#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math,pdb
from typing import List
from tqdm import tqdm

import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling import build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances

from detectron2.modeling.poolers import ROIPooler, cat
# from detectron2.modeling.roi_heads import build_roi_heads
# from detectron2.utils.logger import log_first_n
# from fvcore.nn import giou_loss, smooth_l1_loss

from .loss import SetCriterion, HungarianMatcher
from .head import DynamicHead
from .build import META_ARCH_REGISTRY
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .util.proposal_filter import pairwise_iou,consequent_iou
from .detector import SparseRCNN
from .supcontrast import RoIPoolContrast, AnchorPoolContrast, SupConLoss, MixConLoss

_base_classes = [
    8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 84, 85, 86, 87, 88, 89, 90,
]
_novel_classes = [
    1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72
]
_all_classes = sorted(_base_classes+_novel_classes)
cls_map = {v:i for i,v in enumerate(_all_classes)}
mapped_novel = [cls_map[i] for i in _novel_classes]
mapped_base = [cls_map[i] for i in _base_classes]

__all__ = ["SparseRCNNCoCo"]

logger = logging.getLogger(__name__)

@META_ARCH_REGISTRY.register()
class SparseRCNNCoCo(SparseRCNN):
    """
    Implement SparseRCNNCoCo
    """

    def __init__(self, cfg):
        super(SparseRCNNCoCo,self).__init__(cfg)
        print('This is SparseRCNNCoCo')

        # Contrastive Related Param
        self.proj_dim = cfg.MODEL.PROJ_HEADS.PROJ_DIM
        self.thre_iou = cfg.MODEL.PROJ_HEADS.THRE_IOU
        self.pos_topk = cfg.MODEL.PROJ_HEADS.POS_TOPK
        self.neg_rcnt = cfg.MODEL.PROJ_HEADS.NEG_RCNT
        self.ct_temp = cfg.MODEL.CONTRASTIVE.TEMP_CT
        self.ct_iou = cfg.MODEL.CONTRASTIVE.THRE_IOU
        self.ncon_iou = cfg.MODEL.CONTRASTIVE.NCON_IOU
        self.ct_momentum = cfg.MODEL.CONTRASTIVE.MOMENTUM
        self.reweight_func = cfg.MODEL.CONTRASTIVE.REWEIGHT_FUNC

        self.do_ct = cfg.MODEL.CONTRASTIVE.ENABLED
        self.do_patch = cfg.MODEL.CONTRAPATCH.ENABLED
        self.patch_independ = cfg.MODEL.CONTRAPATCH.INDEPEND
        self.patch_shuffle = cfg.MODEL.CONTRAPATCH.SHUFFLE and self.do_patch

        assert self.do_ct or self.do_patch

        self.roi_gt_proj = RoIPoolContrast(cfg)
        self.patch2query = nn.Linear(self.backbone.bottom_up._out_feature_channels['res5'], self.hidden_dim)
        self.attention_mask = torch.ones(self.num_classes, self.num_classes) * float('-inf')
        self.attention_mask[torch.eye(self.num_classes)==1] = 0

        self.ct_criterion = SupConLoss(self.ct_temp, self.ct_iou, self.reweight_func)
        # self.ct_criterion = MixConLoss(self.ct_temp, self.ct_iou, self.reweight_func)
        self.to(self.device)

    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        src = self.backbone(images.tensor)
        features = list()
        for idx,f in enumerate(self.in_features):
            feature = src[f]
            if self.mod_reweight:
                # feature = decouple_layer(feature,self.mod_decouple) if self.mod_decouple > 0 else feature
                feature = self.affine_layers(feature) if self.mod_share else self.affine_layers[idx](feature)
            features.append(feature)
        
        if self.do_patch:
            patch_features,idx,cls_attn_mask = self.proposal_modulate(batched_inputs)
            cls_attn_mask = (patch_features,idx,cls_attn_mask)
            proposal_boxes = self.init_proposal_boxes.weight[idx].clone()
        else:
            proposal_boxes = self.init_proposal_boxes.weight.clone()
            cls_attn_mask = None

        # Prepare Proposals.
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]
        
        # Prediction.
        outputs_class, outputs_coord, _, outputs_proj = self.head(features, proposal_boxes, self.init_proposal_features.weight,cls_attn_mask)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        outputs_proj = outputs_proj.permute(1,0,2,3).contiguous().view(-1,self.num_proposals*self.num_heads, self.proj_dim)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)

            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                         for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
            loss_dict = self.criterion(output, targets)

            pos_unit,pos_cross,neg_cross,_ = self.gt_pred_iou(outputs_coord,targets)
            pos_index,neg_index = self.get_index(pos_cross, neg_cross)

            pos_feat = outputs_proj[pos_index['id']]
            neg_feat = outputs_proj[neg_index]

            gt_boxes,gt_label = list(),list()
            for x in gt_instances:
                gt_boxes.append(x.gt_boxes)
                gt_label.append(x.gt_classes)
            gt_roi_fmap = self.head.box_pooler(features, gt_boxes)
            gt_roi_feat = self.roi_gt_proj(gt_roi_fmap)
            gt_label = torch.cat(gt_label,dim=0)

            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                         for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
            loss_dict = self.criterion(output, targets)
            all_feat = torch.cat([gt_roi_feat,pos_feat,neg_feat],dim=0)
            all_label = torch.cat([gt_label,torch.Tensor(pos_index['class']).long().to(self.device),-1*torch.ones(neg_feat.size(0)).long().to(self.device)])
            all_iou = torch.cat([torch.ones_like(gt_label).float(),pos_index['iou'],torch.zeros(neg_feat.size(0)).float().to(self.device)])
            loss_dict['loss_ct'] = self.ct_criterion(all_feat,all_label,all_iou)

            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)

            # analysis only
            gt_instances = [x["instances"].to(torch.device('cuda')) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            iou_matrix = pairwise_iou(targets[0], box_pred.view(-1,4)).view(-1,self.num_proposals)
            labels = targets[0]['labels']
            torch.save({'iou':iou_matrix,'labels':labels},'/home/jiawei/MODEL/DetectionAlign/SparseAlign/checkpoints/voc/sparse_rcnn_r101p300/debug/train/{}'.format(batched_inputs[0]['file_name'].split('/')[-1]))
            # pdb.set_trace()

            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results
            else:
                return results
        
    def proposal_modulate(self,batched_inputs):
        idx = torch.randperm(self.num_proposals) if self.patch_shuffle and self.training else torch.arange(self.num_proposals)
        patch_feature = torch.cat([item['patches'][0] for item in batched_inputs])
        patch_label = torch.tensor(np.concatenate([item['patches'][1] for item in batched_inputs]))
        if self.training:
            select = torch.randint(0,len(patch_label),(self.num_proposals,))
            mask_query_patch = (torch.rand(self.num_proposals, 1, device=self.device) > 0.1).float()
        else:
            gt_classes = batched_inputs[0]['instances'].gt_classes
            index = torch.eq(patch_label[:,None],gt_classes[None]).float().sum(dim=-1)
            must_have = (index>0).nonzero(as_tuple=False).squeeze(-1)
            new_add = torch.randint(0,len(patch_label),(self.num_proposals-must_have.size(0),))
            select = torch.cat([must_have,new_add],dim=-1)
            select = torch.randint(0,len(patch_label),(self.num_proposals,))
            mask_query_patch = torch.ones((self.num_proposals,1)).to(self.device)
        patch_label = patch_label[select].to(self.device)
        patch_feature = self.backbone.bottom_up(patch_feature[select].to(self.device))['res5'].mean((-1,-2))
        patch_feature = self.patch2query(patch_feature)
        patch_feature = patch_feature * mask_query_patch
        if self.patch_independ:
            cls_attn_float = torch.eq(patch_label[:,None],patch_label[None,:]).float()
            cls_attn_mask = (cls_attn_float==0)
            # cls_attn_mask = torch.where(cls_attn_float==1,torch.zeros_like(cls_attn_float),torch.ones_like(cls_attn_float)*float('-inf'))
        else:
            cls_attn_mask = None
        return patch_feature,idx.cuda(),cls_attn_mask

    def gt_pred_iou(self, proposals, targets):
        pos_unit,pos_cross = [],[]
        neg_cross,neg_consist = [],[]
        for idx, targets_per_image in enumerate(targets):

            proposals_per_image = proposals[:,idx].contiguous()
            iou_matrix = pairwise_iou(targets_per_image, proposals_per_image.view(-1,4)).view(-1,self.num_heads,self.num_proposals)
            iou_conseq = consequent_iou(proposals_per_image)
            if iou_matrix.size(0)==0:
                pos_unit.append(dict())
                pos_cross.append(dict())
                neg_cross.append(torch.Tensor(np.linspace(0,self.num_proposals-1,self.num_proposals)).long().to(self.device)+self.num_proposals*(self.num_heads-1))
                neg_consist.append((iou_conseq.view(-1)>self.ncon_iou).nonzero(as_tuple=True)[0])
                continue

            iou_maxmat,multi_idx = iou_matrix.max(dim=0)
            multi_idx = targets_per_image['labels'][multi_idx]

            unit_assign = -1.0*torch.ones_like(iou_maxmat)
            unit_assign[iou_maxmat<self.thre_iou[0]] = 0
            unit_assign[iou_maxmat>=self.thre_iou[1]] = 1

            cross_assign = -1.0*torch.ones_like(iou_maxmat)
            cross_assign[1:] = torch.where(unit_assign[1:]==unit_assign[:-1], unit_assign[1:].float(), -1.0*torch.ones_like(cross_assign[1:]))
            cross_assign[1:] = torch.where(torch.logical_and(multi_idx[1:]!=multi_idx[:5], cross_assign[1:]==1), -1.0*torch.ones_like(cross_assign[1:]), cross_assign[1:])

            result = self.class_sort(iou_maxmat,unit_assign,cross_assign,multi_idx)
            pos_unit.append(result[0])
            pos_cross.append(result[1])
            neg_cross.append((cross_assign[-1]==0).nonzero(as_tuple=True)[0]+self.num_proposals*(self.num_heads-1))
            neg_index = torch.logical_and(cross_assign==0,iou_conseq>self.ncon_iou)
            neg_consist.append((neg_index.view(-1)).nonzero(as_tuple=True)[0])

        return pos_unit,pos_cross,neg_cross,neg_consist
    
    def class_sort(self, iou_max, unit_assign, cross_assign, max_index):
        iou_max = iou_max.view(-1)
        unit_assign = unit_assign.view(-1)
        cross_assign = cross_assign.view(-1)
        max_index = max_index.view(-1)
        classes = max_index.unique().cpu().numpy().tolist()

        unit_pos_idx = (unit_assign==1).nonzero(as_tuple=True)[0]
        unit_pos_iou = iou_max[unit_pos_idx].sort(descending=True)
        unit_pos_cls = max_index[unit_pos_idx][unit_pos_iou[1]]
        unit_pos_idx = unit_pos_idx[unit_pos_iou[1]]
        unit_pos = {the_cls:(unit_pos_idx[unit_pos_cls==the_cls],unit_pos_iou[0][unit_pos_cls==the_cls]) for the_cls in classes}

        cross_pos_idx = (cross_assign==1).nonzero(as_tuple=True)[0]
        cross_pos_iou = iou_max[cross_pos_idx].sort(descending=True)
        cross_pos_cls = max_index[cross_pos_idx][cross_pos_iou[1]]
        cross_pos_idx = cross_pos_idx[cross_pos_iou[1]]
        cross_pos = {the_cls:(cross_pos_idx[cross_pos_cls==the_cls],cross_pos_iou[0][cross_pos_cls==the_cls]) for the_cls in classes}

        return unit_pos,cross_pos
    

    def get_index(self, pos_all, neg_all):
        pos_idx = {'class':[],'id':([],[]),'iou':[]}
        for idx,pos_dict in enumerate(pos_all):
            for the_cls,the_idx in pos_dict.items():
                if len(the_idx[0])==0:
                    continue
                pos_idx['class'].extend([the_cls]*min(len(the_idx[0]),self.pos_topk))
                pos_idx['id'][0].extend([idx]*min(len(the_idx[0]),self.pos_topk))
                pos_idx['id'][1].append(the_idx[0][:self.pos_topk])
                pos_idx['iou'].append(the_idx[1][:self.pos_topk])

        pos_idx['id'] = [pos_idx['id'][0],[] if len(pos_idx['id'][1])==0 else torch.cat(pos_idx['id'][1])]
        pos_idx['iou'] = torch.Tensor([]).to(self.device) if len(pos_idx['iou'])==0 else torch.cat(pos_idx['iou'])

        neg_idx = ([],[])
        for idx,neg_item in enumerate(neg_all):
            if len(neg_item)==0:
                    continue
            neg_idx[0].extend([idx]*min(len(neg_item),self.neg_rcnt))
            neg_idx[1].append(neg_item[:self.neg_rcnt])
        neg_idx = (neg_idx[0], torch.cat(neg_idx[1]))
        return pos_idx,neg_idx
    