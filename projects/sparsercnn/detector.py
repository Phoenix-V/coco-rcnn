#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math,pdb
from typing import List

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

from .loss import SetCriterion, HungarianMatcher
from .head import DynamicHead
from .build import META_ARCH_REGISTRY
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.proposal_filter import pairwise_iou,consequent_iou
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .gdl import AffineLayer,decouple_layer

__all__ = ["SparseRCNN"]

@META_ARCH_REGISTRY.register()
class SparseRCNN(nn.Module):
    """
    Implement SparseRCNN
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        # Sparse-RCNN Related Param
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES
        self.num_proposals = cfg.MODEL.SparseRCNN.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        self.num_heads = cfg.MODEL.SparseRCNN.NUM_HEADS
        self.mod_reweight = cfg.MODEL.MODULATE.REWEIGHT
        self.mod_share = cfg.MODEL.MODULATE.SHAREAFF
        self.mod_decouple = cfg.MODEL.MODULATE.DECOUPLE
        if self.mod_decouple > 0:
            assert self.mod_reweight

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility
        
        # Build Proposals.
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)
        
        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())
        if cfg.MODEL.SparseRCNN.TUNE_PROMPT>0:
            self.prompt_proposal = nn.Embedding(cfg.MODEL.SparseRCNN.TUNE_PROMPT, self.hidden_dim)
            self.prompt_boxes = nn.Embedding(cfg.MODEL.SparseRCNN.TUNE_PROMPT, 4)
            nn.init.constant_(self.prompt_boxes.weight[:, :2], 0.5)
            nn.init.constant_(self.prompt_boxes.weight[:, 2:], 1.0)
        else:
           self.prompt_proposal = None
           self.prompt_boxes = None
        # if self.mod_reweight:
        #     if self.mod_share:
        #         self.affine_layers = AffineLayer(num_channels=self.hidden_dim, bias=True)
        #     else:
        #         self.affine_layers = nn.ModuleList([AffineLayer(num_channels=self.hidden_dim, bias=True) for key in self.in_features])
                
        # Loss parameters:
        class_weight = cfg.MODEL.SparseRCNN.CLASS_WEIGHT
        giou_weight = cfg.MODEL.SparseRCNN.GIOU_WEIGHT
        l1_weight = cfg.MODEL.SparseRCNN.L1_WEIGHT
        no_object_weight = cfg.MODEL.SparseRCNN.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.SparseRCNN.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL
        self.tune_until = cfg.MODEL.ROI_HEADS.FREEZE_UNTIL

        # Build Criterion.
        matcher = HungarianMatcher(cfg=cfg,
                                   cost_class=class_weight, 
                                   cost_bbox=l1_weight, 
                                   cost_giou=giou_weight,
                                   use_focal=self.use_focal)
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight,"loss_ct":cfg.MODEL.CONTRASTIVE.WEIGHT}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                local_weight = {k + f"_{i}": v for k, v in weight_dict.items()}
                aux_weight_dict.update(local_weight)
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]
        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=self.num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=no_object_weight,
                                      losses=losses,
                                      use_focal=self.use_focal)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)


        if cfg.MODEL.BACKBONE.BKN_FREEZE:
            for n,p in self.backbone.named_parameters():
                if 'bottom_up' in n and not ('res5' in n):
                    p.requires_grad = False
            print("froze backbone resnet parameters")

        if cfg.MODEL.BACKBONE.FPN_FREEZE:
            for n,p in self.backbone.named_parameters():
                if not ('bottom_up' in n and not ('res5' in n)):
                    p.requires_grad = False
            print("froze backbone pynamid parameters")
        elif cfg.SOLVER.TUNE_MULTIPLIER==0:
            for n,p in self.backbone.named_parameters():
                if 'bottom_up' in n and 'res5' in n:
                    p.requires_grad = False
            print("the res5 block is fixed")

        # Just update the last layer
        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for n,p in self.head.named_parameters():
                if int(n.split('.')[1]) < self.tune_until:
                    if 'class_logits' in n or 'bboxes_delta' in n or 'projection' in n:
                        pass
                    elif 'norm' in n and (not cfg.MODEL.ROI_HEADS.FREEZE_NORM):
                        pass
                    elif '_module' in n:
                        if (int(n.split('_module.')[-1].split('.')[0])%3==0) and cfg.MODEL.ROI_HEADS.FREEZE_MOD:
                            p.requires_grad = False
                        if (int(n.split('_module.')[-1].split('.')[0])%3==1) and cfg.MODEL.ROI_HEADS.FREEZE_NORM:
                            p.requires_grad = False
                    else:
                        p.requires_grad = False
            print("froze dynamic interaction parameters")
        
        if cfg.MODEL.ROI_HEADS.FREEZE_INIT:
            for p in self.init_proposal_features.parameters():
                p.requires_grad = False
            for p in self.init_proposal_boxes.parameters():
                p.requires_grad = False
            print("froze init features parameters")

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
            # if self.mod_reweight:
            #     feature = decouple_layer(feature,self.mod_decouple) if self.mod_decouple > 0 else feature
            #     feature = self.affine_layers(feature) if self.mod_share else self.affine_layers[idx](feature)
            features.append(feature)

        # Prepare Proposals.
        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]
        if self.prompt_boxes is not None:
            prompt_boxes = self.prompt_boxes.weight.clone()
            prompt_boxes = box_cxcywh_to_xyxy(prompt_boxes)
            prompt_boxes = prompt_boxes[None] * images_whwh[:, None, :]
            prompt = (self.prompt_proposal.weight,prompt_boxes)
        else:
            prompt = None

        # Prediction.
        outputs_class, outputs_coord, _, _ = self.head(features, proposal_boxes, self.init_proposal_features.weight,prompt=prompt)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)

            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                         for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
            loss_dict = self.criterion(output, targets)

            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
            
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

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return new_targets

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device).\
                     unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                scores, labels, box_pred, image_sizes
            )):
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh
    