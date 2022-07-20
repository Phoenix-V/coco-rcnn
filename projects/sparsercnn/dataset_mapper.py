#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging

import numpy as np
import torch
import pdb, os
import random
from PIL import Image,ImageFilter

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from torchvision.transforms import transforms

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures.boxes import BoxMode

__all__ = ["SparseRCNNDatasetMapper"]

logger = logging.getLogger(__name__)

def build_transform_gen_sparse(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
        # The following three are newly added
        # tfm_gens.append(T.RandomBrightness(0.85, 1.15))
        # tfm_gens.append(T.RandomContrast(0.85, 1.15))
        # tfm_gens.append(T.RandomSaturation(0.85, 1.15))
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


class SparseRCNNDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by SparseRCNN.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None
        # cfg.INPUT.CROP.TYPE 'absolute_range'
        # cfg.INPUT.CROP.SIZE [384, 600]
        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train
        # self.instance_aug = cfg.INPUT.PATCH_AUG

        self.tfm_gens = build_transform_gen_sparse(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )
        logger.info("Using Augmentation from Original.")

        patch_source = cfg.DATASETS.TRAIN
        patch_source = [item for item in patch_source if 'shot' in item]
        self.do_patch = cfg.MODEL.CONTRAPATCH.ENABLED
        if len(patch_source) == 0:
            assert not self.do_patch
        self.patch_crop = None

        if len(patch_source)>0 and self.do_patch:
            dataset = []
            for patch_src in patch_source:
                dataset.extend(DatasetCatalog.get(patch_src))
            patches = []
            for dict_item in dataset:
                patches.extend(get_annotated_patch_from_img(dict_item,self.img_format))
            self.patch_dict = {}
            for patch,cls in patches:
                if cls not in self.patch_dict:
                    self.patch_dict[cls] = []
                self.patch_dict[cls].append(patch)
            self.query_transform = get_query_transforms(is_train)
            if not is_train:
                patch_label = np.array([item[1] for item in patches])
                patch_image = torch.stack([self.query_transform(item[0]) for item in patches])
                self.patch_crop = (patch_image,patch_label)
        # test = {'file_name': 'datasets/coco/trainval2014/COCO_train2014_000000078468.jpg', 'height': 428, 'width': 640, 'image_id': 78468, 'annotations': [{'iscrowd': 0, 'bbox': [52.78, 84.45, 587.22, 311.88], 'category_id': 7, 'bbox_mode':BoxMode.XYWH_ABS}]}
        # self.test_call(test)
    
    def call(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
            
        if self.crop_gen is None:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                image, transforms = T.apply_transform_gens(
                    self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                )

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        
        patches = [[],[]]
        if not self.do_patch:
            pass
        elif not self.is_train:
            assert not (self.patch_crop is None)
            patches = self.patch_crop
        else:
            for i in range(len(self.patch_dict)):
                patches[0].extend([self.patch_dict[i][torch.randperm(len(self.patch_dict[i]))[0]]])
                patches[1].extend([i])
            patches = [torch.stack([self.query_transform(item) for item in patches[0]]),np.array(patches[1])]
        dataset_dict["patches"] = patches
        return dataset_dict

    def __call__(self, dataset_dict):
        return self.call(dataset_dict)
        # """
        # Args:
        #     dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        # Returns:
        #     dict: a format that builtin models in detectron2 accept
        # """
        # dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        # utils.check_image_size(dataset_dict, image)

        # if self.crop_gen is None:
        #     image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # else:
        #     if np.random.rand() > 0.5:
        #         image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        #     else:
        #         image, transforms = T.apply_transform_gens(
        #             self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
        #         )

        # image_shape = image.shape[:2]  # h, w

        # # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # # Therefore it's important to use torch.Tensor.
        # dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # # print('Keep Annotation for Eval Temporarliy')
        # # if not self.is_train:
        # #     # USER: Modify this if you want to keep them for some reason.
        # #     dataset_dict.pop("annotations", None)
        # #     return dataset_dict

        # if "annotations" in dataset_dict:
        #     # USER: Modify this if you want to keep them for some reason.
        #     for anno in dataset_dict["annotations"]:
        #         anno.pop("segmentation", None)
        #         anno.pop("keypoints", None)

        #     # USER: Implement additional transformations if you have other types of data
        #     annos = [
        #         utils.transform_instance_annotations(obj, transforms, image_shape)
        #         for obj in dataset_dict.pop("annotations")
        #         if obj.get("iscrowd", 0) == 0
        #     ]
        #     instances = utils.annotations_to_instances(annos, image_shape)
        #     dataset_dict["instances"] = utils.filter_empty_instances(instances)
        # return dataset_dict

def get_annotated_patch_from_img(dataset_dict,img_format):
    img = utils.read_image(dataset_dict["file_name"], format=img_format)
    utils.check_image_size(dataset_dict, img)
    img = Image.fromarray(img)
    annotations = dataset_dict['annotations']
    patches = []
    for anno in annotations:
        bbox = anno['bbox']
        patch = img.crop((bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]))
        patches.append((patch,anno['category_id']))
    return patches

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_query_transforms(is_train):
    if is_train:
        # SimCLR style augmentation
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),  HorizontalFlip may cause the pretext too difficult, so we remove it
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

# def make_self_det_transforms(image_set):
#     normalize = T.Compose([
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     # The image of ImageNet is relatively small.
#     scales = [320, 336, 352, 368, 400, 416, 432, 448, 464, 480]

#     if image_set == 'train':
#         return T.Compose([
#             # T.RandomHorizontalFlip(), HorizontalFlip may cause the pretext too difficult, so we remove it
#             T.RandomResize(scales, max_size=600),
#             normalize,
#         ])

#     if image_set == 'val':
#         return T.Compose([
#             T.RandomResize([480], max_size=600),
#             normalize,
#         ])

#     raise ValueError(f'unknown {image_set}')


# def get_random_patch_from_img(img, min_pixel=8):
#     """
#     :param img: original image
#     :param min_pixel: min pixels of the query patch
#     :return: query_patch,x,y,w,h
#     """
#     w, h = img.size
#     min_w, max_w = min_pixel, w - min_pixel
#     min_h, max_h = min_pixel, h - min_pixel
#     sw, sh = np.random.randint(min_w, max_w + 1), np.random.randint(min_h, max_h + 1)
#     x, y = np.random.randint(w - sw) if sw != w else 0, np.random.randint(h - sh) if sh != h else 0
#     patch = img.crop((x, y, x + sw, y + sh))
#     return patch, x, y, sw, sh


# class SelfDet(Dataset):
#     """
#     SelfDet is a dataset class which implements random query patch detection.
#     It randomly crops patches as queries from the given image with the corresponding bounding box.
#     The format of the bounding box is same to COCO.
#     """
#     def __init__(self, root, detection_transform, query_transform, num_patches=10):
#         super(SelfDet, self).__init__()
#         self.root = root
#         self.detection_transform = detection_transform
#         self.query_transform = query_transform
#         self.files = []
#         self.num_patches = num_patches
#         for (troot, _, files) in os.walk(root, followlinks=True):
#             for f in files:
#                 path = os.path.join(troot, f)
#                 self.files.append(path)
#             if len(self.files) > 200:
#                 break
#         print(f'num of files:{len(self.files)}')

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, item):
#         img_path = self.files[item]
#         img = Image.open(img_path).convert("RGB")
#         w, h = img.size
#         if w<=16 or h<=16:
#             return self[(item+1)%len(self)]
#         # the format of the dataset is same with COCO.
#         target = {'orig_size': torch.as_tensor([int(h), int(w)]), 'size': torch.as_tensor([int(h), int(w)])}
#         iscrowd = []
#         labels = []
#         boxes = []
#         area = []
#         patches = []
#         while len(area) < self.num_patches:
#             patch, x, y, sw, sh = get_random_patch_from_img(img)
#             boxes.append([x, y, x + sw, y + sh])
#             area.append(sw * sh)
#             iscrowd.append(0)
#             labels.append(1)
#             patches.append(self.query_transform(patch))
#         target['iscrowd'] = torch.tensor(iscrowd)
#         target['labels'] = torch.tensor(labels)
#         target['boxes'] = torch.tensor(boxes)
#         target['area'] = torch.tensor(area)
#         img, target = self.detection_transform(img, target)
#         return img, torch.stack(patches, dim=0), target