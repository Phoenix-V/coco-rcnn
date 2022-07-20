# -*- coding: utf-8 -*-
#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_sparsercnn_config(cfg):
    """
    Add config for SparseRCNN.
    """
    cfg.MODEL.SparseRCNN = CN()
    cfg.MODEL.SparseRCNN.NUM_CLASSES = 80
    cfg.MODEL.SparseRCNN.NUM_PROPOSALS = 300
    cfg.MODEL.SparseRCNN.TUNE_PROMPT = 0

    # RCNN Head.
    cfg.MODEL.SparseRCNN.NHEADS = 8
    cfg.MODEL.SparseRCNN.DROPOUT = 0.0
    cfg.MODEL.SparseRCNN.DIM_FEEDFORWARD = 2048
    cfg.MODEL.SparseRCNN.ACTIVATION = 'relu'
    cfg.MODEL.SparseRCNN.HIDDEN_DIM = 256
    cfg.MODEL.SparseRCNN.NUM_CLS = 1
    cfg.MODEL.SparseRCNN.NUM_REG = 3
    cfg.MODEL.SparseRCNN.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.SparseRCNN.NUM_DYNAMIC = 2
    cfg.MODEL.SparseRCNN.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.SparseRCNN.CLASS_WEIGHT = 2.0
    cfg.MODEL.SparseRCNN.GIOU_WEIGHT = 2.0
    cfg.MODEL.SparseRCNN.L1_WEIGHT = 5.0
    cfg.MODEL.SparseRCNN.DEEP_SUPERVISION = True
    cfg.MODEL.SparseRCNN.DEEP_CONSISTENCY = 0
    cfg.MODEL.SparseRCNN.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.SparseRCNN.USE_FOCAL = True
    cfg.MODEL.SparseRCNN.ALPHA = 0.25
    cfg.MODEL.SparseRCNN.GAMMA = 2.0
    cfg.MODEL.SparseRCNN.PRIOR_PROB = 0.01

    # Contrastive Loss.
    cfg.MODEL.PROJ_HEADS = CN()
    cfg.MODEL.PROJ_HEADS.PROJ_DIM = 128
    cfg.MODEL.PROJ_HEADS.THRE_IOU = [0.3, 0.7]
    # cfg.MODEL.PROJ_HEADS.PROJ_COM = False
    cfg.MODEL.PROJ_HEADS.POS_TOPK = 10
    cfg.MODEL.PROJ_HEADS.NEG_RCNT = 10
    cfg.MODEL.CONTRASTIVE = CN()
    cfg.MODEL.CONTRASTIVE.ENABLED = False
    cfg.MODEL.CONTRASTIVE.THRE_IOU = 0.8
    cfg.MODEL.CONTRASTIVE.NCON_IOU = 0.9
    cfg.MODEL.CONTRASTIVE.TEMP_CT = 0.2
    # cfg.MODEL.CONTRASTIVE.VERSION = 'V1'
    cfg.MODEL.CONTRASTIVE.REWEIGHT_FUNC = 'none'
    cfg.MODEL.CONTRASTIVE.WEIGHT = 1.0
    cfg.MODEL.CONTRASTIVE.MOMENTUM = 0.8
    cfg.MODEL.CONTRAPATCH = CN()
    cfg.MODEL.CONTRAPATCH.ATTNACT = False
    cfg.MODEL.CONTRAPATCH.ENABLED = False
    cfg.MODEL.CONTRAPATCH.SHUFFLE = True
    cfg.MODEL.CONTRAPATCH.INDEPEND = False
    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.SOLVER.TUNE_MULTIPLIER = 0.1

    cfg.MODEL.MODULATE = CN()
    cfg.MODEL.MODULATE.REWEIGHT=False
    cfg.MODEL.MODULATE.SHAREAFF=False
    cfg.MODEL.MODULATE.DECOUPLE=0.0
 
    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000], 
                                 [64, 10000], [64, 10000],
                                 [64, 10000], [0, 10000],
                                 [0, 10000], [0, 256],
                                 [0, 256], [0, 192],
                                 [0, 192], [0, 96],
                                 [0, 10000])
    
    # PCB Calibration
    cfg.TEST.PCB = CN()
    cfg.TEST.PCB.ENABLED = False
    cfg.TEST.PCB.ALPHA = 0.50
    cfg.TEST.PCB.SCORE_UPPER = 1.0
    cfg.TEST.PCB.SCORE_LOWER = 0.05
    cfg.TEST.PCB.DEPTH = 101
    cfg.TEST.PCB.RES_WEIGHTS = "none"
