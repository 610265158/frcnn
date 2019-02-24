#-*-coding:utf-8-*-

import os
import numpy as np
from easydict import EasyDict as edict

config = edict()
os.environ["CUDA_VISIBLE_DEVICES"] = "2,4"

config.TRAIN = edict()
config.TRAIN.num_gpu = 2
config.TRAIN.batch_size = 1
config.TRAIN.log_interval = 10
config.TRAIN.epoch = 2000
config.TRAIN.train_set_size=17000  ###########u need be sure
config.TRAIN.val_set_size=2800
config.TRAIN.iter_num_per_epoch = config.TRAIN.train_set_size // config.TRAIN.num_gpu // config.TRAIN.batch_size

config.TRAIN.val_iter=config.TRAIN.val_set_size// config.TRAIN.num_gpu // config.TRAIN.batch_size

config.TRAIN.lr_init = 0.001  # initial learning rate
config.TRAIN.lr_decay_every_step = 10*config.TRAIN.iter_num_per_epoch  # evey number of step to decay lr
config.TRAIN.lr_decay_factor = 0.7  # decay lr factor
config.TRAIN.weight_decay_factor = 1.e-7

config.TRAIN.dropout=0.5  ##no use
config.TRAIN.vis=False


##RPN
# anchors -------------------------
config.RPN = edict()
config.RPN.ANCHOR_STRIDE = 16
config.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)   # sqrtarea of the anchor box
config.RPN.ANCHOR_RATIOS = (0.5, 1., 2.)
config.RPN.POSITIVE_ANCHOR_THRESH = 0.7
config.RPN.NEGATIVE_ANCHOR_THRESH = 0.3


# FPN -------------------------
config.FPN = edict()
config.FPN.ANCHOR_STRIDES = (4, 8, 16, 32, 64)  # strides for each FPN level. Must be the same length as ANCHOR_SIZES
config.FPN.PROPOSAL_MODE = 'Level'  # 'Level', 'Joint'
config.FPN.NUM_CHANNEL = 256
config.FPN.NORM = 'None'  # 'None', 'GN'

config.FPN.FRCNN_HEAD_FUNC = 'fastrcnn_2fc_head'
# choices: fastrcnn_2fc_head, fastrcnn_4conv1fc_{,gn_}head
config.FPN.FRCNN_CONV_HEAD_DIM = 256
config.FPN.FRCNN_FC_HEAD_DIM = 1024
config.FPN.MRCNN_HEAD_FUNC = 'maskrcnn_up4conv_head'   # choices: maskrcnn_up4conv_{,gn_}head

# Cascade-RCNN, only available in FPN mode
config.FPN.CASCADE = False

# config.FPN.CASCADE.IOUS = [0.5, 0.6, 0.7]
# config.FPN.CASCADE.BBOX_REG_WEIGHTS = [[10., 10., 5., 5.], [20., 20., 10., 10.], [30., 30., 15., 15.]]


config.FRCNN = edict()
config.FRCNN.BATCH_PER_IM = 512
config.FRCNN.BBOX_REG_WEIGHTS = [10., 10., 5., 5.]  # Better but non-standard setting: [20, 20, 10, 10]
config.FRCNN.FG_THRESH = 0.5
config.FRCNN.FG_RATIO = 0.25  # fg ratio in a ROI batch



config.DATA = edict()
config.DATA.root_path=''
config.DATA.train_txt_path='train.txt'
config.DATA.val_txt_path='val.txt'
config.DATA.NUM_CATEGORY=1
config.DATA.NUM_CLASS = config.DATA.NUM_CATEGORY + 1  # +1 background
config.DATA.cover_small_face=400.



config.BACKBONE = edict()
# basemodel ----------------------
config.BACKBONE.WEIGHTS = ''   # /path/to/weights.npz
config.BACKBONE.RESNET_NUM_BLOCKS = [3, 4, 6, 3]     # for resnet50
# RESNET_NUM_BLOCKS = [3, 4, 23, 3]    # for resnet101
config.BACKBONE.FREEZE_AFFINE = False   # do not train affine parameters inside norm layers
config.BACKBONE.NORM = 'FreezeBN'  # options: FreezeBN, SyncBN, GN, None
config.BACKBONE.FREEZE_AT = 2  # options: 0, 1, 2


config.MODEL = edict()

config.MODEL.mode=True   ###True for train False for eval

config.MODEL.MODE_MASK = False        # FasterRCNN or MaskRCNN
config.MODEL.MODE_FPN = True

config.MODEL.model_path = './model/'  # save directory
config.MODEL.hin = 512  # input size during training , 240
config.MODEL.win = 512
config.MODEL.out_channel=132+3
config.MODEL.net_structure='resnet_v2_50' ######'InceptionResnetV2,resnet_v2_50
config.MODEL.pretrained_model=None
