#-*-coding:utf-8-*-

import os
import numpy as np
from easydict import EasyDict as edict

config = edict()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

config.TRAIN = edict()
config.TRAIN.num_gpu = 4
config.TRAIN.batch_size = 1
config.TRAIN.log_interval = 10
config.TRAIN.epoch = 2000
config.TRAIN.train_set_size=10000  ###########u need be sure
config.TRAIN.val_set_size=1000
config.TRAIN.iter_num_per_epoch = config.TRAIN.train_set_size // config.TRAIN.num_gpu // config.TRAIN.batch_size

config.TRAIN.val_iter=config.TRAIN.val_set_size// config.TRAIN.num_gpu // config.TRAIN.batch_size

config.TRAIN.lr_init = 0.001  # initial learning rate
config.TRAIN.lr_decay_every_step = 10*config.TRAIN.iter_num_per_epoch  # evey number of step to decay lr
config.TRAIN.lr_decay_factor = 0.7  # decay lr factor
config.TRAIN.weight_decay_factor = 1.e-5

config.TRAIN.dropout=0.5  ##no use
config.TRAIN.vis=False


config.TEST = edict()
config.TEST.FRCNN_NMS_THRESH = 0.4

# Smaller threshold value gives significantly better mAP. But we use 0.05 for consistency with Detectron.
# mAP with 1e-4 threshold can be found at https://github.com/tensorpack/tensorpack/commit/26321ae58120af2568bdbf2269f32aa708d425a8#diff-61085c48abee915b584027e1085e1043  # noqa
config.TEST.RESULT_SCORE_THRESH = 0.05
config.TEST.RESULT_SCORE_THRESH_VIS = 0.3   # only visualize confident results
config.TEST.RESULTS_PER_IM = 100

##RPN
# anchors -------------------------
config.RPN = edict()
config.RPN.ANCHOR_STRIDE = 16
#config.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)   # sqrtarea of the anchor box
config.RPN.ANCHOR_SIZES = (16, 32, 64, 128, 256)   # sqrtarea of the anchor box
config.RPN.ANCHOR_RATIOS = (0.75, 1.,1.25)
config.RPN.POSITIVE_ANCHOR_THRESH = 0.7
config.RPN.NEGATIVE_ANCHOR_THRESH = 0.3

# rpn training -------------------------
config.RPN.FG_RATIO = 0.5  # fg ratio among selected RPN anchors
config.RPN.BATCH_PER_IM = 256  # total (across FPN levels) number of anchors that are marked valid
config.RPN.MIN_SIZE = 0
config.RPN.PROPOSAL_NMS_THRESH = 0.7
# Anchors which overlap with a crowd box (IOA larger than threshold) will be ignored.
# Setting this to a value larger than 1.0 will disable the feature.
# It is disabled by default because Detectron does not do this.
config.RPN.CROWD_OVERLAP_THRESH = 9.99
config.RPN.HEAD_DIM = 1024      # used in C4 only


config.RPN.TRAIN_PER_LEVEL_NMS_TOPK = 2000
config.RPN.TEST_PER_LEVEL_NMS_TOPK = 50



# FPN -------------------------
config.FPN = edict()
config.FPN.ANCHOR_STRIDES = (4, 8, 16, 32, 64)  # strides for each FPN level. Must be the same length as ANCHOR_SIZES
config.FPN.PROPOSAL_MODE = 'Level'  # 'Level', 'Joint'
config.FPN.NUM_CHANNEL = 256//8
config.FPN.NORM = 'None'  # 'None', 'GN'

config.FPN.FRCNN_HEAD_FUNC = 'fastrcnn_2fc_head'
# choices: fastrcnn_2fc_head, fastrcnn_4conv1fc_{,gn_}head
config.FPN.FRCNN_CONV_HEAD_DIM = 256
config.FPN.FRCNN_FC_HEAD_DIM = 1024

# Cascade-RCNN, only available in FPN mode
config.FPN.CASCADE = False

# config.FPN.CASCADE.IOUS = [0.5, 0.6, 0.7]
# config.FPN.CASCADE.BBOX_REG_WEIGHTS = [[10., 10., 5., 5.], [20., 20., 10., 10.], [30., 30., 15., 15.]]


config.FRCNN = edict()
config.FRCNN.BATCH_PER_IM = 512//4
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

config.DATA.PIXEL_MEAN = [123.675, 116.28, 103.53]   ###rgb
config.DATA.PIXEL_STD = [58.395, 57.12, 57.375]

config.DATA.hin = 512  # input size during training , 240
config.DATA.win= 512

config.DATA.MAX_SIZE=640


config.BACKBONE = edict()
# basemodel ----------------------

config.MODEL = edict()

config.MODEL.mode=False ###True for train False for eval

config.MODEL.MODE_MASK = False        # FasterRCNN or MaskRCNN
config.MODEL.MODE_FPN = True

config.MODEL.model_path = './model/'  # save directory

config.MODEL.net_structure='ShuffleNetV2' ######'InceptionResnetV2,resnet_v2_50
# config.MODEL.pretrained_model=None
config.MODEL.pretrained_model='./model/epoch_1L2_1e-05.ckpt'
