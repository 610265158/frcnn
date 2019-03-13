#-*-coding:utf-8-*-

import pickle
import tensorflow as tf
import numpy as np
import cv2
import random
from functools import partial
import copy


from net.data import *
from helper.logger import logger
from data.datainfo import data_info
from data.augmentor.augmentation import Pixel_jitter,Fill_img,Random_contrast,\
    Random_brightness,Random_scale_withbbox,Random_flip,Blur_aug,Rotate_with_box

from train_config import config as cfg

from tensorpack.dataflow import BatchData, MultiThreadMapData, PrefetchDataZMQ,DataFromList
def balance(anns):
    res_anns=copy.deepcopy(anns)


    for ann in anns:
        label=ann[-1]
        label = np.array([label.split(' ')], dtype=np.float).reshape((-1, 2))
        bbox = np.array([np.min(label[:, 0]), np.min(label[:, 1]), np.max(label[:, 0]), np.max(label[:, 1])])
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]

        if bbox_width<40 or bbox_height<40:
            res_anns.remove(ann)

        if np.sqrt(np.square(label[37,0]-label[41,0])+np.square(label[37,1]-label[41,1]))/bbox_height<0.02 \
            or np.sqrt(np.square(label[38, 0] - label[40, 0]) + np.square(label[38, 1] - label[40, 1])) / bbox_height < 0.02 \
            or np.sqrt(np.square(label[43,0]-label[47,0])+np.square(label[43,1]-label[47,1]))/bbox_height<0.02 \
            or np.sqrt(np.square(label[44, 0] - label[46, 0]) + np.square(label[44, 1] - label[46, 1])) / bbox_height < 0.02 :
            for i in range(10):
                res_anns.append(ann)
    random.shuffle(res_anns)
    logger.info('befor balance the dataset contains %d images' % (len(anns)))
    logger.info('after balanced the datasets contains %d samples' % (len(res_anns)))
    return res_anns
def get_train_data_list(im_root_path, ann_txt):
    """
    train_im_path : image folder name
    train_ann_path : coco json file name
    """
    logger.info("[x] Get data from {}".format(im_root_path))
    # data = PoseInfo(im_path, ann_path, False)
    data = data_info(im_root_path, ann_txt)
    all_samples=data.get_all_sample()

    return all_samples
def get_data_set(root_path,ana_path):
    data_list=get_train_data_list(root_path,ana_path)
    dataset= DataFromList(data_list, shuffle=True)
    return dataset



def _data_aug_fn(fname, ground_truth,is_training=True):
    """Data augmentation function."""
    ####customed here
    try:

        image = cv2.imread(fname, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = ground_truth.split(' ')
        boxes = []
        for label in labels:
            bbox = np.array(label.split(','), dtype=np.float)
            ##the anchor need ymin,xmin,ymax,xmax
            boxes.append([bbox[1], bbox[0], bbox[3], bbox[2]])

        boxes = np.array(boxes, dtype=np.float)

        ###clip the bbox for the reason that some bboxs are beyond the image
        h_raw_limit, w_raw_limit, _ = image.shape
        boxes[:, 3] = np.clip(boxes[:, 3], 0, w_raw_limit)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, h_raw_limit)
        boxes[boxes < 0] = 0
        #########random scale
        ############## becareful with this func because there is a Infinite loop in its body
        image, boxes=Random_scale_withbbox(image,boxes,target_shape=[cfg.DATA.hin,cfg.DATA.win],jitter=0.3)

        if is_training:
            if random.uniform(0, 1) > 0.5:
                image, boxes =Random_flip(image, boxes)
            image=Pixel_jitter(image,max_=15)
            if random.uniform(0,1)>0.5:
                image=Random_contrast(image)
            if random.uniform(0,1)>0.5:
                image=Random_brightness(image)
            if random.uniform(0,1)>0.5:
                a=[3,5,7]
                k=random.sample(a, 1)[0]
                image=Blur_aug(image,ksize=(k,k))
            if random.uniform(0, 1) > 0.5:
                k = random.uniform(-90, 90)
                image, boxes = Rotate_with_box(image, k, boxes)


        boxes=np.clip(boxes,0,cfg.DATA.hin)
        # ###cove the small faces
        # boxes_clean=[]
        # for i in range(boxes.shape[0]):
        #     box = boxes[i]
        #
        #     if (box[3]-box[1])*(box[2]-box[0])<cfg.DATA.cover_small_face:
        #         image[int(box[0]):int(box[2]),int(box[1]):int(box[3]),:]=0
        #     else:
        #         boxes_clean.append(box)
        boxes=np.array(boxes,dtype=np.float32)
        boxes_refine=np.zeros_like(boxes)
        boxes_refine[:, 0] = boxes[:,1]
        boxes_refine[:, 1] = boxes[:, 0]
        boxes_refine[:, 2] = boxes[:, 3]
        boxes_refine[:, 3] = boxes[:, 2]


        crowd=np.zeros(shape=[boxes_refine.shape[0]])
        klass=np.ones(shape=[boxes_refine.shape[0]])

    except:
        logger.warn('there is an err with %s' % fname)
        image = np.zeros(shape=(cfg.DATA.hin, cfg.DATA.win, 3), dtype=np.float32)
        boxes_refine = np.array([[0, 0, 100, 100]])
        klass = np.array([1])
        crowd = np.array([0])
    ret =prepare_data(image,boxes_refine,klass,crowd)
    return ret


def prepare_data(image,boxes,klass,is_crowd=0):

    boxes = np.copy(boxes)
    im = image
    assert im is not None
    im = im.astype('float32')
    # assume floatbox as input
    assert boxes.dtype == np.float32, "Loader has to return floating point boxes!"

    ret = {'image': im}
    # rpn anchor:
    try:
        if cfg.MODEL.MODE_FPN:
            multilevel_anchor_inputs = get_multilevel_rpn_anchor_input(im, boxes, is_crowd)
            for i, (anchor_labels, anchor_boxes) in enumerate(multilevel_anchor_inputs):

                ret['anchor_labels_lvl{}'.format(i + 2)] = anchor_labels
                ret['anchor_boxes_lvl{}'.format(i + 2)] = anchor_boxes
        else:
            # anchor_labels, anchor_boxes
            ret['anchor_labels'], ret['anchor_boxes'] = get_rpn_anchor_input(im, boxes, is_crowd)

        boxes = boxes[is_crowd == 0]  # skip crowd boxes in training target
        klass = klass[is_crowd == 0]
        ret['gt_boxes'] = boxes
        ret['gt_labels'] = klass
        if not len(boxes):
            raise MalformedData("No valid gt_boxes!")
    except MalformedData as e:
        logger.info("Input {} is filtered for training: {}".format('err', str(e)), 'warn')
        return None


    return ret
def _map_fn(dp,is_training=True):
    fname, annos = dp
    ret=_data_aug_fn(fname,annos,is_training)
    return ret


if __name__=='__main__':
    image = np.zeros(shape=(cfg.DATA.hin, cfg.DATA.win, 3), dtype=np.float32)
    boxes_refine = np.array([[0,0,100,100]])
    klass = np.array([1])
    crowd = np.array([0])
    multilevel_anchor_inputs=get_multilevel_rpn_anchor_input(image, boxes_refine, crowd)
    print()


