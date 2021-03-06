# -*-coding:utf-8-*-


import tensorflow as tf
import tensorflow.contrib.slim as slim
import multiprocessing
import time
import numpy as np
import cv2
import math
from functools import partial
from tensorpack.dataflow import BatchData, MultiThreadMapData, PrefetchDataZMQ, PrefetchOnGPUs

from train_config import config as cfg
from data.utils import get_data_set, _map_fn

from net.faster_rcnn import faster_rcnn

from helper.logger import logger

import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()


class trainner():
    def __init__(self):
        self.train_data_set = get_data_set(cfg.DATA.root_path, cfg.DATA.train_txt_path)
        self.val_data_set = get_data_set(cfg.DATA.root_path, cfg.DATA.val_txt_path)
        self.inputs = []
        self.outputs = []
        self.val_outputs = []
        self.ite_num = 1

        ####train and val aug func
        self.train_map_func = partial(_map_fn, is_training=True)
        self.val_map_func = partial(_map_fn, is_training=False)

    def tower_loss(self, scope, images, labels, boxes, total_anchor, L2_reg, training):
        """Calculate the total loss on a single tower running the model.

        Args:
          scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
          images: Images. 4D tensor of shape [batch_size, height, width, 3].
          labels: Labels. 1D tensor of shape [batch_size].

        Returns:
           Tensor of shape [] containing the total loss for a batch of data
        """

        # Build the portion of the Graph calculating the losses. Note that we will
        # assemble the total_loss using a custom function below.

        # net_out = shufflenet_v2(images, L2_reg, training)

        inputs = {'images': images,
                  'gt_labels': labels,
                  'gt_boxes': boxes,
                  }

        for placeholders in total_anchor:
            inputs[placeholders.name[:-2]] = placeholders

        loss = faster_rcnn(inputs, L2_reg, training)

        regularization_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_loss')

        return loss, regularization_losses

    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.

        Note that this function provides a synchronization point across all towers.

        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """

        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                try:
                    expanded_g = tf.expand_dims(g, 0)
                except:
                    print(_)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def make_data(self, ds, is_training=True):

        if is_training:
            ds = MultiThreadMapData(ds, 15, self.train_map_func, buffer_size=1000, strict=True)
        else:
            ds = MultiThreadMapData(ds, 15, self.val_map_func, buffer_size=1000, strict=True)
        # ds = BatchData(ds, cfg.TRAIN.num_gpu * cfg.TRAIN.batch_size, remainder=True,use_list=False)
        # ds = PrefetchDataZMQ(ds, 2)
        ds = PrefetchOnGPUs(ds, [0])
        ds.reset_state()
        ds = ds.get_data()

        ###########
        # ds = data_set.shuffle(buffer_size=512)  # shuffle before loading images
        # ds = ds.repeat(cfg.TRAIN.epoch)
        # if is_training:
        #     ds = ds.map(self.train_map_func, num_parallel_calls=multiprocessing.cpu_count())  # decouple the heavy map_fn
        # else:
        #     ds = ds.map(self.val_map_func, num_parallel_calls=multiprocessing.cpu_count())  # decouple the heavy map_fn
        # ds = ds.batch(
        #     cfg.TRAIN.num_gpu * cfg.TRAIN.batch_size)  # TODO: consider using tf.contrib.map_and_batch
        #
        # ds = ds.prefetch(5 * cfg.TRAIN.num_gpu)
        # iterator = ds.make_one_shot_iterator()
        # one_element = iterator.get_next()
        # images, labels = one_element
        return ds

    def train_loop(self):
        """Train faces data for a number of epoch."""
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            # Create a variable to count the number of train() calls. This equals the
            # number of batches processed * FLAGS.num_gpus.
            global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)

            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(cfg.TRAIN.lr_init,
                                            global_step,
                                            cfg.TRAIN.lr_decay_every_step // cfg.TRAIN.num_gpu,
                                            decay_rate=cfg.TRAIN.lr_decay_factor,
                                            staircase=True)

            keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            L2_reg = tf.placeholder(tf.float32, name="L2_reg")
            training = tf.placeholder(tf.bool, name="training_flag")

            images_place_holder_list = []
            labels_place_holder_list = []
            boxes_place_holder_list = []
            anchors_place_holder_list = []
            # Create an optimizer that performs gradient descent.
            opt = tf.train.AdamOptimizer(lr)
            # opt = tf.train.MomentumOptimizer(lr,momentum=0.9,use_nesterov=False)
            # Get images and labels

            weights_initializer = slim.xavier_initializer()
            biases_initializer = tf.constant_initializer(0.)
            biases_regularizer = tf.no_regularizer
            weights_regularizer = tf.contrib.layers.l2_regularizer(L2_reg)

            # Calculate the gradients for each model tower.
            tower_grads = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(cfg.TRAIN.num_gpu):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('lztower_%d' % (i)) as scope:
                            with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):

                                if not cfg.DATA.MUTISCALE:
                                    images_ = tf.placeholder(tf.float32, [cfg.DATA.hin, cfg.DATA.win, 3], name="images")
                                else:
                                    images_ = tf.placeholder(tf.float32, [None, None, 3], name="images")
                                boxes_ = tf.placeholder(tf.float32, [None, 4], name="input_boxes")
                                labels_ = tf.placeholder(tf.int64, [None], name="input_labels")
                                ###total anchor
                                total_anchor = []
                                num_anchors = len(cfg.RPN.ANCHOR_RATIOS)
                                for k in range(len(cfg.FPN.ANCHOR_STRIDES)):
                                    total_anchor.extend([
                                        tf.placeholder(tf.int32, (None, None, num_anchors),
                                                       'anchor_labels_lvl{}'.format(k + 2)),
                                        tf.placeholder(tf.float32, (None, None, num_anchors, 4),
                                                       'anchor_boxes_lvl{}'.format(k + 2))])

                                images_place_holder_list.append(images_)
                                labels_place_holder_list.append(labels_)
                                boxes_place_holder_list.append(boxes_)
                                anchors_place_holder_list.append(total_anchor)
                                with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                                                     slim.conv2d_transpose, slim.separable_conv2d,
                                                     slim.fully_connected],
                                                    weights_regularizer=weights_regularizer,
                                                    biases_regularizer=biases_regularizer,
                                                    weights_initializer=weights_initializer,
                                                    biases_initializer=biases_initializer):
                                    loss, l2_loss = self.tower_loss(
                                        scope, images_, labels_, boxes_, total_anchor, L2_reg, training)

                                    # Reuse variables for the next tower.
                                tf.get_variable_scope().reuse_variables()


            # Create a saver.
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)



            # Build an initialization operation to run below.
            init = tf.global_variables_initializer()

            # Start running operations on the Graph. allow_soft_placement must be set to
            # True to build towers on GPU, as some of the ops do not have GPU
            # implementations.

            tf_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            tf_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=tf_config)
            self.sess.run(init)

            if not cfg.MODEL.mode:
                #variables_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                variables_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                print(variables_restore)
                print('......................................................')
                # saver2 = tf.train.Saver(variables_restore)
                variables_restore_n = [v for v in variables_restore if
                                       'logits' not in v.name]  # Conv2d_1c_1x1 Bottleneck
                # print(variables_restore_n)
                saver2 = tf.train.Saver(variables_restore_n)
                saver2.restore(self.sess, cfg.MODEL.pretrained_model)

                saver2.save(self.sess, save_path='./model/inference.ckpt')




    def train(self):
        self.train_loop()




import setproctitle
setproctitle.setproctitle("releasemodel")


trainner=trainner()

trainner.train()

