#-*-coding:utf-8-*-


import tensorflow as tf
import tensorflow.contrib.slim as slim
import multiprocessing
import time
import numpy as np
import cv2
import math
from functools import partial
from tensorpack.dataflow import BatchData, MultiThreadMapData, PrefetchDataZMQ,PrefetchOnGPUs


from train_config import config as cfg
from data.utils import get_data_set,_map_fn


from net.faster_rcnn import faster_rcnn



from helper.logger import logger



import tensorflow.contrib.eager as tfe
 
tf.enable_eager_execution()
class trainner():
    def __init__(self):
        self.train_data_set=get_data_set(cfg.DATA.root_path,cfg.DATA.train_txt_path)
        self.val_data_set = get_data_set(cfg.DATA.root_path,cfg.DATA.val_txt_path)
        self.inputs=[]
        self.outputs=[]
        self.val_outputs=[]
        self.ite_num=1

        ####train and val aug func
        self.train_map_func=partial(_map_fn,is_training=True)
        self.val_map_func=partial(_map_fn,is_training=False)


    def tower_loss(self,scope, images, labels,boxes, total_anchor,L2_reg, training):
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


        #net_out = shufflenet_v2(images, L2_reg, training)


        inputs={'images':images,
                'gt_labels':labels,
                'gt_boxes':boxes,
                }

        for placeholders in total_anchor:
            inputs[placeholders.name[:-2]]=placeholders


        loss = faster_rcnn(inputs,L2_reg, training)
        regularization_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_loss')

        return loss,regularization_losses
    def average_gradients(self,tower_grads):
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
    def make_data(self, ds,is_training=True):

        if is_training:
            ds = MultiThreadMapData(ds, 10, self.train_map_func, buffer_size=100, strict=True)
        else:
            ds = MultiThreadMapData(ds, 10, self.val_map_func, buffer_size=100, strict=True)
        #ds = BatchData(ds, cfg.TRAIN.num_gpu * cfg.TRAIN.batch_size, remainder=True,use_list=False)
        ds = PrefetchDataZMQ(ds, 2)
        ds.reset_state()
        ds=ds.get_data()

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
                                            cfg.TRAIN.lr_decay_every_step,
                                            decay_rate=cfg.TRAIN.lr_decay_factor,
                                            staircase=True)

            keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            L2_reg = tf.placeholder(tf.float32, name="L2_reg")
            training = tf.placeholder(tf.bool, name="training_flag")





            total_loss_show=0.
            images_place_holder_list = []
            labels_place_holder_list = []
            boxes_place_holder_list = []
            anchors_place_holder_list = []
            # Create an optimizer that performs gradient descent.
            opt = tf.train.AdamOptimizer(lr)
            #opt = tf.train.MomentumOptimizer(lr,momentum=0.9,use_nesterov=False)
            # Get images and labels

            weights_initializer = slim.xavier_initializer()
            biases_initializer = tf.constant_initializer(0.)
            biases_regularizer = tf.no_regularizer
            weights_regularizer = tf.contrib.layers.l2_regularizer(L2_reg)

            # Calculate the gradients for each model tower.
            tower_grads = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(cfg.TRAIN.num_gpu):
                    with tf.device('/gpu:%d' % 0):
                        with tf.name_scope('lztower_%d' % (i)) as scope:
                            with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):

                                images_ = tf.placeholder(tf.float32, [cfg.DATA.hin,cfg.DATA.win, 3], name="images")
                                boxes_ = tf.placeholder(tf.float32, [None,4],name="input_boxes")
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
                                   loss,l2_loss = self.tower_loss(
                                        scope, images_, labels_,boxes_,total_anchor, L2_reg, training)

                                    ##use muti gpu ,large batch
                                   if i == cfg.TRAIN.num_gpu - 1:
                                       total_loss = tf.add_n([ *loss,l2_loss])
                                   else:
                                       total_loss = tf.add_n([ *loss])
                                total_loss_show+=total_loss
                                # Reuse variables for the next tower.
                                tf.get_variable_scope().reuse_variables()

                                ##when use batchnorm, updates operations only from the
                                ## final tower. Ideally, we should grab the updates from all towers
                                # but these stats accumulate extremely fast so we can ignore the
                                #  other stats from the other towers without significant detriment.
                                bn_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)

                                # Retain the summaries from the final tower.
                                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                                # Calculate the gradients for the batch of data on this CIFAR tower.
                                grads = opt.compute_gradients(total_loss)

                                # Keep track of the gradients across all towers.
                                tower_grads.append(grads)
            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            grads = self.average_gradients(tower_grads)

            # Add a summary to track the learning rate.
            summaries.append(tf.summary.scalar('learning_rate', lr))
            summaries.append(tf.summary.scalar('total_loss', total_loss))
            # summaries.append(tf.summary.scalar('loc_loss', loc_loss))
            # summaries.append(tf.summary.scalar('cla_loss', cla_loss))
            # summaries.append(tf.summary.scalar('l2_loss', l2_loss))

            # Add histograms for gradients.
            for grad, var in grads:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

            # Apply the gradients to adjust the shared variables.
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.op.name, var))

            if False:
                # Track the moving averages of all trainable variables.
                variable_averages = tf.train.ExponentialMovingAverage(
                    0.9, global_step)
                variables_averages_op = variable_averages.apply(tf.trainable_variables())
                # Group all updates to into a single train op.
                train_op = tf.group(apply_gradient_op, variables_averages_op, *bn_update_ops)
            else:
                train_op = tf.group(apply_gradient_op, *bn_update_ops)

            self.inputs=[images_place_holder_list,boxes_place_holder_list,labels_place_holder_list,anchors_place_holder_list,keep_prob,L2_reg,training]
            self.outputs=[train_op,total_loss_show,loss,l2_loss,lr ]
            self.val_outputs=[grads,total_loss_show,loss,l2_loss,lr ]
            # Create a saver.
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

            # Build the summary operation from the last tower summaries.
            self.summary_op = tf.summary.merge(summaries)

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



            if cfg.MODEL.continue_train:
                #########################restore the params
                variables_restore = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
                print(variables_restore)

                saver2 = tf.train.Saver(variables_restore)
                saver2.restore(self.sess, cfg.MODEL.pretrained_model)

            if cfg.MODEL.pretrained_model is not None:
                #########################restore the params
                variables_restore = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES,scope=cfg.MODEL.net_structure)
                print(variables_restore)
            #    print('......................................................')
            #    # saver2 = tf.train.Saver(variables_restore)
                variables_restore_n = [v for v in variables_restore if
                                       'GN' not in v.name]  # Conv2d_1c_1x1 Bottleneck
                # print(variables_restore_n)
                saver2 = tf.train.Saver(variables_restore_n)
                saver2.restore(self.sess, cfg.MODEL.pretrained_model)
            if not cfg.MODEL.mode:
                variables_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                print(variables_restore)
                print('......................................................')
                # saver2 = tf.train.Saver(variables_restore)
                variables_restore_n = [v for v in variables_restore if
                                       'logits' not in v.name]  # Conv2d_1c_1x1 Bottleneck
                # print(variables_restore_n)
                saver2 = tf.train.Saver(variables_restore_n)
                saver2.restore(self.sess, cfg.MODEL.pretrained_model)

                saver2.save(self.sess, save_path='./model/inference.ckpt')
            #self.coord = tf.train.Coordinator()
            # Start the queue runners.
            # self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

            self.summary_writer = tf.summary.FileWriter(cfg.MODEL.model_path, self.sess.graph)

            train_ds=self.make_data(self.train_data_set,is_training=True)
            val_ds=self.make_data(self.val_data_set,is_training=False)

            ######


            min_loss_control=1000.
            for epoch in range(cfg.TRAIN.epoch):
                self._train(train_ds,epoch)
                #val_loss=self._val(val_ds,epoch)
                # logger.info('**************'
                #            'val_loss %f '%(val_loss))

                #tmp_model_name=cfg.MODEL.model_path + \
                #               'epoch_' + str(epoch ) + \
                #               'L2_' + str(cfg.TRAIN.weight_decay_factor) + \
                #               '.ckpt'
                #logger.info('save model as %s \n'%tmp_model_name)
                #self.saver.save(self.sess, save_path=tmp_model_name)
                
                if 1:
                    #min_loss_control=val_loss
                    low_loss_model_name = cfg.MODEL.model_path + \
                                     'epoch_' + str(epoch) + \
                                     'L2_' + str(cfg.TRAIN.weight_decay_factor)  + '.ckpt'
                    logger.info('A new low loss model  saved as %s \n' % low_loss_model_name)
                    self.saver.save(self.sess, save_path=low_loss_model_name)
            # self.coord.request_stop()  # 停止线程
            # self.coord.join(self.threads)

            self.sess.close()

    def _train(self,train_ds,_epoch):
        for step in range(cfg.TRAIN.iter_num_per_epoch):
            self.ite_num += 1



            ########show_flag check the data
            if cfg.TRAIN.vis:
                examples = next(train_ds)

                print(examples)
                example_image=examples['image']
                # for i in range(cfg.TRAIN.batch_size):
                #     example = examples[i]
                #     images=example['image']
                #
                #
                #     print(example)
                #     # print(example_image.shape)
                #     # print(example_label.shape)
                #     # print(example_matche.shape)
                cv2.namedWindow('img', 0)
                cv2.imshow('img', example_image)
                cv2.waitKey(0)
            else:

                start_time = time.time()
                feed_dict = {}
                for n in range(cfg.TRAIN.num_gpu):

                    examples = next(train_ds)

                    feed_dict[self.inputs[0][n]] = examples['image']
                    feed_dict[self.inputs[1][n]] = examples['gt_boxes']
                    feed_dict[self.inputs[2][n]] = examples['gt_labels']


                    for k in range(len(cfg.FPN.ANCHOR_STRIDES)):
                        feed_dict[self.inputs[3][n][2*k]]=examples['anchor_labels_lvl{}'.format(k + 2)]
                        feed_dict[self.inputs[3][n][2*k+1]] = examples['anchor_boxes_lvl{}'.format(k + 2)]
                        #print(examples['anchor_labels_lvl{}'.format(k + 2)].shape)


                feed_dict[self.inputs[4]] = cfg.TRAIN.dropout
                feed_dict[self.inputs[5]] = cfg.TRAIN.weight_decay_factor
                feed_dict[self.inputs[6]] = True

                fetch_duration = time.time() - start_time

                start_time = time.time()
                _, total_loss_value,loss_value,l2_loss_value,lr_value = \
                    self.sess.run([*self.outputs],
                             feed_dict=feed_dict)

                duration = time.time() - start_time
                run_duration = duration - fetch_duration
                if self.ite_num % cfg.TRAIN.log_interval == 0:
                    num_examples_per_step = cfg.TRAIN.batch_size * cfg.TRAIN.num_gpu
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / cfg.TRAIN.num_gpu



                    logger.info(loss_value)
                    format_str = ('epoch %d: iter %d, '
                                  'total_loss=%.6f '
                                  'l2_loss=%.6f '
                                  'learning rate =%e '
                                  '(%.1f examples/sec; %.3f sec/batch) '
                                  'fetch data time = %.6f'
                                  'run time = %.6f')
                    logger.info(format_str % (_epoch,
                                              self.ite_num,
                                              total_loss_value,
                                              l2_loss_value,
                                              lr_value,
                                              examples_per_sec,
                                              sec_per_batch,
                                              fetch_duration,
                                              run_duration))

                #if self.ite_num % 100 == 0:
                #    summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
                #    self.summary_writer.add_summary(summary_str, self.ite_num)
    def _val(self,val_ds,_epoch):

        all_total_loss=0
        for step in range(cfg.TRAIN.val_iter):

            feed_dict = {}
            for n in range(cfg.TRAIN.num_gpu):

                examples = next(val_ds)

                feed_dict[self.inputs[0][n]] = examples['image']
                feed_dict[self.inputs[1][n]] = examples['gt_boxes']
                feed_dict[self.inputs[2][n]] = examples['gt_labels']

                for k in range(len(cfg.FPN.ANCHOR_STRIDES)):
                    feed_dict[self.inputs[3][n][2 * k]] = examples['anchor_labels_lvl{}'.format(k + 2)]
                    feed_dict[self.inputs[3][n][2 * k + 1]] = examples['anchor_boxes_lvl{}'.format(k + 2)]
                    # print(examples['anchor_labels_lvl{}'.format(k + 2)].shape)

            feed_dict[self.inputs[4]] = 1.
            feed_dict[self.inputs[5]] = 0.
            feed_dict[self.inputs[6]] = False

            _, total_loss_value, loss_value, l2_loss_value, lr_value = \
                self.sess.run([*self.val_outputs],
                              feed_dict=feed_dict)

            all_total_loss+=total_loss_value

        return all_total_loss/cfg.TRAIN.val_iter

    def train(self):
        self.train_loop()





