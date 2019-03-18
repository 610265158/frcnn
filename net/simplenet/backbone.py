#-*-coding:utf-8-*-


import tensorflow as tf

import tensorflow.contrib.slim as slim
from functools import partial

from net.resnet_cp.basemodel import resnet_arg_scope

resnet_arg_scope = partial(resnet_arg_scope, bn_trainable=True)


from net.simplenet.simple_nn import simple_nn




def large_conv(feature,scope):
    branch_1 = slim.separable_conv2d(feature, 16, [1, 7],
                          padding='SAME', activation_fn=tf.nn.relu,
                          scope='lateral_1_1/res{}'.format(5 - scope))
    branch_1 = slim.separable_conv2d(branch_1, 16, [7, 1],
                              padding='SAME', activation_fn=None,
                              scope='lateral_1_2/res{}'.format(5 - scope))

    branch_2 = slim.separable_conv2d(feature, 16, [7, 1],
                                     padding='SAME', activation_fn=tf.nn.relu,
                                     scope='lateral_2_1/res{}'.format(5 - scope))
    branch_2 = slim.separable_conv2d(branch_2, 16, [1, 7],
                                     padding='SAME', activation_fn=None,
                                     scope='lateral_2_2/res{}'.format(5 - scope))


    return tf.add(branch_1,branch_2,name='lateral/res{}'.format(5 - scope))

def create_global_net(blocks, L2_reg,is_training, trainable=True,data_format='NHWC'):
    global_fms = []

    last_fm = None
    initializer = tf.contrib.layers.xavier_initializer()
    for i, block in enumerate(reversed(blocks)):
        with slim.arg_scope(resnet_arg_scope(weight_decay=L2_reg,bn_is_training=is_training,data_format=data_format)):
            # lateral = slim.conv2d(block, 32, [1, 1],
            #     trainable=trainable, weights_initializer=initializer,
            #     padding='SAME', activation_fn=None,
            #     scope='lateral/res{}'.format(5-i))
            lateral=large_conv(block,i)
            if last_fm is not None:

                upsample = tf.keras.layers.UpSampling2D(data_format='channels_last' if data_format=='NHWC' else 'channels_first')(last_fm)
                # upsample = slim.conv2d(upsample, 10, [1, 1],
                #     trainable=trainable, weights_initializer=initializer,
                #     padding='SAME', activation_fn=None,
                #     scope='merge/res{}'.format(5-i),data_format=data_format)

                last_fm = upsample + lateral
            else:
                last_fm = lateral

        global_fms.append(tf.nn.relu(last_fm,name='fpn_relu/res{}'.format(5-i)))



    global_fms.reverse()
    p6 = slim.max_pool2d(
        global_fms[-1], [2, 2], stride=2, padding='SAME', scope='fpn_pool_6', data_format=data_format)

    global_fms.append(p6)
    return global_fms


def add_a_pool(fms,data_format):

    p6 = slim.max_pool2d(
        fms[-1], [3, 3], stride=2, padding='SAME', scope='fpn_pool_6', data_format=data_format)
    fms.append(p6)
    return fms

def plain_resnet50_backbone(image,L2_reg,is_training=True,data_format='NHWC'):


    net_fms = simple_nn(image, L2_reg,is_training)

    print('simplenet backbone output:',net_fms)

    # with tf.variable_scope('CPN'):
    fpn_fms = create_global_net(net_fms, L2_reg,is_training,data_format=data_format)

    #fpn_fms=add_a_pool(net_fms,data_format)

    return fpn_fms


