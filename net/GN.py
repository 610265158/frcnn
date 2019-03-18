#-*-coding:utf-8-*-


import tensorflow as tf

def GroupNorm_nchw(x, group=32, gamma_initializer=tf.constant_initializer(1.)):
    """
    More code that reproduces the paper can be found at https://github.com/ppwwyyxx/GroupNorm-reproduce/.
    """
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims == 4, shape
    chan = shape[1]
    assert chan % group == 0, chan
    group_size = chan // group

    orig_shape = tf.shape(x)
    h, w = orig_shape[2], orig_shape[3]

    x = tf.reshape(x, tf.stack([-1, group, group_size, h, w]))

    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)

    new_shape = [1, group, group_size, 1, 1]

    beta = tf.get_variable('beta', [chan], initializer=tf.constant_initializer())
    beta = tf.reshape(beta, new_shape)

    gamma = tf.get_variable('gamma', [chan], initializer=gamma_initializer)
    gamma = tf.reshape(gamma, new_shape)

    out = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5, name='output')
    return tf.reshape(out, orig_shape, name='output')

def GroupNorm_nhwc(x,G=16,eps=1e-5,gamma_initializer=tf.constant_initializer(1.)):



    with tf.variable_scope('GN'):
        orig_shape = tf.shape(x)

        C = x.get_shape().as_list()[3]

        N,H,W=orig_shape[0],orig_shape[1],orig_shape[2]

        x=tf.reshape(x,[N,H,W,G,C//G])
        mean,var=tf.nn.moments(x,[1,2,4],keep_dims=True)
        x=(x-mean)/tf.sqrt(var+eps)

        x=tf.reshape(x,orig_shape)

        gamma = tf.get_variable('gamma', [C], initializer=gamma_initializer)
        gamma = tf.reshape(gamma, [1,1,1,C])

        beta = tf.get_variable('beta', [C], initializer=tf.constant_initializer())
        beta = tf.reshape(beta, [1, 1, 1, C])
    return x*gamma+beta



def GroupNorm_nhwc__(x,G=16,eps=1e-5,gamma_initializer=tf.constant_initializer(1.)):



    with tf.variable_scope('GN'):
        N,H,W,C=x.get_shape().as_list()
        x=tf.reshape(x,[N,H,W,G,C//G])
        mean,var=tf.nn.moments(x,[1,2,4],keep_dims=True)
        x=(x-mean)/tf.sqrt(var+eps)
        x=tf.reshape(x,[N,H,W,C])

        gamma = tf.get_variable('gamma', [1,1,1,C], initializer=gamma_initializer)
        beta = tf.get_variable('beta', [1,1,1,C], initializer=tf.constant_initializer())

    return x*gamma+beta