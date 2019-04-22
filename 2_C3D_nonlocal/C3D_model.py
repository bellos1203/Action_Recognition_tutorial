import tensorflow as tf
import tensorflow.contrib.slim as slim
from non_local import NonLocalBlock

def C3D(input_data, num_classes, keep_pro=0.5, non_local=False):
    with tf.variable_scope('C3D'):
        with slim.arg_scope([slim.conv3d],
                            padding='SAME',
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            activation_fn=tf.nn.relu,
                            kernel_size=[3, 3, 3],
                            stride=[1, 1, 1]
                            ):
            # Batch * 16 * 112 * 112 * 3
            net = slim.conv3d(input_data, 64, scope='conv1')
            net = slim.max_pool3d(net, kernel_size=[1, 2, 2], stride=[1, 2, 2], padding='SAME', scope='max_pool1')
            # net = NonLocalBlock(net, 64, scope='nonlocal_block_1')

            # Batch * 16 * 56 * 56 * 64
            net = 
            net = 
            if non_local:
                net = NonLocalBlock(net, 128, scope='nonlocal_block_2')

            # Batch * 8 * 28 * 28 * 128 
            net = 
            net = 
            if non_local:
                net = NonLocalBlock(net, 256, scope='nonlocal_block_3')

            # Batch * 4 * 14 * 14 * 256
            net = 
            net = 
            if non_local:
                net = NonLocalBlock(net, 512, scope='nonlocal_block_4')
            # Batch * 2 * 7 * 7 * 512
            net = slim.repeat(net, 2, slim.conv3d, 512, scope='conv5')
            net = slim.max_pool3d(net, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding='SAME', scope='max_pool5')

            # Batch * 1 * 4 * 4 * 512
            net = tf.reshape(net, [-1, 512 * 4 * 4])
            net = slim.fully_connected(net, 4096, weights_regularizer=slim.l2_regularizer(0.0005), scope='fc6')
            net = slim.dropout(net, keep_pro, scope='dropout1')
            net = slim.fully_connected(net, 4096, weights_regularizer=slim.l2_regularizer(0.0005), scope='fc7')
            net = slim.dropout(net, keep_pro, scope='dropout2')
            out = slim.fully_connected(net, num_classes, weights_regularizer=slim.l2_regularizer(0.0005), \
                                       activation_fn=None, scope='out')

            return out


