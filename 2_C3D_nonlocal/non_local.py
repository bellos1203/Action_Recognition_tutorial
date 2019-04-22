import tensorflow as tf
import tensorflow.contrib.slim as slim

def NonLocalBlock(input_x, out_channels, sub_sample=True, is_bn=True, scope='NonLocalBlock'):
    batchsize, clips, height, width, in_channels = input_x.get_shape().as_list()
    with tf.variable_scope(scope) as sc:
        with tf.variable_scope('g') as scope:
            g = slim.conv3d(input_x, out_channels, kernel_size=1, stride=1, scope='g')
            if sub_sample:
                g = slim.max_pool3d(g, [1,2,2], stride=[1,2,2], scope='g_max_pool')

        with tf.variable_scope('phi') as scope:
            '''
            phi = 
            '''
            if sub_sample:
                phi = slim.max_pool3d(phi, [1,2,2], stride=[1,2,2], scope='phi_max_pool')

        with tf.variable_scope('theta') as scope:
            '''
            theta = 
            '''
        
        '''
        g_x = 
        '''
        
        '''
        theta_x = 
        '''
        
        '''
        phi_x = 
        transposed_phi_x = 
        '''
        
        '''
        f =            # (theta, phi) matrix multiplication
        f_softmax =    # softmax
        y =            # (f_softmax, g)
        y =            # reshape
        '''
        
        with tf.variable_scope('w') as scope:
            '''
            w_y =      # Z operation
            '''
            if is_bn:
                w_y = slim.batch_norm(w_y)
        '''
        z =           # add y to x
        '''
    return z

