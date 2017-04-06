import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def GeneratorCNN(z, z_num, output_num, repeat_num, hidden_num, reuse):
    with tf.variable_scope("G", reuse=reuse) as vs:
        x = slim.fully_connected(z, np.prod([z_num, 8, 8]))
        x = tf.reshape(x, [-1, 8, 8, hidden_num])
        
        for idx in range(repeat_num):
            x = tf.nn.elu(slim.conv2d(x, hidden_num, 3, 1))
            x = tf.nn.elu(slim.conv2d(x, hidden_num, 3, 1))
            if idx < repeat_num - 1:
                _, h, w, _ = int_shape(x)
                x = tf.image.resize_nearest_neighbor(x, (h*2, w*2))

        out = slim.conv2d(x, 3, 3, 1)
        #out = tf.nn.tanh(out)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables

def DiscriminatorCNN(x, input_channel, z_num, repeat_num, hidden_num):
    with tf.variable_scope("D") as vs:
        # Encoder
        x = tf.nn.elu(slim.conv2d(x, hidden_num, 3, 1))

        prev_channel_num = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            x = tf.nn.elu(slim.conv2d(x, channel_num, 3, 1))
            x = tf.nn.elu(slim.conv2d(x, channel_num, 3, 1))
            if idx < repeat_num - 1:
                x = slim.conv2d(x, channel_num, 3, 2)
            else:
                pass
                #x = slim.conv2d(x, channel_num, 3, 1)

        conv1_output_dim = [8, 8, channel_num]
        x = slim.fully_connected(tf.reshape(x, [-1, np.prod(conv1_output_dim)]), z_num)

        # Decoder
        conv2_input_dim = [8, 8, hidden_num]
        x = slim.fully_connected(x, np.prod(conv2_input_dim))
        x = tf.reshape(x, [-1] + conv2_input_dim)
        
        for idx in range(repeat_num):
            x = tf.nn.elu(slim.conv2d(x, hidden_num, 3, 1))
            x = tf.nn.elu(slim.conv2d(x, hidden_num, 3, 1))
            if idx < repeat_num - 1:
                _, h, w, _ = int_shape(x)
                x = tf.image.resize_nearest_neighbor(x, (h*2, w*2))

        out = slim.conv2d(x, input_channel, 3, 1)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables
