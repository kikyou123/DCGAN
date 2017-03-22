
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


# In[3]:

def conv2d(input_, output_dim, k_h = 5, k_w = 5, d_h = 2, d_w = 2, stddev = 0.02, name = "conv2d"):
    """Args :
        input_: a feature map [batch_size, height, weight, input_dim]
        output_dim: output feature map channels
        k_h, k_w: kernel size[k_h, k_w, input_dim, output_dim]
        d_h, d_w: stride[1, d_h, d_w, 1]
        stddev: weight initializer sigma
        name : scope
        
        Return:
        output feature map
        """
    
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], 
                            initializer = tf.truncated_normal_initializer(stddev = stddev))
        conv = tf.nn.conv2d(input_, w, strides = [1, d_h, d_w, 1], padding = 'SAME')
        
        b = tf.get_variable('biases', [output_dim], initializer = tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, b)
        
        return conv
    


# In[4]:

def deconv2d(input_, output_shape, k_h = 5, k_w = 5, d_h = 2, d_w = 2, stddev = 0.02, name = "deconv2d", input_dim = None):
    """Args :
        input_: a feature map [batch_size, height, weight, input_dim]
        output_shape: output feature map shape:[batch_size, height, weight, output_dim]
        k_h, k_w: kernel size[k_h, k_w, output_dim, input_dim]
        d_h, d_w: stride[1, d_h, d_w, 1]
        stddev: weight initializer sigma
        name : scope
        
        Return:
        output feature map
        """
    with tf.variable_scope(name):
        if not input_dim:
            w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], 
                                initializer = tf.random_normal_initializer(stddev = stddev))
        else:
            w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_dim], 
                                initializer = tf.random_normal_initializer(stddev = stddev))
            
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape = output_shape, strides = [1, d_h, d_w, 1])
        
        bias = tf.get_variable('biases', [output_shape[-1]], initializer = tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, bias)
        
        return deconv
    
    


# In[ ]:

def conv_cond_concat(x, y):
    """concatennate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], axis = 3)
    
    

def linear(input_, output_size, scope = None, stddev = 0.02, bias_start = 0.0):
    """Arg:
          input_: input tensor of shape [batch_size, input_size]
          output_size: output tensor dim
          
        Return:
          output tensor of shape [batch_size, output_size]
          """
    
    shape = input_.get_shape().as_list()
    
    with tf.variable_scope(scope or 'Linear'):
        matrix = tf.get_variable("Matrix", [shape[1], output_size],
                                 initializer = tf.random_normal_initializer(stddev = stddev))
        bias = tf.get_variable("bias", [output_size], initializer = tf.constant_initializer(0.0))
        
    return tf.matmul(input_, matrix) + bias

def lrelu(x, leak = 0.2, name = 'lrelu'):
    return tf.maximum(x, leak * x)

def relu(x):
    return tf.nn.relu(x)

def sigmoid(x):
    return tf.nn.sigmoid(x)

def batch_norm( x, train = True, epsilon = 1e-5, momentum = 0.9, name = "batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay = momentum, updates_collections = None, epsilon = epsilon, scale = True,
                                        is_training = train, scope = name)



def flatten(x):
    return tf.contrib.layers.flatten(x)