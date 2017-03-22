
# coding: utf-8

# In[1]:

import os

import tensorflow as tf
import numpy as np
from op import *
from utils import *
from tensorflow.examples.tutorials.mnist import input_data


# In[2]:

seed = 45
np.random.seed(seed)
tf.set_random_seed(seed)


# In[3]:

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


# In[4]:

dim = 64
batch_size = 64
z_dim = 100
dfc_dim = 1024
gfc_dim = 1024
learning_rate = 0.0002
beta1 = 0.5
max_epoch = 25
updates_per_epoch = 1000
BATCH = 60000
model_dir = "{}_{}_{}_{}".format('MNIST', batch_size, 28, 28)
y_dim = 10;


# In[5]:

checkpoint_dir = 'checkpoint/CNN_GAN/mnist'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# In[6]:

sample_dir = 'generate_images/CNN_GAN/mnist'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


# In[7]:

log_dir = 'log/CNN_GAN/mnist'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# In[8]:

sample_z = np.random.uniform(-1, 1, [batch_size, z_dim])
sample_num = 64
sample_inputs, sample_labels = mnist.train.next_batch(sample_num)


# In[9]:

# define D-network
def discriminator(image, y = None):
    """create a network that discriminates between images from a dataset and g
    genetated ones
    
    Args:
        image: a batch of real images [batch_size, height, weight, channels]
        y: images labels [batch_size, y_dim]
    Rerurns:
        A tensor that represents the probability of the real image"""
    
    if not y_dim:
        h1 = lrelu(conv2d(image, dim, name = 'd_h1_conv'))
        h2 = lrelu(batch_norm(conv2d(h1, dim * 2, name = 'd_h2_conv'), name = 'd_bn2'))
        h3 = lrelu(batch_norm(linear(tf.reshape(h2, [batch_size, -1]), dfc_dim, 'd_h3_lin'), name = 'd_bn3'))
        h4 = linear(h3, 1, 'd_h4_lin')
    else:
        yb = tf.reshape(y, [batch_size, 1, 1, y_dim])
        x = conv_cond_concat(image, yb)
        h1 = lrelu(conv2d(x, dim, name = 'd_h1_conv'))
        h1 = conv_cond_concat(h1, yb)
        h2 = lrelu(batch_norm(conv2d(h1, dim * 2, name = 'd_h2_conv'), name = 'd_bn2'))
        h2 = tf.concat([tf.reshape(h2, [batch_size, -1]), y], axis = 1)
        h3 = lrelu(batch_norm(linear(h2, dfc_dim, 'd_h3_lin'), name = 'd_bn3'))
        h4 = linear(h3, 1, 'd_h4_lin')
        
    
    return sigmoid(h4)
    


# In[10]:

def generator(z, y = None):
    """Create a network that genetates images
    Args: z: input random noise of size [batch_size, dim_z]
          y: images label [batch_size, y_dim]
    
    Returns:
          A deconvolutional network that generated images of size[batch_size, height, weight, channle]"""
    if not y_dim:
        h0 = relu(batch_norm(linear(z, gfc_dim, 'g_h0_lin'), name = 'g_bn0'))
        h1 = relu(batch_norm(linear(h0, 7 * 7 * dim * 2, 'g_h1_lin'), name = 'g_bn1'))
        h1 = tf.reshape(h1, [-1, 7, 7, dim * 2])
        h2 = relu(batch_norm(deconv2d(h1, [batch_size, 14, 14, dim], name = 'g_h2'), name = 'g_bn2'))
        h3 = deconv2d(h2, [batch_size, 28, 28, 1], name = 'g_h3')
        h3 = sigmoid(h3)
    else:
        yb = tf.reshape(y, [batch_size, 1, 1, y_dim])
        z = tf.concat([z, y], axis = 1)
        h0 = relu(batch_norm(linear(z, gfc_dim, 'g_ho_lin'), name = 'g_bh0'))
        h0 = tf.concat([h0, y], axis = 1)
        h1 = relu(batch_norm(linear(h0, 7 * 7 * dim * 2, 'g_h1_lin'), name = 'g_bn1'))
        h1 = tf.reshape(h1, [-1, 7, 7, dim * 2])
        h1 = conv_cond_concat(h1, yb)
        h2 = relu(batch_norm(deconv2d(h1, [batch_size, 14, 14, dim], name = 'g_h2'), name = 'g_bn2'))
        h2 = conv_cond_concat(h2, yb)
        h3 = deconv2d(h2, [batch_size, 28, 28, 1], name = 'g_h3')
        h3 = sigmoid(h3)
    
    return h3
    


# In[11]:

def train():
    
    if y_dim:
        y = tf.placeholder(tf.float32, shape = [batch_size, y_dim])
    with tf.variable_scope('Gen') as scope:
        z = tf.placeholder(tf.float32, shape = [batch_size, z_dim])
        z_sum = tf.summary.histogram("z", z)
        if y_dim:
            G = generator(z, y)
        else:
            G = generator(z)
        G_sum = tf.summary.image("G", G)
        
    with tf.variable_scope('Disc') as scope:
        x = tf.placeholder(tf.float32, shape = (batch_size, 28, 28, 1))
        if y_dim:
            D1 = discriminator(x, y)
            scope.reuse_variables()
            D2 = discriminator(G, y)
        else:
            D1 = discriminator(x)
            scope.reuse_variables()
            D2 = discriminator(G)
        d1_sum = tf.summary.histogram("d1", D1)
        d2_sum = tf.summary.histogram("d2", D2)
        
    d_loss_real = tf.reduce_mean(-tf.log(D1))
    d_loss_fake = tf.reduce_mean(-tf.log(1 - D2))
    g_loss = tf.reduce_mean(-tf.log(D2))

    d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
    d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)

    d_loss = d_loss_real + d_loss_fake

    g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    d_loss_sum = tf.summary.scalar("d_loss", d_loss)

    d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Disc')
    g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Gen')
     
    d_optim = tf.train.AdamOptimizer(learning_rate, beta1 = beta1).minimize(d_loss, var_list = d_params)
    g_optim = tf.train.AdamOptimizer(learning_rate, beta1 = beta1).minimize(g_loss, var_list = g_params)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4 # 占用GPU40%的显存 
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    
    g_sum = tf.summary.merge([z_sum, d2_sum, G_sum, d_loss_fake_sum, g_loss_sum])
    d_sum = tf.summary.merge([z_sum, d1_sum, d_loss_real_sum, d_loss_sum])
    writer = tf.summary.FileWriter(log_dir, sess.graph)

   # if load(checkpoint_dir):
    #    print(" [*] Load SUCCESS")
   # else:
    #    print(" [!] Load failed...")
        
    counter = 1
    
    for epoch in range(max_epoch):
        batch_idx = BATCH // batch_size
        
        for idx in xrange(0, batch_idx):
            
            batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
            
            images, labels = mnist.train.next_batch(batch_size)
            images = images.astype(np.float32)
            labels = labels.astype(np.float32)
            
            #update D
            _, summary_str = sess.run([d_optim, d_sum], 
                                      feed_dict = {x: np.reshape(images, [batch_size, 28, 28, 1]),
                                                   y: labels,
                                                   z: batch_z})
            writer.add_summary(summary_str, counter)
            
            #update G
            _, summary_str = sess.run([g_optim, g_sum], 
                                      feed_dict = {y:labels,
                                                   z: batch_z})
            writer.add_summary(summary_str, counter)
            
            #update G twice
            #_, summary_str = sess.run([g_optim, g_sum], feed_dict = {z: batch_z})
            #writer.add_summary(summary_str, counter)
            
            counter = counter + 1
            errD = sess.run(d_loss, {z: batch_z, 
                                     x: np.reshape(images, [batch_size, 28, 28, 1]),
                                     y: labels})
            errG = sess.run(g_loss, {z: batch_z, y: labels})
            print ("Epoch: [%2d] [%4d%4d] d_loss: %.8f, g_loss: %.8f" % (epoch, idx, batch_idx,errD, errG ))
            
            # generate samples
            if counter % 100 == 0:
                samples, d1_loss, g1_loss = sess.run([G, d_loss, g_loss],
                                                     feed_dict = {z: sample_z, 
                                                                  x: np.reshape(sample_inputs, [sample_num, 28, 28, 1]),
                                                                  y: sample_labels})
                save_images(samples, [8, 8], './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d1_loss, g1_loss)) 
                
                
            # save parameters
            if counter % 500 == 0:
                save(checkpoint_dir, counter, saver, sess)
    
    sess.close()
        


# In[12]:

def save(checkpoint_dir, step, saver, sess):
    model_name = 'GAN_CNN.model'
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step = step)


# In[ ]:

def load(checkpoint_dir):
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
      print(" [*] Success to read {}".format(ckpt_name))
      return True
    else:
      print(" [*] Failed to find a checkpoint")
      return False
    


# In[ ]:

def main(_):
    train()

if __name__ == '__main__':
    tf.app.run()


# In[ ]:



