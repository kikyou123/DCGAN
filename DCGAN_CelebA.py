
# coding: utf-8

# In[1]:

import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from op import *
from utils import *
from load_hdf import *


# In[2]:

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


# In[3]:

dim = 64
batch_size = 100
z_dim = 100
ndfc = 1024
learning_rate = 0.0002
beta1 = 0.5
max_epoch = 25
ntrain = 202458
model_dir = "{}_{}_{}_{}".format('CelebA', batch_size, 64, 64)
data_dir = '/home/data/houruibing/CelebA'


# In[4]:

tr_data, tr_stream = faces(ntrain = ntrain, batch_size = batch_size, data_dir = data_dir )
tr_handle = tr_data.open()


# In[5]:

checkpoint_dir = 'checkpoint/CNN_GAN/CelebA'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# In[6]:

sample_dir = 'generate_images/CNN_GAN/CelebA'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


# In[7]:

log_dir = 'log/CNN_GAN/CelebA'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# In[8]:

sample_z = np.random.uniform(-1, 1, [batch_size, z_dim])
sample_num = batch_size
tr_handle = tr_data.open()
sample_inputs, = tr_data.get_data(tr_handle, slice(0, batch_size))# int 0-255
sample_inputs = transform(sample_inputs)


# In[9]:

# define D-network
def discriminator(image):
    """create a network that discriminates between images from a dataset and g
    genetated ones
    
    Args:
        image: a batch of real images [batch_size, height, weight, channels]
    Rerurns:
        A tensor that represents the probability of the real image"""
    h1 = lrelu(conv2d(image, dim, name = 'd_h1_conv'))
    h2 = lrelu(batch_norm(conv2d(h1, dim * 2, name = 'd_h2_conv'), name = 'd_bn2'))
    h3 = lrelu(batch_norm(conv2d(h2, dim * 4, name = 'd_h3_conv'), name = 'd_bn3'))
    h4 = lrelu(batch_norm(conv2d(h3, dim * 8, name = 'd_h4_conv'), name = 'd_bn4'))
    h5 = linear(flatten(h4), 1, 'd_h5_lin')
    #h5 = linear(tf.reshape(h4, [batch_size, -1]), 1, 'd_h5_lin')
    #h5 = lrelu(batch_norm(linear(tf.reshape(h4, [batch_size, -1]), ndfc, 'd_h5_lin'), name = 'd_bn5'))
    #h6 = linear(h5, 1, 'd_h6_lin')
    
    return sigmoid(h5)


# In[10]:

def generator(z):
    """Create a network that genetates images
    Args:
          z: input random noise of size [batch_size, dim_z]

    Returns:
          A deconvolutional network that generated images of size[batch_size, height, weight, channle]"""
    h0 = relu(batch_norm(linear(z, dim * 8 * 4 * 4, 'g_h0_lin'), name = 'g_bn0'))
    h0 = tf.reshape(h0, [-1, 4, 4, dim * 8])
    h1 = relu(batch_norm(deconv2d(h0, [batch_size, 8, 8, dim * 4], name = 'g_h1'), name = 'g_bn1'))
    h2 = relu(batch_norm(deconv2d(h1, [batch_size, 16, 16, dim * 2], name = 'g_h2'), name = 'g_bn2'))
    h3 = relu(batch_norm(deconv2d(h2, [batch_size, 32, 32, dim], name = 'g_h3'), name = 'g_bn3'))
    h4 = deconv2d(h3, [batch_size, 64, 64, 3], name = 'g_h4')
    h4 = tf.nn.tanh(h4)
    
    return h4
    


# In[11]:

def train():
    
    with tf.variable_scope('Gen') as scope:
        z = tf.placeholder(tf.float32, shape = [None, z_dim])
        z_sum = tf.summary.histogram("z", z)
        G = generator(z)
        G_sum = tf.summary.image("G", G)
        
    with tf.variable_scope('Disc') as scope:
        x = tf.placeholder(tf.float32, shape = [None, 64, 64, 3])
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
     
    d_optim = tf.train.AdamOptimizer(0.0002, beta1 = beta1).minimize(d_loss, var_list = d_params)
    g_optim = tf.train.AdamOptimizer(0.001, beta1 = beta1).minimize(g_loss, var_list = g_params)
    saver = tf.train.Saver()
    config = tf.ConfigProto()  
    #config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    sess = tf.Session(config = config)
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
        batch_idx = ntrain // batch_size
        idx = -1 # each epoch iter time
        for imb, in tqdm(tr_stream.get_epoch_iterator(), total = ntrain / batch_size):
            idx += 1
            batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
            images = transform(imb)
            
            #update D
            _, summary_str = sess.run([d_optim, d_sum], 
                                      feed_dict = {x: images,
                                                   z: batch_z})
            writer.add_summary(summary_str, counter)
            
            #update G
            _, summary_str = sess.run([g_optim, g_sum], 
                                      feed_dict = {z: batch_z})
            writer.add_summary(summary_str, counter)
            
            #update G twice
            #_, summary_str = sess.run([g_optim, g_sum], feed_dict = {z: batch_z})
            #writer.add_summary(summary_str, counter)
            
            counter = counter + 1
            errD = sess.run(d_loss, {z: batch_z, 
                                     x: images})
            errG = sess.run(g_loss, {z: batch_z})
            print ("Epoch: [%2d] [%4d%4d] d_loss: %.8f, g_loss: %.8f" % (epoch, idx, batch_idx,errD, errG ))
            
            # generate samples
            if counter % 100 == 0:
                samples, d1_loss, g1_loss = sess.run([G, d_loss, g_loss],
                                                     feed_dict = {z: sample_z, 
                                                                  x: sample_inputs})
                samples = inverse_transform(samples)
                save_images(samples, [4, 8], './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
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


# In[13]:

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
    


# In[14]:

def main(_):
     train()

if __name__ == '__main__':
    tf.app.run()

