
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

z_dim = 128
dim = 64
#dfc_dim = 1024
#gfc_dim = 1024
ntrain = 202458 # total traning datasets'
batch_size = 100
max_epoch = 25
beta1 = 0.5
model_dir = "{}_{}_{}_{}".format('CelebA', batch_size, 28, 28)
data_dir = '/home/data/houruibing/CelebA'
is_train = False


# In[4]:

# save model parameters
checkpoint_dir = 'checkpoint/InfoGAN/CelebA'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# In[5]:

#save generate images
sample_dir = 'generate_images/InfoGAN/CelebA'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


# In[6]:

#save model 
log_dir = 'log/InfoGAN/CelebA'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# In[7]:

tr_data, tr_stream = faces(ntrain = ntrain, batch_size = batch_size, data_dir = data_dir )
tr_handle = tr_data.open()


# In[8]:

#define sample input z_noise
sample_num = 100
sample_z_in = np.random.uniform(-1, 1, [sample_num, z_dim])
#define sample images
sample_inputs, = tr_data.get_data(tr_handle, slice(0, sample_num))# int 0-255
sample_inputs = transform(sample_inputs)
#define sample input c_catogorical
sample_lat = []
for i in range(10):
    a = np.array([e for e in range(10)])
    sample_lat.append(a)
sample_lat = np.reshape(sample_lat, [100, 1])
for k in range(9):
    sample_lat1 = []
    for i in range(10):
        for j in range(10):
            sample_lat1.append(i)
    sample_lat1 = np.reshape(sample_lat1, [100, 1])
    sample_lat = np.concatenate([sample_lat, sample_lat1], axis = 1)


# In[9]:

# define G network
def generator(z):
    
    """Args: 
            z: input random noise + structured semantic features
       Return:
             generate image of size [Batch_size, height, weight, channel]"""
    with tf.variable_scope("Gen") as scope:
        h0 = relu(batch_norm(linear(z, dim * 8 * 4 * 4, 'g_h0_lin'), name = 'g_bn0'))
        h0 = tf.reshape(h0, [-1, 4, 4, dim * 8])
        h1 = relu(batch_norm(deconv2d(h0, [batch_size, 8, 8, dim * 4], name = 'g_h1'), name = 'g_bn1'))
        h2 = relu(batch_norm(deconv2d(h1, [batch_size, 16, 16, dim * 2], name = 'g_h2'), name = 'g_bn2'))
        h3 = relu(batch_norm(deconv2d(h2, [batch_size, 32, 32, dim], name = 'g_h3'), name = 'g_bn3'))
        h4 = deconv2d(h3, [batch_size, 64, 64, 3], name = 'g_h4')
        h4 = tf.nn.tanh(h4)
        
        return h4


# In[10]:

#define D network and Q network
def discriminator(image, cat_list = None, conts = 0, reuse = False):
    
    
    """Args:
           image: a batch of images [batch_size, height, weight, channel]
           cat_list: each entry in this list define a categorical variable of a specific size[1, n_catV]
           conts: the number of continous variables.scalar
        Returns:
           d_out: the pro of the input is real images [batch_size, 1]
           q_cat_outs: the distribution of the categorial variable [[batch_size, cat_list[i]]]
           q_cont_mean: the mean of the distribution of the continous variables [batch_size, n_contV]
           q_cont_sigma : the log-sigma2 of the distribution of the continous variables [batch_size, n_contV]
    """
    with tf.variable_scope("Disc") as scope:
        
        if reuse:
            scope.reuse_variables()
            
        h1 = lrelu(conv2d(image, dim, name = 'd_h1_conv'))
        h2 = lrelu(batch_norm(conv2d(h1, dim * 2, name = 'd_h2_conv'), name = 'd_bn2'))
        h3 = lrelu(batch_norm(conv2d(h2, dim * 4, name = 'd_h3_conv'), name = 'd_bn3'))
        h4 = lrelu(batch_norm(conv2d(h3, dim * 8,  name = 'd_h4_conv'), name = 'd_bn4'))
        d_out = sigmoid(linear(flatten(h4), 1, 'd_h5_lin'))
    
    #define q network
    with tf.variable_scope("Q") as scope:
        
        if reuse:
            scope.reuse_variables()
            
        q_a = lrelu(batch_norm(linear(flatten(h4), 128, 'q_h1_lin'), name = 'q_bn1'))
        #continue latent variable
        q_cont_mean = None
        q_cont_sigma = None
        if conts > 0:
            # mean
            q_cont_mean = tf.nn.tanh(linear(q_a, conts, 'q_cont_mean'))
            #log-sigma2
            q_cont_sigma = linear(q_a, conts, 'q_cont_sigma')
        #catagorial variable
        q_cat_outs = []
        for idx, vax in enumerate(cat_list):
            q_cat_a = tf.nn.softmax(linear(q_a, vax, 'q_cat_lin' + str(idx)))
            q_cat_outs.append(q_cat_a)
        
        return d_out, q_cat_outs, q_cont_mean, q_cont_sigma
        
        


# In[11]:

# save model
def save(checkpoint_dir, step, saver, sess):
    model_name = 'InfoGAN.model'
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

# define latent variable
categorical_list = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10] #each entry in this list define a categorical variable of a specific size[1, n_catV]
number_continue = 0 #the number of continous variables

#this placeholders are used for input into the gen and dis, respectively
z_in = tf.placeholder(tf.float32, shape = [None, z_dim])
x = tf.placeholder(tf.float32, shape = [None, 64, 64, 3])

# these placeholders load the latent variables
latent_cat_in  = tf.placeholder(tf.int32, shape = [None, len(categorical_list)])
latent_cat_list = tf.split(latent_cat_in, len(categorical_list), 1)
if number_continue > 0:
    latent_cont_in = tf.placeholder(tf.float32, shape = [None, number_continue])

#z_lat : gen input [batch_size, z_dim + 10 + 2]
z_lat = z_in
oh_list = [] # categorical latent for caculate loss
for idx, vax in enumerate(categorical_list):
    latent_oh = tf.one_hot(tf.reshape(latent_cat_list[idx], [-1]), depth = vax)# [batch_size, vax]
    oh_list.append(latent_oh)
    z_lat = tf.concat([z_lat, latent_oh], axis = 1)
if number_continue > 0:
    z_lat = tf.concat([z_lat, z_in], axis = 1)

z_sum = tf.summary.histogram("z_lat", z_lat)

#generate images
G = generator(z_lat)
G_sum = tf.summary.image("G", G)

#caculate dis outputs
D1, _, _, _ = discriminator(x, categorical_list, number_continue)# reduce pro for real images
D2, q_cat_outs, q_cont_mean, q_cont_sigma = discriminator(G, categorical_list, number_continue, reuse = True)#reduce pro for fake i

#caculate dis and gen loss
d_loss_real = tf.reduce_mean(-tf.log(D1))
d_loss_fake = tf.reduce_mean(-tf.log(1 - D2))
g_loss = tf.reduce_mean(-tf.log(D2))

d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)

d_loss = d_loss_real + d_loss_fake
g_loss_sum = tf.summary.scalar("g_loss", g_loss)
d_loss_sum = tf.summary.scalar("d_loss", d_loss)

#caculate q-net continue variables loss
q_cont_loss = tf.constant(0.0)
if number_continue > 0:
    q_cont_loss = tf.reduce_mean(0.5 * tf.square(q_cont_mean - latent_cont_in) / tf.exp(q_cont_sigma) + 0.5 * q_cont_sigma)

#caculate q-net catagorical variables loss
q_cat_loss = tf.constant(0.0)
for idx, latent_var in enumerate(oh_list):
    q_cat_loss += -(tf.reduce_mean(latent_var * tf.log(q_cat_outs[idx])))

#q-net loss
q_loss = q_cat_loss + q_cont_loss
q_loss_sum = tf.summary.scalar("q_loss", q_loss)

#update param
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]
q_vars = t_vars

#define optimizer
d_optim = tf.train.AdamOptimizer(0.0002, beta1 = beta1).minimize(d_loss, var_list = d_vars)
g_optim = tf.train.AdamOptimizer(0.001, beta1 = beta1).minimize(g_loss, var_list = g_vars)
q_optim = tf.train.AdamOptimizer(0.0002, beta1 = beta1).minimize(q_loss, var_list = q_vars)

#define saver
saver = tf.train.Saver()

#define sess
config = tf.ConfigProto()  
config.gpu_options.allow_growth  = True
sess = tf.Session(config=config)  

#init all variables
init = tf.global_variables_initializer()
sess.run(init)

g_sum = tf.summary.merge([z_sum, G_sum, d_loss_fake_sum, g_loss_sum])
d_sum = tf.summary.merge([z_sum, d_loss_real_sum, d_loss_sum])
q_sum = tf.summary.merge([z_sum, q_loss_sum])
writer = tf.summary.FileWriter(log_dir, sess.graph)

counter = 1 # update time

if is_train:
    
    for epoch in range(max_epoch):
        batch_idx = ntrain // batch_size
        idx = -1 #each epoch iter time
        for imb, in tqdm(tr_stream.get_epoch_iterator(), total = ntrain / batch_size):
            idx += 1
            images = transform(imb)
            zs = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
            if number_continue > 0:
                lcont = np.random.uniform(-1, 1, [batch_size, number_continue]).astype(np.float32)
            lcat = []
            for _, vax in enumerate(categorical_list):
                lcat_i = np.random.randint(0, vax, [batch_size, 1])
                lcat.append(lcat_i)
            lcat = np.reshape(np.array(lcat), [batch_size, len(categorical_list)])
            
            #update D
            _, summary_str = sess.run([d_optim, d_sum],{x: images, z_in: zs, latent_cat_in: lcat})
            writer.add_summary(summary_str, counter)
            
            #update G
            _, summary_str = sess.run([g_optim, g_sum], {z_in: zs, latent_cat_in: lcat})
            writer.add_summary(summary_str, counter)
            
            #update Q
            _, summary_str = sess.run([q_optim, q_sum], {z_in: zs, latent_cat_in: lcat})
            writer.add_summary(summary_str, counter)
            
            counter += 1 #update time add one
            #cacluate loss
            
            errD = sess.run(d_loss, {x: images, z_in: zs, latent_cat_in: lcat})
            errG = sess.run(g_loss, {z_in: zs, latent_cat_in: lcat})
            errQ = sess.run(q_loss, {z_in: zs, latent_cat_in: lcat})
            print ("Epoch: [%2d] [%4d %4d] d_loss: %.8f, g_loss: %.8f, q_loss: %.8f" % (epoch, idx, batch_idx, errD, errG, errQ ))
            
            #generate samples
            if counter % 100 == 0:
                samples, d1_loss, g1_loss, q1_loss = sess.run([G, d_loss, g_loss, q_loss],
                                                             {x: sample_inputs, z_in: sample_z_in, latent_cat_in: sample_lat})
                samples = inverse_transform(samples)
                save_images(samples, [10, 10], './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
                print("[Sample] d_loss: %.8f, g_loss: %.8f, q_loss: %.8f" % (d1_loss, g1_loss, q1_loss)) 
                
                
            # save parameters
            if counter % 500 == 0:
                save(checkpoint_dir, counter, saver, sess)
    
else:
    if load(checkpoint_dir):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")
    sample_directory = 'generate_images/InfoGAN/CelebA/test1'
    if not os.path.exists(sample_directory):
        os.makedirs(sample_directory)
    
    # input noise z each row is the same
    sample_z_in = []
    for i in range(10):
        a = np.random.uniform(-1, 1, [1, z_dim]).astype(np.float32)
        for j in range(10):
            sample_z_in.append(a)
    sample_z_in = np.reshape(np.array(sample_z_in), [100, z_dim])
    
    for n in range(10):
        #define sample input c_catogorical
        sample_lat = []
        for i in range(10):
            a = np.array([e for e in range(10)])
            sample_lat.append(a)
        sample_lat = np.reshape(sample_lat, [100, 1])
        #for k in range(9):
           #sample_lat1 = []
           # for i in range(10):
           #     for j in range(10):
           #         sample_lat1.append(i)
           # sample_lat1 = np.reshape(sample_lat1, [100, 1])
           # sample_lat = np.concatenate([sample_lat, sample_lat1], axis = 1)
        sample_lat1 = np.zeros((100, 9))
        sample_lat = np.concatenate([sample_lat, sample_lat1], axis = 1)
    
        # change cat 3
        sample_lat[:, [0, n]] = sample_lat[:, [n, 0]]
    
        samples = sess.run(G, {z_in: sample_z_in, latent_cat_in: sample_lat})
        save_images(samples, [10, 10], './{}/{}_{:2d}.png'.format(sample_directory, 'cat5', n ))
        print "save success %d" % (n)
