
# coding: utf-8

# In[1]:

import os

import tensorflow as tf
import numpy as np
from op import *
from utils import *
from tensorflow.examples.tutorials.mnist import input_data


# In[2]:

seed = 43
np.random.seed(seed)
tf.set_random_seed(seed)


# In[3]:

mnist = input_data.read_data_sets("MNIST_data/", one_hot = 'True')


# In[4]:

z_dim = 40
dim = 64
dfc_dim = 1024
gfc_dim = 1024
BATCH = 55000 # total Mnist traning datasets'
batch_size = 100
max_epoch = 25
beta1 = 0.5
model_dir = "{}_{}_{}_{}".format('MNIST', batch_size, 28, 28)
is_train = False


# In[5]:

# save model parameters
checkpoint_dir = 'checkpoint/InfoGAN/mnist'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# In[6]:

#save generate images
sample_dir = 'generate_images/InfoGAN/mnist'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


# In[7]:

#save model 
log_dir = 'log/InfoGAN/mnist'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# In[8]:

#define sample input z_noise
sample_num = 100
sample_z_in = np.random.uniform(-1, 1, [sample_num, z_dim])
sample_inputs, sample_labels = mnist.train.next_batch(sample_num)
sample_inputs = np.reshape(sample_inputs, [sample_num, 28, 28, 1])
#define sample input c_catogorical
sample_lat = []
for i in range(10):
    a = np.array([e for e in range(10)])
    sample_lat.append(a)
sample_lat = np.reshape(sample_lat, [100, 1])
#define sample input c_cont
sample_cont_a = []
for i in range(10):
    a = np.linspace(-1, 1, 10)
    sample_cont_a.append(a)
sample_cont_a = np.reshape(sample_cont_a, [100, 1])
sample_cont_b = np.zeros_like(sample_cont_a)
sample_cont = np.hstack([sample_cont_a, sample_cont_b])


# In[9]:

# define G network
def generator(z):
    
    """Args: 
            z: input random noise + structured semantic features
       Return:
             generate image of size [Batch_size, height, weight, channel]"""
    with tf.variable_scope("Gen") as scope:
        h0 = relu(batch_norm(linear(z, gfc_dim, 'g_h0_lin'), name = 'g_bn0'))
        h1 = relu(batch_norm(linear(h0, 7 * 7 * dim * 2, 'g_h1_lin'), name = 'g_bn1'))
        h1 = tf.reshape(h1, [batch_size, 7, 7, dim * 2])
        h2 = relu(batch_norm(deconv2d(h1, [batch_size, 14, 14, dim], k_h = 4, k_w = 4, name = 'g_h2'), name = 'g_bn2'))
        h3 = deconv2d(h2, [batch_size, 28, 28, 1], k_h = 4, k_w = 4, name = 'g_h3')
        h3 = sigmoid(h3)
        
        return h3


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
            
        h1 = lrelu(conv2d(image, dim, k_h = 4, k_w = 4, name = 'd_h1_conv'), leak = 0.1)
        h2 = lrelu(batch_norm(conv2d(h1, dim * 2, k_h = 4, k_w = 4, name = 'd_h2_conv'), name = 'd_bn2'), leak = 0.1)
        h3 = lrelu(batch_norm(linear(flatten(h2), dfc_dim, 'd_h3_lin'), name = 'd_bn3'), leak = 0.1)
        d_out = sigmoid(linear(h3, 1, 'd_h4_lin'))
    
    #define q network
    with tf.variable_scope("Q") as scope:
        
        if reuse:
            scope.reuse_variables()
            
        q_a = lrelu(batch_norm(linear(h3, 128, 'q_h1_lin'), name = 'q_bn1'), leak = 0.1)
        #continue latent variable
        q_cont_outs = None
        if conts > 0:
            q_cont_outs = tf.nn.tanh(linear(q_a, conts, 'q_cont_mean'))
        #catagorial variable
        q_cat_outs = []
        for idx, vax in enumerate(cat_list):
            q_cat_a = tf.nn.softmax(linear(q_a, vax, 'q_cat_lin' + str(idx)))
            q_cat_outs.append(q_cat_a)
        
        return d_out, q_cat_outs, q_cont_outs
        
        


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
categorical_list = [10] #each entry in this list define a categorical variable of a specific size[1, n_catV]
number_continue = 2 #the number of continous variables

#this placeholders are used for input into the gen and dis, respectively
z_in = tf.placeholder(tf.float32, shape = [batch_size, z_dim])
x = tf.placeholder(tf.float32, shape = [batch_size, 28, 28, 1])

# these placeholders load the latent variables
latent_cat_in  = tf.placeholder(tf.int32, shape = [batch_size, len(categorical_list)])
latent_cat_list = tf.split(latent_cat_in, len(categorical_list), 1)
latent_cont_in = tf.placeholder(tf.float32, shape = [batch_size, number_continue])

#z_lat : gen input [batch_size, z_dim + 10 + 2]
z_lat = latent_cont_in
oh_list = [] # categorical latent for caculate loss
for idx, vax in enumerate(categorical_list):
    latent_oh = tf.one_hot(tf.reshape(latent_cat_list[idx], [-1]), depth = vax)# [batch_size, vax]
    oh_list.append(latent_oh)
    z_lat = tf.concat([z_lat, latent_oh], axis = 1)
z_lat = tf.concat([z_lat, z_in], axis = 1)

z_sum = tf.summary.histogram("z_lat", z_lat)

#generate images
G = generator(z_lat)
G_sum = tf.summary.image("G", G)

#caculate dis outputs
D1, _, _= discriminator(x, categorical_list, number_continue)# reduce pro for real images
D2, q_cat_outs, q_cont_outs = discriminator(G, categorical_list, number_continue, reuse = True)#reduce pro for fake i

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
    q_cont_loss = tf.reduce_mean(0.5 * tf.square(q_cont_outs - latent_cont_in))

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
d_optim = tf.train.AdamOptimizer(0.0001, beta1 = beta1).minimize(d_loss, var_list = d_vars)
g_optim = tf.train.AdamOptimizer(0.001, beta1 = beta1).minimize(g_loss, var_list = g_vars)
q_optim = tf.train.AdamOptimizer(0.0001, beta1 = beta1).minimize(q_loss, var_list = q_vars)

#define saver
saver = tf.train.Saver()

#define sess
config = tf.ConfigProto()  
#config.gpu_options.allow_growth  =True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
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
        batch_idx = BATCH // batch_size
        for idx in xrange(0, batch_idx):
            images, labels = mnist.train.next_batch(batch_size)
            images = np.reshape(images.astype(np.float32), [batch_size, 28, 28, 1])
            zs = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
            lcont = np.random.uniform(-1, 1, [batch_size, number_continue]).astype(np.float32)
            lcat = []
            for _, vax in enumerate(categorical_list):
                lcat_i = np.random.randint(0, vax, [batch_size, 1])
                lcat.append(lcat_i)
            lcat = np.reshape(np.array(lcat), [batch_size, len(categorical_list)])
            
            #update D
            _, summary_str = sess.run([d_optim, d_sum],{x: images, z_in: zs, latent_cat_in: lcat, latent_cont_in: lcont})
            writer.add_summary(summary_str, counter)
            
            #update G
            _, summary_str = sess.run([g_optim, g_sum], {z_in: zs, latent_cat_in: lcat, latent_cont_in: lcont})
            writer.add_summary(summary_str, counter)
            
            #update Q
            _, summary_str = sess.run([q_optim, q_sum], {z_in: zs, latent_cat_in: lcat, latent_cont_in: lcont})
            writer.add_summary(summary_str, counter)
            
            counter += 1 #update time add one
            #cacluate loss
            
            errD = sess.run(d_loss, {x: images, z_in: zs, latent_cat_in: lcat, latent_cont_in: lcont})
            errG = sess.run(g_loss, {z_in: zs, latent_cat_in: lcat, latent_cont_in: lcont})
            errQ = sess.run(q_loss, {z_in: zs, latent_cat_in: lcat, latent_cont_in: lcont})
            print ("Epoch: [%2d] [%4d%4d] d_loss: %.8f, g_loss: %.8f, q_loss: %.8f" % (epoch, idx, batch_idx, errD, errG, errQ ))
            
            #generate samples
            if counter % 100 == 0:
                samples, d1_loss, g1_loss, q1_loss = sess.run([G, d_loss, g_loss, q_loss],
                                                             {x: sample_inputs, z_in: sample_z_in, latent_cat_in: sample_lat, latent_cont_in: sample_cont})
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
    sample_directory = 'generate_images/InfoGAN/mnist/test'
    if not os.path.exists(sample_directory):
        os.makedirs(sample_directory)
    
    # input noise z each row is the same
    sample_z_in = []
    for i in range(10):
        a = np.random.uniform(-1, 1, [1, z_dim]).astype(np.float32)
        for j in range(10):
            sample_z_in.append(a)
    sample_z_in = np.reshape(np.array(sample_z_in), [100, z_dim])
    
    # input continue c each row is the same
    sample_cont = []
    for i in range(10):
        a = np.random.uniform(-1, 1, [1, number_continue]).astype(np.float32)
        for j in range(10):
            sample_cont.append(a)
    sample_cont = np.reshape(np.array(sample_cont), [100, number_continue])
    
    #input catagorial c each clo is the same
    sample_lat = []
    for i in range(10):
        a = np.array([e for e in range(10)])
        sample_lat.append(a)
    sample_lat = np.reshape(sample_lat, [100, 1])
    
    samples = sess.run(G, {z_in: sample_z_in, latent_cat_in: sample_lat, latent_cont_in: sample_cont})
    save_images(samples, [10, 10], './{}/{}.png'.format(sample_directory, 'cat'))
    print "save success"

