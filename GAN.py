
# coding: utf-8

# In[3]:

import tensorflow as tf


# In[4]:

import numpy as np


# In[5]:

import matplotlib.pyplot as plt
from scipy.stats import norm
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


# In[6]:

mu, sigma = 4, 0.5
xs = np.linspace(-8, 8, 1000)
plt.plot(xs, norm.pdf(xs, loc = mu, scale = sigma))
plt.savefig('fig0.png')
plt.close()


# In[7]:

batch_size = 15
h_dim = 5
learning_rate = 0.02


# In[8]:

def linear(input1, output_dim, scope = None, stddev = 1.0):
    norm = tf.random_normal_initializer(stddev = stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input1.get_shape()[1], output_dim], initializer=norm)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input1, w) + b

    
    


# In[9]:

def generator(input1, h_dim):
    h0 = tf.nn.softplus(linear(input1, h_dim, 'g0')) 
    x_g = linear(h0, 1, 'g1')
    return x_g


# In[10]:

def discriminator(input1, h_dim):
    h0 = tf.tanh(linear(input1, h_dim * 2, 'd0'))
    h1 = tf.tanh(linear(h0, h_dim * 2, 'd1'))
    h2 = tf.tanh(linear(h1, h_dim * 2, 'd2'))
    h3 = tf.sigmoid(linear(h2, 1, 'd3'))
    return h3


# In[11]:

def optimizer(loss, var_list, initial_learning_rate):
    decay = 0.95
    num_decay_step = 150
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(initial_learning_rate, batch, num_decay_step, decay, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = batch, var_list = var_list)
    return optimizer


# In[12]:

with tf.variable_scope('D_pre'):
    input_node = tf.placeholder(tf.float32, shape = [batch_size, 1]) 
    train_labels = tf.placeholder(tf.float32, shape = [batch_size, 1])
    D_pre = discriminator(input1 = input_node, h_dim = h_dim)
    pre_loss = tf.reduce_mean(tf.square(D_pre - train_labels))
    


# In[13]:

with tf.variable_scope('Gen') as scope:
    z = tf.placeholder(tf.float32, shape = (batch_size, 1))
    G = generator(z, h_dim)


# In[14]:

with tf.variable_scope('Disc') as scope:
    x = tf.placeholder(tf.float32, shape = (batch_size, 1))
    D1 = discriminator(x, h_dim)
    scope.reuse_variables()
    D2 = discriminator(G, h_dim)


# In[15]:

loss_d = tf.reduce_mean(-tf.log(D1) - tf.log(1 - D2))
loss_g = tf.reduce_mean(-tf.log(D2))

d_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'D_pre')
d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Disc' )
g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Gen')

opt_dpre = optimizer(pre_loss, d_pre_params, learning_rate)
opt_d = optimizer(loss_d, d_params, learning_rate)
opt_g = optimizer(loss_g, g_params, learning_rate)


# In[16]:

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)


# In[17]:

def plot_d0(D1, input_node):
    f, ax = plt.subplots(1)
    #p_data
    xs = np.linspace(-8, 8, 1000)
    ax.plot(xs, norm.pdf(xs, loc = mu, scale = sigma), label = 'p_data')
    #decision boundary
    r = 1000
    xs = np.linspace(-8, 8, r)
    ds = np.zeros((r, 1))
    
    for i in range(r / batch_size):
        x = np.reshape(xs[batch_size * i : batch_size * (i + 1)], (batch_size, 1))
        ds[batch_size * i : batch_size * (i + 1)] = sess.run(D1, {input_node: x})
    
    ax.plot(xs, ds, label = 'decision boundary')
    ax.set_ylim(0, 1.1)
    plt.legend()


# In[18]:

plot_d0(D_pre, input_node)
plt.title('Initial Decision Boundary')
plt.savefig('fig1.png')
plt.close()


# In[19]:

lh = np.zeros((1000, 1))
for i in range(1000):
    d = (np.random.random(batch_size) - 0.5) * 10.0
    labels = norm.pdf(d, loc = mu, scale = sigma)
    lh[i], _ = sess.run([pre_loss,opt_dpre], {input_node: np.reshape(d, (batch_size, 1)), train_labels: np.reshape(labels, (batch_size, 1))})
    
    


# In[20]:

plt.plot(lh)
plt.title('Traing Loss')
plt.savefig('fig2')
plt.close()


# In[21]:

plot_d0(D_pre, input_node)
plt.title('pretraning Decision Boundary')
plt.savefig('fig3.png')
plt.close()


# In[22]:

weightD = sess.run(d_pre_params)


# In[23]:

for i, v in enumerate(d_params):
    sess.run(v.assign(weightD[i]))
    


# In[24]:

def plot_fig():
    f, ax = plt.subplots(1)
    #p_data
    xs = np.linspace(-8, 8, 1000)
    ax.plot(xs, norm.pdf(xs, loc = mu, scale = sigma), label = 'p_data')
    #decision boundary
    r = 5000
    xs = np.linspace(-8, 8, r)
    ds = np.zeros((r, 1))
    
    for i in range(r / batch_size):
        x_1 = np.reshape(xs[batch_size * i : batch_size * (i + 1)], (batch_size, 1))
        ds[batch_size * i : batch_size * (i + 1)] = sess.run(D1, {x: x_1})
    
    ax.plot(xs, ds, label = 'decision boundary')
    
    zs = np.linspace(-8, 8, r)
    gs = np.zeros((r, 1))
    for i in range(r / batch_size):
        z_1 = np.reshape(zs[batch_size * i : batch_size * (i + 1)], (batch_size, 1))
        gs[batch_size * i : batch_size * (i + 1)] = sess.run(G, {z: z_1})
    histc, edges = np.histogram(gs, bins = 10)
    ax.plot(np.linspace(-8,8,10), histc/float(r), label='p_g')

    ax.set_ylim(0, 1.1)
    plt.legend()


# In[25]:

plot_fig()
plt.title('Before Training')
plt.savefig('fig4.png')
plt.close()


# In[26]:


TRAIN_ITERS = 1200
k = 1
histd = np.zeros((TRAIN_ITERS, 1))
histg = np.zeros((TRAIN_ITERS, 1))
for i in range(TRAIN_ITERS):
    for j in range(k):
        x_batch = np.random.normal(loc = mu, scale = sigma, size = batch_size)
        x_batch.sort()
        z_batch = np.linspace(-8, 8, batch_size) + np.random.random(batch_size) * 0.01
        histd[i], _ = sess.run([loss_d, opt_d], {x: np.reshape(x_batch, [batch_size, 1]), z: np.reshape(z_batch, [batch_size, 1])})
    z_batch = np.linspace(-8, 8, batch_size) + np.random.random(batch_size) * 0.01
    histg[i], _ = sess.run([loss_g, opt_g], {z: np.reshape(z_batch, [batch_size, 1])})


# In[27]:

plt.subplots(1)
plt.plot(range(TRAIN_ITERS), histd, label = 'obj_d')
plt.plot(range(TRAIN_ITERS), histg, label = 'obj_g')
plt.legend()
plt.savefig('fig5.png')
plt.close()


# In[28]:

plot_fig()
plt.savefig('fig6.png')


# In[ ]:




# In[ ]:




# In[ ]:



