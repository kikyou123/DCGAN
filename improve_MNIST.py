import os

import tensorflow as tf
import numpy as np
from op import *
from utils import *
from utilis_model import  *
from tensorflow.examples.tutorials.mnist import input_data

seed = 45
np.random.seed(seed)
tf.set_random_seed(seed)
rng = np.random.RandomState(seed)

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

beta1 = 0.5
max_epoch = 300
z_dim = 100
batch_size = 100
unlabeled_weight = 1
count = 10
istrain = True

model_dir = "{}_{}_{}_{}".format('mnist', batch_size, 28, 28)

checkpoint_dir = 'checkpoint/improve_GAN/mnist'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

sample_dir = 'generate_images/improve_GAN/mnist'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

log_dir = 'log/improve_GAN/mnist'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

sample_num = 100
sample_z = np.random.uniform(-1, 1, [sample_num, z_dim])
sample_inputs, sample_labels = mnist.train.next_batch(sample_num)

def generator(z):
    """
    :param z: input noise of size[batch_size, dim_Z]
    :return: generator images[batch_size, 28*28]
    """
    with tf.variable_scope("Gen") as scope:
        h0 = tf.nn.softplus(batch_norm(linear(z, 500, 'g_h0_lin'), name = 'g_bn0'))
        h1 = tf.nn.softplus(batch_norm(linear(h0, 500, 'g_h1_lin'), name = 'g_bn1'))
        h2 = tf.nn.sigmoid(linear(h1, 28 * 28, 'g_h2_lin'))
        return h2

def disciminator(x, istrain = istrain, reuse = False):
    """

    :param x: x is input image of size [batch_size, 28 * 28]
    :return: y: [batch_size, 10] yk is the pro x is beglong to the cat k
             h4: the hidden feature map [batch_size, 250]

    """
    with tf.variable_scope("Disc") as scope:
        if reuse:
            scope.reuse_variables()
        x_ = gaussian_noise_layer(x, std = 0.3, istrain = istrain)
        h0 = relu(linear(x_, 1000, 'd_h0_lin'))
        h0_ = gaussian_noise_layer(h0, 0.5, istrain)
        h1 = relu(linear(h0_, 500, 'd_h1_lin'))
        h1_ = gaussian_noise_layer(h1, 0.5, istrain)
        h2 = relu(linear(h1_, 250, 'd_h2_lin'))
        h2_ = gaussian_noise_layer(h2, 0.5, istrain)
        h3 = relu(linear(h2_, 250, 'd_h3_lin'))
        h3_ = gaussian_noise_layer(h3, 0.5, istrain)
        h4 = relu(linear(h3_, 250, 'd_h4_lin'))
        h4_ = gaussian_noise_layer(h4, 0.5, istrain)
        do = linear(h4_, 10, 'd_out')
        return do, h4

#placeholder
z_in = tf.placeholder(tf.float32, shape = [None, z_dim])
x_label = tf.placeholder(tf.float32, shape = [None, 28 * 28])
x_unlabel = tf.placeholder(tf.float32, shape = [None, 28 * 28])
y = tf.placeholder(tf.float32, shape = [None, 10])
z_sum = tf.summary.histogram("z", z_in)

#generate image
G = generator(z_in)
G_sum = tf.summary.histogram("G", G)

#discritor output
output_before_softmax_label, _ = disciminator(x_label, istrain)
output_before_softmax_unlabel, mom_real = disciminator(x_unlabel, istrain, reuse = True)
output_before_softmax_fake, mom_gen = disciminator(G, istrain, reuse = True)

#supervised loss
loss_label = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = output_before_softmax_label))
loss_label_sum = tf.summary.scalar("lossLabel", loss_label)
#unsupervised loss
z_exp_unlabel = tf.reduce_logsumexp(output_before_softmax_unlabel)
z_exp_fake = tf.reduce_logsumexp(output_before_softmax_fake)
loss_unlabel = tf.reduce_mean(-0.5 * z_exp_unlabel + 0.5 * tf.nn.softplus(z_exp_unlabel) + 0.5 * tf.nn.softplus(z_exp_fake))
loss_unlabel_sum = tf.summary.scalar("lossUnlabel", loss_unlabel)
# feature matching loss(generator loss)
loss_gen = tf.reduce_mean(tf.square(mom_gen - mom_real))
loss_gen_sum = tf.summary.scalar("lossGen", loss_gen)

#train error
train_err = tf.reduce_mean(tf.cast(tf.not_equal(tf.arg_max(output_before_softmax_label, 1), tf.arg_max(y, 1)), "float"))


#test error
output_before_softmax, _ = disciminator(x_label, istrain = True, reuse = True)
test_err = tf.reduce_mean(tf.cast(tf.not_equal(tf.arg_max(output_before_softmax, 1), tf.arg_max(y, 1)), "float"))

#tensorflow function for training and testing
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

d_loss = loss_label + unlabeled_weight * loss_unlabel
g_loss = loss_gen

d_optim = tf.train.AdamOptimizer(0.003, beta1 = beta1).minimize(d_loss, var_list = d_vars)
g_optim = tf.train.AdamOptimizer(0.003, beta1 = beta1).minimize(g_loss, var_list = g_vars)

saver = tf.train.Saver()
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)

g_sum = tf.summary.merge([z_sum, G_sum, loss_gen_sum])
d_sum = tf.summary.merge([z_sum, loss_label_sum, loss_unlabel_sum])
writer = tf.summary.FileWriter(log_dir, sess.graph)

#produce input data
train_data, train_label = mnist.train.images, mnist.train.labels
val_data, val_label = mnist.validation.images, mnist.validation.labels
test_data, test_label = mnist.test.images, mnist.test.labels

trainx = np.concatenate([train_data, val_data], axis = 0).astype(np.float32)
trainx_unl = trainx.copy() # for dis loss
trainx_unl2 = trainx.copy() # for gen loss
trainy = np.concatenate([train_label, val_label], axis = 0).astype(np.float32)
nr_batches_train = trainx.shape[0] / batch_size
testx = test_data.astype(np.float32)
testy = test_label.astype(np.float32)
nr_batches_test = test_data.shape[0] / batch_size

#select labeled data[trainx.shape[0], 28 * 28](100)
txs = []
tys = []
for j in range(10):
    txs.append(trainx[np.argmax(trainy, 1) == j][: count])
    tys.append(trainy[np.argmax(trainy, 1) == j][: count])
txs = np.concatenate(txs, axis = 0)
tys = np.concatenate(tys, axis = 0)

counter = 1
#begin train
if istrain:
    for epoch in range(max_epoch):
        #consruct randomly permuted minibatches
        trainx = []
        trainy = []
        for t in range(trainx_unl.shape[0] / txs.shape[0]):
            inds = rng.permutation(txs.shape[0])
            trainx.append(txs[inds])
            trainy.append(tys[inds])
        trainx = np.concatenate(trainx, 0)
        trainy = np.concatenate(trainy, 0)
        trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]
        trainx_unl2 = trainx_unl2[rng.permutation(trainx_unl2.shape[0])]

        #train
        label_loss = 0.
        unlabel_loss = 0.
        err_train = 0.
        gen_loss = 0.
        for idx in range(nr_batches_train):
            batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
            #update D
            _, summary_str, ll, lu, te  = sess.run([d_optim, d_sum, loss_label, loss_unlabel, train_err],
                                      {z_in: batch_z,
                                       x_label: trainx[idx * batch_size: (idx + 1) * batch_size],
                                       y: trainy[idx * batch_size: (idx + 1) * batch_size],
                                       x_unlabel: trainx_unl[idx * batch_size: (idx + 1) * batch_size]
                                       })
            label_loss += ll
            unlabel_loss += lu
            err_train += te
            writer.add_summary(summary_str, counter)
            #update G
            _, summary_str, lg = sess.run([g_optim, g_sum, loss_gen],
                                      {z_in: batch_z,
                                       x_unlabel: trainx_unl2[idx * batch_size: (idx + 1) * batch_size]})
            writer.add_summary(summary_str, counter)
            gen_loss += lg

            counter += 1
             # save parameters
            if counter % 500 == 0:
                save(checkpoint_dir, model_dir = model_dir, step = counter, saver = saver, sess = sess)

        label_loss /= nr_batches_train
        unlabel_loss /= nr_batches_train
        err_train /= nr_batches_train
        gen_loss /= nr_batches_train
        print ("Epoch: [%2d] label_loss: %.8f, unlabel_loss: %.8f, train_err: %.8f, gen_loss: %.8f" % (epoch, label_loss,               unlabel_loss, err_train, gen_loss))

        # generate samples
        samples = sess.run([G], {z_in: sample_z})
        samples = np.reshape(samples, [-1, 28, 28, 1])
        save_images(samples, [10, 10], './{}/train_{:02d}.png'.format(sample_dir, epoch))
        print("[Sample]")

else:
    if load(checkpoint_dir, model_dir = model_dir, saver = saver, sess = sess):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")
    
    # test accuravy
    err_test = 0
    for idx in range(nr_batches_test):
        te = sess.run(test_err, {x_label: testx[idx * batch_size: (idx + 1) * batch_size], y: testy[idx * batch_size: (idx + 1) * batch_size]})
        err_test += te
    print err_test
    
sess.close()
    
    


