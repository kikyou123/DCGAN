
# coding: utf-8

# In[2]:

from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange


# In[6]:

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


# In[4]:

def merge(images, size):
    """Args:
         images: imshow images of shape [batch_size, height, weight, 3]
         size: imshow image contain how much images
    
       Returns: 
          imshow images of size [h * size[0], w * size[1], 3]"""
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h : j * h + h, i * w : i * w + w, :] = image
    return img


# In[5]:

def save_images(images, size, image_path):
    """Args: images of shape[batch_size, height, weight, 3]
             size of shape [n_h, n_w]
             path: name of imsaved images
             """
    return imsave(images, size, image_path)


# In[ ]:
def transform(X, height = 64, weight = 64):
    assert X[0].shape == (height, weight, 3) or X[0].shape == (3, height, weight)
    if X[0].shape == (3, height, weight):
        X = X.transpose(0, 2, 3, 1)
    return np.array(X) / 127.5 - 1

def inverse_transform(images):
    return (images + 1.) / 2.0


def imsize(X):
    """Args:
         X is batch of images[batch_size, 64, 64, 3]
      Return:
         resize images([batch_size, 32, 32, 3)
         """
    batch_size = X.shape[0]
    sample_outs = np.zeros((batch_size, 32, 32, 3))
    for i in range(batch_size):
        sample_outs[i] = scipy.misc.imresize(X[i], (32, 32))
    return sample_outs
