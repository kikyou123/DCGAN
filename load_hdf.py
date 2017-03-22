
# coding: utf-8

# In[8]:

import sys
sys.path.append('..')

import os
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream


def faces(ntrain = None, nval = None, ntest = None, batch_size = 64, data_dir = None):
    path = os.path.join(data_dir, 'face(202458,64,64,3).hdf5')
    tr_data = H5PYDataset(path, which_sets=('train',))
    #te_data = H5PYDataset(path, which_sets=('test',))

    if ntrain is None:
        ntrain = tr_data.num_examples
    #if ntest is None:
     #   ntest = te_data.num_examples
    #if nval is None:
      #  nval = te_data.num_examples

    tr_scheme = ShuffledScheme(examples=ntrain, batch_size=batch_size)
    tr_stream = DataStream(tr_data, iteration_scheme=tr_scheme)

    #te_scheme = SequentialScheme(examples=ntest, batch_size=batch_size)
    #te_stream = DataStream(te_data, iteration_scheme=te_scheme)

    #val_scheme = SequentialScheme(examples=nval, batch_size=batch_size)
    #val_stream = DataStream(tr_data, iteration_scheme=val_scheme)
    return tr_data, tr_stream


# In[ ]:



