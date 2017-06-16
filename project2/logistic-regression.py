
# coding: utf-8

# ### Creating HDF5 data
# 
# Caffe takes inputs in HDF5, database(LevelDB/LMDB) and image(JPG) formats. In this demo, we show how to prepare HDF5 data for Caffe consumption! 
# 
# We follow the same steps that we followed in HW0 to load CIFAR-10 data into python.

# In[ ]:

# the usual startup! 
import numpy as np
from get_cifar10 import load_CIFAR10

cifar10_dir = '/path/to/hw1/1_cs231n/cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# Subsample the data for more efficient code execution in this exercise.
num_training = 49000
num_validation = 1000

# Our validation set will be num_validation points from the original
# training set.
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# Our training set will be the first num_train points from the original
# training set.
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape


# In[ ]:

# caffe needs your data to be in float32
X_train = np.array(X_train,dtype = 'float32')
y_train = np.array(y_train,dtype = 'float32')
X_val = np.array(X_val,dtype = 'float32')
y_val = np.array(y_val,dtype = 'float32')


# In[ ]:

# And, before we feed the data to the classifier, we need to subtract the mean! 
mean_image = np.mean(X_train,axis = 0)
print mean_image.shape

X_train -= mean_image
X_val -= mean_image

print X_train.mean()


# In[ ]:

# If you have read the Caffe tutorial, you will know that Caffe expects input
# in the form of 4-dimensional blobs
# N_images X N_channels X Height X Width
# So, let us reshape them! 

num_channels = 3
height = 32
width = 32

print X_val.shape

X_train = np.reshape(X_train, (num_training,num_channels,height,width))
X_val = np.reshape(X_val, (num_validation,num_channels,height,width))

# Check the data shape again! 
print 'Modified Train data shape: ', X_train.shape
print 'Modified Validation data shape: ', X_val.shape


# In[ ]:

# proceed to creating HDF5. We will create and store the data in folder `hdf5-data` folder 
import h5py
# with h5py.Fil('/path/to/whatever-name-you-want.h5','w') as f:
with h5py.File('hdf5-data/train.h5', 'w') as f:
    f['images'] = X_train # f['name'] - the name field is very important as that is what caffe will recognize
    f['labels'] = y_train
with h5py.File('hdf5-data/test.h5', 'w') as f:
    f['images'] = X_val
    f['labels'] = y_val
print 'HDF5 files are written. Need to make the txt files!'


# Now in the `hdf5-data` folder, you need to create two files `train.txt` and `test.txt` that have the path to the corresponding hdf5 file. As part of this example, these files are provided to you `hdf5-data` folfer. Open these files and make appropriate changes to the path. In case you changed the names of the hdf5 files, you need to change the `.txt` files also before proceeding to training.
# 
# Before starting the training process:
# - make approporiate changes in the `run_logreg.sh` and `logreg_solver.prototxt` files in the path fields. 
# - create a folder for the snapshots to be stored as mentioned against the `snapshot-prefix` field in the solver file
# 
# Run the `run_logreg.sh` file to start the training. 
# 
# Note that for this particular example, the details in the solver file are suited for the softmax classifier alone. For the other problems part of this assignment, a solver file that looks similar is provided. You need to make appropriate changes to obtain better results. You could also try new strategies for altering learning rate, new solver methods, etc. to improve your training process. 

# In[ ]:

# Assuming that you have completed training the classifer, let us plot the training loss vs. iteration. This is an
# example to show the usefullness of logging data. Setting debug_info = 1 in the solver file creates verbose logs
# that can be useful in debugging the network. 

# we neeed matplotlib to plot the graphs for us!
import matplotlib
# This is needed to save images 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# we need to put the log parsing code in the python path
import sys
sys.path.insert(0,'/path/to/caffe/tools/extra/')

from parse_log import parse_log

train_log, test_log = parse_log('/path/to/hw1/2_caffe/logreg/logreg.log')

# view the extracted data
# print train_log
#print test_log

# extract the required fields for plotting 
iters = []
loss = []
# test_interval should be the same as the number of iterations you display so that
# for every entry in iters you have a testing accuracy value. If different, make suitable
# modifications based on len(test_log)
accuracy = []
for i in range(len(train_log)):
    iters.append(train_log[i]['NumIters'])
    loss.append(train_log[i]['loss'])
    accuracy.append(test_log[i]['accuracy']*100)


# In[ ]:

# plot loss
# If your hyper-parameters are tuned, it should decrease exponentially with iterations. 
plt.close()
plt.plot(iters,loss)
# saves iters vs. loss 
plt.savefig('logreg/logreg_loss')


# In[ ]:

# plot accuracy
# if your hyper-parameters are tuned, it should hit a plateau at ~35% accuracy. 
plt.close()
plt.plot(iters,accuracy) 
# save figure
plt.savefig('logreg/logreg_accuracy')

