from f15ece6504.get_cifar10 import load_CIFAR10
# loading some necessary packages
import numpy as np
import random
import matplotlib.pyplot as plt
#matplotlib.pyplot.use('Agg')

# This is to make matplotlib figures appear inline in the
# notebook rather than in a new window.
#%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Some code so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2

# Assuming you have downloaded the CIFAR-10 database in your hw0/f15ece6504/f15ece6504/data folder, 
# we proceed to load the data into python
cifar10_dir = 'f15ece6504/data/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print 'Training data shape: ', X_train.shape
print 'Training labels shape: ', y_train.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
'''
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()
'''
# Subsample the data for more efficient code execution in this exercise.
num_training = 490
num_validation = 10
num_test = 10

# Our validation set will be num_validation points from the original
# training set.
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# Our training set will be the first num_training points from the original
# training set.
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# We use the first num_test points of the original test set as our
# test set.
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# As a sanity check, print out the shapes of the data
print 'Training data shape: ', X_train.shape
print 'Validation data shape: ', X_val.shape
print 'Test data shape: ', X_test.shape


mean_image = np.mean(X_train, axis=0)
print mean_image[:10] # print a few of the elements
#plt.figure(figsize=(4,4))
#plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) # visualize the mean image


X_train -= mean_image
X_val -= mean_image
X_test -= mean_image

# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
# Also, lets transform both data matrices so that each image is a column.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))]).T
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))]).T
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))]).T
# print X_train[0][-1]
print X_train.shape, X_val.shape, X_test.shape

from f15ece6504.classifiers.linear_svm_me1 import svm_loss_vectorized
from f15ece6504.classifiers.softmax import softmax_loss_vectorized
import time

# generate a random SVM weight matrix of small numbers
W = np.random.randn(10,3073) * 0.0001 

# Implement a vectorized code to implement the loss and the gradient of the SVM. 
# Remember that your code should not have any loops. 
# Hint: 
# Look up the broadcasting property of np arrays
# In case you have memory errors, you might want to use float16 or pickle your data
# check np.ndarray.dump()/load() if you consider pickling your data
                    
tic = time.time()
#_, grad_vectorized = svm_loss_vectorized(W, X_train, y_train, 0.00001)
_, grad_vectorized = softmax_loss_vectorized(W, X_train, y_train, 0.00001)
toc = time.time()
print 'Vectorized loss and gradient: computed in %fs' % (toc - tic)

# It should be quite fast and finish within a second if your code is optimized! 

'''
# Now implement SGD in LinearSVM.train() function and run it with the code below
from f15ece6504.classifiers import LinearSVM
svm = LinearSVM()
tic = time.time()
loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=5e4,
                      num_iters=1500, verbose=True)
toc = time.time()
print 'That took %fs' % (toc - tic)
'''

