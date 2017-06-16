
# coding: utf-8

# In[3]:

# Setup
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
#get_ipython().magic(u'matplotlib inline')

# Make sure that caffe is on the python path
caffe_root = '/home/azamosavi/Azamosavi/MachineLearning/caffe-master'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
'''
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
'''
model_prototxt = 'deploy.prototxt'
pretrained_model = os.path.join(caffe_root,'models/bvlc_alexnet/bvlc_alexnet.caffemodel')

caffe.set_mode_cpu()
net = caffe.Classifier(model_prototxt, pretrained_model,
                       mean=np.load(os.path.join(caffe_root,'python/caffe/imagenet/ilsvrc_2012_mean.npy')).mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))


# Sanity check just to see if everything is set up properly.

# In[4]:

image_file = 'cat.jpg'
input_image = caffe.io.load_image(image_file)
prediction = net.predict([input_image])
print("Predicted class is #{}.".format(prediction[0].argmax()))

# load labels
imagenet_labels_filename = os.path.join(caffe_root,'data/ilsvrc12/synset_words.txt')
try:
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
except:
    get_ipython().system(u'../data/ilsvrc12/get_ilsvrc_aux.sh')
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

# sort top k predictions from softmax output
top_k = prediction[0].flatten().argsort()[-1:-6:-1]
print labels[top_k]


# Here, we set the number of iterations, desired label, learning rate and the randomly generated image that we start with.

# In[10]:

n_iterations = 250
input_data = np.random.random((1,3,227,227))
label_index = 281 # cat. 99: goose, 285: cat, 543: dumbbell
label = np.zeros((1,1,1,1000))
label[0,0,0,label_index] = 1;
learning_rate = 10000


# Iteratively perform gradient ascent over input image space to generate visualization.

# In[ ]:

for i in range(n_iterations):
    net.forward(data=input_data)
    # 
    # END OF YOUR CODE
    
    # Perform backward pass for the desired class
    bw = net.backward(**{net.outputs[0]: label})
    
    # Perform gradient ascent over the input image
    # TODO
    diff = bw['data']
    input_data  += learning_rate * diff
    
    # END OF YOUR CODE
    
    if i%20 == 0:
        print("Iteration #{}.".format(i))

print 'Done'


# Normalize and view the class model visualization.

# In[ ]:

data = input_data[0].transpose(1,2,0)
data -= data.min()
data /= data.max()
#plt.imshow(data)
#scipy.misc.imsave(data,'cat_res.jpg')
plt.imsave('cat_res.jpg',data)

