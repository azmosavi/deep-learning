import numpy as np
try:
  from cs231n.im2col_cython import col2im_cython, im2col_cython
except ImportError:
  print 'run the following from the cs231n directory and try again:'
  print 'python setup.py build_ext --inplace'
  print 'You may also need to restart your iPython kernel'

from cs231n.im2col import *


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  #print "xshape :", x.shape
  #print "wshape :", w.shape
  #print "bshpae :", b.shape 
  
  xnew=x.reshape(x.shape[0],w.shape[0])
  out=xnew.dot(w)+b
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  
  doutw = dout.dot(w.T)
  dx = doutw.reshape(x.shape)
  xNew = x.reshape(x.shape[0],w.shape[0])
  dw = xNew.T.dot(dout)
  db = np.sum(dout, axis=0)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out=np.maximum(0,x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  #pass
  dx_tmp=np.reshape([1 if i>0 else 0 for i in (x).reshape(-1)],x.shape)
  dx=dx_tmp*dout
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  #pass
  N, C, H, W = x.shape
  num_filters, _, filter_height, filter_width = w.shape
  stride, pad = conv_param['stride'], conv_param['pad']
  x_padded = np.pad(x,((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
  
  HPrime = (H + 2 * pad - filter_height) / stride + 1
  WPrime = (W + 2 * pad - filter_width) / stride + 1
  out = np.zeros((N, num_filters, HPrime, WPrime))

  for i in range (N):
    xNew = x_padded[i,:,:,:]
    for f in range (num_filters):
      wf=w[f,:,:,:]
      bf=b[f]
      for H in range (HPrime):
        for W in range (WPrime):
          xcols=xNew[:,stride * H: stride * H + filter_height, stride * W : stride * W + filter_width]
          xcol = np.reshape(xcols, -1)
          ws = np.reshape(wf, -1)
          out[i, f, H, W] = xcol.dot(ws) + bf 
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  #pass
  x, w, b, conv_param= cache
  stride, pad = conv_param['stride'], conv_param['pad']
  N, C, H, W = x.shape
  num_filters, _, filter_height, filter_width = w.shape
  HPrime = (H + 2 * pad - filter_height) / stride + 1
  WPrime = (W + 2 * pad - filter_width) / stride + 1

  db = np.sum(dout, axis=(0, 2, 3))

  num_filters, _, filter_height, filter_width = w.shape

  dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
  x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)
  dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)
  

  dx = np.zeros([N, C, H + 2 * pad, W + 2 * pad])
  dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)
  # dx = col2im_indices(dx_cols, x.shape, filter_height, filter_width, pad, stride)
  dx = col2im_cython(dx_cols, x.shape[0], x.shape[1], x.shape[2], x.shape[3], filter_height, filter_width, pad, stride)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  #pass
  N, C, H, W = x.shape
  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']
  HPrime = (H - pool_height) / stride + 1
  WPrime = (W + - pool_width) / stride + 1
  
  out = np.zeros((N, C, HPrime, WPrime))
  for i in range (N):
    for c in range (C):
      xNew=x [ i,c, :,: ]
      for k in range (HPrime):
        for l in range (WPrime):
          xcols=xNew[stride*k : stride*k  + pool_height, stride*l  : stride *l+ pool_width]
          #print 'xcols',xcols.shape
          res = np.reshape(xcols,-1)
          #print 'res',res
          out[i, c, k, l] = res.max(axis=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  #dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  #pass
  x, pool_param = cache
  N, C, H, W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  HPrime = dout.shape[2]
  WPrime = dout.shape[3]
  
  dx = np.zeros((N, C, H, W))
    
  for i in range(N):
    for j in range(C):
      xNew1 = x[i,j,:,:]
      for k in range(HPrime):
        for l in range(WPrime):
          xNew = xNew1[stride * k: stride * k + pool_height, stride * l : stride * l + pool_width]
          rMax=np.max(xNew, axis=1)
          rIndex = np.argmax(rMax)
          cMax=np.max(xNew, axis=0)
          cIndex = np.argmax(cMax)
          dx[i, j, (stride * k) + rIndex, (stride * l) + cIndex] = dout[i, j, k, l]
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

