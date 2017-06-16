import numpy as np
from random import shuffle

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  reg=0.00001
  W = np.random.rand(10,3073) * 0.001 # random weight vector
  scores = W.dot(X) 
  margins = -scores[y,range(scores.shape[1])] +np.log(np.exp(scores).sum())
  loss=(np.sum(margins)/y.shape[0])+reg*((W**2).sum())
  print(loss)
  #dW = np.zeros_like(W)
  dW = np.zeros(W.shape)
  dW=-( np.sum( scores[y,range(scores.shape[1])]-(np.exp(scores[y,range(scores.shape[1])]) /np.exp(scores).sum()) )/y.shape[0] )+reg*(W)
  print(dW)
  print(dW.shape) 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
