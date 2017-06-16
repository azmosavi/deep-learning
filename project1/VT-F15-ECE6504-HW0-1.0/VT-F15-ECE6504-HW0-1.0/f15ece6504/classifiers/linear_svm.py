import numpy as np
from random import shuffle

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.
  Inputs:
  - W: K x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  loss = 0.0
  W = np.random.rand(10,3073) * 0.001 # random weight vector
  dW = np.zeros(W.shape) # initialize the gradient as zero
  delta = 1 # margin of the SVM
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  reg=0.00001 # regularization parameter
  scores = W.dot(X) # Score matrix
  yT=y[np.newaxis].T 
  margins=np.maximum(0, scores - scores[yT[:,0]].diagonal() + delta) #deducting scores of each class from score matrix in vectorized form
  margins_temp=margins
  margins_temp[y,range(scores.shape[1])]=0 ## fill the position of correct class with 0
  #loss=(np.sum(margins)-(y.shape[0]*delta))/y.shape[0]+landa*((W**2).sum())
  loss=(np.sum(margins)- (yT.shape[0]*delta)/y.shape[0])+reg*((W**2).sum()) # average the margin and add the regularization to it
  print ('loss of svm is')
  print(loss)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  W = np.random.rand(10,3073) * 0.001 # random weight vector
  #W = np.random.rand(W.shape[0], W.shape[1]) * 0.001 # random weight vector
  #dW=np.count_nonzero (margins )
  #dW=np.eye(scores.shape[0])*(scores - scores[yT[:,0]].diagonal() + delta)
  margins_temp[margins_temp > 0] = 1
  dW_temp = np.sum(margins_temp, axis=0)
  margins_temp[y, range(X.shape[1])] = -dW_temp
  dW = margins_temp.dot(X.T)/X.shape[1] + reg * W
  print ('Gradient of svm is')
  print(dW)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
'''
  XT=np.transpose(X)
  ## Gradient should be taken w.r.t two terms (w_j and w_{yi}
  inm=np.where((scores - scores[y, range(y.shape[0])]+ delta) > 0, 1, 0) #it gives us positions of elements > 0 in scores matrix
  dW_j=inm.dot(XT)
  dW_i=np.zeros(W.shape)
  dW_itemp=np.transpose(X * np.tile(np.sum(inm, axis=0), (X.shape[0], 1)))
  dW_i[y, :] += - dW_itemp
  dW = dW_j + dW_i
 ''' 
