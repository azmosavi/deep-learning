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
  reg=0.00001
  #loss= eval_loss (X,y,W)
  scores = W.dot(X) 
  margins = -scores[y,range(scores.shape[1])] +np.log(np.exp(scores).sum())
  loss=(np.sum(margins)/y.shape[0])+reg*((W**2).sum())
  print(loss)
  #dW = np.zeros_like(W)
  dW = np.zeros(W.shape)
  dW=-( np.sum( scores[y,range(scores.shape[1])]-(np.exp(scores[y,range(scores.shape[1])]) /np.exp(scores).sum()) )/y.shape[0] )+reg*(W)
  print(dW)
  print(dW.shape)                                                                                                                                    
                                                                                                                                      
  return loss, dW                                                                                                                                      

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.
  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  

def eval_loss (X,y,W):
  loss = 0.0
  #print(y.shape) #49000
  #print(X.shape[0]) #3073

  
  delta = 1 # margin of the SVM
  landa=0.001 # regularization parameter

  for i in xrange (X.shape[1]):
    loss +=L_i(X[:,i], y[i], W)
    
  margins =  scores[y] -np.log(np.exp(scores).sum())
  loss=np.mean(loss)+landa*(W.sum())**2
  print (loss)
  return loss

def L_i(x, y, W):
  
  delta = 1.0 # see notes about delta later in this section
  scores = W.dot(x) # scores becomes of size 10 x 1, the scores for each class
  ###############################################################################
  #### semi-vectorized
  margins =  scores[y] -np.log(np.exp(scores).sum())
  # on y-th position scores[y] - scores[y] canceled and gave delta. We want
  # to ignore the y-th position and only consider margin on max wrong class
  margins[y] = 0
  loss_i = np.sum(margins)
  return loss_i
