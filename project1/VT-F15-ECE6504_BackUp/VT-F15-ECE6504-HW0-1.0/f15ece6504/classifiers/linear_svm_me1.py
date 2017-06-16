import numpy as np
from random import shuffle

def svm_loss_vectorized(W, X, y, reg):
  #loss= eval_loss (X,y,W)
  loss = 0.0
  delta = 1 # margin of the SVM
  reg=0.00001 # regularization parameter
  scores = W.dot(X)
  yT=y[np.newaxis].T
  margins=np.maximum(0, scores - scores[yT[:,0]].diagonal() + delta)
  margins[y,range(scores.shape[1])]=0
  #loss=(np.sum(margins)-(y.shape[0]*delta))/y.shape[0]+landa*((W**2).sum())
  loss=(np.sum(margins)/y.shape[0])+reg*((W**2).sum())
  print(loss)
  W = np.random.rand(10,3073) * 0.001 # random weight vector
  #W = np.random.rand(W.shape[0], W.shape[1]) * 0.001 # random weight vector
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #dW=np.count_nonzero (margins )
  #dW=np.eye(scores.shape[0])*(scores - scores[yT[:,0]].diagonal() + delta)
  print(dW)
  #dW = eval_numerical_gradient(loss, W,X,y)
  return loss, dW
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
  """
  unvectorized version. Compute the multiclass svm loss for a single example (x,y)
  - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
    with an appended bias dimension in the 3073-rd position (i.e. bias trick)
  - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
  - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
  """
'''
  loss = 0.0
  #print(y.shape) #49000
  #print(X.shape[0]) #3073

  
  delta = 1 # margin of the SVM
  landa=0.001 # regularization parameter
  for i in xrange (X.shape[1]):
    loss +=L_i(X[:,i], y[i], W)
    
  loss=np.mean(loss)+landa*(W.sum())**2
  print (loss)
'''
  
    
  
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.

  
  #############################################################################
  #pass
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
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  

def eval_loss (X,y,W):
  loss = 0.0
  #print(y.shape) #49000
  #print(X.shape[0]) #3073

  
  delta = 1 # margin of the SVM
  landa=0.00001 # regularization parameter
  '''
  for i in xrange (X.shape[1]):
    loss +=L_i(X[:,i], y[i], W)
  '''
  scores = W.dot(X)
  yT=y[np.newaxis].T
  #print(y.shape)
  #P = scores[yT[:,0]]
  margins=np.maximum(0, scores - scores[yT[:,0]].diagonal() + delta)
  #margins = np.maximum(0, scores - scores[y] + delta)
  #print(margins.shape)
  #margins[margins[:,:] == delta] = 0
  #loss=np.mean(np.sum(margins))+landa*((W**2).sum())
  loss=(np.sum(margins)-(y.shape[0]*delta))/y.shape[0]+landa*((W**2).sum())
  print (loss)
  return loss

def eval_numerical_gradient(f, x, X, y):
  """ 
  a naive implementation of numerical gradient of f at x 
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """ 

  #fx = f(x) # evaluate function value at original point
  fx = f # evaluate function value at original point
  grad = np.zeros(x.shape)
  h = 0.00001

  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  #iteration = 15
  while not it.finished:
    # evaluate function at x+h
    ix = it.multi_index
    old_value = x[ix]
    x[ix] = old_value + h # increment by h
    fxh = eval_loss (X,y,x)  # evalute f(x + h)
    x[ix] = old_value # restore to previous value (very important!)

    # compute the partial derivative
    grad[ix] = (fxh - fx) / h # the slope
    it.iternext() # step to next dimension

  return grad
'''
def L_i(x, y, W):
  """
  unvectorized version. Compute the multiclass svm loss for a single example (x,y)
  - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
    with an appended bias dimension in the 3073-rd position (i.e. bias trick)
  - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
  - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
  """
  delta = 1.0 # see notes about delta later in this section
  scores = W.dot(x) # scores becomes of size 10 x 1, the scores for each class
  #print(scores.shape)
  #print(y)
  #print(scores[y])
  
  ###############################################################################
  #### semi-vectorized
  margins = np.maximum(0, scores - scores[y] + delta)
  # on y-th position scores[y] - scores[y] canceled and gave delta. We want
  # to ignore the y-th position and only consider margin on max wrong class
  margins[y] = 0
  loss_i = np.sum(margins)
  return loss_i
  ###############################################################################

  ##### Non-vectorized  

  correct_class_score = scores[y]
  D = W.shape[0] # number of classes, e.g. 10
  loss_i = 0.0
  for j in xrange(D): # iterate over all wrong classes
    if j == y:
      # skip for the true class to only loop over incorrect classes
      continue
    # accumulate loss for the i-th example
    loss_i += max(0, scores[j] - correct_class_score + delta)

  #######################################################################################
'''  

  
