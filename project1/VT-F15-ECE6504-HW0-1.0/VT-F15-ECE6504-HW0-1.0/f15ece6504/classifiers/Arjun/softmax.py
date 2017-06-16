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
    scores = np.exp(W.dot(X))  # scores is of size (K, N)
    # Get scores for gt. Integer array indexing.
    # http://cs231n.github.io/python-numpy-tutorial/#numpy
    gt_scores = scores[y, range(y.shape[0])]  # row-vector size (1, N)
    loss = -np.sum(np.log(gt_scores/np.sum(scores, axis=0)))/y.shape[0]
    print loss.shape
    print loss
    # Regularization term
    regularizer = reg * np.sum(np.sum(W*W))
    loss = loss + regularizer
    print loss
    #############################################################################
    # TODO
    # Gradient
    # indicator of whether the class is GT class or not
    ind_gt = np.zeros(scores.shape)
    ind_gt[y, range(scores.shape[1])] = 1
    print ind_gt.shape
    #
    p_j_x = scores/np.tile(np.sum(scores, axis=0), (scores.shape[0], 1))
    print p_j_x.shape
    # (K, N)*(N, D)
    dW = -np.sum((ind_gt - p_j_x).dot(np.transpose(X)))/y.shape[0]
    #
    regularizer = reg * W
    dW = dW + regularizer
    print dW.shape
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
