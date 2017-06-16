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
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    delta = 1  # margin of the SVM
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    scores = W.dot(X)  # scores is of size (K, N)
    # print 'Size of Scores = ', scores.shape
    # Get scores for gt. Integer array indexing.
    # http://cs231n.github.io/python-numpy-tutorial/#numpy
    gt_scores = scores[y, range(y.shape[0])]  # row-vector size (1, N)
    # delta
    # print scores.shape
    delta_matrix = np.zeros(scores.shape)
    delta_matrix.fill(delta)
    # print delta_matrix.shape
    delta_matrix[y, range(scores.shape[1])] = 0
    # subtract gt_score of each data point from each element
    inner_term = scores - gt_scores + delta
    # print inner_term[0]
    # Indices of elements corresponding to: max(0, val)
    max_bool_idx = (inner_term > 0)
    # print max_bool_idx.shape
    max_inner_term = inner_term[max_bool_idx]
    # print max_inner_term.shape
    # Sum columns of max(0, inner_term): Summing over j classes(rows) and N data points (columns)
    loss = np.sum(max_inner_term)
    loss = loss/y.shape[0]
    # Regularization term
    regularizer = reg * np.sum(np.sum(W*W))
    loss = loss + regularizer
    print 'svm loss = ', loss
    # print 'Sum of all elements of W = ', np.sum(np.sum(W))
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
    # replace all non-zero (in our case these are positive) elements with 1
    # indicator_matrix tells us positions of elements > 0 in scores matrix
    # np.where(inner_term > 0, 2, -2) : np.where(inner_term > 0, 2, -2)
    indicator_matrix = np.where(inner_term > 0, 1, 0)  # (K, N)
    # print indicator_matrix
    print indicator_matrix.shape
    # Gradient wrt w_j, i.e., j!=y_i (GT)
    grad_wj = indicator_matrix.dot(np.transpose(X))
    print grad_wj.shape
    #
    grad_wyi = np.zeros(W.shape)  # Gradient wrt w_yi, i.e., j=y_i (GT)
    # Multiply each data point with one multiplier. Broadcast over all rows (dimensions of X)
    # sum(K, N) => (1, N) * (D, N)
    sum_j = np.sum(indicator_matrix, axis=0)
    # print sum_j.shape
    sum_matrix = np.tile(sum_j, (X.shape[0], 1))
    print sum_matrix.shape
    b = np.transpose(X * sum_matrix)
    print b.shape
    # repmat(a, m, n) = tile(a, (m, n))
    grad_wyi[y, :] += - b  # (D, N)
    print grad_wyi.shape
    dW = grad_wyi + grad_wj  # - regularization??
    print dW.shape
    # print 'gradient = ', dW
    # pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
