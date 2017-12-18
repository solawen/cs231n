import numpy as np
from random import shuffle
#from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  C = W.shape[1]
  score = X.dot(W)
  for i,s in enumerate(score, start=0):
    sum = 0
    s -= np.max(s)
    loss += np.log(np.sum(np.exp(s)))
    loss -= s[y[i]]
    dW[:,y[i]] -= X[i]
    for j in xrange(C):
        dW[:,j] += np.exp(s[j])/np.exp(s).sum() * X[i]
  loss = loss / N + reg *np.sum(W*W)
  dW = dW/N + 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
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
  pass
  N = X.shape[0]
  score = X.dot(W)
  score -= score.max(axis = 1).reshape(N,1)
  loss += np.sum(np.log(np.exp(score)))
  loss -= np.sum(score[range(N), y])
  loss = loss / N +reg*np.sum(W*W)
  dW[:,y] -= X.T
  sum = np.sum(np.exp(score),axis = 1)
  dW += (X.T).dot(np.exp(score)/sum.reshape(N,1))
  dW = dW/N + 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

