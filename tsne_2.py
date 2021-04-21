import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
import math
import scipy.linalg as la
from keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance

iris = sns.load_dataset('iris')

# matrix data
X = np.array(iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
specs = np.array(iris['species'])

def squared_euc_dist(X):
    """Calculate squared euclidean distance for all pairs in a data matrix X with d dimensions and n rows.
    Output is a pairwise distance matrix D that is nxn.
    """
    D = distance.squareform(distance.pdist(X, 'sqeuclidean'))
    return D


def p_cond(d_matrix, sigmas):
    """Convert a distances matrix to a matrix of conditional probabilities."""

    sig_2 = np.square(sigmas.reshape((-1, 1)))
    P_cond = np.exp((d_matrix / (2 * sig_2)) - np.max((d_matrix / (2 * sig_2)), axis=1).reshape([-1, 1]))

    # set p_i|i = 0
    np.fill_diagonal(P_cond, 0.)

    P_cond = (P_cond + 1e-10) / (P_cond + 1e-10).sum(axis=1).reshape([-1, 1])

    return P_cond


def binary_search(eval_fn, target, tol=1e-10, max_iter=10000,
                  lower=1e-20, upper=1000.):
    """Perform a binary search over input values to eval_fn.

    # Arguments
        eval_fn: Function that we are optimising over.
        target: Target value we want the function to output.
        tol: Float, once our guess is this close to target, stop.
        max_iter: Integer, maximum num. iterations to search for.
        lower: Float, lower bound of search range.
        upper: Float, upper bound of search range.
    # Returns:
        Float, best input value to function found during search.
    """
    for i in range(max_iter):
        mid = (lower + upper) / 2.
        val = eval_fn(mid)
        if val > target:
            upper = mid
        else:
            lower = mid
        if np.abs(val - target) <= tol:
            break
    return mid


def perp(d_matrix, sigmas):
    """calculate perplexity from distance matrix, sigmas, and conditional probability matrix."""
    P = p_cond(d_matrix, sigmas)
    entropy = -np.sum(P * np.log2(P), axis=1)
    perplexity = 2 ** entropy

    return perplexity


def find_optimal_sigmas(d_matrix, target_perplexity):
    """For each row of distances matrix, find sigma that results
    in target perplexity for that role."""
    sigmas = []
    # For each row of the matrix (each point in our dataset)
    for i in range(d_matrix.shape[0]):
        # Make fn that returns perplexity of this row given sigma
        eval_fn = lambda sigma: \
            perp(d_matrix[i:i + 1, :], np.array(sigma))
        # Binary search over sigmas to achieve target perplexity
        correct_sigma = binary_search(eval_fn, target_perplexity)
        # Append the resulting sigma to our output array
        sigmas.append(correct_sigma)
    return np.array(sigmas)


def q_ij(Y):
    """Calculate joint probabilities over all points given Y, the low-dimensional map of data points. (pg. 2585)"""

    numerator = np.power(1. + (squared_euc_dist(Y)), -1)
    Q = numerator / np.sum(numerator)

    # q_i|i = 0
    np.fill_diagonal(Q, 0.)

    return Q


def p_ij(X, target_perplexity):
    """Calculate joint probabilities in the high dimensional space given data matrix X
    and a target perplexity to find optimal sigmas (pg. 2584).
    """

    d_matrix = -squared_euc_dist(X)

    # optimal sigma for each row of distance matrix
    sigmas = find_optimal_sigmas(d_matrix, target_perplexity)

    # conditional p matrix from optimal sigmas
    p_conditional = p_cond(d_matrix, sigmas)

    # convert conditional P to joint P matrix (pg. 2584)
    n = p_conditional.shape[0]
    p_joint = (p_conditional + p_conditional.T) / (2. * n)

    return p_joint


def grad_C(P, Q, Y):
    """Calculate gradient of cost function (KL) with respect to lower dimensional map points Y (pg. 2586)"""

    pq_diff = (P - Q)[:, :, np.newaxis]

    y_diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]

    y_dist = (np.power(1. + (squared_euc_dist(Y)), -1))[:, :, np.newaxis]

    grad = 4. * (pq_diff * y_diff * y_dist).sum(axis=1)

    return grad


def tsne_opt(X, num_iters=1000, perplexity=30, alpha=10, momentum=0.9):
    """Calculate Y, the optimal low-dimensional representation of data matrix X using optimized TSNE.

    Inputs:
        X: data matrix
        num_iters: number of iterations
        perplexity: target perplexity for calculating optimal sigmas for P probability matrix
        alpha: learning rate
        momentum: momentum to speed up gradient descent algorithm
    """

    # Initialize Y
    Y = (np.random.RandomState(1)).normal(0., 0.0001, [X.shape[0], 2])

    P = p_ij(X, perplexity)

    # Initialise past y_t-1 and y_t-2 values (used for momentum)
    Y_tmin2 = Y
    Y_tmin1 = Y

    # gradient descent with momentum
    for i in range(num_iters):
        Q = q_ij(Y)
        grad = grad_C(P, Q, Y)

        # Update Y using momentum (pg. 2587)
        Y = (Y - alpha * grad) + (momentum * (Y_tmin1 - Y_tmin2))

        # update values of y_t-1 and y_t-2
        Y_tmin2 = Y_tmin1
        Y_tmin1 = Y

    return Y


yout = tsne_opt(X)

sns.scatterplot(yout[:,0], yout[:,1], hue = specs)