import numpy as np
import pandas as pd
import math
import scipy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
import numba
from numba import jit, int32, int64, float32, float64
import pstats



@jit(nopython=True)
def p_ij(d_matrix, perplexity=30.0, tol=1e-6):
    """
    Finds P_ij matrix using binary search to find value of sigma_i

    Inputs: d_matrix- np.array of pairwise distance matrix, with a fixed perplexity

    Output: P-ij matrix
    """

    (n, d) = d_matrix.shape

    P = np.zeros((n, d), dtype=np.float64)
    prec_sum = 0.0

    # precision = 1/2sigma^2
    for i in range(n):
        prec_min = -np.inf
        prec_max = np.inf
        prec = 1.0

        # implement binary search for optimal sigmas
        for j in range(10):  # 10 binary search steps
            sum_p = 0.0
            for k in range(d):
                if k != i:
                    P[i, k] = np.exp(-d_matrix[i, k] * prec)
                    sum_p += P[i, k]

            sum_p_distribution = 0.0

            for k in range(d):
                P[i, k] /= sum_p
                sum_p_distribution += d_matrix[i, k] * P[i, k]

            # Calculate entropy, H matrix
            H = np.log(sum_p) + prec * sum_p_distribution
            H_diff = H - np.log(perplexity)

            # check if entropy is within tolerance
            if np.fabs(H_diff) <= tol:
                break

            if H_diff > 0.0:
                prec_min = prec
                if prec_max == np.inf:
                    prec *= 2.0
                else:
                    prec = (prec + prec_max) / 2.0
            else:
                prec_max = prec
                if prec_min == -np.inf:
                    prec /= 2.0
                else:
                    prec = (prec + prec_min) / 2.0

        prec_sum += prec

    return P


@jit(nopython=True)
def squared_euc_dist(X):
    """Calculate squared euclidean distance for all pairs in a data matrix X with d dimensions and n rows.
    Output is a pairwise distance matrix D that is nxn.
    """
    norms = np.power(X, 2).sum(axis=1)
    D = np.add(np.add(-2 * np.dot(X, X.T), norms).T, norms)

    return D


@jit(nopython=True)
def q_ij(Y):
    """Calculate joint probabilities over all points given Y, the low-dimensional map of data points. (pg. 2585)"""

    numerator = np.power(1. + (squared_euc_dist(Y)), -1)
    Q = numerator / np.sum(numerator)

    # q_i|i = 0
    np.fill_diagonal(Q, 0.)

    return Q


@jit(nopython=True)
def grad_C(P, Q, Y):
    """Estimate the gradient of t-SNE cost with respect to Y."""

    pq_diff = np.expand_dims((P - Q), 2)

    y_diff = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)

    y_dist = np.expand_dims(np.power(1 + squared_euc_dist(Y), -1), 2)

    grad = 4. * (pq_diff * y_diff * y_dist).sum(axis=1)

    return grad


@jit(nopython=True)
def tsne(X, num_iters=1000, perplexity=30, alpha=10, momentum=0.9):
    """Calculate Y, the optimal low-dimensional representation of data matrix X using optimized TSNE.

    Inputs:
        X: data matrix
        num_iters: number of iterations
        perplexity: target perplexity for calculating optimal sigmas for P probability matrix
        alpha: learning rate
        momentum: momentum to speed up gradient descent algorithm
    """

    # Initialize Y
    np.random.seed(0)
    Y = np.random.normal(0, 0.0001, size=(X.shape[0], 2))
    D = squared_euc_dist(X)
    P = p_ij(D)
    P = P + np.transpose(P)
    P = P / np.sum(P)

    # Initialise past y_t-1 and y_t-2 values (used for momentum)
    Y_tmin2 = Y
    Y_tmin1 = Y

    # gradient descent with momentum
    for i in range(num_iters):
        Q = q_ij(Y)
        grad = grad_C(P, Q, Y)

        # Update Y using momentum (pg. 2587)
        Y = (Y - (alpha * grad)) + (momentum * (Y_tmin1 - Y_tmin2))

        # update values of y_t-1 and y_t-2
        Y_tmin2 = Y_tmin1
        Y_tmin1 = Y

    return Y