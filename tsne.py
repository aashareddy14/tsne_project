# load libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
import math

# load iris dataset, which will serve as test dataset for building the algorithm
iris = datasets.load_iris()
X = iris.data

# pairwise distance function and matrix D
def squared_euc_dist(x, y, axis = -1):
    """squared euclidean distance between vectors x and y"""
    diff = x - y
    return (diff**2).sum(axis)

def row_loop_dist(M, distance_func):
    """finds distance matrix using row vectors of M"""
    dist = np.array([[distance_func(M[x, :], M[y, :]) for x in range(M.shape[0])] for y in range(M.shape[0])])
    return dist

D = row_loop_dist(X, squared_euc_dist)

# Compute P_ij matrix (need to perform binary search to find value of sigma_i)
def p_matrix(X, perplexity = 30.0, tol = 1e-5):
    """
    Finds P_ij matrix using binary search to find value of sigma_i

    Inputs: X- np.array of pairwise distance matrix, fixed perplexity

    Output: P-ij matrix
    """
    steps = 10 # maximum number of binary search steps

    (n, d) = X.shape

    P = np.zeros((n, d), dtype=np.float64)
    beta = np.ones((n, 1))
    beta_sum = 0.0
    log_perp = np.log(perplexity)

    for i in range(n):
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0

        # Binary search of precision for i-th conditional distribution
        for j in range(steps):
            sum_Pi = 0.0
            for k in range(d):
                if k != i:
                    P[i, k] = math.exp(-X[i, k] * beta)
                    sum_Pi += P[i, k]

            sum_disti_Pi = 0.0

            for k in range(d):
                P[i, k] /= sum_Pi
                sum_disti_Pi += X[i, k] * P[i, k]

            entropy = np.log(sum_Pi) + beta * sum_disti_Pi
            entropy_diff = entropy - log_perp

            if math.fabs(entropy_diff) <= tol:
                break

            if entropy_diff > 0.0:
                beta_min = beta
                if beta_max == np.inf:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if beta_min == -np.inf:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0

        beta_sum += beta

    return P


# function to calculate Q_ij from Y in the new map

def q_matrix(Y):
    """
    Finds Q_ij matrix

    Inputs: Y- np.array of points in the new space

    Output: Q-ij matrix
    """

    D = row_loop_dist(Y, squared_euc_dist)
    (n, d) = Y.shape
    Q = np.zeros((n, n))
    sum_Qi = 0.0
    for i in range(n):
        for k in range(d):
            if k != i:
                Q[i, k] = math.exp(-D[i, k])
                sum_Qi += Q[i, k]

        sum_disti_Qi = 0.0

        for k in range(d):
            Q[i, k] /= sum_Qi
            sum_disti_Qi += D[i, k] * Q[i, k]

    return Q

# Test Q matrix on X (even though input is normally Y)
Q_ij = q_matrix(X)
Q_ij


# KL divergence function that we will minimize through Gradient Descent
def KL(P, Q):
    """
    Finds KL divergence between P and Q matrices

    Inputs:
        P - similarity matrix of points in the high-dimensional space
        Q - similarity matrix of points in the low-dimensional map space

    Output: KL divergence expression
    """

    kl = 0
    for i in range(P.shape[0]):
        for j in range(Q.shape[0]):
            if Q[i, j] != 0 and P[i, j] != 0:
                kl += P[i, j] * np.log(P[i, j] / Q[i, j])
    return kl

# test
KL(P_ij, Q_ij)

# Function to find gradient of KL divergence
def grad_KL(P, Y):
    """
    Finds the gradient of the KL divergence between P and Q matrices expression

    Inputs:
        P - similarity matrix of points in the high-dimensional space
        Q - similarity matrix of points in the low-dimensional map space

    Output: Gradient of KL divergence expression
    """

    Q = q_matrix(Y)

    for i in range(Y.shape[0]):
        sum_j = 0
        for j in range(Y.shape[0]):
            sum_j += ((Y[i] - Y[j]) * (P[i, j] - Q[i, j]) * (1 + np.linalg.norm(Y[i] - Y[j] ** 2)) ** -1)

    return 4 * sum_j

# test
grad_KL(P_ij, X)

# Perform gradient descent


