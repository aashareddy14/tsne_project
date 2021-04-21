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


def calc_prob_matrix(distances, sigmas):
    """Convert a distances matrix to a matrix of probabilities."""
    two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))

    X = distances / two_sig_sq

    # Subtract max for numerical stability
    e_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))

    # We usually want diagonal probailities to be 0.
    np.fill_diagonal(e_x, 0.)

    # Add a tiny constant for stability of log we take later
    e_x += 1e-8  # numerical stability

    return e_x / e_x.sum(axis=1).reshape([-1, 1])


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


def perplexity(distances, sigmas):
    """calculate perplexity based on the P probability matrix."""
    prob_matrix = calc_prob_matrix(distances, sigmas)
    entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
    perplexity = 2 ** entropy

    return perplexity


def find_optimal_sigmas(distances, target_perplexity):
    """For each row of distances matrix, find sigma that results
    in target perplexity for that role."""
    sigmas = []
    # For each row of the matrix (each point in our dataset)
    for i in range(distances.shape[0]):
        # Make fn that returns perplexity of this row given sigma
        eval_fn = lambda sigma: \
            perplexity(distances[i:i + 1, :], np.array(sigma))
        # Binary search over sigmas to achieve target perplexity
        correct_sigma = binary_search(eval_fn, target_perplexity)
        # Append the resulting sigma to our output array
        sigmas.append(correct_sigma)
    return np.array(sigmas)


def q_tsne(Y):
    """t-SNE: Given low-dimensional representations Y, compute
    matrix of joint probabilities with entries q_ij."""
    distances = -squared_euc_dist(Y)
    inv_distances = np.power(1. - distances, -1)
    np.fill_diagonal(inv_distances, 0.)
    return inv_distances / np.sum(inv_distances), inv_distances


def p_joint(X, target_perplexity):
    """Given a data matrix X, gives joint probabilities matrix.

    # Arguments
        X: Input data matrix.
    # Returns:
        P: Matrix with entries p_ij = joint probabilities.
    """
    # Get the negative euclidian distances matrix for our data
    distances = -squared_euc_dist(X)
    # Find optimal sigma for each row of this distances matrix
    sigmas = find_optimal_sigmas(distances, target_perplexity)
    # Calculate the probabilities based on these optimal sigmas
    p_conditional = calc_prob_matrix(distances, sigmas)
    # Go from conditional to joint probabilities matrix
    P = (p_conditional + p_conditional.T) / (2. * p_conditional.shape[0])
    return P


def tsne_grad(P, Q, Y):
    """Estimate the gradient of t-SNE cost with respect to Y."""
    pq_diff = P - Q
    pq_expanded = np.expand_dims(pq_diff, 2)
    y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)

    # Get Q and distances (distances only used for t-SNE)
    distances = q_tsne(Y)[1]

    # Expand our inv_distances matrix so can multiply by y_diffs
    distances_expanded = np.expand_dims(distances, 2)

    # Multiply this by inverse distances matrix
    y_diffs_wt = y_diffs * distances_expanded

    # Multiply then sum over j's
    grad = 4. * (pq_expanded * y_diffs_wt).sum(1)
    return grad


def estimate_sne(X, num_iters=1000, q_fn=q_tsne, perplexity=30, learning_rate=10, momentum=0.9):
    """Estimates a SNE model.

    # Arguments
        X: Input data matrix.
        y: Class labels for that matrix.
        P: Matrix of joint probabilities.
        rng: np.random.RandomState().
        num_iters: Iterations to train for.
        q_fn: Function that takes Y and gives Q prob matrix.
        plot: How many times to plot during training.
    # Returns:
        Y: Matrix, low-dimensional representation of X.
    """

    # Initialise our 2D representation, numpy RandomState for reproducibility
    rng = np.random.RandomState(1)
    Y = rng.normal(0., 0.0001, [X.shape[0], 2])

    # Obtain matrix of joint probabilities p_ij
    P = p_joint(X, perplexity)

    # Initialise past values (used for momentum)
    if momentum:
        Y_m2 = Y.copy()
        Y_m1 = Y.copy()

    # Start gradient descent loop
    for i in range(num_iters):

        # Get Q and distances (distances only used for t-SNE)
        Q, distances = q_fn(Y)
        # Estimate gradients with respect to Y
        grads = tsne_grad(P, Q, Y)

        # Update Y
        Y = Y - learning_rate * grads
        if momentum:  # Add momentum
            Y += momentum * (Y_m1 - Y_m2)
            # Update previous Y's for momentum
            Y_m2 = Y_m1.copy()
            Y_m1 = Y.copy()

        # Plot sometimes
        # if plot and i % (num_iters / plot) == 0:
        #   categorical_scatter_2d(Y, y, alpha=1.0, ms=6,
        #                         show=True, figsize=(9, 6))

    return Y


yout = estimate_sne(X)

sns.scatterplot(yout[:,0], yout[:,1], hue = specs)