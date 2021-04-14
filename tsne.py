# load iris dataset, which will serve as test dataset for building the algorith
import numpy as np
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
    log_perp = np.log(perplexity)

    for i in range(n):
        for j in range(steps):
            sum_P = 0
            for k in range(d):
                if k != i:
                    P[i,k] = np.exp(-X[i, j] * beta)
                    sum_P += P[i,k]
                sum_dist_P = 0
                P[i,k] /= sum_P
                sum_dist_P += X[i,k] * P[i,k]

        # this is very not done, I am confused







# Compute Q_ij matrix with set variance?



# Perform gradient descent


