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





# Compute Q_ij matrix with set variance?



# Perform gradient descent


