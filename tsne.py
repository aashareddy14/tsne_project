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