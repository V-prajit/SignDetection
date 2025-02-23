import numpy as np
from scipy.spatial.distance import euclidean
def standard_dtw(x, y, distance_func=euclidean):
    n, m = len(x), len(y)    
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = distance_func(x[i-1], y[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],     
                dtw_matrix[i, j-1],     
                dtw_matrix[i-1, j-1]    
            )
    return dtw_matrix[n, m]