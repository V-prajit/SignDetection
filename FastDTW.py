import numpy as np
from scipy.spatial.distance import euclidean


def DTW_Distance(x, y, radius=1):
    distance, _ = fast_dtw(x, y, radius=radius)
    return distance

def fast_dtw(x, y, radius=1, dist_func=euclidean):
    min_size = radius + 2
    
    if len(x) <= min_size or len(y) <= min_size:
        return dtw(x, y, dist_func)

    x_shrunk = _reduce_by_half(x)
    y_shrunk = _reduce_by_half(y)

    _, low_res_path = fast_dtw(x_shrunk, y_shrunk, radius, dist_func)
    
    projected_path = _expand_path(low_res_path, len(x), len(y))
    
    return _dtw_with_window(x, y, projected_path, radius, dist_func)

def _reduce_by_half(sequence):
    return np.array([
        (sequence[i] + sequence[i + 1]) / 2.0
        for i in range(0, len(sequence) - 1, 2)
    ])

def _expand_path(path, len_x, len_y):
    expanded = set()
    for i, j in path:
        i2, j2 = i * 2, j * 2

        for a in range(max(0, i2 - 1), min(len_x, i2 + 2)):
            for b in range(max(0, j2 - 1), min(len_y, j2 + 2)):
                expanded.add((a, b))
    
    return expanded

def _dtw_with_window(x, y, path, radius, dist_func):
    len_x, len_y = len(x), len(y)
    window = _expand_window(path, len_x, len_y, radius)
    
    cost = np.full((len_x, len_y), np.inf)

    if (0, 0) in window:
        cost[0, 0] = dist_func(x[0], y[0])

    for i in range(len_x):
        for j in range(len_y):
            if (i, j) in window:
                current_dist = dist_func(x[i], y[j])

                if i > 0 and j > 0:
                    min_cost = min(
                        cost[i-1, j], 
                        cost[i, j-1],   
                        cost[i-1, j-1]  
                    )
                    cost[i, j] = current_dist + min_cost
                elif i > 0:
                    cost[i, j] = current_dist + cost[i-1, j]
                elif j > 0:
                    cost[i, j] = current_dist + cost[i, j-1]

    path = _backtrack(cost, window)
    
    return cost[-1, -1], path

def _expand_window(path, len_x, len_y, radius):
    window = set()
    for i, j in path:
        for a in range(max(0, i - radius), min(len_x, i + radius + 1)):
            for b in range(max(0, j - radius), min(len_y, j + radius + 1)):
                window.add((a, b))
    return window

def _backtrack(cost_matrix, window):
    i, j = cost_matrix.shape[0] - 1, cost_matrix.shape[1] - 1
    path = [(i, j)]
    
    while i > 0 or j > 0:
        valid_moves = []
        if i > 0 and (i-1, j) in window:
            valid_moves.append((i-1, j))
        if j > 0 and (i, j-1) in window:
            valid_moves.append((i, j-1))
        if i > 0 and j > 0 and (i-1, j-1) in window:
            valid_moves.append((i-1, j-1))
        
        if not valid_moves:
            break
            
        i, j = min(valid_moves, key=lambda x: cost_matrix[x])
        path.append((i, j))
    
    return path[::-1]

def dtw(x, y, dist_func=euclidean):
    len_x, len_y = len(x), len(y)
    cost = np.full((len_x, len_y), np.inf)
    cost[0, 0] = dist_func(x[0], y[0])
    
    for i in range(1, len_x):
        cost[i, 0] = cost[i-1, 0] + dist_func(x[i], y[0])
    for j in range(1, len_y):
        cost[0, j] = cost[0, j-1] + dist_func(x[0], y[j])
    
    for i in range(1, len_x):
        for j in range(1, len_y):
            current_dist = dist_func(x[i], y[j])
            cost[i, j] = current_dist + min(
                cost[i-1, j],   
                cost[i, j-1],   
                cost[i-1, j-1]   
            )
    
    path = _backtrack(cost, {(i, j) for i in range(len_x) for j in range(len_y)})
    return cost[-1, -1], path