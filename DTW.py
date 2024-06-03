import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def DTW_Distance(seq1, seq2):
    distance, path = fastdtw(seq1, seq2, dist=euclidean)
    return distance