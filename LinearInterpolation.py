import numpy as np
from scipy.interpolate import interp1d

def InterpolateAndResample(data, target_size = 20):
    if data.size == 0:
        return np.array([])
        
    nan_data = np.isnan(data[:, 0])

    if np.all(nan_data):
        return np.full((target_size, data.shape[1]), np.nan)
    
    valid_indices = np.arange(len(data))[~nan_data]
    valid_data = data[~nan_data]

    if len(valid_indices) == 1:
        # Only one valid data point, repeat it
        return np.tile(valid_data, (target_size, 1))

    new_indices = np.linspace(valid_indices[0], valid_indices[-1], num=target_size)
    interp_function = [interp1d(valid_indices, valid_data[:, i], kind='linear', fill_value="extrapolate") for i in range (data.shape[1])]
    resampled_data = np.array([function(new_indices) for function in interp_function]).T

    return resampled_data

def calculate_unit_vector(input):
    result = np.full_like(input, np.nan)
    for t in range (1, (len(input)-1)):
        vector = input[t+1] - input[t-1]
        magnitude = np.sqrt(np.sum(vector**2))

        if magnitude != 0:
            unit_vector = vector/magnitude
            result[t] = unit_vector

    return result