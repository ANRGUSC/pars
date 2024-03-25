

import numpy as np
import copy

def normalize_data(matrix):
    # Compute the minimum and maximum of each column (user)
    min_values = np.min(matrix, axis=0)
    max_values = np.max(matrix, axis=0)
    
    # Compute normalization factors for each user
    normalization_factors = max_values - min_values
    
    # Normalize each user's time series
    normalized_matrix = (matrix - min_values) / normalization_factors
    
    return normalized_matrix, min_values, max_values, normalization_factors

def convert_matrix_to_timeseries(matrix):
    return matrix.reshape(-1,matrix.shape[-1])

def convert_timeseries_to_matrix(aggregated_time_series, grid_len, t_total):
    #X = np.zeros((grid_len, grid_len, t_total))
    X = np.zeros((grid_len, grid_len, t_total))


    for i in range(grid_len):
        for j in range(grid_len):
            if (i,j) in aggregated_time_series:
                X[i,j,:] =  aggregated_time_series[(i,j)]

    return X

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

def get_inout_sequences(train_data,tw):
    #train_inout_seq = [create_inout_sequences(train_data[:, i], tw) for i in range(train_data.shape[1])]
    train_inout_seq = [create_inout_sequences(train_data[i,:], tw) for i in range(train_data.shape[0])]

    #print(train_inout_seq)
    train_inout_seq = [item for sublist in train_inout_seq for item in sublist]
    #print(train_inout_seq)
    return train_inout_seq

def sanitize_matrix(x, pnrg, sensitivity = 1.0, epsilon = 1):
    sensitivity = sensitivity
    epsilon = epsilon
    b = sensitivity / epsilon
    x_noisy = x + pnrg.laplace(loc = 0.0, scale = b, size = x.shape)
    #print('x  {}    x_noisy  {}'.format(x,x_noisy))

    return x_noisy

def sanitize_timeseries(timeseries, pnrg, sensitivity = 1.0, epsilon = 1):
    timeseries_new = copy.deepcopy(timeseries)
    sensitivity = sensitivity
    epsilon = epsilon
    b = sensitivity / epsilon
    for key,val in timeseries.items():

        timeseries_new[key] = timeseries_new[key]  + pnrg.laplace(loc = 0.0, scale = b, size = val.shape)

    return timeseries_new






