# Standard libraries
import copy
import numpy as np
import pandas as pd
import pickle
import warnings

# PyTorch for deep learning
import torch
import torch.nn as nn

# Data visualization
import matplotlib.pyplot as plt

# Custom modules and functions
from functions.gen_neighborhoods import (
    gen_neighborhoods_normal_distribution, 
    gen_neighborhoods_uniform_distribution
)
from functions.preprocessing import (
    normalize_data, 
    convert_matrix_to_timeseries, 
    get_inout_sequences, 
    sanitize_matrix, 
    sanitize_timeseries, 
    convert_timeseries_to_matrix
)
from functions.inprocessing import (
    update_grid, 
    calculate_metrics, 
    train, 
    gen_test_consumption_matrices, 
    create_c_sanitized
)
from models.attention_gru import AttentionGRU

# Configuration and warnings
warnings.filterwarnings('ignore')

# Ensure repeatability
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




def gen_workload_random(X, prng2, num_tests):
    """
    Generate random workload for testing
    """
    d=len(X.shape)
    result = []
    for i in range(num_tests):
        mask = np.zeros(X.shape)
        #random_slice =  [np.sort(prng2.choice(np.arange(0, side_len), replace=False, size=(2))) for j in range(d)]
        random_slice =  [np.sort(prng2.choice(np.arange(0, X.shape[j]+1), replace=False, size=(2))) for j in range(d)]
        mask[tuple([slice(*i) for i in random_slice])]=1
        result.append(mask.flatten())
    result = np.array(result)
    return result


def gen_workload_fixed_shape(X, prng2, num_tests,query_size):
    """
    Generate fixed shaped size workload for testing
    """

    d=len(X.shape)
    side_len = X.shape[0]

    result = []
    for i in range(num_tests):
        mask = np.zeros(X.shape)
        
        random_slice =  [np.sort(prng2.choice(np.arange(0, side_len-query_size), replace=False, size=(2))) for j in range(d)]

        random_slice[0][1]= random_slice[0][0]+ query_size
        random_slice[1][1]= random_slice[1][0]+ query_size
        mask[tuple([slice(*i) for i in random_slice])]=1
        result.append(mask.flatten())

    result = np.array(result)
    #print(result)
    return result



def evaluate_performance(C_test_sanitized,true_cons):
    """
    Performance Evaluation.
    """
    name = 'STPT'
    newX = C_test_sanitized
    X = true_cons


    num_tests =200

    seed = 30
    prng2 = np.random.RandomState(seed)
    workload_random = gen_workload_random(X, prng2, num_tests)
    workload_one =    gen_workload_fixed_shape(X, prng2, num_tests,1)
    workload_five =   gen_workload_fixed_shape(X, prng2, num_tests,5)
    workload_ten=     gen_workload_fixed_shape(X, prng2, num_tests,10)

    x_hat_eval = np.dot(workload_random,newX.flatten())
    x_eval = np.dot(workload_random,X.flatten())
    relative_base = np.copy(x_eval)
    relative_base[relative_base == 0] = 20
    diff =  (x_eval-x_hat_eval) / relative_base
    avg_abs_err = np.linalg.norm(diff,1)/ float(diff.size)

    print(name + ' Random Query ')
    print('RE   {}'.format(avg_abs_err))
    #############
    x_hat_eval = np.dot(workload_one,newX.flatten())
    x_eval = np.dot(workload_one,X.flatten())
    relative_base = np.copy(x_eval)
    relative_base[relative_base == 0] = 20
    diff =  (x_eval-x_hat_eval) / relative_base
    avg_abs_err = np.linalg.norm(diff,1)/ float(diff.size)

    print(name + ' size one ')
    print('RE   {}'.format(avg_abs_err))
    #############
    x_hat_eval = np.dot(workload_five,newX.flatten())
    x_eval = np.dot(workload_five,X.flatten())
    relative_base = np.copy(x_eval)
    relative_base[relative_base == 0] = 20
    diff =  (x_eval-x_hat_eval) / relative_base
    avg_abs_err = np.linalg.norm(diff,1)/ float(diff.size)

    print(name + ' size five ')
    print('RE   {}'.format(avg_abs_err))
    #############


    x_hat_eval = np.dot(workload_ten,newX.flatten())
    x_eval = np.dot(workload_ten,X.flatten())
    relative_base = np.copy(x_eval)
    relative_base[relative_base == 0] = 20
    diff =  (x_eval-x_hat_eval) / relative_base
    avg_abs_err = np.linalg.norm(diff,1)/ float(diff.size)

    print(name + ' size ten ')
    print('RE   {}'.format(avg_abs_err))



"""
California Dataset. 
# Update the path as needed. 
"""
data_frame = pd.read_pickle('/home/users/sshaham/PhD_Projects/pars/DP_Publication_of_Electricity_Data/Dataset/CA_sep_dec')
data = pd.DataFrame(data_frame["visits_by_hour"].to_list()).T
data = data.dropna()
data = data.values
clipping_factor = 1.5



"""
Hyperparameters
"""
initial_grid_len =  32
flag_sanitization = True
input_dim     =  1
embed_size    =  128
hidden_dim    =  64
output_dim    =  1
learning_rate =  0.001
pnrg = np.random.RandomState(seed = 1)


"""
Generate Neighborhoods

#UNIFORM
oversampled_columns = np.random.choice(data.shape[1], initial_grid_len*initial_grid_len, replace=True)

#NORMAL
oversampled_columns = np.random.choice(data.shape[1], 300, replace=True)
"""
oversampled_columns = np.random.choice(data.shape[1], initial_grid_len*initial_grid_len, replace=True)
data = data[:, oversampled_columns]
n_users = data.shape[1]
t_total = data.shape[0]


"""
Critical Hyperparameters
"""
depth =      2
tw =         6
batch_size = 32
epochs =     5
t_train = 3*24 #7 days
t_test  = 120
t_val   = 138
eps_tot     = 30
eps_pattern = 10
eps_sanitize = eps_tot - eps_pattern

"""
Normalize Data
"""
normalized_data, min_values, max_values, factors = normalize_data(data)


"""
Generate Neighborhoods

#UNIFORM
cell_coords, user_map = gen_neighborhoods_uniform_distribution(n_users = n_users, grid_size = initial_grid_len)

#NORMAL
cell_coords, user_map = gen_neighborhoods_normal_distribution(pnrg = pnrg, n_users = n_users, grid_size = initial_grid_len)
"""

#For uniform make sure the number of cells and users are the same. 
cell_coords, user_map = gen_neighborhoods_normal_distribution(pnrg = pnrg, n_users = n_users, grid_size = initial_grid_len)

"""
Generate train, test and eval datasets
"""
normalized_data_train = normalized_data[:t_train,:]
normalized_data_test   = normalized_data[300:300+t_test,:]
normalized_data_val  = normalized_data[t_train+t_test:t_train+t_val+t_test, :]

"""
Let us just first store the original consumption matrix
"""

data_test   = data[300+tw : 300+t_test,:]

"""
Let us create the training dataset.
"""
level_len = normalized_data_train.shape[0]//(depth+1)
print('tree meax depth {}   level_len  {}'.format(depth,level_len))
all_training_time_series =[]

for i in range(depth+1):
    normalized_data_train_level = normalized_data_train[i*level_len:(i+1)*level_len]
    converted_grid_len = 2**i
    aggregated_time_series = update_grid(data = normalized_data_train_level, 
                                        initial_grid_len = initial_grid_len, 
                                        converted_grid_len = converted_grid_len, 
                                        cell_coords = cell_coords)
    """
    Noise addition goes here.
    """
    if flag_sanitization:
        mem = copy.deepcopy(aggregated_time_series)
        
        sensitivity = 1/(int(initial_grid_len/converted_grid_len)*int(initial_grid_len/converted_grid_len) )

        aggregated_time_series = sanitize_timeseries(timeseries = mem, 
                                                    pnrg = pnrg, 
                                                    sensitivity = sensitivity, 
                                                    epsilon = eps_pattern/t_train)
    else:
        pass

    """
    Add them to the list
    """
    mem =  list(aggregated_time_series.values())
    all_training_time_series.extend(mem)
    print('Tree depth {}, Num Neighborhoods {}, timeseries length {}'.format(i, len(mem), len(mem[0])))

all_training_time_series = np.vstack(all_training_time_series)
X_train_timeseries = get_inout_sequences(all_training_time_series,tw)

"""
Create Validation Dataset
"""
aggregated_time_series = update_grid( data = normalized_data_val, 
                                     initial_grid_len = initial_grid_len, 
                                     converted_grid_len = initial_grid_len, 
                                     cell_coords = cell_coords)

mem =  list(aggregated_time_series.values())
mem = np.array(mem)
X_val_timeseries = get_inout_sequences(mem, tw)


train_loader = torch.utils.data.DataLoader(X_train_timeseries, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(X_val_timeseries, batch_size=batch_size, shuffle=False)


"""
Model Training
"""
model = AttentionGRU(input_dim = input_dim, embed_size = embed_size, hidden_dim = hidden_dim, output_dim = output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = StepLR(optimizer, step_size=1, gamma=0.0001)  # Reduces the learning rate by a factor of 0.1 every 5 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

model = train(model, criterion, optimizer, scheduler, train_loader, valid_loader, epochs)




"""
Create Test Dataset
"""

aggregated_time_series = update_grid( data = normalized_data_test, 
                                     initial_grid_len = initial_grid_len, 
                                     converted_grid_len = initial_grid_len, 
                                     cell_coords = cell_coords)
mem =  list(aggregated_time_series.values())
mem = np.array(mem)
X_test_timeseries = get_inout_sequences(mem, tw)

"""
Model Test
"""

# Testing
X_test_timeseries = X_test_timeseries[:300]

test_inputs = [X_test_timeseries[i][0] for i in range(len(X_test_timeseries))]
test_targets = [X_test_timeseries[i][1] for i in range(len(X_test_timeseries))]
test_targets = [i[0] for i in test_targets]

test_predictions = []

with torch.no_grad():
    for seq in test_inputs:
        seq = torch.tensor(seq).float()  # Convert to tensor and ensure dtype is float32
        y_pred = model(seq.unsqueeze(0))
        test_predictions.append(y_pred.item())


original_scale_predictions = test_predictions #* (train_max - train_min) + train_min
original_scale_targets = test_targets #* (train_max - train_min) + train_min

#or
# Visualization
plt.figure(figsize=(15, 6))
plt.plot(original_scale_targets, label='True Data')
plt.plot(original_scale_predictions, label='Predictions')
plt.legend()
#plt.title('Test Data vs Predictions (Epsilon = {})'.format(epsilon))
plt.show()



# Example:
y_true = original_scale_targets
y_pred = original_scale_predictions

mape, mae, mse, rmse = calculate_metrics(y_true, y_pred)
print(f"MAPE: {mape:.5f}%")
print(f"MAE: {mae:.5f}")
print(f"MSE: {mse:.5f}")
print(f"RMSE: {rmse:.5f}")




C_test_cons, C_test_pattern = gen_test_consumption_matrices(model, 
                                                            initial_grid_len, 
                                                            aggregated_time_series, 
                                                            normalized_data_test, 
                                                            tw)


aggregated_time_series = update_grid( data = data_test, 
                                     initial_grid_len = initial_grid_len, 
                                     converted_grid_len = initial_grid_len, 
                                     cell_coords = cell_coords)
for key in aggregated_time_series:
    aggregated_time_series[key] = np.clip(aggregated_time_series[key], None, clipping_factor)

C_test_cons = convert_timeseries_to_matrix(aggregated_time_series = aggregated_time_series, 
                                            grid_len = initial_grid_len, 
                                            t_total = data_test.shape[0])


mem = copy.deepcopy(C_test_cons)

#np.save('/tank/users/sshaham/PhD_Projects/P3_DP_Electricity/stored_res/TX_STPT_C_test_cons_uniform.npy', C_test_cons)
#np.save('/tank/users/sshaham/PhD_Projects/P3_DP_Electricity/stored_res/TX_STPT_C_test_cons_normal.npy', C_test_cons)


C_test_sanitized = create_c_sanitized(C_test_cons = mem, 
                        C_test_pattern = C_test_pattern, 
                        quantization_levels = 150,
                        pnrg = pnrg,
                        epsilon_sanitize = eps_sanitize/ (clipping_factor))


evaluate_performance(C_test_sanitized,C_test_cons)



#np.save('/tank/users/sshaham/PhD_Projects/P3_DP_Electricity/stored_res/TX_STPT_C_test_sanitized_uniform.npy', C_test_sanitized)
#np.save('/tank/users/sshaham/PhD_Projects/P3_DP_Electricity/stored_res/TX_STPT_C_test_sanitized_normal.npy', C_test_sanitized)