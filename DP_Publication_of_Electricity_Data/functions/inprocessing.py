

from sklearn.metrics import mean_absolute_error, mean_squared_error
from functions.preprocessing import normalize_data, convert_matrix_to_timeseries, get_inout_sequences, sanitize_matrix, sanitize_timeseries, convert_timeseries_to_matrix
import numpy as np
from collections import defaultdict
import copy
import torch


def calculate_metrics(y_true, y_pred):
    """
    Calculate MAPE, MAE, MSE, and RMSE.
    Args:
        y_true (list or np.array): True values.
        y_pred (list or np.array): Predicted values.
    Returns:
        mape (float): Mean Absolute Percentage Error.
        mae (float): Mean Absolute Error.
        mse (float): Mean Squared Error.
        rmse (float): Root Mean Squared Error.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Avoid division by zero by replacing zeros in y_true with a small value
    y_true = np.where(y_true == 0, 1e-10, y_true)
    
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return mape, mae, mse, rmse

def gen_test_consumption_matrices(model, initial_grid_len, aggregated_time_series, normalized_data_test,tw):
    X = np.zeros((initial_grid_len, initial_grid_len, normalized_data_test.shape[0] - tw))
    X_noisy = np.zeros((initial_grid_len, initial_grid_len, normalized_data_test.shape[0] - tw))
    for i in range(initial_grid_len):
        for j in range(initial_grid_len):
            if (i,j) in aggregated_time_series:
                #X[i,j,:] =  aggregated_time_series[(i,j)]
                s = np.array([aggregated_time_series[(i,j)]])
                X_test_timeseries = get_inout_sequences(s, tw)
                test_inputs = [X_test_timeseries[i][0] for i in range(len(X_test_timeseries))]
                test_targets = [X_test_timeseries[i][1] for i in range(len(X_test_timeseries))]
                test_targets = [i[0] for i in test_targets]
                test_predictions = []
                with torch.no_grad():
                    for seq in test_inputs:
                        seq = torch.tensor(seq).float()  # Convert to tensor and ensure dtype is float32
                        y_pred = model(seq.unsqueeze(0))
                        test_predictions.append(y_pred.item())
                #if i==25 and j==8:
                #    print(test_targets)
                X[i,j,:] = copy.deepcopy(test_targets)
                X_noisy[i,j,:] = copy.deepcopy(test_predictions)
            #else:
            #    print('error')

    return X, X_noisy

def update_grid( data, initial_grid_len, converted_grid_len, cell_coords):

    cell_grid_len = int(initial_grid_len/converted_grid_len)

    d = defaultdict(list)

    for i in range(initial_grid_len):
        for j in range(initial_grid_len):
            i_con = i//cell_grid_len
            j_con = j//cell_grid_len

            d[(i_con,j_con)].append([i,j])


    aggregated_time_series = {}

    # Iterate over each cell and aggregate time series for users in that cell
    for i in range(converted_grid_len):
        for j in range(converted_grid_len):
            # Find users that belong to this cell

            coords_to_find = np.array(d[(i, j)])
            users_in_cell = []

            for coord in coords_to_find:
                matches = np.where((cell_coords == coord).all(axis=1))[0]

                # If matches are found, extend the indices list
                if matches.size > 0:
                    users_in_cell.extend(matches)



            #users_in_cell = np.where((cell_coords in np.array(d[(i,j)])).all(axis=1))[0]
            
            # Aggregate their time series data
            if len(users_in_cell) > 0:
                #aggregated_time_series[(i, j)] = normalized_data[:, users_in_cell].sum(axis=1)
                if data[:, users_in_cell].sum()<=1:
                    continue
                else:
                    aggregated_time_series[(i, j)] = data[:, users_in_cell].mean(axis=1)


    return aggregated_time_series

def train(model, criterion, optimizer, scheduler, train_loader, valid_loader, epochs):
    # Initialize the scheduler
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    # Training
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for seq, labels in train_loader:
            optimizer.zero_grad()
            seq = seq.float()
            labels = labels.float()

            y_pred = model(seq)
            loss = criterion(y_pred.view(-1), labels.view(-1))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        valid_loss = 0.0
        for seq, labels in valid_loader:
            seq = seq.float()
            labels = labels.float()
            y_pred = model(seq)
            loss = criterion(y_pred.view(-1), labels.view(-1))
            valid_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)

        # Step the scheduler with the validation loss
        #scheduler.step(avg_valid_loss)

        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}')    

    return model



def create_c_sanitized(C_test_cons, C_test_pattern, quantization_levels, pnrg, epsilon_sanitize = 10):


    #def create_c_sanitized(C_test_cons, C_test_pattern, quantization_levels):
    mean = np.mean(C_test_pattern)
    std = np.std(C_test_pattern)
    C_test_standardized = (C_test_pattern - mean) / std

    C_test_min = C_test_standardized.min()
    C_test_max = C_test_standardized.max()
    C_test_normalized = (C_test_standardized - C_test_min) / (C_test_max - C_test_min)

    C_test_pattern =C_test_normalized


    quantization = [(1/quantization_levels)*i for i in range(1,quantization_levels) ]
    #print(quantization)
    #print(len(quantization))

    sensitivity = []

    # The first one is just to figure out the sensitivity.

    for idx, i in enumerate(quantization):
        if idx==0:
            
            end = quantization[idx]

            mask =  (C_test_pattern <= end)

            #Step 2: Calculate the sum of these values
            sum_of_values = C_test_cons[mask].sum()

            # Step 3: Count the number of cells in this range
            num_cells = mask.sum()

            if num_cells > 0:
                s_i = np.max(np.sum(mask, axis=2))
                sensitivity.append(s_i)
                #print(' quantization level {}  has {}  cells and sensitivity is {}'.format(idx ,num_cells, s_i))
                #uniform_value = sum_of_values / num_cells
                #C_test_cons[mask] = uniform_value



        elif idx==len(quantization)-1:
            st = quantization[idx-1]
            end = quantization[idx]

            mask = (C_test_pattern > st) & (C_test_pattern <= end)

            #Step 2: Calculate the sum of these values
            sum_of_values = C_test_cons[mask].sum()
            # Step 3: Count the number of cells in this range
            num_cells = mask.sum()

            if num_cells > 0:
                s_i = np.max(np.sum(mask, axis=2))
                sensitivity.append(s_i)
                #print(' quantization level {}  has {}  cells and sensitivity is {}'.format(idx ,num_cells, s_i))

            ####################################################
            st = quantization[idx]
            mask = (C_test_pattern > st) 

            #Step 2: Calculate the sum of these values
            sum_of_values = C_test_cons[mask].sum()
            # Step 3: Count the number of cells in this range
            num_cells = mask.sum()

            if num_cells > 0:
                s_i = np.max(np.sum(mask, axis=2))
                sensitivity.append(s_i)
                #print(' quantization level {}  has {}  cells and sensitivity is {}'.format(idx ,num_cells, s_i))

        else:
            st = quantization[idx-1]
            end = quantization[idx]

            mask = (C_test_pattern > st) & (C_test_pattern <= end)

            #Step 2: Calculate the sum of these values
            sum_of_values = C_test_cons[mask].sum()
            # Step 3: Count the number of cells in this range
            num_cells = mask.sum()
            if num_cells > 0:
                s_i = np.max(np.sum(mask, axis=2))
                sensitivity.append(s_i)
                #print(' quantization level {}  has {}  cells and sensitivity is {}'.format(idx ,num_cells, s_i))




    _sensitivity_sum = 0

    for i in sensitivity:
        _sensitivity_sum += i**(2/3)


    ##############################################################################################################################
    for idx, i in enumerate(quantization):
        if idx==0:
            
            end = quantization[idx]

            mask =  (C_test_pattern <= end)

            #Step 2: Calculate the sum of these values
            sum_of_values = C_test_cons[mask].sum()


            # Step 3: Count the number of cells in this range
            num_cells = mask.sum()

            if num_cells > 0:
                s_i = np.max(np.sum(mask, axis=2))
                sensitivity = s_i
                epsilon = (epsilon_sanitize* (sensitivity**(2/3)))/ _sensitivity_sum
                #print(' quantization level {}  num cells {}  sensitivity {} budget {}'.format(idx ,num_cells, s_i, epsilon))
                b = sensitivity / epsilon
                sum_of_values = sum_of_values + pnrg.laplace(loc = 0.0, scale = b)
                uniform_value = sum_of_values / num_cells
                C_test_cons[mask] = uniform_value



        elif idx==len(quantization)-1:
            st = quantization[idx-1]
            end = quantization[idx]

            mask = (C_test_pattern > st) & (C_test_pattern <= end)

            #Step 2: Calculate the sum of these values
            sum_of_values = C_test_cons[mask].sum()
            # Step 3: Count the number of cells in this range
            num_cells = mask.sum()

            if num_cells > 0:
                s_i = np.max(np.sum(mask, axis=2))
                sensitivity = s_i
                epsilon = (epsilon_sanitize* (sensitivity**(2/3)))/ _sensitivity_sum
                #print(' quantization level {}  num cells {}  sensitivity {} budget {}'.format(idx ,num_cells, s_i, epsilon))
                b = sensitivity / epsilon
                sum_of_values = sum_of_values + pnrg.laplace(loc = 0.0, scale = b)
                uniform_value = sum_of_values / num_cells
                C_test_cons[mask] = uniform_value



            ####################################################
            st = quantization[idx]
            mask = (C_test_pattern > st) 

            #Step 2: Calculate the sum of these values
            sum_of_values = C_test_cons[mask].sum()
            # Step 3: Count the number of cells in this range
            num_cells = mask.sum()

            if num_cells > 0:
                s_i = np.max(np.sum(mask, axis=2))
                sensitivity = s_i
                epsilon = (epsilon_sanitize* (sensitivity**(2/3)))/ _sensitivity_sum
                #print(' quantization level {}  num cells {}  sensitivity {} budget {}'.format(idx ,num_cells, s_i, epsilon))
                b = sensitivity / epsilon
                sum_of_values = sum_of_values + pnrg.laplace(loc = 0.0, scale = b)
                uniform_value = sum_of_values / num_cells
                C_test_cons[mask] = uniform_value

        else:
            st = quantization[idx-1]
            end = quantization[idx]

            mask = (C_test_pattern > st) & (C_test_pattern <= end)

            #Step 2: Calculate the sum of these values
            sum_of_values = C_test_cons[mask].sum()
            # Step 3: Count the number of cells in this range
            num_cells = mask.sum()
            if num_cells > 0:
                s_i = np.max(np.sum(mask, axis=2))
                sensitivity = s_i
                epsilon = (epsilon_sanitize* (sensitivity**(2/3)))/ _sensitivity_sum
                #print(' quantization level {}  num cells {}  sensitivity {} budget {}'.format(idx ,num_cells, s_i, epsilon))
                b = sensitivity / epsilon
                sum_of_values = sum_of_values + pnrg.laplace(loc = 0.0, scale = b)
                uniform_value = sum_of_values / num_cells
                C_test_cons[mask] = uniform_value


    return C_test_cons