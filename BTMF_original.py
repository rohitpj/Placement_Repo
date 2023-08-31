import numpy as np
from numpy.linalg import inv as inv
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod
from scipy.stats import wishart
from scipy.stats import invwishart
from numpy.linalg import solve as solve
from numpy.linalg import cholesky as cholesky_lower
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import solve_triangular as solve_ut
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal as mvnrnd

def mnrnd(M, U, V):
    """
    Generate matrix normal distributed random matrix.
    M is a m-by-n matrix, U is a m-by-m matrix, and V is a n-by-n matrix.
    """
    dim1, dim2 = M.shape
    X0 = np.random.randn(dim1, dim2)
    P = cholesky_lower(U)
    Q = cholesky_lower(V)
    
    return M + P @ X0 @ Q.T

def mvnrnd_pre(mu, Lambda):
    src = normrnd(size = (mu.shape[0],))
    return solve_ut(cholesky_upper(Lambda, overwrite_a = True, check_finite = False), 
                    src, lower = False, check_finite = False, overwrite_b = True) + mu

def cov_mat(mat, mat_bar):
    mat = mat - mat_bar
    return mat.T @ mat

def ar4cast(A, X, Sigma, time_lags, multi_step):
    dim, rank = X.shape
    d = time_lags.shape[0]
    X_new = np.append(X, np.zeros((multi_step, rank)), axis = 0)
    for t in range(multi_step):
        var = A.T @ X_new[dim + t - time_lags, :].reshape(rank * d)
        X_new[dim + t, :] = mvnrnd(var, Sigma)
    return X_new

def sample_factor_w(tau_sparse_mat, tau_ind, W, X, tau, beta0 = 1, vargin = 0):
    """Sampling N-by-R factor matrix W and its hyperparameters (mu_w, Lambda_w)."""
    
    dim1, rank = W.shape
    W_bar = np.mean(W, axis = 0)
    temp = dim1 / (dim1 + beta0)
    var_W_hyper = inv(np.eye(rank) + cov_mat(W, W_bar) + temp * beta0 * np.outer(W_bar, W_bar))
    var_Lambda_hyper = wishart.rvs(df = dim1 + rank, scale = var_W_hyper)
    var_mu_hyper = mvnrnd_pre(temp * W_bar, (dim1 + beta0) * var_Lambda_hyper)
    
    var1 = X.T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ tau_ind.T).reshape([rank, rank, dim1]) + var_Lambda_hyper[:, :, None]
    var4 = var1 @ tau_sparse_mat.T + (var_Lambda_hyper @ var_mu_hyper)[:, None]
    for i in range(dim1):
        W[i, :] = mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])

    return W

def mnrnd(M, U, V):

    dim1, dim2 = M.shape
    X0 = np.random.randn(dim1, dim2)
    P = cholesky_lower(U)
    Q = cholesky_lower(V)
    
    return M + P @ X0 @ Q.T

def sample_var_coefficient(X, time_lags):
    dim, rank = X.shape
    d = time_lags.shape[0]
    tmax = np.max(time_lags)
    
    Z_mat = X[tmax : dim, :]
    Q_mat = np.zeros((dim - tmax, rank * d))
    for k in range(d):
        Q_mat[:, k * rank : (k + 1) * rank] = X[tmax - time_lags[k] : dim - time_lags[k], :]
    var_Psi0 = np.eye(rank * d) + Q_mat.T @ Q_mat
    var_Psi = inv(var_Psi0)
    var_M = var_Psi @ Q_mat.T @ Z_mat
    var_S = np.eye(rank) + Z_mat.T @ Z_mat - var_M.T @ var_Psi0 @ var_M
    Sigma = invwishart.rvs(df = rank + dim - tmax, scale = var_S)
    
    return mnrnd(var_M, var_Psi, Sigma), Sigma

def sample_factor_x(tau_sparse_mat, tau_ind, time_lags, W, X, A, Lambda_x):
    """Sampling T-by-R factor matrix X."""
    
    dim2, rank = X.shape
    tmax = np.max(time_lags)
    tmin = np.min(time_lags)
    d = time_lags.shape[0]
    A0 = np.dstack([A] * d)
    for k in range(d):
        A0[k * rank : (k + 1) * rank, :, k] = 0
    mat0 = Lambda_x @ A.T
    mat1 = np.einsum('kij, jt -> kit', A.reshape([d, rank, rank]), Lambda_x)
    mat2 = np.einsum('kit, kjt -> ij', mat1, A.reshape([d, rank, rank]))
    
    var1 = W.T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ tau_ind).reshape([rank, rank, dim2]) + Lambda_x[:, :, None]
    var4 = var1 @ tau_sparse_mat
    for t in range(dim2):
        Mt = np.zeros((rank, rank))
        Nt = np.zeros(rank)
        Qt = mat0 @ X[t - time_lags, :].reshape(rank * d)
        index = list(range(0, d))
        if t >= dim2 - tmax and t < dim2 - tmin:
            index = list(np.where(t + time_lags < dim2))[0]
        elif t < tmax:
            Qt = np.zeros(rank)
            index = list(np.where(t + time_lags >= tmax))[0]
        if t < dim2 - tmin:
            Mt = mat2.copy()
            temp = np.zeros((rank * d, len(index)))
            n = 0
            for k in index:
                temp[:, n] = X[t + time_lags[k] - time_lags, :].reshape(rank * d)
                n += 1
            temp0 = X[t + time_lags[index], :].T - np.einsum('ijk, ik -> jk', A0[:, :, index], temp)
            Nt = np.einsum('kij, jk -> i', mat1[index, :, :], temp0)
        
        var3[:, :, t] = var3[:, :, t] + Mt
        if t < tmax:
            var3[:, :, t] = var3[:, :, t] - Lambda_x + np.eye(rank)
        X[t, :] = mvnrnd_pre(solve(var3[:, :, t], var4[:, t] + Nt + Qt), var3[:, :, t])

    return X

def sample_precision_tau(sparse_mat, mat_hat, ind):
    var_alpha = 1e-6 + 0.5 * np.sum(ind, axis = 1)
    var_beta = 1e-6 + 0.5 * np.sum(((sparse_mat - mat_hat) ** 2) * ind, axis = 1)
    return np.random.gamma(var_alpha, 1 / var_beta)

def sample_precision_scalar_tau(sparse_mat, mat_hat, ind):
    var_alpha = 1e-6 + 0.5 * np.sum(ind)
    var_beta = 1e-6 + 0.5 * np.sum(((sparse_mat - mat_hat) ** 2) * ind)
    return np.random.gamma(var_alpha, 1 / var_beta)

def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]

def compute_rmse(var, var_hat):
    return  np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])

"""
def BTMF(dense_mat, sparse_mat, init, rank, time_lags, burn_iter, gibbs_iter, option = "factor"):
    original_mat = sparse_mat.copy()
    dim1, dim2 = sparse_mat.shape
    #print("dense_mat shape",dense_mat.shape)
    d = time_lags.shape[0]
    W = init["W"]
    X = init["X"]

    if np.isnan(sparse_mat).any() == False:
        ind = sparse_mat != 0
        pos_obs = np.where(ind)
        pos_test = np.where((dense_mat != 0) & (sparse_mat == 0))

        #print("pos test  shape: ", pos_test)
    elif np.isnan(sparse_mat).any() == True:
        pos_test = np.where((dense_mat != 0) & (np.isnan(sparse_mat)))
        ind = ~np.isnan(sparse_mat)
        pos_obs = np.where(ind)
        sparse_mat[np.isnan(sparse_mat)] = 0

    dense_test = dense_mat[pos_test]
    del dense_mat
    tau = np.ones(dim1)
    W_plus = np.zeros((dim1, rank))
    X_plus = np.zeros((dim2, rank))
    A_plus = np.zeros((rank * d, rank))
    temp_hat = np.zeros(len(pos_test[0]))
    show_iter = 999
    mat_hat_plus = np.zeros((dim1, dim2))
    for it in range(burn_iter + gibbs_iter):
        print("BTMF Iteration:",it)
        tau_ind = tau[:, None] * ind
        tau_sparse_mat = tau[:, None] * sparse_mat
        W = sample_factor_w(tau_sparse_mat, tau_ind, W, X, tau)
        A, Sigma = sample_var_coefficient(X, time_lags)
        X = sample_factor_x(tau_sparse_mat, tau_ind, time_lags, W, X, A, inv(Sigma))
        mat_hat = W @ X.T
        if option == "factor":
            tau = sample_precision_tau(sparse_mat, mat_hat, ind)
        elif option == "pca":
            tau = sample_precision_scalar_tau(sparse_mat, mat_hat, ind)
            tau = tau * np.ones(dim1)
        temp_hat += mat_hat[pos_test]
        if (it + 1) % show_iter == 0 and it < burn_iter:
            temp_hat = temp_hat / show_iter
            print('Iter: {}'.format(it + 1))
            print('MAPE: {:.6}'.format(compute_mape(dense_test, temp_hat)))
            print('RMSE: {:.6}'.format(compute_rmse(dense_test, temp_hat)))
            temp_hat = np.zeros(len(pos_test[0]))
            print()
        if it + 1 > burn_iter:
            W_plus += W
            X_plus += X
            A_plus += A
            mat_hat_plus += mat_hat
    mat_hat = mat_hat_plus / gibbs_iter
    W = W_plus / gibbs_iter
    X = X_plus / gibbs_iter
    A = A_plus / gibbs_iter
    #print('Imputation MAPE: {:.6}'.format(compute_mape(dense_test, mat_hat[:, : dim2][pos_test])))
    #print('Imputation RMSE: {:.6}'.format(compute_rmse(dense_test, mat_hat[:, : dim2][pos_test])))
    #print()
    mat_hat[mat_hat < 0] = 0
    nan_positions = np.isnan(original_mat)
    original_mat[nan_positions] = mat_hat[nan_positions]
    return original_mat, W, X, A
"""
def BTMF(dense_mat, sparse_mat, init, rank, time_lags, burn_iter, gibbs_iter, multi_step = 1, option = "factor"):
    """Bayesian Temporal Matrix Factorization, BTMF."""
    
    dim1, dim2 = sparse_mat.shape
    d = time_lags.shape[0]
    W = init["W"]
    X = init["X"]
    if np.isnan(sparse_mat).any() == False:
        ind = sparse_mat != 0
        pos_obs = np.where(ind)
        pos_test = np.where((dense_mat != 0) & (sparse_mat == 0))
    elif np.isnan(sparse_mat).any() == True:
        pos_test = np.where((dense_mat != 0) & (np.isnan(sparse_mat)))
        ind = ~np.isnan(sparse_mat)
        pos_obs = np.where(ind)
        sparse_mat[np.isnan(sparse_mat)] = 0
    dense_test = dense_mat[pos_test]
    del dense_mat
    tau = np.ones(dim1)
    W_plus = np.zeros((dim1, rank, gibbs_iter))
    A_plus = np.zeros((rank * d, rank, gibbs_iter))
    tau_plus = np.zeros((dim1, gibbs_iter))
    Sigma_plus = np.zeros((rank, rank, gibbs_iter))
    temp_hat = np.zeros(len(pos_test[0]))
    show_iter = 500
    mat_hat_plus = np.zeros((dim1, dim2))
    X_plus = np.zeros((dim2 + multi_step, rank, gibbs_iter))
    mat_new_plus = np.zeros((dim1, multi_step))
    for it in range(burn_iter + gibbs_iter):
        tau_ind = tau[:, None] * ind
        tau_sparse_mat = tau[:, None] * sparse_mat
        W = sample_factor_w(tau_sparse_mat, tau_ind, W, X, tau, beta0 = 1, vargin = 0)
        A, Sigma = sample_var_coefficient(X, time_lags)
        X = sample_factor_x(tau_sparse_mat, tau_ind, time_lags, W, X, A, inv(Sigma))
        mat_hat = W @ X.T
        if option == "factor":
            tau = sample_precision_tau(sparse_mat, mat_hat, ind)
        elif option == "pca":
            tau = sample_precision_scalar_tau(sparse_mat, mat_hat, ind)
            tau = tau * np.ones(dim1)
        temp_hat += mat_hat[pos_test]
        if (it + 1) % show_iter == 0 and it < burn_iter:
            temp_hat = temp_hat / show_iter
            print('Iter: {}'.format(it + 1))
            print('MAPE: {:.6}'.format(compute_mape(dense_test, temp_hat)))
            print('RMSE: {:.6}'.format(compute_rmse(dense_test, temp_hat)))
            temp_hat = np.zeros(len(pos_test[0]))
            print()
        if it + 1 > burn_iter:
            W_plus[:, :, it - burn_iter] = W
            A_plus[:, :, it - burn_iter] = A
            Sigma_plus[:, :, it - burn_iter] = Sigma
            tau_plus[:, it - burn_iter] = tau
            mat_hat_plus += mat_hat
            X0 = ar4cast(A, X, Sigma, time_lags, multi_step)
            X_plus[:, :, it - burn_iter] = X0
            mat_new_plus += W @ X0[dim2 : dim2 + multi_step, :].T
    mat_hat = mat_hat_plus / gibbs_iter
    #print('Imputation MAPE: {:.6}'.format(compute_mape(dense_test, mat_hat[:, : dim2][pos_test])))
    #print('Imputation RMSE: {:.6}'.format(compute_rmse(dense_test, mat_hat[:, : dim2][pos_test])))
    print()
    mat_hat = np.append(mat_hat, mat_new_plus / gibbs_iter, axis = 1)
    mat_hat[mat_hat < 0] = 0
    
    return mat_hat, W_plus, X_plus, A_plus, Sigma_plus, tau_plus

def sample_factor_x_partial(tau_sparse_mat, tau_ind, time_lags, W, X, A, Lambda_x, back_step):
    """Sampling T-by-R factor matrix X."""
    
    dim2, rank = X.shape
    tmax = np.max(time_lags)
    tmin = np.min(time_lags)
    d = time_lags.shape[0]
    A0 = np.dstack([A] * d)
    for k in range(d):
        A0[k * rank : (k + 1) * rank, :, k] = 0
    mat0 = Lambda_x @ A.T
    mat1 = np.einsum('kij, jt -> kit', A.reshape([d, rank, rank]), Lambda_x)
    mat2 = np.einsum('kit, kjt -> ij', mat1, A.reshape([d, rank, rank]))
    
    var1 = W.T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ tau_ind[:, - back_step :]).reshape([rank, rank, back_step]) + Lambda_x[:, :, None]
    var4 = var1 @ tau_sparse_mat[:, - back_step :]
    for t in range(dim2 - back_step, dim2):
        Mt = np.zeros((rank, rank))
        Nt = np.zeros(rank)
        Qt = mat0 @ X[t - time_lags, :].reshape(rank * d)
        index = list(range(0, d))
        if t >= dim2 - tmax and t < dim2 - tmin:
            index = list(np.where(t + time_lags < dim2))[0]
        if t < dim2 - tmin:
            Mt = mat2.copy()
            temp = np.zeros((rank * d, len(index)))
            n = 0
            for k in index:
                temp[:, n] = X[t + time_lags[k] - time_lags, :].reshape(rank * d)
                n += 1
            temp0 = X[t + time_lags[index], :].T - np.einsum('ijk, ik -> jk', A0[:, :, index], temp)
            Nt = np.einsum('kij, jk -> i', mat1[index, :, :], temp0)
        var3[:, :, t + back_step - dim2] = var3[:, :, t + back_step - dim2] + Mt
        X[t, :] = mvnrnd_pre(solve(var3[:, :, t + back_step - dim2], 
                                   var4[:, t + back_step - dim2] + Nt + Qt), var3[:, :, t + back_step - dim2])
    return X

def BTMF_partial(dense_mat, sparse_mat, init, rank, time_lags, burn_iter, gibbs_iter, multi_step = 1, gamma = 10):
    """Bayesian Temporal Matrix Factorization, BTMF."""
    
    dim1, dim2 = sparse_mat.shape
    W_plus = init["W_plus"]
    X_plus = init["X_plus"]
    A_plus = init["A_plus"]
    Sigma_plus = init["Sigma_plus"]
    tau_plus = init["tau_plus"]
    if np.isnan(sparse_mat).any() == False:
        ind = sparse_mat != 0
        pos_obs = np.where(ind)
    elif np.isnan(sparse_mat).any() == True:
        ind = ~np.isnan(sparse_mat)
        pos_obs = np.where(ind)
        sparse_mat[np.isnan(sparse_mat)] = 0
    X_new_plus = np.zeros((dim2 + multi_step, rank, gibbs_iter))
    mat_new_plus = np.zeros((dim1, multi_step))
    back_step = gamma * multi_step
    for it in range(gibbs_iter):
        tau_ind = tau_plus[:, it][:, None] * ind
        tau_sparse_mat = tau_plus[:, it][:, None] * sparse_mat
        X = sample_factor_x_partial(tau_sparse_mat, tau_ind, time_lags, W_plus[:, :, it], 
                                    X_plus[:, :, it], A_plus[:, :, it], inv(Sigma_plus[:, :, it]), back_step)
        X0 = ar4cast(A_plus[:, :, it], X, Sigma_plus[:, :, it], time_lags, multi_step)
        X_new_plus[:, :, it] = X0
        mat_new_plus += W_plus[:, :, it] @ X0[- multi_step :, :].T
    mat_hat = mat_new_plus / gibbs_iter
    mat_hat[mat_hat < 0] = 0
    
    return mat_hat, W_plus, X_new_plus, A_plus, Sigma_plus, tau_plus

def BTMF_forecast(dense_mat, sparse_mat, pred_step, multi_step, rank, time_lags, burn_iter, gibbs_iter, option = "factor", gamma = 10):
    dim1, T = dense_mat.shape
    start_time = T - pred_step
    max_count = int(np.ceil(pred_step / multi_step))
    mat_hat = np.zeros((dim1, max_count * multi_step))
    mape_values=[]
    rmse_values=[]
    #f = IntProgress(min = 0, max = max_count) # instantiate the bar
    #display(f) # display the bar
    for t in range(max_count):
        if t == 0:
            init = {"W": 0.1 * np.random.randn(dim1, rank), "X": 0.1 * np.random.randn(start_time, rank)}
            mat, W, X_new, A, Sigma, tau = BTMF(dense_mat[:, 0 : start_time], 
                sparse_mat[:, 0 : start_time], init, rank, time_lags, burn_iter, gibbs_iter, multi_step, option)
        else:
            init = {"W_plus": W, "X_plus": X_new, "A_plus": A, "Sigma_plus": Sigma, "tau_plus": tau}
            mat, W, X_new, A, Sigma, tau = BTMF_partial(dense_mat[:, 0 : start_time + t * multi_step], 
                sparse_mat[:, 0 : start_time + t * multi_step], init, rank, time_lags, 
                burn_iter, gibbs_iter, multi_step, gamma)
            small_dense_mat = dense_mat[:, start_time : T]
            pos = np.where(small_dense_mat != 0)
            mape_loss = compute_mape(small_dense_mat[pos], mat_hat[pos])  # or compute_mape, based on your preference
            mape_values.append(mape_loss)
            rmse_loss = compute_rmse(small_dense_mat[pos], mat_hat[pos])  # or compute_mape, based on your preference
            rmse_values.append(rmse_loss)
            if t%50 == 0:
                print('Prediction MAPE: {:.6}'.format(mape_loss))
                print('Prediction RMSE: {:.6}'.format(rmse_loss))
        mat_hat[:, t * multi_step : (t + 1) * multi_step] = mat[:, - multi_step :]
        #f.value = t
    small_dense_mat = dense_mat[:, start_time : T]
    pos = np.where(small_dense_mat != 0)
    mape_value = compute_mape(small_dense_mat[pos], mat_hat[pos])
    rmse_value = compute_rmse(small_dense_mat[pos], mat_hat[pos])
    print('Prediction MAPE: {:.6}'.format(mape_value))
    print('Prediction RMSE: {:.6}'.format(rmse_value))
    print()
    return mat_hat, mape_values, rmse_values

from datetime import datetime
from distutils.util import strtobool

import pandas as pd

def convert_tsf_to_dataframe(full_file_path_and_name, replace_missing_vals_with="NaN", value_column_name="series_value",):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )

import pandas as pd

# Read the .tsf file

# Convert to DataFrame

loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe("C:/Users/Rohit/Documents/Exeter-Placement/Challenge/phase_1 data/phase_1_data/phase_1_data.tsf")

print(loaded_data)
#print(loaded_data.shape,frequency, forecast_horizon, contain_missing_values, contain_equal_length)
building_data = loaded_data[loaded_data['series_name'].str.contains('Building')]

max_start_timestamp = building_data['start_timestamp'].max()

building_data['num_timestamps'] = building_data['series_value'].apply(len)

min_timestamps = building_data['num_timestamps'].min()

# Trim each time series to have the same length as the shortest one
building_data['uniform_series'] = building_data['series_value'].apply(lambda x: x[-min_timestamps:])

# Update the 'num_timestamps' column to reflect the new uniform length
building_data['num_timestamps'] = building_data['uniform_series'].apply(len)

import scipy.io
import numpy as np
import time
import datetime as dt

# Assuming loaded_data is already prepared and is in the format of a dense matrix
dense_mat = building_data['uniform_series']  # Adjust this based on how you've prepared loaded_data

  # Assuming you want to work on a copy of the original data

list_of_arrays = [np.array(series) for series in dense_mat]
mape_values = []
rmse_values = []
# Stack these arrays vertically to form a 2D matrix
dense_mat_2d = np.vstack(list_of_arrays)
sparse_mat = dense_mat_2d.copy()
print("dense mat shape",dense_mat_2d.shape)
print(dense_mat_2d)
dense_mat_2d = np.where(dense_mat_2d == 'NaN', np.nan, dense_mat_2d).astype(float)
sparse_mat = np.where(sparse_mat == 'NaN', np.nan, sparse_mat).astype(float)

# Model Setting
rank = 20
pred_step = 2976 
time_lags = np.array([1, 4, 96])  
#time_lags = np.array([1, 96, 384])  # Adjust this based on the seasonality or patterns in your data#
burn_iter = 20
gibbs_iter = 5
multi_step = 1
dim1, dim2 = sparse_mat.shape
init = {"W": 0.1 * np.random.randn(dim1, rank), "X": 0.1 * np.random.randn(dim2, rank)}
print("starting prediction")
# Apply BTMF forecasting for different prediction time horizons (if needed)


start = time.time()
print('Prediction time horizon (delta) = {}.'.format(multi_step))
building_mat_hat, old_mape_values, old_rmse_values = BTMF_forecast(dense_mat_2d, sparse_mat, pred_step, multi_step, rank, time_lags, burn_iter, gibbs_iter)
plt.plot(old_mape_values, marker='o', color='red')
plt.title('Loss over Iterations')
plt.xlabel('Iteration')
plt.ylabel('MAPE Loss)')
plt.grid(True)
filename = "C:/Users/Rohit/Documents/Exeter-Placement/Results/mape_plot_" + dt.datetime.now().strftime('%Y%m%d_%H%M%S') + ".png"
plt.savefig(filename)
plt.show()

plt.plot(old_rmse_values, marker='o', color='red')
plt.title('Loss over Iterations')
plt.xlabel('Iteration')
plt.ylabel('RMSE Loss)')
plt.grid(True)
filename = "C:/Users/Rohit/Documents/Exeter-Placement/Results/rmse_plot_" + dt.datetime.now().strftime('%Y%m%d_%H%M%S') + ".png"
plt.savefig(filename)
plt.show()

print(building_mat_hat)
end = time.time()
print('Running time: %d seconds'%(end - start))
print()
print(building_mat_hat.shape)
plt.figure(figsize=(10, 5))


plt.tight_layout()
plt.show()

