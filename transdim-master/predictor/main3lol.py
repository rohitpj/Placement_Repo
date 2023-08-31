import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv as inv
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod
from scipy.stats import wishart
from scipy.stats import invwishart
from numpy.linalg import solve as solve
from numpy.linalg import cholesky as cholesky_lower
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import solve_triangular as solve_ut

def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]

def compute_rmse(var, var_hat):
    return  np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])

def mvnrnd_pre(mu, Lambda):
    src = normrnd(size = (mu.shape[0],))
    return solve_ut(cholesky_upper(Lambda, overwrite_a = True, check_finite = False), 
                    src, lower = False, check_finite = False, overwrite_b = True) + mu

def mnrnd(M, U, V):

    dim1, dim2 = M.shape
    X0 = np.random.randn(dim1, dim2)
    P = cholesky_lower(U)
    Q = cholesky_lower(V)
    
    return M + P @ X0 @ Q.T

def cov_mat(mat, mat_bar):
    mat = mat - mat_bar
    return mat.T @ mat

def sample_factor_w(tau_sparse_mat, tau_ind, W, X, tau, beta0 = 1, vargin = 0):

    dim1, rank = W.shape
    W_bar = np.mean(W, axis = 0)
    temp = dim1 / (dim1 + beta0)
    var_W_hyper = inv(np.eye(rank) + cov_mat(W, W_bar) + temp * beta0 * np.outer(W_bar, W_bar))
    var_Lambda_hyper = wishart.rvs(df = dim1 + rank, scale = var_W_hyper)
    var_mu_hyper = mvnrnd_pre(temp * W_bar, (dim1 + beta0) * var_Lambda_hyper)

    for i in range(dim1):
        #correction 1
        tau_ind = tau_ind.astype(bool)
        pos0 = tau_ind[i, :]
        Xt = X[pos0, :]
        var_Lambda = Xt.T @ np.diag(tau[i, pos0]) @ Xt + var_Lambda_hyper
        inv_var_Lambda = inv((var_Lambda + var_Lambda.T)/2)
        var_mu = inv_var_Lambda @ (Xt.T @ np.diag(tau[i, pos0]) @ tau_sparse_mat[i, pos0] + var_Lambda_hyper @ var_mu_hyper)
        W[i, :] = mvnrnd_pre(var_mu, inv_var_Lambda)
    return W

def new_sample_factor_w(tau_sparse_mat, tau_ind, W, X, tau, beta0 = 1, vargin = 0):
    """Sampling N-by-R factor matrix W and its hyperparameters (mu_w, Lambda_w)."""
    
    dim1, rank = W.shape
    W_bar = np.mean(W, axis = 0)
    temp = dim1 / (dim1 + beta0)
    #might be self.S problem variable
    var_W_hyper = inv(np.eye(rank) + cov_mat(W, W_bar) + temp * beta0 * np.outer(W_bar, W_bar))
    var_Lambda_hyper = wishart.rvs(df = dim1 + rank, scale = var_W_hyper)
    var_mu_hyper = mvnrnd_pre(temp * W_bar, (dim1 + beta0) * var_Lambda_hyper)
    
    if vargin == 0:
        var1 = X.T
        var2 = kr_prod(var1, var1)
        var3 = (var2 @ tau_ind.T).reshape([rank, rank, dim1]) + var_Lambda_hyper[:, :, None]
        var4 = var1 @ tau_sparse_mat.T + (var_Lambda_hyper @ var_mu_hyper)[:, None]
        for i in range(dim1):
            W[i, :] = mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])

    return W

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

def new_sample_var_coefficient(X, time_lags):
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
        if t >= 0 and t <= tmax - 1:
            Qt = mat0 @ X[t - time_lags, :].reshape(rank * d)
        elif t >= tmax and t <= dim2 - tmin:
            Qt = mat0 @ X[t - time_lags, :].reshape(rank * d)
            index = list(range(0, d))
            Mt = mat2.copy()
            temp = np.zeros((rank * d, len(index)))
            n = 0
            for k in index:
                #temp[n * rank : (n + 1) * rank, :] = X[t - time_lags[k], :]
                temp[n : n + rank, :] = X[t - time_lags[k], :][None, :]


            Nt = np.einsum('ijk, ik -> j', A0, temp)
        elif t >= dim2 - tmin + 1 and t <= dim2 - 1:
            index = list(np.where((t - time_lags >= 0)))[0]
            Qt = mat0[index, :] @ X[t - time_lags[index], :].reshape(rank * len(index))
            Mt = mat2[index, :, :][:, index, :]
            temp = np.zeros((rank * len(index), len(index)))
            n = 0
            for k in index:
                #temp[n * rank : (n + 1) * rank, :] = X[t - time_lags[k], :]
                temp[n * rank : (n + 1) * rank, :] = X[t - time_lags[k], :][None, :]
                n += 1
            Nt = np.einsum('ijk, ik -> j', A0[index, :, :][:, :, index], temp)
        elif t == dim2:
            Qt = np.zeros(rank)
            Mt = np.zeros((rank, rank))
            Nt = np.zeros(rank)
        var_mu = var4[:, t] + var3[:, :, t] @ (Qt - Mt @ X[t, :])
        X[t, :] = mvnrnd_pre(var_mu, var3[:, :, t])
    return X
#sample_factor_x_partial(tau_sparse_mat, tau_ind, time_lags, W_plus[:, :, it], X_plus[:, :, it], A_plus[:, :, it], inv(Sigma_plus[:, :, it]), back_step)
def new_sample_factor_x(tau_sparse_mat, tau_ind, time_lags, W,                 X,                A,                 Lambda_x):
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

def mainBTMF(dense_mat, sparse_mat, init, rank, time_lags, burn_iter, gibbs_iter, option = "factor"):
    """Bayesian Temporal Matrix Factorization, BTMF."""
    mape_losses = []
    rmse_losses = []
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
    print("pos test", pos_test)
    print("dense_test shape: ", dense_test.shape)
    #tau = np.ones(dim1)
    tau = np.ones((dim1, dim2))
    W_plus = np.zeros((dim1, rank))
    X_plus = np.zeros((dim2, rank))
    A_plus = np.zeros((rank * d, rank))
    temp_hat = np.zeros(len(pos_test[0]))
    show_iter = 5
    mat_hat_plus = np.zeros((dim1, dim2))
    
    for it in range(burn_iter + gibbs_iter):
        print("Iteration: ",it)
        W = sample_factor_w(sparse_mat, ind, W, X, tau)
        A, Sigma = sample_var_coefficient(X, time_lags)
        X = new_sample_factor_x(sparse_mat, ind, time_lags, W, X, A, inv(Sigma))
        mat_hat = W @ X.T
        print("dense test, mat_hat[pos test] shape, ", dense_test.shape, mat_hat[pos_test].shape)
        mape_loss = compute_mape(dense_test, mat_hat[pos_test])
        rmse_loss = compute_rmse(dense_test, mat_hat[pos_test])
        mape_losses.append(mape_loss)
        rmse_losses.append(rmse_loss)
        if it + 1 > burn_iter:
            W_plus += W
            X_plus += X
            A_plus += A
            mat_hat_plus += mat_hat
        print("dense test shape", dense_test.shape,"mat hat shape", mat_hat[pos_test].shape)
        print('Iter: {}'.format(it + 1))
        print('MAPE: {:.6}'.format(compute_mape(dense_test, mat_hat[pos_test])))
        print('RMSE: {:.6}'.format(compute_rmse(dense_test, mat_hat[pos_test])))
        print()

    W = W_plus / gibbs_iter
    X = X_plus / gibbs_iter
    A = A_plus / gibbs_iter
    mat_hat = mat_hat_plus / gibbs_iter
    print('Iter: {}'.format(it + 1))
    print("dense test shape", dense_test.shape,"mat hat shape", mat_hat[pos_test].shape)
    print('MAPE: {:.6}'.format(compute_mape(dense_test, mat_hat[pos_test])))
    print('RMSE: {:.6}'.format(compute_rmse(dense_test, mat_hat[pos_test])))
    return mat_hat, W, X, A, mape_losses, rmse_losses

def new_BTMF(dense_mat, sparse_mat, init, rank, time_lags, burn_iter, gibbs_iter, option = "factor"):

    
    dim1, dim2 = sparse_mat.shape
    d = time_lags.shape[0]
    new_mape_losses = []
    new_rmse_losses = []
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
    W_plus = np.zeros((dim1, rank))
    X_plus = np.zeros((dim2, rank))
    A_plus = np.zeros((rank * d, rank))
    temp_hat = np.zeros(len(pos_test[0]))
    show_iter = 200
    mat_hat_plus = np.zeros((dim1, dim2))
    for it in range(burn_iter + gibbs_iter):
        print(it)
        tau_ind = tau[:, None] * ind
        tau_sparse_mat = tau[:, None] * sparse_mat
        #Draw wi ∼N(μ∗w,(Λ∗w)−1)
        W = new_sample_factor_w(tau_sparse_mat, tau_ind, W, X, tau)
        #Draw Σ ∼IW(S∗,ν∗) and A ∼MN (M∗,Ψ∗,Σ):
        A, Sigma = new_sample_var_coefficient(X, time_lags)
        #Draw xt ∼N(μ∗t,Σ∗t)
        X = new_sample_factor_x(tau_sparse_mat, tau_ind, time_lags, W, X, A, inv(Sigma))
        mat_hat = W @ X.T
        if option == "factor":
            #Draw precision τi ∼Gamma(α∗i ,β∗i )
            tau = sample_precision_tau(sparse_mat, mat_hat, ind)
        elif option == "pca":
            #Draw precision τi ∼Gamma(α∗i ,β∗i )
            tau = sample_precision_scalar_tau(sparse_mat, mat_hat, ind)
            tau = tau * np.ones(dim1)
        temp_hat += mat_hat[pos_test]
        #if iter. > m1 then Compute ̃Y = W>X. Collect sample ̃Y . end if: 
        mape_loss = compute_mape(dense_test, mat_hat[pos_test])
        rmse_loss = compute_rmse(dense_test, mat_hat[pos_test])
        new_mape_losses.append(mape_loss)
        new_rmse_losses.append(rmse_loss)
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
    #return ˆY as the average of the m2 samples of ̃Y 
    mat_hat = mat_hat_plus / gibbs_iter
    W = W_plus / gibbs_iter
    X = X_plus / gibbs_iter
    A = A_plus / gibbs_iter
    print('Imputation MAPE: {:.6}'.format(compute_mape(dense_test, mat_hat[:, : dim2][pos_test])))
    print('Imputation RMSE: {:.6}'.format(compute_rmse(dense_test, mat_hat[:, : dim2][pos_test])))
    print()
    mat_hat[mat_hat < 0] = 0
    
    return mat_hat, W, X, A, new_mape_losses, new_rmse_losses
print("hello world")
"""
import time
import scipy.io
import numpy as np
np.random.seed(1000)

print("hello")

#df = pd.read_excel('C:/Users/Rohit/Documents/Exeter-Placement/Archive_Data/Gen_Demand_Data_Sc3_Chausey_Scenario1.xlsx', engine='openpyxl')
#data_array = df.values
dense_tensor = scipy.io.loadmat('C:/Users/Rohit/Documents/Exeter-Placement/transdim-master/datasets/Guangzhou-data-set/tensor.mat')['tensor']
dim = dense_tensor.shape
missing_rate = 0.4 # Random missing (RM)
sparse_tensor = dense_tensor * np.round(np.random.rand(dim[0], dim[1], dim[2]) + 0.5 - missing_rate)
dense_mat = dense_tensor.reshape([dim[0], dim[1] * dim[2]])
sparse_mat = sparse_tensor.reshape([dim[0], dim[1] * dim[2]])
del dense_tensor, sparse_tensor

import time
start = time.time()
dim1, dim2 = sparse_mat.shape
rank = 80
time_lags = np.array([1, 2, 144])
init = {"W": 0.1 * np.random.randn(dim1, rank), "X": 0.1 * np.random.randn(dim2, rank)}
burn_iter = 10
gibbs_iter = 2
mat_hat, W, X, A, mape_losses, rmse_losses = BTMF(dense_mat, sparse_mat, init, rank, time_lags, burn_iter, gibbs_iter)
new_mat_hat, new_W, new_X, new_A, new_mape_losses, new_rmse_losses = new_BTMF(dense_mat, sparse_mat, init, rank, time_lags, burn_iter, gibbs_iter)

# Assuming mape_losses, new_mape_losses, rmse_losses, new_rmse_losses are lists of losses
iterations = np.arange(1, len(mape_losses) + 1)

import pandas as pd
import numpy as np
import scipy.io
import time

# Load the data
df = pd.read_excel('C:/Users/Rohit/Documents/Exeter-Placement/Archive_Data/Gen_Demand_Data_Sc3_Chausey_Scenario1.xlsx', engine='openpyxl')
dense_mat = df.values

# Handle missing values (assuming NaN represents missing values in your .xlsx file)
sparse_mat = np.copy(dense_mat)
sparse_mat[np.isnan(dense_mat)] = 0  # Set NaNs to 0 for the sparse matrix

# Parameters
rank = 80
time_lags = np.array([1, 2, 144])
init = {"W": 0.1 * np.random.randn(dense_mat.shape[0], rank), "X": 0.1 * np.random.randn(dense_mat.shape[1], rank)}
burn_iter = 200
gibbs_iter = 50

# Run BTMF
start = time.time()
mat_hat, W, X, A ,mape_losses,rmse_losses= BTMF(dense_mat, sparse_mat, init, rank, time_lags, burn_iter, gibbs_iter)
end = time.time()

print('Running time: %d seconds'%(end - start))
iterations = np.arange(1, len(mape_losses) + 1)

plt.figure()
plt.plot(iterations, mape_losses, label='Original Method')
plt.plot(iterations, new_mape_losses, label='New Method')
plt.title('MAPE Loss over Iterations')
plt.xlabel('Iteration')
plt.ylabel('MAPE Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(iterations, rmse_losses, label='Original Method')
plt.plot(iterations, new_rmse_losses, label='New Method')
plt.title('RMSE Loss over Iterations')
plt.xlabel('Iteration')
plt.ylabel('RMSE Loss')
plt.legend()
plt.show()
end = time.time()

print('Running time: %d seconds'%(end - start))
"""