import numpy as np
import scipy
from numpy.linalg import inv as inv
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod
from scipy.stats import wishart
from scipy.stats import invwishart
from numpy.linalg import inv
from scipy.stats import wishart, multivariate_normal
from numpy.linalg import solve as solve
from numpy.linalg import cholesky as cholesky_lower
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import solve_triangular as solve_ut




def mvnrnd_pre(mu, Lambda):
    src = normrnd(size = (mu.shape[0],))
    return solve_ut(cholesky_upper(Lambda, overwrite_a = True, check_finite = False), 
                    src, lower = False, check_finite = False, overwrite_b = True) + mu

def cov_mat(mat, mat_bar):
    mat = mat - mat_bar
    return mat.T @ mat

def BTMF_rohit(dense_mat, sparse_mat, init, rank, time_lags, burn_iter, gibbs_iter, option = "factor"):
    dim1, dim2 = sparse_mat.shape
    d = time_lags.shape[0]
    W = np.random.rand(dim1, rank)
    X = np.random.rand(dim2, rank)
    A = np.random.rand(rank * d, rank)

def sample_hyperparameters(W, mu0, beta0, W0, nu0):

    # Compute the parameters for the conditional distributions
    N = W.shape[0]  # Number of rows in W
    rank = W.shape[1]  # Number of columns in W (i.e., the rank)
    W_bar = np.mean(W, axis=0)  # Mean of the rows in W
    S_W = np.dot((W - W_bar).T, W - W_bar) / N  # Covariance of the rows in W
    mu_w_star = (beta0 * mu0 + N * W_bar) / (beta0 + N)
    nu_w_star = nu0 + N
    W_w_star_inv = inv(W0 + S_W + beta0 * N / (beta0 + N) * np.outer(W_bar - mu0, W_bar - mu0))

    # Sample mu_w and Lambda_w from their conditional distributions
    Lambda_w = wishart(df=nu_w_star, scale=W_w_star_inv).rvs()
    mu_w = multivariate_normal(mean=mu_w_star, cov=inv((beta0 + N) * Lambda_w)).rvs()

    return mu_w, Lambda_w

def sample_var_coefficients(X, time_lags, M0, Psi0, S0, nu0):
    # Compute the parameters for the conditional distributions
    T = X.shape[0]  # Number of rows in X
    rank = X.shape[1]  # Number of columns in X (i.e., the rank)
    d = len(time_lags)  # Number of time lags
    Z = X[time_lags.max():]
    Q = np.zeros((T - time_lags.max(), rank * d))
    for k, tau in enumerate(time_lags):
        Q[:, k * rank:(k + 1) * rank] = X[time_lags.max() - tau:-tau]
    Psi_star_inv = inv(Psi0 + Q.T @ Q)
    M_star = Psi_star_inv @ (Psi0 @ M0 + Q.T @ Z)
    S_star = S0 + Z.T @ Z + M0.T @ Psi0 @ M0 - M_star.T @ inv(Psi_star_inv) @ M_star
    nu_star = nu0 + T - time_lags.max()

    # Sample A and Sigma from their conditional distributions
    Sigma = invwishart(df=nu_star, scale=S_star).rvs()
    A = np.reshape(multivariate_normal(mean=M_star.flatten(), cov=np.kron(Sigma, Psi_star_inv)).rvs(), (rank * d, rank))

    return A, Sigma
"""
def sample_factor_w(sparse_mat, ind, W, X, tau, mu_w, Lambda_w):
    # Compute the parameters for the conditional distributions
    dim1 = sparse_mat.shape[0]  # Number of rows in the sparse matrix
    rank = W.shape[1]  # Number of columns in W (i.e., the rank)
    for i in range(dim1):
        ind_i = ind[i, :]
        print(ind_i.shape)
        W[i, :] = multivariate_normal(mean=((tau[i] * X[ind_i, :].T @ sparse_mat[i, ind_i] + Lambda_w @ mu_w) / (tau[i] * X[ind_i, :].T @ X[ind_i, :] + Lambda_w)).flatten(), cov=inv(tau[i] * X[ind_i, :].T @ X[ind_i, :] + Lambda_w)).rvs()

    return W
"""

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



def sample_factor_x(sparse_mat, ind, time_lags, W, X, A, Sigma_inv):
    # Compute the parameters for the conditional distributions
    dim2 = sparse_mat.shape[1]  # Number of columns in the sparse matrix
    rank = W.shape[1]  # Number of columns in W (i.e., the rank)
    d = len(time_lags)  # Number of time lags
    t_max = time_lags.max()
    for t in range(dim2):
        if t < t_max:
            Pt = np.eye(rank)
            Qt = np.zeros(rank)
        else:
            Pt = Sigma_inv
            Qt = Sigma_inv @ np.sum([A[k * rank:(k + 1) * rank, :] @ X[t - time_lags[k], :] for k in range(d)], axis=0)
        ind_t = ind[:, t]
        Mt = np.dot(W[ind_t, :].T, np.multiply(ind_t, W)) + Pt
        Nt = np.dot(W[ind_t, :].T, sparse_mat[ind_t, t]) + Qt
        X[t, :] = multivariate_normal(mean=inv(Mt) @ Nt, cov=inv(Mt)).rvs()

    return X

def sample_precision_tau(sparse_mat, mat_hat, ind, alpha, beta):
    # Compute the parameters for the conditional distributions
    dim1 = sparse_mat.shape[0]  # Number of rows in the sparse matrix
    tau = np.zeros(dim1)
    for i in range(dim1):
        ind_i = ind[i, :]
        alpha_star = alpha + 0.5 * np.sum(ind_i)
        beta_star = beta + 0.5 * np.sum((sparse_mat[i, ind_i] - mat_hat[i, ind_i]) ** 2)
        tau[i] =scipy.stats.gamma(a=alpha_star, scale=1/beta_star).rvs()

    return tau

def BTMF_imputation(sparse_mat, init, rank, time_lags, burn_iter, gibbs_iter):
    # Initialize the factor matrices and hyperparameters
    W = init["W"]
    X = init["X"]
    A = init["A"]
    Sigma_inv = init["Sigma_inv"]
    mu_w = init["mu_w"]
    Lambda_w = init["Lambda_w"]
    alpha = init["alpha"]
    beta = init["beta"]
    tau = init["tau"]
    # Initialize the matrix for storing the imputed values
    mat_hat_plus = np.zeros(sparse_mat.shape)

    # Determine the observed entries
    ind = sparse_mat != 0

    # Run the Gibbs sampling procedure
    for it in range(burn_iter + gibbs_iter):
        W = sample_factor_w(sparse_mat, ind, W, X, tau, mu_w, Lambda_w)
        A, Sigma_inv = sample_var_coefficients(X, time_lags)
        X = sample_factor_x(sparse_mat, ind, time_lags, W, X, A, Sigma_inv)
        tau = sample_precision_tau(sparse_mat, W @ X.T, ind, alpha, beta)

        # After the burn-in period, start collecting samples
        if it >= burn_iter:
            mat_hat_plus += W @ X.T

    # Compute the average of the collected samples to get the imputed matrix
    mat_hat = mat_hat_plus / gibbs_iter

    return mat_hat

import scipy.io

# Load the .mat file
mat = scipy.io.loadmat('C:/Users/Rohit/Documents/Exeter-Placement/transdim-master/datasets/Guangzhou-data-set/tensor.mat')
data = mat['tensor']
rank=7
d=2
init = {
    "W": np.random.rand(data.shape[0], rank),
    "X": np.random.rand(data.shape[1], rank),
    "A": np.random.rand(rank * d, rank),
    "Sigma_inv": np.eye(rank),
    "mu_w": np.zeros(rank),
    "Lambda_w": np.eye(rank),
    "alpha": 1e-6,
    "beta": 1e-6,
    "tau": np.ones(data.shape[0])
}

# Set the rank and time lags
rank = 10  # for example
time_lags = np.array([1, 2, 3])  # for example

# Set the number of burn-in and Gibbs sampling iterations
burn_iter = 1000  # for example
gibbs_iter = 2000  # for example

# Run the BTMF imputation function
imputed_data = BTMF_imputation(data, init, rank, time_lags, burn_iter, gibbs_iter)



"""
def BTMF(dense_mat, sparse_mat, init, rank, time_lags, burn_iter, gibbs_iter, option = "factor"):

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
    W_plus = np.zeros((dim1, rank))
    X_plus = np.zeros((dim2, rank))
    A_plus = np.zeros((rank * d, rank))
    temp_hat = np.zeros(len(pos_test[0]))
    show_iter = 200
    mat_hat_plus = np.zeros((dim1, dim2))
    for it in range(burn_iter + gibbs_iter):
        tau_ind = tau[:, None] * ind
        tau_sparse_mat = tau[:, None] * sparse_mat
        #for i = 1 to N (can be in parallel) do Draw wi ∼N(μ∗w,(Λ∗w)−1)
        W = sample_factor_w(tau_sparse_mat, tau_ind, W, X, tau)
        #Draw Σ ∼IW(S∗,ν∗) and A ∼MN (M∗,Ψ∗,Σ):
        A, Sigma = sample_var_coefficient(X, time_lags)
        #for t = 1 to T do Draw xt ∼N(μ∗t,Σ∗t)
        X = sample_factor_x(tau_sparse_mat, tau_ind, time_lags, W, X, A, inv(Sigma))
        mat_hat = W @ X.T
        if option == "factor":
            #for i = 1 to N do Draw precision τi ∼Gamma(α∗i ,β∗i )
            tau = sample_precision_tau(sparse_mat, mat_hat, ind)
        elif option == "pca":
            #for i = 1 to N do Draw precision τi ∼Gamma(α∗i ,β∗i )
            tau = sample_precision_scalar_tau(sparse_mat, mat_hat, ind)
            tau = tau * np.ones(dim1)
        temp_hat += mat_hat[pos_test]
        #if iter. > m1 then Compute ̃Y = W>X. Collect sample ̃Y . end if: 
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
    
    return mat_hat, W, X, A
"""