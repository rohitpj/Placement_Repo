#implmentation of BMF for time series imputation

import numpy as np
from scipy.stats import wishart, multivariate_normal
from numpy.linalg import inv as inv
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod
from scipy.stats import wishart
from scipy.stats import invwishart
from scipy.stats import multivariate_normal as mvn
from numpy.linalg import solve as solve
from numpy.linalg import cholesky as cholesky_lower
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import solve_triangular as solve_ut
import matplotlib.pyplot as plt
from scipy.stats import expon as exp
import pandas as pd
#import openpyxl as xl

#stole all these functions hehehe

def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]

def compute_rmse(var, var_hat):
    return  np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])

def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')

def mat2ten(mat, dim, mode):
    index = list()
    index.append(mode)
    for i in range(dim.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(dim[index]), order = 'F'), 0, mode)

def mvnrnd_pre(mu, Lambda):
    src = normrnd(size = (mu.shape[0],))
    return solve_ut(cholesky_upper(Lambda, overwrite_a = True, check_finite = False), 
                    src, lower = False, check_finite = False, overwrite_b = True) + mu

def cov_mat(mat, mat_bar):
    mat = mat - mat_bar
    return mat.T @ mat

def sample_hyperparameters(U, H0, beta0, d0, mu0):
    N, r = U.shape
    U_bar = np.mean(U, axis=0)
    S_bar = np.dot(U.T, U) / N
    S0_inv = np.linalg.inv(H0["S0"])
    
    beta_star = beta0 + N
    d_star = d0 + N
    mu_star = (beta0 * mu0 + N * U_bar) / beta_star
    S_star_inv = S0_inv + N * S_bar + (beta0 * N / beta_star) * np.outer((U_bar - mu0), (U_bar - mu0))
    S_star = np.linalg.inv(S_star_inv)
    
    lambda_U = wishart.rvs(df=d_star, scale=S_star)
    mu_U = multivariate_normal.rvs(mean=mu_star, cov=np.linalg.inv(beta_star * lambda_U))
    
    return {"mu": mu_U, "lambda": lambda_U}

def sample_latent_feature_vector(X, V, Hi, sigma, mu_U, lambda_U):
    N, r = V.shape
    VV = np.dot(V.T, V)
    XVT = np.dot(X, V.T)
    lambda_star = lambda_U + sigma * VV
    lambda_star_inv = np.linalg.inv(lambda_star)
    mu_star = lambda_star_inv @ (mu_U @ lambda_U + sigma * XVT)
    return multivariate_normal.rvs(mean=mu_star, cov=lambda_star_inv)

def BPMF(U0, V0, X, H0, Imax, beta0, d0, mu0, sigma):
    N, r = U0.shape
    T = V0.shape[1]
    U = U0
    V = V0
    for i in range(Imax):
        print(i)
        HU = sample_hyperparameters(U, H0, beta0, d0, mu0)
        for j in range(N):
            U[j] = sample_latent_feature_vector(X[j], V, HU, sigma, HU["mu"], HU["lambda"])
        HV = sample_hyperparameters(V.T, H0, beta0, d0, mu0)
        for j in range(T):
            V[:,j] = sample_latent_feature_vector(X[:,j], U, HV, sigma, HV["mu"], HV["lambda"])
    return U, V

# Load the .xlsx file as a pandas DataFrame
#df = pd.read_excel('C:/Users/Rohit/Documents/Summer Internship project/Exeter-Placement/Archive Data/Gen_Demand_Data_Sc3_Chausey_Scenario1.xlsx')
df = pd.read_excel('C:/Users/Rohit/Documents/Exeter-Placement/Archive Data/Gen_Demand_Data_Sc3_Chausey_Scenario1.xlsx')

X = df.values
# Convert the DataFrame to a numpy array
# Set the initial values for U0 and V0
N, T = X.shape  # Assume X is your data matrix
r = 10  # Set the rank
U0 = np.random.normal(size=(N, r))
V0 = np.random.normal(size=(r, T))

# Set the prior hyperparameters
H0 = {"mu": np.zeros(r), "lambda": np.eye(r), "S0": np.eye(r)}
beta0 = 1
d0 = r
mu0 = np.zeros(r)
sigma = 1  # Set sigma to 1, adjust this value according to your problem

# Run the Gibbs sampling algorithm
Imax = 10  # Set the number of iterations
U, V = BPMF(U0, V0, X, H0, Imax, beta0, d0, mu0, sigma)

print("yay")
# Now, U and V are the sampled latent feature matrices


#test_gaussian_prior()
print("yay")
