import numpy as np
from numpy.linalg import inv as inv
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod
from scipy.stats import multivariate_normal, wishart, matrix_normal
from numpy.linalg import solve as solve
from numpy.linalg import cholesky as cholesky_lower
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import solve_triangular as solve_ut
import scipy.io
import unittest
#stolen functions :)

def mvnrnd_pre(mu, Lambda):
    src = normrnd(size = (mu.shape[0],))
    return solve_ut(cholesky_upper(Lambda, overwrite_a = True, check_finite = False), 
                    src, lower = False, check_finite = False, overwrite_b = True) + mu

def cov_mat(mat, mat_bar):
    mat = mat - mat_bar
    return mat.T @ mat

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

import numpy as np
from scipy.stats import multivariate_normal, wishart, gamma

def is_positive_definite(A):
    print("------------------------------------------------")
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            print("Positive definite matrix AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            return True
        except np.linalg.LinAlgError:
            print("negative definite matrix ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
            return False
    else:
        return False

class BTMF:

    def __init__(self, rank, max_iter, burn_in, num_samples, time_lags):
        self.rank = rank
        self.max_iter = max_iter
        self.burn_in = burn_in
        self.num_samples = num_samples
        self.time_lags = time_lags

    def initialize_parameters(self, N, T):
        # Initialize factor matrices
        self.N=N
        self.T=T
        self.W = np.random.normal(size=(N, self.rank))
        self.X = np.random.normal(size=(T, self.rank))

        # Initialize VAR coefficient matrix
        self.A = np.random.normal(size=(self.rank, len(self.time_lags)))

        # Initialize hyperparameters
        self.mw = np.zeros(self.rank)
        self.Lw = N
        self.S = np.eye(self.rank)
        self.S += np.eye(self.S.shape[0]) * 1e-6
        is_positive_definite(self.S)
        print("init positive definite")
        self.M = np.zeros((self.rank, len(self.time_lags)))
        self.C = np.eye(len(self.time_lags))
        self.mt = np.zeros((T, self.rank))
        self.St = np.eye(self.rank)

    def draw_hyperparameters(self):
        # Calculate w_bar and Sw
        w_bar = np.mean(self.W, axis=0)  # Changed self.w to self.W
        Sw = np.sum((self.W - w_bar).T @ (self.W - w_bar), axis=0)  # Changed self.w to self.W
        #var_S = np.eye(rank) + Z_mat.T @ Z_mat - var_M.T @ var_Psi0 @ var_M
        # Update mw, Lw, and Sw
        N = self.W.shape[0]  # Changed self.w to self.W
        b0 = 1  # This is a hyperparameter that you might need to adjust
        m0 = np.zeros(self.rank)  # This is another hyperparameter
        self.mw = (b0 * m0 + N * w_bar) / (b0 + N)
        self.Lw = b0 + N + self.rank + 1
        print("before first change")
        is_positive_definite(self.S)

        self.S = np.eye(self.rank) + Sw + b0 * N / (b0 + N) * np.outer(w_bar - m0, w_bar - m0)
        print("after first change")
        is_positive_definite(self.S)
        # Draw samples from the updated distributions
        mw_sample = np.random.multivariate_normal(self.mw, np.linalg.inv(self.Lw * self.S))
        Lw_sample = scipy.stats.wishart(df=self.Lw, scale=self.S)

        return mw_sample, Lw_sample


    def draw_A_S(self):
        # Calculate Q and Z
        max_h = np.max(self.time_lags)
        Q = np.zeros((self.rank, len(self.time_lags)))
        Z = np.zeros((self.rank, self.rank))
        for t in range(max_h, self.X.shape[0]):
            Q = self.X[t - self.time_lags, :]
            Z += np.outer(self.X[t], self.X[t])

        # Update M and C
        C0_inv = np.eye(len(self.time_lags))  # This is a hyperparameter that you might need to adjust
        M0 = np.zeros((self.rank, len(self.time_lags)))  # This is another hyperparameter
        C_inv = np.linalg.inv(C0_inv + Q.T @ Q)
        self.M = C_inv @ (C0_inv @ M0 + Q.T @ Z)
        self.C = np.linalg.inv(C_inv)

        # Draw samples from the updated distributions
        A_sample = np.random.multivariate_normal(self.M.flatten(), np.kron(self.S, self.C)).reshape(self.rank, len(self.time_lags))
        S_sample = np.random.wishart(df=self.rank, scale=self.S)

        return A_sample, S_sample

    def draw_wi(self, y_i, ti):
        # Calculate mi and Li
        Li_inv = ti * self.X.T @ self.X + self.Lw
        Li = np.linalg.inv(Li_inv)
        mi = Li @ (ti * self.X.T @ y_i + self.Lw @ self.mw)

        # Draw sample from the updated distribution
        wi_sample = np.random.multivariate_normal(mi, Li)

        return wi_sample

    def draw_xt(self, t):
        # Calculate St and mt
        V_t = ...  # This should be the set of observed entries at time t
        wwi_yi_t = sum([self.ti[i] * np.outer(self.w[i], self.y[i, t]) for i in V_t])
        if 1 <= t <= self.T - np.max(self.time_lags):
            Mt = sum([self.A[k].T @ np.linalg.inv(self.S) @ self.A[k] for k in range(self.d) if self.h[k] < t + self.h[k] <= self.T])
            Nt = sum([self.A[k].T @ np.linalg.inv(self.S) @ (self.X[t + self.h[k]] - sum([self.A[l] @ self.X[t + self.h[k] - self.h[l]] for l in range(self.d) if l != k])) for k in range(self.d) if self.h[k] < t + self.h[k] <= self.T])
        else:
            Mt = 0
            Nt = 0
        Pt = np.eye(self.rank) if t in range(1, np.max(self.time_lags) + 1) else np.linalg.inv(self.S)
        Qt = np.zeros(self.rank) if t in range(1, np.max(self.time_lags) + 1) else np.linalg.inv(self.S) @ sum([self.A[l] @ self.X[t - self.h[l]] for l in range(self.d)])
        St_inv = wwi_yi_t + Mt + Pt
        St = np.linalg.inv(St_inv)
        mt = St @ (wwi_yi_t + Nt + Qt)

        # Draw sample from the updated distribution
        xt_sample = np.random.multivariate_normal(mt, St)

        return xt_sample

    def draw_ti(self, i):
        # Calculate ai and bi
        V_i = ...  # This should be the set of observed entries for the i-th time series
        ai = 0.5 * len(V_i) + self.a
        bi = 0.5 * sum([(self.y[i, t] - self.w[i].T @ self.X[t])**2 for t in V_i]) + self.b

        # Draw sample from the updated distribution
        ti_sample = np.random.gamma(ai, 1 / bi)  # Note: numpy's gamma function uses a scale parameter, which is the inverse of the rate parameter

        return ti_sample
    
    def fit(self):
        Y_samples = []
        for iter in range(self.max_iter):
            self.mw, self.Lw = self.draw_hyperparameters()
            for i in range(self.N):
                self.W[i] = self.draw_wi(self.Y[i, :], self.ti[i])
            self.A, self.S = self.draw_A_S()
            is_positive_definite(self.S)
            print("first fit")
            for t in range(self.T):
                self.X[t] = self.draw_xt(t)
            for i in range(self.N):
                self.ti[i] = self.draw_ti(i)
            if iter >= self.m1:
                Y_samples.append(self.W @ self.X.T)
        Y_hat = np.mean(Y_samples, axis=0)
        return Y_hat

mat = scipy.io.loadmat('C:/Users/Rohit/Documents/Exeter-Placement/transdim-master/datasets/Guangzhou-data-set/tensor.mat')

# Convert the tensor to a numpy array
tensor = np.array(mat['tensor'])

# Flatten the tensor to a 2D array (matrix)
Y = tensor.reshape(tensor.shape[0], -1)

# Define the time lag set, the number of burn-in iterations, the number of samples, and the rank
time_lags = np.array([1, 2, 3])
m1 = 1000
m2 = 200
rank = 10

# Create an instance of the BTMF class and fit the model
#model = BTMF(Y, time_lags, m1, m2, rank)
model = BTMF(rank, m1, m2, 10, time_lags)

N, T = Y.shape

# Initialize the parameters
model.initialize_parameters(N, T)
w_sample, Lw_sample = model.draw_hyperparameters()
"""
class TestBTMF(unittest.TestCase):
    def setUp(self):
        self.model = BTMF(rank=10, max_iter=1000, burn_in=200, num_samples=10, time_lags=np.array([1, 2, 3]))
        self.model.initialize_parameters(N=100, T=1000)

    def test_draw_hyperparameters(self):
        mw_sample, Lw_sample = self.model.draw_hyperparameters()

        # Check that mw_sample and Lw_sample are numpy arrays
        self.assertIsInstance(mw_sample, np.ndarray)
        self.assertIsInstance(Lw_sample, np.ndarray)

        # Check that mw_sample and Lw_sample have the correct shape
        self.assertEqual(mw_sample.shape, (self.model.rank,))
        self.assertEqual(Lw_sample.shape, (self.model.rank, self.model.rank))

        # Check that Lw_sample is symmetric
        self.assertTrue(np.allclose(Lw_sample, Lw_sample.T))

        # Check that Lw_sample is positive definite
        self.assertTrue(np.all(np.linalg.eigvals(Lw_sample) > 0))

        # Check that Lw_sample has degrees of freedom greater than or equal to the dimension of the scale matrix
        self.assertTrue(self.model.Lw >= self.model.rank)

if __name__ == '__main__':
    unittest.main()
"""
# Fit the model
#Y_hat = model.fit()

# Print the estimated data matrix
#print(Y_hat)

