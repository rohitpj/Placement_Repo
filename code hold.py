"""
class BTMF:
    def __init__(self, rank, max_iter):
        self.rank = rank
        self.max_iter = max_iter

    def fit(self, Y, V):
        N=Y.size[0]
        T=Y.size[1]
        self.w = np.random.normal(size=(N, self.rank))

        # Initialize temporal factors
        self.x = np.random.normal(size=(T, self.rank))

        # Initialize covariance matrix
        self.S = np.eye(self.rank)

        # Initialize factor loading matrix
        self.A = np.random.normal(size=(self.rank, self.d))

        # Initialize hyperparameters
        self.m0 = np.zeros(self.rank)
        self.W0 = np.eye(self.rank)
        self.n0 = self.rank
        self.b0 = 1
        self.a = 10
        self.b = 10
        self.M0 = np.zeros((self.rank, self.d))
        self.C0 = np.eye(self.d)
        self.S0 = np.eye(self.rank)
        self.mw = np.zeros(self.rank)
        self.Lw = np.eye(self.rank)

        # Initialize n
        self.N = self.rank

        # Initialize S
        self.S = np.eye(self.rank)

        # Initialize M and C
        self.M = np.zeros((self.rank, self.d))
        self.C = np.eye(self.d)

        # Initialize mt and St
        self.mt = np.zeros((T, self.rank))
        self.St = np.eye(self.rank)

        # Initialize spatial factors
        self.W = np.random.normal(size=(N, self.rank))

        # Initialize temporal factors
        self.X = np.random.normal(size=(T, self.rank))

        #begin sampling
        for _ in range(self.max_iter):
            # Draw w_i
            for i in range(N):
                self.w[i] = multivariate_normal.rvs(mean=self.mw, cov=np.linalg.inv(self.Lw))

            # Draw S and A
            self.S = wishart.rvs(df=self.n, scale=self.S)
            self.A = matrix_normal.rvs(mean=self.M, rowcov=self.C, colcov=self.S)

            # Draw x_t
            for t in range(T):
                self.x[t] = multivariate_normal.rvs(mean=self.mt, cov=self.St)


    def impute(self):
        # Use model to impute missing values
        # ...
        pass
"""

        print(isinstance(self.mw, np.ndarray) and self.mw.shape == (self.rank,))

        # Test if Lw is a scalar
        print(isinstance(self.Lw, (int, float, np.integer, float)))

        # Test if S is a numpy array of shape (rank, rank)
        print(isinstance(self.S, np.ndarray) and self.S.shape == (self.rank, self.rank))

        # Test if M is a numpy array of shape (rank, len(time_lags))
        print(isinstance(self.M, np.ndarray) and self.M.shape == (self.rank, len(self.time_lags)))

        # Test if C is a numpy array of shape (len(time_lags), len(time_lags))
        print(isinstance(self.C, np.ndarray) and self.C.shape == (len(self.time_lags), len(self.time_lags)))

        # Test if mt is a numpy array of shape (T, rank)
        print(isinstance(self.mt, np.ndarray) and self.mt.shape == (self.T, self.rank))

        # Test if St is a numpy array of shape (rank, rank)
        print(isinstance(self.St, np.ndarray) and self.St.shape == (self.rank, self.rank))

        # Test if W is a numpy array of shape (N, rank)
        print(isinstance(self.W, np.ndarray) and self.W.shape == (self.N, self.rank))

        # Test if X is a numpy array of shape (T, rank)
        print(isinstance(self.X, np.ndarray) and self.X.shape == (self.T, self.rank))

