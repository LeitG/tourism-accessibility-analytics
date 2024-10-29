import numpy as np
import numbers
from scipy.sparse import csr_matrix

from recommender import Recommender,
def get_rng(seed):
    if seed is None:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{} can not be used to create a numpy.random.RandomState'.format(seed))

def uniform(shape=None, low=0.0, high=1.0, random_state=None, dtype=np.float32):
    return get_rng(random_state).uniform(low, high, shape).astype(dtype)



EPS = 1e-10


class TriRank(Recommender):
    def __init__(
        self,
        R, X, Y,
        name="TriRank",
        alpha=1,
        beta=1,
        gamma=1,
        eta_U=1,
        eta_P=1,
        eta_A=1,
        max_iter=100,
        verbose=True,
        init_params=None,
        seed=None,
    ):
        super().__init__(name)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta_U = eta_U
        self.eta_P = eta_P
        self.eta_A = eta_A
        self.max_iter = max_iter
        self.verbose = verbose
        self.seed = seed
        self.rng = get_rng(seed)
        

        # Init params if provided
        self.init_params = {}
        self.R = self._symmetrical_normalization(csr_matrix(R))
        self.X = self._symmetrical_normalization(csr_matrix(X))
        self.Y = self._symmetrical_normalization(csr_matrix(Y))
        self.p = self.init_params.get("p", None)
        self.a = self.init_params.get("a", None)
        self.u = self.init_params.get("u", None)

        self.r_mat = csr_matrix(self.R)
        self.fit()

    def _init(self):
        if self.p is None:
            self.p = uniform(self.R.shape[1], random_state=self.rng)
        if self.a is None:
            self.a = uniform(self.Y.shape[1], random_state=self.rng)
        if self.u is None:
            self.u = uniform(self.R.shape[0], random_state=self.rng)

    def _symmetrical_normalization(self, matrix: csr_matrix):
        row = []
        col = []
        data = []
        row_norm = np.sqrt(matrix.sum(axis=1).A1)
        col_norm = np.sqrt(matrix.sum(axis=0).A1)
        for i, j in zip(*matrix.nonzero()):
            row.append(i)
            col.append(j)
            data.append(matrix[i, j] / (row_norm[i] * col_norm[j]))

        return csr_matrix((data, (row, col)), shape=matrix.shape)


    def fit(self, val_set=None):
        self._init()

        return self

    def _online_recommendation(self, user):
        # Algorithm 1: Online recommendation line 5
        p_0 = self.r_mat[[user]]
        p_0.data.fill(1)
        p_0 = p_0.toarray().squeeze()
        a_0 = self.Y[user].toarray().squeeze()
        u_0 = np.zeros(self.r_mat.shape[0])
        u_0[user] = 1

        # Algorithm 1: Online training line 6
        if p_0.any():
            p_0 /= np.linalg.norm(p_0, 1)
        if a_0.any():
            a_0 /= np.linalg.norm(a_0, 1)
        if u_0.any():
            u_0 /= np.linalg.norm(u_0, 1)

        # Algorithm 1: Online recommendation line 7
        p = self.p.copy()
        a = self.a.copy()
        u = self.u.copy()

        # Algorithm 1: Online recommendation line 8
        prev_p = p
        prev_a = a
        prev_u = u
        inc = 1
        while True:
            # eq. 4
            u_denominator = self.alpha + self.gamma + self.eta_U + EPS
            # print(self.R.shape, self.alpha.shape)
            u = (
                self.alpha / u_denominator * self.R * p
                + self.gamma / u_denominator * self.Y * a
                + self.eta_U / u_denominator * u_0
            ).squeeze()
            p_denominator = self.alpha + self.beta + self.eta_P + EPS
            p = (
                self.alpha / p_denominator * self.R.T * u
                + self.beta / p_denominator * self.X * a
                + self.eta_P / p_denominator * p_0
            ).squeeze()
            a_denominator = self.gamma + self.beta + self.eta_A + EPS
            a = (
                self.gamma / a_denominator * self.Y.T * u
                + self.beta / a_denominator * self.X.T * p
                + self.eta_P / a_denominator * a_0
            ).squeeze()

            if (self.max_iter > 0 and inc > self.max_iter) or (
                np.all(np.isclose(u, prev_u))
                and np.all(np.isclose(p, prev_p))
                and np.all(np.isclose(a, prev_a))
            ):  # stop when converged
                break
            prev_p, prev_a, prev_u = p, a, u
            inc += 1
        return p
