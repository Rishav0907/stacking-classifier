import numpy as np
import cvxpy as cp


class SVM:
    def __init__(self, C=1):
        self.C = C

    def _kernel_matrix(self, X, Z):
        return X @ Z.T

    def train(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        n_samples = X_train.shape[0]
        y_train = y_train.astype(np.double).reshape(-1, 1)
        y_train = np.squeeze(y_train)

        # Compute the kernel matrix
        K = self._kernel_matrix(X_train, X_train)
        # Add small constant to diagonal for numerical stability
        K += 1e-8 * np.eye(n_samples)

        # Variables
        nu = cp.Variable(n_samples)

        # Objective function
        objective = cp.Maximize(
            cp.sum(nu) - 0.5 * cp.quad_form(cp.multiply(y_train, nu), K))

        # Constraints
        constraints = [nu >= 0, nu <= self.C,
                       cp.sum(cp.multiply(nu, y_train)) == 0]

        # Problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, max_iters=10000)

        nu_values = np.array(nu.value).flatten()
        self.support_vectors = np.where(nu_values > 1e-5)[0]
        self.nu = nu_values[self.support_vectors]
        self.support_vectors_X = X_train[self.support_vectors]
        self.support_vectors_y = y_train[self.support_vectors]

        # Compute the bias term (b)
        self.bias = np.mean(
            self.support_vectors_y - np.sum(
                (self.nu * self.support_vectors_y)[:, None] * K[np.ix_(self.support_vectors, self.support_vectors)], axis=0
            )
        )

    def test(self, X):
        X = np.array(X)
        K = self._kernel_matrix(X, self.support_vectors_X)
        return np.sign(np.sum((self.nu * self.support_vectors_y) * K, axis=1) + self.bias)
