import numpy as np

class PolySVM:
    def __init__(
        self,
        degree=2,
        coef0=1,
        C=1.0,
        gamma=1.0,
        lr=0.001,
        n_iters=1000,
        loss_weight=False,
        pos_weight=5.0
    ):
        self.degree = degree
        self.coef0 = coef0
        self.C = C
        self.gamma = gamma
        self.lr = lr
        self.n_iters = n_iters
        self.loss_weight = loss_weight
        self.pos_weight = pos_weight

    def _poly_kernel(self, X1, X2):
        return (self.gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree

    def fit(self, X, y):
        n_samples = X.shape[0]
        y = np.where(y <= 0, -1, 1)

        self.X = X
        self.y = y
        self.alpha = np.zeros(n_samples)

        K = self._poly_kernel(X, X)

        for _ in range(self.n_iters):
            for i in range(n_samples):
                margin = y[i] * np.sum(self.alpha * y * K[:, i])
                if margin < 1:
                    weight = self.pos_weight if (self.loss_weight and y[i] == 1) else 1.0
                    self.alpha[i] += self.lr * weight
                self.alpha[i] = min(self.alpha[i], self.C)

    def prune_support_vectors(self, threshold=1e-3):
        keep = np.where(np.abs(self.alpha) > threshold)[0]
        self.X = self.X[keep]
        self.y = self.y[keep]
        self.alpha = self.alpha[keep]

    def project(self, X):
        K = self._poly_kernel(X, self.X)
        return np.dot(K, self.alpha * self.y)

    def decision_function(self, X):
        return self.project(X)

    def predict(self, X):
        return np.where(self.project(X) >= 0, 1, 0)
