import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class LogReg(BaseEstimator, ClassifierMixin):
    def __init__(self, gd_type='full', tolerance=1e-4, max_iter=1000,
                 w0=None, eta=1e-2, batch_size=32, random_state=42):
        """
        gd_type: 'full' или 'stochastic'
        tolerance: порог для критерия останова по норме разности весов
        max_iter: максимальное число итераций градиентного спуска
        w0: np.array размера (d + 1) — начальные веса (w0[0] — свободный член)
        eta: длина шага (learning rate)
        batch_size: размер батча для стохастического градиентного спуска
        random_state: seed для выбора батчей
        """
        self.gd_type = gd_type
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w0 = w0
        self.eta = eta
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        """
        X: np.array размера (N, d)
        y: np.array размера (N) из 0 и 1
        ---
        output: self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.classes_ = np.array([0, 1])
        self.loss_history = []
        rng = np.random.default_rng(self.random_state)

        if X.ndim != 2 or y.shape != (len(X),) or len(X) == 0:
            raise ValueError('Нужны непустая матрица X и вектор y той же длины')
        if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
            raise ValueError('X должен быть конечным, y должен содержать только 0 и 1')
        if self.gd_type not in ('full', 'stochastic'):
            raise ValueError('Неизвестный gd_type')
        if self.eta <= 0 or self.tolerance < 0 or self.max_iter < 0:
            raise ValueError('Некорректные параметры обучения')
        if self.batch_size < 1 or int(self.batch_size) != self.batch_size:
            raise ValueError('batch_size должен быть положительным целым числом')
        X = np.column_stack([np.ones(len(X)), X])
        self.w = np.zeros(X.shape[1]) if self.w0 is None else np.array(self.w0, dtype=float, copy=True)
        if self.w.shape != (X.shape[1],) or not np.isfinite(self.w).all():
            raise ValueError('w0 должен содержать d + 1 конечных весов')
        self.n_iter_ = 0
        self.stop_reason_ = 'max_iter'
        for iteration in range(self.max_iter):
            self.loss_history.append(self.calc_loss(X, y))
            if self.gd_type == 'stochastic':
                indices = rng.choice(len(X), size=min(self.batch_size, len(X)), replace=False)
                gradient = self.calc_gradient(X[indices], y[indices])
            else:
                gradient = self.calc_gradient(X, y)
            step = self.eta * gradient
            self.w -= step
            self.n_iter_ = iteration + 1
            if np.linalg.norm(step) < self.tolerance:
                self.stop_reason_ = 'tolerance'
                break

        return self

    def sigmoid(self, z):
        return np.exp(-np.logaddexp(0, -np.asarray(z, dtype=float)))

    def predict_proba(self, X):
        """
        X: np.array размера (N, d)
        ---
        output: np.array размера (N, 2)
        """
        if getattr(self, 'w', None) is None:
            raise Exception('Not trained yet')

        X = np.asarray(X, dtype=float)
        X = np.column_stack([np.ones(len(X)), X])
        probability = self.sigmoid(X @ self.w)
        return np.column_stack([1 - probability, probability])

    def predict(self, X, threshold=0.5):
        """
        X: np.array размера (N, d)
        ---
        output: np.array размера (N) из 0 и 1
        """
        if getattr(self, 'w', None) is None:
            raise Exception('Not trained yet')

        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def calc_gradient(self, X, y):
        """
        X: np.array размера (N, d + 1) — уже с фиктивным признаком
           (N может быть равно batch_size при стохастическом спуске)
        y: np.array размера (N)
        ---
        output: np.array размера (d + 1) — градиент log-loss по self.w
        """
        return X.T @ (self.sigmoid(X @ self.w) - y) / len(y)

    def calc_loss(self, X, y):
        """
        X: np.array размера (N, d + 1) — уже с фиктивным признаком
        y: np.array размера (N)
        ---
        output: float — значение log-loss при текущих self.w
        """
        score = X @ self.w
        return float(np.mean(np.logaddexp(0, score) - y * score))
