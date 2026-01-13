import numpy as np

class StandardScalerScratch:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class KNNRegressorScratch:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        predictions = np.zeros(len(X))

        for i, x in enumerate(X):
            distances = np.zeros(len(self.X_train))

            for j, x_train in enumerate(self.X_train):
                distances[j] = self._euclidean_distance(x, x_train)

            k_indices = np.argsort(distances)[:self.k]
            predictions[i] = np.mean(self.y_train[k_indices])

        return predictions