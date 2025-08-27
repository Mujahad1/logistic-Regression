import numpy as np

class Logistic_regression:
    def __init__(self, num_iters, alpha):
        self.num_iters = num_iters
        self.alpha = alpha

    def log_likelihood(self, y, preds):
        # Fixed double sum and log(0) issue
        eps = 1e-15
        preds = np.clip(preds, eps, 1 - eps)
        return np.sum(y * np.log(preds) + (1 - y) * np.log(1 - preds))

    def fit(self, x, y):
        self.weights = np.zeros(x.shape[1])
        for i in range(self.num_iters):
            preds = self.predict(x)
            dw = np.dot(x.T, (preds - y)) / y.size
            # Gradient descent update
            self.weights -= self.alpha * dw

            # Optional: print loss every 100 iterations
            if i % 100 == 0:
                loss = self.log_likelihood(y, preds)
                print(f"Iteration {i}, Loss: {loss}")

    def predict(self, x):
        # Element-wise multiplication replaced with dot product
        z = np.dot(x, self.weights)
        return 1 / (1 + np.exp(-z))

