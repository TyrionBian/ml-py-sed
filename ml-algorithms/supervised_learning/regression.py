import numpy as np
import math

class l1_regularization():
    """ Regularization for Lasso Regression """
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return self.alpha * np.linalg.norm(w)

    def grad(self, w):
        self.alpha * np.sign(w)

class l2_regularization():
    """ Regularization for Rindge Regression """
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return self.alpha * (w.T @ w)
    
    def grad(self, w):
        return self.alpha * w

class Regression(object):
    """ Base regression model. 
    Parameters:
    -----------
    n_iter: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, n_iter):
        self.n_iter = n_iter

    def initialize_weights(self, n_features):
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, size=(n_features, ))
    
    def fit(self, X:np.ndarray, y):
        X = np.insert(X, 0, 1,axis=1)
        self.training_errors = []
        self.initialize_weights(X.shape[1])

        # gradient descent for n_iterations
        for _ in range(self.n_iter):
            w_prev = np.copy(self.w)
            y_pred = X @ self.w
            mse = np.mean(0.5 * (y - y_pred)**2)
            self.training_errors.append(mse)
            grad = X.T @ (y_pred-y)
            hession = X.T @ (y_pred * (1-y_pred)) @ X
            try:
                w -= np.linalg.solve(hession, grad)
            except np.linalg.LinAlgError:
                break
            if np.allclose(w, w_prev):
                break

        self.w = w

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X @ self.w
        return y_pred

    
class LinearRegression(Regression):
    """Linear model.
    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training. If 
        false then we use batch optimization by least squares.
    """
    def __init__(self, n_iter=100, learning_rate=0.001, gradient_descent=True):
        self.gradient_descent = gradient_descent
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(LinearRegression, self).__init__(n_iter=n_iter)

    def fit(self, X, y):
        # If not gradient descent => Least squares approximation of w
        if not self.gradient_descent:
            X = np.insert(X, 0, 1, axis=1)
            