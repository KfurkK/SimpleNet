import numpy as np
from DNN import NetObject
def sigmoid(x: np.array) -> np.array:
    return 1 / (1 + np.exp(-x))

def dsigmoid(x: np.array) -> np.array:
    s = sigmoid(x)
    return s * (1-s)

def softmax(x: np.array, axis=0) -> np.array:
    exps = np.exp(x - np.max(x, axis=axis, keepdims=True))  # axis=1 for row-wise softmax
    return exps / np.sum(exps, axis=axis, keepdims=True)

def mse(y_pred: np.array, y_true: np.array) -> np.array:
    return np.mean((y_true - y_pred)**2)

def dmse(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size

def cross_entropy(y_pred, y_true):
    eps = 1e-12
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=0))

def dce(y_pred, y_true):
    """CE immediately after softmax
    dz/dL = 1/m (y* - y_true)
    """
    return (y_pred - y_true)


class ReLU(NetObject):
    def forward(self, x):
        self.b = x > 0
        return x * self.b

    def backward(self, grad, lr:None):
        return grad*self.b
    
    def __call__(self, x):
        return self.forward(x)
    
class Softmax(NetObject):
    def __init__(self, axis: int = 0):
        self.axis = axis

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = self._stable_softmax(x)
        return self.y

    def backward(self, dY: np.ndarray, lr: float = 0.0) -> np.ndarray:
        y   = self.y
        axis = self.axis


        dot = np.sum(dY * y, axis=axis, keepdims=True)
        dX  = y * (dY - dot)
        return dX

    def _stable_softmax(self, z: np.ndarray) -> np.ndarray:
        z = z - np.max(z, axis=self.axis, keepdims=True)
        ez = np.exp(z)
        return ez / np.sum(ez, axis=self.axis, keepdims=True)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)    