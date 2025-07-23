import numpy as np
from activations import *
from abc import ABC, abstractmethod

class NetObject:
    @abstractmethod
    def forward(self, ):
        pass
    
    @abstractmethod
    def backward(self, ):
        pass


class LinearLayer(NetObject):
    def __init__(self, num_features: int, out_features: int):
        self.ws = np.random.randn(out_features, num_features) * 0.1 # clipping
        self.bs = np.zeros((out_features, 1)) # initialize as zeros
        
    def forward(self, x) -> np.ndarray:
        self.x = x
        x = self.ws @ x + self.bs
        return x

    def backward(self, dZ: np.array, lr: float) -> np.ndarray:
        m = self.x.shape[1]  # number of samples -> mini batch size
        # (num_classes X mb @ mb X num_features) / mb -> (num_classes X num_features)
        dW = (dZ @ self.x.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        
        dX = self.ws.T @ dZ
        
        # Gradient descent step
        self.ws -= lr * dW
        self.bs -= lr * db

        return dX
    
    def __call__(self, x:int):
        return self.forward(x)
    
 


class Network(NetObject): # single layer consists 5 neurons
    def __init__(self, layers:list, lr:float):
        self.lr = lr
        self.layers = layers

    def forward(self, x:np.array) -> np.array:
        for layer in self.layers:
            x = layer(x)

        return x
    
    def backward(self, dZ: np.array) -> np.array:
        grad = dZ
        for layer in reversed(self.layers):
            grad = layer.backward(grad, self.lr)
        return grad

    def __call__(self, x:np.array) -> np.array:
        return self.forward(x)

if __name__ == "__main__":
    x = np.array([
        [180, 170, 165],  # height
        [75,  65,  55],   # weight
        [30,  22,  25],   # age
        [80,  70,  65],   # pulse
        [12,  11,  10],   # respiration rate
        [98,  97,  99]    # body temp
    ]) # 6x3 ([:, i] must be used when indexing)
    x = x / np.max(x, axis=1, keepdims=True)  # normalize

    y_true = np.array([
        [0, 1, 1],  # male
        [1, 0, 0]   # female
    ])  # 2x3

    LR = 1e-2

    net = Network([
        LinearLayer(6, 3),
        ReLU(),
        LinearLayer(3, 2),
        Softmax()
    ], lr=LR)

    for epoch in range(1_000_000):
        y = net.forward(x)
        
        # calculate loss
        loss = cross_entropy(y, y_true)
    
        # derivative of cross-entropy
        dLoss = dce(y, y_true)
        dA1 = net.backward(dLoss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}", end="\r")

