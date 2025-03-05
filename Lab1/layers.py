import numpy as np
from typing import *
from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.mask = None

    def __str__(self) -> str:
        return "ReLU"

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout[self.mask] = 0 # set non-positive elements to 0
        dx = dout
        return dx

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.out = None

    def __str__(self) -> str:
        return "Sigmoid"
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        out = 1.0 / (1.0 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = dout * (1.0 - self.out) * self.out
        return dx

class Tanh(Layer):
    def __init__(self):
        super().__init__()
        self.out = None

    def __str__(self) -> str:
        return "Tanh"
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.tanh(x)
        self.out = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = dout * (1 - self.out ** 2)
        return dx

class Linear(Layer):
    def __init__(self, W: np.ndarray):
        super().__init__()
        self.W = W
        self.x = None
        self.dW = None

    def __str__(self) -> str:
        return "Linear"
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        out = np.dot(x, self.W)
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        return dx
    
class Affine(Layer):
    def __init__(self, W: np.ndarray, b: np.ndarray):
        super().__init__()
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def __str__(self) -> str:
        return "Affine"
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
    
class SoftmaxWithMSE(Layer):
    def __init__(self):
        super().__init__()
        self.loss = None
        self.y = None
        self.pred_y = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.pred_y = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.pred_y

    def backward(self, y: np.ndarray) -> np.ndarray:
        self.y = y
        self.loss = -np.sum(y * np.log(self.pred_y + 1e-7)) / y.shape[0]
        dx = (self.pred_y - y) / y.shape[0]
        return dx
