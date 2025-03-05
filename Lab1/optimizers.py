import numpy as np
from typing import *
from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, lr: float=0.1):
        self.lr = lr
        pass

    @abstractmethod
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr: float=0.01):
        super().__init__()
        self.lr = lr

    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
        return params
    
class AdaGrad(Optimizer):
    def __init__(self, lr: float=0.01):
        super().__init__()
        self.lr = lr
        self.eps = 1e-7 # term added to the denominator to improve numerical stability
        self.h = None

    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] += grads[key] ** 2
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + self.eps)
        return params