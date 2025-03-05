import numpy as np
from typing import *
from abc import ABC, abstractmethod
from itertools import pairwise

from layers import *
from optimizers import *
from collections import OrderedDict

np.random.seed(16)

class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, epochs_print: int) -> List[float]:
        raise NotImplementedError
    
    @abstractmethod
    def update(self) -> None:
        raise NotImplementedError
    
class MLP(Model):
    def __init__(self, input_size: int, hidden_size: List[int], output_size: int, 
                 lr: float=0.01, optimizer: Optimizer=SGD, layer: Layer=Affine, activation: Layer=Sigmoid):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lr = lr
        self.optimizer = optimizer(lr)
        self.layer = layer
        self.activation = activation

        self.params = {}
        for i, (x, y) in enumerate(pairwise([input_size] + hidden_size + [output_size]), 1):
            self.params['W'+str(i)] = np.random.randn(x, y)
            self.params['b'+str(i)] = np.zeros(y)
        
        self.layers = OrderedDict()
        for i in range(1, len(hidden_size) + 2):
            self.layers[f'Layer{i}'] = self.layer(self.params[f'W{i}'], self.params[f'b{i}'])
            self.layers[f'Activation{i}'] = activation()

        for k, v in self.layers.items():
            print(f"{k} : {v}")
        print()

    def forward(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        self.pred_y = x
        return x
    
    def backward(self, y):
        dout = y
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)
        return dout
    
    def predict(self, x):
        return self.forward(x)
    
    def train(self, x, y, epochs=10000, epochs_print=500):
        loss_list = []
        for idx in range(1, epochs + 1):
            pred_y = self.forward(x)
            loss = np.mean((pred_y - y) ** 2) # MSE
            # dout = (pred_y - y) # derivative of MSE, 用 Sigmoid 的時候可以 train 比較快
            dout = 2 * (pred_y - y) / y.shape[0] # derivative of MSE
            self.backward(dout)
            self.update()
            if idx % epochs_print == 0:
                print(f'epoch {idx:5d} loss : {loss}')
            loss_list.append(loss)
        return loss_list
    
    def update(self):
        grads = {}
        layers = [layer for layer in self.layers.values() if isinstance(layer, self.layer)]
        for i, layer in enumerate(layers, 1):
            grads['W'+str(i)] = layer.dW
            grads['b'+str(i)] = layer.db
        self.params = self.optimizer.update(self.params, grads)

class MLP_wo_activation(Model):
    def __init__(self, input_size: int, hidden_size: List[int], output_size: int, 
                 lr: float=0.01, optimizer: Optimizer=SGD, layer: Layer=Affine, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lr = lr
        self.optimizer = optimizer(lr)
        self.layer = layer

        self.params = {}
        for i, (x, y) in enumerate(pairwise([input_size] + hidden_size + [output_size]), 1):
            self.params['W'+str(i)] = np.random.randn(x, y)
            self.params['b'+str(i)] = np.zeros(y)

        self.layers = OrderedDict()
        for i in range(1, len(hidden_size) + 2):
            self.layers[f'Layer{i}'] = self.layer(self.params[f'W{i}'], self.params[f'b{i}'])

        for k, v in self.layers.items():
            print(f"{k} : {v}")
        print()

    def forward(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        self.pred_y = x
        return x

    def backward(self, y):
        dout = y
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)
        return dout

    def predict(self, x):
        return self.forward(x)

    def train(self, x, y, epochs=10000, epochs_print=500):
        loss_list = []
        for idx in range(1, epochs + 1):
            pred_y = self.forward(x)
            loss = np.mean((pred_y - y) ** 2) # MSE
            dout = 2 * (pred_y - y) / y.shape[0] # derivative of MSE
            self.backward(dout)
            self.update()
            if idx % epochs_print == 0:
                print(f'epoch {idx:5d} loss : {loss}')
            loss_list.append(loss)
        return loss_list

    def update(self):
        grads = {}
        layers = [layer for layer in self.layers.values() if isinstance(layer, self.layer)]
        for i, layer in enumerate(layers, 1):
            grads['W'+str(i)] = layer.dW
            grads['b'+str(i)] = layer.db
        self.params = self.optimizer.update(self.params, grads)