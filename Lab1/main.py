import numpy as np
from typing import *

from layers import *
from utils import *
from optimizers import *
from network import *

def train_and_test(model: Model, x: np.ndarray, y: np.ndarray, epochs: int=10000, epochs_print: int=500, plot: bool=True) -> None:
    losses = model.train(x, y, epochs, epochs_print)
    
    pred_y = model.predict(x)
    pred_y_label = np.where(pred_y > 0.5, 1, 0)
    accuracy = np.mean(y == pred_y_label)

    for i, (y_i, pred_y_i) in enumerate(zip(y, pred_y_label)):
        print(f"Iter: {i} |\t Ground truth: {y_i} |\t Predict: {pred_y_i}")
    print(f"loss={losses[-1]:.5f} accuracy={accuracy*100:.2f}%", end="\n\n")

    if plot:
        show_loss(losses)
        show_result(x, y, pred_y_label)

def run(model: Model, hiddens: List[int]=[10, 10], 
        lr: float=0.1, optimizer: Optimizer=SGD, activation: Layer=Sigmoid,
        epochs: int=10000, epochs_print: int=500) -> None:
    
    np.random.seed(16) # Avoid generating different dataset

    print("Linear dataset: ", end="\n\n")
    x, y = generate_linear(n = 100)
    model_linear = model(input_size = 2, hidden_size = hiddens, output_size = 1,
                       lr=lr, optimizer=optimizer, layer=Affine, activation=activation)
    train_and_test(model_linear, x, y, epochs, epochs_print)

    print("XOR dataset:", end="\n\n")
    x, y = generate_XOR_easy()
    model_xor = model(input_size = 2, hidden_size = hiddens, output_size = 1,
                    lr=lr, optimizer=optimizer, layer=Affine, activation=activation)
    train_and_test(model_xor, x, y, epochs, epochs_print)

if __name__ == "__main__":
    show_data()

    model = MLP(input_size=2, hidden_size=[10, 10], output_size=1,
                lr=0.01, optimizer=SGD, activation=Sigmoid)
    """
        run basic model with default parameters.
        - hidden layer size = [10, 10]
        - learning rate = 0.01
        - activation function = Sigmoid
        - optimizer = SGD
        - epochs = 20000
    """
    # run(model=MLP, hiddens=[10, 10], lr=0.1, optimizer=SGD, activation=Sigmoid, epochs=20000, epochs_print=1000)
    # run(model=MLP, hiddens=[10, 10], lr=0.01, optimizer=SGD, activation=ReLU, epochs=10000, epochs_print=500)
    """
        try different activation function

        - RuLU can still 
    """
    # run(model=MLP, hiddens=[10, 10], lr=0.01, activation=ReLU) # Linear: 100%, XOR: 100%
    # run(model=MLP, hiddens=[10, 10], lr=0.01, activation=Tanh) # Linear: 98%, XOR: 100%
    # run(model=MLP, hiddens=[10, 10], lr=0.01, activation=Tanh, epochs=20000, epochs_print=1000) # Linear: 99%, XOR: 100%
    """
        try different learning rate
    """
    # run(model=MLP, hiddens=[10, 10], lr=1, activation=Sigmoid) # Linear dataset fails
    # run(model=MLP, hiddens=[10, 10], lr=0.1, activation=Sigmoid) 
    # run(model=MLP, hiddens=[10, 10], lr=0.01, activation=Sigmoid) # loss curve is more smooth than lr=0.1
    # run(model=MLP, hiddens=[10, 10], lr=0.001, activation=Sigmoid) # XOR dataset fails
    """
        try different hidden layer size
        when run same epoch, the model with more hidden layer size has better accuracy.
    """
    # run(model=MLP, hiddens=[20, 20], lr=0.01, activation=Sigmoid) # both works
    # run(model=MLP, hiddens=[5, 5], lr=0.01, activation=Sigmoid) # accuracy of XOR dataset is down to 90.48%
    # run(model=MLP, hiddens=[2, 2], lr=0.01, activation=Sigmoid) # accuracy of XOR dataset is down to 76.19%
    # run(model=MLP, hiddens=[5, 5], lr=0.01, activation=Sigmoid, epochs=50000, epochs_print=1000) # both works
    # run(model=MLP, hiddens=[2, 2], lr=0.01, activation=Sigmoid, epochs=50000, epochs_print=1000) # both works
    """
        try different optimizer
    """
    # run(model=MLP, hiddens=[10, 10], lr=0.01, optimizer=AdaGrad, activation=Sigmoid) # accuracy of Linear dataset is down to 99.00%
    # run(model=MLP, hiddens=[10, 10], lr=0.1, optimizer=AdaGrad, activation=Sigmoid)
    """
        try without activation function

        accuracy of Linear dataset is down to 99.00%
        accuracy of XOR is down to 52.38%
    """
    # run(model=MLP_wo_activation, hiddens=[5, 5], lr=0.01, optimizer=SGD)
    # run(model=MLP_wo_activation, hiddens=[2, 2], lr=0.01, optimizer=SGD)
