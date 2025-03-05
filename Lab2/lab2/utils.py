import os
import pickle
from matplotlib import pyplot as plt

if not os.path.exists('./results/'):
    os.makedirs('./results/')

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_losses(losses, title="Loss History"):
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    # plt.show()
    plt.savefig(f'./results/{title}.png')

def show_result(task):
    modes = ["sd", "loso", "losoft"]
    losses = []
    for mode in modes:
        losses.append(load_pickle(f"./model_weight/{task}/{mode}.pt.pkl"))

    plt.figure(figsize=(15, 5))
    plt.suptitle(f'{task}')
    for i, mode in enumerate(modes, 1):
        plt.subplot(1, 3, i)
        plt.plot(losses[i-1], label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{mode}')
        plt.legend()
    # plt.show()
    plt.savefig(f'./results/{task}.png')

if __name__ == '__main__':
    tasks = [
            "Nu22_lr0.01_drop0.5_woDA", # 60.07%/54.86%/76.39% -> basic
            "Nu44_lr0.002_drop0.7_woDA", # 63.11%/60.76%/74.31% -> SD 2nd, LOSO 1st
            "Nu44_lr0.001_drop0.8_wDA", # 63.76%/60.76%/77.78% -> SD 1st, LOSO 1st, FT 1st
            "Nu22_lr0.01_drop0.5_wDA", # 55.21%/57.99%/77.43% -> FT 2nd
        ]
    for task in tasks:
        show_result(task)

    losses = load_pickle("./model_weight/Final/losoft_80.21_Nu44_lr0.001_drop0.8_wDA_Step500.pt.pkl")
    plot_losses(losses, title="LOSO+FT 80.21% Nu44 lr0.001 drop0.8 wDA Step500")