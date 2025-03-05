import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.SCCNet import SCCNet
from Dataloader import MIBCI2aDataset


def sccnet_test(batch_size=32, mode="sd_test",
                model_path='./model_weight/model.pt',
                paras={}):

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} not found")

    test_dataset = MIBCI2aDataset(mode=mode)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SCCNet(**paras)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval() # set model to evaluation mode

    tot = cnt = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # disable gradient calculation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, -1)

            tot += labels.size()[0] # batch size
            cnt += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = cnt / tot * 100

    return acc


if __name__ == '__main__':
    tasks = {
        # Nu22_lr0.01_drop0.5_woDA: 60.07%/54.86%/76.39%
        "SD_Nu22_lr0.01_drop0.5_woDA": {
            "model_path": "./model_weight/Nu22_lr0.01_drop0.5_woDA/sd_final.pt",
            "mode": "sd_test",
            "paras": {"dropoutRate": 0.5, "Nu": 22},
        },
        "LOSO_Nu22_lr0.01_drop0.5_woDA": {
            "model_path": "./model_weight/Nu22_lr0.01_drop0.5_woDA/loso_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.5, "Nu": 22},
        },
        "LOSO+FT_Nu22_lr0.01_drop0.5_woDA": {
            "model_path": "./model_weight/Nu22_lr0.01_drop0.5_woDA/losoft_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.5, "Nu": 22},
        },
        # Nu22_lr0.01_drop0.5_wDA: 55.21%/57.99%/77.43%
        "SD_Nu22_lr0.01_drop0.5_wDA": {
            "model_path": "./model_weight/Nu22_lr0.01_drop0.5_wDA/sd_final.pt",
            "mode": "sd_test",
            "paras": {"dropoutRate": 0.5, "Nu": 22},
        },
        "LOSO_Nu22_lr0.01_drop0.5_wDA": {
            "model_path": "./model_weight/Nu22_lr0.01_drop0.5_wDA/loso_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.5, "Nu": 22},
        },
        "LOSO+FT_Nu22_lr0.01_drop0.5_wDA": {
            "model_path": "./model_weight/Nu22_lr0.01_drop0.5_wDA/losoft_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.5, "Nu": 22},
        },
        # Nu22_lr0.002_drop0.7_woDA: 63.02%/57.99%/73.61%
        "SD_Nu22_lr0.002_drop0.7_woDA": {
            "model_path": "./model_weight/Nu22_lr0.002_drop0.7_woDA/sd_final.pt",
            "mode": "sd_test",
            "paras": {"dropoutRate": 0.7, "Nu": 22},
        },
        "LOSO_Nu22_lr0.002_drop0.7_woDA": {
            "model_path": "./model_weight/Nu22_lr0.002_drop0.7_woDA/loso_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.7, "Nu": 22},
        },
        "LOSO+FT_Nu22_lr0.002_drop0.7_woDA": {
            "model_path": "./model_weight/Nu22_lr0.002_drop0.7_woDA/losoft_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.7, "Nu": 22},
        },
        # Nu22_lr0.002_drop0.8_woDA: 62.28%/59.03%/75.69%
        "SD_Nu22_lr0.002_drop0.8_woDA": {
            "model_path": "./model_weight/Nu22_lr0.002_drop0.8_woDA/sd_final.pt",
            "mode": "sd_test",
            "paras": {"dropoutRate": 0.8, "Nu": 22},
        },
        "LOSO_Nu22_lr0.002_drop0.8_woDA": {
            "model_path": "./model_weight/Nu22_lr0.002_drop0.8_woDA/loso_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.8, "Nu": 22},
        },
        "LOSO+FT_Nu22_lr0.002_drop0.8_woDA": {
            "model_path": "./model_weight/Nu22_lr0.002_drop0.8_woDA/losoft_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.8, "Nu": 22},
        },
        #Nu22_lr0.002_drop0.9_woDA: 62.67%/57.64%/74.31%
        "SD_Nu22_lr0.002_drop0.9_woDA": {
            "model_path": "./model_weight/Nu22_lr0.002_drop0.9_woDA/sd_final.pt",
            "mode": "sd_test",
            "paras": {"dropoutRate": 0.9, "Nu": 22},
        },
        "LOSO_Nu22_lr0.002_drop0.9_woDA": {
            "model_path": "./model_weight/Nu22_lr0.002_drop0.9_woDA/loso_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.9, "Nu": 22},
        },
        "LOSO+FT_Nu22_lr0.002_drop0.9_woDA": {
            "model_path": "./model_weight/Nu22_lr0.002_drop0.9_woDA/losoft_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.9, "Nu": 22},
        },
        # Nu44_lr0.01_drop0.7_wDA: 61.41%/59.72%/73.96%
        "SD_Nu44_lr0.01_drop0.7_wDA": {
            "model_path": "./model_weight/Nu44_lr0.01_drop0.7_wDA/sd_final.pt",
            "mode": "sd_test",
            "paras": {"dropoutRate": 0.7, "Nu": 44},
        },
        "LOSO_Nu44_lr0.01_drop0.7_wDA": {
            "model_path": "./model_weight/Nu44_lr0.01_drop0.7_wDA/loso_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.7, "Nu": 44},
        },
        "LOSO+FT_Nu44_lr0.01_drop0.7_wDA": {
            "model_path": "./model_weight/Nu44_lr0.01_drop0.7_wDA/losoft_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.7, "Nu": 44},
        },
        # Nu44_lr0.001_drop0.7_woDA: 62.15%/59.03%/71.88%
        "SD_Nu44_lr0.001_drop0.7_woDA": {
            "model_path": "./model_weight/Nu44_lr0.001_drop0.7_woDA/sd_final.pt",
            "mode": "sd_test",
            "paras": {"dropoutRate": 0.7, "Nu": 44},
        },
        "LOSO_Nu44_lr0.001_drop0.7_woDA": {
            "model_path": "./model_weight/Nu44_lr0.001_drop0.7_woDA/loso_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.7, "Nu": 44},
        },
        "LOSO+FT_Nu44_lr0.001_drop0.7_woDA": {
            "model_path": "./model_weight/Nu44_lr0.001_drop0.7_woDA/losoft_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.7, "Nu": 44},
        },
        # Nu44_lr0.001_drop0.7_wDA: 63.98%/60.07%/73.61%
        "SD_Nu44_lr0.001_drop0.7_wDA": {
            "model_path": "./model_weight/Nu44_lr0.001_drop0.7_wDA/sd_final.pt",
            "mode": "sd_test",
            "paras": {"dropoutRate": 0.7, "Nu": 44},
        },
        "LOSO_Nu44_lr0.001_drop0.7_wDA": {
            "model_path": "./model_weight/Nu44_lr0.001_drop0.7_wDA/loso_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.7, "Nu": 44},
        },
        "LOSO+FT_Nu44_lr0.001_drop0.7_wDA": {
            "model_path": "./model_weight/Nu44_lr0.001_drop0.7_wDA/losoft_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.7, "Nu": 44},
        },
        # Nu44_lr0.001_drop0.8_wDA: 63.76%/60.76%/77.78%
        "SD_Nu44_lr0.001_drop0.8_wDA": {
            "model_path": "./model_weight/Nu44_lr0.001_drop0.8_wDA/sd_final.pt",
            "mode": "sd_test",
            "paras": {"dropoutRate": 0.8, "Nu": 44},
        },
        "LOSO_Nu44_lr0.001_drop0.8_wDA": {
            "model_path": "./model_weight/Nu44_lr0.001_drop0.8_wDA/loso_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.8, "Nu": 44},
        },
        "LOSO+FT_Nu44_lr0.001_drop0.8_wDA": {
            "model_path": "./model_weight/Nu44_lr0.001_drop0.8_wDA/losoft_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.8, "Nu": 44},
        },
        # Nu44_lr0.001_drop0.75_wDA: 62.89%/60.07%/75.00%
        "SD_Nu44_lr0.001_drop0.75_wDA": {
            "model_path": "./model_weight/Nu44_lr0.001_drop0.75_wDA/sd_final.pt",
            "mode": "sd_test",
            "paras": {"dropoutRate": 0.75, "Nu": 44},
        },
        "LOSO_Nu44_lr0.001_drop0.75_wDA": {
            "model_path": "./model_weight/Nu44_lr0.001_drop0.75_wDA/loso_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.75, "Nu": 44},
        },
        "LOSO+FT_Nu44_lr0.001_drop0.75_wDA": {
            "model_path": "./model_weight/Nu44_lr0.001_drop0.75_wDA/losoft_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.75, "Nu": 44},
        },
        # Nu44_lr0.002_drop0.7_woDA: 63.11%/60.76%/74.31%
        "SD_Nu44_lr0.002_drop0.7_woDA": {
            "model_path": "./model_weight/Nu44_lr0.002_drop0.7_woDA/sd_final.pt",
            "mode": "sd_test",
            "paras": {"dropoutRate": 0.7, "Nu": 44},
        },
        "LOSO_Nu44_lr0.002_drop0.7_woDA": {
            "model_path": "./model_weight/Nu44_lr0.002_drop0.7_woDA/loso_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.7, "Nu": 44},
        },
        "LOSO+FT_Nu44_lr0.002_drop0.7_woDA": {
            "model_path": "./model_weight/Nu44_lr0.002_drop0.7_woDA/losoft_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.7, "Nu": 44},
        },
        # Nu44_lr0.002_drop0.7_wDA: 63.15%/57.29%/75.00%
        "SD_Nu44_lr0.002_drop0.7_wDA": {
            "model_path": "./model_weight/Nu44_lr0.002_drop0.7_wDA/sd_final.pt",
            "mode": "sd_test",
            "paras": {"dropoutRate": 0.7, "Nu": 44},
        },
        "LOSO_Nu44_lr0.002_drop0.7_wDA": {
            "model_path": "./model_weight/Nu44_lr0.002_drop0.7_wDA/loso_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.7, "Nu": 44},
        },
        "LOSO+FT_Nu44_lr0.002_drop0.7_wDA": {
            "model_path": "./model_weight/Nu44_lr0.002_drop0.7_wDA/losoft_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.7, "Nu": 44},
        },
        # Nu44_lr0.002_drop0.8_woDA: 63.32%/58.68%/74.65%
        "SD_Nu44_lr0.002_drop0.8_woDA": {
            "model_path": "./model_weight/Nu44_lr0.002_drop0.8_woDA/sd_final.pt",
            "mode": "sd_test",
            "paras": {"dropoutRate": 0.8, "Nu": 44},
        },
        "LOSO_Nu44_lr0.002_drop0.8_woDA": {
            "model_path": "./model_weight/Nu44_lr0.002_drop0.8_woDA/loso_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.8, "Nu": 44},
        },
        "LOSO+FT_Nu44_lr0.002_drop0.8_woDA": {
            "model_path": "./model_weight/Nu44_lr0.002_drop0.8_woDA/losoft_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.8, "Nu": 44},
        },
        # Nu44_lr0.002_drop0.8_wDA: 60.07%/58.68%/73.26%
        "SD_Nu44_lr0.002_drop0.8_wDA": {
            "model_path": "./model_weight/Nu44_lr0.002_drop0.8_wDA/sd_final.pt",
            "mode": "sd_test",
            "paras": {"dropoutRate": 0.8, "Nu": 44},
        },
        "LOSO_Nu44_lr0.002_drop0.8_wDA": {
            "model_path": "./model_weight/Nu44_lr0.002_drop0.8_wDA/loso_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.8, "Nu": 44},
        },
        "LOSO+FT_Nu44_lr0.002_drop0.8_wDA": {
            "model_path": "./model_weight/Nu44_lr0.002_drop0.8_wDA/losoft_final.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.8, "Nu": 44},
        },
    }

    final_tasks = {
        "SD_63.11_Nu44_lr0.002_drop0.7_woDA": { # 63.11%
            # "model_path": "./model_weight/Nu44_lr0.002_drop0.7_woDA/sd_final.pt",
            "model_path": "./model_weight/Final/sd_63.11_Nu44_lr0.002_drop0.7_woDA.pt",
            "mode": "sd_test",
            "paras": {"dropoutRate": 0.7, "Nu": 44},
        },
        "SD_63.76_Nu44_lr0.001_drop0.8_wDA": { # 63.76%
            # "model_path": "./model_weight/Nu44_lr0.001_drop0.8_wDA/sd_final.pt",
            "model_path": "./model_weight/Final/sd_63.76_Nu44_lr0.001_drop0.8_wDA.pt",
            "mode": "sd_test",
            "paras": {"dropoutRate": 0.8, "Nu": 44},
        },
        "LOSO_60.76_Nu44_lr0.002_drop0.7_woDA": { # 60.76%
            # "model_path": "./model_weight/Nu44_lr0.002_drop0.7_woDA/loso_final.pt",
            "model_path": "./model_weight/Final/loso_60.76_Nu44_lr0.002_drop0.7_woDA.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.7, "Nu": 44},
        },
        "LOSO_60.76_Nu44_lr0.001_drop0.8_wDA": { # 60.76%
            # "model_path": "./model_weight/Nu44_lr0.001_drop0.8_wDA/loso_final.pt",
            "model_path": "./model_weight/Final/loso_60.76_Nu44_lr0.001_drop0.8_wDA.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.8, "Nu": 44},
        },
        "LOSO+FT_77.43_Nu22_lr0.01_drop0.5_wDA": { # 77.43%
            # "model_path": "./model_weight/Nu22_lr0.01_drop0.5_wDA/losoft_final.pt",
            "model_path": "./model_weight/Final/losoft_77.43_Nu22_lr0.01_drop0.5_wDA.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.5, "Nu": 22},
        },
        "LOSO+FT_77.78_Nu44_lr0.001_drop0.8_wDA": { # 77.78%
            # "model_path": "./model_weight/Nu44_lr0.001_drop0.8_wDA/losoft_final.pt",
            "model_path": "./model_weight/Final/losoft_77.78_Nu44_lr0.001_drop0.8_wDA.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.8, "Nu": 44},
        },
        # losoft_80.21_Nu44_lr0.001_drop0.8_wDA_Step500
        "LOSO+FT_80.21_Nu44_lr0.001_drop0.8_wDA_Step500": { # 80.21%
            "model_path": "./model_weight/Final/losoft_80.21_Nu44_lr0.001_drop0.8_wDA_Step500.pt",
            "mode": "loso_test",
            "paras": {"dropoutRate": 0.8, "Nu": 44},
        },
    }

    # for i, (task, info) in enumerate(tasks.items(), 1):
    for i, (task, info) in enumerate(final_tasks.items(), 1):
        acc = sccnet_test(batch_size=32,
                          mode=info["mode"],
                          model_path=info["model_path"],
                          paras=info["paras"])
        print(f'Task: {task:50} accuracy: {acc:.2f}%')

        # import shutil
        # splits = info["model_path"].split('/')
        # paras = splits[-2]
        # mode = splits[-1].split('_')[0]
        # dst_path = f"./model_weight/{mode}_{acc:.2f}_{paras}.pt"
        # shutil.copy(info["model_path"], dst_path)
