# implement your testing script here
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.SCCNet import SCCNet  # 確保 SCCNet 模型定義在 SCCNet.py 中
from Dataloader import MIBCI2aDataset  # 確保數據加載器定義在 dataloader.py 中
from utils import load_model, calculate_accuracy, plot_confusion_matrix

def test_sccnet(model_path, batch_size, mode):
    # 設置設備（GPU 或 CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加載數據集
    if mode == 'LOSO' or mode == 'FT':
        test_dataset = MIBCI2aDataset(mode='LOSO_test')
    elif mode == 'SD':
        test_dataset = MIBCI2aDataset(mode='SD_test')

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 創建模型
    model = SCCNet(numClasses=4, timeSample=438, dropoutRate=0.5).to(device)

    # 加載模型權重
    load_model(model, model_path)

    # 設置模型為評估模式
    model.eval()

    # 初始化變量來計算準確率
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # 不需要計算梯度
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向傳播
            outputs = model(inputs)
            outputs = nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            # 累計正確預測和總樣本數
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 計算準確率
    accuracy = correct / total * 100
    print(f'Accuracy of the model on the test dataset: {accuracy:.2f}%')

    # 繪製混淆矩陣
    # plot_confusion_matrix(all_labels, all_preds, classes=['Left Hand', 'Right Hand', 'Feet', 'Tongue'], normalize=True)

    return accuracy

if __name__ == '__main__':
    model_path = './model_weights/FT80.56_drop0.8_lr0.001_epochs300.pth'
    batch_size = 64
    accuracy = test_sccnet(model_path, batch_size, mode='FT')
    print(f'Test Accuracy: {accuracy:.2f}%')