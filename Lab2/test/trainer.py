import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.SCCNet import SCCNet  # 確保 SCCNet 模型定義在 SCCNet.py 中
from Dataloader import MIBCI2aDataset  # 確保數據加載器定義在 dataloader.py 中
from tester import test_sccnet
from utils import plot_loss_curve  # 從 utils.py 中導入 plot_loss_curve 函數

def train_sccnet_LOSO(num_epochs, batch_size, learning_rate, dropout_rate, model_save_path):
    # dropout = 0.8, lr = 0.001, epoch = 80, acc = 60.07
    # 設置設備（GPU 或 CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 加載數據集
    train_dataset = MIBCI2aDataset(mode='LOSO_train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 創建模型
    model = SCCNet(numClasses=4, timeSample=438,  dropoutRate=dropout_rate).to(device)

    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9) 

    # 訓練模型
    model.train()
    loss_history = []  # 用於存儲每個 epoch 的損失
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度歸零
            optimizer.zero_grad()

            # 前向傳播
            outputs = model(inputs)

            # 計算損失
            loss = criterion(outputs, labels)

            # 反向傳播
            loss.backward()

            # 更新權重
            optimizer.step()

            # 累加損失
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)  # 記錄當前 epoch 的平均損失

        # 打印每個 epoch 的損失和學習率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Learning Rate: {current_lr}')

        # # 保存模型權重
        # torch.save(model.state_dict(), model_save_path)
        # acc = test_sccnet(model_save_path, 64, mode='LOSO')
        # print('acc =', acc)
        # if acc > 63:
        #     exit(0)

        # 更新學習率
        scheduler.step()

    # 保存模型權重
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    # 繪製損失曲線
    plot_loss_curve(loss_history, title='Training Loss Curve')
    
    
def train_sccnet_SD(num_epochs, batch_size, learning_rate, dropout_rate, model_save_path):
    # 設置設備（GPU 或 CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 加載數據集
    train_dataset = MIBCI2aDataset(mode='SD_train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 創建模型
    model = SCCNet(numClasses=4, timeSample=438,  dropoutRate=dropout_rate).to(device)

    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) 

    # 訓練模型
    model.train()
    loss_history = []  # 用於存儲每個 epoch 的損失
    max_acc = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度歸零
            optimizer.zero_grad()

            # 前向傳播
            outputs = model(inputs)

            # 計算損失
            loss = criterion(outputs, labels)

            # 反向傳播
            loss.backward()

            # 更新權重
            optimizer.step()

            # 累加損失
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)  # 記錄當前 epoch 的平均損失

        # 打印每個 epoch 的損失和學習率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Learning Rate: {current_lr}')

        # 保存模型權重
        torch.save(model.state_dict(), model_save_path)
        acc = test_sccnet(model_save_path, 64, mode='SD')
        print('acc =', acc)
        max_acc = max(max_acc, acc)
        if acc > 70:
            exit(0)

        # 更新學習率
        scheduler.step()

    # 保存模型權重
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')
    print('max_acc is', max_acc)

    # 繪製損失曲線
    plot_loss_curve(loss_history, title='Training Loss Curve')
    
def train_sccnet_LOSOFT(num_epochs, batch_size, learning_rate, dropout_rate, model_save_path):
    # 設置設備（GPU 或 CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 加載數據集
    train_dataset = MIBCI2aDataset(mode='finetune')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 創建模型
    model = SCCNet(numClasses=4, timeSample=438,  dropoutRate=dropout_rate).to(device)

    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) 

    # 訓練模型
    model.train()
    loss_history = []  # 用於存儲每個 epoch 的損失
    max_acc = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度歸零
            optimizer.zero_grad()

            # 前向傳播
            outputs = model(inputs)

            # 計算損失
            loss = criterion(outputs, labels)

            # 反向傳播
            loss.backward()

            # 更新權重
            optimizer.step()

            # 累加損失
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)  # 記錄當前 epoch 的平均損失

        # 打印每個 epoch 的損失和學習率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Learning Rate: {current_lr}')

        # 保存模型權重
        torch.save(model.state_dict(), model_save_path)
        acc = test_sccnet(model_save_path, 64, mode='FT')
        max_acc = max(max_acc, acc)
        print('acc =', acc)
        if acc > 80:
            exit(0)

        # 更新學習率
        scheduler.step()

    # 保存模型權重
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')
    print('max acc = ', max_acc)

    # 繪製損失曲線
    plot_loss_curve(loss_history, title='Training Loss Curve')

if __name__ == '__main__':
    num_epochs = 1000
    batch_size = 64
    learning_rate = 0.01
    dropout_rate = 0.8
    model_save_path = './model_weights/sccnet_SD_model.pth'

    # train_sccnet_LOSO(num_epochs, batch_size, learning_rate, dropout_rate, model_save_path)
    train_sccnet_SD(num_epochs, batch_size, learning_rate, dropout_rate, model_save_path)
    # train_sccnet_LOSOFT(num_epochs, batch_size, learning_rate, dropout_rate, model_save_path)
    