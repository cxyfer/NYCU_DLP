import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.SCCNet import *
from Dataloader import *
from tester import *
from utils import *

weight_path = './model_weight/'
if not os.path.exists(weight_path):
    os.makedirs(weight_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sd_train(epochs=1000, batch_size=32, learning_rate=0.01, dropout_rate=0.5, target=70,
               model_path='./model_weight/sd.pt',
               final_model_path='./model_weight/sd_final.pt'):
    print(device)
    
    train_dataset = MIBCI2aDataset(mode='sd_train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SCCNet(numClasses=4, timeSample=438, dropoutRate=dropout_rate).to(device)
    if os.path.exists(final_model_path):
        model.load_state_dict(torch.load(final_model_path))
    elif os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) 

    print(f'Training started with {epochs} epochs, {batch_size} batch size, {learning_rate} learning rate, {dropout_rate} dropout rate, and target accuracy of {target}%')
    model.train()
    loss_history = []
    if os.path.exists(final_model_path):
        max_acc = sccnet_test(batch_size=batch_size, mode="sd_test", model_path=final_model_path)
        print(f'Max Accuracy: {max_acc:.2f}%')
    else:
        max_acc = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for _, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        torch.save(model.state_dict(), model_path)
        acc = sccnet_test(batch_size=batch_size, mode="sd_test", model_path=model_path)
        print(f'[RUN] Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Learning Rate: {current_lr}, Accuracy: {acc:.2f}%')
        if acc > max_acc:
            max_acc = acc
            torch.save(model.state_dict(), final_model_path)
            print(f"[SAVE] Max Accuracy: {max_acc:.2f}% at epoch {epoch + 1}")
        if acc > target:
            break
        scheduler.step()
    
    torch.save(model.state_dict(), model_path)
    print(f'Max Accuracy: {max_acc:.2f}%')

    return loss_history

def loso_train(epochs=1000, batch_size=32, learning_rate=0.01, dropout_rate=0.5, target=70,
               model_path='./model_weight/loso.pt',
               final_model_path='./model_weight/loso_final.pt'):
    mode = 'loso_train'
    print(device)
    
    train_dataset = MIBCI2aDataset(mode='loso_train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SCCNet(numClasses=4, timeSample=438, dropoutRate=dropout_rate).to(device)
    if os.path.exists(final_model_path):
        model.load_state_dict(torch.load(final_model_path))
    elif os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) 

    print(f'Training started with {epochs} epochs, {batch_size} batch size, {learning_rate} learning rate, {dropout_rate} dropout rate, and target accuracy of {target}%')
    model.train()
    loss_history = []
    if os.path.exists(final_model_path):
        max_acc = sccnet_test(batch_size=batch_size, mode="loso_test", model_path=final_model_path)
        print(f'Max Accuracy: {max_acc:.2f}%')
    else:
        max_acc = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for _, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        torch.save(model.state_dict(), model_path)
        acc = sccnet_test(batch_size=batch_size, mode="loso_test", model_path=model_path)
        print(f'[RUN] Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Learning Rate: {current_lr}, Accuracy: {acc:.2f}%')
        if acc > max_acc:
            max_acc = acc
            torch.save(model.state_dict(), final_model_path)
            print(f"[SAVE] Max Accuracy: {max_acc:.2f}% at epoch {epoch + 1}")
        if acc > target:
            break
        scheduler.step()
    
    torch.save(model.state_dict(), model_path)
    print(f'Max Accuracy: {max_acc:.2f}%')

    return loss_history

def losoft_train(epochs=1000, batch_size=32, learning_rate=0.01, dropout_rate=0.5, target=80,
                base_model_path='./model_weight/loso_final.pt',
                model_path='./model_weight/losoft.pt',
                final_model_path='./model_weight/losoft_final.pt'):
    mode = 'finetune'
    print(device)
    
    train_dataset = MIBCI2aDataset(mode='finetune')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SCCNet(numClasses=4, timeSample=438, dropoutRate=dropout_rate).to(device)
    if not os.path.exists(base_model_path):
        raise ValueError(f'Base model path {base_model_path} does not exist')
    model.load_state_dict(torch.load(base_model_path))
    
    if os.path.exists(final_model_path):
        model.load_state_dict(torch.load(final_model_path))
    elif os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) 

    print(f'Training started with {epochs} epochs, {batch_size} batch size, {learning_rate} learning rate, {dropout_rate} dropout rate, and target accuracy of {target}%')
    model.train()
    loss_history = []

    if os.path.exists(final_model_path):
        max_acc = sccnet_test(batch_size=batch_size, mode="loso_test", model_path=final_model_path)
        print(f'Max Accuracy: {max_acc:.2f}%')
    else:
        max_acc = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for _, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        torch.save(model.state_dict(), model_path)
        acc = sccnet_test(batch_size=batch_size, mode="loso_test", model_path=model_path)
        print(f'[RUN] Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Learning Rate: {current_lr}, Accuracy: {acc:.2f}%')
        if acc > max_acc:
            max_acc = acc
            torch.save(model.state_dict(), final_model_path)
            print(f"[SAVE] Max Accuracy: {max_acc:.2f}% at epoch {epoch + 1}")
        if acc > target:
            break
        scheduler.step()
    
    torch.save(model.state_dict(), model_path)
    print(f'Max Accuracy: {max_acc:.2f}%')

    return loss_history

if __name__ == '__main__':
    # loss_history = sd_train(epochs=1000, batch_size=32, learning_rate=1e-2, dropout_rate=0.5, target=70,
    #                           model_path='./model_weight/sd.pt',
    #                           final_model_path='./model_weight/sd_final.pt')
    # print('Training completed.')
    # plot_loss_curve(loss_history)

    # loss_history = loso_train(epochs=1000, batch_size=32, learning_rate=1e-2, dropout_rate=0.5, target=63,
    #                           model_path='./model_weight/loso.pt',
    #                           final_model_path='./model_weight/loso_final.pt')
    # print('Training completed.')
    # plot_loss_curve(loss_history)

    loss_history = losoft_train(epochs=1000, batch_size=32, learning_rate=1e-2, dropout_rate=0.5, target=80,
                                base_model_path='./model_weight/loso_final.pt',
                                model_path='./model_weight/losoft.pt',
                                final_model_path='./model_weight/losoft_final.pt')
    print('Training completed.')
    plot_loss_curve(loss_history)