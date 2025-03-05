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

def train(epochs=1000, batch_size=32, learning_rate=0.01, dropout_rate=0.5, target=70,
          optimizer=optim.Adam, scheduler=optim.lr_scheduler.StepLR,
          step_size=100,
          train_dataset_mode=None, test_dataset_mode=None,
          base_model_path=None, # for finetune
          model_path='./model_weight/sd.pt',
          final_model_path='./model_weight/sd_final.pt'):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_dataset = MIBCI2aDataset(mode=train_dataset_mode)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SCCNet(numClasses=4, timeSample=438, dropoutRate=dropout_rate).to(device)
    if train_dataset_mode == 'finetune':
        if not os.path.exists(base_model_path): # if the base model path does not exist
            raise ValueError(f'Base model path {base_model_path} does not exist')
        model.load_state_dict(torch.load(base_model_path)) # load the base model

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = scheduler(optimizer, step_size=step_size, gamma=0.5)

    max_acc = 0.0
    losses = [] # loss history

    """
        This section deals with the situation of training interruptions and is no longer in use.
    """
    # if os.path.exists(model_path): # if the model exists
    #     model.load_state_dict(torch.load(model_path)) # load the model
    #     with open(f"{model_path}.pkl", 'rb') as f: # load the loss history
    #         losses = pickle.load(f)

    if os.path.exists(final_model_path): # if the final model exists, avoid overwriting
        max_acc = sccnet_test(batch_size=batch_size, mode=test_dataset_mode, model_path=final_model_path)
        print(f'Max Accuracy: {max_acc:.2f}%')

    print(f'Training started with {epochs} epochs, {batch_size} batch size, {learning_rate} learning rate, {dropout_rate} dropout rate, and target accuracy of {target}%.')
    model.train() # set model to train mode
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for _, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() # clear the gradients of all optimized variables
            outputs = model(inputs) # forward pass
            loss = criterion(outputs, labels) # compute the loss
            loss.backward() # backpropagation
            optimizer.step() # update the parameters
            running_loss += loss.item() # accumulate the loss

        avg_loss = running_loss / len(train_loader) # compute the average loss
        losses.append(avg_loss)
        current_lr = optimizer.param_groups[0]['lr'] # get the learning rate
        scheduler.step() # update the learning rate

        torch.save(model.state_dict(), model_path) # save the model
        with open(f"{model_path}.pkl", 'wb') as f: # save the loss history
            pickle.dump(losses, f)
        acc = sccnet_test(batch_size=batch_size, mode=test_dataset_mode, model_path=model_path) # test the model
        print(f"[RUNNING] {epoch:4d}/{epochs} epochs, Loss: {avg_loss:.4f}, LR: {current_lr}, TestAcc: {acc:.2f}%")
        # train_acc = sccnet_test(batch_size=batch_size, mode=train_dataset_mode, model_path=model_path) # test the model
        # print(f"[RUNNING] {epoch:4d}/{epochs} epochs, Loss: {avg_loss:.4f}, LR: {current_lr:.4f}, TrainAcc: {train_acc:.2f}%, TestAcc: {acc:.2f}%")

        if acc > max_acc:
            max_acc = acc # update the max accuracy
            torch.save(model.state_dict(), final_model_path) # save the final model
            print(f"[UPDATE] Max Accuracy: {max_acc:.2f}% at epoch {epoch}")
        if acc > target:
            break

    print(f'Max Accuracy: {max_acc:.2f}%')

    return losses

if __name__ == '__main__':
    # losses1 = train(epochs=500, batch_size=32, learning_rate=1e-3, dropout_rate=0.8, target=70,
    #                 train_dataset_mode='sd_train', test_dataset_mode='sd_test',
    #                 model_path='./model_weight/sd.pt',
    #                 final_model_path='./model_weight/sd_final.pt')
    
    # losses2 = train(epochs=500, batch_size=32, learning_rate=1e-3, dropout_rate=0.8, target=61,
    #                 train_dataset_mode='loso_train', test_dataset_mode='loso_test',
    #                 model_path='./model_weight/loso.pt',
    #                 final_model_path='./model_weight/loso_final.pt')

    losses3 = train(epochs=2000, batch_size=32, learning_rate=1e-3, dropout_rate=0.8, target=81,
                    step_size=500,
                    train_dataset_mode='finetune', test_dataset_mode='loso_test',
                    base_model_path='./model_weight/loso_final.pt',
                    model_path='./model_weight/losoft.pt',
                    final_model_path='./model_weight/losoft_final.pt')
    
    # plot_losses(losses1)
    # plot_losses(losses2)
    # plot_losses(losses3)