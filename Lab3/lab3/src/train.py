import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from oxford_pet import load_dataset
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists("./saved_models/"):
    os.makedirs("./saved_models/")

def train(args):
    # implement the training function here

    assert args.model in ["unet", "resnet34_unet"]

    print(device)

    train_dataset = load_dataset(args.data_path, "train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = load_dataset(args.data_path, "valid")
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    if args.model == 'unet':
        model = UNet(input_channels=3, n_classes=1).to(device)
    else:
        model = ResNet34_UNet(in_channels=3, n_classes=1).to(device)

    # Set the save path
    save_path = os.path.join(args.save_path, args.model)
    model_path = os.path.join(save_path, args.model + '.pth')
    ckpt_path = os.path.join(save_path, args.model + '.ckpt')
    final_model_path = os.path.join(save_path, args.model + '_final.pth')

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(f"Model saved at {save_path}")

    # Set the criterion, optimizer and scheduler
    critirion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    # keep track of performance metrics and some other information
    train_losses = []
    valid_losses = []
    valid_scores = []
    start_epoch = 1
    best_valid_loss = np.inf
    best_valid_score = 0.0

    # load the checkpoint if it exists
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        train_losses = ckpt['train_losses']
        valid_losses = ckpt['valid_losses']
        valid_scores = ckpt['valid_scores']
        best_valid_loss = ckpt['best_valid_loss']
        best_valid_score = ckpt['best_valid_score']
        print(f"Model loaded from {ckpt_path}")

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        # train the model
        model.train() # set the model to training mode
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} - Train"):
            images, masks = batch["image"].to(device), batch["mask"].to(device)
            outputs = model(images)

            # flatten the output and mask tensors for cross entropy loss
            # loss = critirion(outputs, masks)
            loss = critirion(outputs.flatten(start_dim=1), masks.flatten(start_dim=1))
            train_loss += loss.item() * images.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader.dataset)

        # evaluate the model
        model.eval() # set the model to evaluation mode
        valid_loss = 0.0
        valid_score = 0.0 # dice score

        with torch.no_grad(): # without tracking history
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch}/{args.epochs} - Valid"):
                images, masks = batch["image"].to(device), batch["mask"].to(device)

                outputs = model(images)
                # flatten the output and mask tensors for cross entropy loss
                # loss = critirion(outputs, masks)
                loss = critirion(outputs.flatten(start_dim=1), masks.flatten(start_dim=1))
                valid_loss += loss.item() * images.size(0)
                valid_score += dice_score(outputs, masks) * images.size(0)

        valid_loss /= len(valid_loader.dataset)
        valid_score /= len(valid_loader.dataset)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_scores.append(valid_score)

        print(f"Epoch {epoch}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}, Val Score: {valid_score:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        scheduler.step() # update the learning rate

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"{save_path}/{args.model}_best_loss.pth")

        if valid_score > best_valid_score:
            best_valid_score = valid_score
            torch.save(model.state_dict(), f"{save_path}/{args.model}_best_score.pth")

        if epoch % 25 == 0:
            torch.save(model.state_dict(), f"{save_path}/{args.model}_epoch_{epoch}.pth")
            info = {"model": model, "train_losses": train_losses, "valid_losses": valid_losses, "valid_scores": valid_scores}
            with open(f"{save_path}/{args.model}_epoch_{epoch}.pkl", "wb") as f: # save the loss history
                pickle.dump(info, f)

        # save checkpoint
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                'valid_scores': valid_scores,
                'best_valid_loss': best_valid_loss,
                'best_valid_score': best_valid_score
            }, ckpt_path)

    torch.save(model.state_dict(), final_model_path)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default='dataset', help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=500, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument("--model", type=str, default="unet", help="model to use (unet/resnet34_unet)")
    parser.add_argument("--save_path", type=str, default="./saved_models/", help="path to save the model")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)
