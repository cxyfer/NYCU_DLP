import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

def show_dataset_sample(sample, suptitle, save_path=None):
    """
    Show the dataset sample,
    not be used in the training and testing process actually.
    """
    plt.figure(figsize=(5*len(sample), 5))
    plt.suptitle(suptitle)
    for i, (k, v) in enumerate(sample.items()):
        plt.subplot(1, len(sample), i+1)
        if isinstance(v, torch.Tensor): # if the value is a tensor
            v = torch.cpu(v).numpy()
        if v.shape[0] == 1 or v.shape[0] == 3:
            v = np.moveaxis(v, 0, -1) # CHW -> HWC
        plt.imshow(v)
        plt.axis('off')
        plt.title(k)
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def dice_score(pred_mask, gt_mask, smooth=1e-6):
    """
    Compute Dice Score of predicted mask and ground truth mask, 
    the shape of the input tensors should be (batch_size, channels, height, width).

    Arguments:
        pred_mask: predicted mask (PyTorch tensor)
        gt_mask: ground truth mask (PyTorch tensor)
        smooth: smooth factor, to avoid division by zero (float)

    Returns:
        dice: Dice Score (PyTorch tensor)
    """
    # 確保輸入是 PyTorch tensor
    if not isinstance(pred_mask, torch.Tensor):
        pred_mask = torch.tensor(pred_mask)
    if not isinstance(gt_mask, torch.Tensor):
        gt_mask = torch.tensor(gt_mask)

    # 將預測結果二值化（如果還沒有的話）
    pred_mask = (pred_mask > 0.5).float()
    gt_mask = gt_mask.float()
    
    batch_size = pred_mask.shape[0] 
    dice_scores = torch.zeros(batch_size, device=gt_mask.device)

    def calc_dice_score(pred_mask, gt_mask, smooth=1e-6):
        intersection = (gt_mask * pred_mask).sum() # 計算交集 (都是 1 的地方)
        dice = (2. * intersection + smooth) / (gt_mask.sum() + pred_mask.sum() + smooth)
        return dice
    
    for i in range(batch_size):
        dice_scores[i] = calc_dice_score(pred_mask[i], gt_mask[i], smooth)

    return dice_scores.mean()

def load_checkpoint(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")
    ckpt = torch.load(ckpt_path)
    train_losses = ckpt['train_losses']
    valid_losses = ckpt['valid_losses']
    valid_scores = [score.item() for score in ckpt['valid_scores']]
    return dict(train_losses=train_losses, valid_losses=valid_losses, valid_scores=valid_scores)

def show_history(unet_ckpt_path, resnet_ckpt_path, suptitle="History", end_epochs=300):
    ckpt1 = load_checkpoint(unet_ckpt_path)
    ckpt2 = load_checkpoint(resnet_ckpt_path)

    train_losses1, train_losses2 = ckpt1['train_losses'][:end_epochs], ckpt2['train_losses'][:end_epochs]
    valid_losses1, valid_losses2 = ckpt1['valid_losses'][:end_epochs], ckpt2['valid_losses'][:end_epochs]
    valid_scores1, valid_scores2 = ckpt1['valid_scores'][:end_epochs], ckpt2['valid_scores'][:end_epochs]

    plt.figure(figsize=(15, 5))
    plt.suptitle(suptitle)
    plt.subplot(1, 3, 1)
    plt.title('Train Loss')
    plt.plot(train_losses1, label='UNet (lr=1e-5)')
    plt.plot(train_losses2, label='ResNet34_Unet (lr=5e-5)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.title('Valid Loss')
    plt.plot(valid_losses1, label='UNet (lr=1e-5)')
    plt.plot(valid_losses2, label='ResNet34_Unet (lr=5e-5)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.title('Valid Dice Score')
    plt.plot(valid_scores1, label='UNet (lr=1e-5)')
    plt.plot(valid_scores2, label='ResNet34_Unet (lr=5e-5)')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.show()
    # plt.savefig(f'./results/{suptitle}.png')

if __name__ == "__main__":
    """
        Testing for dice_score function
    """
    # pred_mask = torch.tensor([[0, 1, 1], [1, 1, 0], [0, 1, 1]], dtype=torch.float32)
    # gt_mask = torch.tensor([[0, 1, 1], [1, 1, 1], [0, 1, 0]], dtype=torch.float32)
    # pred_mask = pred_mask.unsqueeze(0).unsqueeze(0) # (batch_size, channels, height, width)
    # gt_mask = gt_mask.unsqueeze(0).unsqueeze(0) # (batch_size, channels, height, width)

    # print(pred_mask.shape, gt_mask.shape)

    # score = dice_score(pred_mask, gt_mask)
    # print(f"Dice Score: {score:.4f}")
    """
        Testing for show_history function
    """
    show_history("./saved_models/unet.ckpt",
                 "./saved_models/resnet34_unet.ckpt",
                 "History",
                 end_epochs=300
                 )