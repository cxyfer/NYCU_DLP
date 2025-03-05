import os
import argparse
from tqdm import tqdm

import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from oxford_pet import load_dataset
from utils import *

def predict(model, data, device, desc="Predicting"):
    model.eval()
    images = []
    preds_masks = []
    gt_masks = []
    with torch.no_grad():
        for batch in tqdm(data, desc=desc):
            inputs = batch["image"].to(device)
            outputs = model(inputs)
            images.append(inputs.cpu().numpy())
            preds_masks.append(outputs.cpu().numpy())
            gt_masks.append(batch["mask"].cpu().numpy())
    return dict(images=np.concatenate(images, axis=0), preds_masks=np.concatenate(preds_masks, axis=0), gt_masks=np.concatenate(gt_masks, axis=0))

def predict_single(model, data, device):
    model.eval()
    with torch.no_grad():
        image = torch.from_numpy(data["image"])
        image = image.unsqueeze_(0).to(device)
        pred_mask = model(image).squeeze(0) # (1, 1, H, W) -> (1, H, W)
    return dict(image=data["image"], pred_mask=pred_mask.cpu().numpy(), gt_mask=data["mask"])

def save_images(images, pred_masks, gt_masks, filenames, output_path, alpha=0.5):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for (image, pred_mask, gt_mask, filename) in tqdm(zip(images, pred_masks, gt_masks, filenames), total=len(images), desc="Saving images"):
        image = (image * 255).astype(np.uint8)
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255 # 二值化
        gt_mask = (gt_mask > 0.5).astype(np.uint8) * 255 # 二值化

        image = image.transpose(1, 2, 0)
        pred_mask = pred_mask.transpose(1, 2, 0)
        gt_mask = gt_mask.transpose(1, 2, 0)
                                    
        pred_mask = np.repeat(pred_mask, 3, axis=2)
        gt_mask = np.repeat(gt_mask, 3, axis=2)

        image = Image.fromarray(image)
        pred_mask = Image.fromarray(pred_mask)
        gt_mask = Image.fromarray(gt_mask)
        blend1 = Image.blend(image, pred_mask, alpha=alpha)
        blend2 = Image.blend(image, gt_mask, alpha=alpha)

        blend1.save(os.path.join(output_path, f"{filename}_pred.png"))
        blend2.save(os.path.join(output_path, f"{filename}_gt.png"))

def show_single_result(image, pred_mask, gt_mask, suptitle, alpha=0.5, save_path=None):
    image = (image * 255).astype(np.uint8)
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255 # 二值化
    gt_mask = (gt_mask > 0.5).astype(np.uint8) * 255 # 二值化
                                
    pred_mask = np.repeat(pred_mask, 3, axis=0)
    gt_mask = np.repeat(gt_mask, 3, axis=0)

    blend1 = cv2.addWeighted(image, alpha, pred_mask, 1 - alpha, 0)
    blend2 = cv2.addWeighted(image, alpha, gt_mask, 1 - alpha, 0)

    data = dict(image=image, pred_mask=blend1, gt_mask=blend2)
    show_dataset_sample(data, suptitle, save_path)

def get_args():

    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--data_path', type=str, default='dataset', help='path of the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='batch size')
    parser.add_argument("--model", type=str, default="unet", help="model to use (unet/resnet34_unet)")
    parser.add_argument("--model_path", type=str, default="./saved_models/unet/unet_final.pth", help="path to load the model")
    parser.add_argument("--output_path", type=str, default="output", help="path to save the output")
    parser.add_argument('--idx', '-i', type=int, default=-1, help='index of the image to predict')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.model = "unet" if args.model.lower().startswith("u") else "resnet34_unet"
    assert args.model in ["unet", "resnet34_unet"]

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.model == 'unet':
        model = UNet(input_channels=3, n_classes=1).to(device)
    else:
        model = ResNet34_UNet(in_channels=3, n_classes=1).to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")
    
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    test_dataset = load_dataset(args.data_path, "test")
    if args.idx == -1: # 預測所有資料
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        data = predict(model, test_loader, device, desc="Predicting")
        save_images(data["images"], data["preds_masks"], data["gt_masks"], test_dataset.filenames, os.path.join(args.output_path, f"{args.model}"))
    else:
        data = predict_single(model, test_dataset[args.idx], device)
        dice_score = dice_score(data["pred_mask"], data["gt_mask"])
        print(f"Dice Score of test[{args.idx}]: {dice_score:.4f}")
        suptitle = f"{args.model} - test[{args.idx}]: {test_dataset.filenames[args.idx]} - Dice Score: {dice_score:.4f}"
        show_single_result(data["image"], data["pred_mask"], data["gt_mask"], suptitle, save_path=os.path.join(args.output_path, f"{args.model}_{args.idx}.png"))