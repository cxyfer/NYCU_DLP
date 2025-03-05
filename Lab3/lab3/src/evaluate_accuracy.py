import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import dice_score
from oxford_pet import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_accuracy(model, data, desc=""):
    # implement the evaluation function here
    model.eval()
    tot, acc = 0, 0
    with torch.no_grad():
        for batch in tqdm(data, desc=desc):
            images, masks = batch["image"].to(device), batch["mask"].to(device)
            outputs = model(images)

            outputs = (outputs > 0.5).bool()
            masks = masks.bool()

            acc += (masks == outputs).sum().item()
            tot += masks.size(0) * masks.size(1) * masks.size(2) * masks.size(3) # (batch_size, channels, height, width)
    print(f"{acc} / {tot}")
    return acc / tot

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the UNet and ResNet34-UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default='dataset', help='path of the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='batch size')
    parser.add_argument("--model", type=str, default="unet", help="model to use (unet/resnet34_unet)")
    parser.add_argument("--model_path", type=str, default="./saved_models/unet/unet_final.pth", help="path to load the model")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    assert args.model in ["unet", "resnet34_unet"]

    test_dataset = load_dataset(args.data_path, "test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.model == 'unet':
        model = UNet(input_channels=3, n_classes=1).to(device)
    else:
        model = ResNet34_UNet(in_channels=3, n_classes=1).to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")
    
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    scores = evaluate_accuracy(model, test_loader, desc=f"{os.path.basename(args.model_path)}")
    print(f"Accuracy: {scores:.4f}")
