import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

random.seed(116)
torch.manual_seed(116)

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size
    return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO: ok
        self.current_epoch = current_epoch
        self.type = args.kl_anneal_type
        self.cycle = args.kl_anneal_cycle
        self.ratio = args.kl_anneal_ratio
        self.beta = 0.0 if self.type in ["Cyclical", "Monotonic"] else 1.0
        self.update() # problem: 這行會導致 beta 不會從 0.0 開始

    def update(self):
        # TODO: ok
        self.current_epoch += 1
        if self.type == "Cyclical":
            self.frange_cycle_linear(n_iter=self.current_epoch,
                                     start=self.beta,
                                     stop=1.0,
                                     n_cycle=self.cycle,
                                     ratio=self.ratio)
        elif self.type == "Monotonic":
            self.beta = min(1.0, (self.current_epoch / self.cycle) * self.ratio)
        elif self.type == 'None':
            pass
        else:
            print(self.type)
            raise ValueError("kl_anneal_type must be in ['Cyclical', 'Monotonic', 'None']")

    def get_beta(self):
        # TODO: ok
        return self.beta

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        # TODO: ok
        # problem: 這樣會導致 beta 不會到 stop 的值
        #          應該要寫 n_iter % (n_cycle + 1)
        self.beta = min(stop, ((n_iter % n_cycle) / n_cycle) * ratio)
        # print(f"update beta: {self.beta}")

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args

        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)

        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)

        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)

        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0

        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde

        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
    
        # Logs
        self.loss_list = []
        self.mse_loss_list = []
        self.kl_loss_list = []
        self.val_loss_list = []
        self.val_psnr_list = []

    def forward(self, img, label):
        pass

    def training_stage(self):
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False

            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)

                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])

            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))

            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()


    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0], psnr=self.val_psnr)

    def training_one_step(self, img, label, adapt_TeacherForcing):
        # TODO: ok

        # Set all modules to train mode
        self.frame_transformation.train()
        self.label_transformation.train()
        self.Gaussian_Predictor.train()
        self.Decoder_Fusion.train()
        self.Generator.train()
        
        mse_loss = 0
        kl_loss = 0
        pred_nxt_frame = img[:, 0]
        for i in range(self.train_vi_len - 1):
            # Set teacher forcing
            cur_frame = img[:, i] if adapt_TeacherForcing else pred_nxt_frame
            nxt_frame = img[:, i+1]
            cur_pose = label[:, i]
            nxt_pose = label[:, i+1]

            encoded_cur_frame = self.frame_transformation(cur_frame)
            encoded_nxt_frame = self.frame_transformation(nxt_frame)
            encoded_nxt_pose = self.label_transformation(nxt_pose)
            
            # Prediction
            z, mu, logvar = self.Gaussian_Predictor(encoded_nxt_frame, encoded_nxt_pose) # img, label
            fusion = self.Decoder_Fusion(encoded_cur_frame, encoded_nxt_pose, z) # img, label, parm
            pred_nxt_frame = self.Generator(fusion) # input

            # Loss
            kl_loss += kl_criterion(mu, logvar, self.batch_size) # KL Loss
            mse_loss += self.mse_criterion(pred_nxt_frame, nxt_frame) # MSE Loss
        
        # KL Annealing
        beta = self.kl_annealing.get_beta()
        loss = mse_loss + beta * kl_loss # Total Loss

        # Backpropagation
        self.optim.zero_grad() # Clear gradients
        loss.backward() # Backpropagation
        self.optimizer_step() # Optimizer step

        # Save history
        # problem: train 的時候要寫在 training_stage 中才對 = =
        self.loss_list.append(loss.item())
        self.mse_loss_list.append(mse_loss.item())
        self.kl_loss_list.append(kl_loss.item())

        return loss

    def val_one_step(self, img, label):
        # TODO: ok

        # Set all modules to eval mode
        self.frame_transformation.eval()
        self.label_transformation.eval()
        self.Decoder_Fusion.eval()
        self.Generator.eval()
        
        mse_loss = 0
        psnr_sum = 0
        pred_nxt_frame = img[:, 0]

        # Test
        predicted_img_list = [] # Could add the first frame
        psnr_list = [] 

        for i in range(self.val_vi_len - 1):
            cur_frame, nxt_frame = pred_nxt_frame, img[:, i+1]
            cur_pose, nxt_pose = label[:, i], label[:, i+1]

            encoded_cur_frame = self.frame_transformation(cur_frame)
            encoded_nxt_pose = self.label_transformation(nxt_pose)

            z = torch.randn(1, self.args.N_dim, self.args.frame_H, self.args.frame_W).to(self.args.device)
            fusion = self.Decoder_Fusion(encoded_cur_frame, encoded_nxt_pose, z)
            pred_nxt_frame = self.Generator(fusion)

            # Loss
            mse_loss += self.mse_criterion(pred_nxt_frame, nxt_frame) # MSE Loss
            cur_psnr = Generate_PSNR(pred_nxt_frame, nxt_frame).item() # PSNR
            psnr_sum += cur_psnr

            # Test
            if self.args.test:
                psnr_list.append(cur_psnr) 
                predicted_img_list.append(pred_nxt_frame[0])

        # Logs
        self.val_psnr = psnr_sum / self.val_vi_len # Average PSNR of the validation set
        self.val_loss_list.append(mse_loss.item())
        self.val_psnr_list.append(self.val_psnr)

        if self.args.test:
            # Save gif
            self.make_gif(predicted_img_list, os.path.join(self.args.save_root, f"epoch={self.current_epoch}_val.gif"))

            # Save PSNR-per frame diagram
            plt.plot(psnr_list, label=f"Average PSNR: {round(self.val_psnr, 5)}")
            plt.xlabel("frame")
            plt.ylabel("PSNR")
            plt.title("PSNR-per frame diagram")
            plt.legend()
            plt.savefig(os.path.join(self.args.save_root, f"epoch={self.current_epoch}_val.png"))
            plt.close()

        return mse_loss

    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))

        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False

        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)
        return train_loader

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)
        return val_loader

    def teacher_forcing_ratio_update(self):
        # TODO: ok
        if self.current_epoch >= self.tfr_sde:
            self.tfr = max(0.0, self.tfr - self.tfr_d_step)

    def tqdm_bar(self, mode, pbar, loss, lr, psnr=None):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        if psnr is not None:
            pbar.set_postfix(loss=float(loss), psnr=round(psnr, 6), refresh=False)
        else:
            pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()

    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            # "optimizer": self.state_dict(),
            "optimizer": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch,
            # Logs
            "loss_list" : self.loss_list,
            "mse_loss_list" : self.mse_loss_list,
            "kl_loss_list" : self.kl_loss_list,
            "val_loss_list" : self.val_loss_list,
            "val_psnr_list" : self.val_psnr_list,
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True)
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']

            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

            self.loss_list = checkpoint['loss_list']
            self.mse_loss_list = checkpoint['mse_loss_list']
            self.kl_loss_list = checkpoint['kl_loss_list']
            self.val_loss_list = checkpoint['val_loss_list']
            self.val_psnr_list = checkpoint['val_psnr_list']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):

    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")


    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")

    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")

    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")

    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")




    args = parser.parse_args()

    main(args)
