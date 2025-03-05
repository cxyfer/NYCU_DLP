import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def load_checkpoint(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")
    ckpt = torch.load(ckpt_path)
    return ckpt

def show_history(ckpt_path, suptitle="History"):
    ckpt = load_checkpoint(ckpt_path)

    valid_loss = ckpt['val_loss_list']
    valid_psnr = ckpt['val_psnr_list']

    plt.figure(figsize=(10, 5))
    plt.suptitle(suptitle)
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.plot(valid_loss, label='valid loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title('PSNR')
    plt.plot(valid_psnr, label='valid psnr')
    plt.xlabel('epoch')
    plt.ylabel('PSNR')
    plt.legend()
    plt.savefig(os.path.join(ckpt_path.replace('.ckpt', '.png')))

def simulate_teacher_forcing(epoch=100, tfr=1.0, tfr_sde=10, tfr_d_step=0.1):
    tfr_bak = tfr
    results = []
    for i in range(epoch):
        if i >= tfr_sde:
            tfr = max(0.0, tfr - tfr_d_step)
        results.append(tfr)
    plt.plot(results, label=f"tfr", color="red")
    plt.xlabel("epcoh")
    plt.ylabel("tfr")
    plt.title("Teacher Forcing Ratio")
    plt.legend()
    plt.ylim(-0.1, max(1, max(results)) + 0.1)
    plt.savefig(f"./ckpts/tfr={tfr_bak}.png")
    plt.close()

class kl_annealing():
    def __init__(self, current_epoch=0, type="Cyclical", cycle=10, ratio=1.0):
        self.current_epoch = current_epoch
        self.type = type
        self.cycle = cycle
        self.ratio = ratio
        self.beta = 0.0 if self.type in ["Cyclical", "Monotonic"] else 1.0
        self.update() # problem: 這行會導致 beta 不會從 0.0 開始

    def update(self):
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

    def get_beta(self):
        return self.beta

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        self.beta = min(stop, ((n_iter % n_cycle) / n_cycle) * ratio)

def simulate_kl_annealing(epoch=100, kl_anneal_type="Cyclical", kl_anneal_cycle=10, kl_anneal_ratio=1.0):
    kl = kl_annealing(type=kl_anneal_type, cycle=kl_anneal_cycle, ratio=kl_anneal_ratio)
    results = []
    for i in range(epoch):
        beta = kl.get_beta()
        results.append(beta)
        kl.update()
    plt.plot(results, label=f"beta", color="green")
    plt.xlabel("epcoh")
    plt.ylabel("beta")
    plt.title("KL Annealing")
    plt.legend()
    plt.ylim(-0.1, max(1, max(results)) + 0.1)
    plt.savefig(f"./ckpts/kl_{kl_anneal_type}.png")
    plt.close()
        

if __name__ == "__main__":
    show_history("./ckpts/Cyclical_kl1.0_c10_tfr0.0_epoch=200.ckpt", suptitle="Cyclical_kl1.0_c10_tfr0.0_epoch=200")
    show_history("./ckpts/Monotonic_kl1.0_c10_tfr0.0_epoch=50.ckpt", suptitle="Monotonic_kl1.0_c10_tfr0.0_epoch=50")
    show_history("./ckpts/None_kl1.0_tfr0.0_epoch=48.ckpt", suptitle="None_kl1.0_tfr0.0_epoch=48")
    simulate_teacher_forcing()
    simulate_teacher_forcing(tfr=0.0)
    simulate_kl_annealing()
    simulate_kl_annealing(kl_anneal_type="Monotonic", kl_anneal_cycle=10, kl_anneal_ratio=1.0)
    simulate_kl_annealing(kl_anneal_type="None", kl_anneal_cycle=10, kl_anneal_ratio=1.0)
    