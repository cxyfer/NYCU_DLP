# DLP_LAB4

> [name=313551133 陳軒宇]

[TOC]

## i. Derivate conditional VAE formula (5%)

<center>
<img src="https://i.imgur.com/AfGf4fc.png" width="90%" height="90%" />
</center>

## ii. Introduction (5%)

本實驗的目的在於使用 Conditional VAE 實現影像預測模型。給定一幀畫面和一系列的條件標籤（姿勢），我們希望我們的模型能夠預測未來的幀，生成影像序列，使畫面中的人物做出與給定姿勢相似的動作。

因此，在訓練階段，我們輸入一系列影像和相應的幀，使我們的模型學習如何生成下一幀。由於模型需要生成連續的影像序列，我們使用 VAE 而非普通的 AutoEncoder，因為 VAE 會向模型添加噪聲，使 VAE 能夠生成分佈而非離散變量。

這種方法使得模型能夠捕捉影像序列中的變化和不確定性，從而產生更自然、更連貫的預測結果。通過 Conditional VAE 的應用，我們期望模型能夠根據給定的姿勢信息，生成符合特定動作和風格的影像序列。

## iii. Implementation details (25%)
### 1. How do you write your training/testing protocol (10%)

#### training

```python!
def training_one_step(self, img, label, adapt_TeacherForcing):
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

    return loss
```

```python!
def val_one_step(self, img, label):
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
```

#### testing

```python!
    def val_one_step(self, img, label, idx=0):
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        assert label.shape[0] == 630, "Testing pose seqence should be 630"
        assert img.shape[0] == 1, "Testing video seqence should be 1"
        
        # decoded_frame_list is used to store the predicted frame seq
        # label_list is used to store the label seq
        # Both list will be used to make gif
        decoded_frame_list = [img[0].cpu()]
        label_list = []

        # TODO: ok
        input_frame = img[0]
        for i in range(self.val_vi_len - 1): # 629
            # forward
            frame = self.frame_transformation(input_frame)
            pose = self.label_transformation(label[i])
            z = torch.randn(1, self.args.N_dim, self.args.frame_H, self.args.frame_W).to(self.args.device)
            fusion = self.Decoder_Fusion(frame, pose, z)
            pred_frame = self.Generator(fusion)

            decoded_frame_list.append(pred_frame.cpu())
            label_list.append(label[i].cpu())
            input_frame = pred_frame
        # raise NotImplementedError
            
        # Please do not modify this part, it is used for visulization
        generated_frame = stack(decoded_frame_list).permute(1, 0, 2, 3, 4)
        label_frame = stack(label_list).permute(1, 0, 2, 3, 4)
        
        assert generated_frame.shape == (1, 630, 3, 32, 64), f"The shape of output should be (1, 630, 3, 32, 64), but your output shape is {generated_frame.shape}"
        
        self.make_gif(generated_frame[0], os.path.join(self.args.save_root, f'pred_seq{idx}.gif'))
        
        # Reshape the generated frame to (630, 3 * 64 * 32)
        generated_frame = generated_frame.reshape(630, -1)
        
        return generated_frame
```

### 2. How do you implement reparameterization tricks (5%)

<center>
<img src="https://i.imgur.com/aUYoyLi.png" width="80%" height="90%" />
</center>

```python!
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
```

### 3. How do you set your teacher forcing strategy (5%)

<center>
<img src="https://i.imgur.com/cSVH4Ip.png" width="50%" height="60%" />
<p>tfr=1.0, tfr_sde=10, tfr_d_step=0.1</p>
</center>

```python!
def teacher_forcing_ratio_update(self):
    if self.current_epoch >= self.tfr_sde:
        self.tfr = max(0.0, self.tfr - self.tfr_d_step)
```

### 4. How do you set your kl annealing ratio (5%)

<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1; text-align: center;">
    <img src="https://i.imgur.com/qbYlPFO.png" width="100%" />
    <p>type="Cyclical", cycle=10, ratio=1.0</p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="https://i.imgur.com/dSk0g8U.png" width="100%" />
    <p>type="Monotonic", cycle=10, ratio=1.0</p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="https://i.imgur.com/blgznWq.png" width="100%" />
    <p>type="None"</p>
  </div>
</div>

```python!
class kl_annealing():
    def __init__(self, args, current_epoch=0):
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

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        # problem: 這樣會導致 beta 不會到 stop 的值
        #          應該要寫 n_iter % (n_cycle + 1)
        self.beta = min(stop, ((n_iter % n_cycle) / n_cycle) * ratio)
        # print(f"update beta: {self.beta}")
```

## iv. Analysis & Discussion (25%)
### 1. Plot Teacher forcing ratio (5%)

<center>
<img src="https://i.imgur.com/EPwA2Vx.png" width="50%" height="60%" />
<p>tfr=1.0, tfr_sde=10, tfr_d_step=0.1</p>
</center>

#### a. Analysis & compare with the loss curve

在訓練時我發現使用 teacher forcing 的效果不佳，可能是因為在初始階段的高 teacher forcing ratio 導致無法學習到東西，且後續需要花費更多的 epoch 來修正。由於我希望在盡可能短的 epoch 數量內達到最好的效果，因此我直接不使用 teacher forcing。

### 2. Plot the loss curve while training with different settings. Analyze the difference between them (10%)

雖然訓練的 epoch 數量有所不同，但還是可以看出 Cyclical KL Annealing 可以使 PSNR 階段式的上升，而 Monotonic KL Annealing 由於在超過 kl anneal cycle 後就與不使用 KL Annealing 一樣，所以表現上會比較差。

#### a. With KL annealing (Monotonic)

```bash!
!python Trainer.py --DR ./LAB4_Dataset/ --save_root /content/drive/MyDrive/DLP/Lab4/ckpts/Monotonic_kl1.0_c10_tfr0.0 --ckpt_path /content/drive/MyDrive/DLP/Lab4/ckpts/Monotonic_kl1.0_c10_tfr0.0/epoch=18.ckpt --tfr 0 --tfr_sde 5 --kl_anneal_type Monotonic --kl_anneal_cycle 10 --kl_anneal_ratio 1.0 --num_epoch 33 --fast_train --fast_train_epoch 6 --batch_size 4 --per_save 2
```

<center>
<img src="https://i.imgur.com/7SxBpLd.png" width="90%" height="90%" />
</center>

#### b. With KL annealing (Cyclical)

```bash!
!python Trainer.py --DR ./LAB4_Dataset/ --save_root /content/drive/MyDrive/DLP/Lab4/ckpts/Cyclical_kl1.0_c10_tfr0.0 --tfr 0 --tfr_sde 5 --kl_anneal_type Cyclical --kl_anneal_cycle 10 --kl_anneal_ratio 1.0 --num_epoch 201 --fast_train --fast_train_epoch 6 --batch_size 4 --per_save 2
```

<center>
<img src="https://i.imgur.com/uEZb88O.png" width="90%" height="90%" />
</center>

#### c. Without KL annealing

```bash!
!python Trainer.py --DR ./LAB4_Dataset/ --save_root /content/drive/MyDrive/DLP/Lab4/ckpts/None_kl1.0_tfr0.0 --tfr 0 --kl_anneal_type None --num_epoch 50 --fast_train --fast_train_epoch 6 --batch_size 4 --per_save 2
```

<center>
<img src="https://i.imgur.com/mYbaIso.png" width="90%" height="90%" />
</center>

### 3. Plot the PSNR-per frame diagram in validation dataset (5%)

<center>
<img src="https://i.imgur.com/Np3kE5L.png" width="70%" height="70%" />
</center>

Parameters:
- KL annealing type: `Cyclical`
- KL annealing cycle: $10$
- KL annealing ratio: $1.0$
- Teacher forcing ratio: $0.0$
- epochs: $200$

### 4. Other training strategy analysis (Bonus) (5%)

pass