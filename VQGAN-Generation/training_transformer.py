import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from transformer import VQGANTransformer
from utils import load_data, plot_images
from torch import autocast
from torch.cuda.amp import GradScaler
import math
from torch.optim.lr_scheduler import LambdaLR

class TrainTransformer:
    def __init__(self, args):
        self.model = VQGANTransformer(args).to(device=args.device)
        self.optim, self.scheduler = self.configure_optimizers(args)
        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95))

        # we will compute inside train() once dataloader is known, for now pass dummy scheduler
        def lr_lambda(step):
            return 1.0  # placeholder, will update later

        scheduler = LambdaLR(optimizer, lr_lambda)
        return optimizer, scheduler

    def train(self, args):
        train_dataset = load_data(args)
        scaler = GradScaler()

        num_training_steps = args.epochs * len(train_dataset)

        # LR schedule
        def lr_lambda(step):
            if step < args.warmup_steps:
                return float(max(1, step)) / float(max(1, args.warmup_steps))
            progress = float(step - args.warmup_steps) / float(max(1, num_training_steps - args.warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_lr = 0.01
            return cosine * (1 - min_lr) + min_lr

        self.scheduler.lr_lambdas = [lr_lambda]

        def pkeep_schedule(step):
            progress = step / num_training_steps
            # cosine increase
            return args.pkeep_end - 0.5 * (args.pkeep_end - args.pkeep_start) * (1 + math.cos(math.pi * progress))

        all_loss = 0
        global_step = 0

        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    # update pkeep according to schedule
                    self.model.pkeep = pkeep_schedule(global_step)

                    self.optim.zero_grad()
                    with autocast(device_type='cuda', dtype=torch.float16):
                        imgs = imgs.to(device=args.device)
                        logits, targets = self.model(imgs)
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            targets.reshape(-1))

                    scaler.scale(loss).backward()
                    scaler.step(self.optim)
                    scaler.update()
                    self.scheduler.step()

                    if i == 0:
                        all_loss = loss.item()
                    else:
                        all_loss = all_loss * i / (i + 1) + loss.item() / (i + 1)

                    pbar.set_postfix(
                        Epoch=epoch + 1,
                        Step=global_step,
                        LR=self.scheduler.get_last_lr()[0],
                        Pkeep=round(self.model.pkeep, 3),
                        Transformer_Loss=np.round(all_loss, 4)
                    )
                    pbar.update(0)
                    global_step += 1

            with autocast(device_type='cuda', dtype=torch.float16):
                log, sampled_imgs = self.model.log_images(imgs[0][None])
            vutils.save_image(sampled_imgs.add(1).mul(0.5),os.path.join(f"results_transformer/{args.codebook_optimization}",f"transformer_epoch{epoch + 1}.jpg"),nrow=4)

            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(),os.path.join("checkpoints",f"transformer_{args.codebook_optimization}_epoch{epoch + 1}_{args.codebook_bits}bit_lr{args.learning_rate}_bs{args.batch_size}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer")
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Latent dimension')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image height and width')
    parser.add_argument('--codebook_bits', type=int, default=10,
                        help='number of bits per codebook. No. of codewords in the codebook'
                             ' equals to 2^codebook_bits')
    parser.add_argument("--codebook_optimization", type=str, default='diveq',
                        help='method to optimize VQ codebook: options -> "ste", "ema", "rt",'
                             ' "gumbel_softmax", "nsvq", "diveq", "sfdiveq", "diveq_detach",'
                             ' "sfdiveq_detach" ')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images')
    parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data.')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to trained VQ-VAE checkpoint')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch-size', type=int, default=32, help='Input batch size for training the transformer')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train the transformer')
    parser.add_argument('--learning-rate', type=float, default=4.5e-5, help='Learning rate to train the transformer')

    parser.add_argument('--warmup_steps', type=int, default=10000, help='Number of training iterations to warmup the learning rate')
    parser.add_argument('--pkeep_start', type=float, default=0.5, help='Initial value of pkeep that is the probability of keeping token indices during training')
    parser.add_argument('--pkeep_end', type=float, default=0.95, help='Final value of pkeep that is the probability of keeping token indices during training')

    parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')

    args = parser.parse_args()

    # path to dataset directory
    args.dataset_path = r"./data_dir/celeba_hq_256"

    # device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(f"results_transformer/{args.codebook_optimization}", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Configs from trained VQ-VAE in first stage training -> change them based on your configs
    trained_vqvae_epochs = 100
    trained_vqvae_lr = 2.5e-05
    trained_vqvae_bs = 8

    # Loading the trained VQ-VAE
    args.checkpoint_path = rf"./checkpoints/vqgan_{args.codebook_optimization}_epoch{trained_vqvae_epochs}_{args.codebook_bits}bit_lr{trained_vqvae_lr}_bs{trained_vqvae_bs}.pt"
    print(f"Trained VQ-VAE checkpoint path: {args.checkpoint_path}")

    train_transformer = TrainTransformer(args)