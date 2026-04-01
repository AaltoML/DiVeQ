import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from discriminator import Discriminator
from lpips import LPIPS
from vqgan import VQGAN
from utils import load_data, weights_init
import kagglehub

from torch import autocast
from torch.cuda.amp import GradScaler
from utils import set_temperature

class TrainVQGAN:
    def __init__(self, args):
        self.vqgan = VQGAN(args).to(device=args.device)
        self.discriminator = Discriminator(args).to(device=args.device)

        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)

        self.prepare_training()

        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.vq.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=lr)
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        return opt_vq, opt_disc

    @staticmethod
    def prepare_training():
        os.makedirs(f"results/{args.codebook_optimization}", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)


    def train(self, args):
        train_dataset = load_data(args)
        scaler = GradScaler()
        num_eval_samples = 5
        milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
        scheduler_vq = torch.optim.lr_scheduler.MultiStepLR(self.opt_vq, milestones=milestones, gamma=0.5)
        scheduler_disc = torch.optim.lr_scheduler.MultiStepLR(self.opt_disc, milestones=milestones, gamma=0.5)

        for epoch in range(args.epochs):

            if args.codebook_optimization == 'gumbel_softmax':
                current_temp = set_temperature(epoch, max_epoch=args.epochs)
                self.vqgan.vq.temperature = current_temp # Temperature annealing for GumbelSoftmax

            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    self.discriminator.zero_grad()
                    self.opt_disc.zero_grad()
                    with autocast(device_type='cuda', dtype=torch.float16):
                        imgs = imgs.to(device=args.device)

                        with torch.no_grad():
                            decoded_images, _, _, _ = self.vqgan(imgs)
                        disc_real = self.discriminator(imgs)
                        disc_fake = self.discriminator(decoded_images.detach())
                        d_loss_real = torch.mean(F.relu(1. - disc_real))
                        d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                        if (epoch + 1) > int(args.epochs / 2):
                            disc_factor = args.disc_factor
                        else:
                            disc_factor = 0.0
                        gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)
                    scaler.scale(gan_loss).backward()
                    scaler.step(self.opt_disc)
                    scaler.update()

                    self.vqgan.zero_grad()
                    self.opt_vq.zero_grad()
                    with autocast(device_type='cuda', dtype=torch.float16):
                        decoded_images, indices, perplexity, vq_loss = self.vqgan(imgs)
                        perceptual_loss = self.perceptual_loss(imgs.contiguous(), decoded_images.contiguous())
                        rec_loss = torch.abs(imgs.contiguous() - decoded_images.contiguous())
                        disc_fake = self.discriminator(decoded_images)
                        perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                        perceptual_rec_loss = perceptual_rec_loss.mean()
                        g_loss = -torch.mean(disc_fake)

                        lbd = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                        if args.codebook_optimization in ['ste', 'ema', 'gumbel_softmax', 'rt']:
                            vq_loss = perceptual_rec_loss + vq_loss + (disc_factor * lbd * g_loss)
                        else:
                            vq_loss = perceptual_rec_loss + (disc_factor * lbd * g_loss)

                        recon_loss_mean = rec_loss.mean()
                        perceptual_loss_mean = perceptual_loss.mean()

                    scaler.scale(vq_loss).backward()
                    scaler.step(self.opt_vq)
                    scaler.update()

                    pbar.set_postfix(
                        Epoch=epoch + 1,
                        Recon=np.round(recon_loss_mean.cpu().detach().numpy().item(),5),
                        Percep=np.round(perceptual_loss_mean.cpu().detach().numpy().item(), 5),
                        GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 5),
                        Perplexity=perplexity,
                    )

                    pbar.update(0)

            scheduler_vq.step()
            scheduler_disc.step()

            with torch.no_grad():
                real_fake_images = torch.cat((imgs.add(1).mul(0.5)[:num_eval_samples],torch.clamp(decoded_images.add(1).mul(0.5), min=0.0, max=1)[:num_eval_samples]))
                vutils.save_image(real_fake_images, os.path.join(f"results/{args.codebook_optimization}",f"epoch{epoch+1}.jpg"),nrow=num_eval_samples)

            if (epoch+1) % 10 == 0:
                torch.save(self.vqgan.state_dict(), os.path.join("checkpoints", f"vqgan_{args.codebook_optimization}_epoch{epoch+1}_{args.codebook_bits}bit_lr{args.learning_rate}_bs{args.batch_size}.pt"))
                torch.save(self.discriminator.state_dict(),os.path.join("checkpoints", f"discriminator_{args.codebook_optimization}_epoch{epoch+1}_{args.codebook_bits}bit_lr{args.learning_rate}_bs{args.batch_size}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--embedding_dim', type=int, default=256, help='Latent dimension')
    parser.add_argument('--image_size', type=int, default=256, help='Image height and width')
    parser.add_argument('--codebook_bits', type=int, default=10,
                        help='number of bits per codebook. No. of codewords in the codebook'
                             ' equals to 2^codebook_bits')
    parser.add_argument("--codebook_optimization", type=str, default='diveq',
                        help='method to optimize VQ codebook: options -> "ste", "ema", "rt",'
                             ' "gumbel_softmax", "nsvq", "diveq", "sfdiveq", "diveq_detach",'
                             ' "sfdiveq_detach", "residual_diveq", "residual_sfdiveq",'
                             ' "product_diveq", "product_sfdiveq" ')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch-size', type=int, default=8, help='Input batch size to train the VQ-VAE')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the VQ-VAE')
    parser.add_argument('--learning-rate', type=float, default=2.5e-05, help='Learning rate to train the VQ-VAE')
    parser.add_argument('--disc-factor', type=float, default=0.2)
    parser.add_argument('--rec-loss-factor', type=float, default=1.0, help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1.0, help='Weighting factor for perceptual loss.')

    args = parser.parse_args()

    # path to dataset directory
    os.makedirs("data_dir", exist_ok=True)
    data_path = kagglehub.dataset_download("badasstechie/celebahq-resized-256x256", output_dir="./data_dir")
    args.dataset_path = os.path.join(data_path, "celeba_hq_256")

    # device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_vqgan = TrainVQGAN(args)