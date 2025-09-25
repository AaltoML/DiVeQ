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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from torch import autocast
from torch.cuda.amp import GradScaler
from utils import set_temperature

class TrainVQGAN:
    def __init__(self, args):
        num_codewords = int(2**args.bitrate)
        self.vqgan = VQGAN(args).to(device=args.device)
        self.discriminator = Discriminator(args).to(device=args.device)

        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)
        self.codebook_used_indices = torch.zeros(num_codewords, dtype=torch.int32)

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
        steps_per_epoch = len(train_dataset)
        scaler = GradScaler()
        batch_counter = 0
        total_num_batches = args.epochs * steps_per_epoch
        cb_hist_list = []
        num_eval_samples = 5
        num_codewords = int(2**args.bitrate)
        milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
        scheduler_vq = torch.optim.lr_scheduler.MultiStepLR(self.opt_vq, milestones=milestones, gamma=0.5)
        scheduler_disc = torch.optim.lr_scheduler.MultiStepLR(self.opt_disc, milestones=milestones, gamma=0.5)

        if args.codebook_optimization in ['sfdiveq', 'sfdiveq_detach']:
            os.makedirs(f"hist_{args.codebook_optimization}", exist_ok=True)
            pdf_file = PdfPages(f'./hist_{args.codebook_optimization}/hist_{args.bitrate}bit_var{args.variance}_lr{args.learning_rate}_bs{args.batch_size}.pdf')

        for epoch in range(args.epochs):
            cb_hist = np.zeros((num_codewords - 1,))
            if args.codebook_optimization == 'gumbel_softmax':
                current_temp = set_temperature(epoch, max_epoch=args.epochs)
                self.vqgan.vq.temperature = current_temp

            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    batch_counter += 1
                    self.discriminator.zero_grad()
                    self.opt_disc.zero_grad()
                    with autocast(device_type='cuda', dtype=torch.float16):
                        imgs = imgs.to(device=args.device)

                        with torch.no_grad():
                            if args.codebook_optimization in ['sfdiveq', 'sfdiveq_detach']:
                                if epoch >= args.start_sfvq:
                                    decoded_images, _, _, _ = self.vqgan.forward_sfvq(imgs, epoch, i + 1, steps_per_epoch,args)
                                else:
                                    decoded_images = self.vqgan.forward_sfvq(imgs, epoch, i + 1, steps_per_epoch,args)
                            else:
                                decoded_images, _, _, _ = self.vqgan.forward(imgs)
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
                        if args.codebook_optimization in ['sfdiveq', 'sfdiveq_detach']:
                            if epoch >= args.start_sfvq:
                                decoded_images, min_encoding_indices, q_loss, perplexity = self.vqgan.forward_sfvq(imgs, epoch, i+1, steps_per_epoch, args)
                            else:
                                decoded_images = self.vqgan.forward_sfvq(imgs, epoch, i+1, steps_per_epoch, args)
                        else:
                            decoded_images, min_encoding_indices, q_loss, perplexity = self.vqgan.forward(imgs)
                            with torch.no_grad():
                                self.codebook_used_indices[min_encoding_indices.cpu()] += 1

                        perceptual_loss = self.perceptual_loss(imgs.contiguous(), decoded_images.contiguous())
                        rec_loss = torch.abs(imgs.contiguous() - decoded_images.contiguous())
                        disc_fake = self.discriminator(decoded_images)
                        perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                        perceptual_rec_loss = perceptual_rec_loss.mean()
                        g_loss = -torch.mean(disc_fake)

                        lbd = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                        if args.codebook_optimization in ['ste', 'ema', 'gumbel_softmax', 'rt']:
                            vq_loss = perceptual_rec_loss + q_loss + (disc_factor * lbd * g_loss)
                        elif args.codebook_optimization in ['diveq', 'nsvq', 'diveq_detach', 'sfdiveq', 'sfdiveq_detach']:
                            vq_loss = perceptual_rec_loss + (disc_factor * lbd * g_loss)

                        recon_loss_mean = rec_loss.mean()
                        perceptual_loss_mean = perceptual_loss.mean()

                    scaler.scale(vq_loss).backward()
                    scaler.step(self.opt_vq)
                    scaler.update()

                    if args.codebook_optimization in ['sfdiveq', 'sfdiveq_detach']:
                        if epoch >= args.start_sfvq:
                            cb_hist[torch.unique(min_encoding_indices).cpu().detach().numpy()] += 1
                            pbar.set_postfix(
                                Epoch = epoch+1,
                                Recon=np.round(recon_loss_mean.cpu().detach().numpy().item(), 5),
                                Percep=np.round(perceptual_loss_mean.cpu().detach().numpy().item(), 5),
                                GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 5),
                                Perplexity=np.round(perplexity.cpu().detach().numpy().item(), 2),
                            )
                        else:
                            zero_tensor = torch.tensor(0.0, dtype=torch.float32)
                            pbar.set_postfix(
                                Epoch=epoch + 1,
                                Recon=np.round(recon_loss_mean.cpu().detach().numpy().item(), 5),
                                Percep=np.round(perceptual_loss_mean.cpu().detach().numpy().item(), 5),
                                GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 5),
                            )
                    else:
                        pbar.set_postfix(
                            Epoch=epoch + 1,
                            Recon=np.round(recon_loss_mean.cpu().detach().numpy().item(), 5),
                            Percep=np.round(perceptual_loss_mean.cpu().detach().numpy().item(), 5),
                            GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 5),
                            Perplexity=np.round(perplexity.cpu().detach().numpy().item(), 2),
                        )

                    if args.codebook_optimization in ['diveq', 'nsvq', 'diveq_detach', 'ste', 'ema', 'gumbel_softmax','rt']:
                        if args.codebook_replacement:
                            if (batch_counter % 50 == 0) & (batch_counter <= 5000):
                                self.vqgan.vq.codebook_replacement(50)
                            if (batch_counter % args.replacement_num_batches == 0) & (5000 < batch_counter <= total_num_batches - 3000):
                                self.vqgan.vq.codebook_replacement(args.replacement_num_batches)

                    pbar.update(0)

            cb_hist_list.append(cb_hist)
            scheduler_vq.step()
            scheduler_disc.step()

            with torch.no_grad():
                real_fake_images = torch.cat((imgs.add(1).mul(0.5)[:num_eval_samples],torch.clamp(decoded_images.add(1).mul(0.5), min=0.0, max=1)[:num_eval_samples]))
                vutils.save_image(real_fake_images, os.path.join(f"results/{args.codebook_optimization}",f"e{epoch+1}_lr{args.learning_rate}_bs{args.batch_size}.jpg"),nrow=num_eval_samples)

            if (epoch+1) % 50 == 0:
                torch.save(self.vqgan.state_dict(), os.path.join("checkpoints", f"vqgan_{args.codebook_optimization}_epoch{epoch+1}_{args.bitrate}bit_var{args.variance}_lr{args.learning_rate}_bs{args.batch_size}.pt"))
                torch.save(self.discriminator.state_dict(),os.path.join("checkpoints", f"discriminator_{args.codebook_optimization}_epoch{epoch+1}_{args.bitrate}bit_var{args.variance}_lr{args.learning_rate}_bs{args.batch_size}.pt"))


        if args.codebook_optimization in ['sfdiveq', 'sfdiveq_detach']:
            for i in range(len(cb_hist_list)):
                num_bars = num_codewords - 1
                histogram = np.log10(cb_hist_list[i] + 1)
                fig = plt.figure(figsize=(10, 6))
                plt.bar(np.arange(1, num_bars + 1), height=histogram, width=1)
                plt.title(f'SFVQ Codebook Usage Histogram | Epoch={i + 1}')
                pdf_file.savefig(fig, bbox_inches='tight')
            pdf_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width')
    parser.add_argument('--bitrate', type=int, default=10, help='VQ bitrate')
    parser.add_argument('--codebook_optimization', type=str, default='diveq', help='method to optimize VQ codebook: "ste", "ema", "rt", "gumbel_softmax", "nsvq", "diveq", "sfdiveq", "diveq_detach", "sfdiveq_detach" ')
    parser.add_argument('--codebook_replacement', type=bool, default=True, help='Whether to apply codebook replacement')
    parser.add_argument('--replacement_num_batches', type=int, default=300, help='Number of batches after which to apply codebook replacement')
    parser.add_argument('--discard_threshold', type=float, default=0.01, help='Threshold (percentage) for discarding unused codebook vectors')
    parser.add_argument("--variance", type=float, default=0.01, help="variance of the DIVEQ and SFDIVEQ")
    parser.add_argument("--cbr_method", type=str, default='new', help="Codebook replacement technique to use")
    parser.add_argument("--start_sfvq", type=int, default=2, help="The epoch to start quantizing the latent with SFDIVEQ (only used when using SFDIVEQ or SFDIVEQ_DETACH for training)")
    parser.add_argument('--eps', type=float, default=1e-9, help='Epsilon')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss coefficient')
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
    args.dataset_path = r"path/to/dataset/directory"

    train_vqgan = TrainVQGAN(args)
