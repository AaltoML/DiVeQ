import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from vq import STE, EMA, RT, Gumbel_Softmax, NSVQ, DIVEQ, SFDIVEQ, DIVEQ_DETACH, SFDIVEQ_DETACH
import numpy as np

class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        bitrate = args.bitrate
        self.num_codewords = int(2 ** bitrate)
        self.encoder = Encoder(double_z=False, z_channels=256, resolution=256, in_channels=3, out_ch=3, ch=128,
                               ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16], dropout=0.0).to(device=args.device)
        self.decoder = Decoder(double_z=False, z_channels=256, resolution=256, in_channels=3, out_ch=3, ch=128,
                               ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16], dropout=0.0).to(device=args.device)

        if args.codebook_optimization == 'ste':
            self.vq = STE(args).to(device=args.device)
        elif args.codebook_optimization == 'rt':
            self.vq = RT(args).to(device=args.device)
        elif args.codebook_optimization == 'nsvq':
            self.vq = NSVQ(args).to(device=args.device)
        elif args.codebook_optimization == 'diveq':
            self.vq = DIVEQ(args).to(device=args.device)
        elif args.codebook_optimization == 'diveq_detach':
            self.vq = DIVEQ_DETACH(args).to(device=args.device)
        elif args.codebook_optimization == 'sfdiveq_detach':
            self.vq = SFDIVEQ_DETACH(args).to(device=args.device)
        elif args.codebook_optimization == 'sfdiveq':
            self.vq = SFDIVEQ(args).to(device=args.device)
        elif args.codebook_optimization == 'gumbel_softmax':
            self.vq = Gumbel_Softmax(args).to(device=args.device)
        elif args.codebook_optimization == 'ema':
            self.vq = EMA(args).to(device=args.device)

        if args.codebook_optimization == 'gumbel_softmax':
            self.quant_conv = nn.Conv2d(args.latent_dim, self.num_codewords, 1).to(device=args.device)
        else:
            self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)

        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)

        self.latent_list = []
        self.counter = 1

    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss, perplexity = self.vq(quant_conv_encoded_images)
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(post_quant_conv_mapping)

        return decoded_images, codebook_indices, q_loss, perplexity

    def forward_sfvq(self, imgs, epoch, batch_idx, num_batches, args):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        if epoch >= args.start_sfvq:
            codebook_mapping, codebook_indices, q_loss, perplexity = self.vq(quant_conv_encoded_images)
            post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
            decoded_images = self.decoder(post_quant_conv_mapping)
            return decoded_images, codebook_indices, q_loss, perplexity
        else:
            if (epoch == args.start_sfvq - 1) and (batch_idx >= num_batches - 50):
                z_temp = (quant_conv_encoded_images.permute(0, 2, 3, 1).contiguous()).view(-1, args.latent_dim)
                self.latent_list.append(z_temp.cpu().detach())

                if batch_idx == num_batches:
                    stacked = torch.stack(self.latent_list[0:-1], dim=0).reshape(-1, args.latent_dim)
                    latents = torch.cat((stacked, self.latent_list[-1]), dim=0)

                    initial_cb = torch.zeros((self.num_codewords, args.latent_dim), dtype=torch.float32)
                    hop_size = int(np.floor(latents.shape[0] / self.num_codewords))

                    for jj in range(self.num_codewords):
                        initial_cb[jj] = torch.mean(latents[jj * hop_size:(jj + 1) * hop_size], dim=0)

                    self.vq.codebook.data = initial_cb.to(args.device)

            post_quant_conv_mapping = self.post_quant_conv(quant_conv_encoded_images)
            decoded_images = self.decoder(post_quant_conv_mapping)
            return decoded_images

    def encode(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        return quant_conv_encoded_images

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.conv_out
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        lda = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        lda = torch.clamp(lda, 0, 1e4).detach()
        return 0.8 * lda

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))