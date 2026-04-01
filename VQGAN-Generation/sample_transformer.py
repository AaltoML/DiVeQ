import os
import argparse
import torch
from transformer import VQGANTransformer
from tqdm import tqdm
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description="Sampling")
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

parser.add_argument('--warmup_steps', type=int, default=10000, help='Number of training iterations to warmup the learning rate')
parser.add_argument('--pkeep_start', type=float, default=0.5, help='Initial value of pkeep that is the probability of keeping token indices during training')
parser.add_argument('--pkeep_end', type=float, default=0.95, help='Final value of pkeep that is the probability of keeping token indices during training')
parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')

args = parser.parse_args()

# device
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configs from trained VQ-VAE in first stage training
trained_vqvae_epochs = 100
trained_vqvae_lr = 2.5e-05
trained_vqvae_bs = 8

# Configs from trained Transformer in second stage training
trained_transformer_epochs = 500
trained_transformer_lr = 4.5e-05
trained_transformer_bs = 32

# Number of images to generate
num_samples = 5

if args.codebook_bits == 8:
    top_k = 75
elif args.codebook_bits == 9:
    top_k = 150
elif args.codebook_bits >= 10:
    top_k = 300

os.makedirs(f"generations", exist_ok=True)
generation_path = f"./generations"

# Loading the trained VQ-VAE
args.checkpoint_path = fr"./checkpoints/vqgan_{args.codebook_optimization}_epoch{trained_vqvae_epochs}_{args.codebook_bits}bit_lr{trained_vqvae_lr}_bs{trained_vqvae_bs}.pt"

# Loading the trained transformer
transformer = VQGANTransformer(args).to(args.device)
transformer.load_state_dict(torch.load(os.path.join(f"./checkpoints/", f"transformer_{args.codebook_optimization}_epoch{trained_transformer_epochs}_{args.codebook_bits}bit_lr{trained_transformer_lr}_bs{trained_transformer_bs}.pt")))

k=1
for i in tqdm(range(num_samples)):
    start_indices = torch.zeros((k, 0)).long().to(args.device)
    sos_tokens = torch.ones(start_indices.shape[0], 1) * 0
    sos_tokens = sos_tokens.long().to(args.device)
    sample_indices = transformer.sample(start_indices, sos_tokens, steps=256, top_k=top_k)
    sampled_imgs = transformer.z_to_image(sample_indices)
    sampled_imgs = (sampled_imgs + 1) / 2

    clamped_img = torch.clamp(sampled_imgs, min=0, max=1) * 255
    img_final = (clamped_img.cpu().detach().numpy()).astype(np.uint8)

    im = Image.fromarray(img_final[0].transpose(1,2,0))
    im.save(f"{generation_path}/sample_{i+1}.png")