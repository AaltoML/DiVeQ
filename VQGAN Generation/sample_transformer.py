import os
import argparse
import torch
from transformer import VQGANTransformer
from tqdm import tqdm
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description="VQGAN")
parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension')
parser.add_argument('--image-size', type=int, default=256, help='Image height and width')
parser.add_argument('--bitrate', type=int, default=9, help='VQ bitrate')
parser.add_argument('--codebook_optimization', type=str, default='diveq', help='method used in VQ-VAE to optimize VQ codebook: "ste", "ema", "rt", "gumbel_softmax", "nsvq", "diveq", "sfdiveq", "diveq_detach", "sfdiveq_detach" ')
parser.add_argument("--variance", type=float, default=0.01, help="variance of the DIVEQ and SFDIVEQ")
parser.add_argument("--start_sfvq", type=int, default=2, help="The epoch to start quantizing the latent with SFDIVEQ (only used when using SFDIVEQ or SFDIVEQ_DETACH for training)")
parser.add_argument("--cbr_method", type=str, default='new', help="Codebook replacement technique to use")
parser.add_argument('--sfvq_dithered_inference', dest='sfvq_dithered_inference', action='store_true',help='Enable SFDIVEQ dithered inference, i.e., quantizes to the lines connecting subsequent codewords')
parser.add_argument('--no_sfvq_dithered_inference', dest='sfvq_dithered_inference', action='store_false',help='Disable SFDIVEQ dithered inference, i.e., quantizes only to the codewords')
parser.set_defaults(sfvq_dithered_inference=False)
parser.add_argument('--discard_threshold', type=float, default=0.01, help='Threshold (percentage) for discarding unused codebook vectors')
parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss coefficient')
parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images')
parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data.')
parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to trained VQ-VAE checkpoint')
parser.add_argument('--device', type=str, default="cuda")

parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')

parser.add_argument('--gen_path', type=str, help='path to save generations')

args = parser.parse_args()

dataset_name = 'afhq'

# Configs from trained VQ-VAE in first stage training
trained_vqvae_epochs = 100
trained_vqvae_lr = 2.5e-05
trained_vqvae_bs = 8

# Configs from trained VQ-VAE in first stage training
trained_transformer_epochs = 500
trained_transformer_bs = 32

if dataset_name == 'celeba':
    n = 30000
elif dataset_name == 'afhq':
    n = 15803
elif dataset_name in ['ffhq', 'lsun_bedroom', 'lsun_church']:
    n = 56000

if args.bitrate == 8:
    top_k = 75
elif args.bitrate == 9:
    top_k = 150
else:
    top_k = 300

os.makedirs(f"generations/{args.codebook_optimization}/{args.bitrate}bit/var{args.variance}", exist_ok=True)

# Loading the trained VQ-VAE
args.checkpoint_path = fr"./checkpoints/vqgan_{args.codebook_optimization}_epoch{trained_vqvae_epochs}_{args.bitrate}bit_var{args.variance}_lr{trained_vqvae_lr}_bs{trained_vqvae_bs}.pt"

transformer = VQGANTransformer(args).to("cuda")
# Loading the trained transformer
transformer.load_state_dict(torch.load(os.path.join(f"./checkpoints/", f"transformer_{args.codebook_optimization}_epoch{trained_transformer_epochs}_{args.bitrate}bit_bs{trained_transformer_bs}_var{args.variance}.pt")))

print(f"Bitrate = {args.bitrate} | TOP_K = {top_k}")

k=1
for i in tqdm(range(n)):
    start_indices = torch.zeros((k, 0)).long().to("cuda")
    sos_tokens = torch.ones(start_indices.shape[0], 1) * 0
    sos_tokens = sos_tokens.long().to("cuda")
    sample_indices = transformer.sample(start_indices, sos_tokens, steps=256, top_k=top_k)
    sampled_imgs = transformer.z_to_image(sample_indices)
    sampled_imgs = (sampled_imgs + 1) / 2

    clamped_img = torch.clamp(sampled_imgs, min=0, max=1) * 255
    img_final = (clamped_img.cpu().detach().numpy()).astype(np.uint8)

    im = Image.fromarray(img_final[0].transpose(1,2,0))
    im.save(f"{args.gen_path}/sample_{i+1}.png")