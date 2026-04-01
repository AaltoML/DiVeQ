import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import Dataset_Custom
import argparse
from tqdm import tqdm
import os
from lpips import LPIPS
from torchvision import utils as vutils
import kagglehub

from model import Encoder, Decoder
from vq import STE, EMA, RT, GumbelSoftmax, NSVQ

from diveq import DIVEQ
from sf_diveq import SFDIVEQ
from residual_diveq import ResidualDIVEQ
from residual_sf_diveq import ResidualSFDIVEQ
from product_diveq import ProductDIVEQ
from product_sf_diveq import ProductSFDIVEQ
from diveq_detach import DIVEQDetach
from sf_diveq_detach import SFDIVEQDetach

parser = argparse.ArgumentParser(description="VQVAE")
parser.add_argument("--size", type=int, default=256, help='Image height and width')
parser.add_argument('--embedding_dim', type=int, default=512, help='Latent dimension')
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=5.5e-04)
parser.add_argument('--codebook_bits', type=int, default=10,
                    help='number of bits per codebook. No. of codewords in the codebook'
                         ' equals to 2^codebook_bits')
parser.add_argument("--codebook_optimization", type=str, default='diveq',
                    help='method to optimize VQ codebook: options -> "ste", "ema", "rt",'
                         ' "gumbel_softmax", "nsvq", "diveq", "sfdiveq", "diveq_detach",'
                         ' "sfdiveq_detach", "residual_diveq", "residual_sfdiveq",'
                         ' "product_diveq", "product_sfdiveq" ')
parser.add_argument("--num_codebooks", type=int, default=4,
                    help="No. of codebooks used for Residual VQ or Product VQ")
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument("--data_path", type=str, help="path to training set directory")

args = parser.parse_args()

# path to dataset directory
os.makedirs("data_dir", exist_ok=True)
data_path = kagglehub.dataset_download("badasstechie/celebahq-resized-256x256", output_dir="./data_dir")
args.data_path = os.path.join(data_path, "celeba_hq_256")

# device
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Creating the VQ-VAE model ----------------
class Model(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, num_codebooks, codebook_optimization):
        super(Model, self).__init__()

        self.codebook_optimization = codebook_optimization
        self.embedding_dim = embedding_dim

        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        if self.codebook_optimization == 'gumbel_softmax':
            self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                  out_channels=num_embeddings, kernel_size=1, stride=1)
        else:
            self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                  out_channels=embedding_dim, kernel_size=1, stride=1)

        if self.codebook_optimization == 'diveq':
            self.vq = DIVEQ(num_embeddings, embedding_dim)
        elif self.codebook_optimization == 'sfdiveq':
            self.vq = SFDIVEQ(num_embeddings, embedding_dim)
        elif self.codebook_optimization == 'diveq_detach':
            self.vq = DIVEQDetach(num_embeddings, embedding_dim)
        elif self.codebook_optimization == 'sfdiveq_detach':
            self.vq = SFDIVEQDetach(num_embeddings, embedding_dim)
        elif self.codebook_optimization == 'ste':
            self.vq = STE(num_embeddings, embedding_dim)
        elif self.codebook_optimization == 'rt':
            self.vq = RT(num_embeddings, embedding_dim)
        elif self.codebook_optimization == 'nsvq':
            self.vq = NSVQ(num_embeddings, embedding_dim)
        elif self.codebook_optimization == 'gumbel_softmax':
            self.vq = GumbelSoftmax(num_embeddings, embedding_dim)
        elif self.codebook_optimization == 'ema':
            self.vq = EMA(num_embeddings, embedding_dim)

        # Residual VQ or Product VQ which use multiple codebooks for quantization
        elif self.codebook_optimization == 'residual_diveq':
            self.vq = ResidualDIVEQ(num_embeddings, embedding_dim, num_codebooks)
        elif self.codebook_optimization == 'residual_sfdiveq':
            self.vq = ResidualSFDIVEQ(num_embeddings, embedding_dim, num_codebooks)
        elif self.codebook_optimization == 'product_diveq':
            self.vq = ProductDIVEQ(num_embeddings, embedding_dim, num_codebooks)
        elif self.codebook_optimization == 'product_sfdiveq':
            self.vq = ProductSFDIVEQ(num_embeddings, embedding_dim, num_codebooks)

        print(f"VQ method: {self.vq}")

        self._decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers,
                                num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)

        z_permute = z.permute(0, 2, 3, 1).contiguous()  # Convert BCHW -> BHWC
        z_shape = z_permute.shape
        z_flat = z_permute.view(-1, self.embedding_dim)  # Flatten the input

        if self.codebook_optimization in ['ste', 'ema', 'gumbel_softmax', 'rt']:
            if self.codebook_optimization == 'gumbel_softmax':
                quantized, indices, perplexity, loss = self.vq(z)
            else:
                quantized, indices, perplexity, loss = self.vq(z_flat)
                quantized = (quantized.view(z_shape)).permute(0, 3, 1, 2).contiguous()  # Convert BHWC -> BCHW
        else:
            quantized, indices, perplexity = self.vq(z_flat)
            quantized = (quantized.view(z_shape)).permute(0, 3, 1, 2).contiguous()  # Convert BHWC -> BCHW
            loss = torch.tensor(0.0, dtype=torch.float32, device=x.device) # as a placeholder

        x_recon = self._decoder(quantized)

        return x_recon, indices, perplexity, loss

    def inference(self,x):
        with torch.no_grad():
            z = self._encoder(x)
            z = self._pre_vq_conv(z)
            if self.codebook_optimization == 'gumbel_softmax':
                quantized, indices, perplexity = self.vq.inference(z)
            else:
                z_permute = z.permute(0, 2, 3, 1).contiguous()  # Convert BCHW -> BHWC
                z_shape = z_permute.shape
                z_flat = z_permute.view(-1, self.embedding_dim)  # Flatten the input
                quantized, indices, perplexity = self.vq.inference(z_flat)
                quantized = (quantized.view(z_shape)).permute(0, 3, 1, 2).contiguous()  # Convert BHWC -> BCHW

            x_recon = self._decoder(quantized)
            return x_recon

# ---------------- Some configurations for the model and optimizer ----------------
num_hiddens = 256
num_residual_hiddens = 128
num_residual_layers = 6

os.makedirs("checkpoints", exist_ok=True)
os.makedirs(f"results/{args.codebook_optimization}", exist_ok=True)
num_eval_samples = 5

perceptual_loss = LPIPS().eval().to(device=args.device)

num_embeddings = int(2**args.codebook_bits)
milestones = [int(args.epochs*0.4), int(args.epochs*0.7)]

dataset = Dataset_Custom(args)
training_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=1, drop_last=True)

# Instantiate the vq-vae model
model = Model(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings,
              args.embedding_dim, args.num_codebooks, args.codebook_optimization).to(args.device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=False)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

def set_temperature(epoch_idx, max_epoch=100, start_temp=1.0, min_temp=0.1):
    """
    Sets temperature based on exponential decay from start_temp to min_temp over max_epoch.
    """
    decay_rate = (min_temp / start_temp) ** (1.0 / max_epoch)
    temperature = max(start_temp * (decay_rate ** epoch_idx), min_temp)
    return temperature

model.train()
# ---------------- Training Loop ----------------
for epoch in range(args.epochs):

    if args.codebook_optimization == 'gumbel_softmax':
        current_temp = set_temperature(epoch, max_epoch=args.epochs)
        model.vq.temperature = current_temp # Temperature annealing for GumbelSoftmax

    with tqdm(range(len(training_loader))) as pbar:
        for i, data in zip(pbar, training_loader):
            data = data.to(args.device)
            optimizer.zero_grad()

            data_recon, indices, perplexity, vq_loss = model(data)

            recon_error = F.mse_loss(data_recon, data)
            perceptual_error = torch.mean(perceptual_loss(data, data_recon.clamp(-1,1)))

            if args.codebook_optimization in ['ste', 'ema', 'gumbel_softmax', 'rt']:
                loss = recon_error + (1 * perceptual_error) + vq_loss
            else:
                loss = recon_error + (1 * perceptual_error)

            loss.backward()

            pbar.set_postfix(Epoch=epoch + 1, Recon_Loss=f"{recon_error.item():.6f}",
                             Perc_Loss=f"{perceptual_error.item():.6f}",
                             Perplexity=f"{perplexity}")

            optimizer.step()
            pbar.update(0)

    scheduler.step() # for learning rate scheduling

    # Visualization of reconstructed images (assumption: values of images are in the range of [-1,1])
    with torch.no_grad():
        images = data[0:num_eval_samples]
        recon_images = model.inference(images)
        concat_images = torch.cat((torch.clamp(images.add(1).mul(0.5), min=0.0, max=1.0)
                       , torch.clamp(recon_images.add(1).mul(0.5), min=0.0, max=1.0)))
        vutils.save_image(concat_images, os.path.join(f"results/{args.codebook_optimization}",
                                              f"epoch{epoch + 1}.jpg"), nrow=num_eval_samples)

    # Save the model
    if (epoch + 1) % 1 == 0:
        torch.save(model.state_dict(),f"checkpoints/"
              f"vqvae_{args.codebook_optimization}_epoch{str(epoch + 1)}_"
                          f"{args.codebook_bits}bit_bs{args.batch_size}_lr{args.lr}.pt")