import os
import albumentations
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch

# --------------------------------------------- #
#                  Data Utils
# --------------------------------------------- #

class ImagePaths(Dataset):
    def __init__(self, path, size=None):
        self.size = size

        self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self._length = len(self.images)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example


def load_data(args):
    train_data = ImagePaths(args.dataset_path, size=256)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    return train_loader


# --------------------------------------------- #
#                  Module Utils
#            for Encoder, Decoder etc.
# --------------------------------------------- #

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    full_sample = images["full_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(full_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()


def set_temperature(epoch, max_epoch=100, start_temp=1.0, min_temp=0.1):
    """
    Sets temperature based on exponential decay from start_temp to min_temp over max_epoch.
    """
    decay_rate = (min_temp / start_temp) ** (1.0 / max_epoch)
    temperature = max(start_temp * (decay_rate ** epoch), min_temp)
    return temperature

def cbr_new(codebooks_used, codebooks, num_batches, discarding_threshold, eps, embedding_dim):
    with torch.no_grad():
        unused_indices = torch.where((codebooks_used.cpu() / num_batches) < discarding_threshold)[0]
        used_indices = torch.where((codebooks_used.cpu() / num_batches) >= discarding_threshold)[0]

        unused_count = unused_indices.shape[0]
        used_probs = codebooks_used[used_indices] / torch.sum(codebooks_used[used_indices])
        sampled_indices = torch.from_numpy(np.random.choice(used_indices.numpy(), size=(unused_count,), p=used_probs.numpy()))
        used_codebooks = codebooks[sampled_indices].clone()

        codebooks[unused_indices] *= 0
        codebooks[unused_indices] += used_codebooks[range(unused_count)] + eps * torch.randn((unused_count, embedding_dim), device=used_codebooks.device).clone()

        print(f'\n************* Replaced ' + str(unused_count) + f' codewords *************')
        codebooks_used[:] = 0

    return codebooks_used, codebooks

def cbr_old(codebooks_used, codebooks, num_batches, discarding_threshold, eps, embedding_dim):
        with torch.no_grad():
            unused_indices = torch.where((codebooks_used.cpu() / num_batches) < discarding_threshold)[0]
            used_indices = torch.where((codebooks_used.cpu() / num_batches) >= discarding_threshold)[0]

            unused_count = unused_indices.shape[0]
            used_count = used_indices.shape[0]

            used = codebooks[used_indices].clone()
            if used_count < unused_count:
                used_codebooks = used.repeat(int((unused_count / (used_count + eps)) + 1), 1)
                used_codebooks = used_codebooks[torch.randperm(used_codebooks.shape[0])]
            else:
                used_codebooks = used

            codebooks[unused_indices] *= 0
            codebooks[unused_indices] += used_codebooks[range(unused_count)] + eps * torch.randn((unused_count, embedding_dim), device=used_codebooks.device).clone()

            print(f'\n************* Replaced ' + str(unused_count) + f' codewords *************')
            codebooks_used[:] = 0

        return codebooks_used, codebooks