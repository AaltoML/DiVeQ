import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal, uniform
from torch import einsum
from utils import cbr_new, cbr_old

class STE(nn.Module):
    def __init__(self, args):
        super(STE, self).__init__()
        bitrate = args.bitrate
        self.num_codewords = int(2**bitrate)
        self.latent_dim = args.latent_dim
        self.beta = args.beta
        self.eps = 1e-12
        self.cbr_method = args.cbr_method
        self.discarding_threshold = args.discard_threshold

        # codebook = uniform.Uniform(-1.0 / self.num_codewords, 1.0 / self.num_codewords).sample([self.num_codewords, self.latent_dim])
        codebook = torch.load(f'initial_cb_{bitrate}bit.pt', weights_only=True)
        self.codebook = torch.nn.Parameter(codebook, requires_grad=True)
        self.codebooks_used = torch.zeros(self.num_codewords, dtype=torch.int32)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        distances = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.codebook.t()))

        min_encoding_indices = torch.argmin(distances, dim=1)
        with torch.no_grad():
            self.codebooks_used[min_encoding_indices.cpu()] += 1

        # Compute Perplexity
        encodings = torch.zeros(min_encoding_indices.shape[0], self.num_codewords, device=z.device)
        encodings.scatter_(1, min_encoding_indices.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))

        z_q = self.codebook[min_encoding_indices].view(z.shape)

        loss = F.mse_loss(z.detach(),z_q) + (self.beta * F.mse_loss(z, z_q.detach()))

        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_encoding_indices, loss, perplexity

    def inference(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        distances = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.codebook ** 2, dim=1) - \
                    2 * (torch.matmul(z_flattened, self.codebook.t()))

        min_encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.codebook[min_encoding_indices].view(z.shape)
        return z_q.permute(0, 3, 1, 2), min_encoding_indices

    def codebook_replacement(self, num_batches):
        if self.cbr_method == 'new':
            self.codebooks_used, self.codebook = cbr_new(self.codebooks_used, self.codebook, num_batches,self.discarding_threshold, 1e-9, self.latent_dim)
        elif self.cbr_method == 'old':
            self.codebooks_used, self.codebook = cbr_old(self.codebooks_used, self.codebook, num_batches,self.discarding_threshold, 1e-9, self.latent_dim)


class RT(nn.Module):
    def __init__(self, args):
        super(RT, self).__init__()
        bitrate = args.bitrate
        self.num_codewords = int(2**bitrate)
        self.latent_dim = args.latent_dim
        self.beta = args.beta
        self.eps = 1e-12
        self.cbr_method = args.cbr_method
        self.discarding_threshold = args.discard_threshold

        # codebook = uniform.Uniform(-1.0 / self.num_codewords, 1.0 / self.num_codewords).sample([self.num_codewords, self.latent_dim])
        codebook = torch.load(f'initial_cb_{bitrate}bit.pt', weights_only=True)
        self.codebook = torch.nn.Parameter(codebook, requires_grad=True)
        self.codebooks_used = torch.zeros(self.num_codewords, dtype=torch.int32)

    @staticmethod
    def get_very_efficient_rotation(u, q, e):
        w = ((u + q) / torch.norm(u + q, dim=1, keepdim=True)).detach()
        e = e - 2 * torch.bmm(torch.bmm(e, w.unsqueeze(-1)), w.unsqueeze(1)) + 2 * torch.bmm(
            torch.bmm(e, u.unsqueeze(-1).detach()), q.unsqueeze(1).detach())
        return e

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        distances = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.codebook.t()))

        min_encoding_indices = torch.argmin(distances, dim=1)
        with torch.no_grad():
            self.codebooks_used[min_encoding_indices.cpu()] += 1

        # Compute Perplexity
        encodings = torch.zeros(min_encoding_indices.shape[0], self.num_codewords, device=z.device)
        encodings.scatter_(1, min_encoding_indices.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))

        quantized = self.codebook[min_encoding_indices]

        pre_norm_q = self.get_very_efficient_rotation(z_flattened / (torch.norm(z_flattened, dim=1, keepdim=True) + 1e-6),
                                                      quantized / (torch.norm(quantized, dim=1, keepdim=True) + 1e-6),
                                                      z_flattened.unsqueeze(1)).squeeze()
        q_tilde = pre_norm_q * (torch.norm(quantized, dim=1, keepdim=True) / (torch.norm(z_flattened, dim=1, keepdim=True) + 1e-6)).detach()

        z_q = q_tilde.view(z.shape)

        loss = F.mse_loss(z.detach(),quantized.view(z.shape)) + (self.beta * F.mse_loss(z, quantized.view(z.shape).detach()))

        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_encoding_indices, loss, perplexity

    def inference(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        distances = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.codebook ** 2, dim=1) - \
                    2 * (torch.matmul(z_flattened, self.codebook.t()))

        min_encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.codebook[min_encoding_indices].view(z.shape)
        return z_q.permute(0, 3, 1, 2), min_encoding_indices

    def codebook_replacement(self, num_batches):
        if self.cbr_method == 'new':
            self.codebooks_used, self.codebook = cbr_new(self.codebooks_used, self.codebook, num_batches,self.discarding_threshold, 1e-9, self.latent_dim)
        elif self.cbr_method == 'old':
            self.codebooks_used, self.codebook = cbr_old(self.codebooks_used, self.codebook, num_batches,self.discarding_threshold, 1e-9, self.latent_dim)


class NSVQ(nn.Module):
    def __init__(self, args):
        super(NSVQ, self).__init__()
        bitrate = args.bitrate
        self.num_codewords = int(2 ** bitrate)
        self.latent_dim = args.latent_dim
        self.eps = 1e-12
        self.cbr_method = args.cbr_method
        self.discarding_threshold = args.discard_threshold

        # codebook = uniform.Uniform(-1.0 / self.num_codewords, 1.0 / self.num_codewords).sample([self.num_codewords, self.latent_dim])
        codebook = torch.load(f'initial_cb_{bitrate}bit.pt', weights_only=True)
        self.codebook = torch.nn.Parameter(codebook, requires_grad=True)
        self.codebooks_used = torch.zeros(self.num_codewords, dtype=torch.int32)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        distances = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.codebook.t()))

        min_encoding_indices = torch.argmin(distances, dim=1)
        with torch.no_grad():
            self.codebooks_used[min_encoding_indices.cpu()] += 1

        # Compute Perplexity
        encodings = torch.zeros(min_encoding_indices.shape[0], self.num_codewords, device=z.device)
        encodings.scatter_(1, min_encoding_indices.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))

        z_hard_quantized = self.codebook[min_encoding_indices]

        random_vector = normal.Normal(0, 1).sample(z_flattened.shape).to(z.device)
        norm_quantization_residual = torch.linalg.norm(z_flattened - z_hard_quantized, dim=1, keepdim=True)
        norm_random_vector = torch.linalg.norm(random_vector, dim=1, keepdim=True)
        vq_error = (norm_quantization_residual / (norm_random_vector + self.eps)) * random_vector

        z_q = (z_flattened + vq_error).view(z.shape)

        z_q = z_q.permute(0, 3, 1, 2)

        loss = torch.tensor(0.0, dtype=torch.float32)

        return z_q, min_encoding_indices, loss, perplexity

    def inference(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        distances = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.codebook ** 2, dim=1) - \
                    2 * (torch.matmul(z_flattened, self.codebook.t()))

        min_encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.codebook[min_encoding_indices].view(z.shape)
        return z_q.permute(0, 3, 1, 2), min_encoding_indices

    def codebook_replacement(self, num_batches):
        if self.cbr_method == 'new':
            self.codebooks_used, self.codebook = cbr_new(self.codebooks_used, self.codebook, num_batches,self.discarding_threshold, 1e-9, self.latent_dim)
        elif self.cbr_method == 'old':
            self.codebooks_used, self.codebook = cbr_old(self.codebooks_used, self.codebook, num_batches,self.discarding_threshold, 1e-9, self.latent_dim)


class DIVEQ(nn.Module):
    def __init__(self, args):
        super(DIVEQ, self).__init__()
        bitrate = args.bitrate
        self.num_codewords = int(2 ** bitrate)
        self.latent_dim = args.latent_dim
        self.noise_var = args.variance
        self.eps = 1e-12
        self.cbr_method = args.cbr_method
        self.discarding_threshold = args.discard_threshold

        # codebook = uniform.Uniform(-1.0 / self.num_codewords, 1.0 / self.num_codewords).sample([self.num_codewords, self.latent_dim])
        codebook = torch.load(f'initial_cb_{bitrate}bit.pt', weights_only=True)
        self.codebook = torch.nn.Parameter(codebook, requires_grad=True)
        self.codebooks_used = torch.zeros(self.num_codewords, dtype=torch.int32)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        distances = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.codebook.t()))

        min_encoding_indices = torch.argmin(distances, dim=1)
        with torch.no_grad():
            self.codebooks_used[min_encoding_indices.cpu()] += 1

        # Compute Perplexity
        encodings = torch.zeros(min_encoding_indices.shape[0], self.num_codewords, device=z.device)
        encodings.scatter_(1, min_encoding_indices.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))

        z_hard_quantized = self.codebook[min_encoding_indices]
        direction = z_hard_quantized - z_flattened
        random_vectors = normal.Normal(0, self.noise_var).sample(z_flattened.shape).to(z.device) + direction
        normalized_random_vectors = random_vectors / (torch.linalg.norm(random_vectors, dim=1, keepdim=True) + self.eps)
        error_magnitude = torch.linalg.norm(z_hard_quantized - z_flattened, dim=1, keepdim=True)
        vq_error = error_magnitude * normalized_random_vectors.detach()

        z_q = (z_flattened + vq_error).view(z.shape)
        z_q = z_q.permute(0, 3, 1, 2)

        loss = torch.tensor(0.0, dtype=torch.float32)

        return z_q, min_encoding_indices, loss, perplexity

    def inference(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        distances = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.codebook ** 2, dim=1) - \
                    2 * (torch.matmul(z_flattened, self.codebook.t()))

        min_encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.codebook[min_encoding_indices].view(z.shape)
        return z_q.permute(0, 3, 1, 2), min_encoding_indices

    def codebook_replacement(self, num_batches):
        if self.cbr_method == 'new':
            self.codebooks_used, self.codebook = cbr_new(self.codebooks_used, self.codebook, num_batches,self.discarding_threshold, 1e-9, self.latent_dim)
        elif self.cbr_method == 'old':
            self.codebooks_used, self.codebook = cbr_old(self.codebooks_used, self.codebook, num_batches,self.discarding_threshold, 1e-9, self.latent_dim)


class DIVEQ_DETACH(nn.Module):
    def __init__(self, args):
        super(DIVEQ_DETACH, self).__init__()
        bitrate = args.bitrate
        self.num_codewords = int(2 ** bitrate)
        self.latent_dim = args.latent_dim
        self.eps = 1e-12
        self.cbr_method = args.cbr_method
        self.discarding_threshold = args.discard_threshold

        # codebook = uniform.Uniform(-1.0 / self.num_codewords, 1.0 / self.num_codewords).sample([self.num_codewords, self.latent_dim])
        codebook = torch.load(f'initial_cb_{bitrate}bit.pt', weights_only=True)
        self.codebook = torch.nn.Parameter(codebook, requires_grad=True)
        self.codebooks_used = torch.zeros(self.num_codewords, dtype=torch.int32)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        distances = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.codebook.t()))

        min_encoding_indices = torch.argmin(distances, dim=1)
        with torch.no_grad():
            self.codebooks_used[min_encoding_indices.cpu()] += 1

        # Compute Perplexity
        encodings = torch.zeros(min_encoding_indices.shape[0], self.num_codewords, device=z.device)
        encodings.scatter_(1, min_encoding_indices.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))

        z_hard_quantized = self.codebook[min_encoding_indices]
        error_magnitude = torch.linalg.norm(z_hard_quantized - z_flattened, dim=1, keepdim=True)
        vq_error = error_magnitude * ((z_hard_quantized - z_flattened) / (error_magnitude + 1e-15)).detach()

        z_q = (z_flattened + vq_error).view(z.shape)
        z_q = z_q.permute(0, 3, 1, 2)

        loss = torch.tensor(0.0, dtype=torch.float32)

        return z_q, min_encoding_indices, loss, perplexity

    def inference(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        distances = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.codebook ** 2, dim=1) - \
                    2 * (torch.matmul(z_flattened, self.codebook.t()))

        min_encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.codebook[min_encoding_indices].view(z.shape)
        return z_q.permute(0, 3, 1, 2), min_encoding_indices

    def codebook_replacement(self, num_batches):
        if self.cbr_method == 'new':
            self.codebooks_used, self.codebook = cbr_new(self.codebooks_used, self.codebook, num_batches,self.discarding_threshold, 1e-9, self.latent_dim)
        elif self.cbr_method == 'old':
            self.codebooks_used, self.codebook = cbr_old(self.codebooks_used, self.codebook, num_batches,self.discarding_threshold, 1e-9, self.latent_dim)

class SFDIVEQ(nn.Module):
    def __init__(self, args):
        super(SFDIVEQ, self).__init__()
        bitrate = args.bitrate
        self.num_codewords = int(2 ** bitrate)
        self.latent_dim = args.latent_dim
        self.noise_var = args.variance
        self.eps = 1e-12

        codebook = uniform.Uniform(-1.0 / self.num_codewords, 1.0 / self.num_codewords).sample([self.num_codewords, self.latent_dim])
        self.codebook = torch.nn.Parameter(codebook, requires_grad=True)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        dither = torch.rand((self.num_codewords - 1, 1), device=z.device)
        integer_index = torch.linspace(0, self.num_codewords - 2, self.num_codewords - 1, device=z.device).to(torch.int64)
        c0 = self.codebook[integer_index]
        c1 = self.codebook[integer_index + 1]
        dithered_codebook = ((1 - dither) * c0) + (dither * c1)

        distances = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(dithered_codebook**2, dim=1) - \
            2*(torch.matmul(z_flattened, dithered_codebook.t()))

        min_indices = torch.argmin(distances, dim=1)
        cb_first = self.codebook[min_indices]
        cb_second = self.codebook[min_indices + 1]
        s = dither[min_indices]

        direction_first = cb_first - z_flattened
        random_vectors1 = normal.Normal(0, self.noise_var).sample(z_flattened.shape).to(z.device) + direction_first
        normalized_random_vectors1 = random_vectors1 / torch.linalg.norm(random_vectors1, dim=1, keepdim=True)

        direction_second = cb_second - z_flattened
        random_vectors2 = normal.Normal(0, self.noise_var).sample(z_flattened.shape).to(z.device) + direction_second
        normalized_random_vectors2 = random_vectors2 / torch.linalg.norm(random_vectors2, dim=1, keepdim=True)

        error_magnitude1 = torch.linalg.norm(cb_first - z_flattened, dim=1, keepdim=True)
        error_magnitude2 = torch.linalg.norm(cb_second - z_flattened, dim=1, keepdim=True)
        vq_error1 = error_magnitude1 * ((1 - s) * normalized_random_vectors1).detach()
        vq_error2 = error_magnitude2 * (s * normalized_random_vectors2).detach()

        z_q = (z_flattened + vq_error1 + vq_error2).view(z.shape)

        # Compute Perplexity
        encodings = torch.zeros(min_indices.shape[0], self.num_codewords, device=z.device)
        encodings.scatter_(1, min_indices.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))

        z_q = z_q.permute(0, 3, 1, 2)

        loss = torch.tensor(0.0, dtype=torch.float32)

        return z_q, min_indices, loss, perplexity

    def dithered_inference(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)
        distances = (torch.sum(z_flattened ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebook ** 2, dim=1)
                     - 2 * torch.matmul(z_flattened, self.codebook.t()))
        integer_index = torch.argmin(distances, dim=1).clamp(min=1, max=self.num_codewords - 2)

        cm = self.codebook[integer_index - 1]
        cc = self.codebook[integer_index]
        cp = self.codebook[integer_index + 1]

        remainder_index_m = (((cc - cm) * (z_flattened - cm)).sum(dim=1) / (cc - cm).square().sum(dim=1)).unsqueeze(-1).clamp(min=0., max=1.)
        remainder_index_p = (((cp - cc) * (z_flattened - cc)).sum(dim=1) / (cp - cc).square().sum(dim=1)).unsqueeze(-1).clamp(min=0., max=1.)

        xhat_m = ((1 - remainder_index_m) * cm) + (remainder_index_m * cc)
        xhat_p = ((1 - remainder_index_p) * cc) + (remainder_index_p * cp)
        distance_m = (z_flattened - xhat_m).square().sum(dim=1)
        distance_p = (z_flattened - xhat_p).square().sum(dim=1)

        offset = (distance_p < distance_m).to(torch.int64).squeeze() - 1

        c0 = self.codebook[integer_index + offset]
        c1 = self.codebook[integer_index + offset + 1]

        remainder_index = (((c1 - c0) * (z_flattened - c0)).sum(dim=1) / (c1 - c0).square().sum(dim=1)).clamp(min=0., max=1.)
        quantized = (c0 + (remainder_index.reshape(-1, 1) * (c1 - c0))).view(z.shape)

        min_indices = integer_index + offset

        return quantized.permute(0, 3, 1, 2), min_indices

    def inference(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)
        distances = (torch.sum(z_flattened ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebook ** 2, dim=1)
                     - 2 * torch.matmul(z_flattened, self.codebook.t()))
        min_encoding_indices = torch.argmin(distances, dim=1)

        z_q = self.codebook[min_encoding_indices].view(z.shape)

        return z_q.permute(0, 3, 1, 2), min_encoding_indices

    def get_random_dithered_codebook(self, indices):
        dither = torch.rand((indices.shape[0], 1), device=indices.device)
        c0 = self.codebook[indices]
        c1 = self.codebook[indices + 1]
        sampled_codebook = ((1 - dither) * c0) + (dither * c1)
        return sampled_codebook


class SFDIVEQ_DETACH(nn.Module):
    def __init__(self, args):
        super(SFDIVEQ_DETACH, self).__init__()
        bitrate = args.bitrate
        self.num_codewords = int(2 ** bitrate)
        self.latent_dim = args.latent_dim
        self.noise_var = args.variance
        self.eps = 1e-12

        codebook = uniform.Uniform(-1.0 / self.num_codewords, 1.0 / self.num_codewords).sample([self.num_codewords, self.latent_dim])
        self.codebook = torch.nn.Parameter(codebook, requires_grad=True)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        dither = torch.rand((self.num_codewords - 1, 1), device=z.device)
        integer_index = torch.linspace(0, self.num_codewords - 2, self.num_codewords - 1, device=z.device).to(torch.int64)
        c0 = self.codebook[integer_index]
        c1 = self.codebook[integer_index + 1]
        dithered_codebook = ((1 - dither) * c0) + (dither * c1)

        distances = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(dithered_codebook**2, dim=1) - \
            2*(torch.matmul(z_flattened, dithered_codebook.t()))

        min_indices = torch.argmin(distances, dim=1)
        cb_first = self.codebook[min_indices]
        cb_second = self.codebook[min_indices + 1]
        s = dither[min_indices]

        error_magnitude1 = torch.linalg.norm(cb_first - z_flattened, dim=1, keepdim=True)
        error_magnitude2 = torch.linalg.norm(cb_second - z_flattened, dim=1, keepdim=True)
        vq_error1 = error_magnitude1 * (((1 - s) * (cb_first - z_flattened)) / (error_magnitude1 + 1e-15)).detach()
        vq_error2 = error_magnitude2 * ((s * (cb_second - z_flattened)) / (error_magnitude2 + 1e-15)).detach()

        z_q = (z_flattened + vq_error1 + vq_error2).view(z.shape)

        # Compute Perplexity
        encodings = torch.zeros(min_indices.shape[0], self.num_codewords, device=z.device)
        encodings.scatter_(1, min_indices.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))

        z_q = z_q.permute(0, 3, 1, 2)

        loss = torch.tensor(0.0, dtype=torch.float32)

        return z_q, min_indices, loss, perplexity


class Gumbel_Softmax(nn.Module):
    def __init__(self, args, straight_through=True):
        super(Gumbel_Softmax, self).__init__()
        bitrate = args.bitrate
        self.num_codewords = int(2**bitrate)
        self.latent_dim = args.latent_dim
        self.beta = args.beta
        self.eps = 1e-12
        self.cbr_method = args.cbr_method
        self.discarding_threshold = args.discard_threshold

        self.straight_through = straight_through
        self.temperature = 1.0
        self.kld_scale = 5e-4

        # codebook = uniform.Uniform(-1.0 / self.num_codewords, 1.0 / self.num_codewords).sample([self.num_codewords, self.latent_dim])
        codebook = torch.load(f'initial_cb_{bitrate}bit.pt', weights_only=True)
        self.codebook = nn.Embedding(self.num_codewords, self.latent_dim)
        self.codebook.weight.data = codebook

        self.codebooks_used = torch.zeros(self.num_codewords, dtype=torch.int32)

    def forward(self, z):
        soft_one_hot = F.gumbel_softmax(z, tau=self.temperature, dim=1, hard=self.straight_through)
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(z, dim=1)
        diff = self.kld_scale * torch.sum(qy * torch.log(qy * self.num_codewords + 1e-10), dim=1).mean()

        min_indices = soft_one_hot.argmax(dim=1).flatten()
        encoding_indices = min_indices.unsqueeze(1)
        with torch.no_grad():
            self.codebooks_used[min_indices.cpu()] += 1

        # Compute Perplexity
        encodings = torch.zeros(encoding_indices.shape[0], self.num_codewords, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))

        return z_q, min_indices, diff, perplexity

    def inference(self, z):
        with torch.no_grad():
            soft_one_hot = F.gumbel_softmax(z, tau=1e-10, dim=1, hard=self.straight_through)
            z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)
            min_indices = soft_one_hot.argmax(dim=1).flatten()
        return z_q, min_indices

    def codebook_replacement(self, num_batches):
        if self.cbr_method == 'new':
            self.codebooks_used, self.codebook.weight.data = cbr_new(self.codebooks_used, self.codebook.weight.data,num_batches,self.discarding_threshold, 1e-9,self.latent_dim)
        elif self.cbr_method == 'old':
            self.codebooks_used, self.codebook.weight.data = cbr_old(self.codebooks_used, self.codebook.weight.data,num_batches,self.discarding_threshold, 1e-9,self.latent_dim)


class EMA(nn.Module):
    def __init__(self, args):
        super(EMA, self).__init__()
        bitrate = args.bitrate
        self.num_codewords = int(2**bitrate)
        self.latent_dim = args.latent_dim
        self.beta = args.beta
        self.eps = 1e-12
        self.cbr_method = args.cbr_method
        self.discarding_threshold = args.discard_threshold

        # initial_codebook = uniform.Uniform(-1.0 / self.num_codewords, 1.0 / self.num_codewords).sample([self.num_codewords, self.latent_dim])
        initial_codebook = torch.load(f'initial_cb_{bitrate}bit.pt', weights_only=True)

        self._embedding = nn.Embedding(self.num_codewords, self.latent_dim)
        self._embedding.weight.data = initial_codebook
        self.register_buffer('_ema_cluster_size', torch.zeros(self.num_codewords))
        self._ema_w = nn.Parameter(torch.Tensor(self.num_codewords, self.latent_dim))
        self._ema_w.data.normal_()
        self._decay = 0.99
        self._epsilon = 1e-5

        self.codebooks_used = torch.zeros(self.num_codewords, dtype=torch.int32)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        distances = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self._embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self._embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_codewords, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)

        min_indices = encoding_indices.squeeze(1)
        with torch.no_grad():
            self.codebooks_used[min_indices.cpu()] += 1

        # Quantize and unflatten
        z_q = torch.matmul(encodings, self._embedding.weight).view(z.shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self.num_codewords * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), z_flattened)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        loss = self.beta * F.mse_loss(z_q.detach(), z)

        # Straight Through Estimator
        z_q = z + (z_q - z).detach()

        # Compute Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))

        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_indices , loss, perplexity

    def inference(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        distances = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                    torch.sum(self._embedding.weight ** 2, dim=1) - \
                    2 * (torch.matmul(z_flattened, self._embedding.weight.t()))

        min_indices = torch.argmin(distances, dim=1)
        encoding_indices = min_indices.unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_codewords, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)

        z_q = torch.matmul(encodings, self._embedding.weight).view(z.shape)
        return z_q.permute(0, 3, 1, 2), min_indices

    def codebook_replacement(self, num_batches):
        if self.cbr_method == 'new':
            self.codebooks_used, self._embedding.weight.data = cbr_new(self.codebooks_used, self._embedding.weight.data,num_batches,self.discarding_threshold, 1e-9,self.latent_dim)
        elif self.cbr_method == 'old':
            self.codebooks_used, self._embedding.weight.data = cbr_old(self.codebooks_used, self._embedding.weight.data,num_batches,self.discarding_threshold, 1e-9,self.latent_dim)
