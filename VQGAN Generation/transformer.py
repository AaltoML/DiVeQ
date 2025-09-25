import torch
import torch.nn as nn
import torch.nn.functional as F
from mingpt import GPT
from vqgan import VQGAN

class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super(VQGANTransformer, self).__init__()

        self.sfvq_dithered_inference = args.sfvq_dithered_inference
        self.codebook_optimization = args.codebook_optimization
        if self.sfvq_dithered_inference:
            self.vocab_size = int(2 ** args.bitrate) - 1
        else:
            self.vocab_size = int(2 ** args.bitrate)

        self.sos_token = args.sos_token

        self.vqgan = self.load_vqgan(args)

        transformer_config = {
            "vocab_size": self.vocab_size,
            "block_size": 256,
            "n_layer": 28,
            "n_head": 16,
            "n_embd": 1024
        }
        self.transformer = GPT(**transformer_config)

        self.pkeep = args.pkeep_start

    @staticmethod
    def load_vqgan(args):
        model = VQGAN(args)
        model.load_checkpoint(args.checkpoint_path)
        model = model.eval()
        return model

    @torch.no_grad()
    def encode_to_z(self, x):
        x_embedding = self.vqgan.encode(x)
        if self.sfvq_dithered_inference:
            quant_z, indices = self.vqgan.vq.dithered_inference(x_embedding)
        else:
            quant_z, indices = self.vqgan.vq.inference(x_embedding)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def z_to_image(self, indices, p1=16, p2=16):
        if self.sfvq_dithered_inference:
            ix_to_vectors = self.vqgan.vq.get_random_dithered_codebook(indices.flatten()).reshape(indices.shape[0], p1, p2, 256)
        else:
            if self.codebook_optimization == 'ema':
                ix_to_vectors = self.vqgan.vq._embedding.weight.data[indices.flatten()].reshape(indices.shape[0], p1, p2, 256)
            elif self.codebook_optimization == 'gumbel_softmax':
                ix_to_vectors = self.vqgan.vq.codebook.weight.data[indices.flatten()].reshape(indices.shape[0], p1, p2, 256)
            else:
                ix_to_vectors = self.vqgan.vq.codebook[indices.flatten()].reshape(indices.shape[0], p1, p2, 256)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqgan.decode(ix_to_vectors)
        return image

    def forward(self, x):
        _, indices = self.encode_to_z(x)

        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")

        mask = torch.bernoulli(self.pkeep * torch.ones(indices.shape, device=indices.device))
        mask = mask.round().to(dtype=torch.int64)
        random_indices = torch.randint_like(indices, self.transformer.config.vocab_size)

        new_indices = mask * indices + (1 - mask) * random_indices

        new_indices = torch.cat((sos_tokens, new_indices), dim=1)

        target = indices

        logits, _ = self.transformer(new_indices[:, :-1])

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, top_k=100):
        self.transformer.eval()
        x = torch.cat((c, x), dim=1)
        for k in range(steps):
            logits, _ = self.transformer(x)
            logits = logits[:, -1, :] / temperature

            logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        x = x[:, c.shape[1]:]
        self.transformer.train()
        return x

    @torch.no_grad()
    def log_images(self, x):
        log = dict()

        _, indices = self.encode_to_z(x)
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")

        start_indices = indices[:, :indices.shape[1] // 2]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1] - start_indices.shape[1])
        half_sample = self.z_to_image(sample_indices)

        start_indices = indices[:, :0]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1])
        full_sample = self.z_to_image(sample_indices)

        x_rec = self.z_to_image(indices)

        log["input"] = x
        log["rec"] = x_rec
        log["half_sample"] = half_sample
        log["full_sample"] = full_sample

        return log, torch.concat((x, x_rec, half_sample, full_sample))