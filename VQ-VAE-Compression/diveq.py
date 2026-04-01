import torch
import torch.nn as nn
from torch.distributions import normal
from typing import Tuple
import warnings

class DIVEQ(nn.Module):
    """
    DiVeQ: Differentiable Vector Quantization (VQ) module that allows end-to-end
    training of VQ-based models without any auxiliary losses or hyperparameter tunings.
    The module encompasses codebook replacement function which discards unused codebook
    entries during training.

    Args:
        - num_embeddings (int): Codebook size (No. of codewords).
        - embedding_dim (int): Dimensionality of embeddings.
        - noise_var (float): Variance of the directional noise in DiVeQ.
            Recommended noise_var < 1e-2.
        - replacement_iters (int): Replacement interval (number of training iterations
            to apply codebook replacement). Recommended 50 < replacement_iters < 300.
        - discard_threshold (float): Threshold to discard the codebook entries that are
            used less than this threshold after "replacement_iters" iterations.
            Recommended 0.01 < discard_threshold < 0.05. discard_threshold must be in
            the range of [0,1] such that discard_threshold=0.01 means to discard the
            codebook entries which are used less than 1 percent.
        - perturb_eps (float): Adjusts perturbation/shift magnitude from used codewords
            for codebook replacement.
        - uniform_init (bool): Whether to initialize codebook with uniform distribution.
            If False, the codebook is initialized from a normal distribution.
        - allow_warning (bool): Whether to print the warnings.
        - verbose (bool): Whether to print codebook replacement status.

    Returns:
        - z_q (torch.Tensor): Differentiable quantized input/latent. shape (N, D)
        - indices (torch.Tensor): Selected codebook indices. shape (N, )
        - perplexity (float): Codebook perplexity (average codebook usage)
    """
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            noise_var: float = 0.001,
            replacement_iters: int = 100,
            discard_threshold: float = 0.01,
            perturb_eps: float = 1e-9,
            uniform_init: bool = True,
            allow_warning: bool = True,
            verbose: bool = True,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.noise_var = noise_var
        self.replacement_iters = replacement_iters
        self.discard_threshold = discard_threshold
        self.perturb_eps = perturb_eps
        self.uniform_init = uniform_init
        self.allow_warning = allow_warning
        self.verbose = verbose

        self._check_constraints()

        # ---------------- User warnings ----------------
        if allow_warning:
            if noise_var > 0.01:
                warnings.warn(f"`noise_var` is set to {noise_var}, which is"
                              f" quite large. Values > 0.01 may overshoot"
                              f" nearest-neighbor mapping.", UserWarning)
            if replacement_iters < 50:
                warnings.warn(f"`replacement_iters` is set to"
                              f" {replacement_iters}, which is quite small. Values < 50"
                              f" may cause too early and frequent codebook"
                              f" replacements.", UserWarning)
            elif replacement_iters > 300:
                warnings.warn(f"`replacement_iters` is set to"
                              f" {replacement_iters}, which is quite large."
                              f" Values > 300 may cause too late and sporadic codebook"
                              f" replacements.", UserWarning)

            if discard_threshold > 0.05:
                warnings.warn(f"`discard_threshold` is set to"
                              f" {discard_threshold}, which is quite large."
                              f" Values > 0.05 may discard a portion of suitable but"
                              f" rarely used codewords.", UserWarning)

            if perturb_eps > 1e-6:
                warnings.warn(f"`perturb_eps` is set to {perturb_eps}, which"
                              f" is quite large. Values > 1e-6 may cause big"
                              f" perturbation/shift from used codewords.", UserWarning)

        # ---------------- Codebook initialization ----------------
        if uniform_init:
            codebook = (torch.rand((self.num_embeddings, self.embedding_dim))
                        * (1 / self.num_embeddings))
        else:
            codebook = (torch.randn((self.num_embeddings, self.embedding_dim))
                        * (1 / self.num_embeddings))

        self.codebook = torch.nn.Parameter(codebook, requires_grad=True)

        # ---------------- Tensors used for codebook replacement ----------------
        self.register_buffer("codebook_usage", torch.zeros(self.num_embeddings,
                                                           dtype=torch.int32))
        self.register_buffer("iter_counter", torch.zeros(1, dtype=torch.int32))

    # ---------------- Forward pass (Core API) ----------------
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Args:
            - z (torch.Tensor): input/latent. shape (N, D)

        Returns:
            - z_q (torch.Tensor): Differentiable quantized input/latent. shape (N, D)
            - indices (torch.Tensor): Selected codebook indices. shape (N, )
            - perplexity (float): Codebook perplexity (average codebook usage).
        """
        self._check_input(z)

        # Calculate distances
        distances = (torch.sum(z.pow(2), dim=1, keepdim=True)
                     + torch.sum(self.codebook.pow(2), dim=1)
                     - 2 * torch.matmul(z, self.codebook.t()))

        indices = torch.argmin(distances, dim=1)

        z_hard_quantized = self.codebook[indices] # Non-differentiable quantized input

        direction = z_hard_quantized - z
        random_vectors = (normal.Normal(0, self.noise_var).sample(z.shape)
                          .to(z.device) + direction)
        normalized = random_vectors / torch.linalg.norm(random_vectors, dim=1,
                                                        keepdim=True).clamp_min(1e-12)
        error_magnitude = torch.linalg.norm(z_hard_quantized - z, dim=1, keepdim=True)

        vq_error = error_magnitude * normalized.detach()
        z_q = z + vq_error # Differentiable quantized input

        # Perplexity Computation
        perplexity = self._compute_perplexity(indices)

        # Track used indices for codebook replacement
        with torch.no_grad():
            self.codebook_usage[indices] += 1
            self.iter_counter += 1
            if self.iter_counter.item() % self.replacement_iters == 0:
                self._replace_unused_entries() # Applies codebook replacement

        return z_q, indices, perplexity

    # ---------------- Quantization for Inference ----------------
    @torch.no_grad()
    def inference(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Deterministic hard quantization by mapping the input to the nearest codeword.
        Args:
            - z (torch.Tensor): input/latent. shape (N, D)

        Returns:
            - z_q_hard (torch.Tensor): Hard quantized input/latent. shape (N, D)
            - indices (torch.Tensor): Selected codebook indices. shape (N, )
            - perplexity (float): Codebook perplexity (average codebook usage).
        """
        self._check_input(z)
        distances = (torch.sum(z.pow(2), dim=1, keepdim=True)
                     + torch.sum(self.codebook.pow(2), dim=1)
                     - 2.0 * torch.matmul(z, self.codebook.t()))
        indices = torch.argmin(distances, dim=1)
        perplexity = self._compute_perplexity(indices)
        z_q_hard = self.codebook[indices]
        return z_q_hard, indices, perplexity

    # ---------------- Utility functions ----------------
    def _check_constraints(self,) -> None:
        if self.noise_var <= 0.0:
            raise ValueError("`noise_var` must be a positive float value. To have more"
                             " precise nearest-neighbor assignments, it is recommended"
                             " that noise_var < 1e-2.")
        if (self.replacement_iters <= 0) or (type(self.replacement_iters) is not int):
            raise ValueError("`replacement_iters` must be a positive integer value."
                             " It is recommended that 50 < replacement_iters < 300.")
        if (self.discard_threshold < 0.0) or (self.discard_threshold > 1.0):
            raise ValueError("`discard_threshold` must be in the range of [0,1]. It is"
                             " recommended that 0.01 < discard_threshold < 0.05,"
                             " such that discard_threshold=0.01 means to discard the"
                             " codebook entries which are used less than 1 percent.")

    def _check_input(self, z: torch.Tensor) -> None:
        if z.ndim != 2:
            raise ValueError("DiVeQ input must have the shape of (N, D), where N is"
                             " the No. of input samples,and D is the embedding"
                             " dimensionality.")
        if z.size(1) != self.embedding_dim:
            raise ValueError(f"DiVeQ input.shape[1] must match the embedding"
                             f" dimensionality that is {self.embedding_dim}.")

    def _compute_perplexity(self, indices: torch.Tensor) -> float:
        encodings = torch.zeros(indices.shape[0], self.num_embeddings,
                                device=indices.device)
        encodings.scatter_(1, indices.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity.item()

    def _replace_unused_entries(self) -> None:
        with torch.no_grad():
            usage_ratio = self.codebook_usage / self.replacement_iters
            unused_indices = torch.where(usage_ratio < self.discard_threshold)[0]
            used_indices = torch.where(usage_ratio >= self.discard_threshold)[0]

            if unused_indices.numel() == 0 or used_indices.numel() == 0:
                self.codebook_usage.zero_()
                return

            unused_count = unused_indices.numel()
            used_probs = self.codebook_usage[used_indices] / torch.sum(
                self.codebook_usage[used_indices])
            randomly_sampled_indices = used_probs.multinomial(num_samples=unused_count,
                                                              replacement=True)
            sampled_indices = used_indices[randomly_sampled_indices]
            used_codebooks = self.codebook[sampled_indices].clone()

            self.codebook[unused_indices] = (used_codebooks +
                         self.perturb_eps * torch.randn_like(used_codebooks)).clone()
            self.codebook_usage.zero_()

            if self.verbose:
                print("\n***** Replaced " + str(unused_count) + " codewords *****")

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"noise_var={self.noise_var}, "
            f"replacement_iters={self.replacement_iters}, "
            f"discard_threshold={self.discard_threshold}, "
            f"perturb_eps={self.perturb_eps}, "
            f"uniform_init={self.uniform_init}, "
            f"allow_warning={self.allow_warning}, "
            f"verbose={self.verbose}"
        )