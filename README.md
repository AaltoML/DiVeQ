# **DiVeQ: Differentiable Vector Quantization Using the Reparameterization Trick**

# Abstract
Vector quantization is common in deep models, yet its hard assignments block gradients and hinder end-to-end training. We propose DiVeQ, which treats quantization as adding an error vector that mimics the quantization distortion, keeping the forward pass hard while letting gradients flow. We also present a space-filling variant (SF-DiVeQ) that assigns to a curve constructed by the lines connecting codewords, resulting in less quantization error and full codebook usage. Both methods train end-to-end without requiring auxiliary losses or temperature schedules. On VQ-VAE compression and VQGAN generation across various data sets, they improve reconstruction and sample quality over alternative quantization approaches.

# **VQVAE Compression directory**

- `train.py`: code to train the VQ-VAE model
- `vq.py`: contains the code for all VQ optimization techniques
- `util_funcs.py`: contains some utility functions like codebook replacement

# **VQGAN Generation directory**

- `training_vqgan.py`: code to train the VQ-VAE model
- `training_transformer.py`: code to train the transformer
- `sample_transformer.py`: code to generate images from trained VQGAN
- `compute_fid.py`: code to compute the FID score
- `vq.py`: contains the code for all VQ optimization techniques
- `encoder.py`: contains the code for VQ-VAE encoder
- `decoder.py`: contains the code for VQ-VAE decoder
- `discriminator.py`: contains the code for the discriminator model used for training VQ-VAE
- `vqgan.py`: contains the code to build the VQ-VAE model with the encoder, vector quantization, and decoder
- `transformer.py`: contains the code to build the transformer model
- `mingpt.py`: contains the code for GPT model
- `helper.py`: contains some utility blocks used in building the models such as GroupNorm, ResidualBlock
- `utils.py`: contains some utility functions like codebook replacement

# **Create the conda environment for VQVAE Compression**

Create the environment by passing the following in your terminal in the following order.

`conda create --name vqvae_comp python=3.9.19`  
`conda activate vqvae_comp`

# **Create the conda environment for VQGAN Generation**

Create the environment by passing the following in your terminal in the following order.

`conda create --name vqvae_comp python=3.9.19`  
`conda activate vqvae_comp`


