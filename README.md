# Convolutional Variational Autoencoder for MNIST

A PyTorch implementation of a convolutional Variational Autoencoder (VAE) with advanced training techniques for MNIST digit generation and reconstruction.

## Results

**Final Test Performance:**
- **Reconstruction Loss**: 71.16
- **KL Divergence**: 29.79
- **Total ELBO**: 95.00

## Features

- Convolutional encoder-decoder architecture with batch normalization and dropout
- β-VAE implementation with KL divergence weighting
- KL annealing for balanced reconstruction and regularization
- Cosine learning rate scheduling with warmup phase
- Comprehensive training monitoring and visualization

## Installation

```bash
pip install torch torchvision matplotlib numpy

Training
```bash
python final_vae.py --epochs 120 --beta 0.8 --z-dim 32 --base-lr 1e-4 --max-lr 3e-4 --batch-size 256 --warmup-epochs 10

Command Line Arguments:

    --epochs: Number of training epochs (default: 120)

    --beta: KL divergence weight coefficient (default: 0.8)

    --z-dim: Latent space dimension (default: 32)

    --base-lr: Base learning rate (default: 1e-4)

    --max-lr: Maximum learning rate for cycling (default: 3e-4)

    --batch-size: Training batch size (default: 256)

    --warmup-epochs: Warmup epochs for learning rate (default: 10)

    --out: Output directory (default: "vae_final_out")
    
## Outputs

Training generates:

    best_model.pth: Best performing model weights

    samples_epoch_*.png: Generated samples at different epochs

    recon_epoch_*.png: Input-reconstruction comparisons

    training_history.png: Loss progression plots
    
## Technical Details

## Architecture:

    Encoder: CNN with 32→64→128 channels, batch norm, dropout

    Decoder: Transposed CNN with 128→64→32→1 channels, batch norm

    Latent Space: 32-dimensional Gaussian distribution

## Training Techniques:

    AdamW optimizer with weight decay (1e-5)

    Gradient clipping (1.0)

    KL annealing over 20 epochs

    Cosine learning rate scheduling with warmup

    Data augmentation (random rotation/translation)

## Training Curve

Comprehensive monitoring of:

    Total ELBO loss

    Reconstruction loss

    KL divergence

    Learning rate progression
