#!/usr/bin/env python
"""
Final Optimized Convolutional Variational Autoencoder on MNIST
--------------------------------------------------------------
• Balanced reconstruction and KL loss
• Improved architecture with better regularization
• Learning rate scheduling with warmup
• Comprehensive monitoring
"""

import argparse, os, torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- Optimized Encoder ----------
class ConvEncoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.conv = nn.Sequential(
            # 1×28×28 → 32×14×14
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),
            
            # 32×14×14 → 64×7×7
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),
            
            # 64×7×7 → 128×7×7
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(128 * 7 * 7, z_dim)
        self.fc_logvar = nn.Linear(128 * 7 * 7, z_dim)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar

# ---------- Optimized Decoder ----------
class ConvDecoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3)
        )
        
        self.deconv = nn.Sequential(
            # 128×7×7 → 64×14×14
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),
            
            # 64×14×14 → 32×28×28
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),
            
            # 32×28×28 → 32×28×28
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            
            # 32×28×28 → 1×28×28
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 128, 7, 7)
        return self.deconv(h)

# ---------- VAE ----------
class VAE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.enc = ConvEncoder(z_dim)
        self.dec = ConvDecoder(z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.dec(z)
        return recon_x, mu, logvar

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.z_dim, device=device)
        return self.dec(z)

# ---------- Loss with KL Annealing ----------
def elbo(recon_x, x, mu, logvar, beta, kl_weight=1.0):
    batch_size = x.size(0)
    
    # Clamp to avoid numerical issues
    recon_x = torch.clamp(recon_x, min=1e-8, max=1-1e-8)
    
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size
    
    # KL divergence with annealing
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    kl_loss = kl_loss * kl_weight
    
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss

# ---------- Cosine Learning Rate Scheduler ----------
class CosineLRScheduler:
    def __init__(self, optimizer, base_lr, max_lr, warmup_epochs, total_epochs):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.epoch = 0
        
    def step(self):
        self.epoch += 1
        
        if self.epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr + (self.max_lr - self.base_lr) * (self.epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = max(lr, self.base_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr

# ---------- Training ----------
def train_epoch(model, loader, opt, lr_scheduler, epoch, beta, kl_weight, grad_clip=None):
    model.train()
    total_loss, recon_loss_total, kl_loss_total = 0, 0, 0
    num_batches = 0
    
    for batch_idx, (x, _) in enumerate(loader):
        x = x.to(device)
        opt.zero_grad()
        
        # Update learning rate
        current_lr = lr_scheduler.step()
        
        recon_batch, mu, logvar = model(x)
        loss, recon_loss, kl_loss = elbo(recon_batch, x, mu, logvar, beta, kl_weight)
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
            
        loss.backward()
        
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        opt.step()
        
        total_loss += loss.item()
        recon_loss_total += recon_loss.item()
        kl_loss_total += kl_loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(x)}/{len(loader.dataset)} '
                  f'({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f} '
                  f'LR: {current_lr:.2e} | KL Weight: {kl_weight:.3f}')
    
    if num_batches == 0:
        return 0, 0, 0
        
    return total_loss/num_batches, recon_loss_total/num_batches, kl_loss_total/num_batches

# ---------- Validation ----------
@torch.no_grad()
def validate(model, loader, beta):
    model.eval()
    total_loss, recon_loss_total, kl_loss_total = 0, 0, 0
    num_batches = 0
    
    for x, _ in loader:
        x = x.to(device)
        recon_batch, mu, logvar = model(x)
        loss, recon_loss, kl_loss = elbo(recon_batch, x, mu, logvar, beta, 1.0)
        
        total_loss += loss.item()
        recon_loss_total += recon_loss.item()
        kl_loss_total += kl_loss.item()
        num_batches += 1
    
    return total_loss/num_batches, recon_loss_total/num_batches, kl_loss_total/num_batches

# ---------- Main ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Final Optimized Conv-VAE for MNIST')
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--base-lr", type=float, default=1e-4)
    p.add_argument("--max-lr", type=float, default=3e-4)
    p.add_argument("--z-dim", type=int, default=32)
    p.add_argument("--beta", type=float, default=0.8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--out", type=str, default="vae_final_out")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    
    # Data loading with slight augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(2),
        transforms.RandomAffine(degrees=0, translate=(0.02, 0.02))
    ])
    
    train_ds = datasets.MNIST(".", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(".", train=False, download=True, transform=transforms.ToTensor())
    val_ds = torch.utils.data.Subset(test_ds, range(5000))
    
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_ld = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = VAE(args.z_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-5)
    lr_scheduler = CosineLRScheduler(opt, args.base_lr, args.max_lr, args.warmup_epochs, args.epochs)
    
    print(f"Starting training with z_dim={args.z_dim}, beta={args.beta}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_recon': [], 'val_recon': [], 'train_kl': [], 'val_kl': [], 'lr': []}
    
    for epoch in range(1, args.epochs + 1):
        # KL annealing - gradually increase KL weight
        kl_weight = min(1.0, epoch / 20)  # Ramp up over 20 epochs
        
        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, train_ld, opt, lr_scheduler, epoch, args.beta, kl_weight, args.grad_clip
        )
        
        # Validate (always use full KL weight for validation)
        val_loss, val_recon, val_kl = validate(model, val_ld, args.beta)
        
        current_lr = opt.param_groups[0]['lr']
        history['lr'].append(current_lr)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_recon'].append(train_recon)
        history['val_recon'].append(val_recon)
        history['train_kl'].append(train_kl)
        history['val_kl'].append(val_kl)
        
        print(f'Epoch {epoch:03d}: Train ELBO {train_loss:.2f} | Val ELBO {val_loss:.2f} | '
              f'Recon {train_recon:.2f}/{val_recon:.2f} | KL {train_kl:.2f}/{val_kl:.2f} | '
              f'LR: {current_lr:.2e} | KL Weight: {kl_weight:.3f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.out, 'best_model.pth'))
            print(f"✓ New best model: {val_loss:.2f}")
        
        # Save samples and reconstructions
        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                # Generate samples
                samples = model.sample(64)
                save_image(samples, os.path.join(args.out, f'samples_epoch_{epoch:03d}.png'), nrow=8)
                
                # Save reconstructions
                test_data, _ = next(iter(test_ld))
                test_data = test_data[:8].to(device)
                recon_data, _, _ = model(test_data)
                comparison = torch.cat([test_data, recon_data])
                save_image(comparison, os.path.join(args.out, f'recon_epoch_{epoch:03d}.png'), nrow=8)
    
    # Load best model and final test
    model.load_state_dict(torch.load(os.path.join(args.out, 'best_model.pth')))
    test_loss, test_recon, test_kl = validate(model, test_ld, args.beta)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"Test ELBO: {test_loss:.2f}")
    print(f"Test Reconstruction: {test_recon:.2f}")
    print(f"Test KL: {test_kl:.2f}")
    print(f"{'='*60}")
    
    # Plot comprehensive results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('Total Loss (ELBO)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_recon'], label='Train')
    ax2.plot(history['val_recon'], label='Validation')
    ax2.set_title('Reconstruction Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(history['train_kl'], label='Train')
    ax3.plot(history['val_kl'], label='Validation')
    ax3.set_title('KL Divergence')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('KL Loss')
    ax3.legend()
    ax3.grid(True)
    
    ax4.plot(history['lr'])
    ax4.set_title('Learning Rate Schedule')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_yscale('log')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.2f}")
    print(f"Final test reconstruction: {test_recon:.2f}")
    print(f"Results saved in: {args.out}")