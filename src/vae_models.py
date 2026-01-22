import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import random

class TimeSeriesVAE(nn.Module):
    def __init__(self, input_dim=6, sequence_length=24, latent_dim=8, hidden_dim=128):
        super(TimeSeriesVAE, self).__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.encoder_lstm = nn.LSTM(128, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.encoder_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU()
        )

        self.decoder_lstm = nn.LSTM(hidden_dim, 128, batch_first=True)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, input_dim, kernel_size=5, padding=2)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()

    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.permute(0, 2, 1)
        h, _ = self.encoder_lstm(h)
        h = h[:, -1, :]
        h = self.encoder_fc(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.unsqueeze(1).repeat(1, self.sequence_length, 1)
        h, _ = self.decoder_lstm(h)
        h = h.permute(0, 2, 1)
        return self.decoder_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z
    

class WindPowerDataset(Dataset):
    def __init__(self, timeseries_matrix):
        self.data = timeseries_matrix.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])
    

def vae_loss_function(recon_x, x, mu, logvar, beta=0.5):

    recon_loss = nn.MSELoss(reduction='mean')(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = torch.clamp(kl_loss, max=1000.0)
    kl_loss /= x.size(0) * x.size(1) * x.size(2)
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def train_vae(train_normalised_vae, latent_dim=8, epochs=200, batch_size=64, vae_save_path=None, device='cuda'):

    if vae_save_path and os.path.exists(vae_save_path):
        vae = TimeSeriesVAE(train_normalised_vae.shape[1], 
                           train_normalised_vae.shape[2], latent_dim).to(device)
        vae.load_state_dict(torch.load(vae_save_path, map_location=device))
        vae.eval()
        print(f"VAE loaded from {vae_save_path}")
        return vae
    
    print(f"Training VAE (latent_dim={latent_dim}, epochs={epochs})")
    vae = TimeSeriesVAE(train_normalised_vae.shape[1], train_normalised_vae.shape[2], latent_dim).to(device)

    dataset = WindPowerDataset(train_normalised_vae)

    def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(42)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    optimizer = optim.AdamW(vae.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    vae.train()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        beta = min(1.0, 0.01 + (epoch / (epochs * 0.5)) * 0.99)         
        
        for batch in dataloader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            recon_batch, mu, logvar, z = vae(batch)
            total_loss, recon_loss, kl_loss = vae_loss_function(recon_batch, batch, mu, logvar, beta=beta)
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue
                
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            batch_count += 1
        
        if batch_count == 0:
            continue
            
        epoch_loss /= batch_count
        scheduler.step()
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter > 30:
            print(f"    Early stopping at epoch {epoch}")
            break
            
        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss={epoch_loss:.4f}, β={beta:.3f}")
    
    print(f"  VAE training completed! Best loss: {best_loss:.4f}")

    if vae_save_path is not None:
        torch.save(vae.state_dict(), vae_save_path)
        print(f"  VAE model saved to {vae_save_path}")

    return vae


def extract_vae_features(vae, normalised_vae, device='cuda'):

    vae.eval()
    vae = vae.to(device)

    use_workers = 0 if device == 'cpu' else 4
    use_pin_memory = False if device == 'cpu' else True

    dataset = WindPowerDataset(normalised_vae)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=use_workers, pin_memory=use_pin_memory)
    latent_codes = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device, non_blocking=True)
            mu, logvar = vae.encode(batch)
            z = mu 
            latent_codes.append(z.cpu().numpy())

    latent_matrix = np.vstack(latent_codes)

    if device == 'cuda':
        torch.cuda.empty_cache()

    return latent_matrix
