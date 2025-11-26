import os
import torch
import torch.optim as optim
import sys
sys.path.append('../../')
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import random
from models_pre_encode import SimpleVAE
from models.util import *
import math

os.makedirs('./models', exist_ok=True)
os.makedirs('./embeddings', exist_ok=True)

class RegionGridDataset(Dataset):
    def __init__(self, tensor, region_indices):
        self.tensor = tensor
        self.region_indices = region_indices

    def __len__(self):
        return len(self.region_indices)

    def __getitem__(self, idx):
        r = self.region_indices[idx]
        x = self.tensor[:, :, :, r, :, :]
        return x.squeeze(2), r

def train_vae_for_modality(city, name, data_tensor, device, epochs=1000, batch_size=16, latent_dim=64, use_mean=True):
    total_regions = data_tensor.shape[3]
    train_region_ids = list(range(total_regions - 1))
    test_region_ids = list(range(total_regions))
    shuffled_train_ids = train_region_ids.copy()
    random.shuffle(shuffled_train_ids)
    reverse_index = np.argsort(shuffled_train_ids)

    train_dataset = RegionGridDataset(data_tensor, shuffled_train_ids)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleVAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    reduction_type = 'mean' if use_mean else 'sum'

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        for x, region_ids in train_loader:
            x = x.to(device)
            B, T, H_per_day, H, W = x.shape
            x = x.reshape(B * T * H_per_day, 1, H, W)

            recon, mu, logvar = model(x)

            recon_loss = F.mse_loss(recon, x, reduction=reduction_type)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) if use_mean else \
                      -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            kl_weight = 0.5 * (1 - math.cos(min(epoch / 500, 1.0) * math.pi)) * 0.1
            loss = recon_loss + kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        print(f"Epoch {epoch} | {name} Total Loss: {total_loss:.2f} | Recon: {total_recon:.2f} | KL: {total_kl:.2f}")
        if epoch % 500 == 0:
            torch.save(model.state_dict(), f'./models/{city}_{name}_epoch{epoch}.pt')

    model.eval()
    all_embeddings = [None] * total_regions
    test_dataset = RegionGridDataset(data_tensor, test_region_ids)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for x, region_ids in test_loader:
            x = x.to(device)
            B, T, H_per_day, H, W = x.shape
            x = x.reshape(B * T * H_per_day, 1, H, W)
            mu, _ = model.encode(x)
            mu = mu.reshape(B, T, H_per_day, latent_dim).cpu()

            for i, rid in enumerate(region_ids):
                all_embeddings[rid] = mu[i]

    emb_tensor = torch.stack(all_embeddings, dim=0)
    emb_tensor = emb_tensor.permute(1, 2, 0, 3)
    torch.save(emb_tensor, f'./embeddings/{city}_{name}_embedding.pt')
    print(f"[{city}-{name}] Saved embedding of shape {emb_tensor.shape}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for city in ['XA', 'CD']:
        print(f"=== Processing city: {city} ===")
        prefix = f'../'
        speed, inflow, demand = loaddata_city(city, prefix)
        speed, inflow, demand = process_d_city(speed, inflow, demand)

        speed = speed.reshape(-1, 12, 1, 4, 10, 10)
        inflow = inflow.reshape(-1, 12, 1, 4, 10, 10)
        demand = demand.reshape(-1, 12, 1, 4, 10, 10)

        temp = torch.tensor(np.load(f'../humidity_{city.lower()}.npy')).float().to(device)
        hum = torch.tensor(np.load(f'../humidity_{city.lower()}.npy')).float().to(device)

        temp = temp.reshape(-1, 12, 1, 4, 10, 10)
        hum = hum.reshape(-1, 12, 1, 4, 10, 10)

        for name, data in zip(['speed', 'inflow', 'demand', 'hum', 'temp'],
                              [speed, inflow, demand, hum, temp]):
            train_vae_for_modality(city, name, data, device)

if __name__ == '__main__':
    main()
