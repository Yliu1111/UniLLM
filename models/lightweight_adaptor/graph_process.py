import os
import torch
import torch.optim as optim
import sys
sys.path.append('../../')
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import dense_to_sparse
from models_pre_encode import MultiModalGCNVAE
from models.util import loaddata_city, process_d_city

os.makedirs('./models', exist_ok=True)
os.makedirs('./embeddings', exist_ok=True)

class RegionGCNDataset(Dataset):
    def __init__(self, speed, inflow, demand, region_indices):
        self.speed = speed
        self.inflow = inflow
        self.demand = demand
        self.region_indices = region_indices

    def __len__(self):
        return len(self.region_indices)

    def __getitem__(self, idx):
        r = self.region_indices[idx]
        x_s = self.speed[:, :, :, r, :, :].squeeze(2)
        x_i = self.inflow[:, :, :, r, :, :].squeeze(2)
        x_d = self.demand[:, :, :, r, :, :].squeeze(2)
        return x_s, x_i, x_d, r

def train_multimodal_vae(speed, inflow, demand, speed_edge, inflow_edge, demand_edge, device, city, epochs=1000, batch_size=1):
    total_regions = speed.shape[3]
    train_region_ids = list(range(total_regions - 1))
    test_region_ids = [total_regions - 1]

    train_dataset = RegionGCNDataset(speed, inflow, demand, train_region_ids)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = MultiModalGCNVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def edge_to_index(edge):
        edge = torch.tensor(edge).float()
        return dense_to_sparse(edge)[0].to(device)

    edge_index_s = edge_to_index(speed_edge[0])
    edge_index_i = edge_to_index(inflow_edge[0])
    edge_index_d = edge_to_index(demand_edge[0])

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for xs, xi, xd, _ in train_loader:
            xs, xi, xd = xs.to(device).float(), xi.to(device).float(), xd.to(device).float()
            B, T, H_per_day, H, W = xs.shape
            xs = xs.view(T * H_per_day, 100)
            xi = xi.view(T * H_per_day, 100)
            xd = xd.view(T * H_per_day, 100)

            rs, ri, rd, mu, logvar = model(xs, xi, xd, edge_index_s, edge_index_i, edge_index_d)
            rs = rs.view(-1, 100)
            ri = ri.view(-1, 100)
            rd = rd.view(-1, 100)

            loss_recon = F.mse_loss(rs, xs) + F.mse_loss(ri, xi) + F.mse_loss(rd, xd)
            loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            kl_weight = 0.5 * (1 - math.cos(min(epoch / 500, 1.0) * math.pi)) * 0.1
            loss = loss_recon + kl_weight * loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[{city}] Epoch {epoch} | Total Loss: {total_loss:.2f} | Recon Loss: {loss_recon.item():.2f} | KL Loss: {loss_kl.item():.4f}")
        if epoch % 500 == 0:
            torch.save(model.state_dict(), f'./models/gcnvae_{city}_epoch{epoch}.pt')

    model.eval()
    test_region_ids = list(range(total_regions))
    test_dataset = RegionGCNDataset(speed, inflow, demand, test_region_ids)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    all_embeddings = [None] * total_regions

    with torch.no_grad():
        for xs, xi, xd, rids in test_loader:
            xs, xi, xd = xs.to(device).float(), xi.to(device).float(), xd.to(device).float()
            xs = xs.view(-1, 100)
            xi = xi.view(-1, 100)
            xd = xd.view(-1, 100)

            mu, _ = model.encode(xs, xi, xd, edge_index_s, edge_index_i, edge_index_d)
            mu = mu.view(30, 12, 3, -1).cpu()
            all_embeddings[rids[0]] = mu

    for i, emb in enumerate(all_embeddings):
        if emb is None:
            raise ValueError(f"Embedding for region {i} is missing!")

    emb_tensor = torch.stack(all_embeddings, dim=0)
    emb_tensor = emb_tensor.permute(1, 2, 0, 3, 4)
    torch.save(emb_tensor, f'./embeddings/gcnvae_embedding_{city}.pt')
    print(f'{city}: saved embedding tensor of shape {emb_tensor.shape}')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for city in ['XA', 'CD']:
        print(f"=== Processing city: {city} ===")

        prefix = f'../'
        speed, inflow, demand = loaddata_city(city, prefix)
        speed, inflow, demand = process_d_city(speed, inflow, demand)

        speed_edge = np.load(f'../region_correlation_speed_{city}.npy')
        inflow_edge = np.load(f'../region_correlation_inflow_{city}.npy')
        demand_edge = np.load(f'../region_correlation_demand_{city}.npy')

        speed = speed.reshape(-1, 12, 1, 4, 10, 10)
        inflow = inflow.reshape(-1, 12, 1, 4, 10, 10)
        demand = demand.reshape(-1, 12, 1, 4, 10, 10)

        train_multimodal_vae(speed, inflow, demand, speed_edge, inflow_edge, demand_edge, device, city)

if __name__ == '__main__':
    main()
