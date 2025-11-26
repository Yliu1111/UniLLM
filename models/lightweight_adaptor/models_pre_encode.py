import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SimpleVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super(SimpleVAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.flatten_dim = 64 * 3 * 3
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        x = self.encoder_cnn(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, 64, 3, 3)
        return self.decoder_cnn(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

class SingleGCNEncoder(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=64, latent_dim=64):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class MultiModalGCNVAE(nn.Module):
    def __init__(self, node_num=100, latent_dim=64):
        super().__init__()
        self.encoder_speed = SingleGCNEncoder()
        self.encoder_inflow = SingleGCNEncoder()
        self.encoder_demand = SingleGCNEncoder()

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        self.decoder_speed = nn.Linear(latent_dim, node_num)
        self.decoder_inflow = nn.Linear(latent_dim, node_num)
        self.decoder_demand = nn.Linear(latent_dim, node_num)

    def encode(self, x_s, x_i, x_d, ei_s, ei_i, ei_d):
        h_s = self.encoder_speed(x_s, ei_s)
        h_i = self.encoder_inflow(x_i, ei_i)
        h_d = self.encoder_demand(x_d, ei_d)

        mu_s = self.fc_mu(h_s)
        mu_i = self.fc_mu(h_i)
        mu_d = self.fc_mu(h_d)

        logvar_s = self.fc_logvar(h_s)
        logvar_i = self.fc_logvar(h_i)
        logvar_d = self.fc_logvar(h_d)

        mu = torch.stack([mu_s, mu_i, mu_d], dim=1)
        logvar = torch.stack([logvar_s, logvar_i, logvar_d], dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z_s, z_i, z_d = z[:, 0], z[:, 1], z[:, 2]
        rs = torch.tanh(self.decoder_speed(z_s))
        ri = torch.tanh(self.decoder_inflow(z_i))
        rd = torch.tanh(self.decoder_demand(z_d))
        return rs, ri, rd

    def forward(self, x_s, x_i, x_d, ei_s, ei_i, ei_d):
        mu, logvar = self.encode(x_s, x_i, x_d, ei_s, ei_i, ei_d)
        z = self.reparameterize(mu, logvar)
        rs, ri, rd = self.decode(z)
        return rs, ri, rd, mu, logvar

class TrajLSTMVAE(nn.Module):
    def __init__(self, input_dim=126, hidden_dim=64, latent_dim=64):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder_lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.encoder_lstm(packed)
        h = h_n[-1]
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def encode_sequence(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder_lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        return out

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, lengths):
        max_len = max(lengths)
        decoder_input = z.unsqueeze(1).repeat(1, max_len, 1)
        packed_out, _ = self.decoder_lstm(decoder_input)
        out = self.output_layer(packed_out)
        return out

    def forward(self, x, lengths):
        mu, logvar = self.encode(x, lengths)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, lengths)
        return recon_x, mu, logvar
