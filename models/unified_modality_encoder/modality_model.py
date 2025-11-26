import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import math
import numpy as np

class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)

    def forward(self, x):
        B, T, NM, D = x.shape
        x = x.reshape(B * NM, T, D)
        out, _ = self.attn(x, x, x)
        return out.reshape(B, T, NM, D)

class SpatialCNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)

    def forward(self, x):
        B, T, NM, D = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B * T, D, NM)
        x = self.conv(x)
        x = x.reshape(B, T, -1, NM).permute(0, 1, 3, 2)
        return x

class ModalityAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        attn_score = torch.einsum("btnmd,btnkd->tnmk", Q, K) / (Q.shape[-1] ** 0.5)
        attn_weight = self.softmax(attn_score)
        return torch.einsum("tnmk,btnkd->btnmd", attn_weight, V)

class MultiModalityEncoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=100):
        super().__init__()
        self.temp_attn = TemporalAttention(input_dim)
        self.spatial_cnn = SpatialCNN(input_dim, hidden_dim)
        self.temp_attn2 = TemporalAttention(hidden_dim)
        self.modality_attn = ModalityAttention(hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, inputs):
        modalities = list(inputs.keys())
        x = torch.stack([inputs[m] for m in modalities], dim=3)
        B, T, N, M, D = x.shape
        x = x.permute(0, 1, 3, 2, 4).reshape(B, T, M * N, D)
        x = self.temp_attn(x)
        x = self.spatial_cnn(x)
        x = self.temp_attn2(x)
        x = x.reshape(B, T, M, N, -1).permute(0, 1, 3, 2, 4)
        x = self.modality_attn(x)
        x = self.output_proj(x)
        return x, modalities

class DirectContrastive(nn.Module):
    def __init__(self, channels, num_nodes, num_modals, device):
        super().__init__()
        self.device = device
        self.net = nn.Bilinear(channels, channels, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.proj = nn.Linear(channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, rep):
        B, C, M, N, T = rep.shape
        rep = rep.permute(0, 2, 3, 4, 1)
        h_rl = rep.mean(dim=3)
        h_fk = h_rl[:, torch.randperm(M), :, :]
        cm = self.sigmoid(self.proj(h_rl.mean(dim=2)))
        cm_expand = cm.unsqueeze(2).expand_as(h_rl)
        score_real = self.net(h_rl, cm_expand)
        score_fake = self.net(h_fk, cm_expand)
        logits = torch.cat((score_real, score_fake), dim=-1)
        labels = torch.cat((torch.ones_like(score_real), torch.zeros_like(score_fake)), dim=-1)
        return self.loss_fn(logits, labels)

class UnifiedMultiEncoder(nn.Module):
    def __init__(self, base_encoder, device, num_nodes=100, num_modals=5, n_his=8, channels=64):
        super().__init__()
        self.base_encoder = base_encoder
        self.direct_mssl = DirectContrastive(channels, num_nodes, num_modals, device)

    def forward(self, inputs, raw_inputs):
        base_out, modalities = self.base_encoder(inputs)
        rep = base_out.permute(0, 2, 3, 4, 1)
        loss = self.direct_mssl(rep)
        final_rep = base_out
        return base_out, final_rep, loss, modalities

class FineTuner(nn.Module):
    def __init__(self, model_name='meta-llama/Llama-3.2-1b', hidden_dim=2048, output_dim=100):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        self.embedding_to_hidden = nn.Sequential(
            nn.LayerNorm(100),
            nn.Linear(100, hidden_dim),
            nn.ReLU(),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, input_embeds, attention_mask):
        outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True,
                             return_dict=True)
        last_hidden = outputs.hidden_states[-1]
        return self.regression_head(last_hidden)

def embed_into_instruction_dynamic(tokenized_instruction, input_embeds, fine_tuner, tokenizer, device):
    input_ids = tokenized_instruction["input_ids"].squeeze(0)
    attention_mask = tokenized_instruction["attention_mask"].squeeze(0)
    batch_size, embed_length, _ = input_embeds.shape

    start_token_id = tokenizer.convert_tokens_to_ids("<start>")
    end_token_id = tokenizer.convert_tokens_to_ids("<end>")
    if start_token_id is None or end_token_id is None:
        raise ValueError("Tokenizer does not contain <start> or <end> tokens.")

    start_idx = (input_ids == start_token_id).nonzero(as_tuple=True)[0].item()
    end_idx = (input_ids == end_token_id).nonzero(as_tuple=True)[0].item()

    embedding_layer = fine_tuner.module.model.get_input_embeddings()
    embedding_to_hidden = fine_tuner.module.embedding_to_hidden

    prefix_ids = input_ids[:start_idx]
    suffix_ids = input_ids[end_idx + 1:]

    prefix_embeds = embedding_layer(prefix_ids).to(device)
    suffix_embeds = embedding_layer(suffix_ids).to(device)
    input_embeds_transformed = embedding_to_hidden(input_embeds)

    inputs_embeds_list = []
    attention_mask_list = []

    for i in range(batch_size):
        inputs_embeds = torch.cat([
            prefix_embeds,
            input_embeds_transformed[i],
            suffix_embeds
        ], dim=0)
        attention_mask_dynamic = torch.cat([
            attention_mask[:start_idx],
            torch.ones(embed_length, device=device),
            attention_mask[end_idx + 1:]
        ])
        inputs_embeds_list.append(inputs_embeds)
        attention_mask_list.append(attention_mask_dynamic)

    inputs_embeds = nn.utils.rnn.pad_sequence(inputs_embeds_list, batch_first=True, padding_value=0.0)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

    return inputs_embeds, attention_mask, start_idx
