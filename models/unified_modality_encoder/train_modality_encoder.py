import os
import torch
import torch.optim as optim
import sys

sys.path.append('../../')
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import random
from models.util import *
import math
from modality_model import *
from transformers import AutoTokenizer, AutoModelForCausalLM

def save_encoder_and_embeddings(city, encoder, all_embeddings, save_dir, save_embed_dir):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_embed_dir, exist_ok=True)

    torch.save(encoder.state_dict(), os.path.join(save_dir, f"multi_encoder_{city}.pt"))

    encoder.eval()
    result_list = []

    with torch.no_grad():
        for region_idx in range(4):
            inputs = {
                "speed": all_embeddings["speed"][:, :8, region_idx, :],
                "temperature": all_embeddings["temperature"][:, :8, region_idx, :],
                "humidity": all_embeddings["humidity"][:, :8, region_idx, :],
                "inflow": all_embeddings["inflow"][:, :8, region_idx, :],
                "demand": all_embeddings["demand"][:, :8, region_idx, :],
            }
            for k in inputs:
                inputs[k] = inputs[k].to(next(encoder.parameters()).device)
                inputs[k] = inputs[k].reshape(30, 8, -1, 64)

            output, modalities = encoder(inputs)
            output = output.permute(0, 1, 3, 2, 4)
            result_list.append(output.unsqueeze(2))

    full_embedding = torch.cat(result_list, dim=2)

    for i, modality in enumerate(modalities):
        emb = full_embedding[:, :, :, i, :, :]
        torch.save(emb, os.path.join(save_embed_dir, f"{modality}_embedding_{city}.pt"))
        print(f" Saved {modality} embedding in city {city}: shape {emb.shape}")

def train(city, train_embeddings, raw_inputs, raw_targets, full_embeddings, device, epochs=100, lr=1e-4, alpha=0.2):
    encoder = UnifiedMultiEncoder(
        base_encoder=MultiModalityEncoder(),
        device=device,
        channels=1,
        num_nodes=100,
        num_modals=len(train_embeddings),
        n_his=8
    ).to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    model_name = "meta-llama/Llama-3.2-1b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "<start>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<start>", "<end>"]})

    fine_tuner = FineTuner(model_name=model_name).to(device)
    fine_tuner.model.resize_token_embeddings(len(tokenizer))
    fine_tuner = torch.nn.DataParallel(fine_tuner, device_ids=[0, 1, 2, 3])
    fine_tuner.module.model.eval()
    fine_tuner.train()

    task_list = list(train_embeddings.keys())
    encoder.train()

    for epoch in range(epochs):
        total_loss = 0
        task_loss_sum = {task: 0.0 for task in task_list}
        region_idxs = list(range(3))
        random.shuffle(region_idxs)

        for region_idx in region_idxs:
            embed_inputs = {k: v[:, :, region_idx, :].reshape(30, 8, -1, 64).to(device) for k, v in train_embeddings.items()}
            raw_inputs_region = {k: raw_inputs[k][:, :8, :, region_idx, :, :].reshape(30, 8, -1, 100).to(device) for k in train_embeddings}
            targets = {k: raw_targets[k][:, 8:12, :, region_idx, :, :].reshape(-1, 4, 100).to(device) for k in task_list}

            encoder.zero_grad()
            fine_tuner.zero_grad()

            final_rep, modalities, loss_cl = encoder(embed_inputs, raw_inputs_region)

            task_losses = []
            for task in task_list:
                task_idx = modalities.index(task)
                task_embed = final_rep[:, :, :, task_idx, :].reshape(30, 8, 100)

                instr = f"Given the historical 8 hours data: <start> <end> in urban {task} in city {city} in region {region_idx}, predict the next 4 hours."
                tokenized = tokenizer(instr, return_tensors="pt").to(device)

                embed_input, attn_mask, start_idx = embed_into_instruction_dynamic(
                    tokenized, task_embed, fine_tuner, tokenizer, device
                )

                llm_output = fine_tuner(input_embeds=embed_input, attention_mask=attn_mask)
                pred = llm_output[:, start_idx:start_idx + 4, :]

                loss = F.mse_loss(pred, targets[task])
                task_losses.append(loss)
                task_loss_sum[task] += loss.item()

            task_loss_total = torch.stack(task_losses).mean()
            total_combined_loss = task_loss_total + alpha * loss_cl
            total_combined_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            total_loss += task_loss_total.item() + loss_cl.item()

        avg_task_loss = {task: task_loss_sum[task] / len(region_idxs) for task in task_list}
        print(f"[UnifiedMultiEncoder] {city} Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss / len(region_idxs):.4f}")
        for task, loss_val in avg_task_loss.items():
            print(f"  - Avg {task} loss: {loss_val:.4f}")

    save_encoder_and_embeddings(city, encoder.base_encoder, full_embeddings,
                                f"./modality_models_xa_cd", f"./modality_embeddings_xa_cd", alpha=alpha)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cities = ["XA"]

    for city in cities:
        print(f"\n🚀 Start training on {city}")

        if city == "XA":
            speed = torch.tensor(np.load("../speed_XA.npy")).float().to(device)
            inflow = torch.tensor(np.load("../inflow_XA.npy")).float().to(device)
            demand = torch.tensor(np.load("../demand_XA.npy")).float().to(device)
            temp = torch.tensor(np.load("../temp_xa.npy")).float().to(device)
            hum = torch.tensor(np.load("../humidity_xa.npy")).float().to(device)
            temp_emb = torch.load("../XA_temp_embedding.pt").to(device)
            hum_emb = torch.load("../XA_hum_embedding.pt").to(device)
            gcnvae_emb = torch.load("../gcnvae_embedding_XA.pt").to(device)
        else:
            raise ValueError("Unsupported city")

        speed = speed.reshape(-1, 12, 1, 4, 10, 10)
        inflow = inflow.reshape(-1, 12, 1, 4, 10, 10)
        demand = demand.reshape(-1, 12, 1, 4, 10, 10)
        temp = temp[:30].unsqueeze(1).reshape(-1, 12, 1, 4, 10, 10)
        hum = hum[:30].unsqueeze(1).reshape(-1, 12, 1, 4, 10, 10)
        temp_emb = temp_emb[:30]
        hum_emb = hum_emb[:30]


        speed, demand, inflow = process_small(speed, demand, inflow)

        train_embeddings = {
            "speed": gcnvae_emb[:, :8, :3, 0, :],
            "temperature": temp_emb[:30, :8, :3, :],
            "humidity": hum_emb[:30, :8, :3, :],
            "inflow": gcnvae_emb[:, :8, :3, 1, :],
            "demand": gcnvae_emb[:, :8, :3, 2, :],
        }

        full_embeddings = {
            "speed": gcnvae_emb[:, :8, :, 0, :],
            "temperature": temp_emb[:30, :8, :, :],
            "humidity": hum_emb[:30, :8, :, :],
            "inflow": gcnvae_emb[:, :8, :, 1, :],
            "demand": gcnvae_emb[:, :8, :, 2, :],
        }

        raw_inputs = {
            "speed": speed,
            "inflow": inflow,
            "demand": demand,
            "temperature": temp,
            "humidity": hum,
        }

        raw_targets = raw_inputs
        train(city, train_embeddings, raw_inputs, raw_targets, full_embeddings, device)

if __name__ == '__main__':
    main()
