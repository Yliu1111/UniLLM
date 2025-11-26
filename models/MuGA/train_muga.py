import os
import sys
import time
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.autograd import grad
from collections import deque
import numpy as np

sys.path.append('../')
sys.path.append('../modality_encoder')

from util import *
from finetune_model import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

alpha = 1.0
BUFFER_CAPACITY = 200
REPLAY_K = 2
expert_num = 2


class DayDataset(Dataset):
    def __init__(self, embedding_dir, raw_targets,
                 modalities, pred_len, city, aug=False):
        super().__init__()
        self.modalities = modalities
        self.pred_len = pred_len
        self.n_days = 30
        self.embeds = {
            m: torch.load(
                os.path.join(
                    embedding_dir,
                    f"../modality_encoder/modality_embeddings_xa_cd/{m}_embedding_aug_{aug}_{city}.pt"
                ),
                map_location="cpu"
            ).contiguous()
            for m in modalities
        }
        self.targets = {m: v.contiguous() for m, v in raw_targets.items()}

    def __len__(self):
        return self.n_days

    def __getitem__(self, day):
        sample = {}
        for m in self.modalities:
            x = self.embeds[m][day]
            x = x.squeeze()
            if x.shape[0] == 8 and x.shape[1] == 4:
                inp = x.permute(1, 0, 2).contiguous()
            elif x.shape[0] == 4 and x.shape[1] == 8:
                inp = x.contiguous()
            else:
                raise ValueError(f"Unexpected embed shape {x.shape} for {m}")
            tgt = (
                self.targets[m][day, 8:8 + self.pred_len, 0, :, :, :]
                .permute(1, 0, 2, 3)
                .reshape(4, self.pred_len, 100)
            )
            sample[m] = (inp, tgt)
        return day, sample


def collate_day(batch):
    day_ids, dicts = zip(*batch)
    merged = {}
    for t in dicts[0]:
        inp_list, tgt_list = zip(*[d[t] for d in dicts])
        merged[t] = (torch.stack(inp_list), torch.stack(tgt_list))
    return torch.tensor(day_ids), merged


def solve_pareto_weights(grads: torch.Tensor, max_iter: int = 10):
    G = grads @ grads.T
    n = grads.size(0)
    lam = torch.ones(n, device=grads.device) / n
    for t in range(max_iter):
        alpha = torch.argmin(G @ lam)
        d = torch.zeros_like(lam)
        d[alpha] = 1.
        eta = 2 / (t + 2)
        lam = (1 - eta) * lam + eta * d
    return lam


def compute_hypergradient(fine_tuner, tokenizer,
                          input_embed, target, instruction,
                          device, pred_len):
    tok = tokenizer(instruction, return_tensors="pt").to(device)
    embed_in, attn_mask, _ = embed_into_instruction_dynamic(
        tok, input_embed, fine_tuner, tokenizer, device
    )
    out = fine_tuner(embed_in, attn_mask, pred_len)
    loss = F.mse_loss(out, target, reduction='mean')
    params = [p for p in fine_tuner.parameters() if p.requires_grad]
    g = grad(loss, params, retain_graph=False, create_graph=False)
    return torch.cat([x.reshape(-1) for x in g])


def train_llm_with_muga(
        city, embedding_dir, raw_targets, device,
        save_dir="./finetuned_llm_muga_fast",
        epochs=50, lr=1e-4, pred_len=4,
        BUFFER_CAPACITY=BUFFER_CAPACITY, REPLAY_K=REPLAY_K,
        FW_INTERVAL=4,
        day_bs=16,
        num_workers=4):
    modalities = ["speed", "temperature", "humidity", "inflow", "demand"]
    aug = False
    n_train_region = 3
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1b")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<start>", "<end>"]})
    fine_tuner = torch.nn.DataParallel(
        FineTuner(pred_len, prompt_expert_num=expert_num)
    ).to(device)
    fine_tuner.module.model.resize_token_embeddings(len(tokenizer))
    opt = torch.optim.AdamW(fine_tuner.parameters(), lr=lr)
    fine_tuner.train()
    raw_targets_cpu = {k: v.cpu() for k, v in raw_targets.items()}
    ds = DayDataset(embedding_dir, raw_targets_cpu, modalities, pred_len, city, aug)
    dl = DataLoader(
        ds,
        batch_size=day_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_day
    )
    buffer = {m: deque(maxlen=BUFFER_CAPACITY) for m in modalities}
    lam = torch.ones(len(modalities), device=device) / len(modalities)
    region_step = 0
    for epoch in range(epochs):
        t0 = time.time()
        task_loss_sum = {m: 0. for m in modalities}
        task_cnt_sum = {m: 0 for m in modalities}
        for _, batch_dict in dl:
            if epoch == 0 and region_step == 0:
                for task, (inp, tgt) in batch_dict.items():
                    print(f"{task:<12}  inp={tuple(inp.shape)}  tgt={tuple(tgt.shape)}")
            region_order = list(range(n_train_region))
            random.shuffle(region_order)
            for r in region_order:
                region_step += 1
                if region_step % FW_INTERVAL == 0:
                    hyper_g = []
                    for t in modalities:
                        inp_r = (
                            batch_dict[t][0][:, r]
                            .reshape(-1, 8, 100)
                            .to(device, non_blocking=True)
                        )
                        tgt_r = (
                            batch_dict[t][1][:, r]
                            .reshape(-1, pred_len, 100)
                            .to(device, non_blocking=True)
                        )
                        instr = (
                            f"Given the historical 8 hours data: <start> <end> "
                            f"in urban {t} in city {city} in region {r}, "
                            f"predict the next {pred_len} hours."
                        )
                        g_parts = [
                            compute_hypergradient(
                                fine_tuner, tokenizer,
                                inp_r, tgt_r,
                                instruction=instr, device=device, pred_len=pred_len
                            )
                        ]
                        for (ib_c, tb_c, instr_b, _) in random.sample(
                                buffer[t], k=min(REPLAY_K, len(buffer[t]))):
                            g_parts.append(
                                compute_hypergradient(
                                    fine_tuner, tokenizer,
                                    ib_c.to(device, non_blocking=True),
                                    tb_c.to(device, non_blocking=True),
                                    instr_b, device, pred_len
                                )
                            )
                        hyper_g.append(torch.stack(g_parts).mean(0))
                    lam = solve_pareto_weights(torch.stack(hyper_g))
                    lam = torch.clamp(lam, 0.10, 0.30)
                    lam /= lam.sum()
                opt.zero_grad()
                losses, loss_vals = [], []
                for idx_t, t in enumerate(modalities):
                    inp_r = (
                        batch_dict[t][0][:, r]
                        .reshape(-1, 8, 100)
                        .to(device, non_blocking=True)
                    )
                    tgt_r = (
                        batch_dict[t][1][:, r]
                        .reshape(-1, pred_len, 100)
                        .to(device, non_blocking=True)
                    )
                    instr = (
                        f"Given the historical 8 hours data: <start> <end> "
                        f"in urban {t} in city {city} in region {r}, "
                        f"predict the next {pred_len} hours."
                    )
                    tok = tokenizer(instr, return_tensors="pt").to(device)
                    emb_in, mask, _ = embed_into_instruction_dynamic(
                        tok, inp_r, fine_tuner, tokenizer, device
                    )
                    pred = fine_tuner(emb_in, mask, pred_len)
                    loss = F.mse_loss(pred, tgt_r)
                    losses.append(loss)
                    loss_vals.append(loss.item())
                total_loss = (lam * torch.stack(losses)).sum()
                total_loss.backward()
                opt.step()
                for t, lv in zip(modalities, loss_vals):
                    inp_batch = batch_dict[t][0][:, r].cpu()
                    tgt_batch = batch_dict[t][1][:, r].cpu()
                    buffer[t].append((
                        inp_batch, tgt_batch,
                        f"Given the historical 8 hours data: <start> <end> "
                        f"in urban {t} in city {city} in region {r}, "
                        f"predict the next {pred_len} hours.",
                        lv
                    ))
                    task_loss_sum[t] += lv
                    task_cnt_sum[t] += 1
        elapsed = time.time() - t0
        avg_total = sum(task_loss_sum.values()) / sum(task_cnt_sum.values())
        print(f"[Epoch {epoch + 1}/{epochs}] {elapsed / 60:.1f} min  "
              f"total_loss={avg_total:.4f} λ={lam.cpu().numpy().round(3)}")
        for m in modalities:
            print(f"{m:>12}: {task_loss_sum[m] / task_cnt_sum[m]:.4f}")
    fine_tuner.eval()
    save_dir = f"{save_dir}_forecast_aug_{aug}_{pred_len}h"
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = f"{save_dir}/fine_tuner_router_full_REPLAY_K_{REPLAY_K}_{city}.pt"
    torch.save(
        {
            "fine_tuner": fine_tuner.state_dict(),
            "tokenizer": tokenizer.name_or_path
        },
        ckpt_path
    )
    print("FineTuner saved to", ckpt_path)


@torch.no_grad__()
def evaluate_llm_with_muga(
        city,
        embedding_dir,
        raw_targets,
        device,
        pred_lens=(1, 2, 3, 4),
        num_eval_regions=1,
        aug=False,
        ckpt_prefix="./finetuned_llm_muga_fast",
        day_bs=16,
        num_workers=0):
    modalities = ["speed", "temperature", "humidity", "inflow", "demand"]
    targets_cpu = {k: v.cpu() for k, v in raw_targets.items()}
    all_regions = list(range(targets_cpu["speed"].shape[3]))
    eval_regions = [3]
    print("Eval regions:", eval_regions)
    overall_rmse = {t: [] for t in modalities}
    overall_mae = {t: [] for t in modalities}
    for k in pred_lens:
        print(f"\nEvaluating {k}-hour checkpoint ...")
        ds = DayDataset(embedding_dir, targets_cpu, modalities, pred_len=k, city=city, aug=aug)
        dl = DataLoader(
            ds,
            batch_size=day_bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_day
        )
        ckpt_dir = f"{ckpt_prefix}_forecast_aug_{aug}_{k}h"
        ckpt_path = f"{ckpt_dir}/fine_tuner_router_full_REPLAY_K_{REPLAY_K}_{city}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        tokenizer = AutoTokenizer.from_pretrained(ckpt["tokenizer"])
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.add_special_tokens({"additional_special_tokens": ["<start>", "<end>"]})
        vocab_len = len(tokenizer)
        ft = FineTuner(k, prompt_expert_num=expert_num).to(device)
        ft.model.resize_token_embeddings(vocab_len)
        clean_state = {name.replace("module.", ""): v
                       for name, v in ckpt["fine_tuner"].items()}
        ft.load_state_dict(clean_state, strict=True)
        ft.eval()
        ft.module = ft
        task_rmse_sum = {t: 0.0 for t in modalities}
        task_mae_sum = {t: 0.0 for t in modalities}
        task_cnt = {t: 0 for t in modalities}
        for _, batch_dict in dl:
            for r in eval_regions:
                for t in modalities:
                    inp = batch_dict[t][0][:, r].to(device)
                    tgt = batch_dict[t][1][:, r].to(device)
                    instr = (
                        f"Given the historical 8 hours data: <start> <end> "
                        f"in urban {t} in city {city} in region {r}, "
                        f"predict the next {k} hours."
                    )
                    tok = tokenizer(instr, return_tensors="pt").to(device)
                    emb_in, mask, _ = embed_into_instruction_dynamic(
                        tok, inp, ft, tokenizer, device
                    )
                    pred = ft(emb_in, mask, k)
                    if t == "speed" and r == 3 and k == 1:
                        print("pred[0][0][:5] =", pred[0, 0, :5].cpu().numpy())
                        print("tgt[0][0][:5]  =", tgt[0, 0, :5].cpu().numpy())
                        print("abs error      =", (pred[0, 0, :5] - tgt[0, 0, :5]).abs().cpu().numpy())
                    abs_error = (pred - tgt).abs()
                    rmse_sample = ((pred - tgt) ** 2).mean(dim=-1).sqrt()
                    mae_sample = abs_error.mean(dim=-1)
                    avg_rmse = rmse_sample.mean().item()
                    avg_mae = mae_sample.mean().item()
                    task_rmse_sum[t] += avg_rmse
                    task_mae_sum[t] += avg_mae
                    task_cnt[t] += 1
        print("    task          RMSE     MAE")
        for t in modalities:
            rmse = task_rmse_sum[t] / task_cnt[t]
            mae = task_mae_sum[t] / task_cnt[t]
            overall_rmse[t].append(rmse)
            overall_mae[t].append(mae)
            print(f"    {t:<12}  {rmse:7.4f}  {mae:7.4f}")
    print("\n========= Average over all pred_lens =========")
    print("    task          Avg RMSE    Avg MAE")
    for t in modalities:
        rmse_avg = sum(overall_rmse[t]) / len(overall_rmse[t])
        mae_avg = sum(overall_mae[t]) / len(overall_mae[t])
        print(f"    {t:<12}  {rmse_avg:9.4f}  {mae_avg:9.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cities = ["XA"]
    for city in cities:
        print(f"\nStart MUGA-based LLM training & evaluation on {city}")
        # NOTE: update these paths if you move the data files
        if city == "XA":
            speed = torch.tensor(np.load("../speed_XA.npy")).float().to(device)
            inflow = torch.tensor(np.load("../inflow_XA.npy")).float().to(device)
            demand = torch.tensor(np.load("../demand_XA.npy")).float().to(device)
            temp = torch.tensor(np.load("../temp_xa.npy")).float().to(device)
            hum = torch.tensor(np.load("../humidity_xa.npy")).float().to(device)
        elif city == "CD":
            speed = torch.tensor(np.load("../speed_CD.npy")).float().to(device)
            inflow = torch.tensor(np.load("../inflow_CD.npy")).float().to(device)
            demand = torch.tensor(np.load("../demand_CD.npy")).float().to(device)
            temp = torch.tensor(np.load("../temp_cd.npy")).float().to(device)
            hum = torch.tensor(np.load("../humidity_cd.npy")).float().to(device)
        else:
            raise ValueError(f"Unsupported city: {city}")
        speed, demand, inflow = process_small(speed, demand, inflow)
        speed = speed.reshape(-1, 12, 1, 4, 10, 10)
        inflow = inflow.reshape(-1, 12, 1, 4, 10, 10)
        demand = demand.reshape(-1, 12, 1, 4, 10, 10)
        temp = normalize(temp).unsqueeze(1).reshape(-1, 12, 1, 4, 10, 10)
        hum = hum.unsqueeze(1).reshape(-1, 12, 1, 4, 10, 10)
        raw_targets = {
            "speed": speed,
            "inflow": inflow,
            "demand": demand,
            "temperature": temp,
            "humidity": hum
        }
        embedding_dir = ""
        for pred_len in [1, 2, 3, 4]:
            train_llm_with_muga(city, embedding_dir, raw_targets, device, pred_len=pred_len)
        evaluate_llm_with_muga(
            city,
            embedding_dir,
            raw_targets,
            device=device,
            pred_lens=[1, 2, 3, 4],
            num_eval_regions=1
        )


if __name__ == '__main__':
    main()
