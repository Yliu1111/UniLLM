from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import torch
import torch.nn as nn
import torch.nn.functional as F

class FineTuner(nn.Module):
    def __init__(self, pred_len, model_name='meta-llama/Llama-3.2-1b',
                 hidden_dim=2048, output_dim=100, max_seq_len=64,
                 prompt_len=10, prompt_expert_num=4):
        super().__init__()
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                            r=8, lora_alpha=32, lora_dropout=0.05)
        self.model = get_peft_model(base_model, config)

        self.embedding_to_hidden = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.prompt_len = prompt_len
        self.prompt_expert_num = prompt_expert_num

        self.prompt_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.model.config.hidden_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, prompt_len * self.model.config.hidden_size)
            ) for _ in range(prompt_expert_num)
        ])

        self.router_fusion = nn.Sequential(
            nn.Conv1d(in_channels=prompt_expert_num, out_channels=1, kernel_size=1),
            nn.ReLU(),
        )

        self.sequence_pool = nn.Linear(max_seq_len, pred_len)

        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

    def get_routed_prompt(self, instruction_embed: torch.Tensor):
        expert_prompts = torch.stack([
            expert(instruction_embed).view(self.prompt_len, self.model.config.hidden_size)
            for expert in self.prompt_experts
        ], dim=0)

        expert_prompts = expert_prompts.permute(1, 2, 0)
        expert_prompts = expert_prompts.permute(0, 2, 1)
        fused = self.router_fusion(expert_prompts)
        routed_prompt = fused.squeeze(1)
        return routed_prompt

    def forward(self, input_embeds, attention_mask, pred_len):
        outputs = self.model(inputs_embeds=input_embeds,
                             attention_mask=attention_mask,
                             output_hidden_states=True,
                             return_dict=True)
        last_hidden = outputs.hidden_states[-1]
        reg_out = self.regression_head(last_hidden).transpose(1, 2)

        if reg_out.shape[-1] != self.sequence_pool.in_features:
            if reg_out.shape[-1] < self.sequence_pool.in_features:
                pad_len = self.sequence_pool.in_features - reg_out.shape[-1]
                reg_out = F.pad(reg_out, (0, pad_len))
            else:
                reg_out = reg_out[:, :, :self.sequence_pool.in_features]

        reg_out = self.sequence_pool(reg_out).transpose(1, 2)
        return reg_out


def embed_into_instruction_dynamic(tokenized_instruction, input_embeds,
                                   fine_tuner, tokenizer, device):
    with torch.no_grad():
        input_ids = tokenized_instruction["input_ids"]
        attention_mask_tok = tokenized_instruction["attention_mask"]
        embed_layer = fine_tuner.module.model.get_input_embeddings()
        instruction_embeds = embed_layer(input_ids).squeeze(0)
        instruction_mask = attention_mask_tok.squeeze(0)
        instruction_embed = (instruction_embeds * instruction_mask.unsqueeze(-1)).sum(dim=0) / instruction_mask.sum()

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
    prompt_embed = fine_tuner.module.get_routed_prompt(instruction_embed)

    inputs_embeds_list = []
    attention_mask_list = []

    for i in range(batch_size):
        inputs_embeds = torch.cat([
            prompt_embed,
            prefix_embeds,
            input_embeds_transformed[i],
            suffix_embeds
        ], dim=0)

        attention_mask_dynamic = torch.cat([
            torch.ones(fine_tuner.module.prompt_len, device=device),
            attention_mask[:start_idx],
            torch.ones(embed_length, device=device),
            attention_mask[end_idx + 1:]
        ])

        inputs_embeds_list.append(inputs_embeds)
        attention_mask_list.append(attention_mask_dynamic)

    inputs_embeds = torch.nn.utils.rnn.pad_sequence(inputs_embeds_list, batch_first=True, padding_value=0.0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

    return inputs_embeds, attention_mask, start_idx + fine_tuner.module.prompt_len
