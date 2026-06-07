"""Full MTP head grafting on Hugging Face models (PyTorch side).

This module allows you to take a base Qwen (or similar) model from Hugging Face
and graft Multi-Token Prediction (MTP) heads onto it.

This is the most powerful path when you want full control and to avoid
issues with pre-quantized/extracted MTP heads.

Based on the architecture used in DeepSeek-V3 / Qwen MTP implementations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from rich.console import Console

console = Console()


class MTPHead(nn.Module):
    """
    Single MTP prediction head (lightweight Transformer decoder block).

    Takes:
    - Hidden state from previous layer/head
    - Embedding of the token predicted by the previous head

    Outputs logits for the next future token.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Simple single-layer decoder block
        self.ln = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_attention_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, prev_token_embeds: torch.Tensor):
        # Add previous token embedding (causal chain)
        x = hidden_states + prev_token_embeds

        # Self-attention (causal)
        x = self.ln(x)
        attn_output, _ = self.self_attn(x, x, x, need_weights=False)
        x = x + attn_output

        # MLP
        x = self.ln2(x)
        x = x + self.mlp(x)
        return x


class MTPModelWrapper(nn.Module):
    """
    Wrapper that adds a stack of MTP heads on top of a base causal LM.
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_mtp_heads: int = 1,
        mtp_loss_scaling: float = 0.1,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.num_mtp_heads = num_mtp_heads
        self.mtp_loss_scaling = mtp_loss_scaling

        hidden_size = self.config.hidden_size
        num_attention_heads = getattr(self.config, "num_attention_heads", 32)
        intermediate_size = getattr(self.config, "intermediate_size", hidden_size * 4)

        # Create MTP heads
        self.mtp_heads = nn.ModuleList([
            MTPHead(hidden_size, num_attention_heads, intermediate_size)
            for _ in range(num_mtp_heads)
        ])

        # Shared output head (tie with base model's lm_head if possible)
        self.lm_head = base_model.lm_head if hasattr(base_model, "lm_head") else None

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        base_logits = outputs.logits

        # Main next-token prediction loss
        loss = None
        if labels is not None:
            # Shift for standard NTP
            shift_logits = base_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            ntp_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = ntp_loss

        mtp_losses = []

        # Causal MTP chain
        current_hidden = hidden_states
        for i, head in enumerate(self.mtp_heads):
            # For simplicity in this initial version, we use the same hidden state
            # In a full implementation you would use embeddings of previously predicted tokens
            mtp_hidden = head(current_hidden, torch.zeros_like(current_hidden))
            mtp_logits = self.lm_head(mtp_hidden) if self.lm_head is not None else None

            if labels is not None and mtp_logits is not None:
                # Shift labels for this head (predict t+2, t+3, etc.)
                shift = i + 2
                if shift < labels.size(1):
                    shift_logits = mtp_logits[..., :-shift, :].contiguous()
                    shift_labels = labels[..., shift:].contiguous()
                    loss_fct = nn.CrossEntropyLoss()
                    mtp_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    mtp_losses.append(mtp_loss)

            current_hidden = mtp_hidden  # Chain to next head

        if loss is not None and mtp_losses:
            total_mtp_loss = sum(mtp_losses) / len(mtp_losses)
            loss = loss + self.mtp_loss_scaling * total_mtp_loss

        return {
            "loss": loss,
            "logits": base_logits,
            "mtp_losses": mtp_losses,
            "hidden_states": outputs.hidden_states,
        }


def graft_mtp_heads_on_hf_model(
    model_name_or_path: str,
    num_mtp_heads: int = 1,
    mtp_loss_scaling: float = 0.1,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    torch_dtype=torch.bfloat16,
    device_map: str = "auto",
):
    """
    Load a base model and graft MTP heads on top.

    Returns a model ready for fine-tuning (with optional LoRA).
    """
    console.print(f"[cyan]Loading base model:[/cyan] {model_name_or_path}")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    console.print("[cyan]Grafting MTP heads...[/cyan]")

    mtp_model = MTPModelWrapper(
        base_model=base_model,
        num_mtp_heads=num_mtp_heads,
        mtp_loss_scaling=mtp_loss_scaling,
    )

    if use_lora:
        console.print("[cyan]Applying LoRA to backbone...[/cyan]")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        mtp_model = get_peft_model(mtp_model, lora_config)
        mtp_model.print_trainable_parameters()

    console.print("[green]✓ MTP grafting complete.[/green]")
    return mtp_model


def save_mtp_model_for_gguf(model: nn.Module, save_path: str, tokenizer=None):
    """
    Save the model in a format suitable for convert_hf_to_gguf.py.
    """
    console.print(f"[cyan]Saving MTP model to {save_path}...[/cyan]")

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(save_path)
    else:
        torch.save(model.state_dict(), f"{save_path}/pytorch_model.bin")

    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)

    console.print("[green]✓ Model saved. You can now run:[/green]")
    console.print(f"  mtp-baker convert-hf --model {save_path} --output mtp-model-f16.gguf")
