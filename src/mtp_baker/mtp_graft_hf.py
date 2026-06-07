"""Full MTP head grafting on Hugging Face models (PyTorch side) - Refined v0.6

This is an improved implementation with proper teacher-forcing causal chain
for MTP training (using ground-truth future token embeddings).

This gives you full control to create custom MTP models from any base HF model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

from rich.console import Console

console = Console()


class MTPHead(nn.Module):
    """
    Lightweight single-layer MTP prediction head.

    Implements the causal chain:
    - Takes hidden state from previous stage
    - Adds embedding of the token predicted in the previous step (teacher forcing during training)
    - Processes through attention + MLP
    - Outputs hidden state for next head / logits
    """

    def __init__(self, hidden_size: int, num_attention_heads: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.ln = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_attention_heads, batch_first=True, dropout=0.0)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, prev_token_embeds: torch.Tensor):
        # Causal chain: add embedding of previous predicted token
        x = hidden_states + prev_token_embeds

        # Layer norm + self-attention
        x_norm = self.ln(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_output

        # MLP
        x_norm = self.ln2(x)
        x = x + self.mlp(x_norm)
        return x


class MTPModelWrapper(nn.Module):
    """
    Wraps a base causal language model and adds multiple MTP heads
    with proper causal chaining.
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

        self.mtp_heads = nn.ModuleList([
            MTPHead(hidden_size, num_attention_heads, intermediate_size)
            for _ in range(num_mtp_heads)
        ])

        # Tie lm_head if available
        self.lm_head = getattr(base_model, "lm_head", None)

        # Get embedding layer for teacher forcing
        self.embed_tokens = base_model.get_input_embeddings()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden = outputs.hidden_states[-1]
        base_logits = outputs.logits

        loss = None
        mtp_losses = []

        if labels is not None:
            # === Main Next Token Prediction Loss ===
            shift_logits = base_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            ntp_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = ntp_loss

            # === MTP Heads with proper causal chain (teacher forcing) ===
            current_hidden = last_hidden

            for i, head in enumerate(self.mtp_heads):
                # Get embedding of the ground-truth token at position t + (i+1)
                # This is the key improvement for correct causal MTP
                shift = i + 1
                if shift < input_ids.size(1):
                    prev_token_ids = labels[..., shift:] if labels is not None else input_ids[..., shift:]
                    # Clamp to valid range
                    prev_token_ids = torch.clamp(prev_token_ids, min=0)
                    prev_embeds = self.embed_tokens(prev_token_ids)

                    # Pad to match sequence length if needed
                    if prev_embeds.size(1) < current_hidden.size(1):
                        pad_len = current_hidden.size(1) - prev_embeds.size(1)
                        prev_embeds = torch.nn.functional.pad(prev_embeds, (0, 0, 0, pad_len))
                    elif prev_embeds.size(1) > current_hidden.size(1):
                        prev_embeds = prev_embeds[:, :current_hidden.size(1), :]

                    mtp_hidden = head(current_hidden, prev_embeds)
                else:
                    mtp_hidden = head(current_hidden, torch.zeros_like(current_hidden))

                # Get logits from this head
                if self.lm_head is not None:
                    mtp_logits = self.lm_head(mtp_hidden)
                else:
                    mtp_logits = None

                # Compute MTP loss for this head (predict further into the future)
                if labels is not None and mtp_logits is not None:
                    head_shift = i + 2  # Head 0 predicts t+2, Head 1 predicts t+3, etc.
                    if head_shift < labels.size(1):
                        shift_logits_mtp = mtp_logits[..., :-head_shift, :].contiguous()
                        shift_labels_mtp = labels[..., head_shift:].contiguous()
                        mtp_loss = loss_fct(
                            shift_logits_mtp.view(-1, shift_logits_mtp.size(-1)),
                            shift_labels_mtp.view(-1)
                        )
                        mtp_losses.append(mtp_loss)

                current_hidden = mtp_hidden

            # Combine losses
            if mtp_losses:
                avg_mtp_loss = sum(mtp_losses) / len(mtp_losses)
                loss = loss + self.mtp_loss_scaling * avg_mtp_loss

        return {
            "loss": loss,
            "logits": base_logits,
            "mtp_losses": mtp_losses if mtp_losses else None,
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
    trust_remote_code: bool = True,
):
    """
    Load a base model and graft MTP heads on top.

    Returns a model ready for fine-tuning (with optional LoRA).
    """
    console.print(f"[cyan]Loading base model from Hugging Face:[/cyan] {model_name_or_path}")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )

    console.print(f"[cyan]Grafting {num_mtp_heads} MTP head(s) with improved causal chain...[/cyan]")

    mtp_model = MTPModelWrapper(
        base_model=base_model,
        num_mtp_heads=num_mtp_heads,
        mtp_loss_scaling=mtp_loss_scaling,
    )

    if use_lora:
        console.print("[cyan]Applying LoRA adapters to backbone...[/cyan]")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        mtp_model = get_peft_model(mtp_model, lora_config)
        mtp_model.print_trainable_parameters()

    console.print("[bold green]✓ MTP grafting complete with improved causal chain.[/bold green]")
    return mtp_model


def save_mtp_model_for_gguf(model: nn.Module, save_directory: str, tokenizer=None):
    """
    Save the grafted model so it can be converted to GGUF later.
    """
    console.print(f"[cyan]Saving MTP model to {save_directory}...[/cyan]")

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(save_directory)
    else:
        torch.save(model.state_dict(), f"{save_directory}/pytorch_model.bin")

    if tokenizer is not None:
        tokenizer.save_pretrained(save_directory)

    console.print("[green]✓ Model saved successfully.[/green]")
    console.print("You can now convert it with:")
    console.print(f"  mtp-baker convert-hf --model {save_directory} --output mtp-model-f16.gguf")
