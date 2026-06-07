"""Verification of MTP tensors in GGUF files."""

from __future__ import annotations

from pathlib import Path

from gguf import GGUFReader
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# Common patterns for MTP / draft head tensors in Qwen models
MTP_TENSOR_PATTERNS = [
    "mtp",
    "nextn",
    "draft",
    "spec",
    "ssm_conv1d",      # Seen in some Qwen MTP-related exports
    "head.",
    "mtp_head",
]


def verify_mtp_tensors(gguf_path: str | Path, show_tensors: bool = False) -> str:
    """Check a GGUF file for presence of MTP-related tensors."""
    gguf_path = Path(gguf_path).expanduser().resolve()

    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

    reader = GGUFReader(str(gguf_path))

    all_tensor_names = [t.name for t in reader.tensors]
    mtp_related = []

    for name in all_tensor_names:
        name_lower = name.lower()
        if any(pattern in name_lower for pattern in MTP_TENSOR_PATTERNS):
            mtp_related.append(name)

    # Build report
    lines = []
    lines.append(f"[bold]File:[/bold] {gguf_path}")
    lines.append(f"[bold]Total tensors:[/bold] {len(all_tensor_names)}")
    lines.append(f"[bold]MTP-related tensors found:[/bold] {len(mtp_related)}")

    if mtp_related:
        lines.append("\n[green]✓ MTP tensors detected![/green]")
        if show_tensors:
            table = Table(title="MTP-related Tensors")
            table.add_column("Tensor Name", style="cyan")
            table.add_column("Shape", style="magenta")
            table.add_column("Dtype", style="yellow")

            for name in mtp_related:
                tensor = next((t for t in reader.tensors if t.name == name), None)
                if tensor:
                    shape_str = " × ".join(map(str, tensor.shape)) if tensor.shape else "scalar"
                    table.add_row(name, shape_str, str(tensor.dtype))
            lines.append(table)
    else:
        lines.append("\n[red]✗ No MTP-related tensors detected.[/red]")
        lines.append("This GGUF probably does not have working MTP heads.")

    # Also check for common metadata flags
    metadata_keys = list(reader.metadata.keys())
    mtp_meta = [k for k in metadata_keys if any(p in k.lower() for p in ["mtp", "nextn", "speculative", "draft"])]
    if mtp_meta:
        lines.append(f"\n[bold]MTP-related metadata keys found:[/bold] {len(mtp_meta)}")
        for k in mtp_meta[:5]:
            lines.append(f"  - {k}")

    return "\n".join(str(line) for line in lines)
