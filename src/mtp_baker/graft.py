"""GGUF MTP head grafting logic using the gguf Python library.

This module provides robust merging of MTP draft head tensors into base GGUF files.
It is designed as a reliable workaround for export bugs (e.g. Unsloth truncating
trailing MTP tensors on Qwen models).
"""

from __future__ import annotations

from pathlib import Path

from gguf import GGUFReader, GGUFWriter
from rich.console import Console
from tqdm import tqdm

console = Console()

# Common MTP / draft head tensor name patterns (case-insensitive matching)
MTP_PATTERNS = [
    "mtp",
    "nextn",
    "draft",
    "spec",
    "ssm_conv1d",
    "head.",
    "mtp_head",
]


def _is_mtp_tensor(name: str) -> bool:
    """Check if a tensor name looks like it belongs to MTP draft heads."""
    name_lower = name.lower()
    return any(pattern in name_lower for pattern in MTP_PATTERNS)


def graft_mtp_heads(
    base_path: str | Path,
    heads_path: str | Path,
    output_path: str | Path,
    dry_run: bool = False,
    smart: bool = True,
) -> Path:
    """
    Merge MTP draft head tensors from a separate heads GGUF into a base GGUF.

    This is the recommended workaround when export pipelines (e.g. Unsloth)
    strip or corrupt the MTP layers on Qwen models.

    Args:
        base_path: Path to the base (usually quantized) GGUF
        heads_path: Path to a small GGUF containing only MTP head tensors
        output_path: Where to write the merged result
        dry_run: If True, only analyze and print what would happen
        smart: If True (default), only add tensors that appear to be MTP-related
               and do not already exist in the base (safer).

    Returns:
        Path to the output file (or would-be output in dry-run)
    """
    base_path = Path(base_path).expanduser().resolve()
    heads_path = Path(heads_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if not base_path.exists():
        raise FileNotFoundError(f"Base GGUF not found: {base_path}")
    if not heads_path.exists():
        raise FileNotFoundError(f"MTP heads GGUF not found: {heads_path}")

    console.print(f"[cyan]Reading base GGUF:[/cyan] {base_path.name}")
    base_reader = GGUFReader(str(base_path))

    console.print(f"[cyan]Reading MTP heads GGUF:[/cyan] {heads_path.name}")
    heads_reader = GGUFReader(str(heads_path))

    base_tensors = {t.name: t for t in base_reader.tensors}
    heads_tensors = {t.name: t for t in heads_reader.tensors}

    # Identify MTP tensors in the heads file
    mtp_in_heads = {name: t for name, t in heads_tensors.items() if _is_mtp_tensor(name)}
    non_mtp_in_heads = {name: t for name, t in heads_tensors.items() if not _is_mtp_tensor(name)}

    if dry_run:
        console.print("[yellow]=== DRY RUN — No files will be written ===[/yellow]")

        overlapping = set(base_tensors.keys()) & set(heads_tensors.keys())
        new_mtp = set(mtp_in_heads.keys()) - set(base_tensors.keys())
        new_non_mtp = set(non_mtp_in_heads.keys()) - set(base_tensors.keys())

        console.print(f"\n[bold]Base GGUF:[/bold] {len(base_tensors)} tensors")
        console.print(f"[bold]Heads GGUF:[/bold] {len(heads_tensors)} tensors "
                      f"({len(mtp_in_heads)} look like MTP)")

        if mtp_in_heads:
            console.print(f"\n[green]MTP-related tensors in heads file ({len(mtp_in_heads)}):[/green]")
            for name in sorted(mtp_in_heads.keys())[:15]:
                console.print(f"  • {name}")
            if len(mtp_in_heads) > 15:
                console.print(f"  ... and {len(mtp_in_heads) - 15} more")

        console.print(f"\n[bold]Planned actions (smart={smart}):[/bold]")
        if smart:
            console.print(f"  - Will add {len(new_mtp)} new MTP tensors")
            if new_non_mtp:
                console.print(f"  - [yellow]Skipping {len(new_non_mtp)} non-MTP tensors from heads (smart mode)[/yellow]")
        else:
            console.print(f"  - Will add/overwrite {len(heads_tensors)} tensors from heads file")

        if overlapping:
            console.print(f"  - {len(overlapping)} tensors would be overwritten")

        return output_path

    # === Real grafting ===
    console.print("[cyan]Creating merged GGUF with MTP heads...[/cyan]")

    # Choose which tensors to graft
    if smart:
        tensors_to_graft = mtp_in_heads
        skipped = len(non_mtp_in_heads)
    else:
        tensors_to_graft = heads_tensors
        skipped = 0

    writer = GGUFWriter(str(output_path), arch=base_reader.arch)

    # Copy metadata
    for key, value in base_reader.metadata.items():
        try:
            writer.add_key_value(key, value)
        except Exception:
            pass  # Some metadata types may not transfer perfectly

    # Copy base tensors
    for t in tqdm(base_reader.tensors, desc="Copying base tensors", unit="tensor"):
        writer.add_tensor(t.name, t.data, raw_dtype=t.dtype)

    # Graft selected tensors from heads
    added = 0
    overwritten = 0

    for name, tensor in tqdm(tensors_to_graft.items(), desc="Grafting MTP tensors", unit="tensor"):
        if name in base_tensors:
            overwritten += 1
        else:
            added += 1
        writer.add_tensor(name, tensor.data, raw_dtype=tensor.dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    console.print("\n[bold green]✓ Grafting complete![/bold green]")
    console.print(f"  Added new MTP tensors:     {added}")
    console.print(f"  Overwrote existing:        {overwritten}")
    if skipped:
        console.print(f"  Skipped (non-MTP in smart mode): {skipped}")
    console.print(f"  Total tensors in output:   {len(base_tensors) + added}")

    return output_path
