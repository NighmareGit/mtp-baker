"""GGUF MTP head grafting logic using the gguf Python library."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from gguf import GGUFReader, GGUFWriter
from rich.console import Console
from tqdm import tqdm

console = Console()


def graft_mtp_heads(
    base_path: str | Path,
    heads_path: str | Path,
    output_path: str | Path,
    dry_run: bool = False,
) -> Path:
    """
    Merge MTP draft head tensors from a separate heads GGUF into a base GGUF.

    This is the recommended workaround for models where the original export
    pipeline (e.g. Unsloth) stripped or corrupted the MTP layers.
    """
    base_path = Path(base_path).expanduser().resolve()
    heads_path = Path(heads_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if not base_path.exists():
        raise FileNotFoundError(f"Base GGUF not found: {base_path}")
    if not heads_path.exists():
        raise FileNotFoundError(f"MTP heads GGUF not found: {heads_path}")

    if output_path.exists() and not dry_run:
        # We already handled confirmation in CLI, but double-check here
        pass

    console.print(f"[cyan]Reading base GGUF:[/cyan] {base_path}")
    base_reader = GGUFReader(str(base_path))

    console.print(f"[cyan]Reading MTP heads GGUF:[/cyan] {heads_path}")
    heads_reader = GGUFReader(str(heads_path))

    if dry_run:
        console.print("[yellow]Dry run mode — analyzing tensors only...[/yellow]")

        base_tensors = {t.name: t for t in base_reader.tensors}
        heads_tensors = {t.name: t for t in heads_reader.tensors}

        overlapping = set(base_tensors.keys()) & set(heads_tensors.keys())
        new_mtp = set(heads_tensors.keys()) - set(base_tensors.keys())

        console.print(f"  Base tensors: {len(base_tensors)}")
        console.print(f"  Heads tensors: {len(heads_tensors)}")
        console.print(f"  Overlapping names (will be overwritten by heads): {len(overlapping)}")
        console.print(f"  New MTP tensors to be added: {len(new_mtp)}")

        if new_mtp:
            console.print("[green]New MTP tensors that will be grafted:[/green]")
            for name in sorted(new_mtp)[:10]:
                console.print(f"  - {name}")
            if len(new_mtp) > 10:
                console.print(f"  ... and {len(new_mtp) - 10} more")

        return output_path

    # Real grafting
    console.print("[cyan]Creating new GGUF with grafted MTP heads...[/cyan]")

    writer = GGUFWriter(str(output_path), arch=base_reader.arch)

    # Copy metadata from base
    for key, value in base_reader.metadata.items():
        writer.add_key_value(key, value)

    # Copy all tensors from base first
    base_tensor_names = [t.name for t in base_reader.tensors]
    for t in tqdm(base_reader.tensors, desc="Copying base tensors"):
        writer.add_tensor(t.name, t.data, raw_dtype=t.dtype)

    # Now add/override with MTP head tensors
    heads_tensor_names = [t.name for t in heads_reader.tensors]
    added_count = 0
    overwritten_count = 0

    for t in tqdm(heads_reader.tensors, desc="Grafting MTP head tensors"):
        if t.name in base_tensor_names:
            overwritten_count += 1
        else:
            added_count += 1
        writer.add_tensor(t.name, t.data, raw_dtype=t.dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    console.print(f"[green]Grafting complete.[/green]")
    console.print(f"  Added new MTP tensors: {added_count}")
    console.print(f"  Overwrote existing tensors: {overwritten_count}")
    console.print(f"  Total tensors in output: {len(base_tensor_names) + added_count}")

    return output_path
