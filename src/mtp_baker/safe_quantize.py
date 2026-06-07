"""Safe re-quantization for MTP GGUF models.

This module allows quantizing a model while keeping MTP draft head tensors
at a higher precision (recommended: Q8_0 or f16) to preserve speculative
decoding quality and acceptance rate.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from gguf import GGUFReader, GGUFWriter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from tqdm import tqdm

console = Console()

MTP_PATTERNS = [
    "mtp", "nextn", "draft", "spec", "ssm_conv1d", "head.", "mtp_head"
]


def _is_mtp_tensor(name: str) -> bool:
    name_lower = name.lower()
    return any(p in name_lower for p in MTP_PATTERNS)


def find_llama_quantize() -> Path | None:
    """Try to find the llama-quantize binary."""
    candidates = [
        "llama-quantize",
        "./llama-quantize",
        "build/bin/llama-quantize",
        "../llama.cpp/build/bin/llama-quantize",
    ]
    for name in candidates:
        path = shutil.which(name) or Path(name)
        if isinstance(path, str):
            path = Path(path)
        if path.exists() and path.is_file():
            return path
    return None


def safe_quantize(
    input_path: str | Path,
    output_path: str | Path,
    quant_type: str = "q4_k_m",
    mtp_precision: str = "q8_0",
    llama_quantize_path: str | Path | None = None,
    keep_temp: bool = False,
) -> Path:
    """
    Quantize an MTP GGUF while protecting MTP tensors at higher precision.

    Workflow:
    1. Quantize the full model using llama-quantize.
    2. Re-insert MTP tensors from the original high-precision file
       at the requested higher precision (Q8_0 recommended).

    Args:
        input_path: High-precision MTP GGUF (strongly recommend f16)
        output_path: Final quantized MTP GGUF
        quant_type: Quantization for main model (q4_k_m, q5_k_m, etc.)
        mtp_precision: Precision for MTP tensors (q8_0 recommended)
        llama_quantize_path: Path to llama-quantize binary (auto-detected if None)
        keep_temp: Keep temporary files for debugging

    Returns:
        Path to the final safe quantized GGUF
    """
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input GGUF not found: {input_path}")

    if llama_quantize_path is None:
        llama_quantize_path = find_llama_quantize()

    if llama_quantize_path is None or not Path(llama_quantize_path).exists():
        raise RuntimeError(
            "Could not find 'llama-quantize' binary.\n"
            "Please build llama.cpp and make sure 'llama-quantize' is in your PATH, "
            "or pass the full path with --llama-quantize-path"
        )

    console.print(f"[bold cyan]Safe MTP Quantization[/bold cyan]")
    console.print(f"Input:            {input_path}")
    console.print(f"Output:           {output_path}")
    console.print(f"Main quant:       {quant_type}")
    console.print(f"MTP precision:    {mtp_precision}")

    # Step 1: Quantize the full model
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_quantized = Path(tmpdir) / "quantized.gguf"

        console.print("\n[yellow]Step 1/2:[/yellow] Running llama-quantize on full model...")

        cmd = [
            str(llama_quantize_path),
            str(input_path),
            str(tmp_quantized),
            quant_type,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            if result.stderr:
                console.print(result.stderr)
        except subprocess.CalledProcessError as e:
            console.print(f"[red]llama-quantize failed:[/red]\n{e.stderr}")
            raise

        console.print("[green]✓ Quantization of main model completed.[/green]")

        # Step 2: Re-insert MTP tensors at higher precision
        console.print(f"\n[yellow]Step 2/2:[/yellow] Re-inserting MTP tensors at {mtp_precision} precision...")

        orig_reader = GGUFReader(str(input_path))
        quant_reader = GGUFReader(str(tmp_quantized))

        writer = GGUFWriter(str(output_path), arch=quant_reader.arch)

        # Copy metadata from quantized version
        for key, value in quant_reader.metadata.items():
            try:
                writer.add_key_value(key, value)
            except Exception:
                pass

        mtp_count = 0
        total_tensors = len(quant_reader.tensors)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing tensors...", total=total_tensors)

            for tensor in quant_reader.tensors:
                name = tensor.name

                if _is_mtp_tensor(name):
                    # Take from original high-precision file and use desired precision
                    orig_tensor = next((t for t in orig_reader.tensors if t.name == name), None)
                    if orig_tensor:
                        writer.add_tensor(name, orig_tensor.data, raw_dtype=mtp_precision)
                        mtp_count += 1
                    else:
                        # Fallback to quantized version
                        writer.add_tensor(name, tensor.data, raw_dtype=tensor.dtype)
                else:
                    writer.add_tensor(name, tensor.data, raw_dtype=tensor.dtype)

                progress.update(task, advance=1)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    console.print(f"\n[bold green]✓ Safe quantization complete![/bold green]")
    console.print(f"  MTP tensors upgraded to {mtp_precision}: {mtp_count}")
    console.print(f"  Final file: {output_path}")

    return output_path
