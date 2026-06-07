"""HF to GGUF conversion wrapper with MTP awareness.

This module provides a convenient wrapper around llama.cpp's convert_hf_to_gguf.py
with special handling and recommendations for Qwen models that include MTP heads.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from mtp_baker.verify import verify_mtp_tensors

console = Console()


def find_convert_script() -> Path | None:
    """Try to locate convert_hf_to_gguf.py from llama.cpp."""
    candidates = [
        "convert_hf_to_gguf.py",
        "llama.cpp/convert_hf_to_gguf.py",
        "../llama.cpp/convert_hf_to_gguf.py",
        "examples/convert_hf_to_gguf.py",
    ]
    for name in candidates:
        path = shutil.which(name) or Path(name)
        if isinstance(path, (str, Path)) and Path(path).exists():
            return Path(path)
    return None


def convert_hf_to_gguf(
    model_id_or_path: str,
    output_path: str | Path,
    outtype: str = "f16",
    convert_script: str | Path | None = None,
    extra_args: list[str] | None = None,
) -> Path:
    """
    Convert a Hugging Face model to GGUF with MTP awareness.

    For Qwen models with MTP heads, it is recommended to use a recent
    version of llama.cpp that has proper MTP support in the converter.

    After conversion, it automatically runs verification and gives
    recommendations (e.g. run `graft` if MTP tensors are missing).

    Args:
        model_id_or_path: Hugging Face model ID or local path
        output_path: Where to save the GGUF file
        outtype: Output type (f16 recommended for MTP)
        convert_script: Path to convert_hf_to_gguf.py (auto-detected if None)
        extra_args: Additional arguments to pass to the converter

    Returns:
        Path to the created GGUF file
    """
    output_path = Path(output_path).expanduser().resolve()

    if convert_script is None:
        convert_script = find_convert_script()

    if convert_script is None or not Path(convert_script).exists():
        raise RuntimeError(
            "Could not find 'convert_hf_to_gguf.py'.\n"
            "Please clone llama.cpp and provide the path with --convert-script, "
            "or make sure it's in your PATH / common locations."
        )

    console.print(f"[bold cyan]HF → GGUF Conversion (MTP aware)[/bold cyan]")
    console.print(f"Model:     {model_id_or_path}")
    console.print(f"Output:    {output_path}")
    console.print(f"Outtype:   {outtype}")

    cmd = [
        "python3",
        str(convert_script),
        str(model_id_or_path),
        "--outfile", str(output_path),
        "--outtype", outtype,
    ]

    if extra_args:
        cmd.extend(extra_args)

    console.print("\n[yellow]Running convert_hf_to_gguf.py...[/yellow] (this can take a while)")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        if result.stdout:
            console.print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Conversion failed.[/red]")
        console.print(e.stderr[-3000:] if e.stderr else str(e))
        raise

    console.print(f"\n[green]✓ Conversion completed.[/green]")

    # Automatic MTP verification
    console.print("\n[cyan]Running automatic MTP verification...[/cyan]")
    try:
        report = verify_mtp_tensors(str(output_path))
        console.print(report)
    except Exception as e:
        console.print(f"[yellow]Verification failed to run:[/yellow] {e}")

    # Recommendations
    console.print(Panel.fit(
        "[bold]Next recommended steps:[/bold]\n\n"
        "• If MTP tensors are present → run [green]mtp-baker quantize[/green] with --mtp-precision q8_0\n"
        "• If MTP tensors are missing → run [green]mtp-baker graft[/green] using a heads file\n"
        "• For best results with Qwen MTP models, use a recent llama.cpp build (b9180+)",
        title="MTP Workflow Recommendation",
        border_style="blue"
    ))

    return output_path
