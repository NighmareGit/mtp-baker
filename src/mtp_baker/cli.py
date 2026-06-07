"""Main CLI for mtp-baker using Typer + Rich."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from mtp_baker.graft import graft_mtp_heads
from mtp_baker.verify import verify_mtp_tensors
from mtp_baker.safe_quantize import safe_quantize
from mtp_baker.convert_hf import convert_hf_to_gguf

app = typer.Typer(
    name="mtp-baker",
    help="Reliable toolkit for building and repairing MTP GGUF models (especially Qwen).",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


@app.command()
def graft(
    base: str = typer.Option(..., "--base", "-b", help="Path to base GGUF file"),
    heads: str = typer.Option(..., "--heads", "-h", help="Path to MTP heads GGUF (small file with draft tensors)"),
    output: str = typer.Option(..., "--output", "-o", help="Output path for the merged MTP GGUF"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would happen without writing the file"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite output file if it exists"),
):
    """Graft MTP draft heads into a base GGUF file.

    This is the recommended workaround when Unsloth (or other) export pipelines
    strip MTP tensors from Qwen models.
    """
    console.rule("[bold blue]MTP Head Grafting[/bold blue]")

    table = Table(title="Grafting Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Base GGUF", base)
    table.add_row("MTP Heads GGUF", heads)
    table.add_row("Output", output)
    table.add_row("Dry Run", str(dry_run))
    console.print(table)

    if not force and not dry_run:
        if typer.confirm(f"Output file '{output}' already exists. Overwrite?", default=False):
            pass
        else:
            rprint("[red]Aborted.[/red]")
            raise typer.Exit(1)

    try:
        result = graft_mtp_heads(
            base_path=base,
            heads_path=heads,
            output_path=output,
            dry_run=dry_run,
        )
        if dry_run:
            console.print(Panel.fit("[yellow]Dry run completed. No file was written.[/yellow]"))
        else:
            console.print(Panel.fit(f"[bold green]Success![/bold green] MTP GGUF written to: {result}"))
    except Exception as e:
        console.print(Panel.fit(f"[bold red]Error during grafting:[/bold red]\n{str(e)}", border_style="red"))
        raise typer.Exit(1)


@app.command()
def verify(
    gguf_path: str = typer.Argument(..., help="Path to the GGUF file to verify"),
    show_tensors: bool = typer.Option(False, "--show-tensors", help="List all MTP-related tensors found"),
):
    """Verify that MTP draft head tensors are present and look intact in a GGUF."""
    console.rule("[bold blue]MTP Verification[/bold blue]")

    try:
        report = verify_mtp_tensors(gguf_path, show_tensors=show_tensors)
        console.print(report)
    except Exception as e:
        console.print(Panel.fit(f"[bold red]Verification failed:[/bold red]\n{str(e)}", border_style="red"))
        raise typer.Exit(1)


@app.command()
def quantize(
    input: str = typer.Option(..., "--input", "-i", help="High-precision MTP GGUF (recommended: f16 after grafting)"),
    output: str = typer.Option(..., "--output", "-o", help="Output safe quantized GGUF"),
    base_quant: str = typer.Option("q4_k_m", "--base-quant", help="Quantization type for the main model tensors"),
    mtp_precision: str = typer.Option("q8_0", "--mtp-precision", help="Precision to keep MTP tensors at (q8_0 recommended)"),
    llama_quantize_path: str | None = typer.Option(None, "--llama-quantize-path", help="Path to llama-quantize binary (auto-detected if not provided)"),
):
    """Safely re-quantize an MTP GGUF while protecting MTP head tensors at higher precision.

    This is the recommended way to create production-ready quantized MTP models.
    """
    console.rule("[bold blue]Safe MTP Quantization[/bold blue]")

    try:
        result = safe_quantize(
            input_path=input,
            output_path=output,
            quant_type=base_quant,
            mtp_precision=mtp_precision,
            llama_quantize_path=llama_quantize_path,
        )
        console.print(Panel.fit(f"[bold green]Success![/bold green] Safe quantized MTP model saved to:\n{result}"))
    except Exception as e:
        console.print(Panel.fit(f"[bold red]Quantization failed:[/bold red]\n{str(e)}", border_style="red"))
        raise typer.Exit(1)


@app.command("convert-hf")
def convert_hf(
    model: str = typer.Option(..., "--model", "-m", help="Hugging Face model ID or local path (e.g. Qwen/Qwen3.5-35B-Base)"),
    output: str = typer.Option(..., "--output", "-o", help="Output GGUF file path"),
    outtype: str = typer.Option("f16", "--outtype", help="Output tensor type (f16 recommended for MTP)"),
    convert_script: str | None = typer.Option(None, "--convert-script", help="Path to convert_hf_to_gguf.py"),
):
    """Convert a Hugging Face model to GGUF with MTP awareness.

    Especially useful for Qwen models that may contain MTP heads.
    After conversion it automatically verifies MTP tensors and gives next-step recommendations.
    """
    console.rule("[bold blue]HF → GGUF Conversion[/bold blue]")

    try:
        result = convert_hf_to_gguf(
            model_id_or_path=model,
            output_path=output,
            outtype=outtype,
            convert_script=convert_script,
        )
        console.print(Panel.fit(f"[bold green]Conversion successful![/bold green]\nOutput: {result}"))
    except Exception as e:
        console.print(Panel.fit(f"[bold red]Conversion failed:[/bold red]\n{str(e)}", border_style="red"))
        raise typer.Exit(1)


@app.command()
def info():
    """Show information about mtp-baker and current capabilities."""
    console.print(Panel.fit(
        "[bold]mtp-baker v0.4.0[/bold]\n\n"
        "Created to work around bugs in Unsloth GGUF export that truncate MTP tensors on Qwen models.\n\n"
        "[bold]Current best workflow:[/bold]\n"
        "1. [green]convert-hf[/green] (from raw HF model)  OR  start with a base GGUF\n"
        "2. [green]graft[/green] MTP heads if needed\n"
        "3. [green]verify[/green] the result\n"
        "4. [green]quantize[/green] safely (MTP tensors protected)\n"
        "5. Run in llama.cpp / turboquant / ik_llama.cpp forks",
        title="mtp-baker",
        border_style="blue"
    ))


if __name__ == "__main__":
    app()
