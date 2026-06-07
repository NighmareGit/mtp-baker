"""Main CLI for mtp-baker using Typer + Rich."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from mtp_baker.graft import graft_mtp_heads
from mtp_baker.verify import verify_mtp_tensors

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
    input: str = typer.Option(..., "--input", "-i", help="Input (usually f16 or high precision) MTP GGUF"),
    output: str = typer.Option(..., "--output", "-o", help="Output quantized GGUF"),
    mtp_precision: str = typer.Option("q8_0", "--mtp-precision", help="Quantization type to use for MTP tensors (e.g. q8_0, f16)"),
    base_quant: str = typer.Option("q4_k_m", "--base-quant", help="Quantization type for the main model tensors"),
):
    """Re-quantize an MTP GGUF while protecting the MTP head tensors at higher precision."""
    console.rule("[bold blue]Safe MTP Quantization[/bold blue]")
    rprint("[yellow]Note: Full safe quantization with tensor protection is coming in v0.2.[/yellow]")
    rprint("For now, you can use the standard llama-quantize tool after grafting, "
           "or let me know if you want this implemented next.")


@app.command()
def info():
    """Show information about mtp-baker and current capabilities."""
    console.print(Panel.fit(
        "[bold]mtp-baker v0.1.0[/bold]\n\n"
        "Created to work around bugs in Unsloth GGUF export that truncate MTP tensors on Qwen models.\n\n"
        "Current best workflow: [green]Graft MTP heads[/green] → [green]Verify[/green] → Run in llama.cpp / turboquant fork",
        title="mtp-baker",
        border_style="blue"
    ))


if __name__ == "__main__":
    app()
