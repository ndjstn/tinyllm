"""TinyLLM Command Line Interface."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="tinyllm",
    help="TinyLLM: A Neural Network of LLMs",
    no_args_is_help=True,
)
console = Console()


@app.command()
def version():
    """Show version information."""
    from tinyllm import __version__

    console.print(f"TinyLLM version {__version__}")


@app.command()
def doctor():
    """Check system health and dependencies."""

    async def _check():
        from tinyllm.models import OllamaClient

        console.print("\n[bold]TinyLLM System Check[/bold]\n")

        # Check Ollama
        console.print("Checking Ollama...", end=" ")
        client = OllamaClient()
        try:
            healthy = await client.check_health()
            if healthy:
                console.print("[green]✓ Connected[/green]")
            else:
                console.print("[red]✗ Not responding[/red]")
                return
        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")
            return
        finally:
            await client.close()

        # List models
        console.print("\nAvailable models:")
        client = OllamaClient()
        try:
            models = await client.list_models()
            if models:
                for model in sorted(models):
                    console.print(f"  • {model}")
            else:
                console.print("  [yellow]No models installed[/yellow]")
                console.print("  Run: ollama pull qwen2.5:0.5b")
        finally:
            await client.close()

        console.print("\n[green]System check complete![/green]")

    asyncio.run(_check())


@app.command()
def init(
    config_dir: Path = typer.Option(
        Path("config"),
        "--config",
        "-c",
        help="Configuration directory",
    ),
):
    """Initialize TinyLLM configuration."""
    if config_dir.exists():
        console.print(f"[yellow]Config directory already exists: {config_dir}[/yellow]")
        return

    config_dir.mkdir(parents=True)
    (config_dir / "environments").mkdir()

    # Create default config
    default_config = """# TinyLLM Configuration
version: "1.0"
environment: development

includes:
  - system.yaml
  - models.yaml

graph:
  file: ../graphs/current
"""
    (config_dir / "tinyllm.yaml").write_text(default_config)

    system_config = """# System Configuration
system:
  log_level: INFO
  log_format: json
  data_dir: data
  max_concurrent_requests: 10

ollama:
  host: http://localhost:11434
  timeout_ms: 30000
  max_retries: 3
"""
    (config_dir / "system.yaml").write_text(system_config)

    models_config = """# Model Configuration
models:
  tiers:
    t0:
      name: "Routers"
      models:
        - qwen2.5:0.5b
        - tinyllama
      purpose: "Fast classification"
      vram_estimate_mb: 500

    t1:
      name: "Specialists"
      models:
        - granite-code:3b
        - qwen2.5:3b
        - phi3:mini
      purpose: "Task execution"
      vram_estimate_mb: 3000

    t2:
      name: "Workers"
      models:
        - qwen3:8b
      purpose: "Complex tasks"
      vram_estimate_mb: 6000

    t3:
      name: "Judges"
      models:
        - qwen3:14b
      purpose: "Evaluation"
      vram_estimate_mb: 12000

  default_router: qwen2.5:0.5b
  default_specialist: qwen2.5:3b
  default_judge: qwen3:14b
"""
    (config_dir / "models.yaml").write_text(models_config)

    console.print(f"[green]✓ Created configuration in {config_dir}[/green]")


@app.command()
def run(
    query: str = typer.Argument(..., help="Query to process"),
    trace: bool = typer.Option(False, "--trace", "-t", help="Show execution trace"),
    config_dir: Path = typer.Option(
        Path("config"),
        "--config",
        "-c",
        help="Configuration directory",
    ),
):
    """Run a query through TinyLLM."""
    console.print(f"[dim]Query: {query}[/dim]")
    console.print("[yellow]Note: Full execution not yet implemented[/yellow]")
    console.print("[dim]This is a placeholder - implementation coming soon![/dim]")


@app.command()
def tool(
    tool_name: str = typer.Argument(..., help="Tool to invoke"),
    input_str: str = typer.Argument(..., help="Tool input"),
):
    """Invoke a tool directly."""

    async def _run_tool():
        if tool_name == "calculator":
            from tinyllm.tools.calculator import CalculatorTool, CalculatorInput

            calc = CalculatorTool()
            result = await calc.execute(CalculatorInput(expression=input_str))

            if result.success:
                console.print(f"[green]{result.formatted}[/green]")
            else:
                console.print(f"[red]Error: {result.error}[/red]")
        else:
            console.print(f"[red]Unknown tool: {tool_name}[/red]")
            console.print("Available tools: calculator")

    asyncio.run(_run_tool())


if __name__ == "__main__":
    app()
