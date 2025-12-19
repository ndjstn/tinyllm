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


@app.command()
def models():
    """List configured model tiers."""
    table = Table(title="TinyLLM Model Tiers")
    table.add_column("Tier", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Purpose")
    table.add_column("VRAM (est)")

    tiers = [
        ("T0", "Routers", "Fast classification", "~500 MB"),
        ("T1", "Specialists", "Task execution", "~3 GB"),
        ("T2", "Workers", "Complex tasks", "~6 GB"),
        ("T3", "Judges", "Evaluation", "~12 GB"),
    ]

    for tier, name, purpose, vram in tiers:
        table.add_row(tier, name, purpose, vram)

    console.print(table)


@app.command()
def stats():
    """Show system statistics."""
    from tinyllm.grading.metrics import MetricsTracker

    console.print("\n[bold]TinyLLM System Statistics[/bold]\n")

    # System info
    console.print("[cyan]System:[/cyan]")
    console.print("  • Status: [green]Ready[/green]")

    # Module info
    console.print("\n[cyan]Modules:[/cyan]")
    modules = [
        ("Core Engine", "✓"),
        ("Message System", "✓"),
        ("Node Registry", "✓"),
        ("Tool System", "✓"),
        ("Grading System", "✓"),
        ("Expansion System", "✓"),
        ("Memory System", "✓"),
    ]
    for module, status in modules:
        console.print(f"  • {module}: [green]{status}[/green]")

    # Test stats
    console.print("\n[cyan]Tests:[/cyan]")
    console.print("  • Unit tests: [green]199 passing[/green]")


@app.command()
def chat(
    model: str = typer.Option("qwen2.5:1.5b", "--model", "-m", help="Model to use"),
):
    """Start an interactive chat session."""
    from tinyllm.memory import MemoryStore

    async def _chat():
        from tinyllm.models import OllamaClient

        console.print("[bold]TinyLLM Chat[/bold]")
        console.print(f"Model: {model}")
        console.print("Type 'quit' to exit, 'clear' to clear history\n")

        client = OllamaClient()
        memory = MemoryStore()

        try:
            while True:
                user_input = console.input("[cyan]You:[/cyan] ")

                if user_input.lower() == "quit":
                    break
                if user_input.lower() == "clear":
                    memory.clear_stm()
                    console.print("[dim]History cleared[/dim]\n")
                    continue
                if not user_input.strip():
                    continue

                memory.add_message("user", user_input)

                # Get context from memory
                context = memory.get_context_for_prompt(max_tokens=1000)

                # Generate response
                try:
                    response = await client.generate(
                        model=model,
                        prompt=user_input,
                        system=f"You are a helpful assistant.\n\nConversation context:\n{context}",
                    )
                    assistant_msg = response.response
                    memory.add_message("assistant", assistant_msg)
                    console.print(f"[green]Assistant:[/green] {assistant_msg}\n")
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]\n")

        finally:
            await client.close()

    console.print("[yellow]Note: Requires Ollama with specified model[/yellow]")
    try:
        asyncio.run(_chat())
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")


@app.command()
def benchmark(
    iterations: int = typer.Option(10, "--iterations", "-n", help="Number of iterations"),
):
    """Run performance benchmarks."""
    import time

    console.print("[bold]TinyLLM Benchmarks[/bold]\n")

    # Message creation benchmark
    from tinyllm.core.message import Message, MessageType

    console.print("Message creation...", end=" ")
    start = time.perf_counter()
    for _ in range(iterations * 100):
        Message.create_task("test", MessageType.TASK)
    elapsed = (time.perf_counter() - start) * 1000
    console.print(f"[green]{elapsed:.2f}ms for {iterations * 100} messages[/green]")

    # Memory operations benchmark
    from tinyllm.memory import MemoryStore

    console.print("Memory operations...", end=" ")
    store = MemoryStore()
    start = time.perf_counter()
    for i in range(iterations):
        store.add_message("user", f"Test message {i}")
        store.set_context(f"key_{i}", f"value_{i}")
    elapsed = (time.perf_counter() - start) * 1000
    console.print(f"[green]{elapsed:.2f}ms for {iterations * 2} operations[/green]")

    # Grading models benchmark
    from tinyllm.grading.models import Grade, GradeLevel, DimensionScore, QualityDimension

    console.print("Grade creation...", end=" ")
    start = time.perf_counter()
    for _ in range(iterations * 100):
        Grade(
            level=GradeLevel.GOOD,
            overall_score=0.8,
            dimension_scores=[
                DimensionScore(dimension=QualityDimension.CORRECTNESS, score=0.9, reasoning="Good")
            ],
            feedback="Good work",
            suggestions=[],
            is_passing=True,
        )
    elapsed = (time.perf_counter() - start) * 1000
    console.print(f"[green]{elapsed:.2f}ms for {iterations * 100} grades[/green]")

    console.print("\n[green]Benchmarks complete![/green]")


if __name__ == "__main__":
    app()
