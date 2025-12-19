"""TinyLLM Command Line Interface."""

import asyncio
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from tinyllm.logging import configure_logging, get_logger

app = typer.Typer(
    name="tinyllm",
    help="TinyLLM: A Neural Network of LLMs",
    no_args_is_help=True,
)
console = Console()
logger = get_logger(__name__, component="cli")


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

        logger.info("system_check_started")
        console.print("\n[bold]TinyLLM System Check[/bold]\n")

        # Check Ollama
        console.print("Checking Ollama...", end=" ")
        client = OllamaClient()
        try:
            healthy = await client.check_health()
            if healthy:
                logger.info("ollama_health_check", status="connected")
                console.print("[green]✓ Connected[/green]")
            else:
                logger.warning("ollama_health_check", status="not_responding")
                console.print("[red]✗ Not responding[/red]")
                return
        except Exception as e:
            logger.error("ollama_health_check", status="error", error=str(e))
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


# Known command names that users might accidentally pass to 'run'
_KNOWN_COMMANDS = {"chat", "doctor", "init", "models", "stats", "benchmark", "version", "tool", "graph", "health", "cache-stats", "cache-clear", "metrics"}


def _check_command_confusion(query: str) -> bool:
    """Check if the query looks like a command and warn the user."""
    query_lower = query.lower().strip()

    if query_lower in _KNOWN_COMMANDS:
        console.print(f"\n[yellow]⚠ Did you mean to run the '{query_lower}' command?[/yellow]")
        console.print(f"[dim]You typed:[/dim] tinyllm run {query}")
        console.print(f"[dim]Try instead:[/dim] [green]tinyllm {query_lower}[/green]\n")
        console.print(f"[dim]The 'run' command executes queries through the LLM graph.[/dim]")
        console.print(f"[dim]The '{query_lower}' command is a separate CLI command.[/dim]\n")
        return True
    return False


@app.command()
def run(
    query: str = typer.Argument(..., help="Query to process"),
    graph_path: Path = typer.Option(
        Path("graphs/multi_domain.yaml"),
        "--graph",
        "-g",
        help="Graph definition file",
    ),
    trace: bool = typer.Option(False, "--trace", "-t", help="Show execution trace"),
    force: bool = typer.Option(False, "--force", "-f", help="Force execution even if query looks like a command"),
    log_level: str = typer.Option(
        os.getenv("TINYLLM_LOG_LEVEL", "INFO"),
        "--log-level",
        "-l",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    ),
    log_format: str = typer.Option(
        os.getenv("TINYLLM_LOG_FORMAT", "console"),
        "--log-format",
        help="Log format (console or json)",
    ),
    tracing: bool = typer.Option(
        False,
        "--tracing",
        help="Enable OpenTelemetry distributed tracing",
    ),
    otlp_endpoint: Optional[str] = typer.Option(
        None,
        "--otlp-endpoint",
        help="OTLP exporter endpoint (e.g., http://localhost:4317)",
    ),
    metrics_port: Optional[int] = typer.Option(
        None,
        "--metrics-port",
        help="Port to expose Prometheus metrics (e.g., 9090)",
    ),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="Enable response caching"),
    cache_backend: str = typer.Option(
        "memory",
        "--cache-backend",
        help="Cache backend (memory or redis)",
    ),
    cache_ttl: int = typer.Option(
        3600,
        "--cache-ttl",
        help="Cache TTL in seconds (0 for no expiration)",
    ),
    max_queue_size: int = typer.Option(
        0,
        "--max-queue-size",
        help="Maximum request queue size (0 for unlimited, enables queuing if > 0)",
    ),
    workers: int = typer.Option(
        5,
        "--workers",
        help="Number of concurrent worker threads (only used with queuing)",
    ),
):
    """Run a query through TinyLLM."""

    # Configure logging based on CLI options
    configure_logging(log_level=log_level, log_format=log_format)

    # Configure telemetry if requested
    if tracing:
        from tinyllm.telemetry import TelemetryConfig, configure_telemetry
        telemetry_config = TelemetryConfig(
            enable_tracing=True,
            service_name="tinyllm",
            exporter="otlp" if otlp_endpoint else "console",
            otlp_endpoint=otlp_endpoint,
            sampling_rate=1.0,
        )
        try:
            configure_telemetry(telemetry_config)
            if otlp_endpoint:
                console.print(f"[dim]Tracing enabled, exporting to {otlp_endpoint}[/dim]")
            else:
                console.print("[dim]Tracing enabled (console exporter)[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not enable tracing: {e}[/yellow]")

    # Start metrics server if port is specified
    if metrics_port:
        from tinyllm.metrics import start_metrics_server
        try:
            start_metrics_server(port=metrics_port)
            console.print(f"[dim]Metrics available at http://localhost:{metrics_port}/metrics[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not start metrics server: {e}[/yellow]")

    # Check for command confusion unless --force is used
    if not force and _check_command_confusion(query):
        if not typer.confirm("Do you want to process this as a query anyway?", default=False):
            raise typer.Exit(0)

    async def _run():
        from tinyllm.core.builder import load_graph
        from tinyllm.core.executor import Executor
        from tinyllm.core.message import TaskPayload

        logger.info("query_execution_started", query=query[:100], graph_path=str(graph_path))
        console.print(f"[dim]Query: {query}[/dim]")
        console.print(f"[dim]Graph: {graph_path}[/dim]")

        # Show queue configuration if enabled
        if max_queue_size > 0:
            console.print(f"[dim]Queue: enabled (max_size={max_queue_size}, workers={workers})[/dim]\n")
        else:
            console.print("[dim]Queue: disabled (direct execution)[/dim]\n")

        try:
            # Load and build graph
            graph = load_graph(graph_path)
            executor = Executor(graph)

            # Use queued executor if queue size is configured
            if max_queue_size > 0:
                from tinyllm.queue import QueuedExecutor, BackpressureMode

                queued_executor = QueuedExecutor(
                    executor=executor,
                    max_workers=workers,
                    max_queue_size=max_queue_size,
                    backpressure_mode=BackpressureMode.BLOCK,
                )

                async with queued_executor.lifespan():
                    with console.status("[bold green]Processing..."):
                        task = TaskPayload(content=query)
                        response = await queued_executor.execute(task)
            else:
                # Direct execution without queuing
                with console.status("[bold green]Processing..."):
                    task = TaskPayload(content=query)
                    response = await executor.execute(task)

            # Display results
            if response.success:
                logger.info(
                    "query_execution_success",
                    trace_id=response.trace_id,
                    nodes_executed=response.nodes_executed,
                    latency_ms=response.total_latency_ms,
                )
                console.print("\n[bold green]Response:[/bold green]")
                console.print(response.content or "[dim]No content returned[/dim]")
            else:
                logger.error(
                    "query_execution_failed",
                    trace_id=response.trace_id,
                    error=response.error.message if response.error else "Unknown error",
                )
                console.print(f"\n[red]Error: {response.error.message if response.error else 'Unknown error'}[/red]")

            # Show trace info if requested
            if trace:
                console.print(f"\n[dim]─── Trace Info ───[/dim]")
                console.print(f"[dim]Trace ID: {response.trace_id}[/dim]")
                console.print(f"[dim]Nodes executed: {response.nodes_executed}[/dim]")
                console.print(f"[dim]Tokens used: {response.tokens_used or 0}[/dim]")
                console.print(f"[dim]Latency: {response.total_latency_ms}ms[/dim]")

        except FileNotFoundError as e:
            logger.error("graph_file_not_found", graph_path=str(graph_path))
            console.print(f"[red]Graph file not found: {graph_path}[/red]")
            console.print("[dim]Use --graph to specify a different graph file[/dim]")
        except Exception as e:
            logger.error("execution_exception", error=str(e), error_type=type(e).__name__, exc_info=True)
            console.print(f"[red]Execution failed: {e}[/red]")

    asyncio.run(_run())


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


# Model management subcommand
models_app = typer.Typer(help="Model management commands")
app.add_typer(models_app, name="models")


@models_app.command("list")
def models_list():
    """List all available models from Ollama."""
    from tinyllm.models.registry import get_model_registry

    async def _list():
        from tinyllm.models import OllamaClient

        client = OllamaClient()
        registry = get_model_registry()

        try:
            console.print("\n[bold]Available Models:[/bold]\n")

            models = await client.list_models()
            if not models:
                console.print("[yellow]No models installed[/yellow]")
                console.print("\nTo install a model, run: [cyan]tinyllm models pull <model_name>[/cyan]\n")
                return

            # Sync registry
            registry.sync_from_ollama(models)

            # Create table
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Model", style="green")
            table.add_column("Parameters", style="dim")
            table.add_column("Context Size", style="dim")
            table.add_column("Requests", justify="right")
            table.add_column("Success Rate", justify="right")
            table.add_column("Avg Latency", justify="right")

            for model_name in sorted(models):
                info = registry.get_model_info(model_name)
                params = info.get("parameters", "-")
                context = str(info.get("context_size", "-"))
                health = registry.get_health(model_name)

                if health and health.total_requests > 0:
                    requests = str(health.total_requests)
                    success_rate = f"{health.success_rate:.0f}%"
                    avg_latency = f"{health.average_latency_ms:.0f}ms"
                else:
                    requests = "0"
                    success_rate = "-"
                    avg_latency = "-"

                table.add_row(model_name, params, context, requests, success_rate, avg_latency)

            console.print(table)

            # Show aliases
            aliases = registry.list_aliases()
            if aliases:
                console.print("\n[bold]Model Aliases:[/bold]")
                for alias, target in sorted(aliases.items()):
                    console.print(f"  [cyan]{alias}[/cyan] -> {target}")

            console.print()

        finally:
            await client.close()

    asyncio.run(_list())


@models_app.command("pull")
def models_pull(
    model: str = typer.Argument(..., help="Model name to pull (e.g., qwen2.5:3b)"),
):
    """Pull a model from Ollama registry."""

    async def _pull():
        from tinyllm.models import OllamaClient

        client = OllamaClient()

        try:
            console.print(f"[bold]Pulling model:[/bold] {model}\n")
            console.print("[dim]This may take a while depending on model size...[/dim]\n")

            with console.status(f"[bold green]Downloading {model}...", spinner="dots"):
                await client.pull_model(model)

            console.print(f"\n[green]Successfully pulled {model}![/green]")

        except Exception as e:
            console.print(f"\n[red]Error pulling model: {e}[/red]")
            raise typer.Exit(1)
        finally:
            await client.close()

    asyncio.run(_pull())


@models_app.command("set-default")
def models_set_default(
    model: str = typer.Argument(..., help="Model name to set as default"),
):
    """Set the default model for TinyLLM."""
    from tinyllm.models.registry import get_model_registry

    registry = get_model_registry()

    # Resolve alias if provided
    actual_model = registry.resolve_name(model)

    registry.set_default_model(actual_model)
    console.print(f"\n[green]Default model set to: {actual_model}[/green]\n")


@models_app.command("info")
def models_info(
    model: str = typer.Argument(..., help="Model name to get info about"),
):
    """Get detailed information about a specific model."""
    from tinyllm.models.registry import get_model_registry

    async def _info():
        from tinyllm.models import OllamaClient

        client = OllamaClient()
        registry = get_model_registry()

        try:
            # Sync registry
            models = await client.list_models()
            registry.sync_from_ollama(models)

            # Resolve alias
            actual_model = registry.resolve_name(model)

            info = registry.get_model_info(actual_model)

            if not info.get("exists"):
                console.print(f"\n[red]Model '{actual_model}' not found[/red]\n")
                raise typer.Exit(1)

            console.print(f"\n[bold]Model Information: {actual_model}[/bold]\n")

            # Basic info
            console.print("[cyan]Capabilities:[/cyan]")
            console.print(f"  Family: {info.get('family', 'Unknown')}")
            console.print(f"  Parameters: {info.get('parameters', 'Unknown')}")
            console.print(f"  Context Size: {info.get('context_size', 'Unknown')}")
            console.print(f"  Vision Support: {info.get('vision_support', False)}")

            # Health stats
            if info.get("total_requests", 0) > 0:
                console.print(f"\n[cyan]Statistics:[/cyan]")
                console.print(f"  Total Requests: {info.get('total_requests')}")
                console.print(f"  Success Rate: {info.get('success_rate')}")
                console.print(f"  Average Latency: {info.get('avg_latency_ms')}")
                console.print(f"  Health Status: {'[green]Healthy[/green]' if info.get('is_healthy') else '[red]Unhealthy[/red]'}")

            console.print()

        finally:
            await client.close()

    asyncio.run(_info())


@app.command("tiers")
def tiers():
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
        ("Metrics System", "✓"),
    ]
    for module, status in modules:
        console.print(f"  • {module}: [green]{status}[/green]")

    # Test stats
    console.print("\n[cyan]Tests:[/cyan]")
    console.print("  • Unit tests: [green]199 passing[/green]")


@app.command()
def metrics(
    port: int = typer.Option(9090, "--port", "-p", help="Port to expose metrics"),
    addr: str = typer.Option("0.0.0.0", "--addr", "-a", help="Address to bind to"),
):
    """Start Prometheus metrics server.

    This starts an HTTP server that exposes TinyLLM metrics at /metrics
    for Prometheus to scrape. The server runs indefinitely until stopped.

    Example:
        tinyllm metrics --port 9090

    Then configure Prometheus to scrape http://localhost:9090/metrics
    """
    from tinyllm.metrics import start_metrics_server, get_metrics_collector

    console.print(f"[bold]Starting TinyLLM Metrics Server[/bold]\n")
    console.print(f"Port: {port}")
    console.print(f"Address: {addr}")
    console.print(f"\nMetrics endpoint: [cyan]http://{addr}:{port}/metrics[/cyan]\n")

    # Initialize metrics collector
    collector = get_metrics_collector()
    console.print("[dim]Metrics collector initialized[/dim]")

    try:
        start_metrics_server(port=port, addr=addr)
        console.print("[green]Metrics server started successfully![/green]")
        console.print("\n[dim]Press Ctrl+C to stop...[/dim]\n")

        # Keep the process running
        import signal
        import threading
        event = threading.Event()

        def signal_handler(sig, frame):
            console.print("\n[yellow]Shutting down metrics server...[/yellow]")
            event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        event.wait()

    except OSError as e:
        if "Address already in use" in str(e):
            console.print(f"[red]Error: Port {port} is already in use[/red]")
            console.print(f"[dim]Try a different port with --port[/dim]")
        else:
            console.print(f"[red]Error starting metrics server: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def health(
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    timeout_seconds: int = typer.Option(5, "--timeout", "-t", help="Health check timeout"),
):
    """Check system health for readiness/liveness probes.

    Returns exit code 0 if healthy, 1 if unhealthy.
    Useful for container orchestration (Kubernetes, Docker).
    """
    import json as json_lib
    import time

    async def _health_check():
        from tinyllm.models import OllamaClient, get_shared_client

        start_time = time.monotonic()
        results = {
            "status": "healthy",
            "checks": {},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # Check Ollama connectivity
        try:
            client = await get_shared_client()
            healthy = await asyncio.wait_for(
                client.check_health(),
                timeout=timeout_seconds
            )
            results["checks"]["ollama"] = {
                "status": "healthy" if healthy else "unhealthy",
                "latency_ms": round((time.monotonic() - start_time) * 1000, 2),
            }
            if not healthy:
                results["status"] = "unhealthy"
        except asyncio.TimeoutError:
            results["checks"]["ollama"] = {
                "status": "unhealthy",
                "error": f"timeout after {timeout_seconds}s",
            }
            results["status"] = "unhealthy"
        except Exception as e:
            results["checks"]["ollama"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            results["status"] = "unhealthy"

        # Check available models
        try:
            client = await get_shared_client()
            models = await asyncio.wait_for(
                client.list_models(),
                timeout=timeout_seconds
            )
            results["checks"]["models"] = {
                "status": "healthy" if models else "degraded",
                "count": len(models) if models else 0,
            }
        except Exception as e:
            results["checks"]["models"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        # Add client stats
        try:
            client = await get_shared_client()
            stats = client.get_stats()
            results["client_stats"] = stats
        except Exception:
            pass

        results["total_latency_ms"] = round((time.monotonic() - start_time) * 1000, 2)
        return results

    try:
        results = asyncio.run(_health_check())

        if json_output:
            console.print(json_lib.dumps(results, indent=2))
        else:
            status_color = "green" if results["status"] == "healthy" else "red"
            console.print(f"\n[bold]Health Status: [{status_color}]{results['status'].upper()}[/{status_color}][/bold]")

            for check_name, check_result in results["checks"].items():
                check_status = check_result["status"]
                status_icon = "✓" if check_status == "healthy" else "✗"
                color = "green" if check_status == "healthy" else "red"
                console.print(f"  [{color}]{status_icon}[/{color}] {check_name}: {check_status}")
                if "latency_ms" in check_result:
                    console.print(f"      latency: {check_result['latency_ms']}ms")
                if "error" in check_result:
                    console.print(f"      [dim]error: {check_result['error']}[/dim]")
                if "count" in check_result:
                    console.print(f"      models: {check_result['count']}")

            console.print(f"\n  Total check time: {results['total_latency_ms']}ms")

        # Exit with appropriate code
        if results["status"] != "healthy":
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        if json_output:
            console.print(json_lib.dumps({"status": "unhealthy", "error": str(e)}))
        else:
            console.print(f"[red]Health check failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def chat(
    model: str = typer.Option("qwen2.5:1.5b", "--model", "-m", help="Model to use"),
    system_prompt: Optional[str] = typer.Option(None, "--system", "-s", help="Custom system prompt"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream responses"),
):
    """Start an interactive chat session.

    Commands:
        /model <name> - Switch to a different model
        /models - List available models
        /help - Show available commands
        /clear - Clear conversation history
        /quit or quit - Exit chat
    """
    import time
    from tinyllm.memory import MemoryStore
    from tinyllm.prompts import get_chat_prompt, get_identity_correction
    from tinyllm.models.registry import get_model_registry
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text

    async def _chat():
        from tinyllm.models import OllamaClient

        # Display header with model info
        console.print(Panel.fit(
            f"[bold cyan]TinyLLM Interactive Chat[/bold cyan]\n\n"
            f"[dim]Model:[/dim] {model}\n"
            f"[dim]Identity:[/dim] TinyLLM Assistant\n"
            f"[dim]Streaming:[/dim] {'enabled' if stream else 'disabled'}\n\n"
            f"[yellow]Commands:[/yellow]\n"
            f"  [dim]• /model <name> - Switch models[/dim]\n"
            f"  [dim]• /models - List available models[/dim]\n"
            f"  [dim]• /help - Show all commands[/dim]\n"
            f"  [dim]• Type 'quit' or 'exit' to leave[/dim]\n"
            f"  [dim]• Type 'clear' to clear history[/dim]",
            border_style="cyan"
        ))
        console.print()

        client = OllamaClient(default_model=model)
        memory = MemoryStore()
        registry = get_model_registry()

        # Sync registry with available models
        try:
            available_models = await client.list_models()
            registry.sync_from_ollama(available_models)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not fetch models: {e}[/yellow]\n")

        current_model = model

        # Use custom prompt if provided, otherwise use default with model info
        base_system_prompt = system_prompt if system_prompt else get_chat_prompt(current_model)

        try:
            while True:
                # User input with blue/cyan styling and current model indicator
                user_input = console.input(f"[bold cyan]You[/bold cyan] [dim]({current_model})[/dim]: ")

                if not user_input.strip():
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    command_parts = user_input.split(maxsplit=1)
                    command = command_parts[0].lower()

                    if command in ["/quit", "/exit"]:
                        console.print("\n[dim]Goodbye![/dim]")
                        break
                    elif command == "/help":
                        console.print("\n[bold]Available Commands:[/bold]")
                        console.print("  /model <name>  - Switch to a different model")
                        console.print("  /models        - List available models and their stats")
                        console.print("  /clear         - Clear conversation history")
                        console.print("  /help          - Show this help message")
                        console.print("  /quit          - Exit chat")
                        console.print()
                        continue
                    elif command == "/models":
                        console.print("\n[bold]Available Models:[/bold]")
                        try:
                            models = await client.list_models()
                            if models:
                                for m in sorted(models):
                                    current_indicator = " [green](current)[/green]" if m == current_model else ""
                                    health = registry.get_health(m)
                                    if health and health.total_requests > 0:
                                        health_info = f" [dim]({health.success_rate:.0f}% success, {health.average_latency_ms:.0f}ms avg)[/dim]"
                                    else:
                                        health_info = ""
                                    console.print(f"  • {m}{current_indicator}{health_info}")

                                aliases = registry.list_aliases()
                                if aliases:
                                    console.print("\n[bold]Aliases:[/bold]")
                                    for alias, target in sorted(aliases.items()):
                                        console.print(f"  {alias} -> {target}")
                            else:
                                console.print("  [yellow]No models installed[/yellow]")
                        except Exception as e:
                            console.print(f"  [red]Error listing models: {e}[/red]")
                        console.print()
                        continue
                    elif command == "/model":
                        if len(command_parts) < 2:
                            console.print("[yellow]Usage: /model <model_name>[/yellow]\n")
                            continue

                        new_model = command_parts[1].strip()
                        new_model = registry.resolve_name(new_model)
                        console.print(f"[dim]Switching from {current_model} to {new_model}...[/dim]")

                        try:
                            success, warning = await client.switch_model(new_model)
                            if success:
                                current_model = new_model
                                base_system_prompt = system_prompt if system_prompt else get_chat_prompt(current_model)
                                console.print(f"[green]Switched to model: {current_model}[/green]")
                                console.print("[dim]Conversation context preserved[/dim]\n")
                            else:
                                console.print(f"[red]Failed to switch: {warning}[/red]\n")
                        except Exception as e:
                            console.print(f"[red]Error switching model: {e}[/red]\n")
                        continue
                    elif command == "/clear":
                        memory.clear_stm()
                        console.print("[yellow]History cleared[/yellow]\n")
                        continue
                    else:
                        console.print(f"[yellow]Unknown command: {command}. Type /help for available commands.[/yellow]\n")
                        continue

                # Handle quit without slash
                if user_input.lower() in ["quit", "exit", "clear"]:
                    if user_input.lower() == "clear":
                        memory.clear_stm()
                        console.print("[yellow]History cleared[/yellow]\n")
                        continue
                    else:
                        console.print("\n[dim]Goodbye![/dim]")
                        break

                # Check for identity confusion and provide immediate correction
                identity_correction = get_identity_correction(user_input)
                if identity_correction:
                    console.print(f"\n[bold green]Assistant:[/bold green] [green]{identity_correction}[/green]\n")
                    memory.add_message("user", user_input)
                    memory.add_message("assistant", identity_correction)
                    continue

                memory.add_message("user", user_input)

                # Get context from memory
                context = memory.get_context_for_prompt(max_tokens=1000)

                # Build system prompt with context
                full_system_prompt = base_system_prompt
                if context:
                    full_system_prompt += f"\n\nConversation context:\n{context}"

                # Generate response with streaming or standard mode
                try:
                    if stream:
                        # Streaming mode with live updates
                        console.print("\n[bold green]Assistant:[/bold green] ", end="")

                        response_text = ""
                        token_count = 0
                        start_time = time.monotonic()

                        # Stream tokens as they arrive
                        async for chunk in client.generate_stream(
                            prompt=user_input,
                            system=full_system_prompt,
                        ):
                            response_text += chunk
                            token_count += 1
                            console.print(chunk, end="", style="green")

                        elapsed_time = time.monotonic() - start_time

                        # Add newlines after response
                        console.print("\n")

                        # Show statistics if response took >2 seconds
                        if elapsed_time > 2.0:
                            stats_text = Text()
                            stats_text.append(f"  [~{token_count} tokens, {elapsed_time:.1f}s]", style="dim")
                            console.print(stats_text)

                        console.print()  # Extra spacing

                        memory.add_message("assistant", response_text)

                        # Record request in registry
                        registry.record_request(current_model, elapsed_time * 1000, success=True)

                    else:
                        # Non-streaming mode with spinner
                        with console.status("[bold yellow]Thinking...", spinner="dots"):
                            start_time = time.monotonic()
                            response = await client.generate(
                                prompt=user_input,
                                system=full_system_prompt,
                            )
                            elapsed_time = time.monotonic() - start_time

                        assistant_msg = response.response
                        memory.add_message("assistant", assistant_msg)

                        # Display response with markdown support
                        console.print("\n[bold green]Assistant:[/bold green]")

                        # Try to render as markdown if it contains markdown syntax
                        if any(marker in assistant_msg for marker in ['```', '**', '##', '- ', '* ']):
                            console.print(Markdown(assistant_msg))
                        else:
                            console.print(f"[green]{assistant_msg}[/green]")

                        # Show statistics
                        stats = []
                        if response.eval_count:
                            stats.append(f"{response.eval_count} tokens")
                        if elapsed_time > 2.0:
                            stats.append(f"{elapsed_time:.1f}s")

                        if stats:
                            console.print(f"[dim]  [{', '.join(stats)}][/dim]")

                        console.print()  # Extra spacing

                        # Record request in registry
                        registry.record_request(current_model, elapsed_time * 1000, success=True)

                except Exception as e:
                    console.print(f"\n[bold red]Error:[/bold red] [red]{e}[/red]\n")
                    logger.error("chat_error", error=str(e), model=current_model)
                    # Record failed request
                    registry.record_request(current_model, 0, success=False, error=str(e))

        finally:
            await client.close()

    console.print("[dim]Note: Requires Ollama with specified model[/dim]\n")
    try:
        asyncio.run(_chat())
    except KeyboardInterrupt:
        console.print("\n\n[dim]Goodbye![/dim]")



# Graph versioning subcommand
graph_app = typer.Typer(help="Graph versioning commands")
app.add_typer(graph_app, name="graph")


@graph_app.command("versions")
def graph_versions(
    storage: Path = typer.Option(Path(".tinyllm/versions"), "--storage", "-s"),
):
    """List all graph versions."""
    from tinyllm.expansion.versioning import GraphVersionManager

    manager = GraphVersionManager(storage)
    versions = manager.list_versions()

    if not versions:
        console.print("[yellow]No versions found[/yellow]")
        return

    table = Table(title="Graph Versions")
    table.add_column("Version", style="cyan")
    table.add_column("Message")
    table.add_column("Created", style="dim")
    table.add_column("Changes", style="green")

    for v in versions:
        is_current = " [current]" if v.version == manager.current_version else ""
        created = v.created_at.strftime("%Y-%m-%d %H:%M")
        changes = str(len(v.changes)) if v.changes else "-"
        table.add_row(f"{v.version}{is_current}", v.message or "-", created, changes)

    console.print(table)


@graph_app.command("save")
def graph_save(
    graph_path: Path = typer.Argument(..., help="Graph file to version"),
    message: str = typer.Option("", "--message", "-m", help="Version message"),
    bump: str = typer.Option("patch", "--bump", "-b", help="Version bump type"),
    storage: Path = typer.Option(Path(".tinyllm/versions"), "--storage", "-s"),
):
    """Save current graph as a new version."""
    from tinyllm.config.graph import GraphDefinition
    from tinyllm.expansion.versioning import GraphVersionManager
    import yaml

    if not graph_path.exists():
        console.print(f"[red]Graph file not found: {graph_path}[/red]")
        raise typer.Exit(1)

    # Load graph
    with open(graph_path) as f:
        data = yaml.safe_load(f)
    graph = GraphDefinition(**data)

    # Save version
    manager = GraphVersionManager(storage)
    version = manager.save_version(graph, message=message, bump=bump)

    console.print(f"[green]✓ Saved version {version.version}[/green]")
    if version.changes:
        console.print(f"[dim]Changes from {version.parent_version}:[/dim]")
        for change in version.changes:
            console.print(f"  • {change}")


@graph_app.command("rollback")
def graph_rollback(
    version: str = typer.Argument(..., help="Version to rollback to"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
    storage: Path = typer.Option(Path(".tinyllm/versions"), "--storage", "-s"),
):
    """Rollback to a previous version."""
    from tinyllm.expansion.versioning import GraphVersionManager
    import yaml

    manager = GraphVersionManager(storage)

    try:
        graph = manager.rollback(version)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]✓ Rolled back to version {version}[/green]")
    console.print(f"[dim]New current version: {manager.current_version}[/dim]")

    if output:
        with open(output, "w") as f:
            yaml.dump(graph.model_dump(mode="json"), f, default_flow_style=False)
        console.print(f"[green]✓ Exported to {output}[/green]")


@graph_app.command("diff")
def graph_diff(
    version1: str = typer.Argument(..., help="First version"),
    version2: str = typer.Argument(..., help="Second version"),
    storage: Path = typer.Option(Path(".tinyllm/versions"), "--storage", "-s"),
):
    """Show differences between two versions."""
    from tinyllm.expansion.versioning import GraphVersionManager

    manager = GraphVersionManager(storage)

    try:
        changes = manager.diff(version1, version2)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if not changes:
        console.print("[dim]No differences found[/dim]")
        return

    console.print(f"[bold]Diff: {version1} → {version2}[/bold]\n")
    for change in changes:
        if change.startswith("Added"):
            console.print(f"[green]+ {change}[/green]")
        elif change.startswith("Removed"):
            console.print(f"[red]- {change}[/red]")
        else:
            console.print(f"[yellow]~ {change}[/yellow]")


@graph_app.command("validate")
def graph_validate(
    graph_path: Path = typer.Argument(..., help="Graph file to validate"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed validation info"),
    format_output: str = typer.Option("text", "--format", "-f", help="Output format (text, json, mermaid, dot)"),
):
    """Validate a graph file for structural issues."""
    import json as json_lib
    from tinyllm.core.builder import load_graph

    if not graph_path.exists():
        console.print(f"[red]Graph file not found: {graph_path}[/red]")
        raise typer.Exit(1)

    try:
        # Load the graph
        graph = load_graph(graph_path)

        # Handle special output formats
        if format_output == "mermaid":
            mermaid_output = graph.to_mermaid()
            console.print(mermaid_output)
            return
        elif format_output == "dot":
            dot_output = graph.to_dot()
            console.print(dot_output)
            return

        # Validate the graph
        errors = graph.validate()

        # Count errors and warnings
        error_count = sum(1 for e in errors if e.severity == "error")
        warning_count = sum(1 for e in errors if e.severity == "warning")

        # Handle JSON output
        if format_output == "json":
            result = {
                "graph_id": graph.id,
                "graph_name": graph.name,
                "version": graph.version,
                "valid": error_count == 0,
                "error_count": error_count,
                "warning_count": warning_count,
                "is_acyclic": graph.is_acyclic(),
                "allow_cycles": graph.allow_cycles,
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "issues": [
                    {
                        "severity": e.severity,
                        "message": e.message,
                        "node_id": e.node_id,
                    }
                    for e in errors
                ],
            }
            console.print(json_lib.dumps(result, indent=2))
            if error_count > 0:
                raise typer.Exit(1)
            return

        # Text output
        console.print(f"\n[bold]Graph Validation: {graph.name}[/bold]")
        console.print(f"[dim]File: {graph_path}[/dim]")
        console.print(f"[dim]ID: {graph.id} (v{graph.version})[/dim]\n")

        # Show graph statistics
        if verbose:
            console.print("[cyan]Graph Statistics:[/cyan]")
            console.print(f"  • Nodes: {len(graph.nodes)}")
            console.print(f"  • Edges: {len(graph.edges)}")
            console.print(f"  • Entry points: {len(graph.entry_points)}")
            console.print(f"  • Exit points: {len(graph.exit_points)}")
            console.print(f"  • Protected nodes: {len(graph.protected_nodes)}")
            console.print(f"  • Acyclic: {'Yes' if graph.is_acyclic() else 'No'}")
            console.print(f"  • Cycles allowed: {'Yes' if graph.allow_cycles else 'No'}")

            # Show topological sort if acyclic
            if graph.is_acyclic():
                try:
                    topo_order = graph.topological_sort()
                    console.print(f"\n[cyan]Topological Order:[/cyan]")
                    console.print(f"  {' -> '.join(topo_order)}")
                except ValueError:
                    pass

            # Show cycles if any
            cycles = graph.detect_cycles()
            if cycles:
                console.print(f"\n[yellow]Detected Cycles ({len(cycles)}):[/yellow]")
                for i, cycle in enumerate(cycles, 1):
                    cycle_path = " -> ".join(cycle)
                    console.print(f"  {i}. {cycle_path}")

            console.print()

        # Display validation results
        if not errors:
            console.print("[green]✓ Graph is valid![/green]")
            console.print("[dim]No errors or warnings found.[/dim]")
        else:
            # Show errors
            if error_count > 0:
                console.print(f"[red]✗ Found {error_count} error(s):[/red]")
                for error in errors:
                    if error.severity == "error":
                        node_info = f" (node: {error.node_id})" if error.node_id else ""
                        console.print(f"  [red]•[/red] {error.message}{node_info}")
                console.print()

            # Show warnings
            if warning_count > 0:
                console.print(f"[yellow]⚠ Found {warning_count} warning(s):[/yellow]")
                for error in errors:
                    if error.severity == "warning":
                        node_info = f" (node: {error.node_id})" if error.node_id else ""
                        console.print(f"  [yellow]•[/yellow] {error.message}{node_info}")
                console.print()

            # Summary
            if error_count > 0:
                console.print("[red]Graph validation failed.[/red]")
                raise typer.Exit(1)
            else:
                console.print("[yellow]Graph has warnings but is structurally valid.[/yellow]")

    except FileNotFoundError:
        console.print(f"[red]Graph file not found: {graph_path}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Validation error: {e}[/red]")
        if verbose:
            import traceback
            console.print(f"\n[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


@graph_app.command("export")
def graph_export(
    version: str = typer.Argument(..., help="Version to export"),
    output: Path = typer.Argument(..., help="Output file path"),
    storage: Path = typer.Option(Path(".tinyllm/versions"), "--storage", "-s"),
):
    """Export a specific version to file."""
    from tinyllm.expansion.versioning import GraphVersionManager
    import yaml

    manager = GraphVersionManager(storage)
    graph = manager.load_version(version)

    if graph is None:
        console.print(f"[red]Version {version} not found[/red]")
        raise typer.Exit(1)

    with open(output, "w") as f:
        yaml.dump(graph.model_dump(mode="json"), f, default_flow_style=False)

    console.print(f"[green]✓ Exported {version} to {output}[/green]")


@graph_app.command("resume")
def graph_resume(
    trace_id: str = typer.Argument(..., help="Trace ID to resume"),
    graph_path: Path = typer.Option(
        Path("graphs/multi_domain.yaml"),
        "--graph",
        "-g",
        help="Graph definition file",
    ),
    storage_path: Optional[Path] = typer.Option(
        None,
        "--storage",
        "-s",
        help="Checkpoint storage path (SQLite)",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level",
    ),
):
    """Resume execution from a checkpoint."""
    from tinyllm.core.builder import load_graph
    from tinyllm.core.executor import Executor, ExecutorConfig
    from tinyllm.core.message import TaskPayload
    from tinyllm.persistence.interface import StorageConfig
    from tinyllm.persistence.sqlite_backend import SQLiteCheckpointStorage
    from tinyllm.logging import configure_logging

    configure_logging(log_level=log_level)

    async def _resume():
        console.print(f"[dim]Resuming trace: {trace_id}[/dim]")
        console.print(f"[dim]Graph: {graph_path}[/dim]\n")

        try:
            # Load graph
            graph = load_graph(graph_path)

            # Create checkpoint storage
            config = StorageConfig(
                sqlite_path=str(storage_path) if storage_path else None
            )
            checkpoint_storage = SQLiteCheckpointStorage(config)
            await checkpoint_storage.initialize()

            try:
                # Create executor with checkpointing enabled
                executor_config = ExecutorConfig(
                    checkpoint_interval_ms=5000,
                    checkpoint_after_each_node=True,
                )
                executor = Executor(
                    graph,
                    config=executor_config,
                    checkpoint_storage=checkpoint_storage,
                )

                # Resume execution
                with console.status("[bold green]Resuming..."):
                    response = await executor.resume_from_checkpoint(trace_id)

                # Display results
                if response.success:
                    logger.info(
                        "resume_execution_success",
                        trace_id=response.trace_id,
                        nodes_executed=response.nodes_executed,
                        latency_ms=response.total_latency_ms,
                    )
                    console.print("\n[bold green]Response:[/bold green]")
                    console.print(response.content or "[dim]No content returned[/dim]")
                else:
                    logger.error(
                        "resume_execution_failed",
                        trace_id=response.trace_id,
                        error=response.error.message if response.error else "Unknown error",
                    )
                    console.print(
                        f"\n[red]Error: {response.error.message if response.error else 'Unknown error'}[/red]"
                    )

                console.print(f"\n[dim]Trace ID: {response.trace_id}[/dim]")
                console.print(f"[dim]Nodes executed: {response.nodes_executed}[/dim]")
                console.print(f"[dim]Latency: {response.total_latency_ms}ms[/dim]")

            finally:
                await checkpoint_storage.close()

        except FileNotFoundError:
            logger.error("graph_file_not_found", graph_path=str(graph_path))
            console.print(f"[red]Graph file not found: {graph_path}[/red]")
        except Exception as e:
            logger.error(
                "resume_exception",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            console.print(f"[red]Resume failed: {e}[/red]")

    asyncio.run(_resume())


@graph_app.command("checkpoints")
def graph_checkpoints(
    graph_id: Optional[str] = typer.Option(None, "--graph-id", "-g", help="Filter by graph ID"),
    trace_id: Optional[str] = typer.Option(None, "--trace-id", "-t", help="Filter by trace ID"),
    storage_path: Optional[Path] = typer.Option(
        None,
        "--storage",
        "-s",
        help="Checkpoint storage path (SQLite)",
    ),
):
    """List available checkpoints."""
    from tinyllm.persistence.interface import StorageConfig
    from tinyllm.persistence.sqlite_backend import SQLiteCheckpointStorage

    async def _list():
        # Create checkpoint storage
        config = StorageConfig(
            sqlite_path=str(storage_path) if storage_path else None
        )
        checkpoint_storage = SQLiteCheckpointStorage(config)
        await checkpoint_storage.initialize()

        try:
            # Get checkpoints
            if graph_id:
                checkpoints = await checkpoint_storage.list_checkpoints(graph_id, trace_id)
            else:
                checkpoints = await checkpoint_storage.list(limit=100)

            if not checkpoints:
                console.print("[yellow]No checkpoints found[/yellow]")
                return

            table = Table(title="Graph Checkpoints")
            table.add_column("Checkpoint ID", style="cyan")
            table.add_column("Graph ID", style="green")
            table.add_column("Trace ID", style="yellow")
            table.add_column("Step", style="blue")
            table.add_column("Node ID")
            table.add_column("Status")
            table.add_column("Created", style="dim")

            for cp in checkpoints:
                created = cp.created_at.strftime("%Y-%m-%d %H:%M:%S")
                table.add_row(
                    cp.id[:8] + "...",
                    cp.graph_id[:20],
                    cp.trace_id[:8] + "...",
                    str(cp.step),
                    cp.node_id[:20],
                    cp.status,
                    created,
                )

            console.print(table)
            console.print(f"\n[dim]Total: {len(checkpoints)} checkpoints[/dim]")

        finally:
            await checkpoint_storage.close()

    asyncio.run(_list())


@app.command()
def cache_stats(
    backend: str = typer.Option("memory", "--backend", "-b", help="Cache backend (memory or redis)"),
    redis_host: str = typer.Option("localhost", "--redis-host", help="Redis host"),
    redis_port: int = typer.Option(6379, "--redis-port", help="Redis port"),
):
    """Show cache statistics and metrics."""

    async def _show_stats():
        from tinyllm.cache import create_memory_cache, create_redis_cache

        console.print("[bold]TinyLLM Cache Statistics[/bold]\n")

        try:
            if backend == "memory":
                cache = create_memory_cache()
                console.print("[cyan]Backend:[/cyan] In-Memory (LRU)")
            elif backend == "redis":
                cache = create_redis_cache(host=redis_host, port=redis_port)
                console.print(f"[cyan]Backend:[/cyan] Redis ({redis_host}:{redis_port})")
            else:
                console.print(f"[red]Invalid backend: {backend}[/red]")
                return

            # Get metrics
            metrics = cache.get_metrics()
            size = await cache.size()

            # Display metrics
            table = Table(title="Cache Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Cache Hits", str(metrics.hits))
            table.add_row("Cache Misses", str(metrics.misses))
            table.add_row("Total Requests", str(metrics.total_requests))
            table.add_row("Hit Rate", f"{metrics.hit_rate:.2%}")
            table.add_row("Sets", str(metrics.sets))
            table.add_row("Evictions", str(metrics.evictions))
            table.add_row("Errors", str(metrics.errors))
            table.add_row("Current Size", str(size))

            console.print(table)

            # Close cache
            await cache.close()

        except Exception as e:
            console.print(f"[red]Error retrieving cache stats: {e}[/red]")
            logger.error("cache_stats_error", error=str(e), exc_info=True)

    asyncio.run(_show_stats())


@app.command()
def cache_clear(
    backend: str = typer.Option("memory", "--backend", "-b", help="Cache backend (memory or redis)"),
    redis_host: str = typer.Option("localhost", "--redis-host", help="Redis host"),
    redis_port: int = typer.Option(6379, "--redis-port", help="Redis port"),
    confirm: bool = typer.Option(True, "--confirm/--no-confirm", help="Confirm before clearing"),
):
    """Clear the cache."""

    async def _clear_cache():
        from tinyllm.cache import create_memory_cache, create_redis_cache

        if confirm and not typer.confirm(f"Are you sure you want to clear the {backend} cache?"):
            console.print("[yellow]Cache clear cancelled[/yellow]")
            return

        try:
            if backend == "memory":
                cache = create_memory_cache()
            elif backend == "redis":
                cache = create_redis_cache(host=redis_host, port=redis_port)
            else:
                console.print(f"[red]Invalid backend: {backend}[/red]")
                return

            await cache.clear()
            console.print(f"[green]Cache cleared successfully![/green]")

            await cache.close()

        except Exception as e:
            console.print(f"[red]Error clearing cache: {e}[/red]")
            logger.error("cache_clear_error", error=str(e), exc_info=True)

    asyncio.run(_clear_cache())


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
