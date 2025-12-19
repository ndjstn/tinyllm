"""Example demonstrating OpenTelemetry distributed tracing with TinyLLM.

This example shows how to:
1. Configure OpenTelemetry tracing
2. Trace LLM requests
3. Add custom spans and attributes
4. View trace data in console or export to OTLP

Usage:
    # Console output
    python examples/tracing_example.py

    # Export to Jaeger (requires Jaeger running)
    python examples/tracing_example.py --otlp

Prerequisites:
    pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp

    # For Jaeger:
    docker run -d -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one:latest
    # Then open http://localhost:16686 to view traces
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinyllm.models.client import OllamaClient
from tinyllm.telemetry import (
    TelemetryConfig,
    configure_telemetry,
    get_current_trace_id,
    record_span_event,
    set_span_attribute,
    trace_span,
)


async def main():
    """Demonstrate distributed tracing."""
    import argparse

    parser = argparse.ArgumentParser(description="TinyLLM Tracing Example")
    parser.add_argument(
        "--otlp",
        action="store_true",
        help="Export traces to OTLP/Jaeger instead of console",
    )
    args = parser.parse_args()

    # Configure telemetry
    print("Configuring telemetry...")
    config = TelemetryConfig(
        enable_tracing=True,
        service_name="tinyllm-example",
        exporter="otlp" if args.otlp else "console",
        otlp_endpoint="http://localhost:4317" if args.otlp else None,
        sampling_rate=1.0,  # Sample all traces
    )

    try:
        configure_telemetry(config)
        print(f"✓ Telemetry configured (exporter: {config.exporter})")
        if args.otlp:
            print("  Traces will be sent to http://localhost:4317")
            print("  View traces at http://localhost:16686")
    except ImportError:
        print("✗ OpenTelemetry not installed")
        print("  Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp")
        return

    # Create Ollama client
    client = OllamaClient()

    # Execute a traced operation
    print("\nExecuting traced operations...")
    with trace_span(
        "example.workflow",
        attributes={
            "workflow.type": "text_generation",
            "workflow.version": "1.0",
        },
    ) as span:
        # Get trace ID for correlation
        trace_id = get_current_trace_id()
        if trace_id:
            print(f"Trace ID: {trace_id}")

        # Add custom event
        record_span_event("workflow.started", {"user": "demo"})

        # Step 1: Generate a joke
        with trace_span("step.generate_joke", attributes={"step": 1}):
            print("\n[Step 1] Generating joke...")
            set_span_attribute("joke.category", "programming")

            try:
                response = await client.generate(
                    model="qwen2.5:0.5b",
                    prompt="Tell me a short programming joke",
                    temperature=0.7,
                    max_tokens=100,
                )
                joke = response.response
                print(f"  Joke: {joke[:100]}...")

                # Add result to span
                set_span_attribute("joke.length", len(joke))
                record_span_event("joke.generated", {"success": True})
            except Exception as e:
                print(f"  Error: {e}")
                set_span_attribute("error", str(e))
                return

        # Step 2: Explain the joke
        with trace_span("step.explain_joke", attributes={"step": 2}):
            print("\n[Step 2] Explaining joke...")
            set_span_attribute("explanation.style", "simple")

            try:
                response = await client.generate(
                    model="qwen2.5:0.5b",
                    prompt=f"Explain this joke in one sentence: {joke}",
                    temperature=0.3,
                    max_tokens=100,
                )
                explanation = response.response
                print(f"  Explanation: {explanation[:100]}...")

                set_span_attribute("explanation.length", len(explanation))
                record_span_event("explanation.generated", {"success": True})
            except Exception as e:
                print(f"  Error: {e}")
                set_span_attribute("error", str(e))
                return

        # Mark workflow as complete
        record_span_event("workflow.completed", {"status": "success"})
        print("\n✓ Workflow completed successfully")

    # Clean up
    await client.close()

    print("\nTrace data has been exported.")
    if args.otlp:
        print("View traces in Jaeger at: http://localhost:16686")
        print("Search for service: tinyllm-example")
    else:
        print("Trace spans were printed to console above.")


if __name__ == "__main__":
    asyncio.run(main())
