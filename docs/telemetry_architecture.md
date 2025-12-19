# Telemetry Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         TinyLLM Application                      │
│                                                                   │
│  ┌────────────────┐      ┌─────────────────┐                   │
│  │   CLI / API    │      │  Executor       │                   │
│  │                │─────▶│  (Future)       │                   │
│  └────────────────┘      └─────────────────┘                   │
│         │                         │                              │
│         │                         ▼                              │
│         │                ┌─────────────────┐                   │
│         │                │  Graph Nodes    │                   │
│         │                │  - Entry/Exit   │                   │
│         │                │  - Model        │                   │
│         │                │  - Router       │                   │
│         │                └─────────────────┘                   │
│         │                         │                              │
│         └─────────────────────────┼──────────────────┐          │
│                                   ▼                   │          │
│                          ┌─────────────────┐         │          │
│                          │  OllamaClient   │◀────────┘          │
│                          │  (Traced)       │                    │
│                          └─────────────────┘                    │
│                                   │                              │
│                                   │                              │
└───────────────────────────────────┼──────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   OpenTelemetry SDK          │
                    │   - Tracer Provider          │
                    │   - Span Processors          │
                    │   - Samplers                 │
                    └───────────────────────────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  │                 │                 │
                  ▼                 ▼                 ▼
          ┌──────────────┐  ┌─────────────┐  ┌──────────────┐
          │   Console    │  │    OTLP     │  │   Jaeger     │
          │   Exporter   │  │  Exporter   │  │  (via OTLP)  │
          └──────────────┘  └─────────────┘  └──────────────┘
                  │                 │                 │
                  ▼                 ▼                 ▼
          ┌──────────────┐  ┌─────────────┐  ┌──────────────┐
          │   stdout     │  │  Collector  │  │   Jaeger UI  │
          │              │  │   (OTLP)    │  │              │
          └──────────────┘  └─────────────┘  └──────────────┘
                                    │
                                    ▼
                            ┌─────────────┐
                            │  Backends   │
                            │  - Jaeger   │
                            │  - Tempo    │
                            │  - etc.     │
                            └─────────────┘
```

## Trace Span Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│ graph.execute                                                    │
│ ├─ attributes:                                                   │
│ │  ├─ graph.id: "multi_domain"                                  │
│ │  ├─ execution.trace_id: "abc123..."                           │
│ │  └─ task.content: "What is 2+2?"                              │
│ │                                                                │
│ ├─ node.entry                                                   │
│ │  ├─ attributes:                                               │
│ │  │  ├─ node.id: "entry"                                       │
│ │  │  ├─ node.type: "entry"                                     │
│ │  │  └─ execution.step: 1                                      │
│ │  │                                                             │
│ ├─ node.router                                                  │
│ │  ├─ attributes:                                               │
│ │  │  ├─ node.id: "classifier"                                  │
│ │  │  ├─ node.type: "router"                                    │
│ │  │  └─ execution.step: 2                                      │
│ │  │                                                             │
│ │  └─ llm.generate                                              │
│ │     ├─ attributes:                                            │
│ │     │  ├─ llm.model: "qwen2.5:0.5b"                          │
│ │     │  ├─ llm.prompt_length: 150                             │
│ │     │  ├─ llm.temperature: 0.3                               │
│ │     │  ├─ llm.input_tokens: 45                               │
│ │     │  ├─ llm.output_tokens: 12                              │
│ │     │  └─ llm.total_tokens: 57                               │
│ │     │                                                          │
│ ├─ node.model                                                   │
│ │  ├─ attributes:                                               │
│ │  │  ├─ node.id: "math_specialist"                            │
│ │  │  ├─ node.type: "model"                                    │
│ │  │  └─ execution.step: 3                                     │
│ │  │                                                             │
│ │  └─ llm.generate                                              │
│ │     ├─ attributes:                                            │
│ │     │  ├─ llm.model: "qwen2.5:3b"                            │
│ │     │  ├─ llm.prompt_length: 200                             │
│ │     │  ├─ llm.temperature: 0.3                               │
│ │     │  ├─ llm.input_tokens: 80                               │
│ │     │  ├─ llm.output_tokens: 25                              │
│ │     │  └─ llm.total_tokens: 105                              │
│ │     │                                                          │
│ └─ node.exit                                                    │
│    ├─ attributes:                                               │
│    │  ├─ node.id: "exit"                                        │
│    │  ├─ node.type: "exit"                                      │
│    │  └─ execution.step: 4                                      │
│    │                                                             │
│    └─ final attributes:                                         │
│       ├─ execution.nodes_executed: 4                            │
│       ├─ execution.tokens_used: 162                             │
│       ├─ execution.duration_ms: 450                             │
│       └─ execution.success: true                                │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
┌──────────────┐
│ User Request │
└──────┬───────┘
       │
       ▼
┌────────────────────────────────────────────┐
│ 1. CLI configures telemetry               │
│    - TelemetryConfig created               │
│    - configure_telemetry() called          │
│    - Tracer provider initialized           │
└────────────────┬───────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────┐
│ 2. Executor starts (Future)                │
│    - trace_executor_execution() creates    │
│      root span                             │
│    - Trace ID generated                    │
│    - Added to log context                  │
└────────────────┬───────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────┐
│ 3. Node execution                          │
│    - trace_node_execution() creates        │
│      child span                            │
│    - Node attributes added                 │
│    - Step number recorded                  │
└────────────────┬───────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────┐
│ 4. LLM request                             │
│    - trace_llm_request() creates           │
│      child span                            │
│    - Model, prompt details added           │
│    - Token counts recorded after response  │
└────────────────┬───────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────┐
│ 5. Span completion                         │
│    - Spans closed in reverse order         │
│    - Final attributes added                │
│    - Errors recorded if any                │
└────────────────┬───────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────┐
│ 6. Export                                  │
│    - BatchSpanProcessor batches spans      │
│    - Exporter sends to backend             │
│    - Async, non-blocking                   │
└────────────────────────────────────────────┘
```

## Component Interactions

```
┌─────────────────────────────────────────────────────────────────┐
│                      telemetry.py                                │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ TelemetryConfig                                          │  │
│  │  - enable_tracing: bool                                  │  │
│  │  - service_name: str                                     │  │
│  │  - exporter: str                                         │  │
│  │  - otlp_endpoint: Optional[str]                          │  │
│  │  - sampling_rate: float                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ configure_telemetry(config)                              │  │
│  │  - Creates TracerProvider                                │  │
│  │  - Configures exporter (console/OTLP)                    │  │
│  │  - Sets up span processor                                │  │
│  │  - Configures sampler                                    │  │
│  │  - Sets global tracer                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Tracing Functions                                        │  │
│  │  - trace_span(name, attributes)                          │  │
│  │  - set_span_attribute(key, value)                        │  │
│  │  - record_span_event(name, attrs)                        │  │
│  │  - set_span_error(exception)                             │  │
│  │  - get_current_trace_id()                                │  │
│  │  - get_current_span_id()                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Decorators                                               │  │
│  │  - @traced(span_name, attributes)                        │  │
│  │  - @traced_method(span_name, attributes)                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Helper Functions                                         │  │
│  │  - trace_executor_execution()                            │  │
│  │  - trace_node_execution()                                │  │
│  │  - trace_llm_request()                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Integration with Logging

```
┌─────────────────────────────────────────────────────────────────┐
│ Trace Context + Logging Context                                 │
│                                                                   │
│  trace_span("operation", add_to_logs=True)                      │
│      │                                                            │
│      ├─ Creates OpenTelemetry span                              │
│      │  ├─ Generates trace_id                                   │
│      │  └─ Generates span_id                                    │
│      │                                                            │
│      └─ Binds to logging context                                │
│         ├─ bind_context(trace_id=trace_id)                      │
│         └─ bind_context(span_id=span_id)                        │
│                                                                   │
│  Within span context:                                            │
│      logger.info("event", user_id="123")                        │
│          │                                                        │
│          └─ Log output:                                          │
│             {                                                     │
│               "event": "event",                                  │
│               "user_id": "123",                                  │
│               "trace_id": "abc123...",  ← Automatically added   │
│               "span_id": "def456...",   ← Automatically added   │
│               "timestamp": "...",                                │
│               "level": "info"                                    │
│             }                                                     │
│                                                                   │
│  After span ends:                                                │
│      unbind_context("trace_id", "span_id")                      │
└─────────────────────────────────────────────────────────────────┘
```

## Observability Stack Integration

```
                    ┌─────────────────────┐
                    │   TinyLLM Service   │
                    └──────────┬──────────┘
                               │
               ┌───────────────┼───────────────┐
               │               │               │
               ▼               ▼               ▼
        ┌──────────┐    ┌──────────┐   ┌──────────┐
        │  Traces  │    │   Logs   │   │ Metrics  │
        │ (OTel)   │    │(Structlog│   │(Prometheus)│
        └─────┬────┘    └────┬─────┘   └────┬─────┘
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐    ┌──────────┐   ┌──────────┐
        │  Jaeger  │    │   Loki   │   │ Prometheus│
        │  /Tempo  │    │          │   │          │
        └─────┬────┘    └────┬─────┘   └────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌─────────────────┐
                    │    Grafana      │
                    │  (Unified View) │
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Correlate by   │
                    │  - Trace ID     │
                    │  - Timestamp    │
                    │  - Labels       │
                    └─────────────────┘
```

## Performance Characteristics

```
Operation               Overhead    Notes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Span creation           ~0.1ms      One-time per span
Add attribute           ~0.01ms     Per attribute
Record event            ~0.01ms     Per event
Console export          1-5ms       Synchronous, blocking
OTLP export            <1ms        Async, batched
Sampling decision      <0.01ms     Per trace
Context propagation    <0.01ms     Per span

Total overhead (typical):
- No sampling: 2-10ms per request
- With 10% sampling: 0.2-1ms average
- Tracing disabled: 0ms (zero overhead)
```

## Security Model

```
┌─────────────────────────────────────────────────────────────────┐
│ Sensitive Data Handling                                          │
│                                                                   │
│  ✓ DO include:                                                   │
│    - Request IDs, trace IDs                                     │
│    - User IDs (anonymized)                                      │
│    - Operation names                                             │
│    - Performance metrics                                         │
│    - Error codes                                                 │
│                                                                   │
│  ✗ DON'T include:                                               │
│    - Passwords, API keys                                        │
│    - PII (emails, names, addresses)                             │
│    - Full prompt content (truncate)                             │
│    - Model responses (summarize)                                │
│                                                                   │
│  Mitigation:                                                     │
│    - Truncate long strings                                      │
│    - Redact sensitive patterns                                  │
│    - Use attribute limits                                       │
│    - Configure retention policies                               │
└─────────────────────────────────────────────────────────────────┘
```

## Future Extensions

```
┌─────────────────────────────────────────────────────────────────┐
│ Planned Enhancements                                             │
│                                                                   │
│  1. Full Executor Integration                                   │
│     └─ Trace complete graph execution flow                      │
│                                                                   │
│  2. Node-Level Tracing                                          │
│     ├─ Router decision spans                                    │
│     ├─ Tool execution spans                                     │
│     └─ Transform operation spans                                │
│                                                                   │
│  3. Distributed Tracing                                         │
│     ├─ Context propagation across services                      │
│     ├─ B3/W3C trace context headers                            │
│     └─ Multi-service correlation                                │
│                                                                   │
│  4. Advanced Sampling                                            │
│     ├─ Error-based sampling                                     │
│     ├─ Latency-based sampling                                   │
│     └─ Custom sampler plugins                                   │
│                                                                   │
│  5. Metric Correlation                                          │
│     ├─ Link traces to Prometheus metrics                        │
│     ├─ Exemplars in metrics                                     │
│     └─ Unified observability queries                            │
└─────────────────────────────────────────────────────────────────┘
```
