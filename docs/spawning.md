# Dynamic Node Spawning System

The TinyLLM spawning system provides dynamic node instantiation and management capabilities for runtime graph expansion. This enables adaptive scaling, specialization, and experimentation without requiring graph recompilation.

## Table of Contents

- [Overview](#overview)
- [Core Components](#core-components)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Examples](#examples)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

## Overview

### What is Node Spawning?

Node spawning is the ability to dynamically create new node instances at runtime based on:

- **Performance metrics**: Scale out when nodes are underperforming
- **Workload spikes**: Add capacity when request volume increases
- **Specialization needs**: Create domain-specific variants of general nodes
- **Manual requests**: Programmatic node creation for experimentation

### Key Features

- ✅ **Configurable Limits**: Control max spawns, cooldown periods, and depth
- ✅ **Multiple Triggers**: Manual, performance threshold, workload spike, specialization
- ✅ **Spawn Tracking**: Complete history and metrics for all spawns
- ✅ **Clone Chains**: Track depth of clones-of-clones with max depth enforcement
- ✅ **Validation**: Production-ready error handling and Pydantic models
- ✅ **Metrics**: Real-time statistics on spawn operations and performance

## Core Components

### 1. SpawnTrigger

Enumeration of spawn trigger reasons:

```python
class SpawnTrigger(str, Enum):
    PERFORMANCE_THRESHOLD = "performance_threshold"  # Performance below threshold
    WORKLOAD_SPIKE = "workload_spike"                # Sudden workload increase
    SPECIALIZATION_NEEDED = "specialization_needed"  # Domain specialization
    MANUAL = "manual"                                # Manual spawn request
```

### 2. SpawnConfig

Configuration for spawn behavior:

```python
config = SpawnConfig(
    max_spawns=10,              # Maximum concurrent spawns
    cooldown_ms=5000,           # Cooldown between spawns (ms)
    performance_threshold=0.6,   # Performance trigger threshold
    workload_threshold=100,      # Workload trigger threshold
    enable_auto_spawn=False,     # Enable automatic spawning
    max_spawn_depth=3,           # Maximum clone chain depth
)
```

**Field Descriptions:**

- `max_spawns` (1-100): Maximum number of spawned nodes allowed concurrently
- `cooldown_ms` (100-60000): Minimum time between spawn operations in milliseconds
- `performance_threshold` (0.0-1.0): Performance score below which spawning triggers
- `workload_threshold` (1-10000): Request count above which spawning triggers
- `enable_auto_spawn`: Whether to automatically spawn based on metrics
- `max_spawn_depth` (1-10): Maximum depth of spawn chains (prevents infinite cloning)

### 3. NodeFactory

Factory class for node instantiation:

```python
from tinyllm.expansion import NodeFactory

# Create from definition
node = NodeFactory.create_from_definition(definition)

# Create from template with overrides
node = NodeFactory.create_from_template(
    template=template_definition,
    new_id="specialized_node",
    config_overrides={"model": "qwen2.5:3b"}
)

# Clone node definition
definition = NodeFactory.clone_node_definition(
    node=existing_node,
    new_id="cloned_node",
    modifications={"temperature": 0.9}
)
```

### 4. NodeSpawner

Main spawning orchestrator:

```python
from tinyllm.expansion import NodeSpawner, SpawnConfig

spawner = NodeSpawner(SpawnConfig(max_spawns=5))

# Spawn from template
node = spawner.spawn_node(
    template=template_def,
    specialization={"model": "better-model"},
    trigger=SpawnTrigger.MANUAL
)

# Clone existing node
clone = spawner.clone_node(
    node=existing_node,
    modifications={"temperature": 0.95}
)

# Check if spawning is allowed
if spawner.can_spawn():
    print("Ready to spawn")

# Get spawn history
history = spawner.get_spawn_history(limit=10)

# Get metrics
metrics = spawner.get_metrics()
```

### 5. SpawnRecord

Record of each spawn operation:

```python
class SpawnRecord(BaseModel):
    id: str                      # Unique spawn record ID
    spawned_node_id: str         # ID of spawned node
    template_node_id: str        # ID of template/parent
    trigger: SpawnTrigger        # What triggered spawn
    timestamp: datetime          # When spawned
    specialization: Dict         # Applied specializations
    spawn_depth: int             # Depth in clone chain
    metadata: Dict               # Additional metadata
```

### 6. SpawnMetrics

Aggregate metrics for spawn operations:

```python
class SpawnMetrics(BaseModel):
    total_spawns: int                    # Total spawns created
    active_spawns: int                   # Currently active
    failed_spawns: int                   # Failed attempts
    avg_spawn_latency_ms: float         # Average latency
    spawns_by_trigger: Dict[str, int]   # Counts by trigger
```

## Usage Guide

### Basic Spawning

```python
from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.expansion import NodeSpawner, SpawnConfig, SpawnTrigger

# Create spawner
spawner = NodeSpawner(SpawnConfig(max_spawns=10))

# Define template
template = NodeDefinition(
    id="base_model",
    type=NodeType.MODEL,
    name="Base Model",
    config={"model": "qwen2.5:1.5b", "temperature": 0.7}
)

# Spawn specialized variant
specialized = spawner.spawn_node(
    template=template,
    specialization={
        "model": "qwen2.5:3b",
        "temperature": 0.9
    },
    trigger=SpawnTrigger.SPECIALIZATION_NEEDED
)

print(f"Spawned: {specialized.id}")
```

### Performance-Based Spawning

```python
# Check if performance warrants spawning
current_performance = node.stats.success_rate
should_spawn = spawner.check_performance_trigger(
    node_id=node.id,
    current_performance=current_performance
)

if should_spawn:
    # Spawn upgraded variant
    upgraded = spawner.spawn_node(
        template=template,
        specialization={"model": "qwen2.5:7b"},
        trigger=SpawnTrigger.PERFORMANCE_THRESHOLD
    )
```

### Workload-Based Spawning

```python
# Check if workload warrants spawning
current_workload = get_current_request_count()
should_spawn = spawner.check_workload_trigger(
    node_id=node.id,
    current_workload=current_workload
)

if should_spawn:
    # Spawn additional capacity
    clone = spawner.clone_node(
        node=node,
        trigger=SpawnTrigger.WORKLOAD_SPIKE
    )
```

### Spawn Chain Management

```python
# Create spawn chain: original -> clone -> clone-of-clone
original = spawner.spawn_node(template)
sleep(0.1)  # Wait for cooldown

clone1 = spawner.clone_node(original)
record1 = spawner.get_spawn_record(clone1.id)
print(f"Clone depth: {record1.spawn_depth}")  # 1

sleep(0.1)
clone2 = spawner.clone_node(clone1)
record2 = spawner.get_spawn_record(clone2.id)
print(f"Clone depth: {record2.spawn_depth}")  # 2
```

### Spawn History & Metrics

```python
# Get all spawn history
all_history = spawner.get_spawn_history()

# Filter by trigger type
manual_spawns = spawner.get_spawn_history(
    trigger=SpawnTrigger.MANUAL,
    limit=5
)

# Get children of a template
children = spawner.get_children("template_node_id")

# Get comprehensive metrics
metrics = spawner.get_metrics()
print(f"Total spawns: {metrics.total_spawns}")
print(f"Active: {metrics.active_spawns}")
print(f"Failed: {metrics.failed_spawns}")
print(f"Avg latency: {metrics.avg_spawn_latency_ms:.2f}ms")
print(f"By trigger: {metrics.spawns_by_trigger}")
```

### Despawning Nodes

```python
# Remove from active tracking
success = spawner.despawn_node(node_id)
if success:
    print("Node despawned")
else:
    print("Node not found")
```

## Configuration

### SpawnConfig Options

```python
from tinyllm.expansion import SpawnConfig

# Conservative configuration (production)
conservative = SpawnConfig(
    max_spawns=5,
    cooldown_ms=10000,      # 10 second cooldown
    performance_threshold=0.7,
    workload_threshold=200,
    enable_auto_spawn=False,  # Manual only
    max_spawn_depth=2
)

# Aggressive configuration (development/testing)
aggressive = SpawnConfig(
    max_spawns=20,
    cooldown_ms=1000,       # 1 second cooldown
    performance_threshold=0.5,
    workload_threshold=50,
    enable_auto_spawn=True,   # Auto-spawn enabled
    max_spawn_depth=5
)

# Minimal configuration (experimentation)
minimal = SpawnConfig(
    max_spawns=100,
    cooldown_ms=100,
    performance_threshold=0.0,
    workload_threshold=1,
    enable_auto_spawn=True,
    max_spawn_depth=10
)
```

### Validation Rules

All configuration fields are validated by Pydantic:

- `max_spawns`: Must be 1-100
- `cooldown_ms`: Must be 100-60000
- `performance_threshold`: Must be 0.0-1.0
- `workload_threshold`: Must be 1-10000
- `max_spawn_depth`: Must be 1-10

Invalid values will raise `ValueError` with descriptive messages.

## Examples

### Example 1: A/B Testing Different Models

```python
spawner = NodeSpawner(SpawnConfig(max_spawns=10))

# Spawn variants with different models
variants = []
for model in ["qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b"]:
    variant = spawner.spawn_node(
        template=base_template,
        specialization={"model": model},
        trigger=SpawnTrigger.MANUAL
    )
    variants.append(variant)
    sleep(0.1)  # Wait for cooldown

# Run tests and compare performance
# ...
```

### Example 2: Auto-Scaling Under Load

```python
config = SpawnConfig(
    max_spawns=10,
    workload_threshold=50,
    enable_auto_spawn=True
)
spawner = NodeSpawner(config)

# Monitor and auto-spawn
while True:
    workload = get_request_count()

    if spawner.check_workload_trigger("processor", workload):
        try:
            clone = spawner.clone_node(processor_node)
            print(f"Auto-spawned: {clone.id}")
        except RuntimeError as e:
            print(f"Spawn failed: {e}")

    sleep(1)
```

### Example 3: Domain Specialization

```python
# Create specialized variants for different domains
domains = {
    "math": {"system_prompt": "You are a math expert..."},
    "code": {"system_prompt": "You are a coding expert..."},
    "writing": {"system_prompt": "You are a writing expert..."}
}

specialists = {}
for domain, config in domains.items():
    specialist = spawner.spawn_node(
        template=general_template,
        specialization=config,
        trigger=SpawnTrigger.SPECIALIZATION_NEEDED
    )
    specialists[domain] = specialist
    sleep(0.1)
```

## Best Practices

### 1. Set Appropriate Limits

- **Production**: Conservative limits (max_spawns=5-10, cooldown_ms=5000-10000)
- **Development**: Moderate limits (max_spawns=10-20, cooldown_ms=1000-5000)
- **Testing**: Relaxed limits (max_spawns=50+, cooldown_ms=100)

### 2. Use Auto-Spawn Carefully

- Disable auto-spawn in production initially
- Test auto-spawn behavior in staging
- Monitor spawn metrics closely
- Set conservative thresholds

### 3. Track Spawn Metrics

```python
# Regularly check spawn health
metrics = spawner.get_metrics()
if metrics.failed_spawns > metrics.total_spawns * 0.1:
    print("WARNING: High spawn failure rate")

if metrics.active_spawns == config.max_spawns:
    print("WARNING: At spawn limit")
```

### 4. Clean Up Inactive Spawns

```python
# Despawn underutilized nodes
for record in spawner.get_spawn_history():
    node = get_node(record.spawned_node_id)
    if node.stats.total_executions < 10:
        spawner.despawn_node(record.spawned_node_id)
```

### 5. Use Meaningful Specializations

```python
# Good: Clear, purposeful specialization
spawner.spawn_node(
    template=template,
    specialization={
        "model": "qwen2.5:7b",
        "temperature": 0.9,
        "system_prompt": "Expert in quantum physics"
    }
)

# Bad: Arbitrary changes
spawner.spawn_node(
    template=template,
    specialization={"random_param": 123}
)
```

### 6. Respect Cooldown Periods

```python
# Wait for cooldown between spawns
if not spawner.can_spawn(check_cooldown=True):
    wait_time = calculate_remaining_cooldown(spawner)
    sleep(wait_time)

# Or use check_cooldown=False only when needed
if spawner.can_spawn(check_cooldown=False):
    # Spawn anyway (use sparingly)
    pass
```

## API Reference

### NodeFactory

#### `create_from_definition(definition: NodeDefinition) -> BaseNode`

Create a node instance from a NodeDefinition.

**Parameters:**
- `definition`: Node definition to instantiate

**Returns:** Instantiated node

**Raises:** `ValueError` if node type not registered

#### `create_from_template(template: NodeDefinition, new_id: str, config_overrides: Optional[Dict] = None) -> BaseNode`

Create a node from a template with modifications.

**Parameters:**
- `template`: Template node definition
- `new_id`: ID for the new node
- `config_overrides`: Configuration overrides

**Returns:** New node instance

#### `clone_node_definition(node: BaseNode, new_id: str, modifications: Optional[Dict] = None) -> NodeDefinition`

Create a NodeDefinition by cloning an existing node.

**Parameters:**
- `node`: Node to clone
- `new_id`: ID for cloned node
- `modifications`: Configuration modifications

**Returns:** New node definition

### NodeSpawner

#### `__init__(config: Optional[SpawnConfig] = None)`

Initialize the node spawner.

#### `can_spawn(check_cooldown: bool = True) -> bool`

Check if spawning is currently allowed.

#### `spawn_node(template: NodeDefinition, specialization: Optional[Dict] = None, trigger: SpawnTrigger = SpawnTrigger.MANUAL, spawn_depth: int = 0) -> BaseNode`

Spawn a new node from a template.

#### `clone_node(node: BaseNode, modifications: Optional[Dict] = None, trigger: SpawnTrigger = SpawnTrigger.MANUAL) -> BaseNode`

Clone an existing node with optional modifications.

#### `despawn_node(node_id: str) -> bool`

Remove a spawned node from active tracking.

#### `get_spawn_history(node_id: Optional[str] = None, trigger: Optional[SpawnTrigger] = None, limit: Optional[int] = None) -> List[SpawnRecord]`

Get spawn history with optional filtering.

#### `get_spawn_record(node_id: str) -> Optional[SpawnRecord]`

Get spawn record for a specific node.

#### `get_children(template_id: str) -> List[SpawnRecord]`

Get all spawns created from a template.

#### `get_metrics() -> SpawnMetrics`

Get current spawn metrics.

#### `check_performance_trigger(node_id: str, current_performance: float) -> bool`

Check if performance threshold triggers spawning.

#### `check_workload_trigger(node_id: str, current_workload: int) -> bool`

Check if workload spike triggers spawning.

#### `reset() -> None`

Reset spawner state (for testing/debugging).

#### `get_stats() -> Dict[str, Any]`

Get detailed spawner statistics.

---

**Next Steps:**
- See [examples/spawning_demo.py](../examples/spawning_demo.py) for a complete example
- Review [tests/unit/test_spawning.py](../tests/unit/test_spawning.py) for usage patterns
- Explore the expansion module for integration with the expansion engine
