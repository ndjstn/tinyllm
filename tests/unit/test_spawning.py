"""Tests for dynamic node spawning system."""

import pytest
from datetime import datetime, timedelta
from time import sleep

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.core.node import BaseNode, NodeResult
from tinyllm.core.registry import NodeRegistry
from tinyllm.expansion.spawning import (
    NodeFactory,
    NodeSpawner,
    SpawnConfig,
    SpawnMetrics,
    SpawnRecord,
    SpawnTrigger,
)


# Test node implementation
class TestNode(BaseNode):
    """Simple test node for spawning tests."""

    async def execute(self, message, context):
        """Execute test logic."""
        return NodeResult.success_result(
            output_messages=[message],
            next_nodes=[],
            latency_ms=10,
        )


@pytest.fixture(autouse=True)
def setup_registry():
    """Setup and cleanup node registry for each test."""
    # Register test node type
    NodeRegistry._node_types[NodeType.MODEL] = TestNode
    yield
    # Clean up
    NodeRegistry.clear()


@pytest.fixture
def sample_definition():
    """Create a sample node definition."""
    return NodeDefinition(
        id="test_node",
        type=NodeType.MODEL,
        name="Test Node",
        description="A test node",
        config={
            "timeout_ms": 5000,
            "model": "test-model",
            "temperature": 0.7,
        },
    )


@pytest.fixture
def sample_node(sample_definition):
    """Create a sample node instance."""
    return TestNode(sample_definition)


@pytest.fixture
def spawner():
    """Create a fresh NodeSpawner for each test."""
    return NodeSpawner(SpawnConfig(cooldown_ms=100))


class TestSpawnTrigger:
    """Tests for SpawnTrigger enum."""

    def test_all_triggers_exist(self):
        """All expected triggers should be defined."""
        assert SpawnTrigger.PERFORMANCE_THRESHOLD.value == "performance_threshold"
        assert SpawnTrigger.WORKLOAD_SPIKE.value == "workload_spike"
        assert SpawnTrigger.SPECIALIZATION_NEEDED.value == "specialization_needed"
        assert SpawnTrigger.MANUAL.value == "manual"


class TestSpawnConfig:
    """Tests for SpawnConfig model."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = SpawnConfig()

        assert config.max_spawns == 10
        assert config.cooldown_ms == 5000
        assert config.performance_threshold == 0.6
        assert config.workload_threshold == 100
        assert config.enable_auto_spawn is False
        assert config.max_spawn_depth == 3

    def test_custom_config(self):
        """Should accept custom values."""
        config = SpawnConfig(
            max_spawns=20,
            cooldown_ms=1000,
            performance_threshold=0.5,
            workload_threshold=50,
            enable_auto_spawn=True,
            max_spawn_depth=5,
        )

        assert config.max_spawns == 20
        assert config.cooldown_ms == 1000
        assert config.performance_threshold == 0.5
        assert config.workload_threshold == 50
        assert config.enable_auto_spawn is True
        assert config.max_spawn_depth == 5

    def test_validation_max_spawns(self):
        """Should validate max_spawns range."""
        with pytest.raises(ValueError):
            SpawnConfig(max_spawns=0)

        with pytest.raises(ValueError):
            SpawnConfig(max_spawns=101)

    def test_validation_cooldown(self):
        """Should validate cooldown range."""
        with pytest.raises(ValueError):
            SpawnConfig(cooldown_ms=50)

        with pytest.raises(ValueError):
            SpawnConfig(cooldown_ms=70000)

    def test_validation_performance_threshold(self):
        """Should validate performance threshold range."""
        with pytest.raises(ValueError):
            SpawnConfig(performance_threshold=-0.1)

        with pytest.raises(ValueError):
            SpawnConfig(performance_threshold=1.5)

    def test_validation_workload_threshold(self):
        """Should validate workload threshold range."""
        with pytest.raises(ValueError):
            SpawnConfig(workload_threshold=0)

        with pytest.raises(ValueError):
            SpawnConfig(workload_threshold=20000)

    def test_forbid_extra_fields(self):
        """Should forbid extra fields."""
        with pytest.raises(ValueError):
            SpawnConfig(extra_field="not_allowed")


class TestSpawnRecord:
    """Tests for SpawnRecord model."""

    def test_create_record(self):
        """Should create a spawn record."""
        record = SpawnRecord(
            spawned_node_id="node_1_spawn",
            template_node_id="node_1",
            trigger=SpawnTrigger.MANUAL,
            specialization={"model": "upgraded-model"},
            spawn_depth=0,
        )

        assert record.spawned_node_id == "node_1_spawn"
        assert record.template_node_id == "node_1"
        assert record.trigger == SpawnTrigger.MANUAL
        assert record.specialization["model"] == "upgraded-model"
        assert record.spawn_depth == 0
        assert isinstance(record.timestamp, datetime)
        assert record.id.startswith("spawn_")

    def test_default_values(self):
        """Should have sensible defaults."""
        record = SpawnRecord(
            spawned_node_id="test",
            template_node_id="template",
            trigger=SpawnTrigger.MANUAL,
        )

        assert record.specialization == {}
        assert record.spawn_depth == 0
        assert record.metadata == {}

    def test_forbid_extra_fields(self):
        """Should forbid extra fields."""
        with pytest.raises(ValueError):
            SpawnRecord(
                spawned_node_id="test",
                template_node_id="template",
                trigger=SpawnTrigger.MANUAL,
                invalid_field="value",
            )


class TestSpawnMetrics:
    """Tests for SpawnMetrics model."""

    def test_default_metrics(self):
        """Should have zero defaults."""
        metrics = SpawnMetrics()

        assert metrics.total_spawns == 0
        assert metrics.active_spawns == 0
        assert metrics.failed_spawns == 0
        assert metrics.avg_spawn_latency_ms == 0.0
        assert metrics.spawns_by_trigger == {}

    def test_custom_metrics(self):
        """Should accept custom values."""
        metrics = SpawnMetrics(
            total_spawns=10,
            active_spawns=5,
            failed_spawns=2,
            avg_spawn_latency_ms=150.5,
            spawns_by_trigger={"manual": 8, "performance_threshold": 2},
        )

        assert metrics.total_spawns == 10
        assert metrics.active_spawns == 5
        assert metrics.failed_spawns == 2
        assert metrics.avg_spawn_latency_ms == 150.5
        assert metrics.spawns_by_trigger["manual"] == 8


class TestNodeFactory:
    """Tests for NodeFactory."""

    def test_create_from_definition(self, sample_definition):
        """Should create node from definition."""
        node = NodeFactory.create_from_definition(sample_definition)

        assert isinstance(node, TestNode)
        assert node.id == "test_node"
        assert node.type == NodeType.MODEL
        assert node.name == "Test Node"

    def test_create_from_definition_invalid_type(self):
        """Should raise error for unregistered node type."""
        definition = NodeDefinition(
            id="invalid",
            type=NodeType.ROUTER,  # Not registered in our test
            name="Invalid",
        )

        with pytest.raises(ValueError, match="Unknown node type"):
            NodeFactory.create_from_definition(definition)

    def test_create_from_template(self, sample_definition):
        """Should create node from template with new ID."""
        node = NodeFactory.create_from_template(
            template=sample_definition,
            new_id="new_test_node",
        )

        assert isinstance(node, TestNode)
        assert node.id == "new_test_node"
        assert node.type == NodeType.MODEL
        assert node.name == "Test Node"

    def test_create_from_template_with_overrides(self, sample_definition):
        """Should apply config overrides."""
        node = NodeFactory.create_from_template(
            template=sample_definition,
            new_id="specialized_node",
            config_overrides={
                "model": "better-model",
                "temperature": 0.9,
            },
        )

        assert node.id == "specialized_node"
        assert node._raw_config["model"] == "better-model"
        assert node._raw_config["temperature"] == 0.9
        # Original config value should be preserved
        assert node._raw_config["timeout_ms"] == 5000

    def test_create_from_template_invalid_id(self, sample_definition):
        """Should reject invalid new_id."""
        with pytest.raises(ValueError, match="new_id must be a non-empty string"):
            NodeFactory.create_from_template(
                template=sample_definition,
                new_id="",
            )

        with pytest.raises(ValueError):
            NodeFactory.create_from_template(
                template=sample_definition,
                new_id=None,
            )

    def test_clone_node_definition(self, sample_node):
        """Should clone node definition."""
        definition = NodeFactory.clone_node_definition(
            node=sample_node,
            new_id="cloned_node",
        )

        assert isinstance(definition, NodeDefinition)
        assert definition.id == "cloned_node"
        assert definition.type == sample_node.type
        assert definition.name == sample_node.name
        assert definition.config == sample_node._raw_config

    def test_clone_node_definition_with_modifications(self, sample_node):
        """Should apply modifications when cloning."""
        definition = NodeFactory.clone_node_definition(
            node=sample_node,
            new_id="modified_clone",
            modifications={
                "model": "new-model",
                "new_param": "new_value",
            },
        )

        assert definition.id == "modified_clone"
        assert definition.config["model"] == "new-model"
        assert definition.config["new_param"] == "new_value"
        # Original values preserved
        assert definition.config["timeout_ms"] == 5000

    def test_clone_node_definition_invalid_node(self):
        """Should reject invalid node."""
        with pytest.raises(ValueError, match="node must be a BaseNode instance"):
            NodeFactory.clone_node_definition(
                node="not a node",
                new_id="clone",
            )

    def test_clone_node_definition_invalid_id(self, sample_node):
        """Should reject invalid new_id."""
        with pytest.raises(ValueError, match="new_id must be a non-empty string"):
            NodeFactory.clone_node_definition(
                node=sample_node,
                new_id="",
            )


class TestNodeSpawner:
    """Tests for NodeSpawner."""

    def test_init_default_config(self):
        """Should initialize with default config."""
        spawner = NodeSpawner()

        assert isinstance(spawner.config, SpawnConfig)
        assert spawner.config.max_spawns == 10
        assert spawner.config.cooldown_ms == 5000

    def test_init_custom_config(self):
        """Should initialize with custom config."""
        config = SpawnConfig(max_spawns=5, cooldown_ms=1000)
        spawner = NodeSpawner(config)

        assert spawner.config.max_spawns == 5
        assert spawner.config.cooldown_ms == 1000

    def test_can_spawn_initially(self):
        """Should allow spawning initially."""
        spawner = NodeSpawner()
        assert spawner.can_spawn() is True

    def test_can_spawn_max_limit(self, sample_definition):
        """Should enforce max spawn limit."""
        spawner = NodeSpawner(SpawnConfig(max_spawns=2, cooldown_ms=100))

        # Spawn up to max (disable cooldown check to allow rapid spawns)
        spawner.spawn_node(sample_definition)
        sleep(0.15)  # Wait for cooldown
        spawner.spawn_node(sample_definition)

        # Should not allow more
        assert spawner.can_spawn(check_cooldown=False) is False

    def test_can_spawn_cooldown(self, sample_definition):
        """Should enforce cooldown period."""
        spawner = NodeSpawner(SpawnConfig(cooldown_ms=200))

        # First spawn
        spawner.spawn_node(sample_definition)

        # Should be in cooldown
        assert spawner.can_spawn(check_cooldown=True) is False

        # Should work without cooldown check
        assert spawner.can_spawn(check_cooldown=False) is True

        # Wait for cooldown
        sleep(0.25)
        assert spawner.can_spawn(check_cooldown=True) is True

    def test_spawn_node_basic(self, sample_definition):
        """Should spawn a node from template."""
        spawner = NodeSpawner()

        node = spawner.spawn_node(sample_definition)

        assert isinstance(node, TestNode)
        assert node.id.startswith("test_node_spawn_")
        assert node.type == NodeType.MODEL
        assert node.name == "Test Node"

    def test_spawn_node_with_specialization(self, sample_definition):
        """Should apply specialization parameters."""
        spawner = NodeSpawner()

        node = spawner.spawn_node(
            template=sample_definition,
            specialization={"model": "specialized-model", "temperature": 0.95},
        )

        assert node._raw_config["model"] == "specialized-model"
        assert node._raw_config["temperature"] == 0.95

    def test_spawn_node_with_trigger(self, sample_definition):
        """Should record spawn trigger."""
        spawner = NodeSpawner()

        node = spawner.spawn_node(
            template=sample_definition,
            trigger=SpawnTrigger.PERFORMANCE_THRESHOLD,
        )

        record = spawner.get_spawn_record(node.id)
        assert record.trigger == SpawnTrigger.PERFORMANCE_THRESHOLD

    def test_spawn_node_records_history(self, sample_definition):
        """Should record spawn in history."""
        spawner = NodeSpawner()

        node = spawner.spawn_node(sample_definition)

        history = spawner.get_spawn_history()
        assert len(history) == 1
        assert history[0].spawned_node_id == node.id
        assert history[0].template_node_id == "test_node"

    def test_spawn_node_updates_metrics(self, sample_definition):
        """Should update metrics after spawn."""
        spawner = NodeSpawner()

        spawner.spawn_node(
            sample_definition,
            trigger=SpawnTrigger.WORKLOAD_SPIKE,
        )

        metrics = spawner.get_metrics()
        assert metrics.total_spawns == 1
        assert metrics.active_spawns == 1
        assert metrics.failed_spawns == 0
        assert metrics.spawns_by_trigger["workload_spike"] == 1

    def test_spawn_node_exceeds_limit(self, sample_definition):
        """Should raise error when spawn limit exceeded."""
        spawner = NodeSpawner(SpawnConfig(max_spawns=1))

        spawner.spawn_node(sample_definition)

        with pytest.raises(RuntimeError, match="Cannot spawn"):
            spawner.spawn_node(sample_definition)

    def test_spawn_node_exceeds_depth(self, sample_definition):
        """Should raise error when spawn depth exceeded."""
        spawner = NodeSpawner(SpawnConfig(max_spawn_depth=2))

        with pytest.raises(ValueError, match="Spawn depth .* exceeds max"):
            spawner.spawn_node(sample_definition, spawn_depth=3)

    def test_clone_node_basic(self, sample_node):
        """Should clone an existing node."""
        spawner = NodeSpawner()

        clone = spawner.clone_node(sample_node)

        assert isinstance(clone, TestNode)
        assert clone.id.startswith("test_node_clone_")
        assert clone.type == sample_node.type
        assert clone.name == sample_node.name

    def test_clone_node_with_modifications(self, sample_node):
        """Should apply modifications when cloning."""
        spawner = NodeSpawner()

        clone = spawner.clone_node(
            node=sample_node,
            modifications={"model": "cloned-model"},
        )

        assert clone._raw_config["model"] == "cloned-model"

    def test_clone_node_spawn_depth(self, sample_definition, spawner):
        """Should track spawn depth through clones."""
        # Spawn original
        node1 = spawner.spawn_node(sample_definition)
        record1 = spawner.get_spawn_record(node1.id)
        assert record1.spawn_depth == 0

        sleep(0.15)  # Wait for cooldown

        # Clone the spawn
        node2 = spawner.clone_node(node1)
        record2 = spawner.get_spawn_record(node2.id)
        assert record2.spawn_depth == 1

        sleep(0.15)  # Wait for cooldown

        # Clone the clone
        node3 = spawner.clone_node(node2)
        record3 = spawner.get_spawn_record(node3.id)
        assert record3.spawn_depth == 2

    def test_despawn_node(self, sample_definition):
        """Should remove node from active tracking."""
        spawner = NodeSpawner()

        node = spawner.spawn_node(sample_definition)
        assert len(spawner._active_spawns) == 1

        result = spawner.despawn_node(node.id)
        assert result is True
        assert len(spawner._active_spawns) == 0

    def test_despawn_node_not_found(self):
        """Should return False if node not found."""
        spawner = NodeSpawner()

        result = spawner.despawn_node("nonexistent")
        assert result is False

    def test_get_spawn_history_unfiltered(self, sample_definition, spawner):
        """Should return all spawn history."""
        spawner.spawn_node(sample_definition, trigger=SpawnTrigger.MANUAL)
        sleep(0.15)
        spawner.spawn_node(sample_definition, trigger=SpawnTrigger.WORKLOAD_SPIKE)

        history = spawner.get_spawn_history()
        assert len(history) == 2

    def test_get_spawn_history_filter_by_node(self, sample_definition, spawner):
        """Should filter history by node ID."""
        node1 = spawner.spawn_node(sample_definition)
        sleep(0.15)
        spawner.spawn_node(sample_definition)

        history = spawner.get_spawn_history(node_id=node1.id)
        assert len(history) == 1
        assert history[0].spawned_node_id == node1.id

    def test_get_spawn_history_filter_by_trigger(self, sample_definition, spawner):
        """Should filter history by trigger."""
        spawner.spawn_node(sample_definition, trigger=SpawnTrigger.MANUAL)
        sleep(0.15)
        spawner.spawn_node(sample_definition, trigger=SpawnTrigger.WORKLOAD_SPIKE)
        sleep(0.15)
        spawner.spawn_node(sample_definition, trigger=SpawnTrigger.MANUAL)

        history = spawner.get_spawn_history(trigger=SpawnTrigger.MANUAL)
        assert len(history) == 2
        assert all(r.trigger == SpawnTrigger.MANUAL for r in history)

    def test_get_spawn_history_with_limit(self, sample_definition, spawner):
        """Should limit number of results."""
        for i in range(5):
            spawner.spawn_node(sample_definition)
            if i < 4:  # Don't sleep after last spawn
                sleep(0.15)

        history = spawner.get_spawn_history(limit=3)
        assert len(history) == 3

    def test_get_spawn_history_ordered(self, sample_definition, spawner):
        """Should return history most recent first."""
        node1 = spawner.spawn_node(sample_definition)
        sleep(0.15)  # Wait for cooldown and ensure different timestamps
        node2 = spawner.spawn_node(sample_definition)

        history = spawner.get_spawn_history()
        assert history[0].spawned_node_id == node2.id
        assert history[1].spawned_node_id == node1.id

    def test_get_spawn_record(self, sample_definition):
        """Should get spawn record for specific node."""
        spawner = NodeSpawner()

        node = spawner.spawn_node(sample_definition)
        record = spawner.get_spawn_record(node.id)

        assert record is not None
        assert record.spawned_node_id == node.id

    def test_get_spawn_record_not_found(self):
        """Should return None if record not found."""
        spawner = NodeSpawner()

        record = spawner.get_spawn_record("nonexistent")
        assert record is None

    def test_get_children(self, sample_definition, spawner):
        """Should get all spawns from a template."""
        spawner.spawn_node(sample_definition)
        sleep(0.15)
        spawner.spawn_node(sample_definition)

        children = spawner.get_children("test_node")
        assert len(children) == 2
        assert all(c.template_node_id == "test_node" for c in children)

    def test_get_metrics(self, sample_definition, spawner):
        """Should return current metrics."""
        spawner.spawn_node(sample_definition, trigger=SpawnTrigger.MANUAL)
        sleep(0.15)
        spawner.spawn_node(sample_definition, trigger=SpawnTrigger.MANUAL)

        metrics = spawner.get_metrics()
        assert metrics.total_spawns == 2
        assert metrics.active_spawns == 2
        assert metrics.failed_spawns == 0
        assert metrics.avg_spawn_latency_ms > 0
        assert metrics.spawns_by_trigger["manual"] == 2

    def test_check_performance_trigger_disabled(self):
        """Should not trigger when auto spawn disabled."""
        spawner = NodeSpawner(SpawnConfig(enable_auto_spawn=False))

        result = spawner.check_performance_trigger("node_1", 0.3)
        assert result is False

    def test_check_performance_trigger_below_threshold(self):
        """Should trigger when performance below threshold."""
        spawner = NodeSpawner(
            SpawnConfig(enable_auto_spawn=True, performance_threshold=0.6)
        )

        result = spawner.check_performance_trigger("node_1", 0.5)
        assert result is True

    def test_check_performance_trigger_above_threshold(self):
        """Should not trigger when performance above threshold."""
        spawner = NodeSpawner(
            SpawnConfig(enable_auto_spawn=True, performance_threshold=0.6)
        )

        result = spawner.check_performance_trigger("node_1", 0.8)
        assert result is False

    def test_check_workload_trigger_disabled(self):
        """Should not trigger when auto spawn disabled."""
        spawner = NodeSpawner(SpawnConfig(enable_auto_spawn=False))

        result = spawner.check_workload_trigger("node_1", 200)
        assert result is False

    def test_check_workload_trigger_above_threshold(self):
        """Should trigger when workload above threshold."""
        spawner = NodeSpawner(
            SpawnConfig(enable_auto_spawn=True, workload_threshold=100)
        )

        result = spawner.check_workload_trigger("node_1", 150)
        assert result is True

    def test_check_workload_trigger_below_threshold(self):
        """Should not trigger when workload below threshold."""
        spawner = NodeSpawner(
            SpawnConfig(enable_auto_spawn=True, workload_threshold=100)
        )

        result = spawner.check_workload_trigger("node_1", 50)
        assert result is False

    def test_reset(self, sample_definition, spawner):
        """Should reset spawner state."""
        # Create some state
        spawner.spawn_node(sample_definition)
        sleep(0.15)
        spawner.spawn_node(sample_definition)

        assert len(spawner._spawn_history) == 2
        assert len(spawner._active_spawns) == 2

        spawner.reset()

        assert len(spawner._spawn_history) == 0
        assert len(spawner._active_spawns) == 0
        assert spawner._last_spawn_time is None
        assert spawner.get_metrics().total_spawns == 0

    def test_get_stats(self, sample_definition, spawner):
        """Should return detailed statistics."""
        spawner.spawn_node(sample_definition)

        stats = spawner.get_stats()

        assert "config" in stats
        assert stats["active_spawns"] == 1
        assert stats["total_spawns"] == 1
        assert stats["last_spawn_time"] is not None
        assert "metrics" in stats
        assert stats["spawn_history_size"] == 1

    def test_multiple_spawns_average_latency(self, sample_definition, spawner):
        """Should correctly calculate average spawn latency."""
        # Spawn multiple nodes
        for i in range(5):
            spawner.spawn_node(sample_definition)
            if i < 4:
                sleep(0.15)

        metrics = spawner.get_metrics()
        assert metrics.avg_spawn_latency_ms > 0
        assert metrics.total_spawns == 5

    def test_spawn_node_trigger_counts(self, sample_definition, spawner):
        """Should track spawns by trigger type."""
        spawner.spawn_node(sample_definition, trigger=SpawnTrigger.MANUAL)
        sleep(0.15)
        spawner.spawn_node(sample_definition, trigger=SpawnTrigger.MANUAL)
        sleep(0.15)
        spawner.spawn_node(sample_definition, trigger=SpawnTrigger.WORKLOAD_SPIKE)
        sleep(0.15)
        spawner.spawn_node(
            sample_definition, trigger=SpawnTrigger.PERFORMANCE_THRESHOLD
        )

        metrics = spawner.get_metrics()
        assert metrics.spawns_by_trigger["manual"] == 2
        assert metrics.spawns_by_trigger["workload_spike"] == 1
        assert metrics.spawns_by_trigger["performance_threshold"] == 1

    def test_spawn_preserves_template_metadata(self, sample_definition):
        """Should preserve template metadata in spawn record."""
        spawner = NodeSpawner()

        node = spawner.spawn_node(sample_definition)
        record = spawner.get_spawn_record(node.id)

        assert record.metadata["template_type"] == "model"
        assert record.metadata["template_name"] == "Test Node"

    def test_concurrent_spawn_limit_enforcement(self, sample_definition):
        """Should enforce limit across concurrent spawns."""
        spawner = NodeSpawner(SpawnConfig(max_spawns=3, cooldown_ms=100))

        # Spawn to limit
        spawner.spawn_node(sample_definition)
        sleep(0.15)
        spawner.spawn_node(sample_definition)
        sleep(0.15)
        spawner.spawn_node(sample_definition)

        # Should reject further spawns
        sleep(0.15)
        with pytest.raises(RuntimeError, match="Cannot spawn"):
            spawner.spawn_node(sample_definition)

        # Despawn one
        history = spawner.get_spawn_history()
        spawner.despawn_node(history[0].spawned_node_id)

        # Should now allow spawn
        sleep(0.15)
        spawner.spawn_node(sample_definition)
        assert spawner.get_metrics().active_spawns == 3
