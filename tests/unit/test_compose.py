"""Tests for workflow composition helpers."""

import pytest

from tinyllm.config.graph import NodeType
from tinyllm.core.compose import (
    GateCondition,
    Route,
    TransformOp,
    WorkflowBuilder,
    parallel_consensus_workflow,
    routed_specialist_workflow,
    simple_qa_workflow,
    transform_pipeline_workflow,
)


class TestWorkflowBuilder:
    """Tests for the WorkflowBuilder class."""

    def test_create_simple_workflow(self):
        """Test creating a simple entry→exit workflow."""
        builder = (
            WorkflowBuilder("test.simple", "Test Workflow")
            .add_entry("entry.main")
            .add_exit("exit.main")
            .connect("entry.main", "exit.main")
        )

        definition = builder.build_definition()
        workflow = builder.build()

        assert definition.id == "test.simple"
        assert len(definition.nodes) == 2
        assert len(definition.edges) == 1
        assert workflow.id == "test.simple"

    def test_create_workflow_with_model(self):
        """Test workflow with model node."""
        builder = (
            WorkflowBuilder("test.model", "Model Workflow")
            .add_entry("entry.main")
            .add_model("model.qa", model="qwen2.5:3b", temperature=0.5)
            .add_exit("exit.main")
            .connect("entry.main", "model.qa")
            .connect("model.qa", "exit.main")
        )

        definition = builder.build_definition()
        workflow = builder.build()

        assert len(definition.nodes) == 3
        model_node = definition.get_node("model.qa")
        assert model_node.type == NodeType.MODEL
        assert model_node.config["model"] == "qwen2.5:3b"
        assert model_node.config["temperature"] == 0.5
        assert workflow.get_node("model.qa") is not None

    def test_create_workflow_with_router(self):
        """Test workflow with router node."""
        routes = [
            Route("code", "Code tasks", "model.code"),
            Route("general", "General tasks", "model.general"),
        ]

        builder = (
            WorkflowBuilder("test.router", "Router Workflow")
            .add_entry("entry.main")
            .add_router("router.main", routes=routes, default_route="general")
            .add_model("model.code", model="qwen2.5:3b")
            .add_model("model.general", model="qwen2.5:0.5b")
            .add_exit("exit.main")
            .connect("entry.main", "router.main")
            .connect("router.main", ["model.code", "model.general"])
            .connect(["model.code", "model.general"], "exit.main")
        )

        definition = builder.build_definition()
        workflow = builder.build()

        assert len(definition.nodes) == 5
        router_node = definition.get_node("router.main")
        assert router_node.type == NodeType.ROUTER
        assert len(router_node.config["routes"]) == 2
        assert workflow.get_node("router.main") is not None

    def test_create_workflow_with_gate(self):
        """Test workflow with gate node."""
        conditions = [
            GateCondition("has_code", "'def ' in content", "exit.success"),
            GateCondition("has_function", "'function' in content", "exit.success"),
        ]

        builder = (
            WorkflowBuilder("test.gate", "Gate Workflow")
            .add_entry("entry.main")
            .add_gate(
                "gate.quality",
                conditions=conditions,
                default_target="exit.fallback",
            )
            .add_exit("exit.success")
            .add_exit("exit.fallback", status="fallback")
            .connect("entry.main", "gate.quality")
            .connect("gate.quality", "exit.success")
            .connect("gate.quality", "exit.fallback")
        )

        definition = builder.build_definition()
        gate_node = definition.get_node("gate.quality")
        assert gate_node.type == NodeType.GATE
        assert len(gate_node.config["conditions"]) == 2
        assert gate_node.config["default_target"] == "exit.fallback"

    def test_create_workflow_with_transform(self):
        """Test workflow with transform node."""
        transforms = [
            TransformOp("strip"),
            TransformOp("uppercase"),
            TransformOp("truncate", {"max_length": 100}),
        ]

        builder = (
            WorkflowBuilder("test.transform", "Transform Workflow")
            .add_entry("entry.main")
            .add_transform("transform.pipeline", transforms=transforms)
            .add_exit("exit.main")
            .connect("entry.main", "transform.pipeline")
            .connect("transform.pipeline", "exit.main")
        )

        definition = builder.build_definition()
        transform_node = definition.get_node("transform.pipeline")
        assert transform_node.type == NodeType.TRANSFORM
        assert len(transform_node.config["transforms"]) == 3

    def test_create_workflow_with_loop(self):
        """Test workflow with loop node."""
        builder = (
            WorkflowBuilder("test.loop", "Loop Workflow")
            .add_entry("entry.main")
            .add_loop(
                "loop.main",
                body_node="transform.body",
                condition_type="fixed_count",
                fixed_count=5,
            )
            .add_transform("transform.body", transforms=[TransformOp("uppercase")])
            .add_exit("exit.main")
            .connect("entry.main", "loop.main")
            .connect("loop.main", "exit.main")
        )

        definition = builder.build_definition()
        loop_node = definition.get_node("loop.main")
        assert loop_node.type == NodeType.LOOP
        assert loop_node.config["fixed_count"] == 5
        assert loop_node.config["body_node"] == "transform.body"

    def test_create_workflow_with_fanout(self):
        """Test workflow with fanout node."""
        builder = (
            WorkflowBuilder("test.fanout", "Fanout Workflow")
            .add_entry("entry.main")
            .add_fanout(
                "fanout.parallel",
                target_nodes=["model.a", "model.b", "model.c"],
                aggregation_strategy="majority_vote",
            )
            .add_model("model.a")
            .add_model("model.b")
            .add_model("model.c")
            .add_exit("exit.main")
            .connect("entry.main", "fanout.parallel")
            .connect("fanout.parallel", "exit.main")
        )

        definition = builder.build_definition()
        fanout_node = definition.get_node("fanout.parallel")
        assert fanout_node.type == NodeType.FANOUT
        assert fanout_node.config["aggregation_strategy"] == "majority_vote"
        assert len(fanout_node.config["target_nodes"]) == 3

    def test_many_to_many_connections(self):
        """Test connecting multiple nodes at once."""
        builder = (
            WorkflowBuilder("test.multi", "Multi Workflow")
            .add_entry("entry.main")
            .add_model("model.a")
            .add_model("model.b")
            .add_exit("exit.main")
            .connect("entry.main", ["model.a", "model.b"])
            .connect(["model.a", "model.b"], "exit.main")
        )

        definition = builder.build_definition()

        # Should have 4 edges: entry→a, entry→b, a→exit, b→exit
        assert len(definition.edges) == 4

    def test_build_definition_only(self):
        """Test building just the definition without node instantiation."""
        builder = (
            WorkflowBuilder("test.def", "Definition Test")
            .add_entry("entry.main")
            .add_exit("exit.main")
            .connect("entry.main", "exit.main")
        )

        definition = builder.build_definition()

        assert definition.id == "test.def"
        assert len(definition.nodes) == 2
        assert len(definition.entry_points) == 1
        assert len(definition.exit_points) == 1

    def test_with_metadata(self):
        """Test adding metadata to workflow."""
        builder = (
            WorkflowBuilder("test.meta", "Metadata Test")
            .add_entry("entry.main")
            .add_exit("exit.main")
            .connect("entry.main", "exit.main")
            .with_metadata(
                author="Test Author",
                tags=["test", "example"],
                description="A test workflow",
            )
        )

        definition = builder.build_definition()

        assert definition.metadata.author == "Test Author"
        assert definition.metadata.tags == ["test", "example"]
        assert definition.metadata.description == "A test workflow"

    def test_protect_nodes(self):
        """Test marking nodes as protected."""
        builder = (
            WorkflowBuilder("test.protect", "Protected Test")
            .add_entry("entry.main")
            .add_model("model.important")
            .add_exit("exit.main")
            .connect("entry.main", "model.important")
            .connect("model.important", "exit.main")
            .protect("model.important", "entry.main")
        )

        definition = builder.build_definition()

        assert "model.important" in definition.protected
        assert "entry.main" in definition.protected

    def test_custom_version(self):
        """Test setting custom version."""
        builder = WorkflowBuilder(
            "test.version", "Version Test", version="2.1.0"
        )
        builder.add_entry("entry.main")
        builder.add_exit("exit.main")
        builder.connect("entry.main", "exit.main")

        definition = builder.build_definition()

        assert definition.version == "2.1.0"


class TestConvenienceFunctions:
    """Tests for convenience workflow creation functions."""

    def test_simple_qa_workflow(self):
        """Test simple QA workflow creation."""
        workflow = simple_qa_workflow()

        assert workflow.id == "simple.qa"
        assert workflow.has_node("entry.main")
        assert workflow.has_node("model.qa")
        assert workflow.has_node("exit.main")

    def test_simple_qa_workflow_custom_model(self):
        """Test simple QA with custom model."""
        workflow = simple_qa_workflow(model="qwen2.5:7b")

        model_node = workflow.get_node("model.qa")
        assert model_node is not None

    def test_routed_specialist_workflow(self):
        """Test routed specialist workflow creation."""
        workflow = routed_specialist_workflow()

        assert workflow.id == "routed.specialist"
        assert workflow.has_node("router.main")
        assert workflow.has_node("model.code")
        assert workflow.has_node("model.math")
        assert workflow.has_node("model.general")

    def test_routed_specialist_custom_routes(self):
        """Test routed specialist with custom routes."""
        custom_routes = [
            Route("creative", "Creative writing", "model.creative"),
            Route("technical", "Technical docs", "model.technical"),
        ]

        workflow = routed_specialist_workflow(routes=custom_routes)

        assert workflow.has_node("model.creative")
        assert workflow.has_node("model.technical")
        assert not workflow.has_node("model.code")

    def test_parallel_consensus_workflow(self):
        """Test parallel consensus workflow creation."""
        workflow = parallel_consensus_workflow(num_models=5)

        assert workflow.id == "parallel.consensus"
        assert workflow.has_node("fanout.parallel")
        for i in range(5):
            assert workflow.has_node(f"model.expert_{i}")

    def test_transform_pipeline_workflow(self):
        """Test transform pipeline workflow creation."""
        workflow = transform_pipeline_workflow()

        assert workflow.id == "transform.pipeline"
        assert workflow.has_node("transform.pipeline")

    def test_transform_pipeline_custom_ops(self):
        """Test transform pipeline with custom operations."""
        custom_transforms = [
            TransformOp("uppercase"),
            TransformOp("regex_replace", {"pattern": r"\s+", "replacement": "_"}),
        ]

        workflow = transform_pipeline_workflow(transforms=custom_transforms)

        assert workflow.has_node("transform.pipeline")


class TestWorkflowValidation:
    """Tests for workflow validation."""

    def test_valid_workflow_passes(self):
        """Test that valid workflows pass validation."""
        workflow = (
            WorkflowBuilder("test.valid", "Valid Workflow")
            .add_entry("entry.main")
            .add_model("model.qa")
            .add_exit("exit.main")
            .connect("entry.main", "model.qa")
            .connect("model.qa", "exit.main")
            .build()
        )

        errors = workflow.validate()
        error_msgs = [e for e in errors if e.severity == "error"]
        assert len(error_msgs) == 0

    def test_workflow_with_system_prompt(self):
        """Test model node with system prompt."""
        builder = (
            WorkflowBuilder("test.prompt", "Prompt Workflow")
            .add_entry("entry.main")
            .add_model(
                "model.qa",
                system_prompt="You are a helpful assistant.",
            )
            .add_exit("exit.main")
            .connect("entry.main", "model.qa")
            .connect("model.qa", "exit.main")
        )

        definition = builder.build_definition()
        model_node = definition.get_node("model.qa")
        assert model_node.config["system_prompt"] == "You are a helpful assistant."

    def test_entry_with_required_fields(self):
        """Test entry node with required fields."""
        builder = (
            WorkflowBuilder("test.required", "Required Workflow")
            .add_entry("entry.main", required_fields=["content", "user_id"])
            .add_exit("exit.main")
            .connect("entry.main", "exit.main")
        )

        definition = builder.build_definition()
        entry_node = definition.get_node("entry.main")
        assert entry_node.config["required_fields"] == ["content", "user_id"]
