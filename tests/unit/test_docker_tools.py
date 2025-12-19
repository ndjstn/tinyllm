"""Tests for Docker tools."""

import json
from unittest.mock import MagicMock, patch

import pytest

from tinyllm.tools.docker import (
    # Enums
    ContainerState,
    # Config and dataclasses
    DockerConfig,
    DockerClient,
    DockerManager,
    DockerContainer,
    DockerImage,
    DockerVolume,
    DockerNetwork,
    DockerResult,
    # Input models
    CreateDockerClientInput,
    ListContainersInput,
    GetContainerInput,
    StartContainerInput,
    StopContainerInput,
    RemoveContainerInput,
    GetContainerLogsInput,
    ListImagesInput,
    GetImageInput,
    PullImageInput,
    RemoveImageInput,
    ListVolumesInput,
    CreateVolumeInput,
    RemoveVolumeInput,
    ListNetworksInput,
    GetNetworkInput,
    # Output models
    CreateDockerClientOutput,
    DockerContainersOutput,
    DockerLogsOutput,
    DockerSimpleOutput,
    DockerImagesOutput,
    DockerVolumesOutput,
    DockerNetworksOutput,
    # Tools
    CreateDockerClientTool,
    ListContainersTool,
    GetContainerTool,
    StartContainerTool,
    StopContainerTool,
    RemoveContainerTool,
    GetContainerLogsTool,
    ListImagesTool,
    GetImageTool,
    PullImageTool,
    RemoveImageTool,
    ListVolumesTool,
    CreateVolumeTool,
    RemoveVolumeTool,
    ListNetworksTool,
    GetNetworkTool,
    # Helper functions
    create_docker_config,
    create_docker_client,
    create_docker_manager,
    create_docker_tools,
)


# =============================================================================
# ContainerState Tests
# =============================================================================


class TestContainerState:
    """Tests for ContainerState enum."""

    def test_container_state_values(self):
        """Test container state enum values."""
        assert ContainerState.CREATED == "created"
        assert ContainerState.RUNNING == "running"
        assert ContainerState.PAUSED == "paused"
        assert ContainerState.RESTARTING == "restarting"
        assert ContainerState.REMOVING == "removing"
        assert ContainerState.EXITED == "exited"
        assert ContainerState.DEAD == "dead"

    def test_container_state_count(self):
        """Test all container states are defined."""
        assert len(ContainerState) == 7


# =============================================================================
# DockerConfig Tests
# =============================================================================


class TestDockerConfig:
    """Tests for DockerConfig."""

    def test_config_defaults(self):
        """Test config with default values."""
        config = DockerConfig()
        assert config.host == "unix:///var/run/docker.sock"
        assert config.timeout == 30
        assert config.api_version == "v1.43"
        assert config.tls is False

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = DockerConfig(
            host="tcp://localhost:2375",
            timeout=60,
            api_version="v1.44",
            tls=True,
            cert_path="/path/to/cert.pem",
            key_path="/path/to/key.pem",
            ca_path="/path/to/ca.pem",
        )
        assert config.host == "tcp://localhost:2375"
        assert config.timeout == 60
        assert config.api_version == "v1.44"
        assert config.tls is True


# =============================================================================
# DockerContainer Tests
# =============================================================================


class TestDockerContainer:
    """Tests for DockerContainer dataclass."""

    def test_container_from_api_response(self):
        """Test creating container from API response."""
        api_response = {
            "Id": "abc123def456789",
            "Names": ["/my-container"],
            "Image": "nginx:latest",
            "Status": "Up 2 hours",
            "State": "running",
            "Created": 1704067200,
            "Ports": [
                {"PrivatePort": 80, "PublicPort": 8080, "Type": "tcp", "IP": "0.0.0.0"},
            ],
            "Labels": {"app": "web"},
        }

        container = DockerContainer.from_api_response(api_response)
        assert container.id == "abc123def456"
        assert container.name == "my-container"
        assert container.image == "nginx:latest"
        assert container.status == "Up 2 hours"
        assert container.state == "running"
        assert len(container.ports) == 1
        assert container.ports[0]["public_port"] == 8080
        assert container.labels == {"app": "web"}

    def test_container_from_minimal_response(self):
        """Test creating container from minimal API response."""
        api_response = {
            "Id": "abc123",
            "Names": [],
            "Image": "",
            "Status": "",
            "State": "",
            "Created": 0,
        }

        container = DockerContainer.from_api_response(api_response)
        assert container.id == "abc123"
        assert container.name == ""
        assert container.ports == []

    def test_container_to_dict(self):
        """Test converting container to dictionary."""
        container = DockerContainer(
            id="abc123",
            name="test-container",
            image="alpine:latest",
            status="Up 1 hour",
            state="running",
            created="2024-01-01T00:00:00Z",
            ports=[{"private_port": 80}],
            labels={"env": "test"},
        )

        result = container.to_dict()
        assert result["id"] == "abc123"
        assert result["name"] == "test-container"
        assert result["image"] == "alpine:latest"
        assert result["state"] == "running"


# =============================================================================
# DockerImage Tests
# =============================================================================


class TestDockerImage:
    """Tests for DockerImage dataclass."""

    def test_image_from_api_response(self):
        """Test creating image from API response."""
        api_response = {
            "Id": "sha256:abc123def456789",
            "RepoTags": ["nginx:latest", "nginx:1.24"],
            "Size": 187654321,
            "Created": 1704067200,
            "Labels": {"maintainer": "nginx"},
        }

        image = DockerImage.from_api_response(api_response)
        assert image.id == "abc123def456"
        assert image.repo_tags == ["nginx:latest", "nginx:1.24"]
        assert image.size == 187654321
        assert image.labels == {"maintainer": "nginx"}

    def test_image_from_minimal_response(self):
        """Test creating image from minimal API response."""
        api_response = {
            "Id": "sha256:abc",
            "RepoTags": None,
            "Size": 0,
            "Labels": None,
        }

        image = DockerImage.from_api_response(api_response)
        assert image.id == "abc"
        assert image.repo_tags == []
        assert image.labels == {}

    def test_image_to_dict(self):
        """Test converting image to dictionary."""
        image = DockerImage(
            id="abc123",
            repo_tags=["alpine:latest"],
            size=5242880,  # 5MB
            created="2024-01-01",
            labels={"version": "3.18"},
        )

        result = image.to_dict()
        assert result["id"] == "abc123"
        assert result["repo_tags"] == ["alpine:latest"]
        assert result["size"] == 5242880
        assert result["size_mb"] == 5.0


# =============================================================================
# DockerVolume Tests
# =============================================================================


class TestDockerVolume:
    """Tests for DockerVolume dataclass."""

    def test_volume_from_api_response(self):
        """Test creating volume from API response."""
        api_response = {
            "Name": "my-volume",
            "Driver": "local",
            "Mountpoint": "/var/lib/docker/volumes/my-volume/_data",
            "CreatedAt": "2024-01-01T00:00:00Z",
            "Labels": {"app": "database"},
            "Scope": "local",
        }

        volume = DockerVolume.from_api_response(api_response)
        assert volume.name == "my-volume"
        assert volume.driver == "local"
        assert volume.mountpoint == "/var/lib/docker/volumes/my-volume/_data"
        assert volume.labels == {"app": "database"}

    def test_volume_to_dict(self):
        """Test converting volume to dictionary."""
        volume = DockerVolume(
            name="test-volume",
            driver="local",
            mountpoint="/var/lib/docker/volumes/test-volume/_data",
            created="2024-01-01",
        )

        result = volume.to_dict()
        assert result["name"] == "test-volume"
        assert result["driver"] == "local"


# =============================================================================
# DockerNetwork Tests
# =============================================================================


class TestDockerNetwork:
    """Tests for DockerNetwork dataclass."""

    def test_network_from_api_response(self):
        """Test creating network from API response."""
        api_response = {
            "Id": "abc123def456789",
            "Name": "my-network",
            "Driver": "bridge",
            "Scope": "local",
            "Internal": False,
            "IPAM": {
                "Config": [
                    {"Subnet": "172.18.0.0/16", "Gateway": "172.18.0.1"},
                ],
            },
            "Labels": {"purpose": "app"},
        }

        network = DockerNetwork.from_api_response(api_response)
        assert network.id == "abc123def456"
        assert network.name == "my-network"
        assert network.driver == "bridge"
        assert network.scope == "local"
        assert network.internal is False
        assert len(network.ipam_config) == 1

    def test_network_to_dict(self):
        """Test converting network to dictionary."""
        network = DockerNetwork(
            id="abc123",
            name="test-network",
            driver="bridge",
            scope="local",
            internal=True,
        )

        result = network.to_dict()
        assert result["id"] == "abc123"
        assert result["name"] == "test-network"
        assert result["internal"] is True


# =============================================================================
# DockerResult Tests
# =============================================================================


class TestDockerResult:
    """Tests for DockerResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = DockerResult(
            success=True,
            data={"containers": []},
            status_code=200,
        )
        assert result.success is True
        assert result.data == {"containers": []}
        assert result.error is None

    def test_error_result(self):
        """Test error result."""
        result = DockerResult(
            success=False,
            error="Container not found",
            status_code=404,
        )
        assert result.success is False
        assert result.error == "Container not found"

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = DockerResult(success=True, data={"key": "value"}, status_code=200)
        result_dict = result.to_dict()
        assert result_dict["success"] is True
        assert result_dict["data"] == {"key": "value"}


# =============================================================================
# DockerClient Tests
# =============================================================================


class TestDockerClient:
    """Tests for DockerClient."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return DockerConfig(host="tcp://localhost:2375")

    @pytest.fixture
    def client(self, config):
        """Create test client."""
        return DockerClient(config)

    def test_client_initialization(self, client, config):
        """Test client initialization."""
        assert client.config == config
        assert client._is_unix_socket is False

    def test_unix_socket_client(self):
        """Test Unix socket client."""
        config = DockerConfig(host="unix:///var/run/docker.sock")
        client = DockerClient(config)
        assert client._is_unix_socket is True
        assert client._socket_path == "/var/run/docker.sock"


class TestDockerClientOperations:
    """Tests for DockerClient operations with mocked HTTP."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return DockerConfig(host="tcp://localhost:2375")

    @pytest.fixture
    def client(self, config):
        """Create test client."""
        return DockerClient(config)

    def _mock_response(self, data, status=200):
        """Create mock HTTP response."""
        response = MagicMock()
        response.read.return_value = json.dumps(data).encode()
        response.getcode.return_value = status
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)
        return response

    def test_list_containers(self, client):
        """Test listing containers."""
        api_response = [
            {
                "Id": "abc123def456789",
                "Names": ["/container1"],
                "Image": "nginx",
                "Status": "Up 1 hour",
                "State": "running",
                "Created": 1704067200,
                "Ports": [],
                "Labels": {},
            },
        ]

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(api_response)

            result = client.list_containers()

            assert result.success is True
            assert len(result.data["containers"]) == 1
            assert result.data["containers"][0]["name"] == "container1"

    def test_list_containers_all(self, client):
        """Test listing all containers including stopped."""
        api_response = []

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(api_response)

            result = client.list_containers(all_containers=True)

            assert result.success is True
            # Verify URL contains 'all=true'
            call_args = mock_urlopen.call_args
            request = call_args[0][0]
            assert "all=true" in request.full_url

    def test_get_container(self, client):
        """Test getting a container."""
        api_response = {
            "Id": "abc123def456789",
            "Name": "/my-container",
            "Config": {"Image": "nginx", "Labels": {}},
            "State": {"Status": "running"},
            "Created": "2024-01-01T00:00:00Z",
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(api_response)

            result = client.get_container("abc123")

            assert result.success is True
            assert result.data["container"]["name"] == "my-container"

    def test_start_container(self, client):
        """Test starting a container."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(None, 204)

            result = client.start_container("abc123")

            assert result.success is True

    def test_stop_container(self, client):
        """Test stopping a container."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(None, 204)

            result = client.stop_container("abc123", timeout=30)

            assert result.success is True

    def test_remove_container(self, client):
        """Test removing a container."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(None, 204)

            result = client.remove_container("abc123")

            assert result.success is True

    def test_get_container_logs(self, client):
        """Test getting container logs."""
        log_content = "2024-01-01 INFO: Application started"

        with patch("urllib.request.urlopen") as mock_urlopen:
            response = MagicMock()
            response.read.return_value = json.dumps(log_content).encode()
            response.getcode.return_value = 200
            response.__enter__ = MagicMock(return_value=response)
            response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = response

            result = client.get_container_logs("abc123", tail=50)

            assert result.success is True

    def test_list_images(self, client):
        """Test listing images."""
        api_response = [
            {
                "Id": "sha256:abc123",
                "RepoTags": ["nginx:latest"],
                "Size": 100000000,
                "Created": 1704067200,
                "Labels": {},
            },
        ]

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(api_response)

            result = client.list_images()

            assert result.success is True
            assert len(result.data["images"]) == 1

    def test_get_image(self, client):
        """Test getting an image."""
        api_response = {
            "Id": "sha256:abc123",
            "RepoTags": ["nginx:latest"],
            "Size": 100000000,
            "Created": 1704067200,
            "Labels": {},
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(api_response)

            result = client.get_image("nginx:latest")

            assert result.success is True
            assert result.data["image"]["repo_tags"] == ["nginx:latest"]

    def test_pull_image(self, client):
        """Test pulling an image."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response({}, 200)

            result = client.pull_image("nginx", tag="latest")

            assert result.success is True

    def test_remove_image(self, client):
        """Test removing an image."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response([], 200)

            result = client.remove_image("nginx:latest")

            assert result.success is True

    def test_list_volumes(self, client):
        """Test listing volumes."""
        api_response = {
            "Volumes": [
                {
                    "Name": "my-volume",
                    "Driver": "local",
                    "Mountpoint": "/var/lib/docker/volumes/my-volume/_data",
                },
            ],
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(api_response)

            result = client.list_volumes()

            assert result.success is True
            assert len(result.data["volumes"]) == 1
            assert result.data["volumes"][0]["name"] == "my-volume"

    def test_create_volume(self, client):
        """Test creating a volume."""
        api_response = {
            "Name": "new-volume",
            "Driver": "local",
            "Mountpoint": "/var/lib/docker/volumes/new-volume/_data",
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(api_response)

            result = client.create_volume("new-volume")

            assert result.success is True
            assert result.data["volume"]["name"] == "new-volume"

    def test_remove_volume(self, client):
        """Test removing a volume."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(None, 204)

            result = client.remove_volume("my-volume")

            assert result.success is True

    def test_list_networks(self, client):
        """Test listing networks."""
        api_response = [
            {
                "Id": "abc123",
                "Name": "bridge",
                "Driver": "bridge",
                "Scope": "local",
                "IPAM": {"Config": []},
            },
        ]

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(api_response)

            result = client.list_networks()

            assert result.success is True
            assert len(result.data["networks"]) == 1
            assert result.data["networks"][0]["name"] == "bridge"

    def test_get_network(self, client):
        """Test getting a network."""
        api_response = {
            "Id": "abc123",
            "Name": "my-network",
            "Driver": "bridge",
            "Scope": "local",
            "IPAM": {"Config": []},
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(api_response)

            result = client.get_network("my-network")

            assert result.success is True
            assert result.data["network"]["name"] == "my-network"


# =============================================================================
# DockerManager Tests
# =============================================================================


class TestDockerManager:
    """Tests for DockerManager."""

    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = DockerManager()
        assert manager._clients == {}

    def test_add_client(self):
        """Test adding a client."""
        manager = DockerManager()
        config = DockerConfig()
        client = DockerClient(config)

        manager.add_client("local", client)

        assert "local" in manager._clients
        assert manager.get_client("local") == client

    def test_get_nonexistent_client(self):
        """Test getting a nonexistent client."""
        manager = DockerManager()
        assert manager.get_client("nonexistent") is None

    def test_remove_client(self):
        """Test removing a client."""
        manager = DockerManager()
        config = DockerConfig()
        client = DockerClient(config)
        manager.add_client("test", client)

        result = manager.remove_client("test")

        assert result is True
        assert manager.get_client("test") is None

    def test_remove_nonexistent_client(self):
        """Test removing a nonexistent client."""
        manager = DockerManager()
        result = manager.remove_client("nonexistent")
        assert result is False

    def test_list_clients(self):
        """Test listing clients."""
        manager = DockerManager()
        config = DockerConfig()

        manager.add_client("local", DockerClient(config))
        manager.add_client("remote", DockerClient(config))

        clients = manager.list_clients()

        assert "local" in clients
        assert "remote" in clients
        assert len(clients) == 2


# =============================================================================
# Input/Output Models Tests
# =============================================================================


class TestInputModels:
    """Tests for input models."""

    def test_create_docker_client_input(self):
        """Test CreateDockerClientInput."""
        input_model = CreateDockerClientInput(
            name="local",
            host="unix:///var/run/docker.sock",
            timeout=60,
        )
        assert input_model.name == "local"
        assert input_model.host == "unix:///var/run/docker.sock"
        assert input_model.timeout == 60

    def test_list_containers_input_defaults(self):
        """Test ListContainersInput with defaults."""
        input_model = ListContainersInput()
        assert input_model.client == "default"
        assert input_model.all_containers is False
        assert input_model.filter_status is None

    def test_get_container_input(self):
        """Test GetContainerInput."""
        input_model = GetContainerInput(container_id="abc123")
        assert input_model.container_id == "abc123"

    def test_stop_container_input(self):
        """Test StopContainerInput."""
        input_model = StopContainerInput(container_id="abc123", timeout=30)
        assert input_model.container_id == "abc123"
        assert input_model.timeout == 30

    def test_pull_image_input(self):
        """Test PullImageInput."""
        input_model = PullImageInput(image="nginx", tag="1.24")
        assert input_model.image == "nginx"
        assert input_model.tag == "1.24"

    def test_create_volume_input(self):
        """Test CreateVolumeInput."""
        input_model = CreateVolumeInput(name="my-volume", driver="local")
        assert input_model.name == "my-volume"
        assert input_model.driver == "local"


class TestOutputModels:
    """Tests for output models."""

    def test_create_docker_client_output(self):
        """Test CreateDockerClientOutput."""
        output = CreateDockerClientOutput(success=True, name="local")
        assert output.success is True
        assert output.name == "local"
        assert output.error is None

    def test_docker_containers_output(self):
        """Test DockerContainersOutput."""
        output = DockerContainersOutput(
            success=True,
            containers=[{"id": "abc", "name": "test"}],
        )
        assert output.success is True
        assert len(output.containers) == 1

    def test_docker_logs_output(self):
        """Test DockerLogsOutput."""
        output = DockerLogsOutput(success=True, logs="Application started")
        assert output.success is True
        assert output.logs == "Application started"

    def test_docker_simple_output_error(self):
        """Test DockerSimpleOutput with error."""
        output = DockerSimpleOutput(success=False, error="Container not found")
        assert output.success is False
        assert output.error == "Container not found"


# =============================================================================
# Tool Tests
# =============================================================================


class TestCreateDockerClientTool:
    """Tests for CreateDockerClientTool."""

    @pytest.fixture
    def manager(self):
        """Create test manager."""
        return DockerManager()

    @pytest.fixture
    def tool(self, manager):
        """Create test tool."""
        return CreateDockerClientTool(manager)

    @pytest.mark.asyncio
    async def test_create_client(self, tool, manager):
        """Test creating a client."""
        input_model = CreateDockerClientInput(
            name="local",
            host="unix:///var/run/docker.sock",
        )

        result = await tool.execute(input_model)

        assert result.success is True
        assert result.name == "local"
        assert manager.get_client("local") is not None

    def test_tool_metadata(self, tool):
        """Test tool metadata."""
        assert tool.metadata.id == "create_docker_client"
        assert tool.metadata.category == "utility"


class TestListContainersTool:
    """Tests for ListContainersTool."""

    @pytest.fixture
    def manager(self):
        """Create test manager with mock client."""
        manager = DockerManager()
        config = DockerConfig(host="tcp://localhost:2375")
        manager.add_client("default", DockerClient(config))
        return manager

    @pytest.fixture
    def tool(self, manager):
        """Create test tool."""
        return ListContainersTool(manager)

    @pytest.mark.asyncio
    async def test_list_containers_client_not_found(self, tool):
        """Test listing containers with nonexistent client."""
        input_model = ListContainersInput(client="nonexistent")
        result = await tool.execute(input_model)

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_list_containers_success(self, tool):
        """Test listing containers successfully."""
        api_response = [
            {
                "Id": "abc123",
                "Names": ["/test"],
                "Image": "nginx",
                "Status": "Up",
                "State": "running",
                "Created": 0,
                "Ports": [],
                "Labels": {},
            }
        ]

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(api_response).encode()
            mock_response.getcode.return_value = 200
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            input_model = ListContainersInput(client="default")
            result = await tool.execute(input_model)

            assert result.success is True
            assert len(result.containers) == 1


class TestStartContainerTool:
    """Tests for StartContainerTool."""

    @pytest.fixture
    def manager(self):
        """Create test manager with mock client."""
        manager = DockerManager()
        config = DockerConfig(host="tcp://localhost:2375")
        manager.add_client("default", DockerClient(config))
        return manager

    @pytest.fixture
    def tool(self, manager):
        """Create test tool."""
        return StartContainerTool(manager)

    @pytest.mark.asyncio
    async def test_start_container_success(self, tool):
        """Test starting a container successfully."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b""
            mock_response.getcode.return_value = 204
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            input_model = StartContainerInput(container_id="abc123")
            result = await tool.execute(input_model)

            assert result.success is True

    def test_tool_metadata(self, tool):
        """Test tool metadata."""
        assert tool.metadata.id == "start_docker_container"
        assert tool.metadata.category == "execution"


class TestStopContainerTool:
    """Tests for StopContainerTool."""

    @pytest.fixture
    def manager(self):
        """Create test manager with mock client."""
        manager = DockerManager()
        config = DockerConfig(host="tcp://localhost:2375")
        manager.add_client("default", DockerClient(config))
        return manager

    @pytest.fixture
    def tool(self, manager):
        """Create test tool."""
        return StopContainerTool(manager)

    @pytest.mark.asyncio
    async def test_stop_container_success(self, tool):
        """Test stopping a container successfully."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b""
            mock_response.getcode.return_value = 204
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            input_model = StopContainerInput(container_id="abc123")
            result = await tool.execute(input_model)

            assert result.success is True


class TestListImagesTool:
    """Tests for ListImagesTool."""

    @pytest.fixture
    def manager(self):
        """Create test manager with mock client."""
        manager = DockerManager()
        config = DockerConfig(host="tcp://localhost:2375")
        manager.add_client("default", DockerClient(config))
        return manager

    @pytest.fixture
    def tool(self, manager):
        """Create test tool."""
        return ListImagesTool(manager)

    @pytest.mark.asyncio
    async def test_list_images_success(self, tool):
        """Test listing images successfully."""
        api_response = [
            {
                "Id": "sha256:abc123",
                "RepoTags": ["nginx:latest"],
                "Size": 100000000,
                "Created": 1704067200,
                "Labels": {},
            }
        ]

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(api_response).encode()
            mock_response.getcode.return_value = 200
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            input_model = ListImagesInput(client="default")
            result = await tool.execute(input_model)

            assert result.success is True
            assert len(result.images) == 1


class TestListVolumesTool:
    """Tests for ListVolumesTool."""

    @pytest.fixture
    def manager(self):
        """Create test manager with mock client."""
        manager = DockerManager()
        config = DockerConfig(host="tcp://localhost:2375")
        manager.add_client("default", DockerClient(config))
        return manager

    @pytest.fixture
    def tool(self, manager):
        """Create test tool."""
        return ListVolumesTool(manager)

    @pytest.mark.asyncio
    async def test_list_volumes_success(self, tool):
        """Test listing volumes successfully."""
        api_response = {
            "Volumes": [
                {
                    "Name": "my-volume",
                    "Driver": "local",
                    "Mountpoint": "/var/lib/docker/volumes/my-volume/_data",
                }
            ]
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(api_response).encode()
            mock_response.getcode.return_value = 200
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            input_model = ListVolumesInput(client="default")
            result = await tool.execute(input_model)

            assert result.success is True
            assert len(result.volumes) == 1


class TestListNetworksTool:
    """Tests for ListNetworksTool."""

    @pytest.fixture
    def manager(self):
        """Create test manager with mock client."""
        manager = DockerManager()
        config = DockerConfig(host="tcp://localhost:2375")
        manager.add_client("default", DockerClient(config))
        return manager

    @pytest.fixture
    def tool(self, manager):
        """Create test tool."""
        return ListNetworksTool(manager)

    @pytest.mark.asyncio
    async def test_list_networks_success(self, tool):
        """Test listing networks successfully."""
        api_response = [
            {
                "Id": "abc123",
                "Name": "bridge",
                "Driver": "bridge",
                "Scope": "local",
                "IPAM": {"Config": []},
            }
        ]

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(api_response).encode()
            mock_response.getcode.return_value = 200
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            input_model = ListNetworksInput(client="default")
            result = await tool.execute(input_model)

            assert result.success is True
            assert len(result.networks) == 1


# =============================================================================
# Helper Functions Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_docker_config(self):
        """Test creating Docker config."""
        config = create_docker_config(
            host="tcp://localhost:2375",
            timeout=60,
            api_version="v1.44",
        )

        assert config.host == "tcp://localhost:2375"
        assert config.timeout == 60
        assert config.api_version == "v1.44"

    def test_create_docker_client(self):
        """Test creating Docker client."""
        config = DockerConfig()
        client = create_docker_client(config)

        assert isinstance(client, DockerClient)
        assert client.config == config

    def test_create_docker_manager(self):
        """Test creating Docker manager."""
        manager = create_docker_manager()

        assert isinstance(manager, DockerManager)
        assert manager._clients == {}

    def test_create_docker_tools(self):
        """Test creating Docker tools."""
        manager = DockerManager()
        tools = create_docker_tools(manager)

        assert isinstance(tools, dict)
        assert "create_docker_client" in tools
        # Containers
        assert "list_docker_containers" in tools
        assert "get_docker_container" in tools
        assert "start_docker_container" in tools
        assert "stop_docker_container" in tools
        assert "remove_docker_container" in tools
        assert "get_docker_container_logs" in tools
        # Images
        assert "list_docker_images" in tools
        assert "get_docker_image" in tools
        assert "pull_docker_image" in tools
        assert "remove_docker_image" in tools
        # Volumes
        assert "list_docker_volumes" in tools
        assert "create_docker_volume" in tools
        assert "remove_docker_volume" in tools
        # Networks
        assert "list_docker_networks" in tools
        assert "get_docker_network" in tools
        assert len(tools) == 16

    def test_all_tools_have_correct_manager(self):
        """Test all tools reference the same manager."""
        manager = DockerManager()
        tools = create_docker_tools(manager)

        for tool_name, tool in tools.items():
            assert tool.manager is manager
