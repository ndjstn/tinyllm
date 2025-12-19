"""Tests for Kubernetes tools."""

import json
import ssl
from io import BytesIO
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from tinyllm.tools.kubernetes import (
    # Enums
    K8sResourceType,
    # Config and dataclasses
    K8sConfig,
    K8sClient,
    K8sManager,
    K8sPod,
    K8sDeployment,
    K8sService,
    K8sNamespace,
    K8sConfigMap,
    K8sResult,
    # Input models
    CreateK8sClientInput,
    ListPodsInput,
    GetPodInput,
    GetPodLogsInput,
    DeletePodInput,
    ListDeploymentsInput,
    GetDeploymentInput,
    ScaleDeploymentInput,
    ListServicesInput,
    GetServiceInput,
    ListNamespacesInput,
    GetNamespaceInput,
    ListConfigMapsInput,
    GetConfigMapInput,
    # Output models
    CreateK8sClientOutput,
    K8sPodsOutput,
    K8sLogsOutput,
    K8sSimpleOutput,
    K8sDeploymentsOutput,
    K8sServicesOutput,
    K8sNamespacesOutput,
    K8sConfigMapsOutput,
    # Tools
    CreateK8sClientTool,
    ListPodsTool,
    GetPodTool,
    GetPodLogsTool,
    DeletePodTool,
    ListDeploymentsTool,
    GetDeploymentTool,
    ScaleDeploymentTool,
    ListServicesTool,
    GetServiceTool,
    ListNamespacesTool,
    GetNamespaceTool,
    ListConfigMapsTool,
    GetConfigMapTool,
    # Helper functions
    create_k8s_config,
    create_k8s_client,
    create_k8s_manager,
    create_k8s_tools,
)


# =============================================================================
# K8sResourceType Tests
# =============================================================================


class TestK8sResourceType:
    """Tests for K8sResourceType enum."""

    def test_resource_type_values(self):
        """Test resource type enum values."""
        assert K8sResourceType.POD == "pods"
        assert K8sResourceType.DEPLOYMENT == "deployments"
        assert K8sResourceType.SERVICE == "services"
        assert K8sResourceType.NAMESPACE == "namespaces"
        assert K8sResourceType.CONFIGMAP == "configmaps"
        assert K8sResourceType.SECRET == "secrets"
        assert K8sResourceType.NODE == "nodes"
        assert K8sResourceType.REPLICASET == "replicasets"

    def test_resource_type_count(self):
        """Test all resource types are defined."""
        assert len(K8sResourceType) == 8


# =============================================================================
# K8sConfig Tests
# =============================================================================


class TestK8sConfig:
    """Tests for K8sConfig."""

    def test_config_required_field(self):
        """Test config with required api_server field."""
        config = K8sConfig(api_server="https://k8s.example.com:6443")
        assert config.api_server == "https://k8s.example.com:6443"
        assert config.token is None
        assert config.insecure_skip_tls_verify is False
        assert config.default_namespace == "default"
        assert config.timeout == 30

    def test_config_all_fields(self):
        """Test config with all fields."""
        config = K8sConfig(
            api_server="https://k8s.example.com:6443",
            token="test-token-12345",
            certificate_authority="/path/to/ca.crt",
            client_certificate="/path/to/client.crt",
            client_key="/path/to/client.key",
            insecure_skip_tls_verify=True,
            timeout=60,
            default_namespace="production",
        )
        assert config.api_server == "https://k8s.example.com:6443"
        assert config.token == "test-token-12345"
        assert config.certificate_authority == "/path/to/ca.crt"
        assert config.client_certificate == "/path/to/client.crt"
        assert config.client_key == "/path/to/client.key"
        assert config.insecure_skip_tls_verify is True
        assert config.timeout == 60
        assert config.default_namespace == "production"


# =============================================================================
# K8sPod Tests
# =============================================================================


class TestK8sPod:
    """Tests for K8sPod dataclass."""

    def test_pod_from_api_response(self):
        """Test creating pod from API response."""
        api_response = {
            "metadata": {
                "name": "my-pod",
                "namespace": "default",
                "labels": {"app": "myapp"},
                "creationTimestamp": "2024-01-01T00:00:00Z",
            },
            "spec": {
                "nodeName": "node-1",
                "containers": [
                    {"name": "main"},
                    {"name": "sidecar"},
                ],
            },
            "status": {
                "phase": "Running",
                "podIP": "10.0.0.1",
            },
        }

        pod = K8sPod.from_api_response(api_response)
        assert pod.name == "my-pod"
        assert pod.namespace == "default"
        assert pod.status == "Running"
        assert pod.node_name == "node-1"
        assert pod.ip == "10.0.0.1"
        assert pod.containers == ["main", "sidecar"]
        assert pod.labels == {"app": "myapp"}
        assert pod.creation_timestamp == "2024-01-01T00:00:00Z"

    def test_pod_from_minimal_response(self):
        """Test creating pod from minimal API response."""
        api_response = {
            "metadata": {"name": "minimal-pod"},
            "spec": {},
            "status": {},
        }

        pod = K8sPod.from_api_response(api_response)
        assert pod.name == "minimal-pod"
        assert pod.namespace == ""
        assert pod.status == "Unknown"
        assert pod.node_name is None
        assert pod.containers == []

    def test_pod_to_dict(self):
        """Test converting pod to dictionary."""
        pod = K8sPod(
            name="test-pod",
            namespace="default",
            status="Running",
            node_name="node-1",
            ip="10.0.0.1",
            containers=["app"],
            labels={"app": "test"},
            creation_timestamp="2024-01-01T00:00:00Z",
        )

        result = pod.to_dict()
        assert result["name"] == "test-pod"
        assert result["namespace"] == "default"
        assert result["status"] == "Running"
        assert result["node_name"] == "node-1"
        assert result["ip"] == "10.0.0.1"
        assert result["containers"] == ["app"]
        assert result["labels"] == {"app": "test"}


# =============================================================================
# K8sDeployment Tests
# =============================================================================


class TestK8sDeployment:
    """Tests for K8sDeployment dataclass."""

    def test_deployment_from_api_response(self):
        """Test creating deployment from API response."""
        api_response = {
            "metadata": {
                "name": "my-deployment",
                "namespace": "production",
                "labels": {"app": "myapp"},
                "creationTimestamp": "2024-01-01T00:00:00Z",
            },
            "spec": {
                "replicas": 3,
            },
            "status": {
                "availableReplicas": 3,
                "readyReplicas": 3,
            },
        }

        deployment = K8sDeployment.from_api_response(api_response)
        assert deployment.name == "my-deployment"
        assert deployment.namespace == "production"
        assert deployment.replicas == 3
        assert deployment.available_replicas == 3
        assert deployment.ready_replicas == 3
        assert deployment.labels == {"app": "myapp"}

    def test_deployment_from_minimal_response(self):
        """Test creating deployment from minimal API response."""
        api_response = {
            "metadata": {"name": "minimal-deploy"},
            "spec": {},
            "status": {},
        }

        deployment = K8sDeployment.from_api_response(api_response)
        assert deployment.name == "minimal-deploy"
        assert deployment.replicas == 0
        assert deployment.available_replicas == 0
        assert deployment.ready_replicas == 0

    def test_deployment_to_dict(self):
        """Test converting deployment to dictionary."""
        deployment = K8sDeployment(
            name="test-deploy",
            namespace="default",
            replicas=5,
            available_replicas=4,
            ready_replicas=4,
            labels={"app": "test"},
            creation_timestamp="2024-01-01T00:00:00Z",
        )

        result = deployment.to_dict()
        assert result["name"] == "test-deploy"
        assert result["replicas"] == 5
        assert result["available_replicas"] == 4
        assert result["ready_replicas"] == 4


# =============================================================================
# K8sService Tests
# =============================================================================


class TestK8sService:
    """Tests for K8sService dataclass."""

    def test_service_from_api_response(self):
        """Test creating service from API response."""
        api_response = {
            "metadata": {
                "name": "my-service",
                "namespace": "default",
                "labels": {"app": "myapp"},
                "creationTimestamp": "2024-01-01T00:00:00Z",
            },
            "spec": {
                "type": "LoadBalancer",
                "clusterIP": "10.96.0.1",
                "externalIPs": ["1.2.3.4"],
                "ports": [
                    {"name": "http", "port": 80, "targetPort": 8080, "protocol": "TCP"},
                ],
            },
            "status": {},
        }

        service = K8sService.from_api_response(api_response)
        assert service.name == "my-service"
        assert service.namespace == "default"
        assert service.service_type == "LoadBalancer"
        assert service.cluster_ip == "10.96.0.1"
        assert service.external_ip == "1.2.3.4"
        assert len(service.ports) == 1
        assert service.ports[0]["port"] == 80

    def test_service_with_loadbalancer_ingress(self):
        """Test service with load balancer ingress."""
        api_response = {
            "metadata": {"name": "lb-service", "namespace": "default"},
            "spec": {
                "type": "LoadBalancer",
                "clusterIP": "10.96.0.1",
                "ports": [],
            },
            "status": {
                "loadBalancer": {
                    "ingress": [{"ip": "5.6.7.8"}],
                }
            },
        }

        service = K8sService.from_api_response(api_response)
        assert service.external_ip == "5.6.7.8"

    def test_service_with_hostname_ingress(self):
        """Test service with hostname ingress."""
        api_response = {
            "metadata": {"name": "lb-service", "namespace": "default"},
            "spec": {
                "type": "LoadBalancer",
                "clusterIP": "10.96.0.1",
                "ports": [],
            },
            "status": {
                "loadBalancer": {
                    "ingress": [{"hostname": "lb.example.com"}],
                }
            },
        }

        service = K8sService.from_api_response(api_response)
        assert service.external_ip == "lb.example.com"

    def test_service_to_dict(self):
        """Test converting service to dictionary."""
        service = K8sService(
            name="test-service",
            namespace="default",
            service_type="ClusterIP",
            cluster_ip="10.96.0.1",
            ports=[{"port": 80}],
            labels={"app": "test"},
        )

        result = service.to_dict()
        assert result["name"] == "test-service"
        assert result["service_type"] == "ClusterIP"
        assert result["cluster_ip"] == "10.96.0.1"


# =============================================================================
# K8sNamespace Tests
# =============================================================================


class TestK8sNamespace:
    """Tests for K8sNamespace dataclass."""

    def test_namespace_from_api_response(self):
        """Test creating namespace from API response."""
        api_response = {
            "metadata": {
                "name": "production",
                "labels": {"env": "prod"},
                "creationTimestamp": "2024-01-01T00:00:00Z",
            },
            "status": {
                "phase": "Active",
            },
        }

        namespace = K8sNamespace.from_api_response(api_response)
        assert namespace.name == "production"
        assert namespace.status == "Active"
        assert namespace.labels == {"env": "prod"}

    def test_namespace_from_minimal_response(self):
        """Test creating namespace from minimal API response."""
        api_response = {
            "metadata": {"name": "default"},
            "status": {},
        }

        namespace = K8sNamespace.from_api_response(api_response)
        assert namespace.name == "default"
        assert namespace.status == "Active"

    def test_namespace_to_dict(self):
        """Test converting namespace to dictionary."""
        namespace = K8sNamespace(
            name="test-ns",
            status="Active",
            labels={"team": "platform"},
        )

        result = namespace.to_dict()
        assert result["name"] == "test-ns"
        assert result["status"] == "Active"
        assert result["labels"] == {"team": "platform"}


# =============================================================================
# K8sConfigMap Tests
# =============================================================================


class TestK8sConfigMap:
    """Tests for K8sConfigMap dataclass."""

    def test_configmap_from_api_response(self):
        """Test creating ConfigMap from API response."""
        api_response = {
            "metadata": {
                "name": "my-config",
                "namespace": "default",
                "labels": {"app": "myapp"},
                "creationTimestamp": "2024-01-01T00:00:00Z",
            },
            "data": {
                "config.yaml": "key: value",
                "settings.json": '{"debug": true}',
            },
        }

        configmap = K8sConfigMap.from_api_response(api_response)
        assert configmap.name == "my-config"
        assert configmap.namespace == "default"
        assert configmap.data == {
            "config.yaml": "key: value",
            "settings.json": '{"debug": true}',
        }
        assert configmap.labels == {"app": "myapp"}

    def test_configmap_to_dict(self):
        """Test converting ConfigMap to dictionary."""
        configmap = K8sConfigMap(
            name="test-config",
            namespace="default",
            data={"key": "value"},
            labels={"app": "test"},
        )

        result = configmap.to_dict()
        assert result["name"] == "test-config"
        assert result["data"] == {"key": "value"}


# =============================================================================
# K8sResult Tests
# =============================================================================


class TestK8sResult:
    """Tests for K8sResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = K8sResult(
            success=True,
            data={"pods": []},
            status_code=200,
        )
        assert result.success is True
        assert result.data == {"pods": []}
        assert result.error is None
        assert result.status_code == 200

    def test_error_result(self):
        """Test error result."""
        result = K8sResult(
            success=False,
            error="Not found",
            status_code=404,
        )
        assert result.success is False
        assert result.error == "Not found"
        assert result.status_code == 404

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = K8sResult(success=True, data={"key": "value"}, status_code=200)
        result_dict = result.to_dict()
        assert result_dict["success"] is True
        assert result_dict["data"] == {"key": "value"}
        assert result_dict["status_code"] == 200


# =============================================================================
# K8sClient Tests
# =============================================================================


class TestK8sClient:
    """Tests for K8sClient."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return K8sConfig(
            api_server="https://k8s.example.com:6443",
            token="test-token",
            insecure_skip_tls_verify=True,
        )

    @pytest.fixture
    def client(self, config):
        """Create a test client."""
        return K8sClient(config)

    def test_client_initialization(self, client, config):
        """Test client initialization."""
        assert client.config == config
        assert client._ssl_context is not None

    def test_ssl_context_insecure(self, client):
        """Test SSL context with insecure mode."""
        assert client._ssl_context.check_hostname is False
        assert client._ssl_context.verify_mode == ssl.CERT_NONE

    def test_ssl_context_secure(self):
        """Test SSL context in secure mode."""
        config = K8sConfig(
            api_server="https://k8s.example.com:6443",
            insecure_skip_tls_verify=False,
        )
        client = K8sClient(config)
        assert client._ssl_context.verify_mode != ssl.CERT_NONE

    def test_get_api_version_deployment(self, client):
        """Test getting API version for deployments."""
        version = client._get_api_version(K8sResourceType.DEPLOYMENT)
        assert version == "apis/apps/v1"

    def test_get_api_version_pod(self, client):
        """Test getting API version for pods."""
        version = client._get_api_version(K8sResourceType.POD)
        assert version == "api/v1"


class TestK8sClientOperations:
    """Tests for K8sClient operations with mocked HTTP."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return K8sConfig(
            api_server="https://k8s.example.com:6443",
            token="test-token",
            insecure_skip_tls_verify=True,
        )

    @pytest.fixture
    def client(self, config):
        """Create test client."""
        return K8sClient(config)

    def _mock_response(self, data, status=200):
        """Create mock HTTP response."""
        response = MagicMock()
        response.read.return_value = json.dumps(data).encode()
        response.getcode.return_value = status
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)
        return response

    def test_list_namespaces(self, client):
        """Test listing namespaces."""
        api_response = {
            "items": [
                {"metadata": {"name": "default"}, "status": {"phase": "Active"}},
                {"metadata": {"name": "kube-system"}, "status": {"phase": "Active"}},
            ]
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(api_response)

            result = client.list_namespaces()

            assert result.success is True
            assert len(result.data["namespaces"]) == 2
            assert result.data["namespaces"][0]["name"] == "default"

    def test_get_namespace(self, client):
        """Test getting a namespace."""
        api_response = {
            "metadata": {"name": "production"},
            "status": {"phase": "Active"},
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(api_response)

            result = client.get_namespace("production")

            assert result.success is True
            assert result.data["namespace"]["name"] == "production"

    def test_list_pods(self, client):
        """Test listing pods."""
        api_response = {
            "items": [
                {
                    "metadata": {"name": "pod-1", "namespace": "default"},
                    "spec": {"containers": [{"name": "app"}]},
                    "status": {"phase": "Running"},
                },
            ]
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(api_response)

            result = client.list_pods(namespace="default")

            assert result.success is True
            assert len(result.data["pods"]) == 1
            assert result.data["pods"][0]["name"] == "pod-1"

    def test_list_pods_with_label_selector(self, client):
        """Test listing pods with label selector."""
        api_response = {"items": []}

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(api_response)

            result = client.list_pods(namespace="default", label_selector="app=myapp")

            assert result.success is True
            # Verify the URL contains the label selector
            call_args = mock_urlopen.call_args
            request = call_args[0][0]
            assert "labelSelector=app=myapp" in request.full_url

    def test_get_pod(self, client):
        """Test getting a pod."""
        api_response = {
            "metadata": {"name": "my-pod", "namespace": "default"},
            "spec": {"containers": [{"name": "app"}]},
            "status": {"phase": "Running", "podIP": "10.0.0.1"},
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(api_response)

            result = client.get_pod("my-pod", "default")

            assert result.success is True
            assert result.data["pod"]["name"] == "my-pod"
            assert result.data["pod"]["ip"] == "10.0.0.1"

    def test_delete_pod(self, client):
        """Test deleting a pod."""
        api_response = {"status": "Success"}

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(api_response)

            result = client.delete_pod("my-pod", "default")

            assert result.success is True

    def test_get_pod_logs(self, client):
        """Test getting pod logs."""
        # Logs are returned as plain text, but our mock returns JSON
        log_content = "2024-01-01 INFO: Application started"

        with patch("urllib.request.urlopen") as mock_urlopen:
            response = MagicMock()
            response.read.return_value = json.dumps(log_content).encode()
            response.getcode.return_value = 200
            response.__enter__ = MagicMock(return_value=response)
            response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = response

            result = client.get_pod_logs("my-pod", "default", tail_lines=100)

            assert result.success is True

    def test_list_deployments(self, client):
        """Test listing deployments."""
        api_response = {
            "items": [
                {
                    "metadata": {"name": "my-deploy", "namespace": "default"},
                    "spec": {"replicas": 3},
                    "status": {"availableReplicas": 3, "readyReplicas": 3},
                },
            ]
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(api_response)

            result = client.list_deployments(namespace="default")

            assert result.success is True
            assert len(result.data["deployments"]) == 1
            assert result.data["deployments"][0]["replicas"] == 3

    def test_get_deployment(self, client):
        """Test getting a deployment."""
        api_response = {
            "metadata": {"name": "my-deploy", "namespace": "default"},
            "spec": {"replicas": 3},
            "status": {"availableReplicas": 3, "readyReplicas": 3},
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(api_response)

            result = client.get_deployment("my-deploy", "default")

            assert result.success is True
            assert result.data["deployment"]["name"] == "my-deploy"

    def test_scale_deployment(self, client):
        """Test scaling a deployment."""
        api_response = {
            "metadata": {"name": "my-deploy"},
            "spec": {"replicas": 5},
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(api_response)

            result = client.scale_deployment("my-deploy", "default", 5)

            assert result.success is True
            assert result.data["replicas"] == 5

    def test_list_services(self, client):
        """Test listing services."""
        api_response = {
            "items": [
                {
                    "metadata": {"name": "my-service", "namespace": "default"},
                    "spec": {"type": "ClusterIP", "clusterIP": "10.96.0.1", "ports": []},
                    "status": {},
                },
            ]
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(api_response)

            result = client.list_services(namespace="default")

            assert result.success is True
            assert len(result.data["services"]) == 1
            assert result.data["services"][0]["name"] == "my-service"

    def test_get_service(self, client):
        """Test getting a service."""
        api_response = {
            "metadata": {"name": "my-service", "namespace": "default"},
            "spec": {"type": "ClusterIP", "clusterIP": "10.96.0.1", "ports": []},
            "status": {},
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(api_response)

            result = client.get_service("my-service", "default")

            assert result.success is True
            assert result.data["service"]["name"] == "my-service"

    def test_list_configmaps(self, client):
        """Test listing ConfigMaps."""
        api_response = {
            "items": [
                {
                    "metadata": {"name": "my-config", "namespace": "default"},
                    "data": {"key": "value"},
                },
            ]
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(api_response)

            result = client.list_configmaps(namespace="default")

            assert result.success is True
            assert len(result.data["configmaps"]) == 1

    def test_get_configmap(self, client):
        """Test getting a ConfigMap."""
        api_response = {
            "metadata": {"name": "my-config", "namespace": "default"},
            "data": {"key": "value"},
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response(api_response)

            result = client.get_configmap("my-config", "default")

            assert result.success is True
            assert result.data["configmap"]["name"] == "my-config"


class TestK8sClientErrors:
    """Tests for K8sClient error handling."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        config = K8sConfig(
            api_server="https://k8s.example.com:6443",
            token="test-token",
            insecure_skip_tls_verify=True,
        )
        return K8sClient(config)

    def test_http_error(self, client):
        """Test handling HTTP error."""
        error_body = json.dumps({"message": "Not found"}).encode()
        http_error = HTTPError(
            url="https://k8s.example.com",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=BytesIO(error_body),
        )

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = http_error

            result = client.list_pods()

            assert result.success is False
            assert "404" in result.error
            assert result.status_code == 404

    def test_url_error(self, client):
        """Test handling URL error (connection error)."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = URLError("Connection refused")

            result = client.list_pods()

            assert result.success is False
            assert "Connection error" in result.error

    def test_generic_exception(self, client):
        """Test handling generic exception."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = Exception("Unexpected error")

            result = client.list_pods()

            assert result.success is False
            assert "Unexpected error" in result.error


# =============================================================================
# K8sManager Tests
# =============================================================================


class TestK8sManager:
    """Tests for K8sManager."""

    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = K8sManager()
        assert manager._clients == {}

    def test_add_client(self):
        """Test adding a client."""
        manager = K8sManager()
        config = K8sConfig(api_server="https://k8s.example.com:6443")
        client = K8sClient(config)

        manager.add_client("production", client)

        assert "production" in manager._clients
        assert manager.get_client("production") == client

    def test_get_nonexistent_client(self):
        """Test getting a nonexistent client."""
        manager = K8sManager()
        assert manager.get_client("nonexistent") is None

    def test_remove_client(self):
        """Test removing a client."""
        manager = K8sManager()
        config = K8sConfig(api_server="https://k8s.example.com:6443")
        client = K8sClient(config)
        manager.add_client("test", client)

        result = manager.remove_client("test")

        assert result is True
        assert manager.get_client("test") is None

    def test_remove_nonexistent_client(self):
        """Test removing a nonexistent client."""
        manager = K8sManager()
        result = manager.remove_client("nonexistent")
        assert result is False

    def test_list_clients(self):
        """Test listing clients."""
        manager = K8sManager()
        config = K8sConfig(api_server="https://k8s.example.com:6443")

        manager.add_client("prod", K8sClient(config))
        manager.add_client("staging", K8sClient(config))

        clients = manager.list_clients()

        assert "prod" in clients
        assert "staging" in clients
        assert len(clients) == 2


# =============================================================================
# Input/Output Models Tests
# =============================================================================


class TestInputModels:
    """Tests for input models."""

    def test_create_k8s_client_input(self):
        """Test CreateK8sClientInput."""
        input_model = CreateK8sClientInput(
            name="production",
            api_server="https://k8s.example.com:6443",
            token="my-token",
            insecure_skip_tls_verify=True,
            default_namespace="prod",
        )
        assert input_model.name == "production"
        assert input_model.api_server == "https://k8s.example.com:6443"
        assert input_model.token == "my-token"

    def test_list_pods_input_defaults(self):
        """Test ListPodsInput with defaults."""
        input_model = ListPodsInput()
        assert input_model.client == "default"
        assert input_model.namespace is None
        assert input_model.label_selector is None

    def test_get_pod_input(self):
        """Test GetPodInput."""
        input_model = GetPodInput(name="my-pod", namespace="production")
        assert input_model.name == "my-pod"
        assert input_model.namespace == "production"

    def test_get_pod_logs_input(self):
        """Test GetPodLogsInput."""
        input_model = GetPodLogsInput(
            name="my-pod",
            container="app",
            tail_lines=50,
        )
        assert input_model.name == "my-pod"
        assert input_model.container == "app"
        assert input_model.tail_lines == 50

    def test_scale_deployment_input(self):
        """Test ScaleDeploymentInput."""
        input_model = ScaleDeploymentInput(
            name="my-deploy",
            namespace="production",
            replicas=5,
        )
        assert input_model.name == "my-deploy"
        assert input_model.replicas == 5


class TestOutputModels:
    """Tests for output models."""

    def test_create_k8s_client_output(self):
        """Test CreateK8sClientOutput."""
        output = CreateK8sClientOutput(success=True, name="production")
        assert output.success is True
        assert output.name == "production"
        assert output.error is None

    def test_k8s_pods_output(self):
        """Test K8sPodsOutput."""
        output = K8sPodsOutput(
            success=True,
            pods=[{"name": "pod-1"}, {"name": "pod-2"}],
        )
        assert output.success is True
        assert len(output.pods) == 2

    def test_k8s_logs_output(self):
        """Test K8sLogsOutput."""
        output = K8sLogsOutput(success=True, logs="Application started")
        assert output.success is True
        assert output.logs == "Application started"

    def test_k8s_simple_output_error(self):
        """Test K8sSimpleOutput with error."""
        output = K8sSimpleOutput(success=False, error="Pod not found")
        assert output.success is False
        assert output.error == "Pod not found"


# =============================================================================
# Tool Tests
# =============================================================================


class TestCreateK8sClientTool:
    """Tests for CreateK8sClientTool."""

    @pytest.fixture
    def manager(self):
        """Create test manager."""
        return K8sManager()

    @pytest.fixture
    def tool(self, manager):
        """Create test tool."""
        return CreateK8sClientTool(manager)

    @pytest.mark.asyncio
    async def test_create_client(self, tool, manager):
        """Test creating a client."""
        input_model = CreateK8sClientInput(
            name="production",
            api_server="https://k8s.example.com:6443",
            token="test-token",
        )

        result = await tool.execute(input_model)

        assert result.success is True
        assert result.name == "production"
        assert manager.get_client("production") is not None

    def test_tool_metadata(self, tool):
        """Test tool metadata."""
        assert tool.metadata.id == "create_k8s_client"
        assert tool.metadata.category == "utility"


class TestListPodsTool:
    """Tests for ListPodsTool."""

    @pytest.fixture
    def manager(self):
        """Create test manager with mock client."""
        manager = K8sManager()
        config = K8sConfig(
            api_server="https://k8s.example.com:6443",
            insecure_skip_tls_verify=True,
        )
        manager.add_client("default", K8sClient(config))
        return manager

    @pytest.fixture
    def tool(self, manager):
        """Create test tool."""
        return ListPodsTool(manager)

    @pytest.mark.asyncio
    async def test_list_pods_client_not_found(self, tool):
        """Test listing pods with nonexistent client."""
        input_model = ListPodsInput(client="nonexistent")
        result = await tool.execute(input_model)

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_list_pods_success(self, tool):
        """Test listing pods successfully."""
        api_response = {
            "items": [
                {
                    "metadata": {"name": "pod-1", "namespace": "default"},
                    "spec": {"containers": []},
                    "status": {"phase": "Running"},
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

            input_model = ListPodsInput(client="default")
            result = await tool.execute(input_model)

            assert result.success is True
            assert len(result.pods) == 1


class TestGetPodTool:
    """Tests for GetPodTool."""

    @pytest.fixture
    def manager(self):
        """Create test manager with mock client."""
        manager = K8sManager()
        config = K8sConfig(
            api_server="https://k8s.example.com:6443",
            insecure_skip_tls_verify=True,
        )
        manager.add_client("default", K8sClient(config))
        return manager

    @pytest.fixture
    def tool(self, manager):
        """Create test tool."""
        return GetPodTool(manager)

    @pytest.mark.asyncio
    async def test_get_pod_success(self, tool):
        """Test getting a pod successfully."""
        api_response = {
            "metadata": {"name": "my-pod", "namespace": "default"},
            "spec": {"containers": [{"name": "app"}]},
            "status": {"phase": "Running"},
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(api_response).encode()
            mock_response.getcode.return_value = 200
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            input_model = GetPodInput(name="my-pod", namespace="default")
            result = await tool.execute(input_model)

            assert result.success is True
            assert result.pods[0]["name"] == "my-pod"


class TestDeletePodTool:
    """Tests for DeletePodTool."""

    @pytest.fixture
    def manager(self):
        """Create test manager with mock client."""
        manager = K8sManager()
        config = K8sConfig(
            api_server="https://k8s.example.com:6443",
            insecure_skip_tls_verify=True,
        )
        manager.add_client("default", K8sClient(config))
        return manager

    @pytest.fixture
    def tool(self, manager):
        """Create test tool."""
        return DeletePodTool(manager)

    @pytest.mark.asyncio
    async def test_delete_pod_success(self, tool):
        """Test deleting a pod successfully."""
        api_response = {"status": "Success"}

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(api_response).encode()
            mock_response.getcode.return_value = 200
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            input_model = DeletePodInput(name="my-pod", namespace="default")
            result = await tool.execute(input_model)

            assert result.success is True


class TestScaleDeploymentTool:
    """Tests for ScaleDeploymentTool."""

    @pytest.fixture
    def manager(self):
        """Create test manager with mock client."""
        manager = K8sManager()
        config = K8sConfig(
            api_server="https://k8s.example.com:6443",
            insecure_skip_tls_verify=True,
        )
        manager.add_client("default", K8sClient(config))
        return manager

    @pytest.fixture
    def tool(self, manager):
        """Create test tool."""
        return ScaleDeploymentTool(manager)

    @pytest.mark.asyncio
    async def test_scale_deployment_success(self, tool):
        """Test scaling a deployment successfully."""
        api_response = {"spec": {"replicas": 5}}

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(api_response).encode()
            mock_response.getcode.return_value = 200
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            input_model = ScaleDeploymentInput(
                name="my-deploy",
                namespace="default",
                replicas=5,
            )
            result = await tool.execute(input_model)

            assert result.success is True

    def test_tool_metadata(self, tool):
        """Test tool metadata."""
        assert tool.metadata.id == "scale_k8s_deployment"
        assert tool.metadata.category == "execution"


class TestListServicesTool:
    """Tests for ListServicesTool."""

    @pytest.fixture
    def manager(self):
        """Create test manager with mock client."""
        manager = K8sManager()
        config = K8sConfig(
            api_server="https://k8s.example.com:6443",
            insecure_skip_tls_verify=True,
        )
        manager.add_client("default", K8sClient(config))
        return manager

    @pytest.fixture
    def tool(self, manager):
        """Create test tool."""
        return ListServicesTool(manager)

    @pytest.mark.asyncio
    async def test_list_services_success(self, tool):
        """Test listing services successfully."""
        api_response = {
            "items": [
                {
                    "metadata": {"name": "my-service", "namespace": "default"},
                    "spec": {"type": "ClusterIP", "clusterIP": "10.96.0.1", "ports": []},
                    "status": {},
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

            input_model = ListServicesInput(client="default")
            result = await tool.execute(input_model)

            assert result.success is True
            assert len(result.services) == 1


class TestListNamespacesTool:
    """Tests for ListNamespacesTool."""

    @pytest.fixture
    def manager(self):
        """Create test manager with mock client."""
        manager = K8sManager()
        config = K8sConfig(
            api_server="https://k8s.example.com:6443",
            insecure_skip_tls_verify=True,
        )
        manager.add_client("default", K8sClient(config))
        return manager

    @pytest.fixture
    def tool(self, manager):
        """Create test tool."""
        return ListNamespacesTool(manager)

    @pytest.mark.asyncio
    async def test_list_namespaces_success(self, tool):
        """Test listing namespaces successfully."""
        api_response = {
            "items": [
                {"metadata": {"name": "default"}, "status": {"phase": "Active"}},
                {"metadata": {"name": "kube-system"}, "status": {"phase": "Active"}},
            ]
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(api_response).encode()
            mock_response.getcode.return_value = 200
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            input_model = ListNamespacesInput(client="default")
            result = await tool.execute(input_model)

            assert result.success is True
            assert len(result.namespaces) == 2


class TestListConfigMapsTool:
    """Tests for ListConfigMapsTool."""

    @pytest.fixture
    def manager(self):
        """Create test manager with mock client."""
        manager = K8sManager()
        config = K8sConfig(
            api_server="https://k8s.example.com:6443",
            insecure_skip_tls_verify=True,
        )
        manager.add_client("default", K8sClient(config))
        return manager

    @pytest.fixture
    def tool(self, manager):
        """Create test tool."""
        return ListConfigMapsTool(manager)

    @pytest.mark.asyncio
    async def test_list_configmaps_success(self, tool):
        """Test listing ConfigMaps successfully."""
        api_response = {
            "items": [
                {
                    "metadata": {"name": "my-config", "namespace": "default"},
                    "data": {"key": "value"},
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

            input_model = ListConfigMapsInput(client="default")
            result = await tool.execute(input_model)

            assert result.success is True
            assert len(result.configmaps) == 1


# =============================================================================
# Helper Functions Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_k8s_config(self):
        """Test creating K8s config."""
        config = create_k8s_config(
            api_server="https://k8s.example.com:6443",
            token="my-token",
            insecure_skip_tls_verify=True,
            default_namespace="production",
        )

        assert config.api_server == "https://k8s.example.com:6443"
        assert config.token == "my-token"
        assert config.insecure_skip_tls_verify is True
        assert config.default_namespace == "production"

    def test_create_k8s_client(self):
        """Test creating K8s client."""
        config = K8sConfig(api_server="https://k8s.example.com:6443")
        client = create_k8s_client(config)

        assert isinstance(client, K8sClient)
        assert client.config == config

    def test_create_k8s_manager(self):
        """Test creating K8s manager."""
        manager = create_k8s_manager()

        assert isinstance(manager, K8sManager)
        assert manager._clients == {}

    def test_create_k8s_tools(self):
        """Test creating K8s tools."""
        manager = K8sManager()
        tools = create_k8s_tools(manager)

        assert isinstance(tools, dict)
        assert "create_k8s_client" in tools
        assert "list_k8s_pods" in tools
        assert "get_k8s_pod" in tools
        assert "get_k8s_pod_logs" in tools
        assert "delete_k8s_pod" in tools
        assert "list_k8s_deployments" in tools
        assert "get_k8s_deployment" in tools
        assert "scale_k8s_deployment" in tools
        assert "list_k8s_services" in tools
        assert "get_k8s_service" in tools
        assert "list_k8s_namespaces" in tools
        assert "get_k8s_namespace" in tools
        assert "list_k8s_configmaps" in tools
        assert "get_k8s_configmap" in tools
        assert len(tools) == 14

    def test_all_tools_have_correct_manager(self):
        """Test all tools reference the same manager."""
        manager = K8sManager()
        tools = create_k8s_tools(manager)

        for tool_name, tool in tools.items():
            assert tool.manager is manager
