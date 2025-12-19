"""Kubernetes tools for TinyLLM.

Provides tools for interacting with Kubernetes clusters including:
- Pod management (list, get, delete, logs)
- Deployment management (list, get, scale)
- Service management (list, get)
- Namespace management (list, get)
- ConfigMap management (list, get)
"""

import base64
import json
import logging
import ssl
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.tools.base import BaseTool, ToolMetadata

logger = logging.getLogger(__name__)


class K8sResourceType(str, Enum):
    """Kubernetes resource types."""

    POD = "pods"
    DEPLOYMENT = "deployments"
    SERVICE = "services"
    NAMESPACE = "namespaces"
    CONFIGMAP = "configmaps"
    SECRET = "secrets"
    NODE = "nodes"
    REPLICASET = "replicasets"


@dataclass
class K8sConfig:
    """Kubernetes cluster configuration."""

    api_server: str
    token: Optional[str] = None
    certificate_authority: Optional[str] = None
    client_certificate: Optional[str] = None
    client_key: Optional[str] = None
    insecure_skip_tls_verify: bool = False
    timeout: int = 30
    default_namespace: str = "default"


@dataclass
class K8sPod:
    """Kubernetes Pod representation."""

    name: str
    namespace: str
    status: str
    node_name: Optional[str] = None
    ip: Optional[str] = None
    containers: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    creation_timestamp: Optional[str] = None

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "K8sPod":
        """Create from Kubernetes API response."""
        metadata = data.get("metadata", {})
        spec = data.get("spec", {})
        status = data.get("status", {})

        containers = [c.get("name", "") for c in spec.get("containers", [])]

        return cls(
            name=metadata.get("name", ""),
            namespace=metadata.get("namespace", ""),
            status=status.get("phase", "Unknown"),
            node_name=spec.get("nodeName"),
            ip=status.get("podIP"),
            containers=containers,
            labels=metadata.get("labels", {}),
            creation_timestamp=metadata.get("creationTimestamp"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "namespace": self.namespace,
            "status": self.status,
            "node_name": self.node_name,
            "ip": self.ip,
            "containers": self.containers,
            "labels": self.labels,
            "creation_timestamp": self.creation_timestamp,
        }


@dataclass
class K8sDeployment:
    """Kubernetes Deployment representation."""

    name: str
    namespace: str
    replicas: int
    available_replicas: int
    ready_replicas: int
    labels: Dict[str, str] = field(default_factory=dict)
    creation_timestamp: Optional[str] = None

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "K8sDeployment":
        """Create from Kubernetes API response."""
        metadata = data.get("metadata", {})
        spec = data.get("spec", {})
        status = data.get("status", {})

        return cls(
            name=metadata.get("name", ""),
            namespace=metadata.get("namespace", ""),
            replicas=spec.get("replicas", 0),
            available_replicas=status.get("availableReplicas", 0),
            ready_replicas=status.get("readyReplicas", 0),
            labels=metadata.get("labels", {}),
            creation_timestamp=metadata.get("creationTimestamp"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "namespace": self.namespace,
            "replicas": self.replicas,
            "available_replicas": self.available_replicas,
            "ready_replicas": self.ready_replicas,
            "labels": self.labels,
            "creation_timestamp": self.creation_timestamp,
        }


@dataclass
class K8sService:
    """Kubernetes Service representation."""

    name: str
    namespace: str
    service_type: str
    cluster_ip: Optional[str] = None
    external_ip: Optional[str] = None
    ports: List[Dict[str, Any]] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    creation_timestamp: Optional[str] = None

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "K8sService":
        """Create from Kubernetes API response."""
        metadata = data.get("metadata", {})
        spec = data.get("spec", {})
        status = data.get("status", {})

        external_ips = spec.get("externalIPs", [])
        load_balancer = status.get("loadBalancer", {}).get("ingress", [])
        external_ip = None
        if external_ips:
            external_ip = external_ips[0]
        elif load_balancer:
            external_ip = load_balancer[0].get("ip") or load_balancer[0].get("hostname")

        ports = []
        for port in spec.get("ports", []):
            ports.append({
                "name": port.get("name"),
                "port": port.get("port"),
                "target_port": port.get("targetPort"),
                "protocol": port.get("protocol", "TCP"),
                "node_port": port.get("nodePort"),
            })

        return cls(
            name=metadata.get("name", ""),
            namespace=metadata.get("namespace", ""),
            service_type=spec.get("type", "ClusterIP"),
            cluster_ip=spec.get("clusterIP"),
            external_ip=external_ip,
            ports=ports,
            labels=metadata.get("labels", {}),
            creation_timestamp=metadata.get("creationTimestamp"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "namespace": self.namespace,
            "service_type": self.service_type,
            "cluster_ip": self.cluster_ip,
            "external_ip": self.external_ip,
            "ports": self.ports,
            "labels": self.labels,
            "creation_timestamp": self.creation_timestamp,
        }


@dataclass
class K8sNamespace:
    """Kubernetes Namespace representation."""

    name: str
    status: str
    labels: Dict[str, str] = field(default_factory=dict)
    creation_timestamp: Optional[str] = None

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "K8sNamespace":
        """Create from Kubernetes API response."""
        metadata = data.get("metadata", {})
        status = data.get("status", {})

        return cls(
            name=metadata.get("name", ""),
            status=status.get("phase", "Active"),
            labels=metadata.get("labels", {}),
            creation_timestamp=metadata.get("creationTimestamp"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status,
            "labels": self.labels,
            "creation_timestamp": self.creation_timestamp,
        }


@dataclass
class K8sConfigMap:
    """Kubernetes ConfigMap representation."""

    name: str
    namespace: str
    data: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    creation_timestamp: Optional[str] = None

    @classmethod
    def from_api_response(cls, response: Dict[str, Any]) -> "K8sConfigMap":
        """Create from Kubernetes API response."""
        metadata = response.get("metadata", {})

        return cls(
            name=metadata.get("name", ""),
            namespace=metadata.get("namespace", ""),
            data=response.get("data", {}),
            labels=metadata.get("labels", {}),
            creation_timestamp=metadata.get("creationTimestamp"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "namespace": self.namespace,
            "data": self.data,
            "labels": self.labels,
            "creation_timestamp": self.creation_timestamp,
        }


@dataclass
class K8sResult:
    """Result from Kubernetes API operation."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    status_code: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "status_code": self.status_code,
        }


class K8sClient:
    """Kubernetes API client."""

    def __init__(self, config: K8sConfig):
        """Initialize client.

        Args:
            config: Kubernetes configuration.
        """
        self.config = config
        self._ssl_context = self._create_ssl_context()

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for API calls."""
        if self.config.insecure_skip_tls_verify:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            return context

        context = ssl.create_default_context()
        if self.config.certificate_authority:
            context.load_verify_locations(cafile=self.config.certificate_authority)

        if self.config.client_certificate and self.config.client_key:
            context.load_cert_chain(
                self.config.client_certificate,
                self.config.client_key,
            )

        return context

    def _make_request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> K8sResult:
        """Make HTTP request to Kubernetes API.

        Args:
            method: HTTP method.
            path: API path.
            data: Request body.

        Returns:
            Kubernetes result.
        """
        url = f"{self.config.api_server.rstrip('/')}{path}"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"

        body = None
        if data:
            body = json.dumps(data).encode("utf-8")

        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers=headers,
                method=method,
            )

            with urllib.request.urlopen(
                req,
                timeout=self.config.timeout,
                context=self._ssl_context,
            ) as response:
                response_body = response.read().decode("utf-8")
                response_data = json.loads(response_body) if response_body else None

                return K8sResult(
                    success=True,
                    data=response_data,
                    status_code=response.getcode(),
                )

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
                error_data = json.loads(error_body)
                error_msg = error_data.get("message", error_body)
            except Exception:
                error_msg = error_body or str(e.reason)

            return K8sResult(
                success=False,
                error=f"HTTP {e.code}: {error_msg}",
                status_code=e.code,
            )
        except urllib.error.URLError as e:
            return K8sResult(
                success=False,
                error=f"Connection error: {e.reason}",
            )
        except Exception as e:
            return K8sResult(
                success=False,
                error=str(e),
            )

    def _get_api_version(self, resource: K8sResourceType) -> str:
        """Get API version for resource type."""
        if resource in (K8sResourceType.DEPLOYMENT, K8sResourceType.REPLICASET):
            return "apis/apps/v1"
        return "api/v1"

    # Namespace operations

    def list_namespaces(self) -> K8sResult:
        """List all namespaces."""
        result = self._make_request("GET", "/api/v1/namespaces")

        if result.success and result.data:
            items = result.data.get("items", [])
            namespaces = [K8sNamespace.from_api_response(ns).to_dict() for ns in items]
            result.data = {"namespaces": namespaces}

        return result

    def get_namespace(self, name: str) -> K8sResult:
        """Get a specific namespace."""
        result = self._make_request("GET", f"/api/v1/namespaces/{name}")

        if result.success and result.data:
            result.data = {"namespace": K8sNamespace.from_api_response(result.data).to_dict()}

        return result

    # Pod operations

    def list_pods(
        self,
        namespace: Optional[str] = None,
        label_selector: Optional[str] = None,
    ) -> K8sResult:
        """List pods.

        Args:
            namespace: Namespace to list pods in. If None, lists all namespaces.
            label_selector: Label selector for filtering.

        Returns:
            Result with pod list.
        """
        if namespace:
            path = f"/api/v1/namespaces/{namespace}/pods"
        else:
            path = "/api/v1/pods"

        if label_selector:
            path += f"?labelSelector={label_selector}"

        result = self._make_request("GET", path)

        if result.success and result.data:
            items = result.data.get("items", [])
            pods = [K8sPod.from_api_response(pod).to_dict() for pod in items]
            result.data = {"pods": pods}

        return result

    def get_pod(self, name: str, namespace: str) -> K8sResult:
        """Get a specific pod."""
        result = self._make_request("GET", f"/api/v1/namespaces/{namespace}/pods/{name}")

        if result.success and result.data:
            result.data = {"pod": K8sPod.from_api_response(result.data).to_dict()}

        return result

    def delete_pod(self, name: str, namespace: str) -> K8sResult:
        """Delete a pod."""
        return self._make_request("DELETE", f"/api/v1/namespaces/{namespace}/pods/{name}")

    def get_pod_logs(
        self,
        name: str,
        namespace: str,
        container: Optional[str] = None,
        tail_lines: Optional[int] = None,
    ) -> K8sResult:
        """Get pod logs.

        Args:
            name: Pod name.
            namespace: Namespace.
            container: Container name (required if pod has multiple containers).
            tail_lines: Number of lines from end to return.

        Returns:
            Result with logs.
        """
        path = f"/api/v1/namespaces/{namespace}/pods/{name}/log"

        params = []
        if container:
            params.append(f"container={container}")
        if tail_lines:
            params.append(f"tailLines={tail_lines}")

        if params:
            path += "?" + "&".join(params)

        result = self._make_request("GET", path)

        if result.success:
            # Logs are returned as plain text, not JSON
            result.data = {"logs": result.data}

        return result

    # Deployment operations

    def list_deployments(
        self,
        namespace: Optional[str] = None,
        label_selector: Optional[str] = None,
    ) -> K8sResult:
        """List deployments."""
        if namespace:
            path = f"/apis/apps/v1/namespaces/{namespace}/deployments"
        else:
            path = "/apis/apps/v1/deployments"

        if label_selector:
            path += f"?labelSelector={label_selector}"

        result = self._make_request("GET", path)

        if result.success and result.data:
            items = result.data.get("items", [])
            deployments = [K8sDeployment.from_api_response(d).to_dict() for d in items]
            result.data = {"deployments": deployments}

        return result

    def get_deployment(self, name: str, namespace: str) -> K8sResult:
        """Get a specific deployment."""
        result = self._make_request(
            "GET",
            f"/apis/apps/v1/namespaces/{namespace}/deployments/{name}",
        )

        if result.success and result.data:
            result.data = {"deployment": K8sDeployment.from_api_response(result.data).to_dict()}

        return result

    def scale_deployment(self, name: str, namespace: str, replicas: int) -> K8sResult:
        """Scale a deployment.

        Args:
            name: Deployment name.
            namespace: Namespace.
            replicas: Desired number of replicas.

        Returns:
            Result.
        """
        patch_data = {
            "spec": {
                "replicas": replicas,
            }
        }

        # Use strategic merge patch
        headers = {"Content-Type": "application/strategic-merge-patch+json"}

        url = f"{self.config.api_server.rstrip('/')}/apis/apps/v1/namespaces/{namespace}/deployments/{name}"

        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(patch_data).encode("utf-8"),
                headers=headers,
                method="PATCH",
            )

            with urllib.request.urlopen(
                req,
                timeout=self.config.timeout,
                context=self._ssl_context,
            ) as response:
                response_body = response.read().decode("utf-8")
                response_data = json.loads(response_body) if response_body else None

                return K8sResult(
                    success=True,
                    data={"replicas": replicas},
                    status_code=response.getcode(),
                )

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass

            return K8sResult(
                success=False,
                error=f"HTTP {e.code}: {error_body or e.reason}",
                status_code=e.code,
            )
        except Exception as e:
            return K8sResult(
                success=False,
                error=str(e),
            )

    # Service operations

    def list_services(
        self,
        namespace: Optional[str] = None,
        label_selector: Optional[str] = None,
    ) -> K8sResult:
        """List services."""
        if namespace:
            path = f"/api/v1/namespaces/{namespace}/services"
        else:
            path = "/api/v1/services"

        if label_selector:
            path += f"?labelSelector={label_selector}"

        result = self._make_request("GET", path)

        if result.success and result.data:
            items = result.data.get("items", [])
            services = [K8sService.from_api_response(s).to_dict() for s in items]
            result.data = {"services": services}

        return result

    def get_service(self, name: str, namespace: str) -> K8sResult:
        """Get a specific service."""
        result = self._make_request(
            "GET",
            f"/api/v1/namespaces/{namespace}/services/{name}",
        )

        if result.success and result.data:
            result.data = {"service": K8sService.from_api_response(result.data).to_dict()}

        return result

    # ConfigMap operations

    def list_configmaps(
        self,
        namespace: Optional[str] = None,
        label_selector: Optional[str] = None,
    ) -> K8sResult:
        """List ConfigMaps."""
        if namespace:
            path = f"/api/v1/namespaces/{namespace}/configmaps"
        else:
            path = "/api/v1/configmaps"

        if label_selector:
            path += f"?labelSelector={label_selector}"

        result = self._make_request("GET", path)

        if result.success and result.data:
            items = result.data.get("items", [])
            configmaps = [K8sConfigMap.from_api_response(cm).to_dict() for cm in items]
            result.data = {"configmaps": configmaps}

        return result

    def get_configmap(self, name: str, namespace: str) -> K8sResult:
        """Get a specific ConfigMap."""
        result = self._make_request(
            "GET",
            f"/api/v1/namespaces/{namespace}/configmaps/{name}",
        )

        if result.success and result.data:
            result.data = {"configmap": K8sConfigMap.from_api_response(result.data).to_dict()}

        return result


class K8sManager:
    """Manager for Kubernetes clients."""

    def __init__(self):
        """Initialize manager."""
        self._clients: Dict[str, K8sClient] = {}

    def add_client(self, name: str, client: K8sClient) -> None:
        """Add a Kubernetes client."""
        self._clients[name] = client

    def get_client(self, name: str) -> Optional[K8sClient]:
        """Get a Kubernetes client."""
        return self._clients.get(name)

    def remove_client(self, name: str) -> bool:
        """Remove a Kubernetes client."""
        if name in self._clients:
            del self._clients[name]
            return True
        return False

    def list_clients(self) -> List[str]:
        """List all client names."""
        return list(self._clients.keys())


# Input/Output Models

class CreateK8sClientInput(BaseModel):
    """Input for creating a Kubernetes client."""

    name: str = Field(..., description="Name for the client")
    api_server: str = Field(..., description="Kubernetes API server URL")
    token: Optional[str] = Field(default=None, description="Bearer token for authentication")
    insecure_skip_tls_verify: bool = Field(default=False, description="Skip TLS verification")
    default_namespace: str = Field(default="default", description="Default namespace")


class CreateK8sClientOutput(BaseModel):
    """Output from creating a Kubernetes client."""

    success: bool = Field(description="Whether client was created")
    name: str = Field(description="Client name")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ListPodsInput(BaseModel):
    """Input for listing pods."""

    client: str = Field(default="default", description="Kubernetes client name")
    namespace: Optional[str] = Field(default=None, description="Namespace (None for all)")
    label_selector: Optional[str] = Field(default=None, description="Label selector")


class K8sPodsOutput(BaseModel):
    """Output containing pod list."""

    success: bool = Field(description="Whether operation succeeded")
    pods: Optional[List[Dict[str, Any]]] = Field(default=None, description="Pod list")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class GetPodInput(BaseModel):
    """Input for getting a specific pod."""

    client: str = Field(default="default", description="Kubernetes client name")
    name: str = Field(..., description="Pod name")
    namespace: str = Field(default="default", description="Namespace")


class GetPodLogsInput(BaseModel):
    """Input for getting pod logs."""

    client: str = Field(default="default", description="Kubernetes client name")
    name: str = Field(..., description="Pod name")
    namespace: str = Field(default="default", description="Namespace")
    container: Optional[str] = Field(default=None, description="Container name")
    tail_lines: Optional[int] = Field(default=100, description="Number of lines from end")


class K8sLogsOutput(BaseModel):
    """Output containing logs."""

    success: bool = Field(description="Whether operation succeeded")
    logs: Optional[str] = Field(default=None, description="Log content")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class DeletePodInput(BaseModel):
    """Input for deleting a pod."""

    client: str = Field(default="default", description="Kubernetes client name")
    name: str = Field(..., description="Pod name")
    namespace: str = Field(default="default", description="Namespace")


class K8sSimpleOutput(BaseModel):
    """Simple output for operations without data."""

    success: bool = Field(description="Whether operation succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ListDeploymentsInput(BaseModel):
    """Input for listing deployments."""

    client: str = Field(default="default", description="Kubernetes client name")
    namespace: Optional[str] = Field(default=None, description="Namespace (None for all)")
    label_selector: Optional[str] = Field(default=None, description="Label selector")


class K8sDeploymentsOutput(BaseModel):
    """Output containing deployment list."""

    success: bool = Field(description="Whether operation succeeded")
    deployments: Optional[List[Dict[str, Any]]] = Field(default=None, description="Deployment list")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class GetDeploymentInput(BaseModel):
    """Input for getting a specific deployment."""

    client: str = Field(default="default", description="Kubernetes client name")
    name: str = Field(..., description="Deployment name")
    namespace: str = Field(default="default", description="Namespace")


class ScaleDeploymentInput(BaseModel):
    """Input for scaling a deployment."""

    client: str = Field(default="default", description="Kubernetes client name")
    name: str = Field(..., description="Deployment name")
    namespace: str = Field(default="default", description="Namespace")
    replicas: int = Field(..., description="Desired number of replicas")


class ListServicesInput(BaseModel):
    """Input for listing services."""

    client: str = Field(default="default", description="Kubernetes client name")
    namespace: Optional[str] = Field(default=None, description="Namespace (None for all)")
    label_selector: Optional[str] = Field(default=None, description="Label selector")


class K8sServicesOutput(BaseModel):
    """Output containing service list."""

    success: bool = Field(description="Whether operation succeeded")
    services: Optional[List[Dict[str, Any]]] = Field(default=None, description="Service list")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class GetServiceInput(BaseModel):
    """Input for getting a specific service."""

    client: str = Field(default="default", description="Kubernetes client name")
    name: str = Field(..., description="Service name")
    namespace: str = Field(default="default", description="Namespace")


class ListNamespacesInput(BaseModel):
    """Input for listing namespaces."""

    client: str = Field(default="default", description="Kubernetes client name")


class K8sNamespacesOutput(BaseModel):
    """Output containing namespace list."""

    success: bool = Field(description="Whether operation succeeded")
    namespaces: Optional[List[Dict[str, Any]]] = Field(default=None, description="Namespace list")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class GetNamespaceInput(BaseModel):
    """Input for getting a specific namespace."""

    client: str = Field(default="default", description="Kubernetes client name")
    name: str = Field(..., description="Namespace name")


class ListConfigMapsInput(BaseModel):
    """Input for listing ConfigMaps."""

    client: str = Field(default="default", description="Kubernetes client name")
    namespace: Optional[str] = Field(default=None, description="Namespace (None for all)")
    label_selector: Optional[str] = Field(default=None, description="Label selector")


class K8sConfigMapsOutput(BaseModel):
    """Output containing ConfigMap list."""

    success: bool = Field(description="Whether operation succeeded")
    configmaps: Optional[List[Dict[str, Any]]] = Field(default=None, description="ConfigMap list")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class GetConfigMapInput(BaseModel):
    """Input for getting a specific ConfigMap."""

    client: str = Field(default="default", description="Kubernetes client name")
    name: str = Field(..., description="ConfigMap name")
    namespace: str = Field(default="default", description="Namespace")


# Tools

class CreateK8sClientTool(BaseTool[CreateK8sClientInput, CreateK8sClientOutput]):
    """Tool for creating a Kubernetes client."""

    metadata = ToolMetadata(
        id="create_k8s_client",
        name="Create Kubernetes Client",
        description="Create a Kubernetes API client with credentials",
        category="utility",
    )
    input_type = CreateK8sClientInput
    output_type = CreateK8sClientOutput

    def __init__(self, manager: K8sManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: CreateK8sClientInput) -> CreateK8sClientOutput:
        """Create a Kubernetes client."""
        config = K8sConfig(
            api_server=input.api_server,
            token=input.token,
            insecure_skip_tls_verify=input.insecure_skip_tls_verify,
            default_namespace=input.default_namespace,
        )
        client = K8sClient(config)
        self.manager.add_client(input.name, client)

        return CreateK8sClientOutput(
            success=True,
            name=input.name,
        )


class ListPodsTool(BaseTool[ListPodsInput, K8sPodsOutput]):
    """Tool for listing pods."""

    metadata = ToolMetadata(
        id="list_k8s_pods",
        name="List Kubernetes Pods",
        description="List pods in a Kubernetes cluster",
        category="utility",
    )
    input_type = ListPodsInput
    output_type = K8sPodsOutput

    def __init__(self, manager: K8sManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: ListPodsInput) -> K8sPodsOutput:
        """List pods."""
        client = self.manager.get_client(input.client)

        if not client:
            return K8sPodsOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.list_pods(
            namespace=input.namespace,
            label_selector=input.label_selector,
        )

        if result.success:
            return K8sPodsOutput(success=True, pods=result.data.get("pods"))
        return K8sPodsOutput(success=False, error=result.error)


class GetPodTool(BaseTool[GetPodInput, K8sPodsOutput]):
    """Tool for getting a specific pod."""

    metadata = ToolMetadata(
        id="get_k8s_pod",
        name="Get Kubernetes Pod",
        description="Get details of a specific pod",
        category="utility",
    )
    input_type = GetPodInput
    output_type = K8sPodsOutput

    def __init__(self, manager: K8sManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: GetPodInput) -> K8sPodsOutput:
        """Get a pod."""
        client = self.manager.get_client(input.client)

        if not client:
            return K8sPodsOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.get_pod(input.name, input.namespace)

        if result.success:
            pod = result.data.get("pod")
            return K8sPodsOutput(success=True, pods=[pod] if pod else None)
        return K8sPodsOutput(success=False, error=result.error)


class GetPodLogsTool(BaseTool[GetPodLogsInput, K8sLogsOutput]):
    """Tool for getting pod logs."""

    metadata = ToolMetadata(
        id="get_k8s_pod_logs",
        name="Get Kubernetes Pod Logs",
        description="Get logs from a pod",
        category="utility",
    )
    input_type = GetPodLogsInput
    output_type = K8sLogsOutput

    def __init__(self, manager: K8sManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: GetPodLogsInput) -> K8sLogsOutput:
        """Get pod logs."""
        client = self.manager.get_client(input.client)

        if not client:
            return K8sLogsOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.get_pod_logs(
            name=input.name,
            namespace=input.namespace,
            container=input.container,
            tail_lines=input.tail_lines,
        )

        if result.success:
            logs = result.data.get("logs") if isinstance(result.data, dict) else result.data
            return K8sLogsOutput(success=True, logs=str(logs) if logs else "")
        return K8sLogsOutput(success=False, error=result.error)


class DeletePodTool(BaseTool[DeletePodInput, K8sSimpleOutput]):
    """Tool for deleting a pod."""

    metadata = ToolMetadata(
        id="delete_k8s_pod",
        name="Delete Kubernetes Pod",
        description="Delete a pod from the cluster",
        category="utility",
    )
    input_type = DeletePodInput
    output_type = K8sSimpleOutput

    def __init__(self, manager: K8sManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: DeletePodInput) -> K8sSimpleOutput:
        """Delete a pod."""
        client = self.manager.get_client(input.client)

        if not client:
            return K8sSimpleOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.delete_pod(input.name, input.namespace)

        return K8sSimpleOutput(success=result.success, error=result.error)


class ListDeploymentsTool(BaseTool[ListDeploymentsInput, K8sDeploymentsOutput]):
    """Tool for listing deployments."""

    metadata = ToolMetadata(
        id="list_k8s_deployments",
        name="List Kubernetes Deployments",
        description="List deployments in a Kubernetes cluster",
        category="utility",
    )
    input_type = ListDeploymentsInput
    output_type = K8sDeploymentsOutput

    def __init__(self, manager: K8sManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: ListDeploymentsInput) -> K8sDeploymentsOutput:
        """List deployments."""
        client = self.manager.get_client(input.client)

        if not client:
            return K8sDeploymentsOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.list_deployments(
            namespace=input.namespace,
            label_selector=input.label_selector,
        )

        if result.success:
            return K8sDeploymentsOutput(success=True, deployments=result.data.get("deployments"))
        return K8sDeploymentsOutput(success=False, error=result.error)


class GetDeploymentTool(BaseTool[GetDeploymentInput, K8sDeploymentsOutput]):
    """Tool for getting a specific deployment."""

    metadata = ToolMetadata(
        id="get_k8s_deployment",
        name="Get Kubernetes Deployment",
        description="Get details of a specific deployment",
        category="utility",
    )
    input_type = GetDeploymentInput
    output_type = K8sDeploymentsOutput

    def __init__(self, manager: K8sManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: GetDeploymentInput) -> K8sDeploymentsOutput:
        """Get a deployment."""
        client = self.manager.get_client(input.client)

        if not client:
            return K8sDeploymentsOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.get_deployment(input.name, input.namespace)

        if result.success:
            deployment = result.data.get("deployment")
            return K8sDeploymentsOutput(success=True, deployments=[deployment] if deployment else None)
        return K8sDeploymentsOutput(success=False, error=result.error)


class ScaleDeploymentTool(BaseTool[ScaleDeploymentInput, K8sSimpleOutput]):
    """Tool for scaling a deployment."""

    metadata = ToolMetadata(
        id="scale_k8s_deployment",
        name="Scale Kubernetes Deployment",
        description="Scale a deployment to a specific number of replicas",
        category="execution",
    )
    input_type = ScaleDeploymentInput
    output_type = K8sSimpleOutput

    def __init__(self, manager: K8sManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: ScaleDeploymentInput) -> K8sSimpleOutput:
        """Scale a deployment."""
        client = self.manager.get_client(input.client)

        if not client:
            return K8sSimpleOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.scale_deployment(input.name, input.namespace, input.replicas)

        return K8sSimpleOutput(success=result.success, error=result.error)


class ListServicesTool(BaseTool[ListServicesInput, K8sServicesOutput]):
    """Tool for listing services."""

    metadata = ToolMetadata(
        id="list_k8s_services",
        name="List Kubernetes Services",
        description="List services in a Kubernetes cluster",
        category="utility",
    )
    input_type = ListServicesInput
    output_type = K8sServicesOutput

    def __init__(self, manager: K8sManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: ListServicesInput) -> K8sServicesOutput:
        """List services."""
        client = self.manager.get_client(input.client)

        if not client:
            return K8sServicesOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.list_services(
            namespace=input.namespace,
            label_selector=input.label_selector,
        )

        if result.success:
            return K8sServicesOutput(success=True, services=result.data.get("services"))
        return K8sServicesOutput(success=False, error=result.error)


class GetServiceTool(BaseTool[GetServiceInput, K8sServicesOutput]):
    """Tool for getting a specific service."""

    metadata = ToolMetadata(
        id="get_k8s_service",
        name="Get Kubernetes Service",
        description="Get details of a specific service",
        category="utility",
    )
    input_type = GetServiceInput
    output_type = K8sServicesOutput

    def __init__(self, manager: K8sManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: GetServiceInput) -> K8sServicesOutput:
        """Get a service."""
        client = self.manager.get_client(input.client)

        if not client:
            return K8sServicesOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.get_service(input.name, input.namespace)

        if result.success:
            service = result.data.get("service")
            return K8sServicesOutput(success=True, services=[service] if service else None)
        return K8sServicesOutput(success=False, error=result.error)


class ListNamespacesTool(BaseTool[ListNamespacesInput, K8sNamespacesOutput]):
    """Tool for listing namespaces."""

    metadata = ToolMetadata(
        id="list_k8s_namespaces",
        name="List Kubernetes Namespaces",
        description="List namespaces in a Kubernetes cluster",
        category="utility",
    )
    input_type = ListNamespacesInput
    output_type = K8sNamespacesOutput

    def __init__(self, manager: K8sManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: ListNamespacesInput) -> K8sNamespacesOutput:
        """List namespaces."""
        client = self.manager.get_client(input.client)

        if not client:
            return K8sNamespacesOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.list_namespaces()

        if result.success:
            return K8sNamespacesOutput(success=True, namespaces=result.data.get("namespaces"))
        return K8sNamespacesOutput(success=False, error=result.error)


class GetNamespaceTool(BaseTool[GetNamespaceInput, K8sNamespacesOutput]):
    """Tool for getting a specific namespace."""

    metadata = ToolMetadata(
        id="get_k8s_namespace",
        name="Get Kubernetes Namespace",
        description="Get details of a specific namespace",
        category="utility",
    )
    input_type = GetNamespaceInput
    output_type = K8sNamespacesOutput

    def __init__(self, manager: K8sManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: GetNamespaceInput) -> K8sNamespacesOutput:
        """Get a namespace."""
        client = self.manager.get_client(input.client)

        if not client:
            return K8sNamespacesOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.get_namespace(input.name)

        if result.success:
            namespace = result.data.get("namespace")
            return K8sNamespacesOutput(success=True, namespaces=[namespace] if namespace else None)
        return K8sNamespacesOutput(success=False, error=result.error)


class ListConfigMapsTool(BaseTool[ListConfigMapsInput, K8sConfigMapsOutput]):
    """Tool for listing ConfigMaps."""

    metadata = ToolMetadata(
        id="list_k8s_configmaps",
        name="List Kubernetes ConfigMaps",
        description="List ConfigMaps in a Kubernetes cluster",
        category="utility",
    )
    input_type = ListConfigMapsInput
    output_type = K8sConfigMapsOutput

    def __init__(self, manager: K8sManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: ListConfigMapsInput) -> K8sConfigMapsOutput:
        """List ConfigMaps."""
        client = self.manager.get_client(input.client)

        if not client:
            return K8sConfigMapsOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.list_configmaps(
            namespace=input.namespace,
            label_selector=input.label_selector,
        )

        if result.success:
            return K8sConfigMapsOutput(success=True, configmaps=result.data.get("configmaps"))
        return K8sConfigMapsOutput(success=False, error=result.error)


class GetConfigMapTool(BaseTool[GetConfigMapInput, K8sConfigMapsOutput]):
    """Tool for getting a specific ConfigMap."""

    metadata = ToolMetadata(
        id="get_k8s_configmap",
        name="Get Kubernetes ConfigMap",
        description="Get details of a specific ConfigMap",
        category="utility",
    )
    input_type = GetConfigMapInput
    output_type = K8sConfigMapsOutput

    def __init__(self, manager: K8sManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: GetConfigMapInput) -> K8sConfigMapsOutput:
        """Get a ConfigMap."""
        client = self.manager.get_client(input.client)

        if not client:
            return K8sConfigMapsOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.get_configmap(input.name, input.namespace)

        if result.success:
            configmap = result.data.get("configmap")
            return K8sConfigMapsOutput(success=True, configmaps=[configmap] if configmap else None)
        return K8sConfigMapsOutput(success=False, error=result.error)


# Helper functions

def create_k8s_config(
    api_server: str,
    token: Optional[str] = None,
    insecure_skip_tls_verify: bool = False,
    default_namespace: str = "default",
) -> K8sConfig:
    """Create a Kubernetes configuration."""
    return K8sConfig(
        api_server=api_server,
        token=token,
        insecure_skip_tls_verify=insecure_skip_tls_verify,
        default_namespace=default_namespace,
    )


def create_k8s_client(config: K8sConfig) -> K8sClient:
    """Create a Kubernetes client."""
    return K8sClient(config)


def create_k8s_manager() -> K8sManager:
    """Create a Kubernetes manager."""
    return K8sManager()


def create_k8s_tools(manager: K8sManager) -> Dict[str, BaseTool]:
    """Create all Kubernetes tools."""
    return {
        "create_k8s_client": CreateK8sClientTool(manager),
        # Pods
        "list_k8s_pods": ListPodsTool(manager),
        "get_k8s_pod": GetPodTool(manager),
        "get_k8s_pod_logs": GetPodLogsTool(manager),
        "delete_k8s_pod": DeletePodTool(manager),
        # Deployments
        "list_k8s_deployments": ListDeploymentsTool(manager),
        "get_k8s_deployment": GetDeploymentTool(manager),
        "scale_k8s_deployment": ScaleDeploymentTool(manager),
        # Services
        "list_k8s_services": ListServicesTool(manager),
        "get_k8s_service": GetServiceTool(manager),
        # Namespaces
        "list_k8s_namespaces": ListNamespacesTool(manager),
        "get_k8s_namespace": GetNamespaceTool(manager),
        # ConfigMaps
        "list_k8s_configmaps": ListConfigMapsTool(manager),
        "get_k8s_configmap": GetConfigMapTool(manager),
    }
