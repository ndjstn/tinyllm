"""Docker tools for TinyLLM.

Provides tools for interacting with Docker daemon including:
- Container management (list, get, start, stop, remove, logs)
- Image management (list, get, pull, remove)
- Volume management (list, create, remove)
- Network management (list, get)
"""

import json
import logging
import socket
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.tools.base import BaseTool, ToolMetadata

logger = logging.getLogger(__name__)


class ContainerState(str, Enum):
    """Container state enum."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    RESTARTING = "restarting"
    REMOVING = "removing"
    EXITED = "exited"
    DEAD = "dead"


@dataclass
class DockerConfig:
    """Docker daemon configuration."""

    host: str = "unix:///var/run/docker.sock"
    timeout: int = 30
    api_version: str = "v1.43"
    tls: bool = False
    cert_path: Optional[str] = None
    key_path: Optional[str] = None
    ca_path: Optional[str] = None


@dataclass
class DockerContainer:
    """Docker container representation."""

    id: str
    name: str
    image: str
    status: str
    state: str
    created: str
    ports: List[Dict[str, Any]] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "DockerContainer":
        """Create from Docker API response."""
        names = data.get("Names", [])
        name = names[0].lstrip("/") if names else ""

        ports = []
        for port in data.get("Ports", []):
            ports.append({
                "private_port": port.get("PrivatePort"),
                "public_port": port.get("PublicPort"),
                "type": port.get("Type"),
                "ip": port.get("IP"),
            })

        return cls(
            id=data.get("Id", "")[:12],
            name=name,
            image=data.get("Image", ""),
            status=data.get("Status", ""),
            state=data.get("State", ""),
            created=str(data.get("Created", "")),
            ports=ports,
            labels=data.get("Labels", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "image": self.image,
            "status": self.status,
            "state": self.state,
            "created": self.created,
            "ports": self.ports,
            "labels": self.labels,
        }


@dataclass
class DockerImage:
    """Docker image representation."""

    id: str
    repo_tags: List[str] = field(default_factory=list)
    size: int = 0
    created: str = ""
    labels: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "DockerImage":
        """Create from Docker API response."""
        return cls(
            id=data.get("Id", "").replace("sha256:", "")[:12],
            repo_tags=data.get("RepoTags") or [],
            size=data.get("Size", 0),
            created=str(data.get("Created", "")),
            labels=data.get("Labels") or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "repo_tags": self.repo_tags,
            "size": self.size,
            "size_mb": round(self.size / (1024 * 1024), 2),
            "created": self.created,
            "labels": self.labels,
        }


@dataclass
class DockerVolume:
    """Docker volume representation."""

    name: str
    driver: str
    mountpoint: str
    created: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    scope: str = "local"

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "DockerVolume":
        """Create from Docker API response."""
        return cls(
            name=data.get("Name", ""),
            driver=data.get("Driver", "local"),
            mountpoint=data.get("Mountpoint", ""),
            created=data.get("CreatedAt", ""),
            labels=data.get("Labels") or {},
            scope=data.get("Scope", "local"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "driver": self.driver,
            "mountpoint": self.mountpoint,
            "created": self.created,
            "labels": self.labels,
            "scope": self.scope,
        }


@dataclass
class DockerNetwork:
    """Docker network representation."""

    id: str
    name: str
    driver: str
    scope: str
    internal: bool = False
    ipam_config: List[Dict[str, Any]] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "DockerNetwork":
        """Create from Docker API response."""
        ipam = data.get("IPAM", {})
        ipam_config = ipam.get("Config", [])

        return cls(
            id=data.get("Id", "")[:12],
            name=data.get("Name", ""),
            driver=data.get("Driver", ""),
            scope=data.get("Scope", ""),
            internal=data.get("Internal", False),
            ipam_config=ipam_config,
            labels=data.get("Labels") or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "driver": self.driver,
            "scope": self.scope,
            "internal": self.internal,
            "ipam_config": self.ipam_config,
            "labels": self.labels,
        }


@dataclass
class DockerResult:
    """Result from Docker API operation."""

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


class UnixHTTPHandler(urllib.request.AbstractHTTPHandler):
    """HTTP handler for Unix sockets."""

    def __init__(self, socket_path: str):
        """Initialize handler."""
        super().__init__()
        self.socket_path = socket_path

    def http_open(self, req: urllib.request.Request) -> Any:
        """Open HTTP connection over Unix socket."""
        return self.do_open(self._get_connection, req)

    def _get_connection(self, host: str, timeout: float = 30) -> Any:
        """Get connection to Unix socket."""
        return UnixHTTPConnection(self.socket_path, timeout=timeout)


class UnixHTTPConnection:
    """HTTP connection over Unix socket."""

    def __init__(self, socket_path: str, timeout: float = 30):
        """Initialize connection."""
        self.socket_path = socket_path
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self._response: Optional[bytes] = None

    def connect(self) -> None:
        """Connect to Unix socket."""
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)

    def request(
        self,
        method: str,
        url: str,
        body: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Send HTTP request."""
        if self.sock is None:
            self.connect()

        headers = headers or {}
        if "Host" not in headers:
            headers["Host"] = "localhost"

        request_line = f"{method} {url} HTTP/1.1\r\n"
        header_lines = "".join(f"{k}: {v}\r\n" for k, v in headers.items())

        if body:
            header_lines += f"Content-Length: {len(body)}\r\n"

        request_bytes = (request_line + header_lines + "\r\n").encode()
        if body:
            request_bytes += body

        self.sock.sendall(request_bytes)

    def getresponse(self) -> "UnixHTTPResponse":
        """Get HTTP response."""
        if self.sock is None:
            raise RuntimeError("Not connected")

        response_data = b""
        while True:
            chunk = self.sock.recv(4096)
            if not chunk:
                break
            response_data += chunk
            if b"\r\n\r\n" in response_data:
                header_end = response_data.index(b"\r\n\r\n")
                headers = response_data[:header_end].decode()
                if "Transfer-Encoding: chunked" not in headers:
                    content_length = 0
                    for line in headers.split("\r\n"):
                        if line.lower().startswith("content-length:"):
                            content_length = int(line.split(":")[1].strip())
                            break
                    body = response_data[header_end + 4:]
                    while len(body) < content_length:
                        chunk = self.sock.recv(4096)
                        if not chunk:
                            break
                        body += chunk
                    break

        return UnixHTTPResponse(response_data)

    def close(self) -> None:
        """Close connection."""
        if self.sock:
            self.sock.close()
            self.sock = None


class UnixHTTPResponse:
    """HTTP response from Unix socket."""

    def __init__(self, data: bytes):
        """Initialize response."""
        self.data = data
        self._parse_response()

    def _parse_response(self) -> None:
        """Parse HTTP response."""
        if b"\r\n\r\n" not in self.data:
            self.status = 0
            self.reason = "Invalid response"
            self.body = b""
            return

        header_end = self.data.index(b"\r\n\r\n")
        header_section = self.data[:header_end].decode()
        self.body = self.data[header_end + 4:]

        lines = header_section.split("\r\n")
        status_line = lines[0]
        parts = status_line.split(" ", 2)
        self.status = int(parts[1]) if len(parts) > 1 else 0
        self.reason = parts[2] if len(parts) > 2 else ""

        # Handle chunked encoding
        if "Transfer-Encoding: chunked" in header_section:
            self.body = self._decode_chunked(self.body)

    def _decode_chunked(self, data: bytes) -> bytes:
        """Decode chunked transfer encoding."""
        result = b""
        pos = 0
        while pos < len(data):
            line_end = data.find(b"\r\n", pos)
            if line_end == -1:
                break
            chunk_size_hex = data[pos:line_end].decode().strip()
            if not chunk_size_hex:
                pos = line_end + 2
                continue
            try:
                chunk_size = int(chunk_size_hex, 16)
            except ValueError:
                break
            if chunk_size == 0:
                break
            chunk_start = line_end + 2
            chunk_end = chunk_start + chunk_size
            result += data[chunk_start:chunk_end]
            pos = chunk_end + 2
        return result

    def read(self) -> bytes:
        """Read response body."""
        return self.body

    def getcode(self) -> int:
        """Get status code."""
        return self.status


class DockerClient:
    """Docker API client."""

    def __init__(self, config: DockerConfig):
        """Initialize client.

        Args:
            config: Docker configuration.
        """
        self.config = config
        self._is_unix_socket = config.host.startswith("unix://")
        if self._is_unix_socket:
            self._socket_path = config.host.replace("unix://", "")

    def _make_request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> DockerResult:
        """Make HTTP request to Docker API.

        Args:
            method: HTTP method.
            path: API path.
            data: Request body.
            params: Query parameters.

        Returns:
            Docker result.
        """
        api_path = f"/{self.config.api_version}{path}"
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items())
            api_path += f"?{query}"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Host": "localhost",
        }

        body = None
        if data:
            body = json.dumps(data).encode("utf-8")

        try:
            if self._is_unix_socket:
                return self._unix_socket_request(method, api_path, body, headers)
            else:
                return self._http_request(method, api_path, body, headers)

        except Exception as e:
            return DockerResult(
                success=False,
                error=str(e),
            )

    def _unix_socket_request(
        self,
        method: str,
        path: str,
        body: Optional[bytes],
        headers: Dict[str, str],
    ) -> DockerResult:
        """Make request over Unix socket."""
        conn = UnixHTTPConnection(self._socket_path, timeout=self.config.timeout)
        try:
            conn.connect()
            conn.request(method, path, body, headers)
            response = conn.getresponse()

            response_body = response.read()
            status_code = response.getcode()

            if status_code >= 400:
                try:
                    error_data = json.loads(response_body.decode("utf-8"))
                    error_msg = error_data.get("message", response_body.decode("utf-8"))
                except Exception:
                    error_msg = response_body.decode("utf-8") if response_body else f"HTTP {status_code}"

                return DockerResult(
                    success=False,
                    error=error_msg,
                    status_code=status_code,
                )

            response_data = None
            if response_body:
                try:
                    response_data = json.loads(response_body.decode("utf-8"))
                except json.JSONDecodeError:
                    response_data = response_body.decode("utf-8")

            return DockerResult(
                success=True,
                data=response_data,
                status_code=status_code,
            )

        finally:
            conn.close()

    def _http_request(
        self,
        method: str,
        path: str,
        body: Optional[bytes],
        headers: Dict[str, str],
    ) -> DockerResult:
        """Make HTTP request to TCP host."""
        url = f"http://{self.config.host.replace('tcp://', '')}{path}"

        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers=headers,
                method=method,
            )

            with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                response_body = response.read().decode("utf-8")
                response_data = json.loads(response_body) if response_body else None

                return DockerResult(
                    success=True,
                    data=response_data,
                    status_code=response.getcode(),
                )

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            try:
                error_data = json.loads(error_body)
                error_msg = error_data.get("message", error_body)
            except Exception:
                error_msg = error_body or str(e.reason)

            return DockerResult(
                success=False,
                error=error_msg,
                status_code=e.code,
            )
        except urllib.error.URLError as e:
            return DockerResult(
                success=False,
                error=f"Connection error: {e.reason}",
            )

    # Container operations

    def list_containers(
        self,
        all_containers: bool = False,
        filters: Optional[Dict[str, List[str]]] = None,
    ) -> DockerResult:
        """List containers.

        Args:
            all_containers: Include stopped containers.
            filters: Filter containers.

        Returns:
            Result with container list.
        """
        params = {}
        if all_containers:
            params["all"] = "true"
        if filters:
            params["filters"] = json.dumps(filters)

        result = self._make_request("GET", "/containers/json", params=params)

        if result.success and result.data:
            containers = [DockerContainer.from_api_response(c).to_dict() for c in result.data]
            result.data = {"containers": containers}

        return result

    def get_container(self, container_id: str) -> DockerResult:
        """Get container details."""
        result = self._make_request("GET", f"/containers/{container_id}/json")

        if result.success and result.data:
            # Convert inspect response to container format
            data = result.data
            container = {
                "id": data.get("Id", "")[:12],
                "name": data.get("Name", "").lstrip("/"),
                "image": data.get("Config", {}).get("Image", ""),
                "status": data.get("State", {}).get("Status", ""),
                "state": data.get("State", {}).get("Status", ""),
                "created": data.get("Created", ""),
                "ports": [],
                "labels": data.get("Config", {}).get("Labels", {}),
            }
            result.data = {"container": container}

        return result

    def start_container(self, container_id: str) -> DockerResult:
        """Start a container."""
        return self._make_request("POST", f"/containers/{container_id}/start")

    def stop_container(self, container_id: str, timeout: int = 10) -> DockerResult:
        """Stop a container."""
        return self._make_request(
            "POST",
            f"/containers/{container_id}/stop",
            params={"t": str(timeout)},
        )

    def remove_container(self, container_id: str, force: bool = False) -> DockerResult:
        """Remove a container."""
        params = {}
        if force:
            params["force"] = "true"
        return self._make_request("DELETE", f"/containers/{container_id}", params=params)

    def get_container_logs(
        self,
        container_id: str,
        tail: int = 100,
        timestamps: bool = False,
    ) -> DockerResult:
        """Get container logs."""
        params = {
            "stdout": "true",
            "stderr": "true",
            "tail": str(tail),
        }
        if timestamps:
            params["timestamps"] = "true"

        result = self._make_request("GET", f"/containers/{container_id}/logs", params=params)

        if result.success:
            logs = result.data if isinstance(result.data, str) else ""
            result.data = {"logs": logs}

        return result

    # Image operations

    def list_images(self, all_images: bool = False) -> DockerResult:
        """List images."""
        params = {}
        if all_images:
            params["all"] = "true"

        result = self._make_request("GET", "/images/json", params=params)

        if result.success and result.data:
            images = [DockerImage.from_api_response(img).to_dict() for img in result.data]
            result.data = {"images": images}

        return result

    def get_image(self, image_id: str) -> DockerResult:
        """Get image details."""
        result = self._make_request("GET", f"/images/{image_id}/json")

        if result.success and result.data:
            image = DockerImage.from_api_response(result.data)
            result.data = {"image": image.to_dict()}

        return result

    def pull_image(self, image: str, tag: str = "latest") -> DockerResult:
        """Pull an image."""
        params = {"fromImage": image, "tag": tag}
        return self._make_request("POST", "/images/create", params=params)

    def remove_image(self, image_id: str, force: bool = False) -> DockerResult:
        """Remove an image."""
        params = {}
        if force:
            params["force"] = "true"
        return self._make_request("DELETE", f"/images/{image_id}", params=params)

    # Volume operations

    def list_volumes(self) -> DockerResult:
        """List volumes."""
        result = self._make_request("GET", "/volumes")

        if result.success and result.data:
            volumes_data = result.data.get("Volumes", [])
            volumes = [DockerVolume.from_api_response(v).to_dict() for v in volumes_data]
            result.data = {"volumes": volumes}

        return result

    def create_volume(
        self,
        name: str,
        driver: str = "local",
        labels: Optional[Dict[str, str]] = None,
    ) -> DockerResult:
        """Create a volume."""
        data = {
            "Name": name,
            "Driver": driver,
        }
        if labels:
            data["Labels"] = labels

        result = self._make_request("POST", "/volumes/create", data=data)

        if result.success and result.data:
            volume = DockerVolume.from_api_response(result.data)
            result.data = {"volume": volume.to_dict()}

        return result

    def remove_volume(self, name: str, force: bool = False) -> DockerResult:
        """Remove a volume."""
        params = {}
        if force:
            params["force"] = "true"
        return self._make_request("DELETE", f"/volumes/{name}", params=params)

    # Network operations

    def list_networks(self) -> DockerResult:
        """List networks."""
        result = self._make_request("GET", "/networks")

        if result.success and result.data:
            networks = [DockerNetwork.from_api_response(n).to_dict() for n in result.data]
            result.data = {"networks": networks}

        return result

    def get_network(self, network_id: str) -> DockerResult:
        """Get network details."""
        result = self._make_request("GET", f"/networks/{network_id}")

        if result.success and result.data:
            network = DockerNetwork.from_api_response(result.data)
            result.data = {"network": network.to_dict()}

        return result


class DockerManager:
    """Manager for Docker clients."""

    def __init__(self):
        """Initialize manager."""
        self._clients: Dict[str, DockerClient] = {}

    def add_client(self, name: str, client: DockerClient) -> None:
        """Add a Docker client."""
        self._clients[name] = client

    def get_client(self, name: str) -> Optional[DockerClient]:
        """Get a Docker client."""
        return self._clients.get(name)

    def remove_client(self, name: str) -> bool:
        """Remove a Docker client."""
        if name in self._clients:
            del self._clients[name]
            return True
        return False

    def list_clients(self) -> List[str]:
        """List all client names."""
        return list(self._clients.keys())


# Input/Output Models

class CreateDockerClientInput(BaseModel):
    """Input for creating a Docker client."""

    name: str = Field(..., description="Name for the client")
    host: str = Field(
        default="unix:///var/run/docker.sock",
        description="Docker host (unix:// or tcp://)",
    )
    timeout: int = Field(default=30, description="Request timeout in seconds")
    api_version: str = Field(default="v1.43", description="Docker API version")


class CreateDockerClientOutput(BaseModel):
    """Output from creating a Docker client."""

    success: bool = Field(description="Whether client was created")
    name: str = Field(description="Client name")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ListContainersInput(BaseModel):
    """Input for listing containers."""

    client: str = Field(default="default", description="Docker client name")
    all_containers: bool = Field(default=False, description="Include stopped containers")
    filter_status: Optional[str] = Field(default=None, description="Filter by status")


class DockerContainersOutput(BaseModel):
    """Output containing container list."""

    success: bool = Field(description="Whether operation succeeded")
    containers: Optional[List[Dict[str, Any]]] = Field(default=None, description="Container list")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class GetContainerInput(BaseModel):
    """Input for getting a container."""

    client: str = Field(default="default", description="Docker client name")
    container_id: str = Field(..., description="Container ID or name")


class StartContainerInput(BaseModel):
    """Input for starting a container."""

    client: str = Field(default="default", description="Docker client name")
    container_id: str = Field(..., description="Container ID or name")


class StopContainerInput(BaseModel):
    """Input for stopping a container."""

    client: str = Field(default="default", description="Docker client name")
    container_id: str = Field(..., description="Container ID or name")
    timeout: int = Field(default=10, description="Timeout before killing")


class RemoveContainerInput(BaseModel):
    """Input for removing a container."""

    client: str = Field(default="default", description="Docker client name")
    container_id: str = Field(..., description="Container ID or name")
    force: bool = Field(default=False, description="Force removal")


class GetContainerLogsInput(BaseModel):
    """Input for getting container logs."""

    client: str = Field(default="default", description="Docker client name")
    container_id: str = Field(..., description="Container ID or name")
    tail: int = Field(default=100, description="Number of lines from end")
    timestamps: bool = Field(default=False, description="Include timestamps")


class DockerLogsOutput(BaseModel):
    """Output containing logs."""

    success: bool = Field(description="Whether operation succeeded")
    logs: Optional[str] = Field(default=None, description="Log content")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class DockerSimpleOutput(BaseModel):
    """Simple output for operations without data."""

    success: bool = Field(description="Whether operation succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ListImagesInput(BaseModel):
    """Input for listing images."""

    client: str = Field(default="default", description="Docker client name")
    all_images: bool = Field(default=False, description="Include intermediate images")


class DockerImagesOutput(BaseModel):
    """Output containing image list."""

    success: bool = Field(description="Whether operation succeeded")
    images: Optional[List[Dict[str, Any]]] = Field(default=None, description="Image list")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class GetImageInput(BaseModel):
    """Input for getting an image."""

    client: str = Field(default="default", description="Docker client name")
    image_id: str = Field(..., description="Image ID or name")


class PullImageInput(BaseModel):
    """Input for pulling an image."""

    client: str = Field(default="default", description="Docker client name")
    image: str = Field(..., description="Image name")
    tag: str = Field(default="latest", description="Image tag")


class RemoveImageInput(BaseModel):
    """Input for removing an image."""

    client: str = Field(default="default", description="Docker client name")
    image_id: str = Field(..., description="Image ID or name")
    force: bool = Field(default=False, description="Force removal")


class ListVolumesInput(BaseModel):
    """Input for listing volumes."""

    client: str = Field(default="default", description="Docker client name")


class DockerVolumesOutput(BaseModel):
    """Output containing volume list."""

    success: bool = Field(description="Whether operation succeeded")
    volumes: Optional[List[Dict[str, Any]]] = Field(default=None, description="Volume list")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class CreateVolumeInput(BaseModel):
    """Input for creating a volume."""

    client: str = Field(default="default", description="Docker client name")
    name: str = Field(..., description="Volume name")
    driver: str = Field(default="local", description="Volume driver")


class RemoveVolumeInput(BaseModel):
    """Input for removing a volume."""

    client: str = Field(default="default", description="Docker client name")
    name: str = Field(..., description="Volume name")
    force: bool = Field(default=False, description="Force removal")


class ListNetworksInput(BaseModel):
    """Input for listing networks."""

    client: str = Field(default="default", description="Docker client name")


class DockerNetworksOutput(BaseModel):
    """Output containing network list."""

    success: bool = Field(description="Whether operation succeeded")
    networks: Optional[List[Dict[str, Any]]] = Field(default=None, description="Network list")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class GetNetworkInput(BaseModel):
    """Input for getting a network."""

    client: str = Field(default="default", description="Docker client name")
    network_id: str = Field(..., description="Network ID or name")


# Tools

class CreateDockerClientTool(BaseTool[CreateDockerClientInput, CreateDockerClientOutput]):
    """Tool for creating a Docker client."""

    metadata = ToolMetadata(
        id="create_docker_client",
        name="Create Docker Client",
        description="Create a Docker API client",
        category="utility",
    )
    input_type = CreateDockerClientInput
    output_type = CreateDockerClientOutput

    def __init__(self, manager: DockerManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: CreateDockerClientInput) -> CreateDockerClientOutput:
        """Create a Docker client."""
        config = DockerConfig(
            host=input.host,
            timeout=input.timeout,
            api_version=input.api_version,
        )
        client = DockerClient(config)
        self.manager.add_client(input.name, client)

        return CreateDockerClientOutput(
            success=True,
            name=input.name,
        )


class ListContainersTool(BaseTool[ListContainersInput, DockerContainersOutput]):
    """Tool for listing containers."""

    metadata = ToolMetadata(
        id="list_docker_containers",
        name="List Docker Containers",
        description="List Docker containers",
        category="utility",
    )
    input_type = ListContainersInput
    output_type = DockerContainersOutput

    def __init__(self, manager: DockerManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: ListContainersInput) -> DockerContainersOutput:
        """List containers."""
        client = self.manager.get_client(input.client)

        if not client:
            return DockerContainersOutput(success=False, error=f"Client '{input.client}' not found")

        filters = None
        if input.filter_status:
            filters = {"status": [input.filter_status]}

        result = client.list_containers(
            all_containers=input.all_containers,
            filters=filters,
        )

        if result.success:
            return DockerContainersOutput(success=True, containers=result.data.get("containers"))
        return DockerContainersOutput(success=False, error=result.error)


class GetContainerTool(BaseTool[GetContainerInput, DockerContainersOutput]):
    """Tool for getting a container."""

    metadata = ToolMetadata(
        id="get_docker_container",
        name="Get Docker Container",
        description="Get details of a Docker container",
        category="utility",
    )
    input_type = GetContainerInput
    output_type = DockerContainersOutput

    def __init__(self, manager: DockerManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: GetContainerInput) -> DockerContainersOutput:
        """Get a container."""
        client = self.manager.get_client(input.client)

        if not client:
            return DockerContainersOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.get_container(input.container_id)

        if result.success:
            container = result.data.get("container")
            return DockerContainersOutput(success=True, containers=[container] if container else None)
        return DockerContainersOutput(success=False, error=result.error)


class StartContainerTool(BaseTool[StartContainerInput, DockerSimpleOutput]):
    """Tool for starting a container."""

    metadata = ToolMetadata(
        id="start_docker_container",
        name="Start Docker Container",
        description="Start a Docker container",
        category="execution",
    )
    input_type = StartContainerInput
    output_type = DockerSimpleOutput

    def __init__(self, manager: DockerManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: StartContainerInput) -> DockerSimpleOutput:
        """Start a container."""
        client = self.manager.get_client(input.client)

        if not client:
            return DockerSimpleOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.start_container(input.container_id)
        return DockerSimpleOutput(success=result.success, error=result.error)


class StopContainerTool(BaseTool[StopContainerInput, DockerSimpleOutput]):
    """Tool for stopping a container."""

    metadata = ToolMetadata(
        id="stop_docker_container",
        name="Stop Docker Container",
        description="Stop a Docker container",
        category="execution",
    )
    input_type = StopContainerInput
    output_type = DockerSimpleOutput

    def __init__(self, manager: DockerManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: StopContainerInput) -> DockerSimpleOutput:
        """Stop a container."""
        client = self.manager.get_client(input.client)

        if not client:
            return DockerSimpleOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.stop_container(input.container_id, timeout=input.timeout)
        return DockerSimpleOutput(success=result.success, error=result.error)


class RemoveContainerTool(BaseTool[RemoveContainerInput, DockerSimpleOutput]):
    """Tool for removing a container."""

    metadata = ToolMetadata(
        id="remove_docker_container",
        name="Remove Docker Container",
        description="Remove a Docker container",
        category="execution",
    )
    input_type = RemoveContainerInput
    output_type = DockerSimpleOutput

    def __init__(self, manager: DockerManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: RemoveContainerInput) -> DockerSimpleOutput:
        """Remove a container."""
        client = self.manager.get_client(input.client)

        if not client:
            return DockerSimpleOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.remove_container(input.container_id, force=input.force)
        return DockerSimpleOutput(success=result.success, error=result.error)


class GetContainerLogsTool(BaseTool[GetContainerLogsInput, DockerLogsOutput]):
    """Tool for getting container logs."""

    metadata = ToolMetadata(
        id="get_docker_container_logs",
        name="Get Docker Container Logs",
        description="Get logs from a Docker container",
        category="utility",
    )
    input_type = GetContainerLogsInput
    output_type = DockerLogsOutput

    def __init__(self, manager: DockerManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: GetContainerLogsInput) -> DockerLogsOutput:
        """Get container logs."""
        client = self.manager.get_client(input.client)

        if not client:
            return DockerLogsOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.get_container_logs(
            container_id=input.container_id,
            tail=input.tail,
            timestamps=input.timestamps,
        )

        if result.success:
            logs = result.data.get("logs") if isinstance(result.data, dict) else ""
            return DockerLogsOutput(success=True, logs=logs)
        return DockerLogsOutput(success=False, error=result.error)


class ListImagesTool(BaseTool[ListImagesInput, DockerImagesOutput]):
    """Tool for listing images."""

    metadata = ToolMetadata(
        id="list_docker_images",
        name="List Docker Images",
        description="List Docker images",
        category="utility",
    )
    input_type = ListImagesInput
    output_type = DockerImagesOutput

    def __init__(self, manager: DockerManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: ListImagesInput) -> DockerImagesOutput:
        """List images."""
        client = self.manager.get_client(input.client)

        if not client:
            return DockerImagesOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.list_images(all_images=input.all_images)

        if result.success:
            return DockerImagesOutput(success=True, images=result.data.get("images"))
        return DockerImagesOutput(success=False, error=result.error)


class GetImageTool(BaseTool[GetImageInput, DockerImagesOutput]):
    """Tool for getting an image."""

    metadata = ToolMetadata(
        id="get_docker_image",
        name="Get Docker Image",
        description="Get details of a Docker image",
        category="utility",
    )
    input_type = GetImageInput
    output_type = DockerImagesOutput

    def __init__(self, manager: DockerManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: GetImageInput) -> DockerImagesOutput:
        """Get an image."""
        client = self.manager.get_client(input.client)

        if not client:
            return DockerImagesOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.get_image(input.image_id)

        if result.success:
            image = result.data.get("image")
            return DockerImagesOutput(success=True, images=[image] if image else None)
        return DockerImagesOutput(success=False, error=result.error)


class PullImageTool(BaseTool[PullImageInput, DockerSimpleOutput]):
    """Tool for pulling an image."""

    metadata = ToolMetadata(
        id="pull_docker_image",
        name="Pull Docker Image",
        description="Pull a Docker image from registry",
        category="execution",
    )
    input_type = PullImageInput
    output_type = DockerSimpleOutput

    def __init__(self, manager: DockerManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: PullImageInput) -> DockerSimpleOutput:
        """Pull an image."""
        client = self.manager.get_client(input.client)

        if not client:
            return DockerSimpleOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.pull_image(image=input.image, tag=input.tag)
        return DockerSimpleOutput(success=result.success, error=result.error)


class RemoveImageTool(BaseTool[RemoveImageInput, DockerSimpleOutput]):
    """Tool for removing an image."""

    metadata = ToolMetadata(
        id="remove_docker_image",
        name="Remove Docker Image",
        description="Remove a Docker image",
        category="execution",
    )
    input_type = RemoveImageInput
    output_type = DockerSimpleOutput

    def __init__(self, manager: DockerManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: RemoveImageInput) -> DockerSimpleOutput:
        """Remove an image."""
        client = self.manager.get_client(input.client)

        if not client:
            return DockerSimpleOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.remove_image(input.image_id, force=input.force)
        return DockerSimpleOutput(success=result.success, error=result.error)


class ListVolumesTool(BaseTool[ListVolumesInput, DockerVolumesOutput]):
    """Tool for listing volumes."""

    metadata = ToolMetadata(
        id="list_docker_volumes",
        name="List Docker Volumes",
        description="List Docker volumes",
        category="utility",
    )
    input_type = ListVolumesInput
    output_type = DockerVolumesOutput

    def __init__(self, manager: DockerManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: ListVolumesInput) -> DockerVolumesOutput:
        """List volumes."""
        client = self.manager.get_client(input.client)

        if not client:
            return DockerVolumesOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.list_volumes()

        if result.success:
            return DockerVolumesOutput(success=True, volumes=result.data.get("volumes"))
        return DockerVolumesOutput(success=False, error=result.error)


class CreateVolumeTool(BaseTool[CreateVolumeInput, DockerVolumesOutput]):
    """Tool for creating a volume."""

    metadata = ToolMetadata(
        id="create_docker_volume",
        name="Create Docker Volume",
        description="Create a Docker volume",
        category="execution",
    )
    input_type = CreateVolumeInput
    output_type = DockerVolumesOutput

    def __init__(self, manager: DockerManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: CreateVolumeInput) -> DockerVolumesOutput:
        """Create a volume."""
        client = self.manager.get_client(input.client)

        if not client:
            return DockerVolumesOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.create_volume(name=input.name, driver=input.driver)

        if result.success:
            volume = result.data.get("volume")
            return DockerVolumesOutput(success=True, volumes=[volume] if volume else None)
        return DockerVolumesOutput(success=False, error=result.error)


class RemoveVolumeTool(BaseTool[RemoveVolumeInput, DockerSimpleOutput]):
    """Tool for removing a volume."""

    metadata = ToolMetadata(
        id="remove_docker_volume",
        name="Remove Docker Volume",
        description="Remove a Docker volume",
        category="execution",
    )
    input_type = RemoveVolumeInput
    output_type = DockerSimpleOutput

    def __init__(self, manager: DockerManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: RemoveVolumeInput) -> DockerSimpleOutput:
        """Remove a volume."""
        client = self.manager.get_client(input.client)

        if not client:
            return DockerSimpleOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.remove_volume(name=input.name, force=input.force)
        return DockerSimpleOutput(success=result.success, error=result.error)


class ListNetworksTool(BaseTool[ListNetworksInput, DockerNetworksOutput]):
    """Tool for listing networks."""

    metadata = ToolMetadata(
        id="list_docker_networks",
        name="List Docker Networks",
        description="List Docker networks",
        category="utility",
    )
    input_type = ListNetworksInput
    output_type = DockerNetworksOutput

    def __init__(self, manager: DockerManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: ListNetworksInput) -> DockerNetworksOutput:
        """List networks."""
        client = self.manager.get_client(input.client)

        if not client:
            return DockerNetworksOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.list_networks()

        if result.success:
            return DockerNetworksOutput(success=True, networks=result.data.get("networks"))
        return DockerNetworksOutput(success=False, error=result.error)


class GetNetworkTool(BaseTool[GetNetworkInput, DockerNetworksOutput]):
    """Tool for getting a network."""

    metadata = ToolMetadata(
        id="get_docker_network",
        name="Get Docker Network",
        description="Get details of a Docker network",
        category="utility",
    )
    input_type = GetNetworkInput
    output_type = DockerNetworksOutput

    def __init__(self, manager: DockerManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: GetNetworkInput) -> DockerNetworksOutput:
        """Get a network."""
        client = self.manager.get_client(input.client)

        if not client:
            return DockerNetworksOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.get_network(input.network_id)

        if result.success:
            network = result.data.get("network")
            return DockerNetworksOutput(success=True, networks=[network] if network else None)
        return DockerNetworksOutput(success=False, error=result.error)


# Helper functions

def create_docker_config(
    host: str = "unix:///var/run/docker.sock",
    timeout: int = 30,
    api_version: str = "v1.43",
) -> DockerConfig:
    """Create a Docker configuration."""
    return DockerConfig(
        host=host,
        timeout=timeout,
        api_version=api_version,
    )


def create_docker_client(config: DockerConfig) -> DockerClient:
    """Create a Docker client."""
    return DockerClient(config)


def create_docker_manager() -> DockerManager:
    """Create a Docker manager."""
    return DockerManager()


def create_docker_tools(manager: DockerManager) -> Dict[str, BaseTool]:
    """Create all Docker tools."""
    return {
        "create_docker_client": CreateDockerClientTool(manager),
        # Containers
        "list_docker_containers": ListContainersTool(manager),
        "get_docker_container": GetContainerTool(manager),
        "start_docker_container": StartContainerTool(manager),
        "stop_docker_container": StopContainerTool(manager),
        "remove_docker_container": RemoveContainerTool(manager),
        "get_docker_container_logs": GetContainerLogsTool(manager),
        # Images
        "list_docker_images": ListImagesTool(manager),
        "get_docker_image": GetImageTool(manager),
        "pull_docker_image": PullImageTool(manager),
        "remove_docker_image": RemoveImageTool(manager),
        # Volumes
        "list_docker_volumes": ListVolumesTool(manager),
        "create_docker_volume": CreateVolumeTool(manager),
        "remove_docker_volume": RemoveVolumeTool(manager),
        # Networks
        "list_docker_networks": ListNetworksTool(manager),
        "get_docker_network": GetNetworkTool(manager),
    }
