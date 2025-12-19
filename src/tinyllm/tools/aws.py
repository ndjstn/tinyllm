"""AWS tools for TinyLLM.

This module provides tools for interacting with AWS services
including S3, DynamoDB, and Lambda.
"""

import json
import logging
import hashlib
import hmac
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlencode

from pydantic import BaseModel, Field

from tinyllm.tools.base import BaseTool, ToolMetadata

logger = logging.getLogger(__name__)


class AWSService(str, Enum):
    """AWS services."""

    S3 = "s3"
    DYNAMODB = "dynamodb"
    LAMBDA = "lambda"
    SQS = "sqs"
    SNS = "sns"


@dataclass
class AWSConfig:
    """AWS configuration."""

    access_key_id: str
    secret_access_key: str
    region: str = "us-east-1"
    session_token: Optional[str] = None
    timeout: int = 30


@dataclass
class S3Object:
    """S3 object representation."""

    key: str
    size: Optional[int] = None
    last_modified: Optional[str] = None
    etag: Optional[str] = None
    storage_class: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "size": self.size,
            "last_modified": self.last_modified,
            "etag": self.etag,
            "storage_class": self.storage_class,
        }


@dataclass
class S3Bucket:
    """S3 bucket representation."""

    name: str
    creation_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "creation_date": self.creation_date,
        }


@dataclass
class DynamoDBItem:
    """DynamoDB item representation."""

    attributes: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dynamodb_format(cls, data: Dict[str, Any]) -> "DynamoDBItem":
        """Convert from DynamoDB format to simple dict."""
        def convert_value(value: Dict[str, Any]) -> Any:
            if "S" in value:
                return value["S"]
            elif "N" in value:
                return float(value["N"]) if "." in value["N"] else int(value["N"])
            elif "BOOL" in value:
                return value["BOOL"]
            elif "NULL" in value:
                return None
            elif "L" in value:
                return [convert_value(v) for v in value["L"]]
            elif "M" in value:
                return {k: convert_value(v) for k, v in value["M"].items()}
            elif "SS" in value:
                return set(value["SS"])
            elif "NS" in value:
                return {float(n) if "." in n else int(n) for n in value["NS"]}
            elif "B" in value:
                return value["B"]
            return value

        attributes = {k: convert_value(v) for k, v in data.items()}
        return cls(attributes=attributes)

    @staticmethod
    def to_dynamodb_format(value: Any) -> Dict[str, Any]:
        """Convert a Python value to DynamoDB format."""
        if isinstance(value, str):
            return {"S": value}
        elif isinstance(value, bool):
            return {"BOOL": value}
        elif isinstance(value, (int, float)):
            return {"N": str(value)}
        elif value is None:
            return {"NULL": True}
        elif isinstance(value, list):
            return {"L": [DynamoDBItem.to_dynamodb_format(v) for v in value]}
        elif isinstance(value, dict):
            return {"M": {k: DynamoDBItem.to_dynamodb_format(v) for k, v in value.items()}}
        elif isinstance(value, set):
            if all(isinstance(v, str) for v in value):
                return {"SS": list(value)}
            elif all(isinstance(v, (int, float)) for v in value):
                return {"NS": [str(v) for v in value]}
        return {"S": str(value)}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.attributes


@dataclass
class LambdaFunction:
    """Lambda function representation."""

    function_name: str
    function_arn: Optional[str] = None
    runtime: Optional[str] = None
    handler: Optional[str] = None
    description: Optional[str] = None
    timeout: Optional[int] = None
    memory_size: Optional[int] = None
    last_modified: Optional[str] = None

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "LambdaFunction":
        """Create from Lambda API response."""
        return cls(
            function_name=data.get("FunctionName", ""),
            function_arn=data.get("FunctionArn"),
            runtime=data.get("Runtime"),
            handler=data.get("Handler"),
            description=data.get("Description"),
            timeout=data.get("Timeout"),
            memory_size=data.get("MemorySize"),
            last_modified=data.get("LastModified"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "function_name": self.function_name,
            "function_arn": self.function_arn,
            "runtime": self.runtime,
            "handler": self.handler,
            "description": self.description,
            "timeout": self.timeout,
            "memory_size": self.memory_size,
            "last_modified": self.last_modified,
        }


@dataclass
class AWSResult:
    """Result from AWS API operation."""

    success: bool
    data: Optional[Any] = None
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


class AWSSigner:
    """AWS Signature Version 4 signer."""

    def __init__(self, config: AWSConfig):
        """Initialize signer."""
        self.config = config

    def sign(
        self,
        method: str,
        service: str,
        host: str,
        uri: str,
        headers: Dict[str, str],
        payload: str = "",
        query_params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Sign a request using AWS Signature Version 4.

        Args:
            method: HTTP method.
            service: AWS service name.
            host: Request host.
            uri: Request URI.
            headers: Request headers.
            payload: Request payload.
            query_params: Query parameters.

        Returns:
            Signed headers.
        """
        # Create timestamp
        t = datetime.now(timezone.utc)
        amz_date = t.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = t.strftime("%Y%m%d")

        # Add required headers
        headers["Host"] = host
        headers["X-Amz-Date"] = amz_date

        if self.config.session_token:
            headers["X-Amz-Security-Token"] = self.config.session_token

        # Create canonical request
        canonical_uri = uri
        canonical_querystring = ""
        if query_params:
            canonical_querystring = "&".join(
                f"{quote(k, safe='')}={quote(str(v), safe='')}"
                for k, v in sorted(query_params.items())
            )

        signed_headers = ";".join(k.lower() for k in sorted(headers.keys()))
        canonical_headers = "".join(
            f"{k.lower()}:{headers[k].strip()}\n"
            for k in sorted(headers.keys())
        )

        payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        canonical_request = "\n".join([
            method,
            canonical_uri,
            canonical_querystring,
            canonical_headers,
            signed_headers,
            payload_hash,
        ])

        # Create string to sign
        algorithm = "AWS4-HMAC-SHA256"
        credential_scope = f"{date_stamp}/{self.config.region}/{service}/aws4_request"
        string_to_sign = "\n".join([
            algorithm,
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ])

        # Calculate signature
        def sign_key(key: bytes, msg: str) -> bytes:
            return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

        k_date = sign_key(f"AWS4{self.config.secret_access_key}".encode("utf-8"), date_stamp)
        k_region = sign_key(k_date, self.config.region)
        k_service = sign_key(k_region, service)
        k_signing = sign_key(k_service, "aws4_request")

        signature = hmac.new(
            k_signing,
            string_to_sign.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        # Create authorization header
        authorization = (
            f"{algorithm} "
            f"Credential={self.config.access_key_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, "
            f"Signature={signature}"
        )

        headers["Authorization"] = authorization
        return headers


class AWSClient:
    """Client for AWS APIs."""

    def __init__(self, config: AWSConfig):
        """Initialize client.

        Args:
            config: AWS configuration.
        """
        self.config = config
        self.signer = AWSSigner(config)

    def _make_request(
        self,
        method: str,
        service: AWSService,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        payload: str = "",
        query_params: Optional[Dict[str, str]] = None,
    ) -> AWSResult:
        """Make HTTP request to AWS API.

        Args:
            method: HTTP method.
            service: AWS service.
            endpoint: Full endpoint URL.
            headers: Request headers.
            payload: Request payload.
            query_params: Query parameters.

        Returns:
            AWS result.
        """
        # Parse endpoint
        from urllib.parse import urlparse
        parsed = urlparse(endpoint)
        host = parsed.netloc
        uri = parsed.path or "/"

        headers = headers or {}

        # Sign request
        signed_headers = self.signer.sign(
            method=method,
            service=service.value,
            host=host,
            uri=uri,
            headers=headers,
            payload=payload,
            query_params=query_params,
        )

        # Build URL with query params
        url = endpoint
        if query_params:
            qs = urlencode(query_params)
            url = f"{endpoint}?{qs}"

        try:
            req = urllib.request.Request(
                url,
                data=payload.encode("utf-8") if payload else None,
                headers=signed_headers,
                method=method,
            )

            with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                response_body = response.read().decode("utf-8")

                # Parse response based on content type
                content_type = response.headers.get("Content-Type", "")
                if "json" in content_type:
                    response_data = json.loads(response_body) if response_body else None
                elif "xml" in content_type:
                    response_data = response_body  # Return raw XML
                else:
                    response_data = response_body

                return AWSResult(
                    success=True,
                    data=response_data,
                    status_code=response.getcode(),
                )

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass

            return AWSResult(
                success=False,
                error=f"HTTP {e.code}: {error_body or e.reason}",
                status_code=e.code,
            )
        except urllib.error.URLError as e:
            return AWSResult(
                success=False,
                error=f"Connection error: {e.reason}",
            )
        except Exception as e:
            return AWSResult(
                success=False,
                error=str(e),
            )

    # S3 operations

    def list_buckets(self) -> AWSResult:
        """List S3 buckets.

        Returns:
            AWS result with bucket list.
        """
        endpoint = f"https://s3.{self.config.region}.amazonaws.com/"

        result = self._make_request("GET", AWSService.S3, endpoint)

        if result.success and result.data:
            # Parse XML response (simplified)
            buckets = []
            data = result.data
            if isinstance(data, str) and "<Bucket>" in data:
                import re
                bucket_matches = re.findall(r"<Name>([^<]+)</Name>", data)
                date_matches = re.findall(r"<CreationDate>([^<]+)</CreationDate>", data)
                for i, name in enumerate(bucket_matches):
                    buckets.append(S3Bucket(
                        name=name,
                        creation_date=date_matches[i] if i < len(date_matches) else None,
                    ).to_dict())
            result.data = {"buckets": buckets}

        return result

    def list_objects(
        self,
        bucket: str,
        prefix: Optional[str] = None,
        max_keys: int = 1000,
    ) -> AWSResult:
        """List objects in an S3 bucket.

        Args:
            bucket: Bucket name.
            prefix: Object prefix filter.
            max_keys: Maximum objects to return.

        Returns:
            AWS result with object list.
        """
        endpoint = f"https://{bucket}.s3.{self.config.region}.amazonaws.com/"

        params = {"max-keys": str(max_keys)}
        if prefix:
            params["prefix"] = prefix

        result = self._make_request("GET", AWSService.S3, endpoint, query_params=params)

        if result.success and result.data:
            # Parse XML response (simplified)
            objects = []
            data = result.data
            if isinstance(data, str) and "<Contents>" in data:
                import re
                # Extract object info
                contents = re.findall(r"<Contents>(.*?)</Contents>", data, re.DOTALL)
                for content in contents:
                    key = re.search(r"<Key>([^<]+)</Key>", content)
                    size = re.search(r"<Size>([^<]+)</Size>", content)
                    modified = re.search(r"<LastModified>([^<]+)</LastModified>", content)
                    etag = re.search(r"<ETag>([^<]+)</ETag>", content)

                    if key:
                        objects.append(S3Object(
                            key=key.group(1),
                            size=int(size.group(1)) if size else None,
                            last_modified=modified.group(1) if modified else None,
                            etag=etag.group(1).strip('"') if etag else None,
                        ).to_dict())

            result.data = {"objects": objects}

        return result

    def get_object(self, bucket: str, key: str) -> AWSResult:
        """Get an S3 object.

        Args:
            bucket: Bucket name.
            key: Object key.

        Returns:
            AWS result with object content.
        """
        endpoint = f"https://{bucket}.s3.{self.config.region}.amazonaws.com/{key}"

        return self._make_request("GET", AWSService.S3, endpoint)

    def put_object(
        self,
        bucket: str,
        key: str,
        body: str,
        content_type: str = "text/plain",
    ) -> AWSResult:
        """Put an object to S3.

        Args:
            bucket: Bucket name.
            key: Object key.
            body: Object content.
            content_type: Content type.

        Returns:
            AWS result.
        """
        endpoint = f"https://{bucket}.s3.{self.config.region}.amazonaws.com/{key}"

        headers = {"Content-Type": content_type}

        return self._make_request("PUT", AWSService.S3, endpoint, headers=headers, payload=body)

    def delete_object(self, bucket: str, key: str) -> AWSResult:
        """Delete an S3 object.

        Args:
            bucket: Bucket name.
            key: Object key.

        Returns:
            AWS result.
        """
        endpoint = f"https://{bucket}.s3.{self.config.region}.amazonaws.com/{key}"

        return self._make_request("DELETE", AWSService.S3, endpoint)

    # DynamoDB operations

    def get_item(
        self,
        table_name: str,
        key: Dict[str, Any],
    ) -> AWSResult:
        """Get an item from DynamoDB.

        Args:
            table_name: Table name.
            key: Primary key.

        Returns:
            AWS result with item.
        """
        endpoint = f"https://dynamodb.{self.config.region}.amazonaws.com/"

        # Convert key to DynamoDB format
        dynamodb_key = {k: DynamoDBItem.to_dynamodb_format(v) for k, v in key.items()}

        payload = json.dumps({
            "TableName": table_name,
            "Key": dynamodb_key,
        })

        headers = {
            "Content-Type": "application/x-amz-json-1.0",
            "X-Amz-Target": "DynamoDB_20120810.GetItem",
        }

        result = self._make_request("POST", AWSService.DYNAMODB, endpoint, headers=headers, payload=payload)

        if result.success and result.data and "Item" in result.data:
            item = DynamoDBItem.from_dynamodb_format(result.data["Item"])
            result.data = {"item": item.to_dict()}

        return result

    def put_item(
        self,
        table_name: str,
        item: Dict[str, Any],
    ) -> AWSResult:
        """Put an item to DynamoDB.

        Args:
            table_name: Table name.
            item: Item to put.

        Returns:
            AWS result.
        """
        endpoint = f"https://dynamodb.{self.config.region}.amazonaws.com/"

        # Convert item to DynamoDB format
        dynamodb_item = {k: DynamoDBItem.to_dynamodb_format(v) for k, v in item.items()}

        payload = json.dumps({
            "TableName": table_name,
            "Item": dynamodb_item,
        })

        headers = {
            "Content-Type": "application/x-amz-json-1.0",
            "X-Amz-Target": "DynamoDB_20120810.PutItem",
        }

        return self._make_request("POST", AWSService.DYNAMODB, endpoint, headers=headers, payload=payload)

    def delete_item(
        self,
        table_name: str,
        key: Dict[str, Any],
    ) -> AWSResult:
        """Delete an item from DynamoDB.

        Args:
            table_name: Table name.
            key: Primary key.

        Returns:
            AWS result.
        """
        endpoint = f"https://dynamodb.{self.config.region}.amazonaws.com/"

        # Convert key to DynamoDB format
        dynamodb_key = {k: DynamoDBItem.to_dynamodb_format(v) for k, v in key.items()}

        payload = json.dumps({
            "TableName": table_name,
            "Key": dynamodb_key,
        })

        headers = {
            "Content-Type": "application/x-amz-json-1.0",
            "X-Amz-Target": "DynamoDB_20120810.DeleteItem",
        }

        return self._make_request("POST", AWSService.DYNAMODB, endpoint, headers=headers, payload=payload)

    def query_table(
        self,
        table_name: str,
        key_condition: str,
        expression_values: Dict[str, Any],
        limit: int = 100,
    ) -> AWSResult:
        """Query a DynamoDB table.

        Args:
            table_name: Table name.
            key_condition: Key condition expression.
            expression_values: Expression attribute values.
            limit: Maximum items.

        Returns:
            AWS result with items.
        """
        endpoint = f"https://dynamodb.{self.config.region}.amazonaws.com/"

        # Convert expression values to DynamoDB format
        dynamodb_values = {k: DynamoDBItem.to_dynamodb_format(v) for k, v in expression_values.items()}

        payload = json.dumps({
            "TableName": table_name,
            "KeyConditionExpression": key_condition,
            "ExpressionAttributeValues": dynamodb_values,
            "Limit": limit,
        })

        headers = {
            "Content-Type": "application/x-amz-json-1.0",
            "X-Amz-Target": "DynamoDB_20120810.Query",
        }

        result = self._make_request("POST", AWSService.DYNAMODB, endpoint, headers=headers, payload=payload)

        if result.success and result.data and "Items" in result.data:
            items = [DynamoDBItem.from_dynamodb_format(item).to_dict() for item in result.data["Items"]]
            result.data = {"items": items, "count": result.data.get("Count", len(items))}

        return result

    # Lambda operations

    def list_functions(self) -> AWSResult:
        """List Lambda functions.

        Returns:
            AWS result with function list.
        """
        endpoint = f"https://lambda.{self.config.region}.amazonaws.com/2015-03-31/functions"

        result = self._make_request("GET", AWSService.LAMBDA, endpoint)

        if result.success and result.data and "Functions" in result.data:
            functions = [
                LambdaFunction.from_api_response(f).to_dict()
                for f in result.data["Functions"]
            ]
            result.data = {"functions": functions}

        return result

    def invoke_function(
        self,
        function_name: str,
        payload: Optional[Dict[str, Any]] = None,
        invocation_type: str = "RequestResponse",
    ) -> AWSResult:
        """Invoke a Lambda function.

        Args:
            function_name: Function name.
            payload: Function payload.
            invocation_type: Invocation type.

        Returns:
            AWS result with response.
        """
        endpoint = f"https://lambda.{self.config.region}.amazonaws.com/2015-03-31/functions/{function_name}/invocations"

        headers = {
            "Content-Type": "application/json",
            "X-Amz-Invocation-Type": invocation_type,
        }

        payload_str = json.dumps(payload) if payload else "{}"

        return self._make_request("POST", AWSService.LAMBDA, endpoint, headers=headers, payload=payload_str)


class AWSManager:
    """Manager for AWS clients."""

    def __init__(self):
        """Initialize manager."""
        self._clients: Dict[str, AWSClient] = {}

    def add_client(self, name: str, client: AWSClient) -> None:
        """Add an AWS client."""
        self._clients[name] = client

    def get_client(self, name: str) -> Optional[AWSClient]:
        """Get an AWS client."""
        return self._clients.get(name)

    def remove_client(self, name: str) -> bool:
        """Remove an AWS client."""
        if name in self._clients:
            del self._clients[name]
            return True
        return False

    def list_clients(self) -> List[str]:
        """List all client names."""
        return list(self._clients.keys())


# Pydantic models for tool inputs/outputs


class CreateAWSClientInput(BaseModel):
    """Input for creating an AWS client."""

    name: str = Field(..., description="Name for the client")
    access_key_id: str = Field(..., description="AWS access key ID")
    secret_access_key: str = Field(..., description="AWS secret access key")
    region: str = Field(default="us-east-1", description="AWS region")
    session_token: Optional[str] = Field(default=None, description="Session token for temporary credentials")


class CreateAWSClientOutput(BaseModel):
    """Output from creating an AWS client."""

    success: bool = Field(description="Whether client was created")
    name: str = Field(description="Client name")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ListBucketsInput(BaseModel):
    """Input for listing S3 buckets."""

    client: str = Field(default="default", description="AWS client name")


class S3Output(BaseModel):
    """Output containing S3 data."""

    success: bool = Field(description="Whether operation succeeded")
    buckets: Optional[List[Dict[str, Any]]] = Field(default=None, description="Bucket list")
    objects: Optional[List[Dict[str, Any]]] = Field(default=None, description="Object list")
    content: Optional[str] = Field(default=None, description="Object content")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ListObjectsInput(BaseModel):
    """Input for listing S3 objects."""

    client: str = Field(default="default", description="AWS client name")
    bucket: str = Field(..., description="Bucket name")
    prefix: Optional[str] = Field(default=None, description="Object prefix")
    max_keys: int = Field(default=1000, description="Maximum objects")


class GetObjectInput(BaseModel):
    """Input for getting an S3 object."""

    client: str = Field(default="default", description="AWS client name")
    bucket: str = Field(..., description="Bucket name")
    key: str = Field(..., description="Object key")


class PutObjectInput(BaseModel):
    """Input for putting an S3 object."""

    client: str = Field(default="default", description="AWS client name")
    bucket: str = Field(..., description="Bucket name")
    key: str = Field(..., description="Object key")
    body: str = Field(..., description="Object content")
    content_type: str = Field(default="text/plain", description="Content type")


class DeleteObjectInput(BaseModel):
    """Input for deleting an S3 object."""

    client: str = Field(default="default", description="AWS client name")
    bucket: str = Field(..., description="Bucket name")
    key: str = Field(..., description="Object key")


class SimpleAWSOutput(BaseModel):
    """Simple success/error output."""

    success: bool = Field(description="Whether operation succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class GetDynamoDBItemInput(BaseModel):
    """Input for getting a DynamoDB item."""

    client: str = Field(default="default", description="AWS client name")
    table_name: str = Field(..., description="Table name")
    key: Dict[str, Any] = Field(..., description="Primary key")


class DynamoDBOutput(BaseModel):
    """Output containing DynamoDB data."""

    success: bool = Field(description="Whether operation succeeded")
    item: Optional[Dict[str, Any]] = Field(default=None, description="Item data")
    items: Optional[List[Dict[str, Any]]] = Field(default=None, description="Item list")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class PutDynamoDBItemInput(BaseModel):
    """Input for putting a DynamoDB item."""

    client: str = Field(default="default", description="AWS client name")
    table_name: str = Field(..., description="Table name")
    item: Dict[str, Any] = Field(..., description="Item to put")


class DeleteDynamoDBItemInput(BaseModel):
    """Input for deleting a DynamoDB item."""

    client: str = Field(default="default", description="AWS client name")
    table_name: str = Field(..., description="Table name")
    key: Dict[str, Any] = Field(..., description="Primary key")


class QueryDynamoDBInput(BaseModel):
    """Input for querying DynamoDB."""

    client: str = Field(default="default", description="AWS client name")
    table_name: str = Field(..., description="Table name")
    key_condition: str = Field(..., description="Key condition expression")
    expression_values: Dict[str, Any] = Field(..., description="Expression attribute values")
    limit: int = Field(default=100, description="Maximum items")


class ListFunctionsInput(BaseModel):
    """Input for listing Lambda functions."""

    client: str = Field(default="default", description="AWS client name")


class LambdaOutput(BaseModel):
    """Output containing Lambda data."""

    success: bool = Field(description="Whether operation succeeded")
    functions: Optional[List[Dict[str, Any]]] = Field(default=None, description="Function list")
    response: Optional[Any] = Field(default=None, description="Invocation response")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class InvokeFunctionInput(BaseModel):
    """Input for invoking a Lambda function."""

    client: str = Field(default="default", description="AWS client name")
    function_name: str = Field(..., description="Function name")
    payload: Optional[Dict[str, Any]] = Field(default=None, description="Function payload")
    invocation_type: str = Field(default="RequestResponse", description="Invocation type")


# Tool implementations


class CreateAWSClientTool(BaseTool[CreateAWSClientInput, CreateAWSClientOutput]):
    """Tool for creating an AWS client."""

    metadata = ToolMetadata(
        id="create_aws_client",
        name="Create AWS Client",
        description="Create an AWS API client with credentials",
        category="utility",
    )
    input_type = CreateAWSClientInput
    output_type = CreateAWSClientOutput

    def __init__(self, manager: AWSManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: CreateAWSClientInput) -> CreateAWSClientOutput:
        """Create an AWS client."""
        config = AWSConfig(
            access_key_id=input.access_key_id,
            secret_access_key=input.secret_access_key,
            region=input.region,
            session_token=input.session_token,
        )
        client = AWSClient(config)
        self.manager.add_client(input.name, client)

        return CreateAWSClientOutput(
            success=True,
            name=input.name,
        )


class ListBucketsTool(BaseTool[ListBucketsInput, S3Output]):
    """Tool for listing S3 buckets."""

    metadata = ToolMetadata(
        id="list_s3_buckets",
        name="List S3 Buckets",
        description="List S3 buckets",
        category="utility",
    )
    input_type = ListBucketsInput
    output_type = S3Output

    def __init__(self, manager: AWSManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: ListBucketsInput) -> S3Output:
        """List buckets."""
        client = self.manager.get_client(input.client)

        if not client:
            return S3Output(success=False, error=f"Client '{input.client}' not found")

        result = client.list_buckets()

        if result.success:
            return S3Output(success=True, buckets=result.data.get("buckets"))
        return S3Output(success=False, error=result.error)


class ListObjectsTool(BaseTool[ListObjectsInput, S3Output]):
    """Tool for listing S3 objects."""

    metadata = ToolMetadata(
        id="list_s3_objects",
        name="List S3 Objects",
        description="List objects in an S3 bucket",
        category="utility",
    )
    input_type = ListObjectsInput
    output_type = S3Output

    def __init__(self, manager: AWSManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: ListObjectsInput) -> S3Output:
        """List objects."""
        client = self.manager.get_client(input.client)

        if not client:
            return S3Output(success=False, error=f"Client '{input.client}' not found")

        result = client.list_objects(
            bucket=input.bucket,
            prefix=input.prefix,
            max_keys=input.max_keys,
        )

        if result.success:
            return S3Output(success=True, objects=result.data.get("objects"))
        return S3Output(success=False, error=result.error)


class GetObjectTool(BaseTool[GetObjectInput, S3Output]):
    """Tool for getting an S3 object."""

    metadata = ToolMetadata(
        id="get_s3_object",
        name="Get S3 Object",
        description="Get an object from S3",
        category="utility",
    )
    input_type = GetObjectInput
    output_type = S3Output

    def __init__(self, manager: AWSManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: GetObjectInput) -> S3Output:
        """Get object."""
        client = self.manager.get_client(input.client)

        if not client:
            return S3Output(success=False, error=f"Client '{input.client}' not found")

        result = client.get_object(bucket=input.bucket, key=input.key)

        if result.success:
            return S3Output(success=True, content=result.data)
        return S3Output(success=False, error=result.error)


class PutObjectTool(BaseTool[PutObjectInput, SimpleAWSOutput]):
    """Tool for putting an S3 object."""

    metadata = ToolMetadata(
        id="put_s3_object",
        name="Put S3 Object",
        description="Put an object to S3",
        category="utility",
    )
    input_type = PutObjectInput
    output_type = SimpleAWSOutput

    def __init__(self, manager: AWSManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: PutObjectInput) -> SimpleAWSOutput:
        """Put object."""
        client = self.manager.get_client(input.client)

        if not client:
            return SimpleAWSOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.put_object(
            bucket=input.bucket,
            key=input.key,
            body=input.body,
            content_type=input.content_type,
        )

        if result.success:
            return SimpleAWSOutput(success=True)
        return SimpleAWSOutput(success=False, error=result.error)


class DeleteObjectTool(BaseTool[DeleteObjectInput, SimpleAWSOutput]):
    """Tool for deleting an S3 object."""

    metadata = ToolMetadata(
        id="delete_s3_object",
        name="Delete S3 Object",
        description="Delete an object from S3",
        category="utility",
    )
    input_type = DeleteObjectInput
    output_type = SimpleAWSOutput

    def __init__(self, manager: AWSManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: DeleteObjectInput) -> SimpleAWSOutput:
        """Delete object."""
        client = self.manager.get_client(input.client)

        if not client:
            return SimpleAWSOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.delete_object(bucket=input.bucket, key=input.key)

        if result.success:
            return SimpleAWSOutput(success=True)
        return SimpleAWSOutput(success=False, error=result.error)


class GetDynamoDBItemTool(BaseTool[GetDynamoDBItemInput, DynamoDBOutput]):
    """Tool for getting a DynamoDB item."""

    metadata = ToolMetadata(
        id="get_dynamodb_item",
        name="Get DynamoDB Item",
        description="Get an item from DynamoDB",
        category="utility",
    )
    input_type = GetDynamoDBItemInput
    output_type = DynamoDBOutput

    def __init__(self, manager: AWSManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: GetDynamoDBItemInput) -> DynamoDBOutput:
        """Get item."""
        client = self.manager.get_client(input.client)

        if not client:
            return DynamoDBOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.get_item(table_name=input.table_name, key=input.key)

        if result.success:
            return DynamoDBOutput(success=True, item=result.data.get("item") if result.data else None)
        return DynamoDBOutput(success=False, error=result.error)


class PutDynamoDBItemTool(BaseTool[PutDynamoDBItemInput, SimpleAWSOutput]):
    """Tool for putting a DynamoDB item."""

    metadata = ToolMetadata(
        id="put_dynamodb_item",
        name="Put DynamoDB Item",
        description="Put an item to DynamoDB",
        category="utility",
    )
    input_type = PutDynamoDBItemInput
    output_type = SimpleAWSOutput

    def __init__(self, manager: AWSManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: PutDynamoDBItemInput) -> SimpleAWSOutput:
        """Put item."""
        client = self.manager.get_client(input.client)

        if not client:
            return SimpleAWSOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.put_item(table_name=input.table_name, item=input.item)

        if result.success:
            return SimpleAWSOutput(success=True)
        return SimpleAWSOutput(success=False, error=result.error)


class DeleteDynamoDBItemTool(BaseTool[DeleteDynamoDBItemInput, SimpleAWSOutput]):
    """Tool for deleting a DynamoDB item."""

    metadata = ToolMetadata(
        id="delete_dynamodb_item",
        name="Delete DynamoDB Item",
        description="Delete an item from DynamoDB",
        category="utility",
    )
    input_type = DeleteDynamoDBItemInput
    output_type = SimpleAWSOutput

    def __init__(self, manager: AWSManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: DeleteDynamoDBItemInput) -> SimpleAWSOutput:
        """Delete item."""
        client = self.manager.get_client(input.client)

        if not client:
            return SimpleAWSOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.delete_item(table_name=input.table_name, key=input.key)

        if result.success:
            return SimpleAWSOutput(success=True)
        return SimpleAWSOutput(success=False, error=result.error)


class QueryDynamoDBTool(BaseTool[QueryDynamoDBInput, DynamoDBOutput]):
    """Tool for querying DynamoDB."""

    metadata = ToolMetadata(
        id="query_dynamodb",
        name="Query DynamoDB",
        description="Query a DynamoDB table",
        category="utility",
    )
    input_type = QueryDynamoDBInput
    output_type = DynamoDBOutput

    def __init__(self, manager: AWSManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: QueryDynamoDBInput) -> DynamoDBOutput:
        """Query table."""
        client = self.manager.get_client(input.client)

        if not client:
            return DynamoDBOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.query_table(
            table_name=input.table_name,
            key_condition=input.key_condition,
            expression_values=input.expression_values,
            limit=input.limit,
        )

        if result.success:
            return DynamoDBOutput(success=True, items=result.data.get("items") if result.data else None)
        return DynamoDBOutput(success=False, error=result.error)


class ListFunctionsTool(BaseTool[ListFunctionsInput, LambdaOutput]):
    """Tool for listing Lambda functions."""

    metadata = ToolMetadata(
        id="list_lambda_functions",
        name="List Lambda Functions",
        description="List Lambda functions",
        category="utility",
    )
    input_type = ListFunctionsInput
    output_type = LambdaOutput

    def __init__(self, manager: AWSManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: ListFunctionsInput) -> LambdaOutput:
        """List functions."""
        client = self.manager.get_client(input.client)

        if not client:
            return LambdaOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.list_functions()

        if result.success:
            return LambdaOutput(success=True, functions=result.data.get("functions") if result.data else None)
        return LambdaOutput(success=False, error=result.error)


class InvokeFunctionTool(BaseTool[InvokeFunctionInput, LambdaOutput]):
    """Tool for invoking a Lambda function."""

    metadata = ToolMetadata(
        id="invoke_lambda_function",
        name="Invoke Lambda Function",
        description="Invoke a Lambda function",
        category="execution",
    )
    input_type = InvokeFunctionInput
    output_type = LambdaOutput

    def __init__(self, manager: AWSManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: InvokeFunctionInput) -> LambdaOutput:
        """Invoke function."""
        client = self.manager.get_client(input.client)

        if not client:
            return LambdaOutput(success=False, error=f"Client '{input.client}' not found")

        result = client.invoke_function(
            function_name=input.function_name,
            payload=input.payload,
            invocation_type=input.invocation_type,
        )

        if result.success:
            return LambdaOutput(success=True, response=result.data)
        return LambdaOutput(success=False, error=result.error)


# Convenience functions


def create_aws_config(
    access_key_id: str,
    secret_access_key: str,
    region: str = "us-east-1",
    session_token: Optional[str] = None,
    timeout: int = 30,
) -> AWSConfig:
    """Create an AWS configuration."""
    return AWSConfig(
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        region=region,
        session_token=session_token,
        timeout=timeout,
    )


def create_aws_client(config: AWSConfig) -> AWSClient:
    """Create an AWS client."""
    return AWSClient(config)


def create_aws_manager() -> AWSManager:
    """Create an AWS manager."""
    return AWSManager()


def create_aws_tools(manager: AWSManager) -> Dict[str, BaseTool]:
    """Create all AWS tools."""
    return {
        "create_aws_client": CreateAWSClientTool(manager),
        # S3
        "list_s3_buckets": ListBucketsTool(manager),
        "list_s3_objects": ListObjectsTool(manager),
        "get_s3_object": GetObjectTool(manager),
        "put_s3_object": PutObjectTool(manager),
        "delete_s3_object": DeleteObjectTool(manager),
        # DynamoDB
        "get_dynamodb_item": GetDynamoDBItemTool(manager),
        "put_dynamodb_item": PutDynamoDBItemTool(manager),
        "delete_dynamodb_item": DeleteDynamoDBItemTool(manager),
        "query_dynamodb": QueryDynamoDBTool(manager),
        # Lambda
        "list_lambda_functions": ListFunctionsTool(manager),
        "invoke_lambda_function": InvokeFunctionTool(manager),
    }
