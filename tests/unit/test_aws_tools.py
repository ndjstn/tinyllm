"""Tests for AWS tools module."""

import json
import pytest
from unittest.mock import MagicMock, patch

from tinyllm.tools.aws import (
    # Enums
    AWSService,
    # Config and Client
    AWSConfig,
    AWSClient,
    AWSSigner,
    AWSManager,
    AWSResult,
    # S3 Models
    S3Bucket,
    S3Object,
    ListBucketsInput,
    S3Output,
    ListObjectsInput,
    GetObjectInput,
    PutObjectInput,
    DeleteObjectInput,
    SimpleAWSOutput,
    # DynamoDB Models
    DynamoDBItem,
    GetDynamoDBItemInput,
    DynamoDBOutput,
    PutDynamoDBItemInput,
    DeleteDynamoDBItemInput,
    QueryDynamoDBInput,
    # Lambda Models
    LambdaFunction,
    ListFunctionsInput,
    LambdaOutput,
    InvokeFunctionInput,
    # Tools
    CreateAWSClientInput,
    CreateAWSClientOutput,
    CreateAWSClientTool,
    ListBucketsTool,
    ListObjectsTool,
    GetObjectTool,
    PutObjectTool,
    DeleteObjectTool,
    GetDynamoDBItemTool,
    PutDynamoDBItemTool,
    DeleteDynamoDBItemTool,
    QueryDynamoDBTool,
    ListFunctionsTool,
    InvokeFunctionTool,
    # Helpers
    create_aws_config,
    create_aws_client,
    create_aws_manager,
    create_aws_tools,
)


class TestAWSEnums:
    """Test AWS enums."""

    def test_aws_service_values(self):
        """Test AWSService enum values."""
        assert AWSService.S3 == "s3"
        assert AWSService.DYNAMODB == "dynamodb"
        assert AWSService.LAMBDA == "lambda"


class TestAWSConfig:
    """Test AWSConfig model."""

    def test_create_config(self):
        """Test creating AWS config."""
        config = AWSConfig(
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
        )
        assert config.access_key_id == "AKIAIOSFODNN7EXAMPLE"
        assert config.secret_access_key == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        assert config.region == "us-east-1"
        assert config.session_token is None

    def test_config_with_session_token(self):
        """Test config with session token."""
        config = AWSConfig(
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-west-2",
            session_token="FwoGZXIvYXdzEBYaDK...",
        )
        assert config.session_token == "FwoGZXIvYXdzEBYaDK..."

    def test_config_defaults(self):
        """Test config defaults."""
        config = AWSConfig(
            access_key_id="test",
            secret_access_key="test",
            region="us-east-1",
        )
        assert config.timeout == 30


class TestAWSSigner:
    """Test AWS Signature Version 4 signing."""

    def test_create_signer(self):
        """Test creating a signer."""
        config = AWSConfig(
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
        )
        signer = AWSSigner(config)
        assert signer.config == config

    def test_sign_adds_authorization_header(self):
        """Test that signing adds authorization header."""
        config = AWSConfig(
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
        )
        signer = AWSSigner(config)

        headers = signer.sign(
            method="GET",
            service="s3",
            host="s3.us-east-1.amazonaws.com",
            uri="/",
            headers={},
        )

        assert "Authorization" in headers
        assert "AWS4-HMAC-SHA256" in headers["Authorization"]
        assert "X-Amz-Date" in headers
        assert "Host" in headers

    def test_sign_with_payload(self):
        """Test signing request with payload."""
        config = AWSConfig(
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
        )
        signer = AWSSigner(config)

        payload = '{"TableName": "test"}'
        headers = signer.sign(
            method="POST",
            service="dynamodb",
            host="dynamodb.us-east-1.amazonaws.com",
            uri="/",
            headers={"Content-Type": "application/json"},
            payload=payload,
        )

        assert "Authorization" in headers
        assert "Credential=" in headers["Authorization"]
        assert "SignedHeaders=" in headers["Authorization"]
        assert "Signature=" in headers["Authorization"]

    def test_sign_with_session_token(self):
        """Test signing with session token."""
        config = AWSConfig(
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
            session_token="FwoGZXIvYXdzEBYaDK...",
        )
        signer = AWSSigner(config)

        headers = signer.sign(
            method="GET",
            service="s3",
            host="s3.us-east-1.amazonaws.com",
            uri="/",
            headers={},
        )

        assert "X-Amz-Security-Token" in headers
        assert headers["X-Amz-Security-Token"] == "FwoGZXIvYXdzEBYaDK..."

    def test_sign_with_query_params(self):
        """Test signing with query parameters."""
        config = AWSConfig(
            access_key_id="test",
            secret_access_key="test",
            region="us-east-1",
        )
        signer = AWSSigner(config)

        headers = signer.sign(
            method="GET",
            service="s3",
            host="bucket.s3.us-east-1.amazonaws.com",
            uri="/",
            headers={},
            query_params={"prefix": "test/", "max-keys": "100"},
        )

        assert "Authorization" in headers


class TestAWSClient:
    """Test AWS client."""

    def test_create_client(self):
        """Test creating AWS client."""
        config = AWSConfig(
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
        )
        client = AWSClient(config)
        assert client.config == config
        assert isinstance(client.signer, AWSSigner)


class TestAWSResult:
    """Test AWSResult model."""

    def test_aws_result_success(self):
        """Test successful AWS result."""
        result = AWSResult(
            success=True,
            data={"key": "value"},
            status_code=200,
        )
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.status_code == 200
        assert result.error is None

    def test_aws_result_error(self):
        """Test error AWS result."""
        result = AWSResult(
            success=False,
            error="Something went wrong",
            status_code=500,
        )
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_aws_result_to_dict(self):
        """Test AWSResult to_dict."""
        result = AWSResult(success=True, data={"items": []})
        d = result.to_dict()
        assert "success" in d
        assert "data" in d


class TestDynamoDBItem:
    """Test DynamoDB item and type conversion."""

    def test_to_dynamodb_format_string(self):
        """Test converting string to DynamoDB format."""
        result = DynamoDBItem.to_dynamodb_format("hello")
        assert result == {"S": "hello"}

    def test_to_dynamodb_format_number(self):
        """Test converting number to DynamoDB format."""
        result = DynamoDBItem.to_dynamodb_format(42)
        assert result == {"N": "42"}

        result = DynamoDBItem.to_dynamodb_format(3.14)
        assert result == {"N": "3.14"}

    def test_to_dynamodb_format_bool(self):
        """Test converting boolean to DynamoDB format."""
        result = DynamoDBItem.to_dynamodb_format(True)
        assert result == {"BOOL": True}

        result = DynamoDBItem.to_dynamodb_format(False)
        assert result == {"BOOL": False}

    def test_to_dynamodb_format_none(self):
        """Test converting None to DynamoDB format."""
        result = DynamoDBItem.to_dynamodb_format(None)
        assert result == {"NULL": True}

    def test_to_dynamodb_format_list(self):
        """Test converting list to DynamoDB format."""
        result = DynamoDBItem.to_dynamodb_format(["a", "b", "c"])
        assert result == {"L": [{"S": "a"}, {"S": "b"}, {"S": "c"}]}

    def test_to_dynamodb_format_dict(self):
        """Test converting dict to DynamoDB format."""
        result = DynamoDBItem.to_dynamodb_format({"name": "test", "count": 5})
        assert result == {"M": {"name": {"S": "test"}, "count": {"N": "5"}}}

    def test_from_dynamodb_format_with_record(self):
        """Test creating item from DynamoDB record format."""
        # from_dynamodb_format expects a dict of attribute names to DynamoDB-formatted values
        data = {
            "id": {"S": "user123"},
            "name": {"S": "John Doe"},
            "age": {"N": "30"},
        }
        item = DynamoDBItem.from_dynamodb_format(data)
        assert item.attributes["id"] == "user123"
        assert item.attributes["name"] == "John Doe"
        assert item.attributes["age"] == 30

    def test_from_dynamodb_format_with_complex_record(self):
        """Test creating item from complex DynamoDB record."""
        data = {
            "id": {"S": "user123"},
            "tags": {"L": [{"S": "admin"}, {"S": "active"}]},
            "metadata": {"M": {"created": {"S": "2023-01-01"}}},
        }
        item = DynamoDBItem.from_dynamodb_format(data)
        assert item.attributes["id"] == "user123"
        assert item.attributes["tags"] == ["admin", "active"]
        assert item.attributes["metadata"] == {"created": "2023-01-01"}

    def test_item_to_dict(self):
        """Test DynamoDBItem to_dict."""
        item = DynamoDBItem(attributes={"id": "123", "name": "Test"})
        d = item.to_dict()
        assert d["id"] == "123"
        assert d["name"] == "Test"


class TestS3Models:
    """Test S3 models."""

    def test_s3_bucket(self):
        """Test S3Bucket model."""
        bucket = S3Bucket(
            name="my-bucket",
            creation_date="2023-01-15T10:30:00Z",
        )
        assert bucket.name == "my-bucket"
        assert bucket.creation_date == "2023-01-15T10:30:00Z"

    def test_s3_bucket_to_dict(self):
        """Test S3Bucket to_dict."""
        bucket = S3Bucket(name="test-bucket", creation_date="2023-01-01")
        d = bucket.to_dict()
        assert d["name"] == "test-bucket"

    def test_s3_object(self):
        """Test S3Object model."""
        obj = S3Object(
            key="folder/file.txt",
            size=1024,
            last_modified="2023-06-20T15:45:00Z",
            etag="abc123",
        )
        assert obj.key == "folder/file.txt"
        assert obj.size == 1024
        assert obj.storage_class is None

    def test_s3_object_to_dict(self):
        """Test S3Object to_dict."""
        obj = S3Object(key="test.txt", size=100)
        d = obj.to_dict()
        assert d["key"] == "test.txt"
        assert d["size"] == 100

    def test_list_buckets_input(self):
        """Test ListBucketsInput model."""
        inp = ListBucketsInput(client="default")
        assert inp.client == "default"

    def test_list_buckets_input_default(self):
        """Test ListBucketsInput default."""
        inp = ListBucketsInput()
        assert inp.client == "default"

    def test_list_objects_input(self):
        """Test ListObjectsInput model."""
        inp = ListObjectsInput(
            client="default",
            bucket="my-bucket",
            prefix="folder/",
        )
        assert inp.bucket == "my-bucket"
        assert inp.prefix == "folder/"

    def test_get_object_input(self):
        """Test GetObjectInput model."""
        inp = GetObjectInput(
            client="default",
            bucket="my-bucket",
            key="test.txt",
        )
        assert inp.bucket == "my-bucket"
        assert inp.key == "test.txt"

    def test_put_object_input(self):
        """Test PutObjectInput model."""
        inp = PutObjectInput(
            client="default",
            bucket="my-bucket",
            key="new-file.txt",
            body="content",
            content_type="text/plain",
        )
        assert inp.body == "content"
        assert inp.content_type == "text/plain"


class TestLambdaModels:
    """Test Lambda models."""

    def test_lambda_function(self):
        """Test LambdaFunction model."""
        func = LambdaFunction(
            function_name="my-function",
            function_arn="arn:aws:lambda:us-east-1:123456789:function:my-function",
            runtime="python3.11",
            handler="index.handler",
            memory_size=128,
            timeout=30,
            last_modified="2023-08-10T12:00:00Z",
        )
        assert func.function_name == "my-function"
        assert func.runtime == "python3.11"

    def test_lambda_function_from_api_response(self):
        """Test LambdaFunction.from_api_response."""
        response = {
            "FunctionName": "test-func",
            "FunctionArn": "arn:aws:lambda:us-east-1:123:function:test-func",
            "Runtime": "python3.11",
            "Handler": "main.handler",
            "MemorySize": 256,
            "Timeout": 60,
            "LastModified": "2023-01-01T00:00:00Z",
            "Description": "Test function",
        }
        func = LambdaFunction.from_api_response(response)
        assert func.function_name == "test-func"
        assert func.runtime == "python3.11"
        assert func.memory_size == 256

    def test_lambda_function_to_dict(self):
        """Test LambdaFunction to_dict."""
        func = LambdaFunction(
            function_name="test",
            function_arn="arn:test",
            runtime="python3.11",
        )
        d = func.to_dict()
        assert d["function_name"] == "test"

    def test_invoke_function_input(self):
        """Test InvokeFunctionInput model."""
        inp = InvokeFunctionInput(
            client="default",
            function_name="my-function",
            payload={"key": "value"},
        )
        assert inp.function_name == "my-function"
        assert inp.payload == {"key": "value"}
        assert inp.invocation_type == "RequestResponse"


class TestAWSManager:
    """Test AWS Manager."""

    def test_create_manager(self):
        """Test creating AWS manager."""
        manager = AWSManager()
        assert len(manager.list_clients()) == 0

    def test_add_client(self):
        """Test adding a client."""
        manager = AWSManager()
        config = AWSConfig(
            access_key_id="test",
            secret_access_key="test",
            region="us-east-1",
        )
        client = AWSClient(config)
        manager.add_client("prod", client)

        assert manager.get_client("prod") is not None

    def test_get_client(self):
        """Test getting a client."""
        manager = AWSManager()
        config = AWSConfig(
            access_key_id="test",
            secret_access_key="test",
            region="us-east-1",
        )
        client = AWSClient(config)
        manager.add_client("prod", client)

        retrieved = manager.get_client("prod")
        assert retrieved is not None
        assert retrieved.config == config

    def test_get_nonexistent_client(self):
        """Test getting nonexistent client."""
        manager = AWSManager()
        client = manager.get_client("nonexistent")
        assert client is None

    def test_remove_client(self):
        """Test removing a client."""
        manager = AWSManager()
        config = AWSConfig(
            access_key_id="test",
            secret_access_key="test",
            region="us-east-1",
        )
        client = AWSClient(config)
        manager.add_client("temp", client)

        removed = manager.remove_client("temp")
        assert removed is True
        assert manager.get_client("temp") is None

    def test_remove_nonexistent_client(self):
        """Test removing nonexistent client."""
        manager = AWSManager()
        removed = manager.remove_client("nonexistent")
        assert removed is False

    def test_list_clients(self):
        """Test listing clients."""
        manager = AWSManager()
        config1 = AWSConfig(access_key_id="test1", secret_access_key="test1", region="us-east-1")
        config2 = AWSConfig(access_key_id="test2", secret_access_key="test2", region="us-west-2")

        manager.add_client("prod", AWSClient(config1))
        manager.add_client("dev", AWSClient(config2))

        clients = manager.list_clients()
        assert "prod" in clients
        assert "dev" in clients


class TestCreateAWSClientTool:
    """Test CreateAWSClientTool."""

    @pytest.mark.asyncio
    async def test_create_client_tool(self):
        """Test creating a client via tool."""
        manager = AWSManager()
        tool = CreateAWSClientTool(manager)

        result = await tool.execute(CreateAWSClientInput(
            name="test",
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
        ))

        assert result.success is True
        assert result.name == "test"
        assert manager.get_client("test") is not None


class TestS3Tools:
    """Test S3 tools."""

    @pytest.fixture
    def manager(self):
        """Create a manager with a test client."""
        manager = AWSManager()
        config = AWSConfig(
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
        )
        manager.add_client("default", AWSClient(config))
        return manager

    @pytest.mark.asyncio
    async def test_list_buckets(self, manager):
        """Test listing S3 buckets."""
        tool = ListBucketsTool(manager)

        mock_response = """<?xml version="1.0" encoding="UTF-8"?>
        <ListAllMyBucketsResult>
            <Buckets>
                <Bucket>
                    <Name>bucket1</Name>
                    <CreationDate>2023-01-01T00:00:00.000Z</CreationDate>
                </Bucket>
                <Bucket>
                    <Name>bucket2</Name>
                    <CreationDate>2023-06-15T12:30:00.000Z</CreationDate>
                </Bucket>
            </Buckets>
        </ListAllMyBucketsResult>"""

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_response.encode()
            mock_resp.getcode.return_value = 200
            mock_resp.headers = {"Content-Type": "application/xml"}
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = await tool.execute(ListBucketsInput(client="default"))

        assert result.success is True
        assert len(result.buckets) == 2
        assert result.buckets[0]["name"] == "bucket1"

    @pytest.mark.asyncio
    async def test_list_objects(self, manager):
        """Test listing S3 objects."""
        tool = ListObjectsTool(manager)

        mock_response = """<?xml version="1.0" encoding="UTF-8"?>
        <ListBucketResult>
            <Name>my-bucket</Name>
            <Contents>
                <Key>file1.txt</Key>
                <Size>1024</Size>
                <LastModified>2023-08-01T10:00:00.000Z</LastModified>
                <ETag>"abc123"</ETag>
            </Contents>
            <Contents>
                <Key>folder/file2.txt</Key>
                <Size>2048</Size>
                <LastModified>2023-08-02T15:30:00.000Z</LastModified>
                <ETag>"def456"</ETag>
            </Contents>
        </ListBucketResult>"""

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_response.encode()
            mock_resp.getcode.return_value = 200
            mock_resp.headers = {"Content-Type": "application/xml"}
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = await tool.execute(ListObjectsInput(
                client="default",
                bucket="my-bucket",
            ))

        assert result.success is True
        assert len(result.objects) == 2
        assert result.objects[0]["key"] == "file1.txt"

    @pytest.mark.asyncio
    async def test_get_object(self, manager):
        """Test getting S3 object."""
        tool = GetObjectTool(manager)

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b"Hello, World!"
            mock_resp.getcode.return_value = 200
            mock_resp.headers = {"Content-Type": "text/plain"}
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = await tool.execute(GetObjectInput(
                client="default",
                bucket="my-bucket",
                key="test.txt",
            ))

        assert result.success is True
        assert result.content == "Hello, World!"

    @pytest.mark.asyncio
    async def test_put_object(self, manager):
        """Test putting S3 object."""
        tool = PutObjectTool(manager)

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b""
            mock_resp.getcode.return_value = 200
            mock_resp.headers = {"Content-Type": "application/xml"}
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = await tool.execute(PutObjectInput(
                client="default",
                bucket="my-bucket",
                key="new-file.txt",
                body="New content",
                content_type="text/plain",
            ))

        assert result.success is True

    @pytest.mark.asyncio
    async def test_delete_object(self, manager):
        """Test deleting S3 object."""
        tool = DeleteObjectTool(manager)

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b""
            mock_resp.getcode.return_value = 204
            mock_resp.headers = {}
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = await tool.execute(DeleteObjectInput(
                client="default",
                bucket="my-bucket",
                key="old-file.txt",
            ))

        assert result.success is True

    @pytest.mark.asyncio
    async def test_s3_tool_invalid_client(self, manager):
        """Test S3 tool with invalid client."""
        tool = ListBucketsTool(manager)

        result = await tool.execute(ListBucketsInput(client="nonexistent"))

        assert result.success is False
        assert "not found" in result.error.lower()


class TestDynamoDBTools:
    """Test DynamoDB tools."""

    @pytest.fixture
    def manager(self):
        """Create a manager with a test client."""
        manager = AWSManager()
        config = AWSConfig(
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
        )
        manager.add_client("default", AWSClient(config))
        return manager

    @pytest.mark.asyncio
    async def test_get_item(self, manager):
        """Test getting DynamoDB item."""
        tool = GetDynamoDBItemTool(manager)

        mock_response = json.dumps({
            "Item": {
                "id": {"S": "user123"},
                "name": {"S": "John Doe"},
                "age": {"N": "30"},
            }
        })

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_response.encode()
            mock_resp.getcode.return_value = 200
            mock_resp.headers = {"Content-Type": "application/x-amz-json-1.0"}
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = await tool.execute(GetDynamoDBItemInput(
                client="default",
                table_name="users",
                key={"id": "user123"},
            ))

        assert result.success is True
        assert result.item is not None
        assert result.item["id"] == "user123"
        assert result.item["name"] == "John Doe"
        assert result.item["age"] == 30

    @pytest.mark.asyncio
    async def test_get_item_not_found(self, manager):
        """Test getting nonexistent DynamoDB item."""
        tool = GetDynamoDBItemTool(manager)

        mock_response = json.dumps({})

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_response.encode()
            mock_resp.getcode.return_value = 200
            mock_resp.headers = {"Content-Type": "application/x-amz-json-1.0"}
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = await tool.execute(GetDynamoDBItemInput(
                client="default",
                table_name="users",
                key={"id": "nonexistent"},
            ))

        assert result.success is True
        assert result.item is None

    @pytest.mark.asyncio
    async def test_put_item(self, manager):
        """Test putting DynamoDB item."""
        tool = PutDynamoDBItemTool(manager)

        mock_response = json.dumps({})

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_response.encode()
            mock_resp.getcode.return_value = 200
            mock_resp.headers = {"Content-Type": "application/x-amz-json-1.0"}
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = await tool.execute(PutDynamoDBItemInput(
                client="default",
                table_name="users",
                item={"id": "user456", "name": "Jane Doe", "active": True},
            ))

        assert result.success is True

    @pytest.mark.asyncio
    async def test_delete_item(self, manager):
        """Test deleting DynamoDB item."""
        tool = DeleteDynamoDBItemTool(manager)

        mock_response = json.dumps({})

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_response.encode()
            mock_resp.getcode.return_value = 200
            mock_resp.headers = {"Content-Type": "application/x-amz-json-1.0"}
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = await tool.execute(DeleteDynamoDBItemInput(
                client="default",
                table_name="users",
                key={"id": "user123"},
            ))

        assert result.success is True

    @pytest.mark.asyncio
    async def test_query_table(self, manager):
        """Test querying DynamoDB table."""
        tool = QueryDynamoDBTool(manager)

        mock_response = json.dumps({
            "Items": [
                {
                    "user_id": {"S": "user123"},
                    "order_id": {"S": "order1"},
                    "total": {"N": "99.99"},
                },
                {
                    "user_id": {"S": "user123"},
                    "order_id": {"S": "order2"},
                    "total": {"N": "149.50"},
                },
            ],
            "Count": 2,
            "ScannedCount": 2,
        })

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_response.encode()
            mock_resp.getcode.return_value = 200
            mock_resp.headers = {"Content-Type": "application/x-amz-json-1.0"}
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = await tool.execute(QueryDynamoDBInput(
                client="default",
                table_name="orders",
                key_condition="user_id = :uid",
                expression_values={":uid": "user123"},
            ))

        assert result.success is True
        assert len(result.items) == 2

    @pytest.mark.asyncio
    async def test_dynamodb_tool_invalid_client(self, manager):
        """Test DynamoDB tool with invalid client."""
        tool = GetDynamoDBItemTool(manager)

        result = await tool.execute(GetDynamoDBItemInput(
            client="nonexistent",
            table_name="users",
            key={"id": "123"},
        ))

        assert result.success is False


class TestLambdaTools:
    """Test Lambda tools."""

    @pytest.fixture
    def manager(self):
        """Create a manager with a test client."""
        manager = AWSManager()
        config = AWSConfig(
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
        )
        manager.add_client("default", AWSClient(config))
        return manager

    @pytest.mark.asyncio
    async def test_list_functions(self, manager):
        """Test listing Lambda functions."""
        tool = ListFunctionsTool(manager)

        mock_response = json.dumps({
            "Functions": [
                {
                    "FunctionName": "function1",
                    "FunctionArn": "arn:aws:lambda:us-east-1:123456789:function:function1",
                    "Runtime": "python3.11",
                    "Handler": "index.handler",
                    "MemorySize": 128,
                    "Timeout": 30,
                    "LastModified": "2023-08-01T10:00:00.000+0000",
                },
                {
                    "FunctionName": "function2",
                    "FunctionArn": "arn:aws:lambda:us-east-1:123456789:function:function2",
                    "Runtime": "nodejs18.x",
                    "Handler": "index.handler",
                    "MemorySize": 256,
                    "Timeout": 60,
                    "LastModified": "2023-08-15T14:30:00.000+0000",
                },
            ]
        })

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_response.encode()
            mock_resp.getcode.return_value = 200
            mock_resp.headers = {"Content-Type": "application/json"}
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = await tool.execute(ListFunctionsInput(client="default"))

        assert result.success is True
        assert len(result.functions) == 2
        assert result.functions[0]["function_name"] == "function1"
        assert result.functions[0]["runtime"] == "python3.11"

    @pytest.mark.asyncio
    async def test_invoke_function(self, manager):
        """Test invoking Lambda function."""
        tool = InvokeFunctionTool(manager)

        mock_response = json.dumps({"result": "success", "data": [1, 2, 3]})

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_response.encode()
            mock_resp.getcode.return_value = 200
            mock_resp.headers = {"Content-Type": "application/json"}
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = await tool.execute(InvokeFunctionInput(
                client="default",
                function_name="my-function",
                payload={"input": "test"},
            ))

        assert result.success is True
        assert result.response == {"result": "success", "data": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_lambda_tool_invalid_client(self, manager):
        """Test Lambda tool with invalid client."""
        tool = ListFunctionsTool(manager)

        result = await tool.execute(ListFunctionsInput(client="nonexistent"))

        assert result.success is False


class TestHelperFunctions:
    """Test helper functions."""

    def test_create_aws_config(self):
        """Test create_aws_config helper."""
        config = create_aws_config(
            access_key_id="test",
            secret_access_key="test",
            region="us-east-1",
        )
        assert isinstance(config, AWSConfig)
        assert config.region == "us-east-1"

    def test_create_aws_client(self):
        """Test create_aws_client helper."""
        config = AWSConfig(
            access_key_id="test",
            secret_access_key="test",
            region="us-east-1",
        )
        client = create_aws_client(config)
        assert isinstance(client, AWSClient)

    def test_create_aws_manager(self):
        """Test create_aws_manager helper."""
        manager = create_aws_manager()
        assert isinstance(manager, AWSManager)

    def test_create_aws_tools(self):
        """Test create_aws_tools helper."""
        manager = create_aws_manager()
        tools = create_aws_tools(manager)

        assert len(tools) > 0
        # Check that all expected tools are present (returns a dict)
        assert "create_aws_client" in tools
        assert "list_s3_buckets" in tools
        assert "list_s3_objects" in tools
        assert "get_s3_object" in tools
        assert "put_s3_object" in tools
        assert "delete_s3_object" in tools
        assert "get_dynamodb_item" in tools
        assert "put_dynamodb_item" in tools
        assert "delete_dynamodb_item" in tools
        assert "query_dynamodb" in tools
        assert "list_lambda_functions" in tools
        assert "invoke_lambda_function" in tools


class TestToolMetadata:
    """Test tool metadata."""

    def test_s3_tool_metadata(self):
        """Test S3 tool metadata."""
        manager = AWSManager()
        tool = ListBucketsTool(manager)

        assert tool.metadata.id == "list_s3_buckets"
        assert tool.metadata.name == "List S3 Buckets"
        assert tool.metadata.category == "utility"

    def test_dynamodb_tool_metadata(self):
        """Test DynamoDB tool metadata."""
        manager = AWSManager()
        tool = GetDynamoDBItemTool(manager)

        assert tool.metadata.id == "get_dynamodb_item"
        assert "DynamoDB" in tool.metadata.name or "item" in tool.metadata.name.lower()

    def test_lambda_tool_metadata(self):
        """Test Lambda tool metadata."""
        manager = AWSManager()
        tool = InvokeFunctionTool(manager)

        assert tool.metadata.id == "invoke_lambda_function"
        assert "Lambda" in tool.metadata.name or "Function" in tool.metadata.name
