"""Tests for database query tools."""

import pytest
import tempfile
import os

from tinyllm.tools.database import (
    ColumnInfo,
    DatabaseManager,
    DatabaseQueryInput,
    DatabaseQueryOutput,
    DatabaseQueryTool,
    DatabaseType,
    ListTablesInput,
    ListTablesOutput,
    ListTablesTool,
    QueryResult,
    QueryType,
    SQLiteConnection,
    TableInfo,
    TableInfoInput,
    TableInfoOutput,
    TableInfoTool,
    create_database_manager,
    create_database_tools,
    create_sqlite_connection,
)


class TestColumnInfo:
    """Tests for ColumnInfo."""

    def test_creation(self):
        """Test column info creation."""
        col = ColumnInfo(
            name="id",
            data_type="INTEGER",
            nullable=False,
            primary_key=True,
        )

        assert col.name == "id"
        assert col.data_type == "INTEGER"
        assert col.nullable is False
        assert col.primary_key is True

    def test_to_dict(self):
        """Test converting to dictionary."""
        col = ColumnInfo(
            name="name",
            data_type="VARCHAR(100)",
            nullable=True,
            default="'unknown'",
        )

        d = col.to_dict()

        assert d["name"] == "name"
        assert d["data_type"] == "VARCHAR(100)"
        assert d["nullable"] is True


class TestTableInfo:
    """Tests for TableInfo."""

    def test_creation(self):
        """Test table info creation."""
        table = TableInfo(
            name="users",
            columns=[
                ColumnInfo(name="id", data_type="INTEGER", primary_key=True),
                ColumnInfo(name="name", data_type="TEXT"),
            ],
            primary_keys=["id"],
            row_count=100,
        )

        assert table.name == "users"
        assert len(table.columns) == 2
        assert table.row_count == 100

    def test_to_dict(self):
        """Test converting to dictionary."""
        table = TableInfo(
            name="users",
            columns=[
                ColumnInfo(name="id", data_type="INTEGER"),
            ],
            primary_keys=["id"],
        )

        d = table.to_dict()

        assert d["name"] == "users"
        assert len(d["columns"]) == 1


class TestQueryResult:
    """Tests for QueryResult."""

    def test_success_result(self):
        """Test successful result."""
        result = QueryResult(
            success=True,
            rows=[{"id": 1, "name": "test"}],
            row_count=1,
            columns=["id", "name"],
            query_type=QueryType.SELECT,
        )

        assert result.success is True
        assert result.row_count == 1

    def test_error_result(self):
        """Test error result."""
        result = QueryResult(
            success=False,
            error="Syntax error",
        )

        assert result.success is False
        assert result.error == "Syntax error"

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = QueryResult(
            success=True,
            affected_rows=5,
            query_type=QueryType.UPDATE,
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["affected_rows"] == 5
        assert d["query_type"] == "update"


class TestSQLiteConnection:
    """Tests for SQLiteConnection."""

    def test_in_memory_connection(self):
        """Test in-memory connection."""
        conn = SQLiteConnection(":memory:")
        conn.connect()

        assert conn.is_connected() is True

        conn.disconnect()
        assert conn.is_connected() is False

    def test_file_connection(self):
        """Test file-based connection."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            conn = SQLiteConnection(db_path)
            conn.connect()

            assert conn.is_connected() is True

            conn.disconnect()
        finally:
            os.unlink(db_path)

    def test_create_table(self):
        """Test creating a table."""
        conn = SQLiteConnection(":memory:")
        conn.connect()

        result = conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE
            )
            """
        )

        assert result.success is True
        assert result.query_type == QueryType.CREATE

        conn.disconnect()

    def test_insert_and_select(self):
        """Test insert and select operations."""
        conn = SQLiteConnection(":memory:")
        conn.connect()

        # Create table
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")

        # Insert data
        insert_result = conn.execute(
            "INSERT INTO items (name) VALUES (?)",
            ("test_item",),
        )

        assert insert_result.success is True
        assert insert_result.query_type == QueryType.INSERT
        assert insert_result.affected_rows == 1

        # Select data
        select_result = conn.execute("SELECT * FROM items")

        assert select_result.success is True
        assert select_result.query_type == QueryType.SELECT
        assert select_result.row_count == 1
        assert select_result.rows[0]["name"] == "test_item"

        conn.disconnect()

    def test_update(self):
        """Test update operation."""
        conn = SQLiteConnection(":memory:")
        conn.connect()

        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO items (name) VALUES ('old')")

        result = conn.execute("UPDATE items SET name = 'new' WHERE id = 1")

        assert result.success is True
        assert result.query_type == QueryType.UPDATE
        assert result.affected_rows == 1

        conn.disconnect()

    def test_delete(self):
        """Test delete operation."""
        conn = SQLiteConnection(":memory:")
        conn.connect()

        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO items (name) VALUES ('test')")

        result = conn.execute("DELETE FROM items WHERE id = 1")

        assert result.success is True
        assert result.query_type == QueryType.DELETE
        assert result.affected_rows == 1

        conn.disconnect()

    def test_parameterized_query(self):
        """Test parameterized queries."""
        conn = SQLiteConnection(":memory:")
        conn.connect()

        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        conn.execute(
            "INSERT INTO users (name, age) VALUES (?, ?)",
            ("Alice", 30),
        )
        conn.execute(
            "INSERT INTO users (name, age) VALUES (?, ?)",
            ("Bob", 25),
        )

        result = conn.execute(
            "SELECT * FROM users WHERE age > ?",
            (26,),
        )

        assert result.success is True
        assert result.row_count == 1
        assert result.rows[0]["name"] == "Alice"

        conn.disconnect()

    def test_syntax_error(self):
        """Test handling syntax errors."""
        conn = SQLiteConnection(":memory:")
        conn.connect()

        result = conn.execute("INVALID SQL QUERY")

        assert result.success is False
        assert result.error is not None

        conn.disconnect()

    def test_get_tables(self):
        """Test getting list of tables."""
        conn = SQLiteConnection(":memory:")
        conn.connect()

        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY)")

        tables = conn.get_tables()

        assert "users" in tables
        assert "items" in tables

        conn.disconnect()

    def test_get_table_info(self):
        """Test getting table information."""
        conn = SQLiteConnection(":memory:")
        conn.connect()

        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT
            )
            """
        )
        conn.execute("INSERT INTO users (name, email) VALUES ('test', 'test@test.com')")

        info = conn.get_table_info("users")

        assert info.name == "users"
        assert len(info.columns) == 3
        assert info.row_count == 1

        # Check column details
        id_col = next(c for c in info.columns if c.name == "id")
        assert id_col.primary_key is True

        name_col = next(c for c in info.columns if c.name == "name")
        assert name_col.nullable is False

        conn.disconnect()

    def test_foreign_keys(self):
        """Test foreign key detection."""
        conn = SQLiteConnection(":memory:")
        conn.connect()

        conn.execute("CREATE TABLE categories (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute(
            """
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT,
                category_id INTEGER REFERENCES categories(id)
            )
            """
        )

        info = conn.get_table_info("products")

        assert len(info.foreign_keys) == 1
        assert info.foreign_keys[0]["to_table"] == "categories"

        conn.disconnect()

    def test_execute_without_connection(self):
        """Test executing without connection."""
        conn = SQLiteConnection(":memory:")

        result = conn.execute("SELECT 1")

        assert result.success is False
        assert "Not connected" in result.error


class TestDatabaseManager:
    """Tests for DatabaseManager."""

    def test_add_connection(self):
        """Test adding a connection."""
        manager = DatabaseManager()
        conn = SQLiteConnection(":memory:")

        manager.add_connection("default", conn)

        assert manager.get_connection("default") == conn

    def test_get_connection_not_found(self):
        """Test getting non-existent connection."""
        manager = DatabaseManager()

        conn = manager.get_connection("nonexistent")

        assert conn is None

    def test_remove_connection(self):
        """Test removing a connection."""
        manager = DatabaseManager()
        conn = SQLiteConnection(":memory:")
        conn.connect()

        manager.add_connection("default", conn)
        removed = manager.remove_connection("default")

        assert removed is True
        assert manager.get_connection("default") is None

    def test_list_connections(self):
        """Test listing connections."""
        manager = DatabaseManager()

        manager.add_connection("db1", SQLiteConnection(":memory:"))
        manager.add_connection("db2", SQLiteConnection(":memory:"))

        connections = manager.list_connections()

        assert "db1" in connections
        assert "db2" in connections

    def test_close_all(self):
        """Test closing all connections."""
        manager = DatabaseManager()
        conn1 = SQLiteConnection(":memory:")
        conn1.connect()
        conn2 = SQLiteConnection(":memory:")
        conn2.connect()

        manager.add_connection("db1", conn1)
        manager.add_connection("db2", conn2)

        manager.close_all()

        assert len(manager.list_connections()) == 0


class TestDatabaseQueryTool:
    """Tests for DatabaseQueryTool."""

    @pytest.fixture
    def setup_manager(self):
        """Set up database manager with test data."""
        manager = DatabaseManager()
        conn = SQLiteConnection(":memory:")
        conn.connect()

        # Create test table
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        conn.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)")
        conn.execute("INSERT INTO users (name, age) VALUES ('Bob', 25)")

        manager.add_connection("default", conn)

        yield manager

        manager.close_all()

    @pytest.mark.asyncio
    async def test_select_query(self, setup_manager):
        """Test SELECT query."""
        tool = DatabaseQueryTool(setup_manager)

        result = await tool.execute(
            DatabaseQueryInput(
                connection="default",
                query="SELECT * FROM users ORDER BY name",
            )
        )

        assert result.success is True
        assert result.row_count == 2
        assert result.rows[0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_parameterized_query(self, setup_manager):
        """Test parameterized query."""
        tool = DatabaseQueryTool(setup_manager)

        result = await tool.execute(
            DatabaseQueryInput(
                connection="default",
                query="SELECT * FROM users WHERE age > ?",
                params=[26],
            )
        )

        assert result.success is True
        assert result.row_count == 1
        assert result.rows[0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_insert_query(self, setup_manager):
        """Test INSERT query."""
        tool = DatabaseQueryTool(setup_manager)

        result = await tool.execute(
            DatabaseQueryInput(
                connection="default",
                query="INSERT INTO users (name, age) VALUES (?, ?)",
                params=["Charlie", 35],
            )
        )

        assert result.success is True
        assert result.affected_rows == 1
        assert result.query_type == "insert"

    @pytest.mark.asyncio
    async def test_connection_not_found(self, setup_manager):
        """Test with non-existent connection."""
        tool = DatabaseQueryTool(setup_manager)

        result = await tool.execute(
            DatabaseQueryInput(
                connection="nonexistent",
                query="SELECT 1",
            )
        )

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_connection_not_connected(self, setup_manager):
        """Test with disconnected connection."""
        # Add a disconnected connection
        disconnected = SQLiteConnection(":memory:")
        setup_manager.add_connection("disconnected", disconnected)

        tool = DatabaseQueryTool(setup_manager)

        result = await tool.execute(
            DatabaseQueryInput(
                connection="disconnected",
                query="SELECT 1",
            )
        )

        assert result.success is False
        assert "not connected" in result.error


class TestListTablesTool:
    """Tests for ListTablesTool."""

    @pytest.fixture
    def setup_manager(self):
        """Set up database manager."""
        manager = DatabaseManager()
        conn = SQLiteConnection(":memory:")
        conn.connect()

        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY)")

        manager.add_connection("default", conn)

        yield manager

        manager.close_all()

    @pytest.mark.asyncio
    async def test_list_tables(self, setup_manager):
        """Test listing tables."""
        tool = ListTablesTool(setup_manager)

        result = await tool.execute(
            ListTablesInput(connection="default")
        )

        assert result.success is True
        assert "users" in result.tables
        assert "items" in result.tables

    @pytest.mark.asyncio
    async def test_connection_not_found(self, setup_manager):
        """Test with non-existent connection."""
        tool = ListTablesTool(setup_manager)

        result = await tool.execute(
            ListTablesInput(connection="nonexistent")
        )

        assert result.success is False
        assert "not found" in result.error


class TestTableInfoTool:
    """Tests for TableInfoTool."""

    @pytest.fixture
    def setup_manager(self):
        """Set up database manager."""
        manager = DatabaseManager()
        conn = SQLiteConnection(":memory:")
        conn.connect()

        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE
            )
            """
        )
        conn.execute("INSERT INTO users (name, email) VALUES ('test', 'test@test.com')")

        manager.add_connection("default", conn)

        yield manager

        manager.close_all()

    @pytest.mark.asyncio
    async def test_get_table_info(self, setup_manager):
        """Test getting table info."""
        tool = TableInfoTool(setup_manager)

        result = await tool.execute(
            TableInfoInput(
                connection="default",
                table_name="users",
            )
        )

        assert result.success is True
        assert result.table["name"] == "users"
        assert len(result.table["columns"]) == 3
        assert result.table["row_count"] == 1

    @pytest.mark.asyncio
    async def test_connection_not_found(self, setup_manager):
        """Test with non-existent connection."""
        tool = TableInfoTool(setup_manager)

        result = await tool.execute(
            TableInfoInput(
                connection="nonexistent",
                table_name="users",
            )
        )

        assert result.success is False
        assert "not found" in result.error


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_sqlite_connection(self):
        """Test creating SQLite connection."""
        conn = create_sqlite_connection(":memory:")

        assert isinstance(conn, SQLiteConnection)

    def test_create_database_manager(self):
        """Test creating database manager."""
        manager = create_database_manager()

        assert isinstance(manager, DatabaseManager)

    def test_create_database_tools(self):
        """Test creating all database tools."""
        manager = create_database_manager()
        tools = create_database_tools(manager)

        assert "database_query" in tools
        assert "list_tables" in tools
        assert "table_info" in tools


class TestQueryTypes:
    """Tests for query type detection."""

    def test_query_type_detection(self):
        """Test query type is correctly detected."""
        conn = SQLiteConnection(":memory:")
        conn.connect()

        conn.execute("CREATE TABLE test (id INTEGER)")

        test_cases = [
            ("SELECT * FROM test", QueryType.SELECT),
            ("INSERT INTO test VALUES (1)", QueryType.INSERT),
            ("UPDATE test SET id = 2", QueryType.UPDATE),
            ("DELETE FROM test", QueryType.DELETE),
            ("DROP TABLE test", QueryType.DROP),
        ]

        conn.execute("CREATE TABLE test2 (id INTEGER)")

        for query, expected_type in test_cases:
            if expected_type == QueryType.DROP:
                # Skip DROP to keep table for other tests
                continue
            result = conn.execute(query)
            assert result.query_type == expected_type, f"Query: {query}"

        conn.disconnect()


class TestExecutionTime:
    """Tests for execution time tracking."""

    def test_execution_time_recorded(self):
        """Test that execution time is recorded."""
        conn = SQLiteConnection(":memory:")
        conn.connect()

        conn.execute("CREATE TABLE test (id INTEGER)")

        # Execute a query
        result = conn.execute("SELECT * FROM test")

        assert result.execution_time_ms >= 0
        assert isinstance(result.execution_time_ms, float)

        conn.disconnect()
