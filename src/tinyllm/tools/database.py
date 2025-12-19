"""Database query tools for TinyLLM.

This module provides tools for executing database queries safely,
with support for SQLite, PostgreSQL, and MySQL.
"""

import logging
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field

from tinyllm.tools.base import BaseTool, ToolMetadata

logger = logging.getLogger(__name__)


class DatabaseType(str, Enum):
    """Supported database types."""

    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


class QueryType(str, Enum):
    """Types of SQL queries."""

    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    ALTER = "alter"
    DROP = "drop"
    OTHER = "other"


@dataclass
class ColumnInfo:
    """Information about a database column."""

    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False
    default: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "data_type": self.data_type,
            "nullable": self.nullable,
            "primary_key": self.primary_key,
            "default": self.default,
            "extra": self.extra,
        }


@dataclass
class TableInfo:
    """Information about a database table."""

    name: str
    columns: List[ColumnInfo] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[Dict[str, Any]] = field(default_factory=list)
    indexes: List[str] = field(default_factory=list)
    row_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "columns": [c.to_dict() for c in self.columns],
            "primary_keys": self.primary_keys,
            "foreign_keys": self.foreign_keys,
            "indexes": self.indexes,
            "row_count": self.row_count,
        }


@dataclass
class QueryResult:
    """Result of a database query."""

    success: bool
    rows: List[Dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    affected_rows: int = 0
    columns: List[str] = field(default_factory=list)
    error: Optional[str] = None
    query_type: QueryType = QueryType.OTHER
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "rows": self.rows,
            "row_count": self.row_count,
            "affected_rows": self.affected_rows,
            "columns": self.columns,
            "error": self.error,
            "query_type": self.query_type.value,
            "execution_time_ms": self.execution_time_ms,
        }


class DatabaseConnection(ABC):
    """Abstract database connection interface."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to database."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close database connection."""
        pass

    @abstractmethod
    def execute(
        self,
        query: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> QueryResult:
        """Execute a query.

        Args:
            query: SQL query to execute.
            params: Query parameters.

        Returns:
            Query result.
        """
        pass

    @abstractmethod
    def get_tables(self) -> List[str]:
        """Get list of tables in database."""
        pass

    @abstractmethod
    def get_table_info(self, table_name: str) -> TableInfo:
        """Get information about a table.

        Args:
            table_name: Name of the table.

        Returns:
            Table information.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to database."""
        pass


class SQLiteConnection(DatabaseConnection):
    """SQLite database connection."""

    def __init__(self, database: str = ":memory:"):
        """Initialize SQLite connection.

        Args:
            database: Path to database file or ":memory:".
        """
        self.database = database
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Establish connection."""
        self._conn = sqlite3.connect(self.database)
        self._conn.row_factory = sqlite3.Row

    def disconnect(self) -> None:
        """Close connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._conn is not None

    def execute(
        self,
        query: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> QueryResult:
        """Execute a query."""
        import time

        if not self._conn:
            return QueryResult(
                success=False,
                error="Not connected to database",
            )

        start_time = time.time()

        try:
            cursor = self._conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            # Determine query type
            query_type = self._get_query_type(query)

            # Get results for SELECT queries and PRAGMA queries
            rows = []
            columns = []
            row_count = 0

            # PRAGMA queries return results like SELECT
            is_select_like = query_type == QueryType.SELECT or query.strip().upper().startswith("PRAGMA")
            if is_select_like:
                rows_data = cursor.fetchall()
                if rows_data:
                    columns = list(rows_data[0].keys()) if rows_data else []
                    rows = [dict(row) for row in rows_data]
                    row_count = len(rows)
            else:
                self._conn.commit()

            execution_time = (time.time() - start_time) * 1000

            return QueryResult(
                success=True,
                rows=rows,
                row_count=row_count,
                affected_rows=cursor.rowcount if cursor.rowcount >= 0 else 0,
                columns=columns,
                query_type=query_type,
                execution_time_ms=execution_time,
            )

        except sqlite3.Error as e:
            execution_time = (time.time() - start_time) * 1000
            return QueryResult(
                success=False,
                error=str(e),
                query_type=self._get_query_type(query),
                execution_time_ms=execution_time,
            )

    def _get_query_type(self, query: str) -> QueryType:
        """Determine query type from SQL."""
        query_upper = query.strip().upper()

        if query_upper.startswith("SELECT"):
            return QueryType.SELECT
        elif query_upper.startswith("INSERT"):
            return QueryType.INSERT
        elif query_upper.startswith("UPDATE"):
            return QueryType.UPDATE
        elif query_upper.startswith("DELETE"):
            return QueryType.DELETE
        elif query_upper.startswith("CREATE"):
            return QueryType.CREATE
        elif query_upper.startswith("ALTER"):
            return QueryType.ALTER
        elif query_upper.startswith("DROP"):
            return QueryType.DROP
        else:
            return QueryType.OTHER

    def get_tables(self) -> List[str]:
        """Get list of tables."""
        result = self.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [row["name"] for row in result.rows]

    def get_table_info(self, table_name: str) -> TableInfo:
        """Get table information."""
        # Get column info
        result = self.execute(f"PRAGMA table_info({table_name})")

        columns = []
        primary_keys = []

        for row in result.rows:
            col = ColumnInfo(
                name=row["name"],
                data_type=row["type"],
                nullable=not bool(row["notnull"]),
                primary_key=bool(row["pk"]),
                default=row["dflt_value"],
            )
            columns.append(col)
            if col.primary_key:
                primary_keys.append(col.name)

        # Get foreign keys
        fk_result = self.execute(f"PRAGMA foreign_key_list({table_name})")
        foreign_keys = [
            {
                "from": row["from"],
                "to_table": row["table"],
                "to_column": row["to"],
            }
            for row in fk_result.rows
        ]

        # Get indexes
        idx_result = self.execute(f"PRAGMA index_list({table_name})")
        indexes = [row["name"] for row in idx_result.rows]

        # Get row count
        count_result = self.execute(f"SELECT COUNT(*) as cnt FROM {table_name}")
        row_count = count_result.rows[0]["cnt"] if count_result.rows else 0

        return TableInfo(
            name=table_name,
            columns=columns,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            indexes=indexes,
            row_count=row_count,
        )


class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQL database connection (requires psycopg2)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "postgres",
        user: str = "postgres",
        password: str = "",
    ):
        """Initialize PostgreSQL connection."""
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self._conn = None

    def connect(self) -> None:
        """Establish connection."""
        try:
            import psycopg2
            import psycopg2.extras

            self._conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
            )
        except ImportError:
            raise ImportError("psycopg2 is required for PostgreSQL support")

    def disconnect(self) -> None:
        """Close connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._conn is not None and not self._conn.closed

    def execute(
        self,
        query: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> QueryResult:
        """Execute a query."""
        import time

        if not self._conn:
            return QueryResult(
                success=False,
                error="Not connected to database",
            )

        start_time = time.time()

        try:
            import psycopg2.extras

            cursor = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            query_type = self._get_query_type(query)

            rows = []
            columns = []
            row_count = 0

            if query_type == QueryType.SELECT:
                rows = [dict(row) for row in cursor.fetchall()]
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                row_count = len(rows)
            else:
                self._conn.commit()

            execution_time = (time.time() - start_time) * 1000

            return QueryResult(
                success=True,
                rows=rows,
                row_count=row_count,
                affected_rows=cursor.rowcount if cursor.rowcount >= 0 else 0,
                columns=columns,
                query_type=query_type,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            self._conn.rollback()
            execution_time = (time.time() - start_time) * 1000
            return QueryResult(
                success=False,
                error=str(e),
                query_type=self._get_query_type(query),
                execution_time_ms=execution_time,
            )

    def _get_query_type(self, query: str) -> QueryType:
        """Determine query type."""
        query_upper = query.strip().upper()

        if query_upper.startswith("SELECT"):
            return QueryType.SELECT
        elif query_upper.startswith("INSERT"):
            return QueryType.INSERT
        elif query_upper.startswith("UPDATE"):
            return QueryType.UPDATE
        elif query_upper.startswith("DELETE"):
            return QueryType.DELETE
        elif query_upper.startswith("CREATE"):
            return QueryType.CREATE
        elif query_upper.startswith("ALTER"):
            return QueryType.ALTER
        elif query_upper.startswith("DROP"):
            return QueryType.DROP
        else:
            return QueryType.OTHER

    def get_tables(self) -> List[str]:
        """Get list of tables."""
        result = self.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
            """
        )
        return [row["table_name"] for row in result.rows]

    def get_table_info(self, table_name: str) -> TableInfo:
        """Get table information."""
        # Get columns
        col_result = self.execute(
            """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
            """,
            (table_name,),
        )

        columns = [
            ColumnInfo(
                name=row["column_name"],
                data_type=row["data_type"],
                nullable=row["is_nullable"] == "YES",
                default=row["column_default"],
            )
            for row in col_result.rows
        ]

        # Get primary keys
        pk_result = self.execute(
            """
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            WHERE tc.table_name = %s AND tc.constraint_type = 'PRIMARY KEY'
            """,
            (table_name,),
        )
        primary_keys = [row["column_name"] for row in pk_result.rows]

        # Mark primary key columns
        for col in columns:
            if col.name in primary_keys:
                col.primary_key = True

        # Get row count
        count_result = self.execute(f"SELECT COUNT(*) as cnt FROM {table_name}")
        row_count = count_result.rows[0]["cnt"] if count_result.rows else 0

        return TableInfo(
            name=table_name,
            columns=columns,
            primary_keys=primary_keys,
            row_count=row_count,
        )


class MySQLConnection(DatabaseConnection):
    """MySQL database connection (requires mysql-connector-python)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        database: str = "mysql",
        user: str = "root",
        password: str = "",
    ):
        """Initialize MySQL connection."""
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self._conn = None

    def connect(self) -> None:
        """Establish connection."""
        try:
            import mysql.connector

            self._conn = mysql.connector.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
            )
        except ImportError:
            raise ImportError("mysql-connector-python is required for MySQL support")

    def disconnect(self) -> None:
        """Close connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._conn is not None and self._conn.is_connected()

    def execute(
        self,
        query: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> QueryResult:
        """Execute a query."""
        import time

        if not self._conn:
            return QueryResult(
                success=False,
                error="Not connected to database",
            )

        start_time = time.time()

        try:
            cursor = self._conn.cursor(dictionary=True)

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            query_type = self._get_query_type(query)

            rows = []
            columns = []
            row_count = 0

            if query_type == QueryType.SELECT:
                rows = cursor.fetchall()
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                row_count = len(rows)
            else:
                self._conn.commit()

            execution_time = (time.time() - start_time) * 1000

            return QueryResult(
                success=True,
                rows=rows,
                row_count=row_count,
                affected_rows=cursor.rowcount if cursor.rowcount >= 0 else 0,
                columns=columns,
                query_type=query_type,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            self._conn.rollback()
            execution_time = (time.time() - start_time) * 1000
            return QueryResult(
                success=False,
                error=str(e),
                query_type=self._get_query_type(query),
                execution_time_ms=execution_time,
            )

    def _get_query_type(self, query: str) -> QueryType:
        """Determine query type."""
        query_upper = query.strip().upper()

        if query_upper.startswith("SELECT"):
            return QueryType.SELECT
        elif query_upper.startswith("INSERT"):
            return QueryType.INSERT
        elif query_upper.startswith("UPDATE"):
            return QueryType.UPDATE
        elif query_upper.startswith("DELETE"):
            return QueryType.DELETE
        elif query_upper.startswith("CREATE"):
            return QueryType.CREATE
        elif query_upper.startswith("ALTER"):
            return QueryType.ALTER
        elif query_upper.startswith("DROP"):
            return QueryType.DROP
        else:
            return QueryType.OTHER

    def get_tables(self) -> List[str]:
        """Get list of tables."""
        result = self.execute("SHOW TABLES")
        if result.rows:
            # MySQL returns column name as Tables_in_<dbname>
            key = list(result.rows[0].keys())[0]
            return [row[key] for row in result.rows]
        return []

    def get_table_info(self, table_name: str) -> TableInfo:
        """Get table information."""
        # Get columns
        col_result = self.execute(f"DESCRIBE {table_name}")

        columns = []
        primary_keys = []

        for row in col_result.rows:
            col = ColumnInfo(
                name=row["Field"],
                data_type=row["Type"],
                nullable=row["Null"] == "YES",
                primary_key=row["Key"] == "PRI",
                default=row["Default"],
            )
            columns.append(col)
            if col.primary_key:
                primary_keys.append(col.name)

        # Get row count
        count_result = self.execute(f"SELECT COUNT(*) as cnt FROM {table_name}")
        row_count = count_result.rows[0]["cnt"] if count_result.rows else 0

        return TableInfo(
            name=table_name,
            columns=columns,
            primary_keys=primary_keys,
            row_count=row_count,
        )


class DatabaseManager:
    """Manager for database connections."""

    def __init__(self):
        """Initialize manager."""
        self._connections: Dict[str, DatabaseConnection] = {}

    def add_connection(self, name: str, connection: DatabaseConnection) -> None:
        """Add a database connection.

        Args:
            name: Connection name.
            connection: Database connection.
        """
        self._connections[name] = connection

    def get_connection(self, name: str) -> Optional[DatabaseConnection]:
        """Get a database connection.

        Args:
            name: Connection name.

        Returns:
            Database connection or None.
        """
        return self._connections.get(name)

    def remove_connection(self, name: str) -> bool:
        """Remove a database connection.

        Args:
            name: Connection name.

        Returns:
            True if removed.
        """
        if name in self._connections:
            conn = self._connections.pop(name)
            conn.disconnect()
            return True
        return False

    def list_connections(self) -> List[str]:
        """List all connection names."""
        return list(self._connections.keys())

    def close_all(self) -> None:
        """Close all connections."""
        for conn in self._connections.values():
            conn.disconnect()
        self._connections.clear()


# Pydantic models for tool inputs/outputs


class DatabaseQueryInput(BaseModel):
    """Input for database query tool."""

    connection: str = Field(
        default="default",
        description="Name of the database connection to use",
    )
    query: str = Field(
        ...,
        description="SQL query to execute",
    )
    params: Optional[List[Any]] = Field(
        default=None,
        description="Query parameters for parameterized queries",
    )


class DatabaseQueryOutput(BaseModel):
    """Output from database query tool."""

    success: bool = Field(description="Whether query succeeded")
    rows: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Result rows for SELECT queries",
    )
    row_count: int = Field(
        default=0,
        description="Number of rows returned",
    )
    affected_rows: int = Field(
        default=0,
        description="Number of rows affected for INSERT/UPDATE/DELETE",
    )
    columns: List[str] = Field(
        default_factory=list,
        description="Column names in result",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if query failed",
    )
    query_type: str = Field(
        default="other",
        description="Type of query executed",
    )
    execution_time_ms: float = Field(
        default=0.0,
        description="Query execution time in milliseconds",
    )


class ListTablesInput(BaseModel):
    """Input for list tables tool."""

    connection: str = Field(
        default="default",
        description="Name of the database connection",
    )


class ListTablesOutput(BaseModel):
    """Output from list tables tool."""

    success: bool = Field(description="Whether operation succeeded")
    tables: List[str] = Field(
        default_factory=list,
        description="List of table names",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if operation failed",
    )


class TableInfoInput(BaseModel):
    """Input for table info tool."""

    connection: str = Field(
        default="default",
        description="Name of the database connection",
    )
    table_name: str = Field(
        ...,
        description="Name of the table to get info for",
    )


class TableInfoOutput(BaseModel):
    """Output from table info tool."""

    success: bool = Field(description="Whether operation succeeded")
    table: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Table information",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if operation failed",
    )


# Tool implementations


class DatabaseQueryTool(BaseTool[DatabaseQueryInput, DatabaseQueryOutput]):
    """Tool for executing database queries."""

    metadata = ToolMetadata(
        id="database_query",
        name="Database Query",
        description="Execute SQL queries against a database",
        category="utility",
    )
    input_type = DatabaseQueryInput
    output_type = DatabaseQueryOutput

    def __init__(self, manager: DatabaseManager):
        """Initialize tool.

        Args:
            manager: Database manager with connections.
        """
        self.manager = manager

    async def execute(self, input: DatabaseQueryInput) -> DatabaseQueryOutput:
        """Execute a database query."""
        conn = self.manager.get_connection(input.connection)

        if not conn:
            return DatabaseQueryOutput(
                success=False,
                error=f"Connection '{input.connection}' not found",
            )

        if not conn.is_connected():
            return DatabaseQueryOutput(
                success=False,
                error=f"Connection '{input.connection}' is not connected",
            )

        params = tuple(input.params) if input.params else None
        result = conn.execute(input.query, params)

        return DatabaseQueryOutput(
            success=result.success,
            rows=result.rows,
            row_count=result.row_count,
            affected_rows=result.affected_rows,
            columns=result.columns,
            error=result.error,
            query_type=result.query_type.value,
            execution_time_ms=result.execution_time_ms,
        )


class ListTablesTool(BaseTool[ListTablesInput, ListTablesOutput]):
    """Tool for listing database tables."""

    metadata = ToolMetadata(
        id="list_tables",
        name="List Tables",
        description="List all tables in a database",
        category="utility",
    )
    input_type = ListTablesInput
    output_type = ListTablesOutput

    def __init__(self, manager: DatabaseManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: ListTablesInput) -> ListTablesOutput:
        """List database tables."""
        conn = self.manager.get_connection(input.connection)

        if not conn:
            return ListTablesOutput(
                success=False,
                error=f"Connection '{input.connection}' not found",
            )

        if not conn.is_connected():
            return ListTablesOutput(
                success=False,
                error=f"Connection '{input.connection}' is not connected",
            )

        try:
            tables = conn.get_tables()
            return ListTablesOutput(
                success=True,
                tables=tables,
            )
        except Exception as e:
            return ListTablesOutput(
                success=False,
                error=str(e),
            )


class TableInfoTool(BaseTool[TableInfoInput, TableInfoOutput]):
    """Tool for getting table information."""

    metadata = ToolMetadata(
        id="table_info",
        name="Table Info",
        description="Get detailed information about a database table",
        category="utility",
    )
    input_type = TableInfoInput
    output_type = TableInfoOutput

    def __init__(self, manager: DatabaseManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: TableInfoInput) -> TableInfoOutput:
        """Get table information."""
        conn = self.manager.get_connection(input.connection)

        if not conn:
            return TableInfoOutput(
                success=False,
                error=f"Connection '{input.connection}' not found",
            )

        if not conn.is_connected():
            return TableInfoOutput(
                success=False,
                error=f"Connection '{input.connection}' is not connected",
            )

        try:
            info = conn.get_table_info(input.table_name)
            return TableInfoOutput(
                success=True,
                table=info.to_dict(),
            )
        except Exception as e:
            return TableInfoOutput(
                success=False,
                error=str(e),
            )


# Convenience functions


def create_sqlite_connection(database: str = ":memory:") -> SQLiteConnection:
    """Create a SQLite connection.

    Args:
        database: Path to database or ":memory:".

    Returns:
        SQLite connection.
    """
    return SQLiteConnection(database=database)


def create_postgresql_connection(
    host: str = "localhost",
    port: int = 5432,
    database: str = "postgres",
    user: str = "postgres",
    password: str = "",
) -> PostgreSQLConnection:
    """Create a PostgreSQL connection.

    Args:
        host: Database host.
        port: Database port.
        database: Database name.
        user: Username.
        password: Password.

    Returns:
        PostgreSQL connection.
    """
    return PostgreSQLConnection(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
    )


def create_mysql_connection(
    host: str = "localhost",
    port: int = 3306,
    database: str = "mysql",
    user: str = "root",
    password: str = "",
) -> MySQLConnection:
    """Create a MySQL connection.

    Args:
        host: Database host.
        port: Database port.
        database: Database name.
        user: Username.
        password: Password.

    Returns:
        MySQL connection.
    """
    return MySQLConnection(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
    )


def create_database_manager() -> DatabaseManager:
    """Create a database manager.

    Returns:
        Database manager.
    """
    return DatabaseManager()


def create_database_tools(
    manager: DatabaseManager,
) -> Dict[str, BaseTool]:
    """Create all database tools.

    Args:
        manager: Database manager.

    Returns:
        Dictionary of tool name to tool instance.
    """
    return {
        "database_query": DatabaseQueryTool(manager),
        "list_tables": ListTablesTool(manager),
        "table_info": TableInfoTool(manager),
    }
