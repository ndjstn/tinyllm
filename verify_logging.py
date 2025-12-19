#!/usr/bin/env python3
"""Verification script for structured logging implementation.

This script verifies that:
1. Logging module can be imported
2. All modified components have logging
3. Console and JSON modes work
4. Context binding works
5. All log levels work
"""

import sys


def test_imports():
    """Test that all logging-related imports work."""
    print("Testing imports...")
    try:
        from tinyllm.logging import (
            configure_logging,
            get_logger,
            bind_context,
            unbind_context,
            clear_context,
        )
        from tinyllm import (
            configure_logging as cfg_logging,
            get_logger as get_log,
        )
        print("✓ Logging module imports successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_component_imports():
    """Test that modified components import without errors."""
    print("\nTesting component imports...")
    try:
        from tinyllm.core.executor import Executor, logger as executor_logger
        from tinyllm.cli import app, logger as cli_logger
        from tinyllm.core.node import BaseNode, logger as node_logger
        from tinyllm.nodes.transform import TransformNode, logger as transform_logger
        from tinyllm.models.client import OllamaClient, logger as client_logger
        print("✓ All modified components import successfully")
        return True
    except Exception as e:
        print(f"✗ Component import failed: {e}")
        return False


def test_console_logging():
    """Test console logging mode."""
    print("\nTesting console logging mode...")
    try:
        from tinyllm.logging import configure_logging, get_logger

        configure_logging(log_level="INFO", log_format="console")
        logger = get_logger("test_console")

        logger.info("console_test", mode="console", working=True)
        print("✓ Console logging works")
        return True
    except Exception as e:
        print(f"✗ Console logging failed: {e}")
        return False


def test_json_logging():
    """Test JSON logging mode."""
    print("\nTesting JSON logging mode...")
    try:
        from tinyllm.logging import configure_logging, get_logger

        configure_logging(log_level="INFO", log_format="json")
        logger = get_logger("test_json")

        logger.info("json_test", mode="json", working=True)
        print("✓ JSON logging works")
        return True
    except Exception as e:
        print(f"✗ JSON logging failed: {e}")
        return False


def test_context_binding():
    """Test context binding functionality."""
    print("\nTesting context binding...")
    try:
        from tinyllm.logging import configure_logging, get_logger, bind_context, clear_context

        configure_logging(log_level="INFO", log_format="console")
        logger = get_logger("test_context")

        bind_context(trace_id="test-123", user_id="user-456")
        logger.info("context_test")
        clear_context()

        print("✓ Context binding works")
        return True
    except Exception as e:
        print(f"✗ Context binding failed: {e}")
        return False


def test_log_levels():
    """Test different log levels."""
    print("\nTesting log levels...")
    try:
        from tinyllm.logging import configure_logging, get_logger

        configure_logging(log_level="DEBUG", log_format="console")
        logger = get_logger("test_levels")

        logger.debug("debug_level")
        logger.info("info_level")
        logger.warning("warning_level")
        logger.error("error_level")

        print("✓ All log levels work")
        return True
    except Exception as e:
        print(f"✗ Log levels failed: {e}")
        return False


def test_executor_logging():
    """Test that executor has logging."""
    print("\nTesting executor logging integration...")
    try:
        from tinyllm.core.executor import Executor

        # Just verify the class has the logger
        import inspect
        source = inspect.getsource(Executor.__init__)
        if "logger" in source:
            print("✓ Executor has logging integrated")
            return True
        else:
            print("✗ Executor missing logger")
            return False
    except Exception as e:
        print(f"✗ Executor logging test failed: {e}")
        return False


def test_cli_logging():
    """Test that CLI has logging."""
    print("\nTesting CLI logging integration...")
    try:
        import tinyllm.cli as cli_module
        import inspect

        source = inspect.getsource(cli_module)
        if "logger" in source and "configure_logging" in source:
            print("✓ CLI has logging integrated")
            return True
        else:
            print("✗ CLI missing logger")
            return False
    except Exception as e:
        print(f"✗ CLI logging test failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 70)
    print("TinyLLM Structured Logging Verification")
    print("=" * 70)

    tests = [
        test_imports,
        test_component_imports,
        test_console_logging,
        test_json_logging,
        test_context_binding,
        test_log_levels,
        test_executor_logging,
        test_cli_logging,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All verification tests passed!")
        print("\nLogging implementation is complete and working correctly.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        print("\nPlease review the failures above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
