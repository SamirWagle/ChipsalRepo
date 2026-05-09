"""
Test configuration file for pytest.

Defines fixtures, markers, and settings for all tests.
"""

import os
import sys
import warnings
from pathlib import Path

import pytest
import torch


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests"
    )


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment."""
    # Set random seeds
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Reduce logging noise
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Filter warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    
    yield
    
    # Cleanup after all tests
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


@pytest.fixture(scope="function")
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="function", autouse=True)
def reset_cuda():
    """Reset CUDA state between tests."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test name
        if "slow" in item.nodeid or "integration" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        if "gpu" in item.nodeid or "cuda" in item.nodeid:
            item.add_marker(pytest.mark.gpu)
        
        # Skip GPU tests if CUDA not available
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(pytest.mark.skip(reason="CUDA not available"))


# Custom assertions
def assert_tensor_equal(actual, expected, atol=1e-6):
    """Assert tensors are equal within tolerance."""
    assert torch.allclose(actual, expected, atol=atol), \
        f"Tensors not equal. Max diff: {torch.max(torch.abs(actual - expected))}"


def assert_shape(tensor, expected_shape):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, \
        f"Shape mismatch: got {tensor.shape}, expected {expected_shape}"


def assert_no_nan_inf(tensor):
    """Assert tensor contains no NaN or Inf values."""
    assert not torch.isnan(tensor).any(), "Tensor contains NaN"
    assert not torch.isinf(tensor).any(), "Tensor contains Inf"


# Register custom assertions
pytest.register_assert_rewrite("test_utils")
