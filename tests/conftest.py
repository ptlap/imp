"""
Shared pytest fixtures and configuration for IMP tests.

This module provides common fixtures used across multiple test modules,
including test images, temporary directories, and mock objects.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from PIL import Image


@pytest.fixture
def temp_dir():
    """
    Create temporary directory for test files.
    
    Yields:
        Path: Path to temporary directory
        
    Cleanup:
        Automatically removes directory after test completes
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image_small():
    """
    Create small test image (64x64).
    
    Returns:
        np.ndarray: Small RGB image as float32 array
    """
    return np.random.rand(64, 64, 3).astype(np.float32)


@pytest.fixture
def sample_image_medium():
    """
    Create medium test image (256x256).
    
    Returns:
        np.ndarray: Medium RGB image as float32 array
    """
    return np.random.rand(256, 256, 3).astype(np.float32)


@pytest.fixture
def sample_image_large():
    """
    Create large test image (1024x1024).
    
    Returns:
        np.ndarray: Large RGB image as float32 array
    """
    return np.random.rand(1024, 1024, 3).astype(np.float32)


@pytest.fixture
def sample_grayscale_image():
    """
    Create grayscale test image (all channels equal).
    
    Returns:
        np.ndarray: Grayscale RGB image as float32 array
    """
    gray_value = 0.5
    return np.ones((256, 256, 3), dtype=np.float32) * gray_value


@pytest.fixture
def sample_color_image():
    """
    Create color test image (red).
    
    Returns:
        np.ndarray: Color RGB image as float32 array
    """
    image = np.zeros((256, 256, 3), dtype=np.float32)
    image[:, :, 0] = 1.0  # Red channel
    return image


@pytest.fixture
def sample_noisy_image():
    """
    Create noisy test image.
    
    Returns:
        np.ndarray: Noisy RGB image as float32 array
    """
    clean = np.ones((256, 256, 3), dtype=np.float32) * 0.5
    noise = np.random.normal(0, 0.1, clean.shape).astype(np.float32)
    noisy = np.clip(clean + noise, 0, 1)
    return noisy


def create_test_image_file(path: Path, size: tuple = (256, 256), color: tuple = (255, 0, 0)):
    """
    Helper function to create test image file.
    
    Args:
        path: Path where image should be saved
        size: Image size as (width, height)
        color: RGB color tuple (0-255)
        
    Returns:
        Path: Path to created image file
    """
    img = Image.new('RGB', size, color)
    img.save(path)
    return path


def create_grayscale_image_file(path: Path, size: tuple = (256, 256), gray_value: int = 128):
    """
    Helper function to create grayscale image file.
    
    Args:
        path: Path where image should be saved
        size: Image size as (width, height)
        gray_value: Gray value (0-255)
        
    Returns:
        Path: Path to created image file
    """
    img = Image.new('RGB', size, (gray_value, gray_value, gray_value))
    img.save(path)
    return path


@pytest.fixture
def test_image_file(temp_dir):
    """
    Create test image file in temporary directory.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Path: Path to test image file
    """
    return create_test_image_file(temp_dir / "test.jpg")


@pytest.fixture
def test_grayscale_file(temp_dir):
    """
    Create grayscale test image file in temporary directory.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Path: Path to grayscale test image file
    """
    return create_grayscale_image_file(temp_dir / "gray.jpg")


@pytest.fixture
def sample_metadata():
    """
    Create sample metadata dictionary.
    
    Returns:
        dict: Sample metadata with common fields
    """
    return {
        'original_size': (512, 512),
        'is_grayscale': False,
        'resize_factor': 1.0
    }


# Pytest configuration hooks

def pytest_configure(config):
    """
    Configure pytest with custom settings.
    
    Args:
        config: Pytest config object
    """
    # Register custom markers
    config.addinivalue_line("markers", "unit: Unit tests for individual modules")
    config.addinivalue_line("markers", "integration: Integration tests for pipeline")
    config.addinivalue_line("markers", "slow: Tests that take longer to run")
    config.addinivalue_line("markers", "gpu: Tests that require GPU")


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers automatically.
    
    Args:
        config: Pytest config object
        items: List of collected test items
    """
    for item in items:
        # Add 'unit' marker to all tests in test_*.py files (except test_pipeline.py)
        if "test_pipeline" not in item.nodeid:
            item.add_marker(pytest.mark.unit)
        else:
            # Add 'integration' marker to pipeline tests
            item.add_marker(pytest.mark.integration)
        
        # Add 'gpu' marker to tests that mention GPU in name
        if "gpu" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)
