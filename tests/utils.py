"""
Test utilities and helper functions for IMP tests.

This module provides common utilities for creating test data,
mocking models, and asserting image properties.
"""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional
from unittest.mock import Mock, MagicMock


# ============================================================================
# Test Image Generation Functions
# ============================================================================

def create_test_image(
    size: Tuple[int, int] = (256, 256),
    color: Tuple[int, int, int] = (255, 0, 0),
    dtype: type = np.uint8
) -> np.ndarray:
    """
    Create a test image array with specified size and color.
    
    Args:
        size: Image size as (height, width)
        color: RGB color tuple (0-255)
        dtype: Data type for the array (default: np.uint8)
        
    Returns:
        np.ndarray: Test image array of shape (height, width, 3)
    """
    image = np.zeros((size[0], size[1], 3), dtype=dtype)
    if dtype == np.uint8:
        image[:, :] = color
    else:  # float32
        image[:, :] = [c / 255.0 for c in color]
    return image


def create_grayscale_test_image(
    size: Tuple[int, int] = (256, 256),
    gray_value: int = 128,
    dtype: type = np.uint8
) -> np.ndarray:
    """
    Create a grayscale test image (all channels equal).
    
    Args:
        size: Image size as (height, width)
        gray_value: Gray value (0-255)
        dtype: Data type for the array (default: np.uint8)
        
    Returns:
        np.ndarray: Grayscale test image array
    """
    image = np.ones((size[0], size[1], 3), dtype=dtype)
    if dtype == np.uint8:
        image *= gray_value
    else:  # float32
        image *= (gray_value / 255.0)
    return image


def create_noisy_image(
    size: Tuple[int, int] = (256, 256),
    noise_level: float = 0.1,
    base_value: float = 0.5
) -> np.ndarray:
    """
    Create a noisy test image with Gaussian noise.
    
    Args:
        size: Image size as (height, width)
        noise_level: Standard deviation of noise (0-1 range)
        base_value: Base pixel value before adding noise (0-1 range)
        
    Returns:
        np.ndarray: Noisy test image as float32 in [0, 1] range
    """
    clean = np.ones((size[0], size[1], 3), dtype=np.float32) * base_value
    noise = np.random.normal(0, noise_level, clean.shape).astype(np.float32)
    noisy = np.clip(clean + noise, 0, 1)
    return noisy


def create_gradient_image(
    size: Tuple[int, int] = (256, 256),
    direction: str = 'horizontal'
) -> np.ndarray:
    """
    Create an image with a gradient pattern.
    
    Args:
        size: Image size as (height, width)
        direction: Gradient direction ('horizontal' or 'vertical')
        
    Returns:
        np.ndarray: Gradient image as float32 in [0, 1] range
    """
    image = np.zeros((size[0], size[1], 3), dtype=np.float32)
    
    if direction == 'horizontal':
        gradient = np.linspace(0, 1, size[1])
        image[:, :, :] = gradient[np.newaxis, :, np.newaxis]
    else:  # vertical
        gradient = np.linspace(0, 1, size[0])
        image[:, :, :] = gradient[:, np.newaxis, np.newaxis]
    
    return image


def save_test_image_file(
    path: Path,
    size: Tuple[int, int] = (256, 256),
    color: Tuple[int, int, int] = (255, 0, 0)
) -> Path:
    """
    Create and save a test image file.
    
    Args:
        path: Path where image should be saved
        size: Image size as (width, height) for PIL
        color: RGB color tuple (0-255)
        
    Returns:
        Path: Path to created image file
    """
    img = Image.new('RGB', size, color)
    img.save(path)
    return path


def save_grayscale_image_file(
    path: Path,
    size: Tuple[int, int] = (256, 256),
    gray_value: int = 128
) -> Path:
    """
    Create and save a grayscale test image file.
    
    Args:
        path: Path where image should be saved
        size: Image size as (width, height) for PIL
        gray_value: Gray value (0-255)
        
    Returns:
        Path: Path to created image file
    """
    img = Image.new('RGB', size, (gray_value, gray_value, gray_value))
    img.save(path)
    return path


# ============================================================================
# Mock Model Creation Functions
# ============================================================================

def create_mock_denoiser(output_shape: Optional[Tuple[int, int, int]] = None):
    """
    Create a mock denoiser that returns a processed image.
    
    Args:
        output_shape: Shape of output image (default: same as input)
        
    Returns:
        Mock: Mock denoiser object with denoise method
    """
    mock_denoiser = Mock()
    
    def mock_denoise(image):
        if output_shape:
            return np.random.rand(*output_shape).astype(np.float32)
        else:
            # Return image with slightly reduced noise
            return np.clip(image + np.random.normal(0, 0.01, image.shape), 0, 1).astype(np.float32)
    
    mock_denoiser.denoise = mock_denoise
    mock_denoiser.strength = 10
    
    return mock_denoiser


def create_mock_super_resolver(scale: int = 4):
    """
    Create a mock super-resolution module.
    
    Args:
        scale: Upscaling factor
        
    Returns:
        Mock: Mock super-resolution object with upscale method
    """
    mock_sr = Mock()
    mock_sr.scale = scale
    
    def mock_upscale(image):
        h, w = image.shape[:2]
        return np.random.rand(h * scale, w * scale, 3).astype(np.float32)
    
    mock_sr.upscale = mock_upscale
    mock_sr.load_model = Mock()
    mock_sr.unload_model = Mock()
    
    return mock_sr


def create_mock_preprocessor():
    """
    Create a mock preprocessor.
    
    Returns:
        Mock: Mock preprocessor object with process method
    """
    mock_preprocessor = Mock()
    
    def mock_process(image_path):
        # Return a simple test image and metadata
        image = np.random.rand(256, 256, 3).astype(np.float32)
        metadata = {
            'original_size': (256, 256),
            'is_grayscale': False,
            'resize_factor': 1.0
        }
        return image, metadata
    
    mock_preprocessor.process = mock_process
    mock_preprocessor.max_size = 2048
    
    return mock_preprocessor


def create_mock_realesrgan_model():
    """
    Create a mock Real-ESRGAN model for testing.
    
    Returns:
        Mock: Mock Real-ESRGAN model with enhance method
    """
    mock_model = MagicMock()
    
    def mock_enhance(image, outscale=4):
        h, w = image.shape[:2]
        output = np.random.randint(0, 256, (h * outscale, w * outscale, 3), dtype=np.uint8)
        return output, None
    
    mock_model.enhance = mock_enhance
    
    return mock_model


# ============================================================================
# Assertion Helpers for Image Comparison
# ============================================================================

def assert_image_shape(image: np.ndarray, expected_shape: Tuple[int, int, int]):
    """
    Assert that image has expected shape.
    
    Args:
        image: Image array to check
        expected_shape: Expected shape as (height, width, channels)
        
    Raises:
        AssertionError: If shape doesn't match
    """
    assert image.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {image.shape}"


def assert_image_dtype(image: np.ndarray, expected_dtype: type):
    """
    Assert that image has expected data type.
    
    Args:
        image: Image array to check
        expected_dtype: Expected numpy dtype
        
    Raises:
        AssertionError: If dtype doesn't match
    """
    assert image.dtype == expected_dtype, \
        f"Expected dtype {expected_dtype}, got {image.dtype}"


def assert_image_range(image: np.ndarray, min_val: float = 0.0, max_val: float = 1.0):
    """
    Assert that image values are within expected range.
    
    Args:
        image: Image array to check
        min_val: Minimum expected value
        max_val: Maximum expected value
        
    Raises:
        AssertionError: If values are outside range
    """
    assert image.min() >= min_val, \
        f"Image minimum {image.min()} is below {min_val}"
    assert image.max() <= max_val, \
        f"Image maximum {image.max()} is above {max_val}"


def assert_images_similar(
    image1: np.ndarray,
    image2: np.ndarray,
    tolerance: float = 0.01
):
    """
    Assert that two images are similar within tolerance.
    
    Args:
        image1: First image array
        image2: Second image array
        tolerance: Maximum allowed difference
        
    Raises:
        AssertionError: If images differ by more than tolerance
    """
    assert image1.shape == image2.shape, \
        f"Image shapes don't match: {image1.shape} vs {image2.shape}"
    
    diff = np.abs(image1 - image2).mean()
    assert diff <= tolerance, \
        f"Images differ by {diff}, which exceeds tolerance {tolerance}"


def assert_image_not_identical(image1: np.ndarray, image2: np.ndarray):
    """
    Assert that two images are not identical (useful for testing processing).
    
    Args:
        image1: First image array
        image2: Second image array
        
    Raises:
        AssertionError: If images are identical
    """
    assert not np.array_equal(image1, image2), \
        "Images are identical, but should be different"


def assert_image_dimensions_scaled(
    original: np.ndarray,
    scaled: np.ndarray,
    scale_factor: int
):
    """
    Assert that scaled image has correct dimensions relative to original.
    
    Args:
        original: Original image array
        scaled: Scaled image array
        scale_factor: Expected scaling factor
        
    Raises:
        AssertionError: If dimensions don't match expected scaling
    """
    expected_h = original.shape[0] * scale_factor
    expected_w = original.shape[1] * scale_factor
    
    assert scaled.shape[0] == expected_h, \
        f"Expected height {expected_h}, got {scaled.shape[0]}"
    assert scaled.shape[1] == expected_w, \
        f"Expected width {expected_w}, got {scaled.shape[1]}"


# ============================================================================
# Metadata Helpers
# ============================================================================

def create_sample_metadata(
    original_size: Tuple[int, int] = (512, 512),
    is_grayscale: bool = False,
    resize_factor: float = 1.0
) -> dict:
    """
    Create sample metadata dictionary.
    
    Args:
        original_size: Original image size as (height, width)
        is_grayscale: Whether image is grayscale
        resize_factor: Resize factor applied
        
    Returns:
        dict: Sample metadata dictionary
    """
    return {
        'original_size': original_size,
        'is_grayscale': is_grayscale,
        'resize_factor': resize_factor
    }


def assert_metadata_structure(metadata: dict):
    """
    Assert that metadata has required fields.
    
    Args:
        metadata: Metadata dictionary to check
        
    Raises:
        AssertionError: If required fields are missing
    """
    required_fields = ['original_size', 'is_grayscale', 'resize_factor']
    
    for field in required_fields:
        assert field in metadata, f"Metadata missing required field: {field}"
    
    assert isinstance(metadata['original_size'], tuple), \
        "original_size must be a tuple"
    assert len(metadata['original_size']) == 2, \
        "original_size must have 2 elements"
    assert isinstance(metadata['is_grayscale'], bool), \
        "is_grayscale must be a boolean"
    assert isinstance(metadata['resize_factor'], (int, float)), \
        "resize_factor must be a number"
