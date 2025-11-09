"""
Unit tests for colorization module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from src.models.colorization import ColorizationModule, ColorizationError
from src.utils.exceptions import ModelLoadError, ProcessingError
from tests.utils import (
    create_grayscale_test_image,
    create_test_image,
    assert_image_shape,
    assert_image_dtype,
    assert_image_range
)


# Skip all tests if torch is not available
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


class TestColorizationModule:
    """Test suite for ColorizationModule class"""
    
    @pytest.fixture
    def colorizer(self, tmp_path):
        """Create colorization module with mock weights"""
        weights_path = tmp_path / "ddcolor.pth"
        
        # Create a mock weights file
        mock_model = nn.Conv2d(1, 2, 3, padding=1)
        torch.save({'state_dict': mock_model.state_dict()}, weights_path)
        
        return ColorizationModule(
            weights_path=str(weights_path),
            device='cpu',
            use_fp16=False
        )
    
    @pytest.fixture
    def grayscale_image(self):
        """Create grayscale test image"""
        return create_grayscale_test_image(size=(256, 256), gray_value=128, dtype=np.float32)
    
    @pytest.fixture
    def color_image(self):
        """Create color test image"""
        return create_test_image(size=(256, 256), color=(255, 0, 0), dtype=np.float32)
    
    def test_initialization_default(self, tmp_path):
        """Test ColorizationModule initialization with defaults"""
        weights_path = tmp_path / "ddcolor.pth"
        torch.save({}, weights_path)
        
        colorizer = ColorizationModule(weights_path=str(weights_path))
        
        assert colorizer.weights_path == str(weights_path)
        assert colorizer.device in ['cuda', 'cpu']
        assert colorizer.model is None
    
    def test_initialization_custom_device(self, tmp_path):
        """Test ColorizationModule initialization with custom device"""
        weights_path = tmp_path / "ddcolor.pth"
        torch.save({}, weights_path)
        
        colorizer = ColorizationModule(
            weights_path=str(weights_path),
            device='cpu',
            use_fp16=False
        )
        
        assert colorizer.device == 'cpu'
        assert colorizer.use_fp16 is False
    
    def test_is_grayscale_detection_true(self, colorizer, grayscale_image):
        """Test grayscale detection for grayscale image"""
        result = colorizer.is_grayscale(grayscale_image)
        
        assert result is True
    
    def test_is_grayscale_detection_false(self, colorizer, color_image):
        """Test grayscale detection for color image"""
        result = colorizer.is_grayscale(color_image)
        
        assert result is False
    
    def test_is_grayscale_with_tolerance(self, colorizer):
        """Test grayscale detection with tolerance"""
        # Create image with slight color variation within tolerance
        image = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        image[:, :, 0] += 0.01  # Add small variation to red channel
        
        result = colorizer.is_grayscale(image, tolerance=5)
        
        # Should still be detected as grayscale within tolerance
        assert result is True
    
    def test_is_grayscale_outside_tolerance(self, colorizer):
        """Test grayscale detection outside tolerance"""
        # Create image with significant color variation
        image = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        image[:, :, 0] = 1.0  # Make red channel significantly different
        
        result = colorizer.is_grayscale(image, tolerance=5)
        
        assert result is False
    
    def test_load_model_success(self, colorizer):
        """Test successful model loading"""
        colorizer.load_model()
        
        assert colorizer.model is not None
        assert hasattr(colorizer.model, 'eval')
    
    def test_load_model_already_loaded(self, colorizer):
        """Test loading model when already loaded"""
        colorizer.load_model()
        model_ref = colorizer.model
        
        # Load again
        colorizer.load_model()
        
        # Should be same model instance
        assert colorizer.model is model_ref
    
    def test_load_model_missing_weights(self, tmp_path):
        """Test model loading with missing weights file"""
        colorizer = ColorizationModule(
            weights_path=str(tmp_path / "nonexistent.pth"),
            device='cpu'
        )
        
        with pytest.raises(ModelLoadError, match="Weights file not found"):
            colorizer.load_model()
    
    def test_load_model_no_weights_path(self):
        """Test model loading without weights path"""
        colorizer = ColorizationModule(weights_path=None, device='cpu')
        
        with pytest.raises(ModelLoadError, match="Weights path is required"):
            colorizer.load_model()
    
    def test_colorize_grayscale_image(self, colorizer, grayscale_image):
        """Test colorization of grayscale image"""
        result = colorizer.colorize(grayscale_image)
        
        # Check output properties
        assert_image_shape(result, grayscale_image.shape)
        assert_image_dtype(result, np.float32)
        assert_image_range(result, 0.0, 1.0)
    
    def test_colorize_color_image_bypass(self, colorizer, color_image):
        """Test that color images bypass colorization"""
        result = colorizer.colorize(color_image)
        
        # Should return original image unchanged
        assert np.array_equal(result, color_image)
    
    def test_colorize_invalid_shape_2d(self, colorizer):
        """Test colorization raises error for 2D image"""
        invalid_image = np.ones((256, 256), dtype=np.float32)
        
        with pytest.raises(ColorizationError, match="must have 3 channels"):
            colorizer.colorize(invalid_image)
    
    def test_colorize_invalid_shape_4_channels(self, colorizer):
        """Test colorization raises error for 4-channel image"""
        invalid_image = np.ones((256, 256, 4), dtype=np.float32)
        
        with pytest.raises(ColorizationError, match="must have 3 channels"):
            colorizer.colorize(invalid_image)
    
    def test_colorize_empty_image(self, colorizer):
        """Test colorization raises error for empty image"""
        empty_image = np.array([]).reshape(0, 0, 3).astype(np.float32)
        
        with pytest.raises(ColorizationError, match="Cannot colorize empty image"):
            colorizer.colorize(empty_image)
    
    def test_colorize_small_image(self, colorizer):
        """Test colorization works on small images"""
        small_image = create_grayscale_test_image(size=(64, 64), dtype=np.float32)
        
        result = colorizer.colorize(small_image)
        
        assert_image_shape(result, (64, 64, 3))
        assert_image_dtype(result, np.float32)
    
    def test_colorize_large_image(self, colorizer):
        """Test colorization works on larger images"""
        large_image = create_grayscale_test_image(size=(512, 512), dtype=np.float32)
        
        result = colorizer.colorize(large_image)
        
        assert_image_shape(result, (512, 512, 3))
        assert_image_dtype(result, np.float32)
    
    def test_colorize_loads_model_lazily(self, colorizer, grayscale_image):
        """Test that colorize loads model if not already loaded"""
        assert colorizer.model is None
        
        result = colorizer.colorize(grayscale_image)
        
        assert colorizer.model is not None
    
    def test_unload_model(self, colorizer):
        """Test model unloading"""
        colorizer.load_model()
        assert colorizer.model is not None
        
        colorizer.unload_model()
        
        assert colorizer.model is None
    
    def test_rgb_to_lab_conversion(self, colorizer, grayscale_image):
        """Test RGB to Lab color space conversion"""
        lab_image = colorizer._rgb_to_lab(grayscale_image)
        
        # Check shape
        assert lab_image.shape == grayscale_image.shape
        
        # Check L channel is in reasonable range [0, 100]
        l_channel = lab_image[:, :, 0]
        assert l_channel.min() >= 0
        assert l_channel.max() <= 100
    
    def test_lab_to_rgb_conversion(self, colorizer):
        """Test Lab to RGB color space conversion"""
        # Create Lab image
        lab_image = np.zeros((256, 256, 3), dtype=np.float32)
        lab_image[:, :, 0] = 50  # L channel
        
        rgb_image = colorizer._lab_to_rgb(lab_image)
        
        # Check shape
        assert rgb_image.shape == lab_image.shape
        
        # Check RGB is in [0, 1] range
        assert_image_range(rgb_image, 0.0, 1.0)
    
    def test_colorize_with_fp16(self, tmp_path, grayscale_image):
        """Test colorization with FP16 inference"""
        weights_path = tmp_path / "ddcolor.pth"
        mock_model = nn.Conv2d(1, 2, 3, padding=1)
        torch.save({'state_dict': mock_model.state_dict()}, weights_path)
        
        # Create colorizer with FP16 (will use CPU, so FP16 will be disabled)
        colorizer = ColorizationModule(
            weights_path=str(weights_path),
            device='cpu',
            use_fp16=True  # Will be set to False since device is CPU
        )
        
        result = colorizer.colorize(grayscale_image)
        
        assert_image_shape(result, grayscale_image.shape)
        assert_image_dtype(result, np.float32)
    
    def test_colorize_preserves_resolution(self, colorizer):
        """Test that colorization preserves image resolution"""
        sizes = [(128, 128), (256, 256), (512, 384)]
        
        for size in sizes:
            image = create_grayscale_test_image(size=size, dtype=np.float32)
            result = colorizer.colorize(image)
            
            assert result.shape == image.shape


class TestColorizationError:
    """Test suite for ColorizationError exception"""
    
    def test_colorization_error_is_processing_error(self):
        """Test that ColorizationError inherits from ProcessingError"""
        error = ColorizationError("Test error")
        
        assert isinstance(error, ProcessingError)
        assert isinstance(error, Exception)
    
    def test_colorization_error_message(self):
        """Test ColorizationError message"""
        message = "Colorization failed"
        error = ColorizationError(message)
        
        assert str(error) == message


class TestColorizationIntegration:
    """Integration tests for colorization workflow"""
    
    def test_full_colorization_workflow(self, tmp_path):
        """Test complete colorization workflow"""
        # Setup
        weights_path = tmp_path / "ddcolor.pth"
        mock_model = nn.Conv2d(1, 2, 3, padding=1)
        torch.save({'state_dict': mock_model.state_dict()}, weights_path)
        
        colorizer = ColorizationModule(
            weights_path=str(weights_path),
            device='cpu',
            use_fp16=False
        )
        
        # Create grayscale image
        grayscale = create_grayscale_test_image(size=(256, 256), dtype=np.float32)
        
        # Colorize
        result = colorizer.colorize(grayscale)
        
        # Verify
        assert result.shape == grayscale.shape
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        
        # Cleanup
        colorizer.unload_model()
        assert colorizer.model is None
    
    def test_batch_colorization_with_caching(self, tmp_path):
        """Test colorizing multiple images with model caching"""
        # Setup
        weights_path = tmp_path / "ddcolor.pth"
        mock_model = nn.Conv2d(1, 2, 3, padding=1)
        torch.save({'state_dict': mock_model.state_dict()}, weights_path)
        
        colorizer = ColorizationModule(
            weights_path=str(weights_path),
            device='cpu',
            use_fp16=False
        )
        
        # Create multiple grayscale images
        images = [
            create_grayscale_test_image(size=(256, 256), gray_value=50, dtype=np.float32),
            create_grayscale_test_image(size=(256, 256), gray_value=100, dtype=np.float32),
            create_grayscale_test_image(size=(256, 256), gray_value=150, dtype=np.float32)
        ]
        
        # Colorize all (model should stay loaded)
        results = []
        for image in images:
            result = colorizer.colorize(image)
            results.append(result)
        
        # Verify all results
        for result in results:
            assert result.shape == (256, 256, 3)
            assert result.dtype == np.float32
            assert_image_range(result, 0.0, 1.0)
        
        # Model should still be loaded
        assert colorizer.model is not None
        
        # Cleanup
        colorizer.unload_model()
    
    def test_mixed_grayscale_and_color_images(self, tmp_path):
        """Test processing mix of grayscale and color images"""
        # Setup
        weights_path = tmp_path / "ddcolor.pth"
        mock_model = nn.Conv2d(1, 2, 3, padding=1)
        torch.save({'state_dict': mock_model.state_dict()}, weights_path)
        
        colorizer = ColorizationModule(
            weights_path=str(weights_path),
            device='cpu',
            use_fp16=False
        )
        
        # Create mixed images
        grayscale = create_grayscale_test_image(size=(256, 256), dtype=np.float32)
        color = create_test_image(size=(256, 256), color=(255, 0, 0), dtype=np.float32)
        
        # Process both
        result_gray = colorizer.colorize(grayscale)
        result_color = colorizer.colorize(color)
        
        # Grayscale should be processed
        assert result_gray.shape == grayscale.shape
        
        # Color should be unchanged
        assert np.array_equal(result_color, color)
