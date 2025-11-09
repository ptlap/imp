"""
Unit tests for post-processing module.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import cv2

from src.models.post_processing import PostProcessingModule
from src.utils.exceptions import ProcessingError


class TestPostProcessingModule:
    """Test suite for PostProcessingModule"""
    
    @pytest.fixture
    def module(self):
        """Create default post-processing module"""
        return PostProcessingModule()
    
    @pytest.fixture
    def test_image(self):
        """Create test image (256x256 RGB) with values in [0, 1]"""
        # Create gradient image for testing
        img = np.zeros((256, 256, 3), dtype=np.float32)
        for i in range(3):
            img[:, :, i] = np.linspace(0, 1, 256).reshape(1, -1)
        return img
    
    # Test initialization
    def test_initialization_default(self):
        """Test module can be initialized with default parameters"""
        module = PostProcessingModule()
        
        assert module.white_balance == True
        assert module.auto_levels == True
        assert module.sharpen_strength == 1.0
        assert module.enhance_contrast == True
        assert module.final_denoise == False
    
    def test_initialization_custom(self):
        """Test module can be initialized with custom parameters"""
        module = PostProcessingModule(
            white_balance=False,
            auto_levels=False,
            sharpen_strength=2.5,
            enhance_contrast=False,
            final_denoise=True
        )
        
        assert module.white_balance == False
        assert module.auto_levels == False
        assert module.sharpen_strength == 2.5
        assert module.enhance_contrast == False
        assert module.final_denoise == True
    
    def test_initialization_invalid_sharpen_strength(self):
        """Test initialization fails with invalid sharpen_strength"""
        with pytest.raises(ValueError, match="sharpen_strength must be between"):
            PostProcessingModule(sharpen_strength=-1.0)
        
        with pytest.raises(ValueError, match="sharpen_strength must be between"):
            PostProcessingModule(sharpen_strength=6.0)
    
    # Test process() method
    def test_process_valid_image(self, module, test_image):
        """Test processing valid image"""
        result = module.process(test_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == test_image.shape
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_process_invalid_input_not_array(self, module):
        """Test processing fails with non-array input"""
        with pytest.raises(ProcessingError, match="must be a numpy array"):
            module.process("not an array")
    
    def test_process_invalid_dimensions(self, module):
        """Test processing fails with wrong dimensions"""
        # 2D image
        with pytest.raises(ProcessingError, match="must have 3 dimensions"):
            module.process(np.zeros((256, 256), dtype=np.float32))
        
        # 4D image
        with pytest.raises(ProcessingError, match="must have 3 dimensions"):
            module.process(np.zeros((1, 256, 256, 3), dtype=np.float32))
    
    def test_process_invalid_channels(self, module):
        """Test processing fails with wrong number of channels"""
        with pytest.raises(ProcessingError, match="must have 3 channels"):
            module.process(np.zeros((256, 256, 1), dtype=np.float32))
        
        with pytest.raises(ProcessingError, match="must have 3 channels"):
            module.process(np.zeros((256, 256, 4), dtype=np.float32))
    
    def test_process_empty_image(self, module):
        """Test processing fails with empty image"""
        with pytest.raises(ProcessingError, match="Cannot process empty image"):
            module.process(np.zeros((0, 0, 3), dtype=np.float32))
    
    def test_process_invalid_dtype(self, module):
        """Test processing fails with wrong dtype"""
        with pytest.raises(ProcessingError, match="must be float type"):
            module.process(np.zeros((256, 256, 3), dtype=np.uint8))
    
    def test_process_different_sizes(self, module):
        """Test processing works with different image sizes"""
        sizes = [(128, 128), (256, 256), (512, 512), (1024, 1024)]
        
        for h, w in sizes:
            img = np.random.rand(h, w, 3).astype(np.float32)
            result = module.process(img)
            
            assert result.shape == (h, w, 3)
            assert result.min() >= 0.0
            assert result.max() <= 1.0
    
    def test_process_skip_all_operations(self, test_image):
        """Test processing with all operations disabled"""
        module = PostProcessingModule(
            white_balance=False,
            auto_levels=False,
            sharpen_strength=0.0,
            enhance_contrast=False,
            final_denoise=False
        )
        
        result = module.process(test_image)
        
        # Result should be very close to original (only copy operation)
        assert np.allclose(result, test_image, atol=1e-6)
    
    # Test white balance
    def test_white_balance(self, module, test_image):
        """Test white balance correction"""
        result = module._white_balance(test_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == test_image.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_white_balance_edge_cases(self, module):
        """Test white balance with edge cases"""
        # All black image
        black = np.zeros((256, 256, 3), dtype=np.float32)
        result = module._white_balance(black)
        assert result.shape == black.shape
        
        # All white image
        white = np.ones((256, 256, 3), dtype=np.float32)
        result = module._white_balance(white)
        assert result.shape == white.shape
        
        # Single color image (red)
        red = np.zeros((256, 256, 3), dtype=np.float32)
        red[:, :, 0] = 1.0
        result = module._white_balance(red)
        assert result.shape == red.shape
    
    # Test auto levels
    def test_auto_levels(self, module, test_image):
        """Test auto levels (histogram stretching)"""
        result = module._auto_levels(test_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == test_image.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_auto_levels_low_contrast(self, module):
        """Test auto levels with low contrast image"""
        # Image with values only in [0.4, 0.6] range
        img = np.full((256, 256, 3), 0.5, dtype=np.float32)
        img[:128, :, :] = 0.4
        img[128:, :, :] = 0.6
        
        result = module._auto_levels(img)
        
        # Should stretch to use more of the [0, 1] range
        assert result.shape == img.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_auto_levels_edge_cases(self, module):
        """Test auto levels with edge cases"""
        # All same value (no contrast)
        flat = np.full((256, 256, 3), 0.5, dtype=np.float32)
        result = module._auto_levels(flat)
        assert result.shape == flat.shape
        
        # Already full range
        full_range = np.random.rand(256, 256, 3).astype(np.float32)
        result = module._auto_levels(full_range)
        assert result.shape == full_range.shape
    
    # Test unsharp mask
    def test_unsharp_mask(self, module, test_image):
        """Test unsharp masking"""
        result = module._unsharp_mask(test_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == test_image.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_unsharp_mask_different_strengths(self, test_image):
        """Test unsharp mask with different strength values"""
        strengths = [0.5, 1.0, 2.0, 3.0, 5.0]
        
        for strength in strengths:
            module = PostProcessingModule(sharpen_strength=strength)
            result = module._unsharp_mask(test_image)
            
            assert result.shape == test_image.shape
            assert result.min() >= 0.0
            assert result.max() <= 1.0
    
    def test_unsharp_mask_zero_strength(self, test_image):
        """Test unsharp mask with zero strength (no sharpening)"""
        module = PostProcessingModule(sharpen_strength=0.0)
        result = module._unsharp_mask(test_image)
        
        # With strength=0, result should be close to original
        # (only Gaussian blur and subtraction, but multiplied by 0)
        assert result.shape == test_image.shape
    
    # Test CLAHE contrast enhancement
    def test_enhance_contrast(self, module, test_image):
        """Test CLAHE contrast enhancement"""
        result = module._enhance_contrast(test_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == test_image.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_enhance_contrast_low_contrast_image(self, module):
        """Test CLAHE with low contrast image"""
        # Create low contrast image
        img = np.full((256, 256, 3), 0.5, dtype=np.float32)
        img[:128, :, :] = 0.45
        img[128:, :, :] = 0.55
        
        result = module._enhance_contrast(img)
        
        # CLAHE should increase local contrast
        assert result.shape == img.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    # Test final denoise
    def test_denoise_final(self, module, test_image):
        """Test final denoising"""
        result = module._denoise_final(test_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == test_image.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_denoise_final_noisy_image(self, module):
        """Test final denoising with noisy image"""
        # Create noisy image
        img = np.random.rand(256, 256, 3).astype(np.float32) * 0.5 + 0.25
        noise = np.random.normal(0, 0.05, img.shape).astype(np.float32)
        noisy = np.clip(img + noise, 0, 1)
        
        result = module._denoise_final(noisy)
        
        # Denoised image should be smoother (lower variance)
        assert result.shape == noisy.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    # Test full pipeline with different configurations
    def test_full_pipeline_all_enabled(self, test_image):
        """Test full pipeline with all operations enabled"""
        module = PostProcessingModule(
            white_balance=True,
            auto_levels=True,
            sharpen_strength=1.5,
            enhance_contrast=True,
            final_denoise=True
        )
        
        result = module.process(test_image)
        
        assert result.shape == test_image.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_full_pipeline_selective_operations(self, test_image):
        """Test pipeline with selective operations"""
        # Only sharpen and contrast
        module = PostProcessingModule(
            white_balance=False,
            auto_levels=False,
            sharpen_strength=2.0,
            enhance_contrast=True,
            final_denoise=False
        )
        
        result = module.process(test_image)
        
        assert result.shape == test_image.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    # Test error handling
    def test_white_balance_error_handling(self, module):
        """Test white balance handles errors gracefully"""
        # This should not raise, but return original image
        invalid = np.array([[[np.nan, 0.5, 0.5]]], dtype=np.float32)
        result = module._white_balance(invalid)
        assert result.shape == invalid.shape
    
    def test_process_does_not_modify_input(self, module, test_image):
        """Test that process() does not modify input image"""
        original = test_image.copy()
        result = module.process(test_image)
        
        # Input should remain unchanged
        assert np.array_equal(test_image, original)
        # Result should be different (processed)
        assert not np.array_equal(result, original)
