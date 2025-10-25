"""
Unit tests for denoising module.
"""

import pytest
import numpy as np

from src.models.denoiser import (
    DenoisingModule,
    OpenCVDenoiser,
    NAFNetDenoiser,
    create_denoiser
)
from src.utils.exceptions import ConfigurationError, ProcessingError, ModelLoadError


class TestDenoisingModule:
    """Test suite for base DenoisingModule class"""
    
    def test_base_class_cannot_be_instantiated_directly(self):
        """Test that abstract base class enforces denoise() implementation"""
        # Create a concrete class without implementing denoise()
        class IncompletDenoiser(DenoisingModule):
            pass
        
        # Should raise TypeError when trying to instantiate
        with pytest.raises(TypeError):
            IncompletDenoiser()
    
    def test_strength_validation_too_low(self):
        """Test that strength below 1 raises ConfigurationError"""
        with pytest.raises(ConfigurationError, match="Denoising strength must be between 1 and 100"):
            OpenCVDenoiser(strength=0)
    
    def test_strength_validation_too_high(self):
        """Test that strength above 100 raises ConfigurationError"""
        with pytest.raises(ConfigurationError, match="Denoising strength must be between 1 and 100"):
            OpenCVDenoiser(strength=101)


class TestOpenCVDenoiser:
    """Test suite for OpenCVDenoiser class"""
    
    @pytest.fixture
    def denoiser(self):
        """Create OpenCV denoiser with default strength"""
        return OpenCVDenoiser()
    
    @pytest.fixture
    def test_image(self):
        """Create test image with noise"""
        # Create a simple test image (normalized to [0, 1])
        image = np.ones((512, 512, 3), dtype=np.float32) * 0.5
        # Add some noise
        noise = np.random.normal(0, 0.05, image.shape).astype(np.float32)
        noisy_image = np.clip(image + noise, 0, 1)
        return noisy_image
    
    def test_initialization_default_strength(self):
        """Test OpenCVDenoiser initialization with default strength"""
        denoiser = OpenCVDenoiser()
        
        assert denoiser.strength == 10
        assert denoiser.template_window_size == 7
        assert denoiser.search_window_size == 21
    
    def test_initialization_custom_strength(self):
        """Test OpenCVDenoiser initialization with custom strength"""
        denoiser = OpenCVDenoiser(strength=20)
        
        assert denoiser.strength == 20
    
    def test_denoise_returns_correct_shape(self, denoiser, test_image):
        """Test denoising produces output of correct dimensions"""
        result = denoiser.denoise(test_image)
        
        assert result.shape == test_image.shape
        assert result.shape == (512, 512, 3)
    
    def test_denoise_returns_correct_dtype(self, denoiser, test_image):
        """Test denoising returns float32 in [0, 1] range"""
        result = denoiser.denoise(test_image)
        
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_denoise_with_different_strengths(self, test_image):
        """Test denoising with different strength values"""
        weak_denoiser = OpenCVDenoiser(strength=5)
        strong_denoiser = OpenCVDenoiser(strength=30)
        
        weak_result = weak_denoiser.denoise(test_image)
        strong_result = strong_denoiser.denoise(test_image)
        
        # Both should produce valid outputs
        assert weak_result.shape == test_image.shape
        assert strong_result.shape == test_image.shape
        assert weak_result.dtype == np.float32
        assert strong_result.dtype == np.float32
    
    def test_denoise_reduces_noise(self, denoiser):
        """Test that denoising actually reduces noise variance"""
        # Create clean image
        clean = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        
        # Add significant noise
        noise = np.random.normal(0, 0.1, clean.shape).astype(np.float32)
        noisy = np.clip(clean + noise, 0, 1)
        
        # Denoise
        denoised = denoiser.denoise(noisy)
        
        # Calculate variance (measure of noise)
        noisy_variance = np.var(noisy)
        denoised_variance = np.var(denoised)
        
        # Denoised image should have lower variance
        assert denoised_variance < noisy_variance
    
    def test_denoise_invalid_shape_2d(self, denoiser):
        """Test denoising raises error for 2D image"""
        invalid_image = np.ones((512, 512), dtype=np.float32)
        
        with pytest.raises(ProcessingError, match="must have 3 channels"):
            denoiser.denoise(invalid_image)
    
    def test_denoise_invalid_shape_4_channels(self, denoiser):
        """Test denoising raises error for 4-channel image"""
        invalid_image = np.ones((512, 512, 4), dtype=np.float32)
        
        with pytest.raises(ProcessingError, match="must have 3 channels"):
            denoiser.denoise(invalid_image)
    
    def test_denoise_small_image(self, denoiser):
        """Test denoising works on small images"""
        small_image = np.random.rand(64, 64, 3).astype(np.float32)
        
        result = denoiser.denoise(small_image)
        
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.float32
    
    def test_denoise_large_image(self, denoiser):
        """Test denoising works on larger images"""
        large_image = np.random.rand(1024, 1024, 3).astype(np.float32)
        
        result = denoiser.denoise(large_image)
        
        assert result.shape == (1024, 1024, 3)
        assert result.dtype == np.float32


class TestNAFNetDenoiser:
    """Test suite for NAFNetDenoiser class (placeholder)"""
    
    def test_initialization(self):
        """Test NAFNetDenoiser can be initialized"""
        denoiser = NAFNetDenoiser(strength=10, weights_path="/path/to/weights.pth")
        
        assert denoiser.strength == 10
        assert denoiser.weights_path == "/path/to/weights.pth"
        assert denoiser.device == 'cuda'
        assert denoiser.model is None
    
    def test_initialization_custom_device(self):
        """Test NAFNetDenoiser initialization with custom device"""
        denoiser = NAFNetDenoiser(strength=10, device='cpu')
        
        assert denoiser.device == 'cpu'
    
    def test_load_model_not_implemented(self):
        """Test load_model raises ModelLoadError"""
        denoiser = NAFNetDenoiser()
        
        with pytest.raises(ModelLoadError, match="will be implemented in future"):
            denoiser.load_model()
    
    def test_denoise_not_implemented(self):
        """Test denoise raises ModelLoadError"""
        denoiser = NAFNetDenoiser()
        test_image = np.random.rand(512, 512, 3).astype(np.float32)
        
        with pytest.raises(ModelLoadError, match="will be implemented in future"):
            denoiser.denoise(test_image)


class TestDenoiserFactory:
    """Test suite for create_denoiser factory function"""
    
    def test_create_opencv_denoiser(self):
        """Test factory creates OpenCVDenoiser"""
        denoiser = create_denoiser('opencv', strength=15)
        
        assert isinstance(denoiser, OpenCVDenoiser)
        assert denoiser.strength == 15
    
    def test_create_opencv_denoiser_default_strength(self):
        """Test factory creates OpenCVDenoiser with default strength"""
        denoiser = create_denoiser('opencv')
        
        assert isinstance(denoiser, OpenCVDenoiser)
        assert denoiser.strength == 10
    
    def test_create_nafnet_denoiser(self):
        """Test factory creates NAFNetDenoiser"""
        denoiser = create_denoiser(
            'nafnet',
            strength=20,
            weights_path='/path/to/weights.pth',
            device='cpu'
        )
        
        assert isinstance(denoiser, NAFNetDenoiser)
        assert denoiser.strength == 20
        assert denoiser.weights_path == '/path/to/weights.pth'
        assert denoiser.device == 'cpu'
    
    def test_create_unsupported_type(self):
        """Test factory raises ConfigurationError for unsupported type"""
        with pytest.raises(ConfigurationError, match="Unsupported denoiser type"):
            create_denoiser('unknown_type')
    
    def test_factory_with_config_integration(self):
        """Test factory can be used with config-like parameters"""
        # Simulate config usage
        config_type = 'opencv'
        config_strength = 25
        
        denoiser = create_denoiser(config_type, strength=config_strength)
        
        assert isinstance(denoiser, OpenCVDenoiser)
        assert denoiser.strength == 25
