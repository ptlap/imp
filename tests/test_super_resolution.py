"""
Unit tests for super-resolution module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock torch and related libraries before importing super_resolution module
sys.modules['torch'] = MagicMock()
sys.modules['torch.cuda'] = MagicMock()
sys.modules['basicsr'] = MagicMock()
sys.modules['basicsr.archs'] = MagicMock()
sys.modules['basicsr.archs.rrdbnet_arch'] = MagicMock()

from src.utils.exceptions import ConfigurationError, ProcessingError, ModelLoadError
sys.modules['realesrgan'] = MagicMock()

from src.models.super_resolution import SuperResolutionModule


class TestSuperResolutionModule:
    """Test suite for SuperResolutionModule class"""
    
    def test_initialization_default_parameters(self):
        """Test SuperResolutionModule initialization with default parameters"""
        module = SuperResolutionModule()
        
        assert module.scale == 4
        assert module.tile_size == 512
        assert module.tile_overlap == 64
        assert module.use_fp16 == True or module.use_fp16 == False  # Depends on CUDA availability
        assert module.model is None
        assert module.upsampler is None
    
    def test_initialization_custom_parameters(self):
        """Test SuperResolutionModule initialization with custom parameters"""
        module = SuperResolutionModule(
            scale=2,
            weights_path='/path/to/weights.pth',
            tile_size=256,
            tile_overlap=32,
            device='cpu',
            use_fp16=False
        )
        
        assert module.scale == 2
        assert module.weights_path == '/path/to/weights.pth'
        assert module.tile_size == 256
        assert module.tile_overlap == 32
        assert module.device == 'cpu'
        assert module.use_fp16 == False
    
    def test_initialization_invalid_scale(self):
        """Test initialization raises ConfigurationError for invalid scale"""
        with pytest.raises(ConfigurationError, match="Super-resolution scale must be 2 or 4"):
            SuperResolutionModule(scale=3)
        
        with pytest.raises(ConfigurationError, match="Super-resolution scale must be 2 or 4"):
            SuperResolutionModule(scale=8)
    
    def test_initialization_invalid_tile_overlap(self):
        """Test initialization raises ConfigurationError when tile_overlap >= tile_size"""
        with pytest.raises(ConfigurationError, match="tile_overlap.*must be less than tile_size"):
            SuperResolutionModule(tile_size=512, tile_overlap=512)
        
        with pytest.raises(ConfigurationError, match="tile_overlap.*must be less than tile_size"):
            SuperResolutionModule(tile_size=256, tile_overlap=300)
    
    def test_should_tile_small_image(self):
        """Test _should_tile returns False for small images"""
        module = SuperResolutionModule(tile_size=512)
        
        small_image = np.random.rand(256, 256, 3).astype(np.float32)
        assert module._should_tile(small_image) == False
    
    def test_should_tile_large_image(self):
        """Test _should_tile returns True for large images"""
        module = SuperResolutionModule(tile_size=512)
        
        large_image = np.random.rand(1024, 1024, 3).astype(np.float32)
        assert module._should_tile(large_image) == True
    
    def test_should_tile_exact_size(self):
        """Test _should_tile returns False when image equals tile_size"""
        module = SuperResolutionModule(tile_size=512)
        
        exact_image = np.random.rand(512, 512, 3).astype(np.float32)
        assert module._should_tile(exact_image) == False
    
    def test_should_tile_one_dimension_large(self):
        """Test _should_tile returns True when one dimension exceeds tile_size"""
        module = SuperResolutionModule(tile_size=512)
        
        # Width exceeds tile_size
        wide_image = np.random.rand(256, 1024, 3).astype(np.float32)
        assert module._should_tile(wide_image) == True
        
        # Height exceeds tile_size
        tall_image = np.random.rand(1024, 256, 3).astype(np.float32)
        assert module._should_tile(tall_image) == True
    
    def test_load_model_creates_upsampler(self):
        """Test load_model creates model and upsampler"""
        module = SuperResolutionModule(scale=4, weights_path='/fake/path.pth')
        
        # Mock the imports within load_model
        with patch('src.models.super_resolution.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            
            with patch('builtins.__import__') as mock_import:
                # Create mock classes
                mock_rrdbnet_class = Mock()
                mock_realesrganer_class = Mock()
                mock_model = Mock()
                mock_upsampler = Mock()
                
                mock_rrdbnet_class.return_value = mock_model
                mock_realesrganer_class.return_value = mock_upsampler
                
                # Setup import mocking
                def import_side_effect(name, *args, **kwargs):
                    if 'basicsr.archs.rrdbnet_arch' in name:
                        mock_module = Mock()
                        mock_module.RRDBNet = mock_rrdbnet_class
                        return mock_module
                    elif 'realesrgan' in name:
                        mock_module = Mock()
                        mock_module.RealESRGANer = mock_realesrganer_class
                        return mock_module
                    return Mock()
                
                mock_import.side_effect = import_side_effect
                
                module.load_model()
        
        # Verify model and upsampler were set
        assert module.model is not None
        assert module.upsampler is not None
    
    def test_load_model_only_once(self):
        """Test load_model doesn't reload if model already loaded"""
        module = SuperResolutionModule()
        
        # Set model to non-None to simulate already loaded
        module.model = Mock()
        original_model = module.model
        
        module.load_model()
        
        # Model should remain the same
        assert module.model == original_model
    
    def test_load_model_missing_libraries(self):
        """Test load_model raises ModelLoadError when libraries missing"""
        module = SuperResolutionModule()
        
        # Mock import to fail
        with patch('builtins.__import__', side_effect=ImportError("No module named 'basicsr'")):
            with pytest.raises(ModelLoadError, match="Required libraries not installed"):
                module.load_model()
    
    def test_load_model_missing_weights(self):
        """Test load_model raises ModelLoadError when weights missing"""
        module = SuperResolutionModule(weights_path='/fake/path.pth')
        
        # Mock Path.exists to return False
        with patch('src.models.super_resolution.Path') as mock_path:
            mock_path.return_value.exists.return_value = False
            
            with pytest.raises(ModelLoadError, match="Weights file not found"):
                module.load_model()
    
    def test_upscale_correct_output_dimensions(self):
        """Test upscale produces correct output dimensions"""
        module = SuperResolutionModule(scale=4)
        
        # Create test image
        test_image = np.random.rand(128, 128, 3).astype(np.float32)
        
        # Mock the upsampler
        mock_upsampler = Mock()
        expected_output = np.random.rand(512, 512, 3).astype(np.uint8)
        mock_upsampler.enhance.return_value = (expected_output, None)
        module.upsampler = mock_upsampler
        module.model = Mock()  # Set model to avoid loading
        
        # Mock cv2 color conversion
        with patch('src.models.super_resolution.cv2') as mock_cv2:
            mock_cv2.cvtColor.side_effect = lambda img, code: img
            
            result = module.upscale(test_image)
        
        # Verify output shape is scaled correctly
        assert result.shape == (512, 512, 3)
        assert result.dtype == np.float32
    
    def test_upscale_2x_scale(self):
        """Test upscale with 2x scale factor"""
        module = SuperResolutionModule(scale=2)
        
        test_image = np.random.rand(256, 256, 3).astype(np.float32)
        
        # Mock the upsampler
        mock_upsampler = Mock()
        expected_output = np.random.rand(512, 512, 3).astype(np.uint8)
        mock_upsampler.enhance.return_value = (expected_output, None)
        module.upsampler = mock_upsampler
        module.model = Mock()
        
        # Mock cv2 color conversion
        with patch('src.models.super_resolution.cv2') as mock_cv2:
            mock_cv2.cvtColor.side_effect = lambda img, code: img
            
            result = module.upscale(test_image)
        
        assert result.shape == (512, 512, 3)
    
    def test_upscale_invalid_image_2d(self):
        """Test upscale raises ProcessingError for 2D image"""
        module = SuperResolutionModule()
        
        invalid_image = np.random.rand(128, 128).astype(np.float32)
        
        with pytest.raises(ProcessingError, match="must have 3 channels"):
            module.upscale(invalid_image)
    
    def test_upscale_invalid_image_4_channels(self):
        """Test upscale raises ProcessingError for 4-channel image"""
        module = SuperResolutionModule()
        
        invalid_image = np.random.rand(128, 128, 4).astype(np.float32)
        
        with pytest.raises(ProcessingError, match="must have 3 channels"):
            module.upscale(invalid_image)
    
    def test_upscale_loads_model_if_not_loaded(self):
        """Test upscale automatically loads model if not loaded"""
        module = SuperResolutionModule()
        
        assert module.model is None
        
        test_image = np.random.rand(128, 128, 3).astype(np.float32)
        
        # Mock load_model to set model and upsampler
        def mock_load():
            module.model = Mock()
            mock_upsampler = Mock()
            mock_upsampler.enhance.return_value = (np.random.rand(512, 512, 3).astype(np.uint8), None)
            module.upsampler = mock_upsampler
        
        module.load_model = mock_load
        
        # Mock cv2 color conversion
        with patch('src.models.super_resolution.cv2') as mock_cv2:
            mock_cv2.cvtColor.side_effect = lambda img, code: img
            
            module.upscale(test_image)
        
        # Verify model was loaded
        assert module.model is not None
    
    @patch('src.models.super_resolution.MemoryManager')
    def test_unload_model_clears_memory(self, mock_memory_manager):
        """Test unload_model clears model and GPU memory"""
        module = SuperResolutionModule()
        
        # Set model and upsampler to simulate loaded state
        module.model = Mock()
        module.upsampler = Mock()
        
        module.unload_model()
        
        # Verify model and upsampler are cleared
        assert module.model is None
        assert module.upsampler is None
        
        # Verify MemoryManager.clear_cache was called
        mock_memory_manager.clear_cache.assert_called_once()
        
        # Verify memory logging was called
        assert mock_memory_manager.log_memory_usage.call_count >= 2
    
    @patch('src.models.super_resolution.torch')
    def test_unload_model_cpu_only(self, mock_torch):
        """Test unload_model works without GPU"""
        module = SuperResolutionModule(device='cpu')
        
        module.model = Mock()
        module.upsampler = Mock()
        
        # Mock CUDA not available
        mock_torch.cuda.is_available.return_value = False
        
        module.unload_model()
        
        # Verify model and upsampler are cleared
        assert module.model is None
        assert module.upsampler is None
        
        # Verify GPU cache was not called
        mock_torch.cuda.empty_cache.assert_not_called()
    
    def test_process_with_tiles_delegates_to_upscale(self):
        """Test _process_with_tiles delegates to upscale method"""
        module = SuperResolutionModule(scale=4)
        
        test_image = np.random.rand(1024, 1024, 3).astype(np.float32)
        
        # Mock the upsampler
        mock_upsampler = Mock()
        expected_output = np.random.rand(4096, 4096, 3).astype(np.uint8)
        mock_upsampler.enhance.return_value = (expected_output, None)
        module.upsampler = mock_upsampler
        module.model = Mock()
        
        # Mock cv2 color conversion
        with patch('src.models.super_resolution.cv2') as mock_cv2:
            mock_cv2.cvtColor.side_effect = lambda img, code: img
            
            result = module._process_with_tiles(test_image)
        
        # Verify result has correct shape
        assert result.shape == (4096, 4096, 3)
