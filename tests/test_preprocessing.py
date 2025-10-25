"""
Unit tests for preprocessing module.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image

from src.utils.exceptions import ProcessingError

from src.utils.preprocessing import Preprocessor


class TestPreprocessor:
    """Test suite for Preprocessor class"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance"""
        return Preprocessor(max_size=2048)
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test images"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def create_test_image(self, path: Path, size: tuple, color: tuple = (255, 0, 0)):
        """Helper to create test image"""
        img = Image.new('RGB', size, color)
        img.save(path)
        return path
    
    def create_grayscale_image(self, path: Path, size: tuple, gray_value: int = 128):
        """Helper to create grayscale image"""
        img = Image.new('RGB', size, (gray_value, gray_value, gray_value))
        img.save(path)
        return path
    
    # Test image loading
    def test_load_image_valid_jpg(self, preprocessor, temp_dir):
        """Test loading valid JPG image"""
        img_path = self.create_test_image(temp_dir / "test.jpg", (100, 100))
        result = preprocessor.load_image(str(img_path))
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8
    
    def test_load_image_valid_png(self, preprocessor, temp_dir):
        """Test loading valid PNG image"""
        img_path = self.create_test_image(temp_dir / "test.png", (100, 100))
        result = preprocessor.load_image(str(img_path))
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100, 3)
    
    def test_load_image_nonexistent_file(self, preprocessor):
        """Test loading nonexistent file raises ProcessingError"""
        with pytest.raises(ProcessingError):
            preprocessor.load_image("nonexistent.jpg")
    
    def test_load_image_unsupported_format(self, preprocessor, temp_dir):
        """Test loading unsupported format raises ValueError"""
        # Create a text file with wrong extension
        invalid_path = temp_dir / "test.txt"
        invalid_path.write_text("not an image")
        
        with pytest.raises(ProcessingError, match="Unsupported image format"):
            preprocessor.load_image(str(invalid_path))
    
    # Test grayscale detection
    def test_detect_grayscale_on_color_image(self, preprocessor):
        """Test grayscale detection returns False for color image"""
        # Create color image (red)
        color_image = np.zeros((100, 100, 3), dtype=np.uint8)
        color_image[:, :, 0] = 255  # Red channel
        
        result = preprocessor.detect_grayscale(color_image)
        assert result == False
    
    def test_detect_grayscale_on_grayscale_image(self, preprocessor):
        """Test grayscale detection returns True for grayscale image"""
        # Create grayscale image (all channels equal)
        gray_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        result = preprocessor.detect_grayscale(gray_image)
        assert result == True
    
    def test_detect_grayscale_with_tolerance(self, preprocessor):
        """Test grayscale detection with small channel differences"""
        # Create nearly grayscale image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        image[:, :, 0] = 129  # Slight difference in red channel
        
        # Should be detected as grayscale with default tolerance
        result = preprocessor.detect_grayscale(image, tolerance=2.0)
        assert result == True
    
    def test_detect_grayscale_invalid_shape(self, preprocessor):
        """Test grayscale detection raises error for invalid shape"""
        invalid_image = np.zeros((100, 100), dtype=np.uint8)
        
        with pytest.raises(ProcessingError, match="must have 3 channels"):
            preprocessor.detect_grayscale(invalid_image)
    
    # Test smart resizing
    def test_smart_resize_small_image_no_resize(self, preprocessor):
        """Test small image is not resized"""
        small_image = np.zeros((512, 512, 3), dtype=np.uint8)
        
        resized, scale_factor = preprocessor.smart_resize(small_image)
        
        assert resized.shape == (512, 512, 3)
        assert scale_factor == 1.0
        assert np.array_equal(resized, small_image)
    
    def test_smart_resize_large_image_width(self, preprocessor):
        """Test large image is resized (width exceeds max)"""
        large_image = np.zeros((1000, 3000, 3), dtype=np.uint8)
        
        resized, scale_factor = preprocessor.smart_resize(large_image)
        
        # Max dimension should be 2048
        assert max(resized.shape[:2]) == 2048
        assert resized.shape[0] < 1000  # Height reduced
        assert resized.shape[1] == 2048  # Width is max
        assert scale_factor < 1.0
    
    def test_smart_resize_large_image_height(self, preprocessor):
        """Test large image is resized (height exceeds max)"""
        large_image = np.zeros((3000, 1000, 3), dtype=np.uint8)
        
        resized, scale_factor = preprocessor.smart_resize(large_image)
        
        # Max dimension should be 2048
        assert max(resized.shape[:2]) == 2048
        assert resized.shape[0] == 2048  # Height is max
        assert resized.shape[1] < 1000  # Width reduced
        assert scale_factor < 1.0
    
    def test_smart_resize_maintains_aspect_ratio(self, preprocessor):
        """Test resizing maintains aspect ratio"""
        image = np.zeros((3000, 1500, 3), dtype=np.uint8)
        original_aspect = 3000 / 1500
        
        resized, scale_factor = preprocessor.smart_resize(image)
        new_aspect = resized.shape[0] / resized.shape[1]
        
        # Aspect ratio should be preserved (within rounding error)
        assert abs(original_aspect - new_aspect) < 0.01
    
    # Test normalization
    def test_normalize_converts_to_float(self, preprocessor):
        """Test normalization converts to float32"""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        normalized = preprocessor.normalize(image)
        
        assert normalized.dtype == np.float32
        assert np.allclose(normalized, 1.0)
    
    def test_normalize_range(self, preprocessor):
        """Test normalization produces values in [0, 1]"""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        normalized = preprocessor.normalize(image)
        
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
    
    # Test full process pipeline
    def test_process_returns_tuple(self, preprocessor, temp_dir):
        """Test process returns tuple of image and metadata"""
        img_path = self.create_test_image(temp_dir / "test.jpg", (512, 512))
        
        result = preprocessor.process(str(img_path))
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_process_metadata_structure(self, preprocessor, temp_dir):
        """Test process returns correct metadata structure"""
        img_path = self.create_test_image(temp_dir / "test.jpg", (512, 512))
        
        image, metadata = preprocessor.process(str(img_path))
        
        assert 'original_size' in metadata
        assert 'is_grayscale' in metadata
        assert 'resize_factor' in metadata
        assert metadata['original_size'] == (512, 512)
        assert metadata['is_grayscale'] in [True, False]
        assert isinstance(metadata['resize_factor'], float)
    
    def test_process_small_color_image(self, preprocessor, temp_dir):
        """Test processing small color image"""
        img_path = self.create_test_image(temp_dir / "test.jpg", (512, 512), (255, 0, 0))
        
        image, metadata = preprocessor.process(str(img_path))
        
        assert image.shape == (512, 512, 3)
        assert image.dtype == np.float32
        assert metadata['original_size'] == (512, 512)
        assert metadata['is_grayscale'] == False
        assert metadata['resize_factor'] == 1.0
    
    def test_process_large_image_resizes(self, preprocessor, temp_dir):
        """Test processing large image triggers resize"""
        # Create large image
        large_img = Image.new('RGB', (3000, 2000), (255, 0, 0))
        img_path = temp_dir / "large.jpg"
        large_img.save(img_path)
        
        image, metadata = preprocessor.process(str(img_path))
        
        assert max(image.shape[:2]) <= 2048
        assert metadata['original_size'] == (2000, 3000)
        assert metadata['resize_factor'] < 1.0
    
    def test_process_grayscale_image(self, preprocessor, temp_dir):
        """Test processing grayscale image"""
        img_path = self.create_grayscale_image(temp_dir / "gray.jpg", (512, 512))
        
        image, metadata = preprocessor.process(str(img_path))
        
        assert metadata['is_grayscale'] == True
