"""
Integration tests for pipeline orchestrator.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image
import shutil

from src.pipeline import OldPhotoRestoration
from src.config import Config
from src.utils.exceptions import ProcessingError


class TestOldPhotoRestoration:
    """Test suite for OldPhotoRestoration pipeline"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def test_config(self, temp_dir):
        """Create test configuration"""
        config = Config.default()
        # Disable GPU-dependent modules for testing
        config.models.super_resolution.skip = True
        config.processing.checkpoint_dir = str(temp_dir / "checkpoints")
        config.logging.file = str(temp_dir / "test.log")
        return config
    
    @pytest.fixture
    def pipeline(self, test_config):
        """Create pipeline instance"""
        return OldPhotoRestoration(config=test_config)
    
    def create_test_image(self, path: Path, size: tuple = (256, 256), color: tuple = (255, 0, 0)):
        """Helper to create test image"""
        img = Image.new('RGB', size, color)
        img.save(path)
        return path
    
    # Test pipeline initialization
    def test_pipeline_initialization_with_config(self, test_config):
        """Test pipeline can be initialized with config"""
        pipeline = OldPhotoRestoration(config=test_config)
        
        assert pipeline.config == test_config
        assert pipeline.preprocessor is not None
        assert pipeline.checkpoint_mgr is not None
        assert pipeline.logger is not None
        assert pipeline.denoiser is None  # Lazy loaded
        assert pipeline.super_resolver is None  # Lazy loaded
    
    def test_pipeline_initialization_without_config(self):
        """Test pipeline can be initialized with default config"""
        pipeline = OldPhotoRestoration()
        
        assert pipeline.config is not None
        assert pipeline.preprocessor is not None
        assert pipeline.checkpoint_mgr is not None
    
    # Test single image restoration flow
    def test_restore_basic_flow(self, pipeline, temp_dir):
        """Test basic restoration flow without GPU models"""
        # Create test image
        input_path = self.create_test_image(temp_dir / "input.jpg")
        output_path = temp_dir / "output.png"
        
        # Run restoration (only preprocessing and denoising, SR is skipped)
        result = pipeline.restore(str(input_path), str(output_path))
        
        # Check result
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        
        # Check output file was created
        assert output_path.exists()
    
    def test_restore_without_output_path(self, pipeline, temp_dir):
        """Test restoration without saving to file"""
        input_path = self.create_test_image(temp_dir / "input.jpg")
        
        result = pipeline.restore(str(input_path))
        
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
    
    def test_restore_nonexistent_file(self, pipeline):
        """Test restoration with nonexistent file raises error"""
        with pytest.raises(ProcessingError):
            pipeline.restore("nonexistent.jpg")
    
    def test_restore_creates_output_directory(self, pipeline, temp_dir):
        """Test restoration creates output directory if needed"""
        input_path = self.create_test_image(temp_dir / "input.jpg")
        output_path = temp_dir / "subdir" / "output.png"
        
        result = pipeline.restore(str(input_path), str(output_path))
        
        assert output_path.exists()
        assert output_path.parent.exists()
    
    # Test checkpoint functionality
    def test_restore_with_checkpoints(self, pipeline, temp_dir):
        """Test restoration creates checkpoints"""
        input_path = self.create_test_image(temp_dir / "input.jpg")
        
        # Enable checkpoints
        pipeline.config.processing.checkpoint_enabled = True
        
        result = pipeline.restore(str(input_path))
        
        # Check checkpoints were created
        image_id = input_path.stem
        assert pipeline.checkpoint_mgr.has(f"{image_id}_preprocessed")
        assert pipeline.checkpoint_mgr.has(f"{image_id}_denoised")
    
    def test_restore_resume_from_checkpoint(self, pipeline, temp_dir):
        """Test restoration can resume from checkpoint"""
        input_path = self.create_test_image(temp_dir / "input.jpg")
        
        # Enable checkpoints
        pipeline.config.processing.checkpoint_enabled = True
        
        # First run - creates checkpoints
        result1 = pipeline.restore(str(input_path))
        
        # Second run - should resume from checkpoints
        result2 = pipeline.restore(str(input_path), resume=True)
        
        # Results should be identical
        assert np.allclose(result1, result2)
    
    def test_restore_skip_checkpoint_resume(self, pipeline, temp_dir):
        """Test restoration can skip checkpoint resume"""
        input_path = self.create_test_image(temp_dir / "input.jpg")
        
        # Enable checkpoints
        pipeline.config.processing.checkpoint_enabled = True
        
        # First run
        result1 = pipeline.restore(str(input_path))
        
        # Second run with resume=False
        result2 = pipeline.restore(str(input_path), resume=False)
        
        # Results should still be close (same processing)
        assert np.allclose(result1, result2)
    
    # Test batch processing
    def test_batch_restore_multiple_images(self, pipeline, temp_dir):
        """Test batch processing of multiple images"""
        # Create test images
        input_paths = [
            str(self.create_test_image(temp_dir / f"input_{i}.jpg"))
            for i in range(3)
        ]
        output_dir = temp_dir / "output"
        
        successes, failures = pipeline.batch_restore(input_paths, str(output_dir))
        
        # Check results
        assert len(successes) == 3
        assert len(failures) == 0
        
        # Check output files exist
        for success in successes:
            assert Path(success['output_path']).exists()
    
    def test_batch_restore_skip_existing(self, pipeline, temp_dir):
        """Test batch processing skips already processed images"""
        # Create test images
        input_paths = [
            str(self.create_test_image(temp_dir / f"input_{i}.jpg"))
            for i in range(2)
        ]
        output_dir = temp_dir / "output"
        
        # First run
        successes1, failures1 = pipeline.batch_restore(input_paths, str(output_dir))
        assert len(successes1) == 2
        
        # Second run - should skip existing
        successes2, failures2 = pipeline.batch_restore(input_paths, str(output_dir))
        assert len(successes2) == 2
        
        # Check that images were skipped
        for success in successes2:
            assert success.get('skipped', False) == True
    
    def test_batch_restore_with_failures(self, pipeline, temp_dir):
        """Test batch processing handles failures gracefully"""
        # Create mix of valid and invalid paths
        input_paths = [
            str(self.create_test_image(temp_dir / "input_1.jpg")),
            "nonexistent.jpg",  # This will fail
            str(self.create_test_image(temp_dir / "input_2.jpg"))
        ]
        output_dir = temp_dir / "output"
        
        successes, failures = pipeline.batch_restore(input_paths, str(output_dir), max_retries=1)
        
        # Should have 2 successes and 1 failure
        assert len(successes) == 2
        assert len(failures) == 1
        assert failures[0]['input_path'] == "nonexistent.jpg"
    
    def test_batch_restore_creates_output_dir(self, pipeline, temp_dir):
        """Test batch processing creates output directory"""
        input_paths = [str(self.create_test_image(temp_dir / "input.jpg"))]
        output_dir = temp_dir / "new_output"
        
        assert not output_dir.exists()
        
        successes, failures = pipeline.batch_restore(input_paths, str(output_dir))
        
        assert output_dir.exists()
    
    # Test error handling
    def test_restore_handles_processing_error(self, pipeline, temp_dir):
        """Test restoration handles processing errors"""
        # Create invalid image file
        invalid_path = temp_dir / "invalid.jpg"
        invalid_path.write_text("not an image")
        
        with pytest.raises(ProcessingError):
            pipeline.restore(str(invalid_path))
    
    # Test model lifecycle
    def test_denoiser_lazy_loading(self, pipeline, temp_dir):
        """Test denoiser is loaded lazily"""
        assert pipeline.denoiser is None
        
        input_path = self.create_test_image(temp_dir / "input.jpg")
        pipeline.restore(str(input_path))
        
        # After restoration, denoiser should be unloaded
        assert pipeline.denoiser is None
    
    def test_skip_denoising(self, pipeline, temp_dir):
        """Test denoising can be skipped"""
        pipeline.config.models.denoising.skip = True
        
        input_path = self.create_test_image(temp_dir / "input.jpg")
        result = pipeline.restore(str(input_path))
        
        # Should still return valid result
        assert isinstance(result, np.ndarray)
        # Denoiser should never be loaded
        assert pipeline.denoiser is None
    
    def test_skip_super_resolution(self, pipeline, temp_dir):
        """Test super-resolution can be skipped"""
        pipeline.config.models.super_resolution.skip = True
        
        input_path = self.create_test_image(temp_dir / "input.jpg")
        result = pipeline.restore(str(input_path))
        
        # Should still return valid result
        assert isinstance(result, np.ndarray)
        # Super-resolver should never be loaded
        assert pipeline.super_resolver is None
