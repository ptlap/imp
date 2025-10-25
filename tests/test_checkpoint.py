"""
Unit tests for CheckpointManager.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from src.utils.checkpoint import CheckpointManager
from src.utils.exceptions import ProcessingError


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def checkpoint_manager(temp_checkpoint_dir):
    """Create CheckpointManager instance with temporary directory."""
    return CheckpointManager(checkpoint_dir=temp_checkpoint_dir)


@pytest.fixture
def sample_image():
    """Create sample image array."""
    return np.random.rand(100, 100, 3).astype(np.float32)


@pytest.fixture
def sample_metadata():
    """Create sample metadata dictionary."""
    return {
        'original_size': (200, 200),
        'is_grayscale': False,
        'resize_factor': 0.5
    }


def test_checkpoint_manager_initialization(temp_checkpoint_dir):
    """Test CheckpointManager can be initialized."""
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
    assert manager.checkpoint_dir == Path(temp_checkpoint_dir)
    assert manager.checkpoint_dir.exists()


def test_checkpoint_manager_creates_directory():
    """Test CheckpointManager creates checkpoint directory if it doesn't exist."""
    temp_dir = tempfile.mkdtemp()
    checkpoint_dir = Path(temp_dir) / "checkpoints"
    
    # Directory shouldn't exist yet
    assert not checkpoint_dir.exists()
    
    # Create manager
    manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))
    
    # Directory should now exist
    assert checkpoint_dir.exists()
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_save_checkpoint(checkpoint_manager, sample_image, sample_metadata):
    """Test saving checkpoint with image and metadata."""
    checkpoint_manager.save(sample_image, "test_checkpoint", sample_metadata)
    
    # Check file was created
    checkpoint_path = checkpoint_manager.checkpoint_dir / "test_checkpoint.pkl"
    assert checkpoint_path.exists()


def test_load_checkpoint(checkpoint_manager, sample_image, sample_metadata):
    """Test loading checkpoint returns correct image and metadata."""
    # Save checkpoint
    checkpoint_manager.save(sample_image, "test_checkpoint", sample_metadata)
    
    # Load checkpoint
    loaded_image, loaded_metadata = checkpoint_manager.load("test_checkpoint")
    
    # Verify image
    assert isinstance(loaded_image, np.ndarray)
    assert loaded_image.shape == sample_image.shape
    np.testing.assert_array_almost_equal(loaded_image, sample_image)
    
    # Verify metadata
    assert loaded_metadata == sample_metadata


def test_load_checkpoint_without_metadata(checkpoint_manager, sample_image):
    """Test loading checkpoint saved without metadata."""
    # Save without metadata
    checkpoint_manager.save(sample_image, "test_checkpoint", None)
    
    # Load checkpoint
    loaded_image, loaded_metadata = checkpoint_manager.load("test_checkpoint")
    
    # Verify image loaded correctly
    assert isinstance(loaded_image, np.ndarray)
    np.testing.assert_array_almost_equal(loaded_image, sample_image)
    
    # Metadata should be None
    assert loaded_metadata is None


def test_has_checkpoint(checkpoint_manager, sample_image):
    """Test checking checkpoint existence."""
    # Should not exist initially
    assert not checkpoint_manager.has("test_checkpoint")
    
    # Save checkpoint
    checkpoint_manager.save(sample_image, "test_checkpoint")
    
    # Should exist now
    assert checkpoint_manager.has("test_checkpoint")


def test_load_nonexistent_checkpoint(checkpoint_manager):
    """Test loading nonexistent checkpoint raises ProcessingError."""
    with pytest.raises(ProcessingError):
        checkpoint_manager.load("nonexistent_checkpoint")


def test_clear_checkpoints(checkpoint_manager, sample_image):
    """Test clearing all checkpoints."""
    # Save multiple checkpoints
    checkpoint_manager.save(sample_image, "checkpoint1")
    checkpoint_manager.save(sample_image, "checkpoint2")
    checkpoint_manager.save(sample_image, "checkpoint3")
    
    # Verify they exist
    assert checkpoint_manager.has("checkpoint1")
    assert checkpoint_manager.has("checkpoint2")
    assert checkpoint_manager.has("checkpoint3")
    
    # Clear all checkpoints
    count = checkpoint_manager.clear()
    
    # Should have removed 3 checkpoints
    assert count == 3
    
    # Verify they no longer exist
    assert not checkpoint_manager.has("checkpoint1")
    assert not checkpoint_manager.has("checkpoint2")
    assert not checkpoint_manager.has("checkpoint3")


def test_clear_empty_directory(checkpoint_manager):
    """Test clearing empty checkpoint directory."""
    count = checkpoint_manager.clear()
    assert count == 0


def test_checkpoint_naming_convention(checkpoint_manager, sample_image):
    """Test checkpoint naming follows expected convention."""
    image_id = "photo1"
    
    # Save checkpoints for different steps
    checkpoint_manager.save(sample_image, f"{image_id}_preprocessed")
    checkpoint_manager.save(sample_image, f"{image_id}_denoised")
    checkpoint_manager.save(sample_image, f"{image_id}_sr")
    
    # Verify all exist
    assert checkpoint_manager.has(f"{image_id}_preprocessed")
    assert checkpoint_manager.has(f"{image_id}_denoised")
    assert checkpoint_manager.has(f"{image_id}_sr")


def test_multiple_images_checkpoints(checkpoint_manager, sample_image):
    """Test managing checkpoints for multiple images."""
    # Save checkpoints for multiple images
    for i in range(3):
        image_id = f"photo{i}"
        checkpoint_manager.save(sample_image, f"{image_id}_preprocessed")
        checkpoint_manager.save(sample_image, f"{image_id}_denoised")
    
    # Verify all exist
    for i in range(3):
        image_id = f"photo{i}"
        assert checkpoint_manager.has(f"{image_id}_preprocessed")
        assert checkpoint_manager.has(f"{image_id}_denoised")
    
    # Clear should remove all
    count = checkpoint_manager.clear()
    assert count == 6  # 3 images * 2 checkpoints each
