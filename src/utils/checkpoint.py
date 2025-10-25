"""
Checkpoint management for saving and loading intermediate processing results.
"""

import pickle
import time
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import logging

from .exceptions import ProcessingError

logger = logging.getLogger('imp.checkpoint')


class CheckpointManager:
    """
    Manage processing checkpoints for image restoration pipeline.
    
    Saves intermediate results after each processing step to enable
    resume functionality and debugging.
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"CheckpointManager initialized with directory: {self.checkpoint_dir}")
    
    def save(self, image: np.ndarray, name: str, metadata: Optional[Dict] = None) -> None:
        """
        Save checkpoint with image and metadata.
        
        Args:
            image: Image array to save
            name: Checkpoint name (used as filename without extension)
            metadata: Optional metadata dictionary to save with image
            
        Raises:
            ProcessingError: If checkpoint cannot be saved
        """
        try:
            checkpoint_path = self.checkpoint_dir / f"{name}.pkl"
            
            data = {
                'image': image,
                'metadata': metadata,
                'timestamp': time.time()
            }
            
            logger.debug(f"Saving checkpoint: {name} (shape: {image.shape})")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Checkpoint saved: {name}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint {name}: {str(e)}", exc_info=True)
            raise ProcessingError(f"Failed to save checkpoint {name}: {str(e)}") from e
    
    def load(self, name: str) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Load checkpoint and return image and metadata.
        
        Args:
            name: Checkpoint name (without extension)
            
        Returns:
            Tuple of (image, metadata)
            - image: Loaded image array
            - metadata: Metadata dictionary or None
            
        Raises:
            ProcessingError: If checkpoint doesn't exist or cannot be loaded
        """
        try:
            checkpoint_path = self.checkpoint_dir / f"{name}.pkl"
            
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: {name}")
                raise ProcessingError(f"Checkpoint not found: {name}")
            
            logger.debug(f"Loading checkpoint: {name}")
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            
            if 'image' not in data:
                raise ProcessingError(f"Invalid checkpoint format: missing 'image' key in {name}")
            
            image = data['image']
            logger.info(f"Checkpoint loaded: {name} (shape: {image.shape})")
            return image, data.get('metadata')
            
        except ProcessingError:
            raise
        except Exception as e:
            logger.error(f"Failed to load checkpoint {name}: {str(e)}", exc_info=True)
            raise ProcessingError(f"Failed to load checkpoint {name}: {str(e)}") from e
    
    def has(self, name: str) -> bool:
        """
        Check if checkpoint exists.
        
        Args:
            name: Checkpoint name (without extension)
            
        Returns:
            True if checkpoint exists, False otherwise
        """
        checkpoint_path = self.checkpoint_dir / f"{name}.pkl"
        return checkpoint_path.exists()
    
    def clear(self) -> int:
        """
        Remove all checkpoints from checkpoint directory.
        
        Returns:
            Number of checkpoints removed
        """
        logger.info("Clearing all checkpoints")
        count = 0
        for checkpoint_file in self.checkpoint_dir.glob("*.pkl"):
            try:
                checkpoint_file.unlink()
                count += 1
                logger.debug(f"Removed checkpoint: {checkpoint_file.name}")
            except Exception as e:
                # Continue removing other checkpoints even if one fails
                logger.warning(f"Failed to remove checkpoint {checkpoint_file.name}: {str(e)}")
                pass
        logger.info(f"Cleared {count} checkpoint(s)")
        return count
