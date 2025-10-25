"""
Preprocessing module for image loading and preparation.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
from PIL import Image
import cv2
import logging

from .exceptions import ProcessingError


class Preprocessor:
    """
    Image preprocessing for old photo restoration.
    
    Handles image loading, validation, grayscale detection, resizing,
    and normalization to prepare images for processing pipeline.
    """
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    def __init__(self, max_size: int = 2048, logger: Optional[logging.Logger] = None):
        """
        Initialize preprocessor.
        
        Args:
            max_size: Maximum dimension (width or height) before resizing
            logger: Logger instance for logging events
        """
        self.max_size = max_size
        self.logger = logger or logging.getLogger('imp.preprocessing')
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and validate image from file path.
        
        Args:
            image_path: Path to input image file
            
        Returns:
            Image as numpy array in RGB format (H, W, 3)
            
        Raises:
            ProcessingError: If image file doesn't exist, format is unsupported, or loading fails
        """
        self.logger.debug(f"Loading image: {image_path}")
        
        try:
            path = Path(image_path)
            
            # Check file exists
            if not path.exists():
                self.logger.error(f"Image file not found: {image_path}")
                raise ProcessingError(f"Image file not found: {image_path}")
            
            # Validate format
            if path.suffix not in self.SUPPORTED_FORMATS:
                self.logger.error(f"Unsupported image format: {path.suffix}")
                raise ProcessingError(
                    f"Unsupported image format: {path.suffix}. "
                    f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
                )
            
            # Load image using PIL
            try:
                image = Image.open(image_path)
                original_mode = image.mode
                # Convert to RGB (handles RGBA, grayscale, etc.)
                image = image.convert('RGB')
                # Convert to numpy array
                image_array = np.array(image)
                
                # Validate image dimensions
                if image_array.size == 0:
                    raise ProcessingError(f"Image has zero size: {image_path}")
                
                if image_array.ndim != 3 or image_array.shape[2] != 3:
                    raise ProcessingError(
                        f"Image must have 3 channels after RGB conversion, got shape: {image_array.shape}"
                    )
                
                self.logger.info(f"Image loaded successfully: {image_path} (mode: {original_mode}, shape: {image_array.shape})")
                return image_array
                
            except ProcessingError:
                # Re-raise ProcessingError as-is
                raise
            except Exception as e:
                self.logger.error(f"Failed to load image {image_path}: {str(e)}", exc_info=True)
                raise ProcessingError(f"Failed to load image {image_path}: {str(e)}") from e
                
        except ProcessingError:
            # Re-raise ProcessingError as-is
            raise
        except Exception as e:
            # Wrap any other unexpected errors
            self.logger.error(f"Unexpected error loading image {image_path}: {str(e)}", exc_info=True)
            raise ProcessingError(f"Unexpected error loading image: {str(e)}") from e
    
    def detect_grayscale(self, image: np.ndarray, tolerance: float = 1.0) -> bool:
        """
        Detect if image is grayscale by comparing RGB channels.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            tolerance: Maximum allowed difference between channels
            
        Returns:
            True if image is grayscale, False otherwise
            
        Raises:
            ProcessingError: If image format is invalid
        """
        try:
            if image.ndim != 3 or image.shape[2] != 3:
                raise ProcessingError("Image must have 3 channels (RGB) for grayscale detection")
            
            # Compare R, G, B channels
            r_channel = image[:, :, 0]
            g_channel = image[:, :, 1]
            b_channel = image[:, :, 2]
            
            # Calculate differences
            rg_diff = np.abs(r_channel.astype(float) - g_channel.astype(float))
            rb_diff = np.abs(r_channel.astype(float) - b_channel.astype(float))
            
            # Check if differences are within tolerance
            is_grayscale = (np.mean(rg_diff) <= tolerance and 
                           np.mean(rb_diff) <= tolerance)
            
            self.logger.debug(f"Grayscale detection: {is_grayscale} (RG diff: {np.mean(rg_diff):.2f}, RB diff: {np.mean(rb_diff):.2f})")
            
            return is_grayscale
            
        except ProcessingError:
            raise
        except Exception as e:
            self.logger.error(f"Error during grayscale detection: {str(e)}", exc_info=True)
            raise ProcessingError(f"Failed to detect grayscale: {str(e)}") from e
    
    def smart_resize(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize image if it exceeds max_size while maintaining aspect ratio.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            
        Returns:
            Tuple of (resized_image, scale_factor)
            - resized_image: Resized image or original if no resize needed
            - scale_factor: Ratio of new size to original (1.0 if no resize)
            
        Raises:
            ProcessingError: If resizing fails
        """
        try:
            height, width = image.shape[:2]
            max_dim = max(height, width)
            
            # No resize needed
            if max_dim <= self.max_size:
                self.logger.debug(f"No resize needed: {width}x{height} <= {self.max_size}")
                return image, 1.0
            
            # Calculate scale factor
            scale_factor = self.max_size / max_dim
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Validate new dimensions
            if new_width <= 0 or new_height <= 0:
                raise ProcessingError(f"Invalid resize dimensions: {new_width}x{new_height}")
            
            self.logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height} (scale: {scale_factor:.3f})")
            
            # Resize using INTER_AREA for downscaling
            resized = cv2.resize(
                image, 
                (new_width, new_height), 
                interpolation=cv2.INTER_AREA
            )
            
            return resized, scale_factor
            
        except ProcessingError:
            raise
        except Exception as e:
            self.logger.error(f"Error during image resize: {str(e)}", exc_info=True)
            raise ProcessingError(f"Failed to resize image: {str(e)}") from e
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to [0, 1] range.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Normalized image with values in [0, 1]
        """
        return image.astype(np.float32) / 255.0
    
    def process(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Complete preprocessing pipeline for an image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (processed_image, metadata)
            - processed_image: Preprocessed image as numpy array (H, W, 3)
            - metadata: Dictionary containing:
                - original_size: (height, width) tuple
                - is_grayscale: bool
                - resize_factor: float
                
        Raises:
            ProcessingError: If any preprocessing step fails
        """
        try:
            self.logger.info(f"Starting preprocessing for: {image_path}")
            
            # Load image
            image = self.load_image(image_path)
            
            # Store original size
            original_size = (image.shape[0], image.shape[1])
            
            # Detect grayscale
            is_grayscale = self.detect_grayscale(image)
            
            # Smart resize
            image, resize_factor = self.smart_resize(image)
            
            # Normalize
            self.logger.debug("Normalizing pixel values to [0, 1]")
            image = self.normalize(image)
            
            # Create metadata
            metadata = {
                'original_size': original_size,
                'is_grayscale': is_grayscale,
                'resize_factor': resize_factor
            }
            
            self.logger.info(f"Preprocessing complete - Original: {original_size}, Grayscale: {is_grayscale}, Resize factor: {resize_factor:.3f}")
            
            return image, metadata
            
        except ProcessingError:
            # Re-raise ProcessingError as-is
            raise
        except Exception as e:
            # Wrap any other unexpected errors
            self.logger.error(f"Unexpected error during preprocessing: {str(e)}", exc_info=True)
            raise ProcessingError(f"Preprocessing failed: {str(e)}") from e
