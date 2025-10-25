"""
Denoising module for removing noise and artifacts from images.
"""

from abc import ABC, abstractmethod
import numpy as np
import cv2
from typing import Optional
import logging

from ..utils.memory import MemoryManager
from ..utils.exceptions import ProcessingError, ModelLoadError, ConfigurationError

logger = logging.getLogger(__name__)


class DenoisingModule(ABC):
    """
    Base class for image denoising modules.
    
    Provides abstract interface for different denoising implementations
    (OpenCV, NAFNet, etc.) with common initialization logic.
    """
    
    def __init__(self, strength: int = 10):
        """
        Initialize denoising module.
        
        Args:
            strength: Denoising strength parameter (1-100)
            
        Raises:
            ConfigurationError: If strength is out of valid range
        """
        if strength < 1 or strength > 100:
            raise ConfigurationError(f"Denoising strength must be between 1 and 100, got {strength}")
        
        self.strength = strength
    
    @abstractmethod
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from image.
        
        Args:
            image: Input image as numpy array (H, W, 3) with values in [0, 1]
            
        Returns:
            Denoised image as numpy array (H, W, 3) with values in [0, 1]
        """
        raise NotImplementedError("Subclasses must implement denoise() method")


class OpenCVDenoiser(DenoisingModule):
    """
    OpenCV-based denoising using fastNlMeansDenoisingColored.
    
    Fast CPU-based denoising suitable for quick processing.
    Uses Non-Local Means Denoising algorithm.
    """
    
    def __init__(self, strength: int = 10):
        """
        Initialize OpenCV denoiser.
        
        Args:
            strength: Denoising strength (1-100, default 10)
                     Higher values remove more noise but may blur details
        """
        super().__init__(strength)
        
        # Set template and search window sizes based on strength
        # Template window size: should be odd, typically 7
        self.template_window_size = 7
        
        # Search window size: should be odd, typically 21
        self.search_window_size = 21
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply OpenCV Non-Local Means Denoising to color image.
        
        Args:
            image: Input image as numpy array (H, W, 3) with values in [0, 1]
            
        Returns:
            Denoised image as numpy array (H, W, 3) with values in [0, 1]
            
        Raises:
            ProcessingError: If image format is invalid or denoising fails
        """
        try:
            # Validate input
            if image.ndim != 3 or image.shape[2] != 3:
                raise ProcessingError("Image must have 3 channels (RGB) for denoising")
            
            if image.size == 0:
                raise ProcessingError("Cannot denoise empty image")
            
            logger.info(f"Starting OpenCV denoising with strength={self.strength}")
            logger.debug(f"Image shape: {image.shape}, Template window: {self.template_window_size}, Search window: {self.search_window_size}")
            
            # Log memory before denoising (mainly for consistency, OpenCV is CPU-based)
            MemoryManager.log_memory_usage(logger, "Before OpenCV denoising:")
            
            # Convert from [0, 1] to [0, 255] for OpenCV
            image_uint8 = (image * 255).astype(np.uint8)
            
            # Apply fastNlMeansDenoisingColored
            # Parameters:
            # - h: filter strength for luminance component
            # - hColor: filter strength for color components (usually same as h)
            # - templateWindowSize: size of template patch (should be odd)
            # - searchWindowSize: size of search area (should be odd)
            logger.debug("Applying fastNlMeansDenoisingColored...")
            denoised = cv2.fastNlMeansDenoisingColored(
                image_uint8,
                None,
                h=self.strength,
                hColor=self.strength,
                templateWindowSize=self.template_window_size,
                searchWindowSize=self.search_window_size
            )
            
            # Validate output
            if denoised is None or denoised.size == 0:
                raise ProcessingError("Denoising produced empty output")
            
            # Convert back to [0, 1] range
            denoised_float = denoised.astype(np.float32) / 255.0
            
            # Log memory after denoising
            MemoryManager.log_memory_usage(logger, "After OpenCV denoising:")
            logger.info("OpenCV denoising complete")
            
            return denoised_float
            
        except ProcessingError:
            raise
        except Exception as e:
            logger.error(f"Error during OpenCV denoising: {str(e)}", exc_info=True)
            raise ProcessingError(f"OpenCV denoising failed: {str(e)}") from e


class NAFNetDenoiser(DenoisingModule):
    """
    NAFNet-based denoising (placeholder for future implementation).
    
    High-quality GPU-based denoising using NAFNet model.
    Requires model weights and GPU for inference.
    """
    
    def __init__(self, strength: int = 10, weights_path: Optional[str] = None, device: str = 'cuda'):
        """
        Initialize NAFNet denoiser.
        
        Args:
            strength: Denoising strength (1-100, default 10)
            weights_path: Path to NAFNet model weights
            device: Device for inference ('cuda' or 'cpu')
        """
        super().__init__(strength)
        self.weights_path = weights_path
        self.device = device
        self.model = None
    
    def load_model(self):
        """
        Lazy load NAFNet model.
        
        Raises:
            ModelLoadError: NAFNet support not yet implemented
        """
        # Log memory before loading (for future implementation)
        MemoryManager.log_memory_usage(logger, "Before loading NAFNet model:")
        
        raise ModelLoadError("NAFNet denoising will be implemented in future version")
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply NAFNet denoising (not yet implemented).
        
        Args:
            image: Input image as numpy array (H, W, 3) with values in [0, 1]
            
        Returns:
            Denoised image as numpy array (H, W, 3) with values in [0, 1]
            
        Raises:
            ModelLoadError: NAFNet support not yet implemented
        """
        # Log memory before denoising (for future implementation)
        MemoryManager.log_memory_usage(logger, "Before NAFNet denoising:")
        
        raise ModelLoadError("NAFNet denoising will be implemented in future version")
    
    def unload_model(self):
        """
        Unload NAFNet model and clear GPU memory (for future implementation).
        
        Raises:
            NotImplementedError: NAFNet support not yet implemented
        """
        logger.info("Unloading NAFNet model")
        MemoryManager.log_memory_usage(logger, "Before unloading:")
        
        self.model = None
        MemoryManager.clear_cache()
        
        MemoryManager.log_memory_usage(logger, "After unloading and cache clear:")


def create_denoiser(denoiser_type: str, strength: int = 10, **kwargs) -> DenoisingModule:
    """
    Factory function to create appropriate denoiser based on type.
    
    Args:
        denoiser_type: Type of denoiser ('opencv' or 'nafnet')
        strength: Denoising strength (1-100, default 10)
        **kwargs: Additional arguments for specific denoiser types
        
    Returns:
        Instance of appropriate DenoisingModule subclass
        
    Raises:
        ConfigurationError: If denoiser_type is not supported or configuration is invalid
    """
    try:
        logger.info(f"Creating denoiser: type={denoiser_type}, strength={strength}")
        
        if denoiser_type == 'opencv':
            denoiser = OpenCVDenoiser(strength=strength)
            logger.info("OpenCV denoiser created successfully")
            return denoiser
        elif denoiser_type == 'nafnet':
            denoiser = NAFNetDenoiser(
                strength=strength,
                weights_path=kwargs.get('weights_path'),
                device=kwargs.get('device', 'cuda')
            )
            logger.info("NAFNet denoiser created successfully")
            return denoiser
        else:
            logger.error(f"Unsupported denoiser type: {denoiser_type}")
            raise ConfigurationError(
                f"Unsupported denoiser type: {denoiser_type}. "
                f"Supported types: 'opencv', 'nafnet'"
            )
            
    except ConfigurationError:
        raise
    except Exception as e:
        logger.error(f"Error creating denoiser: {str(e)}", exc_info=True)
        raise ConfigurationError(f"Failed to create denoiser: {str(e)}") from e
