"""
Super-resolution module for upscaling images using Real-ESRGAN.
"""

import numpy as np
from pathlib import Path
from typing import Optional
import cv2
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from ..utils.memory import MemoryManager
from ..utils.exceptions import ProcessingError, ModelLoadError, ConfigurationError, OutOfMemoryError

logger = logging.getLogger(__name__)


class SuperResolutionModule:
    """
    Real-ESRGAN based super-resolution for image upscaling.
    
    Supports 2x and 4x upscaling with tiling strategy for large images,
    FP16 inference for memory efficiency, and lazy model loading.
    """
    
    def __init__(
        self,
        scale: int = 4,
        weights_path: Optional[str] = None,
        tile_size: int = 512,
        tile_overlap: int = 64,
        device: str = 'cuda',
        use_fp16: bool = True
    ):
        """
        Initialize Super-Resolution module.
        
        Args:
            scale: Upscaling factor (2 or 4)
            weights_path: Path to Real-ESRGAN model weights
            tile_size: Size of tiles for processing large images
            tile_overlap: Overlap between tiles in pixels
            device: Device for inference ('cuda' or 'cpu')
            use_fp16: Use FP16 (half precision) for inference
            
        Raises:
            ConfigurationError: If scale is not 2 or 4, or tile configuration is invalid
        """
        if scale not in [2, 4]:
            raise ConfigurationError(f"Super-resolution scale must be 2 or 4, got {scale}")
        
        if tile_overlap >= tile_size:
            raise ConfigurationError(f"tile_overlap ({tile_overlap}) must be less than tile_size ({tile_size})")
        
        if tile_size <= 0:
            raise ConfigurationError(f"tile_size must be positive, got {tile_size}")
        
        if tile_overlap < 0:
            raise ConfigurationError(f"tile_overlap must be non-negative, got {tile_overlap}")
        
        self.scale = scale
        self.weights_path = weights_path
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.device = device if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu'
        self.use_fp16 = use_fp16 and self.device == 'cuda'
        
        # Model will be loaded lazily
        self.model = None
        self.upsampler = None
    
    def load_model(self):
        """
        Lazy load Real-ESRGAN model.
        
        Loads the model only when needed to save memory.
        Uses RealESRGAN from basicsr library.
        
        Raises:
            ModelLoadError: If required libraries are not installed, weights file doesn't exist, or loading fails
        """
        try:
            if self.model is not None:
                logger.debug("Super-Resolution model already loaded, skipping")
                return  # Already loaded
            
            logger.info(f"Loading Real-ESRGAN model (scale={self.scale}x, device={self.device}, fp16={self.use_fp16})")
            
            # Log memory before loading
            MemoryManager.log_memory_usage(logger, "Before loading Super-Resolution model:")
            
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                logger.debug("Successfully imported basicsr and realesrgan libraries")
            except ImportError as e:
                logger.error("Failed to import required libraries for Super-Resolution", exc_info=True)
                raise ModelLoadError(
                    "Required libraries not installed. "
                    "Please install: pip install basicsr realesrgan"
                ) from e
            
            # Check weights path
            if self.weights_path and not Path(self.weights_path).exists():
                logger.error(f"Weights file not found: {self.weights_path}")
                raise ModelLoadError(f"Weights file not found: {self.weights_path}")
            
            if not self.weights_path:
                raise ModelLoadError("Weights path is required but not provided")
            
            logger.info(f"Using weights from: {self.weights_path}")
            
            # Create model architecture
            # RRDBNet: num_in_ch, num_out_ch, num_feat, num_block, num_grow_ch
            logger.debug("Creating RRDBNet architecture...")
            try:
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=self.scale
                )
            except Exception as e:
                logger.error(f"Failed to create RRDBNet architecture: {str(e)}", exc_info=True)
                raise ModelLoadError(f"Failed to create model architecture: {str(e)}") from e
            
            # Create upsampler
            logger.debug(f"Creating RealESRGANer with tile_size={self.tile_size}, tile_overlap={self.tile_overlap}")
            try:
                self.upsampler = RealESRGANer(
                    scale=self.scale,
                    model_path=self.weights_path,
                    model=model,
                    tile=self.tile_size,
                    tile_pad=self.tile_overlap,
                    pre_pad=0,
                    half=self.use_fp16,
                    device=self.device
                )
            except Exception as e:
                logger.error(f"Failed to create RealESRGANer: {str(e)}", exc_info=True)
                raise ModelLoadError(f"Failed to initialize upsampler: {str(e)}") from e
            
            self.model = model
            
            # Log memory after loading
            MemoryManager.log_memory_usage(logger, "After loading Super-Resolution model:")
            logger.info("Real-ESRGAN model loaded successfully")
            
        except ModelLoadError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading Super-Resolution model: {str(e)}", exc_info=True)
            raise ModelLoadError(f"Failed to load Super-Resolution model: {str(e)}") from e
    
    def upscale(self, image: np.ndarray) -> np.ndarray:
        """
        Upscale image using Real-ESRGAN.
        
        Automatically handles tiling for large images and converts
        between RGB and BGR color spaces as needed.
        
        Args:
            image: Input image as numpy array (H, W, 3) with values in [0, 1]
                   Expected in RGB format
            
        Returns:
            Upscaled image as numpy array (H*scale, W*scale, 3) with values in [0, 1]
            
        Raises:
            ProcessingError: If image format is invalid or upscaling fails
            OutOfMemoryError: If GPU runs out of memory during upscaling
        """
        try:
            # Validate input
            if image.ndim != 3 or image.shape[2] != 3:
                raise ProcessingError("Image must have 3 channels (RGB) for super-resolution")
            
            if image.size == 0:
                raise ProcessingError("Cannot upscale empty image")
            
            input_shape = image.shape
            logger.info(f"Starting super-resolution upscaling: {input_shape[1]}x{input_shape[0]} -> {input_shape[1]*self.scale}x{input_shape[0]*self.scale} ({self.scale}x)")
            
            # Check if tiling will be used
            if self._should_tile(image):
                logger.info(f"Image size exceeds tile_size ({self.tile_size}), tiling will be applied")
            
            # Load model if not already loaded
            if self.model is None:
                self.load_model()
            
            # Log memory before upscaling
            MemoryManager.log_memory_usage(logger, "Before upscaling:")
            
            # Convert from [0, 1] to [0, 255] and RGB to BGR for Real-ESRGAN
            logger.debug("Converting image format for Real-ESRGAN processing")
            image_uint8 = (image * 255).astype(np.uint8)
            image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
            
            # Perform upscaling with error handling
            # RealESRGANer.enhance() returns (output, _)
            logger.debug("Running Real-ESRGAN inference...")
            try:
                output_bgr, _ = self.upsampler.enhance(image_bgr, outscale=self.scale)
            except RuntimeError as e:
                # Check if it's an OOM error
                error_msg = str(e).lower()
                if 'out of memory' in error_msg or 'cuda' in error_msg:
                    logger.error(f"GPU out of memory during upscaling: {str(e)}", exc_info=True)
                    raise OutOfMemoryError(
                        f"GPU out of memory during super-resolution. "
                        f"Try reducing tile_size or processing a smaller image. Error: {str(e)}"
                    ) from e
                else:
                    raise ProcessingError(f"Super-resolution inference failed: {str(e)}") from e
            
            # Validate output
            if output_bgr is None or output_bgr.size == 0:
                raise ProcessingError("Super-resolution produced empty output")
            
            # Convert back to RGB and [0, 1] range
            logger.debug("Converting output back to RGB format")
            output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
            output_float = output_rgb.astype(np.float32) / 255.0
            
            # Log memory after upscaling
            MemoryManager.log_memory_usage(logger, "After upscaling:")
            logger.info(f"Super-resolution complete: output shape {output_float.shape}")
            
            return output_float
            
        except (ProcessingError, OutOfMemoryError, ModelLoadError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error during super-resolution: {str(e)}", exc_info=True)
            raise ProcessingError(f"Super-resolution failed: {str(e)}") from e
    
    def _should_tile(self, image: np.ndarray) -> bool:
        """
        Check if image needs tiling based on size.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            True if image should be tiled, False otherwise
        """
        height, width = image.shape[:2]
        # Tile if either dimension exceeds tile_size
        return height > self.tile_size or width > self.tile_size
    
    def _process_with_tiles(self, image: np.ndarray) -> np.ndarray:
        """
        Process large image using tiling strategy.
        
        Divides image into overlapping tiles, processes each independently,
        and merges with feathering in overlap regions for seamless results.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            
        Returns:
            Processed image as numpy array
        """
        # Note: RealESRGANer already handles tiling internally
        # This method is kept for potential custom tiling implementation
        # For now, we rely on the upsampler's built-in tiling
        
        # Log memory during tiling operations
        logger.info(f"Processing with tiling strategy (tile_size={self.tile_size}, overlap={self.tile_overlap})")
        MemoryManager.log_memory_usage(logger, "Before tiled processing:")
        
        result = self.upscale(image)
        
        # Clear cache after tiling to free memory
        logger.debug("Clearing GPU cache after tiled processing")
        MemoryManager.clear_cache()
        MemoryManager.log_memory_usage(logger, "After tiled processing and cache clear:")
        
        return result
    
    def unload_model(self):
        """
        Unload model and clear GPU memory.
        
        Useful for freeing memory after processing is complete.
        """
        logger.info("Unloading Super-Resolution model")
        MemoryManager.log_memory_usage(logger, "Before unloading:")
        
        self.model = None
        self.upsampler = None
        
        # Use MemoryManager for consistent memory cleanup
        MemoryManager.clear_cache()
        
        MemoryManager.log_memory_usage(logger, "After unloading and cache clear:")
