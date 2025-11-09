"""
Colorization module for automatic colorization of grayscale images using DDColor.
"""

import numpy as np
from pathlib import Path
from typing import Optional
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from ..utils.memory import MemoryManager
from ..utils.exceptions import ProcessingError, ModelLoadError, ConfigurationError

logger = logging.getLogger(__name__)


class ColorizationError(ProcessingError):
    """
    Exception raised during colorization operations.
    
    Raised when:
    - Colorization model inference fails
    - Color space conversion fails
    - Invalid image format for colorization
    """
    pass


class ColorizationModule:
    """
    DDColor-based colorization for grayscale images.
    
    Automatically detects grayscale images and applies colorization using
    the DDColor model with Lab color space processing. Supports lazy loading,
    FP16 inference, and graceful error handling.
    """
    
    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = 'cuda',
        use_fp16: bool = True
    ):
        """
        Initialize Colorization module.
        
        Args:
            weights_path: Path to DDColor model weights
            device: Device for inference ('cuda' or 'cpu')
            use_fp16: Use FP16 (half precision) for inference
        """
        self.weights_path = weights_path
        self.device = device if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu'
        self.use_fp16 = use_fp16 and self.device == 'cuda'
        
        # Model will be loaded lazily
        self.model = None
    
    def load_model(self):
        """
        Lazy load DDColor model.
        
        Loads the model only when needed to save memory.
        
        Raises:
            ModelLoadError: If required libraries are not installed, weights file doesn't exist, or loading fails
        """
        try:
            if self.model is not None:
                logger.debug("Colorization model already loaded, skipping")
                return  # Already loaded
            
            logger.info(f"Loading DDColor model (device={self.device}, fp16={self.use_fp16})")
            
            # Log memory before loading
            MemoryManager.log_memory_usage(logger, "Before loading Colorization model:")
            
            # Check PyTorch availability
            if not TORCH_AVAILABLE:
                raise ModelLoadError(
                    "PyTorch not installed. "
                    "Please install: pip install torch torchvision"
                )
            
            # Check weights path
            if self.weights_path and not Path(self.weights_path).exists():
                logger.error(f"Weights file not found: {self.weights_path}")
                raise ModelLoadError(f"Weights file not found: {self.weights_path}")
            
            if not self.weights_path:
                raise ModelLoadError("Weights path is required but not provided")
            
            logger.info(f"Using weights from: {self.weights_path}")
            
            # Load model weights
            logger.debug("Loading DDColor model weights...")
            try:
                # Load state dict with safety checks
                state_dict = torch.load(
                    self.weights_path,
                    map_location=self.device,
                    weights_only=False  # DDColor may have complex state
                )
                
                # Create DDColor model architecture
                # Note: This is a placeholder - actual DDColor model initialization
                # would require the DDColor library or custom implementation
                logger.debug("Creating DDColor model architecture...")
                
                # For now, we'll create a placeholder that will be replaced
                # with actual DDColor model when integrated
                self.model = self._create_ddcolor_model()
                
                # Load weights into model
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    self.model.load_state_dict(state_dict['state_dict'])
                elif isinstance(state_dict, dict):
                    self.model.load_state_dict(state_dict)
                else:
                    raise ModelLoadError("Invalid model weights format")
                
                # Move to device and set to eval mode
                self.model = self.model.to(self.device)
                self.model.eval()
                
                # Convert to FP16 if requested
                if self.use_fp16:
                    logger.debug("Converting model to FP16")
                    self.model = self.model.half()
                
            except Exception as e:
                logger.error(f"Failed to load DDColor model: {str(e)}", exc_info=True)
                raise ModelLoadError(f"Failed to load DDColor model: {str(e)}") from e
            
            # Log memory after loading
            MemoryManager.log_memory_usage(logger, "After loading Colorization model:")
            logger.info("DDColor model loaded successfully")
            
        except ModelLoadError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading Colorization model: {str(e)}", exc_info=True)
            raise ModelLoadError(f"Failed to load Colorization model: {str(e)}") from e
    
    def _create_ddcolor_model(self):
        """
        Create DDColor model architecture.
        
        Returns:
            DDColor model instance
            
        Note:
            This is a placeholder that should be replaced with actual DDColor
            model initialization when the DDColor library is integrated.
        """
        # Placeholder: In production, this would import and create the actual DDColor model
        # from ddcolor import DDColor
        # return DDColor(...)
        
        # For now, return a simple placeholder model
        import torch.nn as nn
        
        class DDColorPlaceholder(nn.Module):
            """Placeholder DDColor model for testing"""
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 2, 3, padding=1)
            
            def forward(self, l_channel):
                return self.conv(l_channel)
        
        return DDColorPlaceholder()
    
    def is_grayscale(self, image: np.ndarray, tolerance: int = 5) -> bool:
        """
        Detect if image is grayscale.
        
        Checks if all RGB channels are equal within tolerance.
        
        Args:
            image: Input image as numpy array (H, W, 3) with values in [0, 1]
            tolerance: Maximum difference between channels (in 0-255 scale)
            
        Returns:
            True if image is grayscale, False otherwise
        """
        # Convert to 0-255 scale for tolerance check
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Check if R == G == B within tolerance
        r_channel = image_uint8[:, :, 0]
        g_channel = image_uint8[:, :, 1]
        b_channel = image_uint8[:, :, 2]
        
        # Calculate max difference between channels
        rg_diff = np.abs(r_channel.astype(np.int16) - g_channel.astype(np.int16))
        rb_diff = np.abs(r_channel.astype(np.int16) - b_channel.astype(np.int16))
        gb_diff = np.abs(g_channel.astype(np.int16) - b_channel.astype(np.int16))
        
        max_diff = np.maximum(np.maximum(rg_diff, rb_diff), gb_diff)
        
        # Image is grayscale if max difference is within tolerance for most pixels
        # Use 95% threshold to allow for some compression artifacts
        grayscale_pixels = np.sum(max_diff <= tolerance)
        total_pixels = image.shape[0] * image.shape[1]
        
        is_gray = (grayscale_pixels / total_pixels) >= 0.95
        
        logger.debug(f"Grayscale detection: {grayscale_pixels}/{total_pixels} pixels within tolerance ({is_gray})")
        
        return is_gray
    
    def colorize(self, image: np.ndarray) -> np.ndarray:
        """
        Colorize grayscale image using DDColor.
        
        Automatically detects if image is grayscale and applies colorization.
        Color images are returned unchanged.
        
        Args:
            image: Input image as numpy array (H, W, 3) with values in [0, 1]
                   Expected in RGB format
            
        Returns:
            Colorized image as numpy array (H, W, 3) with values in [0, 1]
            
        Raises:
            ColorizationError: If image format is invalid or colorization fails
        """
        try:
            # Validate input
            if image.ndim != 3 or image.shape[2] != 3:
                raise ColorizationError("Image must have 3 channels (RGB) for colorization")
            
            if image.size == 0:
                raise ColorizationError("Cannot colorize empty image")
            
            # Check if image is grayscale
            if not self.is_grayscale(image):
                logger.info("Image is already in color, skipping colorization")
                return image
            
            logger.info(f"Starting colorization: {image.shape[1]}x{image.shape[0]}")
            
            # Load model if not already loaded
            if self.model is None:
                self.load_model()
            
            # Log memory before colorization
            MemoryManager.log_memory_usage(logger, "Before colorization:")
            
            # Convert RGB to Lab color space
            logger.debug("Converting RGB to Lab color space")
            lab_image = self._rgb_to_lab(image)
            
            # Extract L channel
            l_channel = lab_image[:, :, 0:1]  # Shape: (H, W, 1)
            
            # Prepare L channel for model input
            l_tensor = torch.from_numpy(l_channel).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 1, H, W)
            l_tensor = l_tensor.to(self.device)
            
            if self.use_fp16:
                l_tensor = l_tensor.half()
            else:
                l_tensor = l_tensor.float()
            
            # Run model inference
            logger.debug("Running DDColor inference...")
            try:
                with torch.no_grad():
                    ab_pred = self.model(l_tensor)  # Predict ab channels
                    
                    # Convert back to float32 and numpy
                    ab_pred = ab_pred.float().squeeze(0).permute(1, 2, 0).cpu().numpy()
                    
            except RuntimeError as e:
                error_msg = str(e).lower()
                if 'out of memory' in error_msg or 'cuda' in error_msg:
                    logger.error(f"GPU out of memory during colorization: {str(e)}", exc_info=True)
                    raise ColorizationError(
                        f"GPU out of memory during colorization. "
                        f"Try processing a smaller image. Error: {str(e)}"
                    ) from e
                else:
                    raise ColorizationError(f"Colorization inference failed: {str(e)}") from e
            
            # Combine L and predicted ab channels
            logger.debug("Combining L and ab channels")
            lab_colorized = np.concatenate([l_channel, ab_pred], axis=2)
            
            # Convert Lab back to RGB
            logger.debug("Converting Lab to RGB color space")
            rgb_colorized = self._lab_to_rgb(lab_colorized)
            
            # Validate output
            if rgb_colorized is None or rgb_colorized.size == 0:
                raise ColorizationError("Colorization produced empty output")
            
            # Ensure output is in valid range [0, 1]
            rgb_colorized = np.clip(rgb_colorized, 0.0, 1.0)
            
            # Log memory after colorization
            MemoryManager.log_memory_usage(logger, "After colorization:")
            logger.info(f"Colorization complete: output shape {rgb_colorized.shape}")
            
            return rgb_colorized
            
        except ColorizationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during colorization: {str(e)}", exc_info=True)
            raise ColorizationError(f"Colorization failed: {str(e)}") from e
    
    def _rgb_to_lab(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to Lab color space.
        
        Args:
            rgb_image: RGB image (H, W, 3) with values in [0, 1]
            
        Returns:
            Lab image (H, W, 3) with L in [0, 100], a in [-128, 127], b in [-128, 127]
        """
        try:
            from skimage import color
            
            # skimage expects RGB in [0, 1] range
            lab_image = color.rgb2lab(rgb_image)
            
            return lab_image
            
        except ImportError:
            # Fallback: simple approximation if skimage not available
            logger.warning("scikit-image not available, using simplified RGB to Lab conversion")
            
            # This is a simplified conversion - in production, use skimage
            # For grayscale images, L channel is just the luminance
            r, g, b = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
            l_channel = 0.299 * r + 0.587 * g + 0.114 * b
            l_channel = l_channel * 100  # Scale to [0, 100]
            
            # For grayscale, a and b should be near 0
            a_channel = np.zeros_like(l_channel)
            b_channel = np.zeros_like(l_channel)
            
            return np.stack([l_channel, a_channel, b_channel], axis=2)
    
    def _lab_to_rgb(self, lab_image: np.ndarray) -> np.ndarray:
        """
        Convert Lab image to RGB color space.
        
        Args:
            lab_image: Lab image (H, W, 3) with L in [0, 100], a in [-128, 127], b in [-128, 127]
            
        Returns:
            RGB image (H, W, 3) with values in [0, 1]
        """
        try:
            from skimage import color
            
            # skimage returns RGB in [0, 1] range
            rgb_image = color.lab2rgb(lab_image)
            
            return rgb_image
            
        except ImportError:
            # Fallback: simple approximation if skimage not available
            logger.warning("scikit-image not available, using simplified Lab to RGB conversion")
            
            # This is a simplified conversion - in production, use skimage
            # Just use L channel as grayscale approximation
            l_channel = lab_image[:, :, 0] / 100.0  # Scale back to [0, 1]
            
            # Create RGB by repeating L channel
            rgb_image = np.stack([l_channel, l_channel, l_channel], axis=2)
            
            return rgb_image
    
    def unload_model(self):
        """
        Unload model and clear GPU memory.
        
        Useful for freeing memory after processing is complete.
        """
        logger.info("Unloading Colorization model")
        MemoryManager.log_memory_usage(logger, "Before unloading:")
        
        self.model = None
        
        # Use MemoryManager for consistent memory cleanup
        MemoryManager.clear_cache()
        
        MemoryManager.log_memory_usage(logger, "After unloading and cache clear:")
