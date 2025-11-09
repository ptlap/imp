"""
Post-processing module for final image enhancements.

Applies color correction, sharpening, and contrast enhancement
to improve the visual quality of restored images.
"""

import numpy as np
import cv2
import logging
from typing import Optional

from ..utils.exceptions import ProcessingError

logger = logging.getLogger(__name__)


class PostProcessingModule:
    """
    Post-processing module for final image enhancements.
    
    Applies a series of image enhancement operations including:
    - White balance correction
    - Auto levels (histogram stretching)
    - Unsharp masking (sharpening)
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Optional final denoising
    
    All operations work on images with values in [0, 1] range.
    """
    
    def __init__(
        self,
        white_balance: bool = True,
        auto_levels: bool = True,
        sharpen_strength: float = 1.0,
        enhance_contrast: bool = True,
        final_denoise: bool = False
    ):
        """
        Initialize post-processing module.
        
        Args:
            white_balance: Apply white balance correction
            auto_levels: Apply histogram stretching
            sharpen_strength: Strength of unsharp masking (0.0 to 5.0)
            enhance_contrast: Apply CLAHE contrast enhancement
            final_denoise: Apply light denoising to smooth artifacts
            
        Raises:
            ValueError: If sharpen_strength is out of valid range
        """
        if sharpen_strength < 0.0 or sharpen_strength > 5.0:
            raise ValueError(f"sharpen_strength must be between 0.0 and 5.0, got {sharpen_strength}")
        
        self.white_balance = white_balance
        self.auto_levels = auto_levels
        self.sharpen_strength = sharpen_strength
        self.enhance_contrast = enhance_contrast
        self.final_denoise = final_denoise
        
        logger.info(
            f"PostProcessingModule initialized: "
            f"white_balance={white_balance}, auto_levels={auto_levels}, "
            f"sharpen_strength={sharpen_strength}, enhance_contrast={enhance_contrast}, "
            f"final_denoise={final_denoise}"
        )
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply post-processing operations to image.
        
        Operations are applied in sequence:
        1. White balance correction (if enabled)
        2. Auto levels (if enabled)
        3. Unsharp masking (if strength > 0)
        4. CLAHE contrast enhancement (if enabled)
        5. Final denoising (if enabled)
        
        Args:
            image: Input image (H, W, 3) with values in [0, 1]
            
        Returns:
            Processed image (H, W, 3) with values in [0, 1]
            
        Raises:
            ProcessingError: If image format is invalid or processing fails
        """
        try:
            # Validate input
            if not isinstance(image, np.ndarray):
                raise ProcessingError("Image must be a numpy array")
            
            if image.ndim != 3:
                raise ProcessingError(f"Image must have 3 dimensions, got {image.ndim}")
            
            if image.shape[2] != 3:
                raise ProcessingError(f"Image must have 3 channels (RGB), got {image.shape[2]}")
            
            if image.size == 0:
                raise ProcessingError("Cannot process empty image")
            
            if image.dtype != np.float32 and image.dtype != np.float64:
                raise ProcessingError(f"Image must be float type, got {image.dtype}")
            
            logger.info(f"Starting post-processing on image shape {image.shape}")
            
            # Work on a copy to avoid modifying input
            result = image.copy()
            
            # 1. White balance
            if self.white_balance:
                logger.debug("Applying white balance correction")
                result = self._white_balance(result)
            
            # 2. Auto levels
            if self.auto_levels:
                logger.debug("Applying auto levels")
                result = self._auto_levels(result)
            
            # 3. Sharpen
            if self.sharpen_strength > 0:
                logger.debug(f"Applying unsharp mask with strength {self.sharpen_strength}")
                result = self._unsharp_mask(result)
            
            # 4. Contrast enhancement
            if self.enhance_contrast:
                logger.debug("Applying CLAHE contrast enhancement")
                result = self._enhance_contrast(result)
            
            # 5. Final denoise
            if self.final_denoise:
                logger.debug("Applying final denoising")
                result = self._denoise_final(result)
            
            logger.info("Post-processing complete")
            return result
            
        except ProcessingError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during post-processing: {str(e)}", exc_info=True)
            raise ProcessingError(f"Post-processing failed: {str(e)}") from e
    
    def _white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply white balance correction using Lab color space.
        
        Adjusts a and b channels toward neutral (128) to correct color cast.
        
        Args:
            image: Input image (H, W, 3) with values in [0, 1]
            
        Returns:
            White balanced image (H, W, 3) with values in [0, 1]
        """
        try:
            # Convert to uint8 for OpenCV
            image_uint8 = (image * 255).astype(np.uint8)
            
            # Convert RGB to Lab color space
            lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
            
            # Split channels
            l, a, b = cv2.split(lab)
            
            # Calculate average a and b channel values
            avg_a = np.average(a)
            avg_b = np.average(b)
            
            # Adjust a and b channels toward neutral (128) with 0.8 factor
            # This corrects color cast while preserving some of the original tone
            a = a.astype(np.float32) - ((avg_a - 128) * 0.8)
            b = b.astype(np.float32) - ((avg_b - 128) * 0.8)
            
            # Clip to valid range [0, 255]
            a = np.clip(a, 0, 255).astype(np.uint8)
            b = np.clip(b, 0, 255).astype(np.uint8)
            
            # Merge channels back
            lab = cv2.merge([l, a, b])
            
            # Convert back to RGB
            result_uint8 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Convert back to float [0, 1]
            result = result_uint8.astype(np.float32) / 255.0
            
            return result
            
        except Exception as e:
            logger.warning(f"White balance failed, returning original image: {str(e)}")
            return image
    
    def _auto_levels(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram stretching for contrast optimization.
        
        Stretches histogram to use full [0, 1] range based on 1% and 99% percentiles.
        
        Args:
            image: Input image (H, W, 3) with values in [0, 1]
            
        Returns:
            Auto-leveled image (H, W, 3) with values in [0, 1]
        """
        try:
            result = image.copy()
            
            # Process each channel independently
            for i in range(3):
                channel = result[:, :, i]
                
                # Compute histogram
                hist, bins = np.histogram(channel.flatten(), 256, [0, 1])
                
                # Compute cumulative distribution function
                cdf = hist.cumsum()
                cdf_normalized = cdf / cdf[-1]
                
                # Find 1% and 99% percentiles
                low = np.searchsorted(cdf_normalized, 0.01)
                high = np.searchsorted(cdf_normalized, 0.99)
                
                # Convert to [0, 1] range
                low = low / 255.0
                high = high / 255.0
                
                # Avoid division by zero
                if high - low < 1e-6:
                    continue
                
                # Stretch histogram
                channel = (channel - low) / (high - low)
                
                # Clip to [0, 1] range
                channel = np.clip(channel, 0, 1)
                
                result[:, :, i] = channel
            
            return result
            
        except Exception as e:
            logger.warning(f"Auto levels failed, returning original image: {str(e)}")
            return image
    
    def _unsharp_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Apply unsharp masking for edge enhancement.
        
        Sharpens image by subtracting blurred version:
        sharpened = original + strength * (original - blurred)
        
        Args:
            image: Input image (H, W, 3) with values in [0, 1]
            
        Returns:
            Sharpened image (H, W, 3) with values in [0, 1]
        """
        try:
            # Gaussian blur with sigma=1.0
            sigma = 1.0
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            
            # Compute sharpened = original + strength * (original - blurred)
            sharpened = image + self.sharpen_strength * (image - blurred)
            
            # Clip to [0, 1] range
            sharpened = np.clip(sharpened, 0, 1)
            
            return sharpened
            
        except Exception as e:
            logger.warning(f"Unsharp mask failed, returning original image: {str(e)}")
            return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE for local contrast enhancement.
        
        Applies Contrast Limited Adaptive Histogram Equalization
        to L channel in Lab color space.
        
        Args:
            image: Input image (H, W, 3) with values in [0, 1]
            
        Returns:
            Contrast enhanced image (H, W, 3) with values in [0, 1]
        """
        try:
            # Convert to uint8 for OpenCV
            image_uint8 = (image * 255).astype(np.uint8)
            
            # Convert RGB to Lab color space
            lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
            
            # Split channels
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel only
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels back
            lab = cv2.merge([l, a, b])
            
            # Convert back to RGB
            result_uint8 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Convert back to float [0, 1]
            result = result_uint8.astype(np.float32) / 255.0
            
            return result
            
        except Exception as e:
            logger.warning(f"CLAHE contrast enhancement failed, returning original image: {str(e)}")
            return image
    
    def _denoise_final(self, image: np.ndarray) -> np.ndarray:
        """
        Apply light denoising to smooth artifacts.
        
        Uses fastNlMeansDenoisingColored with low strength to
        reduce minor artifacts without losing details.
        
        Args:
            image: Input image (H, W, 3) with values in [0, 1]
            
        Returns:
            Denoised image (H, W, 3) with values in [0, 1]
        """
        try:
            # Convert to uint8 for OpenCV
            image_uint8 = (image * 255).astype(np.uint8)
            
            # Apply light denoising with low strength (h=3)
            denoised = cv2.fastNlMeansDenoisingColored(
                image_uint8,
                None,
                h=3,  # Low strength for luminance
                hColor=3,  # Low strength for color
                templateWindowSize=7,
                searchWindowSize=21
            )
            
            # Convert back to float [0, 1]
            result = denoised.astype(np.float32) / 255.0
            
            return result
            
        except Exception as e:
            logger.warning(f"Final denoising failed, returning original image: {str(e)}")
            return image
