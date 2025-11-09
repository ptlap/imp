"""
Custom exception hierarchy for IMP system.

Provides specific exception types for different error scenarios
to enable better error handling and debugging.
"""


class IMPError(Exception):
    """
    Base exception for all IMP-related errors.
    
    All custom exceptions in the IMP system inherit from this base class,
    allowing for catching all IMP-specific errors with a single except clause.
    """
    pass


class ConfigurationError(IMPError):
    """
    Exception raised for configuration validation errors.
    
    Raised when:
    - Configuration file is invalid or malformed
    - Configuration values are out of valid range
    - Required configuration parameters are missing
    - Configuration validation fails
    
    Examples:
        - Invalid denoising type (not 'opencv' or 'nafnet')
        - Super-resolution scale not 2 or 4
        - Tile overlap >= tile size
        - Invalid log level
    """
    pass


class ModelLoadError(IMPError):
    """
    Exception raised when model loading fails.
    
    Raised when:
    - Model weights file not found
    - Model weights download fails
    - Model architecture initialization fails
    - Required libraries for model are not installed
    - GPU/CUDA not available when required
    
    Examples:
        - Real-ESRGAN weights file missing
        - NAFNet weights download timeout
        - basicsr or realesrgan library not installed
    """
    pass


class ProcessingError(IMPError):
    """
    Exception raised during image processing operations.
    
    Raised when:
    - Image file cannot be loaded
    - Image format is unsupported
    - Image processing step fails
    - Invalid image dimensions or format
    - Checkpoint save/load fails
    
    Examples:
        - Corrupted image file
        - Unsupported image format (.bmp, .tiff)
        - Image has wrong number of channels
        - Denoising or upscaling operation fails
    """
    pass


class OutOfMemoryError(IMPError):
    """
    Exception raised when GPU or system memory is exhausted.
    
    Raised when:
    - GPU memory allocation fails
    - System RAM is exhausted
    - Image is too large to process even with tiling
    - Model cannot fit in available memory
    
    Examples:
        - CUDA out of memory during super-resolution
        - Cannot allocate tensor for large image
        - Tiling strategy insufficient for image size
    """
    pass


class ColorizationError(ProcessingError):
    """
    Exception raised during colorization operations.
    
    Raised when:
    - Colorization model inference fails
    - Color space conversion fails
    - Invalid image format for colorization
    
    Examples:
        - DDColor model inference error
        - Lab color space conversion failure
        - GPU out of memory during colorization
    """
    pass


class FaceDetectionError(ProcessingError):
    """
    Exception raised during face detection operations.
    
    Raised when:
    - Face detection model inference fails
    - Invalid image format for face detection
    - Face region extraction fails
    
    Examples:
        - RetinaFace model inference error
        - Invalid bounding box coordinates
        - GPU out of memory during face detection
    """
    pass


class FaceEnhancementError(ProcessingError):
    """
    Exception raised during face enhancement operations.
    
    Raised when:
    - Face enhancement model inference fails
    - Face blending operation fails
    - Invalid face region format
    
    Examples:
        - CodeFormer model inference error
        - Poisson blending failure
        - GPU out of memory during face enhancement
    """
    pass
