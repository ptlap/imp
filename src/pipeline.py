"""
Pipeline orchestrator for old photo restoration.

Main entry point for the IMP system that coordinates preprocessing,
denoising, and super-resolution modules with error handling, checkpointing,
and memory management.
"""

from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import cv2
from tqdm import tqdm

from .config import Config
from .utils.preprocessing import Preprocessor
from .utils.checkpoint import CheckpointManager
from .utils.memory import MemoryManager
from .utils.logging import setup_logger
from .models.denoiser import create_denoiser, DenoisingModule
from .models.super_resolution import SuperResolutionModule
from .utils.exceptions import IMPError, ProcessingError, ModelLoadError, ConfigurationError


class OldPhotoRestoration:
    """
    Main pipeline orchestrator for old photo restoration.
    
    Coordinates preprocessing, denoising, and super-resolution modules
    with lazy model loading, checkpointing, and memory management.
    Supports both single image and batch processing.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize restoration pipeline.
        
        Args:
            config: Configuration object. If None, uses default configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            self.config = config or Config.default()
            
            # Validate configuration
            self.config.validate()
            
            # Setup logger using centralized logging utility
            self.logger = setup_logger(
                name='imp',
                level=self.config.logging.level,
                log_file=self.config.logging.file,
                console_output=True
            )
            self.logger.info("Initializing OldPhotoRestoration pipeline")
            
            # Initialize preprocessor with logger
            self.preprocessor = Preprocessor(
                max_size=self.config.processing.max_image_size,
                logger=self.logger
            )
            self.logger.info(f"Preprocessor initialized with max_size={self.config.processing.max_image_size}")
            
            # Initialize checkpoint manager
            self.checkpoint_mgr = CheckpointManager(self.config.processing.checkpoint_dir)
            self.logger.info(f"Checkpoint manager initialized at {self.config.processing.checkpoint_dir}")
            
            # Lazy loading placeholders for models
            self.denoiser: Optional[DenoisingModule] = None
            self.super_resolver: Optional[SuperResolutionModule] = None
            
            self.logger.info("Pipeline initialization complete")
            
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize pipeline: {str(e)}") from e

    def _load_denoiser(self):
        """
        Lazy load denoising model.
        
        Loads the denoiser only when needed based on configuration.
        Uses factory pattern to create appropriate denoiser type.
        
        Raises:
            ModelLoadError: If denoiser loading fails
            ConfigurationError: If denoiser configuration is invalid
        """
        try:
            if self.denoiser is not None:
                return  # Already loaded
            
            self.logger.info(f"Loading denoiser: {self.config.models.denoising.type}")
            
            self.denoiser = create_denoiser(
                denoiser_type=self.config.models.denoising.type,
                strength=self.config.models.denoising.strength
            )
            
            self.logger.info("Denoiser loaded successfully")
            
        except (ModelLoadError, ConfigurationError):
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading denoiser: {str(e)}", exc_info=True)
            raise ModelLoadError(f"Failed to load denoiser: {str(e)}") from e
    
    def _load_super_resolver(self):
        """
        Lazy load super-resolution model.
        
        Loads the super-resolution model only when needed based on configuration.
        Downloads weights if necessary.
        
        Raises:
            ModelLoadError: If model loading or weight download fails
            ConfigurationError: If super-resolution configuration is invalid
        """
        try:
            if self.super_resolver is not None:
                return  # Already loaded
            
            self.logger.info(f"Loading super-resolution model: {self.config.models.super_resolution.type}")
            
            # Determine weights path
            weights_path = None
            if self.config.models.super_resolution.type == 'realesrgan':
                weights_dir = Path("weights")
                try:
                    weights_dir.mkdir(exist_ok=True)
                except Exception as e:
                    raise ModelLoadError(f"Failed to create weights directory: {str(e)}") from e
                
                # Determine model name based on scale
                if self.config.models.super_resolution.scale == 4:
                    model_name = 'realesrgan-x4plus'
                elif self.config.models.super_resolution.scale == 2:
                    model_name = 'realesrgan-x2plus'
                else:
                    model_name = 'realesrgan-x4plus'  # Default to 4x
                
                weights_path = weights_dir / f"{model_name}.pth"
                
                # Download weights if not present
                if not weights_path.exists():
                    self.logger.info(f"Weights not found, downloading to {weights_path}")
                    try:
                        from .utils.weight_downloader import WeightDownloader
                        downloader = WeightDownloader(weights_dir=str(weights_dir))
                        downloaded_path = downloader.download(model_name=model_name, save_path=str(weights_path))
                        if not downloaded_path:
                            raise ModelLoadError("Failed to download super-resolution weights from all sources")
                        self.logger.info("Weights downloaded successfully")
                    except Exception as e:
                        if isinstance(e, ModelLoadError):
                            raise
                        raise ModelLoadError(f"Failed to download weights: {str(e)}") from e
            
            self.super_resolver = SuperResolutionModule(
                scale=self.config.models.super_resolution.scale,
                weights_path=str(weights_path) if weights_path else None,
                tile_size=self.config.models.super_resolution.tile_size,
                tile_overlap=self.config.models.super_resolution.tile_overlap,
                use_fp16=self.config.models.super_resolution.use_fp16
            )
            
            # Load the model
            self.super_resolver.load_model()
            
            self.logger.info("Super-resolution model loaded successfully")
            
        except (ModelLoadError, ConfigurationError):
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading super-resolution model: {str(e)}", exc_info=True)
            raise ModelLoadError(f"Failed to load super-resolution model: {str(e)}") from e
    
    def _unload_models(self):
        """
        Unload models and clear GPU memory.
        
        Frees up memory by unloading models and clearing GPU cache.
        Safe to call even if models are not loaded.
        """
        self.logger.info("Unloading models and clearing memory")
        
        # Unload denoiser if loaded
        if self.denoiser is not None:
            # NAFNet has unload method, OpenCV doesn't need it
            if hasattr(self.denoiser, 'unload_model'):
                self.denoiser.unload_model()
            self.denoiser = None
        
        # Unload super-resolver if loaded
        if self.super_resolver is not None:
            self.super_resolver.unload_model()
            self.super_resolver = None
        
        # Clear GPU cache
        MemoryManager.clear_cache()
        MemoryManager.log_memory_usage(self.logger, "After unloading models:")

    def restore(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        resume: bool = True
    ) -> np.ndarray:
        """
        Restore old photo through complete processing pipeline.
        
        Executes preprocessing, denoising, and super-resolution in sequence
        with checkpoint support for resume functionality.
        
        Args:
            image_path: Path to input image file
            output_path: Optional path to save restored image
            resume: If True, resume from checkpoint if available
            
        Returns:
            Restored image as numpy array (H, W, 3) with values in [0, 1]
            
        Raises:
            FileNotFoundError: If input image doesn't exist
            ValueError: If image format is invalid
            RuntimeError: If processing fails
        """
        self.logger.info(f"Starting restoration for: {image_path}")
        
        # Generate image ID for checkpoints
        image_id = Path(image_path).stem
        
        try:
            # Step 1: Preprocessing
            self.logger.info("Step 1: Preprocessing")
            checkpoint_name = f"{image_id}_preprocessed"
            
            if resume and self.config.processing.checkpoint_enabled and self.checkpoint_mgr.has(checkpoint_name):
                self.logger.info(f"Loading from checkpoint: {checkpoint_name}")
                image, metadata = self.checkpoint_mgr.load(checkpoint_name)
            else:
                image, metadata = self.preprocessor.process(image_path)
                self.logger.info(f"Preprocessing complete - Size: {image.shape}, Grayscale: {metadata['is_grayscale']}")
                
                if self.config.processing.checkpoint_enabled:
                    self.checkpoint_mgr.save(image, checkpoint_name, metadata)
                    self.logger.info(f"Checkpoint saved: {checkpoint_name}")
            
            # Step 2: Denoising
            if not self.config.models.denoising.skip:
                self.logger.info("Step 2: Denoising")
                checkpoint_name = f"{image_id}_denoised"
                
                if resume and self.config.processing.checkpoint_enabled and self.checkpoint_mgr.has(checkpoint_name):
                    self.logger.info(f"Loading from checkpoint: {checkpoint_name}")
                    image, _ = self.checkpoint_mgr.load(checkpoint_name)
                else:
                    self._load_denoiser()
                    image = self.denoiser.denoise(image)
                    self.logger.info("Denoising complete")
                    
                    # Unload denoiser to free memory
                    self._unload_models()
                    
                    if self.config.processing.checkpoint_enabled:
                        self.checkpoint_mgr.save(image, checkpoint_name)
                        self.logger.info(f"Checkpoint saved: {checkpoint_name}")
            else:
                self.logger.info("Step 2: Denoising skipped (disabled in config)")
            
            # Step 3: Super-resolution
            if not self.config.models.super_resolution.skip:
                self.logger.info("Step 3: Super-resolution")
                checkpoint_name = f"{image_id}_sr"
                
                if resume and self.config.processing.checkpoint_enabled and self.checkpoint_mgr.has(checkpoint_name):
                    self.logger.info(f"Loading from checkpoint: {checkpoint_name}")
                    image, _ = self.checkpoint_mgr.load(checkpoint_name)
                else:
                    self._load_super_resolver()
                    image = self.super_resolver.upscale(image)
                    self.logger.info(f"Super-resolution complete - New size: {image.shape}")
                    
                    # Unload super-resolver to free memory
                    self._unload_models()
                    
                    if self.config.processing.checkpoint_enabled:
                        self.checkpoint_mgr.save(image, checkpoint_name)
                        self.logger.info(f"Checkpoint saved: {checkpoint_name}")
            else:
                self.logger.info("Step 3: Super-resolution skipped (disabled in config)")
            
            # Save result if output path provided
            if output_path:
                # Convert from [0, 1] to [0, 255] and save
                image_uint8 = (image * 255).astype(np.uint8)
                # Convert RGB to BGR for OpenCV
                image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
                
                # Create output directory if needed
                output_dir = Path(output_path).parent
                output_dir.mkdir(parents=True, exist_ok=True)
                
                cv2.imwrite(output_path, image_bgr)
                self.logger.info(f"Result saved to: {output_path}")
            
            self.logger.info(f"Restoration complete for: {image_path}")
            return image
            
        except IMPError:
            # Re-raise IMP-specific errors as-is
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during restoration of {image_path}: {str(e)}", exc_info=True)
            raise ProcessingError(f"Restoration failed: {str(e)}") from e

    def batch_restore(
        self,
        image_paths: List[str],
        output_dir: str,
        max_retries: int = 2
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Restore multiple images in batch with progress tracking.
        
        Processes multiple images sequentially with retry logic for failures.
        Skips already processed images and continues on errors.
        
        Args:
            image_paths: List of input image file paths
            output_dir: Directory to save restored images
            max_retries: Maximum number of retry attempts for failed images
            
        Returns:
            Tuple of (successes, failures)
            - successes: List of dicts with 'input_path' and 'output_path'
            - failures: List of dicts with 'input_path' and 'error'
        """
        self.logger.info(f"Starting batch restoration of {len(image_paths)} images")
        self.logger.info(f"Output directory: {output_dir}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        successes = []
        failures = []
        
        # Track retry attempts
        retry_counts = {path: 0 for path in image_paths}
        
        # Process images with progress bar
        with tqdm(total=len(image_paths), desc="Restoring images") as pbar:
            for image_path in image_paths:
                try:
                    # Generate output path
                    input_filename = Path(image_path).stem
                    output_filename = f"{input_filename}_restored.png"
                    output_file = output_path / output_filename
                    
                    # Skip if already processed
                    if output_file.exists():
                        self.logger.info(f"Skipping already processed: {image_path}")
                        successes.append({
                            'input_path': image_path,
                            'output_path': str(output_file),
                            'skipped': True
                        })
                        pbar.update(1)
                        continue
                    
                    # Attempt restoration
                    self.restore(image_path, str(output_file), resume=True)
                    
                    successes.append({
                        'input_path': image_path,
                        'output_path': str(output_file),
                        'skipped': False
                    })
                    
                    pbar.update(1)
                    
                except Exception as e:
                    retry_counts[image_path] += 1
                    
                    # Retry if under max_retries
                    if retry_counts[image_path] < max_retries:
                        self.logger.warning(
                            f"Failed to process {image_path} (attempt {retry_counts[image_path]}/{max_retries}): {str(e)}"
                        )
                        self.logger.info(f"Retrying {image_path}...")
                        
                        # Clear any partial checkpoints
                        image_id = Path(image_path).stem
                        for suffix in ['preprocessed', 'denoised', 'sr']:
                            checkpoint_name = f"{image_id}_{suffix}"
                            if self.checkpoint_mgr.has(checkpoint_name):
                                try:
                                    checkpoint_path = self.checkpoint_mgr.checkpoint_dir / f"{checkpoint_name}.pkl"
                                    checkpoint_path.unlink()
                                except Exception:
                                    pass
                        
                        # Don't update progress bar, will retry
                        continue
                    else:
                        # Max retries reached
                        self.logger.error(
                            f"Failed to process {image_path} after {max_retries} attempts: {str(e)}"
                        )
                        failures.append({
                            'input_path': image_path,
                            'error': str(e),
                            'attempts': retry_counts[image_path]
                        })
                        pbar.update(1)
        
        # Log summary statistics
        self.logger.info("=" * 60)
        self.logger.info("Batch processing complete")
        self.logger.info(f"Total images: {len(image_paths)}")
        self.logger.info(f"Successful: {len(successes)}")
        self.logger.info(f"Failed: {len(failures)}")
        
        if failures:
            self.logger.info("Failed images:")
            for failure in failures:
                self.logger.info(f"  - {failure['input_path']}: {failure['error']}")
        
        self.logger.info("=" * 60)
        
        return successes, failures
