"""Configuration management for IMP system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml

from .utils.exceptions import ConfigurationError


@dataclass
class DenoisingConfig:
    """Configuration for denoising module."""
    type: str = "opencv"  # "opencv" or "nafnet"
    strength: int = 10
    skip: bool = False


@dataclass
class SuperResolutionConfig:
    """Configuration for super-resolution module."""
    type: str = "realesrgan"
    scale: int = 4
    tile_size: int = 512
    tile_overlap: int = 64
    use_fp16: bool = True
    weights_url: str = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-x4plus.pth"
    skip: bool = False


@dataclass
class ModelsConfig:
    """Configuration for all models."""
    denoising: DenoisingConfig = field(default_factory=DenoisingConfig)
    super_resolution: SuperResolutionConfig = field(default_factory=SuperResolutionConfig)


@dataclass
class ProcessingConfig:
    """Configuration for processing parameters."""
    max_image_size: int = 2048
    save_intermediate: bool = False
    checkpoint_enabled: bool = True
    checkpoint_dir: str = "./checkpoints"


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    file: str = "imp.log"


@dataclass
class Config:
    """Main configuration container for IMP system."""
    models: ModelsConfig = field(default_factory=ModelsConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            Config instance with loaded values
            
        Raises:
            ConfigurationError: If config file doesn't exist, parsing fails, or validation fails
        """
        try:
            config_path = Path(path)
            if not config_path.exists():
                raise ConfigurationError(f"Configuration file not found: {path}")
            
            try:
                with open(config_path, 'r') as f:
                    data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ConfigurationError(f"Failed to parse YAML configuration: {e}") from e
            except Exception as e:
                raise ConfigurationError(f"Failed to read configuration file: {e}") from e
            
            if data is None:
                data = {}
            
            # Parse nested configurations with error handling
            try:
                models_data = data.get('models', {})
                denoising_data = models_data.get('denoising', {})
                super_resolution_data = models_data.get('super_resolution', {})
                processing_data = data.get('processing', {})
                logging_data = data.get('logging', {})
                
                # Create config objects
                denoising_config = DenoisingConfig(**denoising_data)
                super_resolution_config = SuperResolutionConfig(**super_resolution_data)
                models_config = ModelsConfig(
                    denoising=denoising_config,
                    super_resolution=super_resolution_config
                )
                processing_config = ProcessingConfig(**processing_data)
                logging_config = LoggingConfig(**logging_data)
                
                config = cls(
                    models=models_config,
                    processing=processing_config,
                    logging=logging_config
                )
            except TypeError as e:
                raise ConfigurationError(f"Invalid configuration structure: {e}") from e
            
            # Validate configuration
            config.validate()
            
            return config
            
        except ConfigurationError:
            # Re-raise ConfigurationError as-is
            raise
        except Exception as e:
            # Wrap any other unexpected errors
            raise ConfigurationError(f"Unexpected error loading configuration: {e}") from e

    @classmethod
    def default(cls) -> 'Config':
        """
        Create configuration with default values.
        
        Returns:
            Config instance with default values
        """
        return cls()

    def validate(self) -> bool:
        """
        Validate configuration values.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If any configuration value is invalid
        """
        errors = []
        
        # Validate denoising config
        if self.models.denoising.type not in ["opencv", "nafnet"]:
            errors.append(f"Invalid denoising type: {self.models.denoising.type}. Must be 'opencv' or 'nafnet'")
        
        if self.models.denoising.strength < 1 or self.models.denoising.strength > 100:
            errors.append(f"Invalid denoising strength: {self.models.denoising.strength}. Must be between 1 and 100")
        
        # Validate super-resolution config
        if self.models.super_resolution.type not in ["realesrgan"]:
            errors.append(f"Invalid super-resolution type: {self.models.super_resolution.type}. Must be 'realesrgan'")
        
        if self.models.super_resolution.scale not in [2, 4]:
            errors.append(f"Invalid super-resolution scale: {self.models.super_resolution.scale}. Must be 2 or 4")
        
        if self.models.super_resolution.tile_size < 64 or self.models.super_resolution.tile_size > 2048:
            errors.append(f"Invalid tile_size: {self.models.super_resolution.tile_size}. Must be between 64 and 2048")
        
        if self.models.super_resolution.tile_overlap < 0 or self.models.super_resolution.tile_overlap >= self.models.super_resolution.tile_size:
            errors.append(f"Invalid tile_overlap: {self.models.super_resolution.tile_overlap}. Must be between 0 and tile_size")
        
        # Validate processing config
        if self.processing.max_image_size < 256 or self.processing.max_image_size > 8192:
            errors.append(f"Invalid max_image_size: {self.processing.max_image_size}. Must be between 256 and 8192")
        
        # Validate logging config
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging.level not in valid_log_levels:
            errors.append(f"Invalid logging level: {self.logging.level}. Must be one of {valid_log_levels}")
        
        if errors:
            raise ConfigurationError("\n".join(errors))
        
        return True



