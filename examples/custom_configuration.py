"""
Custom configuration example for IMP pipeline.

This script demonstrates different ways to configure the restoration pipeline
including loading from YAML, creating custom configs, and modifying settings.
"""

from pathlib import Path
from src.pipeline import OldPhotoRestoration
from src.config import Config, DenoisingConfig, SuperResolutionConfig, ModelsConfig, ProcessingConfig, LoggingConfig


def example_default_config():
    """Use default configuration."""
    print("=" * 60)
    print("Example 1: Default Configuration")
    print("=" * 60)
    
    # Create pipeline with default config
    pipeline = OldPhotoRestoration()
    
    # Print configuration
    config = pipeline.config
    print("\nDefault Configuration:")
    print(f"  Denoising:")
    print(f"    Type: {config.models.denoising.type}")
    print(f"    Strength: {config.models.denoising.strength}")
    print(f"  Super-Resolution:")
    print(f"    Type: {config.models.super_resolution.type}")
    print(f"    Scale: {config.models.super_resolution.scale}x")
    print(f"    Tile size: {config.models.super_resolution.tile_size}")
    print(f"    FP16: {config.models.super_resolution.use_fp16}")
    print(f"  Processing:")
    print(f"    Max image size: {config.processing.max_image_size}")
    print(f"    Checkpoints: {config.processing.checkpoint_enabled}")
    print(f"  Logging:")
    print(f"    Level: {config.logging.level}")
    print(f"    File: {config.logging.file}")


def example_yaml_config():
    """Load configuration from YAML file."""
    print("\n" + "=" * 60)
    print("Example 2: Load from YAML")
    print("=" * 60)
    
    config_path = "configs/config.yaml"
    
    try:
        # Load config from YAML
        config = Config.from_yaml(config_path)
        pipeline = OldPhotoRestoration(config)
        
        print(f"\n✓ Configuration loaded from {config_path}")
        print(f"  Denoising: {config.models.denoising.type} (strength={config.models.denoising.strength})")
        print(f"  Super-resolution: {config.models.super_resolution.scale}x")
        
    except FileNotFoundError:
        print(f"\n✗ Config file not found: {config_path}")
        print("  Create a config.yaml file in the configs/ directory")
    except Exception as e:
        print(f"\n✗ Error loading config: {e}")


def example_custom_config_programmatic():
    """Create custom configuration programmatically."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Configuration (Programmatic)")
    print("=" * 60)
    
    # Create custom configuration from scratch
    denoising_config = DenoisingConfig(
        type="opencv",
        strength=20,  # Stronger denoising
        skip=False
    )
    
    super_resolution_config = SuperResolutionConfig(
        type="realesrgan",
        scale=2,  # 2x upscaling
        tile_size=256,  # Smaller tiles for memory efficiency
        tile_overlap=32,
        use_fp16=True,
        skip=False
    )
    
    models_config = ModelsConfig(
        denoising=denoising_config,
        super_resolution=super_resolution_config
    )
    
    processing_config = ProcessingConfig(
        max_image_size=1024,  # Smaller max size
        checkpoint_enabled=True,
        checkpoint_dir="./my_checkpoints"
    )
    
    logging_config = LoggingConfig(
        level="DEBUG",  # More verbose logging
        file="my_restoration.log"
    )
    
    config = Config(
        models=models_config,
        processing=processing_config,
        logging=logging_config
    )
    
    # Create pipeline with custom config
    pipeline = OldPhotoRestoration(config)
    
    print("\n✓ Custom configuration created:")
    print(f"  Denoising strength: {config.models.denoising.strength}")
    print(f"  SR scale: {config.models.super_resolution.scale}x")
    print(f"  Tile size: {config.models.super_resolution.tile_size}")
    print(f"  Max image size: {config.processing.max_image_size}")
    print(f"  Log level: {config.logging.level}")


def example_modify_default_config():
    """Modify default configuration."""
    print("\n" + "=" * 60)
    print("Example 4: Modify Default Configuration")
    print("=" * 60)
    
    # Start with default and modify
    config = Config.default()
    
    # Modify specific settings
    config.models.denoising.strength = 15
    config.models.super_resolution.scale = 2
    config.models.super_resolution.tile_size = 256
    config.processing.max_image_size = 1024
    config.logging.level = "DEBUG"
    
    # Validate configuration
    try:
        config.validate()
        print("\n✓ Configuration validated successfully")
    except Exception as e:
        print(f"\n✗ Configuration validation failed: {e}")
        return
    
    # Create pipeline
    pipeline = OldPhotoRestoration(config)
    
    print("\nModified configuration:")
    print(f"  Denoising strength: {config.models.denoising.strength}")
    print(f"  SR scale: {config.models.super_resolution.scale}x")
    print(f"  Tile size: {config.models.super_resolution.tile_size}")


def example_skip_modules():
    """Configure pipeline to skip specific modules."""
    print("\n" + "=" * 60)
    print("Example 5: Skip Specific Modules")
    print("=" * 60)
    
    # Example 5a: Denoising only
    config = Config.default()
    config.models.denoising.skip = False
    config.models.super_resolution.skip = True
    
    pipeline = OldPhotoRestoration(config)
    
    print("\nConfiguration A: Denoising only")
    print("  ✓ Preprocessing")
    print("  ✓ Denoising")
    print("  ✗ Super-resolution (skipped)")
    
    # Example 5b: Super-resolution only
    config = Config.default()
    config.models.denoising.skip = True
    config.models.super_resolution.skip = False
    
    pipeline = OldPhotoRestoration(config)
    
    print("\nConfiguration B: Super-resolution only")
    print("  ✓ Preprocessing")
    print("  ✗ Denoising (skipped)")
    print("  ✓ Super-resolution")


def example_memory_optimized_config():
    """Configuration optimized for low memory environments."""
    print("\n" + "=" * 60)
    print("Example 6: Memory-Optimized Configuration")
    print("=" * 60)
    
    config = Config.default()
    
    # Optimize for memory
    config.models.super_resolution.tile_size = 256  # Smaller tiles
    config.models.super_resolution.tile_overlap = 16  # Less overlap
    config.models.super_resolution.use_fp16 = True  # Use half precision
    config.processing.max_image_size = 1024  # Limit input size
    config.processing.checkpoint_enabled = True  # Enable checkpoints for resume
    
    pipeline = OldPhotoRestoration(config)
    
    print("\n✓ Memory-optimized configuration:")
    print(f"  Tile size: {config.models.super_resolution.tile_size} (smaller)")
    print(f"  Tile overlap: {config.models.super_resolution.tile_overlap} (reduced)")
    print(f"  FP16: {config.models.super_resolution.use_fp16} (enabled)")
    print(f"  Max image size: {config.processing.max_image_size}")
    print(f"  Checkpoints: {config.processing.checkpoint_enabled}")
    print("\nThis configuration uses less GPU memory but may be slower.")


def example_quality_optimized_config():
    """Configuration optimized for maximum quality."""
    print("\n" + "=" * 60)
    print("Example 7: Quality-Optimized Configuration")
    print("=" * 60)
    
    config = Config.default()
    
    # Optimize for quality
    config.models.denoising.strength = 8  # Gentler denoising
    config.models.super_resolution.scale = 4  # Maximum upscaling
    config.models.super_resolution.tile_size = 512  # Larger tiles
    config.models.super_resolution.tile_overlap = 64  # More overlap
    config.processing.max_image_size = 2048  # Allow larger inputs
    
    pipeline = OldPhotoRestoration(config)
    
    print("\n✓ Quality-optimized configuration:")
    print(f"  Denoising strength: {config.models.denoising.strength} (gentle)")
    print(f"  SR scale: {config.models.super_resolution.scale}x (maximum)")
    print(f"  Tile size: {config.models.super_resolution.tile_size} (larger)")
    print(f"  Tile overlap: {config.models.super_resolution.tile_overlap} (more)")
    print(f"  Max image size: {config.processing.max_image_size}")
    print("\nThis configuration prioritizes quality over speed and memory.")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("IMP Configuration Examples")
    print("=" * 60)
    
    # Run all examples
    example_default_config()
    example_yaml_config()
    example_custom_config_programmatic()
    example_modify_default_config()
    example_skip_modules()
    example_memory_optimized_config()
    example_quality_optimized_config()
    
    print("\n" + "=" * 60)
    print("Configuration Complete")
    print("=" * 60)
    print("\nYou can now use these configurations with:")
    print("  pipeline = OldPhotoRestoration(config)")
    print("  restored = pipeline.restore('input.jpg', 'output.png')")
    print("=" * 60)
