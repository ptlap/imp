"""
Basic usage example for IMP pipeline.

This script demonstrates how to use the OldPhotoRestoration pipeline
for single image and batch processing.
"""

from pathlib import Path
from src.pipeline import OldPhotoRestoration
from src.config import Config


def example_single_image():
    """Example: Restore a single image"""
    print("=" * 60)
    print("Example 1: Single Image Restoration")
    print("=" * 60)
    
    # Create pipeline with default configuration
    pipeline = OldPhotoRestoration()
    
    # Restore single image
    input_path = "path/to/your/old_photo.jpg"
    output_path = "path/to/output/restored_photo.png"
    
    try:
        restored_image = pipeline.restore(
            image_path=input_path,
            output_path=output_path,
            resume=True  # Resume from checkpoint if available
        )
        print(f"✓ Image restored successfully!")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Shape: {restored_image.shape}")
    except Exception as e:
        print(f"✗ Restoration failed: {e}")


def example_single_image_with_custom_config():
    """Example: Restore image with custom configuration"""
    print("\n" + "=" * 60)
    print("Example 2: Single Image with Custom Config")
    print("=" * 60)
    
    # Create custom configuration
    config = Config.default()
    config.models.denoising.strength = 15  # Stronger denoising
    config.models.super_resolution.scale = 2  # 2x upscaling instead of 4x
    config.processing.checkpoint_enabled = True
    
    # Create pipeline with custom config
    pipeline = OldPhotoRestoration(config=config)
    
    input_path = "path/to/your/old_photo.jpg"
    output_path = "path/to/output/restored_photo_2x.png"
    
    try:
        restored_image = pipeline.restore(input_path, output_path)
        print(f"✓ Image restored with custom settings!")
        print(f"  Denoising strength: {config.models.denoising.strength}")
        print(f"  Upscaling factor: {config.models.super_resolution.scale}x")
    except Exception as e:
        print(f"✗ Restoration failed: {e}")


def example_batch_processing():
    """Example: Batch process multiple images"""
    print("\n" + "=" * 60)
    print("Example 3: Batch Processing")
    print("=" * 60)
    
    # Create pipeline
    pipeline = OldPhotoRestoration()
    
    # List of images to process
    input_images = [
        "path/to/photo1.jpg",
        "path/to/photo2.jpg",
        "path/to/photo3.jpg",
    ]
    
    output_dir = "path/to/output_batch"
    
    try:
        successes, failures = pipeline.batch_restore(
            image_paths=input_images,
            output_dir=output_dir,
            max_retries=2
        )
        
        print(f"\n✓ Batch processing complete!")
        print(f"  Successful: {len(successes)}")
        print(f"  Failed: {len(failures)}")
        
        if failures:
            print("\n  Failed images:")
            for failure in failures:
                print(f"    - {failure['input_path']}: {failure['error']}")
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")


def example_skip_modules():
    """Example: Skip specific processing modules"""
    print("\n" + "=" * 60)
    print("Example 4: Skip Specific Modules")
    print("=" * 60)
    
    # Create config with modules disabled
    config = Config.default()
    config.models.denoising.skip = False  # Enable denoising
    config.models.super_resolution.skip = True  # Skip super-resolution
    
    pipeline = OldPhotoRestoration(config=config)
    
    input_path = "path/to/your/old_photo.jpg"
    output_path = "path/to/output/denoised_only.png"
    
    try:
        restored_image = pipeline.restore(input_path, output_path)
        print(f"✓ Image processed (denoising only)!")
        print(f"  Denoising: Enabled")
        print(f"  Super-resolution: Skipped")
    except Exception as e:
        print(f"✗ Processing failed: {e}")


def example_load_from_yaml():
    """Example: Load configuration from YAML file"""
    print("\n" + "=" * 60)
    print("Example 5: Load Config from YAML")
    print("=" * 60)
    
    # Load configuration from YAML file
    config_path = "configs/config.yaml"
    
    try:
        config = Config.from_yaml(config_path)
        pipeline = OldPhotoRestoration(config=config)
        
        print(f"✓ Configuration loaded from {config_path}")
        print(f"  Denoising type: {config.models.denoising.type}")
        print(f"  SR scale: {config.models.super_resolution.scale}x")
        print(f"  Checkpoint enabled: {config.processing.checkpoint_enabled}")
        
        # Now use the pipeline...
        # restored = pipeline.restore("input.jpg", "output.png")
        
    except FileNotFoundError:
        print(f"✗ Config file not found: {config_path}")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("IMP Pipeline Usage Examples")
    print("=" * 60)
    print("\nNote: Update the file paths before running these examples!")
    print()
    
    # Run examples (commented out to avoid errors with placeholder paths)
    # Uncomment and update paths to run
    
    # example_single_image()
    # example_single_image_with_custom_config()
    # example_batch_processing()
    # example_skip_modules()
    # example_load_from_yaml()
    
    print("\n" + "=" * 60)
    print("To run these examples:")
    print("1. Update the file paths in each example function")
    print("2. Uncomment the example you want to run")
    print("3. Run: python examples/basic_usage.py")
    print("=" * 60)
