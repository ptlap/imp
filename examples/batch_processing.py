"""
Batch processing example for IMP pipeline.

This script demonstrates how to process multiple images in batch mode
with progress tracking, error handling, and retry logic.
"""

from pathlib import Path
from src.pipeline import OldPhotoRestoration
from src.config import Config


def batch_process_directory(input_dir: str, output_dir: str):
    """
    Process all images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save restored images
    """
    print("=" * 60)
    print("Batch Processing Example")
    print("=" * 60)
    
    # Create pipeline with default configuration
    config = Config.default()
    pipeline = OldPhotoRestoration(config)
    
    # Find all image files in input directory
    input_path = Path(input_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(input_path.glob(f'*{ext}'))
    
    # Convert to strings
    image_paths = [str(p) for p in image_paths]
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    
    print(f"\nFound {len(image_paths)} images to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Process batch with retry logic
    successes, failures = pipeline.batch_restore(
        image_paths=image_paths,
        output_dir=output_dir,
        max_retries=2
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Batch Processing Complete")
    print("=" * 60)
    print(f"Total images: {len(image_paths)}")
    print(f"Successful: {len(successes)}")
    print(f"Failed: {len(failures)}")
    
    if successes:
        print("\nSuccessfully processed:")
        for success in successes[:5]:  # Show first 5
            status = "(skipped)" if success.get('skipped') else "(processed)"
            print(f"  ✓ {Path(success['input_path']).name} {status}")
        if len(successes) > 5:
            print(f"  ... and {len(successes) - 5} more")
    
    if failures:
        print("\nFailed to process:")
        for failure in failures:
            print(f"  ✗ {Path(failure['input_path']).name}")
            print(f"    Error: {failure['error']}")
            print(f"    Attempts: {failure['attempts']}")


def batch_process_with_custom_config(image_paths: list, output_dir: str):
    """
    Process batch with custom configuration.
    
    Args:
        image_paths: List of image file paths
        output_dir: Directory to save restored images
    """
    print("\n" + "=" * 60)
    print("Batch Processing with Custom Config")
    print("=" * 60)
    
    # Create custom configuration
    config = Config.default()
    config.models.denoising.strength = 15  # Stronger denoising
    config.models.super_resolution.scale = 2  # 2x instead of 4x
    config.processing.checkpoint_enabled = True  # Enable checkpoints
    
    print(f"\nConfiguration:")
    print(f"  Denoising strength: {config.models.denoising.strength}")
    print(f"  Super-resolution scale: {config.models.super_resolution.scale}x")
    print(f"  Checkpoints: {'Enabled' if config.processing.checkpoint_enabled else 'Disabled'}")
    print()
    
    # Create pipeline
    pipeline = OldPhotoRestoration(config)
    
    # Process batch
    successes, failures = pipeline.batch_restore(
        image_paths=image_paths,
        output_dir=output_dir,
        max_retries=1
    )
    
    print(f"\nProcessed {len(successes)} images successfully")
    if failures:
        print(f"Failed to process {len(failures)} images")


def selective_batch_processing(image_paths: list, output_dir: str):
    """
    Process batch with selective module execution.
    
    Args:
        image_paths: List of image file paths
        output_dir: Directory to save restored images
    """
    print("\n" + "=" * 60)
    print("Selective Batch Processing (Denoising Only)")
    print("=" * 60)
    
    # Configure to skip super-resolution
    config = Config.default()
    config.models.denoising.skip = False  # Enable denoising
    config.models.super_resolution.skip = True  # Skip super-resolution
    
    print("\nProcessing pipeline:")
    print("  ✓ Preprocessing")
    print("  ✓ Denoising")
    print("  ✗ Super-resolution (skipped)")
    print()
    
    pipeline = OldPhotoRestoration(config)
    
    successes, failures = pipeline.batch_restore(
        image_paths=image_paths,
        output_dir=output_dir
    )
    
    print(f"\nCompleted: {len(successes)} images (denoising only)")


if __name__ == "__main__":
    # Example 1: Process all images in a directory
    # Uncomment and update paths to run
    # batch_process_directory(
    #     input_dir="path/to/input/directory",
    #     output_dir="path/to/output/directory"
    # )
    
    # Example 2: Process specific images with custom config
    # image_list = [
    #     "path/to/photo1.jpg",
    #     "path/to/photo2.jpg",
    #     "path/to/photo3.jpg",
    # ]
    # batch_process_with_custom_config(
    #     image_paths=image_list,
    #     output_dir="path/to/output"
    # )
    
    # Example 3: Selective processing (denoising only)
    # selective_batch_processing(
    #     image_paths=image_list,
    #     output_dir="path/to/output_denoised"
    # )
    
    print("\n" + "=" * 60)
    print("Batch Processing Examples")
    print("=" * 60)
    print("\nTo run these examples:")
    print("1. Update the file paths in the example functions")
    print("2. Uncomment the example you want to run")
    print("3. Run: python examples/batch_processing.py")
    print("=" * 60)
