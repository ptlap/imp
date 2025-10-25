#!/usr/bin/env python3
"""
End-to-end test for WSL environment (without GPU models).
Tests basic pipeline functionality with preprocessing and OpenCV denoising only.
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import OldPhotoRestoration
from src.config import Config


def create_test_image(path: Path, size=(512, 512)):
    """Create a test image with some noise"""
    # Create a simple gradient image with noise
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    # Add some structure
    for i in range(size[0]):
        for j in range(size[1]):
            img[i, j] = [i % 256, j % 256, (i + j) % 256]
    
    # Add noise
    noise = np.random.randint(-20, 20, (*size, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    Image.fromarray(img).save(path)
    return path


def test_pipeline_initialization():
    """Test 1: Pipeline can be initialized"""
    print("\n[Test 1] Pipeline Initialization")
    try:
        pipeline = OldPhotoRestoration()
        print("  ✓ Pipeline initialized with default config")
        
        config = Config.default()
        pipeline = OldPhotoRestoration(config=config)
        print("  ✓ Pipeline initialized with custom config")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_config_loading():
    """Test 2: Configuration can be loaded from YAML"""
    print("\n[Test 2] Configuration Loading")
    try:
        config = Config.from_yaml("configs/config.yaml")
        print("  ✓ Config loaded from YAML")
        
        config.validate()
        print("  ✓ Config validation passed")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_preprocessing_only():
    """Test 3: Preprocessing module works"""
    print("\n[Test 3] Preprocessing Module")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test image
            input_path = tmpdir / "test_input.jpg"
            create_test_image(input_path)
            print(f"  ✓ Created test image: {input_path}")
            
            # Test preprocessing
            from src.utils.preprocessing import Preprocessor
            preprocessor = Preprocessor(max_size=2048)
            
            image, metadata = preprocessor.process(str(input_path))
            print(f"  ✓ Preprocessed image shape: {image.shape}")
            print(f"  ✓ Metadata: {metadata}")
            
            return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_denoising_only():
    """Test 4: OpenCV denoising works (CPU-only)"""
    print("\n[Test 4] OpenCV Denoising (CPU)")
    try:
        from src.models.denoiser import create_denoiser
        
        # Create test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Create denoiser
        denoiser = create_denoiser(denoiser_type="opencv", strength=10)
        print("  ✓ OpenCV denoiser created")
        
        # Denoise
        denoised = denoiser.denoise(test_image)
        print(f"  ✓ Denoised image shape: {denoised.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_without_sr():
    """Test 5: Full pipeline without super-resolution (CPU-only)"""
    print("\n[Test 5] Pipeline without Super-Resolution")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test image
            input_path = tmpdir / "test_input.jpg"
            output_path = tmpdir / "test_output.png"
            create_test_image(input_path, size=(256, 256))
            print(f"  ✓ Created test image: {input_path}")
            
            # Configure pipeline to skip super-resolution
            config = Config.default()
            config.models.super_resolution.skip = True
            config.processing.checkpoint_enabled = False
            config.logging.level = "WARNING"  # Reduce log noise
            
            pipeline = OldPhotoRestoration(config=config)
            print("  ✓ Pipeline created (SR disabled)")
            
            # Restore image
            restored = pipeline.restore(
                image_path=str(input_path),
                output_path=str(output_path),
                resume=False
            )
            
            print(f"  ✓ Image restored successfully")
            print(f"  ✓ Output shape: {restored.shape}")
            print(f"  ✓ Output saved to: {output_path}")
            
            # Verify output file exists
            assert output_path.exists(), "Output file not created"
            print(f"  ✓ Output file verified")
            
            return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_processing():
    """Test 6: Batch processing without super-resolution"""
    print("\n[Test 6] Batch Processing")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create multiple test images
            input_paths = []
            for i in range(3):
                input_path = tmpdir / f"test_input_{i}.jpg"
                create_test_image(input_path, size=(128, 128))
                input_paths.append(str(input_path))
            print(f"  ✓ Created {len(input_paths)} test images")
            
            output_dir = tmpdir / "output"
            
            # Configure pipeline
            config = Config.default()
            config.models.super_resolution.skip = True
            config.processing.checkpoint_enabled = False
            config.logging.level = "WARNING"
            
            pipeline = OldPhotoRestoration(config=config)
            
            # Batch restore
            successes, failures = pipeline.batch_restore(
                image_paths=input_paths,
                output_dir=str(output_dir),
                max_retries=1
            )
            
            print(f"  ✓ Batch processing complete")
            print(f"  ✓ Successes: {len(successes)}")
            print(f"  ✓ Failures: {len(failures)}")
            
            assert len(successes) == 3, f"Expected 3 successes, got {len(successes)}"
            assert len(failures) == 0, f"Expected 0 failures, got {len(failures)}"
            
            return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all end-to-end tests"""
    print("=" * 70)
    print("IMP End-to-End Testing on WSL (CPU-only)")
    print("=" * 70)
    
    tests = [
        test_pipeline_initialization,
        test_config_loading,
        test_preprocessing_only,
        test_denoising_only,
        test_pipeline_without_sr,
        test_batch_processing,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
