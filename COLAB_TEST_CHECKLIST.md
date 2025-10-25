# Google Colab Testing Checklist

This document provides a checklist for testing the IMP pipeline on Google Colab.

## Pre-Test Setup

- [ ] Open Google Colab: https://colab.research.google.com/
- [ ] Upload notebook: `notebooks/01_quick_start.ipynb`
- [ ] Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU
- [ ] Verify GPU is available (run cell 1)

## Test Execution Steps

### 1. Environment Setup (Cells 1-3)
- [ ] Cell 1: GPU check passes and shows T4 GPU (or similar)
- [ ] Cell 2: Repository clones successfully
- [ ] Cell 3: Dependencies install without errors

### 2. Model Preparation (Cell 4)
- [ ] Cell 4: Model weights download successfully
- [ ] Verify weights file exists in `weights/` directory
- [ ] File size should be ~60-70 MB for Real-ESRGAN

### 3. Pipeline Initialization (Cell 5)
- [ ] Cell 5: Pipeline initializes without errors
- [ ] Configuration loads correctly
- [ ] No import errors

### 4. Single Image Restoration (Cells 6-8)
- [ ] Cell 6: Image upload works
- [ ] Cell 7: Restoration completes successfully
  - [ ] Processing time < 30 seconds for 512x512 image
  - [ ] No out-of-memory errors
  - [ ] Output file is created
- [ ] Cell 8: Visualization displays correctly
  - [ ] Before/after comparison shows
  - [ ] Image dimensions are correct (4x upscaling)

### 5. Detail Comparison (Cell 9)
- [ ] Zoom comparison shows improved detail
- [ ] No artifacts or distortions visible

### 6. Download Results (Cell 10)
- [ ] Download works correctly
- [ ] File can be opened locally

### 7. Batch Processing (Cell 11) - Optional
- [ ] Multiple images can be uploaded
- [ ] Batch processing completes
- [ ] Progress bar displays correctly
- [ ] All images process successfully

### 8. Advanced Features (Cells 12-13)
- [ ] Custom configuration works
- [ ] Memory monitoring displays correctly
- [ ] Memory usage < 4GB for standard images

## Performance Targets

| Metric | Target | Actual | Pass/Fail |
|--------|--------|--------|-----------|
| 512x512 processing time | < 5s | ___ s | ___ |
| 1024x1024 processing time | < 20s | ___ s | ___ |
| Peak GPU memory usage | < 4GB | ___ GB | ___ |
| 4x upscaling accuracy | Correct dimensions | ___ | ___ |

## Memory Usage Verification

Test with different image sizes and record memory usage:

| Image Size | Memory Used | Status |
|------------|-------------|--------|
| 256x256 | ___ GB | ___ |
| 512x512 | ___ GB | ___ |
| 1024x1024 | ___ GB | ___ |
| 2048x2048 | ___ GB | ___ |

## Error Handling Tests

- [ ] Test with invalid image format (should fail gracefully)
- [ ] Test with very large image (should use tiling)
- [ ] Test with corrupted image (should show error message)
- [ ] Test without GPU (should fall back to CPU for denoising)

## Quality Verification

- [ ] Denoising reduces visible noise
- [ ] Super-resolution increases sharpness
- [ ] No obvious artifacts or distortions
- [ ] Colors are preserved correctly
- [ ] Edge details are enhanced

## Known Limitations

Document any issues found:

1. _______________________________________________
2. _______________________________________________
3. _______________________________________________

## Test Results Summary

**Date:** _______________
**Tester:** _______________
**Colab GPU Type:** _______________
**Overall Status:** PASS / FAIL

**Notes:**
_______________________________________________
_______________________________________________
_______________________________________________

## Recommendations

Based on testing, document any recommended changes:

1. _______________________________________________
2. _______________________________________________
3. _______________________________________________

---

## Automated Verification Script

For quick verification, you can run this in a Colab cell:

```python
# Quick verification script
import torch
import os
from src.pipeline import OldPhotoRestoration
from src.config import Config
from src.utils.memory import MemoryManager

print("=" * 60)
print("IMP Colab Verification")
print("=" * 60)

# 1. Check GPU
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("✗ No GPU available")

# 2. Check dependencies
try:
    import cv2
    import numpy as np
    from PIL import Image
    import yaml
    print("✓ All dependencies imported")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")

# 3. Check weights
weights_path = "weights/realesrgan-x4plus.pth"
if os.path.exists(weights_path):
    size_mb = os.path.getsize(weights_path) / 1024**2
    print(f"✓ Weights exist ({size_mb:.1f} MB)")
else:
    print("✗ Weights not found")

# 4. Test pipeline initialization
try:
    config = Config.default()
    pipeline = OldPhotoRestoration(config)
    print("✓ Pipeline initialized")
except Exception as e:
    print(f"✗ Pipeline initialization failed: {e}")

# 5. Check memory
if torch.cuda.is_available():
    usage = MemoryManager.get_memory_usage()
    print(f"✓ Current GPU memory: {usage['allocated']:.2f} GB")

print("=" * 60)
print("Verification complete!")
print("=" * 60)
```
