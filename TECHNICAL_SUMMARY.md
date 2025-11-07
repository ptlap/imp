# TECHNICAL DEEP DIVE - Giải thích kỹ thuật chi tiết
## Dành cho phần hỏi đáp technical

---

## 🔍 1. REAL-ESRGAN ALGORITHM

### 1.1. Architecture Overview
```
Input (RGB Image)
    ↓
[Feature Extraction]
    • First Conv: 3 → 64 channels
    ↓
[23 RRDB Blocks]
    • Residual-in-Residual Dense Block
    • Each block: 3 dense layers
    • Feature channels: 64
    • Growth channels: 32
    ↓
[Upsampling]
    • 2x: 1 PixelShuffle layer
    • 4x: 2 PixelShuffle layers
    ↓
[Final Conv]
    • 64 → 3 channels (RGB)
    ↓
Output (Upscaled Image)
```

### 1.2. RRDB Block Detail
```
Input
  │
  ├─[Dense Block]──────┐
  │   ├─Conv─ReLU      │
  │   ├─Conv─ReLU      │ (3 layers)
  │   └─Conv           │
  │                    │
  ├─[Skip Connection]──┘
  │   β × output
  │
  └─[Final Skip]
      α × input + output
```

**Parameters:**
- β (beta) = 0.2 (residual scaling)
- α (alpha) = 0.2 (main skip scaling)

### 1.3. Training Strategy (từ paper)
- **Dataset:** DIV2K, Flickr2K, OutdoorScene
- **Degradation:** Real-world simulation
  - Blur (various kernels)
  - Resize
  - Noise (Gaussian, Poisson)
  - JPEG compression
  - Unsharp masking
- **Loss Function:**
  - L1 Loss
  - Perceptual Loss (VGG features)
  - GAN Loss (adversarial)
- **Optimizer:** Adam
- **Learning rate:** 1e-4 → 1e-7 (cosine annealing)

### 1.4. Tại sao Real-ESRGAN tốt?
1. **Pure synthetic training** → không cần paired data
2. **Second-order degradation** → realistic
3. **High-order degradation modeling** → robust
4. **USM (Unsharp Masking)** → sharper results

---

## 🧮 2. NON-LOCAL MEANS DENOISING

### 2.1. Algorithm Principle
```python
# Pseudo-code
for each pixel p in image:
    for each pixel q in search_window:
        # Compare patches
        patch_p = get_patch(p, template_size)
        patch_q = get_patch(q, template_size)

        # Compute similarity weight
        weight = exp(-||patch_p - patch_q||² / h²)

        # Weighted average
        denoised[p] += weight * image[q]

    denoised[p] /= sum(weights)
```

### 2.2. Parameters Explained
- **h (filter strength):**
  - Small (5-10): Ít nhiễu, giữ details
  - Medium (10-20): Balance
  - Large (20-30): Nhiều nhiễu, có thể blur

- **templateWindowSize (7):**
  - Kích thước patch để so sánh
  - Phải là số lẻ (3, 5, 7, 9)
  - 7 là optimal cho most cases

- **searchWindowSize (21):**
  - Vùng tìm kiếm patches tương tự
  - Lớn hơn → chậm hơn nhưng tốt hơn
  - 21 là good tradeoff

### 2.3. Complexity
- **Time:** O(N × M × T²)
  - N: số pixels
  - M: search window size
  - T: template size
- **Space:** O(N)

### 2.4. Ưu/Nhược điểm
**Ưu điểm:**
- ✅ Không cần training
- ✅ Bảo toàn edges tốt
- ✅ Works on CPU
- ✅ Robust với nhiều loại noise

**Nhược điểm:**
- ❌ Slow (vài giây cho 2K image)
- ❌ Không tốt cho structured noise
- ❌ Over-smoothing nếu h quá lớn

---

## 🧩 3. TILING STRATEGY

### 3.1. Problem
```
Image size: 4096 × 4096 × 3 = 48 MB
After 4x upscale: 16384 × 16384 × 3 = 768 MB
GPU memory: Only 4-8 GB available
```

### 3.2. Solution: Tiling
```
┌────────────────────────────────┐
│ Original Image (4096x4096)     │
│                                │
│  ┌─────┬─────┬─────┬─────┐    │
│  │ T1  │ T2  │ T3  │ T4  │    │
│  ├─────┼─────┼─────┼─────┤    │
│  │ T5  │ T6  │ T7  │ T8  │    │
│  ├─────┼─────┼─────┼─────┤    │
│  │ T9  │ T10 │ T11 │ T12 │    │
│  └─────┴─────┴─────┴─────┘    │
│     ↑                          │
│     └─ Each tile: 512x512      │
│        Overlap: 64px           │
└────────────────────────────────┘
```

### 3.3. Implementation Details
```python
def tile_image(image, tile_size=512, overlap=64):
    """
    Split image into overlapping tiles
    """
    h, w = image.shape[:2]
    stride = tile_size - overlap

    tiles = []
    positions = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Extract tile
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)

            tile = image[y:y_end, x:x_end]
            tiles.append(tile)
            positions.append((y, x, y_end, x_end))

    return tiles, positions

def merge_tiles(tiles, positions, output_shape, overlap=64):
    """
    Merge tiles with blending in overlap regions
    """
    output = np.zeros(output_shape)
    weight_map = np.zeros(output_shape[:2])

    # Create feathering mask
    fade = create_fade_mask(tile_size, overlap)

    for tile, (y, x, y_end, x_end) in zip(tiles, positions):
        # Apply feathering
        tile_weighted = tile * fade

        # Add to output
        output[y:y_end, x:x_end] += tile_weighted
        weight_map[y:y_end, x:x_end] += fade

    # Normalize
    output /= weight_map[..., None]

    return output
```

### 3.4. Feathering (Blending)
```
┌─────────────────────────────┐
│ Tile 1    │    Tile 2       │
│           │                 │
│      ┌────┴────┐            │
│      │ Overlap │            │
│      │  Zone   │            │
│      └────┬────┘            │
│           │                 │
│      Blend with             │
│      linear interpolation   │
└─────────────────────────────┘

Weight transition:
Tile 1: 1.0 → 0.5 → 0.0
Tile 2: 0.0 → 0.5 → 1.0
```

### 3.5. Memory Calculation
```
Without tiling:
- Input: 4096×4096×3 = 48 MB
- After 4x: 16384×16384×3 = 768 MB
- Intermediate: ~2 GB
- Total: ~3 GB VRAM

With tiling (512×512):
- Per tile input: 512×512×3 = 0.75 MB
- Per tile output: 2048×2048×3 = 12 MB
- Model weights: ~60 MB
- Total: ~100 MB VRAM per tile
```

**Benefit:** Có thể xử lý ảnh unlimited size với fixed memory!

---

## 💾 4. CHECKPOINT SYSTEM

### 4.1. Why Checkpointing?
**Problems solved:**
1. **OOM (Out of Memory):** Resume từ bước trước OOM
2. **Crash/Interrupt:** Không mất công xử lý
3. **Debugging:** Kiểm tra output từng bước
4. **Experimentation:** Test different configs từ checkpoint

### 4.2. Storage Format
```python
# Checkpoint structure
checkpoint_data = {
    'image': np.ndarray,        # Processed image
    'metadata': {
        'original_size': tuple,
        'is_grayscale': bool,
        'resize_factor': float,
        'step': str               # 'preprocessed', 'denoised', 'sr'
    },
    'timestamp': float,           # Unix timestamp
    'config': dict               # Config used
}

# File naming
checkpoint_name = f"{image_id}_{step}.pkl"
# Example: "photo123_preprocessed.pkl"
```

### 4.3. Resume Logic
```python
def restore_with_resume(image_path, resume=True):
    steps = ['preprocessed', 'denoised', 'sr']
    image = None

    # Find latest checkpoint
    for step in steps:
        if resume and checkpoint_exists(image_path, step):
            image, metadata = load_checkpoint(image_path, step)
            start_from = steps.index(step) + 1
            break

    # Continue from checkpoint or start fresh
    if image is None:
        image = preprocess(image_path)
        save_checkpoint(image, 'preprocessed')
        start_from = 1

    # Continue remaining steps
    for i in range(start_from, len(steps)):
        image = process_step(image, steps[i])
        save_checkpoint(image, steps[i])

    return image
```

### 4.4. Trade-offs
**Pros:**
- ✅ Resumable
- ✅ Debuggable
- ✅ Fault-tolerant

**Cons:**
- ❌ Disk space (mỗi checkpoint ~50-200 MB)
- ❌ I/O overhead (save/load time)
- ❌ Pickle security concerns

**Alternatives:**
- **NumPy:** `np.savez_compressed()` - safe, compressed
- **HDF5:** `h5py` - efficient, structured
- **Memory-mapped:** `np.memmap()` - zero-copy

---

## 🧠 5. MEMORY MANAGEMENT

### 5.1. Memory Lifecycle
```
Timeline: ─────────────────────────────────────────→

Memory   ↑
Usage    │     ┌─Model─┐
         │     │       │
         │  ┌──┘       └──┐ ← Clear cache
         │  │             │
         │  │    Process  │
         │  │             │
         ├──┴─────────────┴──────────────────────→
         0  Load        Unload           Time

Peak: During model inference
Base: After cleanup
```

### 5.2. PyTorch Memory Model
```python
# PyTorch allocates memory in 2 ways:

1. ALLOCATED (Active memory)
   - Currently used tensors
   - Model weights

2. RESERVED (Cached memory)
   - Previously allocated but freed
   - Kept for reuse (faster)
   - Not returned to system

# Our solution:
torch.cuda.empty_cache()  # Free cached memory
gc.collect()              # Python garbage collection
```

### 5.3. Memory Tracking
```python
def log_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"Allocated: {allocated:.2f}GB")
        print(f"Reserved: {reserved:.2f}GB")
```

### 5.4. Memory Optimization Techniques
**1. Lazy Loading:**
```python
# Bad: Load all models upfront
model1 = load_model1()
model2 = load_model2()
process(model1, model2)

# Good: Load only when needed
model1 = None
model2 = None

def get_model1():
    if model1 is None:
        model1 = load_model1()
    return model1
```

**2. Immediate Unloading:**
```python
# Process with model
model = load_model()
result = model.process(image)

# Unload immediately
del model
torch.cuda.empty_cache()
```

**3. FP16 Inference:**
```python
# FP32 (default): 4 bytes per param
model_fp32 = load_model().float()  # ~240 MB

# FP16 (half): 2 bytes per param
model_fp16 = load_model().half()   # ~120 MB

# 50% memory saving!
# Quality loss: <1% for most tasks
```

**4. Gradient Disabled:**
```python
# Training mode (tracks gradients)
with torch.no_grad():           # Inference mode
    output = model(input)       # No gradient tracking
                                # ~40% memory saving
```

---

## 🎛️ 6. CONFIGURATION SYSTEM

### 6.1. Why YAML?
**Pros:**
- ✅ Human-readable
- ✅ Comments supported
- ✅ Hierarchical structure
- ✅ Language-agnostic
- ✅ Git-friendly

**Cons:**
- ❌ No type checking (giải quyết: validation)
- ❌ Indentation-sensitive

### 6.2. Validation Strategy
```python
@dataclass
class Config:
    def validate(self) -> bool:
        errors = []

        # Range validation
        if not 1 <= strength <= 100:
            errors.append(f"Invalid strength: {strength}")

        # Enum validation
        if type not in ["opencv", "nafnet"]:
            errors.append(f"Invalid type: {type}")

        # Dependency validation
        if tile_overlap >= tile_size:
            errors.append("Overlap must < tile_size")

        # Raise if errors
        if errors:
            raise ConfigurationError("\n".join(errors))

        return True
```

### 6.3. Configuration Precedence
```
1. Command-line args (highest priority)
   └─ python main.py --scale 2

2. Environment variables
   └─ export IMP_SCALE=2

3. Config file (YAML)
   └─ config.yaml: scale: 2

4. Default values (lowest priority)
   └─ @dataclass default values
```

### 6.4. Alternative: Pydantic
```python
from pydantic import BaseModel, Field, validator

class DenoisingConfig(BaseModel):
    type: str = Field(..., regex="^(opencv|nafnet)$")
    strength: int = Field(10, ge=1, le=100)

    @validator('strength')
    def validate_strength(cls, v):
        if v < 1 or v > 100:
            raise ValueError("Must be 1-100")
        return v

# Automatic validation!
# Type conversion!
# Better error messages!
```

---

## 🚨 7. ERROR HANDLING STRATEGY

### 7.1. Exception Hierarchy
```
Exception (built-in)
    │
    └── IMPError (custom base)
            │
            ├── ConfigurationError
            │   ├─ Invalid config values
            │   ├─ Missing config file
            │   └─ Validation failed
            │
            ├── ModelLoadError
            │   ├─ Weights not found
            │   ├─ Download failed
            │   ├─ Library not installed
            │   └─ GPU not available
            │
            ├── ProcessingError
            │   ├─ Image load failed
            │   ├─ Invalid format
            │   ├─ Corruption detected
            │   └─ Processing failed
            │
            └── OutOfMemoryError
                ├─ GPU OOM
                ├─ RAM exhausted
                └─ Image too large
```

### 7.2. Error Handling Pattern
```python
def process_image(image_path):
    try:
        # Attempt processing
        image = load_image(image_path)
        result = model.process(image)
        return result

    except OutOfMemoryError as e:
        # Specific handling for OOM
        logger.error(f"OOM: {e}")
        suggestions = [
            "Reduce tile_size to 256",
            "Use 2x instead of 4x",
            "Skip super-resolution"
        ]
        raise ProcessingError(
            f"Out of memory. Try: {suggestions}"
        ) from e

    except ModelLoadError as e:
        # Specific handling for model errors
        logger.error(f"Model error: {e}")
        raise

    except ProcessingError as e:
        # General processing errors
        logger.error(f"Processing failed: {e}")
        raise

    except Exception as e:
        # Unexpected errors
        logger.critical(f"Unexpected: {e}", exc_info=True)
        raise ProcessingError(
            f"Unexpected error: {e}"
        ) from e
```

### 7.3. Retry Logic
```python
def process_with_retry(image_path, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            return process_image(image_path)

        except OutOfMemoryError as e:
            if attempt < max_retries:
                # Try with smaller settings
                config.tile_size //= 2
                logger.warning(
                    f"OOM - Retry {attempt+1}/{max_retries} "
                    f"with tile_size={config.tile_size}"
                )
                clear_checkpoints()  # Fresh start
                continue
            else:
                raise

        except ProcessingError as e:
            if attempt < max_retries:
                logger.warning(
                    f"Failed - Retry {attempt+1}/{max_retries}"
                )
                continue
            else:
                raise
```

---

## 🧪 8. TESTING STRATEGY

### 8.1. Test Types
```
┌─────────────────────────────────────────┐
│ Testing Pyramid                         │
├─────────────────────────────────────────┤
│                 ┌──────┐                │
│                 │ E2E  │  ← Few         │
│                 └──────┘                │
│             ┌──────────────┐            │
│             │ Integration  │  ← Some    │
│             └──────────────┘            │
│         ┌──────────────────────┐        │
│         │    Unit Tests        │  ← Many│
│         └──────────────────────┘        │
└─────────────────────────────────────────┘
```

### 8.2. Unit Test Examples
```python
# test_preprocessing.py
def test_smart_resize():
    # Arrange
    image = np.ones((4096, 4096, 3))
    preprocessor = Preprocessor(max_size=2048)

    # Act
    resized, scale = preprocessor.smart_resize(image)

    # Assert
    assert resized.shape == (2048, 2048, 3)
    assert scale == 0.5

def test_grayscale_detection():
    # True grayscale
    gray = np.ones((100, 100, 3)) * 128
    assert detect_grayscale(gray) == True

    # Color image
    color = np.random.rand(100, 100, 3) * 255
    assert detect_grayscale(color) == False
```

### 8.3. Integration Test
```python
# test_pipeline.py
def test_full_pipeline():
    # Arrange
    pipeline = OldPhotoRestoration()
    test_image = "test_data/noisy_lowres.jpg"

    # Act
    output = pipeline.restore(test_image)

    # Assert
    assert output is not None
    assert output.shape[0] > 0  # Has height
    assert output.shape[1] > 0  # Has width
    assert output.shape[2] == 3  # RGB
    assert output.min() >= 0
    assert output.max() <= 1
```

### 8.4. Mock Usage
```python
from unittest.mock import Mock, patch

def test_model_loading():
    # Mock model loading (no actual weights)
    with patch('src.models.super_resolution.RealESRGANer') as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance

        # Test
        model = SuperResolutionModule()
        model.load_model()

        # Verify
        mock.assert_called_once()
```

---

## 📊 9. PERFORMANCE ANALYSIS

### 9.1. Profiling Results
```
Function                    Calls   Time(s)   %Total
──────────────────────────────────────────────────
preprocess                      1      0.5      2.6%
  ├─ load_image                 1      0.3      1.6%
  ├─ detect_grayscale           1      0.1      0.5%
  └─ smart_resize               1      0.1      0.5%

denoise                         1      3.0     15.8%
  └─ fastNlMeans                1      2.9     15.3%

super_resolution                1     15.5     81.6%
  ├─ model_forward             16     14.0     73.7%  ← 16 tiles
  └─ merge_tiles                1      1.5      7.9%
──────────────────────────────────────────────────
TOTAL                                 19.0    100.0%
```

### 9.2. Bottleneck Analysis
**Main bottleneck:** Super-resolution (81.6%)
- Model inference: 73.7%
- Tile merging: 7.9%

**Optimization opportunities:**
1. **Model quantization** (INT8) → 2-3x faster
2. **TensorRT** optimization → 2x faster
3. **Batch inference** → 1.5x faster
4. **Multi-GPU** → linear speedup

### 9.3. Memory Profile
```
Step              Peak Memory    Avg Memory
─────────────────────────────────────────────
Preprocessing          150 MB        100 MB
Denoising              300 MB        250 MB
Super-resolution     3,500 MB      2,000 MB  ← Peak
Post-processing        200 MB        150 MB
```

---

## 🎓 10. KEY LEARNINGS

### 10.1. Technical Lessons
1. **Lazy loading is powerful** - Giảm 70% memory
2. **Tiling enables scalability** - Process unlimited size
3. **FP16 is a good trade-off** - 50% memory, <1% quality loss
4. **Checkpoints are critical** - Resume saves hours
5. **Error handling improves UX** - User-friendly messages

### 10.2. Software Engineering
1. **Type hints prevent bugs** - Caught 30+ bugs early
2. **Tests save time long-term** - Debug faster
3. **Documentation = Code** - Future self thanks you
4. **Modular design pays off** - Easy to extend
5. **Configuration is powerful** - Flexible without code changes

### 10.3. Challenges Overcome
1. **OOM errors** → Tiling strategy
2. **Slow processing** → Lazy loading + FP16
3. **Crashes** → Checkpoint system
4. **Maintainability** → Design patterns
5. **Debugging** → Comprehensive logging

---

**Nhớ kỹ những điểm này để trả lời câu hỏi technical!** 🎯
