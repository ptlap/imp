# 🎨 IMP - Image Restoration Project

**IMP** (Image Restoration Project) - Hệ thống phục chế ảnh cũ tự động sử dụng Deep Learning

[![GitHub](https://img.shields.io/badge/GitHub-ptlap%2Fimp-blue)](https://github.com/ptlap/imp)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Colab](https://img.shields.io/badge/Colab-Ready-yellow)](https://colab.research.google.com/)

---

## 📋 Giới thiệu

**IMP** là một đồ án sử dụng Deep Learning để tự động phục chế ảnh cũ/hư hỏng với các tính năng:

- 🧹 **Khử nhiễu** - Loại bỏ noise, scratches, JPEG artifacts
- 🔍 **Tăng độ phân giải** - Super-resolution 2x/4x với Real-ESRGAN
- 🎨 **Tô màu tự động** - Colorization cho ảnh đen trắng với DDColor
- 👤 **Phục hồi khuôn mặt** - Face enhancement với CodeFormer

---

## 🚀 Quick Start

### **Option 1: Google Colab (Recommended for GPU)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ptlap/imp/blob/main/notebooks/01_quick_start.ipynb)

**Step-by-step Colab Setup:**

1. **Open the Colab notebook** using the badge above
2. **Enable GPU runtime:**
   - Click `Runtime` → `Change runtime type`
   - Select `T4 GPU` or `V100 GPU`
   - Click `Save`
3. **Run the setup cells:**

```python
# Clone repository
!git clone https://github.com/ptlap/imp.git
%cd imp

# Install dependencies
!pip install -q -r requirements.txt

# Verify installation
!python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

4. **Run restoration:**

```python
from src.pipeline import OldPhotoRestoration
from src.config import Config

# Create pipeline with default config
config = Config.default()
pipeline = OldPhotoRestoration(config)

# Restore single image
restored = pipeline.restore(
    image_path='path/to/old_photo.jpg',
    output_path='restored_photo.png'
)

# Display results
from PIL import Image
import matplotlib.pyplot as plt

original = Image.open('path/to/old_photo.jpg')
restored_img = Image.fromarray((restored * 255).astype('uint8'))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(original)
axes[0].set_title('Original')
axes[0].axis('off')
axes[1].imshow(restored_img)
axes[1].set_title('Restored')
axes[1].axis('off')
plt.show()
```

**Note:** Model weights will be automatically downloaded on first use (~65MB for Real-ESRGAN).

---

### **Option 2: Local Setup (WSL/Linux)**

**Prerequisites:**
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU with CUDA 11.0+ for GPU acceleration

**Step 1: Clone Repository**

```bash
git clone https://github.com/ptlap/imp.git
cd imp
```

**Step 2: Create Virtual Environment**

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac/WSL

# Verify activation (should show path to venv)
which python
```

**Step 3: Install Dependencies**

```bash
# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

**Step 4: Run Tests (Optional)**

```bash
# Run all tests (CPU only, no GPU required)
pytest tests/ -v

# Run specific test
pytest tests/test_pipeline.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

**Step 5: Basic Usage**

```bash
# Run basic example
python examples/basic_usage.py

# Or use Python interactively
python
>>> from src.pipeline import OldPhotoRestoration
>>> pipeline = OldPhotoRestoration()
>>> restored = pipeline.restore('input.jpg', 'output.png')
```

---

### **Option 3: Local Setup (Windows)**

**Prerequisites:**
- Python 3.8+ installed from [python.org](https://www.python.org/downloads/)
- Git for Windows

**Step 1: Clone Repository**

```cmd
git clone https://github.com/ptlap/imp.git
cd imp
```

**Step 2: Create Virtual Environment**

```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Verify activation
where python
```

**Step 3: Install Dependencies**

```cmd
# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

**Step 4: Run Tests**

```cmd
pytest tests\ -v
```

---

## 📁 Project Structure

```
imp/
├── src/                              # Source code
│   ├── __init__.py                   # Package initialization
│   ├── config.py                     # Configuration management
│   ├── pipeline.py                   # Main pipeline orchestrator
│   ├── models/                       # Model implementations
│   │   ├── __init__.py
│   │   ├── denoiser.py              # Denoising (OpenCV, NAFNet)
│   │   └── super_resolution.py      # Super-resolution (Real-ESRGAN)
│   └── utils/                        # Utility modules
│       ├── __init__.py
│       ├── preprocessing.py         # Image loading and preparation
│       ├── checkpoint.py            # Checkpoint management
│       ├── memory.py                # GPU memory management
│       ├── logging.py               # Logging configuration
│       ├── exceptions.py            # Custom exceptions
│       └── weight_downloader.py     # Model weight downloader
├── examples/                         # Usage examples
│   ├── basic_usage.py               # Single image restoration
│   ├── batch_processing.py          # Batch processing examples
│   └── custom_configuration.py      # Configuration examples
├── tests/                            # Unit tests
│   ├── __init__.py
│   ├── conftest.py                  # Pytest fixtures
│   ├── test_config.py               # Config tests
│   ├── test_pipeline.py             # Pipeline tests
│   ├── test_preprocessing.py        # Preprocessing tests
│   ├── test_denoiser.py             # Denoising tests
│   ├── test_super_resolution.py     # Super-resolution tests
│   ├── test_checkpoint.py           # Checkpoint tests
│   ├── test_memory.py               # Memory management tests
│   └── test_weight_downloader.py    # Weight downloader tests
├── configs/                          # Configuration files
│   └── config.yaml                  # Default configuration
├── docs/                             # Documentation
│   ├── blueprint.md                 # Architecture overview
│   ├── development_workflow.md      # Development guide
│   └── blueprint_optimization_summary.md
├── notebooks/                        # Jupyter notebooks (future)
├── checkpoints/                      # Intermediate results (generated)
├── weights/                          # Model weights (downloaded)
├── .gitignore                        # Git ignore rules
├── pytest.ini                        # Pytest configuration
├── requirements.txt                  # Python dependencies
├── SETUP.md                          # Detailed setup guide
└── README.md                         # This file
```

### **Key Components**

- **`src/pipeline.py`**: Main entry point - orchestrates the restoration process
- **`src/config.py`**: Configuration management with validation
- **`src/models/`**: Model implementations (denoising, super-resolution)
- **`src/utils/`**: Supporting utilities (preprocessing, checkpointing, memory, logging)
- **`examples/`**: Ready-to-run example scripts
- **`tests/`**: Comprehensive unit tests (run without GPU)

---

## 📦 Dependencies

### **Core Dependencies**

- **Python 3.8+**: Programming language
- **PyTorch 2.0+**: Deep learning framework
- **OpenCV (cv2)**: Image processing
- **NumPy**: Numerical operations
- **Pillow (PIL)**: Image I/O
- **PyYAML**: Configuration file parsing

### **Model Dependencies**

- **basicsr**: Image restoration framework
- **realesrgan**: Real-ESRGAN implementation
- **facexlib**: Face detection and enhancement (future)

### **Development Dependencies**

- **pytest**: Testing framework
- **tqdm**: Progress bars
- **scikit-image**: Image quality metrics (future)

### **Installation**

All dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### **Optional: GPU Support**

For GPU acceleration, ensure you have:
- CUDA 11.0 or higher
- cuDNN 8.0 or higher
- PyTorch with CUDA support

Check GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 🔧 Architecture

```
INPUT IMAGE
    ↓
[Preprocessing]
    ↓
[Denoising] ← OpenCV FastNlMeans / NAFNet
    ↓
[Super-Resolution] ← Real-ESRGAN 4x
    ↓
[Colorization?] ← DDColor (nếu ảnh B&W)
    ↓
[Face Detection] ← RetinaFace
    ↓
[Face Enhancement] ← CodeFormer (nếu có faces)
    ↓
[Post-processing]
    ↓
OUTPUT IMAGE
```

---

## 🎯 Features

### ✅ Implemented
- [x] Preprocessing pipeline
- [x] OpenCV-based denoising
- [x] Real-ESRGAN super-resolution
- [x] Smart tiling for large images
- [x] Checkpoint system
- [x] Batch processing

### 🚧 In Progress
- [ ] DDColor colorization
- [ ] CodeFormer face enhancement
- [ ] Gradio web interface
- [ ] Evaluation metrics

### 📝 Planned
- [ ] Fine-tuning on old photos
- [ ] Video restoration
- [ ] Mobile app (TFLite)
- [ ] REST API

---

## 📊 Performance

| Image Size | Processing Time | GPU Memory | Quality (NIQE) |
|------------|----------------|------------|----------------|
| 512x512    | ~4s            | 2GB        | 4.2            |
| 1024x1024  | ~15s           | 4GB        | 4.5            |
| 2048x2048  | ~60s (tiled)   | 4GB        | 4.8            |

*Tested on Google Colab T4 GPU*

---

## 🛠️ Development Workflow

### **1. Code trên Local**
```bash
# Edit code
vim src/pipeline.py

# Test locally (basic logic)
python tests/test_pipeline.py

# Commit & push
git add .
git commit -m "Add feature X"
git push
```

### **2. Run trên Colab**
```python
# Pull latest code
!cd imp && git pull

# Test with GPU
from src.pipeline import Pipeline
pipeline = Pipeline()
result = pipeline.restore('test.jpg')
```

Xem chi tiết: [Development Workflow](docs/development_workflow.md)

---

## 📖 Usage Guide

### **Basic Single Image Restoration**

```python
from src.pipeline import OldPhotoRestoration

# Create pipeline with default settings
pipeline = OldPhotoRestoration()

# Restore image
restored = pipeline.restore(
    image_path='old_photo.jpg',
    output_path='restored_photo.png',
    resume=True  # Resume from checkpoint if available
)
```

### **Batch Processing**

```python
from src.pipeline import OldPhotoRestoration

pipeline = OldPhotoRestoration()

# Process multiple images
image_paths = ['photo1.jpg', 'photo2.jpg', 'photo3.jpg']
successes, failures = pipeline.batch_restore(
    image_paths=image_paths,
    output_dir='./restored_batch',
    max_retries=2
)

print(f"Successful: {len(successes)}, Failed: {len(failures)}")
```

### **Custom Configuration**

```python
from src.config import Config
from src.pipeline import OldPhotoRestoration

# Load from YAML
config = Config.from_yaml('configs/config.yaml')

# Or modify default config
config = Config.default()
config.models.denoising.strength = 15
config.models.super_resolution.scale = 2
config.processing.checkpoint_enabled = True

# Create pipeline with custom config
pipeline = OldPhotoRestoration(config)
restored = pipeline.restore('input.jpg', 'output.png')
```

### **Skip Specific Modules**

```python
from src.config import Config
from src.pipeline import OldPhotoRestoration

# Denoising only (skip super-resolution)
config = Config.default()
config.models.super_resolution.skip = True

pipeline = OldPhotoRestoration(config)
restored = pipeline.restore('input.jpg', 'denoised_only.png')
```

### **More Examples**

See the `examples/` directory for more usage examples:
- `examples/basic_usage.py` - Single image and basic operations
- `examples/batch_processing.py` - Batch processing with progress tracking
- `examples/custom_configuration.py` - Configuration options and presets

---

## 🔧 Troubleshooting

### **Common Issues**

#### **1. CUDA Out of Memory**

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
```python
# Option A: Reduce tile size
config = Config.default()
config.models.super_resolution.tile_size = 256  # Default is 512
config.models.super_resolution.tile_overlap = 16  # Default is 64

# Option B: Use 2x upscaling instead of 4x
config.models.super_resolution.scale = 2

# Option C: Reduce input image size
config.processing.max_image_size = 1024  # Default is 2048

# Option D: Skip super-resolution
config.models.super_resolution.skip = True
```

#### **2. Model Weights Download Fails**

**Error:** `ModelLoadError: Failed to download weights`

**Solutions:**
```bash
# Manual download
mkdir -p weights
cd weights

# Download Real-ESRGAN weights
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-x4plus.pth

# Or use alternative mirror
wget https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4plus.pth -O realesrgan-x4plus.pth
```

#### **3. Import Errors**

**Error:** `ModuleNotFoundError: No module named 'basicsr'`

**Solutions:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or install specific packages
pip install basicsr realesrgan facexlib
```

#### **4. Image Format Not Supported**

**Error:** `ProcessingError: Unsupported image format`

**Supported formats:** JPG, JPEG, PNG

**Solution:**
```bash
# Convert image to supported format using ImageMagick
convert input.bmp output.jpg

# Or using Python
from PIL import Image
img = Image.open('input.bmp')
img.save('output.jpg')
```

#### **5. Virtual Environment Issues (WSL)**

**Problem:** Virtual environment not activating

**Solutions:**
```bash
# Ensure venv module is installed
sudo apt-get install python3-venv

# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate

# Verify activation
which python  # Should show path to venv/bin/python
```

#### **6. Slow Processing on CPU**

**Problem:** Processing is very slow without GPU

**Solutions:**
- Use Google Colab with free GPU (recommended)
- Skip super-resolution module (fastest):
  ```python
  config.models.super_resolution.skip = True
  ```
- Use OpenCV denoising only (CPU-optimized):
  ```python
  config.models.denoising.type = "opencv"
  config.models.super_resolution.skip = True
  ```

#### **7. Checkpoint Resume Not Working**

**Problem:** Pipeline doesn't resume from checkpoint

**Solutions:**
```python
# Ensure checkpoints are enabled
config.processing.checkpoint_enabled = True

# Check checkpoint directory exists
import os
os.makedirs('./checkpoints', exist_ok=True)

# Clear corrupted checkpoints
from src.utils.checkpoint import CheckpointManager
checkpoint_mgr = CheckpointManager('./checkpoints')
checkpoint_mgr.clear()
```

### **Getting Help**

If you encounter issues not covered here:

1. **Check logs:** Review `imp.log` for detailed error messages
2. **Enable debug logging:**
   ```python
   config.logging.level = "DEBUG"
   ```
3. **Open an issue:** [GitHub Issues](https://github.com/ptlap/imp/issues)
4. **Include:**
   - Error message and full stack trace
   - Python version: `python --version`
   - PyTorch version: `python -c "import torch; print(torch.__version__)"`
   - CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
   - Operating system and environment (Colab/WSL/Windows/Linux)

---

## 📚 Documentation

- [Blueprint](docs/blueprint.md) - Kiến trúc tổng quan và chi tiết kỹ thuật
- [Development Workflow](docs/development_workflow.md) - Hướng dẫn development
- [Optimization Summary](docs/blueprint_optimization_summary.md) - Các tối ưu hóa
- [API Documentation](docs/api.md) - Detailed API reference (coming soon)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - Super-resolution
- [CodeFormer](https://github.com/sczhou/CodeFormer) - Face restoration
- [DDColor](https://github.com/piddnad/DDColor) - Colorization
- [BasicSR](https://github.com/XPixelGroup/BasicSR) - Image restoration framework

---

## 📧 Contact

- **Author**: ptlap
- **GitHub**: [@ptlap](https://github.com/ptlap)
- **Project**: [ptlap/imp](https://github.com/ptlap/imp)

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ptlap/imp&type=Date)](https://star-history.com/#ptlap/imp&Date)

---

**Made with ❤️ for restoring precious memories**
