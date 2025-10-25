# 🎨 IMP - Image Restoration Project

Hệ thống phục chế ảnh cũ tự động sử dụng Deep Learning.

## 🚀 Quick Start

### Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ptlap/imp/blob/main/notebooks/01_quick_start.ipynb)

```python
# Clone và cài đặt
!git clone https://github.com/ptlap/imp.git
%cd imp
!pip install -q -r requirements.txt

# Sử dụng
from src.pipeline import OldPhotoRestoration

pipeline = OldPhotoRestoration()
restored = pipeline.restore('input.jpg', 'output.png')
```

### Local Setup

```bash
# Clone repository
git clone https://github.com/ptlap/imp.git
cd imp

# Tạo virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac/WSL
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy example
python examples/basic_usage.py
```

## 📦 Features

- 🧹 **Khử nhiễu** - Loại bỏ noise và scratches
- 🔍 **Tăng độ phân giải** - Super-resolution 2x/4x với Real-ESRGAN
- 🎨 **Tô màu** - Colorization cho ảnh đen trắng (coming soon)
- 👤 **Phục hồi khuôn mặt** - Face enhancement (coming soon)

## 💻 Usage

### Basic Usage

```python
from src.pipeline import OldPhotoRestoration

pipeline = OldPhotoRestoration()
restored = pipeline.restore('old_photo.jpg', 'restored.png')
```

### Batch Processing

```python
image_paths = ['photo1.jpg', 'photo2.jpg', 'photo3.jpg']
successes, failures = pipeline.batch_restore(
    image_paths=image_paths,
    output_dir='./restored_batch'
)
```

### Custom Configuration

```python
from src.config import Config

config = Config.default()
config.models.super_resolution.scale = 2  # 2x thay vì 4x
config.models.denoising.strength = 15

pipeline = OldPhotoRestoration(config)
restored = pipeline.restore('input.jpg', 'output.png')
```

## 📁 Project Structure

```
imp/
├── src/                    # Source code
│   ├── pipeline.py        # Main pipeline
│   ├── config.py          # Configuration
│   ├── models/            # Model implementations
│   └── utils/             # Utilities
├── examples/              # Usage examples
├── tests/                 # Unit tests
├── configs/               # Configuration files
└── requirements.txt       # Dependencies
```

## 🔧 Dependencies

- Python 3.8+
- PyTorch 2.0+
- OpenCV
- Real-ESRGAN
- BasicSR

Xem đầy đủ trong `requirements.txt`.

## 🛠️ Troubleshooting

### CUDA Out of Memory

```python
config = Config.default()
config.models.super_resolution.tile_size = 256
config.models.super_resolution.scale = 2
```

### Slow Processing

Sử dụng Google Colab với GPU miễn phí hoặc skip super-resolution:

```python
config.models.super_resolution.skip = True
```

## 📝 License

MIT License

## 🙏 Acknowledgments

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [BasicSR](https://github.com/XPixelGroup/BasicSR)

---

**Made with ❤️ for restoring precious memories**
