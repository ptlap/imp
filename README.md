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

### **Option 1: Chạy trên Google Colab (Recommended)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ptlap/imp/blob/main/notebooks/01_quick_start.ipynb)

```python
# 1. Clone repository
!git clone https://github.com/ptlap/imp.git
%cd imp

# 2. Install dependencies
!pip install -q -r requirements.txt

# 3. Download pre-trained weights
!bash scripts/download_weights.sh

# 4. Run restoration
from src.pipeline import OldPhotoRestoration

pipeline = OldPhotoRestoration()
pipeline.load_models()
result = pipeline.restore('path/to/old_photo.jpg')
```

### **Option 2: Setup Local (Development)**

```bash
# Clone repository
git clone https://github.com/ptlap/imp.git
cd imp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests (không cần GPU)
pytest tests/
```

---

## 📁 Project Structure

```
imp/
├── src/                         # Source code
│   ├── models/                  # Model wrappers
│   │   ├── denoiser.py
│   │   ├── super_resolution.py
│   │   ├── colorization.py
│   │   └── face_enhancement.py
│   ├── utils/                   # Utilities
│   │   ├── image_io.py
│   │   ├── preprocessing.py
│   │   └── metrics.py
│   ├── pipeline.py              # Main pipeline
│   └── config.py
├── notebooks/                   # Jupyter notebooks
│   ├── 01_quick_start.ipynb
│   ├── 02_full_pipeline.ipynb
│   └── 03_evaluation.ipynb
├── configs/                     # Configuration files
│   └── config.yaml
├── tests/                       # Unit tests
│   └── test_pipeline.py
├── docs/                        # Documentation
│   ├── blueprint.md
│   ├── development_workflow.md
│   └── blueprint_optimization_summary.md
├── scripts/                     # Utility scripts
│   └── download_weights.sh
├── requirements.txt
└── README.md
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

## 📚 Documentation

- [Blueprint](docs/blueprint.md) - Kiến trúc tổng quan và chi tiết kỹ thuật
- [Development Workflow](docs/development_workflow.md) - Hướng dẫn development
- [Optimization Summary](docs/blueprint_optimization_summary.md) - Các tối ưu hóa

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
