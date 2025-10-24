# 🔄 DEVELOPMENT WORKFLOW

## Tổng quan

**Nguyên tắc chính**: 
- ✅ Code trên LOCAL
- ✅ Version control với Git
- ✅ Run/test trên COLAB (có GPU)
- ✅ Dùng PRE-TRAINED models (KHÔNG train!)

---

## 📁 Project Structure

```
imp/                             # LOCAL (máy tính của bạn)
├── .git/                        # Git repository
├── src/                         # Source code
│   ├── models/
│   │   ├── __init__.py
│   │   ├── denoiser.py
│   │   ├── super_resolution.py
│   │   ├── colorization.py
│   │   └── face_enhancement.py
│   ├── utils/
│   │   ├── image_io.py
│   │   ├── preprocessing.py
│   │   └── metrics.py
│   ├── pipeline.py              # Main pipeline
│   └── config.py
├── notebooks/                   # Jupyter notebooks cho Colab
│   ├── 01_quick_start.ipynb
│   ├── 02_full_pipeline.ipynb
│   └── 03_evaluation.ipynb
├── configs/
│   └── config.yaml
├── tests/
│   └── test_pipeline.py
├── docs/
│   ├── blueprint.md
│   └── README.md
├── requirements.txt
└── README.md
```

---

## 🔧 Setup Workflow

### **Step 1: Setup Local Environment**

```bash
# Trên máy tính của bạn (Windows/Mac/Linux)

# 1. Tạo project folder
mkdir imp
cd imp

# 2. Initialize git
git init

# 3. Create virtual environment (optional nhưng recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# 4. Install basic dependencies (cho development)
pip install opencv-python numpy pillow jupyter

# 5. Create project structure
mkdir -p src/models src/utils notebooks configs tests docs
touch src/__init__.py src/models/__init__.py src/utils/__init__.py
```

### **Step 2: Write Code Locally**

```python
# src/pipeline.py (viết trên máy local)

import cv2
import numpy as np
from typing import Optional, Dict

class OldPhotoRestoration:
    """Main restoration pipeline"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.models = {}
    
    def load_models(self):
        """Load pre-trained models (chỉ chạy trên Colab)"""
        # Import heavy libraries chỉ khi cần
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        
        # Load Real-ESRGAN
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                       num_block=23, num_grow_ch=32, scale=4)
        self.models['sr'] = RealESRGANer(
            scale=4,
            model_path='weights/realesrgan-x4plus.pth',
            model=model,
            tile=400,
            half=True
        )
    
    def restore(self, image_path: str) -> np.ndarray:
        """Restore old photo"""
        # Load image
        img = cv2.imread(image_path)
        
        # Denoise
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        # Super-resolution
        if 'sr' in self.models:
            output, _ = self.models['sr'].enhance(img, outscale=4)
        else:
            output = img
        
        return output

# Có thể test basic logic trên local (không cần GPU)
if __name__ == "__main__":
    pipeline = OldPhotoRestoration()
    # Test without loading heavy models
    print("Pipeline initialized successfully!")
```

### **Step 3: Push to GitHub**

```bash
# Trên máy local

# 1. Create .gitignore
cat > .gitignore << EOF
# Python
__pycache__/
*.pyc
*.pyo
venv/
.env

# Weights (KHÔNG commit weights, quá lớn!)
weights/
*.pth
*.ckpt

# Data
data/
results/
checkpoints/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
EOF

# 2. Commit code
git add .
git commit -m "Initial commit: project structure and pipeline"

# 3. Create GitHub repo (trên github.com)
# Tạo repo mới tên "imp"

# 4. Push to GitHub
git remote add origin https://github.com/ptlap/imp.git
git branch -M main
git push -u origin main
```

### **Step 4: Run on Colab**

```python
# notebooks/01_quick_start.ipynb (chạy trên Colab)

# Cell 1: Setup
!nvidia-smi  # Check GPU
!git clone https://github.com/ptlap/imp.git
%cd imp

# Cell 2: Install dependencies
!pip install -q torch torchvision
!pip install -q opencv-python-headless
!pip install -q basicsr realesrgan facexlib

# Cell 3: Download pre-trained weights
!mkdir -p weights
!wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-x4plus.pth -P weights/

# Cell 4: Import your code
import sys
sys.path.append('/content/imp/src')

from pipeline import OldPhotoRestoration

# Cell 5: Initialize and run
pipeline = OldPhotoRestoration()
pipeline.load_models()  # Load pre-trained weights

# Cell 6: Test on sample image
!wget https://example.com/old_photo.jpg -O test.jpg
result = pipeline.restore('test.jpg')

# Cell 7: Display results
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cv2.imread('test.jpg'), cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Restored')
plt.show()

# Cell 8: Save result
cv2.imwrite('restored.jpg', result)

# Cell 9: Download result
from google.colab import files
files.download('restored.jpg')
```

---

## 🔄 Development Cycle

```
┌─────────────────────────────────────────────────────┐
│  1. WRITE CODE (Local)                              │
│     - Edit Python files                             │
│     - Test basic logic (no GPU needed)              │
│     - Write documentation                           │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  2. COMMIT & PUSH (Local → GitHub)                  │
│     git add .                                       │
│     git commit -m "Add feature X"                   │
│     git push                                        │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  3. TEST ON COLAB (GitHub → Colab)                  │
│     - Open Colab notebook                           │
│     - Run: !git pull (nếu đã clone)                 │
│     - hoặc: !git clone (lần đầu)                    │
│     - Test với GPU                                  │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  4. FIX BUGS (nếu có)                               │
│     - Quay lại Local                                │
│     - Fix code                                      │
│     - Repeat cycle                                  │
└─────────────────────────────────────────────────────┘
```

---

## 💡 Best Practices

### **1. Separate Development & Execution**

```python
# src/pipeline.py (local development)
class Pipeline:
    def __init__(self):
        self.models = {}
    
    def load_models(self):
        """Heavy operation - only run on Colab"""
        pass
    
    def restore(self, image):
        """Main logic - can test locally"""
        pass

# notebooks/run_colab.ipynb (Colab execution)
# Import and run the pipeline
```

### **2. Use Configuration Files**

```yaml
# configs/config.yaml (commit to git)
models:
  super_resolution:
    type: "realesrgan"
    scale: 4
    weights_url: "https://github.com/.../realesrgan-x4plus.pth"
  
  colorization:
    type: "ddcolor"
    weights_url: "https://huggingface.co/.../ddcolor.pth"

processing:
  max_image_size: 2048
  tile_size: 512
  use_fp16: true
```

### **3. Modular Code**

```python
# src/models/base.py
class BaseModel:
    """Base class cho tất cả models"""
    def __init__(self, weights_path):
        self.weights_path = weights_path
    
    def load(self):
        raise NotImplementedError
    
    def process(self, image):
        raise NotImplementedError

# src/models/super_resolution.py
class SuperResolution(BaseModel):
    def load(self):
        # Load Real-ESRGAN
        pass
    
    def process(self, image):
        # Run inference
        pass
```

### **4. Testing Strategy**

```python
# tests/test_pipeline.py (chạy local, không cần GPU)
import unittest
from src.pipeline import OldPhotoRestoration

class TestPipeline(unittest.TestCase):
    def test_initialization(self):
        """Test pipeline can be initialized"""
        pipeline = OldPhotoRestoration()
        self.assertIsNotNone(pipeline)
    
    def test_config_loading(self):
        """Test config loading"""
        config = {'max_size': 2048}
        pipeline = OldPhotoRestoration(config)
        self.assertEqual(pipeline.config['max_size'], 2048)

# Run: python -m pytest tests/
```

---

## 🚫 KHÔNG NÊN LÀM

### ❌ **Commit weights vào Git**
```bash
# WRONG - weights quá lớn (100MB - 1GB)
git add weights/realesrgan-x4plus.pth  # ❌ KHÔNG!

# RIGHT - download khi cần
# Trong notebook Colab:
!wget https://github.com/.../weights.pth -P weights/  # ✅ ĐÚNG
```

### ❌ **Code trực tiếp trên Colab**
```python
# WRONG - code trên Colab, mất khi session end
# Viết code dài trong notebook cell  # ❌ KHÔNG!

# RIGHT - code trên local, import vào Colab
from src.pipeline import Pipeline  # ✅ ĐÚNG
```

### ❌ **Train models từ đầu**
```python
# WRONG - mất 2-4 tuần, cần nhiều data
for epoch in range(100):
    train_model()  # ❌ KHÔNG CẦN!

# RIGHT - dùng pre-trained
model.load_state_dict(torch.load('pretrained.pth'))  # ✅ ĐÚNG
```

---

## 📊 Comparison: Local vs Colab

| Task | Local | Colab | Why |
|------|-------|-------|-----|
| **Write code** | ✅ | ❌ | Editor tốt hơn (VSCode, PyCharm) |
| **Git operations** | ✅ | ⚠️ | Dễ quản lý hơn |
| **Test logic** | ✅ | ❌ | Không cần GPU cho basic tests |
| **Run inference** | ❌ | ✅ | Cần GPU (T4, V100) |
| **Process images** | ❌ | ✅ | Cần GPU |
| **Train models** | ❌ | ❌ | KHÔNG CẦN (dùng pre-trained) |
| **Documentation** | ✅ | ❌ | Markdown editor tốt hơn |
| **Debugging** | ✅ | ⚠️ | Debugger tốt hơn |

---

## 🎯 Typical Day Workflow

### **Morning (Local - 2 hours)**
```bash
# 1. Pull latest changes
git pull

# 2. Write new feature
# Edit src/models/colorization.py
# Add colorization support

# 3. Test locally (basic logic)
python tests/test_colorization.py

# 4. Commit
git add src/models/colorization.py
git commit -m "Add colorization module"
git push
```

### **Afternoon (Colab - 1 hour)**
```python
# 1. Open Colab notebook
# 2. Pull latest code
!cd imp && git pull

# 3. Test new feature with GPU
from src.models.colorization import Colorization
colorizer = Colorization()
colorizer.load()
result = colorizer.process(test_image)

# 4. If works: Great! If not: fix on local and repeat
```

### **Evening (Local - 1 hour)**
```bash
# 1. Update documentation
# Edit docs/README.md

# 2. Write report
# Edit docs/report.md

# 3. Commit
git add docs/
git commit -m "Update documentation"
git push
```

---

## 🔑 Key Takeaways

1. **Code trên LOCAL** - Editor tốt, Git dễ, không mất code
2. **Run trên COLAB** - Có GPU, free, không cần setup
3. **KHÔNG train** - Dùng pre-trained models
4. **Git là trung tâm** - Local ↔ GitHub ↔ Colab
5. **Modular code** - Dễ test, dễ maintain
6. **Config files** - Không hardcode paths/parameters
7. **Test locally** - Basic logic không cần GPU
8. **Process on Colab** - Heavy inference cần GPU

---

## 📚 Tools Recommended

### **Local Development:**
- **Editor**: VSCode (với Python extension)
- **Git GUI**: GitHub Desktop hoặc GitKraken
- **Terminal**: Windows Terminal / iTerm2
- **Python**: Python 3.8+ với venv

### **Colab:**
- **Colab Pro**: $10/month (optional nhưng recommended)
- **Google Drive**: Backup weights và results
- **Colab Notebooks**: Jupyter notebooks trên cloud

### **Version Control:**
- **GitHub**: Free (public repo) hoặc private
- **Git**: Command line hoặc GUI

---

**Tóm lại**: Bạn code trên máy local (như bình thường), push lên GitHub, rồi chạy trên Colab (có GPU). KHÔNG cần train gì cả, chỉ dùng pre-trained models! 🚀
