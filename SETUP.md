# IMP Setup Guide

Comprehensive setup instructions for the IMP (Image Restoration Project) across different environments.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Google Colab Setup](#google-colab-setup)
3. [WSL (Windows Subsystem for Linux) Setup](#wsl-setup)
4. [Linux Setup](#linux-setup)
5. [Windows Setup](#windows-setup)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### **All Environments**
- Python 3.8 or higher
- pip package manager
- Git

### **For GPU Acceleration (Optional)**
- CUDA-capable GPU (NVIDIA)
- CUDA Toolkit 11.0 or higher
- cuDNN 8.0 or higher

### **Check Python Version**
```bash
python --version  # or python3 --version
```

If Python is not installed or version is < 3.8, install from:
- **Linux/WSL**: `sudo apt-get install python3.8 python3-pip`
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **macOS**: `brew install python@3.8`

---

## Google Colab Setup

**Best for:** Quick start, GPU access, no local installation

### **Step 1: Open Colab Notebook**

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click `File` → `Open notebook` → `GitHub`
3. Enter: `ptlap/imp`
4. Select: `notebooks/01_quick_start.ipynb`

### **Step 2: Enable GPU**

1. Click `Runtime` → `Change runtime type`
2. Select `Hardware accelerator`: **T4 GPU** or **V100 GPU**
3. Click `Save`

### **Step 3: Install Dependencies**

Run these cells in order:

```python
# Cell 1: Clone repository
!git clone https://github.com/ptlap/imp.git
%cd imp

# Cell 2: Install dependencies
!pip install -q -r requirements.txt

# Cell 3: Verify installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### **Step 4: Run Restoration**

```python
from src.pipeline import OldPhotoRestoration
from src.config import Config

# Create pipeline
config = Config.default()
pipeline = OldPhotoRestoration(config)

# Upload image (use Colab's file upload)
from google.colab import files
uploaded = files.upload()

# Get uploaded filename
image_path = list(uploaded.keys())[0]

# Restore image
restored = pipeline.restore(image_path, 'restored.png')

# Display results
from PIL import Image
import matplotlib.pyplot as plt

original = Image.open(image_path)
restored_img = Image.fromarray((restored * 255).astype('uint8'))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(original)
axes[0].set_title('Original')
axes[0].axis('off')
axes[1].imshow(restored_img)
axes[1].set_title('Restored')
axes[1].axis('off')
plt.tight_layout()
plt.show()

# Download result
files.download('restored.png')
```

**Notes:**
- Colab sessions timeout after ~12 hours of inactivity
- Free tier has usage limits (check [Colab FAQ](https://research.google.com/colaboratory/faq.html))
- Model weights (~65MB) are downloaded automatically on first use

---

## WSL Setup

**Best for:** Windows users who want Linux environment

### **Step 1: Install WSL**

If WSL is not installed:

```powershell
# Open PowerShell as Administrator
wsl --install

# Restart computer
# After restart, set up Ubuntu username and password
```

### **Step 2: Update System**

```bash
# Open WSL terminal
sudo apt-get update
sudo apt-get upgrade -y
```

### **Step 3: Install Python and Dependencies**

```bash
# Install Python 3.8+
sudo apt-get install -y python3.8 python3-pip python3-venv

# Install system dependencies
sudo apt-get install -y git build-essential libopencv-dev

# Verify installation
python3 --version
pip3 --version
```

### **Step 4: Clone Repository**

```bash
# Navigate to home directory
cd ~

# Clone repository
git clone https://github.com/ptlap/imp.git
cd imp
```

### **Step 5: Create Virtual Environment**

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (should show path to venv)
which python
```

### **Step 6: Install Python Dependencies**

```bash
# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# This may take 5-10 minutes
```

### **Step 7: Verify Installation**

```bash
# Run verification script
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from src.config import Config; print('IMP installed successfully!')"
```

### **Step 8: Run Tests**

```bash
# Run all tests
pytest tests/ -v

# Should see all tests passing
```

### **WSL-Specific Notes**

- **File Access**: WSL files are at `\\wsl$\Ubuntu\home\<username>\imp` in Windows Explorer
- **GPU Support**: WSL2 supports CUDA (requires Windows 11 or Windows 10 21H2+)
- **Memory**: WSL uses up to 50% of system RAM by default

---

## Linux Setup

**Best for:** Native Linux users, servers

### **Step 1: Update System**

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### **Step 2: Install Python**

```bash
# Install Python 3.8+
sudo apt-get install -y python3.8 python3-pip python3-venv

# Install system dependencies
sudo apt-get install -y git build-essential libopencv-dev

# Verify
python3 --version
```

### **Step 3: Clone Repository**

```bash
git clone https://github.com/ptlap/imp.git
cd imp
```

### **Step 4: Create Virtual Environment**

```bash
# Create venv
python3 -m venv venv

# Activate
source venv/bin/activate

# Verify
which python  # Should show venv/bin/python
```

### **Step 5: Install Dependencies**

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### **Step 6: (Optional) Install CUDA for GPU**

If you have an NVIDIA GPU:

```bash
# Check GPU
nvidia-smi

# Install CUDA Toolkit (example for CUDA 11.8)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Verify CUDA
nvcc --version
```

### **Step 7: Verify Installation**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
pytest tests/ -v
```

---

## Windows Setup

**Best for:** Native Windows users (without WSL)

### **Step 1: Install Python**

1. Download Python 3.8+ from [python.org](https://www.python.org/downloads/)
2. Run installer
3. **Important**: Check "Add Python to PATH"
4. Click "Install Now"

### **Step 2: Install Git**

1. Download Git from [git-scm.com](https://git-scm.com/download/win)
2. Run installer with default settings

### **Step 3: Clone Repository**

```cmd
# Open Command Prompt or PowerShell
cd %USERPROFILE%\Documents
git clone https://github.com/ptlap/imp.git
cd imp
```

### **Step 4: Create Virtual Environment**

```cmd
# Create venv
python -m venv venv

# Activate
venv\Scripts\activate

# Verify (should show venv\Scripts\python.exe)
where python
```

### **Step 5: Install Dependencies**

```cmd
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### **Step 6: Verify Installation**

```cmd
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
pytest tests\ -v
```

### **Windows-Specific Notes**

- Use `\` instead of `/` for paths
- Use `venv\Scripts\activate` instead of `source venv/bin/activate`
- GPU support requires CUDA-capable GPU and CUDA Toolkit

---

## Verification

### **Quick Verification**

```bash
# Activate virtual environment first
source venv/bin/activate  # Linux/WSL/Mac
# or
venv\Scripts\activate  # Windows

# Run verification
python -c "
from src.pipeline import OldPhotoRestoration
from src.config import Config
print('✓ IMP installed successfully!')
print('✓ All modules imported correctly')
config = Config.default()
print('✓ Configuration loaded')
pipeline = OldPhotoRestoration(config)
print('✓ Pipeline initialized')
print('\nReady to restore images!')
"
```

### **Run Tests**

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### **Test Basic Functionality**

```bash
# Run example script
python examples/basic_usage.py

# Or test interactively
python
>>> from src.pipeline import OldPhotoRestoration
>>> pipeline = OldPhotoRestoration()
>>> print("Pipeline ready!")
```

---

## Troubleshooting

### **Issue: `python` command not found**

**Solution:**
```bash
# Try python3 instead
python3 --version

# Or create alias (Linux/WSL)
echo "alias python=python3" >> ~/.bashrc
source ~/.bashrc
```

### **Issue: `pip` command not found**

**Solution:**
```bash
# Use python -m pip instead
python -m pip --version

# Or install pip
sudo apt-get install python3-pip  # Linux/WSL
```

### **Issue: Permission denied when installing packages**

**Solution:**
```bash
# Don't use sudo with pip in venv
# Make sure venv is activated
source venv/bin/activate

# Then install
pip install -r requirements.txt
```

### **Issue: Virtual environment not activating**

**Solution:**
```bash
# Linux/WSL: Ensure venv module is installed
sudo apt-get install python3-venv

# Recreate venv
rm -rf venv
python3 -m venv venv
source venv/bin/activate
```

### **Issue: Tests failing**

**Solution:**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Run tests with verbose output
pytest tests/ -v -s

# Check specific failing test
pytest tests/test_pipeline.py::test_name -v
```

### **Issue: Import errors**

**Solution:**
```bash
# Ensure you're in the project root directory
cd /path/to/imp

# Ensure venv is activated
source venv/bin/activate

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### **Issue: CUDA not available**

**Solution:**
```bash
# Check GPU
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Next Steps

After successful setup:

1. **Read the README**: `cat README.md` or open in browser
2. **Try examples**: `python examples/basic_usage.py`
3. **Read documentation**: Check `docs/` directory
4. **Start restoring**: See usage guide in README.md

---

## Getting Help

If you encounter issues:

1. Check this troubleshooting section
2. Review logs: `cat imp.log`
3. Search existing issues: [GitHub Issues](https://github.com/ptlap/imp/issues)
4. Open new issue with:
   - Error message
   - Python version: `python --version`
   - OS and environment
   - Steps to reproduce

---

**Happy Restoring! 🎨**
