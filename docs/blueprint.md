# 🎨 BLUEPRINT ĐỒ ÁN: AI PHỤC CHẾ ẢNH CŨ TỰ ĐỘNG

## 📋 THÔNG TIN DỰ ÁN

**Tên đồ án**: Hệ thống phục chế ảnh cũ tự động sử dụng Deep Learning  
**Công nghệ**: Python, PyTorch, Google Colab  
**Thời gian**: 3-4 tháng  
**Mục tiêu**: 
- Khử nhiễu và loại bỏ artifacts
- Tăng độ phân giải (Super-Resolution)
- Tô màu ảnh đen trắng (Colorization)
- Phục hồi khuôn mặt mờ (Face Enhancement)

---

## 🚀 QUICK START (Bắt đầu ngay trong 30 phút!)

### **Step 1: Setup Colab Notebook**
```python
# 1. Tạo notebook mới trên Google Colab
# 2. Enable GPU: Runtime → Change runtime type → GPU (T4)
# 3. Run cells sau:

# Install dependencies
!pip install -q torch torchvision
!pip install -q opencv-python-headless
!pip install -q basicsr realesrgan
!pip install -q facexlib
!pip install -q gradio

# Clone repositories
!git clone https://github.com/xinntao/Real-ESRGAN.git
!git clone https://github.com/sczhou/CodeFormer.git

# Download weights (chọn 1 trong 2 cách)
# Cách 1: Từ GitHub releases
!wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-x4plus.pth -P weights/

# Cách 2: Từ Google Drive (nếu bạn đã backup)
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/weights/*.pth ./weights/
```

### **Step 2: Test Individual Models (15 phút)**
```python
# Test Real-ESRGAN
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2

# Load model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path='weights/realesrgan-x4plus.pth',
    model=model,
    tile=400,
    tile_pad=10,
    pre_pad=0,
    half=True
)

# Test on sample image
img = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
output, _ = upsampler.enhance(img, outscale=4)
cv2.imwrite('output_sr.jpg', output)

print("✓ Super-Resolution works!")
```

### **Step 3: Build Minimal Pipeline (10 phút)**
```python
class MinimalPipeline:
    """Minimal working pipeline - 50 lines of code"""
    
    def __init__(self):
        # Load Real-ESRGAN
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                       num_block=23, num_grow_ch=32, scale=4)
        self.upsampler = RealESRGANer(
            scale=4,
            model_path='weights/realesrgan-x4plus.pth',
            model=model,
            tile=400,
            half=True
        )
    
    def restore(self, image_path):
        """Minimal restoration: denoise + super-resolution"""
        import cv2
        
        # Load
        img = cv2.imread(image_path)
        
        # Denoise (OpenCV)
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        # Super-resolution
        output, _ = self.upsampler.enhance(img, outscale=4)
        
        return output

# Test
pipeline = MinimalPipeline()
result = pipeline.restore('old_photo.jpg')
cv2.imwrite('restored.jpg', result)

print("✓ Minimal pipeline works! You have a working MVP!")
```

### **Step 4: Add Gradio Demo (5 phút)**
```python
import gradio as gr

def restore_ui(image):
    """Gradio interface function"""
    # Save temp
    cv2.imwrite('temp.jpg', image)
    
    # Restore
    result = pipeline.restore('temp.jpg')
    
    return result

# Create interface
demo = gr.Interface(
    fn=restore_ui,
    inputs=gr.Image(label="Upload Old Photo"),
    outputs=gr.Image(label="Restored"),
    title="🎨 AI Photo Restoration",
    description="Upload an old photo and restore it!"
)

# Launch
demo.launch(share=True)  # Creates public link!
```

**🎉 Congratulations! Bạn đã có:**
- ✅ Working pipeline (denoise + super-resolution)
- ✅ Public demo link để share với bạn bè
- ✅ Foundation để build thêm features

**⏭️ Next Steps:**
1. Add colorization (Week 4)
2. Add face enhancement (Week 5)
3. Optimize performance (Week 7)
4. Collect results for report (Week 9)

---

## 🏗️ KIẾN TRÚC TỔNG QUAN

```
┌─────────────────────────────────────────────────────────┐
│                     INPUT IMAGE                          │
│                  (Old/Degraded Photo)                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              MODULE 0: PREPROCESSING                     │
│  - Image loading & validation                            │
│  - Grayscale detection                                   │
│  - Initial resize & normalization                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         MODULE 1: DENOISING & ARTIFACT REMOVAL           │
│  Model: NAFNet hoặc Restormer                            │
│  - Remove Gaussian/Poisson noise                         │
│  - Remove JPEG compression artifacts                     │
│  - Remove scratches & stains                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              MODULE 2: SUPER-RESOLUTION                  │
│  Model: Real-ESRGAN hoặc SwinIR                          │
│  - Upscale 2x/4x                                         │
│  - Enhance details & textures                            │
│  - Reduce blur                                           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
                ┌────┴────┐
                │  Check  │
                │Grayscale│
                └────┬────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
    [IF GRAYSCALE]        [IF COLOR]
          │                     │
          ▼                     │
┌──────────────────────┐        │
│   MODULE 3:          │        │
│   COLORIZATION       │        │
│ Model: DDColor       │        │
│ - AI-based coloring  │        │
└──────────┬───────────┘        │
           │                    │
           └─────────┬──────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              MODULE 4: FACE DETECTION                    │
│  Model: MTCNN hoặc RetinaFace                            │
│  - Detect all faces in image                             │
│  - Extract face regions with padding                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
                ┌────┴────┐
                │ Faces?  │
                └────┬────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
    [IF FACES]            [IF NO FACES]
          │                     │
          ▼                     │
┌──────────────────────┐        │
│   MODULE 5:          │        │
│   FACE ENHANCEMENT   │        │
│ Model: CodeFormer    │        │
│ - Restore face       │        │
│ - Enhance eyes/mouth │        │
│ - Paste back         │        │
└──────────┬───────────┘        │
           │                    │
           └─────────┬──────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              MODULE 6: POST-PROCESSING                   │
│  - Color correction                                      │
│  - Sharpening                                            │
│  - Contrast adjustment                                   │
│  - Final quality check                                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  OUTPUT IMAGE                            │
│                 (Restored Photo)                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 CHI TIẾT KỸ THUẬT TỪNG MODULE

### **MODULE 0: PREPROCESSING**

**Mục đích**: Chuẩn bị ảnh đầu vào cho các module xử lý

**Thuật toán**:
```python
Input: image_path (str)
Output: preprocessed_image (numpy.ndarray), metadata (dict)

Steps:
1. Load image using PIL/OpenCV
2. Convert to RGB if needed
3. Check if grayscale (compare R=G=B channels)
4. Store original dimensions
5. Resize if too large (max dimension = 1024px for memory)
6. Normalize pixel values to [0, 1]
7. Convert to tensor format
```

**Libraries**:
- `PIL (Pillow)` - Image loading
- `OpenCV (cv2)` - Image processing
- `numpy` - Array operations

**Code structure**:
```python
class Preprocessor:
    def __init__(self, max_size=1024):
        self.max_size = max_size
    
    def load_image(self, path):
        # Load and validate
        pass
    
    def detect_grayscale(self, image):
        # Check if B&W
        pass
    
    def resize_if_needed(self, image):
        # Smart resize keeping aspect ratio
        pass
    
    def normalize(self, image):
        # Scale to [0,1]
        pass
```

**Input**: JPG/PNG file (any size)  
**Output**: Tensor shape `[1, 3, H, W]` + metadata dict

---

### **MODULE 1: DENOISING & ARTIFACT REMOVAL**

**Mục đích**: Loại bỏ nhiễu, scratches, JPEG artifacts

**🎯 RECOMMENDED APPROACH (Simplified)**:

**Option A: OpenCV FastNlMeans (BEST for MVP)**
- **Pros**: 
  - ✅ No GPU needed
  - ✅ Fast (< 0.5s for 512x512)
  - ✅ No model weights to download
  - ✅ Good enough for most cases
- **Cons**: 
  - ❌ Not as powerful as deep learning
  - ❌ May over-smooth details
- **Code**:
```python
def denoise_opencv(image, strength=10):
    """Fast denoising using OpenCV"""
    return cv2.fastNlMeansDenoisingColored(
        image, 
        None, 
        h=strength,  # Luminance strength
        hColor=strength,  # Color strength
        templateWindowSize=7,
        searchWindowSize=21
    )
```

**Option B: NAFNet (If you need SOTA quality)**
- **Paper**: "Simple Baselines for Image Restoration" (ECCV 2022)
- **Architecture**: Nonlinear Activation Free Network
- **Params**: ~68M (NAFNet-width64)
- **Pre-trained**: Available on GitHub
- **Pros**: State-of-the-art quality
- **Cons**: Heavy (3GB GPU memory), slower (1-2s)
- **Use when**: You have Colab Pro or processing small batches

**Option C: Restormer (Alternative)**
- **Paper**: "Restormer: Efficient Transformer for High-Resolution Image Restoration" (CVPR 2022)
- **Architecture**: Transformer-based
- **Params**: ~26M
- **Pre-trained**: Available on GitHub
- **Pros**: Lighter than NAFNet
- **Cons**: Still requires GPU

**💡 RECOMMENDATION**: 
- Start with OpenCV for MVP
- Add NAFNet as optional "quality mode" later
- Let users choose: Fast (OpenCV) vs Quality (NAFNet)

**Chi tiết kỹ thuật NAFNet**:

```
Architecture:
- Encoder-Decoder with U-Net structure
- 4 scales (downsample by 2 each level)
- Skip connections between encoder-decoder
- No ReLU/GELU (activation-free)
- Uses SimpleGate: X -> X1, X2 -> X1 ⊙ X2

Key Components:
1. NAFBlock:
   - LayerNorm
   - SimpleGate
   - SimpleChannelAttention (SCA)
   - Skip connection

2. SimpleChannelAttention:
   - Global Average Pooling
   - Conv 1x1
   - Sigmoid activation
```

**Training strategy** (nếu fine-tune):
```yaml
Dataset: 
  - SIDD (real noise)
  - DIV2K + synthetic noise
  
Augmentation:
  - Random crop 256x256
  - Random flip/rotate
  - Add noise: Gaussian σ=[0,50], Poisson λ=[0,30]
  - JPEG compression quality [30,100]
  
Loss function: 
  - L1 loss (Charbonnier loss)
  - Perceptual loss (optional)

Optimizer: AdamW
Learning rate: 2e-4 with Cosine Annealing
Batch size: 16 (hoặc 8 trên Colab)
Epochs: 200
```

**Inference**:
```python
import torch
from nafnet import NAFNet

# Load pre-trained model
model = NAFNet(width=64, num_blks=[1,1,1,28])
model.load_state_dict(torch.load('nafnet_weights.pth'))
model.eval().cuda()

# Denoise
with torch.no_grad():
    denoised = model(noisy_image_tensor)
```

**Input**: `[B, 3, H, W]` noisy image  
**Output**: `[B, 3, H, W]` clean image

---

### **MODULE 2: SUPER-RESOLUTION**

**Mục đích**: Tăng độ phân giải 2x hoặc 4x, làm rõ chi tiết

**Model Option A: Real-ESRGAN (Recommended)**
- **Paper**: "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data" (ICCV 2021)
- **Architecture**: ESRGAN + Real degradation model
- **Scale**: 2x hoặc 4x
- **Pre-trained**: Multiple versions available

**Model Option B: SwinIR**
- **Paper**: "SwinIR: Image Restoration Using Swin Transformer" (ICCV 2021)
- **Architecture**: Swin Transformer blocks
- **Stronger**: Nhưng chậm hơn Real-ESRGAN

**Chi tiết Real-ESRGAN**:

```
Architecture:
- Generator: RRDB (Residual in Residual Dense Block)
- No Discriminator at inference

Components:
1. Shallow Feature Extraction:
   - Conv 3x3
   
2. Deep Feature Extraction:
   - 23x RRDB blocks
   - Each RRDB has 3x Dense blocks
   - Residual scaling β=0.2
   
3. Upsampling:
   - PixelShuffle for 2x (or 2x PixelShuffle for 4x)
   - Conv 3x3
   
4. Reconstruction:
   - Conv 3x3
   - LeakyReLU
   - Conv 3x3
```

**Degradation model** (cho training):
```
1st degradation:
  - Blur: isotropic/anisotropic Gaussian kernel
  - Downsample: bilinear/bicubic/area
  - Noise: Gaussian + Poisson
  - JPEG compression

2nd degradation (repeat above)

Sinc filter (final degradation)
```

**Training** (nếu muốn fine-tune):
```yaml
Dataset: DIV2K, Flickr2K (800 images HR)

HR patch size: 256x256
LR patch size: 64x64 (for 4x SR)

Augmentation:
  - Random flip/rotate
  - Color jitter

Loss:
  - L1 loss: weight=1.0
  - Perceptual loss (VGG19): weight=1.0
  - GAN loss (optional): weight=0.1

Optimizer: Adam
Learning rate: 2e-4 → 1e-7 (decay)
Batch size: 16
Iterations: 400k
```

**Inference**:
```python
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Load model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                num_block=23, num_grow_ch=32, scale=4)

upsampler = RealESRGANer(
    scale=4,
    model_path='RealESRGAN_x4plus.pth',
    model=model,
    tile=400,  # Tile size for large images
    tile_pad=10,
    pre_pad=0,
    half=True  # FP16 for speed
)

# Upscale
output, _ = upsampler.enhance(input_image, outscale=4)
```

**Input**: `[H, W, 3]` low-res BGR image (OpenCV format)  
**Output**: `[4H, 4W, 3]` high-res BGR image

---

### **MODULE 3: COLORIZATION**

**Mục đích**: Tô màu tự động cho ảnh đen trắng

**Model: DDColor (SOTA 2023)**
- **Paper**: "DDColor: Towards Photorealistic Image Colorization via Dual Decoders" (ICCV 2023)
- **Architecture**: Dual-decoder với multi-scale features
- **Pre-trained**: Available on ModelScope/HuggingFace

**Chi tiết DDColor**:

```
Architecture:
┌────────────────────────┐
│   Input Grayscale      │
│      (1 channel)       │
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│   Encoder (ConvNeXt)   │
│   - Extract features   │
│   - Multi-scale        │
└──────────┬─────────────┘
           │
           ├──────────────┬─────────────┐
           │              │             │
           ▼              ▼             ▼
    ┌──────────┐   ┌──────────┐  ┌──────────┐
    │Decoder 1 │   │Decoder 2 │  │Query-based│
    │(Pixel)   │   │(Semantic)│  │Selection  │
    └────┬─────┘   └────┬─────┘  └─────┬────┘
         │              │               │
         └──────┬───────┴───────────────┘
                │
                ▼
        ┌───────────────┐
        │  Color Image  │
        │  (ab channels)│
        └───────────────┘
```

**Key features**:
1. **Dual Decoders**:
   - Pixel decoder: Local color details
   - Semantic decoder: Global color semantics
   
2. **Query-based Color Selection**:
   - Learnable color queries
   - Cross-attention with image features
   
3. **Lab Color Space**:
   - Input: L channel (lightness)
   - Output: ab channels (color)
   - Combine → RGB

**Training** (nếu fine-tune):
```yaml
Dataset: ImageNet (1.3M images)
  - Convert RGB → Lab
  - Use L as input, predict ab

Augmentation:
  - Random crop 256x256
  - Random flip
  - Color jitter on original (for harder cases)

Loss:
  - Smooth L1 loss on ab channels
  - Perceptual loss (optional)

Optimizer: AdamW
Learning rate: 1e-4
Batch size: 32
Epochs: 100
```

**Inference**:
```python
from ddcolor import DDColorModel
import torch
from PIL import Image
import numpy as np

# Load model
model = DDColorModel.from_pretrained('modelscope/ddcolor_paper')
model.eval().cuda()

# Prepare grayscale input
gray_img = Image.open('bw_photo.jpg').convert('L')
gray_tensor = transforms.ToTensor()(gray_img).unsqueeze(0).cuda()

# Colorize
with torch.no_grad():
    colored_tensor = model(gray_tensor)

# Convert to RGB
colored_img = tensor_to_rgb(colored_tensor)
```

**Input**: `[B, 1, H, W]` grayscale image  
**Output**: `[B, 3, H, W]` colorized RGB image

---

### **MODULE 4: FACE DETECTION**

**Mục đích**: Phát hiện khuôn mặt để xử lý riêng

**Model Option A: RetinaFace (Recommended)**
- **Paper**: "RetinaFace: Single-stage Dense Face Localisation in the Wild" (CVPR 2020)
- **Accuracy**: State-of-the-art
- **Speed**: Fast (30fps on GPU)

**Model Option B: MTCNN**
- **Paper**: "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks" (2016)
- **Advantage**: Đơn giản, có sẵn nhiều implementation

**Chi tiết RetinaFace**:

```
Architecture:
- Backbone: ResNet50 or MobileNet0.25
- FPN (Feature Pyramid Network)
- Multi-task branches:
  1. Face classification
  2. Bounding box regression
  3. 5 facial landmarks (eyes, nose, mouth corners)

Output:
- Bounding boxes: [x1, y1, x2, y2]
- Confidence scores: [0, 1]
- 5 landmarks: [(x,y)] × 5
```

**Inference**:
```python
from retinaface import RetinaFace

# Detect faces
faces = RetinaFace.detect_faces(image_path)

# faces = {
#   'face_1': {
#     'facial_area': [x1, y1, x2, y2],
#     'score': 0.99,
#     'landmarks': {
#       'left_eye': [x, y],
#       'right_eye': [x, y],
#       ...
#     }
#   }
# }

# Extract face regions với padding
for face_key, face_data in faces.items():
    x1, y1, x2, y2 = face_data['facial_area']
    
    # Add 30% padding
    w, h = x2-x1, y2-y1
    pad_w, pad_h = int(w*0.3), int(h*0.3)
    
    x1 = max(0, x1-pad_w)
    y1 = max(0, y1-pad_h)
    x2 = min(img_width, x2+pad_w)
    y2 = min(img_height, y2+pad_h)
    
    face_crop = image[y1:y2, x1:x2]
```

**Input**: RGB image (any size)  
**Output**: List of face bounding boxes + landmarks

---

### **MODULE 5: FACE ENHANCEMENT**

**Mục đích**: Phục hồi chi tiết khuôn mặt (mắt, mũi, miệng)

**Model: CodeFormer (SOTA 2022)**
- **Paper**: "Towards Robust Blind Face Restoration with Codebook Lookup Transformer" (NeurIPS 2022)
- **Key Innovation**: Discrete codebook để học face priors
- **Balance**: Quality vs Fidelity (controllable)

**Chi tiết CodeFormer**:

```
Architecture:
┌─────────────────────────────────────┐
│      Input Degraded Face            │
│         (Low quality)               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    Encoder (ConvNeXt-based)         │
│    - Extract features               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    Code Prediction Network          │
│    - Predict codebook indices       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    Codebook Lookup                  │
│    - 1024 learned code entries      │
│    - Each code = 256-dim vector     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    Transformer Decoder              │
│    - 9 Transformer blocks           │
│    - Cross-attention with codes     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    Output Decoder                   │
│    - Upsample to original size      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    Restored Face (High quality)     │
└─────────────────────────────────────┘
```

**Key Components**:

1. **Codebook**:
   - Size: 1024 entries
   - Dimension: 256 per entry
   - Learned from FFHQ dataset
   - Captures common face patterns

2. **Fidelity control** (w parameter):
   - w=0: Maximum quality (more hallucination)
   - w=1: Maximum fidelity (closer to input)
   - Recommended: w=0.5-0.7

3. **Training losses**:
   - Reconstruction loss (L1)
   - Perceptual loss (LPIPS)
   - Adversarial loss (GAN)
   - Identity loss (ArcFace)

**Training details**:
```yaml
Dataset: FFHQ (70k high-quality faces)
  - Synthetic degradation:
    - Blur kernels
    - Downsampling
    - Noise
    - JPEG compression
  
Image size: 512x512

Augmentation:
  - Random flip
  - Color jitter
  
Total loss:
  L = L_rec + λ_per*L_per + λ_adv*L_adv + λ_id*L_id
  λ_per = 1.0, λ_adv = 0.1, λ_id = 1.0

Optimizer: Adam
Learning rate: 2e-4 → 1e-6
Batch size: 16
Iterations: 800k (~2 weeks on 8x V100)
```

**Inference**:
```python
from codeformer import CodeFormer
import torch

# Load model
model = CodeFormer(
    dim_embd=512,
    codebook_size=1024,
    n_head=8,
    n_layers=9,
    connect_list=['32', '64', '128', '256']
).cuda()
model.load_state_dict(torch.load('codeformer.pth'))
model.eval()

# Enhance face
with torch.no_grad():
    # face_tensor: [1, 3, 512, 512]
    restored_face = model(
        face_tensor,
        w=0.7,  # Fidelity weight
        detach_16=True  # For stable training
    )[0]

# Paste back to original image
# Use Poisson blending for seamless integration
import cv2
mask = np.ones(face_crop.shape[:2], dtype=np.uint8) * 255
center = ((x1+x2)//2, (y1+y2)//2)
result = cv2.seamlessClone(
    restored_face, 
    original_image, 
    mask, 
    center, 
    cv2.NORMAL_CLONE
)
```

**Alternative: GFPGAN**
- Similar performance
- Simpler architecture
- Pre-trained weights available
- Code: `from gfpgan import GFPGANer`

**Input**: `[B, 3, 512, 512]` degraded face crop  
**Output**: `[B, 3, 512, 512]` restored face crop

---

### **MODULE 6: POST-PROCESSING**

**Mục đích**: Tinh chỉnh cuối cùng để ảnh tự nhiên

**Operations**:

1. **Color Correction**:
```python
# White balance
def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * 0.8)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * 0.8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

# Auto levels (histogram stretching)
def auto_levels(img):
    for i in range(3):
        hist, bins = np.histogram(img[:,:,i].flatten(), 256, [0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]
        
        # Find 1% and 99% percentiles
        low = np.searchsorted(cdf_normalized, 0.01)
        high = np.searchsorted(cdf_normalized, 0.99)
        
        # Stretch
        img[:,:,i] = np.clip((img[:,:,i] - low) * 255 / (high - low), 0, 255)
    return img
```

2. **Sharpening** (Unsharp Mask):
```python
def unsharp_mask(img, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(img, (0,0), sigma)
    sharpened = cv2.addWeighted(img, 1+strength, blurred, -strength, 0)
    return sharpened
```

3. **Contrast Enhancement** (CLAHE):
```python
def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    lab = cv2.merge([l,a,b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
```

4. **Noise Reduction** (optional, nhẹ):
```python
def denoise_final(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
```

**Pipeline**:
```python
def post_process(image, params):
    # 1. White balance
    if params.get('white_balance', True):
        image = white_balance(image)
    
    # 2. Auto levels
    if params.get('auto_levels', True):
        image = auto_levels(image)
    
    # 3. Sharpen
    sharpen_strength = params.get('sharpen', 1.0)
    if sharpen_strength > 0:
        image = unsharp_mask(image, strength=sharpen_strength)
    
    # 4. Contrast
    if params.get('enhance_contrast', True):
        image = enhance_contrast(image)
    
    # 5. Final denoise (optional)
    if params.get('final_denoise', False):
        image = denoise_final(image)
    
    return image
```

---

## 📊 DATASETS

### **1. Training Datasets**

**Denoising**:
- **SIDD** (Smartphone Image Denoising Dataset)
  - 30,000 noisy images từ 10 smartphone khác nhau
  - Real-world noise
  - Download: https://www.eecs.yorku.ca/~kamel/sidd/
  
- **DIV2K** (800 high-quality images)
  - Thêm synthetic noise
  - Download: https://data.vision.ee.ethz.ch/cvl/DIV2K/

**Super-Resolution**:
- **DIV2K**: HR images (2K resolution)
- **Flickr2K**: 2,650 HR images
- **FFHQ** (nếu focus vào faces): 70,000 faces 1024x1024

**Colorization**:
- **ImageNet**: 1.3M images (convert to grayscale)
- **COCO**: 330K images đa dạng

**Face Restoration**:
- **FFHQ**: 70,000 high-quality faces
- **CelebA-HQ**: 30,000 celebrity faces
- **Synthetic degradation** trên datasets trên

### **2. Test Datasets**

**Old Photos Dataset**:
- **Real-Old**: ~10K ảnh cũ thật từ 1900-1980s
  - Link: Tìm trên GitHub "old photos dataset"
  
- **Tự thu thập**:
  - Archive.org
  - Vintage photo forums
  - Ảnh gia đình cá nhân (xin phép)
  - r/estoration trên Reddit

**Benchmark Datasets**:
- **Set5, Set14, BSD100**: Standard SR benchmarks
- **Urban100**: Complex structures
- **Manga109**: Để test colorization

### **3. Synthetic Degradation Pipeline**

```python
def degrade_image(clean_img):
    """
    Tạo ảnh degraded từ ảnh sạch để training
    """
    img = clean_img.copy()
    
    # 1. Blur
    kernel_size = np.random.choice([7, 9, 11, 13, 15])
    sigma = np.random.uniform(0.5, 3.0)
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    
    # 2. Downsample (simulate low resolution)
    scale = np.random.choice([2, 3, 4])
    h, w = img.shape[:2]
    img = cv2.resize(img, (w//scale, h//scale), 
                     interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, (w, h), 
                     interpolation=cv2.INTER_CUBIC)
    
    # 3. Add noise
    noise_type = np.random.choice(['gaussian', 'poisson', 'mixed'])
    if noise_type == 'gaussian':
        sigma_noise = np.random.uniform(5, 50)
        noise = np.random.normal(0, sigma_noise, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
    elif noise_type == 'poisson':
        img = np.random.poisson(img).astype(np.uint8)
    else:
        # Mixed
        sigma_noise = np.random.uniform(3, 25)
        noise = np.random.normal(0, sigma_noise, img.shape)
        img = np.clip(img + noise, 0, 255)
        img = np.random.poisson(img).astype(np.uint8)
    
    # 4. JPEG compression artifacts
    quality = np.random.randint(30, 90)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    img = cv2.imdecode(encimg, 1)
    
    # 5. Scratches (random lines)
    if np.random.random() < 0.3:  # 30% chance
        num_scratches = np.random.randint(1, 5)
        for _ in range(num_scratches):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
            thickness = np.random.randint(1, 3)
            color = np.random.randint(0, 256)
            cv2.line(img, (x1,y1), (x2,y2), (color,color,color), thickness)
    
    # 6. Stains (random blobs)
    if np.random.random() < 0.2:  # 20% chance
        num_stains = np.random.randint(1, 3)
        for _ in range(num_stains):
            center = (np.random.randint(0, w), np.random.randint(0, h))
            radius = np.random.randint(10, 50)
            color = np.random.randint(0, 256)
            alpha = np.random.uniform(0.3, 0.7)
            overlay = img.copy()
            cv2.circle(overlay, center, radius, (color,color,color), -1)
            img = cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)
    
    return img
```

---

## 💻 IMPLEMENTATION STRUCTURE

### **Project Structure**
```
old_photo_restoration/
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_denoising.ipynb
│   ├── 03_super_resolution.ipynb
│   ├── 04_colorization.ipynb
│   ├── 05_face_enhancement.ipynb
│   ├── 06_full_pipeline.ipynb
│   └── 07_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── denoising.py
│   ├── super_resolution.py
│   ├── colorization.py
│   ├── face_detection.py
│   ├── face_enhancement.py
│   ├── post_processing.py
│   ├── pipeline.py
│   └── utils.py
│
├── models/
│   ├── nafnet/
│   │   ├── __init__.py
│   │   ├── nafnet.py
│   │   └── weights/
│   ├── realesrgan/
│   ├── ddcolor/
│   ├── codeformer/
│   └── download_weights.sh
│
├── data/
│   ├── train/
│   │   ├── clean/
│   │   └── degraded/
│   ├── test/
│   │   └── old_photos/
│   └── results/
│
├── configs/
│   ├── denoising.yaml
│   ├── sr.yaml
│   ├── colorization.yaml
│   └── pipeline.yaml
│
├── requirements.txt
├── setup.py
└── README.md
```

### **Main Pipeline Code**

```python
# src/pipeline.py

import torch
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

from .preprocessing import Preprocessor
from .denoising import DenoisingModel
from .super_resolution import SuperResolutionModel
from .colorization import ColorizationModel
from .face_detection import FaceDetector
from .face_enhancement import FaceEnhancer
from .post_processing import PostProcessor

class OldPhotoRestoration:
    """
    Main pipeline for old photo restoration
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        denoising_model: str = 'nafnet',
        sr_model: str = 'realesrgan',
        sr_scale: int = 4,
        colorization_model: str = 'ddcolor',
        face_model: str = 'codeformer',
        weights_dir: str = './models'
    ):
        self.device = device
        self.weights_dir = Path(weights_dir)
        
        # Initialize modules
        print("Loading models...")
        self.preprocessor = Preprocessor()
        
        self.denoiser = DenoisingModel(
            model_name=denoising_model,
            weights_path=self.weights_dir / denoising_model / 'weights.pth',
            device=device
        )
        
        self.super_resolver = SuperResolutionModel(
            model_name=sr_model,
            scale=sr_scale,
            weights_path=self.weights_dir / sr_model / 'weights.pth',
            device=device
        )
        
        self.colorizer = ColorizationModel(
            model_name=colorization_model,
            weights_path=self.weights_dir / colorization_model / 'weights.pth',
            device=device
        )
        
        self.face_detector = FaceDetector(device=device)
        
        self.face_enhancer = FaceEnhancer(
            model_name=face_model,
            weights_path=self.weights_dir / face_model / 'weights.pth',
            device=device
        )
        
        self.post_processor = PostProcessor()
        
        print("All models loaded successfully!")
    
    def restore(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        skip_denoise: bool = False,
        skip_sr: bool = False,
        skip_colorize: bool = False,
        skip_face: bool = False,
        post_process: bool = True,
        save_intermediate: bool = False
    ) -> Dict:
        """
        Complete restoration pipeline
        
        Args:
            image_path: Path to input image
            output_path: Path to save result (optional)
            skip_*: Skip specific modules
            post_process: Apply post-processing
            save_intermediate: Save intermediate results
            
        Returns:
            Dict containing restored image and metadata
        """
        
        results = {}
        
        # Step 0: Preprocess
        print("Step 0: Preprocessing...")
        img, metadata = self.preprocessor.process(image_path)
        results['original'] = img.copy()
        results['metadata'] = metadata
        
        if save_intermediate:
            cv2.imwrite('step0_preprocessed.png', img)
        
        # Step 1: Denoise
        if not skip_denoise:
            print("Step 1: Denoising...")
            img = self.denoiser.denoise(img)
            results['denoised'] = img.copy()
            if save_intermediate:
                cv2.imwrite('step1_denoised.png', img)
        
        # Step 2: Super-Resolution
        if not skip_sr:
            print("Step 2: Super-Resolution...")
            img = self.super_resolver.upscale(img)
            results['super_resolved'] = img.copy()
            if save_intermediate:
                cv2.imwrite('step2_super_resolved.png', img)
        
        # Step 3: Colorization (if grayscale)
        if not skip_colorize and metadata['is_grayscale']:
            print("Step 3: Colorization...")
            img = self.colorizer.colorize(img)
            results['colorized'] = img.copy()
            if save_intermediate:
                cv2.imwrite('step3_colorized.png', img)
        
        # Step 4-5: Face Detection & Enhancement
        if not skip_face:
            print("Step 4: Detecting faces...")
            faces = self.face_detector.detect(img)
            
            if len(faces) > 0:
                print(f"Found {len(faces)} face(s). Enhancing...")
                img = self.face_enhancer.enhance_faces(
                    img, 
                    faces,
                    fidelity_weight=0.7
                )
                results['face_enhanced'] = img.copy()
                if save_intermediate:
                    cv2.imwrite('step5_face_enhanced.png', img)
            else:
                print("No faces detected, skipping face enhancement")
        
        # Step 6: Post-processing
        if post_process:
            print("Step 6: Post-processing...")
            img = self.post_processor.process(img, params={
                'white_balance': True,
                'auto_levels': True,
                'sharpen': 0.8,
                'enhance_contrast': True
            })
            if save_intermediate:
                cv2.imwrite('step6_post_processed.png', img)
        
        results['final'] = img
        
        # Save final result
        if output_path:
            cv2.imwrite(output_path, img)
            print(f"Saved result to {output_path}")
        
        return results
    
    def batch_restore(
        self,
        input_dir: str,
        output_dir: str,
        **kwargs
    ):
        """
        Restore all images in a directory
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_files = list(input_path.glob('*.jpg')) + \
                     list(input_path.glob('*.png')) + \
                     list(input_path.glob('*.jpeg'))
        
        print(f"Found {len(image_files)} images")
        
        for i, img_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing {img_file.name}...")
            
            try:
                output_file = output_path / f"restored_{img_file.name}"
                self.restore(
                    str(img_file),
                    str(output_file),
                    **kwargs
                )
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")
                continue
        
        print("\nBatch processing complete!")


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    restorer = OldPhotoRestoration(
        device='cuda',
        sr_scale=4
    )
    
    # Single image
    results = restorer.restore(
        image_path='old_photo.jpg',
        output_path='restored.jpg',
        save_intermediate=True
    )
    
    # Batch processing
    # restorer.batch_restore(
    #     input_dir='./data/old_photos',
    #     output_dir='./data/restored'
    # )
```

---

## 🔬 EVALUATION METRICS

### **1. Reference-based Metrics** (khi có ground truth)

**PSNR (Peak Signal-to-Noise Ratio)**:
```python
def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
```
- Higher is better
- Typical range: 20-40 dB
- **Target**: > 28 dB for good restoration

**SSIM (Structural Similarity Index)**:
```python
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(img1, img2):
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray, img2_gray = img1, img2
    
    score, diff = ssim(img1_gray, img2_gray, full=True)
    return score
```
- Range: [0, 1], higher is better
- **Target**: > 0.85

**LPIPS (Learned Perceptual Image Patch Similarity)**:
```python
import lpips

lpips_model = lpips.LPIPS(net='alex').cuda()

def calculate_lpips(img1, img2):
    # Convert to tensor [-1, 1]
    img1_tensor = torch.from_numpy(img1).permute(2,0,1).unsqueeze(0).float() / 127.5 - 1
    img2_tensor = torch.from_numpy(img2).permute(2,0,1).unsqueeze(0).float() / 127.5 - 1
    
    distance = lpips_model(img1_tensor.cuda(), img2_tensor.cuda())
    return distance.item()
```
- Range: [0, 1], lower is better
- More aligned with human perception
- **Target**: < 0.2

### **2. No-reference Metrics** (không cần ground truth)

**NIQE (Natural Image Quality Evaluator)**:
```python
import cv2

def calculate_niqe(image):
    niqe = cv2.quality.QualityNIQE_create()
    score = niqe.compute(image)[0]
    return score
```
- Lower is better
- Range: ~[0, 10]
- **Target**: < 4.5

**BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)**:
```python
import cv2

def calculate_brisque(image):
    brisque = cv2.quality.QualityBRISQUE_create()
    score = brisque.compute(image)[0]
    return score
```
- Lower is better
- Range: ~[0, 100]
- **Target**: < 30

**FID (Fréchet Inception Distance)** - for colorization:
```python
from pytorch_fid import fid_score

# Compare distribution of restored images vs real images
fid_value = fid_score.calculate_fid_given_paths(
    [path_to_restored_images, path_to_real_images],
    batch_size=50,
    device='cuda',
    dims=2048
)
```
- Lower is better
- Measures distribution similarity
- **Target**: < 30 for good colorization

### **3. User Study** (quan trọng nhất!)

```python
# Tạo comparison interface
import gradio as gr

def compare_images(original, restored):
    return gr.HTML(f"""
        <div style="display: flex;">
            <div style="flex: 1;">
                <h3>Original</h3>
                <img src="{original}" />
            </div>
            <div style="flex: 1;">
                <h3>Restored</h3>
                <img src="{restored}" />
            </div>
        </div>
        <div>
            <p>Rate the restoration quality (1-5):</p>
            <input type="range" min="1" max="5" />
        </div>
    """)

# Collect ratings from 20-30 users
# Calculate MOS (Mean Opinion Score)
```

**MOS (Mean Opinion Score)**:
- Scale: 1-5
- 1: Bad, 2: Poor, 3: Fair, 4: Good, 5: Excellent
- **Target**: MOS > 4.0

---

## ⚡ OPTIMIZATION FOR COLAB (ENHANCED)

### **🎯 CRITICAL OPTIMIZATIONS**

#### **1. Lazy Model Loading (Load on Demand)**
```python
class LazyModelLoader:
    """Load models only when needed, unload after use"""
    
    def __init__(self):
        self._models = {}
        self._loaded = {}
    
    def get_model(self, model_name):
        if model_name not in self._loaded:
            print(f"Loading {model_name}...")
            self._loaded[model_name] = self._load_model(model_name)
        return self._loaded[model_name]
    
    def unload_model(self, model_name):
        if model_name in self._loaded:
            del self._loaded[model_name]
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Unloaded {model_name}")

# Usage in pipeline
class OptimizedPipeline:
    def __init__(self):
        self.loader = LazyModelLoader()
    
    def restore(self, image):
        # Load → Use → Unload pattern
        denoiser = self.loader.get_model('denoiser')
        image = denoiser(image)
        self.loader.unload_model('denoiser')
        
        sr_model = self.loader.get_model('super_resolution')
        image = sr_model(image)
        self.loader.unload_model('super_resolution')
        
        # Continue...
        return image
```

**Benefits**:
- ✅ Memory usage: 4GB peak instead of 12GB
- ✅ Fits in Colab free tier
- ✅ Can process larger images

---

#### **2. Smart Image Resizing Strategy**
```python
def smart_resize(image, max_pixels=1024*1024, max_dimension=2048):
    """
    Resize image intelligently based on memory constraints
    
    Rules:
    - If total pixels > max_pixels: resize down
    - If any dimension > max_dimension: resize down
    - Keep aspect ratio
    - Return resize info for upscaling back
    """
    h, w = image.shape[:2]
    total_pixels = h * w
    
    # Calculate resize factor
    if total_pixels > max_pixels:
        scale = np.sqrt(max_pixels / total_pixels)
    elif max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
    else:
        scale = 1.0
    
    if scale < 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Resized from {w}x{h} to {new_w}x{new_h} (scale={scale:.2f})")
        return resized, (h, w, scale)
    
    return image, (h, w, 1.0)

def restore_original_size(image, resize_info):
    """Upscale back to original size"""
    orig_h, orig_w, scale = resize_info
    if scale < 1.0:
        return cv2.resize(image, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
    return image
```

**Benefits**:
- ✅ Prevents OOM errors
- ✅ Faster processing
- ✅ Can restore to original size after

---

#### **3. Efficient Tiling with Overlap Blending**
```python
def process_with_tiles(image, process_fn, tile_size=512, overlap=64):
    """
    Process large image in tiles with smooth blending
    
    Args:
        image: Input image
        process_fn: Function to apply (e.g., super_resolution)
        tile_size: Size of each tile
        overlap: Overlap between tiles for blending
    """
    h, w = image.shape[:2]
    
    # If image small enough, process directly
    if h <= tile_size and w <= tile_size:
        return process_fn(image)
    
    # Calculate tile positions
    stride = tile_size - overlap
    tiles = []
    positions = []
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1, x1 = y, x
            y2, x2 = min(y + tile_size, h), min(x + tile_size, w)
            
            # Extract tile
            tile = image[y1:y2, x1:x2]
            
            # Process tile
            processed_tile = process_fn(tile)
            
            tiles.append(processed_tile)
            positions.append((y1, x1, y2, x2))
            
            # Clear memory after each tile
            torch.cuda.empty_cache()
    
    # Merge tiles with feathering
    result = merge_tiles_with_feathering(tiles, positions, (h, w), overlap)
    return result

def merge_tiles_with_feathering(tiles, positions, output_shape, overlap):
    """Merge tiles with smooth blending in overlap regions"""
    h, w = output_shape
    result = np.zeros((h, w, 3), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)
    
    for tile, (y1, x1, y2, x2) in zip(tiles, positions):
        tile_h, tile_w = tile.shape[:2]
        
        # Create feathering mask
        mask = np.ones((tile_h, tile_w), dtype=np.float32)
        
        # Feather edges
        feather_size = overlap // 2
        for i in range(feather_size):
            alpha = i / feather_size
            mask[i, :] *= alpha  # Top
            mask[-i-1, :] *= alpha  # Bottom
            mask[:, i] *= alpha  # Left
            mask[:, -i-1] *= alpha  # Right
        
        # Add to result with weights
        result[y1:y2, x1:x2] += tile * mask[:, :, np.newaxis]
        weight_map[y1:y2, x1:x2] += mask
    
    # Normalize by weights
    result /= np.maximum(weight_map[:, :, np.newaxis], 1e-6)
    return result.astype(np.uint8)
```

**Benefits**:
- ✅ No visible seams between tiles
- ✅ Can process arbitrarily large images
- ✅ Memory efficient

---

#### **4. Checkpoint System for Long Processing**
```python
import pickle
from pathlib import Path

class CheckpointManager:
    """Save/load intermediate results to survive Colab disconnections"""
    
    def __init__(self, checkpoint_dir='./checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(self, image, step_name, metadata=None):
        """Save intermediate result"""
        checkpoint_path = self.checkpoint_dir / f"{step_name}.pkl"
        data = {
            'image': image,
            'metadata': metadata,
            'timestamp': time.time()
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Saved checkpoint: {step_name}")
    
    def load_checkpoint(self, step_name):
        """Load intermediate result"""
        checkpoint_path = self.checkpoint_dir / f"{step_name}.pkl"
        if checkpoint_path.exists():
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            print(f"✓ Loaded checkpoint: {step_name}")
            return data['image'], data['metadata']
        return None, None
    
    def has_checkpoint(self, step_name):
        """Check if checkpoint exists"""
        return (self.checkpoint_dir / f"{step_name}.pkl").exists()

# Usage in pipeline
class RobustPipeline:
    def __init__(self):
        self.checkpoint_mgr = CheckpointManager()
    
    def restore(self, image_path, resume=True):
        image_id = Path(image_path).stem
        
        # Step 1: Preprocessing
        if resume and self.checkpoint_mgr.has_checkpoint(f"{image_id}_preprocessed"):
            image, meta = self.checkpoint_mgr.load_checkpoint(f"{image_id}_preprocessed")
        else:
            image = self.preprocess(image_path)
            self.checkpoint_mgr.save_checkpoint(image, f"{image_id}_preprocessed")
        
        # Step 2: Denoising
        if resume and self.checkpoint_mgr.has_checkpoint(f"{image_id}_denoised"):
            image, _ = self.checkpoint_mgr.load_checkpoint(f"{image_id}_denoised")
        else:
            image = self.denoise(image)
            self.checkpoint_mgr.save_checkpoint(image, f"{image_id}_denoised")
        
        # Continue for other steps...
        return image
```

**Benefits**:
- ✅ Survive Colab disconnections
- ✅ Resume from last step
- ✅ Save time on re-runs

---

#### **5. Batch Processing with Progress Tracking**
```python
from tqdm.auto import tqdm
import time

class BatchProcessor:
    """Process multiple images with progress tracking and error handling"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.results = []
        self.errors = []
    
    def process_batch(self, image_paths, output_dir, max_retries=2):
        """Process batch of images with retry logic"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        pbar = tqdm(image_paths, desc="Processing images")
        
        for img_path in pbar:
            img_name = Path(img_path).name
            output_path = output_dir / f"restored_{img_name}"
            
            # Skip if already processed
            if output_path.exists():
                pbar.set_postfix({"status": "skipped (exists)"})
                continue
            
            # Try processing with retries
            for attempt in range(max_retries):
                try:
                    pbar.set_postfix({"status": f"processing (attempt {attempt+1})"})
                    
                    result = self.pipeline.restore(img_path)
                    cv2.imwrite(str(output_path), result)
                    
                    self.results.append({
                        'input': img_path,
                        'output': str(output_path),
                        'status': 'success'
                    })
                    
                    pbar.set_postfix({"status": "✓ success"})
                    break
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        # Final attempt failed
                        self.errors.append({
                            'input': img_path,
                            'error': str(e)
                        })
                        pbar.set_postfix({"status": f"✗ failed: {str(e)[:30]}"})
                    else:
                        # Retry after clearing memory
                        torch.cuda.empty_cache()
                        gc.collect()
                        time.sleep(2)
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"Batch processing complete!")
        print(f"✓ Success: {len(self.results)}")
        print(f"✗ Failed: {len(self.errors)}")
        
        if self.errors:
            print(f"\nFailed images:")
            for err in self.errors:
                print(f"  - {err['input']}: {err['error']}")
        
        return self.results, self.errors
```

**Benefits**:
- ✅ Visual progress tracking
- ✅ Automatic retry on failures
- ✅ Skip already processed images
- ✅ Error logging

---

## ⚡ OPTIMIZATION FOR COLAB (ORIGINAL)

### **Memory Management**

```python
import gc
import torch

def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

# Use throughout pipeline
class OptimizedPipeline(OldPhotoRestoration):
    
    def restore(self, image_path, **kwargs):
        # Process in tiles for large images
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        # If image too large, process in tiles
        if h * w > 4096 * 4096:  # > 16MP
            img = self.process_in_tiles(img)
        else:
            img = self.process_whole(img)
        
        clear_memory()
        return img
    
    def process_in_tiles(self, img, tile_size=512, overlap=64):
        """Process large image in tiles with overlap"""
        h, w = img.shape[:2]
        tiles = []
        
        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                tile = img[y:y_end, x:x_end]
                restored_tile = self.process_whole(tile)
                tiles.append((restored_tile, (y, x, y_end, x_end)))
                
                clear_memory()
        
        # Merge tiles with blending
        result = self.merge_tiles(tiles, (h, w))
        return result
```

### **Mixed Precision Training**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():  # FP16
        output = model(batch)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### **Model Quantization** (inference)

```python
# Convert model to INT8
import torch.quantization

model_fp32 = YourModel()
model_fp32.eval()

# Quantization
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

# 4x faster, 4x less memory!
```

### **Gradient Checkpointing**

```python
from torch.utils.checkpoint import checkpoint

class MemoryEfficientModel(nn.Module):
    def forward(self, x):
        # Use checkpointing for memory-heavy layers
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return x
```

---

## 📱 DEMO INTERFACE (Gradio)

```python
import gradio as gr
import cv2
import numpy as np

def restore_image(
    image,
    denoise: bool,
    sr_scale: int,
    colorize: bool,
    enhance_face: bool,
    fidelity_weight: float
):
    """Gradio inference function"""
    
    # Save uploaded image
    temp_path = "temp_input.jpg"
    cv2.imwrite(temp_path, image)
    
    # Restore
    results = restorer.restore(
        temp_path,
        skip_denoise=not denoise,
        skip_sr=(sr_scale == 1),
        skip_colorize=not colorize,
        skip_face=not enhance_face,
        save_intermediate=True
    )
    
    return results['final']

# Create interface
demo = gr.Interface(
    fn=restore_image,
    inputs=[
        gr.Image(label="Upload Old Photo"),
        gr.Checkbox(label="Denoise", value=True),
        gr.Slider(1, 4, step=1, label="Upscale Factor", value=4),
        gr.Checkbox(label="Colorize (if B&W)", value=True),
        gr.Checkbox(label="Enhance Faces", value=True),
        gr.Slider(0, 1, step=0.1, label="Face Fidelity", value=0.7)
    ],
    outputs=gr.Image(label="Restored Photo"),
    title="🎨 AI Old Photo Restoration",
    description="Upload an old/damaged photo and restore it with AI!",
    examples=[
        ["examples/old1.jpg", True, 4, True, True, 0.7],
        ["examples/old2.jpg", True, 2, False, True, 0.5],
    ],
    allow_flagging="never"
)

# Launch
demo.launch(share=True)  # Creates public link
```

### **Advanced Demo với Before/After Slider**:

```python
import gradio as gr

def create_comparison(original, restored):
    """Create side-by-side comparison with slider"""
    return gr.HTML(f"""
        <style>
            .comparison-container {{
                position: relative;
                width: 100%;
                max-width: 800px;
                margin: 0 auto;
            }}
            .comparison-image {{
                width: 100%;
                display: block;
            }}
            .comparison-slider {{
                position: absolute;
                top: 0;
                width: 50%;
                height: 100%;
                overflow: hidden;
            }}
            input[type="range"] {{
                width: 100%;
                margin-top: 20px;
            }}
        </style>
        <div class="comparison-container">
            <img class="comparison-image" src="data:image/png;base64,{to_base64(restored)}" />
            <div class="comparison-slider" id="slider">
                <img class="comparison-image" src="data:image/png;base64,{to_base64(original)}" />
            </div>
        </div>
        <input type="range" min="0" max="100" value="50" 
               oninput="document.getElementById('slider').style.width=this.value+'%'" />
        <p style="text-align: center;">Slide to compare</p>
    """)

with gr.Blocks() as demo:
    gr.Markdown("# 🎨 AI Old Photo Restoration")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Photo")
            denoise = gr.Checkbox(label="Denoise", value=True)
            sr_scale = gr.Slider(1, 4, step=1, value=4, label="Upscale")
            colorize = gr.Checkbox(label="Colorize", value=True)
            enhance_face = gr.Checkbox(label="Enhance Faces", value=True)
            submit_btn = gr.Button("Restore", variant="primary")
        
        with gr.Column():
            output = gr.Image(label="Result")
            comparison = gr.HTML()
    
    submit_btn.click(
        restore_image,
        inputs=[input_image, denoise, sr_scale, colorize, enhance_face],
        outputs=[output, comparison]
    )

demo.launch()
```

---

## �  COMMON PITFALLS & SOLUTIONS

### **Pitfall 1: Model weights không download được**
**Problem**: GitHub releases bị chặn, Google Drive quota exceeded
**Solution**:
```python
# Sử dụng multiple mirrors
WEIGHT_SOURCES = {
    'realesrgan': [
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-x4plus.pth',
        'https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth',
        'https://drive.google.com/uc?id=YOUR_BACKUP_ID'  # Your backup
    ]
}

def download_with_fallback(model_name, save_path):
    """Try multiple sources"""
    for url in WEIGHT_SOURCES[model_name]:
        try:
            print(f"Trying {url}...")
            urllib.request.urlretrieve(url, save_path)
            print("✓ Downloaded successfully")
            return True
        except Exception as e:
            print(f"✗ Failed: {e}")
            continue
    raise Exception("All download sources failed!")
```

### **Pitfall 2: Colab disconnects giữa chừng**
**Problem**: Session timeout sau 12 hours (free tier)
**Solution**:
```python
# Auto-reconnect script (chạy trong browser console)
function KeepAlive() {
    console.log("Keeping session alive...");
    document.querySelector("colab-connect-button").click();
}
setInterval(KeepAlive, 60000); // Every 1 minute

# Hoặc dùng checkpoint system (đã implement ở trên)
```

### **Pitfall 3: Out of Memory khi process large images**
**Problem**: CUDA OOM error với images > 2048x2048
**Solution**:
```python
# Implement automatic downscaling
def safe_process(image, process_fn, max_size=2048):
    h, w = image.shape[:2]
    
    if max(h, w) > max_size:
        # Downscale
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image_small = cv2.resize(image, (new_w, new_h))
        
        # Process
        result_small = process_fn(image_small)
        
        # Upscale back
        result = cv2.resize(result_small, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return result
    else:
        return process_fn(image)
```

### **Pitfall 4: Face enhancement làm mất identity**
**Problem**: CodeFormer với fidelity=0 tạo face quá khác người gốc
**Solution**:
```python
# Use higher fidelity weight (0.7-0.9)
# Add identity loss nếu fine-tune
# Hoặc blend với original face

def blend_faces(original_face, enhanced_face, blend_ratio=0.7):
    """Blend enhanced face with original to preserve identity"""
    return cv2.addWeighted(
        enhanced_face, blend_ratio,
        original_face, 1 - blend_ratio,
        0
    )
```

### **Pitfall 5: Colorization tạo màu không realistic**
**Problem**: DDColor tô màu sai (da xanh, trời vàng, etc.)
**Solution**:
```python
# Post-process colors
def correct_colors(colorized_image):
    """Apply color correction"""
    # Convert to LAB
    lab = cv2.cvtColor(colorized_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Clip extreme values
    a = np.clip(a, 100, 155)  # Reduce extreme red/green
    b = np.clip(b, 100, 155)  # Reduce extreme blue/yellow
    
    # Merge back
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Hoặc cho user chọn color palette
# Hoặc dùng reference image cho color transfer
```

### **Pitfall 6: Results không consistent giữa các runs**
**Problem**: Mỗi lần chạy cho kết quả khác nhau
**Solution**:
```python
# Set random seeds
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call at start of pipeline
set_seed(42)
```

### **Pitfall 7: Evaluation metrics không match visual quality**
**Problem**: PSNR cao nhưng ảnh trông không tốt
**Solution**:
```python
# Use multiple metrics
metrics = {
    'psnr': calculate_psnr(result, reference),  # Pixel-level
    'ssim': calculate_ssim(result, reference),  # Structure
    'lpips': calculate_lpips(result, reference),  # Perceptual
    'niqe': calculate_niqe(result),  # No-reference
}

# Weight them appropriately
# LPIPS và NIQE thường align với human perception hơn PSNR
```

---

## 💡 BEST PRACTICES

### **1. Code Organization**
```
src/
├── models/          # Model wrappers
│   ├── base.py     # Base model class
│   ├── denoiser.py
│   ├── super_resolution.py
│   └── ...
├── utils/          # Utilities
│   ├── image_io.py
│   ├── metrics.py
│   └── visualization.py
├── pipeline.py     # Main pipeline
└── config.py       # Configuration
```

### **2. Configuration Management**
```python
# config.yaml
models:
  denoising:
    type: "opencv"  # or "nafnet"
    strength: 10
  
  super_resolution:
    type: "realesrgan"
    scale: 4
    tile_size: 512
  
  colorization:
    type: "ddcolor"
    enable: true
  
  face_enhancement:
    type: "codeformer"
    fidelity: 0.7

processing:
  max_image_size: 2048
  use_tiling: true
  save_intermediate: false
  checkpoint_enabled: true
```

### **3. Logging**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('restoration.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Usage
logger.info("Starting restoration pipeline")
logger.debug(f"Image size: {h}x{w}")
logger.warning("Image too large, applying tiling")
logger.error(f"Failed to process: {e}")
```

### **4. Testing Strategy**
```python
# tests/test_pipeline.py
import pytest

def test_preprocessing():
    """Test preprocessing module"""
    preprocessor = Preprocessor()
    image = cv2.imread('test_images/sample.jpg')
    result, meta = preprocessor.process(image)
    
    assert result is not None
    assert meta['is_grayscale'] in [True, False]
    assert result.shape[2] == 3  # RGB

def test_pipeline_end_to_end():
    """Test full pipeline"""
    pipeline = OldPhotoRestoration()
    result = pipeline.restore('test_images/old_photo.jpg')
    
    assert result is not None
    assert result.shape[0] > 0 and result.shape[1] > 0

# Run: pytest tests/
```

### **5. Documentation**
```python
def restore_image(
    image_path: str,
    output_path: Optional[str] = None,
    config: Optional[Dict] = None
) -> np.ndarray:
    """
    Restore an old/degraded photo using AI models.
    
    Args:
        image_path: Path to input image (JPG/PNG)
        output_path: Path to save result (optional)
        config: Configuration dict (optional)
            - denoise_strength: int (0-30), default=10
            - sr_scale: int (2 or 4), default=4
            - colorize: bool, default=True
            - enhance_faces: bool, default=True
    
    Returns:
        Restored image as numpy array (H, W, 3)
    
    Raises:
        FileNotFoundError: If image_path doesn't exist
        ValueError: If image format not supported
        RuntimeError: If GPU out of memory
    
    Example:
        >>> result = restore_image('old_photo.jpg', 'restored.jpg')
        >>> print(f"Restored image shape: {result.shape}")
    """
    pass
```

---

## 📝 BÁO CÁO ĐỒ ÁN

### **Cấu trúc báo cáo**

```markdown
# BÁO CÁO ĐỒ ÁN: HỆ THỐNG PHỤC CHẾ ẢNH CŨ TỰ ĐỘNG

## CHƯƠNG 1: GIỚI THIỆU
1.1. Đặt vấn đề
1.2. Mục tiêu đồ án
1.3. Phạm vi nghiên cứu
1.4. Cấu trúc báo cáo

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT
2.1. Các vấn đề trong ảnh cũ
   - Nhiễu (Gaussian, Poisson, Salt & Pepper)
   - Degradation (blur, compression)
   - Scratches, stains, fading
   
2.2. Deep Learning cho Image Restoration
   - CNN architectures
   - Transformer-based methods
   - GAN-based approaches
   - Diffusion models
   
2.3. Các mô hình liên quan
   - Denoising: NAFNet, Restormer
   - Super-Resolution: ESRGAN, SwinIR
   - Colorization: DDColor, DeOldify
   - Face Enhancement: CodeFormer, GFPGAN

## CHƯƠNG 3: PHƯƠNG PHÁP ĐỀ XUẤT
3.1. Kiến trúc tổng thể (pipeline diagram)
3.2. Chi tiết từng module
3.3. Integration strategy
3.4. Optimization techniques

## CHƯƠNG 4: THÍ NGHIỆM VÀ KẾT QUẢ
4.1. Datasets
   - Training data
   - Test data
   - Synthetic degradation
   
4.2. Implementation details
   - Hardware/Software
   - Training parameters
   - Inference time
   
4.3. Evaluation metrics
   - PSNR, SSIM, LPIPS
   - NIQE, BRISQUE
   - User study (MOS)
   
4.4. Kết quả định lượng (tables)
4.5. Kết quả định tính (visual comparisons)
4.6. Ablation study (contribution của từng module)

## CHƯƠNG 5: DEMO ỨNG DỤNG
5.1. Web interface
5.2. User guide
5.3. Example results

## CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
6.1. Tổng kết
6.2. Hạn chế
6.3. Hướng phát triển

TÀI LIỆU THAM KHẢO
PHỤ LỤC
- Source code
- Pre-trained weights links
- Additional results
```

### **Evaluation Tables Template**

```markdown
### Bảng 4.1: So sánh PSNR/SSIM trên test set

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|--------|-------------|---------|---------|
| Input (degraded) | 22.45 | 0.6234 | 0.4521 |
| NAFNet only | 27.83 | 0.8123 | 0.2341 |
| + Real-ESRGAN | 29.12 | 0.8567 | 0.1987 |
| + DDColor | 28.98 | 0.8501 | 0.2034 |
| **Ours (Full)** | **30.45** | **0.8923** | **0.1654** |
| + CodeFormer | **31.23** | **0.9145** | **0.1432** |

### Bảng 4.2: Inference time (512x512 → 2048x2048)

| Module | Time (s) | GPU Memory (GB) |
|--------|----------|-----------------|
| Preprocessing | 0.05 | 0.2 |
| Denoising | 0.82 | 2.1 |
| Super-Resolution | 1.34 | 3.5 |
| Colorization | 0.67 | 1.8 |
| Face Detection | 0.12 | 0.5 |
| Face Enhancement | 0.95 | 2.3 |
| Post-processing | 0.08 | 0.1 |
| **Total** | **4.03** | **10.5** |

*Hardware: Tesla T4 GPU (Colab)*

### Bảng 4.3: User study results (MOS, n=25 users)

| Method | Naturalness ↑ | Detail Quality ↑ | Color Accuracy ↑ | Overall ↑ |
|--------|---------------|------------------|------------------|-----------|
| Input | 2.1 | 1.8 | 2.3 | 2.0 |
| Remini App | 3.8 | 3.9 | 3.7 | 3.8 |
| VanceAI | 4.0 | 4.1 | 3.8 | 4.0 |
| **Ours** | **4.3** | **4.4** | **4.2** | **4.3** |

*Scale: 1 (worst) - 5 (best)*
```

---

## 🚀 TIMELINE & MILESTONES (OPTIMIZED)

### **🎯 CHIẾN LƯỢC: Use Pre-trained Models + Focus on Integration**

**Lý do**: Training from scratch mất quá nhiều thời gian (2-4 tuần/model) và resources. Thay vào đó, sử dụng pre-trained models và tập trung vào:
1. Integration pipeline hiệu quả
2. Optimization cho Colab
3. Demo & evaluation chất lượng cao

---

### **Phase 1: Foundation (Week 1-3)**

**Week 1: Setup & Quick Prototyping**
- [ ] Setup Colab Pro (nếu có budget) hoặc optimize free tier
- [ ] Clone pre-trained model repos:
  - Real-ESRGAN: `git clone https://github.com/xinntao/Real-ESRGAN`
  - CodeFormer: `git clone https://github.com/sczhou/CodeFormer`
  - DDColor: `git clone https://github.com/piddnad/DDColor`
- [ ] Test từng model riêng lẻ với sample images
- [ ] Đo memory usage và inference time
- [ ] **Deliverable**: Working notebook cho mỗi model

**Week 2: Preprocessing & Denoising**
- [ ] Implement Preprocessor class với:
  - Auto-resize cho Colab memory limits
  - Grayscale detection
  - Format conversion utilities
- [ ] Integrate denoising:
  - **Simplified**: Dùng cv2.fastNlMeansDenoisingColored thay vì NAFNet
  - **Reason**: NAFNet quá nặng (68M params), cv2 đủ tốt cho most cases
  - **Alternative**: Nếu cần AI-based, dùng pre-trained NAFNet inference only
- [ ] **Deliverable**: `preprocessing.py` + `denoising.py`

**Week 3: Super-Resolution Integration**
- [ ] Wrap Real-ESRGAN với error handling
- [ ] Implement tiling strategy cho large images
- [ ] Add progress bars với tqdm
- [ ] Memory optimization: clear cache after each step
- [ ] **Deliverable**: `super_resolution.py` working on Colab

---

### **Phase 2: Core Features (Week 4-6)**

**Week 4: Colorization**
- [ ] Integrate DDColor pre-trained model
- [ ] Add fallback: DeOldify (simpler, faster) nếu DDColor OOM
- [ ] Implement color transfer từ reference image (optional feature)
- [ ] Test trên diverse B&W photos
- [ ] **Deliverable**: `colorization.py` với 2 options

**Week 5: Face Enhancement**
- [ ] Integrate RetinaFace cho detection (lightweight)
- [ ] Integrate CodeFormer pre-trained
- [ ] Implement face paste-back với Poisson blending
- [ ] Handle edge cases: no faces, multiple faces, partial faces
- [ ] **Deliverable**: `face_detection.py` + `face_enhancement.py`

**Week 6: Pipeline Integration**
- [ ] Create `OldPhotoRestoration` class
- [ ] Implement sequential pipeline với skip options
- [ ] Add intermediate result saving
- [ ] Implement batch processing
- [ ] **Deliverable**: `pipeline.py` working end-to-end

---

### **Phase 3: Optimization & Polish (Week 7-9)**

**Week 7: Performance Optimization**
- [ ] Profile memory usage từng module
- [ ] Implement smart tiling cho images > 2048px
- [ ] Add FP16 inference cho speed
- [ ] Optimize post-processing (vectorize operations)
- [ ] **Target**: < 30s cho 1024x1024 image trên T4 GPU
- [ ] **Deliverable**: Optimized pipeline

**Week 8: Demo Interface**
- [ ] Build Gradio interface với:
  - Upload/download
  - Module toggles (skip denoise, skip colorize, etc.)
  - Fidelity slider cho face enhancement
  - Before/After comparison slider
- [ ] Add example images
- [ ] Deploy lên Hugging Face Spaces (free hosting!)
- [ ] **Deliverable**: Public demo link

**Week 9: Dataset & Evaluation**
- [ ] Collect 50-100 real old photos (Archive.org, r/estoration)
- [ ] Run pipeline trên test set
- [ ] Calculate metrics: NIQE, BRISQUE (no-reference)
- [ ] Create visual comparison grid
- [ ] **Deliverable**: Results folder với metrics

---

### **Phase 4: Documentation & Finalization (Week 10-12)**

**Week 10: User Study & Refinement**
- [ ] Conduct user study (20-30 people):
  - Google Form với image pairs
  - Rate 1-5 cho quality, naturalness, color
- [ ] Analyze feedback
- [ ] Fine-tune parameters based on feedback
- [ ] **Deliverable**: User study results (MOS scores)

**Week 11: Report Writing**
- [ ] Write Chapters 1-3 (intro, theory, method)
- [ ] Create architecture diagrams (draw.io hoặc Mermaid)
- [ ] Write Chapter 4 (experiments) với tables & figures
- [ ] **Deliverable**: Draft report (80% complete)

**Week 12: Final Polish**
- [ ] Complete Chapter 5-6 (demo, conclusion)
- [ ] Code cleanup & documentation
- [ ] Create README với:
  - Installation guide
  - Usage examples
  - Model weights links
  - Demo screenshots
- [ ] Record demo video (3-5 minutes)
- [ ] **Deliverable**: Final submission package

---

### **⚠️ RISK MITIGATION**

**Risk 1: Colab timeout/disconnection**
- **Solution**: 
  - Save checkpoints after each module
  - Use Colab Pro ($10/month) nếu có budget
  - Implement auto-resume từ last checkpoint

**Risk 2: Out of Memory**
- **Solution**:
  - Implement aggressive tiling
  - Use FP16 inference
  - Process modules sequentially, clear cache between
  - Fallback to smaller models (MobileNet backbone)

**Risk 3: Model weights download fails**
- **Solution**:
  - Mirror weights lên Google Drive
  - Provide multiple download sources
  - Include weights in submission (nếu cho phép)

**Risk 4: Poor results on certain image types**
- **Solution**:
  - Add preprocessing filters (detect image type)
  - Provide manual parameter adjustment in demo
  - Document limitations clearly

---

### **📊 SUCCESS METRICS**

**Minimum Viable Product (MVP)**:
- ✅ Pipeline processes 512x512 image in < 60s
- ✅ Works on Colab free tier (12GB RAM)
- ✅ Handles grayscale + color images
- ✅ Face enhancement works on frontal faces
- ✅ Demo deployed và accessible

**Target Goals**:
- 🎯 NIQE score < 5.0 on test set
- 🎯 User study MOS > 4.0
- 🎯 Process 1024x1024 in < 30s on T4
- 🎯 Batch processing 100 images without crash
- 🎯 Demo has > 100 users (track via HF Spaces)

**Stretch Goals** (nếu còn thời gian):
- 🚀 Fine-tune colorization trên old photo dataset
- 🚀 Add video restoration (frame-by-frame)
- 🚀 Mobile app (TensorFlow Lite conversion)
- 🚀 API endpoint (FastAPI + Docker)

---

## 💰 COST OPTIMIZATION

### **Free Tier Strategy**
```python
# Colab free tier limits:
# - 12GB RAM
# - ~15GB disk
# - 12 hours session
# - T4 GPU (when available)

# Optimization tactics:
1. Load models on-demand, unload after use
2. Process in batches of 5-10 images
3. Use Google Drive for storage
4. Implement checkpointing every 10 images
```

### **Paid Options** (nếu cần)
- **Colab Pro**: $10/month
  - 25GB RAM
  - 24 hour sessions
  - Priority GPU access
  - **Worth it**: Nếu bạn chạy nhiều experiments

- **Hugging Face Spaces**: FREE
  - Host demo permanently
  - 16GB RAM, 2 CPU cores
  - Persistent storage
  - **Highly recommended**

---

## 🔧 SIMPLIFIED ARCHITECTURE (OPTIMIZED)

```
INPUT IMAGE
    ↓
[Preprocessing] ← 0.05s, 0.2GB
    ↓
[Lightweight Denoising] ← 0.3s, 0.5GB (cv2 instead of NAFNet)
    ↓
[Real-ESRGAN 4x] ← 1.5s, 3.5GB (tiled if needed)
    ↓
[Check Grayscale?]
    ↓ (if yes)
[DDColor/DeOldify] ← 0.8s, 2GB
    ↓
[RetinaFace Detection] ← 0.1s, 0.5GB
    ↓ (if faces found)
[CodeFormer Enhancement] ← 1.0s, 2.5GB
    ↓
[Post-processing] ← 0.1s, 0.1GB
    ↓
OUTPUT IMAGE

Total: ~4s for 512x512 → 2048x2048
Memory peak: ~4GB (sequential processing)
```

**Key Changes**:
1. ❌ Remove NAFNet → ✅ Use cv2.fastNlMeansDenoising (faster, lighter)
2. ❌ Remove Restormer option → ✅ Focus on Real-ESRGAN only
3. ❌ Remove SwinIR option → ✅ One model = less complexity
4. ✅ Add fallback options (DeOldify if DDColor fails)
5. ✅ Sequential processing với memory clearing

---

##