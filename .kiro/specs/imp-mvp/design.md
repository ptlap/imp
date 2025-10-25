# Design Document

## Overview

IMP MVP (Minimum Viable Product) là một hệ thống phục chế ảnh cũ được thiết kế để chạy hiệu quả trên cả môi trường local (WSL) và Google Colab. Hệ thống sử dụng kiến trúc modular với các components độc lập có thể được test và maintain riêng biệt.

Thiết kế tập trung vào:
- **Simplicity**: Sử dụng pre-trained models, không train
- **Efficiency**: Lazy loading, memory management, tiling
- **Flexibility**: Configuration-driven, skip modules
- **Reliability**: Error handling, checkpoints, logging

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│              (Python API / Notebook)                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                 Pipeline Orchestrator                    │
│  - Load configuration                                    │
│  - Manage model lifecycle                                │
│  - Execute processing steps                              │
│  - Handle errors and checkpoints                         │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│Preprocessing│ │  Denoising  │ │Super-Resolu-│
│   Module    │ │   Module    │ │tion Module  │
└─────────────┘ └─────────────┘ └─────────────┘
        │            │            │
        └────────────┼────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Utility Services                        │
│  - Image I/O                                             │
│  - Memory Management                                     │
│  - Logging                                               │
│  - Checkpoint Management                                 │
└─────────────────────────────────────────────────────────┘
```

### Processing Flow

```
Input Image Path
    │
    ▼
┌─────────────────────┐
│  Load Configuration │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Load & Validate   │
│       Image         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Preprocessing     │
│  - Resize if needed │
│  - Detect grayscale │
│  - Normalize        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Checkpoint 1     │ ← Save intermediate
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     Denoising       │
│  - OpenCV/NAFNet    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Checkpoint 2     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Super-Resolution   │
│  - Real-ESRGAN      │
│  - Tiling if large  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Checkpoint 3     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Save Result       │
└──────────┬──────────┘
           │
           ▼
    Restored Image
```

## Components and Interfaces

### 1. Configuration Manager

**Purpose**: Load and validate configuration from YAML files

**Interface**:
```python
class Config:
    """Configuration container"""
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load config from YAML file"""
        
    @classmethod
    def default(cls) -> 'Config':
        """Create default configuration"""
        
    def validate(self) -> bool:
        """Validate configuration values"""
```

**Configuration Schema**:
```yaml
models:
  denoising:
    type: "opencv"  # or "nafnet"
    strength: 10
    
  super_resolution:
    type: "realesrgan"
    scale: 4
    tile_size: 512
    tile_overlap: 64
    use_fp16: true
    weights_url: "https://github.com/.../realesrgan-x4plus.pth"

processing:
  max_image_size: 2048
  save_intermediate: false
  checkpoint_enabled: true
  checkpoint_dir: "./checkpoints"

logging:
  level: "INFO"
  file: "imp.log"
```

### 2. Preprocessing Module

**Purpose**: Prepare images for processing

**Interface**:
```python
class Preprocessor:
    """Image preprocessing"""
    
    def __init__(self, max_size: int = 2048):
        self.max_size = max_size
    
    def process(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Load and preprocess image
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (processed_image, metadata)
            - processed_image: numpy array (H, W, 3)
            - metadata: dict with keys:
                - original_size: (height, width)
                - is_grayscale: bool
                - resize_factor: float
        """
        
    def detect_grayscale(self, image: np.ndarray) -> bool:
        """Check if image is grayscale"""
        
    def smart_resize(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resize image if too large, return (resized, scale_factor)"""
```

**Implementation Details**:
- Use PIL/OpenCV for image loading
- Detect grayscale by checking if R=G=B for all pixels (with tolerance)
- Resize using INTER_AREA for downscaling
- Store original dimensions for potential upscaling back

### 3. Denoising Module

**Purpose**: Remove noise and artifacts

**Interface**:
```python
class DenoisingModule:
    """Base class for denoising"""
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from image"""
        raise NotImplementedError

class OpenCVDenoiser(DenoisingModule):
    """OpenCV-based denoising (fast, CPU)"""
    
    def __init__(self, strength: int = 10):
        self.strength = strength
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply fastNlMeansDenoisingColored"""

class NAFNetDenoiser(DenoisingModule):
    """NAFNet-based denoising (quality, GPU)"""
    
    def __init__(self, weights_path: str, device: str = 'cuda'):
        self.weights_path = weights_path
        self.device = device
        self.model = None
    
    def load_model(self):
        """Lazy load NAFNet model"""
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply NAFNet denoising"""
```

**Design Decisions**:
- Default to OpenCV for speed and simplicity
- NAFNet as optional upgrade for quality
- Lazy loading for NAFNet to save memory
- Factory pattern for easy switching

### 4. Super-Resolution Module

**Purpose**: Upscale images using Real-ESRGAN

**Interface**:
```python
class SuperResolutionModule:
    """Real-ESRGAN super-resolution"""
    
    def __init__(
        self,
        scale: int = 4,
        weights_path: str = None,
        tile_size: int = 512,
        tile_overlap: int = 64,
        device: str = 'cuda',
        use_fp16: bool = True
    ):
        self.scale = scale
        self.weights_path = weights_path
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.device = device
        self.use_fp16 = use_fp16
        self.model = None
    
    def load_model(self):
        """Load Real-ESRGAN model"""
    
    def upscale(self, image: np.ndarray) -> np.ndarray:
        """
        Upscale image
        
        Args:
            image: Input image (H, W, 3) BGR format
            
        Returns:
            Upscaled image (H*scale, W*scale, 3)
        """
    
    def _should_tile(self, image: np.ndarray) -> bool:
        """Check if image needs tiling"""
    
    def _process_with_tiles(self, image: np.ndarray) -> np.ndarray:
        """Process large image with tiling"""
```

**Tiling Strategy**:
```python
def _process_with_tiles(self, image: np.ndarray) -> np.ndarray:
    """
    Tiling algorithm:
    1. Divide image into overlapping tiles
    2. Process each tile independently
    3. Merge with feathering in overlap regions
    """
    h, w = image.shape[:2]
    stride = self.tile_size - self.tile_overlap
    
    tiles = []
    positions = []
    
    # Extract tiles
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            tile = extract_tile(image, x, y, self.tile_size)
            processed = self.model.enhance(tile)
            tiles.append(processed)
            positions.append((x, y))
    
    # Merge with feathering
    result = merge_tiles_with_feathering(tiles, positions, output_size)
    return result
```

### 5. Pipeline Orchestrator

**Purpose**: Coordinate all modules and manage execution flow

**Interface**:
```python
class OldPhotoRestoration:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.default()
        self.preprocessor = Preprocessor(self.config.processing.max_image_size)
        self.denoiser = None  # Lazy load
        self.super_resolver = None  # Lazy load
        self.checkpoint_mgr = CheckpointManager(self.config.processing.checkpoint_dir)
        self.logger = setup_logger(self.config.logging)
    
    def restore(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        resume: bool = True
    ) -> np.ndarray:
        """
        Restore old photo
        
        Args:
            image_path: Path to input image
            output_path: Path to save result (optional)
            resume: Resume from checkpoint if available
            
        Returns:
            Restored image as numpy array
        """
    
    def batch_restore(
        self,
        image_paths: List[str],
        output_dir: str,
        max_retries: int = 2
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Restore multiple images
        
        Returns:
            Tuple of (successes, failures)
        """
    
    def _load_denoiser(self):
        """Lazy load denoising model"""
    
    def _load_super_resolver(self):
        """Lazy load super-resolution model"""
    
    def _unload_models(self):
        """Unload models and clear memory"""
```

**Execution Flow**:
```python
def restore(self, image_path: str, output_path: Optional[str] = None, resume: bool = True):
    image_id = Path(image_path).stem
    
    try:
        # Step 1: Preprocessing
        if resume and self.checkpoint_mgr.has(f"{image_id}_preprocessed"):
            image, meta = self.checkpoint_mgr.load(f"{image_id}_preprocessed")
        else:
            image, meta = self.preprocessor.process(image_path)
            if self.config.processing.checkpoint_enabled:
                self.checkpoint_mgr.save(image, f"{image_id}_preprocessed", meta)
        
        # Step 2: Denoising
        if not self.config.models.denoising.skip:
            if resume and self.checkpoint_mgr.has(f"{image_id}_denoised"):
                image, _ = self.checkpoint_mgr.load(f"{image_id}_denoised")
            else:
                self._load_denoiser()
                image = self.denoiser.denoise(image)
                self._unload_models()  # Free memory
                if self.config.processing.checkpoint_enabled:
                    self.checkpoint_mgr.save(image, f"{image_id}_denoised")
        
        # Step 3: Super-resolution
        if not self.config.models.super_resolution.skip:
            if resume and self.checkpoint_mgr.has(f"{image_id}_sr"):
                image, _ = self.checkpoint_mgr.load(f"{image_id}_sr")
            else:
                self._load_super_resolver()
                image = self.super_resolver.upscale(image)
                self._unload_models()
                if self.config.processing.checkpoint_enabled:
                    self.checkpoint_mgr.save(image, f"{image_id}_sr")
        
        # Save result
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image
        
    except Exception as e:
        self.logger.error(f"Failed to restore {image_path}: {e}")
        raise
```

### 6. Memory Management

**Purpose**: Manage GPU memory efficiently

**Interface**:
```python
class MemoryManager:
    """GPU memory management utilities"""
    
    @staticmethod
    def clear_cache():
        """Clear GPU cache"""
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current GPU memory usage"""
        import torch
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3
        }
    
    @staticmethod
    def log_memory_usage(logger, prefix: str = ""):
        """Log memory usage"""
        usage = MemoryManager.get_memory_usage()
        logger.info(f"{prefix} GPU Memory - Allocated: {usage['allocated']:.2f}GB, "
                   f"Reserved: {usage['reserved']:.2f}GB")
```

### 7. Checkpoint Manager

**Purpose**: Save and load intermediate results

**Interface**:
```python
class CheckpointManager:
    """Manage processing checkpoints"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, image: np.ndarray, name: str, metadata: Optional[Dict] = None):
        """Save checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{name}.pkl"
        data = {
            'image': image,
            'metadata': metadata,
            'timestamp': time.time()
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, name: str) -> Tuple[np.ndarray, Optional[Dict]]:
        """Load checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{name}.pkl"
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        return data['image'], data.get('metadata')
    
    def has(self, name: str) -> bool:
        """Check if checkpoint exists"""
        return (self.checkpoint_dir / f"{name}.pkl").exists()
    
    def clear(self):
        """Clear all checkpoints"""
        for f in self.checkpoint_dir.glob("*.pkl"):
            f.unlink()
```

## Data Models

### Image Metadata

```python
@dataclass
class ImageMetadata:
    """Metadata for processed images"""
    original_path: str
    original_size: Tuple[int, int]  # (height, width)
    is_grayscale: bool
    resize_factor: float
    processing_steps: List[str]
    processing_time: float
    timestamp: datetime
```

### Processing Result

```python
@dataclass
class ProcessingResult:
    """Result of image restoration"""
    success: bool
    input_path: str
    output_path: Optional[str]
    restored_image: Optional[np.ndarray]
    metadata: ImageMetadata
    error: Optional[str]
```

## Error Handling

### Error Hierarchy

```python
class IMPError(Exception):
    """Base exception for IMP"""
    pass

class ConfigurationError(IMPError):
    """Configuration validation error"""
    pass

class ModelLoadError(IMPError):
    """Model loading error"""
    pass

class ProcessingError(IMPError):
    """Image processing error"""
    pass

class OutOfMemoryError(IMPError):
    """GPU out of memory error"""
    pass
```

### Error Handling Strategy

1. **Validation Errors**: Fail fast with clear messages
2. **Model Loading Errors**: Try fallback sources, log warnings
3. **Processing Errors**: Log error, save partial results, continue batch
4. **OOM Errors**: Retry with tiling, reduce batch size

## Testing Strategy

### Unit Tests

```python
# tests/test_preprocessor.py
def test_preprocessor_loads_image():
    """Test image loading"""
    
def test_preprocessor_detects_grayscale():
    """Test grayscale detection"""
    
def test_preprocessor_resizes_large_images():
    """Test smart resizing"""

# tests/test_denoiser.py
def test_opencv_denoiser_initialization():
    """Test OpenCV denoiser can be created"""
    
def test_opencv_denoiser_processes_image():
    """Test denoising produces output"""

# tests/test_pipeline.py
def test_pipeline_initialization():
    """Test pipeline can be created"""
    
def test_pipeline_with_mock_models():
    """Test pipeline flow with mocked models"""
```

### Integration Tests

```python
# tests/integration/test_full_pipeline.py
def test_end_to_end_restoration():
    """Test full restoration pipeline"""
    # Requires small test models or mocks
```

### Test Data

- Small test images (< 100KB) committed to repo
- Mock model outputs for testing without GPU
- Synthetic degraded images for validation

## Deployment Considerations

### Local Development (WSL)

- Use venv for isolation
- Install CPU-only dependencies for testing
- Run unit tests without GPU
- Use mock models for integration tests

### Google Colab

- Clone from GitHub
- Download pre-trained weights on first run
- Use GPU for inference
- Save results to Google Drive
- Implement session timeout handling

### Model Weights Management

```python
class WeightDownloader:
    """Download model weights with fallback"""
    
    SOURCES = {
        'realesrgan': [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-x4plus.pth',
            'https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth',
        ]
    }
    
    def download(self, model_name: str, save_path: str) -> bool:
        """Download with fallback sources"""
        for url in self.SOURCES[model_name]:
            try:
                urllib.request.urlretrieve(url, save_path)
                return True
            except Exception as e:
                logger.warning(f"Failed to download from {url}: {e}")
        return False
```

## Performance Targets

| Metric | Target | Environment |
|--------|--------|-------------|
| 512x512 processing time | < 5s | Colab T4 GPU |
| 1024x1024 processing time | < 20s | Colab T4 GPU |
| Peak GPU memory | < 4GB | Colab T4 GPU |
| Preprocessing time | < 0.1s | CPU |
| OpenCV denoising time | < 1s | CPU |

## Security Considerations

- Validate file paths to prevent directory traversal
- Limit maximum image size to prevent DoS
- Sanitize user inputs in configuration
- No execution of user-provided code
- Safe pickle loading (only trusted checkpoints)
