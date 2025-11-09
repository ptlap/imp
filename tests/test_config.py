"""
Unit tests for configuration management.
"""

import pytest
import tempfile
from pathlib import Path
import yaml

from src.config import (
    Config,
    DenoisingConfig,
    SuperResolutionConfig,
    ColorizationConfig,
    FaceDetectionConfig,
    FaceEnhancementConfig,
    ModelsConfig,
    ProcessingConfig,
    LoggingConfig
)
from src.utils.exceptions import ConfigurationError


class TestDenoisingConfig:
    """Test suite for DenoisingConfig"""
    
    def test_default_values(self):
        """Test DenoisingConfig has correct default values"""
        config = DenoisingConfig()
        
        assert config.type == "opencv"
        assert config.strength == 10
        assert config.skip == False
    
    def test_custom_values(self):
        """Test DenoisingConfig with custom values"""
        config = DenoisingConfig(type="nafnet", strength=20, skip=True)
        
        assert config.type == "nafnet"
        assert config.strength == 20
        assert config.skip == True


class TestSuperResolutionConfig:
    """Test suite for SuperResolutionConfig"""
    
    def test_default_values(self):
        """Test SuperResolutionConfig has correct default values"""
        config = SuperResolutionConfig()
        
        assert config.type == "realesrgan"
        assert config.scale == 4
        assert config.tile_size == 512
        assert config.tile_overlap == 64
        assert config.use_fp16 == True
        assert config.skip == False
        assert "realesrgan" in config.weights_url.lower()
    
    def test_custom_values(self):
        """Test SuperResolutionConfig with custom values"""
        config = SuperResolutionConfig(
            scale=2,
            tile_size=256,
            tile_overlap=32,
            use_fp16=False,
            skip=True
        )
        
        assert config.scale == 2
        assert config.tile_size == 256
        assert config.tile_overlap == 32
        assert config.use_fp16 == False
        assert config.skip == True


class TestColorizationConfig:
    """Test suite for ColorizationConfig"""
    
    def test_default_values(self):
        """Test ColorizationConfig has correct default values"""
        config = ColorizationConfig()
        
        assert config.type == "ddcolor"
        assert config.weights_path is None
        assert "ddcolor" in config.weights_url.lower()
        assert config.skip == False
        assert config.use_fp16 == True
    
    def test_custom_values(self):
        """Test ColorizationConfig with custom values"""
        config = ColorizationConfig(
            weights_path="/custom/weights.pth",
            skip=True,
            use_fp16=False
        )
        
        assert config.weights_path == "/custom/weights.pth"
        assert config.skip == True
        assert config.use_fp16 == False


class TestFaceDetectionConfig:
    """Test suite for FaceDetectionConfig"""
    
    def test_default_values(self):
        """Test FaceDetectionConfig has correct default values"""
        config = FaceDetectionConfig()
        
        assert config.type == "retinaface"
        assert config.weights_path is None
        assert "retinaface" in config.weights_url.lower()
        assert config.confidence_threshold == 0.8
        assert config.max_faces == 10
        assert config.skip == False
        assert config.use_fp16 == True
    
    def test_custom_values(self):
        """Test FaceDetectionConfig with custom values"""
        config = FaceDetectionConfig(
            weights_path="/custom/weights.pth",
            confidence_threshold=0.9,
            max_faces=5,
            skip=True,
            use_fp16=False
        )
        
        assert config.weights_path == "/custom/weights.pth"
        assert config.confidence_threshold == 0.9
        assert config.max_faces == 5
        assert config.skip == True
        assert config.use_fp16 == False


class TestFaceEnhancementConfig:
    """Test suite for FaceEnhancementConfig"""
    
    def test_default_values(self):
        """Test FaceEnhancementConfig has correct default values"""
        config = FaceEnhancementConfig()
        
        assert config.type == "codeformer"
        assert config.weights_path is None
        assert "codeformer" in config.weights_url.lower()
        assert config.fidelity == 0.7
        assert config.skip == False
        assert config.use_fp16 == True
    
    def test_custom_values(self):
        """Test FaceEnhancementConfig with custom values"""
        config = FaceEnhancementConfig(
            weights_path="/custom/weights.pth",
            fidelity=0.5,
            skip=True,
            use_fp16=False
        )
        
        assert config.weights_path == "/custom/weights.pth"
        assert config.fidelity == 0.5
        assert config.skip == True
        assert config.use_fp16 == False


class TestProcessingConfig:
    """Test suite for ProcessingConfig"""
    
    def test_default_values(self):
        """Test ProcessingConfig has correct default values"""
        config = ProcessingConfig()
        
        assert config.max_image_size == 2048
        assert config.save_intermediate == False
        assert config.checkpoint_enabled == True
        assert config.checkpoint_dir == "./checkpoints"
    
    def test_custom_values(self):
        """Test ProcessingConfig with custom values"""
        config = ProcessingConfig(
            max_image_size=4096,
            save_intermediate=True,
            checkpoint_enabled=False,
            checkpoint_dir="/custom/path"
        )
        
        assert config.max_image_size == 4096
        assert config.save_intermediate == True
        assert config.checkpoint_enabled == False
        assert config.checkpoint_dir == "/custom/path"


class TestLoggingConfig:
    """Test suite for LoggingConfig"""
    
    def test_default_values(self):
        """Test LoggingConfig has correct default values"""
        config = LoggingConfig()
        
        assert config.level == "INFO"
        assert config.file == "imp.log"
    
    def test_custom_values(self):
        """Test LoggingConfig with custom values"""
        config = LoggingConfig(level="DEBUG", file="/custom/log.txt")
        
        assert config.level == "DEBUG"
        assert config.file == "/custom/log.txt"


class TestConfig:
    """Test suite for main Config class"""
    
    def test_default_config(self):
        """Test Config.default() creates config with default values"""
        config = Config.default()
        
        assert isinstance(config.models, ModelsConfig)
        assert isinstance(config.models.denoising, DenoisingConfig)
        assert isinstance(config.models.super_resolution, SuperResolutionConfig)
        assert isinstance(config.models.colorization, ColorizationConfig)
        assert isinstance(config.models.face_detection, FaceDetectionConfig)
        assert isinstance(config.models.face_enhancement, FaceEnhancementConfig)
        assert isinstance(config.processing, ProcessingConfig)
        assert isinstance(config.logging, LoggingConfig)
    
    def test_config_initialization(self):
        """Test Config can be initialized with custom sub-configs"""
        denoising = DenoisingConfig(strength=15)
        super_res = SuperResolutionConfig(scale=2)
        models = ModelsConfig(denoising=denoising, super_resolution=super_res)
        processing = ProcessingConfig(max_image_size=4096)
        logging = LoggingConfig(level="DEBUG")
        
        config = Config(models=models, processing=processing, logging=logging)
        
        assert config.models.denoising.strength == 15
        assert config.models.super_resolution.scale == 2
        assert config.processing.max_image_size == 4096
        assert config.logging.level == "DEBUG"
    
    def test_from_yaml_valid_config(self):
        """Test loading valid YAML configuration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_content = """
models:
  denoising:
    type: opencv
    strength: 15
    skip: false
  super_resolution:
    type: realesrgan
    scale: 2
    tile_size: 256
    tile_overlap: 32
    use_fp16: false
    skip: false
  colorization:
    type: ddcolor
    skip: false
    use_fp16: true
  face_detection:
    type: retinaface
    confidence_threshold: 0.9
    max_faces: 5
    skip: false
    use_fp16: true
  face_enhancement:
    type: codeformer
    fidelity: 0.5
    skip: false
    use_fp16: true

processing:
  max_image_size: 4096
  save_intermediate: true
  checkpoint_enabled: true
  checkpoint_dir: ./test_checkpoints

logging:
  level: DEBUG
  file: test.log
"""
            f.write(yaml_content)
            config_path = f.name
        
        try:
            config = Config.from_yaml(config_path)
            
            assert config.models.denoising.strength == 15
            assert config.models.super_resolution.scale == 2
            assert config.models.super_resolution.tile_size == 256
            assert config.models.colorization.type == "ddcolor"
            assert config.models.face_detection.confidence_threshold == 0.9
            assert config.models.face_detection.max_faces == 5
            assert config.models.face_enhancement.fidelity == 0.5
            assert config.processing.max_image_size == 4096
            assert config.processing.save_intermediate == True
            assert config.logging.level == "DEBUG"
        finally:
            Path(config_path).unlink()
    
    def test_from_yaml_partial_config(self):
        """Test loading YAML with only some fields specified"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_content = """
models:
  denoising:
    strength: 20
"""
            f.write(yaml_content)
            config_path = f.name
        
        try:
            config = Config.from_yaml(config_path)
            
            # Specified value
            assert config.models.denoising.strength == 20
            # Default values
            assert config.models.denoising.type == "opencv"
            assert config.models.super_resolution.scale == 4
            assert config.processing.max_image_size == 2048
        finally:
            Path(config_path).unlink()
    
    def test_from_yaml_empty_file(self):
        """Test loading empty YAML file uses defaults"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            config_path = f.name
        
        try:
            config = Config.from_yaml(config_path)
            
            # Should have all default values
            assert config.models.denoising.strength == 10
            assert config.models.super_resolution.scale == 4
            assert config.processing.max_image_size == 2048
        finally:
            Path(config_path).unlink()
    
    def test_from_yaml_nonexistent_file(self):
        """Test loading nonexistent file raises ConfigurationError"""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            Config.from_yaml("nonexistent.yaml")
    
    def test_from_yaml_invalid_yaml(self):
        """Test loading invalid YAML raises ConfigurationError"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="Failed to parse YAML"):
                Config.from_yaml(config_path)
        finally:
            Path(config_path).unlink()
    
    def test_validate_valid_config(self):
        """Test validation passes for valid config"""
        config = Config.default()
        
        assert config.validate() == True
    
    def test_validate_invalid_denoising_type(self):
        """Test validation fails for invalid denoising type"""
        config = Config.default()
        config.models.denoising.type = "invalid"
        
        with pytest.raises(ConfigurationError, match="Invalid denoising type"):
            config.validate()
    
    def test_validate_invalid_denoising_strength_low(self):
        """Test validation fails for denoising strength too low"""
        config = Config.default()
        config.models.denoising.strength = 0
        
        with pytest.raises(ConfigurationError, match="Invalid denoising strength"):
            config.validate()
    
    def test_validate_invalid_denoising_strength_high(self):
        """Test validation fails for denoising strength too high"""
        config = Config.default()
        config.models.denoising.strength = 101
        
        with pytest.raises(ConfigurationError, match="Invalid denoising strength"):
            config.validate()
    
    def test_validate_invalid_sr_type(self):
        """Test validation fails for invalid super-resolution type"""
        config = Config.default()
        config.models.super_resolution.type = "invalid"
        
        with pytest.raises(ConfigurationError, match="Invalid super-resolution type"):
            config.validate()
    
    def test_validate_invalid_sr_scale(self):
        """Test validation fails for invalid super-resolution scale"""
        config = Config.default()
        config.models.super_resolution.scale = 3
        
        with pytest.raises(ConfigurationError, match="Invalid super-resolution scale"):
            config.validate()
    
    def test_validate_invalid_tile_size_low(self):
        """Test validation fails for tile_size too low"""
        config = Config.default()
        config.models.super_resolution.tile_size = 32
        
        with pytest.raises(ConfigurationError, match="Invalid tile_size"):
            config.validate()
    
    def test_validate_invalid_tile_size_high(self):
        """Test validation fails for tile_size too high"""
        config = Config.default()
        config.models.super_resolution.tile_size = 4096
        
        with pytest.raises(ConfigurationError, match="Invalid tile_size"):
            config.validate()
    
    def test_validate_invalid_tile_overlap(self):
        """Test validation fails for tile_overlap >= tile_size"""
        config = Config.default()
        config.models.super_resolution.tile_overlap = 512
        
        with pytest.raises(ConfigurationError, match="Invalid tile_overlap"):
            config.validate()
    
    def test_validate_invalid_max_image_size_low(self):
        """Test validation fails for max_image_size too low"""
        config = Config.default()
        config.processing.max_image_size = 100
        
        with pytest.raises(ConfigurationError, match="Invalid max_image_size"):
            config.validate()
    
    def test_validate_invalid_max_image_size_high(self):
        """Test validation fails for max_image_size too high"""
        config = Config.default()
        config.processing.max_image_size = 10000
        
        with pytest.raises(ConfigurationError, match="Invalid max_image_size"):
            config.validate()
    
    def test_validate_invalid_log_level(self):
        """Test validation fails for invalid log level"""
        config = Config.default()
        config.logging.level = "INVALID"
        
        with pytest.raises(ConfigurationError, match="Invalid logging level"):
            config.validate()
    
    def test_validate_invalid_colorization_type(self):
        """Test validation fails for invalid colorization type"""
        config = Config.default()
        config.models.colorization.type = "invalid"
        
        with pytest.raises(ConfigurationError, match="Invalid colorization type"):
            config.validate()
    
    def test_validate_invalid_face_detection_type(self):
        """Test validation fails for invalid face detection type"""
        config = Config.default()
        config.models.face_detection.type = "invalid"
        
        with pytest.raises(ConfigurationError, match="Invalid face detection type"):
            config.validate()
    
    def test_validate_invalid_confidence_threshold_low(self):
        """Test validation fails for confidence threshold too low"""
        config = Config.default()
        config.models.face_detection.confidence_threshold = -0.1
        
        with pytest.raises(ConfigurationError, match="Invalid confidence_threshold"):
            config.validate()
    
    def test_validate_invalid_confidence_threshold_high(self):
        """Test validation fails for confidence threshold too high"""
        config = Config.default()
        config.models.face_detection.confidence_threshold = 1.1
        
        with pytest.raises(ConfigurationError, match="Invalid confidence_threshold"):
            config.validate()
    
    def test_validate_invalid_max_faces_low(self):
        """Test validation fails for max_faces too low"""
        config = Config.default()
        config.models.face_detection.max_faces = 0
        
        with pytest.raises(ConfigurationError, match="Invalid max_faces"):
            config.validate()
    
    def test_validate_invalid_max_faces_high(self):
        """Test validation fails for max_faces too high"""
        config = Config.default()
        config.models.face_detection.max_faces = 21
        
        with pytest.raises(ConfigurationError, match="Invalid max_faces"):
            config.validate()
    
    def test_validate_invalid_face_enhancement_type(self):
        """Test validation fails for invalid face enhancement type"""
        config = Config.default()
        config.models.face_enhancement.type = "invalid"
        
        with pytest.raises(ConfigurationError, match="Invalid face enhancement type"):
            config.validate()
    
    def test_validate_invalid_fidelity_low(self):
        """Test validation fails for fidelity too low"""
        config = Config.default()
        config.models.face_enhancement.fidelity = -0.1
        
        with pytest.raises(ConfigurationError, match="Invalid fidelity"):
            config.validate()
    
    def test_validate_invalid_fidelity_high(self):
        """Test validation fails for fidelity too high"""
        config = Config.default()
        config.models.face_enhancement.fidelity = 1.1
        
        with pytest.raises(ConfigurationError, match="Invalid fidelity"):
            config.validate()
    
    def test_validate_multiple_errors(self):
        """Test validation reports multiple errors"""
        config = Config.default()
        config.models.denoising.strength = 0
        config.models.super_resolution.scale = 3
        config.logging.level = "INVALID"
        
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        
        error_msg = str(exc_info.value)
        assert "denoising strength" in error_msg
        assert "super-resolution scale" in error_msg
        assert "logging level" in error_msg
