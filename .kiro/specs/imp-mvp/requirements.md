# Requirements Document

## Introduction

IMP (Image Restoration Project) là một hệ thống phục chế ảnh cũ tự động sử dụng Deep Learning. Hệ thống sẽ nhận đầu vào là ảnh cũ/hư hỏng và tự động thực hiện các bước xử lý để tạo ra ảnh đã được phục hồi với chất lượng cao hơn.

Phiên bản MVP (Minimum Viable Product) tập trung vào hai chức năng cốt lõi: khử nhiễu và tăng độ phân giải, sử dụng các pre-trained models để đảm bảo có thể hoàn thành trong thời gian ngắn.

## Glossary

- **IMP System**: Hệ thống phục chế ảnh IMP (Image Restoration Project)
- **Degraded Image**: Ảnh đầu vào bị hư hỏng, nhiễu, độ phân giải thấp
- **Restored Image**: Ảnh đầu ra đã được phục hồi
- **Preprocessing Module**: Module tiền xử lý ảnh
- **Denoising Module**: Module khử nhiễu
- **Super-Resolution Module**: Module tăng độ phân giải
- **Pipeline**: Chuỗi xử lý tuần tự các modules
- **Pre-trained Model**: Model đã được train sẵn, không cần train lại
- **Colab Environment**: Môi trường Google Colab với GPU
- **Local Environment**: Môi trường development trên máy local (WSL)
- **Checkpoint**: Điểm lưu trữ trung gian trong quá trình xử lý
- **Tiling**: Kỹ thuật chia ảnh lớn thành các tiles nhỏ để xử lý

## Requirements

### Requirement 1: Project Setup and Structure

**User Story:** As a developer, I want to have a well-organized project structure with proper environment setup, so that I can develop and test the code efficiently on both local (WSL) and Colab environments.

#### Acceptance Criteria

1. THE IMP System SHALL create a project directory structure with separate folders for source code, tests, notebooks, configs, and documentation
2. THE IMP System SHALL provide a virtual environment setup using venv for local WSL development
3. THE IMP System SHALL include a requirements.txt file listing all Python dependencies with version constraints
4. THE IMP System SHALL provide setup instructions for both WSL local environment and Google Colab environment
5. WHERE the developer runs setup on WSL, THE IMP System SHALL create and activate a virtual environment using bash commands

### Requirement 2: Image Preprocessing

**User Story:** As a user, I want the system to automatically prepare my input images for processing, so that images of various formats and sizes can be handled correctly.

#### Acceptance Criteria

1. WHEN an image file path is provided, THE Preprocessing Module SHALL load the image and validate it is a supported format (JPG, PNG, JPEG)
2. THE Preprocessing Module SHALL detect whether the input image is grayscale or color by comparing RGB channel values
3. WHEN the image dimensions exceed 2048 pixels in any dimension, THE Preprocessing Module SHALL resize the image while maintaining aspect ratio
4. THE Preprocessing Module SHALL normalize pixel values to the range [0, 1] for model input
5. THE Preprocessing Module SHALL return both the processed image tensor and metadata including original dimensions and grayscale status

### Requirement 3: Image Denoising

**User Story:** As a user, I want the system to remove noise and artifacts from my old photos, so that the images appear cleaner and clearer.

#### Acceptance Criteria

1. THE Denoising Module SHALL provide OpenCV FastNlMeans denoising as the default lightweight option
2. WHEN processing an image, THE Denoising Module SHALL apply denoising with configurable strength parameter (default 10)
3. THE Denoising Module SHALL process color images using fastNlMeansDenoisingColored with appropriate parameters
4. THE Denoising Module SHALL complete processing of a 512x512 image within 1 second on CPU
5. WHERE GPU is available, THE Denoising Module SHALL support optional NAFNet model for higher quality denoising

### Requirement 4: Super-Resolution

**User Story:** As a user, I want the system to increase the resolution of my low-quality photos, so that I can see more details and have larger printable images.

#### Acceptance Criteria

1. THE Super-Resolution Module SHALL use Real-ESRGAN pre-trained model for upscaling images
2. THE Super-Resolution Module SHALL support 2x and 4x upscaling factors
3. WHEN the input image is larger than 2048x2048 pixels, THE Super-Resolution Module SHALL apply tiling strategy with 512-pixel tiles and 64-pixel overlap
4. THE Super-Resolution Module SHALL use FP16 (half precision) inference when GPU is available to reduce memory usage
5. THE Super-Resolution Module SHALL download pre-trained weights from multiple mirror sources with fallback options

### Requirement 5: Pipeline Integration

**User Story:** As a user, I want to process my images through a single unified pipeline, so that I can easily restore photos without managing individual steps.

#### Acceptance Criteria

1. THE Pipeline SHALL execute preprocessing, denoising, and super-resolution modules in sequence
2. THE Pipeline SHALL provide options to skip individual modules through configuration parameters
3. WHEN processing fails at any step, THE Pipeline SHALL log the error and provide a meaningful error message
4. THE Pipeline SHALL support batch processing of multiple images with progress tracking
5. THE Pipeline SHALL implement a checkpoint system to save intermediate results after each module

### Requirement 6: Memory Management

**User Story:** As a developer, I want the system to manage GPU memory efficiently, so that it can run on Google Colab free tier without out-of-memory errors.

#### Acceptance Criteria

1. THE IMP System SHALL implement lazy model loading to load models only when needed
2. WHEN a module completes processing, THE IMP System SHALL unload the model and clear GPU cache
3. THE IMP System SHALL maintain peak GPU memory usage below 4GB for images up to 1024x1024 pixels
4. WHEN processing large images, THE IMP System SHALL automatically apply tiling to prevent memory overflow
5. THE IMP System SHALL provide memory usage monitoring and logging during processing

### Requirement 7: Configuration Management

**User Story:** As a user, I want to configure processing parameters without modifying code, so that I can easily adjust settings for different types of images.

#### Acceptance Criteria

1. THE IMP System SHALL load configuration from a YAML file specifying model types, parameters, and processing options
2. THE Configuration SHALL include settings for denoising strength, super-resolution scale, and tiling parameters
3. WHEN no configuration file is provided, THE IMP System SHALL use sensible default values
4. THE Configuration SHALL support enabling or disabling individual modules
5. THE IMP System SHALL validate configuration values and provide error messages for invalid settings

### Requirement 8: Error Handling and Logging

**User Story:** As a developer, I want comprehensive error handling and logging, so that I can debug issues and monitor system behavior.

#### Acceptance Criteria

1. THE IMP System SHALL log all major operations including model loading, image processing steps, and errors
2. WHEN an error occurs, THE IMP System SHALL log the full error message and stack trace
3. THE IMP System SHALL provide different log levels (DEBUG, INFO, WARNING, ERROR)
4. THE IMP System SHALL save logs to both console output and a log file
5. WHEN processing multiple images in batch mode, THE IMP System SHALL continue processing remaining images after an error and log all failures

### Requirement 9: Testing Infrastructure

**User Story:** As a developer, I want automated tests for core functionality, so that I can verify the system works correctly and catch regressions early.

#### Acceptance Criteria

1. THE IMP System SHALL include unit tests for preprocessing, denoising, and super-resolution modules
2. THE Tests SHALL verify that modules can be initialized without errors
3. THE Tests SHALL verify that image processing produces output of expected dimensions
4. THE Tests SHALL run on CPU without requiring GPU or large model weights
5. THE Tests SHALL use pytest framework and be executable with a single command

### Requirement 10: Documentation

**User Story:** As a new developer or user, I want clear documentation, so that I can understand how to set up, use, and contribute to the project.

#### Acceptance Criteria

1. THE IMP System SHALL include a README.md with project overview, quick start guide, and installation instructions
2. THE Documentation SHALL provide separate setup instructions for WSL local development and Google Colab
3. THE Documentation SHALL include code examples demonstrating basic usage of the pipeline
4. THE Source Code SHALL include docstrings for all public classes and functions following Python conventions
5. THE Documentation SHALL include a development workflow guide explaining the local-to-Colab workflow
