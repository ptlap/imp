"""
Models module for image restoration.

Provides denoising, super-resolution, and colorization model implementations:
    - denoiser: OpenCV and NAFNet-based denoising
    - super_resolution: Real-ESRGAN based upscaling
    - colorization: DDColor-based colorization for grayscale images

All models support lazy loading and GPU memory management.
"""
