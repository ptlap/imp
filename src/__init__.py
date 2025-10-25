"""
IMP - Image Restoration Project

A deep learning-based system for automatic old photo restoration.
Provides denoising and super-resolution capabilities using pre-trained models.

Main Components:
    - pipeline: Main orchestrator for image restoration
    - config: Configuration management
    - models: Denoising and super-resolution modules
    - utils: Preprocessing, checkpointing, memory management, and logging utilities

Example:
    >>> from src.pipeline import OldPhotoRestoration
    >>> from src.config import Config
    >>> 
    >>> config = Config.default()
    >>> pipeline = OldPhotoRestoration(config)
    >>> restored = pipeline.restore('old_photo.jpg', 'restored.png')
"""

__version__ = "0.1.0"
