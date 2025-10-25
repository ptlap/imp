"""
Memory management utilities for GPU memory tracking and cleanup.
"""

import gc
from typing import Dict, Optional
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class MemoryManager:
    """
    GPU memory management utilities.
    
    Provides static methods for clearing GPU cache, tracking memory usage,
    and logging memory statistics. Handles cases where PyTorch or CUDA
    are not available gracefully.
    """
    
    @staticmethod
    def clear_cache():
        """
        Clear GPU cache and run garbage collection.
        
        Frees up GPU memory by clearing PyTorch's CUDA cache and running
        Python's garbage collector. Safe to call even if CUDA is not available.
        """
        # Run garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """
        Get current GPU memory usage statistics.
        
        Returns:
            Dictionary with memory statistics in GB:
            - allocated: Currently allocated GPU memory
            - reserved: Reserved GPU memory (cached)
            - max_allocated: Peak allocated memory since last reset
            - available: Returns False if CUDA not available
            
        Note:
            If CUDA is not available, returns a dict with 'available': False
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {
                'available': False,
                'allocated': 0.0,
                'reserved': 0.0,
                'max_allocated': 0.0
            }
        
        # Convert bytes to GB
        bytes_to_gb = 1024 ** 3
        
        return {
            'available': True,
            'allocated': torch.cuda.memory_allocated() / bytes_to_gb,
            'reserved': torch.cuda.memory_reserved() / bytes_to_gb,
            'max_allocated': torch.cuda.max_memory_allocated() / bytes_to_gb
        }
    
    @staticmethod
    def log_memory_usage(logger: Optional[logging.Logger] = None, prefix: str = ""):
        """
        Log current GPU memory usage.
        
        Args:
            logger: Logger instance to use. If None, uses root logger
            prefix: Optional prefix string for log message
            
        Example:
            >>> logger = logging.getLogger(__name__)
            >>> MemoryManager.log_memory_usage(logger, "After denoising")
            INFO: After denoising - GPU Memory - Allocated: 1.23GB, Reserved: 2.45GB
        """
        if logger is None:
            logger = logging.getLogger(__name__)
        
        usage = MemoryManager.get_memory_usage()
        
        if not usage.get('available', False):
            logger.info(f"{prefix} GPU Memory - CUDA not available")
            return
        
        log_msg = (
            f"{prefix} GPU Memory - "
            f"Allocated: {usage['allocated']:.2f}GB, "
            f"Reserved: {usage['reserved']:.2f}GB, "
            f"Max Allocated: {usage['max_allocated']:.2f}GB"
        )
        
        logger.info(log_msg)
    
    @staticmethod
    def reset_peak_stats():
        """
        Reset peak memory statistics.
        
        Resets the peak memory tracking to current values. Useful for
        measuring memory usage of specific operations.
        """
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
