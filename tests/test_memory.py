"""
Unit tests for memory management utilities.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock torch before importing memory module
sys.modules['torch'] = MagicMock()
sys.modules['torch.cuda'] = MagicMock()

from src.utils.memory import MemoryManager


class TestMemoryManager:
    """Test suite for MemoryManager class"""
    
    @patch('src.utils.memory.TORCH_AVAILABLE', True)
    @patch('src.utils.memory.torch')
    @patch('src.utils.memory.gc')
    def test_clear_cache_with_cuda(self, mock_gc, mock_torch):
        """Test clear_cache with CUDA available"""
        mock_torch.cuda.is_available.return_value = True
        
        MemoryManager.clear_cache()
        
        # Verify garbage collection was called
        mock_gc.collect.assert_called_once()
        
        # Verify CUDA cache was cleared
        mock_torch.cuda.empty_cache.assert_called_once()
    
    @patch('src.utils.memory.TORCH_AVAILABLE', True)
    @patch('src.utils.memory.torch')
    @patch('src.utils.memory.gc')
    def test_clear_cache_without_cuda(self, mock_gc, mock_torch):
        """Test clear_cache without CUDA available"""
        mock_torch.cuda.is_available.return_value = False
        
        MemoryManager.clear_cache()
        
        # Verify garbage collection was called
        mock_gc.collect.assert_called_once()
        
        # Verify CUDA cache was not called
        mock_torch.cuda.empty_cache.assert_not_called()
    
    @patch('src.utils.memory.TORCH_AVAILABLE', False)
    @patch('src.utils.memory.gc')
    def test_clear_cache_torch_not_available(self, mock_gc):
        """Test clear_cache when torch is not available"""
        MemoryManager.clear_cache()
        
        # Should still call garbage collection
        mock_gc.collect.assert_called_once()
    
    @patch('src.utils.memory.TORCH_AVAILABLE', True)
    @patch('src.utils.memory.torch')
    def test_get_memory_usage_with_cuda(self, mock_torch):
        """Test get_memory_usage with CUDA available"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024 ** 3  # 1 GB
        mock_torch.cuda.memory_reserved.return_value = 2 * 1024 ** 3  # 2 GB
        mock_torch.cuda.max_memory_allocated.return_value = 1.5 * 1024 ** 3  # 1.5 GB
        
        usage = MemoryManager.get_memory_usage()
        
        assert usage['available'] == True
        assert usage['allocated'] == 1.0
        assert usage['reserved'] == 2.0
        assert usage['max_allocated'] == 1.5
    
    @patch('src.utils.memory.TORCH_AVAILABLE', True)
    @patch('src.utils.memory.torch')
    def test_get_memory_usage_without_cuda(self, mock_torch):
        """Test get_memory_usage without CUDA available"""
        mock_torch.cuda.is_available.return_value = False
        
        usage = MemoryManager.get_memory_usage()
        
        assert usage['available'] == False
        assert usage['allocated'] == 0.0
        assert usage['reserved'] == 0.0
        assert usage['max_allocated'] == 0.0
    
    @patch('src.utils.memory.TORCH_AVAILABLE', False)
    def test_get_memory_usage_torch_not_available(self):
        """Test get_memory_usage when torch is not available"""
        usage = MemoryManager.get_memory_usage()
        
        assert usage['available'] == False
        assert usage['allocated'] == 0.0
        assert usage['reserved'] == 0.0
        assert usage['max_allocated'] == 0.0
    
    @patch('src.utils.memory.TORCH_AVAILABLE', True)
    @patch('src.utils.memory.torch')
    def test_log_memory_usage_with_cuda(self, mock_torch):
        """Test log_memory_usage with CUDA available"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024 ** 3
        mock_torch.cuda.memory_reserved.return_value = 2 * 1024 ** 3
        mock_torch.cuda.max_memory_allocated.return_value = 1.5 * 1024 ** 3
        
        mock_logger = Mock(spec=logging.Logger)
        
        MemoryManager.log_memory_usage(mock_logger, "Test prefix")
        
        # Verify logger.info was called
        mock_logger.info.assert_called_once()
        
        # Check log message contains expected information
        log_message = mock_logger.info.call_args[0][0]
        assert "Test prefix" in log_message
        assert "GPU Memory" in log_message
        assert "Allocated: 1.00GB" in log_message
        assert "Reserved: 2.00GB" in log_message
        assert "Max Allocated: 1.50GB" in log_message
    
    @patch('src.utils.memory.TORCH_AVAILABLE', True)
    @patch('src.utils.memory.torch')
    def test_log_memory_usage_without_cuda(self, mock_torch):
        """Test log_memory_usage without CUDA available"""
        mock_torch.cuda.is_available.return_value = False
        
        mock_logger = Mock(spec=logging.Logger)
        
        MemoryManager.log_memory_usage(mock_logger, "Test prefix")
        
        # Verify logger.info was called
        mock_logger.info.assert_called_once()
        
        # Check log message indicates CUDA not available
        log_message = mock_logger.info.call_args[0][0]
        assert "Test prefix" in log_message
        assert "CUDA not available" in log_message
    
    @patch('src.utils.memory.TORCH_AVAILABLE', True)
    @patch('src.utils.memory.torch')
    def test_log_memory_usage_without_logger(self, mock_torch):
        """Test log_memory_usage without providing logger"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024 ** 3
        mock_torch.cuda.memory_reserved.return_value = 2 * 1024 ** 3
        mock_torch.cuda.max_memory_allocated.return_value = 1.5 * 1024 ** 3
        
        # Should not raise error when logger is None
        MemoryManager.log_memory_usage(None, "Test")
    
    @patch('src.utils.memory.TORCH_AVAILABLE', True)
    @patch('src.utils.memory.torch')
    def test_log_memory_usage_empty_prefix(self, mock_torch):
        """Test log_memory_usage with empty prefix"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024 ** 3
        mock_torch.cuda.memory_reserved.return_value = 2 * 1024 ** 3
        mock_torch.cuda.max_memory_allocated.return_value = 1.5 * 1024 ** 3
        
        mock_logger = Mock(spec=logging.Logger)
        
        MemoryManager.log_memory_usage(mock_logger, "")
        
        # Verify logger was called
        mock_logger.info.assert_called_once()
        
        # Log message should still be valid
        log_message = mock_logger.info.call_args[0][0]
        assert "GPU Memory" in log_message
    
    @patch('src.utils.memory.TORCH_AVAILABLE', True)
    @patch('src.utils.memory.torch')
    def test_reset_peak_stats_with_cuda(self, mock_torch):
        """Test reset_peak_stats with CUDA available"""
        mock_torch.cuda.is_available.return_value = True
        
        MemoryManager.reset_peak_stats()
        
        # Verify reset was called
        mock_torch.cuda.reset_peak_memory_stats.assert_called_once()
    
    @patch('src.utils.memory.TORCH_AVAILABLE', True)
    @patch('src.utils.memory.torch')
    def test_reset_peak_stats_without_cuda(self, mock_torch):
        """Test reset_peak_stats without CUDA available"""
        mock_torch.cuda.is_available.return_value = False
        
        MemoryManager.reset_peak_stats()
        
        # Should not call reset when CUDA not available
        mock_torch.cuda.reset_peak_memory_stats.assert_not_called()
    
    @patch('src.utils.memory.TORCH_AVAILABLE', False)
    def test_reset_peak_stats_torch_not_available(self):
        """Test reset_peak_stats when torch is not available"""
        # Should not raise error
        MemoryManager.reset_peak_stats()
    
    @patch('src.utils.memory.TORCH_AVAILABLE', True)
    @patch('src.utils.memory.torch')
    def test_memory_usage_conversion_to_gb(self, mock_torch):
        """Test memory values are correctly converted to GB"""
        mock_torch.cuda.is_available.return_value = True
        
        # Set memory in bytes
        bytes_value = 2.5 * 1024 ** 3  # 2.5 GB in bytes
        mock_torch.cuda.memory_allocated.return_value = bytes_value
        mock_torch.cuda.memory_reserved.return_value = bytes_value * 2
        mock_torch.cuda.max_memory_allocated.return_value = bytes_value * 1.5
        
        usage = MemoryManager.get_memory_usage()
        
        # Verify conversion to GB
        assert abs(usage['allocated'] - 2.5) < 0.01
        assert abs(usage['reserved'] - 5.0) < 0.01
        assert abs(usage['max_allocated'] - 3.75) < 0.01
