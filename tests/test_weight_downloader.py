"""
Unit tests for weight downloader utility.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import urllib.error
import sys

# Mock tqdm before importing weight_downloader
sys.modules['tqdm'] = MagicMock()

from src.utils.weight_downloader import WeightDownloader, DownloadProgressBar
from src.utils.exceptions import ConfigurationError


class TestWeightDownloader:
    """Test suite for WeightDownloader class"""
    
    def test_initialization_default_directory(self):
        """Test WeightDownloader initialization with default directory"""
        with patch('src.utils.weight_downloader.Path.mkdir'):
            downloader = WeightDownloader()
            
            assert downloader.weights_dir == Path('./weights')
    
    def test_initialization_custom_directory(self):
        """Test WeightDownloader initialization with custom directory"""
        with patch('src.utils.weight_downloader.Path.mkdir'):
            downloader = WeightDownloader(weights_dir='/custom/path')
            
            assert downloader.weights_dir == Path('/custom/path')
    
    def test_initialization_creates_directory(self):
        """Test initialization creates weights directory"""
        with patch('src.utils.weight_downloader.Path.mkdir') as mock_mkdir:
            WeightDownloader(weights_dir='/test/path')
            
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    def test_sources_contains_realesrgan_models(self):
        """Test SOURCES dict contains Real-ESRGAN models"""
        assert 'realesrgan-x4plus' in WeightDownloader.SOURCES
        assert 'realesrgan-x2plus' in WeightDownloader.SOURCES
        
        # Verify URLs are lists
        assert isinstance(WeightDownloader.SOURCES['realesrgan-x4plus'], list)
        assert len(WeightDownloader.SOURCES['realesrgan-x4plus']) > 0
    
    def test_download_unknown_model(self):
        """Test download raises ValueError for unknown model"""
        with patch('src.utils.weight_downloader.Path.mkdir'):
            downloader = WeightDownloader()
            
            with pytest.raises(ConfigurationError, match="Unknown model"):
                downloader.download('unknown-model')
    
    @patch('src.utils.weight_downloader.Path')
    def test_download_file_already_exists(self, mock_path_class):
        """Test download returns existing file path if file exists"""
        with patch('src.utils.weight_downloader.Path.mkdir'):
            downloader = WeightDownloader()
            
            # Mock file exists
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path_class.return_value = mock_path
            
            result = downloader.download('realesrgan-x4plus')
            
            # Should return path without downloading
            assert result is not None
    
    @patch('src.utils.weight_downloader.urllib.request.urlretrieve')
    @patch('src.utils.weight_downloader.DownloadProgressBar')
    def test_download_success_first_mirror(self, mock_progress_bar, mock_urlretrieve):
        """Test successful download from first mirror"""
        with patch('src.utils.weight_downloader.Path.mkdir'):
            downloader = WeightDownloader()
            
            # Create a proper mock for the save path
            save_path_mock = Mock(spec=Path)
            save_path_mock.exists.return_value = False  # File doesn't exist initially
            
            # Override the weights_dir to return our mock
            downloader.weights_dir = Mock()
            downloader.weights_dir.__truediv__ = Mock(return_value=save_path_mock)
            
            # Mock progress bar
            mock_bar = MagicMock()
            mock_progress_bar.return_value.__enter__.return_value = mock_bar
            
            result = downloader.download('realesrgan-x4plus')
            
            # Verify download was attempted
            mock_urlretrieve.assert_called_once()
            assert result is not None
    
    @patch('src.utils.weight_downloader.urllib.request.urlretrieve')
    @patch('src.utils.weight_downloader.DownloadProgressBar')
    def test_download_fallback_to_second_mirror(self, mock_progress_bar, mock_urlretrieve):
        """Test download falls back to second mirror on failure"""
        with patch('src.utils.weight_downloader.Path.mkdir'):
            downloader = WeightDownloader()
            
            # Create a proper mock for the save path
            save_path_mock = Mock(spec=Path)
            save_path_mock.exists.return_value = False
            save_path_mock.unlink = Mock()
            
            # Override the weights_dir to return our mock
            downloader.weights_dir = Mock()
            downloader.weights_dir.__truediv__ = Mock(return_value=save_path_mock)
            
            # Mock progress bar
            mock_bar = MagicMock()
            mock_progress_bar.return_value.__enter__.return_value = mock_bar
            
            # First download fails, second succeeds
            mock_urlretrieve.side_effect = [
                urllib.error.URLError("Connection failed"),
                None  # Success on second try
            ]
            
            result = downloader.download('realesrgan-x4plus')
            
            # Verify both mirrors were tried
            assert mock_urlretrieve.call_count == 2
            assert result is not None
    
    @patch('src.utils.weight_downloader.urllib.request.urlretrieve')
    @patch('src.utils.weight_downloader.DownloadProgressBar')
    def test_download_all_mirrors_fail(self, mock_progress_bar, mock_urlretrieve):
        """Test download returns None when all mirrors fail"""
        with patch('src.utils.weight_downloader.Path.mkdir'):
            downloader = WeightDownloader()
            
            # Create a proper mock for the save path
            save_path_mock = Mock(spec=Path)
            save_path_mock.exists.return_value = False
            save_path_mock.unlink = Mock()
            
            # Override the weights_dir to return our mock
            downloader.weights_dir = Mock()
            downloader.weights_dir.__truediv__ = Mock(return_value=save_path_mock)
            
            # Mock progress bar
            mock_bar = MagicMock()
            mock_progress_bar.return_value.__enter__.return_value = mock_bar
            
            # All downloads fail
            mock_urlretrieve.side_effect = urllib.error.URLError("Connection failed")
            
            result = downloader.download('realesrgan-x4plus')
            
            # Should return None after all mirrors fail
            assert result is None
    
    @patch('src.utils.weight_downloader.Path')
    def test_download_custom_save_path(self, mock_path_class):
        """Test download with custom save path"""
        with patch('src.utils.weight_downloader.Path.mkdir'):
            downloader = WeightDownloader()
            
            # Mock file exists at custom path
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path_class.return_value = mock_path
            
            result = downloader.download('realesrgan-x4plus', save_path='/custom/path.pth')
            
            assert result is not None
    
    def test_verify_checksum_no_expected_checksum(self):
        """Test _verify_checksum returns True when no checksum provided"""
        with patch('src.utils.weight_downloader.Path.mkdir'):
            downloader = WeightDownloader()
            
            result = downloader._verify_checksum(Path('/fake/path'), None)
            
            assert result == True
    
    @patch('builtins.open', create=True)
    @patch('src.utils.weight_downloader.hashlib.md5')
    def test_verify_checksum_matching(self, mock_md5, mock_open):
        """Test _verify_checksum returns True for matching checksum"""
        with patch('src.utils.weight_downloader.Path.mkdir'):
            downloader = WeightDownloader()
            
            # Mock file reading
            mock_file = MagicMock()
            mock_file.__enter__.return_value = mock_file
            mock_file.read.side_effect = [b'data', b'']  # Simulate reading chunks
            mock_open.return_value = mock_file
            
            # Mock MD5 hash
            mock_hash = Mock()
            mock_hash.hexdigest.return_value = 'abc123'
            mock_md5.return_value = mock_hash
            
            result = downloader._verify_checksum(Path('/fake/path'), 'abc123')
            
            assert result == True
    
    @patch('builtins.open', create=True)
    @patch('src.utils.weight_downloader.hashlib.md5')
    def test_verify_checksum_not_matching(self, mock_md5, mock_open):
        """Test _verify_checksum returns False for non-matching checksum"""
        with patch('src.utils.weight_downloader.Path.mkdir'):
            downloader = WeightDownloader()
            
            # Mock file reading
            mock_file = MagicMock()
            mock_file.__enter__.return_value = mock_file
            mock_file.read.side_effect = [b'data', b'']
            mock_open.return_value = mock_file
            
            # Mock MD5 hash
            mock_hash = Mock()
            mock_hash.hexdigest.return_value = 'abc123'
            mock_md5.return_value = mock_hash
            
            result = downloader._verify_checksum(Path('/fake/path'), 'different')
            
            assert result == False
    
    @patch('src.utils.weight_downloader.Path')
    def test_get_weight_path_file_exists(self, mock_path_class):
        """Test get_weight_path returns path when file exists"""
        with patch('src.utils.weight_downloader.Path.mkdir'):
            downloader = WeightDownloader()
            
            # Mock file exists
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path.__str__ = lambda self: '/weights/realesrgan-x4plus.pth'
            mock_path_class.return_value = mock_path
            
            # Override weights_dir to use mock
            downloader.weights_dir = Mock()
            downloader.weights_dir.__truediv__ = lambda self, other: mock_path
            
            result = downloader.get_weight_path('realesrgan-x4plus', auto_download=False)
            
            assert result == '/weights/realesrgan-x4plus.pth'
    
    @patch('src.utils.weight_downloader.Path')
    def test_get_weight_path_file_not_exists_no_auto_download(self, mock_path_class):
        """Test get_weight_path returns None when file doesn't exist and auto_download=False"""
        with patch('src.utils.weight_downloader.Path.mkdir'):
            downloader = WeightDownloader()
            
            # Mock file doesn't exist
            mock_path = Mock()
            mock_path.exists.return_value = False
            mock_path_class.return_value = mock_path
            
            # Override weights_dir
            downloader.weights_dir = Mock()
            downloader.weights_dir.__truediv__ = lambda self, other: mock_path
            
            result = downloader.get_weight_path('realesrgan-x4plus', auto_download=False)
            
            assert result is None
    
    @patch('src.utils.weight_downloader.WeightDownloader.download')
    @patch('src.utils.weight_downloader.Path')
    def test_get_weight_path_auto_download(self, mock_path_class, mock_download):
        """Test get_weight_path calls download when auto_download=True"""
        with patch('src.utils.weight_downloader.Path.mkdir'):
            downloader = WeightDownloader()
            
            # Mock file doesn't exist
            mock_path = Mock()
            mock_path.exists.return_value = False
            mock_path_class.return_value = mock_path
            
            # Override weights_dir
            downloader.weights_dir = Mock()
            downloader.weights_dir.__truediv__ = lambda self, other: mock_path
            
            # Mock download
            mock_download.return_value = '/weights/realesrgan-x4plus.pth'
            
            result = downloader.get_weight_path('realesrgan-x4plus', auto_download=True)
            
            # Verify download was called
            mock_download.assert_called_once_with('realesrgan-x4plus')
            assert result == '/weights/realesrgan-x4plus.pth'
