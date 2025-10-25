"""
Utility for downloading model weights with fallback sources.
"""

import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional
import hashlib
from tqdm import tqdm
import logging

from .exceptions import ModelLoadError, ConfigurationError

logger = logging.getLogger('imp.weight_downloader')


class DownloadProgressBar(tqdm):
    """Progress bar for file downloads using tqdm."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Update progress bar.
        
        Args:
            b: Number of blocks transferred
            bsize: Size of each block (in bytes)
            tsize: Total size (in bytes)
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class WeightDownloader:
    """
    Download model weights with fallback sources and integrity verification.
    
    Supports multiple mirror URLs with automatic fallback if download fails.
    Includes progress bar and optional file integrity verification.
    """
    
    # Mirror URLs for Real-ESRGAN weights
    SOURCES = {
        'realesrgan-x4plus': [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-x4plus.pth',
            'https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4plus.pth',
        ],
        'realesrgan-x2plus': [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        ]
    }
    
    # Optional MD5 checksums for integrity verification
    CHECKSUMS = {
        'realesrgan-x4plus': None,  # Add checksum if available
        'realesrgan-x2plus': None,
    }
    
    def __init__(self, weights_dir: str = './weights'):
        """
        Initialize weight downloader.
        
        Args:
            weights_dir: Directory to save downloaded weights
        """
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"WeightDownloader initialized with directory: {self.weights_dir}")
    
    def download(
        self,
        model_name: str,
        save_path: Optional[str] = None,
        verify_checksum: bool = False
    ) -> Optional[str]:
        """
        Download model weights with fallback sources.
        
        Tries each mirror URL in sequence until successful download.
        Optionally verifies file integrity using MD5 checksum.
        
        Args:
            model_name: Name of model (e.g., 'realesrgan-x4plus')
            save_path: Custom save path (optional, defaults to weights_dir/model_name.pth)
            verify_checksum: Verify downloaded file integrity
            
        Returns:
            Path to downloaded file if successful, None otherwise
            
        Raises:
            ConfigurationError: If model_name is not recognized
        """
        if model_name not in self.SOURCES:
            raise ConfigurationError(
                f"Unknown model: {model_name}. "
                f"Available models: {', '.join(self.SOURCES.keys())}"
            )
        
        # Determine save path
        if save_path is None:
            save_path = self.weights_dir / f"{model_name}.pth"
        else:
            save_path = Path(save_path)
        
        # Check if file already exists
        if save_path.exists():
            logger.info(f"Weights already exist at {save_path}")
            if verify_checksum and self.CHECKSUMS.get(model_name):
                if self._verify_checksum(save_path, self.CHECKSUMS[model_name]):
                    logger.info("Checksum verified")
                    return str(save_path)
                else:
                    logger.warning("Checksum verification failed, re-downloading...")
                    save_path.unlink()
            else:
                return str(save_path)
        
        # Try each mirror URL
        urls = self.SOURCES[model_name]
        for i, url in enumerate(urls):
            logger.info(f"Attempting download from mirror {i+1}/{len(urls)}: {url}")
            
            try:
                # Download with progress bar
                with DownloadProgressBar(
                    unit='B',
                    unit_scale=True,
                    miniters=1,
                    desc=f"Downloading {model_name}"
                ) as t:
                    urllib.request.urlretrieve(
                        url,
                        save_path,
                        reporthook=t.update_to
                    )
                
                logger.info(f"Successfully downloaded to {save_path}")
                
                # Verify checksum if requested
                if verify_checksum and self.CHECKSUMS.get(model_name):
                    logger.debug("Verifying checksum...")
                    if self._verify_checksum(save_path, self.CHECKSUMS[model_name]):
                        logger.info("Checksum verified")
                    else:
                        logger.warning("Checksum verification failed")
                        save_path.unlink()
                        continue
                
                return str(save_path)
                
            except urllib.error.URLError as e:
                logger.warning(f"Failed to download from {url}: {e}")
                if save_path.exists():
                    save_path.unlink()
                continue
            except Exception as e:
                logger.error(f"Unexpected error downloading from {url}: {e}", exc_info=True)
                if save_path.exists():
                    save_path.unlink()
                continue
        
        logger.error(f"Failed to download {model_name} from all mirrors")
        return None
    
    def _verify_checksum(self, file_path: Path, expected_md5: str) -> bool:
        """
        Verify file integrity using MD5 checksum.
        
        Args:
            file_path: Path to file to verify
            expected_md5: Expected MD5 hash
            
        Returns:
            True if checksum matches, False otherwise
        """
        if not expected_md5:
            return True
        
        md5_hash = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        
        actual_md5 = md5_hash.hexdigest()
        return actual_md5.lower() == expected_md5.lower()
    
    def get_weight_path(self, model_name: str, auto_download: bool = True) -> Optional[str]:
        """
        Get path to model weights, downloading if necessary.
        
        Args:
            model_name: Name of model
            auto_download: Automatically download if not found
            
        Returns:
            Path to weights file if available, None otherwise
        """
        weight_path = self.weights_dir / f"{model_name}.pth"
        
        if weight_path.exists():
            return str(weight_path)
        
        if auto_download:
            return self.download(model_name)
        
        return None
