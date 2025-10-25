"""
Logging configuration and setup utilities for IMP system.

Provides centralized logging configuration with file and console handlers,
configurable log levels, and consistent formatting across all modules.
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = 'imp',
    level: str = 'INFO',
    log_file: Optional[str] = 'imp.log',
    console_output: bool = True
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Creates a logger with consistent formatting for both file and console output.
    Supports configurable log levels and optional file/console output.
    
    Args:
        name: Logger name (default: 'imp')
        level: Log level as string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Path to log file. If None, no file handler is added
        console_output: If True, add console handler for stdout
        
    Returns:
        Configured logger instance
        
    Raises:
        ValueError: If log level is invalid
    """
    # Validate log level
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if level not in valid_levels:
        raise ValueError(f"Invalid log level: {level}. Must be one of {valid_levels}")
    
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter with timestamp and level
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level))
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    
    # Add file handler if log_file is provided
    if log_file:
        # Create log directory if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str = 'imp') -> logging.Logger:
    """
    Get existing logger by name.
    
    Retrieves a logger that was previously configured with setup_logger().
    If the logger doesn't exist, returns a basic logger.
    
    Args:
        name: Logger name (default: 'imp')
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
