import os
from typing import Union
from werkzeug.datastructures import FileStorage
from pathlib import Path

def validate_image_file(file: FileStorage) -> bool:
    """
    Validate uploaded image file.
    
    Args:
        file: Flask FileStorage object
        
    Returns:
        True if file is valid, False otherwise
    """
    try:
        # Check if file exists
        if not file or file.filename == '':
            return False
        
        # Check file extension
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            return False
        
        # Check file size (10MB limit)
        max_size = 10 * 1024 * 1024  # 10MB in bytes
        
        # Reset file pointer to beginning
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > max_size:
            return False
        
        # Check if file is empty
        if file_size == 0:
            return False
        
        return True
        
    except Exception:
        return False

def validate_filename(filename: str) -> bool:
    """
    Validate filename for security.
    
    Args:
        filename: Filename to validate
        
    Returns:
        True if filename is valid, False otherwise
    """
    try:
        if not filename:
            return False
        
        # Check for path traversal attempts
        if '..' in filename or '/' in filename or '\\' in filename:
            return False
        
        # Check for dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*']
        if any(char in filename for char in dangerous_chars):
            return False
        
        # Check length
        if len(filename) > 255:
            return False
        
        return True
        
    except Exception:
        return False

def validate_image_dimensions(file_path: str, min_width: int = 50, min_height: int = 50) -> bool:
    """
    Validate image dimensions.
    
    Args:
        file_path: Path to image file
        min_width: Minimum allowed width
        min_height: Minimum allowed height
        
    Returns:
        True if dimensions are valid, False otherwise
    """
    try:
        from PIL import Image
        
        with Image.open(file_path) as img:
            width, height = img.size
            
            if width < min_width or height < min_height:
                return False
            
            return True
            
    except Exception:
        return False

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing dangerous characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    try:
        # Remove or replace dangerous characters
        dangerous_chars = {
            '<': '_', '>': '_', ':': '_', '"': '_', 
            '|': '_', '?': '_', '*': '_', '/': '_', '\\': '_'
        }
        
        sanitized = filename
        for char, replacement in dangerous_chars.items():
            sanitized = sanitized.replace(char, replacement)
        
        # Remove multiple consecutive underscores
        while '__' in sanitized:
            sanitized = sanitized.replace('__', '_')
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Ensure filename is not empty
        if not sanitized:
            sanitized = 'image'
        
        return sanitized
        
    except Exception:
        return 'image'
