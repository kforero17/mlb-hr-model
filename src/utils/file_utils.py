"""
Utility functions for file operations.
"""

import os
import logging
from typing import Optional, Literal
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

FileFormat = Literal['csv', 'parquet']

def save_dataframe(
    df: pd.DataFrame,
    filepath: str,
    create_dir: bool = True,
    format: FileFormat = 'parquet'
) -> None:
    """
    Save a DataFrame to a file.
    
    Args:
        df: DataFrame to save
        filepath: Path where to save the file
        create_dir: Whether to create parent directories if they don't exist
        format: File format to use ('csv' or 'parquet')
    """
    if not df.empty:
        if create_dir:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if format == 'parquet':
            df.to_parquet(filepath, index=False)
            logger.info(f"Saved {len(df)} rows to {filepath} (Parquet)")
        else:
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(df)} rows to {filepath} (CSV)")
    else:
        logger.warning(f"No data to save to {filepath}")

def load_dataframe(
    filepath: str,
    format: Optional[FileFormat] = None
) -> Optional[pd.DataFrame]:
    """
    Load a DataFrame from a file.
    
    Args:
        filepath: Path to the file
        format: File format to use ('csv' or 'parquet'). If None, inferred from extension
        
    Returns:
        DataFrame if file exists and is valid, None otherwise
    """
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return None
    
    try:
        # Infer format from extension if not specified
        if format is None:
            format = 'parquet' if filepath.endswith('.parquet') else 'csv'
        
        if format == 'parquet':
            df = pd.read_parquet(filepath)
            logger.info(f"Loaded {len(df)} rows from {filepath} (Parquet)")
        else:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows from {filepath} (CSV)")
        return df
    except Exception as e:
        logger.error(f"Error loading {filepath}: {str(e)}")
        return None

def ensure_directory(directory: str) -> None:
    """
    Ensure a directory exists, create it if it doesn't.
    
    Args:
        directory: Path to the directory
    """
    os.makedirs(directory, exist_ok=True)
    logger.debug(f"Ensured directory exists: {directory}")

def get_file_size(filepath: str) -> Optional[int]:
    """
    Get the size of a file in bytes.
    
    Args:
        filepath: Path to the file
        
    Returns:
        File size in bytes if file exists, None otherwise
    """
    if not os.path.exists(filepath):
        return None
    return os.path.getsize(filepath)

def list_files(directory: str, pattern: str = "*") -> list[str]:
    """
    List files in a directory matching a pattern.
    
    Args:
        directory: Directory to search in
        pattern: Glob pattern to match
        
    Returns:
        List of matching file paths
    """
    if not os.path.exists(directory):
        return []
    
    return [str(f) for f in Path(directory).glob(pattern)]

def convert_to_parquet(csv_file: str, parquet_file: Optional[str] = None) -> None:
    """
    Convert a CSV file to Parquet format.
    
    Args:
        csv_file: Path to the CSV file
        parquet_file: Path for the Parquet file. If None, uses same name as CSV with .parquet extension
    """
    if not os.path.exists(csv_file):
        logger.warning(f"CSV file not found: {csv_file}")
        return
    
    if parquet_file is None:
        parquet_file = os.path.splitext(csv_file)[0] + '.parquet'
    
    try:
        df = pd.read_csv(csv_file)
        df.to_parquet(parquet_file, index=False)
        logger.info(f"Converted {csv_file} to {parquet_file}")
    except Exception as e:
        logger.error(f"Error converting {csv_file} to Parquet: {str(e)}") 