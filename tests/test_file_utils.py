"""
Tests for file utility functions.
"""

import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.file_utils import (
    save_dataframe,
    load_dataframe,
    ensure_directory,
    get_file_size,
    list_files,
    convert_to_parquet
)

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return str(tmp_path)

def test_save_dataframe(sample_dataframe, temp_dir):
    """Test saving DataFrame to different formats."""
    # Test saving as CSV
    csv_filepath = os.path.join(temp_dir, 'test.csv')
    save_dataframe(sample_dataframe, csv_filepath, format='csv')
    assert os.path.exists(csv_filepath)
    assert os.path.getsize(csv_filepath) > 0
    
    # Test saving as Parquet
    parquet_filepath = os.path.join(temp_dir, 'test.parquet')
    save_dataframe(sample_dataframe, parquet_filepath, format='parquet')
    assert os.path.exists(parquet_filepath)
    assert os.path.getsize(parquet_filepath) > 0
    
    # Test saving empty DataFrame
    empty_df = pd.DataFrame()
    empty_filepath = os.path.join(temp_dir, 'empty.csv')
    save_dataframe(empty_df, empty_filepath)
    assert not os.path.exists(empty_filepath)
    
    # Test saving with create_dir=False to non-existent dir raises OSError
    nested_filepath = os.path.join(temp_dir, 'nested', 'test.csv')
    with pytest.raises(OSError):
        save_dataframe(sample_dataframe, nested_filepath, create_dir=False)
    
    # Test saving with create_dir=True
    save_dataframe(sample_dataframe, nested_filepath, create_dir=True)
    assert os.path.exists(nested_filepath)

def test_load_dataframe(sample_dataframe, temp_dir):
    """Test loading DataFrame from different formats."""
    # Test loading CSV
    csv_filepath = os.path.join(temp_dir, 'test.csv')
    save_dataframe(sample_dataframe, csv_filepath, format='csv')
    loaded_df = load_dataframe(csv_filepath, format='csv')
    assert loaded_df is not None
    pd.testing.assert_frame_equal(loaded_df, sample_dataframe)
    
    # Test loading Parquet
    parquet_filepath = os.path.join(temp_dir, 'test.parquet')
    save_dataframe(sample_dataframe, parquet_filepath, format='parquet')
    loaded_df = load_dataframe(parquet_filepath, format='parquet')
    assert loaded_df is not None
    pd.testing.assert_frame_equal(loaded_df, sample_dataframe)
    
    # Test format inference
    loaded_df = load_dataframe(csv_filepath)
    assert loaded_df is not None
    pd.testing.assert_frame_equal(loaded_df, sample_dataframe)
    
    loaded_df = load_dataframe(parquet_filepath)
    assert loaded_df is not None
    pd.testing.assert_frame_equal(loaded_df, sample_dataframe)
    
    # Test loading non-existent file
    non_existent_file = os.path.join(temp_dir, 'non_existent.csv')
    loaded_df = load_dataframe(non_existent_file)
    assert loaded_df is None
    
    # Test loading malformed CSV — pandas parses it leniently, so it returns a DataFrame
    invalid_file = os.path.join(temp_dir, 'invalid.csv')
    with open(invalid_file, 'w') as f:
        f.write('invalid,csv,content\n1,2,3,4')
    loaded_df = load_dataframe(invalid_file)
    assert loaded_df is not None

def test_ensure_directory(temp_dir):
    """Test directory creation."""
    # Test creating new directory
    new_dir = os.path.join(temp_dir, 'new_dir')
    ensure_directory(new_dir)
    assert os.path.exists(new_dir)
    assert os.path.isdir(new_dir)
    
    # Test creating nested directory
    nested_dir = os.path.join(temp_dir, 'nested', 'dir')
    ensure_directory(nested_dir)
    assert os.path.exists(nested_dir)
    assert os.path.isdir(nested_dir)
    
    # Test existing directory
    ensure_directory(temp_dir)
    assert os.path.exists(temp_dir)
    assert os.path.isdir(temp_dir)

def test_get_file_size(sample_dataframe, temp_dir):
    """Test getting file size."""
    # Test CSV file
    csv_filepath = os.path.join(temp_dir, 'test.csv')
    save_dataframe(sample_dataframe, csv_filepath, format='csv')
    size = get_file_size(csv_filepath)
    assert size is not None
    assert size > 0
    
    # Test Parquet file
    parquet_filepath = os.path.join(temp_dir, 'test.parquet')
    save_dataframe(sample_dataframe, parquet_filepath, format='parquet')
    size = get_file_size(parquet_filepath)
    assert size is not None
    assert size > 0
    
    # Test non-existent file
    non_existent_file = os.path.join(temp_dir, 'non_existent.csv')
    size = get_file_size(non_existent_file)
    assert size is None

def test_list_files(temp_dir):
    """Test listing files in directory."""
    # Create test files directly — save_dataframe skips empty DataFrames
    files = ['test1.csv', 'test2.parquet', 'test3.txt']
    for file in files:
        Path(os.path.join(temp_dir, file)).touch()
    
    # Test listing all files
    all_files = list_files(temp_dir)
    assert len(all_files) == len(files)
    assert all(os.path.basename(f) in files for f in all_files)
    
    # Test listing with pattern
    csv_files = list_files(temp_dir, '*.csv')
    assert len(csv_files) == 1
    assert all(f.endswith('.csv') for f in csv_files)
    
    parquet_files = list_files(temp_dir, '*.parquet')
    assert len(parquet_files) == 1
    assert all(f.endswith('.parquet') for f in parquet_files)
    
    # Test non-existent directory
    non_existent_dir = os.path.join(temp_dir, 'non_existent')
    files = list_files(non_existent_dir)
    assert len(files) == 0

def test_convert_to_parquet(sample_dataframe, temp_dir):
    """Test converting CSV to Parquet."""
    # Create a CSV file
    csv_file = os.path.join(temp_dir, 'test.csv')
    save_dataframe(sample_dataframe, csv_file, format='csv')
    
    # Test conversion with default output path
    convert_to_parquet(csv_file)
    parquet_file = os.path.splitext(csv_file)[0] + '.parquet'
    assert os.path.exists(parquet_file)
    
    # Verify the converted file
    loaded_df = load_dataframe(parquet_file, format='parquet')
    assert loaded_df is not None
    pd.testing.assert_frame_equal(loaded_df, sample_dataframe)
    
    # Test conversion with custom output path
    custom_parquet_file = os.path.join(temp_dir, 'custom.parquet')
    convert_to_parquet(csv_file, custom_parquet_file)
    assert os.path.exists(custom_parquet_file)
    
    # Test conversion of non-existent file
    non_existent_file = os.path.join(temp_dir, 'non_existent.csv')
    convert_to_parquet(non_existent_file)
    assert not os.path.exists(os.path.splitext(non_existent_file)[0] + '.parquet') 