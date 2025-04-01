"""
Tests for data collection module with mocked API calls.
"""

import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import requests_mock
from datetime import datetime
from src.data.collect_data import (
    setup_directories,
    fetch_statcast_season,
    fetch_player_data,
    get_weather,
    build_weather_dataset,
    save_data,
    main
)
from config.data_collection_config import (
    SEASONS,
    BALLPARKS,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    STATCAST_FILE,
    BATTER_FILE,
    PITCHER_FILE,
    PARK_FACTORS_FILE,
    WEATHER_FILE
)

@pytest.fixture
def mock_statcast_data():
    """Create sample Statcast data for testing."""
    return pd.DataFrame({
        'game_date': ['2023-06-01', '2023-06-02'],
        'player_name': ['Mike Trout', 'Shohei Ohtani'],
        'player_id': [545361, 669208],
        'game_pk': [123456, 123457],
        'pitch_type': ['FF', 'SL'],
        'release_speed': [95.5, 88.2],
        'release_pos_x': [-1.2, -1.1],
        'release_pos_z': [5.8, 5.7],
        'plate_x': [0.1, -0.2],
        'plate_z': [2.5, 2.4],
        'launch_speed': [105.3, 98.7],
        'launch_angle': [25.5, 22.3],
        'distance': [420, 395],
        'events': ['home_run', 'single'],
        'stand': ['R', 'L'],
        'balls': [2, 1],
        'strikes': [1, 2],
        'pitch_number': [5, 3],
        'inning': [3, 2],
        'inning_topbot': ['Top', 'Bottom'],
        'outs_when_up': [1, 2],
        'home_team': ['LAA', 'LAA'],
        'away_team': ['NYY', 'NYY'],
        'home_score': [3, 2],
        'away_score': [2, 1],
        'stadium': ['Angel Stadium', 'Angel Stadium']
    })

@pytest.fixture
def mock_weather_data():
    """Create sample weather data for testing."""
    return {
        "data": [{
            "temp": 72.5,
            "wind_speed": 8.5,
            "wind_deg": 180.0
        }]
    }

def test_setup_directories():
    """Test directory creation."""
    with patch('os.makedirs') as mock_makedirs:
        setup_directories()
        mock_makedirs.assert_any_call(RAW_DATA_DIR, exist_ok=True)
        mock_makedirs.assert_any_call(PROCESSED_DATA_DIR, exist_ok=True)

@patch('pybaseball.statcast')
def test_fetch_statcast_season(mock_statcast, mock_statcast_data):
    """Test Statcast data fetching with mocked API."""
    mock_statcast.return_value = mock_statcast_data
    
    df = fetch_statcast_season(2023, 2023)
    
    # Check data shape
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert all(col in df.columns for col in [
        'game_date', 'player_name', 'player_id', 'game_pk',
        'pitch_type', 'release_speed', 'launch_speed', 'launch_angle',
        'events', 'stand', 'home_team', 'away_team'
    ])
    
    # Check data types
    assert pd.api.types.is_numeric_dtype(df['release_speed'])
    assert pd.api.types.is_numeric_dtype(df['launch_speed'])
    assert pd.api.types.is_numeric_dtype(df['launch_angle'])

@patch('pybaseball.statcast_batter')
@patch('pybaseball.statcast_pitcher')
def test_fetch_player_data(mock_statcast_pitcher, mock_statcast_batter, mock_statcast_data):
    """Test player data fetching with mocked API."""
    mock_statcast_batter.return_value = mock_statcast_data
    mock_statcast_pitcher.return_value = mock_statcast_data
    
    player_ids = [545361, 669208]
    
    # Test batter data
    batter_df = fetch_player_data(player_ids, 2023, 'batter')
    assert isinstance(batter_df, pd.DataFrame)
    assert 'player_id' in batter_df.columns
    assert all(pid in batter_df['player_id'].values for pid in player_ids)
    
    # Test pitcher data
    pitcher_df = fetch_player_data(player_ids, 2023, 'pitcher')
    assert isinstance(pitcher_df, pd.DataFrame)
    assert 'player_id' in pitcher_df.columns
    assert all(pid in pitcher_df['player_id'].values for pid in player_ids)

def test_get_weather(requests_mock, mock_weather_data):
    """Test weather data fetching with mocked API."""
    api_key = "test_api_key"
    lat, lon = 40.8296, -73.9262
    date = "2023-06-01"
    
    # Mock the API response
    requests_mock.get(
        "http://api.openweathermap.org/data/3.0/onecall/timemachine",
        json=mock_weather_data
    )
    
    weather_data = get_weather(lat, lon, date, api_key)
    
    assert isinstance(weather_data, dict)
    assert 'temp' in weather_data
    assert 'wind_speed' in weather_data
    assert 'wind_deg' in weather_data

def test_build_weather_dataset(requests_mock, mock_weather_data):
    """Test weather dataset building with mocked API."""
    api_key = "test_api_key"
    dates = ["2023-06-01"]
    
    # Mock the API response
    requests_mock.get(
        "http://api.openweathermap.org/data/3.0/onecall/timemachine",
        json=mock_weather_data
    )
    
    df = build_weather_dataset(BALLPARKS, dates, api_key)
    
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in ['date', 'ballpark', 'temp', 'wind_speed', 'wind_deg'])
    assert len(df) == len(BALLPARKS) * len(dates)

def test_save_data(tmp_path, mock_statcast_data):
    """Test data saving functionality."""
    with patch('config.data_collection_config.RAW_DATA_DIR', str(tmp_path)), \
         patch('config.data_collection_config.PROCESSED_DATA_DIR', str(tmp_path)):
        
        raw_file = tmp_path / "test_raw.csv"
        processed_file = tmp_path / "test_processed.csv"
        
        def mock_process(df):
            df['processed'] = True
            return df
        
        save_data(mock_statcast_data, str(raw_file), str(processed_file), mock_process)
        
        # Check if files were created
        assert raw_file.exists()
        assert processed_file.exists()
        
        # Check if data was saved correctly
        saved_raw = pd.read_csv(raw_file)
        saved_processed = pd.read_csv(processed_file)
        
        assert len(saved_raw) == len(mock_statcast_data)
        assert len(saved_processed) == len(mock_statcast_data)
        assert 'processed' in saved_processed.columns

@patch('src.data.collect_data.fetch_statcast_season')
@patch('src.data.collect_data.fetch_player_data')
@patch('src.data.collect_data.park_factors')
@patch('src.data.collect_data.build_weather_dataset')
@patch('src.data.collect_data.setup_directories')
@patch('os.getenv')
def test_main_flow(
    mock_getenv,
    mock_setup_directories,
    mock_build_weather_dataset,
    mock_park_factors,
    mock_fetch_player_data,
    mock_fetch_statcast_season,
    mock_statcast_data,
    tmp_path
):
    """Test main execution flow with mocked dependencies."""
    # Setup mocks
    mock_getenv.return_value = "test_api_key"
    mock_fetch_statcast_season.return_value = mock_statcast_data
    mock_fetch_player_data.return_value = mock_statcast_data
    mock_park_factors.return_value = pd.DataFrame({
        'park_name': ['Yankee Stadium'],
        'hr_factor': [0.95],
        'hr_factor_l': [0.92],
        'hr_factor_r': [0.97],
        'year': [2023]
    })
    mock_build_weather_dataset.return_value = pd.DataFrame({
        'date': ['2023-06-01'],
        'ballpark': ['Yankee Stadium'],
        'temp': [72.5],
        'wind_speed': [8.5],
        'wind_deg': [180.0]
    })
    
    # Patch directory paths
    with patch('config.data_collection_config.RAW_DATA_DIR', str(tmp_path)), \
         patch('config.data_collection_config.PROCESSED_DATA_DIR', str(tmp_path)):
        
        # Run main function
        main()
        
        # Verify function calls
        mock_setup_directories.assert_called_once()
        mock_fetch_statcast_season.assert_called_once()
        mock_fetch_player_data.assert_called()
        mock_park_factors.assert_called_once()
        mock_build_weather_dataset.assert_called_once()
        
        # Verify files were created
        assert (tmp_path / "statcast_data.csv").exists()
        assert (tmp_path / "processed_statcast.csv").exists()
        assert (tmp_path / "park_factors.csv").exists()
        assert (tmp_path / "processed_park_factors.csv").exists()
        assert (tmp_path / "weather_data.csv").exists()
        assert (tmp_path / "processed_weather.csv").exists() 