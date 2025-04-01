"""
Tests for data processing functions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.data.processing import (
    process_statcast_data,
    process_batter_data,
    process_pitcher_data,
    process_park_factors,
    process_weather_data
)

def create_sample_statcast_data():
    """Create sample Statcast data for testing."""
    return pd.DataFrame({
        'game_date': ['2023-06-01', '2023-06-02', '2023-06-03'],
        'player_name': ['Mike Trout', 'Shohei Ohtani', 'Aaron Judge'],
        'player_id': [545361, 669208, 677594],
        'game_pk': [123456, 123457, 123458],
        'pitch_type': ['FF', 'SL', 'FF'],
        'release_speed': [95.5, 88.2, 97.8],
        'release_pos_x': [-1.2, -1.1, -1.3],
        'release_pos_z': [5.8, 5.7, 5.9],
        'plate_x': [0.1, -0.2, 0.3],
        'plate_z': [2.5, 2.4, 2.6],
        'launch_speed': [105.3, 98.7, 106.5],
        'launch_angle': [25.5, 22.3, 28.7],
        'distance': [420, 395, 450],
        'events': ['home_run', 'single', 'home_run'],
        'stand': ['R', 'L', 'R'],
        'balls': [2, 1, 3],
        'strikes': [1, 2, 0],
        'pitch_number': [5, 3, 7],
        'inning': [3, 2, 4],
        'inning_topbot': ['Top', 'Bottom', 'Top'],
        'outs_when_up': [1, 2, 0],
        'home_team': ['LAA', 'LAA', 'NYY'],
        'away_team': ['NYY', 'NYY', 'BOS'],
        'home_score': [3, 2, 5],
        'away_score': [2, 1, 3],
        'stadium': ['Angel Stadium', 'Angel Stadium', 'Yankee Stadium']
    })

def test_process_statcast_data():
    """Test Statcast data processing."""
    df = create_sample_statcast_data()
    processed_df = process_statcast_data(df)
    
    # Check datetime conversion
    assert isinstance(processed_df['game_date'].iloc[0], pd.Timestamp)
    
    # Check derived features
    assert 'is_home_run' in processed_df.columns
    assert 'is_barrel' in processed_df.columns
    assert 'velocity_diff' in processed_df.columns
    assert 'score_diff' in processed_df.columns
    assert 'runners_on' in processed_df.columns
    
    # Check time-based features
    assert 'hour' in processed_df.columns
    assert 'month' in processed_df.columns
    assert 'day_of_week' in processed_df.columns
    
    # Check specific values
    assert processed_df['is_home_run'].iloc[0] == True
    assert processed_df['score_diff'].iloc[0] == 1
    assert processed_df['runners_on'].iloc[0] == 2
    
    # Check edge cases
    assert processed_df['velocity_diff'].iloc[0] == np.nan  # First pitch in game
    assert processed_df['velocity_diff'].iloc[1] == -7.3  # Difference from previous pitch
    assert processed_df['is_barrel'].iloc[2] == True  # High launch speed and angle
    assert processed_df['is_barrel'].iloc[1] == False  # Lower launch speed and angle

def test_process_statcast_data_edge_cases():
    """Test Statcast data processing with edge cases."""
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    processed_df = process_statcast_data(empty_df)
    assert processed_df.empty
    
    # Test with missing columns
    df = create_sample_statcast_data()
    df = df.drop(columns=['launch_speed', 'launch_angle'])
    processed_df = process_statcast_data(df)
    assert 'is_barrel' not in processed_df.columns
    
    # Test with invalid data types
    df = create_sample_statcast_data()
    df['release_speed'] = df['release_speed'].astype(str)
    processed_df = process_statcast_data(df)
    assert processed_df['velocity_diff'].isna().all()

def test_process_batter_data():
    """Test batter data processing."""
    df = create_sample_statcast_data()
    processed_df = process_batter_data(df)
    
    # Check datetime conversion
    assert isinstance(processed_df['game_date'].iloc[0], pd.Timestamp)
    
    # Check derived features
    assert 'is_home_run' in processed_df.columns
    assert 'is_barrel' in processed_df.columns
    assert 'score_diff' in processed_df.columns
    assert 'runners_on' in processed_df.columns
    
    # Check time-based features
    assert 'hour' in processed_df.columns
    assert 'month' in processed_df.columns
    assert 'day_of_week' in processed_df.columns
    
    # Check specific values
    assert processed_df['is_home_run'].iloc[0] == True
    assert processed_df['score_diff'].iloc[0] == 1
    assert processed_df['runners_on'].iloc[0] == 2

def test_process_pitcher_data():
    """Test pitcher data processing."""
    df = create_sample_statcast_data()
    processed_df = process_pitcher_data(df)
    
    # Check datetime conversion
    assert isinstance(processed_df['game_date'].iloc[0], pd.Timestamp)
    
    # Check derived features
    assert 'velocity_diff' in processed_df.columns
    assert 'pitch_height' in processed_df.columns
    assert 'pitch_distance' in processed_df.columns
    assert 'score_diff' in processed_df.columns
    assert 'runners_on' in processed_df.columns
    
    # Check time-based features
    assert 'hour' in processed_df.columns
    assert 'month' in processed_df.columns
    assert 'day_of_week' in processed_df.columns
    
    # Check specific values
    assert processed_df['velocity_diff'].iloc[0] == np.nan  # First pitch in game
    assert processed_df['velocity_diff'].iloc[1] == -7.3  # Difference from previous pitch
    assert processed_df['pitch_height'].iloc[0] == -3.3  # plate_z - release_pos_z
    assert processed_df['pitch_distance'].iloc[0] == pytest.approx(3.31, rel=1e-2)  # sqrt(x^2 + h^2)

def test_process_park_factors():
    """Test park factors processing."""
    df = pd.DataFrame({
        'park_name': ['Yankee Stadium', 'Coors Field', 'Petco Park'],
        'hr_factor': [0.95, 1.15, 0.90],
        'hr_factor_l': [0.92, 1.12, 0.88],
        'hr_factor_r': [0.97, 1.18, 0.91],
        'year': [2023, 2023, 2023]
    })
    processed_df = process_park_factors(df)
    
    # Check derived features
    assert 'hr_factor_diff' in processed_df.columns
    assert 'hr_factor_l_diff' in processed_df.columns
    assert 'hr_factor_r_diff' in processed_df.columns
    assert 'park_type' in processed_df.columns
    
    # Check specific values
    assert processed_df['hr_factor_diff'].iloc[0] == -0.05
    assert processed_df['park_type'].iloc[0] == 'Pitcher'
    assert processed_df['park_type'].iloc[1] == 'Hitter'
    assert processed_df['park_type'].iloc[2] == 'Pitcher'

def test_process_park_factors_edge_cases():
    """Test park factors processing with edge cases."""
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    processed_df = process_park_factors(empty_df)
    assert processed_df.empty
    
    # Test with missing columns
    df = pd.DataFrame({
        'park_name': ['Yankee Stadium'],
        'hr_factor': [0.95],
        'year': [2023]
    })
    processed_df = process_park_factors(df)
    assert 'hr_factor_l_diff' not in processed_df.columns
    assert 'hr_factor_r_diff' not in processed_df.columns
    
    # Test with invalid values
    df = pd.DataFrame({
        'park_name': ['Yankee Stadium'],
        'hr_factor': [np.nan],
        'hr_factor_l': [0.92],
        'hr_factor_r': [0.97],
        'year': [2023]
    })
    processed_df = process_park_factors(df)
    assert processed_df['park_type'].iloc[0] == 'Neutral'  # Default for NaN

def test_process_weather_data():
    """Test weather data processing."""
    df = pd.DataFrame({
        'date': ['2023-06-01', '2023-06-02', '2023-06-03'],
        'ballpark': ['Yankee Stadium', 'Coors Field', 'Petco Park'],
        'temp': [72.5, 68.3, 75.8],
        'wind_speed': [8.5, 12.2, 5.8],
        'wind_deg': [180.0, 225.0, 45.0]
    })
    processed_df = process_weather_data(df)
    
    # Check datetime conversion
    assert isinstance(processed_df['date'].iloc[0], pd.Timestamp)
    
    # Check derived features
    assert 'is_wind_favorable' in processed_df.columns
    assert 'temp_category' in processed_df.columns
    
    # Check time-based features
    assert 'month' in processed_df.columns
    assert 'day_of_week' in processed_df.columns
    
    # Check specific values
    assert processed_df['temp_category'].iloc[0] == 'Warm'
    assert processed_df['temp_category'].iloc[1] == 'Mild'
    assert processed_df['temp_category'].iloc[2] == 'Warm'
    assert processed_df['is_wind_favorable'].iloc[0] == False  # High wind speed
    assert processed_df['is_wind_favorable'].iloc[2] == True  # Low wind speed, favorable direction

def test_process_weather_data_edge_cases():
    """Test weather data processing with edge cases."""
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    processed_df = process_weather_data(empty_df)
    assert processed_df.empty
    
    # Test with missing columns
    df = pd.DataFrame({
        'date': ['2023-06-01'],
        'ballpark': ['Yankee Stadium'],
        'temp': [72.5]
    })
    processed_df = process_weather_data(df)
    assert 'is_wind_favorable' not in processed_df.columns
    
    # Test with invalid values
    df = pd.DataFrame({
        'date': ['2023-06-01'],
        'ballpark': ['Yankee Stadium'],
        'temp': [np.nan],
        'wind_speed': [8.5],
        'wind_deg': [180.0]
    })
    processed_df = process_weather_data(df)
    assert processed_df['temp_category'].iloc[0] == 'Cold'  # Default for NaN 