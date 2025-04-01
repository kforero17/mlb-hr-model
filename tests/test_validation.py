"""
Tests for data validation schemas.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.validation import (
    validate_statcast_data,
    validate_batter_data,
    validate_pitcher_data,
    validate_park_factors,
    validate_weather_data
)

def create_sample_statcast_data():
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

def create_sample_batter_data():
    """Create sample batter data for testing."""
    return pd.DataFrame({
        'game_date': ['2023-06-01', '2023-06-02'],
        'player_id': [545361, 669208],
        'player_name': ['Mike Trout', 'Shohei Ohtani'],
        'game_pk': [123456, 123457],
        'pitch_type': ['FF', 'SL'],
        'release_speed': [95.5, 88.2],
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

def create_sample_pitcher_data():
    """Create sample pitcher data for testing."""
    return pd.DataFrame({
        'game_date': ['2023-06-01', '2023-06-02'],
        'player_id': [669208, 669208],
        'player_name': ['Shohei Ohtani', 'Shohei Ohtani'],
        'game_pk': [123456, 123457],
        'pitch_type': ['FF', 'SL'],
        'release_speed': [95.5, 88.2],
        'release_pos_x': [-1.2, -1.1],
        'release_pos_z': [5.8, 5.7],
        'plate_x': [0.1, -0.2],
        'plate_z': [2.5, 2.4],
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

def create_sample_park_factors():
    """Create sample park factors data for testing."""
    return pd.DataFrame({
        'park_name': ['Yankee Stadium', 'Coors Field'],
        'hr_factor': [0.95, 1.15],
        'hr_factor_l': [0.92, 1.12],
        'hr_factor_r': [0.97, 1.18],
        'year': [2023, 2023]
    })

def create_sample_weather_data():
    """Create sample weather data for testing."""
    return pd.DataFrame({
        'date': ['2023-06-01', '2023-06-02'],
        'ballpark': ['Yankee Stadium', 'Coors Field'],
        'temp': [72.5, 68.3],
        'wind_speed': [8.5, 12.2],
        'wind_deg': [180.0, 225.0]
    })

def test_validate_statcast_data():
    """Test Statcast data validation."""
    df = create_sample_statcast_data()
    validated_df = validate_statcast_data(df)
    assert isinstance(validated_df, pd.DataFrame)
    assert len(validated_df) == 2
    assert all(col in validated_df.columns for col in df.columns)

def test_validate_batter_data():
    """Test batter data validation."""
    df = create_sample_batter_data()
    validated_df = validate_batter_data(df)
    assert isinstance(validated_df, pd.DataFrame)
    assert len(validated_df) == 2
    assert all(col in validated_df.columns for col in df.columns)

def test_validate_pitcher_data():
    """Test pitcher data validation."""
    df = create_sample_pitcher_data()
    validated_df = validate_pitcher_data(df)
    assert isinstance(validated_df, pd.DataFrame)
    assert len(validated_df) == 2
    assert all(col in validated_df.columns for col in df.columns)

def test_validate_park_factors():
    """Test park factors data validation."""
    df = create_sample_park_factors()
    validated_df = validate_park_factors(df)
    assert isinstance(validated_df, pd.DataFrame)
    assert len(validated_df) == 2
    assert all(col in validated_df.columns for col in df.columns)

def test_validate_weather_data():
    """Test weather data validation."""
    df = create_sample_weather_data()
    validated_df = validate_weather_data(df)
    assert isinstance(validated_df, pd.DataFrame)
    assert len(validated_df) == 2
    assert all(col in validated_df.columns for col in df.columns)

def test_invalid_data():
    """Test validation with invalid data."""
    # Test with missing required columns
    invalid_df = pd.DataFrame({
        'game_date': ['2023-06-01'],
        'player_name': ['Mike Trout']
    })
    
    with pytest.raises(ValueError):
        validate_statcast_data(invalid_df)
    
    # Test with wrong data types
    invalid_df = pd.DataFrame({
        'game_date': ['2023-06-01'],
        'player_name': ['Mike Trout'],
        'player_id': ['not_a_number'],
        'game_pk': [123456],
        'pitch_type': ['FF'],
        'release_speed': ['not_a_number'],
        'release_pos_x': [-1.2],
        'release_pos_z': [5.8],
        'plate_x': [0.1],
        'plate_z': [2.5],
        'launch_speed': [105.3],
        'launch_angle': [25.5],
        'distance': [420],
        'events': ['home_run'],
        'stand': ['R'],
        'balls': [2],
        'strikes': [1],
        'pitch_number': [5],
        'inning': [3],
        'inning_topbot': ['Top'],
        'outs_when_up': [1],
        'home_team': ['LAA'],
        'away_team': ['NYY'],
        'home_score': [3],
        'away_score': [2],
        'stadium': ['Angel Stadium']
    })
    
    with pytest.raises(ValueError):
        validate_statcast_data(invalid_df) 