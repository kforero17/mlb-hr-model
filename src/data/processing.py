"""
Data processing and transformation functions.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

def process_statcast_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and clean Statcast data.
    
    Args:
        df: Raw Statcast DataFrame
        
    Returns:
        Processed DataFrame
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Convert date strings to datetime
    processed_df['game_date'] = pd.to_datetime(processed_df['game_date'])
    
    # Add derived features
    processed_df['is_home_run'] = processed_df['events'] == 'home_run'
    if 'launch_speed_angle' in processed_df.columns:
        processed_df['is_barrel'] = processed_df['launch_speed_angle'] == 6
    else:
        processed_df['is_barrel'] = False

    # Calculate pitch velocity features
    processed_df['velocity_diff'] = processed_df.groupby('game_pk')['release_speed'].diff()
    
    # Calculate game state features
    processed_df['score_diff'] = processed_df['home_score'] - processed_df['away_score']
    processed_df['runners_on'] = processed_df['outs_when_up'].apply(lambda x: 3 - x)
    
    # Add time-based features
    processed_df['hour'] = processed_df['game_date'].dt.hour
    processed_df['month'] = processed_df['game_date'].dt.month
    processed_df['day_of_week'] = processed_df['game_date'].dt.dayofweek
    
    return processed_df

def process_batter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and clean batter-specific data.
    
    Args:
        df: Raw batter DataFrame
        
    Returns:
        Processed DataFrame
    """
    processed_df = df.copy()
    
    # Convert date strings to datetime
    processed_df['game_date'] = pd.to_datetime(processed_df['game_date'])
    
    # Add derived features
    processed_df['is_home_run'] = processed_df['events'] == 'home_run'
    if 'launch_speed_angle' in processed_df.columns:
        processed_df['is_barrel'] = processed_df['launch_speed_angle'] == 6
    else:
        processed_df['is_barrel'] = False

    # Calculate game state features
    processed_df['score_diff'] = processed_df['home_score'] - processed_df['away_score']
    processed_df['runners_on'] = processed_df['outs_when_up'].apply(lambda x: 3 - x)

    # Add time-based features
    processed_df['hour'] = processed_df['game_date'].dt.hour
    processed_df['month'] = processed_df['game_date'].dt.month
    processed_df['day_of_week'] = processed_df['game_date'].dt.dayofweek

    return processed_df

def process_pitcher_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and clean pitcher-specific data.
    
    Args:
        df: Raw pitcher DataFrame
        
    Returns:
        Processed DataFrame
    """
    processed_df = df.copy()
    
    # Convert date strings to datetime
    processed_df['game_date'] = pd.to_datetime(processed_df['game_date'])
    
    # Calculate pitch velocity features
    processed_df['velocity_diff'] = processed_df.groupby('game_pk')['release_speed'].diff()
    
    # Calculate pitch location features
    processed_df['pitch_height'] = processed_df['plate_z'] - processed_df['release_pos_z']
    processed_df['pitch_distance'] = np.sqrt(
        processed_df['plate_x']**2 + processed_df['pitch_height']**2
    )
    
    # Calculate game state features
    processed_df['score_diff'] = processed_df['home_score'] - processed_df['away_score']
    processed_df['runners_on'] = processed_df['outs_when_up'].apply(lambda x: 3 - x)
    
    # Add time-based features
    processed_df['hour'] = processed_df['game_date'].dt.hour
    processed_df['month'] = processed_df['game_date'].dt.month
    processed_df['day_of_week'] = processed_df['game_date'].dt.dayofweek
    
    return processed_df

def process_park_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and clean park factors data.
    
    Args:
        df: Raw park factors DataFrame
        
    Returns:
        Processed DataFrame
    """
    processed_df = df.copy()
    
    # Calculate park factor differences
    processed_df['hr_factor_diff'] = processed_df['hr_factor'] - 1.0
    processed_df['hr_factor_l_diff'] = processed_df['hr_factor_l'] - 1.0
    processed_df['hr_factor_r_diff'] = processed_df['hr_factor_r'] - 1.0
    
    # Add park type categorization
    processed_df['park_type'] = pd.cut(
        processed_df['hr_factor'],
        bins=[-np.inf, 0.95, 1.05, np.inf],
        labels=['Pitcher', 'Neutral', 'Hitter']
    )
    
    return processed_df

def process_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and clean weather data.
    
    Args:
        df: Raw weather DataFrame
        
    Returns:
        Processed DataFrame
    """
    processed_df = df.copy()
    
    # Convert date strings to datetime
    processed_df['date'] = pd.to_datetime(processed_df['date'])
    
    # Add derived features
    processed_df['is_wind_favorable'] = (
        (processed_df['wind_speed'] < 10) & 
        (processed_df['wind_deg'].between(0, 45) | processed_df['wind_deg'].between(315, 360))
    )
    
    # Add temperature categories
    processed_df['temp_category'] = pd.cut(
        processed_df['temp'],
        bins=[-np.inf, 60, 70, 80, np.inf],
        labels=['Cold', 'Mild', 'Warm', 'Hot']
    )
    
    # Add time-based features
    processed_df['month'] = processed_df['date'].dt.month
    processed_df['day_of_week'] = processed_df['date'].dt.dayofweek
    
    return processed_df 