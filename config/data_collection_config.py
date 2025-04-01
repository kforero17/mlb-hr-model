"""
Configuration parameters for data collection.
"""

from typing import Dict, List

# Seasons to collect data for
SEASONS: List[int] = [2024]

# Ballpark coordinates for weather data
BALLPARKS: Dict[str, Dict[str, float]] = {
    "Yankee Stadium": {"lat": 40.8296, "lon": -73.9262},
    "Fenway Park": {"lat": 42.3467, "lon": -71.0972},
    "Rogers Centre": {"lat": 43.6414, "lon": -79.3894},
    "Oriole Park at Camden Yards": {"lat": 39.2839, "lon": -76.6217},
    "Tropicana Field": {"lat": 27.7683, "lon": -82.6534},
    "Guaranteed Rate Field": {"lat": 41.8300, "lon": -87.6339},
    "Progressive Field": {"lat": 41.4962, "lon": -81.6852},
    "Comerica Park": {"lat": 42.3390, "lon": -83.0485},
    "Kauffman Stadium": {"lat": 39.0517, "lon": -94.4803},
    "Target Field": {"lat": 44.9817, "lon": -93.2775},
    "Minute Maid Park": {"lat": 29.7573, "lon": -95.3554},
    "Angel Stadium": {"lat": 33.8003, "lon": -117.8827},
    "Oakland Coliseum": {"lat": 37.7516, "lon": -122.2005},
    "T-Mobile Park": {"lat": 47.5915, "lon": -122.3317},
    "Globe Life Field": {"lat": 32.7473, "lon": -97.0944},
    "Truist Park": {"lat": 33.8908, "lon": -84.4678},
    "LoanDepot Park": {"lat": 25.7780, "lon": -80.2195},
    "Citi Field": {"lat": 40.7571, "lon": -73.8458},
    "Citizens Bank Park": {"lat": 39.9057, "lon": -75.1665},
    "Nationals Park": {"lat": 38.8729, "lon": -77.0074},
    "Wrigley Field": {"lat": 41.9484, "lon": -87.6553},
    "Great American Ball Park": {"lat": 39.0978, "lon": -84.5076},
    "American Family Field": {"lat": 43.0280, "lon": -87.9711},
    "PNC Park": {"lat": 40.4469, "lon": -80.0057},
    "Busch Stadium": {"lat": 38.6226, "lon": -90.1928},
    "Chase Field": {"lat": 33.4455, "lon": -112.0667},
    "Coors Field": {"lat": 39.7559, "lon": -104.9942},
    "Dodger Stadium": {"lat": 34.0739, "lon": -118.2400},
    "Petco Park": {"lat": 32.7076, "lon": -117.1570},
    "Oracle Park": {"lat": 37.7786, "lon": -122.3893}
}

# API rate limiting parameters
RATE_LIMIT_DELAY: float = 0.3  # seconds between API calls
WEATHER_RATE_LIMIT_DELAY: float = 1.0  # seconds between weather API calls

# Date ranges for data collection
SEASON_START: str = "03-28"  # 2024 Opening Day
SEASON_END: str = "03-28"    # Starting with just Opening Day for initial test

# File paths for data storage
DATA_DIR: str = "data"
RAW_DATA_DIR: str = f"{DATA_DIR}/raw"
PROCESSED_DATA_DIR: str = f"{DATA_DIR}/processed"

# Raw data files
STATCAST_FILE: str = f"{RAW_DATA_DIR}/statcast_data.csv"
BATTER_FILE: str = f"{RAW_DATA_DIR}/batter_data.csv"
PITCHER_FILE: str = f"{RAW_DATA_DIR}/pitcher_data.csv"
PARK_FACTORS_FILE: str = f"{RAW_DATA_DIR}/park_factors.csv"
WEATHER_FILE: str = f"{RAW_DATA_DIR}/weather_data.csv" 