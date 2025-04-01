"""
Script to run data collection with proper Python path setup.
"""

import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run the data collection script
from src.data.collect_data import main

if __name__ == "__main__":
    main() 