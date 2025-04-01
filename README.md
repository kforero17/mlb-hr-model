# MLB Home Run Prediction Model

This project aims to predict home runs in Major League Baseball using various data sources including Statcast data, weather conditions, and ballpark factors.

## Project Structure

```
mlb-hr-prediction/
├── config/             # Configuration files
├── data/              # Raw and processed data
├── src/               # Source code
│   ├── data/         # Data collection and processing
│   ├── features/     # Feature engineering
│   ├── models/       # ML models
│   └── utils/        # Utility functions
└── tests/            # Unit tests
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:
```
OPENWEATHERMAP_API_KEY=your_key_here
```

## Usage

1. Data Collection:
```bash
python src/data/collect_data.py
```

2. Feature Engineering:
```bash
python src/features/build_features.py
```

3. Model Training:
```bash
python src/models/train_model.py
```

## Data Sources

- Statcast Data (via pybaseball)
- Weather Data (OpenWeatherMap API)
- Ballpark Factors
- Player Statistics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 