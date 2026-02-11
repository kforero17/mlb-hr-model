import pandera as pa
from pandera.typing import Series, DataFrame


class StatcastSchema(pa.DataFrameModel):
    game_date: Series[str]
    player_name: Series[str]
    player_id: Series[int]
    game_pk: Series[int]
    pitch_type: Series[str] = pa.Field(nullable=True)
    release_speed: Series[float] = pa.Field(nullable=True)
    release_pos_x: Series[float] = pa.Field(nullable=True)
    release_pos_z: Series[float] = pa.Field(nullable=True)
    plate_x: Series[float] = pa.Field(nullable=True)
    plate_z: Series[float] = pa.Field(nullable=True)
    launch_speed: Series[float] = pa.Field(nullable=True)
    launch_angle: Series[float] = pa.Field(nullable=True)
    distance: Series[float] = pa.Field(nullable=True)
    events: Series[str] = pa.Field(nullable=True)
    stand: Series[str] = pa.Field(nullable=True)
    balls: Series[int]
    strikes: Series[int]
    pitch_number: Series[int]
    inning: Series[int]
    inning_topbot: Series[str]
    outs_when_up: Series[int]
    home_team: Series[str]
    away_team: Series[str]
    home_score: Series[int]
    away_score: Series[int]
    stadium: Series[str] = pa.Field(nullable=True)

    class Config:
        coerce = True
        strict = False


class BatterSchema(pa.DataFrameModel):
    game_date: Series[str]
    player_id: Series[int]
    player_name: Series[str]
    game_pk: Series[int]
    pitch_type: Series[str] = pa.Field(nullable=True)
    release_speed: Series[float] = pa.Field(nullable=True)
    launch_speed: Series[float] = pa.Field(nullable=True)
    launch_angle: Series[float] = pa.Field(nullable=True)
    distance: Series[float] = pa.Field(nullable=True)
    events: Series[str] = pa.Field(nullable=True)
    stand: Series[str] = pa.Field(nullable=True)
    balls: Series[int]
    strikes: Series[int]
    pitch_number: Series[int]
    inning: Series[int]
    inning_topbot: Series[str]
    outs_when_up: Series[int]
    home_team: Series[str]
    away_team: Series[str]
    home_score: Series[int]
    away_score: Series[int]
    stadium: Series[str] = pa.Field(nullable=True)

    class Config:
        coerce = True
        strict = False


class PitcherSchema(pa.DataFrameModel):
    game_date: Series[str]
    player_id: Series[int]
    player_name: Series[str]
    game_pk: Series[int]
    pitch_type: Series[str] = pa.Field(nullable=True)
    release_speed: Series[float] = pa.Field(nullable=True)
    release_pos_x: Series[float] = pa.Field(nullable=True)
    release_pos_z: Series[float] = pa.Field(nullable=True)
    plate_x: Series[float] = pa.Field(nullable=True)
    plate_z: Series[float] = pa.Field(nullable=True)
    stand: Series[str] = pa.Field(nullable=True)
    balls: Series[int]
    strikes: Series[int]
    pitch_number: Series[int]
    inning: Series[int]
    inning_topbot: Series[str]
    outs_when_up: Series[int]
    home_team: Series[str]
    away_team: Series[str]
    home_score: Series[int]
    away_score: Series[int]
    stadium: Series[str] = pa.Field(nullable=True)

    class Config:
        coerce = True
        strict = False


class ParkFactorsSchema(pa.DataFrameModel):
    park_name: Series[str]
    hr_factor: Series[float]
    hr_factor_l: Series[float]
    hr_factor_r: Series[float]
    year: Series[int]

    class Config:
        coerce = True
        strict = False


class WeatherSchema(pa.DataFrameModel):
    date: Series[str]
    ballpark: Series[str]
    temp: Series[float]
    wind_speed: Series[float]
    wind_deg: Series[float]

    class Config:
        coerce = True
        strict = False


def validate_statcast_data(df: DataFrame) -> DataFrame:
    try:
        return StatcastSchema.validate(df)
    except pa.errors.SchemaError as e:
        raise ValueError(f"Statcast data validation failed: {e}")


def validate_batter_data(df: DataFrame) -> DataFrame:
    try:
        return BatterSchema.validate(df)
    except pa.errors.SchemaError as e:
        raise ValueError(f"Batter data validation failed: {e}")


def validate_pitcher_data(df: DataFrame) -> DataFrame:
    try:
        return PitcherSchema.validate(df)
    except pa.errors.SchemaError as e:
        raise ValueError(f"Pitcher data validation failed: {e}")


def validate_park_factors(df: DataFrame) -> DataFrame:
    try:
        return ParkFactorsSchema.validate(df)
    except pa.errors.SchemaError as e:
        raise ValueError(f"Park factors data validation failed: {e}")


def validate_weather_data(df: DataFrame) -> DataFrame:
    try:
        return WeatherSchema.validate(df)
    except pa.errors.SchemaError as e:
        raise ValueError(f"Weather data validation failed: {e}")
