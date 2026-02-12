from pathlib import Path

ROLLING_WINDOWS: list[int] = [15, 50]

MIN_GAMES_FOR_ROLLING: int = 5
MIN_PA_PER_GAME: int = 1

PLATE_APPEARANCE_EVENTS: list[str] = [
    "single", "double", "triple", "home_run",
    "field_out", "grounded_into_double_play", "force_out",
    "fielders_choice", "fielders_choice_out", "double_play",
    "strikeout", "strikeout_double_play",
    "walk", "hit_by_pitch", "intent_walk",
    "field_error", "sac_fly", "sac_bunt",
    "sac_fly_double_play", "sac_bunt_double_play",
    "catcher_interf", "triple_play",
]

HIT_EVENTS: list[str] = ["single", "double", "triple", "home_run"]

AT_BAT_EXCLUSIONS: list[str] = [
    "walk", "hit_by_pitch", "intent_walk",
    "sac_fly", "sac_bunt", "sac_fly_double_play",
    "sac_bunt_double_play", "catcher_interf",
]

STATCAST_BARREL_CODE: int = 6

HARD_HIT_MIN_EXIT_VELO: float = 95.0
SWEET_SPOT_MIN_ANGLE: float = 8.0
SWEET_SPOT_MAX_ANGLE: float = 32.0
SPRAY_ANGLE_PULL_THRESHOLD: float = 15.0
SPRAY_HOME_X: float = 125.42
SPRAY_HOME_Y: float = 198.27

FASTBALL_TYPES: set[str] = {"FF", "SI", "FC", "FA"}
BREAKING_TYPES: set[str] = {"SL", "CU", "KC", "ST", "SV", "CS", "KN"}
OFFSPEED_TYPES: set[str] = {"CH", "FS", "EP"}

BATTER_ROLLING_FEATURES: list[str] = [
    "hr_rate", "barrel_rate", "avg_exit_velo", "avg_launch_angle",
    "k_rate", "bb_rate", "batting_avg",
]

PITCHER_ROLLING_FEATURES: list[str] = [
    "hr_allowed_rate", "barrel_rate_against", "avg_exit_velo_against",
    "k_rate", "bb_rate", "whip_proxy",
]

DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURE_MATRIX_PATH = PROCESSED_DATA_DIR / "feature_matrix.parquet"

PA_CONTEXT_COLUMNS: list[str] = [
    "inning",
    "outs_when_up",
    "score_diff",
    "is_home",
    "runners_on_base",
    "pa_number_in_game",
]

ENVIRONMENT_FEATURES: list[str] = [
    "park_hr_factor",
    "park_hr_factor_handedness",
    "elevation_ft",
    "roof_type",
    "temp_f",
    "wind_speed_mph",
    "wind_dir_deg",
    "humidity_pct",
    "wind_out_to_cf",
    "air_density_index",
]

WEATHER_DATA_PATH = RAW_DATA_DIR / "weather_games.parquet"

OPPORTUNITY_FEATURES: list[str] = [
    "batting_order_pos",
    "batting_order_avg",
    "team_runs_per_game",
    "expected_pas",
]

BATTER_SKILL_FEATURES: list[str] = [
    "fly_ball_rate",
    "pull_rate",
    "hard_hit_rate",
    "sweet_spot_pct",
    "avg_xslg",
    "avg_xwoba",
    "hr_rate_vs_fastball",
    "hr_rate_vs_breaking",
    "hr_rate_vs_offspeed",
]

PITCHER_SKILL_FEATURES: list[str] = [
    "gb_rate",
    "fb_rate",
    "fastball_pct",
    "breaking_pct",
    "offspeed_pct",
]
