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

BARREL_MIN_EXIT_VELO: float = 98.0
BARREL_MIN_LAUNCH_ANGLE: float = 26.0
BARREL_MAX_LAUNCH_ANGLE: float = 30.0

BATTER_ROLLING_FEATURES: list[str] = [
    "hr_rate", "barrel_rate", "avg_exit_velo", "avg_launch_angle",
    "k_rate", "bb_rate", "batting_avg",
]

PITCHER_ROLLING_FEATURES: list[str] = [
    "hr_allowed_rate", "barrel_rate_against", "avg_exit_velo_against",
    "k_rate", "bb_rate", "whip_proxy",
]

AL_TEAMS: list[str] = [
    "NYY", "BOS", "TOR", "BAL", "TB",
    "CLE", "CWS", "DET", "KC", "MIN",
    "HOU", "LAA", "OAK", "SEA", "TEX",
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
