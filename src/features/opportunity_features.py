import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_PAS_INTERCEPT = 4.72
BASE_PAS_SLOPE = -0.065
LEAGUE_AVG_RUNS = 4.5
TEAM_FACTOR_FLOOR = 0.7
TEAM_FACTOR_CEIL = 1.4
HOME_PA_ADJUSTMENT = -0.06
DEFAULT_LINEUP_POS = 5.0
BATTING_ORDER_ROLLING_WINDOW = 15
TEAM_RUNS_MIN_PERIODS = 5


def derive_batting_order(raw_df: pd.DataFrame) -> pd.DataFrame:
    raw_df = raw_df.copy()
    raw_df["game_date"] = pd.to_datetime(raw_df["game_date"])

    events_df = raw_df[raw_df["events"].notna()].copy()

    home_batters = _assign_order_for_side(events_df, "Bot", is_home=True)
    away_batters = _assign_order_for_side(events_df, "Top", is_home=False)

    result = pd.concat([home_batters, away_batters], ignore_index=True)

    logger.info(
        "Batting order derived: %d batter-game rows across %d games",
        len(result),
        result["game_pk"].nunique(),
    )
    return result


def _assign_order_for_side(
    events_df: pd.DataFrame,
    topbot_value: str,
    is_home: bool,
) -> pd.DataFrame:
    side_df = events_df[events_df["inning_topbot"] == topbot_value].copy()

    first_ab = (
        side_df.groupby(["game_pk", "batter"])["at_bat_number"]
        .min()
        .reset_index()
        .rename(columns={"at_bat_number": "first_ab"})
    )

    first_ab["batting_order_pos"] = (
        first_ab.groupby("game_pk")["first_ab"]
        .rank(method="dense")
        .astype(int)
    )
    first_ab["batting_order_pos"] = first_ab["batting_order_pos"].clip(upper=9)

    game_dates = (
        side_df.groupby("game_pk")["game_date"]
        .first()
        .reset_index()
    )
    first_ab = first_ab.merge(game_dates, on="game_pk", how="left")

    return first_ab[["batter", "game_pk", "game_date", "batting_order_pos"]]


def _compute_batting_order_avg(batting_order_df: pd.DataFrame) -> pd.DataFrame:
    df = batting_order_df.sort_values(["batter", "game_date"]).copy()

    df["batting_order_avg"] = df.groupby("batter")["batting_order_pos"].transform(
        lambda s: s.rolling(
            window=BATTING_ORDER_ROLLING_WINDOW, min_periods=1
        ).mean().shift(1)
    )

    return df[["batter", "game_pk", "batting_order_avg"]]


def compute_team_rolling_runs(
    raw_df: pd.DataFrame, window: int = 30
) -> pd.DataFrame:
    raw_df = raw_df.copy()
    raw_df["game_date"] = pd.to_datetime(raw_df["game_date"])

    game_scores = _derive_game_scores(raw_df)
    team_games = _unpivot_team_runs(game_scores)
    team_games = _apply_rolling_runs(team_games, window)

    logger.info(
        "Team rolling runs computed: %d team-game rows, %d teams",
        len(team_games),
        team_games["team"].nunique(),
    )
    return team_games[["team", "game_pk", "game_date", "team_runs_per_game"]]


def _derive_game_scores(raw_df: pd.DataFrame) -> pd.DataFrame:
    return (
        raw_df.groupby("game_pk")
        .agg(
            home_final=("home_score", "max"),
            away_final=("away_score", "max"),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            game_date=("game_date", "first"),
        )
        .reset_index()
    )


def _unpivot_team_runs(game_scores: pd.DataFrame) -> pd.DataFrame:
    home = game_scores[["game_pk", "game_date", "home_team", "home_final"]].rename(
        columns={"home_team": "team", "home_final": "runs_scored"}
    )
    away = game_scores[["game_pk", "game_date", "away_team", "away_final"]].rename(
        columns={"away_team": "team", "away_final": "runs_scored"}
    )
    return pd.concat([home, away], ignore_index=True)


def _apply_rolling_runs(
    team_games: pd.DataFrame, window: int
) -> pd.DataFrame:
    team_games = team_games.sort_values(["team", "game_date"]).copy()

    team_games["team_runs_per_game"] = team_games.groupby("team")[
        "runs_scored"
    ].transform(
        lambda s: s.rolling(
            window=window, min_periods=TEAM_RUNS_MIN_PERIODS
        ).mean().shift(1)
    )

    return team_games


def add_opportunity_features(
    df: pd.DataFrame,
    batting_order_df: pd.DataFrame,
    team_runs_df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    df = _merge_batting_order(df, batting_order_df)
    df = _merge_team_runs(df, team_runs_df)
    df = _compute_expected_pas(df)

    logger.info(
        "Opportunity features added: batting_order_pos, batting_order_avg, "
        "team_runs_per_game, expected_pas"
    )
    return df


def _merge_batting_order(
    df: pd.DataFrame, batting_order_df: pd.DataFrame
) -> pd.DataFrame:
    order_avg = _compute_batting_order_avg(batting_order_df)

    df = df.merge(
        batting_order_df[["batter", "game_pk", "batting_order_pos"]],
        on=["batter", "game_pk"],
        how="left",
    )
    df = df.merge(
        order_avg[["batter", "game_pk", "batting_order_avg"]],
        on=["batter", "game_pk"],
        how="left",
    )
    return df


def _merge_team_runs(
    df: pd.DataFrame, team_runs_df: pd.DataFrame
) -> pd.DataFrame:
    df["_batting_team"] = np.where(
        df["is_home"] == 1, df["home_team"], df["away_team"]
    )

    df = df.merge(
        team_runs_df[["team", "game_pk", "team_runs_per_game"]],
        left_on=["_batting_team", "game_pk"],
        right_on=["team", "game_pk"],
        how="left",
    )

    df.drop(columns=["_batting_team", "team"], inplace=True, errors="ignore")
    return df


def _resolve_lineup_position(df: pd.DataFrame) -> pd.Series:
    return (
        df["batting_order_avg"]
        .fillna(df["batting_order_pos"])
        .fillna(DEFAULT_LINEUP_POS)
    )


def _compute_expected_pas(df: pd.DataFrame) -> pd.DataFrame:
    lineup_pos = _resolve_lineup_position(df)

    base_pas = BASE_PAS_INTERCEPT + BASE_PAS_SLOPE * (lineup_pos - 1)

    team_factor = (
        df["team_runs_per_game"]
        .fillna(LEAGUE_AVG_RUNS)
        .div(LEAGUE_AVG_RUNS)
        .clip(lower=TEAM_FACTOR_FLOOR, upper=TEAM_FACTOR_CEIL)
    )

    home_adjustment = np.where(df["is_home"] == 1, HOME_PA_ADJUSTMENT, 0.0)

    df["expected_pas"] = base_pas * team_factor + home_adjustment
    return df
