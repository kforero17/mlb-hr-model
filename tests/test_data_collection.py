"""
Tests for data collection module.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.data.collect_data import (
    fetch_statcast_season,
    fetch_player_data,
    collect_season,
    combine_all_seasons,
    main,
    _season_dates,
    _already_collected,
)


@pytest.fixture
def sample_statcast_data():
    return pd.DataFrame({
        "game_date": ["2023-06-01", "2023-06-02"],
        "player_name": ["Mike Trout", "Shohei Ohtani"],
        "batter": [545361, 669208],
        "pitcher": [543037, 605483],
        "game_pk": [123456, 123457],
        "pitch_type": ["FF", "SL"],
        "release_speed": [95.5, 88.2],
        "release_pos_x": [-1.2, -1.1],
        "release_pos_z": [5.8, 5.7],
        "plate_x": [0.1, -0.2],
        "plate_z": [2.5, 2.4],
        "launch_speed": [105.3, 98.7],
        "launch_angle": [25.5, 22.3],
        "distance": [420, 395],
        "events": ["home_run", "single"],
        "stand": ["R", "L"],
        "balls": [2, 1],
        "strikes": [1, 2],
        "pitch_number": [5, 3],
        "inning": [3, 2],
        "inning_topbot": ["Top", "Bottom"],
        "outs_when_up": [1, 2],
        "home_team": ["LAA", "LAA"],
        "away_team": ["NYY", "NYY"],
        "home_score": [3, 2],
        "away_score": [2, 1],
        "stadium": ["Angel Stadium", "Angel Stadium"],
    })


def test_season_dates_known_year():
    start, end = _season_dates(2022)
    assert start == "2022-04-07"
    assert end == "2022-10-05"


def test_season_dates_unknown_year():
    start, end = _season_dates(2030)
    assert start == "2030-03-20"
    assert end == "2030-10-05"


def test_already_collected_missing_file():
    assert not _already_collected("/nonexistent/path.parquet")


@patch("src.data.collect_data.statcast")
def test_fetch_statcast_season(mock_statcast, sample_statcast_data):
    mock_statcast.return_value = sample_statcast_data

    df = fetch_statcast_season(2023)

    assert len(df) == 2
    mock_statcast.assert_called_once_with("2023-03-30", "2023-10-01")


@patch("src.data.collect_data.statcast")
def test_fetch_statcast_season_empty(mock_statcast):
    mock_statcast.return_value = pd.DataFrame()

    df = fetch_statcast_season(2023)

    assert df.empty


@patch("src.data.collect_data.statcast_batter")
def test_fetch_player_data_batter(mock_batter, sample_statcast_data):
    mock_batter.return_value = sample_statcast_data

    df = fetch_player_data([545361], 2023, "batter")

    assert len(df) == 2
    mock_batter.assert_called_once()


@patch("src.data.collect_data.statcast_pitcher")
def test_fetch_player_data_pitcher(mock_pitcher, sample_statcast_data):
    mock_pitcher.return_value = sample_statcast_data

    df = fetch_player_data([543037], 2023, "pitcher")

    assert len(df) == 2
    mock_pitcher.assert_called_once()


@patch("src.data.collect_data.statcast_batter")
def test_fetch_player_data_all_fail(mock_batter):
    mock_batter.side_effect = Exception("API down")

    df = fetch_player_data([1, 2], 2023, "batter")

    assert df.empty


@patch("src.data.collect_data._save_raw_and_processed")
@patch("src.data.collect_data.fetch_player_data")
@patch("src.data.collect_data.fetch_statcast_season")
@patch("src.data.collect_data._already_collected", return_value=False)
def test_collect_season(mock_collected, mock_fetch, mock_player, mock_save, sample_statcast_data):
    mock_fetch.return_value = sample_statcast_data
    mock_player.return_value = sample_statcast_data

    collect_season(2023)

    mock_fetch.assert_called_once_with(2023)
    assert mock_player.call_count == 2
    assert mock_save.call_count == 3


@patch("src.data.collect_data._save_raw_and_processed")
@patch("src.data.collect_data.fetch_statcast_season")
@patch("src.data.collect_data._already_collected", return_value=True)
def test_collect_season_skips_existing(mock_collected, mock_fetch, mock_save):
    collect_season(2023)

    mock_fetch.assert_not_called()
    mock_save.assert_not_called()


@patch("src.data.collect_data.combine_all_seasons")
@patch("src.data.collect_data.collect_season")
@patch("src.data.collect_data.ensure_directory")
def test_main(mock_dir, mock_collect, mock_combine):
    main()

    assert mock_dir.call_count == 2
    assert mock_collect.call_count == 4
    mock_combine.assert_called_once()
