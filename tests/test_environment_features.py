import numpy as np
import pandas as pd
import pytest

from src.features.environment_features import (
    add_environment_features,
    compute_park_hr_factors,
)


def _make_pa_rows(home_team: str, year: int, stand: str, n_pa: int, n_hr: int) -> list[dict]:
    rows = []
    events_pool = ["field_out", "strikeout", "single", "walk"]
    for i in range(n_pa):
        event = "home_run" if i < n_hr else events_pool[i % len(events_pool)]
        rows.append({
            "events": event,
            "game_date": f"{year}-06-15",
            "home_team": home_team,
            "stand": stand,
        })
    return rows


class TestComputeParkHrFactors:

    def test_one_year_lag_factors_from_previous_year(self):
        rows = []
        rows += _make_pa_rows("AAA", 2022, "R", 600, 60)
        rows += _make_pa_rows("BBB", 2022, "R", 600, 15)
        rows += _make_pa_rows("AAA", 2023, "R", 600, 30)
        rows += _make_pa_rows("BBB", 2023, "R", 600, 30)
        raw_df = pd.DataFrame(rows)

        result = compute_park_hr_factors(raw_df)

        factors_2023 = result[result["year"] == 2023]
        assert len(factors_2023) > 0

        aaa_2023 = factors_2023[factors_2023["home_team"] == "AAA"]["park_hr_factor"].iloc[0]
        bbb_2023 = factors_2023[factors_2023["home_team"] == "BBB"]["park_hr_factor"].iloc[0]
        assert aaa_2023 > bbb_2023

        factors_2022 = result[result["year"] == 2022]
        assert len(factors_2022) == 0

    def test_hitter_park_above_one(self):
        rows = []
        rows += _make_pa_rows("HIT", 2022, "R", 600, 60)
        rows += _make_pa_rows("AVG", 2022, "R", 600, 30)
        raw_df = pd.DataFrame(rows)

        result = compute_park_hr_factors(raw_df)

        hit_factor = result[result["home_team"] == "HIT"]["park_hr_factor"].iloc[0]
        assert hit_factor > 1.0

    def test_pitcher_park_below_one(self):
        rows = []
        rows += _make_pa_rows("PIT", 2022, "R", 600, 10)
        rows += _make_pa_rows("AVG", 2022, "R", 600, 50)
        raw_df = pd.DataFrame(rows)

        result = compute_park_hr_factors(raw_df)

        pit_factor = result[result["home_team"] == "PIT"]["park_hr_factor"].iloc[0]
        assert pit_factor < 1.0

    def test_low_sample_defaults_to_one(self):
        rows = []
        rows += _make_pa_rows("SML", 2022, "R", 100, 20)
        rows += _make_pa_rows("BIG", 2022, "R", 600, 30)
        raw_df = pd.DataFrame(rows)

        result = compute_park_hr_factors(raw_df, min_pa_per_park_year=500)

        sml_factor = result[result["home_team"] == "SML"]["park_hr_factor"].iloc[0]
        assert sml_factor == 1.0

    def test_handedness_split_computed(self):
        rows = []
        rows += _make_pa_rows("TST", 2022, "L", 600, 60)
        rows += _make_pa_rows("TST", 2022, "R", 600, 20)
        rows += _make_pa_rows("OTH", 2022, "L", 600, 30)
        rows += _make_pa_rows("OTH", 2022, "R", 600, 30)
        raw_df = pd.DataFrame(rows)

        result = compute_park_hr_factors(raw_df)

        assert "park_hr_factor_handedness" in result.columns

        tst_2023 = result[(result["home_team"] == "TST") & (result["year"] == 2023)]
        left_factor = tst_2023[tst_2023["stand"] == "L"]["park_hr_factor_handedness"].iloc[0]
        right_factor = tst_2023[tst_2023["stand"] == "R"]["park_hr_factor_handedness"].iloc[0]
        assert left_factor != pytest.approx(right_factor, abs=0.01)


class TestComputeWindOutToCf:

    def _make_wind_df(self, home_team: str, wind_dir_deg: float, wind_speed_mph: float) -> pd.DataFrame:
        return pd.DataFrame({
            "home_team": [home_team],
            "wind_dir_deg": [wind_dir_deg],
            "wind_speed_mph": [wind_speed_mph],
        })

    def test_wind_blowing_out_is_positive(self):
        from config.park_config import PARKS
        from src.features.environment_features import _compute_wind_out_to_cf

        cf_bearing = PARKS["NYY"].cf_bearing_deg
        df = self._make_wind_df("NYY", cf_bearing + 180, 10.0)

        result = _compute_wind_out_to_cf(df)

        assert result["wind_out_to_cf"].iloc[0] == pytest.approx(10.0, abs=0.01)

    def test_wind_blowing_in_is_negative(self):
        from config.park_config import PARKS
        from src.features.environment_features import _compute_wind_out_to_cf

        cf_bearing = PARKS["NYY"].cf_bearing_deg
        df = self._make_wind_df("NYY", cf_bearing, 10.0)

        result = _compute_wind_out_to_cf(df)

        assert result["wind_out_to_cf"].iloc[0] == pytest.approx(-10.0, abs=0.01)

    def test_crosswind_is_zero(self):
        from config.park_config import PARKS
        from src.features.environment_features import _compute_wind_out_to_cf

        cf_bearing = PARKS["NYY"].cf_bearing_deg
        df = self._make_wind_df("NYY", cf_bearing + 90, 10.0)

        result = _compute_wind_out_to_cf(df)

        assert result["wind_out_to_cf"].iloc[0] == pytest.approx(0.0, abs=0.01)

    def test_nan_wind_produces_nan(self):
        from src.features.environment_features import _compute_wind_out_to_cf

        df = pd.DataFrame({
            "home_team": ["NYY", "NYY"],
            "wind_dir_deg": [np.nan, 180.0],
            "wind_speed_mph": [10.0, np.nan],
        })

        result = _compute_wind_out_to_cf(df)

        assert pd.isna(result["wind_out_to_cf"].iloc[0])
        assert pd.isna(result["wind_out_to_cf"].iloc[1])


class TestComputeAirDensityIndex:

    def test_sea_level_standard_temp_near_one(self):
        from src.features.environment_features import _compute_air_density_index

        df = pd.DataFrame({"elevation_ft": [0], "temp_f": [70.0]})

        result = _compute_air_density_index(df)

        assert result["air_density_index"].iloc[0] == pytest.approx(1.0, abs=0.03)

    def test_coors_field_below_085(self):
        from src.features.environment_features import _compute_air_density_index

        df = pd.DataFrame({"elevation_ft": [5280], "temp_f": [75.0]})

        result = _compute_air_density_index(df)

        assert result["air_density_index"].iloc[0] < 0.85

    def test_higher_elevation_lower_density(self):
        from src.features.environment_features import _compute_air_density_index

        df = pd.DataFrame({
            "elevation_ft": [0, 5280],
            "temp_f": [70.0, 70.0],
        })

        result = _compute_air_density_index(df)

        assert result["air_density_index"].iloc[0] > result["air_density_index"].iloc[1]

    def test_higher_temp_lower_density(self):
        from src.features.environment_features import _compute_air_density_index

        df = pd.DataFrame({
            "elevation_ft": [0, 0],
            "temp_f": [50.0, 100.0],
        })

        result = _compute_air_density_index(df)

        assert result["air_density_index"].iloc[0] > result["air_density_index"].iloc[1]


class TestAddEnvironmentFeatures:

    @pytest.fixture
    def pa_df(self):
        return pd.DataFrame({
            "game_date": ["2023-06-15", "2023-06-15", "2023-06-16"],
            "home_team": ["NYY", "NYY", "TB"],
            "stand": ["R", "L", "R"],
            "events": ["home_run", "single", "field_out"],
        })

    @pytest.fixture
    def park_factors_df(self):
        return pd.DataFrame({
            "home_team": ["NYY", "NYY", "TB", "TB"],
            "year": [2023, 2023, 2023, 2023],
            "stand": ["R", "L", "R", "L"],
            "park_hr_factor": [1.1, 1.1, 0.9, 0.9],
            "park_hr_factor_handedness": [1.15, 1.05, 0.88, 0.92],
        })

    @pytest.fixture
    def weather_df(self):
        return pd.DataFrame({
            "game_date": ["2023-06-15", "2023-06-16"],
            "home_team": ["NYY", "TB"],
            "temp_c": [25.0, 22.0],
            "humidity_pct": [60.0, 70.0],
            "wind_speed_kmh": [16.0, 10.0],
            "wind_dir_deg": [267.0, 180.0],
        })

    def test_all_ten_columns_present(self, pa_df, park_factors_df, weather_df):
        result = add_environment_features(pa_df, park_factors_df, weather_df)

        expected_cols = [
            "elevation_ft", "roof_type",
            "park_hr_factor", "park_hr_factor_handedness",
            "temp_f", "humidity_pct", "wind_speed_mph", "wind_dir_deg",
            "wind_out_to_cf", "air_density_index",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_missing_weather_produces_nan_weather_columns(self, pa_df, park_factors_df):
        result = add_environment_features(pa_df, park_factors_df, weather_df=None)

        assert result["temp_f"].isna().all()
        assert result["humidity_pct"].isna().all()
        assert result["wind_speed_mph"].isna().all()
        assert result["wind_dir_deg"].isna().all()

    def test_row_count_preserved(self, pa_df, park_factors_df, weather_df):
        result = add_environment_features(pa_df, park_factors_df, weather_df)

        assert len(result) == len(pa_df)

    def test_dome_park_has_nan_wind(self, pa_df, park_factors_df, weather_df):
        result = add_environment_features(pa_df, park_factors_df, weather_df)

        tb_rows = result[result["home_team"] == "TB"]
        assert tb_rows["wind_speed_mph"].isna().all()
        assert tb_rows["wind_dir_deg"].isna().all()
        assert tb_rows["temp_f"].isna().all()
        assert tb_rows["humidity_pct"].isna().all()

    def test_year_column_not_leaked(self, pa_df, park_factors_df, weather_df):
        assert "year" not in pa_df.columns

        result = add_environment_features(pa_df, park_factors_df, weather_df)

        assert "year" not in result.columns


class TestUnitConversions:

    def test_celsius_to_fahrenheit(self):
        from src.features.environment_features import _merge_weather

        df = pd.DataFrame({
            "home_team": ["NYY", "NYY"],
            "game_date": pd.to_datetime(["2023-06-15", "2023-06-16"]),
        })
        weather = pd.DataFrame({
            "home_team": ["NYY", "NYY"],
            "game_date": ["2023-06-15", "2023-06-16"],
            "temp_c": [0.0, 100.0],
            "humidity_pct": [50.0, 50.0],
            "wind_speed_kmh": [0.0, 0.0],
            "wind_dir_deg": [0.0, 0.0],
        })

        result = _merge_weather(df, weather)

        assert result["temp_f"].iloc[0] == pytest.approx(32.0, abs=0.01)
        assert result["temp_f"].iloc[1] == pytest.approx(212.0, abs=0.01)

    def test_kmh_to_mph(self):
        from src.features.environment_features import _merge_weather

        df = pd.DataFrame({
            "home_team": ["NYY"],
            "game_date": pd.to_datetime(["2023-06-15"]),
        })
        weather = pd.DataFrame({
            "home_team": ["NYY"],
            "game_date": ["2023-06-15"],
            "temp_c": [20.0],
            "humidity_pct": [50.0],
            "wind_speed_kmh": [100.0],
            "wind_dir_deg": [0.0],
        })

        result = _merge_weather(df, weather)

        assert result["wind_speed_mph"].iloc[0] == pytest.approx(62.137, abs=0.01)
