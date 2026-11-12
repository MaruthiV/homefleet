"""Batch preparation must preserve the established single-building replay."""
import numpy as np
import pandas as pd
import pytest
from homefleet.data.assemble import map_load_profiles
from homefleet.sim.profiles import map_profile_to_year, year_index


@pytest.mark.parametrize('year', [2019, 2020, 2024, 2026])
def test_batch_mapping_matches_individual_profiles_across_leap_and_dst(year):
    index = year_index(2018)
    profiles = pd.DataFrame({'a': np.arange(len(index), dtype=float),
                             'b': np.arange(len(index), dtype=float) / 7}, index=index)
    profiles.iloc[10:12, 1] = np.nan
    batch = map_load_profiles(profiles, year)
    assert batch.index.equals(year_index(year))
    for column in profiles:
        np.testing.assert_array_equal(batch[column], map_profile_to_year(profiles[column], year))
