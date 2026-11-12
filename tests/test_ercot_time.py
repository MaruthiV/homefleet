import pandas as pd
from homefleet.data.ercot import _to_ct


def test_repeated_delivery_hour_maps_to_distinct_utc_quarter_hours():
    times = _to_ct(pd.Series(['2024-11-03', '2024-11-03']), pd.Series([2, 2]),
                   pd.Series(['N', 'Y']), pd.Series([1, 1]))
    assert times.dt.tz_convert('UTC').tolist() == [pd.Timestamp('2024-11-03 06:00Z'), pd.Timestamp('2024-11-03 07:00Z')]


def test_hour_ending_24_is_last_delivery_hour_not_next_midnight():
    time = _to_ct(pd.Series(['2024-01-10']), pd.Series(['24:00']), pd.Series(['N']), pd.Series([4]))
    assert time.iloc[0] == pd.Timestamp('2024-01-10 23:45', tz='America/Chicago')
