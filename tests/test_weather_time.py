"""Protect actual instants, repeated local hours, and complete local-day coverage."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from homefleet.data import weather


@pytest.mark.parametrize('day,hours,utc_hour', [
    ('2024-01-16', 24, 6), ('2024-07-08', 24, 5),
    ('2024-03-10', 23, 6), ('2024-11-03', 25, 5),
])
def test_local_days_preserve_utc_instants_and_dst(monkeypatch, day, hours, utc_hour):
    attempts = []
    def response(url, params, timeout):
        attempts.append(1)
        if len(attempts) == 1:
            raise weather.requests.ReadTimeout('interrupted transfer')
        assert params['timezone'] == 'UTC' and params['timeformat'] == 'unixtime'
        start = pd.Timestamp(params['start_date'], tz='UTC')
        end = pd.Timestamp(params['end_date'], tz='UTC') + pd.Timedelta(days=1)
        index = pd.date_range(start, end, freq='1h', inclusive='left')
        # Values encode their true instant, so relocalizing a clock string fails.
        seconds = index.asi8 // 10**9
        hourly = {name: seconds.tolist() for name in weather.VARS}
        hourly['time'] = seconds.tolist()
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {'hourly': hourly})
    monkeypatch.setattr(weather.requests, 'get', response)
    monkeypatch.setattr(weather.time, 'sleep', lambda seconds: None)
    frame = weather.fetch_hourly(29.8, -95.4, day, day)
    assert len(frame) == hours
    assert frame.index.tz_convert('UTC')[0].hour == utc_hour
    assert set(frame.index.date) == {pd.Timestamp(day).date()}
    np.testing.assert_array_equal(frame.temperature_2m, frame.index.asi8 // 10**9)
    assert frame.index.is_unique
    repeated = frame.index[frame.index.hour == 1]
    if day == '2024-11-03':
        assert len(repeated) == 2 and repeated[0].utcoffset() != repeated[1].utcoffset()
    if day == '2024-03-10':
        assert 2 not in frame.index.hour


def test_weather_cache_rejects_legacy_clock_and_missing_hours(tmp_path):
    frame = pd.DataFrame({'temperature_2m': [0., 1., 2.]},
                         index=pd.date_range('2024-01-01', periods=3, freq='1h', tz='UTC'))
    path = tmp_path / 'weather.parquet'
    frame.to_parquet(path)
    with pytest.raises(ValueError, match='legacy/unknown weather clock'):
        weather.load_hourly(path)
    frame.attrs['homefleet_weather_clock'] = weather.CLOCK_PROTOCOL
    frame = frame.iloc[[0, 2]]
    frame.to_parquet(path)
    with pytest.raises(ValueError, match='continuous UTC-aligned'):
        weather.load_hourly(path)
