import time
import pandas as pd
import requests

TZ = "America/Chicago"
URL = "https://archive-api.open-meteo.com/v1/archive"
VARS = ["temperature_2m", "wind_speed_10m", "wind_gusts_10m", "precipitation", "shortwave_radiation", "dew_point_2m", "surface_pressure"]
CLOCK_PROTOCOL = 'utc-unix-v1'


def fetch_hourly(lat, lon, start, end, retries=6):
    begin = pd.Timestamp(start, tz=TZ).tz_convert('UTC')
    finish = (pd.Timestamp(end, tz=TZ) + pd.DateOffset(days=1)).tz_convert('UTC')
    params = {"latitude": lat, "longitude": lon, "start_date": begin.date().isoformat(),
              "end_date": (finish - pd.Timedelta('1ns')).date().isoformat(), "hourly": ",".join(VARS),
              "timezone": 'UTC', 'timeformat': 'unixtime', 'temperature_unit': 'celsius',
              'wind_speed_unit': 'kmh', 'precipitation_unit': 'mm'}
    for k in range(retries):
        try:
            r = requests.get(URL, params=params, timeout=120)
        except requests.RequestException as error:
            if k + 1 == retries:
                raise
            delay = min(60 * 2**k, 300)
            print(f'Weather request {type(error).__name__}; retry {k + 2}/{retries} after {delay}s', flush=True)
            time.sleep(delay)
            continue
        if r.status_code in {429, 500, 502, 503, 504} and k + 1 < retries:
            headers = getattr(r, 'headers', {})
            retry_after = headers.get('Retry-After', '')
            delay = max(int(retry_after) if retry_after.isdigit() else 0, min(60 * 2**k, 300))
            print(f'Weather HTTP {r.status_code}; retry {k + 2}/{retries} after {delay}s', flush=True)
            time.sleep(delay)
            continue
        break
    r.raise_for_status()
    payload = r.json()
    df = pd.DataFrame(payload['hourly'])
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert(TZ)
    df = df.set_index('time')
    df = df[(df.index >= begin) & (df.index < finish)]
    expected = pd.date_range(begin, finish, freq='1h', inclusive='left').tz_convert(TZ)
    if not df.index.equals(expected) or df[VARS].isna().any().any():
        raise ValueError('Weather response has missing, duplicate, or invalid hourly observations')
    df.attrs = {'homefleet_weather_clock': CLOCK_PROTOCOL, 'source_hourly_units': payload.get('hourly_units', {})}
    return df


def load_hourly(path):
    frame = pd.read_parquet(path)
    if frame.attrs.get('homefleet_weather_clock') != CLOCK_PROTOCOL:
        raise ValueError(f'{path}: legacy/unknown weather clock; rerun pull_weather.py --force')
    if (frame.index.tz is None or not frame.index.is_unique or frame.index.hasnans or
            not (frame.index.to_series().diff().iloc[1:] == pd.Timedelta('1h')).all()):
        raise ValueError(f'{path}: weather index must be continuous UTC-aligned hourly time')
    return frame
