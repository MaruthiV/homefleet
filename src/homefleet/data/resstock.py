import pandas as pd
from homefleet.paths import DATA_RAW

TZ = "America/Chicago"
HTTP_ROOT = "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/resstock_amy2018_release_2"
COUNTY_GISJOIN = {"48201": "G4802010", "48113": "G4801130", "48439": "G4804390", "48029": "G4800290", "48453": "G4804530", "48085": "G4800850", "48121": "G4801210"}
META_COLS = ["in.county", "in.puma", "in.sqft", "in.geometry_building_type_recs", "in.vintage", "in.heating_fuel", "in.hvac_heating_type", "in.hvac_cooling_type", "in.has_pv", "in.occupants", "in.weather_file_city"]


def load_tx_metadata(path=None):
    path = path or DATA_RAW / "resstock" / "TX_baseline_meta.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f'{HTTP_ROOT}/metadata_and_annual_results/by_state/state=TX/parquet/TX_baseline_metadata_and_annual_results.parquet'
        pd.read_parquet(url).to_parquet(path)
    return pd.read_parquet(path)


def sample_building_ids(fips, n, seed=0, meta=None):
    meta = meta if meta is not None else load_tx_metadata()
    m = meta[(meta["in.geometry_building_type_recs"] == "Single-Family Detached") & (meta["in.county"] == COUNTY_GISJOIN[fips])]
    return m.sample(n=min(n, len(m)), random_state=seed).index.to_numpy()


def load_building_load_kw(bldg_id):
    # interval-ending timestamps in est w/o dst -> interval-start ct; kwh/15min -> kw
    url = f"{HTTP_ROOT}/timeseries_individual_buildings/by_state/upgrade=0/state=TX/{bldg_id}-0.parquet"
    ts = pd.read_parquet(url, columns=["timestamp", "out.electricity.total.energy_consumption"])
    t = pd.to_datetime(ts["timestamp"]) - pd.Timedelta(minutes=15)
    t = t.dt.tz_localize("Etc/GMT+5").dt.tz_convert(TZ)
    return pd.Series(ts["out.electricity.total.energy_consumption"].to_numpy() * 4.0, index=t.to_numpy(), name=str(bldg_id))
