import json
import numpy as np
import pandas as pd
import yaml
from homefleet.paths import DATA_RAW, DATA_PROC, CONFIGS
from homefleet.data.ercot import load_rtm_spp
from homefleet.data.eaglei import county_fraction_panel
from homefleet.data.alerts import county_alert_panel, zone_county_map
from homefleet.data.weather import load_hourly, CLOCK_PROTOCOL
from homefleet.sim.profiles import map_profile_to_year, year_index
from homefleet.sim.pv import pv_kw_from_ghi

TZ = "America/Chicago"


def map_load_profiles(loads, year):
    """Apply the existing local-calendar mapping to every building together."""
    source = loads.copy()
    source.index = source.index.strftime('%m-%d %H:%M')
    source = source[~source.index.duplicated(keep='first')]
    index = year_index(year)
    keys = index.strftime('%m-%d %H:%M').str.replace('02-29', '02-28', regex=False)
    result = source.reindex(keys)
    result.index = index
    return result.ffill().bfill()


def load_splits():
    with open(CONFIGS / "splits.yaml") as f:
        return yaml.safe_load(f)


def prices_for_year(year, zone, idx):
    rt = load_rtm_spp(zones=(zone,))[zone]
    rt = rt[~rt.index.duplicated()].reindex(idx).ffill().bfill()
    da = pd.read_parquet(DATA_RAW / "ercot" / f"dam_spp_{year}.parquet")[zone]
    da = da[~da.index.duplicated()].reindex(idx, method="ffill").bfill()
    return rt.to_numpy(dtype=float), da.to_numpy(dtype=float)


def weather_15min(fips, idx):
    w = load_hourly(DATA_RAW / "weather" / f"openmeteo_{fips}.parquet")
    return w.reindex(idx).interpolate(limit=8).ffill().bfill()


def alerts_for_year(year, idx, fips_list, zmap):
    vt = pd.read_parquet(DATA_RAW / "alerts" / f"vtec_tx_{year}.parquet")
    prev = DATA_RAW / "alerts" / f"vtec_tx_{year - 1}.parquet"
    if prev.exists():
        vt = pd.concat([pd.read_parquet(prev), vt])
    return county_alert_panel(vt, idx, fips_list, zmap=zmap)


def build_metro_year(metro, year, n_loads_file=300, pv_share=0.3, seed=0, out=None):
    sp = load_splits()
    if year <= 2025 and not (DATA_RAW / 'eaglei' / f'eaglei_{year}_TX.csv').exists():
        raise FileNotFoundError(f'Missing observed outage record for {year}; refusing to create a zero-outage historical bundle.')
    zone = sp["metros"][metro]["zone"]
    fips_list = sp["metros"][metro]["counties"]
    idx = year_index(year)
    rt, da = prices_for_year(year, zone, idx)
    loads = pd.read_parquet(DATA_RAW / "resstock" / f"loads_{metro}_n{n_loads_file}.parquet")
    meta = pd.read_parquet(DATA_RAW / "resstock" / f"meta_{metro}_n{n_loads_file}.parquet")
    home_fips = meta["fips"].astype(str).to_numpy()
    L = map_load_profiles(loads, year).to_numpy()
    rng = np.random.default_rng(seed)
    has_pv = rng.random(L.shape[1]) < pv_share
    kw_dc = np.where(has_pv, rng.uniform(5.0, 9.0, L.shape[1]), 0.0)
    W = {f: weather_15min(f, idx) for f in fips_list}
    PV = np.zeros_like(L)
    for j in range(L.shape[1]):
        if has_pv[j]:
            w = W[home_fips[j]]
            PV[:, j] = pv_kw_from_ghi(w["shortwave_radiation"].to_numpy(), w["temperature_2m"].to_numpy(), kw_dc[j])
    years_avail = [y for y in (year,) if (DATA_RAW / "eaglei" / f"eaglei_{y}_TX.csv").exists()]
    if years_avail:
        frac = county_fraction_panel([year], fips=fips_list).reindex(idx).fillna(0.0)
    else:
        frac = pd.DataFrame(0.0, index=idx, columns=fips_list)
    zmap = zone_county_map()
    alerts = alerts_for_year(year, idx, fips_list, zmap)
    hz_files = [DATA_PROC / 'causal' / f"hazard_{f}.parquet" for f in fips_list]
    hazard = None
    if all(p.exists() for p in hz_files):
        hz = [pd.read_parquet(p).reindex(idx, method="ffill").bfill().fillna(0.0) for p in hz_files]
        hazard = np.stack([h.to_numpy(dtype=np.float32) for h in hz], axis=1)
        hazard_h = np.array([int(c) for c in hz[0].columns])
    d = {
        "index": idx.asi8, "price_rt": rt, "price_da": da, "load_kw": L.astype(np.float32), "pv_kw": PV.astype(np.float32),
        "home_fips": home_fips, "home_kw_dc": kw_dc.astype(np.float32), "frac": frac[fips_list].to_numpy(dtype=np.float32),
        "alert": alerts[fips_list].to_numpy(dtype=bool),
        "weather": np.stack([W[f].to_numpy(dtype=np.float32) for f in fips_list], axis=1), "fips_list": np.array(fips_list),
        "weather_cols": np.array(list(W[fips_list[0]].columns)), "zone": zone, "year": year, "metro": metro,
        'weather_clock_protocol': np.array(CLOCK_PROTOCOL),
    }
    if hazard is not None:
        d["hazard"] = hazard
        d["hazard_h"] = hazard_h
    out = out or DATA_PROC / f"{metro}_{year}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **d)
    return out
