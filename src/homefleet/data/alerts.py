import io
import numpy as np
import pandas as pd
import requests
import json
from pathlib import Path
from homefleet.paths import DATA_RAW

URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/watchwarn.py"
WFO_TX = ["HGX", "FWD", "EWX", "CRP", "BRO", "SJN", "SJT", "LUB", "AMA", "MAF", "EPZ", "SHV"]
# storm-watch style triggers: phenomena -> significances that count
STORM_WATCH = {"HU": ("A", "W"), "TR": ("A", "W"), "SV": ("W",), "TO": ("W",), "WS": ("W",), "BZ": ("W",), "IS": ("W",), "HW": ("W",), "SQ": ("W",), "EC": ("W",), "WC": ("W",), "FF": ("W",), "WW": ("Y",)}
TZ = "America/Chicago"


def fetch_vtec(sts, ets, wfos=WFO_TX):
    # nws watch/warning archive, utc issue/expire, ugc = TXC### county or TXZ### zone
    params = [("accept", "csv"), ("sts", sts), ("ets", ets)] + [("wfo[]", w) for w in wfos]
    r = requests.get(URL, params=params, timeout=600)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), low_memory=False)


def zone_county_map(path=None):
    # nws bp dbx: state|zone|cwa|name|state_zone|county|fips|...
    path = path or DATA_RAW / "misc" / "nws_zone_county_bp16ap26.dbx"
    if not path.exists():
        snapshot = Path(__file__).parents[1] / 'assets/zone_county_map.json'
        return {k: set(v) for k, v in json.loads(snapshot.read_text()).items()}
    rows = [l.split("|") for l in open(path, encoding="latin-1").read().splitlines() if l.startswith("TX|")]
    m = {}
    for r in rows:
        m.setdefault(f"TXZ{int(r[1]):03d}", set()).add(r[6])
    return m


def county_alert_panel(vtec, index, fips, triggers=STORM_WATCH, zmap=None):
    # bool frame (index x fips): any qualifying alert active for the county
    zmap = zmap or zone_county_map()
    d = vtec[vtec.gtype == "C"].copy()
    keep = np.zeros(len(d), dtype=bool)
    for ph, sigs in triggers.items():
        keep |= (d.phenomena == ph) & d.significance.isin(sigs)
    d = d[keep]
    iss = pd.to_datetime(d.utc_issue, utc=True).dt.tz_convert(TZ)
    exp = pd.to_datetime(d.utc_expire, utc=True).dt.tz_convert(TZ)
    out = pd.DataFrame(False, index=index, columns=list(fips))
    for ugc, s, e in zip(d.ugc, iss, exp):
        if pd.isna(s) or pd.isna(e):
            continue
        if ugc.startswith("TXC"):
            targets = ["48" + ugc[3:]]
        else:
            targets = list(zmap.get(ugc, []))
        for f in targets:
            if f in out.columns:
                out.loc[s:e, f] = True
    return out
