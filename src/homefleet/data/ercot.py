import io
import json
import zipfile
import pandas as pd
import requests
from homefleet.paths import DATA_RAW

TZ = "America/Chicago"
MIS_LIST = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS?reportTypeId={rtid}"
MIS_DL = "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId={doc}"
HDR = {"User-Agent": "Mozilla/5.0"}
RTID = {"rtm_spp_hist": 13061, "dam_spp_hist": 13060, "dam_as_mcpc_hist": 13091, "rt_as_mcpc_15min_hist": 25570}


def load_rtm_spp(zones=("LZ_HOUSTON", "LZ_NORTH", "LZ_SOUTH"), path=None):
    # 15-min rt spp wide frame, ct index, usd/mwh
    path = path or DATA_RAW / "ercot" / "rtm_all.parquet"
    df = pd.read_parquet(path)
    df = df[df.sp.isin(zones)].drop_duplicates(["ts", "sp"])
    wide = df.pivot(index="ts", columns="sp", values="spp").sort_index()
    wide.index = pd.to_datetime(wide.index, utc=True).tz_convert(TZ)
    return wide


def mis_docs(rtid):
    r = requests.get(MIS_LIST.format(rtid=rtid), headers=HDR, timeout=60)
    r.raise_for_status()
    return [d["Document"] for d in r.json()["ListDocsByRptTypeRes"]["DocumentList"]]


def mis_download(doc_id):
    r = requests.get(MIS_DL.format(doc=doc_id), headers=HDR, timeout=300)
    r.raise_for_status()
    return r.content


def fetch_yearly_csv(rtid, friendly_name):
    # e.g. rtid 13091, 'DAMASMCPC_2023' -> dataframe
    doc = next(d for d in mis_docs(rtid) if d["FriendlyName"] == friendly_name)
    z = zipfile.ZipFile(io.BytesIO(mis_download(doc["DocID"])))
    name = z.namelist()[0]
    if name.endswith(".csv"):
        return pd.read_csv(z.open(name))
    return pd.read_excel(z.open(name), sheet_name=None)


def dam_as_mcpc(year):
    # hourly dam mcpc by service, usd/mwh; ecrs blank before jun 2023, 'REGUP ' has a trailing space upstream
    df = fetch_yearly_csv(RTID["dam_as_mcpc_hist"], f"DAMASMCPC_{year}")
    df.columns = [c.strip() for c in df.columns]
    return df


def _fetch_zip_xlsx(rtid, friendly_name):
    doc = next(d for d in mis_docs(rtid) if d["FriendlyName"] == friendly_name)
    z = zipfile.ZipFile(io.BytesIO(mis_download(doc["DocID"])))
    return pd.ExcelFile(z.open(z.namelist()[0]))


def _to_ct(date, hour_ending, repeated, interval=None):
    # ercot hour ending 1..24 (+ optional 15-min interval 1..4), repeated hour flag for dst fall-back
    d = pd.to_datetime(date)
    he = pd.to_numeric(hour_ending.astype(str).str.slice(0, 2)) if hour_ending.dtype == object else pd.to_numeric(hour_ending)
    minutes = (he - 1) * 60 + (0 if interval is None else (pd.to_numeric(interval) - 1) * 15)
    naive = d + pd.to_timedelta(minutes, unit="m")
    amb = ~(repeated.astype(str).str.strip().str.upper() == "Y")
    return naive.dt.tz_localize(TZ, ambiguous=amb.to_numpy(), nonexistent="shift_forward")


def dam_spp(year):
    # hourly dam spp for lz/hubs, wide frame, interval-start ct index
    x = _fetch_zip_xlsx(RTID["dam_spp_hist"], f"DAMLZHBSPP_{year}")
    parts = [x.parse(s) for s in x.sheet_names if s != 'Report Info']
    df = pd.concat(parts, ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]
    df["ts"] = _to_ct(df["Delivery Date"], df["Hour Ending"], df["Repeated Hour Flag"])
    return df.pivot_table(index="ts", columns="Settlement Point", values="Settlement Point Price").sort_index()


def rtm_spp(year):
    x = _fetch_zip_xlsx(RTID['rtm_spp_hist'], f'RTMLZHBSPP_{year}')
    df = pd.concat([x.parse(s) for s in x.sheet_names if s != 'Report Info'], ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(subset=['Delivery Date', 'Settlement Point Name'])
    df['ts'] = _to_ct(df['Delivery Date'], df['Delivery Hour'], df['Repeated Hour Flag'], df['Delivery Interval'])
    df = df.rename(columns={'Settlement Point Name': 'sp', 'Settlement Point Price': 'spp'})
    df['spp'] = pd.to_numeric(df['spp'], errors='raise')
    df['ts'] = df['ts'].dt.tz_convert('UTC')
    return df[['ts', 'sp', 'spp']].drop_duplicates(['ts', 'sp']).sort_values(['ts', 'sp'])


def rt_as_mcpc_15min(year):
    # post rtc+b real-time as mcpc by 15-min interval (from 2025-12-05), wide by as type
    x = _fetch_zip_xlsx(RTID["rt_as_mcpc_15min_hist"], f"HIST_15Min_RTM_MCPC_{year}")
    parts = []
    for s in x.sheet_names:
        if s == "Report Info":
            continue
        raw = x.parse(s, header=None)
        hdr_rows = raw.index[raw.iloc[:, 3].astype(str).str.strip() == "AS Type"]
        if len(hdr_rows) == 0:
            continue
        h = hdr_rows[0]
        d = raw.iloc[h + 1:].copy()
        d.columns = [str(c).strip() for c in raw.iloc[h]]
        parts.append(d.dropna(subset=["AS Type"]))
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = _to_ct(df["Delivery Date"], df["Delivery Hour"], df["Repeated Hour Flag"], df["Delivery Interval"])
    df["MCPC"] = pd.to_numeric(df["MCPC"], errors="coerce")
    return df.pivot_table(index="ts", columns="AS Type", values="MCPC").sort_index()
