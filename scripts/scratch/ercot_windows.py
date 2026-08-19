import pandas as pd, logging, sys, os, time
logging.disable(logging.CRITICAL)
from gridstatus import Ercot
e = Ercot()
OUT = os.path.dirname(os.path.abspath(__file__))
windows = {
 2019: [('2019-08-12','2019-08-16','Aug 12-16 2019 (EEA1 8/13, 8/15)')],
 2020: [('2020-07-25','2020-07-27','Hurricane Hanna'),('2020-08-26','2020-08-28','Hurricane Laura'),('2020-07-01','2020-08-31','Summer 2020')],
 2021: [('2021-02-13','2021-02-20','Winter Storm Uri'),('2021-06-14','2021-06-15','Jun 14 2021 conservation')],
 2022: [('2022-02-02','2022-02-05','Winter Storm Landon'),('2022-05-13','2022-05-14','May 13 2022 Uri-anniv?'),('2022-07-11','2022-07-21','Jul 2022 heat'),('2022-12-21','2022-12-25','Winter Storm Elliott')],
 2023: [('2023-01-30','2023-02-03','Feb 2023 ice storm (Austin)'),('2023-06-14','2023-06-23','Jun 2023 storms'),('2023-08-16','2023-08-18','Aug 17 2023'),('2023-08-23','2023-08-26','Aug 24-25 2023'),('2023-09-05','2023-09-08','Sep 6 2023 EEA2'),('2023-08-01','2023-09-15','Aug1-Sep15 2023')],
 2024: [('2024-01-13','2024-01-18','Winter Storm Heather'),('2024-05-15','2024-05-18','Houston derecho'),('2024-05-27','2024-05-29','May 28 2024 storms'),('2024-07-07','2024-07-12','Hurricane Beryl'),('2024-08-19','2024-08-21','Aug 20 2024 spike?'),('2024-06-01','2024-09-30','Summer 2024')],
 2025: [('2025-01-19','2025-01-23','Winter Storm Enzo'),('2025-02-17','2025-02-21','Feb 2025 cold'),('2025-03-03','2025-03-05','Mar 4 2025 wind storm'),('2025-05-01','2025-09-30','May-Sep 2025'),('2025-12-05','2025-12-31','Dec 2025 post-RTC+B')],
 2026: [('2026-01-20','2026-01-30','Winter Storm Fern window'),('2026-01-01','2026-02-28','Jan-Feb 2026'),('2026-05-01','2026-08-18','May-Aug 2026')],
}
LOCS = ['LZ_HOUSTON','LZ_NORTH','LZ_SOUTH','LZ_AEN','LZ_CPS','LZ_WEST','HB_HUBAVG']
years = [int(a) for a in sys.argv[1:]] or list(windows)
for y in years:
    pq = f'{OUT}/rtm_spp_{y}.parquet'
    t = time.time()
    if os.path.exists(pq):
        df = pd.read_parquet(pq)
    else:
        try:
            df = e.get_rtm_spp(y)
        except Exception as ex:
            print(y, 'FAILED', repr(ex)[:200]); continue
        df.to_parquet(pq)
    df = df.drop_duplicates(['Interval Start','Location'])  # LZ vs LZEW duplicates
    print(f'\n===== {y}: {len(df)} rows, {df["Interval Start"].min()} .. {df["Interval Start"].max()}  ({time.time()-t:.0f}s)')
    for (s, en, label) in windows[y]:
        print(f'-- {label} [{s}..{en}]')
        for loc in LOCS:
            x = df[df['Location']==loc].set_index('Interval Start').sort_index()
            x = x.loc[s:en, 'SPP'] if len(x) else x
            if len(x)==0: print(f'   {loc}: no data'); continue
            print(f'   {loc:<10} max={x.max():>8.0f} @ {x.idxmax()}  mean={x.mean():>7.1f}  h>=1000={((x>=1000).sum()/4):>6.2f}  h>=4000={((x>=4000).sum()/4):>6.2f}  h>=5000={((x>=5000).sum()/4):>6.2f}  h>=9000={((x>=9000).sum()/4):>6.2f}')
    # whole-year tallies for LZ_HOUSTON and HB_HUBAVG
    for loc in ['LZ_HOUSTON','HB_HUBAVG']:
        x = df[df['Location']==loc].set_index('Interval Start')['SPP']
        print(f'   YEAR {y} {loc}: mean={x.mean():.1f}  h>=1000={(x>=1000).sum()/4:.1f}  h>=4000={(x>=4000).sum()/4:.1f}  max={x.max():.0f} @ {x.idxmax()}')
