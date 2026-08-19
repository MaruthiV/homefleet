import pandas as pd, numpy as np
df=pd.read_parquet('rtm_all.parquet')
df['ts']=pd.to_datetime(df['ts'],utc=True).dt.tz_convert('US/Central')
hub=df[df.sp=='HB_HUBAVG'].set_index('ts')['spp'].sort_index()
lz=df[df.sp.str.startswith('LZ_')].pivot_table(index='ts',columns='sp',values='spp').sort_index()
hubs=df[df.sp.str.startswith('HB_')].pivot_table(index='ts',columns='sp',values='spp').sort_index()
def cap(ts):
    return 9000 if ts.year<2022 else 5000
events={
 'Aug 12-16 2019 heat/EEA1':('2019-08-12','2019-08-16'),
 'Sep 2019 (Sep 3-6)':('2019-09-03','2019-09-06'),
 'Hurricane Hanna Jul 25-27 2020':('2020-07-25','2020-07-27'),
 'Hurricane Laura Aug 26-28 2020':('2020-08-26','2020-08-28'),
 'Uri Feb 14-19 2021':('2021-02-14','2021-02-19'),
 'Uri Feb 11-21 2021 (wide)':('2021-02-11','2021-02-21'),
 'Landon Feb 2-5 2022':('2022-02-02','2022-02-05'),
 'July 2022 (Jul 11-13)':('2022-07-11','2022-07-13'),
 'Elliott Dec 22-25 2022':('2022-12-22','2022-12-25'),
 'Jun 2023 (Jun 15-21)':('2023-06-15','2023-06-21'),
 'Aug 17 2023':('2023-08-17','2023-08-17'),
 'Aug 24-25 2023':('2023-08-24','2023-08-25'),
 'Aug 2023 (whole month)':('2023-08-01','2023-08-31'),
 'Sep 6 2023':('2023-09-06','2023-09-06'),
 'Sep 2023 (whole month)':('2023-09-01','2023-09-30'),
 'Heather Jan 14-17 2024':('2024-01-14','2024-01-17'),
 'Derecho May 16-17 2024':('2024-05-16','2024-05-17'),
 'Houston storms May 28 2024':('2024-05-28','2024-05-28'),
 'Beryl Jul 8-10 2024':('2024-07-08','2024-07-10'),
 'Aug 20 2024':('2024-08-20','2024-08-20'),
 'Aug 2024 (whole month)':('2024-08-01','2024-08-31'),
 'Cora Jan 9-10 2025':('2025-01-09','2025-01-10'),
 'Enzo Jan 20-22 2025':('2025-01-20','2025-01-22'),
 'Feb 18-21 2025':('2025-02-18','2025-02-21'),
 'Mar 4 2025 wind':('2025-03-04','2025-03-04'),
 'Apr 7 2025':('2025-04-07','2025-04-07'),
 'Jul 4-5 2025 floods':('2025-07-04','2025-07-05'),
 'Summer 2025 (Jun-Sep)':('2025-06-01','2025-09-30'),
 'Fern Jan 21-29 2026':('2026-01-21','2026-01-29'),
 '2026 Jan-Aug15 (all)':('2026-01-01','2026-08-15'),
 'Summer 2026 (Jun-Aug15)':('2026-06-01','2026-08-15'),
 'Jul 21-22 2026':('2026-07-21','2026-07-22'),
}
rows=[]
for name,(a,b) in events.items():
    s=pd.Timestamp(a,tz='US/Central'); e=pd.Timestamp(b,tz='US/Central')+pd.Timedelta(days=1)
    h=hub[(hub.index>=s)&(hub.index<e)]
    l=lz[(lz.index>=s)&(lz.index<e)]
    hh=hubs[(hubs.index>=s)&(hubs.index<e)]
    c=cap(s)
    imax=h.idxmax()
    lzmax=l.max().max(); lzarg=l.stack().idxmax()
    rows.append(dict(event=name,hubavg_max=round(h.max(),2),hubavg_max_time=str(imax),
        n15_hub_ge1000=int((h>=1000).sum()),hrs_hub_ge1000=round((h>=1000).sum()/4,2),
        n15_hub_ge_cap=int((h>=c-1).sum()),hrs_hub_ge_cap=round((h>=c-1).sum()/4,2),
        n15_anyLZ_ge1000=int((l.max(axis=1)>=1000).sum()),
        n15_anyLZ_ge_cap=int((l.max(axis=1)>=c-1).sum()),
        lz_max=round(lzmax,2),lz_max_where=f"{lzarg[1]} @ {lzarg[0]}",
        LZ_HOUSTON_max=round(l['LZ_HOUSTON'].max(),2),LZ_HOUSTON_min=round(l['LZ_HOUSTON'].min(),2),
        LZ_NORTH_max=round(l['LZ_NORTH'].max(),2),LZ_AEN_max=round(l['LZ_AEN'].max(),2),LZ_CPS_max=round(l['LZ_CPS'].max(),2),
        hub_mean=round(h.mean(),2)))
out=pd.DataFrame(rows)
pd.set_option('display.width',250); pd.set_option('display.max_columns',30); pd.set_option('display.max_colwidth',45)
print(out.to_string())
out.to_csv('event_stats.csv',index=False)
