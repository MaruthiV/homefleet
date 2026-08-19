import sys, csv, collections, datetime as dt

# Usage: python eaglei_tx_peaks.py eaglei_2021_TX.csv 2021-02-12 2021-02-21 [label]
fn, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
label = sys.argv[4] if len(sys.argv) > 4 else ''
start = dt.datetime.fromisoformat(start); end = dt.datetime.fromisoformat(end)
KEY = {'48201':'Harris','48113':'Dallas','48439':'Tarrant','48453':'Travis','48029':'Bexar',
       '48157':'FortBend','48339':'Montgomery','48167':'Galveston','48039':'Brazoria','48085':'Collin','48121':'Denton','48491':'Williamson'}
tot = collections.defaultdict(int)
cty = {k: collections.defaultdict(int) for k in KEY}
with open(fn) as f:
    r = csv.reader(f)
    for row in r:
        if row[0] == 'fips_code':
            continue
        fips, county, state, out, ts = row[:5]
        t = dt.datetime.fromisoformat(ts)
        if t < start or t > end:
            continue
        try:
            o = int(float(out))
        except ValueError:
            continue
        tot[t] += o
        if fips in KEY:
            cty[fips][t] += o
if not tot:
    print(label, 'NO TX ROWS IN WINDOW'); sys.exit()
pk_t = max(tot, key=tot.get)
print(f"{label} window {start.date()}..{end.date()} (UTC): TX statewide peak customers_out = {tot[pk_t]:,} at {pk_t} UTC")
# hours with >100k and >500k out
n100 = sum(1 for t,v in tot.items() if v > 100_000)/4; n500 = sum(1 for t,v in tot.items() if v > 500_000)/4; n1m = sum(1 for t,v in tot.items() if v > 1_000_000)/4
print(f"   hours(>100k)={n100:.1f}  hours(>500k)={n500:.1f}  hours(>1M)={n1m:.1f}   n_timestamps={len(tot)}")
for fips, name in KEY.items():
    if cty[fips]:
        pt = max(cty[fips], key=cty[fips].get)
        print(f"   {name:<11} peak={cty[fips][pt]:>9,} at {pt} UTC")
