"""Quick check: sin basket without LVS (casino outlier) and sub-period breakdown."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from sin_stocks import data, strategy as st

CACHE = os.path.join(os.path.dirname(__file__), "_cache")
all_prices = data.load_aligned(fetch=False, cache_dir=CACHE)

# Full basket vs no-LVS basket
for basket_name, basket in [
    ("full-6", data.SIN_BASKET),
    ("no-LVS", [t for t in data.SIN_BASKET if t != "LVS"]),
]:
    bp = all_prices[basket]
    ew = st.build_equal_weight(bp)
    spy = np.log(all_prices["SPY"] / all_prices["SPY"].shift(1)).dropna()
    idx = ew.index.intersection(spy.index)
    ew, spy = ew.loc[idx], spy.loc[idx]
    s_sin = st.summarize(ew, label=basket_name)
    diff = ew - spy
    s_diff = st.summarize(diff)
    print(f"{basket_name}: CAGR={s_sin['cagr_pct']:+.2f}% Sharpe={s_sin['sharpe_ann']:+.3f} "
          f"excess={s_diff['cagr_pct']:+.2f}%/yr t={s_diff['tstat']:+.2f}")

# Sub-period breakdown
print()
bp = all_prices[data.SIN_BASKET]
ew = st.build_equal_weight(bp)
spy = np.log(all_prices["SPY"] / all_prices["SPY"].shift(1)).dropna()
idx = ew.index.intersection(spy.index)
ew, spy = ew.loc[idx], spy.loc[idx]

for start, end, label in [
    ("2008-03-18", "2012-12-31", "2008-2012 (GFC era)"),
    ("2013-01-01", "2017-12-31", "2013-2017"),
    ("2018-01-01", "2021-12-31", "2018-2021"),
    ("2022-01-01", "2026-06-12", "2022-2026"),
]:
    sub_ew = ew.loc[start:end]
    sub_spy = spy.loc[start:end]
    s_sin = st.summarize(sub_ew)
    s_spy = st.summarize(sub_spy)
    diff = sub_ew - sub_spy
    s_diff = st.summarize(diff)
    print(f"{label}: SIN={s_sin['cagr_pct']:+.2f}% SPY={s_spy['cagr_pct']:+.2f}% "
          f"excess={s_diff['cagr_pct']:+.2f}% t={s_diff['tstat']:+.2f}")

# Annual excess return
print()
print("Annual excess returns (SIN_EW - SPY):")
for year in range(2009, 2027):
    sub = (ew - spy).loc[f"{year}-01-01":f"{year}-12-31"]
    if len(sub) > 50:
        s = st.summarize(sub)
        print(f"  {year}: {s['cagr_pct']:+.2f}%")
