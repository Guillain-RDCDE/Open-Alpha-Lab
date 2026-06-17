"""Debug why planted alpha isn't detected."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from sin_stocks import data, strategy as st

prices_null, _ = data.synthetic_daily(n_years=15, alpha_bps=0.0, seed=211)
prices_alpha, _ = data.synthetic_daily(n_years=15, alpha_bps=3.0, seed=211)

print("prices_null columns:", prices_null.columns.tolist())
print("prices_alpha columns:", prices_alpha.columns.tolist())

# Check SIN returns directly
sin_null = np.log(prices_null["SIN"] / prices_null["SIN"].shift(1)).dropna()
sin_alpha = np.log(prices_alpha["SIN"] / prices_alpha["SIN"].shift(1)).dropna()
spy_null = np.log(prices_null["SPY"] / prices_null["SPY"].shift(1)).dropna()

print(f"Mean SIN return (null): {sin_null.mean()*1e4:.3f} bps/day")
print(f"Mean SIN return (alpha): {sin_alpha.mean()*1e4:.3f} bps/day")
print(f"Mean SPY return (null): {spy_null.mean()*1e4:.3f} bps/day")

diff_null = sin_null - spy_null
diff_alpha = sin_alpha - spy_null

print(f"Diff null: {diff_null.mean()*1e4:.3f} bps/day")
print(f"Diff alpha: {diff_alpha.mean()*1e4:.3f} bps/day")

s = st.summarize(diff_alpha)
print(f"Diff alpha t-stat: {s['tstat']:.2f}")
