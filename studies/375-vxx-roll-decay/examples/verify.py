"""Reproducible headline run for Study 375 — VXX-Roll-Decay.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached VIX-ETP tape under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vxx_roll_decay import data, strategy as st

print("# VXX-Roll-Decay — short the VIX-futures ETP (VIXY) for the contango carry")
if data.have_real():
    F = data.load_real()
    yrs = (F.index.max() - F.index.min()).days / 365.25
    print(f"real tape      : VIXY (VXX-equivalent) + VIX term structure, {len(F)} days "
          f"({F.index.min().date()} -> {F.index.max().date()}, {yrs:.1f} years)")
    print(f"VIXY decay     : {F['etp'].iloc[0]:,.0f} -> {F['etp'].iloc[-1]:,.2f} "
          f"(x{F['etp'].iloc[-1]/F['etp'].iloc[0]:.1e}) split-adjusted")

    s = st.summarize(F)
    g, n = s["gross"], s["net"]
    print("\n# Short-carry book (constant-notional short, daily rebalanced)")
    print(f"  daily mean   : {s['daily_mean']*100:+.3f}%   HAC(21) t = {s['hac_t']:.2f}   "
          f"placebo p = {s['placebo_p']:.4f}")
    print(f"  frac contango: {s['frac_contango']*100:.1f}% of days (VIX3M>VIX)")
    print(f"  GROSS        : ann {g['ann_ret']*100:+.1f}%  vol {g['ann_vol']*100:.1f}%  "
          f"Sharpe {g['sharpe']:.2f}  maxDD {g['max_dd']*100:.1f}%")
    print(f"  NET(borrow+c): ann {n['ann_ret']*100:+.1f}%  Sharpe {n['sharpe']:.2f}  "
          f"maxDD {n['max_dd']*100:.1f}%")
    print(f"  TAIL         : skew {g['skew']:+.2f}  excess-kurt {g['exkurt']:.1f}  "
          f"worst day {g['worst_day']*100:.1f}% ({g['worst_day_date'].date()})  "
          f"1% VaR {g['var1']*100:.1f}%  1% CVaR {g['cvar1']*100:.1f}%")

    print("\n# Worst 5 short-carry days (the steamroller)")
    for d, v in st.short_carry(F).nsmallest(5).items():
        print(f"  {d.date()}  {v*100:+.1f}%")

    print("\n# Contango-only variant (short ON only when VIX3M>VIX at prior close)")
    sc = st.summarize(F, contango_only=True)
    cn = sc["net"]
    print(f"  net ann {cn['ann_ret']*100:+.1f}%  Sharpe {cn['sharpe']:.2f}  "
          f"maxDD {cn['max_dd']*100:.1f}%  skew {cn['skew']:+.2f}  HAC t {sc['hac_t']:.2f}")

    print("\n# Borrow sensitivity (net Sharpe vs annual short-borrow fee)")
    for b in (0.00, 0.03, 0.10, 0.30):
        sb = st.summarize(F, borrow_annual=b)
        print(f"  borrow={b*100:>4.0f}%/yr : net ann {sb['net']['ann_ret']*100:+.1f}%  "
              f"Sharpe {sb['net']['sharpe']:.2f}")
else:
    print("(no _cache/vol_prices.csv — run data.fetch_prices() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  with NO carry the engine must NOT find a positive carry; a planted carry must light up,")
print("  while the injected crash always reproduces the negative skew / fat tail.")
for edge in (0.0, 0.0030):
    syn = data.synthetic_etp(edge=edge, seed=375)
    ss = st.summarize(syn)
    gg = ss["gross"]
    print(f"  planted edge={edge:.4f}/day: HAC t={ss['hac_t']:+.2f}  placebo p={ss['placebo_p']:.4f}"
          f"  Sharpe={gg['sharpe']:+.2f}  skew={gg['skew']:+.2f}  maxDD={gg['max_dd']*100:.0f}%")
