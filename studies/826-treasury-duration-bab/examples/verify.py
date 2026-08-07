"""Reproducible headline run for Study 826 — Treasury Duration BAB.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached five-ETF panel under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from duration_bab import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")
try:  # keep unicode dashes readable on a cp1252 Windows console
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

print("# Treasury Duration BAB — does betting-against-beta earn a low-risk alpha inside "
      "the curve?")

if not data.have_real():
    print("(cache miss — fetching the five Treasury-ETF closes once)")
    data.fetch()

closes = data.load_series()
print(f"[data] {closes.shape[1]} ETFs, {len(closes)} rows  "
      f"{closes.index.min().date()} -> {closes.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint={data.fingerprint(closes)}")
print("  SURVIVORSHIP: a fixed, currently-listed five-ETF curve ladder — no delisted "
      "Treasury fund is missing here, but the ladder is a design choice named on the "
      "Signal axis.")

panel = data.load_panel()
ret = st.close_returns(panel)
betas = st.rolling_betas(ret, 252).mean()
print("\n[betas] average trailing-252d beta to the equal-weight duration factor:")
print("  " + "  ".join(f"{k} {betas[k]:+.3f}" for k in data.TICKERS)
      + "   (monotone SHY -> TLT, as duration rises)")

book = st.bab_book(ret, 252)
h = st.bab_stats(book, ret)
print(f"\nBAB book: long low-beta (levered to unit beta) / short high-beta, "
      f"{h['n_days']} days")
print("# THE HEADLINE — Frazzini-Pedersen BAB return")
print(f"  BAB   : {h['bab_bps']:+.2f} bps/day  NW(10) t = {h['t_nw']:+.2f}  "
      f"one-sample t = {h['t_1s']:+.2f}  Sharpe = {h['sharpe']:.2f}")
print(f"  legs  : levered-low {h['lev_lo_bps']:+.2f} vs levered-high {h['lev_hi_bps']:+.2f} bps "
      f"(Welch t = {h['welch_t']:+.2f})")
print(f"  factor: alpha {h['alpha_bps']:+.2f} bps/day (NW t = {h['t_alpha']:+.2f}), "
      f"residual beta to factor = {h['beta_resid']:+.3f} (beta-neutral by construction)")
print(f"  cage  : beta_lo {h['beta_lo']:.3f} / beta_hi {h['beta_hi']:.3f} -> "
      f"gross leverage {h['gross_lev']:.2f}x")

print("\n# PLACEBO — column-permute the forward returns into the SAME leverage cage "
      "(1,000 permutations)")
pl = st.placebo_pvalue(ret, n_seeds=20, n_draws_per_seed=50)
sigma = (pl["obs_bps"] - pl["placebo_mean_bps"]) / pl["placebo_sd_bps"]
print(f"  observed {pl['obs_bps']:+.2f} bps vs placebo mean {pl['placebo_mean_bps']:+.3f} "
      f"(sd {pl['placebo_sd_bps']:.3f}) over {pl['n_draws']:,} draws -> right-tail p = "
      f"{pl['p_value']:.5f}  ({sigma:+.2f}sigma vs placebo mean)")
print("  => the real beta-sorted book earns LESS than a random assignment into the same "
      "1/beta cage: the beta SIGNAL adds no value; the positive number is levered carry.")

print("\n# ROBUSTNESS — two eras (split 2018-01-01)")
for lo, hi, lbl in [("2010-01-01", "2018-01-01", "2010-2017"),
                    ("2018-01-01", "2026-07-01", "2018-2026")]:
    sub = book[(book.index >= lo) & (book.index < hi)]
    ts = st.bab_stats(sub, ret)
    print(f"  {lbl}: n={ts['n_days']}  BAB {ts['bab_bps']:+.2f} bps (NW t={ts['t_nw']:+.2f})")

print("\n# THE TIMER — leveraged BAB, costed")
print("  one-way cost x turnover of the levered weights; short leg pays 50 bps/yr borrow")
for cb in (1.0, 5.0):
    tm = st.timer_stats(book, cost_bps=cb, borrow_bps_yr=50.0)
    print(f"  cost={cb:>4.1f} bps: gross {tm['gross_bps']:+.2f} -> net {tm['net_bps']:+.2f} "
          f"bps/day (cost {tm['cost_bps_per_day']:.2f} + borrow {tm['borrow_bps_per_day']:.2f}/day, "
          f"t = {tm['t_net']:+.2f}, Sharpe {tm['sharpe_net']:.2f}, ~{tm['ann_net_pct']:+.1f}%/yr)")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null_t = []
for s_ in range(20):
    p0 = data.synthetic_panel(edge=0.0, seed=826 + s_, n_days=1300)
    null_t.append(st.synthetic_detect(p0)["t_nw"])
null_t = np.asarray(null_t)
print(f"  null (edge=0), 20 seeds: BAB NW t mean {null_t.mean():+.2f} "
      f"(sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t) >= 2).sum()}/20 seeds")
p1 = data.synthetic_panel(edge=0.0015, seed=826, n_days=1600)
sy = st.synthetic_detect(p1)
print(f"  planted (edge=0.0015, seed 826): BAB NW t = {sy['t_nw']:+.2f}, "
      f"Welch t = {sy['welch_t']:+.2f}, residual beta = {sy['beta_resid']:+.3f}")
