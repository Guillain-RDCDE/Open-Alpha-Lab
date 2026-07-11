"""Reproducible headline run for Study 665 — Titanic Syndrome.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached Dow-30 / ^GSPC / SPY tapes
under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import data_stamp  # noqa: E402

from titanic_syndrome import data, strategy as st  # noqa: E402

print("# Titanic Syndrome — does a 52-week-high-with-collapsing-breadth warning predict a decline?")
print(f"basket: {len(data.DOW30)} current Dow-30 members (survivorship-biased — current "
      "membership only, named)")

if not data.have_real():
    print("(cache miss — fetching Dow-30 / ^GSPC / SPY once)")
    data.fetch()

dow, gspc, spy = data.load_real()
print(data_stamp("Dow-30 adjusted closes", dow, asof=data.AS_OF))
print(data_stamp("^GSPC close", gspc, asof=data.AS_OF))
print(data_stamp("SPY close", spy, asof=data.AS_OF))

df = st.titanic_frame(dow, gspc["Close"], spy["Close"])
print(f"common tape: {len(df):,} sessions {df.index.min().date()} -> {df.index.max().date()}")

raw_signal_days = int(df["titanic"].sum())
entries = st.cluster_entries(df["titanic"])
print(f"raw signal sessions: {raw_signal_days}  ->  {len(entries)} clusters "
      f"(consecutive signal days within {st.CLUSTER_DAYS} calendar days merged)")
print("cluster dates:", ", ".join(d.date().isoformat() for d in entries))

print("\n# THE HEADLINE — forward SPY return: signal vs random-entry (drift-matched) vs unconditional")
fwd = st.run_forward_returns(df, entries)
for h in st.HORIZONS:
    row = fwd["by_h"][h]
    s, r, u = row["signal"], row["random"], row["unconditional"]
    print(f"  {h:>2d}d: signal {s['mean_bps']:+7.1f} bps (n={s['n']:>3d}, t_hac={s['t_hac']:+.2f})  "
          f"| random {r['mean_bps']:+7.1f} bps (n={r['n']})  "
          f"| unconditional {u['mean_bps']:+7.1f} bps (n={u['n']})")
    print(f"       Welch t vs random = {row['welch_t_vs_random']:+.2f}   "
          f"Welch t vs unconditional = {row['welch_t_vs_unconditional']:+.2f}")

print("\n# FALSE-ALARM RATE — cluster followed by a >=5% SPY drawdown within 60 sessions?")
fa = st.false_alarm_stats(df, entries)
print(f"  signal decline rate: {fa['signal_decline_rate']*100:.1f}% ({fa['n_clusters']} clusters)  "
      f"vs base rate {fa['base_decline_rate']*100:.1f}% ({fa['n_base']} random dates)   "
      f"Welch t = {fa['welch_t']:+.2f}")
print(f"  false-alarm rate: {fa['false_alarm_rate']*100:.1f}%")

print("\n# THE TIMER — hold SPY, sit in cash for "
      f"{st.TIMER_EXIT_DAYS}d after each cluster ({st.TIMER_COST_BPS:.0f} bps one-way per transition)")
tp = st.timer_performance(df, entries)
bh, tm = tp["buy_hold"], tp["timer"]
print(f"  buy & hold : CAGR {bh['cagr']*100:+.2f}%  vol {bh['vol']*100:.2f}%  "
      f"Sharpe {bh['sharpe']:.2f}  maxDD {bh['maxdd']*100:.1f}%")
print(f"  timer      : CAGR {tm['cagr']*100:+.2f}%  vol {tm['vol']*100:.2f}%  "
      f"Sharpe {tm['sharpe']:.2f}  maxDD {tm['maxdd']*100:.1f}%   "
      f"({tp['n_days_out']} sessions out, {tp['n_transitions']} transitions)")

print("\n# Random-timer control (same cluster count, same fixed exit window, random dates, "
      "1,000 draws)")
rc = st.random_timer_control(df, len(entries), n_draws=1000)
cagrs = rc["cagrs"]
real_cagr = tm["cagr"]
p_beat = float((cagrs >= real_cagr).mean())
print(f"  random-timer CAGR: mean {cagrs.mean()*100:+.2f}%  sd {cagrs.std(ddof=1)*100:.2f}%  "
      f"over {rc['n_draws']} draws")
print(f"  real timer sits at the {(cagrs < real_cagr).mean()*100:.0f}th percentile "
      f"(one-sided p = {p_beat:.3f} of random timing matching or beating it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the Welch detector must NOT systematically fire on a null (correlated, rho=0.40) "
      "panel and must recover a planted post-signal crash. Null checked over 20 seeds.")
null_ts, null_n = [], []
for s_ in range(20):
    panel, sig = data.synthetic_world(crash_bps=0.0, seed=665 + s_)
    d = st.synthetic_detect(panel, sig)
    null_ts.append(d["welch_t"])
    null_n.append(d["n"])
null_ts = np.asarray(null_ts, dtype=float)
valid = np.sum(~np.isnan(null_ts))
print(f"  null (crash=0), 20 seeds: cluster counts {null_n}")
print(f"  mean Welch t = {np.nanmean(null_ts):+.2f}  (sd {np.nanstd(null_ts, ddof=1):.2f}), "
      f"|t|>=2 in {int((np.abs(null_ts) >= 2).sum())}/{valid} valid seeds")
panel, sig = data.synthetic_world(crash_bps=15.0, seed=665)
sy = st.synthetic_detect(panel, sig)
print(f"  planted crash = 15 bps/day for 30d post-signal (seed 665): n={sy['n']} clusters, "
      f"mean {sy['mean_bps']:+.1f} bps  Welch t = {sy['welch_t']:+.2f}")
