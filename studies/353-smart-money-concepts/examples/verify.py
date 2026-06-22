"""Reproducible headline run for Study 353 -- prints every number quoted in docs/results.md
and frozen into notebooks/build_notebooks.py (the ``R`` dict). Offline, deterministic.

    python examples/verify.py

Real tape: SPY daily OHLC (yfinance, cached under _cache/spy_daily.csv). The synthetic
random-walk null and all inference are seed-fixed, so every number reproduces exactly.
"""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from smart_money_concepts import data, strategy as st

HF = 20    # FVG horizon (bars)
HO = 10    # OB forward horizon (bars)

df = data.load_real()
o, c = df["open"].to_numpy(), df["close"].to_numpy()
finger = hashlib.sha256(df[["open", "high", "low", "close"]].round(4)
                        .to_csv().encode()).hexdigest()[:12]

print("# Real tape -- SPY daily OHLC (yfinance)")
print(f"bars           : {len(df)}")
print(f"as-of          : {df.index[-1].date()}  (first {df.index[0].date()})")
print(f"SPY range      : ${df['close'].min():.0f} - ${df['close'].max():.0f}")
print(f"daily vol      : {df['close'].pct_change().std():.4f}")
print(f"fingerprint    : sha256[:12]={finger}")

# ---- Claim (a): fair-value-gap fills ------------------------------------------------------
gaps = st.find_fvgs(df, min_rel=0.001)
syn = data.synthetic_ohlc(n=len(df), sigma=df["close"].pct_change().std(), seed=353)
sg = st.find_fvgs(syn, min_rel=0.001)

print(f"\n# (a) Fair-value gaps -- {len(gaps)} on SPY "
      f"(bull {int((gaps['dir']>0).sum())}, bear {int((gaps['dir']<0).sum())}; "
      f"median width {gaps['width_rel'].median()*100:.2f}%)")
print("  horizon | real fill | placebo zone | random-walk synth | z(real-placebo)")
for H in (5, 10, 20, 40):
    r = st.fvg_fill_stats(df, horizon=H, gaps=gaps)
    p = st.placebo_zone_fill(df, gaps, horizon=H, seed=353)
    s = st.fvg_fill_stats(syn, horizon=H, gaps=sg)
    z = st.two_prop_z(int(round(r["fill_rate"] * r["n"])), r["n"],
                      int(round(p["fill_rate"] * p["n"])), p["n"])
    print(f"  H={H:>2}    |  {r['fill_rate']:.3f}    |   {p['fill_rate']:.3f}      "
          f"|     {s['fill_rate']:.3f}       |   {z:+.2f}")
r20 = st.fvg_fill_stats(df, horizon=HF, gaps=gaps)
print(f"  median time-to-fill (H={HF}, real): {r20['median_ttf']:.0f} bar(s)")

# ---- Claim (b): order-block forward returns -----------------------------------------------
obs = st.find_order_blocks(df, impulse=3, impulse_rel=0.015)
ob = st.ob_forward_returns(df, obs, horizon=HO, lag=1)
rnd = st.random_entry_returns(df, n_entries=ob["n"] * 20, horizon=HO, lag=1, seed=353)
w = st.welch_t(ob["rets"], rnd["rets"])
sobs = st.find_order_blocks(syn, impulse=3, impulse_rel=0.015)
sob = st.ob_forward_returns(syn, sobs, horizon=HO, lag=1)

print(f"\n# (b) Order blocks -- {len(obs)} on SPY "
      f"(bull {int((obs['dir']>0).sum())}, bear {int((obs['dir']<0).sum())}); "
      f"forward {HO}-bar return entering on a revisit (1-bar lag)")
print(f"  OB revisit     : n={ob['n']:>5}  mean={ob['mean']*100:+.4f}%  t={ob['t']:+.2f}")
print(f"  random control : n={rnd['n']:>5}  mean={rnd['mean']*100:+.4f}%  t={rnd['t']:+.2f}")
print(f"  Welch OB-random: diff={w['diff']*100:+.4f}%  t={w['t']:+.2f}  p={w['p']:.3f}")
print(f"  synthetic OB   : n={sob['n']:>5}  mean={sob['mean']*100:+.4f}%  t={sob['t']:+.2f}")
print("  net of one-way costs (entry+exit = 2 legs):")
for bps in (1, 5):
    nc = st.net_of_costs(ob["rets"], bps)
    print(f"    {bps}bp/leg : mean={nc['mean']*100:+.4f}%  t={nc['t']:+.2f}")

# ---- Synthetic positive control -- detector fires on pure noise ---------------------------
print(f"\n# Positive control -- a driftless random walk ALSO leaves 'footprints'")
print(f"  synth FVGs found     : {len(sg)}  (fill-rate at H=20: "
      f"{st.fvg_fill_stats(syn, 20, gaps=sg)['fill_rate']:.3f})")
print(f"  synth order blocks   : {len(sobs)}  (forward edge t={sob['t']:+.2f} -- noise)")
print("  => the patterns are geometry of any random walk; the real tape does not beat them.")
