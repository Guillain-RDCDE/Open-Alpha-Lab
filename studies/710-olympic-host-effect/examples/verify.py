"""Reproducible headline run for Study 710 — Olympic-Host-Effect.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached host-ETF / ^GSPC tapes
under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp, fingerprint  # noqa: E402

from olympic_host_effect import data, strategy as st  # noqa: E402

print("# Olympic-Host-Effect — does the HOST country's stock market rally around its own Games?")

print(f"\nhost calendar: {len(data.HOSTS)} Summer editions {data.HOSTS[0].year} -> "
      f"{data.HOSTS[-1].year} (hardcoded IOC results archive)")
for h in data.HOSTS:
    tag = h.ticker if h.ticker else "NO TICKER (excluded from real-tape panel)"
    print(f"  {h.year} {h.city:<15} {h.country:<15} {tag}")
    if h.note:
        print(f"      note: {h.note}")

if not data.have_real():
    print("\n(cache miss — fetching host ETFs + ^GSPC once)")
    data.fetch()

real = data.load_real()
for t in data.tickers():
    print(data_stamp(t, real[t].to_frame("Close"), asof=data.AS_OF))

df = st.host_abnormal_returns(real)
print(f"\nreal-tape panel: n = {len(df)} of {len(data.HOSTS)} hosts have a contemporaneous "
      "single-country ETF (Athens 2004 excluded, no ticker existed)")
print(f"benchmark: {data.BENCH_TICKER} (named substitute for URTH/ACWI — see data.py)")

print("\n# THE HEADLINE — host-vs-benchmark abnormal return, [-6mo..+2mo] around the Games")
print(df[["year", "city", "ticker", "entry", "exit", "host_ret_pct", "bench_ret_pct",
          "abn_ret_pct"]].to_string(index=False,
          formatters={"host_ret_pct": "{:+.2f}%".format, "bench_ret_pct": "{:+.2f}%".format,
                      "abn_ret_pct": "{:+.2f}%".format}))

s = st.one_sample_t(df["abn_ret_pct"].values)
print(f"\n  n = {s['n']}  mean abnormal return = {s['mean']:+.2f}%  median = {s['median']:+.2f}%  "
      f"sd = {s['sd']:.2f}%")
print(f"  one-sample t = {s['t']:+.3f}  (df={s['n']-1})  two-sided p = {s['p']:.4f}")

w = st.wilcoxon_test(df["abn_ret_pct"].values)
print(f"  Wilcoxon signed-rank: stat = {w['stat']:.2f}  p = {w['p']:.4f}")

b = st.bootstrap_ci(df["abn_ret_pct"].values)
print(f"  percentile bootstrap 95% CI on the mean: [{b['lo']:+.2f}%, {b['hi']:+.2f}%]  "
      f"({b['n_boot']:,} resamples)")

print("\n# Random-window placebo (20 seeds x 500 draws, matched trading-day window length "
      "per ticker)")
pl = st.placebo_pvalue(real, df)
print(f"  observed mean {pl['obs']:+.2f}% vs placebo mean {pl['placebo_mean']:+.2f}% "
      f"(sd {pl['placebo_sd']:.2f}%) over {pl['n_draws']:,} draws -> two-sided p = "
      f"{pl['p_value']:.4f}")

print("\n# Directional myth-check — does a MAJORITY of hosts actually outperform?")
hr = st.outperform_hit_rate(df)
print(f"  {hr['k']}/{hr['n']} hosts outperformed the benchmark = {hr['rate']*100:.1f}%  "
      f"(Wilson 95% [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")
losers = df.loc[df["abn_ret_pct"] <= 0, "city"].tolist()
winner = df.loc[df["abn_ret_pct"] > 0, "city"].tolist()
print(f"  underperformed: {losers}")
print(f"  outperformed:   {winner}")

print("\n# Sensitivity cuts (transparency only — NOT a way to relabel the headline)")
c1 = st.sensitivity_cut(df, ("Beijing",))
print(f"  excl. Beijing (GFC confounder): n={c1['n']} mean={c1['mean']:+.2f}%  t={c1['t']:+.2f}  "
      f"p={c1['p']:.4f}")
c2 = st.sensitivity_cut(df, ("Beijing", "Rio de Janeiro"))
print(f"  excl. Beijing & Rio (outlier)  : n={c2['n']} mean={c2['mean']:+.2f}%  t={c2['t']:+.2f}  "
      f"p={c2['p']:.4f}  <- a post-hoc 2-of-6 cut; flags exactly the snooping risk it warns "
      "against, not evidence")

print("\n# Synthetic positive control — deterministic, no network")
print("  the one-sample-t detector must NOT fire on a null world (effect=0) and must recover a")
print("  planted effect. Null checked over 20 seeds (never a single stream).")
null_ts = np.array([st.synthetic_detect(0.0, seed=710 + s)["t"] for s in range(20)])
print(f"  null (effect=0), 20 seeds: mean t = {null_ts.mean():+.2f}  (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds")
planted = st.synthetic_detect(60.0, seed=710)
print(f"  planted effect=+60.0 pp (seed 710): mean {planted['mean']:+.2f}%  t = {planted['t']:+.2f}")

print("\n# Power curve — how large a TRUE effect would need to be to clear t>=2 at n=6, "
      "given the real panel's own dispersion (200 seeds/effect)")
pc = st.power_curve((0.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0))
print(pc.to_string(index=False, formatters={"effect_pct": "{:+.0f}pp".format,
                                            "power": "{:.0%}".format}))

fp = fingerprint(df.set_index("entry"), cols=["host_ret_pct", "bench_ret_pct", "abn_ret_pct"])
print(f"\nfingerprint(headline panel) = {fp}")
