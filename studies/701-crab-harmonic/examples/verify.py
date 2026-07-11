"""Reproducible headline run for Study 701 — Crab-Harmonic.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily basket tapes
under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import fingerprint as fp_content  # noqa: E402

from crab_harmonic import data, strategy as st  # noqa: E402

PCT = 0.02              # zigzag threshold defining a confirmed swing pivot
COST_BPS = 5.0          # headline one-way cost per leg
HORIZONS = (1, 5, 10)
TIMER_HORIZONS = (1, 3, 5, 10, 20)
BASE_SEEDS = 20         # seeds pooled for the random-day base rate

print("# Crab-Harmonic — does a 1.618 XA extension at D predict a reversal?")
print(f"basket: {', '.join(data.BASKET)}  (identical basket to the sibling harmonic studies)")

if not data.have_real():
    print("(cache miss — fetching daily OHLC once)")
    data.fetch()

bars = {t: data.load_real(t) for t in data.BASKET}
for t in data.BASKET:
    b = bars[t]
    print(f"  {t:5s} {b.index[0].date()} -> {b.index[-1].date()}  n={len(b):,}  "
          f"fp={fp_content(b, cols=['close'])}")


def pooled(placebo: bool, cost: float = 0.0, seed: int = 701, horizons=HORIZONS):
    frames, pivn, candn = [], 0, 0
    per_ticker = {}
    for t in data.BASKET:
        piv, cand, ledger = st.detect_and_scan(bars[t], pct=PCT, placebo=placebo, seed=seed,
                                               cost_bps=cost, horizons=horizons)
        pivn += len(piv)
        candn += len(cand)
        per_ticker[t] = ledger
        if not ledger.empty:
            frames.append(ledger)
    ledger = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return ledger, pivn, candn, per_ticker


print("\n# THE HEADLINE — Crab XABCD (AB retrace 0.382-0.618 of XA, BC retrace 0.382-0.886 "
      "of AB, D = X - 1.618*(A-X), the sharpest extension in the harmonic zoo)")
crab_g, n_piv, n_cand_crab, crab_by_ticker = pooled(placebo=False, cost=0.0)
print(f"  pooled confirmed pivots: {n_piv:,}  |  Crab XABCD candidates: {n_cand_crab:,}  "
      f"|  D-touches: {len(crab_g):,}")

for h in HORIZONS:
    sf = st.summarize(crab_g, f"ret_gross_{h}")
    tag = "  <-- headline horizon" if h == 5 else ""
    print(f"  {h:2d}-day: n={sf['n']:4d} hit={sf['hit_rate']*100:5.1f}% "
          f"mean={sf['mean_bps']:+7.2f}bps  HAC t={sf['t']:+.2f}{tag}")

sf5 = st.summarize(crab_g, "ret_gross_5")
print(f"\n  hit rate (5-day): {sf5['hit_rate']*100:.1f}% (Wilson 95% "
      f"[{sf5['hit_lo']*100:.1f}%, {sf5['hit_hi']*100:.1f}%])  n={sf5['n']}")

print("\n# Per-instrument, 5-day gross")
for t in data.BASKET:
    l_crab = crab_by_ticker[t]
    sf = st.summarize(l_crab, "ret_gross_5")
    print(f"  {t:5s} n={sf['n']:3d} mean={sf['mean_bps']:+7.2f}bps  HAC t={sf['t']:+.2f}")

# --------------------------------------------------------------------------- #
# Signal axis — forward returns at D vs a random-day BASE RATE, Bonferroni-corrected
# --------------------------------------------------------------------------- #
print("\n# SIGNAL AXIS — Crab D-touch fade vs a random-day BASE RATE "
      f"(matched bullish/bearish mix, pooled over {BASE_SEEDS} seeds), Bonferroni-corrected")

dir_mix_all = float((crab_g["reversal_dir"] > 0).mean()) if not crab_g.empty else 0.5


def base_rate_pool(ledger_by_ticker: dict, n_seeds: int = BASE_SEEDS, horizons=HORIZONS,
                   cost: float = 0.0):
    frames = []
    for t in data.BASKET:
        l = ledger_by_ticker[t]
        n_ev = len(l)
        if n_ev == 0:
            continue
        mix = float((l["reversal_dir"] > 0).mean())
        for s in range(n_seeds):
            br = st.base_rate_ledger(bars[t], n_ev, mix, horizons=horizons, cost_bps=cost,
                                     seed=st._seed_from(701, t, s))
            frames.append(br)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


base_g = base_rate_pool(crab_by_ticker)
tests = []  # (label, welch_t, n_crab, n_base)
for h in HORIZONS:
    sf = st.summarize(crab_g, f"ret_gross_{h}")
    sb = st.summarize(base_g, f"ret_gross_{h}")
    wt = st.welch_t(crab_g[f"ret_gross_{h}"], base_g[f"ret_gross_{h}"])
    tag = "  <-- headline horizon" if h == 5 else ""
    print(f"  {h:2d}-day: CRAB mean={sf['mean_bps']:+7.2f}bps (n={sf['n']})   |   "
          f"BASE mean={sb['mean_bps']:+7.2f}bps (n={sb['n']:,})   |   Welch t={wt:+.2f}{tag}")
    if h == 5:
        tests.append(("pooled (5-day)", wt))

print("\n  Per-ticker, 5-day, Crab vs its own matched-mix base rate:")
for t in data.BASKET:
    l_crab = crab_by_ticker[t]
    n_ev = len(l_crab)
    if n_ev == 0:
        print(f"  {t:5s} n=0 (no D-touches)")
        continue
    mix = float((l_crab["reversal_dir"] > 0).mean())
    frames = [st.base_rate_ledger(bars[t], n_ev, mix, horizons=HORIZONS, cost_bps=0.0,
                                  seed=st._seed_from(701, t, s)) for s in range(BASE_SEEDS)]
    br_t = pd.concat(frames, ignore_index=True)
    sf = st.summarize(l_crab, "ret_gross_5")
    sb = st.summarize(br_t, "ret_gross_5")
    wt = st.welch_t(l_crab["ret_gross_5"], br_t["ret_gross_5"])
    tests.append((f"{t} (5-day)", wt))
    print(f"  {t:5s} CRAB n={sf['n']:3d} mean={sf['mean_bps']:+7.2f}bps   |   "
          f"BASE mean={sb['mean_bps']:+7.2f}bps   |   Welch t={wt:+.2f}")

n_tests = len(tests)
thr = st.bonferroni_threshold(0.05, n_tests)
print(f"\n  Bonferroni correction across {n_tests} tests (pooled + 6 per-ticker, 5-day): "
      f"critical |t| = {thr:.2f} (family-wise alpha 0.05, normal approx)")
n_survive = sum(1 for _, wt in tests if np.isfinite(wt) and abs(wt) >= thr)
n_uncorrected = sum(1 for _, wt in tests if np.isfinite(wt) and abs(wt) >= 2.0)
print(f"  tests clearing UNCORRECTED |t|>=2: {n_uncorrected}/{n_tests}   |   "
      f"tests surviving Bonferroni: {n_survive}/{n_tests}")
for label, wt in tests:
    flag = "SURVIVES" if (np.isfinite(wt) and abs(wt) >= thr) else "-"
    print(f"    {label:16s} Welch t={wt:+.2f}  {flag}")

# --------------------------------------------------------------------------- #
# Tradability axis — the fade timer, net of costs
# --------------------------------------------------------------------------- #
print("\n# TRADABILITY AXIS — the fade timer, net of costs "
      "(one round trip = 2 x one-way cost x NAV)")
timer_g, _, _, timer_by_ticker = pooled(placebo=False, cost=0.0, horizons=TIMER_HORIZONS)
for h in TIMER_HORIZONS:
    row = []
    for c in (0.0, 5.0, 10.0):
        r = timer_g[f"ret_gross_{h}"].to_numpy(dtype=float) - 2.0 * c / 1e4
        s = {"mean_bps": float(np.nanmean(r) * 1e4), "t": st.hac_mean_t(r)}
        row.append(f"cost={c:4.1f}bps mean={s['mean_bps']:+7.2f}bps t={s['t']:+.2f}")
    print(f"  {h:2d}-day hold: " + "   |   ".join(row))

# --------------------------------------------------------------------------- #
# Third axis (myth-check) — does the precise 1.618 ratio beat a placebo?
# --------------------------------------------------------------------------- #
print("\n# THIRD AXIS (myth-check) — does the precise 1.618 XA extension beat "
      "a placebo extension zone built off the identical pivots?")
plac_g, _, n_cand_plac, plac_by_ticker = pooled(placebo=True, cost=0.0)
print(f"  placebo candidates: {n_cand_plac:,}  |  placebo D-touches: {len(plac_g):,}")
for h in HORIZONS:
    sf = st.summarize(crab_g, f"ret_gross_{h}")
    sp = st.summarize(plac_g, f"ret_gross_{h}")
    wt = st.welch_t(crab_g[f"ret_gross_{h}"], plac_g[f"ret_gross_{h}"])
    tag = "  <-- headline horizon" if h == 5 else ""
    print(f"  {h:2d}-day: CRAB mean={sf['mean_bps']:+7.2f}bps (n={sf['n']})   |   "
          f"PLACEBO mean={sp['mean_bps']:+7.2f}bps (n={sp['n']})   |   Welch t={wt:+.2f}{tag}")

n_beats = 0
print("\n  Per-instrument, 5-day, Crab vs placebo:")
for t in data.BASKET:
    l_crab = crab_by_ticker[t]
    l_plac = plac_by_ticker[t]
    sf = st.summarize(l_crab, "ret_gross_5")
    sp = st.summarize(l_plac, "ret_gross_5")
    beats = np.isfinite(sf["mean_bps"]) and np.isfinite(sp["mean_bps"]) and sf["mean_bps"] > sp["mean_bps"]
    n_beats += int(beats)
    print(f"  {t:5s} CRAB n={sf['n']:3d} mean={sf['mean_bps']:+7.2f}bps   |   "
          f"PLACEBO n={sp['n']:3d} mean={sp['mean_bps']:+7.2f}bps   |   "
          f"beats placebo: {'yes' if beats else 'no'}")
print(f"\n  Crab beats placebo on {n_beats}/{len(data.BASKET)} tickers (5-day)")

# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
print("\n# Synthetic positive control — deterministic, no network")
print("  pooled across a synthetic 6-name basket per seed (mirrors the real pooled sample);")
print("  the detector must NOT fire on the null (mean_rev=0) across 20 seeds, and must")
print("  recover a planted mean-reversion tendency.")


def pooled_synth(mean_rev: float, seed: int, n_names: int = 6) -> pd.DataFrame:
    frames = []
    for i in range(n_names):
        sbars = data.synthetic_world(mean_rev=mean_rev, seed=seed * 1000 + i, n_days=6300)
        _, _, ledger = st.detect_and_scan(sbars, pct=PCT, cost_bps=0.0)
        if not ledger.empty:
            frames.append(ledger)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


null_ts = []
for s_ in range(20):
    ledger = pooled_synth(0.0, 701 + s_)
    null_ts.append(st.summarize(ledger, "ret_gross_5")["t"])
null_ts = np.asarray(null_ts, dtype=float)
print(f"  null (mean_rev=0), 20 seeds: mean HAC t = {np.nanmean(null_ts):+.2f}  "
      f"(sd {np.nanstd(null_ts, ddof=1):.2f}), |t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds")

planted = pooled_synth(0.12, 701)
sp_planted = st.summarize(planted, "ret_gross_5")
print(f"  planted mean_rev=0.12 (seed 701): n={sp_planted['n']}  mean={sp_planted['mean_bps']:+.2f}bps  "
      f"HAC t={sp_planted['t']:+.2f}")
