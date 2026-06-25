"""Reproducible headline run for Study 470 — Stochastic Momentum Index (Blau).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily tapes under ``_cache/`` if
present (the real-tape numbers), and always runs the synthetic positive control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stochastic_momentum_index import data, strategy as st  # noqa: E402

try:
    from scipy import stats  # for the Welch turn-vs-random test
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False

ASOF = "2026-05-31"


def _load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]


print("# Stochastic Momentum Index — mechanical 'turn out of oversold' on 5 ETFs (yfinance daily)")
if data.have_real():
    spy = _load("SPY")
    span = (spy.index[-1] - spy.index[0]).days / 365.25
    print(f"window   : {spy.index[0].date()} -> {spy.index[-1].date()}  ({span:.1f} years)")
    print(f"SPY fp   : {data.fingerprint(spy)}   tickers: {data.DEFAULT_TICKERS}")
    print(f"params   : N={st.SMI_N} s1={st.SMI_S1} s2={st.SMI_S2} oversold={st.OVERSOLD}")

    print("\n# Pooled SMI-turn vs drift-matched random baseline")
    print(f"  {'H':>3} {'n':>5} {'turn_bps':>9} {'win%':>5} {'1samp_t':>8} "
          f"{'rnd_bps':>8} {'delta':>7} {'net_bps':>8} {'welch_t':>8} {'p':>6}")
    total = 0
    for h in st.HORIZONS:
        tt, rr, ne = [], [], []
        for t in data.DEFAULT_TICKERS:
            b = _load(t)
            hi, lo, c = b["high"], b["low"], b["close"]
            e = st.smi_turn_entries(hi, lo, c)
            re = st.random_entries(c, 1000, seed=7)  # large => stable drift-matched baseline
            tt.append(st.forward_returns(c, e, h))
            rr.append(st.forward_returns(c, re, h))
            ne.append(st.forward_returns(c, e, h, cost_bps=1.0))
            if h == st.HORIZONS[0]:
                total += len(e)
        tt = np.concatenate(tt); rr = np.concatenate(rr); ne = np.concatenate(ne)
        s = st.summarize(tt); sr = st.summarize(rr); sn = st.summarize(ne)
        wt, wp = (stats.ttest_ind(tt, rr, equal_var=False) if HAVE_SCIPY else (float("nan"), float("nan")))
        print(f"  {h:>3} {s['n']:>5} {s['mean_bps']:>9.1f} {s['win']*100:>5.0f} {s['t']:>8.2f} "
              f"{sr['mean_bps']:>8.1f} {s['mean_bps']-sr['mean_bps']:>7.1f} {sn['mean_bps']:>8.1f} "
              f"{wt:>8.2f} {wp:>6.3f}")
    print(f"  total SMI turns (H={st.HORIZONS[0]}): {total}")

    # --- SEED-ROBUSTNESS GUARD ------------------------------------------------
    # A single random-baseline seed can throw a lucky Welch t > 2 that is NOT a
    # real edge (cf. Study 452 spinning-top). The Real stamp is only legitimate
    # if the turn-vs-random Welch t is ROBUST to the baseline seed, so we report
    # the MEAN and SPREAD (min/max) of the Welch t over 20 baseline seeds and the
    # fraction that clear the desk's t >= 2 bar. The turn returns are seed-free;
    # only the random baseline re-draws.
    if HAVE_SCIPY:
        print("\n# Seed-robustness of the turn-vs-random Welch t (20 baseline seeds, n=1000/ticker)")
        print(f"  {'H':>3} {'mean_t':>7} {'min_t':>7} {'max_t':>7} {'frac>=2':>8}")
        # cache per-ticker turn returns once (seed-independent)
        turn_by_h = {h: [] for h in st.HORIZONS}
        closes = {}
        for t in data.DEFAULT_TICKERS:
            b = _load(t)
            closes[t] = b["close"]
            e = st.smi_turn_entries(b["high"], b["low"], b["close"])
            for h in st.HORIZONS:
                turn_by_h[h].append(st.forward_returns(b["close"], e, h))
        turn_by_h = {h: np.concatenate(v) for h, v in turn_by_h.items()}
        for h in st.HORIZONS:
            ts = []
            for sd in range(20):
                rr = np.concatenate([
                    st.forward_returns(closes[t], st.random_entries(closes[t], 1000, seed=sd), h)
                    for t in data.DEFAULT_TICKERS])
                ts.append(stats.ttest_ind(turn_by_h[h], rr, equal_var=False)[0])
            ts = np.asarray(ts)
            print(f"  {h:>3} {ts.mean():>7.2f} {ts.min():>7.2f} {ts.max():>7.2f} "
                  f"{(ts >= 2).mean():>8.2f}")
        print("  (Real stamp robust only where frac>=2 ~ 1.0 across seeds: here 10d & 20d.)")

        # --- STRUCTURE PLACEBO ------------------------------------------------
        # Destroy the SMI's defining structure but keep tape, epoch and the exact
        # per-ticker entry count: place the same number of entries on random dates.
        # If the SMI-turn structure (not drift) carries the signal, the real turn
        # must sit far in the right tail. p = share of count-matched random draws
        # whose pooled mean >= the real turn mean.
        print("\n# Structure placebo: count-matched random entries (300 draws)")
        n_ent = {}
        for t in data.DEFAULT_TICKERS:
            b = _load(t)
            n_ent[t] = len(st.smi_turn_entries(b["high"], b["low"], b["close"]))
        for h in (5, 10, 20):
            obs = turn_by_h[h].mean()
            beats = 0
            for d in range(300):
                rng = np.random.default_rng(d)
                pl = [st.forward_returns(closes[t],
                                         st.random_entries(closes[t], n_ent[t], seed=int(rng.integers(0, 2**31))),
                                         h) for t in data.DEFAULT_TICKERS]
                if np.concatenate(pl).mean() >= obs:
                    beats += 1
            print(f"  H={h:>2} obs={obs*1e4:+.1f} bps  placebo p={(beats + 1) / 301:.4f} "
                  f"(beats={beats}/300)")

    print("\n# Per-ticker, H=20: turn one-sample t and turn-minus-random delta")
    for t in data.DEFAULT_TICKERS:
        b = _load(t)
        hi, lo, c = b["high"], b["low"], b["close"]
        e = st.smi_turn_entries(hi, lo, c)
        re = st.random_entries(c, 1000, seed=7)  # large => stable drift-matched baseline
        s = st.summarize(st.forward_returns(c, e, 20))
        sr = st.summarize(st.forward_returns(c, re, 20))
        print(f"  {t}: ent={len(e):>3} turn={s['mean_bps']:+7.1f}bps t={s['t']:+5.2f} "
              f"rnd={sr['mean_bps']:+7.1f}bps delta={s['mean_bps']-sr['mean_bps']:+7.1f}")

    print("\n# Scrambled-parameter placebo (SPY, H=20, 500 draws)")
    b = _load("SPY")
    pl = st.scrambled_param_placebo(b["high"], b["low"], b["close"], 20, n_draws=500, seed=470)
    print(f"  observed {pl['obs']*1e4:+.1f} bps   placebo p={pl['p_value']:.3f}  "
          f"(>0.05 => the specific SMI parameters not load-bearing)")
else:
    print("(no _cache parquets — call data.load_real(t) once per ticker to build the cache)")

print("\n# Synthetic positive control — deterministic, no network (H=20, n_days=4000)")
print("  edge=0 must NOT reach significance; planted bounce must light up.")
for edge in (0.0, 0.6):
    px, _ = data.synthetic_panel(edge=edge, seed=470, n_days=4000)
    hi, lo, c = px["high"], px["low"], px["close"]
    s = st.summarize(st.forward_returns(c, st.smi_turn_entries(hi, lo, c), 20))
    print(f"  edge={edge:.2f}: n={s['n']:>4}  turn={s['mean_bps']:+7.1f}bps  "
          f"win={s['win']*100:.0f}%  t={s['t']:+.2f}")
