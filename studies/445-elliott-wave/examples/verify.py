"""Reproducible headline run for Study 445 — Elliott Wave.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily tapes under ``_cache/`` if
present (the real-tape numbers), and always runs the synthetic positive control with no network.

    python examples/verify.py

The headline is the **pooled** wave-3 test across the broad-index basket (more swings = more
power); SPY is the named single tape. All runs are trimmed to 2025-12-31 (no partial year).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from elliott_wave import data, strategy as st  # noqa: E402

END = "2025-12-31"
NDRAWS = 5_000
SEED = 445


def pooled_signal(tickers, fn, pct=0.05, **kw):
    """Pool one signal's forward returns + directions across the basket, trimmed to END."""
    rets = {h: [] for h in st.HORIZONS}
    dirs = {h: [] for h in st.HORIZONS}
    n_piv = n_ent = 0
    for tk in tickers:
        if not data.have_real(tk):
            continue
        b = data.load_real(tk)
        b = b[b.index <= END]
        piv = st.zigzag(b["close"].to_numpy(float), pct=pct)
        ent = fn(piv, **kw)
        n_piv += len(piv)
        n_ent += len(ent)
        for h in st.HORIZONS:
            led = st.run_trades(b, ent, h, cost_bps=0.0)
            rets[h].append(led["ret_gross"].to_numpy())
            dirs[h].append(led["dir"].to_numpy())
    return rets, dirs, n_piv, n_ent


def stats_for(r, d, rng):
    """Per-horizon stats + label-shuffle placebo p for a pooled (returns, dirs) array."""
    s = st.summarize(pd.DataFrame({"ret_gross": r, "ret_net": r - 2e-4}), "ret_gross")
    sn = st.summarize(pd.DataFrame({"ret_gross": r, "ret_net": r - 2e-4}), "ret_net")
    mags = r * d
    draws = np.array([(rng.choice([-1, 1], size=len(mags)) * mags).mean()
                      for _ in range(NDRAWS)])
    p = float((draws >= r.mean()).mean())
    return s, sn, p


def main():
    tickers = list(data.TICKERS)
    have = [t for t in tickers if data.have_real(t)]
    print("# Elliott Wave — mechanical ZigZag + Fibonacci proxy (irreducibly subjective method)")
    print(f"# basket cached: {have}")

    if have:
        spy = data.load_real("SPY")
        spy = spy[spy.index <= END]
        span = (spy.index.max() - spy.index.min()).days / 365.25
        print(f"\n# SPY span {spy.index.min().date()} -> {spy.index.max().date()} "
              f"({span:.1f}y, {len(spy)} bars), fingerprint {data.fingerprint(spy)}")

        rng = np.random.default_rng(SEED)
        rets, dirs, npv, nen = pooled_signal(tickers, st.wave3_entries)
        print(f"\n# WAVE-3 prediction (buy with wave 1 after a Fib wave-2), pooled basket")
        print(f"# pivots={npv}  entries={nen}")
        print(f"  {'H':>3} {'n':>5} {'gross_bps':>9} {'net_bps':>8} {'win%':>5} "
              f"{'t':>6} {'hac_t':>6} {'placebo_p':>9}")
        for h in st.HORIZONS:
            r = np.concatenate(rets[h])
            d = np.concatenate(dirs[h])
            s, sn, p = stats_for(r, d, rng)
            print(f"  {h:>3} {s['n_trades']:>5} {s['mean_bps']:>9.0f} {sn['mean_bps']:>8.0f} "
                  f"{s['win_rate']*100:>4.0f} {s['t']:>6.2f} {s['hac_t']:>6.2f} {p:>9.3f}")

        print("\n# COMPLETED-IMPULSE prediction (sell the finished five-wave), pooled basket")
        rets2, dirs2, _, nen2 = pooled_signal(tickers, st.completed_impulse_entries)
        print(f"# entries={nen2}")
        for h in st.HORIZONS:
            r = np.concatenate(rets2[h])
            s = st.summarize(pd.DataFrame({"ret_gross": r, "ret_net": r}), "ret_gross")
            print(f"  H={h:>2}: n={s['n_trades']:>4} bps={s['mean_bps']:>6.0f} "
                  f"win={s['win_rate']*100:>3.0f}% t={s['t']:>5.2f} hac={s['hac_t']:>5.2f}")

        print("\n# ROBUSTNESS — Fibonacci wave-2 band (H=20, pooled)")
        for lo, hi, lab in [(0.50, 0.618, "textbook 50-61.8"),
                            (0.382, 0.786, "wide 38.2-78.6"),
                            (0.618, 1.00, "deep 61.8-100")]:
            rb, db, _, _ = pooled_signal(tickers, st.wave3_entries, fib_lo=lo, fib_hi=hi)
            r = np.concatenate(rb[20])
            s = st.summarize(pd.DataFrame({"ret_gross": r, "ret_net": r}), "ret_gross")
            print(f"  {lab:>17}: n={s['n_trades']:>4} bps={s['mean_bps']:>5.0f} "
                  f"t={s['t']:>5.2f}")

        print("\n# ROBUSTNESS — ZigZag reversal threshold (H=20, pooled)")
        for pct in (0.03, 0.05, 0.08):
            rb, db, _, _ = pooled_signal(tickers, st.wave3_entries, pct=pct)
            r = np.concatenate(rb[20])
            s = st.summarize(pd.DataFrame({"ret_gross": r, "ret_net": r}), "ret_gross")
            print(f"  pct={pct}: n={s['n_trades']:>4} bps={s['mean_bps']:>5.0f} t={s['t']:>5.2f}")

        print("\n# SPY-only wave-3 (the named single tape, lower power)")
        for h in st.HORIZONS:
            res = st.run_experiment(spy, horizon=h, cost_bps=1.0, n_draws=NDRAWS, seed=SEED)
            w = res["wave_gross"]
            print(f"  H={h:>2}: ent={res['n_entries']} bps={w['mean_bps']:>5.0f} "
                  f"win={w['win_rate']*100:>3.0f}% t={w['t']:>5.2f} p={res['placebo_p']:.3f}")
    else:
        print("(no cached tapes — run data.load_real(tk, fetch=True) once to build the cache)")

    print("\n# Synthetic positive control — deterministic, no network (H=10)")
    print("#   edge=0 must NOT manufacture significance; a planted wave-3 edge MUST light up.")
    for edge in (0.0, 0.15):
        bars, _ = data.synthetic_panel(edge=edge, seed=SEED, n_days=9000)
        res = st.run_experiment(bars, horizon=10, cost_bps=1.0, n_draws=NDRAWS, seed=SEED)
        w = res["wave_gross"]
        print(f"  planted edge={edge:+.2f}: entries={res['n_entries']:>3}  "
              f"bps={w['mean_bps']:>6.0f}  win={w['win_rate']*100:>3.0f}%  "
              f"t={w['t']:>6.2f}  hac={w['hac_t']:>6.2f}  placebo_p={res['placebo_p']:.3f}")


if __name__ == "__main__":
    main()
