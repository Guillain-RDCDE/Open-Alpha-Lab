"""Real-tape verification — Study 181 (Ultimate-Oscillator). Regenerates docs/results.md numbers.

Fetches (or reads from cache) a basket of daily tapes, runs the Ultimate Oscillator's
oversold (<30) threshold rule and bullish-divergence rule against a random-direction control,
sweeps hold periods, sweeps costs, and records the honest verdict.  Network is touched only
with --fetch.

    python studies/181-ultimate-oscillator/examples/verify.py            # cache-only
    python studies/181-ultimate-oscillator/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ultimate_oscillator import data, strategy as st  # noqa: E402

# Basket: liquid ETFs with long history.
TICKERS = ["SPY", "QQQ", "IWM", "GLD", "TLT"]


def _pool(tickers, fetch: bool, framing: str = "threshold",
          hold: int = 5, cost: float = 0.0, seed: int | None = None):
    frames = []
    for t in tickers:
        bars = data.fetch_daily(t, fetch=fetch)
        uo = st.ultimate_oscillator(bars)
        if framing == "threshold":
            ent = st.threshold_entries(uo)
        else:
            ent = st.divergence_entries(bars, uo)
        if len(ent) == 0:
            continue
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(bars, ent, hold_days=hold, cost_bps=cost, directions=dirs))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main(fetch: bool) -> None:
    if fetch:
        for t in TICKERS:
            b = data.fetch_daily(t, fetch=True)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")
        print()

    print("=== THRESHOLD FRAMING: UO < 30 / UO > 70, hold 5 days (gross) ===")
    sig = _pool(TICKERS, False, "threshold", hold=5, cost=0.0)
    ctrl = _pool(TICKERS, False, "threshold", hold=5, cost=0.0, seed=181)
    if len(sig):
        s = st.summarize(sig, "ret_gross")
        r = st.summarize(ctrl, "ret_gross")
        print(f"SIGNAL n={s['n_trades']} win={s['win_rate']:.3f} "
              f"mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")
        print(f"RANDOM n={r['n_trades']} win={r['win_rate']:.3f} "
              f"mean={r['mean_bps']:+.2f}bps t={r['tstat']:+.2f}")
    else:
        print("  (no entries — cache absent?)")

    print("\n=== HOLD-PERIOD SWEEP (threshold, gross) ===")
    for h in (1, 3, 5, 10, 20):
        s = st.summarize(_pool(TICKERS, False, "threshold", hold=h, cost=0.0), "ret_gross")
        print(f"hold={h:2d}d  n={s['n_trades']:4d} mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print("\n=== COST SWEEP (threshold, hold 5d) ===")
    for c in (0.0, 0.5, 1.0, 2.0, 5.0):
        s = st.summarize(_pool(TICKERS, False, "threshold", hold=5, cost=c), "ret_net")
        print(f"cost={c:4.1f}bps  net mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")

    print("\n=== DIVERGENCE FRAMING: bullish/bearish divergence, hold 5 days (gross) ===")
    dsig = _pool(TICKERS, False, "divergence", hold=5, cost=0.0)
    dctrl = _pool(TICKERS, False, "divergence", hold=5, cost=0.0, seed=181)
    if len(dsig):
        ds = st.summarize(dsig, "ret_gross")
        dr = st.summarize(dctrl, "ret_gross")
        print(f"SIGNAL n={ds['n_trades']} win={ds['win_rate']:.3f} "
              f"mean={ds['mean_bps']:+.2f}bps t={ds['tstat']:+.2f}")
        print(f"RANDOM n={dr['n_trades']} win={dr['win_rate']:.3f} "
              f"mean={dr['mean_bps']:+.2f}bps t={dr['tstat']:+.2f}")
    else:
        print("  (no divergence entries — cache absent?)")

    print("\n=== PER-TICKER BREAKDOWN (threshold, hold 5d, gross) ===")
    for t in TICKERS:
        try:
            bars = data.fetch_daily(t, fetch=False)
        except FileNotFoundError:
            print(f"  {t}: cache absent")
            continue
        uo = st.ultimate_oscillator(bars)
        ent = st.threshold_entries(uo)
        if len(ent) == 0:
            print(f"  {t}: no entries")
            continue
        s = st.summarize(st.run_trades(bars, ent, hold_days=5, cost_bps=0), "ret_gross")
        print(f"  {t:5s}  n={s['n_trades']:3d} win={s['win_rate']:.3f} "
              f"mean={s['mean_bps']:+.2f}bps t={s['tstat']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
