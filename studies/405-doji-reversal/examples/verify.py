"""Real-tape verification — Study 405 (Doji Reversal). Regenerates docs/results.md numbers.

Reads (or, with --fetch, downloads) the basket's daily tapes, runs the doji detector,
the forward-reversal event study vs the unconditional base rate, the label-shuffle
placebo, the detector-cut sweep, the trend/volume myth-check, the SPY-only check, and the
synthetic positive control. Network is touched only with --fetch.

    python studies/405-doji-reversal/examples/verify.py            # cache-only (offline)
    python studies/405-doji-reversal/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from doji_reversal import data, strategy as st  # noqa: E402

CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))
ASOF = "2025-12-31"
HORIZONS = (1, 3, 5, 10)


def main(fetch: bool) -> None:
    panel = data.load_real(cache_dir=CACHE, fetch=fetch)
    panel = {t: b.loc[:ASOF] for t, b in panel.items()}

    if fetch:
        for t in ["SPY", "AAPL", "MSFT", "JPM", "XOM"]:
            b = panel[t]
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} fp={data.fingerprint(b)}")
    print(f"panel fingerprint: {data.panel_fingerprint(panel)}  (as-of {ASOF})\n")

    res = st.run_experiment(panel, body_frac=0.10, horizons=HORIZONS,
                            cost_bps=2.0, borrow_bps=1.0, n_draws=2000, seed=405)
    print(f"n_dojis={res['n_dojis']}  years={res['years']:.1f}  "
          f"per_name_per_yr={res['dojis_per_name_per_yr']:.1f}\n")

    print("=== Doji reversal vs base rate (gross) ===")
    for h in HORIZONS:
        d = res["per_horizon"][h]
        print(f"h={h:2d}d  n={d['doji']['n']} win={d['doji']['win_rate']:.3f} "
              f"doji={d['doji']['mean_bps']:+6.2f}bps t={d['doji']['tstat']:+5.2f} "
              f"net={d['doji']['net_bps']:+6.2f} | base={d['base']['mean_bps']:+6.2f} "
              f"delta={d['delta_bps']:+6.2f} placebo_p={d['placebo_p']:.3f}")

    print("\n=== Detector-cut robustness (h=5) ===")
    for bf in (0.05, 0.10, 0.15, 0.20):
        r = st.run_experiment(panel, body_frac=bf, horizons=(5,), n_draws=500, seed=405)
        d = r["per_horizon"][5]
        print(f"body_frac={bf:.2f} n={r['n_dojis']:6d} "
              f"abs_t={d['doji']['tstat']:+5.2f} delta={d['delta_bps']:+6.2f} "
              f"placebo_p={d['placebo_p']:.3f}")

    print("\n=== Myth-check: trend filter (h=5, gross) ===")
    for k, v in st.filtered_stats(panel, horizon=5, mode="trend").items():
        print(f"{k:10s} n={v['n']:6d} mean={v['mean_bps']:+6.2f}bps t={v['tstat']:+5.2f}")
    print("=== Myth-check: volume filter (h=5, gross) ===")
    for k, v in st.filtered_stats(panel, horizon=5, mode="volume").items():
        print(f"{k:10s} n={v['n']:6d} mean={v['mean_bps']:+6.2f}bps t={v['tstat']:+5.2f}")

    print("\n=== SPY-only (h=5, gross) ===")
    ev = st.doji_events(panel["SPY"], body_frac=0.10, horizons=(5,))
    s = st.summarize(ev["rev_5"].to_numpy())
    print(f"SPY n={s['n']} mean={s['mean_bps']:+6.2f}bps t={s['tstat']:+5.2f}")

    print("\n=== Synthetic positive control (h=5) ===")
    for edge in (0.0, 0.01, 0.02, 0.04):
        sp, _ = data.synthetic_panel(n_names=28, n_days=3000, edge=edge, seed=405)
        r = st.run_experiment(sp, body_frac=0.10, horizons=(5,), n_draws=500, seed=405)
        d = r["per_horizon"][5]
        print(f"edge={edge:.2f} delta={d['delta_bps']:+7.2f}bps placebo_p={d['placebo_p']:.3f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
