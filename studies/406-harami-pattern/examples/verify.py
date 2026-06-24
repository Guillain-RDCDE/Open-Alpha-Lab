"""Reproducible headline run for Study 406 — the Harami candlestick pattern.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached basket OHLCV under
``_cache/`` if present (the real-tape numbers) and always runs the synthetic control with
no network.

    python examples/verify.py           # cache-only (offline once warmed)
    python examples/verify.py --fetch   # populate/refresh the basket cache
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from harami_pattern import data, strategy as st  # noqa: E402

NDRAWS = 10_000


def main(fetch: bool) -> None:
    if fetch:
        for t in data.BASKET:
            b = data.fetch_one(t)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} ({len(b)} bars)")
        print()

    print("# Harami pattern — event study on a fixed 30-name (29 large-caps + SPY) basket")
    if data.have_real():
        panel = data.load_real(allow_fetch=False)
        mins = [b.index.min() for b in panel.values()]
        maxs = [b.index.max() for b in panel.values()]
        span = (max(maxs) - min(mins)).days / 365.25
        print(f"names          : {len(panel)}  (fingerprint {data.fingerprint(panel)})")
        print(f"window         : {min(mins).date()} -> {max(maxs).date()}  ({span:.1f} years)")

        print("\n# Signed forward return after every harami (long bullish / short bearish)")
        print("# enter next open, exit close +H; net = 2x5bps round trip + 50bps/yr borrow on shorts")
        print(f"  {'H':>3} {'n':>6} {'bull':>5} {'bear':>5} {'mean%':>7} {'baseL%':>7} "
              f"{'win%':>5} {'t_hac':>6} {'t_one':>6} {'p_plac':>7} {'net%':>7}")
        for h in st.HORIZONS:
            s = st.summarize(panel, h, n_draws=NDRAWS)
            print(f"  {h:>3} {s['n_events']:>6} {s['n_bull']:>5} {s['n_bear']:>5} "
                  f"{s['mean']*100:>7.3f} {s['base_long']*100:>7.3f} {s['win']*100:>5.1f} "
                  f"{s['t_hac']:>6.2f} {s['t_one']:>6.2f} {s['p_placebo']:>7.3f} "
                  f"{s['net']*100:>7.3f}")

        print("\n# Split by leg — bullish (long) vs bearish (short)")
        print(f"  {'H':>3} {'leg':>5} {'n':>5} {'mean%':>7} {'t_hac':>6} {'win%':>5}")
        for h in st.HORIZONS:
            ev = st.collect_events(panel, h)
            bull = ev[ev["dir"] > 0]["signed_ret"].to_numpy()
            bear = ev[ev["dir"] < 0]["signed_ret"].to_numpy()
            print(f"  {h:>3} {'bull':>5} {len(bull):>5} {bull.mean()*100:>7.3f} "
                  f"{st.hac_t(bull):>6.2f} {(bull>0).mean()*100:>5.1f}")
            print(f"  {h:>3} {'bear':>5} {len(bear):>5} {bear.mean()*100:>7.3f} "
                  f"{st.hac_t(bear):>6.2f} {(bear>0).mean()*100:>5.1f}")

        print("\n# Myth check (H=1) — does a prior-trend / volume-contraction filter rescue it?")
        print(f"  {'filter':>22} {'n':>6} {'mean%':>7} {'t_hac':>6} {'win%':>5}")
        for label, rt, rv in [("plain", False, False), ("+prior trend", True, False),
                              ("+volume contraction", False, True), ("+both", True, True)]:
            s = st.summarize(panel, 1, placebo=False, require_trend=rt, require_volume=rv)
            print(f"  {label:>22} {s['n_events']:>6} {s['mean']*100:>7.3f} "
                  f"{s['t_hac']:>6.2f} {s['win']*100:>5.1f}")
    else:
        print("(no _cache parquets — run examples/verify.py --fetch once to populate)")

    print("\n# Synthetic positive control — deterministic, no network")
    print("  detector + inference must recover a PLANTED day-after reversal and must NOT")
    print("  manufacture significance when the true edge is 0.")
    for edge in (0.0, 0.004):
        panel_s, truth = data.synthetic_panel(edge=edge, seed=406)
        s = st.summarize(panel_s, 1, n_draws=3_000)
        print(f"  planted edge={edge:+.4f}: events={s['n_events']:>5}  "
              f"mean={s['mean']*100:>6.3f}%  t_hac={s['t_hac']:>6.2f}  "
              f"p_placebo={s['p_placebo']:.3f}  win={s['win']*100:.0f}%")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
