"""Real-tape verification — Study 407 (Dark Cloud Cover & Piercing Line).

Regenerates every number in docs/results.md. Loads the cached 30-name basket, detects
every Dark Cloud Cover / Piercing Line twin by the precise OHLC rule, enters the next open
(one execution lag), and measures the signed forward 1/3/5/10-day return against the
unconditional base rate, with HAC / one-sample / Welch t's, a label-shuffle placebo, a leg
split, a costs charge, the trend/volume myth-check, and the synthetic positive control.

    python studies/407-dark-cloud-piercing/examples/verify.py            # cache-only
    python studies/407-dark-cloud-piercing/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dark_cloud_piercing import data, strategy as st  # noqa: E402

HORIZONS = (1, 3, 5, 10)
N_DRAWS = 10000


def main(fetch: bool) -> None:
    if fetch:
        for t in data.BASKET:
            b = data.load_one(t, allow_fetch=True)
            print(f"{t:5s} {b.index[0].date()}..{b.index[-1].date()} n={len(b)}")
        print()

    panel = data.load_real(allow_fetch=fetch)
    fp = data.fingerprint(panel)
    spans = [(b.index[0], b.index[-1], len(b)) for b in panel.values()]
    start = min(s[0] for s in spans).date()
    end = max(s[1] for s in spans).date()
    print(f"basket: {len(panel)} names | window {start} -> {end} | fingerprint {fp}")

    print("\n=== Signed forward return after every twin (long Piercing / short Dark Cloud) ===")
    print(f"{'H':>3} {'n':>6} {'bull':>5} {'bear':>5} {'mean%':>8} {'baseL%':>8} "
          f"{'win%':>6} {'HACt':>7} {'1samp':>7} {'plac_p':>7} {'net%':>8}")
    for h in HORIZONS:
        s = st.summarize(panel, h, n_draws=N_DRAWS, cost_bps=5.0)
        print(f"{h:>3} {s['n_events']:>6} {s['n_bull']:>5} {s['n_bear']:>5} "
              f"{s['mean']*100:>8.3f} {s['base_long']*100:>8.3f} {s['win']*100:>6.1f} "
              f"{s['t_hac']:>7.2f} {s['t_one']:>7.2f} {s['p_placebo']:>7.3f} {s['net']*100:>8.3f}")

    print("\n=== Leg split (Piercing bullish vs Dark Cloud bearish) ===")
    print(f"{'H':>3} {'bull_n':>7} {'bull%':>8} {'bull_t':>7} {'bw%':>5} "
          f"{'bear_n':>7} {'bear%':>8} {'bear_t':>7} {'rw%':>5}")
    for h in HORIZONS:
        ls = st.leg_split(panel, h)
        b, r = ls["bull"], ls["bear"]
        print(f"{h:>3} {b['n']:>7} {b['mean']*100:>8.3f} {b['t_hac']:>7.2f} {b['win']*100:>5.1f} "
              f"{r['n']:>7} {r['mean']*100:>8.3f} {r['t_hac']:>7.2f} {r['win']*100:>5.1f}")

    print("\n=== Myth check (H=1): does a prior trend / volume spike rescue it? ===")
    print(f"{'filter':>22} {'n':>6} {'mean%':>8} {'HACt':>7} {'win%':>6}")
    for label, kw in (("plain", {}),
                      ("+trend", {"require_trend": True}),
                      ("+volume spike", {"require_volume": True}),
                      ("+both", {"require_trend": True, "require_volume": True})):
        s = st.summarize(panel, 1, placebo=False, **kw)
        print(f"{label:>22} {s['n_events']:>6} {s['mean']*100:>8.3f} "
              f"{s['t_hac']:>7.2f} {s['win']*100:>6.1f}")

    print("\n=== Synthetic positive control (H=1) ===")
    print(f"{'planted edge':>14} {'events':>7} {'mean%':>8} {'HACt':>7} {'plac_p':>7} {'win%':>6}")
    for edge in (0.000, 0.004):
        syn, _ = data.synthetic_panel(edge=edge, seed=407)
        s = st.summarize(syn, 1, n_draws=N_DRAWS)
        print(f"{edge:>14.3f} {s['n_events']:>7} {s['mean']*100:>8.3f} "
              f"{s['t_hac']:>7.2f} {s['p_placebo']:>7.3f} {s['win']*100:>6.1f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
