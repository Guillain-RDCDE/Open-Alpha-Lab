"""Cache-only verification -- Study 298 (Swiftonomics). Regenerates docs/results.md numbers.

Loads the cached LYV + ^GSPC daily panel (cache-only -- no network), runs the
market-model event study over the hardcoded Eras Tour events, and prints the
headline CARs, the cross-event t-test, the placebo p-value, and the costed sleeve.

The cache is populated once with ``data.fetch_prices(fetch=True)`` (the only
network call in the whole study). After that, run from anywhere:

    python studies/298-swiftonomics/examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from swiftonomics import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 298 -- Swiftonomics -- cache-only verification\n")
    print("Data: LYV + ^GSPC daily adjusted closes (_cache/lyv_gspc.parquet)")
    print("      Eras Tour events: hardcoded in data.py (16 events, 2022-2024)\n")

    if not os.path.exists(data.PRICES_CACHE):
        print("No cache found. Populate it once with:")
        print("    python -c \"from swiftonomics import data; data.fetch_prices(fetch=True)\"")
        print("\n(Network is touched ONLY on that explicit fetch=True call.)")
        return

    prices = data.fetch_prices(fetch=False)
    events = data.eras_events()
    fp = data.fingerprint(prices)

    print(f"Data stamp: {prices.index.min().date()}..{prices.index.max().date()}, "
          f"n={len(prices)} days, fingerprint={fp}\n")

    # Primary spec
    study = st.event_study(prices, events, pre=1, post=3)
    inf = st.car_inference(study)
    print("=== Primary spec: market-model event study, window [-1, +3] ===")
    print(f"  events used:      {inf['n_events']}")
    print(f"  mean CAR:         {inf['mean_car_bps']:+.1f} bps")
    print(f"  cross-event t:    {inf['car_t']:+.3f}  (p = {inf['car_p']:.4f})")
    print(f"  hit-rate:         {inf['hit_rate']:.1%}")
    print(f"  AAR HAC t:        {inf['aar_hac_t']:+.3f}")
    print(f"  median beta:      {np.median(study['betas']):.3f}\n")

    # Window robustness
    print("=== Window robustness ===")
    for pre, post in [(1, 1), (1, 3), (1, 5)]:
        stu = st.event_study(prices, events, pre=pre, post=post)
        i = st.car_inference(stu)
        print(f"  [-{pre},+{post}]: mean CAR {i['mean_car_bps']:+.1f} bps, t {i['car_t']:+.3f}, p {i['car_p']:.4f}")
    print()

    # By kind
    print("=== By event kind ===")
    for kind in ["announce", "milestone"]:
        sub = events[events["kind"] == kind]
        stu = st.event_study(prices, sub, pre=1, post=3)
        i = st.car_inference(stu)
        print(f"  {kind:10s}: n={i['n_events']}, mean CAR {i['mean_car_bps']:+.1f} bps, t {i['car_t']:+.3f}")
    print()

    # Placebo
    plc = st.placebo_distribution(prices, n_events=study["n_events"], pre=1, post=3,
                                  n_draws=5000, seed=298)
    p_plc = st.placebo_pvalue(float(study["car"].mean()), plc["mean_cars"])
    print("=== Placebo (random dates, 5000 draws) ===")
    print(f"  placebo std of mean CAR: {plc['placebo_std_bps']:.1f} bps")
    print(f"  placebo two-sided p:     {p_plc:.4f}\n")

    # Tradable sleeve
    print("=== Tradable sleeve (long LYV around events, 3-day hold, 1-day lag) ===")
    for owb in [0.0, 10.0, 20.0]:
        sl = st.tradable_sleeve(prices, events, hold=3, one_way_bps=owb)
        h = sl["hedged_net"]
        print(f"  one-way {owb:4.0f}bps: hedged net {h['mean_bps']:+.1f} bps/event (t {h['t']:+.3f})")
    print()

    print("Verdict: Signal=NONE, Tradability=MIRAGE -- a cultural phenomenon, not a stock signal.")


if __name__ == "__main__":
    main()
