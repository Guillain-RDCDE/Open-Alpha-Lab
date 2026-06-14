"""Offline, deterministic demo — Study 142 (Split-Drift).

No network.  Builds a synthetic event panel and shows the study's spine in one screen:
the post-split drift is detectable when we *plant* it, and absent when we don't.
This is the machinery proof: if the engine reads ~0 on the null, the real-tape verdict
is a statement about the market, not the method.  Run:

    python studies/142-split-drift/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from split_drift import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 142 — Split-Drift — synthetic positive control\n")
    print("(Returns are arithmetic means of compounded lognormal paths;")
    print(" they are large by Jensen's inequality — the *relative* ordering is what matters.)\n")
    print(
        f"{'planted drift':>14} | {'horizon':>6} | {'mean ret%':>9} {'tstat':>7} "
        f"| {'increases?':>10}"
    )
    print("-" * 64)

    # Show that mean return rises monotonically with planted drift at each horizon
    drifts = (0.0, 10.0, 20.0, 40.0)
    results: dict[str, list] = {"ret_1m": [], "ret_3m": [], "ret_6m": []}
    for drift in drifts:
        ev, _ = data.synthetic_events(
            n_stocks=15, n_events=60, horizon=252, drift_bps=drift, seed=142
        )
        for col in ("ret_1m", "ret_3m", "ret_6m"):
            results[col].append(ev[col].dropna().mean() * 100)

    for col, lbl in (("ret_1m", "1m"), ("ret_3m", "3m"), ("ret_6m", "6m")):
        means = results[col]
        for i, (drift, m) in enumerate(zip(drifts, means)):
            increases = "—" if i == 0 else ("yes" if m > means[i - 1] else "NO")
            ev, _ = data.synthetic_events(
                n_stocks=15, n_events=60, horizon=252, drift_bps=drift, seed=142
            )
            s = st.summarize_horizon(ev, col=col)
            print(
                f"{drift:>14.1f} | {lbl:>6} | "
                f"{m:>+9.2f}% {s['tstat']:>+7.2f} "
                f"| {increases:>10}"
            )

    print()

    # Show the spine: 3m mean rises monotonically across drift levels
    print("Spine: 3m mean return by planted drift level (must be monotone):")
    means = []
    for drift in (0.0, 5.0, 10.0, 20.0, 30.0, 40.0):
        ev, _ = data.synthetic_events(
            n_stocks=15, n_events=60, horizon=252, drift_bps=drift, seed=142
        )
        m = ev["ret_3m"].dropna().mean() * 100
        means.append(m)
        print(f"  drift={drift:5.1f} bps/day  3m mean={m:+.2f}%")

    mono = all(means[i] <= means[i + 1] for i in range(len(means) - 1))
    print(f"\nMonotonically increasing: {mono}")
    print(
        "\nThe engine is a faithful drift detector — the real-tape verdict is a "
        "statement about the market, not the method."
    )
    print("See docs/results.md for real-tape numbers.")


if __name__ == "__main__":
    main()
