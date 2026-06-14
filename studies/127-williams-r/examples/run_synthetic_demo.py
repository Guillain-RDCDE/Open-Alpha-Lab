"""Offline, deterministic demo — Study 127 (Williams-R).

No network. Builds a synthetic daily tape and shows the study's spine in one screen:
Williams %R's oversold/overbought zone-entry rule only edges a random-direction coin
when the tape actually carries mean-reversion — and on a martingale (mean_rev = 0)
it is a fair die. Run:

    python studies/127-williams-r/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from williams_r import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 127 — Williams-R — synthetic positive control\n")
    print(f"{'planted mean-rev':>18} | {'%R bps':>9} {'win':>6} {'t':>6} | "
          f"{'random bps':>10} {'t':>6} | %R beats coin?")
    print("-" * 82)
    for mr in (0.00, 0.15, 0.30, -0.15):
        edges = []
        for seed in range(5):
            bars, _ = data.synthetic_daily(n_days=800, mean_rev=mr, seed=seed * 10 + 7)
            wr = st.williams_r(bars)
            ent = st.zone_entries(wr)
            if len(ent) < 5:
                continue
            signal = st.summarize(
                st.run_trades(bars, ent, hold_days=5, cost_bps=0), "ret_gross"
            )
            rand = st.summarize(
                st.run_trades(
                    bars, ent, hold_days=5, cost_bps=0,
                    directions=st.random_directions(len(ent), seed=seed + 1),
                ),
                "ret_gross",
            )
            edges.append((signal["mean_bps"], signal["win_rate"], signal["tstat"],
                          rand["mean_bps"], rand["tstat"]))

        if not edges:
            continue
        sm = sum(e[0] for e in edges) / len(edges)
        sw = sum(e[1] for e in edges) / len(edges)
        st_ = sum(e[2] for e in edges) / len(edges)
        rm = sum(e[3] for e in edges) / len(edges)
        rt_ = sum(e[4] for e in edges) / len(edges)
        beats = "yes" if sm > rm + 1.0 else "no"
        print(f"{mr:>18.2f} | {sm:>+9.2f} {sw:>6.3f} {st_:>+6.2f} | "
              f"{rm:>+10.2f} {rt_:>+6.2f} | {beats}")

    print()
    print("The %R zone entry is a faithful mean-reversion detector: it only edges the coin")
    print("when mean-reversion is planted (mean_rev > 0), and on a martingale (0.00) it is")
    print("a fair die. On the real daily tape mean-reversion is absent — see docs/results.md.")


if __name__ == "__main__":
    main()
