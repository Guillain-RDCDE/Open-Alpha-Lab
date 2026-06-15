"""Offline, deterministic demo — Study 168 (Advance-Decline).

No network. Builds synthetic tapes and shows the study's spine in one screen:
the bearish A/D divergence signal is a timing tool only when divergence is actually
planted — on a null (random-walk) tape it adds nothing over the baseline. Run:

    python studies/168-advance-decline/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from advance_decline import data, strategy as st


def _run(label: str, divergence: float, n_days: int = 756,
         lookback: int = 63, horizon: int = 21) -> None:
    panel, spy, truth = data.synthetic_daily(
        n_stocks=200, n_days=n_days, divergence=divergence, seed=168
    )
    ad  = data.build_ad_line(panel)
    res = st.backtest(spy["close"], ad, lookback=lookback, horizon=horizon)

    ds  = res["div_stats"]
    bs  = res["base_stats"]
    n   = res["n_divergences"]
    dif = res["diff_mean_bps"]

    beats = "yes (planted!)" if dif < -2.0 else "no"
    print(
        f"{label:30s} | n_signals={n:3d} | "
        f"post-div {ds['mean_bps']:+7.1f}bps t={ds['tstat']:+5.2f} | "
        f"baseline {bs['mean_bps']:+7.1f}bps | "
        f"diff {dif:+6.1f}bps | worse than baseline? {beats}"
    )


def main() -> None:
    print("Study 168 — Advance-Decline — synthetic positive control")
    print(f"{'label':30s} | {'n_sig':7s} | {'post-div':>25s} | {'baseline':>22s} | {'diff':>10s} | note")
    print("-" * 120)
    _run("divergence=0.0 (null tape)",  divergence=0.0)
    _run("divergence=1.0 (mild)",       divergence=1.0)
    _run("divergence=2.0 (strong)",     divergence=2.0)
    _run("divergence=3.0 (very strong)", divergence=3.0)
    print()
    print("On a null tape the divergence signal carries no forward timing information.")
    print("Only when breadth genuinely leads the index (planted divergence) does the")
    print("post-signal return differ from the unconditional baseline.")
    print("On the real S&P 500 tape: see docs/results.md — verdict is NONE / MIRAGE.")


if __name__ == "__main__":
    main()
