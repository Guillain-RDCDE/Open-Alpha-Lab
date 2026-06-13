"""Offline positive/negative control for Study 94 (Level-Pegging) — no network.

Shows the harness behaves correctly on panels where we *know* the answer:

- ``mode="broadening"``: the small-cap tail carries the planted premium, so an equal-weight
  index (which over-weights small names vs cap-weight) must WIN — positive, significant alpha.
- ``mode="megacap"``: a few large names carry the premium and the cap-weight index lets them
  compound, so equal-weight must LOSE — negative alpha. This is the 2015-2024 tape in a bottle.

A harness that can't bank EW's edge in a broadening market, nor see it lose in a narrow one,
can't be trusted to read the real RSP-vs-SPY tape.

    python examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from level_pegging import data, strategy  # noqa: E402


def _run(mode: str) -> None:
    frame, truth = data.synthetic_panel(mode=mode, seed=94)
    r = strategy.race(frame)
    ab = strategy.alpha_beta(frame)
    ew, cw = r["ew"], r["cw"]
    print(f"\n=== {mode.upper()} | EW expected to {'WIN' if mode == 'broadening' else 'LOSE'} ===")
    print(f"  EW : Sharpe {ew['sharpe']:.3f}  CAGR {ew['cagr']*100:6.2f}%  final {ew['final']:6.2f}")
    print(f"  CW : Sharpe {cw['sharpe']:.3f}  CAGR {cw['cagr']*100:6.2f}%  final {cw['final']:6.2f}")
    print(f"  EW alpha vs CW = {ab['alpha_ann']*100:+.2f}%/yr (HAC t {ab['t_alpha']:+.2f}), "
          f"beta {ab['beta']:.3f}")
    won = "EW wins" if ew["final"] > cw["final"] else "CW wins"
    print(f"  -> {won} (truth says ew_wins={truth['ew_wins']})")


def main() -> None:
    for mode in ("broadening", "megacap"):
        _run(mode)


if __name__ == "__main__":
    main()
