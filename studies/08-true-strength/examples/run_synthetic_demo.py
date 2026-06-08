"""Offline demo — the whole gauntlet on the synthetic universe, no network, no cache.

Shows the machinery working where the structure is planted: on trend/cycle names the three
oscillators move together (high collinearity, high sign agreement); on noise names they
don't. This is the sanity check the real-data verdict rests on.

    python examples/run_synthetic_demo.py
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from true_strength import collinearity, data


def main():
    frames, truth = data.synthetic_universe(seed=0)
    structured = {t.ticker for t in truth}
    struct = {k: v for k, v in frames.items() if k in structured}
    noise = {k: v for k, v in frames.items() if k not in structured}

    print(f"{len(frames)} names: {len(struct)} structured (trend/cycle), {len(noise)} noise\n")
    print("[level collinearity — structured names]\n", collinearity.level_collinearity(struct).round(3).to_string())
    print("\n[level collinearity — noise names]\n", collinearity.level_collinearity(noise).round(3).to_string())
    print("\n[sign agreement structured vs noise]", {k: round(v, 3) for k, v in collinearity.structure_recall(frames, truth).items()})
    print("[incremental-information R² (TSI on MACD+RSI)]", {k: round(v, 3) if isinstance(v, float) else v for k, v in collinearity.incremental_information(struct).items()})


if __name__ == "__main__":
    main()
