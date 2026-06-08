"""Generate the two narrative notebooks for Study 08 (True-Strength) from source.

Like Studies 01–07, the notebooks are a *generated artefact*: edit the cell text here,
rebuild the skeletons, then execute with nbconvert to embed figures/outputs.

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

The executed path runs on the **offline synthetic universe** — trend/cycle names hidden among
random walks — because the cached real parquets are git-ignored and the desk's reproducible
core must run with no network. The synthetic is where the machinery is *validated* (the three
oscillators agree on planted structure, less on noise); the **NONE / MIRAGE / BUSTED real
verdict** is quoted from [`docs/results.md`](../docs/results.md), produced by
`examples/verify_real.py`. Both notebooks follow the SAME seven desk beats (see
../../../METHODOLOGY.md).
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))           # study root (true_strength/ lives there)
sys.path.insert(0, os.path.abspath("../../.."))      # repo root, for quantlab
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (9.5, 5.2)
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
from true_strength import data, oscillators as osc, backtest, collinearity

# Offline synthetic universe: trend/cycle names among random walks. This VALIDATES the
# machinery (oscillators agree where structure is planted). The real verdict (a liquid
# 174-name basket: NONE / MIRAGE / BUSTED) is in ../docs/results.md via verify_real.py.
frames, truth = data.synthetic_universe(seed=0)
structured = {t.ticker for t in truth}
print(f"{len(frames)} names, {len(structured)} with planted momentum structure")
"""


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md("# Study 08 — True-Strength · for the curious\n\n"
           "**Is the True Strength Index a *truer* read on momentum than the MACD or the RSI — "
           "or the same trade repainted?** No maths needed: we'll just put the three indicators "
           "side by side and see whether they ever disagree.\n\n"
           "> Verdict up top (from the real 174-name run in `../docs/results.md`): the TSI is "
           "**84% reconstructable** from the MACD and RSI, takes the **same position as the MACD "
           "99.4%** of the time, and its equity curve correlates **0.994** with it. Three names, "
           "one trade."),
        code(BOOT),

        md("## 1 · The Claim\n\n"
           "The TSI, says its pitch, *double-smooths* momentum to give a cleaner, **truer** "
           "strength reading than older oscillators. Let's look at the TSI, the MACD and the RSI "
           "on one trending name and ask the obvious question: do they ever tell a different story?"),
        code("name = sorted(structured)[0]\n"
             "close = frames[name]['Close']\n"
             "p = osc.oscillator_panel(close)\n"
             "fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)\n"
             "ax[0].plot(close.index, close.values); ax[0].set_title(f'{name} — price')\n"
             "ax[1].plot(p.index, p['z_tsi'], label='TSI (z)')\n"
             "ax[1].plot(p.index, p['z_macd'], label='MACD (z)', alpha=.8)\n"
             "ax[1].plot(p.index, p['z_rsi'], label='RSI (z)', alpha=.8)\n"
             "ax[1].axhline(0, color='k', lw=.6); ax[1].legend(); ax[1].set_title('three oscillators, same scale')\n"
             "plt.tight_layout(); plt.show()"),

        md("## 2 · So What?\n\n"
           "Traders stack oscillators for *confirmation* — the belief that each says something a "
           "little different, so when they agree you have a stronger bet. If the TSI is really a "
           "repaint of the MACD, that confirmation is an illusion: you're counting one signal "
           "three times and feeling braver for it."),

        md("## 3 · How We'd Know\n\n"
           "Four escalating questions: do the three move together (correlation)? does the TSI add "
           "anything beyond the other two (a spanning test)? do they take the same trade (position "
           "agreement)? and is any standalone profit just the reward for being long stocks, not the "
           "oscillator? First, a sanity check that the machinery works where we *know* the answer."),
        code("rec = collinearity.structure_recall(frames, truth)\n"
             "print('oscillators agree on structured names:', round(rec['agree_structured'], 3))\n"
             "print('oscillators agree on noise names:     ', round(rec['agree_noise'], 3))\n"
             "# They agree more where real momentum structure is planted — the machinery is sound."),

        md("## 4 · The Teardown\n\n"
           "On the structured names, how correlated are the three? And how much of the TSI is just "
           "MACD + RSI?"),
        code("struct = {k: v for k, v in frames.items() if k in structured}\n"
             "print(collinearity.level_collinearity(struct).round(3).to_string())\n"
             "inc = collinearity.incremental_information(struct)\n"
             "print('\\nTSI spanned by MACD+RSI — pooled R²:', round(inc['pooled_r2'], 3))"),
        code("# Do the TSI and MACD ever take a different position? Plot where each is long.\n"
             "pos = backtest.positions_by_oscillator(close, rule='zero')\n"
             "agree = (pos['tsi'] == pos['macd']).mean()\n"
             "print(f'TSI and MACD take the same long/flat stance {agree:.1%} of days on {name}')"),

        md("## 5 · The Verdict\n\n"
           "On the real 174-name universe (`../docs/results.md`):\n\n"
           "- **Signal — distinct? `NONE`.** TSI 84% spanned by MACD+RSI; same position as the MACD "
           "99.4% of days; equity-curve correlation 0.994.\n"
           "- **Tradability — a distinct edge? `MIRAGE`.** The crossover's 0.61 Sharpe is the reward "
           "for being long stocks ~half the time; strip that bias and the TSI's timing Sharpe is 0.05.\n"
           "- **'Truer' than MACD/RSI? `BUSTED`.** Three names, one trade."),

        md("## 6 · Could You Trade It?\n\n"
           "You can — and you'll get an ordinary long-biased momentum book that's identical to a "
           "MACD crossover (which your platform computes for free). The TSI adds nothing to trade and "
           "one more knob to tune."),

        md("## 7 · Going Further\n\n"
           "Regress the TSI out of MACD+RSI and trade the *residual* — we'd bet its Sharpe is ≈ 0. "
           "Or run the same test on the Stochastic, CCI, Williams %R: a little atlas of oscillator "
           "redundancy. See the front-page README for the full thread."),
    ]
    return new_notebook(cells=cells)


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md("# Study 08 — True-Strength · for the quants\n\n"
           "Same seven beats, the rigorous layer: spanning-R², sign agreement, equity-curve ρ, the "
           "long/short alpha-vs-beta cut, and a White (2000) Reality Check on the TSI parameter grid. "
           "Executed on the offline synthetic universe; the headline real numbers are quoted from "
           "`../docs/results.md` (as-of + fingerprint)."),
        code(BOOT),

        md("## 1–3 · Claim, stakes, protocol\n\n"
           "H₁: the TSI carries momentum information not already in MACD+RSI. Null: it is spanned by "
           "them, its position and equity curve are collinear, and any standalone Sharpe is the long "
           "bias. Pre-registered mirage line: R² ≳ 0.8, sign agreement ≳ 0.9 vs MACD, equity ρ ≳ 0.95.\n\n"
           "Machinery sanity — the oscillators must agree more on planted structure than on noise:"),
        code("rec = collinearity.structure_recall(frames, truth)\n"
             "print(rec)"),

        md("## 4 · The Teardown\n\n"
           "### Same shape? Level collinearity and the spanning-R²"),
        code("print(collinearity.level_collinearity(frames).round(3).to_string())\n"
             "print('\\nincremental information (TSI on MACD+RSI):', \n"
             "      {k: round(v,3) if isinstance(v,float) else v for k,v in collinearity.incremental_information(frames).items()})"),

        md("### Same position, same equity curve?"),
        code("print('sign agreement (zero-cross):  ', {k: round(v,3) for k,v in collinearity.sign_agreement(frames,'zero').items()})\n"
             "print('sign agreement (signal-cross):', {k: round(v,3) for k,v in collinearity.sign_agreement(frames,'signal').items()})\n"
             "eq = collinearity.equity_collinearity(frames)\n"
             "print('equity-curve correlation + Sharpes:', {k: round(v,3) for k,v in eq.items()})"),

        md("### Alpha vs beta — the long/short timing cut\n\n"
           "The standalone long/flat Sharpe is partly the equity risk premium (long ~half the time). "
           "Symmetrise to long/short and the unconditional drift cancels, leaving only the "
           "oscillator's *timing*. On the real run that collapses TSI 0.61 → 0.05."),
        code("# long/flat vs long/short Sharpe for the TSI crossover on this synthetic universe\n"
             "lf = backtest.equal_weight_net(frames, lambda c: backtest.tsi_position(c, rule='signal', long_short=False))\n"
             "ls = backtest.equal_weight_net(frames, lambda c: backtest.tsi_position(c, rule='signal', long_short=True))\n"
             "print('long/flat  Sharpe:', round(backtest.annualized_sharpe(lf), 3))\n"
             "print('long/short Sharpe:', round(backtest.annualized_sharpe(ls), 3))"),

        md("### Reality Check on the 24-variant TSI grid\n\n"
           "White (2000): the best-of-grid Sharpe against a mean-zero bootstrap null. On the real run "
           "p ≈ 0 — a faint, *real* generic momentum signal, which is why the verdict is **redundancy**, "
           "not noise."),
        code("rc = collinearity.reality_check_grid(frames, n_boot=500)\n"
             "print({k: (round(v,4) if isinstance(v,float) else v) for k,v in rc.items()})"),

        md("### Cost sweep"),
        code("print(collinearity.cost_sweep(frames, rule='signal').round(3).to_string())"),

        md("### The closing argument — trade the TSI's residual over MACD+RSI\n\n"
           "Regress the TSI out of MACD+RSI (full-sample, the generous steelman) and trade the "
           "*residual* — the part the other two can't reproduce. On the real run it earns Sharpe "
           "**−0.56**: the TSI's unique content is anti-signal, while the raw long/short TSI is "
           "already ≈ +0.05. Nothing left to be the 'true' in True Strength."),
        code("print({k: (round(v,4) if isinstance(v,float) else v)\n"
             "       for k,v in collinearity.orthogonalised_tsi_edge(frames).items()})"),

        md("## 5–7 · Verdict, tradability, going further\n\n"
           "**Signal `NONE`** (spanning R² 0.835, sign agreement 0.994 vs MACD, equity ρ 0.994). "
           "**Tradability `MIRAGE`** (long/short timing Sharpe 0.05 — the 0.61 was beta; the "
           "orthogonalised residual trades *negative*, −0.56). **'Truer'? `BUSTED`.** Next: extend to "
           "the wider oscillator zoo (Stochastic, CCI, %R). Numbers + fingerprint in `../docs/results.md`."),
    ]
    return new_notebook(cells=cells)


def main():
    for fname, nb in [("01_for_the_curious.ipynb", build_curious()),
                      ("02_for_the_quants.ipynb", build_quants())]:
        path = os.path.join(HERE, fname)
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
