"""Generate the two narrative notebooks for Study 11 (Vanishing-Penny) from source.

Like Studies 01–07, the notebooks are a *generated artefact*: edit the cell text here,
rebuild the skeletons, then execute with nbconvert to embed figures/outputs.

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

The executed path runs on the **offline synthetic book** — toy prediction markets whose
arbitrage gaps decay with a *baked-in* 6-minute half-life — because the cached real
parquets are git-ignored and the desk's reproducible core must run with no network. The
synthetic is where the estimator *works* (it recovers a half-life it was given), which is
exactly the point: it proves the code, so the **real verdict** (the penny closes below the
1-minute tape floor — quoted from [`docs/results.md`](../docs/results.md), produced by
`examples/verify_real.py`) is a fact about the market, not a bug. Both notebooks follow the
SAME seven desk beats (see ../../../METHODOLOGY.md).
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))           # study root (prediction_arb/ lives there)
sys.path.insert(0, os.path.abspath("../../.."))      # repo root, for quantlab
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (9.5, 5.2)
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
from prediction_arb import data, arbitrage, robustness

# Offline synthetic book: arbitrage gaps that decay with a KNOWN 6-minute half-life. This
# is where the estimator SHOULD work -- so it validates the machinery. The real verdict
# (the penny closes below the 1-minute tape floor) is in ../docs/results.md via verify_real.py.
gap, truth = data.synthetic_markets(seed=0)
print(f"{gap.shape[1]} markets x {gap.shape[0]} minutes, baked-in half-life = {truth.half_life_min} min")
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
        md(
            "# How fast does a *guaranteed* Polymarket arbitrage close? 🪙\n"
            "### \"\\$40,000,000 in free money\" — tested honestly, in plain English\n\n"
            "Every few weeks a thread goes viral with the same hook: on Polymarket, when "
            "**YES costs \\$0.62 and NO costs \\$0.33**, that's only \\$0.95 — so buying both "
            "guarantees a **free \\$0.05** when the market settles at \\$1. It's true. A "
            "[real 2025 paper](../docs/references.md) documents **\\$40 million** pulled out "
            "this way in a year. The thread says the maths work, the infrastructure exists, "
            "and the only question is whether *you* can grab some.\n\n"
            "So we asked the one question the thread skips: **how long does the free nickel "
            "actually sit there before someone else takes it?**\n\n"
            "> ⚠️ **Not investment advice.** The reproducible core runs on a **synthetic** "
            "market where we *bake in* a known 6-minute half-life (the cached real prices are "
            "git-ignored). That's on purpose: the synthetic is where our stopwatch *works*, "
            "which proves the code — so the real-world result (quoted from "
            "[`../docs/results.md`](../docs/results.md)) is a fact about the market, not a bug.\n\n"
            "*Follows the desk's seven beats ([METHODOLOGY.md](../../../METHODOLOGY.md)). The "
            "rigorous version is the companion,* "
            "[`02_for_the_quants.ipynb`](02_for_the_quants.ipynb)."
        ),
        code(BOOT),

        md(
            "## The answer first 🎯\n\n"
            "| What we asked | The honest answer |\n"
            "|---|---|\n"
            "| Is the free money real? | ✅ **Yes** — risk-free by construction, ~\\$40M "
            "documented, and **161** real gaps opened on our own tape. |\n"
            "| Can our stopwatch measure how fast it closes? | ✅ **Yes** — on the synthetic, it "
            "recovers the baked-in 6-minute half-life exactly. |\n"
            "| How long does a *real* gap last? | ❌ **Under a minute** — 100% of real episodes "
            "are gone by the next minute-sample. |\n"
            "| Could *you* grab it? | ❌ **No** — react in 5 minutes and you keep **3%** of the "
            "nickel; in 30 minutes, basically nothing. |\n\n"
            "> Desk shorthand: **Signal `REAL` · Tradability `MIRAGE` · Execution-moat "
            "`CONFIRMED`** — let's see the method earn them."
        ),

        md(
            "## 1 · The claim 📣\n\n"
            "A viral thread (2.9M views) lays out *\"the exact maths that pulled \\$40,000,000 "
            "out of Polymarket.\"* The core is genuinely true: `YES + NO < \\$1` is a risk-free "
            "trade, it's in a real paper, and the winning wallets are public on-chain — the top "
            "one made **\\$2M over 4,049 trades**. Then the thread pivots to a sign-up link and "
            "an \"airdrop.\" Our job is the gap between *\"this is real\"* and *\"so you can do it.\"*"
        ),
        code(
            "g = gap['MKT00']\n"
            "ep = arbitrage.detect_episodes(g, open_threshold=0.03, name='MKT00')\n"
            "e = max(ep, key=lambda e: abs(e.peak))\n"
            "win = g.loc[e.t_open: e.t_open + pd.Timedelta(minutes=40)]\n"
            "plt.plot(win.index, 100*win.values)\n"
            "plt.axhline(3, ls='--', c='grey'); plt.axhline(0, c='k', lw=0.5)\n"
            "plt.title('One arbitrage opening (synthetic): the free penny, in cents')\n"
            "plt.ylabel('1 - (YES+NO), in cents'); plt.xlabel('time'); plt.show()\n"
            "print(f'peak penny {abs(e.peak)*100:.1f}c, gone in ~{e.duration} min')"
        ),
        md("A gap snaps open, then bleeds back to zero as arbitrageurs close it. The only "
           "question that matters is the **slope of that decay** — its half-life."),

        md(
            "## 2 · So what? 💰\n\n"
            "If the nickel sat there for a minute or two, this would be the cleanest free lunch "
            "in markets: risk-free, documented at \\$40M, open to anyone with a wallet. If "
            "instead it's gone in **seconds**, the same \\$40M is the opposite lesson — a real "
            "edge that is *entirely* about **execution speed**, where everyone slow is just "
            "feeding the fast wallets their exit. The arbitrage being real is exactly what makes "
            "the retail pitch dangerous: the maths check out, so you never suspect the catch."
        ),

        md(
            "## 3 · How we'd know 🔍\n\n"
            "We measure the **half-life** of the gap — minutes until an opened mispricing has "
            "shrunk by half. The rule, set in advance: if it's comfortably longer than a human "
            "reaction, the penny is **reachable**; if it's at or below the finest tape we can "
            "sample (1 minute), it's a **mirage** for anyone not racing the 2-second block. We "
            "prove the stopwatch on the synthetic first, where we know the answer is 6 minutes."
        ),
        code(
            "eps = arbitrage.detect_all(gap, open_threshold=truth.open_threshold)\n"
            "print(f'episodes: {len(eps)}')\n"
            "print(f\"recovered half-life: {arbitrage.time_to_half(eps):.1f} min \"\n"
            "      f\"(baked in: {truth.half_life_min} min)\")\n"
            "robustness.survival_curve(eps)['frac_open'].plot(\n"
            "    title='Synthetic: fraction of arbitrages still open after t minutes')\n"
            "plt.xlabel('minutes since open'); plt.ylabel('fraction still open'); plt.show()"
        ),

        md(
            "## 4 · The teardown 🔬\n\n"
            "On the synthetic the stopwatch nails the baked-in 6 minutes — the machinery is "
            "sound. Now the real world, quoted from the reproducible run "
            "([`../docs/results.md`](../docs/results.md), via `examples/verify_real.py`):\n\n"
            "- **The penny is real:** 13 markets, **161** gaps clear 3¢, median **4¢**.\n"
            "- **And gone by the next sample:** **100%** of episodes are *below the 1-minute "
            "floor* — median duration **1 minute**, too short to even *see* halving.\n"
            "- **The tell:** coarsen the tape and the median episode is *one tick* at every "
            "resolution (1, 2, 5, 15, 30, 60 min) while the count collapses **161 → 2**. Its "
            "true lifetime is below *anything* we can sample.\n"
            "- **So you keep almost none of it:** even granting a generous 1-minute half-life, a "
            "5-minute reaction keeps **3%**, a 30-minute reaction **~0%**."
        ),
        code(
            "print('retail capture vs reaction latency (generous 1-min half-life):')\n"
            "display(robustness.retail_capture_table(1.0).round(4))"
        ),

        md(
            "## 5 · The verdict ⚖️\n\n"
            "**Signal `REAL`** — the free money exists, no argument. **Tradability `MIRAGE`** — "
            "it closes below the 1-minute tape floor, so a human keeps ~0 of it. **Execution-moat "
            "`CONFIRMED`** — a lifetime that hides under any tape is a prize reserved for the "
            "~30-millisecond wallets. The rare case where the signal is bulletproof and the "
            "mirage is *entirely* in who can reach it."
        ),

        md(
            "## 6 · Could you trade it? 🏦\n\n"
            "No — and not because it's fake. Polymarket's order book fills your two legs **one at "
            "a time**: you buy YES at \\$0.30, the book moves, your NO fills at \\$0.78, and the "
            "guaranteed \\$0.40 is now a loss. The fast wallets dodge this by firing both legs "
            "inside one 2-second block from a 30-ms loop; even *they* miss **45%** of the "
            "combinatorial trades. Copy-trading fails too — you see the fill a block late and buy "
            "from the very wallet you copied. And the thread's real ask isn't a trade; it's a "
            "wallet connect and an \"airdrop.\" The product is you."
        ),

        md(
            "## 7 · Going further 🚪\n\n"
            "- **Measure it directly:** rebuild the gap from on-chain `OrderFilled` events and "
            "clock the half-life in *seconds* instead of bounding it at a minute.\n"
            "- **The bigger pot:** the \\$29M *combinatorial* arbitrage (across linked markets), "
            "untouched here.\n"
            "- **Crisis test:** does the gap last any longer when a whole market dislocates at "
            "once (election night, a sports final)?\n\n"
            "The deep version — both estimators, the bootstrap, the resolution sweep, the capture "
            "math — is in [`02_for_the_quants.ipynb`](02_for_the_quants.ipynb)."
        ),
    ]
    return new_notebook(cells=cells, metadata=_meta())


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Vanishing-Penny — the teardown 🪙🔬\n"
            "### The half-life of a risk-free Polymarket gap: episodes, two estimators, the resolution moat\n\n"
            "The rigorous companion to [`01_for_the_curious.ipynb`](01_for_the_curious.ipynb). "
            "Same seven beats, full method. Thesis: the `YES+NO` arbitrage is **real** "
            "(risk-free, ~\\$40M documented) but its **half-life is below the public minute "
            "tape**, so realised retail capture `½^(latency/H)` ≈ 0 — a mirage that is entirely "
            "an execution moat, not a signal failure.\n\n"
            "> ⚠️ **Executed on the synthetic book** (baked-in 6-min half-life), where the "
            "estimator works — this validates the stopwatch end-to-end. The real verdict is "
            "quoted from [`../docs/results.md`](../docs/results.md) (`examples/verify_real.py`). "
            "Fixed seeds; no network."
        ),
        code(BOOT),

        md(
            "## 1 · The claim, as a first-passage hypothesis\n\n"
            "H₀ (the thread's hope): `g = 1 − (p_yes + p_no)` persists seconds-to-minutes — long "
            "enough to place two CLOB legs and bank `g`.\n"
            "H₁ (prior): competing bots close `g` far below human reaction, so the **half-life "
            "`H`** is the whole game and `½^(latency/H) ≈ 0`.\n"
            "We test H₀ by *measuring `H`* — first on synthetic data where `H` is known."
        ),
        code(
            "eps = arbitrage.detect_all(gap, open_threshold=truth.open_threshold)\n"
            "print('episodes:', len(eps))\n"
            "print('summary:', {k:(round(v,3) if isinstance(v,float) else v) for k,v in arbitrage.summary(eps).items()})"
        ),

        md(
            "## 3–4 · Two estimators must agree (or the decay isn't exponential)\n\n"
            "`time_to_half` is assumption-light (median per-episode steps to half its peak); "
            "`fit_half_life` is a pooled through-origin slope of `log(|g|/|g_peak|)` on Δt. On a "
            "clean exponential they coincide — and on the synthetic they land on the baked-in 6."
        ),
        code(
            "print(f\"baked-in   : {truth.half_life_min:.2f} min\")\n"
            "print(f\"time-to-half: {arbitrage.time_to_half(eps):.2f} min\")\n"
            "print(f\"log-lin fit : {arbitrage.fit_half_life(eps):.2f} min\")\n"
            "print('bootstrap CI:', {k:(round(v,2) if isinstance(v,float) else v) for k,v in robustness.bootstrap_half_life(eps, n_boot=2000).items()})"
        ),
        md(
            "> On **real** data these read differently — and that *is* the result: "
            "`half_life_median = nan` because **100% of episodes are below the 1-minute floor** "
            "(`frac_below_floor = 1.0`, median duration 1 min). The bootstrap CI is `nan` by "
            "construction (no finite per-episode half-lives to resample); the `7.72` log-lin fit "
            "survives only on the rare slow tail and is selection-biased upward."
        ),

        md(
            "## 4 · The resolution sweep — the load-bearing caveat, as a number\n\n"
            "Re-detect on a deliberately coarsened tape. On real data the median episode is "
            "*exactly one sample* at **every** fidelity (1→60 min) while the count collapses — the "
            "fingerprint of a timescale below all of them. On the synthetic (true `H` = 6 min) the "
            "sweep instead inflates past 6 only once the grid is coarser than the half-life:"
        ),
        code(
            "display(robustness.resolution_sweep(gap).round(3))"
        ),

        md(
            "## 5 · The verdict, with the numbers\n\n"
            "**Signal `REAL`** (risk-free; 161 real ≥3¢ gaps, median 4¢; ~\\$40M documented). "
            "**Tradability `MIRAGE`** (100% sub-floor, median life 1 min; capture 3% at 5-min "
            "latency, ~0 at 30). **Execution-moat `CONFIRMED`** (median life = sampling interval "
            "at every fidelity; the ~30-ms wallets own it). Real numbers: as-of 2026-06-01, "
            "fingerprint `7baea17d9b7b`, in [`../docs/results.md`](../docs/results.md)."
        ),

        md(
            "## 6 · Could you trade it — the capture integral\n\n"
            "Break-even is moot (the gross is risk-free); the binding quantity is the **capture "
            "fraction** `½^(latency/H)`. With `H` sub-minute and any human `latency` in minutes, "
            "it's rounding error — *before* the two half-spreads and gas the sequential-CLOB fill "
            "costs you. This is the cost sweep's analogue: there's no spread at which a "
            "minutes-late human captures a seconds-lived penny."
        ),
        code(
            "import numpy as np\n"
            "for H in (0.25, 0.5, 1.0, 6.0):\n"
            "    row = {f'{lat}m': round(robustness.retail_capture(H, lat),4) for lat in (1,5,30)}\n"
            "    print(f'H={H:>4} min  capture @', row)"
        ),

        md(
            "## 7 · Going further\n\n"
            "- **Sub-second `H`** from on-chain `OrderFilled` events (the paper's 86M-tx "
            "substrate) — measure the moat instead of bounding it.\n"
            "- **Combinatorial arbitrage** (the \\$29M class): cross-market dependency graph, its "
            "own half-lives and the paper's 45% fill rate.\n"
            "- **Capacity from pre-2026-02-20 depth snapshots** — the professional's constraint "
            "this study brackets.\n\n"
            "Engine: [`../../../quantlab/`](../../../quantlab/). Method: "
            "[`METHODOLOGY.md`](../../../METHODOLOGY.md). Real run: `examples/verify_real.py`."
        ),
    ]
    return new_notebook(cells=cells, metadata=_meta())


def _meta():
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }


def main():
    targets = {
        "01_for_the_curious.ipynb": build_curious(),
        "02_for_the_quants.ipynb": build_quants(),
    }
    for fname, nb in targets.items():
        path = os.path.join(HERE, fname)
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print(f"wrote {fname}  ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
