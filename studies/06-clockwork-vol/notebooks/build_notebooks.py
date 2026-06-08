"""Generate the two narrative notebooks for Study 06 (Clockwork-Vol) from source.

Like Studies 01–05, the notebooks are a *generated artefact*: edit the cell text here,
rebuild the skeletons, then execute with nbconvert to embed figures/outputs.

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

The executed path runs on the **offline synthetic series** — a log-VIX with a *known* fixed
cycle (80d & 40d) buried in AR(1) red noise — because the cached real parquet is git-ignored
and the desk's reproducible core must run with no network. That synthetic is where the
detector *works* (it clears the red-noise envelope at the injected periods and forecasts
their turns), which is exactly the point: it proves the code, so the **null real verdict**
(quoted from [`docs/results.md`](../docs/results.md), produced by `examples/verify_real.py`)
is a fact about the market, not a bug. Both notebooks follow the SAME seven desk beats
(see ../../../METHODOLOGY.md).
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))           # study root (vix_cycles/ lives there)
sys.path.insert(0, os.path.abspath("../../.."))      # repo root, for quantlab
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (9.5, 5.2)
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
from vix_cycles import data, spectral, cycles, backtest, robustness

# Offline synthetic log-VIX: a KNOWN fixed cycle (80d & 40d) hidden in AR(1) red noise. This
# is where the detector SHOULD work — so it validates the machinery. The real verdict (the
# VIX, where it does NOT) is in ../docs/results.md via verify_real.py.
series, injected = data.synthetic_cycle(seed=0)
print(f"{len(series)} sessions, injected cycles {[round(c.period) for c in injected]} (a real clock, by construction)")
"""


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


def _meta():
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the VIX run on a clock? ⏰\n"
            "### \"An 80-day volatility cycle, low formed May 29\" — tested honestly, in plain English\n\n"
            "Every so often a chart goes round showing the **VIX moving in tidy cycles**: an "
            "80-day rhythm, a nested 40-day one, lows and highs you can mark on a calendar weeks "
            "ahead. The [prompting thread](../docs/references.md) even dated them — a cycle low "
            "'formed May 29', a peak due late July. If volatility really ran on a clock like "
            "that, you could *time* it.\n\n"
            "Here's the catch the eye never sees: a slow, drifting, random series — **red noise** — "
            "sprouts exactly these 'cycles' all by itself. So the only honest question is whether "
            "the VIX's cycle is taller than the ones pure noise invents.\n\n"
            "> ⚠️ **Not investment advice.** The reproducible core runs on a **synthetic** series "
            "with a *real* cycle baked in (the cached VIX is git-ignored). That's on purpose: the "
            "synthetic is where the detector *works*, which proves the code — so the flat result on "
            "the real VIX (quoted from [`../docs/results.md`](../docs/results.md)) is a fact about "
            "the market, not a bug.\n\n"
            "*Follows the desk's seven beats ([METHODOLOGY.md](../../../METHODOLOGY.md)). The "
            "rigorous version is the companion,* "
            "[`02_for_the_quants.ipynb`](02_for_the_quants.ipynb)."
        ),
        code(BOOT),

        md(
            "## The answer first 🎯\n\n"
            "| What we asked | The honest answer |\n"
            "|---|---|\n"
            "| Can the detector find a real cycle? | ✅ **Yes** — on a series with a true 80-day "
            "cycle, it lights up well above what noise fakes. |\n"
            "| Does the VIX's 80/40-day 'cycle' clear that bar? | ❌ **No** — its peaks sit *inside* "
            "the red-noise envelope. |\n"
            "| Does the period stay fixed? | ❌ **No** — the 'dominant cycle' wanders window to "
            "window; it has to be re-drawn. |\n"
            "| Could you trade it? | ❌ **No** — the walk-forward forecast is a coin flip; the "
            "cycle trade doesn't beat a scrambled-phase clock. |\n\n"
            "> Desk shorthand: **Signal `NONE` · Tradability `MIRAGE`** — let's see the method earn them."
        ),

        md(
            "## 1 · The claim 📣\n\n"
            "Volatility swings decompose into **fixed-length cycles** — an ~80-session one with a "
            "nested ~40-session one — whose lows and highs you can project forward. Mark the last "
            "low, add a period, and you know roughly when the next turn is due. Drawn on a chart, "
            "it's mesmerising. Let's extract exactly that cycle from a series we *know* has one:"
        ),
        code(
            "dominant = max(injected, key=lambda c: c.amplitude)\n"
            "band = cycles.bandpass(series, dominant.period, width_frac=0.4)\n"
            "tp = cycles.turning_points(series, dominant.period)\n"
            "ax = band.plot(title=f'The extracted ~{dominant.period:.0f}-session cycle (synthetic, real by construction)')\n"
            "for d in tp['lows']: ax.axvline(d, color='g', alpha=0.25)\n"
            "ax.set_ylabel('cycle component (log-VIX)'); plt.show()\n"
            "print(f\"marked {len(tp['lows'])} cycle lows, ~{dominant.period:.0f} sessions apart\")"
        ),
        md("Tidy, isn't it? The whole question is whether the *real* VIX gives you a picture this "
           "clean — or whether noise does too."),

        md(
            "## 2 · So what? 💰\n\n"
            "If volatility ran on a fixed clock, you could time the things that hang off it: buy "
            "the dip when the cycle says a vol *peak* (and a stock *low*) is forming, lift hedges "
            "when it says calm is due. A reliable 80-day vol clock would be one of the cleanest "
            "market-timing edges there is — which is exactly why we should be suspicious of it."
        ),

        md(
            "## 3 · How we'd know 🔍\n\n"
            "Three tests, announced up front. **(1)** Does the cycle's peak stand above the "
            "**red-noise envelope** — the peaks a persistent-but-random series makes on its own? "
            "**(2)** Does the period stay **fixed**, or wander? **(3)** Walking forward, does the "
            "projected cycle **forecast** the next move better than the same cycle with its timing "
            "scrambled? On the synthetic, all three should say *yes* — proving the test works:"
        ),
        code(
            "env = spectral.red_noise_envelope(series, n_sim=600, seed=0)\n"
            "peaks = spectral.significant_peaks(env, q=0.99)\n"
            "print('significant peaks (clear the 99% noise envelope):')\n"
            "print(peaks.round(3).to_string(index=False) if len(peaks) else '  none')\n"
            "skill = backtest.oos_direction_skill(series, band=(20,200), horizon=20, min_train=750, n_null=300, seed=0)\n"
            "print(f\"walk-forward direction skill: {skill['skill']:.0%}  (coin=50%, p={skill['p_value']:.3f})\")"
        ),

        md(
            "## 4 · The teardown 🔬\n\n"
            "On the synthetic, the detector nails the real cycle (peaks above the envelope, "
            "forecast beats the coin). So the machine is sound. Now the **real VIX**, quoted from "
            "the reproducible run ([`../docs/results.md`](../docs/results.md), via "
            "`examples/verify_real.py`):\n\n"
            "- **The claimed cycles aren't significant.** At 40 and 80 sessions the VIX's "
            "periodogram peak sits *inside* the red-noise envelope — noise fakes peaks just as "
            "tall (see results.md for the exact p-values).\n"
            "- **The period won't hold still.** The 'dominant cycle' wanders across a wide range of "
            "lengths window to window — the signature of a curve-fit, not a clock.\n"
            "- **The forecast is a coin flip.** Walking forward, the projected cycle calls the next "
            "move no better than chance, and **no better than a random-phase clock**.\n"
            "- **The rescues don't help** (beat 7): committing to the literal 80d/40d period, or "
            "only trading when a strong cycle is visible, leaves it at a coin flip."
        ),

        md(
            "## 5 · The verdict ⚖️\n\n"
            "**Signal `NONE`** — the VIX's fixed-period cycles don't clear red noise, and their "
            "period doesn't hold. **Tradability `MIRAGE`** — the walk-forward forecast is a coin "
            "flip and the cycle trade doesn't beat a scrambled clock. The synthetic proves the "
            "detector finds real cycles; the VIX simply isn't one. What looks like a clock is the "
            "eye reading rhythm into persistent noise."
        ),

        md(
            "## 6 · Could you trade it? 🏦\n\n"
            "There's nothing to execute: the forecast doesn't beat a coin before a single cost. You "
            "*can't* even buy spot VIX — the honest expression is timing the S&P off the projected "
            "vol cycle, and that doesn't beat a random-phase version of itself. The cycle adds no "
            "information over simply being in the market."
        ),
        code(
            "px = pd.Series(np.exp(np.cumsum(-0.003*(series.values-series.values.mean()))), index=series.index)\n"
            "res = backtest.phase_trade(series, px, band=(20,200), horizon=20, min_train=750, n_null=50, seed=0)\n"
            "print({k: round(v,4) for k,v in res.stats.items() if k in ('sharpe_net','p_value_vs_null','exposure')})"
        ),

        md(
            "## 7 · Going further 🚪\n\n"
            "We didn't leave the rescues as homework — we ran them on the real VIX "
            "([`../docs/extensions.md`](../docs/extensions.md)):\n\n"
            "- **Commit to the literal 80d / 40d clock** (no re-tuning) — still a coin flip.\n"
            "- **Only trade when a strong cycle is visible** (amplitude gate) — fewer trades, same "
            "non-edge.\n"
            "- **Is the cycle in *stocks* instead?** The S&P's 20-week and 4-year clocks don't clear "
            "red noise either.\n\n"
            "Still open: longer VIX-futures-term-structure cycles, and a formal wavelet "
            "time-localization. The deep version — the envelope, the p-values, the walk-forward and "
            "the trade — is in [`02_for_the_quants.ipynb`](02_for_the_quants.ipynb)."
        ),
    ]
    return new_notebook(cells=cells, metadata=_meta())


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Clockwork-Vol — the teardown ⏰🔬\n"
            "### Fixed-period VIX cycles vs an AR(1) red-noise null: periodogram, envelope, walk-forward\n\n"
            "The rigorous companion to [`01_for_the_curious.ipynb`](01_for_the_curious.ipynb). Same "
            "seven beats, full method. Thesis: the VIX's claimed 40/80-session cycles **do not "
            "exceed the AR(1) red-noise envelope**, their period is **unstable**, and a "
            "walk-forward projection carries **no forecast skill** over a random-phase null.\n\n"
            "> ⚠️ **Executed on the synthetic series** (a real injected cycle), where the detector "
            "works — this validates the machinery end-to-end. The real verdict is quoted from "
            "[`../docs/results.md`](../docs/results.md) (`examples/verify_real.py`). Fixed seeds; no "
            "network."
        ),
        code(BOOT),

        md(
            "## 1 · The claim, as a testable hypothesis\n\n"
            "H₁: the log-VIX periodogram has a peak at P∈{40, 80} sessions that **exceeds the "
            "(1−α) AR(1) red-noise quantile**; the period is **stable** across windows; and a "
            "fixed-period projection has **out-of-sample direction skill > ½**, beating a "
            "random-phase null.\n"
            "H₀: the VIX is **red noise** (AR(1), persistence ρ) plus mean reversion — its "
            "periodogram peaks are the broad, tall, *random* peaks autocorrelation manufactures, "
            "and carry no forward information."
        ),
        code(
            "env = spectral.red_noise_envelope(series, n_sim=800, seed=0)\n"
            "print('fitted AR(1) rho:', round(env['rho'],3))\n"
            "print('detection recall vs injected cycles:', robustness.detection_recall(env, injected, q=0.99))\n"
            "pd.DataFrame(robustness.red_noise_pvalues(series, targets=(40.,80.), n_sim=800))"
        ),

        md(
            "## 3–4 · The periodogram against the red-noise envelope\n\n"
            "The core picture: data spectrum vs the 95/99% envelope of AR(1) surrogates. A real "
            "cycle pokes above; a phantom hides inside. On the synthetic the injected periods clear "
            "it — on the VIX (results.md) they don't."
        ),
        code(
            "periods, power = spectral.periodogram(series)\n"
            "m = (periods>=15)&(periods<=400)\n"
            "plt.semilogx(periods[m], power[m], label='data')\n"
            "for q,c in [(0.95,'orange'),(0.99,'red')]:\n"
            "    plt.semilogx(env['periods'][m[:len(env['periods'])]], env['envelope'][q][m[:len(env['periods'])]], c, alpha=.7, label=f'{int(q*100)}% red-noise')\n"
            "plt.gca().invert_xaxis(); plt.xlabel('period (sessions)'); plt.ylabel('power')\n"
            "plt.title('Periodogram vs AR(1) envelope (synthetic: injected peaks clear it)'); plt.legend(); plt.show()"
        ),

        md(
            "## 4 · Period stability — a clock, or a curve-fit?\n\n"
            "Rolling the dominant in-band period: near-constant on the synthetic (a true clock), "
            "wandering on the VIX (results.md). The drift *is* the verdict — a cycle you must "
            "re-tune every quarter isn't a fixed cycle."
        ),
        code(
            "stab = robustness.period_stability(series, band=(20,200), window=1000, step=200)\n"
            "stab.plot(title=f\"dominant period (synthetic: std={stab['dominant_period'].std():.1f} sessions)\")\n"
            "plt.ylabel('period (sessions)'); plt.show()"
        ),

        md(
            "## 4 · Walk-forward forecast skill vs the random-phase null\n\n"
            "The decisive out-of-sample test: fit period+phase on the past, project the next-"
            "horizon direction, score vs a null that keeps period & amplitude but scrambles phase. "
            "Skill ≈ null ⇒ the timing carries nothing. Synthetic clears it; VIX (results.md) does not."
        ),
        code(
            "skill = backtest.oos_direction_skill(series, band=(20,200), horizon=20, min_train=750, n_null=500, seed=0)\n"
            "print({k: (round(v,4) if isinstance(v,float) else v) for k,v in skill.items()})"
        ),

        md(
            "## 5–6 · Verdict & the trade\n\n"
            "**Signal `NONE`**, **Tradability `MIRAGE`** (results.md). The tradeable expression — "
            "long the S&P when the VIX cycle is projected to fall — is scored against buy-and-hold "
            "*and* the random-phase null, with a bootstrap Sharpe CI."
        ),
        code(
            "px = pd.Series(np.exp(np.cumsum(-0.003*(series.values-series.values.mean()))), index=series.index)\n"
            "res = backtest.phase_trade(series, px, band=(20,200), horizon=20, min_train=750, n_null=100, seed=0)\n"
            "print({k: round(v,4) for k,v in res.stats.items()})\n"
            "print('bootstrap Sharpe CI:', {k: round(v,4) for k,v in robustness.bootstrap_sharpe(res.daily.iloc[750:]).items()})"
        ),

        md(
            "## 7 · Going further\n\n"
            "Beat-7 rescues, run on the real VIX in [`../docs/extensions.md`](../docs/extensions.md): "
            "literal 80d/40d periods, an amplitude gate, and the S&P's own 20-week / 4-year clocks — "
            "none clears the bar. Open threads: VIX-futures term-structure cycles and a formal "
            "wavelet (time-frequency) localization to chase a *transient* cycle the global "
            "periodogram would average away."
        ),
    ]
    return new_notebook(cells=cells, metadata=_meta())


def main():
    for name, nb in [("01_for_the_curious", build_curious()), ("02_for_the_quants", build_quants())]:
        path = os.path.join(HERE, f"{name}.ipynb")
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
