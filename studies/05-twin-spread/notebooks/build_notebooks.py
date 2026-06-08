"""Generate the two narrative notebooks for Study 05 (Twin-Spread) from source.

Like Studies 01–04, the notebooks are a *generated artefact*: edit the cell text here,
rebuild the skeletons, then execute with nbconvert to embed figures/outputs.

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

The executed path runs on the **offline synthetic universe** — a toy market with true
cointegrated twins hidden among noise — because the cached real parquets are git-ignored
and the desk's reproducible core must run with no network. That synthetic is where the
rule *works* (the machinery harvests real reversion), which is exactly the point: it
proves the code, so the **flat-to-negative real verdict** (quoted from
[`docs/results.md`](../docs/results.md), produced by `examples/verify_real.py`) is a fact
about the market, not a bug. Both notebooks follow the SAME seven desk beats
(see ../../../METHODOLOGY.md).
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))           # study root (pairs_trading/ lives there)
sys.path.insert(0, os.path.abspath("../../.."))      # repo root, for quantlab
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (9.5, 5.2)
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
from pairs_trading import data, pairs, backtest, robustness

# Offline synthetic universe: true cointegrated twins hidden among noise names. This is
# where the rule SHOULD work — so it validates the machinery. The real verdict (a liquid
# ~170-name basket where it does NOT work) is in ../docs/results.md via verify_real.py.
panel, frames, true_pairs = data.synthetic_universe(seed=0)
print(f"{panel.shape[1]} names, {panel.shape[0]} sessions, {len(true_pairs)} true twins baked in")
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
            "# Does pairs trading still pay after the world copied it? 👯\n"
            "### \"Two stocks drift apart, then snap back\" — tested honestly, in plain English\n\n"
            "Every few months a thread goes viral rediscovering **pairs trading**: find two "
            "stocks that move together, and when one runs ahead, short it and buy the other — "
            "they always snap back. It's the strategy that built the first quant desks, and the "
            "[famous 1999 paper](../docs/references.md) reported ~1.4% a month at near-zero "
            "market risk, *still paying even after it was published*.\n\n"
            "That last part is the hook. Most edges die the moment everyone knows them. Did this "
            "one?\n\n"
            "> ⚠️ **Not investment advice.** The reproducible core runs on a **synthetic** market "
            "with real twins baked in (the cached real prices are git-ignored). That's on "
            "purpose: the synthetic is where the rule *works*, which proves the code — so the "
            "flat real-world result (quoted from [`../docs/results.md`](../docs/results.md)) is a "
            "fact about the market, not a bug.\n\n"
            "*Follows the desk's seven beats ([METHODOLOGY.md](../../../METHODOLOGY.md)). The "
            "rigorous version is the companion,* "
            "[`02_for_the_quants.ipynb`](02_for_the_quants.ipynb)."
        ),
        code(BOOT),

        md(
            "## The answer first 🎯\n\n"
            "| What we asked | The honest answer |\n"
            "|---|---|\n"
            "| Does the rule find genuinely matched pairs? | ✅ **Yes** — given real twins, the "
            "minimum-distance selector recovers them almost perfectly. |\n"
            "| When matched pairs *do* revert, does it profit? | ✅ **Yes** — on the synthetic, "
            "~+0.95%/mo at a Sharpe near 1.7. The machinery works. |\n"
            "| Do real liquid stocks revert like that today? | ❌ **No** — on the real basket the "
            "spread doesn't pay: **negative even before costs**. |\n"
            "| Could you trade it? | ❌ **No edge to trade** — −0.43%/mo net, a −77% drawdown, "
            "market-neutral so there's nowhere to hide. |\n\n"
            "> Desk shorthand: **Signal `NONE` · Tradability `MIRAGE` · Decay `CONFIRMED`** — let's "
            "see the method earn them."
        ),

        md(
            "## 1 · The claim 📣\n\n"
            "Rank pairs of stocks by how tightly their price paths hugged over the past year. Take "
            "the closest ones. When a pair diverges by more than its usual wiggle (2 standard "
            "deviations), **short the one that ran up, buy the one that lagged**, and hold until "
            "they cross back. The pitch: it's *parameter-free* — no knobs to overfit — and it kept "
            "working after 1999."
        ),
        code(
            "formation = panel.iloc[:252]\n"
            "selected = pairs.select_pairs(formation, top_n=len(true_pairs))\n"
            "recall = robustness.selection_recall(selected, true_pairs)\n"
            "print(f'top-{len(true_pairs)} closest pairs recover {recall:.0%} of the true twins')\n"
            "norm = pairs.normalized_prices(formation)\n"
            "p = selected[0]\n"
            "plt.plot(norm.index, norm[p.a], label=p.a)\n"
            "plt.plot(norm.index, norm[p.b], label=p.b)\n"
            "plt.title(f'A matched pair over the formation year  ({p.a} ~ {p.b})')\n"
            "plt.ylabel('normalized price (starts at 1)'); plt.legend(); plt.show()"
        ),
        md("Two paths glued together — that's what 'a pair' means. The bet is that when they "
           "separate, they'll re-glue."),

        md(
            "## 2 · So what? 💰\n\n"
            "Pairs trading is the **origin story of statistical arbitrage**. If a rule this simple, "
            "public, and old still paid ~1.4%/mo at near-zero beta, it'd be free money that "
            "survived being written down — a glaring hole in 'markets are efficient'. If instead "
            "it's been competed to death, it's the textbook case of **alpha decay**: real once, "
            "found, crowded, gone — leaving a rule that still *looks* like it should work."
        ),

        md(
            "## 3 · How we'd know 🔍\n\n"
            "The trap: pairs trading **wins more than half its trades** (lots of little "
            "convergences) even when it loses money overall, because the rare blow-ups (pairs that "
            "break and never come back) are bigger. So we ignore the win rate and ask: does the "
            "**committed capital actually compound up**, and is it positive *before* costs?"
        ),
        code(
            "res = backtest.run(panel, top_n=8, form_len=252, trade_len=126, wait=1)\n"
            "s = res.stats\n"
            "print(f\"win rate: {s['win_rate_net']:.1%}   but the number that matters:\")\n"
            "print(f\"committed monthly net: {s['committed_monthly_net']:+.2%}   Sharpe: {s['sharpe_net']:.2f}\")\n"
            "res.equity.plot(title='Synthetic: committed-capital equity (true twins -> it works)')\n"
            "plt.ylabel('growth of $1'); plt.show()"
        ),

        md(
            "## 4 · The teardown 🔬\n\n"
            "On the synthetic, the rule does its job — the equity grinds up. So the machinery is "
            "sound. Now the real world, quoted from the reproducible run "
            "([`../docs/results.md`](../docs/results.md), via `examples/verify_real.py`):\n\n"
            "- **The selector still works** on real data — it finds tight pairs.\n"
            "- **But the spread doesn't pay.** Modern era (2005–2026): **−0.37%/mo gross**, "
            "**−0.43%/mo net**, Sharpe **−0.44**. Win rate 55.7% — *more winners than losers* — "
            "and still a negative mean.\n"
            "- **Negative even at zero cost** — so it isn't a cost problem; there's no edge.\n"
            "- **Decayed:** good years cluster in 1983–2003; the modern era is mostly red, worst in "
            "2020 / 2022 / 2023. The only green is in crises (2008, 2019, 2025)."
        ),

        md(
            "## 5 · The verdict ⚖️\n\n"
            "**Signal `NONE`** — real pairs don't reconverge into a profit (negative before costs, "
            "bootstrap Sharpe CI below zero). **Tradability `MIRAGE`** — no edge to trade, a −77% "
            "drawdown, market-neutral so no beta to bank. **Decay `CONFIRMED`** — the famous edge "
            "is behind us. The synthetic proves the only thing missing on real data is the one "
            "thing you can't code: stocks that actually revert."
        ),

        md(
            "## 6 · Could you trade it? 🏦\n\n"
            "Unusually clean: **there's nothing to execute well.** The signal is gone before the "
            "first cost, so no venue or sizing rescues it — and these are liquid mega-caps, so you "
            "*could* put real size on a slow, market-neutral bleed. The one place it earns its keep "
            "is a 2008-style dislocation, where everything reverts at once — a hedge with a fuse, "
            "not a standing book."
        ),
        code(
            "print('cost sweep is beside the point when gross < 0:')\n"
            "print(backtest.cost_sweep(panel, top_n=8, half_spread_grid=(0,5,20,50)).round(4)[['committed_monthly_net','sharpe_net']])"
        ),

        md(
            "## 7 · Going further 🚪\n\n"
            "- **A stop-loss** to cap the rare blow-ups (the −77% drawdown is uncapped losers).\n"
            "- **A cointegration filter** so a pair needs an *economic* reason to revert, not luck.\n"
            "- **A far broader universe** — GGR formed from *thousands* of names; we had ~170.\n\n"
            "Each is a concession that the *textbook* rule the thread sells doesn't clear the bar. "
            "The deep version — bootstrap, decay curve, neutrality, costs — is in "
            "[`02_for_the_quants.ipynb`](02_for_the_quants.ipynb)."
        ),
    ]
    return new_notebook(cells=cells, metadata=_meta())


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Twin-Spread — the teardown 👯🔬\n"
            "### GGR (1999) minimum-distance pairs: formation, the 2σ trade, decay, neutrality, costs\n\n"
            "The rigorous companion to [`01_for_the_curious.ipynb`](01_for_the_curious.ipynb). "
            "Same seven beats, full method. Thesis: the **parameter-free** GGR rule, run honestly "
            "on a liquid basket, has **no convergence edge** in the modern era — negative before "
            "costs, market-neutral (β≈0), and short-gamma (win-rate > 50%, mean ≤ 0).\n\n"
            "> ⚠️ **Executed on the synthetic universe** (true twins), where the rule works — this "
            "validates the machinery end-to-end. The real verdict is quoted from "
            "[`../docs/results.md`](../docs/results.md) (`examples/verify_real.py`). Fixed seeds; "
            "no network."
        ),
        code(BOOT),

        md(
            "## 1 · The claim, as a testable hypothesis\n\n"
            "H₁: the top-N minimum-SSD pairs earn a positive **committed-capital** convergence "
            "return, annualised Sharpe CI above 0, market β≈0.\n"
            "H₁′ (sharp): it survives bid-ask costs *and* the post-2002 regime.\n"
            "H₀: minimum-distance ≠ cointegration; convergence is a coin flip whose negative skew "
            "(non-reconverging breaks) pulls the mean to ≤ 0."
        ),
        code(
            "formation = panel.iloc[:252]\n"
            "selected = pairs.select_pairs(formation, top_n=len(true_pairs))\n"
            "print('selection recall on true twins:', robustness.selection_recall(selected, true_pairs))\n"
            "pd.DataFrame([(p.a, p.b, round(p.ssd,4), round(p.sigma,4)) for p in selected],\n"
            "             columns=['a','b','ssd','sigma'])"
        ),

        md(
            "## 3–4 · The trade, and committed-capital P&L\n\n"
            "Open at 2σ, close on the zero-crossing, **wait=1** (act on the triggering close, earn "
            "from the next session — no look-ahead). Committed capital averages all N pairs daily "
            "(idle pairs contribute 0)."
        ),
        code(
            "res = backtest.run(panel, top_n=8, form_len=252, trade_len=126, wait=1)\n"
            "print({k:(round(v,4) if isinstance(v,float) else v) for k,v in res.stats.items()})\n"
            "res.equity.plot(title='committed-capital equity (synthetic twins)'); plt.show()"
        ),

        md(
            "## 4 · Decay — the same rule, year by year\n\n"
            "On the synthetic there's no decay by construction (stationary twins). On **real** data "
            "this is the headline: positive years cluster 1983–2003, the modern era is mostly red. "
            "See [`../docs/results.md`](../docs/results.md)."
        ),
        code(
            "robustness.decay_by_year(panel, top_n=8).round(4)"
        ),

        md(
            "## 4 · Market-neutrality and the bootstrap Sharpe\n\n"
            "A dollar-neutral book should show β≈0 — so the return (good or bad) is the convergence "
            "rule itself, not disguised market exposure. The bootstrap CI is the desk-standard read "
            "on 'could the Sharpe be zero?'"
        ),
        code(
            "mkt = data.market_return(panel)\n"
            "print('neutrality:', {k:round(v,4) for k,v in robustness.market_neutrality(res.daily, mkt).items()})\n"
            "print('bootstrap :', {k:round(v,4) for k,v in robustness.bootstrap_sharpe(res.daily, n_boot=2000).items()})"
        ),
        md(
            "> On **real** data these read: β = −0.00, R² = 0.00 (cleanly neutral), and a modern "
            "bootstrap Sharpe CI of **[−0.81, −0.04]** — *entirely below zero*, 98% of resamples "
            "negative. The full-sample CI [−0.35, +0.13] straddles zero (the thin early universe "
            "dilutes it)."
        ),

        md(
            "## 5 · The verdict, with the numbers\n\n"
            "**Signal `NONE`** (real gross −0.37%/mo, CI below 0, negative at zero cost). "
            "**Tradability `MIRAGE`** (net −0.43%/mo, −77% DD, β≈0, capacity ample so liquidity "
            "isn't the constraint — the missing edge is). **Decay `CONFIRMED`** (best years "
            "1983–2003; modern era red; green only in dislocations — 2008 +0.9%/mo Sharpe 1.30)."
        ),

        md(
            "## 6 · Could you trade it — costs and the short-gamma tail\n\n"
            "The cost sweep on the synthetic is monotone (wider spread never helps); on real data "
            "it starts *already negative at 0 bp* — the tell that there's no edge underneath. The "
            "win-rate-vs-mean gap is the short-gamma signature: bounded gains to the cross, "
            "unbounded loss when a pair breaks."
        ),
        code(
            "sweep = backtest.cost_sweep(panel, top_n=8, half_spread_grid=(0,2,5,10,20,40))\n"
            "display(sweep.round(4))\n"
            "dvol = data.dollar_volume_panel(frames)\n"
            "edge = max(res.stats['mean_trade_net'], 1e-9)*1e4\n"
            "print('capacity (per leg) at a nominal 20bp edge:',\n"
            "      {k:(round(v,1) if isinstance(v,float) else v) for k,v in robustness.capacity(dvol, res.trades, edge_bps=max(edge,20)).items()})"
        ),

        md(
            "## 7 · Going further\n\n"
            "- **Stop-loss** — cap the uncapped short-gamma tail; does it make the skew survivable?\n"
            "- **Cointegration gate** (Engle–Granger / Johansen) instead of raw minimum-SSD.\n"
            "- **A real S&P 1500 cache** — point `data.load_universe` at thousands of names.\n"
            "- **Total-return prices** — re-run off the split-only bias (named, works against the rule).\n"
            "- **The crisis-only book** — is conditional pairs trading (2008/2020 hedge) a real, "
            "sizeable thing?\n\n"
            "Engine: [`../../../quantlab/`](../../../quantlab/). Method: "
            "[`METHODOLOGY.md`](../../../METHODOLOGY.md)."
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
