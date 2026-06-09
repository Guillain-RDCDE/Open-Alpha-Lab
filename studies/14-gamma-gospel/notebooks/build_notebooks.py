"""Generate the two narrative notebooks for Study 14 (Gamma-Gospel) from source.

Like the other studies, the notebooks are a *generated artefact*: edit the cell text here,
rebuild the skeletons, then execute with nbconvert to embed figures/outputs.

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

The executed path runs on the **offline synthetic tape** — two toy books of sessions, one with a
*baked-in genuine gamma effect* and one with *none*, both sharing a VIX-driven confound — because
the real SPY option chain is git-ignored and accumulated over several Alpha Vantage runs. The
synthetic is where the decomposition *works* (it recovers the real effect and unmasks the
trenchcoat), which proves the code; the **real verdict** is produced by `examples/verify.py` into
[`docs/results.md`](../docs/results.md). Both notebooks follow the SAME seven desk beats
(see ../../../METHODOLOGY.md).
"""

from __future__ import annotations

import os
import sys

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))  # repo root, for quantlab
from quantlab.repro import DEFAULT_AS_OF

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))           # study root (gamma_gospel/ lives there)
sys.path.insert(0, os.path.abspath("../../.."))      # repo root, for quantlab
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (9.5, 5.2)
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
from gamma_gospel import data, signals, decompose

# Two offline tapes that share a VIX-driven confound (VIX moves BOTH the GEX sign and the day's
# character). REAL has a genuine gamma effect baked on top (beta_de=0.06); TRENCH has none.
real, t_real = data.synthetic_panel(beta_vol=0.0020, beta_de=0.060, seed=0)
trench, t_tr = data.synthetic_panel(beta_vol=0.0, beta_de=0.0, seed=0)
neg = real["neg_gamma"].astype(bool)
print(f"{len(real)} sessions | VIX on neg-gamma days {real.loc[neg,'vix'].mean():.1f} "
      f"vs pos {real.loc[~neg,'vix'].mean():.1f} | baked-in beta_de REAL={t_real.beta_de}, "
      f"TRENCH={t_tr.beta_de}")
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
            "# Gamma-Gospel — does dealer gamma call the day? 🃏\n"
            "### Is the GEX \"regime read\" a real dealer-hedging signal, or the VIX in a trenchcoat?\n\n"
            "![Signal: Pre-reg](https://img.shields.io/badge/Signal-Pre--reg-8b949e?style=flat-square)\n"
            "![Tradability: Pre-reg](https://img.shields.io/badge/Tradability-Pre--reg-8b949e?style=flat-square)\n\n"
            "A viral thread says price is just the *output*; the real *input* is **dealer hedging** — "
            "long-gamma dealers fade moves (calm **range** day), short-gamma dealers chase them "
            "(violent **trend** day), and the sign of net **GEX**, knowable before the open, calls the "
            "day's character. But there's a suspect hiding in plain sight: **the VIX** — scary days "
            "are both more trending *and* put-heavy (negative-gamma). This study is **pre-registered**: "
            "no real-market results yet, only the expectation that GEX is mostly the volatility regime "
            "relabeled.\n\n"
            "> 📓 **This is the plain-language layer.** The deep companion is "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** Pre-registered design: the stamps above are expectations, "
            "to be earned (or refuted) by the run on the real SPY chain "
            "([`examples/verify.py`](../examples/verify.py) → [`../docs/results.md`](../docs/results.md)). "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## The answer first 🎯\n\n"
            "| What we asked | The honest answer |\n"
            "|---|---|\n"
            "| Are negative-gamma days really more volatile / more trending? | ✅ **Yes, raw** — "
            "but those are also the high-VIX days. |\n"
            "| Does GEX's sign add anything *over the VIX*? | ⏳ **That's the test** — the pitch "
            "lives or dies on whether the gap *survives* controlling for VIX. |\n"
            "| Can our machine even tell a real effect from a relabel? | ✅ **Yes** — below it "
            "recovers a baked-in effect and unmasks a baked-in fake. |\n"
            "| Would a regime *sign* be a trade? | ❌ **Not by itself** — it's a bias on the day's "
            "character, not an entry, and it needs the whole-day hold to express. |\n\n"
            "> Desk shorthand (pre-registered): we expect **Signal `WEAK`** — most of the headline "
            "is the volatility regime relabeled — and **Tradability `MIRAGE`**. The real run "
            "confirms or breaks it."
        ),

        md(
            "## 1 · The claim 📣\n\n"
            "Dealers (market-makers) don't bet on direction — they hedge. Their **gamma** says how "
            "hard they must trade per point of move. Sum it across all strikes (calls add, puts "
            "subtract, under the standard assumption that dealers are long calls and short puts) and "
            "you get net **GEX**. **GEX > 0** (long gamma): dealers fade every move → a *range* day, "
            "vol suppressed. **GEX < 0** (short gamma): dealers amplify every move → a *trend* day, "
            "vol turbocharged. The pitch says the sign — visible before the open — is the most "
            "important thing you can know about the session."
        ),

        md(
            "## 2 · So what? 💰\n\n"
            "If true, you'd know each morning whether to **fade the extremes** (sell premium, trade "
            "range) or **ride breakouts** (momentum) — a regime filter worth real money, and a sign "
            "that a big chunk of intraday behaviour is *mechanically* set by dealer hedging rather "
            "than news. The subtler, likelier story: negative-gamma days really are wilder, but "
            "mostly because **negative gamma is what a high-VIX market looks like** — and you didn't "
            "need an options model to see the VIX. Confuse the two and you're paying \\$7/month to be "
            "told 'it's a scary day' in greek letters."
        ),

        md(
            "## 3 · How we'd know 🔍\n\n"
            "Build a daily panel: the **GEX sign** at the prior close, the next day's **character** "
            "(a range-based vol, and *directional efficiency* = |close−open|/(high−low): ~1 on a "
            "trend day, ~0 on a chop day), and the prior-close **VIX**. First check the raw gap. "
            "Then the decisive move: **partial out the VIX**. If the negative-gamma effect "
            "*survives* the control, GEX carries real information. If it *collapses*, GEX was the "
            "volatility regime in a trenchcoat. We prove the test works on the synthetic, where we "
            "baked the answer in — both versions."
        ),
        code(
            "for name, panel, truth in [('REAL effect', real, t_real), ('TRENCHCOAT', trench, t_tr)]:\n"
            "    raw = decompose.regime_gap(panel, 'de')\n"
            "    p = decompose.partial_over_vix(panel, 'de')\n"
            "    print(f\"{name:12} raw DE gap {raw['gap']:+.3f} (t {raw['t']:+.1f})  ->  \"\n"
            "          f\"survives VIX {p['surviving_coef']:+.3f} (t {p['surviving_t']:+.1f}), \"\n"
            "          f\"{p['survival_share']:.0%} kept  [baked-in beta_de={truth.beta_de}]\")"
        ),
        md("Notice: **both worlds show the same big raw gap** — that's the trap. Only the "
           "VIX-controlled number tells them apart. In the real-effect world ~37% survives near the "
           "baked-in 0.06; in the trenchcoat world it goes to ~0. The raw headline is *not* the "
           "evidence."),

        md(
            "## 4 · The teardown 🔬\n\n"
            "The whole study in one picture: the **raw** regime gap (what the pitch shows you) next "
            "to the **VIX-controlled** gap (what's actually GEX's own information), in both worlds."
        ),
        code(
            "labels = ['raw gap', 'survives VIX-control']\n"
            "fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)\n"
            "for a, (name, panel, truth) in zip(ax, [('Real gamma effect', real, t_real),\n"
            "                                        ('Trenchcoat (no effect)', trench, t_tr)]):\n"
            "    raw = decompose.regime_gap(panel, 'de'); p = decompose.partial_over_vix(panel, 'de')\n"
            "    vals = [raw['gap'], p['surviving_coef']]\n"
            "    a.bar(labels, vals, color=['#888888', '#b22222'])\n"
            "    a.axhline(0, color='k', lw=0.6)\n"
            "    if truth.beta_de: a.axhline(truth.beta_de, color='#1f77b4', ls='--', lw=1.5,\n"
            "                                label=f'baked-in {truth.beta_de}')\n"
            "    a.set_title(name); a.legend()\n"
            "    for i, v in enumerate(vals): a.text(i, v + 0.004, f'{v:+.3f}', ha='center')\n"
            "ax[0].set_ylabel('negative-gamma effect on directional efficiency')\n"
            "plt.suptitle('Same raw gap, opposite truth — only the VIX control separates them')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "**What this means for the real tape.** On real SPY the raw gap will be large — negative "
            "gamma days *are* wilder. The verdict hinges entirely on the right-hand bar: does the "
            "negative-gamma coefficient stay up (near the blue line, real information) or fall to "
            "the floor (trenchcoat)? Given that negative gamma is essentially *defined* by a "
            "put-heavy, high-VIX book, the desk's prior — and the lesson of "
            "[Study 12](../../12-paper-prophet/) — is that the range-vol leg is almost pure VIX, and "
            "the only place GEX might keep something of its own is *directional efficiency*. "
            "[`../docs/results.md`](../docs/results.md) settles it on the real chain."
        ),

        md(
            "## 5 · The verdict ⚖️ *(pre-registered)*\n\n"
            "**Signal `WEAK` (expected)** — there's a real raw gap, but we expect most of it to be "
            "the volatility regime; GEX earns `REAL` only if the VIX-controlled coefficient stays "
            "significant, especially on directional efficiency. **Tradability `MIRAGE` (expected)** "
            "— even a surviving sign is a *bias on the day's character*, not an entry, and expressing "
            "it needs a full-session hold against a round-trip spread. The synthetic proves the test "
            "is honest; the real numbers fill the stamps."
        ),

        md(
            "## 6 · Could you trade it? 🏦\n\n"
            "Suppose the sign *does* survive VIX. You'd still only know *what kind* of day it is — "
            "fade extremes when long-gamma, expect trend when short-gamma. That's a **regime filter**, "
            "not a trade: to monetise it you overlay it on an actual entry, and you pay the spread "
            "either way. A directional-efficiency tilt of a few points won't pay a round-trip on SPY "
            "options or a full-day index hold by itself. The pitch even concedes this — *\"you don't "
            "need to re-architect your strategy… it shifts a few things.\"* A \\$7/month context "
            "feed can be genuinely useful and still not be an edge you can withdraw."
        ),

        md(
            "## 7 · Going further 🚪\n\n"
            "- **The walls and the flip.** Does price actually respect the call-wall / put-wall as "
            "intraday extremes more than random strikes? `compute_gex` returns them; a permutation "
            "null would test it.\n"
            "- **0DTE / charm / vanna.** The pitch's advanced legs — afternoon charm drift, "
            "vanna-driven rallies on IV compression. Each is its own falsifiable sub-study.\n"
            "- **Intraday character.** We grade the day from daily OHLC; with intraday bars you could "
            "test the *mean-reversion vs momentum* claim minute-by-minute, not just end-of-day.\n"
            "- **Run the real chain.** A reliable historical GEX needs open interest by strike, "
            "which (we checked) the free sources don't carry — Alpha Vantage's `HISTORICAL_OPTIONS` "
            "has it but is *premium*. With a paid chain key, `python examples/verify.py --fetch` "
            "accumulates the SPY panel and earns the stamps; that data reality is why this ships "
            "pre-registered.\n\n"
            "The deep version — the HAC nested regression, the beta recovery, the trenchcoat "
            "collapse — is in [`02_for_the_quants.ipynb`](02_for_the_quants.ipynb)."
        ),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Gamma-Gospel — a quantitative teardown 🔬\n"
            "### GEX construction · unobservable dealer assumption · HAC nested regression · baked-in-β recovery · the VIX trenchcoat\n\n"
            "![Signal: Pre-reg](https://img.shields.io/badge/Signal-Pre--reg-8b949e?style=flat-square)\n"
            "![Tradability: Pre-reg](https://img.shields.io/badge/Tradability-Pre--reg-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats.* **Pre-registered** thesis: the GEX-regime claim contains a real but modest "
            "dealer-gamma effect **largely confounded by the volatility level** — a raw negative-gamma "
            "gap in realised vol and directional efficiency that mostly collapses once VIX is "
            "partialled out — and rests on an **unobservable dealer-positioning assumption** (long "
            "calls / short puts) we flag rather than hide.\n\n"
            "> ⚠️ **Not investment advice.** Pre-registered design. Executed here on the **synthetic "
            "tape** — two panels sharing a VIX-driven confound, one with a baked-in genuine gamma "
            "effect (`beta_de=0.06`, `beta_vol=0.002`) and one with none — where the nested HAC "
            "regression recovers exactly what we put in (fixed seeds, no network); the real SPY-chain "
            "stamps come from [`examples/verify.py`](../examples/verify.py) → "
            "[`../docs/results.md`](../docs/results.md), literature in "
            "[`../docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition. House "
            "style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## 1 · The construction, and its load-bearing assumption\n\n"
            "GEX is **not observed** — it is *assumed*. Under the SqueezeMetrics / retail convention "
            "customers buy index puts and overwrite calls, so dealers are long call gamma and short "
            "put gamma:\n\n"
            "$$\\text{GEX} = \\sum_{\\text{calls}} \\Gamma_i\\, \\text{OI}_i\\, \\cdot 100 \\cdot S^2 "
            "\\;-\\; \\sum_{\\text{puts}} \\Gamma_i\\, \\text{OI}_i\\, \\cdot 100 \\cdot S^2.$$\n\n"
            "Only the **sign** is load-bearing here. `compute_gex` implements it; flipping the "
            "call/put open-interest weight flips the sign deterministically, which is what the "
            "unit-tests assert. The dealer-direction assumption is the single biggest modelling "
            "risk — if real dealer inventory differs, the whole map inverts."
        ),
        code(
            "call_heavy = data.synthetic_chain(call_oi_scale=2.0, put_oi_scale=1.0, seed=1)\n"
            "put_heavy = data.synthetic_chain(call_oi_scale=1.0, put_oi_scale=2.0, seed=1)\n"
            "print('call-heavy chain GEX:', round(data.compute_gex(call_heavy, 100.0)['gex'], 3), '(>0, long gamma)')\n"
            "print('put-heavy  chain GEX:', round(data.compute_gex(put_heavy, 100.0)['gex'], 3), '(<0, short gamma)')"
        ),

        md(
            "## 1b · The data-generating process (so we know the truth)\n\n"
            "The synthetic bakes the exact structure the real study must untangle:\n\n"
            "- a persistent **VIX** regime (AR(1)) — *observable*;\n"
            "- `P(neg_gamma) = logistic(γ·(VIX − VIX̄))` — high VIX ⇒ put-heavy ⇒ negative gamma "
            "(**the confound**: VIX drives the regime label);\n"
            "- `rv = a + s·VIX + β_vol·neg_gamma + ε` and `de = clip(c + u·VIX + β_de·neg_gamma + η)` "
            "— each outcome rises with VIX *and*, by `β`, with negative gamma **independently** of "
            "VIX.\n\n"
            "So the raw gap is real by construction, but only the `β` part is *GEX's own*. The whole "
            "verdict is recovering `β` after removing VIX."
        ),
        code(
            "print('confound check — mean VIX by regime:')\n"
            "print(real.groupby(real['neg_gamma'].astype(bool))['vix'].mean().rename({False:'pos-gamma', True:'neg-gamma'}))"
        ),

        md(
            "## 3–4 · The nested regression — the load-bearing number\n\n"
            "`partial_over_vix` fits `y ~ vix` and `y ~ vix + neg_gamma` with Newey-West (HAC) errors "
            "(daily, persistent regimes → naive *t* overstates). It returns the raw gap, the "
            "**surviving** negative-gamma coefficient, its HAC *t*, the incremental R², and the share "
            "of the raw gap that survives. On the real-effect panel it recovers `β`; on the "
            "trenchcoat panel it collapses to zero."
        ),
        code(
            "rows = []\n"
            "for name, panel, truth in [('REAL', real, t_real), ('TRENCH', trench, t_tr)]:\n"
            "    for y in ('rv', 'de'):\n"
            "        p = decompose.partial_over_vix(panel, y)\n"
            "        rows.append({'panel': name, 'outcome': y, 'baked_beta': (truth.beta_vol if y=='rv' else truth.beta_de),\n"
            "                     'raw_gap': p['raw_gap'], 'raw_t': p['raw_t'],\n"
            "                     'surviving': p['surviving_coef'], 'surv_t': p['surviving_t'],\n"
            "                     'share_kept': p['survival_share'], 'dR2': p['delta_r2']})\n"
            "display(pd.DataFrame(rows).set_index(['panel','outcome']).round(4))"
        ),
        md(
            "Read the table: in **REAL**, `surviving` lands on `baked_beta` with |t| ≫ 2 — the genuine "
            "effect is recovered. In **TRENCH**, `raw_gap` is just as large (the confound), but "
            "`surviving` ≈ 0 with |t| < 2 and `dR2` ≈ 0 — the gap was entirely VIX. The raw column is "
            "*identical in character* across the two; only the control distinguishes signal from "
            "relabel. This is the exact apparatus pointed at the real chain."
        ),
        code(
            "# Why the raw gap is a trap: directional efficiency vs VIX, coloured by regime (REAL panel).\n"
            "neg = real['neg_gamma'].astype(bool)\n"
            "plt.scatter(real.loc[~neg,'vix'], real.loc[~neg,'de'], s=9, alpha=0.5, label='pos-gamma', color='#1f77b4')\n"
            "plt.scatter(real.loc[neg,'vix'], real.loc[neg,'de'], s=9, alpha=0.5, label='neg-gamma', color='#b22222')\n"
            "plt.xlabel('prior-close VIX'); plt.ylabel('directional efficiency')\n"
            "plt.title('Negative-gamma days sit at high VIX — the gap is mostly the x-axis'); plt.legend(); plt.show()"
        ),

        md(
            "## 5 · The verdict, with the numbers *(pre-registered)*\n\n"
            "**Signal `WEAK` (expected)** — the raw gap is real but we expect the VIX-controlled "
            "coefficient to retain little, especially for range vol (near-tautologically VIX); GEX "
            "earns `REAL` only if directional efficiency keeps a significant, material coefficient "
            "after VIX. **Tradability `MIRAGE` (expected)** — a surviving sign is a regime *bias*, "
            "not an entry; a few points of directional-efficiency tilt do not pay a round-trip "
            "options/index spread on a whole-day hold. The synthetic validates the machine; "
            "[`../docs/results.md`](../docs/results.md) (via `verify.py`, as-of "
            f"{DEFAULT_AS_OF}) fills the stamps on the real SPY chain."
        ),

        md(
            "## 6 · Could you trade it — the cost a regime bias can't pay\n\n"
            "Grant a surviving directional-efficiency effect of ~`β_de` on the day. That is a tilt in "
            "the *shape* of the session, not a signed entry — to express it you still need a trade, "
            "and you pay the spread (per the house rule, costs against the alpha, not the gross). A "
            "few points of trend-vs-chop probability, held all day on a near-symmetric return, is "
            "swamped by a round-trip SPY-options or index spread. There is no capacity question — SPX "
            "options are deep — because the edge, if any, dies on costs before size matters."
        ),

        md(
            "## 7 · Going further\n\n"
            "- **Walls & flip as levels** — `compute_gex` returns the call/put walls and a crude "
            "gamma flip; test whether intraday highs/lows cluster there vs a permutation null "
            "(Ni–Pearson–Poteshman pinning is the prior).\n"
            "- **Intraday character** — replace daily-OHLC directional efficiency with a "
            "minute-bar variance ratio / mean-reversion statistic, the pitch's literal claim.\n"
            "- **Dealer-sign robustness** — re-run under alternative positioning assumptions "
            "(e.g. add a 0DTE-only book, or flip the put convention) and see how much the sign "
            "series — and the verdict — moves. This is the single biggest modelling risk.\n"
            "- **Magnitude, not just sign** — does *bigger* |GEX| predict more suppression/"
            "amplification, in a continuous regression beyond the sign dummy?\n\n"
            "Engine: [`../../../quantlab/`](../../../quantlab/). Method: "
            "[`METHODOLOGY.md`](../../../METHODOLOGY.md). Real run: `examples/verify.py` → "
            "[`../docs/results.md`](../docs/results.md). Literature: "
            "[`../docs/references.md`](../docs/references.md)."
        ),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "02_for_the_quants.ipynb")


def _meta():
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }


def _write(nb, name):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print(f"wrote {name}  ({len(nb.cells)} cells)")


def main():
    build_curious()
    build_quants()


if __name__ == "__main__":
    main()
