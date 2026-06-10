"""Generate the two narrative notebooks for Study 34 (Aftershock) from source.

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

The executed path runs OFFLINE on the seeded synthetic stock panel where earnings surprises predict a
decaying post-event drift (the machinery) + a null where surprises are noise. This study's *real* tape
(genuine earnings dates + surprises + prices) has no free long-history source, so the real-tape verdict
is explicitly **PENDING / pre-registered** and quoted as such from ../docs/results.md — the synthetic
control is the committed, reproducible proof (cf. Study 27 Steamroller). Both notebooks walk the SAME
seven desk beats.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../../.."))
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5.2)
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
from aftershock import data, strategy, costs, extension

# Offline synthetic control: a stock panel where each earnings surprise carries a small, decaying
# post-event drift (the machinery) + a null where the same surprises are pure noise.
# The REAL PEAD verdict is PENDING an earnings-history fetch (see ../docs/results.md).
panel, events, truth = data.synthetic_pead(seed=34)               # the drift control
panel0, events0, _   = data.synthetic_pead(drift_strength=0.0, seed=34)  # the null
print(f"synthetic control: {truth.n_stocks} stocks x {truth.n_bars} days, {len(events)} events, "
      f"drift_strength {truth.drift_strength} (null=0)")
"""

# Synthetic-control numbers (seed 34, ../docs/results.md, fingerprint 3316fcb4614f)
R = dict(
    fp="3316fcb4614f", n="120", days="3024", ev="5760",
    gross_sh="3.73", gross_cagr="10.2", null_sh="0.40", net5="3.41", turn="0.068", be="57",
    c0="3.734", c2="3.604", c5="3.409", c10="3.083", c20="2.429",
    h5="0.10", h20="2.26", h40="3.24", h60="3.22", h90="3.28",
    h5g="1.10", h40g="3.62",
    car0="+0.0004", car20="+0.0059", car40="+0.0116", car69="+0.0153", carnull="+0.0032",
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Real-tape run?: Pre-reg](https://img.shields.io/badge/Real--tape_run%3F-Pre--reg-8b949e?style=flat-square)\n\n"
)

PENDING = (
    "> ⚠️ **Real-tape run pre-registered and PENDING.** A credible PEAD backtest needs *years* of "
    "reported-earnings dates + surprises per name; no free source supplies that here (yfinance gives only "
    "~6-8 quarters). So this notebook executes on the **synthetic control**, and the real measurement is "
    "**pre-registered** in [`../docs/results.md`](../docs/results.md) — one `examples/verify.py --fetch` "
    "away. This is the desk's pending-fetch pattern (cf. [Study 27](../../27-steamroller/))."
)


def md(t): return new_markdown_cell(t)
def code(t): return new_code_cell(t)


def build_curious():
    cells = [
        md(
            "# Aftershock 📈\n"
            "### A stock beats earnings — and then keeps drifting up for *weeks*. Real? Yes. Tradable? That's the catch.\n\n"
            + BADGES +
            "When a company reports a big earnings *surprise*, the price doesn't snap to its new value and "
            "stop — it keeps **drifting in the surprise's direction for weeks afterward**. That aftershock "
            "is *post-earnings-announcement drift* (PEAD), one of the oldest and sturdiest anomalies in all "
            "of finance (Ball & Brown 1968; Bernard & Thomas 1989). The trade writes itself: go long the "
            "names that just beat, short the names that just missed, and ride the drift. The catch — the "
            "whole lesson of this desk — is that the drift is *small* and hides in exactly the illiquid, "
            "small names that cost the most to trade.\n\n"
            "This is an idea from Kakushadze & Serur's *151 Trading Strategies* (strategy §3.2). We prove "
            "the engine on a synthetic stock panel where every surprise carries a *known* decaying drift "
            "(and a null where surprises are noise), run the long-beat / short-miss book, and reproduce the "
            "textbook drift-decay curve.\n\n"
            "> 📓 **Plain-language layer.** The cost wall, the break-even, the drift-decay curve and the "
            "holding-period sweep are in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            + PENDING + "\n>\n"
            "> ⚠️ **Not investment advice.** Charts below are generated by the code beside them. House style "
            "in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## The answer first 🎯\n\n"
            "| What we asked | The honest answer |\n"
            "|---|---|\n"
            "| Does price keep drifting after an earnings surprise? | 🟩 **Yes.** Real, decades-documented. "
            f"Our control recovers it (gross Sharpe **{R['gross_sh']}**, CAGR **{R['gross_cagr']}%**); the "
            f"null is flat (**{R['null_sh']}**). |\n"
            "| Could you trade it? | 🟨 **Barely.** The drift is small and lives in the *illiquid* names "
            "(Chordia et al. 2009) — real, but thin exactly where it's expensive to harvest. |\n"
            "| Did we measure it on the real tape? | ⚪ **Not yet.** No free source has the *years* of "
            "earnings surprises a credible PEAD needs — the real run is **pre-registered and pending**. |\n\n"
            "> Desk shorthand: **Signal `REAL` · Tradability `FRAGILE` · Real-tape run? `PRE-REG`** — a real "
            "premium, fragile to harvest, with the measurement honestly pending the data."
        ),

        md(
            "## 1 · The claim 📣\n\n"
            "Earnings momentum / PEAD (Kakushadze-Serur §3.2; lineage Ball-Brown 1968, Bernard-Thomas 1989):\n\n"
            "1. **Signal** — each name's most-recent earnings *surprise* (a standardised z-score, SUE), "
            "carried for ~40-60 trading days after the announcement.\n"
            "2. **Dollar-neutral** — long the positive surprises, short the negative ones; legs net to "
            "zero, gross exposure normalised to 1.\n"
            "3. **Roll on the calendar** — names enter on a surprise and leave when it ages out, so the "
            "book turns over slowly.\n\n"
            "The believer's case: investors *under-react* to earnings news — they don't fully price in what "
            "a surprise implies for future earnings — so the surprise keeps predicting return for weeks."
        ),
        code(
            "book = strategy.book_returns(panel, events, cost_bps=0.0)   # GROSS, on the synthetic control\n"
            "eq = (1+book).cumprod()\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(eq.index, eq.values, color='#2ea44f', lw=1.1)\n"
            "ax.set_title('Long-beat / short-miss on a panel that truly drifts — the machinery works')\n"
            "s = strategy.summary(book); print(f\"synthetic gross Sharpe {s['sharpe']:.2f}  (null ~0.40)\")"
        ),

        md(
            "## 2 · So what? 💰\n\n"
            "A real, market-neutral return stream tied to a hard fundamental event — earnings — is the "
            "dream of every fundamental-quant desk: it works in any market direction and rests on a clear "
            "behavioural mechanism. PEAD genuinely *is* real and has survived fifty years of out-of-sample "
            "scrutiny. So the question isn't whether the effect exists — it's whether *you* can keep any of "
            "it after paying to trade a book whose edge concentrates in the costliest names. That gap, "
            "between a real signal and a tradable one, is the whole point of this desk."
        ),

        md(
            "## 3 · How we'd know 🔬\n\n"
            "1. **Real?** Gross-of-cost Sharpe on the control (and flat on a null where surprises are noise).\n"
            "2. **What's the shape?** The drift-decay curve — does the surprise-signed cumulative return "
            "rise then flatten (under-reaction), as Bernard-Thomas found?\n"
            "3. **Tradable?** Turnover, break-even cost, and the holding-period sweep.\n\n"
            "**Fragile/mirage line:** on the real tape, a break-even cost inside the realistic equity "
            "round-trip band (~2-10 bp) — which the liquidity literature predicts, especially in the small "
            "names where PEAD is strongest."
        ),

        md("## 4 · The teardown 🔧\n\n### 4a · The machinery works where the drift is real (control vs null)"),
        code(
            "for label, (pp, ee) in [('drift panel (control)', (panel, events)), ('null (surprise=noise)', (panel0, events0))]:\n"
            "    s = strategy.summary(strategy.book_returns(pp, ee, cost_bps=0.0))\n"
            "    print(f\"{label:24} gross Sharpe {s['sharpe']:+6.2f}  turnover/day {strategy.turnover(pp, ee):.3f}\")"
        ),
        md(
            "### 4b · The aftershock itself — the drift-decay curve\n"
            "Line up every earnings event at day 0 and average the **surprise-signed** cumulative return by "
            "day-since-announcement. If under-reaction is real, it rises for weeks, then flattens — the "
            "Bernard-Thomas (1989) signature."
        ),
        code(
            "curve = extension.drift_decay_curve(panel, events, window=70)\n"
            "curve0 = extension.drift_decay_curve(panel0, events0, window=70)\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(curve.index, curve.values, color='#2ea44f', lw=1.4, label='drift control')\n"
            "ax.plot(curve0.index, curve0.values, color='#999', lw=1.1, ls='--', label='null')\n"
            "ax.axhline(0, color='#ccc', lw=.8)\n"
            "ax.set_xlabel('trading days since the earnings announcement'); ax.set_ylabel('mean surprise-signed cumulative return')\n"
            "ax.set_title('The aftershock: price keeps drifting in the surprise direction, then flattens'); ax.legend()\n"
            f"print(f\"control CAR: day0 {R['car0']}, day20 {R['car20']}, day40 {R['car40']}, day69 {R['car69']}  |  null day69 {R['carnull']}\")"
        ),
        md(
            "### 4c · The cost wall — turnover meets the spread\n"
            "The book rolls slowly (the earnings calendar, not a daily reshuffle), so on the control it "
            "clears costs comfortably. The real tape will be harsher — that's the pending measurement."
        ),
        code(
            "cs = costs.cost_sweep(panel, events)\n"
            "fig, ax = plt.subplots(); ax.axhline(0, color='#999', lw=.8)\n"
            "ax.plot(cs.index, cs['sharpe'], marker='o', color='#c0392b')\n"
            "ax.set_xlabel('cost (bp per unit traded)'); ax.set_ylabel('net Sharpe'); ax.set_title('Cost wall (synthetic control)')\n"
            "print(f\"synthetic turnover/day {strategy.turnover(panel, events):.3f}  break-even {costs.breakeven_cost_bps(panel, events):.0f} bp\")\n"
            "print('On the REAL tape this break-even will be FAR lower -- PEAD is small and lives in high-cost names.')"
        ),

        md("## 5 · The verdict 🧾\n\n"
           f"- **Signal `REAL`** — gross Sharpe {R['gross_sh']} on the control, null flat ({R['null_sh']}); "
           "the drift-decay curve has the textbook rise-then-flatten shape; fifty years of evidence agree.\n"
           "- **Tradability `FRAGILE`** — the drift is small and concentrates in the illiquid, small, "
           "high-cost names (Chordia et al. 2009); the tradable, liquid slice carries almost none.\n"
           "- **Real-tape run? `PRE-REG`** — no free source has the years of earnings surprises required; "
           "the measurement is pre-registered, one `--fetch` away.\n\n"
           "> **Signal `REAL` · Tradability `FRAGILE` · Real-tape run? `PRE-REG`** — real, thin where you "
           "can harvest it, honestly pending the data."),

        md("## 6 · Could you trade it? 💸\n\n"
           "- **Real, but small.** PEAD's drift is a fraction of a percent that plays out over a quarter — "
           "you must hold for weeks to bank it.\n"
           "- **The cruel irony** (Chordia, Goyal, Sadka, Sadka & Shivakumar 2009): PEAD is *strongest in "
           "illiquid stocks* and *weakest in liquid ones* — so the only slice you can trade cheaply at "
           "scale is the slice with almost no drift.\n"
           "- **And it's shrunk.** Like most published anomalies, PEAD has attenuated since it was "
           "documented and arbitraged (McLean & Pontiff 2016). Real and durable, but smaller than its heyday."),

        md(
            "## 7 · Going further 🚪\n\n"
            "### Worked complement — the drift-decay curve & holding-period sweep ([`../docs/extension.md`](../docs/extension.md))\n"
            "How long does the aftershock last, and does the drift outpay the roll cost?\n\n"
            f"- The drift-decay curve rises for ~40-60 days then flattens — so a book must hold for weeks.\n"
            f"- The holding-period sweep peaks around a 40-60 day hold (net Sharpe **{R['h40']}** on the "
            f"control), collapses if you hold too short (5 days → **{R['h5']}**, all turnover, no drift), "
            "and plateaus once the drift has flattened.\n\n"
            "### Other forks\n"
            "- **Liquidity / size tiering** — quantify how much of PEAD lives only in un-tradable small names.\n"
            "- **SUE bucketing** — trade only the extreme-surprise deciles (Bernard-Thomas).\n"
            "- **Earnings- vs price-momentum** — how much drift does the *earnings* surprise add over price momentum (Chordia-Shivakumar 2006)?\n\n"
            "**And the big one:** wire a real earnings-history feed and run `examples/verify.py --fetch` to "
            "fill in [`../docs/results.md`](../docs/results.md). PRs welcome."
        ),
    ]
    return new_notebook(cells=cells, metadata=_meta())


def build_quants():
    cells = [
        md(
            "# Aftershock — a quantitative teardown 🔬\n"
            "### Gross-vs-net Sharpe on control vs null · the drift-decay curve · the cost wall & break-even · the holding-period sweep\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same seven "
            "beats, every claim with its number.* The steelman is §3.2, earnings momentum / PEAD: prices "
            "under-react to the earnings surprise, so it keeps predicting return for weeks (Ball-Brown 1968; "
            "Bernard-Thomas 1989). We confirm the machinery is `REAL` on a synthetic control (gross Sharpe "
            "3.73, null 0.40), reproduce the Bernard-Thomas drift-decay curve, and argue Tradability is "
            "`FRAGILE` from the liquidity literature — with the real-tape run `PRE-REG`.\n\n"
            + PENDING + "\n>\n"
            "> ⚠️ **Not investment advice.** The core executes on the synthetic control; sources in "
            "[`../docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        md(
            "## Beat 0 · Verdict\n\n"
            "| Axis | Stamp | Why |\n"
            "|---|---|---|\n"
            f"| **Signal** — drift after a surprise? | 🟢 `REAL` | Control gross Sharpe **{R['gross_sh']}** "
            f"(CAGR {R['gross_cagr']}%), null flat **{R['null_sh']}**; drift-decay curve rises then flattens; "
            "50 yrs of evidence (Ball-Brown; Bernard-Thomas). |\n"
            f"| **Tradability** | 🟡 `FRAGILE` | Drift is small and concentrates in *illiquid* names "
            "(Chordia et al. 2009); the liquid, scalable slice carries almost none; effect has shrunk. |\n"
            "| **Real-tape run?** | ⚪ `Pre-reg` | No free source has the years of earnings surprises a "
            "credible PEAD needs; apparatus + protocol + mirage-line fixed, one `--fetch` away. |\n\n"
            "> **In one sentence:** post-earnings drift is a real, durable premium our control and the "
            "literature confirm — but it is small and lives in the costliest names, so it is `FRAGILE`, and "
            "the real-tape measurement is pre-registered and pending an earnings-history source.\n\n"
            "*(This notebook executes on the synthetic control; the real numbers are pre-registered in "
            "[`../docs/results.md`](../docs/results.md).)*"
        ),

        md(
            "## Beat 1 · The claim, precisely\n\n"
            "For each name $i$ with most-recent earnings surprise $s_i$ (a standardised z-score / SUE) within "
            "the last $H$ trading days, weight $w_{i,t} \\propto \\big(s_{i,t}-\\bar s_t\\big)$, scaled so "
            "$\\sum_i w_{i,t}=0$ (dollar-neutral) and $\\sum_i|w_{i,t}|=1$, then lagged one day. Claim: "
            "$\\sum_i w_{i,t-1} r_{i,t}>0$ — the surprise predicts post-event return. Null: the same "
            "surprises are drawn but carry no drift, so there is nothing to ride."
        ),
        code(
            "for label, (pp, ee) in [('drift control', (panel, events)), ('null', (panel0, events0))]:\n"
            "    g = strategy.summary(strategy.book_returns(pp, ee, cost_bps=0.0))\n"
            "    print(f\"{label:14} gross Sharpe {g['sharpe']:+6.2f}  turnover/day {strategy.turnover(pp, ee):.3f}\")"
        ),
        md(
            "> 💡 **In plain words.** The control's stocks genuinely drift after a surprise, so the book "
            f"makes money gross (**{R['gross_sh']}**); the null has none (**{R['null_sh']}**). The apparatus "
            "sees PEAD when it's there — so once the real surprises are wired, the result will be about the "
            "*market*, not the machine."
        ),

        md(
            "## Beat 2 · So what?\n\n"
            "Bernard-Thomas (1990) tie the drift to **under-reaction**: investors don't fully price in the "
            "autocorrelation in earnings news, so the surprise leaks into price for a quarter. That makes "
            "PEAD a clean behavioural premium — but the pre-registered prediction, from Chordia et al. "
            "(2009), is that it is *largest where liquidity is scarcest* (small, illiquid names), so it "
            "should be biggest exactly where it costs the most to harvest. Beats 4-6 test that tension."
        ),

        md(
            "## Beat 3 · Pre-registered protocol\n\n"
            "1. **Real?** `book_returns(cost_bps=0)` on control vs null (gross).\n"
            "2. **Shape?** `extension.drift_decay_curve` — surprise-signed CAR by day since event.\n"
            "3. **Tradable?** `strategy.turnover`, `costs.breakeven_cost_bps`, `costs.cost_sweep`.\n"
            "4. **Horizon?** `extension.holding_period_sweep` — does the drift persist long enough to pay?\n\n"
            "**Mirage line (real tape):** break-even cost inside the ~2-10 bp realistic equity round-trip "
            "band — pre-registered before the fetch."
        ),

        md("## Beat 4 · The teardown\n\n### 4a · Real gross on the control, flat on the null (the REAL call)"),
        code(
            "print('Synthetic control (../docs/results.md, fp 3316fcb4614f):')\n"
            "g = strategy.summary(strategy.book_returns(panel, events, cost_bps=0.0))\n"
            "n = strategy.summary(strategy.book_returns(panel0, events0, cost_bps=0.0))\n"
            "print(f'  drift control gross Sharpe {g[\"sharpe\"]:.2f}  CAGR {g[\"cagr\"]*100:.1f}%')\n"
            "print(f'  null          gross Sharpe {n[\"sharpe\"]:.2f}')\n"
            "print('  -> a REAL, recoverable effect; the null proves the diagnostic measures PEAD, not itself.')"
        ),
        md(
            "> 💡 **In plain words.** A market-neutral gross Sharpe of 3.73 on the control vs 0.40 on the "
            "null isn't luck — it's the apparatus correctly seeing the drift that's baked in. On the real "
            "tape the magnitude will be far smaller; the *machine* is validated here."
        ),

        md("### 4b · The drift-decay curve (the under-reaction signature)"),
        code(
            "curve = extension.drift_decay_curve(panel, events, window=70)\n"
            "curve0 = extension.drift_decay_curve(panel0, events0, window=70)\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(curve.index, curve.values, color='#2ea44f', lw=1.4, label='drift control')\n"
            "ax.plot(curve0.index, curve0.values, color='#999', lw=1.1, ls='--', label='null')\n"
            "ax.axhline(0, color='#ccc', lw=.8); ax.legend()\n"
            "ax.set_xlabel('trading days since announcement'); ax.set_ylabel('mean surprise-signed CAR'); ax.set_title('Drift-decay curve (Bernard-Thomas 1989, Fig. 1)')\n"
            f"print(f\"control: day0 {R['car0']}, day20 {R['car20']}, day40 {R['car40']}, day69 {R['car69']}  |  null day69 {R['carnull']}\")"
        ),
        md(
            "> 💡 **In plain words.** The curve climbs for ~40-60 days and then goes flat: the market keeps "
            "re-pricing the surprise for weeks, then finishes. That *shape* is the anomaly — and it tells "
            "you a book must hold for weeks, which sets the turnover and therefore the cost."
        ),

        md("### 4c · The cost wall and the break-even (the FRAGILE call)"),
        code(
            "print('Synthetic cost sweep (net Sharpe):')\n"
            f"print('  0bp {R['c0']}   2bp {R['c2']}   5bp {R['c5']}   10bp {R['c10']}   20bp {R['c20']}')\n"
            "cs = costs.cost_sweep(panel, events)\n"
            "fig, ax = plt.subplots(); ax.axhline(0, color='#999', lw=.8)\n"
            "ax.plot(cs.index, cs['sharpe'], marker='o', color='#c0392b')\n"
            "ax.set_xlabel('cost (bp/unit)'); ax.set_ylabel('net Sharpe'); ax.set_title('Cost wall (synthetic control)')\n"
            "print(f'\\nsynthetic turnover/day {strategy.turnover(panel, events):.3f}  break-even {costs.breakeven_cost_bps(panel, events):.0f} bp')\n"
            "print('On the REAL tape the break-even is expected INSIDE the 2-10 bp band -- the pre-registered mirage-line.')"
        ),

        md("## Beat 5 · The verdict\n\n"
           f"- **`REAL`** (4a, 4b): control gross Sharpe {R['gross_sh']} vs null {R['null_sh']}; drift-decay "
           "curve rises then flattens.\n"
           "- **`FRAGILE`** (4c, Beat 6): drift is small and concentrates in illiquid, high-cost names; the "
           "scalable slice carries little.\n"
           "- **Real-tape run? `PRE-REG`** (Beat 3): protocol + mirage-line fixed; pending an earnings feed.\n\n"
           "> **Signal `REAL` · Tradability `FRAGILE` · Real-tape run? `PRE-REG`.**"),

        md("## Beat 6 · Could you trade it?\n\n"
           "- **Real, but small and slow** — a fraction of a percent of drift over a quarter; you hold for weeks.\n"
           "- **The PEAD paradox** (Chordia et al. 2009): the drift is largest in illiquid names and "
           "smallest in liquid ones — the only cheap-to-trade slice has almost no premium.\n"
           "- **Decay** (McLean & Pontiff 2016): published, arbitraged, attenuated. Real and durable, but "
           "smaller than its 1980s heyday — and the tradable residual is what the pending real run will measure."),

        md(
            "## Beat 7 · Going further\n\n"
            "### 7a · Worked complement — the holding-period sweep ([`../docs/extension.md`](../docs/extension.md))\n"
            "Hold each name's surprise $h$ days; does the drift outpay the roll cost? Synthetic here; real "
            "tape in [`../docs/extension.md`](../docs/extension.md) after `--fetch`."
        ),
        code(
            "hp = extension.holding_period_sweep(panel, events, holds=[5,20,40,60,90], cost_bps=5.0)\n"
            "display(hp.round(3))\n"
            "fig, ax = plt.subplots(); ax.axhline(0, color='#999', lw=.8)\n"
            "ax.plot(hp.index, hp['gross_sharpe'], marker='o', label='gross', color='#2ea44f')\n"
            "ax.plot(hp.index, hp['net_sharpe'], marker='o', label='net @5bp', color='#c0392b')\n"
            "ax.set_xlabel('holding period (days)'); ax.set_ylabel('Sharpe'); ax.legend(); ax.set_title('Holding-period sweep (synthetic)')\n"
            f"print('net Sharpe by hold: 5d {R['h5']}, 20d {R['h20']}, 40d {R['h40']}, 60d {R['h60']}, 90d {R['h90']} -- peaks around a quarter.')"
        ),
        md(
            "> 💡 **In plain words.** Hold too short (5 days) and you pay all the turnover but miss the drift "
            "(net ~0.10); hold ~40-60 days and you bank most of the cumulative drift at low turnover (net "
            "~3.2 on the control); beyond that the drift has flattened, so it plateaus. The sweet spot is "
            "exactly the window over which the decay curve was still rising."
        ),
        md(
            "### 7b · The honest pending status & forks\n"
            "The real measurement is **pre-registered, not done**: the apparatus, the protocol and the "
            "mirage-line are fixed above, and the real `--fetch` run will report the PEAD book's gross/net "
            "Sharpe, HAC *t*, break-even cost, cost & holding sweeps and the real drift-decay curve — each "
            "fingerprinted and as-of pinned in [`../docs/results.md`](../docs/results.md). The expected "
            "shape from the literature is a real but modest gross edge eroded to a thin residual by the "
            "costs and illiquidity of its strongest names — the `REAL` / `FRAGILE` verdict the control and "
            "the literature already earn.\n\n"
            "### 7c · Other forks\n"
            "- **Liquidity / size tiering** — model each tier's honest spread; how much PEAD survives net?\n"
            "- **SUE bucketing** — trade only the extreme-surprise deciles (Bernard-Thomas).\n"
            "- **Earnings- vs price-momentum** — incremental drift from the *earnings* surprise (Chordia-Shivakumar 2006).\n\n"
            "PRs welcome — wire an earnings-history feed and complete the real run, or try a liquidity-tiered book."
        ),
    ]
    return new_notebook(cells=cells, metadata=_meta())


def _meta():
    return {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}


def main():
    nbf.write(build_curious(), os.path.join(HERE, "01_for_the_curious.ipynb"))
    nbf.write(build_quants(), os.path.join(HERE, "02_for_the_quants.ipynb"))
    print("wrote 01_for_the_curious.ipynb and 02_for_the_quants.ipynb")


if __name__ == "__main__":
    main()
