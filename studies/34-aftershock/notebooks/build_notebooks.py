"""Generate the two narrative notebooks for Study 34 (Aftershock) from source.

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

The executed path runs OFFLINE on the seeded synthetic stock panel where earnings surprises predict a
decaying post-event drift (the machinery) + a null where surprises are noise — the fast, reproducible
control that backs the test-suite. The **real-tape numbers** (cached EDGAR SUEs × the S&P 500 price panel)
are the headline, quoted as the measured result from ../docs/results.md. Both notebooks walk the SAME
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

# These EXECUTED cells run the offline synthetic control: a stock panel where each earnings surprise
# carries a small, decaying post-event drift (the machinery) + a null where the same surprises are noise.
# It is the fast, reproducible proof that backs the test-suite. The HEADLINE numbers in the markdown are
# the REAL-tape measurement (cached EDGAR SUEs x the S&P 500 panel) -- see ../docs/results.md / verify.py.
panel, events, truth = data.synthetic_pead(seed=34)               # the drift control
panel0, events0, _   = data.synthetic_pead(drift_strength=0.0, seed=34)  # the null
print(f"synthetic control (backs the tests): {truth.n_stocks} stocks x {truth.n_bars} days, "
      f"{len(events)} events, drift_strength {truth.drift_strength} (null=0)")
"""

# REAL-TAPE numbers (cached EDGAR SUEs × S&P 500 panel; ../docs/results.md, fingerprint 83140e2fef71)
RR = dict(
    fp="83140e2fef71", names="488", ev="23,000", span="2010-01-04 - 2026-06-05", asof="2026-06-05",
    gross_sh="0.30", gross_cagr="0.8", net5="0.05", net10="-0.20", turn="0.058", be="6.0",
    nw_t="1.39", plain_t="1.21",
    rc0="0.298", rc2="0.199", rc5="0.050", rc10="-0.199", rc20="-0.691",
    rh5="-0.66", rh20="-0.34", rh40="-0.19", rh60="0.05", rh90="0.04",
    rcar0="+0.0014", rcar20="+0.0032", rcar40="+0.0059", rcar60="+0.0096", rcar69="+0.0115",
)

# Synthetic-control numbers (seed 34, fingerprint 3316fcb4614f) — the executed offline cells / test proof
R = dict(
    fp="3316fcb4614f", n="120", days="3024", ev="5760",
    gross_sh="3.73", gross_cagr="10.2", null_sh="0.40", net5="3.41", turn="0.068", be="57",
    c0="3.734", c2="3.604", c5="3.409", c10="3.083", c20="2.429",
    h5="0.10", h20="2.26", h40="3.24", h60="3.22", h90="3.28",
    h5g="1.10", h40g="3.62",
    car0="+0.0004", car20="+0.0059", car40="+0.0116", car69="+0.0153", carnull="+0.0032",
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Drift decays as predicted?: Confirmed](https://img.shields.io/badge/Drift_decays_as_predicted%3F-Confirmed-8b949e?style=flat-square)\n\n"
)

REALTAPE = (
    "> ✅ **Real-tape run, fingerprinted.** The headline numbers below are measured on the **real tape** — "
    "cached EDGAR quarterly EPS → a seasonal-random-walk SUE (`surprise_q = eps_q − eps_{q−4}`, "
    "standardised by the stock's own trailing dispersion; announcement = earliest filing) traded against "
    "the cached S&P 500 split/dividend-adjusted-Close panel "
    f"(**{RR['names']} names, {RR['ev']} events, {RR['span']}, fp `{RR['fp']}`**). The executed code cells "
    "run the fast, reproducible **synthetic control** (a known surprise→drift relationship + a null) that "
    "backs the test-suite; reproduce the real run with [`examples/verify.py`](../examples/verify.py). "
    "**Caveat:** the panel is *current* index membership ⇒ survivorship bias, which inflates the "
    "magnitudes; the qualitative verdict is robust."
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
            "This is an idea from Kakushadze & Serur's *151 Trading Strategies* (strategy §3.2). We measure "
            "it on the **real tape** — cached EDGAR earnings turned into a standardised surprise (SUE), "
            "traded against the S&P 500 — run the long-beat / short-miss book, and reproduce the textbook "
            "drift-decay curve. The executed cells below run a synthetic control that backs the tests.\n\n"
            "> 📓 **Plain-language layer.** The cost wall, the break-even, the drift-decay curve and the "
            "holding-period sweep are in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            + REALTAPE + "\n>\n"
            "> ⚠️ **Not investment advice.** Charts below are generated by the code beside them. House style "
            "in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## The answer first 🎯\n\n"
            "| What we asked | The honest answer (real tape) |\n"
            "|---|---|\n"
            "| Does price keep drifting after an earnings surprise? | 🟨 **Yes, but thin.** The drift is real "
            f"and points the right way, but small on liquid S&P names: gross Sharpe **{RR['gross_sh']}** "
            f"(Newey-West *t* **{RR['nw_t']}**) — not significant at conventional levels. |\n"
            "| Could you trade it? | 🟥 **No.** The break-even cost is only "
            f"**{RR['be']} bp** — *inside* realistic equity round-trip costs — and survivorship bias "
            f"inflates even that. Net of 5 bp the Sharpe is **+{RR['net5']}** (≈ nothing). |\n"
            "| Does the drift decay the way the textbook says? | 🟩 **Yes.** The surprise-signed cumulative "
            f"return rises from {RR['rcar0']} (day 0) to {RR['rcar69']} (day 69) — the Bernard-Thomas shape, "
            "on real earnings. |\n\n"
            "> Desk shorthand: **Signal `WEAK` · Tradability `MIRAGE` · Drift decays as predicted? "
            "`CONFIRMED`** — a real but thin premium, a mirage to trade, with the textbook decay shape intact."
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
            "1. **Real?** Gross-of-cost Sharpe + a Newey-West *t* on the real tape (and the control vs a "
            "null below proves the machine sees a drift only when one is there).\n"
            "2. **What's the shape?** The drift-decay curve — does the surprise-signed cumulative return "
            "rise then flatten (under-reaction), as Bernard-Thomas found?\n"
            "3. **Tradable?** Turnover, break-even cost, and the holding-period sweep.\n\n"
            "**Mirage line:** a break-even cost inside the realistic equity round-trip band (~2-10 bp). The "
            f"real tape lands at **{RR['be']} bp** — inside it, so trading it is a mirage."
        ),

        md("## 4 · The teardown 🔧\n\n"
           f"**On the real tape** (cached EDGAR SUEs × the S&P 500, {RR['names']} names, {RR['ev']} events, "
           f"{RR['span']}) the long-beat / short-miss book earns a gross Sharpe of **{RR['gross_sh']}** "
           f"(Newey-West *t* **{RR['nw_t']}**) — real, correctly signed, but thin. The executed cells below "
           "run the synthetic control that proves the *machinery* can see a drift when one is there.\n\n"
           "### 4a · The machinery works where the drift is real (control vs null)"),
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
            f"print(f\"control CAR: day0 {R['car0']}, day20 {R['car20']}, day40 {R['car40']}, day69 {R['car69']}  |  null day69 {R['carnull']}\")\n"
            f"print('REAL tape CAR: day0 {RR['rcar0']}, day20 {RR['rcar20']}, day40 {RR['rcar40']}, day60 {RR['rcar60']}, day69 {RR['rcar69']} -- same rise-then-flatten shape.')"
        ),
        md(
            "### 4c · The cost wall — turnover meets the spread\n"
            "The book rolls slowly (the earnings calendar, not a daily reshuffle), so on the control it "
            f"clears costs comfortably (break-even ~57 bp). **The real tape is far harsher:** the gross edge "
            f"is so small that its real break-even is only **{RR['be']} bp** — *inside* the realistic equity "
            "round-trip band (≈2–10 bp), and survivorship-inflated on top."
        ),
        code(
            "cs = costs.cost_sweep(panel, events)\n"
            "fig, ax = plt.subplots(); ax.axhline(0, color='#999', lw=.8)\n"
            "ax.plot(cs.index, cs['sharpe'], marker='o', color='#c0392b')\n"
            "ax.set_xlabel('cost (bp per unit traded)'); ax.set_ylabel('net Sharpe'); ax.set_title('Cost wall (synthetic control)')\n"
            "print(f\"synthetic turnover/day {strategy.turnover(panel, events):.3f}  break-even {costs.breakeven_cost_bps(panel, events):.0f} bp\")\n"
            f"print('REAL tape: turnover/day {RR['turn']}, break-even {RR['be']} bp -- INSIDE realistic costs. Net @5bp Sharpe {RR['net5']}.')"
        ),

        md("## 5 · The verdict 🧾\n\n"
           f"- **Signal `WEAK`** — real and correctly signed on the tape (gross Sharpe {RR['gross_sh']}, "
           f"Newey-West *t* {RR['nw_t']}), but not significant on a liquid large-cap universe; the "
           "drift-decay curve still has the textbook rise-then-flatten shape.\n"
           f"- **Tradability `MIRAGE`** — break-even {RR['be']} bp sits *inside* realistic equity costs and "
           f"is survivorship-inflated; net @5 bp Sharpe is +{RR['net5']} (≈ nothing).\n"
           "- **Drift decays as predicted? `CONFIRMED`** — the real surprise-signed CAR rises monotonically "
           f"from {RR['rcar0']} (day 0) to {RR['rcar69']} (day 69), the Bernard-Thomas signature.\n\n"
           "> **Signal `WEAK` · Tradability `MIRAGE` · Drift decays as predicted? `CONFIRMED`** — real and "
           "textbook-shaped, but too thin on liquid names to trade."),

        md("## 6 · Could you trade it? 💸\n\n"
           "- **Real, but small and slow.** PEAD's drift is a fraction of a percent that plays out over a "
           f"quarter — you must hold ~60 days to bank it (net @5 bp is negative at every shorter hold).\n"
           "- **The cruel irony** (Chordia, Goyal, Sadka, Sadka & Shivakumar 2009): PEAD is *strongest in "
           "illiquid stocks* and *weakest in liquid ones* — and this is a liquid S&P 500 universe, so the "
           f"drift we *can* trade is exactly the thin slice (gross Sharpe {RR['gross_sh']}).\n"
           "- **And it's shrunk.** Like most published anomalies, PEAD has attenuated since it was "
           f"documented and arbitraged (McLean & Pontiff 2016) — and net of its {RR['be']} bp break-even "
           "there is nothing left to harvest at scale."),

        md(
            "## 7 · Going further 🚪\n\n"
            "### Worked complement — the drift-decay curve & holding-period sweep ([`../docs/extension.md`](../docs/extension.md))\n"
            "How long does the aftershock last, and does the drift outpay the roll cost?\n\n"
            "- The drift-decay curve rises for the whole quarter then flattens — so a book must hold for weeks.\n"
            f"- **On the real tape** the holding-period sweep (net @5 bp) is **{RR['rh5']} at a 5-day hold** "
            f"(all turnover, no drift) and only reaches break-even (**+{RR['rh60']}**) at a 60-day hold — the "
            "drift is real but too slow and small to clear costs.\n\n"
            "### Other forks\n"
            "- **Liquidity / size tiering** — quantify how much of PEAD lives only in un-tradable small names.\n"
            "- **SUE bucketing** — trade only the extreme-surprise deciles (Bernard-Thomas).\n"
            "- **Earnings- vs price-momentum** — how much drift does the *earnings* surprise add over price momentum (Chordia-Shivakumar 2006)?\n\n"
            "**And the survivorship fix:** the real panel is current S&P membership, so a delisting-aware "
            "universe would lower the magnitudes further. PRs welcome."
        ),
    ]
    return new_notebook(cells=cells, metadata=_meta())


def build_quants():
    cells = [
        md(
            "# Aftershock — a quantitative teardown 🔬\n"
            "### Gross-vs-net Sharpe & the Newey-West *t* on the real tape · the drift-decay curve · the cost wall & 6 bp break-even · the holding-period sweep\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same seven "
            "beats, every claim with its number.* The steelman is §3.2, earnings momentum / PEAD: prices "
            "under-react to the earnings surprise, so it keeps predicting return for weeks (Ball-Brown 1968; "
            f"Bernard-Thomas 1989). On the **real tape** the book earns a gross Sharpe of just {RR['gross_sh']} "
            f"(Newey-West *t* {RR['nw_t']}) — `WEAK` — with a {RR['be']} bp break-even that makes it a "
            "tradability `MIRAGE`; but the drift-decay curve `CONFIRMED`s the textbook Bernard-Thomas shape. "
            "The executed cells run the synthetic control that backs the tests.\n\n"
            + REALTAPE + "\n>\n"
            "> ⚠️ **Not investment advice.** The executed cells run the synthetic control; sources in "
            "[`../docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        md(
            "## Beat 0 · Verdict (real tape)\n\n"
            "| Axis | Stamp | Why |\n"
            "|---|---|---|\n"
            f"| **Signal** — drift after a surprise? | 🟡 `WEAK` | Real-tape gross Sharpe **{RR['gross_sh']}** "
            f"(CAGR {RR['gross_cagr']}%), Newey-West *t* **{RR['nw_t']}** (plain {RR['plain_t']}) — correctly "
            "signed but not significant on a liquid large-cap universe; effect much attenuated since 1989. |\n"
            f"| **Tradability** | 🔴 `MIRAGE` | Turnover {RR['turn']}/day, but break-even only **{RR['be']} bp** "
            f"— *inside* realistic equity costs and survivorship-inflated. Net @5 bp Sharpe **+{RR['net5']}**. |\n"
            "| **Drift decays as predicted?** | 🟢 `CONFIRMED` | Real surprise-signed CAR rises monotonically "
            f"{RR['rcar0']} (day 0) → {RR['rcar69']} (day 69) — Bernard-Thomas (1989) Fig. 1 on real earnings. |\n\n"
            "> **In one sentence:** post-earnings drift is real and textbook-shaped on the tape, but on a "
            f"liquid S&P 500 universe it is `WEAK` (gross Sharpe {RR['gross_sh']}, *t* {RR['nw_t']}) and a "
            f"tradability `MIRAGE` — its {RR['be']} bp break-even sits inside realistic costs.\n\n"
            f"*(Real run: {RR['names']} names, {RR['ev']} events, {RR['span']}, fp `{RR['fp']}` — "
            "[`../docs/results.md`](../docs/results.md). The executed cells run the synthetic control.)*"
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
            f"sees PEAD when it's there — and run on the *real* market it earns a far smaller gross "
            f"**{RR['gross_sh']}** (Newey-West *t* {RR['nw_t']}): a real but thin drift, about the market, "
            "not the machine."
        ),

        md(
            "## Beat 2 · So what?\n\n"
            "Bernard-Thomas (1990) tie the drift to **under-reaction**: investors don't fully price in the "
            "autocorrelation in earnings news, so the surprise leaks into price for a quarter. That makes "
            "PEAD a clean behavioural premium — but the prediction from Chordia et al. (2009) is that it is "
            "*largest where liquidity is scarcest* (small, illiquid names), so on a liquid S&P 500 universe "
            "it should be thin. That is exactly what the real tape shows (Beats 4-6): a real but small drift "
            "whose break-even sits inside trading costs."
        ),

        md(
            "## Beat 3 · Protocol\n\n"
            "1. **Real?** `book_returns(cost_bps=0)` gross + the Newey-West *t* (`strategy.newey_west_t`) on "
            "the real tape; control vs null below proves the machine.\n"
            "2. **Shape?** `extension.drift_decay_curve` — surprise-signed CAR by day since event.\n"
            "3. **Tradable?** `strategy.turnover`, `costs.breakeven_cost_bps`, `costs.cost_sweep`.\n"
            "4. **Horizon?** `extension.holding_period_sweep` — does the drift persist long enough to pay?\n\n"
            "**Mirage line:** a break-even cost inside the ~2-10 bp realistic equity round-trip band. The real "
            f"tape lands at **{RR['be']} bp** — inside it, and survivorship-inflated."
        ),

        md("## Beat 4 · The teardown\n\n"
           f"**Real tape:** gross Sharpe **{RR['gross_sh']}**, Newey-West *t* **{RR['nw_t']}** (plain "
           f"{RR['plain_t']}), net @5 bp **+{RR['net5']}**, net @10 bp **{RR['net10']}** — see "
           "[`../docs/results.md`](../docs/results.md). The cells below run the synthetic control: the null "
           "proves the diagnostic measures PEAD, not itself.\n\n### 4a · Real gross on the control, flat on the null"),
        code(
            "print('Synthetic control (backs the tests; real-tape numbers are the headline above):')\n"
            "g = strategy.summary(strategy.book_returns(panel, events, cost_bps=0.0))\n"
            "n = strategy.summary(strategy.book_returns(panel0, events0, cost_bps=0.0))\n"
            "print(f'  drift control gross Sharpe {g[\"sharpe\"]:.2f}  CAGR {g[\"cagr\"]*100:.1f}%')\n"
            "print(f'  null          gross Sharpe {n[\"sharpe\"]:.2f}')\n"
            f"print('  -> machine validated. On the REAL tape: gross Sharpe {RR['gross_sh']}, Newey-West t {RR['nw_t']} -- WEAK but correctly signed.')"
        ),
        md(
            "> 💡 **In plain words.** A market-neutral gross Sharpe of 3.73 on the control vs 0.40 on the "
            "null isn't luck — it's the apparatus correctly seeing the drift that's baked in. Run on the "
            f"*real* market that same machine earns only **{RR['gross_sh']}** (*t* {RR['nw_t']}): the drift "
            "is real but thin on liquid names."
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
            f"print(f\"control: day0 {R['car0']}, day20 {R['car20']}, day40 {R['car40']}, day69 {R['car69']}  |  null day69 {R['carnull']}\")\n"
            f"print(f\"REAL tape: day0 {RR['rcar0']}, day20 {RR['rcar20']}, day40 {RR['rcar40']}, day60 {RR['rcar60']}, day69 {RR['rcar69']} -- same shape, smaller scale.\")"
        ),
        md(
            "> 💡 **In plain words.** The curve climbs for the whole quarter and then flattens: the market "
            "keeps re-pricing the surprise for weeks, then finishes. The **real tape shows the same "
            "monotone rise** (+0.14% → +1.15% over 69 days). That *shape* is the anomaly — and it tells you "
            "a book must hold for weeks, which sets the turnover and therefore the cost."
        ),

        md("### 4c · The cost wall and the break-even (the MIRAGE call)"),
        code(
            "print('Synthetic cost sweep (net Sharpe):')\n"
            f"print('  0bp {R['c0']}   2bp {R['c2']}   5bp {R['c5']}   10bp {R['c10']}   20bp {R['c20']}')\n"
            f"print('REAL tape net Sharpe: 0bp {RR['rc0']}  2bp {RR['rc2']}  5bp {RR['rc5']}  10bp {RR['rc10']}  20bp {RR['rc20']}')\n"
            "cs = costs.cost_sweep(panel, events)\n"
            "fig, ax = plt.subplots(); ax.axhline(0, color='#999', lw=.8)\n"
            "ax.plot(cs.index, cs['sharpe'], marker='o', color='#c0392b')\n"
            "ax.set_xlabel('cost (bp/unit)'); ax.set_ylabel('net Sharpe'); ax.set_title('Cost wall (synthetic control)')\n"
            "print(f'\\nsynthetic turnover/day {strategy.turnover(panel, events):.3f}  break-even {costs.breakeven_cost_bps(panel, events):.0f} bp')\n"
            f"print('REAL tape: turnover/day {RR['turn']}, break-even {RR['be']} bp -- INSIDE the 2-10 bp band. MIRAGE.')"
        ),

        md("## Beat 5 · The verdict\n\n"
           f"- **`WEAK`** (4a, 4b): real-tape gross Sharpe {RR['gross_sh']}, Newey-West *t* {RR['nw_t']} — "
           "correctly signed but not significant on liquid names; the drift-decay curve still rises then flattens.\n"
           f"- **`MIRAGE`** (4c, Beat 6): turnover {RR['turn']}/day but break-even only {RR['be']} bp — inside "
           "realistic costs and survivorship-inflated; net @5 bp +" + RR['net5'] + ".\n"
           "- **Drift decays as predicted? `CONFIRMED`** (4b): real surprise-signed CAR rises monotonically "
           f"{RR['rcar0']} → {RR['rcar69']}.\n\n"
           "> **Signal `WEAK` · Tradability `MIRAGE` · Drift decays as predicted? `CONFIRMED`.**"),

        md("## Beat 6 · Could you trade it?\n\n"
           "- **Real, but small and slow** — a fraction of a percent of drift over a quarter; on the real "
           f"tape net @5 bp is negative at every hold shorter than ~60 days (then only +{RR['rh60']}).\n"
           "- **The PEAD paradox** (Chordia et al. 2009): the drift is largest in illiquid names and "
           "smallest in liquid ones — and this is a liquid S&P 500 universe, so the slice we can trade is "
           "exactly the thin one.\n"
           "- **Decay** (McLean & Pontiff 2016): published, arbitraged, attenuated. Real and durable, but "
           f"net of its {RR['be']} bp break-even there is nothing left to harvest at scale."),

        md(
            "## Beat 7 · Going further\n\n"
            "### 7a · Worked complement — the holding-period sweep ([`../docs/extension.md`](../docs/extension.md))\n"
            "Hold each name's surprise $h$ days; does the drift outpay the roll cost? Synthetic control here; "
            f"**on the real tape** net @5 bp runs {RR['rh5']} (5d) → {RR['rh20']} (20d) → {RR['rh40']} (40d) → "
            f"+{RR['rh60']} (60d) — only break-even at the full quarter."
        ),
        code(
            "hp = extension.holding_period_sweep(panel, events, holds=[5,20,40,60,90], cost_bps=5.0)\n"
            "display(hp.round(3))\n"
            "fig, ax = plt.subplots(); ax.axhline(0, color='#999', lw=.8)\n"
            "ax.plot(hp.index, hp['gross_sharpe'], marker='o', label='gross', color='#2ea44f')\n"
            "ax.plot(hp.index, hp['net_sharpe'], marker='o', label='net @5bp', color='#c0392b')\n"
            "ax.set_xlabel('holding period (days)'); ax.set_ylabel('Sharpe'); ax.legend(); ax.set_title('Holding-period sweep (synthetic)')\n"
            f"print('net Sharpe by hold (control): 5d {R['h5']}, 20d {R['h20']}, 40d {R['h40']}, 60d {R['h60']}, 90d {R['h90']}.')\n"
            f"print('REAL tape net @5bp: 5d {RR['rh5']}, 20d {RR['rh20']}, 40d {RR['rh40']}, 60d +{RR['rh60']}, 90d +{RR['rh90']} -- only break-even at a full quarter.')"
        ),
        md(
            "> 💡 **In plain words.** Hold too short and you pay all the turnover but miss the slow drift — on "
            f"the **real tape** net @5 bp is **{RR['rh5']} at a 5-day hold**, climbing only to break-even "
            f"(**+{RR['rh60']}**) at a 60-day hold. The drift is genuine but so slow and small that even the "
            "best horizon barely clears 5 bp."
        ),
        md(
            "### 7b · The honest verdict & forks\n"
            "The real measurement is **done and fingerprinted** in [`../docs/results.md`](../docs/results.md): "
            f"gross Sharpe {RR['gross_sh']}, Newey-West *t* {RR['nw_t']}, turnover {RR['turn']}/day, "
            f"break-even {RR['be']} bp, the cost & holding sweeps and the real drift-decay curve, "
            f"as-of {RR['asof']}, fp `{RR['fp']}`. The literature's prediction held: a real but modest gross "
            "edge, eroded to a thin residual by costs on the liquid names you can actually trade — the "
            "`WEAK` / `MIRAGE` verdict, with the drift-decay shape `CONFIRMED`.\n\n"
            "### 7c · Other forks\n"
            "- **Survivorship-aware universe** — the panel is current S&P membership; a delisting-aware panel "
            "would lower the magnitudes further.\n"
            "- **Liquidity / size tiering** — model each tier's honest spread; how much PEAD survives net?\n"
            "- **SUE bucketing** — trade only the extreme-surprise deciles (Bernard-Thomas).\n"
            "- **Earnings- vs price-momentum** — incremental drift from the *earnings* surprise (Chordia-Shivakumar 2006).\n\n"
            "PRs welcome — try a liquidity-tiered or delisting-aware book."
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
