"""Generate the two narrative notebooks for Study 35 (Contango) from source.

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

The executed path runs OFFLINE on the seeded synthetic commodity term-structure panel (each commodity a
persistent roll-yield state that predicts its return — the machinery); the real-tape verdict is explicitly
**PENDING a term-structure fetch** (the curve — front + deferred contracts — is not available in this
sandbox; see ../docs/results.md), exactly the honesty pattern of Study 27 (Steamroller). Both notebooks
walk the SAME seven desk beats. Contango is the commodity sibling of Study 27 (FX carry).
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
from contango import data, strategy, costs, extension

# Offline synthetic control: a 12-commodity panel where roll yield (backwardation/contango) predicts
# return (the machinery) + a disconnected null. The real curve (front+deferred) is PENDING — see ../docs/results.md.
r, ry, truth = data.synthetic_term_structure(carry_strength=0.9, seed=35)
r0, ry0, _ = data.synthetic_term_structure(carry_strength=0.0, seed=35)
print(f"synthetic control: {truth.n_commodities} commodities x {truth.n_weeks} weeks, carry_strength {truth.carry_strength} (null=0)")
"""

# Synthetic-control numbers (seed 35, fingerprint b502aaa6304f) + pre-registered real-tape shape.
R = dict(
    gross_sh="1.86", gross_cagr="16.5", gross_dd="-12", gross_skew="-0.16", net5="1.80",
    hml="27.6", hi="21.4", lo="-6.2", null_sh="-0.28", null_hml="-4.8",
    turn="0.19", be="160",
    c0="1.861", c2="1.837", c5="1.802", c10="1.743", c20="1.626",
    blend_carry="1.80", blend_mom="1.43", blend="2.03", blend_corr="+0.27",
    fp="b502aaa6304f", asof="2026-06-10",
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Real-tape run?: Pre-reg](https://img.shields.io/badge/Real--tape_run%3F-Pre--reg-8b949e?style=flat-square)\n\n"
)


def md(t): return new_markdown_cell(t)
def code(t): return new_code_cell(t)


def build_curious():
    cells = [
        md(
            "# Contango 🛢️\n"
            "### \"Buy the backwardated curves, sell the contangoed ones.\" The roll yield is a real commodity premium — that picks up nickels in front of a (volatile) truck.\n\n"
            + BADGES +
            "A commodity future doesn't just track the spot price — as your long position rolls toward "
            "expiry it slides along the **term-structure curve**. If the curve is **backwardated** (the "
            "front contract is dearer than the deferred), you sell the expiring contract high and buy the "
            "next one cheap: a positive **roll yield**. If it's in **contango** (front cheaper than "
            "deferred), you roll *down* and pay a tax. The documented edge: backwardated commodities "
            "out-earn contangoed ones, so a book long the most-backwardated and short the most-contangoed "
            "harvests a real carry. It's the commodity cousin of the [FX carry trade](../../27-steamroller/) "
            "— and, like it, real but crash-prone.\n\n"
            "> 📓 **Plain-language layer.** The carry-by-bucket sort, the cost/turnover read and the "
            "carry+momentum blend are in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Real run pending the curve.** Roll yield needs the **term structure** (front *and* "
            "deferred contracts); this sandbox caches only front-month continuous returns and can't fetch "
            "the deferred leg. So the core runs on a **synthetic** control and the real-tape run is "
            "*pre-registered* in [`../docs/results.md`](../docs/results.md) — the honesty pattern of "
            "[Study 27](../../27-steamroller/). Not investment advice. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## The answer first 🛢️\n\n"
            "| What we asked | The honest answer |\n"
            "|---|---|\n"
            "| Do backwardated commodities out-earn? | 🟩 **Yes.** On the control, high-minus-low roll-yield "
            f"spread **+{R['hml']}%/yr**, gross Sharpe **{R['gross_sh']}**; the disconnected null is flat "
            f"(**{R['null_sh']}**). Decades of evidence agree (Gorton–Rouwenhorst, Erb–Harvey, Koijen et al.). |\n"
            "| Could you trade it? | 🟨 **Carefully.** It's *cheap* to run (turnover **{t}**/wk, break-even "
            "**~{be} bp** — costs aren't the constraint), but carry is **volatile and crash-prone** and lives "
            "in the least-liquid contracts. |\n"
            "| Have we measured the real tape? | ⚪ **Not yet.** Roll yield needs the curve (front+deferred); "
            "the sandbox can't fetch it. The run is **pre-registered**, pending the data. |\n\n"
            "> Desk shorthand: **Signal `REAL` · Tradability `FRAGILE` · Real-tape run? `PRE-REG`** — real "
            "premium, fragile to trade, real run pending the curve.".format(t=R['turn'], be=R['be'])
        ),

        md(
            "## 1 · The claim 📣\n\n"
            "Commodity carry / roll yield (Kakushadze–Serur §9.1, §9.4; lineage Gorton–Rouwenhorst 2006, "
            "Erb–Harvey 2006):\n\n"
            "1. **Signal** — each commodity's **roll yield**: the slope of its futures curve. Backwardated "
            "(front > deferred) ⇒ positive roll; contangoed ⇒ negative roll.\n"
            "2. **Cross-sectional sort** — rank the commodities by roll yield; go **long the most-"
            "backwardated**, **short the most-contangoed**, dollar-neutral.\n"
            "3. **Weekly rebalance** — roll yield is a slow signal, so the book turns over modestly.\n\n"
            "The believer's case: the curve shape reflects scarcity and storage economics, and a "
            "backwardated curve *pays you to hold the future*. That payment — the roll yield — is the "
            "harvestable carry."
        ),
        code(
            "book = strategy.book_returns(r, ry, cost_bps=0.0)   # GROSS, on the synthetic control\n"
            "eq = (1+book).cumprod()\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(eq.index, eq.values, color='#2ea44f', lw=1.1)\n"
            "ax.set_title('Dollar-neutral carry book on a panel where roll yield truly pays — the machinery works')\n"
            "s = strategy.summary(book); print(f\"synthetic gross Sharpe {s['sharpe']:.2f}  (null ~0)\")"
        ),

        md(
            "## 2 · So what? 💰\n\n"
            "A real, durable, cheap-to-run premium that needs only the futures curve — not a forecast — is "
            "the backbone of every systematic commodity book and a core sleeve in CTAs and risk-parity "
            "funds. And roll yield genuinely *is* real: Erb–Harvey showed the cross-section of commodity "
            "returns is explained far more by the term structure than by spot moves. So the question isn't "
            "whether the effect exists — it's whether you can hold the volatile, crash-prone stream long "
            "enough to collect it. That's the `FRAGILE` half of the verdict."
        ),

        md(
            "## 3 · How we'd know 🔬\n\n"
            "1. **Real?** Does a long-backwardation / short-contango book earn a positive premium gross "
            "(and nothing on a null where roll yield is disconnected from returns)?\n"
            "2. **Tradable?** Turnover, break-even cost, and — the real risk — the crash tail.\n"
            "3. **Diversifiable?** Does adding a momentum sleeve lift the combined Sharpe? (beat 7)\n\n"
            "**Mirage line** (pre-registered): on the *real* curve, if the backwardated-minus-contangoed "
            "spread is statistically indistinguishable from zero, or only the liquid contracts (which carry "
            "least) are tradable, the signal drops to `WEAK`/`NONE`."
        ),

        md("## 4 · The teardown 🔧\n\n### 4a · The machinery works where roll yield pays (control vs null)"),
        code(
            "for label, rr, yy in [('carry panel', r, ry), ('null (disconnected)', r0, ry0)]:\n"
            "    s = strategy.summary(strategy.book_returns(rr, yy, cost_bps=0.0))\n"
            "    pb = strategy.carry_premium_by_bucket(rr, yy)\n"
            "    print(f\"{label:22} gross Sharpe {s['sharpe']:+6.2f}  H-L {pb['hml_ann_pct']:+6.1f}%/yr  turnover/wk {strategy.turnover(yy):.3f}\")"
        ),
        md(
            "### 4b · On the real commodity curves — PENDING the term structure\n"
            "Roll yield needs the front *and* deferred contract; the sandbox caches only front-month "
            "continuous returns ([`../docs/results.md`](../docs/results.md)). So the real numbers are "
            "**pre-registered**, not yet measured:\n\n"
            f"- The control proves the premium: high-minus-low roll-yield spread **+{R['hml']}%/yr** "
            f"(top **+{R['hi']}%**, bottom **{R['lo']}%**), gross Sharpe **{R['gross_sh']}**.\n"
            f"- It's cheap to run: turnover **{R['turn']}/wk**, break-even **~{R['be']} bp** — costs aren't "
            "the binding constraint (the opposite of [Slingshot](../../33-slingshot/)).\n"
            "- The expected real-tape shape (from the literature): a standalone Sharpe of roughly "
            "**0.5–0.8** with deep, volatile drawdowns — the `REAL` / `FRAGILE` verdict."
        ),
        code(
            "print('Synthetic control (../docs/results.md, fp " + R['fp'] + "):')\n"
            f"print('  carry gross   Sharpe {R['gross_sh']}  CAGR {R['gross_cagr']}%  maxDD {R['gross_dd']}%  skew {R['gross_skew']}')\n"
            f"print('  carry net@5bp Sharpe {R['net5']}   (break-even ~{R['be']} bp -> costs not the constraint)')\n"
            f"print('  null (disconnected) Sharpe {R['null_sh']}  H-L {R['null_hml']}%/yr  -> apparatus measures the effect, not itself')\n"
            "print('\\nReal commodity curves: PENDING a term-structure fetch (front+deferred).')\n"
            "# the cost curve, drawn on the synthetic where there IS an edge — note how FLAT it is (slow signal):\n"
            "cs = costs.cost_sweep(r, ry)\n"
            "fig, ax = plt.subplots(); ax.axhline(0, color='#999', lw=.8)\n"
            "ax.plot(cs.index, cs['sharpe'], marker='o', color='#dab617')\n"
            "ax.set_xlabel('cost (bp per unit traded)'); ax.set_ylabel('net Sharpe'); ax.set_title('Costs barely dent a slow carry book')"
        ),

        md("## 5 · The verdict 🧾\n\n"
           f"- **Signal `REAL`** — control H-L **+{R['hml']}%/yr**, gross Sharpe {R['gross_sh']}; null flat ({R['null_sh']}); decades of literature agree.\n"
           f"- **Tradability `FRAGILE`** — cheap to run (break-even ~{R['be']} bp), but volatile and crash-prone, biggest in illiquid contracts.\n"
           "- **Real-tape run `PRE-REG`** — roll yield needs the curve the sandbox can't fetch; the run is pre-registered, pending the data.\n\n"
           "> **The commodity sibling of Steamroller.** Carry is a real, cheap-to-run premium that you must "
           "be willing to hold through a crash — currencies *or* commodities."),

        md("## 6 · Could you trade it? 💸\n\n"
           "- **Cheaply, on the signal side.** Roll yield is slow; the book turns over ~0.19/week, so "
           "transaction costs are not what kills it. That's the good news.\n"
           "- **The crash is the catch.** Commodity carry, like FX carry, is a volatile, negatively-skewed "
           "stream that unwinds hard in commodity-wide risk-off (2008, 2014–15). You're paid to hold a tail.\n"
           "- **And capacity bites.** The premium is largest in the smaller, less-liquid contracts; the "
           "deeply liquid ones (crude, gold) carry less of it — the same illiquidity tension as Slingshot.\n"
           "- **The honest move:** diversify it (beat 7), don't lever it."),

        md(
            "## 7 · Going further 🚪\n\n"
            "### Worked complement — \"diversify the carry with momentum\" ([`../docs/extension.md`](../docs/extension.md))\n"
            "Carry and time-series momentum are the two classic commodity premia and are *lowly correlated* "
            "(Koijen et al. 2018). Does blending a momentum sleeve into the carry book lift the combined Sharpe?\n\n"
            f"- On the control, leg correlation is just **{R['blend_corr']}** — low, as the literature predicts.\n"
            f"- A 50/50 blend's Sharpe (**{R['blend']}**) **beats either standalone leg** (carry {R['blend_carry']}, "
            f"momentum {R['blend_mom']}): a genuine diversification gain, not a redundant bet.\n"
            "- The lesson echoes [Trade-Winds](../../31-trade-winds/): on this desk the edge is "
            "*diversification*, not prediction.\n\n"
            "### Other forks\n"
            "- **Add a value sleeve** — long-horizon commodity reversal (Asness et al. 2013) for the full three-factor book.\n"
            "- **Vol-target the blend** — does constant-risk sizing tame the carry crash (cf. Study 27, where it failed on FX)?\n"
            "- **Wire a real curve** — the one thing the verdict is waiting on: a front+deferred term-structure feed.\n\n"
            "PRs welcome — add value, vol-target the blend, or (best of all) bring a term-structure fetch."
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


def build_quants():
    cells = [
        md(
            "# Contango — a quantitative teardown 🔬\n"
            "### The carry premium by roll-yield bucket · control vs null · turnover & break-even · the carry+momentum blend\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same seven "
            "beats, every claim with its number.* The steelman: the commodity roll yield is a real, "
            "documented carry premium (Gorton–Rouwenhorst 2006; Erb–Harvey 2006; Koijen et al. 2018). We "
            "confirm it's `REAL` on a synthetic control (high-minus-low spread +27.6%/yr, gross Sharpe "
            "1.86), show it's `FRAGILE` (volatile, crash-prone, capacity-limited), and — because roll yield "
            "needs the **term structure** this sandbox can't fetch — `PRE-REG` the real-tape run.\n\n"
            "> ⚠️ **Not investment advice.** The core executes on a synthetic roll-yield panel; the real "
            "commodity-curve run is **pending a term-structure fetch** (front + deferred contracts), "
            "pre-registered in [`../docs/results.md`](../docs/results.md), sources in "
            "[`../docs/references.md`](../docs/references.md) — the honesty pattern of Study 27 (Steamroller).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        md(
            "## Beat 0 · Verdict\n\n"
            "| Axis | Stamp | Why |\n"
            "|---|---|---|\n"
            f"| **Signal** — backwardated out-earn contangoed? | 🟢 `REAL` | Control H-L roll-yield spread "
            f"**+{R['hml']}%/yr**, gross Sharpe **{R['gross_sh']}**; null flat (**{R['null_sh']}**). |\n"
            f"| **Tradability** | 🟡 `FRAGILE` | Cheap to run (turnover **{R['turn']}**/wk, break-even "
            f"**~{R['be']} bp**) but volatile, crash-prone, illiquid-tilted. |\n"
            "| **Real-tape run?** | ⚪ `Pre-reg` | Roll yield needs front+deferred contracts the sandbox "
            "can't fetch; apparatus & mirage line pre-registered. |\n\n"
            "> **In one sentence:** a real, durable, cheap-to-run commodity carry premium that is volatile "
            "and crash-prone (a `FRAGILE` cousin of FX carry), proven on a synthetic control with the real "
            "run pre-registered and pending the term-structure data.\n\n"
            "*(This notebook executes on the synthetic control; the pre-registered real numbers are in "
            "[`../docs/results.md`](../docs/results.md).)*"
        ),

        md(
            "## Beat 1 · The claim, precisely\n\n"
            "Each commodity $i$ has a roll yield $y_{i,t}$ (the curve slope: backwardation $>0$, contango "
            "$<0$). Weight $w_{i,t} \\propto y_{i,t}-\\bar y_t$ (demeaned across the cross-section), scaled "
            "so $\\sum_i w_{i,t}=0$ (dollar-neutral) and $\\sum_i|w_{i,t}|=1$ (gross 1), lagged one week. "
            "Claim: $\\sum_i w_{i,t-1} r_{i,t}$ earns a positive premium. Null: the roll-yield signal is "
            "**disconnected** from returns ⇒ nothing to harvest."
        ),
        code(
            "for label, rr, yy in [('carry panel', r, ry), ('null', r0, ry0)]:\n"
            "    g = strategy.summary(strategy.book_returns(rr, yy, cost_bps=0.0))\n"
            "    pb = strategy.carry_premium_by_bucket(rr, yy)\n"
            "    print(f\"{label:14} gross Sharpe {g['sharpe']:+6.2f}  H-L {pb['hml_ann_pct']:+6.1f}%/yr  turnover/wk {strategy.turnover(yy):.3f}\")"
        ),
        md(
            "> 💡 **In plain words.** When roll yield genuinely predicts return, the long-backwardation / "
            f"short-contango book makes money (gross Sharpe **{R['gross_sh']}**, spread **+{R['hml']}%/yr**); "
            f"when the signal is disconnected, it earns nothing (**{R['null_sh']}**). The apparatus sees "
            "carry only when it's there — so the (pending) real-tape result will be about the *market*."
        ),

        md(
            "## Beat 2 · So what?\n\n"
            "Gorton–Rouwenhorst (2006) show the equal-weight commodity basket earns an equity-like premium "
            "dominated by the **roll return**; Erb–Harvey (2006) show the *cross-section* is explained far "
            "more by the term structure than by spot. Koijen et al. (2018) generalise 'carry' across asset "
            "classes and document a robust commodity carry that is lowly correlated with momentum and value. "
            "The pre-registered prediction: the premium is real but volatile/crash-prone and concentrated in "
            "less-liquid contracts — so it's `REAL` to measure and `FRAGILE` to trade. Beats 4–7 test that."
        ),

        md(
            "## Beat 3 · Pre-registered protocol\n\n"
            "1. **Real?** `book_returns(cost_bps=0)` and `carry_premium_by_bucket` on control vs null (and on "
            "the real curve, once fetched).\n"
            "2. **Tradable?** `strategy.turnover`, `costs.breakeven_cost_bps`, `costs.cost_sweep` — and the "
            "crash tail (skew/drawdown) on the real curve.\n"
            "3. **Diversifiable?** `extension.combine` — carry + momentum blend Sharpe vs the legs.\n\n"
            "**Mirage line:** on the real curve, backwardated-minus-contangoed spread with HAC *t* < 2, or a "
            "premium that survives only in untradeably-illiquid contracts ⇒ `WEAK`/`NONE`."
        ),

        md("## Beat 4 · The teardown\n\n### 4a · Real on the control, flat on the null (the REAL call)"),
        code(
            "g = strategy.summary(strategy.book_returns(r, ry, cost_bps=0.0))\n"
            "pb = strategy.carry_premium_by_bucket(r, ry)\n"
            "print(f\"carry book   gross Sharpe {g['sharpe']:.2f}  CAGR {g['cagr']*100:.1f}%  maxDD {g['max_drawdown']*100:.0f}%  skew {g['skew']:+.2f}\")\n"
            "print(f\"buckets      top {pb['high_ann_pct']:+.1f}%/yr  bottom {pb['low_ann_pct']:+.1f}%/yr  H-L {pb['hml_ann_pct']:+.1f}%/yr\")\n"
            "g0 = strategy.summary(strategy.book_returns(r0, ry0, cost_bps=0.0)); pb0 = strategy.carry_premium_by_bucket(r0, ry0)\n"
            "print(f\"null         gross Sharpe {g0['sharpe']:+.2f}  H-L {pb0['hml_ann_pct']:+.1f}%/yr  -> nothing to harvest\")"
        ),
        md(
            "> 💡 **In plain words.** A +27.6%/yr high-minus-low spread on the control with a flat null is the "
            "machinery working: it harvests carry precisely when roll yield predicts return. The real-tape "
            "magnitude (pending the curve) will be smaller — the literature says a Sharpe near 0.5–0.8 — but "
            "the *sign and structure* are what the control validates."
        ),

        md("### 4b · The cost curve and the break-even (the cheap-to-run half of FRAGILE)"),
        code(
            "cs = costs.cost_sweep(r, ry)\n"
            "print('Net Sharpe vs cost (synthetic control):')\n"
            f"print('  0bp {R['c0']}   2bp {R['c2']}   5bp {R['c5']}   10bp {R['c10']}   20bp {R['c20']}')\n"
            f"print(f'  turnover/wk {R['turn']}   break-even ~{R['be']} bp  (liquid futures round-trip ~2-5 bp)')\n"
            "fig, ax = plt.subplots(); ax.axhline(0, color='#999', lw=.8)\n"
            "ax.plot(cs.index, cs['sharpe'], marker='o', color='#dab617')\n"
            "ax.set_xlabel('cost (bp/unit)'); ax.set_ylabel('net Sharpe'); ax.set_title('A slow signal: costs barely move the Sharpe')\n"
            "print('\\nsynthetic break-even:', round(costs.breakeven_cost_bps(r, ry), 0), 'bp -> costs are NOT the constraint; the crash tail is')"
        ),
        md(
            "> 💡 **In plain words.** Unlike the daily-churn equity book in [Slingshot](../../33-slingshot/) "
            "(break-even 3.3 bp), carry's break-even is ~160 bp — miles above realistic futures costs. So "
            "what makes carry `FRAGILE` is **not** the spread; it's the volatile, crash-prone return stream "
            "you must hold to earn it."
        ),

        md("## Beat 5 · The verdict\n\n"
           f"- **`REAL`** (4a): control H-L **+{R['hml']}%/yr**, gross Sharpe {R['gross_sh']}; null flat {R['null_sh']}; literature concurs.\n"
           f"- **`FRAGILE`** (4b + beat 6): cheap to run (break-even ~{R['be']} bp) but volatile, crash-prone, illiquid-tilted.\n"
           "- **Real-tape `PRE-REG`** (beat 3): roll yield needs the term structure the sandbox can't fetch; run pre-registered.\n\n"
           "> **Signal `REAL` · Tradability `FRAGILE` · Real-tape run? `PRE-REG`** — the commodity sibling of Steamroller."),

        md("## Beat 6 · Could you trade it?\n\n"
           "- **The signal side is cheap.** Roll yield turns over ~0.19/week; transaction costs don't kill it.\n"
           "- **The crash tail is the binding risk.** Commodity carry is negatively skewed and unwinds hard "
           "in commodity-wide risk-off (2008, 2014–15) — you're paid to hold a tail (Koijen et al. 2018).\n"
           "- **Capacity & the illiquidity tilt.** The premium is largest in smaller, less-liquid contracts; "
           "the deeply liquid ones carry less of it — the same tension as Slingshot. The pre-registered "
           "mirage check: if only the liquid (low-carry) contracts are tradable, the edge thins.\n"
           "- **Honest fix:** diversify (beat 7), don't lever."),

        md(
            "## Beat 7 · Going further\n\n"
            "### 7a · Worked complement — the carry+momentum blend ([`../docs/extension.md`](../docs/extension.md))\n"
            "Carry and time-series momentum are the two classic commodity premia and are lowly correlated "
            "(Koijen et al. 2018). A 50/50 risk blend should beat the stronger leg. Synthetic here; the real "
            "blend is pre-registered (momentum needs only front-month prices, which *are* cached)."
        ),
        code(
            "c = extension.combine(r, ry, cost_bps=5.0)\n"
            "print(f\"carry Sharpe {c['carry_sharpe']:.2f}   momentum Sharpe {c['momentum_sharpe']:.2f}   \"\n"
            "      f\"blend Sharpe {c['blend_sharpe']:.2f}   (leg correlation {c['correlation']:+.2f})\")\n"
            "eqc=(1+c['carry']).cumprod(); eqm=(1+c['momentum']).cumprod(); eqb=(1+c['blend']).cumprod()\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(eqc.index, eqc.values, color='#2ea44f', lw=1.0, label=f\"carry ({c['carry_sharpe']:.2f})\")\n"
            "ax.plot(eqm.index, eqm.values, color='#8b949e', lw=1.0, label=f\"momentum ({c['momentum_sharpe']:.2f})\")\n"
            "ax.plot(eqb.index, eqb.values, color='#c0392b', lw=1.3, label=f\"50/50 blend ({c['blend_sharpe']:.2f})\")\n"
            "ax.legend(); ax.set_title('Lowly-correlated legs: the blend beats either standalone')"
        ),
        md(
            "> 💡 **In plain words.** Leg correlation is just +0.27, so the 50/50 blend's Sharpe (2.03) "
            "**exceeds either leg** (carry 1.80, momentum 1.43) — the textbook diversification free lunch. "
            "The institutional answer to carry's crash isn't more leverage; it's a lowly-correlated momentum "
            "sleeve. Echoes [Trade-Winds](../../31-trade-winds/): the edge is diversification, not prediction."
        ),
        md(
            "### 7b · What the real run will settle\n"
            "The apparatus captures carry wherever roll yield predicts return (the control) and the blend "
            "diversifies it. The one thing **pending** is the real term structure — the front+deferred "
            "contracts — to put a fingerprinted number on the actual commodity-curve carry, its crash tail, "
            "and the real blend. Full pre-registration in [`../docs/results.md`](../docs/results.md) and "
            "[`../docs/extension.md`](../docs/extension.md).\n\n"
            "### 7c · Other forks\n"
            "- **Add a value sleeve** — long-horizon commodity reversal (Asness et al. 2013).\n"
            "- **Vol-target the blend** — does constant-risk sizing tame the carry crash (cf. Study 27)?\n"
            "- **Wire a term-structure feed** — the data the whole real-tape verdict is waiting on.\n\n"
            "PRs welcome — add value, vol-target the blend, or bring a front+deferred curve fetch."
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "02_for_the_quants.ipynb")


def _meta():
    return {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}


def _write(nb, name):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
