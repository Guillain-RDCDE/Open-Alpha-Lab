"""Generate the two narrative notebooks for Study 37 (Barometer) from source.

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

The executed path runs OFFLINE on the seeded synthetic cross-asset world (two latent, regime-switching
macro states whose lagged momentum predicts returns) — the machinery. The real-tape verdict (cross-asset
proxies + FRED macro) is explicitly **PENDING a reliable FRED macro fetch** (see ../docs/results.md), so
the notebooks quote the synthetic control numbers and pre-register the real run. Both notebooks walk the
SAME seven desk beats. Mirrors Study 27 (Steamroller)'s pending-fetch honesty pattern.
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
from barometer import data, strategy, costs, extension

# Offline synthetic control: a cross-asset world whose latent macro momentum predicts returns (the
# machinery) + a macro_strength=0 null. The real cross-asset/FRED verdict is PENDING (../docs/results.md).
r, m, truth = data.synthetic_macro(macro_strength=1.0, seed=37)
r0, m0, _ = data.synthetic_macro(macro_strength=0.0, seed=37)
print(f"synthetic control: {truth.n_assets} assets x {truth.n_months} months, macro_strength {truth.macro_strength} (null=0)")
"""

# Synthetic-control numbers (seed 37, 50y, gross) — what the executed cells reproduce.
S = dict(
    mm_sh="1.09", mm_ann="5.1", mm_dd="-12", mm_turn="5.6", mm_null="-0.17", mm_be="91",
    inf_sh="0.55", inf_ann="2.3", inf_null="-0.02", inf_be="60",
    reg_inf_up_sh="0.59", reg_inf_up_ann="2.5", reg_inf_dn_sh="0.46", reg_inf_dn_ann="1.9",
    reg_mm_up_sh="0.99", reg_mm_dn_sh="1.33",
    c0="1.09", c5="1.03", c10="0.97", c25="0.78", c50="0.48",
)

BADGES = (
    "![Signal: Real on the level](https://img.shields.io/badge/Signal-Real_on_the_level-2ea44f?style=flat-square)\n"
    "![Signal: Weak on the size](https://img.shields.io/badge/Signal-Weak_on_the_size-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Real-tape run?: Pre-reg](https://img.shields.io/badge/Real--tape_run%3F-Pre--reg-8b949e?style=flat-square)\n\n"
)


def md(t): return new_markdown_cell(t)
def code(t): return new_code_cell(t)


def build_curious():
    cells = [
        md(
            "# Barometer 🌡️\n"
            "### \"Read the macro weather, trade the trend.\" A *real* cross-asset premium — slow, modest, and only a storm-hedge when it storms.\n\n"
            + BADGES +
            "The idea is old and intuitive: the economy has weather. When **growth** is improving, "
            "pro-cyclical assets (stocks, commodities) do well; when **inflation** is rising, *real* "
            "assets (commodities, inflation-linked bonds, gold) protect you while nominal stocks and bonds "
            "suffer. So watch the *trend* in the macro data — the barometer — and lean your portfolio the "
            "way it's pointing. This study asks the honest two-part question: is that real, and can you "
            "trade it?\n\n"
            "> 📓 **Plain-language layer.** The macro-momentum premium, the cost/break-even, and the "
            "regime split that tests the inflation hedge are in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Real run pending one fetch.** The macro state needs **FRED** series (industrial "
            "production, payrolls, CPI, breakeven inflation) that **time out / are intermittent** in this "
            "environment — so, exactly like [Study 27 (Steamroller)](../../27-steamroller/), the core runs "
            "on a **synthetic** cross-asset world and the real cross-asset numbers are "
            "**pre-registered & pending** ([`../docs/results.md`](../docs/results.md)). House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## The answer first 🌡️\n\n"
            "| What we asked | The honest answer |\n"
            "|---|---|\n"
            "| Does the *trend* in macro data predict returns? | 🟩 **Yes — on the level.** On the control "
            f"the macro-momentum book earns Sharpe **{S['mm_sh']}** (+{S['mm_ann']}%/yr); the null is flat "
            f"(**{S['mm_null']}**). |\n"
            "| Is it big? | 🟨 **No — modest & slow.** A ~0.4–0.8 Sharpe with long flat stretches in the "
            "literature (Brooks–Moskowitz 2017). |\n"
            "| Does the inflation hedge pay *when inflation rises*? | 🟨 **Yes, but episodically.** Rising "
            f"inflation Sharpe **{S['reg_inf_up_sh']}** vs falling **{S['reg_inf_dn_sh']}** — it earns its "
            "keep in the storm, drags otherwise. |\n"
            "| Real cross-asset / FRED run? | ⚪ **Pre-registered & pending** a reliable FRED macro fetch. |\n\n"
            "> Desk shorthand: **Signal `REAL`-on-the-level / `WEAK`-on-the-size · Tradability `FRAGILE` · "
            "Real-tape run? `PRE-REG`** — a real, slow, diversifying premium, proven on the control, "
            "pending on the tape."
        ),

        md(
            "## 1 · The claim 📣\n\n"
            "Fundamental macro-momentum & inflation hedging (Kakushadze-Serur §19.2/§19.3; Brooks–Moskowitz "
            "2017; Neville et al. 2021):\n\n"
            "1. **Macro momentum** — go long the assets favoured by *improving* macro momentum: growth-up "
            "lifts pro-cyclical assets, inflation-up favours real assets. Signal = the *change* in the macro "
            "state, lagged so it's tradable.\n"
            "2. **Inflation hedge** — when inflation momentum is positive, overweight **real** assets "
            "(commodities, TIPS, gold) and underweight nominal ones.\n\n"
            "The believer's case: macro data is slow and trends, asset prices adjust slowly to it, and real "
            "assets mechanically protect purchasing power — a real, diversifying premium that shows up "
            "*across* asset classes, not within one."
        ),
        code(
            "book = strategy.book_returns(r, m, kind='macro_momentum', cost_bps=0.0)  # GROSS, on the control\n"
            "eq = (1+book).cumprod()\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(eq.index, eq.values, color='#2ea44f', lw=1.1)\n"
            "ax.set_title('Macro-momentum book on a world that truly trends with the macro barometer')\n"
            "s = strategy.summary(book); print(f\"synthetic gross Sharpe {s['sharpe']:.2f}  (+{100*s['ann_return']:.1f}%/yr, null ~0)\")"
        ),

        md(
            "## 2 · So what? 💰\n\n"
            "A return stream that comes from the *macro cycle* is, almost by construction, **diversifying** "
            "— it doesn't depend on stocks going up, and the inflation leg pays *exactly* when a 60/40 "
            "portfolio is bleeding (the 1970s, 2021–22). That's worth more than its raw Sharpe suggests: a "
            "modest, uncorrelated, crisis-friendly sleeve is the holy grail of portfolio construction (it's "
            "why the desk's [Study 31 Trade-Winds](../../31-trade-winds/) earned its keep). The catch is the "
            "*slowness* and the *episodic* inflation payoff — you have to sit through long flat stretches."
        ),

        md(
            "## 3 · How we'd know 🔬\n\n"
            "1. **Real?** Gross macro-momentum Sharpe on the control (and flat on a `macro_strength=0` null).\n"
            "2. **Tradable?** Turnover, break-even cost — for a *slow* book, is cost even the threat?\n"
            "3. **Does the hedge pay when it should?** Split the inflation book by rising- vs "
            "falling-inflation regime.\n\n"
            "**Mirage line:** if the macro-momentum book were flat on the control, *or* the inflation hedge "
            "paid the *same* in both regimes, the story collapses — there'd be no conditional macro premium "
            "to harvest."
        ),

        md("## 4 · The teardown 🔧\n\n### 4a · The machinery works where macro momentum is real (control vs null)"),
        code(
            "for label, (rr, mm) in [('macro-driven world', (r, m)), ('null (pure noise)', (r0, m0))]:\n"
            "    s = strategy.summary(strategy.book_returns(rr, mm, kind='macro_momentum', cost_bps=0.0))\n"
            "    print(f\"{label:20} macro-mom gross Sharpe {s['sharpe']:+6.2f}  turnover {strategy.turnover_ann(mm,'macro_momentum'):.1f}x/yr\")"
        ),
        md(
            "### 4b · Cost is *not* the threat — the book is slow\n"
            f"Macro signals refresh monthly and trend, so turnover is only **{S['mm_turn']}×/yr** and the "
            f"**break-even cost is ~{S['mm_be']} bp** — far above realistic cross-asset costs. Unlike the "
            "desk's daily-churn books ([Slingshot](../../33-slingshot/)), this one survives costs easily; "
            "the threats are elsewhere (modest Sharpe, long droughts, episodic hedge)."
        ),
        code(
            "cs = costs.cost_sweep(r, m, kind='macro_momentum')\n"
            "display(cs.round(3))\n"
            "fig, ax = plt.subplots(); ax.axhline(0, color='#999', lw=.8)\n"
            "ax.plot(cs.index, cs['sharpe'], marker='o', color='#2ea44f')\n"
            "ax.set_xlabel('cost (bp per unit traded)'); ax.set_ylabel('net Sharpe'); ax.set_title('A slow book barely feels the spread')\n"
            "print('break-even:', round(costs.breakeven_cost_bps(r, m, 'macro_momentum')), 'bp')"
        ),

        md("## 5 · The verdict 🧾\n\n"
           f"- **Signal `REAL` on the level / `WEAK` on the size** — gross Sharpe {S['mm_sh']} on the "
           f"control, null {S['mm_null']}; but the literature says ~0.4–0.8 with long flat stretches.\n"
           f"- **Tradability `FRAGILE`** — cheap to run (break-even ~{S['mm_be']} bp), but modest, slow, "
           "and the inflation hedge is episodic.\n"
           "- **Real-tape run? `PRE-REG`** — pending a reliable FRED macro fetch ([`../docs/results.md`](../docs/results.md)).\n\n"
           "> **A real but humble premium.** Macro momentum is diversifying and crisis-friendly — its value "
           "is in a *portfolio*, not as a standalone Sharpe hero."),

        md("## 6 · Could you trade it? 💸\n\n"
           "- **Yes, cheaply — but patiently.** Low turnover means costs don't kill it (unlike most of this "
           "desk). The price of admission is *time*: macro momentum has multi-year flat stretches.\n"
           "- **The inflation hedge is insurance, not income.** It pays in rising-inflation regimes and "
           "drags otherwise (beat 7) — so size it as a *conditional overlay*, not a permanent sleeve.\n"
           "- **Best used as a sleeve.** A modest, uncorrelated macro book lifts a 60/40's risk-adjusted "
           "return far more than its standalone Sharpe implies — the Trade-Winds lesson again."),

        md(
            "## 7 · Going further 🚪\n\n"
            "### Worked complement — \"does the inflation hedge pay when it's supposed to?\" ([`../docs/extension.md`](../docs/extension.md))\n"
            "We split the inflation book by regime — rising vs falling inflation:\n\n"
            f"- Rising-inflation Sharpe **{S['reg_inf_up_sh']}** (+{S['reg_inf_up_ann']}%/yr) vs falling "
            f"**{S['reg_inf_dn_sh']}** (+{S['reg_inf_dn_ann']}%/yr) — it **pays more in the regime it "
            "targets**, as designed (though it's not free insurance — the real-asset basket has its own drift).\n"
            f"- The broader macro-momentum book is steadier across both regimes (it also rides growth), "
            "confirming the inflation leg is the *conditional* one.\n\n"
            "### Other forks\n"
            "- **Conditional sizing** — only run the inflation tilt when inflation momentum is clearly positive.\n"
            "- **More macro drivers** — add monetary-policy and risk-sentiment momentum (Brooks–Moskowitz use several).\n"
            "- **Combine with trend** — stack on [Trade-Winds](../../31-trade-winds/) for a fuller macro sleeve.\n\n"
            "PRs welcome — and the headline fork is the **real FRED run** itself, pre-registered here and "
            "waiting on a reliable macro fetch."
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


def build_quants():
    cells = [
        md(
            "# Barometer — a quantitative teardown 🔬\n"
            "### Macro-momentum Sharpe vs a null · the (gentle) cost wall & break-even · the inflation-regime split\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same seven "
            "beats, every claim with its number.* The steelman: the trend in fundamental macro data (growth, "
            "inflation) predicts asset returns across classes (Brooks–Moskowitz 2017), and tilting toward "
            "real assets when inflation rises hedges a portfolio in the regimes that hurt it (Neville et al. "
            "2021). We confirm the macro-momentum premium is `REAL` on the control (Sharpe 1.09, null −0.17), "
            "that it's `WEAK`/`FRAGILE` (modest, slow, episodic hedge), and we pre-register the real run.\n\n"
            "> ⚠️ **Real run pending a reliable FRED macro fetch.** The core executes on a synthetic "
            "cross-asset world; the real cross-asset/FRED run is in [`../docs/results.md`](../docs/results.md) "
            "(a **pre-registration** — the daily FRED series time out here), sources in "
            "[`../docs/references.md`](../docs/references.md). Mirrors [Study 27](../../27-steamroller/).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        md(
            "## Beat 0 · Verdict\n\n"
            "| Axis | Stamp | Why |\n"
            "|---|---|---|\n"
            f"| **Signal** — macro trend predicts returns? | 🟢 `REAL` on the level · 🟡 `WEAK` on the size | "
            f"Macro-momentum gross Sharpe **{S['mm_sh']}** (+{S['mm_ann']}%/yr), null **{S['mm_null']}**; "
            "literature ~0.4–0.8 with long flat stretches. |\n"
            f"| **Tradability** | 🟡 `FRAGILE` | Slow book — break-even **~{S['mm_be']} bp**, cost not the "
            "threat; modest Sharpe, long droughts, **episodic** inflation hedge. |\n"
            f"| **Real-tape run?** | ⚪ `Pre-reg` | FRED macro series time out / intermittent here; real run "
            "pending a reliable fetch ([`../docs/results.md`](../docs/results.md)). |\n\n"
            "> **In one sentence:** a real, slow, diversifying cross-asset macro premium whose inflation leg "
            "only earns its keep when inflation is rising — `REAL`-but-`WEAK`, `FRAGILE`, real run "
            "`PRE-REG`.\n\n"
            "*(This notebook executes on the synthetic control; the real numbers are pre-registered in "
            "[`../docs/results.md`](../docs/results.md).)*"
        ),

        md(
            "## Beat 1 · The claim, precisely\n\n"
            "Two latent macro states $g_t$ (growth), $\\pi_t$ (inflation) evolve as persistent, "
            "regime-switching AR(1) processes. The **macro momentum** is $\\Delta x_t = x_t - x_{t-1}$. "
            "Each asset $a$ loads on the *lagged* momentum through fixed signed betas:\n\n"
            "$$ r_{a,t} = s\\cdot\\big(\\beta^g_a\\,\\Delta g_{t-1} + \\beta^\\pi_a\\,\\Delta\\pi_{t-1}\\big)\\,\\sigma_a "
            "+ \\text{noise} $$\n\n"
            "The **macro-momentum book** weights each asset by its macro exposure dotted with the "
            "(z-scored, smoothed) current momentum, dollar-neutral, gross 1, lagged one month. The "
            "**inflation book** overweights real assets when $\\Delta\\pi>0$. Claim: both earn a positive "
            "premium. Null ($s=0$): assets are pure noise — nothing to predict."
        ),
        code(
            "for label, (rr, mm) in [('macro-driven', (r, m)), ('null', (r0, m0))]:\n"
            "    for kind in ('macro_momentum', 'inflation'):\n"
            "        g = strategy.summary(strategy.book_returns(rr, mm, kind=kind, cost_bps=0.0))\n"
            "        print(f\"{label:12} {kind:15} gross Sharpe {g['sharpe']:+6.2f}\")"
        ),
        md(
            "> 💡 **In plain words.** When the macro state really drives returns, both books make money "
            f"(macro-momentum **{S['mm_sh']}**, inflation **{S['inf_sh']}**); when it's pure noise, both are "
            f"flat (**{S['mm_null']}** / **{S['inf_null']}**). The apparatus sees the macro premium only when "
            "it's there — so a real-tape result would be about the *market*, not the method."
        ),

        md(
            "## Beat 2 · So what?\n\n"
            "Macro momentum (Brooks–Moskowitz 2017) is the *fundamental* cousin of price trend (Moskowitz–"
            "Ooi–Pedersen 2012, the desk's [Trade-Winds](../../31-trade-winds/)): both are slow, "
            "low-correlation, crisis-friendly premia. The pre-registered prediction: the macro book is "
            "**real but modest** (a ~0.5 Sharpe), cheap to run (low turnover), and its *inflation leg* is "
            "**conditional** — it should pay in rising-inflation regimes and drag otherwise (Neville et al. "
            "2021; Ang–Bekaert on regime-dependent returns). Beats 4–7 test exactly that."
        ),

        md(
            "## Beat 3 · Pre-registered protocol\n\n"
            "1. **Real?** `book_returns(cost_bps=0)` on control vs null, both books.\n"
            "2. **Tradable?** `strategy.turnover_ann`, `costs.breakeven_cost_bps`, `costs.cost_sweep`.\n"
            "3. **Conditional?** `extension.regime_split` — rising vs falling inflation.\n\n"
            "**Mirage line:** flat macro-momentum book on the control, *or* an inflation hedge that pays the "
            "same in both regimes ⇒ no conditional macro premium. (The real-tape HAC *t* is pre-registered "
            "in [`../docs/results.md`](../docs/results.md), pending the FRED fetch.)"
        ),

        md("## Beat 4 · The teardown\n\n### 4a · Real gross macro-momentum, flat null (the REAL call)"),
        code(
            "cmp = strategy.compare(r, m, cost_bps=0.0)\n"
            "for kind in ('macro_momentum', 'inflation'):\n"
            "    s = cmp[kind]\n"
            "    print(f\"{kind:15} Sharpe {s['sharpe']:+.2f}  ann {100*s['ann_return']:+.1f}%  maxDD {100*s['max_drawdown']:.0f}%  turnover {s['turnover_ann']:.1f}x/yr\")\n"
            "print('\\nReal-tape (PENDING, ../docs/results.md): macro-momentum Sharpe + HAC t, regime split — pre-registered, awaiting a reliable FRED fetch.')"
        ),
        md(
            "> 💡 **In plain words.** A 1.09 gross Sharpe on the control with a flat null is the apparatus "
            "working. The literature tempers the *magnitude*: on real data expect ~0.4–0.8 with long flat "
            "stretches — real, but a portfolio sleeve, not a standalone hero. Hence `REAL` on the level, "
            "`WEAK` on the size."
        ),

        md("### 4b · The (gentle) cost wall and break-even — cost is NOT the threat"),
        code(
            "cs = costs.cost_sweep(r, m, kind='macro_momentum')\n"
            "print('cost sweep (net Sharpe):'); print(cs.round(3).to_string())\n"
            "fig, ax = plt.subplots(); ax.axhline(0, color='#999', lw=.8)\n"
            "ax.plot(cs.index, cs['sharpe'], marker='o', color='#2ea44f')\n"
            "ax.set_xlabel('cost (bp/unit)'); ax.set_ylabel('net Sharpe'); ax.set_title('Slow macro book — barely dented by cost')\n"
            "print('\\nbreak-even macro-momentum:', round(costs.breakeven_cost_bps(r, m, 'macro_momentum')), 'bp  (vs realistic cross-asset costs ~1-5 bp)')"
        ),

        md("## Beat 5 · The verdict\n\n"
           f"- **`REAL` on the level / `WEAK` on the size** (4a): control Sharpe {S['mm_sh']} vs null "
           f"{S['mm_null']}; literature ~0.4–0.8, long droughts.\n"
           f"- **`FRAGILE`** (4b): break-even ~{S['mm_be']} bp (cheap to run), but modest, slow, and the "
           "inflation leg is episodic (Beat 7).\n"
           "- **Real-tape run? `PRE-REG`** — pending a reliable FRED macro fetch.\n\n"
           "> **Signal `REAL`/`WEAK` · Tradability `FRAGILE` · Real-tape run? `PRE-REG`** — a diversifying "
           "macro sleeve, proven on the control, pre-registered on the tape."),

        md("## Beat 6 · Could you trade it?\n\n"
           "- **Cheaply, but patiently.** Low turnover ⇒ costs don't kill it (the rare desk study where "
           "that's true); the cost is sitting through multi-year flat stretches.\n"
           "- **Capacity is large.** Liquid cross-asset proxies (index futures, commodities, TIPS, gold) — "
           "no microstructure ceiling, unlike the single-stock books.\n"
           "- **The inflation hedge is insurance.** It pays in rising-inflation regimes and drags "
           "otherwise — size it conditionally, not permanently (Beat 7)."),

        md(
            "## Beat 7 · Going further\n\n"
            "### 7a · Worked complement — does the inflation hedge pay when it's supposed to? ([`../docs/extension.md`](../docs/extension.md))\n"
            "Split the inflation book by regime — rising vs falling inflation — and ask whether the premium "
            "concentrates where the theory says it must."
        ),
        code(
            "sp = extension.regime_split(r, m, kind='inflation', cost_bps=0.0)\n"
            "display(sp.round(2))\n"
            "spm = extension.regime_split(r, m, kind='macro_momentum', cost_bps=0.0)\n"
            "print('macro-momentum book by regime (steadier, also rides growth):'); print(spm.round(2).to_string())\n"
            "fig, ax = plt.subplots()\n"
            "ax.bar(['rising','falling'], sp['sharpe'].values, color=['#2ea44f','#dab617'])\n"
            "ax.set_ylabel('Sharpe (inflation book)'); ax.set_title('The inflation hedge pays more in the regime it targets')"
        ),
        md(
            "> 💡 **In plain words.** The inflation book earns more when inflation is rising "
            f"(**{S['reg_inf_up_sh']}** vs **{S['reg_inf_dn_sh']}**) — it pays in the storm it's built for. "
            "It's not *free* insurance (the real-asset basket drifts up anyway), but the conditional tilt is "
            "real. The macro-momentum book, riding growth too, is steadier across both regimes — confirming "
            "the inflation leg is the episodic one."
        ),

        md(
            "### 7b · The honest pending real run\n"
            "The real cross-asset / FRED run — macro-momentum Sharpe + HAC *t*, turnover, and the regime "
            "split on the *actual* historical inflation episodes (the 1970s, 2008, 2021–22) — is "
            "**pre-registered** in [`../docs/results.md`](../docs/results.md), awaiting a reliable FRED "
            "macro fetch (the daily series time out in this environment). The apparatus, the null, and the "
            "regime split are fixed *before* those numbers exist — so the real run can confirm or refute, "
            "but not be tuned to taste.\n\n"
            "### 7c · Other forks\n"
            "- **Conditional sizing** — run the inflation tilt only when inflation momentum is clearly positive.\n"
            "- **More drivers** — add monetary-policy & risk-sentiment momentum (Brooks–Moskowitz use several).\n"
            "- **Stack with trend** — combine with [Trade-Winds](../../31-trade-winds/) for a fuller macro sleeve.\n\n"
            "PRs welcome — the headline fork is the **real FRED run** itself."
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
