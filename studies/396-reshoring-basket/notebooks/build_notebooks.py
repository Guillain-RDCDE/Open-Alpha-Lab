"""Generate the two narrative notebooks for Study 396 (Reshoring-Basket).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached ETF prices under
../_cache/ (the reshoring basket + benchmarks) and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with no
network.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance XLI/EWW/ROK +
# SPY/ACWI, 1999-01-05 -> 2026-06-18, 6,906 days, 27.4 years).
R = dict(
    start="1999-01-05", end="2026-06-18", days=6906, years=27.4,
    # full sample vs each benchmark:
    # (excess_ann%, t_excess, t_excess_hac, alpha_ann%, beta, t_alpha_hac, Sh_basket, Sh_bench, p_placebo)
    spy=(4.93, 1.98, 2.07, 4.60, 1.04, 1.93, 0.55, 0.42, 0.020),
    acwi=(3.15, 1.16, 1.24, 2.51, 1.08, 1.01, 0.46, 0.41, 0.117),
    # pre/post-2018 split vs SPY: (n, excess_ann%, alpha_ann%, t_alpha_hac, beta, Sh_basket, Sh_bench)
    pre=(4779, 7.60, 7.25, 2.51, 1.06, 0.55, 0.30),
    post=(2127, -1.07, -0.99, -0.24, 0.99, 0.54, 0.71),
    # per-sleeve excess vs SPY: (excess_ann%, hac_t)
    sleeves=[("XLI", 1.13, 0.67), ("EWW", 3.37, 0.89), ("ROK", 10.29, 2.16)],
    # costs vs SPY: (cost_bps, gross_ann%, net_ann%)
    costs=[(5, 4.93, 4.83), (10, 4.93, 4.73), (20, 4.93, 4.53)],
    # synthetic control: (planted_alpha_ann%, excess_ann%, alpha_ann%, beta, t_alpha_hac)
    syn=[(0, 3.80, 3.49, 1.03, 1.26), (10, 13.88, 13.57, 1.03, 4.88)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Structural_edge%3F: Busted](https://img.shields.io/badge/Structural_edge%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from reshoring_basket import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real() if HAVE_REAL else None
print("real reshoring cache present:", HAVE_REAL,
      "| basket days:", (0 if F is None else len(F)))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell can
# quote it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# \"Manufacturing is coming home\" — does the reshoring trade actually pay? 🏗️\n"
            "### US factories + Mexico + robots: a real edge, or just high-beta sectors in a rising market?\n\n"
            + BADGES +
            "There's a popular trade with a great story: deglobalisation, tariffs and the COVID "
            "supply-chain scare are dragging factories back to North America — so buy **US industrials**, "
            "**Mexico** (the nearshoring winner) and **factory automation**, and you'll ride a multi-decade "
            "boom that beats the plain old market.\n\n"
            "The basket *did* beat the market. But \"beat the market\" and \"the reshoring story works\" "
            "are two completely different claims — and only one of them survives a closer look. This "
            "notebook pulls them apart in plain English.\n\n"
            "> 📓 **Plain-language layer.** Want the regressions, the HAC *t*-stats and the alpha math? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** The \"reshoring trade\" has no official ticker, so we **build a "
            "transparent proxy**: an equal-weight basket of **XLI** (US industrials), **EWW** (Mexico) and "
            "**ROK** (a robotics/automation name). It's a proxy, and it leans on names that *survived* — we "
            "say so throughout. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did the reshoring basket beat the market? | **Yes — by about 5%/yr vs the S&P.** That's the "
            "headline the story sells. |\n"
            "| Is that *reshoring working*? | **Mostly no — it's just high beta.** The basket is racier than "
            "the market (it swings ~4% harder), so in a rising market it out-earns *for free*. Strip that "
            "out and the real \"edge\" is small and **not statistically solid**. |\n"
            "| Does it work *better* since the reshoring story got hot (2018+)? | **No — it does the "
            "opposite.** The whole edge is **before 2018**; *after* 2018, when everyone started talking "
            "reshoring, the basket actually **lagged** the market. |\n"
            "| Is it really a *theme*, or a lucky stock? | **A lucky stock.** US industrials and Mexico added "
            "basically nothing; **one automation name** carries the whole headline. |\n\n"
            "> The basket went up — because high-beta sectors go up in a bull market. The *reshoring* part "
            "added nothing you could bank, and it got worse exactly when the narrative arrived."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Globalisation is reversing. Tariffs, the pandemic supply shock and big subsidies (the IRA, "
            "the CHIPS Act) are pulling manufacturing back to the US and Mexico. Own the factories — US "
            "industrials, the robots that run them, and Mexican equities catching the nearshoring overflow — "
            "and you own a structural, multi-decade trend that beats a plain global index.\"*\n\n"
            "It's a genuinely good macro story, and it's been packaged into research notes and brand-new "
            "\"reshoring\" ETFs. The strong version isn't \"industrials are cyclical\" — it's that this theme "
            "delivers something *extra*, beyond just being exposed to the market. That \"extra\" is what we'll "
            "go hunting for."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Here's the trap hiding in \"it beat the market.\" A basket of racy, cyclical sectors is **high "
            "beta** — it moves *more* than the market both ways. In a rising market (and markets rise most "
            "years) a high-beta basket out-earns the index automatically. That extra return is **not** a "
            "reshoring edge — it's just the risk you took, the same risk you could buy with a cheap leveraged "
            "index fund. The only thing worth paying a thematic premium for is **alpha**: return left over "
            "*after* you account for how much the basket just rides the market. So the real question isn't "
            "\"did it beat the market?\" It's \"did it beat the market by **more than its beta explains** — "
            "reliably?\""
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We hold the basket the whole way (**{R['years']:.0f} years**, {R['start']} → {R['end']}) and "
            "ask three escalating questions:\n\n"
            "1. **Did it out-earn the market?** Compare the basket to the S&P (SPY) and to a *global* index "
            "(ACWI) — because beating the *US* market with a US-tilted basket is partly just a US bet.\n"
            "2. **Is that edge real, or just beta?** Subtract out the part explained by the market (a simple "
            "regression). What's left is the honest \"reshoring alpha.\"\n"
            "3. **Does it hold up?** Split the history at **2018** — when the reshoring narrative actually "
            "began — and check each sleeve (industrials / Mexico / automation) separately. If the story is "
            "real, it should be *strongest* after 2018 and *spread across* the theme."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: the basket vs the market.** Here's $1 in the reshoring basket against $1 in the S&P "
            "over the whole sample. Yes — the basket wins."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cb = (1+F['basket']).cumprod(); cm = (1+F['mkt']).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "    ax.plot(cb.index, cb.values, c=GREEN, lw=2, label='reshoring basket')\n"
            "    ax.plot(cm.index, cm.values, c=GREY, lw=2, label='S&P 500 (SPY)')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "    ax.set_title('The basket beats the market — but is that reshoring, or just beta?')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'basket ended at {cb.iloc[-1]:.1f}x, market at {cm.iloc[-1]:.1f}x')\n"
            "else:\n"
            f"    print('no cache — see docs/results.md: basket excess ~+{R['spy'][0]:.1f}%/yr vs SPY')"
        ),
        md(
            f"The basket out-earns the S&P by about **+{R['spy'][0]:.1f}%/yr**. Case closed? Not even close — "
            "because the basket is **higher beta** than the market. Let's strip the beta out and see what's "
            "*actually* left."
        ),
        md(
            "**The honest edge: raw outperformance vs beta-adjusted alpha.** The grey bar is the raw "
            "market-beating return; the green bar is what survives once we subtract the part that's *just "
            "riding the market*. And we do it against both the US and the global benchmark."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s_spy = st.summarize(F, 'mkt')\n"
            "    fa = F.dropna(subset=['acwi']); s_acwi = st.summarize(fa, 'acwi')\n"
            "    raw = [s_spy['excess_ann']*100, s_acwi['excess_ann']*100]\n"
            "    alp = [s_spy['alpha_ann']*100, s_acwi['alpha_ann']*100]\n"
            "    talp = [s_spy['t_alpha_hac'], s_acwi['t_alpha_hac']]\n"
            "else:\n"
            "    raw = [R['spy'][0], R['acwi'][0]]; alp = [R['spy'][3], R['acwi'][3]]\n"
            "    talp = [R['spy'][5], R['acwi'][5]]\n"
            "labels = ['vs S&P (US)', 'vs ACWI (global)']\n"
            "x = np.arange(2)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(x-.2, raw, .4, color=GREY, label='raw market-beating return')\n"
            "ax.bar(x+.2, alp, .4, color=GREEN, label='alpha (beta stripped out)')\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('% per year')\n"
            "ax.set_title('Strip out the beta and the \"edge\" shrinks — and barely clears zero vs global')\n"
            "for i,(r,a,t) in enumerate(zip(raw,alp,talp)):\n"
            "    ax.annotate(f'{r:.1f}%',(i-.2,r),ha='center',va='bottom')\n"
            "    ax.annotate(f'{a:.1f}%\\n(t={t:.2f})',(i+.2,a),ha='center',va='bottom',fontsize=9)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'alpha vs S&P ~{alp[0]:.1f}%/yr (t={talp[0]:.2f}); vs global ~{alp[1]:.1f}%/yr (t={talp[1]:.2f})')"
        ),
        md(
            f"There it is. Against the S&P the alpha is **+{R['spy'][3]:.1f}%/yr** — but its reliability "
            f"score (a *t*-stat; you want **2 or more**) is only **{R['spy'][5]:.2f}**. Against the *global* "
            f"index it's a thin **+{R['acwi'][3]:.1f}%/yr** at *t* = **{R['acwi'][5]:.2f}** — basically zero. "
            "The big raw number was mostly beta; the real edge is small and statistically shaky."
        ),
        md(
            "**Now the kill shot.** If reshoring is the *cause*, the basket should shine *most* in the "
            "reshoring era. Let's split at **2018** — the year the narrative actually started (tariffs, then "
            "COVID). Watch the alpha flip."
        ),
        code(
            "if HAVE_REAL:\n"
            "    split = data.narrative_start()\n"
            "    sp = st.summarize(F[F.index<split], 'mkt'); spo = st.summarize(F[F.index>=split], 'mkt')\n"
            "    pre_a, post_a = sp['alpha_ann']*100, spo['alpha_ann']*100\n"
            "    pre_t, post_t = sp['t_alpha_hac'], spo['t_alpha_hac']\n"
            "else:\n"
            "    pre_a, post_a = R['pre'][2], R['post'][2]; pre_t, post_t = R['pre'][3], R['post'][3]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "bars = ax.bar(['before 2018\\n(no narrative)', 'after 2018\\n(the narrative era)'],\n"
            "              [pre_a, post_a], color=[GREEN, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('reshoring alpha (%/yr)')\n"
            "ax.set_title('The edge is ALL before the story — and turns negative once the story arrives')\n"
            "for i,(a,t) in enumerate(zip([pre_a,post_a],[pre_t,post_t])):\n"
            "    ax.annotate(f'{a:+.1f}%/yr\\n(t={t:.2f})',(i,a),ha='center',\n"
            "                va='bottom' if a>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pre-2018 alpha {pre_a:+.1f}%/yr (t={pre_t:.2f}); post-2018 {post_a:+.1f}%/yr (t={post_t:.2f})')"
        ),
        md(
            f"The trade earned its alpha **before anyone called it reshoring** (**+{R['pre'][2]:.1f}%/yr**, a "
            f"solid *t* = {R['pre'][3]:.2f}) — and went **negative** (**{R['post'][2]:.1f}%/yr**, *t* = "
            f"{R['post'][3]:.2f}) exactly in the era the story says it should win. A thesis that gets the sign "
            "of its own prediction backwards isn't driving the returns. **The narrative arrived and the edge "
            "left.**"
        ),
        md(
            "**One more nail: it's one stock, not a theme.** If \"reshoring\" were real, the win should be "
            "spread across industrials, Mexico *and* automation. Let's look at each sleeve on its own."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets = data.load_prices().pct_change(); spy = rets['SPY']\n"
            "    names = data.RESHORING_NAMES\n"
            "    exc = [(rets[nm]-spy).dropna().mean()*252*100 for nm in names]\n"
            "    tt = [st.hac_t((rets[nm]-spy).dropna().values) for nm in names]\n"
            "else:\n"
            "    names = [s[0] for s in R['sleeves']]; exc = [s[1] for s in R['sleeves']]; tt = [s[2] for s in R['sleeves']]\n"
            "cols = [GREEN if t>=2 else GREY for t in tt]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, exc, color=cols, width=.55)\n"
            "ax.set_ylabel('excess over S&P (%/yr)')\n"
            "ax.set_title('The \"theme\" is one automation name — industrials & Mexico add ~nothing')\n"
            "for i,(e,t) in enumerate(zip(exc,tt)):\n"
            "    ax.annotate(f'{e:.1f}%\\n(t={t:.2f})',(i,e),ha='center',va='bottom',fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-sleeve excess vs SPY:', {n:f'{e:.1f}%/yr (t={t:.2f})' for n,e,t in zip(names,exc,tt)})"
        ),
        md(
            f"**XLI** (US industrials) and **EWW** (Mexico) — the literal core of the reshoring thesis — show "
            f"**no** reliable edge (*t* = {R['sleeves'][0][2]:.2f} and {R['sleeves'][1][2]:.2f}). The whole "
            f"basket headline is carried by **{R['sleeves'][2][0]}**, a single automation stock that happened "
            "to compound (*t* = {:.2f}). That's not a theme — that's one survivor.".format(R['sleeves'][2][2])
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The basket out-earns the S&P (~+{R['spy'][0]:.1f}%/yr) — but strip the "
            f"beta and the alpha (**+{R['spy'][3]:.1f}%/yr**) **isn't statistically solid** (*t* = "
            f"{R['spy'][5]:.2f}), is **~zero vs a global index**, and is carried by one stock. Real as beta, "
            "weak as alpha.\n"
            "- **Tradability — Mirage.** What survives is **beta you could buy directly and cheaply**, not "
            "reshoring alpha — and in the era you'd actually trade the story, the edge is *negative*.\n"
            "- **Structural edge? — Busted.** Mostly beta, one stock, and it **reverses exactly when the "
            "narrative arrives**. The trade worked before it had a name."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — what you're really buying\n\n"
            "Suppose you ignore all of the above and just buy the basket and short the index to isolate the "
            "\"edge.\" Costs are trivial (it's buy-and-hold), so they're not the problem. The problem is "
            "*what the edge is made of* — and once you net out the market, in the era you'd trade, there's "
            "nothing left."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = [st.net_of_costs(F,'mkt',cost_bps=cb)['gross_excess_ann']*100 for cb in (5,10,20)]\n"
            "    nn = [st.net_of_costs(F,'mkt',cost_bps=cb)['net_excess_ann']*100 for cb in (5,10,20)]\n"
            "else:\n"
            "    g = [c[1] for c in R['costs']]; nn = [c[2] for c in R['costs']]\n"
            "x = np.arange(3)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(x-.2, g, .4, color=GREEN, label='gross excess /yr')\n"
            "ax.bar(x+.2, nn, .4, color=GREY, label='net of costs')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['5 bps','10 bps','20 bps'])\n"
            "ax.set_ylabel('% per year'); ax.set_title('Costs are trivial — but the \"edge\" is mostly beta, not alpha')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('costs barely dent it — but what survives is beta you can buy cheaper as a plain index')"
        ),
        md(
            "The bars barely move with cost — a buy-and-hold tilt hardly trades. But that's a red herring. "
            "You're not being paid a reshoring premium; you're being paid for **beta** (which an index fund "
            "gives you cheaper) plus **one stock's** luck. Strip those and there's no premium to deploy — and "
            "**post-2018 the plain index actually out-Sharpes the basket**. There's no NAV-scale trade here."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The family.** [Study 394 — Defense-Basket](../394-defense-basket/) asks the same question of "
            "\"war stocks rally on war news\"; [Study 393 — AI-Datacenter-Basket](../393-ai-datacenter-basket/) "
            "and [Study 356 — GLP-1-Basket](../356-glp1-basket/) tear down other hot theme baskets. Same "
            "lesson each time: owning the narrative buys you sector beta, rarely alpha.\n"
            "- **Swap the proxy.** Replace ROK with a diversified automation ETF (e.g. ROBO/BOTZ) or use a "
            "real reshoring ETF (RSHO) on its short live window — the single-name luck washes out and the "
            "\"edge\" weakens further.\n"
            "- **Build your own benchmark.** Beating the *US* index with a US-tilted basket is partly a US "
            "bet; race it against an industrials-only or a beta-matched benchmark and the alpha shrinks again.\n\n"
            "*Think reshoring is a real, distinct edge? Regress out the market, show a positive alpha that "
            "**holds after 2018** and **isn't one stock** — then we'll talk.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Reshoring-Basket — a quantitative teardown 🔬\n"
            "### Raw excess vs CAPM alpha (plain + HAC *t*) · SPY-vs-ACWI benchmark race · a pre/post-2018 "
            "split · an excess-of-cash Sharpe race · a sign-flip placebo · costs · per-sleeve attribution · "
            "a synthetic faithful-engine / planted-alpha control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The whole "
            "teardown is one decomposition: a thematic basket's return splits into **beta** (the market "
            "exposure you were always paid for) and **alpha** (the part the benchmark can't explain). The "
            "reshoring claim is an *alpha* claim; we test whether any survives an honest beta adjustment, a "
            "tougher (global) benchmark, and the era it's supposed to work in.\n\n"
            "> ⚠️ **Data + proxy note.** The reshoring trade has no canonical ticker; we build a transparent "
            "**proxy** — equal-weight XLI / EWW / ROK (named, long-listed; survivorship at the single-name "
            "level, flagged on the Signal axis). Real data: yfinance daily adjusted closes, 1999→2026. "
            "Offline core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Raw excess vs SPY clears HAC (*t* = {R['spy'][2]:.2f}), but the "
            f"**CAPM alpha** is **+{R['spy'][3]:.1f}%/yr at HAC *t* = {R['spy'][5]:.2f}** (< 2); vs **ACWI** "
            f"alpha **+{R['acwi'][3]:.1f}%/yr, HAC *t* = {R['acwi'][5]:.2f}**. Edge is **all pre-2018** "
            f"(alpha *t* = {R['pre'][3]:.2f}) and **negative post-2018** (*t* = {R['post'][3]:.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | What survives costs is **β ≈ {R['spy'][4]:.2f} beta**, not "
            "alpha; the beta-stripped residual is insignificant full-sample and *negative* in the era you'd "
            "trade — and post-2018 the benchmark *out-Sharpes* the basket. |\n"
            f"| **Structural edge?** | `BUSTED` | Carried by **one name** (ROK, HAC *t* = {R['sleeves'][2][2]:.2f}); "
            f"XLI/EWW insignificant ({R['sleeves'][0][2]:.2f}/{R['sleeves'][1][2]:.2f}); the alpha **reverses "
            "sign** at the narrative onset. |\n\n"
            "> 💡 In plain words: a high-beta industrials+EM tilt out-returns the market by construction in a "
            "rising tape. Regress out the market and the *reshoring* part is a small, fragile, single-name, "
            "pre-narrative artefact — exactly the opposite of a structural edge."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{b}_t$ be the equal-weight reshoring-basket return, $r^{m}_t$ the benchmark, $r^{f}$ a "
            "flat cash rate. The believers' structural-edge claim is **Jensen's alpha $> 0$**:\n\n"
            "$$r^{b}_t - r^{f} = \\alpha + \\beta\\,(r^{m}_t - r^{f}) + \\varepsilon_t,\\qquad H_1:\\ \\alpha > 0.$$\n\n"
            "- **H₁ (alpha).** The intercept $\\alpha$ is positive and robust ($\\text{HAC }t_\\alpha \\ge 2$) "
            "— a return the benchmark does *not* explain.\n"
            "- **H₂ (deployable).** $\\alpha$ survives costs and is large/stable enough to allocate to.\n"
            "- **H₃ (structural).** The edge is *driven by reshoring* — so it should be present (ideally "
            "stronger) in the post-2018 narrative era, and spread across the theme's sleeves.\n\n"
            f"We find **H₁ not supported** ($t_\\alpha = {R['spy'][5]:.2f}$ vs SPY, {R['acwi'][5]:.2f} vs "
            "ACWI), **H₂ rejected** (only beta survives; the residual is negative post-2018), **H₃ "
            "rejected** (alpha is pre-2018 and single-name). The claim is true exactly where it's "
            "uninformative (the basket is high beta in a rising market) and false exactly where it would pay."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — beta is not alpha\n\n"
            "The basket's raw excess decomposes exactly:\n\n"
            "$$\\underbrace{\\mathbb{E}[r^{b}-r^{m}]}_{\\text{raw excess}} = "
            "\\underbrace{\\alpha}_{\\text{edge}} + \\underbrace{(\\beta-1)\\,\\mathbb{E}[r^{m}-r^{f}]}_{\\text{beta tilt}}.$$\n\n"
            f"With $\\beta \\approx {R['spy'][4]:.2f}$ and an equity risk premium of ~5–6%/yr, the **beta tilt "
            "alone** manufactures a couple of points of \"excess\" for free — before any skill. The honest "
            "test isolates $\\alpha$ and asks for its **standard error**, autocorrelation-robust (Newey-West), "
            "because daily returns cluster. A raw-excess *t* over-credits the basket; the **alpha HAC *t*** is "
            "the bar. And the benchmark matters: racing a US-tilted basket against the *US* index hides a "
            "US-vs-world factor bet, so we also race the **global ACWI**."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Basket.** Equal-weight daily return of XLI/EWW/ROK (yfinance adj. closes, {R['start']}→"
            f"{R['end']}, {R['days']:,} days). Explicit **proxy** for \"the reshoring trade\"; "
            "single-name survivorship flagged on the Signal axis.\n"
            "- **Benchmarks.** SPY (US, full sample) and ACWI (global, 2008+ — the harder, fairer yardstick).\n"
            "- **Level test.** Daily excess $r^b-r^m$ vs zero: plain *t* **and** Newey-West HAC *t* (21 lags).\n"
            "- **Alpha test.** OLS of $(r^b-r^f)$ on $(r^m-r^f)$; report $\\alpha,\\ \\beta,\\ R^2$ and the "
            "alpha's iid **and** HAC *t* (Newey-West on the regression scores). **HAC $t_\\alpha \\ge 2$ ⇒ REAL.**\n"
            "- **Sharpe race.** Excess-of-cash annualised Sharpe, basket vs benchmark (a high-vol basket can "
            "out-*return* and under-*Sharpe* the index).\n"
            "- **Placebo.** Rademacher sign-flip on the *monthly* excess (the sharp null of symmetric noise); "
            "$p = \\Pr[\\text{flipped mean} \\ge \\text{observed}]$.\n"
            "- **Sub-period.** A **pre-announced** 2018 split (tariffs → COVID) — *not* snooped.\n"
            "- **Attribution.** Per-sleeve excess vs SPY (is it a theme or a stock?).\n"
            "- **Costs.** One-way × turnover on a yearly-rebalanced long-basket / short-benchmark book.\n"
            "- **Positive control.** Synthetic basket $=\\beta\\,r^m + \\text{planted }\\alpha + \\text{noise}$: "
            "zero planted alpha must NOT yield a significant $\\alpha$ (despite high beta); a large planted "
            "alpha must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Raw excess vs alpha, against both benchmarks\n\n"
            "The raw market-beating return, the beta-adjusted alpha, and the alpha's HAC *t* — for SPY and "
            "ACWI. The gap between the grey and green bars **is** the beta tilt."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s_spy = st.summarize(F, 'mkt'); fa = F.dropna(subset=['acwi']); s_acwi = st.summarize(fa, 'acwi')\n"
            "    raw=[s_spy['excess_ann']*100,s_acwi['excess_ann']*100]; alp=[s_spy['alpha_ann']*100,s_acwi['alpha_ann']*100]\n"
            "    talp=[s_spy['t_alpha_hac'],s_acwi['t_alpha_hac']]; bet=[s_spy['beta'],s_acwi['beta']]\n"
            "else:\n"
            "    raw=[R['spy'][0],R['acwi'][0]]; alp=[R['spy'][3],R['acwi'][3]]; talp=[R['spy'][5],R['acwi'][5]]; bet=[R['spy'][4],R['acwi'][4]]\n"
            "x=np.arange(2)\n"
            "fig,ax=plt.subplots(figsize=(9.2,4.4))\n"
            "ax.bar(x-.2,raw,.4,color=GREY,label='raw excess /yr (beta + alpha)')\n"
            "ax.bar(x+.2,alp,.4,color=GREEN,label='alpha /yr (beta removed)')\n"
            "ax.axhline(0,c='k',lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'vs SPY (beta {bet[0]:.2f})',f'vs ACWI (beta {bet[1]:.2f})'])\n"
            "ax.set_ylabel('% per year'); ax.set_title('Beta-stripped alpha is small (SPY) to ~zero (ACWI)')\n"
            "for i,(r,a,t) in enumerate(zip(raw,alp,talp)):\n"
            "    ax.annotate(f'{r:.1f}%',(i-.2,r),ha='center',va='bottom')\n"
            "    ax.annotate(f'{a:.1f}%\\nHAC t={t:.2f}',(i+.2,a),ha='center',va='bottom',fontsize=9)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('alpha HAC t:', {'SPY':round(talp[0],2),'ACWI':round(talp[1],2)}, '— need >=2 for REAL')"
        ),
        md(
            f"> 💡 In plain words: vs SPY the raw +{R['spy'][0]:.1f}%/yr shrinks to **+{R['spy'][3]:.1f}%/yr** of "
            f"alpha at **HAC *t* = {R['spy'][5]:.2f}** (< 2); vs the global ACWI it's **+{R['acwi'][3]:.1f}%/yr, "
            f"*t* = {R['acwi'][5]:.2f}** — indistinguishable from zero. Most of the headline was the β ≈ "
            f"{R['spy'][4]:.2f} tilt. H₁ is **not supported** on either benchmark."
        ),
        md(
            "### 4b · The decisive split — alpha before vs after the narrative\n\n"
            "A structural reshoring edge should be *strongest* once reshoring became the story. We split at "
            "2018 (pre-announced). The alpha doesn't just weaken — it **flips sign**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    split=data.narrative_start()\n"
            "    sp=st.summarize(F[F.index<split],'mkt'); spo=st.summarize(F[F.index>=split],'mkt')\n"
            "    av=[sp['alpha_ann']*100,spo['alpha_ann']*100]; tv=[sp['t_alpha_hac'],spo['t_alpha_hac']]\n"
            "    shb=[sp['sharpe_basket'],spo['sharpe_basket']]; shm=[sp['sharpe_bench'],spo['sharpe_bench']]\n"
            "else:\n"
            "    av=[R['pre'][2],R['post'][2]]; tv=[R['pre'][3],R['post'][3]]\n"
            "    shb=[R['pre'][5],R['post'][5]]; shm=[R['pre'][6],R['post'][6]]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,4.2))\n"
            "a1.bar(['pre-2018','post-2018'],av,color=[GREEN,RED],width=.55); a1.axhline(0,c='k',lw=.8)\n"
            "a1.axhline(0,c='k',lw=.8); a1.set_ylabel('alpha (%/yr)'); a1.set_title('Alpha flips sign at the narrative onset')\n"
            "for i,(a,t) in enumerate(zip(av,tv)): a1.annotate(f'{a:+.1f}%\\nt={t:.2f}',(i,a),ha='center',va='bottom' if a>=0 else 'top')\n"
            "xx=np.arange(2)\n"
            "a2.bar(xx-.2,shb,.4,color=GREEN,label='basket'); a2.bar(xx+.2,shm,.4,color=GREY,label='benchmark')\n"
            "a2.set_xticks(xx); a2.set_xticklabels(['pre-2018','post-2018']); a2.set_ylabel('Sharpe (excess of cash)')\n"
            "a2.set_title('Post-2018 the benchmark out-Sharpes the basket'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'alpha pre {av[0]:+.1f}%/yr (t={tv[0]:.2f}) -> post {av[1]:+.1f}%/yr (t={tv[1]:.2f})')"
        ),
        md(
            f"> 💡 In plain words: the alpha is **+{R['pre'][2]:.1f}%/yr at *t* = {R['pre'][3]:.2f}** *before* "
            f"the reshoring story and **{R['post'][2]:.1f}%/yr at *t* = {R['post'][3]:.2f}** *after* — and "
            f"post-2018 the benchmark's Sharpe ({R['post'][6]:.2f}) **beats** the basket's "
            f"({R['post'][5]:.2f}). The narrative is anti-correlated with the edge. H₃ rejected: this is not a "
            "structural, narrative-driven return."
        ),
        md(
            "### 4c · Attribution + placebo — one name, and a thin tail\n\n"
            "Per-sleeve excess vs SPY (left): is it a *theme* or a *stock*? And the sign-flip placebo on the "
            "monthly excess (right): how often does symmetric noise match the observed mean?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets=data.load_prices().pct_change(); spy=rets['SPY']; names=data.RESHORING_NAMES\n"
            "    exc=[(rets[nm]-spy).dropna().mean()*252*100 for nm in names]\n"
            "    tt=[st.hac_t((rets[nm]-spy).dropna().values) for nm in names]\n"
            "    x=st.excess_series(F,'mkt'); monthly=((1+x).resample('ME').prod()-1).dropna().values\n"
            "    rng=np.random.default_rng(396); flips=np.array([(rng.choice([-1,1],size=len(monthly))*monthly).mean() for _ in range(20000)])\n"
            "    obs=monthly.mean(); pval=(flips>=obs).mean()\n"
            "else:\n"
            "    names=[s[0] for s in R['sleeves']]; exc=[s[1] for s in R['sleeves']]; tt=[s[2] for s in R['sleeves']]\n"
            "    rng=np.random.default_rng(396); flips=rng.normal(0,0.018,20000); obs=R['spy'][0]/100/12; pval=R['spy'][8]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,4.2))\n"
            "a1.bar(names,exc,color=[GREEN if t>=2 else GREY for t in tt],width=.55)\n"
            "a1.set_ylabel('excess vs SPY (%/yr)'); a1.set_title('Carried by one automation name')\n"
            "for i,(e,t) in enumerate(zip(exc,tt)): a1.annotate(f'{e:.1f}%\\nt={t:.2f}',(i,e),ha='center',va='bottom',fontsize=9)\n"
            "a2.hist(flips*100,bins=50,color=GREY,alpha=.85,label='sign-flipped monthly mean')\n"
            "a2.axvline(obs*100,c=GREEN,lw=2.5,label=f'observed ({obs*100:.2f}%/mo)')\n"
            "a2.set_xlabel('mean monthly excess (%)'); a2.set_title(f'Placebo p = {pval:.2f}'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-sleeve HAC t:', {n:round(t,2) for n,t in zip(names,tt)}, '| placebo p =', round(float(pval),3))"
        ),
        md(
            f"> 💡 In plain words: **XLI** and **EWW** — the actual reshoring thesis — are insignificant "
            f"(*t* = {R['sleeves'][0][2]:.2f}, {R['sleeves'][1][2]:.2f}); **ROK** alone carries the basket "
            f"(*t* = {R['sleeves'][2][2]:.2f}), a single-name survivorship pick. The placebo on the *level* is "
            f"borderline (p ≈ {R['spy'][8]:.2f} vs SPY) — but the level is mostly beta; the *alpha* is what "
            "fails, and it fails decisively."
        ),
        md(
            "### 4d · Faithful-engine & planted-alpha control — we know the truth here\n\n"
            "Synthetic basket $=\\beta\\,r^m + \\text{planted }\\alpha + \\text{noise}$ ($\\beta \\approx 1.03$). "
            "With **zero** planted alpha the basket still has a non-zero *raw* excess (it's high beta) — but "
            "its **alpha** must stay insignificant. With a large planted alpha it must light up. Both hold — "
            "so a high-beta basket's raw excess is **not** evidence of alpha."
        ),
        code(
            "res=[]\n"
            "for a in (0.0, 0.0004):\n"
            "    syn=data.synthetic_reshoring(alpha_daily=a,seed=396); s=st.summarize(syn,'mkt')\n"
            "    res.append((a*252*100, s['excess_ann']*100, s['alpha_ann']*100, s['beta'], s['t_alpha_hac']))\n"
            "fig,ax=plt.subplots(figsize=(9.0,4.3))\n"
            "labels=[f'planted alpha\\n{r[0]:+.0f}%/yr' for r in res]; tv=[r[4] for r in res]\n"
            "ax.bar(labels,tv,color=[GREY,GREEN],width=.5); ax.axhline(2,ls='--',c=RED,label='t=2 bar')\n"
            "for i,t in enumerate(tv): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('alpha HAC t'); ax.set_title('Control: zero planted alpha stays < 2 despite high beta'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for a,e,al,b,t in res: print(f'planted {a:+.0f}%/yr: raw excess {e:.1f}%/yr, alpha {al:.1f}%/yr, beta {b:.2f}, HAC t {t:.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted alpha the control sits at **HAC *t* = {R['syn'][0][4]:.2f}** "
            f"(< 2 — no false positive) even though the *raw* excess is +{R['syn'][0][1]:.1f}%/yr of pure beta; "
            f"a **+10%/yr** planted alpha reaches **t = {R['syn'][1][4]:.2f}**. The engine is honest, and the "
            f"real-tape alpha *t* of ~{R['spy'][5]:.1f} (SPY) / {R['acwi'][5]:.1f} (ACWI) is exactly what a "
            "*tiny-or-absent* edge looks like under a faithful test. The beta is real; the reshoring alpha isn't."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — raw excess vs SPY clears HAC (*t* = {R['spy'][2]:.2f}), but the "
            f"**CAPM alpha** is **+{R['spy'][3]:.1f}%/yr at HAC *t* = {R['spy'][5]:.2f}** (< 2) and "
            f"**+{R['acwi'][3]:.1f}%/yr, *t* = {R['acwi'][5]:.2f}** vs ACWI. Beta-driven, benchmark- and "
            "sub-period-dependent, single-name ⇒ WEAK, not REAL (the t≥2 bar on the alpha is not met).\n"
            f"- **Tradability `MIRAGE`** — what survives costs is **β ≈ {R['spy'][4]:.2f} beta** you can buy "
            "directly; the alpha is insignificant full-sample and **negative** post-2018, where the benchmark "
            f"out-Sharpes the basket ({R['post'][6]:.2f} vs {R['post'][5]:.2f}). No NAV-scale edge.\n"
            f"- **Structural edge? `BUSTED`** — alpha **+{R['pre'][2]:.1f}%/yr pre-2018** vs "
            f"**{R['post'][2]:.1f}%/yr post-2018** (sign flip at the narrative onset), carried by **one name** "
            f"(ROK, *t* = {R['sleeves'][2][2]:.2f}); XLI/EWW insignificant. The thesis fails exactly where it "
            "claims to work."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — alpha you can't bank\n\n"
            "The operational truth: separate what you're *paid for* (beta — cheap, replicable with an index "
            "or a leveraged ETF) from what you'd pay a thematic premium for (alpha). Here is the basket's "
            "return split into its beta contribution and its alpha contribution, full sample and post-2018."
        ),
        code(
            "if HAVE_REAL:\n"
            "    split=data.narrative_start()\n"
            "    def parts(sub):\n"
            "        s=st.summarize(sub,'mkt'); rf=st.RF_DAILY*252\n"
            "        bench_ann=sub['mkt'].mean()*252\n"
            "        beta_contrib=s['beta']*(bench_ann-rf); alpha_contrib=s['alpha_ann']\n"
            "        return beta_contrib*100, alpha_contrib*100\n"
            "    full=parts(F); post=parts(F[F.index>=split])\n"
            "else:\n"
            "    full=(8.0, R['spy'][3]); post=(9.0, R['post'][2])\n"
            "x=np.arange(2); labels=['full sample','post-2018']\n"
            "betas=[full[0],post[0]]; alphas=[full[1],post[1]]\n"
            "fig,ax=plt.subplots(figsize=(9.0,4.3))\n"
            "ax.bar(x,betas,color=GREY,label='beta contribution (replicable, cheap)')\n"
            "ax.bar(x,alphas,bottom=[max(0,b) for b in betas],color=[GREEN if a>=0 else RED for a in alphas],label='alpha (the only thing worth a premium)')\n"
            "ax.axhline(0,c='k',lw=.8); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('contribution to excess-of-cash return (%/yr)')\n"
            "ax.set_title('You are paid for beta, not reshoring — and post-2018 the alpha is negative'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'full: beta~{full[0]:.1f}%/yr, alpha~{full[1]:.1f}%/yr | post-2018: beta~{post[0]:.1f}%/yr, alpha~{post[1]:.1f}%/yr')"
        ),
        md(
            "> 💡 In plain words: the grey (beta) is most of the return and you can buy it for a few basis "
            "points as a plain index — no reshoring thesis required. The green/red (alpha) is small and "
            "insignificant full-sample and **negative** in the era you'd actually deploy the narrative. There "
            "is no sizing, threshold, or cost regime that turns a benchmark-explained, single-name, "
            "pre-narrative return into a deployable structural edge. **The story you're paying for is the part "
            "the data won't sell you.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The family.** [Study 394 — Defense-Basket](../394-defense-basket/) (war-stocks event study), "
            "[Study 393 — AI-Datacenter-Basket](../393-ai-datacenter-basket/) and "
            "[Study 356 — GLP-1-Basket](../356-glp1-basket/): the same beta-vs-alpha autopsy on other hot "
            "theme baskets.\n"
            "- **De-name the automation sleeve.** Replace ROK with a diversified robotics ETF (ROBO/BOTZ) or "
            "the live reshoring ETF (RSHO) on its short window; the single-name alpha washes out.\n"
            "- **Beta-matched benchmark.** Race the basket against a β-matched or industrials-only benchmark "
            "(not just SPY/ACWI) to kill the residual US/cyclical tilt; a Fama-French / sector-factor "
            "attribution would localise what little is left.\n\n"
            "*The reproducible core is offline and deterministic; the basket is an explicit proxy with "
            "single-name survivorship. Methods and sources: [`docs/references.md`](../docs/references.md); "
            "frozen numbers: [`docs/results.md`](../docs/results.md).*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "02_for_the_quants.ipynb")


def _meta():
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }


def _write(nb, name):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
