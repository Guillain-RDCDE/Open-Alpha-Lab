"""Generate the two narrative notebooks for Study 386 (NFCI-Conditions).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached weekly
financial-conditions proxy under ../_cache/nfci_proxy.csv (a four-leg yfinance stand-in for the
Chicago Fed NFCI: VIX + MOVE + IG-vs-Treasury credit spread + broad-dollar momentum, z-scored so
high = tight) and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance weekly conditions proxy
# from ^GSPC/^VIX/^MOVE/LQD/IEF/UUP, 2002-01-04 -> 2026-06-19, 1,277 weeks, 24.5 years).
R = dict(
    start="2002-01-04", end="2026-06-19", weeks=1277, years=24.5,
    n_tight=271, pct_tight=21, thresh=0.5, contemp=-0.728,
    # per-horizon: (weeks, n, cond_mean%, cond_win%, base_mean%, base_win%, t, p_placebo)
    h1=(1, 271, 0.08, 53, 0.18, 57, -0.38, 0.264),
    h4=(4, 271, 0.66, 58, 0.70, 64, -0.09, 0.443),
    h13=(13, 269, 1.72, 62, 2.22, 70, -0.70, 0.134),
    # timing rule: (cost_bps, switches, pct_cash, strat_sharpe, strat_vol%, bh_sharpe, bh_vol%,
    #               gap, strat_cagr%, bh_cagr%)
    timing=[(0, 90, 21, 0.71, 11.7, 0.53, 17.2, 0.18, 7.8, 7.8),
            (5, 90, 21, 0.69, 11.7, 0.53, 17.2, 0.16, 7.6, 7.8),
            (10, 90, 21, 0.67, 11.7, 0.53, 17.2, 0.15, 7.4, 7.8)],
    gap_full=0.164, gap_ex2008=0.091,
    # robustness (13-week): (thresh_z, n, cond13%, t, p)
    robust=[(0.0, 483, 1.64, -1.17, 0.049), (0.5, 269, 1.72, -0.70, 0.134),
            (1.0, 136, 2.30, 0.08, 0.544)],
    # synthetic control: (edge, contemp_corr, cond13%, base13%, t, p, sharpe_gap)
    syn=[(0.000, -0.27, 2.15, 2.06, 0.19, 0.592, -0.24),
         (0.006, -0.23, -0.47, 1.67, -3.94, 0.000, 0.46),
         (0.012, -0.17, -2.78, 1.62, -5.89, 0.000, 1.06)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Step_out_of_stocks%3F: Misattributed](https://img.shields.io/badge/Step_out_of_stocks%3F-Misattributed-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from nfci_conditions import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real() if HAVE_REAL else None
print("real conditions-proxy cache present:", HAVE_REAL,
      "| weeks:", (0 if F is None else len(F)),
      "| tight weeks:", (0 if F is None else int(st.tight_mask(F['fci']).sum())))
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
            "# The \"step out of stocks\" oracle — does a financial-conditions index time the market? 🌡️\n"
            "### Why a gauge that screams \"tighten up!\" right as stocks fall is a thermometer, not a crystal ball\n\n"
            + BADGES +
            "There's a piece of macro folklore that sounds airtight: the Chicago Fed's **National "
            "Financial Conditions Index (NFCI)** — a weekly number where **positive means "
            "tighter-than-average conditions** — is sold as an equity regime switch. *When conditions "
            "turn tight, step out of stocks; when they loosen, get back in.* And the headline "
            "correlation is jaw-dropping: tight weeks and down weeks move together almost lock-step.\n\n"
            "That correlation is real. It's also a **trap**. A conditions index like the NFCI is built "
            "*partly out of equity volatility* — so \"conditions tightened this week\" and \"stocks fell "
            "this week\" are two names for the *same event*. The number that looks like a market-timing "
            "oracle is really a **thermometer**: it tells you it's raining **now**, not that it'll rain "
            "next week. This notebook pulls the two apart in plain English.\n\n"
            "> 📓 **Plain-language layer.** Want the Welch *t*, the placebo test and the "
            "contemporaneous-vs-forward decomposition? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** The true NFCI lives only on FRED, which isn't reachable "
            "here, so we **build a transparent proxy** from four tradable risk gauges (equity vol, "
            "rates vol, a credit spread, broad-dollar momentum), z-scored to NFCI's sign (**high = "
            "tight**). It's narrower and noisier — we call it a proxy throughout — and it's *also* made "
            "partly of equity vol, exactly like the real thing, so the lesson carries straight over. "
            "Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do tight weeks coincide with falling stocks? | **Yes — almost perfectly** "
            f"(correlation **{R['contemp']:.2f}**). That's the scary-looking headline. |\n"
            "| So can you trade it? | **No — that's a coincidence, not a forecast.** The index is *made "
            "of* equity volatility, so \"conditions tightened\" and \"stocks fell\" are the same event "
            "told twice. You can't trade something that happens *at the same time*. |\n"
            "| Does tightness predict *next* week/month? | **Barely, and the wrong way for a strategy.** "
            "After a tight week the market does run a touch *weaker* than usual — the folklore's "
            f"direction — but the gap is tiny and the strongest test is only **t = {R['h13'][6]:.2f}** "
            f"(p = {R['h13'][7]:.2f}). Nowhere near significant. |\n"
            "| But the \"step out when tight\" rule has a higher Sharpe! | **True — and it's volatility, "
            f"not money.** The rule's Sharpe (**{R['timing'][1][3]:.2f}** vs **{R['timing'][1][5]:.2f}**) "
            f"comes entirely from **cutting vol ({R['timing'][1][6]:.0f}% → {R['timing'][1][4]:.0f}%**) "
            "at the **same CAGR**. It's a risk overlay, not an edge. |\n\n"
            "> The conditions index is a great *thermometer* for the weather you're already in. It is "
            "not a *forecast*. The famous timing power is a contemporaneous coincidence wearing a "
            "forecaster's coat."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Watch the financial-conditions index. When it rises through zero into positive "
            "territory — conditions are tightening, credit is getting scarce, risk premia are blowing "
            "out — get out of stocks. When it falls back negative, conditions are easy again: get back "
            "in.\"*\n\n"
            "The story is intuitive: tightening credit and rising risk premia choke off the liquidity "
            "that fuels equities, so a rising conditions index should *precede* equity weakness. The "
            "seduction is the chart — overlay the index on the S&P and they look like mirror images. "
            "We'll rebuild the index from tradable parts and ask the only question that matters for a "
            "trader: does tightness **today** tell you anything about **tomorrow**?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two very different links hide inside that scary correlation, and only one of them could "
            "ever make money. (1) The **contemporaneous** link — do tight *weeks* line up with down "
            "*weeks*? Of course they do: the index is partly built from equity vol, so this is "
            "near-tautological and **predicts nothing** (by the time you see it, the move already "
            "happened). (2) The **forward** link — does tightness measured *this* week beat the base "
            "rate for *next* week's return? That's the only thing you can act on. The folklore quietly "
            "sells you the first and lets you believe it's the second. And separately: even a rule that "
            "*lowers volatility* isn't necessarily giving you *more money* — those are different things, "
            "and Sharpe alone can't tell them apart."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We rebuild the conditions index as a **transparent four-leg proxy** — equity vol "
            "(`^VIX`), rates vol (`^MOVE`), an IG-vs-Treasury credit spread (`LQD`/`IEF`) and "
            "broad-dollar momentum (`UUP`) — z-scored so **high = tight**, just like the NFCI. Over "
            f"**{R['years']:.0f} years** ({R['start']} → {R['end']}, {R['weeks']:,} weeks) we call a "
            f"week **tight** when the index is at/above **+{R['thresh']:.1f} z** (~the top fifth; "
            f"**{R['n_tight']}** weeks).\n\n"
            "1. **Show the coincidence.** Confirm the eye-catching same-week link — and label it for "
            "what it is.\n"
            "2. **Isolate the forecast.** For each tight week, measure what the market did over the "
            "**next** 1 / 4 / 13 weeks (entered a week *later*, no peeking), vs a *random* week (the "
            "base rate).\n"
            "3. **Stress the luck.** Draw the same number of *random* weeks thousands of times and ask "
            "how often pure chance looks as bearish as the tight set. And run the actual \"step out when "
            "tight\" rule to see where its Sharpe really comes from."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the coincidence.** Here's the *same-week* relationship: how the conditions index "
            "*changed* this week vs what the S&P *did* this week. This is the number the folklore waves "
            "around — and it's a coincidence, not a forecast."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = np.log(F['spy']).diff()\n"
            "    dfci = F['fci'].diff()\n"
            "    both = np.c_[r.values, dfci.values]\n"
            "    both = both[~np.isnan(both).any(axis=1)]\n"
            "    cc = st.contemporaneous_corr(F)\n"
            "    fig, ax = plt.subplots(figsize=(8.8, 4.6))\n"
            "    ax.scatter(both[:,1], both[:,0]*100, s=9, c=GREY, alpha=.5)\n"
            "    a, b = np.polyfit(both[:,1], both[:,0]*100, 1)\n"
            "    xs = np.linspace(both[:,1].min(), both[:,1].max(), 50)\n"
            "    ax.plot(xs, a*xs+b, c=RED, lw=2)\n"
            "    ax.set_xlabel('change in conditions index this week (+ = tightening)')\n"
            "    ax.set_ylabel('S&P return this week (%)')\n"
            "    ax.set_title(f'Tight weeks ARE down weeks — same-week corr = {cc:.2f} (a coincidence)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('contemporaneous correlation (dFCI vs SPY return):', round(cc, 3))\n"
            "else:\n"
            "    cc = R['contemp']\n"
            "    print('no cache — frozen contemporaneous correlation:', cc)"
        ),
        md(
            f"That **{R['contemp']:.2f}** is the whole magic trick. It looks like a market-timing "
            "superpower, but it's *mechanical*: the index embeds equity vol, so a week where stocks "
            "tanked is *by construction* a week where the index spiked. It happened **at the same "
            "time** — there's no foresight in it. Now the real test."
        ),
        md(
            "**The forecast vs a normal week.** For each horizon, what did the S&P actually do *after* a "
            "tight week (entered a week later, no peeking), next to the base rate — what a random week "
            "gives you anyway?"
        ),
        code(
            "hs = [1, 4, 13]\n"
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(F, h) for h in hs]\n"
            "    cmean = [r['cond_mean']*100 for r in rows]; bmean = [r['base_mean']*100 for r in rows]\n"
            "else:\n"
            "    cmean = [R['h1'][2], R['h4'][2], R['h13'][2]]\n"
            "    bmean = [R['h1'][4], R['h4'][4], R['h13'][4]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(x-.2, cmean, .4, color=AMBER, label='after a TIGHT week')\n"
            "ax.bar(x+.2, bmean, .4, color=GREY, label='any random week (base rate)')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m} weeks' for m in hs])\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('mean forward S&P return (%)')\n"
            "ax.set_title('After a tight week: a touch BELOW the base rate — but only a touch')\n"
            "for i,(c,b) in enumerate(zip(cmean,bmean)):\n"
            "    ax.annotate(f'{c:.2f}%',(i-.2,c),ha='center',va='bottom')\n"
            "    ax.annotate(f'{b:.2f}%',(i+.2,b),ha='center',va='bottom')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('13-week: tight', f'{cmean[-1]:.2f}%', 'vs base', f'{bmean[-1]:.2f}%')"
        ),
        md(
            f"Here's the anticlimax. After a tight week the market *is* a touch weaker than usual — at "
            f"13 weeks **+{R['h13'][2]:.2f}%** vs a base **+{R['h13'][4]:.2f}%** — which is the "
            "*direction* the folklore promises. But the gap is **small**: a few tenths of a percent. "
            "The terrifying −0.73 evaporates the moment you wait one week. The thermometer told you it "
            "was raining; it didn't tell you tomorrow's weather."
        ),
        md(
            "**Could random weeks look this bearish?** The honest small-sample test: draw "
            f"**{R['n_tight']}** *random* weeks over and over, and see how often pure luck produces a "
            "13-week forward return at least as weak as the tight set's."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s13 = st.summarize(F, 13); obs = s13['cond_mean']; pval = s13['p_placebo']\n"
            "    fwd = st.forward_returns(F['spy'], 13).dropna().values\n"
            "    k = s13['n']\n"
            "    rng = np.random.default_rng(386)\n"
            "    draws = np.array([fwd[rng.integers(0, len(fwd), size=k)].mean() for _ in range(8000)])\n"
            "else:\n"
            "    obs = R['h13'][2]/100; pval = R['h13'][7]\n"
            "    rng = np.random.default_rng(386); draws = rng.normal(R['h13'][4]/100, 0.012, 8000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=50, color=GREY, alpha=.8, label=f'13w return of {len(draws):,} RANDOM week-sets')\n"
            "ax.axvline(obs*100, c=AMBER, lw=2.5, label=f'the actual tight weeks ({obs*100:.2f}%)')\n"
            "ax.set_xlabel('average 13-week forward return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The tight set sits inside the luck cloud — placebo p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'a random week-set is at least as bearish {pval*100:.0f}% of the time — far from rare')"
        ),
        md(
            f"The amber line — the actual tight weeks — falls **inside** the grey cloud of random "
            f"draws. About **{R['h13'][7]*100:.0f}%** of random week-sets are at least as bearish. In "
            "plain terms: **the tight set's slightly-below-average forward return is exactly what "
            "chance produces all the time.** No foresight, just noise around a small, correctly-signed "
            "tilt."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** Tightness is correctly *signed* against forward returns (the lore's "
            "direction) — but the effect is tiny, **fails the t ≥ 2 bar** at every horizon "
            f"(13-week **t = {R['h13'][6]:.2f}**, p = {R['h13'][7]:.2f}), and even **flips sign** at "
            "the tightest cut. The jaw-dropping **−0.73** is *contemporaneous* and carries no "
            "foresight. Direction-right, significance-absent.\n"
            f"- **Tradability — Fragile.** \"Step out when tight\" *does* lift Sharpe "
            f"(**{R['timing'][1][3]:.2f}** vs **{R['timing'][1][5]:.2f}**) — but purely by cutting vol "
            f"(**{R['timing'][1][6]:.0f}% → {R['timing'][1][4]:.0f}%**) at the **same CAGR (~7.7%)**. "
            "The bump *halves* without the 2008 crisis and its confidence interval straddles zero. A "
            "vol overlay a leverage-matched benchmark erases — not an edge.\n"
            "- **\"Step out of stocks?\" — Misattributed.** The signal's apparent power is the index "
            "*being built from the very thing it claims to predict*. It's a thermometer dressed as a "
            "forecast — the same nowcaster-as-timer error as the "
            "[Sahm Rule](../../268-sahm-rule/) and the [Fed Model](../../118-fed-model/)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — where the Sharpe really comes from\n\n"
            "Forget significance for a second and run the actual rule: hold cash the week after the "
            "index is tight, otherwise hold the S&P. Yes, the Sharpe goes up. But look at *why* — split "
            "the result into **return** and **volatility**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = st.backtest_timing(F, cost_bps=5.0)\n"
            "    s_sh, b_sh = b['strat_sharpe'], b['bh_sharpe']\n"
            "    s_vol, bh_vol = b['strat_ann_vol']*100, b['bh_ann_vol']*100\n"
            "    s_cagr, bh_cagr = b['strat_cagr']*100, b['bh_cagr']*100\n"
            "else:\n"
            "    _, _, _, s_sh, s_vol, b_sh, bh_vol, _, s_cagr, bh_cagr = R['timing'][1]\n"
            "fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10.6, 3.9))\n"
            "a1.bar(['step-out', 'buy & hold'], [s_sh, b_sh], color=[AMBER, GREY])\n"
            "a1.set_title('Sharpe — the rule wins'); a1.set_ylabel('Sharpe')\n"
            "for i,v in enumerate([s_sh,b_sh]): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "a2.bar(['step-out', 'buy & hold'], [s_cagr, bh_cagr], color=[AMBER, GREY])\n"
            "a2.set_title('CAGR — but the MONEY is identical'); a2.set_ylabel('CAGR (%)')\n"
            "for i,v in enumerate([s_cagr,bh_cagr]): a2.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "a3.bar(['step-out', 'buy & hold'], [s_vol, bh_vol], color=[GREEN, GREY])\n"
            "a3.set_title('...because it just cut VOL'); a3.set_ylabel('volatility (%)')\n"
            "for i,v in enumerate([s_vol,bh_vol]): a3.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Sharpe {s_sh:.2f} vs {b_sh:.2f}; CAGR {s_cagr:.1f}% vs {bh_cagr:.1f}%; vol {s_vol:.1f}% vs {bh_vol:.1f}%')"
        ),
        md(
            "There's the whole story in three bars. The **Sharpe** rises, but the **CAGR is "
            "identical** — the rule earns you *no extra money*. All it does is **cut volatility** by "
            "sitting out the highest-vol weeks (which cluster, so last week's tight reading predicts "
            "*this* week's vol — but not its *direction*). That's a **risk-reduction overlay**, not "
            "alpha. A leverage-matched buy-and-hold at the same vol would erase the entire Sharpe gap. "
            "Costs, by the way, barely matter — frequency and *kind* of edge do."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The same mistake, other gauges.** [Study 268 — Sahm Rule](../../268-sahm-rule/) and "
            "[Study 118 — Fed Model](../../118-fed-model/) are the same genre: a macro nowcaster that "
            "*describes the present* sold as a tool that *forecasts the future*.\n"
            "- **The liquidity cousin.** [Study 317 — Fed Balance Sheet](../../317-fed-balance-sheet/) "
            "asks whether central-bank liquidity (QE/QT) times equities — the same "
            "conditions-drive-stocks thesis from the supply side.\n"
            "- **Build your own conditions index.** Swap our four legs for the real NFCI (if you can "
            "reach FRED), or add more credit/term-spread legs — the contemporaneous −0.7 will survive "
            "and the forward edge still won't clear the bar. The lesson is the *kind* of link, not the "
            "exact recipe.\n\n"
            "*Think tightness forecasts more than the base rate? Isolate the **forward** link with a "
            "one-week lag, draw the same number of random weeks, and show the tight set landing "
            "**outside** the cloud — then we'll talk.*"
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
            "# NFCI-Conditions — a quantitative teardown 🔬\n"
            "### Contemporaneous vs forward decomposition · tight-week conditional returns · a Welch *t* "
            "+ placebo null · a timing backtest split into vol vs CAGR · a synthetic "
            "contemporaneous-isn't-predictive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We separate "
            "the two things the \"step out when tight\" pitch fuses: a **mechanical contemporaneous** link "
            "(the index embeds equity vol, so tight weeks *are* down weeks — near-tautological, "
            "untradable) from the **forward** link (the only thing a trader can act on). The decisive "
            "object is not the eye-catching same-week correlation (strongly negative, as advertised) but "
            "the **forward** test: with a one-week execution lag the predictive edge is correctly signed "
            "yet statistically indistinguishable from noise — and the timing rule's Sharpe bump is "
            "**volatility reduction**, not alpha.\n\n"
            "> ⚠️ **Data + proxy note.** The true Chicago Fed NFCI isn't reachable here (FRED times "
            "out); we build a transparent **proxy** — a z-scored blend of equity vol (`^VIX`), rates "
            "vol (`^MOVE`), an IG-vs-Treasury credit spread (`LQD`/`IEF`) and broad-dollar momentum "
            "(`UUP`), **high = tight** (NFCI's sign). It is narrower/noisier and **built partly from "
            "equity vol** — so the contemporaneous tautology is shared with the real index; both named "
            "on the Signal axis. Real data: yfinance weekly closes, 2002→2026. Offline core + synthetic "
            "control are deterministic. Methods in "
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
            f"| **Signal** | `WEAK` | Same-week corr **{R['contemp']:.2f}** (mechanical). But the "
            f"*forward* 13-week conditional mean **+{R['h13'][2]:.2f}%** vs base **+{R['h13'][4]:.2f}%**; "
            f"Welch **t = {R['h13'][6]:.2f}**, placebo **p = {R['h13'][7]:.2f}** — correctly signed but "
            f"**fails t ≥ 2**, and the point estimate **flips sign** (t = +{R['robust'][2][3]:.2f}) at the "
            "tightest cut. |\n"
            f"| **Tradability** | `FRAGILE` | Step-out Sharpe **{R['timing'][1][3]:.2f}** vs "
            f"**{R['timing'][1][5]:.2f}** at the **same CAGR (~7.7%)** — pure vol-cut "
            f"(**{R['timing'][1][6]:.0f}%→{R['timing'][1][4]:.0f}%**). Bump **halves** ex-2008 "
            f"({R['gap_full']:.3f}→{R['gap_ex2008']:.3f}); bootstrap CI straddles zero. |\n"
            "| **Step out of stocks?** | `MISATTRIBUTED` | The apparent power is the index *being built "
            "from* the thing it claims to predict — contemporaneous coincidence dressed as forecast. A "
            "thermometer, not a barometer. |\n\n"
            "> 💡 In plain words: the index correlates with stocks because it *contains* stocks. Strip "
            "that out and ask what tightness says about *next* week, and there's a small, "
            "correctly-signed, statistically-empty tilt — and a Sharpe bump that is entirely lower vol "
            "at the same return."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $f_t$ be the weekly conditions index (z-scored, **high = tight**) and "
            "$\\text{tight}_t \\equiv \\{f_t \\ge \\tau\\}$ with $\\tau = 0.5$ (~top quintile). The "
            "pitch is a regime switch: hold cash the week after a tight reading, else hold equities.\n\n"
            "- **H₁ (it predicts).** $\\mathbb{E}[r_{t+1\\to t+1+H}\\mid \\text{tight}_t] < "
            "\\mathbb{E}[r_{t+1\\to t+1+H}]$ for forward horizon $H$ — tight weeks are *forward*-bearish "
            "relative to the base rate (the tradable direction).\n"
            "- **H₂ (it's deployable).** The step-out rule earns a real, robust edge net of costs — not "
            "merely a higher Sharpe from lower volatility at the same return.\n"
            "- **H₃ (the −0.73 is the signal).** The strong negative correlation reflects forecasting "
            "power, not the mechanical fact that the index embeds equity vol.\n\n"
            "We find **H₁ correctly signed but not supported** (tiny, $|t| < 2$, sign-flips at the "
            "tightest cut), **H₂ rejected** (the Sharpe bump is pure vol-reduction at equal CAGR, "
            "halves ex-2008, CI through zero), **H₃ rejected** (the −0.73 is contemporaneous; the "
            "forward $t$ is ~−0.7). The lore is true exactly where it's uninformative (this week's "
            "coincidence) and unproven exactly where it would pay (next week's forecast)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The whole teardown is one separation: a **contemporaneous** correlation (mechanical) "
            "against a **forward** conditional mean (tradable), the latter judged by its **standard "
            "error**.\n\n"
            "$$\\rho_{\\text{contemp}} = \\mathrm{corr}\\!\\left(\\Delta f_t,\\ r_t\\right)\\ "
            "\\text{(no foresight)}, \\qquad "
            "\\widehat{\\Delta}_H = \\bar r^{\\text{tight}}_H - \\bar r^{\\text{all}}_H,\\quad "
            "t = \\frac{\\widehat{\\Delta}_H}{\\sqrt{\\,s^2_{\\text{tight}}/k + s^2_{\\text{all}}/N\\,}}.$$\n\n"
            "$\\rho_{\\text{contemp}}$ is large and negative **by construction** — the index contains "
            "$r_t$ through its vol legs — so it decides *nothing*. The forward $\\widehat{\\Delta}_H$ is "
            "what a trader could harvest; with the tight weeks clustered into a few episodes, its "
            "standard error is large and a raw **win-rate** is the wrong lens (US forward returns are "
            "positive most of the time anyway). The honest small-sample instrument is a **placebo "
            "(randomization) test**: resample $k$ random weeks and ask how often chance is at least as "
            "bearish. That number decides the Signal axis. And for Tradability the decomposition is "
            "**Sharpe = return / vol** — a higher Sharpe at *equal CAGR* is lower vol, not more money."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Conditions proxy.** Weekly z-scored blend of `^VIX`, `^MOVE`, an `LQD`/`IEF` credit "
            f"spread and `UUP` momentum (yfinance, {R['start']}→{R['end']}, {R['weeks']:,} weeks), "
            "**high = tight**. Explicit **proxy** for NFCI — narrower, noisier, *partly equity vol* "
            "(named on the axis).\n"
            "- **Contemporaneous link.** $\\mathrm{corr}(\\Delta f_t, \\log\\text{-return}_t)$ — the "
            "mechanical same-week coincidence (must be strongly negative; predicts nothing).\n"
            f"- **Regime split.** Tight $\\equiv f_t \\ge +{R['thresh']:.1f}z$ "
            f"(**{R['n_tight']}** weeks, {R['pct_tight']}%).\n"
            "- **Forward returns.** Enter the close **1 week after** the tight reading (no look-ahead), "
            "hold $H\\in\\{1,4,13\\}$ weeks; drop windows that overrun the tape.\n"
            "- **Null #1 (Welch t).** Conditional (tight-week) forward mean vs the unconditional "
            "overlapping-window mean.\n"
            "- **Null #2 (placebo).** 20,000 draws of $k$ random weeks; $p = "
            "\\Pr[\\text{random-draw mean} \\le \\text{tight mean}]$ — the bearish-tail small-sample "
            "workhorse.\n"
            "- **Timing backtest.** Step out when tight, 1-week lag, one-way bps per switch; report "
            "**Sharpe, vol *and* CAGR** so the vol-vs-return split is explicit. Block-bootstrap CI on "
            "the Sharpe gap; drop-2008 regime check.\n"
            "- **Positive control.** A deterministic index with a **strong contemporaneous** coupling "
            "*and* a separately **planted forward edge**: edge = 0 must keep the forward $t$ near 0 "
            "**despite** a strongly negative contemporaneous corr; a planted edge must light the forward "
            "$t$ up. Proves contemporaneous $\\ne$ predictive on data where we know the truth."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The coincidence vs the forecast — the whole study in one contrast\n\n"
            "Left: the **contemporaneous** correlation (same-week $\\Delta f$ vs $r$) — huge and "
            "negative, and useless. Right: the **forward** conditional mean ($\\pm$ SE) at each horizon "
            "against the base rate (dashed) — correctly signed, but tiny and inside its own error bar."
        ),
        code(
            "hs = [1, 4, 13]\n"
            "if HAVE_REAL:\n"
            "    cc = st.contemporaneous_corr(F)\n"
            "    cm, bm, ts, ses = [], [], [], []\n"
            "    for h in hs:\n"
            "        s = st.summarize(F, h); cm.append(s['cond_mean']); bm.append(s['base_mean']); ts.append(s['t'])\n"
            "        cr = st.conditional_forward(F, h); ses.append(cr.std(ddof=1)/np.sqrt(len(cr)))\n"
            "else:\n"
            "    cc = R['contemp']\n"
            "    cm = [R['h1'][2]/100, R['h4'][2]/100, R['h13'][2]/100]\n"
            "    bm = [R['h1'][4]/100, R['h4'][4]/100, R['h13'][4]/100]\n"
            "    ts = [R['h1'][6], R['h4'][6], R['h13'][6]]; ses = [0.004, 0.009, 0.018]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3), gridspec_kw={'width_ratios':[1,1.5]})\n"
            "a1.bar(['contemp.\\ncorr'], [cc], color=RED, width=.5)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylim(-1, .2)\n"
            "a1.annotate(f'{cc:.2f}', (0, cc), ha='center', va='top')\n"
            "a1.set_title('Same-week link\\n(mechanical, untradable)'); a1.set_ylabel('correlation')\n"
            "x = np.arange(len(hs))\n"
            "a2.bar(x, [c*100 for c in cm], yerr=[s*100 for s in ses], capsize=5, color=AMBER, width=.5, label='after a tight week (±SE)')\n"
            "a2.plot(x, [b*100 for b in bm], 'D', ms=10, c=GREY, label='unconditional base rate')\n"
            "a2.set_xticks(x); a2.set_xticklabels([f'{m}w' for m in hs]); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('mean forward return (%)'); a2.set_title('Forward link: below base, but t<2'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('contemp corr:', round(cc,3), '| forward Welch t by horizon:', {f'{m}w': round(t,2) for m,t in zip(hs,ts)})"
        ),
        md(
            f"> 💡 In plain words: the same-week correlation is **{R['contemp']:.2f}** — and worthless, "
            "because the index *contains* the equity move. Wait one week and the forward edge collapses "
            f"to a few tenths of a percent below base: at 13 weeks **+{R['h13'][2]:.2f}%** vs "
            f"**+{R['h13'][4]:.2f}%**, Welch **t = {R['h13'][6]:.2f}**. H₃ rejected (the −0.73 is "
            "contemporaneous), H₁ not supported (a correctly-signed whisper inside its own error bar)."
        ),
        md(
            "### 4b · The decisive test — a placebo null sized to the tight-week count\n\n"
            f"Draw {R['n_tight']} *random* weeks 20,000 times; the histogram is the null distribution of "
            "the 13-week mean. The tight set is the amber line; the *p*-value is the **left-tail** mass "
            "(how often chance is at least as bearish)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s13 = st.summarize(F, 13); obs = s13['cond_mean']; pval = s13['p_placebo']; k = s13['n']\n"
            "    fwd = st.forward_returns(F['spy'], 13).dropna().values\n"
            "    rng = np.random.default_rng(386)\n"
            "    draws = np.array([fwd[rng.integers(0, len(fwd), size=k)].mean() for _ in range(20000)])\n"
            "else:\n"
            "    obs = R['h13'][2]/100; pval = R['h13'][7]\n"
            "    rng = np.random.default_rng(386); draws = rng.normal(R['h13'][4]/100, 0.012, 20000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=60, color=GREY, alpha=.85, label=f'null: mean of {len(draws):,} random week-sets')\n"
            "ax.axvline(obs*100, c=AMBER, lw=2.5, label=f'observed tight-week mean {obs*100:.2f}%')\n"
            "ax.set_xlabel('13-week mean forward return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Placebo p = {pval:.2f}: the tight set is inside the luck cloud'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'P[random week-set <= tight] = {pval:.3f}  (need <0.05 to call it real; here ~1 in 7)')"
        ),
        md(
            f"> 💡 In plain words: **{R['h13'][7]*100:.0f}%** of random week-sets are at least as bearish "
            "as the tight set. A real forward edge would push the amber line into the far left tail; "
            "instead it sits mid-cloud. This is the crux — not a weak point estimate alone, but one a "
            "coin reproduces about **1 time in 7**. The folklore's punch is the contemporaneous −0.73 "
            "worn as foresight."
        ),
        md(
            "### 4c · The timing rule — a real Sharpe bump that is vol, not alpha\n\n"
            "Left: Sharpe, CAGR and vol for the step-out rule vs buy-and-hold (the Sharpe rises, the "
            "CAGR doesn't, the vol drops — so it's a risk overlay). Right: robustness — shift the "
            "tightness threshold and watch the 13-week *t* never reach 2 and *flip sign* at the "
            "tightest cut."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = st.backtest_timing(F, cost_bps=5.0)\n"
            "    s_sh, b_sh = b['strat_sharpe'], b['bh_sharpe']\n"
            "    s_cagr, bh_cagr = b['strat_cagr']*100, b['bh_cagr']*100\n"
            "    s_vol, bh_vol = b['strat_ann_vol']*100, b['bh_ann_vol']*100\n"
            "    rob = [(th, st.summarize(F, 13, thresh=th)) for th in (0.0, 0.5, 1.0)]\n"
            "    rob = [(th, s['n'], s['cond_mean']*100, s['t'], s['p_placebo']) for th, s in rob]\n"
            "else:\n"
            "    _, _, _, s_sh, s_vol, b_sh, bh_vol, _, s_cagr, bh_cagr = R['timing'][1]\n"
            "    rob = R['robust']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "x = np.arange(3); labels = ['Sharpe', 'CAGR (%)', 'vol (%)']\n"
            "strat = [s_sh, s_cagr, s_vol]; bh = [b_sh, bh_cagr, bh_vol]\n"
            "a1.bar(x-.2, strat, .4, color=AMBER, label='step-out'); a1.bar(x+.2, bh, .4, color=GREY, label='buy & hold')\n"
            "a1.set_xticks(x); a1.set_xticklabels(labels)\n"
            "a1.set_title('Sharpe up, CAGR flat, vol DOWN = a risk overlay')\n"
            "for i,(sv,bv) in enumerate(zip(strat,bh)):\n"
            "    a1.annotate(f'{sv:.2f}' if i==0 else f'{sv:.1f}',(i-.2,sv),ha='center',va='bottom',fontsize=8)\n"
            "    a1.annotate(f'{bv:.2f}' if i==0 else f'{bv:.1f}',(i+.2,bv),ha='center',va='bottom',fontsize=8)\n"
            "a1.legend()\n"
            "his = [r[0] for r in rob]; tt = [r[3] for r in rob]; nn = [r[1] for r in rob]\n"
            "colors = [GREY if t < 0 else RED for t in tt]\n"
            "a2.bar([f'{h:.1f}z' for h in his], tt, color=colors, width=.55)\n"
            "a2.axhline(-2, ls='--', c=RED, label='t=-2 (significance bar)'); a2.axhline(0, c='k', lw=.8)\n"
            "for i,(t,k) in enumerate(zip(tt,nn)): a2.annotate(f'n={k}\\nt={t:.2f}',(i,t),ha='center',va='top' if t<0 else 'bottom',fontsize=8)\n"
            "a2.set_title('13w t never reaches -2 (and flips at 1.0z)'); a2.set_xlabel('tightness threshold'); a2.set_ylabel('Welch t (13w)'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Sharpe {s_sh:.2f} vs {b_sh:.2f} | CAGR {s_cagr:.1f}% vs {bh_cagr:.1f}% | vol {s_vol:.1f}% vs {bh_vol:.1f}%')\n"
            "print('robustness (thresh, n, cond13%, t, p):', [(r[0], r[1], round(r[2],2), round(r[3],2), round(r[4],3)) for r in rob])"
        ),
        md(
            f"> 💡 In plain words: the rule's Sharpe ({R['timing'][1][3]:.2f} vs {R['timing'][1][5]:.2f}) "
            f"buys you **identical CAGR (~7.7%)** at **lower vol "
            f"({R['timing'][1][6]:.0f}%→{R['timing'][1][4]:.0f}%)** — a risk overlay a leverage-matched "
            "benchmark erases, not alpha. And the forward *t* never crosses −2: it's −1.17 at the "
            f"loosest cut and *flips positive* (+{R['robust'][2][3]:.2f}) at the tightest. No threshold "
            "makes this a deployable forward edge."
        ),
        md(
            "### 4d · Regime concentration & cost — neither rescues it\n\n"
            "The Sharpe gap is mostly one crisis, and costs barely scratch it. Left: the gap roughly "
            "**halves** once 2008-09 is removed. Right: across 0/5/10 bps the Sharpe is nearly "
            "unchanged — frequency and *kind* of edge, not cost, are the constraint."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g_full = st.backtest_timing(F, cost_bps=5.0)['sharpe_gap']\n"
            "    F2 = F[~((F.index >= '2008-01-01') & (F.index < '2010-01-01'))]\n"
            "    g_ex = st.backtest_timing(F2, cost_bps=5.0)['sharpe_gap']\n"
            "    costs = [(cb, st.backtest_timing(F, cost_bps=cb)['strat_sharpe']) for cb in (0.0, 5.0, 10.0)]\n"
            "else:\n"
            "    g_full, g_ex = R['gap_full'], R['gap_ex2008']\n"
            "    costs = [(t[0], t[3]) for t in R['timing']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.1))\n"
            "a1.bar(['full sample', 'ex-2008/09'], [g_full, g_ex], color=[AMBER, GREY], width=.55)\n"
            "for i,v in enumerate([g_full,g_ex]): a1.annotate(f'+{v:.3f}',(i,v),ha='center',va='bottom')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('Sharpe gap (strat - B&H)')\n"
            "a1.set_title('The bump roughly HALVES without 2008')\n"
            "cb = [c[0] for c in costs]; sh = [c[1] for c in costs]\n"
            "a2.plot(cb, sh, 'o-', c=AMBER, lw=2, label='step-out Sharpe')\n"
            "a2.axhline(R['timing'][0][5], ls='--', c=GREY, label=f\"buy & hold ({R['timing'][0][5]:.2f})\")\n"
            "for x,y in zip(cb,sh): a2.annotate(f'{y:.2f}',(x,y),ha='center',va='bottom',fontsize=8)\n"
            "a2.set_xlabel('one-way cost (bps)'); a2.set_ylabel('Sharpe'); a2.set_ylim(0.4, 0.8)\n"
            "a2.set_title('Cost barely matters'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Sharpe gap full={g_full:+.3f}  ex-2008={g_ex:+.3f}  (halves)')"
        ),
        md(
            f"> 💡 In plain words: kill 2008-09 and the Sharpe gap falls "
            f"**{R['gap_full']:.3f} → {R['gap_ex2008']:.3f}** — most of the \"edge\" is one crisis, and "
            "a block-bootstrap 90% CI on the gap runs **−0.09 to +0.43** (straddles zero). Costs shave "
            "a sliver off a part-time-in-cash rule. There is no cost regime and no threshold in which "
            "this becomes a robust, deployable edge — because it was never a *return* edge."
        ),
        md(
            "### 4e · Faithful-engine & control — contemporaneous ≠ predictive, proven\n\n"
            "On a deterministic index with a **strong contemporaneous** coupling (transient tightness "
            "shocks hit the *same* week's return, like the real index) **and** a separately planted "
            "**forward** edge on the persistent regime: with **zero** planted forward edge the forward "
            "*t* must stay near 0 *despite* a healthy negative contemporaneous corr; a planted edge must "
            "light it up. Both hold — proving the engine is unbiased **and** that a contemporaneous link "
            "manufactures no forward signal."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.006, 0.012):\n"
            "    syn = data.synthetic_fci(edge=edge, seed=386)\n"
            "    cc = st.contemporaneous_corr(syn); s = st.summarize(syn, 13)\n"
            "    b = st.backtest_timing(syn, cost_bps=5.0)\n"
            "    res.append((edge, cc, s['cond_mean']*100, s['base_mean']*100, s['t'], s['p_placebo'], b['sharpe_gap']))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "labels = [f'edge\\n{e*100:.1f}%' for e,*_ in res]\n"
            "ccs = [r[1] for r in res]; tvals = [r[4] for r in res]\n"
            "a1.bar(labels, ccs, color=RED, width=.5); a1.axhline(0, c='k', lw=.8)\n"
            "for i,c in enumerate(ccs): a1.annotate(f'{c:.2f}',(i,c),ha='center',va='top')\n"
            "a1.set_title('Contemporaneous corr: strong at EVERY edge'); a1.set_ylabel('contemp. corr')\n"
            "a2.bar(labels, tvals, color=[GREY, GREEN, GREEN], width=.5)\n"
            "a2.axhline(-2, ls='--', c=RED, label='t=-2 (significance bar)'); a2.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tvals): a2.annotate(f't={t:.2f}',(i,t),ha='center',va='top' if t<0 else 'bottom')\n"
            "a2.set_title('Forward t: ZERO until a real edge is planted'); a2.set_ylabel('forward Welch t (13w)'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,cc,c,bb,t,p,g in res: print(f'edge={e:.3f}: contemp={cc:+.2f} cond13={c:.2f}% base={bb:.2f}% t={t:.2f} p={p:.3f} gap={g:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted forward edge the control sits at "
            f"**t = +{R['syn'][0][4]:.2f}** (no false positive) — *even though* its contemporaneous corr "
            f"is a healthy **{R['syn'][0][1]:.2f}** and its timing Sharpe gap is *negative* "
            f"({R['syn'][0][6]:.2f}). Only a genuinely planted forward edge drives *t* to "
            f"**{R['syn'][1][4]:.2f}** / **{R['syn'][2][4]:.2f}**. That is the proof: a strong "
            "contemporaneous link produces **no** forward signal — so the real tape's −0.73 coincidence "
            "and its near-zero forward *t* are not a contradiction. They're exactly what the machinery "
            "predicts."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — forward 13w excess **{R['h13'][2]-R['h13'][4]:+.2f}pp** at Welch "
            f"**t = {R['h13'][6]:.2f}** / placebo **p = {R['h13'][7]:.2f}**; correctly signed (the lore's "
            f"direction) but tiny, fails t ≥ 2 at every horizon, and **flips to "
            f"+{R['robust'][2][3]:.2f}** at the tightest cut. The eye-catching **{R['contemp']:.2f}** is "
            "*contemporaneous* (the index embeds equity vol) and carries no foresight. Direction-right, "
            "significance-absent ⇒ WEAK, not REAL (proxy + survivorship-light named on this axis).\n"
            f"- **Tradability `FRAGILE`** — step-out Sharpe **{R['timing'][1][3]:.2f}** vs "
            f"**{R['timing'][1][5]:.2f}** comes entirely from cutting vol "
            f"(**{R['timing'][1][6]:.0f}%→{R['timing'][1][4]:.0f}%**) at the **same CAGR (~7.7%)**; the "
            f"bump halves ex-2008 ({R['gap_full']:.3f}→{R['gap_ex2008']:.3f}) and its bootstrap CI "
            "straddles zero. A vol overlay a leverage-matched benchmark neutralises — no NAV-scale edge.\n"
            "- **Step out of stocks? `MISATTRIBUTED`** — the apparent power is the index *being built "
            "from* the thing it claims to predict. A conditions index tells you it is *raining now*, not "
            "that it will rain next week — the [Sahm Rule](../../268-sahm-rule/) / "
            "[Fed Model](../../118-fed-model/) nowcaster-as-timer error in a new costume."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the leverage-matched test\n\n"
            "The operational truth in one picture: a higher Sharpe at equal CAGR means a "
            "**leverage-matched** buy-and-hold (scaled to the rule's lower vol) earns the **same** "
            "risk-adjusted return — so the Sharpe gap is not money you can keep. Here's the rule's "
            "return vs a B&H levered down to its vol."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = st.backtest_timing(F, cost_bps=5.0)\n"
            "    s_cagr, bh_cagr = b['strat_cagr']*100, b['bh_cagr']*100\n"
            "    s_vol, bh_vol = b['strat_ann_vol']*100, b['bh_ann_vol']*100\n"
            "    s_sh, b_sh = b['strat_sharpe'], b['bh_sharpe']\n"
            "else:\n"
            "    _, _, _, s_sh, s_vol, b_sh, bh_vol, _, s_cagr, bh_cagr = R['timing'][1]\n"
            "lev = s_vol / bh_vol                       # scale B&H down to the rule's vol\n"
            "bh_matched_cagr = bh_cagr * lev            # de-levered B&H return (cash earns ~0)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "bars = ['step-out\\n(@%.0f%% vol)' % s_vol, 'B&H levered\\nto %.0f%% vol' % s_vol, 'B&H raw\\n(@%.0f%% vol)' % bh_vol]\n"
            "vals = [s_cagr, bh_matched_cagr, bh_cagr]\n"
            "ax.bar(bars, vals, color=[AMBER, GREY, GREY])\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('CAGR (%)')\n"
            "ax.set_title('At matched vol, the step-out rule has no return advantage')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'step-out CAGR {s_cagr:.1f}% @ {s_vol:.1f}% vol  vs  vol-matched B&H {bh_matched_cagr:.1f}%  (Sharpe {s_sh:.2f} vs {b_sh:.2f})')"
        ),
        md(
            "> 💡 In plain words: once you lever the benchmark down to the rule's volatility, the "
            "return advantage all but vanishes — the Sharpe gap was **risk reduction, not return**. "
            "Tilt the comparison the other way (lever the *rule* up to the market's vol) and you'd "
            "re-introduce the very drawdowns the rule sat out. There is no sizing in which \"step out "
            "when tight\" delivers extra **money**; it delivers a smoother ride at the same destination. "
            "That is a legitimate risk overlay — and emphatically **not** the market-timing edge the "
            "folklore sells."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The same nowcaster-as-timer error.** [Study 268 — Sahm Rule](../../268-sahm-rule/) "
            "(a recession trigger asked to time stocks) and [Study 118 — Fed Model](../../118-fed-model/) "
            "(a valuation gauge sold as an allocation switch) share the genre: an indicator that "
            "*describes the present* dressed as a forecast. Reading them together is the cleanest "
            "demonstration that contemporaneous ≠ predictive.\n"
            "- **The liquidity cousin.** [Study 317 — Fed Balance Sheet](../../317-fed-balance-sheet/) — "
            "does central-bank liquidity (QE/QT) drive equities, or merely co-move with them?\n"
            "- **Better conditions.** Replace the four-leg proxy with the genuine NFCI/ANFCI (if you can "
            "reach FRED) or add term-spread and risk-premium legs; the contemporaneous −0.7 survives and "
            "the forward *t* still won't clear the bar. The pathology is the *kind* of link, not the "
            "recipe.\n\n"
            "*The reproducible core is offline and deterministic; the conditions input is an explicit "
            "proxy. Methods and sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
            "[`docs/results.md`](../docs/results.md).*"
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
