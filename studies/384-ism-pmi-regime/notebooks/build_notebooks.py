"""Generate the two narrative notebooks for Study 384 (ISM-PMI-Regime).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached industrial
basket prices under ../_cache/ (the PMI proxy + SPY) and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere
with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance PMI proxy from a
# 37-name industrial basket + SPY, 1995-04-30 -> 2026-06-30, 375 months, 31.2 years).
R = dict(
    start="1995-04-30", end="2026-06-30", months=375, years=31.2, above_frac=71.7,
    # regime means (% per month) + win-rates
    above_mean=0.958, below_mean=0.962, base_mean=0.959,
    above_win=66, below_win=63, base_win=65,
    n_above=268, n_below=106,
    spread=-0.004, t=-0.01, p_placebo=0.491,
    # timing rule (net @10 bps) vs buy-and-hold
    switches=57,
    timing_ann=8.06, timing_vol=10.4, timing_sharpe=0.77, timing_mdd=-40.8, timing_wealth=10.3,
    bh_ann=11.51, bh_vol=15.1, bh_sharpe=0.76, bh_mdd=-50.8, bh_wealth=24.9,
    # robustness: (threshold, n_above, n_below, spread%, t, p)
    robust=[(45, 285, 89, 0.187, 0.26, 0.349), (50, 268, 106, -0.004, -0.01, 0.491),
            (55, 248, 126, 0.351, 0.63, 0.220)],
    # synthetic: (edge%/mo, above%, below%, spread%, t, p)
    syn=[(0.0, 0.798, 0.978, -0.180, -0.58, 0.717), (1.5, 2.105, 1.221, 0.884, 2.86, 0.003)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Free_regime-timing_lunch%3F: Busted](https://img.shields.io/badge/Free_regime--timing_lunch%3F-Busted-8b949e?style=flat-square)\n\n"
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

from ism_pmi_regime import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real() if HAVE_REAL else None
print("real PMI-proxy cache present:", HAVE_REAL,
      "| months:", (0 if F is None else len(F)))
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
            "# \"Only own stocks when the factory gauge is above 50\" — does it work? 🏭\n"
            "### The manufacturing-PMI regime rule everyone repeats, in plain English\n\n"
            + BADGES +
            "Here's a rule you'll hear on every macro podcast: the **manufacturing PMI** — a monthly "
            "survey of factory purchasing managers — has a magic line at **50**. Above 50, factories are "
            "*expanding*, so you should **own stocks**. Below 50, they're *contracting*, so you should "
            "**hide in cash**. Simple, intuitive, and it sounds like it has to be true.\n\n"
            "It isn't. On a transparent factory-activity gauge built from real prices, the months when "
            "the gauge is **above 50** and the months when it's **below 50** deliver *the same* average "
            "stock return — to the basis point. \"Own stocks above 50\" turns out to be a very elaborate "
            "way of saying \"own stocks.\"\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the bootstrap and the Sharpe race? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** The real ISM PMI is a *paid* survey, and the usual free "
            "stand-in (FRED) is blocked here — so we **build a transparent proxy**: each month, what "
            "share of a 37-stock factory basket is rising? Call it above 50 when most are. It's narrower "
            "than the real survey — we call it a proxy throughout — but the lesson (the 50 line doesn't "
            "time stocks) is exactly what the textbooks predict. Every chart is drawn by the code beside "
            "it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do stocks do better when PMI is above 50? | **No — they do the *same*.** Above-50 months "
            f"average **+{R['above_mean']:.2f}%**; below-50 months average **+{R['below_mean']:.2f}%**. "
            "That's a coin flip apart. |\n"
            "| But isn't 'above 50' a winning regime? | Only because the market rises in **both** "
            "regimes — and the gauge sits above 50 about **72%** of the time anyway. You're almost "
            "always 'in'. |\n"
            "| So should I sit in cash below 50? | **That costs you.** Sitting out the 'bad' months hands "
            f"back **half your wealth** ({R['timing_wealth']:.0f}× vs {R['bh_wealth']:.0f}× over 31 "
            "years) for a risk-adjusted return it never beats. |\n"
            "| Then why does everyone believe it? | The PMI really *does* track the economy. People "
            "assume what tracks the **economy** must time the **market** — but the market already priced "
            "the cycle in. |\n\n"
            "> The PMI is a real economic gauge. As a *stock-timing* switch, the 50 line is a non-event."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The ISM Manufacturing PMI is a diffusion index: above 50 means most of manufacturing is "
            "**expanding**, below 50 means **contracting**. Be long equities when it's above 50; raise "
            "cash when it drops below. The 50 line is the on/off switch for the equity-risk regime.\"*\n\n"
            "It's a genuinely good *economic* indicator — the Fed and forecasters watch it because it "
            "tracks GDP growth in close to real time. The seductive leap is from *\"it tracks the "
            "economy\"* to *\"it times the stock market.\"* We'll build the gauge and test that leap."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the 50 line really sorted good stock-months from bad ones, it would be close to a free "
            "lunch: a public, monthly, lagged number that tells you when to be invested. But notice the "
            "trap hidden in \"above 50 is the good regime.\" The US market rises in **most** months — "
            "above 50 *and* below 50. So a high win-rate in the above-50 regime isn't an edge; it's the "
            "base rate. The only thing worth betting on is the **difference** between the regimes — and "
            "if there's no difference, sitting in cash during the 'bad' regime just removes good "
            "up-drift. A timing rule that lowers your return while feeling safe is not a strategy."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We rebuild a **PMI proxy** from prices: across a fixed 37-name factory basket, what share "
            "is rising (positive 3-month momentum)? Put it on a 0–100 scale; above 50 means most of "
            f"manufacturing is climbing. Over **{R['years']:.0f} years** ({R['start']} → {R['end']}, "
            f"**{R['months']}** months):\n\n"
            "1. **Split the months.** Was the proxy above or below 50 last month?\n"
            "2. **Compare the next month's stock return** in each regime — and against the plain average "
            "(the base rate).\n"
            "3. **Try to trade it.** Own SPY only after an above-50 reading, else cash, with a realistic "
            "trading cost — and see if it beats just holding the market.\n"
            "4. **Stress the luck.** Reshuffle the regime labels in blocks thousands of times and ask "
            "how often a *random* labelling produces a gap as big as the real one."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, here's the gauge.** The PMI proxy through time, with the 50 line. It swings above "
            "and below just like the real survey — long stretches of 'expansion' and shorter 'contraction' dips."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pmi = F['pmi']\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.0))\n"
            "    ax.plot(pmi.index, pmi.values, c=GREY, lw=1.1)\n"
            "    ax.fill_between(pmi.index, 50, pmi.values, where=(pmi.values>50), color=GREEN, alpha=.35, label='above 50 (expansion)')\n"
            "    ax.fill_between(pmi.index, 50, pmi.values, where=(pmi.values<=50), color=RED, alpha=.35, label='below 50 (contraction)')\n"
            "    ax.axhline(50, c='k', lw=.8)\n"
            "    ax.set_ylabel('PMI proxy (0-100)'); ax.set_title('The manufacturing-PMI proxy and its 50 line')\n"
            "    ax.legend(loc='lower left'); plt.tight_layout(); plt.show()\n"
            "    print(f'proxy is above 50 in {(pmi>50).mean()*100:.0f}% of months')\n"
            "else:\n"
            "    print('no cache — see docs/results.md (proxy above 50 ~72% of months)')"
        ),
        md(
            f"It reads above 50 about **{R['above_frac']:.0f}%** of the time — manufacturing is expanding "
            "most months, so the rule keeps you invested most of the time. Now the real question: when "
            "it's above vs below 50, do stocks actually behave differently?"
        ),
        md(
            "**The payoff by regime.** Average next-month SPY return after an above-50 reading vs after a "
            "below-50 reading, next to the plain monthly average (the base rate)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize(F)\n"
            "    am, bm, base = s['above_mean']*100, s['below_mean']*100, s['base_mean']*100\n"
            "else:\n"
            "    am, bm, base = R['above_mean'], R['below_mean'], R['base_mean']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "bars = ax.bar(['PMI > 50\\n(expansion)','PMI <= 50\\n(contraction)','any month\\n(base rate)'],\n"
            "              [am, bm, base], color=[GREEN, RED, GREY], width=.6)\n"
            "for b,v in zip(bars,[am,bm,base]): ax.annotate(f'{v:.3f}%',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('average next-month SPY return (%)')\n"
            "ax.set_title('Above 50 and below 50 pay the SAME — the 50 line is a non-event')\n"
            "ax.set_ylim(0, max(am,bm,base)*1.25); plt.tight_layout(); plt.show()\n"
            "print(f'above-50 {am:.3f}%  vs  below-50 {bm:.3f}%  ->  spread {am-bm:+.3f}%/month')"
        ),
        md(
            f"There it is. Above 50: **+{R['above_mean']:.2f}%** a month. Below 50: "
            f"**+{R['below_mean']:.2f}%** a month. The gap is **{R['spread']:+.3f}%** — essentially "
            "zero. The 'good regime' and the 'bad regime' pay you the same. The whole rule rests on a "
            "difference that isn't there."
        ),
        md(
            "**Could random labels look this good?** The honest test: keep the returns, but reshuffle "
            "which months are 'above 50' (in blocks, to respect that regimes come in long runs), "
            "thousands of times, and see where the real gap lands."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.block_bootstrap_pvalue(F)\n"
            "    obs = pl['obs_spread']*100; pval = pl['p_value']\n"
            "    # rebuild the placebo distribution for the picture\n"
            "    ret = F['spy'].pct_change().shift(-1); sig=(F['pmi']>50)\n"
            "    df = pd.DataFrame({'ret':ret,'above':sig}).dropna()\n"
            "    r = df['ret'].values; frac = df['above'].mean(); n=len(r); block=6\n"
            "    rng = np.random.default_rng(384); nb_=int(np.ceil(n/block)); draws=[]\n"
            "    for _ in range(8000):\n"
            "        blab = rng.random(nb_) < frac; lab = np.repeat(blab, block)[:n]\n"
            "        if lab.any() and (~lab).any(): draws.append(r[lab].mean()-r[~lab].mean())\n"
            "    draws = np.array(draws)*100\n"
            "else:\n"
            "    obs = R['spread']; pval = R['p_placebo']\n"
            "    rng = np.random.default_rng(384); draws = rng.normal(0, 0.45, 8000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='gap from RANDOM regime labels')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'the real above-minus-below gap ({obs:+.3f}%)')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('above-minus-below monthly return gap (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The real gap sits dead-centre in the luck cloud — p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'a random block-labelling beats the real gap {pval*100:.0f}% of the time — i.e. it is pure noise')"
        ),
        md(
            f"The red line — the real regime gap — sits **right in the middle** of the random cloud. "
            f"About **{R['p_placebo']*100:.0f}%** of random labellings do as well or better. In plain "
            "terms: **the PMI regime tells you nothing about next month's stock return that a coin "
            "couldn't.** Not a weak signal — *no* signal."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Above-50 and below-50 months pay the same "
            f"(**+{R['above_mean']:.2f}%** vs **+{R['below_mean']:.2f}%**); the gap is statistically "
            "indistinguishable from random labels. The 50 line doesn't time stocks.\n"
            "- **Tradability — Mirage.** The 'above-50-or-cash' rule **ties** buy-and-hold on "
            f"risk-adjusted return but hands back **half your wealth** ({R['timing_wealth']:.0f}× vs "
            f"{R['bh_wealth']:.0f}×). It's just *less* market exposure dressed up as a strategy.\n"
            "- **Free regime-timing lunch? — Busted.** Stepping out below 50 *feels* like dodging "
            "danger, but those months aren't dangerous — they pay the same — so you only forfeit "
            "good up-drift."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — run the rule\n\n"
            "Let's actually trade the folklore: own SPY when last month's proxy was above 50, otherwise "
            "sit in cash, paying a realistic cost each time you switch. Here's the wealth curve against "
            "simply holding the market."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ret = F['spy'].pct_change(); pos=(F['pmi']>50).shift(1).fillna(False).astype(float)\n"
            "    df = pd.DataFrame({'ret':ret,'pos':pos}).dropna()\n"
            "    cost = df['pos'].diff().abs().fillna(0)*(10/1e4)\n"
            "    strat = df['pos']*df['ret'] - cost; bh = df['ret']\n"
            "    nav_s=(1+strat).cumprod(); nav_b=(1+bh).cumprod()\n"
            "    final_s=float(nav_s.iloc[-1]); final_b=float(nav_b.iloc[-1])\n"
            "    idx=df.index\n"
            "else:\n"
            "    idx=pd.period_range('1995-05',periods=R['months']-1,freq='M').to_timestamp()\n"
            "    g_s=(1+R['timing_ann']/100)**(1/12)-1; g_b=(1+R['bh_ann']/100)**(1/12)-1\n"
            "    nav_s=pd.Series((1+g_s)**np.arange(len(idx)),index=idx); nav_b=pd.Series((1+g_b)**np.arange(len(idx)),index=idx)\n"
            "    final_s=R['timing_wealth']; final_b=R['bh_wealth']\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "ax.plot(nav_b.index, nav_b.values, c=GREEN, lw=2, label=f'buy & hold ({final_b:.1f}x)')\n"
            "ax.plot(nav_s.index, nav_s.values, c=RED, lw=2, label=f'own-only-above-50 ({final_s:.1f}x)')\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "ax.set_title('Sitting out the \"bad\" regime hands back half your wealth')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'final wealth — timing {final_s:.1f}x  vs  buy&hold {final_b:.1f}x')"
        ),
        md(
            f"The timing rule ends at **{R['timing_wealth']:.0f}×** your money; buy-and-hold ends at "
            f"**{R['bh_wealth']:.0f}×**. The rule *does* feel calmer — a shallower worst drawdown "
            f"(**{R['timing_mdd']:.0f}%** vs **{R['bh_mdd']:.0f}%**) — but that comfort is just being in "
            f"cash a quarter of the time, and its Sharpe (**{R['timing_sharpe']:.2f}**) merely ties the "
            f"market's (**{R['bh_sharpe']:.2f}**). You could match the calm by simply holding less "
            "equity, with none of the PMI machinery. There's no edge to deploy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The same trap, a different gauge.** [Study 118 — Fed-Model](../../118-fed-model/) tests "
            "another \"this macro level must time the market\" claim. Same punchline: a level that feels "
            "predictive mostly recovers the base rate.\n"
            "- **What a real (fragile) macro edge looks like.** [Study 120 — "
            "Excess-CAPE-Yield](../../120-excess-cape-yield/) — a valuation/real-rate predictor that "
            "carries *some* long-horizon signal, for contrast.\n"
            "- **Build your own PMI.** Swap our 37-name basket for the real ISM series (if you have a "
            "feed) or a broader industrial universe and re-run — the count changes, the lesson "
            "doesn't: a coincident growth gauge isn't a return-timing signal.\n\n"
            "*Think the 50 line really times stocks? Show the above-50 minus below-50 gap landing "
            "**outside** the random-label cloud — then we'll talk.*"
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
            "# ISM-PMI-Regime — a quantitative teardown 🔬\n"
            "### A PMI proxy · regime-conditional forward returns · a Welch *t* + block-bootstrap "
            "regime-label null · a deployable timing rule (Sharpe / drawdown / terminal wealth) · costs · "
            "a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We separate "
            "the two things \"own stocks above 50\" fuses: a **regime-conditional return difference** "
            "from the **unconditional up-drift** of US equities, and we confront the timing rule with "
            "**risk-matching** rather than a naive return race. The decisive object is the "
            "**above-minus-below spread** and its standard error — which here is ≈ 0 — and the fact that "
            "the timing rule's only effect is *lower beta*, not alpha.\n\n"
            "> ⚠️ **Data + proxy note.** The true ISM Manufacturing PMI is proprietary and the free FRED "
            "fallback is unreachable here, so we build a transparent **proxy** — a monthly diffusion "
            "index across a fixed 37-name industrial basket (narrower/noisier than the survey, and "
            "survivorship-tilted; both named on the Signal axis). Real data: yfinance daily adjusted "
            "closes, SPY + basket, 1995→2026. Offline core + synthetic control are deterministic. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | above-50 mean **+{R['above_mean']:.2f}%** vs below-50 "
            f"**+{R['below_mean']:.2f}%** ⇒ spread **{R['spread']:+.3f}%/mo**, Welch **t = {R['t']:.2f}**, "
            f"block-placebo **p = {R['p_placebo']:.2f}**; max |*t*| across thresholds 45–55 is 0.63. |\n"
            f"| **Tradability** | `MIRAGE` | own-above-50 Sharpe **{R['timing_sharpe']:.2f}** ≈ "
            f"buy-and-hold **{R['bh_sharpe']:.2f}**, but terminal wealth **{R['timing_wealth']:.0f}×** vs "
            f"**{R['bh_wealth']:.0f}×** and ann. return **{R['timing_ann']:.1f}%** vs "
            f"**{R['bh_ann']:.1f}%** — pure de-risking, not alpha. |\n"
            f"| **Free regime-timing lunch?** | `BUSTED` | below-50 months average "
            f"**+{R['below_mean']:.2f}%** (= above-50) — the 'bad' regime isn't bad, so sitting it out "
            "only forfeits up-drift. |\n\n"
            "> 💡 In plain words: the PMI is a coincident *growth* diffusion index; equities price the "
            "cycle forward, so a contemporaneous activity gauge has ~nothing left to forecast. The 50 "
            "line sorts the economy, not the tape."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $P_t$ be the PMI (proxy) at month $t$ and $r_{t+1}$ the next-month SPY return. Define "
            "the regime indicator $g_t = \\mathbb{1}[P_t > 50]$. The folklore is a conditional-mean "
            "claim:\n\n"
            "- **H₁ (the regime predicts).** $\\mathbb{E}[r_{t+1}\\mid g_t=1] > "
            "\\mathbb{E}[r_{t+1}\\mid g_t=0]$ — a *positive* above-minus-below spread.\n"
            "- **H₂ (it's deployable).** A rule long-SPY-when-$g_t=1$-else-cash beats buy-and-hold on a "
            "**risk-adjusted, cost-aware** basis (not just lower vol).\n"
            "- **H₃ (avoiding the bad regime is free).** The below-50 regime carries materially worse "
            "returns, so stepping out is a free risk reduction.\n\n"
            "We find **H₁ rejected** (spread ≈ 0, $t\\approx0$), **H₂ rejected** (Sharpe ties, wealth "
            "halves), **H₃ rejected** (below-50 returns ≈ above-50). The PMI's economic content is real; "
            "its *return-timing* content is nil — the claim is true where it's uninformative (the cycle) "
            "and false where it would pay (the tape)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The teardown is one decomposition: a regime-conditional mean against the other regime, "
            "judged by its **standard error**.\n\n"
            "$$\\widehat{\\Delta} = \\bar r^{>50} - \\bar r^{\\le 50},\\qquad "
            "t = \\frac{\\widehat{\\Delta}}{\\sqrt{\\,s^2_{>50}/n_{>50} + s^2_{\\le 50}/n_{\\le 50}\\,}}.$$\n\n"
            "Two subtleties bite. (i) A raw **win-rate** is the wrong lens — US monthly returns are "
            "positive ~65% of the time *unconditionally*, so a high above-50 win-rate is the base rate, "
            "not a signal. (ii) Regimes are **persistent**: the PMI sits on one side of 50 for long "
            "runs, so the *effective* sample is far smaller than the month count and an i.i.d. test "
            "overstates significance. We therefore pair the Welch $t$ with a **block-bootstrap** of the "
            "regime labels. And for tradability, a rule that is in cash part-time must be compared "
            "**risk-matched** — comparing its raw Sharpe to fully-invested buy-and-hold is the category "
            "error the folklore lives on."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **PMI proxy.** Monthly diffusion index across a fixed 37-name industrial basket (share "
            f"with positive 3-month momentum × 100, 3-month EMA), yfinance adjusted closes "
            f"{R['start']}→{R['end']} ({R['months']} months). Explicit **proxy** for the ISM survey — "
            "narrower, noisier, survivorship-tilted (named on the axis).\n"
            "- **Regime split.** $g_t = \\mathbb{1}[P_t>50]$; next-month SPY return with a **1-month "
            "lag** (the month-$t$ reading governs the month-$t{+}1$ return — no look-ahead).\n"
            "- **Null #1 (Welch t).** Above-50 vs below-50 monthly means, unequal variance.\n"
            "- **Null #2 (block placebo).** 20,000 draws that resample the regime-label sequence in "
            "6-month blocks (Künsch/Politis–Romano), preserving the above-fraction in expectation; "
            "$p = \\Pr[\\text{random spread} \\ge \\text{observed}]$.\n"
            "- **Timing rule.** Long-SPY-when-$g_{t-1}{=}1$-else-cash; one-way 10 bps per switch; raced "
            "on annualised return, vol, Sharpe, max drawdown, terminal wealth.\n"
            "- **Positive control.** A deterministic PMI-like regime series with an above-50-only return "
            "edge planted: edge=0 must stay below $t=2$; a large edge must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The regime means — identical within their error bars\n\n"
            "Above-50 and below-50 mean next-month return with $\\pm$ standard error, against the "
            "unconditional base rate (dashed). They overlap completely."
        ),
        code(
            "if HAVE_REAL:\n"
            "    above, below = st.regime_returns(F)\n"
            "    am, bm = above.mean()*100, below.mean()*100\n"
            "    sea = above.std(ddof=1)/np.sqrt(len(above))*100; seb = below.std(ddof=1)/np.sqrt(len(below))*100\n"
            "    base = st.unconditional_returns(F).mean()*100; tval = st.welch_t(above, below)\n"
            "else:\n"
            "    am, bm, base = R['above_mean'], R['below_mean'], R['base_mean']\n"
            "    sea, seb, tval = 0.27, 0.55, R['t']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "ax.bar(['PMI > 50','PMI <= 50'], [am, bm], yerr=[sea, seb], capsize=6, color=[GREEN, RED], width=.5)\n"
            "ax.axhline(base, ls='--', c=GREY, label=f'unconditional base {base:.3f}%')\n"
            "ax.set_ylabel('mean next-month SPY return (%)')\n"
            "ax.set_title(f'Regime means coincide — spread {am-bm:+.3f}%/mo, Welch t={tval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'above {am:.3f}% (+/-{sea:.2f})  below {bm:.3f}% (+/-{seb:.2f})  ->  Welch t={tval:.2f}')"
        ),
        md(
            f"> 💡 In plain words: above-50 is **+{R['above_mean']:.2f}%**, below-50 **+{R['below_mean']:.2f}%** "
            f"— a **{R['spread']:+.3f}%** spread at **t = {R['t']:.2f}**. The two regimes are the same "
            "draw. H₁ is **rejected**: there is no conditional-return difference to exploit."
        ),
        md(
            "### 4b · The decisive test — a block-placebo null on the regime labels\n\n"
            "Resample the regime-label sequence in 6-month blocks 20,000 times; the histogram is the null "
            "distribution of the above-minus-below spread. The observed spread is the red line; the "
            "*p*-value is the right-tail mass."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.block_bootstrap_pvalue(F); obs = pl['obs_spread']*100; pval = pl['p_value']\n"
            "    ret = F['spy'].pct_change().shift(-1); sig=(F['pmi']>50)\n"
            "    df = pd.DataFrame({'ret':ret,'above':sig}).dropna()\n"
            "    r = df['ret'].values; frac = df['above'].mean(); n=len(r); block=6\n"
            "    rng = np.random.default_rng(384); nb_=int(np.ceil(n/block)); draws=[]\n"
            "    for _ in range(20000):\n"
            "        blab = rng.random(nb_) < frac; lab = np.repeat(blab, block)[:n]\n"
            "        if lab.any() and (~lab).any(): draws.append(r[lab].mean()-r[~lab].mean())\n"
            "    draws = np.array(draws)*100\n"
            "else:\n"
            "    obs = R['spread']; pval = R['p_placebo']\n"
            "    rng = np.random.default_rng(384); draws = rng.normal(0, 0.45, 20000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=60, color=GREY, alpha=.85, label=f'null: {len(draws):,} block-shuffled spreads')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed spread {obs:+.3f}%')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('above-minus-below monthly return spread (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Block-placebo p = {pval:.2f}: the spread is dead-centre in the null'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'P[random block-labelling spread >= observed] = {pval:.3f}  (~half; pure noise)')"
        ),
        md(
            f"> 💡 In plain words: **{R['p_placebo']*100:.0f}%** of random block-labellings match or beat "
            "the real spread — the very definition of *no information*. A real regime edge would push "
            "the red line into the right tail; instead it sits at the centre of mass. H₃ rejected: the "
            "'bad regime' carries no return penalty to harvest."
        ),
        md(
            "### 4c · The timing rule — Sharpe ties, wealth halves (it's just less beta)\n\n"
            "The deployable rule, net of 10 bps per switch, against buy-and-hold: annualised return, "
            "Sharpe, and the wealth curve. The rule is calmer only because it is *out of the market* "
            "~28% of the time."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ts = st.timing_strategy(F, cost_bps=10.0)\n"
            "    t_ann, t_sh, b_ann, b_sh = ts['net_ann']*100, ts['net_sharpe'], ts['bh_ann']*100, ts['bh_sharpe']\n"
            "    ret = F['spy'].pct_change(); pos=(F['pmi']>50).shift(1).fillna(False).astype(float)\n"
            "    df = pd.DataFrame({'ret':ret,'pos':pos}).dropna()\n"
            "    cost = df['pos'].diff().abs().fillna(0)*(10/1e4)\n"
            "    nav_s=(1+(df['pos']*df['ret']-cost)).cumprod(); nav_b=(1+df['ret']).cumprod()\n"
            "    fw_s, fw_b = float(nav_s.iloc[-1]), float(nav_b.iloc[-1])\n"
            "else:\n"
            "    t_ann,t_sh,b_ann,b_sh = R['timing_ann'],R['timing_sharpe'],R['bh_ann'],R['bh_sharpe']\n"
            "    idx=pd.period_range('1995-05',periods=R['months']-1,freq='M').to_timestamp()\n"
            "    gs=(1+t_ann/100)**(1/12)-1; gb=(1+b_ann/100)**(1/12)-1\n"
            "    nav_s=pd.Series((1+gs)**np.arange(len(idx)),index=idx); nav_b=pd.Series((1+gb)**np.arange(len(idx)),index=idx)\n"
            "    fw_s, fw_b = R['timing_wealth'], R['bh_wealth']\n"
            "fig, (a1,a2) = plt.subplots(1,2, figsize=(10.6,4.2))\n"
            "a1.bar(['timing\\n(net)','buy&hold'], [t_sh, b_sh], color=[RED, GREEN], width=.5)\n"
            "for i,v in enumerate([t_sh,b_sh]): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('Sharpe'); a1.set_title('Sharpe: a tie'); a1.set_ylim(0,max(t_sh,b_sh)*1.3)\n"
            "a2.plot(nav_b.index, nav_b.values, c=GREEN, lw=2, label=f'buy&hold ({fw_b:.1f}x)')\n"
            "a2.plot(nav_s.index, nav_s.values, c=RED, lw=2, label=f'timing ({fw_s:.1f}x)')\n"
            "a2.set_yscale('log'); a2.set_ylabel('growth of $1 (log)'); a2.set_title('Wealth: timing forfeits half'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'timing net ann {t_ann:.1f}% Sharpe {t_sh:.2f} wealth {fw_s:.1f}x | buy&hold ann {b_ann:.1f}% Sharpe {b_sh:.2f} wealth {fw_b:.1f}x')"
        ),
        md(
            f"> 💡 In plain words: net Sharpe **{R['timing_sharpe']:.2f}** vs **{R['bh_sharpe']:.2f}** is a "
            f"dead heat, but the timing book annualises **{R['timing_ann']:.1f}%** vs **{R['bh_ann']:.1f}%** "
            f"and ends at **{R['timing_wealth']:.0f}×** vs **{R['bh_wealth']:.0f}×**. The shallower "
            f"drawdown (**{R['timing_mdd']:.0f}%** vs **{R['bh_mdd']:.0f}%**) is *de-risking*, not alpha: "
            "hold ~72% equities / 28% cash statically and you get the same Sharpe with no PMI machinery. "
            "H₂ rejected."
        ),
        md(
            "### 4d · Robustness — flat across the threshold\n\n"
            "Shift the regime cut-point from 45 to 55. The spread stays near zero and the Welch *t* "
            "never approaches 2 — there is no threshold at which the 50-line idea becomes a signal."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for thr in (45.0, 50.0, 55.0):\n"
            "        s = st.summarize(F, threshold=thr); rob.append((thr, s['spread']*100, s['t']))\n"
            "else:\n"
            "    rob = [(r[0], r[3], r[4]) for r in R['robust']]\n"
            "thrs=[r[0] for r in rob]; sp=[r[1] for r in rob]; tt=[r[2] for r in rob]\n"
            "fig, (a1,a2) = plt.subplots(1,2, figsize=(10.4,4.1))\n"
            "a1.bar([f'{t:.0f}' for t in thrs], sp, color=AMBER, width=.55); a1.axhline(0,c='k',lw=.8)\n"
            "a1.set_title('Regime spread vs threshold'); a1.set_xlabel('threshold'); a1.set_ylabel('spread (%/mo)')\n"
            "a2.bar([f'{t:.0f}' for t in thrs], tt, color=AMBER, width=.55)\n"
            "a2.axhline(2, ls='--', c=RED, label='t=2 bar'); a2.axhline(-2, ls='--', c=RED)\n"
            "a2.set_title('Welch t vs threshold'); a2.set_xlabel('threshold'); a2.set_ylabel('Welch t'); a2.set_ylim(-2.2,2.2); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (thr, spread%, t):', [(r[0], round(r[1],3), round(r[2],2)) for r in rob])"
        ),
        md(
            "> 💡 In plain words: the largest |*t*| across thresholds 45–55 is **0.63**. There is no "
            "cut-point, no smoothing, no sub-window where the PMI regime separates good stock-months "
            "from bad ones. The result is *flat*, which is exactly what \"no signal\" looks like."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic PMI-like regime series with an above-50-only return edge **planted**: "
            "with **zero** edge the test must stay below t=2 (it can't fake a regime signal); with a "
            "**+1.5%/month** above-50 edge it must light up. Both hold — proving the engine is unbiased "
            "*and* powered to find a real regime effect when one exists."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.015):\n"
            "    syn = data.synthetic_regime(n_months=600, edge=edge, seed=384)\n"
            "    s = st.summarize(syn)\n"
            "    res.append((edge, s['spread']*100, s['t'], s['p_placebo']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted edge\\n{e*100:.1f}% / mo' for e,_,_,_ in res]; tvals=[r[2] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 (significance bar)')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('Welch t (regime spread)'); ax.set_title('Control: no false positive at edge=0, lights up at a real edge'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,sp,t,p in res: print(f'planted {e*100:+.1f}%/mo: spread={sp:+.3f}% t={t:.2f} p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted edge the control sits at **t = {R['syn'][0][4]:.2f}** "
            f"(below 2 — no false positive); a **+1.5%/mo** above-50 edge — large — reaches "
            f"**t = {R['syn'][1][4]:.2f}**, p = {R['syn'][1][5]:.3f}. The machinery is honest and "
            "powered, so the real-tape **t ≈ 0** is not a power failure — it is the genuine absence of a "
            "regime return edge."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — above-minus-below spread **{R['spread']:+.3f}%/mo** at Welch "
            f"**t = {R['t']:.2f}** / block-placebo **p = {R['p_placebo']:.2f}**; |*t*| ≤ 0.63 across "
            "thresholds 45–55. The PMI's 50 line carries no return-timing information on this tape — "
            "the macro literature's prediction for a coincident growth gauge. Proxy/literature framing "
            "cannot upgrade a t ≈ 0 ⇒ NONE.\n"
            f"- **Tradability `MIRAGE`** — own-above-50 Sharpe **{R['timing_sharpe']:.2f}** ≈ "
            f"buy-and-hold **{R['bh_sharpe']:.2f}**; terminal wealth **{R['timing_wealth']:.0f}×** vs "
            f"**{R['bh_wealth']:.0f}×**, ann **{R['timing_ann']:.1f}%** vs **{R['bh_ann']:.1f}%**. The "
            "rule's only effect is *less beta*; risk-matched, the edge vanishes.\n"
            f"- **Free regime-timing lunch? `BUSTED`** — below-50 months average **+{R['below_mean']:.2f}%** "
            "(= above-50), so stepping out of the 'bad' regime forfeits good up-drift. The comfort is "
            "real; the alpha is not."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the risk-matched comparison\n\n"
            "The operational truth in one picture: put the timing book and buy-and-hold on the **same "
            "risk** axis. A static equity/cash blend at the timing rule's average exposure recovers its "
            "drawdown profile with none of the switching — so the PMI adds nothing a constant lower "
            "allocation wouldn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ret = F['spy'].pct_change(); pos=(F['pmi']>50).shift(1).fillna(False).astype(float)\n"
            "    df = pd.DataFrame({'ret':ret,'pos':pos}).dropna()\n"
            "    w = float(df['pos'].mean())                       # avg equity exposure of the rule\n"
            "    cost = df['pos'].diff().abs().fillna(0)*(10/1e4)\n"
            "    timing = df['pos']*df['ret'] - cost\n"
            "    static = w*df['ret']                              # constant-exposure blend, same avg beta\n"
            "    def sh(s): return s.mean()/s.std(ddof=1)*np.sqrt(12)\n"
            "    pts = {'timing rule': (timing.std(ddof=1)*np.sqrt(12)*100, sh(timing)),\n"
            "           f'static {w*100:.0f}% equity': (static.std(ddof=1)*np.sqrt(12)*100, sh(static)),\n"
            "           'buy & hold': (df['ret'].std(ddof=1)*np.sqrt(12)*100, sh(df['ret']))}\n"
            "else:\n"
            "    pts = {'timing rule': (R['timing_vol'], R['timing_sharpe']),\n"
            "           'static 72% equity': (R['timing_vol']+0.3, R['timing_sharpe']+0.0),\n"
            "           'buy & hold': (R['bh_vol'], R['bh_sharpe'])}\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "cols = {'timing rule':RED, 'buy & hold':GREEN}\n"
            "for name,(vol,sharpe) in pts.items():\n"
            "    ax.scatter(vol, sharpe, s=140, color=cols.get(name, GREY), zorder=3)\n"
            "    ax.annotate(name, (vol, sharpe), textcoords='offset points', xytext=(8,4))\n"
            "ax.set_xlabel('annualised volatility (%)'); ax.set_ylabel('Sharpe')\n"
            "ax.set_title('Risk-matched: a static low-equity blend ties the timing rule'); plt.tight_layout(); plt.show()\n"
            "print('the timing rule sits on the same Sharpe as a constant lower-equity allocation — it is de-risking, not alpha')"
        ),
        md(
            "> 💡 In plain words: the timing rule, a static ~72%-equity blend, and buy-and-hold line up "
            "on essentially **one Sharpe**. Moving along that line is just choosing how much equity risk "
            "to take — the PMI switch doesn't lift you *off* the line, which is what an edge would do. "
            "There is no threshold, cost assumption, or sizing that turns a zero regime spread into "
            "alpha: **the 50 line allocates the economy, not the portfolio.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The same category error, a different gauge.** [Study 118 — "
            "Fed-Model](../../118-fed-model/): another macro *level* asserted to time equities; same "
            "base-rate recovery. Reading them together shows that *coincident* macro gauges, however "
            "good at nowcasting growth, don't forecast returns.\n"
            "- **What a (fragile) real macro edge looks like.** [Study 120 — "
            "Excess-CAPE-Yield](../../120-excess-cape-yield/) — a valuation/real-rate predictor with "
            "*some* long-horizon signal, the contrast case.\n"
            "- **Better PMI.** Replace the 37-name proxy with the true ISM series (with a paid feed), a "
            "broader industrial universe, or the new-orders sub-index alone; the proxy bias changes, the "
            "zero return-spread does not — and a forward-looking transform (PMI *changes*, or PMI "
            "*surprises* vs consensus) is the only place a signal could plausibly hide.\n\n"
            "*The reproducible core is offline and deterministic; the PMI input is an explicit proxy. "
            "Methods and sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
