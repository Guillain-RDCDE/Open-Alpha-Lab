"""Generate the two narrative notebooks for Study 374 (Vol-of-Vol 🌀).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached
``_cache/vol_tape.csv`` (^VVIX / ^VIX / SPY) and otherwise quote the frozen headline numbers
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance ^VVIX/^VIX/SPY,
# 2007-01-03 -> 2026-06-18, 4,887 days, 19.5 years).
R = dict(
    start="2007-01-03", end="2026-06-18", days=4887, years=19.5,
    n_high=1059, frac_high=21.7,
    # per-horizon: (months, n, cond_mean%, cond_win%, base_mean%, base_win%, t, p_plac,
    #               cond_dd%, base_dd%)
    h1=(1, 1059, 1.44, 69, 0.99, 67, 2.35, 0.929, -5.37, -4.13),
    h3=(3, 1046, 3.89, 73, 2.92, 73, 3.51, 0.969, -8.75, -7.66),
    # incremental HAC regression: corr, vix_only_t, vvix_only_t, both_vix_t, both_vvix_t
    inc1=dict(corr=0.41, vix_only_t=1.14, vvix_only_t=2.28, vix_t=0.57, vvix_t=1.57,
              vvix_only_b=0.59, vvix_b=0.49),
    inc3=dict(corr=0.41, vix_only_t=0.82, vvix_only_t=1.74, vix_t=0.39, vvix_t=1.12,
              vvix_only_b=1.27, vvix_b=1.06),
    # robustness: (quantile, n, cond%, welch_t)
    robust=[(0.70, 1567, 1.08, 0.56), (0.80, 1059, 1.44, 2.35), (0.90, 556, 1.60, 2.25)],
    # tradability: net_ann%, net_vol%, net_sharpe, bh_ann%, bh_vol%, bh_sharpe, trades, pct_flat
    trade=dict(net_ann=8.9, net_vol=13.7, net_sh=0.65, bh_ann=12.4, bh_vol=19.6, bh_sh=0.63,
               trades=414, pct_flat=22),
    # synthetic control: (edge, vvix_only_t, vvix_over_vix_t, vvix_b%)
    syn=[(0.0, -0.83, -1.52, -0.50), (0.04, -14.16, -15.84, -13.01),
         (0.10, -17.76, -19.85, -22.82)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Better_than_the_VIX%3F: Busted](https://img.shields.io/badge/Better_than_the_VIX%3F-Busted-8b949e?style=flat-square)\n\n"
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

from vol_of_vol import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real() if HAVE_REAL else None
STATE = st.vvix_state(F) if HAVE_REAL else None
print("real vol-tape cache present:", HAVE_REAL,
      "| high-VVIX days:", (0 if STATE is None else int(STATE.sum())))
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
            "# The volatility of the volatility — does VVIX beat the VIX? 🌀\n"
            "### \"Vol of vol\" is supposed to be a sharper fear gauge than the VIX. Is it? — in plain English\n\n"
            + BADGES +
            "Everyone knows the **VIX**: the market's \"fear gauge,\" the expected wobble of the S&P 500. "
            "Fewer people know its shadow, the **VVIX** — the expected wobble *of the VIX itself*. The "
            "**volatility of the volatility.** The pitch from vol traders is seductive: when the "
            "vol-of-vol spikes, the options market is pricing a shock the plain VIX hasn't caught up to "
            "yet — so VVIX should be an **earlier, sharper** risk alarm than the VIX.\n\n"
            "It's a great story. This notebook checks whether the vol-of-vol actually *adds* anything "
            "over the VIX you can already watch for free — and finds that, once you put both gauges side "
            "by side, the extra alarm is mostly an echo.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the placebo test and the regression "
            "that decides it? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** VVIX (`^VVIX`), VIX (`^VIX`) and SPY come straight from "
            "yfinance; VVIX's history starts in 2007, so we cover every big vol shock since the GFC. "
            "Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does a vol-of-vol spike precede a market drop? | **Not really — the opposite.** After "
            "high-VVIX days the market is, on average, *higher* a month later, because VVIX peaks at "
            "**crash bottoms** (2008, 2020) that then bounce. |\n"
            "| So VVIX is useless? | **No — it flags turbulence.** High-VVIX days are followed by "
            "**deeper drawdowns** (a −5.4% vs −4.1% worst-dip). It marks a *bumpy* month, not a *down* "
            "one. |\n"
            "| But is it **better than the VIX**? | **That's the whole question — and the answer is "
            "no.** Put VIX and VVIX in the same model and the vol-of-vol's edge **vanishes**: it's "
            "~40% correlated with the VIX, so most of its \"signal\" is just the VIX showing through. |\n"
            "| Could you trade it? | **No.** A \"go to cash when vol-of-vol is high\" timer ties "
            "buy-and-hold on risk-adjusted terms while giving up return. |\n\n"
            "> Vol-of-vol is real and even mildly useful as a *turbulence* flag. But as a **sharper "
            "VIX**, it's a redundant cousin — not a new clock."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The VIX tells you how much the market is expected to move. The **VVIX** tells you how "
            "much the **VIX** is expected to move. When vol-of-vol blows out, the options market is "
            "bracing for a tail the VIX hasn't priced yet — so VVIX leads the VIX, and times equity risk "
            "**better** than it.\"*\n\n"
            "VVIX is a real CBOE index: the same model-free recipe as the VIX, applied to **VIX "
            "options**. The picture is intuitive — a violent re-pricing of *uncertainty about "
            "uncertainty* feels like it should be early warning. We'll build the gauge, look at what "
            "follows a spike, and — the part that matters — ask whether it beats the plain VIX."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If vol-of-vol really timed risk **better than the VIX**, it would be worth real money: a "
            "cheap, public series that front-runs the most-watched fear gauge on the planet. But notice "
            "the trap hiding in \"better than the VIX.\" VVIX and VIX move together — when one is high "
            "the other usually is too. So a signal that looks predictive *might* just be the VIX wearing "
            "a costume. The only honest question isn't \"does high VVIX precede trouble?\" (the VIX does "
            "that too) — it's **\"does VVIX add anything *once you already know the VIX*?\"** Everything "
            "below builds to that one test."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We pull **{R['days']:,}** days of VVIX, VIX and SPY ({R['start']} → {R['end']}, "
            f"{R['years']:.0f} years). Then:\n\n"
            "1. **Mark the spikes.** A \"high vol-of-vol\" day = VVIX above its own trailing one-year "
            f"80th percentile (a fair, point-in-time bar). That fires about **{R['frac_high']:.0f}%** of "
            "days.\n"
            "2. **Measure what follows.** After each high-VVIX day, what did SPY do over the next "
            "**1 and 3 months** — return *and* worst drawdown — vs a normal day (the base rate)?\n"
            "3. **The decider.** Race VVIX against the VIX head-to-head: put **both** in one model and "
            "see whether vol-of-vol still says anything once the VIX is accounted for. If it doesn't, "
            "\"better than the VIX\" is busted."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, meet the two gauges.** VVIX (the vol-of-vol) and VIX (plain vol) over the whole "
            "sample. See how their spikes line up — that shared shape is the whole problem."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fig, ax = plt.subplots(figsize=(9.6, 4.4))\n"
            "    ax.plot(F.index, F['vvix'], c=GREEN, lw=.8, label='VVIX (vol of vol)')\n"
            "    ax2 = ax.twinx()\n"
            "    ax2.plot(F.index, F['vix'], c=GREY, lw=.8, alpha=.8, label='VIX (plain vol)')\n"
            "    ax.set_ylabel('VVIX', color=GREEN); ax2.set_ylabel('VIX', color=GREY)\n"
            "    ax2.grid(False)\n"
            "    corr = F['vvix'].corr(F['vix'])\n"
            "    ax.set_title(f'VVIX and VIX move together (corr = {corr:.2f}) — that is the catch')\n"
            "    l1,la1 = ax.get_legend_handles_labels(); l2,la2 = ax2.get_legend_handles_labels()\n"
            "    ax.legend(l1+l2, la1+la2, loc='upper right')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'corr(VVIX, VIX) = {corr:.2f}')\n"
            "else:\n"
            "    print('no cache — VVIX and VIX are ~0.41 correlated; see docs/results.md')"
        ),
        md(
            "**Now, what follows a vol-of-vol spike?** The believers expect *down*. Here's the forward "
            "1-month win-rate after a high-VVIX day next to a normal day — and the surprise is the "
            "direction."
        ),
        code(
            "hs = [1, 3]\n"
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(F, STATE, m) for m in hs]\n"
            "    cwin = [r['cond_win']*100 for r in rows]; bwin = [r['base_win']*100 for r in rows]\n"
            "else:\n"
            "    cwin = [R['h1'][3], R['h3'][3]]; bwin = [R['h1'][5], R['h3'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.bar(x-.2, cwin, .4, color=GREEN, label='after a high-VVIX day')\n"
            "ax.bar(x+.2, bwin, .4, color=GREY, label='any normal day (base rate)')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m} month' + ('s' if m>1 else '') for m in hs])\n"
            "ax.set_ylim(0, 100); ax.set_ylabel('% of the time SPY is higher')\n"
            "ax.set_title('After a vol-of-vol spike the market is MORE likely up, not less')\n"
            "for i,(c,b) in enumerate(zip(cwin,bwin)):\n"
            "    ax.annotate(f'{c:.0f}%',(i-.2,c),ha='center',va='bottom')\n"
            "    ax.annotate(f'{b:.0f}%',(i+.2,b),ha='center',va='bottom')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('1m win-rate after high VVIX:', f'{cwin[0]:.0f}%', ' vs base', f'{bwin[0]:.0f}%')"
        ),
        md(
            f"There's the first twist. After a vol-of-vol spike SPY is higher a month later "
            f"**{R['h1'][3]:.0f}%** of the time — *above* the **{R['h1'][5]:.0f}%** base rate, not below. "
            "VVIX peaks when markets are already **bottoming** (late 2008, March 2020), and bottoms "
            "bounce. So as a \"sell\" signal it points the **wrong way**. Where it *does* bite is "
            "**drawdowns** — let's see that."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cdd = [st.summarize(F, STATE, m)['cond_dd']*100 for m in hs]\n"
            "    bdd = [st.summarize(F, STATE, m)['base_dd']*100 for m in hs]\n"
            "else:\n"
            "    cdd = [R['h1'][8], R['h3'][8]]; bdd = [R['h1'][9], R['h3'][9]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.bar(x-.2, cdd, .4, color=AMBER, label='after a high-VVIX day')\n"
            "ax.bar(x+.2, bdd, .4, color=GREY, label='any normal day (base rate)')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m} month'+('s' if m>1 else '') for m in hs])\n"
            "ax.set_ylabel('average worst drawdown (%)'); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_title('Where VVIX bites: deeper drawdowns (a bumpier month, not a down one)')\n"
            "for i,(c,b) in enumerate(zip(cdd,bdd)):\n"
            "    ax.annotate(f'{c:.1f}%',(i-.2,c),ha='center',va='top')\n"
            "    ax.annotate(f'{b:.1f}%',(i+.2,b),ha='center',va='top')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('1m worst drawdown: high-VVIX', f'{cdd[0]:.1f}%', ' vs base', f'{bdd[0]:.1f}%')"
        ),
        md(
            f"So vol-of-vol *is* a turbulence flag: after a spike the worst dip over the next month is "
            f"**{R['h1'][8]:.1f}%** vs **{R['h1'][9]:.1f}%** normally. A bumpier ride — even if the month "
            "tends to *end* higher. But none of this is the real question. The real question is whether "
            "any of it beats the **VIX**."
        ),
        md(
            "**The decider: VVIX vs VIX, head to head.** We let each gauge predict next month's return "
            "on its own, then put **both** in one race. Watch what happens to VVIX's strength (its "
            "*t*-stat — bigger = stronger) when the VIX joins the model."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = st.incremental_regression(F, 1)\n"
            "    vvix_alone = r['vvix_only_t']; vvix_with_vix = r['vvix_t']\n"
            "else:\n"
            "    vvix_alone = R['inc1']['vvix_only_t']; vvix_with_vix = R['inc1']['vvix_t']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "bars = ax.bar(['VVIX\\nalone', 'VVIX\\nwith VIX in the room'],\n"
            "              [vvix_alone, vvix_with_vix], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(2, ls='--', c=GREY, label='strength needed to call it real (t=2)')\n"
            "for b,v in zip(bars,[vvix_alone, vvix_with_vix]):\n"
            "    ax.annotate(f't={v:.2f}',(b.get_x()+b.get_width()/2, v),ha='center',va='bottom')\n"
            "ax.set_ylabel(\"VVIX's predictive strength (t-stat)\"); ax.set_ylim(0, 2.6)\n"
            "ax.set_title('Add the VIX and the vol-of-vol edge falls below the line'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'VVIX alone: t={vvix_alone:.2f} (clears 2)  ->  with VIX: t={vvix_with_vix:.2f} (fails)')"
        ),
        md(
            f"That's the whole study in one chart. **Alone**, VVIX looks predictive (*t* = "
            f"{R['inc1']['vvix_only_t']:.2f}, above the line). **With the VIX in the room**, it collapses "
            f"to *t* = {R['inc1']['vvix_t']:.2f} — below the bar. The vol-of-vol's apparent edge was "
            "mostly the VIX showing through a 0.41-correlated window. \"Better than the VIX\" — busted."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** High-VVIX days do precede a real *turbulence* signal (deeper "
            f"drawdowns) and even a positive return bump (*t* = {R['h1'][6]:.2f}) — but pointed the "
            "*wrong way* for a sell signal, fragile to the threshold, and **not better than the VIX**.\n"
            "- **Tradability — Mirage.** A \"cash when vol-of-vol is high\" timer ties buy-and-hold on "
            "risk-adjusted terms and gives up return. No edge to deploy.\n"
            "- **Better than the VIX? — Busted.** Once the VIX is in the model, VVIX adds nothing "
            f"robust (*t* {R['inc1']['vvix_only_t']:.2f} → {R['inc1']['vvix_t']:.2f}). It's a redundant "
            "cousin of vol, not a sharper clock."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Suppose you ignored all of that and built the obvious rule: **hold SPY, but go to cash "
            "whenever vol-of-vol is high.** Does dodging the bumpy months pay? Here's that timer against "
            "plain buy-and-hold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bt = st.timing_backtest(F, STATE)\n"
            "    net_sh, bh_sh = bt['net_sharpe'], bt['bh_sharpe']\n"
            "    net_ann, bh_ann = bt['net_ann']*100, bt['bh_ann']*100\n"
            "    net_vol, bh_vol = bt['net_vol']*100, bt['bh_vol']*100\n"
            "else:\n"
            "    t = R['trade']; net_sh, bh_sh = t['net_sh'], t['bh_sh']\n"
            "    net_ann, bh_ann = t['net_ann'], t['bh_ann']; net_vol, bh_vol = t['net_vol'], t['bh_vol']\n"
            "fig, (a1,a2) = plt.subplots(1,2,figsize=(10.2,4.2))\n"
            "a1.bar(['VVIX timer','buy & hold'],[net_sh,bh_sh],color=[AMBER,GREY],width=.55)\n"
            "for i,v in enumerate([net_sh,bh_sh]): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('Sharpe (net of 2 bps)'); a1.set_title('Risk-adjusted: a dead heat'); a1.set_ylim(0,.8)\n"
            "a2.bar(['VVIX timer','buy & hold'],[net_ann,bh_ann],color=[AMBER,GREY],width=.55)\n"
            "for i,v in enumerate([net_ann,bh_ann]): a2.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('annual return (%)'); a2.set_title('Return: the timer gives some up')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'timer Sharpe {net_sh:.2f} vs buy-and-hold {bh_sh:.2f} — a wash; timer earns less')"
        ),
        md(
            f"The timer nets a Sharpe of **{R['trade']['net_sh']:.2f}** — a statistical tie with "
            f"buy-and-hold's **{R['trade']['bh_sh']:.2f}** — while earning **{R['trade']['net_ann']:.1f}%** "
            f"a year against the market's **{R['trade']['bh_ann']:.1f}%**. You bought a calmer ride that "
            "any cash blend would sell you, at the price of return. There's no free lunch hiding in the "
            "vol-of-vol."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The VIX's own shape.** [Study 111 — VIX-term-structure](../111-vix-term-structure/) "
            "asks whether the *slope* of the VIX curve (contango vs backwardation) times risk — a "
            "sibling \"second-order vol\" idea.\n"
            "- **The premium underneath.** [Study 130 — Vol-risk-premium](../130-vol-risk-premium/) — "
            "implied-minus-realized vol, the priced gap these signals sit on top of.\n"
            "- **The cross-section.** [Study 330 — Low-volatility-anomaly](../330-low-volatility-anomaly/) "
            "— does *low* vol pay, or is it beta in disguise?\n"
            "- **Try the VVIX/VIX *ratio*.** A popular variant normalises vol-of-vol by vol. Re-run the "
            "incremental test on it — does the *ratio* beat the VIX where the level didn't?\n\n"
            "*Think vol-of-vol beats the VIX? Put both in one regression and show VVIX clearing t=2 "
            "**with VIX in the model** — then we'll talk.*"
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
            "# Vol-of-Vol — a quantitative teardown 🔬\n"
            "### A high-VVIX state · conditional vs unconditional forward returns/drawdowns · the "
            "decisive VIX+VVIX HAC regression · a block placebo null · a costed long/flat timer · a "
            "synthetic incremental-signal control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The claim "
            "is **incremental**: VVIX times equity risk *better than the VIX*. Because corr(VIX, VVIX) ≈ "
            f"{R['inc1']['corr']:.2f}, a univariate VVIX regression is confounded by VIX, so the only "
            "honest object is the **partial effect** of VVIX *conditional on VIX*, with HAC inference "
            "for the overlapping-window serial correlation. That coefficient, not the raw one, decides "
            "the Signal axis — and it fails the bar.\n\n"
            "> ⚠️ **Data note.** `^VVIX`, `^VIX`, `SPY` (auto-adjusted) from yfinance, "
            f"{R['start']}→{R['end']}, {R['days']:,} days. VVIX/VIX are index levels; SPY is "
            "total-return-ish. Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `WEAK` | Raw 1m Welch **t = {R['h1'][6]:.2f}** (positive — *wrong sign* for "
            f"a sell signal); but VVIX's HAC *t* falls **{R['inc1']['vvix_only_t']:.2f} → "
            f"{R['inc1']['vvix_t']:.2f}** once VIX is in the model, and the raw *t* is "
            f"**{R['robust'][0][3]:.2f}** at the 70th-pct threshold (fragile). |\n"
            f"| **Tradability** | `MIRAGE` | Long/flat timer net Sharpe **{R['trade']['net_sh']:.2f}** vs "
            f"buy-and-hold **{R['trade']['bh_sh']:.2f}** — a wash; earns "
            f"**{R['trade']['net_ann']:.1f}%** vs **{R['trade']['bh_ann']:.1f}%**. |\n"
            f"| **Better than the VIX?** | `BUSTED` | corr(VIX,VVIX) = **{R['inc1']['corr']:.2f}**; in "
            f"the bivariate HAC fit VVIX **t = {R['inc1']['vvix_t']:.2f}** (1m) / "
            f"**{R['inc3']['vvix_t']:.2f}** (3m) — no incremental signal. |\n\n"
            "> 💡 In plain words: VVIX looks like a timer in isolation, but it's a 0.4-correlated "
            "shadow of the VIX. Strip the VIX out and almost nothing's left — and what *is* left points "
            "the wrong way for \"spike = sell.\""
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $V_t$ = VIX, $W_t$ = VVIX, and $r_{t\\to t+H}$ the forward SPY return. The believers' "
            "hypotheses:\n\n"
            "- **H₁ (VVIX predicts).** $\\mathbb{E}[r_{t\\to t+H}\\mid W_t\\ \\text{high}] \\ne "
            "\\mathbb{E}[r_{t\\to t+H}]$ — and in the bearish direction.\n"
            "- **H₂ (incremental over VIX).** In $r_{t\\to t+H} = \\alpha + \\beta\\,z(V_t) + "
            "\\gamma\\,z(W_t) + \\varepsilon$, the partial coefficient $\\gamma$ is significant **with "
            "$\\beta$ in the model** — vol-of-vol adds information the VIX lacks.\n"
            "- **H₃ (deployable).** The edge survives costs and beats buy-and-hold risk-adjusted.\n\n"
            f"We find **H₁ half-true, wrong-signed** (high VVIX → *higher* mean return, real drawdown "
            f"signal), **H₂ rejected** ($\\gamma$ HAC *t* = {R['inc1']['vvix_t']:.2f} once $\\beta$ is "
            "in), **H₃ rejected** (Sharpe tie). The steelman dies on H₂: *incrementality* is the entire "
            "claim, and the partial effect doesn't clear the bar."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — why the partial coefficient is the only honest number\n\n"
            "With $\\mathrm{corr}(z(V), z(W)) \\approx " + f"{R['inc1']['corr']:.2f}" + "$, a univariate "
            "$\\hat\\gamma^{\\text{uni}}$ absorbs the VIX's predictive content through the shared "
            "variance — Frisch–Waugh–Lovell makes the bivariate $\\hat\\gamma$ the coefficient on the "
            "part of VVIX **orthogonal to VIX**. That is exactly \"better than the VIX.\" And because "
            "daily-sampled $H$-day forward returns overlap, $\\varepsilon_t$ is strongly "
            "autocorrelated; naïve OLS SEs are far too small. We use **Newey-West** with the lag set to "
            "the overlap length $H$, so the *t* on $\\hat\\gamma$ is HAC-honest. The raw win-rate is the "
            "wrong lens twice over: US forward returns are positive most of the time *unconditionally*, "
            "and the conditional set sits at crash-recovery dates."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** `^VVIX`, `^VIX`, `SPY` (auto-adjusted), {R['start']}→{R['end']}, "
            f"{R['days']:,} days. VVIX history begins 2007 (covers GFC / 2011 / 2015 / 2018 / 2020).\n"
            "- **State.** High vol-of-vol = $W_t$ above its trailing **252-day 80th percentile** "
            "(causal; no look-ahead).\n"
            "- **Forward objects.** $H\\in\\{1,3\\}$ months (21/63d): forward return **and** worst "
            "peak-to-trough drawdown; enter the close **1 day after** the signal.\n"
            "- **Null #1 (Welch t).** Conditional vs unconditional overlapping forward mean.\n"
            "- **Null #2 (block placebo).** 20,000 contiguous-block resamples sized to the state's "
            "clustering; $p = \\Pr[\\text{draw mean} \\le \\text{conditional mean}]$ (testing for "
            "under-performance).\n"
            "- **The decider (HAC regression).** $r = \\alpha + \\beta z(V) + \\gamma z(W)$ with "
            "Newey-West SEs at lag $H$; VIX-only, VVIX-only, and bivariate fits.\n"
            "- **Costs.** 1-day-lagged long/flat timer, 2 bps one-way per position change, raced "
            "gross/net vs buy-and-hold.\n"
            "- **Positive control.** A synthetic tape where the planted edge lives **only** in the "
            "VIX-orthogonal part of VVIX: $\\gamma$ must be null at edge 0 and light up for a large edge."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Conditional forward returns — positive, wrong-signed, *t* without the control\n\n"
            "Conditional mean forward return (±SE) vs the unconditional base rate (dashed). Positive at "
            "both horizons — the high-VVIX set sits at crash-recovery dates, so a naïve Welch *t* is "
            "**positive**, the opposite of the bearish thesis."
        ),
        code(
            "hs = [1, 3]\n"
            "if HAVE_REAL:\n"
            "    cm, bm, ts, ses = [], [], [], []\n"
            "    for m in hs:\n"
            "        s = st.summarize(F, STATE, m); cm.append(s['cond_mean']); bm.append(s['base_mean']); ts.append(s['t'])\n"
            "        h = st.TRADING_DAYS[m]; fwd = st.forward_return(F['spy'], h).shift(-1).dropna()\n"
            "        cond = fwd[STATE.reindex(fwd.index).fillna(False).values]\n"
            "        ses.append(cond.std(ddof=1)/np.sqrt(len(cond)))\n"
            "else:\n"
            "    cm = [R['h1'][2]/100, R['h3'][2]/100]; bm = [R['h1'][4]/100, R['h3'][4]/100]\n"
            "    ts = [R['h1'][6], R['h3'][6]]; ses = [0.0024, 0.0040]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.bar(x, [c*100 for c in cm], yerr=[s*100 for s in ses], capsize=5, color=GREEN, width=.5, label='after high VVIX (±SE)')\n"
            "ax.plot(x, [b*100 for b in bm], 'D', ms=11, c=GREY, label='unconditional base rate')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m}m' for m in hs]); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward SPY return (%)')\n"
            "ax.set_title('Conditional mean is ABOVE base (wrong sign for a sell signal)'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Welch t by horizon:', {f'{m}m': round(t,2) for m,t in zip(hs,ts)})"
        ),
        md(
            f"> 💡 In plain words: at 1m the conditional mean is **+{R['h1'][2]:.2f}%** vs base "
            f"**+{R['h1'][4]:.2f}%** (Welch *t* = {R['h1'][6]:.2f}) — *positive*. VVIX peaks at bottoms, "
            "and bottoms bounce. As a directional sell signal H₁ is **wrong-signed**; the honest "
            "positive content is in the drawdown (4c) and must still beat the VIX (4b)."
        ),
        md(
            "### 4b · The decisive test — incremental over VIX (HAC regression)\n\n"
            "Three Newey-West fits on the aligned sample: VIX-only, VVIX-only, and the bivariate "
            "VIX+VVIX. The vol-of-vol thesis requires the **bivariate** VVIX *t* to clear 2."
        ),
        code(
            "hs = [1, 3]\n"
            "if HAVE_REAL:\n"
            "    rr = [st.incremental_regression(F, m) for m in hs]\n"
            "    vvix_only = [r['vvix_only_t'] for r in rr]; vvix_both = [r['vvix_t'] for r in rr]\n"
            "    vix_both = [r['vix_t'] for r in rr]\n"
            "else:\n"
            "    vvix_only = [R['inc1']['vvix_only_t'], R['inc3']['vvix_only_t']]\n"
            "    vvix_both = [R['inc1']['vvix_t'], R['inc3']['vvix_t']]\n"
            "    vix_both = [R['inc1']['vix_t'], R['inc3']['vix_t']]\n"
            "x = np.arange(len(hs)); w = .26\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-w, vvix_only, w, color=GREEN, label='VVIX alone (HAC t)')\n"
            "ax.bar(x,   vvix_both, w, color=RED, label='VVIX | VIX in model (HAC t)')\n"
            "ax.bar(x+w, vix_both,  w, color=GREY, label='VIX | VVIX in model (HAC t)')\n"
            "ax.axhline(2, ls='--', c='k', lw=1, label='t = 2 (significance bar)')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m}-month horizon' for m in hs])\n"
            "ax.set_ylabel('HAC (Newey-West) t-stat')\n"
            "ax.set_title('VVIX clears t=2 alone, fails once VIX is controlled'); ax.legend()\n"
            "for i in range(len(hs)):\n"
            "    ax.annotate(f'{vvix_only[i]:.2f}',(i-w,vvix_only[i]),ha='center',va='bottom')\n"
            "    ax.annotate(f'{vvix_both[i]:.2f}',(i,vvix_both[i]),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('VVIX HAC t — alone vs with VIX:', [(round(a,2),round(b,2)) for a,b in zip(vvix_only,vvix_both)])"
        ),
        md(
            f"> 💡 In plain words: the **green** bars (VVIX alone) clear 2 at 1m "
            f"({R['inc1']['vvix_only_t']:.2f}); the **red** bars (VVIX with VIX controlled) sink to "
            f"{R['inc1']['vvix_t']:.2f} / {R['inc3']['vvix_t']:.2f}. The orthogonal-to-VIX part of VVIX "
            "carries no robust signal. **This single chart is the verdict** — H₂ rejected, \"better "
            "than the VIX\" busted."
        ),
        md(
            "### 4c · Robustness + the drawdown bite — neither rescues incrementality\n\n"
            "Left: the raw Welch *t* across VVIX-state thresholds — significant only in a narrow tail "
            "band (fragile). Right: where the signal genuinely (if weakly) bites — forward drawdowns are "
            "deeper after high VVIX. Neither restores an *incremental* edge over VIX."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = [(q,)+(lambda s:(s['n'],s['cond_mean']*100,s['t']))(st.summarize(F, st.vvix_state(F,q=q),1)) for q in (0.70,0.80,0.90)]\n"
            "    cdd = [st.summarize(F, STATE, m)['cond_dd']*100 for m in (1,3)]\n"
            "    bdd = [st.summarize(F, STATE, m)['base_dd']*100 for m in (1,3)]\n"
            "else:\n"
            "    rob = [(q,n,c,t) for (q,n,c,t) in R['robust']]\n"
            "    cdd = [R['h1'][8], R['h3'][8]]; bdd = [R['h1'][9], R['h3'][9]]\n"
            "fig, (a1,a2) = plt.subplots(1,2,figsize=(10.4,4.2))\n"
            "qs=[r[0] for r in rob]; tt=[r[3] for r in rob]; ns=[r[1] for r in rob]\n"
            "a1.bar([f'{q:.2f}' for q in qs], tt, color=AMBER, width=.55)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2'); a1.set_ylim(0,2.6)\n"
            "for i,(t,n) in enumerate(zip(tt,ns)): a1.annotate(f'n={n}',(i,t),ha='center',va='bottom')\n"
            "a1.set_title('Raw Welch t is threshold-fragile'); a1.set_xlabel('VVIX state quantile'); a1.set_ylabel('Welch t'); a1.legend()\n"
            "xx=np.arange(2)\n"
            "a2.bar(xx-.2,cdd,.4,color=AMBER,label='after high VVIX'); a2.bar(xx+.2,bdd,.4,color=GREY,label='base')\n"
            "a2.set_xticks(xx); a2.set_xticklabels(['1m','3m']); a2.set_ylabel('avg worst drawdown (%)'); a2.axhline(0,c='k',lw=.8)\n"
            "a2.set_title('The honest signal: deeper drawdowns'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (q, n, cond%, welch t):', [(r[0], r[1], round(r[2],2), round(r[3],2)) for r in rob])"
        ),
        md(
            f"> 💡 In plain words: drop the bar to the 70th pct and the raw Welch *t* falls to "
            f"**{R['robust'][0][3]:.2f}** — the \"significant\" number lives only in the extreme tail. "
            "The drawdown signal is real (−5.4% vs −4.1% at 1m) but it's a *risk* flag, not a *return* "
            "edge, and it's still just **vol** — the VIX flags the same drawdowns."
        ),
        md(
            "### 4d · Faithful-engine & incremental-power control — we know the truth here\n\n"
            "A synthetic tape where VVIX = a VIX component **plus an independent vol-of-vol component**, "
            "and the planted edge is injected **only** into the VIX-orthogonal part. With **zero** edge "
            "the bivariate VVIX *t* must stay inside ±2 (no false positive); with a large planted edge "
            "it must light up — proving the engine isolates *incremental* signal, not VIX bleed-through."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.04, 0.10):\n"
            "    syn = data.synthetic_tape(edge=edge, seed=374)\n"
            "    r = st.incremental_regression(syn, 1)\n"
            "    res.append((edge, r['vvix_only_t'], r['vvix_t']))\n"
            "labels = [f'planted edge\\n{e*100:.0f}%' for e,_,_ in res]\n"
            "only = [r[1] for r in res]; both = [r[2] for r in res]\n"
            "x = np.arange(len(res)); w=.36\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(x-w/2, only, w, color=GREY, label='VVIX alone')\n"
            "ax.bar(x+w/2, both, w, color=GREEN, label='VVIX | VIX (incremental)')\n"
            "ax.axhline(-2, ls='--', c=RED, label='|t| = 2 bar'); ax.axhline(2, ls='--', c=RED)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('HAC t (1-month)')\n"
            "ax.set_title('Control: incremental t is null at edge 0, lights up for a real edge'); ax.legend()\n"
            "for i,b in enumerate(both): ax.annotate(f'{b:.1f}',(i+w/2,b),ha='center',va='top' if b<0 else 'bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,o,b in res: print(f'planted {e*100:+.0f}%: VVIX-only t={o:.2f} -> VVIX|VIX t={b:.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted edge the incremental VVIX *t* is "
            f"**{R['syn'][0][2]:.2f}** (inside ±2 — no false positive); a planted edge drives it to "
            f"**{R['syn'][1][2]:.1f} / {R['syn'][2][2]:.1f}** with the correct (negative, drawdown-y) "
            "sign. The machinery is honest *and* specifically isolates incrementality. The real-tape "
            f"incremental *t* of ~{R['inc1']['vvix_t']:.1f} is exactly what **no incremental edge** "
            "looks like through this lens."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — raw 1m Welch *t* = {R['h1'][6]:.2f} (positive, wrong-signed) + a "
            f"real drawdown flag, but threshold-fragile (*t* = {R['robust'][0][3]:.2f} at the 70th pct) "
            f"and, decisively, **non-incremental**: VVIX HAC *t* = {R['inc1']['vvix_t']:.2f} once VIX is "
            "in. Literature support + a non-robust, non-incremental estimate ⇒ WEAK, not REAL.\n"
            f"- **Tradability `MIRAGE`** — long/flat timer net Sharpe {R['trade']['net_sh']:.2f} vs "
            f"buy-and-hold {R['trade']['bh_sh']:.2f} (a wash), earning {R['trade']['net_ann']:.1f}% vs "
            f"{R['trade']['bh_ann']:.1f}%. No NAV-scale edge.\n"
            f"- **Better than the VIX? `BUSTED`** — corr(VIX,VVIX) = {R['inc1']['corr']:.2f}; the "
            f"bivariate VVIX *t* is {R['inc1']['vvix_t']:.2f} (1m) / {R['inc3']['vvix_t']:.2f} (3m). The "
            "raw VVIX edge is mostly the VIX leaking through — vol-of-vol is a redundant cousin of vol."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the timer, fully costed\n\n"
            "The operational read: hold SPY, go flat (cash) when vol-of-vol is high, 1-day lag, 2 bps "
            "one-way. Net Sharpe vs buy-and-hold, and what the de-risking actually costs in return."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bt = st.timing_backtest(F, STATE)\n"
            "    vals = [bt['net_sharpe'], bt['bh_sharpe']]; anns=[bt['net_ann']*100, bt['bh_ann']*100]\n"
            "    vols=[bt['net_vol']*100, bt['bh_vol']*100]; nt=bt['n_trades']; pf=bt['pct_flat']*100\n"
            "else:\n"
            "    t=R['trade']; vals=[t['net_sh'],t['bh_sh']]; anns=[t['net_ann'],t['bh_ann']]\n"
            "    vols=[t['net_vol'],t['bh_vol']]; nt=t['trades']; pf=t['pct_flat']\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.2,4.2))\n"
            "a1.bar(['VVIX timer','buy & hold'],vals,color=[AMBER,GREY],width=.55); a1.set_ylim(0,.8)\n"
            "for i,v in enumerate(vals): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('net Sharpe'); a1.set_title('Risk-adjusted: a dead heat')\n"
            "a2.bar(['VVIX timer','buy & hold'],anns,color=[AMBER,GREY],width=.55)\n"
            "for i,v in enumerate(anns): a2.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('annual return (%)'); a2.set_title('Return: the timer underperforms')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'timer: Sharpe {vals[0]:.2f}, {anns[0]:.1f}%/yr, vol {vols[0]:.1f}% ({nt} trades, {pf:.0f}% flat) | B&H Sharpe {vals[1]:.2f}, {anns[1]:.1f}%/yr')"
        ),
        md(
            f"> 💡 In plain words: net Sharpe {R['trade']['net_sh']:.2f} vs {R['trade']['bh_sh']:.2f} is a "
            "statistical tie; the timer trades ~3.5pp of return for ~6pp of vol — a swap any static "
            "cash blend offers. Without an incremental signal over the VIX (which is free), there is no "
            "sizing, threshold or cost regime that turns vol-of-vol into a deployable edge. The "
            "constraint is the **redundancy**, not the costs."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The VIX's slope.** [Study 111 — VIX-term-structure](../111-vix-term-structure/): does "
            "contango/backwardation time risk where the level doesn't? Same incremental-over-VIX "
            "discipline applies.\n"
            "- **The premium underneath.** [Study 130 — Vol-risk-premium](../130-vol-risk-premium/): "
            "implied-minus-realized vol, the priced object these second-order signals sit on.\n"
            "- **The cross-section.** [Study 330 — Low-volatility-anomaly](../330-low-volatility-anomaly/): "
            "is *low* vol a premium or disguised beta?\n"
            "- **The VVIX/VIX ratio.** Re-run `incremental_regression` on $W_t/V_t$ — does normalising "
            "vol-of-vol by vol survive the bivariate HAC test where the level didn't?\n\n"
            "*The reproducible core is offline and deterministic; the decisive test is the bivariate "
            "HAC coefficient. Methods and sources: [`docs/references.md`](../docs/references.md); frozen "
            "numbers: [`docs/results.md`](../docs/results.md).*"
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
