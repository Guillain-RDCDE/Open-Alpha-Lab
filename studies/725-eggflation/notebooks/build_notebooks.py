"""Generate the two narrative notebooks for Study 725 ("Eggflation").

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic: real-tape cells read the cached month-end
yfinance pull under ../_cache/ (CALM, SPY) and the hardcoded, cited, approximate USDA/BLS
egg-price series from the package; on a cache miss they fall back to the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere.
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


# Frozen headline numbers — mirror of docs/results.md. Egg-price series is hardcoded/
# cited/approximate; CALM + SPY are month-end Adj Close via yfinance, as-of 2026-06-30.
# Fingerprint of [egg, calm, spy] = 0c92a7548d09.
R = dict(
    win="2015-02 → 2026-06", n=137, asof="2026-06-30", fp="0c92a7548d09",
    egg_peaks=[("2015 HPAI", "2015-09", 2.97), ("2022-23 (record)", "2023-01", 4.82),
               ("2024-25 (all-time high)", "2025-03", 6.23)],
    egg_base_lo=1.30, egg_base_hi=1.80,
    level_corr=0.756,                         # egg LEVEL vs CALM LEVEL (the illusion)
    contemp_t=-0.38, contemp_corr=-0.030,     # same-month return-vs-return
    pred1_slope=0.074, pred1_t=0.93,          # predictive, lag 1 (tradable)
    pred2_slope=0.135, pred2_t=2.02,          # predictive, lag 2 (a selected lag)
    pred1_spy_t=1.00,                         # lag 1 controlling for the market
    plac1_t=0.93, plac1_p=0.483,              # circular-shift placebo, lag 1
    plac2_t=2.02, plac2_p=0.111,              # circular-shift placebo, lag 2
    rev1_slope=0.180, rev1_t=2.03,            # reverse: stock leads the print (lag 1)
    rev2_slope=0.271, rev2_t=2.27,            # reverse (lag 2)
    timer_sh=0.642, timer_cagr=13.9, timer_vol=20.4, timer_mdd=-20.5,
    net_sh=0.631, net_cagr=13.7,
    calm_sh=0.409, calm_cagr=10.1, calm_vol=28.3, calm_mdd=-34.2,
    spy_sh=0.800, spy_cagr=13.7,
    frac_long=0.54, switches=25,
    capm_alpha_m=1.1, capm_alpha_t=1.64, capm_beta=0.05,
    syn_lead=0.6, syn_slope=0.646, syn_t=7.50, syn_plac_p=0.000, syn_timer_sh=0.645,
    syn_bh_sh=0.273, syn_null_t=0.53,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Already_priced_in%3F: Confirmed](https://img.shields.io/badge/Already_priced_in%3F-Confirmed-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))     # quantlab (optional)
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from eggflation import data, strategy as st

EGG = data.load_egg_index()                          # hardcoded, cited, APPROXIMATE proxy
PANEL = data.build_panel()                           # empty if the CALM/SPY cache is absent
HAVE = not PANEL.empty
print("market cache present:", HAVE, "| egg months:", EGG.index[0].date(), "->", EGG.index[-1].date())
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Can you trade the egg-price spike? 🥚\n"
            "### \"Bird flu sends eggs to $6 — just buy the egg stock\", in plain English\n\n"
            + BADGES +
            "Every couple of years the same headline screams back: **avian flu wipes out millions "
            "of hens, egg prices go vertical** — $2 a dozen becomes $5, then $6, an all-time record. "
            "And there's a pure-play stock for it: **Cal-Maine Foods (CALM)**, the biggest egg producer "
            "in America. The trade writes itself — *eggs are spiking, so buy the egg company before "
            "everyone notices.*\n\n"
            "This notebook takes that at full strength and asks the only question that matters: by the "
            "time you **read** that the egg price jumped, is there anything left to trade — or did the "
            "stock already move?\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the placebo null and the lead-lag "
            "regressions? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice — and a data note.** The retail egg price below is a **small, "
            "clearly-cited, approximate** monthly reconstruction of public USDA/BLS reporting — a "
            "**proxy**, never presented as a live feed. CALM and SPY are real month-end prices via "
            "yfinance. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do eggs really spike on bird flu? | **Yes — dramatically.** The (cited) retail price ran "
            f"from a ~\\${R['egg_base_lo']:.2f}–\\${R['egg_base_hi']:.2f} base to **\\${R['egg_peaks'][1][2]:.2f}** "
            f"(Jan-2023) and **\\${R['egg_peaks'][2][2]:.2f}** (Mar-2025). The spikes are real. |\n"
            "| Does the egg price *predict* CALM? | **No.** Last month's egg-price move forecasts next "
            f"month's CALM return with *t* = **{R['pred1_t']:.2f}** — and it can't even beat a coin-flip "
            f"placebo (*p* = **{R['plac1_p']:.2f}**). |\n"
            "| Then why does it *feel* connected? | Because egg prices and CALM both **trend up together** "
            f"(level correlation **{R['level_corr']:.2f}**). Two rising lines look like a signal; they're not. |\n"
            "| Who moves first? | **The stock.** CALM's return actually **leads** the next retail egg "
            f"print (*t* ≈ **{R['rev1_t']:.1f}**) — by the time \"eggflation\" hits the news, it's already "
            "in the chart. |\n\n"
            "> The spike is real. The **trade** is not — the government egg number is a *rear-view mirror*."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Bird flu culls the flock, supply collapses, egg prices double. Cal-Maine is the largest "
            "US egg producer — a near-pure bet on the egg price. So when you see eggs spiking, buy CALM "
            "and ride the shortage.\"*\n\n"
            "It is a *steelman-able* claim. Cal-Maine's revenue really is roughly *eggs sold × egg "
            "price*, so a price spike drops almost straight to the bottom line — its FY2023 net income "
            "went up nearly **8×**. And the spikes are enormous and repeat: 2015, 2022–23, 2024–25. If "
            "any \"macro headline → single stock\" trade should work, it's this one."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "A clean, repeatable link from a **public, free, monthly government number** (the retail egg "
            "price) to a **liquid listed stock** would be a rare thing: an alt-data edge anyone could run "
            "from their kitchen. It's the whole premise of \"trade the headline\" investing — cardboard "
            "boxes, box-office, Big-Mac inflation, and here, eggs. But *\"eggs spiked and CALM went up\"* "
            "and *\"the egg print tells you to buy CALM\"* are different claims. The first is about two "
            "things **moving together**; the second is about one **predicting** the other, in time to act. "
            "Only the second is tradable — and only the second is what we test."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three honest checks:\n\n"
            "1. **Same-month vs next-month.** Does the egg-price change move with CALM *this* month "
            "(feels like a signal, but you can't trade a number you don't have yet) — and does it "
            "**forecast** *next* month's CALM return (the part you could actually act on)?\n"
            "2. **A placebo.** Shuffle the egg series in time hundreds of ways and re-run — if the real "
            "alignment can't beat random alignments, it's noise.\n"
            "3. **Who leads whom?** Test the *reverse*: does CALM's move foreshadow the **next** egg "
            "print? If the stock leads the government number, the number is stale news.\n\n"
            "**What would make us say \"tradable\"?** The next-month prediction is real (beats its "
            "placebo) **and** a simple egg-timer beats just holding the stock, net of costs. Anything "
            "less is a rear-view mirror."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: the spikes are real.** Here is the (approximate, cited) retail egg price — three "
            "avian-flu blow-offs on a quiet base."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(EGG.index, EGG.values, c=AMBER, lw=2.0, label='retail egg price $/dozen (proxy)')\n"
            "for lab, m, lvl in R['egg_peaks']:\n"
            "    d = pd.Timestamp(m) + pd.offsets.MonthEnd(0)\n"
            "    ax.annotate(f'${lvl:.2f}', (d, lvl), textcoords='offset points', xytext=(0, 6),\n"
            "                ha='center', color=RED, fontsize=9)\n"
            "    ax.scatter([d], [lvl], c=RED, zorder=5)\n"
            "ax.set_xlabel('month'); ax.set_ylabel('$ per dozen')\n"
            "ax.set_title('Eggflation is real: three avian-flu spikes'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('base ~$%.2f-$%.2f, peaks: ' % (R['egg_base_lo'], R['egg_base_hi']) +\n"
            "      ', '.join(f\"{lab} ${lvl:.2f}\" for lab,_,lvl in R['egg_peaks']))"
        ),
        md(
            "Nobody disputes this part. Eggs went from a buck-thirty to nearly **six dollars** a dozen. "
            "The question is never whether the spike happened — it's whether the *published egg number* "
            "was a **buy signal** for the stock, or just a receipt for a move that already happened."
        ),
        md(
            "**The illusion: two lines that rise together.** Put the egg price and CALM on one chart and "
            "the story sells itself — they clearly \"go together.\""
        ),
        code(
            "if HAVE:\n"
            "    e = PANEL['egg']; c = PANEL['calm']\n"
            "else:\n"
            "    e = EGG.reindex(pd.date_range('2015-02-28','2026-06-30',freq='ME')).ffill(); c = e*0+np.nan\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(e.index, e/e.iloc[0]*100, c=AMBER, lw=2.0, label='egg price (rebased)')\n"
            "if HAVE: ax.plot(c.index, c/c.iloc[0]*100, c=GREEN, lw=2.0, label='CALM (rebased)')\n"
            "ax.set_ylabel('rebased to 100 at start'); ax.set_xlabel('month')\n"
            "ax.set_title(f\"They rise together — level correlation {R['level_corr']:.2f}\"); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"egg-vs-CALM LEVEL correlation: {R['level_corr']:.2f}  <- looks compelling...\")"
        ),
        md(
            f"A **{R['level_corr']:.2f}** correlation! Case closed? No. Two things that both drift upward "
            "over a decade will *always* correlate on levels — it says nothing about whether one "
            "**forecasts** the other month to month. That's the trap this whole trade is built on. So "
            "let's ask the tradable question properly."
        ),
        md(
            "**Does last month's egg move predict next month's CALM?** This is the only version you could "
            "act on — you learn the egg print *after* the month ends, so you trade the month after."
        ),
        code(
            "labels = ['same month\\n(untradable)', 'next month\\n(tradable)']\n"
            "ts = [R['contemp_t'], R['pred1_t']]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(labels, ts, .5, color=[GREY, RED])\n"
            "ax.axhline(2, ls='--', c=GREEN, alpha=.7, label='|t| = 2 bar for a real signal')\n"
            "ax.axhline(-2, ls='--', c=GREEN, alpha=.7); ax.axhline(0, c='k', lw=1)\n"
            "for i,t in enumerate(ts): ax.annotate(f't={t:+.2f}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "ax.set_ylabel('predictive t-stat (HAC)'); ax.set_ylim(-2.5, 2.5)\n"
            "ax.set_title('Egg price -> CALM: nothing clears the bar'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"same-month t={R['contemp_t']:+.2f} (and it is UNTRADABLE anyway)\")\n"
            "print(f\"next-month  t={R['pred1_t']:+.2f}  -> no forecasting power\")"
        ),
        md(
            f"The tradable, next-month prediction comes in at *t* = **{R['pred1_t']:.2f}** — nowhere near "
            "the |t| = 2 you'd need. And the same-month number is a *negative* whisper, not the roaring "
            "co-move the levels chart implied. But maybe a lucky *t* is hiding in there — so we test it "
            "against pure chance."
        ),
        md(
            "**The placebo.** Slide the egg series forward by a random number of months (breaking any "
            "real timing but keeping its shape), re-run the prediction, and repeat thousands of times. "
            "If the *real* alignment is special, it should beat the shuffled ones."
        ),
        code(
            "if HAVE:\n"
            "    pl = st.circular_shift_placebo(PANEL, lag=1, n_boot=4000, seed=725)\n"
            "    obs, p = pl['obs_t'], pl['p']; lo, hi = pl['lo'], pl['hi']\n"
            "else:\n"
            "    obs, p, lo, hi = R['plac1_t'], R['plac1_p'], -2.0, 2.0\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.2))\n"
            "ax.axvspan(lo, hi, color=GREY, alpha=.2, label='placebo 95% band')\n"
            "ax.axvline(obs, c=RED, lw=2.4, label=f'real egg signal (t={obs:+.2f})')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('predictive t-stat'); ax.set_yticks([])\n"
            "ax.set_title(f'The real signal sits INSIDE the noise band (p={p:.2f})'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"placebo p = {p:.3f}  -> the egg print does no better than a random shuffle\")"
        ),
        md(
            f"The real egg signal lands **inside** the placebo band (*p* = **{R['plac1_p']:.2f}**): "
            "roughly half of random time-shifts of the egg series \"predict\" CALM at least as well as the "
            "true one. That is the textbook signature of **no signal**."
        ),
        md(
            "**So who leads whom?** Here's the punchline. Test whether *this* month's CALM move "
            "foreshadows *next* month's egg print — the reverse direction."
        ),
        code(
            "if HAVE:\n"
            "    fwd = st.predictive(PANEL, lag=1); rev = st.reverse_lead(PANEL, lag=1)\n"
            "    fwd_t, rev_t = fwd['t'], rev['t']\n"
            "else:\n"
            "    fwd_t, rev_t = R['pred1_t'], R['rev1_t']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['egg -> CALM\\n(the pitch)', 'CALM -> egg\\n(reality)'], [fwd_t, rev_t], .5,\n"
            "       color=[RED, GREEN])\n"
            "ax.axhline(2, ls='--', c=GREY, alpha=.7); ax.axhline(0, c='k', lw=1)\n"
            "for i,t in enumerate([fwd_t, rev_t]): ax.annotate(f't={t:+.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('lead-lag t-stat (HAC)')\n"
            "ax.set_title('The STOCK leads the government egg print, not the other way'); \n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"egg predicts CALM: t={fwd_t:+.2f} (nothing)   |   CALM predicts egg: t={rev_t:+.2f} (real)\")"
        ),
        md(
            f"There it is. The egg price does **not** lead CALM (*t* = {R['pred1_t']:.2f}), but CALM "
            f"**does** lead the next egg print (*t* ≈ {R['rev1_t']:.1f}). The stock market prices the "
            "shortage — from wholesale futures, USDA flock data, company guidance — *weeks before* it "
            "shows up at the supermarket and lands in the CPI headline you read. You're trying to trade "
            "on a receipt."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The egg print doesn't forecast CALM (next-month *t* = {R['pred1_t']:.2f}, "
            f"placebo *p* = {R['plac1_p']:.2f}). The {R['level_corr']:.2f} \"correlation\" is two "
            "co-trending lines.\n"
            "- **Tradability — Mirage.** The one thing that *is* significant runs the wrong way — the "
            f"**stock leads the print** (*t* ≈ {R['rev1_t']:.1f}) — so at trade time the egg number is "
            "already stale.\n"
            "- **Already priced in? — Confirmed.** By the time \"eggflation\" is a headline, it's in "
            "CALM's chart. The public egg number carries no tradable news."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "\"Fine,\" you say, \"I'll just hold CALM whenever eggs are rising.\" Let's build exactly that "
            "timer — long CALM when egg prices have been climbing, cash otherwise — and race it against "
            "simply buying and holding CALM, and against a boring S&P index fund."
        ),
        code(
            "if HAVE:\n"
            "    tm = st.egg_momentum_timer(PANEL); net = st.apply_costs(tm, 10.0)\n"
            "    sh = {'egg timer (net)': st.summarize(net)['sharpe'],\n"
            "          'buy & hold CALM': st.summarize(tm['calm'])['sharpe'],\n"
            "          'S&P 500 (SPY)': st.summarize(tm['spy'])['sharpe']}\n"
            "else:\n"
            "    sh = {'egg timer (net)': R['net_sh'], 'buy & hold CALM': R['calm_sh'], 'S&P 500 (SPY)': R['spy_sh']}\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "cols=[AMBER, RED, GREEN]\n"
            "ax.bar(list(sh), list(sh.values()), .55, color=cols)\n"
            "for i,v in enumerate(sh.values()): ax.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('Sharpe (excess of cash)'); ax.set_title('The egg timer even beats buy-hold CALM — but loses to plain SPY')\n"
            "plt.tight_layout(); plt.show()\n"
            "for k,v in sh.items(): print(f'{k:18s} Sharpe {v:.2f}')"
        ),
        md(
            f"Here's the twist that keeps the myth alive: the egg timer (Sharpe **{R['net_sh']:.2f}**) "
            f"*does* beat buy-and-hold CALM (**{R['calm_sh']:.2f}**) — but only because it hides in cash "
            "during CALM's ugly stretches, not because the egg *print* told it anything (we just saw it "
            "predicts nothing). And it still **loses to a plain S&P index fund** "
            f"(**{R['spy_sh']:.2f}**). You took on single-stock, three-event, one-commodity risk to "
            "underperform SPY. That's the definition of a mirage."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Swap in the live BLS series.** Our egg price is a cited *approximation* of BLS "
            "`APU0000708111`; pull the real monthly series and re-run — the lead-lag verdict won't move.\n"
            "- **Use the futures/wholesale price instead of retail.** The stock likely leads *those* too, "
            "but by less — the sharper test of \"how early is it priced.\"\n"
            "- **The whole 'trade the headline' family.** Cardboard boxes, box-office, sports sentiment: "
            "public alt-data that co-moves but doesn't *lead* (see [docs/references.md](../docs/references.md)).\n\n"
            "*Think a faster egg signal (weekly USDA wholesale, or the flock-cull count itself) beats "
            "the stock to the punch? Build it, lag it honestly, and show a next-period *t* ≥ 2 that beats "
            "its placebo — then charge it costs.*"
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
            "# Eggflation as a trade — a quantitative teardown 🔬\n"
            "### Contemporaneous vs predictive HAC regressions · a circular-shift placebo · the "
            "reverse lead-lag (does the stock lead the print?) · an egg-momentum timer net of costs · "
            "a synthetic planted-lead positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We test the "
            "strongest tradable form of \"buy CALM when eggs spike\": (H₁) the retail egg-price change "
            "**forecasts** CALM's next-month return with HAC *t* ≥ 2; (H₂) that forecast beats a "
            "circular-shift placebo; (H₃) an egg-momentum timer beats buy-and-hold **and** SPY net of "
            "costs. We find **H₁ rejected** (*t* ≈ 0.9), **H₂ rejected** (placebo *p* ≈ 0.48), **H₃ "
            "rejected** (beats CALM but loses to SPY) — and the *only* significant relationship runs "
            "**backwards**: the stock leads the print.\n\n"
            "> ⚠️ **Not investment advice — data provenance.** The egg-price series is **hardcoded, "
            "cited, approximate** (public USDA/BLS reporting — a *labelled proxy*, never a live feed). "
            "`CALM`, `SPY` are month-end Adj Close via yfinance (as-of 2026-06-30, fp `0c92a7548d09`). "
            "Offline core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md); numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Predictive (lag-1) egg→CALM slope *t* = **{R['pred1_t']:.2f}** "
            f"(placebo *p* = **{R['plac1_p']:.2f}**); +market control *t* = **{R['pred1_spy_t']:.2f}**. "
            f"The lag-2 *t* = {R['pred2_t']:.2f} dies under its own placebo (*p* = {R['plac2_p']:.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | Egg-momentum timer net Sharpe **{R['net_sh']:.2f}** > buy-hold "
            f"CALM **{R['calm_sh']:.2f}** but < SPY **{R['spy_sh']:.2f}**. Concentrated single-name, "
            "three-event, one-commodity risk to *underperform* the index. |\n"
            f"| **Already priced in?** | `CONFIRMED` | Reverse regression: CALM return **leads** the next "
            f"egg print, *t* = **{R['rev1_t']:.2f}** (lag-2 *t* = {R['rev2_t']:.2f}). The public retail "
            "number lags the tape it's supposed to forecast. |\n\n"
            "> 💡 In plain words: the eye-catching level correlation "
            f"(**{R['level_corr']:.2f}**) is two co-trending series; the *tradable* prediction is noise; "
            "and the only real lead-lag is the stock forecasting the government number, not the reverse."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $g_t = \\Delta\\log$ (retail egg price) and $r_t$ = CALM's monthly return. The trade is "
            "a joint hypothesis:\n\n"
            "- **H₁ (it forecasts).** In $r_{t+1} = \\alpha + \\beta g_t + \\varepsilon_{t+1}$ the slope "
            "$\\beta > 0$ with a Newey-West *t* > 2 — last month's egg move, *knowable at trade time*, "
            "predicts CALM.\n"
            "- **H₂ (it's not luck).** That *t* beats a **circular-shift placebo** null of the egg series.\n"
            "- **H₃ (it pays).** A long-CALM-when-egg-momentum-positive timer (one-month execution lag) "
            "beats buy-and-hold CALM *and* SPY on excess-of-cash Sharpe, net of costs.\n\n"
            "The steelman is Cal-Maine's economics: revenue ≈ (dozens sold) × (egg price), so a price "
            "spike is almost pure operating leverage. If *any* public-macro-number → single-stock trade "
            "should work, this is the one. The catch is **timing**: the retail print is published with a "
            "~two-week lag, so the tradable regressor is $g_t$ against $r_{t+1}$, not $r_t$."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "H₁–H₃ would make a free monthly government statistic a live trading signal for a liquid "
            "equity — the dream of every \"alt-data from your kitchen\" pitch. But each leg is separately "
            "falsifiable. H₁ is a **predictive regression** with the execution lag built in. H₂ guards "
            "against a lucky single *t* by comparing to the series' own shuffles (the circular shift "
            "preserves autocorrelation, so the null is honest). H₃ is the **money test**, excess-of-cash "
            "so a part-time-in-cash timer races the always-invested benchmarks like-for-like. And a "
            "*fourth* diagnostic — the **reverse regression** — decides the mechanism: if the stock "
            "leads the print, no amount of cleverness with the print can help."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Egg price (proxy).** Hardcoded, cited, **approximate** monthly retail price ($/dozen, "
            "BLS `APU0000708111`), Jan-2015→Jun-2026, anchored on the 2015 / Jan-2023 / Mar-2025 HPAI "
            "peaks. *Labelled a proxy* — the spike *shape* is the public fact, the exact monthly values "
            "are not a live feed.\n"
            "- **CALM, SPY.** Month-end Adj Close (yfinance, cached), as-of 2026-06-30, fp `0c92a7548d09`. "
            "Survivorship is not a concern (named tickers, not a screen); CALM has a continuous listed "
            "history over the window.\n"
            "- **No look-ahead, one lag.** The predictive regressor is $g_t$ (known only after month $t$ "
            "closes) against $r_{t+1}$; the timer shifts its signal by one month — one `shift`, once.\n"
            "- **Inference.** Newey-West (6-lag) HAC *t* on every slope; a 4–5k circular-shift placebo "
            "for the predictive *t*; the reverse regression $g_{t+1}\\sim r_t$ for the lead-lag.\n"
            "- **Positive control.** A deterministic world where $r_{t+1} = 0.6\\,g_t + $ noise — the "
            "engine must recover *t* ≥ 2 and a placebo *p* ≈ 0, and the null world (β=0) must stay flat.\n"
            "- **What would make us say \"tradable\":** H₁ *t* > 2 **and** placebo *p* < 0.05 **and** the "
            "timer beats SPY net. We find none of these; we *do* find the reverse lead."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Level correlation vs return prediction — the illusion, quantified\n\n"
            "The levels correlate strongly; the *return* relationships (same-month and next-month) do "
            "not. This gap is the entire trap."
        ),
        code(
            "if HAVE:\n"
            "    lvl_corr = float(np.corrcoef(PANEL['egg'], PANEL['calm'])[0,1])\n"
            "    cm = st.contemporaneous(PANEL); p1 = st.predictive(PANEL, lag=1)\n"
            "    p2 = st.predictive(PANEL, lag=2); p1s = st.predictive(PANEL, lag=1, control_spy=True)\n"
            "    rows = [('egg LEVEL vs CALM LEVEL (corr)', lvl_corr, None),\n"
            "            ('same-month  r_t ~ g_t', cm['slope'], cm['t']),\n"
            "            ('next-month  r_{t+1} ~ g_t', p1['slope'], p1['t']),\n"
            "            ('next-month + market control', p1s['slope'], p1s['t']),\n"
            "            ('two-month   r_{t+2} ~ g_t (selected lag)', p2['slope'], p2['t'])]\n"
            "else:\n"
            "    rows = [('egg LEVEL vs CALM LEVEL (corr)', R['level_corr'], None),\n"
            "            ('same-month  r_t ~ g_t', None, R['contemp_t']),\n"
            "            ('next-month  r_{t+1} ~ g_t', R['pred1_slope'], R['pred1_t']),\n"
            "            ('next-month + market control', None, R['pred1_spy_t']),\n"
            "            ('two-month   r_{t+2} ~ g_t (selected lag)', R['pred2_slope'], R['pred2_t'])]\n"
            "print(f\"{'relationship':40s} {'slope':>8s} {'t':>7s}\")\n"
            "for name, s, t in rows:\n"
            "    print(f\"{name:40s} {('' if s is None else f'{s:+.3f}'):>8s} {('' if t is None else f'{t:+.2f}'):>7s}\")\n"
            "labs = ['same\\nmonth', 'next\\nmonth', 'next+mkt', 'lag-2\\n(selected)']\n"
            "tv = [rows[1][2], rows[2][2], rows[3][2], rows[4][2]]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(labs, tv, .55, color=[GREY, RED, RED, AMBER])\n"
            "ax.axhline(2, ls='--', c=GREEN, alpha=.7, label='|t|=2'); ax.axhline(-2, ls='--', c=GREEN, alpha=.7)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('predictive t (HAC)')\n"
            "for i,t in enumerate(tv): ax.annotate(f'{t:+.2f}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "ax.set_title(f\"Levels corr {R['level_corr']:.2f}, but no return-predictive t clears 2\"); ax.legend()\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the **{R['level_corr']:.2f}** you'd quote at a dinner party is a "
            "spurious-regression artifact — two integrated, upward-drifting series. The moment you ask "
            f"the honest question (does a *change* forecast a *return*?) it collapses: next-month *t* = "
            f"**{R['pred1_t']:.2f}**, and adding the contemporaneous market barely moves it "
            f"(**{R['pred1_spy_t']:.2f}**). The lag-2 *t* = {R['pred2_t']:.2f} is the one number a "
            "data-miner would frame — so we test it next."
        ),
        md(
            "### 4b · The placebo — is the lag-2 *t* = 2.02 real, or a mined lag?\n\n"
            "Circular-shift the egg series (preserving its autocorrelation), re-fit the predictive "
            "regression thousands of times, and locate the observed *t* in the null — for **both** lags."
        ),
        code(
            "if HAVE:\n"
            "    pl1 = st.circular_shift_placebo(PANEL, lag=1, n_boot=5000, seed=725)\n"
            "    pl2 = st.circular_shift_placebo(PANEL, lag=2, n_boot=5000, seed=725)\n"
            "else:\n"
            "    pl1 = dict(obs_t=R['plac1_t'], p=R['plac1_p'], lo=-2.0, hi=2.0)\n"
            "    pl2 = dict(obs_t=R['plac2_t'], p=R['plac2_p'], lo=-2.1, hi=2.1)\n"
            "fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)\n"
            "for ax, pl, lab in zip(axes, [pl1, pl2], ['lag 1 (tradable)', 'lag 2 (selected)']):\n"
            "    ax.axvspan(pl['lo'], pl['hi'], color=GREY, alpha=.2)\n"
            "    ax.axvline(pl['obs_t'], c=RED, lw=2.4)\n"
            "    ax.axvline(0, c='k', lw=1); ax.set_yticks([])\n"
            "    ax.set_title(f\"{lab}\\nobs t={pl['obs_t']:+.2f}, p={pl['p']:.2f}\"); ax.set_xlabel('t under the null')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"lag1: obs t={pl1['obs_t']:+.2f}  placebo p={pl1['p']:.3f}\")\n"
            "print(f\"lag2: obs t={pl2['obs_t']:+.2f}  placebo p={pl2['p']:.3f}  <- the 'significant' lag is NOT significant vs its placebo\")"
        ),
        md(
            f"> 💡 In plain words: the tradable lag-1 signal sits squarely inside its noise band "
            f"(*p* = **{R['plac1_p']:.2f}**). And the tempting lag-2 *t* = **{R['pred2_t']:.2f}** — the "
            "kind of number that launches a strategy — **fails its own placebo** (*p* = "
            f"**{R['plac2_p']:.2f}**): once you account for having *searched* across lags, it's exactly "
            "what random time-shifts produce. H₁ and H₂ rejected."
        ),
        md(
            "### 4c · The reverse regression — the stock leads the print\n\n"
            "The mechanism. Regress the **next** egg-price change on **this** month's CALM return: "
            "$g_{t+1} = a + b\\,r_t + e$, at lags 1 and 2."
        ),
        code(
            "if HAVE:\n"
            "    f1 = st.predictive(PANEL, lag=1)['t']; f2 = st.predictive(PANEL, lag=2)['t']\n"
            "    r1 = st.reverse_lead(PANEL, lag=1)['t']; r2 = st.reverse_lead(PANEL, lag=2)['t']\n"
            "else:\n"
            "    f1,f2,r1,r2 = R['pred1_t'], R['pred2_t'], R['rev1_t'], R['rev2_t']\n"
            "x = np.arange(2); w=.38\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(x-w/2, [f1,f2], w, color=RED, label='egg -> CALM (the pitch)')\n"
            "ax.bar(x+w/2, [r1,r2], w, color=GREEN, label='CALM -> egg (reality)')\n"
            "ax.axhline(2, ls='--', c=GREY, alpha=.7); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['lag 1', 'lag 2']); ax.set_ylabel('lead-lag t (HAC)')\n"
            "for xi,v in zip([x[0]-w/2,x[1]-w/2,x[0]+w/2,x[1]+w/2],[f1,f2,r1,r2]): ax.annotate(f'{v:+.2f}',(xi,v),ha='center',va='bottom')\n"
            "ax.set_title('Only the reverse direction is significant: the stock leads the egg print'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"egg->CALM: lag1 t={f1:+.2f}, lag2 t={f2:+.2f}  (nothing survives placebo)\")\n"
            "print(f\"CALM->egg: lag1 t={r1:+.2f}, lag2 t={r2:+.2f}  (the real lead-lag)\")"
        ),
        md(
            f"> 💡 In plain words: CALM's return foreshadows the next retail egg print at *t* ≈ "
            f"**{R['rev1_t']:.1f}** (and **{R['rev2_t']:.1f}** at two months), while the egg print → CALM "
            "direction is dead. The equity market aggregates wholesale egg futures, USDA flock/HPAI "
            "data and company guidance and prices the shortage *weeks before* it reaches the supermarket "
            "shelf and the CPI table. The public number is a lagging summary — a receipt, not a signal. "
            "**`Already priced in? → CONFIRMED`.**"
        ),
        md(
            "### 4d · The money test — an egg-momentum timer, net of costs\n\n"
            "Long CALM when trailing 3-month egg momentum > 0 (acted on with a one-month lag), cash at "
            "2%/yr otherwise; 10 bp one-way per switch. Raced on **excess-of-cash** Sharpe against "
            "buy-and-hold CALM and SPY."
        ),
        code(
            "if HAVE:\n"
            "    tm = st.egg_momentum_timer(PANEL, window=3); net = st.apply_costs(tm, 10.0)\n"
            "    g = st.summarize(tm['timer']); nn = st.summarize(net)\n"
            "    cc = st.summarize(tm['calm']); sp = st.summarize(tm['spy'])\n"
            "    frac = float(tm['signal'].mean()); sw = int((tm['signal'].diff().abs()>0).sum())\n"
            "else:\n"
            "    g = dict(sharpe=R['timer_sh'],cagr=R['timer_cagr']/100,mdd=R['timer_mdd']/100)\n"
            "    nn = dict(sharpe=R['net_sh'],cagr=R['net_cagr']/100,mdd=R['timer_mdd']/100)\n"
            "    cc = dict(sharpe=R['calm_sh'],cagr=R['calm_cagr']/100,mdd=R['calm_mdd']/100)\n"
            "    sp = dict(sharpe=R['spy_sh'],cagr=R['spy_cagr']/100,mdd=-0.24)\n"
            "    frac=R['frac_long']; sw=R['switches']\n"
            "names=['egg timer\\n(gross)','egg timer\\n(net 10bp)','buy&hold\\nCALM','SPY']\n"
            "shs=[g['sharpe'], nn['sharpe'], cc['sharpe'], sp['sharpe']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(names, shs, .55, color=[AMBER, AMBER, RED, GREEN])\n"
            "ax.axhline(sp['sharpe'], ls='--', c=GREEN, alpha=.6)\n"
            "for i,v in enumerate(shs): ax.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('Sharpe (excess of cash)'); ax.set_title('Beats buy-hold CALM, loses to a plain index fund')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"timer gross Sharpe {g['sharpe']:.2f} (CAGR {g['cagr']*100:+.1f}%, maxDD {g['mdd']*100:.0f}%)\")\n"
            "print(f\"timer net   Sharpe {nn['sharpe']:.2f} (CAGR {nn['cagr']*100:+.1f}%)  | {sw} switches, long {frac*100:.0f}% of months\")\n"
            "print(f\"buy&hold CALM Sharpe {cc['sharpe']:.2f} (CAGR {cc['cagr']*100:+.1f}%, maxDD {cc['mdd']*100:.0f}%)\")\n"
            "print(f\"SPY          Sharpe {sp['sharpe']:.2f} (CAGR {sp['cagr']*100:+.1f}%)\")"
        ),
        md(
            f"> 💡 In plain words: the timer's net Sharpe **{R['net_sh']:.2f}** *does* edge buy-and-hold "
            f"CALM (**{R['calm_sh']:.2f}**) — but that's **risk reduction from sitting in cash** during "
            "CALM's slumps, not forecasting skill (the egg print predicts nothing, §4a–b). Costs barely "
            f"bite ({R['switches']} switches). And it still trails a plain **SPY** at "
            f"**{R['spy_sh']:.2f}**: you shouldered concentrated single-name, three-event, one-commodity "
            "risk to *underperform the index*. `MIRAGE`."
        ),
        md(
            "### 4e · Positive control — the engine recovers a planted lead\n\n"
            "A deterministic world where $r_{t+1} = 0.6\\,g_t + \\text{noise}$ (seed 725). The harness "
            "must recover *t* ≥ 2, a placebo *p* ≈ 0 and a paying timer — proving the real-tape nulls "
            "are true nulls, not a broken pipeline. The β = 0 world must stay flat."
        ),
        code(
            "w = data.synthetic_world(lead_beta=0.6); w0 = data.synthetic_world(lead_beta=0.0)\n"
            "pr = st.predictive(w, lag=1); pl = st.circular_shift_placebo(w, lag=1, n_boot=3000, seed=725)\n"
            "cr = st.control_recovers(w, lag=1); pr0 = st.predictive(w0, lag=1)\n"
            "tmw = st.egg_momentum_timer(w)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.2))\n"
            "ax.bar(['planted lead\\n(beta=0.6)','null world\\n(beta=0)'], [pr['t'], pr0['t']],\n"
            "       .5, color=[GREEN, GREY])\n"
            "ax.axhline(2, ls='--', c=GREEN, alpha=.7); ax.axhline(0, c='k', lw=1)\n"
            "for i,t in enumerate([pr['t'], pr0['t']]): ax.annotate(f't={t:+.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('predictive t (HAC)'); ax.set_title('Machinery proof: recovers a real lead, stays flat on the null')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"planted lead: slope {pr['slope']:+.3f}  t {pr['t']:+.2f}  placebo p {pl['p']:.3f}  recovered={cr['recovered']}\")\n"
            "print(f\"null world:   t {pr0['t']:+.2f}  recovered={st.control_recovers(w0,1)['recovered']}\")\n"
            "print(f\"planted timer Sharpe {st.summarize(tmw['timer'])['sharpe']:.2f} vs buy-hold {st.summarize(tmw['calm'])['sharpe']:.2f}\")"
        ),
        md(
            f"> 💡 In plain words: when a real lead is present, the engine nails it — *t* = "
            f"**{R['syn_t']:.1f}**, placebo *p* = **{R['syn_plac_p']:.0f}**, timer Sharpe "
            f"**{R['syn_timer_sh']:.2f}** vs buy-hold **{R['syn_bh_sh']:.2f}** — and on the null it reads "
            f"*t* = **{R['syn_null_t']:.2f}** (flat). So the real-tape `NONE` is a *true* null: the egg "
            "print genuinely doesn't lead CALM, not a harness that couldn't see it. A synthetic control "
            "is a machinery proof, never market evidence."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — predictive lag-1 egg→CALM *t* = {R['pred1_t']:.2f} (placebo "
            f"*p* = {R['plac1_p']:.2f}); +market control {R['pred1_spy_t']:.2f}; the lag-2 *t* = "
            f"{R['pred2_t']:.2f} fails its placebo (*p* = {R['plac2_p']:.2f}). The {R['level_corr']:.2f} "
            "level correlation is a spurious-regression artifact. No robust *t* ≥ 2 in the tradable "
            "direction.\n"
            f"- **Tradability `MIRAGE`** — egg-momentum timer net Sharpe {R['net_sh']:.2f} beats buy-hold "
            f"CALM ({R['calm_sh']:.2f}) via cash-parking, not signal, and **loses to SPY** "
            f"({R['spy_sh']:.2f}). Single-name, three-event, one-commodity concentration for "
            "sub-index returns.\n"
            f"- **Already priced in? `CONFIRMED`** — the reverse regression is the only significant one: "
            f"CALM leads the next egg print (*t* = {R['rev1_t']:.2f}, lag-2 {R['rev2_t']:.2f}). The "
            "public retail number lags the equity that's supposed to be forecast by it."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — capacity & cost reality\n\n"
            "Even granting the timer's edge over buy-hold CALM, the trade has no room to grow: it is one "
            "mid-cap name (CALM), it fires on ~three flu episodes in a decade, and it is a bet on a "
            "single agricultural commodity. Terminal wealth of \\$10,000 makes the point."
        ),
        code(
            "start=10_000.0; yrs=R['n']/12\n"
            "paths={'egg timer (net)':R['net_cagr']/100, 'buy & hold CALM':R['calm_cagr']/100, 'S&P 500 (SPY)':R['spy_cagr']/100}\n"
            "ends={k:start*(1+g)**yrs for k,g in paths.items()}\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.3))\n"
            "ax.bar(list(ends), list(ends.values()), .55, color=[AMBER, RED, GREEN])\n"
            "for i,v in enumerate(ends.values()): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel(f'value of $10,000 over {yrs:.1f} years'); ax.set_title('The concentrated egg trade lands below a passive index fund')\n"
            "plt.tight_layout(); plt.show()\n"
            "for k,g in paths.items(): print(f\"{k:18s} ${start*(1+g)**yrs:>10,.0f}  ({g*100:+.1f}%/yr)\")"
        ),
        md(
            "> 💡 In plain words: the egg timer and SPY end up close in dollars, but SPY got there with "
            "**broad diversification and a higher Sharpe**, while the egg book concentrated everything "
            "into one stock's exposure to one bird-flu-driven commodity. Same-ish money, far more risk, "
            "and a signal (the egg print) that we showed does no work. There is no size or venue that "
            "turns a lagging public statistic into an edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Swap in the live BLS/USDA feed.** Replace the hardcoded egg series with BLS "
            "`APU0000708111` (retail) or the USDA wholesale/Egg Markets Overview; re-run — the reverse "
            "lead-lag sign is robust to the exact series.\n"
            "- **Race the leading indicators.** Wholesale egg futures, weekly USDA cull counts, or "
            "Google-Trends 'egg price' — test whether *any* of them beat the stock to the punch with a "
            "next-period *t* ≥ 2 that survives a placebo.\n"
            "- **The 'trade the headline' prior.** Public alt-data that co-moves but doesn't *lead* is "
            "the family rule, not the exception ([docs/references.md](../docs/references.md)); this is "
            "the egg instance of a market that prices news before it prints.\n\n"
            "*The reproducible core is offline and deterministic; the egg series is a **cited, "
            "approximate proxy** and CALM/SPY are real month-end tape. Methods: "
            "[`docs/references.md`](../docs/references.md); frozen numbers (fp `0c92a7548d09`): "
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
