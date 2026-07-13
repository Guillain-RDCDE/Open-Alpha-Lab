"""Generate the two narrative notebooks for Study 728 ("beer is recession-proof").

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic: real-tape cells read the cached month-end
yfinance pulls under ../_cache/ (TAP, SAM, SPY) and the hardcoded, cited NBER recession
windows from the package; on a cache miss they fall back to the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic asymmetric-beta control runs anywhere.
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


# Frozen headline numbers — mirror of docs/results.md (TAP/SAM/SPY month-end Adj Close via
# yfinance, as-of 2026-06-30; NBER recession windows hardcoded/cited).
R = dict(
    win="1995 → 2026",
    n_spy=377, n_sam=367,
    # full-sample risk/return
    tap_cagr=7.32, tap_vol=26.9, tap_sharpe=0.40, tap_mdd=-66.4,
    sam_cagr=6.52, sam_vol=37.4, sam_sharpe=0.36, sam_mdd=-85.4,
    spy_cagr=11.12, spy_vol=15.1, spy_sharpe=0.78, spy_mdd=-50.8,
    # CAPM alpha vs SPY (Newey-West, 6-lag)
    tap_alpha=3.25, tap_beta=0.64, tap_t=0.71, tap_p=0.480,
    sam_alpha=5.91, sam_beta=0.68, sam_t=0.84, sam_p=0.403,
    # bull/bear beta (split at SPY=0)
    tap_down_beta=0.60, tap_up_beta=0.70, tap_asym=-0.10, tap_def=1,
    sam_down_beta=0.87, sam_up_beta=0.79, sam_asym=+0.09, sam_def=0,
    n_down=130, n_up=247,
    # recession-window excess vs SPY (paired t)
    rec_n=31,
    tap_rec_mean=-1.44, tap_rec_excess=0.25, tap_rec_t=0.14, tap_rec_p=0.891,
    tap_rec_cum=-48.1,
    sam_rec_mean=2.42, sam_rec_excess=4.12, sam_rec_t=1.87, sam_rec_p=0.071,
    sam_rec_cum=64.3,
    spy_rec_mean=-1.70, spy_rec_cum=-45.6,
    # per-recession compounded % (stock vs SPY)
    rec_labels=["2001 dot-com", "2008 GFC", "2020 COVID"],
    tap_by_rec=[-13.8, -19.3, -25.4],
    sam_by_rec=[40.8, -10.8, 30.9],
    spy_by_rec=[-7.1, -35.5, -9.2],
    # tradability — $1 buy-and-hold terminal wealth
    tap_wealth=9.19, sam_wealth=6.91, spy_wealth=27.45, hold_yrs=31.4,
    nber_lag=12,
    # synthetic control
    syn_down=0.48, syn_up=0.77, syn_plant_down=0.50, syn_plant_up=1.00,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Counter--cyclical%3F: Busted](https://img.shields.io/badge/Counter--cyclical%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from beer_recession import data, strategy as st

HAVE_PRICES = data.have_prices()
PX = data.load_prices() if HAVE_PRICES else None
print("price cache present:", HAVE_PRICES,
      "| NBER recessions in sample:", [r[0] for r in data.NBER_RECESSIONS])
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Is beer recession-proof? 🍺\n"
            "### \"People drink more when times are hard\" — do the beer stocks actually hold up?\n\n"
            + BADGES +
            "It's one of those things everyone *knows*: when the economy tanks, people don't stop "
            "drinking — they drink **more**. Cheap comfort, a pint at the end of a bad week. So beer "
            "must be **recession-proof**: while the market crashes, a brewer's stock should sail "
            "through, a safe harbour you can actually own.\n\n"
            "This notebook puts that folklore on the clock. We take the two beer stocks you can buy — "
            "**Molson Coors (`TAP`)** and **Boston Beer (`SAM`, the Sam Adams people)** — line them up "
            "next to the **S&P 500**, and ask two blunt questions: *when the market falls, do they fall "
            "less?* and *when the economy is actually in recession, do they beat the index?*\n\n"
            "> 📓 **Plain-language layer.** Want the downside-beta regressions, the Newey-West alpha and "
            "the recession-window *t*-stats? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice — and a data note.** Prices are real (month-end, yfinance). "
            "The recession dates are the **NBER**'s official ones (a cited, dated set of facts — not a "
            "live feed). Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do beer stocks fall *less* than the market? | **Sometimes — and only one of them.** "
            f"Molson Coors is a low-beta stock (β≈{R['tap_beta']:.2f}), so it swings less than the S&P. "
            f"But Boston Beer's down-market beta ({R['sam_down_beta']:.2f}) is *higher* than its "
            f"up-market beta ({R['sam_up_beta']:.2f}) — the **opposite** of defensive. |\n"
            "| Do they beat the S&P *in a recession*? | **No, not reliably.** In 2 of the 3 recessions, "
            "Molson Coors fell **more** than the S&P; over all recession months it still ended down "
            f"**{R['tap_rec_cum']:.0f}%**. Boston Beer's apparent edge is two lucky craft-beer rallies, "
            f"not a beer effect (*t* = {R['sam_rec_t']:.2f}, below the bar). |\n"
            "| Is 'recession-proof beer' real? | **It's folklore.** The only real thing here is *low "
            "beta* — which isn't the same as counter-cyclical, and which you can buy more cheaply "
            "elsewhere. |\n"
            "| Would you have been richer holding beer? | **Not even close.** \\$1 in the S&P became "
            f"**\\${R['spy_wealth']:.0f}**; in Molson Coors **\\${R['tap_wealth']:.0f}**, in Boston "
            f"Beer **\\${R['sam_wealth']:.0f}**. |\n\n"
            "> The barstool wisdom mixes up two different things: *low beta* (real, boring) and "
            "*counter-cyclical* (the exciting claim — and not there)."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Beer is recession-proof. When people lose their jobs they don't cancel the beer — they "
            "cancel the vacation and buy a six-pack instead. So a brewer is a defensive stock: it holds "
            "up when everything else is falling apart. Alcohol is the last thing people give up.\"*\n\n"
            "It's a genuinely *steelman-able* idea. \"Sin\" and staple consumption really is stickier "
            "than, say, cars or holidays — brewers sell a cheap, habitual, everyday product. The finance "
            "version of the folklore is specific and testable: a beer stock should have a **low downside "
            "beta** (it participates less in market falls) and should **out-return the S&P during "
            "recessions**. If that's true, it's a free defensive tilt you can hold forever."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it were true, a beer stock would be a rare thing: an equity that *cushions* your "
            "portfolio exactly when you need it — when the market is crashing and a recession is biting. "
            "That's worth real money: lower drawdowns, a smoother ride, a genuine diversifier you can "
            "hold instead of bonds. But notice the sleight of hand. \"Beer sales hold up in a "
            "recession\" is a claim about **volumes**. \"Beer *stocks* beat the market in a recession\" "
            "is a claim about **prices** — and a stock price already has the whole recession, and "
            "everyone's opinion about beer volumes, baked in. Those two things need not agree at all. "
            "We can check the second one directly."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three honest tests, each against the S&P 500 (SPY):\n\n"
            "1. **Do they fall less?** Split every month into \"market up\" and \"market down,\" and "
            "measure how much each beer stock moves *on the way down* versus *on the way up*. A truly "
            "defensive name drops less than it climbs.\n"
            "2. **Do they beat the index in a recession?** Take the actual NBER recession months (2001, "
            "2008, 2020) and compare each beer stock's return to the S&P's, head-to-head.\n"
            "3. **Could you even trade it?** The NBER only *declares* a recession many months after it "
            "starts — so \"rotate into beer when the recession hits\" is a rule you can't actually "
            "follow in real time.\n\n"
            "**What would make us say \"recession-proof\"?** A clear, consistent tendency to fall less "
            "*and* beat the index across recessions — not one lucky window. Anything less is a barstool "
            "story."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: do they fall less than the market?** For each stock we measure its *bear-market "
            "beta* (how much it moves in months the S&P is down) against its *bull-market beta*. "
            "Defensive means the down-bar is **shorter** than the up-bar."
        ),
        code(
            "if HAVE_PRICES:\n"
            "    spy_r = PX['SPY'].pct_change().dropna()\n"
            "    bb = {t: st.bull_bear_beta(PX[t].pct_change().dropna(), spy_r, split=0.0) for t in ['TAP','SAM']}\n"
            "    tap_d, tap_u = bb['TAP']['down_beta'], bb['TAP']['up_beta']\n"
            "    sam_d, sam_u = bb['SAM']['down_beta'], bb['SAM']['up_beta']\n"
            "else:\n"
            "    tap_d, tap_u = R['tap_down_beta'], R['tap_up_beta']\n"
            "    sam_d, sam_u = R['sam_down_beta'], R['sam_up_beta']\n"
            "x = np.arange(2); fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.bar(x-.2, [tap_d, sam_d], .4, color=RED, label='beta in DOWN markets')\n"
            "ax.bar(x+.2, [tap_u, sam_u], .4, color=GREEN, alpha=.7, label='beta in UP markets')\n"
            "ax.axhline(1.0, ls='--', c=GREY); ax.set_xticks(x); ax.set_xticklabels(['Molson Coors (TAP)','Boston Beer (SAM)'])\n"
            "ax.set_ylabel('beta vs S&P'); ax.set_title('Defensive means the RED bar is shorter than the GREEN'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'TAP: down-beta {tap_d:.2f} vs up-beta {tap_u:.2f}  -> {\"defensive\" if tap_d<tap_u else \"NOT defensive\"}')\n"
            "print(f'SAM: down-beta {sam_d:.2f} vs up-beta {sam_u:.2f}  -> {\"defensive\" if sam_d<sam_u else \"NOT defensive\"}')"
        ),
        md(
            f"Split decision. Molson Coors *is* the mild defensive the folklore imagines — its "
            f"down-beta (**{R['tap_down_beta']:.2f}**) sits below its up-beta (**{R['tap_up_beta']:.2f}**). "
            f"But Boston Beer does the **opposite**: it falls *harder* than it rises "
            f"(**{R['sam_down_beta']:.2f}** down vs **{R['sam_up_beta']:.2f}** up). Two beer stocks, two "
            "opposite answers — not the signature of a real \"beer is defensive\" law. And note both "
            "betas are *below 1*: these are simply low-beta stocks, which is a different, more boring "
            "thing than counter-cyclical."
        ),
        md(
            "**Now the direct test: the recessions themselves.** Here is how each stock did, "
            "compounded, inside each of the three NBER recessions — head-to-head with the S&P."
        ),
        code(
            "labels = R['rec_labels']\n"
            "if HAVE_PRICES:\n"
            "    tb = st.recession_breakdown(PX['TAP'].pct_change().dropna(), spy_r, data.NBER_RECESSIONS)\n"
            "    sb = st.recession_breakdown(PX['SAM'].pct_change().dropna(), spy_r, data.NBER_RECESSIONS)\n"
            "    tapv=[tb[l]['stock']*100 for l in labels]; samv=[sb[l]['stock']*100 for l in labels]; spyv=[tb[l]['bench']*100 for l in labels]\n"
            "else:\n"
            "    tapv, samv, spyv = R['tap_by_rec'], R['sam_by_rec'], R['spy_by_rec']\n"
            "x = np.arange(3); fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "ax.bar(x-.25, tapv, .25, color=AMBER, label='Molson Coors')\n"
            "ax.bar(x,      samv, .25, color='#8e44ad', label='Boston Beer')\n"
            "ax.bar(x+.25, spyv, .25, color=GREEN, label='S&P 500')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('compounded return in the recession (%)')\n"
            "ax.set_title('Recession-proof? Only Boston Beer, and only in 2 lucky windows'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for i,l in enumerate(labels): print(f'{l:14s}  TAP {tapv[i]:+6.1f}%  SAM {samv[i]:+6.1f}%  SPY {spyv[i]:+6.1f}%')"
        ),
        md(
            "Look at what actually happened. Molson Coors — the *defensive* one — fell **more** than the "
            "S&P in 2001 and in 2020; it only \"won\" the 2008 crash. Boston Beer's two big green bars "
            "(2001, 2020) are a small craft-beer growth stock ripping higher for reasons that have "
            "nothing to do with recession beer-drinking — and in **2008, the deepest recession**, it "
            "*fell* too. There is no consistent \"beer beats the market when times are hard\" here."
        ),
        md(
            "**So does the aggregate \"recession outperformance\" survive?** Pool every recession month "
            "and compare each stock's average monthly return to the S&P's."
        ),
        code(
            "if HAVE_PRICES:\n"
            "    tre = st.recession_excess_t(PX['TAP'].pct_change().dropna(), spy_r, data.recession_mask)\n"
            "    sre = st.recession_excess_t(PX['SAM'].pct_change().dropna(), spy_r, data.recession_mask)\n"
            "    tap_ex,tap_t = tre['mean_excess']*100, tre['t']\n"
            "    sam_ex,sam_t = sre['mean_excess']*100, sre['t']\n"
            "else:\n"
            "    tap_ex,tap_t = R['tap_rec_excess'], R['tap_rec_t']\n"
            "    sam_ex,sam_t = R['sam_rec_excess'], R['sam_rec_t']\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "cols=[GREEN if v>0 else RED for v in [tap_ex, sam_ex]]\n"
            "ax.bar(['Molson Coors','Boston Beer'], [tap_ex, sam_ex], .5, color=cols)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('avg monthly return ABOVE the S&P, in recessions (%)')\n"
            "for i,(v,t) in enumerate(zip([tap_ex,sam_ex],[tap_t,sam_t])): ax.annotate(f't={t:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_title('Recession-window edge vs the S&P (with its t-stat)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'TAP recession excess {tap_ex:+.2f}%/mo  t={tap_t:+.2f}  (basically zero)')\n"
            "print(f'SAM recession excess {sam_ex:+.2f}%/mo  t={sam_t:+.2f}  (below the |t|>=2 bar)')"
        ),
        md(
            f"Molson Coors' recession edge is **essentially zero** (*t* = {R['tap_rec_t']:.2f}). Boston "
            f"Beer's looks bigger (**+{R['sam_rec_excess']:.1f}%/mo**) but *t* = **{R['sam_rec_t']:.2f}** "
            "— under the bar we set for \"real,\" and, as the chart above showed, it's two idiosyncratic "
            "rallies rather than a repeatable beer effect. Neither clears the line."
        ),
        md(
            "**And the part the barstool never mentions: what did \"defense\" cost you?** Even if a stock "
            "wobbles less, it's a bad trade if it barely grows. Here's \\$1 left in each for the whole "
            "31 years."
        ),
        code(
            "if HAVE_PRICES:\n"
            "    ends = [st.terminal_wealth(PX[t]) for t in ['SPY','TAP','SAM']]\n"
            "else:\n"
            "    ends = [R['spy_wealth'], R['tap_wealth'], R['sam_wealth']]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(['S&P 500','Molson Coors','Boston Beer'], ends, .55, color=[GREEN, AMBER, '#8e44ad'])\n"
            "for i,v in enumerate(ends): ax.annotate(f'${v:.1f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('$1 invested in 1995 -> ...'); ax.set_title('The price of \"defense\": beer badly lagged the index')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"$1 -> S&P ${ends[0]:.2f}  |  Molson Coors ${ends[1]:.2f}  |  Boston Beer ${ends[2]:.2f}\")"
        ),
        md(
            f"The index turned \\$1 into **\\${R['spy_wealth']:.0f}**. The beer stocks? "
            f"**\\${R['tap_wealth']:.0f}** and **\\${R['sam_wealth']:.0f}** — roughly a *third* of the "
            "index, with *deeper* worst-case drawdowns. Whatever mild cushioning Molson Coors offered on "
            "the way down, you paid for it many times over in growth you gave up."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** No significant excess return anywhere: the recession-window edge "
            f"doesn't clear the bar (best is Boston Beer at *t* = {R['sam_rec_t']:.2f}), the CAPM alphas "
            "are insignificant, and the two names disagree on whether beer is even defensive.\n"
            "- **Tradability — Mirage.** You can't know you're *in* a recession until the NBER says so "
            "months later, and holding beer \"for defense\" cost you two-thirds of the index's return "
            "over 31 years.\n"
            "- **Counter-cyclical? — Busted.** The only real effect is *low beta* (Molson Coors), which "
            "is not the same as counter-cyclical — and Boston Beer isn't even low-beta on the downside."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "Suppose you *believed* it in 1995 and put your \\$10,000 into beer \"for safety.\" Where do "
            "you land in 2026 versus someone who just bought the index and forgot about it?"
        ),
        code(
            "start=10_000.0\n"
            "if HAVE_PRICES:\n"
            "    mult=[st.terminal_wealth(PX[t]) for t in ['SPY','TAP','SAM']]\n"
            "else:\n"
            "    mult=[R['spy_wealth'],R['tap_wealth'],R['sam_wealth']]\n"
            "ends=[start*m for m in mult]\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.3))\n"
            "ax.bar(['S&P index fund','\"defensive\" beer\\n(Molson Coors)','\"defensive\" beer\\n(Boston Beer)'], ends,\n"
            "       color=[GREEN, AMBER, '#8e44ad'], width=.55)\n"
            "for i,v in enumerate(ends): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('value of $10,000, 1995 -> 2026'); ax.set_title('The \"recession hedge\" that cost two-thirds of your money')\n"
            "plt.tight_layout(); plt.show()\n"
            "for l,v in zip(['S&P','Molson Coors','Boston Beer'], ends): print(f'{l:14s} ${v:,.0f}')"
        ),
        md(
            "The index investor ends with roughly **three times** what the beer holder does. And the one "
            "moment the hedge was supposed to earn its keep — a recession — is exactly when you *couldn't "
            "act on it*: the NBER declared the 2008 recession a full year after it began and back-dated "
            "the 2020 trough more than a year later. \"Rotate into beer when the recession hits\" is a "
            "rule you can only follow with a time machine. The recession-proof-beer trade is a story you "
            "tell at the bar, not a position you can hold."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Widen the beer aisle.** Add Anheuser-Busch InBev (`BUD`), Constellation (`STZ`), "
            "Heineken (`HEINY`), Diageo (`DEO`) — does *any* brewer show a consistent recession edge, or "
            "is it always one lucky name?\n"
            "- **Volumes vs prices.** Pull actual beer shipment data (the folklore's real claim) and see "
            "whether even *volumes* rise in recessions — spoiler: it's mostly a shift to cheaper brands, "
            "not more litres.\n"
            "- **The staples cousin.** The same test on tobacco (`MO`, `PM`) or discount retail is the "
            "natural next study — is *anything* truly counter-cyclical, or just low-beta? (see "
            "[docs/references.md](../docs/references.md)).\n\n"
            "*Think a specific brewer beat the S&P net of everything through a recession you can name? "
            "Pull its tape, mark the NBER window, and show it — then check it wasn't one idiosyncratic "
            "rally doing all the work.*"
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
            "# Beer defensiveness — a quantitative teardown 🔬\n"
            "### Bull vs bear beta · Newey-West CAPM alpha vs SPY · the recession-window paired *t* · "
            "the NBER look-ahead that kills the trade · a planted-asymmetric-beta positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We test the "
            "strongest tradable form of \"beer is recession-proof\" on `TAP` (Molson Coors) and `SAM` "
            "(Boston Beer) vs `SPY`: (H₁) **downside defensiveness** — bear-beta < bull-beta < 1; (H₂) "
            "**recession outperformance** — positive recession-window excess with *t* > 2; (H₃) it is "
            "**investable** net of the NBER announcement lag and opportunity cost. We find **H₁ splits** "
            "(TAP yes, SAM no — and it's just low beta), **H₂ rejected** (no leg clears *t* = 2), **H₃ "
            "rejected** (look-ahead + a 3× wealth gap to SPY).\n\n"
            "> ⚠️ **Not investment advice — data provenance.** `TAP`, `SAM`, `SPY` are month-end Adj "
            "Close via yfinance (as-of 2026-06-30; split+dividend adjusted). NBER recession windows are "
            "**hardcoded, cited** official dates (a labelled set of facts, not a feed). Offline core + "
            "synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `NONE` | No leg clears *t* ≥ 2. Recession-window excess: TAP "
            f"*t* = **{R['tap_rec_t']:+.2f}**, SAM *t* = **{R['sam_rec_t']:+.2f}** (n={R['rec_n']} mo). "
            f"CAPM α insignificant (TAP NW *t*={R['tap_t']:+.2f}, SAM *t*={R['sam_t']:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | NBER dates a recession **~{R['nber_lag']} mo ex-post** "
            f"(look-ahead); \\$1 → **\\${R['spy_wealth']:.0f}** SPY vs **\\${R['tap_wealth']:.0f}** TAP / "
            f"**\\${R['sam_wealth']:.0f}** SAM over {R['hold_yrs']:.0f}y. |\n"
            f"| **Counter-cyclical?** | `BUSTED` | Only real effect = *low beta* (TAP β={R['tap_beta']:.2f}); "
            f"but TAP fell **more** than SPY in 2/3 recessions, and SAM's down-beta "
            f"({R['sam_down_beta']:.2f}) **exceeds** its up-beta ({R['sam_up_beta']:.2f}). |\n\n"
            "> 💡 In plain words: one beer stock is mildly low-beta, the other is anti-defensive on the "
            "downside, neither beats the index in recessions at any real significance, and the whole "
            "\"hedge\" would have cost you two-thirds of your money. \"Counter-cyclical\" gets confused "
            "with \"low beta\" — the former isn't here, the latter is cheaper elsewhere."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a beer stock's monthly return be $r^P_t$ and the market's $r^M_t$ (SPY). The folklore "
            "is a joint hypothesis:\n\n"
            "- **H₁ (downside defensiveness).** The bear-market beta is below the bull-market beta and "
            "below 1: $\\;\\beta^- < \\beta^+ \\;\\wedge\\; \\beta^- < 1$, where $\\beta^-$ is the OLS "
            "slope on $\\{t: r^M_t < 0\\}$ and $\\beta^+$ on $\\{t: r^M_t \\ge 0\\}$.\n"
            "- **H₂ (recession outperformance).** Over NBER recession months $\\mathcal{R}$, the mean "
            "excess $\\;\\bar{x} = \\text{mean}_{t\\in\\mathcal{R}}(r^P_t - r^M_t)\\;$ is positive with "
            "$t > 2$.\n"
            "- **H₃ (investable).** After the ~12-month NBER announcement lag (you cannot condition on "
            "$\\mathcal{R}$ in real time) and the full-sample opportunity cost, a beer-tilt beats simply "
            "holding SPY.\n\n"
            "The steelman is that staple/'sin' consumption is income-inelastic, so a brewer's cash flows "
            "*should* be sticky through a downturn. The test is whether that shows up in the **stock**, "
            "**risk-adjusted**, **out-of-sample-tradable** — or only in the barstool."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If H₁–H₃ held, a beer stock would be a genuine equity hedge: it would cushion drawdowns "
            "(H₁), pay off precisely in recessions (H₂), and do so in a way you could actually implement "
            "(H₃). Each leg is separately falsifiable. H₁ is a **conditional-beta** statement — a risk "
            "characteristic, contemporaneous, no lag needed. H₂ is the **event-study** the folklore "
            "actually asserts, but on a tiny sample (three recessions, ~31 months) that is fragile to a "
            "single idiosyncratic name-quarter. H₃ is the **realisability** check: NBER recession dating "
            "is published with a long lag, so any recession-conditional strategy is a look-ahead unless "
            "you can forecast the turn — and a permanent beer tilt must be judged on its full-sample "
            "cost, not its best three windows. The claim needs all three; failing any downgrades it to a "
            "story about low beta."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Data.** `TAP`, `SAM`, `SPY` month-end Adj Close (yfinance, cached; splits + dividends "
            "folded in — *total-return-ish*, labelled as such). Sample 1995-01 → 2026-06 "
            f"(SPY/TAP n={R['n_spy']} monthly returns; SAM from its 1995-11 listing, n={R['n_sam']}).\n"
            "- **Survivorship.** Named tickers, not a screen — but stated honestly: both firms *survived* "
            "the whole window, and SAM in particular is a craft-beer **winner**; the selection points "
            "*for* the defensiveness claim, so we treat any positive result with extra suspicion.\n"
            "- **H₁ test.** Bull/bear conditional betas (split at $r^M=0$); defensive ⟺ "
            "$\\beta^- < \\beta^+ < 1$. No lag (a contemporaneous risk metric).\n"
            "- **H₂ test.** Paired *t* of the recession-window monthly excess $r^P-r^M$ over the union of "
            "NBER recession months; the bar for `REAL` is $t \\ge 2$ in the beer's favour. A "
            "per-recession breakdown checks the aggregate isn't one window.\n"
            "- **CAPM α.** Newey-West (6-lag) HAC *t* of the full-sample alpha vs SPY — is there excess "
            "*after* market beta?\n"
            "- **H₃ / cost (beat 6).** The NBER announcement lag (a look-ahead the folklore ignores) plus "
            "the full-sample terminal-wealth gap to SPY.\n"
            "- **Positive control.** A deterministic stock with a *planted* asymmetric beta "
            "($\\beta^-{=}0.5 < \\beta^+{=}1.0$); the engine must recover $\\beta^- < \\beta^+$ — proof a "
            "null on the real tape is a real null, not a broken detector.\n"
            "- **What would make us say \"defensive/counter-cyclical\":** $\\beta^-<\\beta^+<1$ **on both "
            "names** *and* a recession-window *t* > 2. We find neither."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · H₁ — bull vs bear beta (is it defensive at all?)\n\n"
            "Conditional OLS slope of each beer name on SPY, split by the sign of the market return. "
            "Defensive requires the down-beta *below* the up-beta and below 1."
        ),
        code(
            "spy_r = PX['SPY'].pct_change().dropna() if HAVE_PRICES else None\n"
            "rows = {}\n"
            "for t in ['TAP','SAM']:\n"
            "    if HAVE_PRICES:\n"
            "        bb = st.bull_bear_beta(PX[t].pct_change().dropna(), spy_r, split=0.0)\n"
            "        rows[t] = dict(d=bb['down_beta'], u=bb['up_beta'], f=bb['full_beta'], a=bb['asymmetry'], defn=bb['defensive'])\n"
            "    else:\n"
            "        pre='tap' if t=='TAP' else 'sam'\n"
            "        rows[t] = dict(d=R[f'{pre}_down_beta'], u=R[f'{pre}_up_beta'], f=R[f'{pre}_beta'], a=R[f'{pre}_asym'], defn=R[f'{pre}_def'])\n"
            "x=np.arange(2); fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, [rows['TAP']['d'],rows['SAM']['d']], .4, color=RED, label=r'$\\beta^-$ (down markets)')\n"
            "ax.bar(x+.2, [rows['TAP']['u'],rows['SAM']['u']], .4, color=GREEN, alpha=.7, label=r'$\\beta^+$ (up markets)')\n"
            "ax.axhline(1.0, ls='--', c=GREY); ax.set_xticks(x); ax.set_xticklabels(['TAP','SAM'])\n"
            "ax.set_ylabel('conditional beta vs SPY'); ax.set_title(r'H1: defensive $\\Leftrightarrow \\beta^- < \\beta^+ < 1$ — only TAP, weakly'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for t in ['TAP','SAM']: r=rows[t]; print(f\"{t}: down-beta {r['d']:.2f}  up-beta {r['u']:.2f}  full {r['f']:.2f}  asymmetry {r['a']:+.2f}  defensive={r['defn']}\")"
        ),
        md(
            f"> 💡 In plain words: Molson Coors is a weak textbook defensive — $\\beta^-$="
            f"**{R['tap_down_beta']:.2f}** < $\\beta^+$=**{R['tap_up_beta']:.2f}**, both under 1 (a "
            f"{R['tap_asym']:+.2f} asymmetry). Boston Beer is the **opposite**: $\\beta^-$="
            f"**{R['sam_down_beta']:.2f}** *exceeds* $\\beta^+$=**{R['sam_up_beta']:.2f}** "
            f"({R['sam_asym']:+.2f}) — it participates *more* on the downside. H₁ holds for one name and "
            "fails for the other, and even for TAP it's a small gap on the same low-beta stock. This is "
            "\"low beta,\" not a robust \"beer is defensive\" law."
        ),
        md(
            "### 4b · H₂ — recession-window returns (the direct counter-cyclical test)\n\n"
            "Paired excess $r^P-r^M$ over the union of NBER recession months, and — crucially — the "
            "per-recession breakdown that shows whether any aggregate edge is broad or one window."
        ),
        code(
            "labels=R['rec_labels']\n"
            "if HAVE_PRICES:\n"
            "    tre=st.recession_excess_t(PX['TAP'].pct_change().dropna(), spy_r, data.recession_mask)\n"
            "    sre=st.recession_excess_t(PX['SAM'].pct_change().dropna(), spy_r, data.recession_mask)\n"
            "    tb=st.recession_breakdown(PX['TAP'].pct_change().dropna(), spy_r, data.NBER_RECESSIONS)\n"
            "    sb=st.recession_breakdown(PX['SAM'].pct_change().dropna(), spy_r, data.NBER_RECESSIONS)\n"
            "    tapv=[tb[l]['stock']*100 for l in labels]; samv=[sb[l]['stock']*100 for l in labels]; spyv=[tb[l]['bench']*100 for l in labels]\n"
            "    tt,st_t=tre['t'],sre['t']; tex,sex=tre['mean_excess']*100,sre['mean_excess']*100\n"
            "else:\n"
            "    tapv,samv,spyv=R['tap_by_rec'],R['sam_by_rec'],R['spy_by_rec']\n"
            "    tt,st_t,tex,sex=R['tap_rec_t'],R['sam_rec_t'],R['tap_rec_excess'],R['sam_rec_excess']\n"
            "x=np.arange(3); fig, ax = plt.subplots(figsize=(9.6, 4.6))\n"
            "ax.bar(x-.25, tapv, .25, color=AMBER, label=f'TAP  (agg excess {tex:+.1f}%/mo, t={tt:+.2f})')\n"
            "ax.bar(x,      samv, .25, color='#8e44ad', label=f'SAM  (agg excess {sex:+.1f}%/mo, t={st_t:+.2f})')\n"
            "ax.bar(x+.25, spyv, .25, color=GREEN, label='SPY')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('compounded return in recession (%)'); ax.set_title('H2: no broad recession edge — SAM is 2 idiosyncratic windows'); ax.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "for i,l in enumerate(labels): print(f'{l:14s}  TAP {tapv[i]:+6.1f}%  SAM {samv[i]:+6.1f}%  SPY {spyv[i]:+6.1f}%')\n"
            "print(f'aggregate: TAP excess t={tt:+.2f}   SAM excess t={st_t:+.2f}   (bar is |t|>=2)')"
        ),
        md(
            f"> 💡 In plain words: TAP's recession edge is a rounding error (*t* = {R['tap_rec_t']:+.2f}); "
            f"it fell **more** than SPY in 2001 (−13.8% vs −7.1%) and 2020 (−25.4% vs −9.2%), winning only "
            f"the GFC. SAM's aggregate *t* = {R['sam_rec_t']:+.2f} looks tantalising but the breakdown "
            "indicts it: +40.8% (2001) and +30.9% (2020) are a small growth stock rallying, while in "
            "**2008 — the deepest, most beer-relevant recession — SAM *fell* 10.8%**. An 'edge' that "
            "reverses in the one recession that most tests the thesis is not a beer effect. H₂ rejected."
        ),
        md(
            "### 4c · CAPM α — is there excess after market beta?\n\n"
            "Newey-West (6-lag) HAC regression of each name's monthly return on SPY. `REAL` needs "
            "$t_\\alpha \\ge 2$."
        ),
        code(
            "rows={}\n"
            "for t in ['TAP','SAM']:\n"
            "    if HAVE_PRICES:\n"
            "        nw=st.newey_west_alpha_t(PX[t].pct_change().dropna(), spy_r, 6)\n"
            "        s=st.summarize(PX[t])\n"
            "        rows[t]=dict(alpha=nw['alpha_ann']*100, beta=nw['beta'], t=nw['t_alpha'], cagr=s['cagr']*100, vol=s['vol']*100, sharpe=s['sharpe'], mdd=s['mdd']*100)\n"
            "    else:\n"
            "        pre='tap' if t=='TAP' else 'sam'\n"
            "        rows[t]=dict(alpha=R[f'{pre}_alpha'], beta=R[f'{pre}_beta'], t=R[f'{pre}_t'], cagr=R[f'{pre}_cagr'], vol=R[f'{pre}_vol'], sharpe=R[f'{pre}_sharpe'], mdd=R[f'{pre}_mdd'])\n"
            "labels=list(rows); alphas=[rows[t]['alpha'] for t in labels]; ts=[rows[t]['t'] for t in labels]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3)); cols=[GREEN if a>0 else RED for a in alphas]\n"
            "ax.bar(labels, alphas, .5, color=cols)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised CAPM alpha vs SPY (%)')\n"
            "for i,t in enumerate(labels): ax.annotate(f'NW t={ts[i]:+.2f}',(i,alphas[i]),ha='center',va='bottom')\n"
            "ax.set_title('CAPM alpha: positive point estimates, but |t|<1 — insignificant')\n"
            "plt.tight_layout(); plt.show()\n"
            "for t in labels: r=rows[t]; print(f\"{t}: CAGR {r['cagr']:5.2f}%  vol {r['vol']:4.1f}%  Sharpe {r['sharpe']:.2f}  maxDD {r['mdd']:6.1f}%  |  alpha {r['alpha']:+.2f}%/yr  beta {r['beta']:.2f}  NW t {r['t']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: both point-estimate alphas are positive (TAP +{R['tap_alpha']:.1f}%/yr, "
            f"SAM +{R['sam_alpha']:.1f}%/yr) but neither is distinguishable from zero (NW "
            f"*t* = {R['tap_t']:+.2f}, {R['sam_t']:+.2f}) — and both names posted a *lower* CAGR "
            f"({R['tap_cagr']:.1f}%, {R['sam_cagr']:.1f}%) than SPY ({R['spy_cagr']:.1f}%) with a "
            "*deeper* drawdown. Low beta bought a smoother ride in one name, not alpha, and not "
            "outperformance. The `NONE` stamp is earned."
        ),
        md(
            "### 4d · Positive control — the engine detects a planted defensive stock\n\n"
            "A deterministic stock with a *planted* asymmetric beta ($\\beta^-{=}0.5$, $\\beta^+{=}1.0$, "
            "seed 728). The bull/bear engine must recover $\\beta^- < \\beta^+$ — proving the near-"
            "symmetry on the real tape is a true null, not a blind detector."
        ),
        code(
            "mkt, stock = data.synthetic_defensive(beta_down=0.5, beta_up=1.0)\n"
            "cr = st.control_recovers(stock, mkt, planted_down=0.5, planted_up=1.0)\n"
            "down=mkt[mkt<0]; up=mkt[mkt>=0]\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.3))\n"
            "ax.scatter(mkt, stock, s=10, c=np.where(mkt<0, RED, GREEN), alpha=.6)\n"
            "xs=np.linspace(mkt.min(),0,20); ax.plot(xs, cr['down_beta']*xs, c=RED, lw=2, label=f\"recovered $\\\\beta^-$={cr['down_beta']:.2f}\")\n"
            "xs2=np.linspace(0,mkt.max(),20); ax.plot(xs2, cr['up_beta']*xs2, c=GREEN, lw=2, label=f\"recovered $\\\\beta^+$={cr['up_beta']:.2f}\")\n"
            "ax.axhline(0,c=GREY,lw=.7); ax.axvline(0,c=GREY,lw=.7)\n"
            "ax.set_xlabel('market monthly return'); ax.set_ylabel('stock monthly return')\n"
            "ax.set_title('Planted defensive stock: engine recovers beta- < beta+ (machinery proof)'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"planted down/up 0.50/1.00 -> recovered down {cr['down_beta']:.2f}  up {cr['up_beta']:.2f}  recovered_defensive={cr['recovered_defensive']}\")"
        ),
        md(
            "> 💡 In plain words: when a stock really *is* defensive, the engine sees it cleanly "
            f"(recovered $\\beta^-$≈**{R['syn_down']:.2f}** < $\\beta^+$≈**{R['syn_up']:.2f}**). So the "
            "messy, name-dependent, insignificant result on the real beer tape is a genuine null — not a "
            "detector that can't tell defensive from cyclical. A synthetic control is a machinery proof, "
            "never market evidence."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — no leg clears *t* ≥ 2 in the beer's favour: recession-window excess "
            f"TAP *t*={R['tap_rec_t']:+.2f}, SAM *t*={R['sam_rec_t']:+.2f} (n={R['rec_n']} months, and "
            "SAM's is two idiosyncratic windows); CAPM α insignificant (NW *t*="
            f"{R['tap_t']:+.2f}, {R['sam_t']:+.2f}). H₁ holds only for TAP and only weakly.\n"
            f"- **Tradability `MIRAGE`** — the NBER dates a recession ~{R['nber_lag']} months after it "
            f"starts (a look-ahead), and holding beer \"for defense\" turned \\$1 into "
            f"\\${R['tap_wealth']:.0f}/\\${R['sam_wealth']:.0f} vs SPY's \\${R['spy_wealth']:.0f} over "
            f"{R['hold_yrs']:.0f} years.\n"
            f"- **Counter-cyclical? `BUSTED`** — the only real effect is *low beta* (TAP β="
            f"{R['tap_beta']:.2f}), which is not counter-cyclicality: TAP fell **more** than SPY in 2 of "
            f"3 recessions, and SAM's down-beta ({R['sam_down_beta']:.2f}) exceeds its up-beta "
            f"({R['sam_up_beta']:.2f})."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the capacity & look-ahead reality\n\n"
            "Two walls. **The look-ahead:** any recession-conditional beer tilt needs to know it's in a "
            "recession — but the NBER declared the 2008 recession in Dec-2008 (a year late) and dated the "
            "2020 trough in Jul-2021 (15 months late). **The opportunity cost:** a *permanent* beer tilt "
            "is judged on the full sample, where it badly trails SPY."
        ),
        code(
            "start=10_000.0\n"
            "if HAVE_PRICES:\n"
            "    mult=[st.terminal_wealth(PX[t]) for t in ['SPY','TAP','SAM']]\n"
            "else:\n"
            "    mult=[R['spy_wealth'],R['tap_wealth'],R['sam_wealth']]\n"
            "ends=[start*m for m in mult]\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.bar(['SPY (index)','TAP (Molson Coors)','SAM (Boston Beer)'], ends, .55, color=[GREEN, AMBER, '#8e44ad'])\n"
            "for i,v in enumerate(ends): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(start, ls=':', c=GREY, label='$10,000 start'); ax.set_ylabel('value of $10,000, 1995->2026'); ax.legend()\n"
            "ax.set_title('The permanent \"beer hedge\" ends at ~1/3 of the index')\n"
            "plt.tight_layout(); plt.show()\n"
            "for l,v in zip(['SPY','TAP','SAM'], ends): print(f'{l}: ${v:,.0f}')\n"
            "print(f\"NBER announcement lag ~{R['nber_lag']} months -> recession-conditional entry is a look-ahead\")"
        ),
        md(
            "> 💡 In plain words: there is no version of this that pays. The recession-timing version is "
            "un-implementable (you learn the recession's dates only after it's over); the buy-and-hold "
            "version compounds at a *discount* to the index it was supposed to protect you from. A "
            "defensive tilt is only worth it if the drawdown relief outweighs the return you surrender — "
            "here it emphatically does not, and for SAM the drawdown was actually *worse* "
            f"({R['sam_mdd']:.0f}% vs SPY {R['spy_mdd']:.0f}%)."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Broaden the universe.** Add `BUD`, `STZ`, `HEINY`, `DEO`, `MO`, `PM` and test whether a "
            "*basket* of brewers/sin-stocks shows a defensive tilt the single names don't — pooling "
            "across names is the honest way to beat the tiny-recession-sample problem.\n"
            "- **Better recession conditioning.** Replace the ex-post NBER flag with a *real-time* "
            "recession probability (Sahm rule, yield-curve, claims) and re-run H₃ with an honest, "
            "no-look-ahead entry — the only way the recession-window question becomes tradable.\n"
            "- **Volumes, the folklore's real claim.** Merge TTB/industry beer-shipment data: even if "
            "*volumes* are sticky, the mix shifts to cheaper brands (margin down), which is why the "
            "*stock* need not be defensive at all ([docs/references.md](../docs/references.md)).\n\n"
            "*The reproducible core is offline and deterministic; prices are real (yfinance, "
            "total-return-ish), the NBER windows are **cited official dates**. Methods: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
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
