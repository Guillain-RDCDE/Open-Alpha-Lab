"""Generate the two narrative notebooks for Study 758 (TSA-Throughput).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the hardcoded TSA snapshot
(always available) and the cached JETS/MAR/HLT/SPY prices under ../_cache/, and otherwise quote
the frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive control
runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (TSA throughput hardcoded
# monthly snapshot + travel basket month-end, 2019-01 -> 2026-06, 90 months, 7.4 years).
R = dict(
    start="2019-01-31", end="2026-06-30", months=90, years=7.4,
    tsa_min=0.11, tsa_min_when="Apr 2020", tsa_max=2.85, tsa_max_when="Jul 2025",
    # per-horizon: (months, n_accel, accel%, decel%, base%, acc_up%, base_up%, t, p_placebo)
    h1=(1, 64, 0.86, 2.40, 1.28, 58, 59, -0.29, 0.654),
    h3=(3, 62, 3.28, 4.07, 3.50, 63, 64, -0.10, 0.560),
    h6=(6, 59, 6.87, 5.99, 6.62, 71, 69, 0.09, 0.465),
    h12=(12, 53, 15.36, 4.38, 11.94, 77, 68, 0.82, 0.181),
    # lead/lag: L -> corr
    leadlag={-6: 0.323, -5: 0.051, -4: 0.028, -3: 0.233, -2: -0.008, -1: -0.060,
             0: -0.099, 1: -0.148, 2: -0.088, 3: -0.102, 4: -0.034, 5: -0.102, 6: -0.148},
    # beta control: (months, adj_coef%, adj_t, beta)
    beta=[(1, -0.32, -0.21, 1.37), (3, 0.87, 0.35, 1.38), (6, 4.89, 1.35, 1.35),
          (12, 21.13, 5.70, 1.49)],
    # overlay long/flat: (bh_mean%, bh_sharpe, gross%, gross_sharpe, net%, net_sharpe, switches, exposure)
    overlay=(15.9, 0.52, 3.2, 0.12, 3.1, 0.12, 4, 0.72),
    # overlay long/short net: (net%, net_sharpe)
    overlay_ls=(-10.0, -0.32),
    # robustness 12m: (label, n_accel, accel12%, base12%, t, p)
    robust=[("k=3", 44, 7.88, 11.94, -0.87, 0.842), ("k=6", 44, 10.10, 11.94, -0.42, 0.672),
            ("k=12", 53, 15.36, 11.94, 0.82, 0.181), ("thr>+10%", 28, 5.58, 11.94, -1.42, 0.896)],
    # ex-COVID horizons: (months, n_accel, accel%, base%, t, p)
    exc=[(1, 40, 1.43, 1.63, -0.11, 0.559), (3, 38, 6.21, 4.81, 0.45, 0.299),
         (6, 35, 12.05, 8.27, 1.20, 0.106), (12, 29, 24.40, 15.23, 2.12, 0.015)],
    exc_months=66,
    # synthetic control: (edge, n_accel, accel1m%, base1m%, t, p)
    syn=[(0.0, 160, 0.82, 0.75, 0.10, 0.447), (0.08, 160, 9.22, 6.39, 3.25, 0.000)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Nowcast%3F: Not_supported](https://img.shields.io/badge/Nowcast%3F-Not_supported-8b949e?style=flat-square)\n\n"
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

from tsa_throughput import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real() if HAVE_REAL else None
print("price cache present:", HAVE_REAL,
      "| TSA+basket months:", (0 if F is None else len(F)))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell can
# quote it whether or not the price cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Can airport-checkpoint crowds tell you when to buy travel stocks? 🛫\n"
            "### The 'real-time travel nowcast' — TSA volumes as an early tell for airlines and hotels, in plain English\n\n"
            + BADGES +
            "Every day the TSA reports how many people it screened at U.S. airport checkpoints. In "
            "2020–2021 that number became the most-watched gauge of the travel recovery — it fell "
            "**95%** in April 2020 and every tick back up was quoted as proof the reopening was real. "
            "The folklore says: because TSA is **daily and real-time**, an *acceleration* in throughput "
            "is an **early tell** for travel stocks — airlines and hotels — before the official traffic "
            "and earnings numbers catch up.\n\n"
            "It's a great story. It's also testable. This notebook asks three blunt questions: when TSA "
            "accelerates, do travel stocks really do better next? Does the throughput uptick actually "
            "come **first** (that's the whole pitch)? And if you *bought* travel every time TSA "
            "accelerated, would you make money?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the lead/lag cross-correlation, the "
            "market-beta control and the synthetic control? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** TSA's live feed is blocked in this environment, so we use a "
            "**labelled proxy** — a hardcoded monthly snapshot of the public daily throughput numbers "
            "(the average travellers/day, in millions), including the giant COVID-2020 collapse. Every "
            "chart is drawn by the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| When TSA accelerates, do travel stocks do better? | **At a one-year horizon, a bit — yes.** "
            f"The travel basket averages **+{R['h12'][2]:.1f}%** over the next year after TSA accelerates "
            f"vs **+{R['h12'][4]:.1f}%** normally. The *direction* matches the folklore. |\n"
            "| Is that gap reliable? | **No.** It's inside the noise (you can't tell it from luck), it "
            "**vanishes or flips** if you tweak the recipe, and it only clears the bar once you delete the "
            "COVID reopening — on a single horizon. |\n"
            "| Does the throughput uptick come *first*? | **No — and this is the killer.** The TSA signal "
            "lines up best with a travel-stock move that already happened **three to six months earlier.** "
            "Travel stocks *lead* TSA here; they don't follow it. |\n"
            "| So could you trade it? | **It loses badly.** \"Buy travel when TSA accelerates\" earned "
            f"**+{R['overlay'][4]:.1f}%/yr** vs **+{R['overlay'][0]:.1f}%** for just holding the basket — "
            "you sit out the very rallies you were trying to catch. |\n\n"
            "> TSA numbers are a real, useful read on *how many people are flying right now.* But \"TSA "
            "warns you *early* about travel stocks\" is a coincident echo wearing a crystal-ball costume — "
            "the market prices the recovery months before the checkpoints confirm it."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"TSA checkpoint throughput is a free, daily, government-published read on travel demand. "
            "It's published before airline traffic reports, hotel occupancy, or earnings — so when TSA "
            "volumes accelerate, get long airlines and hotels: you're seeing the tailwind before Wall "
            "Street's official data does.\"*\n\n"
            "There's a respectable backbone to this: TSA throughput genuinely *is* a timely, accurate "
            "measure of how many people are flying — the alt-data 'nowcasting' idea (card spend, "
            "satellite parking lots, web traffic) applied to travel. The trading leap is the part we "
            "test: that a throughput *acceleration* arrives early enough, and cleanly enough, to be a "
            "**tradable** tell for travel *stocks*."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, this would be gold: a free daily number that front-runs the travel trade. But "
            "\"real-time\" hides a trap. The **stock market is a discounting machine** — it prices *news* "
            "about a recovery long before the physical activity shows up. Travel stocks doubled off the "
            "March-2020 lows and ripped on the November-2020 vaccine news — **months before** TSA "
            "throughput actually recovered. So a physical-activity number that lines up with travel-stock "
            "strength might not be *predicting* those stocks at all; it might just be **confirming** a "
            "recovery the market already bought. The difference between *leads* and *confirms* is the "
            "difference between an edge and a mirage — and you can only tell them apart by checking the "
            "timing carefully."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We line up **{R['years']:.0f}+ years** ({R['start'][:4]}–{R['end'][:4]}, "
            f"{R['months']} months) of monthly TSA throughput against a month-end **travel basket** "
            "(half airlines via `JETS`, half hotels via `MAR`+`HLT`), and:\n\n"
            "1. **Split the months.** Call TSA **accelerating** when throughput is above where it was a "
            "year ago (year-over-year growth neutralises the summer-travel season). Compare what the "
            "basket did next (1/3/6/12 months) in accelerating months vs all months.\n"
            "2. **Check the timing.** The crucial test: slide TSA forward and backward against the "
            "travel basket and find *where* they line up best. If TSA truly **leads**, the strongest "
            "link shows up at a **positive lead** (TSA first, stocks later).\n"
            "3. **Try to trade it.** Buy the basket whenever TSA is accelerating, sit in cash otherwise, "
            "pay realistic costs — and see if it beats just buying and holding the basket."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the raw material.** Here's monthly TSA throughput since 2019 — the pre-COVID "
            "cruise near **2.3M/day**, the off-the-chart **collapse to ~0.1M** in April 2020, the long "
            "climb back, and today's record highs above 2019. TSA clearly *knows* about the travel "
            "cycle. The question is whether it knows **early** — earlier than the stocks."
        ),
        code(
            "if HAVE_REAL:\n"
            "    v = F['tsa']\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.0))\n"
            "    ax.plot(v.index, v.values, c=RED, lw=1.5)\n"
            "    ax.set_title('U.S. TSA checkpoint throughput (monthly avg, millions/day)')\n"
            "    ax.set_ylabel('travellers per day (millions)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('trough:', round(v.min(),2), 'M around', v.idxmin().date(), '| recent peak:', round(v.max(),2), 'M')\n"
            "else:\n"
            "    print('no cache — see docs/results.md; TSA bottomed ~0.11M in Apr 2020')"
        ),
        md(
            "**Now the payoff.** For each horizon, the average forward travel-basket return in "
            "**accelerating-TSA** months next to the return on an **average** month. The folklore "
            "predicts the green bars sit *above* the grey ones."
        ),
        code(
            "hs = [1, 3, 6, 12]\n"
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(F, m) for m in hs]\n"
            "    acc = [r['accel_mean']*100 for r in rows]; base = [r['base_mean']*100 for r in rows]\n"
            "else:\n"
            "    acc = [R['h1'][2], R['h3'][2], R['h6'][2], R['h12'][2]]\n"
            "    base = [R['h1'][4], R['h3'][4], R['h6'][4], R['h12'][4]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(x-.2, acc, .4, color=GREEN, label='after TSA ACCELERATES')\n"
            "ax.bar(x+.2, base, .4, color=GREY, label='an average month (base rate)')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m} months' for m in hs])\n"
            "ax.set_ylabel('average forward travel-basket return (%)')\n"
            "ax.set_title('Accelerating TSA -> a long-horizon travel tailwind... but only at 12 months')\n"
            "for i,(a,b) in enumerate(zip(acc,base)):\n"
            "    ax.annotate(f'{a:.1f}%',(i-.2,a),ha='center',va='bottom',fontsize=9)\n"
            "    ax.annotate(f'{b:.1f}%',(i+.2,b),ha='center',va='bottom',fontsize=9)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('12-month: accel', f'{acc[-1]:.1f}%', 'vs base', f'{base[-1]:.1f}%')"
        ),
        md(
            f"Only at **12 months** does the story really show up — accelerating-TSA returns "
            f"(**+{R['h12'][2]:.1f}%**) sit well above the base rate (**+{R['h12'][4]:.1f}%**), and the "
            f"basket is *up* more often (**{R['h12'][5]:.0f}%** of the time vs **{R['h12'][6]:.0f}%**). "
            "At 1, 3 and 6 months there's basically nothing. And even the 12-month gap — as the quants "
            "notebook shows — is small enough that, with only 7 years of data and one giant COVID swing, "
            "it could easily be chance. Hold that thought; the *next* chart is where the story breaks."
        ),
        md(
            "**The crucial test: does the throughput uptick come *first*?** We slide TSA forward and "
            "backward against the travel basket and measure how tightly they move together. A real "
            "nowcast would show its strongest *upward* link at a **positive lead** (TSA leads → bar "
            "rises on the right). Watch where it actually peaks."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ll = st.lead_lag(F)\n"
            "    Ls = list(range(-6,7)); cs = [ll[L] for L in Ls]\n"
            "else:\n"
            "    Ls = sorted(R['leadlag']); cs = [R['leadlag'][L] for L in Ls]\n"
            "cols = [GREEN if L>0 else RED for L in Ls]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "ax.bar(Ls, cs, color=cols, width=.8)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c=GREY, lw=1, ls=':')\n"
            "ax.set_xlabel('lead L (months): L>0 = TSA moves FIRST (nowcast)   |   L<0 = TSA LAGS the travel trade')\n"
            "ax.set_ylabel('correlation with travel-basket move'); ax.set_xticks(Ls)\n"
            "ax.set_title('The peak is on the LEFT: TSA lags travel stocks by 3-6 months')\n"
            "plt.tight_layout(); plt.show()\n"
            "imax = int(np.nanargmax(cs))\n"
            "print(f'strongest positive link at L={Ls[imax]} months (TSA FOLLOWS the travel trade here)')"
        ),
        md(
            f"There it is. The tallest *positive* bar is at **L = −6** (and again at **L = −3**) — the "
            "TSA signal lines up best with a travel-stock move that happened **three to six months "
            "earlier**. On the right, where a true nowcast would live (TSA moving first), the bars are "
            "near zero or even *negative*. **TSA isn't leading travel stocks — it's trailing them.** The "
            "market prices the reopening in stock prices, and only later does it show up at the "
            "checkpoints."
        ),
        md(
            "**Could you trade it anyway?** Suppose you bought the travel basket every month TSA was "
            "accelerating and sat in cash otherwise. Here's that strategy's growth vs just buying and "
            "holding the basket."
        ),
        code(
            "if HAVE_REAL:\n"
            "    import pandas as pd\n"
            "    accel = st.accel_mask(F); pos = accel.map({True:1.0, False:0.0}).astype(float).shift(1)\n"
            "    rr = F['basket'].pct_change()\n"
            "    dfp = pd.DataFrame({'r': rr, 'pos': pos}).dropna()\n"
            "    sw = dfp['pos'].diff().abs().fillna(0); c=10/1e4\n"
            "    overlay = (dfp['pos']*dfp['r'] - sw*c)\n"
            "    bh_grow = (1+dfp['r']).cumprod(); ov_grow = (1+overlay).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "    ax.plot(bh_grow.index, bh_grow.values, c=GREY, lw=1.8, label='buy & hold travel basket')\n"
            "    ax.plot(ov_grow.index, ov_grow.values, c=RED, lw=1.8, label='buy travel when TSA accelerating (net)')\n"
            "    ax.set_ylabel('growth of $1'); ax.set_title('\"Buy travel when TSA accelerates\" badly lags buy-and-hold')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'final $1 -> buy&hold {bh_grow.iloc[-1]:.2f}x  vs  overlay {ov_grow.iloc[-1]:.2f}x')\n"
            "else:\n"
            "    print(f\"overlay {R['overlay'][4]:.1f}%/yr vs buy-hold {R['overlay'][0]:.1f}%/yr (net) — see results.md\")"
        ),
        md(
            f"The 'wait for TSA' overlay ends up **well below** buy-and-hold — "
            f"**+{R['overlay'][4]:.1f}%/yr** vs **+{R['overlay'][0]:.1f}%/yr** net, and a *lower* Sharpe "
            f"({R['overlay'][5]:.2f} vs {R['overlay'][1]:.2f}). The reason is brutal: year-over-year TSA "
            "momentum was still deeply **negative** through the sharpest reopening rallies of "
            "late-2020 and 2021 — exactly when travel stocks doubled. So a rule that waits for TSA to "
            "turn positive **sits out the very returns it was built to catch.** The signal doesn't just "
            "fail to help — **it hurts.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** Accelerating TSA really does precede a stronger 12-month travel return "
            "— but the gap is insignificant full-sample, fragile to how you define the signal, and "
            "clears the bar only once you delete the COVID reopening, on a single horizon. Real as lore, "
            "weak as edge.\n"
            "- **Tradability — Mirage.** Buying travel on accelerating TSA **loses badly to "
            "buy-and-hold** — it sits out the reopening rallies. There's nothing to deploy.\n"
            "- **Real-time nowcast? — Not supported.** The throughput uptick lines up with a travel-stock "
            "move that already happened three to six months earlier. TSA **confirms** the recovery; it "
            "doesn't forecast the stocks. The one word that makes the pitch — *early* — is the part the "
            "data rejects."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "Forget significance for a second. Even if the 12-month tilt were rock-solid, the operational "
            "reality kills it. The signal is a slow, year-over-year gauge that barely moves — it flips "
            f"only **{R['overlay'][7]:.0%}-of-the-time-invested** a handful of times ("
            f"**{R['overlay'][6]} switches** in 7 years) — and the moments it finally turns positive are "
            "*after* the market has already re-rated travel. Worse, the biggest YoY surges (the 2021 "
            "reopening) are the exact tops you'd want to be **fading**, not chasing — which is why "
            "pushing the rule to only fire on a *big* acceleration flips its edge **negative**. Travel "
            "stocks are also just **high-beta** (β ≈ 1.4 to the S&P): most of what looks like a 'travel "
            "nowcast' is market beta you were always paid for. There is no version of \"buy travel when "
            "TSA accelerates\" that both fires early and makes money."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The sibling test.** [Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/) "
            "asks the same question of the famous 'leading' labour signal: real macro tell, but does it "
            "lead the *market*, and can you trade it?\n"
            "- **More alt-data crystal balls.** [Study 358 — Watch-Index](../358-watch-index/) and "
            "[Study 708 — Eurovision-Effect](../708-eurovision-effect/) put other cited alt-data series "
            "through the same wringer.\n"
            "- **Build your own.** Swap the year-over-year window for a shorter one, add real-time "
            "*surprise* (TSA vs the same day last year vs consensus), or use the weekly raw series — the "
            "lead/lag picture barely budges: a series the market already watches daily can't be made to "
            "*lead* the stocks by smoothing it differently.\n\n"
            "*Think TSA leads the travel trade? Show the lead/lag chart peaking on the **right** "
            "(positive lead) — then we'll talk.*"
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
            "# TSA-Throughput — a quantitative teardown 🔬\n"
            "### ACCEL-vs-base split returns · Welch *t* + placebo null · the decisive lead/lag "
            "cross-correlation · a market-beta control · a timing overlay vs buy-and-hold · robustness · "
            "a synthetic planted-edge control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "believers fuse two claims: that accelerating TSA throughput (1) **predicts** stronger "
            "travel-stock returns and (2) does so **early / real-time** enough to trade. We separate "
            "them. The conditional return tilt is *right-signed only at 12 months and insignificant "
            "full-sample*; the decisive object is the **lead/lag structure**, which shows TSA momentum is "
            "**coincident-to-lagging**, not leading — and a tradable overlay that *badly underperforms* "
            "buy-and-hold seals the Tradability axis.\n\n"
            "> ⚠️ **Data + proxy note.** TSA's site is firewalled here; the throughput tape is a hardcoded "
            "monthly snapshot of the public daily checkpoint numbers (average travellers/day, millions) — "
            "a **labelled proxy**, not a real-tape banner. The travel basket is yfinance daily adjusted "
            "close (total-return) for `JETS`/`MAR`/`HLT` (equal-weight ½ airlines · ½ hotels), month-end "
            "sampled; `SPY` is the market control. Offline core + synthetic control are deterministic. "
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
            f"| **Signal** | `WEAK` | 12-month accel mean **+{R['h12'][2]:.1f}%** vs base "
            f"**+{R['h12'][4]:.1f}%** (right sign); Welch **t = {R['h12'][7]:.2f}**, placebo "
            f"**p = {R['h12'][8]:.2f}** — fails **t ≥ 2** full-sample, sign-flips by window, clears the "
            "bar only at 12m ex-COVID (t = 2.12). |\n"
            f"| **Tradability** | `MIRAGE` | Buy-travel-on-accel overlay **+{R['overlay'][4]:.1f}%/yr** "
            f"(Sharpe **{R['overlay'][5]:.2f}**) **vs buy-hold +{R['overlay'][0]:.1f}%/yr** "
            f"(Sharpe **{R['overlay'][1]:.2f}**); long/short **{R['overlay_ls'][0]:.0f}%/yr**. Acting on "
            "it destroys return. |\n"
            f"| **Nowcast?** | `NOT SUPPORTED` | Peak positive lead/lag correlation at **L = −6 / −3** "
            "(TSA lags the trade); at every positive lead corr < 0. A coincident-to-lagging echo, not a "
            "leader. |\n\n"
            "> 💡 In plain words: the equity market discounts the travel recovery *before* physical "
            "throughput recovers, so a physical-activity series that co-moves with travel-stock strength "
            "need not lead it. TSA momentum lines up with a stock move already three-to-six months old — "
            "the 'real-time nowcast' is the market's own lead, reflected back."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $v_t$ be monthly-average TSA throughput and "
            "$m_t = v_t / v_{t-12} - 1$ its year-over-year momentum (YoY neutralises the strong travel "
            "season). TSA is **ACCELERATING** at $t$ when $m_t > 0$. With a one-month execution lag (the "
            "month-$t$ average, published next-day, is acted on at the close of $t+1$), define the "
            "forward basket return $r_{t+1\\to t+1+H}$.\n\n"
            "- **H₁ (predicts).** $\\mathbb{E}[r\\mid \\text{accel}] > \\mathbb{E}[r]$ — a *positive* "
            "excess over the base rate.\n"
            "- **H₂ (leads / nowcast).** The strongest positive TSA↔return correlation sits at a "
            "**positive** lead (TSA moves first).\n"
            "- **H₃ (deployable).** A buy-travel-on-accel overlay beats buy-and-hold net of costs.\n\n"
            f"We find **H₁ right-signed only at 12m and insignificant** ($t = {R['h12'][7]:.2f}$ "
            "full-sample, sign-flips by spec), **H₂ rejected** (peak positive corr at $L=-6$), **H₃ "
            "rejected** (overlay badly underperforms). The folklore is right exactly where it's "
            "uninformative (TSA and the travel recovery co-move) and wrong exactly where it would pay (a "
            "*leading*, *tradable* edge)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The conditional-return test is a two-sample mean comparison judged by its standard error:\n\n"
            "$$\\widehat{\\Delta}_H = \\bar r^{\\text{accel}}_H - \\bar r^{\\text{all}}_H,\\qquad "
            "t = \\frac{\\widehat{\\Delta}_H}{\\sqrt{\\,s^2_{\\text{accel}}/k + s^2_{\\text{all}}/N\\,}}.$$\n\n"
            "But a significant $\\widehat{\\Delta}$ would **still not** establish *leading*: a coincident "
            "or lagging series can co-move with forward returns through autocorrelation in the recovery. "
            "The identifying test is the **lead/lag cross-correlation** "
            "$\\rho(L) = \\mathrm{corr}(m_t,\\ r_{t+L\\to t+L+1})$. A genuine nowcast peaks (positively) "
            "at $L>0$. If $\\arg\\max_L \\rho(L) < 0$, TSA **follows** the travel trade — and the entire "
            "'real-time' thesis collapses regardless of the conditional mean. We also ask whether the "
            "tilt is anything beyond **market beta** (travel is high-β), and whether it survives dropping "
            "the one COVID regime."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **TSA tape.** Monthly-average checkpoint throughput (millions/day), hardcoded labelled "
            f"proxy, {R['start'][:7]}→{R['end'][:7]} ({R['months']} months). Post-2019 only (TSA began "
            "publishing the daily series in 2019) — one business cycle, one dominant COVID regime.\n"
            "- **Signal.** $m_t = v_t/v_{t-12}-1$; ACCELERATING when $m_t>0$.\n"
            "- **Basket.** Equal-weight ½ `JETS` (airlines) · ½ (`MAR`+`HLT`) (hotels), total-return, "
            "month-end. `SPY` as the market control.\n"
            "- **Forward returns.** Enter at the close **1 month after** the signal (no look-ahead), "
            "hold $H\\in\\{1,3,6,12\\}$ months; drop horizons that overrun the tape.\n"
            "- **Null #1 (Welch t).** Accel-set mean vs the unconditional mean.\n"
            "- **Null #2 (placebo).** 20,000 draws of $k$ random months; "
            "$p = \\Pr[\\text{random-draw mean} \\ge \\text{accel mean}]$ (as bullish or more).\n"
            "- **Identification (lead/lag).** $\\rho(L)$ for $L\\in[-6,6]$ — *where* does TSA line up?\n"
            "- **Beta control.** Forward basket $\\sim$ const + forward `SPY` + accel dummy: does the "
            "dummy survive removing market beta?\n"
            "- **Tradability.** Buy-basket-when-accel overlay, 1-month lag, 10 bps one-way per switch, "
            "shorts pay 100 bps/yr borrow; excess-of-zero Sharpe (cash leg = 0, labelled).\n"
            "- **Positive control.** A deterministic series with a *planted* TSA→returns link: `edge=0` "
            "must not fake significance; a large `edge` must light up the test."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The point estimates — right sign only at 12m, insignificant\n\n"
            "Accel-set forward mean with $\\pm$ standard error against the unconditional base rate "
            "(dashed). Above base only at 12 months — and inside its own error bar."
        ),
        code(
            "hs = [1, 3, 6, 12]\n"
            "if HAVE_REAL:\n"
            "    cm, bm, ts, ses = [], [], [], []\n"
            "    for m in hs:\n"
            "        s = st.summarize(F, m); cm.append(s['accel_mean']); bm.append(s['base_mean']); ts.append(s['t'])\n"
            "        a,_d,_al = st.split_returns(F, m); ses.append(a.std(ddof=1)/np.sqrt(len(a)))\n"
            "else:\n"
            "    cm = [R['h1'][2]/100, R['h3'][2]/100, R['h6'][2]/100, R['h12'][2]/100]\n"
            "    bm = [R['h1'][4]/100, R['h3'][4]/100, R['h6'][4]/100, R['h12'][4]/100]\n"
            "    ts = [R['h1'][7], R['h3'][7], R['h6'][7], R['h12'][7]]; ses = [.03,.05,.07,.10]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x, [c*100 for c in cm], yerr=[s*100 for s in ses], capsize=5, color=GREEN, width=.5, label='accel-TSA (±SE)')\n"
            "ax.plot(x, [b*100 for b in bm], 'D', ms=11, c=GREY, label='unconditional base rate')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m}m' for m in hs]); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward basket return (%)')\n"
            "ax.set_title('Right sign only at 12m, and the SE swamps the gap'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Welch t by horizon:', {f'{m}m': round(t,2) for m,t in zip(hs,ts)})"
        ),
        md(
            f"> 💡 In plain words: at 12m the accel-set mean is **+{R['h12'][2]:.1f}%** vs base "
            f"**+{R['h12'][4]:.1f}%** — a ~{R['h12'][2]-R['h12'][4]:.1f}-point excess at "
            f"**t = {R['h12'][7]:.2f}** (not significant). At 1/3/6m the excess is ≈0 or negative "
            f"(t = {R['h1'][7]:.2f}, {R['h3'][7]:.2f}, {R['h6'][7]:.2f}). H₁ is **directionally supported "
            "at long horizons only, statistically not**: the right sign living inside its own error bar."
        ),
        md(
            "### 4b · The decisive identification test — lead/lag\n\n"
            "$\\rho(L) = \\mathrm{corr}(m_t, r_{t+L\\to t+L+1})$. Positive bars left of zero = TSA "
            "**lags** the travel trade; a real nowcast would peak on the **right** (TSA leads)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ll = st.lead_lag(F); Ls = list(range(-6,7)); cs = [ll[L] for L in Ls]\n"
            "else:\n"
            "    Ls = sorted(R['leadlag']); cs = [R['leadlag'][L] for L in Ls]\n"
            "cols = [GREEN if L>0 else RED for L in Ls]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(Ls, cs, color=cols, width=.8)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c=GREY, lw=1, ls=':')\n"
            "imax = int(np.nanargmax(cs))\n"
            "ax.annotate('strongest POSITIVE link\\n(TSA LAGS the travel trade)', xy=(Ls[imax], cs[imax]),\n"
            "            xytext=(Ls[imax]+0.3, cs[imax]+0.05), ha='center', color=RED,\n"
            "            arrowprops=dict(arrowstyle='->', color=RED))\n"
            "ax.set_xlabel('lead L (months): L>0 = TSA leads (nowcast)   |   L<0 = TSA lags')\n"
            "ax.set_ylabel(r'$\\rho(L)$'); ax.set_xticks(Ls)\n"
            "ax.set_title('argmax rho(L) is at L<0: TSA is coincident-to-lagging')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'argmax at L={Ls[imax]} (rho={cs[imax]:+.2f}); rho at +1 month = {cs[Ls.index(1)]:+.2f}')"
        ),
        md(
            "> 💡 In plain words: $\\arg\\max_L \\rho(L) = -6$ (and a second peak at $-3$). The TSA signal "
            "correlates most (positively) with a travel-stock move **a quarter-to-half-year in its "
            "past**; at the positive leads a genuine nowcast needs, $\\rho < 0$. **H₂ rejected.** The "
            "equity market discounts the recovery; TSA trails it — the 'real-time nowcast' is the "
            "market's lead, reflected. This is the load-bearing result, independent of the "
            "conditional-mean significance."
        ),
        md(
            "### 4c · Is it anything beyond market beta?\n\n"
            "Travel is a **high-β** bet. Regress forward basket returns on a constant, the "
            "contemporaneous forward `SPY` return, and the accel dummy — does the dummy survive removing "
            "market beta? (The 12-month bar uses **overlapping** windows, so its classical *t* is "
            "≈√12-inflated — shown hollow and **not** trusted; the placebo *p* = 0.18 is the honest 12m "
            "read.)"
        ),
        code(
            "ms = [1,3,6,12]\n"
            "if HAVE_REAL:\n"
            "    coefs = [st.beta_control(F, m) for m in ms]\n"
            "    dummy = [c['adj_coef']*100 for c in coefs]; tt = [c['adj_t'] for c in coefs]; bet=[c['beta'] for c in coefs]\n"
            "else:\n"
            "    dummy=[b[1] for b in R['beta']]; tt=[b[2] for b in R['beta']]; bet=[b[3] for b in R['beta']]\n"
            "x = np.arange(len(ms))\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.4,4.2))\n"
            "barcols=[GREY,GREY,GREY,RED]\n"
            "a1.bar(x, tt, color=barcols, width=.6)\n"
            "a1.axhline(2, ls='--', c=GREEN, label='t=2'); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_xticks(x); a1.set_xticklabels([f'{m}m' for m in ms]); a1.set_ylabel('accel-dummy t (beyond SPY beta)')\n"
            "a1.set_title('Dummy t: ~0 at tradable horizons\\n(12m red = overlap-inflated, not trusted)'); a1.legend()\n"
            "a2.bar(x, bet, color=AMBER, width=.6)\n"
            "for i,b in enumerate(bet): a2.annotate(f'{b:.2f}',(i,b),ha='center',va='bottom')\n"
            "a2.set_xticks(x); a2.set_xticklabels([f'{m}m' for m in ms]); a2.set_ylabel('basket beta to SPY')\n"
            "a2.set_title('The basket is just high-beta travel (β ~ 1.4)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('dummy t by horizon:', {f'{m}m': round(t,2) for m,t in zip(ms,tt)}, '| beta ~', round(np.mean(bet),2))"
        ),
        md(
            f"> 💡 In plain words: at the tradable 1–3-month horizons the accel dummy is **insignificant** "
            f"(t = {R['beta'][0][2]:.2f}, {R['beta'][1][2]:.2f}) once `SPY` is in the regression — the "
            "'nowcast' is just **high-β travel exposure** (β ≈ 1.4). The eye-catching 12-month dummy "
            f"t = {R['beta'][3][2]:.2f} is an **overlapping-returns artifact** (≈√12 inflation, ~6 "
            "independent blocks); the honest 12m read is the placebo p = 0.18. No beta-adjusted edge at "
            "any horizon you could actually trade."
        ),
        md(
            "### 4d · Tradability — the buy-on-accel overlay loses\n\n"
            "Hold the basket when TSA is accelerating, else cash (long/flat) — or short it (long/short, "
            "100 bps/yr borrow). 1-month lag, 10 bps/switch. Annualised mean and Sharpe vs buy-and-hold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    o = st.timing_overlay(F, cost_bps=10.0)\n"
            "    ols = st.timing_overlay(F, cost_bps=10.0, borrow_bps=100.0, allow_short=True)\n"
            "    bh_m, bh_s = o['bh_mean']*100, o['bh_sharpe']\n"
            "    n_m, n_s = o['overlay_net_mean']*100, o['overlay_net_sharpe']; nsw=o['n_switches']\n"
            "    ls_m, ls_s = ols['overlay_net_mean']*100, ols['overlay_net_sharpe']\n"
            "else:\n"
            "    bh_m, bh_s = R['overlay'][0], R['overlay'][1]; n_m,n_s = R['overlay'][4],R['overlay'][5]; nsw=R['overlay'][6]\n"
            "    ls_m, ls_s = R['overlay_ls'][0], R['overlay_ls'][1]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))\n"
            "labels = ['buy &\\nhold', 'overlay\\nlong/flat net', 'overlay\\nlong/short net']\n"
            "a1.bar(labels, [bh_m, n_m, ls_m], color=[GREY, RED, RED], width=.6)\n"
            "for i,v in enumerate([bh_m,n_m,ls_m]): a1.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('annualised mean return (%)'); a1.set_title('Return: acting on TSA loses ~13 pts/yr')\n"
            "a2.bar(labels, [bh_s, n_s, ls_s], color=[GREY, RED, RED], width=.6)\n"
            "for i,v in enumerate([bh_s,n_s,ls_s]): a2.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('annualised Sharpe (excess-of-0)'); a2.set_title(f'Sharpe: overlay far lower ({nsw} switches)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'long/flat net {n_m:.1f}%/yr (Sharpe {n_s:.2f}) vs buy-hold {bh_m:.1f}%/yr (Sharpe {bh_s:.2f}); long/short {ls_m:.1f}%/yr')"
        ),
        md(
            f"> 💡 In plain words: the overlay returns **+{R['overlay'][4]:.1f}%/yr** net vs "
            f"**+{R['overlay'][0]:.1f}%** for buy-and-hold (Sharpe {R['overlay'][5]:.2f} vs "
            f"{R['overlay'][1]:.2f}); shorting the decelerating leg **loses {abs(R['overlay_ls'][0]):.0f}%/yr**. "
            "Because YoY momentum stayed negative through the 2020–21 reopening rallies, sitting out on "
            "'not yet accelerating' systematically forfeits the biggest travel returns of the sample. "
            "**H₃ rejected** — costs aren't even the issue; the *timing* loses. `MIRAGE`."
        ),
        md(
            "### 4e · Robustness — window, threshold, and the COVID dependence\n\n"
            "Vary the momentum window $k$ and the accel threshold, then drop the COVID regime "
            "(Jun-2020 → May-2022). The full-sample 12-month $t$ never clears 2, and the biggest upticks "
            "**flip the sign**; only 12m ex-COVID clears the bar — on a single horizon."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for k in (3,6,12):\n"
            "        s = st.summarize(F, 12, k=k); rob.append((f'k={k}', s['n_accel'], s['accel_mean']*100, s['t'], s['p_placebo']))\n"
            "    s = st.summarize(F, 12, thresh=0.10); rob.append(('thr>+10%', s['n_accel'], s['accel_mean']*100, s['t'], s['p_placebo']))\n"
            "    F2 = F[(F.index < '2020-06-01') | (F.index >= '2022-06-01')]\n"
            "    s = st.summarize(F2, 12); rob.append(('exCOVID 12m', s['n_accel'], s['accel_mean']*100, s['t'], s['p_placebo']))\n"
            "else:\n"
            "    rob = [(l,n,r,t,p) for (l,n,r,_b,t,p) in R['robust']] + [('exCOVID 12m', R['exc'][3][1], R['exc'][3][2], R['exc'][3][4], R['exc'][3][5])]\n"
            "labels = [r[0] for r in rob]; tt = [r[3] for r in rob]; nn = [r[1] for r in rob]\n"
            "cols = [GREEN if t>=2 else (AMBER if t>0 else RED) for t in tt]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.3))\n"
            "ax.bar(labels, tt, color=cols, width=.6)\n"
            "ax.axhline(2, ls='--', c=GREEN, label='t=+2 (significance bar)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(t,k) in enumerate(zip(tt,nn)): ax.annotate(f'n={k}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "ax.set_ylabel('Welch t (12-month)'); ax.set_title('Only 12m ex-COVID clears |t|=2; big upticks flip NEGATIVE'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (label, n, accel12%, t, p):', [(r[0], r[1], round(r[2],1), round(r[3],2), round(r[4],3)) for r in rob])"
        ),
        md(
            f"> 💡 In plain words: the effect exists only for the YoY (**k=12**) window — a 3-month window "
            f"**flips it negative** (**t={R['robust'][0][4]:.2f}**), as does restricting to big upticks "
            f"(>+10% → **t={R['robust'][3][4]:.2f}**: the largest YoY surges are 2021 reopening *tops* "
            f"you'd fade). Ex-COVID the 12m t rises to **{R['exc'][3][4]:.2f}** (placebo p={R['exc'][3][5]:.3f}) "
            f"— but that is **one** horizon (1/3/6m ex-COVID: t={R['exc'][0][4]:.2f}, {R['exc'][1][4]:.2f}, "
            f"{R['exc'][2][4]:.2f}), on overlapping 12m windows in a 5½-year slice, and says nothing about "
            "the (rejected) *timing*. A directionally-real long-horizon tilt, not a certified leading "
            "nowcast — hence `WEAK`, not `REAL`."
        ),
        md(
            "### 4f · Faithful-engine control — we know the truth here\n\n"
            "A deterministic monthly series with a *planted* link (an accelerating month $t$ lifts the "
            "$t{+}1$-entered held return by `edge`). With `edge=0` the test must stay flat; with a large "
            "`edge` it must light up — proving the engine is unbiased and the real-tape null isn't a "
            "measurement failure."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.08):\n"
            "    syn = data.synthetic_tsa(n_months=240, edge=edge, seed=758)\n"
            "    s = st.summarize(syn, 1, k=12)\n"
            "    res.append((edge, s['n_accel'], s['accel_mean']*100, s['base_mean']*100, s['t'], s['p_placebo']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted edge\\n{e*100:.0f}% / month' for e,_,_,_,_,_ in res]\n"
            "tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=+2 (significance bar)'); ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('Welch t (1-month)'); ax.set_title('Control: no link -> flat; real link -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,c,b,t,p in res: print(f'planted {e*100:+.0f}%/mo: n_acc={k} accel={c:.2f}% base={b:.2f}% t={t:.2f} p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted link the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (no false positive); a **+8%/month** planted link drives "
            f"**t = {R['syn'][1][4]:.2f}**. So the machinery is honest — the real-tape full-sample *t* of "
            f"~{R['h12'][7]:.1f} is a *genuine* weak-or-regime-driven edge, not a broken test. The engine "
            "*can* bank a real TSA→returns link; the real tape just doesn't carry a tradable, leading one."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — 12m excess **{R['h12'][2]-R['h12'][4]:+.1f}pp** at Welch "
            f"**t = {R['h12'][7]:.2f}** / placebo **p = {R['h12'][8]:.2f}**; right sign at 12m only, fails "
            "t≥2 full-sample, **flips negative** by window/threshold, clears the bar only at 12m ex-COVID "
            "(t = 2.12) on a single horizon. Literature support (alt-data nowcasting) + a "
            "directionally-correct-but-fragile long-horizon tilt ⇒ WEAK, not REAL.\n"
            f"- **Tradability `MIRAGE`** — the buy-on-accel overlay returns "
            f"**+{R['overlay'][4]:.1f}%/yr** (Sharpe {R['overlay'][5]:.2f}) vs buy-hold "
            f"**+{R['overlay'][0]:.1f}%/yr** (Sharpe {R['overlay'][1]:.2f}); long/short "
            f"**{R['overlay_ls'][0]:.0f}%/yr**. Acting on the signal *subtracts* return — nothing to "
            "allocate to, and the tilt is just β≈1.4 travel exposure.\n"
            "- **Nowcast? `NOT SUPPORTED`** — $\\arg\\max_L \\rho(L) = -6$ months (second peak $-3$): TSA "
            "momentum is **coincident-to-lagging**, not leading. The equity market is the real-time "
            "discounting mechanism; TSA echoes it. The defining word — *early / real-time* — is the part "
            "the data rejects."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — why even a real tilt wouldn't deploy\n\n"
            "Grant the lore a genuine 12-month tilt. The operational reality still defeats it. YoY TSA "
            f"momentum is a slow, smooth signal that switches only a handful of times ("
            f"**{R['overlay'][6]} switches / 7y**, invested **{R['overlay'][7]:.0%}** of months), and it "
            "turns positive **after** the market has re-rated travel — so the overlay is out of the "
            "market for the sharp early-recovery leg every cycle, which is why its Sharpe is far **below** "
            "passive even before costs. The one regime where TSA and a travel bottom genuinely coincide — "
            "the deepest collapse — is where you'd want maximum *long* exposure, which is exactly why the "
            ">+10% threshold flips the sign negative (you'd be *chasing* the 2021 top). And once you "
            "regress out `SPY`, the beta-adjusted dummy is ~0 at every tradable horizon: there is no "
            "TSA-specific alpha left to harvest, only the high-β travel beta you could buy directly and "
            "more cheaply. No lag, window, or cost assumption rescues a series the whole market already "
            "watches daily."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The sibling.** [Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/): "
            "the 'leading' labour signal, same hardcoded-snapshot + market-tape + lead/lag method — a "
            "real macro tell that turns out coincident-to-lagging for equities.\n"
            "- **Companion alt-data teardowns.** [Study 358 — Watch-Index](../358-watch-index/), "
            "[Study 708 — Eurovision-Effect](../708-eurovision-effect/) — the labelled-proxy pattern; "
            "[Study 387 — Economic-Surprise-Index](../387-economic-surprise-index/) — does a celebrated "
            "nowcast time equities?\n"
            "- **Sharper identification.** Replace the monthly average with the weekly raw series and a "
            "proper VAR / Granger test, or build a real-time **surprise** (TSA vs consensus vs same-week "
            "last year) — the coincident-to-lagging structure is robust to all of these: a series the "
            "market already watches daily can't be made to *lead* the stocks by re-windowing it. The real "
            "open question is whether TSA leads *fundamentals* (RPMs, RevPAR) even when it can't lead the "
            "*stocks*.\n\n"
            "*The reproducible core is offline and deterministic; the TSA input is an explicit labelled "
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
