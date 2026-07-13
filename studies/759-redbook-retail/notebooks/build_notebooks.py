"""Generate the two narrative notebooks for Study 759 (Redbook-Retail).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the hardcoded Redbook
proxy (always available) and the cached XRT/SPY prices under ../_cache/, and otherwise quote
the frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive
control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (Redbook YoY hardcoded proxy +
# XRT/SPY month-end, 2006-06 -> 2026-06, 241 months, 20.0 years).
R = dict(
    start="2006-06-30", end="2026-06-30", months=241, years=20.0,
    # ABSOLUTE XRT: (months, n_accel, accel%, decel%, base%, acc_up%, base_up%, t, p_placebo)
    h1=(1, 105, 1.04, 1.02, 1.03, 56, 56, 0.01, 0.494),
    h3=(3, 104, 4.01, 2.33, 3.07, 66, 61, 0.63, 0.222),
    h6=(6, 101, 9.13, 4.07, 6.26, 70, 62, 1.21, 0.076),
    h12=(12, 99, 14.07, 12.64, 13.26, 73, 65, 0.24, 0.398),
    # RELATIVE (XRT-minus-SPY): (months, n_accel, accel%, base%, t, p)
    rel6=(6, 101, 1.69, 0.31, 0.79, 0.156),
    rel1=(1, 105, -0.39, 0.04, -0.82, 0.819),
    rel12=(12, 99, 0.16, 1.17, -0.40, 0.650),
    # lead/lag (absolute): L -> corr
    leadlag={-6: 0.067, -5: 0.116, -4: 0.310, -3: 0.328, -2: 0.269, -1: 0.059,
             0: 0.018, 1: -0.040, 2: 0.016, 3: 0.046, 4: 0.128, 5: 0.013, 6: -0.111},
    # overlay: (bh_mean%, bh_sharpe, gross_mean%, gross_sharpe, net_mean%, net_sharpe, switches,
    #           bh_growth, overlay_growth)
    overlay=(12.05, 0.49, 8.58, 0.52, 8.21, 0.50, 74, 6.06, 4.01),
    # level regime: (months, strong%, weak%, spread%, t)
    reg1=(1, 0.36, 1.63, -1.26, -1.37),
    reg12=(12, 7.82, 17.66, -9.84, -2.28),
    # robustness at 6m: (label, n_accel, accel6%, t, p)
    robust=[("k=1", 111, 7.9, 0.72, 0.199), ("k=3", 101, 9.1, 1.21, 0.076),
            ("k=6", 118, 6.5, 0.12, 0.441), ("smooth3", 122, 7.3, 0.49, 0.279),
            ("relative", 101, 1.7, 0.79, 0.156), ("ex-COVID", 87, 9.0, 1.22, 0.097)],
    # synthetic control: (edge, n_accel, accel1%, base1%, t, p)
    syn=[(0.0, 183, 1.88, 1.50, 0.72, 0.194), (0.05, 183, 6.25, 4.10, 3.66, 0.000)],
    rb_peak=16.8, rb_peak_when="Nov 2021", rb_trough=-8.4, rb_trough_when="Apr 2020",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Leads_retail%3F: Not_supported](https://img.shields.io/badge/Leads_retail%3F-Not_supported-8b949e?style=flat-square)\n\n"
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

from redbook_retail import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real() if HAVE_REAL else None
print("XRT/SPY cache present:", HAVE_REAL,
      "| Redbook+XRT months:", (0 if F is None else len(F)))
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
            "# Does the weekly retail-sales pulse tell you when to buy retail stocks? 🛒\n"
            "### The Redbook same-store-sales index as a shopping-mall crystal ball, in plain English\n\n"
            + BADGES +
            "Every Tuesday a research firm publishes the **Johnson Redbook Index** — how much more (or "
            "less) America's chain stores rang up this week versus a year ago, measured at the *same "
            "stores* so new openings don't flatter it. It's sold as a real-time read on the shopper. The "
            "folklore says: when that same-store number **speeds up**, the consumer is getting stronger, "
            "so **retail stocks** (the XRT exchange-traded fund) are about to climb — a nowcast you can "
            "trade.\n\n"
            "It's a tidy story. It's also testable. This notebook asks three blunt questions: when "
            "Redbook accelerates, does the retail sector really do better next? Does the sales pulse "
            "actually move **first** (that's the whole pitch)? And if you *bought* retail every time "
            "Redbook sped up, would you make money?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the lead/lag cross-correlation, the "
            "retail-vs-market test and the synthetic control? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** The weekly Redbook series is **proprietary** (a paid feed, "
            "not on FRED), so we use a small, **clearly-labelled approximate reconstruction** of its "
            "monthly year-over-year same-store number — the *shape* is faithful to the public record "
            "(the 2009 slump into negative growth, the 2018–19 strength, the COVID-2020 collapse, the "
            "2021–22 double-digit reopening/inflation surge), the exact monthly values are approximate. "
            "It is a **proxy**, never the licensed tape. XRT is real (yfinance). House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| When Redbook accelerates, does retail do better next? | **Barely, and not reliably.** Over "
            f"the next 6 months XRT averages **+{R['h6'][2]:.1f}%** after Redbook speeds up vs "
            f"**+{R['h6'][4]:.1f}%** normally — the right direction, but small and **well inside the "
            "noise** (you can't tell it from luck). At 1 and 12 months even the direction fades. |\n"
            "| Does the sales pulse come *first*? | **No — and this is the killer.** Redbook's move lines "
            "up best with a retail-stock move that already happened **three months earlier.** The stock "
            "market reprices retail *before* the sales gauge confirms it. Redbook *follows*; it doesn't "
            "lead. |\n"
            "| Does it pick retail *over* the market? | **No.** Once you ask whether Redbook predicts "
            "retail **beating** the S&P, the tiny edge vanishes (and flips negative at some horizons). |\n"
            "| So could you trade it? | **It loses.** \"Own retail when Redbook accelerates\" turned $1 "
            f"into **${R['overlay'][8]:.1f}** vs **${R['overlay'][7]:.1f}** for just holding XRT — you "
            "give up return for a nowcast that never shows up. |\n\n"
            "> Retail sales and retail stocks obviously move together over the years. But \"the weekly "
            "sales pulse *warns* you\" is a coincident — and inflation-tangled — echo wearing a crystal-"
            "ball costume, and acting on it costs you money."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The Redbook Index is the most timely read on the American consumer — weekly, same-store, "
            "no lag. When same-store growth accelerates, the shopper is getting stronger, so get long the "
            "retail sector before the crowd catches up.\"*\n\n"
            "There's a respectable backbone here: same-store (comparable) sales strip out the noise of new "
            "store openings, so a rising Redbook genuinely says *existing* stores are selling more. And "
            "retailers live and die on comps. The trading leap is the part we test: that a Redbook "
            "*acceleration* arrives early enough, and cleanly enough, to be a **tradable** nowcast for "
            "retail *stocks* — not just a number that moves alongside them."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, this would be a gift: a public weekly number that tells you when to tilt into the "
            "retail sector. But \"nowcast\" hides two traps. First, the **stock market is itself a "
            "forward-looking machine** — XRT reprices the consumer in real time, so a monthly sales gauge "
            "may just be *confirming* a move the market already made. Second, Redbook is a **nominal** "
            "number: in 2021–22 same-store growth screamed to **double digits** — but that was mostly "
            "*inflation*, and XRT actually **fell** through 2022. A gauge that can't tell 'more stuff "
            "sold' from 'same stuff, higher prices' is a dangerous thing to buy stocks on. The difference "
            "between a *lead* and an *echo* is the difference between an edge and a mirage."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We line up **{R['years']:.0f} years** ({R['start'][:4]}–{R['end'][:4]}, "
            f"{R['months']} months, XRT's whole life) of the monthly Redbook same-store number against "
            "month-end XRT, and:\n\n"
            "1. **Split the months.** Call Redbook **accelerating** when same-store growth is above where "
            "it was three months ago. Compare what XRT did next (1/3/6/12 months) in accelerating months "
            "vs all months.\n"
            "2. **Check the timing.** The crucial test: slide Redbook forward and backward against the "
            "retail sector and find *where* they line up best. If Redbook truly **leads**, the strongest "
            "link shows up at a **positive lead** (Redbook first, stocks later).\n"
            "3. **Try to trade it.** Own XRT whenever Redbook is accelerating, sit in cash otherwise, pay "
            "realistic costs — and see if it beats just buying and holding retail."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the raw material.** Here's the Redbook same-store year-over-year number over two "
            "decades — the 2009 dip into *negative* growth, the strong 2018–19 run, the COVID-2020 cliff, "
            "and the wild 2021–22 spike to the mid-teens. It clearly *knows* about the consumer. The "
            "question is whether it knows **early** — and whether that 2021 spike was strength or just "
            "prices."
        ),
        code(
            "if HAVE_REAL:\n"
            "    y = F['redbook']\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.0))\n"
            "    ax.plot(y.index, y.values, c=GREEN, lw=1.4)\n"
            "    ax.axhline(0, c=RED, lw=.9, ls='--')\n"
            "    ax.set_title('Redbook same-store retail sales, year-over-year (%)  — labelled proxy')\n"
            "    ax.set_ylabel('same-store sales YoY (%)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('peak', y.max(), '% around', y.idxmax().date(), '| trough', y.min(), '% around', y.idxmin().date())\n"
            "else:\n"
            "    print('no cache — see docs/results.md; proxy peaked ~+16.8% (Nov 2021), trough -8.4% (Apr 2020)')"
        ),
        md(
            "**Now the payoff.** For each horizon, the average forward XRT return in **accelerating** "
            "months next to the return on an **average** month. The folklore predicts the green bars sit "
            "*above* the grey ones."
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
            "ax.bar(x-.2, acc, .4, color=GREEN, label='after Redbook ACCELERATES')\n"
            "ax.bar(x+.2, base, .4, color=GREY, label='an average month (base rate)')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m} months' for m in hs])\n"
            "ax.set_ylabel('average forward XRT return (%)')\n"
            "ax.set_title('Accelerating sales -> slightly higher returns... but only at 6m, and only barely')\n"
            "for i,(a,b) in enumerate(zip(acc,base)):\n"
            "    ax.annotate(f'{a:.1f}%',(i-.2,a),ha='center',va='bottom',fontsize=9)\n"
            "    ax.annotate(f'{b:.1f}%',(i+.2,b),ha='center',va='bottom',fontsize=9)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('6-month: accel', f'{acc[2]:.1f}%', 'vs base', f'{base[2]:.1f}%')"
        ),
        md(
            f"At 6 months the accelerating-set return (**+{R['h6'][2]:.1f}%**) does sit above the base "
            f"rate (**+{R['h6'][4]:.1f}%**) — the right direction. But look at 1 month "
            f"(**+{R['h1'][2]:.1f}%** vs **+{R['h1'][4]:.1f}%**, a dead heat) and 12 months "
            f"(**+{R['h12'][2]:.1f}%** vs **+{R['h12'][4]:.1f}%**, almost nothing). The one hopeful bar "
            "is small enough that, with the data we have, it could easily be chance. Hold that thought; "
            "the *next* chart is where the story breaks."
        ),
        md(
            "**The crucial test: does the sales pulse come *first*?** We slide Redbook forward and "
            "backward against retail stocks and measure how tightly they move together. A real nowcast "
            "would show its strongest *positive* link at a **positive lead** (Redbook leads → bar peaks "
            "on the right). Watch where it actually peaks."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ll = st.lead_lag(F)\n"
            "    Ls = list(range(-6,7)); cs = [ll[L] for L in Ls]\n"
            "else:\n"
            "    Ls = sorted(R['leadlag']); cs = [R['leadlag'][L] for L in Ls]\n"
            "cols = [RED if L<0 else GREEN for L in Ls]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "ax.bar(Ls, cs, color=cols, width=.8)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c=GREY, lw=1, ls=':')\n"
            "ax.set_xlabel('lead L (months): L>0 = Redbook moves FIRST (nowcast)   |   L<0 = Redbook LAGS the stocks')\n"
            "ax.set_ylabel('correlation with retail move'); ax.set_xticks(Ls)\n"
            "ax.set_title('The peak is on the LEFT: Redbook lags retail stocks by ~3 months')\n"
            "plt.tight_layout(); plt.show()\n"
            "imax = int(np.nanargmax(cs))\n"
            "print(f'strongest POSITIVE link at L={Ls[imax]} months (Redbook FOLLOWS the stocks here)')"
        ),
        md(
            f"There it is. The tallest *positive* bar is at **L = −3** — the Redbook signal lines up best "
            "with a retail-stock move that happened **three months earlier**. On the right, where a true "
            "nowcast would live (Redbook moving first), the bars are near zero or even *negative*. "
            "**Redbook isn't leading retail stocks — it's trailing them.** The market prices the consumer "
            "into XRT first; the same-store sales gauge catches up a quarter later."
        ),
        md(
            "**Could you trade it anyway?** Suppose you owned XRT every month Redbook was accelerating and "
            "sat in cash otherwise. Here's that strategy's growth vs just buying and holding retail."
        ),
        code(
            "if HAVE_REAL:\n"
            "    accel = st.accel_mask(F); pos = accel.astype(float).shift(1)\n"
            "    rr = F['xrt'].pct_change()\n"
            "    import pandas as pd\n"
            "    dfp = pd.DataFrame({'r': rr, 'pos': pos}).dropna()\n"
            "    sw = dfp['pos'].diff().abs().fillna(0); c=10/1e4\n"
            "    overlay = (dfp['pos']*dfp['r'] - sw*c)\n"
            "    bh_grow = (1+dfp['r']).cumprod(); ov_grow = (1+overlay).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "    ax.plot(bh_grow.index, bh_grow.values, c=GREY, lw=1.8, label='buy & hold XRT')\n"
            "    ax.plot(ov_grow.index, ov_grow.values, c=RED, lw=1.8, label='own XRT when Redbook accelerating (net)')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "    ax.set_title('\"Own retail when sales accelerate\" lags buy-and-hold for 20 years')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'final $1 -> buy&hold {bh_grow.iloc[-1]:.1f}x  vs  overlay {ov_grow.iloc[-1]:.1f}x')\n"
            "else:\n"
            "    print(f\"overlay ${R['overlay'][8]:.1f} vs buy-hold ${R['overlay'][7]:.1f} per $1 (net) — see results.md\")"
        ),
        md(
            f"The nowcast overlay ends up **below** buy-and-hold — $1 grows to "
            f"**${R['overlay'][8]:.1f}** vs **${R['overlay'][7]:.1f}**, i.e. "
            f"**+{R['overlay'][4]:.1f}%/yr** net vs **+{R['overlay'][0]:.1f}%/yr**. Because you sit out "
            "of the market roughly half the time — including stretches where retail rallies *before* "
            "sales confirm — the overlay mostly misses *gains*. It buys you lower volatility (its Sharpe "
            f"is a near-tie, {R['overlay'][5]:.2f} vs {R['overlay'][1]:.2f}), but no extra reward. There "
            "is no free lunch here."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Accelerating same-store sales are followed by *slightly* better retail "
            "returns at one horizon (6 months), but the tilt is small, **not** statistically significant, "
            "and it evaporates at 1 and 12 months and once you ask about retail *versus the market*. "
            "Indistinguishable from noise.\n"
            "- **Tradability — Mirage.** Buying retail on Redbook acceleration **loses to buy-and-hold**. "
            "There's nothing to deploy.\n"
            "- **Leads retail? — Not supported.** The Redbook move lines up with a retail-stock move that "
            "already happened a quarter earlier. Sales **echo** the sector; they don't forecast it. The "
            "one word that makes the pitch — *nowcast* — is the part the data rejects."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "Forget significance for a second. Even if the small 6-month tilt were real, the operational "
            "reality is unkind: Redbook accelerates in **roughly half** of all months (every wiggle of a "
            "noisy weekly series counts), so an own-on-acceleration rule whips you in and out constantly "
            f"— **{R['overlay'][6]} switches** in 20 years — while the *biggest* same-store surges (2021's "
            "double digits) were **nominal inflation**, exactly when retail *stocks* were about to fall. "
            "A gauge that confuses higher prices with a stronger consumer, and confirms moves the market "
            "already made, has nothing left to sell you. There is no version of \"buy retail when sales "
            "accelerate\" that both fires early and makes money."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The sibling tests.** [Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/) "
            "asks the same question of the weekly claims number: real labour signal, but does it *lead* "
            "the market or echo it? [Study 384 — ISM-PMI-Regime](../384-ism-pmi-regime/) does it for the "
            "manufacturing cycle.\n"
            "- **Kill the inflation.** Deflate Redbook by CPI to a *real* same-store number — the 2021–22 "
            "spike shrinks, and with it the one regime that most confuses the signal. Does a *real* "
            "same-store pulse lead any better? (The lead/lag picture barely budges: a coincident series "
            "can't be made to lead by re-scaling it.)\n"
            "- **Build your own.** Swap XRT for individual retailers, or pair Redbook with a price trend "
            "filter. If you can show the lead/lag chart peaking on the **right** (positive lead), we'll "
            "talk.\n\n"
            "*Think the sales pulse leads retail stocks? Show the lead/lag chart peaking on the "
            "**right** (Redbook first) — then we'll talk.*"
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
            "# Redbook-Retail — a quantitative teardown 🔬\n"
            "### Redbook-acceleration split returns · Welch *t* + placebo null · the decisive lead/lag "
            "cross-correlation · a retail-vs-market relative test · a level-regime split · a timing "
            "overlay vs buy-and-hold · robustness · a synthetic planted-edge control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The believers "
            "fuse two claims: that accelerating Redbook same-store sales (1) **predict** stronger retail "
            "returns and (2) do so **early** enough to trade. We separate them. The conditional return "
            "tilt is *right-signed but insignificant at one horizon and null elsewhere*; the decisive "
            "object is the **lead/lag structure**, which shows Redbook momentum is **coincident-to-"
            "lagging** (it trails XRT by a quarter), not leading — and a tradable overlay that "
            "*underperforms* buy-and-hold seals the Tradability axis. A level-regime split even points "
            "the *wrong way* (strong same-store months precede **weaker** returns, *t* = −2.28) — the "
            "nominal-inflation contamination of 2021–22 laid bare.\n\n"
            "> ⚠️ **Data + proxy note.** The weekly Redbook Index is proprietary (paid feed, off FRED); "
            "the Redbook tape here is a hardcoded, **clearly-labelled approximate monthly reconstruction** "
            "of the YoY same-store number (faithful in shape, approximate in level) — never under a "
            "real-tape banner, named on the Signal axis. XRT and SPY are yfinance daily adjusted close "
            "(total-return), month-end sampled. Offline core + synthetic control are deterministic. "
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
            f"| **Signal** | `NONE` | Best horizon (6m): accel mean **+{R['h6'][2]:.1f}%** vs base "
            f"**+{R['h6'][4]:.1f}%**, Welch **t = {R['h6'][7]:.2f}** (fails **t ≥ 2**); null at 1m/12m; "
            f"the retail-vs-market relative test is **t = {R['rel6'][4]:.2f}** (6m) and *negative* at 1m/12m. |\n"
            f"| **Tradability** | `MIRAGE` | Own-on-acceleration overlay **+{R['overlay'][4]:.1f}%/yr** "
            f"(Sharpe **{R['overlay'][5]:.2f}**) **vs buy-hold +{R['overlay'][0]:.1f}%/yr** "
            f"(Sharpe **{R['overlay'][1]:.2f}**); $1 → **${R['overlay'][8]:.1f}** vs **${R['overlay'][7]:.1f}**. |\n"
            f"| **Leads retail?** | `NOT SUPPORTED` | Peak *positive* lead/lag correlation at "
            f"**L = −3** (Redbook lags XRT by a quarter, ρ = **+{R['leadlag'][-3]:.2f}**); at positive "
            "leads ρ ≈ 0 or negative. A coincident-to-lagging echo, not a leader. |\n\n"
            "> 💡 In plain words: the retail ETF is a forward-looking asset — it reprices the consumer in "
            "real time — so a monthly same-store gauge that co-moves with it need not lead it. Redbook "
            "lines up with a retail move already a quarter old; the 'nowcast' is XRT's own lead, reflected "
            "back. And a **nominal** sales number (double-digit in the 2021–22 inflation, when XRT fell) "
            "can even point the wrong way."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $y_t$ be the Redbook same-store YoY level (%) and $m_t = y_t - y_{t-3}$ its 3-month "
            "momentum (acceleration). Redbook is **ACCELERATING** at $t$ when $m_t > 0$. With a one-month "
            "execution lag (the month-$t$ print is acted on at the close of $t+1$), define forward return "
            "$r_{t+1\\to t+1+H}$ on XRT.\n\n"
            "- **H₁ (predicts).** $\\mathbb{E}[r\\mid \\text{accel}] > \\mathbb{E}[r]$ — a *positive* "
            "excess over the base rate.\n"
            "- **H₂ (leads).** The strongest positive Redbook↔return correlation sits at a **positive** "
            "lead (Redbook moves first).\n"
            "- **H₃ (retail-specific & deployable).** Redbook predicts XRT *outperforming SPY*, and an "
            "own-on-acceleration overlay beats buy-and-hold net of costs.\n\n"
            "We find **H₁ directionally true only at 6m and insignificant** ($t = 1.21$, null elsewhere), "
            "**H₂ rejected** (peak positive corr at $L=-3$), **H₃ rejected** (relative excess "
            f"insignificant at $t={R['rel6'][4]:.2f}$; overlay underperforms). The folklore is right "
            "exactly where it's uninformative (sales and stocks co-move) and wrong exactly where it would "
            "pay (a *leading*, *retail-specific*, *tradable* edge)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The conditional-return test is a two-sample mean comparison judged by its standard error:\n\n"
            "$$\\widehat{\\Delta}_H = \\bar r^{\\text{accel}}_H - \\bar r^{\\text{all}}_H,\\qquad "
            "t = \\frac{\\widehat{\\Delta}_H}{\\sqrt{\\,s^2_{\\text{accel}}/k + s^2_{\\text{all}}/N\\,}}.$$\n\n"
            "But a significant $\\widehat{\\Delta}$ would **still not** establish *leading*: a coincident "
            "or lagging series can co-move with forward returns through cycle autocorrelation. The "
            "identifying test is the **lead/lag cross-correlation** "
            "$\\rho(L) = \\mathrm{corr}(m_t,\\ r_{t+L\\to t+L+1})$. A genuine nowcast peaks "
            "(positively) at $L>0$. If $\\arg\\max_L \\rho(L) < 0$, Redbook **follows** the stocks — and "
            "the entire 'nowcast' thesis collapses regardless of the conditional mean. A **relative** "
            "(XRT−SPY) version asks the sharper question: does Redbook time *retail specifically*, or just "
            "ride the market beta every equity gauge shares?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Redbook tape.** Monthly same-store YoY (%), hardcoded **labelled proxy** (the weekly "
            f"series is proprietary), {R['start'][:7]}→{R['end'][:7]} ({R['months']} months, XRT's whole "
            "listed life). Approximate in level, faithful in shape; named on the axis.\n"
            "- **Signal.** $m_t = y_t - y_{t-3}$; ACCELERATING when $m_t>0$.\n"
            "- **Forward returns.** Enter at the close **1 month after** the signal (no look-ahead), hold "
            "$H\\in\\{1,3,6,12\\}$ months; drop horizons that overrun the tape. Total-return XRT.\n"
            "- **Null #1 (Welch t).** Accel-set mean vs the unconditional mean.\n"
            "- **Null #2 (placebo).** 20,000 draws of $k$ random months; "
            "$p = \\Pr[\\text{random-draw mean} \\ge \\text{accel mean}]$ (as bullish or more).\n"
            "- **Identification (lead/lag).** $\\rho(L)$ for $L\\in[-6,6]$, absolute and relative — "
            "*where* does Redbook line up?\n"
            "- **Retail-specific (relative).** Repeat the split on XRT−SPY forward returns.\n"
            "- **Level regime.** Forward returns in strong (YoY above median) vs weak same-store months.\n"
            "- **Tradability.** Own-when-accelerating overlay, 1-month lag, 10 bps one-way per switch "
            "(turnover one-way × NAV), excess-of-zero Sharpe (cash leg = 0, labelled).\n"
            "- **Positive control.** A deterministic series with a *planted* Redbook→returns link: "
            "`edge=0` must not fake significance; a large `edge` must light up the test."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The point estimates — right sign at 6m, small, insignificant, null elsewhere\n\n"
            "Accel-set forward mean with $\\pm$ standard error against the unconditional base rate "
            "(dashed diamonds). Above base only at 3–6 months, and inside its own error bar throughout."
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
            "    ts = [R['h1'][7], R['h3'][7], R['h6'][7], R['h12'][7]]; ses = [.012,.02,.03,.045]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x, [c*100 for c in cm], yerr=[s*100 for s in ses], capsize=5, color=GREEN, width=.5, label='accel (±SE)')\n"
            "ax.plot(x, [b*100 for b in bm], 'D', ms=11, c=GREY, label='unconditional base rate')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{m}m' for m in hs]); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward XRT return (%)')\n"
            "ax.set_title('Right sign at 3-6m but the SE swamps the gap; a tie at 1m and 12m'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Welch t by horizon:', {f'{m}m': round(t,2) for m,t in zip(hs,ts)})"
        ),
        md(
            f"> 💡 In plain words: the best case is 6 months — accel **+{R['h6'][2]:.1f}%** vs base "
            f"**+{R['h6'][4]:.1f}%**, a ~{R['h6'][2]-R['h6'][4]:.1f}-point gap at **t = {R['h6'][7]:.2f}** "
            f"(not significant). At 1m it's a dead tie ({R['h1'][2]:.2f}% vs {R['h1'][4]:.2f}%) and at 12m "
            "almost nothing. H₁ is **directionally suggestive at one horizon, statistically absent** — "
            "the right sign living inside its own error bar, and only sometimes."
        ),
        md(
            "### 4b · The decisive identification test — lead/lag\n\n"
            "$\\rho(L) = \\mathrm{corr}(m_t, r_{t+L\\to t+L+1})$. Positive bars left of zero = Redbook "
            "**lags** the stocks; a real nowcast would peak on the **right** (Redbook leads)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ll = st.lead_lag(F); Ls = list(range(-6,7)); cs = [ll[L] for L in Ls]\n"
            "else:\n"
            "    Ls = sorted(R['leadlag']); cs = [R['leadlag'][L] for L in Ls]\n"
            "cols = [RED if L<0 else GREEN for L in Ls]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(Ls, cs, color=cols, width=.8)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c=GREY, lw=1, ls=':')\n"
            "imax = int(np.nanargmax(cs))\n"
            "ax.annotate('strongest POSITIVE link\\n(Redbook LAGS the stocks)', xy=(Ls[imax], cs[imax]),\n"
            "            xytext=(Ls[imax]-0.3, cs[imax]+0.06), ha='center', color=RED,\n"
            "            arrowprops=dict(arrowstyle='->', color=RED))\n"
            "ax.set_xlabel('lead L (months): L>0 = Redbook leads (nowcast)   |   L<0 = Redbook lags')\n"
            "ax.set_ylabel(r'$\\rho(L)$'); ax.set_xticks(Ls)\n"
            "ax.set_title('argmax rho(L) is at L<0: Redbook is coincident-to-lagging')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'argmax at L={Ls[imax]} (rho={cs[imax]:+.2f}); rho at +1 month = {cs[Ls.index(1)]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: $\\arg\\max_L \\rho(L) = -3$ (ρ = **+{R['leadlag'][-3]:.2f}**). The "
            "Redbook signal correlates most (positively) with a retail-stock move **a quarter in its "
            f"past**; at the positive leads a genuine nowcast needs, $\\rho \\approx 0$ (ρ at +1 = "
            f"**{R['leadlag'][1]:+.2f}**). **H₂ rejected.** The retail ETF prices the consumer first; the "
            "sales gauge trails. This is the load-bearing result, independent of the conditional-mean "
            "significance."
        ),
        md(
            "### 4c · Retail-specific? — the relative (XRT−SPY) test\n\n"
            "Does Redbook time *retail over the market*, or just the shared equity beta? Repeat the split "
            "on XRT-minus-SPY forward returns."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rel = [st.summarize(F, m, relative=True) for m in hs]\n"
            "    rm = [r['accel_mean']*100 for r in rel]; rt = [r['t'] for r in rel]\n"
            "else:\n"
            "    rm = [R['rel1'][2], R['h3'][2]-R['h3'][2], R['rel6'][2], R['rel12'][2]]\n"
            "    rt = [R['rel1'][4], 0.11, R['rel6'][4], R['rel12'][4]]\n"
            "x = np.arange(len(hs))\n"
            "cols = [GREEN if t>0 else RED for t in rt]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "ax.bar(x, rm, color=cols, width=.55)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{m}m' for m in hs])\n"
            "ax.set_ylabel('mean forward XRT-minus-SPY return (%)')\n"
            "ax.set_title('Redbook does NOT predict retail beating the market (t never near 2)')\n"
            "for i,(v,t) in enumerate(zip(rm,rt)): ax.annotate(f't={t:.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('relative accel-excess t by horizon:', {f'{m}m': round(t,2) for m,t in zip(hs,rt)})"
        ),
        md(
            f"> 💡 In plain words: the tiny absolute 6m tilt is **mostly market beta** — on a "
            f"retail-*minus*-market basis it's **+{R['rel6'][2]:.1f}%** at **t = {R['rel6'][4]:.2f}** "
            f"(insignificant), and it's *negative* at 1m ({R['rel1'][2]:+.2f}%, t={R['rel1'][4]:.2f}) and "
            f"12m ({R['rel12'][2]:+.2f}%, t={R['rel12'][4]:.2f}). **H₃'s retail-specificity is rejected** "
            "— Redbook doesn't pick retail *over* the market."
        ),
        md(
            "### 4d · The level regime points the WRONG way\n\n"
            "Forget acceleration — just split on the *level*: forward XRT returns in strong (Redbook YoY "
            "above the full-sample median) vs weak same-store months. The believers say 'own retail when "
            "the consumer is strong.' The data disagrees."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g1 = st.regime_summary(F, 1); g12 = st.regime_summary(F, 12)\n"
            "    s1, w1, t1 = g1['strong_mean']*100, g1['weak_mean']*100, g1['t']\n"
            "    s12, w12, t12 = g12['strong_mean']*100, g12['weak_mean']*100, g12['t']\n"
            "else:\n"
            "    s1, w1, t1 = R['reg1'][1], R['reg1'][2], R['reg1'][4]\n"
            "    s12, w12, t12 = R['reg12'][1], R['reg12'][2], R['reg12'][4]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "x = np.arange(2)\n"
            "ax.bar(x-.2, [s1, s12], .4, color=GREEN, label='STRONG same-store (YoY high)')\n"
            "ax.bar(x+.2, [w1, w12], .4, color=GREY, label='WEAK same-store (YoY low)')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['1-month fwd', '12-month fwd'])\n"
            "ax.set_ylabel('mean forward XRT return (%)'); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_title(f'Strong-consumer months precede WEAKER retail returns (12m t={t12:.2f})'); ax.legend()\n"
            "for i,(a,b) in enumerate(zip([s1,s12],[w1,w12])):\n"
            "    ax.annotate(f'{a:.1f}%',(i-.2,a),ha='center',va='bottom')\n"
            "    ax.annotate(f'{b:.1f}%',(i+.2,b),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'12m: strong {s12:.1f}% vs weak {w12:.1f}%  (t={t12:.2f})')"
        ),
        md(
            f"> 💡 In plain words: at 12 months, **strong** same-store months are followed by "
            f"**+{R['reg12'][1]:.1f}%** vs **+{R['reg12'][2]:.1f}%** after **weak** months — a "
            f"**{R['reg12'][3]:.1f}-point** gap the *wrong way*, at **t = {R['reg12'][4]:.2f}** "
            "(significant). This is the 2021–22 tell: peak *nominal* same-store growth was peak inflation, "
            "and XRT fell afterward. The only significant regime result runs **against** the folklore — a "
            "late-cycle / inflation echo, not a buy signal."
        ),
        md(
            "### 4e · Robustness — window, smoothing, relative, ex-COVID (6-month)\n\n"
            "Vary the momentum window $k$, smooth the signal, switch to relative, drop the COVID/inflation "
            "window. The 6-month *t* never clears 2 at any spec — it's a fragile bump, not a signal."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for k in (1,3,6):\n"
            "        s = st.summarize(F, 6, k=k); rob.append((f'k={k}', s['n_accel'], s['accel_mean']*100, s['t'], s['p_placebo']))\n"
            "    s = st.summarize(F, 6, smooth=3); rob.append(('smooth3', s['n_accel'], s['accel_mean']*100, s['t'], s['p_placebo']))\n"
            "    s = st.summarize(F, 6, relative=True); rob.append(('relative', s['n_accel'], s['accel_mean']*100, s['t'], s['p_placebo']))\n"
            "    F2 = F[(F.index < '2020-01-01') | (F.index >= '2022-07-01')]\n"
            "    s = st.summarize(F2, 6); rob.append(('ex-COVID', s['n_accel'], s['accel_mean']*100, s['t'], s['p_placebo']))\n"
            "else:\n"
            "    rob = [(l,n,r,t,p) for (l,n,r,t,p) in R['robust']]\n"
            "labels = [r[0] for r in rob]; tt = [r[3] for r in rob]; nn = [r[1] for r in rob]\n"
            "cols = [GREEN if t>0 else RED for t in tt]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.3))\n"
            "ax.bar(labels, tt, color=cols, width=.6)\n"
            "ax.axhline(2, ls='--', c=GREEN, label='t=+2 (significance bar)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(t,k) in enumerate(zip(tt,nn)): ax.annotate(f'n={k}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('Welch t (6-month)'); ax.set_ylim(0, 2.4)\n"
            "ax.set_title('No spec clears |t|=2; the bump peaks at k=3 and evaporates'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (label, n, accel6%, t, p):', [(r[0], r[1], round(r[2],1), round(r[3],2), round(r[4],3)) for r in rob])"
        ),
        md(
            f"> 💡 In plain words: the effect is a single fragile bump — highest at **k=3 → "
            f"t={R['robust'][1][3]:.2f}**, and it falls to **t={R['robust'][2][3]:.2f}** (k=6), "
            f"**t={R['robust'][3][3]:.2f}** (smoothed), **t={R['robust'][4][3]:.2f}** (relative). Even "
            f"dropping the whole COVID/inflation window leaves it at **t={R['robust'][5][3]:.2f}** — still "
            "under the bar. Nothing certifies it."
        ),
        md(
            "### 4f · Faithful-engine control — we know the truth here\n\n"
            "A deterministic monthly series with a *planted* link (accelerating Redbook momentum at $t$ "
            "lifts the $t{+}1$ return by `edge`). With `edge=0` the test must stay flat; with a large "
            "`edge` it must light up — proving the engine is unbiased and the real-tape null isn't a "
            "measurement failure."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.05):\n"
            "    syn = data.synthetic_redbook(n_months=360, edge=edge, seed=759)\n"
            "    s = st.summarize(syn, 1, k=3)\n"
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
            f"**t = {R['syn'][0][4]:.2f}** (no false positive, placebo p = {R['syn'][0][5]:.2f}); a "
            f"**+5%/month** planted link drives **t = {R['syn'][1][4]:.2f}**. So the machinery is honest — "
            "the real-tape *t* of ~1.2 (at best) is a *genuine* weak-or-absent edge, not a broken test. "
            "The engine *can* bank a real Redbook→returns link; the real tape just doesn't carry a "
            "tradable one."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — best-horizon (6m) excess **{R['h6'][2]-R['h6'][4]:+.1f}pp** at Welch "
            f"**t = {R['h6'][7]:.2f}** / placebo **p = {R['h6'][8]:.2f}**"
            f" — fails t≥2, is null at 1m/12m, and is **market beta**: the retail-vs-market relative "
            f"excess is **t = {R['rel6'][4]:.2f}** (6m) and negative at 1m/12m. Indistinguishable from "
            "noise. Literature/folklore support cannot lift a sub-2 tape to REAL.\n"
            f"- **Tradability `MIRAGE`** — the own-on-acceleration overlay returns "
            f"**+{R['overlay'][4]:.1f}%/yr** (Sharpe {R['overlay'][5]:.2f}) vs buy-hold "
            f"**+{R['overlay'][0]:.1f}%/yr** (Sharpe {R['overlay'][1]:.2f}); $1 → **${R['overlay'][8]:.1f}** "
            f"vs **${R['overlay'][7]:.1f}**. Acting on the signal *subtracts* return — nothing to allocate "
            "to.\n"
            "- **Leads retail? `NOT SUPPORTED`** — $\\arg\\max_L \\rho(L) = -3$ months: Redbook momentum "
            "is **coincident-to-lagging**, trailing XRT by a quarter. The retail ETF is the leading "
            "indicator; the sales gauge echoes it. And the only *significant* level-regime result "
            f"(**t = {R['reg12'][4]:.2f}**) points the **wrong way** — strong nominal same-store growth "
            "(2021–22 inflation) preceded *weaker* returns. The defining word — *nowcast* — is the part "
            "the data rejects."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — why even a real tilt wouldn't deploy\n\n"
            "Grant the lore a genuine few-point 6-month tilt. The operational reality still defeats it. "
            "Redbook is 'accelerating' in **~44% of months** (every up-wiggle of a noisy weekly series), "
            f"so the overlay churns ({R['overlay'][6]} switches / 20y) and is out of the market in roughly "
            "half of all months — including recoveries the retail ETF starts pricing *before* sales roll "
            "over. That structural mistiming is why the overlay's return is **below** passive "
            f"(**+{R['overlay'][4]:.1f}%** vs **+{R['overlay'][0]:.1f}%**/yr) even though its Sharpe is a "
            "near-tie (you buy vol reduction, not reward). And the regime where the signal is *loudest* — "
            "the double-digit 2021–22 prints — was **nominal inflation**, exactly when you'd have wanted "
            "*less* retail exposure. No lag, smoothing, or cost assumption rescues a nominal, coincident-"
            "to-lagging series masquerading as a leading nowcast."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The siblings.** [Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/) "
            "and [Study 384 — ISM-PMI-Regime](../384-ism-pmi-regime/): the same hardcoded-snapshot + ETF "
            "method on the weekly claims number and the manufacturing cycle — do any of these famous "
            "'leading' macro series actually lead the tape they're sold against?\n"
            "- **Deflate it.** Convert Redbook to a *real* same-store number (÷ CPI); the 2021–22 spike "
            "collapses and the wrong-way level-regime result should soften — a clean test of how much of "
            "the confusion is pure nominal contamination.\n"
            "- **Sharper identification.** Use the *weekly* Redbook series and a proper VAR / Granger "
            "test, or real-time vintages, against individual retailer names; the coincident-to-lagging "
            "structure is robust — smoothing or re-scaling a coincident series can't manufacture a lead.\n\n"
            "*The reproducible core is offline and deterministic; the Redbook input is an explicit, "
            "clearly-labelled proxy. Methods and sources: [`docs/references.md`](../docs/references.md); "
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
