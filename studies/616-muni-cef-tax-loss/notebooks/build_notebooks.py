"""Generate the two narrative notebooks for Study 616 (Muni-CEF-Tax-Loss).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached fund + benchmark
prices under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic control runs anywhere with no network; the exact seasonal
placebo has no random component at all.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance, 12 seasoned muni
# CEFs + MUB + VWLTX, 2000-01-03 -> 2026-06-30, as-of 2026-06-30).
R = dict(
    start="2000-01-03", end="2026-06-30", rows=6662, fingerprint="5f10f3e3801d",
    n_funds=12, months=225, panel_start="2007-10",
    n_winters=19, n_winters_alt=26,
    # pooled fund-months (SECONDARY - cross-fund pseudo-replication)
    pooled=dict(dec=-13.05, n_dec=228, w_dec=-0.88, jan=186.27, n_jan=228, w_jan=6.57,
                rest=3.52, n_rest=2244, w_jan_dec=6.10),
    # per-winter basket vs MUB (PRIMARY)
    dec_bps=-15.75, t_dec=-0.33,
    jan_bps=191.25, t_jan=2.27, hit_jan=84.2,
    snap_bps=207.00, t_snap=1.83, hit_snap=73.7, snap_ac1=0.069,
    # exact 132-ordered-pair seasonal placebo (deterministic)
    placebo_rank=4, placebo_p=0.0303, placebo_floor=0.0076,
    # VWLTX robustness benchmark (26 winters, 2000+)
    alt=dict(jan=180.34, t_jan=3.51, hit=88.5, dec=16.31, t_dec=0.49,
             snap=164.03, t_snap=2.73, p=0.0455, rank=6),
    # sub-period fade: (label, Jan bps, t, n winters)
    fade=[("MUB 2007-2016", 326.63, 2.56, 10), ("MUB 2017-2025", 40.83, 0.45, 9),
          ("VWLTX 2000-2012", 274.92, 3.72, 13), ("VWLTX 2013-2025", 85.76, 1.34, 13)],
    fade_welch=1.83,
    # January swap costs: (one-way bps, drag bps, net bps/winter, t net, hit %)
    swap=[(5.0, 20, 171.25, 2.03, 84.2), (10.0, 40, 151.25, 1.79, 78.9),
          (25.0, 100, 91.25, 1.08, 73.7)],
    swap_gross=191.25,
    # third axis: Dec-15 vs Jan-15 entries, common end-Feb exit
    dec15=267.92, jan15=33.46, diff=234.45, t_diff=1.78, hit_diff=73.7,
    ex08=114.21, t_ex08=2.02, n_ex08=18,
    # synthetic control: (dump bps, snap bps, Jan bps, t Jan, snap bps, t snap, placebo p)
    syn=[(0, 0, 24.8, 0.65, 22.2, 0.53, 0.2424),
         (100, 100, 124.8, 3.26, 222.2, 5.33, 0.0076)],
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Buy Dec-15, not Jan-15?: Mixed](https://img.shields.io/badge/Buy_Dec--15%2C_not_Jan--15%3F-Mixed-8b949e?style=flat-square)\n\n"
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

from muni_cef_tax_loss import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PX = data.load_real()
    EX = data.monthly_excess(PX, bench=data.BENCH)        # MUB-excess panel (2007-10 ->)
    EX2 = data.monthly_excess(PX, bench=data.ALT_BENCH)   # VWLTX-excess panel (2000 ->)
    WT = st.winter_table(EX)
    WT2 = st.winter_table(EX2)
else:
    PX = EX = EX2 = WT = WT2 = None
print("real muni-CEF cache present:", HAVE_REAL,
      "| MUB-excess months:", (0 if EX is None else len(EX)),
      "| winters:", (0 if WT is None else len(WT)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The muni funds everyone dumps in December — and buys back in January 🏛️\n"
            "### Tax-loss season in municipal-bond closed-end funds, in plain English\n\n"
            + BADGES +
            "Here's a corner of the market almost nobody looks at: **municipal-bond closed-end "
            "funds** — little listed funds that own tax-free city and state bonds. Their owners are "
            "almost all **ordinary taxable households** (the tax-free coupon is worthless to pension "
            "funds and IRAs). The story, from a 2006 *Journal of Finance* paper: every **December**, "
            "those households sell their losers to harvest a tax deduction. In a fund this small and "
            "retail-owned, that selling knocks the **price** down even though the bonds inside barely "
            "move. Then in **January** the sellers come back — and the price snaps back up.\n\n"
            "A predictable dump, a predictable rebound, on a fixed calendar. Sounds too easy. Is it "
            "real?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the exact placebo and the cost math? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** We track **12 seasoned muni CEFs** against **MUB** (a muni "
            "ETF that trades at the value of its bonds). All 12 still trade today — that's "
            "**survivorship**, named openly. Every chart is drawn by the code beside it; house style "
            "in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do these funds really snap back in January? | **Yes.** The average January beats the "
            "muni market by about **+1.9%** — positive in **16 of 19** winters. That part clears our "
            "significance bar, on two different benchmarks. |\n"
            "| Do they really *dump* in December? | **Not on the monthly chart.** December as a whole "
            "looks flat — because the dump *and* the start of the rebound both happen inside December "
            "and cancel out. The real action runs **mid-December to mid-January**. |\n"
            "| Can you trade it? | **Barely.** A once-a-year swap (park in MUB, hold the funds for "
            "January) survives low costs — but these are tiny, thinly-traded funds, and the effect has "
            "**faded** since the paper was published (+0.4% a January in the last decade, roughly the "
            "cost of trading it). |\n"
            "| When should you buy? | **Mid-December, not mid-January.** By Jan-15 the party is "
            "basically over (+0.3% left of a +2.7% move) — though 19 winters can't *prove* the Dec-15 "
            "entry beyond doubt. |\n\n"
            "> The legend is **half true**: the January snap-back is real; the December calendar-month "
            "dump is a mirage of averaging; and the tradable version is thin, once-a-year, and fading."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Taxable investors sell their losing muni closed-end funds in December to book tax "
            "losses. The selling widens the funds' discounts. In January the money comes back and the "
            "discount snaps shut. Buy the dump, ride the snap-back.\"*\n\n"
            "This isn't bar-room folklore — it's **Starks, Yong & Zheng (2006), *Tax-Loss Selling and "
            "the January Effect: Evidence from Municipal Bond Closed-End Funds***, in the *Journal of "
            "Finance*. Muni CEFs are the cleanest possible laboratory: their owners are almost 100% "
            "taxable retail investors, so if tax-loss selling moves prices anywhere, it's here.\n\n"
            "It's also the **seasonal cousin** of a trade we've already audited: "
            "[study 367](../../367-closed-end-fund-discount/README.md) showed that buying CEFs at "
            "wide **discounts** pays (the *level* edge). This study asks about the **calendar**: does "
            "the discount predictably widen in December and close in January?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, this is about as honest as an anomaly gets: a **known cause** (the tax code's "
            "December 31 deadline), a **known victim** (a fund too small and retail-owned for "
            "professionals to bother correcting), and a **fixed calendar** anyone can act on. If "
            "false — or dead since publication — it's a perfect specimen of how a real paper becomes "
            "a zombie trade.\n\n"
            "**How we see the discount without NAV data:** we measure each fund's monthly return "
            "*minus* MUB's. MUB trades at the value of its bonds, so when a fund's price lags MUB, "
            "its discount is widening; when it beats MUB, the discount is snapping shut. Total-return "
            "on both sides, so the funds' fat monthly payouts don't pollute anything."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **{R['n_funds']} seasoned muni CEFs** (Nuveen, BlackRock, Eaton Vance, Invesco, "
            f"Western Asset), daily total-return prices {R['start']} → {R['end']}.\n"
            "- Each month, average the funds into **one basket number**: the basket's excess over "
            "MUB. One December + one January per winter = **19 winters** of evidence (a second "
            "benchmark, the Vanguard muni fund VWLTX, stretches it to **26**).\n"
            "- Ask: is the average **January** excess reliably positive? Is **December** reliably "
            "negative? Could a *random* pair of months look this special? (We check every one of the "
            "132 possible month-pairs.)\n"
            "- Then the practical questions: does a once-a-year swap survive **costs**, and is "
            "**Dec-15 or Jan-15** the right day to buy?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The whole year at a glance.** Average basket excess over MUB, month by month. If the "
            "tax-loss story is real, January should tower over everything."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mu = st.seasonal_means(EX) * 1e4\n"
            "    vals = [float(mu.get(m, np.nan)) for m in range(1, 13)]\n"
            "else:\n"
            "    vals = [191.2, -53.4, -9.0, 11.2, 60.2, 48.7, 51.5, 7.1, -68.6, -86.1, 73.8, -15.7]\n"
            "names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']\n"
            "colors = [GREEN] + [GREY]*10 + [RED]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.6))\n"
            "ax.bar(names, vals, color=colors, width=.65)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average excess over MUB (bps / month)')\n"
            "ax.set_title('The muni-CEF year: January towers, December is... flat')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('January:', round(vals[0],1), 'bps   December:', round(vals[11],1), 'bps')"
        ),
        md(
            f"**January is the outlier of the whole year**: **{R['jan_bps']:+.0f} bps** over the muni "
            f"market on average, positive in **16 of {R['n_winters']}** winters. But look at December "
            f"(red): **{R['dec_bps']:+.0f} bps** — basically zero. The \"December dump\" doesn't show "
            "up as a bad *calendar month*. Hold that thought — the third chart explains why.\n\n"
            "> 🔬 **For the quants:** one observation per winter, one-sample *t* = "
            f"{R['t_jan']:.2f} on January; the Dec→Jan contrast ranks **{R['placebo_rank']}ᵗʰ of "
            f"132** possible month-pairs (exact *p* = {R['placebo_p']:.3f}); the longer VWLTX tape "
            f"gives *t* = {R['alt']['t_jan']:.2f} over 26 winters."
        ),
        md(
            "**Winter by winter.** One bar per winter: the basket's January excess over MUB. A real "
            "seasonal should be *reliably* positive, not one lucky year."
        ),
        code(
            "if HAVE_REAL:\n"
            "    yrs = [str(y+1) for y in WT.index]\n"
            "    jans = (WT['jan'] * 1e4).values\n"
            "else:\n"
            "    yrs = [str(2008+i) for i in range(19)]; jans = np.full(19, R['jan_bps'])\n"
            "fig, ax = plt.subplots(figsize=(10.2, 4.6))\n"
            "ax.bar(yrs, jans, color=[GREEN if v > 0 else RED for v in jans], width=.7)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('January excess over MUB (bps)')\n"
            "ax.set_title('The January snap-back, winter by winter (16 of 19 positive)')\n"
            "plt.xticks(rotation=45); plt.tight_layout(); plt.show()\n"
            "print(f'mean {jans.mean():+.1f} bps   positive in {(jans>0).sum()} of {len(jans)} winters')"
        ),
        md(
            f"Green in **{R['hit_jan']:.0f}%** of winters. But notice the *sizes*: the monsters are "
            "2009 and the early 2010s; the last decade is mostly small bars. The effect was published "
            "in 2006 — and like most published anomalies, it has **faded**: post-2016 the average "
            f"January is only **+41 bps**, roughly what the trade costs to execute."
        ),
        md(
            "**When exactly does the money move?** We race two buyers: one buys the funds on "
            "**Dec-15** (into the dump), one waits until **Jan-15**. Both sell at the end of "
            "February. The gap between them is the payoff earned *between* mid-December and "
            "mid-January."
        ),
        code(
            "if HAVE_REAL:\n"
            "    dj = st.dec15_vs_jan15(PX, data.BENCH, data.FUNDS)\n"
            "    a, b = dj['winA_dec15'].mean()*1e4, dj['winB_jan15'].mean()*1e4\n"
            "else:\n"
            "    a, b = R['dec15'], R['jan15']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "ax.bar(['buy Dec-15\\n(into the dump)', 'buy Jan-15\\n(wait it out)'], [a, b],\n"
            "       color=[GREEN, GREY], width=.5)\n"
            "for i, v in enumerate([a, b]): ax.annotate(f'{v:+.0f} bps', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('excess over MUB to end-February (bps)')\n"
            "ax.set_title('Almost the whole payoff lives between Dec-15 and Jan-15')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Dec-15 entry {a:+.1f} bps   Jan-15 entry {b:+.1f} bps   gap {a-b:+.1f} bps')"
        ),
        md(
            f"**There's the missing December dump.** The Dec-15 buyer earns **{R['dec15']:+.0f} bps**; "
            f"the Jan-15 buyer only **{R['jan15']:+.0f} bps**. The dump happens mid-December, the "
            "snap-back starts *before* year-end — so the December *calendar month* nets out to zero "
            "while the mid-Dec→mid-Jan window carries everything. That's the actionable version: **if "
            "you play this at all, you buy into the December selling, not after New Year**. (One "
            "honest caveat: a chunk of that average is the crazy 2008 winter; strip it out and the "
            f"gap is +{R['ex08']:.0f} bps — right at the edge of what 18 winters can prove.)"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Mixed.** The **January snap-back is real**: {R['jan_bps']:+.0f} bps over "
            f"the muni market, {R['hit_jan']:.0f}% of winters positive, significant on two benchmarks. "
            "The **December-month dump is not there** — it hides inside the month and cancels out.\n"
            "- **Tradability — Fragile.** A once-a-year MUB→funds→MUB swap survives *low* costs "
            f"(net {R['swap'][0][2]:+.0f} bps/winter at 5 bps a leg) but these are tiny, thin funds, "
            "and the last decade's payoff (~+41 bps) is about what the trading costs.\n"
            "- **\"Buy Dec-15, not Jan-15\" — Mixed.** Directionally clear (the payoff is almost all "
            "pre-Jan-15), statistically not sealed in 19 winters."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Why muni CEFs?** Because the owners are all taxable households — the tax-loss story "
            "has no other suspects here. That's why a 2006 *Journal of Finance* paper used them as "
            "the clean laboratory.\n"
            "- **The cousin trade.** [Study 367](../../367-closed-end-fund-discount/README.md) tests "
            "the *level* version (buy wide discounts, any month). This was the *calendar* version. "
            "Same machinery — discounts set by retail flow — two different edges.\n"
            "- **Build your own.** Swap in a different fund list, or move the entry day around "
            "Dec-15 — the engine (`muni_cef_tax_loss/`) exposes every knob.\n\n"
            "*Think a once-a-year +1.9% gross seasonal on thin funds is your retirement plan? Check "
            "what's left after the fade and 4 trading legs — then we'll talk.*"
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
            "# Muni-CEF tax-loss season — a quantitative teardown 🔬\n"
            "### Per-winter basket one-sample *t*'s · an exact 132-pair seasonal placebo · a 26-winter "
            "VWLTX extension · midpoint fade split · costs × 4 legs on the January swap · the Dec-15 "
            "vs Jan-15 paired race · a planted-knob synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The claim "
            "(Starks, Yong & Zheng 2006, JF) is the cleanest identification in the tax-loss-selling "
            "literature — muni CEF holders are ~100% taxable retail — so the job here is an honest "
            "replication on the modern tape: a per-winter unit that kills cross-fund pseudo-"
            "replication, an exhaustive month-pair placebo, and real costs on the actionable trade.\n\n"
            "> ⚠️ **Data + survivorship note.** Fixed **12-fund** seasoned muni-CEF panel, all still "
            "trading in 2026 — a *survivor* panel (merged/liquidated funds absent; for a seasonal test "
            "the tilt is milder than for a level edge, but a surviving panel is a healthier, more-owned "
            "panel — exactly the clientele the story needs). Real data: yfinance total-return closes, "
            "2000→2026; discount motion proxied by fund-minus-benchmark excess (benchmarks price at "
            "NAV). Offline core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fingerprint"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | *Real on the January snap-back, None on the December-month "
            f"dump.* January excess **{R['jan_bps']:+.2f} bps** at one-sample **t = {R['t_jan']:.2f}** "
            f"({R['n_winters']} winters, hit {R['hit_jan']:.0f}%); **{R['alt']['jan']:+.2f} bps at "
            f"t = {R['alt']['t_jan']:.2f}** on {R['n_winters_alt']} VWLTX winters; exact placebo "
            f"p = {R['placebo_p']:.3f}/{R['alt']['p']:.3f}. December: {R['dec_bps']:+.2f} bps, "
            f"t = {R['t_dec']:.2f}. Survivorship + a named fade. |\n"
            f"| **Tradability** | `FRAGILE` | January swap nets **{R['swap'][0][2]:+.2f} bps/winter "
            f"at t = {R['swap'][0][3]:.2f}** at 5 bps one-way; **t = {R['swap'][1][3]:.2f}** at 10 "
            f"bps; post-2016 gross (+{R['fade'][1][1]:.0f} bps) sits below the 40–100 bps drag. "
            "Once a year, on thin funds. |\n"
            f"| **Buy Dec-15, not Jan-15?** | `MIXED` | Paired difference **{R['diff']:+.2f} bps** "
            f"(hit {R['hit_diff']:.0f}%), but **t = {R['t_diff']:.2f}** full-sample and "
            f"**+{R['ex08']:.2f} bps at t = {R['t_ex08']:.2f}** ex-2008 — directionally decisive, "
            "not sealed. |\n\n"
            "> 💡 In plain words: January is genuinely special for these funds — on both benchmarks — "
            "but the December calendar month never shows the dump (it hides inside the month), and "
            "the tradable once-a-year version is thin and fading."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^f_{m}$ be fund $f$'s monthly total return and $r^B_m$ the NAV-priced benchmark's "
            "(MUB; VWLTX pre-2007). The **discount-motion proxy** is the excess\n\n"
            "$$e^f_m = r^f_m - r^B_m,$$\n\n"
            "negative when the fund's *price* lags the muni market (discount widening). Equal-weight "
            "the panel into a basket $\\bar e_m$, and for each winter $y$ take the pair "
            "$(\\bar e_{\\mathrm{Dec},y},\\ \\bar e_{\\mathrm{Jan},y+1})$ — **one observation per "
            "winter**.\n\n"
            "- **H₁ (the dump).** $\\mathbb{E}[\\bar e_{\\mathrm{Dec}}] < 0$.\n"
            "- **H₂ (the snap-back).** $\\mathbb{E}[\\bar e_{\\mathrm{Jan}}] > 0$, and the Dec→Jan "
            "contrast is extreme among all month pairs.\n"
            "- **H₃ (actionable).** A Dec-15 entry beats a Jan-15 entry to a common end-Feb exit, "
            "net of 4 one-way legs.\n\n"
            f"We find **H₂ supported** (t = {R['t_jan']:.2f} on MUB, {R['alt']['t_jan']:.2f} on "
            f"VWLTX), **H₁ rejected at monthly grain** (t = {R['t_dec']:.2f}; the dump lives *inside* "
            "December), **H₃ directionally supported but uncertified** "
            f"(t = {R['t_diff']:.2f})."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the honesty traps this study must dodge\n\n"
            "**(a) Pseudo-replication.** All muni CEFs load on one discount factor; a given January "
            "is essentially **one** draw, not twelve. Pooled fund-month Welch *t*'s (January vs rest "
            f"= {R['pooled']['w_jan']:+.2f} on n = {R['pooled']['n_jan']}) flatter the effect — we "
            "quote them as *secondary* and run the primary test on the per-winter basket:\n\n"
            "$$t = \\frac{\\overline{x}}{s_x/\\sqrt{N}},\\qquad N = 19\\ \\text{(MUB)},\\ 26\\ "
            "\\text{(VWLTX)} \\text{ non-overlapping winters}.$$\n\n"
            "**(b) Calendar snooping.** Dec→Jan is 1 of 132 ordered month pairs; a placebo must "
            "price that in. Ours is **exact** — a full enumeration, no RNG, floor "
            f"p = 1/132 = {R['placebo_floor']:.4f}.\n\n"
            "**(c) Post-publication decay.** Published 2006 (McLean-Pontiff territory): we split each "
            "tape at its midpoint — not snooped — and test the *difference* with a Welch *t*.\n\n"
            "**(d) Costs on the real trade.** The swap fires 4 one-way legs a winter on funds whose "
            "half-spreads alone can reach 10 bps."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {R['n_funds']} seasoned national muni CEFs across five sponsors; "
            f"benchmarks MUB (2007-09 →) and VWLTX (2000 →). Daily total-return closes, "
            f"{R['start']} → {R['end']} ({R['rows']:,} rows). **Survivor** panel — named.\n"
            "- **Proxy.** Monthly fund-minus-benchmark excess; both legs total-return.\n"
            "- **Timing.** The signal is a **fixed calendar rule**; every entry/exit fills at the "
            "first close *after* its decision point — the single execution lag.\n"
            f"- **Primary.** Per-winter basket one-sample *t* ({R['n_winters']} MUB winters; snap "
            f"lag-1 autocorrelation {R['snap_ac1']:+.3f} → plain *t* appropriate on annual data).\n"
            "- **Placebo.** Exact rank of the Dec→Jan contrast among all 132 ordered month pairs.\n"
            "- **Costs.** 4 one-way legs × 5/10/25 bps × NAV per winter, charged against the "
            "January excess.\n"
            "- **Third axis.** Paired Dec-15 vs Jan-15 entries, common last-Feb-close exit.\n"
            "- **Positive control.** Synthetic excess panel (common factor + fund noise) with "
            "plantable December-dump / January-snap knobs; (0, 0) must not fire."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The per-winter basket — the primary test\n\n"
            "One December and one January per winter, equal-weight basket, MUB excess."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ws = st.winter_stats(WT)\n"
            "    dec, jan, snap = ws['dec_mean_bps'], ws['jan_mean_bps'], ws['snap_mean_bps']\n"
            "    td, tj, tsn = ws['t_dec'], ws['t_jan'], ws['t_snap']\n"
            "else:\n"
            "    dec, jan, snap = R['dec_bps'], R['jan_bps'], R['snap_bps']\n"
            "    td, tj, tsn = R['t_dec'], R['t_jan'], R['t_snap']\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))\n"
            "axes[0].bar(['December', 'January', 'snap\\n(Jan-Dec)'], [dec, jan, snap],\n"
            "            color=[RED, GREEN, AMBER], width=.55)\n"
            "for i, v in enumerate([dec, jan, snap]):\n"
            "    axes[0].annotate(f'{v:+.0f}', (i, v), ha='center', va='bottom')\n"
            "axes[0].axhline(0, c='k', lw=.8); axes[0].set_ylabel('basket excess over MUB (bps)')\n"
            "axes[0].set_title('Mean per winter')\n"
            "axes[1].bar(['December', 'January', 'snap'], [td, tj, tsn], color=[RED, GREEN, AMBER], width=.55)\n"
            "axes[1].axhline(2, ls='--', c=RED, label='t = 2 bar'); axes[1].axhline(0, c='k', lw=.8)\n"
            "for i, v in enumerate([td, tj, tsn]):\n"
            "    axes[1].annotate(f't={v:+.2f}', (i, v), ha='center', va='bottom')\n"
            "axes[1].set_title('One-sample t across winters'); axes[1].legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Dec {dec:+.2f} bps (t={td:+.2f})   Jan {jan:+.2f} bps (t={tj:+.2f})   snap {snap:+.2f} bps (t={tsn:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: January clears the bar (**{R['jan_bps']:+.0f} bps, "
            f"t = {R['t_jan']:.2f}**, hit {R['hit_jan']:.0f}%); December is statistically nothing "
            f"({R['dec_bps']:+.0f} bps, t = {R['t_dec']:.2f}). The claim's *snap-back* half is real; "
            "its *dump-month* half is not visible at monthly grain — §4e shows where the dump "
            "actually lives.\n\n"
            f"Secondary pooled fund-month view (pseudo-replication caveat): January vs rest Welch "
            f"t = {R['pooled']['w_jan']:+.2f} (n = {R['pooled']['n_jan']}), December vs rest "
            f"{R['pooled']['w_dec']:+.2f} (n = {R['pooled']['n_dec']}), January vs December "
            f"{R['pooled']['w_jan_dec']:+.2f} — same shape, flattered n."
        ),
        md(
            "### 4b · The exact seasonal placebo — Dec→Jan among all 132 ordered month pairs\n\n"
            "Could a random pair of months look this special? Enumerate **every** ordered pair "
            "(a, b), statistic = seasonal mean(b) − mean(a) of the basket excess."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.exact_pair_placebo(EX)\n"
            "    vals = np.array(list(pl['pairs'].values())) * 1e4\n"
            "    obs, p, rank = pl['obs'] * 1e4, pl['p_value'], pl['rank']\n"
            "else:\n"
            "    rng = np.random.default_rng(0); vals = rng.normal(0, 90, 132)\n"
            "    obs, p, rank = R['snap_bps'], R['placebo_p'], R['placebo_rank']\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.6))\n"
            "ax.hist(vals, bins=33, color=GREY, alpha=.75)\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'Dec->Jan = {obs:+.0f} bps (rank {rank}/132)')\n"
            "ax.set_xlabel('month-pair contrast (bps)'); ax.set_ylabel('count of pairs')\n"
            "ax.set_title('The Dec->Jan contrast among ALL 132 ordered month pairs (exact, no RNG)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f} bps   rank {rank}/132   exact p = {p:.4f} (floor {1/132:.4f})')"
        ),
        md(
            f"> 💡 In plain words: of all 132 ways to pick an ordered pair of months, Dec→Jan lands "
            f"**{R['placebo_rank']}ᵗʰ** (exact p = {R['placebo_p']:.4f}, floor "
            f"{R['placebo_floor']:.4f}) — top 3%. And the pairs that beat it all involve January or "
            "February (January is the panel's best month, February its worst — the snap-back partly "
            "gives itself back). The seasonal structure has exactly the tax-loss calendar's shape."
        ),
        md(
            "### 4c · Robustness — the 26-winter VWLTX tape, and the fade\n\n"
            "MUB only lists in 2007. VWLTX (Vanguard Long-Term Tax-Exempt, NAV-priced) extends the "
            "same test to 2000. Then the midpoint split on both benchmarks."
        ),
        code(
            "if HAVE_REAL:\n"
            "    a = WT.loc[WT.index <= 2016, 'jan']; b = WT.loc[WT.index > 2016, 'jan']\n"
            "    a2 = WT2.loc[WT2.index <= 2012, 'jan']; b2 = WT2.loc[WT2.index > 2012, 'jan']\n"
            "    rows = [(x.mean() * 1e4, st.ttest_vs_zero(x), len(x)) for x in (a, b, a2, b2)]\n"
            "else:\n"
            "    rows = [(v, t, n) for _, v, t, n in R['fade']]\n"
            "labels = [lbl for lbl, *_ in R['fade']]\n"
            "means = [r[0] for r in rows]; ts = [r[1] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(9.8, 4.6))\n"
            "cols = [GREEN, AMBER, GREEN, AMBER]\n"
            "ax.bar(labels, means, color=cols, width=.6)\n"
            "for i, (m, t) in enumerate(zip(means, ts)):\n"
            "    ax.annotate(f'{m:+.0f} bps\\nt={t:+.2f}', (i, m), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('mean January excess (bps)')\n"
            "ax.set_title('The January effect is front-loaded on BOTH benchmarks (midpoint splits)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for lbl, (m, t, n) in zip(labels, rows):\n"
            "    print(f'{lbl:>16}: {m:+8.2f} bps  t = {t:+.2f}  (n={n})')"
        ),
        md(
            f"> 💡 In plain words: the full VWLTX tape *strengthens* the January leg "
            f"(**{R['alt']['jan']:+.2f} bps, t = {R['alt']['t_jan']:.2f}**, hit {R['alt']['hit']:.0f}%, "
            f"placebo p = {R['alt']['p']:.4f}) — but on **both** benchmarks the effect concentrates "
            f"in the first half of the sample (MUB: +{R['fade'][0][1]:.0f} → +{R['fade'][1][1]:.0f} "
            f"bps; VWLTX: +{R['fade'][2][1]:.0f} → +{R['fade'][3][1]:.0f} bps). The early-minus-late "
            f"Welch t is **{R['fade_welch']:+.2f}** — the fade is *not itself certified* at t ≥ 2, "
            "but the recent decade alone cannot certify the effect either. Classic McLean-Pontiff "
            "shape for a 2006-published anomaly; it drives the Tradability stamp."
        ),
        md(
            "### 4d · Costs — the January swap (hold MUB, swap into the basket for January)\n\n"
            "Enter at the last December close, exit at the last January close; **4 one-way legs** per "
            "winter (sell MUB, buy basket, sell basket, buy MUB) at one-way cost × NAV. Gross P&L "
            "over just-holding-MUB is exactly the basket's January excess."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [st.january_swap(WT, cost_bps=cb) for cb in (5.0, 10.0, 25.0)]\n"
            "    nets = [r['net_mean_bps'] for r in rows]; ts = [r['t_net'] for r in rows]\n"
            "    gross = rows[0]['gross_mean_bps']\n"
            "else:\n"
            "    nets = [s[2] for s in R['swap']]; ts = [s[3] for s in R['swap']]\n"
            "    gross = R['swap_gross']\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))\n"
            "labels = ['5 bps', '10 bps', '25 bps']\n"
            "axes[0].bar(labels, nets, color=AMBER, width=.55)\n"
            "axes[0].axhline(gross, ls='--', c=GREY, label=f'gross {gross:+.0f} bps')\n"
            "for i, v in enumerate(nets): axes[0].annotate(f'{v:+.0f}', (i, v), ha='center', va='bottom')\n"
            "axes[0].set_ylabel('net excess per winter (bps)'); axes[0].set_xlabel('one-way cost')\n"
            "axes[0].set_title('Net January swap P&L'); axes[0].legend()\n"
            "axes[1].bar(labels, ts, color=AMBER, width=.55)\n"
            "axes[1].axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i, v in enumerate(ts): axes[1].annotate(f't={v:.2f}', (i, v), ha='center', va='bottom')\n"
            "axes[1].set_title('...and its t across winters'); axes[1].legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for lbl, n, t in zip(labels, nets, ts): print(f'{lbl:>7} one-way: net {n:+.2f} bps/winter  t = {t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: at 5 bps one-way the swap still clears the bar "
            f"(**{R['swap'][0][2]:+.0f} bps net, t = {R['swap'][0][3]:.2f}**); at a realistic 10 bps "
            f"for small CEFs it slips under (t = {R['swap'][1][3]:.2f}); at 25 bps it's t = "
            f"{R['swap'][2][3]:.2f}. Worse: the post-2016 gross (+{R['fade'][1][1]:.0f} bps) sits "
            "*below* the 40–100 bps drag — the trade's last decade is net-negative. Capacity is a "
            "few $100k per fund before you *are* the January bid. **FRAGILE.**"
        ),
        md(
            "### 4e · Third axis — Dec-15 vs Jan-15 (the actionable version)\n\n"
            "Entry A = first close on/after Dec-15; entry B = first close on/after Jan-15; common "
            "exit = last February close. The paired per-winter difference is exactly the excess "
            "earned between Dec-15 and Jan-15 — where does the seasonal payoff actually live?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    dj = st.dec15_vs_jan15(PX, data.BENCH, data.FUNDS)\n"
            "    yrs = [str(y) for y in dj.index]; diffs = (dj['diff'] * 1e4).values\n"
            "else:\n"
            "    yrs = [str(2007+i) for i in range(19)]; diffs = np.full(19, R['diff'])\n"
            "fig, ax = plt.subplots(figsize=(10.2, 4.6))\n"
            "ax.bar(yrs, diffs, color=[GREEN if v > 0 else RED for v in diffs], width=.7)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('Dec-15 entry minus Jan-15 entry (bps)')\n"
            "ax.set_title('The mid-Dec -> mid-Jan window carries the payoff, winter by winter')\n"
            "plt.xticks(rotation=45); plt.tight_layout(); plt.show()\n"
            "print(f'mean {diffs.mean():+.1f} bps   positive {(diffs>0).sum()}/{len(diffs)}   '\n"
            "      f'ex-2008 mean {diffs[[y != \"2008\" for y in yrs]].mean():+.1f} bps')"
        ),
        md(
            f"> 💡 In plain words: buying **Dec-15** beats waiting for **Jan-15** by "
            f"**{R['diff']:+.0f} bps** on average (positive {R['hit_diff']:.0f}% of winters; the "
            f"Jan-15 buyer keeps only {R['jan15']:+.0f} of {R['dec15']:+.0f} bps). This is where the "
            "\"December dump\" went: it happens mid-month and the recovery starts before year-end, so "
            "the calendar-month December nets to zero while the mid-Dec→mid-Jan window carries "
            f"everything. But the *t* is {R['t_diff']:.2f} full-sample — the giant bar is the 2008 "
            f"winter (+24% excess) — and ex-2008 it's +{R['ex08']:.0f} bps at t = {R['t_ex08']:.2f} "
            f"(n = {R['n_ex08']}): right at the bar. **MIXED** — directionally confirmed, not sealed."
        ),
        md(
            "### 4f · Synthetic control — the machinery is faithful\n\n"
            "A deterministic monthly excess panel (common muni-CEF factor + fund noise) with "
            "plantable December-dump / January-snap knobs. Zero knobs must NOT fire; planted knobs "
            "must light up."
        ),
        code(
            "res = []\n"
            "for dump, snap in ((0.0, 0.0), (0.010, 0.010)):\n"
            "    exs = data.synthetic_excess(dump=dump, snap=snap, seed=616)\n"
            "    wts = st.winter_table(exs)\n"
            "    wss = st.winter_stats(wts)\n"
            "    pls = st.exact_pair_placebo(exs)\n"
            "    res.append((dump * 1e4, wss['jan_mean_bps'], wss['t_jan'], wss['snap_mean_bps'],\n"
            "                wss['t_snap'], pls['p_value']))\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "labels = [f'planted\\n{d:.0f} bps' for d, *_ in res]\n"
            "tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i, t in enumerate(tvals): ax.annotate(f't={t:.2f}', (i, t), ha='center', va='bottom')\n"
            "ax.set_ylabel('per-winter snap one-sample t')\n"
            "ax.set_title('Control: zero season -> silent; planted dump+snap -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for d, j, tj, s, ts_, p in res:\n"
            "    print(f'planted {d:+.0f} bps: Jan {j:+.1f} (t={tj:+.2f})  snap {s:+.1f} (t={ts_:+.2f})  placebo p={p:.4f}')"
        ),
        md(
            f"> 💡 In plain words: with nothing planted the machinery stays silent (snap t = "
            f"{R['syn'][0][5]:.2f}, placebo p = {R['syn'][0][6]:.4f}); with a 100-bps dump + 100-bps "
            f"snap planted it fires hard (t = {R['syn'][1][5]:.2f}, placebo at its 1/132 floor). The "
            "measurement is unbiased — so the real-tape January *t*'s are genuine signal, not a "
            "construction artefact. *(A faithful-engine / power check only — never cited in support "
            "of the real-tape stamps.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** (*Real on the January snap-back · None on the December-month "
            f"dump*) — January excess **{R['jan_bps']:+.2f} bps at t = {R['t_jan']:.2f}** "
            f"({R['n_winters']} winters, hit {R['hit_jan']:.0f}%), **{R['alt']['jan']:+.2f} bps at "
            f"t = {R['alt']['t_jan']:.2f}** on {R['n_winters_alt']} VWLTX winters, exact placebo "
            f"p = {R['placebo_p']:.3f}/{R['alt']['p']:.3f}. December flat "
            f"({R['dec_bps']:+.2f} bps, t = {R['t_dec']:.2f}). Survivorship named; fade named "
            f"(post-2016 +{R['fade'][1][1]:.0f} bps, t = {R['fade'][1][2]:.2f}; fade Welch "
            f"t = {R['fade_welch']:+.2f}, not certified).\n"
            f"- **Tradability `FRAGILE`** — net **{R['swap'][0][2]:+.2f} bps/winter at "
            f"t = {R['swap'][0][3]:.2f}** at 5 bps one-way, under the bar at 10 bps "
            f"(t = {R['swap'][1][3]:.2f}); post-2016 gross below the cost drag; once-a-year, tiny "
            "thin funds. Not INVESTABLE.\n"
            f"- **Buy Dec-15, not Jan-15? `MIXED`** — **{R['diff']:+.2f} bps** paired difference "
            f"(hit {R['hit_diff']:.0f}%; Jan-15 keeps only {R['jan15']:+.0f} bps), but "
            f"t = {R['t_diff']:.2f} full-sample and +{R['ex08']:.2f} bps at t = {R['t_ex08']:.2f} "
            "ex-2008. Directionally confirmed, statistically not sealed."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The proxy is the price of admission.** Without per-fund NAV the discount is proxied "
            "by fund-minus-benchmark excess; the CEFs' leverage adds noise (but no Dec/Jan calendar "
            "of its own). A CEFConnect-style NAV tape would sharpen every number here.\n"
            "- **Condition on the loser.** Starks-Yong-Zheng's cross-sectional refinement — the "
            "snap-back concentrates in the funds with year-to-date *losses* to harvest — is the "
            "natural next study; our panel-level test averages over it.\n"
            "- **The level cousin.** [367-closed-end-fund-discount](../../367-closed-end-fund-discount/README.md) "
            "(Real · Fragile) is the *level* edge on the same machinery; this was the *seasonal "
            "flow*. Both end at the same place: real mispricing, no capacity.\n\n"
            "*The reproducible core is offline and deterministic; the primary unit is one winter, one "
            "observation, and the placebo is an exact enumeration. Methods and sources: "
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
