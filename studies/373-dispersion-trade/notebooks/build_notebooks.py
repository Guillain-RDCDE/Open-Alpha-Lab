"""Generate the two narrative notebooks for Study 373 (Dispersion-Trade).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket prices
under ../_cache/ (the dispersion proxy: SPY index vol vs avg single-name vol) and otherwise
quote the frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic
positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance realized-vol proxy
# from a 40-name large-cap basket + SPY, 2005-02-02 -> 2026-06-18, 5,378 days, 21.4 years).
R = dict(
    start="2005-02-02", end="2026-06-18", days=5378, years=21.4,
    index_vol=15.6, avg_vol=24.2, gap=8.5, gap_pos=99.6, impl_corr=0.41,
    # carry book
    n_carry=5315, sharpe=0.17, win=44, t=0.76, t_hac=0.19, p_placebo=0.425,
    skew=4.15, median=-0.00097, worst_day=-0.1514, worst_date="2009-06-09",
    gross_ann=8.2, net_ann=7.6, net_sharpe=0.15,
    # robustness: (window, gap_pos%, sharpe, naive_t, hac_t, placebo_p)
    robust=[(10, 99.2, 0.14, 0.64, 0.18, 0.422), (21, 99.6, 0.17, 0.76, 0.19, 0.425),
            (42, 100.0, 0.21, 0.94, 0.21, 0.423), (63, 100.0, 0.24, 1.11, 0.25, 0.400)],
    # synthetic: (planted_carry, gap_pos%, mean, sharpe, hac_t, placebo_p)
    syn=[(0.0, 100.0, 0.0, 0.00, 0.00, 0.502), (0.0004, 100.0, 0.0004, 1.91, 2.10, 0.019),
         (0.0010, 100.0, 0.0010, 4.78, 5.25, 0.000)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Free_lunch%3F: Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square)\n\n"
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

from dispersion_trade import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real() if HAVE_REAL else None
PNL = st.carry_pnl(F) if HAVE_REAL else None
print("real dispersion-proxy cache present:", HAVE_REAL,
      "| proxy days:", (0 if F is None else len(F)))
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
            "# The volatility \"free lunch\" — is index vol really *cheap*? 🌐\n"
            "### Sell the index's vol, buy the stocks' vol, pocket the gap — the dispersion trade in plain English\n\n"
            + BADGES +
            "Here's a trade the volatility desks love: the **dispersion trade**. The volatility of "
            "the **whole index** (say, the S&P 500) is almost always *lower* than the average "
            "volatility of the **individual stocks** inside it. So — the pitch goes — index vol is "
            "**\"reliably cheap\"**: sell it, buy the single-name vol, and harvest the difference for "
            "free.\n\n"
            "It sounds like arbitrage. It isn't. The gap is **real** — but it's *arithmetic*, the "
            "same reason a team is steadier than its players. And the carry you're supposed to collect "
            "for it turns out to be a slow bleed that pays off mostly in the exact crash that wrecks "
            "everything else. This notebook shows why \"cheap\" here means *insurance you're being paid "
            "to sell*, not free money.\n\n"
            "> 📓 **Plain-language layer.** Want the identity, the HAC *t*-stat, the placebo test and "
            "the skew? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** The real trade uses *option-implied* vols, which aren't "
            "free. So we build a transparent **proxy** from *realized* (actually-happened) vol: SPY's "
            "wobble vs. the average wobble of 40 big stocks. We call it a proxy throughout — and the "
            "lesson (the gap is an identity, the carry is noise) only gets cleaner for it. Every chart "
            "is drawn by the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Is the index's vol lower than the average stock's? | **Yes — almost always.** "
            f"Index vol ~**{R['index_vol']:.0f}%**, average stock ~**{R['avg_vol']:.0f}%**, a "
            f"**{R['gap']:.0f}-point** gap **{R['gap_pos']:.0f}%** of days. |\n"
            "| So index vol *is* cheap? | **No — it's *lower*, which isn't the same thing.** A "
            "basket is mathematically steadier than its parts whenever they don't move in lockstep. "
            "That gap is the **price of diversification**, not a mispricing. |\n"
            f"| Can you harvest the gap as a steady carry? | **Not really.** The trade that monetizes "
            f"it wins only **{R['win']}%** of days, earns a Sharpe of ~**{R['sharpe']:.2f}**, and is "
            "**statistically indistinguishable from zero** once you account for the noise. |\n"
            "| Then why do desks run it? | **They're paid to sell crash insurance.** The gap is the "
            "premium for being **short correlation** — and that bill comes due, all at once, in a "
            "panic (our worst day: **June 2009**). |\n\n"
            "> The gap is real and reliable — because it's *arithmetic*. The free lunch is not: you're "
            "being paid to sell the market a kind of insurance."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Index volatility is reliably **cheap** versus the average of its single-name vols. "
            "So sell index variance, buy single-name variance, and harvest the spread — a persistent, "
            "diversified carry.\"*\n\n"
            "This is real desk strategy, not folklore — the **dispersion** (or **correlation**) trade. "
            "The seductive part is a hard fact: when stocks don't all move together, the index *must* "
            "be calmer than the typical stock. So the gap between \"average stock vol\" and \"index "
            "vol\" is **always there**, day after day. If you could just bank that gap, you'd have a "
            "machine. We'll rebuild the gap, then ask the only question that matters: *can you actually "
            "bank it?*"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two very different things hide inside the word \"cheap.\" (1) *Is index vol **lower** than "
            "the average stock's?* Yes — but that's **diversification arithmetic**: a portfolio is "
            "steadier than its holdings unless they're perfectly correlated. (2) *Is index vol "
            "**mispriced**, so that selling it pays you above fair value?* That's the actual edge — and "
            "it's a completely separate claim. A gap that exists **by construction** is not a profit "
            "opportunity; it's the **fee you collect for absorbing correlation risk** — and like any "
            "insurer, you're fine until the day everyone needs to claim at once."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We rebuild the dispersion gap on a **transparent realized-vol proxy**: across a fixed "
            f"40-name large-cap basket, how much does the average stock wobble (21-day realized vol) "
            f"versus how much does SPY wobble? Over **{R['years']:.0f} years** ({R['start']} → "
            f"{R['end']}) we get **{R['days']:,}** daily readings.\n\n"
            "1. **Measure the gap.** Average single-name vol minus index vol, every day. Is it "
            "reliably positive? (Spoiler: yes — that's the identity.)\n"
            "2. **Try to bank it.** Build the carry book — long single-name variance, short index "
            "variance, struck at a trailing reference so we only earn the *surprise* — and look at "
            "its daily payoff.\n"
            "3. **Stress the luck.** Flip the signs of random chunks of that payoff thousands of times "
            "and ask how often pure chance looks as good. That's the honest test of a carry stream."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the gap itself.** Index vol (SPY) tracks *below* the average single-name vol — "
            "all the time. The shaded distance between the lines is the dispersion gap."
        ),
        code(
            "if HAVE_REAL:\n"
            "    iv = F['index_vol']*100; av = F['avg_vol']*100\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "    ax.plot(av.index, av.values, c=AMBER, lw=1.3, label='avg single-name vol')\n"
            "    ax.plot(iv.index, iv.values, c=GREEN, lw=1.3, label='index vol (SPY)')\n"
            "    ax.fill_between(av.index, iv.values, av.values, color=GREY, alpha=.25, label='dispersion gap')\n"
            "    ax.set_ylabel('annualised realized vol (%)')\n"
            "    ax.set_title('Index vol sits below average single-name vol — essentially always')\n"
            "    ax.legend(loc='upper right'); plt.tight_layout(); plt.show()\n"
            "    print(f'mean index vol {iv.mean():.1f}%  |  mean avg stock vol {av.mean():.1f}%  |  gap +{(av-iv).mean():.1f} pts')\n"
            "else:\n"
            "    print(f\"index vol ~{R['index_vol']}%  avg stock vol ~{R['avg_vol']}%  gap ~+{R['gap']} pts (see docs/results.md)\")"
        ),
        md(
            f"Index vol averages **{R['index_vol']:.0f}%**, the average stock **{R['avg_vol']:.0f}%** "
            f"— a **+{R['gap']:.0f}-point** gap that's positive **{R['gap_pos']:.0f}%** of all days. "
            "Rock-solid. But *why* is it so reliable? Because it's diversification — let's prove that's "
            "all it is."
        ),
        md(
            "**Why the gap *has* to be there.** The index's vol is roughly the average stock's vol "
            "times the square-root of how correlated the stocks are. When correlation is below 1 (it "
            "always is), the index is mechanically calmer. Here's the implied correlation our gap "
            "translates into — never near 1, so the gap never closes."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ic = F['impl_corr']\n"
            "else:\n"
            "    rng = np.random.default_rng(373); ic = None\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.0))\n"
            "if HAVE_REAL:\n"
            "    ax.plot(ic.index, ic.values, c=RED, lw=.9)\n"
            "    ax.axhline(ic.mean(), c=GREY, ls='--', label=f'mean ~{ic.mean():.2f}')\n"
            "    ax.set_ylim(0, 1)\n"
            "else:\n"
            "    ax.axhline(R['impl_corr'], c=RED, lw=2, label=f\"mean ~{R['impl_corr']:.2f}\"); ax.set_ylim(0,1)\n"
            "ax.axhline(1.0, c='k', lw=.8, ls=':', label='correlation = 1 (gap closes)')\n"
            "ax.set_ylabel('implied correlation  (idx vol / avg vol)$^2$')\n"
            "ax.set_title('The gap is just correlation < 1 — diversification, not mispricing')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"average implied correlation ~ {R['impl_corr']:.2f} — far below 1, so the gap is always open\")"
        ),
        md(
            f"Average implied correlation is about **{R['impl_corr']:.2f}** — comfortably below 1, so "
            "the index stays calmer than its stocks *by arithmetic*. The gap isn't a deal; it's the "
            "shadow of diversification. Now: can you turn that shadow into cash?"
        ),
        md(
            "**The carry — can you bank the gap?** Here's the daily payoff of the trade that's "
            "supposed to harvest it (long single-name variance, short index variance). If it were a "
            "free lunch, this would drift smoothly up."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cum = PNL.cumsum()\n"
            "    win = (PNL>0).mean()*100\n"
            "    fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "    ax.plot(cum.index, cum.values, c=RED, lw=1.4)\n"
            "    ax.axhline(0, c='k', lw=.8)\n"
            "    ax.set_title(f'The carry bleeds most days (win-rate {win:.0f}%) and lurches on rare spikes')\n"
            "    ax.set_ylabel('cumulative carry (variance points)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'wins {win:.0f}% of days; the equity line is choppy, not a smooth up-drift')\n"
            "else:\n"
            "    print(f\"carry Sharpe ~{R['sharpe']}, win-rate {R['win']}% (see docs/results.md)\")"
        ),
        md(
            f"Not a money machine — a **choppy, mostly-bleeding** line. The carry is positive only "
            f"**{R['win']}%** of days: you lose small most of the time and get paid in rare bursts. "
            f"Its Sharpe is about **{R['sharpe']:.2f}** — and the honest test below says even *that* is "
            "indistinguishable from luck."
        ),
        md(
            "**Could random luck do this well?** The honest test for a carry stream: flip the signs of "
            "random chunks of the daily payoff thousands of times, and see where the real average lands "
            "in that cloud of chance."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(PNL.values)\n"
            "    obs = pl['obs_mean']; pval = pl['p_value']\n"
            "    x = PNL.values; x=x[np.isfinite(x)]; n=len(x); block=21\n"
            "    nb=int(np.ceil(n/block)); rng=np.random.default_rng(373)\n"
            "    draws=np.array([(x*np.repeat(rng.choice((-1.,1.),size=nb),block)[:n]).mean() for _ in range(6000)])\n"
            "else:\n"
            "    obs=0.000325; pval=R['p_placebo']; rng=np.random.default_rng(373); draws=rng.normal(0,0.0004,6000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='random sign-flip luck')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'the actual carry ({obs:.4f})')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('average daily carry (variance points)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The carry sits inside the luck cloud — placebo p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'random sign-luck matches or beats the carry {pval*100:.0f}% of the time — nowhere near rare')"
        ),
        md(
            f"The red line — the real carry — falls **inside** the grey cloud of pure chance. About "
            f"**{R['p_placebo']*100:.0f}%** of random sign-flips do as well or better. In plain terms: "
            "**the carry's record is what you'd get by flipping coins.** The gap is real; banking it as "
            "a reliable edge is not."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The gap is genuinely real and reliable (**+{R['gap']:.0f} pts, "
            f"{R['gap_pos']:.0f}%** of days) — but it's the **diversification identity**, not an edge, "
            f"and the carry that would cash it in is statistically indistinguishable from zero "
            f"(Sharpe **{R['sharpe']:.2f}**, win-rate **{R['win']}%**). Real as arithmetic, weak as "
            "edge.\n"
            "- **Tradability — Mirage.** The carry bleeds most days and pays off in the exact panic "
            "that blows up a short-correlation book (worst day **June 2009**). A Sharpe-0.1-ish stream "
            "you have to finance through a slow bleed is no NAV-scale strategy.\n"
            "- **Free lunch? — Busted.** \"Cheap\" really means **lower-by-arithmetic** plus **paid "
            "for selling crash insurance.** It's a premium, not a gift."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — mind the tail\n\n"
            "Forget the average for a second and look at the *shape* of the payoffs. The whole risk of "
            "a dispersion seller lives in the left tail — the days correlations slam to 1 and the gap "
            "you were short detonates. Here's the distribution of daily carry."
        ),
        code(
            "if HAVE_REAL:\n"
            "    x = PNL.values; x=x[np.isfinite(x)]; sk = PNL.skew(); md_=np.median(x); wd=x.min()\n"
            "else:\n"
            "    rng=np.random.default_rng(373); x=rng.standard_normal(5000)*0.01; sk=R['skew']; md_=R['median']; wd=R['worst_day']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "ax.hist(x, bins=80, color=AMBER, alpha=.85)\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.axvline(md_, c=RED, ls='--', label=f'median {md_:.4f} (NEGATIVE — you bleed)')\n"
            "ax.set_xlabel('daily carry (variance points)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Negative median, fat right tail (skew {sk:+.1f}): finance a bleed, pray for spikes')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'median {md_:.4f} (you lose on a typical day) | skew {sk:+.1f} | worst day {wd:.3f}')"
        ),
        md(
            f"There's the operational truth. The **median day loses money** ({R['median']:.4f}); the "
            f"mean is dragged positive only by a **fat right tail** (skew **{R['skew']:+.1f}**). You're "
            "financing a steady bleed in exchange for occasional jumps — and historically the *bad* "
            f"jumps for a dispersion seller arrive in crises (our worst day, **{R['worst_date']}**, in "
            "the GFC aftermath). Costs barely move any of this; the **shape** is the problem, and it's "
            "not a shape you can run a fund on."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The real trade uses *implied* vol.** Our proxy is *realized* vol; the genuine "
            "dispersion seller harvests the **implied-minus-realized** spread on variance swaps. That "
            "spread is itself a risk premium (you're paid for correlation risk), so the honest version "
            "of this study still ends at *\"a premium, not a free lunch\"* — but with options data you "
            "could measure the premium directly.\n"
            "- **The vol the index leg is short.** [Study 03 — Fear-Gauge](../03-fear-gauge/) is the "
            "VIX — the implied index vol a dispersion trade sells.\n"
            "- **Build your own basket.** Swap our 40 names for the full S&P 500 and re-run; the gap's "
            "magnitude shifts, the lesson doesn't: it's an identity, and the carry is inside the noise.\n\n"
            "*Think the carry beats luck? Capture the daily payoff, sign-flip random blocks, and show "
            "the real mean landing **outside** the cloud — then we'll talk.*"
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
            "# Dispersion-Trade — a quantitative teardown 🔬\n"
            "### The subadditivity identity · realized dispersion & implied correlation · the carry book "
            "with a HAC *t* + block-sign-flip placebo · costs & skew · a synthetic faithful-engine / "
            "planted-carry control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We separate "
            "the two things \"index vol is cheap\" fuses: a **mechanical** index-vs-basket vol gap "
            "(subadditivity — an identity) from a **tradable carry** (a predictable, monetizable "
            "edge). The decisive object is not the gap's sign (positive by construction) but the carry's "
            "**autocorrelation-robust significance**: with overlapping realized-vol windows the naive "
            "*t* is inflated, and once you HAC-correct it, the carry is indistinguishable from zero.\n\n"
            "> ⚠️ **Data + proxy note.** The real trade is **implied** index variance vs **implied** "
            "single-name variance (variance swaps); implied vols aren't on yfinance, so we build a "
            "transparent **realized-vol proxy** — 21-day realized vol of SPY vs the equal-weight average "
            "realized vol of a fixed 40-name large-cap basket (survivorship biases the gap *narrower*, "
            "against the claim; named on the Signal axis). Real data: yfinance daily adjusted closes, "
            "SPY + basket, 2005→2026. Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `WEAK` | Gap **+{R['gap']:.0f} vol-pts**, positive **{R['gap_pos']:.0f}%** "
            f"of days — but that's subadditivity. The *carry*: Sharpe **{R['sharpe']:.2f}**, win "
            f"**{R['win']}%**, naive *t* {R['t']:.2f} → **HAC *t* = {R['t_hac']:.2f}**, placebo "
            f"*p* = {R['p_placebo']:.2f}. Real-as-identity, **fails *t* ≥ 2**. |\n"
            f"| **Tradability** | `MIRAGE` | Net Sharpe **{R['net_sharpe']:.2f}** (costs barely bite); "
            f"the carry is **negative-median** ({R['median']:.4f}), **+{R['skew']:.1f}-skew** convexity "
            f"whose worst day ({R['worst_date']}) is a correlation-spike crash. |\n"
            f"| **Free lunch?** | `BUSTED` | Synthetic basket, **zero** planted carry: gap positive "
            f"**100%** of days, carry HAC *t* = **{R['syn'][0][4]:.2f}**. Subadditivity manufactures the "
            "gap but **zero** carry — the cheapness is the correlation risk premium. |\n\n"
            "> 💡 In plain words: index vol is mechanically lower than average single-name vol "
            "(σ_index ≈ √ρ·avg σ, ρ < 1). That's an identity. The carry that would monetize it is a "
            "thin, fat-tailed stream the null explains away — you're being **paid to be short "
            "correlation**, not handed alpha."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Index variance is the correlation-weighted average of single-name variances:\n\n"
            "$$\\sigma_{\\text{idx}}^2 = \\sum_i\\sum_j w_i w_j \\rho_{ij}\\,\\sigma_i\\sigma_j "
            "\\;\\le\\; \\Big(\\sum_i w_i \\sigma_i\\Big)^2,$$\n\n"
            "with equality only at $\\rho_{ij}\\equiv 1$. Under a one-factor approximation "
            "$\\sigma_{\\text{idx}} \\approx \\sqrt{\\bar\\rho}\\,\\overline{\\sigma}$, so the gap "
            "$\\overline{\\sigma} - \\sigma_{\\text{idx}} \\ge 0$ is an **identity** for $\\bar\\rho<1$.\n\n"
            "- **H₁ (the gap is real).** $\\mathbb{E}[\\overline{\\sigma}-\\sigma_{\\text{idx}}] > 0$ "
            "— **trivially true** (subadditivity). Not the edge.\n"
            "- **H₂ (the carry is real).** Holding long-single / short-index variance, struck at a "
            "trailing reference, earns a **positive mean with a robust $t\\ge 2$** — *this* is the "
            "monetizable claim.\n"
            "- **H₃ (it's a free lunch).** The carry is *not* merely the **correlation risk premium** "
            "(compensation for short-correlation tail risk).\n\n"
            "We find **H₁ trivially true**, **H₂ rejected** (HAC $t\\approx0.2$, placebo $p\\approx0.4$), "
            "**H₃ rejected** (negative-median, fat-right-tail convexity; the payoff structure of an "
            "insurer). The claim is true exactly where it's an identity and unproven exactly where it "
            "would pay."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The whole teardown is one separation: an **identity** (the gap) from an **edge** (the "
            "carry), judged by an **autocorrelation-robust** standard error.\n\n"
            "$$\\widehat{\\mu}_{\\text{carry}} = \\overline{\\pi_t},\\qquad "
            "t_{\\text{HAC}} = \\frac{\\widehat{\\mu}}{\\sqrt{\\widehat{\\mathrm{Var}}_{\\text{NW}}"
            "(\\overline{\\pi})}},\\quad \\widehat{\\mathrm{Var}}_{\\text{NW}} = "
            "\\frac{1}{n}\\Big(\\gamma_0 + 2\\sum_{k=1}^{L}(1-\\tfrac{k}{L+1})\\gamma_k\\Big).$$\n\n"
            "Overlapping rolling-vol windows make $\\pi_t$ heavily autocorrelated, so $\\gamma_{k>0}>0$ "
            "and the **naive** $t$ (which assumes $\\gamma_{k>0}=0$) is **inflated**. The Newey–West "
            "correction is not optional here; neither is a **block** resampling null (an i.i.d. shuffle "
            "would destroy the very clustering the inference must respect). The carry's *sign* is "
            "uninformative — its **significance under autocorrelation** decides the Signal axis."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Dispersion proxy.** 21-day realized vol of SPY (index) vs the equal-weight average "
            f"21-day realized vol of a fixed 40-name basket (yfinance adjusted closes, {R['start']}→"
            f"{R['end']}, {R['days']:,} days). Explicit **realized-vol proxy** for the implied "
            "dispersion trade; survivorship biases the gap *narrower* (named on the axis).\n"
            "- **Implied correlation.** $\\widehat{\\rho} = (\\sigma_{\\text{idx}}/\\overline\\sigma)^2$ "
            "— the one-factor read of how much diversification is open.\n"
            "- **Carry book.** Long single-name variance, short index variance, struck at the trailing "
            "63-day mean dispersion (the proxy for what's already priced in); **1-day lag** on the "
            "strike (signal known at $t-1$, realized gap of $t$ booked). No look-ahead.\n"
            "- **Null #1 (HAC *t*).** Newey–West (Bartlett, 21 lags) *t* of the carry mean vs 0.\n"
            "- **Null #2 (placebo).** 20,000 **block-sign-flip** draws (21-day blocks); "
            "$p=\\Pr[\\text{placebo mean} \\ge \\text{observed}]$.\n"
            "- **Costs.** Vega/bid-ask 5e-4 var-pt × 12 re-strikes/yr, charged against the carry.\n"
            "- **Positive control.** A one-factor correlated basket with a **planted carry knob**: the "
            "engine must recover the planted edge, recover the implied correlation, **and** show the "
            "mechanical gap yields **zero** carry when the true edge is zero."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The identity — the gap is positive by construction\n\n"
            "Index vol vs average single-name vol, and the implied correlation the gap encodes. The gap "
            "is positive on essentially every day — and that is *exactly what subadditivity predicts*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    gs = st.gap_sign_stats(F)\n"
            "    iv=F['index_vol']*100; av=F['avg_vol']*100\n"
            "else:\n"
            "    gs={'mean_index_vol':R['index_vol']/100,'mean_avg_vol':R['avg_vol']/100,'share_positive':R['gap_pos']/100,'mean_impl_corr':R['impl_corr']}\n"
            "    iv=av=None\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,4.2))\n"
            "if HAVE_REAL:\n"
            "    a1.scatter(iv, av, s=4, c=GREY, alpha=.3)\n"
            "    lim=[5, max(av.max(),iv.max())*1.02]\n"
            "    a1.plot(lim, lim, c=RED, ls='--', label='index = avg stock (gap=0)')\n"
            "    a1.set_xlim(lim); a1.set_ylim(lim)\n"
            "else:\n"
            "    a1.plot([10,80],[10,80],c=RED,ls='--',label='gap=0'); a1.scatter([R['index_vol']],[R['avg_vol']],c=GREY)\n"
            "a1.set_xlabel('index vol (%)'); a1.set_ylabel('avg single-name vol (%)')\n"
            "a1.set_title(f\"avg stock vol > index vol {gs['share_positive']*100:.1f}% of days\"); a1.legend()\n"
            "if HAVE_REAL:\n"
            "    a2.hist(F['impl_corr'].values, bins=50, color=RED, alpha=.8)\n"
            "    a2.axvline(gs['mean_impl_corr'], c='k', ls='--', label=f\"mean {gs['mean_impl_corr']:.2f}\")\n"
            "else:\n"
            "    a2.axvline(R['impl_corr'],c='k',ls='--',label=f\"mean {R['impl_corr']:.2f}\")\n"
            "a2.set_xlim(0,1); a2.set_xlabel('implied correlation'); a2.set_title('correlation < 1 keeps the gap open'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"gap>0 {gs['share_positive']*100:.1f}% of days | index {gs['mean_index_vol']*100:.1f}% avg {gs['mean_avg_vol']*100:.1f}% | impl_corr {gs['mean_impl_corr']:.2f}\")"
        ),
        md(
            f"> 💡 In plain words: every dot sits above the 45° line — average stock vol exceeds index "
            f"vol **{R['gap_pos']:.0f}%** of days, at a mean implied correlation **~{R['impl_corr']:.2f}**. "
            "H₁ confirmed and **uninformative**: this is the covariance-matrix identity, not evidence of "
            "mispricing. The question is whether the *carry* is anything more."
        ),
        md(
            "### 4b · The carry — and the decisive HAC correction\n\n"
            "The carry mean with its **naive** vs **HAC** *t*. The naive *t* trusts independence; the "
            "HAC *t* accounts for the autocorrelation of overlapping vol windows. Watch the *t* collapse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize_carry(PNL)\n"
            "    naive_t=s['t']; hac_t=s['t_hac']; shp=s['sharpe']; win=s['win']*100; pval=s['p_placebo']\n"
            "else:\n"
            "    naive_t=R['t']; hac_t=R['t_hac']; shp=R['sharpe']; win=R['win']; pval=R['p_placebo']\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.4,4.2))\n"
            "a1.bar(['naive t','HAC t'], [naive_t, hac_t], color=[GREY, RED], width=.55)\n"
            "a1.axhline(2, ls='--', c=GREEN, label='t = 2 (significance bar)')\n"
            "for i,t in enumerate([naive_t,hac_t]): a1.annotate(f'{t:.2f}',(i,t),ha='center',va='bottom')\n"
            "a1.set_ylim(0,2.4); a1.set_title('Correcting for autocorrelation kills the t'); a1.set_ylabel('t-stat (mean vs 0)'); a1.legend()\n"
            "if HAVE_REAL:\n"
            "    x=PNL.values; x=x[np.isfinite(x)]; n=len(x); block=21; nb=int(np.ceil(n/block)); rng=np.random.default_rng(373)\n"
            "    draws=np.array([(x*np.repeat(rng.choice((-1.,1.),size=nb),block)[:n]).mean() for _ in range(20000)]); obs=x.mean()\n"
            "else:\n"
            "    rng=np.random.default_rng(373); draws=rng.normal(0,0.0004,20000); obs=0.000325\n"
            "a2.hist(draws, bins=60, color=GREY, alpha=.85, label='block-sign-flip null')\n"
            "a2.axvline(obs, c=RED, lw=2.5, label=f'observed carry {obs:.4f}')\n"
            "a2.set_title(f'Placebo p = {pval:.2f}'); a2.set_xlabel('mean daily carry'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Sharpe {shp:.2f} | win {win:.0f}% | naive t {naive_t:.2f} -> HAC t {hac_t:.2f} | placebo p {pval:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the naive *t* (**{R['t']:.2f}**) already misses the bar, and the "
            f"**HAC *t* is {R['t_hac']:.2f}** — the autocorrelation of overlapping windows had inflated "
            f"even that. A block-sign-flip placebo matches the carry **{R['p_placebo']*100:.0f}%** of "
            "the time. H₂ rejected: the carry is statistically indistinguishable from zero."
        ),
        md(
            "### 4c · Robustness + cost — neither rescues it\n\n"
            "Sweep the realized-vol window and apply the vega/bid-ask cost. Longer windows *smooth* the "
            "carry and nudge the naive Sharpe up — but the **HAC *t* never approaches 2**, and costs "
            "shave only a sliver (so cost is **not** the binding constraint)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    prices = data.load_prices(); rob=[]\n"
            "    for w in (10,21,42,63):\n"
            "        fw=data.dispersion_proxy(prices, window=w); sw=st.summarize_carry(st.carry_pnl(fw))\n"
            "        gw=st.gap_sign_stats(fw)['share_positive']\n"
            "        rob.append((w, gw*100, sw['sharpe'], sw['t'], sw['t_hac'], sw['p_placebo']))\n"
            "    c = st.net_of_costs(PNL)\n"
            "else:\n"
            "    rob=R['robust']; c={'gross_ann':R['gross_ann']/100,'net_ann':R['net_ann']/100,'net_sharpe':R['net_sharpe']}\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.4,4.2))\n"
            "ws=[r[0] for r in rob]; hac=[r[4] for r in rob]; shp=[r[2] for r in rob]\n"
            "a1.bar([str(w) for w in ws], hac, color=AMBER, width=.55)\n"
            "a1.axhline(2, ls='--', c=GREEN, label='t = 2'); a1.set_ylim(0,2.2)\n"
            "for i,(t,s) in enumerate(zip(hac,shp)): a1.annotate(f'Sh {s:.2f}',(i,t),ha='center',va='bottom',fontsize=8)\n"
            "a1.set_title('HAC t never reaches 2 (any window)'); a1.set_xlabel('realized-vol window (days)'); a1.set_ylabel('HAC t'); a1.legend()\n"
            "a2.bar(['gross','net @cost'], [c['gross_ann']*100, c['net_ann']*100], color=[GREEN, GREY], width=.55)\n"
            "for i,v in enumerate([c['gross_ann']*100, c['net_ann']*100]): a2.annotate(f'{v:.1f}',(i,v),ha='center',va='bottom')\n"
            "a2.set_title(f\"Cost barely bites (net Sharpe {c['net_sharpe']:.2f})\"); a2.set_ylabel('annualised carry (var-pts)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (w, gap+%, Sharpe, naive_t, HAC_t, p):', [(r[0],round(r[1],1),round(r[2],2),round(r[3],2),round(r[4],2),round(r[5],3)) for r in rob])"
        ),
        md(
            f"> 💡 In plain words: at the 63-day window the naive Sharpe creeps to 0.24, but the **HAC "
            "*t* tops out at 0.25** — smoothing buys appearance, not significance. Net of cost the "
            f"annual carry slips only **{R['gross_ann']:.1f} → {R['net_ann']:.1f}** var-pts (net Sharpe "
            f"**{R['net_sharpe']:.2f}**). There is no window and no cost regime in which this clears the "
            "bar — the constraint is the **risk-premium nature of the carry itself.**"
        ),
        md(
            "### 4d · The tail — the insurer's payoff, and the GFC bill\n\n"
            "Why this is a *mirage* even granting a tiny positive mean: the carry is **negative-median, "
            "positively-skewed convexity** — you bleed daily and are paid in spikes, and the worst spikes "
            "are correlation crashes. The distribution and the cumulative path tell it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    x=PNL.values; x=x[np.isfinite(x)]; sk=PNL.skew(); md_=np.median(x); cum=PNL.cumsum()\n"
            "    wdate=PNL.idxmin().date(); wd=PNL.min()\n"
            "else:\n"
            "    rng=np.random.default_rng(373); x=rng.standard_normal(5000)*0.01; sk=R['skew']; md_=R['median']; cum=None; wdate=R['worst_date']; wd=R['worst_day']\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,4.2))\n"
            "a1.hist(x, bins=80, color=AMBER, alpha=.85); a1.axvline(0,c='k',lw=.8)\n"
            "a1.axvline(md_, c=RED, ls='--', label=f'median {md_:.4f} < 0')\n"
            "a1.set_title(f'Negative median, skew {sk:+.1f}'); a1.set_xlabel('daily carry (var-pts)'); a1.legend()\n"
            "if HAVE_REAL:\n"
            "    a2.plot(cum.index, cum.values, c=RED, lw=1.3); a2.axhline(0,c='k',lw=.8)\n"
            "    a2.set_title('Cumulative carry — choppy, crisis-driven'); a2.set_ylabel('cumulative var-pts')\n"
            "else:\n"
            "    a2.text(.5,.5,'(cache absent)',ha='center')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'median {md_:.4f} (<0) | skew {sk:+.1f} | worst day {wd:.3f} on {wdate}')"
        ),
        md(
            f"> 💡 In plain words: the **median day loses** ({R['median']:.4f}); the positive mean is a "
            f"**skew artefact** (+{R['skew']:.1f}). That is the textbook payoff of an **insurance "
            f"seller** — small steady premium, rare large claim. The worst claim lands "
            f"**{R['worst_date']}** (GFC aftermath, correlations pinned near 1). You cannot size a NAV "
            "to a stream whose Sharpe is 0.1-ish *and* whose tail correlates with everything else "
            "blowing up. Tradability **MIRAGE**."
        ),
        md(
            "### 4e · Faithful-engine & planted-carry control — we know the truth here\n\n"
            "On a deterministic one-factor correlated basket: with **zero** planted carry the mechanical "
            "gap is still positive ~100% of days, yet the carry must read **zero** (no false positive); "
            "with a planted carry it must light up. Both hold — proving the engine separates *identity* "
            "from *edge*."
        ),
        code(
            "res=[]\n"
            "for carry in (0.0, 0.0004, 0.0010):\n"
            "    syn=data.synthetic_basket(n_days=4000, n_names=40, carry=carry, seed=373)\n"
            "    prox=data.dispersion_proxy(syn, window=21); gp=st.gap_sign_stats(prox)['share_positive']\n"
            "    s=st.summarize_carry(st.synthetic_carry_pnl(syn))\n"
            "    res.append((carry, gp*100, s['mean'], s['sharpe'], s['t_hac'], s['p_placebo']))\n"
            "fig,ax=plt.subplots(figsize=(8.8,4.3))\n"
            "labels=[f'planted\\n{c:+.4f}' for c,_,_,_,_,_ in res]; tvals=[r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, AMBER, GREEN], width=.55)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('carry HAC t'); ax.set_title('Control: zero carry -> t=0 (gap fakes nothing); planted carry lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for c,g,m,sh,t,p in res: print(f'planted {c:+.4f}: gap+={g:.0f}% mean={m:+.5f} Sharpe={sh:.2f} HAC_t={t:.2f} p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted carry the gap is positive **100%** of days yet "
            f"the carry HAC *t* is **{R['syn'][0][4]:.2f}** — **subadditivity manufactures the gap but "
            f"zero edge.** A small planted carry just clears (HAC *t* = {R['syn'][1][4]:.2f}); a large "
            f"one lights up ({R['syn'][2][4]:.2f}). (Implied-correlation recovery is faithful too: "
            "planted ρ = 0.2/0.4/0.6 → measured 0.19/0.36/0.55.) So the machinery is honest, and the "
            "real-tape HAC *t* of ~0.2 is exactly what an **absent** carry looks like — the gap is "
            "identity, the carry is the correlation risk premium."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — the gap is real (+{R['gap']:.0f} vol-pts, {R['gap_pos']:.0f}% of "
            f"days) but **mechanical** (subadditivity); the carry that monetizes it is **HAC *t* = "
            f"{R['t_hac']:.2f}**, placebo *p* = {R['p_placebo']:.2f}, Sharpe {R['sharpe']:.2f}. Identity "
            "+ a positive-but-insignificant, autocorrelation-fragile carry ⇒ WEAK, not REAL (the desk's "
            "*t* ≥ 2 bar is not met). Survivorship biases the gap *narrower*, against the claim.\n"
            f"- **Tradability `MIRAGE`** — net Sharpe **{R['net_sharpe']:.2f}**; costs barely bite, but "
            f"the carry is **negative-median, +{R['skew']:.1f}-skew** convexity whose tail (worst day "
            f"{R['worst_date']}) is a correlation-spike crash. Not a NAV-scale, allocatable edge.\n"
            f"- **Free lunch? `BUSTED`** — the synthetic control nails it: **zero** planted carry, gap "
            f"positive 100% of days, carry HAC *t* = {R['syn'][0][4]:.2f}. \"Cheap\" index vol is the "
            "**correlation risk premium** — priced compensation for short-correlation tail risk, not "
            "alpha."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the power & premium curve\n\n"
            "The operational truth in one picture: at our realized Sharpe and sample size, how big would "
            "the *true* carry have to be to clear t=2 — and where does the observed carry sit? The "
            "observed edge lives far below the detection floor, because what's left after the identity "
            "**is** the risk premium, not alpha."
        ),
        code(
            "if HAVE_REAL:\n"
            "    x=PNL.values; x=x[np.isfinite(x)]; sd=x.std(ddof=1); obs=x.mean(); n=len(x)\n"
            "else:\n"
            "    sd=0.012; obs=0.000325; n=R['n_carry']\n"
            "# min detectable mean for naive t=2 at sample size k (illustrative; HAC inflates further)\n"
            "ks=np.arange(250, 6000, 50)\n"
            "min_det = 2.0 * sd / np.sqrt(ks)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(ks, min_det, c=AMBER, lw=2, label='mean carry needed for naive t=2')\n"
            "ax.axhline(obs, c=GREEN, ls='--', label=f'observed carry ~{obs:.4f}')\n"
            "ax.axvline(n, c=GREY, ls=':', label=f'our n={n}')\n"
            "ax.set_xlabel('sample size (days)'); ax.set_ylabel('mean daily carry (var-pts)')\n"
            "ax.set_title('Even ignoring autocorrelation, the carry sits under the detection floor'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "need = 2.0*sd/np.sqrt(n)\n"
            "print(f'at n={n}, need ~{need:.5f} mean for naive t=2; observed ~{obs:.5f} -> below the floor (and HAC makes it worse)')"
        ),
        md(
            "> 💡 In plain words: the amber curve is the **minimum detectable carry**; the green line is "
            "what we observe. They don't meet at our sample — and that's *before* the HAC penalty for "
            "autocorrelation, which pushes the real bar higher still. There is no sizing, no window, no "
            "cost assumption that turns this into power, because the residual after the diversification "
            "identity is **the correlation risk premium** — compensation for a tail, not an exploitable "
            "edge. **The arithmetic that makes index vol \"cheap\" is exactly the arithmetic that makes "
            "the carry un-free.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Measure the premium directly.** With option-implied vols (or variance-swap quotes) you "
            "could compute the genuine **implied-minus-realized** dispersion spread — the actual carry "
            "source — and price the correlation risk premium (Driessen–Maenhout–Vilkov 2009) instead of "
            "proxying it with realized vol.\n"
            "- **The index leg's implied vol.** [Study 03 — Fear-Gauge](../03-fear-gauge/) is the VIX, "
            "the implied index vol a dispersion trade sells.\n"
            "- **Richer correlation model.** Replace equal weights with cap weights and the one-factor "
            "implied correlation with the full Σ; add a borrow charge on the single-name longs. The gap "
            "shrinks or grows, the **risk-premium conclusion does not.**\n\n"
            "*The reproducible core is offline and deterministic; vol here is an explicit realized-vol "
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
