"""Generate the two narrative notebooks for Study 505 (Left-Tail-Momentum).

    python notebooks/build_notebooks.py

Both notebooks follow the desk's seven beats. Synthetic figures run anywhere, offline and
deterministic; the real-panel cells use the cached yfinance parquets under ../_cache/ if
present and otherwise quote the frozen headline numbers in ``R`` (a mirror of docs/results.md,
as-of 2026-06-26).
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


# Frozen real-panel headline numbers -- mirror of docs/results.md (as-of 2026-06-26).
R = dict(
    n_months=135, year_start=2015, year_end=2026, n_tickers=48,
    ls_mean=-25.39, ls_vol=24.78, ls_sharpe=-1.025, ls_t=-3.436, ls_hac=-3.471,
    ls_hit=37.0, ls_dd=-96.1, turnover=12.0,
    long_mean=12.45, short_mean=37.84,
    ls_net=-26.04, ls_net_t=-3.523,
    placebo_real_mo=-2.116, placebo_p=0.000,
    fp_prices="8791a7217b02", fp_spy="f9f0cfe2b487",
    sweep={"var5": (-25.39, -3.436), "var1": (-20.64, -2.811), "worst": (-18.85, -3.119)},
    lookback={126: (-20.08, -2.755), 252: (-25.39, -3.436), 504: (-25.82, -3.079)},
    ctrl={0.00: (3.89, 0.840, 10), 0.04: (5.05, 1.091, 15), 0.08: (6.09, 1.317, 25)},
    annual={2015: -8.3, 2016: -35.5, 2017: -27.8, 2018: -9.9, 2019: -16.3, 2020: -38.2,
            2021: -24.8, 2022: 13.0, 2023: -46.5, 2024: -12.1, 2025: -34.1, 2026: -26.4},
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from left_tail_momentum import data, strategy as st

def _have_cache():
    study = os.path.abspath("..")
    return (os.path.exists(os.path.join(study, "_cache", "ltm_prices.parquet"))
            and os.path.exists(os.path.join(study, "_cache", "ltm_spy.parquet")))

HAVE_REAL = _have_cache()
print("yfinance cache present:", HAVE_REAL)

if HAVE_REAL:
    prices, spy = data.fetch_panel()
    book = st.long_short_book(prices, kind="var5", lookback=252, min_stocks=20,
                              cost_bps=5.0, borrow_ann_bps=50.0)
    s_ls   = st.summary(book["ls_ret"])
    s_net  = st.summary(book["ls_ret_net"])
    s_long = st.summary(book["long_ret"])
    s_short= st.summary(book["short_ret"])
    print(f"N months: {s_ls['n']} | LS mean: {s_ls['mean']*100:+.2f}%/yr"
          f" | one-sample t: {s_ls['tstat']:+.3f}")
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Left-Tail-Momentum -- do the worst-crashed stocks keep crashing?\n"
            "### Atilgan-Bali-Demirtas-Gunaydin (2020): under-reaction to bad news, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Left%20tail%20keeps%20losing%3F: Busted](https://img.shields.io/badge/Left%20tail%20keeps%20losing%3F-Busted-8b949e?style=flat-square)\n\n"
            "In 2020, Atilgan, Bali, Demirtas and Gunaydin published a striking finding: stocks "
            "with the **worst left-tail risk** -- the deepest recent crashes, the lowest "
            "Value-at-Risk -- *keep* underperforming. Investors under-react to extreme bad news, "
            "so the left tail has **momentum**: crashed stocks keep crashing. The trade is to go "
            "**long the safe tail** and **short the crashed tail**.\n\n"
            "We test that on a large-cap S&P 500 survivor basket using yfinance daily prices, "
            "naming the survivorship bias up front. The result is a clean lesson in how a real "
            "academic effect can **invert** once you put it on tradable survivors.\n\n"
            "> **This is the plain-language layer.** Want the VaR signal, the placebo null, the "
            "seed-robust control, and the year-by-year breakdown? See the companion "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do crashed stocks keep crashing? | **No -- the opposite.** On survivors the "
            f"crashed-tail leg *out-earns* the safe leg (+{R['short_mean']:.1f}% vs "
            f"+{R['long_mean']:.1f}%/yr). |\n"
            "| Does the ABDG long-safe/short-crashed book pay? | **No.** It loses "
            f"**{R['ls_mean']:.1f}%/yr** at one-sample *t* = **{R['ls_t']:.2f}** -- strongly "
            "significant, but the *wrong sign*. |\n"
            "| Is the signal real at all? | **Yes -- as reversal.** Placebo p = "
            f"**{R['placebo_p']:.3f}**: no random sort is this extreme. The left tail "
            "**rebounds** on survivors. |\n"
            "| Is there a survivorship problem? | **Yes, and it's the whole story.** The "
            "crashers that delisted -- the short leg ABDG needs -- are deleted from the basket. |\n\n"
            "> A real, robust cross-sectional signal that points the **reverse** of the claim. "
            "On large-cap survivors, a deep recent crash was a *buy*, not a sell."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim\n\n"
            "> *\"Stocks with extreme negative returns -- the worst left tails -- are "
            "under-priced for *bad* news, not good. Investors under-react to crashes, so the "
            "low-VaR stocks keep underperforming. Short them, buy the safe tail: left-tail "
            "momentum.\"*\n\n"
            "-- Atilgan, Bali, Demirtas & Gunaydin (2020), *Journal of Financial Economics*\n\n"
            "The intuition is behavioural: a sharp crash is salient *bad* news, and investors "
            "are slow to mark down a stock all the way. So the bad news keeps bleeding into "
            "prices over the following month -- momentum in the left tail. The competing force "
            "is **short-run reversal** (Jegadeesh 1990): last month's losers bounce. ABDG argue "
            "the under-reaction wins; on our basket, reversal does."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "If true, left-tail momentum is a behavioural alpha uncorrelated with the classic "
            "factors -- a way to harvest the market's slow digestion of bad news. It would also "
            "validate a whole family of 'sell the crashers' systematic strategies.\n\n"
            "The questions for us:\n"
            "1. **Does the continuation survive on a realistic large-cap panel?**\n"
            "2. **Or does short-run reversal dominate -- do crashed survivors bounce?**\n"
            "3. **What does survivorship do to a left-tail sort specifically?**"
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 -- How would we even know?\n\n"
            "Three disciplines keep us honest:\n\n"
            "1. **Signal from the past only.** The left-tail VaR at each month-end uses just the "
            "trailing year of returns; the holding return is the *next* month, entered one "
            "trading day later (one execution lag, no same-bar fill).\n"
            "2. **A placebo null.** Re-randomise which stocks are 'safe' vs 'crashed' 200 times. "
            "If a random sort matches the real one, there is no signal.\n"
            "3. **Name the survivorship bias.** The basket is the names *still trading in 2026*. "
            "The worst-tail names of the past -- the ones that crashed into delisting -- are gone. "
            "Those are exactly the names the short leg needs, so a survivor basket's 'crashed' "
            "stocks are disproportionately *rebounds*."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md("## 4 -- The teardown\n\n**First, does the engine work when the effect is planted?**"),
        code(
            "# Synthetic positive control: plant a left-tail-momentum drift and check recovery.\n"
            "# (3 seeds here for speed; docs/results.md averages 20 seeds.)\n"
            "ctrl = st.synthetic_control(data, premiums=(0.0, 0.04, 0.08), n_seeds=3,\n"
            "                            kind='var5', lookback=252, base_seed=505)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if m > 0 else RED for m in ctrl['mean_ls_pct']]\n"
            "ax.bar(ctrl['premium'].astype(str), ctrl['mean_ls_pct'], color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted left-tail-momentum premium')\n"
            "ax.set_ylabel('long-safe/short-crashed mean (%/yr)')\n"
            "ax.set_title('Synthetic control: engine recovers the effect, correct (positive) sign')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            "The engine is faithful: plant left-tail momentum and the long-safe/short-crashed "
            "mean rises **with the correct positive sign**. So if the real tape comes out "
            "*negative*, that is the **market**, not a broken detector."
        ),
        md("**Now the honest test on the real yfinance panel.**"),
        code(
            "if HAVE_REAL:\n"
            "    long_m = s_long['mean']*100; short_m = s_short['mean']*100\n"
            "    ls_m = s_ls['mean']*100; ls_t = s_ls['tstat']\n"
            "else:\n"
            "    long_m, short_m = R['long_mean'], R['short_mean']\n"
            "    ls_m, ls_t = R['ls_mean'], R['ls_t']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "labels = ['Long\\n(safe tail)', 'Short\\n(crashed tail)']\n"
            "vals = [long_m, short_m]\n"
            "ax.bar(labels, vals, color=[AMBER, RED], width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('mean annual return (%/yr)')\n"
            "ax.set_title(f'Crashed leg WINS: long-safe/short-crashed = {ls_m:+.1f}%/yr  (t={ls_t:+.2f})')\n"
            "for i, v in enumerate(vals):\n"
            "    ax.text(i, v + 0.6, f'{v:+.1f}%', ha='center', fontsize=11)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Long (safe): {long_m:+.2f}%/yr | Short (crashed): {short_m:+.2f}%/yr')\n"
            "print(f'ABDG book (long-safe - short-crashed): {ls_m:+.2f}%/yr | one-sample t = {ls_t:+.3f}')"
        ),
        md(
            f"The crashed-tail leg earns **+{R['short_mean']:.1f}%/yr** -- *more* than the safe "
            f"leg's **+{R['long_mean']:.1f}%/yr**. So the ABDG continuation book loses "
            f"**{R['ls_mean']:.1f}%/yr** at *t* = **{R['ls_t']:.2f}**: a strong, significant "
            "signal pointing the **opposite** way to the hypothesis. On survivors, the left tail "
            "**rebounds**."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- NONE.** The ABDG continuation does not appear -- it *inverts*. "
            f"Long-safe/short-crashed loses {R['ls_mean']:.1f}%/yr at *t* = {R['ls_t']:.2f}. "
            "There is a real, robust signal (placebo p = "
            f"{R['placebo_p']:.3f}), but it is **reversal**, not the claimed momentum -- so on "
            "the hypothesis as written, NONE.\n"
            f"- **Tradability -- MIRAGE.** A {R['ls_dd']:.0f}% drawdown short book whose short "
            "leg is the decade's winners. Costs are tiny and don't change a thing.\n"
            "- **Left tail keeps losing? -- BUSTED.** A deep recent crash was a buy, not a "
            "sell, on survivors."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "Not in the stated direction. The continuation book is short the best performers "
            "and carries a -96% drawdown. The *reversal* it accidentally uncovers is the known "
            "short-run-reversal effect (Jegadeesh 1990), which is itself notoriously hard to "
            "trade net of costs in liquid large-caps. And the whole thing rides on a survivor "
            "basket: the firms that crashed and *stayed* down -- ABDG's real short candidates "
            "-- delisted and are simply not here.\n\n"
            "ABDG's result lives on the **full** US universe (including the crash-into-delisting "
            "names) with costly-arbitrage tilts (illiquid, high-idio-vol stocks). A tradable "
            "large-cap survivor sleeve cannot reproduce it."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Full universe.** Re-run on a delisting-complete CRSP/Compustat panel; the "
            "crash-into-delisting names are the whole short leg.\n"
            "- **Costly-arbitrage tilt.** ABDG's effect is strongest in illiquid, "
            "high-idiosyncratic-vol names -- large-caps are the *least* fertile ground.\n"
            "- **The mirror, the right tail.** [Study 53 -- Jackpot](../../53-jackpot/) and "
            "[Study 365 -- Lottery-MAX](../../365-lottery-max-effect/) test the lottery (MAX) "
            "effect -- the right-tail twin -- both None x Mirage on survivors.\n"
            "- **Co-crash risk.** [Study 332 -- Downside-Beta](../../332-downside-beta/) prices "
            "a different left-tail exposure.\n"
            "- **The cousin factor.** [Study 238 -- Betting-Against-Beta](../../238-betting-against-beta/) "
            "is the closest cross-sectional-risk relative.\n\n"
            "*Think left-tail momentum survives once you add the delisted crashers and tilt to "
            "illiquid names? Fork this, swap in a delisting-complete universe, and show t > 2 "
            "in the right direction, net of borrow. That is the bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Left-Tail-Momentum -- a quantitative teardown\n"
            "### yfinance survivor panel * VaR sort * long-short * one-sample t * placebo null\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Left%20tail%20keeps%20losing%3F: Busted](https://img.shields.io/badge/Left%20tail%20keeps%20losing%3F-Busted-8b949e?style=flat-square)\n\n"
            "The quantitative companion to the "
            "[notebook for the curious](01_for_the_curious.ipynb) -- same seven beats, every "
            "claim carrying its standard error. We test ABDG (2020): sort on trailing 5%/1% VaR "
            "(or worst day), long the safe tail, short the crashed tail, one execution lag, "
            "monthly hold.\n\n"
            "> **Not investment advice.** Real data: yfinance daily adjusted-close, 48 S&P 500 "
            "large-cap survivors, 2014-2026, as-of 2026-06-26. Methods in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship bias is named:** the basket is the names still trading in 2026; "
            "the worst-tail names that delisted -- the short leg ABDG needs -- are absent."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Long-safe/short-crashed **{R['ls_mean']:.2f}%/yr**, "
            f"one-sample *t* = **{R['ls_t']:.3f}** (HAC {R['ls_hac']:.3f}) -- significant but "
            "**wrong sign** vs the continuation claim. |\n"
            f"| **Tradability** | `MIRAGE` | Sharpe **{R['ls_sharpe']:.3f}**, max DD "
            f"**{R['ls_dd']:.1f}%**, short leg = decade's winners. |\n"
            "| **Left tail keeps losing?** | `BUSTED` | Loses 11 of 12 years; crashed survivors "
            "rebound. |\n\n"
            "> A real, robust cross-sectional signal -- pointing **reverse** of ABDG (2020). "
            "On survivors, short-run reversal beats left-tail momentum."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $\\text{VaR}^{5}_{i,t}$ = the 5th-percentile daily return of stock $i$ over the "
            "trailing year at $t$ (a negative number; more negative = worse left tail). ABDG "
            "(2020) assert:\n\n"
            "- **H1 (continuation).** Stocks with the worst (most negative) VaR keep "
            "underperforming next month: $E[r^{\\text{worst}}_{t+1}] < E[r^{\\text{safe}}_{t+1}]$, "
            "so the spread $r^{\\text{safe}} - r^{\\text{worst}} > 0$.\n"
            "- **H2 (under-reaction).** The driver is slow incorporation of bad news, strongest "
            "where arbitrage is costly (illiquid, high-idio-vol).\n\n"
            "We test H1 directly on large-caps. The competing hypothesis is short-run reversal "
            "(Jegadeesh 1990): $r^{\\text{worst}} - r^{\\text{safe}} > 0$ instead. The data pick "
            "**reversal** -- the spread is large and *negative*."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what? -- the economic stakes\n\n"
            "Left-tail momentum is one of the cleaner *behavioural* anomalies in the post-2010 "
            "literature -- a slow-bleeding under-reaction to bad news. If it survives on tradable "
            "large-caps, it is a deployable short signal. If it inverts -- as here -- it is a "
            "case study in how survivorship and reversal conspire to flip a published factor's "
            "sign on the part of the universe you can actually hold."
        ),

        # ---- BEAT 3 -- PROTOCOL ----------------------------------------------
        md(
            "## 3 -- The protocol\n\n"
            "- **Signal.** Trailing 252-day left-tail statistic per stock: `var5` (5% quantile), "
            "`var1` (1% quantile), or `worst` (min daily return).\n"
            "- **Ranking.** Monthly, sort the 48 names; quintile groups (or median split if thin).\n"
            "- **Long leg.** Equal-weight the safe tail (least-negative VaR).\n"
            "- **Short leg.** Equal-weight the crashed tail (most-negative VaR).\n"
            "- **Execution lag.** Enter the close one trading day after the month-end signal; "
            "hold the next calendar month. No same-bar fill.\n"
            "- **Inference.** One-sample *t* on the monthly spread (bar |t| >= 2), HAC *t* "
            "alongside, a 200-draw placebo label-shuffle null.\n"
            "- **Costs.** 5 bps/leg x turnover + 50 bps/yr borrow on the short leg; gross vs net.\n"
            "- **Universe caveat.** 48 large-cap survivors -- survivorship-biased."
        ),

        # ---- BEAT 4 -- TEARDOWN ----------------------------------------------
        md("## 4 -- The teardown"),
        md(
            "### 4a -- Positive control: the engine is a faithful detector (seed-robust)\n\n"
            "Plant a left-tail-momentum premium across many seeds and average the one-sample t "
            "(Welch-style, so no claim rests on one lucky RNG). The mean should rise with the "
            "premium and stay positive. (3 seeds for the figure; docs/results.md uses 20.)"
        ),
        code(
            "ctrl = st.synthetic_control(data, premiums=(0.0, 0.04, 0.08), n_seeds=3,\n"
            "                            kind='var5', lookback=252, base_seed=505)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if m > 0 else RED for m in ctrl['mean_ls_pct']]\n"
            "ax.bar(ctrl['premium'].astype(str), ctrl['mean_ls_pct'], color=col)\n"
            "for i, (m, t) in enumerate(zip(ctrl['mean_ls_pct'], ctrl['mean_t'])):\n"
            "    ax.text(i, m + 0.15, f't={t:+.2f}', ha='center', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted premium'); ax.set_ylabel('long-safe/short-crashed (%/yr)')\n"
            "ax.set_title('Positive control: monotone & correct-signed in the planted premium')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))\n"
            "print('\\nFrozen 20-seed control (docs/results.md):')\n"
            "for p,(m,t,sh) in {0.00:(3.89,0.840,10),0.04:(5.05,1.091,15),0.08:(6.09,1.317,25)}.items():\n"
            "    print(f'  premium {p:+.2f}: mean {m:+.2f}%/yr | avg t {t:+.3f} | share(t>=2) {sh}%')"
        ),
        md(
            "Faithful but **under-powered** at 48 names: even a planted premium of 0.08 averages "
            "*t* ~1.3 (only 25% of seeds clear t>=2). So the engine could only confirm a *large* "
            "real continuation -- yet the real tape is significant in the **opposite** sign, which "
            "no faithful detector confuses for continuation."
        ),
        md("### 4b -- The real long-short and its legs"),
        code(
            "if HAVE_REAL:\n"
            "    ls_m = s_ls['mean']*100; ls_t = s_ls['tstat']; ls_hac = s_ls['hac_t']\n"
            "    lo_m = s_long['mean']*100; sh_m = s_short['mean']*100\n"
            "    net_m = s_net['mean']*100; net_t = s_net['tstat']\n"
            "else:\n"
            "    ls_m, ls_t, ls_hac = R['ls_mean'], R['ls_t'], R['ls_hac']\n"
            "    lo_m, sh_m = R['long_mean'], R['short_mean']\n"
            "    net_m, net_t = R['ls_net'], R['ls_net_t']\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "axes[0].bar(['Long\\n(safe)', 'Short\\n(crashed)'], [lo_m, sh_m],\n"
            "            color=[AMBER, RED], width=0.55)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_ylabel('mean annual return (%/yr)')\n"
            "axes[0].set_title('Crashed leg out-earns the safe leg')\n"
            "\n"
            "# gross vs net spread\n"
            "axes[1].bar(['gross', 'net'], [ls_m, net_m], color=[RED, RED], width=0.5, alpha=0.85)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('long-safe/short-crashed (%/yr)')\n"
            "axes[1].set_title(f'ABDG book loses: gross {ls_m:+.1f}% (t={ls_t:+.2f}), net {net_m:+.1f}%')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Spread gross {ls_m:+.2f}%/yr | one-sample t {ls_t:+.3f} | HAC t {ls_hac:+.3f}')\n"
            "print(f'Spread net   {net_m:+.2f}%/yr | one-sample t {net_t:+.3f}')"
        ),
        md(
            f"> The spread is **{R['ls_mean']:.2f}%/yr** at one-sample *t* = **{R['ls_t']:.3f}** "
            f"(HAC {R['ls_hac']:.3f}). Costs barely move it (turnover ~{R['turnover']:.0f}%/mo). "
            "The signal is real -- and the *reverse* of ABDG."
        ),
        md("### 4c -- The placebo: is the (wrong-sign) signal real, or noise?"),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(prices, kind='var5', lookback=252, min_stocks=20,\n"
            "                           n_shuffles=200, seed=505)\n"
            "    real_mo = pl['real_mean']*100; pval = pl['p_value']\n"
            "else:\n"
            "    real_mo, pval = R['placebo_real_mo'], R['placebo_p']\n"
            "print(f'Real mean spread/month: {real_mo:+.3f}%')\n"
            "print(f'Placebo p-value (200 random sorts): {pval:.3f}')\n"
            "print('Interpretation: the real left-tail sort is more extreme (negative) than'\n"
            "      f' {(1-pval)*100:.0f}% of random sorts -> the reversal is real, not noise.')"
        ),
        md("### 4d -- Flavour & lookback sweep -- is the inversion robust?"),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for kind in ('var5', 'var1', 'worst'):\n"
            "        b = st.long_short_book(prices, kind=kind, lookback=252, min_stocks=20)\n"
            "        s = st.summary(b['ls_ret'])\n"
            "        rows.append({'spec': kind, 'mean_%': s['mean']*100, 't': s['tstat']})\n"
            "    for lb in (126, 252, 504):\n"
            "        if len(prices) > lb:\n"
            "            b = st.long_short_book(prices, kind='var5', lookback=lb, min_stocks=20)\n"
            "            s = st.summary(b['ls_ret'])\n"
            "            rows.append({'spec': f'lb{lb}', 'mean_%': s['mean']*100, 't': s['tstat']})\n"
            "else:\n"
            "    for k,(m,t) in R['sweep'].items(): rows.append({'spec': k, 'mean_%': m, 't': t})\n"
            "    for lb,(m,t) in R['lookback'].items(): rows.append({'spec': f'lb{lb}', 'mean_%': m, 't': t})\n"
            "sweep = pd.DataFrame(rows)\n"
            "fig, ax = plt.subplots(figsize=(10, 4.3))\n"
            "ax.bar(sweep['spec'], sweep['mean_%'], color=RED, alpha=0.85)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (m, t) in enumerate(zip(sweep['mean_%'], sweep['t'])):\n"
            "    ax.text(i, m - 1.2, f't={t:+.1f}', ha='center', fontsize=8)\n"
            "ax.set_ylabel('long-safe/short-crashed (%/yr)')\n"
            "ax.set_title('Every signal flavour and lookback inverts (all |t| > 2.7)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(sweep.round(3).to_string(index=False))"
        ),
        md("### 4e -- Equity curve and the -96% drawdown"),
        code(
            "if HAVE_REAL:\n"
            "    eq = (1 + book['ls_ret']).cumprod()\n"
            "    dd = (eq / eq.cummax() - 1) * 100\n"
            "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)\n"
            "    ax1.plot(book.index, eq.values, c=RED, lw=2)\n"
            "    ax1.axhline(1, c='k', lw=1, ls='--')\n"
            "    ax1.set_ylabel('cumulative wealth (ABDG book)')\n"
            "    ax1.set_title('The continuation book bleeds out -- long-safe/short-crashed')\n"
            "    ax2.fill_between(book.index, dd.values, 0, color=RED, alpha=0.4)\n"
            "    ax2.set_ylabel('drawdown (%)'); ax2.set_xlabel('date')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'Max drawdown: {dd.min():.1f}% | Hit rate: {s_ls[\"hit_rate\"]*100:.1f}%')\n"
            "else:\n"
            "    print(f'Frozen: max DD = {R[\"ls_dd\"]:.1f}% | hit rate = {R[\"ls_hit\"]:.1f}%')"
        ),
        md("### 4f -- Year-by-year"),
        code(
            "if HAVE_REAL:\n"
            "    bc = book.copy(); bc['year'] = bc.index.year\n"
            "    annual = bc.groupby('year')['ls_ret'].apply(lambda x: float((1+x).prod()-1))*100\n"
            "    yrs, vals = list(annual.index), list(annual.values)\n"
            "else:\n"
            "    yrs, vals = list(R['annual'].keys()), list(R['annual'].values())\n"
            "fig, ax = plt.subplots(figsize=(10, 4.0))\n"
            "col = [GREEN if v > 0 else RED for v in vals]\n"
            "ax.bar([str(y) for y in yrs], vals, color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('ABDG book return (%/yr)')\n"
            "ax.set_title('Loses in 11 of 12 years -- the lone winner (2022) is the bear market')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({y: round(v,1) for y, v in zip(yrs, vals)})"
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `NONE`** -- the ABDG continuation inverts: {R['ls_mean']:.2f}%/yr at "
            f"one-sample *t* = {R['ls_t']:.3f} (HAC {R['ls_hac']:.3f}), placebo p = "
            f"{R['placebo_p']:.3f}. Real, robust -- but *reversal*, not momentum. On the claim "
            "as written, NONE.\n"
            f"- **Tradability `MIRAGE`** -- Sharpe {R['ls_sharpe']:.3f}, max DD {R['ls_dd']:.1f}%, "
            "the short leg is the decade's winners; costs are immaterial to a wrong-sign book.\n"
            "- **Left tail keeps losing? `BUSTED`** -- crashed survivors rebound; the book loses "
            "11 of 12 years."
        ),

        # ---- BEAT 6 -- TRADABILITY -------------------------------------------
        md(
            "## 6 -- Could you trade it?\n\n"
            "No -- and the *reason* is the lesson. ABDG's continuation needs the names that "
            "crashed and **stayed down** (often into delisting). A survivor basket deletes them "
            "by construction, leaving 'crashed' names that are disproportionately rebounds, so "
            "the sort picks up **short-run reversal** instead. The reversal is itself thin and "
            "hard to monetise net of costs in liquid large-caps. There is no deployable trade in "
            "either direction here -- the wrong-sign continuation, nor the thin reversal it masks."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Delisting-complete universe.** A CRSP/Compustat panel with delisting returns "
            "is the only fair test of H1 -- the short leg is the crash-into-delisting names.\n"
            "- **Costly-arbitrage tilt.** ABDG's effect concentrates in illiquid, high-idio-vol "
            "names; large-caps are the worst place to look.\n"
            "- **Expected Shortfall.** ABDG also use ES (average of the tail) -- swap "
            "`left_tail_signal` for a mean-below-quantile variant.\n"
            "- **[Study 53 -- Jackpot](../../53-jackpot/)** & **[365 -- Lottery-MAX](../../365-lottery-max-effect/)**: "
            "the right-tail mirror, both None x Mirage on survivors.\n"
            "- **[Study 332 -- Downside-Beta](../../332-downside-beta/)** and "
            "**[238 -- Betting-Against-Beta](../../238-betting-against-beta/)**: cousin "
            "cross-sectional-risk factors on the same desk engine.\n\n"
            "*Think left-tail momentum holds once you add the delisted crashers and tilt to "
            "illiquid names? Fork this, swap the universe, and show t > 2 in the right direction "
            "net of borrow. That is the bar.*"
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
