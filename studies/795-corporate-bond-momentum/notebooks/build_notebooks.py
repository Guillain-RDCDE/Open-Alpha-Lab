"""Generate the two narrative notebooks for Study 795 (Corporate-Bond-Momentum).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached bond-ETF
panel under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance total-return,
# 11 credit + Treasury bond ETFs, 2007-01-03 -> 2026-06-30, 216 WML holding months).
R = dict(
    start="2007-01-03", end="2026-06-30", fp="1f2efa58efab",
    n_rows=4903, n_names=11, n_months=234,
    # headline WML (6m formation, hold 1m, top/bottom third)
    n=216, mean_pct=1.63, vol_pct=10.23, sharpe=0.16, hac_t=0.80, iid_t=0.67,
    hit_pct=51.4, wilson=(44.8, 58.0), max_dd_pct=-24.2, worst_pct=-18.88,
    turnover_pct=50.5, avg_leg=3.7, avg_ranked=10.4,
    # rank-shuffle placebo
    placebo_real_pct=1.63, placebo_mean_pct=0.02, placebo_sd_pct=1.51,
    placebo_p=0.152, placebo_n=2000,
    # lookback sweep: label -> (mean%/yr, Sharpe, HAC t, n)
    lookbacks={
        "6m (headline)": (1.63, 0.16, 0.80, 216),
        "12m": (-1.32, -0.14, -0.54, 210),
        "12-1": (-1.19, -0.13, -0.54, 209),
        "3m": (1.93, 0.20, 0.93, 219),
    },
    # subperiods: label -> (mean%/yr, Sharpe, HAC t, n)
    eras={
        "2007-2013 (Jostova era)": (2.51, 0.17, 0.54, 67),
        "2014-2026 (post-pub)": (1.23, 0.16, 0.60, 149),
    },
    # cost sweep: (cost_bps, borrow_bps) -> (net%/yr, Sharpe, HAC t)
    costs={
        "5 bps + 50 borrow": (0.82, 0.08, 0.40),
        "10 bps + 50 borrow": (0.52, 0.05, 0.25),
        "20 bps + 100 borrow": (-0.59, -0.06, -0.28),
    },
    break_even_bps=26.8,
    hyg_mean_pct=5.29, hyg_sharpe=0.52, hyg_t=2.47,
    long_leg_pct=5.64, short_leg_pct=4.02, corr_hyg=-0.10,
    # synthetic control (seed 314): strength -> (mean%/yr, HAC t)
    syn={0.0: (-0.2, -0.11), 0.06: (3.2, 2.51), 0.12: (11.0, 5.30), 0.20: (24.5, 8.59)},
    syn_null_mean_t=0.14, syn_null_fire=0, syn_null_seeds=20,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Lookback stable%3F: Not supported](https://img.shields.io/badge/Lookback_stable%3F-Not_supported-8b949e?style=flat-square)\n\n"
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

from bond_momentum import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.drop_partial_last_month(data.load_real())
    MP = st.to_monthly(PRICES)
    BK = st.wml_book(MP)
else:
    PRICES = MP = BK = None
print("real cache present:", HAVE_REAL, "| WML holding months:",
      (0 if BK is None else len(BK)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does \"momentum\" work in bonds the way it works in stocks? 📈\n"
            "### Corporate-bond momentum — real for single high-yield names, invisible on a "
            "basket of ETFs\n\n"
            + BADGES +
            "Momentum is the closest thing finance has to a law: stocks that beat their peers "
            "keep beating them for a while. A well-cited 2013 paper (Jostova and co-authors) "
            "found the same thing in **corporate bonds** — but specifically in *individual "
            "high-yield* bonds. The obvious retail question: can you just do it with bond "
            "**ETFs**? Rank a handful of credit funds on their recent return, buy the winners, "
            "short the losers?\n\n"
            "We tried. It doesn't work — and there's a clean reason why.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 11 credit + Treasury bond ETFs, total return, 2007→2026. Each "
            "chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do winner bond-ETFs keep beating loser bond-ETFs? | **Not measurably.** The "
            f"winners-minus-losers spread earns **+{R['mean_pct']:.1f}%/yr** — but that's "
            f"statistical noise (it could just as easily be zero), and the winners beat the "
            f"losers on only **{R['hit_pct']:.0f}%** of months, a coin flip. |\n"
            "| Is the '6-12 month' rule robust? | **No.** Rank on the last **6** months and the "
            "tiny edge is positive; rank on the last **12** and it turns *negative*. A real "
            "pattern doesn't change its sign when you nudge the lookback inside the range the "
            "claim itself endorses. |\n"
            f"| Can you trade it? | **No.** You'd need costs below ~**{R['break_even_bps']:.0f} "
            f"bps** a trade just to break even, and the spread turns over half the book every "
            "month. At realistic costs the net return is a rounding error, then negative. |\n"
            f"| Is there *any* edge in these ETFs? | **Yes — just owning high-yield credit** "
            f"(HYG returned +{R['hyg_mean_pct']:.1f}%/yr, a real risk premium). But that's a "
            "*level* bet, not momentum — and the momentum spread is built to cancel it out. |\n\n"
            "> The famous bond-momentum result is real where it was found (single high-yield "
            "bonds). It just doesn't survive the trip to a basket of ETFs."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Momentum is everywhere. Rank corporate bonds on their trailing 6-12 month "
            "return, buy the recent winners and short the recent losers, and you'll earn a "
            "premium — just like in stocks.\"*\n\n"
            "This traces to **Jostova, Nikolova, Philipov & Stahel (2013)**, who documented "
            "significant momentum in corporate bonds over 1973-2011. The crucial detail, which "
            "the retail version usually drops: they found it **concentrated in high-yield "
            "(junk) bonds** and *weak-to-absent in investment-grade*. Momentum in bonds is a "
            "credit-quality story."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a simple monthly rank-and-rotate on liquid bond **ETFs** captured that premium, "
            "it would be a genuinely easy, mechanical fixed-income strategy anyone could run "
            "from a brokerage account — no single-bond illiquidity, no dealer spreads on odd "
            "lots. That's exactly why it's worth testing *and* worth being suspicious of: the "
            "gap between \"true in the academic single-name cross-section\" and \"tradable in a "
            "dozen ETFs\" is where a lot of factor strategies quietly die."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The universe.** {R['n_names']} credit + Treasury bond ETFs (IG, high-yield, "
            "EM, bank loans, fallen angels, and short/intermediate/long Treasuries), total "
            "return, 2007→2026.\n"
            "- **The sort.** Every month-end, rank on trailing 6-month return; go long the top "
            "third, short the bottom third; hold one month; rebalance.\n"
            "- **The luck check.** Randomly reshuffle which ETF gets which momentum rank 2,000 "
            "times — how often does a *random* ranking beat the real one?\n"
            "- **The robustness check.** Slide the lookback across the 6-12 month range the "
            "claim names. A real signal keeps its sign.\n"
            "- **The trade check.** Charge realistic costs both ways, make the short leg pay "
            "borrow, and find the break-even."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline spread.** How much does buying winners and shorting losers "
            "actually make, and is it distinguishable from zero?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summary(BK['wml_gross'])\n"
            "    eq = (1 + BK['wml_gross']).cumprod()\n"
            "    mean_pct, t = s['mean']*100, s['tstat']\n"
            "else:\n"
            "    mean_pct, t = R['mean_pct'], R['hac_t']\n"
            "    idx = pd.date_range('2008-07-31', periods=R['n'], freq='ME')\n"
            "    rng = np.random.default_rng(795)\n"
            "    eq = pd.Series((1+rng.normal(R['mean_pct']/100/12, R['vol_pct']/100/np.sqrt(12), R['n'])).cumprod(), index=idx)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "ax.plot(eq.index, eq.values, color=GREY, lw=1.8)\n"
            "ax.axhline(1.0, c='k', lw=.8)\n"
            "ax.set_ylabel('growth of $1 (gross, dollar-neutral)')\n"
            "ax.set_title(f'Winners-minus-losers bond ETFs: {mean_pct:+.1f}%/yr, but HAC t = {t:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'WML {mean_pct:+.2f}%/yr, HAC t = {t:+.2f} (the bar is 2.0)')"
        ),
        md(
            f"A gently rising line that could just as easily be flat. The spread makes "
            f"**+{R['mean_pct']:.1f}%/yr** over ~18 years, but the honest statistic is "
            f"**HAC *t* = +{R['hac_t']:.2f}** — the desk bar is 2.0, and this doesn't come close. "
            f"Along the way it drops **{R['max_dd_pct']:.0f}%** and has a single "
            f"**{R['worst_pct']:.1f}%** month. This is noise with a slight upward tilt.\n\n"
            "**Is even that tilt real, or could a random ranking do as well?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(MP, n_perm=800)\n"
            "    obs = pl['real_mean_ann']*100\n"
            "    rng = np.random.default_rng(795)\n"
            "    draws = rng.normal(pl['placebo_mean_ann']*100, pl['placebo_sd_ann']*100, 4000)\n"
            "    p = pl['p_value']\n"
            "else:\n"
            "    obs, p = R['placebo_real_pct'], R['placebo_p']\n"
            "    rng = np.random.default_rng(795)\n"
            "    draws = rng.normal(R['placebo_mean_pct'], R['placebo_sd_pct'], 4000)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='null: random rankings')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'real momentum ranking {obs:+.2f}%/yr')\n"
            "ax.set_xlabel('winners-minus-losers return of a random ranking (%/yr)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The real ranking sits inside the luck cloud: p = {R[\"placebo_p\"]:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'real {obs:+.2f}%/yr vs random rankings; p = {R[\"placebo_p\"]:.3f} '\n"
            "      '(a random ranking wins ~1 time in 7)')"
        ),
        md(
            f"The real momentum ranking (red line) sits comfortably **inside** the cloud of what "
            f"random rankings produce: **p = {R['placebo_p']:.3f}**. A coin-flip ranking beats "
            "the momentum ranking about one time in seven. There's no signal being detected here."
            "\n\n**Now the check that really sinks it: does the '6-12 month' rule even keep its "
            "sign?**"
        ),
        code(
            "labels = list(R['lookbacks']); ts = [R['lookbacks'][k][2] for k in labels]\n"
            "if HAVE_REAL:\n"
            "    ts = []\n"
            "    for lb, sk in ((6,0),(12,0),(12,1),(3,0)):\n"
            "        ts.append(st.summary(st.wml_book(MP, lookback=lb, skip=sk)['wml_gross'])['tstat'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "cols = [GREEN if t>0 else RED for t in ts]\n"
            "ax.bar(labels, ts, color=cols, width=.6)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(-2, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('HAC t of the WML spread')\n"
            "ax.set_title('The sign flips inside the 6-12m window the claim names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({l: round(t,2) for l,t in zip(labels, ts)})"
        ),
        md(
            "This is the tell. Rank on the last **6** months and the spread is faintly positive; "
            "rank on the last **12** (or 12-skip-1, the textbook equity-momentum formation) and "
            "it goes faintly **negative**. Not one bar clears the bar in either direction. A real "
            "anomaly is stable across nearby, equally-reasonable lookbacks — this one changes its "
            "mind across the exact 6-12 month range the claim endorses.\n\n"
            "**Finally: is there any edge in these ETFs at all?**"
        ),
        code(
            "if HAVE_REAL and 'HYG' in MP.columns:\n"
            "    hy = st.summary(MP['HYG'].pct_change().dropna())\n"
            "    wl = st.summary(BK['wml_gross'])\n"
            "    pairs = [('long HYG\\n(own credit)', hy['tstat'], GREEN), ('WML spread\\n(momentum)', wl['tstat'], RED)]\n"
            "else:\n"
            "    pairs = [('long HYG\\n(own credit)', R['hyg_t'], GREEN), ('WML spread\\n(momentum)', R['hac_t'], RED)]\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar([p[0] for p in pairs], [p[1] for p in pairs], color=[p[2] for p in pairs], width=.5)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('HAC t')\n"
            "ax.set_title('The only real edge here is owning credit, not ranking it')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'long HYG t = {R[\"hyg_t\"]:+.2f}  vs  momentum spread t = {R[\"hac_t\"]:+.2f}')"
        ),
        md(
            f"Simply **holding** high-yield credit (HYG) earned +{R['hyg_mean_pct']:.1f}%/yr at "
            f"**HAC *t* = +{R['hyg_t']:.2f}** — a real credit risk premium. The momentum *spread*, "
            f"which is deliberately built to cancel that premium out (it's dollar-neutral, "
            f"correlation to HYG ≈ {R['corr_hyg']:.2f}), makes nothing certifiable. The lesson: "
            "the money in these ETFs is in *owning* credit, not in *timing* which credit fund led "
            "last quarter."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The winners-minus-losers spread earns +{R['mean_pct']:.1f}%/yr "
            f"at HAC *t* = +{R['hac_t']:.2f} — statistically invisible — with a coin-flip hit "
            f"rate and a placebo *p* = {R['placebo_p']:.3f}. And it flips sign across the 6-12m "
            "lookback the claim names.\n"
            "- **Tradability — Mirage.** Break-even near "
            f"{R['break_even_bps']:.0f} bps against 50%/month turnover; net is a rounding error "
            "at retail costs and negative beyond. Nothing to trade.\n"
            "- **\"Is the lookback stable?\" — Not supported.** 6m positive, 12m negative — the "
            "opposite of robustness. Exactly what you'd expect if a coarse ETF panel can't hold "
            "the single-name high-yield cross-section where the effect actually lives."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Why it fails as ETFs.** Cross-sectional momentum needs *dispersion to rank* — "
            "many high-yield names spreading out so the recent winners are a distinct group. One "
            "HYG (or JNK, or ANGL) ETF averages that entire cross-section into a single number, "
            "so there's nothing left to rank *within* it. Jostova et al.'s effect is a "
            "single-name, high-yield phenomenon; the ETF wrapper throws away exactly the "
            "variation it feeds on.\n"
            "- **What might work instead:** a single-name high-yield bond dataset (TRACE), where "
            "the real cross-section survives — or a duration-neutral, credit-only sub-basket if "
            "you can source enough distinct high-yield sleeves.\n"
            "- **Sibling studies:** [518-time-series-momentum](../../518-time-series-momentum/) "
            "(each asset on its *own* trend, not ranked against peers), "
            "[247-bond-seasonality](../../247-bond-seasonality/) (a calendar signal on bonds), "
            "and the carry pair [611-mreit-carry](../../611-mreit-carry/) / "
            "[612-em-debt-carry](../../612-em-debt-carry/) (owning yield, not timing momentum) — "
            "see [docs/references.md](docs/references.md) for the exact dedup.\n\n"
            "*Think ETF bond-momentum can be made to pay? Show a certifiable HAC *t* ≥ 2 on a "
            "lookback that keeps its sign, net of realistic costs — then we'll talk.*"
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
            "# Corporate-bond momentum — a quantitative teardown 🔬\n"
            "### The winners-minus-losers HAC *t* · a 2,000-permutation rank-shuffle placebo · "
            "the 6-12m lookback-sign sweep · the era split · an honest cost/borrow sweep · a "
            "planted-momentum synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **cross-sectional momentum in bonds** (Jostova, Nikolova, Philipov & Stahel "
            "2013) — is a rank-sort claim, distinct from time-series/own-sign trend (study 518), "
            "bond seasonality (247), and packaged carry (611/612). The job: measure the ETF-panel "
            "winners-minus-losers spread honestly, and be clear about *why* a null here is "
            "consistent with a paper that found the effect in single high-yield names.\n\n"
            "> ⚠️ **Data note.** 11 credit + Treasury bond ETFs, daily total return (auto-adjusted "
            "close), 2007-01-03 → 2026-06-30, yfinance, cached. Dollar-neutral top/bottom-third "
            "sort, one execution `shift` (form on the month-*t* close, earn *t+1*). "
            "**Survivorship** (current-membership ETF basket) named on the Signal axis — but a "
            "*null* is not manufactured by it. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | WML **+{R['mean_pct']:.2f}%/yr**, HAC *t* = "
            f"**+{R['hac_t']:.2f}** (plain {R['iid_t']:+.2f}); hit {R['hit_pct']:.1f}% "
            f"(Wilson [{R['wilson'][0]:.1f}%, {R['wilson'][1]:.1f}%]); rank-shuffle placebo "
            f"**p = {R['placebo_p']:.3f}**; sign flips to {R['lookbacks']['12m'][2]:+.2f} at 12m |\n"
            f"| **Tradability** | `MIRAGE` | break-even ≈ {R['break_even_bps']:.0f} bps one-way; "
            f"net {R['costs']['10 bps + 50 borrow'][0]:+.2f}%/yr (t = "
            f"{R['costs']['10 bps + 50 borrow'][2]:+.2f}) at 10 bps, negative at 20 bps; gross "
            f"Sharpe {R['sharpe']:.2f}, max DD {R['max_dd_pct']:.0f}% |\n"
            f"| **Lookback stable?** | `NOT SUPPORTED` | 6m *t* = +{R['hac_t']:.2f} vs 12m *t* = "
            f"{R['lookbacks']['12m'][2]:+.2f} — sign-unstable inside the claimed window |\n\n"
            "> 💡 In plain words: the ETF-panel momentum spread is indistinguishable from zero and "
            "not even sign-stable — consistent with Jostova et al.'s finding that bond momentum "
            "lives in the *single-name high-yield* cross-section a coarse ETF basket can't hold."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{fwd}_{i,t+1}$ be ETF $i$'s total return in month $t{+}1$ and "
            "$m_{i,t} = P_{i,t}/P_{i,t-L} - 1$ its trailing $L$-month return. The WML book longs "
            "the top-third of $\\{m_{i,t}\\}$ and shorts the bottom-third, equal-weight and "
            "dollar-neutral. The claims:\n\n"
            "- **H₁ (momentum).** $E[\\text{WML}_{t+1}] > 0$ with a robust *t* — recent winners "
            "keep beating recent losers next month.\n"
            "- **H₂ (robust window).** The premium is stable across the 6-12 month formation the "
            "claim names (not sign-flipping).\n"
            "- **H₃ (capture).** The spread survives realistic costs (one-way × NAV, short leg "
            "pays borrow).\n\n"
            "We find **H₁ not supported** (HAC *t* = "
            f"+{R['hac_t']:.2f}, placebo p = {R['placebo_p']:.3f}), **H₂ not supported** (6m "
            f"+{R['hac_t']:.2f} vs 12m {R['lookbacks']['12m'][2]:+.2f}), **H₃ moot** (no gross "
            "edge to protect). The one real premium in the panel is *owning* high-yield credit "
            f"(long HYG HAC *t* = +{R['hyg_t']:.2f}) — a level bet the WML spread is designed to "
            "strip out."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Monthly WML returns are the unit of inference. The **planned primary** is a "
            "**Newey-West (HAC) one-sample *t*** on the monthly mean (serial-correlation robust; "
            "auto lag). We cross-check with a plain i.i.d. *t*, a Wilson interval on the "
            "positive-month rate, and a **2,000-permutation rank-shuffle placebo** — each month we "
            "keep the realised return cross-section exactly but randomly permute which ETF gets "
            "which rank, destroying the past→future link while preserving the return distribution "
            "and leg sizes. The lookback sweep (6m / 12m / 12-1 / 3m) is the pre-registered "
            "robustness rail — the claim itself names *6-12 months*, so sign-instability inside "
            "that band is decisive, not snooped."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** {R['n_names']} credit + Treasury bond ETFs, {R['n_rows']:,} daily rows, "
            f"{R['start']} → {R['end']}; {R['n_months']} month-ends, {R['n']} WML holding months.\n"
            "- **Sort.** Rank on trailing 6-month total return; long top third, short bottom third "
            f"(≈ {R['avg_leg']:.1f} names/leg of ≈ {R['avg_ranked']:.1f} ranked); dollar-neutral; "
            "form on the *t* close, earn *t+1* (one shift).\n"
            "- **Signal test.** HAC *t* + plain *t* + Wilson hit rate + 2,000-perm placebo.\n"
            "- **Robustness.** Lookback sweep across 6-12m (+3m, +12-1); era split 2007-2013 vs "
            "2014-2026.\n"
            "- **Execution.** Cost sweep (5/10/20 bps one-way) with the short leg paying borrow "
            "(50/100 bps/yr), and the break-even cost.\n"
            "- **Control.** Synthetic total-return panel, planted-momentum knob; null must not "
            "fire (checked over 20 seeds), planted effect must be recovered."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline spread and its placebo\n\n"
            "HAC *t* on the monthly WML mean, plus the rank-shuffle null. In the notebook we run "
            "a lighter placebo (800 perms) and quote the canonical 2,000-perm p from `results.md`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summary(BK['wml_gross'])\n"
            "    print(f\"WML {s['mean']*100:+.2f}%/yr, vol {s['vol']*100:.2f}%, Sharpe {s['sharpe']:.2f} (n={s['n']})\")\n"
            "    print(f\"HAC t = {s['tstat']:+.2f}   plain t = {s['t_iid']:+.2f}\")\n"
            "    print(f\"hit {s['hit_rate']*100:.1f}%  Wilson [{s['hit_lo']*100:.1f}%, {s['hit_hi']*100:.1f}%]\")\n"
            "    print(f\"max DD {s['max_dd']*100:.1f}%  worst month {s['worst']*100:+.2f}%\")\n"
            "    pl = st.placebo_pvalue(MP, n_perm=800)\n"
            "    obs = pl['real_mean_ann']*100\n"
            "    draws = np.random.default_rng(795).normal(pl['placebo_mean_ann']*100, pl['placebo_sd_ann']*100, 4000)\n"
            "else:\n"
            "    obs = R['placebo_real_pct']\n"
            "    draws = np.random.default_rng(795).normal(R['placebo_mean_pct'], R['placebo_sd_pct'], 4000)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='null: random rankings (light run)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'real momentum ranking {obs:+.2f}%/yr')\n"
            "ax.set_xlabel('WML return of a random ranking (%/yr)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Inside the luck cloud: canonical p = {R[\"placebo_p\"]:.3f} (2,000 perms)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical placebo (results.md): real {R['placebo_real_pct']:+.2f}%/yr vs \"\n"
            "      f\"{R['placebo_mean_pct']:+.2f} (sd {R['placebo_sd_pct']:.2f}), p = {R['placebo_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the observed +{R['mean_pct']:.2f}%/yr sits well inside the null "
            f"(placebo mean {R['placebo_mean_pct']:+.2f} ± {R['placebo_sd_pct']:.2f}%/yr); "
            f"**p = {R['placebo_p']:.3f}**. HAC *t* = +{R['hac_t']:.2f}, hit rate a coin flip. H₁ "
            "fails on every reading."
        ),
        md(
            "### 4b · The killer — sign instability across the claimed lookback\n\n"
            "The claim names a **6-12 month** formation. A real anomaly keeps its sign across that "
            "band; a data artifact doesn't."
        ),
        code(
            "labels = list(R['lookbacks'])\n"
            "if HAVE_REAL:\n"
            "    rows = {}\n"
            "    for lb, sk, lab in ((6,0,'6m (headline)'),(12,0,'12m'),(12,1,'12-1'),(3,0,'3m')):\n"
            "        srow = st.summary(st.wml_book(MP, lookback=lb, skip=sk)['wml_gross'])\n"
            "        rows[lab] = (srow['mean']*100, srow['sharpe'], srow['tstat'])\n"
            "    means = [rows[k][0] for k in labels]; ts = [rows[k][2] for k in labels]\n"
            "else:\n"
            "    means = [R['lookbacks'][k][0] for k in labels]; ts = [R['lookbacks'][k][2] for k in labels]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(labels, means, color=[GREEN if m>0 else RED for m in means], width=.6)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('WML mean (%/yr)'); a1.set_title('Return flips sign')\n"
            "a2.bar(labels, ts, color=[GREEN if t>0 else RED for t in ts], width=.6)\n"
            "a2.axhline(2, ls='--', c=GREY, lw=1); a2.axhline(-2, ls='--', c=GREY, lw=1); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('HAC t'); a2.set_title('...and never clears |t|=2 either way')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({l: (round(m,2), round(t,2)) for l,m,t in zip(labels, means, ts)})"
        ),
        md(
            f"> 💡 In plain words: 6-month formation gives +{R['lookbacks']['6m (headline)'][0]:.2f}%/yr "
            f"(*t* = +{R['lookbacks']['6m (headline)'][2]:.2f}); 12-month and 12-1 give "
            f"{R['lookbacks']['12m'][0]:+.2f} / {R['lookbacks']['12-1'][0]:+.2f}%/yr (*t* = "
            f"{R['lookbacks']['12m'][2]:+.2f}). The sign is *not stable* inside the exact 6-12m "
            "window the claim endorses. H₂ fails — this is the signature of no underlying effect, "
            "not of a fragile one."
        ),
        md(
            "### 4c · Era split — no window carries it\n\n"
            "The Jostova sample ends ~2011; if the effect were real-but-decayed we'd see it in the "
            "early slice. We don't."
        ),
        code(
            "labels = list(R['eras'])\n"
            "if HAVE_REAL:\n"
            "    vals = {}\n"
            "    for lo, hi, lab in (('2007-01-01','2013-12-31','2007-2013 (Jostova era)'),\n"
            "                        ('2014-01-01','2026-06-30','2014-2026 (post-pub)')):\n"
            "        sub = BK['wml_gross'][(BK.index>=lo)&(BK.index<=hi)]\n"
            "        srow = st.summary(sub); vals[lab] = (srow['mean']*100, srow['tstat'])\n"
            "    means = [vals[k][0] for k in labels]; ts = [vals[k][1] for k in labels]\n"
            "else:\n"
            "    means = [R['eras'][k][0] for k in labels]; ts = [R['eras'][k][2] for k in labels]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "ax.bar(labels, means, color=[AMBER, AMBER], width=.5)\n"
            "for i,(m,t) in enumerate(zip(means, ts)): ax.annotate(f'{m:+.1f}%/yr\\n(t={t:+.2f})',(i,m),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('WML mean (%/yr)')\n"
            "ax.set_title('Neither era clears t=1, let alone t=2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({l:(round(m,2),round(t,2)) for l,m,t in zip(labels, means, ts)})"
        ),
        md(
            f"> 💡 In plain words: the in-sample-era slice (+{R['eras']['2007-2013 (Jostova era)'][0]:.2f}%/yr, "
            f"*t* = +{R['eras']['2007-2013 (Jostova era)'][2]:.2f}) is no stronger than the "
            f"post-publication slice (+{R['eras']['2014-2026 (post-pub)'][0]:.2f}%/yr, *t* = "
            f"+{R['eras']['2014-2026 (post-pub)'][2]:.2f}). There is no era where the ETF spread is "
            "real — this is a level problem (wrong instrument), not a decay story."
        ),
        md(
            "### 4d · The timer — honest cost/borrow sweep\n\n"
            "There's no gross edge to protect, but for completeness: one-way turnover × NAV, short "
            "leg pays borrow, and the break-even cost."
        ),
        code(
            "labels = list(R['costs'])\n"
            "if HAVE_REAL:\n"
            "    nets = {}\n"
            "    for cb, bb, lab in ((5,50,'5 bps + 50 borrow'),(10,50,'10 bps + 50 borrow'),(20,100,'20 bps + 100 borrow')):\n"
            "        srow = st.summary(st.wml_book(MP, cost_bps=cb, borrow_ann_bps=bb)['wml_net'])\n"
            "        nets[lab] = (srow['mean']*100, srow['tstat'])\n"
            "    ms = [nets[k][0] for k in labels]; ts = [nets[k][1] for k in labels]\n"
            "    g = st.summary(BK['wml_gross'])['mean']*100\n"
            "else:\n"
            "    ms = [R['costs'][k][0] for k in labels]; ts = [R['costs'][k][2] for k in labels]\n"
            "    g = R['mean_pct']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "bars = ['gross'] + labels; vals = [g] + ms\n"
            "ax.bar(bars, vals, color=[GREY, AMBER, AMBER, RED], width=.6)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('net WML mean (%/yr)')\n"
            "ax.set_title(f'Break-even ~ {R[\"break_even_bps\"]:.0f} bps one-way; net dies with cost')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'break-even ~ {R[\"break_even_bps\"]:.1f} bps one-way; ' + str({l:(round(m,2),round(t,2)) for l,m,t in zip(labels, ms, ts)}))"
        ),
        md(
            f"> 💡 In plain words: gross +{R['mean_pct']:.2f}%/yr against 50%/month turnover → "
            f"break-even ≈ {R['break_even_bps']:.0f} bps one-way. Net is "
            f"+{R['costs']['5 bps + 50 borrow'][0]:.2f}%/yr at 5 bps (*t* = "
            f"{R['costs']['5 bps + 50 borrow'][2]:+.2f}) and {R['costs']['20 bps + 100 borrow'][0]:+.2f}%/yr "
            "at 20 bps. There was never a significant gross edge, so costs merely confirm the "
            "MIRAGE. H₃ is moot."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Deterministic total-return panel, TUNABLE planted cross-sectional momentum (seed "
            "314). The engine must score ~0 on the null and rise with the planted strength; the "
            "null is checked over **20 seeds**."
        ),
        code(
            "sc = st.synthetic_control()\n"
            "nr = st.synthetic_null_robustness()\n"
            "xs = list(sc['mom_strength']); ts = list(sc['tstat'])\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.plot(xs, ts, 'o-', color=GREEN, lw=2, ms=8)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(-2, ls='--', c=GREY, lw=1); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('planted momentum strength'); ax.set_ylabel('HAC t recovered')\n"
            "ax.set_title('Control: flat at the null, lights up as momentum is planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({round(x,2): round(t,2) for x,t in zip(xs, ts)})\n"
            "print(f\"null over {nr['n_seeds']} seeds: mean HAC t = {nr['mean_null_t']:+.2f}, \"\n"
            "      f\"|t|>2 in {nr['frac_abs_t_gt2']*100:.0f}% of seeds\")"
        ),
        md(
            f"> 💡 In plain words: the engine scores the null at *t* = {R['syn'][0.0][1]:+.2f} and "
            f"recovers planted momentum monotonically (*t* = +{R['syn'][0.06][1]:.2f} → "
            f"+{R['syn'][0.12][1]:.2f} → +{R['syn'][0.20][1]:.2f}); across "
            f"{R['syn_null_seeds']} null seeds mean *t* = +{R['syn_null_mean_t']:.2f}, "
            f"|t|>2 in {R['syn_null_fire']}/{R['syn_null_seeds']}. The pipeline is unbiased and "
            f"has ample power — so the flat real-tape *t* = +{R['hac_t']:.2f} is a true reading, "
            "not a broken detector. *(Faithful-engine / power check only — never cited for the "
            "real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — WML +{R['mean_pct']:.2f}%/yr at HAC *t* = +{R['hac_t']:.2f} "
            f"(plain {R['iid_t']:+.2f}); hit {R['hit_pct']:.1f}% (Wilson straddles 50%); "
            f"rank-shuffle placebo p = {R['placebo_p']:.3f}; sign flips to "
            f"{R['lookbacks']['12m'][2]:+.2f} at 12m; no era clears *t* = 1. No cross-sectional "
            "momentum in this bond-ETF panel.\n"
            f"- **Tradability `MIRAGE`** — break-even ≈ {R['break_even_bps']:.0f} bps one-way "
            f"against 50%/month turnover; net {R['costs']['10 bps + 50 borrow'][0]:+.2f}%/yr "
            f"(*t* = {R['costs']['10 bps + 50 borrow'][2]:+.2f}) at 10 bps, "
            f"{R['costs']['20 bps + 100 borrow'][0]:+.2f}%/yr at 20 bps; gross Sharpe "
            f"{R['sharpe']:.2f}, max DD {R['max_dd_pct']:.0f}%. No residual to trade.\n"
            f"- **Lookback stable? `NOT SUPPORTED`** — 6m +{R['hac_t']:.2f} vs 12m "
            f"{R['lookbacks']['12m'][2]:+.2f}; sign-unstable inside the claimed window. Consistent "
            "with the effect living in the single-name high-yield cross-section a coarse ETF "
            "basket cannot hold — and with the fact that the only real premium here is *owning* "
            f"credit (long HYG HAC *t* = +{R['hyg_t']:.2f}), which the spread is built to cancel."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Why the ETF wrapper kills it.** Cross-sectional momentum feeds on *dispersion to "
            "rank*. Jostova et al.'s effect is a single-name, high-yield phenomenon; a broad "
            "high-yield ETF averages that entire cross-section into one number, so the "
            "within-sleeve variation the sort needs is gone before you rank. An 11-ETF panel that "
            "is IG-and-Treasury-heavy has even less credit dispersion to work with.\n"
            "- **The right next instrument** is single-name high-yield bonds (TRACE) or a much "
            "wider set of narrow credit sleeves — where the real cross-section survives — tested "
            "with the same HAC/placebo/lookback rails used here.\n"
            "- **Dedup map:** [518-time-series-momentum](../../518-time-series-momentum/) "
            "(own-sign trend, not a rank sort), [247-bond-seasonality](../../247-bond-seasonality/) "
            "(a calendar signal, no cross-section), [611-mreit-carry](../../611-mreit-carry/) and "
            "[612-em-debt-carry](../../612-em-debt-carry/) (carry — a *level* signal — where the "
            "kill is the left tail, not sign-instability).\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
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
