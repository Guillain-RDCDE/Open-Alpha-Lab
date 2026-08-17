"""Generate the two narrative notebooks for Study 947 (The Buffer Ladder).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        01_for_the_curious.ipynb 02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below, which mirrors ``docs/results.md``; the only live cells run the fast
synthetic control and a small closed-form correlation demonstration, and they are labelled
as synthetic wherever they appear. No cell reads the cache, so the notebooks execute on a
fresh checkout.
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


# --------------------------------------------------------------------------- #
# Frozen real-tape headline — mirror of docs/results.md.
# BUFR vs PJAN/PAPR/PJUL/POCT vs SPY/BIL, daily total return, excess-of-cash,
# 2020-08-11 -> 2026-06-30, 5 bps one-way, as-of 2026-06-30.
# --------------------------------------------------------------------------- #
R = dict(
    start="2020-08-11", end="2026-06-30", n_days=1478, n_race=1477, n_matched=1225,
    matched_start="2021-08-12", fp="489cd6cd95e2",

    # excess-of-cash arms: (ann return %, vol %, excess Sharpe, HAC t)
    bufr=(7.93, 10.21, 0.777, 2.06),
    pjan=(6.61, 8.62, 0.767, 2.08),
    papr=(5.32, 7.75, 0.687, 1.82),
    pjul=(7.21, 8.20, 0.879, 2.40),
    poct=(7.33, 7.77, 0.943, 2.65),
    diy=(6.61, 7.74, 0.853, 2.30),
    diy_matched=(7.34, 11.12, 0.661, 1.59),
    mix_bufr=(6.09, 9.80, 0.621, 1.50),
    spy=(13.73, 16.92, 0.812, 2.17),

    # absolute (lived) max drawdowns, %
    dd_bufr=-13.73, dd_diy=-10.82, dd_pjan=-11.93, dd_papr=-11.87,
    dd_pjul=-10.69, dd_poct=-10.22, dd_spy=-24.50,

    # gaps: (pp/yr, HAC t, tracking error %, dSharpe)
    gap_diy=(1.33, 1.18, 3.52, -0.076),
    gap_pjan=(1.32, 1.26, 3.78, 0.010),
    gap_papr=(2.61, 1.74, 4.70, 0.090),
    gap_pjul=(0.73, 0.55, 4.01, -0.102),
    gap_poct=(0.61, 0.40, 4.37, -0.166),
    gap_matched=(-0.96, -1.69, 2.35, -0.057),
    gap_mix=(0.30, 0.31, 2.66, -0.018),
    gap_diy_mix=(1.26, 1.39, 2.29, 0.043),

    beta_bufr=0.579, beta_diy=0.439, corr_bufr_diy=0.960,

    # bootstrap: (point, ci_low, ci_high, frac_negative %)
    boot_diy=(1.33, -0.81, 3.22, 10.2),
    boot_matched=(-0.96, -1.91, -0.05, 98.4),
    boot_sharpe=(-0.076, -0.209, 0.034, 91.2),

    # block-length sensitivity of the ONE CI that excludes zero (beta-matched gap):
    # (block days, mean ci_low, mean ci_high, seeds excluding zero, seeds tried)
    boot_block_sens=((5, -2.20, 0.26, 0, 3), (10, -2.00, 0.06, 0, 3),
                     (21, -1.87, -0.06, 3, 3), (42, -1.81, -0.13, 3, 3),
                     (63, -1.78, -0.14, 3, 3)),

    # entry-point luck
    spread_mean=4.53, spread_median=4.51, spread_max=8.70,
    sd_single_mean=6.77, sd_basket=6.61, var_reduction=2.4, pair_corr=0.889,
    var_reduction_daily=4.2, var_reduction_closed_form=4.2,

    # eras
    era_e=(726, 0.639, 0.768, 1.08, 0.54, -1.22, -1.18),
    era_l=(751, 0.961, 0.945, 1.57, 1.36, -0.79, -1.14),

    # sweeps
    cost0=(1.32, 1.18, -0.96, -1.69), cost25=(1.33, 1.19, -0.95, -1.66),
    fee00=(1.33, 1.18, -0.96, -1.69), fee20=(1.53, 1.36, -0.76, -1.34),
    fee40=(1.73, 1.54, -0.56, -0.98),
    fee_single_pct=0.79, fee_extra_pct=0.20,

    # calendar years: year -> (BUFR, PJAN, PAPR, PJUL, POCT, DIY, spread, SPY)
    years={
        2021: (11.88, 8.80, 7.51, 7.20, 9.45, 8.25, 2.26, 28.73),
        2022: (-7.57, -5.29, -4.29, -2.08, -1.25, -3.20, 4.04, -18.18),
        2023: (19.63, 18.18, 16.45, 19.87, 20.12, 18.66, 3.67, 26.18),
        2024: (14.68, 13.45, 12.28, 13.76, 9.55, 12.26, 4.21, 24.89),
        2025: (12.44, 11.29, 6.58, 12.78, 10.99, 10.40, 6.19, 17.72),
    },

    # synthetic control
    syn_planted=(3.80, 4.16, 0.36, 4.84),
    syn_null_fee=(-0.20, 0.16, 0.36, 0.18),
    syn_null_clean=(0.00, 0.36, 0.36, 0.42),
    syn_null_mean=0.11, syn_null_sd=0.87, syn_null_maxt=1.69, syn_null_fires=0,
)


HEADER = f"""# Study 947 — The Buffer Ladder 🪜

**Does laddering a buffer fund add anything you could not do yourself with four trades?**

A buffer ETF's terms — the stated buffer against losses, the stated cap on gains — are
struck once a year on a named month. So the payoff you actually get depends on *when you
bought*. The industry's answer is the **laddered wrapper**: one ticker (**BUFR**) holding a
spread of vintages so entry timing averages out, for a management fee **on top of** the
underlying funds' expense ratios.

The obvious retort: buy the vintages yourself and equal-weight them. Four trades.

We race **BUFR** against each of its four quarterly vintages (**PJAN / PAPR / PJUL /
POCT**), against an equal-weight **DIY basket** of them, against a **beta-matched** DIY
ladder, and against the dumb **SPY/BIL** mix — all **excess-of-cash**, all total-return,
{R['start']} → {R['end']} ({R['n_race']:,} days), 5 bps one-way.

*Every real number below is the frozen headline from `docs/results.md` (Fingerprint
`{R['fp']}`); the live cells run offline synthetic demonstrations and say so. As-of
2026-06-30.*
"""


# --------------------------------------------------------------------------- #
# Live cells (offline, deterministic, clearly synthetic)
# --------------------------------------------------------------------------- #
IMPORTS = (
    "import os, sys\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "sys.path.insert(0, os.path.abspath(os.path.join('..', '..', '..')))\n"
    "import numpy as np\n"
    "from buffer_ladder import data, strategy as st"
)

CORR_DEMO = (
    "# SYNTHETIC — no market data. How much variance does averaging N legs remove,\n"
    "# as a function of how correlated those legs are? Closed form: for N equally\n"
    "# correlated legs of equal variance, sd(basket)/sd(leg) = sqrt((1 + (N-1)*rho) / N).\n"
    "def variance_reduction(rho, n=4):\n"
    "    return (1.0 - np.sqrt((1.0 + (n - 1) * rho) / n)) * 100.0\n"
    "\n"
    "for rho in (0.0, 0.25, 0.50, 0.75, 0.889, 0.95):\n"
    "    tag = '   <-- the four Power Buffer vintages' if rho == 0.889 else ''\n"
    "    print('correlation %.3f  ->  averaging 4 legs cuts sd by %5.1f%%%s'\n"
    "          % (rho, variance_reduction(rho), tag))"
)

SYNTH_CONTROL = (
    "# SYNTHETIC control — the machinery proof. Never supports the real-tape stamp.\n"
    "for tag, ss, fee in [('planted premium', 1.0, 0.002),\n"
    "                     ('null, fee only ', 0.0, 0.002),\n"
    "                     ('null, no fee   ', 0.0, 0.000)]:\n"
    "    px, truth = data.synthetic_panel(signal_strength=ss, extra_fee_ann=fee, seed=947)\n"
    "    d = st.synthetic_detect(px, truth)\n"
    "    print('%s : planted %+5.2f pp/yr -> recovered %+5.2f (error %+.2f), HAC t %+.2f'\n"
    "          % (tag, d['expected_gap_pp'], d['gap_ann_pp'], d['error_pp'], d['t_hac']))\n"
    "\n"
    "nulls = [st.synthetic_detect(*data.synthetic_panel(signal_strength=0.0,\n"
    "                                                   extra_fee_ann=0.0, seed=947 + s))\n"
    "         for s in range(8)]\n"
    "ts = np.array([n['t_hac'] for n in nulls])\n"
    "gaps = np.array([n['gap_ann_pp'] for n in nulls])\n"
    "print('\\nnull across 8 seeds: mean gap %+.2f pp/yr (sd %.2f), max |t| %.2f, fires on %d/8'\n"
    "      % (gaps.mean(), gaps.std(ddof=1), np.abs(ts).max(), int((np.abs(ts) >= 2).sum())))"
)


def build_curious():
    b, d, sp = R["bufr"], R["diy"], R["spy"]
    gd, gm = R["gap_diy"], R["gap_matched"]
    nb = new_notebook()
    cells = [
        md(HEADER),

        md("## 1. The problem laddering claims to solve\n\n"
           "PJAN resets every January, PAPR every April, and so on. Buy one of them in the "
           "middle of its outcome period and you inherit whatever the market has already "
           "done to that vintage's buffer and cap. Pick the wrong door and you get a "
           "different year from your neighbour who picked the right one.\n\n"
           "So: how big is that difference, really? Here is the gap between the "
           "best-performing and worst-performing vintage over every rolling one-year "
           f"window on the tape.\n\n"
           f"| | Best minus worst vintage, rolling 1 year |\n|---|--:|\n"
           f"| Mean | **{R['spread_mean']:.2f} pp** |\n"
           f"| Median | {R['spread_median']:.2f} pp |\n"
           f"| Worst case | {R['spread_max']:.2f} pp |\n\n"
           "About four and a half percentage points, typically. Not nothing — but hold that "
           "number, because the next section is where the story turns."),

        md("## 2. The catch — the four vintages are almost the same fund\n\n"
           f"The four vintages have a **{R['pair_corr']:.3f}** daily correlation with each "
           "other. They all track the same index with a similar damping; the reset month "
           "shifts the strikes a little, but the underlying is identical.\n\n"
           "That matters enormously, because averaging things that move together removes "
           "almost no risk. Averaging things that move independently removes a lot. Here is "
           "the arithmetic, and it is not a market fact — it is a fact about averages:"),
        code(IMPORTS),
        code(CORR_DEMO),

        md(f"## 3. What that means on the real tape\n\n"
           f"On the real tape the DIY basket's one-year return standard deviation is "
           f"**{R['sd_basket']:.2f}%** against **{R['sd_single_mean']:.2f}%** for the "
           f"average single vintage — a **{R['var_reduction']:.1f}%** reduction. Roughly a "
           f"sixth of a percentage point.\n\n"
           "That is the size of the prize laddering is competing for. And it is free: four "
           "trades and a rebalance reminder get you all of it.\n\n"
           "> 🔬 **For the quants** — the cell above is the equally-correlated-legs closed "
           "form sd(basket)/sd(leg) = √((1 + (N−1)ρ)/N). At ρ = 0.889 and N = 4 it predicts "
           "a **4.2%** cut in daily standard deviation; the tape delivers **4.20%** — the "
           "arithmetic is exact. On a one-year *holding period*, where each vintage's path "
           "through its own buffer and cap matters, the realised cut is smaller still at "
           "**2.4%**. Either way there is nothing left over for the wrapper to have been "
           "clever about."),

        md("## 4. So did the wrapper win anyway?\n\n"
           "On raw return, yes — and this is where it gets interesting."),
        code(
            "R = dict(bufr=%r, diy=%r, spy=%r, gap_diy=%r,\n"
            "         beta_bufr=%r, beta_diy=%r, gap_matched=%r)\n"
            "print('BUFR (the ladder) : %%+.2f%%%% a year, vol %%.2f%%%%, excess Sharpe %%+.3f'\n"
            "      %% R['bufr'][:3])\n"
            "print('DIY basket        : %%+.2f%%%% a year, vol %%.2f%%%%, excess Sharpe %%+.3f'\n"
            "      %% R['diy'][:3])\n"
            "print()\n"
            "print('the wrapper earned %%+.2f pp/yr more ... and its Sharpe is %%+.3f LOWER'\n"
            "      %% (R['gap_diy'][0], R['gap_diy'][3]))"
            % (R["bufr"], R["diy"], R["spy"], R["gap_diy"],
               R["beta_bufr"], R["beta_diy"], R["gap_matched"])
        ),

        md(f"## 5. Because the wrapper is not a better ladder — it is a bolder one\n\n"
           f"BUFR's sensitivity to the S&P 500 is **{R['beta_bufr']:.3f}**. The "
           f"four-vintage DIY basket's is **{R['beta_diy']:.3f}**. The wrapper simply holds "
           f"more equity exposure — and in a five-year stretch where the S&P returned "
           f"{sp[0]:+.1f}% a year excess of cash, more equity exposure means more return.\n\n"
           f"That is not a laddering premium. It is a beta you can buy for the price of an "
           f"index fund. Top the DIY basket up with SPY until it carries the *same* beta as "
           f"the wrapper, and the wrapper **loses** by **{gm[0]:+.2f} pp/yr** "
           f"(HAC *t* = {gm[1]:+.2f}) — about the size of the extra fee layer it charges.\n\n"
           f"The tell is in the risk numbers: the wrapper's worst loss on the tape was "
           f"**{R['dd_bufr']:.1f}%**, against **{R['dd_diy']:.1f}%** for the home-made basket "
           f"and **{R['dd_poct']:.1f}%** for POCT held alone. The product you bought *for "
           f"downside protection* protected you least."),

        md("## 6. The single year that mattered\n\n"
           "There has been exactly one genuine down-year in BUFR's life. Here is how each "
           "arm did in it, alongside the rest of the tape (nominal total return, %):\n\n"
           "| Year | BUFR | PJAN | PAPR | PJUL | POCT | DIY basket | SPY |\n"
           "|---|--:|--:|--:|--:|--:|--:|--:|\n" +
           "\n".join(
               f"| {y} | **{v[0]:+.2f}** | {v[1]:+.2f} | {v[2]:+.2f} | {v[3]:+.2f} | "
               f"{v[4]:+.2f} | **{v[5]:+.2f}** | {v[7]:+.2f} |"
               for y, v in R["years"].items()) +
           "\n\nBUFR out-returned the DIY basket every complete year — and in **2022**, the "
           "only year anyone actually needed a buffer, it lost **more than double** what the "
           "home-made basket lost, and more than every single one of its own constituents. "
           "That one row is the whole beta story in miniature."),

        md("## 7. Is any of this statistically real? (No.)\n\n"
           f"That is the honest headline. **Not one comparison on this tape clears the "
           f"desk's |*t*| = 2 bar, in either direction.**\n\n"
           f"| | Gap | HAC *t* | Bootstrap 95% CI |\n|---|--:|--:|--:|\n"
           f"| Wrapper vs DIY basket | {gd[0]:+.2f} pp/yr | {gd[1]:+.2f} | "
           f"[{R['boot_diy'][1]:+.2f}, {R['boot_diy'][2]:+.2f}] |\n"
           f"| Wrapper vs **beta-matched** DIY | {gm[0]:+.2f} pp/yr | {gm[1]:+.2f} | "
           f"[{R['boot_matched'][1]:+.2f}, {R['boot_matched'][2]:+.2f}] |\n\n"
           "The beta-matched line is the one to be careful with: its bootstrap CI does clear "
           "zero at the 21-day block used above, but flip to a 5- or 10-day block and it "
           "straddles zero again (notebook 02, §5). The exclusion is the block length "
           "talking, so the HAC *t* of −1.69 is what we stamp on.\n\n"
           "Five-point-nine years and one down-year is simply not enough tape to resolve a "
           "one-percentage-point effect. What we *can* say without a *t*-test is the "
           "mechanism: the vintages are nearly the same fund, so laddering them removes "
           "nearly nothing, and the wrapper's visible edge is beta."),

        md("## 8. Live check — the machinery is not broken (offline synthetic)\n\n"
           "Before believing a null, check the detector can find something. On a synthetic "
           "panel with a *planted* laddering premium it recovers it cleanly; on a null panel "
           "it stays quiet. Nothing below touches market data."),
        code(SYNTH_CONTROL),

        md(f"## Verdict\n\n"
           f"- **Signal — None.** No comparison clears |*t*| = 2: {gd[0]:+.2f} pp/yr vs the "
           f"DIY basket (*t* = {gd[1]:+.2f}) and {gm[0]:+.2f} pp/yr vs a beta-matched DIY "
           f"ladder (*t* = {gm[1]:+.2f}). The wrapper's apparent edge is **beta** "
           f"({R['beta_bufr']:.3f} vs {R['beta_diy']:.3f}), and on risk-adjusted terms it is "
           f"*worse* than the basket it wraps ({b[2]:+.3f} vs {d[2]:+.3f}) with a deeper "
           f"drawdown. The entry-point luck it exists to average away is worth a "
           f"**{R['var_reduction']:.1f}%** variance reduction, because the vintages are "
           f"{R['pair_corr']:.3f} correlated.\n"
           f"- **Tradability — Mirage.** Nothing to bank in either direction. And one "
           f"caveat that matters more than the *t*-stats: there is exactly **one** laddered "
           f"wrapper with this history, over 5.9 years containing a single down-year. This "
           f"is an n-of-1 product test, not a cross-section.\n"
           f"- **What the ladder is genuinely good for:** one trade instead of four, no "
           f"rebalance calendar, no vintage to choose. That convenience is real, and this "
           f"study does not price it. It is just not an edge."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    gd, gm, gx = R["gap_diy"], R["gap_matched"], R["gap_mix"]
    cells = [
        md("# Study 947 — The Buffer Ladder — the teardown\n\n"
           "The excess-of-cash Sharpe race, the Newey-West gap tests, paired block-bootstrap "
           "CIs, the beta decomposition, the entry-point-luck dispersion, an era cut, a cost "
           "sweep, a declared-proxy fee sweep, and the live synthetic control.\n\n"
           f"Every real number is frozen from `docs/results.md` (Fingerprint `{R['fp']}`), "
           f"{R['start']} → {R['end']}. Daily **total-return** closes throughout, TR vs TR. "
           "**One execution lag:** every estimated weight — the basket's rebalance target and "
           "every beta — is formed on data through the close of day *t* and applied at *t+1*."),

        code("R = %r" % (R,)),

        md("## 1. Two windows, never mixed\n\n"
           f"Arms needing no estimated weight (wrapper, vintages, DIY basket) race over "
           f"**n = {R['n_race']:,}**. The beta-matched arms burn the first 252 days on an "
           f"expanding, one-day-lagged OLS beta and race over **n = {R['n_matched']:,}** "
           f"(from {R['matched_start']}). Reported separately throughout."),

        md("## 2. The arms, excess-of-cash (BIL total return subtracted)"),
        code(
            "rows = [('BUFR (ladder)', R['bufr'], R['n_race']),\n"
            "        ('PJAN', R['pjan'], R['n_race']),\n"
            "        ('PAPR', R['papr'], R['n_race']),\n"
            "        ('PJUL', R['pjul'], R['n_race']),\n"
            "        ('POCT', R['poct'], R['n_race']),\n"
            "        ('DIY basket', R['diy'], R['n_race']),\n"
            "        ('DIY beta-matched', R['diy_matched'], R['n_matched']),\n"
            "        ('SPY/BIL mix @BUFR beta', R['mix_bufr'], R['n_matched']),\n"
            "        ('SPY', R['spy'], R['n_race'])]\n"
            "print('%-24s %6s %8s %7s %9s %7s' % ('arm', 'n', 'ann %', 'vol %', 'exSharpe', 'HAC t'))\n"
            "for name, v, n in rows:\n"
            "    print('%-24s %6d %+8.2f %7.2f %+9.3f %+7.2f' % (name, n, v[0], v[1], v[2], v[3]))"
        ),

        md("**Read the vol column before the return column.** The wrapper returns most of the "
           "buffer arms and has the highest vol of them, so its excess Sharpe (+0.777) sits "
           "*below* the DIY basket's (+0.853) and below POCT held alone (+0.943). The "
           "risk-adjusted ranking inverts the raw-return ranking."),

        md("## 3. The gaps — wrapper minus each DIY alternative\n\n"
           "HAC *t* on the daily return difference (Jobson-Korkie in Newey-West form; the "
           "cash leg cancels because both arms are excess-of-cash)."),
        code(
            "print('%-28s %6s %10s %8s %8s %9s' % ('comparison','n','gap pp/yr','HAC t','TE %','dSharpe'))\n"
            "for name, key, n in [('vs DIY basket','gap_diy',R['n_race']),\n"
            "                     ('vs PJAN','gap_pjan',R['n_race']),\n"
            "                     ('vs PAPR','gap_papr',R['n_race']),\n"
            "                     ('vs PJUL','gap_pjul',R['n_race']),\n"
            "                     ('vs POCT','gap_poct',R['n_race']),\n"
            "                     ('vs beta-matched DIY','gap_matched',R['n_matched']),\n"
            "                     ('vs beta-matched SPY/BIL','gap_mix',R['n_matched']),\n"
            "                     ('DIY vs beta-matched mix','gap_diy_mix',R['n_matched'])]:\n"
            "    g = R[key]\n"
            "    print('%-28s %6d %+10.2f %+8.2f %8.2f %+9.3f' % (name, n, g[0], g[1], g[2], g[3]))\n"
            "print()\n"
            "print('|t| >= 2 anywhere in this table? %s'\n"
            "      % any(abs(R[k][1]) >= 2 for k in ('gap_diy','gap_pjan','gap_papr','gap_pjul',\n"
            "                                        'gap_poct','gap_matched','gap_mix','gap_diy_mix')))"
        ),

        md(f"## 4. The beta decomposition — where the +{gd[0]:.2f} pp/yr actually comes from\n\n"
           f"Expanding-window, one-day-lagged OLS beta on SPY excess returns (identical to "
           f"the full-sample in-sample estimate to three decimals, which is itself worth "
           f"noting — the exposures are stable):\n\n"
           f"| | SPY-beta | Ann. excess return |\n|---|--:|--:|\n"
           f"| BUFR | **{R['beta_bufr']:.3f}** | {R['bufr'][0]:+.2f}% |\n"
           f"| DIY basket | **{R['beta_diy']:.3f}** | {R['diy'][0]:+.2f}% |\n"
           f"| Difference | {R['beta_bufr'] - R['beta_diy']:+.3f} | "
           f"{R['bufr'][0] - R['diy'][0]:+.2f} pp |\n\n"
           f"The beta gap is {R['beta_bufr'] - R['beta_diy']:.3f}; SPY's excess return over "
           f"the window was {R['spy'][0]:+.2f}%/yr; "
           f"{R['beta_bufr'] - R['beta_diy']:.3f} × {R['spy'][0]:.2f} ≈ "
           f"{(R['beta_bufr'] - R['beta_diy']) * R['spy'][0]:+.2f} pp/yr — which is more than "
           f"the entire {gd[0]:+.2f} pp/yr gap. Hold beta constant and the wrapper is "
           f"**{gm[0]:+.2f} pp/yr** behind (*t* = {gm[1]:+.2f}), with the two series "
           f"{R['corr_bufr_diy']:.3f} correlated day to day.\n\n"
           f"> 💡 **In plain words** — the wrapper did not ladder better. It just held more "
           f"stock, in five years when holding more stock paid."),

        md("## 5. Block bootstrap (2,000 draws, 21-day blocks, paired resampling)\n\n"
           "Both arms are resampled on the *same* block indices, so the 0.960 correlation "
           "between them survives the resampling — the correct construction for a difference "
           "between two near-identical funds."),
        code(
            "for name, key in [('gap vs DIY basket', 'boot_diy'),\n"
            "                  ('gap vs beta-matched DIY', 'boot_matched'),\n"
            "                  ('Sharpe gap vs DIY basket', 'boot_sharpe')]:\n"
            "    p, lo, hi, neg = R[key]\n"
            "    excl = 'EXCLUDES zero' if lo * hi > 0 else 'straddles zero'\n"
            "    print('%-26s %+7.3f  95%% CI [%+7.3f, %+7.3f]  frac<0 %5.1f%%  -> %s'\n"
            "          % (name, p, lo, hi, neg, excl))"
        ),

        md(f"**One CI in this study excludes zero — so we audited it.** The beta-matched gap's "
           f"bootstrap CI [{R['boot_matched'][1]:+.2f}, {R['boot_matched'][2]:+.2f}] *just* "
           f"clears zero while its HAC *t* is only **{gm[1]:+.2f}**. The block length is a free "
           f"parameter of the bootstrap, not a fact about the tape, so the first question is "
           f"whether the exclusion survives changing it."),
        code(
            "# Block-length sensitivity of the beta-matched gap's CI (frozen real-tape run,\n"
            "# regenerated by examples/verify.py; 3 RNG seeds x 2,000 draws per block).\n"
            "print('%-8s %22s %s' % ('block', 'mean 95% CI', 'excludes zero'))\n"
            "n_excl = n_all = 0\n"
            "for block, lo, hi, k, s in R['boot_block_sens']:\n"
            "    print('%5dd   [%+7.2f, %+7.2f]      %d/%d seeds' % (block, lo, hi, k, s))\n"
            "    n_excl += k; n_all += s\n"
            "print('\\nexcludes zero on %d of %d (block, seed) settings' % (n_excl, n_all))"
        ),

        md(f"**It does not.** Flip to a 5- or 10-day block and the same gap straddles zero: "
           f"the exclusion holds on 9 of 15 (block, seed) settings and fails on 6. A result "
           f"that changes verdict when you change a nuisance parameter is not a result. The "
           f"HAC *t* of **{gm[1]:+.2f}** — which prices the autocorrelation directly instead of "
           f"resampling around it — is the honest summary, and it is what this study is "
           f"stamped on: suggestive of a small fee drag, **short of |*t*| ≥ 2, and not "
           f"rescued by the bootstrap**. For contrast, the synthetic panel's *planted* premium "
           f"excludes zero at **every** block length — that is what a real gap looks like "
           f"under the same sweep, and `tests/test_strategy.py` asserts it."),

        md("## 6. Entry-point luck, and why averaging barely touches it\n\n"
           f"| Measure | Value |\n|---|--:|\n"
           f"| Rolling 1-yr best-minus-worst vintage spread (mean / median / max) | "
           f"{R['spread_mean']:.2f} / {R['spread_median']:.2f} / {R['spread_max']:.2f} pp |\n"
           f"| SD of rolling 1-yr returns, average single vintage | {R['sd_single_mean']:.2f}% |\n"
           f"| SD of rolling 1-yr returns, equal-weight basket | {R['sd_basket']:.2f}% |\n"
           f"| **Variance reduction — 1-year holding period** | "
           f"**{R['var_reduction']:.1f}%** |\n"
           f"| **Variance reduction — daily returns** | "
           f"**{R['var_reduction_daily']:.1f}%** |\n"
           f"| Closed form √((1 + 3ρ)/4) at that ρ | "
           f"**{R['var_reduction_closed_form']:.1f}%** |\n"
           f"| Mean pairwise daily correlation | **{R['pair_corr']:.3f}** |\n\n"
           f"This is the load-bearing measurement in the study and it needs no inference at "
           f"all. The closed form for N equally-correlated legs, "
           f"sd(basket)/sd(leg) = √((1 + (N−1)ρ)/N), predicts "
           f"**{R['var_reduction_closed_form']:.1f}%** at ρ = {R['pair_corr']:.3f}, N = 4; "
           f"the daily tape delivers **{R['var_reduction_daily']:.1f}%**. The arithmetic is "
           f"exact. On a one-year holding period the realised cut is smaller still "
           f"({R['var_reduction']:.1f}%), because each vintage's path through its own buffer "
           f"and cap does not average as cleanly as its daily noise does. Laddering delivers "
           f"precisely what the correlation says it must — which is almost nothing."),
        code(IMPORTS),
        code(CORR_DEMO),

        md(f"## 7. Era cut (split 2023-07-01)\n\n"
           f"| Era | n | exSharpe wrapper / DIY | vs DIY basket | vs beta-matched DIY |\n"
           f"|---|--:|--:|--:|--:|\n"
           f"| 2020-08 → 2023-06 | {R['era_e'][0]} | {R['era_e'][1]:+.3f} / "
           f"{R['era_e'][2]:+.3f} | {R['era_e'][3]:+.2f} pp (*t* = {R['era_e'][4]:+.2f}) | "
           f"**{R['era_e'][5]:+.2f} pp** (*t* = {R['era_e'][6]:+.2f}) |\n"
           f"| 2023-07 → 2026-06 | {R['era_l'][0]} | {R['era_l'][1]:+.3f} / "
           f"{R['era_l'][2]:+.3f} | {R['era_l'][3]:+.2f} pp (*t* = {R['era_l'][4]:+.2f}) | "
           f"**{R['era_l'][5]:+.2f} pp** (*t* = {R['era_l'][6]:+.2f}) |\n\n"
           "The beta-matched shortfall is negative in both halves and significant in neither. "
           "Nothing flips sign; nothing crosses the bar. Note the second era covers the "
           "high-short-rate regime, where the excess-of-cash framing bites hardest — and the "
           "conclusion does not move."),

        md("## 8. Cost sweep — and which way it cuts\n\n"
           f"| One-way cost | vs DIY basket | vs beta-matched DIY |\n|---|--:|--:|\n"
           f"| 0 bps (gross) | {R['cost0'][0]:+.2f} pp (*t* = {R['cost0'][1]:+.2f}) | "
           f"{R['cost0'][2]:+.2f} pp (*t* = {R['cost0'][3]:+.2f}) |\n"
           f"| 25 bps | {R['cost25'][0]:+.2f} pp (*t* = {R['cost25'][1]:+.2f}) | "
           f"{R['cost25'][2]:+.2f} pp (*t* = {R['cost25'][3]:+.2f}) |\n\n"
           "Friction is charged one-way × NAV on the **DIY** arms only — the wrapper's own "
           "trading is already inside its NAV. So a higher cost can only *flatter* the "
           "wrapper, and even at a punitive 25 bps it still fails to beat a beta-matched DIY "
           "ladder. Rebalancing the basket never / annually / quarterly / monthly moves its "
           "excess Sharpe by less than 0.001 in all four cases — four legs correlated 0.889 "
           "barely drift apart. No short leg is required on the real tape, so no borrow is "
           "charged; the machinery charges it if the matched weight ever goes negative."),

        md(f"## 9. The declared PROXY — the fee layer, and its sweep\n\n"
           f"A single Power Buffer vintage quotes **{R['fee_single_pct']:.2f}%/yr**; the "
           f"laddered wrapper adds a management fee on top of those acquired-fund fees, an "
           f"incremental layer we **assume** at **{R['fee_extra_pct']:.2f}%/yr**. This is a "
           f"quoted number, not a tape measurement — published NAV returns are already net of "
           f"whatever was actually charged. It is used only to build a 'had the layer been "
           f"waived' counterfactual, and it is swept:\n\n"
           f"| Extra layer waived | vs DIY basket | vs beta-matched DIY |\n|---|--:|--:|\n"
           f"| +0.00%/yr | {R['fee00'][0]:+.2f} pp (*t* = {R['fee00'][1]:+.2f}) | "
           f"{R['fee00'][2]:+.2f} pp (*t* = {R['fee00'][3]:+.2f}) |\n"
           f"| +{R['fee_extra_pct']:.2f}%/yr (our assumption) | {R['fee20'][0]:+.2f} pp "
           f"(*t* = {R['fee20'][1]:+.2f}) | {R['fee20'][2]:+.2f} pp "
           f"(*t* = {R['fee20'][3]:+.2f}) |\n"
           f"| +0.40%/yr (generous upper bound) | {R['fee40'][0]:+.2f} pp "
           f"(*t* = {R['fee40'][1]:+.2f}) | {R['fee40'][2]:+.2f} pp "
           f"(*t* = {R['fee40'][3]:+.2f}) |\n\n"
           f"The assumed layer accounts for a fifth to a half of the beta-matched shortfall, "
           f"and no value in the swept range turns the wrapper into a winner. The verdict does "
           f"not rest on the guess."),

        md("## 10. Synthetic control — the detector is unbiased (offline)\n\n"
           "It never supports the real-tape stamp; it only proves the null is a fact about "
           "the tape rather than a broken harness. The panel plants a laddering premium on "
           "top of an equal-weight vintage basket, net of a planted fee, with realistic "
           "wrapper tracking noise."),
        code(SYNTH_CONTROL),

        md(f"| Panel | Planted | Recovered | Error | HAC *t* |\n|---|--:|--:|--:|--:|\n"
           f"| Planted premium (4%/yr less a 0.20%/yr fee) | "
           f"{R['syn_planted'][0]:+.2f} | **{R['syn_planted'][1]:+.2f}** | "
           f"{R['syn_planted'][2]:+.2f} | **{R['syn_planted'][3]:+.2f}** |\n"
           f"| Null, fee only | {R['syn_null_fee'][0]:+.2f} | {R['syn_null_fee'][1]:+.2f} | "
           f"{R['syn_null_fee'][2]:+.2f} | {R['syn_null_fee'][3]:+.2f} |\n"
           f"| Clean null | {R['syn_null_clean'][0]:+.2f} | {R['syn_null_clean'][1]:+.2f} | "
           f"{R['syn_null_clean'][2]:+.2f} | {R['syn_null_clean'][3]:+.2f} |\n\n"
           f"Across 8 null seeds: mean gap {R['syn_null_mean']:+.2f} pp/yr (sd "
           f"{R['syn_null_sd']:.2f}), max |*t*| {R['syn_null_maxt']:.2f}, fires on "
           f"**{R['syn_null_fires']}/8**."),

        md(f"## Verdict\n\n"
           f"- **Signal — None.** No gap clears |*t*| = 2 in either direction: "
           f"{gd[0]:+.2f} pp/yr vs the DIY basket (*t* = {gd[1]:+.2f}, bootstrap CI "
           f"[{R['boot_diy'][1]:+.2f}, {R['boot_diy'][2]:+.2f}]) and {gm[0]:+.2f} pp/yr vs a "
           f"beta-matched DIY ladder (*t* = {gm[1]:+.2f}). The wrapper's headline "
           f"outperformance is beta ({R['beta_bufr']:.3f} vs {R['beta_diy']:.3f}), and "
           f"risk-adjusted it is *behind* the basket it wraps "
           f"({R['bufr'][2]:+.3f} vs {R['diy'][2]:+.3f}) with a "
           f"{abs(R['dd_bufr']) - abs(R['dd_diy']):.1f} pp deeper drawdown. The entry-point "
           f"luck laddering exists to remove is worth a {R['var_reduction']:.1f}% variance "
           f"reduction at ρ = {R['pair_corr']:.3f}. **Survivorship:** the surviving flagships "
           f"of a category that has shuttered products, and an **n-of-1** wrapper over 5.9 "
           f"years with one down-year.\n"
           f"- **Tradability — Mirage.** Nothing to bank. Long the wrapper for laddering pays "
           f"a fee layer for a 2.4% variance cut plus a beta tilt an index fund sells "
           f"cheaper; short the wrapper against a beta-matched DIY ladder targets "
           f"{gm[0]:+.2f} pp/yr across {gm[2]:.2f}% tracking error at *t* = {gm[1]:+.2f} — a "
           f"coin-flip with a borrow bill. Both arms merely **tie** the dumb beta-matched "
           f"SPY/BIL mix ({gx[0]:+.2f} pp/yr, *t* = {gx[1]:+.2f}), reproducing Study 624's "
           f"result one layer up the wrapper stack."),
    ]
    nb["cells"] = cells
    return nb


def main():
    for name, nb in [("01_for_the_curious.ipynb", build_curious()),
                     ("02_for_the_quants.ipynb", build_quants())]:
        nb["metadata"] = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        }
        path = os.path.join(HERE, name)
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print(f"wrote {path} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
