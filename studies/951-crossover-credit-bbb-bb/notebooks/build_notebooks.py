"""Generate the two narrative notebooks for Study 951 (The Crossover Rung).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from
the frozen ``R`` dict below (the single mirror of ``docs/results.md``); the only live cells
run the fast **synthetic** control, and they are never placed under a real-tape banner.
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


# Frozen real-tape headline — the single mirror of docs/results.md. Daily total-return
# closes, excess of BIL, adjusted on IEF (duration) + SPY (equity beta).
# Window: 2012-04-11 -> 2026-06-30, ANGL inception gated. Alphas are annualised %.
R = dict(
    start="2012-04-11", end="2026-06-30", n_days=3575, n_reg=3574, fp="b2e46afc7601",
    # the ladder: excess Sharpe, excess return %/yr, vol %, abs max DD %, alpha %/yr, t
    agg=(0.113, 0.55, 4.86, -18.4, -0.48, -0.94, 0.71, 0.07, 0.84),
    lqd=(0.241, 1.89, 7.82, -25.0, -0.67, -0.49, 0.97, 0.18, 0.67),
    angl=(0.602, 5.57, 9.25, -29.3, +0.84, +0.42, 0.34, 0.35, 0.39),
    hyg=(0.437, 3.42, 7.82, -22.0, -1.67, -1.23, 0.27, 0.37, 0.62),
    # head-to-head: adjusted alpha %/yr, HAC t, raw diff %/yr, raw t
    ah_alpha=2.51, ah_t=2.01, ah_raw=2.15, ah_raw_t=1.96,
    al_alpha=1.51, al_t=1.15, al_raw=3.68, al_raw_t=2.09,
    # HYG - LQD on the SAME ANGL-gated window as the two rows above (comparable);
    # its own longest BIL-gated window (2007-05 -> 2026-06, n=4801) reads +0.05 / +0.03
    # and is reported separately, never mixed into the table.
    hl_alpha=-1.00, hl_t=-0.83, hl_raw=1.53, hl_raw_t=0.92,
    hl_long_alpha=0.05, hl_long_t=0.03, hl_long_n=4801,
    # bootstrap (2000 draws, 21-day blocks, joint resample of y and factors)
    ah_ci=(0.38, 4.74), ah_neg=1.3, al_ci=(-0.61, 3.90), al_neg=8.0,
    # era cut at 2019-01-01
    era_e_n=1691, era_e_ah=4.47, era_e_ah_t=2.63, era_e_al=3.94, era_e_al_t=2.02,
    era_l_n=1882, era_l_ah=1.50, era_l_ah_t=0.92, era_l_al=-0.05, era_l_al_t=-0.03,
    # calendar-year ANGL - HYG excess difference, %
    cal=[(2012, 2.12), (2013, 1.15), (2014, 0.63), (2015, 3.60), (2016, 10.65),
         (2017, 3.41), (2018, -3.92), (2019, 3.48), (2020, 8.48), (2021, 2.98),
         (2022, -3.74), (2023, 0.82), (2024, -1.77), (2025, 0.47), (2026, 0.58)],
    # jackknife
    jk_pass=5, jk_n=15, jk_drop16=(1.84, 1.43), jk_drop20=(2.32, 2.42),
    drop_both=(1.55, 1.61), drop_both_n=3069,
    # HAC bandwidth sweep on ANGL - HYG
    lags=[(5, 1.95), (8, 2.01), (10, 2.01), (21, 2.25), (42, 2.54)],
    # independent tapes
    faln_ushy=(0.63, 0.48, 2178), faln_hyg=(1.78, 1.34, 2519), angl_ushy=(0.18, 0.14, 2178),
    # all four crossover/broad-HY pairs on the ONE window the four funds share
    # (2017-10 -> 2026-06, n=2178) — shows the sibling collapse is mostly the window,
    # not fund identity, so it must not be counted as a separate failure
    same_win=[("ANGL - HYG (headline)", 0.96, 0.67), ("FALN - HYG", 1.42, 0.97),
              ("FALN - USHY", 0.63, 0.48), ("ANGL - USHY", 0.18, 0.14)],
    same_win_n=2178,
    late_ladder=[("LQD", 0.041, -1.12), ("FALN", 0.364, -1.01), ("ANGL", 0.312, -1.46),
                 ("USHY", 0.311, -1.64), ("HYG", 0.244, -2.42)],
    # expression A (own ANGL instead of HYG) and B (long/short, borrow swept)
    swap_gain=0.165,
    borrow=[(0, 2.53, 2.11, 2.14, 0.342), (25, 2.28, 1.90, 1.89, 0.302),
            (50, 2.03, 1.69, 1.64, 0.262), (100, 1.53, 1.28, 1.14, 0.182)],
    turnover=0.38,
    # synthetic control
    syn_planted=3.00, syn_recovered=3.58, syn_t=4.64,
    syn_null_mean=0.10, syn_null_sd=0.60, syn_null_fire=0, syn_null_n=8,
)


HEADER = f"""# Study 951 — The Crossover Rung 🪜

**Is the BBB/BB boundary the best-paid rung on the credit ladder?**

Credit is sold as a ladder — aggregate, investment grade, *crossover*, high yield — and the
folklore says the best-paid step is the **boundary** between investment grade and junk.
The story is institutional: when an issuer is downgraded out of investment grade, mandates
and insurance capital rules *force* holders to sell, while the natural buyers are slow or
constrained. Somebody should be collecting the discount.

We race the whole ladder on daily **total-return** closes — **AGG → LQD → ANGL → HYG** —
every rung **excess of cash** (BIL) and then adjusted for **duration** (IEF) and **equity
beta** (SPY), so the winner is not simply whoever carries the most risk.
Window {R['start']} → {R['end']} ({R['n_days']:,} days), gated by ANGL's inception.

*Real-tape numbers below are the frozen headline (`docs/results.md`, Fingerprint
`{R['fp']}`); the live cells run the offline **synthetic** control only. As-of 2026-06-30.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The rungs, and what each one actually paid\n\n"
           "Excess-of-cash **Sharpe** is reward per unit of risk — how much you were paid for "
           "the bumps you wore. Walk up the ladder and it rises, peaks on the crossover rung, "
           "and then *falls* as you go deeper into high yield. The ladder has a hump, and the "
           "hump is on the boundary. That much of the folklore is simply true."),
        code(
            "R = dict(agg=%r, lqd=%r, angl=%r, hyg=%r)\n"
            "names = {'agg':'AGG  aggregate     ', 'lqd':'LQD  investment grade',\n"
            "         'angl':'ANGL crossover     ', 'hyg':'HYG  broad high yield'}\n"
            "for k in ('agg','lqd','angl','hyg'):\n"
            "    s, exr, vol, dd = R[k][0], R[k][1], R[k][2], R[k][3]\n"
            "    bar = '#' * int(round(s * 30))\n"
            "    print('%%s  Sharpe %%+.3f  %%-20s  earned %%+.2f%%%%/yr, worst fall %%.1f%%%%'\n"
            "          %% (names[k], s, bar, exr, dd))"
            % (R["agg"], R["lqd"], R["angl"], R["hyg"])
        ),
        md("> 🔬 **For the quants** — Sharpe here is *excess of cash*: each rung's daily total "
           "return minus BIL's, so the 2023-2026 era of 5% short rates does not flatter a bond "
           "fund that merely earned the bill rate. Drawdowns are *absolute* (what you lived "
           "through), not excess."),
        md("## 2. Why the raw ladder is a rigged race\n\n"
           "The crossover rung earned more *because it took different risks*: more equity-like "
           "credit risk than investment grade, longer duration than deep high yield. To ask "
           "whether the boundary is genuinely **better paid** you have to strip both out — "
           "hold a matching amount of Treasuries (IEF) and stocks (SPY) against each rung and "
           "see what is left over. That leftover is the *alpha*."),
        code(
            "R = dict(ah_alpha=%r, ah_t=%r, ah_raw=%r, al_alpha=%r, al_t=%r, al_raw=%r,\n"
            "         hl_alpha=%r, hl_t=%r)\n"
            "print('crossover vs broad high yield : raw %%+.2f%%%%/yr -> adjusted %%+.2f%%%%/yr  (t = %%+.2f)'\n"
            "      %% (R['ah_raw'], R['ah_alpha'], R['ah_t']))\n"
            "print('crossover vs investment grade : raw %%+.2f%%%%/yr -> adjusted %%+.2f%%%%/yr  (t = %%+.2f)'\n"
            "      %% (R['al_raw'], R['al_alpha'], R['al_t']))\n"
            "print('broad high yield vs IG        :                    adjusted %%+.2f%%%%/yr  (t = %%+.2f)'\n"
            "      %% (R['hl_alpha'], R['hl_t']))\n"
            "print()\n"
            "print('t = 2 is the desk bar. One leg lands ON it; the other does not land at all.')"
            % (R["ah_alpha"], R["ah_t"], R["ah_raw"], R["al_alpha"], R["al_t"], R["al_raw"],
               R["hl_alpha"], R["hl_t"])
        ),
        md(f"So the claim splits, and only half of it half-survives. Against **broad high "
           f"yield** the boundary pays **+{R['ah_alpha']:.2f}%/yr** — but at *t* = "
           f"**+{R['ah_t']:.2f}**, sitting *on* the significance bar rather than past it. "
           f"Against **investment grade** — the half that would make it a genuine *rung* "
           f"rather than just a better high-yield fund — it pays "
           f"**+{R['al_alpha']:.2f}%/yr at *t* = +{R['al_t']:.2f}**. Nothing. And the step "
           f"from investment grade to broad high yield paid **{R['hl_alpha']:.2f}%/yr** on "
           f"the same window (and **+{R['hl_long_alpha']:.2f}%/yr** on its own longer "
           f"19-year one): once you adjust for the risks, that rung was free money for "
           f"nobody, whichever sample you read it on."),
        md("## 3. Two years carry the whole thing\n\n"
           "Look at the crossover rung's advantage over broad high yield, year by year. It is "
           "not a steady drip of carry. It is two enormous years surrounded by noise — **2016** "
           "(the energy and mining downgrade wave recovering off the February lows) and "
           "**2020** (the COVID fallen-angel wave: Ford, Occidental, Kraft Heinz). That is the "
           "forced-seller mechanism firing, exactly as the theory says — and it says nothing "
           "about the years in between."),
        code(
            "cal = %r\n"
            "for y, v in cal:\n"
            "    mark = '  <-- downgrade wave' if v > 6 else ''\n"
            "    bar = ('+' if v >= 0 else '-') * max(1, int(round(abs(v))))\n"
            "    print('%%d  %%+6.2f%%%%  %%-12s%%s' %% (y, v, bar, mark))"
            % (R["cal"],)
        ),
        md(f"## 4. Delete a year and the significance evaporates\n\n"
           f"The honest stress test: remove one calendar year at a time and re-estimate. A "
           f"broad premium barely notices. This one **survives only "
           f"{R['jk_pass']} of {R['jk_n']}** single-year deletions. Take out 2016 alone and it "
           f"falls to **+{R['jk_drop16'][0]:.2f}%/yr at *t* = +{R['jk_drop16'][1]:.2f}**; take "
           f"out both wave years and it is **+{R['drop_both'][0]:.2f}%/yr at *t* = "
           f"+{R['drop_both'][1]:.2f}** over the remaining {R['drop_both_n']:,} days. Same "
           f"sign, a third smaller, and no longer distinguishable from luck.\n\n"
           f"The era cut says the same thing in a different voice: "
           f"**+{R['era_e_ah']:.2f}%/yr (*t* = +{R['era_e_ah_t']:.2f})** before 2019, "
           f"**+{R['era_l_ah']:.2f}%/yr (*t* = +{R['era_l_ah_t']:.2f})** after. And the recent "
           f"era is *not* a quiet one for the mechanism — it contains the largest downgrade "
           f"wave in history. The engine fired and the premium still did not show up."),
        md(f"## 5. Swap the funds and it disappears\n\n"
           f"ANGL is not the crossover rung; it is *one fund's version of it*. There is a "
           f"second fallen-angel fund (**FALN**) and a second broad high-yield fund "
           f"(**USHY**). Run the identical race on the sibling pair and the premium is "
           f"**+{R['faln_ushy'][0]:.2f}%/yr at *t* = +{R['faln_ushy'][1]:.2f}** — the right "
           f"sign, and essentially nothing.\n\n"
           f"But be careful how much of that you charge to the *funds*. The sibling pair only "
           f"exists from 2017, and on that same window the **headline** pair reads "
           f"+{R['same_win'][0][1]:.2f}%/yr (*t* = +{R['same_win'][0][2]:.2f}) as well. Most "
           f"of the collapse is the calendar, not the holdings — this is section 4 wearing a "
           f"different hat, not a second, independent piece of bad news."),
        md(f"## 6. What you would actually be buying\n\n"
           f"The cheap way to own this is not a hedge fund trade — it is a **fund swap**: hold "
           f"ANGL instead of HYG inside a high-yield sleeve you already have. One trade, "
           f"pennies of spread, fees already inside the tape, and over the full window it "
           f"bought **+{R['swap_gain']:.3f}** of excess Sharpe.\n\n"
           f"The price of admission is written in the drawdown column of the first table: "
           f"**−{abs(R['angl'][3]):.1f}%** for the crossover fund against "
           f"**−{abs(R['hyg'][3]):.1f}%** for broad high yield in March 2020, and 18% more "
           f"day-to-day volatility. You are not being handed a premium; you are changing what "
           f"kind of risk you own, and being paid a contested amount for it."),
        md("## 7. Is the measuring stick honest? (live, synthetic)\n\n"
           "Before believing a borderline number, check the instrument. We build a fake credit "
           "ladder where we *know* the crossover rung is paid an extra 3%/yr, and a second one "
           "where we know it is paid nothing. The same estimator must find the first and stay "
           "quiet on the second. **This cell is synthetic — it is a proof about the tool, "
           "never evidence about credit.**"),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from crossover_credit import data, strategy as st\n"
            "planted, truth = data.synthetic_panel(signal_strength=1.0, seed=951)\n"
            "d = st.synthetic_detect(planted)\n"
            "print('planted %+.2f%%/yr  ->  recovered %+.2f%%/yr (t = %+.2f)'\n"
            "      % (truth['alpha_planted']*100, d['alpha_ann']*100, d['t_alpha']))\n"
            "nulls = [st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=951+s)[0])\n"
            "         for s in range(8)]\n"
            "a = np.array([x['alpha_ann'] for x in nulls]) * 100\n"
            "t = np.array([x['t_alpha'] for x in nulls])\n"
            "print('null x8: alpha mean %+.2f%%/yr (sd %.2f), |t| >= 2 fires %d/8'\n"
            "      % (a.mean(), a.std(ddof=1), (abs(t) >= 2).sum()))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** The ladder really does hump on the boundary by raw "
           f"reward-per-risk, and the premium's sign is positive everywhere we look. But it "
           f"beats broad high yield only *at* the bar (*t* = +{R['ah_t']:.2f}), survives only "
           f"{R['jk_pass']}/{R['jk_n']} year deletions, and is three times smaller after 2018. "
           f"(The sibling-fund pair reads *t* = +{R['faln_ushy'][1]:.2f}, but that is mostly "
           f"the same late window rather than a separate failure.) Against investment "
           f"grade there is nothing at all. Right shape, unproven premium.\n"
           f"- **Tradability — Fragile.** The switch is genuinely cheap — no borrow, no timing, "
           f"one trade — so this is not a cost mirage. But what you buy is a "
           f"{abs(R['angl'][3]) - abs(R['hyg'][3]):.1f} pp deeper worst drawdown in exchange "
           f"for an episodic, downgrade-wave payoff. A tilt worth holding if you want that "
           f"exposure; not a premium to size."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 951 — The Crossover Rung — the teardown\n\n"
           "Four rungs (AGG, LQD, ANGL, HYG) on daily total-return closes, excess of BIL, "
           "regressed jointly on IEF (duration) and SPY (equity beta). The headline statistic "
           "is the two-factor alpha on the **return difference** between rungs, with a "
           "Newey-West HAC *t*. Then: a joint block-bootstrap CI, an era cut, a one-year-out "
           "jackknife, an HAC-bandwidth sweep, the sibling-fund pair, the borrow sweep, and "
           "the live synthetic control.\n\n"
           "Every real number is frozen from `docs/results.md` (Fingerprint `%s`), window "
           "%s → %s, n = %s regression days. "
           "**Proxies:** fallen-angel ETFs stand in for the crossover rung (no free daily "
           "BBB−/BB+ tape); IEF/SPY are a two-factor adjustment, not a credit-factor model, "
           "and their betas are **fitted in sample** (full-window OLS, in-sample intercept — "
           "standard attribution, and the fitted hedge is never traded); the short-leg "
           "borrow rate is an assumption and is swept."
           % (R["fp"], R["start"], R["end"], f"{R['n_reg']:,}")),
        code("R = %r" % (R,)),
        md("## 1. The ladder — levels and loadings\n\n"
           "Columns: excess-of-cash Sharpe, annualised excess return, annualised vol, "
           "*absolute* max drawdown, two-factor alpha and its HAC *t*, the two betas, R².\n\n"
           "> 💡 **In plain words** — the first column is what each rung paid per unit of "
           "wobble; the alpha column is what is left after you hold the matching amount of "
           "Treasuries and stocks against it."),
        code(
            "hdr = '%-6s %9s %10s %8s %9s %11s %8s %7s %7s %6s'\n"
            "print(hdr % ('rung','exSharpe','excess/yr','vol','maxDD','alpha/yr','t','bIEF','bSPY','R2'))\n"
            "for k, nm in [('agg','AGG'), ('lqd','LQD'), ('angl','ANGL'), ('hyg','HYG')]:\n"
            "    s, exr, vol, dd, a, t, bd, be, r2 = R[k]\n"
            "    print('%-6s %+9.3f %+9.2f%% %7.2f%% %+8.1f%% %+10.2f%% %+8.2f %7.2f %7.2f %6.2f'\n"
            "          % (nm, s, exr, vol, dd, a, t, bd, be, r2))\n"
            "print()\n"
            "print('The raw excess-Sharpe ladder humps on the crossover rung (%.3f) --' % R['angl'][0])\n"
            "print('but no single rung alpha clears |t| = 2 on its own.')"
        ),
        md("## 2. Head-to-head — alpha on the return difference\n\n"
           "Both legs are excess-of-cash, so the cash leg cancels in the difference and the "
           "intercept is exactly the duration- and equity-adjusted premium of one rung over "
           "another (the Jobson-Korkie return-difference form, HAC standard errors). All "
           "three pairs run on the **same ANGL-gated window** so the rows are comparable; "
           "HYG − LQD is the one pair that does not need ANGL, and its own longest "
           f"(BIL-gated, n = {R['hl_long_n']:,}) window is printed underneath, labelled, "
           "rather than mixed into the table."),
        code(
            "rows = [('ANGL - HYG (boundary vs deep HY)', R['ah_alpha'], R['ah_t'], R['ah_raw'], R['ah_raw_t']),\n"
            "        ('ANGL - LQD (boundary vs IG)     ', R['al_alpha'], R['al_t'], R['al_raw'], R['al_raw_t']),\n"
            "        ('HYG  - LQD (deep HY vs IG)      ', R['hl_alpha'], R['hl_t'], R['hl_raw'], R['hl_raw_t'])]\n"
            "print('%-34s %10s %7s %10s %7s' % ('pair','alpha/yr','HAC t','raw/yr','raw t'))\n"
            "for nm, a, t, raw, rt in rows:\n"
            "    print('%-34s %+9.2f%% %+7.2f %+9.2f%% %+7.2f' % (nm, a, t, raw, rt))\n"
            "print()\n"
            "print('  (all three rows: same ANGL-gated window, n = %d)' % R['n_reg'])\n"
            "print()\n"
            "print('ANGL - LQD: the adjustment removes %.0f%% of the raw difference'\n"
            "      % ((1 - R['al_alpha']/R['al_raw'])*100))\n"
            "print('HYG  - LQD: the IG->HY step paid %+.2f%%/yr adjusted -- nothing at all' % R['hl_alpha'])\n"
            "print('            on its own longest window (n = %d) it reads %+.2f%%/yr (t = %+.2f)'\n"
            "      % (R['hl_long_n'], R['hl_long_alpha'], R['hl_long_t']))"
        ),
        md("## 3. Block-bootstrap CI (2,000 draws, 21-day blocks)\n\n"
           "Blocks are resampled **jointly** across the difference series and the factor tape, "
           "so the betas are re-estimated on every draw and the interval carries their "
           "estimation error, not just the intercept's."),
        code(
            "print('ANGL - HYG : alpha %+.2f%%  95%% CI [%+.2f%%, %+.2f%%]  share<0 %.1f%%'\n"
            "      % (R['ah_alpha'], R['ah_ci'][0], R['ah_ci'][1], R['ah_neg']))\n"
            "print('ANGL - LQD : alpha %+.2f%%  95%% CI [%+.2f%%, %+.2f%%]  share<0 %.1f%%'\n"
            "      % (R['al_alpha'], R['al_ci'][0], R['al_ci'][1], R['al_neg']))\n"
            "print()\n"
            "print('The HY leg clears zero with %.2f pp to spare at the lower bound.' % R['ah_ci'][0])\n"
            "print('The IG leg does not clear zero.')"
        ),
        md("## 4. Era cut (split 2019-01-01)\n\n"
           "> 💡 **In plain words** — the premium is a first-half phenomenon by a factor of "
           "three, and the second half is *not* a quiet period for the mechanism: it contains "
           "the March-2020 fallen-angel wave, the largest downgrade cohort on record."),
        code(
            "print('%-12s %6s %12s %8s' % ('pair','n','alpha/yr','HAC t'))\n"
            "print('%-12s %6d %+11.2f%% %+8.2f' % ('ANGL-HYG e', R['era_e_n'], R['era_e_ah'], R['era_e_ah_t']))\n"
            "print('%-12s %6d %+11.2f%% %+8.2f' % ('ANGL-HYG l', R['era_l_n'], R['era_l_ah'], R['era_l_ah_t']))\n"
            "print('%-12s %6d %+11.2f%% %+8.2f' % ('ANGL-LQD e', R['era_e_n'], R['era_e_al'], R['era_e_al_t']))\n"
            "print('%-12s %6d %+11.2f%% %+8.2f' % ('ANGL-LQD l', R['era_l_n'], R['era_l_al'], R['era_l_al_t']))\n"
            "print()\n"
            "print('ANGL-HYG decays %.1fx across the split; ANGL-LQD goes to zero.'\n"
            "      % (R['era_e_ah'] / R['era_l_ah']))"
        ),
        md("## 5. Influence — calendar years and the one-year-out jackknife\n\n"
           "The forced-seller mechanism predicts an *episodic* payoff, and that is exactly what "
           "the tape shows — which is also what destroys the significance."),
        code(
            "tot = sum(v for _, v in R['cal'])\n"
            "waves = sum(v for y, v in R['cal'] if y in (2016, 2020))\n"
            "for y, v in R['cal']:\n"
            "    print('%d %+6.2f%%%s' % (y, v, '   <- downgrade wave' if y in (2016, 2020) else ''))\n"
            "print()\n"
            "print('2016 + 2020 alone = %+.2f pp of the %+.2f pp cumulative sum (%.0f%%)'\n"
            "      % (waves, tot, 100*waves/tot))\n"
            "print('|t| >= 2 survives %d/%d single-year deletions' % (R['jk_pass'], R['jk_n']))\n"
            "print('drop 2016      : alpha %+.2f%% (t = %+.2f)' % R['jk_drop16'])\n"
            "print('drop 2020      : alpha %+.2f%% (t = %+.2f)' % R['jk_drop20'])\n"
            "print('drop both      : alpha %+.2f%% (t = %+.2f)  over %d days'\n"
            "      % (R['drop_both'][0], R['drop_both'][1], R['drop_both_n']))"
        ),
        md("## 6. HAC bandwidth and the sibling-fund pair\n\n"
           "The bandwidth sweep runs in the *unhelpful* direction — the *t* is lowest at short "
           "bandwidths and rises with the kernel — so the headline (default `4(n/100)^(2/9)` = "
           "8 lags) is not a cherry-picked wide window. The fund swap looks like the harder "
           "test — replace **both** legs with their siblings and the premium is gone — but "
           "the sibling funds only exist from 2017, so the last block re-runs all four pairs "
           "on the one window they share. The headline pair collapses there too: the swap is "
           "mostly the era cut in disguise."),
        code(
            "for lg, t in R['lags']:\n"
            "    print('HAC lags %3d : t = %+.2f%s' % (lg, t, '   <- default rule of thumb' if lg == 8 else ''))\n"
            "print()\n"
            "print('%-28s %10s %8s %7s' % ('pair','alpha/yr','HAC t','n'))\n"
            "print('%-28s %+9.2f%% %+8.2f %7d' % ('ANGL - HYG  (headline)', R['ah_alpha'], R['ah_t'], R['n_reg']))\n"
            "for nm, key in [('FALN - USHY (both swapped)','faln_ushy'),\n"
            "                ('FALN - HYG  (cross leg)','faln_hyg'),\n"
            "                ('ANGL - USHY (HY leg)','angl_ushy')]:\n"
            "    a, t, n = R[key]\n"
            "    print('%-28s %+9.2f%% %+8.2f %7d' % (nm, a, t, n))\n"
            "print()\n"
            "print('ladder on the FALN-gated 2017-2026 window (excess Sharpe, alpha):')\n"
            "for nm, s, a in R['late_ladder']:\n"
            "    print('  %-5s %+.3f  %+.2f%%' % (nm, s, a))\n"
            "print('  the hump survives; every alpha is negative and every |t| < 1.4')\n"
            "print()\n"
            "print('window-vs-fund: all four pairs on the shared window (n = %d)' % R['same_win_n'])\n"
            "for nm, a, t in R['same_win']:\n"
            "    print('  %-22s %+6.2f%%   t %+5.2f' % (nm, a, t))\n"
            "print('  the HEADLINE pair loses %.2f pp on this window alone --'\n"
            "      % (R['ah_alpha'] - R['same_win'][0][1]))\n"
            "print('  so the sibling swap is mostly the era cut restated, not a third failure')"
        ),
        md("## 7. The tradable expressions — costs, borrow, and one execution lag\n\n"
           "**A.** Own ANGL instead of HYG: no shorting, no timing, a single switch charged "
           "on **both** sides (sell HYG *and* buy ANGL, 2 × 5 bps one-way) and amortised "
           "over 14 years — ~0.007%/yr of drag — fund fees already inside the total-return "
           "tape.\n\n"
           "**B.** The pure spread (long ANGL, short HYG), dollar-neutral, reset monthly: the "
           "weights implied by the closes through the last trading day of month *t* are acted "
           "on at *t*+1 — **the study's single execution lag** — where the drift turnover "
           f"({R['turnover']:.2f}× per year) is charged at 5 bps one-way × NAV. The short leg "
           "pays borrow, which is an **assumption**, so it is swept."),
        code(
            "print('A. own ANGL instead of HYG: excess-Sharpe gain %+.3f, drawdown %.1f%% -> %.1f%%'\n"
            "      % (R['swap_gain'], R['hyg'][3], R['angl'][3]))\n"
            "print()\n"
            "print('B. long ANGL / short HYG, monthly reset, turnover %.2fx/yr' % R['turnover'])\n"
            "print('%8s %10s %8s %9s %8s' % ('borrow','alpha/yr','HAC t','mean/yr','Sharpe'))\n"
            "for b, a, t, m, s in R['borrow']:\n"
            "    flag = '   <- realistic ETF borrow' if b == 50 else ''\n"
            "    print('%6d bp %+9.2f%% %+8.2f %+8.2f%% %+8.3f%s' % (b, a, t, m, s, flag))"
        ),
        md("## 8. Synthetic control — the estimator is unbiased\n\n"
           "A three-rung ladder built from a duration factor and an equity factor, with a "
           "planted premium on the crossover rung. It must be recovered when present and "
           "absent when not. **Synthetic only — this never supports a real-tape stamp.**"),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from crossover_credit import data, strategy as st\n"
            "planted, truth = data.synthetic_panel(signal_strength=1.0, seed=951)\n"
            "d = st.synthetic_detect(planted)\n"
            "print('planted %+.2f%%/yr -> recovered %+.2f%%/yr (t = %+.2f)'\n"
            "      % (truth['alpha_planted']*100, d['alpha_ann']*100, d['t_alpha']))\n"
            "print('recovered betas: dur %.2f, eq %.2f  (planted rung is dur 0.35 / eq 0.35 vs HY 0.25 / 0.40)'\n"
            "      % (d['betas']['dur'], d['betas']['eq']))\n"
            "nulls = [st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=951+s)[0])\n"
            "         for s in range(8)]\n"
            "a = np.array([x['alpha_ann'] for x in nulls]) * 100\n"
            "t = np.array([x['t_alpha'] for x in nulls])\n"
            "print('null x8: alpha mean %+.2f%%/yr (sd %.2f), |t| >= 2 fires %d/8'\n"
            "      % (a.mean(), a.std(ddof=1), (abs(t) >= 2).sum()))\n"
            "half = st.synthetic_detect(data.synthetic_panel(signal_strength=0.5, seed=951)[0])\n"
            "print('half strength: %+.2f%%/yr (t = %+.2f) -- the knob is monotone'\n"
            "      % (half['alpha_ann']*100, half['t_alpha']))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** The excess-Sharpe ladder humps on the boundary "
           f"({R['angl'][0]:+.3f} crossover vs {R['hyg'][0]:+.3f} broad HY, "
           f"{R['lqd'][0]:+.3f} IG) and the crossover premium's sign is positive in every era, "
           f"every fund pair and every bandwidth. It fails on robustness, not on sign. "
           f"(i) The headline duration- and equity-adjusted alpha vs broad high yield is "
           f"**{R['ah_alpha']:+.2f}%/yr at HAC *t* = {R['ah_t']:+.2f}** — on the bar, with a "
           f"bootstrap CI of [{R['ah_ci'][0]:+.2f}%, {R['ah_ci'][1]:+.2f}%] whose lower bound "
           f"is {R['ah_ci'][0]:.2f} pp — and it survives only {R['jk_pass']}/{R['jk_n']} "
           f"one-year-out deletions ({R['jk_drop16'][1]:+.2f} without 2016; "
           f"{R['drop_both'][1]:+.2f} without both wave years). (ii) It is a first-era effect: "
           f"{R['era_e_ah']:+.2f}%/yr (*t* = {R['era_e_ah_t']:+.2f}) before 2019 against "
           f"{R['era_l_ah']:+.2f}%/yr (*t* = {R['era_l_ah_t']:+.2f}) after, despite the late "
           f"era containing the record downgrade wave. (iii) Both legs swapped for siblings: "
           f"{R['faln_ushy'][0]:+.2f}%/yr (*t* = {R['faln_ushy'][1]:+.2f}) — but on that same "
           f"2017-2026 window the headline pair itself reads only "
           f"{R['same_win'][0][1]:+.2f}%/yr (*t* = {R['same_win'][0][2]:+.2f}), so this is "
           f"(ii) restated plus about 1.2 pp of fund-identity noise, not a third independent "
           f"failure: two pieces of adverse evidence, not three. (iv) The IG side of "
           f"the boundary — the half that makes it a *rung* — is absent: "
           f"{R['al_alpha']:+.2f}%/yr (*t* = {R['al_t']:+.2f}), CI [{R['al_ci'][0]:+.2f}%, "
           f"{R['al_ci'][1]:+.2f}%], and {R['era_l_al']:+.2f}%/yr recently. The synthetic "
           f"control recovers a planted {R['syn_planted']:.2f}%/yr at *t* = "
           f"{R['syn_t']:+.2f} and fires {R['syn_null_fire']}/{R['syn_null_n']} on the null, "
           f"so the borderline reading is the tape's, not the harness's. *Survivorship: six "
           f"living ETFs; failed crossover funds are not on this tape. One US credit history, "
           f"two downgrade waves.*\n"
           f"- **Tradability — Fragile.** Not a cost mirage: expression A is one trade with "
           f"≤ 1 bp/yr amortised drag, no borrow, fees inside the tape, and it delivered "
           f"{R['swap_gain']:+.3f} of excess Sharpe. Fragile because the admission price is a "
           f"{abs(R['angl'][3]) - abs(R['hyg'][3]):.1f} pp deeper worst drawdown "
           f"({R['hyg'][3]:.1f}% → {R['angl'][3]:.1f}%) and 18% more vol, for a premium that is "
           f"episodic, era-contingent and not measurable at all on the post-2017 tape; and the pure "
           f"long/short expression decays to *t* = {R['borrow'][2][2]:+.2f} at a realistic "
           f"{R['borrow'][2][0]:.0f} bps borrow. A tilt, not a harvest."),
    ]
    nb["cells"] = cells
    return nb


def main():
    for name, nb in [("01_for_the_curious", build_curious()),
                     ("02_for_the_quants", build_quants())]:
        path = os.path.join(HERE, f"{name}.ipynb")
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("wrote", path)


if __name__ == "__main__":
    main()
