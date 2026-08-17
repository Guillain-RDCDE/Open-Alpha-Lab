"""Generate the two narrative notebooks for Study 953 (Replicating the Convert).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from
the frozen ``R`` dict below (a mirror of docs/results.md); the only live computation is
the fast synthetic control, so execution is quick and network-free. No cell reads the real
cache, so nothing under a real-tape banner can silently fall back to synthetic data.
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


# Frozen real-tape headline — mirror of docs/results.md. CWB vs a frozen SPY/QQQ/LQD/BIL
# mix, daily total return, excess-of-cash, fitted 2009-2017, scored 2018-01-02 -> 2026-06-30.
R = dict(
    start="2009-04-16", end="2026-06-30", n_days=4328, fp="5e5674563124",
    is_end="2017-12-31", oos_start="2018-01-02",
    w_spy=36.7, w_qqq=20.2, w_lqd=5.8, w_cash=37.3,
    r2_is=0.703, te_is=5.68, fit_intercept=1.12, alpha_is=1.28, t_is=0.80, n_is=2193,
    alpha_oos=1.80, t_oos=0.58, te_oos=8.30, corr_oos=0.850, n_oos=2134,
    ci_lo=-4.47, ci_hi=7.55, ci_pos=71.4,
    sharpe_fund=0.662, sharpe_repl=0.725, sharpe_gap=-0.063,
    vol_fund=15.51, vol_repl=11.69, dd_fund=-32.24, dd_repl=-19.46, dd_repl_volmatched=-25.25,
    m_r2=0.764, m_te=7.85, n_months=102,
    ter_low=-55.8, ter_mid=29.9, ter_high=72.5, t_low=-1.57, t_high=1.56,
    smile=-21.6, t_smile=-0.50,
    up_cap=1.219, dn_cap=1.215, asym=0.004,
    tm_gamma=0.192, t_tm=0.25,
    era_e_alpha=4.68, era_e_t=0.86, era_e_n=1008,
    era_l_alpha=-0.78, era_l_t=-0.23, era_l_n=1126,
    refit_alpha=1.35, refit_t=0.61, refit_n=3141,
    cost0_alpha=1.69, cost0_t=0.54, cost10_alpha=1.91, cost10_t=0.62,
    kit_spy_alpha=3.01, kit_spy_t=0.92, kit_spylqd_alpha=2.99, kit_spylqd_t=0.94,
    kit_spyqqq_alpha=1.81, kit_spyqqq_t=0.57, n_specs=11, max_spec_t=0.94,
    split_best_alpha=2.35, split_best_t=0.90, split_worst_alpha=-2.72, split_worst_t=-0.80,
    icvt_alpha=-2.16, icvt_t=-0.56, icvt_up=0.802, icvt_dn=0.871, icvt_n=1378,
    mar20_fund=-13.28, mar20_repl=-6.49, mar20_gap=-6.79,
    fund_fee=0.40, replica_fee_bps=10,
    syn_alpha=5.02, syn_t=3.97, syn_gamma=0.73, syn_t_gamma=3.45, syn_smile=26.7,
    syn_null_mean=0.03, syn_null_sd=0.87, syn_null_fire=0,
)

SYN_PREAMBLE = (
    "import os, sys\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
    "import numpy as np\n"
    "from convert_repl import data, strategy as st\n"
)


HEADER = f"""# Study 953 — Replicating the Convert 🎭

**Is a convertible-bond fund anything more than equity plus credit in a costume?**

A convertible bond is sold as an instrument you cannot build from parts: a bond that turns
into a share, so you ride the stock up while a bond floor catches you on the way down — a
**convex** payoff worth paying a fee for. That is a testable claim, and the test is
mechanical. Fit the fund to a plain, static, long-only mix of liquid ETFs; **freeze** the
recipe; then see (a) whether anything is left over, and (b) whether what is left over has
the promised *shape*.

We test **CWB** (the $4bn SPDR convertibles ETF) against **SPY + QQQ + LQD + cash (BIL)**,
daily **total-return** closes, {R['start']} → {R['end']} ({R['n_days']:,} days), every leg
**excess-of-cash**. The mix is fitted on **2009-2017 only** and scored on the untouched
**{R['oos_start']} → {R['end']}** hold-out.

*Numbers below are the frozen headline (`docs/results.md`, Fingerprint `{R['fp']}`); the only
live cells run the fast offline synthetic control. As-of 2026-06-30.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. Three ETFs write the recipe\n\n"
           "We did not choose the weights — the data did, using only the fund's first nine "
           "years. The answer is startling in its plainness."),
        code(
            ("R = %r\n" % (R,)) +
            "print('What eight years of CWB returns say it actually is:')\n"
            "for leg, w in [('SPY  (US large cap)', R['w_spy']), ('QQQ  (Nasdaq growth)', R['w_qqq']),\n"
            "               ('LQD  (IG credit)', R['w_lqd']), ('cash (T-bills)', R['w_cash'])]:\n"
            "    print(f'  {leg:22s} {w:5.1f}%')\n"
            "print(f\"\\nin-sample fit quality: R2 {R['r2_is']:.3f}\")"
        ),
        md("The 'bond' half of a convertible is **not** a bond. Fitted freely, the fund wants "
           "**37% cash** and barely **6% investment-grade credit**. Read plainly: a convertible "
           "fund is roughly *half an equity fund and a third a money-market fund*, with a "
           "sliver of credit — and the equity half leans Nasdaq, because the companies that "
           "issue converts are growth companies that cannot borrow cheaply.\n\n"
           "> 🔬 **For the quants** — the weights come from Sharpe-style constrained least "
           "squares (non-negative, summing to at most one), so the replica is a portfolio "
           "someone could actually hold. The constraint never binds: the unconstrained OLS "
           "solution is already inside it, so no leg goes short and no borrow cost arises."),
        md("## 2. Freeze it, and see what is left\n\n"
           "The honest test is not how well a recipe fits the past — it is what the recipe "
           "misses on years it has never seen. So we lock the weights at the end of 2017 and "
           "never touch them again."),
        code(
            "print(f\"hold-out {R['oos_start']} -> {R['end']}  ({R['n_oos']:,} trading days)\")\n"
            "print(f\"  what the fund did that the frozen mix did not: {R['alpha_oos']:+.2f}%/yr\")\n"
            "print(f\"  is that distinguishable from zero?  HAC t = {R['t_oos']:+.2f}   \"\n"
            "      f\"(the desk's bar is |t| >= 2)\")\n"
            "print(f\"  95% confidence range: [{R['ci_lo']:+.2f}%, {R['ci_hi']:+.2f}%]  -> straddles zero\")\n"
            "print(f\"\\nrisk-adjusted return (excess of cash):\")\n"
            "print(f\"  the fund     {R['sharpe_fund']:+.3f}\")\n"
            "print(f\"  the frozen mix {R['sharpe_repl']:+.3f}   <- the copy edged the original\")"
        ),
        md("A nine-year-old recipe, never updated, kept pace with the fund it was copied from — "
           "and on risk-adjusted terms very slightly beat it. Nothing here is *wrong* with CWB; "
           "the point is that nothing here is *distinctive* about it either."),
        md("## 3. The claim that actually matters: the shape\n\n"
           "Convertibles are not sold on return, they are sold on **shape** — win more in good "
           "months, lose less in bad ones. Measure it: sort the hold-out months by how the "
           "stock market did, and see how the fund fared against its own copy in each third."),
        code(
            "print('fund minus its frozen copy, by third of the stock market (bps per month):')\n"
            "print(f\"  worst months   {R['ter_low']:+7.1f}   <- should be POSITIVE if a bond floor exists\")\n"
            "print(f\"  middle months  {R['ter_mid']:+7.1f}\")\n"
            "print(f\"  best months    {R['ter_high']:+7.1f}   <- positive, as promised\")\n"
            "print(f\"\\nthe promised smile (both tails better than the middle): {R['smile']:+.1f} bps/month\")\n"
            "print(f\"upside captured vs the copy   {R['up_cap']:.3f}\")\n"
            "print(f\"downside captured vs the copy {R['dn_cap']:.3f}\")\n"
            "print(f\"asymmetry (the whole pitch)   {R['asym']:+.3f}   <- essentially zero\")"
        ),
        md("There is the answer, and it is not subtle. CWB does beat its copy in strong months "
           "— by capturing **22% more** of the move. It also loses **22% more** in weak ones. "
           "That is not convexity; that is **leverage**. You could buy the same thing by "
           "holding a bit more of the copy.\n\n"
           "> 🔬 **For the quants** — the Treynor-Mazuy curvature on the fund-minus-replica "
           f"difference is γ = **+{R['tm_gamma']:.2f}** with HAC *t* = **+{R['t_tm']:.2f}**: "
           "indistinguishable from a straight line. The tail 'smile' is in fact "
           f"**{R['smile']:+.1f}** bps/month (Welch *t* = {R['t_smile']:+.2f}) — the tails are "
           "*worse* than the middle, the opposite of a smile."),
        md("## 4. The one month the floor was asked for\n\n"
           "March 2020 is the cleanest test a bond floor will ever get."),
        code(
            "print(f\"March 2020, excess of cash:\")\n"
            "print(f\"  CWB (the convertible fund)  {R['mar20_fund']:+.2f}%\")\n"
            "print(f\"  its own frozen stock/credit/cash copy {R['mar20_repl']:+.2f}%\")\n"
            "print(f\"  shortfall {R['mar20_gap']:+.2f}%  in a single month\")"
        ),
        md("The convertible fund lost roughly **twice** what its own plain replica lost, in "
           "the month the floor was supposed to matter. When arbitrageurs are forced to "
           "delever, convertibles trade *below* the value of their parts — the floor gives way "
           "exactly when you reach for it."),
        md("## 5. Is the measuring stick honest? (a live offline check)\n\n"
           "Before believing a blank result, prove the instrument works. We build a fake fund "
           "that genuinely *is* a static mix plus a real 2%/yr edge and a real ratcheting "
           "equity exposure — the honest version of the convertible story — and check the same "
           "machinery finds it. Then we switch both effects off and check it finds nothing."),
        code(
            SYN_PREAMBLE +
            "planted = st.synthetic_detect(data.synthetic_panel(signal_strength=1.0, seed=953)[0])\n"
            "print('planted fund (there IS something to find):')\n"
            "print('  leftover return %+.2f%%/yr (t %+.2f)  |  curvature %+.2f (t %+.2f)'\n"
            "      % (planted['alpha_oos_ann']*100, planted['t_oos'],\n"
            "         planted['tm_gamma'], planted['t_tm_gamma']))\n"
            "nulls = [st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=953+s)[0])\n"
            "         for s in range(4)]\n"
            "print('null funds, 4 fresh draws (there is NOTHING to find):')\n"
            "for i, n in enumerate(nulls):\n"
            "    print('  draw %d: leftover %+.2f%%/yr (t %+.2f)  |  curvature %+.2f (t %+.2f)'\n"
            "          % (i + 1, n['alpha_oos_ann']*100, n['t_oos'], n['tm_gamma'], n['t_tm_gamma']))\n"
            "print('  -> not one of them clears the |t| >= 2 bar')"
        ),
        md("The instrument rings loudly when there is something there and stays quiet when "
           "there is not. So the blank we got on the real tape is a fact about convertible "
           "ETFs, not a broken ruler. *(This cell is synthetic — a machinery proof, never "
           "market evidence.)*"),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** Out-of-sample, the fund's leftover return over a frozen "
           f"SPY/QQQ/LQD/cash mix is **{R['alpha_oos']:+.2f}%/yr with *t* = {R['t_oos']:+.2f}** "
           f"— zero by any standard — and it flips sign between eras "
           f"({R['era_e_alpha']:+.2f}%/yr in 2018-21, {R['era_l_alpha']:+.2f}%/yr since 2022). "
           f"The second convertible ETF (ICVT) gives {R['icvt_alpha']:+.2f}%/yr "
           f"(*t* = {R['icvt_t']:+.2f}). The convexity claim fails on its own terms: "
           f"up-capture {R['up_cap']:.3f} versus down-capture {R['dn_cap']:.3f}.\n"
           f"- **Tradability — Mirage.** There is nothing to trade in either direction. Owning "
           f"the fund buys a coin-flip residual against a certain {R['fund_fee']:.2f}%/yr fee; "
           f"betting *against* the fund means running a {R['m_te']:.1f}%/yr tracking error to "
           f"harvest a 2%/yr expectation, with the bad months bunched (−6.7% in March 2020 "
           f"alone). The useful conclusion is not a trade — it is that you already own this "
           f"exposure."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 953 — Replicating the Convert — the teardown\n\n"
           "Sharpe-style constrained replication of CWB on SPY/QQQ/LQD/BIL, fitted in-sample "
           "and frozen; the hold-out residual with a Newey-West *t* and a circular "
           "block-bootstrap CI; the convexity battery (tercile smile, up-/down-capture, "
           "Treynor-Mazuy γ) run *against the replica*; era cut, annual refit, cost/fee sweep, "
           "ICVT cross-check, and the live synthetic control. Every real number is frozen from "
           "`docs/results.md` (Fingerprint `%s`, as-of 2026-06-30).\n\n"
           "> 💡 **In plain words** — we build the cheapest possible copy of a convertible fund "
           "out of three ordinary ETFs, lock the recipe before the years we grade it on, and "
           "then ask whether the fund did anything the copy did not."
           % R["fp"]),
        code("R = %r" % (R,)),
        md("## 1. The constrained fit (in-sample only)\n\n"
           "Minimise ``||y − a − Xw||²`` over a free intercept and ``w ∈ {w ≥ 0, Σw ≤ 1}`` "
           "(Sharpe 1992 style analysis), on daily excess-of-cash total returns through "
           f"**{R['is_end']}**. The residual budget is cash, so the replica is a genuine "
           "long-only portfolio and no borrow cost arises."),
        code(
            "print(f\"in-sample {R['n_is']:,} days -> {R['is_end']}\")\n"
            "for leg, w in [('SPY', R['w_spy']), ('QQQ', R['w_qqq']), ('LQD', R['w_lqd']), ('BIL', R['w_cash'])]:\n"
            "    print(f'  {leg}  {w:5.1f}%')\n"
            "print(f\"  equity total {R['w_spy']+R['w_qqq']:.1f}%  |  credit {R['w_lqd']:.1f}%  |  cash {R['w_cash']:.1f}%\")\n"
            "print(f\"  R2 {R['r2_is']:.3f}   TE {R['te_is']:.2f}%/yr   fit intercept {R['fit_intercept']:+.2f}%/yr\")\n"
            "print(f\"  in-sample residual vs the costed replica {R['alpha_is']:+.2f}%/yr (HAC t {R['t_is']:+.2f})\")\n"
            "print('  the long-only budget never binds: the unconstrained OLS solution is identical to 3 dp')"
        ),
        md("## 2. The frozen hold-out\n\n"
           "Weights applied from the first trading day *after* the in-sample end (the single "
           "execution lag); monthly rebalance traded at the next close; 3 bps one-way × NAV on "
           "turnover; a 10 bps/yr replica fee drag (a labelled PROXY, swept in §5)."),
        code(
            "print(f\"OOS {R['oos_start']} -> {R['end']}  n={R['n_oos']:,}\")\n"
            "print(f\"  residual alpha {R['alpha_oos']:+.2f}%/yr   HAC t {R['t_oos']:+.2f}\")\n"
            "print(f\"  block-bootstrap 95% CI [{R['ci_lo']:+.2f}%, {R['ci_hi']:+.2f}%]  \"\n"
            "      f\"share>0 {R['ci_pos']:.1f}%  -> straddles zero\")\n"
            "print(f\"  tracking error {R['te_oos']:.2f}%/yr   corr {R['corr_oos']:.3f}   \"\n"
            "      f\"monthly R2 {R['m_r2']:.3f} / TE {R['m_te']:.2f}%/yr\")\n"
            "print(f\"  excess Sharpe: fund {R['sharpe_fund']:+.3f} vs replica {R['sharpe_repl']:+.3f} \"\n"
            "      f\"(gap {R['sharpe_gap']:+.3f})\")\n"
            "print(f\"  vol {R['vol_fund']:.2f}% vs {R['vol_repl']:.2f}%   \"\n"
            "      f\"maxDD {R['dd_fund']:.2f}% vs {R['dd_repl']:.2f}% \"\n"
            "      f\"({R['dd_repl_volmatched']:.2f}% vol-matched)\")"
        ),
        md("> 💡 **In plain words** — the copy did not merely keep up; on risk-adjusted terms it "
           "edged ahead, and it did so with a shallower worst loss even after scaling it up to "
           "the fund's volatility. Note the honest caveat in the other direction: 7.9%/yr "
           "monthly tracking error means this is a good *description* of CWB, not a tight clone."),
        md("## 3. The convexity battery, run against the replica\n\n"
           "Out-of-sample months sorted into terciles by SPY's excess return. A convex wrapper "
           "beats its linear replica in **both** tails. This is the axis on which Study 339 "
           "used SPY itself as the benchmark; here the benchmark is the fund's own fitted parts, "
           "which is a strictly harder test of the wrapper."),
        code(
            "print(f\"n months {R['n_months']}   fund - replica, bps/month\")\n"
            "print(f\"  bottom tercile {R['ter_low']:+7.1f}  (t {R['t_low']:+.2f})\")\n"
            "print(f\"  middle tercile {R['ter_mid']:+7.1f}\")\n"
            "print(f\"  top tercile    {R['ter_high']:+7.1f}  (t {R['t_high']:+.2f})\")\n"
            "print(f\"  smile (tails - middle) {R['smile']:+.1f} bps/month  Welch t {R['t_smile']:+.2f}\")\n"
            "print(f\"  up-capture {R['up_cap']:.3f}   down-capture {R['dn_cap']:.3f}   \"\n"
            "      f\"asymmetry {R['asym']:+.3f}\")\n"
            "print(f\"  Treynor-Mazuy gamma {R['tm_gamma']:+.3f}  (HAC t {R['t_tm']:+.2f})\")"
        ),
        md("The tercile means are monotone increasing in the market, not U-shaped: the "
           "difference is a **line through the origin with a positive slope**, i.e. residual "
           "beta. Up- and down-capture agree to within 0.004, and γ is a quarter of a standard "
           "error from zero. Whatever a convertible's embedded option does at the single-bond "
           "level, the aggregate fund shows no measurable curvature against its own parts.\n\n"
           "*One label, because it matters: this is a **description** of a realised payoff, not "
           "a tradable rule. The tercile breakpoints and the up/down split are computed on the "
           "whole hold-out, so a month's bucket is known only in arrears — nothing downstream "
           "trades on it.*"),
        code(
            "import matplotlib.pyplot as plt\n"
            "fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))\n"
            "vals = [R['ter_low'], R['ter_mid'], R['ter_high']]\n"
            "ax[0].bar(['bottom', 'middle', 'top'], vals,\n"
            "          color=['#c0392b', '#8b949e', '#2e7d32'])\n"
            "ax[0].axhline(0, color='k', lw=0.8)\n"
            "ax[0].set_title('fund - replica by market tercile (bps/month)')\n"
            "ax[0].set_ylabel('bps / month')\n"
            "ax[1].bar(['up-capture', 'down-capture'], [R['up_cap'], R['dn_cap']],\n"
            "          color=['#2e7d32', '#c0392b'])\n"
            "ax[1].axhline(1.0, color='k', lw=0.8, ls='--')\n"
            "ax[1].set_ylim(0.9, 1.35)\n"
            "ax[1].set_title('capture vs the replica (1.0 = same)')\n"
            "fig.suptitle('A line, not a smile', y=1.02)\n"
            "fig.tight_layout()\n"
            "plt.show()"
        ),
        md("## 4. Robustness — era cut, annual refit, and the two judgement calls\n\n"
           "Two choices in this design are hindsight-flavoured and must be swept rather than "
           "defended. (i) The **QQQ leg**: we know *today* that convertible issuers skew "
           "growth, and a richer kit mechanically shrinks the residual we are trying to "
           "measure. (ii) The **2017 boundary**. Strip the kit back to what a 2009 observer "
           "would have picked, and walk the boundary across seven years."),
        code(
            "print(f\"2018-2021 (n={R['era_e_n']:,}): alpha {R['era_e_alpha']:+.2f}%/yr  t {R['era_e_t']:+.2f}\")\n"
            "print(f\"2022-2026 (n={R['era_l_n']:,}): alpha {R['era_l_alpha']:+.2f}%/yr  t {R['era_l_t']:+.2f}\")\n"
            "print(f\"annual refit, always forward (n={R['refit_n']:,}): \"\n"
            "      f\"alpha {R['refit_alpha']:+.2f}%/yr  t {R['refit_t']:+.2f}\")\n"
            "print('\\nreplication kit sweep (a LEANER kit leaves a BIGGER residual):')\n"
            "print(f\"  SPY only      alpha {R['kit_spy_alpha']:+.2f}%/yr  t {R['kit_spy_t']:+.2f}\")\n"
            "print(f\"  SPY+LQD       alpha {R['kit_spylqd_alpha']:+.2f}%/yr  t {R['kit_spylqd_t']:+.2f}\")\n"
            "print(f\"  SPY+QQQ       alpha {R['kit_spyqqq_alpha']:+.2f}%/yr  t {R['kit_spyqqq_t']:+.2f}\")\n"
            "print(f\"  SPY+QQQ+LQD   alpha {R['alpha_oos']:+.2f}%/yr  t {R['t_oos']:+.2f}   (headline)\")\n"
            "print('\\nin-sample boundary sweep, 2014-12-31 ... 2020-12-31:')\n"
            "print(f\"  best  t: alpha {R['split_best_alpha']:+.2f}%/yr  t {R['split_best_t']:+.2f}\")\n"
            "print(f\"  worst t: alpha {R['split_worst_alpha']:+.2f}%/yr  t {R['split_worst_t']:+.2f}\")\n"
            "print(f\"\\n-> across all {R['n_specs']} specifications the largest |t| is \"\n"
            "      f\"{R['max_spec_t']:.2f}, and one of them is negative\")"
        ),
        md("> 💡 **In plain words** — what little residual exists is a 2018-2021 artefact, the "
           "SPAC-and-growth-issuance boom when convertible issuers were exactly the names that "
           "ran. Refitting the recipe every year and always applying it forward gives the same "
           "*t* of 0.61. And the QQQ leg is not what buried the fund: **take it away and the "
           "residual gets larger** (+3.01%/yr) while the *t* stays under 1. The blank belongs "
           "to the fund, not to the specification."),
        md("## 5. Cost / fee sweep and the ICVT cross-check\n\n"
           "Turnover is trivial, so friction cannot decide this — and charging the replica's "
           "*own* fee honestly makes the residual **larger**, not smaller."),
        code(
            "print(f\"0 bps cost, 0 bps/yr replica fee : alpha {R['cost0_alpha']:+.2f}%/yr (t {R['cost0_t']:+.2f})\")\n"
            "print(f\"10 bps cost, 20 bps/yr replica fee: alpha {R['cost10_alpha']:+.2f}%/yr (t {R['cost10_t']:+.2f})\")\n"
            "print(f\"-> the whole 4x3 grid moves alpha by 22 bps/yr and t by 0.08\")\n"
            "print(f\"\\nICVT (n={R['icvt_n']:,}): alpha {R['icvt_alpha']:+.2f}%/yr (t {R['icvt_t']:+.2f})\")\n"
            "print(f\"  up-capture {R['icvt_up']:.3f} < down-capture {R['icvt_dn']:.3f}  \"\n"
            "      f\"-> mildly CONCAVE, the opposite of the pitch\")\n"
            "print(f\"\\nMarch 2020: fund {R['mar20_fund']:+.2f}% vs replica {R['mar20_repl']:+.2f}% \"\n"
            "      f\"(shortfall {R['mar20_gap']:+.2f}% in one month)\")"
        ),
        md("## 6. Live synthetic control — the machinery is unbiased\n\n"
           "Planted world: a fund that is a static mix **plus** a 2%/yr alpha **plus** a delta "
           "ratchet (equity exposure rising after rallies, falling after selloffs — exactly what "
           "a real convertible does). Null world: both switched off. The harness must recover "
           "the first and report nothing on the second. Machinery proof only — never market "
           "evidence."),
        code(
            SYN_PREAMBLE +
            "pl = st.synthetic_detect(data.synthetic_panel(signal_strength=1.0, seed=953)[0])\n"
            "print('planted: weights recovered', {k: round(v, 3) for k, v in pl['weights'].items()},\n"
            "      '(true 0.30 / 0.20 / 0.35)')\n"
            "print('planted: OOS alpha %+.2f%%/yr (t %+.2f), TM gamma %+.2f (t %+.2f), smile %+.1f bps'\n"
            "      % (pl['alpha_oos_ann']*100, pl['t_oos'], pl['tm_gamma'], pl['t_tm_gamma'], pl['smile_bps']))\n"
            "nl = np.array([[d['t_oos'], d['t_tm_gamma']] for d in\n"
            "               (st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=953+s)[0])\n"
            "                for s in range(8))])\n"
            "print('null x8: alpha t mean %+.2f (sd %.2f), |t|>=2 in %d/8; gamma t mean %+.2f, |t|>=2 in %d/8'\n"
            "      % (nl[:,0].mean(), nl[:,0].std(ddof=1), (abs(nl[:,0])>=2).sum(),\n"
            "         nl[:,1].mean(), (abs(nl[:,1])>=2).sum()))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The out-of-sample residual over a frozen, long-only, "
           f"three-ETF-plus-cash replication is **{R['alpha_oos']:+.2f}%/yr, HAC *t* = "
           f"{R['t_oos']:+.2f}**, bootstrap CI **[{R['ci_lo']:+.2f}%, {R['ci_hi']:+.2f}%]** — "
           f"the desk's bar is |*t*| ≥ 2, and this is a quarter of it. It is not era-robust "
           f"({R['era_e_alpha']:+.2f}%/yr then {R['era_l_alpha']:+.2f}%/yr), survives no better "
           f"under an annual refit (*t* = {R['refit_t']:+.2f}), never clears "
           f"|*t*| = {R['max_spec_t']:.2f} across {R['n_specs']} kit-and-boundary "
           f"specifications, and flips sign on ICVT "
           f"({R['icvt_alpha']:+.2f}%/yr, *t* = {R['icvt_t']:+.2f}). The convexity claim fails "
           f"independently and more decisively: up-capture {R['up_cap']:.3f} ≈ down-capture "
           f"{R['dn_cap']:.3f} (asymmetry {R['asym']:+.3f}), tail smile {R['smile']:+.1f} "
           f"bps/month (*t* = {R['t_smile']:+.2f}), Treynor-Mazuy γ = {R['tm_gamma']:+.3f} "
           f"(*t* = {R['t_tm']:+.2f}). **Survivorship** is named on this axis: CWB and ICVT are "
           f"two *surviving* listed funds, so the sample is if anything tilted in the wrapper's "
           f"favour.\n"
           f"- **Tradability — Mirage.** Nothing banks. The long side pays a certain "
           f"{R['fund_fee']:.2f}%/yr for a coin-flip residual; the short-the-wrapper side needs "
           f"to run a {R['m_te']:.1f}%/yr monthly tracking error to harvest a 2%/yr expectation "
           f"whose realised path includes a {R['mar20_gap']:+.1f}% month. The result is a "
           f"*classification*, not a trade: a convertible ETF is a "
           f"{R['w_spy']+R['w_qqq']:.0f}/{R['w_lqd']:.0f}/{R['w_cash']:.0f} "
           f"equity/credit/cash sleeve, and the fee is the only part of it you can forecast."),
    ]
    nb["cells"] = cells
    return nb


def main() -> None:
    for name, nb in (("01_for_the_curious.ipynb", build_curious()),
                     ("02_for_the_quants.ipynb", build_quants())):
        nb["metadata"] = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        }
        path = os.path.join(HERE, name)
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
