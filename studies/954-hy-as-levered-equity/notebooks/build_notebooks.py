"""Generate the two narrative notebooks for Study 954 (High Yield in Disguise).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every **real-tape** number is quoted from the
frozen ``R`` dict below (a mirror of ``docs/results.md``); the only live cells run the fast
offline **synthetic** control, and they are always introduced as synthetic — never under a
real-tape banner.
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
# Frozen real-tape headline — the single mirror of docs/results.md.
# HYG vs a held-out w*SPY + (1-w)*IEF blend, excess-of-cash (BIL), total return,
# 252-day estimation window, month-end freeze, 2 bps one-way, 2008-06-02 -> 2026-06-30.
# --------------------------------------------------------------------------- #
R = dict(
    stamp_start="2007-05-30", stamp_end="2026-06-30", stamp_n=4802, fp="c08a8d88f0f9",
    start="2008-06-02", end="2026-06-30", n_days=4548,
    w_mean=0.450, w_min=0.269, w_max=0.631, short_max=0.000, turnover=0.34,
    r2_d=0.468, r2_w=0.565, r2_m=0.517, r2_q=0.591,
    te_d=8.15, te_w=7.45, te_m=7.18, te_q=6.26,
    resid_ann=-2.16, t_resid=-1.25,
    # hy_cagr / rp_cagr are EXCESS-of-cash CAGRs (the series the Sharpe race is run on);
    # *_cagr_lived are the absolute total-return CAGRs a holder actually saw. Cash (BIL)
    # compounded at +1.26%/yr, which is exactly the difference between the two.
    hy_sharpe=0.395, hy_cagr=3.85, hy_cagr_lived=5.17, hy_vol=11.12, hy_dd=-32.40, hy_t=1.73,
    rp_sharpe=0.731, rp_cagr=6.34, rp_cagr_lived=7.68, rp_vol=8.96, rp_dd=-20.70, rp_t=3.39,
    cash_cagr_lived=1.26,
    gap=-0.336, t_gap=-2.07, gap_pp_per_yr=3.7,
    ci_hy_lo=-0.049, ci_hy_hi=0.900, ci_hy_neg=4.4,
    ci_rp_lo=0.314, ci_rp_hi=1.205, ci_rp_neg=0.1,
    ci_gap_pt=-0.427, ci_gap_lo=-0.805, ci_gap_hi=-0.035, ci_gap_neg=98.4,
    era_e_n=2163, era_e_hy=0.480, era_e_rp=0.770, era_e_gap=-0.290, era_e_t=-1.19,
    era_e_resid=-1.07, era_e_tres=-0.34,
    era_l_n=2385, era_l_hy=0.298, era_l_rp=0.696, era_l_gap=-0.398, era_l_t=-2.05,
    era_l_resid=-3.15, era_l_tres=-1.97,
    c08_dd_hy=-32.4, c08_dd_rp=-20.7, c08_ret_hy=-9.2, c08_ret_rp=-8.9,
    c20_dd_hy=-22.0, c20_dd_rp=-13.1, c20_ret_hy=-6.9, c20_ret_rp=1.4,
    c22_dd_hy=-15.5, c22_dd_rp=-18.9, c22_ret_hy=-11.0, c22_ret_rp=-15.7,
    cost0_gap=-0.337, cost0_t=-2.07, cost25_gap=-0.328, cost25_t=-2.02,
    win126_gap=-0.250, win126_t=-1.58, win252_gap=-0.336, win252_t=-2.07,
    win504_gap=-0.373, win504_t=-2.32, win756_gap=-0.389, win756_t=-2.41,
    # duration-leg sweep (same 4,548-day common sample): SHY 1-3y, IEI 3-7y,
    # IEF 7-10y (headline), TLT 20y+ — Treasuries only, so no credit enters the bench.
    leg_shy_w=0.357, leg_shy_gap=-0.297, leg_shy_t=-1.93, leg_shy_resid=-0.62,
    leg_iei_w=0.393, leg_iei_gap=-0.323, leg_iei_t=-2.07, leg_iei_resid=-1.33,
    leg_ief_w=0.450, leg_ief_gap=-0.336, leg_ief_t=-2.07, leg_ief_resid=-2.16,
    leg_tlt_w=0.614, leg_tlt_gap=-0.344, leg_tlt_t=-1.92, leg_tlt_resid=-4.19,
    jnk_n=4399, jnk_w=0.443, jnk_r2=0.487, jnk_gap=-0.268, jnk_t=-1.64,
    ushy_n=1923, ushy_w=0.367, ushy_r2=0.630, ushy_gap=-0.286, ushy_t=-1.24,
    er_hyg=0.49, er_blend=0.125, er_diff=0.365,
)


HEADER = f"""# Study 954 — High Yield in Disguise 🎭

**Is a high-yield bond fund just equity and Treasuries in a costume — and if not, does the
difference pay?**

The trade-desk aphorism says a junk bond is a senior claim on a leveraged company: equity
risk on the way down, capped on the way up. If that is the whole story, a high-yield fund
should be reproducible as a simple blend of **SPY** (equity) and **IEF** (Treasury
duration) — and you would be paying a bond fund's fee for a mixture you could hold yourself.

We fit that blend to **HYG** out of sample — a trailing 252-day constrained regression,
the weight frozen at each month-end and applied to the *following* month — and race the two
**excess-of-cash** (BIL), over {R['start']} → {R['end']} ({R['n_days']:,} days) of daily
**total-return** closes, 2 bps one-way on the blend's rebalance.

*Every real-tape number below is the frozen headline from `docs/results.md`
(Fingerprint `{R['fp']}`, as-of 2026-06-30). The live cells at the end run an offline
**synthetic** control and are labelled as such.*
"""


# --------------------------------------------------------------------------- #
# 01 — for the curious
# --------------------------------------------------------------------------- #
def build_curious():
    cells = [
        md(HEADER),

        md("## 1. What the fitted recipe turned out to be\n\n"
           "Solve for the equity share `w` that best explains HYG's daily moves, refit it "
           "every month, and never let the fit see the future. Over eighteen years the "
           "answer barely wandered."),
        code(
            "R = dict(w_mean=%r, w_min=%r, w_max=%r, turnover=%r, short_max=%r)\n"
            "print('fitted recipe for a high-yield fund: %%.0f%%%% equity + %%.0f%%%% Treasuries'\n"
            "      %% (R['w_mean']*100, (1-R['w_mean'])*100))\n"
            "print('the equity share ranged from %%.2f to %%.2f over 18 years' %% (R['w_min'], R['w_max']))\n"
            "print('never short, never levered (max short notional %%.3f); %%.2f x NAV traded per year'\n"
            "      %% (R['short_max'], R['turnover']))"
            % (R["w_mean"], R["w_min"], R["w_max"], R["turnover"], R["short_max"])
        ),
        md("So high yield is **not** levered equity. It is roughly **45% equity and 55% "
           "Treasuries** — *de*-levered equity, with a big bond sleeve attached.\n\n"
           "> 🔬 **For the quants:** the weight comes from regressing `r_HY − r_IEF` on "
           "`r_SPY − r_IEF`. Subtracting the duration leg from both sides imposes "
           "*weights sum to one* exactly, so the slope is the equity share of a fully "
           "funded blend — no cash is created or destroyed and no leverage sneaks in."),

        md("## 2. The costume does not fit\n\n"
           "If high yield really were the blend, the blend would explain almost all of it. "
           "It explains about half — and stretching the return horizon (which would flatter "
           "a fund whose bonds are marked stale) barely helps."),
        code(
            "R = dict(r2_d=%r, r2_w=%r, r2_m=%r, r2_q=%r, te_d=%r, te_q=%r)\n"
            "for label, r2, in [('daily', R['r2_d']), ('weekly', R['r2_w']),\n"
            "                   ('monthly', R['r2_m']), ('quarterly', R['r2_q'])]:\n"
            "    print('%%-10s the blend explains %%.0f%%%% of high yield' %% (label, r2*100))\n"
            "print()\n"
            "print('left over: %%.1f%%%% a year of tracking error (daily), %%.1f%%%% (quarterly)'\n"
            "      %% (R['te_d'], R['te_q']))"
            % (R["r2_d"], R["r2_w"], R["r2_m"], R["r2_q"], R["te_d"], R["te_q"])
        ),
        md("Roughly **7 percentage points a year** of high yield's movement is something "
           "neither the stock market nor the Treasury market does. That is credit risk — "
           "real, distinct, and yours to be paid for. **The costume story fails.**"),

        md("## 3. But the distinct risk paid nothing\n\n"
           "So the fund is not a copy. The next question is the only one that matters to an "
           "owner: for the same amount of risk taken, who ended up with more money?"),
        code(
            "R = dict(hy_sharpe=%r, rp_sharpe=%r, hy_cagr=%r, rp_cagr=%r,\n"
            "         hy_lived=%r, rp_lived=%r, cash_lived=%r,\n"
            "         hy_vol=%r, rp_vol=%r, hy_dd=%r, rp_dd=%r, gap=%r, t_gap=%r,\n"
            "         gap_pp_per_yr=%r)\n"
            "print('what the statement showed (total return, cash made %%+.2f%%%% a year):'\n"
            "      %% R['cash_lived'])\n"
            "print('  HYG, held 18 years : %%+.2f%%%% a year, %%.1f%%%% volatility, worst loss %%.1f%%%%'\n"
            "      %% (R['hy_lived'], R['hy_vol'], R['hy_dd']))\n"
            "print('  the 45/55 blend    : %%+.2f%%%% a year, %%.1f%%%% volatility, worst loss %%.1f%%%%'\n"
            "      %% (R['rp_lived'], R['rp_vol'], R['rp_dd']))\n"
            "print()\n"
            "print('the same race after subtracting cash (what the Sharpe compares):')\n"
            "print('  above cash: %%+.2f%%%% a year for the fund, %%+.2f%%%% for the blend'\n"
            "      %% (R['hy_cagr'], R['rp_cagr']))\n"
            "print('  return per unit of risk: %%.3f for the fund, %%.3f for the blend'\n"
            "      %% (R['hy_sharpe'], R['rp_sharpe']))\n"
            "print('gap %%+.3f  (statistical t = %%+.2f)  ~%%.1f pp/yr at matched volatility'\n"
            "      %% (R['gap'], R['t_gap'], R['gap_pp_per_yr']))"
            % (R["hy_sharpe"], R["rp_sharpe"], R["hy_cagr"], R["rp_cagr"],
               R["hy_cagr_lived"], R["rp_cagr_lived"], R["cash_cagr_lived"],
               R["hy_vol"], R["rp_vol"], R["hy_dd"], R["rp_dd"], R["gap"], R["t_gap"],
               R["gap_pp_per_yr"])
        ),
        md("The homemade blend won on **every** count over these eighteen years: more "
           "return, less volatility, a shallower worst loss. Part of that is simply the "
           f"fee — HYG charges **{R['er_hyg']:.2f}%** a year against **{R['er_blend']:.3f}%** "
           f"for the blend, a **{R['er_diff']:.2f} pp** head start — but the fee is only "
           "about a sixth of the gap. The rest is the credit risk failing to pay.\n\n"
           "> 🔬 **For the quants:** the *t* on that gap is "
           f"**{R['t_gap']:+.2f}** and the block-bootstrap CI is "
           f"[{R['ci_gap_lo']:+.2f}, {R['ci_gap_hi']:+.2f}] — clear of zero, but barely. "
           "The direction is unanimous across four estimation windows, four Treasury "
           "maturities, both eras and all three funds; the *significance* is not — swap "
           f"the Treasury leg for SHY ({R['leg_shy_t']:+.2f}) or TLT ({R['leg_tlt_t']:+.2f}) "
           "and the headline drops back under the bar. The magnitude is one good year away "
           "from being unremarkable. That "
           "is why the Signal stamp is Mixed and not Real."),

        md("## 4. Except in 2022 — and that is the catch\n\n"
           "Three crises, three verdicts. Two of them are lopsided wins for the blend, "
           "because when *credit* is what breaks, the blend's Treasury sleeve rallies while "
           "high-yield spreads blow out. The third goes the other way."),
        code(
            "R = dict(c08_dd_hy=%r, c08_dd_rp=%r, c20_dd_hy=%r, c20_dd_rp=%r,\n"
            "         c20_ret_hy=%r, c20_ret_rp=%r, c22_dd_hy=%r, c22_dd_rp=%r,\n"
            "         c22_ret_hy=%r, c22_ret_rp=%r)\n"
            "print('2008 credit crisis : fund %%.1f%%%% drawdown vs blend %%.1f%%%%'\n"
            "      %% (R['c08_dd_hy'], R['c08_dd_rp']))\n"
            "print('2020 Covid crash   : fund %%.1f%%%% drawdown vs blend %%.1f%%%%'\n"
            "      %% (R['c20_dd_hy'], R['c20_dd_rp']))\n"
            "print('   over that quarter the fund lost %%.1f%%%%; the blend made %%+.1f%%%%'\n"
            "      %% (R['c20_ret_hy'], R['c20_ret_rp']))\n"
            "print()\n"
            "print('2022 rate shock    : fund %%.1f%%%% drawdown vs blend %%.1f%%%%  <- the blend LOSES'\n"
            "      %% (R['c22_dd_hy'], R['c22_dd_rp']))\n"
            "print('   over that year the fund lost %%.1f%%%%; the blend lost %%.1f%%%%'\n"
            "      %% (R['c22_ret_hy'], R['c22_ret_rp']))"
            % (R["c08_dd_hy"], R["c08_dd_rp"], R["c20_dd_hy"], R["c20_dd_rp"],
               R["c20_ret_hy"], R["c20_ret_rp"], R["c22_dd_hy"], R["c22_dd_rp"],
               R["c22_ret_hy"], R["c22_ret_rp"])
        ),
        md("2022 was not a credit event — it was a **pure interest-rate** event, and the "
           "blend carries far more interest-rate risk than the fund does (high-yield bonds "
           "are shorter and pay more coupon). Swapping the fund for the blend is not a free "
           "lunch: it is **trading credit risk for duration risk**. Over these eighteen "
           "years that trade paid; in the one year rates were the story, it cost 4.7 pp."),

        md("## 5. Is the harness honest? (a live, offline synthetic check)\n\n"
           "Everything above is a *frozen* real-tape number. The two cells below are "
           "**synthetic** — a made-up world where we control the answer — run live to prove "
           "the machinery finds an effect when one is planted and stays quiet when it is "
           "not.\n\n"
           "The synthetic fund is genuinely 45% equity + 55% duration *plus* a credit shock "
           "of fixed size. In world A the shock is paid nothing; in world B — the null — it "
           "is paid exactly enough to keep the fund level with its blend."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from hy_replication import data, strategy as st\n"
            "\n"
            "for label, ss in [('A  shock paid nothing ', 1.0), ('B  shock fairly paid  ', 0.0)]:\n"
            "    prices, truth = data.synthetic_panel(signal_strength=ss, seed=954)\n"
            "    d = st.synthetic_detect(prices)\n"
            "    print('%s: recipe found %.3f (planted %.2f) | blend ahead by %+.3f'\n"
            "          % (label, d['w_mean'], truth['w_true'], -d['excess_sharpe_gap']))\n"
            "print()\n"
            "print('SYNTHETIC, not the real tape: the recipe is recovered either way, but the')\n"
            "print('blend only WINS when the extra risk is genuinely unpaid. World B is one')\n"
            "print('draw of a null whose spread is 0.13, so its small lead is noise: across')\n"
            "print('8 seeds world B averages -0.05 and never fires, world A averages -0.44.')"
        ),

        md(f"## Verdict\n\n"
           f"- **Signal — Mixed.** Half the folklore is simply wrong: high yield is *not* a "
           f"repackaged equity position — a held-out 45/55 blend reproduces under half of it "
           f"(R² {R['r2_d']:.2f}) and leaves ~7 pp/yr of genuinely different risk. The other "
           f"half — that you are not paid for that difference — is true in **every** cut we "
           f"took, but only just: *t* = {R['t_gap']:+.2f}, bootstrap CI "
           f"[{R['ci_gap_lo']:+.2f}, {R['ci_gap_hi']:+.2f}], with JNK, USHY and the first era "
           f"all short of the bar.\n"
           f"- **Tradability — Fragile.** The swap is cheap, low-turnover and gained "
           f"{R['rp_cagr_lived'] - R['hy_cagr_lived']:+.1f} pp/yr of lived CAGR with a "
           f"{abs(R['hy_dd']) - abs(R['rp_dd']):.1f} pp shallower worst loss — but it is a "
           f"substitution, not an edge, and it hands you a different risk. In the one year "
           f"that risk showed up (2022) the blend lost "
           f"{abs(R['c22_ret_rp']) - abs(R['c22_ret_hy']):.1f} pp more than the fund it "
           f"replaced."),
    ]
    nb = new_notebook()
    nb["cells"] = cells
    return nb


# --------------------------------------------------------------------------- #
# 02 — for the quants
# --------------------------------------------------------------------------- #
def build_quants():
    cells = [
        md("# Study 954 — High Yield in Disguise — the teardown\n\n"
           "The constrained held-out replication, R² by return horizon, the excess-of-cash "
           "Sharpe race, the vol-matched Newey-West *t*, block-bootstrap CIs, the era cut, "
           "the crisis table, the cost and estimation-window sweeps, the two cross-check "
           "funds, and a live **synthetic** control.\n\n"
           "**Construction.** For each day *t*, `w` is the OLS slope of `r_HY − r_IEF` on "
           "`r_SPY − r_IEF` over the trailing 252 trading days — the constraint that weights "
           "sum to one is imposed by subtracting the duration leg from both sides, so `w` is "
           "literally the equity share of a fully funded blend. `w` is frozen at each "
           "**calendar month-end** and applied to the **following** month: that freeze is the "
           "study's **single execution lag** (the weight in force on day *t* was fitted on "
           "returns ending at least one trading day earlier, and is never re-fitted "
           "intra-month). No second lag is stacked on it.\n\n"
           "**Frictions.** The replication pays 2 bps one-way × NAV on rebalance turnover "
           "(a PROXY, swept) and 50 bps/yr borrow on any short leg (a PROXY that is inert — "
           "the fitted `w` never left [0, 1]). The fund pays **nothing**: it is bought once "
           "and held, so the race is deliberately generous to it. Both arms are excess-of-cash "
           "(BIL total return), both tapes are **total return** (`auto_adjust=True`).\n\n"
           "All real numbers are frozen from `docs/results.md` "
           "(Fingerprint `%s`, as-of 2026-06-30)." % R["fp"]),

        code("R = %r\nprint('frozen real-tape headline loaded: %%d fields, fingerprint %%s' %% (len(R), R['fp']))"
             % (R,)),

        md("## 1. The held-out weight path\n\n"
           "> 💡 **In plain words:** we are asking what mixture of stocks and government "
           "bonds behaved most like a junk-bond fund, and we only ever use yesterday's "
           "answer to judge today."),
        code(
            "print(f\"held-out record : {R['start']} -> {R['end']}  n={R['n_days']:,}\")\n"
            "print(f\"data stamp      : {R['stamp_start']} -> {R['stamp_end']}  \"\n"
            "      f\"n={R['stamp_n']:,}  fp={R['fp']}\")\n"
            "print(f\"equity share w  : mean {R['w_mean']:.3f}  range [{R['w_min']:.3f}, {R['w_max']:.3f}]\")\n"
            "print(f\"max short notional {R['short_max']:.3f} -> the borrow PROXY is identically inert\")\n"
            "print(f\"turnover {R['turnover']:.2f} x NAV / yr -> the cost PROXY has almost no room to bite\")"
        ),

        md("## 2. Replication quality by horizon — the 'costume' test\n\n"
           "A fund whose bonds were marked stale would look badly replicated daily and well "
           "replicated quarterly. The R² is flat in the horizon, so the unexplained part is "
           "economic, not an artefact of marking."),
        code(
            "for h, r2, te in [('daily', R['r2_d'], R['te_d']), ('weekly', R['r2_w'], R['te_w']),\n"
            "                  ('monthly', R['r2_m'], R['te_m']), ('quarterly', R['r2_q'], R['te_q'])]:\n"
            "    print(f\"{h:>10s}: R^2 {r2:.3f}   tracking error {te:5.2f}%/yr\")\n"
            "print()\n"
            "print(f\"residual (HY - replication): {R['resid_ann']:+.2f}%/yr   HAC t = {R['t_resid']:+.2f}\")\n"
            "print('-> the replication FAILS as a replication: ~half the variance is credit-specific.')"
        ),

        md("## 3. The excess-of-cash race and the vol-matched HAC *t*\n\n"
           "Both arms minus BIL's total return; drawdowns are absolute (lived). The gap is "
           "tested as the Newey-West *t* on the daily difference of the two series after each "
           "is scaled to unit realised volatility — the Jobson-Korkie Sharpe comparison in "
           "HAC form.\n\n"
           "> 💡 **In plain words:** dial both portfolios to the same riskiness, then ask "
           "which one ended up with more money, and whether the difference is bigger than "
           "chance."),
        code(
            "print(f\"HYG          : exSharpe {R['hy_sharpe']:+.3f}  exCAGR {R['hy_cagr']:+.2f}%  \"\n"
            "      f\"vol {R['hy_vol']:.2f}%  MaxDD {R['hy_dd']:.2f}%  HAC t {R['hy_t']:+.2f}\")\n"
            "print(f\"replication  : exSharpe {R['rp_sharpe']:+.3f}  exCAGR {R['rp_cagr']:+.2f}%  \"\n"
            "      f\"vol {R['rp_vol']:.2f}%  MaxDD {R['rp_dd']:.2f}%  HAC t {R['rp_t']:+.2f}\")\n"
            "print(f\"lived (absolute) CAGR: HYG {R['hy_cagr_lived']:+.2f}%  \"\n"
            "      f\"replication {R['rp_cagr_lived']:+.2f}%  cash {R['cash_cagr_lived']:+.2f}%\"\n"
            "      f\"  <- exCAGR is net of cash; the drawdowns above are absolute\")\n"
            "print()\n"
            "print(f\"excess-Sharpe gap (HY - repl): {R['gap']:+.3f}   vol-matched HAC t = {R['t_gap']:+.2f}\")\n"
            "print('   (the vol match uses full-sample realised vols: an EX-POST test')\n"
            "print('    statistic, not a path anyone could have levered to in advance)')\n"
            "print(f\"-> about {R['gap_pp_per_yr']:.1f} pp/yr of excess return forgone at matched vol\")\n"
            "print(f\"-> of which ~{R['er_diff']:.2f} pp is the fee gap \"\n"
            "      f\"(HYG {R['er_hyg']:.2f}% vs blend {R['er_blend']:.3f}%, a PROXY decomposition)\")"
        ),

        md("## 4. Block-bootstrap CIs (2,000 draws, 21-day blocks)"),
        code(
            "print(f\"HYG exSharpe        : {R['hy_sharpe']:+.3f}  95% CI \"\n"
            "      f\"[{R['ci_hy_lo']:+.3f}, {R['ci_hy_hi']:+.3f}]  share<0 {R['ci_hy_neg']:.1f}%\")\n"
            "print(f\"replication exSharpe: {R['rp_sharpe']:+.3f}  95% CI \"\n"
            "      f\"[{R['ci_rp_lo']:+.3f}, {R['ci_rp_hi']:+.3f}]  share<0 {R['ci_rp_neg']:.1f}%\")\n"
            "print(f\"vol-matched gap     : {R['ci_gap_pt']:+.3f}  95% CI \"\n"
            "      f\"[{R['ci_gap_lo']:+.3f}, {R['ci_gap_hi']:+.3f}]  share<0 {R['ci_gap_neg']:.1f}%\")\n"
            "print('-> the gap CI excludes zero, but the upper end sits at -0.04. Marginal.')"
        ),

        md("## 5. Era cut (split 2017-01-01)\n\n"
           "Re-uses the already held-out series, so both halves inherit exactly the "
           "out-of-sample weights the full run used."),
        code(
            "print(f\"2008-2016 (n={R['era_e_n']:,}): HY {R['era_e_hy']:+.3f} / repl {R['era_e_rp']:+.3f}  \"\n"
            "      f\"gap {R['era_e_gap']:+.3f} (t={R['era_e_t']:+.2f})  \"\n"
            "      f\"residual {R['era_e_resid']:+.2f}%/yr (t={R['era_e_tres']:+.2f})\")\n"
            "print(f\"2017-2026 (n={R['era_l_n']:,}): HY {R['era_l_hy']:+.3f} / repl {R['era_l_rp']:+.3f}  \"\n"
            "      f\"gap {R['era_l_gap']:+.3f} (t={R['era_l_t']:+.2f})  \"\n"
            "      f\"residual {R['era_l_resid']:+.2f}%/yr (t={R['era_l_tres']:+.2f})\")\n"
            "print('-> same sign in both halves, wider in the recent one; only the recent one clears |t|=2.')"
        ),

        md("## 6. The crisis table — where the two arms actually diverge\n\n"
           "> 💡 **In plain words:** the blend beats the fund when the crisis is about "
           "companies defaulting, and loses when the crisis is about interest rates."),
        code(
            "rows = [('2008 GFC', R['c08_dd_hy'], R['c08_dd_rp'], R['c08_ret_hy'], R['c08_ret_rp']),\n"
            "        ('2020 Covid', R['c20_dd_hy'], R['c20_dd_rp'], R['c20_ret_hy'], R['c20_ret_rp']),\n"
            "        ('2022 rates', R['c22_dd_hy'], R['c22_dd_rp'], R['c22_ret_hy'], R['c22_ret_rp'])]\n"
            "print(f\"{'episode':<12s}{'DD hy':>9s}{'DD repl':>10s}{'ret hy':>9s}{'ret repl':>10s}   winner\")\n"
            "for tag, ddh, ddr, rh, rr in rows:\n"
            "    win = 'replication' if ddr > ddh else 'the fund'\n"
            "    print(f\"{tag:<12s}{ddh:>8.1f}%{ddr:>9.1f}%{rh:>8.1f}%{rr:>9.1f}%   {win}\")\n"
            "print()\n"
            "print('2022 is a pure duration event: the blend holds far more interest-rate risk.')\n"
            "print('The swap is credit risk -> duration risk, not risk -> no risk.')"
        ),

        md("## 7. Sensitivity — cost PROXY and estimation window\n\n"
           "Turnover is 0.34 ×NAV/yr, so friction cannot explain the result; the estimation "
           "window is the one design choice with real bite, and it trades weight stability "
           "against how much of 2008 survives into the out-of-sample record."),
        code(
            "print('cost sweep (one-way bps on the replication):')\n"
            "print(f\"   0 bps: gap {R['cost0_gap']:+.3f} (t={R['cost0_t']:+.2f})\")\n"
            "print(f\"   2 bps: gap {R['win252_gap']:+.3f} (t={R['win252_t']:+.2f})   <- headline\")\n"
            "print(f\"  25 bps: gap {R['cost25_gap']:+.3f} (t={R['cost25_t']:+.2f})   <- 12x the headline, unchanged\")\n"
            "print()\n"
            "print('estimation-window sweep:')\n"
            "for w, g, t, s in [(126, R['win126_gap'], R['win126_t'], '2007-12'),\n"
            "                   (252, R['win252_gap'], R['win252_t'], '2008-06'),\n"
            "                   (504, R['win504_gap'], R['win504_t'], '2009-06'),\n"
            "                   (756, R['win756_gap'], R['win756_t'], '2010-06')]:\n"
            "    print(f\"  {w:>3d} d (OOS from {s}): gap {g:+.3f} (t={t:+.2f})\")\n"
            "print('-> sign unanimous, magnitude monotone in the window, t straddles the bar.')"
        ),

        md("## 7b. Sensitivity — which Treasury is *the* duration leg?\n\n"
           "IEF (7-10y) is a **design choice**, not a fact of the tape: a high-yield fund's "
           "own duration is nearer 3-4 years, so SHY (1-3y) and IEI (3-7y) are at least as "
           "defensible and TLT (20y+) is the aggressive end. The weight re-fits itself to "
           "whatever leg it is handed, so the fit always works — the question is whether "
           "the *conclusion* does. Treasuries only: AGG or LQD on the bench would smuggle "
           "the credit risk under test into the benchmark. Same common 4,548-day sample."),
        code(
            "print(f\"{'leg':<12s}{'w':>7s}{'residual':>11s}{'gap':>9s}{'t':>8s}\")\n"
            "for tag, w, res, g, t in [\n"
            "        ('SHY  1-3y', R['leg_shy_w'], R['leg_shy_resid'], R['leg_shy_gap'], R['leg_shy_t']),\n"
            "        ('IEI  3-7y', R['leg_iei_w'], R['leg_iei_resid'], R['leg_iei_gap'], R['leg_iei_t']),\n"
            "        ('IEF 7-10y', R['leg_ief_w'], R['leg_ief_resid'], R['leg_ief_gap'], R['leg_ief_t']),\n"
            "        ('TLT  20y+', R['leg_tlt_w'], R['leg_tlt_resid'], R['leg_tlt_gap'], R['leg_tlt_t'])]:\n"
            "    flag = '   <- headline' if tag.startswith('IEF') else ''\n"
            "    print(f\"{tag:<12s}{w:>7.3f}{res:>10.2f}%{g:>9.3f}{t:>8.2f}{flag}\")\n"
            "print()\n"
            "print('-> the SIGN is leg-proof: every maturity hands the blend the higher Sharpe.')\n"
            "print('-> the t is NOT: 2.07 is the top of a 1.92-2.07 range, so the bar is')\n"
            "print('   cleared in the middle and missed at both ends of the curve.')\n"
            "print('-> the residual is the least robust number in the study: -0.62%/yr against')\n"
            "print('   SHY, -4.19%/yr against TLT. Most of the headline -2.16%/yr is the price')\n"
            "print('   of the maturity mismatch, not a measurement of unpaid credit risk.')"
        ),

        md("## 8. Cross-checks — the other two high-yield funds"),
        code(
            "print(f\"JNK  (n={R['jnk_n']:,}): w {R['jnk_w']:.3f}  R^2 {R['jnk_r2']:.3f}  \"\n"
            "      f\"gap {R['jnk_gap']:+.3f} (t={R['jnk_t']:+.2f})\")\n"
            "print(f\"USHY (n={R['ushy_n']:,}): w {R['ushy_w']:.3f}  R^2 {R['ushy_r2']:.3f}  \"\n"
            "      f\"gap {R['ushy_gap']:+.3f} (t={R['ushy_t']:+.2f})\")\n"
            "print('-> same sign and size, but neither clears |t|=2 on its own shorter sample.')\n"
            "print('Survivorship: these are the funds still listed; dead HY ETFs are absent,')\n"
            "print('which flatters the fund side - so the negative finding is conservative.')"
        ),

        md("## 9. Live synthetic control — the machinery is unbiased\n\n"
           "**This section is synthetic, not the real tape.** The generator builds a fund "
           "that *is* `w_true × equity + (1 − w_true) × duration` plus an idiosyncratic "
           "credit shock of fixed size; `signal_strength` changes only what that shock is "
           "*paid*. At `1` it carries a 3%/yr give-up (the planted effect); at `0` it earns "
           "exactly the premium that keeps the fund's Sharpe level with the blend's (the "
           "null). The weight must be recovered in **both** worlds."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from hy_replication import data, strategy as st\n"
            "\n"
            "for ss, tag in [(1.0, 'uncompensated (planted)'), (0.0, 'fairly paid (null)  ')]:\n"
            "    prices, truth = data.synthetic_panel(signal_strength=ss, seed=954)\n"
            "    d = st.synthetic_detect(prices)\n"
            "    print(f\"{tag}: w_hat {d['w_mean']:.3f} (true {truth['w_true']:.2f})  \"\n"
            "          f\"R^2 {d['r2']:.3f}  residual {d['residual_ann']*100:+.2f}%/yr \"\n"
            "          f\"(t={d['t_residual']:+.2f})  gap {d['excess_sharpe_gap']:+.3f} \"\n"
            "          f\"(t={d['t_gap']:+.2f})\")"
        ),
        code(
            "for ss, tag in [(1.0, 'planted'), (0.0, 'null   ')]:\n"
            "    gaps = np.array([\n"
            "        st.synthetic_detect(data.synthetic_panel(signal_strength=ss, seed=954 + s)[0])['excess_sharpe_gap']\n"
            "        for s in range(8)\n"
            "    ])\n"
            "    print(f\"{tag} x8 seeds: gap mean {gaps.mean():+.3f} (sd {gaps.std(ddof=1):.3f}), \"\n"
            "          f\"|gap| >= 0.35 on {(abs(gaps) >= 0.35).sum()}/8\")\n"
            "print()\n"
            "print('SYNTHETIC ONLY. The detector fires on the planted give-up and is centred on')\n"
            "print('zero when the same extra risk is fairly paid -> the real-tape gap is a fact')\n"
            "print('about the high-yield tape, not a biased harness.')"
        ),

        md(f"## Verdict\n\n"
           f"- **Signal — Mixed.** Two claims, two answers. *Replication* is **refuted**: "
           f"R² = {R['r2_d']:.3f} daily and {R['r2_q']:.3f} quarterly with ~7 pp/yr of "
           f"tracking error, so high yield carries a real, distinct credit exposure and is "
           f"not a repackaged equity position. *Compensation* leans one way everywhere — "
           f"the gap is negative in all four estimation windows "
           f"({R['win126_gap']:+.3f} to {R['win756_gap']:+.3f}), both eras "
           f"({R['era_e_gap']:+.3f} / {R['era_l_gap']:+.3f}) and all three funds — but the "
           f"headline HAC *t* is only {R['t_gap']:+.2f}, the bootstrap CI "
           f"[{R['ci_gap_lo']:+.3f}, {R['ci_gap_hi']:+.3f}] clears zero by 0.04, the "
           f"standalone residual *t* is {R['t_resid']:+.2f}, JNK ({R['jnk_t']:+.2f}), "
           f"USHY ({R['ushy_t']:+.2f}) and the early era ({R['era_e_t']:+.2f}) all miss the "
           f"bar, and swapping the duration leg for SHY ({R['leg_shy_t']:+.2f}) or TLT "
           f"({R['leg_tlt_t']:+.2f}) drops the headline back under it. Unanimous direction, "
           f"marginal size — Mixed, not Real.\n"
           f"- **Tradability — Fragile.** The substitution is cheap and mechanical "
           f"({R['turnover']:.2f} ×NAV/yr, unchanged at 25 bps, `w` inside [0, 1] so no "
           f"leverage or borrow) and it delivered {R['rp_cagr_lived'] - R['hy_cagr_lived']:+.1f} pp/yr of lived CAGR "
           f"with a {abs(R['hy_dd']) - abs(R['rp_dd']):.1f} pp shallower worst loss. But it "
           f"is a *substitution*, not an alpha — you keep the same two risk premia and drop "
           f"a fund fee worth {R['er_diff']:.2f} pp — and it is regime-conditional: 2022 "
           f"cost the blend {abs(R['c22_ret_rp']) - abs(R['c22_ret_hy']):.1f} pp more than "
           f"the fund, because the trade is credit risk for duration risk. Size it as an "
           f"exposure decision, not as an edge."),
    ]
    nb = new_notebook()
    nb["cells"] = cells
    return nb


def main() -> None:
    for name, builder in [("01_for_the_curious", build_curious),
                          ("02_for_the_quants", build_quants)]:
        nb = builder()
        path = os.path.join(HERE, f"{name}.ipynb")
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print(f"wrote {path} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
