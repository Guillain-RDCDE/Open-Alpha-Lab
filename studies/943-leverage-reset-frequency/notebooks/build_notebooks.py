"""Generate the two narrative notebooks for Study 943 (Reset Frequency).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from
the frozen ``R`` dict below (mirroring docs/results.md); the only live cells run the fast
synthetic control, so execution is quick and network-free. No synthetic figure ever sits
under a real-tape banner.
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


# Frozen real-tape headline — mirror of docs/results.md. SPY on margin at ^IRX + 50 bps,
# monthly vs daily leverage reset, raced against SSO (2x) and UPRO (3x), excess-of-cash.
R = dict(
    asof="2026-06-30",
    spread_bps=50, cost_bps=2, maintenance=25,
    # --- 2x sleeve --------------------------------------------------------- #
    x2_start="2007-05-31", x2_end="2026-06-30", x2_n=4799, x2_fp="8f71ea1922ec",
    x2_fund_sh=0.504, x2_fund_cagr=14.33, x2_fund_vol=38.9, x2_fund_dd=-84.7, x2_fund_term=12.8,
    x2_day_sh=0.527, x2_day_cagr=15.45, x2_day_vol=39.6, x2_day_dd=-84.2, x2_day_term=15.4,
    x2_mon_sh=0.549, x2_mon_cagr=17.13, x2_mon_vol=42.8, x2_mon_dd=-82.5, x2_mon_term=20.3,
    x2_spy_sh=0.613, x2_spy_cagr=10.70, x2_spy_vol=19.8, x2_spy_dd=-55.2, x2_spy_term=6.9,
    x2_diff_bps=1.052, x2_t=2.79, x2_adv=0.022,
    x2_adv_ci_lo=-0.010, x2_adv_ci_hi=0.062, x2_adv_frac_neg=8.7,
    x2_mon_ci_lo=0.168, x2_mon_ci_hi=0.951, x2_day_ci_lo=0.122, x2_day_ci_hi=0.941,
    x2_vs_fund=0.045, x2_t_vs_fund=3.41, x2_fee_leg=1.21, x2_t_fee=3.82,
    x2_w_mean=1.998, x2_w_min=1.78, x2_w_max=3.24,
    x2_months=229, x2_slope=-1.250, x2_slope_t=-6.83,
    x2_chop=0.381, x2_chop_t=4.59, x2_trend=-0.164, x2_trend_t=-8.40,
    x2_pred_slope=0.115, x2_pred_t=0.72, x2_switch=0.060, x2_switch_t=2.49,
    x2_era_e_adv=0.047, x2_era_e_t=2.26, x2_era_l_adv=0.004, x2_era_l_t=1.56,
    # The era contrast is NOT significant - the house bar wants the difference tested.
    x2_era_diff_bps=-0.836, x2_era_diff_t=-1.15,
    x3_era_diff_bps=-2.369, x3_era_diff_t=-0.30,
    x2_sp0_adv=0.021, x2_sp200_adv=0.025, x2_c0_adv=0.021, x2_c10_adv=0.029,
    # --- 3x sleeve --------------------------------------------------------- #
    x3_start="2009-06-26", x3_end="2026-06-30", x3_n=4276, x3_fp="4a95749c18bb",
    x3_fund_sh=0.791, x3_fund_cagr=32.95, x3_fund_vol=51.3, x3_fund_dd=-76.8, x3_fund_term=125.5,
    x3_day_sh=0.808, x3_day_cagr=34.17, x3_day_vol=51.3, x3_day_dd=-76.2, x3_day_term=146.6,
    x3_mon_sh=0.239, x3_mon_cagr=4.05, x3_mon_vol=18.8, x3_mon_dd=-48.3, x3_mon_term=2.0,
    x3_spy_sh=0.911, x3_spy_cagr=15.17, x3_spy_term=11.0,
    x3_diff_bps=-14.678, x3_t=-3.62, x3_adv=-0.569,
    x3_adv_ci_lo=-1.085, x3_adv_ci_hi=0.017,
    x3_fee_leg=0.94, x3_t_fee=3.71,
    x3_w_mean=2.944, x3_w_mean_free=2.983, x3_vol_free=58.5,
    x3_dd_free=-82.2, x3_m30_adv=0.124, x3_w_max_free=8.42, x3_w_max_25=3.84, x3_liq="2011-08-08",
    x3_months=204, x3_slope=-3.158, x3_slope_t=-7.22,
    x3_chop=0.950, x3_chop_t=3.91, x3_trend=-0.492, x3_trend_t=-8.29,
    x3_pred_slope=0.659, x3_pred_t=1.61, x3_switch=0.041, x3_switch_t=1.04,
    x3_era_e_adv=-0.581, x3_era_e_t=-2.80, x3_era_l_adv=-0.563, x3_era_l_t=-2.48,
    x3_m0_adv=0.003, x3_m0_term=227.3, x3_m15_adv=-0.285, x3_m15_liq="2020-03-20",
    x3_m30_liq_daily="2010-05-06",
    # --- 2008 stress on SPY ------------------------------------------------ #
    st_start="2004-01-05", st_end="2026-06-30", st_n=5652,
    st_2x_mon=37.56, st_2x_day=27.91,
    st_3x_mon_free=80.43, st_3x_mon_free_w=12.80, st_3x_mon_25=0.96, st_3x_liq="2008-09-29",
    st_3x_day=33.75,
    st_4x_mon_free=0.00, st_4x_liq="2008-10-24", st_4x_day=17.37,
    # --- synthetic control ------------------------------------------------- #
    syn_chop_gap=0.357, syn_chop_t=3.74, syn_chop_slope=-1.99,
    syn_trend_gap=-0.403, syn_trend_t=-2.47, syn_trend_slope=-2.65,
    syn_null_gap=0.033, syn_null_t=0.27, syn_null_slope=-2.29,
)


HEADER = f"""# Study 943 — Reset Frequency ⚖️

**Leveraged ETFs reset their leverage every single evening. Would resetting monthly
instead have been better?**

Every forum thread about TQQQ and UPRO blames the **daily reset** for "volatility decay",
and prescribes the same fix: reset monthly and keep the leverage without the drag. That is
a mechanical claim, so we build the alternative for real — **SPY on margin**, financed at
**^IRX + {R['spread_bps']} bps**, levered back to 2x / 3x **once a month** and left to
drift in between — and race it, **excess-of-cash**, against a daily-reset replication and
against **SSO** and **UPRO** themselves.

Windows: 2x on {R['x2_start']} → {R['x2_end']} ({R['x2_n']:,} days, fingerprint
`{R['x2_fp']}`); 3x on {R['x3_start']} → {R['x3_end']} ({R['x3_n']:,} days, fingerprint
`{R['x3_fp']}`). Total-return closes (`auto_adjust=True`); `^IRX` is a *yield*, not a
price. One execution lag: the reset is decided at the month-end close and is in force the
next session.

*Numbers below are the frozen headline (`docs/results.md`); the live cells run only the
offline synthetic control. As-of {R['asof']}.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),

        md("## 1. What a reset actually is\n\n"
           "A 2x fund is a margin account: for every £1 of yours it holds £2 of the index "
           "and owes £1 of cash. If the index falls 10%, your £1 of equity becomes £0.80 "
           "while you still hold £1.80 of index — your *effective* leverage has quietly "
           "risen from 2.0 to 2.25. Resetting means trading back to exactly 2.0.\n\n"
           "**Daily reset** (what SSO and UPRO do): every evening, without fail. "
           "**Monthly reset** (the folklore fix): once a month, and whatever the market "
           "does in between, you live with it."),

        md("## 2. The 2x race — the folklore is *almost* right, and it doesn't matter"),
        code(
            "R = " + repr(R) + "\n"
            "rows = [('SSO (the real 2x fund)', 'x2_fund'), ('daily-reset replication', 'x2_day'),\n"
            "        ('MONTHLY-reset replication', 'x2_mon'), ('SPY (1x, reference)', 'x2_spy')]\n"
            "print(f\"{'arm':<26s} {'exSharpe':>9s} {'CAGR':>8s} {'vol':>7s} {'worst DD':>9s} {'x money':>9s}\")\n"
            "for label, k in rows:\n"
            "    print(f\"{label:<26s} {R[k+'_sh']:+9.3f} {R[k+'_cagr']:+7.2f}% {R[k+'_vol']:6.1f}% \"\n"
            "          f\"{R[k+'_dd']:8.1f}% {R[k+'_term']:8.1f}x\")\n"
            "print()\n"
            "print(f\"monthly minus daily: {R['x2_diff_bps']:+.2f} bps/day (t = {R['x2_t']:+.2f}) \"\n"
            "      f\"-> a REAL extra return\")\n"
            "print(f\"but risk-adjusted   : {R['x2_adv']:+.3f} of Sharpe, 95% CI \"\n"
            "      f\"[{R['x2_adv_ci_lo']:+.3f}, {R['x2_adv_ci_hi']:+.3f}] -> indistinguishable from zero\")"
        ),
        md(f"The monthly reset turned £1 into **£{R['x2_mon_term']:.1f}** where the daily "
           f"reset made **£{R['x2_day_term']:.1f}** and SSO itself made "
           f"**£{R['x2_fund_term']:.1f}**. That looks like a win — until you notice the "
           f"volatility went from {R['x2_day_vol']:.1f}% to {R['x2_mon_vol']:.1f}% at the "
           f"same time. Per unit of risk taken, the gain is **{R['x2_adv']:+.3f}** of "
           f"Sharpe, and the confidence interval straddles zero.\n\n"
           f"> 🔬 **For the quants** — the extra return is not free alpha, and it is not "
           f"extra leverage either: mean exposure is {R['x2_w_mean']:.2f}, the same "
           f"2.00 the daily arm runs. It is extra *risk*. Between resets the monthly "
           f"account's leverage floats — minimum {R['x2_w_min']:.2f}, **maximum "
           f"{R['x2_w_max']:.2f}** (October 2008) — and that float is the whole of the "
           f"extra volatility."),

        md("## 3. Where the difference *comes from* — and why the folklore is upside down"),
        code(
            "print('monthly-minus-daily gap, by the shape of the month (percentage points/month)')\n"
            "print(f\"  2x: choppy months {R['x2_chop']:+.3f} (t {R['x2_chop_t']:+.2f})  |  \"\n"
            "      f\"trending months {R['x2_trend']:+.3f} (t {R['x2_trend_t']:+.2f})\")\n"
            "print(f\"  3x: choppy months {R['x3_chop']:+.3f} (t {R['x3_chop_t']:+.2f})  |  \"\n"
            "      f\"trending months {R['x3_trend']:+.3f} (t {R['x3_trend_t']:+.2f})\")"
        ),
        md("In a **choppy** month the daily reset really does rebalance into every "
           "reversal, and the monthly reset wins. In a **trending** month the daily reset "
           "*compounds the trend* — it adds exposure as you make money — and the monthly "
           "reset loses. The daily reset is not the villain of the story; it is the hero "
           "of every straight-line month, and the villain only of the zig-zag ones.\n\n"
           "> 🔬 **For the quants** — this split is *arithmetic*, not a signal. Section 6 "
           "shows the same relationship appears on a pure random walk. And it is not "
           f"forecastable: use last month's shape to choose this month's reset and the "
           f"relationship collapses to *t* = {R['x2_pred_t']:+.2f}."),

        md("## 4. The 3x sleeve — where the story ends abruptly"),
        code(
            "print(f\"3x, {R['x3_start']} -> {R['x3_end']}\")\n"
            "print(f\"  UPRO (the real fund)          : x{R['x3_fund_term']:.1f} money, \"\n"
            "      f\"excess Sharpe {R['x3_fund_sh']:+.3f}\")\n"
            "print(f\"  daily-reset replication       : x{R['x3_day_term']:.1f} money, \"\n"
            "      f\"excess Sharpe {R['x3_day_sh']:+.3f}\")\n"
            "print(f\"  MONTHLY-reset margin account  : x{R['x3_mon_term']:.1f} money, \"\n"
            "      f\"excess Sharpe {R['x3_mon_sh']:+.3f}\")\n"
            "print(f\"  ... margin-called on {R['x3_liq']}; the proceeds then sit in cash,\")\n"
            "print(\"      earning the cash leg and exactly zero excess return, "
            "for fifteen years.\")"
        ),
        md(f"Left alone, the 3x monthly account's leverage drifted to **"
           f"{R['x3_w_max_free']:.2f}x** in March 2020. A broker does not leave you alone: "
           f"at a standard 25% maintenance requirement the position was liquidated on "
           f"**{R['x3_liq']}**, in the post-downgrade selloff, and never came back. "
           f"(We credit the liquidated account the cash rate afterwards — a margin "
           f"call should cost you the equity it destroyed, not an invented drag for "
           f"the next fifteen years.)\n\n"
           f"And on the longer SPY tape — which reaches the 2008 crash that UPRO, launched "
           f"in 2009, never saw — a 3x monthly account is called on **{R['st_3x_liq']}**, "
           f"and a **4x monthly account went to negative equity** on {R['st_4x_liq']}. A "
           f"4x *daily*-reset account, on the same tape, could not: it ended at "
           f"×{R['st_4x_day']:.1f}. **That is what the daily reset buys you.**"),

        md("## 5. So what *is* the monthly reset's edge over the funds?"),
        code(
            "print(f\"monthly reset vs SSO       : {R['x2_vs_fund']:+.3f} Sharpe (t {R['x2_t_vs_fund']:+.2f})\")\n"
            "print(f\"  of which reset frequency  : {R['x2_adv']:+.3f}\")\n"
            "print(f\"  of which the fund's fee + tracking drag: {R['x2_fee_leg']:+.2f}%/yr \"\n"
            "      f\"(t {R['x2_t_fee']:+.2f}) -- nothing to do with resetting\")"
        ),
        md("About half of it is simply the ~0.9% expense ratio and tracking slippage you "
           "avoid by doing it yourself — a real saving, but a *fee* story, not a *reset* "
           "story. And it is only yours if you can borrow near the T-bill rate: widen the "
           "financing spread to 200 bps and that leg shrinks from +0.056 to +0.010."),

        md("## 6. Live check — the machinery is unbiased (offline synthetic)\n\n"
           "We plant a world that is deliberately **choppy** (yesterday's move partly "
           "reverses), one that is deliberately **trending**, and one that is a pure "
           "**random walk** — all with the same volatility. The monthly reset must win in "
           "the first, lose in the second, and do nothing at all in the third."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from reset_freq import data, strategy as st\n"
            "for tag, phi, ss in [('choppy  ', -0.15, 1.0), ('trending', 0.15, 1.0), ('random walk', -0.15, 0.0)]:\n"
            "    g = [st.synthetic_detect(data.synthetic_daily(phi=phi, signal_strength=ss,\n"
            "                                                  n_years=12, seed=943+s)[0])\n"
            "         for s in range(3)]\n"
            "    print(f\"{tag:<12s}: monthly-minus-daily gap {np.mean([d['mean_gap_bps'] for d in g]):+.3f} bps/day\"\n"
            "          f\"   (path-shape slope {np.mean([d['chop_slope'] for d in g]):+.2f})\")"
        ),
        md("Right sign in both planted worlds, **zero on the random walk** — the detector "
           "is honest. Note the last column: the path-shape slope is strongly negative "
           "*even on the random walk*. That is the proof that the trending-versus-choppy "
           "split in section 3 is arithmetic about the path, and never evidence of an edge."),

        md(f"## Verdict\n\n"
           f"- **Signal — Mixed.** *Real on the return, absent on the Sharpe; positive "
           f"at 2x, negative at 3x.* Reset frequency genuinely moves returns (2x: "
           f"{R['x2_diff_bps']:+.2f} bps/day, HAC *t* = {R['x2_t']:+.2f}) and its "
           f"direction is real and well identified — but the *claim* that monthly is "
           f"**better** fails: {R['x2_adv']:+.3f} of Sharpe with a CI straddling zero, "
           f"unforecastable, and "
           f"{R['x3_adv']:+.3f} at 3x. (The second half of the sample looks weaker, "
           f"{R['x2_era_l_adv']:+.3f} against {R['x2_era_e_adv']:+.3f}, but the era "
           f"difference itself is *t* = {R['x2_era_diff_t']:+.2f} — not a decay we "
           f"can claim.)\n"
           f"- **Tradability — Mirage.** The monthly reset's real product is uncontrolled "
           f"leverage — {R['x3_w_max_free']:.2f}x at the peak, a margin call in 2011, "
           f"negative equity at 4x in 2008 — all of it at the *same* average leverage "
           f"the daily arm ran. The daily reset's supposed vice is the feature "
           f"that keeps you solvent, and it costs about two hundredths of a Sharpe."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 943 — Reset Frequency — the teardown\n\n"
           "The excess-of-cash race at 2x and 3x, the Newey-West difference *t*, paired "
           "block-bootstrap CIs on the Sharpe advantage, the efficiency-ratio "
           "decomposition (and the proof that it is arithmetic), the predictive version, "
           "the era cut, three sweeps — financing spread, cost, maintenance margin — the "
           "2008 stress, and the live synthetic control.\n\n"
           f"Real numbers are frozen from `docs/results.md` (fingerprints "
           f"`{R['x2_fp']}` for the 2x sleeve, `{R['x3_fp']}` for the 3x). Construction: "
           f"exposure `w` on the index funded at `^IRX + {R['spread_bps']} bps`, "
           f"`r_p = w·r − (w−1)·f − cost`, one execution lag on the reset, "
           f"{R['cost_bps']} bps one-way on |Δw|·NAV, {R['maintenance']}% maintenance "
           "margin. Long-only, so there is no stock borrow; the only borrow is cash, and "
           "its spread is swept. A margin-called arm is credited the cash leg for "
           "the rest of the sample, so a call costs it the equity it destroyed and "
           "nothing more."),
        code("R = " + repr(R)),

        md("## 1. The 2x race, excess-of-cash\n\n"
           "> 💡 **In plain words** — three ways to hold twice the S&P: buy the fund, do it "
           "yourself and rebalance nightly, or do it yourself and rebalance monthly."),
        code(
            "for label, k in [('SSO (fund)', 'x2_fund'), ('daily-synth', 'x2_day'),\n"
            "                 ('monthly-synth', 'x2_mon'), ('SPY 1x', 'x2_spy')]:\n"
            "    print(f\"{label:<14s} exSharpe {R[k+'_sh']:+.3f}  CAGR {R[k+'_cagr']:+6.2f}%  \"\n"
            "          f\"vol {R[k+'_vol']:5.1f}%  maxDD {R[k+'_dd']:6.1f}%  terminal x{R[k+'_term']:.1f}\")\n"
            "print(f\"\\nmonthly - daily : {R['x2_diff_bps']:+.3f} bps/day  HAC t {R['x2_t']:+.2f}  \"\n"
            "      f\"Sharpe advantage {R['x2_adv']:+.3f}\")\n"
            "print(f\"paired block-bootstrap CI on the advantage: \"\n"
            "      f\"[{R['x2_adv_ci_lo']:+.3f}, {R['x2_adv_ci_hi']:+.3f}]  \"\n"
            "      f\"({R['x2_adv_frac_neg']:.1f}% of resamples negative)\")\n"
            "print(f\"monthly Sharpe CI [{R['x2_mon_ci_lo']:+.3f}, {R['x2_mon_ci_hi']:+.3f}]  \"\n"
            "      f\"daily Sharpe CI [{R['x2_day_ci_lo']:+.3f}, {R['x2_day_ci_hi']:+.3f}] -- superimposed\")\n"
            "print(f\"exposure path (monthly arm): mean {R['x2_w_mean']:.3f}  min {R['x2_w_min']:.2f}  \"\n"
            "      f\"max {R['x2_w_max']:.2f}\")"
        ),
        md("The mean-return difference clears |*t*| = 2; the **Sharpe** difference does "
           "not. Mean exposure is 2.00 on the nose, so this is not a crude leverage "
           "mismatch at 2x — it is the path convexity, and it is worth two hundredths of a "
           "Sharpe."),

        md("## 2. Decomposing the gap versus the fund\n\n"
           "> 💡 **In plain words** — beating SSO is two separate wins: rebalancing less "
           "often, and not paying SSO's fee. Only the first one is this study's subject."),
        code(
            "print(f\"monthly-synth - SSO        : {R['x2_vs_fund']:+.3f} Sharpe (HAC t {R['x2_t_vs_fund']:+.2f})\")\n"
            "print(f\"  reset-frequency leg      : {R['x2_adv']:+.3f}\")\n"
            "print(f\"  fee/tracking leg (daily-synth - SSO): {R['x2_fee_leg']:+.2f}%/yr \"\n"
            "      f\"(HAC t {R['x2_t_fee']:+.2f})\")\n"
            "print(f\"3x: fee/tracking leg (daily-synth - UPRO): {R['x3_fee_leg']:+.2f}%/yr \"\n"
            "      f\"(HAC t {R['x3_t_fee']:+.2f})\")"
        ),

        md("## 3. Trending versus choppy — the efficiency-ratio decomposition\n\n"
           "Per month, regress the log gap (monthly − daily, pp) on "
           "ER = |Σ log r| / Σ|log r| on SPY. ER = 1 is a straight-line month; ER ≈ 0 is a "
           "round trip.\n\n"
           "> 💡 **In plain words** — did the month go somewhere, or did it thrash about?"),
        code(
            "print(f\"2x ({R['x2_months']} months): slope {R['x2_slope']:+.3f} pp per unit ER  \"\n"
            "      f\"HAC t {R['x2_slope_t']:+.2f}\")\n"
            "print(f\"   choppy tercile {R['x2_chop']:+.3f} pp (t {R['x2_chop_t']:+.2f})   \"\n"
            "      f\"trending tercile {R['x2_trend']:+.3f} pp (t {R['x2_trend_t']:+.2f})\")\n"
            "print(f\"3x ({R['x3_months']} months): slope {R['x3_slope']:+.3f} pp per unit ER  \"\n"
            "      f\"HAC t {R['x3_slope_t']:+.2f}\")\n"
            "print(f\"   choppy tercile {R['x3_chop']:+.3f} pp (t {R['x3_chop_t']:+.2f})   \"\n"
            "      f\"trending tercile {R['x3_trend']:+.3f} pp (t {R['x3_trend_t']:+.2f})\")\n"
            "print()\n"
            "print(f\"LAGGED (tradable) version: 2x slope {R['x2_pred_slope']:+.3f} \"\n"
            "      f\"(t {R['x2_pred_t']:+.2f}); 3x slope {R['x3_pred_slope']:+.3f} (t {R['x3_pred_t']:+.2f})\")\n"
            "print(f\"naive switch rule (monthly after a choppy month, IN-SAMPLE median threshold): \"\n"
            "      f\"2x {R['x2_switch']:+.3f} pp/mo (t {R['x2_switch_t']:+.2f}), \"\n"
            "      f\"3x {R['x3_switch']:+.3f} pp/mo (t {R['x3_switch_t']:+.2f})\")"
        ),
        md(f"The contemporaneous slope is enormous and the sign is unambiguous — and "
           f"**both are mechanical**. Section 7 measures the same slope of "
           f"{R['syn_null_slope']:+.2f} on an *iid random walk*, where by construction "
           f"there is nothing to find. A month that happened to be choppy always favours "
           f"the monthly reset; that is the compounding algebra, not information. What is "
           f"*not* mechanical is the unconditional mean gap, which is zero on the null and "
           f"{R['x2_diff_bps']:+.2f} bps/day on the real 2x tape — a leverage-lens "
           f"restatement of the post-2000 negative daily autocorrelation of the S&P.\n\n"
           f"> 💡 **In plain words** — you cannot pick next month's reset frequency, "
           f"because you cannot know in advance whether next month will trend."),

        md("## 4. The 3x sleeve and the maintenance-margin sweep\n\n"
           "> 💡 **In plain words** — the answer at 3x depends entirely on how patient "
           "your lender is, so here is every level of patience."),
        code(
            "for label, k in [('UPRO (fund)', 'x3_fund'), ('daily-synth', 'x3_day'), ('monthly-synth', 'x3_mon')]:\n"
            "    print(f\"{label:<14s} exSharpe {R[k+'_sh']:+.3f}  CAGR {R[k+'_cagr']:+6.2f}%  \"\n"
            "          f\"terminal x{R[k+'_term']:.1f}\")\n"
            "print(f\"monthly - daily: {R['x3_diff_bps']:+.2f} bps/day  HAC t {R['x3_t']:+.2f}  \"\n"
            "      f\"Sharpe advantage {R['x3_adv']:+.3f}  CI [{R['x3_adv_ci_lo']:+.3f}, {R['x3_adv_ci_hi']:+.3f}]\")\n"
            "print()\n"
            "print('maintenance-margin sweep (PROXY):')\n"
            "print(f\"   0%: advantage {R['x3_m0_adv']:+.3f}  peak exposure {R['x3_w_max_free']:.2f}x  \"\n"
            "      f\"terminal x{R['x3_m0_term']:.1f}  never called\")\n"
            "print(f\"  15%: advantage {R['x3_m15_adv']:+.3f}  called {R['x3_m15_liq']}\")\n"
            "print(f\"  25%: advantage {R['x3_adv']:+.3f}  called {R['x3_liq']}  \"\n"
            "      f\"terminal x{R['x3_mon_term']:.1f}\")\n"
            "print(f\"  30%: even the DAILY-reset margin account is called ({R['x3_m30_liq_daily']}, \"\n"
            "      f\"the flash crash)\")"
        ),
        md(f"Read the 0% row as the fair-fight upper bound: with a lender who never calls, "
           f"the 3x monthly reset's Sharpe advantage is **{R['x3_m0_adv']:+.3f}** — it "
           f"converts a ×{R['x3_day_term']:.1f} into a ×{R['x3_m0_term']:.1f} on mean "
           f"exposure **{R['x3_w_mean_free']:.2f}** — the daily arm's own 3.00 — by "
           f"running {R['x3_vol_free']:.1f}% vol against {R['x3_day_vol']:.1f}% and a "
           f"{R['x3_dd_free']:.1f}% drawdown against {R['x3_day_dd']:.1f}%. Same "
           f"leverage, more risk, no Sharpe. The 30% row is the deeper point: at a "
           f"broker requirement most retail accounts actually face **no** 3x margin "
           f"account survives (the {R['x3_m30_adv']:+.3f} there is two dead arms, not "
           f"a win) — which is why these things are funds."),

        md("## 5. Era cut and the other two sweeps\n\n"
           "> 💡 **In plain words** — does the answer depend on when you looked, on how "
           "expensive your borrowing is, or on your commissions? On this tape, none of "
           "the three: the halves look different but are not statistically "
           "distinguishable."),
        code(
            "print(f\"2x era cut: 2007-2016 {R['x2_era_e_adv']:+.3f} (t {R['x2_era_e_t']:+.2f})  |  \"\n"
            "      f\"2017-2026 {R['x2_era_l_adv']:+.3f} (t {R['x2_era_l_t']:+.2f})\")\n"
            "print(f\"  test of the DIFFERENCE (gap on an era dummy): \"\n"
            "      f\"{R['x2_era_diff_bps']:+.3f} bps/day, HAC t {R['x2_era_diff_t']:+.2f} \"\n"
            "      f\"-- NOT distinguishable, so no decay is claimed\")\n"
            "print(f\"3x era cut: 2009-2016 {R['x3_era_e_adv']:+.3f} (t {R['x3_era_e_t']:+.2f})  |  \"\n"
            "      f\"2017-2026 {R['x3_era_l_adv']:+.3f} (t {R['x3_era_l_t']:+.2f})   \"\n"
            "      f\"difference {R['x3_era_diff_bps']:+.3f} bps/day \"\n"
            "      f\"(t {R['x3_era_diff_t']:+.2f})\")\n"
            "print()\n"
            "print(f\"2x financing spread   0 bps: adv {R['x2_sp0_adv']:+.3f}   \"\n"
            "      f\"200 bps: adv {R['x2_sp200_adv']:+.3f}\")\n"
            "print(f\"2x trading cost       0 bps: adv {R['x2_c0_adv']:+.3f}   \"\n"
            "      f\"10 bps one-way: adv {R['x2_c10_adv']:+.3f}\")\n"
            "print('(both arms hold nearly the same average exposure, so financing and cost\\n'\n"
            "      ' hit them almost identically -- neither sweep can rescue or kill the result)')"
        ),

        md("## 6. The 2008 stress UPRO never saw\n\n"
           "> 💡 **In plain words** — UPRO launched after the crash. So we replay the "
           "construction on SPY through it."),
        code(
            "print(f\"SPY {R['st_start']} -> {R['st_end']} ({R['st_n']:,} days), ^IRX accrual as cash\")\n"
            "print(f\"  2x monthly x{R['st_2x_mon']:.1f}   vs 2x daily x{R['st_2x_day']:.1f}  (never called)\")\n"
            "print(f\"  3x monthly, no lender limit: x{R['st_3x_mon_free']:.1f}, \"\n"
            "      f\"peak exposure {R['st_3x_mon_free_w']:.2f}x\")\n"
            "print(f\"  3x monthly, 25% maintenance: x{R['st_3x_mon_25']:.2f}, called {R['st_3x_liq']}\")\n"
            "print(f\"  3x daily                   : x{R['st_3x_day']:.1f}\")\n"
            "print(f\"  4x monthly, no lender limit: x{R['st_4x_mon_free']:.2f} -- NEGATIVE EQUITY \"\n"
            "      f\"{R['st_4x_liq']}\")\n"
            "print(f\"  4x daily                   : x{R['st_4x_day']:.1f} -- cannot go below zero\")"
        ),
        md("The 4x pair is the cleanest statement of what the daily reset is *for*. A "
           "constant-leverage fund's terminal value is bounded below by zero by "
           "construction; a drift-then-reset margin account's is not."),

        md("## 7. Live synthetic control — the machinery is unbiased\n\n"
           "An AR(1) choppiness knob at **fixed total volatility**, so only the path shape "
           "changes. Liquidation disabled, seeds 943+."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from reset_freq import data, strategy as st\n"
            "for tag, phi, ss in [('choppy  phi=-0.15', -0.15, 1.0),\n"
            "                     ('trending phi=+0.15', 0.15, 1.0),\n"
            "                     ('iid null phi= 0.00', -0.15, 0.0)]:\n"
            "    d = [st.synthetic_detect(data.synthetic_daily(phi=phi, signal_strength=ss,\n"
            "                                                  n_years=12, seed=943+s)[0])\n"
            "         for s in range(4)]\n"
            "    print(f\"{tag}: gap {np.mean([x['mean_gap_bps'] for x in d]):+.3f} bps/day \"\n"
            "          f\"(HAC t {np.mean([x['t_gap'] for x in d]):+.2f})  \"\n"
            "          f\"ER slope {np.mean([x['chop_slope'] for x in d]):+.2f}\")"
        ),
        md("Correct sign in both planted worlds, centred at zero on the null — and the ER "
           "slope stays strongly negative in all three, which is exactly the caveat of "
           "section 3, pinned as a live measurement (and as a unit test)."),

        md(f"## Verdict\n\n"
           f"- **Signal — Mixed.** *Real on the return, absent on the Sharpe; positive "
           f"at 2x, negative at 3x.* The reset-frequency effect on *returns* is real and "
           f"clears the bar at 2x ({R['x2_diff_bps']:+.2f} bps/day, HAC *t* = "
           f"{R['x2_t']:+.2f}), with the theory-predicted conditional sign in both sleeves "
           f"(slope *t* = {R['x2_slope_t']:+.2f} / {R['x3_slope_t']:+.2f}) and a synthetic "
           f"control that recovers it and stays quiet on the null. The claim under test "
           f"nevertheless fails: risk-adjusted, the 2x advantage is {R['x2_adv']:+.3f} "
           f"with CI [{R['x2_adv_ci_lo']:+.3f}, {R['x2_adv_ci_hi']:+.3f}], the lagged "
           f"(tradable) form is *t* = {R['x2_pred_t']:+.2f}, and the 3x advantage is "
           f"{R['x3_adv']:+.3f} (*t* = {R['x3_t']:+.2f}) — or {R['x3_m0_adv']:+.3f} even "
           f"with an infinitely patient lender. The post-2017 half is weaker "
           f"({R['x2_era_l_adv']:+.3f} against {R['x2_era_e_adv']:+.3f}) but the era "
           f"difference is *t* = {R['x2_era_diff_t']:+.2f}, so that is a hint, not a "
           f"decay. Survivorship note: SSO and UPRO are the survivors, and the closed "
           f"leveraged funds are not on this tape.\n"
           f"- **Tradability — Mirage.** Nothing bankable is on offer as a *reset* edge. "
           f"The monthly reset sells uncontrolled leverage at an unchanged *average* "
           f"leverage — peak {R['x3_w_max_free']:.2f}x on a {R['x3_w_mean_free']:.2f} mean, "
           f"a call on {R['x3_liq']}, negative equity at 4x in 2008 — and roughly half of "
           f"its apparent advantage over the funds is their {R['x2_fee_leg']:.2f}%/yr "
           f"fee-and-tracking drag, which is a different study's question and is only "
           f"yours if you borrow near the bill rate."),
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
