"""Generate the two narrative notebooks for Study 958 (Spot ETF Basis).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic apart from one clearly-labelled real-tape
chart that reads the shared cache (and says so plainly if the cache is absent, rather
than quietly drawing synthetic data under a real banner). Every real-tape number is
quoted from the frozen ``R`` dict below, which mirrors ``docs/results.md``; the live
cells run the fast synthetic control.
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


# Frozen real-tape headline — mirror of docs/results.md. Total-return closes,
# as-of 2026-06-30, BITO/BTC-USD window 2021-10-20 -> 2026-06-30.
R = dict(
    start="2021-10-20", end="2026-06-30", n_days=1177, fp="b7f384a06a79",
    launch="2024-01-11", n_post=618,
    # ruler calibration
    ibit_trend=-0.250, ibit_se=0.049, ibit_t=-5.09, ibit_naive=-0.416, ibit_naive_t=-0.07,
    fbtc_trend=-0.205, fbtc_se=0.047, fbtc_t=-4.33, fbtc_naive=-0.459, fbtc_naive_t=-0.08,
    fee_ibit=0.25, fee_fbtc=0.25, fee_bito=0.95,
    # drags
    full_trend=-5.686, full_se=0.159, full_t=-35.77, full_naive=-5.530, full_naive_t=-1.36,
    post_trend=-5.972, post_se=0.204, post_t=-29.34, post_naive=-5.645, post_naive_t=-1.02,
    vs_ibit=-5.722, vs_ibit_t=-30.68, vs_ibit_naive=-5.228, vs_ibit_naive_t=-7.24,
    vs_fbtc=-5.767, vs_fbtc_t=-29.58,
    # non-overlapping monthly cross-check (plain t, no HAC): drag, t, n months
    full_mo=-5.278, full_mo_t=-2.09, post_mo=-5.629, post_mo_t=-2.17,
    vs_ibit_mo=-5.475, vs_ibit_mo_t=-9.83, vs_fbtc_mo=-5.501, vs_fbtc_mo_t=-8.92,
    mo_n_full=56, mo_n_post=29,
    # trend-fit residual diagnostics (DF t below ~-3.4 = stationary = the trend t means something)
    ar1_btc=0.740, df_btc=-7.46, ar1_ibit=0.981, df_ibit=-3.00, ar1_fbtc=0.982, df_fbtc=-2.51,
    boot_full_lo=-10.45, boot_full_hi=-0.66, boot_ibit_lo=-6.62, boot_ibit_hi=-3.84,
    lag_t={20: -35.77, 60: -21.78, 120: -16.51, 250: -13.90},
    lag_t_post={20: -29.34, 60: -20.03, 120: -17.50, 250: -17.67},
    lag_t_era={20: -9.50, 60: -7.02, 120: -6.57, 250: -7.34},
    lag_t_matched={20: -0.08, 60: -0.09, 120: -0.10, 250: -0.10},
    # implied basis
    pre_drag=-2.548, pre_cash=2.89, pre_basis=4.49, pre_excess=1.60,
    post_drag=-5.972, post_cash=4.42, post_basis=9.44, post_excess=5.02,
    fee_lo_basis=9.64, fee_hi_basis=9.24,
    # era tests
    era_pre=-2.564, era_post=-5.969, era_chg=-3.405, era_se=0.359, era_t=-9.50,
    m_pre=-7.316, m_post=-7.351, m_chg=-0.035, m_se=0.444, m_t=-0.08, m_n=502,
    plac_n=44, plac_med=4.48, plac_max=10.48, plac_rank=7, plac_frac=16,
    # matched-window sweep: half-width months -> (pre, post, change, t, n)
    win={
        6: (-7.40, -9.60, -2.20, -2.47, 252),
        9: (-7.19, -8.08, -0.89, -1.31, 379),
        12: (-7.32, -7.35, -0.03, -0.08, 502),
        18: (-2.76, -7.26, -4.50, -6.43, 753),
        24: (-2.23, -6.68, -4.45, -12.04, 1003),
    },
    # calendar years: drag vs BTC-USD, HAC t, drag vs IBIT, cash, implied basis
    years={
        2022: (-2.61, -3.42, None, 1.43, 3.09),
        2023: (-7.08, -24.39, None, 5.01, 11.14),
        2024: (-7.41, -21.57, -7.18, 5.20, 11.66),
        2025: (-5.31, -30.41, -5.08, 4.15, 8.51),
        2026: (-3.13, -7.80, -3.07, 3.49, 5.67),
    },
    cyc_corr=-0.32, cyc_t=-1.89, cyc_t_ols=-15.04, cyc_neff=8.3,
    # harvest inference cross-checks
    h2_t250=3.74, h2_mo_t=4.35, h2_mo_pos="28/30",
    # harvest
    h0=5.39, h0_sharpe=2.83, h0_t=7.31,
    h1=4.27, h1_sharpe=2.25, h1_t=5.81,
    h2=3.28, h2_sharpe=1.72, h2_t=4.47, h2_dd=-0.67, h2_vol=1.90,
    h5=0.30, h5_sharpe=0.16, h5_t=0.41,
    boot_sh_lo=0.95, boot_sh_hi=2.48,
    yr24_net=4.50, yr25_net=3.22, yr26_net=0.99,
    yr24_gross=6.67, yr25_gross=5.29, yr26_gross=3.06,
    fbtc_pair=3.20, fbtc_pair_t=4.31,
    # synthetic control
    syn_fee_planted=-0.25, syn_fee_read=-0.285,
    syn_planted_chg=7.00, syn_read_chg=7.31, syn_read_t=33.21,
    syn_null_chg=0.31, syn_null_t=1.43, syn_null_mean=0.09, syn_null_sd=0.22, syn_null_fire=0,
)

BOOT = (
    "import os, sys\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
)


HEADER = f"""# Study 958 — Spot ETF Basis 🧊

**After the January-2024 spot bitcoin ETFs, is the futures-vs-spot carry still there?**

Until 2024 the only US-listed, 40-Act routes to bitcoin were a *futures* ETF (BITO, from
October 2021) and a closed-end *trust* (GBTC). Both charged their holders a wedge: the
futures wrapper paid a rich CME basis every day it held the front contract, the trust
swung between a +40% premium and a −49% discount. The tidy story says the spot ETFs, which
create and redeem physical bitcoin at NAV, should have **compressed** both.

We test that on **BITO vs BTC-USD** ({R['start']} → {R['end']}, {R['n_days']:,} sessions)
with **IBIT** and **FBTC** as independent spot rulers from {R['launch']}. Daily
**total-return** closes — BITO's distributions are reinvested; price-only would invent
most of the "drag".

*Numbers below are the frozen headline (`docs/results.md`, Fingerprint `{R['fp']}`); the
live cells run the fast synthetic control. As-of 2026-06-30.*
"""


def build_curious():
    cells = [
        md(HEADER),
        md("## 1. What a wrapper costs you\n\n"
           "An ETF that cannot legally hold bitcoin buys *next month's* bitcoin instead. "
           "Next month's bitcoin has been persistently more expensive than today's — that "
           "gap is the **basis** — and every month the fund watches the gap melt away and "
           "then pays it again. The bill shows up as a slow, silent bleed against the coin. "
           "Here is what that bill actually was."),
        code(
            "R = dict(post_trend=%r, post_t=%r, vs_ibit=%r, vs_ibit_t=%r,\n"
            "         post_mo_t=%r, vs_ibit_mo_t=%r,\n"
            "         post_basis=%r, post_excess=%r, fee_bito=%r)\n"
            "print('BITO vs bitcoin, since the spot ETFs launched: %%+.2f%%%%/yr "
            "(t = %%.1f, or %%.1f on one reading a month)'\n"
            "      %% (R['post_trend'], R['post_t'], R['post_mo_t']))\n"
            "print('BITO vs IBIT, same trading hours          : %%+.2f%%%%/yr "
            "(t = %%.1f, or %%.1f on one reading a month)'\n"
            "      %% (R['vs_ibit'], R['vs_ibit_t'], R['vs_ibit_mo_t']))\n"
            "print()\n"
            "print('of which the fund fee is only %%.2f%%%% -- the rest is the futures carry'\n"
            "      %% R['fee_bito'])\n"
            "print('implied basis %%+.1f%%%%/yr, i.e. %%+.1f%%%%/yr ON TOP of cash'\n"
            "      %% (R['post_basis'], R['post_excess']))"
            % (R["post_trend"], R["post_t"], R["vs_ibit"], R["vs_ibit_t"],
               R["post_mo_t"], R["vs_ibit_mo_t"],
               R["post_basis"], R["post_excess"], R["fee_bito"])
        ),
        md("So the carry is emphatically **still there**: a bitcoin dollar held through the "
           "futures wrapper loses about **six cents a year** to a bitcoin dollar held "
           "through the spot wrapper. The spot ETFs did not make that go away.\n\n"
           "The second line is the one to trust. Comparing a fund to the *coin* means "
           "comparing a 4pm New York price to a midnight UTC one; comparing it to another "
           "fund does not. Measured fund-against-fund, the gap survives even the bluntest "
           "possible test — take one reading a month, 29 readings, no clever statistics — "
           "at *t* = −9.8. Against the coin the same blunt test reads only −2.2."),
        md("## 2. Can we trust the ruler? Ask it to measure something we already know\n\n"
           "Comparing a fund to bitcoin is harder than it sounds: bitcoin trades 24/7 and "
           "the price we quote is stamped at midnight UTC, while the funds stop trading at "
           "4pm in New York. Sixteen hours of bitcoin is a *lot* of noise.\n\n"
           "The fix is to measure the gap as a **trend line** through every day, instead of "
           "just comparing the first and last day. And the way to check it works is to point "
           "it at a cost we already know exactly — a spot ETF's published fee."),
        code(
            "cal = %r\n"
            "for tk, trend, naive, fee in cal:\n"
            "    print('%%s: trend-line reads %%+.3f%%%%/yr | first-vs-last reads %%+.3f%%%%/yr | "
            "published fee %%+.2f%%%%/yr'\n"
            "          %% (tk, trend, naive, -fee))"
            % ([("IBIT", R["ibit_trend"], R["ibit_naive"], R["fee_ibit"]),
                ("FBTC", R["fbtc_trend"], R["fbtc_naive"], R["fee_fbtc"])],)
        ),
        md("> 🔬 **For the quants.** Because daily log differences telescope, "
           "`252 × mean(daily diff)` *is* the two-endpoint estimator — it discards every "
           "observation in between and inherits both endpoints' intraday offset. The "
           "trend slope uses all *n* points. On the real tape it reads IBIT's 25 bp fee to "
           "the basis point (*t* = −5.1) where the endpoint estimator cannot tell it from "
           "zero (*t* = −0.07)."),
        md(f"## 3. The actual question: did the launch change anything?\n\n"
           f"Split the whole sample at {R['launch']} and the answer looks dramatic — and "
           f"backwards. The bleed *widened*, from {R['era_pre']:.2f}%/yr to "
           f"{R['era_post']:.2f}%/yr.\n\n"
           f"But that comparison is rigged. The pre-launch window is mostly **2022**, the "
           f"crypto bear market, when nobody wanted levered bitcoin and the futures curve "
           f"barely sloped at all. Compare like with like — the twelve months either side of "
           f"the launch — and the answer is a flat nothing.\n\n"
           f"One caveat we owe you: twelve months is *a* choice, and it happens to be the "
           f"flattest one. Six, nine, eighteen and twenty-four months either side all say "
           f"the bleed got **worse**, not better. So the honest summary is not \"nothing "
           f"changed\" but \"nothing got cheaper\" — which is still the opposite of what the "
           f"tidy story predicted."),
        code(
            "print('full-sample split : %%+.2f%%%%/yr before  ->  %%+.2f%%%%/yr after   (change %%+.2f pp)'\n"
            "      %% (%r, %r, %r))\n"
            "print('matched 12 months : %%+.2f%%%%/yr before  ->  %%+.2f%%%%/yr after   (change %%+.2f pp, t = %%.2f)'\n"
            "      %% (%r, %r, %r, %r))\n"
            "print()\n"
            "print('and the calendar years either side of the launch:')"
            % (R["era_pre"], R["era_post"], R["era_chg"],
               R["m_pre"], R["m_post"], R["m_chg"], R["m_t"])
        ),
        code(
            "for y, row in %r.items():\n"
            "    tag = ' <- launch year' if y == 2024 else ''\n"
            "    part = ' (H1 only)' if y == 2026 else ''\n"
            "    print('%%d: drag %%+6.2f%%%%/yr   implied basis %%+6.2f%%%%/yr%%s%%s'\n"
            "          %% (y, row[0], row[4], tag, part))"
            % (R["years"],)
        ),
        md(f"**2023 and 2024 are the same number.** The wrapper wedge that the spot ETFs "
           f"*did* close was the other one — GBTC's discount, which snapped to NAV within "
           f"weeks (see [study 618](../../618-gbtc-premium-cycle/)). The futures basis was "
           f"never an access wedge; it is a **financing** rate, and it is priced by how "
           f"badly levered longs want exposure. That is why it is fat in the 2023-24 "
           f"melt-up and thin in the 2022 and 2026 drawdowns."),
        md(f"## 4. The trap we nearly fell into\n\n"
           f"That full-sample break had a *t* of {R['era_t']:.2f} — hugely 'significant'. "
           f"So we ran the identical test at {R['plac_n']} **arbitrary** dates that mean "
           f"nothing. The median arbitrary date produces |*t*| = {R['plac_med']:.2f}, and "
           f"the launch ranks only **{R['plac_rank']} of {R['plac_n']}**. A slowly wandering "
           f"series manufactures a 'significant' before/after break almost anywhere you cut "
           f"it. The event date was not special."),
        md("## 5. Live check — the machinery is unbiased (offline synthetic)\n\n"
           "**This cell uses synthetic data, not the real tape.** We build a toy world with "
           "a known wrapper fee, a known futures carry and the same midnight-vs-4pm "
           "timestamp problem, then plant a compression at the event date. The detector must "
           "find it — and must stay quiet when the carry is large but unchanged."),
        code(
            BOOT +
            "from etf_basis import data, strategy as st\n"
            "planted = st.synthetic_detect(*data.synthetic_panel(signal_strength=1.0, seed=958))\n"
            "null    = st.synthetic_detect(*data.synthetic_panel(signal_strength=0.0, seed=958))\n"
            "print('SYNTHETIC world, compression planted at %+.2f pp -> detector reads %+.2f pp (t = %+.1f)'\n"
            "      % (planted['planted_change_pct'], planted['era_change_pct'], planted['era_t']))\n"
            "print('SYNTHETIC world, NO compression (same big carry) -> detector reads %+.2f pp (t = %+.2f)'\n"
            "      % (null['era_change_pct'], null['era_t']))\n"
            "print('SYNTHETIC ruler on a planted %.2f%%/yr fee -> reads %+.3f%%/yr'\n"
            "      % (null['planted_fee_pct'], null['spot_etf_drag_pct']))"
        ),
        md(f"## 6. Is there money in it?\n\n"
           f"If the futures wrapper bleeds ~6%/yr against the spot wrapper, buy one and "
           f"short the other. It works — while your broker lends you BITO cheaply. At a 2% "
           f"borrow fee and 5 bps of trading cost the pair nets **+{R['h2']:.2f}%/yr** with "
           f"a Sharpe of {R['h2_sharpe']:.2f} and a worst loss of {abs(R['h2_dd']):.2f}%. At "
           f"a 5% borrow fee it nets **+{R['h5']:.2f}%/yr** — nothing. And it is fading: "
           f"+{R['yr24_net']:.2f}% in 2024, +{R['yr25_net']:.2f}% in 2025, "
           f"+{R['yr26_net']:.2f}% in the first half of 2026.\n\n"
           f"The version that always works needs no borrow at all: **own the spot wrapper "
           f"instead of the futures one.** That is not alpha — it is not paying a bill you "
           f"do not have to pay."),
        md(f"## Verdict\n\n"
           f"- **Signal — Mixed.** The carry is unambiguously alive "
           f"({R['vs_ibit']:+.2f}%/yr fund-against-fund, which survives even a "
           f"one-reading-a-month test at *t* = {R['vs_ibit_mo_t']:.2f}; "
           f"+{R['post_excess']:.1f}%/yr "
           f"above cash), on a ruler that reads a known 25 bp fee exactly. But the claim "
           f"under test — that the spot ETFs compressed it — is **rejected**: matched "
           f"twelve-month windows differ by {R['m_chg']:+.2f} pp (*t* = {R['m_t']:.2f}), and "
           f"the impressive full-sample break has the wrong sign and ranks "
           f"{R['plac_rank']}/{R['plac_n']} against placebo dates.\n"
           f"- **Tradability — Fragile.** Long spot / short futures nets "
           f"+{R['h2']:.2f}%/yr at 2% borrow, dies at 5%, and is shrinking year by year. "
           f"Bankable only for someone with cheap, reliable borrow — and the risk-free "
           f"alternative (owning the spot wrapper) is cost avoidance, not an edge."),
    ]
    nb = new_notebook()
    nb["cells"] = cells
    return nb


def build_quants():
    cells = [
        md("# Study 958 — Spot ETF Basis — the teardown\n\n"
           "The trend-slope tracking-difference estimator and its fee calibration, the "
           "broken-trend era test at the 2024-01-11 spot-ETF launch, a 44-date placebo "
           "sweep, HAC bandwidths out to 250 lags, the implied-basis decomposition with its "
           "fee sweep, the borrow-swept harvest, and the planted-compression synthetic "
           "control. Every real-tape number is frozen from `docs/results.md` "
           "(Fingerprint `%s`, as-of 2026-06-30)." % R["fp"]),
        code("R = %r" % (R,)),
        md("## 1. The estimator\n\n"
           "For a wrapper *W* and reference *S*, the tracking difference is "
           "$c_t = \\log(W_t/W_0) - \\log(S_t/S_0)$ on the wrapper's sessions. A constant "
           "annual bleed makes $c_t$ a straight line, so the drag is the slope.\n\n"
           "Two estimators of that slope. The **naive** one, "
           "$252\\,\\overline{\\Delta c}$, telescopes to $(c_T - c_0)/T$ — the two-endpoint "
           "estimator. The **trend** one is the HAC-robust OLS slope of $c_t$ on calendar "
           "time in years. They target the same quantity; only one of them survives a "
           "reference stamped at a different hour (`BTC-USD` at 00:00 UTC, the ETFs at "
           "16:00 New York).\n\n"
           "The calibration test uses a *known* cost: a spot wrapper's own expense ratio."),
        code(
            "print(f\"IBIT: trend {R['ibit_trend']:+.3f}%/yr (se {R['ibit_se']:.3f}, t {R['ibit_t']:+.2f})  \"\n"
            "      f\"naive {R['ibit_naive']:+.3f}%/yr (t {R['ibit_naive_t']:+.2f})  \"\n"
            "      f\"published fee {-R['fee_ibit']:+.2f}%/yr\")\n"
            "print(f\"FBTC: trend {R['fbtc_trend']:+.3f}%/yr (se {R['fbtc_se']:.3f}, t {R['fbtc_t']:+.2f})  \"\n"
            "      f\"naive {R['fbtc_naive']:+.3f}%/yr (t {R['fbtc_naive_t']:+.2f})  \"\n"
            "      f\"published fee {-R['fee_fbtc']:+.2f}%/yr (waived to 2024-07-31)\")"
        ),
        md("> 💡 **In plain words.** Point both rulers at a fund whose cost we already know. "
           "The trend ruler reads BlackRock's 25 bp fee as 25 bp. The endpoint ruler reads "
           "it as anything between −0.4% and nothing at all. Everything downstream uses the "
           "first one."),
        md("## 2. The drags, and the assumption-light cross-checks\n\n"
           "BITO against the coin (timestamp-offset reference) and against each spot wrapper "
           "(matched closes, no offset at all), read three ways: the efficient **trend** "
           "slope with a HAC *t*, the **endpoint** estimator, and the mean of "
           "**non-overlapping month-end gaps** with an ordinary *t* and no HAC assumption "
           "whatsoever.\n\n"
           "Read the *t*'s from right to left. The trend slope regresses a serially "
           "dependent level on time, so its HAC *t* is an upper bound on the evidence, not "
           "the evidence; against the *coin* the honest estimators fall to |*t*| ≈ 2.1. "
           "The certification therefore rests on the **matched-close** pairs, where all "
           "three agree at |*t*| between 7.2 and 9.8 — and all four point estimates sit "
           "within 50 bp/yr of each other regardless."),
        code(
            "rows = [('BITO vs BTC-USD full', R['full_trend'], R['full_t'], R['full_naive'], R['full_naive_t'], R['full_mo'], R['full_mo_t'], R['mo_n_full']),\n"
            "        ('BITO vs BTC-USD post', R['post_trend'], R['post_t'], R['post_naive'], R['post_naive_t'], R['post_mo'], R['post_mo_t'], R['mo_n_post']),\n"
            "        ('BITO vs IBIT   post', R['vs_ibit'], R['vs_ibit_t'], R['vs_ibit_naive'], R['vs_ibit_naive_t'], R['vs_ibit_mo'], R['vs_ibit_mo_t'], R['mo_n_post']),\n"
            "        ('BITO vs FBTC   post', R['vs_fbtc'], R['vs_fbtc_t'], None, None, R['vs_fbtc_mo'], R['vs_fbtc_mo_t'], R['mo_n_post'])]\n"
            "print(f\"{'pair':22s} {'trend (HAC t)':>22} {'endpoint (t)':>20} {'monthly non-overlap (t)':>28}\")\n"
            "for name, tr, t, nv, nvt, mo, mot, mon in rows:\n"
            "    nvs = f\"{nv:+.3f} ({nvt:+.2f})\" if nv is not None else '-'\n"
            "    print(f\"{name:22s} {f'{tr:+.3f} ({t:+.2f})':>22} {nvs:>20} \"\n"
            "          f\"{f'{mo:+.3f} ({mot:+.2f}, {mon} m)':>28}\")\n"
            "print()\n"
            "print(f\"block-bootstrap CI on the naive drag: vs IBIT [{R['boot_ibit_lo']:+.2f}, {R['boot_ibit_hi']:+.2f}]%/yr  \"\n"
            "      f\"vs BTC-USD [{R['boot_full_lo']:+.2f}, {R['boot_full_hi']:+.2f}]%/yr\")\n"
            "print()\n"
            "print('trend-fit residual diagnostics (DF t below -3.4 rejects a unit root):')\n"
            "for tag, a, d in [('vs BTC-USD', R['ar1_btc'], R['df_btc']),\n"
            "                  ('vs IBIT   ', R['ar1_ibit'], R['df_ibit']),\n"
            "                  ('vs FBTC   ', R['ar1_fbtc'], R['df_fbtc'])]:\n"
            "    verdict = 'stationary -> the trend t means something' if d < -3.4 else \\\n"
            "              'NOT rejected -> discount that trend t, read the monthly column'\n"
            "    print(f\"  BITO {tag}  AR(1) {a:+.3f}  DF t {d:+.2f}   {verdict}\")"
        ),
        md("### HAC bandwidth sensitivity\n\n"
           "The residual basis wanders at multi-month scale, so a 20-lag Bartlett window is "
           "generous. Widening it to a full trading year does not change any conclusion."),
        code(
            "print(f\"{'lags':>5} {'full drag t':>12} {'post drag t':>12} {'era chg t':>10} {'matched t':>10}\")\n"
            "for lg in (20, 60, 120, 250):\n"
            "    print(f\"{lg:5d} {R['lag_t'][lg]:12.2f} {R['lag_t_post'][lg]:12.2f} \"\n"
            "          f\"{R['lag_t_era'][lg]:10.2f} {R['lag_t_matched'][lg]:10.2f}\")"
        ),
        md("## 3. Drag → implied basis\n\n"
           "A fully collateralised long futures wrapper earns "
           "$r_S - b + r_c - f$ per year, so its drag against spot is $-b + r_c - f$ and\n\n"
           "$$ b = r_c - f - \\text{drag}, \\qquad b - r_c = -f - \\text{drag}. $$\n\n"
           "The excess-of-cash basis does not depend on the cash assumption at all. "
           "**ASSUMPTIONS:** $f$ = BITO's 0.95% published expense ratio; $r_c$ = BIL's "
           "realised total return as the collateral-yield proxy. Both are swept."),
        code(
            "print(f\"pre-launch : drag {R['pre_drag']:+.3f}%/yr  cash {R['pre_cash']:.2f}%  \"\n"
            "      f\"-> basis {R['pre_basis']:+.2f}%/yr  (excess of cash {R['pre_excess']:+.2f}%/yr)\")\n"
            "print(f\"post-launch: drag {R['post_drag']:+.3f}%/yr  cash {R['post_cash']:.2f}%  \"\n"
            "      f\"-> basis {R['post_basis']:+.2f}%/yr  (excess of cash {R['post_excess']:+.2f}%/yr)\")\n"
            "print(f\"fee sweep (PROXY 0.95%): 0.75% -> {R['fee_lo_basis']:+.2f}%/yr, \"\n"
            "      f\"1.15% -> {R['fee_hi_basis']:+.2f}%/yr  (one-for-one, no qualitative change)\")"
        ),
        md("## 4. The era test — and why the significant version is the wrong one\n\n"
           "Broken trend "
           "$c_t = a + b t + g\\,\\mathbb{1}\\{t>t_0\\} + h\\,(t-t_0)\\mathbb{1}\\{t>t_0\\}$, "
           "with $h$ the change in slope (**positive = the carry compressed**) and $g$ "
           "absorbing any one-day jump so it cannot leak into $h$."),
        code(
            "print(f\"full sample    : pre {R['era_pre']:+.3f}  post {R['era_post']:+.3f}  \"\n"
            "      f\"change {R['era_chg']:+.3f} pp (se {R['era_se']:.3f}, t {R['era_t']:+.2f})\")\n"
            "print(f\"matched +/-12m : pre {R['m_pre']:+.3f}  post {R['m_post']:+.3f}  \"\n"
            "      f\"change {R['m_chg']:+.3f} pp (se {R['m_se']:.3f}, t {R['m_t']:+.2f})  n={R['m_n']}\")\n"
            "print()\n"
            "print('the full-sample pre-window is mostly the 2022 backwardation, so the two')\n"
            "print('windows are not comparable; the matched cut is the more honest one.')\n"
            "print()\n"
            "print('but ONE window is a CHOICE, so here is the whole family:')\n"
            "print(f\"{'window':>8} {'pre':>8} {'post':>8} {'change':>9} {'HAC t':>8} {'n':>6}\")\n"
            "for m, (pre, post, chg, t, n) in R['win'].items():\n"
            "    flag = '  <- the flattest of the family, i.e. the headline' if m == 12 else ''\n"
            "    print(f\"  +/-{m:2d}m {pre:8.2f} {post:8.2f} {chg:9.2f} {t:8.2f} {n:6d}{flag}\")\n"
            "print()\n"
            "print('NOT ONE width compresses (a compression would be a POSITIVE change);')\n"
            "print('several are significantly wrong-signed. The conclusion survives the choice,')\n"
            "print('but the +/-12m headline is the reading most favourable to it -- said out loud.')"
        ),
        md("### Placebo split sweep\n\n"
           "Bertrand-Duflo-Mullainathan in one picture: with a serially correlated outcome, "
           "a single before/after break is far less informative than its *t* suggests. Rerun "
           "the identical broken-trend test at every month-start with at least six months on "
           "each side."),
        code(
            "print(f\"{R['plac_n']} arbitrary split dates: median |t| {R['plac_med']:.2f}, max |t| {R['plac_max']:.2f}\")\n"
            "print(f\"the launch's |t| = {abs(R['era_t']):.2f} ranks {R['plac_rank']}/{R['plac_n']} \"\n"
            "      f\"({R['plac_frac']}% of arbitrary dates are at least as extreme)\")"
        ),
        md("## 5. Calendar-year fingerprint — cycle, not plumbing"),
        code(
            "print(f\"{'year':>6} {'drag vs BTC':>12} {'HAC t':>8} {'vs IBIT':>9} {'cash':>6} {'basis':>8}\")\n"
            "for y, (d, t, di, c, b) in R['years'].items():\n"
            "    dis = f\"{di:+.2f}\" if di is not None else '  -'\n"
            "    note = '  <- launch' if y == 2024 else ('  (H1)' if y == 2026 else '')\n"
            "    print(f\"{y:6d} {d:12.2f} {t:8.2f} {dis:>9} {c:6.2f} {b:8.2f}{note}\")\n"
            "print()\n"
            "print(f\"rolling-126d drag vs trailing 126d BTC return: corr {R['cyc_corr']:.2f} \"\n"
            "      f\"(HAC(250) t {R['cyc_t']:+.2f}) -- suggestive, NOT certified\")\n"
            "print(f\"  the same regression's naive OLS t is {R['cyc_t_ols']:+.2f}: the windows overlap\\n\"\n"
            "      f\"  126 deep and only ~{R['cyc_neff']:.1f} of them are independent, so that number is\\n\"\n"
            "      f\"  shown purely to size the inflation. strategy.cycle_regression reproduces both.\")"
        ),
        md("### The real tape, drawn\n\n"
           "The cumulative log tracking difference of each wrapper against the coin. This "
           "cell reads the **shared real cache**; if the cache is absent it says so and "
           "draws nothing rather than substituting synthetic data under a real banner."),
        code(
            BOOT +
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "from etf_basis import data, strategy as st\n"
            "if not data.have_real():\n"
            "    print('real cache absent -- chart skipped (no synthetic substitute under a real banner)')\n"
            "else:\n"
            "    px = data.load_prices()\n"
            "    fig, ax = plt.subplots(figsize=(9, 4.2))\n"
            "    for tk, colour in [('BITO', '#c0392b'), ('IBIT', '#2ea44f'), ('FBTC', '#1f77b4')]:\n"
            "        c = st.cumulative_diff(px[tk].dropna(), px['BTC-USD'].dropna()) * 100\n"
            "        ax.plot(c.index, c.values, color=colour, lw=1.2, label=tk)\n"
            "    ax.axvline(pd.Timestamp(data.LAUNCH), color='0.4', ls='--', lw=1)\n"
            "    ax.annotate('spot ETFs launch', xy=(pd.Timestamp(data.LAUNCH), ax.get_ylim()[0]),\n"
            "                xytext=(6, 8), textcoords='offset points', color='0.35', fontsize=9)\n"
            "    ax.axhline(0, color='0.7', lw=0.8)\n"
            "    ax.set_ylabel('cumulative tracking difference vs BTC-USD (log %)')\n"
            "    ax.set_title('REAL TAPE -- total return, as-of 2026-06-30')\n"
            "    ax.legend(loc='lower left', frameon=False)\n"
            "    fig.tight_layout()\n"
            "    plt.show()"
        ),
        md("> 💡 **In plain words.** The green and blue lines (spot ETFs) are nearly flat — "
           "they cost you their fee and nothing else. The red line (futures ETF) walks "
           "steadily downhill, and its slope does not kink at the dashed launch line."),
        md("## 6. Tradability — long IBIT / short BITO\n\n"
           "Equal notional, reset every 21 sessions. **The single execution lag:** the reset "
           "uses the close of day *t* and is live from *t+1*; between resets the weights "
           "drift with the legs. The short leg pays borrow (an ASSUMPTION, swept) every day "
           "it is on; both legs pay one-way cost × NAV on the notional traded at each "
           "reset.\n\n"
           "**Financing (an ASSUMPTION).** The book is dollar-neutral and self-financing — "
           "the short proceeds pay for the long leg and the rebate on them is assumed to "
           "offset the cash the long leg gives up — so what follows is already **excess of "
           "cash on both legs**, with no risk-free rate left to subtract. `borrow` is the "
           "*incremental* loan fee on top of that wash, which is why the 5% row matters: it "
           "is the world where the rebate collapses."),
        code(
            "print(f\"{'borrow':>7} {'net %/yr':>10} {'Sharpe':>8} {'HAC t':>8}\")\n"
            "for lbl, ann, sh, t in [('0%', R['h0'], R['h0_sharpe'], R['h0_t']),\n"
            "                        ('1%', R['h1'], R['h1_sharpe'], R['h1_t']),\n"
            "                        ('2%', R['h2'], R['h2_sharpe'], R['h2_t']),\n"
            "                        ('5%', R['h5'], R['h5_sharpe'], R['h5_t'])]:\n"
            "    print(f\"{lbl:>7} {ann:10.2f} {sh:8.2f} {t:8.2f}\")\n"
            "print(f\"\\nat 2% borrow / 5 bps: vol {R['h2_vol']:.2f}%, worst DD {R['h2_dd']:.2f}%, \"\n"
            "      f\"bootstrap Sharpe CI [{R['boot_sh_lo']:+.2f}, {R['boot_sh_hi']:+.2f}]\")\n"
            "print(f\"  inference cross-checks: HAC(250) t {R['h2_t250']:+.2f}; non-overlapping \"\n"
            "      f\"monthly t {R['h2_mo_t']:+.2f} ({R['h2_mo_pos']} months positive)\")\n"
            "print(f\"FBTC as the long leg: {R['fbtc_pair']:+.2f}%/yr net (t {R['fbtc_pair_t']:+.2f})\")\n"
            "print(f\"\\nby year (2% borrow, 5 bps): gross {R['yr24_gross']:+.2f}/{R['yr25_gross']:+.2f}/{R['yr26_gross']:+.2f}%  \"\n"
            "      f\"net {R['yr24_net']:+.2f}/{R['yr25_net']:+.2f}/{R['yr26_net']:+.2f}%  (2024/2025/2026H1)\")"
        ),
        md("## 7. Synthetic control — the machinery is unbiased\n\n"
           "**Synthetic data, not the real tape.** A planted world: a spot wrapper bleeding a "
           "known fee, a futures wrapper bleeding `fee + basis − collateral`, and every "
           "wrapper close stamped at a different hour from the reference. The detector must "
           "recover a planted compression *and* stay quiet when the carry is large but "
           "unchanged — the exact failure mode that would have manufactured our headline."),
        code(
            BOOT +
            "import numpy as np\n"
            "from etf_basis import data, strategy as st\n"
            "pl = st.synthetic_detect(*data.synthetic_panel(signal_strength=1.0, seed=958))\n"
            "nl = st.synthetic_detect(*data.synthetic_panel(signal_strength=0.0, seed=958))\n"
            "print(f\"SYNTHETIC ruler : planted fee {nl['planted_fee_pct']:+.2f}%/yr -> reads {nl['spot_etf_drag_pct']:+.3f}%/yr\")\n"
            "print(f\"SYNTHETIC planted compression {pl['planted_change_pct']:+.2f} pp -> reads \"\n"
            "      f\"{pl['era_change_pct']:+.2f} pp (t {pl['era_t']:+.2f})\")\n"
            "print(f\"SYNTHETIC null (same big carry, no break) -> reads \"\n"
            "      f\"{nl['era_change_pct']:+.2f} pp (t {nl['era_t']:+.2f})\")\n"
            "nulls = np.array([st.synthetic_detect(*data.synthetic_panel(signal_strength=0.0, seed=958+s))['era_change_pct']\n"
            "                  for s in range(8)])\n"
            "print(f\"SYNTHETIC null x8 seeds: mean {nulls.mean():+.2f} pp (sd {nulls.std(ddof=1):.2f}), \"\n"
            "      f\"|change| >= 2 pp in {(np.abs(nulls) >= 2.0).sum()}/8\")\n"
            "print(f\"SYNTHETIC endpoint estimator on the same world: {nl['fut_drag_naive_pct']:+.2f}%/yr \"\n"
            "      f\"vs trend {nl['fut_drag_trend_pct']:+.2f}%/yr\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Mixed.** The futures carry is still there at Real-grade strength, "
           f"and it is the **matched-close** measure that certifies it: "
           f"{R['vs_ibit']:+.2f}%/yr against IBIT, holding on every estimator including "
           f"non-overlapping monthly gaps with no HAC at all "
           f"(*t* = {R['vs_ibit_mo_t']:.2f}; endpoint *t* = {R['vs_ibit_naive_t']:.2f}; the "
           f"trend *t* = {R['vs_ibit_t']:.1f} is reported but discounted — that residual is "
           f"near a unit root, DF *t* = {R['df_ibit']:.2f}). Against the coin it reads "
           f"{R['post_trend']:+.2f}%/yr "
           f"(trend *t* = {R['post_t']:.1f}, {R['lag_t_post'][250]:.1f} at 250 lags — but only "
           f"{R['post_mo_t']:.2f} monthly: sixteen hours of bitcoin sits in that ruler). An "
           f"implied front basis of "
           f"{R['post_basis']:+.1f}%/yr, {R['post_excess']:+.1f}%/yr above cash — on a ruler "
           f"calibrated to read a 25 bp fee exactly. The **compression claim is rejected**: "
           f"matched twelve-month windows read {R['m_pre']:.2f} vs {R['m_post']:.2f}%/yr "
           f"(change {R['m_chg']:+.2f} pp, *t* = {R['m_t']:.2f}) — and **no** symmetric width "
           f"from ±6 to ±24 months compresses, several being significantly wrong-signed, so "
           f"the answer does not turn on the window (though ±12m is the flattest member of "
           f"that family and is labelled as such); the full-sample break is "
           f"wrong-signed and ranks {R['plac_rank']}/{R['plac_n']} against placebo dates; the "
           f"gradual 2025-26 fade tracks the bitcoin cycle (corr {R['cyc_corr']:.2f}, "
           f"*t* = {R['cyc_t']:.2f} — suggestive only). Synthetic control fires on a planted "
           f"break ({R['syn_read_chg']:+.2f} pp, *t* = {R['syn_read_t']:.1f}) and is silent on "
           f"the null (mean {R['syn_null_mean']:+.2f} pp, {R['syn_null_fire']}/8). No "
           f"survivorship — three continuously listed instruments — though BITO is the "
           f"*surviving* futures wrapper, so the class-wide toll is if anything understated.\n"
           f"- **Tradability — Fragile.** +{R['h2']:.2f}%/yr net at 2% borrow and 5 bps, "
           f"excess of cash on both legs "
           f"(Sharpe {R['h2_sharpe']:.2f}, *t* = {R['h2_t']:+.2f}, {R['h2_t250']:+.2f} at 250 "
           f"lags, non-overlapping monthly *t* = {R['h2_mo_t']:+.2f}, bootstrap CI "
           f"[{R['boot_sh_lo']:+.2f}, {R['boot_sh_hi']:+.2f}], worst drawdown "
           f"{abs(R['h2_dd']):.2f}%), dead at 5% borrow (+{R['h5']:.2f}%/yr, "
           f"*t* = {R['h5_t']:+.2f}), decaying year by year, and capacity-bounded by BITO's "
           f"borrow pool. The riskless version — owning the spot wrapper — is cost avoidance, "
           f"not alpha."),
    ]
    nb = new_notebook()
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
