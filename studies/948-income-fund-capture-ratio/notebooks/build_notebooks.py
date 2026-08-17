"""Generate the two narrative notebooks for Study 948 (Capture Ratio).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        01_for_the_curious.ipynb 02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below (a mirror of ``docs/results.md``); the only live cells run the fast
synthetic control and the estimator's null-bias check, which are synthetic by construction
and are labelled as such — no synthetic cell ever sits under a real-tape banner.
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
# 14 income ETFs vs SPY/QQQ/IWM, cash = BIL, monthly total returns, as-of 2026-06-30.
# --------------------------------------------------------------------------- #
R = dict(
    start="2007-06-30", end="2026-06-30", n_obs=327, fp="01c183fa5741",
    n_funds=14, bonferroni=2.91,
    # fund: (bench, n, up, down, spread, ci_lo, ci_hi, p_pos, conv, t_conv)
    scorecard={
        "QYLD": ("QQQ", 150, 0.497, 0.559, -0.062, -0.21, 0.11, 0.23, -0.35, -2.62),
        "JEPQ": ("QQQ", 49, 0.712, 0.689, 0.023, -0.13, 0.24, 0.64, -0.44, -5.58),
        "NUSI": ("QQQ", 78, 0.630, 0.466, 0.164, -0.03, 0.39, 0.95, -0.20, -1.65),
        "XYLD": ("SPY", 156, 0.641, 0.711, -0.070, -0.22, 0.10, 0.20, -0.35, -2.14),
        "JEPI": ("SPY", 73, 0.592, 0.571, 0.021, -0.16, 0.20, 0.60, -0.14, -0.72),
        "PBP": ("SPY", 222, 0.554, 0.614, -0.060, -0.19, 0.09, 0.21, -0.37, -3.69),
        "DIVO": ("SPY", 114, 0.772, 0.729, 0.042, -0.10, 0.19, 0.73, 0.01, 0.07),
        "SPYI": ("SPY", 46, 0.745, 0.720, 0.026, -0.05, 0.16, 0.71, -0.36, -5.49),
        "SCHD": ("SPY", 176, 0.860, 0.839, 0.020, -0.18, 0.24, 0.56, -0.09, -0.69),
        "VYM": ("SPY", 229, 0.865, 0.873, -0.008, -0.14, 0.13, 0.46, -0.10, -1.23),
        "DVY": ("SPY", 229, 0.800, 0.819, -0.018, -0.24, 0.21, 0.46, -0.21, -1.39),
        "SPHD": ("SPY", 164, 0.713, 0.754, -0.041, -0.34, 0.27, 0.39, -0.28, -1.16),
        "NOBL": ("SPY", 152, 0.822, 0.894, -0.072, -0.28, 0.13, 0.24, -0.15, -0.85),
        "RYLD": ("IWM", 86, 0.565, 0.593, -0.028, -0.21, 0.20, 0.43, -0.53, -3.15),
    },
    n_pos_nominal=0, n_pos_bonferroni=0, n_neg_nominal=6,
    concave_funds=["QYLD", "JEPQ", "XYLD", "PBP", "SPYI", "RYLD"],
    spread_median=-0.013, spread_mean=-0.004, spread_min=-0.072, spread_max=0.164,
    ci_covers_zero=14,
    # CAPM alpha (excess-of-cash, HAC 6): fund -> (beta, alpha_bps, t_alpha, kink_bps)
    capm={
        "QYLD": (0.52, -18.6, -1.24, 51.6), "JEPQ": (0.65, 3.8, 0.22, 99.4),
        "NUSI": (0.54, 33.6, 1.43, 81.0), "XYLD": (0.68, -17.4, -1.17, 40.4),
        "JEPI": (0.56, -2.6, -0.21, 23.3), "PBP": (0.63, -18.8, -1.56, 48.8),
        "DIVO": (0.73, 5.7, 0.45, 3.9), "SPYI": (0.71, -1.6, -0.17, 58.9),
        "SCHD": (0.83, 3.8, 0.23, 18.4), "VYM": (0.87, -3.3, -0.28, 14.7),
        "DVY": (0.83, -7.1, -0.34, 30.1), "SPHD": (0.77, -13.9, -0.48, 31.1),
        "NOBL": (0.84, -12.4, -0.81, 12.7), "RYLD": (0.58, -16.5, -0.61, 120.1),
    },
    max_abs_t_alpha=1.56,
    # Sharpe race: fund -> (sharpe_fund, sharpe_bench, gap, t_gap)
    sharpe={
        "QYLD": (0.633, 0.984, -0.351, -3.99), "XYLD": (0.610, 0.896, -0.286, -3.94),
        "PBP": (0.391, 0.678, -0.287, -3.66), "SPYI": (0.976, 1.017, -0.041, -2.62),
        "JEPQ": (0.977, 0.992, -0.015, -2.20), "JEPI": (0.766, 0.941, -0.175, -2.13),
        "NOBL": (0.639, 0.862, -0.223, -1.87), "SPHD": (0.597, 0.944, -0.347, -1.60),
        "RYLD": (0.259, 0.451, -0.193, -1.56), "DIVO": (0.814, 0.847, -0.034, -1.46),
        "VYM": (0.567, 0.644, -0.077, -1.22), "DVY": (0.471, 0.644, -0.173, -1.16),
        "SCHD": (0.858, 0.966, -0.109, -0.96), "NUSI": (1.086, 0.920, 0.166, -1.28),
    },
    n_beating_bench=1,
    # Era cut (split 2021-01-01): fund -> (spread_early, spread_late)
    era={
        "QYLD": (-0.107, -0.016), "XYLD": (-0.193, 0.052), "PBP": (-0.112, 0.073),
        "DIVO": (0.017, 0.057), "SCHD": (0.025, 0.002), "VYM": (-0.037, 0.092),
        "DVY": (-0.025, 0.112), "SPHD": (-0.116, 0.034), "NOBL": (-0.024, -0.125),
    },
    rho=-0.117, rho_p=0.756, rho_n=9, sign_agreement=0.44,
    # Assumption sweeps
    spy_median=-0.003, spy_pos=0, spy_neg=5,
    geo_flip="13 of 14 turn negative under the geometric convention",
    geo_examples={"QYLD": (-0.062, -0.178), "RYLD": (-0.028, -0.171),
                  "DIVO": (0.042, -0.038)},
    # Traded beta-neutral spread (36m rolling beta through t, held t+1)
    best_traded="DVY", best_gross_bps=0.8, best_gross_t=0.04,
    borrow_sweep=[(0, 0.6, 0.03, 0.01), (50, -2.7, -0.13, -0.04),
                  (100, -6.0, -0.29, -0.08), (200, -12.7, -0.62, -0.18)],
    traded_examples={"QYLD": (114, -24.2, -1.36, -28.8, -1.62),
                     "XYLD": (120, -27.9, -1.87, -34.1, -2.28),
                     "PBP": (186, -18.9, -1.55, -24.0, -1.96),
                     "SCHD": (140, -4.1, -0.22, -11.6, -0.62),
                     "DVY": (193, 0.8, 0.04, -6.0, -0.29)},
    # Unadjusted corporate action found and repaired in the vendor tape
    split_ticker="NUSI", split_date="2025-02-18", split_ratio=2.000000,
    split_raw=dict(down=-0.309, spread=0.939, alpha=185.6, t_alpha=1.25, traded=331.5),
    split_fixed=dict(down=0.466, spread=0.164, alpha=33.6, t_alpha=1.43, traded=53.0),
    ci_covers_zero_raw=13,
    # Fund-matched null: fund -> (observed, null_beta, excess_beta, p, null_ab, excess_ab)
    matched={
        "QYLD": (-0.062, 0.031, -0.093, 0.886, -0.058, -0.004),
        "JEPQ": (0.023, 0.052, -0.029, 0.644, 0.069, -0.046),
        "NUSI": (0.164, 0.047, 0.117, 0.127, 0.185, -0.021),
        "XYLD": (-0.070, 0.023, -0.093, 0.897, -0.077, 0.006),
        "JEPI": (0.021, 0.054, -0.032, 0.640, 0.040, -0.019),
        "PBP": (-0.060, 0.024, -0.084, 0.919, -0.077, 0.017),
        "DIVO": (0.042, 0.028, 0.015, 0.429, 0.057, -0.015),
        "SPYI": (0.026, 0.063, -0.037, 0.745, 0.053, -0.028),
        "SCHD": (0.020, 0.011, 0.009, 0.447, 0.034, -0.013),
        "VYM": (-0.008, 0.009, -0.016, 0.614, -0.009, 0.002),
        "DVY": (-0.018, 0.011, -0.030, 0.619, -0.027, 0.009),
        "SPHD": (-0.041, 0.014, -0.054, 0.644, -0.067, 0.027),
        "NOBL": (-0.072, 0.008, -0.080, 0.783, -0.063, -0.009),
        "RYLD": (-0.028, 0.036, -0.063, 0.732, -0.027, -0.000),
    },
    matched_n_sig=0, matched_min_p=0.127, matched_median_excess=-0.035,
    matched_ab_median_err=0.014, matched_ab_max_err=0.046,
    # Best traded spread ignoring the 60-month history floor
    best_any="NUSI", best_any_bps=53.0, best_any_t=1.60, best_any_n=42,
    # The estimator's own null bias (synthetic, zero truth, UNCONDITIONAL)
    null_n=140, null_spread_mean=0.111, null_spread_median=0.081,
    null_spread_sd=0.120, null_frac_pos=0.82,
    null_conv_mean=-0.003, null_conv_sd=0.071,
    # Synthetic control
    syn_corr=0.912, syn_hits_pos=1, syn_hits_neg=3, syn_n_funds=7,
    syn_null_fires=3, syn_null_tests=56, syn_null_expected=2.8,
)

BOOT = (
    "import os, sys\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
    "from capture_ratio import data, strategy as st\n"
)

HEADER = f"""# Study 948 — Capture Ratio 📉

**Do income funds really "give up a little upside to avoid a lot of downside"?**

That sentence is on every covered-call and high-dividend fact sheet, and it is a claim you
can *measure*. Take a fund's monthly returns, split the months by whether its benchmark was
up or down, and ask what share of each it kept:

* **up-capture** — the share of the benchmark's good months the fund kept,
* **down-capture** — the share of the bad months it took,
* **capture spread** — up minus down. **Positive means the promise is true.**

We score **{R['n_funds']} US income ETFs** — nine option-writers and five dividend funds as
a no-options control — against the index each is marketed against (SPY / QQQ / IWM), on
monthly **total** returns (they pay most of their return out as distributions), cash leg
BIL, {R['start']} → {R['end']} ({R['n_obs']} month-ends), as-of 2026-06-30.

*Real numbers below are the frozen headline (`docs/results.md`, fingerprint
`{R['fp']}`). The live cells run the offline synthetic checks and say so.*
"""


def _scorecard_code():
    return (
        "R_SCORE = " + repr(R["scorecard"]) + "\n"
        "print(f\"{'fund':6}{'bench':6}{'n':>5}{'up':>8}{'down':>8}{'spread':>9}"
        "{'95% CI':>18}{'conv t':>9}\")\n"
        "for f, (b, n, up, dn, sp, lo, hi, p, cv, t) in R_SCORE.items():\n"
        "    ci = f'[{lo:+.2f}, {hi:+.2f}]'\n"
        "    print(f\"{f:6}{b:6}{n:5d}{up:8.3f}{dn:8.3f}{sp:+9.3f}{ci:>18}{t:+9.2f}\")\n"
    )


# --------------------------------------------------------------------------- #
# 01 — for the curious
# --------------------------------------------------------------------------- #
def build_curious():
    cells = [
        md(HEADER),
        md("## 1. The promise, and the one number that tests it\n\n"
           "A fund that keeps **70%** of the good months and only **40%** of the bad ones is "
           "doing something genuinely clever: it has bent the payoff in your favour. A fund "
           "that keeps 70% of the good months and 70% of the bad ones has done nothing you "
           "could not do yourself by holding 70% of the index and putting the rest in the "
           "bank.\n\n"
           "So the honest scorecard is not the two numbers. It is the **gap** between them."),
        md("## 2. What the tape says\n\n"
           "Fourteen funds. Up-capture, down-capture, the spread, and a 95% confidence "
           "interval on that spread. Read the last two columns."),
        code(_scorecard_code()),
        md(f"## 3. The two numbers are the same number\n\n"
           f"Look down the `up` and `down` columns: they track each other almost perfectly. "
           f"QYLD keeps 50% of the good months and takes 56% of the bad ones. SCHD: 86% and "
           f"84%. VYM: 87% and 87%. The median spread across the whole panel is "
           f"**{R['spread_median']:+.3f}** — that is, nothing.\n\n"
           f"And the confidence intervals swallow zero for **{R['ci_covers_zero']} of "
           f"{R['n_funds']}** funds. Not one fund has a spread you could bet on.\n\n"
           f"> 🔬 **For the quants** — the spread's regression twin is the Henriksson-Merton "
           f"convexity coefficient `b_up − b_dn`, which has an honest standard error. Zero of "
           f"fourteen funds reach a nominal |*t*| ≥ 2 on the positive side. Six reach it on "
           f"the **negative** side ({', '.join(R['concave_funds'])}) — every one of them an "
           f"option-writer, delivering exactly the concave payoff it sold."),
        md("## 4. The ruler is bent — it reads positive on nothing at all\n\n"
           "Here is the part that changes how you read every fund fact sheet you will ever "
           "see. We built synthetic funds that are, by construction, **plain fractions of "
           "the index** — no cleverness, no options, true capture spread exactly zero — and "
           "measured them with the industry's own formula.\n\n"
           "*(The cell below is synthetic by construction — it is a test of the measuring "
           "tape, not of any real fund.)*"),
        code(
            BOOT
            + "nb = st.null_spread_distribution(\n"
              "    data.synthetic_panel(signal_strength=0.0, seed=948 + s) for s in range(6))\n"
              "print(f\"{nb['n_obs']} synthetic funds, every one a plain fraction of the index\")\n"
              "print(f\"true capture spread : 0.000 for all of them\")\n"
              "print(f\"MEASURED spread     : mean {nb['spread_mean']:+.3f}, \"\n"
              "      f\"positive {nb['spread_frac_positive']:.0%} of the time\")\n"
              "print(f\"regression twin     : mean {nb['convexity_mean']:+.3f} (honest)\")\n"
        ),
        md(f"The capture spread is a *ratio of two averages*, and the bottom of the "
           f"down-capture ratio is a small negative number. Divide by something small and "
           f"noisy and the answer wanders — upwards, as it happens. On data with a true "
           f"spread of **exactly zero** the industry formula reads positive about "
           f"**{R['null_frac_pos']:.0%}** of the time, with an average of "
           f"**{R['null_spread_mean']:+.3f}**.\n\n"
           f"**But we are not allowed to just subtract that from the real table**, and it is "
           f"worth saying why, because it is the kind of shortcut that turns a true finding "
           f"into an overstated one. The {R['null_spread_mean']:+.3f} averages over *many "
           f"possible market histories*. The real funds were measured against **one** history "
           f"— the actual one — and once you hold that fixed, most of the wobble disappears.\n\n"
           f"So we redid it properly: rebuild each real fund a thousand times as a "
           f"**plain scaled index with no cleverness in it** — its own beta, its own noise, "
           f"the real market's real months — and ask how often that fake fund's spread beats "
           f"the real fund's. Answer: **{R['matched_n_sig']} of {R['n_funds']}** real funds "
           f"stand out (the closest call is *p* = {R['matched_min_p']:.3f}), and the typical "
           f"real fund reads **{R['matched_median_excess']:+.3f}** *below* its own dumb twin. "
           f"Smaller than the naive correction would have claimed — and still pointing the "
           f"same way."),
        md(f"## 5. NUSI — the one fund that looked convex, and the split that did it\n\n"
           f"Before any of the above, one number had to be thrown out. On "
           f"**{R['split_date']}** NUSI's price doubled in a single day — an exact "
           f"×{R['split_ratio']:.6f}, with QQQ unmoved. That is not a return, it is a "
           f"**1-for-2 reverse split** that the data vendor's 'adjusted' prices had failed to "
           f"back out, and it was worth a fake **+100% month**.\n\n"
           f"Left in, it made NUSI look like the one fund that works: down-capture "
           f"**{R['split_raw']['down']:+.3f}** (negative!), spread "
           f"**{R['split_raw']['spread']:+.3f}**, alpha **{R['split_raw']['alpha']:+.1f} "
           f"bps/month**. Repaired: down-capture **{R['split_fixed']['down']:+.3f}**, spread "
           f"**{R['split_fixed']['spread']:+.3f}**, alpha "
           f"**{R['split_fixed']['alpha']:+.1f} bps/month**. It also moved the panel headline "
           f"— the confidence interval covers zero for {R['ci_covers_zero']}/{R['n_funds']} "
           f"funds now, rather than {R['ci_covers_zero_raw']}/{R['n_funds']}.\n\n"
           f"NUSI still has the panel's largest spread at "
           f"**{R['scorecard']['NUSI'][4]:+.3f}**, and it still is not evidence: the interval "
           f"[{R['scorecard']['NUSI'][5]:+.2f}, {R['scorecard']['NUSI'][6]:+.2f}] covers zero, "
           f"and its regression twin is **{R['scorecard']['NUSI'][8]:+.2f}** — *concave*, like "
           f"every other option-writer."),
        md(f"## 6. Does last decade's scorecard predict the next one?\n\n"
           f"No. Split the sample at 2021 and rank the nine funds with enough history on "
           f"both sides: the rank correlation between the two eras is "
           f"**{R['rho']:+.3f}** (*p* = {R['rho_p']:.2f}) and the spread keeps the *same sign* "
           f"in only **{R['sign_agreement']:.0%}** of cases — worse than a coin. Five of the nine "
           f"funds flip outright. Whatever a capture spread describes, it is a sample, not "
           f"a fund."),
        code(
            "R_ERA = " + repr(R["era"]) + "\n"
            "RHO, PVAL, NOBS = " + repr((R["rho"], R["rho_p"], R["rho_n"])) + "\n"
            "print(f\"{'fund':6}{'2007-2020':>12}{'2021-2026':>12}   flip?\")\n"
            "for f, (a, b) in R_ERA.items():\n"
            "    print(f\"{f:6}{a:+12.3f}{b:+12.3f}   {'yes' if a * b < 0 else 'no'}\")\n"
            "print(f\"\\nSpearman rho between the two eras: {RHO:+.3f} \"\n"
            "      f\"(p = {PVAL:.2f}, n = {NOBS})\")\n"
        ),
        md(f"## 7. So what *are* you buying?\n\n"
           f"A **smaller position in the index, plus a distribution schedule**. Every fund's "
           f"beta sits between **0.52 and 0.87**, and not one of them has measurable alpha — "
           f"the largest |*t*| on alpha across all fourteen is **{R['max_abs_t_alpha']:.2f}**, "
           f"and it is *negative*. **{R['n_beating_bench']} of {R['n_funds']}** beats its own "
           f"benchmark on risk-adjusted return after subtracting cash — NUSI, and it wins on "
           f"lower volatility, not higher return (its mean monthly excess return is *below* "
           f"QQQ's).\n\n"
           f"The reduced drawdowns are real — eleven of fourteen funds cut the worst loss. "
           f"But so does owning less of the index, and that costs 0.03% a year instead of "
           f"0.35–0.68%."),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** Not one of the fourteen funds shows a positive capture "
           f"spread at even a nominal |*t*| ≥ 2, never mind the **{R['bonferroni']:.2f}** bar "
           f"that fourteen simultaneous tests demand. The interval covers zero for "
           f"{R['ci_covers_zero']}/{R['n_funds']}, there is no rank persistence across eras "
           f"(rho = {R['rho']:+.3f}), and against its own zero-cleverness twin not one fund "
           f"stands out (smallest *p* = {R['matched_min_p']:.3f}). The only robust finding "
           f"runs the other way — six "
           f"option-writers are significantly **concave**, which is what selling calls does "
           f"by definition. Survivorship flatters the table: the funds that closed are gone "
           f"from the tape.\n"
           f"- **Tradability — Mirage.** Nothing to buy and nothing to short. The "
           f"beta-neutral spread that would monetise the scorecard earns "
           f"**{R['best_gross_bps']:+.1f} bps/month gross** at its very best "
           f"(*t* = {R['best_gross_t']:+.2f}) and is negative once the short leg pays 50 bps "
           f"of borrow.\n"
           f"- **The honest reading.** These products are fairly priced beta reduction with a "
           f"cash-flow calendar attached. That is a legitimate thing to want. It is simply "
           f"not the asymmetry the sentence on the fact sheet describes."),
    ]
    nb = new_notebook()
    nb["cells"] = cells
    return nb


# --------------------------------------------------------------------------- #
# 02 — for the quants
# --------------------------------------------------------------------------- #
def build_quants():
    cells = [
        md(f"# Study 948 — Capture Ratio — the teardown\n\n"
           f"The capture spread as an estimand: a joint block-bootstrap CI, the "
           f"Henriksson-Merton convexity twin with a HAC *t* judged against a Bonferroni bar, "
           f"CAPM alphas, the excess-of-cash Sharpe race, an era cut with rank persistence, "
           f"the assumption sweeps, the beta-neutral traded arm with cost × borrow, and the "
           f"live synthetic control.\n\n"
           f"**Design.** Monthly simple **total** returns (`auto_adjust=True`), months "
           f"partitioned by the *benchmark's* sign, cash leg BIL, per-fund windows from each "
           f"fund's own inception, {R['start']} → {R['end']}. **One execution lag, and only "
           f"in the traded arm**: the 36-month rolling beta is estimated through month *t* "
           f"and held over *t*+1. Capture ratios are a contemporaneous attribution — nothing "
           f"is traded on them, so no lag applies. Costs are one-way bps × NAV; the short "
           f"index leg pays borrow; both are swept.\n\n"
           f"Every real number is frozen from `docs/results.md` (fingerprint `{R['fp']}`)."),
        code("R = %r" % ({k: v for k, v in R.items()
                          if k in ("start", "end", "n_obs", "fp", "n_funds", "bonferroni",
                                   "spread_median", "ci_covers_zero", "max_abs_t_alpha",
                                   "rho", "rho_p", "rho_n", "sign_agreement",
                                   "n_beating_bench", "null_spread_mean", "null_frac_pos",
                                   "matched_n_sig", "matched_min_p",
                                   "matched_median_excess")},)),
        md(f"## 0. Data hygiene first — one unadjusted corporate action\n\n"
           f"`data.split_artifacts` sweeps all eighteen daily series for one-day price ratios "
           f"that land on a real share-count factor (3:2, 2:1, 3:1, 4:1, 5:1, 10:1 and their "
           f"reverse twins, 3% tolerance, ±35% jump threshold). The whitelist is deliberately "
           f"short: a dense list of every *n/m* would tile the number line and smooth a "
           f"genuine crash away. It fires **exactly once**."),
        code(
            "R_SPLIT = " + repr({"ticker": R["split_ticker"], "date": R["split_date"],
                                 "ratio": R["split_ratio"], "raw": R["split_raw"],
                                 "fixed": R["split_fixed"],
                                 "ci0_raw": R["ci_covers_zero_raw"],
                                 "ci0_fixed": R["ci_covers_zero"]}) + "\n"
            "print(f\"{R_SPLIT['ticker']} {R_SPLIT['date']}: one-day ratio \"\n"
            "      f\"{R_SPLIT['ratio']:.6f} -> read as a 1-for-2 reverse split, back-adjusted\")\n"
            "print()\n"
            "print(f\"{'':10}{'down-cap':>10}{'spread':>9}{'alpha bps':>11}{'t':>7}{'traded bps':>12}\")\n"
            "for tag in ('raw', 'fixed'):\n"
            "    d = R_SPLIT[tag]\n"
            "    print(f\"{tag:10}{d['down']:+10.3f}{d['spread']:+9.3f}\"\n"
            "          f\"{d['alpha']:+11.1f}{d['t_alpha']:+7.2f}{d['traded']:+12.1f}\")\n"
            "print()\n"
            "print(f\"panel CI-covers-zero count: {R_SPLIT['ci0_raw']} raw -> \"\n"
            "      f\"{R_SPLIT['ci0_fixed']} repaired (of 14)\")\n"
        ),
        md("`auto_adjust=True` is supposed to handle this and did not. Left in, the fake "
           "+100% month made NUSI the panel's only fund with a CI excluding zero — i.e. it "
           "would have been the study's one 'result'. Everything below runs on the repaired "
           "tape; `data.load_prices(repair_splits=False)` reproduces the contaminated one."),
        md("## 1. The scorecard\n\n"
           "`spread` = up-capture − down-capture (arithmetic, ratio of means). CI = 95% "
           "circular block bootstrap, 2,000 draws, 3-month blocks, fund and benchmark "
           "resampled **jointly**. `convexity t` = HAC (Newey-West, 6 lags) *t* on "
           "`b_up − b_dn` from the piecewise fit on excess-of-cash returns."),
        code(_scorecard_code()),
        code(
            "N_POS, N_BONF, N_NEG = " + repr((R["n_pos_nominal"], R["n_pos_bonferroni"],
                                              R["n_neg_nominal"])) + "\n"
            "CONCAVE = " + repr(R["concave_funds"]) + "\n"
            "print(f\"Bonferroni bar for {R['n_funds']} simultaneous tests: \"\n"
            "      f\"|t| >= {R['bonferroni']:.2f}\")\n"
            "print(f\"funds with convexity t >= +2 (nominal) : {N_POS}\")\n"
            "print(f\"funds clearing the Bonferroni bar      : {N_BONF}\")\n"
            "print(f\"funds with convexity t <= -2 (CONCAVE) : {N_NEG}  -> {CONCAVE}\")\n"
            "print(f\"bootstrap CI covers zero for {R['ci_covers_zero']}/{R['n_funds']} funds\")\n"
            "print(f\"cross-sectional spread: median {R['spread_median']:+.3f}\")\n"
        ),
        md("> 💡 **In plain words** — nothing in the table is convex. The only funds whose "
           "payoff shape is statistically distinguishable from a straight line are bent the "
           "*wrong* way, and they are exactly the funds that write options."),
        md(f"## 2. The estimator's null bias — why the ratio cannot be read at face value\n\n"
           f"The capture spread is a difference of two ratios of sample means whose down-leg "
           f"denominator is a small negative number. Ratio estimators of that shape are "
           f"biased in finite samples. We measure it directly on zero-truth synthetic panels "
           f"(**synthetic by construction** — this is a property of the estimator, not of any "
           f"fund)."),
        code(
            BOOT
            + "nb = st.null_spread_distribution(\n"
              "    data.synthetic_panel(signal_strength=0.0, seed=948 + s) for s in range(8))\n"
              "print(f\"{nb['n_obs']} fund-panels, planted true capture spread = 0.000 for every one\")\n"
              "print(f\"arithmetic spread : mean {nb['spread_mean']:+.3f}  median {nb['spread_median']:+.3f}\"\n"
              "      f\"  sd {nb['spread_sd']:.3f}  positive {nb['spread_frac_positive']:.0%}\")\n"
              "print(f\"HM convexity      : mean {nb['convexity_mean']:+.3f}  \"\n"
              "      f\"median {nb['convexity_median']:+.3f}  sd {nb['convexity_sd']:.3f}  (unbiased)\")\n"
        ),
        md(f"On the published run this is **mean {R['null_spread_mean']:+.3f}, positive "
           f"{R['null_frac_pos']:.0%} of the time** for the ratio versus "
           f"**{R['null_conv_mean']:+.3f}** for the regression twin.\n\n"
           f"**This number must not be subtracted from the real panel.** It is an "
           f"*unconditional* bias — it averages over random benchmark paths. A real capture "
           f"spread is computed on **one** realised benchmark path, and conditioning on that "
           f"path removes most of it. Quoting {R['null_spread_mean']:+.3f} against the real "
           f"median would be mixing tapes and would overstate the case roughly threefold. The "
           f"real-tape version is section 2b."),
        md(f"## 2b. The fund-matched null — the bias measured on each fund's own sample\n\n"
           f"`strategy.matched_null_spread` rebuilds each fund 1,000 times as a "
           f"**zero-convexity twin**: the fund's own beta and residual vol (circular block "
           f"bootstrap of the OLS residuals) on the **real** benchmark and cash paths, "
           f"re-scored with the same estimator. Two variants:\n\n"
           f"* **null(β)** — zero alpha too, i.e. a plain scaled index position. This is the "
           f"test, and it has power: on the planted synthetic panel the genuinely convex fund "
           f"clears it at *p* = 0.000 and the concave ones sit at the far tail.\n"
           f"* **null(α,β)** — keeps the fund's own fitted alpha. A **decomposition, not a "
           f"test**.\n\n"
           f"**Stated because it cuts against us:** null(β) also fires on a perfectly *linear* "
           f"fund carrying a positive alpha, since the capture spread of a linear fund is "
           f"`a·(1/mean_up − 1/mean_down)` and `mean_down < 0`, so the two terms add. A small "
           f"*p* there means 'beats a scaled index', **not** 'is convex'. Convexity as such is "
           f"the HM column in section 1, whose free intercept separates the two."),
        code(
            "R_MATCH = " + repr(R["matched"]) + "\n"
            "M = " + repr((R["matched_n_sig"], R["matched_min_p"], R["matched_median_excess"],
                           R["matched_ab_median_err"], R["matched_ab_max_err"])) + "\n"
            "print(f\"{'fund':6}{'observed':>10}{'null(b)':>9}{'excess':>9}{'p':>7}\"\n"
            "      f\"{'null(a,b)':>11}{'excess':>9}\")\n"
            "for f, (o, nb_, eb, p, nab, eab) in R_MATCH.items():\n"
            "    print(f\"{f:6}{o:+10.3f}{nb_:+9.3f}{eb:+9.3f}{p:7.3f}{nab:+11.3f}{eab:+9.3f}\")\n"
            "print()\n"
            "print(f\"funds reaching p < 0.05 vs a plain scaled index: {M[0]}/14 \"\n"
            "      f\"(smallest p = {M[1]:.3f})\")\n"
            "print(f\"median excess spread vs that twin             : {M[2]:+.3f}\")\n"
            "print(f\"null(a,b) reproduces the observed spread to a median |error| of \"\n"
            "      f\"{M[3]:.3f} (max {M[4]:.3f})\")\n"
        ),
        md(f"Two readings, and the second is the sharper one:\n\n"
           f"1. **{R['matched_n_sig']} of {R['n_funds']}** funds' spreads reach *p* < 0.05 "
           f"against a plain scaled index (smallest {R['matched_min_p']:.3f}); the median "
           f"excess is **{R['matched_median_excess']:+.3f}** — real, and about a third of what "
           f"the unconditional bias would have implied.\n"
           f"2. The alpha-carrying twin reproduces every observed spread to a median error of "
           f"**{R['matched_ab_median_err']:.3f}** (max {R['matched_ab_max_err']:.3f}). The "
           f"capture spread carries **no information beyond (alpha, beta)**. It is a "
           f"re-encoding of the CAPM fit, not an independent measurement of payoff shape — "
           f"which is the strongest form of this study's claim.\n\n"
           f"> 💡 **In plain words** — the industry's ruler reads long, and once you correct "
           f"it against each fund's own history it turns out to be measuring the fund's alpha "
           f"and beta all over again, in worse units."),
        md("## 3. Is it just beta? CAPM alpha, and the kink intercept\n\n"
           "Single-beta market model on excess-of-cash monthly returns, HAC(6). The **kink "
           "intercept** is the piecewise fit's intercept — the fund's excess return in a flat "
           "benchmark month, i.e. the premium an option-writer collects. It is *not* alpha: a "
           "kinked fit lifts the intercept mechanically and the same fit hands it back "
           "through a truncated up-beta."),
        code(
            "R_CAPM = " + repr(R["capm"]) + "\n"
            "print(f\"{'fund':6}{'beta':>7}{'alpha bps/mo':>15}{'t':>8}{'kink bps/mo':>14}\")\n"
            "for f, (b, a, t, k) in R_CAPM.items():\n"
            "    print(f\"{f:6}{b:7.2f}{a:+15.1f}{t:+8.2f}{k:+14.1f}\")\n"
            "print(f\"\\nmax |t| on alpha across the panel: {R['max_abs_t_alpha']:.2f}\"\n"
            "      f\"  -> no fund adds anything beyond a smaller index position\")\n"
        ),
        md("## 4. Excess-of-cash Sharpe race (both sides excess of BIL)\n\n"
           "`gap` = fund Sharpe − benchmark Sharpe; the *t* is HAC(6) on the **monthly return "
           "difference**, which is the Jobson-Korkie comparison in its Newey-West form (the "
           "cash leg cancels in the difference)."),
        code(
            "R_SH = " + repr(R["sharpe"]) + "\n"
            "print(f\"{'fund':6}{'fund Sh':>10}{'bench Sh':>10}{'gap':>9}{'HAC t':>8}\")\n"
            "for f, (sf, sb, g, t) in R_SH.items():\n"
            "    print(f\"{f:6}{sf:+10.3f}{sb:+10.3f}{g:+9.3f}{t:+8.2f}\")\n"
            "print(f\"\\nfunds beating their own benchmark: \"\n"
            "      f\"{R['n_beating_bench']}/{R['n_funds']}\")\n"
        ),
        md("## 5. Robustness — era cut, rank persistence, and the assumption sweeps"),
        code(
            "R_ERA = " + repr(R["era"]) + "\n"
            "RHO, PVAL, NOBS, AGREE = " + repr((R["rho"], R["rho_p"], R["rho_n"],
                                                R["sign_agreement"])) + "\n"
            "print('era cut, split 2021-01-01 (>= 24 months each side):')\n"
            "for f, (a, b) in R_ERA.items():\n"
            "    flag = 'SIGN FLIP' if a * b < 0 else ''\n"
            "    print(f\"  {f:6} early {a:+.3f}   late {b:+.3f}   {flag}\")\n"
            "print(f\"\\nSpearman rho = {RHO:+.3f} (p = {PVAL:.3f}, n = {NOBS}), \"\n"
            "      f\"sign agreement {AGREE:.0%}\")\n"
        ),
        code(
            "SPY_MED, SPY_POS, SPY_NEG = " + repr((R["spy_median"], R["spy_pos"],
                                                   R["spy_neg"])) + "\n"
            "GEO = " + repr(R["geo_examples"]) + "\n"
            "print(\"ASSUMPTION 1 - the fund -> benchmark map (hand-assigned from each\")\n"
            "print(\"               fund's stated index). Sweep: force SPY for every fund.\")\n"
            "print(f\"  median spread {SPY_MED:+.3f}  |  t_convexity >= +2: {SPY_POS}\"\n"
            "      f\"  |  <= -2: {SPY_NEG}\")\n"
            "print()\n"
            "print('ASSUMPTION 2 - the capture convention. Sweep: arithmetic vs geometric.')\n"
            "print('  " + R["geo_flip"] + "')\n"
            "for f, (a, g) in GEO.items():\n"
            "    print(f\"    {f:6} arithmetic {a:+.3f} -> geometric {g:+.3f}\")\n"
        ),
        md("Neither assumption moves the verdict: forcing SPY on everyone still leaves "
           "**zero** funds with a positive convexity *t*, and the geometric convention pushes "
           "**every** fund's spread down."),
        md("## 6. Tradability — the beta-neutral traded spread\n\n"
           "Long the fund, short `beta_hat` of the benchmark, remainder in cash. `beta_hat` = "
           "36-month rolling OLS slope **through month *t***, held over ***t*+1** (the one "
           "execution lag). Cost 5 bps one-way × NAV on the notional turned over; the short "
           "index leg pays borrow."),
        code(
            "R_TR = " + repr(R["traded_examples"]) + "\n"
            "BEST, BEST_BPS, BEST_T = " + repr((R["best_traded"], R["best_gross_bps"],
                                                R["best_gross_t"])) + "\n"
            "BORROW = " + repr(R["borrow_sweep"]) + "\n"
            "print(f\"{'fund':6}{'n':>5}{'gross bps/mo':>15}{'t':>8}\"\n"
            "      f\"{'net(5bps,100bp)':>18}{'t':>8}\")\n"
            "for f, (n, g, tg, nt, tn) in R_TR.items():\n"
            "    print(f\"{f:6}{n:5d}{g:+15.1f}{tg:+8.2f}{nt:+18.1f}{tn:+8.2f}\")\n"
            "ANY = " + repr((R["best_any"], R["best_any_bps"], R["best_any_t"],
                             R["best_any_n"])) + "\n"
            "print(f\"\\nlargest gross spread in the panel, no history floor: {ANY[0]} at \"\n"
            "      f\"{ANY[1]:+.1f} bps/mo on {ANY[3]} months (t = {ANY[2]:+.2f})\")\n"
            "print(f\"best spread in the panel with >= 60 months: {BEST} at \"\n"
            "      f\"{BEST_BPS:+.1f} bps/mo gross (t = {BEST_T:+.2f})\")\n"
            "print('\\nborrow sweep on that best case:')\n"
            "for b, m, t, s in BORROW:\n"
            "    print(f\"  borrow {b:4d} bps -> {m:+6.1f} bps/mo \"\n"
            "          f\"(t = {t:+.2f}, Sharpe {s:+.2f})\")\n"
        ),
        md("> 💡 **In plain words** — the best beta-neutral version of this trade with a real "
           "history makes less than a basis point a month before the short leg is paid for, "
           "and loses money once it is. The one that looks better (NUSI, 42 months) is short, "
           "not significant, and belongs to the fund whose tape needed a split repaired."),
        md("## 7. Live synthetic control — the machinery is unbiased\n\n"
           "**Synthetic by construction.** A planted cross-section (one genuinely convex fund, "
           "three genuinely concave, three flat) must be recovered; a null cross-section "
           "(every fund plain linear beta) must stay quiet at roughly the nominal rate. This "
           "proves the real-tape zero is a property of the funds, not of the harness — it "
           "never supports the stamp."),
        code(
            BOOT
            + "planted, tp = data.synthetic_panel(signal_strength=1.0, seed=948)\n"
              "dp = st.synthetic_detect(planted, tp, n_boot=200)\n"
              "print(f\"planted: corr(measured spread, planted spread) = {dp['corr_measured_true']:+.3f}\")\n"
              "print(f\"         {dp['n_hits_positive']} positive / {dp['n_hits_negative']} negative \"\n"
              "      f\"convexity hits of {dp['n_funds']} funds (planted: 1 convex, 3 concave)\")\n"
              "tbl = dp['table'][['true_spread', 'spread', 'convexity', 't_convexity']]\n"
              "print(tbl.round(3).to_string())\n"
              "fires = 0\n"
              "for s in range(6):\n"
              "    z, tz = data.synthetic_panel(signal_strength=0.0, seed=948 + s)\n"
              "    d = st.synthetic_detect(z, tz, n_boot=1)\n"
              "    fires += d['n_hits_positive'] + d['n_hits_negative']\n"
              "print(f\"\\nnull x6 seeds: |t| >= 2 fires on {fires}/42 fund-tests \"\n"
              "      f\"(nominal 5% predicts ~2.1)\")\n"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The estimand is the capture spread, and it is zero. "
           f"**{R['n_pos_nominal']} of {R['n_funds']}** funds reach a nominal positive "
           f"|*t*| ≥ 2 on the convexity coefficient; **{R['n_pos_bonferroni']}** clear the "
           f"family-wise bar of {R['bonferroni']:.2f}; the bootstrap CI covers zero for "
           f"{R['ci_covers_zero']}/{R['n_funds']}; against each fund's own zero-convexity "
           f"twin, on its own sample and the real benchmark path, {R['matched_n_sig']}/"
           f"{R['n_funds']} reach *p* < 0.05 (smallest {R['matched_min_p']:.3f}) and the "
           f"median excess spread is {R['matched_median_excess']:+.3f}; and the ranking has no "
           f"era-to-era persistence (rho = {R['rho']:+.3f}, *p* = {R['rho_p']:.2f}). The "
           f"largest positive (NUSI, {R['scorecard']['NUSI'][4]:+.3f}) needed an unadjusted "
           f"reverse split removed before it stopped reading "
           f"{R['split_raw']['spread']:+.3f}, and still carries a *negative* convexity "
           f"coefficient. What is robust is the opposite sign: {R['n_neg_nominal']} "
           f"significantly concave funds, all option-writers — an accounting identity of "
           f"selling calls. **Survivorship** is named here: the panel is the funds still "
           f"listed at the as-of, so the closed ones flatter it.\n"
           f"- **Tradability — Mirage.** Beta 0.52–0.87, no alpha anywhere "
           f"(max |*t*| = {R['max_abs_t_alpha']:.2f}, and negative), "
           f"{R['n_beating_bench']}/{R['n_funds']} beating its benchmark on excess-of-cash "
           f"Sharpe — NUSI, on lower vol, with a HAC *t* of −1.28 on the monthly return "
           f"difference. The beta-neutral arm tops out at {R['best_any_bps']:+.1f} bps/month "
           f"gross on {R['best_any_n']} months (*t* = {R['best_any_t']:+.2f}) and, among funds "
           f"with a real history, at {R['best_gross_bps']:+.1f} bps/month "
           f"(*t* = {R['best_gross_t']:+.2f}) — which dies at 50 bps of borrow. The drawdown "
           f"reduction is real and is bought with beta, which is available for 3 bps.\n"
           f"- **Methodological note worth keeping.** The arithmetic capture spread is a "
           f"**biased** estimator — positive {R['null_frac_pos']:.0%} of the time on a truth "
           f"of exactly zero — and once that bias is measured against each fund's own sample "
           f"it turns out the spread reproduces the fund's (alpha, beta) fit to within "
           f"{R['matched_ab_median_err']:.3f}. Any fund scorecard quoting up- and "
           f"down-capture without an interval is reporting a CAPM regression in worse units."),
    ]
    nb = new_notebook()
    nb["cells"] = cells
    return nb


def main() -> None:
    for name, nb in (("01_for_the_curious.ipynb", build_curious()),
                     ("02_for_the_quants.ipynb", build_quants())):
        path = os.path.join(HERE, name)
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print(f"wrote {path} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
