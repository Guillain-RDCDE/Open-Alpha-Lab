"""Generate the two narrative notebooks for Study 914 (Securities-Lending Offset).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below (a mirror of docs/results.md); the only live cells run the fast
offline synthetic control, and they are never presented under a real-tape banner.
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
# Monthly log total returns, auto_adjust=True, as-of 2026-06-30.
# --------------------------------------------------------------------------- #
R = dict(
    fp="d080dc025da8", asof="2026-06-30", n_days_frame=8411,
    # per-pair: (window, n, drift_bp, delta_er_bp, resid_bp, t, ci_lo, ci_hi, te_pct, floor_bp, beta)
    pairs=[
        ("SPY", "IVV", "US large cap", "2000-06", "2026-06", 313,
         -1.4, 6.5, 5.1, 1.04, -5.3, 15.1, 0.52, 20, 1.002),
        ("EEM", "IEMG", "Emerging markets", "2012-11", "2026-06", 164,
         -58.0, 61.0, 3.0, 0.11, -50.1, 56.2, 1.24, 67, 1.015),
        ("EFA", "IEFA", "Developed ex-US", "2012-11", "2026-06", 164,
         -21.4, 26.0, 4.6, 0.24, -33.6, 43.4, 0.82, 44, 0.993),
        ("VEA", "IEFA", "Developed ex-US (cross-sponsor)", "2012-11", "2026-06", 164,
         52.7, -4.0, 48.7, 1.14, -25.1, 137.0, 1.26, 68, 1.022),
        ("IWM", "IJR", "US small cap", "2000-06", "2026-06", 313,
         -127.3, 13.0, -114.3, -1.50, -264.3, 30.0, 3.75, 147, 1.013),
    ],
    pooled_resid=3.7, pooled_t=0.31, pooled_lo=-20.0, pooled_hi=27.2,
    pooled_te=0.55, pooled_floor=30, pooled_n=164,
    pooled_early=-0.6, pooled_early_t=-0.05,
    pooled_all=2.1, pooled_all_t=0.08,
    spy_bound=5.3, spy_ci_bound=15.1,
    era=[("SPY-IVV", 5.6, 0.90, 4.1, 0.64),
         ("EEM-IEMG", 14.9, 0.58, -6.8, -0.13),
         ("EFA-IEFA", -23.3, -0.89, 36.7, 1.34),
         ("VEA-IEFA", -16.2, -0.67, 125.8, 1.55),
         ("IWM-IJR", -159.3, -2.06, 6.5, 0.03)],
    er_sweep=[(0.68, 59.0, 1.0, 0.04), (0.70, 61.0, 3.0, 0.11),
              (0.72, 63.0, 5.0, 0.18), (0.75, 66.0, 8.0, 0.29)],
    # The decisive sweep: the CHEAP leg's fee, which every sponsor CUT during the sample.
    # (pair, cheap ticker, [(label, ER_b, dER_bp, resid_bp, t), ...])
    er_sweep_cheap=[
        ("SPY-IVV", "IVV", [("earliest", 0.0945, 0.0, -1.4, -0.28),
                            ("mid", 0.0650, 2.9, 1.6, 0.32),
                            ("today", 0.0300, 6.5, 5.1, 1.04)]),
        ("EEM-IEMG", "IEMG", [("earliest", 0.1800, 52.0, -6.0, -0.22),
                              ("mid", 0.1300, 57.0, -1.0, -0.03),
                              ("today", 0.0900, 61.0, 3.0, 0.11)]),
        ("EFA-IEFA", "IEFA", [("earliest", 0.1400, 19.0, -2.4, -0.13),
                              ("mid", 0.0900, 24.0, 2.6, 0.13),
                              ("today", 0.0700, 26.0, 4.6, 0.24)]),
    ],
    # Whole table re-run under three labelled fee schedules (both legs moved).
    fee_scenarios=[
        ("today (headline)", [5.1, 3.0, 4.6, 48.7, -114.3]),
        ("approx time-weighted", [2.1, 0.0, 3.1, 49.7, -122.8]),
        ("earliest in-sample", [1.2, -1.0, -0.4, 50.7, -127.3]),
    ],
    # True drift-rebalancing turnover vs the friction actually charged (bp/yr).
    turnover=[("IEMG/EEM", 3.36, 0.10), ("IEFA/EFA", 2.28, 0.07),
              ("IVV/SPY", 1.00, 0.03), ("IJR/IWM", 9.49, 0.28)],
    charged_bp=6.0,
    daily=[("SPY-IVV", 4.0, 0.29, 2.20), ("EEM-IEMG", 4.0, 0.12, 1.75),
           ("EFA-IEFA", 4.0, 0.18, 1.32), ("VEA-IEFA", 47.7, 1.33, 1.83),
           ("IWM-IJR", -114.3, -1.49, 5.13)],
    trade=[("IEMG/EEM", 47.0, 0.38, 1.69, 22.0, -3.0, -53.0, 1.24),
           ("IEFA/EFA", 16.5, 0.20, 0.85, -8.5, -33.5, -83.5, 0.82),
           ("IVV/SPY", -5.2, -0.10, -1.02, -30.2, -55.2, -105.2, 0.53),
           ("IJR/IWM", 110.4, 0.29, 1.45, 85.4, 60.4, 10.4, 3.78)],
    ex_sharpe=[("EEM", 0.26), ("IEMG", 0.38), ("IWM", 0.43),
               ("IJR", 0.47), ("SPY", 0.64), ("IVV", 0.64)],
    syn_planted=60.0, syn_recovered=63.1, syn_t=4.89,
    syn_null_mean=2.7, syn_null_sd=13.9, syn_null_fire=0,
    syn_pool=64.7, syn_pool_t=10.37,
)


HEADER = f"""# Study 914 — Securities-Lending Offset 📚

**Index funds lend their shares to short sellers. Does the money come back to you?**

The pitch is seductive and, in the sponsors' own reports, true: an index fund lends the
shares it holds, collects a fee, and credits most of it back to the fund. In hard-to-borrow
corners — small caps, emerging markets — that revenue is supposed to offset a real slice of
the expense ratio, so the fund's **tracking difference** should land *better* than `−ER`.
"Our EM fund is effectively free," goes the brochure version.

We test it on **five same-asset-class ETF pairs** — SPY/IVV, EEM/IEMG, EFA/IEFA, VEA/IEFA,
IWM/IJR — using daily **total-return** closes aggregated to monthly, as-of {R['asof']}.

*Real numbers below are the frozen headline (`docs/results.md`, Fingerprint `{R['fp']}`).
The live cells run the offline synthetic control only.*
"""


def _pair_rows_code():
    return (
        "pairs = %r\n"
        "hdr = 'pair            class                     resid    t      95%% CI          TE    floor'\n"
        "print(hdr); print('-' * len(hdr))\n"
        "for a, b, k, s, e, n, dr, der, res, t, lo, hi, te, fl, beta in pairs:\n"
        "    print('%%-4s-%%-5s %%-25s %%+7.1f %%+6.2f  [%%+7.1f,%%+7.1f] %%5.2f%%%% %%4dbp'\n"
        "          %% (a, b, k[:25], res, t, lo, hi, te, fl))"
        % (R["pairs"],)
    )


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea, and the one number that decides it\n\n"
           "Two funds hold nearly the same basket. One charges more. If nothing else were "
           "going on, the dearer fund should trail the cheaper one by *exactly* the fee "
           "difference. So take the realised gap and add the fee gap back:\n\n"
           "> **residual = realised return gap + fee gap**\n\n"
           "If that residual is **zero**, fees explained everything. If it is **positive** "
           "for the dearer fund, that fund handed back something extra — which is what "
           "lending revenue is supposed to look like from the outside.\n\n"
           "> 🔬 *For the quants:* the residual is measured on month-end log total returns "
           "(daily differencing between two funds is dominated by dividend-date mismatch), "
           "with an HAC *t* and a 6-month-block bootstrap CI. Expense ratios are a labelled "
           "**assumption**, not tape, and are swept."),
        code(_pair_rows_code()),
        md(f"## 2. The tight pairs say: fees explain everything\n\n"
           f"For the three pairs that genuinely track the same thing, the residual is "
           f"**+3.0, +4.6 and +5.1 basis points a year** — with *t*-statistics of 0.11, 0.24 "
           f"and 1.04. That is nothing. EEM trailed IEMG by 58 bp/yr; its fee disadvantage is "
           f"61 bp/yr. The gap *is* the fee, to within three basis points.\n\n"
           f"Pool the three and you get **{R['pooled_resid']:+.1f} bp/yr "
           f"(*t* = {R['pooled_t']:+.2f})**, CI [{R['pooled_lo']:+.1f}, {R['pooled_hi']:+.1f}]."),
        md("## 3. The clean experiment nobody set up on purpose\n\n"
           "**SPY cannot lend its shares.** It was built in 1993 as a unit investment trust, "
           "and that structure forbids securities lending. **IVV can, and does.** Same index, "
           "same 500 stocks, tightest tracking on the desk.\n\n"
           "If lending revenue reached shareholders in any visible way, IVV should beat SPY "
           "by *more* than the fee gap. Over 26 years **SPY trailed IVV by 1.4 bp/yr** — "
           "against a fee gap that is somewhere between 0 and 6.5 bp/yr depending on which "
           "year's fee schedule you use (IVV's fee fell from 0.0945% to 0.03% over exactly "
           "this window; see §4). The two funds ran neck and neck.\n\n"
           "That is a **bound**, not a measurement: any IVV lending advantage visible in "
           "returns is within a few basis points a year, and the bootstrap CI caps it at "
           f"**≈{R['spy_ci_bound']:.0f} bp/yr**. Which way the small leftover points is "
           "decided by the fee assumption, not by the tape."),
        code(
            "spy = dict(zip(['a', 'b', 'klass', 'start', 'end', 'n', 'drift', 'dER',\n"
            "                'resid', 't', 'ci_lo', 'ci_hi', 'te', 'floor', 'beta'], {spy!r}))\n"
            "print('SPY (cannot lend) vs IVV (lends): {{start}} -> {{end}}, {{n}} months'.format(**spy))\n"
            "print('  realised gap  %+7.1f bp/yr   (SPY minus IVV) <- this is the tape'\n"
            "      % spy['drift'])\n"
            "print('  residual under TODAY\\'s fees      (dER {{dER:+.1f}} bp): {{resid:+.1f}} bp/yr'\n"
            "      '  HAC t = {{t:+.2f}}'.format(**spy))\n"
            "print('  residual under 2000-era fees     (dER  +0.0 bp): {early:+.1f} bp/yr'\n"
            "      '  HAC t = {early_t:+.2f}')\n"
            "print('  95% CI on the headline: [{{ci_lo:+.1f}}, {{ci_hi:+.1f}}] bp/yr'.format(**spy))\n"
            "print()\n"
            "print('  -> the SIGN flips with the fee assumption; only the MAGNITUDE is the tape\\'s.')\n"
            "print('  -> bound: any IVV passthrough visible in returns <= {bound:.0f} bp/yr.')"
            .format(spy=R["pairs"][0], bound=R["spy_ci_bound"],
                    early=R["er_sweep_cheap"][0][2][0][3],
                    early_t=R["er_sweep_cheap"][0][2][0][4])
        ),
        md("## 4. The catch that decides the sign: fees were cut\n\n"
           "Every one of these residuals is a *difference of two fees*, and we do not have a "
           "tape of fees — we have today's published numbers. That matters here more than "
           "anywhere, because **the cheap fund in each pair had its fee cut hard during the "
           "sample** (IVV 0.0945% → 0.03%, IEMG 0.18% → 0.09%, IEFA 0.14% → 0.07%) while the "
           "expensive one barely moved.\n\n"
           "Charging the cheap fund its *current* fee for its whole history therefore makes "
           "the fee gap look bigger than it was, which pushes every residual **up** — in "
           "exactly the direction that makes a lending offset look absent. Undo that, and all "
           "three tight-pair residuals cross zero:"),
        code(
            "rows = %r\n"
            "print('pair       cheap leg   fee assumption      dER      residual      t')\n"
            "for name, cheap, sweep in rows:\n"
            "    for label, er_b, der, res, t in sweep:\n"
            "        print('%%-10s %%-11s %%-9s %%.4f%%%%  %%+7.1f bp  %%+7.1f bp  %%+6.2f'\n"
            "              %% (name, cheap, label, er_b, der, res, t))\n"
            "print()\n"
            "print('The residual is smaller than the error in the fee input. So the honest')\n"
            "print('output is a MAGNITUDE (|residual| <= ~6 bp/yr), with no direction attached.')"
            % (R["er_sweep_cheap"],)
        ),
        md("## 5. The big numbers are the wrong kind of big\n\n"
           "Two pairs *do* show large residuals — and both are red herrings:\n\n"
           "- **IWM − IJR: −114 bp/yr.** Those are Russell 2000 and S&P SmallCap 600, two "
           "different index families with a well-documented quality gap. The fee gap is 13 bp. "
           "A number nine times too large, in a pair that does not share a benchmark, is "
           "telling you about index construction, not about lending.\n"
           "- **VEA − IEFA: +49 bp/yr.** FTSE and MSCI disagree about which countries count "
           "as developed. Korea and Canada sit in one and not the other.\n\n"
           "> 🔬 *For the quants:* neither clears |*t*| = 2 over the full sample, and neither "
           "is stable across the 2020 era split."),
        md("## 6. Why the tape could never have settled this\n\n"
           "Here is the uncomfortable arithmetic. Two same-class funds still drift apart by "
           "0.5% to 3.8% a year at random. Over the sample we have, that random drift means "
           "the smallest effect we could ever call statistically real is **20 to 147 basis "
           "points a year**.\n\n"
           "And the thing we are hunting — net lending revenue credited to a broad index "
           "fund — is worth perhaps **1 to 15 basis points a year**.\n\n"
           "So this study cannot *measure* the offset. It can only **bound** it. That bound "
           "is the honest deliverable, and it is a useful one: whatever the lending desks "
           "earn, what reaches you as measurable *relative* return is smaller than roughly "
           "6 bp/yr on every same-sponsor pair, on any fee assumption."),
        md("## 7. Live check — the machinery works (offline synthetic)\n\n"
           "Before believing a zero, check the detector can find a one. On a *simulated* pair "
           "where we plant 60 bp/yr of lending passthrough by hand, the same estimator must "
           "find it; on a simulated pair with no passthrough it must stay silent."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..', '..', '..')))\n"
            "import numpy as np\n"
            "from lending_offset import data, strategy as st\n"
            "planted = st.synthetic_detect(*data.synthetic_daily(signal_strength=1.0, seed=914))\n"
            "print('planted +%.0f bp/yr -> recovered %+.1f bp/yr  (t = %+.2f)'\n"
            "      % (planted['planted_bp'], planted['residual_bp'], planted['t_hac']))\n"
            "nulls = [st.synthetic_detect(*data.synthetic_daily(signal_strength=0.0, seed=914 + s))\n"
            "         for s in range(8)]\n"
            "res = np.array([n['residual_bp'] for n in nulls])\n"
            "ts  = np.array([n['t_hac'] for n in nulls])\n"
            "print('null x8            -> mean %+.1f bp/yr (sd %.1f), fires |t|>=2 in %d/8'\n"
            "      % (res.mean(), res.std(ddof=1), (abs(ts) >= 2).sum()))"
        ),
        md("## 8. Could you at least trade the fee gap?\n\n"
           "Barely. Long IEMG, short EEM harvests the 61 bp fee difference: **+47 bp/yr** "
           "gross, Sharpe +0.38. Then the short leg pays borrow. At 25 bp of borrow it is "
           "+22 bp. At **50 bp of borrow — which is what an EM ETF actually costs to "
           "borrow — it is −3 bp.** Gone.\n\n"
           "The practical lesson is the boring one: if you want the cheap fund's return, "
           "**own the cheap fund**. Do not try to short the expensive one to capture the "
           "difference."),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The lending offset does not show up in returns. The tight "
           f"pairs' residuals are **+3.0 / +4.6 / +5.1 bp/yr** (|*t*| ≤ 1.04), pooled "
           f"**{R['pooled_resid']:+.1f} bp/yr (*t* = {R['pooled_t']:+.2f})** — and "
           f"**{R['pooled_early']:+.1f} bp/yr** if the funds are charged the fees they "
           f"actually charged at the start of the sample. The one structural control — SPY, "
           f"which legally *cannot* lend — ran level with IVV. The two large residuals are "
           f"index-composition wedges. Honest caveats: the test is **underpowered by design** "
           f"(noise floor 20–147 bp/yr against a 1–15 bp effect); **the fee assumption, not "
           f"the tape, decides the residual's sign**, so only its magnitude is reported; and "
           f"every fund here is a **survivor**.\n"
           f"- **Tradability — Mirage.** No lending spread to bank. The fee gap itself is "
           f"tradable only until borrow reaches ~50 bp/yr, at which point it is exactly zero.\n"
           f"- **What you should take away.** The offset is real in the annual reports and "
           f"invisible in the tape. Choose funds on the fee you can see, not on the lending "
           f"revenue you are promised."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 914 — Securities-Lending Offset — the teardown\n\n"
           "The estimand, the five-pair table, the structural SPY/IVV control and its "
           "confounds, the two-sided fee-history sweep that decides the residual's sign, the "
           "pooled residuals, the era cut, the frequency check, the borrow-swept pair trade, "
           "the power arithmetic, and the live synthetic control. Every real number is frozen "
           "from `docs/results.md` (Fingerprint `%s`, as-of %s)." % (R["fp"], R["asof"])),
        md("## The estimand\n\n"
           "For funds $a$, $b$ in the same asset class, on month-end log total returns:\n\n"
           "$$\\text{residual}(a,b) \\;=\\; 12\\cdot\\overline{\\left(\\log r_{a,t} - \\log r_{b,t}\\right)} \\;+\\; (\\mathrm{ER}_a - \\mathrm{ER}_b)$$\n\n"
           "Under the null of *no differential lending passthrough and perfect replication*, "
           "the residual is zero. The lending-offset claim predicts a positive residual for "
           "the leg whose lending programme returns more.\n\n"
           "Three design points, stated once:\n\n"
           "1. **No signal, so no execution lag — and none is faked.** The residual is an "
           "accounting identity on contemporaneous returns. The *tradable* leg is a **static** "
           "pair of two fixed tickers: there is no forecast and nothing to lag. Its friction "
           "is booked once a year in January (a cost convention, not a lag), on the full "
           "notional of both legs — 6.0 bp/yr against a true drift-rebalancing cost of "
           "0.03–0.29 bp/yr, i.e. charged 20–200× over.\n"
           "2. **Frequency.** Monthly. Daily total-return closes for two funds paying on "
           "different dates carry large transient noise (SPY−IVV: 2.20% daily TE vs 0.52% "
           "monthly). Daily is reported as a robustness check and agrees to 1 bp.\n"
           "3. **Non-tape inputs, and the one that binds.** Expense ratios are an "
           "**assumption** (current published net ERs held constant) — and since every cheap "
           "leg's fee was *cut* during the sample, that assumption inflates every residual. "
           "**Both** legs are swept and the whole table is re-run under three fee schedules; "
           "the sign does not survive it, so only a magnitude is reported. Borrow is an "
           "**assumption**, swept 0→100 bp.\n\n"
           "> 💡 *In plain words:* take the return gap between two near-identical funds, add "
           "back the fee gap, and see whether anything is left over."),
        code("R = %r" % (R,)),
        md("## The five pairs"),
        code(
            "hdr = ('pair       class                     window            n   drift    dER   '\n"
            "       'resid      t        95% CI           TE    floor   beta')\n"
            "print(hdr); print('-' * len(hdr))\n"
            "for a, b, k, s, e, n, dr, der, res, t, lo, hi, te, fl, beta in R['pairs']:\n"
            "    print('%-4s-%-5s %-25s %s->%s %4d %+7.1f %+6.1f %+8.1f %+6.2f '\n"
            "          '[%+7.1f,%+7.1f] %5.2f%% %4dbp %6.3f'\n"
            "          % (a, b, k[:25], s, e, n, dr, der, res, t, lo, hi, te, fl, beta))\n"
            "print()\n"
            "print('pooled (tight pairs, equal weight, n=%d): %+.1f bp/yr  t=%+.2f  '\n"
            "      'CI[%+.1f,%+.1f]  TE %.2f%%  floor %dbp'\n"
            "      % (R['pooled_n'], R['pooled_resid'], R['pooled_t'], R['pooled_lo'],\n"
            "         R['pooled_hi'], R['pooled_te'], R['pooled_floor']))"
        ),
        md("**Reading it.** Betas within 1.00 ± 0.03 confirm these are same-exposure pairs. "
           "The three same-sponsor pairs return residuals of 3–5 bp/yr with |*t*| ≤ 1.04. The "
           "two cross-family pairs (VEA−IEFA, IWM−IJR) return large residuals that are "
           "composition wedges: IWM−IJR's −114 bp/yr against a 13 bp fee gap is the "
           "Russell-2000 / S&P-600 quality differential, and it does not clear |*t*| = 2 "
           "either.\n\n"
           "> 💡 *In plain words:* the funds that really track the same thing show nothing; "
           "the funds that show something do not really track the same thing."),
        md("## The structural control — SPY cannot lend\n\n"
           "SPY is a unit investment trust (1993 vintage): it may not lend securities and may "
           "not reinvest dividends between distributions. IVV is an open-end fund that lends. "
           "Identical index, 0.52%/yr relative tracking error — the tightest pair available "
           "anywhere, and 26 years of it.\n\n"
           "A lending offset in IVV predicts a **negative** SPY−IVV residual (SPY should lose "
           "*more* than its fee handicap). The tape's own contribution is a realised drift of "
           "**−1.4 bp/yr**; everything after that is the fee assumption.\n\n"
           "**Two confounds are baked in and are not separable**, so this control bounds a "
           "*net* structural difference rather than IVV's lending revenue: SPY's UIT structure "
           "also forbids reinvesting dividends between distributions (a cash drag penalising "
           "SPY), and SPY's own fee never moved while IVV's fell fourfold."),
        code(
            "a, b, k, s, e, n, dr, der, res, t, lo, hi, te, fl, beta = R['pairs'][0]\n"
            "early = R['er_sweep_cheap'][0][2][0]\n"
            "print('SPY - IVV   %s -> %s   n=%d months   TE %.2f%%/yr   beta %.3f'\n"
            "      % (s, e, n, te, beta))\n"
            "print('  realised drift (THE TAPE)        %+.1f bp/yr' % dr)\n"
            "print('  + fee gap, today\\'s ERs   %+6.1f -> residual %+.1f bp/yr  HAC t %+.2f'\n"
            "      % (der, res, t))\n"
            "print('  + fee gap, 2000-era ERs  %+6.1f -> residual %+.1f bp/yr  HAC t %+.2f'\n"
            "      % (early[2], early[3], early[4]))\n"
            "print('  95%% CI on the headline [%+.1f, %+.1f] bp/yr' % (lo, hi))\n"
            "print()\n"
            "print('  -> the SIGN is the fee assumption\\'s, not the tape\\'s. What the tape')\n"
            "print('     delivers is a BOUND: any IVV passthrough visible in returns is')\n"
            "print('     <= %.0f bp/yr at 95%% confidence.' % hi)\n"
            "print('  -> and it is a bound on a NET quantity: SPY\\'s UIT dividend cash drag')\n"
            "print('     pushes the same residual the other way and cannot be separated out.')"
        ),
        md("## The fee-history sweep — the assumption that owns the sign\n\n"
           "The residual is a difference of two fees plus a drift, and fees are **not tape**. "
           "Every cheap leg here was cut hard during the sample (IVV 0.0945→0.03, IEMG "
           "0.18→0.09, IEFA 0.14→0.07, IJR 0.20→0.06) while the dear legs barely moved. "
           "Holding today's ERs constant therefore overstates every fee gap and biases every "
           "residual **upward** — the direction that makes an offset look absent. Sweeping "
           "only the dear leg would be a one-sided sweep, so both legs are swept and the whole "
           "table is re-run under three labelled schedules.\n\n"
           "*(The endpoints are published headline ERs; the mid value is an approximate "
           "time-weighting of the published cut schedule — an ASSUMPTION, bracketed by the "
           "endpoints, so nothing rests on it being exact.)*"),
        code(
            "print('CHEAP-leg sweep (dear leg pinned at today\\'s ER):')\n"
            "print('pair       cheap  assumption   ER_b      dER      residual      t')\n"
            "for name, cheap, sweep in R['er_sweep_cheap']:\n"
            "    for label, er_b, der_, res_, t_ in sweep:\n"
            "        print('%-10s %-6s %-10s %.4f%%  %+7.1f bp  %+7.1f bp  %+6.2f'\n"
            "              % (name, cheap, label, er_b, der_, res_, t_))\n"
            "print()\n"
            "print('WHOLE TABLE re-run, both legs moved:')\n"
            "print('%-22s %8s %9s %9s %9s %9s' % ('scenario', 'SPY-IVV', 'EEM-IEMG',\n"
            "                                     'EFA-IEFA', 'VEA-IEFA', 'IWM-IJR'))\n"
            "for label, vals in R['fee_scenarios']:\n"
            "    print('%-22s ' % label + ' '.join('%+8.1f' % v for v in vals))\n"
            "print()\n"
            "print('pooled (tight pairs): %+.1f bp (t %+.2f) on today\\'s fees, %+.1f bp (t %+.2f)'\n"
            "      % (R['pooled_resid'], R['pooled_t'], R['pooled_early'], R['pooled_early_t']))\n"
            "print('pooled (ALL FIVE, today\\'s fees): %+.1f bp (t %+.2f) - the class filter is'\n"
            "      % (R['pooled_all'], R['pooled_all_t']))\n"
            "print('shown rather than assumed: dropping the two wedge pairs changes nothing.')\n"
            "print()\n"
            "print('=> the tight-pair residuals are SMALLER than the uncertainty in the fee')\n"
            "print('   input. Magnitude survives (|resid| <= ~6 bp/yr); the sign does not.')"
        ),
        md("## Era cut (split 2020-01-01)"),
        code(
            "print('pair        early resid      t     |  late resid       t')\n"
            "for name, er, et, lr, lt in R['era']:\n"
            "    print('%-10s %+10.1f bp %+6.2f  | %+10.1f bp %+6.2f' % (name, er, et, lr, lt))\n"
            "print()\n"
            "print('Only SPY-IVV is stable across eras - and it is stable at zero.')\n"
            "print('The only |t|>2 among the RESIDUAL estimates (IWM-IJR early) flips sign')\n"
            "print('after 2020: an index-methodology fact, not a fee or lending fact - and it')\n"
            "print('points the opposite way to the lending story anyway.')"
        ),
        md("## The expense-ratio sweep, dear leg\n\n"
           "The smaller half of the assumption, for completeness: EEM's ER was 0.75% at "
           "IEMG's launch and is 0.70% now. The residual moves one-for-one with the assumed "
           "fee gap here too, but the dear legs moved far less than the cheap ones, so this "
           "band leaves the magnitude untouched."),
        code(
            "print('EEM-IEMG, sweeping the assumed EEM expense ratio:')\n"
            "for er, der, res, t in R['er_sweep']:\n"
            "    print('  ER_EEM = %.2f%%  ->  dER %+.1f bp  residual %+.1f bp/yr  t = %+.2f'\n"
            "          % (er, der, res, t))\n"
            "print('\\nacross the full band the residual stays within +-10 bp/yr and |t| < 0.3')"
        ),
        md("## Frequency robustness — daily instead of monthly"),
        code(
            "print('pair        daily resid       t     daily TE   (monthly TE)')\n"
            "mte = {('%s-%s' % (p[0], p[1])): p[12] for p in R['pairs']}\n"
            "for name, res, t, te in R['daily']:\n"
            "    print('%-10s %+10.1f bp %+6.2f   %5.2f%%      %5.2f%%'\n"
            "          % (name, res, t, te, mte[name]))\n"
            "print('\\nsame answers; daily TE is 2-4x larger purely from dividend-date mismatch,')\n"
            "print('which is why the monthly series carries the headline.')"
        ),
        md("## The power arithmetic — why this tape can only bound\n\n"
           "The noise floor is $2\\,\\sigma_{TE}/\\sqrt{T}$: the smallest annual effect the "
           "sample can call real at |*t*| = 2. Set it beside a realistic net lending yield "
           "for a broad index fund (1–15 bp/yr, per Blocher & Whaley 2016)."),
        code(
            "print('pair        TE/yr   years   noise floor   plausible lending yield')\n"
            "for a, b, k, s, e, n, dr, der, res, t, lo, hi, te, fl, beta in R['pairs']:\n"
            "    print('%-4s-%-5s %5.2f%%  %5.1f   %6d bp    1-15 bp' % (a, b, te, n / 12.0, fl))\n"
            "print('\\npooled: floor %d bp/yr against a 1-15 bp effect -> the study is'\n"
            "      % R['pooled_floor'])\n"
            "print('underpowered BY CONSTRUCTION. The reportable quantity is the CI edge,')\n"
            "print('not the point estimate: a bound, not a measurement.')"
        ),
        md("## The tradable leg — long cheap / short dear, with borrow\n\n"
           "Dollar-neutral and **static**: two fixed tickers, no signal, so no execution lag "
           "is applied and none is claimed. The arithmetic is a book restored to equal "
           "notional each month end; friction is charged once a year on the **full** notional "
           "of both legs (6.0 bp/yr, booked in January — a cost convention). That charge is "
           "deliberately conservative, as the turnover check below shows. Self-financing, so "
           "the return is already excess-of-cash and the Sharpe is comparable to any other "
           "excess-of-cash Sharpe on the desk — with the standing caveat that a real short "
           "rebate lands *below* the cash rate, which is what the borrow sweep is for."),
        code(
            "print('cost conservatism: true drift-rebalancing turnover vs what is charged')\n"
            "for name, turn, true_bp in R['turnover']:\n"
            "    print('  %-10s turnover %5.2f%%/yr -> %.2f bp/yr true  vs %.1f bp/yr charged'\n"
            "          % (name, turn, true_bp, R['charged_bp']))\n"
            "print('  (charged 20-200x over: the trade is not flattered by its cost model)')\n"
            "print()\n"
            "print('trade       vol     0bp borrow          25bp      50bp      100bp')\n"
            "for name, g0, sh, t, b25, b50, b100, vol in R['trade']:\n"
            "    print('%-10s %5.2f%%  %+7.1f bp (Sh %+.2f, t %+.2f)  %+7.1f  %+7.1f  %+7.1f'\n"
            "          % (name, vol, g0, sh, t, b25, b50, b100))\n"
            "print()\n"
            "print('IEMG/EEM harvests the 61 bp FEE gap, not lending revenue - and it is')\n"
            "print('exactly zero at 50 bp of borrow, inside the range EM ETFs borrow at.')\n"
            "print('IJR/IWM +110 bp is the small-cap index quality premium (TE 3.78%, t +1.45),')\n"
            "print('a different study entirely.')\n"
            "print()\n"
            "print('context - each leg excess-of-cash Sharpe vs BIL:')\n"
            "print('  ' + '  '.join('%s %+.2f' % (k, v) for k, v in R['ex_sharpe']))"
        ),
        md("## Live synthetic control — the estimator is unbiased and powered\n\n"
           "Planted world: the dearer fund is handed 60 bp/yr of passthrough on top of its "
           "fee; the estimator must recover it. Null world: fees and composition noise only; "
           "the estimator must stay quiet. This is simulated data — it never supports the "
           "real-tape stamp, it only certifies the harness."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..', '..', '..')))\n"
            "import numpy as np\n"
            "from lending_offset import data, strategy as st\n"
            "pl = st.synthetic_detect(*data.synthetic_daily(signal_strength=1.0, seed=914))\n"
            "print('planted %+.0f bp/yr -> recovered %+.1f bp/yr (error %+.1f)  t=%+.2f  TE %.2f%%'\n"
            "      % (pl['planted_bp'], pl['residual_bp'], pl['error_bp'], pl['t_hac'], pl['te_pct']))\n"
            "half = st.synthetic_detect(*data.synthetic_daily(signal_strength=0.5, seed=914))\n"
            "print('half strength %+.0f bp -> %+.1f bp  t=%+.2f  (monotone in the knob)'\n"
            "      % (half['planted_bp'], half['residual_bp'], half['t_hac']))\n"
            "nulls = [st.synthetic_detect(*data.synthetic_daily(signal_strength=0.0, seed=914 + s))\n"
            "         for s in range(8)]\n"
            "res = np.array([n['residual_bp'] for n in nulls])\n"
            "ts  = np.array([n['t_hac'] for n in nulls])\n"
            "print('null x8: mean %+.1f bp (sd %.1f), |t|>=2 in %d/8'\n"
            "      % (res.mean(), res.std(ddof=1), (abs(ts) >= 2).sum()))\n"
            "frames, tr = data.synthetic_panel(signal_strength=1.0, seed=914)\n"
            "pool = st.synthetic_panel_detect(frames, tr)\n"
            "print('pooled %d planted pairs: %+.1f bp  t=%+.2f  (pooling buys power, as on the desk)'\n"
            "      % (pool['n_pairs'], pool['residual_bp'], pool['t_hac']))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The fee-adjusted residual on the three same-sponsor pairs is "
           f"**+3.0 / +4.6 / +5.1 bp/yr**, |*t*| ≤ 1.04, every CI straddling zero; pooled it is "
           f"**{R['pooled_resid']:+.1f} bp/yr (*t* = {R['pooled_t']:+.2f})**, CI "
           f"[{R['pooled_lo']:+.1f}, {R['pooled_hi']:+.1f}] — and "
           f"**{R['pooled_early']:+.1f} bp/yr (*t* = {R['pooled_early_t']:+.2f})** under "
           f"earliest-in-sample fees. The structural control — SPY, a unit investment trust "
           f"that legally **cannot** lend, against IVV, which does — shows a realised drift of "
           f"only −1.4 bp/yr over 26 years, bounding any visible IVV passthrough at "
           f"**≈{R['spy_ci_bound']:.0f} bp/yr** (95% CI). The two large residuals (IWM−IJR "
           f"−114 bp, VEA−IEFA +49 bp) are index-composition wedges and neither clears "
           f"|*t*| = 2. The synthetic control recovers a planted +60 bp at "
           f"*t* = {R['syn_t']:+.2f} and is silent on the null "
           f"({R['syn_null_mean']:+.1f} bp, {R['syn_null_fire']}/8), so the harness is sound. "
           f"Named limits: **power** (noise floor 20–147 bp/yr against a 1–15 bp effect); "
           f"**the fee assumption owns the sign**, so a magnitude is reported and no "
           f"direction; the SPY/IVV control **confounds** lending with SPY's UIT dividend cash "
           f"drag; and **fund survivorship** (all ten funds are survivors).\n"
           f"- **Tradability — Mirage.** No lending spread exists to bank. The fee gap is "
           f"tradable at +47 bp/yr gross (IEMG/EEM) and is **exactly zero at 50 bp of "
           f"borrow** — inside the realistic range. Own the cheap fund; do not short the dear "
           f"one.\n"
           f"- **The deliverable is a magnitude bound — not an estimate, and not a sign.** "
           f"Whatever the lending desks earn, what reaches the shareholder as measurable "
           f"relative return is under ≈6 bp/yr on every same-sponsor pair under every fee "
           f"scenario (CI edge ≈15 bp/yr on the tightest pair, ≈25 bp/yr pooled). Real in the "
           f"annual report; at its plausible size, invisible in this tape — which was never "
           f"capable of resolving it either way."),
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
