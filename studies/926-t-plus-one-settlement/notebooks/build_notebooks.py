"""Generate the two narrative notebooks for Study 926 (T+1).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below, which mirrors ``docs/results.md``; the only live cells run the
fast synthetic control, which is clearly labelled as synthetic and never sits under a
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


# Frozen real-tape headline — mirror of docs/results.md. SPY/IWM/EEM/EFA/BIL daily
# total-return OHLC, difference-in-difference around the 2024-05-28 T+1 switch,
# symmetric 524-day windows, as-of 2026-06-30.
R = dict(
    switch="2024-05-28", asof="2026-06-30", fp="1dea7df6b22f", n_rows=4802,
    pre_start="2022-04-26", pre_end="2024-05-24",
    post_start="2024-05-28", post_end="2026-06-30", n_each=524,
    # DiD, treated minus SPY, post minus pre
    efa_ron=-0.064, efa_ron_t=-0.02, efa_abson=1.080, efa_abson_t=0.35,
    efa_share=0.023, efa_share_t=1.03, efa_abscc=8.668, efa_abscc_t=2.18,
    eem_ron=4.827, eem_ron_t=1.11, eem_abson=8.281, eem_abson_t=1.87,
    eem_share=-0.017, eem_share_t=-0.73, eem_abscc=19.209, eem_abscc_t=3.39,
    iwm_ron=0.675, iwm_ron_t=0.25, iwm_abson=6.000, iwm_abson_t=2.24,
    iwm_share=0.016, iwm_share_t=0.78, iwm_abscc=2.656, iwm_abscc_t=0.63,
    # levels of the overnight variance share
    efa_share_pre=0.531, efa_share_post=0.559,
    eem_share_pre=0.589, eem_share_post=0.578,
    spy_share_pre=0.415, spy_share_post=0.420,
    spy_abscc_pre=82.97, spy_abscc_post=69.13,
    efa_abscc_pre=78.95, efa_abscc_post=73.78,
    eem_abscc_pre=85.76, eem_abscc_post=91.13,
    # bootstrap CIs on the DiD
    ci_efa_share=(-0.023, 0.069), ci_efa_abscc=(-0.93, 18.02),
    ci_eem_share=(-0.059, 0.025), ci_eem_abscc=(4.64, 34.44),
    # window-length sweep on abs_cc
    efa_abscc_w=((63, 12.83, 1.76), (126, 8.78, 1.43), (252, -8.17, -1.53), (524, 8.67, 2.18)),
    eem_abscc_w=((63, 8.94, 1.14), (126, 10.95, 1.22), (252, -10.71, -1.56), (524, 19.21, 3.39)),
    efa_share_w=((63, 0.137, 2.31), (126, 0.079, 1.97), (252, 0.038, 1.18), (524, 0.023, 1.03)),
    eem_share_w=((63, 0.003, 0.04), (126, -0.024, -0.47), (252, -0.008, -0.21), (524, -0.017, -0.73)),
    # placebo switch dates: (leg, outcome, real |t|, median |t|, max |t|, exceeding, n)
    placebo=(
        ("EFA", "r_on", 0.02, 1.48, 3.71, 88, 88),
        ("EFA", "abs_on", 0.35, 2.77, 10.77, 82, 88),
        ("EFA", "on_var_share", 1.03, 1.94, 8.48, 69, 88),
        ("EFA", "abs_cc", 2.18, 4.97, 10.64, 69, 88),
        ("EEM", "r_on", 1.11, 2.21, 5.05, 47, 81),
        ("EEM", "abs_on", 1.87, 3.45, 5.21, 59, 81),
        ("EEM", "on_var_share", 0.73, 2.78, 6.22, 66, 81),
        ("EEM", "abs_cc", 3.39, 6.26, 10.08, 61, 81),
        ("IWM", "r_on", 0.25, 1.82, 4.25, 89, 93),
        ("IWM", "abs_on", 2.24, 2.63, 10.44, 55, 93),
        ("IWM", "on_var_share", 0.78, 1.92, 6.00, 72, 93),
        ("IWM", "abs_cc", 0.63, 2.05, 7.14, 76, 93),
    ),
    # excess-of-cash Sharpe vs BIL, pre -> post
    sharpe=(("SPY", 0.52, 0.90, 0.34), ("IWM", 0.13, 0.80, 0.66),
            ("EEM", 0.06, 1.09, 1.14), ("EFA", 0.42, 0.79, 0.36)),
    # turn-of-month overlay, 3 bps one-way per leg, 50 bps/yr borrow
    efa_tom_gross=(-1.73, 5.38), efa_tom_net=(-3.42, 3.68),
    efa_tom_did=7.11, efa_tom_did_t=0.57, efa_tom_post_t=0.40, efa_tom_sharpe=0.17,
    eem_tom_gross=(10.75, 17.80), eem_tom_net=(9.05, 16.11),
    eem_tom_did=7.06, eem_tom_did_t=0.43, eem_tom_post_t=1.40, eem_tom_sharpe=0.96,
    tom_on_days=100,
    sweep=((0.0, 0.0, 17.80, 1.55), (3.0, 50.0, 16.11, 1.40), (10.0, 100.0, 12.41, 1.08)),
    # synthetic control
    syn_pl_ron=16.47, syn_pl_ron_t=2.73, syn_pl_abson=13.94, syn_pl_abson_t=3.75,
    syn_nl_ron=1.05, syn_nl_ron_t=0.19, syn_nl_abson=1.06, syn_nl_abson_t=0.31,
)


HEADER = f"""# Study 926 — T+1 ⏱️

**On 28 May 2024 the United States halved its settlement cycle. Did the tape notice?**

US cash equities, ETFs and corporate bonds moved from **T+2 to T+1** settlement on
Tuesday **{R['switch']}** (SEC Rule 15c6-1(a) as amended). Europe, the UK and most of Asia
did not move. That leaves a fund like **EFA** — US-listed shares that now settle in one
day, European and Japanese holdings that still settle in two — carrying a genuine
settlement mismatch, while **SPY** carries none.

So we run a **difference-in-difference**: measure the same four daily quantities on the
treated funds and on SPY, and ask whether the *gap* between them changed at the switch.
The windows are symmetric — **{R['n_each']} trading days either side**,
{R['pre_start']} → {R['pre_end']} against {R['post_start']} → {R['post_end']}.

*Real-tape numbers below are the frozen headline (`docs/results.md`, return fingerprint
`{R['fp']}`, {R['n_rows']:,} daily rows); the live cells run the offline synthetic control
and are labelled as such. Total-return closes (`auto_adjust=True`). As-of {R['asof']}.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. What actually changed\n\n"
           "Before May 2024, when you bought a US share you paid for it two business days\n"
           "later. After May 2024, one. Nothing about *what* the share is worth changed —\n"
           "only how quickly money and stock have to arrive.\n\n"
           "The interesting case is a fund like **EFA**, which owns European and Japanese\n"
           "shares but trades in New York. Its own shares now settle in one day; the things\n"
           "it owns still settle in two. Someone has to bridge that day — with cash, with\n"
           "borrowed stock, with an FX swap. If that cost were large enough, you might expect\n"
           "to see it somewhere in the price: a jumpier overnight session, a different split\n"
           "between what the fund does while New York sleeps and what it does while New York\n"
           "trades, or a different pattern around month-end when flows are heaviest.\n\n"
           "> 🔬 **For the quants** — the four outcomes come from the exact identity\n"
           "> `(1 + r_overnight)(1 + r_intraday) = (1 + r_close-to-close)`. No model, no\n"
           "> parameter, no fitting; `auto_adjust=True` scales open and close by the same\n"
           "> factor, so the identity survives dividend adjustment."),
        md("## 2. The headline: nothing moved\n\n"
           "The measure that matters is the **overnight share of the day's risk** — of all the\n"
           "variance a fund produces in 24 hours, how much lands between the closing bell and\n"
           "the next open. If settlement pressure had migrated into the overnight window, this\n"
           "is where it would show."),
        code(
            "R = dict(efa_share=%r, efa_share_t=%r, eem_share=%r, eem_share_t=%r,\n"
            "         efa_share_pre=%r, efa_share_post=%r,\n"
            "         eem_share_pre=%r, eem_share_post=%r,\n"
            "         spy_share_pre=%r, spy_share_post=%r)\n"
            "print('overnight share of daily variance, before -> after 28 May 2024')\n"
            "print('  SPY (control) : %%.3f -> %%.3f' %% (R['spy_share_pre'], R['spy_share_post']))\n"
            "print('  EFA (treated) : %%.3f -> %%.3f' %% (R['efa_share_pre'], R['efa_share_post']))\n"
            "print('  EEM (treated) : %%.3f -> %%.3f' %% (R['eem_share_pre'], R['eem_share_post']))\n"
            "print()\n"
            "print('difference-in-difference (treated minus SPY, after minus before):')\n"
            "print('  EFA %%+.3f  (t = %%+.2f)' %% (R['efa_share'], R['efa_share_t']))\n"
            "print('  EEM %%+.3f  (t = %%+.2f)  <- opposite sign to EFA' %% (R['eem_share'], R['eem_share_t']))"
            % (R["efa_share"], R["efa_share_t"], R["eem_share"], R["eem_share_t"],
               R["efa_share_pre"], R["efa_share_post"], R["eem_share_pre"],
               R["eem_share_post"], R["spy_share_pre"], R["spy_share_post"])
        ),
        md(f"Two funds that should have been treated the same way move in **opposite\n"
           f"directions**, neither anywhere near significance. Note also the *levels*: EFA and\n"
           f"EEM put ~{R['efa_share_pre']:.0%}–{R['eem_share_pre']:.0%} of their daily risk into the "
           f"overnight window against SPY's ~{R['spy_share_pre']:.0%} — but that is not a settlement fact. "
           f"**Their \"overnight\" contains the entire European and Asian trading session.** That "
           f"mechanical confound is why this study can only ever ask about the *change*, never "
           f"about the level."),
        md("## 3. The two results that looked real — and why they aren't\n\n"
           "Two numbers in the whole battery cleared a *t* of 2. Both are cautionary tales.\n\n"
           f"**Total volatility (EEM, *t* = {R['eem_abscc_t']:+.2f}).** Look at where it comes from: "
           f"SPY's own daily move shrank from {R['spy_abscc_pre']:.0f} to {R['spy_abscc_post']:.0f} bps "
           f"while EEM's went from {R['eem_abscc_pre']:.0f} to {R['eem_abscc_post']:.0f}. The 'effect' is "
           f"the US mega-cap tape calming down after 2024 — a fact about SPY, tagged onto EEM by "
           f"the subtraction.\n\n"
           f"**Overnight volatility (IWM, *t* = {R['iwm_abson_t']:+.2f}).** IWM is US small caps: its "
           f"shares *and* its holdings both moved to T+1, so it has no mismatch to feel. It is the "
           f"placebo, and the placebo fired."),
        md("## 4. The test that settles it\n\n"
           "Pretend the settlement change happened on some other date — every quarter across the\n"
           "sample — and run exactly the same analysis. If 28 May 2024 is special, its *t* should\n"
           "stand out. Here is where it actually ranks:"),
        code(
            "placebo = %r\n"
            "print('leg  outcome        real |t|   median fake |t|   fake dates that beat it')\n"
            "for leg, oc, real, med, mx, exc, n in placebo:\n"
            "    print('%%-4s %%-13s %%7.2f %%15.2f %%18s' %% (leg, oc, real, med, '%%d / %%d' %% (exc, n)))"
            % (R["placebo"],)
        ),
        md("On **every** outcome and **every** fund, the *typical* made-up date produces a bigger\n"
           "difference-in-difference than the real one does. The EEM total-volatility result — the\n"
           "study's largest *t* — is beaten by 61 of 81 arbitrary dates whose median is 6.26.\n\n"
           "The reason is not subtle: 2022–2026 contains a rate-hiking cycle, a rate-cutting cycle,\n"
           "a tariff shock and a volatility regime change. Cut that tape anywhere and you will find\n"
           "a 'structural break'. **28 May 2024 is one of the least remarkable dates you could\n"
           "have picked.**"),
        md("## 5. Could you trade it anyway?\n\n"
           "The one tradable version: buy EFA (or EEM), sell SPY, hold the spread across the\n"
           "turn of the month — the window where settlement and rebalancing flows are heaviest —\n"
           "with a one-day execution lag, 3 bps a side and 50 bps/yr of borrow on the short leg.\n"
           "The lag is not cosmetic: because the calendar signal fires *on* the month-end\n"
           "session, the days actually held are the **first four trading days of each month**,\n"
           "and the month-end session itself never makes it into the book."),
        code(
            "R = dict(efa_net=%r, eem_net=%r, efa_did=%r, efa_did_t=%r,\n"
            "         eem_did=%r, eem_did_t=%r, eem_post_t=%r, eem_sharpe=%r, on_days=%r)\n"
            "print('long EFA / short SPY : %%+.2f -> %%+.2f bps per on-day (net)'\n"
            "      %% R['efa_net'])\n"
            "print('long EEM / short SPY : %%+.2f -> %%+.2f bps per on-day (net)'\n"
            "      %% R['eem_net'])\n"
            "print()\n"
            "print('change at the switch: EFA %%+.2f bps (t %%+.2f) | EEM %%+.2f bps (t %%+.2f)'\n"
            "      %% (R['efa_did'], R['efa_did_t'], R['eem_did'], R['eem_did_t']))\n"
            "print('EEM post-period alone: Sharpe %%+.2f but t = %%+.2f on only %%d on-days'\n"
            "      %% (R['eem_sharpe'], R['eem_post_t'], R['on_days']))"
            % (R["efa_tom_net"], R["eem_tom_net"], R["efa_tom_did"], R["efa_tom_did_t"],
               R["eem_tom_did"], R["eem_tom_did_t"], R["eem_tom_post_t"],
               R["eem_tom_sharpe"], R["tom_on_days"])
        ),
        md("The EEM spread's post-period Sharpe of +0.96 is the prettiest number in the study, and\n"
           "it is not a T+1 effect: it was paying +10.75 bps per on-day *before* the switch too.\n"
           "The **change** — the only thing this study is testing — is +7 bps with a *t* of +0.43,\n"
           "and stays there at every cost and borrow assumption we swept.\n\n"
           "> 🔬 **For the quants** — a constant per-trade cost cancels in a difference-in-\n"
           "> difference by construction, which is why the DiD is flat across the whole cost ×\n"
           "> borrow surface. That invariance is reported deliberately rather than hidden."),
        md("## 6. Live check — the machinery works (offline synthetic)\n\n"
           "The cells below build a **synthetic** two-fund world in which we *plant* a settlement\n"
           "break — the treated fund's overnight drift and volatility jump on the switch date —\n"
           "and a second world where nothing happens. The estimator must find the first and stay\n"
           "silent on the second. Nothing here touches the real tape."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from t_plus_one import data, strategy as st\n"
            "pl_panel, pl_truth = data.synthetic_panel(signal_strength=1.0, seed=926)\n"
            "nl_panel, nl_truth = data.synthetic_panel(signal_strength=0.0, seed=926)\n"
            "pl = st.synthetic_detect(pl_panel, pl_truth)\n"
            "nl = st.synthetic_detect(nl_panel, nl_truth)\n"
            "print('SYNTHETIC (not the real tape)')\n"
            "print('planted break: overnight drift DiD %+.2f bps (t %+.2f) -- must fire'\n"
            "      % (pl['did_r_on_bps'], pl['t_r_on']))\n"
            "print('null world   : overnight drift DiD %+.2f bps (t %+.2f) -- must stay quiet'\n"
            "      % (nl['did_r_on_bps'], nl['t_r_on']))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The overnight risk split does not move: "
           f"{R['efa_share']:+.3f} (*t* = {R['efa_share_t']:+.2f}) for EFA, "
           f"{R['eem_share']:+.3f} (*t* = {R['eem_share_t']:+.2f}) for EEM — opposite signs, both "
           f"insignificant, both bootstrap CIs across zero. The two significant results are a "
           f"total-vol measure driven by SPY's own compression and an overnight-vol move in the "
           f"domestic **placebo**. And the placebo-date distribution buries everything: the median "
           f"made-up date beats the real one on every outcome.\n"
           f"- **Tradability — Mirage.** The month-end spread changes by ~+7 bps per on-day with "
           f"*t* between +0.4 and +0.6, gross, at every cost and borrow assumption. There is "
           f"nothing to size.\n"
           f"- **The honest reading.** A settlement cycle is plumbing. It changed who funds what "
           f"overnight — FX swaps, stock loan recalls, ETF creation baskets — none of which appear "
           f"in a price file. A daily open/close tape was never going to see it, and it didn't."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 926 — T+1 — the teardown\n\n"
           "The difference-in-difference specification, HAC inference, block-bootstrap CIs, the\n"
           "window-length sign flip, the placebo-switch-date distribution, the excess-of-cash "
           "Sharpe race, the costed turn-of-month overlay and the live synthetic control. Every "
           "real number is frozen from `docs/results.md` (return fingerprint `%s`, as-of %s)."
           % (R["fp"], R["asof"])),
        md("## Specification\n\n"
           "For each daily outcome `y` and each treated leg *i*, form the difference against the\n"
           "control, `d_t = y_i,t − y_SPY,t`, restrict to symmetric windows, and estimate\n\n"
           "$$d_t = \\alpha + \\beta \\cdot \\mathbb{1}[t \\geq \\text{2024-05-28}] + \\varepsilon_t$$\n\n"
           "with Newey-West standard errors at bandwidth $\\lfloor 4(n/100)^{2/9} \\rfloor$. "
           "$\\beta$ is the DiD. The common market factor cancels inside `d_t`, which is what makes "
           "the estimator well-behaved against the 2022–2026 macro path — but it cannot cancel the "
           "**structural** difference that EFA's and EEM's overnight window contains the entire "
           "foreign cash session.\n\n"
           "> 💡 **In plain words** — we are not asking whether EFA behaves differently from SPY. "
           "Of course it does. We are asking whether the *size of that difference* changed on one "
           "specific day."),
        code("R = %r" % (R,)),
        md("## The four outcomes, three legs"),
        code(
            "rows = [('EFA','r_on',R['efa_ron'],R['efa_ron_t']),\n"
            "        ('EFA','abs_on',R['efa_abson'],R['efa_abson_t']),\n"
            "        ('EFA','on_var_share',R['efa_share'],R['efa_share_t']),\n"
            "        ('EFA','abs_cc',R['efa_abscc'],R['efa_abscc_t']),\n"
            "        ('EEM','r_on',R['eem_ron'],R['eem_ron_t']),\n"
            "        ('EEM','abs_on',R['eem_abson'],R['eem_abson_t']),\n"
            "        ('EEM','on_var_share',R['eem_share'],R['eem_share_t']),\n"
            "        ('EEM','abs_cc',R['eem_abscc'],R['eem_abscc_t']),\n"
            "        ('IWM*','r_on',R['iwm_ron'],R['iwm_ron_t']),\n"
            "        ('IWM*','abs_on',R['iwm_abson'],R['iwm_abson_t']),\n"
            "        ('IWM*','on_var_share',R['iwm_share'],R['iwm_share_t']),\n"
            "        ('IWM*','abs_cc',R['iwm_abscc'],R['iwm_abscc_t'])]\n"
            "print('leg   outcome           DiD    HAC t   flag')\n"
            "for leg, oc, did, t in rows:\n"
            "    flag = '<-- |t| > 2' if abs(t) >= 2 else ''\n"
            "    print('%-5s %-13s %+8.3f %+7.2f   %s' % (leg, oc, did, t, flag))\n"
            "print('\\n* IWM is the DOMESTIC placebo-treated leg: shares and holdings both T+1.')\n"
            "print('  windows: %s -> %s vs %s -> %s, %d trading days each'\n"
            "      % (R['pre_start'], R['pre_end'], R['post_start'], R['post_end'], R['n_each']))"
        ),
        md("Three points. First, `on_var_share` — the outcome with the clearest mechanical link to\n"
           "a settlement change — is insignificant on both treated legs and **carries opposite\n"
           "signs**. Second, `abs_cc` fires on both treated legs, but the driver is the control:\n"
           "SPY's mean absolute daily move fell 82.97 → 69.13 bps while EFA's fell 78.95 → 73.78.\n"
           "Third, the placebo leg fires on `abs_on`, which is a direct falsification of the\n"
           "settlement interpretation."),
        md("## Block-bootstrap CIs on the DiD (2,000 draws, 21-day circular blocks)\n\n"
           "Pre and post halves are resampled separately in blocks, and the statistic recomputed\n"
           "as `mean(post) − mean(pre)`."),
        code(
            "for tag, pt, ci in [('EFA on_var_share', R['efa_share'], R['ci_efa_share']),\n"
            "                    ('EFA abs_cc     ', R['efa_abscc'], R['ci_efa_abscc']),\n"
            "                    ('EEM on_var_share', R['eem_share'], R['ci_eem_share']),\n"
            "                    ('EEM abs_cc     ', R['eem_abscc'], R['ci_eem_abscc'])]:\n"
            "    zero = 'includes 0' if ci[0] <= 0 <= ci[1] else 'excludes 0'\n"
            "    print('%s  point %+8.3f  95%% CI [%+8.3f, %+8.3f]  %s' % (tag, pt, ci[0], ci[1], zero))"
        ),
        md("## Window-length sweep — the sign flip\n\n"
           "An event study has no eras in the buy-and-hold sense; the honest substitute is to\n"
           "measure the same break over one quarter, six months, one year and two years either\n"
           "side. A real break survives all four."),
        code(
            "for tag, sweep in [('EFA on_var_share', R['efa_share_w']),\n"
            "                   ('EEM on_var_share', R['eem_share_w']),\n"
            "                   ('EFA abs_cc      ', R['efa_abscc_w']),\n"
            "                   ('EEM abs_cc      ', R['eem_abscc_w'])]:\n"
            "    cells = '  '.join('+/-%3dd %+8.3f (t %+5.2f)' % row for row in sweep)\n"
            "    print('%s  %s' % (tag, cells))\n"
            "print('\\nBoth abs_cc results CHANGE SIGN at a one-year window and change back at two.')"
        ),
        md("## Placebo switch dates — the falsification\n\n"
           "Re-run the identical DiD pretending the change happened every 63 trading days instead,\n"
           "excluding the year either side of the true date so the real break cannot leak into its\n"
           "own null (Bertrand-Duflo-Mullainathan 2004). Then rank the true |*t*|."),
        code(
            "print('leg  outcome        real|t|  median|t|   max|t|   exceeding   percentile')\n"
            "for leg, oc, real, med, mx, exc, n in R['placebo']:\n"
            "    pct = 100.0 * (n - exc) / n\n"
            "    print('%-4s %-13s %7.2f %10.2f %8.2f %8s %11.0f%%'\n"
            "          % (leg, oc, real, med, mx, '%d / %d' % (exc, n), pct))\n"
            "print('\\nOn every row the MEDIAN fake date produces a larger |t| than the real one.')\n"
            "print('The headline EEM abs_cc t = +3.39 sits at the 25th percentile of noise.')"
        ),
        md("## Excess-of-cash Sharpe race (vs BIL), pre vs post\n\n"
           "Both arms excess of the *same* cash leg — which matters here, since the pre-window sits\n"
           "at ~5% short rates and the post-window walks them down."),
        code(
            "print('ETF   exSharpe pre -> post   Welch t')\n"
            "for tk, pre, post, t in R['sharpe']:\n"
            "    print('%-4s   %+.2f  ->  %+.2f        %+.2f' % (tk, pre, post, t))\n"
            "print('\\nEverything improved, including the control. That is a bull market, not a')\n"
            "print('settlement cycle: no Welch t is remotely near 2.')"
        ),
        md("## The costed overlay and its assumption sweep\n\n"
           "Dollar-neutral long EFA/EEM, short SPY, on a turn-of-month calendar signal (last\n"
           "trading day + first three). Position for *t+1* set from the calendar known through *t*\n"
           "— **one execution lag**, which shifts the *held* window forward: the book is on for the\n"
           "**first four trading days of each month** and flat on the month-end session itself\n"
           "(every on-day has within-month rank 0–3). Costs are one-way × NAV on **each** leg at\n"
           "every position change; the short leg pays borrow. Both are ASSUMPTIONS, not\n"
           "measurements, so both are swept. The post-period Sharpes below are annualised on a\n"
           "series that is flat ~80% of days — read the HAC *t*, not the Sharpe."),
        code(
            "print('spread                gross pre->post      net pre->post      DiD net (t)   post t')\n"
            "print('long EFA / short SPY  %+6.2f -> %+6.2f     %+6.2f -> %+6.2f     %+5.2f (%+.2f)   %+.2f'\n"
            "      % (R['efa_tom_gross'][0], R['efa_tom_gross'][1], R['efa_tom_net'][0],\n"
            "         R['efa_tom_net'][1], R['efa_tom_did'], R['efa_tom_did_t'], R['efa_tom_post_t']))\n"
            "print('long EEM / short SPY  %+6.2f -> %+6.2f     %+6.2f -> %+6.2f     %+5.2f (%+.2f)   %+.2f'\n"
            "      % (R['eem_tom_gross'][0], R['eem_tom_gross'][1], R['eem_tom_net'][0],\n"
            "         R['eem_tom_net'][1], R['eem_tom_did'], R['eem_tom_did_t'], R['eem_tom_post_t']))\n"
            "print('\\ncost x borrow sweep (post-period net, long EEM / short SPY, bps per on-day):')\n"
            "for c, b, net, t in R['sweep']:\n"
            "    print('  cost %5.1f bps, borrow %6.1f bps/yr -> net %+6.2f (t %+.2f)' % (c, b, net, t))\n"
            "print('\\nNever clears |t| = 2, not even gross and borrow-free. The DiD is flat across')\n"
            "print('the surface because a constant cost cancels in a difference-in-difference.')"
        ),
        md("## Live synthetic control — the estimator is unbiased\n\n"
           "A **synthetic** two-fund panel with a planted break (treated leg's overnight drift +8\n"
           "bps/day, +45 bps on turn-of-month days, +30% idiosyncratic overnight vol, from the\n"
           "switch onwards) and a null panel where nothing happens. Nothing below reads the real\n"
           "tape."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from t_plus_one import data, strategy as st\n"
            "pl_panel, pl_truth = data.synthetic_panel(signal_strength=1.0, seed=926)\n"
            "pl = st.synthetic_detect(pl_panel, pl_truth)\n"
            "expected = pl_truth['planted_on_drift_bps'] + pl_truth['planted_tom_bps'] * pl_truth['tom_frac']\n"
            "print('SYNTHETIC WORLD (never the real tape)')\n"
            "print('planted overnight drift shift: %+.2f bps expected, %+.2f recovered (t %+.2f)'\n"
            "      % (expected, pl['did_r_on_bps'], pl['t_r_on']))\n"
            "print('planted overnight vol lift   : DiD abs_on %+.2f bps (t %+.2f)'\n"
            "      % (pl['did_abs_on_bps'], pl['t_abs_on']))\n"
            "nulls = []\n"
            "for s in range(6):\n"
            "    p, tr = data.synthetic_panel(signal_strength=0.0, seed=926 + s)\n"
            "    nulls.append(st.synthetic_detect(p, tr)['t_r_on'])\n"
            "nulls = np.array(nulls)\n"
            "print('null x6: t on overnight drift mean %+.2f (sd %.2f), |t| >= 2 in %d/6'\n"
            "      % (nulls.mean(), nulls.std(ddof=1), int((np.abs(nulls) >= 2).sum())))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The mechanism-linked outcome, `on_var_share`, gives "
           f"{R['efa_share']:+.3f} (*t* = {R['efa_share_t']:+.2f}) on EFA and "
           f"{R['eem_share']:+.3f} (*t* = {R['eem_share_t']:+.2f}) on EEM — opposite signs, both "
           f"insignificant, both block-bootstrap CIs straddling zero. The two |t| ≥ 2 results are "
           f"`abs_cc` (a total-vol measure whose entire movement is SPY's own compression, and "
           f"which **flips sign** at a ±252-day window) and `abs_on` on the **domestic placebo** "
           f"IWM. The placebo-switch-date distribution is decisive: on all twelve reported rows the "
           f"median arbitrary date exceeds the true one, and the study's largest *t* (+3.39) is "
           f"beaten by 61 of 81 fake dates whose median is +6.26. The synthetic control recovers a "
           f"planted break cleanly ({R['syn_pl_ron']:+.2f} bps, *t* = {R['syn_pl_ron_t']:+.2f}) and "
           f"is silent on the null (*t* = {R['syn_nl_ron_t']:+.2f}), so the miss is the tape's, not "
           f"the harness's.\n"
           f"- **Tradability — Mirage.** The turn-of-month spread's change at the switch is "
           f"+7 bps per on-day with HAC *t* of +0.43 (EEM) and +0.57 (EFA), invariant across the "
           f"whole cost × borrow surface and never clearing |t| = 2 even gross. The EEM spread's "
           f"post-period Sharpe of +0.96 rests on 100 on-days with its own *t* of +1.40, and was "
           f"already positive pre-switch.\n"
           f"- **Scope, honestly.** All assumptions are labelled in `docs/results.md`: the 3 bps "
           f"one-way cost, the 50 bps/yr borrow, the hardcoded event date and the choice of which "
           f"funds count as treated. The channels where a T+1 effect would actually live — FX swap "
           f"funding, stock-loan recall timing, ETF creation/redemption fails — are not in a price "
           f"file, and this study makes no claim about them."),
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
