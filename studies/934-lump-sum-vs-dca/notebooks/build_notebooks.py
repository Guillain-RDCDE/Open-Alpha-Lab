"""Generate the two narrative notebooks for Study 934 (Lump Sum vs DCA).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below (a mirror of ``docs/results.md``); the only live cells run the
fast synthetic control, and they are never placed under a real-tape banner.
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


# Frozen real-tape headline — mirror of docs/results.md. SPY / IEF vs BIL, daily
# total-return closes, 12-month horizon, 12 monthly tranches, 1 bp one-way,
# 2007-05-30 -> 2026-06-30, as-of 2026-06-30.
R = dict(
    start="2007-05-30", end="2026-06-30", n_days=4802, n_win=217,
    fp_spy="edef65f148a6", fp_ief="9803a2a6157d",
    first_start="2007-06-01", last_end="2026-06-01",
    win=76.0, win_lo=69.9, win_hi=81.2,
    mean_gap=5.05, median_gap=6.09, sd_gap=9.93,
    p05=-10.52, p95=18.73, worst_gap=-30.65, best_gap=43.93,
    t_hac=3.19, t_nonoverlap=2.18, boot_lo=1.57, boot_hi=7.90, boot_neg=0.2,
    lump_mean=12.35, dca_mean=7.29, lump_worst=-45.85, dca_worst=-36.13,
    sd_lump=0.1701, sd_dca=0.0999, disp_ratio=0.587,
    cheap_n=61, cheap_win=93.4, cheap_gap=10.10, cheap_t=9.66, cheap_ret=22.1,
    mid_n=60, mid_win=80.0, mid_gap=5.64, mid_t=5.22, mid_ret=13.8,
    str_n=60, str_win=73.3, str_gap=3.43, str_t=2.38, str_ret=8.5,
    dd_n=59, dd_win=72.9, dd_gap=5.26, dd_med=7.48, dd_t=1.50, dd_ret=13.8,
    hi_n=158, hi_win=77.2, hi_gap=4.98, hi_t=3.63, hi_ret=11.8,
    era_e_n=115, era_e_win=70.4, era_e_gap=4.09, era_e_t=1.65,
    era_l_n=102, era_l_win=82.4, era_l_gap=6.14, era_l_t=3.39,
    long_start="2000-01-03", long_n=305, long_win=74.4, long_gap=4.47, long_t=3.25,
    long_t_no=2.36, long_e_gap=0.67, long_e_t=0.26, long_l_gap=6.90, long_l_t=6.01,
    dec00_n=119, dec00_win=60.5, dec00_gap=0.67, dec00_t=0.26, dec00_ret=1.6,
    dec10_n=120, dec10_win=85.0, dec10_gap=6.10, dec10_t=6.61, dec10_ret=13.5,
    dec20_n=66, dec20_win=80.3, dec20_gap=8.36, dec20_t=3.06, dec20_ret=17.6,
    long_em_gap=0.03, long_em_t=0.06,
    # The exposure control — DCA's analytic average weight is (n+1)/2n = 13/24.
    em_w=54.2, em_gap=-0.04, em_win=53.5, em_t=-0.08, em_t_no=-0.06,
    em_lo=-1.18, em_hi=0.98, em_sd=0.0926,
    dm_w=58.4, dm_gap=0.43, dm_t=0.78, dm_lo=-0.80, dm_hi=1.48,
    rd_lump=0.651, rd_dca=0.608, xs_lump=11.11, xs_dca=6.06,
    ief_em_gap=0.08, ief_em_t=0.47, ief_em_lo=-0.26, ief_em_hi=0.42,
    cost0=5.05, cost25=5.04,
    tick0=5.05, tick1=5.16, tick5=5.60, tick10=6.15,
    zero_cash_gap=5.64, zero_cash_t=3.57,
    tr3_gap=0.81, tr6_gap=2.18, tr12_gap=5.05, tr24_gap=11.61,
    tr3_win=65.0, tr6_win=70.9, tr12_win=76.0, tr24_win=86.8,
    tr3_disp=0.727, tr6_disp=0.639, tr12_disp=0.587, tr24_disp=0.497,
    ief_win=59.0, ief_gap=1.02, ief_t=1.45, ief_lo=-0.36, ief_hi=2.29,
    ief_disp=0.628,
    syn_pl_gap=3.74, syn_pl_win=62.7, syn_pl_seeds="12/12",
    syn_nl_gap=0.13, syn_nl_win=46.6, syn_nl_seeds="6/12",
    syn_fa_gap=-3.15, syn_fa_win=32.0, syn_fa_seeds="0/12",
)


HEADER = f"""# Study 934 — Lump Sum vs DCA 💸

**A windfall lands. Send it all in on Monday, or drip it in over a year?**

The most repeated piece of retail money advice makes three promises at once: averaging in
gets you a *better average price*, *less risk*, and *more money*. We test all three on
the tape, over **every start month** of the sample.

The setup: $1, twelve months, valued on the same terminal date either way. **Lump sum**
buys the whole dollar at the start. **DCA** buys 1/12 at each of twelve month-ends, and —
unlike almost every published version of this test — the money still waiting sits in
**BIL** and earns the **real T-bill yield** (0% for six of these years, ~5% for three).
One execution lag: decided at a month-end close, filled at the next day's close.

Real tape: **SPY vs BIL**, daily total-return closes, {R['start']} → {R['end']}
({R['n_win']} start months), 1 bp one-way. Bond-heavy variant: **IEF vs BIL**.

*Every real number below is frozen from `docs/results.md` (SPY fingerprint
`{R['fp_spy']}`); the live cells run only the offline synthetic control. As-of 2026-06-30.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. Who finishes richer?\n\n"
           "Roll the whole experiment forward one month at a time and count. Over "
           f"{R['n_win']} start months from {R['first_start']} to {R['last_end']}:"),
        code(
            "R = " + repr({k: R[k] for k in ("win", "win_lo", "win_hi", "mean_gap",
                                             "median_gap", "lump_mean", "dca_mean",
                                             "t_hac")}) + "\n"
            "print(f\"lump sum finishes richer in {R['win']:.1f}% of start months  \"\n"
            "      f\"(95% CI {R['win_lo']:.1f}%-{R['win_hi']:.1f}%)\")\n"
            "print(f\"average gap: {R['mean_gap']:+.2f} cents on every dollar invested   \"\n"
            "      f\"(median {R['median_gap']:+.2f})\")\n"
            "print(f\"average 12-month outcome: lump {R['lump_mean']:+.2f}%   vs   DCA {R['dca_mean']:+.2f}%\")"
        ),
        md("## 2. Why — and it is not a market view\n\n"
           "Nothing here is a forecast. Stocks pay a premium *on average*, and every "
           "dollar sitting in the queue is a dollar not being paid it. The T-bill yield "
           "the waiting money earns closes part of the gap but nowhere near all of it: "
           f"crediting the real BIL path instead of the usual 0% assumption gives DCA back "
           f"**{R['zero_cash_gap'] - R['mean_gap']:.2f} cents** of a "
           f"**{R['zero_cash_gap']:.2f}-cent** lead. \n\n"
           "> 🔬 **For the quants** — the windows overlap by up to eleven months, so the "
           f"headline *t* is Newey-West with 12 lags (**{R['t_hac']:+.2f}**), backed by a "
           f"fully non-overlapping check (**{R['t_nonoverlap']:+.2f}**) and a 12-month "
           f"block bootstrap CI of **[{R['boot_lo']:+.2f}, {R['boot_hi']:+.2f}]** cents."),
        md("## 3. The part of the advice that is true\n\n"
           f"DCA really does lower risk. Its outcomes land in a band a little over **half** "
           f"as wide as the lump sum's (dispersion ratio **{R['disp_ratio']:.3f}**), and its "
           f"worst twelve months lose **{R['dca_worst']:.1f}%** where the lump sum's lose "
           f"**{R['lump_worst']:.1f}%**. That is genuine — but look at *how* it is bought: "
           "by holding less equity for longer. The calm is a property of the **weight**, "
           "not of the schedule, so you can have exactly the same ride by picking a smaller "
           "stock weight and investing it at once — no twelve-month queue required. What you "
           "cannot do is keep the five cents *and* the calm: those cents are the extra risk. "
           "Section 3b puts a number on that."),
        code(
            "R = " + repr({k: R[k] for k in ("sd_lump", "sd_dca", "disp_ratio",
                                             "lump_worst", "dca_worst", "worst_gap",
                                             "best_gap")}) + "\n"
            "print(f\"spread of outcomes : lump {R['sd_lump']:.4f}   DCA {R['sd_dca']:.4f}   \"\n"
            "      f\"-> ratio {R['disp_ratio']:.3f}\")\n"
            "print(f\"worst 12 months    : lump {R['lump_worst']:.1f}%   DCA {R['dca_worst']:.1f}%\")\n"
            "print(f\"worst / best gap   : {R['worst_gap']:+.1f} cents / {R['best_gap']:+.1f} cents\")\n"
            "print( '                     -> going all-in is right three times in four,'\n"
            "       ' and expensively wrong the fourth')"
        ),
        md("## 3b. So what are those five cents, really?\n\n"
           "Not a timing skill — an ownership difference. Spread over twelve months, "
           f"the DCA plan owns the market for only **{R['em_w']:.1f}%** of the year on "
           "average (the first tranche is invested all twelve months, the last one for "
           "none of it). So put that number on the table and race DCA against the *boring* "
           f"portfolio: **{R['em_w']:.1f}% in stocks and the rest in T-bills, bought at "
           "the start and left alone**.\n\n"
           f"The five cents vanish: **{R['em_gap']:+.2f} cents**, a coin flip "
           f"({R['em_win']:.1f}% of months), with an interval of "
           f"[{R['em_lo']:+.2f}, {R['em_hi']:+.2f}] cents around zero. DCA was never "
           "buying worse prices — it was just owning less stock, and it was paid "
           "accordingly."),
        code(
            "R = " + repr({k: R[k] for k in ("mean_gap", "em_w", "em_gap", "em_win",
                                             "em_lo", "em_hi", "sd_dca", "em_sd")}) + "\n"
            "print(f\"lump sum (100% invested) vs DCA        : {R['mean_gap']:+.2f} cents\")\n"
            "print(f\"static {R['em_w']:.1f}% stocks + bills vs DCA : {R['em_gap']:+.2f} cents  \"\n"
            "      f\"({R['em_win']:.1f}% of months, CI [{R['em_lo']:+.2f}, {R['em_hi']:+.2f}])\")\n"
            "print(f\"same calm ride                         : spread {R['em_sd']:.4f} vs DCA {R['sd_dca']:.4f}\")\n"
            "print()\n"
            "print('-> the twelve tranches add nothing the weight had not already given you')"
        ),
        md("## 4. \"But surely DCA wins when the market looks expensive?\"\n\n"
           "That is the version of the advice worth testing, so we tested it — with "
           "hindsight, which is the friendliest possible framing. Split the start months by "
           "how stretched SPY was against its own three-year average (a **price proxy**, not "
           "CAPE), and separately by whether you were starting inside a drawdown:\n\n"
           "| Starting from | lump wins | average gap |\n|---|--:|--:|\n"
           f"| a cheap market | {R['cheap_win']:.1f}% | {R['cheap_gap']:+.2f}c |\n"
           f"| a middling market | {R['mid_win']:.1f}% | {R['mid_gap']:+.2f}c |\n"
           f"| a **stretched** market | {R['str_win']:.1f}% | {R['str_gap']:+.2f}c |\n"
           f"| **10%+ below the high** | {R['dd_win']:.1f}% | {R['dd_gap']:+.2f}c |\n\n"
           "The advantage shrinks where the story says it should — and **never crosses "
           "zero**. There was no starting condition, in nineteen years, in which drip-feeding "
           "beat sending it. Averaging in is not buying the fear; it is just owning less."),
        md("## 5. Two more things worth knowing\n\n"
           f"**Taking longer costs more.** Three tranches costs {R['tr3_gap']:+.2f}c, six "
           f"{R['tr6_gap']:+.2f}c, twelve {R['tr12_gap']:+.2f}c, twenty-four "
           f"**{R['tr24_gap']:+.2f}c** — and each of them smooths the ride a little more. "
           "The comfort and the bill scale together; pick your point on that line knowingly.\n\n"
           f"**Bonds are a different question.** Put the windfall into a Treasury sleeve "
           f"(IEF) instead and the lump sum's edge collapses to {R['ief_gap']:+.2f}c, no "
           "longer distinguishable from zero. The prize was never a trick of timing — it was "
           "the risk premium of whatever you were buying, collected sooner. Small premium, "
           "small prize."),
        md("## 6. Live check — the machinery has no thumb on the scale (offline synthetic)\n\n"
           "Before believing a result that agrees this neatly with theory, make the harness "
           "prove it can say the opposite. On simulated tapes with a **planted** premium the "
           "lump sum must win; with **no** premium the answer must sit on a coin flip; on a "
           "**falling** tape DCA must win. Twelve independent 25-year paths per world."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from lump_vs_dca import data, strategy as st\n"
            "\n"
            "def small(signal_strength, seed):\n"
            "    return data.synthetic_daily(n_years=15, signal_strength=signal_strength, seed=seed)\n"
            "\n"
            "for ss, label in [(1.0, 'rising  (premium planted)'),\n"
            "                  (0.0, 'flat    (the null)     '),\n"
            "                  (-1.0, 'falling (premium is negative)')]:\n"
            "    c = st.synthetic_control(ss, seeds=range(934, 940), synth=small)\n"
            "    print('%s : mean gap %+6.2f cents, lump wins on %3.0f%% of paths'\n"
            "          % (label, c['mean_gap_cents'], 100 * c['frac_seeds_lump_wins']))"
        ),
        md("## Verdict\n\n"
           f"- **Signal — Real.** The lump sum wins **{R['win']:.1f}%** of start months by "
           f"**{R['mean_gap']:+.2f} cents on the dollar**, HAC *t* = **{R['t_hac']:+.2f}**, "
           f"bootstrap CI [{R['boot_lo']:+.2f}, {R['boot_hi']:+.2f}] clear of zero, same sign "
           f"in every cut and on the {R['long_start'][:4]}-2026 history "
           f"(*t* = {R['long_t']:+.2f}). The advice is not merely unproven; it is backwards.\n"
           f"- **Tradability — Mirage.** There is nothing to bank. Match the exposure — a "
           f"static {R['em_w']:.1f}% stock portfolio against DCA — and the whole gap is "
           f"{R['em_gap']:+.2f} cents (*t* = {R['em_t']:+.2f}, interval "
           f"[{R['em_lo']:+.2f}, {R['em_hi']:+.2f}]). It is beta you were always paid for, "
           f"which is why it dies on bonds ({R['ief_gap']:+.2f}c) and died through the 2000s "
           f"({R['dec00_gap']:+.2f}c, *t* = {R['dec00_t']:+.2f}).\n"
           f"- **Does DCA lower risk? — Confirmed.** Dispersion ratio {R['disp_ratio']:.3f}, "
           f"worst window {R['dca_worst']:.1f}% vs {R['lump_worst']:.1f}%. Real, and "
           "available more cheaply by owning less stock."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 934 — Lump Sum vs DCA — the teardown\n\n"
           "Terminal-wealth race over every start month: win rate with a Wilson interval, "
           "the overlap-corrected HAC *t* on the mean gap, a non-overlapping check, a "
           "12-month block bootstrap, the **exposure-matched control** that decides what the "
           "gap is made of, the dispersion read, the conditional cuts, era and decade cuts, "
           "a long-history extension, the cost/ticket/tranche sweeps, the bond-heavy variant "
           "and the two-sided synthetic control.\n\n"
           "**Design.** $1, twelve months, both arms valued on the same terminal date. "
           "Purchases are decided at a month-end close and executed at the next trading "
           "day's close — one lag, applied identically to the lump-sum buy and to all "
           "twelve tranches. Uninvested DCA balance accrues **BIL total return**. Costs "
           "one-way × NAV; no shorting anywhere, so no borrow leg.\n\n"
           "Real numbers frozen from `docs/results.md` (SPY fingerprint `%s`, IEF `%s`), "
           "as-of 2026-06-30." % (R["fp_spy"], R["fp_ief"])),
        code("R = %r" % (R,)),
        md("## 1. Headline and its inference\n\n"
           "Monthly starts with twelve-month horizons overlap by up to eleven months, so "
           "the naive *t* is meaningless. Three independent corrections:"),
        code(
            "print(f\"n windows {R['n_win']}   {R['first_start']} -> {R['last_end']}\")\n"
            "print(f\"win rate      : {R['win']:.1f}%  Wilson 95% CI [{R['win_lo']:.1f}%, {R['win_hi']:.1f}%]\")\n"
            "print(f\"mean gap      : {R['mean_gap']:+.2f}c   median {R['median_gap']:+.2f}c   sd {R['sd_gap']:.2f}c\")\n"
            "print(f\"HAC t (12 lag): {R['t_hac']:+.2f}\")\n"
            "print(f\"non-overlap t : {R['t_nonoverlap']:+.2f}   (every 12th start, averaged over all 12 phases)\")\n"
            "print(f\"boot 95% CI   : [{R['boot_lo']:+.2f}, {R['boot_hi']:+.2f}]c   share<0 {R['boot_neg']:.1f}%\")"
        ),
        md("> 💡 **In plain words** — three different ways of admitting the windows share "
           "tape, and all three still say the lump sum wins by about five cents on the "
           "dollar. Whether those cents are *timing* is section 1b."),
        md("## 1b. Exposure or timing? — the control that sets the Tradability stamp\n\n"
           "A DCA schedule is not only later into the market, it is **less in** it. Tranche "
           "*j* is invested for (n−j)/n of the window, so its time-weighted exposure is the "
           "analytic **(n+1)/2n = 13/24 = 54.2%** — nothing fitted, nothing peeked at. A raw "
           "terminal-wealth race therefore pits a full-beta portfolio against a half-beta "
           "one, and any positive premium hands the lump sum a win it did not earn by "
           "timing.\n\n"
           "The control: race DCA against a **static** portfolio holding that same 54.2% in "
           "the asset and the rest in the same BIL leg for the whole window. Both arms are "
           "then read excess of the **same** cash leg — never one raw against one excess."),
        code(
            "print(f\"headline  lump vs DCA       : {R['mean_gap']:+.2f}c  \"\n"
            "      f\"HAC t {R['t_hac']:+.2f}  CI [{R['boot_lo']:+.2f}, {R['boot_hi']:+.2f}]\")\n"
            "print(f\"matched   {R['em_w']:.1f}% static vs DCA : {R['em_gap']:+.2f}c  \"\n"
            "      f\"HAC t {R['em_t']:+.2f}  CI [{R['em_lo']:+.2f}, {R['em_hi']:+.2f}]  \"\n"
            "      f\"(non-overlap t {R['em_t_no']:+.2f}, win {R['em_win']:.1f}%)\")\n"
            "print(f\"          {R['dm_w']:.1f}% static vs DCA : {R['dm_gap']:+.2f}c  \"\n"
            "      f\"HAC t {R['dm_t']:+.2f}  CI [{R['dm_lo']:+.2f}, {R['dm_hi']:+.2f}]  \"\n"
            "       '(dispersion-matched weight = IN-SAMPLE fit, sensitivity only)')\n"
            "print()\n"
            "print(f\"reward per unit dispersion, both excess of the same cash leg:\")\n"
            "print(f\"   lump {R['xs_lump']:+.2f}% -> {R['rd_lump']:.3f}   \"\n"
            "      f\"DCA {R['xs_dca']:+.2f}% -> {R['rd_dca']:.3f}\")\n"
            "print(f\"long tape ({R['long_start'][:4]}-2026), same control: \"\n"
            "      f\"{R['long_em_gap']:+.2f}c (t {R['long_em_t']:+.2f})\")"
        ),
        md("> 💡 **In plain words** — every cent of the headline is the extra beta. Own "
           "DCA's average weight for twelve months and you land where DCA lands, with the "
           "same dispersion, without the schedule. That is the whole finding, and it is why "
           "the second stamp is red."),
        md("## 2. Dispersion — the risk claim, stated separately from the return claim"),
        code(
            "print(f\"terminal wealth  lump: mean {R['lump_mean']:+.2f}%  sd {R['sd_lump']:.4f}  worst {R['lump_worst']:.2f}%\")\n"
            "print(f\"                 DCA : mean {R['dca_mean']:+.2f}%  sd {R['sd_dca']:.4f}  worst {R['dca_worst']:.2f}%\")\n"
            "print(f\"dispersion ratio (DCA/lump) = {R['disp_ratio']:.3f}\")\n"
            "print(f\"gap distribution: p5 {R['p05']:+.2f}c   p95 {R['p95']:+.2f}c   \"\n"
            "      f\"worst {R['worst_gap']:+.2f}c   best {R['best_gap']:+.2f}c\")"
        ),
        md("The dispersion cut is real and mechanical: DCA's average equity exposure over "
           "the window is ~54% of the lump sum's, so its variance is roughly a third and its "
           "SD roughly six tenths. Nothing about *prices* is being improved."),
        md("## 3. Conditional cuts — hindsight terciles, PROXY valuation\n\n"
           "`stretch0` = level ÷ trailing 3-year mean at the start date. It is a **price "
           "proxy** for expensive, not CAPE (no earnings data enters this study), and the "
           "terciles are cut in-sample, so this answers \"did such a state exist?\" and not "
           "\"could you trade it?\". The trailing mean needs a three-year runway inside the "
           "sample, so the terciles cover 181 of the 217 starts; the drawdown cut uses all "
           "217."),
        code(
            "rows = [('cheap tercile', R['cheap_n'], R['cheap_win'], R['cheap_gap'], R['cheap_t'], R['cheap_ret']),\n"
            "        ('middle tercile', R['mid_n'], R['mid_win'], R['mid_gap'], R['mid_t'], R['mid_ret']),\n"
            "        ('stretched tercile', R['str_n'], R['str_win'], R['str_gap'], R['str_t'], R['str_ret']),\n"
            "        ('start >=10% below high', R['dd_n'], R['dd_win'], R['dd_gap'], R['dd_t'], R['dd_ret']),\n"
            "        ('start near the highs', R['hi_n'], R['hi_win'], R['hi_gap'], R['hi_t'], R['hi_ret'])]\n"
            "print(f\"{'start state':24s} {'n':>4s} {'win%':>7s} {'gap':>8s} {'t':>7s} {'12m ret':>9s}\")\n"
            "for name, n, w, g, t, r in rows:\n"
            "    print(f\"{name:24s} {n:4d} {w:6.1f}% {g:+7.2f}c {t:+7.2f} {r:+8.1f}%\")\n"
            "print('\\nno cut crosses zero; the weakest t is the drawdown cut (n=59)')"
        ),
        md("> 💡 **In plain words** — the one place the advice should shine, starting from "
           "a drawdown, is the one place the evidence is thinnest (*t* = +1.50) — and even "
           "there the sign is still against it."),
        md("## 4. Era cuts, decades, and the long-history extension\n\n"
           "BIL's 2007 inception gates the honest cash leg. The extension backwards runs "
           "under a **0% cash ASSUMPTION**, and is floored at a **pinned 2000-01-03** rather "
           "than at whatever SPY history the *shared* `studies/_cache` happens to hold — "
           "other studies re-pull the same tickers with their own start dates, and a "
           "robustness number that moves with someone else's fetch is not a robustness "
           "number. (An earlier draft of this study quoted a 1993-start run; it no longer "
           "reproduces, and the conclusion it supported was wrong. The decade table is what "
           "replaced it.)"),
        code(
            "print(f\"2007-06..2016-12 (n={R['era_e_n']}): win {R['era_e_win']:.1f}%  gap {R['era_e_gap']:+.2f}c  t={R['era_e_t']:+.2f}\")\n"
            "print(f\"2017-01..2025-06 (n={R['era_l_n']}): win {R['era_l_win']:.1f}%  gap {R['era_l_gap']:+.2f}c  t={R['era_l_t']:+.2f}\")\n"
            "print()\n"
            "print(f\"long history {R['long_start'][:4]}-2026 (n={R['long_n']}, 0% cash ASSUMPTION, pinned start): \"\n"
            "      f\"win {R['long_win']:.1f}%  gap {R['long_gap']:+.2f}c  HAC t {R['long_t']:+.2f}  \"\n"
            "      f\"non-overlap t {R['long_t_no']:+.2f}\")\n"
            "for tag, n, w, g, t, r in [('2000s', R['dec00_n'], R['dec00_win'], R['dec00_gap'], R['dec00_t'], R['dec00_ret']),\n"
            "                           ('2010s', R['dec10_n'], R['dec10_win'], R['dec10_gap'], R['dec10_t'], R['dec10_ret']),\n"
            "                           ('2020s', R['dec20_n'], R['dec20_win'], R['dec20_gap'], R['dec20_t'], R['dec20_ret'])]:\n"
            "    print(f\"   {tag}: n={n:3d}  win {w:5.1f}%  gap {g:+6.2f}c  t={t:+5.2f}  mean 12m SPY {r:+5.1f}%\")\n"
            "print(f\"   exposure-matched on the long tape: {R['long_em_gap']:+.2f}c (t {R['long_em_t']:+.2f})\")"
        ),
        md("> 💡 **In plain words** — across the lost decade, the decade whose equities paid "
           f"**{R['dec00_ret']:+.1f}%** a year over twelve months, the lump sum's advantage is "
           f"**{R['dec00_gap']:+.2f}c** with *t* = **{R['dec00_t']:+.2f}**: gone, not merely "
           "smaller. The advantage is the size of the premium that showed up — exactly what "
           "the exposure control in section 1b predicts."),
        md("## 5. Costs — the one that cancels and the one that does not\n\n"
           "Both arms put the same $1 to work, so a **proportional** one-way cost is charged "
           "on the same notional in each and cancels in the difference. A **fixed ticket** "
           "does not: DCA pays it twelve times. Ticket amounts and the $10,000 windfall are "
           "ASSUMPTIONS, swept rather than asserted."),
        code(
            "print(f\"proportional: 0 bps {R['cost0']:+.2f}c  ->  25 bps {R['cost25']:+.2f}c   (cancels, by construction)\")\n"
            "print(f\"fixed ticket on a $10,000 windfall:\")\n"
            "for tk, g in [(0, R['tick0']), (1, R['tick1']), (5, R['tick5']), (10, R['tick10'])]:\n"
            "    print(f\"   ${tk:2d}/trade -> mean gap {g:+.2f}c\")\n"
            "print(f\"\\ncash-leg assumption: 0% cash gives {R['zero_cash_gap']:+.2f}c (t={R['zero_cash_t']:+.2f}) \"\n"
            "      f\"vs {R['mean_gap']:+.2f}c crediting the real BIL path\")"
        ),
        md("## 6. Tranche length and the bond-heavy variant"),
        code(
            "print('tranches   win%     gap    dispersion ratio')\n"
            "for k, w, g, d in [(3, R['tr3_win'], R['tr3_gap'], R['tr3_disp']),\n"
            "                   (6, R['tr6_win'], R['tr6_gap'], R['tr6_disp']),\n"
            "                   (12, R['tr12_win'], R['tr12_gap'], R['tr12_disp']),\n"
            "                   (24, R['tr24_win'], R['tr24_gap'], R['tr24_disp'])]:\n"
            "    print(f'   {k:2d}     {w:5.1f}%  {g:+6.2f}c        {d:.3f}')\n"
            "print()\n"
            "print(f\"IEF (7-10y Treasuries): win {R['ief_win']:.1f}%  gap {R['ief_gap']:+.2f}c  \"\n"
            "      f\"t={R['ief_t']:+.2f}  boot CI [{R['ief_lo']:+.2f}, {R['ief_hi']:+.2f}] -> includes zero\")\n"
            "print(f\"   exposure-matched on IEF: {R['ief_em_gap']:+.2f}c (t {R['ief_em_t']:+.2f}, \"\n"
            "      f\"CI [{R['ief_em_lo']:+.2f}, {R['ief_em_hi']:+.2f}])\")\n"
            "print()\n"
            "print('tranche count only moves average exposure: (n+1)/2n = 66.7 / 58.3 / 54.2 / 52.1%')"
        ),
        md("> 💡 **In plain words** — the bond result is the honest boundary of the finding: "
           "the lump sum's edge *is* the premium of the asset you are buying, so where the "
           "premium is thin the edge is statistically invisible. And the tranche sweep is the "
           "same fact along a second axis — return and dispersion move together at every "
           "point on that line because the only thing the tranche count changes is the "
           "average weight."),
        md("## 7. The two-sided synthetic control (live, offline)\n\n"
           "A harness that only ever crowns the lump sum would look 'validated' by a rising "
           "tape. The control runs three planted worlds and must get all three right. One "
           "25-year path of a 16%-vol asset carries a ~3 pp standard error on its own drift, "
           "so the control is read across seeds — a single draw genuinely can land the wrong "
           "side of a 7 pp planted premium."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from lump_vs_dca import data, strategy as st\n"
            "\n"
            "def small(signal_strength, seed):\n"
            "    return data.synthetic_daily(n_years=15, signal_strength=signal_strength, seed=seed)\n"
            "\n"
            "for ss in (1.0, 0.0, -1.0):\n"
            "    c = st.synthetic_control(ss, seeds=range(934, 940), synth=small)\n"
            "    print(f\"ss={ss:+.1f}: mean gap {c['mean_gap_cents']:+6.2f}c (sd {c['sd_gap_cents']:.2f}), \"\n"
            "          f\"mean win rate {c['mean_win_rate']:.1%}, lump wins on {c['frac_seeds_lump_wins']:.0%} of seeds\")"
        ),
        md("The 12-seed version quoted in `docs/results.md` reads "
           f"**{R['syn_pl_gap']:+.2f}c** ({R['syn_pl_seeds']} seeds) on the planted world, "
           f"**{R['syn_nl_gap']:+.2f}c** ({R['syn_nl_seeds']}) on the null and "
           f"**{R['syn_fa_gap']:+.2f}c** ({R['syn_fa_seeds']}) on the falling tape — the "
           "harness crowns DCA when DCA deserves it."),
        md("## Verdict\n\n"
           f"- **Signal — Real.** Win rate **{R['win']:.1f}%** (Wilson CI "
           f"[{R['win_lo']:.1f}%, {R['win_hi']:.1f}%], clear of 50%), mean gap "
           f"**{R['mean_gap']:+.2f}c/$1**, HAC *t* **{R['t_hac']:+.2f}**, non-overlapping *t* "
           f"**{R['t_nonoverlap']:+.2f}**, bootstrap CI "
           f"**[{R['boot_lo']:+.2f}, {R['boot_hi']:+.2f}]**. Sign-stable across both eras "
           f"({R['era_e_gap']:+.2f}c / {R['era_l_gap']:+.2f}c), all five conditional cuts, "
           f"every cost assumption, and the {R['long_n']}-window {R['long_start'][:4]}-2026 "
           f"extension (*t* = {R['long_t']:+.2f}). The synthetic control recovers a planted "
           "effect of either sign and is silent on the null. The stamp is about the "
           "terminal-wealth gap existing — not about what causes it.\n"
           f"- **Tradability — Mirage.** Not eaten by costs (they cancel): there is nothing "
           f"to eat. Matched for exposure, the gap is **{R['em_gap']:+.2f}c** with *t* = "
           f"**{R['em_t']:+.2f}** and a bootstrap CI of "
           f"**[{R['em_lo']:+.2f}, {R['em_hi']:+.2f}]**, and the two arms earn the same "
           f"reward per unit of dispersion ({R['rd_lump']:.3f} vs {R['rd_dca']:.3f}, both "
           "excess of the same cash leg). This is the desk's textbook Mirage — *it's just "
           f"beta you were always paid for*: it vanishes in the 2000s ({R['dec00_gap']:+.2f}c, "
           f"*t* = {R['dec00_t']:+.2f}) and on bonds ({R['ief_gap']:+.2f}c, CI through zero). "
           "Free and correct to act on **if** the equity weight is already chosen; not an "
           "edge to bank, scale or repeat.\n"
           f"- **Does DCA lower risk? — Confirmed.** Dispersion ratio "
           f"**{R['disp_ratio']:.3f}**, worst window {R['dca_worst']:.2f}% vs "
           f"{R['lump_worst']:.2f}%. It is bought with average exposure, not with better "
           f"entry prices: the static {R['em_w']:.1f}% portfolio has the same dispersion "
           f"({R['em_sd']:.4f} vs {R['sd_dca']:.4f}) and the same terminal wealth, without "
           "the schedule."),
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
