"""Generate the two narrative notebooks for Study 925 (Front-End Trend).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
single frozen ``R`` dict below, which mirrors ``docs/results.md``; the only live cells run
the fast offline synthetic control, and they are never presented under a real-tape banner.
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
# Frozen real-tape headline — the single source of truth, mirroring docs/results.md.
# ^IRX 3-month change -> TLT / BIL, raced excess-of-cash against static IEF.
# 2007-05-30 -> 2026-06-30, 2 bps one-way, one execution lag.
# --------------------------------------------------------------------------- #
R = dict(
    start="2007-05-30", end="2026-06-30", n_days=4800, fp="c05691fe8719",
    lookback=63, cost_bps=2.0, in_frac=0.478, n_switches=321, switches_per_year=17,

    sw_sharpe=0.154, sw_sharpe_gross=0.185, sw_cagr=1.09, sw_vol=10.98, sw_dd=-28.2,
    sw_t=0.73,
    ief_sharpe=0.293, ief_cagr=1.82, ief_vol=6.99, ief_dd=-23.9, ief_t=1.37,
    tlt_sharpe=0.184, tlt_cagr=1.66, tlt_vol=15.24, tlt_dd=-48.4, tlt_t=0.89,
    rnd_sharpe=-0.316, rnd_cagr=-3.74, rnd_dd=-47.2,

    adv=-0.139, t_vs_ief=-0.20, t_vs_tlt=-0.52,
    adv_one_seed=0.469, t_one_seed=2.01,

    ci_sw_lo=-0.283, ci_sw_hi=0.554, ci_sw_neg=24.3,
    ci_ief_lo=-0.136, ci_ief_hi=0.704, ci_ief_neg=9.3,
    ci_diff_lo=-0.498, ci_diff_hi=0.221, ci_diff_neg=76.4,

    rnd_net_adv=0.268, rnd_net_sd=0.177, rnd_net_t=1.20, rnd_net_share=20,
    rnd_gross_adv=0.060, rnd_gross_sd=0.175, rnd_gross_t=0.29, rnd_gross_share=0,
    rnd_flips=2400,

    # the flip-free exposure control: constant 47.8% TLT / 52.2% BIL, rebalanced daily
    blend_w=47.8, blend_turnover=91,
    blend_net_sharpe=0.182, blend_net_adv=-0.028, blend_net_t=0.23,
    blend_gross_sharpe=0.184, blend_gross_adv=0.001, blend_gross_t=0.43,

    on_bp=1.69, on_n=2263, on_t=0.87, off_bp=0.59, off_n=2473, off_t=0.36,
    spread_bp=1.09, spread_hac=0.43, spread_welch=0.39,

    era_e_n=2100, era_e_ief=0.73, era_e_sw=0.28, era_e_adv=-0.452, era_e_t=-0.71,
    era_e_frac=58, era_e_dd_ief=-10.4, era_e_dd_sw=-27.1,
    era_l_n=2635, era_l_ief=-0.11, era_l_sw=0.03, era_l_adv=0.138, era_l_t=0.44,
    era_l_frac=39, era_l_dd_ief=-23.9, era_l_dd_sw=-28.2,

    lb21_adv=0.056, lb21_t=0.99, lb42_adv=-0.178, lb42_t=-0.41,
    lb63_adv=-0.139, lb63_t=-0.20, lb126_adv=-0.122, lb126_t=-0.11,
    lb252_adv=0.083, lb252_t=1.25,

    cost0_adv=-0.108, cost0_t=-0.01, cost10_adv=-0.264, cost10_t=-0.96,
    cost25_adv=-0.496, cost25_t=-2.35,

    # every calendar year, no selection: (year, switch %, IEF %, TLT %, gap %, in-duration %)
    # gaps are quoted from verify.py (computed unrounded), not re-derived from the columns
    years=((2007, 6.7, 5.5, 7.3, 1.2, 99), (2008, 26.1, 17.9, 34.0, 8.2, 78),
           (2009, -21.5, -6.6, -21.8, -14.9, 67), (2010, -2.4, 9.4, 9.0, -11.7, 47),
           (2011, 26.4, 15.6, 34.0, 10.7, 73), (2012, 4.5, 3.7, 2.4, 0.8, 20),
           (2013, -14.8, -6.1, -13.4, -8.7, 59), (2014, 24.9, 9.1, 27.3, 15.9, 73),
           (2015, -11.3, 1.5, -1.8, -12.8, 38), (2016, 2.7, 1.0, 1.2, 1.7, 24),
           (2017, -1.4, 2.6, 9.2, -3.9, 2), (2018, 1.7, 1.0, -1.6, 0.7, 0),
           (2019, 6.9, 8.0, 14.1, -1.2, 67), (2020, 6.1, 10.0, 18.2, -3.9, 83),
           (2021, -15.2, -3.3, -4.6, -11.9, 56), (2022, 1.4, -15.2, -31.2, 16.5, 0),
           (2023, 21.0, 3.6, 2.8, 17.3, 15), (2024, -9.4, -0.6, -8.1, -8.8, 71),
           (2025, 8.8, 8.0, 4.2, 0.8, 72), (2026, 1.8, -0.1, 1.0, 1.9, 47)),
    yrs_beat=11, yrs_total=20, gap_mean=-0.10, gap_median=0.76,

    y2022_sw=1.4, y2022_ief=-15.2, y2022_tlt=-31.2, y2022_sessions=0, y2022_total=251,
    y2023_sw=21.0, y2023_ief=3.6, y2023_days=37, y2023_tlt_window=14.7,
    y2023_strat_window=16.2, y2023_bil=4.9,

    shy_sharpe=0.331, shy_adv=0.061, shy_t=1.43, shy_frac=85, shy_switches=68,
    shy_gross_rnd=0.172, shy_gross_rnd_t=1.36,
    shy_blend_w=85.2, shy_blend_sharpe=0.177,
    shy_blend_adv=0.153, shy_blend_t=1.92,
    shy_blend_gross_adv=0.158, shy_blend_gross_t=1.97,
    shy_blend_early_adv=0.005, shy_blend_early_t=0.51, shy_blend_early_frac=99,
    shy_blend_late_adv=0.243, shy_blend_late_t=1.52,
    shy_blend_ex22_adv=0.156, shy_blend_ex22_t=1.24,
    syn_blend_adv=1.545, syn_blend_t=4.45,

    syn_pl_adv=1.78, syn_pl_t=4.66, syn_pl_mean=2.15, syn_pl_fire=10,
    syn_nl_mean=-0.050, syn_nl_sd=0.203, syn_nl_fire=0,
)

BANNER = (
    f"*Real-tape numbers below are the frozen headline from "
    f"[`docs/results.md`](../docs/results.md) — SHY/IEF/TLT/BIL total-return closes plus "
    f"the `^IRX` yield, {R['start']} → {R['end']} ({R['n_days']:,} days), Fingerprint "
    f"`{R['fp']}`, as-of 2026-06-30. Live cells run the offline synthetic control only, "
    f"and are labelled as such.*"
)


# --------------------------------------------------------------------------- #
# 01 — for the curious
# --------------------------------------------------------------------------- #
HEADER_01 = f"""# Study 925 — Front-End Trend

**If short-term interest rates have been falling for three months, should you go and buy
the long bond?**

Central banks do not change their minds at random. When the Federal Reserve starts
cutting, it usually keeps cutting for a year or more; when it starts hiking, likewise. So
here is a rule almost anyone can follow. Look at the yield on the 3-month Treasury bill
(`^IRX`). Compare it with where it stood three months ago.

- **Rates falling?** Own long-dated Treasuries (**TLT**) — falling rates push bond prices
  up, and the longest bonds move the most.
- **Rates rising or flat?** Sit in Treasury bills (**BIL**) and collect the short rate.

Check daily, act the next morning, pay 2 bps of your portfolio each time you switch.

The benchmark it has to beat is deliberately unglamorous: **just holding IEF**, the 7-10
year Treasury fund, from start to finish and never thinking about rates again.

{BANNER}
"""


def build_curious():
    cells = [
        md(HEADER_01),

        md("## 1. The scoreboard\n\n"
           "Everything below is measured **above cash** — we subtract what Treasury bills "
           "themselves paid. That matters enormously here: for a third of this sample bills "
           "paid nearly nothing and for another stretch they paid 5%, and a rule that spends "
           "half its life in bills would otherwise get credit for the Fed's generosity rather "
           "than for its own cleverness.\n\n"
           "> 🔬 **For the quants** — every arm is a daily simple excess return over BIL's "
           "own total return, so the cash leg cancels in every pairwise difference and the "
           "Sharpe race is like-for-like."),
        code(
            "R = " + repr(R) + "\n"
            "rows = [('switch (the rule)', R['sw_sharpe'], R['sw_cagr'], R['sw_vol'], R['sw_dd']),\n"
            "        ('static IEF',       R['ief_sharpe'], R['ief_cagr'], R['ief_vol'], R['ief_dd']),\n"
            "        ('static TLT',       R['tlt_sharpe'], R['tlt_cagr'], R['tlt_vol'], R['tlt_dd']),\n"
            "        ('random control',   R['rnd_sharpe'], R['rnd_cagr'], float('nan'), R['rnd_dd'])]\n"
            "print(f\"{'arm':18s}{'exSharpe':>10s}{'exCAGR':>9s}{'vol':>8s}{'worst loss':>12s}\")\n"
            "for name, sh, cg, vol, dd in rows:\n"
            "    v = '   n/a' if vol != vol else f'{vol:6.2f}%'\n"
            "    print(f'{name:18s}{sh:+10.3f}{cg:+8.2f}%{v:>8s}{dd:+11.1f}%')\n"
            "print()\n"
            "print(f\"advantage of the rule over just holding IEF: {R['adv']:+.3f} Sharpe\")\n"
            "print(f\"how sure are we?  HAC t = {R['t_vs_ief']:+.2f}   (the desk's bar is |t| >= 2)\")"
        ),

        md(f"## 2. The rule is not just worse — it is worse in every way\n\n"
           f"A timing rule is allowed to earn less if it makes the ride smoother. This one "
           f"does not. Against simply holding IEF it delivers a **lower** return above cash "
           f"({R['sw_cagr']:.2f}% vs {R['ief_cagr']:.2f}%), **more** volatility "
           f"({R['sw_vol']:.1f}% vs {R['ief_vol']:.1f}%) and a **deeper** worst loss "
           f"({R['sw_dd']:.1f}% vs {R['ief_dd']:.1f}%) — while asking you to trade about "
           f"**{R['switches_per_year']} times a year**. There is no dimension on which it wins."),

        md(f"## 3. So why does everyone remember this rule working?\n\n"
           f"Because of two spectacular years.\n\n"
           f"**2022.** Rates rose relentlessly all year. The rule was in Treasury bills for "
           f"**{R['y2022_total'] - R['y2022_sessions']} of {R['y2022_total']} sessions** — it "
           f"never once owned duration — and finished **{R['y2022_sw']:+.1f}%**, while IEF lost "
           f"**{R['y2022_ief']:.1f}%** and TLT lost **{R['y2022_tlt']:.1f}%**. A 16-point win "
           f"in the year that broke the 60/40 portfolio.\n\n"
           f"**2023.** The rule owned long bonds for just **{R['y2023_days']} sessions** — "
           f"6 November to 29 December — during which **TLT rose "
           f"{R['y2023_tlt_window']:+.1f}%**; the rule's own return over that stretch was "
           f"{R['y2023_strat_window']:+.1f}% (it sat out the one down day inside the window). "
           f"Bills paid {R['y2023_bil']:.1f}% for the other eleven months. Result: "
           f"**{R['y2023_sw']:+.1f}%** against IEF's {R['y2023_ief']:+.1f}%.\n\n"
           f"Those two years are genuine — not artefacts, not look-ahead. So here is the "
           f"whole scorecard, every year, nothing left out."),
        code(
            "print(f\"{'year':>6s}{'switch':>10s}{'IEF':>10s}{'TLT':>10s}{'gap':>10s}{'in duration':>13s}\")\n"
            "for y, sw, ief, tlt, gap, frac in R['years']:\n"
            "    print(f'{y:>6d}{sw:+9.1f}%{ief:+9.1f}%{tlt:+9.1f}%{gap:+9.1f}%{frac:>12d}%')\n"
            "print()\n"
            "print(f\"beats IEF in {R['yrs_beat']} of {R['yrs_total']} years; \"\n"
            "      f\"mean gap {R['gap_mean']:+.2f} pp/yr, median {R['gap_median']:+.2f} pp/yr\")\n"
            "print('a coin-toss hit rate with a slightly negative mean -- what no information looks like.')"
        ),

        md("### The wins that are not timing\n\n"
           "Look at 2011 (+10.7 over IEF) and 2014 (+15.9). Those are bigger gaps than most of "
           "the losses, and they have nothing to do with reading the front end: the rule simply "
           "happened to be holding TLT while TLT returned +34% and +27%. It **lagged its own "
           "instrument** in both years — it beat IEF only because twenty-year duration beats "
           "eight-year duration in a bond bull market. The bill for that beta arrives in 2009 "
           "(−14.9), 2015 (−12.8) and 2021 (−11.9), when it held the same duration into the "
           "sell-off.\n\n"
           "Strip out the beta and the timing record really is 2022 and 2023 — two episodes in "
           "nineteen years."),

        md(f"## 4. \"But it beat a coin flip!\" — the trap\n\n"
           f"The standard fairness check is a **random control**: a rule that owns long bonds "
           f"on *random* days, but just as often ({R['in_frac']:.0%} of the time). If our rule "
           f"beats it, the *choice of days* was worth something.\n\n"
           f"On one random draw ours wins by a convincing **{R['adv_one_seed']:+.3f}** Sharpe. "
           f"But a coin flipped 4,800 times is one draw of many, so we ran **30** of them — and "
           f"then ran them again with the trading costs switched off:\n\n"
           f"| | advantage over random | how sure |\n"
           f"|---|--:|--:|\n"
           f"| after costs | **{R['rnd_net_adv']:+.3f}** | *t* = {R['rnd_net_t']:+.2f} |\n"
           f"| **before costs** | **{R['rnd_gross_adv']:+.3f}** | *t* = {R['rnd_gross_t']:+.2f} |\n\n"
           f"The win evaporates when costs come off. The reason is mundane: a coin flip changes "
           f"its mind about **{R['rnd_flips']:,}** times over the sample; our rule changes its "
           f"mind **{R['n_switches']}** times. We were not out-*picking* the coin, we were "
           f"out-*sitting* it. That is a real advantage over a maniac, but it is not a "
           f"forecast.\n\n"
           f"**The fair version of that test.** Give the rule an opponent with its exact average "
           f"exposure and *no* switching at all: a fixed **{R['blend_w']:.1f}% TLT / "
           f"{100 - R['blend_w']:.1f}% bills** portfolio, held throughout. Before costs the two "
           f"are level ({R['sw_sharpe_gross']:+.3f} vs {R['blend_gross_sharpe']:+.3f} — a gap "
           f"of **{R['blend_gross_adv']:+.3f}**); after costs the "
           f"lazy blend is **ahead** ({R['blend_net_sharpe']:+.3f} vs {R['sw_sharpe']:+.3f}). "
           f"Nineteen years of decisions bought exactly nothing that owning the average would "
           f"not have given you for free."),

        md(f"## 5. And it depends entirely on a number we made up\n\n"
           f"Why three months? No reason — it just sounds sensible. Try the neighbours:\n\n"
           f"| lookback | advantage over IEF |\n"
           f"|---|--:|\n"
           f"| 1 month | {R['lb21_adv']:+.3f} |\n"
           f"| 2 months | {R['lb42_adv']:+.3f} |\n"
           f"| **3 months** (ours) | **{R['lb63_adv']:+.3f}** |\n"
           f"| 6 months | {R['lb126_adv']:+.3f} |\n"
           f"| 12 months | {R['lb252_adv']:+.3f} |\n\n"
           f"The answer changes **sign three times** across a perfectly reasonable grid, and "
           f"never gets big enough to matter in either direction. When the conclusion depends "
           f"on an arbitrary knob, the conclusion is the knob."),

        md("## 6. Is the test itself any good? (live, offline, synthetic)\n\n"
           "Before believing a null result you have to prove the measuring device works. So we "
           "build a **fake world** where short rates genuinely do trend — where this month's "
           "rate move really does predict next month's — and check that the rule finds it. Then "
           "we build a world where rate moves are pure coin flips and check that the rule finds "
           "nothing.\n\n"
           "> 🔬 **For the quants** — the knob is the AR(1) coefficient on daily rate "
           "*increments*, with unconditional rate volatility held fixed, so the two worlds "
           "differ in predictability and not in risk. Nothing in this cell touches the real "
           "tape."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..', '..', '..')))\n"
            "from front_end_trend import data, strategy as st\n"
            "\n"
            "trending = st.synthetic_detect(data.synthetic_daily(signal_strength=1.0, seed=925)[0])\n"
            "coinflip = st.synthetic_detect(data.synthetic_daily(signal_strength=0.0, seed=925)[0])\n"
            "print('SYNTHETIC (not the real tape)')\n"
            "print(f\"world where rates really trend : Sharpe advantage \"\n"
            "      f\"{trending['excess_sharpe_adv']:+.2f}  -> the rule finds it easily\")\n"
            "print(f\"world of pure coin-flip rates  : Sharpe advantage \"\n"
            "      f\"{coinflip['excess_sharpe_adv']:+.2f}  -> and it correctly finds nothing\")\n"
            "print()\n"
            "print('So the measuring device works. The blank on the real tape is the tape\\'s.')"
        ),

        md(f"## Verdict\n\n"
           f"- **Signal — None.** Against simply holding IEF, the front-end trend rule is behind "
           f"by **{R['adv']:+.3f}** Sharpe with a *t* of {R['t_vs_ief']:+.2f} — no signal, and "
           f"the point estimate has the wrong sign. The answer flips with the lookback window, "
           f"flips between the first and second halves of the sample, and the one apparently "
           f"impressive result (beating a coin flip) turns out to be a turnover artefact that "
           f"vanishes before costs. Against a lazy fixed-weight portfolio of the same average "
           f"exposure the rule is level before costs and behind after them.\n"
           f"- **Tradability — Mirage.** Lower return, higher volatility, deeper drawdown, "
           f"~{R['switches_per_year']} round trips a year, and it beats its benchmark in only "
           f"{R['yrs_beat']} of {R['yrs_total']} years for a mean gap of "
           f"{R['gap_mean']:+.2f} pp/yr. You would be paying to make your bond sleeve worse.\n"
           f"- **What is true.** The rule really did sit out all of 2022 and really did hold "
           f"duration through the late-2023 rally. Both are honest — and its other big years "
           f"(2011, 2014) are long-duration beta, not timing. A nineteen-year record whose "
           f"timing content is two loud episodes is the exact shape of luck, and the tape here "
           f"cannot tell it from anything else."),
    ]
    nb = new_notebook()
    nb["cells"] = cells
    return nb


# --------------------------------------------------------------------------- #
# 02 — for the quants
# --------------------------------------------------------------------------- #
HEADER_02 = f"""# Study 925 — Front-End Trend — the teardown

**Signal.** `s_t = 1[ y_t − y_{{t−63}} < 0 ]` where `y` is `^IRX`, the CBOE 13-week
Treasury bill discount yield in percent — a **price-only yield level**, never a return.
`s` is shifted forward one day: exactly **one** execution lag, formed through the close of
`t`, acted at `t+1`, applied in `rate_trend_signal` and nowhere else.

**Book.** `s = 1` → long TLT; `s = 0` → long BIL. Binary, unlevered, **no short leg
anywhere**, so no borrow is paid or assumed. Costs are `cost_bps × 1e-4 × |Δs|` — one-way,
on NAV, on switch days only.

**Race.** All arms excess-of-cash (minus BIL's own total return): the switch, static IEF
(7-10y, the honest static-duration benchmark), static TLT (the instrument held always), a
frequency-matched random control, and a **matched constant-weight blend** carrying the
rule's average exposure with zero flips. Inference: Newey-West HAC *t* on the daily return
*difference* (Jobson-Korkie form), a paired circular block bootstrap on the Sharpe
difference, an era cut, a lookback sweep, a cost sweep, a HAC day-level conditional test,
and a seed-swept random control run gross **and** net.

{BANNER}
"""


def build_quants():
    cells = [
        md(HEADER_02),
        code("R = " + repr(R) + "\n"
             "print(f\"frozen headline: {R['start']} -> {R['end']}, n={R['n_days']}, fp={R['fp']}\")\n"
             "print(f\"in duration {R['in_frac']:.1%} of days, {R['n_switches']} switches \"\n"
             "      f\"(~{R['switches_per_year']}/yr)\")"),

        md("## 1. The excess-of-cash race\n\n"
           "> 💡 **In plain words** — every arm has T-bill returns subtracted, so nobody gets "
           "paid for merely sitting in cash while cash yielded 5%."),
        code(
            "print(f\"{'arm':16s}{'exSharpe':>10s}{'exCAGR':>9s}{'vol':>8s}{'maxDD':>9s}{'HAC t':>8s}\")\n"
            "print(f\"{'switch':16s}{R['sw_sharpe']:+10.3f}{R['sw_cagr']:+8.2f}%{R['sw_vol']:7.2f}%\"\n"
            "      f\"{R['sw_dd']:+8.1f}%{R['sw_t']:+8.2f}\")\n"
            "print(f\"{'static IEF':16s}{R['ief_sharpe']:+10.3f}{R['ief_cagr']:+8.2f}%{R['ief_vol']:7.2f}%\"\n"
            "      f\"{R['ief_dd']:+8.1f}%{R['ief_t']:+8.2f}\")\n"
            "print(f\"{'static TLT':16s}{R['tlt_sharpe']:+10.3f}{R['tlt_cagr']:+8.2f}%{R['tlt_vol']:7.2f}%\"\n"
            "      f\"{R['tlt_dd']:+8.1f}%{R['tlt_t']:+8.2f}\")\n"
            "print(f\"{'random (1 seed)':16s}{R['rnd_sharpe']:+10.3f}{R['rnd_cagr']:+8.2f}%\"\n"
            "      f\"{'':8s}{R['rnd_dd']:+8.1f}%\")\n"
            "print()\n"
            "print(f\"adv vs static IEF : {R['adv']:+.3f}   HAC t on daily return diff = {R['t_vs_ief']:+.2f}\")\n"
            "print(f\"adv vs static TLT :         HAC t on daily return diff = {R['t_vs_tlt']:+.2f}\")\n"
            "print('the switch arm is worse on Sharpe, MORE volatile and DEEPER in drawdown than IEF.')"
        ),

        md("## 2. Paired block bootstrap on the Sharpe *difference*\n\n"
           "2,000 draws, 21-day circular blocks, **same block indices drawn for both arms** so "
           "the paired structure and the shared cash leg survive resampling."),
        code(
            "print(f\"switch exSharpe      {R['sw_sharpe']:+.3f}  95% CI [{R['ci_sw_lo']:+.3f}, {R['ci_sw_hi']:+.3f}]  share<0 {R['ci_sw_neg']:.1f}%\")\n"
            "print(f\"static IEF exSharpe  {R['ief_sharpe']:+.3f}  95% CI [{R['ci_ief_lo']:+.3f}, {R['ci_ief_hi']:+.3f}]  share<0 {R['ci_ief_neg']:.1f}%\")\n"
            "print(f\"DIFFERENCE           {R['adv']:+.3f}  95% CI [{R['ci_diff_lo']:+.3f}, {R['ci_diff_hi']:+.3f}]  share<0 {R['ci_diff_neg']:.1f}%\")\n"
            "print('the difference CI straddles zero with 76% of the mass on the losing side.')"
        ),

        md("## 3. The random control is a turnover trap\n\n"
           "A frequency-matched Bernoulli control flips roughly `2p(1−p)N` times. With "
           "`p = 0.478` and `N = 4,800` that is ~2,400 flips against the rule's 321 — so at any "
           "positive cost the control pays ~7× the friction it never chose. One seed is also "
           "one draw. Sweep 30 seeds, and run it **gross**.\n\n"
           "> 💡 **In plain words** — the rule did not pick better days than the coin. It just "
           "traded less often than the coin, and we were charging the coin for that."),
        code(
            "print(f\"single seed (the tempting number): adv {R['adv_one_seed']:+.3f}  t {R['t_one_seed']:+.2f}\")\n"
            "print()\n"
            "print(f\"{'':10s}{'adv mean':>10s}{'sd':>8s}{'t mean':>9s}{'share t>=2':>12s}\")\n"
            "print(f\"{'net 2bps':10s}{R['rnd_net_adv']:+10.3f}{R['rnd_net_sd']:8.3f}\"\n"
            "      f\"{R['rnd_net_t']:+9.2f}{R['rnd_net_share']:>11d}%\")\n"
            "print(f\"{'GROSS':10s}{R['rnd_gross_adv']:+10.3f}{R['rnd_gross_sd']:8.3f}\"\n"
            "      f\"{R['rnd_gross_t']:+9.2f}{R['rnd_gross_share']:>11d}%\")\n"
            "print()\n"
            "print(f\"~{R['rnd_flips']:,} control flips vs {R['n_switches']} rule switches -> \"\n"
            "      'the net advantage is friction accounting, not timing.')"
        ),

        md("### 3b. The control the random one should have been\n\n"
           "Remove the friction confound entirely: hold the rule's **average** exposure with "
           "**no flips at all** — a constant `w = 47.8%` TLT / 52.2% BIL book rebalanced daily "
           "(drift trades only, ~91%/yr of turnover, charged at the same cost the rule pays). "
           "Same duration on average, no regime switching, no seed to draw. "
           "`strategy.matched_blend_race` does this in one call.\n\n"
           "> 💡 **In plain words** — if the timing is worth anything, it has to beat simply "
           "owning that average all the time."),
        code(
            "print(f\"{'':10s}{'switch':>10s}{'blend':>10s}{'adv':>10s}{'ret-diff t':>12s}\")\n"
            "print(f\"{'net 2bps':10s}{R['sw_sharpe']:+10.3f}{R['blend_net_sharpe']:+10.3f}\"\n"
            "      f\"{R['blend_net_adv']:+10.3f}{R['blend_net_t']:+12.2f}\")\n"
            "print(f\"{'GROSS':10s}{R['sw_sharpe_gross']:+10.3f}\"\n"
            "      f\"{R['blend_gross_sharpe']:+10.3f}{R['blend_gross_adv']:+10.3f}\"\n"
            "      f\"{R['blend_gross_t']:+12.2f}\")\n"
            "print()\n"
            "print('gross: the same number to two decimals. net: the flip-free blend wins.')\n"
            "print('321 timing decisions in nineteen years are collectively worth zero.')"
        ),

        md("## 4. Day-level conditional test — does the signal carry *any* information?\n\n"
           "Strip out portfolio construction and cost entirely: take the excess-of-cash daily "
           "return of TLT and split it by what the rule said the day before.\n\n"
           "The spread's *t* is the **HAC** one — the Newey-West *t* of "
           "`e·(s − p)/(p(1−p))`, whose mean is exactly the difference in bucket means. Welch's "
           "iid *t* is printed for reference only: the buckets are selected by a signal that "
           "persists for months, so daily returns inside a bucket are not independent draws."),
        code(
            "print(f\"own-duration days : {R['on_bp']:+.2f} bp/d  n={R['on_n']}  HAC t {R['on_t']:+.2f}\")\n"
            "print(f\"sit-in-bills days : {R['off_bp']:+.2f} bp/d  n={R['off_n']}  HAC t {R['off_t']:+.2f}\")\n"
            "print(f\"spread            : {R['spread_bp']:+.2f} bp/d   HAC t = {R['spread_hac']:+.2f}\"\n"
            "      f\"   (Welch t = {R['spread_welch']:+.2f}, iid, reference only)\")\n"
            "print('right sign, no significance -- 4,736 daily observations cannot separate the buckets.')"
        ),

        md("## 5. Era cut and lookback sweep — the sign is not stable\n\n"
           "The signal is built on the full yield history then sliced, so the late era does not "
           "pay a fresh 63-day warmup (that would be a different rule, not a robustness check); "
           "no look-ahead is introduced because the signal still uses past yields only."),
        code(
            "print(f\"2007-2015 (n={R['era_e_n']}): IEF {R['era_e_ief']:+.2f} / switch {R['era_e_sw']:+.2f}  \"\n"
            "      f\"adv {R['era_e_adv']:+.3f} (t={R['era_e_t']:+.2f})  in-duration {R['era_e_frac']}%\")\n"
            "print(f\"2016-2026 (n={R['era_l_n']}): IEF {R['era_l_ief']:+.2f} / switch {R['era_l_sw']:+.2f}  \"\n"
            "      f\"adv {R['era_l_adv']:+.3f} (t={R['era_l_t']:+.2f})  in-duration {R['era_l_frac']}%\")\n"
            "print('  -> sign flips between halves; neither half is significant.')\n"
            "print()\n"
            "for lb, a, t in [(21, R['lb21_adv'], R['lb21_t']), (42, R['lb42_adv'], R['lb42_t']),\n"
            "                 (63, R['lb63_adv'], R['lb63_t']), (126, R['lb126_adv'], R['lb126_t']),\n"
            "                 (252, R['lb252_adv'], R['lb252_t'])]:\n"
            "    mark = '  <- headline' if lb == 63 else ''\n"
            "    print(f'lookback {lb:3d}d: adv {a:+.3f} (t={t:+.2f}){mark}')\n"
            "print('  -> three sign changes across a reasonable grid. The conclusion is the knob.')"
        ),

        md("## 6. Cost sweep and the SHY cross-check\n\n"
           "No short leg, so no borrow: the only friction is switch cost, one-way × NAV. The "
           "cross-check swaps the `^IRX` yield change for **SHY's 12-minus-1-month total-return "
           "momentum** — the same bet read off a tradable instrument instead of an index yield."),
        code(
            "print(f\"gross (0 bps): adv {R['cost0_adv']:+.3f} (t={R['cost0_t']:+.2f})  <- already negative\")\n"
            "print(f\"10 bps       : adv {R['cost10_adv']:+.3f} (t={R['cost10_t']:+.2f})\")\n"
            "print(f\"25 bps       : adv {R['cost25_adv']:+.3f} (t={R['cost25_t']:+.2f})  <- the only |t|>=2 here, wrong way\")\n"
            "print()\n"
            "print(f\"SHY 12-1 momentum variant: switch {R['shy_sharpe']:+.3f}, adv {R['shy_adv']:+.3f} \"\n"
            "      f\"(t={R['shy_t']:+.2f}), in-duration {R['shy_frac']}%, {R['shy_switches']} switches\")\n"
            "print(f\"  gross vs random over 30 seeds: {R['shy_gross_rnd']:+.3f} (t={R['shy_gross_rnd_t']:+.2f})\")\n"
            "print(f\"  vs ITS matched blend (w={R['shy_blend_w']:.1f}% TLT): \"\n"
            "      f\"adv {R['shy_blend_adv']:+.3f} (t={R['shy_blend_t']:+.2f}) net, \"\n"
            "      f\"{R['shy_blend_gross_adv']:+.3f} (t={R['shy_blend_gross_t']:+.2f}) GROSS\")\n"
            "print(f\"    2007-2015 (in duration {R['shy_blend_early_frac']}% of days): \"\n"
            "      f\"adv {R['shy_blend_early_adv']:+.3f} (t={R['shy_blend_early_t']:+.2f})\")\n"
            "print(f\"    2016-2026: adv {R['shy_blend_late_adv']:+.3f} (t={R['shy_blend_late_t']:+.2f})\"\n"
            "      f\"   |  full sample minus 2022: adv {R['shy_blend_ex22_adv']:+.3f} \"\n"
            "      f\"(t={R['shy_blend_ex22_t']:+.2f})\")"
        ),

        md(f"**The strongest number in the study, and why it is still not a badge.** The SHY "
           f"variant beats a fixed-weight book of its *own* average exposure by "
           f"**{R['shy_blend_adv']:+.3f}** net and **{R['shy_blend_gross_adv']:+.3f}** gross "
           f"(*t* = {R['shy_blend_t']:+.2f} / {R['shy_blend_gross_t']:+.2f}) — not a turnover "
           f"artefact, and the right comparison for a rule that is long duration "
           f"{R['shy_frac']}% of the time. It is still under |*t*| ≥ 2, and it is not stable: "
           f"in 2007-2015 the rule held duration on {R['shy_blend_early_frac']}% of days — it "
           f"made no decisions to be right about — and its advantage there is "
           f"{R['shy_blend_early_adv']:+.3f}. Everything it has comes from the second half, "
           f"where the 2022 hiking cycle dominates; drop 2022 and the *t* falls to "
           f"{R['shy_blend_ex22_t']:+.2f}. One alternative specification, one era, under the "
           f"bar. Interesting; not evidence."),

        md("## 7. Live synthetic control — the harness is unbiased (offline)\n\n"
           "The knob is the AR(1) coefficient `phi` on the short rate's daily *increments*, with "
           "the unconditional daily rate volatility rescaled by `sqrt(1 − phi²)` so the planted "
           "and null worlds carry identical risk and differ only in predictability. Duration legs "
           "are priced off the same path with `r = carry/252 − D·dy + noise`.\n\n"
           "> 💡 **In plain words** — we prove the detector works by planting the effect and "
           "watching it fire, then removing it and watching it go quiet."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..', '..', '..')))\n"
            "import numpy as np\n"
            "from front_end_trend import data, strategy as st\n"
            "\n"
            "pl = st.synthetic_detect(data.synthetic_daily(signal_strength=1.0, seed=925)[0])\n"
            "print('SYNTHETIC (never supports the real-tape stamp)')\n"
            "print(f\"planted trending front end: adv {pl['excess_sharpe_adv']:+.3f} (t={pl['t_diff_vs_mid']:+.2f})\")\n"
            "nl = np.array([st.synthetic_detect(f)['excess_sharpe_adv']\n"
            "               for f, _ in data.synthetic_panel(n_seeds=10, signal_strength=0.0)])\n"
            "pp = np.array([st.synthetic_detect(f)['excess_sharpe_adv']\n"
            "               for f, _ in data.synthetic_panel(n_seeds=10, signal_strength=1.0)])\n"
            "print(f\"planted x10: mean {pp.mean():+.3f}, adv>=0.4 in {(pp >= 0.4).sum()}/10\")\n"
            "print(f\"null    x10: mean {nl.mean():+.3f} (sd {nl.std(ddof=1):.3f}), \"\n"
            "      f\"|adv|>=0.4 in {(np.abs(nl) >= 0.4).sum()}/10\")\n"
            "\n"
            "cd_pl = st.conditional_day_test(*[data.synthetic_daily(signal_strength=1.0, seed=925)[0][c]\n"
            "                                  for c in ('rate', 'long', 'cash')])\n"
            "cd_nl = st.conditional_day_test(*[data.synthetic_daily(signal_strength=0.0, seed=925)[0][c]\n"
            "                                  for c in ('rate', 'long', 'cash')])\n"
            "print(f\"day-level spread  planted {cd_pl['spread_bp']:+.2f} bp/d \"\n"
            "      f\"(HAC t {cd_pl['spread_hac_t']:+.2f})   null {cd_nl['spread_bp']:+.2f} bp/d \"\n"
            "      f\"(HAC t {cd_nl['spread_hac_t']:+.2f})\")\n"
            "\n"
            "sig_pl = st.rate_trend_signal(data.synthetic_daily(signal_strength=1.0, seed=925)[0]['rate'])\n"
            "f_pl = data.synthetic_daily(signal_strength=1.0, seed=925)[0]\n"
            "mb = st.matched_blend_race(f_pl['long'], f_pl['mid'], f_pl['cash'], sig_pl)\n"
            "print(f\"matched-blend control on the planted world: adv {mb['adv_vs_blend']:+.3f} \"\n"
            "      f\"(t {mb['t_diff_vs_blend']:+.2f}) -> real timing does beat its own average exposure\")"
        ),

        md(f"## Verdict\n\n"
           f"- **Signal — None.** Excess-of-cash Sharpe advantage over static IEF is "
           f"**{R['adv']:+.3f}** (HAC *t* = {R['t_vs_ief']:+.2f}); versus static TLT the "
           f"difference *t* is {R['t_vs_tlt']:+.2f}. The paired bootstrap difference CI is "
           f"**[{R['ci_diff_lo']:+.3f}, {R['ci_diff_hi']:+.3f}]** with {R['ci_diff_neg']:.0f}% of "
           f"draws negative. The sign flips across eras ({R['era_e_adv']:+.2f} / "
           f"{R['era_l_adv']:+.2f}) and three times across the lookback grid. The day-level "
           f"conditional spread is {R['spread_bp']:+.2f} bp/d at HAC *t* = {R['spread_hac']:+.2f}. "
           f"The apparent win over a random control is a **turnover artefact**: "
           f"{R['rnd_net_adv']:+.3f} net but only **{R['rnd_gross_adv']:+.3f} gross** across 30 "
           f"seeds (mean *t* {R['rnd_gross_t']:+.2f}, {R['rnd_gross_share']}/30 clearing *t* ≥ 2); "
           f"against the flip-free blend of identical exposure it is "
           f"**{R['blend_gross_adv']:+.3f} gross, {R['blend_net_adv']:+.3f} net**. "
           f"Nothing clears |*t*| ≥ 2 in the claim's direction. The synthetic control fires at "
           f"adv **{R['syn_pl_mean']:+.2f}** ({R['syn_pl_fire']}/10) on a planted trending front "
           f"end and sits at **{R['syn_nl_mean']:+.3f}** ({R['syn_nl_fire']}/10) on the null, so "
           f"the blank is the tape's, not the harness's; the blend control separates there too "
           f"(adv {R['syn_blend_adv']:+.3f}, *t* {R['syn_blend_t']:+.2f}).\n"
           f"- **The one number that argues back.** The SHY 12-1 variant beats *its* matched "
           f"blend by {R['shy_blend_adv']:+.3f} net / {R['shy_blend_gross_adv']:+.3f} gross "
           f"(*t* {R['shy_blend_t']:+.2f} / {R['shy_blend_gross_t']:+.2f}) — the best result on "
           f"this desk's tape for the idea, under the bar, and worth "
           f"{R['shy_blend_early_adv']:+.3f} in the era where the rule held duration "
           f"{R['shy_blend_early_frac']}% of days.\n"
           f"- **Tradability — Mirage.** Lower excess Sharpe, higher vol ({R['sw_vol']:.1f}% vs "
           f"{R['ief_vol']:.1f}%), deeper drawdown ({R['sw_dd']:.1f}% vs {R['ief_dd']:.1f}%), "
           f"~{R['switches_per_year']} round trips a year, negative gross and more negative at "
           f"every cost tested, ahead of IEF in {R['yrs_beat']}/{R['yrs_total']} calendar years "
           f"for a mean gap of {R['gap_mean']:+.2f} pp/yr. Nothing to bank.\n"
           f"- **The honest residue.** 2022 ({R['y2022_sessions']} of {R['y2022_total']} sessions "
           f"in duration, +16.5 points over IEF) and late 2023 ({R['y2023_days']} sessions long "
           f"while TLT rose {R['y2023_tlt_window']:+.1f}%) are real, and they are the whole "
           f"*timing* record: the other big winning years (2011 +10.7, 2014 +15.9) are 20-year "
           f"duration beta the rule rode up and then paid back in 2009, 2015 and 2021. Two "
           f"episodes in nineteen years is not a sample; it is the shape luck takes when it is "
           f"loud."),
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
