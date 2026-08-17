"""Generate the two narrative notebooks for Study 930 (When-Issued Window).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below, which mirrors ``docs/results.md``; the only live cells run the
fast synthetic control, and they are labelled as synthetic wherever they appear. No cell
sits under a real-tape banner while computing synthetic numbers.
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
# 26 US spin-offs, distributions 2012-05-01 -> 2025-02-24, alpha vs SPY,
# total-return closes, 10 bps one-way x NAV + 40 bp/yr SPY borrow, as-of 2026-06-30.
# --------------------------------------------------------------------------- #
R = dict(
    asof="2026-06-30", fp="e78725a81ebe", n_events=26, n_tickers=52,
    first="2012-05-01", last="2025-02-24", per_year=2.0, n_delisted=4,
    # HEADLINE legs are GROSS: the child alpha is negative, so charging a long position's
    # frictions to it would inflate the effect. Net is quoted beside, never headlined.
    par_mean=-1.45, par_med=-2.28, par_hit=0.42, par_t=-0.26, par_hac=-0.47,
    par_ci_lo=-11.43, par_ci_hi=9.47,
    c5_mean=-4.54, c5_med=-4.36, c5_hit=0.31, c5_t=-2.73, c5_hac=-3.02,
    c5_ci_lo=-7.67, c5_ci_hi=-1.42, c5_hit_lo=0.17, c5_hit_hi=0.50, c5_sd=8.47,
    c10_mean=-3.47, c10_med=-3.09, c10_hit=0.31, c10_t=-1.61, c10_hac=-2.20,
    c10_ci_lo=-7.61, c10_ci_hi=0.63,
    c21_mean=-1.02, c21_med=-1.87, c21_hit=0.46, c21_t=-0.41, c21_hac=-0.58,
    c21_ci_lo=-5.75, c21_ci_hi=3.73,
    comb_mean=1.88, comb_med=2.80, comb_hit=0.62, comb_t=1.28, comb_hac=1.44,
    comb_ci_lo=-1.02, comb_ci_hi=4.69,
    # the same five legs, NET of all frictions (long version)
    n_par=-2.10, n_par_t=-0.38, n_c5=-4.77, n_c5_t=-2.87, n_c5_hac=-3.17,
    n_c10=-3.71, n_c10_t=-1.71, n_c21=-1.27, n_c21_t=-0.51, n_comb=1.62, n_comb_t=1.10,
    # five pre-specified legs = one family (Westfall-Young max-|t| bootstrap)
    fw_legs=5, fw_p_single=0.0088, fw_p_fwe=0.0452,
    # eras (gross) [net]
    era_e_n=6, era_e_mean=-3.77, era_e_t=-2.91,
    era_l_n=20, era_l_mean=-4.77, era_l_t=-2.23,
    # benchmark swap (gross)
    iwm_mean=-4.41, iwm_t=-2.48, mdy_mean=-4.64, mdy_t=-2.62,
    # robustness (gross)
    jk_lo=-3.36, jk_hi=-2.44, dec_n=19, dec_mean=-5.09, dec_t=-2.88,
    cost0=-4.56, cost0_t=-2.74, cost25=-5.07, cost25_t=-3.05, cost50=-5.57, cost50_t=-3.35,
    # the load-bearing anchor, checked rather than asserted
    anchor_exact=25, anchor_tail=24, anchor_tail_lo=3, anchor_tail_hi=15,
    # proxy sweeps
    rw_m3=-6.85, rw_m3_t=-2.74, rw_m1=-7.76, rw_m1_t=-3.78,
    rw_p1=-3.31, rw_p1_t=-1.76, rw_p3=2.24, rw_p3_t=1.62,
    ann_m5=-3.78, ann_p5=-0.89,
    ratio_half=1.75, ratio_two=1.41, ratio_eq=0.37,
    # exploratory shape
    caar_m3=4.09, caar_m1=1.86, caar_p1=-1.08, caar_p2=-3.90, caar_p3=-6.09,
    caar_p5=-4.48, caar_p10=-3.49, caar_p21=-0.79, caar_n=24,
    h3_mean=-5.91, h3_t=-3.65, h3_hac=-4.49, h3_hit=0.12, h63_mean=-0.74, h63_t=-0.19,
    # short-side economics
    sh0=4.32, sh0_t=2.60, sh0_hit=0.65, sh0_ci_lo=1.20, sh0_ci_hi=7.45,
    sh25=3.82, sh25_t=2.30, sh25_ci_lo=0.70, sh25_ci_hi=6.95,
    sh50=3.33, sh50_t=2.00, sh50_hit=0.62, sh50_ci_lo=0.20, sh50_ci_hi=6.46,
    sh100=2.33, sh100_t=1.40, sh100_ci_lo=-0.79, sh100_ci_hi=5.46,
    # daily sleeve — the calendar is the study's own window, not the shared cache's depth
    sl5_days=118, sl5_of=3228, sl5_bps=-75.9, sl5_t=-1.84, sl5_lo=-133.6, sl5_hi=-9.8,
    sl21_days=453, sl21_of=3244, sl21_bps=-4.2, sl21_t=-0.27,
    # synthetic control
    syn_planted_child=6.63, syn_planted_child_t=4.43, syn_planted_par=21.72,
    syn_planted_par_t=2.60, syn_null_fire=0,
)

BANNER = f"""*Real-tape numbers below are frozen from [`docs/results.md`](../docs/results.md)
(Fingerprint `{R['fp']}`, as-of {R['asof']}). Live cells run the **synthetic** control only
and say so.*"""

SETUP = (
    "import os, sys\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
    "import numpy as np\n"
    "from when_issued import data, strategy as st\n"
)


def build_curious():
    nb = new_notebook()
    cells = [
        md(f"""# Study 930 — When-Issued Window

**When a company splits in two, is the freshly cut-loose half a bargain?**

Between the day a break-up is announced and the day the new company actually trades
normally, there is a strange in-between zone. The child already has a price — a
*when-issued* price, good only if the split completes — but nobody quite owns it yet.
Folklore says this zone is full of free money: the parent re-rates upward on the news,
and the child gets dumped on day one by index funds and pension managers who never chose
to own it, so you can pick it up cheap.

We took **{R['n_events']} liquid US spin-offs ({R['first'][:4]}–{R['last'][:4]})** and
measured all three pieces of that story against the S&P 500.

{BANNER}"""),
        md("## 1. Three windows, three stories\n\n"
           "- **The parent**, from the day after the break-up is announced to the last close "
           "before the child trades normally. Does dismantling a conglomerate re-rate what is "
           "left behind?\n"
           "- **The child**, over its first 5, 10 and 21 normal trading sessions. This is the "
           "famous one: Joel Greenblatt's *forced sellers*.\n"
           "- **Both halves together**, held a month. Is the sum of the parts worth more than "
           "the whole was?\n\n"
           "Everything is measured against SPY over exactly the same calendar days, and we "
           "always wait one full session before acting — so no result here depends on being "
           "quicker than a newspaper. The figures below are *before* trading costs on purpose: "
           "the answer turns out to be a **loss**, and charging costs to a loss would only "
           "flatter the finding. Costs appear in §5, on the trade you would actually place."),
        code(
            "R = dict(par_mean=%r, par_t=%r, c5_mean=%r, c5_t=%r, c21_mean=%r, c21_t=%r,\n"
            "         comb_mean=%r, comb_t=%r)\n"
            "print('parent, announcement -> distribution : %%+6.2f%%%%  (t = %%+.2f)'\n"
            "      %% (R['par_mean'], R['par_t']))\n"
            "print('child, first 5 sessions             : %%+6.2f%%%%  (t = %%+.2f)   <-- the only real one'\n"
            "      %% (R['c5_mean'], R['c5_t']))\n"
            "print('child, first 21 sessions            : %%+6.2f%%%%  (t = %%+.2f)'\n"
            "      %% (R['c21_mean'], R['c21_t']))\n"
            "print('parent + child, first 21 sessions   : %%+6.2f%%%%  (t = %%+.2f)'\n"
            "      %% (R['comb_mean'], R['comb_t']))"
            % (R["par_mean"], R["par_t"], R["c5_mean"], R["c5_t"],
               R["c21_mean"], R["c21_t"], R["comb_mean"], R["comb_t"])
        ),
        md(f"## 2. The surprise: the child is not cheap, it is *falling*\n\n"
           f"Buy the child at the close of its very first normal session and hold it a week, "
           f"and you lose **{abs(R['c5_mean']):.1f}%** to the index. Only "
           f"**{int(round(R['c5_hit'] * R['n_events']))} of {R['n_events']}** spin-offs beat "
           f"SPY over that week. That is not a rounding error — it is the exact opposite of "
           f"what the forced-seller story predicts.\n\n"
           f"The parent's long wait from announcement to distribution turns out to be nothing "
           f"much at all ({R['par_mean']:+.2f}%, *t* = {R['par_t']:+.2f}), and putting the two "
           f"halves back together does nothing either ({R['comb_mean']:+.2f}%, "
           f"*t* = {R['comb_t']:+.2f}). Four of the five things we measured are noise. The "
           f"fifth is real and points the wrong way."),
        md(f"""## 3. What is probably going on

Think about who *has* to trade on the day of the distribution. Every index fund that owns
the parent is handed the child. The child immediately joins the parent's index — so those
funds cannot sell it, they have to **hold** it, and the funds that track the child's *new*
index have to **buy** it. On day one there is a wall of price-insensitive buying.

Then it stops. Over the next three sessions the child gives back
**{abs(R['caar_p3']):.1f}%** relative to the index, and by the end of the month it has
clawed most of that back ({R['caar_p21']:+.2f}%). That shape — a pop, a slide, a partial
recovery — is the classic footprint of mechanical demand, not of forced selling.

> 🔬 **For the quants:** the trough sits at three sessions ({R['h3_mean']:+.2f}%,
> *t* = {R['h3_t']:+.2f}, only {int(round(R['h3_hit'] * R['n_events']))}/{R['n_events']}
> positive) and is gone by a quarter ({R['h63_mean']:+.2f}%, *t* = {R['h63_t']:+.2f}). Only
> 5 / 10 / 21 sessions were pre-specified, so the 3-session number carries an unpriced
> multiple-comparison penalty and is a shape diagnostic, never the headline."""),
        md(f"## 4. Is it just a few disasters dragging the average?\n\n"
           f"No. Drop any single spin-off from the sample and the *t*-statistic stays between "
           f"**{R['jk_lo']:+.2f}** and **{R['jk_hi']:+.2f}**. Split the sample in 2020 and it "
           f"is negative in both halves ({R['era_e_mean']:+.2f}% before, "
           f"{R['era_l_mean']:+.2f}% after). Spin-off children are usually smaller than the "
           f"S&P 500, so we re-ran it against the small-cap and mid-cap indices too: "
           f"{R['iwm_mean']:+.2f}% against IWM, {R['mdy_mean']:+.2f}% against MDY. Same "
           f"answer everywhere."),
        md(f"""## 5. So can you trade it?

Only by **shorting** — and that is where it gets awkward. Shorting a company that started
trading five days ago means borrowing shares that barely exist yet: tiny float, no settled
lending supply, most of the stock sitting in index funds. We swept the borrow cost from
free to punitive:

| You pay to borrow | You keep, per trade |
|---|--:|
| nothing (fantasy) | **{R['sh0']:+.2f}%** (*t* = {R['sh0_t']:+.2f}) |
| 25%/yr | {R['sh25']:+.2f}% |
| 50%/yr | {R['sh50']:+.2f}% |
| 100%/yr | {R['sh100']:+.2f}% — and now the confidence interval crosses zero |

And there are only about **{R['per_year']:.0f} of these a year**. Two trades a year, each
one a coin-flip with a {R['c5_sd']:.1f}-point standard deviation, in a name your broker may
simply refuse to locate. Real effect, awful business."""),
        md("## 6. Live check — the measuring stick is straight (synthetic, not the real tape)\n\n"
           "Before believing a negative number, check the ruler. We build a **fake** world of "
           "spin-offs where we *plant* a known parent run-up and a known child drift, and run "
           "the identical code on it. It must find what we planted — and find nothing when we "
           "plant nothing."),
        code(
            SETUP +
            "\n# SYNTHETIC world (not the real tape) — the machinery proof.\n"
            "px, ev, truth = data.synthetic_daily(signal_strength=1.0, seed=930)\n"
            "planted = st.synthetic_detect(px, ev)\n"
            "px0, ev0, _ = data.synthetic_daily(signal_strength=0.0, seed=930)\n"
            "null = st.synthetic_detect(px0, ev0)\n"
            "print('planted world: we buried a %+.2f%% child drift ->'\n"
            "      ' the estimator found %+.2f%% (t = %+.2f)'\n"
            "      % (truth['planted_child_alpha_21d']*100,\n"
            "         planted['child21_mean_pct'], planted['child21_t']))\n"
            "print('null world   : we buried nothing            ->'\n"
            "      ' the estimator found %+.2f%% (t = %+.2f)'\n"
            "      % (null['child21_mean_pct'], null['child21_t']))"
        ),
        md(f"""## Verdict

- **Signal — Real, with the sign inverted, and only just.** The child's first five
  regular-way sessions under-perform SPY by **{R['c5_mean']:+.2f}%** (*t* = {R['c5_t']:+.2f},
  HAC {R['c5_hac']:+.2f}, bootstrap CI [{R['c5_ci_lo']:+.2f}, {R['c5_ci_hi']:+.2f}]% clear of
  zero), in both eras and against three different benchmarks. That is a genuine effect —
  it is just not the bargain the folklore advertises. The parent run-up and the
  sum-of-the-parts pop are both nothing. Two things to keep it in proportion: we measured
  five things, and after paying for all five looks the odds of seeing this by luck are
  **{R['fw_p_fwe']:.1%}** — real, but a whisker inside the line, not a mile. And the sample
  is {R['n_events']} hand-picked liquid spins, with {R['n_delisted']} more lost because the
  child was later acquired and delisted; treat the size of the number with more suspicion
  than its sign.
- **Tradability — Fragile.** Harvesting it means shorting a five-day-old spin-off about
  twice a year. Survives a 50%/yr borrow ({R['sh50']:+.2f}%), does not survive 100%/yr
  ({R['sh100']:+.2f}%, CI through zero), and starting three sessions late turns
  {R['sh0']:+.2f}% into {-R['rw_p3']:+.2f}%."""),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md(f"""# Study 930 — When-Issued Window — the teardown

Three windows around a US spin-off, on one hand-compiled table of **{R['n_events']}** liquid
US events (distributions {R['first']} → {R['last']}, ≈{R['per_year']:.0f}/yr,
{R['n_tickers']} tickers):

1. **parent**, close of the session *after* the announcement → last close *before* the
   child's first regular-way session (never crossing the distribution, so Yahoo's
   inconsistent spin-off adjustment never enters);
2. **child**, first **5 / 10 / 21** regular-way sessions, entered at the *first*
   regular-way close so the day-one print is excluded;
3. **parent + child**, ratio-weighted, 21 sessions from that same close.

Every leg is `asset − SPY` over identical calendar sessions: a cash-neutral spread, so
excess-of-cash *is* the spread. Frictions: **10 bps one-way × NAV** on the spin-off side
(round trip = 2×), 1 bp on SPY, **40 bp/yr borrow on the short index leg**. Total-return
closes (`auto_adjust=True`). One execution lag, everywhere.

{BANNER}"""),
        code("R = %r" % (R,)),
        md("## 1. The three legs\n\n"
           "Headline figures are **gross**. The child's alpha is negative, so charging a long "
           "position's round-trip costs to it would make the effect look bigger than it is; the "
           "net column is printed beside it to show frictions are not what create it, and the "
           "costed number that matters lives in §5, on the side you would actually trade.\n\n"
           "> 💡 **In plain words:** five numbers were pre-specified. Four are noise. The one "
           "that fires points the opposite way to the story it was built to test — and once you "
           "pay for having taken five looks, it clears the 5% line by a whisker, not a mile."),
        code(
            "print(f\"{'leg':34s}{'mean':>9s}{'med':>9s}{'hit':>7s}{'t':>8s}{'HAC t':>8s}   95% CI\")\n"
            "rows = [('parent  ann -> distribution', R['par_mean'], R['par_med'], R['par_hit'], R['par_t'], R['par_hac'], R['par_ci_lo'], R['par_ci_hi']),\n"
            "        ('child   first  5 sessions',  R['c5_mean'],  R['c5_med'],  R['c5_hit'],  R['c5_t'],  R['c5_hac'],  R['c5_ci_lo'],  R['c5_ci_hi']),\n"
            "        ('child   first 10 sessions',  R['c10_mean'], R['c10_med'], R['c10_hit'], R['c10_t'], R['c10_hac'], R['c10_ci_lo'], R['c10_ci_hi']),\n"
            "        ('child   first 21 sessions',  R['c21_mean'], R['c21_med'], R['c21_hit'], R['c21_t'], R['c21_hac'], R['c21_ci_lo'], R['c21_ci_hi']),\n"
            "        ('parent+child first 21',      R['comb_mean'],R['comb_med'],R['comb_hit'],R['comb_t'],R['comb_hac'],R['comb_ci_lo'],R['comb_ci_hi'])]\n"
            "for lbl, m, md_, h, t, hac, lo, hi in rows:\n"
            "    print(f\"{lbl:34s}{m:+8.2f}%{md_:+8.2f}%{h:7.2f}{t:+8.2f}{hac:+8.2f}   [{lo:+.2f}, {hi:+.2f}]%\")\n"
            "print(f\"\\nnet of a LONG position's frictions: parent {R['n_par']:+.2f}% \"\n"
            "      f\"(t={R['n_par_t']:+.2f}) | child+5d {R['n_c5']:+.2f}% (t={R['n_c5_t']:+.2f}, \"\n"
            "      f\"HAC {R['n_c5_hac']:+.2f}) | +10d {R['n_c10']:+.2f}% (t={R['n_c10_t']:+.2f}) | \"\n"
            "      f\"+21d {R['n_c21']:+.2f}% (t={R['n_c21_t']:+.2f}) | comb {R['n_comb']:+.2f}% \"\n"
            "      f\"(t={R['n_comb_t']:+.2f})\")\n"
            "print('the 5-session leg clears |t|>=2 gross AND net, with the sign inverted vs the claim')\n"
            "print()\n"
            "print(f\"but five legs were pre-specified, so the family is priced: Westfall-Young\")\n"
            "print(f\"max-|t| bootstrap -> single-test p = {R['fw_p_single']:.4f}, \"\n"
            "      f\"FAMILY-WISE p over {R['fw_legs']} legs = {R['fw_p_fwe']:.4f}  <- survives, narrowly\")"
        ),
        md("## 2. Robustness on the 5-session leg\n\n"
           "Era cut by distribution date, a size-matched benchmark swap (children are small/mid "
           "caps and SPY is not), a leave-one-out jackknife, de-clustering to one event per "
           "calendar quarter, and a cost sweep.\n\n"
           "> 💡 **In plain words:** costs make a *negative* number more negative, so they "
           "cannot be what manufactured it."),
        code(
            "print(f\"eras : pre-2020 n={R['era_e_n']:2d} {R['era_e_mean']:+.2f}% (t={R['era_e_t']:+.2f})  |  \"\n"
            "      f\"2020+ n={R['era_l_n']:2d} {R['era_l_mean']:+.2f}% (t={R['era_l_t']:+.2f})   -> negative in both\")\n"
            "print(f\"bench: SPY {R['c5_mean']:+.2f}% (t={R['c5_t']:+.2f})  IWM {R['iwm_mean']:+.2f}% \"\n"
            "      f\"(t={R['iwm_t']:+.2f})  MDY {R['mdy_mean']:+.2f}% (t={R['mdy_t']:+.2f})   -> not a size tilt\")\n"
            "print(f\"jack : leave-one-out t in [{R['jk_lo']:+.2f}, {R['jk_hi']:+.2f}]   -> no single event carries it\")\n"
            "print(f\"clust: one event per quarter n={R['dec_n']} {R['dec_mean']:+.2f}% (t={R['dec_t']:+.2f})\")\n"
            "print(f\"cost : 0bps {R['cost0']:+.2f}% (t={R['cost0_t']:+.2f}) | 25bps {R['cost25']:+.2f}% \"\n"
            "      f\"(t={R['cost25_t']:+.2f}) | 50bps {R['cost50']:+.2f}% (t={R['cost50_t']:+.2f})\")"
        ),
        md("## 3. The three non-tape inputs, swept\n\n"
           "The event table is an **assumption**: announcement dates, first regular-way sessions "
           "and distribution ratios are hand-compiled from Form 10 / 8-K filings and the press. "
           "Each is swept.\n\n"
           "> 💡 **In plain words:** the regular-way anchor is the one that matters. Start "
           "earlier — inside the when-issued tail — and the effect is *stronger*; start three "
           "sessions late and it flips sign. That is a description of a three-to-five-session "
           "effect, but it also means the dates have to be right to the session."),
        code(
            "print('regular-way anchor (the load-bearing proxy):')\n"
            "for lbl, m, t in [('-3 sessions', R['rw_m3'], R['rw_m3_t']), ('-1 session ', R['rw_m1'], R['rw_m1_t']),\n"
            "                  (' 0 (table) ', R['n_c5'], R['n_c5_t']), ('+1 session ', R['rw_p1'], R['rw_p1_t']),\n"
            "                  ('+3 sessions', R['rw_p3'], R['rw_p3_t'])]:\n"
            "    flag = '   <- sign flips' if m > 0 else ''\n"
            "    print(f'   {lbl}: {m:+6.2f}% (t={t:+.2f}){flag}')\n"
            "print(f\"announcement date (parent leg): -5 {R['ann_m5']:+.2f}% .. +5 {R['ann_p5']:+.2f}% \"\n"
            "      f\"-> null at every setting\")\n"
            "print(f\"distribution ratio (combined) : x0.5 {R['ratio_half']:+.2f}% | x1 {R['n_comb']:+.2f}% | \"\n"
            "      f\"x2 {R['ratio_two']:+.2f}% | equal-weight {R['ratio_eq']:+.2f}% -> null under every weighting\")\n"
            "print()\n"
            "print('is the anchor on the right side of the transition? checked, not asserted:')\n"
            "print(f\"  {R['anchor_exact']}/{R['n_events']} stated dates are themselves a session on the\"\n"
            "      ' child tape (VLTO snaps 3 sessions forward: Yahoo has no 2023-10-02/03)')\n"
            "print(f\"  {R['anchor_tail']}/{R['n_events']} children show a genuine when-issued tail of \"\n"
            "      f\"{R['anchor_tail_lo']}-{R['anchor_tail_hi']} sessions BEFORE the anchor\")\n"
            "print('  -> the anchor is not set early, which is the direction that would inflate this')"
        ),
        md("## 4. Event-time shape — EXPLORATORY\n\n"
           "CAAR relative to the first regular-way close; negative event time sits **inside the "
           "when-issued period**. Only 5 / 10 / 21 were pre-specified — everything else on this "
           "grid was searched after seeing the data and carries an unpriced multiple-comparison "
           "penalty. It is here for the mechanism, never for the stamp."),
        code(
            "for k, v in [(-3, R['caar_m3']), (-1, R['caar_m1']), (0, 0.0), (1, R['caar_p1']),\n"
            "             (2, R['caar_p2']), (3, R['caar_p3']), (5, R['caar_p5']),\n"
            "             (10, R['caar_p10']), (21, R['caar_p21'])]:\n"
            "    bar = '#' * int(abs(v) * 3)\n"
            "    print(f\"  session {k:+3d}: {v:+6.2f}%  {bar}\")\n"
            "print(f\"\\ntrough at +3 sessions: {R['h3_mean']:+.2f}% (t={R['h3_t']:+.2f}, HAC {R['h3_hac']:+.2f}, \"\n"
            "      f\"{R['h3_hit']:.0%} positive); gone by +63d: {R['h63_mean']:+.2f}% (t={R['h63_t']:+.2f})\")\n"
            "print(f\"n={R['caar_n']} events have >=3 when-issued sessions on the tape\")"
        ),
        md("## 5. Harvesting a negative alpha — the short side, and the borrow it needs\n\n"
           "The trade is: short the child, long SPY, five sessions. The child's borrow rate is "
           "**not on this tape** — it is an assumption, and for a name five days old it is the "
           "assumption that decides everything.\n\n"
           "> 💡 **In plain words:** at any plausible borrow the arithmetic works; the question "
           "is whether a locate exists at all."),
        code(
            "print(f\"{'child borrow':>14s}{'mean':>9s}{'t':>8s}{'hit':>7s}   95% CI\")\n"
            "rows = [('0%/yr', R['sh0'], R['sh0_t'], R['sh0_hit'], R['sh0_ci_lo'], R['sh0_ci_hi']),\n"
            "        ('25%/yr', R['sh25'], R['sh25_t'], R['sh0_hit'], R['sh25_ci_lo'], R['sh25_ci_hi']),\n"
            "        ('50%/yr', R['sh50'], R['sh50_t'], R['sh50_hit'], R['sh50_ci_lo'], R['sh50_ci_hi']),\n"
            "        ('100%/yr', R['sh100'], R['sh100_t'], R['sh50_hit'], R['sh100_ci_lo'], R['sh100_ci_hi'])]\n"
            "for lbl, m, t, h, lo, hi in rows:\n"
            "    flag = '   <- CI through zero' if lo < 0 else ''\n"
            "    print(f\"{lbl:>14s}{m:+8.2f}%{t:+8.2f}{h:7.2f}   [{lo:+.2f}, {hi:+.2f}]%{flag}\")\n"
            "print(f\"\\ndaily hedged sleeve, 5d window: {R['sl5_days']} active of {R['sl5_of']} \"\n"
            "      f\"sessions, {R['sl5_bps']:+.1f} bps/day, HAC t={R['sl5_t']:+.2f}, \"\n"
            "      f\"block-boot CI [{R['sl5_lo']:+.1f}, {R['sl5_hi']:+.1f}] bps\")\n"
            "print(f\"21d window: {R['sl21_days']} of {R['sl21_of']} sessions, \"\n"
            "      f\"{R['sl21_bps']:+.1f} bps/day (t={R['sl21_t']:+.2f}) -- diluted to nothing\")\n"
            "print(f\"idle {1 - R['sl5_days']/R['sl5_of']:.0%} of even its OWN window (first distribution\"\n"
            "      ' -> last exit, not whatever pre-history the shared cache holds for SPY):')\n"
            "print('a capacity statement about 2 events/yr, not evidence either way, which is')\n"
            "print('why no fund-level Sharpe is quoted')"
        ),
        md("## 6. Live synthetic control — the estimator is unbiased (synthetic, not the real tape)\n\n"
           "A planted world (a real parent run-up and a real child drift) must fire; a null world "
           "must not. Same code path as the real tape: `event_panel` on a synthetic frame that "
           "uses the loader's schema."),
        code(
            SETUP +
            "\n# SYNTHETIC worlds only — this cell never touches the real tape.\n"
            "px, ev, truth = data.synthetic_daily(signal_strength=1.0, seed=930)\n"
            "pl = st.synthetic_detect(px, ev)\n"
            "print(f\"planted (child 21d alpha {truth['planted_child_alpha_21d']:.2%}): \"\n"
            "      f\"parent {pl['parent_mean_pct']:+.2f}% (t={pl['parent_t']:+.2f}), \"\n"
            "      f\"child21 {pl['child21_mean_pct']:+.2f}% (t={pl['child21_t']:+.2f})\")\n"
            "nulls = [st.synthetic_detect(*data.synthetic_daily(signal_strength=0.0, seed=930+s)[:2])\n"
            "         for s in range(8)]\n"
            "pt = np.array([d['parent_t'] for d in nulls]); ct = np.array([d['child21_t'] for d in nulls])\n"
            "print(f\"null x8: parent t mean {pt.mean():+.2f} (|t|>=2 in {(abs(pt)>=2).sum()}/8), \"\n"
            "      f\"child21 t mean {ct.mean():+.2f} (|t|>=2 in {(abs(ct)>=2).sum()}/8)\")\n"
            "print('an unbiased ruler -> the real-tape result is a property of spin-offs, not the harness')"
        ),
        md(f"""## Verdict

- **Signal — Real, sign inverted, family-wise *p* = {R['fw_p_fwe']:.4f}.**
  {R['c5_mean']:+.2f}% gross over the child's first five regular-way sessions
  (*t* = {R['c5_t']:+.2f}, HAC {R['c5_hac']:+.2f}, bootstrap CI
  [{R['c5_ci_lo']:+.2f}, {R['c5_ci_hi']:+.2f}]% clear of zero, hit rate {R['c5_hit']:.2f}
  [{R['c5_hit_lo']:.2f}, {R['c5_hit_hi']:.2f}]; {R['n_c5']:+.2f}% net of a long position's
  frictions, which is why gross is the headline). Negative gross and net, in both eras, versus
  SPY / IWM / MDY, under a jackknife ({R['jk_lo']:+.2f}…{R['jk_hi']:+.2f}) and after
  de-clustering ({R['dec_t']:+.2f}) — |*t*| ≥ 2.2 on every cut. The other four pre-specified
  legs — the parent run-up ({R['par_mean']:+.2f}%, *t* = {R['par_t']:+.2f}), the 10- and
  21-session child windows, and the sum-of-the-parts ({R['comb_mean']:+.2f}%,
  *t* = {R['comb_t']:+.2f}) — are null, and the Westfall-Young max-|*t*| bootstrap over all
  five puts the family-wise *p* at **{R['fw_p_fwe']:.4f}**: inside 5%, but not comfortably.
  The forced-*seller* prediction is rejected; the shape (pop, three-session slide, month-long
  fill-in) is the index-inclusion footprint of forced *buying*. Survivorship named on this
  axis: {R['n_events']} curated liquid spins, plus {R['n_delisted']} dropped because the child
  was acquired and delisted. Unit beta is assumed throughout — the IWM/MDY swap is the check.
- **Tradability — Fragile.** The harvesting trade is a five-session short of a five-day-old
  listing: {R['sh0']:+.2f}%/trade (*t* = {R['sh0_t']:+.2f}) gross of borrow,
  {R['sh50']:+.2f}% at 50%/yr, {R['sh100']:+.2f}% with a CI through zero at 100%/yr — and
  the borrow is the one input the tape cannot price. ~{R['per_year']:.0f} trades a year,
  {R['c5_sd']:.1f} pp per-trade dispersion, and a {R['rw_p3']:+.2f}% sign flip if the anchor
  is three sessions late. Not bankable as a strategy; useful as a warning about buying the
  first print."""),
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
