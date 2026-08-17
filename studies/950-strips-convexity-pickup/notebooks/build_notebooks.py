"""Generate the two narrative notebooks for Study 950 (Zero-Coupon Convexity).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below, which mirrors ``docs/results.md``; the only live cells run the
fast synthetic control, and they are never presented under a real-tape banner.
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


# Frozen real-tape headline — mirror of docs/results.md. EDV (20-30y STRIPS) vs a
# duration-matched TLT+BIL mix, total-return, excess-of-cash, 2009-02-02 -> 2026-06-30.
R = dict(
    start="2009-02-02", end="2026-06-30", n_days=4377, n_months=208, fp="a669055b6e7a",
    L_mean=1.420, L_sd=0.097, L_lo=1.19, L_hi=1.62,
    a_mean=2.65, a_vol=21.41, a_sharpe=0.124, a_t=0.57,
    b_mean=3.19, b_vol=21.42, b_sharpe=0.149, b_t=0.71,
    sp_mean=-0.53, sp_vol=4.77, sp_sharpe=-0.213, sp_t=-0.87,
    vol_ratio=1.000, sp_bp_mo=-3.98, sp_bp_t=-1.08, sp_hit=44.2,
    # asymmetry regression, squared NET monthly move
    dy2_a=-8.75, dy2_a_t=-1.50, dy2_b1=-0.7217, dy2_b1_t=-2.73,
    dy2_b2=110.5, dy2_b2_t=0.84, dy2_25=6.90, dy2_50=27.62, dy2_be=28.1, dy2_r2=0.079,
    # asymmetry regression, realised variance
    rv_a=-12.74, rv_a_t=-1.88, rv_b1=-0.7108, rv_b1_t=-2.55,
    rv_b2=154.7, rv_b2_t=1.31, rv_25=9.67, rv_50=38.69, rv_be=28.7, rv_r2=0.084,
    median_absdy=15.0,
    # move-size buckets (raw / after removing the residual linear exposure)
    bk_q_n=70, bk_q_dy=4.1, bk_q=-0.94, bk_q_t=-0.15, bk_q_h=-1.22,
    bk_m_n=69, bk_m_dy=14.5, bk_m=-7.51, bk_m_t=-1.22, bk_m_h=-7.07,
    bk_l_n=69, bk_l_dy=32.4, bk_l=-3.52, bk_l_t=-0.43, bk_l_h=-2.38,
    # what the fitted b2 predicts each bucket should earn, and the noise on the gap
    bk_q_pred=0.27, bk_l_pred=12.93, bk_gap_se=11.6,
    # bootstrap
    boot_lo=-11.01, boot_hi=3.42, boot_neg=86.8,
    b2_dy2_lo=-163.3, b2_dy2_hi=338.2, b2_dy2_neg=27.5,
    b2_rv_lo=-41.0, b2_rv_hi=516.2, b2_rv_neg=5.9,
    # eras
    e_n=106, e_sp=-4.21, e_sp_t=-0.67, e_a=-12.78, e_a_t=-1.44, e_b2=179.9, e_b2_t=0.85, e_b1_t=-2.06,
    l_n=101, l_sp=-4.17, l_sp_t=-1.18, l_a=-4.18, l_a_t=-0.94, l_b2=21.8, l_b2_t=0.39, l_b1_t=-2.41,
    # sweeps
    cost0_sp=-4.00, cost25_sp=-3.78,
    fin0_sp=-4.85, fin0_t=-1.31, fin100_sp=-1.35, fin100_t=-0.37,
    w126_sp=-0.09, w126_b2=87.1, w504_sp=-1.18, w504_b2=98.2,
    # ZROZ cross-check
    z_start="2010-12-01", z_months=186, z_L=1.549, z_volratio=0.992,
    z_sp=-2.06, z_sp_t=-0.46, z_b2_dy2=208.7, z_b2_dy2_t=1.69,
    z_b2_rv=51.7, z_b2_rv_t=0.27, z_a_dy2=-11.43, z_a_dy2_t=-1.75, z_b1_t=-1.40,
    # ZROZ era cut — the only cuts in the study that clear |t| = 2, and the sign flip
    # that stops them counting. (2010-2017 vs 2018-2026.)
    ze_b2_dy2=403.8, ze_b2_dy2_t=2.14, ze_b2_rv=618.9, ze_b2_rv_t=2.88,
    ze_a_rv=-33.19, zl_b2_rv=-123.5, zl_b2_rv_t=-1.81, zl_a_rv=5.09,
    # full 12-cut census (2 funds x 3 eras x 2 specs) — see docs/results.md
    grid_n=12, grid_b2_pos=11, grid_a_neg=11, grid_t2=2, grid_max_t=2.88,
    edv_max_t=1.87,
    # synthetic control
    syn_b2=260.3, syn_b2_t=6.31, syn_a=-27.05, syn_a_t=-7.41, syn_volratio=1.002,
    syn_null_t=-0.93, syn_null_sd=1.12, syn_null_fire=1,
)


HEADER = f"""# Study 950 — Zero-Coupon Convexity 📐

**Does a zero-coupon Treasury fund pay you for convexity, or just for duration?**

A bond's price curves against its yield. That curvature — *convexity* — is supposed to be a
gift: for the same duration, a more convex position gains more on a big rally than it loses
on an equally big sell-off. The zero-coupon Treasury funds (**EDV**, 20-30y STRIPS;
**ZROZ**, 25y+) sit further out the curve than the ordinary long-bond fund **TLT**, so per
dollar of duration they carry more curvature. If that curvature is being *paid*, the zero
should beat a **duration-matched** TLT + T-bill mix in **large-move months** and lose a
little in quiet ones.

We match the duration on the realised beta of each leg to **the same rate factor** (the
daily change in the 30-year yield, `^TYX`), race the two arms **excess-of-cash**, and test
the **asymmetry explicitly** — a regression on the squared rate move — rather than reading
it off an average. EDV vs the matched mix, {R['start']} → {R['end']} ({R['n_days']:,} days,
{R['n_months']} months).

*Every real number below is the frozen headline (`docs/results.md`, Fingerprint
`{R['fp']}`); the only live cells run the fast offline synthetic control. As-of 2026-06-30.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. What convexity is, in one breath\n\n"
           "If rates move 1%, a 20-year bond loses about 20% — *roughly*. The word doing the "
           "work is *roughly*: the loss on a rise is slightly smaller than the gain on a fall "
           "of the same size, because the price-yield line bends. That bend is convexity. "
           "The bigger the move, the more the bend matters — which is why any real test has "
           "to look at the **size** of the move, not the average month.\n\n"
           "> 🔬 **For the quants** — formally `ΔP/P ≈ −D·Δy + ½·C·Δy²`. The first term is "
           "duration and it is what we neutralise; the whole study lives in the second term."),
        md("## 2. First, make the two arms carry the same rate risk\n\n"
           f"The zero fund is simply *longer*. Solving the match from realised sensitivity to "
           f"the 30-year yield says you need **{R['L_mean']:.2f} units of TLT** (funded with "
           f"T-bills) to carry as much rate risk as one unit of EDV — the STRIPS fund runs "
           f"about **42% more duration per dollar**. Do that, and the two arms end up with "
           f"volatilities that agree to three decimal places."),
        code(
            "R = dict(L_mean=%r, a_vol=%r, b_vol=%r, vol_ratio=%r, sp_mean=%r, sp_bp_mo=%r,\n"
            "         sp_bp_t=%r, sp_hit=%r)\n"
            "print('one unit of EDV  ~ %%.2f units of TLT (rest in T-bills)' %% R['L_mean'])\n"
            "print('volatility  EDV arm %%.2f%%%%   matched mix %%.2f%%%%   ratio %%.3f'\n"
            "      %% (R['a_vol'], R['b_vol'], R['vol_ratio']))\n"
            "print('the leftover spread: %%+.2f%%%%/yr  = %%+.2f bp per month (t %%+.2f), '\n"
            "      'positive in only %%.1f%%%% of months'\n"
            "      %% (R['sp_mean'], R['sp_bp_mo'], R['sp_bp_t'], R['sp_hit']))"
            % (R["L_mean"], R["a_vol"], R["b_vol"], R["vol_ratio"], R["sp_mean"],
               R["sp_bp_mo"], R["sp_bp_t"], R["sp_hit"])
        ),
        md(f"## 3. The honest headline — the average says nothing\n\n"
           f"Over seventeen and a half years the zero fund ends up **{R['sp_mean']:+.2f}%/yr** "
           f"behind the duration-matched mix — with a *t* of {R['sp_t']:+.2f} and a bootstrap "
           f"range of [{R['boot_lo']:+.1f}, {R['boot_hi']:+.1f}] bp/month. That is a shrug, not "
           f"a result. But an average was never the right test: convexity is supposed to pay "
           f"in the **big** months and cost you in the quiet ones."),
        md("## 4. So: does it pay in the big months?\n\n"
           "Sort the months into three buckets by how far the 30-year yield actually moved. "
           "If the convexity story were true, the numbers should climb from left to right."),
        code(
            "rows = [('quiet', %r, %r, %r, %r), ('middling', %r, %r, %r, %r),\n"
            "        ('large', %r, %r, %r, %r)]\n"
            "print('bucket      n   mean |move|   spread      t     (after removing the\\n"
            "                                                        residual duration leak)')\n"
            "for name, n, dy, sp, t in rows:\n"
            "    print('%%-9s %%3d   %%6.1f bp   %%+7.2f bp  %%+5.2f' %% (name, n, dy, sp, t))\n"
            "print()\n"
            "print('hedged: quiet %%+.2f   middling %%+.2f   large %%+.2f  bp/month'\n"
            "      %% (%r, %r, %r))"
            % (R["bk_q_n"], R["bk_q_dy"], R["bk_q"], R["bk_q_t"],
               R["bk_m_n"], R["bk_m_dy"], R["bk_m"], R["bk_m_t"],
               R["bk_l_n"], R["bk_l_dy"], R["bk_l"], R["bk_l_t"],
               R["bk_q_h"], R["bk_m_h"], R["bk_l_h"])
        ),
        md(f"They do not climb. The large-move bucket is **{R['bk_l']:+.2f} bp/month** — "
           f"negative — and it stays negative after we strip out the small amount of "
           f"leftover duration the match could not remove. On the rawest possible reading, "
           f"the convexity paycheck simply does not show up."),
        md(f"## 5. The regression is kinder — and still not enough\n\n"
           f"Fit the shape properly (`spread = a + b1·move + b2·move²`) and the signs come out "
           f"**exactly as the textbook says they should**: a positive curvature term "
           f"(**{R['dy2_b2']:+.1f}**, worth about **{R['dy2_25']:+.1f} bp** in a 25 bp month and "
           f"**{R['dy2_50']:+.1f} bp** in a 50 bp one) and a negative intercept "
           f"(**{R['dy2_a']:+.2f} bp/month** — the price you pay for it). That pattern repeats in "
           f"**{R['grid_b2_pos']} of the {R['grid_n']}** cuts we ran (two funds × three eras × "
           f"two ways of measuring the move — the whole census, printed in "
           f"[docs/results.md](../docs/results.md)).\n\n"
           f"And yet the *t*-statistic on that curvature term is **{R['dy2_b2_t']:+.2f}**. "
           f"The desk's bar is |*t*| ≥ 2, and exactly **{R['grid_t2']} of {R['grid_n']}** cuts "
           f"reach it — both of them the *same* 84 months of the cross-check fund ZROZ "
           f"(2010-2017: **+{R['ze_b2_dy2_t']:.2f}** and **+{R['ze_b2_rv_t']:.2f}**), counted "
           f"once per regressor. What happens next door settles it: on **2018-2026** that same "
           f"fund gives a curvature term of **{R['zl_b2_rv']:+.1f}** with a *positive* "
           f"intercept — the story running backwards — and EDV, the fund in the headline, never "
           f"gets past **+{R['edv_max_t']:.2f}** in any era. A shape that reaches significance "
           f"in one window and inverts in the one beside it is a coin, not a premium."),
        md(f"## 6. The number that settles it — a 28 bp breakeven\n\n"
           f"Take the fit at face value and ask the practical question: **how far must rates "
           f"move in a month before the curvature gain repays the carry you gave up?** The "
           f"answer is **{R['dy2_be']:.0f} basis points**. The *median* month in this sample "
           f"moved **{R['median_absdy']:.0f} bp**. So even on its own most flattering terms, the "
           f"convexity trade is under water in roughly two months out of three, and needs the "
           f"tail to bail it out."),
        md(f"## 7. One more inconvenient detail\n\n"
           f"EDV holds 20-30 year STRIPS; TLT holds 20-year-plus coupon bonds. They are not at "
           f"the same point on the curve, and the match — solved against a single 30-year "
           f"factor — cannot fully fix that. What is left is about **+0.72 years** of residual "
           f"duration (*t* = {R['dy2_b1_t']:+.2f}), and it is there in **both** halves of the "
           f"sample. In other words, a good part of what this \"convexity spread\" trades is "
           f"the 20s-versus-30s curve. That is a legitimate position — it is just not the one "
           f"the story is selling."),
        md("## 8. Is the tool broken, or is the effect small? (live, offline)\n\n"
           "Fair question. So we build a synthetic bond world where the long leg *genuinely* "
           "carries a big convexity pickup and genuinely pays carry for it, and a null world "
           "where it does not — and run the identical machinery on both."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from zero_convexity import data, strategy as st\n"
            "planted = [st.synthetic_detect(data.synthetic_panel(signal_strength=1.0, seed=950+s)[0])\n"
            "           for s in range(6)]\n"
            "null    = [st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=950+s)[0])\n"
            "           for s in range(6)]\n"
            "def show(tag, runs):\n"
            "    b2 = np.array([r['b2'] for r in runs]); t = np.array([r['b2_t'] for r in runs])\n"
            "    a  = np.array([r['a_bp_mo'] for r in runs])\n"
            "    print('%-14s curvature term %+8.1f  (t %+5.2f)  price of it %+7.2f bp/mo  '\n"
            "          '[6 worlds, |t|>=2 in %d]'\n"
            "          % (tag, b2.mean(), t.mean(), a.mean(), (abs(t) >= 2).sum()))\n"
            "show('planted world', planted)\n"
            "show('null world', null)"
        ),
        md("The detector fires hard when there is something to find and goes quiet when there "
           "is not. So the real-tape silence is a fact about the **Treasury curve**, not about "
           "the harness."),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** The shape is right almost everywhere we look — positive "
           f"curvature term and negative intercept in **{R['grid_b2_pos']} of {R['grid_n']}** "
           f"cuts — but the headline *t* is **{R['dy2_b2_t']:+.2f}**, the bootstrap interval "
           f"straddles zero, and the large-move bucket is negative. The "
           f"**{R['grid_t2']} cuts that do clear 2** are one window of the cross-check fund "
           f"whose neighbouring window flips both signs, so nothing here replicates; on top of "
           f"that a significant **+0.72 yr** of residual duration means part of the spread is a "
           f"curve trade. Directionally right, statistically uncertified.\n"
           f"- **Tradability — Mirage.** The spread earns **{R['sp_mean']:+.2f}%/yr** and needs a "
           f"**{R['dy2_be']:.0f} bp** month just to break even against a median month of "
           f"**{R['median_absdy']:.0f} bp**. Costs and financing never change the sign — this one "
           f"fails at the signal stage, not the friction stage.\n"
           f"- **What the zero fund *is* good for:** more duration per dollar (about 42% more). "
           f"That is a real, useful property. It is just not a convexity paycheck."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 950 — Zero-Coupon Convexity — the teardown\n\n"
           "The rate-factor duration match, the excess-of-cash race, the asymmetry regression "
           "in both quadratic specifications, the move-size buckets (raw and linearly hedged), "
           "block-bootstrap CIs on the mean and on *b2*, the era cut, three sweeps, the ZROZ "
           "cross-check and the live synthetic control. Every real number is frozen from "
           "`docs/results.md` (Fingerprint `%s`).\n\n"
           "**Design.** Arm A = 100%% EDV. Arm B = `L`·TLT + `(1−L)`·BIL, `L` = ratio of the "
           "rolling 252-day OLS slopes of each leg's daily excess return on `Δ^TYX`, read at "
           "month end and traded the following month (**one execution lag**). Both arms "
           "excess-of-cash. 3 bps one-way × NAV on the mix's turnover; a **PROXY** 25 bp/yr "
           "financing spread on `(L−1)`. Both frictions fall on Arm B only, so the race is "
           "tilted *towards* the claim." % R["fp"]),
        code("R = %r" % (R,)),
        md("## 1. The match — does it actually neutralise duration?"),
        code(
            "print(f\"L on TLT: mean {R['L_mean']:.3f}  sd {R['L_sd']:.3f}  \"\n"
            "      f\"range [{R['L_lo']:.2f}, {R['L_hi']:.2f}]\")\n"
            "print(f\"A: 100% EDV        mean {R['a_mean']:+.2f}%/yr  vol {R['a_vol']:.2f}%  \"\n"
            "      f\"Sharpe {R['a_sharpe']:+.3f}  HAC t {R['a_t']:+.2f}\")\n"
            "print(f\"B: matched mix     mean {R['b_mean']:+.2f}%/yr  vol {R['b_vol']:.2f}%  \"\n"
            "      f\"Sharpe {R['b_sharpe']:+.3f}  HAC t {R['b_t']:+.2f}\")\n"
            "print(f\"A - B (spread)     mean {R['sp_mean']:+.2f}%/yr  vol {R['sp_vol']:.2f}%  \"\n"
            "      f\"Sharpe {R['sp_sharpe']:+.3f}  HAC t {R['sp_t']:+.2f}\")\n"
            "print(f\"vol ratio A/B = {R['vol_ratio']:.3f}  -> a clean duration match\")\n"
            "print(f\"monthly: {R['sp_bp_mo']:+.2f} bp/mo (t {R['sp_bp_t']:+.2f}), \"\n"
            "      f\"hit rate {R['sp_hit']:.1f}%, bootstrap 95% CI \"\n"
            "      f\"[{R['boot_lo']:+.2f}, {R['boot_hi']:+.2f}] bp/mo, share<0 {R['boot_neg']:.1f}%\")"
        ),
        md("> 💡 **In plain words** — the two arms end up with the *same* interest-rate risk "
           "(volatilities 21.41% vs 21.42%). Anything left over is not duration. Over the whole "
           "sample that leftover is slightly negative and statistically nothing."),
        md("## 2. The headline — `diff = a + b1·Δy + b2·Q`, HAC(6)\n\n"
           "`Q` is either the squared net monthly move `Δy²` (a bond repriced from one month-end "
           "yield to the next) or the realised variance `Σ Δy_t²` of the daily factor inside the "
           "month (the gamma P&L a daily-marked fund actually accrues). Theory: `b2 > 0`, "
           "`a < 0`, `b1 ≈ 0`."),
        code(
            "for tag, a, at, b1, b1t, b2, b2t, p25, p50, be, r2 in [\n"
            "    ('dy^2 (net move) ', R['dy2_a'], R['dy2_a_t'], R['dy2_b1'], R['dy2_b1_t'],\n"
            "     R['dy2_b2'], R['dy2_b2_t'], R['dy2_25'], R['dy2_50'], R['dy2_be'], R['dy2_r2']),\n"
            "    ('realised variance', R['rv_a'], R['rv_a_t'], R['rv_b1'], R['rv_b1_t'],\n"
            "     R['rv_b2'], R['rv_b2_t'], R['rv_25'], R['rv_50'], R['rv_be'], R['rv_r2'])]:\n"
            "    print(f\"[{tag}]  R2={r2:.3f}\")\n"
            "    print(f\"   a  = {a:+7.2f} bp/mo (t {at:+.2f})   b1 = {b1:+.4f} (t {b1t:+.2f}) \"\n"
            "          f\"-> residual duration {-b1:+.2f} yr\")\n"
            "    print(f\"   b2 = {b2:+7.1f}        (t {b2t:+.2f})   worth {p25:+.2f} bp at 25 bp, \"\n"
            "          f\"{p50:+.2f} bp at 50 bp; breakeven move {be:.1f} bp\")\n"
            "print(f\"\\nmedian |monthly move| = {R['median_absdy']:.1f} bp \"\n"
            "      f\"-> the breakeven sits at roughly the 67th percentile of months\")\n"
            "print(f\"bootstrap b2 [dy^2]: 95% CI [{R['b2_dy2_lo']:+.1f}, {R['b2_dy2_hi']:+.1f}], \"\n"
            "      f\"share<0 {R['b2_dy2_neg']:.1f}%\")\n"
            "print(f\"bootstrap b2 [rv  ]: 95% CI [{R['b2_rv_lo']:+.1f}, {R['b2_rv_hi']:+.1f}], \"\n"
            "      f\"share<0 {R['b2_rv_neg']:.1f}%\")"
        ),
        md("> 💡 **In plain words** — the curvature term points the right way and is about the "
           "right size for a 7.5-year duration gap. It is simply drowned in noise: a 25 bp month "
           "is supposed to hand you ~7 bp, and the month-to-month scatter is ~140 bp."),
        md("## 3. The same thing without a model — move-size buckets\n\n"
           "Terciles of |Δy|, raw and after regressing out the residual linear exposure "
           "(in-sample, descriptive — used to look at *shape*, never to claim a return)."),
        code(
            "print(f\"{'bucket':10s}{'n':>5s}{'|dy|':>10s}{'spread':>12s}{'t':>8s}{'hedged':>12s}\")\n"
            "for nm, n, dy, sp, t, h in [\n"
            "    ('quiet', R['bk_q_n'], R['bk_q_dy'], R['bk_q'], R['bk_q_t'], R['bk_q_h']),\n"
            "    ('middling', R['bk_m_n'], R['bk_m_dy'], R['bk_m'], R['bk_m_t'], R['bk_m_h']),\n"
            "    ('large', R['bk_l_n'], R['bk_l_dy'], R['bk_l'], R['bk_l_t'], R['bk_l_h'])]:\n"
            "    print(f\"{nm:10s}{n:5d}{dy:9.1f}bp{sp:+11.2f}bp{t:+8.2f}{h:+11.2f}bp\")\n"
            "print('\\nno monotone climb -> the raw asymmetry is absent.')\n"
            "gap = R['bk_l_pred'] - R['bk_q_pred']\n"
            "print(f\"consistency check: at b2 = {R['dy2_b2']:+.0f} the fit predicts \"\n"
            "      f\"{R['bk_q_pred']:+.2f} bp in the quiet bucket and {R['bk_l_pred']:+.2f} bp \"\n"
            "      f\"in the large one\")\n"
            "print(f\"  -> a gap of {gap:.1f} bp/mo against a standard error on that gap of \"\n"
            "      f\"{R['bk_gap_se']:.1f} bp: about ONE standard error.\")\n"
            "print('  The bucket table and the regression do not disagree - the effect is simply')\n"
            "print('  at the edge of what 208 months can resolve.')"
        ),
        md("## 4. Era cut (split 2018-01-01)"),
        code(
            "print(f\"2009-2017 (n={R['e_n']:3d} mo): spread {R['e_sp']:+.2f} bp/mo (t {R['e_sp_t']:+.2f})  \"\n"
            "      f\"a {R['e_a']:+.2f} (t {R['e_a_t']:+.2f})  b2 {R['e_b2']:+7.1f} (t {R['e_b2_t']:+.2f})  \"\n"
            "      f\"b1 t {R['e_b1_t']:+.2f}\")\n"
            "print(f\"2018-2026 (n={R['l_n']:3d} mo): spread {R['l_sp']:+.2f} bp/mo (t {R['l_sp_t']:+.2f})  \"\n"
            "      f\"a {R['l_a']:+.2f} (t {R['l_a_t']:+.2f})  b2 {R['l_b2']:+7.1f} (t {R['l_b2_t']:+.2f})  \"\n"
            "      f\"b1 t {R['l_b1_t']:+.2f}\")\n"
            "print('\\nb2 keeps its sign but collapses 8x into the era that CONTAINS 2020 and 2022 -')\n"
            "print('the two biggest rate shocks of the sample. The only era-stable feature is the')\n"
            "print('residual duration leak (b1 t = %+.2f / %+.2f).' % (R['e_b1_t'], R['l_b1_t']))"
        ),
        md("## 5. Sweeps — costs, the financing PROXY, and the hedge lookback"),
        code(
            "print(f\"cost   0 bps: spread {R['cost0_sp']:+.2f} bp/mo   |   \"\n"
            "      f\"cost  25 bps: spread {R['cost25_sp']:+.2f} bp/mo   (turnover is a rounding error)\")\n"
            "print(f\"financing  0 bp/yr: {R['fin0_sp']:+.2f} bp/mo (t {R['fin0_t']:+.2f})   |   \"\n"
            "      f\"100 bp/yr: {R['fin100_sp']:+.2f} bp/mo (t {R['fin100_t']:+.2f})\")\n"
            "print('   ^ the PROXY. A WIDER spread makes the mix worse and FLATTERS the zero leg -')\n"
            "print('     even at a punitive 100 bp/yr the zero still does not get ahead.')\n"
            "print(f\"window 126d: spread {R['w126_sp']:+.2f} bp/mo, b2 {R['w126_b2']:+.1f}   |   \"\n"
            "      f\"252d: {R['sp_bp_mo']:+.2f}, b2 {R['dy2_b2']:+.1f}   |   \"\n"
            "      f\"504d: {R['w504_sp']:+.2f}, b2 {R['w504_b2']:+.1f}\")"
        ),
        md("> 💡 **In plain words** — nothing about the plumbing decides this study. Change the "
           "costs, change the borrowing assumption, change how the hedge is estimated: the "
           "curvature term stays positive and stays insignificant, and the mean spread wobbles "
           "around zero. That *is* the finding."),
        md("## 6. Cross-check — ZROZ (25y+ zeros)"),
        code(
            "print(f\"{R['z_start']} -> 2026-06-30, {R['z_months']} months, L {R['z_L']:.3f}, \"\n"
            "      f\"vol ratio {R['z_volratio']:.3f}\")\n"
            "print(f\"spread {R['z_sp']:+.2f} bp/mo (t {R['z_sp_t']:+.2f})\")\n"
            "print(f\"b2[dy^2] {R['z_b2_dy2']:+.1f} (t {R['z_b2_dy2_t']:+.2f})   \"\n"
            "      f\"b2[rv] {R['z_b2_rv']:+.1f} (t {R['z_b2_rv_t']:+.2f})  <- a factor of 4 apart\")\n"
            "print(f\"a[dy^2] {R['z_a_dy2']:+.2f} bp/mo (t {R['z_a_dy2_t']:+.2f}); \"\n"
            "      f\"b1 t {R['z_b1_t']:+.2f} (leak milder - ZROZ sits closer to the 30y point)\")\n"
            "print('\\nsame signs, same absence of significance on the full sample, and an estimate')\n"
            "print('that is not stable across the two quadratic specifications. Nothing to bank.')"
        ),
        md(f"## 6b. The full cut census — and the only two cuts that clear |*t*| = 2\n\n"
           f"A robustness claim is only checkable if **every** cut the design implies is on the "
           f"page. The design implies **{R['grid_n']}**: two funds × three eras × two quadratic "
           f"specifications. `strategy.cut_grid` / `grid_census` run and print all of them "
           f"(`examples/verify.py`), so no sentence in this study can quote a favourable "
           f"subset.\n\n"
           f"An earlier draft reported six of them, called the sign pattern unanimous, and said "
           f"the best *t* anywhere was +{R['z_b2_dy2_t']:.2f}. Fitting the six it had not run — "
           f"the ZROZ era cut — overturned both statements."),
        code(
            f"census = dict(n={R['grid_n']}, b2_pos={R['grid_b2_pos']}, a_neg={R['grid_a_neg']},\n"
            f"              t2={R['grid_t2']}, max_t={R['grid_max_t']}, edv_max_t={R['edv_max_t']})\n"
            "print('census over the full grid of %d cuts:' % census['n'])\n"
            "print('  b2 > 0 in %2d/%d   a < 0 in %2d/%d   |t| >= 2 in %d/%d'\n"
            "      % (census['b2_pos'], census['n'], census['a_neg'], census['n'],\n"
            "         census['t2'], census['n']))\n"
            "print('  largest |t| ANYWHERE: %+.2f  (ZROZ, 2010-2017, realised variance)'\n"
            "      % census['max_t'])\n"
            f"print('  the two cuts that clear 2: ZROZ 2010-2017  b2 {R['ze_b2_dy2']:+.1f} "
            f"(t {R['ze_b2_dy2_t']:+.2f}) [dy^2]')\n"
            f"print('%38s b2 {R['ze_b2_rv']:+.1f} (t {R['ze_b2_rv_t']:+.2f}) [rv]' % '')\n"
            f"print('  ... and the era NEXT DOOR, same fund, same spec:')\n"
            f"print('     ZROZ 2018-2026  b2 {R['zl_b2_rv']:+.1f} (t {R['zl_b2_rv_t']:+.2f}), "
            f"a {R['zl_a_rv']:+.2f} bp/mo  <- BOTH SIGNS FLIP')\n"
            f"print('  EDV, the headline fund, never exceeds +{R['edv_max_t']:.2f} in any cut.')"
        ),
        md(f"> 💡 **How to read those two cuts.** They are not two findings: they are one "
           f"84-month window of the *cross-check* fund, fit twice with two near-identical "
           f"regressors, un-corrected for {R['grid_n']} looks — roughly the rate at which a "
           f"null throws up a |*t*| ≥ 2 by chance. The decisive fact is the neighbour: on "
           f"2018-2026, the era that actually contains 2020 and 2022, the same fund and the "
           f"same specification produce `b2` = **{R['zl_b2_rv']:+.1f}** with a **positive** "
           f"intercept — convexity running backwards. Reported, not promoted: this is what "
           f"`WEAK` (\"fragile to method or selection\") is for, and it is why the badge did "
           f"not move up."),
        md("## 7. The synthetic control — is the harness unbiased? (live, offline)\n\n"
           "Planted world: the long leg carries 2.5× the short leg's convexity per unit of "
           "duration and pays 30 bp/month of carry for it — `b2` must come out strongly "
           "positive and `a` strongly negative. Null world: the duration-matched mix is "
           "convexity-matched too — both must be silent. This proves the machinery; it never "
           "supports a real-tape stamp."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from zero_convexity import data, strategy as st\n"
            "pl = st.synthetic_detect(data.synthetic_panel(signal_strength=1.0, seed=950)[0])\n"
            "print(f\"planted: b2 {pl['b2']:+.1f} (t {pl['b2_t']:+.2f})  a {pl['a_bp_mo']:+.2f} bp/mo \"\n"
            "      f\"(t {pl['a_t']:+.2f})  b1 t {pl['b1_t']:+.2f}  vol ratio {pl['vol_ratio']:.3f}\")\n"
            "ts = np.array([st.synthetic_detect(\n"
            "        data.synthetic_panel(signal_strength=0.0, seed=950+s)[0])['b2_t']\n"
            "      for s in range(8)])\n"
            "print(f\"null x8: b2 t mean {ts.mean():+.2f} (sd {ts.std(ddof=1):.2f}), \"\n"
            "      f\"|t|>=2 in {(np.abs(ts)>=2).sum()}/8\")\n"
            "print('\\nplanted effect resolved at t>5; null centred on zero -> the real-tape silence')\n"
            "print('is a property of the Treasury curve, not of the regression.')"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** The convexity coefficient is positive in "
           f"**{R['grid_b2_pos']} of {R['grid_n']}** fund × era × specification cuts and the "
           f"intercept negative in {R['grid_a_neg']} of {R['grid_n']} — the sign pattern of "
           f"\"convexity is real but priced\", with one cut (ZROZ 2018-2026, realised variance) "
           f"flipping both. The headline does not clear the bar: `b2` = "
           f"**{R['dy2_b2']:+.1f}** (HAC *t* = **{R['dy2_b2_t']:+.2f}**), "
           f"**{R['rv_b2']:+.1f}** (*t* = {R['rv_b2_t']:+.2f}) on realised variance; the "
           f"**{R['grid_t2']} cuts that do** clear it (ZROZ 2010-2017, "
           f"**+{R['ze_b2_dy2_t']:.2f}** / **+{R['ze_b2_rv_t']:.2f}**) are one window of the "
           f"cross-check fund, uncorrected for {R['grid_n']} looks, inverted by the era beside "
           f"it; bootstrap CIs "
           f"[{R['b2_dy2_lo']:+.0f}, {R['b2_dy2_hi']:+.0f}] and "
           f"[{R['b2_rv_lo']:+.0f}, {R['b2_rv_hi']:+.0f}]; the raw large-move bucket is "
           f"**{R['bk_l']:+.2f} bp/mo**, i.e. the wrong sign; and a significant "
           f"**+0.72 yr** residual duration (*t* = {R['dy2_b1_t']:+.2f}, era-stable) means part "
           f"of the spread is a 20s-30s curve trade. The synthetic control resolves a planted "
           f"pickup at *t* = {R['syn_b2_t']:+.2f} and is silent on the null "
           f"({R['syn_null_fire']}/8), so the miss is the tape's, not the harness's. "
           f"**Survivorship / selection:** four named, still-listed funds and one official "
           f"yield series, but EDV and ZROZ are the *only* US zero-coupon Treasury ETFs that "
           f"survived to be testable.\n"
           f"- **Tradability — Mirage.** The spread earns **{R['sp_mean']:+.2f}%/yr** "
           f"(monthly Sharpe {R['sp_sharpe']:+.3f}, *t* {R['sp_t']:+.2f}); the fitted convexity "
           f"needs a **{R['dy2_be']:.0f} bp** month to repay its own carry against a median "
           f"month of **{R['median_absdy']:.0f} bp**; the estimate does not survive the change of "
           f"quadratic regressor on ZROZ ({R['z_b2_dy2']:+.0f} → {R['z_b2_rv']:+.0f}). Costs and "
           f"the financing proxy are immaterial. The zero fund buys you **more duration per "
           f"dollar** — a real and useful property — and nothing else you can measure."),
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
