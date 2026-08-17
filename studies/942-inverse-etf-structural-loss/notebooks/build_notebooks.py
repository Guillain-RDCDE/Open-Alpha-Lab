"""Generate the two narrative notebooks for Study 942 (The Inverse Tax).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministically. Every real-tape number is quoted from the
frozen ``R`` dict below, which mirrors ``docs/results.md``; the only live cells run the fast
synthetic control, and they are never placed under a real-tape banner.
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


# Frozen real-tape headline — mirror of docs/results.md. SH/PSQ/SDS vs a daily-reset direct
# short, total-return closes, 2007-05-31 -> 2026-06-30, base proxies c=1, borrow 30 bps,
# rebalance cost 1 bp one-way x NAV.
R = dict(
    start="2007-05-31", end="2026-06-30", n_days=4799, fp="b9485964fb9d",
    credit=1.0, borrow_bps=30.0, cost_bps=1.0,
    irx_ann=1.47, bil_ann=1.36, irx_bil_diff=0.11,
    sh_gap=1.06, sh_t=5.10, sh_t_iid=2.10, sh_sd=13.8, sh_ci_lo=0.71, sh_ci_hi=1.45,
    psq_gap=0.58, psq_t=2.50, psq_sd=16.6, psq_ci_lo=0.20, psq_ci_hi=0.97,
    sds_gap=1.19, sds_t=3.78, sds_sd=21.6, sds_ci_lo=0.69, sds_ci_hi=1.74,
    # Same-mandate counterweight: the fund charged the index's dividends too.
    sh_sm=-0.79, sh_sm_t=-2.76, psq_sm=-0.23, psq_sm_t=-0.94,
    sds_sm=-2.51, sds_sm_t=-4.87,
    be_sm_sh=0.46, be_sm_psq=0.84, be_sm_sds=0.14,
    sm_zirp=-2.52, sm_zirp_t=-6.61, sm_norm=1.59, sm_norm_t=4.20,
    # HAC lag profile of the SH gap (Bartlett truncation, 0 = i.i.d.).
    hac_profile=((0, 2.10), (1, 2.82), (2, 3.21), (5, 4.61), (9, 5.10),
                 (21, 5.50), (63, 4.34), (126, 3.27), (252, 2.44)),
    gap_acf1=-0.44,
    sh_exsharpe=-0.574, dir_exsharpe=-0.627, sharpe_diff=0.052,
    sh_cagr=-11.22, dir_cagr=-12.16, sh_vol=19.7, dir_vol=19.8,
    sh_dd=-94.7, dir_dd=-95.4,
    sh_raw=2.18, sh_div=1.85, sh_fin=1.74, sh_er=0.89, sh_resid=-0.52,
    sh_beta=0.9914, sh_gamma=1.187, sh_gamma_px=1.400, sh_alpha=0.33,
    psq_raw=1.69, psq_div=0.81, psq_fin=2.01, psq_er=0.95, psq_resid=-0.18,
    psq_beta=0.9917, psq_gamma=1.371, psq_gamma_px=1.263,
    sds_raw=3.39, sds_div=3.70, sds_fin=2.80, sds_er=0.89, sds_resid=-2.21,
    sds_beta=0.9823, sds_gamma=0.954, sds_gamma_px=1.164,
    spy_yield=1.85, qqq_yield=0.81,
    be_sh=1.72, be_psq=1.39, be_sds=1.41,
    era_e_n=2163, era_e_gap=0.16, era_e_t=0.39, era_e_rf=0.49,
    era_l_n=2636, era_l_gap=1.79, era_l_t=9.54, era_l_rf=2.26,
    zirp_n=2785, zirp_gap=-0.46, zirp_t=-1.83, zirp_rf=0.15,
    norm_n=2014, norm_gap=3.15, norm_t=11.37, norm_rf=3.29,
    c0_b0=2.22, c0_b0_t=8.74, c1_b30=1.06, c1_b30_t=5.10,
    c15_b0=0.02, c15_b0_t=0.09, c2_b0=-0.72, c2_b0_t=-3.81,
    c2_b30=-0.41, c2_b30_t=-2.19,
    cost0=1.01, cost0_t=4.90, cost10=1.42, cost10_t=6.82,
    sh_drag_63=-0.34, sh_drag_63_med=-0.26, sh_wrap_63=0.27,
    sds_drag_63=-1.12, sds_drag_63_med=-0.76, sds_wrap_63=0.38,
    sds_drag_252=-2.68, sh_drag_252=-0.80,
    syn_plant=-2.77, syn_plant_t=-22.06, syn_null=0.23, syn_null_t=1.84,
    syn_null_mean=0.027, syn_null_sd=0.159, syn_null_fire=0,
    syn_plant_mean=-2.973, syn_seeds=12,
)


HEADER = f"""# Study 942 — The Inverse Tax 🔻

**Is an inverse ETF really a worse short than shorting the index yourself?**

The standard warning on SH, PSQ and SDS levies three charges: you pay an *expense ratio* a
direct shorter does not, you eat the *daily-reset path drag*, and you hand the sponsor the
*financing* a short seller would collect as a rebate. So — the story goes — anyone who can
borrow the shares should short the index outright.

We build the honest alternative and race it: a directly-short book rebalanced daily to the
same −k× NAV, which **owes the index's dividends**, pays a stock-loan fee, and collects
whatever rebate its broker credits (modelled off `^IRX`). Real tape {R['start']} → {R['end']}
({R['n_days']:,} days), daily total-return closes plus price-only SPY/QQQ closes for the
dividend leg.

*Numbers below are the frozen headline run (`docs/results.md`, Fingerprint `{R['fp']}`); the
only live cells run the fast offline synthetic control. As-of 2026-06-30.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The three charges, and what they are worth\n\n"
           "Before any arithmetic, notice what the story leaves out. The ProShares inverse "
           "S&P 500 and Nasdaq-100 funds track the **price** index — so their holder never "
           "owes the dividends a share-short is charged for. And the funds hold their whole "
           "NAV in collateral and swaps struck against a short rate, while a *retail* short "
           "seller is usually paid nothing at all on the proceeds. Two credits the folklore "
           "never counts, against three charges it does."),
        md(f"## 2. The result — the fund wins, at the financing you actually get\n\n"
           f"Base case: your own cash earns the bill rate, the short proceeds earn nothing "
           f"(the ordinary retail arrangement), 30 bps to borrow the shares, 1 bp a day of "
           f"rebalancing cost on the self-managed book."),
        code(
            ("R = %r\n" % (R,))
            + "print('gap = inverse ETF minus an honest direct short (positive = the FUND wins)')\n"
            "print(f\"  SH  (-1x S&P 500) : {R['sh_gap']:+.2f}%/yr   HAC t {R['sh_t']:+.2f}\")\n"
            "print(f\"  PSQ (-1x Nasdaq)  : {R['psq_gap']:+.2f}%/yr   HAC t {R['psq_t']:+.2f}\")\n"
            "print(f\"  SDS (-2x S&P 500) : {R['sds_gap']:+.2f}%/yr   HAC t {R['sds_t']:+.2f}\")\n"
            "print(f\"  SH bootstrap 95% CI: [{R['sh_ci_lo']:+.2f}, {R['sh_ci_hi']:+.2f}] \"\n"
            "      f\"-- clear of zero\")"
        ),
        md(f"The folklore says that column should be **negative**. On all three funds it is "
           f"positive, and on SH it is positive with a *t* of **{R['sh_t']:.2f}**.\n\n"
           f"> 🔬 **For the quants:** the *t* is Newey-West on the daily return difference "
           f"(the plain i.i.d. *t* is {R['sh_t_iid']:+.2f}; the gap's lag-1 autocorrelation is "
           f"{R['gap_acf1']:.2f}, which is why HAC *raises* it), and the interval is a circular "
           f"block bootstrap (2,000 draws, 21-day blocks) on the annualised mean gap. Both arms "
           f"are raced excess-of-cash against BIL: excess Sharpe {R['sh_exsharpe']:+.3f} (fund) "
           f"vs {R['dir_exsharpe']:+.3f} (direct)."),
        md("## 2b. The catch — that race is not quite apples to apples\n\n"
           "SH tracks the **price** index. Someone who shorts SPY shares is short the "
           "**total return** and has to hand over the dividends. So part of what we just "
           "measured is not the fund being *better* — it is the fund being short a "
           "*different, slightly easier* thing. Charge the fund those dividends too and run "
           "the race again:"),
        code(
            "print('same-mandate race: BOTH arms charged the index dividends')\n"
            "for tag, head, sm, t in [\n"
            "    ('SH ', R['sh_gap'],  R['sh_sm'],  R['sh_sm_t']),\n"
            "    ('PSQ', R['psq_gap'], R['psq_sm'], R['psq_sm_t']),\n"
            "    ('SDS', R['sds_gap'], R['sds_sm'], R['sds_sm_t']),\n"
            "]:\n"
            "    print(f\"  {tag}: headline {head:+.2f}%/yr  ->  same-mandate {sm:+.2f}%/yr \"\n"
            "          f\"(HAC t {t:+.2f})\")\n"
            "print('\\nthe sign flips. both signs are significant. that is the whole study.')"
        ),
        md(f"**Both framings are legitimate, and they disagree.** If your question is *\"buy "
           f"the fund or short the shares?\"*, the dividends are part of the answer and the fund "
           f"wins. If your question is *\"is the wrapper an efficient way to carry a given short "
           f"exposure?\"*, you take the dividends out and the fund loses — {R['sh_sm']:+.2f} %/yr "
           f"on SH, {R['sds_sm']:+.2f} on the −2×. This is why the Signal stamp is **Mixed** "
           f"rather than a clean yes or no."),
        md("## 3. Where the money is — the four terms\n\n"
           "Strip the direct book of *all* interest and the raw gap breaks into four pieces. "
           "The expense ratio is the only one the folklore names, and it is the smallest."),
        code(
            "print(f\"SH, raw gap {R['sh_raw']:+.2f}%/yr, against a direct short \"\n"
            "      f\"paid no interest at all:\")\n"
            "for label, value in [\n"
            "    ('dividends a share-short owes, the fund does not', R['sh_div']),\n"
            "    ('short-rate interest the fund credits on NAV',     R['sh_fin']),\n"
            "    ('the expense ratio',                               -R['sh_er']),\n"
            "    ('the wrapper residual (swap spread, slippage)',    R['sh_resid']),\n"
            "]:\n"
            "    print(f\"   {value:+6.2f}%/yr  {label}\")"
        ),
        md(f"SPY paid **{R['spy_yield']:.2f}%/yr** of dividends over this window — a bill a "
           f"share-short foots and an SH holder does not. And the fund credits about "
           f"**{R['sh_gamma']:.2f} units** of the 13-week bill rate on its NAV, where a retail "
           f"shorter collects one. Together those two dwarf the {R['sh_er']:.2f}% expense "
           f"ratio and the {abs(R['sh_resid']):.2f}% the wrapper genuinely costs."),
        md(f"## 4. The daily reset is real — and it is not the fund's fault\n\n"
           f"Over a quarter, keeping a book at −2× costs **{R['sds_drag_63']:.2f} pp** versus "
           f"putting a short on once and leaving it (median {R['sds_drag_63_med']:.2f}); at −1× "
           f"it costs {R['sh_drag_63']:.2f} pp. That drag is genuine. But it belongs to the "
           f"*discipline of holding constant leverage*, not to the wrapper: a self-managed book "
           f"rebalanced to −1× every day pays exactly the same bill. Once you compare the fund "
           f"with a **daily-reset** book instead of a static one, the wrapper's own column is "
           f"small and, if anything, slightly positive ({R['sds_wrap_63']:+.2f} pp per quarter "
           f"on SDS). Blaming the reset on the ETF is a category error."),
        md(f"## 5. So when *is* the folklore right? When money is free.\n\n"
           f"Cut the sample by the prevailing 13-week bill rate — a level you know at the start "
           f"of each day, so this is not hindsight."),
        code(
            "print(f\"bills below 1% (n={R['zirp_n']:,}, avg {R['zirp_rf']:.2f}%/yr): \"\n"
            "      f\"gap {R['zirp_gap']:+.2f}%/yr   HAC t {R['zirp_t']:+.2f}\")\n"
            "print(f\"bills above 1% (n={R['norm_n']:,}, avg {R['norm_rf']:.2f}%/yr): \"\n"
            "      f\"gap {R['norm_gap']:+.2f}%/yr   HAC t {R['norm_t']:+.2f}\")\n"
            "print()\n"
            "print(f\"break-even: the fund wins until your broker credits you {R['be_sh']:.2f} \"\n"
            "      f\"units of the bill rate.\")\n"
            "print('retail is about 1.0; a prime-brokerage account is about 2.0.')"
        ),
        md(f"In a zero-rate world there is no financing to pass through, so the expense ratio "
           f"and the reset residual are all that is left and the fund is indeed the marginally "
           f"worse short — **{R['zirp_gap']:.2f}%/yr** on the product-vs-product race (*t* = "
           f"{R['zirp_t']:.2f}, short of the bar) and **{R['sm_zirp']:.2f}%/yr** on the "
           f"same-mandate one (*t* = {R['sm_zirp_t']:.2f}, well past it). When bills pay "
           f"{R['norm_rf']:.1f}% the fund beats a retail direct short by "
           f"**{R['norm_gap']:+.2f}%/yr**. The \"inverse tax\" is a ZIRP-era memory."),
        md("## 6. Live check — the machinery is unbiased (offline synthetic, no real data)\n\n"
           "We simulate an inverse fund that really *does* leak a known 3%/yr, and a second "
           "one that is a costless replicate. The harness must find the first and stay quiet "
           "on the second."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from inverse_tax import data, strategy as st\n"
            "taxed, t1 = data.synthetic_daily(signal_strength=1.0, seed=942)\n"
            "fair,  t0 = data.synthetic_daily(signal_strength=0.0, seed=942)\n"
            "d1, d0 = st.synthetic_detect(taxed), st.synthetic_detect(fair)\n"
            "print('planted a %+.2f%%/yr tax -> measured %+.2f%%/yr (t %+.2f)'\n"
            "      % (t1['expected_gap_ann']*100, d1['gap_ann_pct'], d1['gap_t']))\n"
            "print('planted no tax at all   -> measured %+.2f%%/yr (t %+.2f)'\n"
            "      % (d0['gap_ann_pct'], d0['gap_t']))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Mixed.** There is a real, tightly measured structural gap "
           f"(**{R['sh_gap']:+.2f} %/yr** on SH, HAC *t* = **{R['sh_t']:+.2f}**, replicated on "
           f"PSQ and SDS) — but it is a *subsidy*, not the advertised tax, at the financing an "
           f"ordinary shorter actually gets. Its sign is decided by three things outside the "
           f"price tape: the level of short rates, what your broker pays you, and whether you "
           f"charge the fund for dividends it does not owe (do that and SH reads "
           f"{R['sh_sm']:+.2f} %/yr, *t* {R['sh_sm_t']:+.2f}).\n"
           f"- **Tradability — Fragile.** Worth about a point a year at retail terms, "
           f"{R['norm_gap']:+.2f} %/yr at 2023-2026 rates — but it is an implementation choice "
           f"on a hedge, not a return stream (both arms lost {abs(R['sh_cagr']):.0f}-"
           f"{abs(R['dir_cagr']):.0f} %/yr while the index quintupled), and it reverses for a "
           f"prime broker and evaporates at zero rates.\n"
           f"- **What was busted.** The expense ratio is the *smallest* of the three charges; "
           f"the daily-reset drag is not the wrapper's; and the financing leg — the one that "
           f"carries the claim — points the other way for everyone who is not a prime broker."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 942 — The Inverse Tax — the teardown\n\n"
           "The replicate's accounting, the HAC *t* and block-bootstrap CI on the daily gap, "
           "the financing-passthrough regression, the break-even rebate credit, the era and "
           "rate-regime cuts, both proxy sweeps, the path-drag table, and the live synthetic "
           "control. Every real number is frozen from `docs/results.md` "
           "(Fingerprint `%s`)." % R["fp"]),
        md("## The two arms, stated exactly\n\n"
           "Per $1 of NAV, no timing signal anywhere:\n\n"
           "1. **fund** — the inverse ETF's own daily total return (SH, PSQ, SDS).\n"
           "2. **direct** — a directly-short book rebalanced daily to the same −k× NAV:\n\n"
           "$$ r^{direct}_t \\;=\\; k\\,r^{index,TR}_t \\;+\\; c\\,|k|\\,rf_t \\;-\\; "
           "\\text{borrow}_t \\;-\\; \\text{cost}_t $$\n\n"
           "The **total-return** leg is deliberate: a share-short owes the distributions. "
           "`rf` is the `^IRX` act/360 accrual over calendar days at the *previous* close's "
           "level. **One execution lag, applied identically to both arms:** the exposure set "
           "at the close of `t−1` earns day `t`'s return and finances at `t−1`'s rate.\n\n"
           "`c` (rebate credit) and the borrow fee are **PROXY / ASSUMPTION** inputs, not tape — "
           "both are swept below, and the break-even `c` is reported.\n\n"
           "> 💡 **In plain words:** we rebuild, day by day, the P&L of someone who shorted the "
           "index themselves — paying the dividends, paying to borrow the shares, earning "
           "whatever interest their broker hands back — and put it next to the ETF."),
        code("R = %r" % (R,)),
        md("## Headline — the daily gap"),
        code(
            "for tag, g, t, sd, lo, hi in [\n"
            "    ('SH  (-1x SPY)', R['sh_gap'],  R['sh_t'],  R['sh_sd'],  R['sh_ci_lo'],  R['sh_ci_hi']),\n"
            "    ('PSQ (-1x QQQ)', R['psq_gap'], R['psq_t'], R['psq_sd'], R['psq_ci_lo'], R['psq_ci_hi']),\n"
            "    ('SDS (-2x SPY)', R['sds_gap'], R['sds_t'], R['sds_sd'], R['sds_ci_lo'], R['sds_ci_hi']),\n"
            "]:\n"
            "    print(f\"{tag}: gap {g:+.2f}%/yr  HAC t {t:+.2f}  daily sd {sd:.1f} bps  \"\n"
            "          f\"boot 95% CI [{lo:+.2f}, {hi:+.2f}]\")\n"
            "print(f\"\\nn={R['n_days']:,} days, {R['start']} -> {R['end']}, \"\n"
            "      f\"c={R['credit']:.0f}, borrow={R['borrow_bps']:.0f} bps, cost={R['cost_bps']:.0f} bp\")\n"
            "print(f\"excess-of-cash (minus BIL): fund exSharpe {R['sh_exsharpe']:+.3f} vs \"\n"
            "      f\"direct {R['dir_exsharpe']:+.3f} (diff {R['sharpe_diff']:+.3f}); \"\n"
            "      f\"absolute CAGR {R['sh_cagr']:+.2f}% vs {R['dir_cagr']:+.2f}%\")\n"
            "print(f\"financing cross-check: ^IRX accrual {R['irx_ann']:.2f}%/yr vs BIL TR \"\n"
            "      f\"{R['bil_ann']:.2f}%/yr (wedge {R['irx_bil_diff']:+.2f} pp)\")"
        ),
        md("Both arms are short books over a sample in which the index rose roughly five-fold, "
           "so both excess Sharpes are deeply negative and neither is interpretable on its own. "
           "The **gap** is the estimand."),
        md("## The same-mandate counterweight — where the sign comes from\n\n"
           "The headline race is *product vs product*. It is **not** an exposure-matched race: "
           "SH/PSQ/SDS track the **price** index, while a share-short is short the "
           "**total-return** stream and owes the distributions. That term is the largest single "
           "piece of the raw gap. Debit the fund `|k| ×` the realised distribution yield and both "
           "arms sit on the total-return leg — the strictly like-for-like comparison:"),
        code(
            "hdr = f\"{'fund':5s} {'headline':>10s} {'same-mandate':>14s} {'HAC t':>8s} {'c* (same)':>10s}\"\n"
            "print(hdr); print('-'*len(hdr))\n"
            "for f, h, sm, t, c in [\n"
            "    ('SH',  R['sh_gap'],  R['sh_sm'],  R['sh_sm_t'],  R['be_sm_sh']),\n"
            "    ('PSQ', R['psq_gap'], R['psq_sm'], R['psq_sm_t'], R['be_sm_psq']),\n"
            "    ('SDS', R['sds_gap'], R['sds_sm'], R['sds_sm_t'], R['be_sm_sds']),\n"
            "]:\n"
            "    print(f\"{f:5s} {h:+10.2f} {sm:+14.2f} {t:+8.2f} {c:10.2f}\")\n"
            "print(f\"\\nSH same-mandate by regime: ZIRP {R['sm_zirp']:+.2f}%/yr \"\n"
            "      f\"(t {R['sm_zirp_t']:+.2f})   bills>=1% {R['sm_norm']:+.2f}%/yr \"\n"
            "      f\"(t {R['sm_norm_t']:+.2f})\")"
        ),
        md("**The estimand's sign is a definitional choice and both signs clear |*t*| ≥ 2.** "
           "Neither framing is wrong; quoting only one would be. The stamp is Mixed for exactly "
           "this reason, and no robustness cut rescues a single sign."),
        md(f"## How much of the *t* is the HAC bandwidth?\n\n"
           f"The daily gap's lag-1 autocorrelation is **{R['gap_acf1']:.2f}** — the ETF's closing "
           f"print against the index close, bid-ask bounce and premium/discount noise that "
           f"reverses the next day. HAC therefore *reduces* the variance relative to i.i.d., so "
           f"the automatic-bandwidth *t* is not comparable to a naive one. The profile, not the "
           f"point:"),
        code(
            "print('Bartlett lags -> HAC t on the SH gap (0 = i.i.d.)')\n"
            "for lag, t in R['hac_profile']:\n"
            "    mark = '   <- automatic bandwidth' if lag == 9 else ''\n"
            "    print(f\"  {lag:4d}: {t:+.2f}{mark}\")\n"
            "ts = [t for _, t in R['hac_profile']]\n"
            "print(f\"\\nrange {min(ts):+.2f} to {max(ts):+.2f}; \"\n"
            "      f\"|t|>=2 at every truncation, but the conservative read is the i.i.d. one.\")"
        ),
        md("## Decomposition of the raw gap (c = 0, no borrow, no cost)\n\n"
           "γ is the OLS loading of the fund's daily return on `|k|·rf` in\n\n"
           "$$ r^{fund}_t = \\alpha + \\beta\\,(k\\,r^{index,TR}_t) + \\gamma\\,(|k|\\,rf_t) + "
           "\\varepsilon_t $$\n\n"
           "so the financing passthrough is **measured**, not assumed. The expense ratio is the "
           "one stated (non-tape) term; the residual absorbs swap spread, reset slippage and "
           "tracking error."),
        code(
            "hdr = f\"{'fund':5s} {'raw':>7s} {'divs':>7s} {'financing':>10s} {'-ER':>7s} {'residual':>9s} {'beta':>7s} {'gamma':>7s}\"\n"
            "print(hdr); print('-'*len(hdr))\n"
            "for f, raw, dv, fin, er, res, b, g in [\n"
            "    ('SH',  R['sh_raw'],  R['sh_div'],  R['sh_fin'],  R['sh_er'],  R['sh_resid'],  R['sh_beta'],  R['sh_gamma']),\n"
            "    ('PSQ', R['psq_raw'], R['psq_div'], R['psq_fin'], R['psq_er'], R['psq_resid'], R['psq_beta'], R['psq_gamma']),\n"
            "    ('SDS', R['sds_raw'], R['sds_div'], R['sds_fin'], R['sds_er'], R['sds_resid'], R['sds_beta'], R['sds_gamma']),\n"
            "]:\n"
            "    print(f\"{f:5s} {raw:+7.2f} {dv:+7.2f} {fin:+10.2f} {-er:+7.2f} {res:+9.2f} {b:7.4f} {g:7.3f}\")\n"
            "print(f\"\\nall %/yr. index yields: SPY {R['spy_yield']:.2f}%, QQQ {R['qqq_yield']:.2f}%; \"\n"
            "      f\"bills {R['irx_ann']:.2f}%\")\n"
            "print('\\nSPEC CHECK - gamma re-estimated on the PRICE leg (what the funds track):')\n"
            "for f, kabs, g_tr, g_px in [('SH', 1.0, R['sh_gamma'], R['sh_gamma_px']),\n"
            "                            ('PSQ', 1.0, R['psq_gamma'], R['psq_gamma_px']),\n"
            "                            ('SDS', 2.0, R['sds_gamma'], R['sds_gamma_px'])]:\n"
            "    print(f\"  {f:4s} gamma {g_tr:.3f} (TR leg) vs {g_px:.3f} (price leg) \"\n"
            "          f\"-> financing term moves {(g_px-g_tr)*kabs*R['irx_ann']:+.2f}%/yr\")\n"
            "print('  the RAW GAP is spec-free; only the financing/residual SPLIT moves.')"
        ),
        md(f"β ≈ 0.99 on the −1× funds and {R['sds_beta']:.4f} on the −2×: the funds deliver "
           f"essentially their stated exposure, so the gap is *not* tracking failure. "
           f"γ = **{R['sh_gamma']:.3f}** (SH) and **{R['psq_gamma']:.3f}** (PSQ) — between one "
           f"and two units of the bill rate on NAV, which is the crux: a retail direct shorter "
           f"is at `c = 1`.\n\n"
           f"> 💡 **In plain words:** the funds hand back more short-rate interest than an "
           f"ordinary shorter is credited, and they never pay the index's dividends. Those two "
           f"credits are worth more than the fees they charge."),
        md("## Break-even rebate credit\n\n"
           "The gap is exactly linear in `c` (each unit costs the direct book `|k|·rf`), so the "
           "indifference point is a single number the reader can place their own account against."),
        code(
            "for f, c in [('SH', R['be_sh']), ('PSQ', R['be_psq']), ('SDS', R['be_sds'])]:\n"
            "    print(f\"{f:4s}: c* = {c:.2f} units of the bill rate\")\n"
            "print('\\nretail account ~ c=1.0  ->  the FUND wins')\n"
            "print('prime brokerage ~ c=2.0 ->  the DIRECT SHORT wins')"
        ),
        md("## Robustness — era cut and rate-regime cut\n\n"
           "The rate-regime partition uses the *previous* close's 13-week bill level, known at "
           "the start of each day: an ex-ante split, not a look-ahead sort."),
        code(
            "print(f\"2007-2015 (n={R['era_e_n']:,}, bills {R['era_e_rf']:.2f}%/yr): \"\n"
            "      f\"gap {R['era_e_gap']:+.2f}%/yr  t {R['era_e_t']:+.2f}\")\n"
            "print(f\"2016-2026 (n={R['era_l_n']:,}, bills {R['era_l_rf']:.2f}%/yr): \"\n"
            "      f\"gap {R['era_l_gap']:+.2f}%/yr  t {R['era_l_t']:+.2f}\")\n"
            "print()\n"
            "print(f\"bills <1%  (n={R['zirp_n']:,}, avg {R['zirp_rf']:.2f}%/yr): \"\n"
            "      f\"gap {R['zirp_gap']:+.2f}%/yr  t {R['zirp_t']:+.2f}  <- the folklore's regime\")\n"
            "print(f\"bills >=1% (n={R['norm_n']:,}, avg {R['norm_rf']:.2f}%/yr): \"\n"
            "      f\"gap {R['norm_gap']:+.2f}%/yr  t {R['norm_t']:+.2f}\")"
        ),
        md(f"The gap is not a constant — it is a rate story, and the era cut is the same "
           f"statement in calendar clothing. Note that even in ZIRP, its own best regime, the "
           f"claimed tax reaches only *t* = {R['zirp_t']:.2f}: it is never *robustly* "
           f"demonstrated anywhere on this tape."),
        md("## Proxy sweeps — the honest sensitivity"),
        code(
            "print('SH, gap %/yr (HAC t) across the two PROXY inputs:')\n"
            "print(f\"  c=0.0, borrow=  0 bps : {R['c0_b0']:+.2f} ({R['c0_b0_t']:+.2f})\")\n"
            "print(f\"  c=1.0, borrow= 30 bps : {R['c1_b30']:+.2f} ({R['c1_b30_t']:+.2f})   <- base case\")\n"
            "print(f\"  c=1.5, borrow=  0 bps : {R['c15_b0']:+.2f} ({R['c15_b0_t']:+.2f})   <- indifference\")\n"
            "print(f\"  c=2.0, borrow= 30 bps : {R['c2_b30']:+.2f} ({R['c2_b30_t']:+.2f})\")\n"
            "print(f\"  c=2.0, borrow=  0 bps : {R['c2_b0']:+.2f} ({R['c2_b0_t']:+.2f})   <- the ONLY corner\")\n"
            "print( \"                                             that reproduces the claimed tax\")\n"
            "print()\n"
            "print('rebalance-cost sweep on the direct book (c=1, 30 bps borrow):')\n"
            "print(f\"  gross (0 bps): {R['cost0']:+.2f}%/yr (t {R['cost0_t']:+.2f})   \"\n"
            "      f\"10 bps: {R['cost10']:+.2f}%/yr (t {R['cost10_t']:+.2f})\")\n"
            "print('  costs only WIDEN the fund advantage - it is the direct book that must be re-levered daily')"
        ),
        md("## Path drag — and its correct owner\n\n"
           "Non-overlapping holding windows. `daily-reset − static` is the pure constant-leverage "
           "path drag of Cheng-Madhavan / Avellaneda-Zhang; `fund − daily-reset` is what the "
           "wrapper itself adds."),
        code(
            "print(f\"SH  63d: daily-reset - static {R['sh_drag_63']:+.2f} pp \"\n"
            "      f\"(med {R['sh_drag_63_med']:+.2f})  |  fund - daily-reset {R['sh_wrap_63']:+.2f} pp\")\n"
            "print(f\"SDS 63d: daily-reset - static {R['sds_drag_63']:+.2f} pp \"\n"
            "      f\"(med {R['sds_drag_63_med']:+.2f})  |  fund - daily-reset {R['sds_wrap_63']:+.2f} pp\")\n"
            "print(f\"SH 252d: {R['sh_drag_252']:+.2f} pp     SDS 252d: {R['sds_drag_252']:+.2f} pp\")\n"
            "print()\n"
            "print('ratio of the -2x drag to the -1x drag at 63d: %.1fx  (theory (k^2-k)/2 -> 3x)'\n"
            "      % (R['sds_drag_63'] / R['sh_drag_63']))"
        ),
        md("The drag scales with |k| about as `(k² − k)/2 · σ²` predicts. But it sits entirely in "
           "the **`daily-reset − static`** column — the column a self-managed constant-leverage "
           "book pays too. The wrapper's own column is small and slightly positive.\n\n"
           "> 💡 **In plain words:** the decay is real, and it is the price of keeping your "
           "short the same size every day. Buying the fund does not create it; doing it "
           "yourself does not avoid it."),
        md("## Live synthetic control — the machinery is unbiased\n\n"
           "Offline, no cache, fixed seeds. A simulated fund with a **planted** 3%/yr tax must "
           "be caught with the right sign and size; a costless replicate must read zero. This "
           "proves the harness measures what it claims — it never supports the real-tape stamp."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from inverse_tax import data, strategy as st\n"
            "seeds = tuple(942 + s for s in range(12))\n"
            "null  = [st.synthetic_detect(p) for p, _ in data.synthetic_panel(seeds=seeds, signal_strength=0.0)]\n"
            "plant = [st.synthetic_detect(p) for p, _ in data.synthetic_panel(seeds=seeds, signal_strength=1.0)]\n"
            "ng = np.array([d['gap_ann_pct'] for d in null]);  nt = np.array([d['gap_t'] for d in null])\n"
            "pg = np.array([d['gap_ann_pct'] for d in plant])\n"
            "print('null  x%d: gap mean %+.3f%%/yr (sd %.3f), |t|>=2 on %d/%d seeds'\n"
            "      % (len(seeds), ng.mean(), ng.std(ddof=1), (abs(nt) >= 2).sum(), len(seeds)))\n"
            "print('plant x%d: gap mean %+.3f%%/yr (sd %.3f) against a planted -3.00, right sign %d/%d'\n"
            "      % (len(seeds), pg.mean(), pg.std(ddof=1), (pg < 0).sum(), len(seeds)))"
        ),
        code(
            "d = st.decompose(st.synthetic_frame(\n"
            "        data.synthetic_daily(signal_strength=1.0, tax_ann_full=0.01,\n"
            "                             div_yield_ann=0.02, passthrough=1.5,\n"
            "                             rate_ann=0.04, seed=942)[0]),\n"
            "        fund='FUND', index='IDX', k=-1.0, expense_ratio_pct=0.0)\n"
            "print('decomposition on a tape with a planted 2.00% dividend yield and a 1.500 passthrough:')\n"
            "print('  recovered dividend yield %.2f%%   recovered gamma %.3f   beta %.4f'\n"
            "      % (d['div_yield_ann_pct'], d['gamma'], d['beta']))"
        ),
        md("## What this does not establish\n\n"
           "- **Survivorship.** SH, PSQ and SDS survived from 2006-2007 to today; the industry "
           "has liquidated a long tail of inverse and leveraged-inverse products whose wrapper "
           "residuals are not in this sample. The −0.52 %/yr residual on SH is a *survivor's* "
           "residual, and the three tickers are an explicit, hindsight-flavoured universe pick "
           "(largest and longest-lived of their kind).\n"
           "- **No recall / buy-in risk** on the direct book: it is assumed able to borrow SPY "
           "and QQQ every day at a flat fee. That charge is zero here and biases *against* the "
           "fund, whose shareholder cannot be recalled.\n"
           "- **No tax, no margin mechanics**, and the bill rate is `^IRX`, a discount quote "
           f"rather than a broker rate sheet (BIL cross-check bounds the error at "
           f"{R['irx_bil_diff']:.2f} pp/yr).\n"
           "- **The financing/residual split is spec-dependent** (γ on the TR leg vs the price "
           "leg); the raw gap is not."),
        md(f"## Verdict\n\n"
           f"- **Signal — Mixed.** The structural gap is real and precisely estimated: "
           f"**{R['sh_gap']:+.2f} %/yr** on SH with HAC *t* = **{R['sh_t']:+.2f}** "
           f"(i.i.d. *t* {R['sh_t_iid']:+.2f}) and a "
           f"bootstrap CI [{R['sh_ci_lo']:+.2f}, {R['sh_ci_hi']:+.2f}] clear of zero, "
           f"replicated on PSQ ({R['psq_gap']:+.2f}, *t* {R['psq_t']:+.2f}) and SDS "
           f"({R['sds_gap']:+.2f}, *t* {R['sds_t']:+.2f}). But **its sign is a choice, not a "
           f"measurement**: on the same-mandate race it is {R['sh_sm']:+.2f} %/yr "
           f"(*t* {R['sh_sm_t']:+.2f}) on SH and {R['sds_sm']:+.2f} (*t* {R['sds_sm_t']:+.2f}) "
           f"on SDS — the folklore's sign, also significant. On top of that the break-even "
           f"rebate credit is **{R['be_sh']:.2f}** units of the bill rate, the gap swings from "
           f"{R['zirp_gap']:+.2f} %/yr in ZIRP to {R['norm_gap']:+.2f} %/yr when bills pay "
           f"{R['norm_rf']:.1f}%, and the early era is a coin-flip "
           f"({R['era_e_gap']:+.2f} %/yr, *t* {R['era_e_t']:+.2f}). Mixed, not Real.\n"
           f"- **Tradability — Fragile.** Bankable only as an implementation choice on a hedge "
           f"sleeve: both arms lost {abs(R['sh_cagr']):.1f}%/yr and "
           f"{abs(R['dir_cagr']):.1f}%/yr absolute over a sample in which the index quintupled. "
           f"The plausible range of the two proxies spans the sign change, the edge evaporates "
           f"at zero rates, and the −2× fund still carries a {R['sds_resid']:.2f} %/yr wrapper "
           f"residual on top of {R['sds_drag_63']:.2f} pp a quarter of reset drag.\n"
           f"- **The claim, itemised.** Expense ratio: real, and the *smallest* term. Daily-reset "
           f"drag: real, and not the wrapper's — a self-managed constant-leverage book pays it "
           f"identically. Financing-and-dividends: the leg that carries the whole claim, and on "
           f"this tape it is genuinely two-sided — it favours the fund at retail terms on a "
           f"product-vs-product comparison, and favours the direct short for a prime broker, at "
           f"zero rates, or on a same-mandate comparison."),
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
