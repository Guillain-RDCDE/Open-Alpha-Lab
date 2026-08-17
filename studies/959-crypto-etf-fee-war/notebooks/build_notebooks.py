"""Generate the two narrative notebooks for Study 959 (Crypto Fee War).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below, which mirrors ``docs/results.md``; the only live cells run the
fast synthetic control, and they are never placed under a real-tape heading.
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
# Ten US spot-bitcoin ETFs, common window 2024-01-11 -> 2026-06-30, as-of 2026-06-30.
# --------------------------------------------------------------------------- #
R = dict(
    start="2024-01-11", end="2026-06-30", n_days=618, n_months=29, fp="8463d167f5aa",
    # the measurement floor
    sd_vs_bench=134.7, sd_vs_peer=8.9, floor_ratio=15.2,
    det_vs_bench=481.0, det_vs_peer=66.0,
    # the headline pair
    cheap="EZBC", dear="GBTC", spread=145.8, spread_t=8.43, pos_months=26,
    ci_lo=110.5, ci_hi=187.0, frac_neg=0.0, sd_month=12.4, sd_daily=10.2,
    # every cheap wrapper against GBTC
    vs_gbtc={"IBIT": (133.6, 7.04, 24), "FBTC": (136.3, 8.23, 25), "ARKB": (139.9, 7.31, 25),
             "BITB": (140.1, 9.77, 27), "HODL": (161.7, 10.56, 28), "BRRR": (140.7, 9.36, 26),
             "BTCO": (147.2, 5.80, 24), "EZBC": (145.8, 8.43, 26), "BTCW": (139.8, 6.76, 24)},
    # tracking table: fee, endpoint, slope, t_slope, monthly-vs-BTC, t, vs-cohort, t
    track={
        "IBIT": (25, -41.3, -25.0, -5.16, -15.7, -0.13, 5.1, 0.38),
        "FBTC": (25, -45.6, -20.5, -4.28, -13.1, -0.12, 7.8, 1.04),
        "ARKB": (21, -46.5, -20.8, -4.37, -9.5, -0.08, 11.4, 1.10),
        "BITB": (20, -49.9, -18.0, -3.70, -9.3, -0.09, 11.6, 1.88),
        "HODL": (20, -23.9, 0.2, 0.05, 12.3, 0.11, 33.2, 4.10),
        "BRRR": (25, -56.9, -21.1, -4.40, -8.7, -0.08, 12.2, 1.63),
        "BTCO": (25, -25.1, -19.4, -4.12, -2.2, -0.02, 18.7, 1.29),
        "EZBC": (19, -29.9, -15.5, -3.33, -3.6, -0.03, 17.3, 1.66),
        "BTCW": (25, -19.1, -17.2, -3.62, -9.6, -0.09, 11.3, 0.74),
        "GBTC": (150, -79.4, -156.8, -33.22, -149.4, -1.39, -128.5, -9.57),
    },
    # anchor robustness of the headline (it telescopes to an endpoint estimate, so its two
    # anchors are tested directly): trims off both ends, and every session as an anchor
    trim={1: (141.7, 7.78, 27), 2: (140.5, 7.18, 25), 3: (135.6, 6.06, 23)},
    slope_all=141.3, slope_acf1=0.80, slope_nine_med=140.1,
    slope_nine_lo=131.8, slope_nine_hi=157.1,
    # era cut, split 2025-01-01
    era_e_n=11, era_e=186.5, era_e_t=5.76, era_e_pos=9,
    era_l_n=18, era_l=120.9, era_l_t=9.10, era_l_pos=17,
    # the fee-rank test: (spearman, p_perm, crit_5pct, slope, r2)
    rank={
        "headline / all ten": (-0.642, 0.0514, 0.642, -1.124, 0.978),
        "headline / cheap nine": (-0.495, 0.1792, 0.688, -1.522, 0.247),
        "blended / all ten": (-0.498, 0.1472, 0.644, -1.083, 0.985),
        "blended / cheap nine": (-0.310, 0.4076, 0.678, -1.645, 0.552),
    },
    tier_spread=6.0,
    # waiver event study: (end, pre, post, step, welch_t, expected)
    waiver={
        "IBIT": ("2025-01-10", -20.9, 21.1, 42.0, 0.73, -13),
        "FBTC": ("2024-07-31", 18.3, 5.0, -13.3, -0.30, -25),
        "ARKB": ("2024-07-11", 34.6, 6.6, -28.0, -0.22, -21),
        "BITB": ("2024-07-11", 4.0, 13.2, 9.2, 0.18, -20),
        "HODL": ("2025-03-31", 28.9, 37.2, 8.3, 0.25, -20),
        "BTCO": ("2024-07-11", -23.8, 27.5, 51.2, 0.47, -25),
        "EZBC": ("2024-08-02", 26.1, 15.0, -11.1, -0.11, -19),
        "BTCW": ("2024-07-11", 19.0, 9.6, -9.3, -0.17, -25),
    },
    # out-of-cohort control: Grayscale's own Mini Trust
    mini_months=23, mini_vs_gbtc=130.6, mini_vs_gbtc_t=2.84, mini_vs_gbtc_pos=20,
    mini_vs_cheap=-0.0, mini_vs_cheap_t=-0.00, mini_fee_gap=135,
    # the ownership race
    race_n=617, race_switches=7,
    sh_cheap=0.3467, cagr_cheap=9.67, tot_cheap=25.36,
    sh_dear=0.3365, cagr_dear=9.13, tot_dear=23.84,
    sh_rot=0.3457, cagr_rot=9.64, tot_rot=25.28,
    sharpe_gap=0.0102, vol_ann=49.9,
    # acting on it
    be_2bp=10.0, be_5bp=25.0, be_25bp=125.2,
    tax_20_100=7.2, tax_20_300=11.1, tax_238_300=13.5,
    ls_gross=137.8, ls_dead_at=150.0,
    # synthetic control
    syn_pl_pair=124.4, syn_pl_planted=131.0, syn_pl_t=6.55, syn_pl_slope=-0.970, syn_pl_r2=0.995,
    syn_nl_pair=-7.9, syn_nl_t=-0.44, syn_nl_slope=0.039,
    syn_null_seeds=120, syn_null_rho=0.016, syn_null_fire_pct=4.2, syn_null_max_t=0.92,
    syn_pow_seeds=24, syn_pow_t=7.00, syn_pow_slope=-1.010, syn_pow_slope_sd=0.045,
    syn_pow_rank_fire=9,
)


HEADER = f"""# Study 959 — Crypto Fee War ₿

**Ten wrappers, one coin, fees from 19 bp to 150 bp. Does the tape hand the fee back?**

On **2024-01-11** the SEC let ten US spot-bitcoin ETFs start trading on the same morning.
They hold the same asset, strike at the same 16:00 New York close, and were written from
near-identical prospectuses. What they do *not* share is the price: Franklin charges
**19 bp** a year, BlackRock and Fidelity **25**, and Grayscale's converted trust charges
**150** — an eightfold spread, the widest any ETF category has ever launched with. Several
launches also waived their fee outright for the first six to twelve months.

That is a natural experiment. The sponsor fee is *contractually* accrued out of NAV every
day, so unlike almost everything on this desk it cannot fail to exist — the only question is
how much of it a public daily tape can actually see.

We measure it on the full cohort against **BTC-USD** and against each other,
{R['start']} → {R['end']} ({R['n_days']} sessions, {R['n_months']} complete months).

*Numbers below are the frozen headline run (`docs/results.md`, fingerprint `{R['fp']}`,
as-of 2026-06-30). The only live cells run the offline synthetic control, and they say so.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),

        md("## 1. The same coin, at two prices\n\n"
           "Imagine two shops selling the identical bar of gold, in the identical vault, "
           "with the identical opening hours. One charges you 0.20% a year to keep it there; "
           "the other charges 1.50%. Nothing else differs. Over a decade the second shop "
           "quietly takes about an eighth of your gold.\n\n"
           "That is precisely the US spot-bitcoin ETF market since January 2024. So the "
           "question is embarrassingly simple: **do the expensive shop's customers actually "
           "end up with less?**"),
        code(
            "R = dict(track=%r)\n"
            "print('fund    fee bp/yr   what it delivered vs the pack (bp/yr)')\n"
            "for tk, v in R['track'].items():\n"
            "    print('%%-6s %%8d %%18.1f' %% (tk, v[0], v[6]))"
            % (R["track"],)
        ),
        md(f"Read the right-hand column against **zero**, not against each other. Nine funds "
           f"sit in a band from +5 to +33 basis points a year — that band *is* the noise. And "
           f"then there is GBTC at **{R['track']['GBTC'][6]:.1f} bp/yr**, which is not noise: "
           f"it is almost exactly the {R['track']['GBTC'][0] - 20:.0f} bp of extra fee it "
           f"charges.\n\n"
           f"> 🔬 **For the quants.** The column is each fund's mean non-overlapping monthly "
           f"log return minus the equal-weight cohort mean, annualised. GBTC's *t* is "
           f"**{R['track']['GBTC'][7]:+.2f}**; the best of the other nine is "
           f"**{R['track']['HODL'][7]:+.2f}** and the rest are under +2."),

        md("## 2. Why you cannot just compare a fund to bitcoin\n\n"
           "The obvious test is to line each fund up against the price of bitcoin and see who "
           "falls behind. It does not work, and the reason is a clock.\n\n"
           "Bitcoin trades every minute of every day. The ETFs are priced once, at 16:00 in "
           "New York. So the gap between a fund and 'bitcoin' on any given day is mostly the "
           "**move bitcoin made while the fund was shut** — overnight, over the weekend. That "
           "is enormous, and it has nothing to do with fees."),
        code(
            "R = dict(sd_vs_bench=%r, sd_vs_peer=%r, floor_ratio=%r, det_vs_bench=%r, det_vs_peer=%r,\n"
            "         n_months=%r)\n"
            "print('one day of fund-minus-bitcoin  wobbles by %%6.1f bp' %% R['sd_vs_bench'])\n"
            "print('one day of fund-minus-fund     wobbles by %%6.1f bp   (%%.0fx smaller)'\n"
            "      %% (R['sd_vs_peer'], R['floor_ratio']))\n"
            "print()\n"
            "print('so in %%d months, the smallest yearly fee gap you could prove is:' %% R['n_months'])\n"
            "print('  against bitcoin itself : %%5.0f bp/yr' %% R['det_vs_bench'])\n"
            "print('  against another fund   : %%5.0f bp/yr' %% R['det_vs_peer'])\n"
            "print()\n"
            "print('a 20 bp/yr fee is 0.08 bp per day. good luck.')"
            % (R["sd_vs_bench"], R["sd_vs_peer"], R["floor_ratio"],
               R["det_vs_bench"], R["det_vs_peer"], R["n_months"])
        ),
        md(f"Comparing the funds **to each other** deletes the clock problem entirely — they "
           f"are all shut at the same moment, so the overnight move is common to all of them "
           f"and cancels. The noise falls by **{R['floor_ratio']:.0f}×**. Everything that "
           f"follows is a fund-versus-fund comparison, for exactly this reason.\n\n"
           f"> 🔬 **For the quants.** The stub is mean-zero, so a fund-versus-spot estimate is "
           f"*unbiased* — it is merely useless. GBTC's true leak shows up in that column at "
           f"**{R['track']['GBTC'][4]:.1f} bp/yr** with a *t* of only "
           f"**{R['track']['GBTC'][5]:+.2f}**: the right answer, unprovable."),

        md("## 3. The answer: yes, where the fee gap is big\n\n"
           "Line the cheapest wrapper up against the most expensive one and the fee is simply "
           "*there* — not to the basis point (the honest range is about 135 to 145 a year, "
           "depending on exactly which days you anchor on), but unmistakably there."),
        code(
            "R = dict(cheap=%r, dear=%r, spread=%r, spread_t=%r, pos_months=%r, n_months=%r,\n"
            "         ci_lo=%r, ci_hi=%r, vs_gbtc=%r)\n"
            "print('%%s minus %%s: %%+.1f bp per year   (t = %%+.2f)'\n"
            "      %% (R['cheap'], R['dear'], R['spread'], R['spread_t']))\n"
            "print('positive in %%d of %%d months, 95%%%% confidence [%%+.1f, %%+.1f]'\n"
            "      %% (R['pos_months'], R['n_months'], R['ci_lo'], R['ci_hi']))\n"
            "print()\n"
            "print('and it is not one lucky pairing - every cheap fund vs GBTC:')\n"
            "for tk, (s, t, p) in R['vs_gbtc'].items():\n"
            "    print('  %%-6s %%+7.1f bp/yr   t=%%+6.2f   %%2d/%%d months ahead' %% (tk, s, t, p, R['n_months']))"
            % (R["cheap"], R["dear"], R["spread"], R["spread_t"], R["pos_months"], R["n_months"],
               R["ci_lo"], R["ci_hi"], R["vs_gbtc"])
        ),
        md("Nine separate funds, nine separate confirmations, all pointing the same way and "
           "all landing on the same number — around **135 basis points a year**, which is "
           "what you get when you subtract 20 from 150.\n\n"
           "> 🔬 **For the quants.** These are not nine independent draws — they share the "
           "GBTC leg — but they *are* nine independent cheap legs, and the dispersion across "
           "them (134 to 162 bp/yr) is a fair read of how much wrapper-specific noise sits on "
           "top of the fee."),

        md("## 4. The cleanest experiment on the desk\n\n"
           "In July 2024 Grayscale did something that could not have been better designed if "
           "we had asked. It launched a **second** bitcoin ETF — the Mini Trust — spun out of "
           "the first one's own coins. Same sponsor. Same custodian. Same coin. Same 16:00 "
           "strike. Same everything, except the fee: **15 bp instead of 150**."),
        code(
            "R = dict(mini_months=%r, mini_vs_gbtc=%r, mini_vs_gbtc_t=%r, mini_vs_gbtc_pos=%r,\n"
            "         mini_vs_cheap=%r, mini_fee_gap=%r)\n"
            "print('Grayscale cheap twin vs Grayscale flagship, over %%d months:' %% R['mini_months'])\n"
            "print('  measured gap  %%+7.1f bp/yr   (t = %%+.2f, ahead in %%d/%%d months)'\n"
            "      %% (R['mini_vs_gbtc'], R['mini_vs_gbtc_t'], R['mini_vs_gbtc_pos'], R['mini_months']))\n"
            "print('  fee gap       %%+7.1f bp/yr' %% R['mini_fee_gap'])\n"
            "print()\n"
            "print('and the same cheap twin against an outside cheap fund: %%+.1f bp/yr - nothing.'\n"
            "      %% R['mini_vs_cheap'])"
            % (R["mini_months"], R["mini_vs_gbtc"], R["mini_vs_gbtc_t"], R["mini_vs_gbtc_pos"],
               R["mini_vs_cheap"], R["mini_fee_gap"])
        ),
        md("The gap is the fee. Not Grayscale's competence, not its custodian, not its "
           "trading desk — the fee, handed back to within a few basis points, by the sponsor's "
           "own two products."),

        md("## 5. What the tape *cannot* see — and it is most of the story\n\n"
           "Two of the three things we set out to test simply do not survive contact with the "
           "data, and the reason is the same in both cases: the effect is smaller than the "
           "measurement."),
        code(
            "R = dict(tier_spread=%r, det_vs_peer=%r, rank=%r, waiver=%r)\n"
            "print('inside the cheap tier, the WHOLE fee spread is %%.0f bp/yr' %% R['tier_spread'])\n"
            "print('the smallest gap this tape can prove is         %%.0f bp/yr' %% R['det_vs_peer'])\n"
            "print()\n"
            "print('so ranking funds by realised tracking:')\n"
            "for tag, (rho, p, crit, slope, r2) in R['rank'].items():\n"
            "    verdict = 'significant' if p < 0.05 else 'NOT significant'\n"
            "    print('  %%-24s rank corr %%+.3f, p=%%.3f  -> %%s' %% (tag, rho, p, verdict))\n"
            "print()\n"
            "print('and the fee waivers expiring - a step that SHOULD be negative:')\n"
            "for tk, (end, pre, post, step, t, exp) in R['waiver'].items():\n"
            "    print('  %%-6s expected %%+4d, measured %%+6.1f  (t=%%+5.2f)  %%s'\n"
            "          %% (tk, exp, step, t, 'wrong sign' if step > 0 else ''))"
            % (R["tier_spread"], R["det_vs_peer"], R["rank"], R["waiver"])
        ),
        md(f"**Ranking IBIT against FBTC on realised tracking is reading tea leaves.** Their "
           f"fees differ by nothing at all; the widest gap inside the cheap tier is "
           f"{R['tier_spread']:.0f} bp/yr and the tape resolves {R['det_vs_peer']:.0f}. And a "
           f"waiver expiring is a **1.7 bp step in a monthly series that wobbles by 12 bp** — "
           f"four of the eight measured steps come out with the wrong sign, which is exactly "
           f"what pure noise looks like.\n\n"
           f"That is not evidence the waivers were fictional. It is a demonstration that a "
           f"free daily price series cannot see them.\n\n"
           f"> 🔬 **For the quants.** The all-ten rank correlation is "
           f"{R['rank']['headline / all ten'][0]:+.3f} with a permutation *p* of "
           f"{R['rank']['headline / all ten'][1]:.4f} — sitting exactly on the critical value "
           f"the null can attain ({R['rank']['headline / all ten'][2]:.3f}), because five funds "
           f"are tied at 25 bp. Meanwhile the *regression* of tracking difference on fee has "
           f"slope {R['rank']['headline / all ten'][3]:+.3f} and R² "
           f"{R['rank']['headline / all ten'][4]:.3f}. The level is nailed; the ranking is not."),

        md("## 6. So what is 135 basis points a year actually worth?\n\n"
           "This is where the study turns awkward. The fee effect is one of the most "
           "statistically certain things on this desk — and it is nearly worthless as a "
           "*trade*."),
        code(
            "R = dict(sh_cheap=%r, sh_dear=%r, sh_rot=%r, tot_cheap=%r, tot_dear=%r, tot_rot=%r,\n"
            "         sharpe_gap=%r, vol_ann=%r, race_switches=%r, be_2bp=%r, be_25bp=%r,\n"
            "         tax_20_100=%r, tax_238_300=%r, ls_gross=%r, ls_dead_at=%r)\n"
            "print('owning the cheapest wrapper : total %%+.2f%%%%   Sharpe %%+.4f' %% (R['tot_cheap'], R['sh_cheap']))\n"
            "print('owning the priciest wrapper : total %%+.2f%%%%   Sharpe %%+.4f' %% (R['tot_dear'], R['sh_dear']))\n"
            "print('chasing last quarter\\'s best: total %%+.2f%%%%   Sharpe %%+.4f  (%%d switches)'\n"
            "      %% (R['tot_rot'], R['sh_rot'], R['race_switches']))\n"
            "print()\n"
            "print('the fee advantage in Sharpe terms: %%+.4f  - against %%.0f%%%% annual volatility'\n"
            "      %% (R['sharpe_gap'], R['vol_ann']))\n"
            "print()\n"
            "print('switching at 2 bp costs        : repaid in %%.0f days' %% R['be_2bp'])\n"
            "print('switching at 25 bp costs       : repaid in %%.0f days' %% R['be_25bp'])\n"
            "print('switching with a taxed +100%%%% gain: repaid in %%.1f YEARS' %% R['tax_20_100'])\n"
            "print('switching with a taxed +300%%%% gain: repaid in %%.1f YEARS' %% R['tax_238_300'])"
            % (R["sh_cheap"], R["sh_dear"], R["sh_rot"], R["tot_cheap"], R["tot_dear"], R["tot_rot"],
               R["sharpe_gap"], R["vol_ann"], R["race_switches"], R["be_2bp"], R["be_25bp"],
               R["tax_20_100"], R["tax_238_300"], R["ls_gross"], R["ls_dead_at"])
        ),
        md(f"Bitcoin moves about **{R['vol_ann']:.0f}% a year**. Against that, 135 basis points "
           f"is a **{R['sharpe_gap']:+.4f}** change in Sharpe ratio — you would never see it in "
           f"a performance chart. Chasing last quarter's best tracker actually ends up *behind* "
           f"simply buying the cheap fund and never touching it again.\n\n"
           f"And this is why Grayscale can still charge 150 bp in a market where the going "
           f"rate is 20: anyone who bought GBTC before 2024 is sitting on an enormous embedded "
           f"gain, and selling to save 135 bp/yr means paying capital gains **today**. At a "
           f"+300% gain that takes **{R['tax_238_300']:.0f} years** to earn back. The dear fund "
           f"is not surviving on merit; it is surviving on a tax lock-in."),

        md("## 7. Live check — the machinery is honest (offline synthetic)\n\n"
           "Nothing below touches the real tape. We build a fake world where wrappers really "
           "are shaved by a known fee ladder, and a second fake world where they all charge "
           "the same thing while the published fee sheet still *looks* dispersed. The tools "
           "must find the first and stay silent on the second."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from fee_war import data, strategy as st\n"
            "for ss, tag in [(1.0, 'planted fee ladder'), (0.0, 'null: everyone charges the same')]:\n"
            "    px, truth = data.synthetic_panel(signal_strength=ss, seed=959)\n"
            "    d = st.synthetic_detect(px, truth, n_perm=2000)\n"
            "    print('%-34s measured %+7.1f bp/yr (planted %3.0f)  t=%+6.2f'\n"
            "          % (tag, d['pair_spread_bpy'], d['planted_gap_bpy'], d['pair_t']))"
        ),
        md("The planted fee is recovered; the null world stays flat. So the real-tape result "
           "is a fact about the funds, not an artefact of the tools."),

        md(f"## Verdict\n\n"
           f"- **Signal — Real.** The fee is delivered: **{R['spread']:+.1f} bp/yr** between "
           f"cheapest and priciest with *t* = **{R['spread_t']:+.2f}**, ahead in "
           f"{R['pos_months']} of {R['n_months']} months, confirmed on nine separate wrappers "
           f"and by Grayscale's own cheap twin (**{R['mini_vs_gbtc']:+.1f} bp/yr** against a "
           f"{R['mini_fee_gap']} bp fee gap). What is *not* real is the fine print: inside the "
           f"19–25 bp tier the ranking is unreadable, and the waiver expiries are invisible.\n"
           f"- **Tradability — Fragile.** It is a purchase decision, not a strategy. Buying a "
           f"20 bp wrapper today instead of a 150 bp one is free, permanent and worth ~135 "
           f"bp/yr. Everything else fails: the advantage is **{R['sharpe_gap']:+.4f}** of "
           f"Sharpe against 50% volatility, the rotation rule loses to doing nothing, and a "
           f"taxed switch takes {R['tax_20_100']:.0f}–{R['tax_238_300']:.0f} years to repay."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md(f"# Study 959 — Crypto Fee War — the teardown\n\n"
           f"Three tracking-difference estimators and where the naive one lies; the "
           f"fund-versus-spot measurement floor that makes the cross-sectional instrument "
           f"necessary; the HAC *t* and moving-block bootstrap on the cheapest-minus-priciest "
           f"spread; a Spearman rank test reported against its **attainable** critical value; "
           f"the waiver event study; the excess-of-cash ownership race with one execution lag; "
           f"the borrow and tax sweeps; and the live synthetic control.\n\n"
           f"Cohort: the ten US spot-bitcoin ETFs launched 2024-01-11, vs **BTC-USD**. Window "
           f"{R['start']} → {R['end']}, {R['n_days']} sessions, {R['n_months']} complete "
           f"months. Every real number is frozen from `docs/results.md` (fingerprint "
           f"`{R['fp']}`, as-of 2026-06-30)."),
        code("R = %r" % (R,)),

        md("## 1. The measurement floor decides the estimator\n\n"
           "BTC-USD is a 24/7 quote; the wrappers strike at 16:00 New York. The fund-minus-"
           "benchmark difference therefore carries a mean-zero **clock stub**; the "
           "fund-minus-fund difference does not, because the stub is common."),
        code(
            "print(f\"daily sd, fund - BTC-USD : {R['sd_vs_bench']:7.1f} bp\")\n"
            "print(f\"daily sd, fund - peer    : {R['sd_vs_peer']:7.1f} bp   (ratio {R['floor_ratio']:.1f}x)\")\n"
            "print()\n"
            "print(f\"smallest |t|=2-detectable gap on {R['n_months']} months:\")\n"
            "print(f\"  vs BTC-USD : {R['det_vs_bench']:5.0f} bp/yr\")\n"
            "print(f\"  vs a peer  : {R['det_vs_peer']:5.0f} bp/yr\")\n"
            "print(f\"a 20 bp/yr fee = {20/252:.3f} bp/day.\")"
        ),
        md("> 💡 **In plain words.** Bitcoin keeps moving after the funds close, so measuring a "
           "fund against bitcoin is like weighing a letter on a bathroom scale during an "
           "earthquake. Weigh two letters against each other and the earthquake cancels."),

        md("## 2. Three estimators of one quantity\n\n"
           "`endpoint` (two anchors, annualised), `trend slope` (OLS of the log ratio on time, "
           "HAC se, 618 anchors), `monthly` (mean non-overlapping complete-calendar-month "
           "difference, HAC *t*), plus the **cohort-relative** monthly figure in which the "
           "benchmark's stub cancels entirely."),
        code(
            "hdr = ('fund', 'fee', 'endpt', 'slope', 't', 'vsBTC', 't', 'vsPack', 't')\n"
            "print('%-6s %5s %8s %8s %7s %9s %7s %9s %7s' % hdr)\n"
            "for tk, v in R['track'].items():\n"
            "    print('%-6s %5d %8.1f %8.1f %7.2f %9.1f %7.2f %9.1f %7.2f'\n"
            "          % (tk, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]))"
        ),
        md(f"Three things to read off this table.\n\n"
           f"1. **The endpoint estimator lies by 70 bp/yr.** It puts GBTC's leak at "
           f"{R['track']['GBTC'][1]:.1f} bp/yr — roughly half the truth — because 2024-01-11 was "
           f"GBTC's conversion day and its closing discount that afternoon is baked into the "
           f"first anchor. Two anchors, one dislocated, wrong answer.\n"
           f"2. **The vs-BTC column is unusable.** GBTC's {R['track']['GBTC'][4]:.1f} bp/yr — the "
           f"one unambiguously correct number in it — carries *t* = "
           f"{R['track']['GBTC'][5]:+.2f}. The floor is {R['det_vs_bench']:.0f} bp/yr.\n"
           f"3. **The trend-slope column looks sharp and is over-confident.** Its residual is a "
           f"persistent AR(1) premium/discount, so the HAC correction is not enough; it is "
           f"quoted, never leaned on. Every headline is the monthly estimator."),
        md(f"> 💡 **In plain words.** Measure a slow leak by comparing the level on two "
           f"particular days and you are at the mercy of what happened on those two days. "
           f"Measuring month by month does **not** make that go away — log increments "
           f"telescope, so the monthly mean is still an endpoint estimate, just between two "
           f"*better* days (the contaminated conversion close is gone). What it genuinely "
           f"adds is a dispersion: 29 separate signs, and a *t*. The anchors themselves are "
           f"checked in §3b.\n\n"
           f"**HODL is the exception worth naming:** {R['track']['HODL'][6]:+.1f} bp/yr against "
           f"the pack at *t* = {R['track']['HODL'][7]:+.2f} — beating the cohort by more than "
           f"its own 20 bp fee. A wrapper cannot out-earn the coin it holds; the residual is a "
           f"quoting artefact plus the cohort's longest waiver (scheduled to 2025-03-31). Named, "
           f"not promoted to a finding."),

        md("## 3. The headline — a pure-tape pair spread\n\n"
           "Fund-minus-fund monthly tracking difference, annualised. Uses **no fee input**. "
           "HAC *t* at 3 monthly lags; moving-block bootstrap (5,000 draws, 3-month blocks)."),
        code(
            "print(f\"{R['cheap']} - {R['dear']}: {R['spread']:+.1f} bp/yr   HAC t = {R['spread_t']:+.2f}\")\n"
            "print(f\"  positive in {R['pos_months']}/{R['n_months']} months\")\n"
            "print(f\"  block-bootstrap 95% CI [{R['ci_lo']:+.1f}, {R['ci_hi']:+.1f}] bp/yr, \"\n"
            "      f\"share of resamples < 0: {R['frac_neg']:.4f}\")\n"
            "print(f\"  monthly sd {R['sd_month']:.1f} bp | daily sd {R['sd_daily']:.1f} bp\")\n"
            "print()\n"
            "print('every cheap wrapper against GBTC (nine independent cheap legs):')\n"
            "for tk, (s, t, p) in R['vs_gbtc'].items():\n"
            "    print(f'  {tk:6s} {s:+8.1f} bp/yr   t={t:+6.2f}   {p:2d}/{R[\"n_months\"]}')\n"
            "vals = [v[0] for v in R['vs_gbtc'].values()]\n"
            "print(f'\\n  dispersion across the nine: {min(vals):.1f} to {max(vals):.1f} bp/yr'\n"
            "      f'  (fee gap 150-20 = 130)')"
        ),

        md("## 3b. Is the headline an artefact of its own two anchors?\n\n"
           "The monthly mean telescopes, so it *is* an endpoint estimate between the first "
           "and last complete month-end. So test those anchors: trim months off **both** "
           "ends, and re-estimate with every session as an anchor (the OLS trend slope)."),
        code(
            "print('%-38s %10s %8s' % ('estimator', 'bp/yr', 'HAC t'))\n"
            "print('%-38s %+10.1f %+8.2f' % ('headline (29 months)', R['spread'], R['spread_t']))\n"
            "for k, (v, t, n) in R['trim'].items():\n"
            "    print('%-38s %+10.1f %+8.2f' % (f'trim {k} month(s) off both ends (n={n})', v, t))\n"
            "print('%-38s %+10.1f %8s' % ('all-618-anchor OLS trend slope', R['slope_all'], 'n/a'))\n"
            "print('%-38s %+10.1f %8s' % ('same slope, median of the nine legs', R['slope_nine_med'], 'n/a'))\n"
            "print(f\"\\nevery anchor-free version returns 135-142 bp/yr; the headline is ~5 bp rich.\")"
        ),
        md(f"**The finding survives its anchors, and the headline is slightly rich.** "
           f"Trimming three months off each end still gives {R['trim'][3][0]:.1f} bp/yr at "
           f"*t* = {R['trim'][3][1]:+.2f}; using all 618 sessions as anchors gives "
           f"{R['slope_all']:.1f} bp/yr, and the median of the same slope across the nine "
           f"cheap legs is {R['slope_nine_med']:.1f}. The extra ~5 bp in the headline is the "
           f"tail of GBTC's discount closing in early 2024 — the same thing the era cut shows "
           f"in §4. **Carry away ~140 bp/yr, not 145.8.** (The trend slope's own HAC *t* is "
           f"not quoted: its residual is a persistent AR(1) premium, acf1 = "
           f"{R['slope_acf1']:.2f}, so that *t* is over-confident. It is used here only as a "
           f"point estimate that no single day can move.)"),

        md("## 4. Era cut (split 2025-01-01)"),
        code(
            "print(f\"2024      (n={R['era_e_n']:2d} months): {R['era_e']:+7.1f} bp/yr  \"\n"
            "      f\"t={R['era_e_t']:+5.2f}  {R['era_e_pos']}/{R['era_e_n']} positive\")\n"
            "print(f\"2025-2026 (n={R['era_l_n']:2d} months): {R['era_l']:+7.1f} bp/yr  \"\n"
            "      f\"t={R['era_l_t']:+5.2f}  {R['era_l_pos']}/{R['era_l_n']} positive\")"
        ),
        md(f"Positive and significant in **both** halves. 2024 runs hot "
           f"({R['era_e']:.0f} bp/yr) — GBTC's discount was still closing and its outflow was "
           f"heaviest — and the later era, at {R['era_l']:.0f} bp/yr, is the cleaner "
           f"steady-state read. Note the *t* is **higher** in the calmer era: as the discount "
           f"noise faded the contractual fee showed through more clearly, not less."),

        md("## 5. Does the fee *ranking* predict the outcome ranking?\n\n"
           "Spearman(fee, cohort-relative TD) against a permutation null (exact for ≤ 8 funds, "
           "20,000 sampled above), reported alongside the **attainable** 5% critical value — a "
           "fee vector with five funds tied at 25 bp cannot generate a fine rank statistic. "
           "Beside it, the cross-sectional pass-through regression TD = a + b·fee, where a "
           "wrapper that leaks its fee and nothing else gives b = −1."),
        code(
            "print('%-24s %8s %8s %10s %8s %7s' % ('sheet / universe', 'rho', 'p_perm', 'needs|rho|', 'slope b', 'R2'))\n"
            "for tag, (rho, p, crit, slope, r2) in R['rank'].items():\n"
            "    print('%-24s %+8.3f %8.4f %10.3f %+8.3f %7.3f' % (tag, rho, p, crit, slope, r2))\n"
            "print()\n"
            "print(f\"inside the cheap tier the whole fee spread is {R['tier_spread']:.0f} bp/yr;\"\n"
            "      f\" the floor is {R['det_vs_peer']:.0f} bp/yr.\")"
        ),
        md(f"Two facts, both true, neither allowed to swallow the other.\n\n"
           f"- **The pass-through is essentially exact**: slope "
           f"{R['rank']['headline / all ten'][3]:+.3f} (waiver-blended "
           f"{R['rank']['blended / all ten'][3]:+.3f}) with R² "
           f"{R['rank']['headline / all ten'][4]:.3f}. The published fee sheet explains 98% of "
           f"the cross-section.\n"
           f"- **The rank test does not clear — it has no room to**: *p* = "
           f"{R['rank']['headline / all ten'][1]:.4f} against an attainable critical value of "
           f"{R['rank']['headline / all ten'][2]:.3f}. Drop GBTC and nothing is significant on "
           f"any fee sheet (*p* = {R['rank']['headline / cheap nine'][1]:.2f} to "
           f"{R['rank']['blended / cheap nine'][1]:.2f}), and R² falls to "
           f"{R['rank']['headline / cheap nine'][4]:.3f}.\n\n"
           f"This is a power statement, not a contradiction: the cross-section contains exactly "
           f"**one** resolvable fee fact, and the regression finds it while the rank statistic "
           f"asks a finer question than {R['n_months']} months can answer."),
        md("> 💡 **In plain words.** The fee sheet is right about who is expensive. It is not, "
           "and cannot be, right about who is 22 bp versus 25 bp — that difference is thinner "
           "than the measuring instrument."),

        md("## 6. The waiver event study\n\n"
           "Step in each fund's cohort-relative monthly TD at its **scheduled** waiver end "
           "(itself an ASSUMPTION — several were AUM-capped and ended early). A waiver that "
           "expires should print a negative step of roughly the headline fee."),
        code(
            "print('%-6s %-12s %8s %8s %8s %8s %9s' % ('fund','waiver end','pre','post','step','welch t','expected'))\n"
            "wrong = 0\n"
            "for tk, (end, pre, post, step, t, exp) in R['waiver'].items():\n"
            "    wrong += int(step > 0)\n"
            "    print('%-6s %-12s %+8.1f %+8.1f %+8.1f %+8.2f %+9d' % (tk, end, pre, post, step, t, exp))\n"
            "print(f'\\nsteps with the WRONG sign: {wrong}/{len(R[\"waiver\"])}; '\n"
            "      f'max |welch t| = {max(abs(v[4]) for v in R[\"waiver\"].values()):.2f}')"
        ),
        md("**Nothing.** No step clears |*t*| = 0.8 and half have the wrong sign. A 20 bp/yr "
           "waiver expiry is a **1.7 bp step in a monthly series with a 12 bp sd**, observed "
           "with six to twelve months on each side — the test is under-powered by roughly an "
           "order of magnitude before it starts. BRRR and GBTC cannot be tested at all (too few "
           "months on one side). This is a null result about the *instrument*, not about the "
           "waivers."),

        md(f"## 7. Out-of-cohort control — the same sponsor, two fees\n\n"
           f"Grayscale's Bitcoin Mini Trust (ticker BTC, 15 bp) launched 2024-07-31, spun out "
           f"of GBTC's own coins. Same sponsor, custodian, coin and strike; "
           f"{R['mini_fee_gap']} bp of fee difference and nothing else."),
        code(
            "print(f\"BTC - GBTC ({R['mini_months']} months): {R['mini_vs_gbtc']:+7.1f} bp/yr  \"\n"
            "      f\"t={R['mini_vs_gbtc_t']:+5.2f}  {R['mini_vs_gbtc_pos']}/{R['mini_months']} positive\")\n"
            "print(f\"BTC - {R['cheap']} ({R['mini_months']} months): {R['mini_vs_cheap']:+7.1f} bp/yr  \"\n"
            "      f\"t={R['mini_vs_cheap_t']:+5.2f}\")\n"
            "print(f\"\\nfee difference: {R['mini_fee_gap']} bp/yr. measured: {R['mini_vs_gbtc']:.1f}.\")"
        ),
        md("Against its own flagship the cheap twin gains the fee. Against an outside cheap "
           "wrapper it is indistinguishable from zero. Sponsor identity is held fixed and the "
           "effect survives — it is the fee, not the firm."),

        md("## 8. The ownership race, excess-of-cash (2 bp one-way, one execution lag)\n\n"
           "Three ways to choose a wrapper. `own_cheapest` and `own_priciest` never trade. "
           "`rotate_winner` ranks on trailing three-month cohort-relative TD at each quarter "
           "end and switches at the **next** session's close. No short leg, so no borrow; the "
           "cash leg is BIL's actual total return."),
        code(
            "print('%-16s %14s %8s %8s %10s' % ('arm','excess Sharpe','CAGR','vol','total'))\n"
            "print('%-16s %+14.4f %+8.2f%% %8.1f%% %+10.2f%%'\n"
            "      % ('own_cheapest', R['sh_cheap'], R['cagr_cheap'], R['vol_ann'], R['tot_cheap']))\n"
            "print('%-16s %+14.4f %+8.2f%% %8.1f%% %+10.2f%%'\n"
            "      % ('own_priciest', R['sh_dear'], R['cagr_dear'], R['vol_ann'], R['tot_dear']))\n"
            "print('%-16s %+14.4f %+8.2f%% %8.1f%% %+10.2f%%'\n"
            "      % ('rotate_winner', R['sh_rot'], R['cagr_rot'], R['vol_ann'], R['tot_rot']))\n"
            "print(f\"\\nSharpe gap cheapest - priciest: {R['sharpe_gap']:+.4f}\")\n"
            "print(f\"rotation: {R['race_switches']} switches, and it ends BEHIND doing nothing \"\n"
            "      f\"({R['tot_rot']:+.2f}% vs {R['tot_cheap']:+.2f}%)\")"
        ),
        md(f"The most statistically certain result on this page is worth "
           f"**{R['sharpe_gap']:+.4f}** of Sharpe, because the asset carries "
           f"{R['vol_ann']:.0f}% annualised volatility and the fee is 1.2 bp a month. The "
           f"rotation rule pays {R['race_switches']} switches to arrive behind where "
           f"buy-and-never-trade already was."),

        md("## 9. Costs, tax and borrow — the three ways to act\n\n"
           "One-way × NAV throughout. The tax rate, the embedded gain and the borrow rate are "
           "**ASSUMPTIONS** (no free tape carries them), so each is swept rather than picked."),
        code(
            "print('(1) switch cost - repaid in:')\n"
            "for c, d in [(2.0, R['be_2bp']), (5.0, R['be_5bp']), (25.0, R['be_25bp'])]:\n"
            "    print(f'      {c:5.1f} bp one-way -> {d:6.1f} days')\n"
            "print('\\n(2) taxable switch - repaid in:')\n"
            "print(f'      20.0%  tax, +100% gain -> {R[\"tax_20_100\"]:5.1f} years')\n"
            "print(f'      20.0%  tax, +300% gain -> {R[\"tax_20_300\"]:5.1f} years')\n"
            "print(f'      23.8%  tax, +300% gain -> {R[\"tax_238_300\"]:5.1f} years')\n"
            "print('\\n(3) long cheap / short GBTC, net of borrow:')\n"
            "for b in (0.0, 50.0, 100.0, 150.0, 300.0):\n"
            "    net = R['ls_gross'] - b\n"
            "    print(f'      borrow {b:6.1f} bp/yr -> net {net:+8.1f} bp/yr  '\n"
            "          f'{\"alive\" if net > 0 else \"DEAD\"}')"
        ),
        md(f"The long/short survives to roughly **{R['ls_gross']:.0f} bp/yr of borrow** and dies "
           f"above it — on a permanently hard-to-borrow legacy trust in continuous outflow. The "
           f"desk does not own a borrow tape, so the honest statement is that whether this "
           f"nets anything is decided entirely by a number the study cannot observe.\n\n"
           f"> 💡 **In plain words.** You can prove the expensive fund leaks 135 bp a year. "
           f"Getting paid for knowing it requires either buying fresh (free), or shorting a "
           f"stock that is expensive to borrow (unknown), or selling something you owe tax on "
           f"(a decade)."),

        md("## 10. Live synthetic control — never supports the stamp\n\n"
           "A planted world where each wrapper really is shaved by its published fee, and a "
           "null world where every wrapper charges the cohort average while the published sheet "
           "still shows the same dispersion. The pair spread and the pass-through must recover "
           "the first and stay silent on the second."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from fee_war import data, strategy as st\n"
            "for ss, tag in [(1.0, 'planted fee ladder'), (0.0, 'null: one fee, dispersed sheet')]:\n"
            "    px, truth = data.synthetic_panel(signal_strength=ss, seed=959)\n"
            "    d = st.synthetic_detect(px, truth, n_perm=3000)\n"
            "    print('%-32s pair %+7.1f (planted %3.0f) t=%+6.2f | rho %+.3f p=%.3f | b=%+.3f R2=%.3f'\n"
            "          % (tag, d['pair_spread_bpy'], d['planted_gap_bpy'], d['pair_t'],\n"
            "             d['spearman'], d['p_perm'], d['pass_through_slope'], d['pass_through_r2']))\n"
            "ts, ps = [], []\n"
            "for s in range(12):\n"
            "    px, truth = data.synthetic_panel(signal_strength=0.0, seed=959 + s)\n"
            "    d = st.synthetic_detect(px, truth, n_perm=1500)\n"
            "    ts.append(abs(d['pair_t'])); ps.append(d['p_perm'])\n"
            "print('\\nnull x12: max |pair t| %.2f (never >= 2), rank test fires at 5%% on %d/12'\n"
            "      % (max(ts), sum(1 for p in ps if p < 0.05)))"
        ),
        code(
            "print(f\"frozen synthetic run (docs/results.md):\")\n"
            "print(f\"  planted: pair {R['syn_pl_pair']:+.1f} bp/yr (planted {R['syn_pl_planted']:.0f}), \"\n"
            "      f\"t={R['syn_pl_t']:+.2f}, pass-through b={R['syn_pl_slope']:+.3f} R2={R['syn_pl_r2']:.3f}\")\n"
            "print(f\"  null   : pair {R['syn_nl_pair']:+.1f} bp/yr, t={R['syn_nl_t']:+.2f}, \"\n"
            "      f\"b={R['syn_nl_slope']:+.3f}\")\n"
            "print(f\"  null size, {R['syn_null_seeds']} seeds: mean rho {R['syn_null_rho']:+.3f}, \"\n"
            "      f\"rank test fires on {R['syn_null_fire_pct']:.1f}% (nominal 5%), \"\n"
            "      f\"max |pair t| {R['syn_null_max_t']:.2f}\")\n"
            "print(f\"  power,     {R['syn_pow_seeds']} seeds: mean pair t {R['syn_pow_t']:+.2f}, \"\n"
            "      f\"slope {R['syn_pow_slope']:+.3f} +/- {R['syn_pow_slope_sd']:.3f}, \"\n"
            "      f\"rank test fires on {R['syn_pow_rank_fire']}/{R['syn_pow_seeds']}\")"
        ),
        md(f"The harness reproduces, on a planted world, **exactly the split the real tape "
           f"shows**: the pair spread and the pass-through regression see the fee (mean *t* "
           f"{R['syn_pow_t']:+.2f}, slope {R['syn_pow_slope']:+.3f} ± {R['syn_pow_slope_sd']:.3f}), "
           f"while the rank statistic — handed a sheet with five funds tied at one number — "
           f"fires on only {R['syn_pow_rank_fire']}/{R['syn_pow_seeds']} seeds. On the null it "
           f"is correctly sized ({R['syn_null_fire_pct']:.1f}% at nominal 5%) and the pair "
           f"instrument's |*t*| never once reaches 2 across {R['syn_null_seeds']} seeds. The "
           f"rank test's failure on the real tape is a property of tied fee sheets, not a "
           f"broken estimator."),

        md(f"## Verdict\n\n"
           f"- **Signal — Real.** The headline is a pure-tape fund-versus-fund spread using no "
           f"fee input: **{R['spread']:+.1f} bp/yr, HAC *t* = {R['spread_t']:+.2f}**, bootstrap "
           f"CI [{R['ci_lo']:+.1f}, {R['ci_hi']:+.1f}] entirely above zero, "
           f"{R['pos_months']}/{R['n_months']} months positive, significant in **both** eras "
           f"({R['era_e']:.0f} then {R['era_l']:.0f} bp/yr), replicated on **nine** independent "
           f"cheap legs (*t* = {min(v[1] for v in R['vs_gbtc'].values()):.1f} to "
           f"{max(v[1] for v in R['vs_gbtc'].values()):.1f}). Cross-sectional pass-through of "
           f"fee into tracking difference: **{R['rank']['headline / all ten'][3]:+.3f}, R² = "
           f"{R['rank']['headline / all ten'][4]:.3f}**. The sponsor-fixed control (Grayscale's "
           f"own 15 bp twin vs its 150 bp flagship) returns **{R['mini_vs_gbtc']:+.1f} bp/yr** "
           f"against a {R['mini_fee_gap']} bp fee gap.\n"
           f"- **What is not real.** The fine-grained ranking (cheap-tier *p* = "
           f"{R['rank']['headline / cheap nine'][1]:.2f}–{R['rank']['blended / cheap nine'][1]:.2f}; "
           f"{R['tier_spread']:.0f} bp of fee spread against a {R['det_vs_peer']:.0f} bp/yr "
           f"floor) and the waiver expiries (no step clears |*t*| = 0.8, four of eight the wrong "
           f"sign).\n"
           f"- **Tradability — Fragile.** {R['sharpe_gap']:+.4f} of Sharpe against "
           f"{R['vol_ann']:.0f}% vol; the rotation rule loses to doing nothing; the long/short "
           f"dies above ~{R['ls_gross']:.0f} bp/yr of unobservable borrow; a taxed switch takes "
           f"{R['tax_20_100']:.0f}–{R['tax_238_300']:.0f} years. Bankable only as a **purchase "
           f"decision**: buy the 20 bp wrapper, never trade it.\n"
           f"- **Sample.** {R['n_months']} months, one bitcoin cycle, one cohort. The fee "
           f"arithmetic will not change; everything else should be re-read in five years."),
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
