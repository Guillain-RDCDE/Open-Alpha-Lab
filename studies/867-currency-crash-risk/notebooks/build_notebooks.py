"""Generate the two narrative notebooks for Study 867 (Currency Crash Risk).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from
the frozen ``R`` dict (mirroring docs/results.md); the live cells run only the fast
synthetic positive control, so execution is quick and network-free.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance weekly FX,
# 8 currencies vs USD, 2003-12-12 -> 2026-06-26; long top-3 / short bottom-3 carry basket).
R = dict(
    start="2003-12-12", end="2026-06-26", n_ccy=8, n_weeks=1177,
    basket_skew=-1.39, skew_t=-1.51,
    premium_bps=6.33, premium_ann=3.29, premium_t=1.73, sharpe=0.34,
    worst_week=-13.30, max_dd=-36.0, calm_ann=12.6, off_ann=-173.6,
    slope=-0.312, t_slope=-2.41, r2=0.49, spearman=-0.833, spearman_p=0.008,
    hi_leg_skew=-1.08, lo_leg_skew=0.38, leg_diff=-1.46,
    placebo_obs=-1.39, placebo_mean=-0.001, placebo_sd=0.815, placebo_p=0.0335,
    era_early_skew=-1.41, era_early_skew_t=-1.21, era_early_prem=3.70, era_early_prem_t=1.19,
    era_early_slope_t=-3.39, era_early_spearman=-0.786, era_early_n=577,
    era_late_skew=-1.11, era_late_skew_t=-1.92, era_late_prem=2.90, era_late_prem_t=1.30,
    era_late_slope_t=-1.58, era_late_spearman=-0.833, era_late_n=600,
    timer_0_net=1.21, timer_0_sh=0.13, timer_50_net=0.71, timer_50_sh=0.07, timer_50_t=0.37,
    timer_100_net=0.21,
    null_skew_t_mean=0.02, null_skew_t_sd=0.73, null_fire=0, null_slope_mean=0.0019,
    planted_skew=-3.50, planted_slope=-0.696, planted_slope_t=-9.98, planted_spearman=-0.976,
)


HEADER = f"""# Study 867 — Currency Crash Risk 💥

**Do high-carry currencies go up by the stairs and down by the elevator?**

Brunnermeier, Nagel & Pedersen (2008) argue the carry trade's premium is compensation
for **crash risk**: high-interest (carry) currencies are **negatively skewed** — gentle
appreciation while you earn the rate differential, punctuated by violent unwinds. The
higher the carry, the deeper the skew, and a long-high / short-low carry basket inherits
that crash tail. We test both halves on a weekly 8-currency tape vs USD
({R['start']} → {R['end']}, {R['n_ccy']} currencies incl. the notorious high-carry MXN).

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: fixed current membership (no de-pegged legs) —
magnitudes are an upper bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "You earn the interest differential by holding a high-yield currency (MXN, NZD, "
           "AUD) funded in a low-yield one (JPY, CHF). Most weeks it drifts gently up — "
           "*the stairs*. Then a risk-off shock hits, everyone unwinds at once, and it "
           "gaps down — *the elevator*. That asymmetry is **negative skew**, and BNP say "
           "it deepens with the carry. So we (a) check whether higher-carry currencies are "
           "more negatively skewed and (b) measure the crash tail of the carry basket."),
        code(
            "import numpy as np, pandas as pd\n"
            f"R = dict(basket_skew={R['basket_skew']!r}, worst_week={R['worst_week']!r}, "
            f"max_dd={R['max_dd']!r}, calm_ann={R['calm_ann']!r}, off_ann={R['off_ann']!r}, "
            f"premium_ann={R['premium_ann']!r}, spearman={R['spearman']!r})\n"
            "print('carry basket realized skew: %+.2f  (deeply negative = the crash tail)' % R['basket_skew'])\n"
            "print('  worst single week %+.1f%%   max drawdown %+.1f%%' % (R['worst_week'], R['max_dd']))\n"
            "print('  calm weeks %+.1f%%/yr  vs  worst-5%% weeks %+.1f%%/yr (annualised)' % (R['calm_ann'], R['off_ann']))\n"
            "print('  premium you are paid for it: %+.2f%%/yr' % R['premium_ann'])\n"
            "print('  higher carry -> more negative skew: Spearman rank corr %+.2f' % R['spearman'])"
        ),
        md("## 2. Is the skew tied to the carry ordering? A live synthetic control\n\n"
           "We plant the crash in a seeded toy world (`edge>0`: a fat negative factor tail "
           "whose loading rises with carry) and check the detector recovers a negative "
           "skew-carry slope — and stays *silent* on the null (`edge=0`, symmetric factor, "
           "no crash asymmetry). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from fx_crash import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=867, n_weeks=1000))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.02, seed=867, n_weeks=1000))\n"
            "print('null world   : skew-carry slope %+.3f  basket skew %+.2f  (should be ~0)' % (null['slope'], null['basket_skew']))\n"
            "print('planted world: skew-carry slope %+.3f  basket skew %+.2f  (should light up)' % (planted['slope'], planted['basket_skew']))"
        ),
        md("## 3. The honest verdict — real crash, but a weak/borderline signal and no paycheck\n\n"
           f"On the real tape the carry basket is deeply negatively skewed "
           f"(**{R['basket_skew']:+.2f}**, worst week {R['worst_week']:+.1f}%, max DD "
           f"{R['max_dd']:+.1f}%) and higher carry clearly predicts more negative skew "
           f"(Spearman **{R['spearman']:+.2f}**, permutation *p* = {R['spearman_p']:.3f}) — "
           f"the Brunnermeier-et-al signature is genuinely there and correctly signed. **But** "
           f"the strict significance bar is not cleared: a Newey-West *t* on the basket's own "
           f"skewness is only **{R['skew_t']:+.2f}** (the skew *t* is structurally low-powered "
           f"against rare crashes). And the premium you are paid for the tail is weak "
           f"(**{R['premium_ann']:+.2f}%/yr**, *t* = {R['premium_t']:+.2f}), collapsing to "
           f"**{R['timer_50_net']:+.2f}%/yr** after a modest borrow — pennies in front of a "
           f"steamroller. **Signal: Weak. Tradability: Mirage.**"),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 867 — Currency Crash Risk — the teardown\n\n"
           "The basket crash skew (Newey-West *t* on the standardised-cubed residuals), the "
           "skew-carry cross-section (slope + Spearman), the label-shuffle placebo, the "
           "crash-conditional split, the two-era cut, the costed timer, and the 20-seed "
           "synthetic control."),
        code("R = %r" % (R,)),
        md("## The headline — the carry basket's crash skew + premium\n\n"
           "Dollar-neutral long top-3 / short bottom-3 carry basket, weekly."),
        code(
            "print(f\"realized skew : {R['basket_skew']:+.2f}   NW(6) skew t = {R['skew_t']:+.2f}\")\n"
            "print(f\"premium       : {R['premium_bps']:+.2f} bps/wk ({R['premium_ann']:+.2f}%/yr)  \"\n"
            "      f\"NW t = {R['premium_t']:+.2f}  Sharpe {R['sharpe']:.2f}\")\n"
            "print(f\"crash shape   : worst week {R['worst_week']:+.1f}%  max DD {R['max_dd']:+.1f}%\")\n"
            "print(f\"crash split   : calm {R['calm_ann']:+.1f}%/yr vs worst-5% weeks {R['off_ann']:+.1f}%/yr\")"
        ),
        md("## The skew-carry cross-section — higher carry, more negative skew?"),
        code(
            "print(f\"slope(skew on carry) = {R['slope']:+.3f}  t = {R['t_slope']:+.2f}  R2 = {R['r2']:.2f}\")\n"
            "print(f\"Spearman rank corr   = {R['spearman']:+.3f}  (permutation p = {R['spearman_p']:.3f})\")\n"
            "print(f\"high-carry leg skew {R['hi_leg_skew']:+.2f} vs low-carry leg skew {R['lo_leg_skew']:+.2f} \"\n"
            "      f\"(diff {R['leg_diff']:+.2f})\")"
        ),
        md("## Placebo — shuffle which currency owns which carry (2,000 relabelings)"),
        code(
            "print(f\"observed basket skew {R['placebo_obs']:+.2f} vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> left-tail p = {R['placebo_p']:.4f}\")"
        ),
        md("## Robustness — two eras (split 2015-01-01)"),
        code(
            "print(f\"2004-2014 (n={R['era_early_n']}): basket skew {R['era_early_skew']:+.2f} (t {R['era_early_skew_t']:+.2f}) \"\n"
            "      f\"| premium {R['era_early_prem']:+.2f}%/yr (t {R['era_early_prem_t']:+.2f}) \"\n"
            "      f\"| skew-carry slope t {R['era_early_slope_t']:+.2f} Spearman {R['era_early_spearman']:+.3f}\")\n"
            "print(f\"2015-2026 (n={R['era_late_n']}): basket skew {R['era_late_skew']:+.2f} (t {R['era_late_skew_t']:+.2f}) \"\n"
            "      f\"| premium {R['era_late_prem']:+.2f}%/yr (t {R['era_late_prem_t']:+.2f}) \"\n"
            "      f\"| skew-carry slope t {R['era_late_slope_t']:+.2f} Spearman {R['era_late_spearman']:+.3f}\")"
        ),
        md("## The timer — can you get paid for the crash risk?\n\n"
           "Costed carry book: 2 bps/side rebalance + borrow on the short leg."),
        code(
            "print(f\"borrow   0 bps/yr: net {R['timer_0_net']:+.2f}%/yr  Sharpe {R['timer_0_sh']:.2f}\")\n"
            "print(f\"borrow  50 bps/yr: net {R['timer_50_net']:+.2f}%/yr  Sharpe {R['timer_50_sh']:.2f}  (t {R['timer_50_t']:+.2f})\")\n"
            "print(f\"borrow 100 bps/yr: net {R['timer_100_net']:+.2f}%/yr\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the powered skew-carry detector must NOT fire on the null and must recover "
           "a planted carry-crash relation. (The basket *skew t* is deliberately low-powered "
           "against rare crashes — even a planted skew of −3.5 returns skew t ≈ −1.)"),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from fx_crash import data, strategy as st\n"
            "null_slope = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=867+s, n_weeks=1000))['slope'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: skew-carry slope mean {null_slope.mean():+.4f} (sd {null_slope.std(ddof=1):.4f})\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.02, seed=867, n_weeks=1000))\n"
            "print(f\"planted (edge=0.02): basket skew {planted['basket_skew']:+.2f}, skew-carry slope {planted['slope']:+.3f} (Spearman {planted['spearman']:+.3f})\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** The BNP crash-skew signature is genuinely present and "
           f"correctly signed: basket skew **{R['basket_skew']:+.2f}** (label-shuffle *p* = "
           f"{R['placebo_p']:.3f}), a strongly monotone skew-carry cross-section (Spearman "
           f"**{R['spearman']:+.2f}**, permutation *p* = {R['spearman_p']:.3f}), stable in "
           f"sign across both eras, textbook crash accounting (worst week "
           f"{R['worst_week']:+.1f}%, max DD {R['max_dd']:+.1f}%). But the strict green bar — "
           f"a robust Newey-West |t| ≥ 2 holding across sub-eras — is not cleared: the "
           f"basket-skew NW *t* is only **{R['skew_t']:+.2f}** ({R['era_early_skew_t']:+.2f} / "
           f"{R['era_late_skew_t']:+.2f} by era) and the slope *t* falls to "
           f"{R['era_late_slope_t']:+.2f} late. The 20-seed synthetic control fires on "
           f"{R['null_fire']}/20 nulls and recovers a planted relation cleanly, so the "
           f"borderline real result is honest, not machinery.\n"
           f"- **Tradability — Mirage.** The premium is weak (**{R['premium_ann']:+.2f}%/yr**, "
           f"*t* = {R['premium_t']:+.2f}) and collapses to **{R['timer_50_net']:+.2f}%/yr** "
           f"(Sharpe {R['timer_50_sh']:.2f}) at 50 bps/yr borrow — pennies in front of a "
           f"−36% steamroller."),
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
