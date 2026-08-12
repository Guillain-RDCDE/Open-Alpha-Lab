"""Generate the two narrative notebooks for Study 849 (Dry January / Veganuary).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from the
frozen ``R`` dict (mirroring docs/results.md); the live cells run only the fast synthetic
positive control, so execution is quick and network-free.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily total-return
# closes, 8 tickers, 1999-01-04 -> 2026-06-30; abnormal return = group - SPY; monthly).
R = dict(
    start="1999-01-04", end="2026-06-30", as_of="2026-06-30", fingerprint="e5210651db30",
    n_rows=6914, n_tickers=8,
    # January abnormal returns
    alc_jan_n=27, alc_jan_mean=-1.20, alc_jan_t1s=-1.28, alc_jan_tnw=-1.46, alc_jan_hit="10/27",
    plant_jan_n=7, plant_jan_mean=12.71, plant_jan_t1s=1.23, plant_jan_tnw=1.82, plant_jan_hit="5/7",
    stap_jan_n=27, stap_jan_mean=-0.62, stap_jan_t1s=-0.80,
    spread_jan_n=7, spread_jan_mean=14.46, spread_jan_t1s=1.33, spread_jan_tnw=1.76, spread_jan_hit="5/7",
    # February hangover
    alc_feb_mean=0.12, alc_feb_t1s=0.13,
    # placebo (directional, rank of Jan among 12 months)
    alc_placebo_rank=1, alc_placebo_p=0.083,
    plant_placebo_rank=1, plant_placebo_p=0.083,
    # two-era robustness (alcohol Jan)
    era_early_n=13, era_early_mean=-1.58, era_early_t=-1.37,
    era_late_n=14, era_late_mean=-0.85, era_late_t=-0.57,
    # timer
    timer5_gross=14.46, timer5_cost=0.24, timer5_net=14.22, timer5_t=1.31,
    timer10_net=14.02, timer10_t=1.29,
    # synthetic control
    null_t=-0.20, null_fire=0, edge2_t=1.39, edge2_fire=20, edge5_t=3.77, edge5_fire=90,
)


HEADER = f"""# Study 849 — Dry January / Veganuary 🍸🥦

**Do the January "Dry January" (abstain from alcohol) and "Veganuary" (go plant-based) waves
move the stocks?**

If the two cultural campaigns shift consumer demand hard and predictably enough, they should
leave a January seasonal in the **abnormal** return (`group − SPY`): the alcohol names
(`BUD STZ TAP DEO SAM`) *down*, plant-based `BYND` *up*, staples `XLP` flat. We read the
January (and February) abnormal return across every year on yfinance daily total-return
closes ({R['start']} → {R['end']}, {R['n_tickers']} tickers).

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. As-of {R['as_of']}, fingerprint `{R['fingerprint']}`.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Millions pledge to skip alcohol (Dry January) or eat plant-based (Veganuary) "
           "every January, and drinks makers routinely flag a soft January. If that demand "
           "shift reached prices it would show up as a January dip in the alcohol names and a "
           "January pop in a plant-based name — measured *against the market* (`− SPY`), so "
           "the market's own turn-of-year seasonality is stripped out. Because January is "
           "always January, the test is zero look-ahead by construction."),
        code(
            "R = dict(alc_jan_mean=%r, alc_jan_t1s=%r, plant_jan_mean=%r, plant_jan_t1s=%r,\n"
            "         spread_jan_mean=%r, spread_jan_t1s=%r, stap_jan_mean=%r)\n"
            "print('January abnormal return (group - SPY):')\n"
            "print('  alcohol basket : %%+.2f%%%%  (t = %%+.2f)  <- Dry January drag' %% (R['alc_jan_mean'], R['alc_jan_t1s']))\n"
            "print('  plant  (BYND)  : %%+.2f%%%%  (t = %%+.2f)  <- Veganuary lift (only 7 yrs!)' %% (R['plant_jan_mean'], R['plant_jan_t1s']))\n"
            "print('  staples (XLP)  : %%+.2f%%%%            <- flat control' %% R['stap_jan_mean'])\n"
            "print('  plant - alcohol: %%+.2f%%%%  (t = %%+.2f)  <- pure thesis' %% (R['spread_jan_mean'], R['spread_jan_t1s']))"
            % (R["alc_jan_mean"], R["alc_jan_t1s"], R["plant_jan_mean"], R["plant_jan_t1s"],
               R["spread_jan_mean"], R["spread_jan_t1s"], R["stap_jan_mean"])
        ),
        md("## 2. Is the detector honest? A live synthetic control\n\n"
           "We plant a tunable January abnormal-return seasonal in a seeded toy world "
           "(`edge>0`) and check the monthly-seasonality detector recovers it — and that it "
           "stays *silent* on the null (`edge=0`, no seasonal). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from dry_january import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=849), month=1)\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.05, seed=849), month=1)\n"
            "print('null world   : Jan dummy NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: Jan dummy NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — the right footprint, too faint to trade\n\n"
           f"The folklore points **exactly the right way**: in January the alcohol basket is "
           f"the year's *most-negative* month (rank {R['alc_placebo_rank']}/12) and `BYND` the "
           f"*most-positive* (rank {R['plant_placebo_rank']}/12). But the only well-sampled leg "
           f"— the alcohol drag over {R['alc_jan_n']} Januaries — is just **{R['alc_jan_mean']:+.2f}%** "
           f"at *t* = **{R['alc_jan_t1s']:+.2f}**, and the eye-catching "
           f"**{R['spread_jan_mean']:+.2f}%** plant-minus-alcohol spread rests on only "
           f"**{R['spread_jan_n']}** noisy `BYND` years (*t* = {R['spread_jan_t1s']:+.2f}). "
           f"The February 'hangover' is absent ({R['alc_feb_mean']:+.2f}%, *t* = {R['alc_feb_t1s']:+.2f}). "
           f"Nothing clears |*t*| ≥ 2. The synthetic control shows a *real* +5% seasonal would "
           f"fire {R['edge5_fire']}% of the time, so this is a genuine but sub-threshold hint. "
           "**Signal: Weak. Tradability: Mirage.**"),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 849 — Dry January / Veganuary — the teardown\n\n"
           "The January/February abnormal returns, the Newey-West January-dummy *t*, the "
           "twelve-month calendar placebo, the two-era robustness cut, the costed timer, and "
           "the 20-seed synthetic control."),
        code("R = %r" % (R,)),
        md("## The headline — January abnormal return (group − SPY), one obs/year"),
        code(
            "print(f\"alcohol         (n={R['alc_jan_n']:>2}) : {R['alc_jan_mean']:+.2f}%  \"\n"
            "      f\"t_1s = {R['alc_jan_t1s']:+.2f}  NW t = {R['alc_jan_tnw']:+.2f}  hit {R['alc_jan_hit']}\")\n"
            "print(f\"plant  (BYND)   (n={R['plant_jan_n']:>2}) : {R['plant_jan_mean']:+.2f}%  \"\n"
            "      f\"t_1s = {R['plant_jan_t1s']:+.2f}  NW t = {R['plant_jan_tnw']:+.2f}  hit {R['plant_jan_hit']}\")\n"
            "print(f\"staples (XLP)   (n={R['stap_jan_n']:>2}) : {R['stap_jan_mean']:+.2f}%  \"\n"
            "      f\"t_1s = {R['stap_jan_t1s']:+.2f}   <- flat control\")\n"
            "print(f\"plant - alcohol (n={R['spread_jan_n']:>2}) : {R['spread_jan_mean']:+.2f}%  \"\n"
            "      f\"t_1s = {R['spread_jan_t1s']:+.2f}  NW t = {R['spread_jan_tnw']:+.2f}  hit {R['spread_jan_hit']}\")"
        ),
        md("## Calendar placebo — is January special, or one of twelve months?\n\n"
           "Rank January's mean abnormal return among all twelve calendar months (directional)."),
        code(
            "print(f\"alcohol (left tail) : Jan is rank {R['alc_placebo_rank']}/12  -> p = {R['alc_placebo_p']:.3f}\")\n"
            "print(f\"plant   (right tail): Jan is rank {R['plant_placebo_rank']}/12  -> p = {R['plant_placebo_p']:.3f}\")\n"
            "print('Both land January at rank 1/12 in the predicted direction — but a pre-specified')\n"
            "print('month placebo can do no better than p = 1/12 = 0.083, which misses 0.05.')"
        ),
        md("## Robustness — alcohol January abnormal return, two eras (split 2013)"),
        code(
            "print(f\"1999-2012 (n={R['era_early_n']}): {R['era_early_mean']:+.2f}%  t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2013-2026 (n={R['era_late_n']}): {R['era_late_mean']:+.2f}%  t = {R['era_late_t']:+.2f}\")\n"
            "print('Right sign in both halves, but fades to t = -0.57 in the modern era.')"
        ),
        md("## The timer — long-plant / short-alcohol every January, costed\n\n"
           "Enter December close, exit January close; 2 legs × round-trip + short borrow."),
        code(
            "for tag,g,c,n,t in [('5 bps',R['timer5_gross'],R['timer5_cost'],R['timer5_net'],R['timer5_t']),\n"
            "                    ('10 bps',R['timer5_gross'],R['timer5_cost']*2,R['timer10_net'],R['timer10_t'])]:\n"
            "    print(f\"{tag:>6} one-way: gross {g:+.2f}% -> net {n:+.2f}%/yr (t = {t:+.2f}, only 7 obs)\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted seasonal."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from dry_january import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=849+s), month=1)['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds : NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.05, seed=849), month=1)\n"
            "print(f\"planted (edge=0.05)    : NW t = {planted['t_nw']:+.2f}, beta = {planted['beta_pct']:+.2f}%\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** The Dry-January / Veganuary footprint is *directionally "
           f"perfect* — alcohol is the year's most-negative calendar month "
           f"({R['alc_jan_mean']:+.2f}%, rank 1/12) and `BYND` the most-positive (rank 1/12), "
           f"sign-consistent across both eras — but the only well-sampled leg reaches just "
           f"*t* = {R['alc_jan_t1s']:+.2f} ({R['alc_jan_n']} Januaries), the plant/spread side "
           f"rides {R['spread_jan_n']} noisy `BYND` years (*t* = {R['spread_jan_t1s']:+.2f}), "
           f"the February hangover is absent, and **no cut clears |*t*| ≥ 2**. The 20-seed "
           f"control fires {R['edge5_fire']}% on a genuine +5% seasonal and "
           f"{R['null_fire']}/20 on the null, so the whiff is real-but-sub-threshold.\n"
           f"- **Tradability — Mirage.** The long-plant / short-alcohol January timer nets "
           f"**{R['timer5_net']:+.2f}%/yr** at 5 bps — but on {R['spread_jan_n']} observations "
           f"at *t* = {R['timer5_t']:+.2f}, driven by one small-cap's early tape, "
           f"capacity-trivial and inside a multi-cut search. Nothing bankable."),
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
