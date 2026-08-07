"""Generate the two narrative notebooks for Study 809 (Signed Jump Variation).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily OHLC,
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; trailing-21d signed jump
# variation (RS+ - RS-)/RV sort, long bottom30% (low SJ) / short top30% (high SJ)).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_days=4126,
    spread_bps=-1.71, t_nw=-1.36, t_1s=-1.29,
    lo_bps=7.20, hi_bps=8.91, welch_t=-0.66, gross_sharpe=-0.32,
    placebo_obs=-1.71, placebo_mean=0.025, placebo_sd=0.899,
    placebo_p=0.97400, placebo_sigma_left=1.9, placebo_draws=1000,
    era_early_bps=-0.35, era_early_t=-0.26, era_early_n=1992,
    era_late_bps=-2.98, era_late_t=-1.44, era_late_n=2134,
    timer_1_gross=-1.71, timer_1_cost=2.14, timer_1_net=-3.85, timer_1_t=-2.89,
    timer_5_gross=-1.71, timer_5_cost=10.14, timer_5_net=-11.85, timer_5_t=-8.91,
    null_mean_t=-0.31, null_sd_t=0.82, null_fire=0,
    planted_t=3.66, planted_welch=3.57,
)


HEADER = f"""# Study 809 — Signed Jump Variation ⚡📉

**Do stocks whose recent variance is *downside*-dominated go on to earn *more*?**

Barndorff-Nielsen, Kinnebrock & Shephard (2010) split realized variance by the **sign of the
return** — upside `RS+ = Σ r²·1(r>0)` vs downside `RS- = Σ r²·1(r<0)` — and Bollerslev, Li &
Zhao (2020) show the **signed jump variation** `SJ = (RS+ − RS-)/RV` is priced *negatively*:
the **downside** ("bad" volatility) names carry a **premium**, the **upside** ("good"
volatility) names under-earn. A long **low-SJ** / short **high-SJ** book should earn a positive
spread. We take the self-contained daily version on a liquid US cross-section
({R['start']} → {R['end']}, {R['n_names']} names).

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: current-membership mega-caps — magnitudes are an upper
bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Split a month's daily variance into the part from **up** days (`RS+`, 'good' "
           "volatility) and the part from **down** days (`RS-`, 'bad' volatility). The "
           "signed jump `SJ = (RS+ − RS-)/RV` is positive when the wobble is mostly on the "
           "upside, negative when it is mostly on the downside. The theory: downside "
           "volatility is genuinely feared, so bearing it is *paid* — buy the "
           "downside-heavy names, sell the lottery-like upside-heavy ones."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, lo_bps=%r, hi_bps=%r, gross_sharpe=%r)\n"
            "print('long low-SJ / short high-SJ spread: %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  low-SJ (downside) book %%+.2f bps vs high-SJ (upside) book %%+.2f bps'\n"
            "      %% (R['lo_bps'], R['hi_bps']))\n"
            "print('  gross spread Sharpe (before cost): %%.2f' %% R['gross_sharpe'])"
            % (R["spread_bps"], R["t_nw"], R["lo_bps"], R["hi_bps"], R["gross_sharpe"])
        ),
        md("## 2. Is the sort just noise? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world (`edge>0`) and check the detector "
           "recovers it — and that it stays *silent* on the null (`edge=0`, signed jump "
           "present but unpriced). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from signed_jump import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=809, n_assets=40, n_days=1200))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0024, seed=809, n_assets=40, n_days=1500))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — the famous edge does *not* replicate here\n\n"
           f"On this liquid mega-cap tape the long-low-SJ / short-high-SJ spread is "
           f"**{R['spread_bps']:+.2f} bps/day** with NW *t* = **{R['t_nw']:+.2f}** — "
           f"**not significant** (|*t*| < 2), and the point estimate even leans the "
           f"**wrong way**: the upside-dominated ('good' volatility) names, if anything, "
           f"*out-earned* the downside names (the permutation null centres at 0 with sd "
           f"{R['placebo_sd']:.2f} bps; the observed value is only ~{R['placebo_sigma_left']:.1f}σ "
           f"into the *left* tail). The seeded synthetic control recovers a *planted* "
           f"Bollerslev-Li-Zhao relation cleanly, so this is a genuine **absence** on the "
           f"mega-cap survivor universe, not a bug — the signed-jump premium is a "
           f"smaller-cap phenomenon that does not survive on 50 mega-caps. "
           f"**Signal: None** (the claimed edge is absent), **Tradability: Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 809 — Signed Jump Variation — the teardown\n\n"
           "The per-leg splits, the Newey-West spread *t*, the pooled Welch book test, the "
           "1,000-permutation placebo, the two-era robustness cut, the costed timer, and the "
           "20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The headline — long-low-SJ / short-high-SJ spread\n\n"
           "Daily equal-weight bottom-30% (low SJ) minus top-30% (high SJ) signed-jump spread."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : low-SJ {R['lo_bps']:+.2f} vs high-SJ {R['hi_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:.2f} (before cost)\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> right-tail p = {R['placebo_p']:.5f} \"\n"
            "      f\"(~{R['placebo_sigma_left']:.1f}sigma into the LEFT tail)\")"
        ),
        md("## Robustness — two eras (split 2018-01-01)"),
        code(
            "print(f\"2010-2017 (n={R['era_early_n']}): {R['era_early_bps']:+.2f} bps  NW t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2018-2026 (n={R['era_late_n']}): {R['era_late_bps']:+.2f} bps  NW t = {R['era_late_t']:+.2f}\")"
        ),
        md("## The timer — can you get paid for it?\n\n"
           "2 sides × one-way cost × NAV per day on the long-short book; short pays 50 bps/yr borrow."),
        code(
            "for tag,g,c,n,t in [('1 bp',R['timer_1_gross'],R['timer_1_cost'],R['timer_1_net'],R['timer_1_t']),\n"
            "                    ('5 bps',R['timer_5_gross'],R['timer_5_cost'],R['timer_5_net'],R['timer_5_t'])]:\n"
            "    print(f\"{tag:>5} one-way: gross {g:+.2f} -> net {n:+.2f} bps/day (cost {c:.2f}/day, t={t:+.2f})\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted relation."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from signed_jump import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=809+s, n_assets=40, n_days=1200))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0024, seed=809, n_assets=40, n_days=1500))\n"
            "print(f\"planted (edge=0.0024): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The claimed Bollerslev-Li-Zhao negative-signed-jump premium does"
           f" **not** replicate on 50 liquid US mega-caps: the long-low-SJ / short-high-SJ spread"
           f" is **{R['spread_bps']:+.2f} bps/day** (NW *t* = **{R['t_nw']:+.2f}**, |*t*| < 2) — "
           f"insignificant *and* mildly *wrong-signed* (the permutation null centres at 0, sd "
           f"{R['placebo_sd']:.2f} bps; observed ~{R['placebo_sigma_left']:.1f}σ into the left "
           f"tail), and flat in both eras (*t* = {R['era_early_t']:+.2f} / "
           f"{R['era_late_t']:+.2f}). The 20-seed synthetic control recovers a *planted* relation "
           f"cleanly (*t* = {R['planted_t']:+.2f}, fires on {R['null_fire']}/20 nulls), so the "
           f"absence is real, not machinery. Survivorship biases the magnitude.\n"
           f"- **Tradability — Mirage.** The specified book loses money net at any cost: at 1 bp "
           f"one-way the friction ({R['timer_1_cost']:.2f} bps/day) already dwarfs the "
           f"{abs(R['spread_bps']):.2f} bps (insignificant) gross edge, net "
           f"**{R['timer_1_net']:+.2f} bps/day** (*t* = {R['timer_1_t']:+.2f}); at 5 bps "
           f"**{R['timer_5_net']:+.2f} bps/day**."),
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
