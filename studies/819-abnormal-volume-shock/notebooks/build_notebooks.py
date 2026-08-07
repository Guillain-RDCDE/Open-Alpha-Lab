"""Generate the two narrative notebooks for Study 819 (Abnormal-Volume Shock).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily OHLC +
# Volume, total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; 5d mean of 60d
# standardised abnormal volume, long top30% / short bottom30%).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_days=4083,
    spread_bps=0.55, t_nw=0.54, t_1s=0.52,
    hi_bps=7.44, lo_bps=6.89, welch_t=0.21, gross_sharpe=0.13,
    placebo_obs=0.55, placebo_mean=0.015, placebo_sd=0.865,
    placebo_p=0.266, placebo_sigma_right=0.62, placebo_draws=1000,
    era_early_bps=-0.48, era_early_t=-0.40, era_early_n=1949,
    era_late_bps=1.49, era_late_t=0.94, era_late_n=2134,
    timer_1_gross=0.55, timer_1_cost=2.14, timer_1_net=-1.59, timer_1_t=-1.49,
    timer_5_gross=0.55, timer_5_cost=10.14, timer_5_net=-9.59, timer_5_t=-8.98,
    null_mean_t=-0.27, null_sd_t=1.05, null_fire=2,
    planted_t=21.64, planted_welch=22.46,
)


HEADER = f"""# Study 819 — Abnormal-Volume Shock 📊📣

**Do stocks that print abnormally heavy volume go on to drift *up*?**

Garfinkel & Sokobin (2006) argue that trading volume a name's own recent norm cannot
explain is a footprint of **attention / opinion divergence**, and that higher unexplained
volume is followed by a **positive subsequent drift**. We take the self-contained daily
version on a liquid US cross-section ({R['start']} → {R['end']}, {R['n_names']} names):
each name's **standardised abnormal volume** `(V − mean_60)/std_60`, averaged over a 5-day
formation window, then a long-high / short-low sort.

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: current-membership mega-caps — magnitudes are an upper
bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "When a name suddenly trades *far more* than its own 60-day norm, that burst is "
           "a footprint of fresh information and disagreement about it. If attention and "
           "disagreement resolve slowly, the name should keep drifting for a few days. "
           "So: measure how many sigmas above its own benchmark each name is trading, "
           "buy the loud names, sell the quiet ones."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, hi_bps=%r, lo_bps=%r, gross_sharpe=%r)\n"
            "print('long high-avol / short low-avol spread: %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  high-avol book %%+.2f bps vs low-avol book %%+.2f bps'\n"
            "      %% (R['hi_bps'], R['lo_bps']))\n"
            "print('  gross spread Sharpe (before cost): %%.2f' %% R['gross_sharpe'])"
            % (R["spread_bps"], R["t_nw"], R["hi_bps"], R["lo_bps"], R["gross_sharpe"])
        ),
        md("## 2. Is the sort real? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world (`edge>0`: an attention shock that "
           "inflates volume *and* lifts the forward return) and check the detector "
           "recovers it — and that it stays *silent* on the null (`edge=0`, abnormal "
           "volume present but unpriced). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from volume_shock import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=819, n_assets=40, n_days=1200))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0020, seed=819, n_assets=40, n_days=1500))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — the famous drift does *not* show up here\n\n"
           f"On this liquid mega-cap tape the long-high-avol / short-low-avol spread is "
           f"**{R['spread_bps']:+.2f} bps/day** with NW *t* = **{R['t_nw']:+.2f}** — the "
           f"sign matches Garfinkel & Sokobin, but the magnitude is a rounding error: the "
           f"permutation null centres at 0 (sd {R['placebo_sd']:.2f} bps) and the observed "
           f"value is only ~{R['placebo_sigma_right']:.1f}σ into the right tail "
           f"(p = {R['placebo_p']:.2f}). It even flips sign across the two eras "
           f"({R['era_early_t']:+.2f} / {R['era_late_t']:+.2f}). The seeded synthetic "
           f"control recovers a *planted* attention→drift relation overwhelmingly "
           f"(*t* = {R['planted_t']:+.1f}), so this is a genuine absence on mega-caps, not a "
           "dead engine — the disagreement drift is an earnings-window / small-cap effect "
           "diluted to nothing here. **Signal: None**, **Tradability: Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 819 — Abnormal-Volume Shock — the teardown\n\n"
           "The per-leg splits, the Newey-West spread *t*, the pooled Welch book test, "
           "the 1,000-permutation placebo, the two-era robustness cut, the costed timer, "
           "and the 20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The headline — long-high-avol / short-low-avol spread\n\n"
           "Daily equal-weight top-30% minus bottom-30% abnormal-volume spread."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : high-avol {R['hi_bps']:+.2f} vs low-avol {R['lo_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:.2f} (before cost)\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> p = {R['placebo_p']:.5f}  \"\n"
            "      f\"(~{R['placebo_sigma_right']:.2f} sigma into the right tail)\")"
        ),
        md("## Robustness — two eras (split 2018-01-01)"),
        code(
            "print(f\"2010-2017 (n={R['era_early_n']}): {R['era_early_bps']:+.2f} bps  NW t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2018-2026 (n={R['era_late_n']}): {R['era_late_bps']:+.2f} bps  NW t = {R['era_late_t']:+.2f}\")\n"
            "print('  -> the sign FLIPS between halves; neither is significant')"
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
            "from volume_shock import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=819+s, n_assets=40, n_days=1200))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0020, seed=819, n_assets=40, n_days=1500))\n"
            "print(f\"planted (edge=0.0020): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The claimed Garfinkel-Sokobin abnormal-volume drift does "
           f"**not** replicate as a tradable spread on 50 liquid US mega-caps: the "
           f"long-high-avol / short-low-avol spread is **{R['spread_bps']:+.2f} bps/day** "
           f"(NW *t* = **{R['t_nw']:+.2f}**) — correctly signed but statistically absent, "
           f"only ~{R['placebo_sigma_right']:.2f}σ into the placebo (p = {R['placebo_p']:.2f}), "
           f"and it **flips sign** across the two eras (*t* = {R['era_early_t']:+.2f} / "
           f"{R['era_late_t']:+.2f}). The synthetic control recovers a *planted* relation "
           f"overwhelmingly (*t* = {R['planted_t']:+.1f}; null ≈ N(0,1)), so the flat result "
           f"is the data, not machinery.\n"
           f"- **Tradability — Mirage.** The right-signed {R['timer_1_gross']:+.2f} bps/day "
           f"gross tilt is dwarfed by the {R['timer_1_cost']:.2f} bps/day friction at 1 bp "
           f"one-way, net **{R['timer_1_net']:+.2f} bps/day** (*t* = {R['timer_1_t']:+.2f}); "
           f"at 5 bps **{R['timer_5_net']:+.2f} bps/day**."),
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
