"""Generate the two narrative notebooks for Study 871 (The Rank Effect).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily OHLC,
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; trailing-42d return
# rank, long middle 40% / short both 20% tails).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_days=4104,
    spread_bps=-1.65, t_nw=-1.86, t_1s=-1.88,
    mid_bps=6.71, ext_bps=8.36, welch_t=-0.64, gross_sharpe=-0.46,
    lvl_mid_pct=2.47, lvl_ext_pct=3.87,
    lc_spread_bps=0.11, lc_t_nw=0.18, lc_t_1s=0.17,
    placebo_obs=-1.65, placebo_mean=0.036, placebo_sd=0.789,
    placebo_p=0.98100, placebo_sigma_left=2.14, placebo_draws=1000,
    era_early_bps=-0.58, era_early_t=-0.50, era_early_n=1970,
    era_late_bps=-2.63, era_late_t=-1.99, era_late_n=2134,
    timer_1_gross=-1.65, timer_1_cost=2.14, timer_1_net=-3.79, timer_1_t=-4.31,
    timer_5_gross=-1.65, timer_5_cost=10.14, timer_5_net=-11.79, timer_5_t=-13.42,
    null_mean_t=0.24, null_sd_t=0.96, null_fire=1,
    planted_raw_t=4.14, planted_lc_t=2.33,
)


HEADER = f"""# Study 871 — The Rank Effect 🏅

**Do the best- and worst-ranked names in a portfolio go on to *under-earn* the middle?**

Hartzmark (2015) finds that investors disproportionately **sell the best- and
worst-ranked positions** in their portfolio — the salience of the extremes drives the
trade, not the raw return. That should put predictable selling pressure on the top- and
bottom-ranked names. We take the self-contained cross-sectional proxy on a liquid US
cross-section ({R['start']} → {R['end']}, {R['n_names']} names): each day rank by trailing
return, **long the middle, short both tails**, and — crucially — **control for the raw
return level**.

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: current-membership mega-caps — magnitudes are an upper
bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Look at your portfolio sorted by return. The **top** name and the **bottom** "
           "name jump out — they are *salient*. Hartzmark shows investors sell those "
           "extremes far more than the boring middle-ranked names, regardless of the "
           "actual return level. If everyone dumps the extremes, the top- and "
           "bottom-ranked names should face selling pressure and **under-earn the "
           "middle** next period. So: rank the cross-section, buy the middle, sell both "
           "tails — and make sure the raw return level isn't doing the work."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, mid_bps=%r, ext_bps=%r, "
            "lc_spread_bps=%r, lc_t_nw=%r, gross_sharpe=%r)\n"
            "print('long-middle / short-extremes spread: %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  middle book %%+.2f bps vs extremes book %%+.2f bps'\n"
            "      %% (R['mid_bps'], R['ext_bps']))\n"
            "print('  AFTER controlling for the raw return level: %%+.2f bps/day (t = %%+.2f)'\n"
            "      %% (R['lc_spread_bps'], R['lc_t_nw']))\n"
            "print('  gross spread Sharpe (before cost): %%.2f' %% R['gross_sharpe'])"
            % (R["spread_bps"], R["t_nw"], R["mid_bps"], R["ext_bps"],
               R["lc_spread_bps"], R["lc_t_nw"], R["gross_sharpe"])
        ),
        md("## 2. Is the sort just lucky? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world (`edge>0`: extreme-ranked names "
           "carry a forward penalty) and check the detector recovers it — raw *and* after "
           "controlling for the level — and that it stays *silent* on the null (`edge=0`, "
           "names still ranked but rank carries no forward information). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from rank_effect import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=871, n_assets=40, n_days=1200))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0016, seed=871, n_assets=40, n_days=1500))\n"
            "print('null world   : raw spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: raw spread NW t = %+.2f  (should light up)' % planted['t_nw'])\n"
            "print('planted world: level-controlled NW t = %+.2f  (survives the level control)' % planted['lc_t_nw'])"
        ),
        md("## 3. The honest verdict — the famous effect leaves *no* footprint here\n\n"
           f"On this liquid mega-cap tape the long-middle / short-extremes spread is "
           f"**{R['spread_bps']:+.2f} bps/day** with NW *t* = **{R['t_nw']:+.2f}** — the "
           f"**wrong sign** (the claim wants the extremes to under-earn, i.e. a *positive* "
           f"spread) and **not significant** (|t| < 2). Worse for the story: once you "
           f"**control for the raw return level** — the whole point of the rank effect — "
           f"the spread collapses to **{R['lc_spread_bps']:+.2f} bps/day** "
           f"(*t* = {R['lc_t_nw']:+.2f}), a flat zero. The tiny raw tilt was just momentum "
           f"in the tails, not a rank-position effect. The seeded synthetic control recovers "
           f"a *planted* rank-extremity relation cleanly, so this is a genuine absence, not "
           f"a broken sort — the rank effect is a **retail-position, trading-behaviour** "
           f"phenomenon that leaves no tradable cross-sectional signal on 50 mega-caps. "
           f"**Signal: None**, **Tradability: Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 871 — The Rank Effect — the teardown\n\n"
           "The per-leg splits, the Newey-West spread *t*, the level-controlled residual "
           "spread, the pooled Welch book test, the 1,000-permutation placebo, the two-era "
           "robustness cut, the costed timer, and the 20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The headline — long-middle / short-extremes spread\n\n"
           "Daily equal-weight middle-40% minus both-20%-tails rank-extremity spread."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : middle {R['mid_bps']:+.2f} vs extremes {R['ext_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"level neutral : middle trail {R['lvl_mid_pct']:+.2f}%  vs extremes {R['lvl_ext_pct']:+.2f}%\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:+.2f} (before cost)\")"
        ),
        md("## Controlling for the raw return level — the effect vanishes\n\n"
           "Residualise each day's forward return on a quadratic in the standardised "
           "trailing-return level (remove *any* smooth momentum/reversal curve), then "
           "re-measure the middle-minus-extremes spread. What survives is the pure "
           "rank-*position* effect."),
        code(
            "print(f\"level-controlled spread: {R['lc_spread_bps']:+.2f} bps/day  NW(10) t = {R['lc_t_nw']:+.2f}  \"\n"
            "      f\"(one-sample t = {R['lc_t_1s']:+.2f})  -> a flat zero\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> right-tail p = {R['placebo_p']:.5f} \"\n"
            "      f\"(~{R['placebo_sigma_left']:.2f}sigma into the WRONG (left) tail)\")"
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
            "from rank_effect import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=871+s, n_assets=40, n_days=1200))['lc_t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: level-ctrl NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0016, seed=871, n_assets=40, n_days=1500))\n"
            "print(f\"planted (edge=0.0016): raw NW t = {planted['t_nw']:+.2f}, level-ctrl NW t = {planted['lc_t_nw']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** Hartzmark's rank effect leaves **no** cross-sectional "
           f"return footprint on 50 liquid US mega-caps. The specified long-middle / "
           f"short-extremes spread is **{R['spread_bps']:+.2f} bps/day** (NW *t* = "
           f"**{R['t_nw']:+.2f}**) — *wrong-signed* and insignificant — and once you "
           f"**control for the raw return level** it collapses to "
           f"**{R['lc_spread_bps']:+.2f} bps/day** (*t* = {R['lc_t_nw']:+.2f}), a flat zero. "
           f"Not robust across eras (*t* = {R['era_early_t']:+.2f} / {R['era_late_t']:+.2f}); "
           f"the observed value sits ~{R['placebo_sigma_left']:.2f}σ into the *wrong* tail of "
           f"a 1,000-permutation placebo. The 20-seed synthetic control recovers a *planted* "
           f"relation cleanly (level-ctrl *t* = {R['planted_lc_t']:+.2f}, fires on "
           f"{R['null_fire']}/20 nulls), so the absence is real, not machinery.\n"
           f"- **Tradability — Mirage.** The specified book loses money gross and net "
           f"(**{R['timer_1_net']:+.2f} bps/day** at 1 bp one-way, {R['timer_5_net']:+.2f} at "
           f"5 bps); even the data-mined sign-flip is eaten by the "
           f"{R['timer_1_cost']:.2f} bps/day round-trip friction at 1 bp."),
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
