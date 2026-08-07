"""Generate the two narrative notebooks for Study 813 (Maximum-Drawdown Anomaly).

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
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; trailing-252d maximum
# drawdown sort, long bottom30% calm / short top30% distressed; spread = calm - distressed).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_days=3895,
    spread_bps=-4.35, t_nw=-2.36, t_1s=-2.34,
    lo_bps=5.34, hi_bps=9.68, welch_t=-1.50, gross_sharpe=-0.59,
    placebo_obs=-4.35, placebo_mean=0.046, placebo_sd=1.049,
    placebo_p=1.00000, placebo_sigma_left=4.19, placebo_draws=1000,
    era_early_bps=-2.41, era_early_t=-1.10, era_early_n=1761,
    era_late_bps=-5.94, era_late_t=-2.09, era_late_n=2134,
    timer_1_gross=-4.35, timer_1_cost=2.14, timer_1_net=-6.48, timer_1_t=-3.49,
    timer_5_gross=-4.35, timer_5_cost=10.14, timer_5_net=-14.48, timer_5_t=-7.79,
    flip_1_net=2.21, flip_1_t=1.19, flip_5_net=-5.79, flip_5_t=-3.11,
    null_mean_t=-0.53, null_sd_t=1.07, null_fire=2,
    planted_t=9.03, planted_welch=9.18,
    fingerprint="357fd262912f",
)


HEADER = f"""# Study 813 — Maximum-Drawdown Anomaly 📉

**When a stock just took its deepest 12-month drawdown, does it keep sinking — or bounce?**

Sort a cross-section of stocks on each name's **trailing 12-month maximum drawdown** (the
largest peak-to-trough decline of its cumulative total return). The distress story says the
deepest-drawdown names keep **under-earning**; the reversal story says they **rebound**. We
take no prior — we sort into fractiles on a liquid US cross-section
({R['start']} → {R['end']}, {R['n_names']} names), measure the forward long-short spread, and
report the sign we actually find.

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: current-membership mega-caps — the deepest drawdowns of
all (names that fell and never recovered) are exactly what this survivor panel deletes, so
magnitudes are an upper bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Each name's **maximum drawdown** is how far it fell from its own running peak "
           "over the last 12 months — a pure, price-based distress gauge. Rank the "
           "cross-section: the calm names (shallow drawdown) on one side, the recently "
           "battered names (deep drawdown) on the other. Then watch what happens *next*. "
           "Two rival stories: **distress** (the wounded keep bleeding) vs **reversal** "
           "(the wounded bounce). Only the tape settles it."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, lo_bps=%r, hi_bps=%r, gross_sharpe=%r)\n"
            "print('long calm / short distressed spread: %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  calm book %%+.2f bps vs distressed book %%+.2f bps'\n"
            "      %% (R['lo_bps'], R['hi_bps']))\n"
            "print('  spread = calm - distressed; NEGATIVE => the distressed names OUT-earned (a rebound)')"
            % (R["spread_bps"], R["t_nw"], R["lo_bps"], R["hi_bps"], R["gross_sharpe"])
        ),
        md("## 2. Is the sort just lucky? A live synthetic control\n\n"
           "We plant a *distress* effect in a seeded toy world (`edge>0`: fragile names "
           "have deep drawdowns AND low forward returns) and check the detector recovers "
           "it — and that it stays *silent* on the null (`edge=0`, drawdowns present but "
           "unpriced). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from max_drawdown import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=813, n_assets=40, n_days=1500))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.004, seed=813, n_assets=40, n_days=1500))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up POSITIVE = distress)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — a fragile *reversal*, not a distress premium\n\n"
           f"On this liquid mega-cap tape the long-calm / short-distressed spread is "
           f"**{R['spread_bps']:+.2f} bps/day** with NW *t* = **{R['t_nw']:+.2f}** — the "
           f"*negative* sign means the **distressed** (deep-drawdown) names actually "
           f"**out-earned** the calm ones (distressed book **{R['hi_bps']:+.2f}** vs calm "
           f"**{R['lo_bps']:+.2f}** bps). So on mega-caps it's the **reversal** story, not "
           f"distress: the wounded bounced. But the effect is **not robust** — it lives in "
           f"the {R['era_late_n']:,}-day 2018–2026 era (*t* = {R['era_late_t']:+.2f}) and is "
           f"absent 2010–2017 (*t* = {R['era_early_t']:+.2f}). And it does not pay: the "
           f"specified book loses money outright, and even the profitable *rebound* "
           f"direction earns only +{R['flip_1_net']:.2f} bps/day net at a fantasy 1 bp cost "
           f"(*t* = {R['flip_1_t']:+.2f}, not significant) and dies at 5 bps. "
           f"**Signal: Weak** (a significant but era-fragile rebound), **Tradability: "
           f"Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 813 — Maximum-Drawdown Anomaly — the teardown\n\n"
           "The per-leg splits, the Newey-West spread *t*, the pooled Welch book test, "
           "the 1,000-permutation placebo, the two-era robustness cut, the costed timer "
           "(both directions), and the 20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The headline — long-calm / short-distressed spread\n\n"
           "Daily equal-weight bottom-30% (shallow drawdown) minus top-30% (deep drawdown) "
           "spread. `spread = calm - distressed`; a negative value means the distressed "
           "names out-earned (a rebound)."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : calm {R['lo_bps']:+.2f} vs distressed {R['hi_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:.2f} (calm-minus-distressed, before cost)\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> right-tail p = {R['placebo_p']:.5f}\")\n"
            "print(f\"observed sits ~{R['placebo_sigma_left']:.2f} sigma into the LEFT tail -> the (reversal) spread is not a lucky sort\")"
        ),
        md("## Robustness — two eras (split 2018-01-01)\n\n"
           "The decisive honesty check: does the rebound hold across time?"),
        code(
            "print(f\"2010-2017 (n={R['era_early_n']}): {R['era_early_bps']:+.2f} bps  NW t = {R['era_early_t']:+.2f}  <- NOT significant\")\n"
            "print(f\"2018-2026 (n={R['era_late_n']}): {R['era_late_bps']:+.2f} bps  NW t = {R['era_late_t']:+.2f}  <- carries the full-sample result\")"
        ),
        md("## The timer — can you get paid for it (either direction)?\n\n"
           "2 sides × one-way cost × NAV per day on the long-short book; short pays 50 bps/yr borrow."),
        code(
            "print('specified book (long calm / short distressed):')\n"
            "for tag,g,c,n,t in [('1 bp',R['timer_1_gross'],R['timer_1_cost'],R['timer_1_net'],R['timer_1_t']),\n"
            "                    ('5 bps',R['timer_5_gross'],R['timer_5_cost'],R['timer_5_net'],R['timer_5_t'])]:\n"
            "    print(f\"  {tag:>5} one-way: gross {g:+.2f} -> net {n:+.2f} bps/day (cost {c:.2f}/day, t={t:+.2f})\")\n"
            "print('sign-flipped REBOUND book (long distressed / short calm):')\n"
            "for tag,n,t in [('1 bp',R['flip_1_net'],R['flip_1_t']),('5 bps',R['flip_5_net'],R['flip_5_t'])]:\n"
            "    print(f\"  {tag:>5} one-way: net {n:+.2f} bps/day (t={t:+.2f})\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted "
           "distress relation (deep drawdown -> low forward return -> positive calm-minus-distressed spread)."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from max_drawdown import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=813+s, n_assets=40, n_days=1500))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.004, seed=813, n_assets=40, n_days=1500))\n"
            "print(f\"planted (edge=0.004): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** Sorting on the trailing 12-month maximum drawdown, the "
           f"long-calm / short-distressed spread is **{R['spread_bps']:+.2f} bps/day** "
           f"(NW *t* = **{R['t_nw']:+.2f}**): the *distressed* names **out-earned** — a "
           f"**drawdown reversal**, one of the two outcomes the claim entertained. The "
           f"1,000-permutation placebo confirms it isn't a lucky sort "
           f"(~{R['placebo_sigma_left']:.1f}σ into the left tail). But it is **not robust "
           f"across eras** (*t* = {R['era_early_t']:+.2f} in 2010–2017 vs "
           f"{R['era_late_t']:+.2f} in 2018–2026), so it clears the pooled |t|≥2 bar only "
           f"marginally and on one half of the sample — **Weak, not Real**. The 20-seed "
           f"synthetic control recovers a *planted distress* relation cleanly (*t* = "
           f"{R['planted_t']:+.2f}) and is quiet on the null. Survivorship biases the "
           f"magnitude (the deepest drawdowns — permanent losers — are absent).\n"
           f"- **Tradability — Mirage.** The specified book loses money "
           f"(**{R['timer_1_net']:+.2f} bps/day** net at 1 bp). The profitable *rebound* "
           f"direction earns only **+{R['flip_1_net']:.2f} bps/day** net at a fantasy 1 bp "
           f"(*t* = {R['flip_1_t']:+.2f} — not significant) and turns negative "
           f"({R['flip_5_net']:+.2f}) by 5 bps. No paycheck either way."),
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
