"""Generate the two narrative notebooks for Study 808 (Continuing Overreaction).

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
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; monthly CO(12,skip1)
# weighted signed momentum, long top30% / short bottom30%).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_months=184,
    spread_bps=15.46, t_nw=0.58, t_1s=0.55,
    hi_bps=152.94, lo_bps=137.48, welch_t=0.31, gross_sharpe=0.14,
    placebo_obs=15.46, placebo_mean=1.33, placebo_sd=20.82,
    placebo_p=0.26100, placebo_sigma=0.68, placebo_draws=1000,
    era_early_bps=8.23, era_early_t=0.26, era_early_n=82,
    era_late_bps=21.27, era_late_t=0.54, era_late_n=102,
    timer_1_gross=15.46, timer_1_cost=6.17, timer_1_net=9.29, timer_1_t=0.33,
    timer_5_gross=15.46, timer_5_cost=14.17, timer_5_net=1.29, timer_5_t=0.05,
    null_mean_t=-0.47, null_sd_t=0.95, null_fire=1,
    planted_t=8.61, planted_welch=5.86,
)


HEADER = f"""# Study 808 — Continuing Overreaction 🔁

**Does a name on a persistent recent up-streak keep running?**

Byun, Lim & Yun (2016) build a **weighted signed-momentum** score — a recency-weighted
sum of the *signs* of a stock's recent monthly returns (recent months count most). A
high positive score marks a consistent up-streak ("continuing overreaction"), which is
supposed to predict the cross-section *positively*: the streak keeps going. We take the
self-contained monthly version on a liquid US cross-section ({R['start']} → {R['end']},
{R['n_names']} names).

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: current-membership mega-caps — magnitudes are an upper
bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Take a stock's last 12 monthly returns, drop the most recent month, and just "
           "look at the **signs**: up, up, down, up… Weight the *recent* months more and "
           "add them up (normalised) to get a score between −1 and +1. A score near +1 is "
           "a consistent recent up-streak. The behavioural story ('continuing "
           "overreaction') says investors keep chasing that streak, so it should keep "
           "running — buy the high-CO names, sell the low-CO ones."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, hi_bps=%r, lo_bps=%r, gross_sharpe=%r)\n"
            "print('long high-CO / short low-CO spread: %%+.2f bps/month (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  high-CO book %%+.2f bps vs low-CO book %%+.2f bps'\n"
            "      %% (R['hi_bps'], R['lo_bps']))\n"
            "print('  gross spread Sharpe (before cost, ann.): %%.2f' %% R['gross_sharpe'])"
            % (R["spread_bps"], R["t_nw"], R["hi_bps"], R["lo_bps"], R["gross_sharpe"])
        ),
        md("## 2. Is the sort just lucky? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world (`edge>0`: a persistent monthly "
           "trend state drives both past signs and the forward month) and check the "
           "detector recovers it — and that it stays *silent* on the null (`edge=0`, signs "
           "are coin-flips). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from continuing_overreaction import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=808, n_assets=40, n_days=1800))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.02, seed=808, n_assets=40, n_days=1800))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — the famous edge does *not* replicate here\n\n"
           f"On this liquid mega-cap tape the long-high-CO / short-low-CO spread is "
           f"**{R['spread_bps']:+.2f} bps/month** with NW *t* = **{R['t_nw']:+.2f}** — the "
           f"*right sign* (a whisper of continuation) but **statistically indistinguishable "
           f"from zero**: a monthly Sharpe of {R['gross_sharpe']:.2f}, the high-CO and low-CO "
           f"books within a rounding error ({R['hi_bps']:+.0f} vs {R['lo_bps']:+.0f} bps), and "
           f"only ≈{R['placebo_sigma']:+.2f}σ into a 1,000-permutation placebo (p = "
           f"{R['placebo_p']:.3f}). The seeded synthetic control recovers a *planted* "
           f"continuation cleanly, so this is a genuine null on the mega-cap universe, not a "
           f"bug — the continuing-overreaction premium simply does not bite on 50 well-arbitraged "
           f"mega-caps. **Signal: None** (the claimed edge is absent), **Tradability: Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 808 — Continuing Overreaction — the teardown\n\n"
           "The per-leg splits, the Newey-West spread *t*, the pooled Welch book test, "
           "the 1,000-permutation placebo, the two-era robustness cut, the costed timer, "
           "and the 20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The headline — long-high-CO / short-low-CO spread\n\n"
           "Monthly equal-weight top-30% minus bottom-30% CO spread."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/month  NW(6) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : high-CO {R['hi_bps']:+.2f} vs low-CO {R['lo_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:.2f} (before cost, ann.)\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.2f} \"\n"
            "      f\"(sd {R['placebo_sd']:.2f}) -> p = {R['placebo_p']:.5f} ({R['placebo_sigma']:+.2f}sigma)\")"
        ),
        md("## Robustness — two eras (split 2018-01-01)"),
        code(
            "print(f\"2010-2017 (n={R['era_early_n']}): {R['era_early_bps']:+.2f} bps  NW t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2018-2026 (n={R['era_late_n']}): {R['era_late_bps']:+.2f} bps  NW t = {R['era_late_t']:+.2f}\")"
        ),
        md("## The timer — can you get paid for it?\n\n"
           "2 sides × one-way cost × NAV per monthly rebalance; short pays 50 bps/yr borrow."),
        code(
            "for tag,g,c,n,t in [('1 bp',R['timer_1_gross'],R['timer_1_cost'],R['timer_1_net'],R['timer_1_t']),\n"
            "                    ('5 bps',R['timer_5_gross'],R['timer_5_cost'],R['timer_5_net'],R['timer_5_t'])]:\n"
            "    print(f\"{tag:>5} one-way: gross {g:+.2f} -> net {n:+.2f} bps/month (cost {c:.2f}/reb, t={t:+.2f})\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted continuation."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from continuing_overreaction import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=808+s, n_assets=40, n_days=1800))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.02, seed=808, n_assets=40, n_days=1800))\n"
            "print(f\"planted (edge=0.02): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The Byun-Lim-Yun continuing-overreaction premium does **not**"
           f" replicate on 50 liquid US mega-caps: the long-high-CO / short-low-CO spread is"
           f" **{R['spread_bps']:+.2f} bps/month** (NW *t* = **{R['t_nw']:+.2f}**) — the right"
           f" sign but statistically zero, ≈{R['placebo_sigma']:+.2f}σ in a 1,000-permutation"
           f" placebo (p = {R['placebo_p']:.3f}), flat in both eras (*t* = {R['era_early_t']:+.2f}"
           f" / {R['era_late_t']:+.2f}). The 20-seed synthetic control recovers a *planted*"
           f" continuation cleanly (*t* = {R['planted_t']:+.2f}, fires on {R['null_fire']}/20"
           f" nulls), so this is a genuine null, not machinery. Survivorship biases the magnitude"
           f" upward if anything.\n"
           f"- **Tradability — Mirage.** The book is insignificant gross; the monthly round-trip"
           f" friction eats it to **{R['timer_1_net']:+.2f} bps/month** (*t* = {R['timer_1_t']:+.2f})"
           f" at 1 bp and **{R['timer_5_net']:+.2f} bps** (*t* = {R['timer_5_t']:+.2f}) at 5 bps."),
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
