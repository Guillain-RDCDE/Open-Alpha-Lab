"""Generate the two narrative notebooks for Study 807 (Salience-Theory Returns).

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
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; trailing-21d salience-
# theory value sort, long bottom30% low-ST / short top30% high-ST).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_days=4125,
    fingerprint="357fd262912f",
    spread_bps=-1.00, t_nw=-0.78, t_1s=-0.74,
    lo_bps=7.49, hi_bps=8.49, welch_t=-0.37, gross_sharpe=-0.18,
    placebo_obs=-1.00, placebo_mean=0.009, placebo_sd=0.896,
    placebo_p=0.86700, placebo_sigma=-1.13, placebo_draws=1000,
    era_early_bps=-0.80, era_early_t=-0.60, era_early_n=1991,
    era_late_bps=-1.19, era_late_t=-0.55, era_late_n=2134,
    timer_1_gross=-1.00, timer_1_cost=2.14, timer_1_net=-3.14, timer_1_t=-2.32,
    timer_5_gross=-1.00, timer_5_cost=10.14, timer_5_net=-11.14, timer_5_t=-8.23,
    null_mean_t=-0.26, null_sd_t=0.91, null_fire=1,
    planted_t=4.31, planted_welch=4.46,
)


HEADER = f"""# Study 807 — Salience-Theory Returns ✨📉

**Do stocks whose recent *salient* days were UP go on to earn *less*?**

Cosemans & Frehen (2021), applying the **Bordalo-Gennaioli-Shleifer salience** model, argue
that investors over-weight a stock's most **salient** (attention-grabbing) days. Over the
trailing month each day's salience versus the market is
`σ = |rᵢ−rₘ| / (|rᵢ|+|rₘ|+θ)`, θ=0.1; the days are ranked and given declining decision
weights `δ^rank` (δ≈0.7); the **salience-theory value** `ST` is the salience-weighted mean of
market-excess returns. A **high** ST — the salient days were *up* — marks an over-priced name
that should **under-earn**, so a long **low-ST** / short **high-ST** book should earn a
positive spread. We take the self-contained daily version on a liquid US cross-section
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
           "Salience theory says a few big, attention-grabbing days do the thinking for us. "
           "If a stock's most *salient* recent days were **up** (versus the market), "
           "salience-loving investors over-value it — so its *future* return should be "
           "lower. Rank each name's month by day-salience, over-weight the salient days, and "
           "measure whether they were up or down: buy the low-ST names, sell the high-ST "
           "ones."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, lo_bps=%r, hi_bps=%r, gross_sharpe=%r)\n"
            "print('long low-ST / short high-ST spread: %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  low-ST book %%+.2f bps vs high-ST book %%+.2f bps'\n"
            "      %% (R['lo_bps'], R['hi_bps']))\n"
            "print('  gross spread Sharpe (before cost): %%.2f' %% R['gross_sharpe'])"
            % (R["spread_bps"], R["t_nw"], R["lo_bps"], R["hi_bps"], R["gross_sharpe"])
        ),
        md("## 2. Is the machinery honest? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world (`edge>0`) and check the detector "
           "recovers it — and that it stays *silent* on the null (`edge=0`, salience present "
           "but unpriced). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from salience_theory import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=807, n_assets=40, n_days=1200))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0016, seed=807, n_assets=40, n_days=1500))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — the edge does *not* show up here\n\n"
           f"On this liquid mega-cap tape the long-low-ST / short-high-ST spread is "
           f"**{R['spread_bps']:+.2f} bps/day** with NW *t* = **{R['t_nw']:+.2f}** — "
           f"statistically **indistinguishable from zero**. The low-ST book (+{R['lo_bps']:.2f} "
           f"bps) and the high-ST book (+{R['hi_bps']:.2f} bps) earn essentially the same; the "
           f"permutation null centres at zero (sd {R['placebo_sd']:.2f} bps) and the observed "
           f"value sits a mere ~{abs(R['placebo_sigma']):.1f}σ away (p = {R['placebo_p']:.2f}). "
           "The seeded synthetic control recovers a *planted* salience relation cleanly, so the "
           "flat real-tape result is a genuine null, not a broken detector — the Cosemans-Frehen "
           "premium is documented on the *broad* cross-section (including small, illiquid names) "
           "and simply does not appear on 50 mega-caps. **Signal: None**, and after costs "
           "**Tradability: Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 807 — Salience-Theory Returns — the teardown\n\n"
           "The per-leg splits, the Newey-West spread *t*, the pooled Welch book test, the "
           "1,000-permutation placebo, the two-era robustness cut, the costed timer, and the "
           "20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The headline — long-low-ST / short-high-ST spread\n\n"
           "Daily equal-weight bottom-30% (low ST) minus top-30% (high ST) spread."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : low-ST {R['lo_bps']:+.2f} vs high-ST {R['hi_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:.2f} (before cost)\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> p = {R['placebo_p']:.5f} (observed ~{R['placebo_sigma']:+.2f} sigma)\")"
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
            "from salience_theory import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=807+s, n_assets=40, n_days=1200))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0016, seed=807, n_assets=40, n_days=1500))\n"
            "print(f\"planted (edge=0.0016): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The Cosemans-Frehen salience-theory premium does **not** show "
           f"up on 50 liquid US mega-caps: the long-low-ST / short-high-ST spread is "
           f"**{R['spread_bps']:+.2f} bps/day** (NW *t* = **{R['t_nw']:+.2f}**) — indistinguishable "
           f"from zero, flat in both eras (*t* = {R['era_early_t']:+.2f} / {R['era_late_t']:+.2f}), "
           f"~{abs(R['placebo_sigma']):.1f}σ from the permutation-null centre (p = {R['placebo_p']:.2f}). "
           f"The synthetic control recovers a *planted* relation cleanly (*t* = {R['planted_t']:+.2f}), "
           f"so the flat tape is a real null, not machinery. The effect is a broad-cross-section "
           f"(small-cap-inclusive) phenomenon; survivorship on mega-caps kills what little is left.\n"
           f"- **Tradability — Mirage.** There is no gross edge to harvest ({R['spread_bps']:+.2f} bps/day), "
           f"and the book bleeds net: at 1 bp one-way the {R['timer_1_cost']:.2f} bps/day friction makes "
           f"it **{R['timer_1_net']:+.2f} bps/day** (*t* = {R['timer_1_t']:+.2f}); at 5 bps "
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
