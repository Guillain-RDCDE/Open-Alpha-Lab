"""Generate the two narrative notebooks for Study 810 (Price Delay).

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
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; weekly delay sort,
# trailing-52w, contemp market + 4 weekly lags, long top30% (HIGH delay) / short
# bottom30% (LOW delay)). Reproduce with examples/verify.py.
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_weeks=804,
    spread_bps=2.45, t_nw=0.41, t_1s=0.43,
    long_bps=36.73, short_bps=34.28, welch_t=0.20, gross_sharpe=0.11,
    placebo_obs=2.45, placebo_mean=-0.057, placebo_sd=4.684,
    placebo_p=0.30100, placebo_draws=1000,
    era_early_bps=1.04, era_early_t=0.17, era_early_n=360,
    era_late_bps=3.75, era_late_t=0.39, era_late_n=443,
    timer_1_gross=2.45, timer_1_cost=2.96, timer_1_net=-0.51, timer_1_t=-0.09,
    timer_5_gross=2.45, timer_5_cost=10.96, timer_5_net=-8.51, timer_5_t=-1.49,
    null_mean_t=0.15, null_sd_t=1.22, null_fire=1,
    planted_t=10.83, planted_welch=1.79,
    fingerprint="357fd262912f",
)


HEADER = f"""# Study 810 — Price Delay ⏳

**Do stocks that price market news *slowly* go on to earn a premium?**

Hou & Moskowitz (2005) argue that a stock into which market-wide information diffuses
**slowly** — one whose return responds to the market with a lag — should command a
**return premium** over a stock that prices the same news promptly. Their **delay**
measure: regress a name's weekly return on the contemporaneous market plus four weekly
lags of the market over a trailing year, and read off how much of the explained variance
the *lagged* terms carry (`delay = 1 − R²_contemp-only / R²_with-lags`). Sort **long
HIGH-delay / short LOW-delay**. We test it on a liquid US cross-section
({R['start']} → {R['end']}, {R['n_names']} names).

*Numbers below are the frozen headline (`docs/results.md`, fingerprint
`{R['fingerprint']}`); the live cells run the fast synthetic control. Survivorship:
current-membership mega-caps — magnitudes are an upper bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Some stocks react to market news the moment it lands; others lag — a small, "
           "neglected, hard-to-arbitrage name may only catch up a week later. Hou & "
           "Moskowitz measure that lag as **price delay**: run the market and four weekly "
           "lags against a name's return, and see how much of the co-movement only shows "
           "up in the *lagged* terms. The thesis: **slow** names are riskier / more "
           "neglected, so they should pay a premium. Buy the laggards, sell the prompt "
           "names."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, long_bps=%r, short_bps=%r, gross_sharpe=%r)\n"
            "print('long high-delay / short low-delay spread: %%+.2f bps/week (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  high-delay book %%+.2f bps vs low-delay book %%+.2f bps'\n"
            "      %% (R['long_bps'], R['short_bps']))\n"
            "print('  gross spread Sharpe (before cost): %%.2f' %% R['gross_sharpe'])"
            % (R["spread_bps"], R["t_nw"], R["long_bps"], R["short_bps"], R["gross_sharpe"])
        ),
        md("## 2. Is the sort just lucky? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world (`knob>0`: high-delay names load on "
           "the *lagged* market **and** earn a premium) and check the detector recovers "
           "it — and that it stays *silent* on the null (`knob=0`: the lag structure is "
           "there but unpriced). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from price_delay import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(knob=0.0, seed=811, n_assets=40, n_days=2000))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(knob=0.0018, seed=810, n_assets=40, n_days=2000))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — the famous edge does *not* show up here\n\n"
           f"On this liquid mega-cap tape the long-high-delay / short-low-delay spread is "
           f"**{R['spread_bps']:+.2f} bps/week** with NW *t* = **{R['t_nw']:+.2f}** — the "
           f"*right sign* (high-delay names did edge out low-delay ones) but statistically "
           f"**indistinguishable from zero**: a column-permutation placebo puts the "
           f"observed value at only p ≈ {R['placebo_p']:.2f}, and it is a coin-flip in both "
           f"halves of the sample (*t* = {R['era_early_t']:+.2f} / {R['era_late_t']:+.2f}). "
           f"The seeded synthetic control recovers a *planted* delay premium cleanly "
           f"(*t* = {R['planted_t']:+.2f}) and stays silent on the null, so the flat "
           f"real-tape result is genuine, not a broken engine — the delay premium is a "
           f"**small / illiquid / neglected-stock** phenomenon that does not survive on 50 "
           f"mega-caps, every one of which is priced in milliseconds. And once you charge "
           f"the weekly round-trip, even the tiny positive gross edge turns **negative** "
           f"(net {R['timer_1_net']:+.2f} bps/week at 1 bp one-way). **Signal: None**, "
           f"**Tradability: Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 810 — Price Delay — the teardown\n\n"
           "The per-leg splits, the Newey-West spread *t*, the pooled Welch book test, the "
           "1,000-permutation placebo, the two-era robustness cut, the costed timer, and "
           "the 20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The headline — long-high-delay / short-low-delay spread\n\n"
           "Weekly equal-weight top-30% minus bottom-30% price-delay spread "
           "(trailing-52-week delay, contemporaneous market + 4 weekly lags)."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/week  NW(6) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : high-delay {R['long_bps']:+.2f} vs low-delay {R['short_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:.2f} (before cost)\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)\n\n"
           "Keep the delay sort, but read each week's forward return from a column-permuted "
           "panel (signal → outcome link broken). If the edge were real the observed spread "
           "would sit far in the right tail; here it does not."),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> p = {R['placebo_p']:.5f}\")"
        ),
        md("## Robustness — two eras (split 2018-01-01)"),
        code(
            "print(f\"2010-2017 (n={R['era_early_n']}): {R['era_early_bps']:+.2f} bps  NW t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2018-2026 (n={R['era_late_n']}): {R['era_late_bps']:+.2f} bps  NW t = {R['era_late_t']:+.2f}\")"
        ),
        md("## The timer — can you get paid for it?\n\n"
           "2 sides × one-way cost × NAV per **week** on the long-short book; short pays 50 bps/yr borrow."),
        code(
            "for tag,g,c,n,t in [('1 bp',R['timer_1_gross'],R['timer_1_cost'],R['timer_1_net'],R['timer_1_t']),\n"
            "                    ('5 bps',R['timer_5_gross'],R['timer_5_cost'],R['timer_5_net'],R['timer_5_t'])]:\n"
            "    print(f\"{tag:>5} one-way: gross {g:+.2f} -> net {n:+.2f} bps/week (cost {c:.2f}/week, t={t:+.2f})\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted premium."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from price_delay import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(knob=0.0, seed=811+s, n_assets=40, n_days=2000))['t_nw'] for s in range(8)])\n"
            "print(f\"null (knob=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(knob=0.0018, seed=810, n_assets=40, n_days=2000))\n"
            "print(f\"planted (knob=0.0018): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The claimed Hou-Moskowitz delay premium does **not** "
           f"replicate on 50 liquid US mega-caps: the long-high-delay / short-low-delay "
           f"spread is **{R['spread_bps']:+.2f} bps/week** (NW *t* = **{R['t_nw']:+.2f}**) — "
           f"the right sign but a coin-flip (placebo p ≈ {R['placebo_p']:.2f}), flat in both "
           f"eras (*t* = {R['era_early_t']:+.2f} / {R['era_late_t']:+.2f}). The 20-seed "
           f"synthetic control recovers a *planted* premium cleanly (*t* = "
           f"{R['planted_t']:+.2f}, fires on ~{R['null_fire']}/20 nulls, the expected 5% "
           f"false-positive rate), so the null result is real, not machinery. The delay "
           f"premium is a small / illiquid / neglected-stock effect; mega-caps are exactly "
           f"where it should not appear. Survivorship biases the magnitude upward.\n"
           f"- **Tradability — Mirage.** Even the tiny positive gross edge dies on contact "
           f"with costs: at 1 bp one-way the weekly friction ({R['timer_1_cost']:.2f} "
           f"bps/week) already exceeds the {R['spread_bps']:.2f} bps gross, net "
           f"**{R['timer_1_net']:+.2f} bps/week** (*t* = {R['timer_1_t']:+.2f}); at 5 bps "
           f"**{R['timer_5_net']:+.2f} bps/week**."),
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
