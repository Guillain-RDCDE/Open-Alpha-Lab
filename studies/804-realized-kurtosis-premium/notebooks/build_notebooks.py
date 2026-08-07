"""Generate the two narrative notebooks for Study 804 (Realized-Kurtosis Premium).

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
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; trailing-21d realized
# kurtosis sort, long top30% / short bottom30%).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_days=4125,
    spread_bps=1.78, t_nw=1.79, t_1s=1.76,
    hi_bps=8.76, lo_bps=6.97, welch_t=0.69, gross_sharpe=0.43,
    placebo_obs=1.78, placebo_mean=-0.004, placebo_sd=0.825,
    placebo_p=0.01600, placebo_sigma_right=2.16, placebo_draws=1000,
    era_early_bps=0.13, era_early_t=0.11, era_early_n=1991,
    era_late_bps=3.33, era_late_t=2.12, era_late_n=2134,
    timer_1_gross=1.78, timer_1_cost=2.14, timer_1_net=-0.35, timer_1_t=-0.35,
    timer_5_gross=1.78, timer_5_cost=10.14, timer_5_net=-8.35, timer_5_t=-8.23,
    null_mean_t=0.35, null_sd_t=0.90, null_fire=0,
    planted_t=12.81, planted_welch=15.36,
)


HEADER = f"""# Study 804 — Realized-Kurtosis Premium 🎲📊

**Do stocks with fat-tailed recent returns earn a cross-sectional premium?**

Amaya, Christoffersen, Jacobs & Vasquez (2015) — the paper famous for the negative
realized-**skewness** relation — also tests realized **kurtosis** (a name's recent
fat-tailedness, the fourth moment) and finds it a **weak / ambiguous** predictor, mostly
subsumed by skewness and volatility. We take the self-contained daily version on a liquid
US cross-section ({R['start']} → {R['end']}, {R['n_names']} names) and sort **long the
high-kurt / short the low-kurt** names.

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: current-membership mega-caps — magnitudes are an upper
bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "**Kurtosis** is how *fat-tailed* a return distribution is — how often a stock "
           "throws a big move (either direction) versus a calm one. A high-kurtosis name "
           "lives in fits and starts; a low-kurtosis name grinds. If investors dislike (or "
           "love) fat tails, kurtosis might be priced. But kurtosis is *symmetric* — it "
           "mixes up-tail and down-tail — so whatever premium a tail carries is mostly "
           "already captured by **skewness** (which side the tail is on) and **volatility** "
           "(how big it is). That is why the paper flags it as the *weak* one."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, hi_bps=%r, lo_bps=%r, gross_sharpe=%r)\n"
            "print('long high-kurt / short low-kurt spread: %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  high-kurt book %%+.2f bps vs low-kurt book %%+.2f bps'\n"
            "      %% (R['hi_bps'], R['lo_bps']))\n"
            "print('  gross spread Sharpe (before cost): %%.2f' %% R['gross_sharpe'])"
            % (R["spread_bps"], R["t_nw"], R["hi_bps"], R["lo_bps"], R["gross_sharpe"])
        ),
        md("## 2. Is the sort just lucky? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world (`edge>0`, fat-tailed names earn more) "
           "and check the detector recovers it — and that it stays *silent* on the null "
           "(`edge=0`, kurtosis present but unpriced). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from realized_kurtosis import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=804, n_assets=40, n_days=1500))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.010, seed=804, n_assets=40, n_days=1500))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — the edge is *weak*, exactly as the paper says\n\n"
           f"On this liquid mega-cap tape the long-high-kurt / short-low-kurt spread is "
           f"**{R['spread_bps']:+.2f} bps/day** with NW *t* = **{R['t_nw']:+.2f}** — the "
           f"*right sign* (high-kurt names did edge out low-kurt ones), but **below the "
           f"|t| ≥ 2 bar** the desk requires to call an edge real. The pooled book Welch *t* "
           f"is a limp **{R['welch_t']:+.2f}**, and the whole (weak) effect lives in the "
           f"second half of the sample — it is a flat **zero** in 2010–2017 (*t* = "
           f"{R['era_early_t']:+.2f}). The seeded synthetic control recovers a *planted* "
           f"relation cleanly and never fires on the null, so this is a faithful weak "
           f"measurement, not a bug. **Signal: Weak** (right-signed but sub-threshold), "
           f"**Tradability: Mirage** — the {R['spread_bps']:+.2f} bps gross edge is smaller "
           f"than the {R['timer_1_cost']:.2f} bps/day round-trip cost at even 1 bp one-way."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 804 — Realized-Kurtosis Premium — the teardown\n\n"
           "The per-leg splits, the Newey-West spread *t*, the pooled Welch book test, "
           "the 1,000-permutation placebo, the two-era robustness cut, the costed timer, "
           "and the 20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The headline — long-high-kurt / short-low-kurt spread\n\n"
           "Daily equal-weight top-30% minus bottom-30% realized-kurtosis spread."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : high-kurt {R['hi_bps']:+.2f} vs low-kurt {R['lo_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:.2f} (before cost)\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)\n\n"
           "Note the tension: the placebo puts the spread ~2.2σ into the right tail (p≈0.016), "
           "but it ignores the serial correlation of the overlapping-window signal — which the "
           "Newey-West *t* (+1.79) does account for. The robust read is the sub-threshold one."),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> right-tail p = {R['placebo_p']:.5f} \"\n"
            "      f\"(~{R['placebo_sigma_right']:+.2f} sigma)\")"
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
            "from realized_kurtosis import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=804+s, n_assets=40, n_days=1500))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.010, seed=804, n_assets=40, n_days=1500))\n"
            "print(f\"planted (edge=0.010): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** The realized-kurtosis premium is, as the source paper itself"
           f" reports, the **weak / ambiguous** sibling of realized skewness. On 50 liquid US"
           f" mega-caps the long-high-kurt / short-low-kurt spread is **{R['spread_bps']:+.2f}"
           f" bps/day** — the *right sign*, but NW *t* = **{R['t_nw']:+.2f}** (below the |t| ≥ 2"
           f" bar), the book Welch *t* is {R['welch_t']:+.2f}, and it is a flat zero in 2010–2017"
           f" (*t* = {R['era_early_t']:+.2f}), surfacing only marginally in 2018–2026 (*t* ="
           f" {R['era_late_t']:+.2f}). The 20-seed synthetic control fires on {R['null_fire']}/20"
           f" nulls and recovers a planted relation cleanly (*t* = {R['planted_t']:+.2f}), so this"
           f" is a faithful weak measurement. Survivorship biases the magnitude.\n"
           f"- **Tradability — Mirage.** The {R['spread_bps']:+.2f} bps/day gross edge is already"
           f" smaller than the {R['timer_1_cost']:.2f} bps/day round-trip friction at 1 bp one-way"
           f" (net **{R['timer_1_net']:+.2f} bps/day**, *t* = {R['timer_1_t']:+.2f}); at 5 bps"
           f" **{R['timer_5_net']:+.2f} bps/day**."),
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
