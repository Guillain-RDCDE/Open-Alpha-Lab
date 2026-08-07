"""Generate the two narrative notebooks for Study 816 (Drawdown Duration).

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
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; trailing-252d
# time-underwater sort, long top30% [high UW] / short bottom30% [low UW]).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_rows=4147, n_days=3895,
    fingerprint="357fd262912f",
    spread_bps=-1.12, t_nw=-0.80, t_1s=-0.81,
    hi_bps=6.62, lo_bps=7.75, welch_t=-0.41, gross_sharpe=-0.20,
    placebo_obs=-1.12, placebo_mean=-0.037, placebo_sd=1.007,
    placebo_p=0.261, placebo_sigma=1.12, placebo_draws=1000,
    era_early_bps=0.50, era_early_t=0.28, era_early_n=1761,
    era_late_bps=-2.46, era_late_t=-1.17, era_late_n=2134,
    timer_1_gross=-1.12, timer_1_cost=2.14, timer_1_net=-3.26, timer_1_t=-2.34,
    timer_5_gross=-1.12, timer_5_cost=10.14, timer_5_net=-11.26, timer_5_t=-8.07,
    null_mean_t=0.10, null_sd_t=1.15, null_fire=2,
    planted_t=-15.31, planted_welch=-14.44,
)


HEADER = f"""# Study 816 — Drawdown Duration ⏱️📉

**Does how *long* a name spent underwater over the past year predict its future return?**

A drawdown has two moments: its **depth** (how far a name fell) and its **duration** (how
long it stayed down). We take the duration side as **time-underwater** — the fraction of
the trailing year a name's cumulative total return sat **below its running high-water
mark** — and ask whether the market **pays** for bearing persistent-drawdown names (they
rebound), or whether they simply **keep sinking**. Liquid US cross-section
({R['start']} → {R['end']}, {R['n_names']} names). Honest sign.

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: current-membership mega-caps — the names that stayed
underwater and died are absent, so any "losers keep sinking" tilt is understated.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Track a name's cumulative total return and its **high-water mark** (the highest "
           "level it has reached so far). Whenever the curve is *below* that mark, the name "
           "is **underwater**. Add up the underwater days over the last year and divide by "
           "the year: that fraction is **time-underwater**. A name always making fresh highs "
           "is near 0; a persistent laggard is near 1. Does that persistent-drawdown risk "
           "earn a premium — or keep sinking?"),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, hi_bps=%r, lo_bps=%r, gross_sharpe=%r)\n"
            "print('long high-underwater / short low-underwater spread: %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  high-underwater book %%+.2f bps vs low-underwater book %%+.2f bps'\n"
            "      %% (R['hi_bps'], R['lo_bps']))\n"
            "print('  gross spread Sharpe (before cost): %%.2f' %% R['gross_sharpe'])"
            % (R["spread_bps"], R["t_nw"], R["hi_bps"], R["lo_bps"], R["gross_sharpe"])
        ),
        md("## 2. A tiny time-underwater example\n\n"
           "A price that rises to a peak then drifts below it is underwater exactly on the "
           "days strictly beneath its running high-water mark. This is the whole signal, "
           "vectorised: `cumprod` → `cummax` → `curve < hwm` → rolling mean."),
        code(
            "px = np.array([100,101,102,103,104,103,102,101,100,99], float)\n"
            "curve = px / px[0]\n"
            "hwm = np.maximum.accumulate(curve)\n"
            "uw = (curve < hwm).astype(int)\n"
            "print('underwater flag per day:', list(uw))\n"
            "print('time-underwater over these 10 days: %.0f%%' % (100*uw.mean()))"
        ),
        md("## 3. Is the sort just noise? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world (`knob>0`: low-drift names stay "
           "underwater and keep sinking) and check the detector recovers it — and that it "
           "stays *silent* on the null (`knob=0`, time-underwater present but unpriced). "
           "No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from drawdown_duration import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(knob=0.0, seed=816, n_assets=40, n_days=1200))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(knob=0.0010, seed=816, n_assets=40, n_days=1500))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up, negative)' % planted['t_nw'])"
        ),
        md("## 4. The honest verdict — no signal, no paycheck\n\n"
           f"On this liquid mega-cap tape the long-high-underwater / short-low-underwater spread "
           f"is **{R['spread_bps']:+.2f} bps/day** with NW *t* = **{R['t_nw']:+.2f}** — a "
           f"statistical **zero**. It sits ≈{R['placebo_sigma']:.2f}σ from a 1,000-permutation "
           f"null (two-sided p = {R['placebo_p']:.2f}), and its sign even *flips* between eras "
           f"(early *t* = {R['era_early_t']:+.2f}, late *t* = {R['era_late_t']:+.2f}). The market "
           "neither pays a persistent-drawdown premium nor keeps sinking the underwater names. "
           "The seeded synthetic control recovers a *planted* relation cleanly, so this flatness "
           "is a genuine absence of signal, not a broken sort. **Signal: None**, "
           "**Tradability: Mirage** (the book loses to costs either way)."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 816 — Drawdown Duration — the teardown\n\n"
           "The per-leg splits, the Newey-West spread *t*, the pooled Welch book test, the "
           "1,000-permutation two-sided placebo, the two-era robustness cut, the costed timer, "
           "and the 20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The headline — long-high-underwater / short-low-underwater spread\n\n"
           "Daily equal-weight top-30% (high time-underwater) minus bottom-30% (low) spread, "
           f"n = {R['n_days']:,} days, as-of {R['end']}, fingerprint `{R['fingerprint']}`."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : high-UW {R['hi_bps']:+.2f} vs low-UW {R['lo_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:.2f} (before cost)\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations, two-sided)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> {R['placebo_sigma']:.2f} sigma, two-sided p = {R['placebo_p']:.3f}\")"
        ),
        md("## Robustness — two eras (split 2018-01-01)"),
        code(
            "print(f\"2010-2017 (n={R['era_early_n']}): {R['era_early_bps']:+.2f} bps  NW t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2018-2026 (n={R['era_late_n']}): {R['era_late_bps']:+.2f} bps  NW t = {R['era_late_t']:+.2f}\")\n"
            "print('sign flips across eras and neither half is significant -> a textbook null')"
        ),
        md("## The timer — can you get paid for it?\n\n"
           "2 sides × one-way cost × NAV per day on the long-short book; short pays 50 bps/yr borrow."),
        code(
            "for tag,g,c,n,t in [('1 bp',R['timer_1_gross'],R['timer_1_cost'],R['timer_1_net'],R['timer_1_t']),\n"
            "                    ('5 bps',R['timer_5_gross'],R['timer_5_cost'],R['timer_5_net'],R['timer_5_t'])]:\n"
            "    print(f\"{tag:>5} one-way: gross {g:+.2f} -> net {n:+.2f} bps/day (cost {c:.2f}/day, t={t:+.2f})\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted relation "
           "(low-drift names stay underwater and keep sinking -> a *negative* high-minus-low spread)."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from drawdown_duration import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(knob=0.0, seed=816+s, n_assets=40, n_days=1200))['t_nw'] for s in range(8)])\n"
            "print(f\"null (knob=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(knob=0.0010, seed=816, n_assets=40, n_days=1500))\n"
            "print(f\"planted (knob=0.0010): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The fraction of the trailing year a mega-cap spent underwater does"
           f" **not** predict its forward return: the long-high-underwater / short-low-underwater"
           f" spread is **{R['spread_bps']:+.2f} bps/day** (NW *t* = **{R['t_nw']:+.2f}**), "
           f"≈{R['placebo_sigma']:.2f}σ from a {R['placebo_draws']:,}-permutation null "
           f"(two-sided p = {R['placebo_p']:.2f}), with a sign that *flips* between eras "
           f"(*t* = {R['era_early_t']:+.2f} / {R['era_late_t']:+.2f}). The 20-seed synthetic control"
           f" recovers a *planted* relation cleanly (*t* = {R['planted_t']:+.2f}, fires on "
           f"{R['null_fire']}/20 nulls), so the flat real-tape result is a genuine absence of signal."
           f" Survivorship understates any 'losers keep sinking' tilt.\n"
           f"- **Tradability — Mirage.** The book loses gross ({R['spread_bps']:+.2f} bps/day) and net"
           f" (**{R['timer_1_net']:+.2f} bps/day** at 1 bp, {R['timer_5_net']:+.2f} at 5 bps); the "
           f"{R['timer_1_cost']:.2f} bps/day round-trip friction eats even the sign-flip at a mere 1 bp."),
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
