"""Generate the two narrative notebooks for Study 812 (Corwin-Schultz Spread).

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
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; trailing-21d Corwin-
# Schultz spread sort, long top30% (illiquid) / short bottom30% (liquid)).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_days=4125,
    fingerprint="357fd262912f", median_spread_bps=12.7,
    spread_bps=4.45, t_nw=3.24, t_1s=3.06,
    long_bps=10.64, short_bps=6.19, welch_t=1.62, gross_sharpe=0.76,
    placebo_obs=4.45, placebo_mean=-0.019, placebo_sd=0.916,
    placebo_p=0.00000, placebo_sigma=4.88, placebo_draws=1000,
    era_early_bps=3.48, era_early_t=2.03, era_early_n=1991,
    era_late_bps=5.35, era_late_t=2.52, era_late_n=2134,
    timer_1_gross=4.45, timer_1_cost=2.14, timer_1_net=2.31, timer_1_t=1.59,
    timer_1_sharpe=0.39, timer_1_ann=5.8,
    timer_5_gross=4.45, timer_5_cost=10.14, timer_5_net=-5.69, timer_5_t=-3.91,
    timer_5_sharpe=-0.97, timer_5_ann=-14.3,
    null_mean_t=-0.17, null_sd_t=0.79, null_fire=1,
    planted_t=10.09, planted_welch=10.22,
)


HEADER = f"""# Study 812 — Corwin-Schultz Spread 📏

**Can you read a stock's bid-ask spread off its daily high and low — and does the illiquid
name pay a premium?**

Corwin & Schultz (2012) show the daily **high** transacts near the *ask* and the **low**
near the *bid*, so the high-low range hides the spread; comparing single-day ranges with
the two-day range isolates it. A high estimated spread proxies **illiquidity**, and
illiquid assets should earn more (Amihud-Mendelson). We take the self-contained daily
version on a liquid US cross-section ({R['start']} → {R['end']}, {R['n_names']} names) and
sort **long high-spread / short low-spread**.

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: current-membership mega-caps — magnitudes are an upper
bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Every day, the printed **high** is about where someone paid the *ask* and the "
           "**low** is about where someone hit the *bid* — so the daily high-low range is "
           "inflated by the spread. But price *variance* grows with time while the spread "
           "does not, so contrasting a single day's range with a two-day range lets you "
           "back out the spread. High spread = illiquid name; illiquid names should be "
           "cheaper and earn more. Buy the illiquid, sell the liquid."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, long_bps=%r, short_bps=%r, gross_sharpe=%r, median_spread_bps=%r)\n"
            "print('median daily CS spread across mega-caps: ~%%.1f bps (a sane effective spread)'\n"
            "      %% R['median_spread_bps'])\n"
            "print('long high-spread / short low-spread: %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  illiquid book %%+.2f bps vs liquid book %%+.2f bps'\n"
            "      %% (R['long_bps'], R['short_bps']))\n"
            "print('  gross spread Sharpe (before cost): %%.2f' %% R['gross_sharpe'])"
            % (R["spread_bps"], R["t_nw"], R["long_bps"], R["short_bps"],
               R["gross_sharpe"], R["median_spread_bps"])
        ),
        md("## 2. Is the sort just lucky? A live synthetic control\n\n"
           "We plant the premium in a seeded toy world (`edge>0`): each name gets a "
           "persistent spread that both widens its high-low range *and* lifts its return. "
           "The detector must recover it — and stay *silent* on the null (`edge=0`, spreads "
           "present but unpriced). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from corwin_schultz import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=812, n_assets=40, n_days=1200))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.08, seed=812, n_assets=40, n_days=1500))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — the premium is *real*, the paycheck is *fragile*\n\n"
           f"On this liquid mega-cap tape the long-high-spread / short-low-spread book earns "
           f"**{R['spread_bps']:+.2f} bps/day** with NW *t* = **{R['t_nw']:+.2f}** — the "
           f"illiquid names genuinely out-earned the liquid ones, the correct sign, and it "
           f"holds in both halves of the sample (*t* = {R['era_early_t']:+.2f} / "
           f"{R['era_late_t']:+.2f}) and sits {R['placebo_sigma']:+.2f}σ into the right tail of "
           f"a 1,000-permutation placebo. A **rare green** for this desk. But the catch: at an "
           f"idealised 1 bp one-way the net is still positive (**{R['timer_1_net']:+.2f} "
           f"bps/day**) yet no longer significant (*t* = {R['timer_1_t']:+.2f}), and at a "
           f"realistic 5 bps it goes to **{R['timer_5_net']:+.2f} bps/day**. The long leg *is* "
           f"the illiquid names — where real spreads are widest — so you pay the premium to "
           f"collect it. **Signal: Real**, **Tradability: Fragile**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 812 — Corwin-Schultz Spread — the teardown\n\n"
           "The per-leg books, the Newey-West spread *t*, the pooled Welch book test, "
           "the 1,000-permutation placebo, the two-era robustness cut, the costed timer, "
           "and the 20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The estimator itself\n\n"
           "Daily `S = 2(e^α−1)/(1+e^α)` from consecutive 2-day highs/lows (negatives "
           "floored at 0), averaged over a trailing month. The median mega-cap estimate is a "
           "sensible effective spread — a sanity check before the sort."),
        code(
            "print(f\"median daily CS spread across names : ~{R['median_spread_bps']:.1f} bps\")\n"
            "print(f\"panel Close fingerprint             : {R['fingerprint']}  (as-of {R['end']})\")"
        ),
        md("## The headline — long-high-spread / short-low-spread\n\n"
           "Daily equal-weight top-30% (illiquid) minus bottom-30% (liquid) estimated-spread sort."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : high-spread {R['long_bps']:+.2f} vs low-spread {R['short_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:.2f} (before cost)\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> p = {R['placebo_p']:.5f}\")\n"
            "print(f\"observed is {R['placebo_sigma']:+.2f} sigma into the RIGHT tail of the null\")"
        ),
        md("## Robustness — two eras (split 2018-01-01)"),
        code(
            "print(f\"2010-2017 (n={R['era_early_n']}): {R['era_early_bps']:+.2f} bps  NW t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2018-2026 (n={R['era_late_n']}): {R['era_late_bps']:+.2f} bps  NW t = {R['era_late_t']:+.2f}\")"
        ),
        md("## The timer — can you get paid for it?\n\n"
           "2 sides × one-way cost × NAV per day on the long-short book; short pays 50 bps/yr "
           "borrow. (The flat charge is *generous* — the illiquid long leg is where real "
           "spreads are widest.)"),
        code(
            "for tag,g,c,n,t,sh,an in [\n"
            "    ('1 bp',R['timer_1_gross'],R['timer_1_cost'],R['timer_1_net'],R['timer_1_t'],R['timer_1_sharpe'],R['timer_1_ann']),\n"
            "    ('5 bps',R['timer_5_gross'],R['timer_5_cost'],R['timer_5_net'],R['timer_5_t'],R['timer_5_sharpe'],R['timer_5_ann'])]:\n"
            "    print(f\"{tag:>5} one-way: gross {g:+.2f} -> net {n:+.2f} bps/day \"\n"
            "          f\"(cost {c:.2f}/day, t={t:+.2f}, Sharpe {sh:+.2f}, ~{an:+.1f}%/yr)\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted premium."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from corwin_schultz import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=812+s, n_assets=40, n_days=1200))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.08, seed=812, n_assets=40, n_days=1500))\n"
            "print(f\"planted (edge=0.08): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Real.** The Corwin-Schultz illiquidity premium **replicates with the "
           f"correct sign** on 50 liquid US mega-caps: long high-spread / short low-spread is "
           f"**{R['spread_bps']:+.2f} bps/day** (NW *t* = **{R['t_nw']:+.2f}**), significant in "
           f"both eras (*t* = {R['era_early_t']:+.2f} / {R['era_late_t']:+.2f}), "
           f"{R['placebo_sigma']:+.2f}σ into the right tail of a 1,000-permutation placebo. The "
           f"20-seed synthetic control recovers a *planted* premium cleanly (*t* = "
           f"{R['planted_t']:+.2f}, fires on {R['null_fire']}/20 nulls). A rare green — it "
           f"survives even where an illiquidity effect should be weakest. Survivorship biases "
           f"the magnitude upward.\n"
           f"- **Tradability — Fragile.** The gross premium is real but lives inside its own "
           f"cost band: at 1 bp one-way net **{R['timer_1_net']:+.2f} bps/day** but *t* only "
           f"{R['timer_1_t']:+.2f}; at 5 bps **{R['timer_5_net']:+.2f} bps/day** (*t* = "
           f"{R['timer_5_t']:+.2f}). The long leg is the illiquid names — you pay the very "
           f"spread the premium is compensating."),
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
