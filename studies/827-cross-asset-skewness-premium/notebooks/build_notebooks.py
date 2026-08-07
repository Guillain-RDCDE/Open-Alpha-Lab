"""Generate the two narrative notebooks for Study 827 (Cross-Asset Skewness Premium).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily total-return
# closes, nine asset-class ETFs, 2007-01-03 -> 2026-06-30; trailing-126d realized skew,
# monthly long bottom-1/3 / short top-1/3).
R = dict(
    start="2007-01-03", end="2026-06-30", n_classes=9, n_months=227, median_n=9,
    fingerprint="9ce7d7c0e243",
    spread_bps=13.73, t_nw=0.62, t_1s=0.64,
    lo_bps=63.24, hi_bps=49.51, welch_t=0.39, sharpe=0.15,
    placebo_obs=13.73, placebo_mean=-0.421, placebo_sd=17.101,
    placebo_p=0.2040, placebo_sigma=0.83, placebo_draws=1000,
    era_early_bps=-15.21, era_early_t=-0.45, era_early_n=107,
    era_late_bps=39.54, era_late_t=1.39, era_late_n=120,
    w63_bps=22.06, w63_t=1.04, w252_bps=5.95, w252_t=0.25,
    timer_1_gross=13.73, timer_1_cost=6.17, timer_1_net=7.57, timer_1_t=0.35, timer_1_ann=0.9,
    timer_5_gross=13.73, timer_5_cost=14.17, timer_5_net=-0.43, timer_5_t=-0.02, timer_5_ann=-0.1,
    null_mean_t=0.01, null_sd_t=1.22, null_fire=2, planted_t=2.52, planted_welch=2.87,
)


HEADER = f"""# Study 827 — Cross-Asset Skewness Premium 🎲🌐

**Does the single-name "lottery names under-earn" effect carry up to whole *asset classes*?**

The single-name realized-skewness reversal (Amaya, Christoffersen, Jacobs & Vasquez 2015;
Study 803) says the most right-skewed **stocks** go on to earn *less*. Here we ask its
asset-class analogue: measure each of nine asset-class ETFs' **trailing realized skewness**,
each month go **long the low-skew / short the high-skew** classes, and see whether low-skew
classes out-earn. Real tape {R['start']} → {R['end']}, {R['n_classes']} classes.

*Numbers below are the frozen headline (`docs/results.md`, fingerprint `{R['fingerprint']}`); the
live cells run the fast synthetic control. Survivorship: fixed current-membership class-proxy
ETFs — milder than a single-name universe, named on the Signal axis.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "In the stock cross-section, a right-skewed name has a fat *upside* tail — the "
           "occasional big up-day — and lottery-loving investors overpay for it, so it "
           "under-earns. Do investors do the same *across* asset classes: bidding up "
           "whichever class (a commodity spike, an EM melt-up) recently looked most "
           "lottery-like? Sort the nine classes on their trailing third moment; buy the "
           "boring low-skew ones, sell the lottery-like high-skew ones."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, lo_bps=%r, hi_bps=%r, sharpe=%r)\n"
            "print('long low-skew / short high-skew spread: %%+.2f bps/month (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  low-skew book %%+.2f bps vs high-skew book %%+.2f bps'\n"
            "      %% (R['lo_bps'], R['hi_bps']))\n"
            "print('  gross spread Sharpe (before cost, annualised): %%.2f' %% R['sharpe'])"
            % (R["spread_bps"], R["t_nw"], R["lo_bps"], R["hi_bps"], R["sharpe"])
        ),
        md("## 2. Is the sort just lucky? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world of nine synthetic classes (`edge>0`) "
           "and check the detector recovers it — and that it stays *silent* on the null "
           "(`edge=0`, skew present but unpriced). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from cross_asset_skew import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=827, n_assets=9, n_days=3000))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.004, seed=827, n_assets=9, n_days=4000))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up above |t|=2)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — the premium does *not* carry to asset classes\n\n"
           f"On the nine-class tape the long-low-skew / short-high-skew spread is "
           f"**{R['spread_bps']:+.2f} bps/month** with NW *t* = **{R['t_nw']:+.2f}** — the *sign* "
           f"is in the claimed direction (low-skew classes did edge out high-skew ones), but the "
           f"magnitude is **statistically zero**. A 1,000-permutation placebo puts it only "
           f"~{R['placebo_sigma']:.1f}σ into the right tail (p = {R['placebo_p']:.2f}), and the spread "
           f"even *flips sign* across the two eras ({R['era_early_bps']:+.0f} bps early, "
           f"{R['era_late_bps']:+.0f} bps late). With just nine classes there is too little skew "
           f"dispersion for the effect to exist. The seeded synthetic control shows a real premium "
           f"of plausible size *would* have fired, so this is an honest null, not a broken sort. "
           f"**Signal: None**, **Tradability: Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 827 — Cross-Asset Skewness Premium — the teardown\n\n"
           "The per-leg books, the Newey-West spread *t*, the pooled Welch book test, the "
           "1,000-permutation asset-label placebo, the two-era and multi-window robustness cut, "
           "the costed monthly timer, and the 20-seed synthetic control."),
        code("R = %r" % (R,)),
        md("## The headline — long-low-skew / short-high-skew spread\n\n"
           "Monthly equal-weight bottom-⅓ minus top-⅓ realized-skew spread across nine classes "
           f"(n = {R['n_months']} months)."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/month  NW(6) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : low-skew {R['lo_bps']:+.2f} vs high-skew {R['hi_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['sharpe']:.2f} (before cost, annualised)\")"
        ),
        md("## Placebo — asset-label-permute the forward returns (1,000 permutations)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> p = {R['placebo_p']:.4f}  (~{R['placebo_sigma']:+.2f} sigma)\")"
        ),
        md("## Robustness — two eras (split 2016-07-01) and the window sweep"),
        code(
            "print(f\"2007-2016 (n={R['era_early_n']}): {R['era_early_bps']:+.2f} bps  NW t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2016-2026 (n={R['era_late_n']}): {R['era_late_bps']:+.2f} bps  NW t = {R['era_late_t']:+.2f}\")\n"
            "print(f\" 63d window : {R['w63_bps']:+.2f} bps  NW t = {R['w63_t']:+.2f}\")\n"
            "print(f\"252d window : {R['w252_bps']:+.2f} bps  NW t = {R['w252_t']:+.2f}\")\n"
            "print('  -> sign flips across eras; never clears |t|=2 at any lookback')"
        ),
        md("## The timer — can you get paid for it?\n\n"
           "2 sides × one-way cost × NAV per monthly rebalance; short pays 50 bps/yr borrow."),
        code(
            "for tag,g,c,n,t in [('1 bp',R['timer_1_gross'],R['timer_1_cost'],R['timer_1_net'],R['timer_1_t']),\n"
            "                    ('5 bps',R['timer_5_gross'],R['timer_5_cost'],R['timer_5_net'],R['timer_5_t'])]:\n"
            "    print(f\"{tag:>5} one-way: gross {g:+.2f} -> net {n:+.2f} bps/mo (cost {c:.2f}/mo, t={t:+.2f})\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted relation "
           "even on a nine-asset cross-section."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from cross_asset_skew import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=827+s, n_assets=9, n_days=3000))['t_nw'] for s in range(20)])\n"
            "print(f\"null (edge=0), 20 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/20\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.004, seed=827, n_assets=9, n_days=4000))\n"
            "print(f\"planted (edge=0.004): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The single-name realized-skewness reversal does **not** carry up "
           f"to the asset-class level. The long-low-skew / short-high-skew spread across nine class "
           f"ETFs is **{R['spread_bps']:+.2f} bps/month** (NW *t* = **{R['t_nw']:+.2f}**) — right-signed "
           f"but statistically zero, ~{R['placebo_sigma']:.1f}σ into the placebo right tail "
           f"(p = {R['placebo_p']:.2f}), flipping sign across the two eras "
           f"(*t* = {R['era_early_t']:+.2f} / {R['era_late_t']:+.2f}) and insignificant at every "
           f"lookback. The 20-seed synthetic control fires on a *planted* relation "
           f"(*t* = {R['planted_t']:+.2f}) and stays silent on the null ({R['null_fire']}/20), so the "
           f"flat real result is a genuine absence of edge, not machinery.\n"
           f"- **Tradability — Mirage.** The gross spread is already insignificant (Sharpe "
           f"{R['sharpe']:.2f}); a token 5 bps one-way cost erases it to zero "
           f"(net **{R['timer_1_net']:+.2f} bps/mo** at 1 bp, {R['timer_5_net']:+.2f} at 5 bps). "
           f"Nothing survives friction because there was nothing there."),
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
