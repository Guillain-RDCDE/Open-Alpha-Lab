"""Generate the two narrative notebooks for Study 872 (Nominal-Price Illusion).

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
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; nominal price-level
# sort, long cheapest30% / short priciest30%; fingerprint 357fd262912f).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_days=4146,
    fingerprint="357fd262912f",
    cheap_name="T", cheap_val=20, dear_name="CAT", dear_val=1063,
    spread_bps=2.92, t_nw=3.01, t_1s=2.83,
    lo_bps=8.75, hi_bps=5.82, welch_t=1.14,
    lo_vol=19.2, lo_skew=-0.26, lo_sharpe=1.15,
    hi_vol=17.8, hi_skew=-0.44, hi_sharpe=0.82,
    placebo_obs=2.92, placebo_mean=0.019, placebo_sd=1.218,
    placebo_p=0.0140, placebo_sigma=2.4, placebo_draws=1000,
    era_early_bps=2.66, era_early_t=2.11, era_early_n=2012,
    era_late_bps=3.18, era_late_t=2.16, era_late_n=2134,
    timer_1_gross=2.92, timer_1_cost=2.14, timer_1_net=0.79, timer_1_t=0.76,
    timer_5_gross=2.92, timer_5_cost=10.14, timer_5_net=-7.21, timer_5_t=-6.98,
    null_mean_t=-0.15, null_sd_t=0.92, null_fire=0,
    planted_t=-3.69, planted_welch=-4.01,
    planted_lo_vol=19.8, planted_lo_skew=0.89, planted_hi_vol=6.3, planted_hi_skew=0.33,
)


HEADER = f"""# Study 872 — Nominal-Price Illusion 🪙

**A $10 stock and a $500 stock — is the cheap one an over-priced lottery ticket?**

Kumar (2009) and Birru & Wang (2016) argue the raw **nominal share price** is a pure
*money illusion*: value = price × shares, so the dollar price of one share carries **no**
information about a firm — yet retail lottery demand piles into low-priced names, and
investors over-estimate how much a cheap share can "grow". If that demand over-prices
cheap-looking stocks, low-priced names should carry the lottery look (more volatility,
more right-skew) and **lower risk-adjusted returns**. We sort a liquid US cross-section on
its nominal price level ({R['start']} → {R['end']}, {R['n_names']} names).

*Numbers below are the frozen headline (`docs/results.md`, fingerprint
`{R['fingerprint']}`); the live cells run the fast synthetic control. Survivorship + proxy:
current-membership mega-caps are rarely cheap, and the Close is split-back-adjusted — an
honest nominal-price proxy. Honest low power.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Two firms worth the same can trade at $10 or $500 a share — the number is set "
           "by an arbitrary share count, nothing real. But a $10 share *feels* cheaper and "
           "*feels* like it has more room to run, so lottery-hunting retail money crowds "
           "into low-priced names. If they over-pay, the cheap names should be **over-priced "
           "lotteries**: more volatile, more right-skewed, and — the payoff prediction — "
           "**lower risk-adjusted returns**. Sort on the price level; short the cheap, buy "
           "the dear."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, lo_bps=%r, hi_bps=%r, lo_sharpe=%r, hi_sharpe=%r,\n"
            "         cheap_name=%r, cheap_val=%r, dear_name=%r, dear_val=%r)\n"
            "print('price range at as-of: cheapest %%s $%%d .. priciest %%s $%%d (no single-digit names)'\n"
            "      %% (R['cheap_name'], R['cheap_val'], R['dear_name'], R['dear_val']))\n"
            "print('long cheap / short dear spread: %%+.2f bps/day (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  cheap book %%+.2f bps vs dear book %%+.2f bps' %% (R['lo_bps'], R['hi_bps']))\n"
            "print('  cheap Sharpe %%+.2f vs dear Sharpe %%+.2f' %% (R['lo_sharpe'], R['hi_sharpe']))"
            % (R["spread_bps"], R["t_nw"], R["lo_bps"], R["hi_bps"], R["lo_sharpe"],
               R["hi_sharpe"], R["cheap_name"], R["cheap_val"], R["dear_name"], R["dear_val"])
        ),
        md("## 2. Is the sort just lucky? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world (`edge>0`: cheap names look "
           "lottery-like *and* under-earn) and check the detector recovers it — and that it "
           "stays *silent* on the null (`edge=0`: cheap names still look lottery-like but "
           "the price predicts nothing about returns). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from nominal_price import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=872, n_assets=40, n_days=1200))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0016, seed=872, n_assets=40, n_days=1500))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should be NEGATIVE: cheap under-earns)' % planted['t_nw'])\n"
            "print('planted lottery look: cheap vol %.1f%% skew %+.2f  vs  dear vol %.1f%% skew %+.2f'\n"
            "      % (planted['lo_vol']*100, planted['lo_skew'], planted['hi_vol']*100, planted['hi_skew']))"
        ),
        md("## 3. The honest verdict — the illusion does *not* pay here\n\n"
           f"On this liquid mega-cap tape the long-cheap / short-dear spread is "
           f"**{R['spread_bps']:+.2f} bps/day** with NW *t* = **{R['t_nw']:+.2f}** — "
           f"significant, but with the **opposite sign** to the claim: here the low-priced "
           f"names actually *out-earned* the expensive ones, and with a **higher** Sharpe "
           f"({R['lo_sharpe']:+.2f} vs {R['hi_sharpe']:+.2f}). The catch is baked into the "
           f"universe — mega-caps are **rarely cheap** (the cheapest name is ~${R['cheap_val']}, "
           f"no true low-dollar lottery stocks exist here), so the retail-lottery segment the "
           f"theory targets is simply absent, and within mega-caps the lower-dollar names are "
           f"the more value-tilted ones that quietly did well. The seeded synthetic control "
           f"recovers a *planted* under-earn relation cleanly, so this is a genuine "
           f"sign-reversal, not a bug. **Signal: None** (the claimed edge is absent), "
           f"**Tradability: Mirage** (even the sign-flip book dies at cost)."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 872 — Nominal-Price Illusion — the teardown\n\n"
           "The per-leg splits, the Newey-West spread *t*, the pooled Welch book test, the "
           "per-book vol / skew / Sharpe (the risk-adjusted read), the 1,000-permutation "
           "placebo, the two-era robustness cut, the costed timer, and the 20-seed synthetic "
           "control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The headline — long-cheap / short-dear spread (`lo − hi`)\n\n"
           "Daily equal-weight cheapest-30% minus priciest-30% price-level spread."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : cheap {R['lo_bps']:+.2f} vs dear {R['hi_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")"
        ),
        md("## The risk-adjusted lottery read — more risk for less reward?\n\n"
           "The over-priced-lottery hypothesis is about *risk-adjusted* underperformance. "
           "It fails on both legs: the cheap book is only slightly more volatile, and its "
           "Sharpe is *higher*, not lower."),
        code(
            "print(f\"cheap book : vol {R['lo_vol']:.1f}%/yr  skew {R['lo_skew']:+.2f}  Sharpe {R['lo_sharpe']:+.2f}\")\n"
            "print(f\"dear  book : vol {R['hi_vol']:.1f}%/yr  skew {R['hi_skew']:+.2f}  Sharpe {R['hi_sharpe']:+.2f}\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations, two-sided)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> two-sided p = {R['placebo_p']:.4f} \"\n"
            "      f\"(~{R['placebo_sigma']:.1f} sigma into the right tail)\")"
        ),
        md("## Robustness — two eras (split 2018-01-01)"),
        code(
            "print(f\"2010-2017 (n={R['era_early_n']}): {R['era_early_bps']:+.2f} bps  NW t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2018-2026 (n={R['era_late_n']}): {R['era_late_bps']:+.2f} bps  NW t = {R['era_late_t']:+.2f}\")\n"
            "print('the (wrong-sign) out-performance of cheap names holds in BOTH halves -> robust None, not noise')"
        ),
        md("## The timer — can you get paid for it (even the reversed book)?\n\n"
           "2 sides × one-way cost × NAV per day on the long-short book; short (dear) pays "
           "50 bps/yr borrow."),
        code(
            "for tag,g,c,n,t in [('1 bp',R['timer_1_gross'],R['timer_1_cost'],R['timer_1_net'],R['timer_1_t']),\n"
            "                    ('5 bps',R['timer_5_gross'],R['timer_5_cost'],R['timer_5_net'],R['timer_5_t'])]:\n"
            "    print(f\"{tag:>5} one-way: gross {g:+.2f} -> net {n:+.2f} bps/day (cost {c:.2f}/day, t={t:+.2f})\")\n"
            "print('even the data-mined sign-flip is insignificant at 1 bp and negative at 5 bps -> Mirage')"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted "
           "under-earn relation (a *negative* spread), while planting the lottery look."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from nominal_price import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=872+s, n_assets=40, n_days=1200))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0016, seed=872, n_assets=40, n_days=1500))\n"
            "print(f\"planted (edge=0.0016): NW t = {planted['t_nw']:+.2f} (negative = cheap under-earns), Welch t = {planted['welch_t']:+.2f}\")\n"
            "print(f\"planted lottery look: cheap vol {planted['lo_vol']*100:.1f}% skew {planted['lo_skew']:+.2f}  vs  dear vol {planted['hi_vol']*100:.1f}% skew {planted['hi_skew']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The nominal-price money-illusion premium does **not** "
           f"replicate on 50 liquid US mega-caps: the long-cheap / short-dear spread is "
           f"**{R['spread_bps']:+.2f} bps/day** (NW *t* = **{R['t_nw']:+.2f}**) — significant "
           f"but *opposite in sign* to the claim (cheap names out-earned, with a *higher* "
           f"Sharpe, {R['lo_sharpe']:+.2f} vs {R['hi_sharpe']:+.2f}), holding in both eras "
           f"(*t* = {R['era_early_t']:+.2f} / {R['era_late_t']:+.2f}), ≈{R['placebo_sigma']:.1f}σ "
           f"into the right tail of a 1,000-permutation placebo. The synthetic control recovers "
           f"a *planted* under-earn relation cleanly (*t* = {R['planted_t']:+.2f}, fires on "
           f"{R['null_fire']}/20 nulls), so the sign-reversal is real, not machinery. Mega-caps "
           f"are *rarely cheap* — the lottery segment is absent (honest low power).\n"
           f"- **Tradability — Mirage.** Even the data-mined sign-flip dies: net "
           f"**{R['timer_1_net']:+.2f} bps/day** at 1 bp one-way but *insignificant* "
           f"(*t* = {R['timer_1_t']:+.2f}), and **{R['timer_5_net']:+.2f} bps/day** at 5 bps. "
           f"No version survives the {R['timer_1_cost']:.2f} bps/day round-trip friction."),
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
