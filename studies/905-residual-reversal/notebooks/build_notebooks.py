"""Generate the two narrative notebooks for Study 905 (Residual Reversal).

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
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; weekly market-model
# residual, long past-week-residual-loser / short-winner, top-60% dollar-volume screen).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_weeks=808, fp="357fd262912f",
    spread_bps=-0.38, t_nw=-0.05, t_1s=-0.04,
    lo_bps=34.12, hi_bps=34.51, welch_t=-0.03, gross_sharpe=-0.01, hit=51.1,
    raw_bps=-0.03, raw_t=-0.00,
    noscreen_bps=2.31, noscreen_t=0.32,
    placebo_obs=-0.38, placebo_mean=0.063, placebo_sd=5.348, placebo_p=0.513,
    placebo_draws=1000,
    era_early_bps=2.18, era_early_t=0.26, era_early_n=364,
    era_late_bps=-3.18, era_late_t=-0.23, era_late_n=443,
    timer_1_gross=-0.38, timer_1_cost=2.96, timer_1_net=-3.35, timer_1_t=-0.39,
    timer_5_gross=-0.38, timer_5_cost=10.96, timer_5_net=-11.35, timer_5_t=-1.32,
    null_mean_t=0.17, null_sd_t=0.78, null_fire=0,
    planted_t=24.99, planted_raw_t=13.23,
)


HEADER = f"""# Study 905 — Residual Reversal ↩️

**Strip the factor out of weekly reversal — does the *cleaned* signal finally pay?**

Blitz, Huij, Lansdorp & Verbeek (2013) argue the classic one-week reversal (buy last
week's losers, sell its winners) mostly harvests **bid-ask bounce** and **common-factor**
moves, so it dies at the spread. Their fix: regress each name's weekly return on the
market, keep the **residual**, and reverse on *that*, on a liquid subset. We run the
self-contained version on a liquid US cross-section ({R['start']} → {R['end']},
{R['n_names']} names), with the RAW weekly reversal placed beside it as the foil.

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: current-membership mega-caps — magnitudes are an upper
bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "A stock that dropped last week is a reversal *buy* — unless it dropped because "
           "the **whole market** dropped, in which case it is not over-sold at all, just "
           "beta doing its job. Raw reversal can't tell the two apart, so it keeps buying "
           "high-beta names into market rebounds and paying the bid-ask spread for the "
           "privilege. The **residual** reversal first removes the market move (a "
           "market-model regression) and reverses only on what's left — the genuinely "
           "idiosyncratic wobble — on liquid names where the bounce is small."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, raw_bps=%r, raw_t=%r, lo_bps=%r, hi_bps=%r, gross_sharpe=%r)\n"
            "print('residual reversal (liquid, factor-cleaned): %%+.2f bps/wk (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('raw reversal (same screen, no cleaning)    : %%+.2f bps/wk (NW t = %%+.2f)'\n"
            "      %% (R['raw_bps'], R['raw_t']))\n"
            "print('  residual-loser book %%+.2f bps vs residual-winner book %%+.2f bps'\n"
            "      %% (R['lo_bps'], R['hi_bps']))\n"
            "print('  gross spread Sharpe (before cost): %%.2f' %% R['gross_sharpe'])"
            % (R["spread_bps"], R["t_nw"], R["raw_bps"], R["raw_t"],
               R["lo_bps"], R["hi_bps"], R["gross_sharpe"])
        ),
        md("## 2. Does the cleaner really clean? A live synthetic control\n\n"
           "We plant a weekly residual mean-reversion in a seeded toy world (`edge>0`) on "
           "top of a strong common factor, and check that the **residual** detector "
           "recovers it while the **raw** detector is muddied by the factor — and that "
           "both stay *silent* on the null (`edge=0`). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from resid_reversal import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=905, n_assets=40, n_days=1600))\n"
            "plan = st.synthetic_detect(data.synthetic_panel(edge=0.35, seed=905, n_assets=40, n_days=2000))\n"
            "print('null world   : residual NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: residual NW t = %+.2f  vs RAW NW t = %+.2f (factor-muddied)'\n"
            "      % (plan['t_nw'], plan['raw_t_nw']))"
        ),
        md("## 3. The honest verdict — on 50 mega-caps, even the *clean* signal is flat\n\n"
           f"The residual reversal, factor-stripped and liquidity-screened exactly as the "
           f"recipe asks, earns **{R['spread_bps']:+.2f} bps/week** with NW *t* = "
           f"**{R['t_nw']:+.2f}** — indistinguishable from zero. The raw foil is just as "
           f"dead (**{R['raw_bps']:+.2f}** bps, *t* = {R['raw_t']:+.2f}); dropping the "
           f"liquidity screen barely moves it ({R['noscreen_bps']:+.2f} bps, *t* = "
           f"{R['noscreen_t']:+.2f}). The permutation placebo sits dead-centre (*p* = "
           f"{R['placebo_p']:.2f}) and the two eras flip sign "
           f"({R['era_early_bps']:+.2f} → {R['era_late_bps']:+.2f} bps), both "
           f"insignificant. The synthetic control proves the cleaner *works* (it recovers "
           f"a planted residual reversal at *t* = {R['planted_t']:.0f}, silent on "
           f"{R['null_fire']}/20 nulls), so this is a genuine **absence of edge**, not "
           f"broken code: short-term reversal — residual or raw — is a small-cap / "
           f"illiquid-breadth phenomenon, and 50 liquid mega-caps is exactly where it is "
           f"not. **Signal: None**, **Tradability: Mirage** (costs turn the flat gross "
           f"deeply negative: **{R['timer_1_net']:+.2f} bps/wk** net at just 1 bp)."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 905 — Residual Reversal — the teardown\n\n"
           "The residual construction, the residual-vs-raw race, the Newey-West spread "
           "*t*, the pooled Welch book test, the 1,000-permutation placebo, the two-era "
           "cut, the costed timer, and the 20-seed synthetic control."),
        code("R = %r" % (R,)),
        md("## The headline — long residual-loser / short residual-winner, liquid subset\n\n"
           "Weekly equal-weight bottom-30% minus top-30% market-model-residual spread, "
           "top-60% by trailing dollar volume. The RAW weekly reversal is the foil."),
        code(
            "print(f\"residual reversal : {R['spread_bps']:+.2f} bps/wk  NW(8) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}  (n={R['n_weeks']} wks)\")\n"
            "print(f\"  books           : loser {R['lo_bps']:+.2f} vs winner {R['hi_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f}), gross Sharpe {R['gross_sharpe']:.2f}, hit {R['hit']:.1f}%\")\n"
            "print(f\"raw reversal (foil): {R['raw_bps']:+.2f} bps/wk  NW t = {R['raw_t']:+.2f}\")\n"
            "print(f\"residual, NO screen: {R['noscreen_bps']:+.2f} bps/wk  NW t = {R['noscreen_t']:+.2f}\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> p = {R['placebo_p']:.3f}  (dead centre)\")"
        ),
        md("## Robustness — two eras (split 2018-01-01)"),
        code(
            "print(f\"2010-2017 (n={R['era_early_n']}): {R['era_early_bps']:+.2f} bps  NW t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2018-2026 (n={R['era_late_n']}): {R['era_late_bps']:+.2f} bps  NW t = {R['era_late_t']:+.2f}\")\n"
            "print('  -> the sign flips across eras; nothing to stand on.')"
        ),
        md("## The timer — can you get paid for it?\n\n"
           "2 sides × one-way cost × NAV per weekly rebalance; short pays 50 bps/yr borrow."),
        code(
            "for tag,g,c,n,t in [('1 bp',R['timer_1_gross'],R['timer_1_cost'],R['timer_1_net'],R['timer_1_t']),\n"
            "                    ('5 bps',R['timer_5_gross'],R['timer_5_cost'],R['timer_5_net'],R['timer_5_t'])]:\n"
            "    print(f\"{tag:>5} one-way: gross {g:+.2f} -> net {n:+.2f} bps/wk (cost {c:.2f}/wk, t={t:+.2f})\")"
        ),
        md("## Synthetic positive control — the cleaner is real, the null is real\n\n"
           "Live: the residual detector must recover a planted residual reversal, beat the "
           "factor-muddied raw detector, and NOT fire on the null."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from resid_reversal import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=905+s, n_assets=40, n_days=1600))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: residual NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "plan = st.synthetic_detect(data.synthetic_panel(edge=0.35, seed=905, n_assets=40, n_days=2000))\n"
            "print(f\"planted (edge=0.35): residual NW t = {plan['t_nw']:+.2f}  >>  raw NW t = {plan['raw_t_nw']:+.2f} (muddied)\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** On 50 liquid US mega-caps the factor-cleaned, "
           f"liquidity-screened weekly residual reversal earns **{R['spread_bps']:+.2f} "
           f"bps/week** (NW *t* = **{R['t_nw']:+.2f}**) — a flat line. The raw foil is "
           f"equally dead ({R['raw_bps']:+.2f} bps), the placebo is dead-centre "
           f"(*p* = {R['placebo_p']:.2f}), and the two eras flip sign "
           f"({R['era_early_t']:+.2f} / {R['era_late_t']:+.2f}). The 20-seed synthetic "
           f"control recovers a *planted* residual reversal cleanly (*t* = "
           f"{R['planted_t']:.0f}, above the factor-muddied raw *t* = "
           f"{R['planted_raw_t']:.0f}; fires on {R['null_fire']}/20 nulls), so the null is "
           f"real, not machinery: short-term reversal lives in small/illiquid breadth, not "
           f"in mega-caps. *Survivorship biases the magnitude upward — and it is still "
           f"zero.*\n"
           f"- **Tradability — Mirage.** A gross edge of essentially zero cannot survive "
           f"the {R['timer_1_cost']:.2f} bps/week round-trip friction of a fully-turning "
           f"weekly book: net **{R['timer_1_net']:+.2f} bps/wk** at 1 bp one-way, "
           f"**{R['timer_5_net']:+.2f}** at 5 bps. Even the slightly-positive un-screened "
           f"gross ({R['noscreen_bps']:+.2f} bps) is eaten many times over."),
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
