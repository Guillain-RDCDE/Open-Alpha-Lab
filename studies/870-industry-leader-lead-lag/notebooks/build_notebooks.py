"""Generate the two narrative notebooks for Study 870 (Industry-Leader Lead-Lag).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily OHLCV,
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; weekly W-FRI returns,
# 8 sectors, largest-cap leaders, long up-leader / short down-leader followers).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_weeks=859,
    spread_bps=-3.64, t_nw=-0.77, t_1s=-0.69,
    up_bps=28.79, dn_bps=35.99, welch_t=-1.99, gross_sharpe=-0.17,
    placebo_obs=-3.64, placebo_mean=3.465, placebo_sd=4.935,
    placebo_p=0.928, placebo_draws=1000,
    era_early_bps=-13.96, era_early_t=-2.16, era_early_n=415,
    era_late_bps=5.96, era_late_t=0.89, era_late_n=443,
    dyn_bps=-3.87, dyn_t=-0.81,
    timer_1_gross=-3.64, timer_1_cost=2.96, timer_1_net=-6.60, timer_1_t=-1.24,
    timer_5_gross=-3.64, timer_5_cost=10.96, timer_5_net=-14.60, timer_5_t=-2.75,
    null_mean_t=-0.04, null_sd_t=0.79, null_fire=0,
    planted_t=20.79, planted_welch=17.91,
)


HEADER = f"""# Study 870 — Industry-Leader Lead-Lag 👑

**Does the *biggest* name in a sector lead the rest?**

Hou (2007) finds that information diffuses **within an industry** from the largest firm
outward: the bellwether's return this week foreshadows its smaller peers' returns next
week (slow within-industry diffusion). Trade it — long the followers whose leader *rose*,
short those whose leader *fell*. We take the self-contained weekly version on a liquid US
cross-section ({R['start']} → {R['end']}, {R['n_names']} names in 8 sectors).

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship + leader designation: current-membership mega-caps,
largest-cap leaders — magnitudes are an upper bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Big, closely-followed firms price sector news first; their smaller industry "
           "peers, watched by fewer eyes, catch up a beat later. So the **leader's** move "
           "this week should tip the **followers'** move next week. Buy the followers of "
           "leaders that rose, sell the followers of leaders that fell."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_bps=%r, t_nw=%r, up_bps=%r, dn_bps=%r, gross_sharpe=%r, placebo_p=%r)\n"
            "print('long up-leader / short down-leader followers spread: %%+.2f bps/week (NW t = %%+.2f)'\n"
            "      %% (R['spread_bps'], R['t_nw']))\n"
            "print('  followers after up-leader %%+.2f bps vs after down-leader %%+.2f bps'\n"
            "      %% (R['up_bps'], R['dn_bps']))\n"
            "print('  gross weekly Sharpe (before cost): %%.2f' %% R['gross_sharpe'])\n"
            "print('  placebo right-tail p: %%.3f (nothing special in the true alignment)' %% R['placebo_p'])"
            % (R["spread_bps"], R["t_nw"], R["up_bps"], R["dn_bps"], R["gross_sharpe"], R["placebo_p"])
        ),
        md("## 2. Is the sort even wired up? A live synthetic control\n\n"
           "We plant a leader→follower diffusion in a seeded toy world (`edge>0`) and check "
           "the detector recovers it — and that it stays *silent* on the null (`edge=0`, "
           "leaders and followers independent). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from leader_lag import data, strategy as st\n"
            "secs, lds = data.synthetic_sectors(), data.synthetic_leaders()\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=870, n_weeks=260), secs, lds)\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.6, seed=870, n_weeks=320), secs, lds)\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — the famous edge does *not* replicate here\n\n"
           f"On this liquid mega-cap tape the long-up-leader / short-down-leader followers "
           f"spread is **{R['spread_bps']:+.2f} bps/week** with NW *t* = **{R['t_nw']:+.2f}** — "
           f"statistically indistinguishable from zero, and if anything leaning the *wrong* way "
           f"(followers earned a touch *more* after their leader **fell**, Welch *t* = "
           f"{R['welch_t']:+.2f}). The permutation null actually centres *above* the observed "
           f"value (right-tail p = {R['placebo_p']:.2f}), and the sign flips era-to-era "
           f"({R['era_early_bps']:+.1f} bps early vs {R['era_late_bps']:+.1f} late). The seeded "
           "synthetic control recovers a *planted* diffusion emphatically, so the sort works — "
           "there is simply no lead-lag to harvest here. Slow within-industry diffusion is a "
           "small-and-illiquid-firm effect; 50 mega-caps price sector news near-simultaneously. "
           "**Signal: None**, **Tradability: Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 870 — Industry-Leader Lead-Lag — the teardown\n\n"
           "The weekly spread Newey-West *t*, the pooled Welch leg test, the 1,000-permutation "
           "placebo, the two-era robustness cut, the dollar-volume leader re-designation, the "
           "costed timer, and the 20-seed synthetic control."),
        code(
            "R = %r" % (R,)
        ),
        md("## The headline — long up-leader / short down-leader followers\n\n"
           "Weekly equal-per-sector spread: `mean_s sign(leader_w) · mean_followers(ret_{w+1})`."),
        code(
            "print(f\"spread        : {R['spread_bps']:+.2f} bps/week  NW(6) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}  (n = {R['n_weeks']} weeks)\")\n"
            "print(f\"legs          : after up-leader {R['up_bps']:+.2f} vs after down-leader {R['dn_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"gross Sharpe  : {R['gross_sharpe']:.2f} (before cost)\")"
        ),
        md("## Placebo — shuffle the lead→lag week alignment (1,000 permutations)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> right-tail p = {R['placebo_p']:.3f}\")\n"
            "print('the null sits ABOVE the observed value — nothing special in the true alignment')"
        ),
        md("## Robustness — two eras (split 2018-01-01) and a dollar-volume leader re-designation"),
        code(
            "print(f\"2010-2017 (n={R['era_early_n']}): {R['era_early_bps']:+.2f} bps  NW t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2018-2026 (n={R['era_late_n']}): {R['era_late_bps']:+.2f} bps  NW t = {R['era_late_t']:+.2f}\")\n"
            "print(f\"$-vol leaders   : {R['dyn_bps']:+.2f} bps  NW t = {R['dyn_t']:+.2f} (same non-result)\")\n"
            "print('sign FLIPS across eras and clears |t|>=2 only in the WRONG direction')"
        ),
        md("## The timer — can you get paid for it?\n\n"
           "2 sides × one-way cost × NAV per week on the long-short book; short pays 50 bps/yr borrow."),
        code(
            "for tag,g,c,n,t in [('1 bp',R['timer_1_gross'],R['timer_1_cost'],R['timer_1_net'],R['timer_1_t']),\n"
            "                    ('5 bps',R['timer_5_gross'],R['timer_5_cost'],R['timer_5_net'],R['timer_5_t'])]:\n"
            "    print(f\"{tag:>5} one-way: gross {g:+.2f} -> net {n:+.2f} bps/week (cost {c:.2f}/wk, t={t:+.2f})\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted diffusion."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from leader_lag import data, strategy as st\n"
            "secs, lds = data.synthetic_sectors(), data.synthetic_leaders()\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=870+s, n_weeks=200), secs, lds)['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.6, seed=870, n_weeks=320), secs, lds)\n"
            "print(f\"planted (edge=0.6): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** Hou's industry-leader lead-lag does **not** replicate on 50 liquid "
           f"US mega-caps: the long-up-leader / short-down-leader followers spread is "
           f"**{R['spread_bps']:+.2f} bps/week** (NW *t* = **{R['t_nw']:+.2f}**) — indistinguishable "
           f"from zero and on the *wrong* side if anything (Welch *t* = {R['welch_t']:+.2f}), "
           f"unremarkable against a 1,000-permutation placebo (p = {R['placebo_p']:.2f}), "
           f"sign-flipping across eras ({R['era_early_bps']:+.1f} / {R['era_late_bps']:+.1f} bps), and "
           f"unchanged under a dollar-volume leader re-designation ({R['dyn_bps']:+.2f} bps). The 20-seed "
           f"synthetic control recovers a *planted* diffusion cleanly (*t* = {R['planted_t']:+.2f}, fires "
           f"on {R['null_fire']}/20 nulls), so this is a true absence — the effect lives among small "
           f"illiquid firms this survivor panel omits.\n"
           f"- **Tradability — Mirage.** The specified book loses money gross and net "
           f"({R['timer_1_net']:+.2f} bps/week at 1 bp, {R['timer_5_net']:+.2f} at 5 bps); even the "
           f"data-mined sign-flip is eaten by the {R['timer_1_cost']:.2f} bps/week friction at 1 bp."),
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
