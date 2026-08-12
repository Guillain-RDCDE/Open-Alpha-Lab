"""Generate the two narrative notebooks for Study 878 (Economic Policy Uncertainty).

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


# Frozen real-tape headline numbers — mirror of docs/results.md. Signal = VIX proxy (the
# newspaper EPU feed was unreachable in-environment); SPY+VIX real, 1993-02-28 -> 2026-06-30.
R = dict(
    source="vix_proxy", start="1993-02-28", end="2026-06-30", n=401, fp="d922d69b4b52",
    unc_lo=9.5, unc_hi=59.9, unc_mean=19.6,
    # forward realized vol on level: h -> (t, r2)
    rv_lvl={1: (10.21, 0.507), 3: (9.44, 0.388), 6: (7.41, 0.301), 12: (6.48, 0.231)},
    rv_chg={1: 3.39, 3: 3.09, 6: 2.49, 12: 2.73},
    # forward return on level: h -> (t, r2)
    ret_lvl={1: (0.65, 0.003), 3: (0.82, 0.010), 6: (1.24, 0.017), 12: (1.04, 0.010)},
    ret_chg={1: -0.49, 3: 0.36, 6: 0.86, 12: 0.26},
    era_early_ret_t=0.09, era_early_rv_t=6.20, era_early_n=191,
    era_late_ret_t=4.36, era_late_rv_t=8.18, era_late_n=210,
    placebo_ret3_p=0.258, placebo_ret6_p=0.221, placebo_rv3_p=0.000,
    timer_leanin_sharpe=0.49, timer_leanin_ann=5.9,
    timer_derisk_sharpe=0.57, timer_derisk_ann=5.2,
    timer_bh_sharpe=0.77, timer_bh_ann=11.4,
    null_ret_t=0.10, null_rv_t=1.12, planted_ret_t=4.96, planted_rv_t=14.26,
    null_fire=1, null_seeds=10,
)


HEADER = f"""# Study 878 — Economic Policy Uncertainty ❓📰

**Does a spike in *policy uncertainty* tell you anything about the *future* — higher
volatility, or a return you get paid for bearing it?**

Baker, Bloom & Davis' newspaper-based **EPU** index is the canonical "how uncertain is
policy" gauge. It is sold on two stories: high EPU should precede **higher equity vol**
(the vol story) and **higher forward returns** as compensation (the risk-premium story).
We test both directly on the aggregate US market ({R['start']} → {R['end']}, {R['n']} months).

> **Data-honesty note.** The real Baker-Bloom-Davis newspaper feed was **network-unreachable**
> from the build environment, so the signal here is a **labelled VIX proxy** (real CBOE
> implied vol) — a market-based stand-in, *never* the newspaper index. See `docs/references.md`.

*Numbers below are the frozen headline (`docs/results.md`, fingerprint `{R['fp']}`); the live
cells run the fast synthetic control.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea\n\n"
           "When policy is uncertain — debt-ceiling brinkmanship, an election, a war — the "
           "newspapers fill with the word *uncertainty*. The story goes that markets get "
           "**choppier** (higher vol) and that you should be **paid extra** to hold stocks "
           "through the fog. The desk's prior is blunter: an uncertainty index is a "
           "*thermometer*, not a *crystal ball* — it spikes **with** the sell-off, not before "
           "the recovery."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(rv_t3=%r, rv_r2_3=%r, ret_t3=%r, ret_r2_3=%r,\n"
            "         placebo_ret3_p=%r, bh_sharpe=%r, leanin_sharpe=%r)\n"
            "print('LEG 1  forward vol   on uncertainty (3m): HAC t = %%+.2f  (R2 = %%.3f)'\n"
            "      %% (R['rv_t3'], R['rv_r2_3']))\n"
            "print('LEG 2  forward return on uncertainty (3m): HAC t = %%+.2f  (R2 = %%.3f)'\n"
            "      %% (R['ret_t3'], R['ret_r2_3']))\n"
            "print('       return-leg block-shuffle placebo p = %%.3f  (broken-link null)'\n"
            "      %% R['placebo_ret3_p'])\n"
            "print('       timing on it: Sharpe %%.2f  vs  buy-and-hold %%.2f'\n"
            "      %% (R['leanin_sharpe'], R['bh_sharpe']))"
            % (R["rv_lvl"][3][0], R["rv_lvl"][3][1], R["ret_lvl"][3][0], R["ret_lvl"][3][1],
               R["placebo_ret3_p"], R["timer_bh_sharpe"], R["timer_leanin_sharpe"])
        ),
        md("## 2. Is the machinery even honest? A live synthetic control\n\n"
           "We build a seeded toy world where the *previous* month's uncertainty genuinely "
           "drives *this* month's vol and return (`edge>0`), and check the detector recovers "
           "**both** legs — and stays silent on the null (`edge=0`). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from epu import data, strategy as st\n"
            "null = st.synthetic_detect(*data.synthetic(360, 0.0, 0.0, 878), horizon=3)\n"
            "planted = st.synthetic_detect(*data.synthetic(360, 0.02, 0.6, 878), horizon=3)\n"
            "print('null    world: ret_t=%+.2f  rv_t=%+.2f  (should be ~0)' % (null['ret_t'], null['rv_t']))\n"
            "print('planted world: ret_t=%+.2f  rv_t=%+.2f  (both light up)' % (planted['ret_t'], planted['rv_t']))"
        ),
        md("## 3. The honest verdict\n\n"
           f"On the real tape the **vol leg fires hard** (HAC *t* = "
           f"**+{R['rv_lvl'][3][0]:.2f}** at 3m) — but that is nearly **mechanical**: the "
           f"VIX proxy *is* the market's implied vol, so of course it tracks realized vol; "
           f"it's a *coincident thermometer*, not a forward edge. The **return leg — the "
           f"part you'd actually get paid for — is dead**: HAC *t* = "
           f"**+{R['ret_lvl'][3][0]:.2f}** (R² ≈ {R['ret_lvl'][3][1]:.2f}), a placebo p of "
           f"**{R['placebo_ret3_p']:.2f}**, and what little post-2009 significance appears "
           f"(*t* = +{R['era_late_ret_t']:.2f}) **vanishes pre-2009** "
           f"(*t* = +{R['era_early_ret_t']:.2f}) — a single-era recovery-drift artefact. And "
           f"no uncertainty-timed rule beats buy-and-hold (Sharpe "
           f"{R['timer_leanin_sharpe']:.2f} vs {R['timer_bh_sharpe']:.2f}). "
           f"**Signal: None** (uncertainty is contemporaneous), **Tradability: Mirage.**"),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 878 — Economic Policy Uncertainty — the teardown\n\n"
           "The two predictive regressions (forward vol AND forward return on the uncertainty "
           "level/change) with Newey-West HAC *t*, the two-era cut, the block-shuffle placebo, "
           "the costed timer, and the live synthetic control. *Signal = labelled VIX proxy — "
           "the newspaper EPU feed was unreachable in-environment (see `docs/references.md`).*"),
        code("R = %r" % (R,)),
        md("## Leg 1 — forward realized vol on the uncertainty level (the 'vol story')"),
        code(
            "for h in (1,3,6,12):\n"
            "    t,r2 = R['rv_lvl'][h]\n"
            "    print(f\"h={h:>2}m: HAC t = {t:+6.2f}   R2 = {r2:.3f}   (on change: t = {R['rv_chg'][h]:+.2f})\")\n"
            "print('\\n-> strongly significant everywhere, BUT VIX *is* implied vol: this is the'\n"
            "      ' near-mechanical variance-risk-premium fact, a coincident reading, not an edge.')"
        ),
        md("## Leg 2 — forward SPY return on the uncertainty level (the 'risk-premium story')"),
        code(
            "for h in (1,3,6,12):\n"
            "    t,r2 = R['ret_lvl'][h]\n"
            "    print(f\"h={h:>2}m: HAC t = {t:+6.2f}   R2 = {r2:.3f}   (on change: t = {R['ret_chg'][h]:+.2f})\")\n"
            "print('\\n-> not one horizon clears |t|=2; R2 ~ 0.01. The risk-premium claim fails.')"
        ),
        md("## Robustness — two eras (split 2009-01-01), horizon 3m"),
        code(
            "print(f\"1993-2008 (n={R['era_early_n']}): RET t = {R['era_early_ret_t']:+.2f} | RV t = {R['era_early_rv_t']:+.2f}\")\n"
            "print(f\"2009-2026 (n={R['era_late_n']}): RET t = {R['era_late_ret_t']:+.2f} | RV t = {R['era_late_rv_t']:+.2f}\")\n"
            "print('-> the return leg lives ONLY post-2009 (recovery drift) and is absent before: a single-era artefact.')"
        ),
        md("## Placebo — block-shuffle the regressor (broken link), 1,000 draws"),
        code(
            "print(f\"return h=3: p = {R['placebo_ret3_p']:.3f}   return h=6: p = {R['placebo_ret6_p']:.3f}\")\n"
            "print(f\"vol    h=3: p = {R['placebo_rv3_p']:.3f}  (the vol slope is real; the return slope is not)\")"
        ),
        md("## The timer — lean INTO high uncertainty vs buy-and-hold"),
        code(
            "print(f\"lean-in : ann {R['timer_leanin_ann']:+.1f}%  Sharpe {R['timer_leanin_sharpe']:.2f}\")\n"
            "print(f\"de-risk : ann {R['timer_derisk_ann']:+.1f}%  Sharpe {R['timer_derisk_sharpe']:.2f}\")\n"
            "print(f\"buy-hold: ann {R['timer_bh_ann']:+.1f}%  Sharpe {R['timer_bh_sharpe']:.2f}  <- neither rule beats it\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased (live)\n\n"
           "The detector must recover planted forward relations on both legs and stay silent on the null."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from epu import data, strategy as st\n"
            "null = st.synthetic_detect(*data.synthetic(360, 0.0, 0.0, 878), horizon=3)\n"
            "planted = st.synthetic_detect(*data.synthetic(360, 0.02, 0.6, 878), horizon=3)\n"
            "print(f\"null    : ret_t {null['ret_t']:+.2f}  rv_t {null['rv_t']:+.2f}\")\n"
            "print(f\"planted : ret_t {planted['ret_t']:+.2f}  rv_t {planted['rv_t']:+.2f}\")\n"
            "nr = np.array([st.synthetic_detect(*data.synthetic(300,0.0,0.0,878+s),3)['ret_t'] for s in range(10)])\n"
            "print(f\"null 10 seeds: ret_t mean {nr.mean():+.2f} (sd {nr.std(ddof=1):.2f}), |t|>=2 in {(abs(nr)>=2).sum()}/10\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The risk-premium claim fails: forward-return HAC *t* = "
           f"+{R['ret_lvl'][1][0]:.2f} … +{R['ret_lvl'][6][0]:.2f} across 1–12m (R² ≈ 0.01), "
           f"placebo p = {R['placebo_ret3_p']:.2f}, and the post-2009 flicker "
           f"(*t* = +{R['era_late_ret_t']:.2f}) is absent pre-2009 "
           f"(*t* = +{R['era_early_ret_t']:.2f}). The vol leg is significant "
           f"(*t* = +{R['rv_lvl'][3][0]:.2f}) but **mechanical** (VIX ≈ implied vol) and "
           f"contemporaneous. Uncertainty is a thermometer, not a crystal ball. *Signal is a "
           f"labelled VIX proxy — the newspaper EPU feed was unreachable in-environment.*\n"
           f"- **Tradability — Mirage.** No uncertainty-timed rule beats buy-and-hold "
           f"(Sharpe {R['timer_leanin_sharpe']:.2f} / {R['timer_derisk_sharpe']:.2f} vs "
           f"{R['timer_bh_sharpe']:.2f}); it loses on signal, not costs."),
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
