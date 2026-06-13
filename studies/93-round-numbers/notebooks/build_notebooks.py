"""Build the two narrative notebooks for Study 93 (Round-Numbers).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both walk the seven desk beats at two altitudes. Figures are computed live from the
``round_numbers`` package on the cached real tape; the frozen ``R`` dict mirrors
``docs/results.md`` for the verdict prose so the two never drift.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-12).
R = dict(
    as_of="2026-06-12", fp="84df82af343c", n_pooled=35936,
    chi2_pool=380.4, p_pool="2.0e-76", frac_near=11.2,
    chi2_msft=67.7, p_msft="4.3e-11", chi2_aapl=50.3, p_aapl="9.5e-08",
    chi2_spy=34.3, chi2_gspc=16.9,
    fade_trades=4642, fade_bps=-0.53, fade_t=-0.18,
    fade_net_bps=-10.53, cost_bps=5.0,
    rand_bps=-0.10, rand_sd=2.77, beats_pct=43, n_seeds=200,
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study root (round_numbers/)
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5.5)
import numpy as np, pandas as pd
from round_numbers import data, strategy

AS_OF = "2026-06-12"
BAND, HORIZON = 0.004, 1
names = data.load_basket(as_of=AS_OF)
fp = data.fingerprint([c.to_numpy() for _, _, c in names])
for t, g, c in names:
    print(f"{t:6} grid={g:6.0f}  {len(c):,} rows  {c.index[0].date()} -> {c.index[-1].date()}")
print("basket fingerprint =", fp)
"""


def md(t):
    return new_markdown_cell(t)


def code(t):
    return new_code_cell(t)


HERO_BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Beats a random level?: Not supported](https://img.shields.io/badge/Beats_a_random_level%3F-Not_supported-8b949e?style=flat-square)\n\n"
)

# A reusable cell that draws the pooled distance-to-round histogram.
CLUSTER_FIG = (
    "allc = np.concatenate([c.to_numpy() for _, _, c in names])\n"
    "r = strategy.clustering_test(allc, 1.0)\n"
    "dist = strategy.distance_to_round(allc, 1.0)\n"
    "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
    "ax.hist(dist, bins=10, range=(0, 0.5), alpha=.85, edgecolor='white')\n"
    "ax.axhline(len(dist)/10, color='C3', ls='--', lw=2, label='uniform expectation')\n"
    "ax.set_xlabel('distance from close to nearest whole dollar'); ax.set_ylabel('count')\n"
    "ax.set_title(f\"Pooled clustering: chi2 = {r['chi2']:.0f}, p = {r['p']:.1e} \"\n"
    "             f\"({r['frac_near']*100:.1f}% within 5% of a round number vs 10% uniform)\")\n"
    "ax.legend(); plt.show()\n"
    "print(f\"pooled n={r['n']:,}  chi2={r['chi2']:.1f}  p={r['p']:.1e}  frac_near={r['frac_near']*100:.1f}%\")"
)

# A reusable cell that runs the fade and the random-level control.
FADE_FIG = (
    "real = strategy.pooled_fade(names, band=BAND, horizon=HORIZON)\n"
    "t_real = strategy.hac_tstat(real)\n"
    "net = strategy.net_of_costs(real, cost_bps=5.0)\n"
    "rand = strategy.random_level_pool_means(names, n_seeds=200, band=BAND, horizon=HORIZON)\n"
    "beats = (real.mean() > rand).mean()\n"
    "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
    "ax.hist(rand*1e4, bins=30, alpha=.85, label='random-level fade (200 seeds)')\n"
    "ax.axvline(real.mean()*1e4, color='C3', lw=2, label=f'round-number fade = {real.mean()*1e4:+.2f} bps')\n"
    "ax.axvline(0, color='k', lw=1, ls=':')\n"
    "ax.set_xlabel('mean return per trade (bps)'); ax.set_ylabel('count'); ax.legend()\n"
    "ax.set_title(f'Round-number fade beats random levels {beats*100:.0f}% of seeds (HAC t = {t_real:+.2f})')\n"
    "plt.show()\n"
    "print(f'real fade: trades {len(real):,}  mean {real.mean()*1e4:+.2f} bps  HAC t {t_real:+.2f}')\n"
    "print(f'net of 5 bps/leg: {net.mean()*1e4:+.2f} bps/trade')\n"
    "print(f'random-level: {rand.mean()*1e4:+.2f} bps (sd {rand.std()*1e4:.2f})  -> real beats {beats*100:.0f}% of seeds')"
)


def build_curious():
    cells = [
        md(
            "# Round Numbers 🔢\n"
            "### Prices pile up at whole dollars — but can you *trade* the magnetism?\n\n"
            + HERO_BADGES +
            "Traders swear prices are *magnetised* by round numbers — that a stock approaching $200 "
            "or the S&P nearing 6,000 will stall and bounce, so you can fade the approach. There are "
            "two claims hiding in there. One is true: prices really do **cluster** at round levels. "
            "The other — that you can **make money** off it — is not.\n\n"
            "> 📓 **This is the plain-language layer.** The chi-square, the HAC *t* and the "
            "random-level control are in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same "
            "story, deeper.\n>\n"
            "> ⚠️ **Not investment advice.** Educational, reproducible research — every chart below is "
            "generated by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do prices cluster at round numbers? | **Yes** — **{R['frac_near']}%** of closes land within "
            f"5% of a whole dollar, vs the **10%** pure chance predicts (chi² = **{R['chi2_pool']:.0f}**). |\n"
            f"| Does nearness to a round number predict a bounce? | **No** — fading it earns **{R['fade_bps']} bps** "
            f"a trade, statistically a flat zero. |\n"
            f"| Could you trade it? | **No** — after costs each fade loses **~{abs(R['fade_net_bps']):.0f} bps**, "
            "and it can't even beat fading *random* levels. |"
        ),
        md(
            "## 1 · The claim\n\n"
            "*\"Prices are drawn to round numbers. A name climbing toward $200 stalls there; the index "
            "pauses at every thousand. Fade the approach — short into the round number, buy the bounce — "
            "and collect.\"* It's some of the oldest lore on the trading floor."
        ),
        md(
            "## 2 · So what?\n\n"
            "If round numbers really repelled price, you'd have a free, ever-present mean-reversion "
            "signal on every liquid name in the world. That's a big enough prize that it deserves a "
            "fair test — and a careful one, because *clustering* and *forecastability* are two very "
            "different things that the lore quietly mashes together."
        ),
        md(
            "## 3 · How would we even know?\n\n"
            "Two separate tests. **(a)** Do closes pile up near round numbers more than chance? We "
            "measure how far each close sits from its nearest whole dollar and check the shape. "
            "**(b)** Does sitting near a round number *forecast* a reversal? We fade it — short just "
            "below the level, long just above — enter **one day later**, and see what the next day "
            "brings. Then the honest part: we compare it to fading **random** price levels at the same "
            "frequency. If round-ness matters, the real fade should beat the random one."
        ),
        md("## 4 · The teardown — let's actually look\n\nFirst, **do prices cluster?** Here is the "
           "distance from every close in the basket to its nearest whole dollar:"),
        code(CLUSTER_FIG),
        md(
            f"The first bar towers over the uniform line: **{R['frac_near']}%** of closes sit within a "
            f"nickel of a whole dollar against the **10%** chance would give. With {R['n_pooled']:,} "
            "observations that's not luck (chi² ≈ "
            f"{R['chi2_pool']:.0f}). **Clustering is real** — exactly what sixty years of research says."
        ),
        md("Now the part the lore actually sells: **can you trade it?** We fade every approach to a "
           "round level and line it up against fading *random* levels of the same spacing."),
        code(FADE_FIG),
        md(
            f"The red line — the real round-number fade — sits **right in the middle** of the random-level "
            f"cloud (it wins only **{R['beats_pct']}%** of the {R['n_seeds']} random draws, a coin-flip "
            "leaning the *wrong* way). And its average trade is **negative even before costs**. The "
            "magnetism is in the *ruler*, not the *future*."
        ),
        md(
            "## 5 · The verdict\n\n"
            f"- **Clustering → Real.** {R['frac_near']}% near a round number vs 10% uniform, chi² "
            f"{R['chi2_pool']:.0f} (p ≈ {R['p_pool']}). Prices genuinely pile up at round levels.\n"
            f"- **Forecast → None.** The fade earns **{R['fade_bps']} bps**/trade (HAC *t* {R['fade_t']}) "
            f"and beats random levels only {R['beats_pct']}% of the time.\n"
            f"- **Trade it? → Mirage.** Net of {R['cost_bps']:.0f} bps/leg each fade loses "
            f"**~{abs(R['fade_net_bps']):.0f} bps**."
        ),
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing to trade. The signal doesn't pay even gross, so costs are almost beside the "
            "point — but for completeness, a realistic 5 bps/leg turns the −0.5 bps gross into about "
            f"**−{abs(R['fade_net_bps']):.0f} bps a trade**. A clustering fact is not a forecast; the "
            "round numbers tell you where price has *been*, not where it's *going*."
        ),
        md(
            "## 7 · Going further 🚪\n\n"
            "- Does the *intraday* tape (where Harris found the strongest clustering) hide a fadeable "
            "edge the daily close washes out?\n"
            "- Are **00-levels** (whole hundreds) more magnetic than mere whole dollars? Tighten the "
            "grid and see.\n"
            "- Could clustering be a *liquidity* signal instead — more limit orders resting at round "
            "prices — rather than a directional one? Fork it and test."
        ),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "01_for_the_curious.ipynb")


def build_quants():
    cells = [
        md(
            "# Round Numbers — a quantitative teardown 🔬\n"
            "### Real tape · chi-square clustering · HAC fade · random-level control · cost sweep\n\n"
            + HERO_BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same seven "
            "beats, every claim now carrying its standard error.* We separate the two things the lore "
            "conflates — price-level **clustering** (real, and old news) and **forecastability** of a "
            "fade (absent) — and show the fade can't even beat a random-level placebo.\n\n"
            "> ⚠️ **Not investment advice.** Basket ^GSPC/SPY/AAPL/MSFT, daily (`quantlab.data`, Yahoo); "
            "single names in `raw` (as-traded nominal), index/ETF in `split_only`; fade returns demeaned "
            "by each tape's drift; one-day execution lag. Sources in "
            "[`docs/references.md`](../docs/references.md), reproducible run in "
            "[`docs/results.md`](../docs/results.md).\n>\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Why |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | Real on the level — pooled chi² **{R['chi2_pool']:.0f}** (p ≈ "
            f"{R['p_pool']}), {R['frac_near']}% within 5% of a round number. None on the forecast — fade "
            f"**{R['fade_bps']} bps**/trade, HAC *t* **{R['fade_t']}**. |\n"
            f"| **Tradability** | `MIRAGE` | Negative gross; net of {R['cost_bps']:.0f} bps/leg "
            f"**{R['fade_net_bps']} bps**/trade. No edge to scale. |\n"
            f"| **Beats a random level?** | `NOT SUPPORTED` | Real fade beats random-level fade "
            f"{R['beats_pct']}% of {R['n_seeds']} seeds — a coin-flip. |\n\n"
            "> 💡 **In plain words:** the magnetism is a property of the *ruler* (where prices like to "
            "sit), not of the *future* (where they go next)."
        ),
        md(
            "## 1 · The claim, steelmanned\n\n"
            "- **H₁ (clustering):** closes are non-uniformly distributed in their distance to the "
            "nearest round level — they pile up near it.\n"
            "- **H₂ (forecast):** that proximity predicts a reversal away from the level — a fade earns a "
            "positive, autocorrelation-robust return.\n"
            "- **H₃ (information):** the *round* levels do this better than *arbitrary* levels at the "
            "same frequency.\n\n"
            "H₁ can be true while H₂ and H₃ are false — and that is exactly the split we find."
        ),
        md("## 2 · So what? — what rides on each answer\n\nIf H₂/H₃ held, every liquid asset would carry "
           "a permanent, free mean-reversion signal. The clustering literature (Osborne 1962, Harris "
           "1991) all but guarantees H₁; the open question — and the one the lore cashes in — is H₂."),
        md("## 3 · How we'd know — the protocol\n\nChi-square on the folded distance-to-round "
           "(per name + pooled) · fade as a drift-demeaned fixed-horizon event study with one-day lag · "
           "HAC *t* on the pooled trade returns · random-level control (phase-shifted grid, 200 seeds) · "
           "cost sweep. **Mirage trigger, declared up front:** a fade *t* below 2, or a real fade that "
           "fails to beat the random-level placebo."),
        md("## 4 · The teardown\n\n**(a) Clustering** — per name and pooled chi-square:"),
        code(
            "rows = []\n"
            "for t, g, c in names:\n"
            "    r = strategy.clustering_test(c.to_numpy(), g)\n"
            "    rows.append([t, r['n'], r['chi2'], r['p'], r['frac_near']*100])\n"
            "allc = np.concatenate([c.to_numpy() for _, _, c in names])\n"
            "rp = strategy.clustering_test(allc, 1.0)\n"
            "rows.append(['POOLED($)', rp['n'], rp['chi2'], rp['p'], rp['frac_near']*100])\n"
            "tbl = pd.DataFrame(rows, columns=['name', 'n', 'chi2', 'p', 'within 5% (%)']).set_index('name')\n"
            "tbl"
        ),
        code(CLUSTER_FIG),
        md(
            "> 💡 **In plain words:** the single names (AAPL, MSFT) cluster hard; the index barely. "
            "That's the Harris (1991) shape — clustering is strongest in higher-priced, more-volatile "
            "single stocks. The *existence* of clustering is not in doubt."
        ),
        md("**(b) Forecast** — the fade, with HAC inference, vs the random-level placebo (200 seeds):"),
        code(FADE_FIG),
        md(
            "> 💡 **In plain words:** the real fade lands in the middle of the random-level cloud and "
            "leans negative. Round-ness adds no information a trader can bank; you'd do no better — "
            "slightly worse — fading arbitrary price levels."
        ),
        md("**Cost sweep** — there is no break-even, because the gross edge is already below zero:"),
        code(
            "real = strategy.pooled_fade(names, band=BAND, horizon=HORIZON)\n"
            "for cb in [0, 1, 2, 5, 10]:\n"
            "    m = strategy.net_of_costs(real, cost_bps=cb).mean()*1e4\n"
            "    print(f'cost {cb:2d} bps/leg -> {m:+7.2f} bps/trade')"
        ),
        md(
            "## 5 · The verdict\n\n"
            f"Signal `MIXED` (clustering real: chi² {R['chi2_pool']:.0f}, p ≈ {R['p_pool']}; forecast "
            f"none: fade {R['fade_bps']} bps, HAC *t* {R['fade_t']}). Tradability `MIRAGE` "
            f"({R['fade_net_bps']} bps/trade net). Beats a random level? `NOT SUPPORTED` "
            f"({R['beats_pct']}% of {R['n_seeds']} seeds). The harness *can* detect a planted bounce "
            "(the synthetic positive control banks it at HAC *t* ≈ +5.7) — there simply isn't one in the "
            "real tape."
        ),
        md(
            "## 6 · Could you trade it?\n\n"
            "Capacity is irrelevant when the gross edge is negative. The binding fact is economic: a "
            "lumpy price *distribution* implies nothing about the *conditional mean of the next return*. "
            "The round numbers mark where price rests, not where it travels — and a spread-paying fade of "
            "them is a slow bleed."
        ),
        md(
            "## 7 · Going further\n\n"
            "- Re-run on **intraday** bars (Harris's home turf) — does a higher-frequency fade survive?\n"
            "- Decompose clustering by **price decade** and **volatility** to reproduce the Harris "
            "cross-section explicitly.\n"
            "- Recast the round level as a **liquidity** marker (resting limit orders) rather than a "
            "directional one and test whether it predicts *spread*, not *return*."
        ),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "02_for_the_quants.ipynb")


def _meta():
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }


def _write(nb, name):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
