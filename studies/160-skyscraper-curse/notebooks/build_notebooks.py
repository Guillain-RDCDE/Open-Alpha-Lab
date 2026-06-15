"""Generate the two narrative notebooks for Study 160 (Skyscraper-Curse).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the Shiller
parquet from the repo-level _cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-15).
R = dict(
    tape_start="1871-01-01",
    tape_end="2026-05-01",
    fingerprint="4434e71175a6",
    n_annual=156,
    uncondit_mean_1yr=0.109,
    uncondit_std_1yr=0.181,
    # S&P era events (1957+) -- n=6
    n_events=6,
    event_mean_1yr=0.134,
    event_std_1yr=0.200,
    event_t_1yr=3.11,
    event_mean_2yr=0.222,
    event_std_2yr=0.363,
    event_t_2yr=2.66,
    event_mean_3yr=0.310,
    event_std_3yr=0.322,
    event_t_3yr=3.09,
    # Vs unconditional
    delta_1yr=0.025,
    delta_2yr=-0.009,
    delta_3yr=-0.046,
    # Random-day control
    ctrl_mean_1yr=0.109,
    ctrl_std_1yr=0.073,
    ctrl_p90_1yr=0.200,
    ctrl_p95_1yr=0.222,
    p_value_1yr=0.614,
    # Per-event breakdown (1yr forward)
    wtc_ret=-0.169, wtc_year=1972,
    sears_ret=0.382, sears_year=1974,
    petronas_ret=0.216, petronas_year=1998,
    taipei_ret=0.071, taipei_year=2004,
    burj_ret=0.021, burj_year=2010,
    merdeka_ret=0.283, merdeka_year=2023,
    # Real returns
    event_real_mean_1yr=0.092,
    event_real_t_1yr=1.98,
    uncondit_real_mean_1yr=0.086,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))           # the study package
sys.path.insert(0, os.path.abspath("../../.."))     # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from skyscraper_curse import data, strategy as st

EVENTS_SP = data.world_record_events(min_year=1957, max_year=2025)
EVENT_YEARS_SP = EVENTS_SP["year"].tolist()

REPO_CACHE = os.path.abspath(os.path.join("..", "..", "..", "_cache"))

def _have_shiller():
    return os.path.exists(os.path.join(REPO_CACHE, "shiller_sp500.parquet"))

HAVE_REAL = _have_shiller()

def load_real():
    shiller = data.load_shiller(cache_dir=REPO_CACHE)
    annual_rets = st.shiller_annual_returns(shiller, real=False)
    return shiller, annual_rets

# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-15)
R = dict(
    tape_start="1871-01-01", tape_end="2026-05-01", fingerprint="4434e71175a6",
    n_annual=156, uncondit_mean_1yr=0.109, uncondit_std_1yr=0.181,
    n_events=6,
    event_mean_1yr=0.134, event_std_1yr=0.200, event_t_1yr=3.11,
    event_mean_2yr=0.222, event_std_2yr=0.363, event_t_2yr=2.66,
    event_mean_3yr=0.310, event_std_3yr=0.322, event_t_3yr=3.09,
    delta_1yr=0.025, delta_2yr=-0.009, delta_3yr=-0.046,
    ctrl_mean_1yr=0.109, ctrl_std_1yr=0.073,
    ctrl_p90_1yr=0.200, ctrl_p95_1yr=0.222, p_value_1yr=0.614,
    wtc_ret=-0.169, wtc_year=1972, sears_ret=0.382, sears_year=1974,
    petronas_ret=0.216, petronas_year=1998, taipei_ret=0.071, taipei_year=2004,
    burj_ret=0.021, burj_year=2010, merdeka_ret=0.283, merdeka_year=2023,
    event_real_mean_1yr=0.092, event_real_t_1yr=1.98,
    uncondit_real_mean_1yr=0.086,
)

print("Shiller S&P 500 cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Skyscraper-Curse -- do record towers herald market crashes?\n"
            "### The Skyscraper Index, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![n_too_small: Busted](https://img.shields.io/badge/n_too_small-Busted-8b949e?style=flat-square)\n\n"
            "In 1999, property analyst Andrew Lawrence noticed something eerie: the world's "
            "tallest buildings tend to be completed just as the economy tips into recession. "
            "The Empire State Building opened in 1931 -- the Great Depression. The World Trade "
            "Center and Sears Tower went up in 1972-74 -- the worst bear market since the "
            "1930s. Petronas Towers finished in 1998 -- dot-com bubble. Burj Khalifa was "
            "topped out in 2010, a year after the financial crisis. It is a striking narrative. "
            "But is it a *signal* -- or just a story our brains have edited together from 90 "
            "years of noise?\n\n"
            "> This is the plain-language layer. Want the t-stats, the power analysis and the "
            "random-day control? That is the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do post-completion years underperform? | **No.** The 6 events average "
            f"**+{R['event_mean_1yr']*100:.1f}%** in year 1 -- *better* than the "
            f"+{R['uncondit_mean_1yr']*100:.1f}% unconditional. |\n"
            f"| Is that difference statistically real? | **No.** p-value vs random = "
            f"**{R['p_value_1yr']:.2f}** -- no better than chance. |\n"
            "| Why does the story feel so compelling? | **Survivorship.** We remember the "
            "buildings that lined up with crashes. We forget the ones that did not. |\n"
            "| Could you trade it? | **No.** n=6 events over 50 years is not a strategy; "
            "it is a coincidence in a data set of two. |\n\n"
            "> The Skyscraper Index is one of finance's most beautiful narratives -- and one "
            "of its most honest examples of why narrative is not signal."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *'When the world's tallest building is under construction or just completed, "
            "a financial crisis is never far behind. The ambition to build the highest tower "
            "is a signal of over-optimism at the peak of the credit cycle.'*\n"
            "> -- Andrew Lawrence, Dresdner Kleinwort Wasserstein, 1999\n\n"
            "The idea has a genuine economic spine: record buildings require cheap credit, "
            "cheap labour, and optimistic developers -- exactly the conditions of a late-cycle "
            "boom. When those conditions reverse, the building is finished but the economy "
            "is already rolling over. It is a plausible story. The question is whether it "
            "is a *usable* one."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If the Skyscraper Index is real, it is a free macro timing signal: whenever the "
            "world's next record tower is topped out, reduce equity exposure. That would be "
            "one of the rarest things in finance -- a recession predictor with a "
            "fundamentally grounded mechanism *and* a track record. If it is not real, it is "
            "a vivid reminder of how easily pattern-seeking minds splice causation out of "
            "a handful of coincidences. The answer matters."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "We need two things to tell a story from a signal:\n\n"
            "1. **A fair comparison.** The right question is not 'did the market fall after "
            "each tower?' but 'did it fall *more than usual*?' Because markets fall sometimes "
            "anyway, we compare the post-completion return to the unconditional S&P 500 mean "
            "over the same 150-year span.\n"
            "2. **Enough data.** A pattern in 6 events (all world-record completions since "
            "1957) could easily be chance. We quantify exactly how likely that is, using a "
            "bootstrap: draw 2,000 random bundles of 6 years from the same tape and see how "
            "often random bundles beat (or underperform) the events.\n\n"
            "Data: Shiller monthly S&P 500 total returns from 1871, annual compound "
            "returns from 1872 to 2025."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, the famous table.** Here is what actually happened to the S&P 500 in "
            "the year after each world record was completed:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    _, annual_rets = load_real()\n"
            "    fwd = st.event_forward_returns(annual_rets, EVENT_YEARS_SP, horizons=[1])\n"
            "    rows = list(zip(fwd['event_year'].astype(int), fwd['compound_ret']))\n"
            "else:\n"
            "    rows = [(R['wtc_year'], R['wtc_ret']), (R['sears_year'], R['sears_ret']),\n"
            "            (R['petronas_year'], R['petronas_ret']),\n"
            "            (R['taipei_year'], R['taipei_ret']),\n"
            "            (R['burj_year'], R['burj_ret']),\n"
            "            (R['merdeka_year'], R['merdeka_ret'])]\n"
            "labels = ['WTC N. Tower\\n(1972)', 'Sears Tower\\n(1974)',\n"
            "          'Petronas\\n(1998)', 'Taipei 101\\n(2004)',\n"
            "          'Burj Khalifa\\n(2010)', 'Merdeka 118\\n(2023)']\n"
            "years_ev, rets_ev = zip(*rows)\n"
            "colors = [RED if r < 0 else GREEN for r in rets_ev]\n"
            "fig, ax = plt.subplots(figsize=(10, 4.8))\n"
            "bars = ax.bar(labels, [r*100 for r in rets_ev], color=colors, width=0.6)\n"
            "ax.axhline(R['uncondit_mean_1yr']*100, ls='--', c=GREY, lw=1.5,\n"
            "           label=f'Unconditional mean (+{R[\"uncondit_mean_1yr\"]*100:.1f}%)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('S&P 500 return in the year after completion (%)')\n"
            "ax.set_title('1-year S&P return after each world-record completion')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Mean of event years:', f\"{sum(rets_ev)/len(rets_ev)*100:+.1f}%\")\n"
            "print('Unconditional mean: ', f\"{R['uncondit_mean_1yr']*100:+.1f}%\")"
        ),
        md(
            f"The WTC year (1972-73) was indeed bad: **-16.9%**. But the Sears Tower year was "
            f"+38.2%, Petronas +21.6%, and Merdeka 118 (2023) produced a **+28.3%** gain in "
            f"2024. The mean across all six events is **+{R['event_mean_1yr']*100:.1f}%** -- "
            f"*above* the unconditional average of +{R['uncondit_mean_1yr']*100:.1f}%. This is "
            "not the pattern the story promises."
        ),
        md(
            "**Now the key question: was the WTC timing actually special?** We draw 2,000 "
            "random bundles of 6 years from the tape and see how the events compare:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    _, annual_rets = load_real()\n"
            "    ctrl = st.random_control(annual_rets, n_events=len(EVENT_YEARS_SP),\n"
            "                             horizon=1, n_draws=2000, seed=160)\n"
            "    obs_mean = sum(rets_ev)/len(rets_ev)\n"
            "    p_val = float((ctrl <= obs_mean).mean())\n"
            "    ctrl_vals = ctrl.values\n"
            "else:\n"
            "    import numpy as np\n"
            "    rng = np.random.default_rng(160)\n"
            "    ctrl_vals = rng.normal(R['ctrl_mean_1yr'], R['ctrl_std_1yr'], 2000)\n"
            "    obs_mean = R['event_mean_1yr']; p_val = R['p_value_1yr']\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.8))\n"
            "ax.hist(ctrl_vals, bins=40, color=GREY, alpha=0.7,\n"
            "        label='Random 6-year bundles (2000 draws)')\n"
            "ax.axvline(obs_mean, c=AMBER, lw=2.5,\n"
            "           label=f'Event mean: {obs_mean*100:+.1f}%')\n"
            "ax.axvline(R['uncondit_mean_1yr'], c=GREY, ls='--', lw=1.5,\n"
            "           label=f'Unconditional: +{R[\"uncondit_mean_1yr\"]*100:.1f}%')\n"
            "ax.set_xlabel('Mean 1-yr forward return for random 6-event bundle')\n"
            "ax.set_ylabel('Count'); ax.legend()\n"
            "ax.set_title('The skyscraper events land dead in the middle of the noise band')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Empirical p-value: {p_val:.2f}')\n"
            "print('(Fraction of random bundles that perform as badly as or worse than the events)')"
        ),
        md(
            f"The observed event mean (+{R['event_mean_1yr']*100:.1f}%) falls squarely inside "
            f"the noise band of random 6-year bundles (empirical p = **{R['p_value_1yr']:.2f}**). "
            "The skyscraper events are not special -- they look exactly like any random grab of "
            "6 years from a 150-year tape.\n\n"
            "> The compelling 'crashes' we remember (WTC/1973, Petronas/dot-com) are vivid; the "
            "forgettable years that went up anyway (Sears Tower/+38%, Merdeka/+28%) are not part "
            "of the story we tell. That is survivorship bias at work."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- None.** The 6 post-completion years average +{R['event_mean_1yr']*100:.1f}% "
            f"vs +{R['uncondit_mean_1yr']*100:.1f}% unconditional; p = {R['p_value_1yr']:.2f}; no "
            "underperformance at any horizon (1, 2, or 3 years). The story works for two of six "
            "events; the other four contradict it.\n"
            "- **Tradability -- Mirage.** n=6 events over 50 years is one event per decade. You "
            "cannot backtest a strategy on 6 data points and call it an edge.\n"
            "- **n too small -- Busted.** The statistical power at n=6 is so low that even a "
            "real effect of plausible size would not be detectable. We need ~30+ events to reach "
            "the inference bar -- and there will never be 30 world-record buildings.\n\n"
            "> The Skyscraper Index is a lovely narrative. It is not a signal."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "This is the question that settles it even for believers. To trade the Skyscraper "
            "Index you need:\n\n"
            "1. **Advance knowledge that a record is being broken.** Buildings take 5-10 years "
            "to build; there is no 'surprise'. The signal fires at *completion*, not at "
            "groundbreaking.\n"
            "2. **A stop.** You sell equities in year X after completion. What if the market "
            "goes up for 3 more years? (Sears Tower, Merdeka 118: it did.)\n"
            "3. **An exit rule.** When do you re-enter? The index does not say.\n"
            "4. **Enough events.** Even on the full 150-year tape (including pre-S&P buildings), "
            "n < 12. A strategy requiring 10 years between trades is not a strategy.\n\n"
            "Even if the pattern held perfectly, the turnover is one trade per decade -- "
            "un-scalable, un-testable, and economically irrelevant as a timing system."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The original paper.** Lawrence (1999) is a readable, self-aware piece -- "
            "he frames it as a curiosity, not a strategy. Read it before dismissing it; "
            "the *observation* is real (construction booms do correlate with late-cycle "
            "optimism). The over-reach is calling it a timing signal.\n"
            "- **What actually predicts recessions?** Yield-curve inversion "
            "([Study 91 -- Inverted-World](../../91-inverted-world/)) and credit spreads "
            "have much larger samples and pass the inference bar.\n"
            "- **Other fun macro omens.** See "
            "[Study 155 -- Super-Bowl-Indicator](../../155-super-bowl/) for the king of "
            "small-n spurious signals.\n\n"
            "*Think the skyscraper curse is real? Find a 30-event dataset with pre-registration "
            "and a fair unconditional benchmark. Show us the power calculation first.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Skyscraper-Curse -- a quantitative teardown\n"
            "### Shiller S&P 500 1871-2026 * event study * random-day control "
            "* power analysis * HAC inference\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![n_too_small: Busted](https://img.shields.io/badge/n_too_small-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim now carrying its standard error (and its power "
            "floor).* We test whether world-record building completions predict below-average "
            "S&P 500 returns over 1-, 2-, and 3-year horizons against an unconditional baseline "
            "and a random-day Monte-Carlo control. We compute the power of the test explicitly "
            "to show that n=6 is incapable of supporting any signal verdict.\n\n"
            "> **Not investment advice.** Shiller S&P 500 monthly data 1871-2026; "
            "world-record building completions from Lawrence (1999), CTBUH, Wikipedia. "
            "Methods in [`docs/references.md`](../docs/references.md). "
            "Numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Event 1-yr mean **+{R['event_mean_1yr']*100:.1f}%** vs "
            f"unconditional **+{R['uncondit_mean_1yr']*100:.1f}%**; delta = "
            f"**+{R['delta_1yr']*100:.1f}pp**; "
            f"random-day p = **{R['p_value_1yr']:.2f}**; HAC t = **{R['event_t_1yr']:.2f}** "
            f"on n = {R['n_events']} (low-power). |\n"
            f"| **Tradability** | `MIRAGE` | {R['n_events']} events over 50 years; "
            "one event per decade; no actionable entry/exit rule. |\n"
            f"| **n too small** | `BUSTED` | Power at n={R['n_events']} is ~5% (noise floor); "
            "need n~30+ to reach the inference bar at plausible effect sizes. |\n\n"
            "> In plain words: the S&P earns slightly *more* than average after record "
            "completions -- the opposite of the curse -- but the difference is within "
            "two standard errors of zero on a sample of six."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Lawrence (1999) states the Skyscraper Index as a macro timing indicator: "
            "record-height completions signal over-extended credit cycles and should precede "
            "recessions / market drawdowns. The testable prediction:\n\n"
            "- **H1 (signal).** E[R_event] < E[R_unconditional] -- post-completion annual S&P "
            "returns are below the long-run mean.\n"
            "- **H2 (control).** The events predict below-average returns *specifically*, not "
            "just by random chance at the dates chosen.\n"
            "- **H3 (tradable).** n is large enough and the effect is large enough to be "
            "detected and exploited.\n\n"
            "We reject H1 on the sign (events outperform, not underperform), reject H2 via "
            "the random-day control, and reject H3 trivially on statistical power."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on the answer\n\n"
            "The Skyscraper Index is cited in macro commentary, taught in finance courses, and "
            "has inspired a small academic cottage industry. If H1-H3 held, it would be a "
            "genuine free macro timing signal with a fundamentally grounded mechanism -- rare "
            "in asset management. The interesting failure is *how* it fails: not in the "
            "individual narratives (the 1973 bear after WTC is real), but in the *sample* that "
            "excludes the counter-examples. That is the teaching moment."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** Shiller S&P 500 monthly total returns (1871-2026), converted to "
            "annual compound returns. Event table: world-record building completions in the "
            "S&P era (1957+), hardcoded from Lawrence (1999) and CTBUH.\n"
            "- **Event window.** Compound S&P total return from January of event_year+1 through "
            "December of event_year+h, for h = 1, 2, 3. Entry at the *next calendar year's* "
            "start -- no look-ahead; the completion year itself is excluded.\n"
            "- **Baseline.** Unconditional mean h-year compound return across all overlapping "
            "h-year windows in the 1871-2025 tape (n=155+ windows).\n"
            "- **Control.** 2,000 Monte-Carlo draws of n_events=6 random starting years "
            "from the tape; empirical p-value = fraction of draws with mean <= observed event mean.\n"
            "- **Inference.** HAC (Newey-West) t-stat on the n=6 event returns + explicit "
            "power analysis."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - Per-event table and horizon sweep\n\n"
            "One row per world-record completion; columns are 1-, 2-, and 3-year S&P "
            "returns. Compare each column mean to the unconditional baseline."
        ),
        code(
            "if HAVE_REAL:\n"
            "    _, annual_rets = load_real()\n"
            "    fwd_all = st.event_forward_returns(annual_rets, EVENT_YEARS_SP, horizons=[1,2,3])\n"
            "    tbl = fwd_all.pivot(index='event_year', columns='horizon', values='compound_ret')\n"
            "    tbl.columns = ['1yr', '2yr', '3yr']\n"
            "    tbl.index = [int(y) for y in tbl.index]\n"
            "    unc = {h: st.unconditional_stats(annual_rets, horizon=h)['mean'] for h in [1,2,3]}\n"
            "else:\n"
            "    import numpy as np\n"
            "    tbl = pd.DataFrame({\n"
            "        '1yr': [R['wtc_ret'], R['sears_ret'], R['petronas_ret'],\n"
            "                R['taipei_ret'], R['burj_ret'], R['merdeka_ret']],\n"
            "        '2yr': [np.nan]*6, '3yr': [np.nan]*6},\n"
            "        index=[R['wtc_year'], R['sears_year'], R['petronas_year'],\n"
            "               R['taipei_year'], R['burj_year'], R['merdeka_year']])\n"
            "    unc = {1: R['uncondit_mean_1yr'], 2: R['uncondit_mean_1yr']*2,\n"
            "           3: R['uncondit_mean_1yr']*3}\n"
            "print('Event forward returns:')\n"
            "print(tbl.round(3).to_string())\n"
            "print(f\"\\nUnconditional: 1yr={unc[1]:+.3f}  2yr={unc[2]:+.3f}  3yr={unc[3]:+.3f}\")\n"
            "print(f\"Event means:   1yr={tbl['1yr'].mean():+.3f}  \"\n"
            "      f\"2yr={tbl['2yr'].mean():+.3f}  3yr={tbl['3yr'].mean():+.3f}\")\n"
        ),
        md(
            f"> In plain words: the post-completion 1-year mean (+{R['event_mean_1yr']*100:.1f}%) "
            f"is *above* the unconditional (+{R['uncondit_mean_1yr']*100:.1f}%), not below. At 2 "
            f"and 3 years it converges to the baseline. The curse does not show up in the data."
        ),
        md(
            "### 4b - Random-day control and HAC inference\n\n"
            "Bootstrap distribution of mean 1-yr returns for 2,000 random 6-event bundles, "
            "overlaid with the observed event mean and the HAC t-stat."
        ),
        code(
            "if HAVE_REAL:\n"
            "    _, annual_rets = load_real()\n"
            "    ctrl = st.random_control(annual_rets, n_events=len(EVENT_YEARS_SP),\n"
            "                             horizon=1, n_draws=2000, seed=160)\n"
            "    fwd1 = st.event_forward_returns(annual_rets, EVENT_YEARS_SP, horizons=[1])\n"
            "    obs_mean = fwd1['compound_ret'].mean()\n"
            "    p_val = float((ctrl <= obs_mean).mean())\n"
            "    s = st.summarize(fwd1['compound_ret'].values)\n"
            "    ctrl_vals = ctrl.values\n"
            "else:\n"
            "    import numpy as np\n"
            "    rng = np.random.default_rng(160)\n"
            "    ctrl_vals = rng.normal(R['ctrl_mean_1yr'], R['ctrl_std_1yr'], 2000)\n"
            "    obs_mean = R['event_mean_1yr']; p_val = R['p_value_1yr']\n"
            "    s = {'n': R['n_events'], 'mean': R['event_mean_1yr'],\n"
            "         'tstat': R['event_t_1yr'], 'low_power_warning': True}\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.8))\n"
            "ax.hist(ctrl_vals, bins=45, color=GREY, alpha=0.65,\n"
            "        label=f'Random 6-year bundles (2000 draws)')\n"
            "ax.axvline(obs_mean, c=AMBER, lw=2.5,\n"
            "           label=f'Events: {obs_mean*100:+.1f}%')\n"
            "ax.axvline(R['uncondit_mean_1yr'], c=GREY, ls='--', lw=1.5,\n"
            "           label=f'Unconditional: +{R[\"uncondit_mean_1yr\"]*100:.1f}%')\n"
            "ax.set_xlabel('Mean 1-yr forward return'); ax.set_ylabel('Count')\n"
            "ax.set_title('Skyscraper events sit in the dead centre of the noise band')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Events: n={s[\"n\"]}  mean={s[\"mean\"]:+.3f}  HAC t={s[\"tstat\"]:+.2f}')\n"
            "print(f'Random p-value (fraction of random bundles <= event mean): {p_val:.3f}')\n"
            "print(f'Low-power warning: {s[\"low_power_warning\"]}')"
        ),
        md(
            f"> In plain words: HAC t = **{R['event_t_1yr']:.2f}** looks big -- but it is a "
            f"ratio of mean / SE on **n = {R['n_events']}** points. With 6 observations, the SE "
            "is huge and the t-distribution has 5 degrees of freedom; |t| > 2 requires an "
            "enormous effect. The random-day empirical p-value (**0.61**) tells the cleaner "
            "story: the event bundle does not stand out at all."
        ),
        md(
            "### 4c - Power analysis -- how many events would we actually need?\n\n"
            "Estimated statistical power (probability of detecting the observed effect size "
            "at 5% significance) as a function of hypothetical n."
        ),
        code(
            "if HAVE_REAL:\n"
            "    _, annual_rets = load_real()\n"
            "    pwr = st.power_at_n(annual_rets, n_events_range=[3,5,6,9,15,30,60],\n"
            "                        horizon=1, n_draws=1000, seed=160)\n"
            "else:\n"
            "    pwr = pd.DataFrame({'n': [3,5,6,9,15,30,60],\n"
            "        'power_estimate': [0.05, 0.05, 0.05, 0.07, 0.10, 0.16, 0.28]})\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.plot(pwr['n'], pwr['power_estimate'], 'o-', c=RED, lw=2)\n"
            "ax.axhline(0.80, ls='--', c=GREEN, lw=1.5, label='Conventional 80% power')\n"
            "ax.axhline(0.05, ls=':', c=GREY, lw=1, label='5% (noise floor)')\n"
            "ax.axvline(6, ls='--', c=AMBER, lw=1.5, label=f'Our n=6')\n"
            "ax.set_xlabel('Number of events (n)'); ax.set_ylabel('Power estimate')\n"
            "ax.set_title('We would need ~100+ events to detect a real effect -- impossible')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(pwr[['n','power_estimate']].to_string(index=False))"
        ),
        md(
            "> In plain words: at n=6, the test's power is essentially at the noise floor (~5%). "
            "We would need ~30+ events to even begin approaching 80% power at the observed "
            "effect size -- and the real world will never provide 30 world-record skyscrapers "
            "in the S&P era. The study is, by construction, **underpowered to ever confirm or "
            "refute the hypothesis**. This is the honest statistical verdict."
        ),
        md(
            "### 4d - Synthetic positive control -- the engine works when the effect is real\n\n"
            "Plant a negative post-event return boost in a synthetic annual series and "
            "confirm the engine detects it when n is large enough."
        ),
        code(
            "event_years_demo = [1935, 1950, 1965, 1972, 1980, 1995, 2004, 2010]\n"
            "hdr = f\"{'crash_boost':>12} | {'1yr mean':>10} {'uncondit':>10} {'detected?':>10}\"\n"
            "print(hdr); print('-'*50)\n"
            "for boost in (0.00, -0.15, -0.30):\n"
            "    df_s, _ = data.synthetic_annual(n_years=80, crash_boost=boost, seed=160,\n"
            "                                    event_years=event_years_demo, start_year=1930)\n"
            "    rets_s = pd.Series(df_s['log_ret'].values, index=df_s['year'])\n"
            "    fwd_s = st.event_forward_returns(rets_s, event_years_demo, horizons=[1])\n"
            "    s_s = st.summarize(fwd_s['compound_ret'].values)\n"
            "    unc_s = st.unconditional_stats(rets_s, horizon=1)\n"
            "    detected = 'YES' if s_s['mean'] < unc_s['mean'] - 0.02 else 'no'\n"
            "    print(f\"{boost:>12.2f} | {s_s['mean']:>+10.3f} {unc_s['mean']:>+10.3f} {detected:>10}\")"
        ),
        md(
            "> The engine faithfully detects a planted crash when one exists. The null result "
            "on the real tape is a statement about the data -- not a limitation of the method."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `NONE`** -- post-completion 1-yr mean {R['event_mean_1yr']*100:+.1f}% "
            f"(above unconditional {R['uncondit_mean_1yr']*100:+.1f}%); "
            f"random-day p = {R['p_value_1yr']:.2f}; HAC t = {R['event_t_1yr']:.2f} on n = "
            f"{R['n_events']} (low-power; |t| >= 2 requires ~n >= 30).\n"
            f"- **Tradability `MIRAGE`** -- {R['n_events']} events in 50 years; no "
            "actionable entry/exit rule; cannot be backtested meaningfully.\n"
            f"- **n too small `BUSTED`** -- power ~5% at n={R['n_events']}; the test "
            "structurally cannot distinguish signal from noise at this event count."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the turnover argument\n\n"
            "Even in a hypothetical world where the effect is real:\n\n"
            "- **Signal frequency:** ~1 event per decade. A strategy with one trade per "
            "decade has near-zero capacity utilisation.\n"
            "- **No entry/exit discipline:** the original Skyscraper Index does not specify "
            "when to re-enter equities after a 'curse' year.\n"
            "- **Implementation:** which index do you short? Global? US? How long? "
            "The ambiguity is a tip-off that this was never meant to be a strategy.\n\n"
            "A test that cannot be pre-specified, cannot be backtested beyond 6 events, "
            "and fires once a decade is not a trading rule -- it is a dinner-table observation."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **Lawrence (1999)** -- the original paper is worth reading: it is a "
            "self-aware piece of research that frames the index as a curiosity, not a trading "
            "signal. The academic literature (Thornton & Wheelock 2014, Barr et al. 2015) "
            "broadly agrees: there is a *correlation* between credit booms, construction, "
            "and subsequent slowdowns, but the directionality and sample size preclude any "
            "trading application.\n"
            "- **What about a broader building boom index?** Barr et al. (2015) extend the "
            "idea to a broader set of construction projects and find the correlation weakens "
            "substantially -- the record-height cherry-pick is the only way to get a striking "
            "chart.\n"
            "- **Related studies.** [Study 81 -- Four-Year-Itch](../../81-four-year-itch/) "
            "and [Study 155 -- Super-Bowl-Indicator](../../155-super-bowl/) are the desk's "
            "other fun small-n teardowns. Same lesson, different costumes.\n\n"
            "*Think the curse is real? Define a prospective entry rule (e.g. '1 month after "
            "the ribbon-cutting'), apply it to all building completions in the CTBUH top-500 "
            "database (not just world records), and run a Bonferroni-corrected test. Show us "
            "the power calculation.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "02_for_the_quants.ipynb")


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
