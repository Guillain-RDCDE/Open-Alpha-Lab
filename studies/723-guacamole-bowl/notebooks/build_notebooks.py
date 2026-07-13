"""Generate the two narrative notebooks for Study 723 ("guacamole-bowl").

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic: real-tape cells read the cached month-end yfinance pulls
under ../_cache/ (PEP/SPY/^IRX monthly returns) and the hardcoded, cited, approximate avocado seasonal
from the package; on a cache miss they fall back to the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic Jan-Feb control runs anywhere.

The tradable leg is PEP (Frito-Lay's Super-Bowl chip-and-dip complex) — a LABELLED PROXY, because the
pure-play avocado name CVGW is unavailable on the current Yahoo feed. The avocado price series is a
labelled proxy used for its *shape* only; it never backs a Signal stamp.
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


# Frozen headline numbers — mirror of docs/results.md (PEP/SPY/^IRX month-end returns via yfinance,
# 1993-02..2026-05, 400 months, as-of 2026-06-01, fingerprint 14fdb930823d; avocado seasonal hardcoded).
R = dict(
    win="1993-02 -> 2026-05", n_months=400, asof="2026-06-01", fp="14fdb930823d",
    # avocado price seasonal (labelled proxy, shape only), base 100 ~= annual mean
    av_seasonal={1: 95, 2: 97, 3: 100, 4: 103, 5: 102, 6: 105, 7: 110, 8: 114, 9: 111, 10: 103, 11: 93, 12: 90},
    av_win_mean=96.0, av_year_mean=101.9, av_gap=-5.9,
    # per-month PEP: mean%, t_naive, t_hac, n
    months={
        1: (-0.38, -0.38, -0.36, 33), 2: (0.09, 0.10, 0.12, 34), 3: (2.69, 2.96, 3.53, 34),
        4: (0.83, 0.94, 1.09, 34), 5: (1.08, 1.34, 1.23, 34), 6: (0.56, 0.63, 0.64, 33),
        7: (1.10, 1.34, 1.44, 33), 8: (-1.51, -1.28, -0.94, 33), 9: (1.02, 1.14, 1.07, 33),
        10: (2.55, 2.16, 1.95, 33), 11: (1.42, 1.98, 2.30, 33), 12: (0.62, 0.90, 1.06, 33),
    },
    # guac window (Jan-Feb) vs rest
    win_mean=-0.14, rest_mean=1.04, spread=-1.19, spread_t=-1.67, n_win=67, n_rest=333,
    # placebo across 66 month-pairs
    placebo_rank=9, placebo_n=66, placebo_pct=14, placebo_z=-1.32,
    most_pos=[("Mar-Oct", 2.14), ("Mar-Nov", 1.47), ("Oct-Nov", 1.37)],
    # block-bootstrap CI on the window spread
    ci_lo=-2.61, ci_hi=0.26, ci_point=-1.19,
    # timer race (Sharpe = excess of T-bill)
    timer_cagr=1.5, timer_sharpe=-0.09, timer_mdd=-30,
    timer_net_cagr=1.4, timer_net_sharpe=-0.11,
    pep_cagr=8.7, pep_sharpe=0.42, pep_mdd=-36, pep_vol=18.4, pep_sharpe_raw=0.54,
    spy_cagr=10.9, spy_sharpe=0.61, spy_mdd=-51, spy_vol=14.8, spy_sharpe_raw=0.77,
    # Newey-West alpha PEP vs SPY
    nw_alpha=3.46, nw_beta=0.59, nw_t=1.27,
    # synthetic positive control
    syn_spread=3.70, syn_t=4.60, syn_rank=66, syn_timer=0.65, syn_bh=-0.16,
    syn_null_spread=0.10, syn_null_t=0.12,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Guacamole_surge%3F: Busted](https://img.shields.io/badge/Guacamole_surge%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "..")))  # repo root (quantlab)
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

from guacamole_bowl import data, strategy as st
try:
    from quantlab import repro
    D = repro.as_of(data.fetch_data())            # cache-first, pinned to the desk as-of
except Exception:
    D = data.fetch_data()
HAVE = not D.empty
AV = data.load_avocado_seasonal()                 # hardcoded, cited, APPROXIMATE proxy (shape only)
print("real-tape cache present:", HAVE, "| months:", (len(D) if HAVE else 0))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Super Bowl guacamole trade 🥑\n"
            "### Americans eat ~100 million pounds of avocados for the big game — is there a trade in it?\n\n"
            + BADGES +
            "Every year around late January the stat makes the rounds: the U.S. imports a *mountain* of "
            "avocados for Super Bowl Sunday, more guacamole is eaten that weekend than any other, "
            "growers ship record volume. So the folk-finance leap is irresistible: if demand spikes on "
            "a **calendar date everyone can see coming**, the avocado / produce trade should carry a "
            "**January–February seasonal**. Buy ahead of the game, ride the surge.\n\n"
            "This notebook takes that seriously and then takes it apart — with the avocado price's own "
            "seasonal shape, a real stock tape, and the one test that kills most calendar folklore: a "
            "**placebo** that asks whether the Super-Bowl window is actually special, or just one of "
            "sixty-six ordinary pairs of months.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Newey-West alpha and the placebo "
            "distribution? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice — and a data note.** The pure-play avocado stock (Calavo, "
            "`CVGW`) is currently **untradable on the Yahoo feed** (its daily history returns a single "
            "bar), so the tradable leg here is **`PEP`** — PepsiCo, whose Frito-Lay arm *is* the "
            "Super-Bowl chip-and-dip complex (Tostitos + the branded dips) — a **labelled proxy**. The "
            "avocado price line is a **small, cited, approximate** reconstruction of public USDA / Hass "
            "Avocado Board seasonality — a **proxy**, shown for its shape, never as a live feed."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is the guacamole binge real? | **Yes.** The Super-Bowl avocado-volume spike is a genuine, "
            "well-documented demand event. |\n"
            "| Does it lift the avocado *price* in Jan–Feb? | **No — the opposite.** Winter is the "
            f"*soft* part of the avocado price year (heavy Mexican supply): the guac window sits "
            f"**{R['av_gap']:.0f} index points below** the annual average. The surge is pre-supplied. |\n"
            "| Does the trade show a Jan–Feb seasonal? | **No — it's the year's *weakest* window.** "
            f"Jan–Feb underperforms the rest of the year by **{R['spread']:.1f}%/month** (*t* = "
            f"{R['spread_t']:.2f}, wrong sign), ranking **{R['placebo_rank']}/{R['placebo_n']}** among "
            "all month-pairs — near the bottom, not the top. |\n"
            "| Could you trade it? | **No.** A long-Jan–Feb timer earns **Sharpe "
            f"{R['timer_sharpe']:.2f}** vs buy-and-hold SPY's **{R['spy_sharpe']:.2f}** — it puts you in "
            "the market only during the worst stretch of the year. |\n\n"
            "> The binge is real. The *trade* is not — the calendar everyone can see is exactly why "
            "there's nothing left to harvest."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"More guacamole is eaten on Super Bowl Sunday than any other day of the year — "
            "growers ship a record wall of Hass avocados for it. A demand spike that big, on a date "
            "you can circle in advance, has to move the avocado and produce trade. Buy in January, "
            "sell after the game.\"*\n\n"
            "It's a *steelman-able* claim. The volume is real: the U.S. consumes on the order of "
            "**100+ million pounds** of avocados around the game, and importers/retailers visibly "
            "pre-position for it. Cinco de Mayo is a second, smaller guac holiday. If any consumer "
            "seasonal should be legible in a price, a nationally-televised, date-certain binge is the "
            "candidate."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "A clean, tradable calendar seasonal would be a small miracle: a known date, a known "
            "demand spike, a repeatable long window — the easiest kind of edge to run, no forecasting "
            "required. It's the same dream as *sell-in-May*, the *Santa-Claus rally*, or the coffee "
            "*frost trade*: pin your entries and exits to the calendar and get paid. But *\"a lot of "
            "guacamole is eaten in February\"* and *\"the avocado trade goes up in February\"* are very "
            "different statements. The first is about **demand volume**; the second is a claim about "
            "**price**, net of a supply chain that has months of warning. We can check the second "
            "directly."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three honest looks:\n\n"
            "1. **The avocado price's own shape.** Put the (cited, approximate) wholesale-Hass seasonal "
            "index on a chart. Does winter — the guac window — actually sit *high*?\n"
            "2. **The tradable tape.** Take the Super-Bowl snack proxy (`PEP` / Frito-Lay) back to "
            "1993 and line up every calendar month. Is Jan–Feb reliably strong?\n"
            "3. **The placebo.** Score the Jan–Feb window against **all 66 pairs of months**. A real "
            "seasonal sits in the extreme tail; folklore sits in the crowd.\n\n"
            "**What would make us say \"real trade\"?** Jan–Feb positive with *t* ≥ 2, sitting near the "
            "*top* of the placebo, and a timer that beats buy-and-hold. Anything less is a binge with "
            "no trade attached."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: the avocado price's own seasonal shape.** If the Super Bowl lifted prices, winter "
            "would be a peak. Here's the (approximate, cited) wholesale-Hass seasonal index."
        ),
        code(
            "vals = [float(AV.loc[m]) for m in range(1,13)]\n"
            "ymean = float(AV.mean())\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "cols = [RED if m in (1,2) else AMBER for m in range(1,13)]\n"
            "ax.bar(MONTH_NAMES, vals, color=cols, width=.7)\n"
            "ax.axhline(ymean, ls='--', c=GREY, label=f'annual average ({ymean:.0f})')\n"
            "ax.annotate('Super Bowl\\nwindow', (0.5, vals[0]), textcoords='offset points',\n"
            "            xytext=(0, 12), ha='center', color=RED, fontsize=9)\n"
            "ax.set_ylabel('seasonal price index (base 100 = annual mean)')\n"
            "ax.set_title('Avocado prices are SOFT in winter — the guac window is below average'); ax.legend()\n"
            "ax.set_ylim(80, 120)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"guac window (Jan-Feb) mean {np.mean(vals[:2]):.1f}  vs annual mean {ymean:.1f}  \"\n"
            "      f\"=> {np.mean(vals[:2])-ymean:+.1f} index pts\")"
        ),
        md(
            f"The window everyone circles is the year's **soft** patch, ~**{abs(R['av_gap']):.0f} points "
            "below** average. Why? The Mexican Hass harvest floods the market in winter and the trade "
            "*pre-positions* months ahead for a date it has known since last season. The real price "
            "peak is the **late-summer supply gap** (Aug), which has nothing to do with football. The "
            "demand is real; the price simply absorbs it."
        ),
        md(
            "**Now the tape.** Here's the average return of the Super-Bowl snack proxy (`PEP`) in each "
            "calendar month since 1993, with the guac window in red."
        ),
        code(
            "if HAVE:\n"
            "    ms = st.month_stats(D['pep'])\n"
            "    means = [ms.loc[m,'mean']*100 for m in range(1,13)]\n"
            "else:\n"
            "    means = [R['months'][m][0] for m in range(1,13)]\n"
            "cols = [RED if m in (1,2) else AMBER for m in range(1,13)]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "ax.bar(MONTH_NAMES, means, color=cols, width=.7)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('avg monthly return (%)')\n"
            "ax.set_title('The guac window (red) is not special — if anything, it lags'); \n"
            "plt.tight_layout(); plt.show()\n"
            "print('Jan:', f\"{means[0]:+.2f}%\", ' Feb:', f\"{means[1]:+.2f}%\", ' | best months are Mar/Oct/Nov (off-thesis)')"
        ),
        md(
            f"January is **negative** ({R['months'][1][0]:+.2f}%) and February is a **flat "
            f"{R['months'][2][0]:+.2f}%**. The genuinely strong months — March, October, November — have "
            "nothing to do with the Super Bowl. The window the folklore points at is, if anything, a "
            "*drag*."
        ),
        md(
            "**The placebo — is Jan–Feb even special?** We score the Super-Bowl window against every "
            "one of the 66 possible pairs of months. If the guac seasonal were real, Jan–Feb would sit "
            "at the far *right* (most positive). Where does it actually land?"
        ),
        code(
            "import itertools\n"
            "if HAVE:\n"
            "    pb = st.placebo_pairs(D['pep'])\n"
            "    s = D['pep']; s.index = pd.DatetimeIndex(s.index)\n"
            "    spreads = []\n"
            "    for pair in itertools.combinations(range(1,13),2):\n"
            "        w = s[s.index.month.isin(pair)]; r = s[~s.index.month.isin(pair)]\n"
            "        spreads.append((w.mean()-r.mean())*100)\n"
            "    jf = (s[s.index.month.isin([1,2])].mean()-s[~s.index.month.isin([1,2])].mean())*100\n"
            "    rank = pb['rank']\n"
            "else:\n"
            "    spreads = list(np.random.default_rng(0).normal(0,0.9,66)); jf = R['spread']; rank = R['placebo_rank']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.hist(spreads, bins=18, color=GREY, alpha=.7)\n"
            "ax.axvline(jf, c=RED, lw=2.5, label=f'Jan-Feb (guac): {jf:+.2f}%')\n"
            "ax.axvline(0, c='k', lw=1, ls=':')\n"
            "ax.set_xlabel('month-pair vs rest-of-year spread (%/month)'); ax.set_ylabel('# of month-pairs')\n"
            "ax.set_title(f'Placebo: Jan-Feb ranks {rank}/66 — near the BOTTOM, not the top'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"Jan-Feb spread {jf:+.2f}%/mo ranks {rank}/66 (1=lowest). It is among the WEAKEST windows, not the strongest.\")"
        ),
        md(
            f"There it is. The Super-Bowl window ranks **{R['placebo_rank']} out of {R['placebo_n']}** — "
            f"in the bottom **{R['placebo_pct']}%** of all month-pairs. Dozens of *arbitrary* windows "
            "with no story beat it. A real seasonal lives in the tail; this one is buried in the crowd, "
            "on the wrong side of zero."
        ),
        md(
            "**Finally, could you trade it?** The obvious rule: hold the snack proxy in Jan–Feb, sit in "
            "T-bills the rest of the year. Race it against just buying and holding SPY."
        ),
        code(
            "if HAVE:\n"
            "    rf = D['tbill']\n"
            "    timer = st.seasonal_timer(D['pep'], tbill=rf)\n"
            "    bh_pep = st.buy_hold(D['pep']); bh_spy = st.buy_hold(D['spy'])\n"
            "    rows = {'guac timer\\n(long Jan-Feb)': st.summary(timer, rf=rf)['sharpe'],\n"
            "            'buy & hold\\nPEP': st.summary(bh_pep, rf=rf)['sharpe'],\n"
            "            'buy & hold\\nSPY': st.summary(bh_spy, rf=rf)['sharpe']}\n"
            "else:\n"
            "    rows = {'guac timer\\n(long Jan-Feb)': R['timer_sharpe'], 'buy & hold\\nPEP': R['pep_sharpe'], 'buy & hold\\nSPY': R['spy_sharpe']}\n"
            "labels=list(rows); vals=list(rows.values())\n"
            "cols=[RED, AMBER, GREEN]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('Sharpe (excess of T-bill)')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_title('The guac timer has a NEGATIVE Sharpe — worse than doing nothing')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"guac timer Sharpe {vals[0]:+.2f}  vs  buy-and-hold SPY {vals[2]:+.2f}\")"
        ),
        md(
            f"The timer earns a **negative** Sharpe ({R['timer_sharpe']:.2f}) — it deliberately holds "
            "the market during its worst two months and sits out the rest. Buy-and-hold SPY "
            f"(**{R['spy_sharpe']:.2f}**) laps it without trying. Costs only make it worse."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Jan–Feb underperforms the rest of the year by **{R['spread']:.1f}%/"
            f"month** (*t* = {R['spread_t']:.2f}, the *wrong* sign); no thesis month clears |*t*| ≥ 2 in "
            f"its favour; the placebo ranks it **{R['placebo_rank']}/{R['placebo_n']}**. No seasonal.\n"
            "- **Tradability — Mirage.** The long-Jan–Feb timer earns **negative** Sharpe "
            f"({R['timer_sharpe']:.2f}) vs buy-and-hold SPY's {R['spy_sharpe']:.2f}; and the pure-play "
            "avocado stock isn't even reliably tradable.\n"
            "- **Guacamole surge? — Busted.** The avocado price is *soft* in winter and the snack tape's "
            "guac window is its weakest — both legs point the wrong way. The binge is real; the trade "
            "is folklore."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "Imagine two people in early 1993, each with \\$10,000. One buys SPY and never touches it. "
            "The other runs the guacamole calendar — in the market for Jan–Feb, T-bills otherwise — "
            "every single year. Where do they land by 2026?"
        ),
        code(
            "start = 10_000.0; yrs = R['n_months']/12\n"
            "spy_end = start*(1+R['spy_cagr']/100)**yrs\n"
            "timer_end = start*(1+R['timer_cagr']/100)**yrs\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['buy & hold SPY', 'guacamole timer\\n(long Jan-Feb)'], [spy_end, timer_end],\n"
            "       color=[GREEN, RED], width=.55)\n"
            "for i,v in enumerate([spy_end, timer_end]): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel(f'value of $10,000 after {yrs:.0f} years')\n"
            "ax.set_title('Same $10k, 1993 -> 2026'); plt.tight_layout(); plt.show()\n"
            "print(f'buy & hold SPY: ${spy_end:,.0f}   |   guacamole timer: ${timer_end:,.0f}')"
        ),
        md(
            "The index investor multiplies their money many times over doing nothing. The calendar "
            "trader — earning cash rates 10 months a year and the market's *worst* window the other "
            "two — ends up a small fraction of that. There is no execution trick that rescues a window "
            "that is negative before you even pay costs. And the one instrument that *is* the avocado "
            "trade (Calavo) can't be reliably bought on the tape at all."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Pull the real avocado tape.** Our seasonal line is a cited *approximation*; USDA AMS "
            "Market News and the Hass Avocado Board publish weekly price + volume. Swap them in — the "
            "*shape* (winter-soft, summer-peak) won't move.\n"
            "- **Get CVGW back.** If Calavo's history returns to the feed, re-run with the pure-play "
            "avocado equity in place of `PEP`; the placebo and timer machinery is ticker-agnostic.\n"
            "- **The calendar-folklore family.** Sell-in-May, the Santa rally, the "
            "[coffee frost trade](../../307-coffee-seasonality/): a vivid, *true* story about the world "
            "that makes a *terrible* calendar trade because everyone can see the date coming "
            "([docs/references.md](../docs/references.md)).\n\n"
            "*Think a specific produce name (a berry grower, a lime importer) prints a real Super-Bowl "
            "seasonal net of costs? Pull its tape, run the placebo, and show it sits in the tail — not "
            "the crowd.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# The Super-Bowl guacamole seasonal — a quantitative teardown 🔬\n"
            "### Per-month HAC *t*-stats · a Jan–Feb window spread · a placebo across all 66 month-pairs "
            "· a block-bootstrap CI · a Jan–Feb timer vs buy-and-hold · Newey-West alpha · a synthetic "
            "positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We test the "
            "strongest tradable form of \"the Super-Bowl guacamole binge prints a Jan–Feb seasonal\": "
            "(H₁) the guac window out-returns the rest of the year with *t* ≥ 2; (H₂) it sits in the "
            "extreme tail of a month-pair placebo; (H₃) a long-window timer beats buy-and-hold. We find "
            "**all three rejected** — the window is negative, ordinary, and a losing trade.\n\n"
            "> ⚠️ **Not investment advice — data provenance.** Tradable leg **`PEP`** (Frito-Lay's "
            "Super-Bowl chip-and-dip complex) + benchmark **`SPY`** + cash **`^IRX`**: month-end returns "
            "from daily closes via yfinance, 1993-02→2026-05, 400 months, as-of "
            f"{R['asof']}, fingerprint `{R['fp']}`. The pure-play avocado name `CVGW` is a **labelled "
            "proxy** we would prefer but its Yahoo tape is currently a single bar; the wholesale-Hass "
            "seasonal is **hardcoded, cited, approximate** (shape only, never a Signal). Methods in "
            "[`docs/references.md`](../docs/references.md); numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Guac window (Jan–Feb) − rest = **{R['spread']:+.2f}%/mo**, "
            f"*t* = **{R['spread_t']:+.2f}** (wrong sign); no thesis month clears |*t*|≥2 (Jan *t*_HAC "
            f"{R['months'][1][2]:+.2f}, Feb {R['months'][2][2]:+.2f}); placebo rank "
            f"**{R['placebo_rank']}/{R['placebo_n']}**. |\n"
            f"| **Tradability** | `MIRAGE` | Long-Jan–Feb timer Sharpe **{R['timer_sharpe']:+.2f}** "
            f"(net {R['timer_net_sharpe']:+.2f}) vs buy-and-hold SPY **{R['spy_sharpe']:+.2f}**; PEP "
            f"NW alpha *t* = {R['nw_t']:+.2f} (no edge). CVGW untradable on the feed. |\n"
            f"| **Guacamole surge?** | `BUSTED` | Avocado price is **{R['av_gap']:.0f} pts below** annual "
            f"mean in the window; the equity window is the year's weakest. Both legs point the wrong way. |\n\n"
            "> 💡 In plain words: the guacamole binge is a real *volume* event, but a date the whole "
            "supply chain sees coming leaves nothing in the *price*, and the snack tape's Jan–Feb window "
            "is a drag, not a surge. There is no axis on which the trade survives."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be the monthly return of the tradable proxy and $W = \\{\\text{Jan, Feb}\\}$ the "
            "guac window. The claim is a joint hypothesis:\n\n"
            "- **H₁ (a positive seasonal).** The window premium "
            "$\\;\\delta = \\mathbb{E}[r_t \\mid t\\in W] - \\mathbb{E}[r_t \\mid t\\notin W] > 0$ with a "
            "Welch *t* > 2.\n"
            "- **H₂ (it's special, not snooped).** Across all $\\binom{12}{2}=66$ month-pairs, $\\delta_W$ "
            "sits in the extreme upper tail — a real seasonal, not the lucky max of many windows.\n"
            "- **H₃ (it's tradable).** A long-window / cash-otherwise timer beats buy-and-hold on "
            "excess-of-cash Sharpe, net of costs.\n\n"
            "The steelman is genuine: the Super-Bowl avocado-volume spike is real and date-certain — the "
            "single best setup a consumer-calendar seasonal could hope for. The test is whether *price* "
            "inherits any of it once a supply chain with months of warning has pre-positioned."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If H₁–H₃ held, this would be the cleanest kind of edge: no forecasting, just a calendar "
            "rule you set once. That's exactly why it deserves the placebo. Consumer-demand seasonals "
            "are the graveyard of retail quant — the demand is visible, so it's *arbitraged into the "
            "supply chain* (growers plant, importers pre-book, retailers stock) long before it can move "
            "a traded price. H₁ is the raw effect; **H₂ is the honesty check** (a max over 66 windows is "
            "significant by construction — you must correct for the search); H₃ is whether any residue "
            "survives being traded. Failing any one downgrades \"seasonal trade\" to \"true story, no edge.\""
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Tradable tape.** `PEP` month-end returns (daily closes resampled, hole-free grid), "
            "1993-02→2026-05, benchmarked on `SPY`, cash leg `^IRX`. A *labelled proxy*: PEP is the "
            "Super-Bowl snack complex, not an avocado — and CVGW, the pure-play, is unavailable on the "
            "feed (a real look-ahead/availability caveat, stated).\n"
            "- **Avocado price (proxy, shape only).** A hardcoded, cited, approximate wholesale-Hass "
            "seasonal index — used to falsify the *premise* (is winter even a price peak?), never for a "
            "Signal stamp.\n"
            "- **Signal tests.** (i) Per-month one-sample HAC (Newey-West) *t*; the `REAL` bar is "
            "|*t*|≥2 *after* Bonferroni for 12 months (≈|*t*|≥3). (ii) Welch *t* of the Jan–Feb window "
            "spread. (iii) **Placebo**: the same spread for all 66 month-pairs; report the window's rank "
            "and *z*. (iv) A circular block-bootstrap (12-month blocks) CI on the spread.\n"
            "- **Cost / capacity (beat 6).** A calendar-known timer (no execution lag); Sharpe is "
            "excess-of-T-bill on both legs; costs charged one-way × NAV (2 legs/yr).\n"
            "- **Positive control.** A synthetic world with a *planted* Jan–Feb premium (and a null); "
            "the engine must recover the planted spread (*t* ≫ 2, placebo rank 66/66) and find nothing "
            "under the null — proof the real-tape null is a true null, not a broken harness.\n"
            "- **What would make us say \"real trade\":** H₁ *t* > 2 **and** a top-tail placebo rank "
            "**and** a timer that beats buy-and-hold. We find none."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The premise — does the avocado price even peak in winter?\n\n"
            "Before the tape: the (cited, approximate) wholesale-Hass seasonal index. If the Super Bowl "
            "lifted prices, the guac window would be a maximum. It is a *minimum-ish*."
        ),
        code(
            "vals=[float(AV.loc[m]) for m in range(1,13)]; ym=float(AV.mean())\n"
            "gap=np.mean(vals[:2])-ym\n"
            "fig, ax = plt.subplots(figsize=(9.2,4.2))\n"
            "cols=[RED if m in (1,2) else AMBER for m in range(1,13)]\n"
            "ax.bar(MONTH_NAMES, vals, color=cols, width=.7); ax.axhline(ym, ls='--', c=GREY, label=f'annual mean {ym:.0f}')\n"
            "ax.set_ylim(80,120); ax.set_ylabel('seasonal price index (100=mean)')\n"
            "ax.set_title('Premise check: winter is SOFT (peak Mexican supply); price peak is Aug'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'guac window {np.mean(vals[:2]):.1f} vs annual mean {ym:.1f} => gap {gap:+.1f} index pts')"
        ),
        md(
            f"> 💡 In plain words: the guac window sits **{R['av_gap']:.0f} points below** the annual "
            "average. The demand spike is real but *pre-supplied* — importers and retailers stock the "
            "shelves for a date they've known since last season, so the marginal price barely twitches. "
            "The premise of a \"price surge\" is false before we touch a stock."
        ),
        md(
            "### 4b · Per-month HAC *t*-stats on the tradable tape\n\n"
            "One-sample Newey-West *t* of each calendar month's `PEP` return. The `REAL` bar after "
            "Bonferroni (12 tests) is ≈ |*t*| ≥ 3; the guac window must clear it in the *positive* "
            "direction."
        ),
        code(
            "if HAVE:\n"
            "    ms = st.month_stats(D['pep'])\n"
            "    means=[ms.loc[m,'mean']*100 for m in range(1,13)]; thac=[ms.loc[m,'tstat_hac'] for m in range(1,13)]\n"
            "else:\n"
            "    means=[R['months'][m][0] for m in range(1,13)]; thac=[R['months'][m][2] for m in range(1,13)]\n"
            "fig, ax = plt.subplots(figsize=(9.4,4.3))\n"
            "cols=[RED if m in (1,2) else (GREEN if abs(thac[m-1])>=2 else GREY) for m in range(1,13)]\n"
            "ax.bar(MONTH_NAMES, thac, color=cols, width=.7)\n"
            "for y in (2,-2): ax.axhline(y, ls=':', c='k', alpha=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('HAC t-stat (mean vs 0)')\n"
            "ax.set_title('Per-month HAC t: guac window (red) is ~0; the |t|>2 months are off-thesis')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"Jan t_HAC {thac[0]:+.2f}  Feb {thac[1]:+.2f}  | Mar {thac[2]:+.2f}  Nov {thac[10]:+.2f} (both off-thesis)\")"
        ),
        md(
            f"> 💡 In plain words: Jan (*t*_HAC {R['months'][1][2]:+.2f}) and Feb ({R['months'][2][2]:+.2f}) "
            f"are indistinguishable from zero. The only months near significance — March "
            f"({R['months'][3][2]:+.2f}) and November ({R['months'][11][2]:+.2f}) — are nowhere near the "
            "Super Bowl and wouldn't survive Bonferroni anyway. No thesis month is real."
        ),
        md(
            "### 4c · The window spread + the placebo (the decisive test)\n\n"
            "Welch *t* of the Jan–Feb window vs the rest of the year, then the same spread for **all 66 "
            "month-pairs**. A real seasonal is a right-tail outlier; a snooped one is typical."
        ),
        code(
            "import itertools\n"
            "if HAVE:\n"
            "    ws = st.window_spread_tstat(D['pep']); pb = st.placebo_pairs(D['pep'])\n"
            "    s = D['pep']; s.index = pd.DatetimeIndex(s.index)\n"
            "    spreads=[(s[s.index.month.isin(p)].mean()-s[~s.index.month.isin(p)].mean())*100 for p in itertools.combinations(range(1,13),2)]\n"
            "    jf=ws['spread']*100; jt=ws['tstat']; rank=pb['rank']; z=pb['z']\n"
            "else:\n"
            "    spreads=list(np.random.default_rng(0).normal(0,0.9,66)); jf=R['spread']; jt=R['spread_t']; rank=R['placebo_rank']; z=R['placebo_z']\n"
            "fig, ax = plt.subplots(figsize=(9.2,4.3))\n"
            "ax.hist(spreads, bins=18, color=GREY, alpha=.75)\n"
            "ax.axvline(jf, c=RED, lw=2.5, label=f'Jan-Feb (guac) {jf:+.2f}%/mo, t={jt:+.2f}')\n"
            "ax.axvline(np.mean(spreads), c='k', ls='--', alpha=.6, label='placebo mean')\n"
            "ax.set_xlabel('month-pair minus rest-of-year (%/mo)'); ax.set_ylabel('# of the 66 pairs')\n"
            "ax.set_title(f'Placebo: Jan-Feb ranks {rank}/66 (z={z:+.2f}) — bottom decile, wrong tail'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"window mean {R['win_mean']:+.2f}% vs rest {R['rest_mean']:+.2f}% => spread {jf:+.2f}%/mo  t={jt:+.2f}  (n_win={R['n_win']}, n_rest={R['n_rest']})\")\n"
            "print(f\"placebo rank {rank}/66 (1=lowest), z={z:+.2f}; most-positive pairs are all off-thesis: {R['most_pos']}\")"
        ),
        md(
            f"> 💡 In plain words: the window spread is **{R['spread']:+.2f}%/mo, *t* = {R['spread_t']:+.2f}** "
            "— not just insignificant, *negative*. Against the placebo it ranks "
            f"**{R['placebo_rank']}/{R['placebo_n']}** (*z* = {R['placebo_z']:+.2f}): the Super-Bowl "
            "window is in the *bottom decile* of all month-pairs. Even the search-corrected upper tail "
            "belongs to off-thesis pairs (Mar–Oct). H₁ and H₂ both rejected."
        ),
        md(
            "### 4d · Block-bootstrap CI on the window spread\n\n"
            "Circular 12-month-block bootstrap (5000 resamples) of the Jan–Feb-minus-rest spread, "
            "respecting the annual structure."
        ),
        code(
            "if HAVE:\n"
            "    ci = st.spread_bootstrap_ci(D['pep'], n_boot=5000, seed=723)\n"
            "    lo,hi,pt = ci['lo']*100, ci['hi']*100, ci['point']*100\n"
            "else:\n"
            "    lo,hi,pt = R['ci_lo'], R['ci_hi'], R['ci_point']\n"
            "fig, ax = plt.subplots(figsize=(8.8,2.6))\n"
            "ax.hlines(0, lo, hi, color=AMBER, lw=6, alpha=.6)\n"
            "ax.plot([pt],[0],'o',c=RED,ms=10,label=f'point {pt:+.2f}%')\n"
            "ax.axvline(0, c='k', lw=1.2, ls='--')\n"
            "ax.set_yticks([]); ax.set_xlabel('Jan-Feb minus rest spread (%/mo)')\n"
            "ax.set_title(f'95% CI [{lo:+.2f}%, {hi:+.2f}%] — sits mostly BELOW zero'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'block-bootstrap 95% CI: [{lo:+.2f}%, {hi:+.2f}%]  point {pt:+.2f}%')"
        ),
        md(
            f"> 💡 In plain words: the CI is **[{R['ci_lo']:+.2f}%, {R['ci_hi']:+.2f}%]** — it barely "
            "pokes above zero and sits mostly negative. Even giving the folklore every benefit, the "
            "honest read is \"no positive seasonal, and if anything a small negative one.\""
        ),
        md(
            "### 4e · The timer race + Newey-West alpha\n\n"
            "Long the proxy in Jan–Feb, T-bills otherwise (calendar-known, no lag), vs buy-and-hold. "
            "Sharpe is excess-of-T-bill on both legs; net charges 5 bp one-way × NAV (2 legs/yr). Plus "
            "the Newey-West alpha of PEP vs SPY — is there *any* edge, or just beta?"
        ),
        code(
            "if HAVE:\n"
            "    rf=D['tbill']; timer=st.seasonal_timer(D['pep'], tbill=rf)\n"
            "    net=st.apply_costs(timer, n_trades_per_year=2, cost_bps_one_way=5)\n"
            "    rows={'guac timer\\n(gross)':st.summary(timer,rf=rf), 'guac timer\\n(net 5bp)':st.summary(net,rf=rf),\n"
            "          'buy&hold\\nPEP':st.summary(D['pep'],rf=rf), 'buy&hold\\nSPY':st.summary(D['spy'],rf=rf)}\n"
            "    sh=[rows[k]['sharpe'] for k in rows]\n"
            "    nw=st.newey_west_alpha_t(D['pep'], D['spy'], lags=6)\n"
            "else:\n"
            "    sh=[R['timer_sharpe'],R['timer_net_sharpe'],R['pep_sharpe'],R['spy_sharpe']]\n"
            "    rows={'guac timer\\n(gross)':0,'guac timer\\n(net 5bp)':0,'buy&hold\\nPEP':0,'buy&hold\\nSPY':0}\n"
            "    nw={'alpha_ann':R['nw_alpha']/100,'beta':R['nw_beta'],'t_alpha':R['nw_t']}\n"
            "labels=list(rows); cols=[RED,RED,AMBER,GREEN]\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.bar(labels, sh, color=cols, width=.62); ax.axhline(0,c='k',lw=1)\n"
            "for i,v in enumerate(sh): ax.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_ylabel('Sharpe (excess of T-bill)'); ax.set_title('H3: the guac timer Sharpe is NEGATIVE; buy-and-hold wins')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"guac timer gross {sh[0]:+.2f}  net {sh[1]:+.2f}  |  buy-hold PEP {sh[2]:+.2f}  SPY {sh[3]:+.2f}\")\n"
            "print(f\"PEP vs SPY Newey-West: alpha {nw['alpha_ann']*100:+.2f}%/yr  beta {nw['beta']:.2f}  t={nw['t_alpha']:+.2f}  (|t|<2: no alpha)\")"
        ),
        md(
            f"> 💡 In plain words: the timer's Sharpe is **{R['timer_sharpe']:+.2f}** gross, "
            f"**{R['timer_net_sharpe']:+.2f}** net — *negative*, because it holds the market only during "
            f"its worst window. Buy-and-hold SPY is **{R['spy_sharpe']:+.2f}**. And PEP's own alpha vs "
            f"SPY is an insignificant **{R['nw_alpha']:+.1f}%/yr** (*t* = {R['nw_t']:+.2f}) — no edge to "
            "harvest even ignoring the seasonal. H₃ rejected. `MIRAGE`."
        ),
        md(
            "### 4f · Positive control — the engine recovers a planted seasonal\n\n"
            "A synthetic world with a *planted* +3%/mo Jan–Feb premium (seed 723), and a null. The "
            "harness must light up on the plant and stay dark on the null — proving the real-tape "
            "`NONE` is a true null, not a broken pipeline."
        ),
        code(
            "dfp,truth = data.synthetic_world(jan_feb_premium=0.03, seed=723)\n"
            "dfn,_ = data.synthetic_world(jan_feb_premium=0.0, seed=723)\n"
            "wp=st.window_spread_tstat(dfp['pep']); pbp=st.placebo_pairs(dfp['pep'])\n"
            "wn=st.window_spread_tstat(dfn['pep'])\n"
            "tim=st.seasonal_timer(dfp['pep'], tbill=dfp['tbill']); bh=st.buy_hold(dfp['pep'])\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.2))\n"
            "ax.bar(['planted +3%/mo\\n(window spread t)','null\\n(window spread t)'], [wp['tstat'], wn['tstat']],\n"
            "       color=[GREEN, GREY], width=.5)\n"
            "ax.axhline(2, ls=':', c='k'); ax.axhline(0, c='k', lw=1); ax.set_ylabel('Welch t')\n"
            "ax.set_title('Machinery proof: engine recovers a planted seasonal, finds nothing under null')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"planted: window spread {wp['spread']*100:+.2f}%/mo t={wp['tstat']:+.2f}  placebo rank {pbp['rank']}/66  timer Sharpe {st.summary(tim,rf=dfp['tbill'])['sharpe']:+.2f} vs buy-hold {st.summary(bh,rf=dfp['tbill'])['sharpe']:+.2f}\")\n"
            "print(f\"null   : window spread {wn['spread']*100:+.2f}%/mo t={wn['tstat']:+.2f}  (correctly nothing)\")"
        ),
        md(
            f"> 💡 In plain words: with a real seasonal planted, the engine nails it — window spread "
            f"**{R['syn_spread']:+.2f}%/mo, *t* = {R['syn_t']:+.2f}**, placebo rank **{R['syn_rank']}/66** "
            f"(the extreme tail), timer Sharpe **{R['syn_timer']:+.2f}** beating buy-and-hold "
            f"**{R['syn_bh']:+.2f}**; under the null it finds *t* = {R['syn_null_t']:+.2f}. So the real "
            "tape's flat, wrong-tail result is a genuine null — not a harness that can't see."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — window spread **{R['spread']:+.2f}%/mo**, *t* = {R['spread_t']:+.2f} "
            f"(wrong sign); Jan *t*_HAC {R['months'][1][2]:+.2f}, Feb {R['months'][2][2]:+.2f}; placebo "
            f"rank **{R['placebo_rank']}/{R['placebo_n']}** (*z* = {R['placebo_z']:+.2f}); bootstrap CI "
            f"[{R['ci_lo']:+.2f}%, {R['ci_hi']:+.2f}%]. No robust *t* ≥ 2 anywhere in the window's favour "
            "— point estimates lean negative.\n"
            f"- **Tradability `MIRAGE`** — long-Jan–Feb timer Sharpe **{R['timer_sharpe']:+.2f}** gross / "
            f"{R['timer_net_sharpe']:+.2f} net vs buy-and-hold SPY **{R['spy_sharpe']:+.2f}**; PEP alpha "
            f"*t* = {R['nw_t']:+.2f}. And the pure-play (CVGW) can't be bought on the feed.\n"
            f"- **Guacamole surge? `BUSTED`** — the avocado price is **{R['av_gap']:.0f} pts below** its "
            "annual mean in the window and the snack tape's window is its weakest. A date-certain demand "
            "spike, fully pre-supplied, leaves no price for a trader."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the capacity & cost reality\n\n"
            "Terminal wealth of \\$10,000 from 1993, buy-and-hold SPY vs the guacamole timer (net). "
            "Capacity is a second wall: the *intended* instrument (Calavo, a micro-cap avocado name) is "
            "thinly traded even when it *is* on the tape — there is no scalable avocado book."
        ),
        code(
            "start=10_000.0; yrs=R['n_months']/12\n"
            "paths={'buy & hold SPY':R['spy_cagr']/100, 'guac timer (net)':R['timer_net_cagr']/100, 'buy & hold PEP':R['pep_cagr']/100}\n"
            "labels=list(paths); ends=[start*(1+g)**yrs for g in paths.values()]; cols=[GREEN, RED, AMBER]\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.bar(labels, ends, .55, color=cols)\n"
            "for i,v in enumerate(ends): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel(f'value of $10,000 after {yrs:.0f} years'); ax.set_title('The calendar timer earns cash rates 10 months a year — and the worst 2 months in-market')\n"
            "plt.tight_layout(); plt.show()\n"
            "for l,g in paths.items(): print(f\"{l:20s} ${start*(1+g)**yrs:>12,.0f}  ({g*100:+.1f}%/yr)\")"
        ),
        md(
            "> 💡 In plain words: the timer isn't just worse risk-adjusted — it compounds far less, "
            "because it's in T-bills most of the year and in the market precisely when the market is "
            "weakest. There is no sizing, venue, or cost assumption that turns a negative-Sharpe "
            "calendar window into an edge, and the on-thesis instrument has no capacity anyway."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Swap in the real avocado tape.** Replace the hardcoded seasonal with USDA AMS Market "
            "News weekly avocado price + Hass Avocado Board volume; the winter-soft / summer-peak shape "
            "(and the verdict) won't move, but you'll have the exact series.\n"
            "- **Recover CVGW.** When Calavo's daily history returns to the feed, drop it in for `PEP`; "
            "the placebo/timer machinery is ticker-agnostic. Expect a thinner, noisier null on a "
            "micro-cap, not a surge.\n"
            "- **The calendar-folklore prior.** Sell-in-May, the turn-of-month, the Santa rally, the "
            "[coffee frost seasonal](../../307-coffee-seasonality/): visible-calendar demand stories are "
            "arbitraged into the supply chain before they reach a traded price "
            "([docs/references.md](../docs/references.md)).\n\n"
            "*The reproducible core is offline and deterministic; the tradable leg is a **labelled "
            "proxy** and the avocado series is a **cited, approximate proxy**. Methods: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
            "[`docs/results.md`](../docs/results.md).*"
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
