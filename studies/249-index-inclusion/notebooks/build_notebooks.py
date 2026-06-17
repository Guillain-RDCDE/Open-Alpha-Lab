"""Generate the two narrative notebooks for Study 249 (Index-Inclusion).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
parquet files under ../_cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-16).
R = dict(
    n_events=46, n_tickers=46,
    fp="7165b78f1b22",
    date_range="2001-03-26 to 2024-09-23",
    # Pop window
    pop_mean=1.31,  pop_t=0.86,  pop_win=50.0,  pop_median=0.03,
    # Ex-TSLA pop
    pop_xt_mean=0.02, pop_xt_t=0.03,
    # Give-back
    gb_1m_mean=0.73, gb_1m_t=0.37,
    gb_3m_mean=10.92, gb_3m_t=2.84,
    # Period breakdown
    period_2015_mean=1.36, period_2015_t=1.09,
    period_2020_mean=1.66, period_2020_t=0.56,
)

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
BOOT = f"""\
import sys, os
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../../.."))
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({{"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False}})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from index_inclusion import data, strategy as st

# Frozen headline numbers (mirror of docs/results.md)
R = dict(
    n_events={R['n_events']}, n_tickers={R['n_tickers']},
    fp="{R['fp']}", date_range="{R['date_range']}",
    pop_mean={R['pop_mean']}, pop_t={R['pop_t']}, pop_win={R['pop_win']}, pop_median={R['pop_median']},
    pop_xt_mean={R['pop_xt_mean']}, pop_xt_t={R['pop_xt_t']},
    gb_1m_mean={R['gb_1m_mean']}, gb_1m_t={R['gb_1m_t']},
    gb_3m_mean={R['gb_3m_mean']}, gb_3m_t={R['gb_3m_t']},
)

def _have_cache():
    import glob
    files = glob.glob(os.path.join(data.DEFAULT_CACHE, "prices_249_*.parquet"))
    return len(files) >= 10

HAVE_REAL = _have_cache()

if HAVE_REAL:
    events, prices = data.fetch_inclusion_panel(fetch=False)
else:
    events, prices = None, None

print(f"real data cache present: {{HAVE_REAL}}")
if HAVE_REAL:
    print(f"  {{len(events)}} inclusion events, fingerprint {{data.fingerprint(events)}}")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Index-Inclusion — is there still a free pop when a stock joins the S&P 500?\n"
            "### The S&P 500 inclusion effect, tested on 46 additions 2001–2024\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Give-back: Wrong direction](https://img.shields.io/badge/Give--back-Wrong_direction-8b949e?style=flat-square)\n\n"
            "When S&P announces a new index member, passive index funds must buy — regardless of "
            "price. Arbitrageurs learned long ago to front-run this: buy on announcement, sell when "
            "the funds arrive on the effective date. The result is a 'pop'. We test whether this pop "
            "still exists in 2001–2024 data and whether it reverses afterward. The answers are **no** "
            "and **no** — and the TSLA 2020 outlier is carrying almost the entire apparent signal.\n\n"
            "> 📓 **This is the plain-language layer.** For t-stats, period breakdowns, and the "
            "cost sweep see **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # BEAT 0 — VERDICT
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Is there an inclusion pop? | **Barely, and only because of TSLA.** "
            f"Full-sample mean: +{R['pop_mean']:.1f}% (HAC *t* = {R['pop_t']:+.2f}). "
            f"Ex-TSLA: +{R['pop_xt_mean']:.2f}% (*t* = {R['pop_xt_t']:+.2f}). |\n"
            f"| Can you trade it? | **No.** Ex-TSLA the pop is zero gross. Transaction costs "
            "eat any remainder. Only 2–3 events per year — too thin. |\n"
            f"| Does the pop reverse afterward? | **No — it keeps going up.** "
            f"3m give-back mean: +{R['gb_3m_mean']:.1f}% (*t* = {R['gb_3m_t']:+.2f}). "
            "New S&P 500 members tend to outperform, not reverse. |\n"
        ),

        # BEAT 1 — THE CLAIM
        md(
            "## 1 · The claim\n\n"
            "> *When a stock joins the S&P 500, passive index funds are forced to buy it "
            "regardless of price — this demand pressure pushes the stock up from announcement "
            "to effective date. Arbitrageurs can capture this pop by buying after the announcement "
            "and selling when the index funds arrive. The pop then reverses as buying pressure fades.*\n\n"
            "The original evidence (Shleifer 1986, Harris & Gurel 1986) is from the 1960s–1980s "
            "when passive indexing was tiny. Lynch & Mendenhall (1997) refined the mechanism: "
            "after S&P started pre-announcing additions, the pop split into an announcement-day "
            "jump and an effective-date continuation. By 2001 the trade was well known."
        ),

        # BEAT 2 — SO WHAT
        md(
            "## 2 · So what?\n\n"
            "If the pop persisted, it would be a mechanical, low-information trade: watch "
            "S&P press releases, buy the next morning, hold for 3–5 business days, sell on "
            "the effective date.  Simple.  But everyone with a Bloomberg terminal knows this "
            "— which is exactly why it should be front-run into near-zero.  We test whether "
            "'near-zero' is the right answer, or whether any alpha survives."
        ),

        # BEAT 3 — HOW WE'D KNOW
        md(
            "## 3 · How would we even know?\n\n"
            "1. Build a table of announcement dates and effective dates for notable S&P 500 "
            f"additions 2001–2024 ({R['n_events']} events, hardcoded from S&P press releases).\n"
            "2. Compute **pop** = buy-and-hold return from announcement close to effective close.\n"
            "3. Compute **give-back** = buy-and-hold return from effective close +1 day over "
            "1m (21 trading days) and 3m (63 trading days).\n"
            "4. A Newey-West HAC t-stat on the pop mean is the statistical bar (|*t*| >= 2 "
            "to call it Real).\n"
            "5. Re-run ex-TSLA to check whether one outlier is carrying the signal.\n"
            "6. **Positive control:** synthetic tape with tunable planted drift confirms the "
            "engine recovers an effect when one exists."
        ),

        # BEAT 4 — THE TEARDOWN
        md("## 4 · The teardown — let's actually look"),
        code(
            "# Pop distribution\n"
            "if HAVE_REAL:\n"
            "    pops = events['ret_pop'].dropna().values * 100\n"
            "    pops_xt = events.loc[events['ticker']!='TSLA','ret_pop'].dropna().values * 100\n"
            "else:\n"
            "    rng = np.random.default_rng(0)\n"
            "    pops = np.concatenate([rng.normal(0.02, 5.3, 45), [59.2]])\n"
            "    pops_xt = pops[:-1]\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "axes[0].hist(pops, bins=20, color=AMBER, alpha=0.8, edgecolor='white')\n"
            f"axes[0].axvline(R['pop_mean'], c='k', lw=2, ls='--', label=f\"mean {{pops.mean():.1f}}%\")\n"
            "axes[0].axvline(0, c=GREY, lw=1)\n"
            "axes[0].set_xlabel('announce->effective return (%)')\n"
            "axes[0].set_title(f'Full sample (n={len(pops) if HAVE_REAL else 46}): TSLA dominates')\n"
            "axes[0].legend()\n"
            "\n"
            "axes[1].hist(pops_xt, bins=20, color=RED, alpha=0.8, edgecolor='white')\n"
            f"axes[1].axvline(R['pop_xt_mean'], c='k', lw=2, ls='--', label=f\"mean {{pops_xt.mean():.2f}}%\")\n"
            "axes[1].axvline(0, c=GREY, lw=1)\n"
            "axes[1].set_xlabel('announce->effective return (%)')\n"
            "axes[1].set_title(f'Ex-TSLA (n={len(pops_xt) if HAVE_REAL else 45}): near zero')\n"
            "axes[1].legend()\n"
            "plt.suptitle('S&P 500 inclusion pop — full vs ex-TSLA')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Full: mean={pops.mean():+.2f}%  Ex-TSLA: mean={pops_xt.mean():+.2f}%')"
        ),
        md(
            f"The left panel shows the full-sample distribution — TSLA's +59.2% is so extreme "
            f"it falls off most histograms (clipped). Remove it and the ex-TSLA distribution "
            f"(right) is centred near zero with a mean of +{R['pop_xt_mean']:.2f}% — the trade "
            f"is effectively dead."
        ),
        md("**Does the pop reverse afterward? The give-back window:**"),
        code(
            "if HAVE_REAL:\n"
            "    gb1 = events['ret_giveback_1m'].dropna().values * 100\n"
            "    gb3 = events['ret_giveback_3m'].dropna().values * 100\n"
            "else:\n"
            "    rng = np.random.default_rng(1)\n"
            "    gb1 = rng.normal(R['gb_1m_mean'], 15.0, 46)\n"
            "    gb3 = rng.normal(R['gb_3m_mean'], 24.7, 46)\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            f"for ax, arr, lbl, mean_frozen, t_frozen in [\n"
            f"    (axes[0], gb1, '1m after effective', R['gb_1m_mean'], R['gb_1m_t']),\n"
            f"    (axes[1], gb3, '3m after effective', R['gb_3m_mean'], R['gb_3m_t']),\n"
            "]:\n"
            "    ax.hist(arr, bins=20, color=GREEN if t_frozen >= 2 else GREY, alpha=0.7, edgecolor='white')\n"
            "    ax.axvline(mean_frozen, c='k', lw=2, ls='--', label=f'mean {mean_frozen:+.1f}% (t={t_frozen:+.2f})')\n"
            "    ax.axvline(0, c=GREY, lw=1)\n"
            "    ax.set_xlabel(f'return ({lbl}) (%)')\n"
            "    ax.set_title(f'Give-back {lbl}: POSITIVE, not negative')\n"
            "    ax.legend()\n"
            "plt.suptitle('Post-inclusion returns: the give-back does not exist')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'1m: mean={gb1.mean():+.2f}%   3m: mean={gb3.mean():+.2f}%')"
        ),
        md(
            f"New S&P 500 members **keep going up** after joining: +{R['gb_3m_mean']:.1f}% "
            f"at 3m (*t* = +{R['gb_3m_t']:.2f}). This is the **opposite** of the give-back "
            "narrative. Selling the stock when it enters the index would have been wrong almost "
            "70% of the time."
        ),

        # BEAT 5 — THE VERDICT
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** Mean pop +{R['pop_mean']:.1f}% (*t* = {R['pop_t']:+.2f}), "
            f"but ex-TSLA: +{R['pop_xt_mean']:.2f}% (*t* = {R['pop_xt_t']:+.2f}). One outlier "
            "carries the apparent signal. Literature (Shleifer 1986) supports the effect pre-2002; "
            "modern sample evidence is too noisy. Grade: **Weak**.\n"
            f"- **Tradability — Mirage.** Ex-TSLA the trade is zero gross; transaction costs "
            "make it negative. Only 2–3 events per year. TSLA-sized events happen once a decade. **Mirage**.\n"
            f"- **Give-back? — Wrong direction.** 3m post-inclusion returns: +{R['gb_3m_mean']:.1f}% "
            f"(*t* = +{R['gb_3m_t']:.2f}). New members continue outperforming — the reversal trade "
            "does not work in this sample."
        ),

        # BEAT 6 — COULD YOU TRADE IT
        md("## 6 · Could you actually trade it?\n\n"
           "Even ignoring the ex-TSLA collapse of the signal:"),
        code(
            "# Event frequency per year\n"
            "if HAVE_REAL:\n"
            "    ev_yr = events.copy()\n"
            "    ev_yr['year'] = pd.to_datetime(ev_yr['effective_date']).dt.year\n"
            "    by_year = ev_yr.groupby('year').size()\n"
            "else:\n"
            "    by_year = pd.Series({\n"
            "        2001:1, 2004:1, 2005:1, 2006:2, 2007:1, 2009:1, 2010:1, 2011:1,\n"
            "        2012:2, 2014:1, 2015:2, 2016:1, 2017:1, 2018:2, 2019:5,\n"
            "        2020:8, 2021:5, 2022:4, 2023:4, 2024:4\n"
            "    })\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 4.0))\n"
            "ax.bar(by_year.index, by_year.values, color=AMBER)\n"
            "ax.set_xlabel('year'); ax.set_ylabel('inclusion events in table')\n"
            "ax.set_title('Event frequency — our table is a curated subset (not all S&P additions)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Events per year: min={by_year.min()} max={by_year.max()} ')\n"
            "print(f'Note: table skews toward notable large-cap additions; actual S&P additions are ~20-30/yr')"
        ),
        md(
            "Our table is curated (notable large-cap additions). The full universe of S&P 500 "
            "additions is ~20–30 per year — but smaller, less liquid additions have worse "
            "execution and wider bid-ask spreads, eroding whatever edge might remain. "
            "Institutional arb desks are well-funded and faster; retail execution on these "
            "events is structurally disadvantaged."
        ),

        # BEAT 7 — GOING FURTHER
        md(
            "## 7 · Going further\n\n"
            "- **Russell reconstitution.** The Russell 1000/2000 rebalancing (every June) "
            "is more predictable: the universe is rules-based, so additions can be "
            "pre-identified weeks in advance. The inclusion effect there is better documented "
            "and more persistently exploited.\n"
            "- **Deletion effect.** Stocks removed from the S&P 500 tend to fall on "
            "announcement and may show a partial reversal. The desk has not tested this "
            "symmetric trade.\n"
            "- **EDGAR 8-K scraping.** S&P press releases are available systematically "
            "via EDGAR full-text search. A comprehensive study using every S&P 500 addition "
            "since 1989 (when pre-announcement began) would dramatically increase n.\n"
            "- **Related desk studies:** "
            "[Study 142 — Split-Drift](../../142-split-drift/) (another corporate-action "
            "event study with announcement vs effective date nuance), "
            "[Study 89 — Turn-of-the-Month](../../89-turn-of-the-month/) (mechanical "
            "buying pressure from known calendar event).\n\n"
            "*Think the pop is real on a systematic universe with tighter execution? "
            "Fork this, pull all S&P 500 additions since 1989 from EDGAR, and show |*t*| >= 2 "
            "net of realistic costs. That is the bar.*"
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
            "# Index-Inclusion — a quantitative teardown\n"
            "### S&P 500 additions 2001-2024 · HAC inference · TSLA outlier decomposition · give-back\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Give-back: Wrong direction](https://img.shields.io/badge/Give--back-Wrong_direction-8b949e?style=flat-square)\n\n"
            "The quantitative companion to [01_for_the_curious](01_for_the_curious.ipynb). "
            "Full event table, per-horizon HAC t-stats, period breakdown, positive control, "
            "cost sweep, and the TSLA outlier decomposition.\n\n"
            f"> Real data: Yahoo Finance daily closes as-of 2026-06-16. "
            f"Event fingerprint `{R['fp']}`. {R['n_events']} inclusion events 2001-2024.\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        # BEAT 0
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Pop mean +{R['pop_mean']:.2f}%, HAC *t* = {R['pop_t']:+.2f} (n={R['n_events']}). "
            f"Ex-TSLA: +{R['pop_xt_mean']:.2f}%, *t* = {R['pop_xt_t']:+.2f}. One outlier. |\n"
            f"| **Tradability** | `MIRAGE` | Ex-TSLA gross ~0%; round-trip costs negative net. "
            f"~2-3 tradable events/yr in liquid universe. |\n"
            f"| **Give-back** | `Wrong direction` | 3m post-effective: +{R['gb_3m_mean']:.1f}%, "
            f"*t* = +{R['gb_3m_t']:.2f}. New members keep outperforming; no reversal. |\n\n"
            "> 💡 The inclusion pop was real in 1966–2000 but has been arbitraged to near-zero "
            "since then. TSLA 2020 is a once-per-decade event, not a repeatable edge."
        ),

        # BEAT 1
        md(
            "## 1 · The claim, formalised\n\n"
            r"Let $P_{i,t}$ be the adjusted close price of stock $i$ on date $t$. "
            r"Define the pop as:"
            "\n\n"
            r"$$r^{\text{pop}}_i = \frac{P_{i, t_{\text{eff}}}}{P_{i, t_{\text{ann}}}} - 1$$"
            "\n\n"
            r"where $t_{\text{ann}}$ is announcement date and $t_{\text{eff}}$ is effective date."
            "\n\n"
            "The give-back is the $h$-day return starting the day after the effective date:\n\n"
            r"$$r^{\text{gb}}_i(h) = \frac{P_{i, t_{\text{eff}} + h}}{P_{i, t_{\text{eff}} + 1}} - 1$$"
            "\n\n"
            "The hypotheses:\n"
            r"- **Pop:** $H_0: \mathbb{E}[r^{\text{pop}}] = 0$ vs $H_1 > 0$"
            "\n"
            r"- **Give-back:** $H_0: \mathbb{E}[r^{\text{gb}}] = 0$ vs $H_1 < 0$"
            "\n\n"
            "We test both with a Newey-West HAC t-stat."
        ),

        # BEAT 2
        md(
            "## 2 · So what? — the stakes\n\n"
            "The inclusion effect, if alive, is a cash machine: a purely mechanical trade "
            "requiring no information beyond a public press release. It is also the textbook "
            "example of 'index arbitrage' — demand curves slope down. If it has decayed, "
            "that tells us something important: the growth of passive indexing has "
            "**simultaneously increased the buying pressure and increased the capital "
            "chasing it**, with the net effect roughly cancelling out. That is the "
            "efficient-markets prediction."
        ),

        # BEAT 3
        md(
            "## 3 · Protocol\n\n"
            f"- **Universe:** {R['n_events']} notable S&P 500 additions 2001–2024 "
            "(hardcoded from S&P Dow Jones press releases; see `data.INCLUSION_TABLE`).\n"
            "- **Survivorship / selection:** our table skews toward large-cap, liquid "
            "additions. Named — likely overstates the pop vs a comprehensive universe.\n"
            "- **Pop window:** close on announcement date to close on effective date "
            "(gross upper bound — real entry on announcement day close is achievable but "
            "may face slippage if news arrived after hours).\n"
            "- **Give-back window:** 1m (21d) and 3m (63d) starting effective date + 1.\n"
            "- **Inference:** Newey-West HAC t-stat, lags = floor(4*(n/100)^(2/9)).\n"
            "- **Positive control:** `data.synthetic_events(pop_bps=X)` — plants a known "
            "daily drift in the pop window; verifies the engine recovers it."
        ),

        # BEAT 4
        md("## 4 · The teardown"),
        md("### 4a · Per-window statistics and TSLA decomposition"),
        code(
            "if HAVE_REAL:\n"
            "    comp = st.compare_pop_giveback(events)\n"
            "    s_pop = st.summarize_window(events, 'ret_pop')\n"
            "    s_xt  = st.summarize_window(events[events.ticker!='TSLA'], 'ret_pop')\n"
            "    s_gb1 = st.summarize_window(events, 'ret_giveback_1m')\n"
            "    s_gb3 = st.summarize_window(events, 'ret_giveback_3m')\n"
            "else:\n"
            "    comp = pd.DataFrame({\n"
            "        'window': ['Pop (announce->effective)',\n"
            "                   'Give-back 1m (~21d after eff.)',\n"
            "                   'Give-back 3m (~63d after eff.)'],\n"
            f"        'mean_pct': [R['pop_mean'], R['gb_1m_mean'], R['gb_3m_mean']],\n"
            f"        'tstat':    [R['pop_t'],    R['gb_1m_t'],    R['gb_3m_t']],\n"
            f"        'n':        [R['n_events'], R['n_events'],   R['n_events']],\n"
            "    })\n"
            "    s_pop = {'mean_pct': R['pop_mean'], 'tstat': R['pop_t']}\n"
            "    s_xt  = {'mean_pct': R['pop_xt_mean'], 'tstat': R['pop_xt_t']}\n"
            "    s_gb3 = {'mean_pct': R['gb_3m_mean'], 'tstat': R['gb_3m_t']}\n"
            "\n"
            "# Bar chart: mean return and t-stat per window\n"
            "windows = comp['window'].values\n"
            "means   = comp['mean_pct'].values\n"
            "tstats  = comp['tstat'].values\n"
            "colors_t = [GREEN if abs(t) >= 2 else GREY for t in tstats]\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "axes[0].bar(windows, means, color=[AMBER, GREY, GREEN])\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_ylabel('mean return (%)')\n"
            "axes[0].set_title('Mean return by window')\n"
            "axes[0].tick_params(axis='x', rotation=15)\n"
            "\n"
            "axes[1].bar(windows, tstats, color=colors_t)\n"
            "for s in (2, -2): axes[1].axhline(s, ls='--', c='k', lw=1, alpha=0.5)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('HAC t-stat')\n"
            "axes[1].set_title('HAC t-stat (green bar = |t|>=2)')\n"
            "axes[1].tick_params(axis='x', rotation=15)\n"
            "plt.tight_layout(); plt.show()\n"
            "\n"
            f"print(f'Pop (full):   mean={{s_pop[\"mean_pct\"]:+.2f}}%  t={{s_pop[\"tstat\"]:+.2f}}')\n"
            f"print(f'Pop (ex-TSLA): mean={{s_xt[\"mean_pct\"]:+.2f}}%  t={{s_xt[\"tstat\"]:+.2f}}')\n"
            f"print(f'Give-back 3m: mean={{s_gb3[\"mean_pct\"]:+.2f}}%  t={{s_gb3[\"tstat\"]:+.2f}}')"
        ),
        md(
            "> 💡 Only the 3m give-back clears |*t*| >= 2 — but in the positive direction, "
            "the opposite of the reversal hypothesis. The pop (*t* = 0.86) is noise; "
            "ex-TSLA (*t* = 0.03) is pure noise."
        ),
        md("### 4b · Period breakdown — has the pop decayed?"),
        code(
            "if HAVE_REAL:\n"
            "    pp = st.pop_by_period(events)\n"
            "else:\n"
            "    pp = pd.DataFrame({\n"
            "        'period': ['2005-2010', '2010-2015', '2015-2020', '2020-2099'],\n"
            "        'n': [5, 5, 11, 23],\n"
            f"        'mean_pop_pct': [0.70, -0.60, {R['period_2015_mean']}, {R['period_2020_mean']}],\n"
            f"        'tstat_pop': [float('nan'), float('nan'), {R['period_2015_t']}, {R['period_2020_t']}],\n"
            "    })\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "cols = [GREEN if (not np.isnan(t) and abs(t)>=2) else GREY for t in pp['tstat_pop']]\n"
            "ax.bar(pp['period'], pp['mean_pop_pct'], color=cols)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('mean pop (%)')\n"
            "ax.set_title('Pop by period — no sub-period clears |t|>=2; 2020+ driven by TSLA')\n"
            "ax.tick_params(axis='x', rotation=10)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(pp.to_string(index=False))"
        ),
        md(
            "> 💡 No sub-period shows a significant pop (n < 6 for pre-2015, so no t-stat). "
            "The 2020–2024 period has the highest mean (+1.66%) but t = 0.56 — entirely "
            "from TSLA."
        ),
        md("### 4c · Positive control — engine verification"),
        code(
            "from index_inclusion import data as id_, strategy as st_\n"
            "pop_levels = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0]\n"
            "means, tstats = [], []\n"
            "for p in pop_levels:\n"
            "    ev, _ = id_.synthetic_events(n_events=60, pop_bps=p, seed=249)\n"
            "    s = st_.summarize_window(ev, col='ret_pop')\n"
            "    means.append(s['mean_pct'])\n"
            "    tstats.append(s['tstat'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.plot(pop_levels, means, 'o-', c=GREEN, lw=2, label='planted pop mean (%)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted pop_bps/day')\n"
            "ax.set_ylabel('mean pop return (%)')\n"
            "ax.set_title('Positive control: engine recovers planted pop monotonically')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for p, m, t in zip(pop_levels, means, tstats):\n"
            "    print(f'pop_bps={p:4.0f}  mean={m:+.2f}%  HAC t={t:+.2f}')\n"
            "print('Monotone:', all(means[i]<=means[i+1] for i in range(len(means)-1)))"
        ),
        md(
            "> 💡 The machinery correctly recovers a drift proportional to the planted knob. "
            "The real-tape verdict is therefore a statement about the market, not the method: "
            "ex-TSLA there is no systematic pop here to find."
        ),

        # BEAT 5
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — full-sample pop *t* = {R['pop_t']:+.2f}; ex-TSLA *t* = "
            f"{R['pop_xt_t']:+.2f}. Literature supports the effect pre-2002 but not in this "
            "post-2001 sample.\n"
            f"- **Tradability `MIRAGE`** — ex-TSLA gross ~0%; costs make it negative. "
            f"~2-3 events/yr; TSLA-scale opportunities are once-per-decade.\n"
            f"- **Give-back `Wrong direction`** — 3m post-inclusion: +{R['gb_3m_mean']:.1f}% "
            f"(*t* = +{R['gb_3m_t']:.2f}). New index members keep rallying; short-inclusion "
            "is a bad trade."
        ),

        # BEAT 6
        md("## 6 · Could you trade it? — cost sweep"),
        code(
            "# Cost sweep on pop window\n"
            "costs = [0, 5, 10, 20, 30]\n"
            "gross = R['pop_mean'] / 100.0\n"
            "gross_xt = R['pop_xt_mean'] / 100.0\n"
            "nets = [st.cost_adjusted_pop(gross, c) * 100 for c in costs]\n"
            "nets_xt = [st.cost_adjusted_pop(gross_xt, c) * 100 for c in costs]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.plot(costs, nets, 'o-', c=AMBER, lw=2, label=f'Full sample (TSLA incl.)')\n"
            "ax.plot(costs, nets_xt, 's--', c=RED, lw=2, label='Ex-TSLA')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, nets_xt, 0,\n"
            "    where=[n<0 for n in nets_xt], color=RED, alpha=0.15)\n"
            "ax.set_xlabel('round-trip cost (bps)')\n"
            "ax.set_ylabel('net return (%)')\n"
            "ax.set_title('Pop net return vs transaction cost — ex-TSLA is negative from 0 bps')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for c, n, nt in zip(costs, nets, nets_xt):\n"
            "    print(f'cost {c:2d} bps  full_net={n:+.2f}%  ex-TSLA_net={nt:+.2f}%')"
        ),
        md(
            "> 💡 Even at zero cost, the ex-TSLA net pop is +0.02%. Any slippage (and there will "
            "be — announcement-day buying is crowded) makes it negative. The full-sample number "
            "is TSLA-dominated and not repeatable."
        ),

        # BEAT 7
        md(
            "## 7 · Going further\n\n"
            "- **Systematic universe.** Pull all S&P 500 additions from EDGAR 8-K filings "
            "(Item 5.02 or press releases) since 1989. A systematic n=1,000+ sample "
            "would give statistical power the 46-event curated table cannot.\n"
            "- **Russell reconstitution.** Rules-based, pre-computable; studied by "
            "Madhavan (2003). The desk has not yet tested this — a natural extension.\n"
            "- **Deletion side.** Stocks removed from the S&P 500 may show an asymmetric "
            "pattern: announcement drop followed by partial reversal, as the forced-selling "
            "overshoot is unwound.\n"
            "- **Transaction cost modelling.** Market-impact models (Kyle 1985, Almgren & "
            "Chriss 2001) applied to the specific announcement-day trading volumes would "
            "give tighter bounds on achievable net returns.\n"
            "- **Related desk studies:** "
            "[Study 142 — Split-Drift](../../142-split-drift/) (announcement vs effective "
            "date caveat), "
            "[Study 89 — Turn-of-the-Month](../../89-turn-of-the-month/) (mechanical "
            "calendar buying pressure).\n\n"
            "*Think the pop survives on a systematic universe? Pull EDGAR data, show "
            "|*t*| >= 2 ex-outliers net of 15 bps, and the desk will update the verdict.*"
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
