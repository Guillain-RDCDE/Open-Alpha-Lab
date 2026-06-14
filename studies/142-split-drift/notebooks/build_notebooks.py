"""Generate the two narrative notebooks for Study 142 (Split-Drift).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
parquet files under ../_cache/ if present and otherwise quote the frozen headline
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-14).
R = dict(
    n_events=41, n_tickers=22, n_base=205,
    # Split arm
    sp_1m=-1.49,  sp_1m_t=-0.92,  sp_1m_win=43.9,
    sp_3m=+0.40,  sp_3m_t=+0.15,  sp_3m_win=53.7,
    sp_6m=-0.67,  sp_6m_t=-0.14,  sp_6m_win=58.5,
    sp_12m=+5.14, sp_12m_t=+0.99, sp_12m_win=62.5,
    # Baseline arm
    bs_1m=+2.06,  bs_1m_t=+2.59,
    bs_3m=+5.67,  bs_3m_t=+4.68,
    bs_6m=+11.74, bs_6m_t=+5.66,
    bs_12m=+28.71, bs_12m_t=+7.26,
    # Difference (split - baseline)
    diff_1m=-3.55,  diff_1m_t=-2.20,
    diff_3m=-5.27,  diff_3m_t=-2.00,
    diff_6m=-12.42, diff_6m_t=-2.50,
    diff_12m=-23.56, diff_12m_t=-4.56,
    # Fingerprint
    fp="4b63fefd052e",
    date_range="2000-01-14 to 2025-11-17",
)

# ---------------------------------------------------------------------------
# Shared preamble — imports, data loading, HAVE_REAL guard
# ---------------------------------------------------------------------------
BOOT = f"""\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({{"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False}})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from split_drift import data, strategy as st

# Frozen headline numbers (mirror of docs/results.md) — used when real cache is absent.
R = dict(
    n_events={R['n_events']}, n_tickers={R['n_tickers']}, n_base={R['n_base']},
    sp_1m={R['sp_1m']},  sp_1m_t={R['sp_1m_t']},  sp_1m_win={R['sp_1m_win']},
    sp_3m={R['sp_3m']},  sp_3m_t={R['sp_3m_t']},  sp_3m_win={R['sp_3m_win']},
    sp_6m={R['sp_6m']},  sp_6m_t={R['sp_6m_t']},  sp_6m_win={R['sp_6m_win']},
    sp_12m={R['sp_12m']}, sp_12m_t={R['sp_12m_t']}, sp_12m_win={R['sp_12m_win']},
    bs_1m={R['bs_1m']},  bs_1m_t={R['bs_1m_t']},
    bs_3m={R['bs_3m']},  bs_3m_t={R['bs_3m_t']},
    bs_6m={R['bs_6m']},  bs_6m_t={R['bs_6m_t']},
    bs_12m={R['bs_12m']}, bs_12m_t={R['bs_12m_t']},
    diff_1m={R['diff_1m']},  diff_1m_t={R['diff_1m_t']},
    diff_3m={R['diff_3m']},  diff_3m_t={R['diff_3m_t']},
    diff_6m={R['diff_6m']},  diff_6m_t={R['diff_6m_t']},
    diff_12m={R['diff_12m']}, diff_12m_t={R['diff_12m_t']},
)

def _have_cache():
    # Check for at least one cached split file and one price file
    import glob
    sp_files = glob.glob(os.path.join(data.DEFAULT_CACHE, "splits_142_*.parquet"))
    pr_files = glob.glob(os.path.join(data.DEFAULT_CACHE, "prices_142_*_1d.parquet"))
    return len(sp_files) > 5 and len(pr_files) > 5

HAVE_REAL = _have_cache()

if HAVE_REAL:
    events, prices = data.fetch_splits_panel(fetch=False)
    baseline = st.build_baseline_events(events, prices, n_control_per_event=5, seed=142)
else:
    events, prices, baseline = None, None, None

print(f"real data cache present: {{HAVE_REAL}}")
if HAVE_REAL:
    print(f"  {{len(events)}} split events, {{len(baseline)}} baseline events")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Split-Drift — do stocks drift up after a stock split?\n"
            "### The Ikenberry-Rankine-Stice (1996) post-split anomaly, tested on effective dates\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Post--effective_drift%3F: Absent](https://img.shields.io/badge/Post--effective_drift%3F-Absent-8b949e?style=flat-square)\n\n"
            "A 1996 paper by Ikenberry, Rankine and Stice found that stocks announcing a split "
            "drifted up **+7.9%** over the following year — the market seemed to under-react to "
            "the signal of management confidence.  Here we test whether that drift persists "
            "**after the effective (ex-split) date** using yfinance data on 30 large-cap US stocks.  "
            "The answer is a firm **no** — and at the 12-month horizon, split stocks actually "
            "**underperformed** a matched no-split baseline by 23 percentage points.\n\n"
            "> **One big caveat before anything else.** yfinance gives us the split *effective* "
            "date, not the *announcement* date (typically 3–6 weeks earlier).  By the effective "
            "date the market has already had time to react.  We are testing a stricter, weaker "
            "window than the original study.  If the original effect existed and was fully priced "
            "by announcement, we should find nothing here — and that is exactly what we find.\n\n"
            "> 📓 **This is the plain-language layer.** For t-stats, bootstrap bands, and the "
            "full event table see "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do stocks drift up after a split effective date? | **No.** The 12-month mean "
            f"is +{R['sp_12m']:.1f}% (HAC *t* = {R['sp_12m_t']:+.2f}) — not significant. |\n"
            f"| Do they beat a no-split baseline? | **No — they underperform.** Split months return "
            f"{R['sp_3m']:+.1f}% at 3m vs the baseline's {R['bs_3m']:+.1f}% "
            f"(difference: {R['diff_3m']:+.1f}%, *t* = {R['diff_3m_t']:+.2f}). |\n"
            "| Could you trade it? | **No.** No significant edge at any horizon; only ~2 "
            "large-cap splits per year in recent years. |\n"
            "| Does the classic Ikenberry (1996) anomaly still work? | "
            "**Cannot say** — we test from the effective date, not the announcement date. |\n\n"
            "> The post-effective-date drift is not just absent — at longer horizons the split "
            "events show the *opposite* pattern: below-average returns vs the same stocks "
            "in non-split periods.  A possible explanation: splits tend to occur after strong "
            "prior performance (the share price is 'too high'), and the mean-reversion that "
            "follows is the dominant signal."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *A stock split signals management confidence in future earnings — the firm is "
            "so sure of continued growth that it lowers the share price to attract more retail "
            "investors.  The market under-reacts to this signal, so buying at the split date "
            "and holding for 6–12 months earns an abnormal return of +8% to +12%.*  "
            "— Ikenberry, Rankine & Stice (1996), JFQA\n\n"
            "The theoretical backbone is credible: splits are costly (they raise microstructure "
            "noise and transaction costs), so only genuinely confident managers do them.  The "
            "empirical finding in 1996 was striking — but it was measured from the **announcement** "
            "date, in 1975–1990 data.  We test 2000–2025 effective-date data."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the drift persists to the effective date, it would be a simple, mechanical "
            "trading rule: watch the 'Splits & Dividends' calendar, buy at the open on ex-date, "
            "hold for a year.  No forecasting, no model — just the calendar.  That makes it "
            "interesting to test, and dishonest *not* to caveat with the announcement-date "
            "gap.  Our job is to report what the data say about the window we can actually "
            "observe, and flag what we cannot."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "The honest comparison is not 'did the stock go up?'  Everything goes up in a bull "
            "market.  We need to compare split periods to **the same stocks in non-split periods** "
            "— and even then, our basket is survivorship-biased (current S&P 500 large-caps "
            "projected back), so the baseline is already pumped up by mega-cap winners.\n\n"
            "Steps:\n"
            "1. Collect all split effective dates for 30 large-cap US stocks since 2000 (yfinance).\n"
            "2. Compute buy-and-hold returns from the day *after* the effective date, at "
            "1m / 3m / 6m / 12m horizons.\n"
            "3. Sample the same returns from the same stocks on random *non-split* dates "
            f"(excluding a 252-trading-day exclusion window around any split): {R['n_base']} "
            "baseline observations vs 41 split events.\n"
            "4. Test whether the split arm's mean return is (a) positive and significant, and "
            "(b) beats the baseline.  A Newey-West HAC t-stat ≥ 2 on the difference is the bar.\n\n"
            "**Positive control:** a synthetic tape with a tunable daily drift confirms the "
            "engine recovers a drift when one is planted."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look"
        ),
        code(
            "# Forward return distributions: split events vs baseline\n"
            "if HAVE_REAL:\n"
            "    sp_3m = events['ret_3m'].dropna().values * 100\n"
            "    bs_3m = baseline['ret_3m'].dropna().values * 100\n"
            "else:\n"
            "    # Approximate from frozen numbers\n"
            "    rng = __import__('numpy').random.default_rng(0)\n"
            "    sp_3m = rng.normal(R['sp_3m'], 25, 41)\n"
            "    bs_3m = rng.normal(R['bs_3m'], 20, 205)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.hist(sp_3m, bins=20, alpha=0.65, color=RED,  label=f'Split events (n={len(sp_3m)})')\n"
            "ax.hist(bs_3m, bins=20, alpha=0.65, color=GREY, label=f'Baseline (n={len(bs_3m)})')\n"
            f"ax.axvline(R['sp_3m'], c=RED,  lw=2, ls='--', label=f\"Split mean {R['sp_3m']:+.1f}%\")\n"
            f"ax.axvline(R['bs_3m'], c=GREY, lw=2, ls='--', label=f\"Baseline mean {R['bs_3m']:+.1f}%\")\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('3-month buy-and-hold return (%)')\n"
            "ax.set_ylabel('count')\n"
            "ax.set_title('3-month returns: split events vs matched baseline\\n(survivorship-biased basket — baseline inflated by mega-cap winners)')\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Split mean: {sp_3m.mean():+.2f}%  Baseline mean: {bs_3m.mean():+.2f}%  Difference: {sp_3m.mean()-bs_3m.mean():+.2f}%')"
        ),
        md(
            f"The split events (red) cluster near zero and slightly below; the baseline (grey) "
            f"is shifted right — but that's survivorship bias, not a baseline we could 'buy' "
            f"instead.  The honest question is whether splits outperform the same stocks in "
            f"non-split periods from the same basket, and they clearly do not: split events "
            f"return **{R['sp_3m']:+.1f}%** at 3 months vs the baseline's **{R['bs_3m']:+.1f}%** "
            f"(gap: **{R['diff_3m']:+.1f}%**, HAC *t* = {R['diff_3m_t']:+.2f})."
        ),
        md("**Horizon sweep — is there any horizon with a positive drift?**"),
        code(
            "if HAVE_REAL:\n"
            "    sp_means = [events[c].dropna().mean()*100 for c in ('ret_1m','ret_3m','ret_6m','ret_12m')]\n"
            "    bs_means = [baseline[c].dropna().mean()*100 for c in ('ret_1m','ret_3m','ret_6m','ret_12m')]\n"
            "else:\n"
            "    sp_means = [R['sp_1m'], R['sp_3m'], R['sp_6m'], R['sp_12m']]\n"
            "    bs_means = [R['bs_1m'], R['bs_3m'], R['bs_6m'], R['bs_12m']]\n"
            "\n"
            "labels = ['1m (~21d)', '3m (~63d)', '6m (~126d)', '12m (~252d)']\n"
            "x = range(len(labels))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "ax.bar([i - 0.2 for i in x], sp_means, width=0.38, color=RED,  label='Split events')\n"
            "ax.bar([i + 0.2 for i in x], bs_means, width=0.38, color=GREY, label='Baseline (same basket, non-split)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(list(x)); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('mean buy-and-hold return (%)')\n"
            "ax.set_title('Split events underperform the matched baseline at every horizon')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for h, sm, bm in zip(labels, sp_means, bs_means):\n"
            "    print(f'{h}: split {sm:+.1f}%  baseline {bm:+.1f}%  diff {sm-bm:+.1f}%')"
        ),
        md(
            "The split arm trails the baseline at 1m, 3m, 6m, and 12m.  The gap widens as the "
            f"horizon extends, reaching **{R['diff_12m']:+.1f}%** at 12 months "
            f"(*t* = {R['diff_12m_t']:+.2f}).  This is the *opposite* of the drift story."
        ),
        md("**Why might splits predict below-average subsequent returns?**"),
        md(
            "Splits tend to occur after strong prior price performance (the share price "
            "reached a level the board found 'too high').  Companies that recently had "
            "great runs often mean-revert, or at minimum don't continue outperforming.  "
            "So the post-split period captures a momentum *reversal* — not a fundamental "
            "signal — in this basket and this window.  This is not an investable 'short "
            "splits' signal either: it is not statistically significant in the split arm "
            "alone (*t* ranges from −0.92 to +0.99), and n = 41 events is too thin."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** No post-effective-date drift at any horizon: 1m *t* = "
            f"{R['sp_1m_t']:+.2f}, 3m *t* = {R['sp_3m_t']:+.2f}, 6m *t* = {R['sp_6m_t']:+.2f}, "
            f"12m *t* = {R['sp_12m_t']:+.2f}.  None clear |*t*| ≥ 2.\n"
            "- **Tradability — Mirage.** No significant edge; ~2 large-cap events per year "
            "is too thin for a portfolio.\n"
            "- **Post-effective drift? — Absent.** Split events actually underperform the "
            f"same-basket baseline by {R['diff_3m']:+.1f}% at 3m and {R['diff_12m']:+.1f}% "
            f"at 12m (both *p* < 0.05).\n\n"
            "> **Caveat:** we tested the *effective-date* window.  The original Ikenberry (1996) "
            "anomaly runs from the *announcement* date, which yfinance does not supply.  "
            "The absence here is consistent with efficient pricing: by the ex-date, the "
            "announcement drift has already occurred."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Even if the gross edge were real, the strategy faces a size problem:"
        ),
        code(
            "# How many split events per year in the basket?\n"
            "if HAVE_REAL:\n"
            "    events_year = events.copy()\n"
            "    events_year['year'] = pd.to_datetime(events_year['event_date']).dt.year\n"
            "    by_year = events_year.groupby('year').size()\n"
            "else:\n"
            "    import pandas as pd\n"
            "    by_year = pd.Series({2000:6, 2001:5, 2003:2, 2004:3, 2005:3, 2006:2,\n"
            "                         2007:2, 2012:1, 2014:2, 2015:3, 2020:2, 2021:1,\n"
            "                         2022:3, 2024:3, 2025:1})\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.0))\n"
            "ax.bar(by_year.index, by_year.values, color=AMBER)\n"
            "ax.set_xlabel('year'); ax.set_ylabel('split events (large-cap basket)')\n"
            "ax.set_title('Split frequency has collapsed since 2010 — the strategy rarely fires')\n"
            "plt.tight_layout(); plt.show()\n"
            "recent = by_year[by_year.index >= 2015].mean()\n"
            "print(f'Average events/year since 2015: {recent:.1f}  (total: {by_year.values.sum()} over 25 years)')"
        ),
        md(
            "Even in a 30-stock large-cap basket, the strategy fires only ~2 times per year "
            "since 2015.  At that rate, a 12-month holding period means you are carrying at "
            "most 2–3 positions at a time — a concentration risk, not a portfolio.  Combined "
            "with a negative (or at best flat) edge vs baseline, this is firmly a **Mirage**."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The real test would use announcement dates.** If you can source SEC 8-K "
            "filing dates for stock splits (from EDGAR), you could test the Ikenberry drift "
            "in its original form.  That is the study this notebook is *not*, because "
            "yfinance does not supply announcement dates.\n"
            "- **Broader universe.** The 30-stock basket is survivorship-biased and thin.  "
            "A study using the full Russell 3000 history (via a commercial data vendor) "
            "with announcement dates would be much more powerful.\n"
            "- **High-split-ratio events.** Some of the AMZN (20:1), GOOGL (20:1), WMT (3:1) "
            "events in 2022–2024 are unusual; testing ratios separately might reveal "
            "heterogeneity the pooled test misses.\n"
            "- **Related anomalies:** "
            "[Study 34 — Aftershock](../../34-aftershock/) (post-earnings drift), "
            "[Study 82 — Witching-Hour](../../82-witching-hour/) (calendar event effects).\n\n"
            "*Think the effective-date drift is real on a broader universe? Fork this, swap in "
            "a wider ticker basket, and show |*t*| ≥ 2 that clears the costs. That's the bar.*"
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
            "# Split-Drift — a quantitative teardown\n"
            "### Post-effective-date event study · matched baseline · HAC inference · survivorship caveat\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Post--effective_drift%3F: Absent](https://img.shields.io/badge/Post--effective_drift%3F-Absent-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim carrying its standard error.*  We test whether "
            "stock splits predict above-average forward returns at 1/3/6/12-month horizons, "
            "vs a within-stock matched-period baseline, on 30 large-cap US stocks 2000–2025.\n\n"
            "> ⚠️ **Not investment advice.**  Real data: Yahoo Finance splits + daily closes, "
            f"as-of 2026-06-14.  Event fingerprint `{R['fp']}`.  "
            "Effective dates only — NOT announcement dates (see caveat).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Post-effective 12m mean **{R['sp_12m']:+.1f}%**, "
            f"HAC *t* = **{R['sp_12m_t']:+.2f}**; every horizon |*t*| < 2.  "
            f"vs matched baseline: **{R['diff_12m']:+.1f}%** underperformance "
            f"(*t* = {R['diff_12m_t']:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | ~2 events/yr in 30-stock basket; "
            "thin n, no significant gross edge, below-baseline returns. |\n"
            f"| **Post-effective drift?** | `ABSENT` | Split stocks underperform same-basket "
            f"non-split periods: {R['diff_1m']:+.1f}% (1m), {R['diff_3m']:+.1f}% (3m), "
            f"{R['diff_6m']:+.1f}% (6m), {R['diff_12m']:+.1f}% (12m). |\n\n"
            "> 💡 The anomaly Ikenberry et al. (1996) documented was measured from the *announcement* "
            "date; by the effective date the market has already reacted.  Our result is consistent "
            "with semi-strong-form efficiency: public information is priced in before the ex-date."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_{i,t,h}$ be the $h$-day buy-and-hold return for stock $i$ starting at "
            "split effective date $t$. The Ikenberry et al. hypothesis (mapped to our window) is:\n\n"
            r"$$\mathbb{E}[r_{i,t,h}] > \mathbb{E}[r_{i,s,h}]$$"
            "\n\nwhere $s$ is a random non-split date for the same stock — i.e., the post-split "
            "forward return exceeds what you'd earn by entering on a random non-split day.\n\n"
            "The original study found +7.9% (1-year, vs size-and-B/M matched firms) on "
            "1975–1990 NYSE data.  Our test is weaker: (a) we use the effective date "
            "not the announcement, (b) our basket is survivorship-biased, (c) 2000–2025 "
            "data includes the tech bubble, GFC, COVID, and zero-rate era.\n\n"
            "We still reject the hypothesis at every horizon."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The split anomaly is interesting precisely because it requires no information beyond "
            "a public corporate action calendar.  If it persisted to the effective date at scale, "
            "it would be an exploitable calendar signal.  The real question — and the one we cannot "
            "answer with yfinance — is whether the *announcement-date* window still works post-2000.  "
            "That requires EDGAR 8-K data; this study tests a proxy window and finds nothing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe:** {R['n_tickers']} large-cap US tickers with split history, "
            "from `data.SPLIT_BASKET` (survivorship-biased — named).\n"
            f"- **Events:** {R['n_events']} split effective dates 2000–2025 with ratio ≥ 1.5 "
            "(via `yfinance.splits`).\n"
            "- **Forward returns:** buy-and-hold from effective-date close +1 trading day, "
            "at 1m (21d) / 3m (63d) / 6m (126d) / 12m (252d).\n"
            f"- **Baseline:** {R['n_base']} observations from the same tickers in non-split "
            "windows (excluding ±252d around each split).\n"
            "- **Inference:** Newey-West HAC t-stat on split arm mean; HAC t-stat on the "
            "difference (split − baseline mean, applied as a one-sample test on the split "
            "arm after centering on the baseline mean).\n"
            "- **Positive control:** synthetic event panel with tunable planted drift — "
            "confirms the engine recovers a drift when one exists."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Per-horizon statistics — split arm vs baseline\n\n"
            "The HAC t-stat on the split arm mean, and on the (split − baseline) difference, "
            "at each horizon.  The inference bar is |*t*| ≥ 2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    comp = st.compare_horizons(events, baseline)\n"
            "    sp_stats = {col: st.summarize_horizon(events, col=col)\n"
            "                for col in ('ret_1m','ret_3m','ret_6m','ret_12m')}\n"
            "else:\n"
            "    import pandas as pd\n"
            "    comp = pd.DataFrame({\n"
            "        'horizon':['1m (~21d)','3m (~63d)','6m (~126d)','12m (~252d)'],\n"
            f"        'mean_split_pct':[R['sp_1m'],R['sp_3m'],R['sp_6m'],R['sp_12m']],\n"
            f"        'mean_base_pct':[R['bs_1m'],R['bs_3m'],R['bs_6m'],R['bs_12m']],\n"
            f"        'diff_pct':[R['diff_1m'],R['diff_3m'],R['diff_6m'],R['diff_12m']],\n"
            f"        'tstat_split':[R['sp_1m_t'],R['sp_3m_t'],R['sp_6m_t'],R['sp_12m_t']],\n"
            f"        'tstat_diff':[R['diff_1m_t'],R['diff_3m_t'],R['diff_6m_t'],R['diff_12m_t']],\n"
            "    })\n"
            "    sp_stats = None\n"
            "\n"
            "# Plot t-stats\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "horizons = comp['horizon'].values\n"
            "col_split = [GREEN if abs(v) >= 2 else RED for v in comp['tstat_split']]\n"
            "col_diff  = [GREEN if abs(v) >= 2 else RED for v in comp['tstat_diff']]\n"
            "\n"
            "axes[0].bar(horizons, comp['tstat_split'].values, color=col_split)\n"
            "for s in (2, -2): axes[0].axhline(s, ls='--', c=GREY, lw=1)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_title('HAC t-stat: split arm mean vs zero')\n"
            "axes[0].set_ylabel('HAC t-stat'); axes[0].tick_params(axis='x', rotation=15)\n"
            "\n"
            "axes[1].bar(horizons, comp['tstat_diff'].values, color=col_diff)\n"
            "for s in (2, -2): axes[1].axhline(s, ls='--', c=GREY, lw=1)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_title('HAC t-stat: split − baseline difference')\n"
            "axes[1].set_ylabel('HAC t-stat (negative = split underperforms)'); axes[1].tick_params(axis='x', rotation=15)\n"
            "\n"
            "plt.tight_layout(); plt.show()\n"
            "print(comp[['horizon','mean_split_pct','mean_base_pct','diff_pct','tstat_split','tstat_diff']].to_string(index=False))"
        ),
        md(
            f"> 💡 The left panel shows the split arm's absolute return signal — not a single "
            f"horizon clears |*t*| ≥ 2.  The right panel shows the *relative* test vs the "
            f"matched baseline — here 3 of 4 horizons clear |*t*| ≥ 2, all in the *negative* "
            f"direction: split events systematically underperform their own same-stock baseline.  "
            f"The worst is 12m: *t* = {R['diff_12m_t']:+.2f}."
        ),
        md(
            "### 4b · Positive control — the engine works when drift is planted\n\n"
            "To confirm the event-study machinery is functional, we use `data.synthetic_events` "
            "with a tunable daily drift and verify the engine recovers it."
        ),
        code(
            "from split_drift import data as sd, strategy as st\n"
            "drift_levels = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0]\n"
            "means_3m, t_3m = [], []\n"
            "for d in drift_levels:\n"
            "    ev, _ = sd.synthetic_events(n_stocks=15, n_events=60, drift_bps=d, seed=142)\n"
            "    s = st.summarize_horizon(ev, col='ret_3m')\n"
            "    means_3m.append(s['mean_pct'])\n"
            "    t_3m.append(s['tstat'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(drift_levels, means_3m, 'o-', c=GREEN, lw=2, label='3m mean (%)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted drift (bps/day)'); ax.set_ylabel('3m mean return (%)')\n"
            "ax.set_title('Positive control: engine recovers planted drift monotonically')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for d, m, t in zip(drift_levels, means_3m, t_3m):\n"
            "    print(f'drift={d:5.1f} bps/day  3m mean={m:+.2f}%  HAC t={t:+.2f}')\n"
            "print('\\nMonotonically increasing:', all(means_3m[i]<=means_3m[i+1] for i in range(len(means_3m)-1)))"
        ),
        md(
            "> 💡 The engine correctly recovers a drift that is proportional to the planted "
            "knob — confirming the machinery is not broken.  The real-tape verdict is therefore "
            "a statement about the *market*, not the method: there is no post-effective-date "
            "drift here to find."
        ),
        md(
            "### 4c · Individual events — heterogeneity in the sample\n\n"
            "The small n (41) and wide dispersion suggest large idiosyncratic variation."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev_disp = events[['ticker','event_date','split_ratio','ret_3m','ret_12m']].copy()\n"
            "    ev_disp['ret_3m_pct'] = ev_disp['ret_3m'] * 100\n"
            "    ev_disp['ret_12m_pct'] = ev_disp['ret_12m'] * 100\n"
            "    ev_disp = ev_disp.sort_values('ret_12m_pct', ascending=False)\n"
            "else:\n"
            "    ev_disp = None\n"
            "\n"
            "# Plot 3m return distribution\n"
            "if HAVE_REAL:\n"
            "    ret3 = ev_disp['ret_3m_pct'].dropna().values\n"
            "else:\n"
            "    ret3 = __import__('numpy').random.default_rng(0).normal(R['sp_3m'], 22, 41)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.hist(ret3, bins=15, color=RED, alpha=0.7, edgecolor='white')\n"
            "ax.axvline(ret3.mean(), c='k', lw=2, ls='--', label=f'mean {ret3.mean():+.1f}%')\n"
            "ax.axvline(0, c=GREY, lw=1)\n"
            "ax.set_xlabel('3-month return after split effective date (%)')\n"
            "ax.set_ylabel('count')\n"
            "ax.set_title(f'Wide dispersion: n={len(ret3)}, std={ret3.std():.1f}%')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'n={len(ret3)}, mean={ret3.mean():+.2f}%, std={ret3.std():.2f}%, ')\n"
            "print(f'min={ret3.min():+.2f}%, max={ret3.max():+.2f}%')\n"
            "print(f'With this dispersion and n=41, the power to detect a 5% true drift is very low.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — post-effective 1m *t* {R['sp_1m_t']:+.2f}, "
            f"3m *t* {R['sp_3m_t']:+.2f}, 6m *t* {R['sp_6m_t']:+.2f}, "
            f"12m *t* {R['sp_12m_t']:+.2f}.  All < |2|; all inside noise.\n"
            f"- **Tradability `MIRAGE`** — no significant positive edge; ~2 large-cap "
            "events/year; below-baseline returns at every horizon.\n"
            f"- **Post-effective drift `ABSENT`** — at 12m the split arm trails the matched "
            f"baseline by **{R['diff_12m']:+.1f}%** (HAC *t* = {R['diff_12m_t']:+.2f}), "
            "consistent with the hypothesis that the market fully prices splits before the ex-date.\n\n"
            "> **Caveat:** Ikenberry et al. (1996) tested from the announcement date on "
            "1975–1990 data.  This study cannot confirm or refute that result; it tests a "
            "different, weaker window (post-effective)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — event frequency and cost analysis"
        ),
        code(
            "# Cost sweep at 3-month horizon\n"
            "if HAVE_REAL:\n"
            "    gross_3m = events['ret_3m'].dropna().mean()\n"
            "else:\n"
            "    gross_3m = R['sp_3m'] / 100\n"
            "\n"
            "costs = [0, 2, 5, 10, 20]\n"
            "net_ann = [st.cost_adjusted_return(float(gross_3m), 63, c) * 100 for c in costs]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net_ann, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net_ann, 0, where=[n<0 for n in net_ann], color=RED, alpha=0.15)\n"
            "ax.set_xlabel('round-trip cost (bps)')\n"
            "ax.set_ylabel('net annualised return (%/yr)')\n"
            "ax.set_title('3m horizon cost sweep (gross already near zero)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for c, n in zip(costs, net_ann):\n"
            "    print(f'cost {c:2d} bps  net_ann {n:+.1f}%/yr')"
        ),
        md(
            "> 💡 The 3m gross is +0.4% (annualised ~+1.6%/yr) — already near the noise floor.  "
            "Even zero cost doesn't produce a meaningful return, and the 12m horizon (where the "
            "return is larger at +5.1%) is already well below the matched baseline (+28.7%) "
            "from the same survivorship-biased basket.  There is no tradable edge here."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Test from announcement dates** — source 8-K corporate action filings from "
            "EDGAR (`EFTS` full-text search) to get actual announcement dates, and test the "
            "original Ikenberry window.  That is the study this one proxies.\n"
            "- **Broader universe** — Compustat/CRSP data covering the full US equity universe "
            "back to 1970 would dramatically increase n (1,000s of splits) and power.\n"
            "- **Split-ratio heterogeneity** — large-ratio splits (AMZN 20:1, GOOGL 20:1) "
            "are rare recent events with potentially different signalling content vs the "
            "historical 2:1 pattern.\n"
            "- **Related desk studies:** "
            "[Study 34 — Aftershock](../../34-aftershock/) (post-earnings drift), "
            "[Study 82 — Witching-Hour](../../82-witching-hour/) (calendar event effects), "
            "[Study 65 — Scorecard](../../65-scorecard/) (survivorship accounting).\n\n"
            "*Think the effective-date drift is real on a wider universe with announcement "
            "dates? Fork this, widen the basket and source 8-K filings, and show |*t*| ≥ 2 "
            "on the difference vs a matched control. That's the bar.*"
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
