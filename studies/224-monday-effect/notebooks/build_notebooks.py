"""Build the two narrative notebooks for Study 224 (Monday Effect).

    python notebooks/build_notebooks.py

Both walk the seven desk beats at two altitudes. Figures are computed live from the
monday_effect package on the cached real tape; the frozen R dict mirrors docs/results.md
so the two never drift.
"""

from __future__ import annotations

import os
import sys

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-16).
R = dict(
    n=8400, start="1993-01-29", end="2026-06-12", fp="eb1569cc9e97",
    mon_mean=5.64, tue_mean=7.17, wed_mean=6.20, thu_mean=1.46, fri_mean=3.32,
    mon_t=2.06, tue_t=2.64, wed_t=2.21, thu_t=0.52, fri_t=1.33,
    mon_diff=1.08, mon_diff_t=0.36, tue_diff=3.03, tue_diff_t=1.00,
    mon_pre=4.78, mon_post=0.09, mon_change=-4.69, mon_change_t=-0.77,
    mon_cagr=1.34, mon_sharpe=0.196, mon_dd=-41.7, mon_tim=19,
    skip_cagr=7.30, skip_sharpe=0.512, skip_dd=-48.9, skip_tim=81,
    b_cagr=10.82, b_sharpe=0.646, b_dd=-55.2,
    mon_cagr_gap=-9.47, skip_cagr_gap=-3.52,
    mon_sharpe_gap=-0.450, skip_sharpe_gap=-0.133,
)

BOOT = "\n".join([
    "import sys, os",
    'sys.path.insert(0, os.path.abspath(".."))',
    'sys.path.insert(0, os.path.abspath("../../.."))',
    "%matplotlib inline",
    "import matplotlib.pyplot as plt",
    'plt.rcParams["figure.figsize"] = (10, 5.5)',
    "import numpy as np, pandas as pd",
    'pd.set_option("display.float_format", lambda v: f"{v:,.3f}")',
    "from monday_effect import data, strategy",
    "",
    'AS_OF = "2026-06-16"',
    'frame = data.load_real("SPY", mode="total_return").loc[:AS_OF]',
    'close = frame["close"]',
    "wm = strategy.weekday_means(close)",
    "bh = strategy.buy_and_hold(close)",
    "mon_only = strategy.backtest(close, strategy.monday_only_position(close), cost_bps=1.0)",
    "skip_mon = strategy.backtest(close, strategy.skip_monday_position(close), cost_bps=1.0)",
    'print(f"SPY total return: {len(close):,} rows  {close.index[0].date()} -> {close.index[-1].date()}  fingerprint={data.fingerprint(frame)}")',
    "print(wm.round(2))",
])

HERO_BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Monday Effect still there?: Busted](https://img.shields.io/badge/Monday_Effect_still_there%3F-Busted-8b949e?style=flat-square)\n\n"
)


def md(t):
    return new_markdown_cell(t)


def code(t):
    return new_code_cell(t)


def J(*lines):
    return "\n".join(lines)


def build_curious():
    bar_code = J(
        "fig, ax = plt.subplots(figsize=(9, 5))",
        "colors = ['C3' if d=='Mon' else ('C1' if d=='Thu' else 'C0') for d in wm.index]",
        "ax.bar(wm.index, wm['mean_bps'], color=colors, alpha=.85)",
        "ax.axhline(0, color='k', lw=.8)",
        "for d, v in wm['mean_bps'].items():",
        "    ax.text(d, v + .15, f'{v:+.1f}', ha='center', fontsize=10)",
        "ax.set_ylabel('Mean daily return (bps)')",
        "ax.set_title('SPY -- average close-to-close return by weekday (1993-2026)')",
        "plt.tight_layout(); plt.show()",
    )
    equity_code = J(
        "fig, ax = plt.subplots(figsize=(10, 5.5))",
        "ax.plot(bh['equity'], label=f\"Buy & hold (CAGR {bh['cagr']*100:.1f}%)\", lw=1.4)",
        "ax.plot(skip_mon['equity'], label=f\"Skip Monday (CAGR {skip_mon['cagr']*100:.1f}%)\", lw=1.2, alpha=.85)",
        "ax.plot(mon_only['equity'], label=f\"Buy Monday only (CAGR {mon_only['cagr']*100:.1f}%)\", lw=1.2, alpha=.85)",
        "ax.set_yscale('log'); ax.set_ylabel('Growth of $1 (log)'); ax.legend()",
        "ax.set_title('Calendar rules vs simply holding SPY')",
        "plt.tight_layout(); plt.show()",
    )
    cells = [
        md(J(
            "# The Monday Effect",
            "### Is Monday still the market worst day, or did that ghost die decades ago?",
            "",
            HERO_BADGES,
            "For decades the folklore has said Monday is the worst trading day -- the",
            "*Monday Effect* -- so you should avoid it. We measure the average close-to-close",
            "return on each weekday of SPY (dividends included) and ask: is Monday really",
            "negative? (**No**). Could you trade around it? (**No**).",
            "",
            "> This is the plain-language layer. HAC t-stats and the pre/post-2000 split are in",
            "> **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.",
            ">",
            "> Not investment advice. Educational, reproducible research.",
        )),
        code(BOOT),
        md(J(
            "## The answer first",
            "",
            "| Question | Answer |",
            "|---|---|",
            f"| Is Monday negative? | **No** -- Monday averages **+{R['mon_mean']} bps**, positive. |",
            f"| Weakest day? | **Thursday** (+{R['thu_mean']} bps). |",
            f"| Can you trade around it? | **No** -- buy-Monday earns **{R['mon_cagr']}%/yr** vs **{R['b_cagr']}%/yr** B&H (**{abs(R['mon_cagr_gap']):.0f} pts/yr less**). |",
        )),
        md(J(
            "## 1 -- The claim",
            "",
            "Kenneth French (1980) found significantly negative Monday close-to-close returns",
            "on 1953-1977 S&P data. The question is whether it survived decades of arbitrage.",
        )),
        md("## 2 -- The teardown\n\nFirst, the bar chart of weekday returns."),
        code(bar_code),
        md(
            f"**Monday (red) is positive** (+{R['mon_mean']} bps). Thursday (orange) is weakest "
            f"(+{R['thu_mean']} bps). All days positive -- the negative-Monday effect is gone."
        ),
        md("Now the tradability test: literal rules vs holding SPY."),
        code(equity_code),
        md(J(
            f"Not close. Buy-Monday: {R['mon_cagr']}%/yr vs {R['b_cagr']}%/yr B&H",
            f"(**{abs(R['mon_cagr_gap']):.0f} pts/yr lost** sitting in cash 81% of time).",
            f"Skip-Monday: {R['skip_cagr']}%/yr ({abs(R['skip_cagr_gap']):.0f} pts/yr lost).",
            "Avoiding a positive day costs you the equity premium.",
        )),
        md(J(
            "## 3 -- The verdict",
            "",
            f"- **Monday Effect still there? Busted.** Monday is +{R['mon_mean']} bps, positive pre- and post-2000.",
            f"- **Signal -- None.** Monday-rest +{R['mon_diff']} bps (HAC t +{R['mon_diff_t']:.2f}): wrong sign.",
            f"- **Trade it? Mirage.** Buy-Monday -{abs(R['mon_cagr_gap']):.0f} pts/yr, skip-Monday -{abs(R['skip_cagr_gap']):.0f} pts/yr vs B&H.",
        )),
        md(J(
            "## 4 -- Going further",
            "",
            "- Extend to **^GSPC** 1950s data (price-only) to find when the effect vanished.",
            "- Is there a *Monday-after-down-Friday* conditional effect?",
            "- International evidence in markets with less arbitrage capacity?",
        )),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "01_for_the_curious.ipynb")


def build_quants():
    contrast_code = J(
        "mon = strategy.contrast(close, 0)",
        "tue = strategy.contrast(close, 1)",
        "print(f\"Monday  - rest: {mon['diff_bps']:+.2f} bps   HAC t {mon['hac_t']:+.2f}\")",
        "print(f\"Tuesday - rest: {tue['diff_bps']:+.2f} bps   HAC t {tue['hac_t']:+.2f}\")",
        "print('Monday is positive and non-significant: effect absent, not merely weak.')",
    )
    sp_code = J(
        "sp = strategy.subperiod_effect(close, 0, cut='2000-01-01')",
        "print(f\"Monday pre-2000:  {sp['pre_diff_bps']:+.2f} bps\")",
        "print(f\"Monday post-2000: {sp['post_diff_bps']:+.2f} bps\")",
        "print(f\"Change: {sp['change_bps']:+.2f} bps  HAC t(change): {sp['hac_t_change']:+.2f}\")",
        "print('Monday positive in BOTH halves. Cannot certify decay.')",
    )
    timer_code = J(
        "import pandas as pd",
        "tbl = pd.DataFrame({",
        "    'CAGR %':  [mon_only['cagr']*100, skip_mon['cagr']*100, bh['cagr']*100],",
        "    'Vol %':   [mon_only['vol']*100, skip_mon['vol']*100, bh['vol']*100],",
        "    'Sharpe':  [mon_only['sharpe'], skip_mon['sharpe'], bh['sharpe']],",
        "    'MaxDD %': [mon_only['max_dd']*100, skip_mon['max_dd']*100, bh['max_dd']*100],",
        "    'TiM %':   [mon_only['time_in_market']*100, skip_mon['time_in_market']*100, bh['time_in_market']*100],",
        "}, index=['Buy Monday only', 'Skip Monday', 'Buy & hold'])",
        "tbl.round(2)",
    )
    cells = [
        md(J(
            "# The Monday Effect -- a quantitative teardown",
            "### Real total-return tape -- per-weekday HAC inference -- pre/post-2000 difference",
            "",
            HERO_BADGES,
            "The deep companion to the [curious notebook](01_for_the_curious.ipynb).",
            "We test three assertions: Monday is negative (false), a calendar rule beats",
            "the index (mirage), and the effect may have decayed (not testable -- never",
            "negative on this tape).",
            "",
            "> Not investment advice. SPY daily, total-return; cash 0%; 1 bp/switch; no execution lag.",
            "> Sources: docs/references.md. Run: docs/results.md.",
            "",
            "**vs Study 90 (Weekend):** Study 90 used overnight returns; this study uses",
            "close-to-close (French 1980 convention). Both find no negative Monday.",
        )),
        code(BOOT),
        md(J(
            "## Verdict, up front",
            "",
            "| Axis | Stamp | Why |",
            "|---|---|---|",
            f"| **Signal** | NONE | Monday-rest **+{R['mon_diff']} bps** (HAC *t* **+{R['mon_diff_t']:.2f}**), wrong sign and insignificant. |",
            f"| **Tradability** | MIRAGE | Buy-Monday **{R['mon_cagr']}%** CAGR vs B&H **{R['b_cagr']}%** ({R['mon_cagr_gap']:.1f} pts/yr). |",
            f"| **Monday Effect still there?** | BUSTED | Monday pre-2000 +{R['mon_pre']} bps, post-2000 +{R['mon_post']} bps; change HAC t {R['mon_change_t']}. |",
        )),
        md(J(
            "## 1 -- Protocol",
            "",
            "Per-weekday mean with HAC Newey-West *t* (daily returns mildly autocorrelated) --",
            "Monday-vs-rest contrast as difference of means with HAC SEs in quadrature --",
            "pre-2000/post-2000 split with **test of the change** on the pre-registered millennium cut --",
            "literal timers net of 1 bp/switch. **None trigger:** Monday contrast wrong sign and |t| << 2.",
        )),
        md("## 2 -- Per-weekday means and the Monday contrast"),
        code("wm.round(3)"),
        code(contrast_code),
        md(J(
            "Monday-vs-rest: **+1.08 bps** (HAC t +0.36), **wrong sign**.",
            "With five weekday tests in play, no contrast would survive a snooping correction.",
            "The negative-Monday Effect is absent from this tape entirely.",
        )),
        md("## 3 -- Pre/post-2000 split with a test of the change"),
        code(sp_code),
        md(J(
            "The Monday Effect predates the SPY era. French found it on 1953-1977 data;",
            "on the 1993-2026 tape Monday is positive in both sub-periods.",
            "Change t = -0.77: not significant. We cannot certify decay -- nothing was there.",
        )),
        md("## 4 -- Literal timers net of 1 bp/switch"),
        code(timer_code),
        md(J(
            "Lost equity premium on sat-out days swamps any weekday tilt.",
            "Buy-Monday sits in cash 81% of the time and loses 9.5 pts/yr to buy-and-hold.",
        )),
        md(J(
            "## 5 -- The verdict",
            "",
            f"Signal NONE (Monday-rest +{R['mon_diff']} bps, HAC t +{R['mon_diff_t']:.2f} -- wrong sign),",
            f"Tradability MIRAGE (buy-Monday {R['mon_cagr_gap']:.1f} pts/yr, skip-Monday {R['skip_cagr_gap']:.1f} pts/yr vs B&H),",
            f"Monday Effect still there? BUSTED (positive pre- and post-2000, change t {R['mon_change_t']}).",
            "French 1980 is not certifiable on the 1993-2026 SPY tape.",
        )),
        md(J(
            "## 6 -- Going further",
            "",
            "- Extend to **price-only ^GSPC** from 1950s to locate exactly when the effect vanished.",
            "- Apply a **White Reality Check** over all five weekday rules for a snooping-aware p-value.",
            "- **Conditional Monday:** Monday-after-a-down-Friday -- does the unconditional flat hide a tilt?",
        )),
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
