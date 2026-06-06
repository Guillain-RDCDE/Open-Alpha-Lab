"""Generate the two narrative notebooks from source, then they are executed.

Why a builder script instead of hand-edited .ipynb? Reproducibility: the
notebooks are a *generated artefact*. Edit the cell text here, re-run
``python notebooks/build_notebooks.py`` to rebuild the skeletons, then execute
with nbconvert to embed the figures/outputs:

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Two audiences, two files:
  01_for_the_curious  — plain-language story, no jargon
  02_for_the_quants   — real data, critique, statistics, execution realism
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# A header every notebook runs to find the package and use inline figures.
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5.5)
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
"""


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Why do stock markets make their money *overnight*? 🌙\n"
            "### A real market anomaly — and why it's subtler than it looks\n\n"
            "Over the last 30 years, almost all the gains of the world's big stock "
            "markets piled up **while the market was closed** (from one day's close to "
            "the next morning's open). The **daytime session** (open → close) is nearly "
            "flat. Strange, isn't it?\n\n"
            "This notebook tells the story **without jargon**. For the rigorous version "
            "(real data, statistics, costs), head to "
            "[`02_for_the_quants.ipynb`](02_for_the_quants.ipynb).\n\n"
            "> ⚠️ **This is not investment advice.** Educational and research tool."
        ),
        code(BOOT),
        md(
            "## 1. The pattern: the night rises, the day stalls\n\n"
            "Let's build a *toy* market that is completely honest: each night it drifts a "
            "hair upward (+3 basis points, i.e. +0.03%), each day a hair downward, and "
            "**everything else is pure noise — no fraud, no conspiracy**. What does the "
            "night/day decomposition show?"
        ),
        code(
            "from overnight import decompose, diagnostics\n\n"
            "ohlc = diagnostics.synthetic_ohlc(overnight_bias_bps=3, intraday_bias_bps=-1, seed=0)\n"
            "dec = decompose.decompose(ohlc)\n\n"
            "ax = plt.subplot()\n"
            "ax.plot(dec.index, dec['cum_overnight']*100, label='Overnight (close→open)', lw=2)\n"
            "ax.plot(dec.index, dec['cum_intraday']*100, label='Intraday (open→close)', lw=2)\n"
            "ax.plot(dec.index, dec['cum_close_close']*100, label='Buy & hold', color='grey', lw=1.2)\n"
            "ax.set_title('Toy market — no fraud, just a 3 bps overnight bias')\n"
            "ax.set_ylabel('Cumulative return (%)'); ax.legend(); ax.grid(alpha=.3)\n"
            "plt.show()\n"
            "s = decompose.summary(dec)\n"
            "print(f\"Overnight cumulative: {s.loc['overnight','cum_return']*100:+.0f}%   \"\n"
            "      f\"Intraday cumulative: {s.loc['intraday','cum_return']*100:+.0f}%\")"
        ),
        md(
            "**We just reproduced Knuteson's pattern with zero manipulation.** A tiny "
            "constant bias, repeated ~250 nights a year for 32 years, is enough. The "
            "night *looks* magical. Hold that thought: a very small edge, repeated "
            "thousands of times, becomes huge. That's our first trap."
        ),
        md(
            "## 2. Trap #1 — the scale lies (the magic of compounding)\n\n"
            "The article's charts use a **logarithmic** scale, where you quickly read "
            "\"billions of %\". But where does that dizzying number come from? Not from "
            "fraud: from **compounding**. Look at what a simple constant bias becomes, by "
            "size and horizon:"
        ),
        code(
            "table = diagnostics.compounding_table()\n"
            "diagnostics.format_compounding(table)"
        ),
        md(
            "A bias of **1 basis point per night** — totally innocent, undetectable — "
            "compounds to three digits over 30 years. At 30 bps, you reach *trillions* of "
            "percent. **The explosion comes from the exponent, not from a conspiracy.** "
            "Any spectacular magnitude shown on a log axis must pass this sanity check first."
        ),
        md(
            "## 3. Trap #2 — dirty data manufactures the signal\n\n"
            "Free data (Yahoo) mis-handles some *splits* and dividends, especially in "
            "emerging markets. A handful of corrupted prices is enough to **mechanically "
            "shift return from the day into the night**. Demonstration on a perfectly flat "
            "market (zero bias) where we dirty 3 prices:"
        ),
        code(
            "flat = diagnostics.synthetic_ohlc(overnight_bias_bps=0, intraday_bias_bps=0, seed=1)\n"
            "clean = decompose.decompose(flat)\n"
            "dirty = decompose.decompose(diagnostics.inject_split_artifact(flat, factor=1.5))\n"
            "print(f\"Overnight cumulative  BEFORE: {clean['cum_overnight'].iloc[-1]*100:+.1f}%\")\n"
            "print(f\"Overnight cumulative  AFTER : {dirty['cum_overnight'].iloc[-1]*100:+.1f}%   (3 dirtied prices)\")\n"
            "flags = diagnostics.flag_suspicious_returns(dirty)\n"
            "print(f\"\\nThe automatic detector flags {len(flags)} suspicious day(s):\")\n"
            "flags[['r_overnight','r_intraday']]"
        ),
        md(
            "Three data errors, and the \"overnight performance\" flips from red to bright "
            "green. This is exactly the mechanism behind the wild numbers of certain "
            "emerging markets in the article. **Before crying scandal, check your data.**"
        ),
        md(
            "## 4. Trap #3 — fees erase the gain\n\n"
            "Suppose the night effect is real (it partly is). Can you *trade* it? The "
            "\"buy at the close, sell at the open\" strategy pays the bid/ask spread "
            "**twice a day, ~250 days a year**. Let's see what's left once we subtract "
            "realistic costs:"
        ),
        code(
            "from overnight import backtest\n"
            "sweep = backtest.cost_sweep(dec, roundtrip_bps=(0,1,2,3,5,8))\n"
            "view = sweep.copy()\n"
            "view['cagr_net'] = (view['cagr_net']*100).map('{:+.2f}%'.format)\n"
            "view['sharpe_net'] = view['sharpe_net'].map('{:+.2f}'.format)\n"
            "view['max_drawdown'] = (view['max_drawdown']*100).map('{:.0f}%'.format)\n"
            "view.columns = ['Net annual return', 'Net Sharpe', 'Worst drawdown']\n"
            "view.index.name = 'Round-trip cost (bps)'\n"
            "view"
        ),
        md(
            "At **0 fees**, the Sharpe is decent (~0.7). At a realistic **5 basis points** "
            "round-trip, the gain turns **negative**. This is precisely the fate of the "
            "\"night effect\" ETFs NSPY and NIWM: launched June 2022, **liquidated August "
            "2023** after heavy underperformance.\n\n"
            "> *A strategy that's beautiful on paper is worth no more than the paper until "
            "it has paid the real costs of execution.*"
        ),
        md(
            "## In a nutshell\n\n"
            "| | |\n|---|---|\n"
            "| ✅ **The fact is real** | the night did outperform the day, over decades |\n"
            "| ⚠️ **But the numbers are inflated** | compounding + log scale + dirty data |\n"
            "| ❌ **And hard to exploit** | fees erase the edge |\n\n"
            "The night/day anomaly is a beautiful case study: **real, fascinating, but to "
            "be handled with rigour**. For the numbers-on-real-data version — China test, "
            "artefacts, statistics, beta vs alpha — see "
            "[`02_for_the_quants.ipynb`](02_for_the_quants.ipynb).\n\n"
            "*Further reading:* the original articles and the academic literature (with a "
            "map of who argues what) are listed in [`docs/references.md`](../docs/references.md); "
            "grab the PDFs with `python papers/download_papers.py`."
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
            "# The overnight anomaly — quantitative analysis\n"
            "### Real data, critical teardown, statistics, execution realism\n\n"
            "Rigorous version of the [notebook for the curious](01_for_the_curious.ipynb). "
            "It tackles four questions a sceptical quant asks immediately:\n\n"
            "1. **Does the pattern hold on real data, everywhere?** (spoiler: no — and that's instructive)\n"
            "2. **Is the overnight Sharpe distinguishable from zero? Is it alpha or disguised beta?**\n"
            "3. **Does it survive real execution costs?**\n"
            "4. **Is the whole thing reproducible?**\n\n"
            "> ⚠️ **Not investment advice.** Data: Yahoo! Finance via `yfinance`, adjustment "
            "mode `split_only` (choice documented in §3.3). First run hits the network."
        ),
        code(BOOT + "\nfrom overnight import data, decompose, diagnostics, backtest, stats\n"),
        md(
            "## 1. The pattern on 10 world indices (ETFs)\n\n"
            "We decompose each ETF into night / day and read the **annualised Sharpe of "
            "the overnight leg** — the only number that matters to judge an *edge*."
        ),
        code(
            "rows, decs = {}, {}\n"
            "for tk, label in data.WORLD_INDICES.items():\n"
            "    try:\n"
            "        dec = decompose.decompose(data.fetch(tk, mode='split_only'))\n"
            "    except Exception as e:\n"
            "        print(f'{tk}: FAILED {e}'); continue\n"
            "    decs[tk] = dec\n"
            "    s = decompose.summary(dec)\n"
            "    rows[tk] = {\n"
            "        'market': label,\n"
            "        'night cum %': s.loc['overnight','cum_return']*100,\n"
            "        'day cum %': s.loc['intraday','cum_return']*100,\n"
            "        'Sharpe night': s.loc['overnight','sharpe'],\n"
            "        'Sharpe day': s.loc['intraday','sharpe'],\n"
            "        'suspicious days': len(diagnostics.flag_suspicious_returns(dec)),\n"
            "    }\n"
            "table = pd.DataFrame(rows).T\n"
            "table"
        ),
        md(
            "Quick read: the **US (SPY, QQQ)** and **Brazil** show the classic pattern "
            "(huge night, weak or negative day, night Sharpe ~0.7). But look at **Europe "
            "(UK, Germany, France)** and **Japan**: the pattern is **inverted**. That's "
            "not a detail — it's the heart of the teardown."
        ),
        md(
            "## 2. Critical teardown\n\n"
            "### 2.1 The foreign-ETF inversion: we're measuring the time zone, not an anomaly\n\n"
            "Let's visualise night vs day Sharpe by market:"
        ),
        code(
            "t = table.sort_values('Sharpe night')\n"
            "x = np.arange(len(t)); w = 0.4\n"
            "fig, ax = plt.subplots(figsize=(11,5))\n"
            "ax.bar(x-w/2, t['Sharpe night'].astype(float), w, label='Sharpe night', color='#2c7fb8')\n"
            "ax.bar(x+w/2, t['Sharpe day'].astype(float), w, label='Sharpe day', color='#fdae61')\n"
            "ax.axhline(0, color='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels(t.index)\n"
            "ax.set_ylabel('Annualised Sharpe'); ax.set_title('Night vs day by market — the sign flips for foreign ETFs')\n"
            "ax.legend(); ax.grid(axis='y', alpha=.3); plt.show()"
        ),
        md(
            "**Microstructure explanation.** EWU, EWG, EWQ, EWJ are ETFs listed in New "
            "York but whose underlying (London, Frankfurt, Paris, Tokyo) **trades during "
            "the US night**. For these products, the \"overnight\" window (US clock, "
            "close→open) **contains the home market's session**, while the US \"intraday\" "
            "window falls when the underlying is largely closed (the price only moves via "
            "NAV arbitrage).\n\n"
            "In other words: **the night/day split is relative to the listing clock**. "
            "Applied to an instrument offset from its market, it mostly measures the time "
            "zone — not an \"anomaly\". A single global manipulator would struggle to "
            "explain why the sign depends on where the ETF is *listed*. This is a major "
            "caution, and a reminder: **always check what the time window actually "
            "captures for the chosen instrument.**"
        ),
        md(
            "### 2.2 The China test and the T+1 rule\n\n"
            "For Chinese stocks, the literature (Qiao and Dam 2020) documents an "
            "**inverted** pattern (positive day, negative night), cleanly explained by the "
            "**T+1** rule (shares bought one day cannot be sold before the next). Our ETF "
            "proxy (FXI, US-listed) doesn't directly test A-shares, but we note a much "
            "weaker signal than for the US:"
        ),
        code(
            "if 'FXI' in decs:\n"
            "    s = decompose.summary(decs['FXI'])\n"
            "    print('China (FXI) — night Sharpe: {:+.2f}  vs  US (SPY): {:+.2f}'.format(\n"
            "        s.loc['overnight','sharpe'], decompose.summary(decs['SPY']).loc['overnight','sharpe']))\n"
            "    print('A universal, orchestrated pattern should be more geographically homogeneous.')"
        ),
        md(
            "### 2.3 Compounding and selection\n\n"
            "As shown in the notebook for the curious, log-scale magnitudes are dominated "
            "by **compounding** (a 1 bps bias → +124% over 32 years) and by **selection** "
            "(the most spectacular figures are, by the author's own admission, \"the 25 "
            "most problematic\"). The **suspicious-days counter** (column in §1) stays low "
            "here because the US ETFs are clean — the artefacts live mostly in the **raw "
            "emerging spot indices** (e.g. `^BSESN`), not these ETFs. A caveat to keep for "
            "any reproduction of the famous India Figure 8."
        ),
        md(
            "## 3. Statistical rigour & risk\n\n"
            "### 3.1 Is the overnight Sharpe distinguishable from zero? (bootstrap)\n\n"
            "Could a Sharpe of 0.77 on a finite sample be noise? 95% confidence interval "
            "by bootstrap (2000 resamples):"
        ),
        code(
            "for tk in ['SPY','QQQ','EWZ','FXI']:\n"
            "    if tk not in decs: continue\n"
            "    r = stats.sharpe_ci_bootstrap(decs[tk]['r_overnight'], n_boot=2000, seed=0)\n"
            "    print(f\"{tk:4s} night Sharpe = {r['sharpe']:+.2f}  \"\n"
            "          f\"95% CI [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]  \"\n"
            "          f\"P(Sharpe<0) = {r['frac_negative']:.1%}  (n={r['n_obs']})\")"
        ),
        md(
            "For SPY/QQQ/Brazil the interval clears zero by a wide margin: the effect is "
            "**statistically real**. So the question is not *\"does it exist?\"* but *\"is "
            "it exploitable alpha?\"* — hence the next two sections."
        ),
        md(
            "### 3.2 Alpha, or disguised beta?\n\n"
            "Holding the market *every night* means carrying **gap risk** permanently. So "
            "part of the \"overnight alpha\" is a **beta risk premium**, not a distinct "
            "edge. We regress the night leg on the market (close-close): "
            "`r_night = α + β·r_market + ε`."
        ),
        code(
            "d = stats.beta_decomposition(decs['SPY'], leg='overnight')\n"
            "print('SPY — overnight leg regressed on the market (close-close):')\n"
            "print(f\"  beta = {d['beta']:.2f}   R^2 = {d['r_squared']:.2f}\")\n"
            "print(f\"  mean overnight return = {d['mean_leg_bps']:.2f} bps/day\")\n"
            "print(f\"     of which beta*market = {d['beta_contrib_bps']:.2f} bps   (risk premium)\")\n"
            "print(f\"     of which resid. alpha = {d['alpha_daily_bps']:.2f} bps   ({d['alpha_ann_pct']:+.1f}%/yr)\")"
        ),
        md(
            "Two readings, both unfavourable to the easy-edge thesis:\n\n"
            "- **~40% of the overnight return is beta** (beta ≈ 0.33 on SPY): holding the "
            "market overnight is partly a plain **gap-risk premium**, not distinct alpha.\n"
            "- More importantly, the **residual alpha (~1.9 bps/day) is *below* the "
            "break-even cost (~3.25 bps)** computed in §4: even the \"non-beta\" part of "
            "the return does not survive execution fees. That's the nail in the coffin — "
            "you must charge costs against the alpha, not the gross return."
        ),
        md(
            "### 3.3 Sensitivity to the dividend-adjustment mode\n\n"
            "The adjustment mode is **not a detail**: a stock goes ex-dividend at the "
            "**open**, so adjustment shifts return between night and day. Compare "
            "`split_only` (default, keeps the ex-div gap in the night) vs `total_return` "
            "(fully adjusted) on SPY:"
        ),
        code(
            "spy_tr = decompose.decompose(data.fetch('SPY', mode='total_return'))\n"
            "spy_so = decs['SPY']\n"
            "print('SPY — overnight cumulative:')\n"
            "print(f\"  split_only   : {spy_so['cum_overnight'].iloc[-1]*100:+,.0f}%\")\n"
            "print(f\"  total_return : {spy_tr['cum_overnight'].iloc[-1]*100:+,.0f}%\")\n"
            "print('The choice shifts the level of the night leg -> document it in any published figure.')"
        ),
        md(
            "## 4. Execution realism\n\n"
            "Cost model per night held:\n"
            "`cost = 2*(0.5*spread + commission + slippage) + financing`. "
            "The **factor 2** (you cross the spread to buy *and* to sell, ~252×/yr) is the "
            "killer. Break-even and sweep on SPY:"
        ),
        code(
            "be = backtest.breakeven_cost_bps(spy_so)\n"
            "print(f'Break-even round-trip cost = {be:.2f} bps/night')\n"
            "sweep = backtest.cost_sweep(spy_so, roundtrip_bps=(0,1,2,3,5,8))\n"
            "sweep.round(3)"
        ),
        md(
            "The gross edge survives only a few basis points. Three retail-side aggravating "
            "factors, not captured by this *optimistic* backtest:\n\n"
            "- **Execution price ≠ academic prints**: the anomaly is measured on the "
            "close/open auctions, inaccessible to retail; at T±5 min you're in continuous "
            "trading, with a wider spread.\n"
            "- **CFD / MT5 swap**: the overnight financing charged each night can erase the "
            "edge on its own (check `swap_long` before any trade — exactly the safeguard "
            "implemented in the repo's MT5 connector).\n"
            "- **Capacity / slippage**: market impact grows with order size, especially on "
            "thin auctions."
        ),
        md(
            "## 5. Reproducibility\n\n"
            "- **Determinism**: all synthetic data and the bootstrap are *seeded*.\n"
            "- **Tests**: `pytest` verifies the decomposition identity "
            "`(1+r_night)(1+r_day)=(1+r_cc)` (error ~1e-16) and cost monotonicity.\n"
            "- **CI**: GitHub Actions reruns tests + the offline demo on Python 3.10–3.12.\n"
            "- **Data**: local parquet cache (`_cache/`), explicit adjustment mode.\n"
            "- **Generated notebooks**: `python notebooks/build_notebooks.py` then "
            "`nbconvert --execute` — the figure you read is the executed output, not a screenshot."
        ),
        md(
            "## 6. Honest verdict\n\n"
            "Three levels never to be conflated:\n\n"
            "1. **The empirical fact is REAL** and well documented (Cooper, Cliff, and "
            "Gulen 2008; Berkman et al. 2012; Lou, Polk, and Skouras 2019; Boyarchenko, "
            "Larsen, and Whelan 2023). Credit to Knuteson for publishing data and code.\n"
            "2. **The magnitudes are INFLATED** — 30-year compounding + log scale, data "
            "artefacts, selection/survivorship bias.\n"
            "3. **The attribution to orchestrated fraud is NOT proven** — the foreign-ETF "
            "inversion (time zone) and the Chinese case (T+1; Qiao and Dam 2020) favour "
            "**microstructure** explanations, and most of the \"overnight return\" is "
            "**gap beta**, not alpha.\n\n"
            "And even assuming the edge is real: **it does not survive real execution "
            "costs** — as the 2023 liquidation of the NSPY / NIWM ETFs showed. A beautiful "
            "anomaly for understanding microstructure; a poor retail trading strategy."
        ),
        md(
            "## References\n\n"
            "Author–date (Chicago / *JFE*). Full list, a literature map and BibTeX in the "
            "repo: [`docs/references.md`](../docs/references.md), "
            "[`references.bib`](../references.bib).\n\n"
            "- Berkman, H., P. D. Koch, L. Tuttle, and Y. J. Zhang. 2012. \"Paying Attention: "
            "Overnight Returns and the Hidden Cost of Buying at the Open.\" *JFQA* 47 (4): 715–741.\n"
            "- Boyarchenko, N., L. C. Larsen, and P. Whelan. 2023. \"The Overnight Drift.\" "
            "*Review of Financial Studies* 36 (9): 3502–3547.\n"
            "- Cooper, M. J., M. T. Cliff, and H. Gulen. 2008. \"Return Differences between "
            "Trading and Non-Trading Hours: Like Night and Day.\" Working paper, SSRN 1004081.\n"
            "- Knuteson, B. 2019, 2020, 2022, 2023. Overnight/intraday return series "
            "(arXiv:1912.01708, 2010.01727, 2201.00223; SSRN 4619084).\n"
            "- Lou, D., C. Polk, and S. Skouras. 2019. \"A Tug of War: Overnight Versus "
            "Intraday Expected Returns.\" *JFE* 134 (1): 192–213.\n"
            "- Qiao, K., and L. Dam. 2020. \"The Overnight Return Puzzle and the 'T+1' "
            "Trading Rule in Chinese Stock Markets.\" *Journal of Financial Markets* 50: 100534."
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
