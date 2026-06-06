"""Generate the two narrative notebooks from source, then they are executed.

Why a builder script instead of hand-edited .ipynb? Reproducibility: the
notebooks are a *generated artefact*. Edit the cell text here, re-run
``python notebooks/build_notebooks.py`` to rebuild the skeletons, then execute
with nbconvert to embed the figures/outputs:

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Two audiences, two files:
  01_for_the_curious  — plain-language story + why it matters (the stakes)
  02_for_the_quants   — real data, critique, statistics, microstructure, capacity
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

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
            "### A real market anomaly, what's at stake, and why it's subtler than it looks\n\n"
            "Over the last 30 years, almost all the gains of the world's big stock "
            "markets piled up **while the market was closed** — from one day's close to "
            "the next morning's open. The **daytime session** (open → close) is nearly "
            "flat. Someone even claims this is the fingerprint of a giant fraud.\n\n"
            "This notebook tells the story **without jargon**, explains **why it matters**, "
            "and shows why the truth is more interesting than the headline. For the "
            "rigorous version (statistics, microstructure, capacity), see "
            "[`02_for_the_quants.ipynb`](02_for_the_quants.ipynb).\n\n"
            "> ⚠️ **This is not investment advice.** Educational and research tool."
        ),
        code(BOOT),
        md(
            "## Why should you care? (the stakes)\n\n"
            "This isn't a trivia question. Three very different things hang on the answer:\n\n"
            "- **A fraud accusation.** One researcher (Bruce Knuteson) argues the pattern is "
            "the signature of a single huge hedge fund secretly manipulating the world's "
            "markets — a *multi-trillion-dollar* claim against the financial system. If "
            "true, it's one of the biggest scandals in finance. If not, it's a careless "
            "accusation. **Which it is depends entirely on the numbers.**\n"
            "- **Your savings.** If \"stocks only go up at night\" were a free lunch, you "
            "could just buy at the close and sell at the open. Funds tried exactly that "
            "(the *NSPY* and *NIWM* ETFs). They launched in 2022 and were **shut down in "
            "2023** after losing to the market. Understanding *why* protects you from the "
            "next shiny promise.\n"
            "- **How markets actually work.** The pattern is real and well documented. The "
            "interesting question is *what causes it* — and the honest answer teaches you "
            "more about markets than any get-rich scheme.\n\n"
            "Our job here: separate **what's true** from **what's exaggerated** from "
            "**what's unproven** — with code you can run yourself."
        ),
        md(
            "## Why would a market move at night *at all*?\n\n"
            "Before crying fraud, notice there are perfectly ordinary reasons the "
            "*closed* hours could carry most of the return:\n\n"
            "1. **News doesn't wait for the bell.** Companies report earnings *after* the "
            "close; economic data and overseas events land overnight. Prices jump at the "
            "next open to absorb it.\n"
            "2. **The world keeps trading while you sleep.** When New York is closed, "
            "Tokyo and London are open. A US-listed fund on foreign stocks will do most of "
            "its *real* moving during America's night.\n"
            "3. **You get paid to hold risk you can't escape.** Holding stocks overnight "
            "means bearing gap risk while you *cannot* trade. Markets tend to pay a premium "
            "for bearing risk — so some overnight return is just **compensation for risk**, "
            "not magic.\n"
            "4. **The night is simply longer.** The closed window (evening + the whole next "
            "morning, plus entire weekends) spans far more calendar time than the 6.5-hour "
            "trading day. More hours → more drift. *(The quant notebook makes this precise — "
            "and it dissolves much of the mystery.)*\n\n"
            "Keep these in mind: each is a boring, fraud-free reason the night can win."
        ),
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
            "**We just reproduced the scary pattern with zero manipulation.** A tiny "
            "constant bias, repeated ~250 nights a year for decades, is enough. The night "
            "*looks* magical. Now three traps that inflate the story — then the punchline."
        ),
        md(
            "## 2. Trap #1 — the scale lies (the magic of compounding)\n\n"
            "The headline charts use a **logarithmic** scale, where you quickly read "
            "\"billions of %\". Where does that dizzying number come from? Not fraud: "
            "**compounding**. Look at what a simple constant bias becomes, by size and horizon:"
        ),
        code(
            "table = diagnostics.compounding_table()\n"
            "diagnostics.format_compounding(table)"
        ),
        md(
            "A bias of **1 basis point per night** — totally innocent, undetectable — "
            "compounds to three digits over 30 years. At 30 bps, you reach *trillions* of "
            "percent. **The explosion comes from the exponent, not from a conspiracy.**"
        ),
        md(
            "## 3. Trap #2 — dirty data manufactures the signal\n\n"
            "Free data (Yahoo) mis-handles some *splits* and dividends, especially in "
            "emerging markets. A handful of corrupted prices is enough to **mechanically "
            "shift return from the day into the night**. On a perfectly flat market we "
            "dirty just 3 prices:"
        ),
        code(
            "flat = diagnostics.synthetic_ohlc(overnight_bias_bps=0, intraday_bias_bps=0, seed=1)\n"
            "clean = decompose.decompose(flat)\n"
            "dirty = decompose.decompose(diagnostics.inject_split_artifact(flat, factor=1.5))\n"
            "print(f\"Overnight cumulative  BEFORE: {clean['cum_overnight'].iloc[-1]*100:+.1f}%\")\n"
            "print(f\"Overnight cumulative  AFTER : {dirty['cum_overnight'].iloc[-1]*100:+.1f}%   (3 dirtied prices)\")\n"
            "flags = diagnostics.flag_suspicious_returns(dirty)\n"
            "print(f\"\\nThe automatic detector flags {len(flags)} suspicious day(s).\")"
        ),
        md(
            "Three data errors, and the \"overnight performance\" flips from red to bright "
            "green. This is the mechanism behind the wildest emerging-market numbers. "
            "**Before crying scandal, check your data.**"
        ),
        md(
            "## 4. Trap #3 — fees erase the gain\n\n"
            "Suppose the night effect is real (it partly is). Can you *trade* it? Buying "
            "every close and selling every open pays the spread **twice a day, ~250 days a "
            "year**. What's left after realistic costs?"
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
            "round-trip, the gain turns **negative**. This is exactly what sank the NSPY / "
            "NIWM \"night effect\" ETFs: launched June 2022, **liquidated August 2023**.\n\n"
            "> *A strategy that's beautiful on paper is worth no more than the paper until "
            "it has paid the real costs of execution.*"
        ),
        md(
            "## So… fraud, or not?\n\n"
            "Here's the honest verdict, and why it's more interesting than the headline:\n\n"
            "| | |\n|---|---|\n"
            "| ✅ **The fact is real** | the night really did outperform the day, for decades |\n"
            "| ⚠️ **But the numbers are inflated** | compounding + log scale + dirty data |\n"
            "| 🕐 **And partly an illusion** | the night is simply *longer* — see the quant notebook |\n"
            "| ❌ **And not tradable** | fees erase the edge (RIP, night ETFs) |\n"
            "| 🔍 **Fraud? Not proven** | ordinary risk + market plumbing explain it just as well |\n\n"
            "The pattern is genuine and fascinating — but \"the market is rigged at night\" "
            "is a much bigger claim than the evidence supports. The quant notebook shows, "
            "with proper statistics, *why a careful researcher stays sceptical*: "
            "[`02_for_the_quants.ipynb`](02_for_the_quants.ipynb).\n\n"
            "*Further reading:* the original articles and the academic literature (with a "
            "map of who argues what) are in [`docs/references.md`](../docs/references.md); "
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
            "# The overnight anomaly — a quantitative teardown\n"
            "### Real data · robust inference · microstructure · capacity · falsifiability\n\n"
            "The rigorous companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "We take the overnight-return pattern seriously *as a fact* and then ask the "
            "questions that decide what it **means**:\n\n"
            "1. Is it statistically real once returns aren't assumed i.i.d.? **(yes — and we say so)**\n"
            "2. Are we comparing like with like? **(the clock illusion)**\n"
            "3. Is it alpha or just beta you carry overnight? **(mostly beta)**\n"
            "4. Has it decayed since publication? **(sharply)**\n"
            "5. Could anyone actually harvest it at scale? **(no — capacity is tiny)**\n"
            "6. Does any of this prove *manipulation*? **(no — and here's the likelihood argument)**\n\n"
            "> ⚠️ **Not investment advice.** Data: Yahoo! Finance via `yfinance`, mode "
            "`split_only`. First run hits the network. Methods: Newey & West (1987), "
            "Lo (2002), McLean & Pontiff (2016), Almgren et al. (2005) — see "
            "[`docs/references.md`](../docs/references.md)."
        ),
        code(
            BOOT + "\nfrom overnight import data, decompose, diagnostics, backtest, stats, analytics\n"
        ),
        md(
            "## 1. The pattern on 10 world indices (ETFs)\n\n"
            "Decompose each ETF into night / day and read the **annualised Sharpe of the "
            "overnight leg** — the only number that matters for an *edge*."
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
            "    }\n"
            "table = pd.DataFrame(rows).T\n"
            "table"
        ),
        md(
            "The US (SPY, QQQ) and Brazil show the classic shape (night Sharpe ~0.7); "
            "Europe and Japan are **inverted** (§4.1). First, is any of this real?"
        ),
        md(
            "## 2. Is it real? Honest, autocorrelation-robust inference\n\n"
            "A pretty cumulative chart proves nothing. Daily returns are autocorrelated and "
            "heteroskedastic, so we use **Newey-West (HAC)** t-stats on the mean and the "
            "**Lo (2002)** standard error on the Sharpe. We *want* to give the anomaly its "
            "fairest hearing."
        ),
        code(
            "print('Overnight leg — HAC (Newey-West) and Lo (2002) inference:\\n')\n"
            "for tk in ['SPY','QQQ','EWZ','FXI']:\n"
            "    if tk not in decs: continue\n"
            "    h  = analytics.mean_tstat_hac(decs[tk]['r_overnight'])\n"
            "    hi = analytics.mean_tstat_hac(decs[tk]['r_intraday'])\n"
            "    s  = analytics.sharpe_with_se(decs[tk]['r_overnight'])\n"
            "    print(f\"{tk:4s}  overnight {h['mean_bps']:+.2f} bps/day  HAC t={h['tstat']:+.2f}   \"\n"
            "          f\"Sharpe {s['sharpe_ann']:+.2f} (t={s['tstat']:+.2f})    [intraday HAC t={hi['tstat']:+.2f}]\")"
        ),
        md(
            "For SPY the overnight mean carries a **HAC t ≈ 5** — unambiguously real — while "
            "the *intraday* mean is insignificant (t ≈ 1). So we concede the central "
            "empirical fact without flinching. The fight is **not** \"is it real?\" but "
            "\"what is it?\". The rest of this notebook is that fight."
        ),
        md(
            "## 3. The clock illusion — are we even comparing like with like?\n\n"
            "Here is the single most under-appreciated point in the whole debate. The "
            "overnight window is **not** 6.5 hours like the trading day — it runs from the "
            "16:00 close to the 09:30 open (**17.5h on weeknights**) and across **whole "
            "weekends and holidays (65h+)**. If return accrues with calendar time, the "
            "night *should* dominate simply because it is longer. Let's normalise."
        ),
        code(
            "tn = analytics.time_normalized_summary(decs['SPY'])\n"
            "display(tn.round(3))\n"
            "rs = tn.loc['overnight','mean_bps_per_session'] / tn.loc['intraday','mean_bps_per_session']\n"
            "rh = tn.loc['overnight','mean_bps_per_hour']    / tn.loc['intraday','mean_bps_per_hour']\n"
            "print(f'SPY  overnight-vs-intraday advantage   per SESSION: {rs:.1f}x   per CALENDAR HOUR: {rh:.1f}x')"
        ),
        md(
            "The average SPY overnight window is **~28 hours** (weekends drag the mean up), "
            "versus 6.5 intraday. Per *session* the night out-earns the day by ~4×; per "
            "*calendar hour* that collapses to ~1.3×. **Most of the famous gap is just the "
            "night being longer.** A residual per-hour premium survives — consistent with a "
            "genuine but modest overnight risk premium — but the \"day is dead, night is "
            "magic\" framing is largely a unit error. Any serious treatment must put both "
            "legs on the same clock; the headline rarely does."
        ),
        md(
            "## 4. Critical teardown of the cross-section\n\n"
            "### 4.1 The foreign-ETF inversion: we're measuring the *time zone*\n"
        ),
        code(
            "t = table.sort_values('Sharpe night')\n"
            "x = np.arange(len(t)); w = 0.4\n"
            "fig, ax = plt.subplots(figsize=(11,5))\n"
            "ax.bar(x-w/2, t['Sharpe night'].astype(float), w, label='Sharpe night', color='#2c7fb8')\n"
            "ax.bar(x+w/2, t['Sharpe day'].astype(float),   w, label='Sharpe day',   color='#fdae61')\n"
            "ax.axhline(0, color='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels(t.index)\n"
            "ax.set_ylabel('Annualised Sharpe'); ax.set_title('Night vs day by market — the sign flips for foreign ETFs')\n"
            "ax.legend(); ax.grid(axis='y', alpha=.3); plt.show()"
        ),
        md(
            "EWU/EWG/EWQ/EWJ are US-listed ETFs whose underlying (London, Frankfurt, Paris, "
            "Tokyo) **trades during the US night**. For them the \"overnight\" window "
            "*contains the home session*, and the US \"intraday\" window is when the "
            "underlying is shut. The night/day split is **relative to the listing clock**, "
            "not a universal anomaly. A single global manipulator would struggle to explain "
            "why the sign depends on *where the ETF is listed* — but a time-zone account "
            "explains it instantly."
        ),
        md(
            "### 4.2 The China test (T+1) and multiple testing\n\n"
            "China's \"T+1\" rule (shares bought today can't be sold until tomorrow) "
            "predicts an *inverted* pattern (Qiao and Dam 2020) — a natural experiment that "
            "again favours microstructure over manipulation. And with a cross-section of "
            "markets, remember **multiple testing**: pick \"the 25 most problematic\" from a "
            "large universe and significance is guaranteed by selection alone."
        ),
        code(
            "ts = {tk: analytics.sharpe_with_se(d['r_overnight'])['tstat'] for tk, d in decs.items()}\n"
            "n_sig = sum(abs(v) > 1.96 for v in ts.values())\n"
            "print(f'{n_sig}/{len(ts)} markets: |t| > 1.96 on the overnight Sharpe.')\n"
            "print(f'Under H0 (no effect), expected false positives at 5%: {0.05*len(ts):.1f}.')\n"
            "print('Overnight Sharpe t-stats, sorted:')\n"
            "for k, v in sorted(ts.items(), key=lambda kv: -kv[1]):\n"
            "    print(f'   {k:4s} {v:+.1f}')"
        ),
        md(
            "## 5. Alpha, or beta you carry in the dark?\n\n"
            "Holding the market *every night* means bearing **gap risk** permanently — so "
            "part of the \"overnight alpha\" is a beta risk premium. Regress the night leg "
            "on the market (`r_night = α + β·r_market + ε`) and split the mean."
        ),
        code(
            "d = stats.beta_decomposition(decs['SPY'], leg='overnight')\n"
            "be = backtest.breakeven_cost_bps(decs['SPY'])\n"
            "print('SPY overnight leg vs the market (close-close):')\n"
            "print(f\"  beta = {d['beta']:.2f}   R^2 = {d['r_squared']:.2f}\")\n"
            "print(f\"  mean overnight        = {d['mean_leg_bps']:.2f} bps/day\")\n"
            "print(f\"    of which beta*market = {d['beta_contrib_bps']:.2f} bps  (gap risk premium)\")\n"
            "print(f\"    of which alpha       = {d['alpha_daily_bps']:.2f} bps  ({d['alpha_ann_pct']:+.1f}%/yr)\")\n"
            "print(f\"  break-even round-trip cost = {be:.2f} bps  ->  residual alpha {'survives' if d['alpha_daily_bps']>be else 'is BELOW cost'}\")"
        ),
        md(
            "~40% of the overnight return is plain market beta carried overnight. Worse, the "
            "**residual alpha (~1.9 bps) sits below the break-even trading cost (~3.3 bps)**: "
            "even the non-beta part doesn't survive execution. You must charge costs against "
            "the *alpha*, not the gross return — and there's nothing left."
        ),
        md(
            "## 6. Has the edge decayed? (post-publication alpha decay)\n\n"
            "Documented anomalies tend to weaken once the world reads about them "
            "(McLean and Pontiff 2016). The overnight effect has been in the literature "
            "since Cooper, Cliff, and Gulen (2008). Trailing 5-year overnight Sharpe:"
        ),
        code(
            "rsh = analytics.rolling_sharpe(decs['SPY']['r_overnight'])\n"
            "ax = plt.subplot()\n"
            "ax.plot(rsh.index, rsh.values, lw=1.6, color='#2c7fb8')\n"
            "ax.axhline(0, color='k', lw=.6); ax.axhline(0.5, color='grey', ls='--', lw=.8)\n"
            "ax.set_title('SPY overnight — trailing 5-year Sharpe (it fades)')\n"
            "ax.set_ylabel('Annualised Sharpe (5y rolling)'); ax.grid(alpha=.3); plt.show()\n"
            "print(f'{rsh.index[0].date()}: {rsh.iloc[0]:.2f}   ->   {rsh.index[-1].date()}: {rsh.iloc[-1]:.2f}   '\n"
            "      f'(min {rsh.min():.2f}, max {rsh.max():.2f})')"
        ),
        md(
            "From a Sharpe near **2 in the late 1990s** to about **0.5 today** — a textbook "
            "decay profile. An edge that is fading as it becomes known is the opposite of "
            "what you'd expect from an *ongoing, deliberate* manipulation scheme."
        ),
        md(
            "## 7. Capacity: could anyone actually harvest it?\n\n"
            "Even a real, free edge is worthless if it can't hold money. Using the "
            "square-root market-impact law (Almgren et al. 2005), find the AUM at which "
            "round-trip impact equals the gross overnight edge."
        ),
        code(
            "spy_ohlc = data.fetch('SPY', mode='split_only')\n"
            "cap = analytics.capacity_estimate(decs['SPY'], spy_ohlc)\n"
            "print(f\"Gross edge          : {cap['edge_bps']:.2f} bps/night\")\n"
            "print(f\"Daily volatility    : {cap['daily_vol_bps']:.0f} bps\")\n"
            "print(f\"ADV                 : {cap['adv_shares']:,.0f} shares  @ ${cap['price']:.0f}\")\n"
            "print(f\"Participation cap   : {cap['participation_rate']*1e4:.2f} bps of ADV\")\n"
            "print(f\"Capacity (coef=1)   : ${cap['capacity_usd']/1e6:,.1f}M before impact eats the edge\")"
        ),
        md(
            "Order-of-magnitude (the impact coefficient is venue-specific), but the message "
            "is robust: the strategy holds only **single-digit millions** of SPY before its "
            "own trading impact erases a 3-bps edge. This cuts **both** ways against the "
            "headline. Retail can't scale it — and, more tellingly, **a fund large enough "
            "to *move world markets* could not quietly profit from an edge this thin**: the "
            "impact of trading at that size would dwarf the premium. The alleged mechanism "
            "is self-defeating at the scale required to matter."
        ),
        md(
            "## 8. Does any of this prove manipulation? A likelihood argument\n\n"
            "Knuteson's thesis: a large quant firm expands its book when the market is "
            "illiquid (near the open, moving prices up) and contracts it when liquid. The "
            "pattern is certainly *consistent* with that. But consistency is not evidence. "
            "Frame it as a likelihood ratio for the observed pattern *D*:\n\n"
            "$$\\Lambda = \\frac{P(D \\mid \\text{manipulation})}{P(D \\mid \\text{risk premium} + \\text{microstructure})}$$\n\n"
            "- The basic night>day pattern is **highly probable under *both*** hypotheses "
            "(overnight risk premium, clientele effects, news timing, and the clock "
            "illusion all predict it). When *D* is roughly as likely under each, "
            "$\\Lambda \\approx 1$ — the headline pattern **discriminates almost nothing**.\n"
            "- The **discriminating** evidence points the other way: the foreign-ETF "
            "inversion (listing clock) and the Chinese T+1 inversion are *predicted* by "
            "microstructure and *awkward* for one global manipulator → $\\Lambda < 1$.\n"
            "- The **capacity** result makes the manipulation mechanism economically "
            "self-defeating at world scale.\n\n"
            "What *would* move $\\Lambda$ decisively: participant-level order data showing "
            "systematic open-auction imbalances from one book, characteristic intraday "
            "reversals, or a cross-sectional signature tied to a specific firm. Knuteson "
            "doesn't have it; neither do we. Absent discriminating data, **parsimony "
            "favours risk premium + microstructure over a secret world-spanning fraud.**"
        ),
        md(
            "## 9. Reproducibility\n\n"
            "- **Determinism**: synthetic data, bootstrap and all seeds are fixed.\n"
            "- **Tests**: `pytest` checks the decomposition identity "
            "`(1+r_night)(1+r_day)=(1+r_cc)` (~1e-16), the cost model, the stats and the "
            "analytics (HAC, Lo SE, calendar-time, capacity).\n"
            "- **CI**: GitHub Actions reruns everything on Python 3.10–3.12.\n"
            "- **Generated notebooks**: `python notebooks/build_notebooks.py` then "
            "`nbconvert --execute` — every figure here is an executed output, not a screenshot."
        ),
        md(
            "## 10. Verdict\n\n"
            "A disciplined reading of the same data the pamphlet uses:\n\n"
            "1. **The fact is REAL.** Overnight returns dominate intraday with HAC "
            "t-stats around 5. Full credit to Knuteson for publishing data and code.\n"
            "2. **The magnitude is oversold.** Log-scale compounding, data artefacts, "
            "selection — and above all the **clock illusion**: per calendar hour the night's "
            "edge is ~1.3×, not 4×.\n"
            "3. **What's left is mostly beta and is fading.** ~40% of the overnight return "
            "is gap-risk premium; the residual alpha is below trading costs; and the "
            "5-year Sharpe has decayed from ~2 to ~0.5.\n"
            "4. **It isn't tradable and isn't scalable.** Costs flip it negative; capacity "
            "is single-digit millions.\n"
            "5. **Manipulation is unproven.** The pattern is weak evidence; the "
            "discriminating evidence (inversions, capacity) favours risk premium + "
            "microstructure.\n\n"
            "A beautiful, real, and *largely explained* feature of market microstructure — "
            "wrongly dressed up as both a free lunch and a global fraud. The honest, "
            "fully-reproducible answer is more interesting than either."
        ),
        md(
            "## References\n\n"
            "Author–date (Chicago / *JFE*). Full list, literature map and BibTeX: "
            "[`docs/references.md`](../docs/references.md), [`references.bib`](../references.bib).\n\n"
            "Almgren, R., C. Thum, E. Hauptmann, and H. Li. 2005. \"Direct Estimation of "
            "Equity Market Impact.\" *Risk* 18 (7). · "
            "Berkman, H., P. D. Koch, L. Tuttle, and Y. J. Zhang. 2012. *JFQA* 47 (4): 715–741. · "
            "Boyarchenko, N., L. C. Larsen, and P. Whelan. 2023. \"The Overnight Drift.\" *RFS* 36 (9): 3502–3547. · "
            "Cooper, M. J., M. T. Cliff, and H. Gulen. 2008. \"Like Night and Day.\" Working paper. · "
            "Knuteson, B. 2019–2023. arXiv:1912.01708, 2010.01727, 2201.00223; SSRN 4619084. · "
            "Lo, A. W. 2002. \"The Statistics of Sharpe Ratios.\" *FAJ* 58 (4): 36–52. · "
            "Lou, D., C. Polk, and S. Skouras. 2019. *JFE* 134 (1): 192–213. · "
            "McLean, R. D., and J. Pontiff. 2016. *Journal of Finance* 71 (1): 5–32. · "
            "Newey, W. K., and K. D. West. 1987. *Econometrica* 55 (3): 703–708. · "
            "Qiao, K., and L. Dam. 2020. *Journal of Financial Markets* 50: 100534."
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
