"""Generate the two narrative notebooks for Study 334 (ARK-Innovation).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells try the cached daily
parquet under ../_cache/ and otherwise fall back to the synthetic tape with a clear
banner — so the notebook re-runs for any reader, online or off. The frozen real-tape
headline numbers live in one dict, ``R`` (mirroring docs/results.md, as-of 2026-06-18).
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-18).
R = dict(
    asof="2026-06-18",
    last_complete="2026-05-29",
    n_arkk=2910,
    fp_arkk="a43361b569ef", fp_qqq="8da71ba017ad", fp_xlk="4a6a71a151dd",
    # ARKK buy-and-hold arc
    cagr_full=13.8, total_full=346.0,
    cagr_boom=40.3, total_boom=737.0, peak_date="2021-02-12", peak_px=154.0, incep_px=18.4,
    cagr_bust=-11.3, total_bust=-47.0, dd=-81.0, trough_date="2022-12-28", trough_px=29.0,
    # overlay race (gross, real ARKK)
    trend_mean=6.45, trend_sharpe=0.60, trend_t=2.09, trend_inv=58,
    tsmom_mean=4.16, tsmom_sharpe=0.39, tsmom_t=1.36,
    mr_mean=0.74, mr_sharpe=0.09, mr_t=0.33,
    bh_sharpe=0.53, bh_t=1.88,
    trend_ci_lo=0.22, trend_ci_hi=12.3,
    excess_sharpe_diff=0.07, qqq_sharpe=0.93, qqq_cagr=19.7,
    # cost sweep
    cost0=6.45, cost0_t=2.09, cost20=6.33, cost20_t=2.05,
    # behaviour gap (literature, real)
    wealth_destroyed_lo=7.0, wealth_destroyed_hi=10.0,
    # synthetic behaviour-gap control
    null_twr=11.6, null_dwr=15.7, null_gap=-4.1,
    bb_twr=1.9, bb_dwr=-4.8, bb_gap=6.6,
    # synthetic momentum control
    mom_null_t=-0.15, mom_planted_t=5.24,
)


# ---------------------------------------------------------------------------
# Shared analysis preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from ark_innovation import data, strategy as st

TICKERS = ["ARKK", "QQQ", "XLK"]
CUT = pd.Timestamp("2026-06-01")  # exclusive — drop the partial month

def _have_cache():
    return all(os.path.exists(data._cache_path(t, data.DEFAULT_CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def real_close(t):
    px = data.fetch_prices(t, fetch=False)
    return px["close"][px.index < CUT]

print("real daily cache present:", HAVE_REAL)
"""

# Synthetic fallback builder — a price tape standing in for ARKK when offline.
FALLBACK = """\
# Offline fallback: a deterministic boom-bust tape stands in for the real ARKK series.
# (Clearly bannered as synthetic; real headline numbers are quoted from docs/results.md.)
def synthetic_arkk():
    fr, _ = data.synthetic_hype_cycle(n_months=180, bust=0.04, chase=15.0, seed=334)
    # upsample the monthly NAV to a daily-ish business index for the overlay charts
    daily = data.synthetic_daily(n_days=2910, momentum=0.05, drift=0.0003, seed=334)
    return daily["close"]
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# ARK-Innovation — was chasing Cathie Wood a momentum win, or a buy-the-top machine? 🚀\n"
            "### The most-hyped ETF of 2021, and the gap between what it *returned* and what its investors *earned*\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![A_buy--the--top_machine%3F: Confirmed](https://img.shields.io/badge/A_buy--the--top_machine%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "In 2020 ARKK rose about **150%**, Cathie Wood was on every screen, and money "
            "poured in. Then it fell **−81%**. Here's the twist this notebook is about: over its "
            "*whole* life ARKK's price actually did fine — up **+13.8%/yr** from inception. So how "
            "did so many people lose money on a fund that went *up*? Because almost nobody held it "
            "from the start — the cash arrived near the top. This notebook tells that story in "
            "plain English, and shows the one trick that separates a fund's advertised return from "
            "the return its investors actually got.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the dollar-weighted IRR "
            "machinery and the capacity math? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. The real-tape numbers are pinned in "
            "[docs/results.md](../docs/results.md) (as-of " + R["asof"] + "). House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + FALLBACK),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first 🎯\n\n"
            "| Question | The honest answer |\n"
            "|---|---|\n"
            "| Was it a **momentum** win you could trade? | **Barely, and not really.** Only a slow "
            "trend rule beat the noise — and only by hiding in cash through the crash. |\n"
            "| Could you have **traded** that edge profitably? | **No.** A plain Nasdaq-100 (QQQ) "
            "holder both avoided ARKK's crash *and* compounded faster. |\n"
            "| Was it a **buy-the-top machine**? | **Yes.** The fund went up over its life, but the "
            "average dollar that bought it lost money — it arrived at the peak. |\n\n"
            f"The fund peaked **{R['peak_date']}** after a **+{R['total_boom']:.0f}%** run, then fell "
            f"**{R['dd']:.0f}%**. That shape is the whole story."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "*Innovation is the future, and a star stock-picker riding it will crush the index. ARKK "
            "is how you buy that future — and it's already proven it, up 150% in a year.* That was "
            "the pitch in late 2020, and millions of investors bought it. The implicit promise: "
            "**momentum** — what's been going up will keep going up, and a concentrated bet on the "
            "winners beats a boring index fund."
        ),

        # ---- BEAT 2 — SO WHAT? -----------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the pitch were right, ARKK buyers would have beaten the Nasdaq. If it were *wrong* "
            "in the specific way hype usually goes wrong, two things would be true at once: the fund "
            "could still post a decent long-run number **and** its investors could collectively lose "
            "money — because they bought late. Telling those two apart is the difference between "
            "*'innovation investing works'* and *'innovation marketing works'*. Real money rides on "
            "which one it is."
        ),
        code(
            "# The arc: ARKK from inception, with the 2021 peak marked.\n"
            "if HAVE_REAL:\n"
            "    c = real_close('ARKK'); tape = 'REAL ARKK (cache)'\n"
            "else:\n"
            "    c = synthetic_arkk(); tape = 'SYNTHETIC fallback (offline)'\n"
            "peak_date = c.idxmax()\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(c.index, c.values, color=GREEN, lw=1.4)\n"
            "ax.axvline(peak_date, color=RED, ls='--', lw=1, label=f'peak {peak_date.date()}')\n"
            "ax.set_title(f'ARKK price arc — {tape}'); ax.set_ylabel('price (total-return proxy)')\n"
            "ax.legend(); plt.show()\n"
            "print('banner:', tape)\n"
            f"print('Pinned real arc: +{R['total_boom']:.0f}% to the {R['peak_date']} peak, then {R['dd']:.0f}%, ending +{R['cagr_full']:.1f}%/yr')"
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two clean tests, decided in advance:\n\n"
            "1. **Was it a momentum win you could trade?** Turn ARKK's price into simple long/flat "
            "rules — ride the trend, ride short-term momentum, or buy the dips — and see if any of "
            "them reliably beat just holding the fund (and, more importantly, holding QQQ). If none "
            "do, there was no tradable timing edge.\n"
            "2. **Did its investors make money?** Compare the fund's **buy-and-hold return** (what one "
            "dollar held the whole time earned) with its **dollar-weighted return** (what the average "
            "dollar earned, accounting for *when* the money actually arrived). If the second is far "
            "below the first, it was a buy-the-top machine.\n\n"
            "**What would make us say 'no buy-the-top problem':** the two returns come out about "
            "equal. We'll show on a synthetic fund exactly what each outcome looks like."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "### 4a · The momentum rules barely work\n\n"
            "Three simple long/flat overlays on the real ARKK tape. Only the **slow trend rule** "
            "(50-day average above the 200-day) beats the noise — and it does it by sitting in cash "
            "through the 2021-22 crash, not by clever timing. Short-term momentum and buy-the-dip "
            "are both indistinguishable from random."
        ),
        code(
            "# Overlay race on the (real or synthetic) ARKK tape.\n"
            "rows = []\n"
            "for name, sig in [('trend 50/200', st.ma_crossover_signal(c, 50, 200)),\n"
            "                  ('momentum 1d', st.ts_momentum_signal(c, 1)),\n"
            "                  ('buy-the-dip', st.mean_reversion_signal(c, 5, 1.0))]:\n"
            "    d = st.overlay_returns(c, sig, cost_bps=0.0)\n"
            "    s = st.summarize_overlay(d, 'gross')\n"
            "    rows.append((name, round(s['sharpe'], 2), round(s['hac_t'], 2), f\"{s['pct_invested']*100:.0f}%\"))\n"
            "bh = c.pct_change().fillna(0.0).to_numpy()\n"
            "rows.append(('just hold ARKK', round(st.sharpe(bh), 2), round(st.hac_tstat(bh), 2), '100%'))\n"
            "tbl = pd.DataFrame(rows, columns=['rule', 'Sharpe', 'robust t', '% in market'])\n"
            "print('tape:', tape); display(tbl)\n"
            f"print('(pinned real: trend t=+{R['trend_t']}, momentum t=+{R['tsmom_t']}, dip t=+{R['mr_t']})')"
        ),
        md(
            "### 4b · The buy-the-top trap, shown on a fund we built\n\n"
            "Now the heart of it. We build two synthetic funds. **Fund A** just goes up steadily, "
            "with steady investor money. **Fund B** booms and busts (like ARKK), and its investors "
            "*chase* — they pile in after the run-up, right before the fall. Watch what happens to "
            "the gap between the fund's return and the *average investor's* return."
        ),
        code(
            "# Two funds: a steady grower vs a boom-bust with performance-chasing flows.\n"
            "labels, gaps, twrs, dwrs = [], [], [], []\n"
            "for label, bust, chase in [('A: steady grower', 0.0, 0.0),\n"
            "                           ('B: boom-bust + chasing', 0.04, 15.0)]:\n"
            "    fr, _ = data.synthetic_hype_cycle(n_months=180, bust=bust, chase=chase, seed=334)\n"
            "    g = st.behaviour_gap(fr['price'], fr['flow'], st.MONTHS_PER_YEAR)\n"
            "    labels.append(label); gaps.append(g['gap']*100)\n"
            "    twrs.append(g['twr_cagr']*100); dwrs.append(g['dwr_irr']*100)\n"
            "fig, ax = plt.subplots()\n"
            "x = np.arange(len(labels)); w = 0.35\n"
            "ax.bar(x - w/2, twrs, w, label='fund return (buy & hold)', color=GREEN)\n"
            "ax.bar(x + w/2, dwrs, w, label='what the average investor earned', color=RED)\n"
            "ax.axhline(0, color='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('% per year'); ax.set_title('The buy-the-top gap (synthetic illustration)')\n"
            "ax.legend(); plt.show()\n"
            "print('SYNTHETIC illustration of the mechanism — not market evidence.')\n"
            "for l, t, dd in zip(labels, twrs, dwrs):\n"
            "    print(f'  {l:24s} fund {t:+.1f}%/yr   investor {dd:+.1f}%/yr')"
        ),
        md(
            f"On the steady grower the average investor does *fine* — there's no gap. On the "
            f"boom-bust fund with chasing flows, the **fund returned ~+{R['bb_twr']:.0f}%/yr while the "
            f"average dollar earned ~{R['bb_dwr']:.0f}%/yr** — a **{R['bb_gap']:.0f}-point** hole. That "
            "is the buy-the-top machine, in miniature. Real ARKK is the same story at full scale: "
            f"Morningstar estimated it destroyed roughly **\\${R['wealth_destroyed_lo']:.0f}–"
            f"{R['wealth_destroyed_hi']:.0f} billion** of investor wealth even as its price rose."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Mixed.** Only the slow trend rule beat the noise, and only as crash "
            "insurance; real momentum and buy-the-dip did nothing.\n"
            "- **Tradability — Mirage.** Even that rule lost to just holding the Nasdaq. There was no "
            "money in timing ARKK.\n"
            "- **A buy-the-top machine? — Confirmed.** The fund went up; its investors, on average, "
            "did not."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT? ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            f"The trend rule trades rarely, so costs barely touch it. That's not the problem. The "
            f"problem is it never earns more than a boring QQQ index fund (Sharpe **{R['qqq_sharpe']:.2f}** "
            f"vs the overlay's **{R['trend_sharpe']:.2f}**), while making you hold a single volatile fad "
            "fund. The 'edge' is dodging a crash a diversified investor never had to take. The lesson "
            "isn't a strategy — it's *don't chase, and don't concentrate.*"
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- Plug **real ARKK share-count / AUM history** into `data.fetch_flows` (it's not on the "
            "free daily feed) to compute the *exact* dollar-weighted return instead of quoting the "
            "Morningstar estimate.\n"
            "- Repeat for the other ARK funds (ARKG, ARKW) and the broader thematic-ETF zoo — the "
            "buy-the-top pattern is a *product-design* phenomenon, not an ARKK quirk.\n"
            "- Fork the synthetic generator's `chase` and `bust` knobs to map how steep the price "
            "round-trip has to be before the behaviour gap turns a winning fund into a losing trade."
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
            "# ARK-Innovation — a quantitative teardown 🔬\n"
            "### Real ARKK tape · trend/momentum/reversion overlay race · dollar-weighted IRR · HAC inference · block-bootstrap CIs · capacity\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![A_buy--the--top_machine%3F: Confirmed](https://img.shields.io/badge/A_buy--the--top_machine%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We race three long/flat "
            "overlays on ARKK's real daily tape under autocorrelation-robust inference, then turn to "
            "the headline: the time-weighted vs dollar-weighted return gap (Dichev 2007) that "
            "decides what ARKK's investors actually earned.\n\n"
            "> ⚠️ **Not investment advice.** Data: Yahoo! daily auto-adjusted (total-return-proxy) "
            "closes for ARKK / QQQ / XLK since 2014-10-31, cache-first. Real headline numbers are "
            f"pinned in [docs/results.md](../docs/results.md) (as-of {R['asof']}); methods in "
            "[`docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition. House "
            "style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + FALLBACK),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | The decisive number |\n"
            "|---|---|---|\n"
            f"| **Signal** | Mixed | Trend 50/200 HAC *t* = **+{R['trend_t']}** (CI low **+{R['trend_ci_lo']} bps**); "
            f"ts-momentum *t* = +{R['tsmom_t']}, mean-rev *t* = +{R['mr_t']} → None |\n"
            f"| **Tradability** | Mirage | Overlay excess Sharpe over ARKK B&H **+{R['excess_sharpe_diff']}**; "
            f"vs QQQ B&H Sharpe **{R['qqq_sharpe']}** it loses outright |\n"
            f"| **Buy-the-top machine?** | Confirmed | +{R['total_boom']:.0f}% boom, {R['dd']:.0f}% bust; "
            f"dollar-weighted return deeply negative (~\\${R['wealth_destroyed_lo']:.0f}–{R['wealth_destroyed_hi']:.0f}bn destroyed) |\n\n"
            "> 💡 **In plain words.** One trend rule technically clears the bar, but only by going to "
            "cash in the crash; nothing here beats an index fund; and the average dollar in ARKK lost "
            "money even though the price went up."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Three testable hypotheses:\n\n"
            "- **H₁ (momentum).** ARKK's daily returns carry tradable persistence — a trend / "
            "time-series-momentum overlay earns a positive risk-adjusted return with HAC *t* ≥ 2.\n"
            "- **H₂ (tradable edge).** That overlay beats *both* ARKK buy-and-hold and a QQQ "
            "buy-and-hold on an excess-of-cash Sharpe basis.\n"
            "- **H₃ (investor experience).** The dollar-weighted (money-weighted IRR) return ARKK "
            "investors earned is far below its time-weighted (buy-and-hold) return — the "
            "performance-chasing penalty (Dichev 2007; Friesen-Sapp 2007)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "H₁/H₂ test whether *innovation momentum* was a real, harvestable edge or just beta you "
            "could have bought cheaper. H₃ tests whether the **fund's posted return** (the number ARK "
            "advertised) bears any resemblance to **investors' realised return** — the gap is the "
            "literature's cleanest indictment of thematic-ETF hype (Ben-David et al. 2023)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "1. **Decompose** ARKK's arc into boom (→2021 peak) and bust segments — an exact identity, "
            "no fitting.\n"
            "2. **Robust inference** — Newey-West HAC *t* on each overlay's daily return; circular "
            "block-bootstrap CI (preserves volatility clustering).\n"
            "3. **Alpha vs beta** — excess-of-cash Sharpe race vs ARKK and QQQ buy-and-hold (a "
            "long/flat overlay is part-time in cash, so we compare excess-to-excess).\n"
            "4. **Capacity / cost** — one-way × NAV cost sweep, one documented execution lag (signal "
            "at close *t* → return *t+1*).\n"
            "5. **The behaviour gap** — money-weighted IRR vs time-weighted CAGR, with the mechanism "
            "validated on a deterministic synthetic positive control."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown\n\n### 4a · The boom-bust decomposition"),
        code(
            "if HAVE_REAL:\n"
            "    c = real_close('ARKK'); tape = 'REAL ARKK (cache)'\n"
            "else:\n"
            "    c = synthetic_arkk(); tape = 'SYNTHETIC fallback (offline)'\n"
            "pk = c.idxmax()\n"
            "boom = st.time_weighted_return(c[c.index <= pk], st.TRADING_DAYS_PER_YEAR)\n"
            "bust = st.time_weighted_return(c[c.index >= pk], st.TRADING_DAYS_PER_YEAR)\n"
            "full = st.time_weighted_return(c, st.TRADING_DAYS_PER_YEAR)\n"
            "print('tape:', tape)\n"
            "print(f'  inception -> peak ({pk.date()}): {boom[\"cagr\"]*100:+.1f}%/yr  ({boom[\"total_return\"]*100:+.0f}%)')\n"
            "print(f'  peak -> end:                 {bust[\"cagr\"]*100:+.1f}%/yr  ({bust[\"total_return\"]*100:+.0f}%)')\n"
            "print(f'  full sample:                 {full[\"cagr\"]*100:+.1f}%/yr  ({full[\"total_return\"]*100:+.0f}%)')\n"
            f"print('  pinned real: boom +{R['cagr_boom']}%/yr (+{R['total_boom']:.0f}%), bust {R['cagr_bust']}%/yr ({R['total_bust']:.0f}%), full +{R['cagr_full']}%/yr')"
        ),
        md(
            "### 4b · The overlay race under HAC inference + block-bootstrap CI\n\n"
            "> 💡 **In plain words.** Does any simple timing rule on ARKK beat the noise? We charge "
            "each one a robust *t*-stat and a bootstrap confidence interval."
        ),
        code(
            "rows = []\n"
            "overlays = {'trend 50/200': st.ma_crossover_signal(c, 50, 200),\n"
            "            'ts-momentum 1d': st.ts_momentum_signal(c, 1),\n"
            "            'mean-rev z<-1': st.mean_reversion_signal(c, 5, 1.0)}\n"
            "for name, sig in overlays.items():\n"
            "    d = st.overlay_returns(c, sig, cost_bps=0.0)\n"
            "    s = st.summarize_overlay(d, 'gross')\n"
            "    lo, hi = st.block_bootstrap_ci(d['gross'].to_numpy(), block=21, n_boot=1000)\n"
            "    rows.append((name, round(s['mean_bps'], 2), round(s['sharpe'], 2),\n"
            "                 round(s['hac_t'], 2), f'[{lo*1e4:+.2f},{hi*1e4:+.2f}]', f\"{s['pct_invested']*100:.0f}%\"))\n"
            "bh = c.pct_change().fillna(0.0).to_numpy()\n"
            "rows.append(('ARKK buy&hold', round(np.nanmean(bh)*1e4, 2), round(st.sharpe(bh), 2),\n"
            "             round(st.hac_tstat(bh), 2), '-', '100%'))\n"
            "tbl = pd.DataFrame(rows, columns=['overlay', 'mean bps/d', 'Sharpe', 'HAC t', '95% CI bps', '% inv'])\n"
            "print('tape:', tape); display(tbl)\n"
            f"print('(pinned real: trend t=+{R['trend_t']} CI [{R['trend_ci_lo']:+.2f},{R['trend_ci_hi']:+.2f}], ts-mom t=+{R['tsmom_t']}, mr t=+{R['mr_t']})')"
        ),
        md(
            "### 4c · Alpha vs beta — the excess-of-cash Sharpe race\n\n"
            "> 💡 **In plain words.** The trend overlay sits in cash ~42% of the time, so comparing "
            "its raw Sharpe to a fully-invested index is unfair. We subtract cash from both and race "
            "it against ARKK *and* QQQ buy-and-hold."
        ),
        code(
            "trend = st.overlay_returns(c, st.ma_crossover_signal(c, 50, 200), 0.0)['gross'].to_numpy()\n"
            "arkk_bh = c.pct_change().fillna(0.0).to_numpy()\n"
            "if HAVE_REAL:\n"
            "    q = real_close('QQQ'); qqq_bh = q.pct_change().fillna(0.0).to_numpy()\n"
            "    m = min(len(trend), len(qqq_bh)); src = 'REAL QQQ'\n"
            "else:\n"
            "    qqq_bh = data.synthetic_daily(n_days=len(c), momentum=0.0, drift=0.0006, seed=7)['close'].pct_change().fillna(0.0).to_numpy()\n"
            "    m = min(len(trend), len(qqq_bh)); src = 'SYNTHETIC QQQ stand-in'\n"
            "vs_arkk = st.excess_sharpe(trend, arkk_bh)\n"
            "vs_qqq = st.excess_sharpe(trend[:m], qqq_bh[:m])\n"
            "print('benchmark source:', src)\n"
            "print(f'  trend overlay excess Sharpe : {vs_arkk[\"strat_excess_sharpe\"]:.2f}')\n"
            "print(f'  ARKK buy&hold excess Sharpe : {vs_arkk[\"bench_excess_sharpe\"]:.2f}  (diff {vs_arkk[\"diff\"]:+.2f})')\n"
            "print(f'  QQQ  buy&hold excess Sharpe : {vs_qqq[\"bench_excess_sharpe\"]:.2f}  (diff {vs_qqq[\"diff\"]:+.2f})')\n"
            f"print('  pinned real: overlay vs ARKK diff +{R['excess_sharpe_diff']}; QQQ B&H Sharpe {R['qqq_sharpe']}')"
        ),
        md(
            "### 4d · The behaviour gap — time-weighted vs dollar-weighted, with a positive control\n\n"
            "> 💡 **In plain words.** The IRR machinery: what did the *average dollar* earn, given "
            "when the money arrived? We plant a known gap on a synthetic fund and recover it (a "
            "machinery proof — never market evidence), then state the real figure from the literature."
        ),
        code(
            "rows = []\n"
            "for label, bust_, chase in [('null: steady grower', 0.0, 0.0),\n"
            "                            ('boom-bust + chasing', 0.04, 15.0)]:\n"
            "    fr, _ = data.synthetic_hype_cycle(n_months=180, bust=bust_, chase=chase, seed=334)\n"
            "    g = st.behaviour_gap(fr['price'], fr['flow'], st.MONTHS_PER_YEAR)\n"
            "    rows.append((label, round(g['twr_cagr']*100, 1), round(g['dwr_irr']*100, 1), round(g['gap']*100, 1)))\n"
            "tbl = pd.DataFrame(rows, columns=['synthetic fund', 'time-weighted %/yr', 'dollar-weighted %/yr', 'gap pts'])\n"
            "display(tbl)\n"
            "print('SYNTHETIC positive control — the IRR engine recovers the planted gap.')\n"
            f"print('Real ARKK (literature): fund time-weighted +{R['cagr_full']}%/yr, but the dollar-weighted')\n"
            f"print('return was deeply negative — Morningstar est. ~${R['wealth_destroyed_lo']:.0f}-{R['wealth_destroyed_hi']:.0f}bn of investor wealth destroyed.')"
        ),
        md(
            "### 4e · The momentum engine's own positive control\n\n"
            "Sanity check that the overlay machinery *can* detect momentum when it exists — so its "
            "near-silence on real ARKK is a finding, not a broken pipeline."
        ),
        code(
            "for label, mom in [('null: no momentum', 0.0), ('planted momentum', 0.20)]:\n"
            "    s = data.synthetic_daily(n_days=3000, momentum=mom, drift=0.0, seed=334)\n"
            "    d = st.overlay_returns(s['close'], st.ts_momentum_signal(s['close'], 1), 0.0)\n"
            "    t = st.summarize_overlay(d, 'gross')['hac_t']\n"
            "    print(f'  {label:20s} HAC t = {t:+.2f}')\n"
            f"print('  (pinned: null t={R['mom_null_t']}, planted t=+{R['mom_planted_t']})')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Mixed.** Trend 50/200 clears the bar (HAC *t* = +{R['trend_t']}, block-bootstrap "
            f"CI [{R['trend_ci_lo']:+.2f}, {R['trend_ci_hi']:+.2f}] bps) but only as crash avoidance; "
            f"ts-momentum (+{R['tsmom_t']}) and mean-reversion (+{R['mr_t']}) are None. *Real on the slow "
            "trend · None on momentum/reversion.*\n"
            f"- **Tradability — Mirage.** Overlay excess Sharpe over ARKK B&H is +{R['excess_sharpe_diff']}, "
            f"and it loses outright to QQQ B&H (Sharpe {R['qqq_sharpe']}). No investable edge.\n"
            f"- **Buy-the-top machine? — Confirmed.** Fund +{R['cagr_full']}%/yr, but a +{R['total_boom']:.0f}% "
            f"boom into a {R['dd']:.0f}% bust with assets concentrated near the top → dollar-weighted "
            "return deeply negative."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it?\n\n"
            f"The trend overlay is robust to costs (still HAC *t* = +{R['cost20_t']} at 20 bps round-trip) "
            "because it trades rarely — but cost was never the binding constraint. **Capacity and "
            "selection are:** the only value the rule adds is sidestepping a drawdown a diversified "
            f"QQQ holder (Sharpe {R['qqq_sharpe']}, +{R['qqq_cagr']}%/yr) never suffered. You'd be taking "
            "concentrated single-fad risk to *underperform* an index fund. The honest break-even isn't "
            "a cost level — it's the realisation that the trade shouldn't be put on at all."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Real flows.** Drop ARKK's monthly shares-outstanding into `data.fetch_flows` to "
            "replace the literature estimate with a computed dollar-weighted IRR off this exact engine.\n"
            "- **Cross-section.** Run the behaviour gap across the thematic-ETF universe (opt into the "
            "survivorship guard — dead thematic ETFs bias the gap *downward*, so the survivors "
            "understate it) to test Ben-David et al.'s product-level buy-the-top claim.\n"
            "- **Regime split.** The trend overlay's whole edge is one crash; test it pre-2021 alone "
            "(it should vanish) to confirm it's crash-insurance, not timing skill."
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
