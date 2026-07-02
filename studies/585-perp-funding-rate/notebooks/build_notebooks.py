"""Generate the two narrative notebooks for Study 585 (Perp-Funding-Rate).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The reproducible core is
**synthetic** — funding-rate history is not free (see ../perp_funding_rate/data.py) — so every
signal cell runs on the deterministic seeded panel, offline. The frozen headline numbers live in
one dict ``R`` (mirroring docs/results.md). The BTC/ETH price tape is loaded cache-first only to
make the missing-funding-data limitation concrete; it never produces a funding signal.
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


# Frozen synthetic headline numbers — mirror of docs/results.md (seed 585, 2000 periods, planted
# contrarian_beta = -0.012, horizon 3, panel fp 8121108453d2; as-of 2026-06-30). All from RUNNING
# the engine; the synthetic panel is deterministic so these reproduce exactly.
R = dict(
    fp_panel="8121108453d2", fp_px="db49481bb5f1",
    n=2000, k=400, beta=-0.012,
    cold_mean=1.40, hot_mean=-1.60, spread=3.01, ls_t=8.08, placebo_p=0.0005,
    slope=-0.0111, slope_t=-9.44, corr=-0.21,
    net=2.47, cost_bps=6.0, drag_bps=30.0,
    # horizon sweep: (label, spread%, slope_t)
    hor=[("1 (~8h)", 2.88, -15.58), ("3 (~1d)", 3.01, -9.44),
         ("6 (~2d)", 3.61, -8.07), ("9 (~3d)", 3.10, -6.26)],
    # null single-seed
    null_slope_t=0.75, null_ls_t=-0.98, null_placebo=0.344,
    # seed-robust control: (beta, mean slope-t over 25 seeds)
    ctrl=[(0.0, -0.17), (-0.006, -5.34), (-0.012, -10.51), (-0.020, -17.40)],
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from perp_funding_rate import data, strategy as st

# The reproducible core is SYNTHETIC — funding history is not free. The panel is deterministic
# (seed 585), so every number below reproduces exactly, offline.
BETA = -0.012
PANEL, TRUTH = data.synthetic_panel(contrarian_beta=BETA)
print(f"synthetic funding panel: {len(PANEL)} periods, planted contrarian_beta={BETA}, "
      f"fp {data.fingerprint(PANEL)}")

# The BTC/ETH PRICE tape (free) — loaded ONLY to make the missing-funding limitation concrete.
PX = data.fetch_btc_eth_prices(fetch=False)
HAVE_PX = not PX.empty
print("free BTC/ETH price tape present:", HAVE_PX,
      "" if not HAVE_PX else f"({len(PX)} rows, funding-rate tape is NOT free -> synthetic core)")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Perp Funding Rates — when longs get greedy, is a flush coming? 🪙\n"
            "### Reading the crowd's leverage off the funding rate, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "A crypto **perpetual swap** ('perp') is a futures contract with no expiry date. To keep "
            "its price glued to the real spot price, exchanges make traders pay each other a small "
            "fee every 8 hours — the **funding rate**. When lots of people are aggressively *long* "
            "and levered, the perp trades a touch above spot, funding goes **positive**, and the "
            "longs pay the shorts. When everyone has capitulated and gone short, funding goes "
            "**negative** and the shorts pay.\n\n"
            "So funding is a live read on **how the crowd is positioned**. The folklore: when funding "
            "runs *hot* (very positive), the longs are over-crowded and over-levered — and a "
            "**flush** (a sharp drop that liquidates them) is due. That makes hot funding a "
            "**contrarian** signal: fade the crowd.\n\n"
            "We'd love to test that on the real tape. We **can't** — and that turns out to be the "
            "whole story.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is the 'hot funding → flush' idea plausible? | **Yes.** It's a well-liked crypto-desk "
            "story with real mechanics behind it — funding really does track how levered the crowd "
            "is. |\n"
            "| Did we prove it on the real market? | **No — we *can't*.** BTC/ETH *prices* are free, "
            "but the historical **funding-rate tape** (the whole signal) is locked behind **paid** "
            "exchange APIs. No funding data → no real test. |\n"
            "| So what *did* we do? | Built a **synthetic** world where we *plant* the effect, and "
            "showed our engine catches it cleanly — and stays silent when nothing is planted. |\n"
            "| Could you trade it if it were real? | **Barely.** It's the highest-turnover trade "
            "around (rebalance every 8 hours), and the short leg literally *pays* the funding it's "
            "betting on. |\n\n"
            "> The honest verdict: the idea is plausible, the machinery works, but the data that would "
            "make it a **real** result isn't free — so we cap it below `REAL` and say so out loud."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"When perpetual-futures funding rates run hot, longs are over-leveraged and due for a "
            "flush.\"* — crypto-desk folklore (Coinglass funding heatmaps, exchange & data-vendor "
            "research).\n\n"
            "Why it's tempting: funding is one of the cleanest *positioning* gauges in any market. In "
            "stocks you guess at crowding; in crypto perps the exchange literally *charges* the "
            "crowded side. If the levered longs are all on one boat, a small wobble can cascade into "
            "forced liquidations — a flush. Fade the hot funding, buy the cold panic."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If it worked, it'd be a rare thing: a *positioning* signal you can read off a public "
            "number, on the most liquid 24/7 market there is. It would also say something deep — that "
            "crowds in crypto are so reflexive that their own leverage predicts their own undoing. "
            "The desk has poked at crypto's price and calendar and sentiment signals already "
            "([133](../../133-crypto-seasonality/), [175](../../175-crypto-weekend/), "
            "[210](../../210-crypto-trend/), [251](../../251-crypto-reversal/), "
            "[325](../../325-crypto-fear-greed/)); this is the one whose whole input is the *funding "
            "rate* — and that's exactly the input we can't get for free."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Get the funding tape.** For each 8-hour period, the funding rate.\n"
            "2. **Mark 'hot' and 'cold'.** Standardize funding against its recent history, so 'hot' "
            "means unusually high *right now*.\n"
            "3. **Look forward.** After hot funding, did the price fall (a flush)? After cold funding, "
            "did it bounce? The contrarian claim says yes.\n\n"
            "The catch that sinks the whole thing: **step 1 isn't free.** yfinance gives us BTC/ETH "
            "*prices* — but funding history lives behind paid exchange APIs (Binance, Bybit, "
            "Coinglass, Amberdata). Let's *see* the free price tape, then face the gap:"
        ),
        code(
            "if HAVE_PX:\n"
            "    fig, ax = plt.subplots(figsize=(9, 4.2))\n"
            "    ax.plot(PX.index, PX['BTC-USD'], color=AMBER, lw=1.2, label='BTC-USD (free)')\n"
            "    ax.set_ylabel('BTC price (USD)'); ax.set_yscale('log')\n"
            "    ax.set_title('The price tape is free. The FUNDING tape (the signal) is NOT.')\n"
            "    ax.legend(loc='upper left'); plt.tight_layout(); plt.show()\n"
            "    print('We have', len(PX), 'daily BTC/ETH closes -> but ZERO funding-rate history.')\n"
            "else:\n"
            "    print('price cache absent (offline) — the point stands: funding history is not free.')"
        ),
        md(
            "That price line is *all* a no-key stack gets. The funding rate — the entire signal — "
            "isn't in it. So we do the only honest thing: build a **synthetic** funding world where "
            "we control the truth, prove the engine works there, and refuse to claim a real result we "
            "can't earn.\n\n"
            "*Timing note:* in the synthetic world the funding rate at each period is paired with the "
            "**next** period's returns — we never let the future leak into the signal."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — the synthetic world\n\n"
            "We build a made-up but realistic funding series (it drifts and mean-reverts, like real "
            "funding) and **plant** a contrarian effect: hot funding really does lead to lower "
            "forward returns. Then we sort periods into a *hot* fifth and a *cold* fifth and compare "
            "what happened next.\n\n"
            "> ⚙️ **Everything below is synthetic** (a planted effect), *not* the market. It shows "
            "what a *real* signal would look like — and proves our detector isn't blind."
        ),
        code(
            "b = st.tail_buckets(PANEL); ls = st.long_short_tstat(b)\n"
            "cold_m, hot_m, spr, t = b['cold_mean']*100, b['hot_mean']*100, b['spread']*100, ls['t']\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['COLD funding\\n(panic shorts)','HOT funding\\n(crowded longs)'],\n"
            "       [cold_m, hot_m], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'Planted contrarian world: cold bounces (+{cold_m:.1f}%), hot flushes ({hot_m:.1f}%)')\n"
            "fig.suptitle('SYNTHETIC (planted effect) — not the market', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Cold +{cold_m:.1f}%  vs  Hot {hot_m:.1f}%  ->  spread {spr:+.1f}% (t {t:+.2f})')"
        ),
        md(
            f"On the planted tape the contrarian trade works by design: the cold-funding fifth bounced "
            f"**+{R['cold_mean']}%**, the hot-funding fifth flushed **{R['hot_mean']}%**, a spread of "
            f"**+{R['spread']}%** (*t* **+{R['ls_t']}**). That's what a *real* funding signal would "
            "look like — clean and strong.\n\n"
            "**But does our engine only ever say 'yes'?** The crucial test: turn the effect **off** "
            "(plant nothing) and check it goes quiet."
        ),
        code(
            "pn, _ = data.synthetic_panel(contrarian_beta=0.0)\n"
            "bn = st.tail_buckets(pn); lsn = st.long_short_tstat(bn)\n"
            "spr_planted = st.tail_buckets(PANEL)['spread']*100\n"
            "spr_null = bn['spread']*100\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['planted effect','NULL\\n(nothing planted)'], [spr_planted, spr_null],\n"
            "       color=[GREEN, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('cold - hot spread %')\n"
            "ax.set_title('Engine fires on the planted effect, goes quiet at the null')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'planted spread {spr_planted:+.1f}% (t {st.long_short_tstat(st.tail_buckets(PANEL))[\"t\"]:+.2f})  |  '\n"
            "      f'null spread {spr_null:+.1f}% (t {lsn[\"t\"]:+.2f})')"
        ),
        md(
            f"Good — at the null the spread collapses toward zero (*t* ≈ {R['null_ls_t']}, inside the "
            "noise band). The detector isn't a yes-machine: it fires on a real planted effect and "
            "stays quiet when there's nothing there. So if the market signal were real, we'd catch "
            "it. We just can't point the engine at a real funding tape."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Not because the idea is bad — because we **cannot test it for "
            "real**. Funding history isn't free, so there's no real number, and the desk only stamps "
            "`REAL` when a real tape clears the bar. The synthetic control proves the engine works; "
            "that earns respect, not a `REAL`.\n"
            "- **Tradability — Mirage.** Even granting the effect, it's an 8-hourly rebalance and the "
            "short leg *pays* the funding it trades on. Costs eat it."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Barely, even in the best case. The signal refreshes every 8 hours, so you'd rebalance "
            "three times a day — the highest turnover on the desk, and every reform pays taker fees "
            "and slippage. Worse, the *hot-funding short* leg means you are **paying the positive "
            "funding** to hold the short: the trade's own thesis (funding is high) is also its running "
            "cost. On the planted tape a +3.01% gross spread nets +2.47% *per rebalance* — but that's "
            "a made-up number; on a real, noisy funding tape the net edge would be a small difference "
            "of large, expensive numbers."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The free-data crypto studies.** [133 Seasonality](../../133-crypto-seasonality/), "
            "[175 Weekend](../../175-crypto-weekend/), [210 Trend](../../210-crypto-trend/), "
            "[251 Reversal](../../251-crypto-reversal/), [325 Fear-Greed](../../325-crypto-fear-greed/) "
            "— crypto signals you *can* test, because their input is the free price tape.\n"
            "- **The other synthetic-only studies.** [273 Lego](../../273-lego-returns/), "
            "[275 Whisky](../../275-whisky-cask/), [276 Sneakers](../../276-sneaker-resale/) — same "
            "honesty: no real free tape, so build synthetic and say so.\n\n"
            "*Have a paid funding feed (Coinglass / Amberdata / a Binance key)? Fork this, drop the "
            "real funding tape into `data.py`, and show whether the cold-minus-hot spread clears "
            "t = 2 on the live market — after 8-hourly costs.*"
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
            "# Perp Funding Rates — a quantitative teardown 🔬\n"
            "### trailing-z funding signal · quintile sort · two-sample *t* · label-shuffle placebo · period-level slope · horizon sweep · costs & funding drag · seed-robust synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* The contrarian funding claim: "
            "standardized funding predicts the forward return with a **negative** sign (hot funding → "
            "lower return). The catch is the data: funding history is **not free**, so the "
            "reproducible core is a deterministic **synthetic** panel with a planted effect, and the "
            "study is capped below `REAL` on the SIGNAL axis by house rule.\n\n"
            "> ⚠️ **Not investment advice.** The signal cells are **synthetic** (seed 585, planted "
            f"`contrarian_beta`, panel fp `{R['fp_panel']}`) — funding-rate history is paid exchange "
            "data. The free BTC/ETH price tape is shown only to make the data gap concrete. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | No real tape (funding history is paid). Synthetic control: "
            f"spread **+{R['spread']}%** (two-sample *t* **+{R['ls_t']}**, placebo *p* {R['placebo_p']}), "
            f"slope *t* **{R['slope_t']}**, sign stable 1→9 periods, flat at the null — the engine "
            "works, but synthetic-only caps at `NONE`. |\n"
            f"| **Tradability** | `MIRAGE` | 8-hourly rebalance; short leg pays the funding it keys on. "
            f"Synthetic gross +{R['spread']}% → net +{R['net']}% ({R['cost_bps']:.0f} bps/leg + "
            f"{R['drag_bps']:.0f} bps drag) *per rebalance*. |\n\n"
            "> 💡 In plain words: the machinery is sound (the seed-robust control proves it), but the "
            "input that would make this a real test is behind a paywall — so `NONE`, honestly."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $f_t$ be the funding rate at period $t$, $z_t$ its trailing standardized value, and "
            "$r_{t\\to t+h}$ the forward return over the next $h$ funding periods.\n\n"
            "- **H₁ (the contrarian long-short).** $\\mathbb{E}[r \\mid \\text{cold } z] - "
            "\\mathbb{E}[r \\mid \\text{hot } z] > 0$ at *t* ≥ 2 (cold bounces, hot flushes).\n"
            "- **H₂ (period-level).** $\\text{slope of } r \\text{ on } z < 0$ (hotter funding, lower "
            "forward return).\n"
            "- **H₃ (robustness).** The sign of H₁/H₂ is stable across forward horizons.\n"
            "- **H₄ (net).** H₁ survives 8-hourly costs and the short-side funding drag.\n\n"
            "We can only test these on a **synthetic** tape (funding history is not free). There, with "
            "a planted effect, **H₁–H₃ hold cleanly and H₄ nets positive** — but that certifies the "
            "*engine*, not the market. On the real tape, all four are **untested** for lack of data — "
            "which is why the Signal is `NONE`, not `REAL`."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **data gap**. Funding is arguably the cleanest positioning gauge in "
            "any market — the exchange charges the crowded side directly — so a real contrarian edge "
            "here would be a genuine anomaly. Two things stop us certifying it: (a) the **funding "
            "tape is paid** (Binance/Bybit `fapi`, Coinglass, Amberdata, Kaiko), so a no-key stack "
            "has no real series; and (b) even granting the effect, the **8-hourly turnover** and the "
            "short leg's **funding carry** make it structurally expensive. This is the one crypto "
            "signal on the desk whose *entire input* is unfree — unlike the price/calendar/sentiment "
            "studies ([133](../../133-crypto-seasonality/) … [325](../../325-crypto-fear-greed/))."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $z_t$ = trailing standardized funding (rolling window, min 20 obs), so "
            "'hot'/'cold' is defined against the recent past — fully public at $t$.\n"
            "- **Sort.** Quintile tails (≈400 periods each): cold (lowest $z$) vs hot (highest $z$).\n"
            "- **Inference.** Two-sample (Welch) *t* on cold − hot forward returns; a **label-shuffle "
            "placebo** null (2000 perms).\n"
            "- **Period-level.** OLS of forward return on $z$ — the *sign* of the slope is the claim.\n"
            "- **Robustness.** Re-run over forward horizons 1, 3, 6, 9 funding periods; read the sign.\n"
            "- **Frictions.** Round-trip cost per leg + a short-side funding drag (you pay the "
            "positive funding to hold the hot short).\n"
            "- **Positive control.** A deterministic synthetic world with a planted effect "
            "(`contrarian_beta < 0`) and a null — averaged over 25 seeds (house rule).\n\n"
            "Timing: funding is observed at $t$ (public at that close); the position is entered at "
            "that close and held over the strictly-future forward window. No look-ahead. The "
            "same-bar fill is a documented convention; a one-period-later entry leaves the sign and "
            "stamp unchanged."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown (synthetic — planted effect)"),
        md(
            "### 4a · The sort — H₁\n\n"
            "Cold vs hot forward returns on the planted synthetic tape, with the two-sample *t* and "
            "the placebo null."
        ),
        code(
            "b = st.tail_buckets(PANEL); ls = st.long_short_tstat(b)\n"
            "p = st.placebo_pvalue(PANEL, n_perm=2000, seed=585)\n"
            "cold_m, hot_m, spr, t = b['cold_mean']*100, b['hot_mean']*100, b['spread']*100, ls['t']\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['COLD (z low)','HOT (z high)'], [cold_m, hot_m], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'cold - hot = {spr:+.1f}%  (two-sample t {t:+.2f}, placebo p {p:.4f})')\n"
            "fig.suptitle('SYNTHETIC (planted effect) — not the market', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Cold +{cold_m:.1f}% | Hot {hot_m:.1f}% | spread {spr:+.1f}% | t {t:+.2f} | placebo p {p:.4f}')"
        ),
        md(
            f"> 💡 In plain words: on the planted tape H₁ holds strongly — spread **+{R['spread']}%** "
            f"(*t* +{R['ls_t']}), placebo *p* = {R['placebo_p']}. This is the *positive control*: it "
            "shows the sort and its *t*-machinery recover a real cold-bounce / hot-flush when one "
            "exists. It is **not** a market result."
        ),
        md(
            "### 4b · The period-level slope — H₂\n\n"
            "Regress forward return on standardized funding. A *negative* slope is the contrarian "
            "effect."
        ),
        code(
            "reg = st.funding_regression(PANEL)\n"
            "slope, slope_t, corr = reg['slope'], reg['slope_t'], reg['corr']\n"
            "z = PANEL['z_funding'].to_numpy(); y = PANEL['fwd_ret'].to_numpy()*100\n"
            "fig, ax = plt.subplots(figsize=(8, 4.6))\n"
            "ax.scatter(z, y, s=6, color=GREY, alpha=.4)\n"
            "xs = np.linspace(z.min(), z.max(), 50)\n"
            "ax.plot(xs, (reg['slope']*xs + (y.mean()/100 - reg['slope']*z.mean()))*100, c=RED, lw=2)\n"
            "ax.set_xlabel('standardized funding z (higher = hotter)'); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'slope {slope:.4f}/z-unit (t {slope_t:+.2f}) — NEGATIVE = contrarian')\n"
            "fig.suptitle('SYNTHETIC (planted effect) — not the market', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'slope {slope:.4f}/z-unit, slope-t {slope_t:+.2f}, corr(funding, fwd) {corr:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: H₂ recovered — slope **{R['slope']}** per z-unit (*t* {R['slope_t']}), "
            f"the contrarian sign, with corr {R['corr']}. Hotter funding → lower forward return, by "
            "construction. The detector reads the planted sign correctly."
        ),
        md(
            "### 4c · Robustness — H₃ (does the sign hold across horizons?)\n\n"
            "Re-run the sort and slope over forward horizons of 1, 3, 6 and 9 funding periods."
        ),
        code(
            "rows = st.horizon_sweep(data, BETA, horizons=(1,3,6,9))\n"
            "tab = pd.DataFrame([(l, s*100, t) for (l,s,t) in rows],\n"
            "                   columns=['horizon_periods','spread%','slope_t']).set_index('horizon_periods')\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "cols = [GREEN if s>0 else RED for s in tab['spread%']]\n"
            "ax.bar(range(len(tab)), tab['spread%'], color=cols, width=.55)\n"
            "ax.set_xticks(range(len(tab))); ax.set_xticklabels([f'{h}p' for h in tab.index])\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('cold - hot spread %')\n"
            "ax.set_title('Planted sign is stable across horizons (all green, all t < -2)')\n"
            "fig.suptitle('SYNTHETIC (planted effect) — not the market', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(2).to_string())"
        ),
        md(
            "> 💡 In plain words: H₃ holds on the synthetic tape — the contrarian sign is stable and "
            "the slope-*t* stays past −2 from ~8h out to ~3 days. So the engine is not horizon-fragile. "
            "(Again: engine, not market.)"
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — no real funding tape to test (paid data), so H₁–H₄ are "
            f"**untested on the market**. The synthetic control clears every bar (spread +{R['spread']}%, "
            f"slope *t* {R['slope_t']}, placebo {R['placebo_p']}, sign-stable, flat at null) — proving "
            "the engine, not the anomaly. House rule: synthetic-only caps below `REAL`.\n"
            f"- **Tradability `MIRAGE`** — 8-hourly rebalance, short leg pays the funding it keys on; "
            f"synthetic gross +{R['spread']}% → net +{R['net']}%.\n"
            "- **Data-availability limitation** named openly on the Signal axis (like "
            "[273](../../273-lego-returns/) / [275](../../275-whisky-cask/) / "
            "[276](../../276-sneaker-resale/))."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the funding drag\n\n"
            "Charge the synthetic spread per 8-hourly rebalance: a round-trip cost on each leg plus a "
            "punitive short-side **funding drag** (the hot short *pays* funding)."
        ),
        code(
            "b = st.tail_buckets(PANEL)\n"
            "gross = b['spread']*100\n"
            "net = st.net_spread(b, cost_bps=6.0, funding_drag_bps=30.0)*100\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross','net\\n(costs+funding drag)'], [gross, net], color=[GREEN, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('cold - hot spread % (per rebalance)')\n"
            "ax.set_title(f'Synthetic edge shrinks with costs: {gross:+.1f}% -> {net:+.1f}%')\n"
            "fig.suptitle('SYNTHETIC (planted effect) — not the market', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:+.1f}% -> net {net:+.1f}% (6 bps/leg + 30 bps funding drag, per 8h rebalance)')"
        ),
        md(
            "> 💡 In plain words: even a *planted, clean* edge gives back a chunk to costs, and this is "
            "per-rebalance — three times a day. On a real, noisy funding tape (no planted effect) the "
            "net would be a small difference of large, expensive numbers. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the seed-robust synthetic control\n\n"
            "Is the engine a faithful detector, or does it always print a signal? Plant a contrarian "
            "effect (`contrarian_beta < 0`) of increasing strength and watch the period-level slope-*t* "
            "go negative — **averaged over 25 seeds** so no single lucky seed can fake it (house rule)."
        ),
        code(
            "betas = [0.0, -0.003, -0.006, -0.009, -0.012, -0.016, -0.020]\n"
            "ts = [st.synthetic_mean_t(data, contrarian_beta=b, n_seeds=25) for b in betas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(betas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1, label='t = -2 bar (contrarian)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted contrarian_beta (more negative = stronger effect)')\n"
            "ax.set_ylabel('mean period slope-t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted effect — flat at the null, past -2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for b, t in zip(betas, ts): print(f'beta {b:+.3f} -> mean slope-t {t:+.2f}')"
        ),
        md(
            "The slope-*t* is ≈ 0 at the null and falls past −2 as the planted effect grows — a "
            "faithful detector. So the study's limitation is the **missing real funding tape**, not a "
            "broken engine. The perp-funding contrarian idea stays `NONE` on this desk until someone "
            "pipes in a paid funding feed and clears *t* = 2 on the live market — after 8-hourly costs. "
            "For crypto signals we *can* test on free data, see "
            "[251 Crypto-Reversal](../../251-crypto-reversal/) and "
            "[325 Crypto-Fear-Greed](../../325-crypto-fear-greed/)."
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
