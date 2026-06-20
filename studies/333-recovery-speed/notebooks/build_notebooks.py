"""Generate the two narrative notebooks for Study 333 (Recovery-Speed).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
cells run anywhere, offline and deterministic (the positive control + null). The
real-tape cells use the cached daily parquets under ../_cache/ if present and otherwise
quote the frozen headline numbers in ``R`` (mirroring docs/results.md), so the notebook
re-runs for any reader — and every cell banners which tape it is showing.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-05-29,
# fingerprint 51c5a3a1d9e4, 18 survivor large-caps + SPY).
R = dict(
    asof="2026-05-29", fp="51c5a3a1d9e4", n_names=18,
    # pooled across 4 drawdowns, market-neutral, gross (long%, short%, LS%, t, CI)
    p63_long=-1.6, p63_short=-4.4, p63_ls=2.8, p63_t=0.58,
    p126_long=1.0, p126_short=-0.3, p126_ls=1.3, p126_t=0.23, p126_lo=-9.4, p126_hi=13.0,
    p252_ls=1.5, p252_t=0.13,
    # the cherry-pick
    gfc_ls=22.7, gfc_t=2.78,
    # cost sweep (pooled 126, net, %)
    cost0=1.1, cost5=0.9, cost10=0.7, cost20=0.3,
    # synthetic control / null
    syn_null_ls=3.9, syn_null_t=0.97, syn_null_p=0.203,
    syn_ctrl_ls=19.1, syn_ctrl_t=3.30, syn_ctrl_p=0.003,
)


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

from recovery_speed import data, strategy as st

def _have_real():
    try:
        data.load_real(fetch=False, allow_survivorship_bias=True)
        return True
    except Exception:
        return False

HAVE_REAL = _have_real()

def real_panel():
    panel, mkt, meta = data.load_real(allow_survivorship_bias=True)
    panel = panel[panel.index <= "2026-05-31"]; mkt = mkt[mkt.index <= "2026-05-31"]
    return panel, mkt, meta

print("real survivor cache present:", HAVE_REAL)
print("NOTE: real cells fall back to frozen docs/results.md numbers when offline.")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Recovery-Speed — do the fastest names to bounce back keep leading?\n"
            "### A piece of post-crash folklore, sorted, pooled, and taken apart in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![A_fast--recoverers_rule%3F: Cherry--picked](https://img.shields.io/badge/A_fast--recoverers_rule%3F-Cherry--picked-8b949e?style=flat-square)\n\n"
            "After every crash someone says it: *the stocks that climb back to their old highs "
            "fastest are the strong ones, and they keep winning.* It sounds obviously true — "
            "strength begets strength. This notebook ranks names by **recovery speed** after a "
            "drawdown and asks whether the fast recoverers actually keep leading. Spoiler: across "
            "four real crashes, no — and the one crash where it 'works' is a textbook cherry-pick.\n\n"
            "> 📓 **This is the plain-language layer.** Want the cross-sectional sort, the pooled "
            "t-stats and the synthetic control? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it; offline it falls back to a synthetic tape and says so. House "
            "style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do fast recoverers keep leading? | **No.** Pooled across four crashes the "
            f"fast-minus-slow spread is **+{R['p126_ls']:.1f}% at six months (*t* = +{R['p126_t']:.2f})** — "
            "indistinguishable from zero. |\n"
            "| But doesn't it work after the GFC? | **Only after the GFC.** That one event prints "
            f"**+{R['gfc_ls']:.0f}%** (*t* = +{R['gfc_t']:.2f}). Pick that crash, ignore the rest, and "
            "you have a 'strategy.' That's cherry-picking. |\n"
            "| Could the data be hiding losers? | **Yes — and it helps the claim.** Our names are "
            "*survivors*; the firms that crashed and never came back are gone. That bias works "
            "*for* 'fast recoverers win' — and it still fails. |\n"
            "| So what is it? | A plausible story with **no edge once you stop snooping**. |\n\n"
            "> Recovery-speed momentum is the kind of idea that's true in hindsight and absent in "
            "advance: every bull market has a poster child that roared back, but you can't sort on "
            "it ahead of time and get paid."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"After a deep drawdown, the names that recover to their old highs **fastest** are "
            "the strongest companies — buy the fast recoverers and they keep leading the laggards.\"*\n\n"
            "It's a cross-sectional cousin of momentum, conditioned on a crash. The intuition is "
            "real-sounding: a stock that V-bounces while the market is still on its back must have "
            "something going for it. The bet is that 'something' persists."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it were true it would be a gift: crashes are exactly when everyone is paralysed, "
            "and a simple rule — *rank by recovery speed, buy the top, short the bottom* — would "
            "turn the chaos into a signal. It would also tell us something deep about markets: that "
            "the snap-back ranks companies by hidden quality. The flip side is just as instructive. "
            "If the rule only works on the one crash you happened to look at, it's a perfect lesson "
            "in how a handful of events plus a survivor universe can manufacture a 'strategy' out of "
            "noise."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three honesty checks:\n\n"
            "1. **Pool every crash, don't pick one.** Find *all* the big drawdowns in the sample "
            "(2008, 2020, 2022, 2025), run the fast-minus-slow book in each, and pool the results. "
            "One event is a coincidence; four is a test.\n"
            "2. **Strip out the market.** A recovery rally is a market move; high-beta names bounce "
            "faster *and* keep rallying just because stocks are rising. We measure the long-short "
            "**net of the market**, so we don't mistake beta for skill.\n"
            "3. **Mind the dead.** Our universe is survivors — the firms that crashed and died are "
            "missing, which *flatters* the claim. If it fails even then, it fails.\n\n"
            "And a sanity check on the machine itself: a synthetic market where we *plant* a real "
            "recovery-momentum edge, to prove the engine can find one when it's there."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the cherry-pick vs the reality.** Here's the fast-minus-slow six-month spread "
            "for the single deepest crash (the GFC) next to the honest average across all four:"
        ),
        code(
            "labels = ['GFC only\\n(1 crash)', 'Pooled\\n(4 crashes)']\n"
            "if HAVE_REAL:\n"
            "    panel, mkt, meta = real_panel()\n"
            "    evs = st.multi_events(mkt)\n"
            "    ev0 = st.detect_drawdown(mkt, min_depth=0.18)\n"
            "    gfc = st.run_event(panel, mkt, ev0, fwd_days=126, market_neutral=True)\n"
            "    gfc_ls = gfc['ls_gross']*100; gfc_t = st.spread_tstat(gfc)\n"
            "    pl = st.pooled_spread(panel, mkt, evs, fwd_days=126, market_neutral=True)\n"
            "    pool_ls = pl['ls_gross']*100; pool_t = pl['spread_t']\n"
            "    banner = 'REAL tape (survivor large-caps, as-of %s)' % meta.get('market','SPY')\n"
            "else:\n"
            f"    gfc_ls, gfc_t = {R['gfc_ls']}, {R['gfc_t']}\n"
            f"    pool_ls, pool_t = {R['p126_ls']}, {R['p126_t']}\n"
            "    banner = 'FROZEN docs/results.md numbers (offline)'\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "b = ax.bar(labels, [gfc_ls, pool_ls], color=[AMBER, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('fast − slow, 6-month (%)')\n"
            "ax.set_title('One crash looks like a strategy; four crashes look like noise')\n"
            "for bar_, t_ in zip(b, [gfc_t, pool_t]):\n"
            "    ax.annotate(f't={t_:+.2f}', (bar_.get_x()+bar_.get_width()/2, bar_.get_height()),\n"
            "                ha='center', va='bottom')\n"
            "ax.text(0.99, 0.02, banner, transform=ax.transAxes, ha='right', va='bottom',\n"
            "        fontsize=8, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'GFC alone: {gfc_ls:+.1f}% (t={gfc_t:+.2f})   Pooled: {pool_ls:+.1f}% (t={pool_t:+.2f})')"
        ),
        md(
            f"There it is. The GFC alone hands you **+{R['gfc_ls']:.0f}%** at *t* = +{R['gfc_t']:.2f} — "
            f"a 'strategy.' Pool all four crashes and it collapses to **+{R['p126_ls']:.1f}%** at "
            f"*t* = +{R['p126_t']:.2f}, with a confidence range "
            f"(**{R['p126_lo']:.0f}% to +{R['p126_hi']:.0f}%**) that swallows zero whole. The signal "
            "was never there — one lucky episode was."
        ),
        md(
            "**Now the sanity check: can the machine even find a real effect?** We build a synthetic "
            "market and *plant* genuine recovery-momentum (fast recoverers really do keep leading), "
            "then turn the knob off. The engine should bank the planted edge and find nothing in the "
            "null."
        ),
        code(
            "rows = []\n"
            "for edge, lab in [(0.0, 'No edge (null)'), (0.0015, 'Planted edge (control)')]:\n"
            "    p, m, _ = data.synthetic_panel(n_names=80, n_days=1500, recovery_edge=edge, seed=333)\n"
            "    ev = st.detect_drawdown(m, min_depth=0.10)\n"
            "    res = st.edge_vs_shuffle(p, m, ev, n_shuffle=200, seed=1)\n"
            "    rows.append((lab, res['real_ls']*100, res['spread_t']))\n"
            "dfm = pd.DataFrame(rows, columns=['tape','ls%','t'])\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "b = ax.bar(dfm['tape'], dfm['ls%'], color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('fast − slow spread (%)')\n"
            "ax.set_title('SYNTHETIC: the engine finds the edge only when it is really there')\n"
            "for bar_, t_ in zip(b, dfm['t']):\n"
            "    ax.annotate(f't={t_:+.2f}', (bar_.get_x()+bar_.get_width()/2, bar_.get_height()),\n"
            "                ha='center', va='bottom')\n"
            "ax.text(0.99, 0.02, 'SYNTHETIC control tape (offline, deterministic)',\n"
            "        transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dfm.round(2).to_string(index=False))"
        ),
        md(
            f"On the synthetic tape the engine is faithful: with no planted edge the spread is noise "
            f"(*t* = +{R['syn_null_t']:.2f}), and with a real edge planted it banks it cleanly "
            f"(*t* = +{R['syn_ctrl_t']:.2f}). So the flat *real* result isn't a broken detector — "
            "the market simply doesn't carry the effect."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Pooled across four crashes the fast-minus-slow spread is "
            f"+{R['p126_ls']:.1f}% at six months (*t* = +{R['p126_t']:.2f}), CI "
            f"[{R['p126_lo']:.0f}%, +{R['p126_hi']:.0f}%]. Noise.\n"
            "- **Tradability — Mirage.** There's no gross edge to charge costs against — it's gone "
            "before the first dollar of fees.\n"
            "- **A fast-recoverers rule? — Cherry-picked.** It exists only if you report the one "
            "crash that worked and ignore the other three — on a universe that already drops the "
            "dead."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's almost nothing to trade, but for completeness: even gross, the pooled spread "
            "sits inside its own error bars. Charging realistic costs and short-borrow just walks a "
            "tiny non-edge toward zero:"
        ),
        code(
            "costs = [0, 5, 10, 20]\n"
            "if HAVE_REAL:\n"
            "    panel, mkt, _ = real_panel(); evs = st.multi_events(mkt)\n"
            "    net = [st.pooled_spread(panel, mkt, evs, fwd_days=126, cost_bps=c,\n"
            "                            borrow_bps_ann=50.0)['ls_net']*100 for c in costs]\n"
            "    banner = 'REAL tape (pooled, 4 crashes)'\n"
            "else:\n"
            f"    net = [{R['cost0']}, {R['cost5']}, {R['cost10']}, {R['cost20']}]\n"
            "    banner = 'FROZEN docs/results.md numbers (offline)'\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps, one-way × NAV, both legs) + 50 bps/yr borrow')\n"
            "ax.set_ylabel('net fast − slow, 6-month (%)')\n"
            "ax.set_title('Nothing to charge costs against — the edge was never there')\n"
            "ax.text(0.99, 0.95, banner, transform=ax.transAxes, ha='right', va='top',\n"
            "        fontsize=8, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net spread by cost:', [round(x,2) for x in net])"
        ),
        md(
            "Unlike the desk's *fragile* edges (real but thin), this isn't a cost story at all. The "
            "gross number is already inside the noise — costs are irrelevant when there's no signal "
            "to erode."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Bigger, dead-included universe.** Our 18 survivors flatter the claim; the real test "
            "is a point-in-time universe with the delisted names. We expect it to look *worse*, not "
            "better.\n"
            "- **Is it just momentum?** A fast recoverer is a recent winner. Strip out plain "
            "cross-sectional momentum and see if anything recovery-specific is left (we doubt it).\n"
            "- **The exit, not the entry.** Maybe recovery speed times *risk* rather than *return* — "
            "fast recoverers as a lower-drawdown sleeve. A different question worth its own study.\n\n"
            "*Think your favourite crash's poster-child proves it? Fork this, add the drawdown you "
            "have in mind to the event list, and watch the pooled t-stat refuse to move.*"
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
            "# Recovery-Speed — a quantitative teardown 🔬\n"
            "### Cross-sectional sort · drawdown detection · pooled HAC/bootstrap inference · "
            "market-neutral race · the single-event cherry-pick · synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![A_fast--recoverers_rule%3F: Cherry--picked](https://img.shields.io/badge/A_fast--recoverers_rule%3F-Cherry--picked-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether a "
            "recovery-speed cross-sectional sort, pooled across every independent drawdown, carries "
            "a market-neutral long-short edge — and we show exactly how a single event manufactures "
            "significance.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo daily bars, 18 survivor large-caps + "
            "SPY, 2004–2026, as-of "
            f"{R['asof']} (fingerprint `{R['fp']}`); the offline core and tests run on a "
            "deterministic synthetic panel. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Pooled (4 crashes), market-neutral 126-day spread "
            f"**+{R['p126_ls']:.1f}%**, two-sample/HAC *t* = **+{R['p126_t']:.2f}**, bootstrap CI "
            f"**[{R['p126_lo']:.0f}%, +{R['p126_hi']:.0f}%]**; 63-day *t* = +{R['p63_t']:.2f}, "
            f"252-day *t* = +{R['p252_t']:.2f}. |\n"
            f"| **Tradability** | `MIRAGE` | Gross spread already inside its CI; net at 20 bps + "
            f"borrow = +{R['cost20']:.1f}%. No edge to charge costs against. |\n"
            f"| **A fast-recoverers rule?** | `CHERRY--PICKED` | Single deepest drawdown (GFC) prints "
            f"+{R['gfc_ls']:.0f}% at *t* = +{R['gfc_t']:.2f}; pooling the other three kills it. |\n\n"
            "> 💡 In plain words: the effect is one good crash wide. Pool honestly and it's gone — "
            "and the survivor universe was already helping it."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let an event be a market drawdown with peak $p$, trough $\\tau$, and recovery bar $c$ "
            "(first bar regaining a fraction of the loss). Score name $i$ by recovery speed\n\n"
            "$$\\text{speed}_i=\\frac{P_i[c]-P_i[\\tau]}{P_i[p]-P_i[\\tau]}\\Big/(c-\\tau),$$\n\n"
            "rank into quintiles, go long the top, short the bottom, hold $h$ bars from $c{+}1$.\n\n"
            "- **H₁ (signal).** The market-neutral long-short forward return is $>0$, pooled across "
            "all events.\n"
            "- **H₂ (not just one event).** It is not driven by a single drawdown.\n"
            "- **H₃ (not just beta/momentum).** It survives measuring returns in excess of the "
            "market.\n\n"
            "We find **H₁ rejected** (pooled *t* ≈ 0), **H₂ decisive** (only the GFC carries it), "
            "and **H₃ moot** (nothing left to attribute)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "A robust recovery-speed sort would be a rare *conditional* anomaly — a signal that only "
            "switches on after crashes, when capital is scarcest and the payoff highest. That makes "
            "the failure mode equally valuable: with only ~4 drawdowns per two decades, the "
            "*effective sample size of events is tiny*, so a single episode can dominate any naive "
            "average. Naming that — and quantifying how survivorship tilts the deck — is worth more "
            "than the binary stamp."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Event detection.** Deepest drawdown on the running max of SPY; recovery declared at "
            "the first bar regaining 85% of the peak-to-trough loss. Walk forward to enumerate "
            "*independent* episodes.\n"
            "- **Score & sort.** Per-name recovery speed at $c$; quintile long-short, equal weight.\n"
            "- **Execution.** One lag: signal at $c$'s close, enter at $c{+}1$. Costs one-way × NAV "
            "on both legs; short leg pays borrow.\n"
            "- **Market-neutral.** Forward returns in excess of the market over the same window.\n"
            "- **Inference.** Two-sample / HAC *t* on the pooled long-short cross-section; bootstrap "
            "CI by resampling names within each leg.\n"
            "- **Controls.** A single-event reading (the cherry-pick), and a synthetic panel with a "
            "plantable recovery-momentum edge + a shuffle-the-scores null.\n\n"
            "**Survivorship is named on the Signal axis:** the universe is current survivors, which "
            "biases the long-short *upward* — the slow-recoverers that died are absent from the short "
            "leg. The opt-in guard (`allow_survivorship_bias=True`) gates the loader."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Pooled inference across every drawdown\n\n"
            "The headline: the market-neutral long-short spread by horizon, pooled across the four "
            "detected drawdowns, with bootstrap CIs."
        ),
        code(
            "horizons = [63, 126, 252]\n"
            "if HAVE_REAL:\n"
            "    panel, mkt, meta = real_panel(); evs = st.multi_events(mkt)\n"
            "    ev_years = [mkt.index[e['trough']].year for e in evs]\n"
            "    res = [st.pooled_spread(panel, mkt, evs, fwd_days=h, market_neutral=True) for h in horizons]\n"
            "    ls = [r['ls_gross']*100 for r in res]; ts = [r['spread_t'] for r in res]\n"
            "    lo = [r['ci_lo']*100 for r in res]; hi = [r['ci_hi']*100 for r in res]\n"
            "    banner = f'REAL tape · events {ev_years}'\n"
            "else:\n"
            f"    ls = [{R['p63_ls']}, {R['p126_ls']}, {R['p252_ls']}]\n"
            f"    ts = [{R['p63_t']}, {R['p126_t']}, {R['p252_t']}]\n"
            f"    lo = [-6.3, {R['p126_lo']}, -19.9]; hi = [12.1, {R['p126_hi']}, 24.5]\n"
            "    banner = 'FROZEN docs/results.md numbers (offline)'\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "x = np.arange(len(horizons))\n"
            "ax.errorbar(x, ls, yerr=[np.array(ls)-np.array(lo), np.array(hi)-np.array(ls)],\n"
            "            fmt='o', color=RED, capsize=6, ms=9, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in horizons])\n"
            "ax.set_ylabel('fast − slow, market-neutral (%)')\n"
            "ax.set_title('Every horizon: spread indistinguishable from zero (CI straddles 0)')\n"
            "for xi, t_ in zip(x, ts): ax.annotate(f't={t_:+.2f}', (xi, 0.5), ha='center', color=GREY)\n"
            "ax.text(0.99, 0.02, banner, transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "for h, l, t in zip(horizons, ls, ts): print(f'{h}d: LS={l:+.1f}%  t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: at the canonical six-month horizon the pooled spread is "
            f"+{R['p126_ls']:.1f}% with a confidence interval from {R['p126_lo']:.0f}% to "
            f"+{R['p126_hi']:.0f}%. You cannot reject zero at any horizon. **H₁ rejected.**"
        ),
        md(
            "### 4b · The single-event cherry-pick\n\n"
            "Where does the folklore come from? From doing the test on *one* crash. The GFC alone "
            "vs the pool:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    panel, mkt, _ = real_panel(); evs = st.multi_events(mkt)\n"
            "    ev0 = st.detect_drawdown(mkt, min_depth=0.18)\n"
            "    g = st.run_event(panel, mkt, ev0, fwd_days=126, market_neutral=True)\n"
            "    gfc_ls, gfc_t = g['ls_gross']*100, st.spread_tstat(g)\n"
            "    pl = st.pooled_spread(panel, mkt, evs, fwd_days=126, market_neutral=True)\n"
            "    pool_ls, pool_t = pl['ls_gross']*100, pl['spread_t']\n"
            "    per = [(mkt.index[e['trough']].year,\n"
            "            st.run_event(panel, mkt, e, fwd_days=126, market_neutral=True)['ls_gross']*100)\n"
            "           for e in evs]\n"
            "    banner = 'REAL tape'\n"
            "else:\n"
            f"    gfc_ls, gfc_t, pool_ls, pool_t = {R['gfc_ls']}, {R['gfc_t']}, {R['p126_ls']}, {R['p126_t']}\n"
            "    per = [(2009, 22.7), (2020, -3.0), (2022, -8.0), (2025, 1.0)]\n"
            "    banner = 'FROZEN illustrative split (offline)'\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "yrs = [str(y) for y, _ in per]; vals = [v for _, v in per]\n"
            "cols = [AMBER if v > 0 else GREY for v in vals]\n"
            "ax.bar(yrs, vals, color=cols, width=.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axhline(pool_ls, ls='--', c=RED, label=f'pooled avg {pool_ls:+.1f}%')\n"
            "ax.set_ylabel('fast − slow, 6-month (%)'); ax.set_title('Per-crash spread: one big winner, the rest noise')\n"
            "ax.legend(); ax.text(0.99, 0.02, banner, transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'GFC alone: {gfc_ls:+.1f}% (t={gfc_t:+.2f})  vs pooled {pool_ls:+.1f}% (t={pool_t:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: the GFC contributes +{R['gfc_ls']:.0f}% almost single-handedly "
            f"(*t* = +{R['gfc_t']:.2f} on its own). It is the deepest drawdown — the one with the most "
            "cross-sectional dispersion to mine — so it dominates any naive average. Reporting it "
            "alone is the snoop. **H₂ is the whole story.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — pooled market-neutral spread +{R['p126_ls']:.1f}% at 126d, "
            f"*t* = +{R['p126_t']:.2f}, CI [{R['p126_lo']:.0f}%, +{R['p126_hi']:.0f}%]; "
            f"+{R['p63_t']:.2f}/+{R['p252_t']:.2f} at 63/252d. Below the bar everywhere.\n"
            f"- **Tradability `MIRAGE`** — no gross edge; net at 20 bps + borrow +{R['cost20']:.1f}%.\n"
            f"- **A fast-recoverers rule? `CHERRY--PICKED`** — GFC alone +{R['gfc_ls']:.0f}% "
            f"(*t* = +{R['gfc_t']:.2f}); the other three crashes erase it, on a survivor universe "
            "biased *for* the claim."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the (irrelevant) cost sweep\n\n"
            "For form's sake: net pooled spread vs cost. Since the gross is inside the noise, this "
            "just confirms there is nothing to harvest."
        ),
        code(
            "costs = [0, 5, 10, 20]\n"
            "if HAVE_REAL:\n"
            "    panel, mkt, _ = real_panel(); evs = st.multi_events(mkt)\n"
            "    net = [st.pooled_spread(panel, mkt, evs, fwd_days=126, cost_bps=c, borrow_bps_ann=50.0)['ls_net']*100 for c in costs]\n"
            "    banner = 'REAL tape (pooled)'\n"
            "else:\n"
            f"    net = [{R['cost0']}, {R['cost5']}, {R['cost10']}, {R['cost20']}]\n"
            "    banner = 'FROZEN docs/results.md numbers (offline)'\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=RED, lw=2); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps, one-way × NAV, both legs) + 50 bps/yr borrow')\n"
            "ax.set_ylabel('net 6-month spread (%)'); ax.set_ylim(min(0, min(net)-0.5), max(net)+1)\n"
            "ax.set_title('A non-edge, gently nudged toward zero by costs')\n"
            "ax.text(0.99, 0.95, banner, transform=ax.transAxes, ha='right', va='top', fontsize=8, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net by cost:', [round(x,2) for x in net])"
        ),
        md(
            "> 💡 In plain words: when the gross edge is already statistically zero, the cost axis is "
            "a formality. This is a *mirage*, not a *fragile* edge — there is no there there."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print zero? Plant a tunable "
            "recovery-momentum edge on a synthetic panel and sweep its strength against the "
            "shuffle-the-scores null."
        ),
        code(
            "edges = [0.0, 0.0005, 0.0010, 0.0015, 0.0020]\n"
            "ls, ts = [], []\n"
            "for e in edges:\n"
            "    p, m, _ = data.synthetic_panel(n_names=80, n_days=1500, recovery_edge=e, seed=333)\n"
            "    ev = st.detect_drawdown(m, min_depth=0.10)\n"
            "    r = st.run_event(p, m, ev)\n"
            "    ls.append(r['ls_gross']*100); ts.append(st.spread_tstat(r))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.plot(edges, ls, 'o-', c=GREEN, lw=2); ax.axhline(0, c='k', lw=1)\n"
            "ax2 = ax.twinx(); ax2.plot(edges, ts, 's--', c=GREY, lw=1.4); ax2.axhline(2, ls=':', c=GREY)\n"
            "ax.set_xlabel('planted recovery_edge'); ax.set_ylabel('fast − slow spread (%)', color=GREEN)\n"
            "ax2.set_ylabel('spread t', color=GREY)\n"
            "ax.set_title('SYNTHETIC: the engine banks a real edge monotonically — it is faithful')\n"
            "ax.text(0.01, 0.95, 'SYNTHETIC control tape (offline, deterministic)',\n"
            "        transform=ax.transAxes, ha='left', va='top', fontsize=8, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('spread% by edge:', [round(x,1) for x in ls]); print('t by edge:', [round(x,2) for x in ts])"
        ),
        md(
            f"The spread and its *t* rise monotonically with the planted edge — ~0 at the null "
            f"(*t* = +{R['syn_null_t']:.2f}), clearing the bar at a real edge (*t* = "
            f"+{R['syn_ctrl_t']:.2f}, shuffle-*p* = {R['syn_ctrl_p']:.3f}). So the engine is honest, "
            "and the flat real-tape result is a statement about the market: **recovery speed carries "
            "no forward information once you stop cherry-picking the crash.**\n\n"
            "Open forks: a point-in-time universe *including the delisted* (expected to look worse), "
            "and an orthogonalisation against plain cross-sectional momentum (a fast recoverer is "
            "just a recent winner)."
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
