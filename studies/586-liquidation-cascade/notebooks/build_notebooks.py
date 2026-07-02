"""Generate the two narrative notebooks for Study 586 (Liquidation-Cascade).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). Every code cell runs
fully offline and deterministic on the synthetic tape in ``liquidation_cascade/data.py`` — there is
no real forced-liquidation feed (paid product), so ``HAVE_REAL`` is False in the reproducible core
and the notebooks compute the headline numbers live from the synthetic **null** world. The frozen
dict ``R`` mirrors docs/results.md so the prose and the recomputed figures agree.
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30; synthetic NULL world,
# 1,500 days, seed 586, bounce_alpha=0, horizon=5, q=.95; frame fp b3d7960bf851).
R = dict(
    fp_frame="b3d7960bf851", n=1500, n_event=74, event_pct=4.93,
    horizon=5, q=0.95,
    event_mean=-1.20, baseline_mean=-0.29, gap=-0.96, t=-0.92, placebo_p=0.34,
    net_gap=-1.16, cost_bps=10.0,
    best_gap=2.92, best_net=2.72, best_t=1.27, best_nevent=15,  # h=5 q=.99 curve-fit cut
    # robustness sweep rows: (horizon, q, gap_pct, t, n_event)
    sweep=[
        (1, 0.90, -0.40, -1.06, 139), (1, 0.95, -0.64, -1.30, 74), (1, 0.99, 0.21, 0.19, 15),
        (3, 0.90, -0.35, -0.57, 139), (3, 0.95, 0.09, 0.12, 74), (3, 0.99, 0.96, 0.61, 15),
        (5, 0.90, -1.10, -1.37, 139), (5, 0.95, -0.96, -0.92, 74), (5, 0.99, 2.92, 1.27, 15),
        (10, 0.90, -1.45, -1.38, 135), (10, 0.95, -0.70, -0.55, 72), (10, 0.99, -0.32, -0.12, 15),
        (20, 0.90, -1.69, -1.09, 134), (20, 0.95, -0.94, -0.43, 71), (20, 0.99, -1.48, -0.39, 14),
    ],
    # synthetic control: (bounce_alpha, mean event-t over 25 seeds)
    ctrl=[(0.000, 0.13), (0.005, 1.05), (0.010, 1.98), (0.015, 2.89), (0.020, 3.74), (0.030, 5.24)],
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

from liquidation_cascade import data, strategy as st

HORIZON, Q, SEED = 5, 0.95, 586

# The reproducible core is SYNTHETIC: there is no free forced-liquidation feed.
REAL = data.fetch_series(fetch=False)          # empty by design (no free tape)
HAVE_REAL = not REAL.empty

# The headline synthetic NULL world (bounce_alpha = 0 => liquidations carry no forward info).
FRAME_NULL, TRUTH = data.synthetic_series(bounce_alpha=0.0, horizon=HORIZON, seed=SEED)
print("real liquidation tape present:", HAVE_REAL, "(none is free — study is synthetic-only)")
print("synthetic null world:", len(FRAME_NULL), "days, frame fp",
      data.fingerprint(FRAME_NULL[["ret", "liq"]]))
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Liquidation Cascades — does crypto bounce after the blood? 💥\n"
            "### When a wave of forced sell-offs hits, do you buy the dip — or catch a falling knife?\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "Here's a story every crypto trader has heard. When the market drops hard, exchanges "
            "**forcibly close** the accounts of traders who borrowed too much — a *liquidation "
            "cascade*. Millions of dollars of positions get dumped at market, all at once, whether "
            "the seller likes it or not. The folklore says: that selling is *mechanical*, not "
            "informed — it's panic and margin calls, not people who know something — so the price "
            "overshoots to the downside and then **snaps back**. A big liquidation wave, the story "
            "goes, is a **capitulation bottom**: buy the blood.\n\n"
            "So does it work? We'd love to check it against real liquidation data. There's a catch — "
            "and it decides the whole study.\n\n"
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
            "| Can we test 'buy the blood' on **real** data? | **No — not for free.** The per-day "
            "forced-liquidation numbers you'd need (Coinglass, Amberdata) are a *paid* product with "
            "no usable free history. So we can't earn a `REAL` verdict here. |\n"
            "| What does the honest fallback show? | A **synthetic null world** — a realistic BTC-like "
            "tape where, by construction, liquidations carry *no* forward information. The engine "
            "should find **nothing**. It does. |\n"
            f"| Is there a bounce in that null world? | **No.** After a big-liquidation day, the next "
            f"5 days returned **{R['event_mean']}%** vs the everyday baseline **{R['baseline_mean']}%** — "
            f"a gap of **{R['gap']}%** the *wrong* way, and a shuffle test can't tell it from noise "
            f"(*p* = {R['placebo_p']}). |\n"
            "| Does the engine even *work*? | **Yes** — when we *plant* a real bounce, it lights up "
            "past the significance bar. So the null is honest, not a broken detector. |\n\n"
            "> The story is plausible (fire-sale selling really does overshoot in other markets). But "
            "the data to test it isn't free, and in a fair synthetic null there's no bounce to find."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"A big wave of forced liquidations is a capitulation bottom — the mechanical selling "
            "overshoots, so crypto bounces. Buy the blood.\"*\n\n"
            "Why it *sounds* smart: when the market falls, leveraged longs get **margin-called** and "
            "the exchange closes them at market — forced sellers who *have* to sell, right now, at "
            "any price. That's not people with bad news; it's plumbing. Classic finance says "
            "forced, liquidity-driven selling pushes price *below* fair value and then reverses "
            "(Shleifer-Vishny fire sales; Coval-Stafford's forced-selling *reversal*). So the "
            "liquidation spike should mark the low."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If it's true, it's a clean, mechanical buy signal — no fundamentals, no news to read, "
            "just *watch the liquidation ticker and buy the spike*. That's exactly the kind of "
            "simple, seductive rule that ends up on every crypto-Twitter thread. The desk has poked "
            "at crypto from the calendar side ([133 seasonality](../../133-crypto-seasonality/), "
            "[175 weekend](../../175-crypto-weekend/)) and the price side "
            "([210 trend](../../210-crypto-trend/), [251 reversal](../../251-crypto-reversal/)); "
            "here we go at the **forced-liquidation flow** itself."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Flag the cascades.** A day is a *liquidation event* if that day's forced-liquidation "
            "flow is in the top slice (say top 5%) of the recent past.\n"
            "2. **Look forward, not back.** Measure the return over the *next* few days — *after* the "
            "event day. (The same-day crash is the cascade itself; the *bounce* is what happens "
            "next.)\n"
            "3. **Compare to normal.** Did the days after a cascade beat an ordinary day? A real "
            "bounce means yes, and by enough that a coin-flip shuffle can't fake it.\n\n"
            "**The catch that decides everything:** step 1 needs a *forced-liquidation series*, and "
            "there's no free one with real history. So we do the honest thing — build a realistic "
            "synthetic tape where we *control* whether a bounce exists, and show (a) the engine, and "
            "(b) that a fair **null** (no bounce planted) shows nothing.\n\n"
            "*Timing:* we act at the event day's close (we know the day's liquidations by then) and "
            "hold the next few days. No peeking at the future."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a liquidation cascade even look like? Here's the synthetic tape: price "
            "on top, forced-liquidation flow below. Notice the liquidation spikes land on the *red* "
            "days — that's the plumbing (down move → margin calls)."
        ),
        code(
            "f = FRAME_NULL\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,\n"
            "                             gridspec_kw={'height_ratios':[2,1]})\n"
            "a1.plot(f.index, f['price'], color=GREY, lw=1.1); a1.set_ylabel('price (synthetic)')\n"
            "a1.set_title('A synthetic BTC-style tape with a forced-liquidation channel')\n"
            "ev = st.event_flag(f['liq'], q=Q)\n"
            "a2.fill_between(f.index, 0, f['liq']/1e6, color=GREY, lw=0, alpha=.6)\n"
            "a2.scatter(f.index[ev], (f['liq']/1e6)[ev], color=RED, s=14, zorder=3,\n"
            "           label=f'liquidation events (top {int((1-Q)*100)}%)')\n"
            "a2.set_ylabel('liq flow ($M)'); a2.legend(loc='upper left')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('liquidation-event days:', int(ev.sum()), 'of', len(ev),\n"
            "      f'({100*ev.mean():.1f}%)')"
        ),
        md(
            "**Now the question: do the days *after* a cascade bounce?** Compare the 5-day forward "
            "return on event days vs every day."
        ),
        code(
            "es = st.event_study(FRAME_NULL, horizon=HORIZON, q=Q)\n"
            "em, bm, gap, t = es['event_mean']*100, es['baseline_mean']*100, es['gap']*100, es['t']\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['after a\\nCASCADE', 'a NORMAL\\nday'], [em, bm], color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('next-5-day return %')\n"
            "ax.set_title(f'The folklore says the red bar should be HIGHER. It isn\\'t (gap {gap:+.2f}%)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'after cascade {em:+.2f}%  |  normal day {bm:+.2f}%  ->  gap {gap:+.2f}% (t {t:+.2f})')"
        ),
        md(
            f"There's no bounce. In this fair null world the days *after* a liquidation cascade "
            f"returned **{R['event_mean']}%** vs a normal-day baseline of **{R['baseline_mean']}%** — "
            f"a gap of **{R['gap']}%**, if anything the *wrong* way, and well inside the noise. Exactly "
            "what an honest detector should say when there's nothing to find."
        ),
        md(
            "**Maybe it works at a different horizon or a bigger cutoff?** Sweep them — and watch the "
            "sign refuse to settle down."
        ),
        code(
            "sw = st.robustness_sweep(FRAME_NULL, horizons=(1,3,5,10,20), quantiles=(0.90,0.95,0.99))\n"
            "piv = sw.pivot(index='horizon', columns='q', values='gap_pct')\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "im = ax.imshow(piv.values, cmap='RdYlGn', aspect='auto', vmin=-3, vmax=3)\n"
            "ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f'top {int((1-q)*100)}%' for q in piv.columns])\n"
            "ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)\n"
            "ax.set_xlabel('event cutoff'); ax.set_ylabel('hold (days)')\n"
            "for i in range(piv.shape[0]):\n"
            "    for j in range(piv.shape[1]):\n"
            "        ax.text(j, i, f'{piv.values[i,j]:+.1f}', ha='center', va='center', fontsize=9)\n"
            "ax.set_title('Post-cascade gap % (green=bounce, red=bleed) — no stable sign')\n"
            "fig.colorbar(im, ax=ax, shrink=.8, label='gap %'); plt.tight_layout(); plt.show()\n"
            "print('sign flips across the grid; every cell is |t|<2')"
        ),
        md(
            "The gap flips between green (bounce) and red (bleed) with no pattern, and every cell is "
            "statistically weak. A 'signal' whose direction depends on which horizon and cutoff you "
            "pick is not a signal."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** We can't test it on real data for free, and in a fair synthetic null "
            f"there's no bounce (gap {R['gap']}%, *t* {R['t']}, shuffle *p* {R['placebo_p']}), with a "
            "sign that flips across horizons and cutoffs.\n"
            "- **Tradability — Mirage.** Unreachable data, a wrong-signed gap, and a crypto round-trip "
            "cost on top. Nothing to harvest."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Not on a free stack — you can't even *see* the signal without a paid liquidation feed. "
            "And even granting the data, the null world already shows the post-cascade move has no "
            "stable sign, so you'd be paying crypto round-trip costs to trade noise. The one cut that "
            "looks like a bounce (a long horizon at the most extreme 1% cutoff) rests on a *handful* "
            "of events with a *t* well under 2 — the signature of a curve-fit, not an edge."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Point it at real data.** If you have a Coinglass/Amberdata liquidation feed, stage a "
            "daily `['ret','liq']` frame at `_cache/liquidation_series.parquet` and the *same* engine "
            "runs on the real tape — that's the seam left open in `data.py`.\n"
            "- **The cousins.** [251 crypto-reversal](../../251-crypto-reversal/) and "
            "[210 crypto-trend](../../210-crypto-trend/) are the price-only versions of 'does crypto "
            "snap back or keep going?' — useful priors for which way a real bounce would lean.\n\n"
            "*Think the bounce is real? Fork this, drop in a real liquidation series, and show the "
            "post-cascade gap clearing t = 2 the right way across horizons — the positive control "
            "shows exactly what that would look like.*"
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
            "# Liquidation Cascades — a quantitative teardown 🔬\n"
            "### forced-liquidation event flag · forward-return event study · two-sample *t* · label-shuffle placebo · horizon × threshold sweep · costs · seed-robust synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We flag large "
            "forced-liquidation days, measure the *forward* return after them, and test whether it "
            "beats the unconditional baseline (the capitulation-bounce) — on a synthetic tape, "
            "because no free real forced-liquidation series exists.\n\n"
            "> ⚠️ **Not investment advice.** This study is **synthetic-only**: the per-day "
            "forced-liquidation USD series the claim needs is a paid product (Coinglass/Amberdata) "
            "with no usable free history, so a `REAL` stamp (robust *t* ≥ 2 on a **real** tape) is "
            f"unreachable and the study is capped at `WEAK`/`NONE`. Headline = the synthetic **null** "
            f"world (1,500 days, frame fp `{R['fp_frame']}`, as-of 2026-06-30). Methods in "
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
            f"| **Signal** | `NONE` | No free real tape → cannot earn `REAL`. Null world: event−baseline "
            f"gap **{R['gap']}%** (two-sample *t* **{R['t']}**, placebo *p* {R['placebo_p']}); sign "
            "flips across the horizon × threshold grid. |\n"
            f"| **Tradability** | `MIRAGE` | Unreachable paid feed; wrong-signed headline gap; even the "
            f"best cut (h=5, top 1%, {R['best_nevent']} events, *t* {R['best_t']}) is a curve-fit; "
            f"net of a {2*R['cost_bps']:.0f} bps round-trip. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), but there is no "
            "free data to point it at, and a fair null shows no bounce and no stable sign."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $L_t$ be day $t$'s forced-liquidation flow, $E_t = \\mathbb{1}[L_t \\ge Q_q(\\text{trailing } L)]$ "
            "the *event* flag (top $1-q$ tail), and $r^{(H)}_{t} = \\prod_{k=1}^{H}(1+r_{t+k})-1$ the "
            "return over the *next* $H$ days (strictly after $t$).\n\n"
            "- **H₁ (the bounce).** $\\mathbb{E}[r^{(H)} \\mid E{=}1] - \\mathbb{E}[r^{(H)}] > 0$ at "
            "*t* ≥ 2 (post-cascade returns beat the baseline).\n"
            "- **H₂ (robustness).** The sign of H₁ is stable across $H$ and $q$.\n"
            "- **H₃ (net).** H₁ survives the round-trip cost.\n"
            "- **H₀ (data).** All of the above require a *real* $L_t$ series. **None is free.**\n\n"
            "We find **H₀ binding** (synthetic-only), **H₁ not met** in the null world (gap wrong "
            "sign, *t* < 2), **H₂ rejected** (sign flips), **H₃ moot**. The positive control shows the "
            "engine *would* reject H₀-conditional-on-data if a bounce were truly there."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why**. The mechanism is real elsewhere: forced, liquidity-motivated "
            "selling overshoots and *reverses* in equities (Coval-Stafford 2007), and reversal "
            "premia spike when liquidity evaporates (Nagel 2012) — precisely the regime a "
            "liquidation-event filter selects. So the *prior* for a crypto capitulation-bounce is not "
            "silly. What kills the study is **H₀**: the conditioning variable (aggregate "
            "forced-liquidation USD) is a paid data product, so on a no-key stack the hypothesis is "
            "**untestable on real data**, and the honest output is the machinery + a null."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Event flag.** $E_t = \\mathbb{1}[L_t \\ge Q_q(\\text{trailing 252d})]$ — a "
            "*point-in-time* trailing quantile (no future levels inform the threshold).\n"
            "- **Forward return.** $r^{(H)}_t$ over days $t{+}1..t{+}H$ — the same-day cascade is "
            "excluded by construction (that crash is the event, not the bounce).\n"
            "- **Event study.** Welch two-sample *t* of event vs non-event forward returns.\n"
            "- **Placebo.** Shuffle the event labels against the fixed forward returns (2,000 perms).\n"
            "- **Robustness.** $H \\in \\{1,3,5,10,20\\}$ × $q \\in \\{.90,.95,.99\\}$; read the sign.\n"
            "- **Costs.** One entry + one exit per event, each × NAV (a long — no borrow).\n"
            "- **Positive control.** A synthetic world with a planted bounce (`bounce_alpha > 0`) and "
            "a null — averaged over 25 seeds (house rule).\n\n"
            "Timing: the event is known at the event day's close ($L_t$, $r_t$ complete), so we enter "
            "at that close and hold days $t{+}1..t{+}H$. No future data enters the flag or the return "
            "(no look-ahead). The same-close entry is the single documented convention; the sweep and "
            "the control use it identically."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The event study — H₁\n\n"
            "Event vs non-event forward returns, with the two-sample *t* and the placebo null."
        ),
        code(
            "es = st.event_study(FRAME_NULL, horizon=HORIZON, q=Q)\n"
            "p = st.placebo_pvalue(FRAME_NULL, horizon=HORIZON, q=Q, n_perm=2000, seed=586)\n"
            "em, bm, gap, t = es['event_mean']*100, es['baseline_mean']*100, es['gap']*100, es['t']\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['EVENT\\n(post-cascade)', 'BASELINE\\n(all days)'], [em, bm], color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel(f'next-{HORIZON}-day return %')\n"
            "ax.set_title(f'Event - baseline = {gap:+.2f}%  (two-sample t {t:+.2f}, placebo p {p:.3f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'event {em:+.2f}% | baseline {bm:+.2f}% | gap {gap:+.2f}% | t {t:+.2f} | placebo p {p:.3f}'\n"
            "      f' | n_event {es[\"n_event\"]}')"
        ),
        md(
            f"> 💡 In plain words: H₁ **not met** — the post-cascade gap is **{R['gap']}%** (*t* "
            f"{R['t']}), the *wrong* sign, and the placebo *p* = {R['placebo_p']} says it's "
            "indistinguishable from noise. No capitulation-bounce in the null world."
        ),
        md(
            "### 4b · Robustness — H₂ (does the sign hold?)\n\n"
            "The event−baseline gap and its *t* across every horizon × threshold."
        ),
        code(
            "sw = st.robustness_sweep(FRAME_NULL, horizons=(1,3,5,10,20), quantiles=(0.90,0.95,0.99))\n"
            "sw['q'] = sw['q'].map(lambda x: f'top {int((1-x)*100)}%')\n"
            "print(sw.round(3).to_string(index=False))\n"
            "piv = sw.pivot(index='horizon', columns='q', values='gap_pct')\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "im = ax.imshow(piv.values, cmap='RdYlGn', aspect='auto', vmin=-3, vmax=3)\n"
            "ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)\n"
            "ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)\n"
            "ax.set_xlabel('event cutoff'); ax.set_ylabel('hold (days)')\n"
            "for i in range(piv.shape[0]):\n"
            "    for j in range(piv.shape[1]):\n"
            "        ax.text(j, i, f'{piv.values[i,j]:+.1f}', ha='center', va='center', fontsize=9)\n"
            "ax.set_title('Post-cascade gap % — sign flips, no cell clears |t|=2'); \n"
            "fig.colorbar(im, ax=ax, shrink=.8, label='gap %'); plt.tight_layout(); plt.show()"
        ),
        md(
            "> 💡 In plain words: H₂ **rejected.** The gap flips sign across the grid and *no* cell "
            "reaches |*t*| ≥ 2. The lone strongly-positive cut (h=5, top 1%) rests on "
            f"{R['best_nevent']} events at *t* {R['best_t']} — a curve-fit, not an edge."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — H₀ binding (no free real tape → no `REAL`); H₁ not met in the null "
            f"(gap {R['gap']}%, *t* {R['t']}, wrong sign); H₂ rejected (sign flips, no |t| ≥ 2).\n"
            f"- **Tradability `MIRAGE`** — unreachable paid feed; wrong-signed headline gap; the best "
            f"cut is a {R['best_nevent']}-event curve-fit; net of a {2*R['cost_bps']:.0f} bps "
            "round-trip there is nothing to harvest."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs\n\n"
            "The headline gap is the wrong sign before costs; even the single positive cut barely "
            "survives the round-trip and rests on a handful of events."
        ),
        code(
            "es = st.event_study(FRAME_NULL, horizon=HORIZON, q=Q)\n"
            "gap = es['gap']*100; net = st.net_gap(es['gap'], cost_bps=10.0)*100\n"
            "# the 'best' curve-fit cut for contrast\n"
            "esb = st.event_study(FRAME_NULL, horizon=5, q=0.99)\n"
            "bgap = esb['gap']*100; bnet = st.net_gap(esb['gap'], cost_bps=10.0)*100\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "x = np.arange(2); w = .35\n"
            "ax.bar(x-w/2, [gap, bgap], w, label='gross', color=RED)\n"
            "ax.bar(x+w/2, [net, bnet], w, label='net (20 bps r/t)', color=GREY)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['headline\\n(h=5, top 5%)', f'best cut\\n(h=5, top 1%, {esb[\"n_event\"]} ev)'])\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('event - baseline gap %'); ax.legend()\n"
            "ax.set_title('Wrong sign on the headline; the one positive cut is a handful of events')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'headline gross {gap:+.2f}% -> net {net:+.2f}% | best-cut gross {bgap:+.2f}% -> net {bnet:+.2f}% ({esb[\"n_event\"]} events)')"
        ),
        md(
            "> 💡 In plain words: nothing to charge costs against on the headline (it *loses* before "
            "frictions), and the positive cut is too thin to bank. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a real bounce "
            "(`bounce_alpha > 0`) of increasing strength and watch the event-study *t* rise — "
            "averaged over 25 seeds so no single lucky seed can fake it."
        ),
        code(
            "alphas = [0.0, 0.005, 0.010, 0.015, 0.020, 0.030]\n"
            "ts = [st.synthetic_mean_t(data, bounce_alpha=a, horizon=HORIZON, q=Q, n_seeds=25) for a in alphas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(alphas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = +2 bar (bounce)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted bounce_alpha (bigger = stronger capitulation-bounce)')\n"
            "ax.set_ylabel('mean event-study t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted bounce — flat at the null, past +2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for a, t in zip(alphas, ts): print(f'bounce_alpha {a:+.3f} -> mean event-t {t:+.2f}')"
        ),
        md(
            "The event-*t* is ≈ 0 at the null (no hallucinated bounce) and climbs past +2 as the "
            "planted bounce grows — a faithful detector. So the null-world result is a statement about "
            "*a world where liquidations carry no forward information*, and the real-data verdict is "
            "gated entirely by **H₀**: no free forced-liquidation tape exists, so the honest stamp is "
            "`NONE` × `MIRAGE`. Point the engine at a paid Coinglass/Amberdata feed (the seam in "
            "`data.py`) and it will tell you, honestly, whether the bounce is there."
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
