"""Generate the two narrative notebooks for Study 561 (ETF-Flow-Momentum).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). This is a
**synthetic-only** study — there is no real ETF-flow tape (see docs/results.md), so ``HAVE_REAL``
is always False and every figure is drawn *live* from the deterministic synthetic generator in
``etf_flow_momentum/data.py``. The dict ``R`` below mirrors the frozen headline numbers in
docs/results.md and is used for the verdict tables and prose; the code cells recompute the
synthetic numbers themselves so the notebook is fully reproducible offline.
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


# Frozen synthetic headline numbers — mirror of docs/results.md (as-of 2026-06-30; seed 561;
# 16 ETFs x 120 months; null fp 9cd505b99a53, planted-edge fp dca0e2e0ba52).
R = dict(
    fp_null="9cd505b99a53", fp_edge="dca0e2e0ba52", n_etfs=16, n_months=120,
    # null world (flow_alpha = 0) — the honest baseline
    null_ann=2.88, null_t=0.78, null_hit=51.7, null_placebo=0.418, null_slope_t=0.32, null_corr=0.008,
    # planted-edge illustration (flow_alpha = 0.005)
    edge_alpha=0.005, edge_ann=15.99, edge_t=4.35, edge_hit=64.2, edge_placebo=0.0005,
    edge_slope_t=4.21, edge_corr=0.11, edge_net=14.15, cost_bps=3.0, borrow_bps=40.0,
    # robustness (planted world): (frac, ann%, t)
    rob=[(0.25, 17.1, 4.04), (0.33, 16.0, 4.35), (0.50, 8.8, 3.39)],
    # seed-robust control (25 seeds): (flow_alpha, mean spread-t)
    ctrl=[(-0.008, -5.87), (-0.005, -3.57), (0.0, 0.25), (0.003, 2.54), (0.005, 4.07), (0.008, 6.36)],
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

from etf_flow_momentum import data, strategy as st

# Synthetic-only study: there is no real ETF-flow tape (yfinance exposes one stale
# shares-outstanding scalar, not a daily history). Every figure below is drawn LIVE from the
# deterministic synthetic generator, seed 561.
HAVE_REAL = not data.fetch_panel(fetch=False).shape[0] > 0
HAVE_REAL = False  # explicit: no real tape exists for this study
print("real ETF-flow tape present:", HAVE_REAL, "(synthetic-only study — the free data does not exist)")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# ETF-Flow-Momentum — do the ETFs pulling in the most money keep winning? 💸\n"
            "### Or is chasing the hot sector fund a trap? — in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "When a sector ETF gets popular, new money pours in: the fund *creates* new shares to "
            "meet demand, so its assets swell. The **flow-momentum** idea says those swelling, "
            "in-favour funds keep leading — money flows to winners and winners keep winning. The "
            "opposite idea — call it the **trap** — says heavy inflows mean the sector is crowded and "
            "over-extended, so chasing it is exactly how you buy the top.\n\n"
            "Both stories have serious academic backing, and they point in *opposite directions*. So "
            "which is it? Here's the catch that shapes this whole study: **you can't cheaply measure "
            "ETF flows.** The clean daily 'how many shares does this fund have today' number is a paid "
            "data feed; the free stuff gives you one stale figure. So we can't test this on the real "
            "tape at all — we build a realistic *synthetic* world instead, and see what an honest test "
            "would say.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "control? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it, on a deterministic synthetic panel. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do the biggest-inflow ETFs keep winning? | **Maybe — the textbooks genuinely disagree.** "
            "One camp says flows persist (winners keep winning); an equally serious camp says flows "
            "*reverse* (crowding trap). The sign is disputed. |\n"
            "| Can we settle it on real data? | **No.** Honest ETF flows need a paid daily "
            "shares-outstanding feed. Free data (yfinance) gives one stale number. So this is a "
            "**synthetic-only** study — it can never be certified `REAL`. |\n"
            f"| What does an honest *no-edge* world look like? | A **coin.** Spread +{R['null_ann']}%/yr, "
            f"*t* +{R['null_t']}, wins {R['null_hit']}% of months, shuffle-test *p* {R['null_placebo']}. "
            "Nothing there. |\n"
            "| Does the engine work? | **Yes** — plant a real edge (either sign) and it catches it; "
            "leave it at zero and it stays flat. (See beat 7.) |\n\n"
            "> So: `Weak` on signal (the literature says *maybe*, the sign is contested, and there is "
            "no real tape to certify it), `Mirage` on tradability (you can't even cheaply *measure* "
            "the signal, let alone trade it)."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"Sector ETFs pulling in the most new money keep outperforming.\"* — flow momentum "
            "(Ben-Rephael, Kandel & Wohl 2012; Clifford et al. 2014).\n\n"
            "The intuition: money is smart-ish and sticky. When investors pile into, say, a "
            "semiconductor ETF, that buying pressure lifts the underlying stocks, momentum feeds on "
            "itself, and the fund keeps leading. The mirror claim — the **trap** — is just as famous "
            "(Frazzini & Lamont's *\"Dumb Money\"* 2008; Ben-David, Franzoni & Moussawi 2018): retail "
            "flows chase the top, non-fundamental demand mean-reverts, and the crowded fund gives it "
            "back. Same signal, opposite sign."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If flows *persist*, there's a dead-simple sector-rotation trade: each month, buy the ETFs "
            "with the biggest inflows, avoid (or short) the ones bleeding money. If flows *reverse*, "
            "the exact same trade is a slow-motion way to buy high and sell low. Getting the sign "
            "wrong doesn't cost you nothing — it costs you *double*. That's why 'the sign is disputed' "
            "isn't a footnote; it's the whole ballgame."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Measure flow.** Each month, for each ETF: how much new money came in (net "
            "creation-unit inflows, as a % of assets)? Positive = money flooding in; negative = "
            "money leaving.\n"
            "2. **Sort.** Split the ETFs into a *high-inflow* third and a *low-inflow* third.\n"
            "3. **Wait one month.** Did the high-inflow third beat the low-inflow third next month? "
            "The claim says yes; the trap says no.\n\n"
            "The problem lives in step 1. That clean daily flow number is a **paid** data field. The "
            "free version is one stale scalar — useless for a monthly panel. So we can't run this on "
            "the real tape without smuggling in bad data. Instead we build a *synthetic* ETF-flow "
            "world where we control the truth, and check what an honest test would find.\n\n"
            "*Timing:* the flow signal at month *t* uses only flows realised through month *t*; we "
            "enter at that month's close and hold the next month. No peeking ahead — one clean "
            "execution lag."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — the honest *no-edge* world\n\n"
            "First, the sanity check that matters most: if flows had **no** predictive power at all, "
            "what would our test show? Let's build exactly that world (the null) and sort it."
        ),
        code(
            "panel_null, _ = data.synthetic_panel(flow_alpha=0.0, seed=561)   # NO edge planted\n"
            "s = st.spread_stats(panel_null, frac=0.33)\n"
            "p = st.placebo_pvalue(panel_null, frac=0.33, n_perm=2000, seed=561)\n"
            "series = st.monthly_spread_series(panel_null, frac=0.33)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.2))\n"
            "ax.bar(series.index, series.values*100, color=[GREEN if v>0 else RED for v in series.values], width=1.0)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(s['mean_m']*100, c=GREY, ls='--', lw=1.5, label=f\"mean {s['mean_m']*100:+.2f}%/mo\")\n"
            "ax.set_xlabel('month'); ax.set_ylabel('inflow - outflow spread %/mo')\n"
            "ax.set_title(f\"NULL world: spread {s['ann']*100:+.1f}%/yr, t {s['t']:+.2f}, placebo p {p:.3f} — a coin\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"NULL: ann {s['ann']*100:+.2f}% | t {s['t']:+.2f} | hit {s['hit']*100:.1f}% | placebo p {p:.3f}\")"
        ),
        md(
            f"There's the honest baseline. With no real edge, the flow spread wobbles around zero: "
            f"**+{R['null_ann']}%/yr**, *t* **+{R['null_t']}**, it wins only **{R['null_hit']}%** of "
            f"months, and the shuffle test *p* = **{R['null_placebo']}** (about a coin). This is what "
            "'chasing flows does nothing' looks like — and with no real tape, we can't rule out that "
            "the real world is exactly this."
        ),
        md(
            "**But wait — could a real edge exist and we'd just miss it?** No. Watch: plant a small, "
            "realistic flow-momentum edge and re-run the same sort."
        ),
        code(
            "panel_edge, _ = data.synthetic_panel(flow_alpha=0.005, seed=561)  # modest edge planted\n"
            "s_null = st.spread_stats(panel_null, 0.33); s_edge = st.spread_stats(panel_edge, 0.33)\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "ax.bar(['NULL world\\n(no edge)', 'EDGE world\\n(modest flow momentum)'],\n"
            "       [s_null['ann']*100, s_edge['ann']*100], color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('inflow - outflow spread %/yr')\n"
            "ax.set_title('The engine is not blind: it sees a planted edge and stays flat without one')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"null {s_null['ann']*100:+.1f}%/yr (t {s_null['t']:+.2f})  vs  edge {s_edge['ann']*100:+.1f}%/yr (t {s_edge['t']:+.2f})\")"
        ),
        md(
            f"So the detector works: a planted edge shows up loud (**+{R['edge_ann']}%/yr**, *t* "
            f"**+{R['edge_t']}**), the null stays flat. The reason we still say `Weak`, not `Real`, is "
            "**not** a broken engine — it's that we have **no real ETF-flow tape** to point it at, and "
            "the literature can't even agree on the sign."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The claim has genuine academic support, but it fights an equally "
            "documented *reversal* (crowding) effect, and there is **no free real tape** to certify "
            "it. On the honest null the spread is a coin. Synthetic-only ⇒ capped below `Real`.\n"
            "- **Tradability — Mirage.** You can't cheaply *measure* ETF flows on a retail stack, so "
            "there is nothing to trade — regardless of which sign the truth turns out to be."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Not on free data. The whole trade hinges on a clean daily *shares-outstanding* series to "
            "compute flows — a paid vendor field. And even if you bought that feed, you'd be betting "
            "on a signal whose *sign* the academic literature can't agree on: flow persistence vs. "
            "crowding reversal. ETFs themselves are cheap and liquid (in our planted world, costs "
            f"barely dent it: **+{R['edge_ann']}%** gross → **+{R['edge_net']}%** net) — but cheap "
            "frictions don't help when you can't measure the signal in the first place. `Mirage`."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The control (beat 7 of the quant notebook).** We plant the edge at *both* signs — "
            "momentum **and** the reversal trap — and the engine catches each, staying flat at zero. "
            "Proof it's the *data* that's missing, not the test.\n"
            "- **The real test needs the paid feed.** Re-run this with a genuine daily "
            "shares-outstanding panel (Bloomberg / issuer files) across a full cycle, and settle the "
            "sign dispute on the tape.\n"
            "- **Cousins on the desk.** [378 ETF-NAV-Premium](../../378-etf-nav-premium/) trades the "
            "ETF *price-vs-NAV* gap; [507](../../507-cross-sectional-momentum/) and "
            "[518](../../518-time-series-momentum/) trade *return* momentum. This one is about "
            "**flow**.\n\n"
            "*Think flow momentum is real (or a real trap)? Fork this, wire in a paid daily "
            "shares-outstanding feed, and show the inflow-minus-outflow spread clearing t = 2 — in "
            "*either* direction — on the real tape.*"
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
            "# ETF-Flow-Momentum — a quantitative teardown 🔬\n"
            "### monthly flow sort · one-sample *t* on the spread · label-shuffle placebo · Fama-MacBeth pooled slope · leg-fraction sweep · costs & borrow · both-signs synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We build a synthetic ETF-flow "
            "panel whose knob `flow_alpha` plants either sign of the flow→forward-return relation, and "
            "test whether the highest-inflow ETFs earn the highest next-month returns (momentum) or "
            "the lowest (the crowding trap).\n\n"
            "> ⚠️ **Not investment advice. Synthetic-only study** — there is no free real tape for ETF "
            "creation-unit flows (yfinance exposes one stale shares-outstanding scalar), so per the "
            "desk rubric this is capped below `Real`. Every figure is drawn live from the "
            "deterministic generator (seed 561). Methods in "
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
            f"| **Signal** | `WEAK` | Null-world spread **+{R['null_ann']}%/yr**, one-sample *t* "
            f"**+{R['null_t']}**, placebo *p* **{R['null_placebo']}**, hit {R['null_hit']}% — a coin. "
            "Literature supports *a* flow effect but the *sign* is disputed; **no real tape** ⇒ can't "
            "certify `Real`. |\n"
            f"| **Tradability** | `MIRAGE` | Honest ETF flow needs a paid daily shares-outstanding "
            f"feed — unmeasurable on a retail stack. (Planted world: gross +{R['edge_ann']}% → net "
            f"+{R['edge_net']}% after {R['cost_bps']:.0f} bps/leg + {R['borrow_bps']:.0f} bps borrow — "
            "mild, but moot without the signal.) |\n\n"
            "> 💡 In plain words: the engine is a faithful detector (control below), but the evidence "
            "available is synthetic-only and the null is indistinguishable from noise — so the verdict "
            "is about the *data gap*, not a failed test."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $F_{i,t}$ be ETF $i$'s net creation-unit flow in month $t$ (fraction of assets) and "
            "$r_{i,t+1}$ its next-month return.\n\n"
            "- **H₁ (flow momentum / long-short).** Sorting on $F_t$, "
            "$\\mathbb{E}[r_{t+1}\\mid\\text{high flow}] - \\mathbb{E}[r_{t+1}\\mid\\text{low flow}] > 0$ "
            "at *t* ≥ 2 (inflows keep winning).\n"
            "- **H₁ʹ (the trap).** The same spread is **< 0** (crowding reversal).\n"
            "- **H₂ (firm-level).** The within-month slope of $r_{t+1}$ on $F_t$ is non-zero and "
            "sign-stable.\n"
            "- **H₃ (net).** Whatever the sign, it survives costs and the short-leg borrow.\n\n"
            "The literature splits on H₁ vs H₁ʹ. **We cannot test any of them on a real tape** (no "
            "flow data), so we (a) show the null is a coin, and (b) prove the engine would resolve "
            "H₁/H₁ʹ *if* the data existed (the both-signs control, beat 7)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on the sign\n\n"
            "The content is the **sign war**. **Persistence camp:** Ben-Rephael-Kandel-Wohl (2012) — "
            "flows are sticky and positively related to near-term returns. **Reversal camp:** "
            "Frazzini-Lamont (2008) *\"Dumb Money\"* and Ben-David-Franzoni-Moussawi (2018) — "
            "flow-driven non-fundamental demand mean-reverts. A rotation trade built on the wrong sign "
            "loses *twice*. And underneath both sits the measurement wall: honest flow = daily "
            "Δ(shares outstanding) × NAV, a paid field. That is why this study is synthetic-only — the "
            "same data-availability convention as [273 lego-returns](../../273-lego-returns/) and "
            "[276 sneaker-resale](../../276-sneaker-resale/)."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** Each month, the cross-section of net flows $F_{i,t}$ (fraction of assets).\n"
            "- **Sort.** Tercile tails (≈5 ETFs each): inflow leg (top $F$) long, outflow leg (bottom "
            "$F$) short, equal-weight, dollar-neutral, hold one month.\n"
            "- **Headline.** A **one-sample *t*** on the monthly long-short spread series.\n"
            "- **Placebo.** Shuffle the flow labels against forward returns *within each month* "
            "(2000 perms); read the mean spread's tail probability.\n"
            "- **Firm-level.** A Fama-MacBeth-style pooled slope of $r_{t+1}$ on within-month "
            "$z(F_t)$, with *t* from the per-month slope estimates (months = independent units).\n"
            "- **Robustness.** Re-run at leg fractions 25% / 33% / 50%.\n"
            "- **Frictions.** Monthly-rebalance one-way cost per leg + an annual short borrow (ETFs "
            "are cheap/liquid, so both are modest but honest).\n"
            "- **Positive control.** A deterministic world with a planted edge of **either** sign, "
            "averaged over 25 seeds (house rule ≥ 20).\n\n"
            "Timing: the signal at month $t$ uses flows through $t$ only; enter at the $t$-close, hold "
            "$t{+}1$. One execution lag, no look-ahead, no same-bar fill."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The sort on the null — H₁\n\n"
            "Inflow vs outflow forward returns with no planted edge: the one-sample *t* and the "
            "placebo null."
        ),
        code(
            "panel_null, _ = data.synthetic_panel(flow_alpha=0.0, seed=561)\n"
            "s = st.spread_stats(panel_null, 0.33)\n"
            "p = st.placebo_pvalue(panel_null, 0.33, n_perm=2000, seed=561)\n"
            "series = st.monthly_spread_series(panel_null, 0.33)\n"
            "fig, ax = plt.subplots(1, 2, figsize=(11, 4.1))\n"
            "ax[0].hist(series.values*100, bins=20, color=GREY, alpha=.85)\n"
            "ax[0].axvline(0, c='k', lw=1); ax[0].axvline(s['mean_m']*100, c=RED, ls='--', lw=1.5)\n"
            "ax[0].set_xlabel('monthly spread %'); ax[0].set_ylabel('count')\n"
            "ax[0].set_title(f\"mean {s['mean_m']*100:+.2f}%/mo — centred on ~0\")\n"
            "cum = (1+series).cumprod()\n"
            "ax[1].plot(cum.index, cum.values, c=GREY, lw=1.6)\n"
            "ax[1].axhline(1, c='k', lw=1); ax[1].set_xlabel('month'); ax[1].set_ylabel('growth of $1')\n"
            "ax[1].set_title(f\"null spread book: t {s['t']:+.2f}, placebo p {p:.3f}\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"NULL ann {s['ann']*100:+.2f}% | t {s['t']:+.2f} | hit {s['hit']*100:.1f}% | placebo p {p:.3f} | n_months {s['n_months']}\")"
        ),
        md(
            f"> 💡 In plain words: on the null, H₁ is **not rejected** because there's nothing to "
            f"reject — spread **+{R['null_ann']}%/yr**, *t* **+{R['null_t']}**, placebo *p* "
            f"**{R['null_placebo']}**. A coin. This is the honest baseline the real (unmeasurable) "
            "world could well be."
        ),
        md(
            "### 4b · The pooled slope on the null — H₂\n\n"
            "Fama-MacBeth pooled slope of next-month return on within-month z(flow). On the null it "
            "should be ~0."
        ),
        code(
            "reg = st.pooled_slope(panel_null)\n"
            "df = panel_null.copy()\n"
            "df['zflow'] = df.groupby('month')['flow'].transform(lambda x: (x-x.mean())/x.std(ddof=0) if x.std(ddof=0)>0 else x*0)\n"
            "df['dret'] = df.groupby('month')['fwd_ret'].transform(lambda x: x-x.mean())\n"
            "fig, ax = plt.subplots(figsize=(8, 4.6))\n"
            "ax.scatter(df['zflow'], df['dret']*100, s=8, color=GREY, alpha=.4)\n"
            "xs = np.linspace(df['zflow'].min(), df['zflow'].max(), 50)\n"
            "ax.plot(xs, reg['slope']*xs*100, c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('within-month z(flow)'); ax.set_ylabel('within-month de-meaned fwd return %')\n"
            "ax.set_title(f\"null pooled slope t {reg['t']:+.2f}, corr {reg['corr']:+.3f} — flat\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"null slope-t {reg['t']:+.2f} | corr(flow, fwd) {reg['corr']:+.3f} | n {reg['n']}\")"
        ),
        md(
            f"> 💡 In plain words: H₂ flat on the null — slope-*t* **+{R['null_slope_t']}**, corr "
            f"**+{R['null_corr']}**. No firm-level flow→return link when none is planted."
        ),
        md(
            "### 4c · Robustness across leg fractions (planted world)\n\n"
            "To show the *engine's* behaviour is stable, switch on a modest edge and sweep the leg "
            "fraction. (On the null every fraction is ~0, so the sweep is only informative once an "
            "edge exists.)"
        ),
        code(
            "panel_edge, _ = data.synthetic_panel(flow_alpha=0.005, seed=561)\n"
            "tab = st.robustness_sweep(panel_edge, fracs=(0.25, 0.33, 0.50))\n"
            "fig, ax = plt.subplots(figsize=(8, 4.2))\n"
            "ax.bar([str(f) for f in tab.index], tab['ann']*100, color=GREEN, width=.5)\n"
            "for x,(f,row) in enumerate(tab.iterrows()):\n"
            "    ax.text(x, row['ann']*100+.4, f\"t {row['t']:+.1f}\", ha='center', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xlabel('leg fraction'); ax.set_ylabel('spread %/yr')\n"
            "ax.set_title('Planted-world spread: sign & significance stable across leg fractions')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(3).to_string())"
        ),
        md(
            "> 💡 In plain words: in the planted world the spread is positive and significant at every "
            "leg fraction (shrinking toward 50% as legs dilute) — the engine's sort is robust; the "
            "*data*, not the sort, is what's missing."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — null spread a coin (*t* +{R['null_t']}, placebo *p* "
            f"{R['null_placebo']}); literature supports *a* flow effect but the sign is disputed; no "
            "real tape ⇒ cannot certify `Real`.\n"
            f"- **Tradability `MIRAGE`** — ETF flows are unmeasurable on free data; the signal itself "
            "is out of reach, so there is nothing to trade."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow (planted world)\n\n"
            "*If* the signal existed and were real, would ETF frictions eat it? Charge a monthly "
            "rebalance cost per leg + a short borrow on the planted-edge book."
        ),
        code(
            "net = st.net_annualised(panel_edge, frac=0.33, cost_bps=3.0, borrow_ann_bps=40.0)\n"
            "labels = ['gross', 'net\\n(costs+borrow)']\n"
            "vals = [net['gross_ann']*100, net['net_ann']*100]\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(labels, vals, color=[GREEN, AMBER], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('spread %/yr')\n"
            "ax.set_title(f\"planted world: gross {vals[0]:+.1f}% -> net {vals[1]:+.1f}% (3 bps/leg + 40 bps borrow)\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"gross {net['gross_ann']*100:+.2f}% -> net {net['net_ann']*100:+.2f}% | cost {net['cost_ann']*100:.2f}% | borrow {net['borrow_ann']*100:.2f}%\")"
        ),
        md(
            f"> 💡 In plain words: ETFs are cheap and liquid, so frictions are mild (gross "
            f"+{R['edge_ann']}% → net +{R['edge_net']}%). But that is a *planted* world — mild costs "
            "don't rescue tradability when you cannot measure the signal at all. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the both-signs synthetic control\n\n"
            "The decisive check: is the engine a faithful detector, or does it always print ~0? Plant "
            "the flow effect at a range of strengths and **both signs** — momentum (+) and the "
            "crowding trap (−) — averaged over 25 seeds so no lucky seed can fake it."
        ),
        code(
            "alphas = [-0.008, -0.005, -0.003, 0.0, 0.003, 0.005, 0.008]\n"
            "ts = [st.synthetic_mean_t(data, flow_alpha=a, n_seeds=25, frac=0.33) for a in alphas]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "cols = [RED if a<0 else (GREY if a==0 else GREEN) for a in alphas]\n"
            "ax.plot(alphas, ts, '-', c='k', lw=1, zorder=1)\n"
            "ax.scatter(alphas, ts, c=cols, s=70, zorder=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(-2, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, c='k', lw=.6, alpha=.4)\n"
            "ax.set_xlabel('planted flow_alpha  (<0 = crowding trap, >0 = flow momentum)')\n"
            "ax.set_ylabel('mean spread-t (25 seeds)')\n"
            "ax.set_title('The engine catches BOTH signs — flat at the null, past +/-2 as the effect grows')\n"
            "plt.tight_layout(); plt.show()\n"
            "for a, t in zip(alphas, ts): print(f'flow_alpha {a:+.3f} -> mean spread-t {t:+.2f}')"
        ),
        md(
            "The mean spread-*t* is ≈ 0 at the null, drives **negative** as we plant the crowding trap "
            "and **positive** as we plant momentum — the detector resolves *either* sign cleanly. So "
            "the `WEAK` verdict is a statement about the **evidence available** (synthetic-only, no "
            "real ETF-flow tape), not a broken test. Wire in a paid daily shares-outstanding feed and "
            "this same engine would settle the flow-momentum-vs-crowding sign war on the tape. For the "
            "ETF *price-vs-NAV* signal see [378 ETF-NAV-Premium](../../378-etf-nav-premium/); for "
            "*return* momentum see [507](../../507-cross-sectional-momentum/) and "
            "[518](../../518-time-series-momentum/)."
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
