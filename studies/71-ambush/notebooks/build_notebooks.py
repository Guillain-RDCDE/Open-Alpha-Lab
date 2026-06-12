"""Build the two narrative notebooks for Study 71 (Ambush).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both walk the seven desk beats; the code cells run the REAL tape offline (the repo-root
``_cache/`` parquets, resolved by absolute path from ``ambush.data``). The headline
numbers quoted in PROSE come from the single REAL dict below, which mirrors
``docs/results.md`` (bench rule: one source of truth; every verify.py re-run that moves
a number triggers a rebuild here).
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# ── the pinned real run — mirrors docs/results.md (as-of 2026-06-01) ──────────
REAL = dict(
    asof="2026-06-01", lo="1993-01-29", hi="2026-06-01", n_days=8391,
    fp_spy="68198e0c321b", fp_vix="1d63caa5b039",
    lift=[0.55, 1.91, 9.21, 17.18, 42.19], lift_n=[3392, 2772, 1590, 574, 62],
    k3_bp=19.6, k3_t=3.06, k3_n=636, k4_bp=42.2, k4_t=2.43, k2_t=4.66,
    rc_p=0.009, dec_pre=17.6, dec_post=15.3, dec_tchange=0.17,
    k3_full=0.42, k3_is=0.52, k3_oos=0.28, k3_exc=1.23, k3_tim=7.6, k3_tr=15,
    k2_full=0.41, k4_oos=0.01, bh_full=0.42, bh_oos=0.62,
    ci_full="[+0.10, +0.73]", ci_oos="[-0.23, +0.79]",
    gross=1.63, fin=0.21, cost=0.19, breakeven_full=7, breakeven_oos=5,
    med_w=0.52, stops_yr=2.1, max_dd=-8.9, worst=-2.62,
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Rarity defeats costs?: Confirmed](https://img.shields.io/badge/Rarity_defeats_costs%3F-Confirmed-8b949e?style=flat-square)\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))        # the study package (ambush/)
sys.path.insert(0, os.path.abspath("../../.."))  # repo root (quantlab/)
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5.5)
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

from ambush import data, signals, strategy
from quantlab.repro import DEFAULT_AS_OF, as_of, fingerprint

spy = as_of(data.spy_frame())
vix = as_of(data.vix_series().to_frame())["vix"]
rf  = as_of(data.rf_series().to_frame())["rf"]
print(f"SPY {spy.index.min().date()} -> {spy.index.max().date()} ({len(spy)} sessions), "
      f"as-of {DEFAULT_AS_OF}")
print("fingerprints:", fingerprint(spy.round(6)), "/", fingerprint(vix.to_frame().round(6)))
"""

LIFT_CELL = """\
lift = strategy.lift_table(spy, vix)
display(lift.round(2))

ax = lift["next_bps"].plot.bar(color=["#8b949e"]*2 + ["#dab617"] + ["#2ea44f"]*2, rot=0)
ax.set_xlabel("signals firing at today's close (of 4)")
ax.set_ylabel("next-day SPY return (bps)")
ax.set_title("The ambush ladder — next-day return climbs with the confluence count")
for i, (v, n) in enumerate(zip(lift["next_bps"], lift["n"])):
    ax.annotate(f"{v:+.1f}\\nn={n}", (i, v), ha="center", va="bottom", fontsize=9)
plt.tight_layout(); plt.show()
"""

HAC_CELL = """\
from quantlab.analytics import mean_tstat_hac
for k in (2, 3, 4):
    t = mean_tstat_hac(strategy.armed_stream(spy, vix, k=k))
    print(f"armed K>={k}: {t['mean_bps']:+5.1f} bp/day | HAC t = {t['tstat']:+.2f} | n = {t['n']}")
"""

DECAY_CELL = """\
dec = strategy.premium_change(spy, vix, k=3, split="2015")
print(f"armed-day premium over the rest: {dec['premium_pre_bp']:+.1f} bp/day before 2015 "
      f"(Welch t {dec['welch_t_pre']:+.2f}) -> {dec['premium_post_bp']:+.1f} bp/day since "
      f"(t {dec['welch_t_post']:+.2f})")
print(f"test of the *difference*: t_change = {dec['t_change']:+.2f}  ->  no detectable decay")
"""

RC_CELL = """\
from quantlab import bayes
rc = bayes.reality_check(strategy.variant_panel(spy, vix, rf), n_boot=2000, seed=0)
print(f"family K in {{1,2,3,4}} (announced in the pre-registration, all of it):")
print(f"best net Sharpe {rc['observed_max_sharpe']:+.2f} -> Reality-Check p = "
      f"{rc['reality_check_pvalue']:.3f} (stationary bootstrap, {rc['n_boot']} draws)")
"""

BOOKS_CELL = """\
rows, curves = [], {}
for k in (1, 2, 3, 4):
    led = strategy.book(spy, vix, rf, k=k)
    s  = strategy.summary(led["net_excess"], led)
    so = strategy.summary(led["net_excess"]["2015":], led["2015":])
    rows.append({"book": f"K>={k}", "sharpe": s["sharpe"], "excess/yr": s["ann_excess"],
                 "time in mkt": s["time_in_market"], "trades/yr": s["trades_per_year"],
                 "OOS sharpe": so["sharpe"]})
    curves[f"K>={k}"] = (1 + led["net_excess"]).cumprod()
bh = strategy.bh_excess(spy, rf)
rows.append({"book": "SPY B&H (excess)", "sharpe": strategy.summary(bh)["sharpe"],
             "excess/yr": strategy.summary(bh)["ann_excess"], "time in mkt": 1.0,
             "trades/yr": 0, "OOS sharpe": strategy.summary(bh["2015":])["sharpe"]})
display(pd.DataFrame(rows).set_index("book").round(3))

ax = curves["K>=3"].plot(label="ambush K>=3 (net excess)", color="#2ea44f")
(1 + bh).cumprod().plot(ax=ax, label="SPY buy-and-hold (excess)", color="#8b949e", alpha=0.8)
ax.axvline(pd.Timestamp("2015-01-01"), ls="--", c="k", lw=1, label="IS | OOS split")
ax.set_yscale("log"); ax.legend(); ax.set_ylabel("growth of $1 (excess-of-cash)")
ax.set_title("The ambush book vs the market, both excess-of-cash (log scale)")
plt.tight_layout(); plt.show()
"""

SWEEP_CELL = """\
sw_full = strategy.cost_sweep(spy, vix, rf)
sw_oos  = strategy.cost_sweep(spy["2014-10":], vix, rf)   # 3-month warm-up for the vol estimator
ax = sw_full["net_sharpe"].plot(marker="o", label="full sample")
sw_oos["net_sharpe"].plot(ax=ax, marker="s", label="OOS (2015->)")
ax.axhline(0, c="k", lw=1); ax.axvline(1.0, ls="--", c="#2ea44f", label="a liquid US500 CFD (~1 bp)")
ax.set_xlabel("one-way spread (bps)"); ax.set_ylabel("net excess Sharpe, K>=3")
ax.set_title("Where the edge dies — and how far away that is")
ax.legend(); plt.tight_layout(); plt.show()
"""

RISK_CELL = """\
led = strategy.book(spy, vix, rf, k=3)
held = led[led["pos"] > 0]
eq = (1 + led["net_excess"]).cumprod()
dd = eq / eq.cummax() - 1
print(f"exposure when armed: median {held['pos'].median():.2f}x, max {held['pos'].max():.2f}x")
print(f"stop fires {led['stopped'].sum() / (len(led)/252):.1f}x/yr | "
      f"worst single day {led['net_excess'].min():+.2%} | max drawdown {dd.min():.1%}")
ax = dd.plot(color="#c0392b"); ax.set_title("Drawdown of the net excess book (K>=3) — the 1%/day budget at work")
ax.set_ylabel("drawdown"); plt.tight_layout(); plt.show()
"""


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    R = REAL
    cells = [
        md(
            "# Study 71 — Ambush 🪤\n"
            "### Four S&P 500 edges died paying the spread. What if you only trade the days they all fire at once?\n\n"
            + BADGES + "\n"
            "This bench spent seventy studies watching short-term stock-market edges die the same death: "
            "the pattern is real, but it trades so often that the bid-ask spread eats it alive. This study "
            "flips the logic — keep four of those dead edges as *filters*, stay out of the market almost "
            "always, and strike only on the rare day they agree. Rarity, not signal strength, is the cost defence.\n\n"
            "> 📓 **This is the plain-language layer.** Want the statistics, the bootstrap and the "
            "cost maths? That's the companion notebook, **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** "
            "— same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** An educational, reproducible research tool: every chart below "
            "is generated by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),
        md(
            "## The answer first 🎯\n\n"
            f"| Question | Answer |\n|---|---|\n"
            f"| Is the confluence premium real? | **Yes** — a ≥3-signal day is followed by **{R['k3_bp']:+.1f} bp** "
            f"next session vs {R['lift'][0]:+.2f} bp on a quiet day, and the climb is perfectly monotone |\n"
            f"| Did it fade like its ingredients? | **No** — {R['dec_pre']:+.1f} → {R['dec_post']:+.1f} bp/day across 2015, "
            "statistically indistinguishable |\n"
            f"| Does rarity beat the spread? | **Yes** — ~{R['k3_tr']} trades/yr leave costs eating only ~a quarter "
            "of the gross (the daily version of these edges lost *everything* to costs) |\n"
            f"| So… can you get rich? | **Not so fast** — in market only {R['k3_tim']:.0f}% of days under a strict 1%/day "
            f"risk budget, it earns **{R['k3_exc']:+.2f}%/yr** over cash. A real edge, honestly small |\n\n"
            "The rest of this notebook is *why* you should believe each line."
        ),
        md(
            "## 1 · The claim\n\n"
            "Markets keep a little mean-reversion kindness for whoever provides liquidity on ugly days: "
            "buy when a day closes crushed at the bottom of its range ([study 19](../../19-rubber-band/)), "
            "around the turn of the month ([study 42](../../42-last-call/)), after a red close "
            "([study 13](../../13-crimson-hour/)), when the VIX is jumpy ([study 03](../../03-fear-gauge/)). "
            "Each of those is *real* in the data — and each one, traded alone, **lost money net** because it "
            "trades almost every day and the spread never sleeps. The ambush claim: those four edges share the "
            "same fingerprint (panicky selling that overshoots), so the day **three or four fire together** "
            "should carry their *stacked* premium — and such days are rare enough that costs can't reach them."
        ),
        md(
            "## 2 · So what?\n\n"
            "If true, it's the first strategy this bench has met where the *cost problem is solved by design* "
            "instead of wished away. And it's exactly the shape a small CFD account can trade: no contract-size "
            "constraint, one liquid instrument (US500), a couple of trades a month, every position sized so a "
            "bad day costs at most **1% of the account**. If false, we've shown that stacking dead edges just "
            "makes a deader edge — also worth knowing, since 'combine weak signals' is the most common retail fantasy "
            "(and [study 38](../../38-chorus/) already showed *averaging* them adds nothing)."
        ),
        md(
            "## 3 · How would we even know?\n\n"
            "Everything was frozen **before** the test, in writing "
            "([docs/preregistration.md](../docs/preregistration.md)): the four definitions and their thresholds "
            "(all inherited from the source studies — nothing tuned here), the confluence rule (long only, K≥3, "
            "family K∈{1..4} declared and statistically corrected for), the CFD cost model (1 bp one-way + "
            "financing over cash), the risk overlay (vol targeting, 1%/day budget, intraday stop), the 2015 "
            "in/out-of-sample split, and the exact criteria for each verdict stamp. The harness also had to pass "
            "a lie-detector first: it must *find* a premium planted in a synthetic tape, and find *nothing* in a "
            "random walk (it does — `examples/run_synthetic_demo.py`)."
        ),
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First exhibit: line up every trading day since 1993 by how many of the four signals fired at its "
            "close, and ask what SPY did *the next day*. No sizing, no costs — just the raw ladder."
        ),
        code(LIFT_CELL),
        md(
            f"A quiet day is worth half a basis point the next morning. A four-alarm day is worth **{R['lift'][4]:+.1f} bp** "
            f"— eighty times more. The climb is perfectly ordered, on {R['n_days']:,} sessions of real tape.\n\n"
            "🔬 *For the quants:* the armed stream clears the bench's robust-inference bar — "
            f"HAC *t* = **{R['k3_t']:+.2f}** at K≥3 — and White's Reality Check across every K we announced puts "
            f"the family at **p = {R['rc_p']:.3f}**. Details next door.\n\n"
            "Second exhibit: has it rotted, like every ingredient did on its own?"
        ),
        code(DECAY_CELL),
        md(
            "Now the part that kills most paper edges — make it a *book*. Long only on confluence days, "
            "vol-targeted size, a hard stop at −1% of the account, 1 bp spread each way plus overnight "
            "financing, cash earning the T-bill the rest of the time. Raced fairly against just *holding* SPY "
            "(both measured over cash):"
        ),
        code(BOOKS_CELL),
        code(SWEEP_CELL),
        md(
            "## 5 · The verdict\n\n"
            f"**Signal `REAL`** — the ladder is monotone, the K≥3 premium carries HAC *t* = {R['k3_t']:+.2f}, "
            "the family is multiple-testing-corrected, and the premium hasn't faded.\n\n"
            f"**Tradability `FRAGILE`** — the book survives costs with a wide moat (break-even ~{R['breakeven_full']} bp "
            f"vs ~1 bp real), but it spends {R['k3_tim']:.0f}% of its life in the market and earns {R['k3_exc']:+.2f}%/yr "
            f"over cash. Out-of-sample its Sharpe is **{R['k3_oos']:+.2f}** — positive, but below the **0.30** bar we "
            "froze in advance, with a confidence interval that still touches zero. We called that FRAGILE before "
            "we knew the number; it stays FRAGILE.\n\n"
            "**Rarity defeats costs? `CONFIRMED`** — the design goal, and the study's actual discovery: the same "
            "ingredients that lose >100% of their edge to costs traded daily keep ~75% of it traded ~15×/yr."
        ),
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Mechanically, yes — that's what it was designed for. One liquid instrument, signals computable at "
            "the close in a spreadsheet, ~15 entries a year, position small (median ~0.5× the account), a stop "
            "that fires twice a year, and a worst 33-year drawdown of −9% on the excess line. The honest catch "
            "is the *prize*: ~1.2%/yr over cash on its own is dinner money, not a living — it only makes sense "
            "**stacked on top of** whatever else the account does (it's flat 92% of the time, so it stacks "
            "cleanly). And the 1% budget holds *at the stop*: an overnight gap straight through it can cost more "
            "(worst real day: −2.6%). CFDs add a second honest catch: the quoted spread is the *good* case — "
            "weekend financing, widened spreads at the close, and your broker's hedging all tax the same ~20 bp/trade edge."
        ),
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The natural upgrade** — intraday entries (buy the stop-run at 15:50, not the close) would need "
            "minute bars; the bench's daily tape can't see it.\n"
            "- **A fifth ingredient** — overnight-only holding ([study 01](../../01-overnight-anomaly/)) on "
            "armed days: exit at the open, halve the financing.\n"
            "- **Fork it** — every threshold is one constant in [`ambush/signals.py`](../ambush/signals.py); the "
            "pre-registration tells you exactly which knobs were *not* allowed to move, so you know what an "
            "honest variant looks like.\n\n"
            "*Reproduce: `python examples/verify.py` (offline, fingerprinted) · machinery proof: "
            "`python examples/run_synthetic_demo.py` · full numbers: [docs/results.md](../docs/results.md).*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    R = REAL
    cells = [
        md(
            "# Ambush — a quantitative teardown 🔬\n"
            "### Pre-registered confluence · HAC inference · Reality Check · a costed, risk-budgeted CFD book\n\n"
            + BADGES + "\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — same seven "
            "beats, every claim now carrying its standard error. The object under test is a *meta-signal*: "
            "the count of four pre-existing, individually cost-dead S&P 500 signals firing on the same close.\n\n"
            "> ⚠️ **Not investment advice.** SPY split-only daily OHLC + ^VIX raw closes (1993 → as-of "
            f"{R['asof']}), ^IRX as the per-day cash rate; protocol and thresholds frozen in "
            "[`docs/preregistration.md`](../docs/preregistration.md) before the run; sources in "
            "[`docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition — so this "
            "notebook still reads even if you skim the maths. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Why |\n|---|---|---|\n"
            f"| Signal | `REAL` | monotone lift {R['lift'][0]:+.2f} → {R['lift'][4]:+.1f} bp across K = 0…4; "
            f"armed K≥3 stream {R['k3_bp']:+.1f} bp/day, HAC *t* = **{R['k3_t']:+.2f}** (n = {R['k3_n']}); "
            f"RC over the announced family **p = {R['rc_p']:.3f}**; no decay across 2015 (*t*-change {R['dec_tchange']:+.2f}) |\n"
            f"| Tradability | `FRAGILE` | net book survives costs (break-even ~{R['breakeven_full']} bp vs ~1 bp real) "
            f"but OOS Sharpe **{R['k3_oos']:+.2f}** < the pre-registered 0.30 bar, CI {R['ci_oos']} spans 0 |\n"
            f"| Rarity defeats costs? | `CONFIRMED` | spread+financing = {R['fin'] + R['cost']:.2f}%/yr of a {R['gross']:.2f}%/yr "
            "gross (~25%), vs >100% for the same signals traded daily (study 19) |\n\n"
            "> 💡 **In plain words** — the trap catches real prey, the toll booth can't eat it, but the prey is small."
        ),
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Liquidity provision after forced selling earns a premium (Nagel 2012; Connors & Alvarez's IBS rule; "
            "Lakonishok–Smidt and McConnell–Xu on the month-turn; Whaley's fear gauge). On this bench each "
            "expression of that premium is `REAL` gross and `MIRAGE` net — killed by daily turnover, not by the "
            "tape (studies 01, 03, 13, 19, 42; study 38 adds that *averaging* signals into a composite forecast "
            "also fails). Three testable hypotheses, frozen in advance:\n\n"
            "- **H₁** next-day return is monotone increasing in the confluence count;\n"
            "- **H₂** the K≥3 armed stream clears HAC *t* ≥ 2, surviving a Reality Check over the announced K family;\n"
            "- **H₃** the K≥3 book — vol-target + 1%/day budget + stop, net of 1 bp one-way + financing — keeps a "
            "positive excess Sharpe out-of-sample (≥ 0.30 with CI > 0 for `INVESTABLE`)."
        ),
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "H₁∧H₂ alone would already be a bench first: a *new* statistically certified signal built from four "
            "dead ones — evidence that decay lives in the *trading*, not the *tape*. H₃ decides whether the "
            "rarity defence is an actual mechanism or another mirage: at ~15 one-way pairs/yr, a 1 bp toll costs "
            "~3 bp/yr against a ~125 bp/yr gross — if *that* book still dies net, confluence added nothing. "
            "The risk overlay is not decoration: it is what makes the result transferable to the smallest real "
            "account (a CFD has no contract-size floor; the 1%/day budget is the user's spec, enforced in sizing "
            "*and* as a stop)."
        ),
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "1. **Decompose** — the lift table: next-day raw close-to-close SPY return bucketed by today's count; "
            "no parameters, no overlay.\n"
            "2. **Robust inference** — Newey–West *t* on the armed stream; Welch test of the premium *change* "
            "across the announced 2015 split; circular-block-bootstrap CI on the book's Sharpe; White (2000) "
            "Reality Check on the stationary bootstrap across K∈{1,2,3,4} — the whole family we looked at, "
            "declared before looking.\n"
            "3. **Critique** — one execution lag exactly (signals at close *t* earn *t+1*; the TOM leg is "
            "calendar-known and marks *tomorrow's* membership before receiving that same lag); split-only data "
            "choice stated; VIX forward-filled across holiday gaps, never backfilled.\n"
            "4. **Alpha vs beta** — the book is raced **excess-of-cash vs excess-of-cash** against SPY B&H "
            "(a 7.6%-in-market book on raw Sharpe vs an always-long benchmark would just measure who held more "
            "T-bills — bench rule).\n"
            "5. **Execution & capacity** — cost sweep to break-even, full and OOS; financing charged per held "
            "night; stop exits pay the spread too.\n"
            "6. **Controls** — planted-premium tape must light up, seeded random walk must stay dark "
            "(`tests/test_synth_controls.py`, run in CI).\n\n"
            "> 💡 **In plain words** — we wrote the rules of the game, the referee, and the losing conditions "
            "on paper first, then played once."
        ),
        md("## 4 · The teardown\n\n### 4.1 The lift is monotone (H₁)"),
        code(LIFT_CELL),
        md("### 4.2 …and statistically real under HAC errors (H₂, first leg)"),
        code(HAC_CELL),
        md(
            "> 💡 **In plain words** — even letting the noisy days lean on each other (Newey–West), the ≥3-signal "
            "premium is more than three standard errors from zero.\n\n"
            "### 4.3 No decay across the pre-registered split — unlike every ingredient"
        ),
        code(DECAY_CELL),
        md(
            "Each ingredient's own study found gross decay (19: Sharpe halved; 42: premium −94% post-2008; "
            "67: gone entirely). The *confluence* premium is statistically flat across 2015 — consistent with the "
            "crowding story: everyone arbitrages the everyday expression of liquidity provision; the 7%-of-days "
            "tail keeps paying.\n\n"
            "### 4.4 The selection correction (H₂, second leg)"
        ),
        code(RC_CELL),
        md(
            "> 💡 **In plain words** — 'the best of four variants' is significant by construction unless you "
            "punish yourself for having looked at four; after the punishment, p ≈ 0.01.\n\n"
            "### 4.5 The book (H₃): overlay, costs, IS/OOS"
        ),
        code(BOOKS_CELL),
        md(
            f"Full-sample the K≥3 book matches the market's excess Sharpe ({R['k3_full']:+.2f} vs {R['bh_full']:+.2f}) "
            f"while exposed {R['k3_tim']:.0f}% of the time; OOS it earns {R['k3_oos']:+.2f} against the market's great "
            f"decade ({R['bh_oos']:+.2f}). K≥4 is the cautionary row: {R['k4_oos']:+.2f} OOS on ~2 trades/yr — "
            "62 events in 33 years is a sample, not a strategy.\n\n"
            "### 4.6 Cost anatomy and the moat"
        ),
        code(SWEEP_CELL),
        md("### 4.7 The risk overlay, audited"),
        code(RISK_CELL),
        md(
            "## 5 · The verdict\n\n"
            f"- **H₁ monotone:** confirmed, {R['lift'][0]:+.2f} → {R['lift'][1]:+.2f} → {R['lift'][2]:+.2f} → "
            f"{R['lift'][3]:+.2f} → {R['lift'][4]:+.2f} bp.\n"
            f"- **H₂ real:** HAC *t* = {R['k3_t']:+.2f} (K≥3), RC p = {R['rc_p']:.3f}, decay *t*-change "
            f"{R['dec_tchange']:+.2f} ⇒ **Signal `REAL`**.\n"
            f"- **H₃ tradable:** full-sample net Sharpe {R['k3_full']:+.2f} (CI {R['ci_full']}), OOS "
            f"{R['k3_oos']:+.2f} (CI {R['ci_oos']}) — positive, surviving costs, **but** below the frozen 0.30 bar "
            "with a CI through zero ⇒ **Tradability `FRAGILE`**, by the letter of the pre-registration.\n"
            f"- **Mechanism:** costs+financing take {(R['fin']+R['cost'])/R['gross']:.0%} of gross at ~{R['k3_tr']} "
            "trades/yr ⇒ **Rarity defeats costs `CONFIRMED`** — the bench's first deliberate, working cost defence."
        ),
        md(
            "## 6 · Could you trade it?\n\n"
            f"**Venue** — a US500 CFD (no size floor) or MES futures (cheaper financing, ~$5 notional floor per "
            f"0.25 pt tick is irrelevant at this size). **Costs** — the moat is real: break-even ~{R['breakeven_full']} bp "
            f"one-way full-sample (~{R['breakeven_oos']} bp OOS) against ~0.5–1 bp on a tight US500 quote; CFD "
            "weekend financing and close-auction spread widening shave it but don't close it. **Risk** — median "
            f"exposure {R['med_w']:.2f}×, stop {R['stops_yr']:.1f}×/yr, max excess drawdown {R['max_dd']:.1f}%, worst day "
            f"{R['worst']:.2f}% (gap through the stop — the 1% budget binds at the stop, not through a gap; an "
            "overnight index gap >2× the stop distance is rare but real). **Capacity** — effectively unlimited at "
            "retail size; the premium is ~20 bp on ~$0.5×NAV fifteen times a year. **The candid line** — "
            f"{R['k3_exc']:+.2f}%/yr excess is a *sleeve*, not a strategy: its value is that it is uncorrelated by "
            "construction with being long (flat 92% of days) and costs nothing to hold. Anyone selling this as a "
            "get-rich machine is selling the lift table without the time-in-market column."
        ),
        md(
            "## 7 · Going further\n\n"
            "- **Overnight-only exit** (close→open on armed days, study 01's leg): halves financing and the "
            "gap-risk window; needs the open auction's effective spread.\n"
            "- **Sizing by count** — w ∝ (K−2) instead of flat: the lift table says K=4 is worth 2.5× K=3; 62 "
            "events can't certify it (that's *why* it wasn't pre-registered).\n"
            "- **Cross-market replication** — the same four definitions run unchanged on QQQ/EWJ/EWG tapes "
            "already in `_cache/`; a confluence premium that exists *only* on SPY would be a red flag worth "
            "publishing either way.\n"
            "- **PR welcome** — `ambush/` is ~300 lines; every threshold is a named constant; "
            "`tests/` pins the accounting (one lag, costs both legs, stop, budget) so a fork can't silently cheat.\n\n"
            "*Numbers in prose: [`docs/results.md`](../docs/results.md) (fingerprints "
            f"`{R['fp_spy']}` / `{R['fp_vix']}`, as-of {R['asof']}). Reproduce: `python examples/verify.py`.*"
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
