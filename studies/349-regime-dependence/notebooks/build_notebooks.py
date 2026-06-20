"""Generate the two narrative notebooks for Study 349 (Regime-Dependence).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells read the cached
price parquet under ../_cache/ if present and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-05-31).
R = dict(
    window="2002-08 → 2026-05", months=286, fp="1573a701e3bd",
    # trend overlay
    trend_sharpe=0.90, trend_t=4.54,
    trend_s2000=0.97, trend_s2010=0.89, trend_s2020=0.94,
    trend_cv=0.04, trend_share=43, trend_ex_sharpe=0.91, trend_ex_t=3.84,
    trend_gap=-4.9, trend_gap_lo=-16.2, trend_gap_hi=5.9,
    # 60/40
    mix_sharpe=0.97, mix_t=5.09,
    mix_s2000=0.66, mix_s2010=1.46, mix_s2020=0.86,
    mix_cv=0.34, mix_share=48, mix_ex_sharpe=0.76, mix_ex_t=2.92,
    mix_gap=2.1, mix_gap_lo=-5.4, mix_gap_hi=10.3,
    # synthetic control
    dur_sharpe=0.34, dur_cv=0.42, dur_share=26, dur_ex_sharpe=0.30, dur_ex_t=2.14,
    fit_sharpe=0.41, fit_cv=1.02, fit_share=55, fit_ex_sharpe=0.22, fit_ex_t=1.56,
    fit_gap_lo=4.5, fit_gap_hi=33.8,
    dur_gap_lo=-14.7, dur_gap_hi=20.2,
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

from regime_dependence import data, strategy as st

def _have_cache():
    return os.path.exists(data._cache_path(data.DEFAULT_CACHE))

HAVE_REAL = _have_cache()
print("real price cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Regime-Dependence — was it skill, or just one decade? 🪟\n"
            "### A strategy that ruled the 2010s: durable edge, or a bet on one era?\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Did_ruling_one_decade_prove_skill%3F: It_depends](https://img.shields.io/badge/Did_ruling_one_decade_prove_skill%3F-It_depends-8b949e?style=flat-square)\n\n"
            "Every backtest hands you one number — a Sharpe ratio over the whole sample — and it's "
            "tempting to read a big one as proof of skill. But a strategy can earn a *spectacular* "
            "full-sample Sharpe while quietly owing more than half of it to a single lucky decade. "
            "This notebook builds two strategies where we *know* the answer, and shows the one trick "
            "that tells a durable edge apart from a one-era bet.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the bootstrap and the "
            "drop-a-decade re-estimate? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does a big full-sample Sharpe prove skill? | **No — not on its own.** A strategy can "
            "post a top-tier Sharpe while earning most of its edge in one decade and ~nothing in the "
            "others. |\n"
            "| How do you catch it? | **Split the tape by decade, then drop the best one.** A durable "
            "edge barely flinches; a one-era bet collapses. |\n"
            "| What did the real strategies do? | A **12-month trend** overlay scored ~0.9 in *every* "
            f"decade (rock-steady). A **static 60/40** ruled the 2010s (Sharpe {R['mix_s2010']}) but "
            f"still survived dropping it ({R['mix_ex_sharpe']}). |\n"
            "| Is this a strategy you can trade? | **No — it's a ruler, not a trade.** It tells you "
            "whether to *believe* a backtest. |\n\n"
            "> The headline Sharpe is an average over regimes. Average away the regimes and you can't "
            "see which decade paid the bill — so split first, *then* marvel."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"This strategy printed a Sharpe above 1 for twenty years. That's not luck — that's an "
            "edge.\"*\n\n"
            "It's the most natural inference in finance, and it's how most strategies get sold. The "
            "strong version: a long, high full-sample Sharpe is sufficient evidence of a durable, "
            "repeatable edge. The textbook cautionary tale is the **60/40 portfolio** — a glorious "
            "multi-decade record that, after 2022, a lot of people realised had ridden a 40-year fall "
            "in interest rates. Was it a timeless mix, or a bet on one regime that happened to last a "
            "long time?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a strategy's edge lives in one regime, then the moment the regime turns the edge is "
            "gone — and you'll have sized your whole plan around a number that was never going to "
            "repeat. This is the single most expensive mistake in quant investing: mistaking *one "
            "long good era* for *a law of markets*. The fix is cheap and almost nobody does it — "
            "before you trust a backtest, ask it how it would have done **without its best decade**."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We can't peek inside a real strategy to *know* whether its edge is durable — so first we "
            "build a fake world where we planted the answer ourselves:\n\n"
            "1. **A durable strategy** — a small edge present in *every* decade.\n"
            "2. **A regime-fitted strategy** — a big edge in *one* decade and nothing the rest of the "
            "time. We rig it so its full-sample Sharpe is **as good as or better than** the durable "
            "one's.\n\n"
            "Then we apply the lens — decade-by-decade Sharpe, and the drop-the-best-decade test — and "
            "check it correctly fingers the fake. *Only then* do we point it at real strategies "
            "(a trend overlay and a 60/40 mix). If the lens can't catch a planted fake, it proves "
            "nothing on the real tape."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the two synthetic strategies.** Same market, same length. One has a steady edge "
            "everywhere; the other has a giant edge in regime 2 and nothing elsewhere. Look at their "
            "Sharpe *decade by decade*:"
        ),
        code(
            "df, reg, truth = data.synthetic_regimes()\n"
            "rs_dur = st.regime_sharpes(df['durable'], reg)\n"
            "rs_fit = st.regime_sharpes(df['regime_fit'], reg)\n"
            "x = np.arange(len(rs_dur))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "w = 0.38\n"
            "ax.bar(x - w/2, rs_dur.values, w, color=GREEN, label='Durable (edge every regime)')\n"
            "ax.bar(x + w/2, rs_fit.values, w, color=RED, label='Regime-fitted (edge ONE regime)')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'regime {i}' for i in rs_dur.index])\n"
            "ax.set_ylabel('Sharpe within the regime'); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_title('One of these earns its keep everywhere; the other in a single regime')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('full-sample Sharpe  durable:', round(st.sharpe(df['durable']),2),\n"
            "      '  regime-fitted:', round(st.sharpe(df['regime_fit']),2))"
        ),
        md(
            f"Here's the trap in one line: the **regime-fitted** strategy's full-sample Sharpe "
            f"(~{R['fit_sharpe']}) is actually **higher** than the durable one's (~{R['dur_sharpe']}). "
            "If you only ever looked at the headline number, you'd pick the *worse* strategy. The "
            "decade-by-decade bars give it away instantly — one bar towers, the rest are flat."
        ),
        md(
            "**Now the test that needs no chart-reading: drop each strategy's best decade and "
            "re-measure.** A durable edge shrugs it off; a one-era bet falls apart."
        ),
        code(
            "dur = st.leave_hot_regime_out(df['durable'], reg)\n"
            "fit = st.leave_hot_regime_out(df['regime_fit'], reg)\n"
            "labels = ['Durable', 'Regime-fitted']\n"
            "full = [dur['sharpe_full'], fit['sharpe_full']]\n"
            "exhot = [dur['sharpe_ex_hot'], fit['sharpe_ex_hot']]\n"
            "x = np.arange(2); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(x - w/2, full, w, color=GREY, label='full sample')\n"
            "ax.bar(x + w/2, exhot, w, color=[GREEN, RED], label='best decade removed')\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('Sharpe')\n"
            "ax.set_title('Drop the best decade: the durable edge survives, the bet collapses')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"durable: {dur['sharpe_full']:.2f} -> {dur['sharpe_ex_hot']:.2f}  \"\n"
            "      f\"(t {dur['t_full']:.2f} -> {dur['t_ex_hot']:.2f})\")\n"
            "print(f\"fitted : {fit['sharpe_full']:.2f} -> {fit['sharpe_ex_hot']:.2f}  \"\n"
            "      f\"(t {fit['t_full']:.2f} -> {fit['t_ex_hot']:.2f})\")"
        ),
        md(
            f"The durable strategy barely moves ({R['dur_sharpe']} → {R['dur_ex_sharpe']}, still "
            f"statistically significant). The regime-fitted one drops to ~{R['fit_ex_sharpe']} and its "
            f"significance evaporates — its 'edge' *was* the one decade. **The headline Sharpe lied; "
            "the drop-a-decade test told the truth.**"
        ),
        md(
            "**Finally, the real tape.** Two well-worn strategies on US markets: a 12-month trend "
            "overlay and a static 60/40. How stable is each, decade by decade?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets = data.load_real(fetch=False)\n"
            "    trend = st.apply_trend(rets['equity'], lookback=12)\n"
            "    mix = st.apply_6040(rets['equity'], rets['bond'])\n"
            "    rt = st.regime_sharpes(trend, st.decade_label(trend.index))\n"
            "    rm = st.regime_sharpes(mix, st.decade_label(mix.index))\n"
            "    decs = sorted(set(rt.index) | set(rm.index))\n"
            "    tv = [rt.get(d, np.nan) for d in decs]; mv = [rm.get(d, np.nan) for d in decs]\n"
            "else:\n"
            f"    decs = [2000, 2010, 2020]\n"
            f"    tv = [{R['trend_s2000']}, {R['trend_s2010']}, {R['trend_s2020']}]\n"
            f"    mv = [{R['mix_s2000']}, {R['mix_s2010']}, {R['mix_s2020']}]\n"
            "x = np.arange(len(decs)); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(x - w/2, tv, w, color=GREEN, label='Trend (12m)')\n"
            "ax.bar(x + w/2, mv, w, color=AMBER, label='60/40')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{d}s' for d in decs])\n"
            "ax.set_ylabel('Sharpe in the decade')\n"
            "ax.set_title('Trend is flat across decades; 60/40 ruled the 2010s')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('trend by decade:', [f'{v:+.2f}' for v in tv])\n"
            "print('60/40 by decade:', [f'{v:+.2f}' for v in mv])"
        ),
        md(
            f"The trend overlay is almost boringly stable — ~0.9 in every decade. The 60/40 clearly "
            f"**ruled the 2010s** ({R['mix_s2010']}) versus the 2000s ({R['mix_s2000']}) and 2020s "
            f"({R['mix_s2020']}). On this short, three-decade tape both real strategies *survive* "
            "dropping their best decade — but the 60/40 leaned on one era far more than the trend did. "
            "The lens shows you exactly how much."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** This is a method, not a tradable effect; and on the real tape the "
            "decade differences are too small (three decades) to resolve from noise. The planted "
            "synthetic answer is a *machinery proof*, never market evidence.\n"
            "- **Tradability — Mirage.** There's nothing to buy. It's a ruler for *other* strategies.\n"
            "- **Did ruling one decade prove skill? — It depends, and now you can tell.** A strategy "
            "can post a great headline Sharpe and still be a one-era bet (we proved it on a planted "
            "fake). Split by decade and drop the best one, and the durable edges keep standing while "
            "the bets fall over."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "You don't trade a ruler — but it changes *what* you'll trade. Run any candidate strategy "
            "through it before you commit capital: if its edge halves (or vanishes) when you remove "
            "its best decade, you're looking at a regime bet, not a skill, and you should size it like "
            "one (or pass). The cheap discipline this study sells: **never accept a full-sample Sharpe "
            "without asking what's left after its luckiest era.** That one question would have saved a "
            "lot of 60/40 investors a lot of surprise in 2022."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Pick the regimes that matter.** We split by calendar decade — but you can split by "
            "rate regime, by VIX level, by bull/bear. The lens is the same; the regimes are your "
            "hypothesis about *what* the strategy is really betting on.\n"
            "- **Run it on the desk's other studies.** Take any 'Real' strategy on this bench and ask "
            "it the drop-best-decade question. Some will hold; some will wobble.\n"
            "- **Longer history.** Our bond leg only lists from 2002. Swap in a longer Treasury total-"
            "return series and you get five-plus decades — and far more power in the gap test.\n\n"
            "*Think your favourite backtest is skill? Fork this, split its returns by decade, and drop "
            "the best one. If the edge walks out the door with that decade, it was never yours.*"
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
            "# Regime-Dependence — a quantitative teardown 🔬\n"
            "### Decade-conditional Sharpe · drop-best-decade HAC *t* · cross-regime gap bootstrap CI · "
            "a planted positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Did_ruling_one_decade_prove_skill%3F: It_depends](https://img.shields.io/badge/Did_ruling_one_decade_prove_skill%3F-It_depends-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* The object under test is a "
            "**method of evaluation**: can decade-conditional Sharpe, a drop-best-decade re-estimate, "
            "and a cross-regime gap CI separate a durable edge from a strategy that merely *ruled one "
            "era* — a separation a single full-sample Sharpe cannot make.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo total-return monthly, `^SP500TR` + `IEF`, "
            "2002–2026 (as-of 2026-05-31; the bond leg caps the history at three decades — see beat 3); "
            "the offline core and tests run on a deterministic synthetic regime panel. Methods in "
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
            f"| **Signal** | `NONE` | A methodology demo certifies no tradable effect; the real-tape "
            f"cross-regime gap CIs straddle zero (trend [{R['trend_gap_lo']}, {R['trend_gap_hi']}], "
            f"60/40 [{R['mix_gap_lo']}, {R['mix_gap_hi']}] %/yr — 3 regimes, under-powered). The "
            "planted control is a machinery proof, not market evidence. |\n"
            f"| **Tradability** | `MIRAGE` | The product is a measuring instrument, not a strategy. |\n"
            f"| **Did ruling one decade prove skill?** | `IT DEPENDS` | On synthetic ground truth a "
            f"regime-fitted strategy with the *higher* full-sample Sharpe ({R['fit_sharpe']} vs "
            f"{R['dur_sharpe']}) collapses on drop-best-decade (HAC *t* {R['fit_ex_t']} < 2) while the "
            f"durable edge survives ({R['dur_ex_t']} ≥ 2). The lens reverses the headline ranking. |\n\n"
            "> 💡 In plain words: the full-sample Sharpe is an estimator that averages over regimes, so "
            "it is blind to *where* the edge lives. Conditioning on regime — and then deleting the best "
            "one — is the missing diagnostic."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a strategy's monthly excess returns be $r_t$, partitioned into regimes "
            "$g\\in\\{1,\\dots,G\\}$. The full-sample Sharpe is "
            "$\\widehat{SR} = \\bar r / \\hat\\sigma_r \\cdot \\sqrt{12}$.\n\n"
            "- **H₀ (the believer's claim).** A large $\\widehat{SR}$ over a long sample implies a "
            "durable edge: $SR_g \\approx \\widehat{SR}$ for all $g$.\n"
            "- **H₁ (regime-fitting).** The edge is concentrated: there exists a regime $g^\\star$ with "
            "$SR_{g^\\star} \\gg SR_{g\\neq g^\\star}\\approx 0$, yet $\\widehat{SR}$ is still large "
            "because $g^\\star$ dominates the pooled moments.\n\n"
            "The two are **observationally identical at the level of $\\widehat{SR}$** — that is the "
            "whole problem. The discriminating statistics: (i) the per-regime $SR_g$ and their "
            "dispersion; (ii) the leave-$g^\\star$-out Sharpe and its HAC *t*; (iii) a block-bootstrap "
            "CI on $\\bar r_{g^\\star} - \\bar r_{\\neg g^\\star}$. We plant H₁ in synthetic data and "
            "show the lens recovers it."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Bailey & López de Prado's *deflated Sharpe* punishes a Sharpe for the number of trials it "
            "took to find; regime-fitting is the temporal cousin — the 'trial' is the *window*. If a "
            "20-year Sharpe of 1.0 is really a 10-year Sharpe of 1.6 stapled to a 10-year Sharpe of "
            "0.2, then your forward expectation, conditional on the regime turning, is the 0.2 — and "
            "your position sizing built on 1.0 is levered to a number that no longer exists. The dollar "
            "stakes are a drawdown you didn't budget for. The desk's contribution is to make the "
            "diagnosis a *two-line* procedure: split, then drop the best."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Positive control (synthetic).** `data.synthetic_regimes` paints $G=6$ regimes of 120 "
            "months and emits two return series on the *same* market factor: `durable` (a constant "
            "edge `edge_durable` in every regime) and `regime_fit` (an edge `edge_hot` in one regime, "
            "zero elsewhere), tuned so the *full-sample* Sharpe of `regime_fit` ≥ that of `durable`.\n"
            "- **The lens.** Per-regime Sharpe (`regime_sharpes`); a stability score — the CV of the "
            "per-regime Sharpe and the share of cumulative excess earned in the best regime "
            "(`stability_score`); the leave-best-regime-out Sharpe and HAC *t* (`leave_hot_regime_out`); "
            "and a block-bootstrap CI on the best-minus-rest mean gap (`hot_minus_rest_ci`).\n"
            "- **Inference.** Newey-West HAC *t* on the level and on the ex-best re-estimate; circular "
            "block-bootstrap (Politis-Romano) for the gap CI. Excess-vs-excess: the trend overlay's "
            "in-cash months earn the cash rate (0 here), so its series *is* the excess return.\n"
            "- **Real tape.** `^SP500TR` + `IEF` total return, monthly, 2002–2026, split by calendar "
            "decade. **Limitation, named:** `IEF` lists from 2002, so only three regimes exist — the "
            "gap test is low-powered, and we say so rather than over-claim.\n"
            "- **One execution lag.** The trend signal known at the close of month *t* sets the "
            "position for *t+1* (a single `shift`); the 60/40 weight is calendar-known and needs none."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The trap — equal headline Sharpe, opposite stability\n\n"
            "Per-regime Sharpe for the two synthetic strategies. The pooled Sharpe cannot distinguish "
            "them; the conditional Sharpe does at a glance."
        ),
        code(
            "df, reg, truth = data.synthetic_regimes()\n"
            "rs_dur = st.regime_sharpes(df['durable'], reg)\n"
            "rs_fit = st.regime_sharpes(df['regime_fit'], reg)\n"
            "ss_dur = st.stability_score(df['durable'], reg)\n"
            "ss_fit = st.stability_score(df['regime_fit'], reg)\n"
            "x = np.arange(len(rs_dur)); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(9.3, 4.4))\n"
            "ax.bar(x - w/2, rs_dur.values, w, color=GREEN,\n"
            "       label=f'Durable (full SR {st.sharpe(df[\"durable\"]):.2f}, CV {ss_dur[\"cv\"]:.2f})')\n"
            "ax.bar(x + w/2, rs_fit.values, w, color=RED,\n"
            "       label=f'Regime-fitted (full SR {st.sharpe(df[\"regime_fit\"]):.2f}, CV {ss_fit[\"cv\"]:.2f})')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x)\n"
            "ax.set_xticklabels([f'g{i}' for i in rs_dur.index]); ax.set_ylabel('Sharpe within regime')\n"
            "ax.set_title('Equal (better!) pooled Sharpe, wildly different regime profile')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"share of excess in best regime  durable {ss_dur['share_top_regime']*100:.0f}%  \"\n"
            "      f\"fitted {ss_fit['share_top_regime']*100:.0f}%\")"
        ),
        md(
            f"> 💡 In plain words: the regime-fitted strategy's pooled Sharpe (~{R['fit_sharpe']}) "
            f"*beats* the durable one's (~{R['dur_sharpe']}), yet it earns **{R['fit_share']}%** of its "
            f"cumulative edge in a single regime (vs **{R['dur_share']}%** for the durable). The "
            f"coefficient of variation of the per-regime Sharpe — {R['fit_cv']} vs {R['dur_cv']} — is "
            "the single dial that ranks them correctly."
        ),
        md(
            "### 4b · The decisive test — drop the best regime, re-estimate HAC *t*\n\n"
            "The clean, parameter-free diagnostic: delete the regime each strategy did best in and "
            "ask whether the edge is still significant on what remains."
        ),
        code(
            "dur = st.leave_hot_regime_out(df['durable'], reg)\n"
            "fit = st.leave_hot_regime_out(df['regime_fit'], reg)\n"
            "labels = ['Durable', 'Regime-fitted']\n"
            "t_full = [dur['t_full'], fit['t_full']]; t_ex = [dur['t_ex_hot'], fit['t_ex_hot']]\n"
            "x = np.arange(2); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(x - w/2, t_full, w, color=GREY, label='HAC t, full sample')\n"
            "ax.bar(x + w/2, t_ex, w, color=[GREEN, RED], label='HAC t, best regime removed')\n"
            "ax.axhline(2, ls=':', c='k'); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('HAC t-stat')\n"
            "ax.set_title('Drop the best regime: durable stays t>2, the bet falls under')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"durable  t {dur['t_full']:.2f} -> {dur['t_ex_hot']:.2f}  \"\n"
            "      f\"SR {dur['sharpe_full']:.2f} -> {dur['sharpe_ex_hot']:.2f}\")\n"
            "print(f\"fitted   t {fit['t_full']:.2f} -> {fit['t_ex_hot']:.2f}  \"\n"
            "      f\"SR {fit['sharpe_full']:.2f} -> {fit['sharpe_ex_hot']:.2f}\")"
        ),
        md(
            f"> 💡 In plain words: both strategies clear *t* = 2 on the full sample. Remove the best "
            f"regime and the durable edge stays significant (*t* {R['dur_ex_t']}) while the "
            f"regime-fitted one drops to *t* {R['fit_ex_t']} — **below the bar**. The lens reverses the "
            "verdict the headline Sharpe implied. This is the positive control passing: the harness "
            "*can* catch a planted regime bet, so a clean pass on the real tape means something."
        ),
        md(
            "### 4c · The cross-regime gap CI — is the concentration even resolvable?\n\n"
            "A block-bootstrap CI on (best-regime mean − rest-of-sample mean). Entirely above zero ⇒ "
            "the edge is genuinely concentrated; straddling zero ⇒ not resolvable from noise."
        ),
        code(
            "g_fit = st.hot_minus_rest_ci(df['regime_fit'], reg, n_boot=2000)\n"
            "g_dur = st.hot_minus_rest_ci(df['durable'], reg, n_boot=2000)\n"
            "names = ['Regime-fitted', 'Durable']\n"
            "mids = [g_fit['gap_ann']*100, g_dur['gap_ann']*100]\n"
            "los = [g_fit['lo']*100, g_dur['lo']*100]; his = [g_fit['hi']*100, g_dur['hi']*100]\n"
            "y = np.arange(2)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 3.6))\n"
            "for i,(m,lo,hi,c) in enumerate(zip(mids,los,his,[RED,GREEN])):\n"
            "    ax.plot([lo,hi],[i,i], c=c, lw=3)\n"
            "    ax.plot(m, i, 'o', c=c, ms=9)\n"
            "ax.axvline(0, c='k', lw=1); ax.set_yticks(y); ax.set_yticklabels(names)\n"
            "ax.set_xlabel('best regime minus rest (%/yr), 95% block-bootstrap CI')\n"
            "ax.set_title('Fitted: gap resolvable above zero. Durable: straddles zero')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"fitted gap CI [{los[0]:+.1f}, {his[0]:+.1f}]  durable gap CI [{los[1]:+.1f}, {his[1]:+.1f}]\")"
        ),
        md(
            f"> 💡 In plain words: the regime-fitted strategy's best-minus-rest gap CI is "
            f"**[{R['fit_gap_lo']}, {R['fit_gap_hi']}]%/yr — entirely above zero**, correctly flagging "
            f"the concentration. The durable strategy's gap straddles zero "
            f"([{R['dur_gap_lo']}, {R['dur_gap_hi']}]) — no single regime stands out, exactly as "
            "planted. The CI tells you *whether you can even trust* the per-regime contrast."
        ),
        md(
            "### 4d · The real tape — the lens on canonical strategies\n\n"
            "Trend overlay vs 60/40, split by decade, run through the same machinery."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets = data.load_real(fetch=False)\n"
            "    trend = st.apply_trend(rets['equity'], lookback=12)\n"
            "    mix = st.apply_6040(rets['equity'], rets['bond'])\n"
            "    rows = []\n"
            "    for nm, s in [('Trend (12m)', trend), ('60/40', mix)]:\n"
            "        g = st.decade_label(s.index)\n"
            "        lo = st.leave_hot_regime_out(s, g); ss = st.stability_score(s, g)\n"
            "        rows.append((nm, st.sharpe(s), ss['cv'], int(lo['hot_regime']),\n"
            "                     lo['sharpe_ex_hot'], lo['t_ex_hot']))\n"
            "    summ = pd.DataFrame(rows, columns=['strategy','full SR','CV','best decade',\n"
            "                                       'SR ex-best','HAC t ex-best']).set_index('strategy')\n"
            "else:\n"
            f"    summ = pd.DataFrame({{\n"
            f"        'full SR':[{R['trend_sharpe']}, {R['mix_sharpe']}],\n"
            f"        'CV':[{R['trend_cv']}, {R['mix_cv']}],\n"
            f"        'best decade':[2000, 2010],\n"
            f"        'SR ex-best':[{R['trend_ex_sharpe']}, {R['mix_ex_sharpe']}],\n"
            f"        'HAC t ex-best':[{R['trend_ex_t']}, {R['mix_ex_t']}]}},\n"
            f"        index=['Trend (12m)','60/40'])\n"
            "display(summ.round(2))"
        ),
        md(
            f"> 💡 In plain words: the trend overlay is the durable archetype — CV **{R['trend_cv']}**, "
            f"and dropping its best decade leaves HAC *t* = **{R['trend_ex_t']}**. The 60/40 leaned on "
            f"the 2010s (CV **{R['mix_cv']}**), but it *still* survives removal (*t* = "
            f"**{R['mix_ex_t']}** > 2) — on this tape it is regime-*sensitive*, not regime-*fitted*. "
            "Neither real strategy is the synthetic fraud — but the lens quantifies exactly how much "
            "each owes to its best era."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — no tradable effect (it's a method); and the real-tape cross-regime "
            f"gaps are not resolvable from noise with only three decades (gap CIs straddle zero). The "
            "synthetic control is a machinery proof, never market evidence.\n"
            f"- **Tradability `MIRAGE`** — a measuring instrument, not a strategy.\n"
            f"- **Did ruling one decade prove skill? `IT DEPENDS`** — *confirmed on ground truth* that "
            f"a one-era bet can out-Sharpe a durable edge on the headline ({R['fit_sharpe']} vs "
            f"{R['dur_sharpe']}) yet collapse on drop-best-decade (*t* {R['fit_ex_t']}); the lens "
            "(conditional Sharpe + drop-best + gap CI) recovers the truth a single number hides."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — turning the ruler on your book\n\n"
            "There is no edge to harvest — the deliverable is a *gate* you put in front of every "
            "candidate strategy. The operational rule: accept a backtest's Sharpe only if (i) its "
            "per-regime CV is modest, (ii) it clears HAC *t* > 2 with its best regime removed, and "
            "(iii) its best-minus-rest gap CI does **not** sit entirely above zero. The trend overlay "
            "passes all three; many a 'Sharpe 2' backtest on this bench would not. The cost of running "
            "the gate is two function calls; the cost of skipping it is a 2022."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Better regimes than the calendar.** Decades are a crude partition. Condition on the "
            "rate regime, the inflation regime, or a Hamilton (1989) Markov-switching state, and the "
            "test gains power and economic meaning.\n"
            "- **Deflated for the split.** Combine the drop-best-decade test with the deflated Sharpe "
            "(Bailey–López de Prado) to penalise *both* the number of strategies tried and the number "
            "of regimes searched.\n"
            "- **Longer bond history.** Replace `IEF` with a spliced long Treasury total-return series "
            "to recover five-plus decades and a powered gap test on the real 60/40.\n"
            "- **Audit the bench.** Run this lens across every `REAL`-stamped study here and publish "
            "which edges survive losing their best decade.\n\n"
            "*A full-sample Sharpe is a confession averaged over regimes. Split it, drop the best "
            "decade, and read what's left — that is the number you'll actually live with.*"
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
