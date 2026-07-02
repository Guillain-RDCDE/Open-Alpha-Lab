"""Generate the two narrative notebooks for Study 558 (Failures-To-Deliver).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). Everything runs
**offline and deterministically** on the synthetic FTD panel — there is no real FTD tape (see
../failures_to_deliver/data.py), so ``HAVE_REAL`` is always False and the synthetic null world
*is* the reproducible headline. The dict ``R`` mirrors docs/results.md exactly.
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30; null world, seed 558,
# 60 names x 756 days; panel fp c0ab49b6db5b).
R = dict(
    fp_null="c0ab49b6db5b", n_events=829, horizon=5, spike_z=2.5,
    null_raw_pct=0.153, null_baseline_pct=0.182, null_abn_pct=-0.029,
    null_t=-0.136, null_placebo_p=0.513, null_net_pct=-0.188,
    # sweep (h, z, n, abn%, t)
    sweep=[(1, 2.0, 1871, -0.142, -2.227), (1, 2.5, 1382, -0.161, -2.232),
           (1, 3.0, 1053, -0.156, -1.872), (3, 2.0, 1222, -0.211, -1.529),
           (3, 2.5, 948, -0.084, -0.556), (3, 3.0, 764, -0.041, -0.237),
           (5, 2.0, 1009, -0.199, -1.004), (5, 2.5, 829, -0.029, -0.136),
           (5, 3.0, 694, -0.007, -0.03), (10, 2.0, 812, -0.172, -0.535),
           (10, 2.5, 700, 0.228, 0.67), (10, 3.0, 606, 0.224, 0.619)],
    sweep_t_lo=-2.232, sweep_t_hi=0.67,
    null30_mean_t=0.20, null30_sd_t=1.116, null30_frac_gt2=0.067,
    # control (alpha, mean t over 25 seeds)
    ctrl=[(0.0, 0.14), (0.002, 1.783), (0.004, 3.419), (0.006, 5.045), (0.010, 8.259)],
    on_alpha=0.006, on_abn_pct=1.052, on_t=4.916, on_placebo_p=0.001,
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

from failures_to_deliver import data, strategy as st

# There is NO free FTD tape (SEC bulk CUSIP flat file, no prices) -> synthetic-only, offline.
HAVE_REAL = not data.fetch_panel().empty   # always False by construction
print("real FTD tape present:", HAVE_REAL, "(synthetic-only study — no free FTD data)")

# The reproducible headline is the deterministic seed-558 NULL world (folklore off).
NULL_PANEL, _ = data.synthetic_panel(squeeze_alpha=0.0, seed=558)
print("null panel fp:", data.fingerprint(NULL_PANEL), "|", NULL_PANEL['name'].nunique(),
      "names x", NULL_PANEL['day'].nunique(), "days")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Failures-To-Deliver — do settlement fails foreshadow a squeeze? 📉\n"
            "### The meme-stock belief that 'FTD spike ⇒ shorts trapped ⇒ pop', in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Squeeze_predictor%3F: Busted](https://img.shields.io/badge/Squeeze_predictor%3F-Busted-8b949e?style=flat-square)\n\n"
            "When you sell a share you're supposed to *deliver* it a couple of days later. Sometimes "
            "the seller doesn't — that's a **failure-to-deliver (FTD)**, a settlement fail, and the "
            "SEC publishes a running tally. During the meme-stock frenzy a story took hold: a **spike "
            "in FTD means short-sellers are trapped and can't deliver**, so a **short squeeze** — a "
            "violent upward move — must be coming. Buy the FTD spike, ride the pop.\n\n"
            "It's a great story. We test it the honest way: build a realistic fake market where FTD "
            "spikes and stock moves are known to be *unrelated*, then ask whether an event study "
            "(look at what happens right after each spike) can tell the truth — and whether the "
            "seductive 'post-spike pop' survives.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** And a hard data caveat: there is **no free FTD price "
            "tape** (the SEC file is a bulk, twice-a-month, CUSIP-keyed list with no prices), so this "
            "study is deliberately **synthetic-only** — every chart is drawn by the code beside it."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| After an FTD spike, is there a pop? | **No.** On a market built with *no* link between "
            f"fails and returns, the post-spike abnormal return is **{R['null_abn_pct']}%** over 5 "
            f"days (*t* {R['null_t']}). Basically zero. |\n"
            f"| Is that just because our fake is boring? | **No** — plant a *real* pop and the engine "
            f"catches it easily (*t* +{R['on_t']} at a modest planted effect). The flat result is a "
            "*true* negative. |\n"
            f"| But I've seen convincing post-FTD-spike charts! | **So has everyone.** If you try "
            f"enough horizons and thresholds, pure noise will hand you a scary-looking *t* > 2 — we "
            "show exactly that below. |\n"
            "| Could you trade it? | **No** — and you can't even *get* the data cleanly. The FTD file "
            "is twice a month, lagged, keyed by CUSIP, with no prices. |\n\n"
            "> The squeeze-from-fails story is microstructure **folklore**. Fails are mostly "
            "settlement plumbing and the cost of borrowing, not a cornered short."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"When failures-to-deliver spike, shorts are trapped — a squeeze is coming, so the "
            "stock pops.\"* — meme-stock-era microstructure folklore.\n\n"
            "The intuition feels airtight: if sellers *can't deliver*, surely they're desperate "
            "shorts who'll be forced to buy back at any price. But real fails happen for dull "
            "reasons — market-makers legitimately fail while hedging, options assignments settle "
            "late, borrow is briefly expensive. A fail is a plumbing event far more often than a "
            "cornered-short event."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If the folklore were true it would be a money printer: watch the SEC fails file, buy "
            "every spike, collect the squeeze. The desk has already looked at the meme phenomenon "
            "itself ([213 Meme-Stocks](../../213-meme-stocks/)) and at short interest "
            "([262 Short-Interest](../../262-short-interest/)); here we go straight at the specific "
            "**FTD-spike-as-squeeze-predictor** signal."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Spot the spikes.** For each stock, flag the days its fails jump far above its own "
            "recent normal (a trailing z-score) — using only the *past*, so no cheating.\n"
            "2. **Look forward.** After each spike, measure the next 5 days' return.\n"
            "3. **Compare to the market.** Subtract what the average stock did over the same days, so "
            "we isolate the *abnormal* pop the folklore promises — not just a rising tide.\n\n"
            "Two honesty guards: we only count spikes far enough apart that their forward windows "
            "don't overlap (overlapping windows fake confidence), and we enter *after* the spike day "
            "(no peeking at the move we're trying to predict).\n\n"
            "*Data reality:* we can't do this on the real SEC file (twice-monthly, CUSIP-keyed, no "
            "prices), so we build a **faithful synthetic market** where we *know* the truth — fails "
            "and returns are unrelated — and check the engine gets it right."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**What happens in the 5 days after an FTD spike, versus the market?**"
        ),
        code(
            "es = st.event_study(NULL_PANEL, horizon=5, spike_z=2.5)\n"
            "raw = es['mean_fwd']*100; base = es['baseline_mean']*100; abn = es['excess']*100\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['post-spike\\n(5d)','market\\nbaseline','ABNORMAL\\n(pop)'],\n"
            "       [raw, base, abn], color=[GREY, GREY, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'The promised pop is ~zero: abnormal {abn:+.2f}% (t {es[\"t\"]:+.2f})')\n"
            "fig.suptitle('synthetic NULL world — fails & returns unrelated by construction', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'{es[\"n_events\"]} spike events | post-spike {raw:+.2f}% | market {base:+.2f}% | ABNORMAL {abn:+.2f}% (t {es[\"t\"]:+.2f})')"
        ),
        md(
            f"There's the answer. The post-spike **abnormal** return is **{R['null_abn_pct']}%** — a "
            f"rounding error — with a *t* of **{R['null_t']}**. No pop. And that's the truth we built "
            "in: in this world fails carry *no* information about future returns."
        ),
        md(
            "**\"But I've seen charts where the pop is obvious!\"** Here's the trap. Let's try lots of "
            "horizons (1, 3, 5, 10 days) and spike thresholds — on the *same noise* — and see what "
            "the biggest *t* we can find is:"
        ),
        code(
            "sw = st.robustness_sweep(NULL_PANEL, horizons=(1,3,5,10), thresholds=(2.0,2.5,3.0))\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "labels = [f\"h{int(r.horizon)}/z{r.spike_z}\" for r in sw.itertuples()]\n"
            "cols = [RED if abs(t)>2 else GREY for t in sw['t']]\n"
            "ax.bar(range(len(sw)), sw['t'], color=cols, width=.7)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(range(len(sw)))\n"
            "ax.set_xticklabels(labels, rotation=60, fontsize=8); ax.set_ylabel('t-stat')\n"
            "ax.set_title('On PURE NOISE, cherry-picking a cut fakes |t| > 2 (red)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('sweep t range:', round(sw['t'].min(),2), 'to', round(sw['t'].max(),2))"
        ),
        md(
            f"There it is: on a market with **nothing** in it, the 1-day cut prints a *t* of about "
            f"**{R['sweep_t_lo']}** (the *wrong* sign, even). The sweep *t* runs "
            f"**{R['sweep_t_lo']} → {R['sweep_t_hi']}**. Hunt across enough horizons and thresholds "
            "and noise *will* hand you a significant-looking result. That striking post-FTD-spike "
            "chart you saw online is almost certainly one lucky cut."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Post-spike abnormal return {R['null_abn_pct']}% (*t* {R['null_t']}), "
            f"placebo *p* {R['null_placebo_p']}. And with no free FTD tape, `REAL` is off the table "
            "anyway.\n"
            "- **Tradability — Mirage.** Nothing to harvest before costs; the FTD-spiking name is the "
            "costly-to-borrow tail; and the raw data isn't cleanly reachable.\n"
            "- **Squeeze predictor? — Busted.** A sweep on pure noise fakes a *t* > 2 — the classic "
            "multiple-comparisons mirage."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No — twice over. First, there's no edge to trade. Second, the raw signal isn't even "
            "usable: the SEC fails file lands *twice a month*, lagged by the settlement cycle, keyed "
            "by CUSIP (not ticker), with no prices attached. By the time you've stitched it to a "
            "clean, survivorship-free price panel, any fleeting 'signal' is long gone — and the names "
            "you'd be buying are exactly the hard-to-borrow ones a short trade hates."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The neighbours.** [213 Meme-Stocks](../../213-meme-stocks/) (the return phenomenon) "
            "and [262 Short-Interest](../../262-short-interest/) (short interest / days-to-cover).\n"
            "- **The engine isn't blind.** In the quant notebook we *plant* a real post-spike pop and "
            "watch the same test catch it past *t* +2 — proof the flat null here is a true negative.\n\n"
            "*Think fails really do predict squeezes? Fork this, get a real CUSIP-linked FTD + price "
            "panel (survivorship-clean), and show the post-spike abnormal return clearing t = 2.*"
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
            "# Failures-To-Deliver — a quantitative teardown 🔬\n"
            "### Market-adjusted event study · one-sample *t* · label-shuffle placebo · horizon×threshold sweep · costs & borrow · seed-robust synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Squeeze_predictor%3F: Busted](https://img.shields.io/badge/Squeeze_predictor%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether the forward "
            "return after a failures-to-deliver spike is abnormally positive (the squeeze folklore).\n\n"
            "> ⚠️ **Not investment advice.** **Synthetic-only, offline, deterministic** — there is no "
            "free FTD tape (SEC bulk CUSIP flat file, no prices), so this study *cannot* earn a "
            f"`REAL` stamp. Reproducible headline: the seed-558 null world (panel fp `{R['fp_null']}`, "
            "as-of 2026-06-30). Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Post-spike **abnormal** return **{R['null_abn_pct']}%** (5d, "
            f"market-adjusted), one-sample *t* **{R['null_t']}**, placebo *p* {R['null_placebo_p']}. "
            "No free real tape ⇒ `REAL` impossible. |\n"
            f"| **Tradability** | `MIRAGE` | Nothing before costs; net {R['null_net_pct']}% (5 bps/leg "
            "+ 300 bps borrow); raw FTD file is semi-monthly, lagged, CUSIP-keyed, price-free. |\n"
            f"| **Squeeze predictor?** | `BUSTED` | Sweep on pure noise fakes \\|*t*\\| up to "
            f"**{abs(R['sweep_t_lo']):.2f}**; across 30 null seeds {R['null30_frac_gt2']*100:.0f}% "
            "individually print \\|*t*\\| > 2. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), but with no "
            "planted effect it correctly finds nothing — and a horizon/threshold grid manufactures "
            "significance on noise."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a name have an FTD-spike event at day $t$ and let $r_{t+1:t+h}$ be its forward "
            "cumulative return; let $m_{t+1:t+h}$ be the cross-name (market) forward return. Define "
            "the **abnormal** post-spike return $a = r - m$.\n\n"
            "- **H₁ (the squeeze).** $\\mathbb{E}[a] > 0$ at *t* ≥ 2 (an abnormal pop follows spikes).\n"
            "- **H₂ (robustness).** The sign/significance of H₁ is stable across horizons $h$ and "
            "spike thresholds $z$.\n"
            "- **H₃ (net).** H₁ survives costs and the hard-to-borrow short leg.\n\n"
            "We find **H₁ rejected** on the honest null (abnormal ≈ 0, *t* ≈ 0), **H₂ rejected** (a "
            "sweep manufactures |*t*| > 2 on noise), and **H₃ moot** (no gross edge to defend). The "
            "positive control shows the *engine* would accept H₁ if a real pop existed — so this is a "
            "true negative, and the data-availability wall caps the study at `NONE` regardless."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why**. The microstructure literature says fails are dominated by "
            "market-making and cost-of-borrow, not cornered shorts — Boni (2006), Evans-Geczy-Musto-"
            "Reed (2009), Fotak-Raman-Yadav (2014), who find fails associate with *better* liquidity "
            "and *lower* volatility, the opposite of an impending squeeze. Our null world encodes "
            "exactly that: fails (a persistent, spiky AR(1) process) are statistically independent of "
            "returns. The question is whether an honest event study *confirms* the null instead of "
            "hallucinating a pop."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Events.** Per name, flag days where the trailing-60d FTD z-score $\\ge z$ (causal, "
            "past-only). Enforce a **refractory period** of one horizon so kept events' forward "
            "windows are disjoint (overlapping windows are dependent and inflate the *t*).\n"
            "- **Abnormal return.** $a = r_{t+1:t+h} - m_{t+1:t+h}$ (market-adjusted), $h = 5$ days.\n"
            "- **Inference.** One-sample *t* on the abnormal returns (H0: mean = 0); a **label-shuffle "
            "placebo** null (move the spike labels to random non-event days).\n"
            "- **Robustness.** Sweep $h \\in \\{1,3,5,10\\}$ × $z \\in \\{2.0,2.5,3.0\\}$ — and read it "
            "as a multiple-comparisons demonstration.\n"
            "- **Frictions.** Round-trip cost on the long + a punitive borrow (FTD-spikers are the "
            "hard-to-borrow tail).\n"
            "- **Positive control.** A deterministic synthetic world with a planted pop "
            "(`squeeze_alpha > 0`) and a null — averaged over 25 seeds (house rule).\n\n"
            "Timing: the spike is read at day $t$; entry is at $t$'s close and the forward return is "
            "$t+1..t+h$. No future data enters the signal (no look-ahead). The one-bar entry lag is "
            "the single documented convention."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The event study — H₁\n\n"
            "Market-adjusted abnormal post-spike return, its one-sample *t*, and the placebo null."
        ),
        code(
            "es = st.event_study(NULL_PANEL, horizon=5, spike_z=2.5)\n"
            "p = st.placebo_pvalue(NULL_PANEL, horizon=5, spike_z=2.5, n_perm=1000, seed=558)\n"
            "abn, t = es['excess']*100, es['t']\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['raw post-spike','market baseline','ABNORMAL'],\n"
            "       [es['mean_fwd']*100, es['baseline_mean']*100, abn], color=[GREY, GREY, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('5-day forward return %')\n"
            "ax.set_title(f'Abnormal post-spike = {abn:+.2f}%  (one-sample t {t:+.2f}, placebo p {p:.3f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'{es[\"n_events\"]} events | abnormal {abn:+.3f}% | t {t:+.2f} | placebo p {p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: H₁ **rejected.** The abnormal post-spike return is "
            f"**{R['null_abn_pct']}%** (*t* {R['null_t']}) and the placebo *p* = {R['null_placebo_p']} "
            "is a coin-flip — the spike labels carry no forward information, exactly as built."
        ),
        md(
            "### 4b · Robustness — H₂ and the multiple-comparisons trap\n\n"
            "Sweep horizons × thresholds **on the same null world**. Watch a cut fake significance."
        ),
        code(
            "sw = st.robustness_sweep(NULL_PANEL, horizons=(1,3,5,10), thresholds=(2.0,2.5,3.0))\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "labels = [f\"h{int(r.horizon)}/z{r.spike_z}\" for r in sw.itertuples()]\n"
            "cols = [RED if abs(t)>2 else GREY for t in sw['t']]\n"
            "ax.bar(range(len(sw)), sw['t'], color=cols, width=.7)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(range(len(sw))); ax.set_xticklabels(labels, rotation=60, fontsize=8)\n"
            "ax.set_ylabel('t-stat'); ax.set_title('Sweep on pure noise: the h1 cut fakes |t| > 2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(sw.round(3).to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words: H₂ **rejected.** The sweep *t* runs **{R['sweep_t_lo']} → "
            f"{R['sweep_t_hi']}**; the 1-day cut alone clears \\|*t*\\| > 2 (wrong sign) on a market "
            "with nothing in it. A signal you must hunt for across a grid is a data-mining artifact."
        ),
        md(
            "### 4c · The luck of one seed — why a single striking chart proves nothing\n\n"
            "Run the headline event study on 30 independent null worlds (seeds 558–587)."
        ),
        code(
            "ts = []\n"
            "for s in range(558, 588):\n"
            "    pn, _ = data.synthetic_panel(squeeze_alpha=0.0, seed=s)\n"
            "    ts.append(st.event_study(pn, horizon=5, spike_z=2.5)['t'])\n"
            "ts = np.array(ts)\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "ax.hist(ts, bins=12, color=GREY, edgecolor='w')\n"
            "ax.axvline(2, ls='--', c=RED); ax.axvline(-2, ls='--', c=RED); ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('abnormal-return t (per null seed)'); ax.set_ylabel('count')\n"
            "ax.set_title(f'Null t: mean {ts.mean():+.2f}, sd {ts.std(ddof=1):.2f}, |t|>2 in {np.mean(np.abs(ts)>2)*100:.0f}% of seeds')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'mean {ts.mean():+.3f} | sd {ts.std(ddof=1):.3f} | frac|t|>2 {np.mean(np.abs(ts)>2):.3f}')"
        ),
        md(
            f"> 💡 In plain words: averaged over seeds the *t* is ≈ 0 (mean {R['null30_mean_t']}, sd "
            f"{R['null30_sd_t']}), but **{R['null30_frac_gt2']*100:.0f}%** of individual null seeds "
            "print \\|*t*\\| > 2 with no effect present. One convincing post-FTD-spike backtest is one "
            "draw from this histogram."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — H₁ rejected (abnormal {R['null_abn_pct']}%, *t* {R['null_t']}, "
            f"placebo *p* {R['null_placebo_p']}); no free real tape ⇒ `REAL` impossible.\n"
            f"- **Tradability `MIRAGE`** — nothing before costs; net {R['null_net_pct']}%; raw data "
            "semi-monthly, lagged, CUSIP-keyed, price-free.\n"
            "- **Squeeze predictor? `BUSTED`** — sweep fakes |*t*| > 2 on noise; sign-unstable across "
            "cuts."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "There is no gross edge to defend, and the short leg is punitive."
        ),
        code(
            "es = st.event_study(NULL_PANEL, horizon=5, spike_z=2.5)\n"
            "gross = es['excess']*100\n"
            "net = st.net_event_return(es['excess'], cost_bps=5.0, borrow_ann_bps=300.0, horizon_days=5)*100\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross\\n(abnormal)','net\\n(costs+borrow)'], [gross, net], color=[GREY, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('abnormal post-spike return %')\n"
            "ax.set_title(f'Nothing before costs ({gross:+.2f}%), worse after ({net:+.2f}%)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:+.3f}% -> net {net:+.3f}% (5 bps/leg + 300 bps borrow, 5d hold)')"
        ),
        md(
            "> 💡 In plain words: there is nothing to charge costs against — the abnormal return is "
            "already ~0, and the borrow (FTD-spikers are the hard-to-borrow tail) only deepens the "
            "hole. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a *real* post-spike "
            "pop (`squeeze_alpha > 0`) of increasing strength and watch the abnormal-*t* climb past "
            "+2 — averaged over 25 seeds so no single lucky seed can fake it."
        ),
        code(
            "alphas = [0.0, 0.002, 0.004, 0.006, 0.010]\n"
            "ts = [st.synthetic_mean_t(a, n_seeds=25) for a in alphas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(alphas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = +2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted squeeze_alpha (0 = null, larger = stronger pop)')\n"
            "ax.set_ylabel('mean abnormal-t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted squeeze — flat at the null, past +2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for a, t in zip(alphas, ts): print(f'alpha {a:+.3f} -> mean abnormal-t {t:+.2f}')"
        ),
        md(
            f"The abnormal-*t* is **{R['ctrl'][0][1]}** at the null and climbs to **+{R['ctrl'][-1][1]}** "
            "as the planted pop grows — so the engine is a faithful detector. The flat null is "
            f"therefore a **true negative**: a planted world at `squeeze_alpha={R['on_alpha']}` gives "
            f"abnormal **+{R['on_abn_pct']}%** at *t* **+{R['on_t']}** (placebo *p* {R['on_placebo_p']}), "
            "yet the honest FTD folklore delivers nothing. Combined with the *no-free-FTD-tape* wall, "
            "the verdict is `NONE` × `MIRAGE`, squeeze-predictor claim `BUSTED`. Neighbours: "
            "[213 Meme-Stocks](../../213-meme-stocks/), [262 Short-Interest](../../262-short-interest/)."
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
