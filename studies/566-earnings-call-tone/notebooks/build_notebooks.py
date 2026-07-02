"""Generate the two narrative notebooks for Study 566 (Earnings-Call-Tone).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). This is a SYNTHETIC-ONLY
study: every cell runs offline and deterministic on the seeded event panel; the real-tape fetch
returns an empty frame by design, so ``HAVE_REAL`` is always False and the notebook recomputes the
headline numbers live from the same generator that produced ``docs/results.md`` (mirrored in ``R``).
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30; seed 566; planted
# tone_beta=0.020; 480 calls; panel fp 3ec051a6458a).
R = dict(
    fp_panel="3ec051a6458a", fp_world="1c74d461b0e5",
    n_firms=40, n_quarters=12, n_calls=480, tone_beta=0.020,
    naive_slope=0.0251, naive_t=12.05,
    ctrl_slope=0.0176, ctrl_t=7.95, surprise_slope=0.0194,
    placebo_p=0.0005,
    ls_upbeat=2.65, ls_guarded=-3.11, ls_spread=5.76, ls_t=9.48,
    gross_ann=23.0, net_ann=21.7,
    null_naive_t=6.50, null_ctrl_t=-0.19,
    # robustness sub-panels: (label, ctrl slope, ctrl t)
    win=[("Q01-Q03", 0.0208, 4.44), ("Q04-Q06", 0.0129, 3.28),
         ("Q07-Q09", 0.0210, 5.01), ("Q10-Q12", 0.0185, 3.61)],
    # synthetic control: (tone_beta, mean ctrl slope-t over 25 seeds)
    ctrl_curve=[(0.0, -0.19), (0.01, 4.63), (0.02, 9.45), (0.04, 19.10), (0.06, 28.74)],
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

from earnings_call_tone import data, strategy as st

TONE_BETA = 0.020
PANEL, TRUTH = data.synthetic_panel(tone_beta=TONE_BETA, seed=566)

# This study is SYNTHETIC-ONLY: no free real tape exists, so the real fetch is an empty stub.
REAL = data.fetch_panel(fetch=False)
HAVE_REAL = not REAL.empty
print("synthetic panel:", len(PANEL), "calls | fp", data.fingerprint(PANEL))
print("real tape present:", HAVE_REAL, "(synthetic-only study — real fetch is an empty stub by design)")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Earnings-Call Tone — does sounding upbeat move the stock? 🗣️\n"
            "### Reading the *mood* of a company's earnings call, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Confounded_by_the_number%3F: Named](https://img.shields.io/badge/Confounded_by_the_number%3F-Named-8b949e?style=flat-square)\n\n"
            "Four times a year, a company reports its results and holds a **conference call** where "
            "management talks through the quarter. Some calls sound *upbeat* ('record demand', 'strong "
            "momentum'); others sound *guarded* ('challenging environment', 'cautious outlook'). The "
            "claim — a linguistic cousin of the famous **post-earnings-announcement drift** — is that "
            "this **tone** predicts where the stock goes in the days *after* the call. Upbeat words → "
            "the stock keeps climbing; guarded words → it keeps sliding.\n\n"
            "It sounds plausible, and the academic literature broadly supports it. But there's a trap "
            "hiding in plain sight, and this notebook is really about that trap.\n\n"
            "> 📓 **This is the plain-language layer.** Want the regressions, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice** — and note this is a **synthetic-only** study: there is no "
            "free feed of scored transcripts, so every chart is drawn on a seeded, deterministic panel. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do upbeat calls beat guarded calls afterwards? | **On the surface, yes — hugely.** The "
            f"upbeat third earned **+{R['ls_upbeat']}%** after the call, the guarded third "
            f"**{R['ls_guarded']}%** — a **+{R['ls_spread']}%** gap. |\n"
            "| So is tone a money machine? | **Careful.** Companies sound upbeat *because* the quarter "
            "was good. Most of that gap is just the earnings **number** drifting — not the words. |\n"
            f"| What happens if you strip out the number? | The tone edge **shrinks** (from a slope *t* "
            f"of {R['naive_t']} to {R['ctrl_t']}). A real, but much smaller, linguistic signal remains "
            "— *in this planted world.* |\n"
            "| Can we prove it on real data? | **No.** Scored transcripts joined to returns cost money; "
            "there's no free feed. So the honest stamp is **Weak** — the textbooks say real, this desk "
            "can't certify it. |\n\n"
            "> The moral: **tone and the earnings number are entangled.** Any 'call tone predicts "
            "returns' claim that doesn't subtract the number is mostly re-discovering post-earnings "
            "drift with a fancier vocabulary."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"The emotional tone of an earnings call predicts the stock's drift afterwards.\"* — a "
            "linguistic cousin of PEAD (Price, Doran, Peterson & Bliss 2012; Loughran & McDonald 2011).\n\n"
            "We already know the *number* drifts: a company that beats expectations tends to keep rising "
            "for weeks (post-earnings-announcement drift). The new claim is that the **words** carry "
            "*extra* information — how confident or hedged management sounds — beyond what the number "
            "already told you. If true, you could read the transcript's mood and trade the drift."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If tone genuinely adds information *beyond the number*, it's a real edge — and a whole "
            "industry of transcript-scraping NLP is justified. If it's just the number in disguise, "
            "then all that machine-reading is re-selling you plain old earnings drift. The desk has "
            "the numeric drifts on file ([363 PEAD](../../363-pead-drift/), "
            "[534 revenue-surprise](../../534-revenue-surprise-drift/)); here we ask whether the *words* "
            "beat the *number*."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Score the tone.** For each call, count upbeat vs guarded words → a *net tone* number "
            "(positive = upbeat).\n"
            "2. **Measure the drift.** Track the stock's market-adjusted return *after* the call (its "
            "'CAR').\n"
            "3. **The honest twist — subtract the number.** Because upbeat calls follow good quarters, "
            "we must ask: does tone predict the drift *once you already know the earnings surprise*? "
            "That's the real test.\n\n"
            "*Timing:* the tone is known *at* the call (the transcript is public that evening); we only "
            "measure the drift *afterwards* and enter the next session. No peeking at the future.\n\n"
            "*The catch:* there's no free database of scored transcripts + returns, so we run this on a "
            "**synthetic** panel where we *know* the truth — which lets us prove the method works and "
            "show exactly how the trap springs."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Upbeat calls vs guarded calls: who drifted more afterwards?**"
        ),
        code(
            "ls = st.tone_long_short(PANEL, frac=0.3)\n"
            "up, gd, spr, t = ls['upbeat_mean']*100, ls['guarded_mean']*100, ls['spread']*100, ls['t']\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['UPBEAT third','GUARDED third'], [up, gd], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('post-call drift (CAR) %')\n"
            "ax.set_title(f'Upbeat +{up:.1f}%  vs  Guarded {gd:.1f}%  ->  gap {spr:+.1f}% (t {t:+.1f})')\n"
            "fig.suptitle('synthetic panel (480 calls) — looks like free money', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'upbeat +{up:.2f}% | guarded {gd:.2f}% | gap {spr:+.2f}% (two-sample t {t:+.2f})')"
        ),
        md(
            f"A **+{R['ls_spread']}%** gap with a *t* around {R['ls_t']:.0f} — that looks like a slam "
            "dunk. But remember: the upbeat calls are *also* the ones that beat on the number. So how "
            "much of this is the words, and how much is just the earnings surprise drifting? Let's "
            "strip the number out."
        ),
        code(
            "naive = st.tone_regression(PANEL, control_surprise=False)\n"
            "ctrl  = st.tone_regression(PANEL, control_surprise=True)\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['tone ALONE\\n(naive)','tone AFTER subtracting\\nthe number'],\n"
            "       [naive['tone_t'], ctrl['tone_t']], color=[RED, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar')\n"
            "ax.set_ylabel('tone slope t-stat'); ax.legend()\n"
            "ax.set_title(f\"The tone edge shrinks once you subtract the number ({naive['tone_t']:.1f} -> {ctrl['tone_t']:.1f})\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"naive tone-t {naive['tone_t']:.2f}  ->  surprise-controlled tone-t {ctrl['tone_t']:.2f}\")"
        ),
        md(
            f"The naive tone signal (*t* {R['naive_t']}) shrinks to *t* {R['ctrl_t']} once we control "
            "for the earnings number — a big chunk of the 'tone edge' was really the number drifting. "
            "In *this* world a genuine linguistic residual survives (it was planted). **But watch what "
            "happens when there is truly no tone effect at all:**"
        ),
        code(
            "pnull, _ = data.synthetic_panel(tone_beta=0.0, seed=566)   # tone adds NOTHING here\n"
            "nn = st.tone_regression(pnull, control_surprise=False)['tone_t']\n"
            "cc = st.tone_regression(pnull, control_surprise=True)['tone_t']\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "ax.bar(['naive\\n(fooled)','controlled\\n(honest)'], [nn, cc], color=[RED, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('tone slope t-stat')\n"
            "ax.set_title(f'NULL world (tone truly useless): naive still says t={nn:.1f}, honest says t={cc:.1f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null world -> naive tone-t {nn:+.2f} (FALSE positive) | controlled {cc:+.2f} (correct)')"
        ),
        md(
            f"There's the trap. In a world where tone tells you **nothing** beyond the number, the naive "
            f"reading *still* prints *t* ≈ +{R['null_naive_t']} — a confident **false positive** — while "
            f"the honest, number-controlled reading sits at ≈ {R['null_ctrl_t']}, correctly flat. Anyone "
            "who scores transcripts and skips this control is measuring earnings drift and calling it "
            "'tone'."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The literature says call tone predicts drift (after controlling for "
            "the number), and our engine banks it when it's planted — but there is **no free real tape** "
            "to certify it, so it's capped at Weak.\n"
            "- **Tradability — Fragile.** The tradable version needs vendor-grade transcript scoring and "
            "clean event-time returns to separate the words from the number; the honest residual is thin.\n"
            "- **Confounded by the number? — Named.** Tone and the earnings surprise are entangled; the "
            "naive read is a false positive at the null. This confound is stated up front."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Barely. The naive long-short looks rich, but it's mostly earnings drift you could get more "
            "cheaply from the number itself. The *pure* tone edge is small and needs expensive, "
            "carefully-scored transcripts to isolate — plus you'd be *shorting* the guarded names, which "
            "are exactly the ones already falling and costly to borrow. Fragile at best."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The number it rides on.** [363 PEAD-drift](../../363-pead-drift/) and "
            "[534 revenue-surprise-drift](../../534-revenue-surprise-drift/) are the *numeric* drifts this "
            "linguistic cousin sits on top of.\n"
            "- **The real test needs real transcripts.** Score a genuine transcript corpus (Loughran-McDonald "
            "lexicon), join to event-time CAR, and run the *surprise-controlled* slope — that's the only "
            "way to earn a `REAL` stamp here.\n\n"
            "*Have a scored-transcript tape? Drop it in `_cache/` in the study schema and re-run — the "
            "engine will read it and the notebook will light up the real numbers.*"
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
            "# Earnings-Call Tone — a quantitative teardown 🔬\n"
            "### net-tone → post-call CAR · naive vs surprise-controlled slope · the false-positive at the null · label-shuffle placebo · tone long-short with costs & borrow · four-window sweep · synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Confounded_by_the_number%3F: Named](https://img.shields.io/badge/Confounded_by_the_number%3F-Named-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We regress post-call CAR on net "
            "tone, control for the numeric surprise to isolate the **linguistic** edge, and show that "
            "skipping that control manufactures a false positive.\n\n"
            "> ⚠️ **Not investment advice.** **Synthetic-only** study: no free feed of scored "
            f"transcripts × event-time CAR exists, so the panel is deterministic (seed 566, planted "
            f"`tone_beta={R['tone_beta']}`, 480 calls, panel fp `{R['fp_panel']}`) and the real fetch is "
            "an empty stub — the Signal axis is **capped at `WEAK`**. Methods in "
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
            f"| **Signal** | `WEAK` | Surprise-controlled tone slope-*t* **+{R['ctrl_t']}** (placebo *p* "
            f"{R['placebo_p']}) on the planted panel, flat **{R['null_ctrl_t']}** at the null — but **no "
            "real tape** exists, so `REAL` (robust *t* ≥ 2 on real data) is unreachable → capped `WEAK`. |\n"
            f"| **Tradability** | `FRAGILE` | Naive tone long-short gross **+{R['gross_ann']}%/yr** → net "
            f"**+{R['net_ann']}%/yr**, but it is PEAD-inflated; the *isolated* edge needs vendor-grade "
            "scoring to capture. |\n"
            f"| **Confounded by the number?** | `NAMED` | At the null the *naive* slope-*t* averages "
            f"**+{R['null_naive_t']}** (false positive) vs controlled **{R['null_ctrl_t']}**. |\n\n"
            "> 💡 In plain words: the engine is faithful (the synthetic control proves it), but a `REAL` "
            "stamp needs a real tape we don't have, and the naive read is a confound — hence `WEAK` × "
            "`FRAGILE`, confound `NAMED`."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $T_i$ be call $i$'s net tone, $S_i$ its numeric earnings surprise, and $C_i$ its "
            "post-call CAR.\n\n"
            "- **H₁ (naive).** slope of $C$ on $T$ alone $> 0$ at *t* ≥ 2.\n"
            "- **H₂ (the real claim).** slope of $C$ on $T$ **controlling for $S$** $> 0$ at *t* ≥ 2 — "
            "tone adds information *beyond the number*.\n"
            "- **H₃ (robustness).** the controlled sign holds across sub-panels.\n"
            "- **H₄ (net).** a tradable tone long-short survives costs + the guarded-leg borrow.\n\n"
            "The honest test is **H₂**. H₁ is a trap: because $\\text{corr}(T,S) > 0$, H₁ can be "
            "'confirmed' even when tone is useless (H₂ false). We show exactly that below."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is **partialling-out**. Upbeat calls follow beats, so $T$ and $S$ are "
            "collinear; the naive slope of $C$ on $T$ absorbs the PEAD coefficient on $S$. Only the "
            "*controlled* tone coefficient (Frisch-Waugh-Lovell) isolates the linguistic channel. This "
            "maps to the desk's numeric drifts — [363 PEAD](../../363-pead-drift/), "
            "[534 revenue-surprise](../../534-revenue-surprise-drift/) — which are the $S$ that $T$ must "
            "beat, and is distinct from the aggregate-mood studies "
            "([259 news-tone](../../259-news-tone/), [392 glassdoor](../../392-glassdoor-sentiment/))."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Score.** Net tone $T$ = (positive − negative) sentiment-word share (Loughran-McDonald "
            "lexicon); numeric surprise $S$ = standardised SUE.\n"
            "- **Outcome.** Post-call CAR $C$ (market-adjusted return *after* the call).\n"
            "- **Naive slope.** OLS $C \\sim T$ — the PEAD-inflated read (H₁).\n"
            "- **Controlled slope.** OLS $C \\sim [T, S]$ — the *tone* coefficient is the isolated edge (H₂).\n"
            "- **Placebo.** Shuffle $T$ against $(C, S)$, refit the controlled slope, read the tail (2000 perms).\n"
            "- **Long-short.** Long the upbeat tone tail, short the guarded tail; costs + borrow.\n"
            "- **Robustness.** Controlled slope-*t* over four contiguous sub-panels.\n"
            "- **Positive control.** Deterministic worlds with planted `tone_beta` (0 = null), averaged "
            "over 25 seeds (house rule).\n\n"
            "Timing: tone is known at the call; CAR is measured strictly after; entry is the next "
            "session. One documented execution lag, applied identically in the sweep and the control."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · Naive vs surprise-controlled slope — H₁ vs H₂\n\n"
            "The whole study in one cell: the tone slope before and after subtracting the number."
        ),
        code(
            "naive = st.tone_regression(PANEL, control_surprise=False)\n"
            "ctrl  = st.tone_regression(PANEL, control_surprise=True)\n"
            "p = st.placebo_pvalue(PANEL, n_perm=2000, seed=566)\n"
            "tab = pd.DataFrame({\n"
            "    'regression': ['C ~ tone (naive)', 'C ~ [tone, surprise] (controlled)'],\n"
            "    'tone_slope': [naive['tone_slope'], ctrl['tone_slope']],\n"
            "    'tone_t': [naive['tone_t'], ctrl['tone_t']],\n"
            "}).set_index('regression')\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(range(2), tab['tone_t'], color=[RED, GREEN], width=.5)\n"
            "ax.set_xticks(range(2)); ax.set_xticklabels(['naive','controlled'])\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar'); ax.legend()\n"
            "ax.set_ylabel('tone slope t-stat')\n"
            "ax.set_title(f\"Tone slope-t: naive {naive['tone_t']:.1f} -> controlled {ctrl['tone_t']:.1f} (placebo p {p:.4f})\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(4).to_string()); print('placebo p (controlled slope):', round(p,4))"
        ),
        md(
            f"> 💡 In plain words: H₁ 'passes' (naive *t* +{R['naive_t']}) but is inflated by PEAD; **H₂ "
            f"is the honest one** — controlled tone-*t* **+{R['ctrl_t']}** (placebo *p* {R['placebo_p']}). "
            "On this planted world a genuine linguistic residual survives; the placebo confirms it sits "
            "in the tail, not the bulk."
        ),
        md(
            "### 4b · The false-positive at the null — why the control is not optional\n\n"
            "Rebuild the world with `tone_beta = 0` (tone adds *nothing* beyond the number) and rerun "
            "both slopes."
        ),
        code(
            "pnull, _ = data.synthetic_panel(tone_beta=0.0, seed=566)\n"
            "nn = st.tone_regression(pnull, control_surprise=False)['tone_t']\n"
            "cc = st.tone_regression(pnull, control_surprise=True)['tone_t']\n"
            "# average over 25 seeds so it's not one lucky draw\n"
            "nn_bar = st.synthetic_mean_t(data, 0.0, n_seeds=25, control_surprise=False)\n"
            "cc_bar = st.synthetic_mean_t(data, 0.0, n_seeds=25, control_surprise=True)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['naive\\n(this seed)','controlled\\n(this seed)','naive\\n(25-seed mean)','controlled\\n(25-seed mean)'],\n"
            "       [nn, cc, nn_bar, cc_bar], color=[RED, GREEN, RED, GREEN], width=.6)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('tone slope t-stat'); ax.set_title('NULL world: naive is a confident false positive; controlled is flat')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null: naive t {nn:+.2f} (mean {nn_bar:+.2f}) | controlled t {cc:+.2f} (mean {cc_bar:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: with tone genuinely useless, the naive slope-*t* averages "
            f"**+{R['null_naive_t']}** across 25 seeds — a rock-solid **false positive** — while the "
            f"controlled slope-*t* averages **{R['null_ctrl_t']}**, correctly flat. The surprise control "
            "is the difference between measuring tone and re-labelling PEAD."
        ),
        md(
            "### 4c · Robustness — H₃ (does the controlled sign hold?)\n\n"
            "The surprise-controlled slope over four contiguous sub-panels of quarters."
        ),
        code(
            "rows = st.robustness_windows(PANEL, n_windows=4)\n"
            "tab = pd.DataFrame(rows, columns=['window','ctrl_slope','ctrl_t']).set_index('window')\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "cols = [GREEN if t>0 else RED for t in tab['ctrl_t']]\n"
            "ax.bar(range(len(tab)), tab['ctrl_t'], color=cols, width=.55)\n"
            "ax.set_xticks(range(len(tab))); ax.set_xticklabels(tab.index)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar'); ax.axhline(0, c='k', lw=1); ax.legend()\n"
            "ax.set_ylabel('controlled tone slope t'); ax.set_title('Controlled sign holds across sub-panels (planted world)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(3).to_string())"
        ),
        md(
            "> 💡 In plain words: H₃ holds *when the effect is truly planted* — the controlled slope is "
            "positive with *t* > 3 in all four sub-panels. On a real tape this sign-stability is exactly "
            "the open question this desk can't resolve without the data."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — H₂ holds on the planted panel (controlled *t* +{R['ctrl_t']}, placebo "
            f"*p* {R['placebo_p']}), but **no real tape** exists → `REAL` unreachable, capped `WEAK`.\n"
            f"- **Tradability `FRAGILE`** — naive long-short net +{R['net_ann']}%/yr is PEAD-inflated; the "
            "isolated edge needs vendor-grade scoring to capture, with a borrow on the guarded short.\n"
            f"- **Confounded by the number? `NAMED`** — naive slope-*t* +{R['null_naive_t']} at the null "
            "is a false positive; only the controlled slope isolates the words."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the long-short, costs and the borrow\n\n"
            "The naive tone long-short, annualised, net of quarterly turnover and a guarded-leg borrow."
        ),
        code(
            "ls = st.tone_long_short(PANEL, frac=0.3)\n"
            "gross = ls['spread']*4.0*100\n"
            "net = st.net_spread(ls, turns_per_year=4.0, cost_bps=5.0, borrow_ann_bps=50.0)*100\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross\\n(~4 calls/yr)','net\\n(costs+borrow)'], [gross, net], color=[AMBER, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised tone long-short %')\n"
            "ax.set_title(f'Naive long-short: gross {gross:+.0f}% -> net {net:+.0f}% (PEAD-inflated)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:+.1f}% -> net {net:+.1f}% (4 turns/yr x 5 bps/leg + 50 bps borrow)')"
        ),
        md(
            f"> 💡 In plain words: net **+{R['net_ann']}%/yr** *looks* investable — but this is the "
            "**naive** book, so it's mostly earnings drift you could harvest more cheaply from the number. "
            "The pure tone residual is thin and vendor-dependent. `FRAGILE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the controlled engine a faithful detector, or does it always print ~0? Plant a linguistic "
            "drift of increasing strength (`tone_beta`) and watch the controlled slope-*t* climb — "
            "averaged over 25 seeds so no single lucky seed can fake it."
        ),
        code(
            "betas = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]\n"
            "ts = [st.synthetic_mean_t(data, tone_beta=b, n_seeds=25) for b in betas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(betas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted tone_beta (stronger linguistic drift ->)')\n"
            "ax.set_ylabel('mean controlled tone slope-t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted linguistic edge — flat at the null, past +2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for b, t in zip(betas, ts): print(f'tone_beta {b:+.3f} -> mean controlled slope-t {t:+.2f}')"
        ),
        md(
            "The controlled slope-*t* is ≈ 0 at the null and climbs monotonically past +2 as the planted "
            "linguistic drift grows — the detector is faithful. So the honest limitation is **data "
            "availability**, not a broken engine: with no free tape of scored transcripts × event-time "
            "CAR, this desk can't lift the claim above `WEAK`. For the numeric drifts it rides on, see "
            "[363 PEAD-drift](../../363-pead-drift/) and "
            "[534 revenue-surprise-drift](../../534-revenue-surprise-drift/)."
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
