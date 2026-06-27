"""Generate the two narrative notebooks for Study 540 (Distress-Risk-Anomaly).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic figures
run anywhere, offline and deterministic; the real-tape cells use the cached prices + fundamentals
under ../_cache/ if present and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md), so the notebook re-runs for any reader offline.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-26; scored
# 2024-06-28, forward to 2026-06-25; 39 survivors; panel fp 0835f4217788).
R = dict(
    fp_px="9bf6c3a922f0", fp_fn="5f664306cf34", fp_panel="0835f4217788",
    n=39, k=12, split="2024-06-28", end="2026-06-25",
    safe_mean=34.1, distressed_mean=82.7, spread=-48.6, ls_t=-2.64, placebo_p=0.033,
    slope=11.0, slope_t=1.99, corr=0.31,
    net=-50.8, cost_bps=5.0, borrow_bps=100.0,
    # robustness windows: (label, spread%, slope_t)
    win=[("2021-06 -> 2023-06", 22.0, -2.02), ("2022-06 -> 2024-06", 26.9, -1.46),
         ("2023-06 -> 2025-06", -45.6, 0.42), ("2024-06 -> 2026-06", -48.6, 1.99)],
    # synthetic control: (alpha, mean slope-t over 25 seeds)
    ctrl=[(0.0, -0.04), (-0.06, -0.87), (-0.12, -1.69), (-0.20, -2.78)],
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

from distress_risk_anomaly import data, strategy as st

SPLIT, END = "2024-06-28", "2026-06-25"

def load_real():
    \"\"\"Cache-first real cross-section (empty frame offline).\"\"\"
    px = data.fetch_prices(fetch=False)
    fn = data.fetch_fundamentals(fetch=False)
    if px.empty or not fn:
        return pd.DataFrame()
    return st.build_panel(px, fn, split_date=SPLIT, end_date=END)

PANEL = load_real()
HAVE_REAL = not PANEL.empty
print("real distress panel present:", HAVE_REAL,
      "" if not HAVE_REAL else f"({len(PANEL)} names, scored {SPLIT} -> {END})")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Distress Puzzle — do the riskiest firms earn the *least*? 💀\n"
            "### Sorting blue chips by how 'distressed' they look, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Distress_puzzle_on_the_tape%3F: Mixed](https://img.shields.io/badge/Distress_puzzle_on_the_tape%3F-Mixed-8b949e?style=flat-square)\n\n"
            "Here's one of finance's strangest findings. A famous 2008 study "
            "(Campbell, Hilscher & Szilagyi) built a model to predict which companies are most likely "
            "to **go bankrupt** — and then found that those most-likely-to-fail firms went on to earn "
            "the **lowest** stock returns. That's backwards: you'd expect to be *paid* for holding "
            "risky, fragile companies. Instead, holding them lost you money. It's called the "
            "**distress puzzle**.\n\n"
            "So we test it the cheap way: take 39 big-name stocks, score each one for 'distress' "
            "(lots of debt, weak profits, jumpy share price), and see whether the *safe* ones beat "
            "the *distressed* ones.\n\n"
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
            f"| Did the *distressed* firms earn less (the puzzle)? | **No — the opposite.** Over "
            f"2024-26 the distressed bucket earned **+{R['distressed_mean']}%** vs the safe bucket's "
            f"**+{R['safe_mean']}%**. The risky names roughly *doubled* the safe ones. |\n"
            f"| Is that just luck? | **No** — a shuffle test puts the odds at *p* = {R['placebo_p']}. "
            "On this window the *anti*-puzzle is real. |\n"
            f"| So is the puzzle fake? | **It depends when you look.** In two earlier windows the "
            "puzzle *did* show up; in the two recent ones it flipped. The sign isn't stable. |\n"
            "| Could you trade it? | **No.** A signal that flips sign with the window, on a basket "
            "that has *already* dropped every company that went bust, isn't a trade. |\n\n"
            "> The puzzle is real in the textbooks — but the textbook version needs the firms that "
            "*actually failed*. Our basket only has survivors, and 2024-26 was a melt-up that "
            "rewarded exactly the debt-heavy, jumpy names the puzzle says should lose."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"The most financially distressed firms earn anomalously low returns.\"* — the "
            "distress puzzle (Campbell, Hilscher & Szilagyi 2008; Dichev 1998).\n\n"
            "The intuition for why it's a *puzzle*: risk is supposed to be rewarded. A company on the "
            "edge of bankruptcy is risky, so its stock should be cheap and earn high returns to "
            "compensate. Instead, the data says the opposite — distressed stocks *underperform*. "
            "Maybe investors gamble on lottery-like distressed names and overpay; maybe the losers "
            "just keep losing. Either way, the sign is backwards from the textbook."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If true, it breaks the most basic rule everyone learns: more risk → more reward. And it "
            "suggests an easy trade — *own the safe firms, avoid (or short) the distressed ones*. "
            "The desk has already looked at the two famous bankruptcy **scores** "
            "([123 Altman-Z](../../123-altman-z/), [230 Ohlson-O](../../230-ohlson-o-score/)); here "
            "we go straight at the **return puzzle** itself: build a distress score, sort, and look "
            "at who won."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Score distress.** For each firm: a lot of debt (leverage), weak profits (low return "
            "on assets) and a jumpy share price (high volatility) all mean *more* distressed. Add "
            "them up.\n"
            "2. **Sort.** Split the 39 names into a *safe* third and a *distressed* third.\n"
            "3. **Compare forward returns.** Did the safe third beat the distressed third over the "
            "next two years? The puzzle says yes (safe wins).\n\n"
            "Catch: our basket is companies **still alive in 2026** — so the ones that actually went "
            "bankrupt (the heart of the real effect) are missing. We'll see that bias bite.\n\n"
            "*Timing:* the score uses only data up to the scoring day's close (trailing volatility + "
            "the last statements), so it's fully known at that close; we enter there and hold two "
            "years. No peeking at the future."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Who won — the safe third or the distressed third?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = st.bucket_returns(PANEL, frac=0.3); ls = st.long_short_tstat(b)\n"
            "    safe_m, dist_m, spr, t = b['safe_mean']*100, b['distressed_mean']*100, b['spread']*100, ls['t']\n"
            "    banner = 'REAL tape (39 survivors, 2024-06 -> 2026-06)'\n"
            "else:\n"
            f"    safe_m, dist_m, spr, t = {R['safe_mean']}, {R['distressed_mean']}, {R['spread']}, {R['ls_t']}\n"
            "    banner = 'frozen real numbers (offline)'\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['SAFE third','DISTRESSED third'], [safe_m, dist_m], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'The puzzle says SAFE wins. Here the DISTRESSED third won (+{dist_m:.0f}% vs +{safe_m:.0f}%)')\n"
            "fig.suptitle(banner, fontsize=9, color=GREY); plt.tight_layout(); plt.show()\n"
            "print(f'Safe +{safe_m:.1f}%  vs  Distressed +{dist_m:.1f}%  ->  spread {spr:+.1f}% (t {t:+.2f})')"
        ),
        md(
            f"There it is, **backwards**. The puzzle predicts the *safe* names win. On 2024-26 the "
            f"**distressed** names won by miles (+{R['distressed_mean']}% vs +{R['safe_mean']}%) — the "
            f"long-safe/short-distressed spread is **{R['spread']}%**, a clean loss the *wrong* way. "
            "Why? The 'distressed' bucket here is banks and the high-volatility AI winners — and they "
            "led a melt-up."
        ),
        md(
            "**Is the puzzle just *gone*, or does it depend on when you look?** Run the same sort over "
            "four different two-year windows:"
        ),
        code(
            "wins = " + repr([(w[0], w[1]) for w in R["win"]]) + "\n"
            "if HAVE_REAL:\n"
            "    px = data.fetch_prices(fetch=False); fn = data.fetch_fundamentals(fetch=False)\n"
            "    spans = [('2021-06-28','2023-06-28'),('2022-06-28','2024-06-28'),\n"
            "             ('2023-06-28','2025-06-25'),('2024-06-28','2026-06-25')]\n"
            "    labs = ['2021-23','2022-24','2023-25','2024-26']\n"
            "    spr = []\n"
            "    for (sp,en) in spans:\n"
            "        pn = st.build_panel(px, fn, split_date=sp, end_date=en)\n"
            "        spr.append(st.bucket_returns(pn, 0.3)['spread']*100 if not pn.empty else np.nan)\n"
            "else:\n"
            "    labs = ['2021-23','2022-24','2023-25','2024-26']; spr = [w[1] for w in wins]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "cols = [GREEN if s>0 else RED for s in spr]\n"
            "ax.bar(labs, spr, color=cols, width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('safe - distressed spread %')\n"
            "ax.set_title('The puzzle (green, safe wins) shows up early, then FLIPS (red)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('spreads by window:', [round(s,1) for s in spr])"
        ),
        md(
            "So the sign isn't stable. In **2021-23** and **2022-24** the puzzle *appears* — the safe "
            "names win (+22%, +27%). In **2023-25** and **2024-26** it **flips** — the distressed "
            "names win. A 'signal' whose direction changes with the calendar is not something you can "
            "bet on."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** On the headline window the safe-minus-distressed spread is the "
            f"*wrong* sign ({R['spread']}%, *t* {R['ls_t']}), and it flips across windows. No bankable "
            "puzzle here.\n"
            "- **Tradability — Mirage.** A survivor basket, annual rebalance, with the distressed leg "
            "being the costly-to-borrow tail. Nothing to harvest.\n"
            "- **Distress puzzle on the tape? — Mixed.** It *does* appear in two of four windows and "
            "vanishes in the other two — survivorship and a high-beta melt-up explain the rest."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No. Even setting aside the wrong sign, the trade would mean *shorting* the most "
            "distressed names — exactly the stocks that are expensive and hard to borrow — on a "
            "basket that has already quietly deleted every company that went bankrupt. The real "
            "puzzle lives in the firms that *failed*; a blue-chip survivor sort can't see them."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The scores it's built on.** [123 Altman-Z](../../123-altman-z/) and "
            "[230 Ohlson-O](../../230-ohlson-o-score/) are the static bankruptcy *classifiers*; this "
            "study is the *return* puzzle.\n"
            "- **The real test needs the dead.** Re-run this with delisted/bankrupt firms included "
            "(a survivorship-free universe) and a full cycle — that's where the puzzle earns its "
            "keep.\n\n"
            "*Think the puzzle is real and we just missed it? Fork this, add bankrupt names and a "
            "2008-2009 window, and show the safe-minus-distressed spread clearing t = 2 the right "
            "way.*"
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
            "# The Distress Puzzle — a quantitative teardown 🔬\n"
            "### CHS-style distress score · tercile sort · two-sample *t* · label-shuffle placebo · firm-level slope · four-window sign-flip · costs & borrow · synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Distress_puzzle_on_the_tape%3F: Mixed](https://img.shields.io/badge/Distress_puzzle_on_the_tape%3F-Mixed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We build a "
            "Campbell-Hilscher-Szilagyi-style distress score, sort the cross-section, and test whether "
            "the most-distressed firms earn the lowest returns (the puzzle) — or, here, the highest.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance daily prices + a fundamentals "
            f"snapshot, 39 survivors (as-of 2026-06-26, panel fp `{R['fp_panel']}`); the offline core "
            "and tests run on a deterministic synthetic world. Methods in "
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
            f"| **Signal** | `NONE` | Safe − distressed spread **{R['spread']}%** (two-sample *t* "
            f"**{R['ls_t']}**, placebo *p* {R['placebo_p']}) — the *wrong* sign; firm slope-*t* "
            f"**+{R['slope_t']}** (positive = anti-puzzle); sign flips across windows. |\n"
            f"| **Tradability** | `MIRAGE` | Survivor basket, annual rebalance, distressed short = "
            f"costly borrow. Gross {R['spread']}% → net {R['net']}% (5 bps/leg + {R['borrow_bps']:.0f} "
            "bps borrow). |\n"
            f"| **Distress puzzle on the tape?** | `MIXED` | Present (slope-*t* −2.02 / −1.46) in "
            "2021-23 & 2022-24; inverted in 2023-25 & 2024-26. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), but on this "
            "survivor basket over this window the puzzle is absent-to-inverted and sign-unstable."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $D_i$ be firm $i$'s CHS-style distress score and $r_i$ its forward return.\n\n"
            "- **H₁ (the puzzle / long-short).** $\\mathbb{E}[r \\mid \\text{safe}] - "
            "\\mathbb{E}[r \\mid \\text{distressed}] > 0$ at *t* ≥ 2 (safe beats distressed).\n"
            "- **H₂ (firm-level).** $\\text{slope of } r \\text{ on } D < 0$ (more distress, less "
            "return).\n"
            "- **H₃ (robustness).** The sign of H₁/H₂ is stable across scoring windows.\n"
            "- **H₄ (net).** H₁ survives costs and the distressed-leg borrow.\n\n"
            "We find **H₁ rejected** (wrong sign on the headline window), **H₂ rejected** (slope is "
            "*positive*), **H₃ rejected** (sign flips), and **H₄ moot** (the trade is the wrong sign "
            "before costs)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why**. The puzzle is well documented on full-universe, "
            "survivorship-free data driven by firms that *fail*. Two forces kill it here: (a) "
            "**survivorship** — the basket already dropped every bankruptcy, removing the extreme "
            "left tail that makes distressed stocks lose on average; and (b) a **high-beta melt-up** "
            "(2023-26) that rewarded the levered, volatile names the score flags as distressed. That "
            "maps to [123 Altman-Z](../../123-altman-z/)'s weak distress reading on the same kind of "
            "survivor panel."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Score.** $D = z(\\text{leverage}) - z(\\text{ROA}) + z(\\text{equity vol})$, each "
            "z-scored across the basket (higher = more distressed). The three observable CHS legs.\n"
            "- **Sort.** Tercile tails (≈12 names each): safe (lowest $D$) vs distressed (highest).\n"
            "- **Inference.** Two-sample (Welch) *t* on the safe − distressed forward returns; a "
            "**label-shuffle placebo** null (2000 perms).\n"
            "- **Firm-level.** OLS of forward return on $D$ — the *sign* of the slope is the puzzle.\n"
            "- **Robustness.** Re-sort over four scoring windows; read the sign.\n"
            "- **Frictions.** Round-trip cost per leg + an annual borrow on the distressed short.\n"
            "- **Positive control.** A deterministic synthetic world with a planted puzzle "
            "(`distress_alpha < 0`) and a null — averaged over 25 seeds (house rule).\n\n"
            "Timing: distress is scored from trailing data (the realised-vol window ending at the "
            "split close) + the last statements, so the signal is fully public at that close; we "
            "enter at that close and hold over the forward window. No future data enters the score "
            "(no look-ahead). The same-close fill is a documented convention — on a ~2-year hold a "
            "one-bar-later entry moves the headline spread ~1pt and changes no stamp."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The sort — H₁\n\n"
            "Safe vs distressed forward returns, with the two-sample *t* and the placebo null."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = st.bucket_returns(PANEL, 0.3); ls = st.long_short_tstat(b)\n"
            "    p = st.placebo_pvalue(PANEL, n_perm=2000, seed=540)\n"
            "    safe_m, dist_m, spr, t = b['safe_mean']*100, b['distressed_mean']*100, b['spread']*100, ls['t']\n"
            "else:\n"
            f"    safe_m, dist_m, spr, t, p = {R['safe_mean']}, {R['distressed_mean']}, {R['spread']}, {R['ls_t']}, {R['placebo_p']}\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['SAFE','DISTRESSED'], [safe_m, dist_m], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'Safe - distressed = {spr:+.1f}%  (two-sample t {t:+.2f}, placebo p {p:.3f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Safe +{safe_m:.1f}% | Distressed +{dist_m:.1f}% | spread {spr:+.1f}% | t {t:+.2f} | placebo p {p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: H₁ **rejected, with the sign reversed.** The spread is **{R['spread']}%** "
            f"(*t* {R['ls_t']}), and the placebo *p* = {R['placebo_p']} says the inversion is real on "
            "this window — the distressed names genuinely *out*-earned the safe ones."
        ),
        md(
            "### 4b · The firm-level slope — H₂\n\n"
            "Regress forward return on the distress score across all 39 names. A *negative* slope "
            "would be the puzzle."
        ),
        code(
            "if HAVE_REAL:\n"
            "    reg = st.cross_section_regression(PANEL)\n"
            "    d = st.distress_score(PANEL); y = PANEL['forward_ret']*100\n"
            "    slope, slope_t, corr = reg['slope']*100, reg['slope_t'], reg['corr']\n"
            "    fig, ax = plt.subplots(figsize=(8, 4.6))\n"
            "    ax.scatter(d, y, s=28, color=GREY)\n"
            "    xs = np.linspace(d.min(), d.max(), 50)\n"
            "    ax.plot(xs, (reg['slope']*xs + (y.mean()/100 - reg['slope']*d.mean()))*100, c=RED, lw=2)\n"
            "    ax.set_xlabel('distress score (higher = more distressed)'); ax.set_ylabel('forward return %')\n"
            "    ax.set_title(f'Slope {slope:+.1f}%/unit (t {slope_t:+.2f}) — POSITIVE = anti-puzzle')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            f"    slope, slope_t, corr = {R['slope']}, {R['slope_t']}, {R['corr']}\n"
            "print(f'slope {slope:+.1f}%/unit, slope-t {slope_t:+.2f}, corr(distress, fwd) {corr:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: H₂ **rejected** — the slope is **+{R['slope']}%** per distress unit "
            f"(*t* +{R['slope_t']}), the *opposite* sign to the puzzle. More distress went with *more* "
            f"return (corr +{R['corr']}) on this window."
        ),
        md(
            "### 4c · Robustness — H₃ (does the sign hold?)\n\n"
            "The same sort over four scoring windows."
        ),
        code(
            "wins = " + repr(R["win"]) + "\n"
            "if HAVE_REAL:\n"
            "    px = data.fetch_prices(fetch=False); fn = data.fetch_fundamentals(fetch=False)\n"
            "    spans = [('2021-06-28','2023-06-28'),('2022-06-28','2024-06-28'),\n"
            "             ('2023-06-28','2025-06-25'),('2024-06-28','2026-06-25')]\n"
            "    rows = []\n"
            "    for (lab,_,_),(sp,en) in zip(wins, spans):\n"
            "        pn = st.build_panel(px, fn, split_date=sp, end_date=en)\n"
            "        bb = st.bucket_returns(pn, 0.3); rr = st.cross_section_regression(pn)\n"
            "        rows.append((lab, bb['spread']*100, rr['slope_t']))\n"
            "else:\n"
            "    rows = wins\n"
            "tab = pd.DataFrame(rows, columns=['window','spread%','slope_t']).set_index('window')\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "cols = [GREEN if s>0 else RED for s in tab['spread%']]\n"
            "ax.bar(range(len(tab)), tab['spread%'], color=cols, width=.55)\n"
            "ax.set_xticks(range(len(tab))); ax.set_xticklabels(tab.index, rotation=15)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('safe - distressed spread %')\n"
            "ax.set_title('Sign flips: puzzle present (green) early, inverted (red) recently')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(2).to_string())"
        ),
        md(
            "> 💡 In plain words: H₃ **rejected.** Spread **+22% / +27%** (puzzle present, slope-*t* "
            "−2.02 / −1.46) in 2021-23 & 2022-24, then **−46% / −49%** (inverted) in 2023-25 & "
            "2024-26. A sign that depends on the window is not a signal."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — H₁ rejected (spread {R['spread']}%, *t* {R['ls_t']}, wrong sign); "
            f"H₂ rejected (slope *t* +{R['slope_t']}); H₃ rejected (sign flips).\n"
            f"- **Tradability `MIRAGE`** — survivor basket, annual rebalance, distressed-leg borrow; "
            f"gross {R['spread']}% → net {R['net']}%.\n"
            "- **Distress puzzle on the tape? `MIXED`** — present in 2 of 4 windows, inverted in the "
            "other 2."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "The trade is the wrong sign before costs; the borrow is a footnote."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = st.bucket_returns(PANEL, 0.3)\n"
            "    gross = b['spread']*100\n"
            "    net = st.net_spread(b, cost_bps=5.0, borrow_ann_bps=100.0, holding_years=2.0)*100\n"
            "else:\n"
            f"    gross, net = {R['spread']}, {R['net']}\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross','net\\n(costs+borrow)'], [gross, net], color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('safe - distressed spread %')\n"
            "ax.set_title(f'Wrong sign before costs ({gross:+.0f}%), worse after ({net:+.0f}%)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:+.1f}% -> net {net:+.1f}% (5 bps/leg + 100 bps borrow, 2y hold)')"
        ),
        md(
            "> 💡 In plain words: there is nothing to charge costs against — the long-safe/short-distressed "
            "book *loses* before frictions on this window. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a real puzzle "
            "(`distress_alpha < 0`) of increasing strength and watch the firm-level slope-*t* go "
            "negative — averaged over 25 seeds so no single lucky seed can fake it."
        ),
        code(
            "alphas = [0.0, -0.04, -0.08, -0.12, -0.16, -0.20]\n"
            "ts = [st.synthetic_mean_t(data, distress_alpha=a, n_seeds=25) for a in alphas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot([a for a in alphas], ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1, label='t = -2 bar (puzzle)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted distress_alpha (more negative = stronger puzzle)')\n"
            "ax.set_ylabel('mean firm slope-t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted puzzle — flat at the null, past -2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for a, t in zip(alphas, ts): print(f'alpha {a:+.2f} -> mean slope-t {t:+.2f}')"
        ),
        md(
            "The slope-*t* is ≈ 0 at the null and falls past −2 as the planted puzzle grows — so the "
            "engine is a faithful detector. The real-tape result is therefore a statement about **this "
            "survivor basket on this window**: the CHS distress puzzle is absent-to-inverted and "
            "sign-unstable here. For the bankruptcy *scores* it builds on, see "
            "[123 Altman-Z](../../123-altman-z/) and [230 Ohlson-O](../../230-ohlson-o-score/)."
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
