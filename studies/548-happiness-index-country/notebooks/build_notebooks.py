"""Generate the two narrative notebooks for Study 548 (Happiness-Index-Country).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic figures run
anywhere, offline and deterministic; the real-tape cells use the cached country-ETF panel under
../_cache/ if present and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-30; scored
# 2024-03-20, forward to 2026-06-26; 24 investable countries; panel fp 90b4d9c25e41).
R = dict(
    fp_px="483868490082", fp_panel="90b4d9c25e41",
    n=24, k=7, split="2024-03-20", end="2026-06-26",
    happy_mean=44.6, gloomy_mean=87.0, spread=-42.4, ls_t=-1.65, placebo_p=0.066,
    rho=-0.19, rho_t=-0.90, slope=-69.1, slope_t=-1.92, corr=-0.38,
    net=-43.8, cost_bps=5.0, borrow_bps=60.0,
    # robustness by WHR edition: (label, spread%, slope_t, rho)
    win=[("2019-21", 24.4, 1.80, 0.45), ("2021-23", 2.7, 0.90, 0.03),
         ("2022-24", -7.1, 0.26, -0.23), ("2024-26", -42.4, -1.92, -0.19)],
    # synthetic control (40 seeds): (alpha, mean slope-t, mean spread-t)
    ctrl=[(0.00, -0.11, -0.13), (0.03, 0.33, 0.29), (0.06, 0.78, 0.71),
          (0.10, 1.38, 1.27), (0.15, 2.13, 1.96)],
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

from happiness_index_country import data, strategy as st

SPLIT, END = "2024-03-20", "2026-06-26"

def load_real():
    \"\"\"Cache-first real cross-section (empty frame offline).\"\"\"
    return data.fetch_panel(fetch=False, split_date=SPLIT, end_date=END)

PANEL = load_real()
HAVE_REAL = not PANEL.empty
print("real happiness panel present:", HAVE_REAL,
      "" if not HAVE_REAL else f"({len(PANEL)} countries, scored {SPLIT} -> {END})")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Happiness Trade — do the *happiest* countries' markets win? 😊\n"
            "### Sorting the world's stock markets by the World Happiness Report, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Spurious%3F: Busted](https://img.shields.io/badge/Spurious%3F-Busted-8b949e?style=flat-square)\n\n"
            "Every year the **World Happiness Report** ranks countries from happiest to gloomiest — the "
            "Nordics on top, as usual. Here's a tempting story: happy, well-run, optimistic societies "
            "*must* have better stock markets, right? So you could buy the happiest countries, avoid "
            "(or short) the gloomiest, and coast.\n\n"
            "We test it the honest way: take the 24 countries you can actually *buy* (each has a "
            "single-country exchange-traded fund), line them up by their happiness rank, and see "
            "whether the happy third beat the gloomy third.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the rank correlation and "
            "the placebo test? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Did the *happy* countries' markets win? | **No — the opposite.** Over 2024-26 the "
            f"gloomy third earned **+{R['gloomy_mean']}%** vs the happy third's **+{R['happy_mean']}%**. "
            "The gloomy markets roughly *doubled* the happy ones. |\n"
            f"| Is happiness even *linked* to returns? | **No** — the rank correlation is only "
            f"**{R['rho']}** (basically zero). |\n"
            f"| Is it just this window? | **Yes-ish, and that's the point.** In 2019-21 the happy "
            "markets *did* edge ahead; then it faded and flipped. The direction changes with the year. |\n"
            "| Could you trade it? | **No.** A signal built on ~24 countries that flips sign depending "
            "on which report you read is a coin toss, not an edge. |\n\n"
            "> This is a textbook **spurious correlation**: with only two dozen countries, some "
            "happiness-vs-returns pattern will always appear by luck — and vanish next window."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"The happiest nations have the best-performing stock markets.\"* — an alt-data folklore.\n\n"
            "The intuition sounds airtight: happier countries tend to be richer, better governed, more "
            "trusting, more stable — surely their companies do better and their markets reward you. "
            "It's the kind of story that makes a great chart and a terrible trade."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If true, it's an easy, feel-good global strategy: own the happy countries, skip the "
            "gloomy ones. The desk has poked at *mood* signals before "
            "([257 AAII-Sentiment](../../257-aaii-sentiment/), "
            "[335 Buzz-Sentiment-ETF](../../335-buzz-sentiment-etf/)) — but those time *one* market. "
            "This is different: a **cross-country** sort on a societal well-being index. It's also a "
            "clean lesson in how spurious correlations are born."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Rank by happiness.** Use the World Happiness Report 2024 rank (1 = happiest).\n"
            "2. **Keep the ones you can buy.** Only 24 countries have a liquid single-country ETF — "
            "that's the *investable* universe.\n"
            "3. **Sort & compare.** Split into a *happy* third and a *gloomy* third, and compare their "
            "market returns over the next ~2 years.\n\n"
            "Catch: 24 countries is *tiny*. With so few data points, almost any pattern can show up "
            "by chance — which is exactly what we'll see.\n\n"
            "*Timing:* the happiness rank is the report published *before* the holding window, so it's "
            "fully known when we 'buy'. No peeking at the future."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Who won — the happy third or the gloomy third?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = st.bucket_returns(PANEL, frac=0.3); ls = st.long_short_tstat(b)\n"
            "    happy_m, gloom_m, spr, t = b['happy_mean']*100, b['gloomy_mean']*100, b['spread']*100, ls['t']\n"
            "    banner = 'REAL tape (24 country ETFs, 2024-03 -> 2026-06)'\n"
            "else:\n"
            f"    happy_m, gloom_m, spr, t = {R['happy_mean']}, {R['gloomy_mean']}, {R['spread']}, {R['ls_t']}\n"
            "    banner = 'frozen real numbers (offline)'\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['HAPPY third','GLOOMY third'], [happy_m, gloom_m], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'The folklore says HAPPY wins. Here the GLOOMY third won (+{gloom_m:.0f}% vs +{happy_m:.0f}%)')\n"
            "fig.suptitle(banner, fontsize=9, color=GREY); plt.tight_layout(); plt.show()\n"
            "print(f'Happy +{happy_m:.1f}%  vs  Gloomy +{gloom_m:.1f}%  ->  spread {spr:+.1f}% (t {t:+.2f})')"
        ),
        md(
            f"There it is, **backwards**. The folklore predicts the *happy* markets win. On 2024-26 the "
            f"**gloomy** markets won by miles (+{R['gloomy_mean']}% vs +{R['happy_mean']}%) — the "
            f"long-happy/short-gloomy spread is **{R['spread']}%**. The 'gloomy' bucket here is Korea "
            "(which more than *tripled*), Spain, Italy and South Africa: whatever regions happened to "
            "rip this window, they weren't the Nordics."
        ),
        md(
            "**Is happiness *linked* to returns at all?** The cleanest single check is the *rank "
            "correlation*: line countries up by happiness, line them up by return, and see if the two "
            "orderings match."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rc = st.rank_correlation(PANEL); rho = rc['rho']\n"
            "    d = PANEL['happiness']; y = PANEL['forward_ret']*100\n"
            "else:\n"
            f"    rho = {R['rho']}\n"
            "    d = None; y = None\n"
            "if HAVE_REAL:\n"
            "    fig, ax = plt.subplots(figsize=(8, 4.6))\n"
            "    ax.scatter(d, y, s=32, color=GREY)\n"
            "    for name in PANEL.index:\n"
            "        ax.annotate(name[:3], (PANEL.loc[name,'happiness'], PANEL.loc[name,'forward_ret']*100),\n"
            "                    fontsize=7, alpha=.6)\n"
            "    ax.set_xlabel('happiness (1.0 = happiest country)'); ax.set_ylabel('forward return %')\n"
            "    ax.set_title(f'Happiness vs returns: rank correlation = {rho:+.2f} (basically nothing)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "print(f'Spearman rank correlation(happiness, return) = {rho:+.2f}')"
        ),
        md(
            f"The rank correlation is **{R['rho']}** — essentially zero. If happiness drove returns "
            "you'd see a clear upward tilt; instead it's a shotgun blast. Happiness and market returns "
            "just aren't lined up."
        ),
        md(
            "**Maybe we just picked a bad window?** Run the same sort using four different World "
            "Happiness Reports:"
        ),
        code(
            "wins = " + repr([(w[0], w[1]) for w in R["win"]]) + "\n"
            "if HAVE_REAL:\n"
            "    spans = [('2019-03-20','2021-03-20'),('2021-03-19','2023-03-19'),\n"
            "             ('2022-03-18','2024-03-18'),('2024-03-20','2026-06-26')]\n"
            "    labs = ['2019-21','2021-23','2022-24','2024-26']\n"
            "    spr = []\n"
            "    for (sp,en) in spans:\n"
            "        pn = data.fetch_panel(fetch=False, split_date=sp, end_date=en)\n"
            "        spr.append(st.bucket_returns(pn, 0.3)['spread']*100 if not pn.empty else np.nan)\n"
            "else:\n"
            "    labs = [w[0] for w in wins]; spr = [w[1] for w in wins]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "cols = [GREEN if s>0 else RED for s in spr]\n"
            "ax.bar(labs, spr, color=cols, width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('happy - gloomy spread %')\n"
            "ax.set_title('The happiness edge (green) shows up once, then FADES and FLIPS (red)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('spreads by WHR edition:', [round(s,1) for s in spr])"
        ),
        md(
            "So the sign isn't stable. In **2019-21** the happy markets *did* edge ahead (+24%). Then "
            "it fades to noise (+3%, −7%) and in **2024-26** it flips hard negative (−42%). A 'signal' "
            "whose direction depends on which report you happen to read is the definition of a "
            "spurious correlation."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The spread is the *wrong* sign ({R['spread']}%, *t* {R['ls_t']}), the "
            f"rank correlation is {R['rho']}, and the sign flips across reports. And with only 24 "
            "countries you could never get a trustworthy answer anyway.\n"
            "- **Tradability — Mirage.** A 24-name sort, ~7 per bucket, wrong sign before costs. "
            "Nothing to harvest.\n"
            "- **Spurious correlation? — Busted.** This is what a made-up cross-country pattern looks "
            "like: tiny sample, near-zero correlation, sign that flips with the calendar."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No. Even ignoring the wrong sign, you're betting a whole strategy on the ordering of ~24 "
            "countries — and whatever *actually* drives them (economic cycle, currencies, which region "
            "is hot) has nothing to do with the happiness survey. This window, emerging and Asian "
            "markets led, so 'gloomy' won. Next window it'll be something else. That's not an edge; "
            "it's a lottery ticket with a happy sticker on it."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The mood-signal cousins.** [257 AAII-Sentiment](../../257-aaii-sentiment/) and "
            "[335 Buzz-Sentiment-ETF](../../335-buzz-sentiment-etf/) test *time-series* sentiment on a "
            "single market — a different axis from this cross-country sort.\n"
            "- **The synthetic control (quant notebook).** We *plant* a real happiness effect in a toy "
            "world and show the engine catches it — proving the null result here is the data, not a "
            "broken test.\n\n"
            "*Think happiness really does predict markets? Fork this, add a bigger country universe and "
            "a longer history, and show the happy-minus-gloomy spread clearing t = 2 the right way.*"
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
            "# The Happiness Trade — a quantitative teardown 🔬\n"
            "### WHR rank sort · Welch two-sample *t* · Spearman rank correlation · label-shuffle placebo · country-level slope · four-edition sign-flip · costs & borrow · synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Spurious%3F: Busted](https://img.shields.io/badge/Spurious%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We sort the investable "
            "cross-section by World Happiness Report rank and test whether the happiest nations' "
            "markets beat the gloomiest.\n\n"
            "> ⚠️ **Not investment advice.** Illustrative real data: yfinance daily total returns for "
            f"24 single-country ETFs joined to the WHR-2024 rank (as-of 2026-06-30, panel fp "
            f"`{R['fp_panel']}`); the offline core and tests run on a deterministic synthetic world. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition.\n"
            ">\n"
            "> ⚠️ **Synthetic-first / small-N.** The investable cross-section is ~24 countries and no "
            "clean aligned happiness × tradable-index panel exists, so a robust real-tape *t* ≥ 2 is "
            "unreachable — this study is capped below REAL by design (named on the Signal axis)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Happy − gloomy spread **{R['spread']}%** (Welch *t* "
            f"**{R['ls_t']}**, placebo *p* {R['placebo_p']}) — wrong sign; Spearman rho "
            f"**{R['rho']}** (*t* {R['rho_t']}); slope-*t* {R['slope_t']}; sign flips across editions. |\n"
            f"| **Tradability** | `MIRAGE` | 24-name sort, ~7/bucket, sign-unstable. Gross "
            f"{R['spread']}% → net {R['net']}% (5 bps/leg + {R['borrow_bps']:.0f} bps borrow). |\n"
            f"| **Spurious correlation?** | `BUSTED` | n = 24, rank correlation {R['rho']}, direction "
            "flips edition-to-edition — the textbook signature. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), but on ~24 "
            "countries the happiness sort is absent-to-inverted, insignificant, and sign-unstable."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $H_i$ be country $i$'s happiness score (from its WHR rank) and $r_i$ its forward "
            "market return.\n\n"
            "- **H₁ (the sort / long-short).** $\\mathbb{E}[r \\mid \\text{happy}] - "
            "\\mathbb{E}[r \\mid \\text{gloomy}] > 0$ at *t* ≥ 2 (happy beats gloomy).\n"
            "- **H₂ (rank correlation).** $\\rho_{\\text{Spearman}}(H, r) > 0$, stably.\n"
            "- **H₃ (country-level).** slope of $r$ on $H$ $> 0$.\n"
            "- **H₄ (robustness).** The sign of H₁–H₃ is stable across WHR editions.\n\n"
            "We find **H₁ rejected** (wrong sign), **H₂ rejected** (rho ≈ 0), **H₃ rejected** (slope "
            "leans negative), and **H₄ rejected** (sign flips) — and, structurally, n = 24 is too "
            "small to certify anything either way."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why it's spurious**. Happiness rank is entangled with a dozen macro "
            "variables — GDP, governance, sector mix, currency regime — so any cross-country return "
            "sort will load on whichever macro factor led the window. Here (2024-26) that was "
            "EM/Asia, so the *gloomy* bucket won. With only ~24 investable countries and ~7 per "
            "bucket, the standard errors swamp any true effect. This is the same synthetic-only, "
            "no-clean-panel structure as [273 Lego-Returns](../../273-lego-returns/) and "
            "[276 Sneaker-Resale](../../276-sneaker-resale/)."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Score.** Happiness $H \\in [0,1]$ from WHR rank (best rank → 1.0), monotone so the "
            "slope's sign is the folklore's sign.\n"
            "- **Sort.** Tercile tails (≈7 countries each): happy (best ranks) vs gloomy (worst).\n"
            "- **Inference.** Welch two-sample *t* on happy − gloomy forward returns; a "
            "**label-shuffle placebo** null (2000 perms); the **Spearman rank correlation** (the "
            "single number the folklore lives on).\n"
            "- **Country-level.** OLS of forward return on $H$ — the *sign* of the slope is the "
            "folklore.\n"
            "- **Robustness.** Re-sort over four WHR editions; read the sign.\n"
            "- **Frictions.** Round-trip cost per leg + a modest borrow on the gloomy short.\n"
            "- **Positive control.** A deterministic synthetic world with a planted effect "
            "(`rank_alpha > 0`) and a null — averaged over 40 seeds (house rule ≥ 20).\n\n"
            "Timing: happiness is the *published* WHR rank preceding the window, so the signal is "
            "public at entry; we enter at that close and hold over the forward window. No look-ahead."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The sort — H₁\n\n"
            "Happy vs gloomy forward returns, with the Welch *t* and the placebo null."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = st.bucket_returns(PANEL, 0.3); ls = st.long_short_tstat(b)\n"
            "    p = st.placebo_pvalue(PANEL, n_perm=2000, seed=548)\n"
            "    happy_m, gloom_m, spr, t = b['happy_mean']*100, b['gloomy_mean']*100, b['spread']*100, ls['t']\n"
            "else:\n"
            f"    happy_m, gloom_m, spr, t, p = {R['happy_mean']}, {R['gloomy_mean']}, {R['spread']}, {R['ls_t']}, {R['placebo_p']}\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['HAPPY','GLOOMY'], [happy_m, gloom_m], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'Happy - gloomy = {spr:+.1f}%  (Welch t {t:+.2f}, placebo p {p:.3f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Happy +{happy_m:.1f}% | Gloomy +{gloom_m:.1f}% | spread {spr:+.1f}% | t {t:+.2f} | placebo p {p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: H₁ **rejected, with the sign reversed.** The spread is "
            f"**{R['spread']}%** (*t* {R['ls_t']}), and the placebo *p* = {R['placebo_p']} says even the "
            "inversion is not clean on n = 24 — a non-result, not an anti-signal."
        ),
        md(
            "### 4b · The rank correlation & country-level slope — H₂ / H₃\n\n"
            "Spearman rho (the folklore's headline number) and an OLS of forward return on happiness."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rc = st.rank_correlation(PANEL); reg = st.cross_section_regression(PANEL)\n"
            "    d = PANEL['happiness']; y = PANEL['forward_ret']*100\n"
            "    rho, slope, slope_t, corr = rc['rho'], reg['slope']*100, reg['slope_t'], reg['corr']\n"
            "    fig, ax = plt.subplots(figsize=(8, 4.6))\n"
            "    ax.scatter(d, y, s=30, color=GREY)\n"
            "    xs = np.linspace(d.min(), d.max(), 50)\n"
            "    ax.plot(xs, (reg['slope']*xs + (y.mean()/100 - reg['slope']*d.mean()))*100, c=RED, lw=2)\n"
            "    ax.set_xlabel('happiness (1.0 = happiest)'); ax.set_ylabel('forward return %')\n"
            "    ax.set_title(f'Spearman rho {rho:+.2f} | slope {slope:+.0f}%/unit (t {slope_t:+.2f}) — NOT positive')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            f"    rho, slope, slope_t, corr = {R['rho']}, {R['slope']}, {R['slope_t']}, {R['corr']}\n"
            "print(f'Spearman rho {rho:+.2f} | slope {slope:+.1f}%/unit | slope-t {slope_t:+.2f} | Pearson corr {corr:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: H₂ and H₃ **rejected.** The rank correlation is **{R['rho']}** "
            f"(*t* {R['rho_t']}) — the folklore's headline number is essentially zero. The OLS slope "
            f"leans *negative* ({R['slope']}%/unit, *t* {R['slope_t']}) but doesn't clear |*t*| = 2, "
            "and (next panel) its sign isn't stable."
        ),
        md(
            "### 4c · Robustness — H₄ (does the sign hold across WHR editions?)\n\n"
            "The same sort over four World Happiness Reports."
        ),
        code(
            "wins = " + repr(R["win"]) + "\n"
            "if HAVE_REAL:\n"
            "    spans = [('2019-03-20','2021-03-20'),('2021-03-19','2023-03-19'),\n"
            "             ('2022-03-18','2024-03-18'),('2024-03-20','2026-06-26')]\n"
            "    labs = [w[0] for w in wins]\n"
            "    rows = []\n"
            "    for lab,(sp,en) in zip(labs, spans):\n"
            "        pn = data.fetch_panel(fetch=False, split_date=sp, end_date=en)\n"
            "        bb = st.bucket_returns(pn, 0.3); rr = st.cross_section_regression(pn); rc = st.rank_correlation(pn)\n"
            "        rows.append((lab, bb['spread']*100, rr['slope_t'], rc['rho']))\n"
            "else:\n"
            "    rows = wins\n"
            "tab = pd.DataFrame(rows, columns=['edition','spread%','slope_t','rho']).set_index('edition')\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "cols = [GREEN if s>0 else RED for s in tab['spread%']]\n"
            "ax.bar(range(len(tab)), tab['spread%'], color=cols, width=.55)\n"
            "ax.set_xticks(range(len(tab))); ax.set_xticklabels(tab.index)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('happy - gloomy spread %')\n"
            "ax.set_title('Sign flips: edge present (green) in 2019-21, faded then inverted (red)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(2).to_string())"
        ),
        md(
            "> 💡 In plain words: H₄ **rejected.** Spread **+24%** (2019-21, slope-*t* +1.80) → **+3% / "
            "−7%** (noise) → **−42%** (2024-26). A sign that depends on which report you read is not a "
            "signal — it is a spurious correlation caught in the act."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — H₁ rejected (spread {R['spread']}%, *t* {R['ls_t']}, wrong sign); "
            f"H₂ rejected (rho {R['rho']}); H₃ rejected (slope *t* {R['slope_t']}); H₄ rejected (sign "
            "flips). Structurally uncertifiable at n = 24.\n"
            f"- **Tradability `MIRAGE`** — 24-name sort, ~7/bucket, sign-unstable; gross {R['spread']}% "
            f"→ net {R['net']}%.\n"
            "- **Spurious correlation? `BUSTED`** — tiny sample, near-zero correlation, sign flips "
            "edition-to-edition."
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
            "    net = st.net_spread(b, cost_bps=5.0, borrow_ann_bps=60.0, holding_years=2.0)*100\n"
            "else:\n"
            f"    gross, net = {R['spread']}, {R['net']}\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross','net\\n(costs+borrow)'], [gross, net], color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('happy - gloomy spread %')\n"
            "ax.set_title(f'Wrong sign before costs ({gross:+.0f}%), worse after ({net:+.0f}%)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:+.1f}% -> net {net:+.1f}% (5 bps/leg + 60 bps borrow, 2y hold)')"
        ),
        md(
            "> 💡 In plain words: there is nothing to charge costs against — the long-happy/short-gloomy "
            "book *loses* before frictions on this window. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector on n = 24, or does it always print ~0? Plant a real "
            "happiness effect (`rank_alpha > 0`) of increasing strength and watch the country-level "
            "slope-*t* climb — averaged over 40 seeds so no single lucky seed can fake it."
        ),
        code(
            "alphas = [0.0, 0.03, 0.06, 0.10, 0.15]\n"
            "ts  = [st.synthetic_mean_t(data, rank_alpha=a, n_seeds=40) for a in alphas]\n"
            "tss = [st.synthetic_mean_spread_t(data, rank_alpha=a, n_seeds=40) for a in alphas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(alphas, ts, 'o-', c=GREEN, lw=2, label='firm slope-t')\n"
            "ax.plot(alphas, tss, 's--', c=AMBER, lw=2, label='long-short Welch-t')\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted rank_alpha (more positive = stronger happiness effect)')\n"
            "ax.set_ylabel('mean t (40 seeds)')\n"
            "ax.set_title('The harness banks a planted effect — flat at the null, past +2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for a, t, ts2 in zip(alphas, ts, tss): print(f'alpha {a:+.2f} -> slope-t {t:+.2f}, spread-t {ts2:+.2f}')"
        ),
        md(
            "The stats sit at ≈ 0 at the null and climb past +2 as the planted effect grows — so the "
            "engine is a faithful detector even on 24 countries. The real-tape non-result is therefore "
            "a statement about **this cross-section**: the World Happiness sort is absent, "
            "insignificant, and sign-unstable — a spurious correlation, `BUSTED`. For the *time-series* "
            "mood cousins, see [257 AAII-Sentiment](../../257-aaii-sentiment/) and "
            "[335 Buzz-Sentiment-ETF](../../335-buzz-sentiment-etf/)."
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
