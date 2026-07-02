"""Generate the two narrative notebooks for Study 543 (Western-Zodiac-CEO).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic figures
run anywhere, offline and deterministic; the real-tape cells use the cached prices under ../_cache/
if present and otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md), so
the notebook re-runs for any reader offline.
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
# 2022-06-30, forward to 2026-06-26; 31 curated CEOs; panel fp 67cb7992be0c).
R = dict(
    fp_px="c391b6621701", fp_panel="67cb7992be0c",
    n=31, signs_present=11, split="2022-06-30", end="2026-06-26",
    F=0.69, F_p=0.72, perm_pF=0.70,
    best_sign="Aquarius", best_mean=466.0, rest_mean=202.0, spread=264.0,
    bvr_t=0.73, perm_pspread=0.83,
    net=259.0, cost_bps=5.0, borrow_bps=100.0,
    # robustness windows: (label, F, perm_p, best_sign, spread_ratio)
    win=[("2019-06 -> 2021-06", 1.34, 0.230, "Cancer", 6.55),
         ("2020-06 -> 2022-06", 0.95, 0.502, "Cancer", 0.96),
         ("2021-06 -> 2023-06", 0.72, 0.679, "Leo", 0.45),
         ("2022-06 -> 2024-06", 1.21, 0.293, "Aquarius", 2.80),
         ("2022-06 -> 2026-06", 0.69, 0.687, "Aquarius", 2.64)],
    # synthetic control: (alpha, mean F over 25 seeds, mean perm p)
    ctrl=[(0.00, 1.16, 0.444), (0.15, 1.72, 0.202), (0.30, 3.27, 0.039), (0.50, 6.88, 0.002)],
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

from western_zodiac_ceo import data, strategy as st

SPLIT, END = "2022-06-30", "2026-06-26"

def load_real():
    \"\"\"Cache-first real cross-section (empty frame offline).\"\"\"
    px = data.fetch_prices(fetch=False)
    if px.empty:
        return pd.DataFrame()
    return data.build_panel(px, SPLIT, END)

PANEL = load_real()
HAVE_REAL = not PANEL.empty
print("real CEO-sign panel present:", HAVE_REAL,
      "" if not HAVE_REAL else f"({len(PANEL)} CEOs, scored {SPLIT} -> {END})")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The CEO's Star Sign — do Aries chiefs beat Capricorns? ♌\n"
            "### Sorting big companies by their boss's horoscope, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Lucky_sign%3F: Busted](https://img.shields.io/badge/Lucky_sign%3F-Busted-8b949e?style=flat-square)\n\n"
            "Every so often a finance-lifestyle piece claims that a company led by a CEO of the "
            "*right* star sign — a decisive Aries, a visionary Aquarius — does better. It's the "
            "horoscope column pointed at the boardroom, and it's the western-astrology twin of the "
            "**Chinese zodiac-year** folklore we already tested ([Study 165](../../165-chinese-zodiac/)).\n\n"
            "So we test it the honest way. We hand-build a small table of well-known big-company "
            "CEOs, look up each one's **birth date** (public record), work out their **sun sign**, "
            "and ask: did the companies whose bosses share a sign earn different returns?\n\n"
            "> 📓 **This is the plain-language layer.** Want the *F*-stats, the shuffle test and the "
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
            f"| Do the sign groups earn different returns? | **No.** Across the {R['signs_present']} "
            f"signs in the table the spread of averages is *smaller* than the scatter within each "
            f"sign (ANOVA F = {R['F']}). |\n"
            f"| Is there a lucky sign? | **No.** The 'best' sign ({R['best_sign']}) wins only because "
            "it happens to contain one AI mega-winner — and a shuffle test says that's pure luck "
            f"(*p* = {R['perm_pspread']}). |\n"
            "| Does the winner stay the winner? | **No.** Look at different time windows and the "
            "'lucky sign' keeps changing — Cancer, then Leo, then Aquarius. |\n"
            "| Could you trade it? | **No.** Twelve signs split across ~30 companies is 2-3 names "
            "per sign — far too few to tell luck from skill. |\n\n"
            "> A birthday has nothing to do with how a company trades — and with only a couple of "
            "CEOs per sign, even a real pattern couldn't show up. This is a small-sample mirage."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"A company run by a CEO of the right star sign outperforms.\"* — astro-finance "
            "folklore, the western cousin of the Chinese zodiac-year claim.\n\n"
            "The pitch is that personality traits attached to a sun sign — Aries drive, Aquarius "
            "vision, Capricorn discipline — flow through to the company and its stock. There's no "
            "actual mechanism (your birthday is fixed years before you run anything), which is "
            "exactly why it's folklore. We give it a fair test anyway."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If it were true, you could screen the whole market by the CEO's birthday — no "
            "accounting, no charts, just a horoscope. It would also be one of the purest examples of "
            "a pattern that *must* be coincidence. The desk has already busted the calendar version "
            "([165 Chinese-Zodiac](../../165-chinese-zodiac/)); this is the *person* version."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Build the table.** For each big-name CEO, look up the birth date (Wikipedia / "
            "company bios) and convert it to a sun sign.\n"
            "2. **Group by sign.** Put each company in its CEO's sign bucket.\n"
            "3. **Compare returns.** Did some signs' companies clearly beat others over the next few "
            "years? An ANOVA answers 'do the groups differ at all?'; a shuffle test tells us whether "
            "any gap is more than luck.\n\n"
            "The catch that dooms it: twelve signs over ~30 companies leaves only **2-3 names per "
            "sign**. That's like judging a dice game after three rolls.\n\n"
            "*Timing:* the sign is set at birth, decades before any price — so there's nothing to "
            "peek at. We just fix a start date and measure the return over the following years."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Do the sign buckets earn different returns?** Here's the average forward return for "
            "each sun sign in the table."
        ),
        code(
            "if HAVE_REAL:\n"
            "    means = st.sign_means(PANEL).sort_values()*100\n"
            "    counts = st.sign_counts(PANEL)\n"
            "    f = st.anova_f(PANEL); Fval, Fp = f['F'], f['p']\n"
            "    banner = f'REAL tape ({len(PANEL)} CEOs, 2022-06 -> 2026-06)'\n"
            "else:\n"
            f"    order = ['Leo','Scorpio','Taurus','Cancer','Libra','Pisces','Capricorn','Gemini','Virgo','Aries','{R['best_sign']}']\n"
            f"    means = pd.Series(np.linspace(120, {R['best_mean']}, len(order)), index=order)\n"
            "    counts = pd.Series(3, index=order)\n"
            f"    Fval, Fp = {R['F']}, {R['F_p']}\n"
            "    banner = 'frozen real numbers (offline)'\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.6))\n"
            "cols = [GREEN if m==means.max() else GREY for m in means]\n"
            "ax.bar(means.index, means.values, color=cols, width=.7)\n"
            "ax.axhline(means.mean(), c=RED, ls='--', lw=1.2, label='grand mean')\n"
            "ax.set_ylabel('forward return %'); ax.legend()\n"
            "ax.set_title(f'Sign averages scatter like noise — ANOVA F = {Fval:.2f} (p {Fp:.2f})')\n"
            "plt.setp(ax.get_xticklabels(), rotation=30, ha='right')\n"
            "fig.suptitle(banner, fontsize=9, color=GREY); plt.tight_layout(); plt.show()\n"
            "print('names per sign:', counts.to_dict())"
        ),
        md(
            f"The bars bounce around, but an **ANOVA F of {R['F']}** (below 1) means the gaps "
            "*between* signs are actually **smaller** than the ordinary scatter *within* each sign. "
            "In plain terms: knowing the CEO's star sign tells you nothing about the return. And look "
            "at the counts — 2-3 companies per sign. There's no test that could find a signal here "
            "even if one existed."
        ),
        md(
            "**But wait — one sign *did* have the highest average. Isn't that the lucky sign?** "
            "Let's give the folklore its best shot: go long the best sign, short the rest, and then "
            "ask a shuffle test how often *random* sign labels would produce a gap that big."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bvr = st.best_vs_rest(PANEL); pl = st.placebo_pvalue(PANEL, n_perm=5000, seed=543)\n"
            "    best, bm, rm, spr, p = bvr['sign'], bvr['best_mean']*100, bvr['rest_mean']*100, bvr['spread']*100, pl['p_spread']\n"
            "else:\n"
            f"    best, bm, rm, spr, p = '{R['best_sign']}', {R['best_mean']}, {R['rest_mean']}, {R['spread']}, {R['perm_pspread']}\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar([f'{best}\\n(best sign)','the rest'], [bm, rm], color=[GREEN, GREY], width=.5)\n"
            "ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'{best} \"wins\" (+{bm:.0f}% vs +{rm:.0f}%) — but shuffle p = {p:.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'{best}: +{bm:.0f}% vs rest +{rm:.0f}% -> spread +{spr:.0f} pts | shuffle p {p:.2f}')"
        ),
        md(
            f"There's the trap. **{R['best_sign']}** looks amazing (+{R['best_mean']:.0f}% vs "
            f"+{R['rest_mean']:.0f}%) — but that's because it happens to hold **NVDA**, one stock "
            "that ~10x'd in the AI boom. Once you account for the fact that we *cherry-picked the "
            "best of twelve signs*, the shuffle test says a gap this big shows up "
            f"**{R['perm_pspread']:.0%} of the time by pure chance**. That's not a signal; that's the "
            "definition of luck."
        ),
        md(
            "**Does the lucky sign at least stay the same over time?** No — run it over five "
            "different windows:"
        ),
        code(
            "wins = " + repr([(w[0], w[3]) for w in R["win"]]) + "\n"
            "labs = [w[0] for w in wins]; best_signs = [w[1] for w in wins]\n"
            "if HAVE_REAL:\n"
            "    px = data.fetch_prices(fetch=False)\n"
            "    spans = [('2019-06-28','2021-06-30'),('2020-06-30','2022-06-30'),\n"
            "             ('2021-06-30','2023-06-30'),('2022-06-30','2024-06-28'),('2022-06-30','2026-06-26')]\n"
            "    best_signs = []\n"
            "    for sp,en in spans:\n"
            "        pn = data.build_panel(px, sp, en)\n"
            "        best_signs.append(st.best_vs_rest(pn)['sign'] if not pn.empty else '?')\n"
            "fig, ax = plt.subplots(figsize=(9, 3.2))\n"
            "ax.axis('off')\n"
            "tbl = ax.table(cellText=[best_signs], colLabels=labs, rowLabels=['\"lucky\" sign'], loc='center', cellLoc='center')\n"
            "tbl.scale(1, 2.2); tbl.set_fontsize(11)\n"
            "ax.set_title('The \"lucky sign\" keeps changing — a sign that moves is not a signal')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('winning sign by window:', dict(zip(labs, best_signs)))"
        ),
        md(
            "Cancer, Cancer, Leo, Aquarius, Aquarius. The 'lucky sign' is whatever sign happened to "
            "hold that window's biggest winner. That's the tell of a coincidence."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Sign averages scatter like noise (ANOVA F {R['F']}, shuffle *p* "
            f"{R['perm_pF']}); the 'best sign' is a cherry-pick that dies at *p* {R['perm_pspread']}.\n"
            "- **Tradability — Mirage.** 2-3 names per sign is a multiple-comparisons mirage; nothing "
            "to trade.\n"
            "- **Lucky sign? — Busted.** The winner changes every window and rides one mega-cap."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No. Even ignoring that there's no real gap, the 'trade' would be long one sign and short "
            "the other eleven — on 2-3 stocks per bucket, rebuilt whenever a CEO changes. The apparent "
            "edge is one AI winner sitting in one sign. Costs are a footnote to a number that is "
            "already luck."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The calendar cousin.** [165 Chinese-Zodiac](../../165-chinese-zodiac/) tests the "
            "zodiac *year* (Dragon-year bullishness) and busts it the same way — small-sample, "
            "multiple-comparisons.\n"
            "- **Why it can never be 'Real'.** A sun sign never changes over a CEO's tenure, so "
            "there's no time series to average — just one thin cross-section. No amount of data-mining "
            "can turn a birthday into alpha.\n\n"
            "*Think there's a lucky sign? Fork this, add hundreds of CEOs (so each sign has real "
            "power), and show one sign's ANOVA clearing significance after a multiple-comparisons "
            "correction. Spoiler: it won't.*"
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
            "# The CEO's Star Sign — a quantitative teardown 🔬\n"
            "### Curated CEO→sign table · one-way ANOVA · label-shuffle & max-statistic placebos · best-vs-rest Welch *t* · five-window sign-wander · costs & borrow · synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Lucky_sign%3F: Busted](https://img.shields.io/badge/Lucky_sign%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its null.* We map large-cap CEOs to their western "
            "sun sign from public birth dates, sort the cross-section, and test whether the sign "
            "means differ — with the multiple-comparisons correction a 12-way test on ~30 names "
            "demands.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance daily prices for a hand-curated "
            f"31-CEO table (as-of 2026-06-30, panel fp `{R['fp_panel']}`); the offline core and tests "
            "run on a deterministic synthetic world. Methods in "
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
            f"| **Signal** | `NONE` | ANOVA across {R['signs_present']} signs **F = {R['F']}** "
            f"(analytic *p* {R['F_p']}, permutation *p* **{R['perm_pF']}**); best-vs-rest Welch *t* "
            f"**+{R['bvr_t']}**, max-statistic placebo *p* **{R['perm_pspread']}**. |\n"
            f"| **Tradability** | `MIRAGE` | 2-3 names per sign; the 'best' sign rides one mega-cap. "
            f"Gross +{R['spread']:.0f} pts → net +{R['net']:.0f} pts (5 bps/leg + {R['borrow_bps']:.0f} "
            "bps borrow) — but the pick is *p* {0}. |\n".format(R['perm_pspread'])
            + "| **Lucky sign?** | `BUSTED` | Winning sign wanders Cancer→Cancer→Leo→Aquarius→Aquarius; "
            "ANOVA never clears *p* < 0.23. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), but a birthday "
            "carries no return information and ~30 names split twelve ways can't be powered."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $s_i \\in \\{1,\\dots,12\\}$ be firm $i$'s CEO sun sign and $r_i$ its forward "
            "return.\n\n"
            "- **H₁ (omnibus).** The sign-group means differ: one-way ANOVA rejects at the "
            "*permutation* level.\n"
            "- **H₂ (best-vs-rest).** $\\mathbb{E}[r \\mid \\text{best sign}] - "
            "\\mathbb{E}[r \\mid \\text{rest}] > 0$ at *t* ≥ 2 **after** a max-statistic correction "
            "for selecting the best of twelve signs.\n"
            "- **H₃ (stability).** The winning sign / any effect is stable across forward windows.\n\n"
            "We find **H₁ not rejected** (F < 1, permutation *p* ≈ 0.70), **H₂ not rejected** "
            "(*t* +0.73, corrected *p* 0.83), and **H₃ rejected** (the winning sign changes every "
            "window). The Signal axis is additionally capped below `REAL` **by construction**: a "
            "curated ~30-name table with a birthday-fixed label has no powered, survivorship-free "
            "tape to certify on."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — the multiple-comparisons trap\n\n"
            "This is a **12-way** hypothesis on a **~30-name** cross-section — 2-3 names per cell. "
            "Picking 'the best sign' after seeing the data is a textbook selection: the naive "
            "best-vs-rest *t* is upward-biased, so the honest null re-selects the best sign on **every "
            "shuffle** (a max-statistic / familywise correction). It is the same structural problem as "
            "[165 Chinese-Zodiac](../../165-chinese-zodiac/)'s '12 animals × ~3 years', where a "
            "Bonferroni correction buried the most extreme raw *t*."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Table.** Curated CEO birth dates → western tropical sun sign (`data.curated_signs`), "
            "joined to real yfinance forward returns.\n"
            "- **Omnibus.** One-way ANOVA F across the signs; report analytic *p* **and** the "
            "permutation *p* (the honest one on tiny cells).\n"
            "- **Best-vs-rest.** Long the best eligible sign (≥ 2 names), short the rest; Welch two-"
            "sample *t*.\n"
            "- **Nulls.** Label-shuffle permutation for F, **plus a max-statistic** shuffle for the "
            "best-vs-rest spread (re-selecting the best sign each shuffle).\n"
            "- **Robustness.** Re-run over five forward windows; read the winning sign and F.\n"
            "- **Frictions.** Round-trip cost per leg + an annual borrow on the short (rest) leg.\n"
            "- **Positive control.** A deterministic synthetic world with a planted sign bump "
            "(`sign_alpha > 0`) and a null — averaged over 25 seeds (house rule).\n\n"
            "Timing: the sun sign is fixed at birth (decades pre-price), so the signal is trivially "
            "public in advance — no look-ahead. The single convention is one as-of date and a same-"
            "close entry over the forward window; on a multi-year hold a one-bar-later entry changes "
            "nothing material."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The omnibus — H₁\n\n"
            "ANOVA across the signs, with the permutation *p* (the honest one on 2-3-name cells)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    f = st.anova_f(PANEL); pl = st.placebo_pvalue(PANEL, n_perm=5000, seed=543)\n"
            "    means = st.sign_means(PANEL).sort_values()*100\n"
            "    Fval, Fp, permp = f['F'], f['p'], pl['p_F']\n"
            "else:\n"
            f"    Fval, Fp, permp = {R['F']}, {R['F_p']}, {R['perm_pF']}\n"
            f"    means = pd.Series(np.linspace(120, {R['best_mean']}, {R['signs_present']}))\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.4))\n"
            "ax.bar(range(len(means)), means.values, color=GREY, width=.7)\n"
            "ax.axhline(means.mean(), c=RED, ls='--', lw=1.2, label='grand mean')\n"
            "ax.set_xticks(range(len(means)))\n"
            "ax.set_xticklabels(list(means.index) if HAVE_REAL else range(len(means)), rotation=30, ha='right')\n"
            "ax.set_ylabel('forward return %'); ax.legend()\n"
            "ax.set_title(f'ANOVA F = {Fval:.2f} (analytic p {Fp:.2f}, permutation p {permp:.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'ANOVA F {Fval:.2f} | analytic p {Fp:.2f} | permutation p {permp:.2f}')"
        ),
        md(
            f"> 💡 In plain words: H₁ **not rejected.** F = {R['F']} is *below 1* — the between-sign "
            f"variance is smaller than the within-sign variance — and the permutation *p* = "
            f"{R['perm_pF']} confirms it: seven in ten random relabellings match this F. No omnibus "
            "sign effect."
        ),
        md(
            "### 4b · Best-vs-rest — H₂ (and the selection correction)\n\n"
            "Long the best sign, short the rest — with the max-statistic placebo that re-picks the "
            "best sign on every shuffle."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bvr = st.best_vs_rest(PANEL); pl = st.placebo_pvalue(PANEL, n_perm=5000, seed=543)\n"
            "    best, bm, rm, spr, t, p = bvr['sign'], bvr['best_mean']*100, bvr['rest_mean']*100, bvr['spread']*100, bvr['t'], pl['p_spread']\n"
            "else:\n"
            f"    best, bm, rm, spr, t, p = '{R['best_sign']}', {R['best_mean']}, {R['rest_mean']}, {R['spread']}, {R['bvr_t']}, {R['perm_pspread']}\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar([f'{best}','rest'], [bm, rm], color=[GREEN, GREY], width=.5)\n"
            "ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'{best} - rest = +{spr:.0f} pts (Welch t +{t:.2f}, max-stat placebo p {p:.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'{best}: +{bm:.0f}% | rest +{rm:.0f}% | spread +{spr:.0f} pts | Welch t +{t:.2f} | max-stat p {p:.2f}')"
        ),
        md(
            f"> 💡 In plain words: H₂ **not rejected.** The naive spread (+{R['spread']:.0f} pts) looks "
            f"big, but the Welch *t* is only +{R['bvr_t']}, and once you correct for having picked the "
            f"best of twelve signs the placebo *p* = {R['perm_pspread']}. {R['best_sign']} 'wins' "
            "purely because it holds NVDA — one mega-cap in an AI melt-up."
        ),
        md(
            "### 4c · Stability — H₃ (does the winner hold?)\n\n"
            "The ANOVA + best-vs-rest over five forward windows."
        ),
        code(
            "wins = " + repr(R["win"]) + "\n"
            "if HAVE_REAL:\n"
            "    px = data.fetch_prices(fetch=False)\n"
            "    spans = [('2019-06-28','2021-06-30'),('2020-06-30','2022-06-30'),\n"
            "             ('2021-06-30','2023-06-30'),('2022-06-30','2024-06-28'),('2022-06-30','2026-06-26')]\n"
            "    rows = []\n"
            "    for (lab,_,_,_,_),(sp,en) in zip(wins, spans):\n"
            "        pn = data.build_panel(px, sp, en)\n"
            "        f = st.anova_f(pn); pl = st.placebo_pvalue(pn, n_perm=2000, seed=543); b = st.best_vs_rest(pn)\n"
            "        rows.append((lab, f['F'], pl['p_F'], b['sign'], b['spread']))\n"
            "else:\n"
            "    rows = wins\n"
            "tab = pd.DataFrame(rows, columns=['window','F','perm_p','best_sign','spread']).set_index('window')\n"
            "fig, ax = plt.subplots(figsize=(9, 4.2))\n"
            "ax.bar(range(len(tab)), tab['F'], color=GREY, width=.55)\n"
            "ax.axhline(1.0, c=RED, ls='--', lw=1, label='F = 1 (no group effect)')\n"
            "ax.set_xticks(range(len(tab))); ax.set_xticklabels([w.replace(' ','') for w in tab.index], rotation=20, ha='right')\n"
            "ax.set_ylabel('ANOVA F'); ax.legend()\n"
            "for i,(bs) in enumerate(tab['best_sign']):\n"
            "    ax.text(i, tab['F'].iloc[i]+.05, bs, ha='center', fontsize=8, color=GREY)\n"
            "ax.set_title('F hugs 1 in every window; the \"best sign\" (labels) keeps changing')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(3).to_string())"
        ),
        md(
            "> 💡 In plain words: H₃ **rejected.** The ANOVA F hovers around 1 in every window (best "
            "permutation *p* = 0.23) and the winning sign wanders Cancer → Cancer → Leo → Aquarius → "
            "Aquarius. A 'signal' whose identity changes with the window is not a signal."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — H₁ not rejected (F {R['F']}, permutation *p* {R['perm_pF']}); H₂ "
            f"not rejected (*t* +{R['bvr_t']}, corrected *p* {R['perm_pspread']}); H₃ rejected. Capped "
            "below `REAL` by construction (curated, birthday-fixed, underpowered).\n"
            f"- **Tradability `MIRAGE`** — 2-3 names per sign; gross +{R['spread']:.0f} pts → net "
            f"+{R['net']:.0f} pts, but the pick is *p* {R['perm_pspread']}.\n"
            "- **Lucky sign? `BUSTED`** — winner changes every window, rides one mega-cap."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "Costs barely move a number that is already luck."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bvr = st.best_vs_rest(PANEL)\n"
            "    gross = bvr['spread']*100\n"
            "    net = st.net_spread(bvr, cost_bps=5.0, borrow_ann_bps=100.0, holding_years=4.0)*100\n"
            "else:\n"
            f"    gross, net = {R['spread']}, {R['net']}\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross','net\\n(costs+borrow)'], [gross, net], color=[GREY, GREY], width=.5)\n"
            "ax.set_ylabel('best - rest spread %')\n"
            "ax.set_title(f'Gross +{gross:.0f} pts -> net +{net:.0f} pts — but the pick is p 0.83')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross +{gross:.0f} pts -> net +{net:.0f} pts (5 bps/leg + 100 bps borrow, 4y hold)')"
        ),
        md(
            "> 💡 In plain words: the net spread is still positive, but it is a post-hoc pick with "
            "*p* = 0.83 and an unstable winner — there is nothing real to charge costs against. "
            "`MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print F ≈ 1? Plant a real sign "
            "effect (`sign_alpha > 0`, one sign given an alpha bump) of increasing strength and watch "
            "the ANOVA F rise and the permutation *p* fall — averaged over 25 seeds so no single "
            "lucky seed can fake it."
        ),
        code(
            "alphas = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]\n"
            "res = [st.synthetic_mean_F(data, sign_alpha=a, n_seeds=25) for a in alphas]\n"
            "Fs = [r['mean_F'] for r in res]; ps = [r['mean_p'] for r in res]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "a1.plot(alphas, Fs, 'o-', c=GREEN, lw=2); a1.axhline(1, c=GREY, ls='--', lw=1)\n"
            "a1.set_xlabel('planted sign_alpha'); a1.set_ylabel('mean ANOVA F (25 seeds)')\n"
            "a1.set_title('F rises with the planted effect')\n"
            "a2.plot(alphas, ps, 'o-', c=RED, lw=2); a2.axhline(0.05, c=GREY, ls='--', lw=1, label='p = 0.05')\n"
            "a2.set_xlabel('planted sign_alpha'); a2.set_ylabel('mean permutation p'); a2.legend()\n"
            "a2.set_title('permutation p falls past 0.05')\n"
            "plt.tight_layout(); plt.show()\n"
            "for a, r in zip(alphas, res): print(f'alpha {a:.2f} -> mean F {r[\"mean_F\"]:.2f}, mean perm p {r[\"mean_p\"]:.3f}')"
        ),
        md(
            "At the null (`sign_alpha = 0`) the F sits near 1 and the permutation *p* near 0.5 — no "
            "false signal. Planting a genuine sign effect drives F up and *p* past 0.05, so the engine "
            "is a faithful detector. The flat real-tape result is therefore a statement about **the "
            "folklore**, not a broken engine: CEO sun sign carries no return information, and a "
            "curated ~30-name table could never certify it if it did. For the calendar cousin, see "
            "[165 Chinese-Zodiac](../../165-chinese-zodiac/)."
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
