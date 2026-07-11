"""Generate the two narrative notebooks for Study 660 (Carry-Everywhere).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached wide
daily-close tape under ../_cache/ and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with
no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance FX/Treasury/
# equity/commodity ETFs, 2007-07 -> 2026-06, 228 months; fingerprint ed7c280e84fe).
R = dict(
    start="2007-07", end="2026-06", n=228, fp="ed7c280e84fe",
    sleeves={
        "FX": dict(ann=-0.48, vol=10.39, sharpe=-0.05, t=-0.20, skew=-0.74, maxdd=-39.60),
        "BOND": dict(ann=1.71, vol=5.62, sharpe=0.30, t=1.33, skew=0.23, maxdd=-20.82),
        "EQ": dict(ann=-3.90, vol=11.05, sharpe=-0.35, t=-1.44, skew=-0.07, maxdd=-60.71),
        "CMD": dict(ann=2.41, vol=7.14, sharpe=0.34, t=1.39, skew=0.75, maxdd=-23.99),
    },
    corr={
        ("FX", "BOND"): -0.24, ("FX", "EQ"): -0.11, ("FX", "CMD"): -0.26,
        ("BOND", "EQ"): -0.06, ("BOND", "CMD"): 0.30, ("EQ", "CMD"): -0.05,
    },
    combo_ann=-0.07, combo_vol=3.79, combo_sharpe=-0.02, combo_t=-0.08,
    combo_skew=0.01, combo_maxdd=-18.51, combo_ci=(-0.45, 0.43),
    ivw=dict(FX=0.191, BOND=0.352, EQ=0.179, CMD=0.277),
    ivw_ann=0.48, ivw_sharpe=0.14, ivw_t=0.62,
    ex_eq_ann=1.21, ex_eq_sharpe=0.30, ex_eq_t=1.41,
    turnover=dict(FX=0.048, BOND=0.018, EQ=0.074, CMD=0.091),
    net5_ann=-0.26, net5_sharpe=-0.07, net5_t=-0.31,
    net10_ann=-0.30, net10_sharpe=-0.08, net10_t=-0.35,
    gfc=dict(combo=-0.96, other=-0.001, FX=-27.09, BOND=4.69, EQ=13.66, CMD=8.43),
    covid=dict(combo=2.26, other=-0.016, FX=-4.62, BOND=4.62, EQ=-10.68, CMD=20.87),
    syn_null_mean=-0.15, syn_null_sd=1.13, syn_null_fire=1, syn_null_seeds=20,
    syn_planted_t=2.50, syn_planted_sharpe=0.56, syn_planted_ann=2.06,
    syn_crash_t=-0.06, syn_crash_sharpe=-0.01, syn_crash_skew=-2.84,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Crashes_everywhere_at_once%3F: Busted](https://img.shields.io/badge/Crashes_everywhere_at_once%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from carry_everywhere import data as dt, strategy as st

HAVE_REAL = dt.have_real()
if HAVE_REAL:
    CLOSES = dt.load_real()
    MRET = dt.monthly_returns(CLOSES)
    SL = st.all_sleeves(MRET)
    CB = st.combo(SL)
else:
    CLOSES = MRET = SL = CB = None
print("real cache present:", HAVE_REAL, "| sleeve-months:", (0 if SL is None else len(SL)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does \"carry\" really pay off everywhere? 🌍💰\n"
            "### FX, bonds, dividend stocks, commodities — one signal, four markets, "
            "and a combo that turns out to add up to **nothing**\n\n"
            + BADGES +
            "\"Carry\" is the simplest idea in investing: hold the thing that pays you to "
            "hold it. Borrow a low-interest currency, lend a high one, pocket the gap. Buy "
            "the dividend stock instead of the growth stock, pocket the yield. A famous 2018 "
            "paper (Koijen, Moskowitz, Pedersen & Vrugt) argued this isn't just an FX trick — "
            "the same logic pays off in bonds, equities and commodities too, and because the "
            "four markets barely talk to each other, blending them should give you a smoother, "
            "*more reliable* payday than any one alone.\n\n"
            "We rebuilt all four sleeves on free data and blended them. Here's what happened.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the bootstrap and the cost "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** Four *static* baskets (composition fixed from each market's "
            "textbook carry classification, never re-fit to the data — like reading a public "
            "calendar, this has zero look-ahead), 2007-07 → 2026-06, 19 years including the "
            "2008 and 2020 crashes. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does each carry sleeve pay on its own? | **Not convincingly.** Bond carry "
            f"({R['sleeves']['BOND']['ann']:+.1f}%/yr) and commodity carry "
            f"({R['sleeves']['CMD']['ann']:+.1f}%/yr) lean positive but aren't statistically "
            f"solid; FX carry is flat ({R['sleeves']['FX']['ann']:+.1f}%/yr); equity carry is "
            f"outright negative ({R['sleeves']['EQ']['ann']:+.1f}%/yr). |\n"
            f"| Does blending the four fix it? | **No.** The equal-weight combo earns "
            f"**{R['combo_ann']:+.2f}%/yr** — statistically indistinguishable from zero. "
            "Diversifying four so-so ingredients doesn't bake a cake. |\n"
            "| Does it at least crash on cue, proving it's \"real risk you're paid to bear\"? | "
            f"**No — and that's the interesting part.** In 2008 the FX leg cratered "
            f"**{R['gfc']['FX']:+.1f}%** while the other three sleeves were solidly *positive*. "
            f"In 2020 the equity leg cratered **{R['covid']['EQ']:+.1f}%** while the commodity "
            f"leg *spiked* **{R['covid']['CMD']:+.1f}%**. The crashes never line up. |\n"
            "| So what actually happened? | On this tape, there's simply no premium there to "
            "protect — combining four legs that individually miss the bar just gives you a "
            "quieter zero. |\n\n"
            "> Four markets, one idea, zero payoff — but for an oddly reassuring reason: they "
            "don't fall apart together either."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"In every asset class — currencies, bonds, stocks, commodities — the asset "
            "that pays you more to hold it (all else equal) tends to keep paying you more. "
            "Carry works everywhere, and because the carry premium in one market is basically "
            "unrelated to carry in another, you can diversify across them and end up with a "
            "smoother, more dependable return than any single carry trade offers.\"*\n\n"
            "It's the kind of claim that sounds almost too tidy — the same intuition, four "
            "different costumes. Koijen, Moskowitz, Pedersen & Vrugt formalized it in 2018 "
            "with decades of institutional data. We're asking whether it survives a rebuild "
            "on the free data anyone can pull today."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, this is close to a holy grail for a small allocator: a genuinely "
            "diversifying source of return, built from instruments anyone can trade (ETFs and "
            "spot FX, no exotic derivatives required), with a story that doesn't depend on "
            "correctly forecasting where any market is *headed* — only on collecting what it "
            "already pays you to wait. That's the pitch behind entire multi-strategy \"carry "
            "fund\" products. So: does each leg actually pay, does blending them help, and does "
            "the whole thing crash exactly when a risk premium is supposed to?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **Four fixed baskets**, one per asset class, built from each market's textbook "
            "high-carry vs low-carry pairing — never re-fit to the data:\n"
            "  - **FX**: long AUD+NZD (classic high-yielders), short JPY+CHF (classic funders)\n"
            "  - **Bond**: long IEF (7-10y Treasuries), short SHY (1-3y) — the term-spread trade\n"
            "  - **Equity**: long VYM (high dividend yield), short VUG (growth, low yield)\n"
            "  - **Commodity**: long DBC (a roll-optimized broad basket), short GSG (a naive-roll "
            "basket) — isolating the roll-yield component\n"
            "- **The test.** Does each sleeve earn a statistically real premium (a *t*-stat "
            "clearing 2)? Does an equal blend of the four beat any one of them? And do the "
            f"legs really unravel together in **2008 GFC** and **2020 COVID** — the two "
            "windows the carry-crash story is built on?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the four sleeves on their own.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    anns = {c: SL[c].mean()*12*100 for c in SL.columns}\n"
            "else:\n"
            "    anns = {k: v['ann'] for k, v in R['sleeves'].items()}\n"
            "labels = ['FX\\n(AUD+NZD/JPY+CHF)','BOND\\n(IEF/SHY)','EQ\\n(VYM/VUG)','CMD\\n(DBC/GSG)']\n"
            "vals = [anns[k] for k in ['FX','BOND','EQ','CMD']]\n"
            "cols = [GREEN if v>0 else RED for v in vals]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.6))\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.2f}%/yr',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('gross annualised return')\n"
            "ax.set_title('Four carry sleeves: two lean positive, one is flat, one is negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({k: round(v,2) for k,v in anns.items()})"
        ),
        md(
            f"Bond and commodity carry lean positive (**{R['sleeves']['BOND']['ann']:+.2f}%/yr** "
            f"and **{R['sleeves']['CMD']['ann']:+.2f}%/yr**) but — spoiler for the quants "
            "notebook — neither clears the statistical bar alone. FX carry, the textbook trade, "
            f"is roughly flat (**{R['sleeves']['FX']['ann']:+.2f}%/yr**) over a period dominated "
            "by the post-2008 near-zero-rate years. Equity carry is the surprise laggard "
            f"(**{R['sleeves']['EQ']['ann']:+.2f}%/yr**) — dividend stocks (VYM) lost outright to "
            "growth stocks (VUG) through the biggest mega-cap tech bull market in a generation, "
            "and our simple proxy can't tell that apart from a genuine carry failure.\n\n"
            "**Now, the blend.** Does averaging the four help?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    c_ann = CB.mean()*12*100\n"
            "else:\n"
            "    c_ann = R['combo_ann']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.6))\n"
            "ax.bar(['equal-weight\\ncarry-everywhere combo'], [c_ann], color=GREY, width=.45)\n"
            "ax.annotate(f'{c_ann:+.2f}%/yr', (0, c_ann), ha='center',\n"
            "    va='top' if c_ann<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylim(-4.5, 3.0)\n"
            "ax.set_ylabel('gross annualised return')\n"
            "ax.set_title('The blend: not the average of hope, the average of the numbers above')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'combo: {c_ann:+.2f}%/yr')"
        ),
        md(
            f"**{R['combo_ann']:+.2f}%/yr.** Essentially nothing. The quants notebook shows the "
            f"combo's *t*-stat is **{R['combo_t']:+.2f}** — you cannot even confidently say the "
            "sign is right, let alone bank the number. Blending four ingredients where two are "
            "weak, one is flat and one is negative doesn't average up to a premium; it averages "
            "toward whatever the four actually summed to, which here is close to zero.\n\n"
            "**Finally, the crash test.** If carry really is \"a premium for selling insurance,\" "
            "the payoff should evaporate — hard — exactly when markets panic. Let's check both "
            "2008 and 2020:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    gfc = {c: st.crisis_stats(SL[c], dt.GFC_WINDOW)['cum_return_pct'] for c in SL.columns}\n"
            "    covid = {c: st.crisis_stats(SL[c], dt.COVID_WINDOW)['cum_return_pct'] for c in SL.columns}\n"
            "else:\n"
            "    gfc = {k: R['gfc'][k] for k in ['FX','BOND','EQ','CMD']}\n"
            "    covid = {k: R['covid'][k] for k in ['FX','BOND','EQ','CMD']}\n"
            "keys = ['FX','BOND','EQ','CMD']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.6), sharey=True)\n"
            "for ax, d, title in [(a1, gfc, '2008 GFC (Aug-Nov)'), (a2, covid, '2020 COVID (Feb-Apr)')]:\n"
            "    vals = [d[k] for k in keys]\n"
            "    cols = [GREEN if v>0 else RED for v in vals]\n"
            "    ax.bar(keys, vals, color=cols, width=.6)\n"
            "    for i,v in enumerate(vals): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',\n"
            "        va='top' if v<0 else 'bottom')\n"
            "    ax.axhline(0, c='k', lw=.8); ax.set_title(title)\n"
            "a1.set_ylabel('cumulative sleeve return in the window')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('GFC:', gfc); print('COVID:', covid)"
        ),
        md(
            f"They don't move together. In 2008, FX carry cratered **{R['gfc']['FX']:+.1f}%** "
            "while the other three sleeves were comfortably positive. In 2020, equity carry "
            f"cratered **{R['covid']['EQ']:+.1f}%** while commodity carry *spiked* "
            f"**{R['covid']['CMD']:+.1f}%** (oil's April-2020 negative-price shock hit the naive-"
            "roll commodity index far harder than the roll-optimized one, blowing the spread "
            "wide open in the *carry basket's* favor). The result: the **combo** was roughly "
            f"flat in 2008 (**{R['gfc']['combo']:+.2f}%**) and actually up in 2020 "
            f"(**{R['covid']['combo']:+.2f}%**). The \"synchronized carry crash\" of finance "
            "folklore just doesn't show up here — the legs crash on their own schedules, and "
            "that happens to cancel out."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Four legs, none clearing a statistical bar alone, and an "
            f"equal-weight blend of **{R['combo_ann']:+.2f}%/yr** that is indistinguishable "
            "from zero. Diversifying four so-so ingredients gives you a quieter zero, not a "
            "premium.\n"
            "- **Tradability — Mirage.** There's nothing to size, and costs push the already-"
            "flat number slightly negative.\n"
            "- **\"Crashes everywhere at once?\" — Busted.** In both 2008 and 2020 the legs "
            "moved in *different* directions, at *different* times, cancelling rather than "
            "compounding. That's a genuinely interesting (if quiet) finding: the diversification "
            "logic KMPV describe held up structurally — it just had nothing positive to protect."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This is a proxy problem as much as a signal problem.** Real KMPV-style carry "
            "uses live deposit rates, actual futures curves and dozens of instruments per asset "
            "class, re-ranked monthly. Our four static 2-vs-2 baskets on free ETF/spot data are "
            "a much coarser instrument — a fair, honest rebuild, but not the original paper's "
            "own (richer) test.\n"
            "- **The equity leg is the weak link worth re-testing.** Dividend yield conflates "
            "carry with the value factor; a cleaner carry proxy (e.g. options-implied dividend "
            "swaps, or short-rate-adjusted earnings yield) might separate the two.\n"
            "- **Sibling studies:** [364-fx-carry-trade](../../364-fx-carry-trade/) does the FX "
            "leg properly on its own; [612-em-debt-carry](../../612-em-debt-carry/) does one "
            "packaged-carry sleeve in depth; [638-value-momentum-everywhere](../../638-value-"
            "momentum-everywhere/) runs the identical multi-sleeve combo architecture for a "
            "*different* signal pair — and lands on the same kind of statistical zero.\n\n"
            "*Think a richer carry proxy would flip this? Show a combo that clears *t* ≥ 2 net "
            "of costs on free data, and we'll rerun the teardown.*"
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
            "# Carry-Everywhere — a quantitative teardown 🔬\n"
            "### Four HAC-*t* sleeve legs · a correlation matrix · a block-bootstrap combo "
            "Sharpe CI · inverse-vol and ex-equity combo variants · a turnover-based cost "
            "sweep · the 2008/2020 crisis ledger · a synthetic control with a plantable "
            "synchronized-crash factor\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "Koijen-Moskowitz-Pedersen-Vrugt (2018) claim carry pays in every asset class and "
            "diversifies well; the job here is to measure four honest proxies, combine them, "
            "and ask what actually survives inference.\n\n"
            "> ⚠️ **Data note.** yfinance total-return daily closes for 11 tickers "
            f"(FX spot + Treasury/dividend/commodity ETFs + BIL cash reference), "
            f"{R['start']} → {R['end']} ({R['n']} monthly observations), fingerprint "
            "`" + R["fp"] + "`. Four **static** long/short sleeves, composition fixed ex ante "
            "(zero look-ahead). No survivorship — all legs are diversified ETFs / spot FX, not "
            "single-name panels. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | equal-weight combo **{R['combo_ann']:+.2f}%/yr**, "
            f"Sharpe **{R['combo_sharpe']:+.2f}**, HAC **t = {R['combo_t']:+.2f}**, "
            f"bootstrap 95% Sharpe CI **[{R['combo_ci'][0]:+.2f}, {R['combo_ci'][1]:+.2f}]** |\n"
            f"| **Tradability** | `MIRAGE` | net at 5/10 bps: "
            f"**{R['net5_ann']:+.2f}% / {R['net10_ann']:+.2f}%/yr**, Sharpe "
            f"{R['net5_sharpe']:+.2f} / {R['net10_sharpe']:+.2f} |\n"
            f"| **Crashes everywhere at once?** | `BUSTED` | 2008 combo "
            f"{R['gfc']['combo']:+.2f}% (FX leg alone {R['gfc']['FX']:+.1f}%); 2020 combo "
            f"{R['covid']['combo']:+.2f}% (EQ leg alone {R['covid']['EQ']:+.1f}%, CMD leg "
            f"{R['covid']['CMD']:+.1f}%) |\n\n"
            "> 💡 In plain words: four legs that individually miss the bar, blended, still "
            "miss the bar — but they miss it *quietly*, because they don't fall apart on the "
            "same day."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{(s)}_t$ be sleeve $s \\in \\{\\text{FX, BOND, EQ, CMD}\\}$'s monthly return "
            "and $C_t = \\tfrac{1}{4}\\sum_s r^{(s)}_t$ the equal-weight combo. The claims:\n\n"
            "- **H₁ (carry pays, per sleeve).** $E[r^{(s)}_t] > 0$ with $t \\ge 2$, in each of "
            "the four asset classes.\n"
            "- **H₂ (low correlation).** $\\text{corr}(r^{(s)}_t, r^{(s')}_t)$ is small across "
            "sleeve pairs — the KMPV diversification premise.\n"
            "- **H₃ (the combo is a robust premium).** $E[C_t] > 0$ with $t \\ge 2$, and "
            "ideally with a *higher* Sharpe than any individual sleeve.\n"
            "- **H₄ (compensation for crash risk).** $C_t$ (and each $r^{(s)}_t$) is sharply "
            "negative in the canonical carry-unwind windows (2008 GFC, 2020 COVID).\n\n"
            "We find **H₂ supported** (|ρ| ≤ 0.30 throughout), **H₁ and H₃ not supported** "
            "(no leg, and not the combo, clears *t* ≥ 2), and **H₄ not supported** in either "
            "window — the legs move violently but not *together*."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Each sleeve's monthly return is a **single time series**, not a paired group "
            "split, so the planned primary is a **Newey-West (HAC, Bartlett kernel) *t*** on "
            "the one-sample mean (automatic-lag selection, $L = \\lfloor 4(n/100)^{2/9} "
            "\\rfloor$). Because a point-estimate Sharpe invites over-reading, the combo also "
            "gets a **circular block-bootstrap** 95% CI (block = 6 months, 2,000 draws — "
            "i.i.d. resampling would destroy the serial correlation the inference exists to "
            "respect). The crisis-window test uses two **hardcoded** facts (no fitting): the "
            "2008 GFC (Aug-Nov, the Lehman aftermath) and 2020 COVID (Feb-Apr, the crash and "
            "initial rebound)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Sleeves.** FX long AUD+NZD / short JPY+CHF; BOND long IEF / short SHY; EQ "
            "long VYM / short VUG; CMD long DBC / short GSG — all **static**, fixed ex ante.\n"
            f"- **Tape.** {R['n']} monthly observations, {R['start']} → {R['end']} "
            f"(fingerprint `{R['fp']}`). As-of 2026-06-30 (last complete month).\n"
            "- **Headline.** HAC *t* per sleeve and combo; block-bootstrap Sharpe CI on the "
            "combo.\n"
            "- **Robustness.** Inverse-vol-weighted combo; an ex-equity (3-sleeve) combo.\n"
            "- **Execution.** Monthly rebalance to par weights at month-end close — the "
            "basket composition never changes, so there is no signal-formation lag to "
            "document (analogous to a public calendar: zero look-ahead by construction).\n"
            "- **Costs.** One-way cost × first-order rebalance turnover (weight × |leg "
            "return|) per sleeve, plus a short-leg ETF borrow spread on EQ/CMD only (FX spot "
            "and Treasury legs pay none — the carry differential itself *is* the financing "
            "cost).\n"
            "- **Control.** Synthetic 4-sleeve world, tunable planted carry mean AND a tunable "
            "shared crash factor; the null must not fire across ≥ 10 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The four sleeves — HAC *t*, skew, drawdown\n\n"
            "One-sample Newey-West *t* on each sleeve's monthly mean; skew and max drawdown "
            "flag the shape risk a mean alone hides."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = {}\n"
            "    for c in SL.columns:\n"
            "        h = st.hac_mean(SL[c])\n"
            "        rows[c] = dict(ann=SL[c].mean()*12*100, vol=SL[c].std(ddof=1)*np.sqrt(12)*100,\n"
            "                       sharpe=st.sharpe(SL[c]), t=h['t'], skew=st.skewness(SL[c]),\n"
            "                       maxdd=st.max_drawdown(SL[c])*100)\n"
            "else:\n"
            "    rows = R['sleeves']\n"
            "keys = ['FX','BOND','EQ','CMD']\n"
            "anns = [rows[k]['ann'] for k in keys]; ts = [rows[k]['t'] for k in keys]\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.0, 6.6), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "a1.bar(keys, anns, color=[GREEN if v>0 else RED for v in anns], width=.6)\n"
            "for i,v in enumerate(anns): a1.annotate(f'{v:+.2f}%',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('ann. return %')\n"
            "a1.set_title('Sleeve returns and their HAC t (dashed = |t|=2 bar)')\n"
            "a2.bar(keys, ts, color=[RED if abs(t)>=2 else GREY for t in ts], width=.6)\n"
            "a2.axhline(0, c='k', lw=.8); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.set_ylabel('HAC t')\n"
            "plt.tight_layout(); plt.show()\n"
            "for k in keys: print(f\"{k}: ann {rows[k]['ann']:+.2f}%  Sharpe {rows[k]['sharpe']:+.2f}  \"\n"
            "                     f\"HAC t={rows[k]['t']:+.2f}  skew={rows[k]['skew']:+.2f}  \"\n"
            "                     f\"maxDD={rows[k]['maxdd']:.1f}%\")"
        ),
        md(
            "> 💡 In plain words: not one bar crosses the dashed *t* = ±2 line. Commodity roll-"
            f"yield carry comes closest (*t* = {R['sleeves']['CMD']['t']:+.2f}), bond term-"
            f"spread carry is similar (*t* = {R['sleeves']['BOND']['t']:+.2f}); both are "
            "directionally right but individually uncertifiable. FX carry is a coin flip "
            f"(*t* = {R['sleeves']['FX']['t']:+.2f}) and equity carry is negative "
            f"(*t* = {R['sleeves']['EQ']['t']:+.2f}, confounded with the growth/value cycle)."
        ),
        md(
            "### 4b · Correlation — the diversification premise, checked directly\n\n"
            "KMPV's case for combining sleeves rests on low pairwise correlation. We check it "
            "on our own tape rather than assume it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    C = SL.corr()\n"
            "else:\n"
            "    keys = ['FX','BOND','EQ','CMD']\n"
            "    C = pd.DataFrame(np.eye(4), index=keys, columns=keys)\n"
            "    for (a,b), v in R['corr'].items():\n"
            "        C.loc[a,b] = C.loc[b,a] = v\n"
            "fig, ax = plt.subplots(figsize=(5.6, 5.0))\n"
            "im = ax.imshow(C.values, vmin=-1, vmax=1, cmap='RdBu_r')\n"
            "ax.set_xticks(range(4)); ax.set_xticklabels(C.columns)\n"
            "ax.set_yticks(range(4)); ax.set_yticklabels(C.index)\n"
            "for i in range(4):\n"
            "    for j in range(4):\n"
            "        ax.text(j, i, f'{C.values[i,j]:+.2f}', ha='center', va='center',\n"
            "                color='white' if abs(C.values[i,j])>0.5 else 'black')\n"
            "ax.set_title('Sleeve correlation matrix — genuinely low')\n"
            "plt.colorbar(im, fraction=0.046, pad=0.04)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(C.round(2))"
        ),
        md(
            "> 💡 In plain words: every off-diagonal entry is |ρ| ≤ 0.30. The structural premise "
            "holds — these four legs really don't move together. It just doesn't rescue a "
            "premium that isn't there: diversification narrows the *distribution* around a "
            "mean, it can't move the mean itself."
        ),
        md(
            "### 4c · The headline combo, with a bootstrap CI (not a point estimate)\n\n"
            "Equal-weight (1/4 each), then a circular block-bootstrap (block = 6 months) on "
            "the annualised Sharpe."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c_ann, c_vol = CB.mean()*12*100, CB.std(ddof=1)*np.sqrt(12)*100\n"
            "    c_sh, c_h = st.sharpe(CB), st.hac_mean(CB)\n"
            "    lo, hi = st.block_bootstrap_sharpe_ci(CB)\n"
            "else:\n"
            "    c_ann, c_vol, c_sh = R['combo_ann'], R['combo_vol'], R['combo_sharpe']\n"
            "    c_h = {'t': R['combo_t']}; lo, hi = R['combo_ci']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.4))\n"
            "ax.errorbar([0], [c_sh], yerr=[[c_sh-lo],[hi-c_sh]], fmt='o', color=RED,\n"
            "            capsize=8, markersize=10, elinewidth=2)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlim(-1, 1); ax.set_xticks([0]); ax.set_xticklabels(['equal-weight combo'])\n"
            "ax.set_ylabel('annualised Sharpe (95% block-bootstrap CI)')\n"
            "ax.set_title(f'Sharpe {c_sh:+.2f}, CI [{lo:+.2f}, {hi:+.2f}] — straddles zero')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'combo: ann {c_ann:+.2f}%  vol {c_vol:.2f}%  Sharpe {c_sh:+.2f}  '\n"
            "      f\"HAC t={c_h['t']:+.2f}  bootstrap CI=[{lo:+.2f}, {hi:+.2f}]\")"
        ),
        md(
            f"> 💡 In plain words: the point estimate (Sharpe {R['combo_sharpe']:+.2f}) is "
            f"already near zero, and the 95% CI **[{R['combo_ci'][0]:+.2f}, "
            f"{R['combo_ci'][1]:+.2f}]** is wide enough that we can't even confidently claim "
            "the *sign*. This is the desk's honest bar failing cleanly, not a marginal miss."
        ),
        md(
            "### 4d · Robustness — does a smarter blend rescue it?\n\n"
            "Two natural alternatives: weight sleeves by inverse volatility (equal *risk* "
            "contribution, not equal notional), and drop the weakest leg (equity)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ivw = st.inv_vol_weights(SL)\n"
            "    cb_iv = st.combo(SL, ivw)\n"
            "    iv_ann, iv_sh, iv_t = cb_iv.mean()*12*100, st.sharpe(cb_iv), st.hac_mean(cb_iv)['t']\n"
            "    cb3 = st.combo(SL[['FX','BOND','CMD']])\n"
            "    ex_ann, ex_sh, ex_t = cb3.mean()*12*100, st.sharpe(cb3), st.hac_mean(cb3)['t']\n"
            "else:\n"
            "    iv_ann, iv_sh, iv_t = R['ivw_ann'], R['ivw_sharpe'], R['ivw_t']\n"
            "    ex_ann, ex_sh, ex_t = R['ex_eq_ann'], R['ex_eq_sharpe'], R['ex_eq_t']\n"
            "names = ['equal-weight\\n(headline)', 'inverse-vol\\nweighted', 'ex-equity\\n3-sleeve']\n"
            "anns = [R['combo_ann'] if not HAVE_REAL else CB.mean()*12*100, iv_ann, ex_ann]\n"
            "ts = [R['combo_t'] if not HAVE_REAL else st.hac_mean(CB)['t'], iv_t, ex_t]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "cols = [RED if abs(t)<2 else GREEN for t in ts]\n"
            "ax.bar(names, anns, color=cols, width=.55)\n"
            "for i,(a,t) in enumerate(zip(anns, ts)):\n"
            "    ax.annotate(f'{a:+.2f}%/yr\\n(t={t:+.2f})',(i,a),ha='center',\n"
            "        va='top' if a<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('gross annualised return')\n"
            "ax.set_title('No construction choice clears t >= 2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'inv-vol: {iv_ann:+.2f}% (t={iv_t:+.2f})  ex-EQ: {ex_ann:+.2f}% (t={ex_t:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: risk-weighting helps a little (*t* = {R['ivw_t']:+.2f}) and "
            f"dropping the weak equity leg helps more (*t* = {R['ex_eq_t']:+.2f}) — but even "
            "the best variant we tried stays under the bar. We are not p-hacking a construction "
            "until one clears 2; we show the natural alternatives and report all of them, "
            "clearing bar or not."
        ),
        md(
            "### 4e · Costs — turnover is real but small; there's no gross edge to protect\n\n"
            "First-order monthly rebalance turnover (weight × |leg return|) per sleeve, plus a "
            "short-leg ETF borrow spread on EQ (40 bps/yr) and CMD (25 bps/yr) only — FX and "
            "Treasury legs pay none."
        ),
        code(
            "if HAVE_REAL:\n"
            "    to = st.all_turnover(MRET).mean()\n"
            "    n5, n10 = st.combo_net(MRET, 5.0), st.combo_net(MRET, 10.0)\n"
            "    n5a, n10a = n5.mean()*12*100, n10.mean()*12*100\n"
            "    n5s, n10s = st.sharpe(n5), st.sharpe(n10)\n"
            "else:\n"
            "    to = pd.Series(R['turnover'])\n"
            "    n5a, n10a = R['net5_ann'], R['net10_ann']\n"
            "    n5s, n10s = R['net5_sharpe'], R['net10_sharpe']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(to.index, to.values, color=GREY, width=.55)\n"
            "a1.set_ylabel('avg monthly one-way turnover (frac NAV)')\n"
            "a1.set_title('Turnover is thin — these are static baskets')\n"
            "gross = R['combo_ann'] if not HAVE_REAL else CB.mean()*12*100\n"
            "a2.bar(['gross','net 5bps','net 10bps'], [gross, n5a, n10a],\n"
            "       color=[GREY, AMBER, AMBER], width=.6)\n"
            "for i,v in enumerate([gross, n5a, n10a]): a2.annotate(f'{v:+.2f}%',(i,v),\n"
            "    ha='center', va='top' if v<0 else 'bottom')\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('ann. return %'); a2.set_title('Already ~zero gross; net is more negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('turnover:', to.round(3).to_dict())\n"
            "print(f'net5 {n5a:+.2f}% (Sharpe {n5s:+.2f})  net10 {n10a:+.2f}% (Sharpe {n10s:+.2f})')"
        ),
        md(
            "> 💡 In plain words: costs are not the villain here (turnover tops out around 9% "
            "of NAV per month, on the commodity leg) — there's simply no gross edge for costs "
            "to eat. Net just makes an already-flat number a touch more negative."
        ),
        md(
            "### 4f · The crisis ledger — 2008 GFC and 2020 COVID, leg by leg\n\n"
            "Two hardcoded windows (facts, not fit): the Lehman aftermath and the COVID crash. "
            "If carry is compensation for a synchronized crash risk, this is where it should "
            "show up hardest."
        ),
        code(
            "if HAVE_REAL:\n"
            "    gfc = {c: st.crisis_stats(SL[c], dt.GFC_WINDOW)['cum_return_pct'] for c in SL.columns}\n"
            "    gfc['COMBO'] = st.crisis_stats(CB, dt.GFC_WINDOW)['cum_return_pct']\n"
            "    covid = {c: st.crisis_stats(SL[c], dt.COVID_WINDOW)['cum_return_pct'] for c in SL.columns}\n"
            "    covid['COMBO'] = st.crisis_stats(CB, dt.COVID_WINDOW)['cum_return_pct']\n"
            "else:\n"
            "    gfc = dict(R['gfc']); covid = dict(R['covid'])\n"
            "keys = ['FX','BOND','EQ','CMD','COMBO']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)\n"
            "for ax, d, title in [(a1, gfc, '2008 GFC (Aug-Nov)'), (a2, covid, '2020 COVID (Feb-Apr)')]:\n"
            "    vals = [d[k] for k in keys]\n"
            "    cols = [GREEN if v>0 else RED for v in vals[:-1]] + [AMBER]\n"
            "    ax.bar(keys, vals, color=cols, width=.6)\n"
            "    for i,v in enumerate(vals): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',\n"
            "        va='top' if v<0 else 'bottom', fontsize=9)\n"
            "    ax.axhline(0, c='k', lw=.8); ax.set_title(title)\n"
            "a1.set_ylabel('cumulative return in the window')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('GFC:', {k: round(v,2) for k,v in gfc.items()})\n"
            "print('COVID:', {k: round(v,2) for k,v in covid.items()})"
        ),
        md(
            f"> 💡 In plain words: the FX leg alone lost **{R['gfc']['FX']:+.1f}%** in the GFC "
            f"window — a genuine carry crash — but it happened while BOND ({R['gfc']['BOND']:+.1f}%), "
            f"EQ ({R['gfc']['EQ']:+.1f}%) and CMD ({R['gfc']['CMD']:+.1f}%) carry were all "
            f"positive, netting the combo to **{R['gfc']['combo']:+.2f}%** — near flat. COVID "
            f"flips the actor: EQ carry lost **{R['covid']['EQ']:+.1f}%** while CMD carry "
            f"*gained* **{R['covid']['CMD']:+.1f}%** (the naive-roll commodity index, more "
            "exposed to the April-2020 negative-oil-price shock, cratered harder than the "
            f"roll-optimized one — widening rather than closing the carry spread). Combo: "
            f"**{R['covid']['combo']:+.2f}%**. H₄ — the synchronized crash — is **Busted** on "
            "this tape, in both textbook windows."
        ),
        md(
            "### 4g · Faithful-engine & power control — including a plantable crash factor\n\n"
            "Synthetic four-sleeve world: independent noise around a TUNABLE planted carry "
            "mean, all sleeves loaded on a TUNABLE shared crash factor. The null (no carry, no "
            "crash) is checked over **20 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    d = st.synthetic_detect(carry_bps_mo=0.0, crash_beta=0.0, seed=660 + s_)\n"
            "    null_ts.append(d['t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "planted = st.synthetic_detect(carry_bps_mo=40.0, crash_beta=0.0, seed=660)\n"
            "planted_crash = st.synthetic_detect(carry_bps_mo=40.0, crash_beta=1.0, seed=660)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null: no carry, no crash (20 seeds)')\n"
            "ax.scatter([1], [planted['t']], color=GREEN, s=100, zorder=5,\n"
            "           label='planted carry, no crash')\n"
            "ax.scatter([2], [planted_crash['t']], color=RED, s=100, zorder=5,\n"
            "           label='SAME planted carry + a synchronized crash')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks([0,1,2]); ax.set_xticklabels(['null x20','planted\\ncarry','planted carry\\n+ crash'])\n"
            "ax.set_ylabel('HAC t (combo)')\n"
            "ax.set_title('A real premium, swamped by a shared tail factor — a plausible mechanism')\n"
            "ax.legend(fontsize=8); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t={null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "print(f\"planted (no crash): Sharpe {planted['sharpe']:+.2f}  t={planted['t']:+.2f}\")\n"
            "print(f\"planted + crash: Sharpe {planted_crash['sharpe']:+.2f}  t={planted_crash['t']:+.2f}  \"\n"
            "      f\"skew={planted_crash['skew']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: the null never fires falsely more than expected "
            f"({R['syn_null_fire']}/20 seeds, matching the ~5% rate a two-sided *t*-test allows), "
            f"and a planted +40 bps/mo carry premium lights up cleanly (*t* = "
            f"{R['syn_planted_t']:+.2f}). Bolt a **shared** crash factor onto the exact same "
            f"premium — nothing else changed — and the *t*-stat collapses to "
            f"{R['syn_crash_t']:+.2f} while skew turns sharply negative "
            f"({R['syn_crash_skew']:+.2f}). This shows *mechanically* how a real cross-asset "
            "carry premium could fail to certify on any single sample if it loads on a common "
            "tail — which makes it all the more notable that the **real** 2008/2020 windows "
            "above did **not** show that synchronized pattern. *(Faithful-engine / power check "
            "only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — equal-weight combo **{R['combo_ann']:+.2f}%/yr**, Sharpe "
            f"**{R['combo_sharpe']:+.2f}**, HAC *t* = **{R['combo_t']:+.2f}**, bootstrap 95% "
            f"Sharpe CI **[{R['combo_ci'][0]:+.2f}, {R['combo_ci'][1]:+.2f}]**. No individual "
            f"leg clears *t* ≥ 2 (best: CMD *t* = {R['sleeves']['CMD']['t']:+.2f}); the best "
            f"alternate construction (ex-equity 3-sleeve) reaches only *t* = "
            f"{R['ex_eq_t']:+.2f}. Correlations are genuinely low (|ρ| ≤ 0.30) — the "
            "diversification premise held; it just had a near-zero mean to diversify.\n"
            f"- **Tradability `MIRAGE`** — net of costs and short-leg borrow: "
            f"**{R['net5_ann']:+.2f}%/yr** (5 bps) to **{R['net10_ann']:+.2f}%/yr** (10 bps), "
            f"Sharpe {R['net5_sharpe']:+.2f} / {R['net10_sharpe']:+.2f}. Nothing to size.\n"
            f"- **\"Crashes everywhere at once?\" `BUSTED`** — 2008 combo "
            f"{R['gfc']['combo']:+.2f}% despite the FX leg's {R['gfc']['FX']:+.1f}% carry "
            f"unwind (offset by BOND/EQ/CMD); 2020 combo {R['covid']['combo']:+.2f}% despite "
            f"the EQ leg's {R['covid']['EQ']:+.1f}% slide (offset by CMD's "
            f"{R['covid']['CMD']:+.1f}% spike). The legs crash — on different clocks."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The proxy, not the theory, is the likely culprit.** KMPV's original test uses "
            "live deposit rates, dozens of instruments per asset class and monthly re-ranking; "
            "our four static 2-vs-2 baskets are a coarser instrument on free data. A richer "
            "rebuild (real futures curves for commodities, a full G10 cross-section for FX, an "
            "options-implied dividend-swap proxy for equities) is the natural sequel.\n"
            "- **The equity leg deserves its own dedicated study** — dividend yield vs growth "
            "mechanically nets out to the value factor over 2007-2026's growth-dominated "
            "regime; disentangling \"carry\" from \"value\" in equities is a real methodological "
            "question the original paper handles with more instruments than we can here.\n"
            "- **Dedup map:** [364-fx-carry-trade](../../364-fx-carry-trade/) (the FX leg, done "
            "properly, its own *t* = 1.14), [612-em-debt-carry](../../612-em-debt-carry/) (one "
            "packaged-carry sleeve in depth), [638-value-momentum-everywhere](../../638-value-"
            "momentum-everywhere/) (same combo architecture, a different signal pair, the same "
            "kind of statistical zero).\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
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
