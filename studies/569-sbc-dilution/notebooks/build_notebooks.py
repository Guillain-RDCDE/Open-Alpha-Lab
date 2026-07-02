"""Generate the two narrative notebooks for Study 569 (SBC-Dilution).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic figures
run anywhere, offline and deterministic; the real-tape cells use the cached prices + shares + SBC
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-30; formation
# 2016 -> 2024, held t->t+1; 31 survivors; joint panel fp bbec9e324ac1).
R = dict(
    fp_px="db57cbb8d955", fp_sh="9629ac4ff009", fp_sbc="75f937c63b8e", fp_panel="bbec9e324ac1",
    n_names=31, n_years=9,
    long_mean=17.4, short_mean=32.6, ls_mean=-15.2, ls_t=-2.01,
    win=22.0, sharpe=-0.67, placebo_p=0.998,
    net=-15.9, net_t=-2.10, cost_bps=10.0, borrow_bps=50.0,
    # each-leg cut: (label, years, ls_mean%, t)
    legs=[("composite", 9, -15.2, -2.01), ("SBC-only", 3, -41.3, -1.92),
          ("dilution-only", 9, -8.2, -1.37)],
    # width sweep: (frac, ls_mean%, t)
    widths=[(0.2, -13.8, -1.91), (0.3, -15.2, -2.01), (0.4, -16.5, -1.99)],
    # synthetic control (n_years=6, 25 seeds): (edge, mean_t, ls_mean%)
    ctrl=[(0.0, 0.06, -0.3), (0.10, 1.00, 4.8), (0.20, 1.81, 9.9), (0.30, 2.42, 15.0)],
    short_leg="NVDA, META, AMZN, MSFT, CRM, AMD, ORCL, QCOM, INTC",
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

from sbc_dilution import data, strategy as st

def load_real():
    \"\"\"Cache-first real (prices, shares, sbc); drop the partial current year.\"\"\"
    if not data.have_real():
        return None
    px, sh, sbc = data.load_real()
    return data.drop_partial_last_year(px, sh, sbc, asof=pd.Timestamp("2026-06-30"))

REAL = load_real()
HAVE_REAL = REAL is not None
if HAVE_REAL:
    PX, SH, SBC = REAL
    print("real SBC/dilution panel present:", HAVE_REAL,
          f"({PX.shape[1]} names, formation {PX.index[0].year+1} -> {PX.index[-1].year})")
else:
    print("real panel absent — using frozen headline numbers (offline)")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# SBC-Dilution — do the biggest stock-printers quietly dilute you into worse returns? 🧾\n"
            "### Sorting big companies by how much of themselves they hand out each year, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Replicates_here%3F: Busted](https://img.shields.io/badge/Replicates_here%3F-Busted-8b949e?style=flat-square)\n\n"
            "Here's a tidy Wall-Street idea. When a company pays its employees in **stock** instead of "
            "cash (stock-based comp, 'SBC'), that's a real cost — it quietly slices everyone else's "
            "ownership a little thinner every year (**dilution**). But because SBC barely dents the "
            "headline profit number, the theory goes, investors *under-notice* it — so the companies "
            "handing out the most stock should go on to earn the **lowest** returns.\n\n"
            "So we test it the cheap way: take 31 big-name stocks, score each one for how much it "
            "dilutes (lots of stock-comp, fast-growing share count), and see whether the *lean* ones "
            "beat the *heavy diluters*.\n\n"
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
            f"| Did the *lean* firms beat the *heavy diluters* (the anomaly)? | **No — the opposite.** "
            f"Over 9 years the long-lean / short-diluters trade lost **{R['ls_mean']}%/yr**. The heavy "
            f"stock-printers *won*. |\n"
            f"| Is that just luck? | **No** — a shuffle test beats the real sort **{R['placebo_p']*100:.1f}%** "
            "of the time. The real sort sits deep in the *losing* tail. |\n"
            f"| Could you trade it? | **No.** It points the wrong way, wins only {R['win']:.0f}% of years, "
            "and the stocks you'd short are exactly the hard-to-borrow mega-caps. |\n"
            "| So is the anomaly fake? | **Not in general** — but it needs the full market and the firms "
            "that diluted themselves to death. Our basket only has survivors, in an AI melt-up that "
            "rewarded the biggest stock-printers. |\n\n"
            "> The idea is real in the textbooks — but the textbook version needs the companies that "
            "*actually failed*, and a normal market. Our basket is 31 survivors during a growth boom "
            "that paid the heavy diluters the most."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"Firms that dilute shareholders the most — via heavy stock-based comp — earn lower "
            "future returns.\"* — an accounting hidden-cost anomaly, a cousin of the share-issuance "
            "factor (Pontiff-Woodgate 2008).\n\n"
            "The intuition: stock-based comp is a genuine expense (you're paying people with slices of "
            "the company), but it hides in a footnote instead of the profit line. If investors "
            "under-weight it, the heavy stock-printers look cheaper than they are — and go on to "
            "disappoint. The lean companies that *shrink* their share count (buybacks) should quietly "
            "win."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If true, it's an easy filter: avoid (or short) the serial diluters, own the lean "
            "buyback-ers. And it says something uncomfortable — that a big chunk of 'growth' is really "
            "just the company printing itself smaller. The desk has already looked at the pure "
            "share-count version ([519 Net-Share-Issuance](../../519-net-share-issuance/)); here we add "
            "the **stock-comp** flavour and go straight at the *return* question."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Score dilution.** For each firm each year: how much stock-comp it pays (as a share of "
            "revenue) **plus** how fast its share count is growing. Add them up — higher = more "
            "dilutive.\n"
            "2. **Sort.** Split the 31 names into a *lean* third and a *heavy-diluter* third.\n"
            "3. **Compare next-year returns.** Did the lean third beat the diluter third? The anomaly "
            "says yes.\n\n"
            "Catch: our basket is companies **still alive in 2026** — so any firm that diluted itself "
            "into the ground is missing. And we only get a few years of stock-comp data. We'll see both "
            "bite.\n\n"
            "*Timing:* the score uses only data public at year-end; we trade the *next* year and hold. "
            "No peeking at the future."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Who won — the lean third or the heavy-diluter third?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize(PX, SH, SBC, frac=0.3, placebo=False)\n"
            "    lean_m, dil_m, ls = s['long_mean']*100, s['short_mean']*100, s['ls_mean']*100\n"
            "    banner = f'REAL tape ({PX.shape[1]} survivors, formation 2016 -> 2024)'\n"
            "else:\n"
            f"    lean_m, dil_m, ls = {R['long_mean']}, {R['short_mean']}, {R['ls_mean']}\n"
            "    banner = 'frozen real numbers (offline)'\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['LEAN third\\n(low dilution)','HEAVY-DILUTER third'], [lean_m, dil_m], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('avg next-year return %')\n"
            "ax.set_title(f'The anomaly says LEAN wins. Here the DILUTERS won (+{dil_m:.0f}% vs +{lean_m:.0f}%/yr)')\n"
            "fig.suptitle(banner, fontsize=9, color=GREY); plt.tight_layout(); plt.show()\n"
            "print(f'Lean +{lean_m:.1f}%/yr  vs  Diluters +{dil_m:.1f}%/yr  ->  long-short {ls:+.1f}%/yr')"
        ),
        md(
            f"There it is, **backwards**. The anomaly predicts the *lean* names win. On 2016-24 the "
            f"**heavy diluters** won by miles (+{R['short_mean']}% vs +{R['long_mean']}%/yr) — the "
            f"long-lean / short-diluters trade is **{R['ls_mean']}%/yr**, a clean loss the *wrong* way. "
            f"Why? The heavy-dilution bucket is the SBC-soaked AI mega-caps ({R['short_leg']}) — and "
            "they led a melt-up."
        ),
        md(
            "**Is it just luck?** Shuffle which company gets which dilution score, thousands of times, "
            "and see how often a *random* sort beats the real one:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pb = st.placebo_pvalue(PX, SH, SBC, frac=0.3, n_draws=5000, seed=569)\n"
            "    p = pb['p_value']\n"
            "else:\n"
            f"    p = {R['placebo_p']}\n"
            "fig, ax = plt.subplots(figsize=(7, 3.2))\n"
            "ax.barh(['random shuffle beats\\nthe real sort'], [p*100], color=RED)\n"
            "ax.axvline(50, ls='--', c=GREY, lw=1); ax.set_xlim(0, 100); ax.set_xlabel('% of shuffles')\n"
            "ax.set_title(f'A random relabelling beats the real dilution sort {p*100:.1f}% of the time')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'placebo p = {p:.3f}  (the real sort is worse than {p*100:.1f}% of random ones)')"
        ),
        md(
            f"The real sort loses to a coin-flip relabelling **{R['placebo_p']*100:.1f}%** of the time. "
            "So this isn't just a weak signal — on this window it's a genuine *inversion*: the diluters "
            "really did out-earn the lean names."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The long-lean / short-diluters spread is the *wrong* sign "
            f"({R['ls_mean']}%/yr), the placebo beats it {R['placebo_p']*100:.1f}% of the time, and it "
            f"wins only {R['win']:.0f}% of years. No bankable anomaly here.\n"
            "- **Tradability — Mirage.** A survivor basket, annual rebalance, with the diluter short "
            "leg being the costly-to-borrow mega-cap tech. Nothing to harvest.\n"
            "- **Replicates here? — Busted.** The effect that lives in the full market doesn't survive "
            "a 31-name survivor sort in a growth melt-up."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No. Even ignoring the wrong sign, the trade means *shorting* the heaviest diluters — which "
            "here are NVDA, META, Amazon and friends, exactly the stocks that are expensive and hard to "
            "borrow — on a basket that has already quietly deleted every company that diluted itself to "
            "death. The real anomaly lives in the firms that *failed*; a blue-chip survivor sort can't "
            "see them."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Its pure cousin.** [519 Net-Share-Issuance](../../519-net-share-issuance/) is the same "
            "idea with just the share count (no stock-comp leg) — and lands the same place.\n"
            "- **The real test needs the dead.** Re-run this with delisted / diluted-to-death firms "
            "included (a survivorship-free universe), deep point-in-time SBC data, and a full cycle — "
            "that's where the anomaly earns its keep.\n\n"
            "*Think the anomaly is real and we just missed it? Fork this, add serial-diluter blow-ups "
            "and a non-melt-up window, and show the long-lean / short-diluters spread clearing t = 2 "
            "the right way.*"
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
            "# SBC-Dilution — a quantitative teardown 🔬\n"
            "### dilution score (z(SBC/rev)+z(share growth)) · annual sort · one-sample *t* · label-shuffle placebo · each-leg & width robustness · costs & borrow · synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Replicates_here%3F: Busted](https://img.shields.io/badge/Replicates_here%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We build an SBC/dilution score, "
            "sort the cross-section annually, and test whether the lean names beat the heavy diluters "
            "(the anomaly) — or, here, the reverse.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance year-end prices + split-adjusted shares "
            f"+ an SBC/revenue snapshot, {R['n_names']} survivors (as-of 2026-06-30, joint panel fp "
            f"`{R['fp_panel']}`); the offline core and tests run on a deterministic synthetic world. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Long-lean / short-diluters spread **{R['ls_mean']}%/yr** "
            f"(one-sample *t* **{R['ls_t']}**, placebo *p* {R['placebo_p']}) — the *wrong* sign; wins "
            f"{R['win']:.0f}% of years; wrong sign in every leg & width. |\n"
            f"| **Tradability** | `MIRAGE` | Survivor basket, annual rebalance, diluter short = costly "
            f"borrow. Gross {R['ls_mean']}% → net {R['net']}% ({R['cost_bps']:.0f} bps/leg + "
            f"{R['borrow_bps']:.0f} bps borrow). |\n"
            "| **Replicates here?** | `BUSTED` | A canonical accounting factor, honestly replicated on a "
            "small survivor basket over an AI melt-up, inverts — sign reverses, *t* never clears +2, "
            "placebo unbeaten. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it, with the *right* "
            "sign), but on this survivor basket over this window the anomaly is inverted and stably so."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $D_i$ be firm $i$'s dilution score (heavy SBC + fast share growth) and $r_i$ its "
            "next-year return.\n\n"
            "- **H₁ (the anomaly / long-short).** $\\mathbb{E}[r \\mid \\text{lean}] - "
            "\\mathbb{E}[r \\mid \\text{diluter}] > 0$ at *t* ≥ 2 (lean beats diluters).\n"
            "- **H₂ (each leg).** The sign holds for SBC-intensity alone *and* share-growth alone.\n"
            "- **H₃ (robustness).** The sign is stable across quantile widths.\n"
            "- **H₄ (net).** H₁ survives costs and the diluter-leg borrow.\n\n"
            "We find **H₁ rejected** (wrong sign, *t* −2.0), **H₂ rejected** (both legs negative), "
            "**H₃ rejected the wrong way** (stably negative), and **H₄ moot** (the trade is the wrong "
            "sign before costs)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why**. The SBC/dilution anomaly is documented on full-universe, "
            "survivorship-free data driven by firms that over-issue and disappoint. Two forces kill it "
            "here: (a) **survivorship** — the basket already dropped every firm that diluted itself to "
            "death, removing the left tail that makes diluters lose on average; and (b) an **AI/growth "
            "melt-up** (2020-24) that rewarded the SBC-heavy mega-caps the score flags as diluters. "
            "That echoes [519 Net-Share-Issuance](../../519-net-share-issuance/) and "
            "[540 Distress-Risk](../../540-distress-risk-anomaly/), which invert on the same kind of "
            "survivor tape."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Score.** $D = z(\\text{SBC}/\\text{rev}) + z(\\Delta\\text{shares})$, each z-scored "
            "across the basket each year (higher = more dilutive). Shares are **split-adjusted** so a "
            "split isn't mistaken for issuance.\n"
            "- **Sort.** Tercile tails (≈9 names each): lean (lowest $D$) vs diluters (highest).\n"
            "- **Inference.** One-sample *t* of the annual long-short mean vs 0; a **label-shuffle "
            "placebo** null (permute which name carries which score).\n"
            "- **Robustness.** Each leg alone (SBC-only, dilution-only); a quantile-width sweep.\n"
            "- **Frictions.** One-way cost × turnover on both legs + an annual borrow on the diluter "
            "short.\n"
            "- **Positive control.** A deterministic synthetic world with a planted anomaly "
            "(`edge > 0`, heavy-diluters underperform) and a null — averaged over 25 seeds (house "
            "rule).\n\n"
            "Timing: the score is public at year-end *t*; we trade *t→t+1* and hold (one execution "
            "lag). No future data enters the score. **Data limits, named:** yfinance exposes only a "
            "shallow ~4-year SBC history, so the SBC leg is a recent snapshot forward-filled onto the "
            "grid; the basket is large-cap **survivors**. Both cap the stamp — a survivor / snapshot "
            "study can never be `REAL`."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The sort — H₁\n\n"
            "Lean vs diluter next-year returns, with the one-sample *t* and the placebo null."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize(PX, SH, SBC, frac=0.3, placebo=True)\n"
            "    lean_m, dil_m, ls, t, p = (s['long_mean']*100, s['short_mean']*100, s['ls_mean']*100,\n"
            "                               s['t'], s['p_placebo'])\n"
            "else:\n"
            f"    lean_m, dil_m, ls, t, p = {R['long_mean']}, {R['short_mean']}, {R['ls_mean']}, {R['ls_t']}, {R['placebo_p']}\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['LEAN','DILUTERS'], [lean_m, dil_m], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('avg next-year return %')\n"
            "ax.set_title(f'Lean - diluters = {ls:+.1f}%/yr  (one-sample t {t:+.2f}, placebo p {p:.3f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Lean +{lean_m:.1f}%/yr | Diluters +{dil_m:.1f}%/yr | LS {ls:+.1f}%/yr | t {t:+.2f} | placebo p {p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: H₁ **rejected, with the sign reversed.** The spread is "
            f"**{R['ls_mean']}%/yr** (*t* {R['ls_t']}), and the placebo *p* = {R['placebo_p']} says the "
            "real sort is worse than 998 of every 1000 random relabellings — the diluters genuinely "
            "*out*-earned the lean names on this window."
        ),
        md(
            "### 4b · Each leg alone — H₂\n\n"
            "Does the sign survive when we sort on SBC intensity alone, or share growth alone?"
        ),
        code(
            "legs = " + repr(R["legs"]) + "\n"
            "if HAVE_REAL:\n"
            "    rows = [(r['signal'], r['n_years'], r['ls_mean']*100, r['t'])\n"
            "            for r in st.leg_sweep(PX, SH, SBC, frac=0.3)]\n"
            "else:\n"
            "    rows = legs\n"
            "tab = pd.DataFrame(rows, columns=['leg','years','ls_mean%','t']).set_index('leg')\n"
            "fig, ax = plt.subplots(figsize=(8, 4.0))\n"
            "cols = [GREEN if v>0 else RED for v in tab['ls_mean%']]\n"
            "ax.bar(range(len(tab)), tab['ls_mean%'], color=cols, width=.55)\n"
            "ax.set_xticks(range(len(tab))); ax.set_xticklabels(tab.index)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('long-short %/yr')\n"
            "ax.set_title('Every leg points the WRONG way (all red = anomaly inverted)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(2).to_string())"
        ),
        md(
            "> 💡 In plain words: H₂ **rejected.** Composite **−15.2%**, SBC-only **−41.3%** (3 snapshot "
            "years), dilution-only **−8.2%** — all negative. A hidden cost being under-priced would show "
            "a *positive* spread in at least one leg; none does."
        ),
        md(
            "### 4c · Robustness — H₃ (does the width matter?)\n\n"
            "The same sort at three quantile widths."
        ),
        code(
            "widths = " + repr(R["widths"]) + "\n"
            "if HAVE_REAL:\n"
            "    rows = [(r['frac'], r['ls_mean']*100, r['t']) for r in st.width_sweep(PX, SH, SBC)]\n"
            "else:\n"
            "    rows = widths\n"
            "tab = pd.DataFrame(rows, columns=['frac','ls_mean%','t']).set_index('frac')\n"
            "fig, ax = plt.subplots(figsize=(8, 4.0))\n"
            "ax.bar([str(f) for f in tab.index], tab['ls_mean%'], color=RED, width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('long-short %/yr'); ax.set_xlabel('tail fraction')\n"
            "ax.set_title('Stably inverted: near -15%/yr at every width (t ~ -2)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(2).to_string())"
        ),
        md(
            "> 💡 In plain words: H₃ **rejected the wrong way.** The inversion isn't a tail-width "
            "artefact — it's the same, near *t* −2, at 20/30/40% cuts. Stable, but stably backwards."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — H₁ rejected (spread {R['ls_mean']}%, *t* {R['ls_t']}, wrong sign); "
            "H₂ rejected (every leg negative); H₃ rejected (stably inverted).\n"
            f"- **Tradability `MIRAGE`** — survivor basket, annual rebalance, diluter-leg borrow; "
            f"gross {R['ls_mean']}% → net {R['net']}%.\n"
            "- **Replicates here? `BUSTED`** — the full-market anomaly inverts on 31 survivors."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "The trade is the wrong sign before costs; the borrow just deepens it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = st.net_of_costs(PX, SH, SBC, frac=0.3, cost_bps=10.0, borrow_ann_bps=50.0)\n"
            "    gross, net = c['gross_mean']*100, c['net_mean']*100\n"
            "else:\n"
            f"    gross, net = {R['ls_mean']}, {R['net']}\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross','net\\n(costs+borrow)'], [gross, net], color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('long-short %/yr')\n"
            "ax.set_title(f'Wrong sign before costs ({gross:+.0f}%/yr), worse after ({net:+.0f}%/yr)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:+.1f}%/yr -> net {net:+.1f}%/yr (10 bps/leg x 2 + 50 bps borrow)')"
        ),
        md(
            "> 💡 In plain words: there is nothing to charge costs against — the long-lean / "
            "short-diluters book *loses* before frictions on this window. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a real anomaly "
            "(`edge > 0`, heavy-diluters underperform) of increasing strength and watch the long-short "
            "*t* go **positive** — averaged over 25 seeds so no single lucky seed can fake it."
        ),
        code(
            "ctrl = " + repr(R["ctrl"]) + "\n"
            "edges = [0.0, 0.10, 0.20, 0.30]\n"
            "rows = [st.synthetic_control(data.synthetic_panel, edges=[e], n_years=6, n_seeds=25)[0]\n"
            "        for e in edges]\n"
            "ts = [r['mean_t'] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(edges, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = +2 bar (anomaly)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted edge (larger = stronger anomaly)')\n"
            "ax.set_ylabel('mean long-short t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted anomaly the RIGHT way — flat at null, past +2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for e, r in zip(edges, rows): print(f\"edge {e:+.2f} -> mean t {r['mean_t']:+.2f}, ls_mean {r['ls_mean']*100:+.1f}%\")"
        ),
        md(
            "The long-short *t* is ≈ 0 at the null and rises past **+2** as the planted anomaly grows — "
            "so the engine is a faithful detector *and* points the right way when the effect is real. "
            "The real-tape **negative** *t* is therefore a statement about **this survivor basket on "
            "this window**: the SBC/dilution anomaly is inverted and stably so here. For the pure "
            "share-count cousin, see [519 Net-Share-Issuance](../../519-net-share-issuance/)."
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
