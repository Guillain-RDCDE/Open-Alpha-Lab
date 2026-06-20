"""Generate the two narrative notebooks for Study 328 (Benford-Law).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures (the Benford positive control and the non-Benford anti-control) run anywhere,
offline and deterministic; the real-tape cells use the shared/quantlab parquet cache if
present and otherwise fall back to the synthetic tape, clearly bannered, quoting the
frozen headline numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs
end-to-end for any reader.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-01).
R = dict(
    # Single-name PRICE conformity (split-adjusted) — they FAIL Benford
    spy_dec=1.24, spy_mad=0.0416, spy_chi=1407,
    qqq_dec=1.57, qqq_mad=0.0528,
    gld_dec=1.08, gld_mad=0.0821,
    tlt_dec=0.33, tlt_mad=0.1338,
    aapl_dec=3.80, aapl_mad=0.0313,
    msft_dec=3.78, msft_mad=0.0710,
    # RETURNS conform (span many orders of magnitude in absolute value)
    spy_ret_mad=0.0086,
    nigrini_nonconf=0.015,
    # The forensic cross-sectional sort (long-conforming / short-deviant), real panel
    panel_names=366, panel_n=317, panel_fp="ad9a46a7c9bc",
    main_ann=0.0876, main_t=3.58, main_ci_lo=0.0035, main_ci_hi=0.0116,
    pre_t=2.02, post_t=4.32,
    chi_ann=0.0554, chi_t=2.49,
    net10_ann=0.0656, net10_t=2.65,
    randbase_ann=0.0862, randbase_t=3.40,
    # The confounds
    corr_dev_trail=-0.227,            # deviation vs trailing 12m return
    mom_ann=-0.0339, mom_t=-1.10,     # plain 12m momentum on the same panel (so it's NOT momentum)
    # Synthetic controls
    syn_ben_dec=2.57, syn_ben_mad=0.0175,
    syn_non_dec=0.11, syn_non_mad=0.1831,
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from benford_law import data, strategy as st

def have_real(ticker="SPY"):
    return data.cache_available(ticker)

HAVE_REAL = have_real()
print("real price cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Benford's Law — do stock prices really obey it, and can the deviation flag trouble? 🔢\n"
            "### A forensic-accounting trick, pointed at the market, in plain English\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![A_forensic_red--flag%3F: Not_supported](https://img.shields.io/badge/A_forensic_red--flag%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "There's a famous bit of number-magic called **Benford's Law**: in many real-world "
            "datasets the first digit is a **1** about 30% of the time, a **9** less than 5% — not "
            "the flat ~11% each you'd guess. Auditors use it to catch faked numbers. The viral "
            "trading version says: *if a stock's prices stray from Benford's Law, something's wrong "
            "— and you can trade the deviation.* This notebook checks whether prices obey the law at "
            "all, and whether the deviation is worth a dime.\n\n"
            "> **This is the plain-language layer.** Want the χ² tests, the survivorship confound and "
            "the HAC *t*-stats? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do *prices* obey Benford's Law? | **A single stock's prices — no.** SPY's first-digit "
            f"deviation (MAD **{R['spy_mad']:.3f}**) is way past the 'nonconformity' line "
            f"({R['nigrini_nonconf']:.3f}) — because SPY only ranged ~{R['spy_dec']:.1f} decades and "
            "sat in the 400s for years. |\n"
            "| Then what *does* obey it? | **Returns.** A day's % move spans tiny to huge, so its "
            f"leading digit is beautifully Benford (SPY abs-returns MAD **{R['spy_ret_mad']:.3f}**). "
            "The folklore points at the wrong column. |\n"
            "| Does a 'deviation' flag trouble? | **No.** A high price-deviation just means the "
            "price lived in a *narrow range* — that's geometry, not fraud or weakness. |\n"
            "| Can you trade the deviation? | **No.** The only place a deviation-sort looks "
            f"profitable (*t* ≈ +{R['main_t']:.1f}) is a **survivorship-biased** list of today's "
            "winners — a universe you couldn't have known in advance. |\n\n"
            "> Benford's Law is a real tool — for *auditing reported figures*. Stretched into a "
            "price-trading signal it measures the width of a price range and the luck of survival, "
            "and calls it 'integrity.'"
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Stock prices, like most natural data, follow Benford's Law. When a stock's prices "
            "**deviate** from the expected first-digit distribution, it's a red flag — a sign of "
            "manipulation or instability. Screen for high-deviation names and avoid (or short) "
            "them.\"*\n\n"
            "— the popular forensic-trading framing (descended from Nigrini's audit work and "
            "Ley's 1996 finding that the S&P 500 *index* digits broadly conform)\n\n"
            "The respectable core is real: Benford's Law genuinely catches fabricated accounting "
            "numbers, and the **index over a century** does conform. The leap under test is whether "
            "an **individual security's price** conforms, whether a *deviation* means trouble, and "
            "whether you can *trade* it."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it worked, you'd have a free fraud-and-fragility detector: rank the market by a "
            "first-digit score, dodge the dodgy names, pocket the difference. It would also be a "
            "deep statement — that price *integrity* leaves a fingerprint in the digits. But "
            "Benford's Law has a precondition that's easy to forget: it only holds for numbers that "
            "**span several orders of magnitude on a log scale.** Whether a stock price meets that "
            "bar — and what a 'deviation' therefore *means* — is the whole game."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three checks:\n\n"
            "1. **Does the law even hold?** Take real prices, count the leading digits, compare to "
            "Benford. Do the same for *returns*. (We build two synthetic tapes first — one we "
            "*know* should conform, one we *know* shouldn't — to prove our measuring stick works.)\n"
            "2. **What does a deviation track?** Correlate a name's deviation with its price *range* "
            "and its recent *return* — is it integrity, or just geometry?\n"
            "3. **Can you trade it?** Every month, short the most-deviant names and long the most-"
            "conforming, and measure the spread honestly — then ask what survives once you remember "
            "the list of names is *today's* survivors.\n\n"
            "If the deviation is just price-range, and the trade only works on survivors, the verdict "
            "is **mirage**."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, does our measuring stick work?** Two synthetic price tapes: one that wanders "
            "across orders of magnitude (should be Benford), one stuck in a narrow band (shouldn't)."
        ),
        code(
            "sb, tb = data.synthetic_benford(); sn, tn = data.synthetic_nonbenford()\n"
            "pmf = st.benford_pmf(); digs = np.arange(1, 10)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)\n"
            "a1.bar(digs, st.digit_frequencies(sb), color=GREEN, alpha=.8, label='wide-range price')\n"
            "a1.plot(digs, pmf, 'ko-', lw=2, label='Benford')\n"
            "a1.set_title(f'Wide-range walk → Benford (MAD {st.mad(sb):.3f})'); a1.legend()\n"
            "a1.set_xlabel('leading digit'); a1.set_ylabel('frequency')\n"
            "a2.bar(digs, st.digit_frequencies(sn), color=RED, alpha=.8, label='range-bound price')\n"
            "a2.plot(digs, pmf, 'ko-', lw=2, label='Benford')\n"
            "a2.set_title(f'Range-bound price → NOT Benford (MAD {st.mad(sn):.3f})'); a2.legend()\n"
            "a2.set_xlabel('leading digit')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'positive control spans {tb[\"log_range_decades\"]:.2f} decades; '\n"
            "      f'anti-control spans {tn[\"log_range_decades\"]:.2f} decades')"
        ),
        md(
            "The stick works: a price that ranges over decades is Benford; a price pinned to a band "
            "isn't (its digits are all 4s and 5s). **Benford is a statement about *range*.** Hold "
            "that thought."
        ),
        md(
            "**Now the real thing.** Here's a single ETF's actual prices vs Benford — and, beside it, "
            "the same ETF's *returns*:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    spy = data.load_real('SPY')\n"
            "    px_freq = st.digit_frequencies(spy); px_mad = st.mad(spy)\n"
            "    rt_freq = st.digit_frequencies(spy.pct_change().abs().dropna()); rt_mad = st.mad(spy.pct_change().abs().dropna())\n"
            "    banner = 'REAL TAPE — SPY split-adjusted'\n"
            "else:\n"
            "    sb, _ = data.synthetic_benford(); nb_, _ = data.synthetic_nonbenford()\n"
            "    px_freq = st.digit_frequencies(nb_); px_mad = st.mad(nb_)\n"
            "    rt_freq = st.digit_frequencies(sb); rt_mad = st.mad(sb)\n"
            "    banner = 'SYNTHETIC FALLBACK (no cache) — illustrative only'\n"
            "pmf = st.benford_pmf(); digs = np.arange(1, 10)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)\n"
            "a1.bar(digs, px_freq, color=RED, alpha=.8); a1.plot(digs, pmf, 'ko-', lw=2)\n"
            "a1.set_title(f'PRICES fail Benford (MAD {px_mad:.3f})'); a1.set_xlabel('leading digit'); a1.set_ylabel('freq')\n"
            "a2.bar(digs, rt_freq, color=GREEN, alpha=.8); a2.plot(digs, pmf, 'ko-', lw=2)\n"
            "a2.set_title(f'RETURNS conform (MAD {rt_mad:.3f})'); a2.set_xlabel('leading digit')\n"
            "fig.suptitle(banner, fontsize=10, color=GREY); plt.tight_layout(); plt.show()\n"
            "print(banner)"
        ),
        md(
            f"There's the punchline of the science: a **single** stock's *prices* do **not** obey "
            f"Benford (SPY MAD **{R['spy_mad']:.3f}**, far past the {R['nigrini_nonconf']:.3f} "
            "nonconformity line) — it never spanned enough decades and dwelt in the 400s for years. "
            f"The thing that *does* conform is **returns** (MAD **{R['spy_ret_mad']:.3f}**), because "
            "a daily move ranges from a whisper to a crash. The folklore has it backwards."
        ),
        md(
            "**So can you trade the 'deviation' anyway?** Short the most-deviant names, long the "
            "most-conforming, rebalanced monthly. On a panel of today's large caps it looks like a "
            "winner — until you notice *which* names are on the list:"
        ),
        code(
            "labels = ['Deviation sort\\n(survivor panel)', 'Plain 12-month\\nmomentum (same panel)']\n"
            f"anns = [{R['main_ann']*100:.2f}, {R['mom_ann']*100:.2f}]\n"
            f"ts = [{R['main_t']:.2f}, {R['mom_t']:.2f}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "b = ax.bar(labels, anns, color=[AMBER, GREY], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised spread (%/yr)')\n"
            "ax.set_title('Looks tradable — on a survivorship-biased universe')\n"
            "for bar_, t_ in zip(b, ts):\n"
            "    ax.annotate(f't={t_:+.2f}', (bar_.get_x()+bar_.get_width()/2, bar_.get_height()),\n"
            "                ha='center', va='bottom' if bar_.get_height()>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('Deviation sort t={R['main_t']:+.2f} — but the panel is CURRENT members (survivors).')"
        ),
        md(
            f"The deviation sort prints **+{R['main_ann']*100:.1f}%/yr** at *t* = +{R['main_t']:.1f}. "
            "But the universe is **current** large caps reconstructed after the fact — the textbook "
            "survivorship trap. A name flagged 'low-deviation / conforming' is, on this list, a name "
            "that *kept climbing across decades* — i.e. one that survived. We're not detecting "
            "integrity; we're re-discovering that survivors survived. (It isn't even plain momentum: "
            f"vanilla 12-month momentum on the same panel is *negative*, {R['mom_ann']*100:+.1f}%/yr.)"
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Mixed.** The deviation sort clears the bar (*t* ≈ +{R['main_t']:.1f}) — but "
            "only on a survivorship-biased panel, and the premise ('prices obey Benford') is false "
            "for a single name. Significant-looking, uncertifiable.\n"
            "- **Tradability — Mirage.** It needs a year of trailing prices *and* a universe you "
            "couldn't have known in real time. The deviation is a proxy for price-range and survival.\n"
            "- **A forensic red-flag for trouble? — Not supported.** A single price's deviation is "
            "geometry (how narrow its range), not manipulation. The conforming quantity is *returns*, "
            "which carries no such 'trouble' reading at all."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Set survivorship aside for a second and just ask about costs and capacity. The score "
            "needs a **252-day trailing window** before it says anything, the spread is modest, and "
            "the whole apparent edge evaporates the moment you swap the survivor list for a real, "
            "point-in-time universe (which we can't even build offline here — that's the tell). "
            "Costs are the *least* of its problems: even at 10 bps round-trip the survivor-panel "
            f"number is still +{R['net10_ann']*100:.1f}%/yr — but it's +{R['net10_ann']*100:.1f}% of "
            "a mirage. There is nothing here to size."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Where Benford is genuinely useful.** On *reported financials* (revenue, earnings) — "
            "its home turf — first-digit screens really do flag manipulation. That's the EDGAR data "
            "the desk caches; a fork could screen reported fundamentals, not prices.\n"
            "- **Second-digit and last-digit tests.** Nigrini's full toolkit goes beyond the first "
            "digit; do they carry any more information on prices? (Bet: still no — same range "
            "problem.)\n"
            "- **The honest control.** Re-run the sort on a *point-in-time* membership panel (through "
            "the survivorship opt-in guard) and watch the *t* collapse. That's the experiment that "
            "settles it.\n\n"
            "*Think the digits hide a real signal? Fork this, build a point-in-time universe, and "
            "show the deviation sort surviving once the survivors stop choosing themselves.*"
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
            "# Benford's Law on the market — a quantitative teardown 🔬\n"
            "### First-digit conformity (χ² / Nigrini MAD) · cross-sectional sort · survivorship & "
            "level-base confounds · HAC inference · synthetic controls\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![A_forensic_red--flag%3F: Not_supported](https://img.shields.io/badge/A_forensic_red--flag%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether single-name "
            "prices conform to Benford, what a 'deviation' actually measures, and whether a "
            "deviation-sorted long/short carries return information once the survivorship and "
            "fabricated-level confounds are confronted.\n\n"
            "> **Not investment advice.** Real data: the desk's shared `quantlab.data` price cache "
            "(SPY/QQQ/… split-adjusted) and the current-membership `daily_panel`; as-of 2026-06-01; "
            "the offline core and tests run on deterministic synthetic tapes. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | Long-conforming/short-deviant sort: ann **+{R['main_ann']*100:.1f}%**, "
            f"HAC *t* = **+{R['main_t']:.2f}**, block-boot CI [{R['main_ci_lo']*100:.2f}%, "
            f"{R['main_ci_hi']*100:.2f}%] **per period** — but on a **survivorship-biased** panel; "
            "premise (single-name prices ~ Benford) is **false**. |\n"
            f"| **Tradability** | `MIRAGE` | Needs a 252-day window *and* a point-in-time universe "
            "you can't reconstruct; the score proxies price-range & survival, not a tradable edge. |\n"
            f"| **A forensic red-flag?** | `NOT SUPPORTED` | SPY price MAD **{R['spy_mad']:.3f}** "
            f"(≫ {R['nigrini_nonconf']:.3f} nonconformity) is *range*, not integrity; *returns* "
            f"conform (MAD {R['spy_ret_mad']:.3f}). Deviation ⟂ momentum ({R['mom_t']:+.2f}). |\n\n"
            "> 💡 In plain words: a famous audit tool, mis-aimed. Benford lives on quantities that "
            "span orders of magnitude; a single price doesn't, so its 'deviation' is geometry. The "
            "one place a deviation-sort pays is a survivor panel — exactly the bias that fabricates "
            "cross-sectional premia."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Benford: $P(D_1=d) = \\log_{10}(1 + 1/d)$, $d\\in\\{1,\\dots,9\\}$. Conformity via "
            "Pearson $\\chi^2$ and Nigrini's MAD $= \\tfrac1 9\\sum_d |f_d - p_d|$.\n\n"
            "- **H₁ (prices conform).** A single security's price first-digits ~ Benford "
            "(MAD below the ~0.015 nonconformity line).\n"
            "- **H₂ (deviation = integrity).** A high deviation reflects manipulation/instability, "
            "not a mechanical property of the price path.\n"
            "- **H₃ (deviation predicts returns).** Sorting on a trailing deviation score yields a "
            "long-conforming/short-deviant spread with HAC $t \\ge 2$ on a *valid* universe.\n\n"
            "We find **H₁ rejected** (single-name prices fail; *returns* conform), **H₂ rejected** "
            "(deviation tracks price-range), and **H₃ supported only on a survivorship-biased panel** "
            "— i.e. not validly."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If H₁–H₃ held, Benford would be a costless integrity-and-return screen and a statement "
            "that price *honesty* is encoded in digits. The interesting content is *why* it fails: "
            "the law is scale-invariant and needs a log-uniform spread (Hill 1995); a single price "
            "over 30 years spans ~1–2 decades and clusters, so its first digit is dominated by "
            "wherever it dwelt. Naming that mechanism — and showing the apparent trade is a survivor "
            "artifact — is worth more than the binary verdict."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Conformity.** $\\chi^2$ and MAD of leading digits, on real prices and returns, vs "
            "two synthetic controls (a wide-range walk that *must* conform; a range-bound price that "
            "*must* fail) — the harness's positive + anti control.\n"
            "- **What deviation measures.** Cross-sectional correlation of a name's trailing-window "
            "deviation with its price-range and its trailing return.\n"
            "- **The sort.** Every `rebal` days, short the top-tercile deviant names, long the "
            "bottom tercile, equal-weight; **one execution lag** (form on $t$, earn $t{+}1\\!\\to$ "
            "rebalance); **costs one-way × NAV** (~2× NAV per full long/short rebuild); HAC *t* + "
            "circular block-bootstrap CI on the spread.\n"
            "- **The confounds.** Re-run with randomised base prices (kill the fabricated-level "
            "artifact), with χ² instead of MAD, net of 10 bps, and contrast with plain momentum — "
            "then name **survivorship** explicitly (the panel is current members)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The harness discriminates (positive + anti control)\n\n"
            "Before any market data: prove the measuring stick separates Benford from non-Benford."
        ),
        code(
            "sb, tb = data.synthetic_benford(); sn, tn = data.synthetic_nonbenford()\n"
            "rows = [('wide-range walk (control+)', tb['log_range_decades'], st.mad(sb), st.chi_square(sb)),\n"
            "        ('range-bound price (control−)', tn['log_range_decades'], st.mad(sn), st.chi_square(sn))]\n"
            "ctl = pd.DataFrame(rows, columns=['tape','decades','MAD','chi2'])\n"
            "print(ctl.to_string(index=False))\n"
            "assert st.mad(sb) < st.mad(sn) and st.chi_square(sb) < st.chi_square(sn)\n"
            "print('\\nHarness OK: the conforming tape scores far lower on both statistics.')"
        ),
        md(
            f"> 💡 In plain words: the control that spans {R['syn_ben_dec']:.1f} decades sits at MAD "
            f"~{R['syn_ben_mad']:.3f} (near conformity); the range-bound one at "
            f"~{R['syn_non_mad']:.3f} (gross nonconformity). The stick reads range."
        ),
        md(
            "### 4b · H₁ — single-name prices fail; returns conform\n\n"
            "Conformity of real split-adjusted prices vs absolute returns, across names of differing "
            "price-range width."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in ['SPY','QQQ','GLD','TLT','AAPL','MSFT']:\n"
            "        try:\n"
            "            s = data.load_real(t)\n"
            "            dec = float(np.log10(s.max()/s.min()))\n"
            "            recs.append((t, dec, st.mad(s), st.mad(s.pct_change().abs().dropna())))\n"
            "        except Exception:\n"
            "            pass\n"
            "    conf = pd.DataFrame(recs, columns=['ticker','decades','price_MAD','ret_MAD'])\n"
            "    banner = 'REAL TAPE'\n"
            "else:\n"
            "    conf = pd.DataFrame({'ticker':['SPY','QQQ','GLD','TLT','AAPL','MSFT'],\n"
            f"        'decades':[{R['spy_dec']},{R['qqq_dec']},{R['gld_dec']},{R['tlt_dec']},{R['aapl_dec']},{R['msft_dec']}],\n"
            f"        'price_MAD':[{R['spy_mad']},{R['qqq_mad']},{R['gld_mad']},{R['tlt_mad']},{R['aapl_mad']},{R['msft_mad']}],\n"
            f"        'ret_MAD':[{R['spy_ret_mad']}]*6}})\n"
            "    banner = 'FROZEN HEADLINE (no cache)'\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.4))\n"
            "x = np.arange(len(conf)); w=.38\n"
            "ax.bar(x-w/2, conf['price_MAD'], w, color=RED, label='price MAD')\n"
            "ax.bar(x+w/2, conf['ret_MAD'], w, color=GREEN, label='abs-return MAD')\n"
            f"ax.axhline({R['nigrini_nonconf']}, ls='--', c=GREY, label='Nigrini nonconformity (0.015)')\n"
            "ax.set_xticks(x); ax.set_xticklabels(conf['ticker']); ax.set_ylabel('MAD vs Benford')\n"
            "ax.set_title(f'{banner}: prices fail Benford, returns conform'); ax.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(conf.round(4).to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words: **every** single-name *price* MAD is above the 0.015 line "
            f"(SPY {R['spy_mad']:.3f}, TLT {R['tlt_mad']:.3f}); even AAPL/MSFT, which span ~3.8 "
            "decades, only reach ~0.03–0.07 because the path *dwells*. **Returns** sit near "
            f"{R['spy_ret_mad']:.3f} — acceptable conformity. **H₁ rejected**, and the folklore's "
            "target column (price) is the wrong one."
        ),
        md(
            "### 4c · H₃ — the deviation sort, and the confounds\n\n"
            "Long-conforming / short-deviant, monthly, on the current-membership panel — with the "
            "robustness battery that exposes what it really is."
        ),
        code(
            "labels = ['MAD sort\\n(main)', 'χ² sort', 'net 10bps', 'random\\nbase px', 'plain\\nmomentum']\n"
            f"anns = [{R['main_ann']*100:.2f},{R['chi_ann']*100:.2f},{R['net10_ann']*100:.2f},{R['randbase_ann']*100:.2f},{R['mom_ann']*100:.2f}]\n"
            f"ts   = [{R['main_t']:.2f},{R['chi_t']:.2f},{R['net10_t']:.2f},{R['randbase_t']:.2f},{R['mom_t']:.2f}]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))\n"
            "c = [AMBER if t>=2 else GREY for t in ts]\n"
            "a1.bar(labels, anns, color=c); a1.axhline(0, c='k', lw=1)\n"
            "a1.set_ylabel('annualised spread (%/yr)'); a1.set_title('Spread (survivor panel)')\n"
            "a1.tick_params(axis='x', labelsize=8)\n"
            "a2.bar(labels, ts, color=c); a2.axhline(2, ls='--', c=GREY); a2.axhline(0, c='k', lw=1)\n"
            "a2.set_ylabel('HAC t-stat'); a2.set_title('|t| — robust to costs/level, but on survivors')\n"
            "a2.tick_params(axis='x', labelsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('Deviation sort survives the level-base and cost checks (t≈{R['main_t']:.1f}); '\n"
            f"      'it is NOT plain momentum (mom t={R['mom_t']:+.2f}); the uncontrolled confound is SURVIVORSHIP.')"
        ),
        md(
            "> 💡 In plain words: the spread clears |t|=2 and shrugs off costs and the "
            "fabricated-level test — and it isn't momentum (momentum is negative here). That makes "
            "it *look* like a discovery. But the universe is **today's** index members, the textbook "
            "survivorship bias the desk has watched *manufacture* significant premia before. On this "
            "list, 'conforming' ⇔ 'climbed across decades' ⇔ 'survived'. **H₃ is supported only on an "
            "invalid universe**, so the Signal is `MIXED`, not `REAL`."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — sort HAC *t* = +{R['main_t']:.2f} (ann +{R['main_ann']*100:.1f}%), "
            f"robust to χ² (+{R['chi_t']:.2f}), 10 bps (+{R['net10_t']:.2f}) and randomised base "
            f"prices (+{R['randbase_t']:.2f}); but on a survivorship-biased panel and built on a "
            "false premise (H₁). Significant-looking, uncertifiable on a point-in-time universe.\n"
            "- **Tradability `MIRAGE`** — needs a 252-day window and a membership list you couldn't "
            "have known; the score is a price-range/survival proxy.\n"
            f"- **A forensic red-flag? `NOT SUPPORTED`** — price MAD ({R['spy_mad']:.3f}) is range, "
            f"not integrity; returns conform ({R['spy_ret_mad']:.3f}); deviation ⟂ momentum "
            f"({R['mom_t']:+.2f})."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the capacity & survivorship wall\n\n"
            "The cost axis is not where this dies — the survivor-panel spread survives 10 bps. It "
            "dies on **validity**: there is no point-in-time universe behind the number. The honest "
            "stress test is to rebuild the panel through the survivorship opt-in guard "
            "(`allow_survivorship_bias`) and watch the *t* fall toward the synthetic null (where, by "
            "construction, a deviation sort on an unrelated panel has |t| < 2). Until then, the "
            f"+{R['main_ann']*100:.1f}%/yr is a number you cannot bank."
        ),
        code(
            "# The synthetic null: a panel where price paths are UNRELATED to any future return.\n"
            "# A deviation sort there must NOT clear |t|=2 — confirming the engine isn't a\n"
            "# number-printer, and that the real-panel t is about the (biased) UNIVERSE, not the rule.\n"
            "rng = np.random.default_rng(328)\n"
            "dates = data._decorative_index(2000)\n"
            "cols = {f'N{j}': 20.0*np.exp(np.cumsum(rng.normal(3e-4, .02, len(dates)))) for j in range(30)}\n"
            "null = pd.DataFrame(cols, index=dates)\n"
            "res = st.forensic_backtest(null, window=252, rebal=42, quantile=0.3)\n"
            "print(f\"synthetic-null deviation sort: n={res['n_periods']} ann={res['ann_mean']:+.4f} \"\n"
            "      f\"HAC t={res['tstat']:+.2f}  (|t|<2 → no edge without a biased universe)\")"
        ),
        md(
            "> 💡 In plain words: on a panel with no planted link, the same rule prints |t| < 2 — the "
            "engine is honest. So the real-panel significance is a property of the **survivor "
            "universe**, not of any forensic content in the digits."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Aim Benford where it belongs.** Reported financials (the EDGAR cache the desk "
            "keeps) are its native habitat; a first-digit screen on revenue/earnings is the "
            "defensible use — prices were never the right input.\n"
            "- **Point-in-time universe.** Rebuild the panel with historical membership through the "
            "survivorship guard and re-run 4c; the prediction is *t* → the synthetic-null band.\n"
            "- **Higher-order digit tests.** Second-digit / summation tests (Nigrini) — same range "
            "problem on prices, likely same non-result.\n\n"
            "*The digits of a price encode its range and its survival, not its honesty. Fork this, "
            "give it an honest universe or an honest input (financials), and see what's left.*"
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
