"""Generate the two narrative notebooks for Study 565 (Filing-Readability).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). This is a **synthetic-only**
study — there is no free real 10-K-text tape (see data.py), so ``HAVE_REAL`` is ``False`` everywhere,
every code cell runs offline and deterministically on the synthetic world, and NO cell is drawn under
a real-tape banner. The dict ``R`` mirrors the exact numbers in docs/results.md.
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


# Frozen synthetic headline numbers — mirror of docs/results.md (as-of 2026-06-30;
# synthetic_panel(n_stocks=400, obf_alpha=-0.10, seed=565); panel fp 53136605f951).
# NB: these are a MACHINERY PROOF on planted synthetic data, never market evidence.
R = dict(
    fp_panel="53136605f951", n=400, k=120, obf_alpha=-0.10,
    ic=-0.343, ic_t=-7.11, slope=-2.18, slope_t=-8.11, corr=-0.38,
    readable_mean=13.6, murky_mean=1.5, spread=12.1, ls_t=6.44, placebo_p=0.0005,
    risk_slope=0.030, risk_t=36.25,
    gross=12.1, net=10.4, cost_bps=5.0, borrow_bps=150.0,
    # per-leg IC sweep: (leg, ic, ic_t)
    legs=[("fog", -0.285, -5.85), ("length_kw", -0.300, -6.16),
          ("filesize_mb", -0.285, -5.83), ("composite", -0.343, -7.11)],
    # synthetic control: (alpha, mean IC over 25 seeds, mean IC-t)
    ctrl=[(0.0, -0.008, -0.17), (-0.04, -0.125, -2.52), (-0.10, -0.284, -5.82),
          (-0.16, -0.410, -8.69), (-0.24, -0.535, -11.91)],
    null_ls_t=1.05, null_placebo_p=0.30,
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

from filing_readability import data, strategy as st

# Synthetic-only study: no free real 10-K-text tape exists (see data.py). The real-tape stub is
# empty by construction, so HAVE_REAL is False and every cell below runs on the deterministic
# synthetic world — NEVER under a real-tape banner.
REAL = data.fetch_panel(fetch=False)
HAVE_REAL = not REAL.empty

# The calibrated headline synthetic world (the LM anomaly planted at obf_alpha = -0.10).
PANEL, TRUTH = data.synthetic_panel(n_stocks=400, obf_alpha=-0.10, seed=565)
print("real 10-K-text tape present:", HAVE_REAL, "(none exists on a free stack — synthetic-only)")
print("synthetic panel:", len(PANEL), "firms, fp", data.fingerprint(PANEL))
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Readability Anomaly — do murky 10-Ks hide bad news? 📄\n"
            "### Scoring annual reports by how *hard they are to read*, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n\n"
            "Every year, a public company files a 10-K — a long legal document describing its business "
            "and risks. Some are crisp and clear. Others are **enormous, dense, and murky**. A famous "
            "2014 study (Loughran & McDonald) found something suspicious: the companies with the "
            "*hardest-to-read, longest, biggest* filings went on to earn **worse** stock returns and "
            "were **riskier**. The story is **obfuscation** — if you have bad news to bury, you write "
            "a longer, murkier report and hope nobody reads to the end. And the market, it seems, "
            "underreacts.\n\n"
            "So we test it: score each firm's filing for 'murkiness' (a high fog index, lots of words, "
            "a big file), and see whether the *readable* filers beat the *murky* ones.\n\n"
            "> ⚠️ **A big honest caveat up front.** To test this *for real* you'd need every company's "
            "10-K **text** lined up with the *right* firm's *later* returns — and that clean, "
            "point-in-time dataset simply **isn't free**. So this notebook runs on a **synthetic** "
            "world calibrated to the research. It proves our measuring machine works; it can't be "
            "'market proof'. That's exactly why the Signal stamp is **Weak**, not Real.\n"
            ">\n"
            "> 📓 Want the *t*-stats, the placebo test and the cost maths? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**. Not investment advice."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do murkier filings earn worse returns (the anomaly)? | **Yes — in the research, and in "
            f"our calibrated test.** The readable third earned **+{R['readable_mean']}%** vs the murky "
            f"third's **+{R['murky_mean']}%** — a **+{R['spread']}%** spread the right way. |\n"
            f"| Is it just luck? | **No** (in the synthetic world) — a shuffle test puts the odds at "
            f"*p* = {R['placebo_p']}. And the machine stays *flat* when we turn the effect off. |\n"
            "| So is it proven on real stocks? | **We can't say.** There's no free dataset of 10-K "
            "*text* lined up with returns. We can only show the *machine* works. |\n"
            "| Could you trade it? | **Fragile.** In the model the trade pays after costs — but the "
            "'murky' names you'd short are the small, illiquid, expensive-to-borrow ones, and we have "
            "no real tape to size it. |\n\n"
            "> The anomaly is real in the journals (it's a *Journal of Finance* result). Our engine "
            "banks it cleanly on calibrated data. What's missing is the one thing money cares about: "
            "a **real tape** to certify it on. Hence **Weak**, not Real."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"Firms whose 10-K filings are harder to read — higher fog, longer, bigger file — earn "
            "lower future returns and carry higher risk.\"* — the readability anomaly (Loughran & "
            "McDonald 2014; Li 2008; You & Zhang 2009).\n\n"
            "Why would that be? **Managers with bad news to bury write murkier reports.** A company "
            "whose earnings are about to disappoint has an incentive to make its filing longer and "
            "denser, so the bad bits are harder to find. Investors, faced with a 200-page fog, "
            "**underreact** — they don't fully price the buried signal — so the murky firms quietly "
            "drift *down* over the following months. Clear filing, nothing to hide; murky filing, "
            "**red flag**."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If true, it means the *shape* of a document — not even what it says, just how hard it is "
            "to read — carries a tradable signal. That's a striking idea: obfuscation as alpha. And it "
            "suggests a simple rule: **own the clear filers, avoid (or short) the murky ones.** The "
            "desk has looked at filing *sentiment / tone* elsewhere ([257 AAII](../../257-aaii-sentiment/), "
            "[335 Buzz](../../335-buzz-sentiment-etf/), [392 Glassdoor](../../392-glassdoor-sentiment/)); "
            "here we go at the *readability* itself — the structure, not the mood."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Score murkiness.** For each firm: a high **fog index** (long sentences, complex "
            "words), a **long** document, a **big file**. Add them up — higher = harder to read.\n"
            "2. **Sort.** Split the firms into a *readable* third and a *murky* third.\n"
            "3. **Compare forward returns.** Did the readable third beat the murky third *after* the "
            "filing came out? The anomaly says yes.\n\n"
            "Catch — the one that caps this whole study: to do this on *real* companies you need every "
            "10-K's **text**, parsed, lined up with the *right* firm's *later* return, without "
            "survivorship bias. **That dataset isn't free.** So we build a **synthetic** world "
            "calibrated to the research, where we *know* the true effect, and check the machine reads "
            "it correctly.\n\n"
            "*Timing:* the readability is known the instant the 10-K is filed; the return is measured "
            "strictly *afterwards*. No peeking at the future."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Who won — the readable third or the murky third?** (Calibrated synthetic world.)"
        ),
        code(
            "b = st.bucket_returns(PANEL, frac=0.3); ls = st.long_short_tstat(b)\n"
            "read_m, murk_m, spr, t = b['readable_mean']*100, b['murky_mean']*100, b['spread']*100, ls['t']\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['READABLE third','MURKY third'], [read_m, murk_m], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'Readable filers won (+{read_m:.0f}%) — murky filers lagged (+{murk_m:.0f}%)')\n"
            "fig.suptitle('synthetic world calibrated to the LM anomaly (NOT market evidence)', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Readable +{read_m:.1f}%  vs  Murky +{murk_m:.1f}%  ->  spread {spr:+.1f}% (t {t:+.2f})')"
        ),
        md(
            f"There it is, the **right way**: the readable third earned **+{R['readable_mean']}%**, the "
            f"murky third only **+{R['murky_mean']}%** — a **+{R['spread']}%** spread (*t* +{R['ls_t']}). "
            "Remember: this is the *machine* reading an effect we *planted*. The real question is "
            "whether the machine is honest — so let's turn the effect **off** and check it doesn't "
            "hallucinate a signal."
        ),
        md(
            "**Does the machine cry wolf?** Same engine, but on a world where readability has *no* real "
            "link to returns (the null). A good detector should print ~nothing."
        ),
        code(
            "panel0, _ = data.synthetic_panel(n_stocks=400, obf_alpha=0.0, seed=565)\n"
            "b0 = st.bucket_returns(panel0, 0.3); ls0 = st.long_short_tstat(b0)\n"
            "spr0 = b0['spread']*100; t0 = ls0['t']\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "ax.bar(['effect ON\\n(planted)','effect OFF\\n(null)'], [spr, spr0], color=[GREEN, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('readable - murky spread %')\n"
            "ax.set_title(f'ON: +{spr:.0f}% (t {t:+.1f})   vs   OFF: {spr0:+.0f}% (t {t0:+.1f}) — flat')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'planted spread {spr:+.1f}% (t {t:+.2f})  |  null spread {spr0:+.1f}% (t {t0:+.2f})')"
        ),
        md(
            "Turn the effect off and the spread collapses to noise (*t* ≈ 1, nowhere near 2). So the "
            "machine only 'finds' the anomaly when it's actually there — it's a faithful detector, not "
            "a signal-manufacturing machine. That's the whole point of the synthetic control."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The anomaly is a real, replicated journal result and our engine banks "
            f"it cleanly on calibrated data (spread +{R['spread']}%, *t* +{R['ls_t']}) while staying "
            "flat at the null. But **`Real` needs a real tape at *t* ≥ 2**, and no free 10-K-text tape "
            "exists — so literature-real + no tape = **Weak**.\n"
            "- **Tradability — Fragile.** In the model the trade pays after costs, but the murky short "
            "leg is the small/illiquid, hard-to-borrow tail, with no real capacity estimate.\n\n"
            "> This is the desk's honest posture for a claim that's real in the books but has no free "
            "tape to test — the same as [whisky-cask](../../275-whisky-cask/) and "
            "[lego-returns](../../273-lego-returns/)."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Fragile. Even granting the effect, the trade means *shorting* the murkiest filers — which "
            "skew small, illiquid, and expensive to borrow — and rebalancing once a year around filing "
            "dates. In the plausibility model it survives a punitive borrow (gross "
            f"+{R['gross']}% → net +{R['net']}%), but without a real tape there's no honest capacity, "
            "turnover, or borrow-availability number. You can't size a trade you can't measure."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The modern descendant.** *Lazy Prices* (Cohen, Malloy & Nguyen 2020): it's not just "
            "how murky a filing is, but how much it *changed* year-over-year — big text changes predict "
            "lower returns too.\n"
            "- **The real test needs the text.** Drop a real EDGAR-text + point-in-time-return panel "
            "into `data.fetch_panel()` (the stub is ready for it) and re-run — that's the only way to "
            "move this off `Weak`.\n\n"
            "*Think you can build the real panel? Fork this, wire EDGAR full-text + a survivorship-free "
            "return tape into the stub, and show the readable-minus-murky spread clearing t = 2 on the "
            "actual tape.*"
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
            "# The Readability Anomaly — a quantitative teardown 🔬\n"
            "### LM obfuscation score · Spearman IC + Fisher-z *t* · tercile long-short · label-shuffle placebo · risk leg · per-leg IC sweep · costs & borrow · synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We build a Loughran-McDonald-style "
            "**obfuscation score** (fog + length + file size), sort the cross-section, and test whether "
            "the least-readable filings earn the lowest returns (the anomaly).\n\n"
            "> ⚠️ **Not investment advice — and read this first.** There is **no free real tape** for this "
            "claim: a point-in-time 10-K-text + survivorship-free return panel isn't a no-key retail "
            "artifact. So every number below is from a **deterministic synthetic world** calibrated to "
            f"the LM literature (panel fp `{R['fp_panel']}`) — a **machinery proof, never market "
            "evidence**. `Real` needs *t* ≥ 2 on a **real** tape (METHODOLOGY → the inference bar); this "
            "study is therefore capped at `Weak`. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Synthetic IC **{R['ic']}** (Fisher-z *t* **{R['ic_t']}**); "
            f"long-short **+{R['spread']}%** (two-sample *t* **+{R['ls_t']}**, placebo *p* {R['placebo_p']}); "
            f"risk leg *t* **+{R['risk_t']:.0f}**. All the right sign — but **synthetic**; no real tape "
            "can certify it. |\n"
            f"| **Tradability** | `FRAGILE` | Gross +{R['gross']}% → net +{R['net']}% (5 bps/leg + "
            f"{R['borrow_bps']:.0f} bps borrow); murky short = small/illiquid tail; no real capacity "
            "estimate. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it, flat at the null) "
            "and recovers the LM anomaly with the right sign and huge *t* on planted data — but the "
            "desk certifies `Real` only from a **real** tape, which doesn't exist for free here."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $O_i$ be firm $i$'s obfuscation score (higher = less readable) and $r_i$ its "
            "*post-filing* forward return.\n\n"
            "- **H₁ (the anomaly / IC).** $\\text{IC} = \\text{Spearman}(O, r) < 0$ at *t* ≥ 2 "
            "(murkier filings, lower returns).\n"
            "- **H₂ (long-short).** $\\mathbb{E}[r \\mid \\text{readable}] - \\mathbb{E}[r \\mid "
            "\\text{murky}] > 0$ at *t* ≥ 2.\n"
            "- **H₃ (risk leg).** $\\text{slope of post-filing vol on } O > 0$ (murkier, riskier).\n"
            "- **H₄ (robustness).** The sign holds across the individual legs (fog, length, file size).\n"
            "- **H₅ (real tape).** H₁–H₄ replicate on a *real* EDGAR-text + return panel. **Untestable "
            "here** — no free tape — which is exactly why the ceiling is `Weak`.\n\n"
            "On the calibrated synthetic world we find **H₁–H₄ all supported with the right sign**; "
            "**H₅ we cannot reach**."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why**. The LM anomaly is well documented on real, survivorship-free "
            "EDGAR data driven by the *obfuscation* mechanism (Li 2008: bad-earnings firms file "
            "less-readable reports; You & Zhang 2009: investors underreact to complex 10-Ks). Two things "
            "keep it off a free retail stack: (a) the 10-K **text** panel isn't a free artifact "
            "(yfinance has none); and (b) a survivor return tape would strip the murkiest filers that "
            "later fail — biasing *against* the anomaly. So we prove the *machine* on synthetic data and "
            "name the tape gap. Sibling posture to [275 whisky-cask](../../275-whisky-cask/) and "
            "[273 lego-returns](../../273-lego-returns/)."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Score.** $O = z(\\text{fog}) + z(\\text{length}) + z(\\text{filesize})$, each z-scored "
            "across the cross-section (higher = less readable). The three observable LM legs.\n"
            "- **IC.** Spearman rank correlation of $O$ vs forward return, with a Fisher-z *t* "
            "($\\text{atanh}(\\text{IC})\\cdot\\sqrt{n-3}$). The *sign* is the anomaly.\n"
            "- **Long-short.** Tercile tails (120 names each): readable (lowest $O$) vs murky (highest); "
            "two-sample (Welch) *t*; a **label-shuffle placebo** null (2000 perms).\n"
            "- **Risk leg.** OLS of post-filing vol on $O$ — slope > 0 is the LM 'higher risk' leg.\n"
            "- **Per-leg sweep.** IC of fog / length / file size alone (LM: file size is cleanest).\n"
            "- **Frictions.** Round-trip cost per leg + an annual borrow on the murky short.\n"
            "- **Positive control.** A deterministic synthetic world with a planted anomaly "
            "(`obf_alpha < 0`) and a null — averaged over 25 seeds (house rule).\n\n"
            "Timing: readability is public the instant the 10-K is filed; the forward return is measured "
            "strictly *after* (a documented ≥1-day execution lag). The synthetic panel bakes this in "
            "(`forward_ret` is post-filing by construction) — no look-ahead."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The information coefficient — H₁\n\n"
            "The Spearman rank correlation of obfuscation vs forward return, with its Fisher-z *t*. A "
            "*negative* IC is the anomaly."
        ),
        code(
            "reg = st.cross_section_regression(PANEL)\n"
            "icd = st.information_coefficient(PANEL)\n"
            "o = st.readability_score(PANEL); y = PANEL['forward_ret']*100\n"
            "ic, ic_t = icd['ic'], icd['ic_t']\n"
            "fig, ax = plt.subplots(figsize=(8, 4.6))\n"
            "ax.scatter(o, y, s=14, color=GREY, alpha=.5)\n"
            "xs = np.linspace(o.min(), o.max(), 50)\n"
            "b0 = y.mean()/100 - reg['slope']*o.mean()\n"
            "ax.plot(xs, (reg['slope']*xs + b0)*100, c=RED, lw=2)\n"
            "ax.set_xlabel('obfuscation score (higher = less readable)'); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'IC {ic:+.3f} (Fisher-z t {ic_t:+.2f}) — NEGATIVE = the anomaly')\n"
            "fig.suptitle('synthetic world (machinery proof, not market evidence)', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'IC {ic:+.3f}, IC-t {ic_t:+.2f}, n {icd[\"n\"]} | firm slope {reg[\"slope\"]*100:+.2f}%/unit (t {reg[\"slope_t\"]:+.2f}), corr {reg[\"corr\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: H₁ **supported (on planted data).** The IC is **{R['ic']}** (*t* "
            f"{R['ic_t']}) — murkier filings earn lower returns, exactly the anomaly's sign. The "
            f"firm-level slope agrees (**{R['slope']}%**/unit, *t* {R['slope_t']}). This is the machine "
            "reading a planted effect; the null check (beat 7) shows it doesn't do this on noise."
        ),
        md(
            "### 4b · The long-short and its placebo — H₂\n\n"
            "Tercile tails, readable − murky, with the two-sample *t* and the label-shuffle null."
        ),
        code(
            "b = st.bucket_returns(PANEL, 0.3); ls = st.long_short_tstat(b)\n"
            "p = st.placebo_pvalue(PANEL, n_perm=2000, seed=565)\n"
            "read_m, murk_m, spr, t = b['readable_mean']*100, b['murky_mean']*100, b['spread']*100, ls['t']\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['READABLE','MURKY'], [read_m, murk_m], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('forward return %')\n"
            "ax.set_title(f'Readable - murky = {spr:+.1f}%  (two-sample t {t:+.2f}, placebo p {p:.4f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Readable +{read_m:.1f}% | Murky +{murk_m:.1f}% | spread {spr:+.1f}% | t {t:+.2f} | placebo p {p:.4f}')"
        ),
        md(
            f"> 💡 In plain words: H₂ **supported.** The spread is **+{R['spread']}%** (*t* +{R['ls_t']}); "
            f"the placebo *p* = {R['placebo_p']} puts it deep in the tail of the shuffled null. On the "
            "synthetic world this is a genuine cross-sectional signal, not a sorting artifact."
        ),
        md(
            "### 4c · The risk leg + the per-leg IC sweep — H₃ & H₄\n\n"
            "Post-filing volatility on obfuscation (H₃: slope > 0), and the IC of each individual leg "
            "vs the composite (H₄: the sign is stable across legs; LM say file size is the cleanest)."
        ),
        code(
            "rr = st.risk_regression(PANEL)\n"
            "sweep = st.leg_ic_sweep(PANEL)\n"
            "legs = list(sweep.keys()); ics = [sweep[l][0] for l in legs]; tts = [sweep[l][1] for l in legs]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "# risk leg\n"
            "o = st.readability_score(PANEL); pv = PANEL['post_vol']\n"
            "a1.scatter(o, pv, s=12, color=GREY, alpha=.5)\n"
            "xs = np.linspace(o.min(), o.max(), 50)\n"
            "a1.plot(xs, rr['slope']*xs + (pv.mean() - rr['slope']*o.mean()), c=RED, lw=2)\n"
            "a1.set_xlabel('obfuscation'); a1.set_ylabel('post-filing vol')\n"
            "a1.set_title(f'Risk leg: slope {rr[\"slope\"]:+.3f} (t {rr[\"slope_t\"]:+.1f}) > 0')\n"
            "# per-leg IC sweep\n"
            "cols = [RED if l!='composite' else GREEN for l in legs]\n"
            "a2.barh(range(len(legs)), ics, color=cols)\n"
            "a2.set_yticks(range(len(legs))); a2.set_yticklabels(legs)\n"
            "a2.axvline(0, c='k', lw=1); a2.set_xlabel('IC (Spearman)')\n"
            "a2.set_title('Every leg negative; composite strongest')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('risk slope', round(rr['slope'],3), 't', round(rr['slope_t'],1))\n"
            "for l in legs: print(f'  {l:12s} IC {sweep[l][0]:+.3f} (t {sweep[l][1]:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: H₃ **supported** — post-filing vol rises with obfuscation (slope "
            f"+{R['risk_slope']}, *t* +{R['risk_t']:.0f}), the LM 'higher risk' leg. H₄ **supported** — "
            "fog, length and file size each carry the same negative IC (≈ −0.29 to −0.30), and the "
            f"composite is strongest (**{R['ic']}**). No single proxy is perfect (LM's exact point); the "
            "blend wins."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — H₁–H₄ all supported with the right sign on calibrated data (IC "
            f"{R['ic']}, *t* {R['ic_t']}; long-short +{R['spread']}%, *t* +{R['ls_t']}; risk leg *t* "
            f"+{R['risk_t']:.0f}), but **H₅ (real tape) is unreachable** — no free 10-K-text panel. "
            "`Real` needs *t* ≥ 2 on a **real** tape; literature-real + no tape = `Weak`.\n"
            f"- **Tradability `FRAGILE`** — gross +{R['gross']}% → net +{R['net']}% (5 bps/leg + "
            f"{R['borrow_bps']:.0f} bps borrow), but the murky short is the small/illiquid tail and "
            "there's no real capacity estimate."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "In the model the book pays after a punitive borrow — but the murky short leg is exactly the "
            "hard-to-borrow tail, and no real tape means no honest capacity number."
        ),
        code(
            "b = st.bucket_returns(PANEL, 0.3)\n"
            "gross = b['spread']*100\n"
            "net = st.net_spread(b, cost_bps=5.0, borrow_ann_bps=150.0, holding_years=1.0)*100\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross','net\\n(costs+borrow)'], [gross, net], color=[GREEN, AMBER], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('readable - murky spread %')\n"
            "ax.set_title(f'Survives frictions in the model: +{gross:.0f}% -> +{net:.0f}%')\n"
            "fig.suptitle('synthetic; no real capacity/turnover estimate exists', fontsize=9, color=GREY)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:+.1f}% -> net {net:+.1f}% (5 bps/leg + 150 bps borrow, 1y hold)')"
        ),
        md(
            "> 💡 In plain words: it clears costs *in the model*, so tradability isn't `Mirage` — but "
            "with a small/illiquid short leg and no real tape to size capacity or borrow availability, "
            "it isn't `Investable` either. `Fragile` is the honest middle."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print a negative IC? Plant an anomaly "
            "of increasing strength (`obf_alpha` more negative) and watch the mean IC-*t* go negative — "
            "and confirm it sits at ≈ 0 at the **null**. Averaged over 25 seeds so no single lucky seed "
            "can fake it (house rule)."
        ),
        code(
            "alphas = [0.0, -0.04, -0.08, -0.10, -0.16, -0.24]\n"
            "res = [st.synthetic_mean_ic(data, obf_alpha=a, n_seeds=25) for a in alphas]\n"
            "tts = [r['ic_t'] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(alphas, tts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1, label='t = -2 bar (anomaly)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted obf_alpha (more negative = stronger anomaly)')\n"
            "ax.set_ylabel('mean IC-t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted anomaly — flat at the null, past -2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for a, r in zip(alphas, res): print(f'obf_alpha {a:+.2f} -> mean IC {r[\"ic\"]:+.3f}, mean IC-t {r[\"ic_t\"]:+.2f}')"
        ),
        md(
            "The IC-*t* is ≈ 0 at the null (**−0.17**) and falls past −2 from `obf_alpha = -0.04` "
            "onward — so the engine is a faithful detector that neither misses a real anomaly nor "
            "manufactures one on noise. The synthetic result is therefore a clean **machinery proof**: "
            "the harness *would* bank the LM readability anomaly if it could ever see a real 10-K-text "
            "tape. Because no such free tape exists, the honest stamp stays **`WEAK`**. For the desk's "
            "*sentiment* studies (a different textual signal), see "
            "[257 AAII](../../257-aaii-sentiment/) and [392 Glassdoor](../../392-glassdoor-sentiment/)."
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
