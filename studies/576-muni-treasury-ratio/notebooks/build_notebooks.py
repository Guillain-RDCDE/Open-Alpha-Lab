"""Generate the two narrative notebooks for Study 576 (Muni-Treasury-Ratio).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic figures run
anywhere, offline and deterministic; the real-tape cells use the cached MUB/IEF series under
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-30; MUB/IEF,
# 2009-06-30 -> 2026-06-29; 4,275 days; tape fp 738df551786b, analysis fp c3e43ab615e6).
R = dict(
    fp_tape="738df551786b", fp_analysis="c3e43ab615e6",
    n_days=4275, n_obs=4087, start="2009-06-30", end="2026-06-29",
    ratio_mean=1.30, ratio_min=0.81, ratio_max=2.42,
    slope_pct=0.080, hac_t=0.43, r2=0.003,           # headline h=63 (slope in %/sigma)
    q5=0.51, q1=-0.09, spread=0.60, naive_t=4.12, placebo_p=0.0005,
    nonoverlap_spread=0.74, nonoverlap_t=0.51, nonoverlap_n=65,
    net=0.35, cost_bps=3.0, borrow_bps=50.0,
    # horizon sweep: (horizon, slope %/sigma, HAC-t, quintile spread%)
    sweep=[(21, 0.026, 0.39, 0.17), (63, 0.080, 0.43, 0.60),
           (126, 0.169, 0.59, 1.45), (252, 0.554, 1.28, 2.64)],
    # synthetic control: (timing_beta, mean HAC-t over 25 seeds)
    ctrl=[(0.0, 0.15), (0.03, 0.80), (0.06, 1.45), (0.10, 2.25), (0.15, 3.13)],
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

from muni_treasury_ratio import data, strategy as st

WARMUP = "2009-06-30"    # drop the first year (trailing-yield warm-up)

def load_real():
    \"\"\"Cache-first real analysis frame (empty offline).\"\"\"
    df = data.fetch_series(fetch=False)
    if df.empty:
        return pd.DataFrame()
    return df.loc[WARMUP:]

DF = load_real()
HAVE_REAL = not DF.empty
print("real MUB/IEF tape present:", HAVE_REAL,
      "" if not HAVE_REAL else f"({len(DF)} days, {DF.index.min().date()} -> {DF.index.max().date()})")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Muni-Treasury Ratio — does 'cheap munis' time the bond market? 🏛️\n"
            "### Using the muni-vs-Treasury yield ratio as a rich/cheap dial, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "Municipal bonds ('munis') pay interest that's exempt from federal income tax, so they can "
            "pay a *lower* yield than a Treasury and still leave a high-tax investor better off. The "
            "muni desk's oldest gauge is the **muni-Treasury ratio**: the muni yield divided by the "
            "Treasury yield. When that ratio is unusually **high**, munis look **cheap** (they're paying "
            "a big slice of the Treasury yield *despite* the tax break) and are supposed to outperform; "
            "when it's **low**, munis look **rich** and should lag.\n\n"
            "So we test it the cheap way: take a muni ETF (**MUB**) and a Treasury ETF (**IEF**), build "
            "their yield ratio, and see whether a high ratio really does predict munis beating "
            "Treasuries over the next few months.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the overlap trap and the "
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
            f"| Do cheap munis (high ratio) really outperform next? | **Yes — in direction.** The "
            f"cheapest-muni fifth of days earned **+{R['q5']}%** vs the richest fifth's **{R['q1']}%** "
            f"over the next 3 months — a **+{R['spread']}%** edge, the *right* way. |\n"
            f"| Is that a real, bankable signal? | **No.** Once you account for the fact that "
            "overlapping 3-month windows aren't independent, the *t*-stat falls from a shiny "
            f"**+{R['naive_t']}** to just **+{R['nonoverlap_t']}**. The edge is inside the noise. |\n"
            f"| Does it work at other horizons? | **Same story** — right sign at 1, 3, 6, 12 months, "
            "but the honest *t* never clears 2 (best is +1.28 at a year). |\n"
            "| Could you trade it? | **No.** A +0.35% net wobble that isn't statistically real, built "
            "on a coarse yield *proxy* rather than the curve a desk actually quotes. |\n\n"
            "> The ratio leans the right way — but 'leans the right way' and 'is a signal you can bet "
            "on' are different things. The gap between them is what this study measures."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"When the muni-Treasury yield ratio is high, munis are cheap and will outperform; when "
            "it's low, munis are rich and will lag.\"* — muni-desk folklore (the daily 10-year "
            "AAA-muni / Treasury ratio).\n\n"
            "The intuition is genuinely sensible. Munis are tax-exempt, so in a fair world they'd yield "
            "roughly *(1 − your tax rate)* times a Treasury. If the ratio climbs well above that, munis "
            "are handing you extra yield for the same-ish risk — a bargain — and the price should "
            "eventually catch up (munis outperform). If it sinks, munis are expensive. It's a "
            "mean-reversion story dressed as relative value."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If it worked, it'd be a clean dial: overweight munis when the ratio is high, underweight "
            "when low. But there's a catch baked into the claim — the ratio is **also a tax artifact**. "
            "It moves when people change their minds about future tax rates, not just when munis get "
            "cheap. So a 'high ratio' can mean 'bargain' *or* 'the market expects lower taxes' — and "
            "those imply opposite trades. The desk has looked at the Treasury *curve* as a timer "
            "([132 Yield-Curve-Steepener](../../132-yield-curve-steepener/)); here we go at the "
            "muni-vs-Treasury *ratio*."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Build the ratio.** Muni yield (from MUB's payouts) ÷ Treasury yield (from IEF's "
            "payouts), each day.\n"
            "2. **Score cheap/rich.** Compare today's ratio to its own last year — a *high* z-score "
            "means munis look cheap versus recent history.\n"
            "3. **Look forward.** Did munis beat Treasuries over the next 3 months when the ratio was "
            "high? Sort days into fifths by the ratio and compare.\n\n"
            "Catch we'll hit hard: 3-month forward windows *overlap* — today's window shares 62 of its "
            "63 days with yesterday's. That makes the data look like far more independent evidence than "
            "it is, and it's exactly how a nothing-signal can print a big *t*-stat.\n\n"
            "*Timing:* the ratio uses only data up to today's close; we act at that close and hold "
            "forward. No peeking."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Did the cheap-muni days really lead to muni outperformance?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = st.quintile_spread(DF, 252, 63)\n"
            "    q5, q1, spr = b['q5_mean']*100, b['q1_mean']*100, b['spread']*100\n"
            "    banner = 'REAL tape (MUB/IEF, 2009-06 -> 2026-06)'\n"
            "else:\n"
            f"    q5, q1, spr = {R['q5']}, {R['q1']}, {R['spread']}\n"
            "    banner = 'frozen real numbers (offline)'\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.4))\n"
            "ax.bar(['CHEAP munis\\n(high ratio, Q5)','RICH munis\\n(low ratio, Q1)'], [q5, q1],\n"
            "       color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('next 3-month muni - Treasury excess %')\n"
            "ax.set_title(f'Cheap munis did outperform next: +{q5:.2f}% vs {q1:.2f}% (spread +{spr:.2f}%)')\n"
            "fig.suptitle(banner, fontsize=9, color=GREY); plt.tight_layout(); plt.show()\n"
            "print(f'Q5 (cheap) +{q5:.2f}%  vs  Q1 (rich) {q1:+.2f}%  ->  spread {spr:+.2f}%')"
        ),
        md(
            f"There's the folklore, pointing the **right way**: the cheap-muni days (Q5) beat the "
            f"rich-muni days (Q1) by **+{R['spread']}%** over the next quarter. On its own that looks "
            f"like a win — and the naive *t*-stat is a juicy **+{R['naive_t']}**. So why isn't this a "
            "`REAL` signal?"
        ),
        md(
            "**Because the windows overlap.** Watch what happens when we only use *non-overlapping* "
            "3-month windows — genuinely independent bets, no shared days:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ss = st.spread_tstat(st.quintile_spread(DF, 252, 63))\n"
            "    a = st.aligned(DF, 252, 63); sub = a.iloc[::63]\n"
            "    z = sub['ratio_z'].values; y = sub['fwd_excess'].values\n"
            "    hi = z >= np.quantile(z, 0.8); lo = z <= np.quantile(z, 0.2)\n"
            "    va, vb = y[hi].var(ddof=1), y[lo].var(ddof=1)\n"
            "    tt = (y[hi].mean()-y[lo].mean())/np.sqrt(va/hi.sum()+vb/lo.sum())\n"
            "    naive_t, nonov_t, nn = ss['t'], tt, len(sub)\n"
            "else:\n"
            f"    naive_t, nonov_t, nn = {R['naive_t']}, {R['nonoverlap_t']}, {R['nonoverlap_n']}\n"
            f"SPR = {R['spread']}\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "ax.bar(['naive t\\n(overlapping)','honest t\\n(non-overlap)'], [naive_t, nonov_t],\n"
            "       color=[GREY, RED], width=.5)\n"
            "ax.axhline(2, ls='--', c=GREEN, lw=1.2, label='t = 2 bar')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('t-stat on the same spread')\n"
            "ax.set_title(f'Same +{SPR}% spread: t {naive_t:.1f} -> {nonov_t:.1f} once windows are independent')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'naive overlapping t {naive_t:+.2f}  ->  honest non-overlap t {nonov_t:+.2f} (n={nn})')"
        ),
        md(
            f"There's the trick. The **same** +{R['spread']}% spread carries a *t* of +{R['naive_t']} "
            f"when you let the windows overlap, but only **+{R['nonoverlap_t']}** on independent "
            f"windows (just {R['nonoverlap_n']} of them). The overlap manufactured almost all the "
            "significance. The signal is real in *direction* but not in *strength*."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** Right sign at every horizon, literature behind it — but no honest "
            "*t* ≥ 2 on the tape (the overlap-corrected *t* is +0.5).\n"
            "- **Tradability — Mirage.** A +0.35% net wobble that isn't statistically real, on a coarse "
            "yield proxy rather than the curve a desk trades."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Not on this evidence. The edge clears trading costs on paper (+0.60% gross → +0.35% net), "
            "but it isn't statistically distinguishable from zero, and we built the ratio from ETF "
            "*payout* yields — a coarse stand-in for the AAA-muni / Treasury *par* curve a real muni "
            "desk quotes. Worse, the ratio is partly a **tax bet** (it moves with expected tax rates), "
            "so a 'cheap' reading can mean two opposite things. A dial you can't read cleanly isn't a "
            "dial."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The Treasury-only cousin.** [132 Yield-Curve-Steepener](../../132-yield-curve-steepener/) "
            "times the long bond off the *curve slope* — same 'right sign, weak *t*' shape.\n"
            "- **The real test needs the real curve.** Re-run this off MMD/AAA-GO muni yields vs "
            "matched-maturity Treasuries (not ETF payout proxies), and separate the tax-expectation "
            "component from the relative-value one.\n\n"
            "*Think the ratio really times munis and we just used too coarse a proxy? Fork this, plug in "
            "the MMD curve, and show the muni-minus-Treasury spread clearing t = 2 on independent "
            "windows.*"
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
            "# The Muni-Treasury Ratio — a quantitative teardown 🔬\n"
            "### Distribution-yield ratio · trailing z · HAC predictive regression · quintile sort · the overlap illusion · placebo null · four-horizon sweep · costs & borrow · synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We build the MUB/IEF "
            "distribution-yield ratio, z-score it, and test whether it predicts the forward "
            "muni-minus-Treasury excess return — with a **Newey-West** *t* that respects the overlap.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance daily total-return prices + trailing "
            f"distribution yields for MUB & IEF, 2009-06 → 2026-06 (as-of 2026-06-30, analysis fp "
            f"`{R['fp_analysis']}`); the offline core and tests run on a deterministic synthetic world. "
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
            f"| **Signal** | `WEAK` | Predictive slope *right sign* every horizon, but HAC *t* "
            f"**+{R['hac_t']}** (h = 63), best **+1.28** (h = 252) — sub-2. Quintile spread "
            f"**+{R['spread']}%** naive *t* **+{R['naive_t']}** collapses to **+{R['nonoverlap_t']}** "
            "on non-overlapping windows. |\n"
            f"| **Tradability** | `MIRAGE` | Gross **+{R['spread']}%** → net **+{R['net']}%** "
            f"({R['cost_bps']:.0f} bps/leg + {R['borrow_bps']:.0f} bps borrow); clears costs but "
            "rests on an insignificant, overlap-inflated edge on a distribution-yield proxy. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), and the ratio "
            "leans the right way — but the real-tape edge doesn't clear the inference bar once overlap "
            "is handled honestly."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $z_t$ be the trailing z-score of the M/T ratio at $t$ and $x_{t\\to t+h}$ the forward "
            "muni-minus-Treasury excess return.\n\n"
            "- **H₁ (timing).** slope of $x$ on $z > 0$ at HAC *t* ≥ 2 (high ratio → munis "
            "outperform).\n"
            "- **H₂ (long-short).** $\\mathbb{E}[x \\mid \\text{Q5}] - \\mathbb{E}[x \\mid \\text{Q1}] "
            "> 0$, and *survives the overlap correction*.\n"
            "- **H₃ (robustness).** The sign and significance hold across horizons $h \\in \\{21,63,126,"
            "252\\}$.\n"
            "- **H₄ (net).** H₁/H₂ survive costs and the Treasury-leg borrow.\n\n"
            "We find **H₁ partially** (right sign, but *t* sub-2), **H₂ rejected as a real effect** "
            "(spread positive but the overlap-corrected *t* is +0.5), **H₃ split** (sign stable, "
            "significance never reached), **H₄ moot** (the spread nets positive but isn't real)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why**. Two things blunt the ratio here: (a) **overlap** — 63-day "
            "forward windows share 62/63 of their days, so a naive *t* over-counts evidence by an order "
            "of magnitude (Newey-West 1987 exists precisely for this); and (b) the **tax artifact** — "
            "the ratio's level and moves reflect expected marginal tax rates and muni-market "
            "segmentation as much as relative value (Chalmers 1998; Ang-Bhansali-Xing 2010), so it is "
            "not a clean valuation signal. That maps to [132 Yield-Curve-Steepener](../../132-yield-curve-steepener/)'s "
            "'right sign, weak *t*' reading on a related rate-timing rule."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $z_t = $ trailing 252-day z-score of $\\text{mt\\_ratio}_t = "
            "\\text{yield}^{\\text{MUB}}_t / \\text{yield}^{\\text{IEF}}_t$ (higher = munis cheaper).\n"
            "- **Target.** $x_{t\\to t+h}$ = compounded muni − Treasury return over $(t, t+h]$.\n"
            "- **Headline.** OLS of $x$ on $z$ with a **Newey-West HAC** SE (lag ≈ the overlap "
            "horizon) — the honest predictive *t*.\n"
            "- **Long-short.** Quintile sort on $z$; Q5 − Q1 forward-excess spread, two-sample *t*, "
            "plus a **non-overlapping** cross-check and a **label-shuffle placebo** (2000 perms).\n"
            "- **Robustness.** Re-run at $h \\in \\{21,63,126,252\\}$; read sign and *t*.\n"
            "- **Frictions.** Round-trip cost per leg + an annual borrow on the short Treasury leg.\n"
            "- **Positive control.** A deterministic synthetic world with a planted timing effect "
            "(`timing_beta > 0`) and a null — averaged over 25 seeds (house rule).\n\n"
            "Timing: $z_t$ uses only data through $t$; we enter at $t$'s close and hold over $(t, "
            "t+h]$. No future data enters the signal (no look-ahead). The same-close fill is a "
            "documented convention — on a multi-week hold a one-bar-later entry moves the headline "
            "HAC-*t* by a hair and changes no stamp."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline predictive regression — H₁\n\n"
            "Forward excess on the ratio z, Newey-West *t* on the slope. Right sign, sub-2 *t*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    reg = st.headline_test(DF, 252, 63)\n"
            "    a = st.aligned(DF, 252, 63)\n"
            "    slope, hac_t, r2 = reg['slope'], reg['t'], reg['r2']\n"
            "    fig, ax = plt.subplots(figsize=(8, 4.6))\n"
            "    ax.scatter(a['ratio_z'], a['fwd_excess']*100, s=8, color=GREY, alpha=.4)\n"
            "    xs = np.linspace(a['ratio_z'].min(), a['ratio_z'].max(), 50)\n"
            "    b0 = a['fwd_excess'].mean() - reg['slope']*a['ratio_z'].mean()\n"
            "    ax.plot(xs, (reg['slope']*xs + b0)*100, c=GREEN, lw=2)\n"
            "    ax.set_xlabel('M/T ratio z (higher = munis cheaper)'); ax.set_ylabel('forward muni - Tsy excess %')\n"
            "    ax.set_title(f'Slope +{slope*100:.3f}%/sigma (HAC t {hac_t:+.2f}) - right sign, sub-2')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            f"    slope, hac_t, r2 = {R['slope_pct']/100}, {R['hac_t']}, {R['r2']}\n"
            "print(f'slope {slope*100:+.3f}%/sigma  HAC-t {hac_t:+.2f}  R2 {r2:.4f}')"
        ),
        md(
            f"> 💡 In plain words: H₁ **half-passes** — the slope is the *right* sign "
            f"(+{R['slope_pct']}% excess per 1σ-cheaper ratio) but the HAC *t* is **+{R['hac_t']}**, far "
            f"from the *t* ≥ 2 bar, and R² ≈ {R['r2']}: the ratio explains almost none of the forward "
            "variance."
        ),
        md(
            "### 4b · The quintile spread and the overlap illusion — H₂\n\n"
            "The tradable Q5 − Q1 spread looks significant — until you remove the overlap."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = st.quintile_spread(DF, 252, 63); ss = st.spread_tstat(b)\n"
            "    p = st.placebo_pvalue(DF, 252, 63, n_perm=2000, seed=576)\n"
            "    a = st.aligned(DF, 252, 63); sub = a.iloc[::63]\n"
            "    z = sub['ratio_z'].values; y = sub['fwd_excess'].values\n"
            "    hi = z >= np.quantile(z, 0.8); lo = z <= np.quantile(z, 0.2)\n"
            "    va, vb = y[hi].var(ddof=1), y[lo].var(ddof=1)\n"
            "    nonov_t = (y[hi].mean()-y[lo].mean())/np.sqrt(va/hi.sum()+vb/lo.sum())\n"
            "    spr, naive_t, nn = b['spread']*100, ss['t'], len(sub)\n"
            "else:\n"
            f"    spr, naive_t, p, nonov_t, nn = {R['spread']}, {R['naive_t']}, {R['placebo_p']}, {R['nonoverlap_t']}, {R['nonoverlap_n']}\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "ax.bar(['naive t\\n(overlapping)','honest t\\n(non-overlap)'], [naive_t, nonov_t],\n"
            "       color=[GREY, RED], width=.5)\n"
            "ax.axhline(2, ls='--', c=GREEN, lw=1.2, label='t = 2 bar'); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('t-stat on the Q5 - Q1 spread')\n"
            "ax.set_title(f'Same +{spr:.2f}% spread: t {naive_t:.1f} -> {nonov_t:.1f} on independent windows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'spread {spr:+.2f}%  naive t {naive_t:+.2f}  placebo p {p:.4f}  ->  non-overlap t {nonov_t:+.2f} (n={nn})')"
        ),
        md(
            f"> 💡 In plain words: H₂ **rejected as a real effect.** The +{R['spread']}% spread is "
            f"economically real and the placebo *p* = {R['placebo_p']} looks decisive — but both are "
            f"driven by overlap: on {R['nonoverlap_n']} *independent* 63-day windows the Welch *t* is "
            f"only **+{R['nonoverlap_t']}**. The naive +{R['naive_t']} was the window-overlap talking."
        ),
        md(
            "### 4c · Robustness — H₃ (sign vs significance across horizons)\n\n"
            "The same regression at 1, 3, 6, 12 months."
        ),
        code(
            "sweep = " + repr(R["sweep"]) + "\n"
            "if HAVE_REAL:\n"
            "    rows = [(h, sl, t, spr) for (h, sl, t, spr) in st.horizon_sweep(DF, 252, (21,63,126,252))]\n"
            "    rows = [(h, sl*100, t, spr*100) for (h, sl, t, spr) in rows]\n"
            "else:\n"
            "    rows = sweep\n"
            "tab = pd.DataFrame(rows, columns=['horizon','slope%/sig','HAC_t','spread%']).set_index('horizon')\n"
            "fig, ax = plt.subplots(figsize=(8.3, 4.3))\n"
            "ax.bar([str(h) for h in tab.index], tab['HAC_t'], color=AMBER, width=.55)\n"
            "ax.axhline(2, ls='--', c=GREEN, lw=1.2, label='t = 2 bar'); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('forward horizon (trading days)'); ax.set_ylabel('HAC slope-t')\n"
            "ax.set_title('Right sign, rising with horizon - but never clears t = 2')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(tab.round(3).to_string())"
        ),
        md(
            "> 💡 In plain words: H₃ **splits** — the sign is *stable and correct* at every horizon "
            "(the folklore's direction is not in doubt) and the spread grows to +2.64% at a year, but "
            "the HAC *t* rises only to +1.28. Direction: yes. Significance: never."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — H₁ right sign but HAC *t* +{R['hac_t']} (best +1.28); H₂'s spread "
            f"is real in size but its significance is overlap (+{R['naive_t']} → +{R['nonoverlap_t']}); "
            "H₃ sign-stable, never significant.\n"
            f"- **Tradability `MIRAGE`** — gross +{R['spread']}% → net +{R['net']}%; clears costs but "
            "rests on an insignificant, overlap-inflated edge on a distribution-yield proxy."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "The spread clears costs numerically; that is not the problem — the significance is."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = st.quintile_spread(DF, 252, 63)\n"
            "    gross = b['spread']*100\n"
            "    net = st.net_spread(b, cost_bps=3.0, borrow_ann_bps=50.0, horizon=63)*100\n"
            "else:\n"
            f"    gross, net = {R['spread']}, {R['net']}\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "ax.bar(['gross','net\\n(costs+borrow)'], [gross, net], color=[GREEN, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('Q5 - Q1 forward excess %')\n"
            "ax.set_title(f'Clears costs (+{gross:.2f}% -> +{net:.2f}%) - but the edge is not significant')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:+.2f}% -> net {net:+.2f}% (3 bps/leg + 50 bps borrow, 0.25y hold)')"
        ),
        md(
            "> 💡 In plain words: a positive net number on a statistically-insignificant, "
            "overlap-inflated, tax-contaminated proxy signal is not a trade. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a real timing effect "
            "(`timing_beta > 0`) of increasing strength and watch the HAC slope-*t* climb — averaged "
            "over 25 seeds so no single lucky seed can fake it."
        ),
        code(
            "betas = [0.0, 0.03, 0.06, 0.10, 0.15]\n"
            "ts = [st.synthetic_mean_t(data, timing_beta=b, horizon=63, n_seeds=25) for b in betas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(betas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted timing_beta (larger = stronger timing effect)')\n"
            "ax.set_ylabel('mean HAC slope-t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted timer - flat at the null, past +2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for b, t in zip(betas, ts): print(f'timing_beta {b:+.2f} -> mean HAC-t {t:+.2f}')"
        ),
        md(
            "The HAC *t* is ≈ 0 at the null and climbs past +2 as the planted timer grows — so the "
            "engine is a faithful detector. The real-tape *t* = +0.43 is therefore a statement about "
            "**this tape and this proxy**: the muni-Treasury ratio leans the right way but doesn't "
            "clear the bar. For the Treasury-curve cousin, see "
            "[132 Yield-Curve-Steepener](../../132-yield-curve-steepener/)."
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
