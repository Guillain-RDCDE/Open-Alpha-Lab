"""Generate the two narrative notebooks for Study 861 (Debt-Maturity Rollover Risk).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached prices + EDGAR
events under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (EDGAR maturity legs + yfinance
# prices, 32 names, period ends 2008-07 -> 2026-05, as-of 2026-06-30).
R = dict(
    as_of="2026-06-30", n_names=32, n_events=1732,
    end_lo="2008-07-31", end_hi="2026-05-03", fp_prices="6067630c220d",
    # primary claim long-short (long low-share / short high-share, terciles, staleness 400)
    n_months=203, avg_n=26.2, ls_span_lo="2009-08-31", ls_span_hi="2026-06-30",
    ls_mean_bps=46.7, ls_ann=5.61, ls_t_iid=2.65, ls_t_nw=3.22, ls_sharpe=0.64,
    ls_hit=59, ls_long_bps=143.1, ls_short_bps=96.4, ls_turn=0.103, ls_cum=2.42,
    ls200_mean_bps=41.1, ls200_t_nw=2.69,
    scaled_mean_bps=34.1, scaled_t_nw=2.67, scaled_sharpe=0.48,
    t_halves=3.87, t_quartiles=2.69, t_quintiles=2.27,
    jk_min=2.14, jk_min_name="OKE", jk_max=3.72, jk_max_name="GILD",
    # pooled event drift  horizon -> (n, high/top%, low/bot%, ls low-high%, t, win%, placebo p)
    drift={
        21: (1730, 0.86, 1.34, 0.48, 1.29, 52, 0.105),
        63: (1710, 2.96, 4.26, 1.30, 2.02, 53, 0.018),
        126: (1686, 6.02, 8.85, 2.83, 2.85, 52, 0.000),
    },
    mono63=(4.26, 2.42, 2.96), mono126=(8.85, 5.62, 6.02),   # low -> high share
    # era split at 2022
    era_early_n=149, era_early_bps=32.6, era_early_t=2.10,
    era_late_n=54, era_late_bps=85.7, era_late_t=2.95,
    # tradability net   (cost, borrow) -> (net bps, net ann, nw t, sharpe)
    net={(10, 50): (40.5, 4.86, 2.79, 0.56), (20, 100): (34.3, 4.11, 2.36, 0.47),
         (30, 200): (23.9, 2.86, 1.64, 0.33), (50, 300): (11.4, 1.37, 0.78, 0.16)},
    # synthetic control
    syn_null_mean=-0.41, syn_null_sd=1.33, syn_null_fire=2, syn_null_seeds=12,
    syn_planted_bps=226.5, syn_planted_t=7.13,
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Rate-era bite: Confirmed](https://img.shields.io/badge/Rate--era_bite%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from debt_maturity import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PX, EV = data.load_real()
else:
    PX = EV = None
print("real cache present:", HAVE_REAL, "| events:", (0 if EV is None else len(EV)),
      "| names:", (0 if EV is None else EV['ticker'].nunique()))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The debt that has to be refinanced *soon*. Do those firms lose? ⏳\n"
            "### A balance-sheet maturity signal that — unusually for this desk — actually predicts "
            "returns, and bites hardest exactly when the story says it should\n\n"
            + BADGES +
            "Most of what we test here turns out to be folklore. This one doesn't. Every company "
            "that borrows has a **maturity wall**: some of its debt is due within a year and has "
            "to be *rolled over* — refinanced at whatever interest rate and credit mood prevail "
            "when it comes due. A firm living on the short end of that curve is exposed: if rates "
            "jump or lenders get nervous between now and the maturity date, it refinances at a "
            "worse price (or, in a real crunch, can't refinance at all). The claim is that the "
            "market **under-prices** this, so the firms with the biggest short-term slice quietly "
            "**under-earn** — and that the penalty is worst when rates are *rising*.\n\n"
            "We took that literally. It holds.\n\n"
            "> 📓 **Plain-language layer.** Want the Newey-West *t*-stats, the jackknife, the "
            "placebo and the cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 32 large, debt-carrying US filers that report a clean maturity "
            "split on EDGAR, 2008→2026; a genuinely **thin, uneven panel**. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do high-short-term-debt firms **under-earn**? | **Yes — and it clears the bar.** "
            f"Buy the safe (termed-out) names, short the ones with the biggest maturity wall, and "
            f"the spread is **+{R['ls_ann']:.1f}%/yr** with a robust *t* of **+{R['ls_t_nw']:.2f}** "
            f"(the bar is 2). Right-signed, exactly as claimed. |\n"
            f"| Does it bite harder when **rates rise**? | **Yes.** The penalty is ~2.6× bigger in "
            f"the 2022+ hiking cycle (**+{R['era_late_bps']:.0f} bps/mo**) than before "
            f"(**+{R['era_early_bps']:.0f} bps/mo**) — the exact mechanism the story names. |\n"
            "| Is it just one lucky stock or one lucky decade? | **No.** Drop any single name and "
            "it still clears the bar; both halves of the sample are significant; a pooled "
            "event-drift check agrees. |\n"
            f"| Can you actually **trade** it? | **Barely.** Net of realistic costs + borrow it "
            f"survives (+{R['net'][(20,100)][1]:.1f}%/yr, *t* = +{R['net'][(20,100)][2]:.2f}), but "
            f"the Sharpe is a modest {R['net'][(20,100)][3]:.2f} and it **breaks** if costs run "
            "higher. A real edge, one cost-bump from vanishing. |\n\n"
            "> A rare green on this desk — but read the fine print: much of it is the *safe* leg "
            "winning, which may be part quality/defensive tilt, not pure rollover magic."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A firm with a big chunk of debt maturing within a year has to refinance it soon. "
            "If rates rise or credit tightens, that hurts — so the market should, but doesn't "
            "fully, discount those firms. Bet against the ones with the biggest short-term "
            "maturity wall.\"*\n\n"
            "It's a real corner of finance: **rollover risk** (He & Xiong 2012) is a genuine "
            "driver of credit risk, and firms with debt maturing into the 2008 crisis really did "
            "cut investment (Almeida et al. 2012). The twist is the *return* prediction — that "
            "high-rollover-risk firms **under**-earn — which lines up with the (still-debated) "
            "**distress anomaly**: risky firms puzzlingly earning *less*, not more."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "The signal is one ratio off the balance sheet — the share of a firm's debt due within "
            "a year — and it needs no estimates feed, no alt-data. If it *reliably* sorted returns "
            "it would be a clean, cheap edge. On this desk that is usually the cue for it to "
            "evaporate under a real test. This time it mostly doesn't, which is worth understanding."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The signal.** For each of {R['n_names']} big debt-carrying names, the "
            "**short-term-debt share** = (debt due this year) ÷ (total debt), known only on the "
            "**filing date** of the 10-Q/10-K (never before — no peeking).\n"
            "- **The return test.** Each month, rank the names, buy the *lowest*-share third "
            "(safe), short the *highest*-share third (rollover risk), hold for the next month. If "
            "the risky names under-earn, that spread makes money.\n"
            "- **The rates test.** Split at 2022 (the Fed hiking cycle). The story says the penalty "
            "should be bigger there.\n"
            "- **The mirage checks.** Does dropping any one stock kill it? Does a coin-flip "
            "relabelling of the names beat the real ranking? Does it survive trading costs?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Buy safe, short the maturity wall.** Here's a dollar in the long-short (long "
            "low-share, short high-share), gross of costs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.calendar_ls(PX, EV, signal_col='st_share', n_buckets=3, min_names=6, staleness_days=400, long_high=False)\n"
            "    s = st.calendar_ls_stats(ls)\n"
            "    ann, tnw = s['ann_pct'], s['t_nw']\n"
            "    cum = (1+ls['ls']).cumprod()\n"
            "else:\n"
            "    ann, tnw = R['ls_ann'], R['ls_t_nw']\n"
            "    cum = None\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "if cum is not None:\n"
            "    ax.plot(cum.index, cum.values, color=GREEN, lw=1.8)\n"
            "    ax.axhline(1.0, c='k', lw=.8)\n"
            "    ax.set_ylabel('growth of $1 (gross, long-short)')\n"
            "    ax.set_title(f'Safe minus risky: +{ann:.1f}%/yr, robust t = {tnw:+.2f} (clears the bar)')\n"
            "else:\n"
            "    ax.text(.5,.5,'run with cache for the equity curve',ha='center'); ax.set_axis_off()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'long-short: +{ann:.1f}%/yr gross, Newey-West t = {tnw:+.2f} (the bar is 2)')"
        ),
        md(
            f"It slopes up and — unusually — the statistics back it: **+{R['ls_ann']:.1f}%/yr** "
            f"gross growing \\$1 to \\${R['ls_cum']:.2f} over {R['n_months']} months, with a "
            f"robust *t* of **+{R['ls_t_nw']:.2f}**. The firms with the biggest near-term maturity "
            f"wall really did under-earn the termed-out names by ~{R['ls_mean_bps']:.0f} bps a "
            "month. Right sign, over the bar.\n\n"
            "**Now the part the claim really hangs on: does it bite harder when rates rise?** Split "
            "the same long-short at 2022, the start of the hiking cycle."
        ),
        code(
            "if HAVE_REAL:\n"
            "    e = st.era_split(ls, split='2022-01-01')\n"
            "    eb, et, lb, lt = e['early_bps'], e['early_t'], e['late_bps'], e['late_t']\n"
            "else:\n"
            "    eb, et, lb, lt = R['era_early_bps'], R['era_early_t'], R['era_late_bps'], R['era_late_t']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "ax.bar(['2009-2021\\n(low/falling rates)','2022-2026\\n(hiking / tight credit)'], [eb, lb],\n"
            "       color=[GREY, GREEN], width=.55)\n"
            "for i,(v,t_) in enumerate([(eb,et),(lb,lt)]): ax.annotate(f'{v:+.0f} bps/mo\\n(t={t_:+.2f})',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('long-short mean (bps/mo)')\n"
            "ax.set_title('Rollover risk bites ~2.6x harder in the rising-rate era')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'2009-2021: {eb:+.0f} bps/mo (t={et:+.2f})   |   2022-2026: {lb:+.0f} bps/mo (t={lt:+.2f})')"
        ),
        md(
            f"There it is: **+{R['era_early_bps']:.0f} bps/mo** in the easy-money years, "
            f"**+{R['era_late_bps']:.0f} bps/mo** once the Fed started hiking — about 2.6× bigger, "
            "both statistically significant. When refinancing actually got expensive, the "
            "short-term-debt-heavy names paid for it. That's not a story bolted on after the fact; "
            "it's the mechanism the claim named, showing up in the exact sub-period it predicted.\n\n"
            "**Is it a fluke of one stock?** Drop each name in turn and re-run — if the whole thing "
            "leans on one lucky ticker, it should collapse."
        ),
        code(
            "print('Drop-one-name jackknife (frozen): every one of 32 refits keeps t > 2')\n"
            f"print('   worst: drop {R['jk_min_name']} -> t = +{R['jk_min']:.2f}   best: drop {R['jk_max_name']} -> t = +{R['jk_max']:.2f}')\n"
            "fig, ax = plt.subplots(figsize=(8.2, 3.4))\n"
            f"ax.barh(['worst drop\\n({R['jk_min_name']})','best drop\\n({R['jk_max_name']})'], [{R['jk_min']}, {R['jk_max']}], color=GREEN, height=.5)\n"
            "ax.axvline(2, ls='--', c=RED, lw=1.2); ax.set_xlabel('Newey-West t after dropping that name')\n"
            "ax.set_title('No single stock carries it — all 32 refits stay over the bar')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"Every single drop-one refit stays above *t* = 2 (worst case +{R['jk_min']:.2f}). "
            "It isn't one stock, and it isn't one decade. This is about as robust as a signal on a "
            "small panel gets."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** High-short-term-share firms under-earn by ~+{R['ls_ann']:.1f}%/yr "
            f"(robust *t* = +{R['ls_t_nw']:.2f}), right-signed, holding in both eras and surviving "
            "a drop-one jackknife. The claim is confirmed on this tape.\n"
            "- **Rate-era bite — Confirmed.** The penalty is ~2.6× larger in the 2022+ hiking "
            "cycle — rollover risk really does bite when rates rise.\n"
            f"- **Tradability — Fragile.** Net of realistic costs it survives (+"
            f"{R['net'][(20,100)][1]:.1f}%/yr, *t* = +{R['net'][(20,100)][2]:.2f}), but the Sharpe "
            f"is a slim {R['net'][(20,100)][3]:.2f} and a harsher cost assumption knocks it under "
            "the bar. Present, not bankable.\n\n"
            "> The honest one-liner: *the maturity wall is a real, priced-too-slowly risk — but "
            "the tradable slice of it is thin, and part of the win is simply the safe, "
            "big-cap leg being a nice place to hide.*"
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Is it rollover, or is it quality?** Much of the spread is the *low-share* (safe, "
            "termed-out mega-cap) leg out-earning — the middle and high thirds are close. So some "
            "of this may be a **quality / low-volatility / defensive** tilt wearing a maturity "
            "costume. Isolating the pure rollover effect (controlling for size and leverage) is "
            "the next cut.\n"
            "- **Survivorship cuts *for* the result.** Our basket is current survivors — it can't "
            "include the firms whose maturity wall actually sank them. That omission makes the "
            "measured penalty **conservative**: the worst high-share names are missing.\n"
            "- **Sibling studies:** [540-distress-risk-anomaly](../../540-distress-risk-anomaly/), "
            "[123-altman-z](../../123-altman-z/) and [230-ohlson-o-score](../../230-ohlson-o-score/) "
            "rank on distress *probability*; [154-leverage-anomaly](../../154-leverage-anomaly/) "
            "ranks on the *amount* of leverage. This one is alone in ranking on the debt's "
            "**maturity mix** — *when* it comes due, not *how much* there is. See "
            "[docs/references.md](docs/references.md) for the exact dedup.\n\n"
            "*Think the alpha is pure rollover and not a quality tilt? Strip out size and leverage, "
            "show the residual spread survives on the size you'd actually run — then we'll talk.*"
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
            "# Debt-Maturity Rollover Risk — a quantitative teardown 🔬\n"
            "### A point-in-time calendar-time tercile long-short (Newey-West) · a drop-one-name "
            "jackknife · a pooled event-drift cross-check with a label-shuffle placebo · the 2022 "
            "rate-era cut · a cost/borrow stress · a 12-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **firms with a high short-term-debt share under-earn, worst when rates rise** "
            "— is, unusually for this desk, **supported** on the real tape. This is distinct from "
            "every distress sibling: [540](../../540-distress-risk-anomaly/), "
            "[123](../../123-altman-z/), [230](../../230-ohlson-o-score/) rank on default "
            "*probability*; [154](../../154-leverage-anomaly/) on the *amount* of leverage. This "
            "ranks on the debt's **maturity composition**.\n\n"
            "> ⚠️ **Data note.** EDGAR `DebtCurrent` / `LongTermDebtCurrent` / "
            "`LongTermDebtNoncurrent` + yfinance adjusted closes, "
            + str(R["n_names"]) + " names, ends "
            + R["end_lo"] + " → " + R["end_hi"] + ", as-of " + R["as_of"] + ". Point-in-time on "
            "the **filing date**. Survivorship named on the Signal axis (current-survivors basket, "
            "which drops the actual rollover blow-ups — conservative here). Numbers in "
            "[`docs/results.md`](../docs/results.md) (prices fingerprint `" + R["fp_prices"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** (returns) | `REAL` | claim tercile long-short (long low-share / short "
            f"high-share) **{R['ls_mean_bps']:+.1f} bps/mo** (+{R['ls_ann']:.1f}%/yr gross), "
            f"one-sample *t* = {R['ls_t_iid']:+.2f}, **Newey-West *t* = {R['ls_t_nw']:+.2f}**; "
            f"drop-one jackknife all > +2; asset-scaled NW *t* = {R['scaled_t_nw']:+.2f} |\n"
            f"| **Tradability** | `FRAGILE` | net of 20 bps + 100 bps borrow: NW *t* = "
            f"{R['net'][(20,100)][2]:+.2f}, Sharpe {R['net'][(20,100)][3]:.2f}; but breaks at "
            f"30 bps + 200 bps (NW *t* = {R['net'][(30,200)][2]:+.2f}) |\n"
            f"| **Rate-era bite?** | `CONFIRMED` | 2022+ era +{R['era_late_bps']:.0f} bps "
            f"(NW *t* = {R['era_late_t']:+.2f}) vs pre-2022 +{R['era_early_bps']:.0f} bps "
            f"(NW *t* = {R['era_early_t']:+.2f}) — ~2.6× larger when rates rise |\n\n"
            "> 💡 In plain words: a genuine, right-signed, robust return penalty on rollover risk "
            "— rare on this desk — that behaves exactly as the rates story predicts, but whose "
            "*tradable* residual after costs is thin and one assumption from vanishing."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $S_{i,q}$ be name $i$'s short-term-debt share at fiscal quarter $q$, disclosed on "
            "filing date $F_{i,q}$:\n\n"
            "$$S_{i,q} = \\frac{\\text{DebtCurrent} + \\text{LongTermDebtCurrent}}"
            "{\\text{DebtCurrent} + \\text{LongTermDebtCurrent} + \\text{LongTermDebtNoncurrent}}$$\n\n"
            "the fraction of the debt stack maturing within a year, known at $F_{i,q}$. The claims:\n\n"
            "- **H₁ (under-earning).** A cross-sectional long-short **long low-$S$ / short high-$S$** "
            "earns a positive forward return spread — high-rollover-risk firms under-earn.\n"
            "- **H₂ (rate-conditional).** That spread is **larger in the 2022+ rising-rate era**.\n"
            "- **H₃ (tradable).** It survives realistic long-short costs + borrow.\n\n"
            "We find **H₁ supported** (NW *t* = "
            f"{R['ls_t_nw']:+.2f}, jackknife-robust), **H₂ supported** (era +"
            f"{R['era_late_bps']:.0f} vs +{R['era_early_bps']:.0f} bps), and **H₃ only marginally** "
            "(survives the base cost model, breaks under stress). A *negative* significant spread "
            "would have been the wrong sign — a risk premium, not a penalty — and stamped `NONE`; "
            "the sign here is right and matches the distress-anomaly literature (Campbell-Hilscher-"
            "Szilagyi 2008)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The primary is a **calendar-time** long-short, not a pooled event study, because "
            "balance-sheet signals are persistent and filings cluster: a calendar series of "
            "monthly long-short returns lets a **Newey-West (6-lag) HAC *t*** do the honest work "
            "the desk's `REAL` bar is written against. The panel is thin, so we sort into "
            "**terciles** and require ≥ 6 names. Because this is a *positive*, we hold it to a "
            "higher bar than usual: a **drop-one-name jackknife** (does any single stock carry "
            "it?), **half/quartile/quintile** re-sorts, a **pooled event-drift + label-shuffle "
            "placebo** cross-check, and a **cost/borrow stress** — all before stamping green."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** {R['n_events']:,} (ticker, filing) debt quarters across {R['n_names']} "
            f"names, ends {R['end_lo']} → {R['end_hi']}, each stamped with its 10-Q/10-K filing "
            "date (point-in-time; a missing short-term leg is a genuine zero).\n"
            "- **Primary.** Monthly tercile long-short (long low-$S$ / short high-$S$), one "
            "execution lag; Newey-West + one-sample *t*, Sharpe, hit rate.\n"
            "- **Robustness.** Drop-one-name jackknife (32 refits); half/quartile/quintile sorts; "
            "staleness 200 vs 400 days; the asset-scaled signal `(DC+LC)/Assets`.\n"
            "- **Cross-check.** Pooled event drift over 21/63/126 trading days, one-day-lag entry, "
            "low-minus-high tercile, one-sample *t* + 10k-draw placebo, and the monotonicity "
            "picture.\n"
            "- **Rate-era.** Calendar long-short split at 2022-01-01 (plus 2017/2019 alternates).\n"
            "- **Execution.** Long-short net of one-way cost × turnover (both legs) + short borrow, "
            "stressed 10/50 → 50/300 bps.\n"
            "- **Control.** Synthetic panel, planted-penalty knob; the null must not fire across "
            "12 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The primary — calendar-time tercile long-short (Newey-West)\n\n"
            "Long the low-share (safe) tercile, short the high-share (rollover-risk) tercile each "
            "month, earn next month's return. Decisive statistic: the HAC *t* of the monthly series."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.calendar_ls(PX, EV, signal_col='st_share', n_buckets=3, min_names=6, staleness_days=400, long_high=False)\n"
            "    s = st.calendar_ls_stats(ls)\n"
            "    ls_sc = st.calendar_ls(PX, EV, signal_col='st_debt_assets', staleness_days=400, long_high=False)\n"
            "    s_sc = st.calendar_ls_stats(ls_sc)\n"
            "    print(f\"calendar long-short: {s['mean_bps']:+.1f} bps/mo (+{s['ann_pct']:.2f}%/yr gross), \"\n"
            "          f\"n={s['n_months']} months, avg cross-section {s['avg_n']:.1f}\")\n"
            "    print(f\"  one-sample t = {s['t_iid']:+.2f}   Newey-West(6) t = {s['t_nw']:+.2f}   \"\n"
            "          f\"Sharpe {s['sharpe']:.2f}   hit {s['hit']*100:.0f}%   turnover {s['avg_turnover']:.3f}\")\n"
            "    print(f\"  asset-scaled signal: {s_sc['mean_bps']:+.1f} bps/mo, NW t = {s_sc['t_nw']:+.2f}, Sharpe {s_sc['sharpe']:.2f}\")\n"
            "    cum = (1+ls['ls']).cumprod(); nser = ls['n']\n"
            "else:\n"
            "    cum = None; nser = None\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "if cum is not None:\n"
            "    a1.plot(cum.index, cum.values, color=GREEN, lw=1.8); a1.axhline(1, c='k', lw=.8)\n"
            "    a1.set_ylabel('growth of $1 (gross)'); a1.set_title(f\"Long-short: +{R['ls_ann']:.1f}%/yr, NW t = {R['ls_t_nw']:+.2f}\")\n"
            "    a2.plot(nser.index, nser.values, color=GREY, lw=1.5); a2.set_ylabel('names in cross-section')\n"
            "    a2.set_title('Thin & uneven: ~26 names on average, sparser early')\n"
            "else:\n"
            "    for a in (a1,a2): a.text(.5,.5,'run with cache',ha='center'); a.set_axis_off()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"frozen: {R['ls_mean_bps']:+.1f} bps/mo, NW t = {R['ls_t_nw']:+.2f}, \"\n"
            "      f\"staleness-200 NW t = {R['ls200_t_nw']:+.2f}, asset-scaled NW t = {R['scaled_t_nw']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: **+{R['ls_mean_bps']:.0f} bps/month** (~+{R['ls_ann']:.1f}%/yr "
            f"gross), and the HAC *t* is **+{R['ls_t_nw']:.2f}** — over the bar. Every "
            f"specification agrees: staleness-200 NW *t* = +{R['ls200_t_nw']:.2f}, asset-scaled "
            f"(the maturity-wall *level*, not just its share) NW *t* = +{R['scaled_t_nw']:.2f}, and "
            f"the half/quartile/quintile sorts read +{R['t_halves']:.2f} / +{R['t_quartiles']:.2f} "
            f"/ +{R['t_quintiles']:.2f}. Right-signed and robust."
        ),
        md(
            "### 4b · Drop-one-name jackknife — not a single-stock artifact\n\n"
            "A green stamp on a 32-name panel demands proof it isn't one lucky ticker. Refit the "
            "primary NW *t* dropping each name in turn (frozen; live-run in `examples/verify.py`)."
        ),
        code(
            "jk = {'min': (R['jk_min'], R['jk_min_name']), 'max': (R['jk_max'], R['jk_max_name'])}\n"
            "fig, ax = plt.subplots(figsize=(8.6, 3.4))\n"
            "ax.barh([f\"worst drop ({jk['min'][1]})\", f\"best drop ({jk['max'][1]})\"],\n"
            "        [jk['min'][0], jk['max'][0]], color=GREEN, height=.5)\n"
            "ax.axvline(2, ls='--', c=RED, lw=1.2); ax.set_xlabel('Newey-West t after dropping that name')\n"
            "ax.set_title('All 32 drop-one refits stay above t = 2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"jackknife: every one of 32 refits keeps NW t > 2 \"\n"
            "      f\"(worst drop {jk['min'][1]} -> +{jk['min'][0]:.2f}, best drop {jk['max'][1]} -> +{jk['max'][0]:.2f})\")"
        ),
        md(
            f"> 💡 In plain words: the weakest the signal ever gets is NW *t* = +{R['jk_min']:.2f} "
            f"(dropping {R['jk_min_name']}) — still over the bar. No single name is doing the work. "
            "That is the check most spurious 'anomalies' on this desk fail; this one passes it."
        ),
        md(
            "### 4c · The rate-era cut — H₂, the claim's own prediction\n\n"
            "Split the calendar long-short at 2022-01-01 (the hiking cycle). The claim says the "
            "penalty should be *larger* there."
        ),
        code(
            "if HAVE_REAL:\n"
            "    e = st.era_split(ls, split='2022-01-01')\n"
            "    eb, et, en = e['early_bps'], e['early_t'], e['early_n']\n"
            "    lb, lt, ln = e['late_bps'], e['late_t'], e['late_n']\n"
            "else:\n"
            "    eb, et, en = R['era_early_bps'], R['era_early_t'], R['era_early_n']\n"
            "    lb, lt, ln = R['era_late_bps'], R['era_late_t'], R['era_late_n']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.2))\n"
            "ax.bar([f'2009-2021\\n(n={en})', f'2022-2026\\n(n={ln})'], [eb, lb], color=[GREY, GREEN], width=.5)\n"
            "for i,(v,t_) in enumerate([(eb,et),(lb,lt)]): ax.annotate(f'{v:+.0f} bps\\n(NW t={t_:+.2f})',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('long-short mean (bps/mo)')\n"
            "ax.set_title('Rollover penalty ~2.6x bigger in the rising-rate era')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'2009-2021: {eb:+.1f} bps NW t={et:+.2f} (n={en})  |  2022-2026: {lb:+.1f} bps NW t={lt:+.2f} (n={ln})')"
        ),
        md(
            f"> 💡 In plain words: +{R['era_early_bps']:.0f} bps (NW *t* = {R['era_early_t']:+.2f}) "
            f"pre-2022, +{R['era_late_bps']:.0f} bps (NW *t* = {R['era_late_t']:+.2f}) in the "
            "hiking cycle — both significant, and ~2.6× bigger when refinancing actually got "
            "expensive. The effect is not a static factor; it is **rate-conditional**, exactly as "
            "H₂ predicts. (Alternate splits agree: post-2017 +62 bps *t* +2.83, post-2019 +65 bps "
            "*t* +2.51.)"
        ),
        md(
            "### 4d · The cross-check — pooled event drift + placebo + monotonicity\n\n"
            "Bucket all events by share; low-minus-high forward drift with a label-shuffle null. "
            "The claim predicts a *decreasing* tercile ladder (high share earns least)."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for h in st.HORIZONS:\n"
            "        es = st.event_summary(PX, EV, horizon=h, n_buckets=3, n_draws=4000, long_high=False)\n"
            "        rows.append((h, es['ls_mean']*100, es['t'], es['ls_win']*100, es['p_placebo']))\n"
            "    mono = st.bucket_means(st.event_drift_frame(PX, EV, horizon=63), 3)*100\n"
            "else:\n"
            "    for h in st.HORIZONS:\n"
            "        d = R['drift'][h]; rows.append((h, d[3], d[4], d[5], d[6]))\n"
            "    mono = np.array(R['mono63'])\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "hs = [r[0] for r in rows]; ts = [r[2] for r in rows]\n"
            "a1.bar([f'{h}d' for h in hs], ts, color=[GREY,GREEN,GREEN], width=.55)\n"
            "a1.axhline(2, ls='--', c=RED, lw=1); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_ylabel('one-sample t (low-minus-high drift)'); a1.set_title('Builds with horizon; clears |t|=2 by a quarter')\n"
            "a2.bar(['low\\nshare','mid','high\\nshare'], mono, color=[GREEN, GREY, GREY], width=.55)\n"
            "for i,v in enumerate(mono): a2.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('3-month forward return'); a2.set_title('Concentrated in the safe (low-share) leg')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h,ls_,t,win,p in rows:\n"
            "    print(f'  H={h:>3}d: low-minus-high {ls_:+.2f}%  t={t:+.2f}  win={win:.0f}%  placebo p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the pooled drift **builds with horizon** — "
            f"{R['drift'][21][3]:+.2f}% at 21d ({R['drift'][21][4]:+.2f}), "
            f"{R['drift'][63][3]:+.2f}% at 63d ({R['drift'][63][4]:+.2f}), "
            f"{R['drift'][126][3]:+.2f}% at 126d ({R['drift'][126][4]:+.2f}) — placebo *p* "
            f"{R['drift'][63][6]:.3f}/{R['drift'][126][6]:.3f}. But the terciles are **not a clean "
            f"ladder** ({R['mono63'][0]:+.2f}% / {R['mono63'][1]:+.2f}% / {R['mono63'][2]:+.2f}% "
            "low→high): the win is concentrated in the **low-share (safe) leg** out-earning, not a "
            "smooth penalty across the whole distribution. Honest caveat, flagged for the "
            "quality-tilt worry below."
        ),
        md(
            "### 4e · Tradability — the timer (stressed)\n\n"
            "Calendar long-short net of one-way costs × turnover (both legs) + short borrow, across "
            "four cost/borrow assumptions."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for cb, bb in [(10.,50.),(20.,100.),(30.,200.),(50.,300.)]:\n"
            "        nt = st.calendar_ls_net(ls, cost_bps=cb, borrow_bps_ann=bb)\n"
            "        rows.append((cb, bb, nt['net_ann_pct'], nt['net_t_nw'], nt['net_sharpe']))\n"
            "else:\n"
            "    for (cb,bb),v in R['net'].items(): rows.append((cb, bb, v[1], v[2], v[3]))\n"
            "labels = [f'{int(cb)}+{int(bb)}' for cb,bb,_,_,_ in rows]\n"
            "ts = [r[3] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "cols = [GREEN if t>=2 else RED for t in ts]\n"
            "ax.bar(labels, ts, color=cols, width=.55)\n"
            "for i,(cb,bb,a,t,sh) in enumerate(rows): ax.annotate(f't={t:+.2f}\\n(Sh {sh:.2f})',(i,t),ha='center',va='bottom')\n"
            "ax.axhline(2, ls='--', c='k', lw=1); ax.set_ylabel('net Newey-West t'); ax.set_xlabel('cost bps + borrow bps/yr')\n"
            "ax.set_title('Survives the base case, breaks under stress = FRAGILE')\n"
            "plt.tight_layout(); plt.show()\n"
            "for cb,bb,a,t,sh in rows: print(f'  cost {int(cb)}bps + borrow {int(bb)}bps/yr: +{a:.2f}%/yr net, NW t={t:+.2f}, Sharpe {sh:.2f}')"
        ),
        md(
            f"> 💡 In plain words: turnover is low (~{R['ls_turn']:.2f}/mo — a slow balance-sheet "
            f"signal), so at 20 bps + 100 bps borrow the edge **survives** "
            f"(+{R['net'][(20,100)][1]:.1f}%/yr, NW *t* = {R['net'][(20,100)][2]:+.2f}, Sharpe "
            f"{R['net'][(20,100)][3]:.2f}). But push to 30 bps + 200 bps and it slips under the bar "
            f"(NW *t* = {R['net'][(30,200)][2]:+.2f}); at 50 bps + 300 bps it's gone "
            f"({R['net'][(50,300)][2]:+.2f}). A real net edge on the base model that is one "
            "cost-bump from vanishing — **FRAGILE**, not a robust `INVESTABLE`."
        ),
        md(
            "### 4f · Faithful-engine & power control\n\n"
            "Synthetic price + signal panel with a TUNABLE planted penalty (high-share names drift "
            "down). The null (edge = 0) is checked over **12 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(12):\n"
            "    p0, e0 = data.synthetic_panel(edge=0.0, seed=861 + s_)\n"
            "    null_ts.append(st.synthetic_detect(p0, e0)['t_nw'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "p1, e1 = data.synthetic_panel(edge=0.15, seed=861)\n"
            "planted_t = st.synthetic_detect(p1, e1)['t_nw']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12,.12,12), null_ts, color=GREY, s=45, label='null worlds (edge=0), 12 seeds')\n"
            "ax.scatter([1], [planted_t], color=GREEN, s=95, zorder=5, label='planted penalty (edge=0.15)')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=.6)\n"
            "ax.set_xticks([0,1]); ax.set_xticklabels(['null x 12','planted'])\n"
            "ax.set_ylabel('claim long-short Newey-West t')\n"
            "ax.set_title('Control: the null barely fires; a planted penalty lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null: mean NW t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/12  |  planted NW t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 12 null worlds the detector averages NW *t* = "
            f"{R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and crosses |t|=2 only "
            f"{R['syn_null_fire']}/12 times — about what chance gives you. A planted penalty reads "
            f"NW *t* = {R['syn_planted_t']:.2f}. The machinery is unbiased and powered, so the "
            f"real-tape +{R['ls_t_nw']:.2f} is a genuine effect, not a broken pipeline. *(Power "
            "check only — never cited in support of a real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal (returns) `REAL`** — claim tercile long-short {R['ls_mean_bps']:+.1f} "
            f"bps/mo (+{R['ls_ann']:.1f}%/yr gross), one-sample *t* = {R['ls_t_iid']:+.2f}, "
            f"**Newey-West *t* = {R['ls_t_nw']:+.2f}**; drop-one jackknife all > +2 (min "
            f"+{R['jk_min']:.2f}); half/quartile/quintile +{R['t_halves']:.2f}/"
            f"+{R['t_quartiles']:.2f}/+{R['t_quintiles']:.2f}; asset-scaled +{R['scaled_t_nw']:.2f}; "
            f"pooled event drift clears |t|=2 by a quarter (placebo *p* ≤ {R['drift'][63][6]:.02f}). "
            "Right-signed, robust, tape-confirmed.\n"
            f"- **Tradability `FRAGILE`** — net of 20 bps + 100 bps borrow: +"
            f"{R['net'][(20,100)][1]:.2f}%/yr, NW *t* = {R['net'][(20,100)][2]:+.2f}, Sharpe "
            f"{R['net'][(20,100)][3]:.2f}; but breaks at 30 bps + 200 bps (NW *t* = "
            f"{R['net'][(30,200)][2]:+.2f}). Present, not bankable.\n"
            f"- **Rate-era bite `CONFIRMED`** — 2022+ +{R['era_late_bps']:.0f} bps (NW *t* = "
            f"{R['era_late_t']:+.2f}) vs pre-2022 +{R['era_early_bps']:.0f} bps (NW *t* = "
            f"{R['era_early_t']:+.2f}); ~2.6× larger when rates rise, as claimed."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Rollover, or quality in disguise?** The tercile picture (§4d) shows the spread is "
            "concentrated in the **low-share (safe, termed-out mega-cap) leg**. That raises the "
            "real possibility that part of this is a **quality / low-volatility / size** tilt, not "
            "pure maturity-timing causation. The decisive next cut is a spread residualised on "
            "size, leverage and a low-vol factor — does the rollover effect survive on its own?\n"
            "- **Survivorship is conservative here.** The current-survivors basket omits the firms "
            "whose maturity wall actually sank them — the extreme high-share left tail — so the "
            "measured penalty is if anything an *under*-estimate of the true short-vs-long spread.\n"
            "- **Dedup map:** [540-distress-risk-anomaly](../../540-distress-risk-anomaly/), "
            "[123-altman-z](../../123-altman-z/), [230-ohlson-o-score](../../230-ohlson-o-score/) "
            "rank on default *probability* (a composite hazard); "
            "[154-leverage-anomaly](../../154-leverage-anomaly/) on the *amount* of leverage. None "
            "ranks on the debt's **maturity composition** — *when* it comes due — which is this "
            "study's own axis.\n\n"
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
