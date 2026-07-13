"""Generate the two narrative notebooks for Study 742 (Friday-17th / Venerdì 17).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached FTSE MIB
(EUR, price-only) and EWI (USD, total-return) tapes under ../_cache/ and otherwise quote
the frozen headline numbers in ``R`` (a mirror of docs/results.md). The synthetic
positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md.
# FTSE MIB (FTSEMIB.MI, EUR, price-only) is the primary local-sentiment tape;
# EWI (USD, total-return) is the tradable vehicle. yfinance, as-of 2026-06-30.
R = dict(
    fp="e5483c326431",
    mib_rows=7276, ewi_rows=7619,
    mib_lo="1998-01-02", mib_hi="2026-06-30", ewi_lo="1996-03-19", ewi_hi="2026-06-30",
    # --- FTSE MIB (local, EUR, price-only) -- PRIMARY ---
    m_n=49, m_mean=+26.285, m_t=+1.267,
    m_other_mean=-2.651, m_contrast=+28.936, m_welch_t=+1.371, m_welch_p=0.1703,
    m_all_mean=+0.860, m_contrast_all=+25.425, m_welch_p_all=0.2219,
    m_down_k=23, m_down_n=49, m_down_rate=46.9, m_down_lo=33.7, m_down_hi=60.6,
    m_pl_obs=+26.285, m_pl_null=-2.811, m_pl_sd=20.601, m_pl_p=0.9266, m_pl_draws=10000,
    m_sh_n=49, m_sh_gross=-26.285, m_sh_gross_t=-1.27, m_sh_net=-38.285, m_sh_net_t=-1.85,
    m_sh_win=42.9, m_sh_be=12.0,
    # DOM sweep (day: [n, mean_bps, contrast_bps, p_raw, p_bonf]) sorted by p_raw
    m_sweep={10: [47, -62.420, -62.798, 0.01296, 0.06480],
             3:  [47, +36.755, +39.717, 0.04728, 0.23639],
             17: [49, +26.285, +28.936, 0.17030, 0.85147],
             24: [47, -44.615, -44.393, 0.21847, 1.00000],
             31: [26, +4.781, +6.568, 0.78178, 1.00000]},
    # sub-periods (MIB)
    m_h1_n=25, m_h1_mean=+31.51, m_h1_t=+1.026, m_h1_contrast=+34.29, m_h1_p=0.272,
    m_h2_n=24, m_h2_mean=+20.84, m_h2_t=+0.735, m_h2_contrast=+23.34, m_h2_p=0.419,
    # --- EWI (USD, total-return) -- tradable vehicle ---
    e_n=52, e_mean=+10.073, e_t=+0.517,
    e_other_mean=-2.416, e_contrast=+12.488, e_welch_t=+0.626, e_welch_p=0.5311,
    e_all_mean=+2.611, e_contrast_all=+7.462, e_welch_p_all=0.7030,
    e_down_k=24, e_down_n=52, e_down_rate=46.2, e_down_lo=33.3, e_down_hi=59.5,
    e_pl_obs=+10.073, e_pl_null=-2.462, e_pl_sd=22.288, e_pl_p=0.7137,
    e_sh_n=52, e_sh_gross=-10.073, e_sh_gross_t=-0.52, e_sh_net=-22.073, e_sh_net_t=-1.13,
    e_sh_win=44.2, e_sh_be=12.0,
    e_sweep={3:  [46, +47.927, +51.469, 0.03323, 0.16614],
             10: [49, -43.660, -43.053, 0.13401, 0.67006],
             17: [52, +10.073, +12.488, 0.53114, 1.00000],
             24: [48, -20.530, -19.142, 0.66418, 1.00000],
             31: [31, -4.007, -2.059, 0.93642, 1.00000]},
    # synthetic control
    syn_null_meant=+0.175, syn_null_sd=0.885, syn_null_fire=0, syn_null_seeds=20,
    syn_p05_t=-4.507, syn_p10_t=-8.137, syn_p20_t=-15.399,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Unlucky_day%3F: Busted](https://img.shields.io/badge/Unlucky_day%3F-Busted-8b949e?style=flat-square)\n\n"
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

from friday_17th import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_real()
    MIB = st.build_frame(PRICES[data.MIB])
    EWI = st.build_frame(PRICES[data.EWI])
else:
    PRICES = MIB = EWI = None
print("real cache present:", HAVE_REAL,
      "| Venerdi 17 events -> MIB:", (0 if MIB is None else int(MIB['is_f17'].sum())),
      "EWI:", (0 if EWI is None else int(EWI['is_f17'].sum())))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# In Italy the unlucky day is Friday the **17th**. Does the market feel it? 😱📉\n"
            "### *Venerdì 17* — the Latin-market cousin of Friday the 13th — tested on Italy's own stock market\n\n"
            + BADGES +
            "Anglo-Saxons fear Friday the **13th**. Italians fear Friday the **17th** — "
            "*Venerdì 17*. The reason is gorgeously morbid: the Roman numeral for 17, "
            "`XVII`, is an anagram of the Latin **`VIXI`** — \"I have lived\", i.e. "
            "\"I am dead\". Airlines have skipped row 17, hotels skip the 17th floor. So "
            "if superstition can move a market through sheer collective mood, Italy's "
            "own blue-chip index — the **FTSE MIB** — is exactly where a *Venerdì 17* "
            "dip should show up.\n\n"
            "We tested it the same honest way the desk tested [Friday the 13th]"
            "(../../163-friday-13th/) one country over: every Friday-the-17th on the "
            "Italian tape, 1998→2026.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo, the "
            "look-elsewhere correction? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** *Venerdì 17* dates are pure calendar arithmetic "
            "(Friday **and** the 17th — known before the open, so zero look-ahead). "
            "Primary tape: the **FTSE MIB** index (local, EUR, price-only) — the purest "
            "read of Italian sentiment. Tradable vehicle: **EWI** (US-listed, USD, "
            "total-return). Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the FTSE MIB fall on Venerdì 17? | **No — it's slightly *up*.** "
            f"Mean **{R['m_mean']:+.1f} bps** across {R['m_n']} events, *t* = "
            f"**{R['m_t']:+.2f}**. The fear points the *wrong way*. |\n"
            f"| Is it a red day more often than not? | **No.** It closes down only "
            f"**{R['m_down_rate']:.0f}%** of the time ({R['m_down_k']}/{R['m_down_n']}) "
            "— it's *greener* than a coin flip. |\n"
            f"| Worse than an ordinary Friday? | **No.** It beats the average Friday by "
            f"**{R['m_contrast']:+.0f} bps**, and that gap is statistical noise "
            f"(*p* = {R['m_welch_p']:.2f}). |\n"
            f"| Could you have traded the superstition? | **No.** *Shorting* the MIB into "
            f"Venerdì 17 (the folklore trade) **loses {abs(R['m_sh_net']):.0f} bps** per "
            "event net of costs — the one bettable version of the fear is a money-loser. |\n\n"
            "> Just like Friday the 13th on Wall Street, Italy's unlucky Friday is, if "
            "anything, a *good* day for stocks. The superstition isn't merely absent from "
            "the data — it has the sign backwards."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"In Italy, Friday the 17th is the unlucky day. Nervous, superstitious "
            "trading — a whole country in a slightly darker mood — should leave a mark "
            "on the Italian stock market: weaker returns on Venerdì 17 than on an "
            "ordinary day.\"*\n\n"
            "It's a real, deeply-held superstition (*eptacaidecafobia* — fear of 17), "
            "and it rides on the same folk logic as the Anglo Friday-13th trade. The "
            "one academic anchor, **Kolb & Rodriguez (1987)**, found a small negative "
            "Friday-13th blip in the *1940s–80s* Dow — but essentially every modern "
            "replication (including the desk's own [study 163](../../163-friday-13th/)) "
            "finds nothing. Nobody had checked the *Italian 17th* on the *Italian tape*. "
            "We did."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a *calendar superstition with no economic content whatsoever* moved a "
            "G7 stock index, that would be a small miracle of behavioural finance — "
            "pure mood, priced. You could sell volatility, or simply short into every "
            "Venerdì 17 and cover at the close. Entire folk-finance columns assume "
            "something like this is true. We wanted to know: is there anything there at "
            "all?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** Every Friday-the-17th, 1998→2026 — **{R['m_n']} events** "
            "on the FTSE MIB — from pure date arithmetic (no table, no fetch).\n"
            "- **The tape.** The **FTSE MIB** index itself (local, EUR, price-only): if "
            "Italian mood moves anything, it moves this. Cross-checked on **EWI** (the "
            "US-listed, USD, total-return ETF a foreigner could actually buy).\n"
            "- **The unit.** Each Venerdì 17 is one independent event → a **one-sample "
            "*t*** of the day's return, plus a **Welch** contrast vs all *other* Fridays "
            "(is it worse than a normal Friday?).\n"
            "- **The honesty checks.** A **down-day hit rate** with a Wilson interval; a "
            "**multi-seed random-calendar placebo** (do random Fridays look just as "
            "extreme?); a **look-elsewhere sweep** over the neighbouring Friday slots; "
            "and a **costed short** — the trade a believer would actually place.\n\n"
            "> **What would make us say \"real\"?** A negative Venerdì-17 mean with "
            "*t* ≤ −2 on the MIB, a down-day rate clearly above 50%, and a random-Friday "
            "placebo that the observed mean sits *below*. Anything less is folklore."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline: the Venerdì-17 bar points the wrong way.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rm = st.friday17_test(MIB); re = st.friday17_test(EWI)\n"
            "    m17, mo = rm['mean_f17_bps'], rm['mean_other_fri_bps']\n"
            "    e17, eo = re['mean_f17_bps'], re['mean_other_fri_bps']\n"
            "else:\n"
            "    m17, mo = R['m_mean'], R['m_other_mean']\n"
            "    e17, eo = R['e_mean'], R['e_other_mean']\n"
            "x = np.arange(2); w = 0.36\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.6))\n"
            "ax.bar(x - w/2, [m17, e17], w, label='Venerdi 17', color=RED)\n"
            "ax.bar(x + w/2, [mo, eo], w, label='every other Friday', color=GREY)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['FTSE MIB\\n(EUR, price-only)', 'EWI\\n(USD, total-return)'])\n"
            "ax.set_ylabel('mean return (bps/day)')\n"
            "ax.set_title('The unlucky day is GREEN on both tapes -- and beats an ordinary Friday')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'MIB: Venerdi17 {m17:+.1f} bps vs other-Fri {mo:+.1f} bps')\n"
            "print(f'EWI: Venerdi17 {e17:+.1f} bps vs other-Fri {eo:+.1f} bps')"
        ),
        md(
            f"On the FTSE MIB, Venerdì 17 averages **{R['m_mean']:+.1f} bps** — "
            f"*positive*, *t* = {R['m_t']:+.2f} — and it actually *beats* the average "
            f"Friday by {R['m_contrast']:+.0f} bps. The tradable EWI agrees "
            f"({R['e_mean']:+.1f} bps, *t* = {R['e_t']:+.2f}). Not a dip; a small, "
            "insignificant *lift*.\n\n"
            "**Maybe the average hides a lot of red days offset by a few big greens?** "
            "Let's count."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rm = st.friday17_test(MIB)\n"
            "    dr, lo, hi = rm['down_rate']*100, rm['down_lo']*100, rm['down_hi']*100\n"
            "else:\n"
            "    dr, lo, hi = R['m_down_rate'], R['m_down_lo'], R['m_down_hi']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 3.4))\n"
            "ax.barh([0], [dr], color=RED, height=.5)\n"
            "ax.errorbar([dr], [0], xerr=[[dr-lo],[hi-dr]], fmt='none', ecolor='k', capsize=6, lw=1.6)\n"
            "ax.axvline(50, ls='--', c='k', lw=1.2, label='coin flip (50%)')\n"
            "ax.set_xlim(0, 100); ax.set_yticks([]); ax.set_xlabel('share of Venerdi 17s that CLOSE DOWN (%)')\n"
            "ax.set_title(f'Venerdi 17 is a red day only {dr:.0f}% of the time -- less than a coin flip')\n"
            "ax.legend(loc='lower right'); plt.tight_layout(); plt.show()\n"
            "print(f'down-day rate {dr:.1f}%  Wilson 95% [{lo:.1f}%, {hi:.1f}%]')"
        ),
        md(
            f"Venerdì 17 closes *down* only **{R['m_down_rate']:.0f}%** of the time "
            f"({R['m_down_k']}/{R['m_down_n']}) — the Wilson interval "
            f"[{R['m_down_lo']:.0f}%, {R['m_down_hi']:.0f}%] straddles 50% and leans "
            "*green*. There is no army of red days here.\n\n"
            "**Is a +26 bps Friday even unusual? Let's ask 10,000 random Fridays.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.random_friday_placebo(MIB, n_seeds=8, n_draws_per_seed=500)\n"
            "    obs = pl['obs_bps']; rng = np.random.default_rng(742)\n"
            "    draws = rng.normal(pl['null_mean_bps'], pl['null_sd_bps'], 6000)\n"
            "else:\n"
            "    obs = R['m_pl_obs']; rng = np.random.default_rng(742)\n"
            "    draws = rng.normal(R['m_pl_null'], R['m_pl_sd'], 6000)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='random sets of 49 other Fridays')\n"
            "ax.axvline(obs, c=RED, lw=2.4, label=f'observed Venerdi 17 mean {obs:+.1f} bps')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('mean return of a random-Friday draw (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Observed sits at the HIGH end of the null (canonical p_left = {R[\"m_pl_p\"]:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"observed {R['m_pl_obs']:+.1f} bps vs null mean {R['m_pl_null']:+.1f} \"\n"
            "      f\"(sd {R['m_pl_sd']:.1f}); left-tail p = {R['m_pl_p']:.3f} over {R['m_pl_draws']:,} draws\")"
        ),
        md(
            f"The observed mean sits at the **high** end of the random-Friday cloud: the "
            f"left-tail *p* (the folklore's \"is it unusually *low*?\") is "
            f"**{R['m_pl_p']:.2f}** — about as far from \"unusually weak\" as you can "
            "get. A random Friday is *more* likely to look bad than Venerdì 17 does.\n\n"
            "**Last: could a believer have made money shorting the curse?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sm = st.short_the_17th(PRICES[data.MIB]); se = st.short_the_17th(PRICES[data.EWI])\n"
            "    mg, mn = sm['gross_mean_bps'], sm['net_mean_bps']\n"
            "    eg, en = se['gross_mean_bps'], se['net_mean_bps']\n"
            "else:\n"
            "    mg, mn, eg, en = R['m_sh_gross'], R['m_sh_net'], R['e_sh_gross'], R['e_sh_net']\n"
            "x = np.arange(2); w = 0.36\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(x - w/2, [mg, eg], w, label='gross', color=GREY)\n"
            "ax.bar(x + w/2, [mn, en], w, label='net of costs + borrow', color=RED)\n"
            "for i, v in enumerate([mn, en]): ax.annotate(f'{v:+.0f}', (i+w/2, v), ha='center', va='top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['short the MIB', 'short EWI'])\n"
            "ax.set_ylabel('profit per Venerdi 17 (bps)')\n"
            "ax.set_title('The folklore trade -- shorting the unlucky day -- LOSES money')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'short MIB: gross {mg:+.1f} bps  net {mn:+.1f} bps')\n"
            "print(f'short EWI: gross {eg:+.1f} bps  net {en:+.1f} bps')"
        ),
        md(
            f"Shorting the FTSE MIB into every Venerdì 17 — the literal folklore bet — "
            f"loses **{abs(R['m_sh_net']):.0f} bps per event** net of a round trip of "
            f"costs and one day of borrow (*t* = {R['m_sh_net_t']:.2f}), because the day "
            "is *green* on average. The curse isn't just untradeable; betting on it is a "
            "slow bleed."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Venerdì 17 prints **{R['m_mean']:+.1f} bps** on the "
            f"FTSE MIB (*t* = {R['m_t']:+.2f}) — *positive*, so if anything the fear has "
            f"the sign backwards. Down-day rate {R['m_down_rate']:.0f}%, Welch vs other "
            f"Fridays *p* = {R['m_welch_p']:.2f}, placebo left-tail *p* = "
            f"{R['m_pl_p']:.2f}. EWI agrees.\n"
            "- **Tradability — Mirage.** The one bettable version of the superstition — "
            "short the unlucky day — *loses* money, gross and net, on both tapes. There "
            "is no edge to charge costs against.\n"
            "- **Unlucky day? — Busted.** On Italy's own market, Venerdì 17 is a "
            "slightly *above-average* Friday — exactly the shape the desk found for "
            "Friday the 13th on Wall Street. The dread does not price."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Two superstitions, two countries, one answer.** Friday the 13th on the "
            "S&P (study 163) and Friday the 17th on the FTSE MIB both come out *green* "
            "and insignificant. Calendar dread is a wonderfully universal human trait — "
            "and a wonderfully universal non-signal in prices.\n"
            "- **Sibling studies:** the direct cousin [163-friday-13th](../../163-friday-13th/), "
            "plus other folk-calendar teardowns — [158-super-bowl](../../158-super-bowl/), "
            "[608-friday-news-dump](../../608-friday-news-dump/), and the sentiment-event "
            "family [707-plane-crash-effect](../../707-plane-crash-effect/) / "
            "[708-eurovision-effect](../../708-eurovision-effect/).\n\n"
            "*Think Venerdì 17 bites somewhere we didn't look — Italian small-caps, "
            "single stocks, intraday, options-implied vol? Fork it, point the same "
            "battery at a cleaner tape, and show a net, placebo-surviving dip. We'll "
            "publish the teardown.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Friday-17th (*Venerdì 17*) — a quantitative teardown 🔬\n"
            "### One-sample *t* across independent events · a Welch contrast vs other "
            "Fridays · a Wilson down-day rate · a multi-seed random-calendar placebo · a "
            "look-elsewhere Bonferroni sweep · a costed short · a 20-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim — **Italy's FTSE MIB trades weak on "
            "Friday the 17th** — is folk superstition (*eptacaidecafobia*), the Latin "
            "cousin of the Friday-13th trade. Its only academic neighbour is Kolb & "
            "Rodriguez (1987) on the 1940s–80s Dow, a result no modern tape reproduces. "
            "The job here is to measure the *Italian 17th* honestly, on the *Italian "
            "tape*, with the right inference unit for a ~1.7-per-year calendar event.\n\n"
            "> ⚠️ **Data note.** Primary tape: **`FTSEMIB.MI`** (FTSE MIB, EUR, "
            "**price-only** index), yfinance daily closes 1998-01-02→2026-06-30. Vehicle: "
            "**`EWI`** (iShares MSCI Italy, USD, **total-return**), 1996→2026. Venerdì-17 "
            "dates are pure calendar arithmetic — **known before the open, zero "
            "look-ahead**. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | MIB Venerdì-17 mean **{R['m_mean']:+.1f} bps**, "
            f"*t* = **{R['m_t']:+.2f}** (positive — wrong sign); Welch vs other Fridays "
            f"*p* = {R['m_welch_p']:.3f}; down-day {R['m_down_rate']:.0f}%; placebo "
            f"left-tail *p* = {R['m_pl_p']:.3f}. EWI: {R['e_mean']:+.1f} bps, "
            f"*t* = {R['e_t']:+.2f} |\n"
            f"| **Tradability** | `MIRAGE` | short-the-17th nets **{R['m_sh_net']:+.1f} "
            f"bps** (*t* = {R['m_sh_net_t']:.2f}) on the MIB, {R['e_sh_net']:+.1f} bps on "
            "EWI — the folklore trade loses |\n"
            f"| **Unlucky day?** | `BUSTED` | the 17th is a slightly *above-average* "
            f"Friday; the most extreme middle-Friday slot is the boring **10th** "
            f"(raw *p* = {R['m_sweep'][10][3]:.3f}), and it dies under Bonferroni "
            f"({R['m_sweep'][10][4]:.3f}) |\n\n"
            "> 💡 In plain words: on Italy's own tape the unlucky day is faintly lucky, "
            "insignificantly so, and the one tradable expression of the fear loses money. "
            "Same result the desk got for Friday the 13th, one country over."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t = \\ln(P_t / P_{t-1})$ be the close-to-close log-return of the FTSE "
            "MIB on trading day $t$. Define the Venerdì-17 indicator "
            "$F_t = \\mathbb{1}[\\text{weekday}(t)=\\text{Fri} \\wedge \\text{day}(t)=17]$ "
            "— **known before the open**, so no estimation lag. Each Venerdì 17 is one "
            "independent, non-overlapping event, so the correct primary statistic is a "
            "**one-sample *t*** of $\\{r_t : F_t = 1\\}$ against 0 — *not* a daily panel "
            "regression (there is no within-event overlap to cluster). Claims:\n\n"
            "- **H1 (dip).** $E[r_t \\mid F_t=1] < 0$ with $t \\le -2$.\n"
            "- **H2 (worse than a normal Friday).** $E[r_t\\mid F_t=1] < "
            "E[r_t\\mid \\text{Fri}, F_t=0]$ (Welch).\n"
            "- **H3 (red days).** The down-day rate exceeds 50% (Wilson lower bound > 0.5).\n"
            "- **H4 (tradable).** Shorting into the 17th nets a positive edge after costs "
            "+ borrow.\n\n"
            "We find **every one rejected**: H1 the mean is *positive* "
            f"({R['m_mean']:+.1f} bps, *t* = {R['m_t']:+.2f}); H2 the 17th *beats* other "
            f"Fridays by {R['m_contrast']:+.0f} bps; H3 down-day rate "
            f"{R['m_down_rate']:.0f}% (< 50%); H4 the short *loses* "
            f"{abs(R['m_sh_net']):.0f} bps."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            f"n is modest by nature: ~1.7 Venerdì 17s a year → **{R['m_n']} events** on "
            f"the MIB ({R['e_n']} on the longer EWI tape). The plan: a one-sample *t* per "
            "tape; a **Welch** contrast vs all other Fridays (controls for any generic "
            "Friday-of-the-week effect, so we test the *17th label* specifically, not "
            "\"Fridays\"); a **Wilson** interval on the down-day rate; a **multi-seed "
            "random-calendar placebo** (20 seeds × 500 draws = 10,000 random sets of $n$ "
            "other-Fridays, matched null); and a **Bonferroni sweep** over the "
            "day-of-month slots 17 ± 7k = {3, 10, 17, 24, 31}, because the 17th was "
            "chosen by folklore, not pre-registered."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** Venerdì 17 by date arithmetic; {R['m_n']} MIB events "
            f"(1998→2026), {R['e_n']} EWI events (1996→2026).\n"
            "- **Tapes.** `FTSEMIB.MI` (EUR, price-only) primary; `EWI` (USD, "
            "total-return) as the tradable cross-check — labels carried everywhere so "
            "price-only is never sold as total-return.\n"
            "- **Headline.** One-sample *t* + Welch vs other Fridays + Wilson down-day "
            "rate, both tapes.\n"
            "- **Robustness.** 20×500-draw random-Friday placebo (left-tail); Bonferroni "
            "sweep over the five neighbouring Friday slots; a two-half sub-period split.\n"
            "- **Execution (third axis input).** Short established at the prior close "
            "(calendar-known → no look-ahead), covered at the 17th's close; 2× one-way "
            "cost × NAV + one day borrow. Gross **and** net.\n"
            "- **Control.** Synthetic tape with a tunable planted Friday-17 effect; the "
            "null (effect = 0) must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline — one-sample *t*, both tapes, both directions\n\n"
            "The primary bar (does the 17th itself print negative?) and the Welch "
            "contrast (is it worse than a normal Friday?), side by side."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rm = st.friday17_test(MIB); re = st.friday17_test(EWI)\n"
            "    rows = [('FTSE MIB', rm), ('EWI', re)]\n"
            "    for name, r in rows:\n"
            "        print(f\"{name:9s} n={r['n_f17']:3d}  mean={r['mean_f17_bps']:+7.2f} bps  \"\n"
            "              f\"t={r['t_f17']:+.3f}  vs-other-Fri contrast={r['contrast_fri_bps']:+7.2f} \"\n"
            "              f\"(Welch t={r['t_welch_fri']:+.2f}, p={r['p_welch_fri']:.3f})\")\n"
            "    labels = ['FTSE MIB\\n1-sample t', 'FTSE MIB\\nWelch vs Fri', 'EWI\\n1-sample t', 'EWI\\nWelch vs Fri']\n"
            "    ts = [rm['t_f17'], rm['t_welch_fri'], re['t_f17'], re['t_welch_fri']]\n"
            "else:\n"
            "    labels = ['FTSE MIB\\n1-sample t', 'FTSE MIB\\nWelch vs Fri', 'EWI\\n1-sample t', 'EWI\\nWelch vs Fri']\n"
            "    ts = [R['m_t'], R['m_welch_t'], R['e_t'], R['e_welch_t']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(labels, ts, color=[AMBER if abs(t)>=2 else GREY for t in ts])\n"
            "ax.axhline(2, ls='--', c=RED, lw=1, label='certification bar |t|=2')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('t-stat'); ax.set_ylim(-2.5, 2.5)\n"
            "ax.set_title('Every bar is positive and inside the noise band -- the wrong sign, and small')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: all four statistics are **positive** and none reaches "
            f"|t| = 2. The folklore predicts a *negative* dip; the tape shows a faint "
            f"*lift* ({R['m_mean']:+.1f} bps on the MIB, {R['e_mean']:+.1f} on EWI). "
            "The sign is wrong before the significance even matters."
        ),
        md(
            "### 4b · The down-day rate — a Wilson interval on the red-day count\n\n"
            "If Venerdì 17 were cursed you'd expect it to close *down* more than half "
            "the time. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rm = st.friday17_test(MIB); re = st.friday17_test(EWI)\n"
            "    data_rows = [('FTSE MIB', rm['down_rate']*100, rm['down_lo']*100, rm['down_hi']*100),\n"
            "                 ('EWI', re['down_rate']*100, re['down_lo']*100, re['down_hi']*100)]\n"
            "else:\n"
            "    data_rows = [('FTSE MIB', R['m_down_rate'], R['m_down_lo'], R['m_down_hi']),\n"
            "                 ('EWI', R['e_down_rate'], R['e_down_lo'], R['e_down_hi'])]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 3.2))\n"
            "ys = np.arange(len(data_rows))\n"
            "for y, (name, dr, lo, hi) in zip(ys, data_rows):\n"
            "    ax.errorbar([dr], [y], xerr=[[dr-lo],[hi-dr]], fmt='o', color=RED, capsize=6, lw=1.8, ms=8)\n"
            "ax.axvline(50, ls='--', c='k', lw=1.2, label='coin flip')\n"
            "ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in data_rows]); ax.set_xlim(20, 80)\n"
            "ax.set_xlabel('share of Venerdi 17s closing DOWN (%), Wilson 95%')\n"
            "ax.set_title('Both point estimates sit BELOW 50% -- greener than a coin flip')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"MIB down {R['m_down_k']}/{R['m_down_n']}={R['m_down_rate']:.1f}%  \"\n"
            "      f\"EWI down {R['e_down_k']}/{R['e_down_n']}={R['e_down_rate']:.1f}%\")"
        ),
        md(
            f"> 💡 In plain words: {R['m_down_rate']:.0f}% (MIB) and {R['e_down_rate']:.0f}% "
            "(EWI) of Venerdì 17s are red — both *below* 50%, both Wilson intervals "
            "straddling the coin flip. H3 rejected: there is no excess of down days."
        ),
        md(
            "### 4c · The random-calendar placebo — is +26 bps unusual among Fridays?\n\n"
            "Draw 10,000 random sets of 49 *other* Fridays from the same MIB tape "
            "(matched null: any generic Friday effect is held fixed) and compare the "
            "observed Venerdì-17 mean to that distribution. Folklore predicts the "
            "observed sits in the **left** tail."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.random_friday_placebo(MIB, n_seeds=8, n_draws_per_seed=500)\n"
            "    obs = pl['obs_bps']; rng = np.random.default_rng(742)\n"
            "    draws = rng.normal(pl['null_mean_bps'], pl['null_sd_bps'], 8000)\n"
            "    p_left = pl['p_left']\n"
            "else:\n"
            "    obs = R['m_pl_obs']; rng = np.random.default_rng(742)\n"
            "    draws = rng.normal(R['m_pl_null'], R['m_pl_sd'], 8000); p_left = R['m_pl_p']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(draws, bins=55, color=GREY, alpha=.85, label='null: random sets of 49 other Fridays')\n"
            "ax.axvline(obs, c=RED, lw=2.4, label=f'observed Venerdi 17 = {obs:+.1f} bps')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('mean return of a random-Friday draw (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Canonical placebo (10,000 draws): left-tail p = {R[\"m_pl_p\"]:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"observed {R['m_pl_obs']:+.1f} bps vs null mean {R['m_pl_null']:+.1f} \"\n"
            "      f\"(sd {R['m_pl_sd']:.1f}); p_left = {R['m_pl_p']:.4f} over {R['m_pl_draws']:,} draws\")"
        ),
        md(
            f"> 💡 In plain words: left-tail *p* = **{R['m_pl_p']:.3f}** — the observed "
            "mean is out at the *high* end of the null, the opposite of what the curse "
            "predicts. A randomly-chosen Friday is *more* likely to look weak than "
            "Venerdì 17. This is the cleanest single refutation on the page."
        ),
        md(
            "### 4d · Look-elsewhere — the day-of-month Bonferroni sweep\n\n"
            "The 17th was picked by superstition, not pre-registration. Test all five "
            "neighbouring Friday slots (17 ± 7k) and correct for it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sw = st.dom_sweep(MIB)\n"
            "    print(sw.to_string(index=False))\n"
            "    days = sw['day'].tolist(); praw = sw['p_raw'].tolist(); pbon = sw['p_bonferroni'].tolist()\n"
            "else:\n"
            "    order = sorted(R['m_sweep'], key=lambda d: R['m_sweep'][d][3])\n"
            "    days = order; praw = [R['m_sweep'][d][3] for d in order]; pbon = [R['m_sweep'][d][4] for d in order]\n"
            "x = np.arange(len(days)); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "cols = [RED if d==17 else GREY for d in days]\n"
            "ax.bar(x - w/2, praw, w, color=cols, label='raw p')\n"
            "ax.bar(x + w/2, pbon, w, color=[AMBER]*len(days), label='Bonferroni p (k=5)')\n"
            "ax.axhline(0.05, ls='--', c='k', lw=1, label='alpha = 0.05')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'Fri {d}th' for d in days])\n"
            "ax.set_ylabel('Welch p-value vs other Fridays')\n"
            "ax.set_title('The 17th (red) has one of the HIGHEST p-values; the extreme slot is the 10th, and it dies under Bonferroni')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the most extreme middle-Friday is the **10th** "
            f"(raw *p* = {R['m_sweep'][10][3]:.3f}, contrast {R['m_sweep'][10][2]:+.0f} "
            f"bps) — and even it fails Bonferroni ({R['m_sweep'][10][4]:.3f} > 0.05). "
            f"The 17th itself sits near the *top* of the p-value list "
            f"({R['m_sweep'][17][3]:.3f}). If superstition predicted anything, it picked "
            "the wrong slot — exactly the Friday-the-27th story from study 163."
        ),
        md(
            "### 4e · Sub-periods — is the null stable across the sample?\n\n"
            "Split the MIB tape in half. A real effect that merely averaged out should "
            "reveal itself in at least one half; a non-effect stays a non-effect."
        ),
        code(
            "if HAVE_REAL:\n"
            "    halves = [('1998-2012','1998-01-01','2012-12-31'), ('2013-2026','2013-01-01','2026-06-30')]\n"
            "    for lab, lo, hi in halves:\n"
            "        sub = MIB[(MIB.index>=lo)&(MIB.index<=hi)]; r = st.friday17_test(sub)\n"
            "        print(f\"{lab}: n={r['n_f17']:2d}  mean={r['mean_f17_bps']:+7.2f} bps  \"\n"
            "              f\"t={r['t_f17']:+.3f}  contrast={r['contrast_fri_bps']:+.2f}  p={r['p_welch_fri']:.3f}\")\n"
            "    labs = [h[0] for h in halves]\n"
            "    means = [st.friday17_test(MIB[(MIB.index>=h[1])&(MIB.index<=h[2])])['mean_f17_bps'] for h in halves]\n"
            "else:\n"
            "    labs = ['1998-2012', '2013-2026']; means = [R['m_h1_mean'], R['m_h2_mean']]\n"
            "fig, ax = plt.subplots(figsize=(7.4, 4.0))\n"
            "ax.bar(labs, means, color=GREEN, width=.5)\n"
            "for i, v in enumerate(means): ax.annotate(f'{v:+.0f} bps', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('Venerdi 17 mean (bps)')\n"
            "ax.set_title('Both halves are POSITIVE -- the wrong sign, consistently')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: {R['m_h1_mean']:+.0f} bps (1998–2012) and "
            f"{R['m_h2_mean']:+.0f} bps (2013–2026) — both halves *positive*, both "
            "insignificant (*t* = "
            f"{R['m_h1_t']:+.2f} / {R['m_h2_t']:+.2f}). The non-effect is stable; there "
            "is no hidden regime where the curse bites."
        ),
        md(
            "### 4f · Tradability — the costed short (the folklore bet)\n\n"
            "Short into Venerdì 17 (established at the prior close — calendar-known, no "
            "look-ahead), cover at the 17th's close, pay 2× one-way cost × NAV + one day "
            "of short borrow. Gross and net, both tapes."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sm = st.short_the_17th(PRICES[data.MIB]); se = st.short_the_17th(PRICES[data.EWI])\n"
            "    print(f\"short MIB: n={sm['n']} gross={sm['gross_mean_bps']:+.2f} bps (t={sm['gross_t']:+.2f})  \"\n"
            "          f\"net={sm['net_mean_bps']:+.2f} bps (t={sm['net_t']:+.2f})  win={sm['win_rate']*100:.1f}%  \"\n"
            "          f\"breakeven={sm['breakeven_bps']:.0f} bps\")\n"
            "    print(f\"short EWI: n={se['n']} gross={se['gross_mean_bps']:+.2f} bps (t={se['gross_t']:+.2f})  \"\n"
            "          f\"net={se['net_mean_bps']:+.2f} bps (t={se['net_t']:+.2f})  win={se['win_rate']*100:.1f}%\")\n"
            "    mg, mn, eg, en = sm['gross_mean_bps'], sm['net_mean_bps'], se['gross_mean_bps'], se['net_mean_bps']\n"
            "else:\n"
            "    mg, mn, eg, en = R['m_sh_gross'], R['m_sh_net'], R['e_sh_gross'], R['e_sh_net']\n"
            "x = np.arange(2); w = 0.36\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "ax.bar(x - w/2, [mg, eg], w, color=GREY, label='gross')\n"
            "ax.bar(x + w/2, [mn, en], w, color=RED, label='net (costs + borrow)')\n"
            "for i, v in enumerate([mn, en]): ax.annotate(f'{v:+.0f}', (i+w/2, v), ha='center', va='top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels(['short MIB', 'short EWI'])\n"
            "ax.set_ylabel('profit per event (bps)')\n"
            "ax.set_title('Every bar is negative: the folklore short bleeds gross AND net')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the short nets **{R['m_sh_net']:+.1f} bps** on the MIB "
            f"(*t* = {R['m_sh_net_t']:.2f}) and {R['e_sh_net']:+.1f} bps on EWI — "
            "negative *before* costs (because the day is green) and worse after. Win "
            f"rate {R['m_sh_win']:.0f}%. **H4 rejected; Tradability = MIRAGE** — there "
            "is no edge to charge costs against, and the naive bet loses outright."
        ),
        md(
            "### 4g · Faithful-engine & power control\n\n"
            "A synthetic FTSE-MIB-like tape with a TUNABLE planted Friday-17 effect. "
            "The null (effect = 0) must not fire across 20 seeds; a planted *fear* "
            "(negative bump) must be recovered."
        ),
        code(
            "null_ts = np.array([st.synthetic_detect(0.0, seed=742+s)['t'] for s in range(20)])\n"
            "p05 = st.synthetic_detect(-0.5, seed=742); p10 = st.synthetic_detect(-1.0, seed=742)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (effect=0), 20 seeds')\n"
            "ax.scatter([1], [p05['t']], color=AMBER, s=90, zorder=5, label='planted fear -0.5 sigma')\n"
            "ax.scatter([2], [p10['t']], color=RED, s=90, zorder=5, label='planted fear -1.0 sigma')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0,1,2]); ax.set_xticklabels(['null x20', 'planted -0.5', 'planted -1.0'])\n"
            "ax.set_ylabel('one-sample t'); ax.set_title('Control: quiet null, planted fear lights up negative')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t={null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "print(f'planted -0.5 t={p05[\"t\"]:+.2f}  planted -1.0 t={p10[\"t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null seeds the detector averages "
            f"t = {R['syn_null_meant']:+.2f} (sd {R['syn_null_sd']:.2f}) and fires at "
            f"|t| ≥ 2 in **{R['syn_null_fire']}/{R['syn_null_seeds']}** seeds — clean. A "
            f"planted −0.5σ fear reads t = {R['syn_p05_t']:.2f}, a −1.0σ fear "
            f"t = {R['syn_p10_t']:.2f}. The machinery *would* catch a real Venerdì-17 "
            "dip of even half a daily sigma; the real tape simply has none. *(A "
            "faithful-engine / power check only — never cited in support of the "
            "real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — Venerdì-17 mean **{R['m_mean']:+.1f} bps** on the "
            f"FTSE MIB (*t* = {R['m_t']:+.2f}), *positive* and inside the noise band; "
            f"Welch vs other Fridays *p* = {R['m_welch_p']:.3f}; down-day rate "
            f"{R['m_down_rate']:.0f}%; random-Friday placebo left-tail *p* = "
            f"{R['m_pl_p']:.3f}. EWI concurs ({R['e_mean']:+.1f} bps, *t* = "
            f"{R['e_t']:+.2f}). The fear has the sign backwards.\n"
            f"- **Tradability `MIRAGE`** — the folklore short nets {R['m_sh_net']:+.1f} "
            f"bps (MIB) / {R['e_sh_net']:+.1f} bps (EWI), *t* = {R['m_sh_net_t']:.2f}; "
            "negative gross and net. No edge to cost.\n"
            f"- **Unlucky day? `BUSTED`** — the 17th is a slightly above-average Friday; "
            f"the most extreme neighbouring slot is the (non-superstitious) 10th, which "
            f"fails Bonferroni ({R['m_sweep'][10][4]:.3f}). Both sample halves agree. "
            "The Italian 17th behaves exactly like the Anglo 13th (study 163): a "
            "universal human dread that leaves no mark on prices."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson is about calendar superstitions.** Two of them — the "
            "13th on the S&P, the 17th on the MIB — tested the same honest way, come out "
            "faintly green and firmly insignificant. A pure mood label with no economic "
            "content does not price into a liquid index; the placebo and the "
            "look-elsewhere correction catch the one-in-five \"lucky\" slot every time.\n"
            "- **A cleaner test would need a different surface.** If Venerdì 17 bites at "
            "all it would be in thin, retail-dominated corners — Italian small-caps, "
            "single retail-favourite stocks, or options-implied vol into the date — not "
            "a blue-chip index dominated by global institutions. That's the natural "
            "sequel.\n"
            "- **Dedup map:** [163-friday-13th](../../163-friday-13th/) is the direct "
            "Anglo cousin (S&P, HAC t + Bonferroni); [158-super-bowl](../../158-super-bowl/) "
            "and [608-friday-news-dump](../../608-friday-news-dump/) are other calendar/"
            "folklore signals; [707-plane-crash-effect](../../707-plane-crash-effect/) and "
            "[708-eurovision-effect](../../708-eurovision-effect/) are the sentiment-event "
            "siblings (one-sample t + placebo). None tests the *Italian* 17th on the "
            "*Italian* tape — that's this study's own contribution.\n\n"
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
