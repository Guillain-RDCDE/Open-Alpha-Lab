"""Generate the two narrative notebooks for Study 608 (Friday News Dump).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached EDGAR
panel + prices under ../_cache/ when present and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic machinery control runs
anywhere.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (3,122 negative 8-Ks
# + 696 earnings-control, 2004-08-23 -> 2026-06-30, as-of 2026-06-30).
R = dict(
    as_of="2026-06-30", n_filings=3818, n_bad=3122, n_ctrl=696,
    n_by_class=dict(ceo_exit=1051, nonreliance=1044, impairment=1027),
    n_mapped=1234, n_car=1027, first="2004-08-27", last="2026-06-15",
    fp_events="e566cab9c587", fp_spy="c9b5637c9a7f", n_sessions=5658,
    main=dict(
        n_fri=210, n_oth=817,
        ar0_fri=65.4, ar0_oth=-70.4, ar0_t_fri=0.83, ar0_t_oth=-1.79,
        ar0_gap_t=1.54,
        drift_fri=-45.6, drift_oth=-56.3, drift_t_fri=-0.40, drift_t_oth=-1.06,
        gap=10.7, gap_t=0.08, gap_t_wins=0.02, p_perm=0.523,
        pooled=-54.1, pooled_t=-1.12, pooled_t_wins=-1.45, car010_gap_t=1.07,
        pm_n=130, pm_drift=-2.2, pm_drift_t=-0.02, pm_gap=59.5, pm_gap_t=0.39,
    ),
    # (label, n, n_fri, gap_bps, gap_t, pooled_bps, pooled_t)
    robust=[
        ("nonreliance", 254, 61, 247.6, 0.82, -6.9, -0.05),
        ("impairment", 526, 91, 109.0, 0.88, -48.7, -1.26),
        ("ceo_exit", 247, 58, -400.6, -1.39, -114.2, -0.88),
        ("2004-2015", 344, 62, -63.7, -0.41, -11.2, -0.17),
        ("2016-2026", 683, 148, 48.0, 0.28, -75.8, -1.17),
    ],
    shares=dict(
        friday=(21.3, 7.9, 10.67), after_close=(57.9, 47.6, 4.94),
        friday_pm=(12.8, 1.4, 15.15),
        wd_bad=[19.7, 19.6, 19.4, 20.0, 21.3],
        wd_ctrl=[12.6, 22.7, 29.0, 27.7, 7.9],
        friday_null=20.3, friday_vs_null_z=1.40,
    ),
    # (label, n, per_year, gross_bps, t_gross, hit, cost_bps, net_bps, t_net)
    trades=[
        ("Friday filers, borrow 5%/yr", 210, 9.6, 45.6, 0.40, 0.56, 41.8, 3.8, 0.03),
        ("Friday filers, borrow 2%/yr", 210, 9.6, 45.6, 0.40, 0.56, 29.9, 15.7, 0.14),
        ("Friday-PM only, borrow 5%/yr", 130, 6.0, 2.2, 0.02, 0.60, 41.8, -39.6, -0.28),
        ("Friday-PM only, borrow 2%/yr", 130, 6.0, 2.2, 0.02, 0.60, 29.9, -27.7, -0.19),
    ],
    path_fri=[65.4, 28.1, 45.2, 17.0, 64.9, 111.0, 80.7, 57.7, 42.5, 23.8, 19.8],
    path_oth=[-70.4, -106.1, -100.7, -136.1, -109.1, -133.7, -126.7, -145.5,
              -132.9, -137.4, -126.7],
    # (label, friday_share_pct, gap_bps, gap_t, placebo_p)
    syn=[("null", 21.5, -26.2, -0.38, 0.340),
         ("drift -300 bps Fri", 21.5, -326.2, -4.69, 0.000),
         ("hide p_fri=0.35", 37.1, -22.3, -0.38, 0.364)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Friday_night_dump%3F: Mixed](https://img.shields.io/badge/Friday_night_dump%3F-Mixed-8b949e?style=flat-square)\n\n"
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

from friday_news_dump import data, strategy as st

AS_OF = "2026-06-30"
HAVE_REAL = data.have_real()
print("real cache present:", HAVE_REAL,
      "(offline CI runs quote the frozen numbers in R — mirror of docs/results.md)")
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"

WD = "Mon Tue Wed Thu Fri".split()


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    m = R["main"]
    s = R["shares"]
    cells = [
        md(
            "# The Friday news dump 🗑️\n"
            "### Do companies bury bad news on Friday night — and does the stock keep "
            "sliding while nobody looks?\n\n"
            + BADGES +
            "Every PR handbook knows the trick: got something ugly to disclose? File it "
            "**Friday after the market closes**. Reporters are gone, investors are at "
            "dinner, and by Monday it's old news. Finance has a famous paper behind the "
            "folklore — DellaVigna & Pollet (2009) found that Friday earnings news gets "
            "a weaker immediate reaction and a bigger *delayed* one: the market "
            "underreacts, then drifts.\n\n"
            "We put both halves under the microscope with the SEC's own filing logs: "
            f"**{R['n_bad']:,} unambiguously-bad 8-K filings** (accounting restatements, "
            "asset impairments, CEO exits) timestamped **to the second**, 2004-2026, plus "
            f"{R['n_ctrl']} routine earnings filings as the control.\n\n"
            "> 📓 **Plain-language layer.** Want the Welch *t*'s, the permutation placebo "
            "and the cost stack? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do companies time their bad news? | **Yes — but into the *evening*, not "
            f"into Friday.** {s['after_close'][0]:.0f}% of bad 8-Ks land after the 4pm "
            f"close (routine earnings: {s['after_close'][1]:.0f}%). |\n"
            "| Is Friday night really the dump ground? | **Only compared to other news.** "
            f"Bad news hits Friday-after-close {s['friday_pm'][0]:.1f}% of the time vs "
            f"{s['friday_pm'][1]:.1f}% for earnings — a 9× gap. But bad news is spread "
            "almost perfectly flat across the week "
            f"(Friday {s['friday'][0]:.1f}% vs a {s['friday_null']:.1f}% coin-flip "
            "baseline). Friday only *looks* special because ordinary news avoids it. |\n"
            "| Does Friday-filed bad news drift down while nobody looks? | **No.** Over "
            "the next ten sessions Friday filers slid "
            f"{m['drift_fri']:.0f} bps vs {m['drift_oth']:.0f} bps for Mon-Thu filers — "
            "statistically identical (the gap is 11 bps of noise, *t* ≈ 0.1). |\n"
            "| Can you short the Friday dump and get paid? | **No.** After realistic "
            "costs and borrow the trade makes ~4 bps per event (*t* = 0.03), ten times "
            "a year. |\n\n"
            "> The *hiding* is real — companies really do slip bad news out after hours. "
            "The *market-falls-for-it* half is what fails: by the next close the price "
            "has moved, and there is no extra slide left to harvest."
        ),

        md(
            "## 1 · The claim\n\n"
            "> *\"File it Friday after the close. The market underreacts to news nobody "
            "reads, so the stock drifts down for weeks — short the dump.\"*\n\n"
            "It is plausible folklore with an academic pedigree: limited attention is one "
            "of the best-documented facts in behavioural finance, and DellaVigna-Pollet "
            "measured exactly this pattern on Friday *earnings* announcements (1995-2006 "
            "data). We test the sharper version: not earnings that might be good or bad, "
            "but 8-K classes that are **bad by construction** —\n\n"
            f"- **Item 4.02** ({R['n_by_class']['nonreliance']:,} filings): \"don't rely "
            "on our previous accounts\" — a restatement is coming;\n"
            f"- **Item 2.06** ({R['n_by_class']['impairment']:,}): material impairments "
            "— assets just lost value;\n"
            f"- **Item 5.02 CEO exits** ({R['n_by_class']['ceo_exit']:,}): the boss "
            "resigned.\n\n"
            "Each filing carries the EDGAR **acceptance timestamp** — the second it became "
            "public — so \"Friday after the close\" is measured exactly, not guessed."
        ),

        md(
            "## 2 · The hiding — real, but it's about the *clock*, not the *calendar*\n\n"
            "First the disclosure-timing question on the **full** panel of "
            f"{R['n_bad']:,} bad filings (no survivorship here — dead companies' filings "
            "count too). If firms dumped bad news on Friday, the Friday bar below should "
            "tower over the rest of the week."
        ),
        code(
            "s = R['shares']\n"
            "x = np.arange(5); w = 0.38\n"
            "fig, ax = plt.subplots()\n"
            "ax.bar(x - w/2, s['wd_bad'], w, color=RED, label='bad news (4.02 / 2.06 / CEO exit)')\n"
            "ax.bar(x + w/2, s['wd_ctrl'], w, color=GREY, label='routine earnings (Item 2.02)')\n"
            "ax.axhline(s['friday_null'], ls='--', c='k', lw=1,\n"
            "           label=f\"uniform week ({s['friday_null']:.1f}%)\")\n"
            "ax.set_xticks(x, ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])\n"
            "ax.set_ylabel('share of filings (%)')\n"
            "ax.set_title('When the news lands: weekday mix, 2004-2026 (full panel, no survivorship)')\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"bad news on Friday          : {s['friday'][0]:.1f}%  (uniform null \"\n"
            "      f\"{s['friday_null']:.1f}% -> z = {s['friday_vs_null_z']:+.2f}, not significant)\")\n"
            "print(f\"bad news after the close    : {s['after_close'][0]:.1f}% vs earnings \"\n"
            "      f\"{s['after_close'][1]:.1f}%  (z = {s['after_close'][2]:+.2f})\")\n"
            "print(f\"bad news Friday-after-close : {s['friday_pm'][0]:.1f}% vs earnings \"\n"
            "      f\"{s['friday_pm'][1]:.1f}%  (z = {s['friday_pm'][2]:+.2f})\")\n"
        ),
        md(
            "The red bars are almost **flat** — bad news is *not* piled onto Friday. The "
            "grey bars explain the legend anyway: routine earnings **avoid** Friday "
            f"(only {s['friday'][1]:.1f}%), so *relative to what you're used to reading*, "
            "Friday is disproportionately grim. And the genuine hiding shows up on the "
            f"clock, not the calendar: **{s['after_close'][0]:.0f}% of bad 8-Ks land "
            "after 4pm** — the Friday-evening *combination* is 9× more common for bad "
            "news than for earnings.\n\n"
            "> 🔬 **For the quants.** Friday share of bad news vs the uniform-calendar "
            f"null: binomial z = +{s['friday_vs_null_z']:.2f}. Vs the earnings control: "
            f"two-proportion z = +{s['friday'][2]:.2f}. Both are true at once — that "
            "contrast IS the finding."
        ),

        md(
            "## 3 · The drift — does the market fall for it?\n\n"
            "Now the tradable half. For every bad filing we map its timestamp to the "
            "first session whose **close** reflects the news (a Friday-9pm filing lands "
            "on Monday), subtract the market's move (SPY), and watch ten sessions. "
            "DellaVigna-Pollet predicts the **red line** (Friday filers) should sink "
            "*further* than the grey one as the ignored news soaks in."
        ),
        code(
            "m = R['main']\n"
            "t = np.arange(11)\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(t, R['path_fri'], marker='o', color=RED,\n"
            "        label=f\"Friday filers (n={m['n_fri']})\")\n"
            "ax.plot(t, R['path_oth'], marker='o', color=GREY,\n"
            "        label=f\"Mon-Thu filers (n={m['n_oth']})\")\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axvspan(0.5, 10.5, alpha=0.06, color=RED)\n"
            "ax.text(5.5, ax.get_ylim()[1]*0.9, 'the would-be drift window (+1..+10)',\n"
            "        ha='center', fontsize=9, color=RED)\n"
            "ax.set_xlabel('sessions since the news was public (day 0 = first close after filing)')\n"
            "ax.set_ylabel('mean cumulative abnormal return (bps vs SPY)')\n"
            "ax.set_title('Bad news, then... nothing much: survivor tape, 1,027 events 2004-2026')\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"drift day +1..+10: Friday {m['drift_fri']:+.1f} bps vs Mon-Thu \"\n"
            "      f\"{m['drift_oth']:+.1f} bps -> gap {m['gap']:+.1f} bps, Welch t = {m['gap_t']:+.2f}\")\n"
            "print(f\"label-permutation placebo p = {m['p_perm']:.3f}  (a fair coin does this half the time)\")\n"
        ),
        md(
            "Both groups wobble around **flat**. The Friday line ends the window a whole "
            f"**{m['gap']:.0f} bps** *above* the weekday line — the wrong direction for "
            "the story, and pure noise (*t* ≈ 0.1). Splitting by news class or by decade "
            "gives signs both ways, all insignificant.\n\n"
            "**One honest caveat, in the open:** our price tape only covers companies "
            "that still exist — just "
            f"{R['n_mapped']:,} of {R['n_bad']:,} filers map to a live ticker. The firms "
            "that restated and *died* aren't here, so genuine bad-news drift is "
            "understated. That censoring hits Friday and Monday filers alike, though — "
            "and the Friday-vs-weekday **gap**, which is what the folklore claims, is "
            "absent on the tape we can see.\n\n"
            "> 🔬 **For the quants.** Pooled drift across all bad news: "
            f"{m['pooled']:+.1f} bps/10d, *t* = {m['pooled_t']:.2f} — right sign, below "
            "the bar. The full inference table is in notebook 02."
        ),

        md(
            "## 4 · \"Short the dump\" as a trade\n\n"
            "Suppose you shorted every Friday-filed bad name at the first close after the "
            "news (never catching the day-0 hit — that's gone by the time you can trade), "
            "hedged with SPY, covered ten sessions later:\n\n"
            "| rule | trades/yr | gross/event | net/event (costs + borrow) |\n|---|--:|--:|--:|\n"
            + "\n".join(
                f"| {lab} | ~{py:.0f} | {g:+.1f} bps (*t* = {tg:+.2f}) | "
                f"**{n:+.1f} bps** (*t* = {tn:+.2f}) |"
                for lab, _, py, g, tg, _, _, n, tn in R["trades"][:2]
            ) +
            "\n\nTen trades a year, in exactly the small caps where borrow is expensive "
            "and spreads are wide, for a net edge statistically indistinguishable from "
            "zero. The Friday-*evening*-only variant is net **negative**. There is no "
            "trade here."
        ),

        md(
            "## 5 · Verdict\n\n"
            "| Axis | Stamp |\n|---|---|\n"
            "| Signal — Friday-filed bad news drifts more | **NONE** — gap "
            f"*t* = {m['gap_t']:+.2f}, placebo p = {m['p_perm']:.2f} (survivor tape, named) |\n"
            "| Tradability — short the dump | **MIRAGE** — +4 bps/event net, *t* = 0.03 |\n"
            "| Myth-check — firms dump bad news Friday night | **MIXED** — the dump is "
            "real but it's an *after-hours* dump; the Friday tilt is folklore born of "
            "everything-else avoiding Friday |\n\n"
            "The PR departments are guilty as charged — they really do wait for the "
            "closing bell. The *market*, though, doesn't stay fooled long enough to pay "
            "you: by the next close the damage is priced, Friday or not.\n\n"
            "---\n"
            f"*As-of {R['as_of']}; panel fingerprint `{R['fp_events']}`, SPY tape "
            f"`{R['fp_spy']}`. Reproduce with `python examples/verify.py`. Sources & "
            "method: [docs/references.md](../docs/references.md) · "
            "[docs/results.md](../docs/results.md). Research & education, not investment "
            "advice.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    nbf.write(nb, os.path.join(HERE, "01_for_the_curious.ipynb"))


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    m = R["main"]
    s = R["shares"]
    cells = [
        md(
            "# Friday News Dump — the quant teardown 🗑️📉\n\n"
            + BADGES +
            "**Claim (DellaVigna-Pollet 2009, transposed to unambiguous bad news).** "
            "Negative corporate disclosures filed on Friday — especially after the close "
            "— receive attention-limited underreaction and exhibit stronger negative "
            "post-event drift than identical weekday disclosures.\n\n"
            "**Design.** Stratified panel of negative 8-Ks (Item 4.02 non-reliance, "
            "Item 2.06 impairments, Item 5.02 CEO resignations) from the EDGAR full-text "
            "search API, 2004-08-23 → 2026-06-30, acceptance timestamps to the second; "
            "Item 2.02 earnings 8-Ks as the timing control. Market-adjusted CARs vs SPY; "
            "Welch *t* on the Friday-vs-Mon-Thu drift gap; label-permutation placebo; "
            "penny-stock screen at $1. ONE execution convention: day 0 = first session "
            "whose close reflects the filing.\n\n"
            "> 💡 **In plain words:** we check whether the market really sleeps through "
            "Friday-night bad news (it doesn't), and separately whether companies really "
            "try the trick (they do — but after hours, not specifically on Friday)."
        ),
        code(BOOT_CELL),

        md(
            "## 0 · Data stamp\n\n"
            f"| Field | Value |\n|---|---|\n"
            f"| Filing panel | {R['n_filings']:,} 8-Ks — {R['n_bad']:,} negative "
            f"(ceo_exit {R['n_by_class']['ceo_exit']:,} · nonreliance "
            f"{R['n_by_class']['nonreliance']:,} · impairment "
            f"{R['n_by_class']['impairment']:,}) + {R['n_ctrl']} earnings-control |\n"
            "| Sampling | first 12 eligible FTS hits per class × quarter (8 control) — "
            "documented, weekday-orthogonal cap; pure `8-K` only, item codes verified |\n"
            f"| Return panel | {R['n_mapped']:,}/{R['n_bad']:,} filers map to a live "
            f"ticker (SEC current map); full −1..+10 window + ≥$1 screen → "
            f"**{R['n_car']:,} events** ({R['first']} → {R['last']}), "
            f"{m['n_fri']} Friday-filed |\n"
            f"| Fingerprints | events `{R['fp_events']}` · SPY `{R['fp_spy']}` "
            f"({R['n_sessions']:,} sessions) · as-of {R['as_of']} |\n"
            "| Survivorship | **SEVERE and named**: 60% of filers are gone from the "
            "current ticker map — drift magnitudes are understated; the timing shares "
            "(§3) use the full panel and carry no survivorship |\n\n"
            "> 💡 The survivorship censoring hits Friday and Mon-Thu filers alike, so the "
            "Friday-vs-weekday *gap* — the claim under test — remains a fair contrast on "
            "the visible tape; the *level* of drift is the biased quantity."
        ),
        code(
            "# live recompute (cache-first; CI without cache quotes R)\n"
            "if HAVE_REAL:\n"
            "    panel = data.load_panel(require_prices=True)\n"
            "    px = data.load_prices(asof=AS_OF)\n"
            "    kept, armat = st.event_ars(panel, px)\n"
            "    fri = kept['friday'].to_numpy()\n"
            "    g = st.group_stats(fri, armat)\n"
            "    print(f\"events {len(kept)} | Friday {g['n_fri']} | gap \"\n"
            "          f\"{g['gap_bps']:+.1f} bps Welch t={g['gap_t']:+.2f} | pooled drift \"\n"
            "          f\"{g['pooled_drift_bps']:+.1f} bps t={g['pooled_drift_t']:+.2f}\")\n"
            "    assert abs(g['gap_t'] - R['main']['gap_t']) < 0.05, 'drifted from results.md!'\n"
            "else:\n"
            "    print('cache absent -> quoting frozen R (mirror of docs/results.md)')\n"
        ),

        md(
            "## 1 · The main test — reaction and drift, Friday vs Mon-Thu\n\n"
            "| group | n | AR day 0 | drift CAR[+1..+10] | drift *t* |\n|---|--:|--:|--:|--:|\n"
            f"| Friday filers | {m['n_fri']} | {m['ar0_fri']:+.1f} bps "
            f"(*t* = {m['ar0_t_fri']:+.2f}) | {m['drift_fri']:+.1f} bps | "
            f"{m['drift_t_fri']:+.2f} |\n"
            f"| Mon-Thu filers | {m['n_oth']} | {m['ar0_oth']:+.1f} bps "
            f"(*t* = {m['ar0_t_oth']:+.2f}) | {m['drift_oth']:+.1f} bps | "
            f"{m['drift_t_oth']:+.2f} |\n\n"
            f"**Gap = {m['gap']:+.1f} bps, Welch *t* = {m['gap_t']:+.2f}** "
            f"(1%-winsorized *t* = {m['gap_t_wins']:+.2f}); permutation placebo "
            f"**p = {m['p_perm']:.3f}** (2,000 seeded draws). CAR[0..+10] gap Welch "
            f"*t* = {m['car010_gap_t']:+.2f} — Friday filers end *less* negative. "
            f"Friday-after-close subset (n = {m['pm_n']}): drift {m['pm_drift']:+.1f} bps "
            f"(*t* = {m['pm_drift_t']:+.2f}), gap vs rest {m['pm_gap']:+.1f} bps "
            f"(*t* = {m['pm_gap_t']:+.2f}).\n\n"
            f"Pooled drift, all bad news: **{m['pooled']:+.1f} bps/10d**, "
            f"*t* = {m['pooled_t']:.2f} (winsorized {m['pooled_t_wins']:.2f}) — the sign "
            "the literature predicts, below the *t* ≥ 2 bar on this survivor tape.\n\n"
            "> 💡 **In plain words:** whatever day the bad news lands, the next ten "
            "sessions look the same — a slow, statistically unremarkable bleed of about "
            "half a percent. No extra Friday slide."
        ),
        code(
            "t = np.arange(11)\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(t, R['path_fri'], marker='o', color=RED,\n"
            "        label=f\"Friday (n={R['main']['n_fri']})\")\n"
            "ax.plot(t, R['path_oth'], marker='o', color=GREY,\n"
            "        label=f\"Mon-Thu (n={R['main']['n_oth']})\")\n"
            "if HAVE_REAL:  # redraw live from the cache on top (must overlay exactly)\n"
            "    pf = (armat[fri].mean(axis=0).cumsum() * 1e4)\n"
            "    po = (armat[~fri].mean(axis=0).cumsum() * 1e4)\n"
            "    ax.plot(t, pf, lw=1, ls=':', color=RED)\n"
            "    ax.plot(t, po, lw=1, ls=':', color=GREY)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('sessions since day 0'); ax.set_ylabel('mean cumulative AR (bps vs SPY)')\n"
            "ax.set_title('Mean cumulative abnormal return around negative 8-Ks (frozen + live overlay)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
        ),

        md(
            "## 2 · Robustness — six cells, six nothings\n\n"
            "| slice | n (Friday) | gap bps | Welch *t* | pooled drift bps (*t*) |\n"
            "|---|--:|--:|--:|--:|\n"
            + "\n".join(
                f"| {lab} | {n} ({nf}) | {gb:+.1f} | {gt:+.2f} | {pb:+.1f} ({pt:+.2f}) |"
                for lab, n, nf, gb, gt, pb, pt in R["robust"]
            ) +
            "\n\nSigns both ways, every |*t*| < 1.4. The one seemingly-interesting cell "
            "(CEO exits, gap −400 bps) is *t* = −1.39 across 58 Friday events and flips "
            "sign in the other two classes — exactly what multiple-comparison noise looks "
            "like.\n\n"
            "> 💡 **In plain words:** slice it by kind of bad news or by decade — no "
            "slice shows the Friday effect."
        ),
        code(
            "# permutation placebo, live (reduced draws in-notebook; canonical p in R)\n"
            "if HAVE_REAL:\n"
            "    p = st.placebo_gap(fri, armat, n_perm=300, seed=608)\n"
            "    print(f'placebo (300 draws, live) p = {p:.3f}   | canonical (2,000 draws) '\n"
            "          f\"p = {R['main']['p_perm']:.3f}\")\n"
            "else:\n"
            "    print(f\"canonical placebo p = {R['main']['p_perm']:.3f} (2,000 draws, seeded)\")\n"
        ),

        md(
            "## 3 · Third axis — the hiding itself (full panel, zero survivorship)\n\n"
            "| share | bad news | earnings control | two-prop z |\n|---|--:|--:|--:|\n"
            f"| Friday | {s['friday'][0]:.1f}% | {s['friday'][1]:.1f}% | "
            f"+{s['friday'][2]:.2f} |\n"
            f"| after 16:00 ET | {s['after_close'][0]:.1f}% | {s['after_close'][1]:.1f}% "
            f"| +{s['after_close'][2]:.2f} |\n"
            f"| Friday after close | {s['friday_pm'][0]:.1f}% | {s['friday_pm'][1]:.1f}% "
            f"| +{s['friday_pm'][2]:.2f} |\n\n"
            f"Against the **uniform-calendar null** (Fridays = {s['friday_null']:.1f}% of "
            f"EDGAR-open days) the bad-news Friday share is z = +{s['friday_vs_null_z']:.2f} "
            "— *not* significant. The weekday mix of bad news is flat "
            f"({' / '.join(f'{x:.1f}' for x in s['wd_bad'])}%); the *control* is what "
            f"deserts Friday ({s['friday'][1]:.1f}%). The unambiguous hiding margin is "
            "the **clock**: after-hours (z = +4.9) and the Friday-PM combination "
            "(z = +15.2, a 9× ratio) — consistent with Niessner (2015) and deHaan et al. "
            "(2015).\n\n"
            "> 💡 **In plain words:** companies genuinely wait for the closing bell to "
            "release bad news. They do NOT particularly wait for Friday — Friday just "
            "looks dirty because good news avoids it."
        ),
        code(
            "x = np.arange(5); w = 0.38\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "ax1.bar(x - w/2, R['shares']['wd_bad'], w, color=RED, label='bad news')\n"
            "ax1.bar(x + w/2, R['shares']['wd_ctrl'], w, color=GREY, label='earnings ctrl')\n"
            "ax1.axhline(R['shares']['friday_null'], ls='--', c='k', lw=1)\n"
            "ax1.set_xticks(x, ['Mon','Tue','Wed','Thu','Fri']); ax1.set_ylabel('% of filings')\n"
            "ax1.set_title('weekday mix'); ax1.legend()\n"
            "pairs = [('Friday', 'friday'), ('after close', 'after_close'),\n"
            "         ('Friday PM', 'friday_pm')]\n"
            "xb = np.arange(3)\n"
            "ax2.bar(xb - w/2, [R['shares'][k][0] for _, k in pairs], w, color=RED)\n"
            "ax2.bar(xb + w/2, [R['shares'][k][1] for _, k in pairs], w, color=GREY)\n"
            "for i, (_, k) in enumerate(pairs):\n"
            "    ax2.text(i, R['shares'][k][0] + 1, f\"z=+{R['shares'][k][2]:.1f}\",\n"
            "             ha='center', fontsize=9)\n"
            "ax2.set_xticks(xb, [p for p, _ in pairs]); ax2.set_title('timing shares (%)')\n"
            "plt.tight_layout(); plt.show()\n"
        ),

        md(
            "## 4 · Tradability — the cost stack\n\n"
            "Entry at the day-0 close (the ONE lag — the filing is public before that "
            "close by construction, so the rule is implementable but never captures the "
            "day-0 hit), cover at close(+10), SPY-hedged; 10 bps one-way × 2 stock legs "
            "+ 1 bp × 2 hedge legs; shorts pay borrow.\n\n"
            "| rule | n (~/yr) | gross/event | costs | net/event |\n|---|--:|--:|--:|--:|\n"
            + "\n".join(
                f"| {lab} | {n} (~{py:.0f}) | {g:+.1f} bps (*t* = {tg:+.2f}, hit "
                f"{int(h*100)}%) | {c:.0f} bps | **{nt:+.1f} bps** (*t* = {tn:+.2f}) |"
                for lab, n, py, g, tg, h, c, nt, tn in R["trades"]
            ) +
            "\n\nGross never clears *t* = 0.5; net is a rounding error or negative. "
            "Capacity makes it worse: the median 4.02/2.06 filer is a small cap with "
            "real HTB risk.\n\n"
            "> 💡 **In plain words:** even free of costs there's nothing here; with "
            "costs it's a slow bleed of commissions."
        ),

        md(
            "## 5 · Synthetic machinery control *(never market evidence)*\n\n"
            "Deterministic planted-effect world (seed 608): the harness must stay quiet "
            "at the null and light up when a Friday drift or a Friday-hiding propensity "
            "is planted."
        ),
        code(
            "print(f\"{'world':22s} {'fri share':>9s} {'gap bps':>9s} {'Welch t':>8s} {'placebo p':>10s}\")\n"
            "for lab, kw in [('null', {}), ('drift -300 bps Fri', dict(drift_fri=-0.03)),\n"
            "                ('hide p_fri=0.35', dict(p_fri=0.35))]:\n"
            "    pl, am = data.synthetic_world(**kw)\n"
            "    gs = st.group_stats(pl['friday'].to_numpy(), am)\n"
            "    pp = st.placebo_gap(pl['friday'].to_numpy(), am, n_perm=300, seed=608)\n"
            "    print(f\"{lab:22s} {pl['friday'].mean()*100:8.1f}% {gs['gap_bps']:+9.1f} \"\n"
            "          f\"{gs['gap_t']:+8.2f} {pp:10.3f}\")\n"
            "print('\\ncanonical (500-draw) rows, mirror of results.md:')\n"
            "for lab, fs, gb, gt, pp in R['syn']:\n"
            "    print(f'  {lab:22s} fri {fs:4.1f}%  gap {gb:+7.1f} bps  t {gt:+5.2f}  p {pp:.3f}')\n"
        ),

        md(
            "## 6 · Verdict\n\n"
            "- **Signal — NONE.** Friday-vs-weekday drift gap "
            f"{m['gap']:+.1f} bps, Welch *t* = {m['gap_t']:+.2f}, winsorized "
            f"{m['gap_t_wins']:+.2f}, permutation p = {m['p_perm']:.3f}; all six "
            "robustness cells |*t*| < 1.4 with signs both ways; pooled bad-news drift "
            f"*t* = {m['pooled_t']:.2f}. Survivorship (39.5% mapped) understates drift "
            "*levels* and is named — but the *gap* the claim needs is absent on the "
            "visible tape.\n"
            "- **Tradability — MIRAGE.** +3.8 bps/event net (*t* = 0.03) at ten "
            "trades/yr; the Friday-PM variant is net negative; HTB small caps.\n"
            "- **Friday-night dump? — MIXED.** After-hours dumping is emphatic "
            "(z = +4.9; Friday-PM 12.8% vs 1.4%, z = +15.2) but the Friday *calendar* "
            "tilt of bad news is z = +1.4 vs uniform — the folklore mistakes ordinary "
            "news avoiding Friday for bad news seeking it.\n\n"
            "**The honesty checks:** one documented execution lag; costs one-way × NAV "
            "with borrow on shorts; survivorship named on the Signal axis; the timing "
            "axis uses the survivorship-free full panel; synthetic control quoted only "
            "as machinery proof; documented weekday-orthogonal sampling cap.\n\n"
            "---\n"
            f"*As-of {R['as_of']}; fingerprints events `{R['fp_events']}` · SPY "
            f"`{R['fp_spy']}`. Reproduce: `python examples/verify.py`. Literature map: "
            "[docs/references.md](../docs/references.md).*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    nbf.write(nb, os.path.join(HERE, "02_for_the_quants.ipynb"))


if __name__ == "__main__":
    build_curious()
    build_quants()
    print("notebooks written.")
