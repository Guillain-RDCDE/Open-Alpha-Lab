"""Generate the two narrative notebooks for Study 621 (Share-Class Spreads).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached four-ticker tape
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance split-adjusted
# closes, BRK pair 1996-05-09 -> 2026-06-30, GOOG pair 2014-04-03 -> 2026-06-30).
R = dict(
    as_of="2026-06-30", fingerprint="145603f5d518",
    brk_days=7583, brk_start="1996-05-09", goog_days=3078, goog_start="2014-04-03",
    # violations: (threshold bps, share %, days)
    viol=[(0, 35.80, 2715), (10, 18.08, 1371), (50, 1.77, 134), (100, 0.24, 18)],
    worst_up_bps=194, worst_up_date="2020-03-13", worst_dn_bps=-700, worst_dn_date="2009-02-20",
    asym_brk=13.0, brk_dn50=22.95, brk_up50=1.77,
    asym_goog=0.7, goog_up50=40.16, goog_dn50=26.74,
    # BRK discount distribution
    mean_bps=-35.5, hac_t=-12.53, hac_lags=10, median_bps=-8.2, std_bps=82,
    p1_bps=-385.7, p99_bps=62.7,
    # eras: (label, mean bps, HAC t, % days above parity)
    brk_eras=[("1996-2009", -55.1, -10.82, 33.9), ("2010-2019", -7.1, -5.99, 44.9),
              ("2020-2026", -37.7, -10.83, 25.9)],
    # GOOGL voting premium
    goog_mean_bps=51.6, goog_t=6.45, goog_std=149, goog_min=-435, goog_max=531,
    goog_eras=[("2014-2017", 237.5, 28.59, 0), ("2018-2021", -4.8, -0.65, 50),
               ("2022-2026", -53.9, -14.74, 90)],
    # half-lives
    rho_brk=0.8566, hl_brk=4.5, rho_goog=0.9919, hl_goog=85.2,
    # pairs trade: diagnostic + honest net rows (cost bps, net %/yr, HAC t)
    pairs_diag=(8.42, 9.38, 33), pairs=[(2.0, 0.21, 0.38), (5.0, -0.95, -1.69),
                                        (10.0, -2.87, -4.75)],
    # switch overlay: diagnostic (thr1%, net, t, switches, %days in B) + honest rows
    # honest: (thr %, net %/yr, t, pre2010 net, pre t, post2010 net, post t)
    sw_diag=(4.27, 6.02, 158, 23),
    switch=[(0.5, -0.56, -1.03, -1.06, -0.93, -0.15, -0.39),
            (1.0, 0.14, 0.28, 0.13, 0.12, 0.14, 0.43),
            (2.0, 0.30, 0.73, 0.67, 0.74, -0.01, -0.04),
            (3.0, 0.42, 1.18, 0.76, 0.95, 0.14, 1.00)],
    # synthetic control: (label, mean bps, HAC t, %days > +50, %days < -50)
    syn=[("PLANTED bound -40 bps", -54.9, -33.86, 0.00, 52.64),
         ("NULL no bound, 0", -3.1, -1.50, 14.40, 16.98)],
)

BADGES = (
    "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Tradable for A-holders?: Busted](https://img.shields.io/badge/Tradable_for_A--holders%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from share_class_spreads import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PX = data.load_real()
    PX = PX[PX.index <= "2026-06-30"]              # frozen as-of (docs/results.md)
    BRK = st.gap_series(PX, "BRK-A", "BRK-B", data.BRK_PARITY, start=data.BRKB_START)
    GOO = st.gap_series(PX, "GOOG", "GOOGL", 1.0, start=data.GOOG_START)
else:
    PX = BRK = GOO = None
print("real tape cached:", HAVE_REAL,
      "| BRK pair days:", (0 if BRK is None else len(BRK)),
      "| GOOG pair days:", (0 if GOO is None else len(GOO)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The share that can never cost too much 🔀\n"
            "### Berkshire's 1/1500 bound vs the Google twins — a law of one price you can watch work, in plain English\n\n"
            + BADGES +
            "Berkshire Hathaway trades under two tickers. One share of **BRK.A** costs about as much "
            "as a house (~$750,000); one share of **BRK.B** costs about as much as a dinner for two "
            "(~$500). And there's a rule, written into the company's charter, that ties them together: "
            "**any A share can be converted, at any time, into exactly 1,500 B shares** — but *never* "
            "the other way.\n\n"
            "That one-way door has a beautiful consequence, spelled out by Warren Buffett himself: the "
            "B **can never cost more than 1/1500th of the A** (if it did, anyone holding an A could "
            "mint 1,500 Bs and sell them for free money — and they do). But because the door only "
            "opens one way, the B **can trade at a discount**, and nobody can arbitrage it shut.\n\n"
            "Google (Alphabet) also trades under two tickers — **GOOGL** (one vote) and **GOOG** (no "
            "vote) — but with **no conversion door at all**. Same company, same economics, no leash.\n\n"
            "So: does the market really enforce Buffett's bound, every day, for thirty years? And is "
            "the B-discount the free lunch it looks like?\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the violation thresholds and the "
            "cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Daily split-adjusted closes (yfinance), BRK pair from the "
            "B's first trade (1996), Google pair from the 2014 class-C split, frozen as-of 2026-06-30. "
            "Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the market enforce the 1/1500 bound? | **Yes, visibly.** In 30 years the B closed "
            "more than half a percent *above* the bound on only **1.8% of days** (never more than "
            "~2%, and that was the March-2020 panic) — while it spent **23% of days** more than half "
            "a percent *below*. Capped on top, free-falling underneath: exactly what a one-way door "
            "does. |\n"
            "| What about the Google twins? | **No leash at all.** The voting share traded **+2.4% "
            "rich** in 2014–17, then **cheaper than the non-voting share 90% of days** since 2022. A "
            "spread with no conversion door just… wanders (it takes ~4 months to un-stretch, vs ~1 "
            "week for Berkshire's). |\n"
            "| Is the B-discount free money for A-holders? | **No — and this is the fun part.** On "
            "paper, switching into the cheap B and back earns +4.3%/yr. But that profit assumes you "
            "trade *at the exact prices that made the discount look big* — mostly stale prints of the "
            "barely-traded A share. Wait one day for a real fill and the edge collapses to **~0.1%/yr, "
            "statistically nothing**, in every era. |\n\n"
            "> The bound is **real**. The reason it's real is that *someone else* — A-holders with a "
            "conversion right — enforces it instantly. Which is precisely why there's nothing left "
            "for you."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"BRK.B can never cost more than 1/1500th of BRK.A — a one-way conversion bound the "
            "market enforces daily. GOOG vs GOOGL has no bound at all.\"*\n\n"
            "This isn't market folklore — it's in **Berkshire's own shareholder memo** "
            "([compab.pdf](https://www.berkshirehathaway.com/compab.pdf)): each A converts into "
            "1,500 B at the holder's option; B never converts back; so the B \"can never sell for "
            "anything more than a tiny fraction above\" parity, but \"can sell at a discount.\" The "
            "Google twins are the perfect control group: two classes, same company, **no conversion "
            "right in either direction**."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "This is the cleanest natural experiment in finance: the **law of one price** with the "
            "arbitrage mechanism *visible in the charter*. If the bound holds where conversion exists "
            "and fails where it doesn't, you're watching arbitrage — not luck — hold prices together. "
            "And it sets up the tempting question every A-holder eventually asks: when my B twin goes "
            "on sale at a 2% discount, isn't switching classes free money?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"Take every trading day since the B was born ({R['brk_start']}, "
            f"{R['brk_days']:,} days) and compute the **gap**: how far the B sits above or below "
            "exactly 1/1500th of the A (split-adjusted, so one number works for the whole history). "
            "Then:\n\n"
            "1. **Count the violations.** Days the B closed *above* parity, by how much.\n"
            "2. **Compare the two tails.** A one-way door should cap the top and leave the bottom "
            "free.\n"
            f"3. **Run the twin with no door.** Same math on GOOGL vs GOOG ({R['goog_days']:,} days "
            "since 2014).\n"
            "4. **Try to eat the discount.** Simulate an A-holder switching into the cheap B and "
            "back — with the one honesty rule that kills most paper edges: you trade at the *next* "
            "day's close, not at the price that generated the signal.\n\n"
            "One caveat we name up front: an A share trades a few hundred times a day, so its 4 p.m. "
            "print is often minutes stale — tiny apparent \"premiums\" of a few hundredths of a "
            "percent are measurement noise, not broken arbitrage. We judge the bound where prints "
            "can't hide: half a percent and beyond."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The gap itself, thirty years of it.** Watch the ceiling."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = BRK*1e4\n"
            "    fig, ax = plt.subplots(figsize=(10, 4.8))\n"
            "    ax.plot(g.index, g.values, lw=.5, color=GREY)\n"
            "    ax.axhline(0, color='k', lw=1)\n"
            "    ax.axhline(50, color=RED, lw=1.2, ls='--', label='parity + 50 bps (the ceiling)')\n"
            "    ax.fill_between(g.index, 0, g.where(g>0).values, color=RED, alpha=.55, label='B above parity')\n"
            "    ax.fill_between(g.index, 0, g.where(g<0).values, color=GREEN, alpha=.45, label='B below parity (discount)')\n"
            "    ax.set_ylabel('BRK gap: 1500*B/A - 1 (bps)'); ax.set_ylim(-750, 260)\n"
            "    ax.set_title('Capped on top, free underneath: the one-way conversion bound at work')\n"
            "    ax.legend(loc='lower right'); plt.tight_layout(); plt.show()\n"
            "    print(f'days B > parity+50bps: {(BRK>0.005).mean()*100:.2f}%   days B < parity-50bps: {(BRK<-0.005).mean()*100:.2f}%')\n"
            "else:\n"
            "    print('cache missing - frozen numbers:', R['brk_up50'], '% above vs', R['brk_dn50'], '% below +/-50bps')"
        ),
        md(
            f"The picture *is* the verdict: above the line the gap is a thin red film (beyond "
            f"+50 bps on just **{R['brk_up50']:.1f}%** of days, worst **+{R['worst_up_bps']} bps** in "
            f"the {R['worst_up_date'][:4]} COVID panic), below it green stalactites reaching "
            f"**{R['worst_dn_bps']} bps** in the 2009 crisis. The B spent **{R['brk_dn50']:.0f}%** of "
            f"days more than 50 bps cheap — **{R['asym_brk']:.0f}× more often** than it spent rich. "
            "That asymmetry is the one-way door, printed on the tape."
        ),
        md(
            "**Now the twin with no door.** Same company, two Google share classes, no conversion "
            "right — the spread has nothing to hold onto."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)\n"
            "    for ax, s, lbl, col in [(axes[0], BRK*1e4, 'BRK gap (bounded)', GREEN),\n"
            "                            (axes[1], GOO*1e4, 'GOOGL-GOOG spread (no bound)', AMBER)]:\n"
            "        ax.hist(np.clip(s.values, -500, 500), bins=80, color=col, alpha=.85)\n"
            "        ax.axvline(0, color='k', lw=1); ax.axvline(50, color=RED, ls='--', lw=1)\n"
            "        ax.set_title(lbl); ax.set_xlabel('gap (bps, clipped at +/-500)')\n"
            "    axes[0].set_ylabel('days')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'GOOGL premium: mean {GOO.mean()*1e4:+.0f} bps, range {GOO.min()*1e4:+.0f} to {GOO.max()*1e4:+.0f} bps')\n"
            "else:\n"
            "    print('cache missing - GOOGL premium mean', R['goog_mean_bps'], 'bps, range', R['goog_min'], 'to', R['goog_max'])"
        ),
        md(
            f"Berkshire's histogram slams into a wall just past zero; Google's sprawls both ways "
            f"(**{R['goog_min']} to +{R['goog_max']} bps**). And the Google spread isn't even stable "
            f"in *sign*: the voting share was **+{R['goog_eras'][0][1]:.0f} bps rich** in 2014–17, "
            f"then **{R['goog_eras'][2][1]:.0f} bps cheap** since 2022 — below the non-voting class "
            f"on **{R['goog_eras'][2][3]:.0f}% of days** — because Alphabet's buybacks concentrate in "
            "the non-voting class. No door, no law. A stretched Berkshire gap heals in about **a "
            f"week** (half-life {R['hl_brk']:.1f} trading days); a stretched Google gap takes about "
            f"**four months** ({R['hl_goog']:.0f} days)."
        ),
        md(
            "**The free-lunch test.** The B trades at a real, persistent discount (average "
            f"**{R['mean_bps']:.0f} bps** — statistically rock-solid). So: switch into the B whenever "
            "it's ≥1% cheap, switch back at parity, pocket the difference? Here's that strategy under "
            "two rules — *paper* (you magically trade at the closing prints that defined the "
            "discount) and *honest* (you trade at the next day's close, like a human)."
        ),
        code(
            "labels = ['paper fill\\n(at the signal print)', 'honest fill\\n(next close)']\n"
            "vals = [R['sw_diag'][0], R['switch'][1][1]]\n"
            "ts = [R['sw_diag'][1], R['switch'][1][2]]\n"
            "if HAVE_REAL:\n"
            "    a = st.switch_overlay(PX, 'BRK-A', 'BRK-B', data.BRK_PARITY, thr=0.01, cost_bps=5.0, fill='same')\n"
            "    b = st.switch_overlay(PX, 'BRK-A', 'BRK-B', data.BRK_PARITY, thr=0.01, cost_bps=5.0, fill='next')\n"
            "    vals = [a['net_ann_pct'], b['net_ann_pct']]; ts = [a['hac_t'], b['hac_t']]\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.4))\n"
            "ax.bar(labels, vals, color=[AMBER, GREY], width=.55)\n"
            "for i,(v,t) in enumerate(zip(vals, ts)):\n"
            "    ax.annotate(f'{v:+.2f}%/yr\\n(t={t:+.2f})', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('excess vs just holding A (net %/yr)')\n"
            "ax.set_title('The B-discount \"free lunch\" exists only at prices you could never get')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'paper {vals[0]:+.2f}%/yr (t={ts[0]:+.2f})  ->  honest {vals[1]:+.2f}%/yr (t={ts[1]:+.2f})')"
        ),
        md(
            f"From **+{R['sw_diag'][0]:.1f}%/yr (t = {R['sw_diag'][1]:.1f})** to "
            f"**+{R['switch'][1][1]:.2f}%/yr (t = {R['switch'][1][2]:.2f})** — the entire edge lived "
            "inside the prints that generated the signal. When the discount \"appears,\" it's often "
            "the A share's stale, wide closing print; by the first close at which you could actually "
            "trade, the gap has already snapped most of the way back (that one-week half-life cuts "
            "both ways). The quants notebook shows the same collapse at every threshold — 0.5%, 1%, "
            "2%, 3% — in every era, and for the hedge-fund version of the trade too."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** The one-way bound is on the tape: beyond +50 bps on only "
            f"**{R['brk_up50']:.1f}%** of days (max +{R['worst_up_bps']} bps, in a panic) vs "
            f"**{R['brk_dn50']:.0f}%** of days below — a {R['asym_brk']:.0f}× asymmetry — plus a "
            f"persistent average discount of **{R['mean_bps']:.0f} bps** that is statistically "
            "unambiguous. The unleashed Google twins confirm the mechanism by lacking it.\n"
            "- **Tradability — Mirage.** Every scheme to harvest the gap dies under an honest fill "
            "(+0.2%/yr at best, statistically zero), before you even mention the $750k minimum A "
            "share.\n"
            "- **\"Free money for A-holders\"? — Busted.** The +4.3%/yr paper edge is a "
            "trade-at-the-signal-print illusion; delayed one close it reads t ≤ 1.2 everywhere."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Why does the discount exist at all?** Because the door only opens one way, closing "
            "it from below needs someone to *sell A and buy B* in size — and A's illiquidity makes "
            "that slow. The bound is enforced in seconds; the discount heals in a week.\n"
            "- **The Google lesson.** \"Voting premium\" sounds like a constant of nature; it flipped "
            "sign when buybacks tilted to the non-voting class. Spreads without a mechanism are "
            "narratives, not laws.\n"
            "- **Build your own.** The engine takes any two-class pair — try Heico A/HEI, Alphabet "
            "against other bridgeless twins, or a CEF against its NAV "
            "([367-closed-end-fund-discount](../../367-closed-end-fund-discount/README.md)).\n\n"
            "*Think you can catch the discount faster than the conversion arbitrageurs? Show the "
            "switch overlay clearing t = 2 with a next-close fill — then we'll talk.*"
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
            "# Share-Class Spreads — a quantitative teardown 🔬\n"
            "### Bound violations by threshold + tail asymmetry · HAC t on the structural discount, "
            "by era · the bounded/unbounded half-life contrast · fill-at-print vs next-close collapse "
            "on two trading rules · a planted-bound synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim is contractual, not statistical — Berkshire's charter gives every A-holder a "
            "1→1,500 conversion option, B-to-A does not exist, and Alphabet's twins have no bridge "
            "at all — so the job is to *measure the enforcement* and then to test, honestly, whether "
            "anything is left to trade.\n\n"
            "> ⚠️ **Data note.** yfinance daily split-adjusted closes; BRK pair "
            + f"{R['brk_start']} → {R['as_of']} ({R['brk_days']:,} days, adjusted parity a constant "
            "1,500), GOOG pair "
            + f"{R['goog_start']} → {R['as_of']} ({R['goog_days']:,} days). Price-only returns "
            "(neither BRK class ever paid a dividend; Google's 2024+ dividends are identical across "
            "classes and cancel in the ratio). No panel, no survivorship screen — two named pairs, "
            "nothing searched over. BRK-A's close is stale/wide (a few hundred trades/day at ~$750k), "
            "so sub-10-bps 'premiums' are print noise; the bound is judged at 50–100 bps. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fingerprint"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | B beyond parity +50 bps on **{R['brk_up50']:.2f}%** of days vs "
            f"**{R['brk_dn50']:.2f}%** below (asym **{R['asym_brk']:.0f}×**; GOOG control "
            f"{R['asym_goog']:.1f}×); mean discount **{R['mean_bps']:.1f} bps**, "
            f"**HAC t = {R['hac_t']:.2f}**, significant in all three eras; half-life "
            f"{R['hl_brk']:.1f}d vs {R['hl_goog']:.0f}d unbounded. |\n"
            f"| **Tradability** | `MIRAGE` | Honest one-lag z-score pairs trade nets "
            f"**{R['pairs'][0][1]:+.2f}%/yr (t = {R['pairs'][0][2]:+.2f})** at 2 bps, "
            f"{R['pairs'][1][1]:+.2f}%/yr at 5 bps (borrow paid); the paper "
            f"+{R['pairs_diag'][0]:.1f}%/yr (t = {R['pairs_diag'][1]:.1f}) exists only at the "
            "signal's own prints. $750k minimum A lot; conversion is one-way and broker-mediated. |\n"
            f"| **Tradable for A-holders?** | `BUSTED` | Switch overlay: fill-at-print "
            f"**+{R['sw_diag'][0]:.2f}%/yr (t = {R['sw_diag'][1]:.2f})** → next-close fill "
            f"**{R['switch'][1][1]:+.2f}%/yr (t = {R['switch'][1][2]:+.2f})**; t ≤ 1.2 at every "
            "threshold in every era. |\n\n"
            "> 💡 In plain words: the bound is real and mechanically enforced; the discount is real "
            "and statistically decisive; and everything tradable about either evaporates at the "
            "first honest fill."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $g_t = 1500\\,B_t/A_t - 1$ on split-adjusted closes (negative = B cheap; the 2010 "
            "50:1 split folds the pre-2010 ratio of 30 into a constant 1,500). The conversion right "
            "implies a one-sided no-arbitrage condition:\n\n"
            "$$g_t \\le \\varepsilon \\quad\\text{(a few bps of frictions)},\\qquad g_t \\text{ unbounded below}.$$\n\n"
            "- **H₁ (the bound).** $\\Pr[g_t > 50\\text{ bps}] \\approx 0$, upward excursions small, "
            "brief, crisis-clustered; the GOOGL−GOOG spread $s_t$ shows no such cap.\n"
            "- **H₂ (the structural discount).** $\\mathbb{E}[g_t] < 0$ with an "
            "autocorrelation-robust t (the gap has ρ ≈ 0.86, so a naive t would overstate n).\n"
            "- **H₃ (tradability / the myth).** Neither a long-cheap/short-rich pairs rule nor an "
            "A-holder class switch survives one honest execution lag + costs.\n\n"
            "We find **H₁ and H₂ decisively supported**, and **H₃ decisively rejected** — the paper "
            "profits are a fill-at-the-signal-print artefact."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "This pair is the cleanest *identified* law-of-one-price experiment on US tape: the "
            "arbitrage mechanism is written in the charter (compare Lamont-Thaler's Palm/3Com, where "
            "the blocked mechanism produced a 2-month 30% violation). If $g_t$ is capped where the "
            "conversion exists and $s_t$ wanders where it doesn't, price discipline is *caused* by "
            "the mechanism, not by market efficiency in the abstract. The methodological stake is "
            "just as sharp: close-stamped daily data on an illiquid leg **manufactures** relative-"
            "value alpha under a same-close fill — this study quantifies that mirage at t = 6–9 "
            "before killing it with one lag."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** yfinance split-adjusted closes; BRK pair {R['brk_days']:,} days "
            f"({R['brk_start']} →), GOOG pair {R['goog_days']:,} days ({R['goog_start']} →); as-of "
            f"{R['as_of']}, fingerprint `{R['fingerprint']}`.\n"
            "- **Bound.** Violation shares at +0/+10/+50/+100 bps; ±50 bps tail asymmetry; same for "
            "the GOOG control. Sub-10-bps excursions are non-synchronous-close noise, named as such.\n"
            "- **Discount.** Mean $g_t$ with Newey-West t (Bartlett, lags = ⌊4(n/100)^{2/9}⌋ = "
            f"{R['hac_lags']}), distribution, era slices (1996–2009 / 2010–2019 / 2020–2026), AR(1) "
            "half-lives for both pairs.\n"
            "- **Trading rules.** (a) Dollar-neutral z-score pairs (entry |z|>1.5 on 252d, exit "
            "|z|<0.25, short leg pays 50 bps/yr borrow); (b) A-holder switch overlay (hold B while "
            "≥ thr cheap, back to A at parity), thresholds 0.5–3%. Exactly **one execution lag**: "
            "signal at close *t*, filled at close *t+1*, first return accrues *t+1→t+2*. The "
            "`fill='same'` variant (fill at close *t*) is the diagnostic that locates the mirage. "
            "Costs 2/5/10 bps one-way × NAV per leg; a switch = 2 legs.\n"
            "- **Control.** A deterministic synthetic two-class world with a planted mean discount "
            "and a hard one-way bound, plus a symmetric unbounded null the detector must NOT flag."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Bound enforcement — violations by threshold, and the tail asymmetry\n\n"
            "The one-way conversion predicts a hard ceiling and a free floor. Count both tails."
        ),
        code(
            "if HAVE_REAL:\n"
            "    vb = st.violation_stats(BRK); vg = st.violation_stats(GOO)\n"
            "    rows = [(thr, vb[f'above_{thr}bps_share']*100, vb[f'above_{thr}bps_days']) for thr in (0,10,50,100)]\n"
            "    asym = (vb['dn50_share']*100, vb['up50_share']*100, vb['asym_ratio'],\n"
            "            vg['up50_share']*100, vg['dn50_share']*100)\n"
            "else:\n"
            "    rows = R['viol']; asym = (R['brk_dn50'], R['brk_up50'], R['asym_brk'], R['goog_up50'], R['goog_dn50'])\n"
            "print('BRK-B above parity by more than:')\n"
            "for thr, sh, dd in rows: print(f'   {thr:>4d} bps : {sh:5.2f}% of days ({dd} days)')\n"
            "print(f'tail asymmetry beyond +/-50bps: below {asym[0]:.2f}% vs above {asym[1]:.2f}%  ->  {asym[2]:.1f}x')\n"
            "print(f'GOOG control: above {asym[3]:.2f}% vs below {asym[4]:.2f}%  (no bound: symmetric)')\n"
            "if HAVE_REAL:\n"
            "    fig, ax = plt.subplots(figsize=(10, 4.6))\n"
            "    ax.plot(BRK.index, BRK.values*1e4, lw=.5, color=GREY)\n"
            "    ax.axhline(50, color=RED, ls='--', lw=1.2, label='+50 bps ceiling')\n"
            "    ax.axhline(0, color='k', lw=1)\n"
            "    ax.fill_between(BRK.index, 0, (BRK*1e4).where(BRK>0).values, color=RED, alpha=.6)\n"
            "    ax.fill_between(BRK.index, 0, (BRK*1e4).where(BRK<0).values, color=GREEN, alpha=.4)\n"
            "    ax.set_ylabel('g = 1500*B/A - 1 (bps)'); ax.set_ylim(-750, 260)\n"
            "    ax.set_title('30 years of the one-way bound: capped above, free below'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: beyond +50 bps — where stale prints can't hide — the \"violations\" "
            f"are **{R['brk_up50']:.2f}%** of days ({R['viol'][2][2]} days in 30 years, clustered in "
            f"1996–98, 2008 and March 2020; worst **+{R['worst_up_bps']} bps** on "
            f"{R['worst_up_date']}), against **{R['brk_dn50']:.2f}%** of days below — a "
            f"**{R['asym_brk']:.0f}×** asymmetry. The GOOG pair is {R['asym_goog']:.1f}× — i.e., "
            "symmetric. The ceiling exists exactly where the conversion right exists."
        ),
        md(
            "### 4b · The structural discount — HAC t, by era\n\n"
            "The gap is heavily autocorrelated (ρ ≈ 0.86), so the mean gets a Newey-West t."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ds = st.dist_stats(BRK)\n"
            "    eras = []\n"
            "    for lbl, s in [('1996-2009', BRK[:'2010-01-20']), ('2010-2019', BRK['2010-01-21':'2019']), ('2020-2026', BRK['2020':])]:\n"
            "        d2 = st.dist_stats(s); eras.append((lbl, d2['mean_bps'], d2['hac_t']))\n"
            "    print(f\"mean {ds['mean_bps']:+.1f} bps  HAC t = {ds['hac_t']:+.2f} (lags={ds['hac_lags']}, n={ds['n']:,})\")\n"
            "    print(f\"median {ds['median_bps']:+.1f}  std {ds['std_bps']:.0f}  p1 {ds['p1_bps']:+.1f}  p99 {ds['p99_bps']:+.1f}\")\n"
            "else:\n"
            "    eras = [(l, m, t) for l, m, t, _ in R['brk_eras']]\n"
            "    print(f\"mean {R['mean_bps']:+.1f} bps  HAC t = {R['hac_t']:+.2f} (frozen)\")\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar([e[0] for e in eras], [e[1] for e in eras], color=GREEN, width=.55)\n"
            "for i, e in enumerate(eras): ax.annotate(f'{e[1]:+.1f} bps\\nHAC t={e[2]:+.1f}', (i, e[1]), ha='center', va='top')\n"
            "ax.axhline(0, color='k', lw=1); ax.set_ylabel('mean gap (bps)')\n"
            "ax.set_title('The B discount is small, persistent, and significant in every era')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('eras:', [(e[0], round(e[1],1), round(e[2],2)) for e in eras])"
        ),
        md(
            f"> 💡 In plain words: the B has averaged **{R['mean_bps']:.1f} bps cheap** for 30 years "
            f"(**HAC t = {R['hac_t']:.2f}**), from {R['brk_eras'][0][1]:.0f} bps in the wild early "
            f"era to {R['brk_eras'][1][1]:.0f} bps in the placid 2010s and back to "
            f"{R['brk_eras'][2][1]:.0f} bps since 2020. The distribution is the bound's fingerprint: "
            f"p99 = **+{R['p99_bps']:.0f} bps** (capped) vs p1 = **{R['p1_bps']:.0f} bps** (free)."
        ),
        md(
            "### 4c · The unbounded control — GOOGL vs GOOG\n\n"
            "No conversion bridge, so the voting spread should wander — and it does more than that: "
            "it changed sign."
        ),
        code(
            "if HAVE_REAL:\n"
            "    geras = []\n"
            "    for lbl, s in [('2014-2017', GOO[:'2017']), ('2018-2021', GOO['2018':'2021']), ('2022-2026', GOO['2022':])]:\n"
            "        d2 = st.dist_stats(s); geras.append((lbl, d2['mean_bps'], d2['hac_t'], (s<0).mean()*100))\n"
            "else:\n"
            "    geras = R['goog_eras']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "cols = [GREEN if e[1] > 0 else RED for e in geras]\n"
            "ax.bar([e[0] for e in geras], [e[1] for e in geras], color=cols, width=.55)\n"
            "for i, e in enumerate(geras):\n"
            "    ax.annotate(f'{e[1]:+.0f} bps\\nt={e[2]:+.1f}', (i, e[1]), ha='center',\n"
            "                va='bottom' if e[1] > 0 else 'top')\n"
            "ax.axhline(0, color='k', lw=1); ax.set_ylabel('mean GOOGL-GOOG spread (bps)')\n"
            "ax.set_title('The voting premium is not a law: +238 bps, then negative (buyback tilt to class C)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('GOOG eras:', [(e[0], round(e[1],1), round(e[2],2), round(e[3])) for e in geras])"
        ),
        md(
            f"> 💡 In plain words: the \"voting premium\" was **+{R['goog_eras'][0][1]:.0f} bps** "
            f"(t = {R['goog_eras'][0][2]:.1f}) in 2014–17, statistically zero in 2018–21, and "
            f"**{R['goog_eras'][2][1]:.0f} bps** (t = {R['goog_eras'][2][2]:.1f}) since 2022, with "
            f"GOOGL *below* GOOG on {R['goog_eras'][2][3]:.0f}% of days — Alphabet's buybacks "
            "concentrate in class C. Spread std **" + f"{R['goog_std']}" + " bps** vs the bounded "
            f"pair's {R['std_bps']} bps; half-life **{R['hl_goog']:.0f} trading days** vs "
            f"**{R['hl_brk']:.1f}** (AR(1) ρ {R['rho_goog']:.4f} vs {R['rho_brk']:.4f}). No "
            "mechanism, no anchor."
        ),
        md(
            "### 4d · Tradability — the fill convention is the whole result\n\n"
            "Both rules, both conventions. `fill='same'` books the trade at the very close prints "
            "that generated the signal (only achievable by front-running your own data); "
            "`fill='next'` waits for the first close you could genuinely trade — the desk's one "
            "honest lag."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pd_ = st.pairs_trade(PX, 'BRK-A', 'BRK-B', data.BRK_PARITY, fill='same', cost_bps=0.0, borrow_bps_yr=0.0)\n"
            "    pairs_rows = [(cb, st.pairs_trade(PX, 'BRK-A', 'BRK-B', data.BRK_PARITY, fill='next', cost_bps=cb))\n"
            "                  for cb in (2.0, 5.0, 10.0)]\n"
            "    pdg = (pd_['gross_ann_pct'], pd_['gross_t'])\n"
            "    prow = [(cb, r['net_ann_pct'], r['net_t']) for cb, r in pairs_rows]\n"
            "    sd = st.switch_overlay(PX, 'BRK-A', 'BRK-B', data.BRK_PARITY, thr=0.01, cost_bps=5.0, fill='same')\n"
            "    sh = st.switch_overlay(PX, 'BRK-A', 'BRK-B', data.BRK_PARITY, thr=0.01, cost_bps=5.0, fill='next')\n"
            "    sdg = (sd['net_ann_pct'], sd['hac_t']); shn = (sh['net_ann_pct'], sh['hac_t'])\n"
            "else:\n"
            "    pdg = R['pairs_diag'][:2]; prow = R['pairs']\n"
            "    sdg = R['sw_diag'][:2]; shn = (R['switch'][1][1], R['switch'][1][2])\n"
            "labels = ['pairs trade\\n(z-score, L/S)', 'A-holder switch\\n(thr=1%)']\n"
            "paper = [pdg[0], sdg[0]]; honest = [prow[1][1], shn[0]]\n"
            "x = np.arange(2); w = 0.36\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.6))\n"
            "ax.bar(x - w/2, paper, w, color=AMBER, label='paper: fill at the signal print (gross / net@5bps)')\n"
            "ax.bar(x + w/2, honest, w, color=GREY, label='honest: next-close fill, net @ 5 bps')\n"
            "for i, v in enumerate(paper): ax.annotate(f'{v:+.1f}%', (i - w/2, v), ha='center', va='bottom')\n"
            "for i, v in enumerate(honest): ax.annotate(f'{v:+.2f}%', (i + w/2, v), ha='center', va='bottom')\n"
            "ax.axhline(0, color='k', lw=1); ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('%/yr'); ax.set_title('One honest execution lag deletes both strategies')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'pairs: paper gross {pdg[0]:+.2f}%/yr (t={pdg[1]:+.2f}) -> honest net: ' +\n"
            "      ', '.join(f'{cb:.0f}bps {v:+.2f}%/yr (t={t:+.2f})' for cb, v, t in prow))\n"
            "print(f'switch thr=1%: paper {sdg[0]:+.2f}%/yr (t={sdg[1]:+.2f}) -> honest {shn[0]:+.2f}%/yr (t={shn[1]:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: the pairs trade reads **+{R['pairs_diag'][0]:.1f}%/yr at "
            f"t = {R['pairs_diag'][1]:.1f}** if you pretend to trade the signal's own prints, and "
            f"**{R['pairs'][0][1]:+.2f}%/yr at t = {R['pairs'][0][2]:+.2f}** (2 bps, borrow paid) the "
            "moment you can't — negative at 5 bps and beyond. The signal *is* the microstructure "
            "noise of BRK-A's close. Add the access walls (a ~$750k minimum lot on the A leg, "
            "one-way broker-mediated conversion, a 35-bps average prize) and Tradability is a "
            "textbook **MIRAGE**."
        ),
        md(
            "### 4e · Third axis — was the discount *ever* a signal for A-holders?\n\n"
            "Full threshold × era grid for the honest-fill switch overlay (net @ 5 bps, 2 legs per "
            "switch). If deep discounts were ever genuinely harvestable — say pre-2010, when −2% "
            "days were common — it must show here."
        ),
        code(
            "if HAVE_REAL:\n"
            "    grid = []\n"
            "    for thr in (0.005, 0.01, 0.02, 0.03):\n"
            "        sw = st.switch_overlay(PX, 'BRK-A', 'BRK-B', data.BRK_PARITY, thr=thr, cost_bps=5.0, fill='next')\n"
            "        e = sw['excess']; pre, post = e[:'2009'], e['2010':]\n"
            "        tp, _, _ = st.hac_t(pre.values); tq, _, _ = st.hac_t(post.values)\n"
            "        grid.append((thr*100, sw['net_ann_pct'], sw['hac_t'],\n"
            "                     pre.mean()*252*100, tp, post.mean()*252*100, tq))\n"
            "else:\n"
            "    grid = R['switch']\n"
            "print(f\"{'thr':>5} | {'full net':>9} {'t':>6} | {'pre-2010':>9} {'t':>6} | {'2010+':>9} {'t':>6}\")\n"
            "for g in grid:\n"
            "    print(f'{g[0]:4.1f}% | {g[1]:+8.2f}% {g[2]:+6.2f} | {g[3]:+8.2f}% {g[4]:+6.2f} | {g[5]:+8.2f}% {g[6]:+6.2f}')\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "x = np.arange(len(grid)); w = 0.36\n"
            "ax.bar(x - w/2, [g[4] for g in grid], w, color=GREY, label='pre-2010 HAC t')\n"
            "ax.bar(x + w/2, [g[6] for g in grid], w, color=AMBER, label='2010+ HAC t')\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 bar'); ax.axhline(0, color='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{g[0]:.1f}%' for g in grid])\n"
            "ax.set_xlabel('switch threshold (B this much below parity)'); ax.set_ylabel('HAC t of net excess vs holding A')\n"
            "ax.set_ylim(-2, 3); ax.set_title('No threshold, no era clears the bar under an honest fill')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: **never.** Across thresholds 0.5–3% and both eras the honest net "
            f"excess sits between {min(g[1] for g in R['switch']):+.2f}%/yr and "
            f"{max(g[1] for g in R['switch']):+.2f}%/yr with **t between "
            f"{min(g[2] for g in R['switch']):+.2f} and {max(g[2] for g in R['switch']):+.2f}** — "
            "even the wild pre-2010 discounts (−7% at the 2009 trough) snapped back by the first "
            f"close you could trade (half-life {R['hl_brk']:.1f} days). The B-discount has **never** "
            "been a tradable signal for A-holders: **BUSTED**."
        ),
        md(
            "### 4f · Faithful-engine control — we know the truth here\n\n"
            "A deterministic two-class world (seed 621): shared GBM fundamental, AR(1) relative-value "
            "gap, class-specific print noise. World 1 plants a −40 bps mean discount **and** a hard "
            "one-way bound; world 2 is a symmetric unbounded null with zero mean gap. The detector "
            "(mean-gap HAC t + ±50 bps tail asymmetry) must flag the first and stay quiet on the "
            "second."
        ),
        code(
            "rows = []\n"
            "for lbl, kw in [('PLANTED bound + -40bps', dict(mean_discount=-0.004, bound=True)),\n"
            "                ('NULL no bound, 0 gap', dict(mean_discount=0.0, bound=False))]:\n"
            "    spx = data.synthetic_pair(seed=621, **kw)\n"
            "    det = st.bound_detector(spx, data.BRK_PARITY)\n"
            "    rows.append((lbl, det['mean_bps'], det['hac_t'], det['up50_share']*100, det['dn50_share']*100))\n"
            "    print(f\"{lbl:24s}: mean {det['mean_bps']:+6.1f} bps  HAC t = {det['hac_t']:+7.2f}  \"\n"
            "          f\"above +50bps {det['up50_share']*100:5.2f}%  below -50bps {det['dn50_share']*100:5.2f}%\")\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "ax.bar([r[0] for r in rows], [abs(r[2]) for r in rows], color=[GREEN, GREY], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='|t| = 2 bar')\n"
            "for i, r in enumerate(rows): ax.annotate(f'|t|={abs(r[2]):.1f}', (i, abs(r[2])), ha='center', va='bottom')\n"
            "ax.set_ylabel('|HAC t| of the mean gap'); ax.set_title('Planted bound lights up; unbounded null stays quiet')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the planted world reads mean {R['syn'][0][1]:+.1f} bps at "
            f"**HAC t = {R['syn'][0][2]:+.2f}** with a zero upward tail; the null reads "
            f"{R['syn'][1][1]:+.1f} bps at **t = {R['syn'][1][2]:+.2f}** with near-symmetric tails "
            f"({R['syn'][1][3]:.1f}% vs {R['syn'][1][4]:.1f}%). The machinery detects exactly what is "
            "planted and nothing else. *(A machinery proof — never cited in support of the stamps.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — the one-way bound is on the tape: >+50 bps on "
            f"**{R['brk_up50']:.2f}%** of days (>+100 bps: {R['viol'][3][1]:.2f}%; worst "
            f"+{R['worst_up_bps']} bps, {R['worst_up_date']}) vs {R['brk_dn50']:.2f}% below "
            f"(**{R['asym_brk']:.0f}×** asymmetry), mean discount **{R['mean_bps']:.1f} bps** at "
            f"**HAC t = {R['hac_t']:.2f}** (all eras significant), half-life {R['hl_brk']:.1f}d vs "
            f"{R['hl_goog']:.0f}d for the unbounded GOOG control whose spread even flipped sign. "
            "Clears t ≥ 2 on the real tape many times over.\n"
            f"- **Tradability `MIRAGE`** — honest one-lag pairs trade: {R['pairs'][0][1]:+.2f}%/yr "
            f"(t = {R['pairs'][0][2]:+.2f}) at 2 bps, {R['pairs'][1][1]:+.2f}%/yr at 5 bps, "
            f"{R['pairs'][2][1]:+.2f}%/yr at 10 bps; the +{R['pairs_diag'][0]:.1f}%/yr paper number "
            "is a same-print fill artefact. $750k minimum lot, one-way conversion, 35-bps prize.\n"
            f"- **Tradable for A-holders? `BUSTED`** — switch overlay t ≤ "
            f"{max(g[2] for g in R['switch']):.2f} at every threshold in every era under an honest "
            f"fill (vs t = {R['sw_diag'][1]:.2f} at the fill-at-print illusion)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The bound is a free option with one owner class.** Only A-holders can mint B; that "
            "privilege is why enforcement is instant on top and slow underneath (closing the "
            "discount needs slow-moving long-A/short-B capital — Shleifer-Vishny limits in "
            "miniature).\n"
            "- **The mirage generalises.** Any relative-value study with one stale leg (ADR pairs, "
            "A/H twins, holdco stubs) manufactures t > 5 alpha under a same-close fill. Report the "
            "fill convention or report nothing.\n"
            "- **Siblings on the desk.** [05-twin-spread](../../05-twin-spread/README.md) trades "
            "*statistical* pairs with no contractual anchor; "
            "[367-closed-end-fund-discount](../../367-closed-end-fund-discount/README.md) has a NAV "
            "and no conversion; [618-gbtc-premium-cycle](../../618-gbtc-premium-cycle/README.md) "
            "shows what happens when a one-way bridge breaks; "
            "[620-a-h-premium](../../620-a-h-premium/README.md) segments by market access. This "
            "study is the limiting case: a bound that is *contractual*, tested against its own "
            "bridgeless twin.\n\n"
            "*The reproducible core is offline and deterministic; every number here is printed by "
            "[`examples/verify.py`](../examples/verify.py) and frozen in "
            "[`docs/results.md`](../docs/results.md).*"
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
