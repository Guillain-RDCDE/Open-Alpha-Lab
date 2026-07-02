"""Generate the two narrative notebooks for Study 563 (Secondary-Offering-Drift).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached event prices under
../_cache/ (the 27 offering tickers + SPY) and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance abnormal (stock-SPY)
# drift after 34 notable seasoned/secondary offerings + SPY, 2008-01-02 -> 2026-06-30, panel fp
# 75dc79931e2c). The claim predicts a NEGATIVE drift; the tape delivers a POSITIVE one.
R = dict(
    start="2008-01-02", end="2026-06-30", fp="75dc79931e2c", n_events=34,
    # per-horizon: (months, n, ab_mean%, ab_median%, loss%, t, p_placebo_left)
    h1=(1, 34, 10.60, 0.46, 50, 1.73, 0.985),
    h3=(3, 34, 7.83, 3.34, 47, 1.32, 0.737),
    h6=(6, 34, 8.21, 6.19, 41, 0.99, 0.531),
    h12=(12, 34, 29.44, 3.71, 47, 1.78, 0.563),
    # short-the-issuer P&L: (months, short_gross%, short_net%)
    cost=[(1, -10.60, -10.95), (3, -7.83, -8.68), (6, -8.21, -9.81)],
    # size split at 6m: (label, n, ab6_mean%, t, p)
    size=[("BIG (>=median $1.08bn)", 17, 6.73, 0.84, 0.477),
          ("SMALL (<median)", 17, 9.69, 0.65, 0.601)],
    # synthetic control (25 seeds), 6m: (planted_edge, mean_t, ab6_mean%)
    syn=[(0.0, 0.68, 4.50), (-0.10, -0.20, -0.60), (-0.20, -1.13, -5.45), (-0.30, -2.10, -10.06)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Dilution_drift%3F: Busted](https://img.shields.io/badge/Dilution_drift%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from secondary_offering_drift import data, strategy as st

HAVE_REAL = data.have_real()
PRICES, EVENTS = data.load_real() if HAVE_REAL else (None, None)
if HAVE_REAL:
    PRICES = PRICES.loc[:"2026-06-30"]           # drop the partial current bar (as-of stamp)
print("real event-price cache present:", HAVE_REAL,
      "| offerings in table:", len(data.event_table()))
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Secondary-Offering-Drift — does a stock keep sliding after it sells fresh shares? 💧\n"
            "### When a company prints new stock to raise cash, in plain English\n\n"
            + BADGES +
            "Here's a piece of market folklore that *sounds* airtight. When a public company needs "
            "cash, it can **sell a fresh slug of new shares** to the market — a *secondary* or "
            "*follow-on* offering. Two things are supposed to happen: (1) **dilution** — the pie is "
            "now cut into more slices, so each existing share is worth a little less; and (2) a "
            "**bad signal** — management chose to sell equity *now*, which whispers 'we think our "
            "stock is about as expensive as it's going to get.' Put together, the stock is supposed "
            "to **drift down for months** afterward.\n\n"
            "So we test it the cheap way: take **34** notable, recognisable offerings, and see "
            "whether the stock really slid in the months after — *beyond* whatever the whole market "
            "was doing.\n\n"
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
            f"| Did the stock slide after the offering (the claim)? | **No — the opposite.** Over the "
            f"next month the issuers *beat* the market by **+{R['h1'][2]:.0f}%** on average; over the "
            f"next *year*, by **+{R['h12'][2]:.0f}%**. The 'diluted' stocks *ripped*. |\n"
            "| Is that rip a real edge, then? | **No** — it never clears the significance bar (best "
            f"*t* = +{R['h12'][5]}), and it's driven by a handful of moonshots. Positive on average, "
            "but inside the noise. |\n"
            "| So is the dilution story just wrong? | **On this basket, backwards.** The names we can "
            "recognise and name are growth-story high-fliers (Tesla, crypto miners, pandemic winners) "
            "that raised cash *at strength* — and kept climbing. |\n"
            "| Could you trade it (short the issuer)? | **No.** Shorting these names *loses* money "
            "before costs, and they're exactly the expensive-to-borrow stocks. |\n\n"
            "> The dilution effect is real in the textbooks (and there *is* a small negative pop on "
            "the *day* of the announcement). But the months-long *slide* the folklore promises isn't "
            "here — our sample is survivors and headliners that grew into their new shares."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"After a company sells a fresh slug of shares in a secondary offering, the stock keeps "
            "sliding.\"* — the dilution + negative-signalling story (Loughran-Ritter 1995; "
            "Myers-Majluf 1984).\n\n"
            "Why it *should* be true: more shares outstanding means each one owns a smaller piece of "
            "the same company (dilution), and a manager who sells equity is implicitly saying the "
            "stock is fully priced (a bearish signal). Both point the same way — down. The academic "
            "'new issues puzzle' even documents issuers *under*performing for years afterward."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If true, it's a clean short: when a company you follow prints stock, fade it. It's the "
            "mirror image of the **buyback** story ([Study 368](../../368-buyback-drift/) — buy back "
            "shares, drift up) and the discrete-event cousin of the **net-share-issuance** factor "
            "([Study 519](../../519-net-share-issuance/) — diluters underperform). We go straight at "
            "the *event*: mark each offering, and look at what the stock did next — beyond the market."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Mark the offerings.** 34 notable follow-on / secondary raises (ticker, pricing "
            "date, size).\n"
            "2. **Strip the market.** For each, measure the stock's return *minus SPY's* over the "
            "next 1 / 3 / 6 / 12 months — the **abnormal** return. That way a bull (or bear) market "
            "can't masquerade as offering drift.\n"
            "3. **Enter one day later.** We act the day *after* the offering is public — no peeking, "
            "and it steps past the small same-day pop so we measure the *drift*, not the jump.\n\n"
            "Catch: our list is companies we can *recognise* — which skews to survivors and headline "
            "growth names. The messy distressed dilutions, where the bad signal bites hardest, are "
            "under-represented. We'll see that bias flip the whole result."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Did the stock slide (claim) or climb (tape) after the offering, beyond the market?**"
        ),
        code(
            "hor = [1, 3, 6, 12]\n"
            "if HAVE_REAL:\n"
            "    means = [st.summarize(PRICES, EVENTS, m)['ab_mean']*100 for m in hor]\n"
            "    banner = 'REAL tape (34 offerings, abnormal vs SPY)'\n"
            "else:\n"
            f"    means = [{R['h1'][2]}, {R['h3'][2]}, {R['h6'][2]}, {R['h12'][2]}]\n"
            "    banner = 'frozen real numbers (offline)'\n"
            "fig, ax = plt.subplots(figsize=(8, 4.4))\n"
            "cols = [GREEN if m>0 else RED for m in means]\n"
            "ax.bar([f'{m}m' for m in hor], means, color=cols, width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('abnormal drift %  (stock - SPY)')\n"
            "ax.set_title('The claim says DOWN. Every horizon drifted UP (issuers beat the market)')\n"
            "fig.suptitle(banner, fontsize=9, color=GREY); plt.tight_layout(); plt.show()\n"
            "print('abnormal drift by horizon:', [round(m,1) for m in means])"
        ),
        md(
            f"There it is, **backwards**. The dilution story predicts a slide. On these 34 offerings "
            f"the issuers *beat* the market at every horizon — **+{R['h1'][2]:.0f}%** at one month, "
            f"**+{R['h12'][2]:.0f}%** at a year. The 'diluted' names climbed. Why? The recognisable "
            "offerings are growth-story high-fliers raising cash while their stock was hot — and they "
            "stayed hot."
        ),
        md(
            "**Is that climb a real signal, or a few moonshots dragging the average up?** Compare the "
            "*average* drift to the *typical* (median) one, and count how often the stock actually "
            "fell (the 'loss-rate' — a real down-drift would push it above 50%)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s6 = st.summarize(PRICES, EVENTS, 6)\n"
            "    mean6, med6, loss6 = s6['ab_mean']*100, s6['ab_median']*100, s6['loss']*100\n"
            "else:\n"
            f"    mean6, med6, loss6 = {R['h6'][2]}, {R['h6'][3]}, {R['h6'][4]}\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "ax.bar(['mean drift','median drift'], [mean6, med6], color=[GREEN, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('6-month abnormal drift %')\n"
            "ax.set_title(f'6m: mean +{mean6:.0f}% but stock fell only {loss6:.0f}% of the time (~coin flip)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'6m mean +{mean6:.1f}% | median +{med6:.1f}% | fell {loss6:.0f}% of the time')"
        ),
        md(
            f"The stock fell only **{R['h6'][4]:.0f}%** of the time at six months — a coin flip, if "
            "anything *below* it. The mean is dragged up by a few explosive winners; there is no "
            "reliable down-drift to short. Whatever the textbooks say, *this* basket did not slide."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The drift is the *wrong sign* (up, not down) at every horizon and "
            "never clears the significance bar. No slide to bank.\n"
            "- **Tradability — Mirage.** The trade would be to *short* the issuer — which *lost* money "
            "here, before you even pay the (steep) borrow on these hard-to-borrow names.\n"
            "- **Dilution drift on the tape? — Busted.** On recognisable, survived growth-story "
            "raisers, the issuers out-ran the market for a year. The effect the folklore promises is "
            "absent-to-inverted on this sample."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No. The tradable version of 'the stock slides' is to **short the issuer** — but on this "
            "basket that short *loses* money, because the stocks went up. And these are exactly the "
            "kind of hot, hard-to-borrow names (Tesla, crypto miners) where shorting is expensive and "
            "dangerous. Wrong sign, worst possible names to be short. There's no trade here."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The factor version.** [Study 519 — Net-Share-Issuance](../../519-net-share-issuance/) "
            "tests the *continuous* version (rank firms by how much stock they printed) on a broad "
            "basket.\n"
            "- **The mirror image.** [Study 368 — Buyback-Drift](../../368-buyback-drift/): does a "
            "stock drift *up* after a buyback? (Faintly, but not significantly either.)\n"
            "- **The real test needs the ugly deals.** Re-run this with *distressed* dilutions "
            "(near-bankruptcy raises, deep-discount rights issues) and a survivorship-free universe — "
            "that's where the negative signal should bite.\n\n"
            "*Think the slide is real and we just cherry-picked winners? Fork this, add 100 messy "
            "small-cap distressed offerings, and show the abnormal drift clearing t = −2 the way the "
            "dilution story predicts.*"
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
            "# Secondary-Offering-Drift — a quantitative teardown 🔬\n"
            "### abnormal-return event study · forward drift vs zero · one-sample Welch *t* · same-names left-tail placebo · size-split cut · short-the-issuer costs & borrow · synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We mark 34 notable "
            "follow-on / secondary offerings and test whether the issuer drifts **down** (dilution + "
            "negative signal) — or, here, up.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance daily adjusted closes for 27 offering "
            f"tickers + SPY (as-of 2026-06-30, panel fp `{R['fp']}`); the offline core and tests run "
            "on a deterministic synthetic world. Methods in "
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
            f"| **Signal** | `NONE` | Abnormal drift **+{R['h1'][2]:.0f} / +{R['h3'][2]:.0f} / "
            f"+{R['h6'][2]:.0f} / +{R['h12'][2]:.0f}%** at 1/3/6/12m — the *wrong* sign (claim = down); "
            f"best *t* **+{R['h12'][5]}** (< 2); left-tail placebo *p* {R['h6'][6]}–{R['h1'][6]}. |\n"
            f"| **Tradability** | `MIRAGE` | Short-the-issuer P&L **{R['cost'][2][1]:+.0f}%** gross / "
            f"**{R['cost'][2][2]:+.0f}%** net at 6m (10 bps + 300 bps/yr borrow). Wrong sign before "
            "costs. |\n"
            f"| **Dilution drift on the tape?** | `BUSTED` | Positive drift in both size buckets; "
            "issuers out-ran SPY for a year. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), but on this "
            "hand-picked roster of recognisable growth-story raisers the dilution drift is "
            "absent-to-inverted and never significant."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $a_i$ be offering $i$'s **abnormal** forward return (stock − SPY) over horizon $H$.\n\n"
            "- **H₁ (the drift).** $\\mathbb{E}[a] < 0$ at *t* ≤ −2 (issuers slide beyond the "
            "market).\n"
            "- **H₂ (placebo).** The offering set sits in the *left* tail of a same-names "
            "random-date null.\n"
            "- **H₃ (dose-response).** *Bigger* offerings ⇒ *more* negative drift.\n"
            "- **H₄ (tradable).** Shorting the issuer, net of borrow, is profitable.\n\n"
            "We find **H₁ rejected with the sign reversed** (drift is positive), **H₂ rejected** "
            "(the set sits in the *right* tail), **H₃ rejected** (both size halves drift up), and "
            "**H₄ rejected** (the short loses before costs)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why**. The dilution/new-issues effect is documented on *broad, "
            "survivorship-free* universes and over *long* horizons, and driven partly by the "
            "distressed, deep-discount raises. Two forces kill it here: (a) **selection** — a "
            "hand-picked table of recognisable offerings skews to *survivors* and *growth stories* "
            "raising at strength; and (b) the discrete-event framing captures a handful of moonshots "
            "(Tesla's capital raises, crypto-miner ATMs) whose post-raise rallies swamp the average. "
            "The continuous factor on a full basket lives in "
            "[519 Net-Share-Issuance](../../519-net-share-issuance/); the mirror event in "
            "[368 Buyback-Drift](../../368-buyback-drift/)."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Event.** 34 notable follow-on / secondary offerings (ticker, pricing date, $bn).\n"
            "- **Measure.** Abnormal return $a = (P^{stock}_H/P^{stock}_0 - 1) - "
            "(P^{SPY}_H/P^{SPY}_0 - 1)$ at $H \\in \\{1,3,6,12\\}$ months.\n"
            "- **Entry.** One trading day *after* the offering is public (no look-ahead; past the "
            "announcement-day pop).\n"
            "- **Inference.** One-sample Welch *t* vs 0; a **same-names left-tail placebo** null "
            "(20,000 random-date redraws, ask $P[\\text{placebo mean} \\le \\text{offering mean}]$).\n"
            "- **Robustness.** Median split on deal size.\n"
            "- **Frictions.** The tradable trade is a **short**: P&L $= -a$, charged a round-trip "
            "cost + a pro-rated 300 bps/yr borrow.\n"
            "- **Positive control.** A deterministic synthetic world with a planted *negative* drift "
            "and a null — averaged over 25 seeds (house rule)."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The drift vs zero — H₁ & H₂\n\n"
            "Abnormal drift at each horizon with the one-sample Welch *t* and the left-tail placebo *p*."
        ),
        code(
            "hor = [1, 3, 6, 12]\n"
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(PRICES, EVENTS, m) for m in hor]\n"
            "    tab = [(f'{m}m', s['n'], s['ab_mean']*100, s['ab_median']*100, s['loss']*100, s['t'], s['p_placebo'])\n"
            "           for m, s in zip(hor, rows)]\n"
            "else:\n"
            f"    tab = [('1m',)+{R['h1'][1:]}, ('3m',)+{R['h3'][1:]}, ('6m',)+{R['h6'][1:]}, ('12m',)+{R['h12'][1:]}]\n"
            "import pandas as pd\n"
            "df = pd.DataFrame(tab, columns=['H','n','mean%','median%','loss%','t','placebo_p']).set_index('H')\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "cols = [GREEN if m>0 else RED for m in df['mean%']]\n"
            "ax1.bar(df.index, df['mean%'], color=cols, width=.6); ax1.axhline(0, c='k', lw=1)\n"
            "ax1.set_ylabel('abnormal drift %'); ax1.set_title('Drift is POSITIVE (claim = negative)')\n"
            "ax2.bar(df.index, df['t'], color=GREY, width=.6)\n"
            "ax2.axhline(2, ls='--', c=GREEN, lw=1); ax2.axhline(-2, ls='--', c=RED, lw=1); ax2.axhline(0, c='k', lw=1)\n"
            "ax2.set_ylabel('Welch t vs 0'); ax2.set_title('|t| < 2 everywhere (no significance either way)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(df.round(3).to_string())"
        ),
        md(
            f"> 💡 In plain words: **H₁ rejected with the sign reversed** and **H₂ rejected.** The "
            f"drift is *positive* (+{R['h1'][2]:.0f}% to +{R['h12'][2]:.0f}%), *t* never leaves "
            f"(−2, +2), and the left-tail placebo *p* is {R['h6'][6]}–{R['h1'][6]} — the offering set "
            "sits on the *high* side of the random-date distribution, the opposite of a slide."
        ),
        md(
            "### 4b · Dose-response — H₃ (does a bigger offering slide more?)\n\n"
            "Median-split on deal size; the dilution story predicts the BIG bucket slides *more*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = st.size_split(PRICES, EVENTS, 6)\n"
            "    labs = [r['bucket'] for r in rows]; vals = [r['ab_mean']*100 for r in rows]\n"
            "    ts = [r['t'] for r in rows]\n"
            "else:\n"
            f"    labs = [{R['size'][0][0]!r}, {R['size'][1][0]!r}]\n"
            f"    vals = [{R['size'][0][2]}, {R['size'][1][2]}]; ts = [{R['size'][0][3]}, {R['size'][1][3]}]\n"
            "fig, ax = plt.subplots(figsize=(8, 4.2))\n"
            "ax.bar(range(len(labs)), vals, color=[GREEN if v>0 else RED for v in vals], width=.5)\n"
            "ax.set_xticks(range(len(labs))); ax.set_xticklabels([l.split(' ')[0] for l in labs])\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('6m abnormal drift %')\n"
            "ax.set_title('Both size buckets drift UP — dose-response fails')\n"
            "plt.tight_layout(); plt.show()\n"
            "for l, v, t in zip(labs, vals, ts): print(f'{l:>22}: 6m drift {v:+.1f}%  (t {t:+.2f})')"
        ),
        md(
            "> 💡 In plain words: **H₃ rejected.** Both halves drift up (BIG "
            f"+{R['size'][0][2]:.0f}%, SMALL +{R['size'][1][2]:.0f}%), neither significant. 'More "
            "shares sold ⇒ bigger slide' is not in this data."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — H₁ rejected (drift +{R['h6'][2]:.0f}% at 6m, wrong sign); H₂ "
            f"rejected (right-tail placebo *p* {R['h6'][6]}); best *t* +{R['h12'][5]} (< 2).\n"
            f"- **Tradability `MIRAGE`** — short-the-issuer P&L {R['cost'][2][1]:+.0f}% gross → "
            f"{R['cost'][2][2]:+.0f}% net at 6m (wrong sign before the borrow).\n"
            "- **Dilution drift on the tape? `BUSTED`** — positive in both size buckets; issuers "
            "out-ran SPY for a year."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the short, costs and the borrow\n\n"
            "The tradable expression is a **short** of the issuer, so its P&L is $-a$. Because the "
            "drift is positive, the short loses; the punitive borrow only deepens it."
        ),
        code(
            "hor = [1, 3, 6]\n"
            "if HAVE_REAL:\n"
            "    gross = [st.net_of_costs(PRICES, EVENTS, m)['short_gross']*100 for m in hor]\n"
            "    net = [st.net_of_costs(PRICES, EVENTS, m)['short_net']*100 for m in hor]\n"
            "else:\n"
            f"    gross = [{R['cost'][0][1]}, {R['cost'][1][1]}, {R['cost'][2][1]}]\n"
            f"    net = [{R['cost'][0][2]}, {R['cost'][1][2]}, {R['cost'][2][2]}]\n"
            "x = np.arange(len(hor)); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(8, 4.2))\n"
            "ax.bar(x-w/2, gross, w, label='short gross', color=RED)\n"
            "ax.bar(x+w/2, net, w, label='short net (cost+borrow)', color=GREY)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels([f'{m}m' for m in hor])\n"
            "ax.set_ylabel('short-the-issuer P&L %'); ax.legend()\n"
            "ax.set_title('Shorting the issuer LOSES at every horizon (wrong sign before costs)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for m, g, n in zip(hor, gross, net): print(f'{m}m: short gross {g:+.1f}% -> net {n:+.1f}%')"
        ),
        md(
            "> 💡 In plain words: there is nothing to charge costs against — the short *loses* before "
            "frictions, and these are the worst possible names to be short. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector of a *real* downside drift, or does it always print "
            "~0? Plant a genuine negative drift of increasing strength and watch the Welch *t* go "
            "negative — averaged over 25 seeds so no single lucky seed can fake it."
        ),
        code(
            "edges = [0.0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30]\n"
            "rows = [st.synthetic_control(data.synthetic_events, edges=(e,), months=6, n_seeds=25)[0] for e in edges]\n"
            "ts = [r['mean_t'] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(edges, ts, 'o-', c=RED, lw=2)\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1, label='t = -2 bar (downside drift)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted abnormal drift (more negative = stronger slide)')\n"
            "ax.set_ylabel('mean Welch t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted slide — flat at the null, past -2 only for a HUGE drift')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for e, t in zip(edges, ts): print(f'planted {e:+.2f} -> mean t {t:+.2f}')"
        ),
        md(
            "The seed-averaged *t* is ≈ +0.7 at the null and only falls past −2 when the planted "
            "slide is a huge −30% over six months — so the engine is a faithful detector, **and** ~34 "
            "single-name events are too few to catch any slide of plausible size (at the null, some "
            "individual seeds still cross |t| = 2 by luck, which is exactly why we average). The "
            "real-tape result is therefore a statement about **this survivor/visibility-selected "
            "basket**: the dilution drift is absent-to-inverted here. For the factor and the mirror "
            "event, see [519 Net-Share-Issuance](../../519-net-share-issuance/) and "
            "[368 Buyback-Drift](../../368-buyback-drift/)."
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
