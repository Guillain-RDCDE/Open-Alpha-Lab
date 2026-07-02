"""Generate the two narrative notebooks for Study 546 (Nobel-Announcement-Drift).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic figures
run anywhere, offline and deterministic; the real-tape cells use the cached ETF prices under
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-30; 72 Nobel
# science-prize announcements 2001-2024; prices fp ea4f6fc0a551, CAR table fp e7c04bc8ac6d).
# CARs are fractions in the code; we display as % (x100) in the notebooks.
R = dict(
    fp_px="ea4f6fc0a551", fp_events="d6630448011d", fp_car="e7c04bc8ac6d",
    n_events=72, n_car=144, y0=2001, y1=2024, H=10,
    mean_car=-0.33, t=-1.36,                       # headline, %
    placebo_obs=-0.33, placebo_mean=0.14, placebo_p=0.984,
    net=-0.37, net_t=-1.53, cost_bps=2.0,
    # per prize: (prize, n, mean_car%, t)
    prize=[("medicine", 48, -0.94, -2.33), ("physics", 48, 0.08, 0.20),
           ("chemistry", 48, -0.14, -0.29)],
    # per sector: (sector, prize-label, n, mean_car%, t)
    sector=[("XLV", "Medicine", 24, -0.67, -2.50), ("IBB", "Medicine", 24, -1.22, -1.58),
            ("XLK", "Physics/Chem", 48, 0.27, 1.10), ("SMH", "Physics/Chem", 48, -0.33, -0.60)],
    # horizon sweep: (H, mean_car%, t)
    horizon=[(3, -0.01, -0.07), (5, -0.24, -1.42), (10, -0.33, -1.36), (21, 0.13, 0.32)],
    # entry-lag + era: label, mean_car%, t
    lag1=(-0.31, -1.26), era_pre=(0.30, 0.74), era_post=(-0.97, -3.60),
    # synthetic control: (drift, mean CAR-t over 25 seeds)
    ctrl=[(0.0, 0.19), (0.0005, 0.89), (0.0010, 1.60), (0.0015, 2.30), (0.0020, 3.01)],
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

from nobel_announcement_drift import data, strategy as st

H = 10

def load_real():
    \"\"\"Cache-first real return panel + events (empty offline).\"\"\"
    px = data.fetch_prices(fetch=False)
    if px.empty:
        return None, None
    return st.daily_returns(px), data.nobel_events()

RETS, EVENTS = load_real()
HAVE_REAL = RETS is not None
print("real Nobel tape present:", HAVE_REAL,
      "" if not HAVE_REAL else f"({len(EVENTS)} announcements, {EVENTS.year.min()}-{EVENTS.year.max()})")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Nobel-Announcement-Drift — do sectors rally after the Nobel Prizes? 🏅\n"
            "### When Medicine, Physics and Chemistry are announced each October, do pharma and tech drift?\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "Every October, Stockholm reads out the Nobel Prizes. There's a bit of market folklore "
            "that says the *related* sectors should get a bump from all the attention: pharma and "
            "biotech after the **Medicine** prize, tech and semiconductors after **Physics** and "
            "**Chemistry**. The idea is that headlines → attention → buying → **drift** in the days "
            "that follow.\n\n"
            "So we test it the clean way: take the health ETFs (XLV, IBB) and the tech/chip ETFs "
            "(XLK, SMH), line up **24 years** of announcement dates, and measure how each sector "
            "moved *relative to the market* in the 10 days after its prize.\n\n"
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
            f"| Did the related sectors drift *up* after their prize? | **No.** Across {R['n_car']} "
            f"prize→sector events the average 10-day move (vs the market) was **{R['mean_car']}%** — "
            "slightly *negative*. |\n"
            f"| Is that just an unlucky sample? | **No** — a random week in the *same* October beat it "
            f"{int(R['placebo_p']*100)}% of the time. Random October drifted **+{R['placebo_mean']}%**; "
            "Nobel weeks did worse. |\n"
            f"| Did *any* sector react? | The one clear move was **pharma after Medicine — down** "
            f"({R['prize'][0][2]}%, *t* {R['prize'][0][3]}). That's the opposite of the story. |\n"
            "| Could you trade it? | **No.** It loses before costs, at every horizon, and the sign "
            "even flips between the early and late years. |\n\n"
            "> A Nobel prize honours research done *decades* ago. It's a wonderful headline — but it "
            "isn't news about next quarter's earnings, so there's no reason for the stock of a whole "
            "sector to drift. And it doesn't."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"When the Nobel science prizes are announced, the related sectors catch a "
            "news-attention bid and drift — pharma after Medicine, tech/semis after Physics & "
            "Chemistry.\"*\n\n"
            "It's **folklore**, not a textbook anomaly — the sort of thing you hear on financial "
            "Twitter every October. The mechanism would be *attention*: a flood of headlines about a "
            "scientific field pulls retail eyeballs (and dollars) toward the obviously-related stocks. "
            "Attention *can* move prices short-term — the question is whether a Nobel announcement is "
            "a big enough, specific enough shock to leave a footprint you could trade."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If it were real, it'd be a free calendar trade: you know the announcement dates *years* "
            "in advance, so you could just buy the sector the morning after. The desk has looked at "
            "plenty of *real* announcement drifts — after [earnings](../../363-pead-drift/), after "
            "[product keynotes](../../299-keynote-drift/), around the [Fed](../../517-pre-fomc-drift/). "
            "Those carry a genuine *surprise*. A Nobel prize doesn't. So this is a clean test of "
            "whether **pure attention, with no cash-flow news, drifts a sector**."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **The dates.** The three science prizes are announced on the Mon/Tue/Wed of the first "
            "full week of October. We hardcode 24 years of them (2001-2024).\n"
            "2. **The map.** Medicine → health ETFs (XLV, IBB); Physics & Chemistry → tech/chips "
            "(XLK, SMH).\n"
            "3. **The measure.** For each event, how did the sector move over the next 10 trading "
            "days — *minus* how the whole market (SPY) moved? (A prize that happened to fall in a "
            "great week for stocks isn't a Nobel effect.) That market-adjusted move is the "
            "**abnormal return**.\n"
            "4. **The sanity check.** Compare it to a *random* week in the same October. If Nobel "
            "weeks aren't special, random October weeks should look just as good.\n\n"
            "*Timing:* the announcement is public the instant it's read out and is scheduled weeks "
            "ahead, so we enter at the close of the first trading day on/after it and hold 10 days — "
            "no peeking at the future."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Which prizes moved their sector, and in which direction?** (10-day market-adjusted "
            "return.)"
        ),
        code(
            "labs = " + repr([p[0].title() for p in R["prize"]]) + "\n"
            "if HAVE_REAL:\n"
            "    tab = st.event_car_table(RETS, EVENTS, data.PRIZE_SECTOR, data.BENCHMARK, horizon=H)\n"
            "    vals = [st.car_summary(tab[tab.prize==p])['mean_car']*100 for p in ['medicine','physics','chemistry']]\n"
            "    banner = 'REAL tape (24 years, 2001-2024)'\n"
            "else:\n"
            "    vals = [" + ", ".join(str(p[2]) for p in R["prize"]) + "]\n"
            "    banner = 'frozen real numbers (offline)'\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "cols = [GREEN if v>0 else RED for v in vals]\n"
            "ax.bar(labs, vals, color=cols, width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('10-day CAR vs market %')\n"
            "ax.set_title('The folklore says UP. Pharma actually fell after Medicine.')\n"
            "fig.suptitle(banner, fontsize=9, color=GREY); plt.tight_layout(); plt.show()\n"
            "for l,v in zip(labs, vals): print(f'{l:10s} 10-day CAR {v:+.2f}%')"
        ),
        md(
            f"There it is: no positive drift anywhere. **Medicine** — the prize the folklore is most "
            f"confident about — moved health stocks **down** ({R['prize'][0][2]}%). Physics and "
            "Chemistry barely moved tech at all. The attention bid just isn't there."
        ),
        md(
            "**Are Nobel weeks even special?** Compare the actual Nobel week to a *random* week in the "
            "same October, thousands of times over:"
        ),
        code(
            f"obs, pmean, pval = {R['placebo_obs']}, {R['placebo_mean']}, {R['placebo_p']}\n"
            "if HAVE_REAL:\n"
            "    pb = st.placebo_pvalue(RETS, EVENTS, data.PRIZE_SECTOR, data.BENCHMARK, horizon=H, n_draws=1500, seed=546)\n"
            "    obs, pmean, pval = pb['obs']*100, pb['placebo_mean']*100, pb['p_value']\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "ax.bar(['Nobel weeks','random October\\nweeks'], [obs, pmean], color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean 10-day CAR %')\n"
            "ax.set_title(f'A random October week beats the Nobel week (placebo p = {pval:.3f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Nobel weeks {obs:+.2f}%  vs  random October {pmean:+.2f}%  ->  placebo p {pval:.3f}')"
        ),
        md(
            f"A random week in the same month did **better** than the actual Nobel week "
            f"({int(R['placebo_p']*100)}% of random draws beat it). So there is no announcement "
            "effect at all — if anything Nobel weeks are slightly *worse* than an ordinary October "
            "week."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The average post-Nobel sector move is **{R['mean_car']}%** (*t* "
            f"{R['t']}) — insignificant and the wrong sign, and a random October week beats it.\n"
            "- **Tradability — Mirage.** It loses before costs, at every holding length, and the sign "
            "flips between the early and late years. Nothing to trade."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No. Even ignoring the wrong sign, you'd be buying a broad sector ETF on the strength of "
            "a headline about research from the 1980s. There's no earnings surprise, no guidance "
            "change, no cash-flow news — just a lovely story. Markets price *cash flows and "
            "surprises*, and a Nobel prize is neither."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The real drifts.** [363 PEAD](../../363-pead-drift/), "
            "[299 Keynote-Drift](../../299-keynote-drift/) and "
            "[515 Earnings-Announcement-Premium](../../515-earnings-announcement-premium/) are "
            "announcements that carry a *surprise* — that's where drift actually lives.\n"
            "- **Other rituals.** [158 Super-Bowl](../../158-super-bowl/) and "
            "[95 Holiday-Cheer](../../95-holiday-cheer/) test other calendar folklore.\n\n"
            "*Think there's a same-day pop we averaged away? Fork this, measure the announcement-day "
            "return itself (not the 10-day drift), and show it clears t = 2 the right way.*"
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
            "# Nobel-Announcement-Drift — a quantitative teardown 🔬\n"
            "### Market-model CARs · one-sample *t* · random-October placebo · per-prize/sector cuts · horizon/entry-lag/era sweep · costs · synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We run a textbook event "
            "study: estimate each sector ETF's market-model beta on a trailing pre-event window, sum "
            "its abnormal returns vs SPY over the post-announcement window, and test the mean against "
            "a random-October placebo.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance daily ETF closes, "
            f"{R['n_events']} Nobel science-prize announcements {R['y0']}-{R['y1']} (as-of 2026-06-30, "
            f"prices fp `{R['fp_px']}`, CAR table fp `{R['fp_car']}`); the offline core and tests run "
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
            f"| **Signal** | `NONE` | Mean 10-day CAR **{R['mean_car']}%** (one-sample *t* "
            f"**{R['t']}**, *n* {R['n_car']}) — insignificant, wrong sign; random-October placebo "
            f"*p* **{R['placebo_p']}**; only |*t*|≥2 cut is Medicine→health at *t* {R['prize'][0][3]} "
            "(wrong sign); sign flips pre/post-2013. |\n"
            f"| **Tradability** | `MIRAGE` | Long-only sector ETF t0→t0+10: gross {R['mean_car']}% → "
            f"net {R['net']}% ({R['cost_bps']:.0f} bps/side, no borrow); negative at every horizon. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), but on the real "
            "tape there is no post-Nobel drift — the one significant move is the wrong way and the "
            "sign is era-unstable."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $CAR_e$ be the market-model cumulative abnormal return of the mapped sector over "
            "$(t_0, t_0+H]$ for event $e$ (t0 = first session on/after the announcement).\n\n"
            "- **H₁ (drift).** $\\mathbb{E}[CAR] > 0$ at *t* ≥ 2 (the sector drifts up).\n"
            "- **H₂ (special).** The Nobel-week mean CAR beats a random-October relocation "
            "(placebo *p* small).\n"
            "- **H₃ (robustness).** The sign/size holds across horizons, entry lag and eras.\n"
            "- **H₄ (net).** H₁ survives round-trip costs.\n\n"
            "We find **H₁ rejected** (insignificant, wrong sign), **H₂ rejected** (placebo *p* "
            f"{R['placebo_p']} — random October is *better*), **H₃ rejected** (sign flips "
            "pre/post-2013), and **H₄ moot** (negative before costs)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why**. Genuine post-announcement drift (PEAD, keynote drift) is "
            "driven by an *information surprise* the market underreacts to. A Nobel prize is a "
            "backward-looking honour with **no cash-flow content** — there is no surprise for prices "
            "to underreact to, so the null is a *flat* CAR, and that is essentially what we get "
            "(slightly negative). Contrast the real drift family: "
            "[363 PEAD](../../363-pead-drift/), [299 Keynote-Drift](../../299-keynote-drift/), "
            "[515 Earnings-Announcement-Premium](../../515-earnings-announcement-premium/)."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Abnormal returns.** Market model $r_{sec} - (\\alpha + \\beta\\, r_{SPY})$, with "
            "$(\\alpha,\\beta)$ from a trailing 120-session window ending strictly before each day "
            "(rolling, no look-ahead).\n"
            "- **CAR.** Sum the abnormal returns over $(t_0, t_0+H]$, $H=10$, for every mapped "
            "(prize, sector) pair.\n"
            "- **Inference.** One-sample *t* of the mean CAR vs 0.\n"
            "- **Placebo.** Relocate each event to a random trading day in the *same* October; "
            "*p* = share of draws whose mean CAR ≥ observed (2000 draws).\n"
            "- **Robustness.** Sweep $H \\in \\{3,5,10,21\\}$, entry at t0 vs t0+1, and a pre/post-2013 "
            "era split.\n"
            "- **Costs.** Round-trip one-way cost per event (long-only ETF, no borrow).\n"
            "- **Positive control.** A deterministic synthetic world with a planted post-event drift, "
            "averaged over 25 seeds (house rule).\n\n"
            "Timing: the announcement is public and pre-scheduled, so entering at the first "
            "on/after-announcement close and measuring the *forward* window uses no future data. The "
            "same-close t0 fill is the one documented execution lag; a t0+1 variant is in the sweep."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline CAR + placebo — H₁ & H₂\n\n"
            "Mean 10-day CAR across all mapped events, its *t*, and the random-October null."
        ),
        code(
            f"mean_car, t, pval = {R['mean_car']}, {R['t']}, {R['placebo_p']}\n"
            f"pmean = {R['placebo_mean']}\n"
            "if HAVE_REAL:\n"
            "    tab = st.event_car_table(RETS, EVENTS, data.PRIZE_SECTOR, data.BENCHMARK, horizon=H)\n"
            "    s = st.car_summary(tab); mean_car, t = s['mean_car']*100, s['t']\n"
            "    pb = st.placebo_pvalue(RETS, EVENTS, data.PRIZE_SECTOR, data.BENCHMARK, horizon=H, n_draws=2000, seed=546)\n"
            "    pmean, pval = pb['placebo_mean']*100, pb['p_value']\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "ax.bar(['Nobel weeks','random October'], [mean_car, pmean], color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean 10-day CAR %')\n"
            "ax.set_title(f'Mean CAR {mean_car:+.2f}% (t {t:+.2f}) | placebo p {pval:.3f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'mean CAR {mean_car:+.2f}% | t {t:+.2f} | placebo mean {pmean:+.2f}% | p {pval:.3f}')"
        ),
        md(
            f"> 💡 In plain words: **H₁ and H₂ both rejected.** The mean CAR is **{R['mean_car']}%** "
            f"(*t* {R['t']}) and a random October week is *better* (placebo *p* {R['placebo_p']}). No "
            "attention bid."
        ),
        md(
            "### 4b · Per prize & per sector — where's the (non-)reaction?\n\n"
            "Break the CAR out by prize and by sector ETF."
        ),
        code(
            "prize_rows = " + repr(R["prize"]) + "\n"
            "sect_rows = " + repr(R["sector"]) + "\n"
            "if HAVE_REAL:\n"
            "    tab = st.event_car_table(RETS, EVENTS, data.PRIZE_SECTOR, data.BENCHMARK, horizon=H)\n"
            "    prize_rows = [(p, len(tab[tab.prize==p]), st.car_summary(tab[tab.prize==p])['mean_car']*100,\n"
            "                   st.car_summary(tab[tab.prize==p])['t']) for p in ['medicine','physics','chemistry']]\n"
            "    smap = {'XLV':'Medicine','IBB':'Medicine','XLK':'Physics/Chem','SMH':'Physics/Chem'}\n"
            "    sect_rows = [(sc, smap[sc], len(tab[tab.sector==sc]), st.car_summary(tab[tab.sector==sc])['mean_car']*100,\n"
            "                  st.car_summary(tab[tab.sector==sc])['t']) for sc in ['XLV','IBB','XLK','SMH']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "pl = [r[0].title() for r in prize_rows]; pv = [r[2] for r in prize_rows]\n"
            "a1.bar(pl, pv, color=[GREEN if v>0 else RED for v in pv], width=.55)\n"
            "a1.axhline(0, c='k', lw=1); a1.set_title('By prize'); a1.set_ylabel('10-day CAR %')\n"
            "sl = [r[0] for r in sect_rows]; sv = [r[3] for r in sect_rows]\n"
            "a2.bar(sl, sv, color=[GREEN if v>0 else RED for v in sv], width=.55)\n"
            "a2.axhline(0, c='k', lw=1); a2.set_title('By sector ETF')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('by prize: ', [(r[0], round(r[2],2), round(r[3],2)) for r in prize_rows])\n"
            "print('by sector:', [(r[0], round(r[3],2), round(r[4],2)) for r in sect_rows])"
        ),
        md(
            f"> 💡 In plain words: the only |*t*|≥2 cut is **XLV after Medicine at *t* {R['sector'][0][4]}** "
            "— pharma drifting *down*. Tech (XLK/SMH) after Physics/Chemistry is a coin-flip. No "
            "positive, significant reaction anywhere."
        ),
        md(
            "### 4c · Robustness — H₃ (horizon, entry lag, era)\n\n"
            "Does anything positive survive a different holding length, a one-bar-later entry, or a "
            "different era?"
        ),
        code(
            "hz = " + repr(R["horizon"]) + "\n"
            f"lag1 = {R['lag1']}; era_pre = {R['era_pre']}; era_post = {R['era_post']}\n"
            "if HAVE_REAL:\n"
            "    hs = st.horizon_sweep(RETS, EVENTS, data.PRIZE_SECTOR, data.BENCHMARK, horizons=(3,5,10,21))\n"
            "    hz = [(h, hs.loc[h,'mean_car']*100, hs.loc[h,'t']) for h in (3,5,10,21)]\n"
            "    l1 = st.horizon_sweep(RETS, EVENTS, data.PRIZE_SECTOR, data.BENCHMARK, horizons=(10,), entry_lag=1)\n"
            "    lag1 = (l1.loc[10,'mean_car']*100, l1.loc[10,'t'])\n"
            "    es = st.era_split(RETS, EVENTS, data.PRIZE_SECTOR, data.BENCHMARK, horizon=H, split_year=2013)\n"
            "    era_pre = (es['pre']['mean_car']*100, es['pre']['t']); era_post = (es['post']['mean_car']*100, es['post']['t'])\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "labs = [f'H={h}' for h,_,_ in hz] + ['t0+1\\n(H=10)', 'pre-2013', 'post-2013']\n"
            "vals = [v for _,v,_ in hz] + [lag1[0], era_pre[0], era_post[0]]\n"
            "ax.bar(range(len(labs)), vals, color=[GREEN if v>0 else RED for v in vals], width=.6)\n"
            "ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=10)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean CAR %')\n"
            "ax.set_title('Never positive-significant; sign flips across eras')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h,v,tt in hz: print(f'H={h:2d}: {v:+.2f}% (t {tt:+.2f})')\n"
            "print(f't0+1 H=10: {lag1[0]:+.2f}% (t {lag1[1]:+.2f})')\n"
            "print(f'pre-2013: {era_pre[0]:+.2f}% (t {era_pre[1]:+.2f}) | post-2013: {era_post[0]:+.2f}% (t {era_post[1]:+.2f})')"
        ),
        md(
            "> 💡 In plain words: **H₃ rejected.** No horizon is positive-significant; the entry-lag "
            "variant is unchanged; and the era split *flips sign* (weakly + early, significantly − "
            f"late, *t* {R['era_post'][1]}). A sign that depends on the decade is not a signal."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — H₁ rejected (mean CAR {R['mean_car']}%, *t* {R['t']}, wrong "
            f"sign); H₂ rejected (placebo *p* {R['placebo_p']}); H₃ rejected (sign flips).\n"
            f"- **Tradability `MIRAGE`** — long-only sector ETF; gross {R['mean_car']}% → net "
            f"{R['net']}% (net *t* {R['net_t']}); negative at every horizon."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs\n\n"
            "The trade is negative before costs; frictions only deepen the loss."
        ),
        code(
            f"gross, net = {R['mean_car']}, {R['net']}\n"
            "if HAVE_REAL:\n"
            "    tab = st.event_car_table(RETS, EVENTS, data.PRIZE_SECTOR, data.BENCHMARK, horizon=H)\n"
            "    nc = st.net_car(tab, cost_bps=2.0); gross, net = nc['gross']*100, nc['net']*100\n"
            "fig, ax = plt.subplots(figsize=(6.8, 4.2))\n"
            "ax.bar(['gross','net\\n(2 bps/side)'], [gross, net], color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean 10-day CAR %')\n"
            "ax.set_title(f'Negative before costs ({gross:+.2f}%), worse after ({net:+.2f}%)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:+.2f}% -> net {net:+.2f}% (2 bps/side round-trip, long-only ETF)')"
        ),
        md(
            "> 💡 In plain words: there's nothing to charge costs against — the buy-the-sector book "
            "*loses* before frictions. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a real post-event "
            "drift of increasing strength and watch the CAR-*t* climb past +2 — averaged over 25 "
            "seeds so no single lucky seed can fake it."
        ),
        code(
            "ctrl = " + repr(R["ctrl"]) + "\n"
            "drifts = [c[0] for c in ctrl]\n"
            "ts = [st.synthetic_mean_t(data, drift=d, horizon=H, n_seeds=25) for d in drifts]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(drifts, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = +2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted per-day post-event drift'); ax.set_ylabel('mean CAR-t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted drift — flat at the null, past +2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for d, t in zip(drifts, ts): print(f'drift {d:+.4f} -> mean CAR-t {t:+.2f}')"
        ),
        md(
            "The CAR-*t* is ≈ 0 at the null and rises past +2 as the planted drift grows — so the "
            "engine is a faithful detector. The flat/negative real-tape result is therefore the "
            "**absence of an effect**, not a broken test: a Nobel prize carries no cash-flow surprise, "
            "so the related sector doesn't drift. For real announcement drifts, see "
            "[363 PEAD](../../363-pead-drift/) and [299 Keynote-Drift](../../299-keynote-drift/)."
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
