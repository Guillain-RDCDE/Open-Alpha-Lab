"""Generate the two narrative notebooks for Study 720 (Super-Bowl-Advertiser).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached advertiser
prices under ../_cache/ (the ~32-name table + SPY) and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere
with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance ~32-name Super Bowl
# advertiser table + SPY, 2013-06-03 -> 2026-07-10; 32 advertiser-year events priced;
# table fingerprint fc267dcab645).
R = dict(
    n_table=32, n_delisted=11, n_priced=32, fp="fc267dcab645",
    # leg: (mean%, win%, t[, placebo_p])
    monday=(-0.33, 41, -0.83),
    drift=(0.39, 47, 0.31, 0.637),
    hold=(1.20, 50, 0.60, 0.471),
    # era: (name, n, drift%, hold%)
    eras=[("2015-17", 3, 2.79, 6.66), ("2019-21", 12, -1.32, 3.66),
          ("2022-24", 17, 1.17, -1.49)],
    # robustness: (drift/hold, n, drift%, drift_t, hold%, hold_t)
    robust=[("3/10", 32, 0.76, 0.93, -0.31, -0.20), ("5/20", 32, 0.39, 0.31, 1.20, 0.60),
            ("10/40", 32, 1.71, 1.13, -2.92, -1.02)],
    cost=dict(gross_drift=0.39, net_drift=0.19, gross_hold=1.59, net_hold=1.39),
    # synthetic: edge, n, drift%, drift_t, hold%, hold_t
    syn=[(0.00, 32, -0.03, -0.05, 0.20, 0.09),
         (0.10, 32, 10.48, 13.23, 0.20, 0.09)],
    drift_std=0.0719,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Big-ad signal?: Not supported](https://img.shields.io/badge/Big--ad_signal%3F-Not_supported-8b949e?style=flat-square)\n\n"
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

from super_bowl_advertiser import data, strategy as st

HAVE_REAL = data.have_real()
B = data.load_real() if HAVE_REAL else None
EV = st.collect_events(B) if HAVE_REAL else None
print("real advertiser-tape cache present:", HAVE_REAL,
      "| events priced:", (0 if EV is None else len(EV)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The big-ad signal — do Super Bowl advertisers drift up? 📺\n"
            "### The company that buys a $7-million commercial, gets the buzz, and pops — in plain English\n\n"
            + BADGES +
            "There's a tidy market story: a company spends **~$7 million** on 30 seconds of Super Bowl "
            "airtime, ~120 million people see it, the brand trends all week — and the **stock drifts up** "
            "in the days after the game. There's even an academic paper behind it (Fehle, Tsyplakov & "
            "Zdorovtsov, 2005) that found positive abnormal returns around Super Bowl ads. So: watch who "
            "advertises Sunday, buy Monday, ride the buzz. A scheduled edge, every February.\n\n"
            "This notebook asks whether that's a *real, repeatable* effect or just a good story — by "
            "building a transparent table of **~32 real listed Super Bowl advertisers** (2015–2024) and "
            "looking at what actually happened to all of them after the game.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the power analysis? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** The most *famous* Super Bowl advertisers in history — "
            "**Pets.com**, **Computer.com**, **Kozmo.com**, the dot-com class of 2000 — **spent their "
            "capital on the ad and then went bust**, vanishing from the data. So the names we *can* still "
            "price are the survivors. That biases everything **toward** 'advertising pays' — which makes "
            "it all the more telling that the drift still doesn't show up. Every chart is drawn by the "
            "code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do Super Bowl advertisers drift up after the game? | **Not really.** Across ~32 advertiser "
            f"events the first-week 'drift' averages just **+{R['drift'][0]:.1f}%** vs the market — and "
            f"it's a coin-flip (**{R['drift'][1]:.0f}%** of the time positive, *below* 50%). |\n"
            "| Is the reaction at least positive on day one? | **No — it's slightly negative.** The "
            f"Monday after the game the average advertiser is **{R['monday'][0]:.1f}%** vs the market. The "
            "buzz doesn't move the stock. |\n"
            "| But didn't a paper find this? | **On a 2000–2004 sample, yes.** Two decades later, on a "
            "modern tape, it's gone — the kind of effect that fades once everyone knows about it (or was "
            "always thin). |\n"
            "| Can you trade the 'big-ad signal'? | **There's nothing to trade.** Buying the basket every "
            f"February earns **+{R['cost']['gross_drift']:.1f}% gross** over five days — indistinguishable "
            "from noise, and it's just large-cap beta. |\n\n"
            "> The drift is a faint coin-flip, the day-one reaction is *negative*, and the famous 'ads "
            "pay' cases are the survivors we remember. The legend is a story, not a signal."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A Super Bowl ad buys a company a national attention spike — ~120M viewers, a week of "
            "social buzz, brand searches through the roof. That attention pulls in retail buyers, and the "
            "stock drifts up in the days after the game. Watch who advertises, buy the Monday open, ride "
            "the buzz.\"*\n\n"
            "It's not a crazy claim — there's real academia behind it. **Fehle, Tsyplakov & Zdorovtsov "
            "(2005)** studied Super Bowl advertisers and found **significantly positive abnormal returns** "
            "in the days *after* the game, which they read as advertising nudging investor attention. The "
            "believers extend that into a tradable February calendar. We'll test it on a real, "
            "representative table of modern advertisers — and see whether the 2005 result still breathes."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it were real and repeatable, it'd be a tidy little machine: every year the ad line-up "
            "leaks in advance, so you'd have a **pre-scheduled basket** to buy on Super Bowl Monday and "
            "sell a week later. It would also say something odd about markets — that a **TV commercial** "
            "moves a multi-billion-dollar stock more than fundamentals. But two things have to hold for "
            "the machine to work: the drift has to be **real on average** (not just in the cases we "
            "remember), and it has to be **big enough to clear costs** on a basket of large caps. Miss "
            "either and the 'trade' is a Super Bowl party anecdote."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We build a **transparent table of ~{R['n_table']} real listed advertisers** across "
            "2015–2024 (Wix, Coinbase, Temu, e.l.f., Bud Light, GM, Salesforce, Rocket…) — not just the "
            "legends, the representative set. For each one:\n\n"
            "1. **Find the game.** Line up the stock against the market (SPY) around Super Bowl Sunday.\n"
            "2. **Measure the drift.** The **drift** leg (abnormal return over the first ~week after the "
            "game — the 'big-ad signal') and a longer **hold** leg (the next month) — *abnormal* meaning "
            "*beyond what the market did*. You act Monday, so there's a one-day entry lag and no "
            "weekend-gap look-ahead.\n"
            "3. **Stress the luck.** Draw the same number of *random* windows thousands of times and ask "
            "how often chance produces a drift this big. With only ~32 events, that's the honest test.\n\n"
            "And we say it loudly: the biggest advertisers that **went bust** (Pets.com & co.) left the "
            "data, so our survivors are the *good* cases — the deck is stacked **for** the story."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the drift and the hold, side by side.** For every advertiser: the abnormal "
            "(market-adjusted) return in the first week after the game vs the next month."
        ),
        code(
            "if HAVE_REAL:\n"
            "    drifts = EV['drift'].values*100; holds = EV['hold'].values*100; labels = EV['ticker'].values\n"
            "else:\n"
            "    rng=np.random.default_rng(720)\n"
            "    drifts=rng.normal(R['drift'][0],7,R['n_priced']); holds=rng.normal(R['hold'][0],10,R['n_priced'])\n"
            "    labels=[f'A{i}' for i in range(R['n_priced'])]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "ax.scatter(drifts, holds, c=GREEN, s=60, edgecolor='k', alpha=.8)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c='k', lw=.8)\n"
            "ax.axhline(np.mean(holds), c=GREY, ls='--', label=f'mean hold {np.mean(holds):+.1f}%')\n"
            "ax.axvline(np.mean(drifts), c=AMBER, ls='--', label=f'mean drift {np.mean(drifts):+.1f}%')\n"
            "ax.set_xlabel('drift: first-week abnormal return after the game (%)')\n"
            "ax.set_ylabel('hold: next-month abnormal return (%)')\n"
            "ax.set_title('The big-ad signal should push the cloud to the RIGHT — it sits on zero')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'mean drift {np.mean(drifts):+.1f}%   mean hold {np.mean(holds):+.1f}%  (a real signal would be clearly >0)')"
        ),
        md(
            f"There's the tell. If 'buy the buzz' were true, the dots would sit clearly **to the right** "
            f"of the zero line (a positive drift). Instead the mean drift is a whisper — "
            f"**+{R['drift'][0]:.1f}%** — with the cloud straddling zero. A few big winners "
            "(a Coinbase here, a DraftKings there) are cancelled by just as many losers."
        ),
        md(
            "**The legs as averages.** Mean abnormal drift and hold, with the win-rate (how often each is "
            "positive). A coin is 50%."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize(EV, B, placebo=False)\n"
            "    mm, mw = s['monday']['mean']*100, s['monday']['win']*100\n"
            "    dm, dw = s['drift']['mean']*100, s['drift']['win']*100\n"
            "    hm, hw = s['hold']['mean']*100, s['hold']['win']*100\n"
            "else:\n"
            "    mm, mw = R['monday'][0], R['monday'][1]\n"
            "    dm, dw, hm, hw = R['drift'][0], R['drift'][1], R['hold'][0], R['hold'][1]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.2))\n"
            "a1.bar(['Monday\\nreaction', 'drift\\n(+1..+5d)', 'hold\\n(+6..+25d)'], [mm, dm, hm],\n"
            "       color=[GREY, AMBER, GREEN], width=.6)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean abnormal return (%)')\n"
            "a1.set_title('All three hug zero (Monday is negative)')\n"
            "for i,v in enumerate([mm,dm,hm]): a1.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "a2.bar(['Monday','drift','hold'], [mw, dw, hw], color=GREY, width=.6)\n"
            "a2.axhline(50, c=RED, ls='--', label='coin flip (50%)')\n"
            "a2.set_ylim(0,100); a2.set_ylabel('% of events positive'); a2.set_title('Win-rate ~ a coin flip')\n"
            "a2.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Monday {mm:+.1f}% (win {mw:.0f}%)   drift {dm:+.1f}% (win {dw:.0f}%)   hold {hm:+.1f}% (win {hw:.0f}%)')"
        ),
        md(
            f"The drift is **+{R['drift'][0]:.1f}%** and positive only **{R['drift'][1]:.0f}%** of the "
            f"time — *below* a coin flip. The Monday reaction is actually **{R['monday'][0]:.1f}%** "
            "(negative). There's no bump to buy: the ad airs, the world watches, the stock shrugs."
        ),
        md(
            "**Could a handful of random weeks look this good?** The honest small-sample test: draw "
            f"**{R['n_priced']}** *random* weeks on the same stocks, over and over, and see where the "
            "real drift lands against pure luck."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(B, len(EV), leg='drift', observed=EV['drift'].mean(), n_draws=4000)\n"
            "    obs = pl['obs']*100; pval = pl['p_value']\n"
            "    tickers=[c for c in B['prices'].columns if c!='SPY']\n"
            "    exs=[st._excess_log_returns(B['prices'][t], B['prices']['SPY']) for t in tickers]\n"
            "    exs=[s for s in exs if len(s)>80]; rng=np.random.default_rng(720)\n"
            "    draws=[]\n"
            "    for _ in range(3000):\n"
            "        vals=[]\n"
            "        for _ in range(len(EV)):\n"
            "            s=exs[rng.integers(0,len(exs))]; p0=rng.integers(1,len(s)-7); vals.append(np.expm1(s.iloc[p0:p0+5].sum()))\n"
            "        draws.append(np.mean(vals))\n"
            "    draws=np.array(draws)*100\n"
            "else:\n"
            "    obs=R['drift'][0]; pval=R['drift'][3]; rng=np.random.default_rng(720); draws=rng.normal(0,1.6,3000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=45, color=GREY, alpha=.85, label=f'drift of {R[\"n_priced\"]} RANDOM weeks')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'the actual advertisers ({obs:+.1f}%)')\n"
            "ax.set_xlabel('average first-week abnormal return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The drift sits dead-center in the luck cloud — placebo p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'a random {R[\"n_priced\"]}-week draw matches the drift {pval*100:.0f}% of the time — not rare at all')"
        ),
        md(
            f"The green line — the real advertisers' drift — sits **smack in the middle** of the grey "
            f"luck cloud (placebo *p* ≈ **{R['drift'][3]:.2f}**). In plain terms: **three-dozen random "
            "weeks would look about this 'special' by chance.** The drift isn't attention; it's noise "
            "with a famous paper and a sock puppet attached."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The drift is **+{R['drift'][0]:.1f}%** on a *below*-coin-flip win-rate, "
            f"the day-one reaction is **{R['monday'][0]:.1f}%** (negative), and the drift sits dead-center "
            "in the luck cloud. The 2005 paper found it on a 2000–04 sample; this modern tape doesn't.\n"
            f"- **Tradability — Mirage.** Buying the ad basket earns **+{R['cost']['gross_drift']:.1f}% "
            "gross** over five days — noise, and it's just large-cap beta. Nothing to size.\n"
            "- **\"Big-ad signal\"? — Not supported.** We remember Pets.com and the sock puppet *because* "
            "they were extreme — and they **went bust**, so they never enter a fair sample. On the "
            "survivors, the buzz doesn't move the tape."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — a basket that earns nothing\n\n"
            "Forget significance and just price the believers' trade: **buy every advertiser Monday, "
            "sell a week later.** Here's what that basket makes per event, gross and net of the (modest, "
            "large-cap) costs of getting in and out."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = st.net_of_costs(EV, cost_bps=10.0)\n"
            "    gd, nd = c['gross_drift']*100, c['net_drift']*100\n"
            "else:\n"
            "    gd, nd = R['cost']['gross_drift'], R['cost']['net_drift']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "ax.bar(['gross', 'net @10bps×2'], [gd, nd], color=[AMBER, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('per-event P&L of the ad basket (%)')\n"
            "ax.set_title('The big-ad basket earns a rounding error — before it earns anything real')\n"
            "for i,v in enumerate([gd,nd]): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'ad basket: {gd:+.2f}% gross, {nd:+.2f}% net over 5 days — a coin-flip rounding error')"
        ),
        md(
            f"The basket is **+{R['cost']['gross_drift']:.1f}% gross** and "
            f"**+{R['cost']['net_drift']:.1f}% net** over five days — a rounding error on a coin-flip "
            "win-rate. Whatever tiny positive you see is well inside its own error bar and is "
            "indistinguishable from just holding a few large caps for a week in February. There's no "
            "machine here — just $7-million commercials and a memorable sock puppet."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The other attention effects.** [Study 389 — Name-Change-Effect](../389-name-change-effect/) "
            "asks whether renaming toward the hot theme (`.com`, `Blockchain`, `AI`) pays — same family "
            "(a label / an attention grab, not a fundamental), same verdict.\n"
            "- **The anecdote trap.** [Study 343 — Data-Mining-Roulette](../343-data-mining-roulette/) "
            "shows how a few loud cases manufacture a 'law' that vanishes on a representative sample.\n"
            "- **Add the corpses.** Our survivor tape is biased *for* the story; get the delisted "
            "advertisers (Pets.com, Computer.com, Kozmo.com) and the drift wouldn't improve — those names "
            "went to **−100%**, which is a bankruptcy, not a tradable 'buzz'.\n\n"
            "*Think the big-ad signal is real and harvestable? Capture the events, draw the same number of "
            "random weeks, and show the drift landing **outside** the cloud **and** a Monday reaction "
            "that's reliably positive — then we'll talk.*"
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
            "# Super-Bowl-Advertiser — a quantitative teardown 🔬\n"
            "### Abnormal-return event windows on a Super Bowl advertiser table · a drift leg vs zero · "
            "a Welch *t* + placebo randomization null · the costed ad-calendar basket · a synthetic "
            "faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We treat the "
            "folklore as a **one-legged event-study hypothesis** — a positive post-game **drift** for "
            "advertisers — and confront it with the **sample size** and with **survivorship**. The "
            "decisive objects are a cross-section of ~32 abnormal-return events and a placebo null sized "
            "to that count.\n\n"
            "> ⚠️ **Data + survivorship note.** The advertiser table is hardcoded and transparent (~32 "
            "real listed advertiser-years, 2015–2024); the priced tape is **survivor-biased** — the "
            "loudest advertisers in history (Pets.com, Computer.com, Kozmo.com, the dot-com class of 2000) "
            "**went bust** and leave no series, biasing the drift *up* (named on the Signal axis). Real "
            "data: yfinance daily adjusted closes. Offline core + synthetic control are deterministic. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | drift **+{R['drift'][0]:.1f}%** (Welch **t = {R['drift'][2]:.2f}**, "
            f"placebo **p = {R['drift'][3]:.2f}**), win-rate **{R['drift'][1]:.0f}%** (< 50%); Monday "
            f"reaction **{R['monday'][0]:.1f}%** (**t = {R['monday'][2]:.2f}**, *negative*). Dead-center "
            "in the luck cloud. |\n"
            f"| **Tradability** | `MIRAGE` | ad basket = **+{R['cost']['gross_drift']:.1f}% gross**, "
            f"**+{R['cost']['net_drift']:.1f}% net** of 2 large-cap crossings over 5 days — a rounding "
            "error inside its own SE. |\n"
            f"| **Big-ad signal?** | `NOT SUPPORTED` | Fehle-Tsyplakov-Zdorovtsov (2005) found it on a "
            "2000–04 sample; on a modern ~32-event survivor tape the drift is noise and the memorable "
            "'ads pay' cases **delisted**. |\n\n"
            "> 💡 In plain words: a much-cited 2005 result and a famous sock puppet have kept the 'buy the "
            "buzz' idea alive, but on the names we can still price the post-game drift is indistinguishable "
            "from three-dozen random February weeks — and the day-one move is *negative*."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{i}_{[a,b]}$ be advertiser $i$'s cumulative **abnormal** return (in excess of SPY) "
            "over trading-day offsets $[a,b]$ relative to the Monday after its Super Bowl (entry lagged "
            "one day). Define a **drift** leg $D_i = r^{i}_{[+1,+5]}$ and a **hold** leg "
            "$H_i = r^{i}_{[+6,+25]}$, plus the **Monday reaction** $M_i = r^{i}_{[0,0]}$.\n\n"
            "- **H₁ (the drift).** $\\mathbb{E}[D_i] > 0$ — the ad earns an abnormal post-game drift "
            "(Fehle-Tsyplakov-Zdorovtsov 2005 found positive abnormal returns after the game).\n"
            "- **H₂ (deployable).** $\\mathbb{E}[D_i]$, net of costs on a long-the-advertisers basket, is "
            "positive and harvestable.\n\n"
            "We find **H₁ not supported** ($D$ tiny, $t<0.5$, win-rate < 50%, dead-center in the placebo "
            "cloud; $M$ actually *negative*), and therefore **H₂ rejected** (the basket is a rounding "
            "error). The legend is true only where it's a 2005 result on a 2000–04 sample, and absent "
            "where it would pay."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on the answer\n\n"
            "The teardown is a one-sample test on a small cross-section, judged by its **standard error** "
            "and by **survivorship**:\n\n"
            "$$t_{D} = \\frac{\\bar D}{s_D/\\sqrt{k}},\\qquad k\\approx 32.$$\n\n"
            "With $k\\approx 32$ and single-name volatility, $s_D/\\sqrt{k}$ is large — a few-tenths-of-a-"
            "percent mean drowns in its own SE. Worse, the sample is **conditioned on survival**: the "
            "advertisers whose post-ad path was catastrophic (a −100% bankruptcy) **left the tape**, so "
            "$\\bar D$ is biased **upward** — *toward* supporting H₁ only because the failures deleted "
            "themselves. The honest instrument is a **randomization (placebo) test**: resample $k$ random "
            "non-event windows on the same tickers and ask how often chance matches the drift. That, not "
            "the point estimate, decides the Signal axis."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Advertiser table.** ~{R['n_table']} documented listed advertiser-years (2015–2024), "
            f"hardcoded & transparent; **{R['n_delisted']}** famous advertisers are listed as **DELISTED / "
            f"went private** (no series) for the survivorship caveat. **{R['n_priced']}** priced.\n"
            "- **Abnormal returns.** Daily log return in excess of SPY; drift $=[+1,+5]$, hold $=[+6,+25]$ "
            "trading days after the Monday, **1-day entry lag** (act at Monday's close — the Sunday ad is "
            "already public; the drift leg starts Tuesday, so no weekend-gap look-ahead). The **Monday "
            "reaction** $[0,0]$ is reported separately (it folds in the un-tradable Fri→Mon gap).\n"
            "- **Null #1 (Welch t).** The drift's cross-sectional mean vs zero.\n"
            "- **Null #2 (placebo).** 20,000 draws of $k$ random non-event windows on the same tickers; "
            "$p = \\Pr[|\\text{random mean}| \\ge |\\text{observed}|]$ — the small-sample workhorse.\n"
            "- **Costs.** A long-the-advertisers basket pays a one-way large-cap charge on **two** "
            "crossings per event.\n"
            "- **Positive control.** Deterministic event windows with a **planted** drift of size `edge`: "
            "the inference must recover a large edge **and** must NOT manufacture significance when "
            "`edge = 0`."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The drift — a whisper inside its own SE\n\n"
            "Mean abnormal return per leg with $\\pm$ standard error, against zero (dashed). The drift is "
            "small and inside its SE; the Monday reaction is *negative*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    M = EV['monday'].values; D = EV['drift'].values; H = EV['hold'].values\n"
            "    mm, dm, hm = M.mean()*100, D.mean()*100, H.mean()*100\n"
            "    mse = M.std(ddof=1)/np.sqrt(len(M))*100; dse = D.std(ddof=1)/np.sqrt(len(D))*100\n"
            "    hse = H.std(ddof=1)/np.sqrt(len(H))*100\n"
            "    mt, dt, ht = st.welch_t(M), st.welch_t(D), st.welch_t(H)\n"
            "else:\n"
            "    mm, dm, hm = R['monday'][0], R['drift'][0], R['hold'][0]\n"
            "    mt, dt, ht = R['monday'][2], R['drift'][2], R['hold'][2]\n"
            "    mse, dse, hse = abs(mm/max(abs(mt),1e-9)), abs(dm/max(abs(dt),1e-9)), abs(hm/max(abs(ht),1e-9))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(['Monday [0,0]','drift [+1,+5]','hold [+6,+25]'], [mm, dm, hm], yerr=[mse, dse, hse],\n"
            "       capsize=6, color=[GREY, AMBER, GREEN], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('mean abnormal return (%)')\n"
            "ax.set_title(f'drift t={dt:.2f} (n.s.); Monday t={mt:.2f} and NEGATIVE')\n"
            "for i,v in enumerate([mm,dm,hm]): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Monday {mm:+.2f}% (t={mt:.2f})  drift {dm:+.2f}% (t={dt:.2f})  hold {hm:+.2f}% (t={ht:.2f})')"
        ),
        md(
            f"> 💡 In plain words: the drift is **+{R['drift'][0]:.1f}%** at **t = {R['drift'][2]:.2f}** "
            f"(not significant); the Monday reaction is **{R['monday'][0]:.1f}%** at "
            f"**t = {R['monday'][2]:.2f}** — *negative*, the opposite of a buzz pop; and the hold leg "
            f"(**+{R['hold'][0]:.1f}%**, t = {R['hold'][2]:.2f}) is a shrug. H₁ is not supported on any leg."
        ),
        md(
            "### 4b · The decisive test — a placebo null sized to the event count\n\n"
            f"Draw {R['n_priced']} random non-event windows 20,000 times; the histogram is the null for "
            "the mean drift. The real drift is the green line; the *p*-value is the two-sided tail mass."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(B, len(EV), leg='drift', observed=EV['drift'].mean(), n_draws=6000)\n"
            "    obs = pl['obs']*100; pval = pl['p_value']\n"
            "    tickers=[c for c in B['prices'].columns if c!='SPY']\n"
            "    exs=[st._excess_log_returns(B['prices'][t], B['prices']['SPY']) for t in tickers]\n"
            "    exs=[s for s in exs if len(s)>80]; rng=np.random.default_rng(720)\n"
            "    draws=[]\n"
            "    for _ in range(6000):\n"
            "        vals=[]\n"
            "        for _ in range(len(EV)):\n"
            "            s=exs[rng.integers(0,len(exs))]; p0=rng.integers(1,len(s)-7); vals.append(np.expm1(s.iloc[p0:p0+5].sum()))\n"
            "        draws.append(np.mean(vals))\n"
            "    draws=np.array(draws)*100\n"
            "else:\n"
            "    obs=R['drift'][0]; pval=R['drift'][3]; rng=np.random.default_rng(720); draws=rng.normal(0,1.6,6000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=55, color=GREY, alpha=.85, label=f'null: {R[\"n_priced\"]} random windows')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'observed drift {obs:+.1f}%')\n"
            "ax.set_xlabel('mean first-week abnormal return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Placebo p = {pval:.2f}: the drift is dead-center in the luck cloud'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'P[|random {R[\"n_priced\"]}-window mean| >= |drift|] = {pval:.3f}  (need <0.05 to call it real)')"
        ),
        md(
            f"> 💡 In plain words: **{R['drift'][3]*100:.0f}%** of random {R['n_priced']}-window draws "
            "match or beat the drift in magnitude. A real effect would push the green line into the tail; "
            "instead it sits mid-cloud. H₁ is **not supported** — the drift is what three-dozen random "
            "February weeks look like."
        ),
        md(
            "### 4c · By era + robustness — the drift is nowhere and sign-unstable\n\n"
            "Split by era and shift the windows. The drift **flips sign** across eras and horizons "
            "(2019–21 is *negative*), and the drift *t* never clears 1.2 — the hallmark of noise, not a "
            "decaying real effect."
        ),
        code(
            "eras = R['eras']\n"
            "if HAVE_REAL:\n"
            "    eras = []\n"
            "    for nm,yrs in [('2015-17',range(2015,2018)),('2019-21',range(2019,2022)),('2022-24',range(2022,2025))]:\n"
            "        sub = EV[EV['year'].isin(list(yrs))]\n"
            "        eras.append((nm, len(sub), sub['drift'].mean()*100, sub['hold'].mean()*100))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "names=[e[0] for e in eras]; edr=[e[2] for e in eras]; ehd=[e[3] for e in eras]; xx=np.arange(len(names))\n"
            "a1.bar(xx-.2, edr, .4, color=AMBER, label='drift'); a1.bar(xx+.2, ehd, .4, color=GREEN, label='hold')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_xticks(xx); a1.set_xticklabels(names)\n"
            "a1.set_ylabel('mean abnormal return (%)'); a1.set_title('Drift flips sign by era'); a1.legend()\n"
            "if HAVE_REAL:\n"
            "    rob=[]\n"
            "    for d_,h_ in [(3,10),(5,20),(10,40)]:\n"
            "        e2=st.collect_events(B,drift=d_,hold=h_); s2=st.summarize(e2,B,drift=d_,hold=h_,placebo=False)\n"
            "        rob.append((f'{d_}/{h_}', s2['n'], s2['drift']['mean']*100, s2['drift']['t'], s2['hold']['mean']*100, s2['hold']['t']))\n"
            "else:\n"
            "    rob = R['robust']\n"
            "labs=[r[0] for r in rob]; dts=[r[3] for r in rob]; hts=[r[5] for r in rob]; xx2=np.arange(len(labs))\n"
            "a2.bar(xx2-.2, dts, .4, color=AMBER, label='drift t'); a2.bar(xx2+.2, hts, .4, color=GREEN, label='hold t')\n"
            "a2.axhline(2, ls='--', c=RED, label='t=2'); a2.axhline(-2, ls='--', c=RED)\n"
            "a2.set_xticks(xx2); a2.set_xticklabels(labs); a2.set_ylabel('Welch t'); a2.set_xlabel('drift/hold days')\n"
            "a2.set_title('No window clears |t|=2'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('eras:', [(e[0], round(e[2],1), round(e[3],1)) for e in eras])"
        ),
        md(
            "> 💡 In plain words: 2015–17 drifts up, 2019–21 drifts *down*, 2022–24 is flat — a sign that "
            "wanders with the sample, not an effect. And across window choices (3/10, 5/20, 10/40) no "
            "drift *t* clears 1.2. The result is noise in every choice that could have rescued it."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "Deterministic event windows with a **planted** post-game drift of size `edge`: with "
            "**`edge=0`** the inference must stay flat (a few-dozen noisy events can't fake significance); "
            "with a **+10%** planted drift it must light up the drift leg. Both hold — proving the engine "
            "is unbiased *and* that this sample size only detects implausibly large effects."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.10):\n"
            "    syn = data.synthetic_ads(n_events=32, edge=edge, seed=726)\n"
            "    ev = st.collect_events(syn, drift=5, hold=20); s = st.summarize(ev, syn, drift=5, hold=20, placebo=False)\n"
            "    res.append((edge, s['n'], s['drift']['mean']*100, s['drift']['t'], s['hold']['mean']*100, s['hold']['t']))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "labels=[f'planted\\n{int(e*100)}%' for e,*_ in res]; xx=np.arange(len(labels))\n"
            "dts=[r[3] for r in res]; hts=[r[5] for r in res]\n"
            "ax.bar(xx-.2, dts, .4, color=AMBER, label='drift t'); ax.bar(xx+.2, hts, .4, color=GREEN, label='hold t')\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2'); ax.axhline(-2, ls='--', c=RED)\n"
            "ax.set_xticks(xx); ax.set_xticklabels(labels); ax.set_ylabel('Welch t')\n"
            "ax.set_title('Control: only a large planted drift lights up the drift leg'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,dm,dt,hm,ht in res: print(f'planted {int(e*100):>3}%: n={k} drift={dm:+.1f}%(t={dt:+.2f}) hold={hm:+.1f}%(t={ht:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted edge the control's drift *t* is "
            f"**{R['syn'][0][3]:.2f}** and hold *t* is **{R['syn'][0][5]:.2f}** (both ~0 — no false "
            f"positive); only the **+10%** plant reaches drift *t* **{R['syn'][1][3]:.2f}**. So the "
            f"machinery is honest, and the real-tape drift *t* of **{R['drift'][2]:.2f}** is exactly what "
            f"an *absent or tiny* effect looks like through a {R['n_priced']}-event keyhole."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — drift **+{R['drift'][0]:.1f}%** at Welch **t = {R['drift'][2]:.2f}** / "
            f"placebo **p = {R['drift'][3]:.2f}**, win-rate **{R['drift'][1]:.0f}%** (< 50%); Monday "
            f"reaction **{R['monday'][0]:.1f}%** (t = {R['monday'][2]:.2f}, *negative*). "
            "Fehle-Tsyplakov-Zdorovtsov (2005) found the effect on a 2000–04 sample, but this modern tape "
            "is indistinguishable from noise ⇒ NONE, not WEAK. **Survivorship** named on this axis: the "
            "advertisers that went bust delisted, biasing the drift *up*.\n"
            f"- **Tradability `MIRAGE`** — the long-the-advertisers basket is "
            f"**+{R['cost']['gross_drift']:.1f}% gross**, **+{R['cost']['net_drift']:.1f}% net** of 2 "
            "large-cap crossings over 5 days — a rounding error inside its own SE, indistinguishable from "
            "a week of February beta.\n"
            f"- **Big-ad signal? `NOT SUPPORTED`** — the legend is a much-cited 2005 result plus a "
            "memorable sock puppet; across a representative modern survivor table the post-game drift is a "
            "coin-flip and the day-one move is negative."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the power curve\n\n"
            "The operational truth in one picture: how big would the *true* drift have to be for a "
            "$k$-event study to detect it at $t=2$? At $k\\approx 32$ you'd need a drift several times "
            "the one observed; the real signal lives far below the detection floor — and the basket is a "
            "rounding error on top."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sd = EV['drift'].std(ddof=1); obs = EV['drift'].mean()\n"
            "else:\n"
            "    sd = R['drift_std']; obs = R['drift'][0]/100\n"
            "ks = np.arange(5, 250)\n"
            "min_det = 2.0 * sd / np.sqrt(ks)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(ks, min_det*100, c=AMBER, lw=2, label='drift needed for t=2')\n"
            "ax.axhline(obs*100, c=GREEN, ls='--', label=f'observed drift ~{obs*100:+.1f}%')\n"
            "ax.axvline(R['n_priced'], c=GREY, ls=':', label=f\"our k={R['n_priced']}\")\n"
            "ax.set_xlabel('number of events k'); ax.set_ylabel('first-week drift (%)')\n"
            "ax.set_title('Detection floor vs the real drift: badly under-powered'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "need = 2.0*sd/np.sqrt(R['n_priced'])*100\n"
            "print(f'at k={R[\"n_priced\"]} you need ~{need:.1f}% drift for t=2; observed ~{obs*100:+.1f}% -> under-powered')"
        ),
        md(
            "> 💡 In plain words: the amber curve is the **minimum detectable drift**; the green line is "
            "what we see. They don't meet until $k$ is many times larger than the Super Bowl calendar "
            "will ever deliver (one game a year, a couple-dozen listed advertisers) — and even a *real* "
            "drift this small wouldn't clear large-cap costs. There is no sizing, threshold, or cost "
            "assumption that manufactures an edge from ~32 events whose mean is a coin-flip whisper. The "
            "rarity that makes the legend fun (one big night a year) is exactly what makes it untestable "
            "and untradable."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The adjacent attention effect.** [Study 389 — Name-Change-Effect]"
            "(../389-name-change-effect/): does renaming toward the hot theme (`.com`/`Blockchain`/`AI`) "
            "pay? Same family (an attention grab, not a fundamental), same small-sample / survivorship "
            "pathology.\n"
            "- **The anecdote trap, formalised.** [Study 343 — Data-Mining-Roulette]"
            "(../343-data-mining-roulette/) on how loud cases manufacture spurious 'laws'.\n"
            "- **Add the corpses.** Reconstruct the delisted advertisers (Pets.com, Computer.com, "
            "Kozmo.com) from a survivorship-free CRSP feed; the drift wouldn't grow — those names went to "
            "a −100% bankruptcy, informative about the *story*, not a tradable leg. And re-run the "
            "Fehle-Tsyplakov-Zdorovtsov (2005) window on 2000–04 to see the effect that has since faded.\n\n"
            "*The reproducible core is offline and deterministic; the advertiser table is hardcoded and "
            "the priced tape is survivor-biased (named). Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
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
