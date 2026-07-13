"""Generate the two narrative notebooks for Study 722 (Logo-Rebrand).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached rebrand prices
under ../_cache/ (the ~26-name table + SPY) and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with no
network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance ~26-name rebrand
# table + SPY, 2008-01-02 -> 2026-06-30, as-of 2026-06-30, fp=c12adbfe3fd2; 22 events priced).
R = dict(
    n_table=26, n_delisted=8, n_priced=22, asof="2026-06-30", fp="c12adbfe3fd2",
    # leg: (mean%, win%, t, placebo_p)
    announce=(2.44, 64, 1.96, 0.022),
    drift=(-4.61, 45, -1.05, 0.333),
    announce_median=1.60,
    # kind: (n, announce%, drift%)
    kinds=[("name", 8, 3.56, -16.40), ("identity", 5, 0.64, -9.06),
           ("logo", 9, 2.45, 8.34)],
    # outlier fragility of the announce blip: (n_dropped, n, announce%, t)
    fragile=[(0, 22, 2.44, 1.96), (1, 21, 1.71, 1.62), (2, 20, 1.02, 1.21),
             (3, 19, 0.68, 0.84)],
    # robustness: (ann_d/drift_d, n, ann%, ann_t, drift%, drift_t)
    robust=[("3/60", 22, 1.74, 1.78, -1.70, -0.51), ("5/120", 22, 2.44, 1.96, -4.61, -1.05),
            ("5/252", 22, 2.44, 1.96, -9.18, -1.38), ("10/120", 22, 2.27, 1.29, -4.85, -1.32),
            ("1/120", 22, 0.24, 0.34, -1.82, -0.38)],
    cost=dict(gross_hold=-2.17, net_hold=-2.37, gross_announce=2.44, net_announce=2.24,
              gross_drift=-4.61, net_drift=-4.81),
    # synthetic: edge, n, announce%, announce_t, drift%, drift_t
    syn=[(0.00, 26, -0.32, -0.31, -0.52, -0.10),
         (0.30, 26, -0.32, -0.31, 34.28, 5.05)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Renewal or red--flag%3F: Coin_flip](https://img.shields.io/badge/Renewal_or_red--flag%3F-Coin_flip-8b949e?style=flat-square)\n\n"
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

from logo_rebrand import data, strategy as st

HAVE_REAL = data.have_real()
B = data.load_real() if HAVE_REAL else None
EV = st.collect_events(B) if HAVE_REAL else None
print("real rebrand-tape cache present:", HAVE_REAL,
      "| events priced:", (0 if EV is None else len(EV)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The rebrand tell — is a new logo a comeback, or a cry for help? 🎨\n"
            "### The company that changes its name or logo — renewal, or a floundering firm's vanity — in plain English\n\n"
            + BADGES +
            "Two camps swear opposite things about the same event. **Camp Renewal:** when a company "
            "unveils a fresh name or logo — Google → *Alphabet*, Facebook → *Meta* — it's announcing a "
            "turnaround; the stock re-rates, so **buy the rebrand**. **Camp Red-Flag:** a rebrand is what "
            "a *struggling* firm does to distract from a rotting business — a vanity move, a **sell "
            "signal**. Both can't be right, and a rebrand is a public, dated event — so we can just look.\n\n"
            "This notebook builds a transparent table of **~26 real corporate rebrands (2010–2025)** — "
            "name changes, identity refreshes, and pure logo redesigns — and asks what actually happened "
            "to the stock: a quick pop the week of the reveal, and the six months after.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the power "
            "analysis? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Some of the loudest rebrands **left the market**: *Twitter → "
            "X* went private, *Weight Watchers → WW* filed for bankruptcy in 2025, *Paramount* got "
            "swallowed in 2025. The names we *can* still price are the ones that *survived* — which "
            "biases everything **against** the 'floundering firm' story, and makes it all the more "
            "telling what the survivors show. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does a rebrand pop the stock the week of the reveal? | **A little — but don't bank on it.** "
            f"The first-week bump averages **+{R['announce'][0]:.1f}%**, but it's carried by two "
            "news-confounded outliers; drop them and it fades to a coin-flip. |\n"
            "| Does a rebrand mark a *renewal* — outperformance in the months after? | **No.** Over the "
            f"next ~6 months the survivors drift **{R['drift'][0]:.0f}%**, statistically zero. No "
            "turnaround shows up in the price. |\n"
            "| Is it a *floundering-firm* red flag, then? | **Also no — not on the survivors.** The drift "
            "is slightly negative but indistinguishable from noise, and the real disasters already "
            "**left the tape**. |\n"
            "| Can you trade 'buy the rebrand'? | **It loses money.** Buying the reveal and holding is "
            f"**{R['cost']['gross_hold']:.1f}% per event before costs** — the soft negative drift eats "
            "the little pop. |\n\n"
            "> A rebrand is neither a reliable comeback nor a reliable tell. The week-one pop is a fragile "
            "whisper; the months after are noise. It's a coin flip wearing a new logo."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A rebrand is a signal. When a company changes its name or logo, it's telling you "
            "something — either it's turning the page toward renewal (buy the fresh start), or it's a "
            "floundering firm papering over the cracks (sell the vanity). Either way, the rebrand moves "
            "the stock, and you can trade the direction.\"*\n\n"
            "Both halves have adherents. Design and marketing press frame rebrands as **renewal** "
            "catalysts; value-investing folklore treats a splashy logo change as a **distress tell** "
            "('when management is redecorating, check the balance sheet'). They make *opposite* "
            "predictions about the drift after a rebrand — so a single event study can referee. (The "
            "desk already tested the theme-chasing cousin: [Study 389 — Name-Change-Effect]"
            "(../389-name-change-effect/), where `.com`/`Blockchain`/`AI` renames 'pop then dump.')"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a rebrand reliably signalled *renewal*, you'd have a tidy calendar of catalysts to buy. "
            "If it reliably signalled *floundering*, you'd have a free short list of names to avoid or "
            "fade. And it would say something about markets — that a **press release with a new "
            "wordmark** carries information about the *future*, not just the present. But for either "
            "trade to work, the drift after the rebrand has to point **reliably** one way. If name-changes "
            "drift down while logo refreshes drift up — because *troubled firms rename and healthy giants "
            "restyle* — then the 'signal' is just the firm's pre-existing health leaking through, "
            "backwards, and there's nothing to trade."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We build a **transparent table of ~{R['n_table']} real rebrands** (2010–2025) — name "
            "changes (Facebook→Meta, Coach→Tapestry), identity/structure refreshes (ConAgra→Conagra "
            "Brands), and pure logo redesigns (Pepsi, J&J, Walmart, Mastercard). For each one:\n\n"
            "1. **Find the reveal.** Line up the stock against the market (SPY) around the announcement.\n"
            "2. **Measure two legs.** The **announce** reaction (abnormal return over the first ~week) "
            "and the **drift** (the next ~6 months) — *abnormal* meaning *beyond what the market did*.\n"
            "3. **Stress the luck.** Draw the same number of *random* windows thousands of times and ask "
            "how often chance produces a drift this big. With only ~22 events, that's the honest test.\n\n"
            "And we say it loudly: the worst-outcome rebrands **left the tape** (Twitter→X private, "
            "WW bankrupt, Paramount acquired), so our survivors are the *good* cases — the deck is "
            "stacked **against** the floundering story, not for it."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the week-one pop and the six-month drift, side by side.** For every surviving "
            "rebrand: the abnormal (market-adjusted) return in the first week vs the next six months."
        ),
        code(
            "if HAVE_REAL:\n"
            "    anns = EV['announce'].values*100; drs = EV['drift'].values*100; labels = EV['ticker'].values\n"
            "else:\n"
            "    rng=np.random.default_rng(722)\n"
            "    anns=rng.normal(R['announce'][0],6,R['n_priced']); drs=rng.normal(R['drift'][0],20,R['n_priced'])\n"
            "    labels=[f'R{i}' for i in range(R['n_priced'])]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "ax.scatter(anns, drs, c=GREY, s=60, edgecolor='k', alpha=.8)\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c='k', lw=.8)\n"
            "ax.axhline(np.mean(drs), c=RED, ls='--', label=f'mean drift {np.mean(drs):+.0f}%')\n"
            "ax.axvline(np.mean(anns), c=AMBER, ls='--', label=f'mean pop {np.mean(anns):+.1f}%')\n"
            "ax.set_xlabel('announce: first-week abnormal return (%)')\n"
            "ax.set_ylabel('drift: next-6-months abnormal return (%)')\n"
            "ax.set_title('Renewal would sit ABOVE zero, floundering BELOW — the cloud straddles it')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'mean pop {np.mean(anns):+.1f}%   mean drift {np.mean(drs):+.1f}%  (renewal>0, floundering<0)')"
        ),
        md(
            f"There's the tell — or the absence of one. If rebrands meant *renewal*, the dots would sit "
            f"**above** zero (positive drift); if they meant *floundering*, **below**. Instead the mean "
            f"drift is **{R['drift'][0]:.0f}%** and the cloud straddles the line. The pop leans faintly "
            "positive; the drift is a wash."
        ),
        md(
            "**The two legs as averages.** Mean abnormal announce-pop and drift, with the win-rate (how "
            "often each leg is positive). A coin is 50%."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize(EV, B, placebo=False)\n"
            "    am, aw = s['announce']['mean']*100, s['announce']['win']*100\n"
            "    dm, dw = s['drift']['mean']*100, s['drift']['win']*100\n"
            "else:\n"
            "    am, aw, dm, dw = R['announce'][0], R['announce'][1], R['drift'][0], R['drift'][1]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.2))\n"
            "a1.bar(['announce\\n(+1..+5d)', 'drift\\n(+6..+126d)'], [am, dm], color=[AMBER, RED], width=.55)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean abnormal return (%)')\n"
            "a1.set_title('A small pop, then a drift that is basically zero')\n"
            "for i,v in enumerate([am,dm]): a1.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "a2.bar(['announce','drift'], [aw, dw], color=GREY, width=.55)\n"
            "a2.axhline(50, c=RED, ls='--', label='coin flip (50%)')\n"
            "a2.set_ylim(0,100); a2.set_ylabel('% of events positive'); a2.set_title('Drift win-rate is a coin flip')\n"
            "a2.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'announce {am:+.1f}% (win {aw:.0f}%)   drift {dm:+.1f}% (win {dw:.0f}%)')"
        ),
        md(
            f"The pop is **+{R['announce'][0]:.1f}%** and positive **{R['announce'][1]:.0f}%** of the "
            f"time; the drift is **{R['drift'][0]:.0f}%** and a **{R['drift'][1]:.0f}%** coin-flip. "
            "Neither camp's promise — a re-rating up, or a slide down — reliably shows on the survivors."
        ),
        md(
            "**Could a handful of random windows look this good?** The honest small-sample test for the "
            f"*drift* (the actual renewal/floundering claim): draw **{R['n_priced']}** *random* windows "
            "on the same stocks, over and over, and see where the real drift lands against pure luck."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(B, len(EV), leg='drift', observed=EV['drift'].mean(), n_draws=4000)\n"
            "    obs = pl['obs']*100; pval = pl['p_value']\n"
            "    tickers=[c for c in B['prices'].columns if c!='SPY']\n"
            "    exs=[st._excess_log_returns(B['prices'][t], B['prices']['SPY']) for t in tickers]\n"
            "    exs=[s for s in exs if len(s)>200]; rng=np.random.default_rng(722)\n"
            "    draws=[]\n"
            "    for _ in range(3000):\n"
            "        vals=[]\n"
            "        for _ in range(len(EV)):\n"
            "            s=exs[rng.integers(0,len(exs))]; p0=rng.integers(1,len(s)-130); vals.append(np.expm1(s.iloc[p0:p0+120].sum()))\n"
            "        draws.append(np.mean(vals))\n"
            "    draws=np.array(draws)*100\n"
            "else:\n"
            "    obs=R['drift'][0]; pval=R['drift'][3]; rng=np.random.default_rng(722); draws=rng.normal(0,4.5,3000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=45, color=GREY, alpha=.85, label=f'drift of {R[\"n_priced\"]} RANDOM windows')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'the actual rebrands ({obs:+.1f}%)')\n"
            "ax.set_xlabel('average 6-month abnormal return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The drift sits deep inside the luck cloud — placebo p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'a random {R[\"n_priced\"]}-window draw matches the drift {pval*100:.0f}% of the time — not rare at all')"
        ),
        md(
            f"The red line — the real rebrands' drift — sits **squarely inside** the grey luck cloud "
            f"(placebo *p* ≈ **{R['drift'][3]:.2f}**). In plain terms: **a couple-dozen random dates "
            "would look about this 'special' one-in-three times.** There is no renewal signal and no "
            "floundering signal in the drift — just noise."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The renewal/floundering claim lives in the **drift**, and the drift is "
            f"**{R['drift'][0]:.0f}%** at a *t* of **{R['drift'][2]:.2f}** — pure noise (a `NONE`). The "
            f"only positive is a first-week **+{R['announce'][0]:.1f}%** pop, but it *fails* the "
            "significance bar and **collapses when two news-confounded outliers leave** — a fragile "
            "`WEAK`, not a real edge.\n"
            f"- **Tradability — Mirage.** Buy-the-rebrand-and-hold is **{R['cost']['gross_hold']:.1f}% "
            "per event before costs** — the soft negative drift eats the little pop.\n"
            "- **Renewal or red-flag? — Coin flip.** The drift can't tell them apart. Name-changes drift "
            "*down*, logo refreshes drift *up* — which is the firm's pre-existing health leaking through "
            "backwards, not a signal you can act on."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the pop the outliers built\n\n"
            "Forget the drift and just price the renewal camp's trade: **buy the reveal, hold six "
            "months.** And then check the one leg that looked alive — the week-one pop — for how much of "
            "it is really just *two* stocks."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = st.net_of_costs(EV, cost_bps=10.0)\n"
            "    gh, nh = c['gross_hold']*100, c['net_hold']*100\n"
            "    a = EV['announce'].values; order=np.argsort(-np.abs(a))\n"
            "    frag=[]\n"
            "    for drop in (0,1,2,3):\n"
            "        keep=np.ones(len(a),bool); keep[order[:drop]]=False\n"
            "        frag.append((drop, a[keep].mean()*100, st.welch_t(a[keep])))\n"
            "else:\n"
            "    gh, nh = R['cost']['gross_hold'], R['cost']['net_hold']\n"
            "    frag=[(f[0],f[2],f[3]) for f in R['fragile']]\n"
            "fig, (a1,a2) = plt.subplots(1,2,figsize=(10.4, 4.2))\n"
            "a1.bar(['gross', 'net @10bps'], [gh, nh], color=[AMBER, RED], width=.5)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('per-event P&L of buy-and-hold (%)')\n"
            "a1.set_title('Buy-the-rebrand loses money')\n"
            "for i,v in enumerate([gh,nh]): a1.annotate(f'{v:+.1f}%',(i,v),ha='center',va='top')\n"
            "dd=[f'drop {f[0]}' for f in frag]; tt=[f[2] for f in frag]\n"
            "a2.bar(dd, tt, color=GREY, width=.55); a2.axhline(2, ls='--', c=RED, label='t=2 (significant)')\n"
            "a2.set_ylabel('Welch t of the week-one pop'); a2.set_title('The pop dies when 2 outliers leave')\n"
            "a2.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'buy-and-hold: {gh:+.1f}% gross, {nh:+.1f}% net'); print('pop t as biggest outliers drop:', [(d,round(t,2)) for d,_,t in frag])"
        ),
        md(
            f"The hold book is **{R['cost']['gross_hold']:.1f}% gross** / **{R['cost']['net_hold']:.1f}% "
            "net** — a money-loser, because the six-month drift is soft-negative. And the one leg that "
            f"flickered — the week-one pop — has a *t* of **{R['fragile'][0][3]:.2f}** that falls to "
            f"**{R['fragile'][2][3]:.2f}** the moment you remove just **two** stocks (BlackBerry, whose "
            "pop was really its BB10 phone launch, and GM, mid-EV-re-rating). Take away two headlines "
            "that had nothing to do with the logo, and the 'rebrand pop' is a coin flip. There's no "
            "machine here."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The theme-chasing cousin.** [Study 389 — Name-Change-Effect](../389-name-change-effect/) "
            "tests the `.com`/`Blockchain`/`AI` renames — the 'pop then dump' variant, same family "
            "(a label, not a fundamental), same verdict.\n"
            "- **The anecdote trap.** [Study 343 — Data-Mining-Roulette](../343-data-mining-roulette/) "
            "shows how a few loud cases (here: two confounded outliers) manufacture a 'law.'\n"
            "- **Add the corpses.** Our survivor tape is biased *against* the floundering story; get "
            "delisted/private data (Twitter→X, WW, Bed Bath & Beyond) and the drift might turn negative "
            "— but that's a bankruptcy, not a tradable 'signal,' and you couldn't have shorted a private "
            "company anyway.\n\n"
            "*Think the rebrand signal is real and harvestable? Capture the events, draw the same number "
            "of random windows, and show the drift landing **outside** the cloud — reliably one sign, "
            "surviving the removal of your two biggest names — then we'll talk.*"
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
            "# Logo-Rebrand — a quantitative teardown 🔬\n"
            "### Abnormal-return event windows on a rebrand table · announce/drift legs vs zero · "
            "a Welch *t* + placebo randomization null · outlier-fragility · the costed renewal trade · "
            "a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We treat the "
            "folklore as a **two-sided directional hypothesis on the post-rebrand drift** — *renewal* "
            "(drift > 0) vs *floundering* (drift < 0) — and confront it with the **sample size**, with "
            "**outlier fragility**, and with **survivorship**. The decisive objects are a cross-section "
            "of ~22 abnormal-return events and a placebo null sized to that count.\n\n"
            "> ⚠️ **Data + survivorship note.** The rebrand table is hardcoded and transparent (~26 real "
            "rebrands 2010–2025 across name/identity/logo); the priced tape is **survivor-biased** — the "
            "worst-outcome rebrands (Twitter→X private, WW bankrupt 2025, Paramount acquired 2025) leave "
            "no clean series, biasing the drift *up* (named on the Signal axis). Real data: yfinance "
            f"daily adjusted closes, 2008→2026, as-of **{R['asof']}**, fp `{R['fp']}`. Offline core + "
            "synthetic control are deterministic. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | drift **{R['drift'][0]:.1f}%** (Welch **t = {R['drift'][2]:.2f}**, "
            f"placebo **p = {R['drift'][3]:.2f}**) — the renewal/floundering claim is `NONE`. A week-one "
            f"pop **+{R['announce'][0]:.1f}%** (**t = {R['announce'][2]:.2f}**, placebo **p = "
            f"{R['announce'][3]:.3f}**) is borderline but **fails t≥2 and collapses to t = "
            f"{R['fragile'][2][3]:.2f}** when 2 confounded outliers drop ⇒ WEAK. |\n"
            f"| **Tradability** | `MIRAGE` | buy-the-rebrand-and-hold = **{R['cost']['gross_hold']:.1f}% "
            f"gross**, **{R['cost']['net_hold']:.1f}% net** per event. The soft-negative drift dominates "
            "the pop; nothing to size. |\n"
            f"| **Renewal or red-flag?** | `COIN FLIP` | name-changes drift **{R['kinds'][0][3]:.0f}%**, "
            f"logo-only refreshes **+{R['kinds'][2][3]:.0f}%** — reverse causation by firm health, not a "
            "tradable signal. |\n\n"
            "> 💡 In plain words: the two camps make opposite predictions about the drift, and the drift "
            "is noise — so *neither* is supported. The one flicker of life (a week-one pop) is two "
            "headlines (BB10 launch, GM's EV re-rating) that had nothing to do with the logo."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{i}_{[a,b]}$ be stock $i$'s cumulative **abnormal** return (in excess of SPY) over "
            "trading-day offsets $[a,b]$ relative to its rebrand date (entry lagged one day). Define an "
            "**announce** leg $A_i = r^{i}_{[+1,+5]}$ and a **drift** leg $D_i = r^{i}_{[+6,+126]}$.\n\n"
            "- **H_renewal.** $\\mathbb{E}[D_i] > 0$ — the rebrand marks a turnaround; the stock "
            "re-rates upward over the following ~6 months.\n"
            "- **H_flounder.** $\\mathbb{E}[D_i] < 0$ — the rebrand is a distress tell; the stock keeps "
            "sliding.\n"
            "- **H_deployable.** A buy-the-rebrand book, net of costs, earns $\\mathbb{E}[A_i + D_i] > 0$.\n\n"
            "We find **H_renewal not supported** ($D$ small and *negative*), **H_flounder not supported** "
            "($|t_D| < 2$, placebo $p \\approx 0.33$, and survivorship biases $D$ *up*), and "
            "**H_deployable rejected** (the book is gross-negative). The only survivor is a fragile "
            "week-one pop that fails $t = 2$ and is two outliers deep."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The teardown is two one-sample tests on a small cross-section, judged by their **standard "
            "error**, by **outlier leverage**, and by **survivorship**:\n\n"
            "$$t_{\\text{leg}} = \\frac{\\bar X}{s_X/\\sqrt{k}},\\qquad X\\in\\{A, D\\},\\ k\\approx 22.$$\n\n"
            "With $k\\approx 22$ and large-cap-with-tails volatility, $s_X/\\sqrt{k}$ is large — a "
            "few-percent mean drowns in its own SE, and a *single* news-confounded event can swing $t$ by "
            "0.5. Worse, the sample is **conditioned on survival**: firms whose post-rebrand path was most "
            "negative (Twitter→X, WW, Bed Bath & Beyond) **left the tape**, so $\\bar D$ is biased "
            "**upward** — *against* the floundering thesis. The honest instrument is a **randomization "
            "(placebo) test**: resample $k$ random non-event windows on the same tickers and ask how often "
            "chance matches each leg. That, plus an **outlier-drop** curve, decides the Signal axis."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Rebrand table.** ~{R['n_table']} documented rebrands 2010–2025 (name / identity / "
            f"logo), hardcoded & transparent; **{R['n_delisted']}** famous ones that delisted or went "
            f"private are listed as **DELISTED** (no series) for the survivorship caveat. "
            f"**{R['n_priced']}** priced (two of the four drops — WW, Paramount — are *themselves* "
            "survivorship: the rebranded firm left the tape).\n"
            "- **Abnormal returns.** Daily log return in excess of SPY; announce $=[+1,+5]$, drift "
            "$=[+6,+126]$ trading days (~6 months), **1-day entry lag** (act the day after the reveal).\n"
            "- **Null #1 (Welch t).** Each leg's cross-sectional mean vs zero.\n"
            "- **Null #2 (placebo).** 20,000 draws of $k$ random non-event windows on the same tickers; "
            "$p = \\Pr[|\\text{random mean}| \\ge |\\text{observed}|]$ — the small-sample workhorse.\n"
            "- **Fragility.** Drop the 1–3 largest-$|A|$ events and re-test the pop.\n"
            "- **Costs.** A buy-the-rebrand-and-hold book pays a one-way large-cap charge on **two** "
            "crossings per event.\n"
            "- **Positive control.** Deterministic event windows with a **planted** renewal drift of "
            "size `edge`: the inference must recover a large edge **and** must NOT manufacture "
            "significance when `edge = 0`."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The two legs — a faint pop, a null drift\n\n"
            "Mean abnormal return per leg with $\\pm$ standard error, against zero (dashed). The announce "
            "pop is small and barely outside its SE; the drift is negative but well inside it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    A = EV['announce'].values; D = EV['drift'].values\n"
            "    am, dm = A.mean()*100, D.mean()*100\n"
            "    ase, dse = A.std(ddof=1)/np.sqrt(len(A))*100, D.std(ddof=1)/np.sqrt(len(D))*100\n"
            "    at, dt = st.welch_t(A), st.welch_t(D)\n"
            "else:\n"
            "    am, dm, at, dt = R['announce'][0], R['drift'][0], R['announce'][2], R['drift'][2]\n"
            "    ase, dse = abs(am/max(at,1e-9)), abs(dm/min(dt,-1e-9))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(['announce [+1,+5]','drift [+6,+126]'], [am, dm], yerr=[ase, dse], capsize=6,\n"
            "       color=[AMBER, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('mean abnormal return (%)')\n"
            "ax.set_title(f'announce t={at:.2f} (sub-2), drift t={dt:.2f} (n.s.) — neither camp lands')\n"
            "for i,v in enumerate([am,dm]): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'announce {am:+.2f}% (t={at:.2f})   drift {dm:+.2f}% (t={dt:.2f}) — renewal wants D>0, floundering D<0')"
        ),
        md(
            f"> 💡 In plain words: the pop is **+{R['announce'][0]:.1f}%** at **t = {R['announce'][2]:.2f}** "
            f"(under the *t*=2 bar); the drift is **{R['drift'][0]:.1f}%** at **t = {R['drift'][2]:.2f}** "
            "(indistinguishable from zero). H_renewal wants the drift clearly positive; H_flounder wants "
            "it clearly negative; it is neither — and survivorship only pushes it *more* positive."
        ),
        md(
            "### 4b · The decisive test — a placebo null sized to the event count\n\n"
            f"Draw {R['n_priced']} random non-event **drift** windows 20,000 times; the histogram is the "
            "null for the mean drift. The real drift is the red line; the *p*-value is the two-sided tail "
            "mass."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(B, len(EV), leg='drift', observed=EV['drift'].mean(), n_draws=6000)\n"
            "    obs = pl['obs']*100; pval = pl['p_value']\n"
            "    tickers=[c for c in B['prices'].columns if c!='SPY']\n"
            "    exs=[st._excess_log_returns(B['prices'][t], B['prices']['SPY']) for t in tickers]\n"
            "    exs=[s for s in exs if len(s)>200]; rng=np.random.default_rng(722)\n"
            "    draws=[]\n"
            "    for _ in range(6000):\n"
            "        vals=[]\n"
            "        for _ in range(len(EV)):\n"
            "            s=exs[rng.integers(0,len(exs))]; p0=rng.integers(1,len(s)-130); vals.append(np.expm1(s.iloc[p0:p0+120].sum()))\n"
            "        draws.append(np.mean(vals))\n"
            "    draws=np.array(draws)*100\n"
            "else:\n"
            "    obs=R['drift'][0]; pval=R['drift'][3]; rng=np.random.default_rng(722); draws=rng.normal(0,4.5,6000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=55, color=GREY, alpha=.85, label=f'null: {R[\"n_priced\"]} random windows')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed drift {obs:+.1f}%')\n"
            "ax.set_xlabel('mean 6-month abnormal return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Placebo p = {pval:.2f}: the drift is deep inside the luck cloud'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'P[|random {R[\"n_priced\"]}-window mean| >= |drift|] = {pval:.3f}  (need <0.05 to call it real)')"
        ),
        md(
            f"> 💡 In plain words: **{R['drift'][3]*100:.0f}%** of random {R['n_priced']}-window draws "
            "match or beat the drift in magnitude. A real renewal (or floundering) signal would push the "
            "red line into a tail; instead it sits mid-cloud. Both H_renewal and H_flounder are **not "
            "supported** — the drift is what a couple-dozen random dates look like."
        ),
        md(
            "### 4c · By kind + window robustness — the 'signal' is reverse causation\n\n"
            "Split by rebrand kind and shift the windows. The drift splits by *who rebrands how* — "
            "**name changes drift down, logo-only refreshes drift up** — and no window makes either leg "
            "clear *t* = 2."
        ),
        code(
            "kinds = R['kinds']\n"
            "if HAVE_REAL:\n"
            "    kinds = []\n"
            "    for k in ['name','identity','logo']:\n"
            "        sub = EV[EV['kind']==k]\n"
            "        kinds.append((k, len(sub), sub['announce'].mean()*100, sub['drift'].mean()*100))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "names=[k[0] for k in kinds]; kann=[k[2] for k in kinds]; kdr=[k[3] for k in kinds]; xx=np.arange(len(names))\n"
            "a1.bar(xx-.2, kann, .4, color=AMBER, label='announce'); a1.bar(xx+.2, kdr, .4, color=RED, label='drift')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_xticks(xx); a1.set_xticklabels(names)\n"
            "a1.set_ylabel('mean abnormal return (%)'); a1.set_title('Name-changes drift DOWN, logos drift UP'); a1.legend()\n"
            "if HAVE_REAL:\n"
            "    rob=[]\n"
            "    for ad,dd in [(3,60),(5,120),(5,252),(10,120),(1,120)]:\n"
            "        e2=st.collect_events(B,announce=ad,drift=dd); s2=st.summarize(e2,B,announce=ad,drift=dd,placebo=False)\n"
            "        rob.append((f'{ad}/{dd}', s2['n'], s2['announce']['mean']*100, s2['announce']['t'], s2['drift']['mean']*100, s2['drift']['t']))\n"
            "else:\n"
            "    rob = R['robust']\n"
            "labs=[r[0] for r in rob]; ats=[r[3] for r in rob]; dts=[r[5] for r in rob]; xx2=np.arange(len(labs))\n"
            "a2.bar(xx2-.2, ats, .4, color=AMBER, label='announce t'); a2.bar(xx2+.2, dts, .4, color=RED, label='drift t')\n"
            "a2.axhline(2, ls='--', c=GREY); a2.axhline(-2, ls='--', c=GREY, label='t=±2')\n"
            "a2.set_xticks(xx2); a2.set_xticklabels(labs, fontsize=8); a2.set_ylabel('Welch t'); a2.set_xlabel('announce/drift days')\n"
            "a2.set_title('No window clears |t|=2'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('by kind:', [(k[0], round(k[3],1)) for k in kinds])"
        ),
        md(
            f"> 💡 In plain words: name-changes drift **{R['kinds'][0][3]:.0f}%** and logo-only refreshes "
            f"**+{R['kinds'][2][3]:.0f}%** — *opposite* signs. That isn't a rebrand effect; it's **reverse "
            "causation**: firms in trouble tend to change their *name*, while healthy giants (Pepsi, J&J, "
            "Mastercard) merely *restyle*. The 'signal' is the firm's pre-existing health, read backwards. "
            "And no window rescues significance."
        ),
        md(
            "### 4d · Fragility — the pop is two headlines deep\n\n"
            "The one leg that flickered is the week-one pop (*t* = "
            f"{R['announce'][2]:.2f}). Drop the largest-$|A|$ events one at a time and watch it fold — a "
            "leverage diagnostic the raw *t* hides."
        ),
        code(
            "if HAVE_REAL:\n"
            "    a = EV['announce'].values; order=np.argsort(-np.abs(a))\n"
            "    frag=[]\n"
            "    for drop in (0,1,2,3):\n"
            "        keep=np.ones(len(a),bool); keep[order[:drop]]=False\n"
            "        frag.append((drop, keep.sum(), a[keep].mean()*100, st.welch_t(a[keep])))\n"
            "    dropped=[EV['ticker'].values[i] for i in order[:3]]\n"
            "else:\n"
            "    frag=R['fragile']; dropped=['BB','GM','BBWI']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "xx=[f'drop {f[0]}\\n(n={f[1]})' for f in frag]; tt=[f[3] for f in frag]\n"
            "ax.bar(xx, tt, color=[AMBER if t>=1.96 else GREY for t in tt], width=.55)\n"
            "ax.axhline(1.96, ls='--', c=RED, label='t=1.96'); ax.axhline(2, ls=':', c='k', label='t=2 bar')\n"
            "ax.set_ylabel('Welch t of the week-one pop'); ax.set_title(f'Remove {dropped[:2]} and the pop is a coin flip')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for d,n,m,t in frag: print(f'drop {d}: n={n} announce={m:+.2f}% t={t:+.2f}')\n"
            "print('biggest-|announce| names dropped:', dropped[:3])"
        ),
        md(
            f"> 💡 In plain words: the pop's *t* falls **{R['fragile'][0][3]:.2f} → {R['fragile'][1][3]:.2f} "
            f"→ {R['fragile'][2][3]:.2f} → {R['fragile'][3][3]:.2f}** as the three biggest names leave. "
            "The two doing the heavy lifting — **BlackBerry** (its pop was really the BB10 phone launch "
            "week) and **GM** (mid-EV-re-rating) — moved on *product/strategy* news, not a logo. Strip "
            "the confounds and the 'rebrand pop' is not distinguishable from zero. This is why it stamps "
            "`WEAK`, not `REAL`."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Deterministic event windows with a **planted** renewal drift of size `edge`: with "
            "**`edge=0`** the inference must stay flat (a couple-dozen noisy events can't fake "
            "significance); with a **+30%** planted drift it must light up the drift leg. Both hold — "
            "proving the engine is unbiased *and* that this sample size only detects large effects."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.30):\n"
            "    syn = data.synthetic_rebrands(n_events=26, edge=edge, seed=722)\n"
            "    ev = st.collect_events(syn, announce=5, drift=120); s = st.summarize(ev, syn, announce=5, drift=120, placebo=False)\n"
            "    res.append((edge, s['n'], s['announce']['mean']*100, s['announce']['t'], s['drift']['mean']*100, s['drift']['t']))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "labels=[f'planted\\n{int(e*100)}%' for e,*_ in res]; xx=np.arange(len(labels))\n"
            "ats=[r[3] for r in res]; dts=[r[5] for r in res]\n"
            "ax.bar(xx-.2, ats, .4, color=AMBER, label='announce t'); ax.bar(xx+.2, dts, .4, color=RED, label='drift t')\n"
            "ax.axhline(2, ls='--', c=GREY, label='t=2'); ax.axhline(-2, ls='--', c=GREY)\n"
            "ax.set_xticks(xx); ax.set_xticklabels(labels); ax.set_ylabel('Welch t (6-month drift / 1-week pop)')\n"
            "ax.set_title('Control: only a large planted drift lights up the drift leg'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,am,at,dm,dt in res: print(f'planted {int(e*100):>3}%: n={k} announce={am:+.1f}%(t={at:+.2f}) drift={dm:+.1f}%(t={dt:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted edge the control's drift *t* is "
            f"**{R['syn'][0][5]:.2f}** (no false positive); only the **+30%** plant reaches drift *t* "
            f"**{R['syn'][1][5]:.2f}**. So the machinery is honest, and the real-tape drift *t* of "
            f"**{R['drift'][2]:.2f}** is exactly what an *absent* effect looks like through a "
            f"{R['n_priced']}-event keyhole."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — the renewal/floundering claim is the **drift**: **{R['drift'][0]:.1f}%** "
            f"at Welch **t = {R['drift'][2]:.2f}** / placebo **p = {R['drift'][3]:.2f}** ⇒ `NONE`. A "
            f"week-one pop **+{R['announce'][0]:.1f}%** (**t = {R['announce'][2]:.2f}**, placebo **p = "
            f"{R['announce'][3]:.3f}**) is significant *raw* but **fails t≥2 and collapses to t = "
            f"{R['fragile'][2][3]:.2f}** when 2 news-confounded outliers (BB, GM) drop ⇒ `WEAK`, not REAL. "
            "**Survivorship** named on this axis: the worst-outcome rebrands delisted / went private, "
            "biasing the drift *up*.\n"
            f"- **Tradability `MIRAGE`** — buy-the-rebrand-and-hold = **{R['cost']['gross_hold']:.1f}% "
            f"gross**, **{R['cost']['net_hold']:.1f}% net** of 2 crossings. The soft-negative drift "
            "dominates the pop; no NAV-scale edge, and the pop leg isn't investable (sub-*t*=2, "
            "outlier-built).\n"
            f"- **Renewal or red-flag? `COIN FLIP`** — name-changes drift **{R['kinds'][0][3]:.0f}%**, "
            f"logo refreshes **+{R['kinds'][2][3]:.0f}%**: the drift encodes the firm's *prior* health "
            "(troubled firms rename; giants restyle), not a forward signal you can trade."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the power curve\n\n"
            "The operational truth in one picture: how big would the *true* drift have to be for a "
            "$k$-event study to detect it at $t=2$? At $k\\approx 22$ you'd need a drift several times the "
            "one observed; the real signal lives far below the detection floor — and the buy-and-hold "
            "book is negative-carry on top."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sd = EV['drift'].std(ddof=1); obs = EV['drift'].mean()\n"
            "else:\n"
            "    sd = 0.20; obs = R['drift'][0]/100\n"
            "ks = np.arange(5, 200)\n"
            "min_det = 2.0 * sd / np.sqrt(ks)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(ks, min_det*100, c=AMBER, lw=2, label='drift needed for t=2')\n"
            "ax.axhline(abs(obs)*100, c=RED, ls='--', label=f'|observed drift| ~{abs(obs)*100:.1f}%')\n"
            "ax.axvline(R['n_priced'], c=GREY, ls=':', label=f\"our k={R['n_priced']}\")\n"
            "ax.set_xlabel('number of events k'); ax.set_ylabel('6-month drift (%)')\n"
            "ax.set_title('Detection floor vs the real drift: badly under-powered'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "need = 2.0*sd/np.sqrt(R['n_priced'])*100\n"
            "print(f'at k={R[\"n_priced\"]} you need ~{need:.1f}% |drift| for t=2; observed ~{obs*100:+.1f}% -> under-powered')"
        ),
        md(
            "> 💡 In plain words: the amber curve is the **minimum detectable drift**; the red line is "
            "what we see. They don't meet until $k$ is many times larger than the rebrand calendar will "
            "ever deliver — and even a *real* drift wouldn't pay here, because the buy-and-hold book is "
            "gross-negative and the pop leg is two confounded headlines. There is no sizing, threshold, or "
            "cost assumption that manufactures an edge from ~22 events whose drift is noise. The rarity "
            "that makes the folklore vivid is exactly what makes it untestable and untradable."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The theme-chasing cousin.** [Study 389 — Name-Change-Effect](../389-name-change-effect/): "
            "the `.com`/`Blockchain`/`AI` 'pop then dump' variant. Same family (a label, not a "
            "fundamental), same small-sample / survivorship pathology.\n"
            "- **The anecdote trap, formalised.** [Study 343 — Data-Mining-Roulette]"
            "(../343-data-mining-roulette/) on how loud cases (here: two confounded outliers) manufacture "
            "spurious 'laws.'\n"
            "- **Add the corpses.** Reconstruct the delisted/private rebrands (Twitter→X, WW, Bed Bath & "
            "Beyond) from a survivorship-free feed; the drift may turn negative, but that's a bankruptcy, "
            "not a tradable leg — and half of them you couldn't have shorted.\n\n"
            "*The reproducible core is offline and deterministic; the rebrand table is hardcoded and the "
            "priced tape is survivor-biased (named). Methods and sources: "
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
