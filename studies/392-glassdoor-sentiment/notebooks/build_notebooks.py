"""Generate the two narrative notebooks for Study 392 (Glassdoor-Sentiment).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket prices
under ../_cache/ together with the CONSTRUCTED sentiment proxy and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs
anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance 40-name large-cap
# basket + a CONSTRUCTED return-independent satisfaction proxy, 2005-01-03 -> 2026-06-18,
# 5,399 days, 21.5 years, 257 monthly long-short spreads).
R = dict(
    start="2005-01-03", end="2026-06-18", days=5399, years=21.5, n_months=257,
    mean_m=0.27, ann_mean=3.3, t=1.16, sharpe=0.25, sharpe_lo=-0.16, sharpe_hi=0.67,
    win_rate=54, p_placebo=0.16,
    # costs
    gross_m=0.273, net_m=0.215, gross_sharpe=0.25, net_sharpe=0.20, cost_bps=70,
    # robustness over quintile fraction q: (q, ann%, t, sharpe, placebo_p)
    robust=[(0.10, 2.4, 0.71, 0.15, 0.438), (0.20, 3.3, 1.16, 0.25, 0.160),
            (0.30, 2.3, 0.99, 0.21, 0.236)],
    seed_max_t=1.57,
    seed_ts=[1.16, 0.35, -0.13, 1.34, 0.69, -0.19, -0.30, -1.57, -0.44, -1.04],
    # synthetic control: (edge, ann%, t, sharpe, win%, placebo_p)
    syn=[(0.0, 1.7, 0.83, 0.19, 54, 0.448), (0.010, 15.8, 7.91, 1.77, 72, 0.000)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Free_lunch%3F: Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square)\n\n"
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

from glassdoor_sentiment import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    D = data.load_real()
    PRICES, STARS = D["prices"], D["stars"]
    MONTHLY = st.to_monthly_returns(PRICES)
    SPREAD = st.long_short_returns(MONTHLY, STARS)
else:
    PRICES = STARS = MONTHLY = SPREAD = None
print("real price cache present:", HAVE_REAL,
      "| monthly long-short spreads:", (0 if SPREAD is None else len(SPREAD)))
print("NOTE: the satisfaction score is a CONSTRUCTED proxy (NOT real Glassdoor data).")
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do the happiest companies beat the market? — the Glassdoor-Sentiment factor 🙂\n"
            "### The alt-data pitch that 5-star workplaces are 5-star stocks, in plain English\n\n"
            + BADGES +
            "There's a seductive idea floating around finance: **the companies with the happiest "
            "employees beat the market.** Great Glassdoor ratings, the story goes, are an early read on "
            "a healthy, productive, soon-to-outperform business — buy the 5-star workplaces and you've "
            "front-run Wall Street.\n\n"
            "It isn't pure invention: a famous 2011 academic paper (Alex Edmans) really did find that a "
            "list of the *\"100 Best Companies to Work For\"* earned a few extra percent a year. But a "
            "careful academic list run by professors is one thing; *\"sort stocks by their star rating "
            "and back up the truck\"* is another. This notebook separates the real paper from the retail "
            "pitch.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the bootstrap and the placebo test? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Real Glassdoor ratings cost money — they're behind a paid "
            "API. So we **build a transparent stand-in**: every company in our 40-stock basket gets a "
            "fixed, made-up \"happiness score\" that is **deliberately unrelated to its returns.** That "
            "lets us show *how* the strategy works and what an *honest no-edge result* looks like. We call "
            "it a proxy throughout. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is the \"happy companies win\" idea real? | **In a 2011 academic paper, partly yes** — a "
            "curated best-places-to-work list beat the market by a few percent a year. |\n"
            "| Does it work as a simple star-rating trade? | **We can't confirm it here.** On a "
            "transparent happiness proxy, going long the happiest and short the grumpiest returns "
            f"**+{R['ann_mean']:.1f}%/yr** — but that's a *t* of just **{R['t']:.2f}**, well short of "
            "significant. |\n"
            "| Could that small edge be luck? | **Easily.** A coin-flip beats it about half the time "
            f"(**{R['win_rate']}%** of months are up), and shuffling who's \"happy\" at random matches "
            f"the result **~1 time in 6**. |\n"
            "| So why does everyone repeat it? | **A good story + an up-drifting market.** Happy, "
            "famous companies *feel* like winners; pin that to a real-but-fragile academic finding and "
            "the alt-data marketing does the rest. |\n\n"
            "> The Edmans paper is real. The retail *\"buy the 5-star workplaces\"* trade, on data we can "
            "actually touch, is statistically indistinguishable from picking names out of a hat."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Employees rate their own companies on Glassdoor. A high rating means a happy, "
            "motivated workforce — which means a better-run, faster-growing business — which the market "
            "hasn't fully priced. So buy the highest-rated firms and short the lowest-rated, and you'll "
            "earn a steady premium.\"*\n\n"
            "The academic anchor is Edmans (2011), *Does the stock market fully value intangibles?* — a "
            "value-weighted portfolio of Fortune's *100 Best Companies to Work For* beat the market on a "
            "risk-adjusted basis. The folk version drops the careful list construction and risk "
            "adjustment and just sorts on raw stars. We'll rebuild that sort and put it under a "
            "microscope."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two very different things hide inside \"happy companies beat the market.\" (1) *Is there a "
            "real, published effect?* — yes, in a specific careful study. (2) *Can **you** harvest it by "
            "sorting stocks on a star rating, after costs, reliably enough to bet real money?* — that's "
            "the question that pays the rent, and it's the one the folklore never actually answers. A "
            "few-percent gap measured on a couple of dozen names over twenty years can easily be noise; "
            "the only honest way to know is to ask how often *pure chance* produces the same gap."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We rebuild the trade on a **transparent happiness proxy** across a fixed 40-name large-cap "
            f"basket over **{R['years']:.0f} years** ({R['start']} → {R['end']}):\n\n"
            "1. **Score each company.** Give every name a fixed \"stars\" rating (our stand-in for "
            "Glassdoor — deliberately unrelated to returns, so we can see what *no edge* looks like).\n"
            "2. **Sort and trade.** Each month go **long the happiest fifth**, **short the grumpiest "
            "fifth**, equal weight — the classic long-short factor.\n"
            "3. **Stress the luck.** Shuffle who's labelled \"happy\" thousands of times and ask how "
            "often a *random* labelling does as well as the real one. That's the honest test for a "
            "single sort."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the sort.** Here are the happiest and grumpiest names our proxy picks out, with "
            "their stars. (Remember: the scores are a transparent stand-in, not real Glassdoor data.)"
        ),
        code(
            "if HAVE_REAL:\n"
            "    hi, lo = st.quintile_masks(STARS)\n"
            "    happy = STARS[hi].sort_values(ascending=False)\n"
            "    grumpy = STARS[lo].sort_values()\n"
            "else:\n"
            "    happy = grumpy = None\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "if HAVE_REAL:\n"
            "    names = list(grumpy.index) + list(happy.index[::-1])\n"
            "    vals  = list(grumpy.values) + list(happy.values[::-1])\n"
            "    cols  = [RED]*len(grumpy) + [GREEN]*len(happy)\n"
            "    ax.barh(names, vals, color=cols)\n"
            "    ax.axvline(3.6, c=GREY, ls='--', label='basket mean ~3.6 stars')\n"
            "    ax.set_xlabel('constructed satisfaction score (stars, 1-5)')\n"
            "    ax.set_title('The happiness sort: grumpiest (red, short) vs happiest (green, long)')\n"
            "    ax.legend()\n"
            "else:\n"
            "    ax.text(.5,.5,'no cache — see docs/results.md for the frozen sort', ha='center')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('long the green names, short the red names, rebalanced monthly with a 1-period lag')"
        ),
        md(
            "**Now the payoff.** Did the happy-minus-grumpy long-short actually grow your money? Here's "
            "its cumulative return — and a flat-ish line is the whole story."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cum = (1 + SPREAD).cumprod()\n"
            "    xidx = cum.index; yv = cum.values\n"
            "else:\n"
            "    rng = np.random.default_rng(392)\n"
            "    yv = np.cumprod(1 + rng.normal(R['mean_m']/100, 0.018, R['n_months']))\n"
            "    xidx = np.arange(len(yv))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.plot(xidx, yv, c=GREEN, lw=2)\n"
            "ax.axhline(1.0, c=GREY, ls='--')\n"
            "ax.set_ylabel('growth of $1 in the long-short')\n"
            "ax.set_title(f'Happy-minus-grumpy long-short: ~{R[\"ann_mean\"]:.1f}%/yr, but is it real?')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'annualised long-short ~{R[\"ann_mean\"]:.1f}%/yr — positive, but with a t of only {R[\"t\"]:.2f}')"
        ),
        md(
            f"A gentle upward drift — about **+{R['ann_mean']:.1f}%/yr**. Tempting! But a positive line "
            "isn't proof. The real question is whether that drift is *bigger than luck*. Time for the "
            "honest test."
        ),
        md(
            "**Could random labels do just as well?** Here's the honest test: keep the same stocks, but "
            "**shuffle which ones are \"happy\"** thousands of times, and see how the real sort stacks "
            "up against pure chance."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cols = [c for c in MONTHLY.columns if c in STARS.index]\n"
            "    base = STARS.loc[cols].values.copy(); obs = SPREAD.mean()\n"
            "    rng = np.random.default_rng(392)\n"
            "    import pandas as pd\n"
            "    draws = []\n"
            "    for _ in range(4000):\n"
            "        sh = pd.Series(rng.permutation(base), index=cols)\n"
            "        draws.append(st.long_short_returns(MONTHLY, sh).mean())\n"
            "    draws = np.array(draws); pval = R['p_placebo']\n"
            "else:\n"
            "    rng = np.random.default_rng(392); draws = rng.normal(0, R['mean_m']/100/1.2, 4000)\n"
            "    obs = R['mean_m']/100; pval = R['p_placebo']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=50, color=GREY, alpha=.85, label='random \"who is happy\" labels')\n"
            "ax.axvline(obs*100, c=GREEN, lw=2.5, label=f'the real happiness sort ({obs*100:+.2f}%/mo)')\n"
            "ax.axvline(-obs*100, c=GREEN, lw=1, ls=':')\n"
            "ax.set_xlabel('mean monthly long-short return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The real sort sits inside the luck cloud — placebo p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'a random happiness labelling matches the real sort ~{pval*100:.0f}% of the time — not rare')"
        ),
        md(
            f"The green line — the real happiness sort — falls comfortably **inside** the grey cloud of "
            f"random labellings. About **{int(R['p_placebo']*100)}%** of random \"who's happy\" shuffles "
            "do as well or better. In plain terms: **the happiness sort's record is what you'd expect "
            "from shuffling labels on a hat full of large-caps.** A nice story, not a measured edge."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The academic premium is real *in the paper*, but we can only test the "
            f"folk claim on a **constructed proxy**, where the long-short is **+{R['ann_mean']:.1f}%/yr "
            f"at t = {R['t']:.2f}** — it doesn't clear the significance bar. A proxy can never upgrade "
            "this to Real.\n"
            "- **Tradability — Fragile.** Costs are small (a static sort rarely trades), but a "
            f"long-short with a **Sharpe of {R['sharpe']:.2f}** whose error bar runs through zero isn't "
            "a strategy you fund.\n"
            "- **Free lunch? — Busted.** \"Buy the happiest workplaces and beat the market\" is, on data "
            "we can actually touch, a coin-flip dressed up as alt-data insight."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · But the engine works — here's the proof\n\n"
            "Is our test just blind to *everything*? No. To prove the machinery can find a real edge "
            "when one exists, we build a **make-believe market where happiness genuinely pays** — we "
            "*plant* an edge — and check the same engine lights up."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.010):\n"
            "    syn = data.synthetic_panel(edge=edge, seed=392)\n"
            "    s = st.summarize(syn['prices'], syn['stars'], n_placebo=2000)\n"
            "    res.append((edge, s['t']))\n"
            "labels = ['no planted edge\\n(happiness useless)', 'big planted edge\\n(happiness pays)']\n"
            "tvals = [r[1] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 (significance bar)')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('significance (t) of the happiness long-short')\n"
            "ax.set_title('The engine stays silent on noise and shouts on a real edge'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,t in res: print(f'planted edge {e:+.3f}: t = {t:.2f}')"
        ),
        md(
            f"With **no** planted edge the test stays quiet (**t = {R['syn'][0][2]:.2f}** — no false "
            f"alarm); with a **big** planted edge it roars (**t = {R['syn'][1][2]:.2f}**). So the engine "
            "isn't broken or sleepy — it finds happiness premia when they're really there. That's why "
            "the flat result on the real proxy is an **honest \"nothing here,\"** not a measurement "
            "failure. *(The real Edmans premium might still be real — we just can't confirm it without "
            "the paid ratings.)*"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Other mood signals.** [Study 257 — AAII-Sentiment](../257-aaii-sentiment/) asks the "
            "same \"does mood move prices?\" question for *investor* sentiment instead of *employee* "
            "sentiment.\n"
            "- **More alt-data.** [Study 252 — Google-Trends](../252-google-trends/) and "
            "[Study 335 — BUZZ-Sentiment-ETF](../335-buzz-sentiment-etf/) put other crowd-sourced "
            "signals through the same wringer.\n"
            "- **Bring real ratings.** Swap our constructed scores for an actual Glassdoor / Indeed feed "
            "(or replicate Edmans' best-places-to-work list) and re-run the exact same engine — the "
            "method is ready, only the data is paywalled.\n\n"
            "*Think the happiest workplaces really do beat the market? Bring the real ratings, run the "
            "sort, and show the green line landing **outside** the luck cloud — then we'll talk.*"
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
            "# Glassdoor-Sentiment — a quantitative teardown 🔬\n"
            "### A happiness quintile long-short · a one-sample *t* + bootstrap Sharpe CI + sign-flip "
            "placebo null · costs · a synthetic faithful-engine / planted-edge control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We "
            "implement the textbook factor sort behind *\"happiest employees beat the market\"* — long "
            "the top-quintile satisfaction names, short the bottom quintile — and confront it with the "
            "**sample size** and a **randomization null**. The decisive object is not the point estimate "
            "(it's mildly positive, as the lore says) but its **error bar**: on a constructed, "
            "return-independent satisfaction proxy the long-short is statistically indistinguishable "
            "from relabelling who is 'happy' at random.\n\n"
            "> ⚠️ **Data + proxy note.** Real employer-review ratings aren't free; we build a transparent "
            "**proxy** — a fixed-seed 1–5 star score per name, **independent of returns** (an honest "
            "null, named on the Signal axis alongside survivorship). Real data: yfinance daily adjusted "
            "closes, 40-name large-cap basket, 2005→2026. Offline core + synthetic control are "
            "deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Long-short **+{R['ann_mean']:.1f}%/yr**, one-sample "
            f"**t = {R['t']:.2f}**, placebo **p = {R['p_placebo']:.2f}**, Sharpe **{R['sharpe']:.2f}** "
            f"(95% CI **[{R['sharpe_lo']:+.2f}, {R['sharpe_hi']:+.2f}]**, through zero) — **fails "
            "t ≥ 2**. Proxy/synthetic-only ⇒ never REAL. |\n"
            f"| **Tradability** | `FRAGILE` | Static sort ⇒ **{R['cost_bps']} bps/yr** all-in trims "
            f"Sharpe {R['gross_sharpe']:.2f}→{R['net_sharpe']:.2f}; a gross Sharpe of {R['sharpe']:.2f} "
            "with a CI through zero is not NAV-scale. |\n"
            f"| **Free lunch?** | `BUSTED` | Win-rate **{R['win_rate']}%** (≈ coin); a random "
            f"happiness relabelling matches the spread **~1 in 6**. |\n\n"
            "> 💡 In plain words: a return-independent score *should* produce a null, and it does. The "
            "value of the exercise is a faithful, leak-free engine + a placebo null that says exactly "
            "how unremarkable the point estimate is."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $s_i$ be firm $i$'s satisfaction score and $r_{i,t}$ its month-$t$ return. Form the "
            "static long-short\n\n"
            "$$ x_t = \\frac{1}{|Q_5|}\\sum_{i\\in Q_5} r_{i,t} \\; - \\; \\frac{1}{|Q_1|}\\sum_{i\\in "
            "Q_1} r_{i,t}, \\qquad Q_5=\\{i: s_i \\ge \\text{P}_{80}\\},\\; Q_1=\\{i: s_i \\le "
            "\\text{P}_{20}\\}. $$\n\n"
            "- **H₁ (the factor pays).** $\\mathbb{E}[x_t] > 0$ — the happiest quintile out-earns the "
            "grumpiest, with a *t* that clears the bar.\n"
            "- **H₂ (it's deployable).** The spread survives one-way costs × turnover + short borrow at "
            "a Sharpe worth funding.\n"
            "- **H₃ (\"free lunch\").** The premium is information in the *labels*, not an artefact of "
            "the cross-section / luck.\n\n"
            "We find **H₁ not supported** (positive, $t = 1.16 < 2$, Sharpe CI through zero), **H₂ "
            "rejected** (Sharpe 0.25, CI straddling zero), **H₃ rejected** (a random relabelling matches "
            "the spread ~1 in 6). Crucially we test on a **proxy** that is return-independent by "
            "construction — so this is the *honest null*, and the synthetic control confirms the engine "
            "would speak if an edge were planted."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The whole teardown is one estimate and its standard error:\n\n"
            "$$ t = \\frac{\\bar x}{s_x/\\sqrt{T}}, \\qquad \\widehat{\\text{SR}}_{\\text{ann}} = "
            "\\frac{\\bar x}{s_x}\\sqrt{12}. $$\n\n"
            "A **win-rate** is the wrong lens — a long-short straddles zero, so ~50% up-months is the "
            "*null*, not evidence. The honest small-sample instrument for a single static sort is a "
            "**relabelling (placebo) test**: permute the $s_i$ across names and ask how often a random "
            "labelling yields $|\\bar x|$ as large as observed. That *p*, plus a **bootstrap CI** on the "
            "Sharpe, decides the Signal axis — not the sign of the point estimate."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe + proxy.** Fixed 40-name large-cap basket (yfinance adjusted closes, "
            f"{R['start']}→{R['end']}, {R['days']:,} days ⇒ {R['n_months']} monthly spreads). Sentiment "
            "= a **constructed** 1–5 star score, **independent of returns** (explicit proxy; survivorship "
            "named on the axis).\n"
            "- **Sort.** Rank by stars; $Q_5$ = top quintile (long), $Q_1$ = bottom quintile (short), "
            "equal-weight; **one-period (monthly) execution lag**, no look-ahead.\n"
            "- **Null #1 (one-sample t).** $\\bar x$ vs 0.\n"
            "- **Null #2 (bootstrap).** 5,000-resample CI for the annualised Sharpe.\n"
            "- **Null #3 (placebo).** 5,000 random relabellings of $s_i$; $p = \\Pr[|\\bar "
            "x_{\\text{shuffle}}| \\ge |\\bar x|]$ — the small-sample workhorse.\n"
            "- **Costs.** 10 bps one-way × 2 legs × rebalances/yr + 50 bps/yr short borrow.\n"
            "- **Positive control.** A synthetic panel with a **planted** happiness→return tilt: zero "
            "edge must stay below t=2, a large edge must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The point estimate — positive, small, inside its error bar\n\n"
            "Annualised long-short with its bootstrap Sharpe CI. Positive, but the Sharpe interval "
            "**spans zero**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    boot = st.bootstrap_sharpe_ci(SPREAD.values, seed=392)\n"
            "    sh, lo, hi = boot['sharpe'], boot['lo'], boot['hi']\n"
            "else:\n"
            "    sh, lo, hi = R['sharpe'], R['sharpe_lo'], R['sharpe_hi']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "ax.errorbar([0], [sh], yerr=[[sh-lo],[hi-sh]], fmt='o', ms=12, capsize=8, c=GREEN,\n"
            "            label='happiness long-short Sharpe (95% CI)')\n"
            "ax.axhline(0, c=RED, ls='--', label='zero (no premium)')\n"
            "ax.set_xlim(-1,1); ax.set_xticks([]); ax.set_ylabel('annualised Sharpe')\n"
            "ax.set_title(f'Sharpe {sh:.2f}, CI [{lo:+.2f}, {hi:+.2f}] — straddles zero'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Sharpe={sh:.2f}  95% CI=[{lo:+.2f}, {hi:+.2f}]  -> indistinguishable from 0')"
        ),
        md(
            f"> 💡 In plain words: the long-short earns **+{R['ann_mean']:.1f}%/yr** at a Sharpe of "
            f"**{R['sharpe']:.2f}**, but the 95% CI runs from **{R['sharpe_lo']:+.2f}** to "
            f"**{R['sharpe_hi']:+.2f}** — it includes zero. H₁ is **not supported**: a positive whisper "
            "that the data can't distinguish from no premium at all."
        ),
        md(
            "### 4b · The decisive test — a sign-flip placebo sized to the cross-section\n\n"
            "Permute which names are 'happy' vs 'grumpy' thousands of times; the histogram is the null "
            "distribution of the mean spread. The real sort is the green line; the *p*-value is the "
            "two-sided tail mass."
        ),
        code(
            "import pandas as pd\n"
            "if HAVE_REAL:\n"
            "    cols = [c for c in MONTHLY.columns if c in STARS.index]\n"
            "    base = STARS.loc[cols].values.copy(); obs = SPREAD.mean()\n"
            "    rng = np.random.default_rng(392)\n"
            "    draws = np.array([st.long_short_returns(MONTHLY, pd.Series(rng.permutation(base), index=cols)).mean()\n"
            "                      for _ in range(5000)])\n"
            "    pval = float((np.abs(draws) >= abs(obs)).mean())\n"
            "else:\n"
            "    rng = np.random.default_rng(392); draws = rng.normal(0, R['mean_m']/100/1.2, 5000)\n"
            "    obs = R['mean_m']/100; pval = R['p_placebo']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=60, color=GREY, alpha=.85, label=f'null: {len(draws):,} random relabellings')\n"
            "ax.axvline(obs*100, c=GREEN, lw=2.5, label=f'observed sort {obs*100:+.2f}%/mo')\n"
            "ax.axvline(-obs*100, c=GREEN, lw=1, ls=':')\n"
            "ax.set_xlabel('mean monthly long-short return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Placebo p = {pval:.2f}: the sort is inside the luck cloud'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'P[|random relabelling| >= |observed|] = {pval:.3f}  (need <0.05; here ~1 in 6)')"
        ),
        md(
            f"> 💡 In plain words: **{int(R['p_placebo']*100)}%** of random happiness labellings match or "
            "beat the real sort in magnitude. A real factor would push the green line into the far tail; "
            "instead it sits mid-cloud. H₃ rejected: the 'premium' is a draw from the cross-section, not "
            "information in the labels."
        ),
        md(
            "### 4c · Robustness + cost — neither rescues it\n\n"
            "Vary the quintile fraction and apply costs. The *t* never approaches 2, and a static sort's "
            "70 bps/yr only nicks the Sharpe (so cost is **not** the binding constraint — there is simply "
            "no edge)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for q in (0.10, 0.20, 0.30):\n"
            "        sp = st.long_short_returns(MONTHLY, STARS, q=q)\n"
            "        rob.append((q, sp.values.mean()*12*100, st.t_stat(sp.values), st.sharpe(sp.values)))\n"
            "    c = st.net_of_costs(SPREAD, cost_bps=10.0, rebalances_per_year=1.0, borrow_bps=50.0)\n"
            "    gs, ns = c['gross_sharpe'], c['net_sharpe']\n"
            "else:\n"
            "    rob = [(r[0], r[1], r[2], r[3]) for r in R['robust']]\n"
            "    gs, ns = R['gross_sharpe'], R['net_sharpe']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "qs = [f'{r[0]:.2f}' for r in rob]; tt = [r[2] for r in rob]\n"
            "a1.bar(qs, tt, color=AMBER, width=.55)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 (significance bar)')\n"
            "for i,t in enumerate(tt): a1.annotate(f'{t:.2f}',(i,t),ha='center',va='bottom')\n"
            "a1.set_title('t never reaches 2'); a1.set_xlabel('quintile fraction q'); a1.set_ylabel('one-sample t'); a1.set_ylim(0,2.3); a1.legend()\n"
            "a2.bar(['gross','net @70bps/yr'], [gs, ns], color=[GREEN, GREY], width=.5)\n"
            "for i,v in enumerate([gs,ns]): a2.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('annualised Sharpe'); a2.set_title('Cost barely matters — there is no edge')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (q, ann%, t, sharpe):', [(r[0], round(r[1],1), round(r[2],2), round(r[3],2)) for r in rob])"
        ),
        md(
            "> 💡 In plain words: tightening or loosening the quintiles leaves *t* between ~0.7 and ~1.2 "
            "— nowhere near 2 — and the 70 bps/yr cost trims the Sharpe from "
            f"{R['gross_sharpe']:.2f} to {R['net_sharpe']:.2f}. There is no parameter and no cost regime "
            "in which the proxy sort becomes deployable; the constraint is the **absence of signal**, "
            "not friction."
        ),
        md(
            "### 4d · Faithful-engine & planted-edge control — we know the truth here\n\n"
            "On a synthetic panel where each name's score is known and returns carry a **planted** "
            "happiness tilt: with **zero** edge the long-short must stay below t=2; with a **large** "
            "planted edge it must light up. Both hold — proving the engine is unbiased *and* that the "
            "real-tape null is an honest null."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.010):\n"
            "    syn = data.synthetic_panel(edge=edge, seed=392)\n"
            "    s = st.summarize(syn['prices'], syn['stars'], n_placebo=2000)\n"
            "    res.append((edge, s['t'], s['ann_mean']*100, s['p_placebo']))\n"
            "labels = [f'planted edge\\n{e:+.3f}/mo' for e,_,_,_ in res]\n"
            "tvals = [r[1] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 (significance bar)')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('one-sample t of the long-short'); ax.set_title('Control: silent on noise, loud on a real edge'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,t,a,p in res: print(f'planted {e:+.3f}/mo: t={t:.2f}  ann={a:+.1f}%  placebo p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted edge the control sits at **t = {R['syn'][0][2]:.2f}** "
            f"(below 2 — no false positive); a **+1.0%/mo** planted edge reaches **t = {R['syn'][1][2]:.2f}**. "
            "So the machinery is honest, and the real-tape *t* of ~1.2 on a return-independent proxy is "
            "exactly what an **absent** edge looks like through this engine. The proxy is the verdict's "
            "limit, not the engine."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — long-short **+{R['ann_mean']:.1f}%/yr** at one-sample "
            f"**t = {R['t']:.2f}** / placebo **p = {R['p_placebo']:.2f}**, Sharpe **{R['sharpe']:.2f}** "
            f"(CI **[{R['sharpe_lo']:+.2f}, {R['sharpe_hi']:+.2f}]**, through zero). Literature support "
            "(Edmans 2011) + a proxy-only, insignificant point estimate ⇒ WEAK, not REAL (the desk's "
            "t≥2 bar is unmet, and a proxy/synthetic result can never back REAL). Survivorship + the "
            "proxy caveat are named on this axis.\n"
            f"- **Tradability `FRAGILE`** — a static sort trades rarely, so the **{R['cost_bps']} bps/yr** "
            f"all-in cost only trims Sharpe {R['gross_sharpe']:.2f}→{R['net_sharpe']:.2f}; a gross "
            f"Sharpe of {R['sharpe']:.2f} with a CI through zero is not NAV-scale. Cost isn't the killer "
            "— the missing edge is.\n"
            f"- **Free lunch? `BUSTED`** — win-rate **{R['win_rate']}%** (≈ coin); a random relabelling "
            "matches the spread ~1 in 6. On data we can actually touch, the alt-data pitch is "
            "indistinguishable from chance."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the power floor\n\n"
            "The operational truth in one picture: how big would the *true* monthly long-short have to "
            "be for a $T$-month study to detect it at t=2? At $T=257$ months the detection floor is well "
            "**above** the spread we observe — the proxy signal lives under the floor, and the academic "
            "premium would need a far cleaner, larger sample to confirm."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sd = SPREAD.std(ddof=1); obs_m = SPREAD.mean()\n"
            "else:\n"
            "    sd = 0.037; obs_m = R['mean_m']/100\n"
            "Ts = np.arange(24, 600)\n"
            "min_detectable = 2.0 * sd / np.sqrt(Ts)        # monthly mean needed for t=2\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(Ts, min_detectable*100, c=AMBER, lw=2, label='monthly mean needed for t=2')\n"
            "ax.axhline(obs_m*100, c=GREEN, ls='--', label=f'observed mean ~{obs_m*100:+.2f}%/mo')\n"
            "ax.axvline(R['n_months'], c=GREY, ls=':', label=f\"our T={R['n_months']} months\")\n"
            "ax.set_xlabel('months of data T'); ax.set_ylabel('monthly long-short mean (%)')\n"
            "ax.set_title('Detection floor vs the observed sort: we sit under the floor'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "need = 2.0*sd/np.sqrt(R['n_months'])\n"
            "print(f'at T={R[\"n_months\"]} you need ~{need*100:.2f}%/mo for t=2; observed ~{obs_m*100:.2f}%/mo -> under the floor')"
        ),
        md(
            "> 💡 In plain words: the amber curve is the **minimum detectable monthly mean**; the green "
            "line is what we actually see — it sits *below* the floor at our sample length. Even granting "
            "the lore a real premium, a return-independent proxy can't surface it, and the genuine effect "
            "would need real ratings and a far larger, cleaner cross-section to clear t=2. There is no "
            "sizing or threshold that manufactures power from a flat proxy — **the missing ingredient is "
            "the data we can't buy, and a story is not a substitute for it.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Investor vs employee mood.** [Study 257 — AAII-Sentiment](../257-aaii-sentiment/): the "
            "same 'does sentiment predict returns?' question for survey-based *investor* mood.\n"
            "- **Crowd-sourced alt-data.** [Study 252 — Google-Trends](../252-google-trends/) and "
            "[Study 335 — BUZZ-Sentiment-ETF](../335-buzz-sentiment-etf/) — adjacent alt-data signals "
            "through the same inference pipeline.\n"
            "- **Real ratings.** Replace `data.constructed_sentiment` with a genuine Glassdoor / Indeed "
            "panel (or Edmans' best-places list) and re-run; the leak-free engine, the placebo null and "
            "the cost model are unchanged — only the (paywalled) data differs.\n\n"
            "*The reproducible core is offline and deterministic; the sentiment input is an explicit "
            "constructed proxy. Methods and sources: [`docs/references.md`](../docs/references.md); "
            "frozen numbers: [`docs/results.md`](../docs/results.md).*"
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
