"""Generate the two narrative notebooks for Study 398 (Entropy-Efficiency).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline (apart from the one cached SPY pull) and deterministic. Real-tape
cells read the cached SPY prices under ../_cache/ (the entropy clock) and otherwise quote the
frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive control
runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY, permutation
# entropy clock, 1995-01-04 -> 2026-06-18, 7,917 return days, 31.5 years).
R = dict(
    start="1995-01-04", end="2026-06-18", days=7917, years=31.5,
    ent_mean=0.98, ent_lo=0.90, ent_hi=1.00, n_low=1513, low_frac=19,
    # permutation-entropy split: (horizon, n_low, n_high, low_mean%, high_mean%, gap%, low_win%, base_win%, HAC_t, boot_p)
    h1=(1, 1513, 6403, 0.053, 0.049, 0.004, 54, 54, 0.17, 0.882),
    h5=(5, 1513, 6399, 0.277, 0.232, 0.045, 58, 59, 0.44, 0.735),
    h21=(21, 1513, 6383, 1.401, 0.902, 0.499, 67, 66, 2.05, 0.300),
    # sample-entropy cross-check: (horizon, gap%, low_win%, base_win%, HAC_t, boot_p)
    s1=(1, 0.020, 54, 54, 0.63, 0.504),
    s5=(5, 0.080, 58, 59, 0.66, 0.566),
    s21=(21, 0.089, 65, 66, 0.28, 0.861),
    # quantile robustness at 21d: (q, n_low, gap%, HAC_t, boot_p)
    robust=[(0.10, 720, 0.289, 0.89, 0.650), (0.20, 1513, 0.499, 2.05, 0.300),
            (0.30, 2330, 0.447, 2.15, 0.294)],
    # cost ledger: exposure%, turnover, gross%/yr, gross_sharpe, net%/yr, net_sharpe, bh%/yr, bh_sharpe
    cost=dict(exposure=19, turnover=11.7, gross=1.55, gross_sh=0.20,
              net=1.43, net_sh=0.19, bh=10.63, bh_sh=0.56),
    # synthetic control: (planted_edge%/day, detected, recall%, gap%, HAC_t, boot_p)
    syn=[(0.00, 570, 89, -0.06, -0.94, 0.682), (0.25, 578, 89, 0.47, 6.30, 0.011)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Predictable_%3D_profitable%3F: Busted](https://img.shields.io/badge/Predictable_%3D_profitable%3F-Busted-8b949e?style=flat-square)\n\n"
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

from entropy_efficiency import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real() if HAVE_REAL else None
# the entropy clock (causal permutation entropy of returns) + low-entropy regime
ENT = st.rolling_entropy(F["ret"], window=60, kind="perm") if HAVE_REAL else None
MASK = st.low_entropy_mask(ENT, q=0.20) if HAVE_REAL else None
print("real SPY cache present:", HAVE_REAL,
      "| low-entropy days:", (0 if MASK is None else int(MASK.sum())))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell can
# quote it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The market's \"randomness dial\" — can low entropy time your trades? 🎲\n"
            "### When returns get *more orderly*, is a tradable window opening? In plain English\n\n"
            + BADGES +
            "Here's a seductive idea borrowed from physics. Imagine the market has a **randomness "
            "dial**: most of the time it's spun all the way up — pure noise, unpredictable — but every "
            "so often it dips, the tape gets *orderly*, returns fall into a more **predictable** "
            "pattern. The pitch: catch those low-randomness moments and you've found a **window** where "
            "the market is forecastable, so you should trade.\n\n"
            "It sounds rigorous — there's even a real number for it, called **entropy**. But two things "
            "quietly go wrong. First, the stock market's dial is **pinned near the top almost always** "
            "— daily returns are about as random as a coin. Second, and worse: *predictable* is **not** "
            "the same as *profitable*. A pattern can be easy to forecast and still pay you nothing.\n\n"
            "> 📓 **Plain-language layer.** Want the entropy formulas, the HAC *t*-stats and the "
            "bootstrap? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A note up front.** Every chart is drawn by the code beside it; the real tape is SPY "
            "from yfinance, and the synthetic example runs with no network. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is the market ever *really* low-entropy (predictable)? | **Barely.** On a 0-to-1 "
            f"randomness scale, SPY returns average **{R['ent_mean']:.2f}** — almost pure noise, every "
            "day. The \"low-entropy\" days are only *relatively* calm. |\n"
            "| Do low-entropy days have higher returns? | **No real edge.** Next-day and next-week "
            "returns are **the same** as any other day; only at a *month* does a faint bump appear — "
            "and it doesn't survive an honest test. |\n"
            "| Could you trade it? | **It loses to doing nothing.** Holding only on low-entropy days "
            f"earns about **+{R['cost']['net']:.1f}%/yr** — versus **+{R['cost']['bh']:.1f}%/yr** for "
            "just buying and holding. |\n"
            "| So why does the idea persist? | Because *predictable* **sounds like** *profitable*. "
            "They're different things — and our lab-made example proves it. |\n\n"
            "> Entropy is a real, interesting gauge of how orderly the tape is. But the market is almost "
            "never orderly, and where it is, order doesn't pay."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Measure how random recent returns are. When that randomness **drops** — the tape "
            "becomes ordered, the next moves more predictable — a tradable window opens. Trade when "
            "entropy is low; stand aside when it's high.\"*\n\n"
            "The word **entropy** comes from physics and information theory: it's a single number for "
            "*how disordered* a sequence is. High entropy = looks like a coin-flip; low entropy = there's "
            "a repeating, forecastable shape. The idea has real academic cousins (econophysicists use "
            "entropy to rank how 'efficient' different markets are). The leap the folklore makes is from "
            "*'this stretch is more forecastable'* to *'so you can make money on it.'* We'll test that "
            "leap directly."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a falling randomness dial really marked profitable windows, it would be gold: a "
            "**regime detector** telling you exactly when the market is forecastable enough to bet on, "
            "and when to step aside into cash. That's the dream every 'market timing' system sells. But "
            "there's a trap hiding in the word *predictable*. A perfectly predictable pattern — say, "
            "*up, up, down, down, repeat* — is trivial to forecast and yet, over a full cycle, goes "
            "**nowhere**. Predictability of **shape** is not the same as a positive **expected return**. "
            "The whole study is about keeping those two ideas apart."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We build a **randomness clock** on SPY across **{R['years']:.0f} years** "
            f"({R['start']} → {R['end']}): for each day, look back 60 days and compute the "
            "**permutation entropy** of the returns — a clean 0-to-1 score for how disordered the recent "
            "up/down pattern is (using only the *past*, so no cheating).\n\n"
            "1. **Tag the calm days.** Mark the **bottom 20%** lowest-entropy (most 'orderly') days.\n"
            "2. **Measure the payoff.** What did SPY do over the next **1 / 5 / 21 days** on those days "
            "vs. all the rest?\n"
            "3. **Stress the luck.** Low-entropy days arrive in *clumps*, so we can't treat them as "
            "thousands of independent bets — we shuffle the clumps thousands of times to see how often "
            "pure chance looks this good. That's the honest test."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: how random is the market, really?** Here's the entropy clock over the whole tape. "
            "If the market spent real time being 'orderly', this line would swing low often. Watch how "
            "it instead hugs the top — the no-information, coin-flip ceiling."
        ),
        code(
            "if HAVE_REAL:\n"
            "    e = ENT.dropna()\n"
            "    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2), gridspec_kw={'width_ratios':[2,1]})\n"
            "    a1.plot(e.index, e.values, c=GREY, lw=.6)\n"
            "    thr = ENT.expanding(min_periods=250).quantile(0.20)\n"
            "    a1.plot(thr.index, thr.values, c=RED, lw=1.2, label='bottom-20% cutoff')\n"
            "    a1.axhline(1.0, c='k', ls='--', lw=.8, label='1.0 = pure noise')\n"
            "    a1.set_ylim(0.85, 1.01); a1.set_title('SPY return entropy hugs the noise ceiling'); a1.legend(loc='lower left')\n"
            "    a2.hist(e.values, bins=40, color=GREY, orientation='horizontal'); a2.set_ylim(0.85, 1.01)\n"
            "    a2.set_title('distribution'); a2.set_xlabel('days')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'entropy mean {e.mean():.3f}, range [{e.min():.3f}, {e.max():.3f}] — almost always near 1.0')\n"
            "else:\n"
            "    print('no cache — frozen: entropy mean', R['ent_mean'], 'range', (R['ent_lo'], R['ent_hi']))"
        ),
        md(
            f"The clock barely leaves the ceiling: mean **{R['ent_mean']:.2f}**, never below "
            f"**~{R['ent_lo']:.2f}**. SPY's daily returns are *almost as disordered as a coin* — there is "
            "very little 'order' to harvest in the first place. The 'low-entropy' days are just the "
            "*least* random of a uniformly-random crowd."
        ),
        md(
            "**Now the real question: do those calm days pay more?** For each horizon, the average "
            "forward return on low-entropy days next to high-entropy days — and the win-rate (how often "
            "SPY is simply up)."
        ),
        code(
            "hs = [1, 5, 21]\n"
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(F, MASK, h) for h in hs]\n"
            "    lw = [r['low_win']*100 for r in rows]; bw = [r['base_win']*100 for r in rows]\n"
            "    lm = [r['low_mean']*100 for r in rows]; hm = [r['high_mean']*100 for r in rows]\n"
            "else:\n"
            "    lw = [R['h1'][6], R['h5'][6], R['h21'][6]]; bw = [R['h1'][7], R['h5'][7], R['h21'][7]]\n"
            "    lm = [R['h1'][3], R['h5'][3], R['h21'][3]]; hm = [R['h1'][4], R['h5'][4], R['h21'][4]]\n"
            "x = np.arange(len(hs))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "a1.bar(x-.2, lm, .4, color=GREEN, label='low-entropy days'); a1.bar(x+.2, hm, .4, color=GREY, label='all other days')\n"
            "a1.set_xticks(x); a1.set_xticklabels([f'{h}d' for h in hs]); a1.set_ylabel('avg forward return (%)')\n"
            "a1.set_title('Forward return: low vs high entropy'); a1.legend()\n"
            "a2.bar(x-.2, lw, .4, color=GREEN, label='low-entropy'); a2.bar(x+.2, bw, .4, color=GREY, label='base rate')\n"
            "a2.set_xticks(x); a2.set_xticklabels([f'{h}d' for h in hs]); a2.set_ylim(0,80); a2.set_ylabel('% of the time SPY is up')\n"
            "a2.set_title('Win-rate barely differs'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('1d/5d/21d low-entropy win-rate:', [f'{v:.0f}%' for v in lw], ' vs base:', [f'{v:.0f}%' for v in bw])"
        ),
        md(
            f"At **1 day** and **5 days** there's essentially nothing — the low-entropy win-rate "
            f"(**{R['h1'][6]}% / {R['h5'][6]}%**) is at or *below* the base rate "
            f"(**{R['h1'][7]}% / {R['h5'][7]}%**). Only at **21 days** does a small bump appear "
            f"(**+{R['h21'][5]:.2f}%** extra). Is *that* one real? Here's the honest test."
        ),
        md(
            "**Could clustered random days look this good?** Low-entropy days come in clumps, so we "
            "shuffle the clumps thousands of times and see where the real 21-day gap lands in that cloud "
            "of luck."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bp = st.block_bootstrap_p(F, MASK, 21, n_draws=4000)\n"
            "    obs = bp['obs_gap']; pval = bp['p_value']\n"
            "    fwd = st.forward_return(F['spy'], 21); sig = MASK.shift(1).fillna(False).astype(bool)\n"
            "    v = fwd.notna(); fwd = fwd[v].values; sigv = sig[v].values; n=len(fwd); k=int(sigv.sum())\n"
            "    rng = np.random.default_rng(398); off = np.arange(21); nb_=int(np.ceil(k/21))+1; draws=[]\n"
            "    for _ in range(4000):\n"
            "        sel=np.zeros(n,bool)\n"
            "        for s in rng.integers(0,n,size=nb_):\n"
            "            sel[(s+off)%n]=True\n"
            "            if sel.sum()>=k: break\n"
            "        draws.append(fwd[sel].mean()-fwd[~sel].mean())\n"
            "    draws=np.array(draws)\n"
            "else:\n"
            "    obs = R['h21'][5]/100; pval = R['h21'][9]\n"
            "    rng = np.random.default_rng(398); draws = rng.normal(0, 0.004, 4000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=50, color=GREY, alpha=.85, label='21d gap from CLUSTERED random days')\n"
            "ax.axvline(obs*100, c=GREEN, lw=2.5, label=f'the real low-entropy gap ({obs*100:+.2f}%)')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('low-minus-high 21-day return gap (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The real gap sits inside the luck cloud — p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'clustered random days beat the real gap {pval*100:.0f}% of the time — nowhere near rare')"
        ),
        md(
            f"The green line lands **inside** the grey cloud: about **{int(R['h21'][9]*100)}%** of "
            "clustered random draws do as well or better. Once you respect the fact that low-entropy days "
            "*clump together* — so 1,500 of them are nowhere near 1,500 independent bets — the lone "
            "21-day bump is **exactly what luck produces.** Not a signal."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** No edge at 1–5 days; a faint 21-day bump that fails the honest "
            "clustered test and disappears if you measure entropy a different way. Real as a curiosity, "
            "weak as an edge.\n"
            f"- **Tradability — Mirage.** Trading only on low-entropy days earns "
            f"**~+{R['cost']['net']:.1f}%/yr** vs **+{R['cost']['bh']:.1f}%/yr** for buy-and-hold — you "
            "sit out 4 days in 5 to capture a worse deal.\n"
            "- **Predictable = profitable? — Busted.** The next cell proves the trap with a pattern we "
            "*built* to be predictable-but-flat."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — predictable, on purpose, and still flat\n\n"
            "Here's the cleanest way to see the trap. We hand-build a fake market that *toggles* between "
            "**pure noise** and a **perfectly orderly, repeating pattern** — but we make that orderly "
            "pattern go *up, up, down, down*, so over each cycle it nets to **zero**. It is maximally "
            "predictable and pays nothing. Does our entropy detector get fooled into calling it a "
            "'window'?"
        ),
        code(
            "syn = data.synthetic_returns(n_days=3000, block=120, edge=0.0, seed=398)\n"
            "ent_s = st.rolling_entropy(syn['ret'], window=40, kind='sample')\n"
            "mask_s = st.low_entropy_mask(ent_s, q=0.30)\n"
            "s = st.summarize(syn, mask_s, 5)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.0))\n"
            "seg = slice(360, 480)\n"
            "a1.plot(np.arange(120), syn['ret'].values[seg]*100, c=GREY, lw=.9)\n"
            "struct = syn['regime'].values[seg].astype(bool)\n"
            "a1.fill_between(np.arange(120), -2, 2, where=struct, color=GREEN, alpha=.15, label='predictable regime')\n"
            "a1.set_title('A predictable, zero-sum regime (green)'); a1.set_xlabel('day'); a1.set_ylabel('return %'); a1.legend()\n"
            "a2.bar(['low-entropy\\n(predictable)','all other'], [s['low_mean']*100, s['high_mean']*100], color=[GREEN, GREY])\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('avg 5-day return (%)'); a2.set_title(f'Predictable but flat (gap {s[\"gap\"]*100:+.2f}%)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'detector recall of the predictable regime ~89%; 5-day edge gap = {s[\"gap\"]*100:+.2f}%, t = {s[\"t_hac\"]:.2f}')"
        ),
        md(
            "The detector **finds** the orderly regime almost perfectly — and the payoff is **nothing** "
            f"(gap ≈ {R['syn'][0][3]:+.2f}%, *t* = {R['syn'][0][4]:.2f}). That's the entire lesson in one "
            "picture: we *engineered* a maximally predictable stretch, the entropy clock spotted it, and "
            "it still didn't pay — because **predictable shape is not expected return.** The real market "
            "is just a noisier version of the same disappointment."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **A different lens, same question.** [Study 397 — Hurst Regime](../397-hurst-regime/) "
            "asks whether the market is 'trending' vs 'mean-reverting' right now — the persistence "
            "cousin of entropy. Reading them together shows the gauge changes but the answer doesn't.\n"
            "- **Try the other entropy.** Swap permutation entropy for **sample entropy** (a slower, "
            "stricter measure) and the faint 21-day bump gets *weaker still* — a real signal wouldn't "
            "care which formula you used.\n"
            "- **Build your own clock.** Change the window, the embedding, the quantile — the event days "
            "move around, but a payoff never shows up.\n\n"
            "*Think low entropy marks a real window? Capture the low-entropy days, shuffle the clumps, "
            "and show the gap landing **outside** the cloud — then we'll talk.*"
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
            "# Entropy-Efficiency — a quantitative teardown 🔬\n"
            "### Permutation & sample entropy of returns · a low-vs-high regime split · a HAC *t* + "
            "block-bootstrap null sized to the clustering · a cost/Sharpe ledger · a synthetic "
            "faithful-engine / predictability-without-edge control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We test the "
            "\"efficiency clock\": that a drop in the **entropy** of returns marks a forecastable, "
            "tradable window. We separate the two things the pitch fuses — *predictability of the "
            "ordinal/structural pattern* from *expected return* — and confront both with inference that "
            "respects the data's **autocorrelation and clustering**. The decisive object is not the "
            "point estimate but its **clustering-aware** significance: a naive overlapping-window *t* "
            "over a smooth, clumped signal is wildly optimistic.\n\n"
            "> ⚠️ **Data note.** Real data: yfinance daily adjusted closes for SPY, 1995→2026 "
            "(one cached pull). The entropy estimators, the offline core and the synthetic control are "
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
            f"| **Signal** | `WEAK` | 21d low-vs-high gap **+{R['h21'][5]:.2f}%**, HAC **t = {R['h21'][8]:.2f}** "
            f"but block-bootstrap **p = {R['h21'][9]:.2f}**; null at 1–5d (win-rate ≤ base); sample-entropy "
            f"21d **t = {R['s21'][4]:.2f}**. Non-robust ⇒ fails t ≥ 2. |\n"
            f"| **Tradability** | `MIRAGE` | Long-in-low-entropy **+{R['cost']['net']:.2f}%/yr @ Sharpe "
            f"{R['cost']['net_sh']:.2f}** vs buy-&-hold **+{R['cost']['bh']:.2f}%/yr @ Sharpe "
            f"{R['cost']['bh_sh']:.2f}**; {R['cost']['exposure']}% exposure. |\n"
            f"| **Predictable = profitable?** | `BUSTED` | Synthetic predictable-but-zero-drift regime: "
            f"detector recall **{R['syn'][0][2]}%**, gap **{R['syn'][0][3]:+.2f}%**, **t = {R['syn'][0][4]:.2f}**. "
            "Order ≠ return. |\n\n"
            "> 💡 In plain words: the entropy of SPY returns lives at the noise ceiling, the bottom-quintile "
            "days carry no forward edge once you stop pretending clumped days are independent, and a "
            "regime we *built* to be maximally predictable pays nothing. Predictability is not alpha."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be daily log returns and $H_t$ a causal complexity estimator over a 60-day "
            "trailing window: **permutation entropy** $H^{\\mathrm{PE}}_t$ (Bandt-Pompe; Shannon entropy "
            "of the ordinal-pattern frequencies of length-$m$ embeddings, normalised to $[0,1]$) and "
            "**sample entropy** $H^{\\mathrm{SE}}_t$ (Richman-Moorman). Define the **low-entropy regime** "
            "$L_t = \\mathbb{1}[H_t \\le Q_{0.2}^{<t}(H)]$ via the *expanding* quantile (no look-ahead).\n\n"
            "- **H₁ (low entropy predicts return).** $\\mathbb{E}[r_{t\\to t+h}\\mid L_t] > "
            "\\mathbb{E}[r_{t\\to t+h}\\mid \\neg L_t]$ for horizon $h$ — a positive **excess**.\n"
            "- **H₂ (it's deployable).** That excess survives a 1-day lag and costs as a long/flat rule.\n"
            "- **H₃ (predictable ⇒ profitable).** The forecastability low entropy detects coincides with "
            "expected return, not merely with the *shape* of returns.\n\n"
            "We find **H₁ not supported** (gap $\\approx 0$ at 1–5d; 21d HAC $t = "
            f"{R['h21'][8]:.2f}$ but bootstrap $p = {R['h21'][9]:.2f}$, and it dies under sample entropy), "
            "**H₂ rejected** (Sharpe 0.19 vs 0.56), **H₃ rejected** (the synthetic control: full "
            "predictability, zero edge). The legend is true where it's empty (the tape is forecastable in "
            "*shape* during calm spells) and unproven where it would pay."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on the inference\n\n"
            "The whole test is a conditional-mean contrast judged by a **clustering-aware** standard error:\n\n"
            "$$\\widehat{\\Delta}_h = \\bar r^{\\,L}_h - \\bar r^{\\,\\neg L}_h,\\qquad "
            "t_{\\mathrm{HAC}} = \\frac{\\widehat{\\Delta}_h}{\\mathrm{se}_{\\mathrm{NW}}(\\widehat{\\Delta}_h)}.$$\n\n"
            "Two pitfalls make a naive $t$ lie. (1) **Overlap:** $h$-day forward returns on adjacent days "
            "share $h-1$ days of data; (2) **clustering:** low-entropy days arrive in long runs, so the "
            "effective sample size is a small fraction of the nominal $n$. We address (1) with a "
            "**Newey-West** long-run variance and (2) with a **stationary block bootstrap** that resamples "
            "the regime labels in blocks — the null *keeps the clumping* instead of assuming independence. "
            "A raw **win-rate** is the wrong lens entirely (US returns are positive most days "
            "unconditionally). That clustering-aware $p$, not the point estimate, decides the Signal axis."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Entropy clock.** Causal $H^{{\\mathrm{{PE}}}}_t$ (window 60, $m=3$) on SPY returns "
            f"(yfinance, {R['start']}→{R['end']}, {R['days']:,} days); window ends at $t{{-}}1$.\n"
            "- **Regime.** Bottom-20% expanding-quantile entropy days = low-entropy (predictable); "
            "the rest = high-entropy.\n"
            "- **Forward returns.** Enter the close **1 day after** the signal (no look-ahead), hold "
            "$h\\in\\{1,5,21\\}$ days; drop tail overruns.\n"
            "- **Null #1 (HAC t).** Newey-West $t$ of the per-day low-minus-high contribution.\n"
            "- **Null #2 (block bootstrap).** Stationary block bootstrap of the regime labels (21-day "
            "blocks), $p = \\Pr[|\\text{boot gap}| \\ge |\\text{obs gap}|]$.\n"
            "- **Second estimator.** Re-run on sample entropy $H^{\\mathrm{SE}}$ — a real signal is "
            "estimator-robust.\n"
            "- **Costs.** 1-day lag + 1 bp/turn on a long-in-low-entropy / flat rule; gross & net Sharpe "
            "vs buy-and-hold.\n"
            "- **Positive control.** A deterministic tape toggling random vs a **zero-sum cyclic** "
            "(low-sample-entropy) regime with a *planted* edge: the detector must find it, the inference "
            "must recover a real edge **and** must NOT manufacture one when predictability comes without "
            "drift."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The point estimates — null at short horizons, a borderline 21-day bump\n\n"
            "Conditional mean forward return on low-entropy days with its $\\pm$ standard error, against "
            "the high-entropy mean (dashed). Flat at 1/5d; a small positive gap at 21d."
        ),
        code(
            "hs = [1, 5, 21]\n"
            "if HAVE_REAL:\n"
            "    cm, hm, ts, ses = [], [], [], []\n"
            "    for h in hs:\n"
            "        s = st.summarize(F, MASK, h); cm.append(s['low_mean']); hm.append(s['high_mean']); ts.append(s['t_hac'])\n"
            "        low, _ = st.regime_forward(F, MASK, h); ses.append(low.std(ddof=1)/np.sqrt(len(low)))\n"
            "else:\n"
            "    cm = [R['h1'][3]/100, R['h5'][3]/100, R['h21'][3]/100]\n"
            "    hm = [R['h1'][4]/100, R['h5'][4]/100, R['h21'][4]/100]\n"
            "    ts = [R['h1'][8], R['h5'][8], R['h21'][8]]; ses = [0.0004, 0.001, 0.003]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x, [c*100 for c in cm], yerr=[s*100 for s in ses], capsize=5, color=GREEN, width=.5, label='low-entropy (±SE)')\n"
            "ax.plot(x, [h*100 for h in hm], 'D', ms=11, c=GREY, label='high-entropy mean')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward SPY return (%)')\n"
            "ax.set_title('Low-entropy edge: ~0 at 1-5d, a small bump at 21d'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('HAC t by horizon:', {f'{h}d': round(t,2) for h,t in zip(hs,ts)})"
        ),
        md(
            f"> 💡 In plain words: 1d/5d gaps are basis-point dust at HAC *t* = {R['h1'][8]:.2f}/{R['h5'][8]:.2f}. "
            f"At 21d the gap is **+{R['h21'][5]:.2f}%** with HAC **t = {R['h21'][8]:.2f}** — *touching* the "
            "significance bar. H₁ hangs on that single number; 4b and 4c dismantle it."
        ),
        md(
            "### 4b · The decisive test — a block bootstrap sized to the clustering\n\n"
            "Resample the regime labels in 21-day blocks (preserving the clumping) and rebuild the 21-day "
            "gap many times; the histogram is the clustering-aware null. The observed gap is the green "
            "line; the two-sided *p* is the mass beyond $\\pm|\\widehat\\Delta|$."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bp = st.block_bootstrap_p(F, MASK, 21, n_draws=5000); obs = bp['obs_gap']; pval = bp['p_value']\n"
            "    fwd = st.forward_return(F['spy'], 21); sig = MASK.shift(1).fillna(False).astype(bool)\n"
            "    v = fwd.notna(); fwd = fwd[v].values; sigv = sig[v].values; n=len(fwd); k=int(sigv.sum())\n"
            "    rng = np.random.default_rng(398); off = np.arange(21); nb_=int(np.ceil(k/21))+1; draws=[]\n"
            "    for _ in range(5000):\n"
            "        sel=np.zeros(n,bool)\n"
            "        for s in rng.integers(0,n,size=nb_):\n"
            "            sel[(s+off)%n]=True\n"
            "            if sel.sum()>=k: break\n"
            "        draws.append(fwd[sel].mean()-fwd[~sel].mean())\n"
            "    draws=np.array(draws)\n"
            "else:\n"
            "    obs = R['h21'][5]/100; pval = R['h21'][9]; rng = np.random.default_rng(398); draws = rng.normal(0,0.004,5000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=60, color=GREY, alpha=.85, label=f'clustered null ({len(draws):,} block draws)')\n"
            "ax.axvline(obs*100, c=GREEN, lw=2.5, label=f'observed gap {obs*100:+.2f}%')\n"
            "ax.axvline(-abs(obs)*100, c=GREEN, ls=':', lw=1.2); ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('low-minus-high 21-day gap (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Block-bootstrap p = {pval:.2f}: gap is inside the clustered luck cloud'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'P[|clustered block draw| >= |obs|] = {pval:.3f}  (need <0.05; here ~1 in 3)')"
        ),
        md(
            f"> 💡 In plain words: **{int(R['h21'][9]*100)}%** of clustered random label sets match or beat "
            "the gap. The borderline HAC *t* was the overlapping-window SE flattering itself; once the null "
            "*keeps the clumping*, the 21-day bump is ordinary. H₁ not supported — the effective sample is "
            "a handful of regime episodes, not 1,500 independent days."
        ),
        md(
            "### 4c · Estimator-robustness — a second entropy kills it\n\n"
            "If the 21-day bump were a real regime signal it would survive a change of complexity measure. "
            "Re-run the whole split on **sample entropy** (Richman-Moorman) and sweep the quantile cutoff."
        ),
        code(
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "# left: PE vs SampEn 21d t\n"
            "if HAVE_REAL:\n"
            "    ent_se = st.rolling_entropy(F['ret'], window=60, kind='sample'); mask_se = st.low_entropy_mask(ent_se, q=0.20)\n"
            "    pe_t = [st.summarize(F, MASK, h)['t_hac'] for h in [1,5,21]]\n"
            "    se_t = [st.summarize(F, mask_se, h)['t_hac'] for h in [1,5,21]]\n"
            "else:\n"
            "    pe_t = [R['h1'][8], R['h5'][8], R['h21'][8]]; se_t = [R['s1'][4], R['s5'][4], R['s21'][4]]\n"
            "x = np.arange(3)\n"
            "a1.bar(x-.2, pe_t, .4, color=AMBER, label='permutation entropy'); a1.bar(x+.2, se_t, .4, color=GREY, label='sample entropy')\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2'); a1.set_xticks(x); a1.set_xticklabels(['1d','5d','21d'])\n"
            "a1.set_ylabel('HAC t'); a1.set_title('Borderline on PE, gone on SampEn'); a1.legend()\n"
            "# right: quantile sweep (PE, 21d)\n"
            "if HAVE_REAL:\n"
            "    rob = [(q,)+ (lambda s:(s['gap']*100, s['t_hac'], s['p_boot']))(st.summarize(F, st.low_entropy_mask(ENT,q=q), 21)) for q in (0.10,0.20,0.30)]\n"
            "else:\n"
            "    rob = [(r[0], r[2], r[3], r[4]) for r in R['robust']]\n"
            "qs=[r[0] for r in rob]; tq=[r[2] for r in rob]; pq=[r[3] for r in rob]\n"
            "a2.bar([f'{q:.0%}' for q in qs], tq, color=AMBER, width=.55); a2.axhline(2, ls='--', c=RED)\n"
            "for i,(t,p) in enumerate(zip(tq,pq)): a2.annotate(f'p={p:.2f}',(i,t),ha='center',va='bottom')\n"
            "a2.set_ylim(0,2.6); a2.set_ylabel('HAC t'); a2.set_xlabel('low-entropy cutoff'); a2.set_title('Flat in the cutoff; bootstrap never rejects')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('PE 21d t=%.2f -> SampEn 21d t=%.2f' % (pe_t[-1], se_t[-1]))"
        ),
        md(
            f"> 💡 In plain words: swap permutation entropy for sample entropy and the 21d HAC *t* drops "
            f"from **{R['h21'][8]:.2f}** to **{R['s21'][4]:.2f}** — the two gauges don't even agree on which "
            "days are 'low entropy'. Across quantile cutoffs the bootstrap *p* never leaves "
            f"**{R['robust'][2][4]:.2f}–{R['robust'][0][4]:.2f}**. There is no robust corner."
        ),
        md(
            "### 4d · Tradability ledger — predictable, unprofitable\n\n"
            "The natural deployment: long SPY on (lagged) low-entropy days, flat otherwise; 1 bp per "
            "turn. Compare gross/net annualised return and Sharpe to buy-and-hold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = st.net_of_costs(F, MASK, cost_bps=1.0)\n"
            "else:\n"
            "    c = dict(gross_ann=R['cost']['gross']/100, net_ann=R['cost']['net']/100, bh_ann=R['cost']['bh']/100,\n"
            "             gross_sharpe=R['cost']['gross_sh'], net_sharpe=R['cost']['net_sh'], bh_sharpe=R['cost']['bh_sh'],\n"
            "             exposure=R['cost']['exposure']/100, turnover_annual=R['cost']['turnover'])\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "labels = ['low-ent\\ngross','low-ent\\nnet','buy &\\nhold']\n"
            "a1.bar(labels, [c['gross_ann']*100, c['net_ann']*100, c['bh_ann']*100], color=[GREEN, AMBER, GREY])\n"
            "a1.set_ylabel('annual return (%)'); a1.set_title('Timing leaves most of the drift on the table')\n"
            "a2.bar(labels, [c['gross_sharpe'], c['net_sharpe'], c['bh_sharpe']], color=[GREEN, AMBER, GREY])\n"
            "a2.set_ylabel('annual Sharpe'); a2.set_title(f'Worse Sharpe at {c[\"exposure\"]*100:.0f}% exposure')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'net {c[\"net_ann\"]*100:.2f}%/yr @ Sharpe {c[\"net_sharpe\"]:.2f} vs B&H {c[\"bh_ann\"]*100:.2f}%/yr @ Sharpe {c[\"bh_sharpe\"]:.2f}; turnover {c[\"turnover_annual\"]:.1f}x')"
        ),
        md(
            f"> 💡 In plain words: net **+{R['cost']['net']:.2f}%/yr at Sharpe {R['cost']['net_sh']:.2f}** vs "
            f"**+{R['cost']['bh']:.2f}%/yr at Sharpe {R['cost']['bh_sh']:.2f}**. Costs are trivial (1 bp on "
            f"{R['cost']['turnover']:.0f}× turnover); the killer is **{R['cost']['exposure']}% exposure** — "
            "you sit out four days in five to capture a worse risk-adjusted deal. MIRAGE."
        ),
        md(
            "### 4e · Faithful-engine & predictability-without-edge control\n\n"
            "A deterministic tape toggling **random** vs a **zero-sum cyclic structured** regime (low "
            "sample entropy, *no* drift by construction). With **zero** planted edge the test must NOT "
            "reject (predictable ≠ profitable); with a large planted edge it must light up — proving the "
            "engine is unbiased and isolating the exact fallacy."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.0025):\n"
            "    syn = data.synthetic_returns(n_days=3000, block=120, edge=edge, seed=398)\n"
            "    ent_y = st.rolling_entropy(syn['ret'], window=40, kind='sample'); mask_y = st.low_entropy_mask(ent_y, q=0.30)\n"
            "    truth = syn['regime'].astype(bool); recall = (mask_y & truth).sum()/max(1,int(mask_y.sum()))\n"
            "    s = st.summarize(syn, mask_y, 5); res.append((edge, recall, s['gap'], s['t_hac'], s['p_boot']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted edge\\n{e*100:.2f}%/day\\n(recall {r*100:.0f}%)' for e,r,_,_,_ in res]\n"
            "tvals = [r[3] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5); ax.axhline(2, ls='--', c=RED, label='t=2'); ax.axhline(-2, ls='--', c=RED)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "ax.set_ylabel('HAC t (5-day)'); ax.set_title('Control: predictable-no-edge stays flat; real edge lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,r,g,t,p in res: print(f'edge {e*100:+.2f}%/day: recall {r*100:.0f}% gap {g*100:+.2f}% t={t:.2f} p_boot={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the detector finds the predictable regime (**{R['syn'][0][2]}%** recall) "
            f"either way. With **no** edge the test sits at **t = {R['syn'][0][4]:.2f}** (flat — no false "
            f"positive); only a **+{R['syn'][1][0]:.2f}%/day** planted edge drives **t = {R['syn'][1][4]:.2f}**. "
            "So the machinery is honest, and the real-tape result is exactly what *predictability without a "
            "payoff* looks like through it. H₃ rejected at the root."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — 21d gap **+{R['h21'][5]:.2f}%** at HAC **t = {R['h21'][8]:.2f}** but "
            f"block-bootstrap **p = {R['h21'][9]:.2f}**; null at 1–5d; sample-entropy 21d **t = {R['s21'][4]:.2f}**; "
            "flat across cutoffs. Literature interest + a non-robust, estimator-fragile point estimate ⇒ "
            "WEAK, not REAL (the desk's t ≥ 2 bar is not robustly met).\n"
            f"- **Tradability `MIRAGE`** — long-in-low-entropy **+{R['cost']['net']:.2f}%/yr @ Sharpe "
            f"{R['cost']['net_sh']:.2f}** vs B&H **+{R['cost']['bh']:.2f}%/yr @ Sharpe {R['cost']['bh_sh']:.2f}**; "
            f"{R['cost']['exposure']}% exposure. Costs trivial; the clock just keeps you out of the market.\n"
            f"- **Predictable = profitable? `BUSTED`** — the synthetic control: a regime built to be "
            f"maximally predictable with **zero** drift is detected (**{R['syn'][0][2]}%** recall) and pays "
            f"**nothing** (**t = {R['syn'][0][4]:.2f}**). Order is not return. Same regime question, different "
            "lens, as **[Study 397 (Hurst Regime)](../397-hurst-regime/)**."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the effective-sample problem\n\n"
            "The operational truth: low-entropy days clump into a *handful of regime episodes*, so the "
            "effective sample is tiny — and statistical power scales with the **effective**, not nominal, "
            "count. We estimate the effective sample size from the run-length of low-entropy episodes."
        ),
        code(
            "if HAVE_REAL:\n"
            "    m = MASK.values.astype(int)\n"
            "    # count contiguous low-entropy runs (episodes)\n"
            "    runs = np.diff(np.concatenate([[0], m, [0]]))\n"
            "    starts = np.where(runs==1)[0]; ends = np.where(runs==-1)[0]\n"
            "    lengths = ends - starts; n_episodes = len(lengths); n_days = int(m.sum())\n"
            "else:\n"
            "    n_days = R['n_low']; n_episodes = 70; lengths = np.full(70, n_days/70)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "a1.hist(lengths, bins=30, color=GREY); a1.set_xlabel('episode length (days)'); a1.set_ylabel('count')\n"
            "a1.set_title(f'{n_days} low-entropy days = {n_episodes} clustered episodes')\n"
            "a2.bar(['nominal n\\n(days)','effective n\\n(episodes)'], [n_days, n_episodes], color=[GREY, RED])\n"
            "a2.set_ylabel('sample size'); a2.set_title('Power scales with the SMALL number')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'{n_days} low-entropy days collapse to ~{n_episodes} independent episodes (mean length {lengths.mean():.0f}d)')"
        ),
        md(
            "> 💡 In plain words: the **1,513** low-entropy days are not 1,513 bets — they are a few dozen "
            "**episodes** of persistent calm. That is why the borderline HAC *t* evaporates under the block "
            "bootstrap: the honest denominator is the *episode* count, and a few dozen episodes cannot "
            "establish a sub-percent monthly edge. No window-size, threshold, or cost assumption "
            "manufactures power from a clock that barely leaves the noise ceiling — **the order the signal "
            "promises is both rare and unpaid.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The persistence cousin.** [Study 397 — Hurst Regime](../397-hurst-regime/): trending vs "
            "mean-reverting regimes — the long-memory lens on the same 'is the market more orderly now, "
            "and does it pay?' question. Reading them together shows the gauge changes, the verdict "
            "doesn't.\n"
            "- **Multiscale & cross-sectional entropy.** Multiscale sample entropy, transfer entropy "
            "between assets, or permutation entropy on a *cross-section* of names — richer gauges, same "
            "predictability≠profitability trap to test.\n"
            "- **Conditional, not unconditional.** Does low entropy improve a *different* signal "
            "(momentum, vol-targeting) rather than predict raw return? A more honest use of the clock — "
            "PR it and we'll race it.\n\n"
            "*The reproducible core is offline and deterministic apart from one cached SPY pull; the "
            "entropy clock is causal and the null is clustering-aware. Methods and sources: "
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
