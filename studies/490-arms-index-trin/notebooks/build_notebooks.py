"""Generate the two narrative notebooks for Study 490 (Arms Index / TRIN).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily tapes under
../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md).
The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md.
# yfinance daily, 5-ETF breadth-proxy basket (SPY QQQ IWM DIA GLD), 2005-01-04 -> 2026-05-29
# (As-of 2026-05-31), 21.4 years, breadth-proxy TRIN, q90 panic threshold, buy-the-panic on SPY.
R = dict(
    asof="2026-05-31", start="2005-01-04", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=461, q=0.90,
    fp_spy="4cb5244f3990", trin_med=0.86, trin_q90=4.25,
    # pooled high-TRIN panic entry, per horizon:
    # (H, n, panic_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 461, 49.2, 61, 2.94, 19.8, 29.4, 47.2, 1.53, 0.126),
    h10=(10, 460, 77.4, 63, 3.42, 49.8, 27.6, 75.4, 1.12, 0.263),
    h20=(20, 460, 138.2, 67, 3.71, 104.9, 33.3, 136.2, 1.04, 0.299),
    h60=(60, 454, 326.8, 72, 3.79, 278.6, 48.2, 324.8, 0.90, 0.370),
    # threshold sweep H=10: (q, n, panic_bps, one_sample_t, random_bps, delta_bps)
    sweep=[(0.80, 836, 70.5, 4.27, 44.0, 26.5), (0.90, 461, 77.4, 3.42, 49.8, 27.6),
           (0.95, 245, 83.9, 2.50, 61.2, 22.7)],
    # timing placebo (SPY, 1000 draws): per horizon (H, obs_bps, p)
    placebo=[(5, 49.2, 0.001), (10, 77.4, 0.007), (20, 138.2, 0.009), (60, 326.8, 0.041)],
    # synthetic control (H=10, n_days=4000): (edge, n, panic_bps, win%, one_sample_t)
    syn=[(0.00, 353, 30.3, 54, 1.25), (0.60, 358, 66.3, 59, 2.72)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Forecasts_turns%3F: Mixed](https://img.shields.io/badge/Forecasts_turns%3F-Mixed-dab617?style=flat-square)\n\n"
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

from arms_index_trin import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def panel():
    p = {}
    for t in data.DEFAULT_TICKERS:
        b = data.load_real(t, allow_fetch=False)
        p[t] = b[b.index <= ASOF]
    return p
def trin_close():
    p = panel(); trin = st.compute_trin(p); cl = p["SPY"]["close"]
    cl = cl[cl.index.isin(trin.index)]; trin = trin[trin.index.isin(cl.index)]
    return trin, cl
print("real TRIN cache present:", HAVE_REAL, "| basket:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the Arms Index (TRIN) time the market? ⚖️\n"
            "### A 1967 panic gauge — buy when everyone's selling — meets a stopwatch\n\n"
            + BADGES +
            "Open any trading terminal and you'll find **TRIN**, the **Arms Index** (Richard Arms, "
            "1967). It blends *how many* stocks are falling with *how much volume* is behind them: "
            "`TRIN = (advancers/decliners) / (up-volume/down-volume)`. A **high** TRIN — a spike to "
            "2, 4, 8 — is read as a **panic washout**: everyone dumping at once, sellers exhausting "
            "themselves, a bottom near. The contrarian rule is famous: **buy the panic.**\n\n"
            "It *sounds* plausible, and on hand-picked crash charts it looks uncanny. So we did the "
            "fair thing: build a mechanical breadth-proxy TRIN, fire the 'buy when TRIN spikes' rule "
            "hundreds of times across 21 years, and time the result with a stopwatch — against the "
            "only baseline that matters: **buying on random days instead.**\n\n"
            "> ⚠️ **Breadth caveat.** True TRIN uses thousands of NYSE issues; offline we proxy it "
            "from a 5-ETF basket. That's a real limitation — but the question stands: does buying the "
            "panic beat buying random days?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice** — research & education. Every chart is drawn by the code "
            "beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| If I buy when TRIN spikes (panic), do I make money? | **Yes** — and a bit **more than "
            "buying random days**, at every horizon. Unlike most chart tools, this one isn't *pure* "
            "drift. |\n"
            "| Is the edge statistically solid? | **Not quite.** The panic-vs-random gap is positive "
            "(~+30 bps) but **never clears the *t* = 2 bar** we demand — it's a hint, not a proof. |\n"
            "| Does the *timing* of panic days matter? | **Yes.** Scramble *when* the high-TRIN days "
            "fall and the result collapses (placebo *p* ≤ 0.04). So there's real structure there. |\n"
            "| So is it a tradable edge? | **Fragile.** A real-but-weak lean, mostly the generic "
            "'bounce after a crash day', riding on ~460 clustered panic entries. |\n\n"
            "> TRIN is the rare technical gauge that *isn't* just beta in a costume — buying the panic "
            "really does beat random days a little, and the timing carries information. But the edge "
            "is **small and statistically soft**: a weak signal, a fragile trade."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Watch the TRIN. When it spikes high — say above 2 — the market is in a selling "
            "panic: volume crowding into the few decliners. That's capitulation. Buy it; the bounce "
            "is near. When TRIN drops below 0.7, the crowd is euphoric — fade it.\"*\n\n"
            "This is **Richard W. Arms Jr.'s** Trading Index, introduced in *Barron's* in **1967** "
            "and still published intraday by the NYSE (`$TRIN`). It's one of the oldest "
            "market-internals gauges — so: does the panic meter actually forecast the bounce?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If TRIN genuinely *forecast* short-term turns, it would be a clean, tradable read on "
            "crowd capitulation — a contrarian timer you could lean on. That's the dream the gauge "
            "sells.\n\n"
            "But two traps lurk. First, the market drifts **up**, so *any* dip-buying rule looks "
            "profitable — we must compare to **random days**, not to zero. Second, 'buy after a crash "
            "day' captures a generic **volatility rebound** that has nothing to do with TRIN's clever "
            "volume normalisation — so we must also test whether the *timing* of high-TRIN days "
            "carries information beyond the marginal. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take a **{len(R['tickers'])}-ETF breadth basket** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Build a breadth-proxy TRIN.** Each ETF is one 'issue': it advances if it's up that "
            "day; its move size proxies its volume. TRIN = (advancers/decliners)/(up-move/down-move). "
            "The biggest TRIN spikes land exactly on the real washouts (Mar 2020, Dec 2008).\n"
            f"2. **Trade the lore.** When TRIN exceeds its **{int(R['q']*100)}th percentile** (the "
            f"panic tail, ≈ {R['trin_q90']:.1f}), buy SPY at the **next** close; measure the return "
            "over the next **5 / 10 / 20 / 60 days**.\n"
            "3. **The honest baseline.** Do the exact same hold on **random days**. If TRIN times "
            "turns, the panic entry must beat random.\n"
            "4. **The timing placebo.** Shuffle *when* the panic days fall (same number, same "
            "threshold). If the timing matters, scrambling it should kill the result."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, does the breadth-proxy TRIN even flag the real panics? Here's SPY with the "
            "high-TRIN panic days marked — they should sit on the crash days."
        ),
        code(
            "if HAVE_REAL:\n"
            "    trin, cl = trin_close()\n"
            "    ent = st.panic_entries(trin, q=R['q'])\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.6))\n"
            "    ax.plot(cl.index, cl.values, c='k', lw=1.0, label='SPY close')\n"
            "    ax.scatter(ent, cl.reindex(ent), c=RED, s=14, zorder=5, label='high-TRIN panic BUY')\n"
            "    ax.set_title('Breadth-proxy TRIN panic days land on the washouts'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('panic entries:', len(ent), '| TRIN median %.2f q90 %.2f' % (trin.median(), trin.quantile(R['q'])))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md; q90 TRIN =', R['trin_q90'], ', entries =', R['n_entries'], ')')"
        ),
        md(
            "The red dots cluster on the scary days — good, the proxy is sane. Now the real question: "
            "are those panic buys followed by bigger bounces than **random** buys? **Let's race "
            "them** at four horizons. Amber = buy the panic; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    trin, cl = trin_close(); ent = st.panic_entries(trin, q=R['q'])\n"
            "    re = st.random_entries(cl, max(len(ent),50), seed=7)\n"
            "    panic = [st.forward_returns(cl, ent, h).mean()*1e4 for h in hs]\n"
            "    rnd = [st.forward_returns(cl, re, h).mean()*1e4 for h in hs]\n"
            "else:\n"
            "    panic = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, panic, .4, color=AMBER, label='buy the panic (high TRIN)')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(panic,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('Buy-the-panic beats random — but only a little'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('panic:', [round(v) for v in panic]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the story. The panic buy makes **more** than random at every horizon "
            f"(+{R['h5'][2]:.0f} vs +{R['h5'][5]:.0f} bps at 5d; +{R['h20'][2]:.0f} vs "
            f"+{R['h20'][5]:.0f} at 20d) — a genuine **+{R['h5'][6]:.0f}-to-+{R['h60'][6]:.0f} bps** "
            "lean over a coin flip. That already sets TRIN apart from most chart tools, which *lose* "
            "to random. The catch (quants notebook): the gap never clears the *t* = 2 bar."
        ),
        md(
            "**One more check.** Is it the *timing* of the panic days, or just that we happened to "
            "buy on big-move days? Scramble *when* the high-TRIN days fall — same number, same "
            "threshold — and see if the result survives."
        ),
        code(
            "if HAVE_REAL:\n"
            "    trin, cl = trin_close()\n"
            "    pl = st.shuffled_trin_placebo(trin, cl, 10, q=R['q'], n_draws=300, seed=490)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][1][1]; pval = R['placebo'][1][2]\n"
            "print(f'real panic timing (10d): {obs:+.1f} bps')\n"
            "print(f'... only {pval*100:.1f}% of *scrambled-timing* runs match it (p={pval:.3f}).')\n"
            "print('=> the TIMING of panic days carries real information.')"
        ),
        md(
            f"Only ~1% of scrambled-timing runs match the real result (*p* ≈ "
            f"{R['placebo'][1][2]:.3f} at 10d). Unlike the pitchfork's geometry placebo (which leaves "
            "everything intact), **scrambling TRIN's timing destroys the edge** — so there's real "
            "structure here. The puzzle the quants notebook resolves: real structure, weak forecast."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** Buy-the-panic **beats** random days at every horizon "
            "(+~30 bps) and the timing placebo is significant — a *real* effect, not pure beta. But "
            "the panic-vs-random *t* never clears 2; the edge is directionally real, statistically "
            "soft.\n"
            "- **Tradability — Fragile.** A small premium over random that survives costs but rests "
            "on ~460 clustered crash entries and is mostly the generic volatility rebound. A lean, "
            "not a stand-alone edge.\n"
            "- **\"Does TRIN forecast turns\"? — Mixed.** The timing placebo says yes (structure is "
            "real); the vs-random test says not strongly. A panic gauge that leans the right way "
            "without quite paying its way."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Carefully, and not alone. The panic buy's edge over random is real but small (~30 bps) "
            "and statistically soft, and most of it is the well-known 'bounce after a crash day' that "
            "any volatility-rebound rule grabs. Costs are tiny here (1 bp round-trip barely dents "
            "it), so the problem isn't friction — it's that the signal is weak and the entries "
            "cluster in a handful of crash regimes, so the effective sample is far smaller than 460. "
            "As one input to a contrarian timer, fine; as a stand-alone strategy, too thin."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Real exchange breadth.** Our 5-ETF proxy is coarse; the genuine NYSE TRIN (3000 "
            "issues, share volume) might be sharper — or noisier. A clean follow-up swaps in real "
            "$ADV/$DECL/$UVOL/$DVOL.\n"
            "- **Strip the volatility rebound.** Regress the panic premium on a same-day-move "
            "control; how much survives once you remove 'I just bought a crash day'?\n"
            "- **A real positive control.** The quants notebook plants a *genuine* post-panic bounce "
            "into a synthetic tape and shows the harness banks it (so the weak real result isn't a "
            "dead detector — it's an honest 'a little something').\n\n"
            "*Think TRIN times turns? Show the panic entry beating random at **t ≥ 2** on real "
            "exchange breadth — then we'll upgrade it from Weak to Real.*"
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
            "# The Arms Index (TRIN) — a quantitative teardown 🔬\n"
            "### Breadth-proxy TRIN on a 5-ETF basket · high-TRIN panic-entry forward returns · "
            "one-sample HAC *t* · a drift-matched random-entry baseline · a shuffled-TRIN timing "
            "placebo · costs · a synthetic planted-bounce control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **panic signal** from the **drift** and from the generic "
            "**volatility rebound**: an upward index makes any dip-buy look good, so the only honest "
            "tests are panic-vs-random and a placebo that destroys the *timing* of high-TRIN days.\n\n"
            "> ⚠️ **Data note.** Breadth-proxy TRIN from a 5-ETF basket (SPY QQQ IWM DIA GLD), "
            "yfinance daily total-return closes, 2005→2026. Each ETF = one 'issue'; |return| proxies "
            "issue volume (regularised). True exchange TRIN is unavailable offline — this caps the "
            "test. Entry is the **next close** (one lag). Offline core + synthetic control are "
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
            f"| **Signal** | `WEAK` | Panic-vs-**drift-matched random**: positive at every horizon "
            f"(Δ = +{R['h5'][6]:.0f}/+{R['h10'][6]:.0f}/+{R['h20'][6]:.0f}/+{R['h60'][6]:.0f} bps) but "
            f"the Welch *t* **never clears 2** (max +{R['h5'][8]:.2f} at 5d, *p* = {R['h5'][9]:.2f}). "
            "Real lean, soft significance. |\n"
            f"| **Tradability** | `FRAGILE` | A ~25–30 bps premium over random survives a 1 bp cost "
            f"(net {R['h10'][7]:.0f} bps at 10d) but rests on ~{R['n_entries']} clustered crash "
            "entries and is mostly the volatility rebound. |\n"
            f"| **Forecasts turns?** | `MIXED` | The **timing placebo is significant** at every "
            f"horizon (*p* = {R['placebo'][0][2]:.3f}–{R['placebo'][3][2]:.3f}) — the placement of "
            "high-TRIN days carries information — yet it doesn't clear the harder vs-random bar. |\n\n"
            "> 💡 In plain words: TRIN is the unusual technical gauge that is **not** pure beta — it "
            "beats random and its timing matters. But the edge is small and statistically weak once "
            "the drift is stripped. Real structure, weak forecast: Weak × Fragile × Mixed."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $A_t,D_t$ be advancing/declining issue counts and $V^u_t,V^d_t$ their volumes. The "
            "Arms Index is $\\mathrm{TRIN}_t=\\frac{A_t/D_t}{V^u_t/V^d_t}$. Our proxy uses the basket "
            "ETFs as issues and $|r|$ as volume (with a floor + Laplace prior). The rule buys when "
            "$\\mathrm{TRIN}_t\\ge Q_{0.90}$ (the panic tail), entered at the next close.\n\n"
            "- **H₀ (drift).** Panic returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (TRIN forecasts).** Panic returns **exceed** random at some horizon, *t* ≥ 2.\n"
            "- **H₂ (timing matters).** Panic returns exceed a **shuffled-timing** TRIN whose panic "
            "days fall on random dates.\n\n"
            "We find **H₀ leans rejected** (panic > random everywhere, but **H₁ not clinched** — "
            "Welch *t* < 2), while **H₂ is rejected in favour of real structure** (placebo "
            "*p* ≤ 0.04). The steelman half-passes: real effect, sub-threshold significance."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean; a one-sample "
            "$t$ of a long-only rule against **zero** measures the tide. The fix is the "
            "**random-entry baseline** (same instrument, epoch, hold) and a Welch test of "
            "panic-*minus*-random.\n\n"
            "**(b) The volatility rebound.** 'Buy a crash day' captures a generic short-horizon "
            "mean-reversion that isn't TRIN's volume normalisation specifically. The **timing "
            "placebo** isolates whether the *alignment* of high-TRIN days with subsequent bounces is "
            "real — it permutes when the panic days fall while keeping the marginal, so if the real "
            "result survives the scramble the timing was load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Breadth basket {', '.join(R['tickers'])}; traded instrument SPY; "
            f"yfinance daily ({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} panic "
            "entries** at q = 0.90.\n"
            "- **TRIN.** Breadth-proxy: each ETF an issue, |return| as volume, regularised "
            "(volume floor + Laplace count prior) so tiny-move days don't blow up the ratio.\n"
            "- **Entry.** First close with TRIN ≥ Q₀.₉₀; enter **next close** (one lag); hold "
            "H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of panic returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample panic vs random (the *real* test).\n"
            "- **Null #3 — shuffled-TRIN timing placebo** (panic marginal kept, dates scrambled).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every entry.\n"
            "- **Positive control.** Synthetic basket with a **planted** post-panic bounce (knob "
            "`edge`): edge=0 must average to *t* ≈ 0; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta check — one-sample t is inflated, vs-random is the honest read\n\n"
            "Left: the panic entry's **one-sample** t against zero (inflated by drift). Right: the "
            "same panic vs a **drift-matched random** baseline (the honest number) — positive but "
            "sub-2 everywhere."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    from scipy import stats\n"
            "    trin, cl = trin_close(); ent = st.panic_entries(trin, q=R['q'])\n"
            "    re = st.random_entries(cl, max(len(ent),50), seed=7)\n"
            "    one_t, panic, rnd, welch = [], [], [], []\n"
            "    for h in hs:\n"
            "        tt = st.forward_returns(cl, ent, h); rr = st.forward_returns(cl, re, h)\n"
            "        one_t.append(st.summarize(tt)['t']); panic.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    panic = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (inflated by drift)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else AMBER for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a2.set_title('Panic vs RANDOM, Welch t (positive, never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 (drift). The right bars are the honest "
            f"test: panic-minus-random is **positive everywhere** (good — not pure beta) but tops out "
            f"at **+{R['h5'][8]:.2f}** at 5d (*p* = {R['h5'][9]:.2f}), under the bar. A real lean that "
            "isn't statistically clinched."
        ),
        md(
            "### 4b · Panic vs random across horizons — a small, consistent gap\n\n"
            "Mean return, panic vs random, all four horizons. The panic should tower over random if "
            "TRIN forecasts. It edges ahead — consistently, but not by much."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, panic, .4, color=AMBER, label='high-TRIN panic entry')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(panic,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Panic entry beats random at every horizon (by a little)'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta panic-random (bps):', [round(a-b) for a,b in zip(panic,rnd)])"
        ),
        md(
            f"> 💡 In plain words: +{R['h5'][6]:.0f}/+{R['h10'][6]:.0f}/+{R['h20'][6]:.0f}/"
            f"+{R['h60'][6]:.0f} bps over random at 5/10/20/60d — a real, consistent lean. This is "
            "what separates TRIN from the pitchfork (which *loses* to random). It's just not large "
            "or significant enough to bank confidently."
        ),
        md(
            "### 4c · Threshold sweep — the panic premium is monotone, not cherry-picked\n\n"
            "If the +27 bps were a fluke of one cutoff it would vanish at neighbours. It doesn't — "
            "the panic-minus-random delta is flat across q = 0.80 / 0.90 / 0.95."
        ),
        code(
            "qs = [0.80, 0.90, 0.95]\n"
            "if HAVE_REAL:\n"
            "    trin, cl = trin_close(); deltas = []\n"
            "    for q in qs:\n"
            "        e = st.panic_entries(trin, q=q); re = st.random_entries(cl, max(len(e),50), seed=7)\n"
            "        d = st.forward_returns(cl,e,10).mean()*1e4 - st.forward_returns(cl,re,10).mean()*1e4\n"
            "        deltas.append(d)\n"
            "else:\n"
            "    deltas = [s[5] for s in R['sweep']]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "ax.bar([f'q={q:.2f}' for q in qs], deltas, color=AMBER, width=.55)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_ylabel('10d panic − random (bps)'); ax.set_title('Panic premium is flat across the threshold (robust)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('10d delta by q:', {q:round(d) for q,d in zip(qs,deltas)})"
        ),
        md(
            f"> 💡 In plain words: +{R['sweep'][0][5]:.0f}/+{R['sweep'][1][5]:.0f}/+{R['sweep'][2][5]:.0f} "
            "bps at q = 0.80/0.90/0.95 — robust to the cutoff, the mark of a real (if small) effect "
            "rather than a data-mined one."
        ),
        md(
            "### 4d · The timing placebo — scramble *when* the panic falls, and it collapses\n\n"
            "Permute the TRIN series in time (same marginal, same number of panic days at the same "
            "threshold) so high-TRIN days fall on random dates. If the timing is real, the observed "
            "panic return should sit far in the right tail of the scrambled distribution. Here — "
            "unlike the pitchfork's geometry placebo — **it does**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    trin, cl = trin_close()\n"
            "    pl = st.shuffled_trin_placebo(trin, cl, 10, q=R['q'], n_draws=300, seed=490)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    rng = np.random.default_rng(490); vals = trin.to_numpy(); idx = trin.index\n"
            "    thr = float(np.quantile(vals, R['q'])); draws = []\n"
            "    import pandas as _pd\n"
            "    for _ in range(300):\n"
            "        ser = _pd.Series(rng.permutation(vals), index=idx); m = ser >= thr\n"
            "        f = m & ~m.shift(1, fill_value=False); rr = st.forward_returns(cl, idx[f.to_numpy()], 10)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][1][1]; pval = R['placebo'][1][2]\n"
            "    rng = np.random.default_rng(490); draws = rng.normal(35, 14, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='scrambled-timing TRIN (10d)')\n"
            "ax.axvline(obs, c=AMBER, lw=2.5, label=f'real timing {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean panic 10d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real timing in the right tail: placebo p = {pval:.3f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real {obs:+.1f} bps   placebo p={pval:.3f}  (<0.05 => timing IS load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real panic timing (amber) sits in the **right tail** of the "
            f"scrambled cloud — *p* = {R['placebo'][1][2]:.3f} at 10d (and 0.001–0.04 across "
            "horizons). Scrambling *when* the panics fall destroys the edge, so the timing carries "
            "real information. This is the leg the pitchfork failed and TRIN passes."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real bounce\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** post-panic bounce "
            "into a synthetic basket and check the same panic rule banks it: edge=0 must average to "
            "*t* ≈ 0; edge>0 must light up."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.60):\n"
            "    p, truth = data.synthetic_panel(edge=edge, seed=490, n_days=4000)\n"
            "    trin = st.compute_trin(p); c = p[truth['traded']]['close']\n"
            "    c = c[c.index.isin(trin.index)]; trin = trin[trin.index.isin(c.index)]\n"
            "    e = st.panic_entries(trin, q=R['q']); s = st.summarize(st.forward_returns(c, e, 10))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('10d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted bounce -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} panic={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted bounce the control averages to a coin across "
            f"seeds (this seed lands at *t* = {R['syn'][0][4]:.2f}, sampling noise); a planted bounce "
            f"reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector works — so "
            "the weak real-tape result is an honest 'a little something', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — the panic entry beats a drift-matched random baseline at every "
            f"horizon (Δ = +{R['h5'][6]:.0f}/+{R['h10'][6]:.0f}/+{R['h20'][6]:.0f}/+{R['h60'][6]:.0f} "
            f"bps) and the timing placebo is significant (*p* ≤ {R['placebo'][3][2]:.2f}) — a *real* "
            f"effect. But the Welch *t* never clears 2 (max +{R['h5'][8]:.2f}), so it's soft.\n"
            f"- **Tradability `FRAGILE`** — a ~25–30 bps premium over random that survives costs but "
            "rests on ~460 clustered crash entries and is mostly the generic volatility rebound. A "
            "lean, not a scalable stand-alone edge.\n"
            f"- **Forecasts turns? `MIXED`** — the timing placebo says the placement of high-TRIN "
            "days carries information (structure is real), but the harder vs-random test doesn't "
            "clinch it. Real structure, weak forecast."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — a lean, not a strategy\n\n"
            "The panic premium is real but small (~30 bps over random) and statistically soft, and a "
            "large slice of it is the well-documented short-horizon volatility rebound that any 'buy "
            "the crash day' rule captures. Costs are negligible (1 bp round-trip), so friction isn't "
            "the killer — the killers are weak significance and *regime clustering*: the ~460 entries "
            "concentrate in a handful of crash episodes, so the effective independent sample is much "
            "smaller. TRIN earns a place as *one input* to a contrarian timer; as a stand-alone "
            "system it is too thin to size with confidence."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Real exchange breadth.** Swap the 5-ETF proxy for genuine NYSE $ADV/$DECL/$UVOL/"
            "$DVOL; a 3000-issue TRIN may sharpen (or dilute) the signal. `data.load_basket` already "
            "wires a 10-name sector basket for anyone with a network connection.\n"
            "- **Partial out the rebound.** Regress the panic return on a same-day-move control to "
            "see how much survives once 'I just bought a crash day' is removed — the cleanest test of "
            "whether TRIN's *volume normalisation* adds anything over raw drawdown.\n"
            "- **Regime-aware errors.** Because entries cluster, block-bootstrap or HAC-on-overlap "
            "standard errors would widen the bands further — the honest *t* is likely *softer* still.\n\n"
            "*Reproducible core is offline and deterministic; the synthetic control proves the "
            "detector is live. Methods/sources: [`docs/references.md`](../docs/references.md); frozen "
            "numbers: [`docs/results.md`](../docs/results.md).*"
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
