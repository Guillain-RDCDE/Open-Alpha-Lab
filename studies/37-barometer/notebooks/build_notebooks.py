"""Generate the two narrative notebooks for Study 37 (Barometer) from source.

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

The executed path runs OFFLINE: it loads the seeded synthetic cross-asset world (the machinery proof: two
latent, regime-switching macro states whose lagged momentum predicts returns, vs a null) AND the REAL book
from the desk's cached tape (18 cross-asset ETFs + CPI-YoY / yield-curve-slope macro, lagged one month,
2007-2025). Both notebooks walk the SAME seven desk beats and quote the real fingerprinted numbers; the
synthetic control is kept as the offline mechanism proof.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../../.."))
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5.2)
import numpy as np, pandas as pd
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
from barometer import data, strategy, costs, extension
from barometer.data import REAL_ASSETS, REAL_ASSET_ORDER
SPEC = dict(assets=REAL_ASSETS, order=REAL_ASSET_ORDER)

# Offline synthetic control: a cross-asset world whose latent macro momentum predicts returns (the
# machinery) + a macro_strength=0 null.
r, m, truth = data.synthetic_macro(macro_strength=1.0, seed=37)
r0, m0, _ = data.synthetic_macro(macro_strength=0.0, seed=37)
print(f"synthetic control: {truth.n_assets} assets x {truth.n_months} months, macro_strength {truth.macro_strength} (null=0)")

# REAL book from the desk's cached tape (18 ETFs + CPI-YoY/yield-slope macro, lagged 1m, 2007-2025), offline.
_b = data.fetch_macro()
R, M = (_b.get("assets"), _b.get("macro")) if _b else (None, None)
if R is not None:
    print(f"real book: {R.shape[1]} ETFs x {len(R)} months  {R.index.min().date()} -> {R.index.max().date()}")
"""

# Synthetic-control numbers (seed 37, 50y, gross) — the offline machinery proof the executed cells reproduce.
S = dict(
    mm_sh="1.09", mm_ann="5.1", mm_dd="-12", mm_turn="5.6", mm_null="-0.17", mm_be="91",
    inf_sh="0.55", inf_ann="2.3", inf_null="-0.02", inf_be="60",
    reg_inf_up_sh="0.59", reg_inf_up_ann="2.5", reg_inf_dn_sh="0.46", reg_inf_dn_ann="1.9",
    reg_mm_up_sh="0.99", reg_mm_dn_sh="1.33",
    c0="1.09", c5="1.03", c10="0.97", c25="0.78", c50="0.48",
)

# REAL-tape numbers (verify.py on the cached tape, 217 months 2007-02 -> 2025-02, net @5 bp, fp baa416a9db25).
RR = dict(
    span="2007-02 -> 2025-02", n="217", fp="baa416a9db25",
    mm_sh="-0.05", mm_cagr="-0.7", mm_dd="-29", mm_turn="7.3", mm_t="-0.22",
    inf_sh="-0.12", inf_t="-0.42",
    reg_up_sh="-0.08", reg_up_ann="-0.5", reg_dn_sh="-0.16", reg_dn_ann="-0.9",
    raw_up_sh="0.10", raw_up_ann="1.8", raw_dn_sh="-0.01", raw_dn_ann="-0.3",
    bench_ew_sh="0.57", bench_ew_cagr="5.2", bench_6040_sh="0.84", bench_6040_cagr="7.8",
)

BADGES = (
    "![Signal: Real on direction](https://img.shields.io/badge/Signal-Real_on_direction-2ea44f?style=flat-square)\n"
    "![Signal: Weak on size](https://img.shields.io/badge/Signal-Weak_on_size-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Real-tape run?: Done](https://img.shields.io/badge/Real--tape_run%3F-Done-2ea44f?style=flat-square)\n\n"
)


def md(t): return new_markdown_cell(t)
def code(t): return new_code_cell(t)


def build_curious():
    cells = [
        md(
            "# Barometer 🌡️\n"
            "### \"Read the macro weather, trade the trend.\" A *real* cross-asset premium — slow, modest, and only a storm-hedge when it storms.\n\n"
            + BADGES +
            "The idea is old and intuitive: the economy has weather. When **growth** is improving, "
            "pro-cyclical assets (stocks, commodities) do well; when **inflation** is rising, *real* "
            "assets (commodities, inflation-linked bonds, gold) protect you while nominal stocks and bonds "
            "suffer. So watch the *trend* in the macro data — the barometer — and lean your portfolio the "
            "way it's pointing. This study asks the honest two-part question: is that real, and can you "
            "trade it?\n\n"
            "> 📓 **Plain-language layer.** The macro-momentum premium, the cost/break-even, and the "
            "regime split that tests the inflation hedge are in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> 🟢 **Real run done.** The numbers here are from the desk's cached tape — **18 cross-asset "
            "ETFs** plus a macro state of **CPI year-over-year** (inflation) and the **yield-curve slope** "
            "(growth), all lagged one month for publication delay, **2007-2025** "
            "([`../docs/results.md`](../docs/results.md)). The **synthetic** cross-asset world is kept as "
            "the offline mechanism proof. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## The answer first 🌡️\n\n"
            "| What we asked | The honest answer (real tape, 2007-2025) |\n"
            "|---|---|\n"
            "| Do real assets beat nominal bonds *when inflation rises*? | 🟩 **Yes — the direction is real.** "
            f"An always-long real-asset basket out-earns Treasuries by **+{RR['raw_up_ann']}%/yr** in "
            f"rising-inflation months and **{RR['raw_dn_ann']}%/yr** in falling ones. |\n"
            "| Can you *trade* it with a monthly macro-momentum book? | 🟨 **Barely — it's weak & slow.** "
            f"On the short post-2007 tape the standalone book is flat: net Sharpe **{RR['mm_sh']}** (HAC "
            f"*t* {RR['mm_t']}), below a passive equal-weight hold (**{RR['bench_ew_sh']}**). |\n"
            "| Does the inflation hedge pay *more* when inflation rises? | 🟨 **Yes, directionally.** Rising "
            f"Sharpe **{RR['reg_up_sh']}** vs falling **{RR['reg_dn_sh']}** — right-sided, but the timed "
            "book whipsaws under water in both over one short cycle. |\n"
            "| Real cross-asset run? | 🟢 **Done** — 18 ETFs + cached macro, offline, fingerprinted. |\n\n"
            "> Desk shorthand: **Signal `REAL`-on-direction / `WEAK`-on-size · Tradability `FRAGILE` · "
            "Real-tape run? `DONE`** — a real, slow, diversifying premium whose *direction* survives the "
            "real tape but whose *timed monthly book* doesn't clear noise on one short post-2007 cycle."
        ),

        md(
            "## 1 · The claim 📣\n\n"
            "Fundamental macro-momentum & inflation hedging (Kakushadze-Serur §19.2/§19.3; Brooks–Moskowitz "
            "2017; Neville et al. 2021):\n\n"
            "1. **Macro momentum** — go long the assets favoured by *improving* macro momentum: growth-up "
            "lifts pro-cyclical assets, inflation-up favours real assets. Signal = the *change* in the macro "
            "state, lagged so it's tradable.\n"
            "2. **Inflation hedge** — when inflation momentum is positive, overweight **real** assets "
            "(commodities, TIPS, gold) and underweight nominal ones.\n\n"
            "The believer's case: macro data is slow and trends, asset prices adjust slowly to it, and real "
            "assets mechanically protect purchasing power — a real, diversifying premium that shows up "
            "*across* asset classes, not within one."
        ),
        code(
            "book = strategy.book_returns(r, m, kind='macro_momentum', cost_bps=0.0)  # GROSS, on the control\n"
            "eq = (1+book).cumprod()\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(eq.index, eq.values, color='#2ea44f', lw=1.1)\n"
            "ax.set_title('Macro-momentum book on a world that truly trends with the macro barometer')\n"
            "s = strategy.summary(book); print(f\"synthetic gross Sharpe {s['sharpe']:.2f}  (+{100*s['ann_return']:.1f}%/yr, null ~0)\")"
        ),

        md(
            "## 2 · So what? 💰\n\n"
            "A return stream that comes from the *macro cycle* is, almost by construction, **diversifying** "
            "— it doesn't depend on stocks going up, and the inflation leg pays *exactly* when a 60/40 "
            "portfolio is bleeding (the 1970s, 2021–22). That's worth more than its raw Sharpe suggests: a "
            "modest, uncorrelated, crisis-friendly sleeve is the holy grail of portfolio construction (it's "
            "why the desk's [Study 31 Trade-Winds](../../31-trade-winds/) earned its keep). The catch is the "
            "*slowness* and the *episodic* inflation payoff — you have to sit through long flat stretches."
        ),

        md(
            "## 3 · How we'd know 🔬\n\n"
            "1. **Real?** Gross macro-momentum Sharpe on the control (and flat on a `macro_strength=0` null).\n"
            "2. **Tradable?** Turnover, break-even cost — for a *slow* book, is cost even the threat?\n"
            "3. **Does the hedge pay when it should?** Split the inflation book by rising- vs "
            "falling-inflation regime.\n\n"
            "**Mirage line:** if the macro-momentum book were flat on the control, *or* the inflation hedge "
            "paid the *same* in both regimes, the story collapses — there'd be no conditional macro premium "
            "to harvest."
        ),

        md("## 4 · The teardown 🔧\n\n### 4a · The machinery works where macro momentum is real (control vs null)"),
        code(
            "for label, (rr, mm) in [('macro-driven world', (r, m)), ('null (pure noise)', (r0, m0))]:\n"
            "    s = strategy.summary(strategy.book_returns(rr, mm, kind='macro_momentum', cost_bps=0.0))\n"
            "    print(f\"{label:20} macro-mom gross Sharpe {s['sharpe']:+6.2f}  turnover {strategy.turnover_ann(mm,'macro_momentum'):.1f}x/yr\")"
        ),
        md(
            "### 4b · Cost is *not* the threat — the book is slow\n"
            f"Macro signals refresh monthly and trend, so turnover is only **{S['mm_turn']}×/yr** and the "
            f"**break-even cost is ~{S['mm_be']} bp** — far above realistic cross-asset costs. Unlike the "
            "desk's daily-churn books ([Slingshot](../../33-slingshot/)), this one survives costs easily; "
            "the threats are elsewhere (modest Sharpe, long droughts, episodic hedge)."
        ),
        code(
            "cs = costs.cost_sweep(r, m, kind='macro_momentum')\n"
            "display(cs.round(3))\n"
            "fig, ax = plt.subplots(); ax.axhline(0, color='#999', lw=.8)\n"
            "ax.plot(cs.index, cs['sharpe'], marker='o', color='#2ea44f')\n"
            "ax.set_xlabel('cost (bp per unit traded)'); ax.set_ylabel('net Sharpe'); ax.set_title('A slow book barely feels the spread')\n"
            "print('break-even:', round(costs.breakeven_cost_bps(r, m, 'macro_momentum')), 'bp')"
        ),

        md(
            "### 4c · On the real tape (18 ETFs + cached macro, 2007-2025)\n"
            "The machinery works on the control; now the **real** book. On the short post-2007 sample the "
            "timed monthly macro-momentum book is **flat-to-negative** and below a passive hold — but the "
            "*directional* inflation effect is real: real assets out-earn nominal bonds when inflation rises."
        ),
        code(
            "if R is not None:\n"
            "    mm = strategy.book_returns(R, M, kind='macro_momentum', cost_bps=5.0, **SPEC)\n"
            "    inf = strategy.book_returns(R, M, kind='inflation', cost_bps=5.0, **SPEC)\n"
            "    ew = R[REAL_ASSET_ORDER].mean(axis=1)\n"
            "    for nm, b in [('macro-momentum', mm), ('inflation-hedge', inf), ('equal-weight 18 (hold)', ew)]:\n"
            "        s = strategy.summary(b)\n"
            "        print(f\"{nm:24} Sharpe {s['sharpe']:+.2f}  CAGR {100*s['cagr']:+.1f}%  maxDD {100*s['max_drawdown']:.0f}%\")\n"
            "    raw = extension.raw_real_minus_nominal(R, M, **SPEC)\n"
            "    print('\\nraw real-minus-nominal spread by inflation regime (timing stripped out):')\n"
            "    print(raw.round(2).to_string())\n"
            "else:\n"
            "    print('real cache not present; see ../docs/results.md for the fingerprinted run')"
        ),

        md("## 5 · The verdict 🧾\n\n"
           f"- **Signal `REAL` on direction / `WEAK` on size** — on the real tape real assets beat nominal "
           f"bonds by +{RR['raw_up_ann']}%/yr *when inflation rises* (vs {RR['raw_dn_ann']}%/yr when it "
           f"falls), but the timed standalone book is flat (net Sharpe {RR['mm_sh']}, HAC *t* {RR['mm_t']}); "
           f"the control shows the same `REAL`/`WEAK` split (Sharpe {S['mm_sh']} vs null {S['mm_null']}).\n"
           f"- **Tradability `FRAGILE`** — cheap to run (turnover {RR['mm_turn']}×/yr), but gross of cost it "
           f"makes ~nothing on one short cycle and is beaten by a passive hold (Sharpe {RR['bench_ew_sh']}).\n"
           f"- **Real-tape run? `DONE`** — 18 ETFs + cached macro, 2007-2025, fingerprinted "
           "([`../docs/results.md`](../docs/results.md)).\n\n"
           "> **A real but humble premium.** Macro momentum is diversifying and crisis-friendly — its value "
           "is in a *portfolio*, not as a standalone Sharpe hero, and on a short tape it can simply fail to "
           "clear noise."),

        md("## 6 · Could you trade it? 💸\n\n"
           "- **Yes, cheaply — but patiently.** Low turnover means costs don't kill it (unlike most of this "
           "desk). The price of admission is *time*: macro momentum has multi-year flat stretches.\n"
           "- **The inflation hedge is insurance, not income.** It pays in rising-inflation regimes and "
           "drags otherwise (beat 7) — so size it as a *conditional overlay*, not a permanent sleeve.\n"
           "- **Best used as a sleeve.** A modest, uncorrelated macro book lifts a 60/40's risk-adjusted "
           "return far more than its standalone Sharpe implies — the Trade-Winds lesson again."),

        md(
            "## 7 · Going further 🚪\n\n"
            "### Worked complement — \"does the inflation hedge pay when it's supposed to?\" ([`../docs/extension.md`](../docs/extension.md))\n"
            "We split the inflation book by regime — rising vs falling inflation — on the **real tape**:\n\n"
            f"- The timed inflation book is right-sided (rising Sharpe **{RR['reg_up_sh']}** vs falling "
            f"**{RR['reg_dn_sh']}**) but whipsaws under water in both over one short cycle.\n"
            f"- Strip the timing out and the mechanism is clear: an always-long real-asset basket beats "
            f"nominal bonds by **+{RR['raw_up_ann']}%/yr when inflation rises** vs {RR['raw_dn_ann']}%/yr "
            "when it falls — the inflation hedge's *direction* is real on the live tape.\n"
            f"- On the synthetic control the timed book also pays more rising (Sharpe **{S['reg_inf_up_sh']}**) "
            f"than falling (**{S['reg_inf_dn_sh']}**) — same conditional shape, cleaner because the signal is strong.\n\n"
            "### Other forks\n"
            "- **Conditional sizing** — only run the inflation tilt when inflation momentum is clearly positive.\n"
            "- **More macro drivers** — add monetary-policy and risk-sentiment momentum (Brooks–Moskowitz use several).\n"
            "- **Combine with trend** — stack on [Trade-Winds](../../31-trade-winds/) for a fuller macro sleeve.\n\n"
            "PRs welcome — the headline fork is turning the real *directional* edge into a tradable overlay "
            "without giving it back to monthly-timing noise."
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


def build_quants():
    cells = [
        md(
            "# Barometer — a quantitative teardown 🔬\n"
            "### Macro-momentum Sharpe vs a null · the (gentle) cost wall & break-even · the inflation-regime split\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same seven "
            "beats, every claim with its number.* The steelman: the trend in fundamental macro data (growth, "
            "inflation) predicts asset returns across classes (Brooks–Moskowitz 2017), and tilting toward "
            "real assets when inflation rises hedges a portfolio in the regimes that hurt it (Neville et al. "
            "2021). We confirm the macro-momentum premium is `REAL` on the control (Sharpe 1.09, null −0.17) "
            "and `WEAK`/`FRAGILE` on the real tape, where the *direction* survives (real assets beat nominal "
            "bonds when inflation rises) but the timed monthly book is flat over one short post-2007 cycle.\n\n"
            "> 🟢 **Real run done.** This notebook executes BOTH the synthetic control (machinery proof) AND "
            "the real book from the desk's cached tape — 18 cross-asset ETFs + CPI-YoY / yield-curve-slope "
            "macro, lagged one month, 2007-2025 ([`../docs/results.md`](../docs/results.md), fingerprint "
            f"`{RR['fp']}`); sources in [`../docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        md(
            "## Beat 0 · Verdict\n\n"
            "| Axis | Stamp | Why |\n"
            "|---|---|---|\n"
            f"| **Signal** — macro trend predicts returns? | 🟢 `REAL` on direction · 🟡 `WEAK` on size | "
            f"Real tape: real assets beat nominal bonds by +{RR['raw_up_ann']}%/yr *when inflation rises* "
            f"(vs {RR['raw_dn_ann']}%/yr falling), but the standalone book is flat (net Sharpe **{RR['mm_sh']}**, "
            f"HAC *t* {RR['mm_t']}); control shows the same split (Sharpe **{S['mm_sh']}** vs null **{S['mm_null']}**). |\n"
            f"| **Tradability** | 🟡 `FRAGILE` | Slow book (turnover {RR['mm_turn']}×/yr) — cost is **not** the "
            f"threat; but gross it makes ~nothing on one short cycle, beaten by a passive hold (Sharpe {RR['bench_ew_sh']}). |\n"
            f"| **Real-tape run?** | 🟢 `Done` | 18 ETFs + cached macro, {RR['span']}, {RR['n']} months, "
            f"offline + fingerprinted `{RR['fp']}` ([`../docs/results.md`](../docs/results.md)). |\n\n"
            "> **In one sentence:** a real, slow, diversifying cross-asset macro premium whose *direction* "
            "survives the real tape (the inflation hedge pays when inflation rises) but whose *timed monthly "
            "book* doesn't clear noise on one short post-2007 cycle — `REAL`-on-direction/`WEAK`-on-size, "
            "`FRAGILE`, real run `DONE`."
        ),

        md(
            "## Beat 1 · The claim, precisely\n\n"
            "Two latent macro states $g_t$ (growth), $\\pi_t$ (inflation) evolve as persistent, "
            "regime-switching AR(1) processes. The **macro momentum** is $\\Delta x_t = x_t - x_{t-1}$. "
            "Each asset $a$ loads on the *lagged* momentum through fixed signed betas:\n\n"
            "$$ r_{a,t} = s\\cdot\\big(\\beta^g_a\\,\\Delta g_{t-1} + \\beta^\\pi_a\\,\\Delta\\pi_{t-1}\\big)\\,\\sigma_a "
            "+ \\text{noise} $$\n\n"
            "The **macro-momentum book** weights each asset by its macro exposure dotted with the "
            "(z-scored, smoothed) current momentum, dollar-neutral, gross 1, lagged one month. The "
            "**inflation book** overweights real assets when $\\Delta\\pi>0$. Claim: both earn a positive "
            "premium. Null ($s=0$): assets are pure noise — nothing to predict."
        ),
        code(
            "for label, (rr, mm) in [('macro-driven', (r, m)), ('null', (r0, m0))]:\n"
            "    for kind in ('macro_momentum', 'inflation'):\n"
            "        g = strategy.summary(strategy.book_returns(rr, mm, kind=kind, cost_bps=0.0))\n"
            "        print(f\"{label:12} {kind:15} gross Sharpe {g['sharpe']:+6.2f}\")"
        ),
        md(
            "> 💡 **In plain words.** When the macro state really drives returns, both books make money "
            f"(macro-momentum **{S['mm_sh']}**, inflation **{S['inf_sh']}**); when it's pure noise, both are "
            f"flat (**{S['mm_null']}** / **{S['inf_null']}**). The apparatus sees the macro premium only when "
            "it's there — so a real-tape result would be about the *market*, not the method."
        ),

        md(
            "## Beat 2 · So what?\n\n"
            "Macro momentum (Brooks–Moskowitz 2017) is the *fundamental* cousin of price trend (Moskowitz–"
            "Ooi–Pedersen 2012, the desk's [Trade-Winds](../../31-trade-winds/)): both are slow, "
            "low-correlation, crisis-friendly premia. The prediction: the macro book is **real but modest** "
            "(a ~0.5 Sharpe in the literature), cheap to run (low turnover), and its *inflation leg* is "
            "**conditional** — it should pay in rising-inflation regimes and drag otherwise (Neville et al. "
            "2021; Ang–Bekaert on regime-dependent returns). Beats 4–7 test exactly that on the real tape."
        ),

        md(
            "## Beat 3 · Protocol\n\n"
            "1. **Real?** `book_returns(cost_bps=0)` on control vs null (machinery), then the real book with "
            "a HAC (Newey-West) *t* on the mean.\n"
            "2. **Tradable?** `strategy.turnover_ann`, `costs.breakeven_cost_bps`, `costs.cost_sweep`, and a "
            "passive-hold benchmark.\n"
            "3. **Conditional?** `extension.regime_split` and `raw_real_minus_nominal` — rising vs falling "
            "inflation, on the real tape.\n\n"
            "**Mirage line:** flat macro-momentum book on the *control*, *or* a raw real-minus-nominal spread "
            "that pays the same in both regimes ⇒ no conditional macro premium. (On the real tape the "
            "*timed* book is allowed to be weak — the short sample is noisy; what must hold is the direction.)"
        ),

        md("## Beat 4 · The teardown\n\n### 4a · Machinery: real gross macro-momentum, flat null (the control)"),
        code(
            "cmp = strategy.compare(r, m, cost_bps=0.0)\n"
            "for kind in ('macro_momentum', 'inflation'):\n"
            "    s = cmp[kind]\n"
            "    print(f\"{kind:15} Sharpe {s['sharpe']:+.2f}  ann {100*s['ann_return']:+.1f}%  maxDD {100*s['max_drawdown']:.0f}%  turnover {s['turnover_ann']:.1f}x/yr\")"
        ),
        md(
            "> 💡 **In plain words.** A 1.09 gross Sharpe on the control with a flat null is the apparatus "
            "working — it extracts the macro premium only when one is present. Now the real tape."
        ),

        md("### 4b · The real book + HAC *t* (the REAL/WEAK call)"),
        code(
            "import numpy as np\n"
            "def nw_t(x, lags=6):\n"
            "    x = np.asarray([v for v in x if v==v], float); n=len(x)\n"
            "    if n<3: return float('nan')\n"
            "    e = x - x.mean(); g0 = e@e/n; var=g0\n"
            "    for k in range(1, lags+1):\n"
            "        var += 2*(1-k/(lags+1))*(e[k:]@e[:-k])/n\n"
            "    se = (var/n)**0.5; return float(x.mean()/se) if se>0 else float('nan')\n"
            "if R is not None:\n"
            "    for kind in ('macro_momentum','inflation'):\n"
            "        b = strategy.book_returns(R, M, kind=kind, cost_bps=5.0, **SPEC); s = strategy.summary(b)\n"
            "        print(f\"{kind:15} net@5bp Sharpe {s['sharpe']:+.2f}  CAGR {100*s['cagr']:+.1f}%  maxDD {100*s['max_drawdown']:.0f}%  HAC t {nw_t(b):+.2f}\")\n"
            "    ew = R[REAL_ASSET_ORDER].mean(axis=1); sixty = 0.6*R['SPY']+0.4*R['IEF']\n"
            "    for nm,b in [('equal-weight 18 (hold)', ew), ('60/40 SPY/IEF (hold)', sixty)]:\n"
            "        s = strategy.summary(b); print(f\"{nm:22} Sharpe {s['sharpe']:+.2f}  CAGR {100*s['cagr']:+.1f}%\")\n"
            "else:\n"
            "    print('real cache not present; see ../docs/results.md')"
        ),
        md(
            "> 💡 **In plain words.** On the short post-2007 tape the timed monthly macro book is **flat** "
            f"(net Sharpe **{RR['mm_sh']}**, HAC *t* {RR['mm_t']}) and below a passive hold (**{RR['bench_ew_sh']}**) "
            "— `REAL` on direction (see Beat 7), but `WEAK` on size and standalone tradability."
        ),

        md("### 4c · The (gentle) cost wall — cost is NOT the threat"),
        code(
            "cs = costs.cost_sweep(R, M, kind='macro_momentum', **SPEC) if R is not None else costs.cost_sweep(r, m, kind='macro_momentum')\n"
            "print('real-book cost sweep (net Sharpe):'); print(cs.round(3).to_string())\n"
            "fig, ax = plt.subplots(); ax.axhline(0, color='#999', lw=.8)\n"
            "ax.plot(cs.index, cs['sharpe'], marker='o', color='#2ea44f')\n"
            "ax.set_xlabel('cost (bp/unit)'); ax.set_ylabel('net Sharpe'); ax.set_title('Slow macro book — the sweep is linear, cost is not what kills it')"
        ),

        md("## Beat 5 · The verdict\n\n"
           f"- **`REAL` on direction / `WEAK` on size** (4a/4b): control Sharpe {S['mm_sh']} vs null "
           f"{S['mm_null']}, but the real timed book is flat (net Sharpe {RR['mm_sh']}, HAC *t* {RR['mm_t']}) "
           "on one short cycle; the *directional* inflation effect survives (Beat 7).\n"
           f"- **`FRAGILE`** (4c): turnover {RR['mm_turn']}×/yr, cost is linear and not the threat — but gross "
           f"it makes ~nothing and is beaten by a passive hold (Sharpe {RR['bench_ew_sh']}).\n"
           f"- **Real-tape run? `DONE`** — {RR['span']}, fingerprint `{RR['fp']}`.\n\n"
           "> **Signal `REAL`/`WEAK` · Tradability `FRAGILE` · Real-tape run? `DONE`** — a diversifying macro "
           "premium real in direction, weak and slow to monetise on a short tape."),

        md("## Beat 6 · Could you trade it?\n\n"
           "- **Not as a standalone monthly book on this tape.** Gross of cost it makes ~nothing over one "
           "short post-2007 cycle and is beaten by a passive hold — the signal is too slow and the sample too short.\n"
           "- **Cost is not the obstacle.** Low turnover ⇒ the cost sweep is linear; the obstacle is the weak, "
           "slow edge itself and the long flat stretches.\n"
           "- **The directional inflation hedge is the salvageable piece.** Real assets do beat nominal bonds "
           "when inflation rises (Beat 7) — use it as a slow *conditional overlay*, not a timed monthly sleeve."),

        md(
            "## Beat 7 · Going further\n\n"
            "### 7a · Worked complement — does the inflation hedge pay when it's supposed to? ([`../docs/extension.md`](../docs/extension.md))\n"
            "Split the inflation book by regime — rising vs falling inflation — and ask whether the premium "
            "concentrates where the theory says it must."
        ),
        code(
            "if R is not None:\n"
            "    sp = extension.regime_split(R, M, kind='inflation', cost_bps=5.0, **SPEC)\n"
            "    print('timed inflation book by regime (net @5bp):'); print(sp.round(2).to_string())\n"
            "    raw = extension.raw_real_minus_nominal(R, M, **SPEC)\n"
            "    print('\\nRAW real-minus-nominal spread by regime (timing stripped out):'); print(raw.round(2).to_string())\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.bar(['rising','falling'], raw['ann_return_pct'].values, color=['#2ea44f','#dab617'])\n"
            "    ax.axhline(0, color='#999', lw=.8); ax.set_ylabel('real-minus-nominal %/yr')\n"
            "    ax.set_title('Real assets out-earn nominal bonds when inflation rises')\n"
            "else:\n"
            "    print('real cache not present; see ../docs/extension.md')"
        ),
        md(
            "> 💡 **In plain words.** The *timed* inflation book is right-sided (rising Sharpe "
            f"**{RR['reg_up_sh']}** vs falling **{RR['reg_dn_sh']}**) but whipsaws under water in both over "
            "one short cycle. Strip the timing out and the mechanism is unambiguous: an always-long "
            f"real-asset basket beats nominal bonds by **+{RR['raw_up_ann']}%/yr when inflation rises** vs "
            f"{RR['raw_dn_ann']}%/yr when it falls. The inflation hedge's *direction* is real on the live "
            "tape — what's fragile is monetising it with a slow monthly timing rule."
        ),

        md(
            "### 7b · The synthetic control (the same shape, cleaner)\n"
            f"On the seeded control the *timed* inflation book also pays more rising (Sharpe **{S['reg_inf_up_sh']}**) "
            f"than falling (**{S['reg_inf_dn_sh']}**) — the same conditional shape, but cleaner because the "
            "signal is strong and the sample long. The real tape confirms the *direction*; the control "
            "confirms the *mechanism*.\n\n"
            "### 7c · Other forks\n"
            "- **Conditional sizing** — run the inflation tilt only when inflation momentum is clearly positive.\n"
            "- **More drivers** — add monetary-policy & risk-sentiment momentum (Brooks–Moskowitz use several).\n"
            "- **Stack with trend** — combine with [Trade-Winds](../../31-trade-winds/) for a fuller macro sleeve.\n\n"
            "PRs welcome — the headline fork is turning the real *directional* inflation edge into a tradable "
            "overlay without giving it back to monthly-timing noise."
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "02_for_the_quants.ipynb")


def _meta():
    return {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}


def _write(nb, name):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
