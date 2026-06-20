"""Generate the two narrative notebooks for Study 332 (Downside-Beta).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached daily
panel under ../_cache/ if present and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md), so the notebook re-runs for any reader. Every cell banners
which tape it is showing — synthetic cells never wear a real-tape banner.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-19).
R = dict(
    n_months=250, n_tickers=40, span="2005-2026",
    fp_panel="9da245a5bd56", fp_mkt="a9c845a0f216",
    # gross long-short spreads
    dbeta_ann=12.17, dbeta_t=2.56, dbeta_sr=0.59, dbeta_hit=0.59,
    dbeta_ci_lo=19, dbeta_ci_hi=176,
    beta_ann=10.46, beta_t=2.09, beta_sr=0.48, beta_hit=0.56,
    reldb_ann=-0.46, reldb_t=-0.14, reldb_sr=-0.03, reldb_hit=0.48,
    reldb_ci_lo=-58, reldb_ci_hi=50,
    # cost sweep (one-way bps -> net ann %, t)
    cost0_ann=12.17, cost0_t=2.56,
    cost5_ann=9.77, cost5_t=2.06,
    cost10_ann=7.37, cost10_t=1.55,
    cost20_ann=2.57, cost20_t=0.54,
    # synthetic positive/null control
    syn_null_ann=-8.41, syn_null_t=-0.85,
    syn_pos_ann=30.02, syn_pos_t=2.87,
    syn_corr=0.92,
    acx_premium=6.0,  # Ang-Chen-Xing's published ~6%/yr
)


# ---------------------------------------------------------------------------
# Shared analysis preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

from downside_beta import data, strategy as st

# --- the real tape is cache-first; fall back to the synthetic control offline ---
HAVE_REAL = os.path.exists(data.PANEL_CACHE) and os.path.exists(data.MARKET_CACHE)
if HAVE_REAL:
    try:
        rdf_real, mkt_real = data.load_real(fetch=False)
        # drop the in-progress month
        last = rdf_real.index[-1]
        cut = pd.Timestamp(last.year, last.month, 1) - pd.Timedelta(days=1)
        rdf_real, mkt_real = rdf_real.loc[:cut], mkt_real.loc[:cut]
        print(f"REAL tape: {rdf_real.shape[0]} days x {rdf_real.shape[1]} tickers, "
              f"{rdf_real.index[0].date()} -> {rdf_real.index[-1].date()}")
    except Exception as e:
        HAVE_REAL = False
        print("Real cache unreadable, falling back to synthetic:", e)
if not HAVE_REAL:
    print("No real cache -- notebook will run on the SYNTHETIC tape and quote frozen "
          "real numbers from docs/results.md.")
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Downside-Beta -- do stocks that crash *with* the market pay you more? \U0001F4C9\n"
            "### Ang, Chen & Xing (2006), tested honestly -- and asked the one question that decides it\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Distinct from beta%3F: Not_supported](https://img.shields.io/badge/Distinct_from_beta%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "A famous 2006 paper found that stocks whose price falls *hardest when the whole "
            "market falls* -- high **downside beta** -- go on to earn more, about **6% a year** "
            "extra. The idea is intuitive: a stock that abandons you exactly when everything "
            "else is sinking is a lousy hedge, so you should be paid to hold it. This notebook "
            "asks two things in plain language: is that premium real on today's tape, and -- the "
            "question that actually decides it -- is it anything more than the ordinary reward "
            "for holding *risky* (high-beta) stocks?\n\n"
            "> This is the plain-language layer. The downside-beta maths, the HAC *t*-stats, the "
            "beta control and the cost sweep live in the companion "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** Reproducible research: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT ------------------------------------------------
        md(
            "## The answer first \U0001F3AF\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do high downside-beta stocks earn more? | **On the raw sort, yes** -- about "
            f"+{R['dbeta_ann']:.0f}%/yr (HAC *t* = +{R['dbeta_t']:.2f}) on our tape. |\n"
            f"| Is it a *downside* story, or just a *beta* story? | **Just beta.** A plain-beta "
            f"sort earns nearly the same (+{R['beta_ann']:.0f}%/yr), and the part of downside "
            f"beta that beta *doesn't* explain earns **{R['reldb_ann']:+.1f}%/yr** (*t* = "
            f"{R['reldb_t']:+.2f}) -- nothing. |\n"
            f"| Could you trade it? | **No.** At 10 bps costs the spread slips below the "
            f"significance bar, and what's left is plain market exposure you can buy cheaper. |\n\n"
            "So: a real-looking number that is really the oldest premium in finance (beta) "
            "wearing a downside costume."
        ),
        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "Ordinary finance measures a stock's risk by its **beta**: how much it moves when "
            "the market moves, up *or* down, treated the same. Ang, Chen & Xing argued that "
            "investors don't feel up and down the same way -- losing money when *everything* is "
            "crashing hurts far more than missing out on a rally. So they split beta in two:\n\n"
            "- **Downside beta (β⁻)** -- how much a stock moves on days the market is *down*.\n"
            "- **Upside beta (β⁺)** -- how much it moves on days the market is *up*.\n\n"
            "Their finding: stocks with high β⁻ earn a premium of roughly "
            f"**{R['acx_premium']:.0f}% per year** over low-β⁻ stocks, and -- the strong "
            "claim -- this is *not* just the ordinary beta premium in disguise."
        ),
        # ---- BEAT 2 -- SO WHAT? ----------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, it's a genuinely new risk factor: a way to earn extra return by "
            "deliberately holding the stocks that hurt most in crashes -- and a reason the plain "
            "CAPM mis-prices the cross-section. If *false* -- if β⁻ only pays because "
            "high-β⁻ stocks are just high-β stocks -- then the 'downside premium' is "
            "a relabelling of something we've known since 1964, and you'd be paying turnover and "
            "shorting costs to harvest market beta the hard way."
        ),
        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "The test is a sort. Each month, rank every stock by its trailing downside beta, "
            "buy the top fifth, short the bottom fifth, hold a month, repeat. If the long-short "
            "earns a positive, statistically real return, the premium exists.\n\n"
            "**The trap, announced in advance:** high-β⁻ stocks are usually just "
            "high-β stocks. So we run the *exact same sort on plain beta* as a control, and "
            "-- the decisive test -- we sort on **relative downside beta** (β⁻ minus "
            "β), the part of downside risk that ordinary beta does *not* explain. If that "
            "earns nothing, the downside story is busted, however good the raw number looks."
        ),
        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 · The teardown -- let's actually look\n\n"
            "First, a sanity check on a **synthetic market we built ourselves**, where we *know* "
            "the answer. We plant a real downside premium in one tape and none in another, and "
            "confirm our sorting machine finds it only where it exists. Then we turn the same "
            "machine on the real tape."
        ),
        code(
            "# --- machinery proof on the SYNTHETIC tape (we know the truth here) ---\n"
            "rdf_null, mkt_null, _ = data.synthetic_panel(downside_premium=0.0, seed=332)\n"
            "rdf_pos,  mkt_pos,  _ = data.synthetic_panel(downside_premium=0.0025, seed=332)\n"
            "res_null = st.monthly_quantile_returns(rdf_null, mkt_null, signal_col='beta_dn')\n"
            "res_pos  = st.monthly_quantile_returns(rdf_pos,  mkt_pos,  signal_col='beta_dn')\n"
            "s_null = st.summarize(res_null['spread']); s_pos = st.summarize(res_pos['spread'])\n"
            "print(f\"SYNTHETIC null   : spread {s_null['mean_ann']*100:6.1f}%/yr  t={s_null['tstat']:+.2f}\")\n"
            "print(f\"SYNTHETIC priced : spread {s_pos['mean_ann']*100:6.1f}%/yr  t={s_pos['tstat']:+.2f}\")\n"
            "print('The machine finds the premium only when we plant one. Good.')"
        ),
        code(
            "# --- the real question: raw beta_dn sort vs the beta control (REAL tape if cached) ---\n"
            "if HAVE_REAL:\n"
            "    tape = 'REAL'\n"
            "    rr = {c: st.summarize(st.monthly_quantile_returns(rdf_real, mkt_real, signal_col=c, min_stocks=8)['spread'])\n"
            "          for c in ['beta_dn','beta','rel_dbeta']}\n"
            "    dbeta_ann, dbeta_t = rr['beta_dn']['mean_ann']*100, rr['beta_dn']['tstat']\n"
            "    beta_ann,  beta_t  = rr['beta']['mean_ann']*100,    rr['beta']['tstat']\n"
            "    reldb_ann, reldb_t = rr['rel_dbeta']['mean_ann']*100, rr['rel_dbeta']['tstat']\n"
            "else:\n"
            "    tape = 'FROZEN (docs/results.md)'\n"
            f"    dbeta_ann, dbeta_t = {R['dbeta_ann']}, {R['dbeta_t']}\n"
            f"    beta_ann,  beta_t  = {R['beta_ann']}, {R['beta_t']}\n"
            f"    reldb_ann, reldb_t = {R['reldb_ann']}, {R['reldb_t']}\n"
            "labels = ['Downside beta\\n(β⁻)', 'Plain beta\\n(β)', 'Relative β⁻\\n(β⁻−β)']\n"
            "vals = [dbeta_ann, beta_ann, reldb_ann]; ts = [dbeta_t, beta_t, reldb_t]\n"
            "cols = [AMBER, GREY, RED]\n"
            "fig, ax = plt.subplots()\n"
            "bars = ax.bar(labels, vals, color=cols)\n"
            "for b, t in zip(bars, ts):\n"
            "    ax.text(b.get_x()+b.get_width()/2, b.get_height(), f't={t:+.2f}', ha='center',\n"
            "            va='bottom' if b.get_height()>=0 else 'top', fontsize=11, fontweight='bold')\n"
            "ax.axhline(0, color='k', lw=.8)\n"
            "ax.set_ylabel('long-short spread (%/yr)')\n"
            "ax.set_title(f'The downside premium is just the beta premium  [{tape} tape]')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'[{tape}]  beta_dn {dbeta_ann:+.1f}%/yr (t={dbeta_t:+.2f}) | '\n"
            "      f'beta {beta_ann:+.1f}%/yr (t={beta_t:+.2f}) | rel_dbeta {reldb_ann:+.1f}%/yr (t={reldb_t:+.2f})')"
        ),
        md(
            "The picture says it all. The raw downside-beta sort (amber) looks like a real "
            "premium. The plain-beta sort (grey) earns almost exactly the same. And the "
            "*relative* downside beta (red) -- the part that would make this a genuinely new "
            "factor -- earns **nothing**. The premium is beta, relabelled."
        ),
        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal -- MIXED.** The raw β⁻ spread is statistically real on this "
            f"tape (+{R['dbeta_ann']:.0f}%/yr, *t* = +{R['dbeta_t']:.2f}). The *downside-"
            f"specific* part is not ({R['reldb_ann']:+.1f}%/yr, *t* = {R['reldb_t']:+.2f}). Real "
            "on the raw sort, none on the downside story.\n"
            "- **Distinct from plain beta? -- NOT SUPPORTED.** The whole premium survives in "
            "the beta sort and vanishes once beta is removed.\n"
            "- **Tradability -- MIRAGE.** It dies under realistic costs, and the part that "
            "doesn't is market beta you can buy far more cheaply."
        ),
        # ---- BEAT 6 -- COULD YOU TRADE IT? ----------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Even the *raw* (beta-flavoured) spread is a heavy monthly long-short: you rebalance "
            "both legs every month, and the short leg is high-beta names that cost the most to "
            "borrow. Here's what plausible trading costs do to it."
        ),
        code(
            "costs = [0, 5, 10, 20]\n"
            "if HAVE_REAL:\n"
            "    res = st.monthly_quantile_returns(rdf_real, mkt_real, signal_col='beta_dn', min_stocks=8)\n"
            "    net_ann = [st.summarize(st.net_spread(res, one_way_bps=c))['mean_ann']*100 for c in costs]\n"
            "    net_t   = [st.summarize(st.net_spread(res, one_way_bps=c))['tstat'] for c in costs]\n"
            "    tape = 'REAL'\n"
            "else:\n"
            f"    net_ann = [{R['cost0_ann']}, {R['cost5_ann']}, {R['cost10_ann']}, {R['cost20_ann']}]\n"
            f"    net_t   = [{R['cost0_t']}, {R['cost5_t']}, {R['cost10_t']}, {R['cost20_t']}]\n"
            "    tape = 'FROZEN'\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot(costs, net_ann, 'o-', color=RED, lw=2)\n"
            "for c, a, t in zip(costs, net_ann, net_t):\n"
            "    ax.annotate(f't={t:+.2f}', (c, a), textcoords='offset points', xytext=(0,8), ha='center')\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('net spread (%/yr)')\n"
            "ax.set_title(f'Costs eat the (beta-flavoured) spread  [{tape} tape]')\n"
            "ax.axhline(0, color='k', lw=.8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'[{tape}] net spread by cost (bps):', dict(zip(costs, [round(a,1) for a in net_ann])))"
        ),
        # ---- BEAT 7 -- GOING FURTHER ----------------------------------------
        md(
            "## 7 · Going further \U0001F6AA\n\n"
            "- **Broaden the universe.** Our basket is 40 survivors; the original paper used the "
            "full CRSP cross-section. A wider, survivorship-free panel might revive a small "
            "*incremental* premium -- or kill it further.\n"
            "- **Coskewness.** Harvey-Siddique (2000) argue the priced thing is *coskewness* "
            "(crashing with the market), a close cousin -- worth running side by side.\n"
            "- **The mirror.** Betting-Against-Beta (Study 238) says *low* beta is what pays. "
            "Two famous papers, opposite tilts, same beta machinery -- fork this and race them."
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Downside-Beta -- a quantitative teardown \U0001F52C\n"
            "### Cross-sectional pricing · β⁻ vs β vs β⁻−β · HAC inference · block bootstrap · capacity\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Distinct from beta%3F: Not_supported](https://img.shields.io/badge/Distinct_from_beta%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) "
            "-- *same seven beats, every claim now carrying its standard error.* We test "
            "Ang-Chen-Xing (2006) as a cross-sectional pricing claim and separate the raw "
            "downside-beta sort from its incremental (over-beta) content, which is what decides "
            "whether 'downside risk' is a factor or a costume.\n\n"
            "> **Not investment advice.** Yahoo daily adjusted closes (total-return adjusted) "
            "for a large-cap S&P 500 basket, equal-weight market proxy; downside beta on a "
            "trailing 252-day window; one-month execution lag; HAC *t* (Newey-West) and circular "
            "block-bootstrap CIs. Sources: [`docs/references.md`](../docs/references.md), pinned "
            "run: [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> \U0001F4A1 **The `\U0001F4A1 In plain words` notes** translate each result back into "
            "intuition. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Why |\n|---|---|---|\n"
            f"| **Signal** | Mixed | Raw β⁻ spread +{R['dbeta_ann']:.2f}%/yr, HAC "
            f"*t* = +{R['dbeta_t']:.2f}; but relative β⁻−β earns "
            f"{R['reldb_ann']:+.2f}%/yr, *t* = {R['reldb_t']:+.2f} -- real on the raw sort, none "
            "on the downside-specific part. Survivorship-biased basket. |\n"
            f"| **Tradability** | Mirage | Net of 10 bps one-way, *t* falls to +{R['cost10_t']:.2f} "
            "(below the bar), before short borrow; the tradeable part is plain beta. |\n"
            "| **Distinct from plain beta?** | Not supported | β⁻−β earns "
            f"nothing (*t* = {R['reldb_t']:+.2f}); the premium is the beta premium relabelled. |\n\n"
            "> \U0001F4A1 **In plain words.** The downside-beta sort 'works' the way a thermometer "
            "in a sauna 'works' -- it reads hot, but it's measuring the room (beta), not a new "
            "thing (downside risk)."
        ),
        # ---- BEATS 1-7 -------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Ang, Chen & Xing (2006) decompose market beta around the market mean:\n\n"
            "$$\\beta^- = \\frac{\\mathrm{cov}(r_i, r_m \\mid r_m < \\mu_m)}{\\mathrm{var}(r_m \\mid r_m < \\mu_m)}, "
            "\\qquad \\beta^+ = \\frac{\\mathrm{cov}(r_i, r_m \\mid r_m > \\mu_m)}{\\mathrm{var}(r_m \\mid r_m > \\mu_m)}.$$\n\n"
            "- **H₁ (the headline):** sorting on β⁻ yields a positive high-minus-low "
            "spread with HAC *t* ≥ 2.\n"
            "- **H₂ (the decisive one):** the spread survives controlling for plain β -- "
            "operationalised as a sort on **β⁻ − β** (their own relative "
            "downside beta) earning a positive, significant spread.\n"
            "- **H₃ (tradability):** the net-of-cost spread keeps *t* ≥ 2 at realistic "
            "costs.\n\n"
            "We use μ_m = 0 (down = a negative market day) as the canonical threshold; a "
            "market-mean threshold is available as a robustness knob."
        ),
        md(
            "## 2 · So what? -- what rides on each answer\n\n"
            "H₂ is the whole game. A raw β⁻ premium that vanishes once β is "
            "removed is not a new factor; it is the CAPM market premium, harvested through an "
            "expensive long-short instead of a cheap index. Confirming H₁ while rejecting "
            "H₂ is exactly the kind of 'real but misattributed' result this desk exists to "
            "flag."
        ),
        md(
            "## 3 · How we'd know -- the protocol\n\n"
            "**Decompose** → estimate (β, β⁻, β⁺) on a trailing "
            "252-day window per name. **Sort** → monthly quintile long-short on each of "
            "β⁻, β, β⁻−β, with one execution lag (signal at "
            "*t*−1 month-end earns month *t*). **Robust inference** → Newey-West HAC "
            "*t* and circular block-bootstrap CIs. **Control** → random same-sized "
            "partitions (concentration null). **Capacity** → one-way × NAV cost sweep, "
            "both legs. **Positive control** → synthetic tape with a planted premium."
        ),
        md("## 4 · The teardown"),
        md(
            "### 4a · Positive/null control -- the harness is a faithful detector\n\n"
            "Before trusting any real number, prove the machine detects the effect it claims to. "
            "On a synthetic firm × day panel we plant a downside premium (or none) and check "
            "the sort recovers it (or doesn't)."
        ),
        code(
            "rows = []\n"
            "for prem, label in [(0.0, 'null'), (0.0025, 'priced')]:\n"
            "    rdf, mkt, truth = data.synthetic_panel(downside_premium=prem, seed=332)\n"
            "    res = st.monthly_quantile_returns(rdf, mkt, signal_col='beta_dn')\n"
            "    s = st.summarize(res['spread'])\n"
            "    lo, hi = st.block_bootstrap_ci(res['spread'], n_boot=1000)\n"
            "    rows.append({'tape': label, 'spread %/yr': s['mean_ann']*100, 'HAC t': s['tstat'],\n"
            "                 'CI lo bps/mo': lo*1e4, 'CI hi bps/mo': hi*1e4, 'n': s['n']})\n"
            "    if prem > 0:\n"
            "        panel = st.beta_panel(rdf, mkt, window_end=rdf.index[-1])\n"
            "        corr = np.corrcoef(panel['rel_dbeta'].to_numpy(), truth['gamma'])[0,1]\n"
            "ctrl = pd.DataFrame(rows).set_index('tape'); display(ctrl.round(2))\n"
            "print(f'corr(estimated β⁻−β, planted downside loading) = {corr:.2f} '\n"
            "      '-- the sort really keys on downside risk when downside risk is what varies.')"
        ),
        md(
            "> \U0001F4A1 **In plain words.** The machine bags the planted premium (*t* ≈ +2.9) "
            "and finds nothing when there's nothing to find (*t* ≈ −0.9). It is a "
            "trustworthy detector -- so when it finds the *real*-tape premium is all-beta, we "
            "believe it. (This synthetic result is a machinery proof; it can never *back* a "
            "stamp.)"
        ),
        md("### 4b · The real tape -- β⁻ vs β vs β⁻−β"),
        code(
            "if HAVE_REAL:\n"
            "    tape = 'REAL'\n"
            "    out = {}\n"
            "    for c in ['beta_dn', 'beta', 'rel_dbeta']:\n"
            "        res = st.monthly_quantile_returns(rdf_real, mkt_real, signal_col=c, min_stocks=8)\n"
            "        s = st.summarize(res['spread'])\n"
            "        lo, hi = st.block_bootstrap_ci(res['spread'], n_boot=1000)\n"
            "        out[c] = {'spread %/yr': s['mean_ann']*100, 'HAC t': s['tstat'],\n"
            "                  'ann SR': s['sharpe_ann'], 'hit': s['hit_rate'],\n"
            "                  'CI lo bps/mo': lo*1e4, 'CI hi bps/mo': hi*1e4, 'n': s['n']}\n"
            "    real_tbl = pd.DataFrame(out).T\n"
            "else:\n"
            "    tape = 'FROZEN (docs/results.md)'\n"
            "    real_tbl = pd.DataFrame({\n"
            f"        'beta_dn':   {{'spread %/yr': {R['dbeta_ann']}, 'HAC t': {R['dbeta_t']}, 'ann SR': {R['dbeta_sr']}, 'hit': {R['dbeta_hit']}, 'CI lo bps/mo': {R['dbeta_ci_lo']}, 'CI hi bps/mo': {R['dbeta_ci_hi']}, 'n': {R['n_months']}}},\n"
            f"        'beta':      {{'spread %/yr': {R['beta_ann']}, 'HAC t': {R['beta_t']}, 'ann SR': {R['beta_sr']}, 'hit': {R['beta_hit']}, 'CI lo bps/mo': np.nan, 'CI hi bps/mo': np.nan, 'n': {R['n_months']}}},\n"
            f"        'rel_dbeta': {{'spread %/yr': {R['reldb_ann']}, 'HAC t': {R['reldb_t']}, 'ann SR': {R['reldb_sr']}, 'hit': {R['reldb_hit']}, 'CI lo bps/mo': {R['reldb_ci_lo']}, 'CI hi bps/mo': {R['reldb_ci_hi']}, 'n': {R['n_months']}}},\n"
            "    }).T\n"
            "print(f'[{tape}] monthly long-short spreads, gross:')\n"
            "display(real_tbl.round(2))"
        ),
        md(
            "> \U0001F4A1 **In plain words.** Read the three rows top to bottom: β⁻ clears "
            "the bar, β clears it too at nearly the same size, and β⁻−β "
            "(the only one that would make 'downside' a distinct factor) is a flat line through "
            "zero. H₁ holds; H₂ fails."
        ),
        md(
            "## 5 · The verdict -- the decisive statistics in one place\n\n"
            f"- **Signal MIXED.** β⁻: +{R['dbeta_ann']:.2f}%/yr, HAC *t* = "
            f"+{R['dbeta_t']:.2f}, 95% block-boot CI [+{R['dbeta_ci_lo']}, +{R['dbeta_ci_hi']}] "
            f"bps/mo (excludes 0). β⁻−β: {R['reldb_ann']:+.2f}%/yr, *t* = "
            f"{R['reldb_t']:+.2f}, CI [{R['reldb_ci_lo']}, +{R['reldb_ci_hi']}] bps/mo "
            "(straddles 0). Real on the raw sort · None on the downside-specific part.\n"
            "- **Distinct from plain beta? NOT SUPPORTED.** β sort earns "
            f"+{R['beta_ann']:.2f}%/yr (*t* = +{R['beta_t']:.2f}); removing β removes the "
            "whole premium.\n"
            f"- **Tradability MIRAGE.** Net *t*: {R['cost0_t']:+.2f} (0bps) → "
            f"{R['cost5_t']:+.2f} (5) → {R['cost10_t']:+.2f} (10) → {R['cost20_t']:+.2f} "
            "(20), before short borrow.\n\n"
            "Survivorship: the basket is current S&P 500 names projected backwards (opt-in "
            "guard), so even the raw β⁻ leg is an upper bound."
        ),
        md("## 6 · Could you trade it? -- cost sweep, both legs, one-way × NAV"),
        code(
            "costs = [0, 5, 10, 20]\n"
            "if HAVE_REAL:\n"
            "    res = st.monthly_quantile_returns(rdf_real, mkt_real, signal_col='beta_dn', min_stocks=8)\n"
            "    net_ann = [st.summarize(st.net_spread(res, one_way_bps=c))['mean_ann']*100 for c in costs]\n"
            "    net_t   = [st.summarize(st.net_spread(res, one_way_bps=c))['tstat'] for c in costs]\n"
            "    tape = 'REAL'\n"
            "else:\n"
            f"    net_ann = [{R['cost0_ann']}, {R['cost5_ann']}, {R['cost10_ann']}, {R['cost20_ann']}]\n"
            f"    net_t   = [{R['cost0_t']}, {R['cost5_t']}, {R['cost10_t']}, {R['cost20_t']}]\n"
            "    tape = 'FROZEN'\n"
            "sweep = pd.DataFrame({'one-way bps': costs, 'net %/yr': net_ann, 'HAC t': net_t}).set_index('one-way bps')\n"
            "print(f'[{tape}] cost sweep on the β⁻ long-short:'); display(sweep.round(2))\n"
            "print('Break-even on significance lands between 5 and 10 bps -- before any short borrow.')"
        ),
        md(
            "> \U0001F4A1 **In plain words.** A monthly beta-tilted long-short is expensive to run, "
            "and the only thing it harvests is the market premium -- which a buy-and-hold index "
            "fund delivers at a few basis points a year. You'd be paying a steakhouse markup for "
            "a supermarket steak."
        ),
        md(
            "## 7 · Going further\n\n"
            "- **Full CRSP / survivorship-free panel** -- does a small incremental premium "
            "survive on the wider, fairer cross-section the original used?\n"
            "- **Coskewness (Harvey-Siddique 2000)** as a competing measure of crash-comovement, "
            "run head-to-head with β⁻−β.\n"
            "- **Threshold sensitivity** -- μ_m = market mean vs 0; conditional (downside in "
            "*bad* states only) vs unconditional downside beta.\n"
            "- **The BAB mirror** -- race against Study 238 (low beta pays). PRs welcome: fork "
            "`downside_beta/strategy.py`, swap the signal column, re-run the gauntlet."
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
