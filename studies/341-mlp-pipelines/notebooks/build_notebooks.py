"""Generate the two narrative notebooks for Study 341 (MLP-Pipelines).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic MLP
figures run anywhere, offline and deterministic; the real-tape cells read the cached monthly
parquet under the shared _cache/ if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader, online or off, and
banners which tape each cell shows.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-05-31, fp 328f05124cbf).
R = dict(
    # Energy beta vs XLE total return (HAC t on the slope, R2, CI)
    amlp_beta=0.74, amlp_beta_t=5.26, amlp_r2=0.64, amlp_ci_lo=0.49, amlp_ci_hi=0.97,
    mlpa_beta=0.78, mlpa_beta_t=5.19, mlpa_r2=0.65,
    amza_beta=1.14, amza_beta_t=5.46, amza_r2=0.68,
    # The race vs SPY total return
    amlp_cagr=5.3, amlp_sh=0.34, amlp_dd=-73, amlp_n=189,
    mlpa_cagr=3.8, mlpa_sh=0.28, mlpa_dd=-72, mlpa_n=169,
    amza_cagr=-0.8, amza_sh=0.20, amza_dd=-88, amza_n=139,
    spy_cagr=15.4, spy_sh=1.06, spy_dd=-24,
    xle_cagr=8.7, xle_sh=0.44, xle_dd=-65,
    amlp_spread_spy=-7.7, amlp_spread_xle=-3.8,
    amlp_t_spy=-1.47, amlp_t_xle=-0.91,
    # Up/down capture vs XLE (energy)
    amlp_up=0.68, amlp_dn=0.66, amza_up=1.01, amza_dn=1.09,
    # The income illusion: distribution vs NAV vs return-of-capital share
    amlp_dist=7.8, amlp_price=-2.3, amlp_roc=1.00, amlp_total=5.4,
    mlpa_dist=7.9, mlpa_price=-3.8, mlpa_roc=1.00, mlpa_total=3.9,
    amza_dist=14.3, amza_price=-13.4, amza_roc=1.00, amza_total=-0.7,
)


_R_REPR = repr(R)

BOOT = (
"""\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
# Frozen real-tape headline numbers (mirror of docs/results.md, as-of 2026-05-31).
# Used only on a cache miss so the notebook re-runs identically offline.
R = """ + _R_REPR + """
"""
"""\
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from mlp_pipelines import data, strategy as st

AS_OF = pd.Timestamp("2026-05-31")
FUNDS = ["AMLP", "MLPA", "AMZA"]

def load_real():
    panel = data.fetch_panel()      # cache-first, empty on miss
    split = data.fetch_price_and_dist()
    if not panel.empty:
        panel = panel[panel.index <= AS_OF]
    if not split.empty:
        split = split[split.index <= AS_OF]
    return panel, split

PANEL, SPLIT = load_real()
HAVE_REAL = (not PANEL.empty) and ("XLE" in PANEL.columns) and ("AMLP" in PANEL.columns)
print("real monthly cache present:", HAVE_REAL,
      "" if not HAVE_REAL else f"({PANEL.index.min().date()}..{PANEL.index.max().date()})")
""")


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Pipeline MLPs — a fat-yield free lunch, or a value trap? ⛽\n"
            "### AMLP pays a 7–8% 'income'. Where does it come from, and what are you really holding?\n\n"
            "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Fat--yield_free_lunch%3F: Busted](https://img.shields.io/badge/Fat--yield_free_lunch%3F-Busted-8b949e?style=flat-square)\n\n"
            "**Master Limited Partnerships** — the companies that own the oil & gas *pipelines* — are "
            "marketed as a bond-like income machine: own the toll-road of energy, collect a steady "
            "**7–8% distribution**, and let the cash roll in. The most popular wrapper, **AMLP** (the "
            "Alerian MLP ETF), is held by income investors who want yield without (they think) the "
            "swings of stocks. This notebook asks the only two questions that matter — **where does "
            "that 'income' actually come from, and what are you really exposed to?**\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the energy-beta regression "
            "and the capture math? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Where does the ~8% 'yield' come from? | **Your own money.** AMLP's NAV *fell* "
            f"{R['amlp_price']:+.1f}%/yr while it paid out {R['amlp_dist']:.1f}% — so **100% of the "
            f"distribution is return of capital**, principal handed back to you. |\n"
            f"| What are you really holding? | **A leveraged bet on energy.** AMLP's beta to the "
            f"energy sector (XLE) is **{R['amlp_beta']:.2f}** with a rock-solid statistical *t* "
            f"(**{R['amlp_beta_t']:.1f}**). It is an oil trade wearing an income costume. |\n"
            f"| Is it bond-like? | **The opposite.** It lost **{R['amlp_dd']}%** peak-to-trough — "
            f"three times the market's {R['spy_dd']}% drawdown. Bonds don't do that. |\n"
            f"| Did the income make up for it? | **No.** Total return {R['amlp_cagr']:.1f}%/yr vs the "
            f"market's {R['spy_cagr']:.1f}%, at a third of the Sharpe ({R['amlp_sh']:.2f} vs "
            f"{R['spy_sh']:.2f}). |\n\n"
            "> The distribution is real cash — but it's mostly your own capital, sitting on top of a "
            "volatile energy bet. '7–8% income' is a **framing trick**: it points at the payout and "
            "away from the total return and the risk, which are what actually decide your outcome."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Pipeline MLPs are the toll-roads of energy. They earn fee-based cash flows that don't "
            "depend on the oil price, and pass it through as a high, stable distribution. Own the "
            "infrastructure, collect a 7–8% yield — bond-like income with a touch of inflation "
            "protection.\"*\n\n"
            "— the shared pitch behind the MLP / midstream-income category (AMLP, MLPA, AMZA …)\n\n"
            "There's a kernel of truth: pipelines really do collect fee-based, volume-driven revenue, "
            "and some of it is genuinely fee-based. The question is whether the *fund* delivers a "
            "bond-like income stream — or whether you're buying a leveraged energy bet and being "
            "handed back your own capital as 'yield'."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Billions of retirement dollars sit in MLP income products bought by people who hear "
            "'7% yield, toll-road of energy' and picture a safe bond substitute. If the 'income' is "
            "really their own capital being liquidated — on top of an asset that crashes with the oil "
            "price — they are spending down principal while believing it's intact, *and* carrying "
            "equity-grade (worse) crash risk they were told they'd avoided. AMLP fell ~50% in the "
            "2014–16 oil bust and ~60% in the March-2020 crash. Naming exactly *what* the yield is, "
            "and *what* you're actually exposed to, is the whole game."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three clean tests:\n\n"
            "1. **Split the 'income' from the NAV.** A fund's total return = what the share price "
            "(NAV) does **plus** the cash it pays out. If the NAV is *falling* while it pays a fat "
            "yield, the yield is just your own capital coming back — *return of capital*.\n"
            "2. **Measure what it really tracks.** Regress AMLP's monthly return on the energy sector "
            "(XLE). A bond-like income sleeve would barely move with energy (beta ≈ 0); a leveraged "
            "energy bet has a big, statistically solid beta.\n"
            "3. **Race it, and look at the crash.** Compare AMLP to just owning SPY (the market) and "
            "XLE (energy) on **total return**, and at the worst drawdown. Bond-like income should be "
            "calm; an energy bet won't be.\n\n"
            "Tape: monthly total returns for AMLP, MLPA, AMZA, SPY, XLE, USO (Yahoo)."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, where the 'yield' comes from.** For each fund, split the annual return into the "
            "NAV (price) and the distribution, and ask how much of that distribution was backed by "
            "the NAV actually growing:"
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL and not SPLIT.empty:\n"
            "    for f in ['AMLP','MLPA','AMZA']:\n"
            "        try:\n"
            "            both = pd.concat([SPLIT[(f,'price')], SPLIT[(f,'dist')]], axis=1).dropna()\n"
            "        except KeyError:\n"
            "            continue\n"
            "        ill = st.income_illusion(both.iloc[:,0], both.iloc[:,1])\n"
            "        rows.append((f, ill['dist_yield']*100, ill['price_cagr']*100, ill['return_of_capital_share']*100))\n"
            "    src = 'REAL TAPE'\n"
            "if not rows:\n"
            "    rows = [('AMLP',R['amlp_dist'],R['amlp_price'],R['amlp_roc']*100),\n"
            "            ('MLPA',R['mlpa_dist'],R['mlpa_price'],R['mlpa_roc']*100),\n"
            "            ('AMZA',R['amza_dist'],R['amza_price'],R['amza_roc']*100)]\n"
            "    src = 'frozen real numbers (docs/results.md)'\n"
            "dfi = pd.DataFrame(rows, columns=['fund','dist%','price_cagr%','roc%'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.6))\n"
            "x = np.arange(len(dfi))\n"
            "ax.bar(x, dfi['dist%'], width=.5, color=AMBER, label='distribution paid (the \"yield\")')\n"
            "ax.bar(x, dfi['price_cagr%'], width=.5, color=[GREEN if v>=0 else RED for v in dfi['price_cagr%']],\n"
            "       alpha=.9, label='what the NAV actually did')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels(dfi['fund'])\n"
            "ax.set_ylabel('%/yr'); ax.legend()\n"
            "ax.set_title(f'The \"income\" vs the NAV  ·  {src}')\n"
            "for xi, r_ in zip(x, dfi['roc%']):\n"
            "    ax.annotate(f'{r_:.0f}% is\\nreturn of capital', (xi, -15.5), ha='center', va='top', fontsize=8, color=GREY)\n"
            "ax.set_ylim(-20, max(dfi['dist%'])*1.15)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dfi.round(1).to_string(index=False))"
        ),
        md(
            f"The bar is brutal for all three: every MLP fund's **NAV shrank** while it paid a fat "
            f"distribution, so the **entire** payout was return of capital — your own money on a round "
            f"trip. AMZA, the leveraged one, paid a {R['amza_dist']:.0f}% 'yield' while its NAV bled "
            f"{abs(R['amza_price']):.0f}%/yr — a textbook capital-destruction machine. The 'income' is "
            "real cash, but it's principal and gains you already had, not yield created from thin air."
        ),
        md(
            "**Now, what are you really holding?** Regress each fund on the energy sector (XLE) and "
            "look at the beta — how much it moves for each 1% move in energy:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = []\n"
            "    for f in FUNDS:\n"
            "        sub = PANEL[[f,'XLE']].dropna()\n"
            "        if sub.empty: continue\n"
            "        eb = st.energy_beta(sub[f], sub['XLE'])\n"
            "        bb.append((f, eb['beta'], eb['beta_t'], eb['r2']))\n"
            "    src='REAL TAPE'\n"
            "else:\n"
            "    bb=[('AMLP',R['amlp_beta'],R['amlp_beta_t'],R['amlp_r2']),\n"
            "        ('MLPA',R['mlpa_beta'],R['mlpa_beta_t'],R['mlpa_r2']),\n"
            "        ('AMZA',R['amza_beta'],R['amza_beta_t'],R['amza_r2'])]\n"
            "    src='frozen real numbers (docs/results.md)'\n"
            "dfb = pd.DataFrame(bb, columns=['fund','beta','t','r2'])\n"
            "fig, ax = plt.subplots(figsize=(9.5,4.6))\n"
            "ax.bar(dfb['fund'], dfb['beta'], color=AMBER)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, r_ in dfb.iterrows():\n"
            "    ax.annotate(f\"t={r_['t']:.1f}\\nR²={r_['r2']:.2f}\", (i, r_['beta']+.03), ha='center', fontsize=9, color=GREY)\n"
            "ax.set_ylabel('beta to the energy sector (XLE)')\n"
            "ax.set_title(f'An \"income\" fund that is really an energy bet  ·  {src}')\n"
            "ax.set_ylim(0, max(dfb['beta'])*1.25)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dfb.round(2).to_string(index=False))"
        ),
        md(
            f"There's the costume coming off. AMLP moves **{R['amlp_beta']:.2f}** for every 1% in "
            f"energy, with a *t* of **{R['amlp_beta_t']:.1f}** — that's not a bond, it's a coiled "
            "energy trade. A genuine fee-based toll-road would barely flinch when oil moves; these "
            "funds lurch with it (and AMZA, leveraged, moves more than 1-for-1)."
        ),
        md(
            "**And the crash.** Bond-like income is supposed to be *calm*. Here's the worst "
            "peak-to-trough loss of AMLP vs just owning the market (SPY) and energy (XLE):"
        ),
        code(
            "if HAVE_REAL:\n"
            "    dd = []\n"
            "    for f,nm in [('AMLP','AMLP'),('SPY','SPY (market)'),('XLE','XLE (energy)')]:\n"
            "        s = PANEL[f].dropna(); dd.append((nm, st.max_drawdown(s)*100, st.cagr(s)*100))\n"
            "    src='REAL TAPE'\n"
            "else:\n"
            "    dd=[('AMLP',R['amlp_dd'],R['amlp_cagr']),('SPY (market)',R['spy_dd'],R['spy_cagr']),\n"
            "        ('XLE (energy)',R['xle_dd'],R['xle_cagr'])]\n"
            "    src='frozen real numbers (docs/results.md)'\n"
            "dfd = pd.DataFrame(dd, columns=['asset','maxdd%','cagr%'])\n"
            "fig, ax = plt.subplots(figsize=(9.5,4.6))\n"
            "col=[RED, GREEN, AMBER]\n"
            "ax.bar(dfd['asset'], dfd['maxdd%'], color=col)\n"
            "ax.set_ylabel('worst peak-to-trough drawdown (%)')\n"
            "ax.set_title(f'\"Bond-like income\" lost 3x the market  ·  {src}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dfd.round(1).to_string(index=False))"
        ),
        md(
            f"A 'bond-like' sleeve that loses **{abs(R['amlp_dd'])}%** is a contradiction in terms. "
            f"AMLP's drawdown is *deeper than the energy sector's own* and three times the market's — "
            "and over the full window it earned a third of the market's return at a third of the "
            "Sharpe. The income never came close to paying for the ride."
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** AMLP's energy exposure is unmistakable: beta {R['amlp_beta']:.2f} "
            f"to XLE with HAC *t* = **{R['amlp_beta_t']:.1f}** (CI {R['amlp_ci_lo']:.2f}–"
            f"{R['amlp_ci_hi']:.2f}), and the distribution is **{R['amlp_roc']*100:.0f}% return of "
            "capital**. What's real is exactly the *opposite* of the pitch: it's an energy bet that "
            "hands back your own money.\n"
            "- **Tradability — Mirage.** As a 'bond-like income' sleeve it's strictly worse than the "
            f"thing it's sold against: a {abs(R['amlp_dd'])}% drawdown, a third of the market's "
            "Sharpe, and a self-financed yield.\n"
            "- **Fat-yield free lunch? — Busted.** The yield isn't free and it isn't bond-like — "
            "it's leverage on oil with a return-of-capital coupon."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "You can absolutely *buy* AMLP — it's liquid. The question is what for. As a deliberate, "
            "sized **bullish bet on energy / midstream**, it's a legitimate (if expensive, and "
            "tax-quirky) instrument — just know that's what it is. As a 'safe high-yield bond "
            "substitute' for a retiree, it's the wrong tool by a mile: you'd be holding a leveraged "
            "oil trade that liquidates your principal to manufacture the headline yield, and that has "
            "historically halved in energy busts. The cash isn't fake; the **free lunch** — and the "
            "**bond-like safety** — are."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The option-overlay cousin.** [Study 337 — Covered-Call-ETF](../../337-covered-call-etf/): "
            "the *same* return-of-capital income illusion, engineered with call options instead of "
            "pipeline distributions.\n"
            "- **The single-name version.** [Study 57 — Yield-Trap](../../57-yield-trap/): "
            "high-dividend *stocks* on total return — the same 'the yield isn't free' lesson.\n"
            "- **The bond-costume cousin.** [Study 338 — Preferred-Stocks](../../338-preferred-stocks/): "
            "another 'bond-like' high-yield sleeve that turns out to carry equity-grade crash risk.\n"
            "- **Fork it.** Does midstream finally pay when energy *rises*? Re-run the race over only "
            "oil-bull sub-periods — the one regime where the leveraged-energy read could turn the "
            "story positive (and check whether the distribution still erodes the NAV even then).\n\n"
            "*Think the post-2021 'midstream renaissance' fixed it? Fork this, condition on the "
            "energy-up regime, and see whether the income finally stops being return of capital.*"
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
            "# Pipeline MLPs — a quantitative teardown 🔬\n"
            "### Energy-beta regression (HAC *t*) · SPY/XLE total-return race · return-of-capital "
            "decomposition · drawdown & capture · synthetic MLP replicator\n\n"
            "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Fat--yield_free_lunch%3F: Busted](https://img.shields.io/badge/Fat--yield_free_lunch%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We regress three MLP 'income' "
            "ETFs (AMLP, MLPA, AMZA) on the energy sector with an autocorrelation-robust *t*, decompose "
            "each fund's distribution into NAV vs return of capital, race them against SPY and XLE total "
            "return, and confirm the engine on a deterministic synthetic MLP replicator.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo monthly bars, total return "
            "(`auto_adjust`) for the regression & race + split-only price & dividends for the "
            "decomposition; as-of **2026-05-31** (partial month dropped); the offline core and tests "
            "run on a deterministic synthetic MLP world. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT + "\nfrom mlp_pipelines.strategy import energy_beta, race, capture, income_illusion, replicate_mlp\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | AMLP energy beta **{R['amlp_beta']:.2f}** to XLE, **HAC *t* = "
            f"{R['amlp_beta_t']:.2f}** (CI [{R['amlp_ci_lo']:.2f}, {R['amlp_ci_hi']:.2f}], R² "
            f"{R['amlp_r2']:.2f}); distribution **{R['amlp_roc']*100:.0f}% return of capital**. The "
            "energy-bet identity is real; so is the NAV erosion. |\n"
            f"| **Tradability** | `MIRAGE` | Sharpe {R['amlp_sh']:.2f} vs SPY {R['spy_sh']:.2f}; max "
            f"drawdown **{R['amlp_dd']}%** (deeper than XLE's {R['xle_dd']}% and 3× SPY's "
            f"{R['spy_dd']}%); CAGR {R['amlp_cagr']:.1f}% vs {R['spy_cagr']:.1f}%. |\n"
            f"| **Fat-yield free lunch?** | `BUSTED` | The 'bond-like 8% income' is a leveraged energy "
            f"bet (beta {R['amlp_beta']:.2f}) paying back your own NAV ({R['amlp_roc']*100:.0f}% ROC). |\n\n"
            "> 💡 In plain words: AMLP isn't a high-yield bond — it's an oil-beta sandwich. The "
            "distribution is your own capital; the price line is XLE with leverage and a bleed. The "
            "**energy beta is the real signal**, and it's the very thing the 'safe income' pitch denies."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "An MLP fund's monthly total return decomposes as\n\n"
            "$$r^{mlp}_t = \\beta\\, r^{energy}_t + \\delta + d_t + \\varepsilon_t,$$\n\n"
            "an energy-factor exposure $\\beta$, a structural NAV drift $\\delta$, the distribution "
            "$d_t$, and idiosyncratic noise. The marketed claim is $\\beta \\approx 0$ (fee-based, "
            "oil-insensitive), $\\delta \\approx 0$ (NAV preserved) and a high steady $d_t$ — i.e. "
            "bond-like income.\n\n"
            "- **H₁ (bond-like, oil-insensitive).** The energy beta $\\beta \\approx 0$.\n"
            "- **H₂ (the income is yield).** Distribution backed by NAV growth (return-of-capital "
            "share ≈ 0), not self-financed.\n"
            "- **H₃ (bond-like risk).** Drawdown and Sharpe comparable to a bond, not to equities.\n"
            "- **H₄ (free lunch).** Total return matches the market with the yield on top.\n\n"
            "We find **H₁ rejected** (β ≈ 0.74, HAC *t* ≈ 5.3), **H₂ rejected** (100% return of "
            "capital), **H₃ rejected** (−73% drawdown, Sharpe 0.34), and **H₄ rejected** (trails SPY "
            "by ~8%/yr)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If $\\beta$ is large and the ROC share is ~1, the product is mis-sold: it is an energy "
            "trade marketed as fixed income, financing its headline yield by liquidating principal. "
            "The structural drag is well-documented — the C-corp-wrapper tax accrual on a fund of "
            "MLPs (the post-2016 AMLP write-down), plus distribution coverage below 1× — but the "
            "interesting empirical content is whether, **net of the energy exposure and the NAV "
            "bleed**, the package behaves like the bond-substitute it is sold as. It does not; we put "
            "three live funds on the stand with HAC inference and the distribution-vs-NAV "
            "decomposition the marketing never shows."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Energy beta (the Signal).** Regress fund monthly TR on XLE TR; **OLS slope with a "
            "Newey-West HAC standard error** on the slope (so the *t* survives energy's volatility "
            "clustering), R², and a **block-bootstrap 95% CI** for β (block = 6 months).\n"
            "- **Decompose.** Split-only price (NAV) + dividend stream → distribution yield, price "
            "CAGR, and the **return-of-capital share** = the fraction of the distribution not backed "
            "by NAV growth.\n"
            "- **Race.** Fund vs SPY *and* XLE on aligned monthly TR (both fully invested → an honest "
            "excess-vs-excess comparison). Mean spread, HAC *t*, block-bootstrap CI, plus drawdown "
            "and up/down capture vs energy.\n"
            "- **Positive control.** A deterministic synthetic MLP replicator (β·energy + bleed + "
            "distribution); it must recover the planted β and read ~0 on the null.\n\n"
            "Three funds: AMLP, MLPA, AMZA. As-of 2026-05-31, partial month dropped, "
            "content-fingerprinted."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The energy beta — the inference-bar number for the Signal\n\n"
            "Regress each MLP fund on XLE total return; the HAC *t* on the slope is the decisive "
            "statistic. A bond-like income sleeve (H₁) must show β ≈ 0."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for f in FUNDS:\n"
            "        sub = PANEL[[f,'XLE']].dropna()\n"
            "        if sub.empty: continue\n"
            "        eb = energy_beta(sub[f], sub['XLE'])\n"
            "        recs.append((f, eb['n'], eb['beta'], eb['beta_t'], eb['ci_lo'], eb['ci_hi'], eb['r2']))\n"
            "    dft = pd.DataFrame(recs, columns=['fund','n','beta','HAC_t','ci_lo','ci_hi','r2'])\n"
            "    src='REAL TAPE'\n"
            "else:\n"
            "    dft = pd.DataFrame({'fund':FUNDS,'n':[R['amlp_n'],R['mlpa_n'],R['amza_n']],\n"
            "        'beta':[R['amlp_beta'],R['mlpa_beta'],R['amza_beta']],\n"
            "        'HAC_t':[R['amlp_beta_t'],R['mlpa_beta_t'],R['amza_beta_t']],\n"
            "        'ci_lo':[R['amlp_ci_lo'],0.53,0.74],'ci_hi':[R['amlp_ci_hi'],1.02,1.48],\n"
            "        'r2':[R['amlp_r2'],R['mlpa_r2'],R['amza_r2']]})\n"
            "    src='frozen real numbers (docs/results.md)'\n"
            "fig, ax = plt.subplots(figsize=(9.5,4.6))\n"
            "ax.bar(dft['fund'], dft['HAC_t'], color=[GREEN if abs(t)>=2 else RED for t in dft['HAC_t']])\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('HAC t-stat of the energy (XLE) beta')\n"
            "ax.set_title(f'The energy exposure is unmistakable — all three clear t=2  ·  {src}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dft.round(2).to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words: AMLP's beta to energy is {R['amlp_beta']:.2f} with HAC *t* = "
            f"**{R['amlp_beta_t']:.2f}** and a bootstrap CI wholly above zero ({R['amlp_ci_lo']:.2f}–"
            f"{R['amlp_ci_hi']:.2f}); energy alone explains {R['amlp_r2']*100:.0f}% of the variance. "
            "**H₁ (oil-insensitive) is decisively rejected** — this is the real, robust signal, and "
            "it is the opposite of the pitch."
        ),
        md(
            "### 4b · The income illusion — distribution vs NAV vs return of capital\n\n"
            "Split the distribution from the NAV and size how much of the 'yield' is self-financed."
        ),
        code(
            "rows=[]\n"
            "if HAVE_REAL and not SPLIT.empty:\n"
            "    for f in FUNDS:\n"
            "        try: both=pd.concat([SPLIT[(f,'price')],SPLIT[(f,'dist')]],axis=1).dropna()\n"
            "        except KeyError: continue\n"
            "        ill=income_illusion(both.iloc[:,0],both.iloc[:,1])\n"
            "        rows.append((f, ill['dist_yield']*100, ill['price_cagr']*100, ill['total_cagr']*100, ill['return_of_capital_share']))\n"
            "    src='REAL TAPE'\n"
            "if not rows:\n"
            "    rows=[('AMLP',R['amlp_dist'],R['amlp_price'],R['amlp_total'],R['amlp_roc']),\n"
            "          ('MLPA',R['mlpa_dist'],R['mlpa_price'],R['mlpa_total'],R['mlpa_roc']),\n"
            "          ('AMZA',R['amza_dist'],R['amza_price'],R['amza_total'],R['amza_roc'])]\n"
            "    src='frozen real numbers (docs/results.md)'\n"
            "dfi=pd.DataFrame(rows, columns=['fund','dist%','price_cagr%','total%','roc_share'])\n"
            "fig,ax=plt.subplots(figsize=(9.5,4.6))\n"
            "ax.bar(dfi['fund'], dfi['roc_share']*100, color=[RED if v>0.9 else AMBER for v in dfi['roc_share']])\n"
            "ax.axhline(100, ls='--', c=GREY); ax.set_ylim(0,110)\n"
            "ax.set_ylabel('% of the distribution that is RETURN OF CAPITAL')\n"
            "ax.set_title(f'The \"income\" is 100% your own money  ·  {src}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dfi.round(2).to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words: all three are **{R['amlp_roc']*100:.0f}% return of capital** — the "
            f"NAV fell while they paid double-digit-or-near distributions. AMZA paid {R['amza_dist']:.0f}% "
            f"while its NAV bled {abs(R['amza_price']):.0f}%/yr to a *negative* total return. **H₂ "
            "rejected.**"
        ),
        md(
            "### 4c · The race & the crash — bond-like risk? (H₃/H₄)\n\n"
            "Total-return spread vs SPY and XLE, the Sharpe gap, and the worst drawdown — the test of "
            "whether this is calm income or a leveraged energy bet."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs=[]\n"
            "    for f in FUNDS:\n"
            "        for b in ['SPY','XLE']:\n"
            "            sub=PANEL[[f,b]].dropna()\n"
            "            if sub.empty: continue\n"
            "            r=race(sub[f],sub[b])\n"
            "            recs.append((f, b, r['spread_ann_pct'], r['spread_t'], r['fund_sharpe'], r['bench_sharpe'], r['fund_maxdd']*100, r['bench_maxdd']*100))\n"
            "    dfr=pd.DataFrame(recs, columns=['fund','vs','spread%/yr','HAC_t','f_Sh','b_Sh','f_DD%','b_DD%'])\n"
            "    src='REAL TAPE'\n"
            "else:\n"
            "    dfr=pd.DataFrame({'fund':['AMLP','AMLP'],'vs':['SPY','XLE'],\n"
            "        'spread%/yr':[R['amlp_spread_spy'],R['amlp_spread_xle']],'HAC_t':[R['amlp_t_spy'],R['amlp_t_xle']],\n"
            "        'f_Sh':[R['amlp_sh'],R['amlp_sh']],'b_Sh':[R['spy_sh'],R['xle_sh']],\n"
            "        'f_DD%':[R['amlp_dd'],R['amlp_dd']],'b_DD%':[R['spy_dd'],R['xle_dd']]})\n"
            "    src='frozen real numbers (docs/results.md)'\n"
            "amlp = dfr[dfr['fund']=='AMLP']\n"
            "fig, axes = plt.subplots(1,2,figsize=(11,4.4))\n"
            "axes[0].bar(['AMLP','SPY','XLE'],[R['amlp_sh'],R['spy_sh'],R['xle_sh']] if not HAVE_REAL else\n"
            "            [st.sharpe(PANEL['AMLP'].dropna()), st.sharpe(PANEL['SPY'].dropna()), st.sharpe(PANEL['XLE'].dropna())],\n"
            "            color=[RED,GREEN,AMBER]); axes[0].set_title('Sharpe'); axes[0].set_ylabel('annualised Sharpe')\n"
            "axes[1].bar(['AMLP','SPY','XLE'],[R['amlp_dd'],R['spy_dd'],R['xle_dd']] if not HAVE_REAL else\n"
            "            [st.max_drawdown(PANEL['AMLP'].dropna())*100, st.max_drawdown(PANEL['SPY'].dropna())*100, st.max_drawdown(PANEL['XLE'].dropna())*100],\n"
            "            color=[RED,GREEN,AMBER]); axes[1].set_title('Worst drawdown'); axes[1].set_ylabel('max drawdown %')\n"
            "fig.suptitle(f'AMLP: a third of the market\\'s Sharpe, 3x the drawdown  ·  {src}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dfr.round(2).to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words: AMLP trails SPY by {abs(R['amlp_spread_spy']):.1f}%/yr and even XLE "
            f"by {abs(R['amlp_spread_xle']):.1f}%/yr (the spread *t* is sub-2 — the return give-up is "
            f"economically large but, on this single fund, not statistically *certain*), at a Sharpe of "
            f"{R['amlp_sh']:.2f} and a **{R['amlp_dd']}%** drawdown deeper than energy's own. **H₃/H₄ "
            "rejected:** the risk is worse than equities, not bond-like."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — the *energy-bet identity* is the robust signal: AMLP β = "
            f"{R['amlp_beta']:.2f} to XLE, HAC *t* = **{R['amlp_beta_t']:.2f}**, CI "
            f"[{R['amlp_ci_lo']:.2f}, {R['amlp_ci_hi']:.2f}], R² {R['amlp_r2']:.2f}; distribution "
            f"{R['amlp_roc']*100:.0f}% return of capital. (The total-return *underperformance* is "
            "large but its single-fund spread *t* is sub-2 — we stamp on the energy beta, which "
            "clears the bar emphatically, not on the spread.)\n"
            f"- **Tradability `MIRAGE`** — Sharpe {R['amlp_sh']:.2f} vs SPY {R['spy_sh']:.2f}, "
            f"{R['amlp_dd']}% drawdown, self-financed yield. A worse bond *and* a worse stock.\n"
            "- **Fat-yield free lunch? `BUSTED`** — H₁/H₂/H₃/H₄ all rejected; it's leverage on oil "
            "with a return-of-capital coupon."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the cost is the structure, not the spread\n\n"
            "The funds are liquid; the explicit fee is ~0.85%, but the real 'cost' is the NAV bleed "
            "(tax accrual + sub-1× distribution coverage) and the un-hedged energy beta — both already "
            "inside the total return and the drawdown above. There's no realistic cost knob that flips "
            "the verdict; the loss is structural. The synthetic below makes the mechanism explicit: "
            "with a planted bleed and a fat distribution, the return-of-capital share is 1 regardless "
            "of trading frictions."
        ),
        code(
            "# Synthetic: the distribution is return of capital whenever NAV drift <= 0, at any cost level.\n"
            "energy = data.synthetic_mlp(beta=0.0, dist=0.0, nav_drift=0.0, idio_vol=0.0, seed=341)[0]['energy_tr']\n"
            "drifts = np.linspace(0.002, -0.006, 25)\n"
            "roc = []\n"
            "for dft_ in drifts:\n"
            "    rep = replicate_mlp(energy, beta=0.9, dist=0.0065, nav_drift=dft_)\n"
            "    ill = st.income_illusion(rep['price'], rep['dist'])\n"
            "    roc.append(ill['return_of_capital_share']*100)\n"
            "fig,ax=plt.subplots(figsize=(9.0,4.3))\n"
            "ax.plot(drifts*100, roc, 'o-', c=AMBER, lw=2)\n"
            "ax.axhline(100, ls='--', c=GREY); ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('monthly NAV drift (%) — bleed to the right'); ax.set_ylabel('return-of-capital share (%)')\n"
            "ax.set_title('Once the NAV bleeds, the whole distribution is your own capital')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At NAV drift <= 0, ROC share saturates at 100% — exactly what the real funds show.')"
        ),
        md(
            "> 💡 In plain words: the moment the NAV stops growing, *every dollar of distribution* is "
            "principal coming back. The real funds sit on the right-hand (bleeding) end of this curve."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the energy-beta detector faithful, or does it always print a big β? Sweep the planted "
            "β on the replicator and watch the recovered HAC *t* track it — and the null (β = 0) reads "
            "flat."
        ),
        code(
            "energy = data.synthetic_mlp(beta=0.0, dist=0.0, nav_drift=0.0, idio_vol=0.0, seed=341)[0]['energy_tr']\n"
            "betas = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]\n"
            "ts, recov = [], []\n"
            "for b in betas:\n"
            "    panel,_ = data.synthetic_mlp(beta=b, dist=0.0065, nav_drift=-0.0035, seed=341)\n"
            "    eb = st.energy_beta(panel['mlp_tr'], panel['energy_tr'])\n"
            "    ts.append(eb['beta_t']); recov.append(eb['beta'])\n"
            "fig,ax=plt.subplots(figsize=(9.0,4.3))\n"
            "ax.plot(betas, recov, 'o-', c=GREEN, lw=2, label='recovered beta')\n"
            "ax.plot(betas, betas, ls='--', c=GREY, label='planted beta (truth)')\n"
            "ax.set_xlabel('planted energy beta'); ax.set_ylabel('recovered beta'); ax.legend()\n"
            "ax.set_title('The detector is faithful: recovered beta tracks the planted beta')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('HAC t by planted beta:', [round(t,1) for t in ts], '— ~0 at the null, large as beta grows.')"
        ),
        md(
            "The detector reads β ≈ 0 (and *t* ≈ 0) at the null and recovers the planted slope as it "
            "grows — so the real-tape β ≈ 0.74, *t* ≈ 5.3 is a statement about **AMLP**, not an "
            "artefact of the harness. For the option-overlay version of the same return-of-capital "
            "income illusion, see [Study 337 — Covered-Call-ETF](../../337-covered-call-etf/); for "
            "the high-dividend single-name version, [Study 57 — Yield-Trap](../../57-yield-trap/)."
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
