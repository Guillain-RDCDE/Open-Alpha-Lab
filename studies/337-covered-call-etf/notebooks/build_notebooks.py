"""Generate the two narrative notebooks for Study 337 (Covered-Call-ETF).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
buy-write figures run anywhere, offline and deterministic; the real-tape cells read the cached
monthly parquet under the shared _cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader, online or
off, and banners which tape each cell shows.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-05-31).
R = dict(
    # The race vs SPY total return (HAC t, bootstrap CI bps/mo, capture)
    qyld_cagr=8.1, qyld_t=-3.21, qyld_spread=-6.4, qyld_sh=0.76, qyld_up=0.60, qyld_dn=0.62, qyld_n=149,
    xyld_cagr=8.0, xyld_t=-3.84, xyld_spread=-7.3, xyld_sh=0.74, xyld_up=0.63, xyld_dn=0.71, xyld_n=155,
    ryld_cagr=5.0, ryld_t=-2.90, ryld_spread=-11.4, ryld_sh=0.40, ryld_up=0.57, ryld_dn=0.76, ryld_n=85,
    jepi_cagr=10.4, jepi_t=-2.08, jepi_spread=-8.5, jepi_sh=1.00, jepi_up=0.60, jepi_dn=0.62, jepi_n=72,
    jepq_cagr=17.7, jepq_t=-0.15, jepq_spread=-0.5, jepq_sh=1.18, jepq_up=0.93, jepq_dn=0.88, jepq_n=48,
    spy_cagr=14.0, spy_sh=0.95,
    # The income illusion: distribution vs NAV vs return-of-capital share
    qyld_dist=11.2, qyld_price=-2.7, qyld_roc=1.00,
    xyld_dist=8.0, xyld_price=0.1, xyld_roc=0.99,
    ryld_dist=12.3, ryld_price=-6.4, ryld_roc=1.00,
    jepi_dist=8.8, jepi_price=1.5, jepi_roc=0.83,
    jepq_dist=11.5, jepq_price=5.6, jepq_roc=0.52,
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

from covered_call_etf import data, strategy as st

AS_OF = pd.Timestamp("2026-05-31")
FUNDS = ["JEPI", "JEPQ", "QYLD", "XYLD", "RYLD"]

def load_real():
    panel = data.fetch_panel()      # cache-first, empty on miss
    split = data.fetch_price_and_dist()
    if not panel.empty:
        panel = panel[panel.index <= AS_OF]
    if not split.empty:
        split = split[split.index <= AS_OF]
    return panel, split

PANEL, SPLIT = load_real()
HAVE_REAL = (not PANEL.empty) and ("SPY" in PANEL.columns)
print("real monthly cache present:", HAVE_REAL,
      "" if not HAVE_REAL else f"({PANEL.index.min().date()}..{PANEL.index.max().date()})")
""")


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Covered-Call ETFs — is the 'income' free, or is it your own money? 💸\n"
            "### JEPI, QYLD and friends pay a fat monthly yield — but where does it come from?\n\n"
            "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Yield_without_giving_up_returns%3F: Busted](https://img.shields.io/badge/Yield_without_giving_up_returns%3F-Busted-8b949e?style=flat-square)\n\n"
            "Buy-write 'income' ETFs are everywhere: **JEPI**, **JEPQ**, **QYLD**, **XYLD**, **RYLD** "
            "— funds that hold the stock market, sell call options against it, and pay out an "
            "eye-catching **8–12% monthly yield**. The pitch is irresistible: *keep your equity "
            "returns, and collect a juicy income on top.* This notebook asks the only two questions "
            "that matter — **where does that 'income' actually come from, and do you really keep "
            "your returns?**\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the bootstrap CIs and "
            "the capture math? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Where does the 12% 'yield' come from? | **Your own money.** For QYLD and RYLD the "
            f"NAV *fell* ({R['qyld_price']:+.0f}% and {R['ryld_price']:+.0f}%/yr) while they paid "
            f"out double digits — so **100% of the distribution is return of capital**. |\n"
            f"| Do you keep your equity returns? | **No.** Every fund with a real track record "
            f"trailed **SPY total return** — QYLD by {R['qyld_spread']:.0f}%/yr, JEPI by "
            f"{R['jepi_spread']:.0f}%/yr — at a *lower* Sharpe. |\n"
            "| Why? | **Capped upside.** Writing calls keeps only ~60% of the up months but still "
            "takes ~62–76% of the down months. You sold your best months for a premium that didn't "
            "cover them. |\n"
            "| So what is the 'income'? | A **payout schedule**, not a free lunch. The cash is real; "
            "it's just (mostly) gains and principal you already owned, handed back to you. |\n\n"
            "> The funds aren't a scam — the cash is really paid. But '8–12% income' is a "
            "**framing trick**: it points at a payout and away from the total return, which is what "
            "actually decides whether you have more money at the end."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Earn a high, consistent monthly income from the stocks you already own. Our "
            "covered-call strategy generates premium income while you stay invested in the market — "
            "yield without giving up your equity returns.\"*\n\n"
            "— the shared marketing of the buy-write ETF category (JEPI, QYLD, XYLD, RYLD …)\n\n"
            "There's a real engine under the hood. Selling call options does collect a genuine "
            "premium (the *volatility risk premium* — index options are, on average, a bit "
            "over-priced). The question is whether that premium is **free money on top** of your "
            "returns, or whether you're **selling something** to get it."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "These funds hold **tens of billions of dollars**, much of it from retirees who hear "
            "'8% income' and picture a bond that pays like a stock. If the 'income' is really their "
            "own capital being liquidated, they are slowly spending down their nest egg while "
            "believing it's intact — and underperforming a plain index fund the whole way. That's "
            "not an academic distinction; it's the difference between a portfolio that lasts and one "
            "that quietly shrinks. Naming exactly *what* the yield is matters."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two clean tests:\n\n"
            "1. **Split the 'income' from the NAV.** A fund's total return = what the share price "
            "(NAV) does **plus** the cash it pays out. If the NAV is *falling* while it pays a fat "
            "yield, the yield is just your own capital coming back — *return of capital*.\n"
            "2. **Race it against just buying SPY.** Compare each fund to the S&P 500 on **total "
            "return** (dividends and distributions reinvested, fees included) over its whole life. "
            "If the pitch is true, the buy-write should roughly match SPY. If it gives up returns, "
            "it trails.\n\n"
            "Tape: monthly total returns for JEPI, JEPQ, QYLD, XYLD, RYLD and SPY (Yahoo)."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the punchline: where the 'yield' comes from.** For each fund, split the "
            "annual return into the NAV (price) and the distribution, and ask how much of that "
            "distribution was backed by the NAV actually growing:"
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL and not SPLIT.empty:\n"
            "    for f in ['QYLD','RYLD','XYLD','JEPI','JEPQ']:\n"
            "        try:\n"
            "            both = pd.concat([SPLIT[(f,'price')], SPLIT[(f,'dist')]], axis=1).dropna()\n"
            "        except KeyError:\n"
            "            continue\n"
            "        ill = st.income_illusion(both.iloc[:,0], both.iloc[:,1])\n"
            "        rows.append((f, ill['dist_yield']*100, ill['price_cagr']*100, ill['return_of_capital_share']*100))\n"
            "    src = 'REAL TAPE'\n"
            "if not rows:\n"
            "    rows = [('QYLD',R['qyld_dist'],R['qyld_price'],R['qyld_roc']*100),\n"
            "            ('RYLD',R['ryld_dist'],R['ryld_price'],R['ryld_roc']*100),\n"
            "            ('XYLD',R['xyld_dist'],R['xyld_price'],R['xyld_roc']*100),\n"
            "            ('JEPI',R['jepi_dist'],R['jepi_price'],R['jepi_roc']*100),\n"
            "            ('JEPQ',R['jepq_dist'],R['jepq_price'],R['jepq_roc']*100)]\n"
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
            "    ax.annotate(f'{r_:.0f}% is\\nreturn of capital', (xi, -8.5), ha='center', va='top', fontsize=8, color=GREY)\n"
            "ax.set_ylim(-13, max(dfi['dist%'])*1.15)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dfi.round(1).to_string(index=False))"
        ),
        md(
            f"For **QYLD** and **RYLD** the bar is brutal: they paid {R['qyld_dist']:.0f}% and "
            f"{R['ryld_dist']:.0f}% 'yields' while their NAV *shrank* — so the entire distribution "
            "was **return of capital**, your own money on a round trip. Even the better-run **JEPI** "
            f"is ~{R['jepi_roc']*100:.0f}% return of capital. The 'income' is real cash, but it's "
            "overwhelmingly principal and gains you already had, not yield created out of thin air."
        ),
        md(
            "**Now the race.** Forget the yield for a second — what matters is total return. Here's "
            "each fund vs just owning SPY (total return, fees and all), per year over its life:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rr = []\n"
            "    for f in ['JEPI','QYLD','XYLD','RYLD','JEPQ']:\n"
            "        sub = PANEL[[f,'SPY']].dropna()\n"
            "        if sub.empty: continue\n"
            "        r = st.race(sub[f], sub['SPY'])\n"
            "        rr.append((f, r['fund_cagr']*100, r['bench_cagr']*100))\n"
            "    src = 'REAL TAPE'\n"
            "else:\n"
            "    rr = [('JEPI',R['jepi_cagr'],18.0),('QYLD',R['qyld_cagr'],14.0),\n"
            "          ('XYLD',R['xyld_cagr'],14.7),('RYLD',R['ryld_cagr'],16.0),('JEPQ',R['jepq_cagr'],18.0)]\n"
            "    src = 'frozen real numbers (docs/results.md)'\n"
            "dfr = pd.DataFrame(rr, columns=['fund','fund_cagr','spy_cagr'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.6))\n"
            "x = np.arange(len(dfr)); w=.38\n"
            "ax.bar(x-w/2, dfr['fund_cagr'], w, color=AMBER, label='the buy-write fund')\n"
            "ax.bar(x+w/2, dfr['spy_cagr'], w, color=GREEN, label='just owning SPY')\n"
            "ax.set_xticks(x); ax.set_xticklabels(dfr['fund']); ax.set_ylabel('total return %/yr')\n"
            "ax.legend(); ax.set_title(f'Every fund with a track record trails SPY  ·  {src}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dfr.round(1).to_string(index=False))"
        ),
        md(
            f"Green beats amber almost everywhere. QYLD trailed SPY by ~{abs(R['qyld_spread']):.0f}%/yr, "
            f"JEPI by ~{abs(R['jepi_spread']):.0f}%/yr, RYLD by ~{abs(R['ryld_spread']):.0f}%/yr — and "
            "all at a *lower* Sharpe (more on that for the quants). The one near-tie, **JEPQ**, only "
            "exists because it was born into a roaring Nasdaq bull market that its options barely "
            "capped — a lucky 4-year window, not a vindicated design."
        ),
        md(
            "**Why does holding the market lose to the market?** Because covered calls **cap your "
            "upside**. Here's the up-vs-down capture of the systematic writers — how much of the "
            "good months and the bad months they keep:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cc = []\n"
            "    for f in ['QYLD','XYLD','RYLD','JEPI']:\n"
            "        sub = PANEL[[f,'SPY']].dropna()\n"
            "        if sub.empty: continue\n"
            "        c = st.capture(sub[f], sub['SPY'])\n"
            "        cc.append((f, c['up_capture'], c['down_capture']))\n"
            "    src='REAL TAPE'\n"
            "else:\n"
            "    cc=[('QYLD',R['qyld_up'],R['qyld_dn']),('XYLD',R['xyld_up'],R['xyld_dn']),\n"
            "        ('RYLD',R['ryld_up'],R['ryld_dn']),('JEPI',R['jepi_up'],R['jepi_dn'])]\n"
            "    src='frozen real numbers (docs/results.md)'\n"
            "dfc = pd.DataFrame(cc, columns=['fund','up','dn'])\n"
            "fig, ax = plt.subplots(figsize=(9.5,4.6))\n"
            "x=np.arange(len(dfc)); w=.38\n"
            "ax.bar(x-w/2, dfc['up']*100, w, color=GREEN, label='keeps of the UP months')\n"
            "ax.bar(x+w/2, dfc['dn']*100, w, color=RED, label='keeps of the DOWN months')\n"
            "ax.axhline(100, ls='--', c=GREY); ax.set_xticks(x); ax.set_xticklabels(dfc['fund'])\n"
            "ax.set_ylabel('% of the market move kept'); ax.legend()\n"
            "ax.set_title(f'Capped upside: keeps less of the good months than the bad  ·  {src}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dfc.round(2).to_string(index=False))"
        ),
        md(
            "There's the mechanism in one picture. You keep only ~60% of the rallies but ~62–76% "
            "of the sell-offs. The call premium you collect simply doesn't pay for the upside you "
            "hand away — so over a full cycle you fall behind."
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** Every fund with a real history trails SPY total return with a "
            f"strong statistical *t* (QYLD −3.2, XYLD −3.8, RYLD −2.9, JEPI −2.1), and the "
            "distribution is 83–100% return of capital. The under-performance and the income "
            "illusion are real, not noise.\n"
            "- **Tradability — Mirage.** As a way to *own equities* it's strictly worse: lower "
            "Sharpe, capped upside, and a 'yield' that's mostly self-financed.\n"
            "- **Yield without giving up returns? — Busted.** You give up the returns to "
            "*manufacture* the yield. The promise is precisely what the data refutes."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "You can absolutely *buy* these funds — they're liquid and cheap to trade. The question "
            "is whether you *should*, and for what. As a high-cash-flow sleeve in a taxable bucket "
            "where you genuinely want spendable cash and accept lower growth, a buy-write is a "
            "defensible (if expensive) tool. As a 'better way to own the market' or a retirement "
            "growth engine, it's the wrong instrument: you'd have ended up with more money — and "
            "could have *sold shares* for the same cash flow with more control — by just holding a "
            "low-cost index fund. The cash isn't fake; the **free lunch** is."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The sister study.** [Study 62 — Premium-Seller](../../62-premium-seller/) races "
            "QYLD against *its own underlying* (QQQ) and dissects the capture asymmetry — the "
            "same family, a different cut.\n"
            "- **The dividend cousin.** [Study 57 — Yield-Trap](../../57-yield-trap/): high-dividend "
            "*stocks* on total return — the same 'the yield isn't free' lesson on single names.\n"
            "- **Fork it.** Does the buy-write win in a flat, choppy decade (when the upside it "
            "caps never shows up)? Re-run the race over only sideways sub-periods and see whether "
            "the premium finally pays — the one regime where the pitch could be true.\n\n"
            "*Think JEPQ proves the new generation cracked it? Fork this, wait for the next bear "
            "market, and re-run the race — the design's whole claim is that it protects you then.*"
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
            "# Covered-Call ETFs — a quantitative teardown 🔬\n"
            "### SPY-total-return race · HAC *t* + block-bootstrap CI · return-of-capital "
            "decomposition · up/down capture · synthetic buy-write replicator\n\n"
            "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Yield_without_giving_up_returns%3F: Busted](https://img.shields.io/badge/Yield_without_giving_up_returns%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We race five buy-write ETFs "
            "(JEPI, JEPQ, QYLD, XYLD, RYLD) against SPY total return with an autocorrelation-robust "
            "*t* and a block-bootstrap CI, decompose each fund's distribution into NAV vs return of "
            "capital, and confirm the engine on a deterministic synthetic replicator.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo monthly bars, total return "
            "(`auto_adjust`) for the race + split-only price & dividends for the decomposition; "
            "as-of **2026-05-31** (partial month dropped); the offline core and tests run on a "
            "deterministic synthetic buy-write. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT + "\nfrom covered_call_etf.strategy import race, capture, income_illusion, replicate_buywrite\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | vs SPY TR: QYLD **HAC *t* = {R['qyld_t']:.2f}**, XYLD "
            f"{R['xyld_t']:.2f}, RYLD {R['ryld_t']:.2f}, JEPI {R['jepi_t']:.2f} — all with a "
            f"bootstrap CI wholly below 0; distribution {R['qyld_roc']*100:.0f}% / "
            f"{R['ryld_roc']*100:.0f}% return-of-capital for QYLD/RYLD. |\n"
            f"| **Tradability** | `MIRAGE` | Lower Sharpe ({R['qyld_sh']:.2f} vs SPY "
            f"{R['spy_sh']:.2f}), up-capture ~0.60 vs down-capture ~0.62–0.76 — capped upside "
            "not paid for by the premium. |\n"
            f"| **Yield without giving up returns?** | `BUSTED` | The promise is rejected: QYLD "
            f"trails SPY by {R['qyld_spread']:.1f}%/yr, JEPI by {R['jepi_spread']:.1f}%/yr; JEPQ's "
            f"tie (*t* = {R['jepq_t']:.2f}) is a 48-month QQQ-bull artefact (up-capture "
            f"{R['jepq_up']:.2f}). |\n\n"
            "> 💡 In plain words: the buy-write doesn't add yield to your returns — it converts "
            "returns *into* yield (and loses a bit in the conversion). The high distribution is "
            "mostly your own NAV; the total return lags the index it holds."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "A buy-write is long the index and short a one-month call, so its total return is\n\n"
            "$$r^{bw}_t = \\min(r^{idx}_t,\\,K) + p_t,$$\n\n"
            "the index capped at the strike $K$ plus the call premium $p_t$. The marketed claim is "
            "that the premium compensates for the cap, i.e. $\\mathbb{E}[r^{bw}] \\approx "
            "\\mathbb{E}[r^{idx}]$ *with* a high, steady payout.\n\n"
            "- **H₁ (no return give-up).** The monthly spread $r^{bw}_t - r^{spy}_t$ has mean ≥ 0.\n"
            "- **H₂ (the income is yield).** The distribution is backed by NAV growth (return-of-"
            "capital share ≈ 0), not self-financed.\n"
            "- **H₃ (risk-improved).** The buy-write Sharpe ≥ SPY's.\n"
            "- **H₄ (the new generation is different).** JEPI/JEPQ's OTM/ELN design escapes the cap.\n\n"
            "We find **H₁ rejected** (robustly negative spread), **H₂ rejected** (83–100% return of "
            "capital), **H₃ rejected** (lower Sharpe), and **H₄ unsupported** (JEPI still trails at "
            "*t* = −2.1; JEPQ's tie is a short, lucky window)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The volatility risk premium is real (Bakshi-Kapadia 2003), so *some* edge exists in "
            "selling index options — the interesting content is not whether the premium exists but "
            "whether, **net of the upside forfeited**, the package beats the index it holds. "
            "Israelov-Nielsen (2015) argued analytically that it does not; we put five live funds, "
            "including the newer actively-overlaid JEPI/JEPQ, on the stand with robust inference and "
            "the distribution-vs-NAV decomposition the marketing never shows."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Race.** Fund vs SPY on aligned monthly **total returns** (both fully invested → an "
            "honest excess-vs-excess comparison; no cash leg to mis-net). Mean spread, **Newey-West "
            "HAC *t***, **circular block-bootstrap 95% CI** (block = 6 months).\n"
            "- **Decompose.** Split-only price (NAV) + dividend stream → distribution yield, price "
            "CAGR, and the **return-of-capital share** = the fraction of the distribution not backed "
            "by NAV growth.\n"
            "- **Capture.** Up- and down-capture vs SPY, conditioned on the benchmark's sign.\n"
            "- **Positive control.** A deterministic synthetic buy-write replicator (cap + premium); "
            "it must recover the planted effect and read ~0 on the null.\n\n"
            "Five funds: JEPI, JEPQ, QYLD, XYLD, RYLD. As-of 2026-05-31, partial month dropped, "
            "content-fingerprinted."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The race — robust inference on the spread vs SPY total return\n\n"
            "The inference-bar number: the HAC *t* on the monthly return spread, per fund, with its "
            "bootstrap CI. A fund that 'gives yield without giving up returns' must **not** show a "
            "robustly negative spread."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for f in FUNDS:\n"
            "        sub = PANEL[[f,'SPY']].dropna()\n"
            "        if sub.empty: continue\n"
            "        r = race(sub[f], sub['SPY'])\n"
            "        recs.append((f, r['n'], r['spread_ann_pct'], r['spread_t'],\n"
            "                     r['spread_ci_lo_bps'], r['spread_ci_hi_bps'],\n"
            "                     r['fund_sharpe'], r['bench_sharpe']))\n"
            "    dft = pd.DataFrame(recs, columns=['fund','n','spread%/yr','HAC_t','ci_lo','ci_hi','f_Sh','spy_Sh'])\n"
            "    src='REAL TAPE'\n"
            "else:\n"
            "    dft = pd.DataFrame({'fund':FUNDS,\n"
            "        'n':[R['jepi_n'],R['jepq_n'],R['qyld_n'],R['xyld_n'],R['ryld_n']],\n"
            "        'spread%/yr':[R['jepi_spread'],R['jepq_spread'],R['qyld_spread'],R['xyld_spread'],R['ryld_spread']],\n"
            "        'HAC_t':[R['jepi_t'],R['jepq_t'],R['qyld_t'],R['xyld_t'],R['ryld_t']],\n"
            "        'ci_lo':[-122,-46,-77,-82,-145],'ci_hi':[-4,43,-20,-26,-27],\n"
            "        'f_Sh':[R['jepi_sh'],R['jepq_sh'],R['qyld_sh'],R['xyld_sh'],R['ryld_sh']],\n"
            "        'spy_Sh':[1.13,1.11,0.95,1.01,0.96]})\n"
            "    src='frozen real numbers (docs/results.md)'\n"
            "fig, ax = plt.subplots(figsize=(9.5,4.6))\n"
            "col=[GREEN if abs(t)<2 else RED for t in dft['HAC_t']]\n"
            "ax.bar(dft['fund'], dft['HAC_t'], color=col)\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('HAC t-stat of (fund − SPY) monthly spread')\n"
            "ax.set_title(f'Four of five trail SPY below t=-2  ·  {src}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dft.round(2).to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words: QYLD, XYLD, RYLD and JEPI all clear |*t*| = 2 on the **negative** "
            "side, with bootstrap CIs entirely below zero — the return give-up is statistically "
            f"real. Only JEPQ ties (*t* = {R['jepq_t']:.2f}), and its CI straddles zero on just 48 "
            "months. **H₁ rejected.**"
        ),
        md(
            "### 4b · The income illusion — distribution vs NAV vs return of capital\n\n"
            "Split the distribution from the NAV and size how much of the 'yield' is self-financed."
        ),
        code(
            "rows=[]\n"
            "if HAVE_REAL and not SPLIT.empty:\n"
            "    for f in ['JEPI','JEPQ','QYLD','XYLD','RYLD']:\n"
            "        try: both=pd.concat([SPLIT[(f,'price')],SPLIT[(f,'dist')]],axis=1).dropna()\n"
            "        except KeyError: continue\n"
            "        ill=income_illusion(both.iloc[:,0],both.iloc[:,1])\n"
            "        rows.append((f, ill['dist_yield']*100, ill['price_cagr']*100, ill['total_cagr']*100, ill['return_of_capital_share']))\n"
            "    src='REAL TAPE'\n"
            "if not rows:\n"
            "    rows=[('JEPI',R['jepi_dist'],R['jepi_price'],10.5,R['jepi_roc']),\n"
            "          ('JEPQ',R['jepq_dist'],R['jepq_price'],17.8,R['jepq_roc']),\n"
            "          ('QYLD',R['qyld_dist'],R['qyld_price'],8.2,R['qyld_roc']),\n"
            "          ('XYLD',R['xyld_dist'],R['xyld_price'],8.1,R['xyld_roc']),\n"
            "          ('RYLD',R['ryld_dist'],R['ryld_price'],5.1,R['ryld_roc'])]\n"
            "    src='frozen real numbers (docs/results.md)'\n"
            "dfi=pd.DataFrame(rows, columns=['fund','dist%','price_cagr%','total%','roc_share'])\n"
            "fig,ax=plt.subplots(figsize=(9.5,4.6))\n"
            "ax.bar(dfi['fund'], dfi['roc_share']*100, color=[RED if v>0.9 else AMBER for v in dfi['roc_share']])\n"
            "ax.axhline(100, ls='--', c=GREY); ax.set_ylim(0,110)\n"
            "ax.set_ylabel('% of the distribution that is RETURN OF CAPITAL')\n"
            "ax.set_title(f'The \"income\" is mostly your own money  ·  {src}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dfi.round(2).to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words: QYLD and RYLD are **{R['qyld_roc']*100:.0f}% / "
            f"{R['ryld_roc']*100:.0f}% return of capital** — the NAV fell while they paid double "
            f"digits. Even JEPI is {R['jepi_roc']*100:.0f}%. JEPQ's {R['jepq_roc']*100:.0f}% is the "
            "lowest only because a +5.6%/yr QQQ NAV did the lifting — the overlay didn't. "
            "**H₂ rejected.**"
        ),
        md(
            "### 4c · Up/down capture — the capped-upside signature\n\n"
            "The structural reason the spread is negative: the calls shave the up months harder "
            "than the premium cushions the down months."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cc=[]\n"
            "    for f in FUNDS:\n"
            "        sub=PANEL[[f,'SPY']].dropna()\n"
            "        if sub.empty: continue\n"
            "        c=capture(sub[f],sub['SPY']); cc.append((f,c['up_capture'],c['down_capture']))\n"
            "    dfc=pd.DataFrame(cc,columns=['fund','up','dn']); src='REAL TAPE'\n"
            "else:\n"
            "    dfc=pd.DataFrame({'fund':FUNDS,'up':[R['jepi_up'],R['jepq_up'],R['qyld_up'],R['xyld_up'],R['ryld_up']],\n"
            "        'dn':[R['jepi_dn'],R['jepq_dn'],R['qyld_dn'],R['xyld_dn'],R['ryld_dn']]}); src='frozen real numbers (docs/results.md)'\n"
            "fig,ax=plt.subplots(figsize=(9.5,4.6))\n"
            "ax.scatter(dfc['dn'],dfc['up'],s=120,color=AMBER,zorder=3)\n"
            "for _,r_ in dfc.iterrows(): ax.annotate(r_['fund'],(r_['dn'],r_['up']),xytext=(6,4),textcoords='offset points')\n"
            "lim=[0.3,1.05]; ax.plot(lim,lim,ls='--',c=GREY,label='symmetric (up=down)')\n"
            "ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_xlabel('down-capture (kept of DOWN months)')\n"
            "ax.set_ylabel('up-capture (kept of UP months)'); ax.legend()\n"
            "ax.set_title(f'Below the line = keeps less upside than downside  ·  {src}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dfc.round(2).to_string(index=False))"
        ),
        md(
            "> 💡 In plain words: every systematic writer sits **below** the diagonal — up-capture "
            "< down-capture, the asymmetry that guarantees long-run under-performance in a rising "
            "market. **H₃ rejected** (the Sharpe table in 4a confirms the risk isn't improved "
            "either). JEPQ is the only point near the line — its OTM overlay barely bit, in a "
            "one-directional QQQ tape."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — vs SPY TR: QYLD *t* {R['qyld_t']:.2f}, XYLD {R['xyld_t']:.2f}, "
            f"RYLD {R['ryld_t']:.2f}, JEPI {R['jepi_t']:.2f}; bootstrap CIs wholly below 0; "
            f"distribution {R['qyld_roc']*100:.0f}–{int(R['xyld_roc']*100)}% return of capital for the "
            "systematic writers.\n"
            f"- **Tradability `MIRAGE`** — lower Sharpe ({R['qyld_sh']:.2f} vs {R['spy_sh']:.2f}), "
            "capped upside, self-financed 'income'. A worse way to own equities.\n"
            f"- **Yield without giving up returns? `BUSTED`** — H₁/H₂/H₃ all rejected; JEPQ's "
            f"tie (*t* = {R['jepq_t']:.2f}, up-capture {R['jepq_up']:.2f}) is a 48-month bull-market "
            "artefact, not evidence H₄ holds."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the cost is the cap, not the fee\n\n"
            "The funds are liquid and the *explicit* fee is small; the real 'cost' is the forfeited "
            "upside, already inside the total-return spread above. The break-even framing: for the "
            "buy-write to match SPY, the premium would have to fully fund the cap — which the "
            "capture asymmetry shows it does not. There is no realistic cost knob that flips the "
            "verdict; the loss is structural, not frictional."
        ),
        code(
            "# Synthetic break-even: how much premium per month would a +1.8%/mo cap need to match the index?\n"
            "idx = data.synthetic_buywrite(cap=np.inf, premium=0.0, seed=337)[0]['index_tr']\n"
            "prems = np.linspace(0.0, 0.012, 25)\n"
            "spreads = []\n"
            "for p in prems:\n"
            "    rep = replicate_buywrite(idx, cap=0.018, premium=p)\n"
            "    spreads.append((rep['total_gross'].mean() - idx.mean())*1e4)\n"
            "be = prems[np.argmin(np.abs(spreads))]*100\n"
            "fig,ax=plt.subplots(figsize=(9.0,4.3))\n"
            "ax.plot(prems*100, spreads, 'o-', c=AMBER, lw=2)\n"
            "ax.axhline(0,c='k',lw=1); ax.axvline(be,ls='--',c=GREY)\n"
            "ax.set_xlabel('monthly call premium collected (%)'); ax.set_ylabel('buy-write − index (bps/mo)')\n"
            "ax.set_title(f'Synthetic break-even: a +1.8%/mo cap needs ~{be:.2f}%/mo premium to match')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Break-even premium ~{be:.2f}%/mo; real at-the-money index call premia rarely sustain that net of the cap.')"
        ),
        md(
            "> 💡 In plain words: the synthetic shows exactly how much premium the cap would have to "
            "earn to break even — and real index-option premia, net of the upside given away in "
            "trending markets, don't get there. The cap is the cost, and it's not negotiable."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print 'loses'? Sweep the planted "
            "cap tightness on the replicator and watch the spread vs the index move from ~0 (no cap) "
            "to deeply negative (tight cap) — and the null (no cap, no premium) is a dead heat."
        ),
        code(
            "caps = [0.10, 0.05, 0.03, 0.025, 0.02, 0.015]\n"
            "idx = data.synthetic_buywrite(cap=np.inf, premium=0.0, seed=337)[0]['index_tr']\n"
            "ts=[]\n"
            "for c in caps:\n"
            "    rep = replicate_buywrite(idx, cap=c, premium=0.0065)\n"
            "    _, t = st.newey_west_t((rep['total_gross']-idx).to_numpy())\n"
            "    ts.append(t)\n"
            "fig,ax=plt.subplots(figsize=(9.0,4.3))\n"
            "ax.plot([c*100 for c in caps], ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(-2, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.invert_xaxis(); ax.set_xlabel('monthly cap strike (%) — tighter to the right')\n"
            "ax.set_ylabel('HAC t of (buy-write − index)')\n"
            "ax.set_title('The engine is a faithful detector: tighter cap → more negative spread')\n"
            "plt.tight_layout(); plt.show()\n"
            "# null check\n"
            "rep0 = replicate_buywrite(idx, cap=np.inf, premium=0.0)\n"
            "print('NULL (no cap, no premium): spread =', round(float((rep0['total_gross']-idx).abs().max()),12), '(a dead heat)')"
        ),
        md(
            "The detector reads ~0 with a loose cap and falls past *t* = −2 as the cap tightens — so "
            "the real-tape result is a statement about **the funds**, not an artefact of the "
            "harness. For the capture-asymmetry cut of the same effect on QYLD-vs-QQQ, see "
            "[Study 62 — Premium-Seller](../../62-premium-seller/); for the single-name dividend "
            "version, [Study 57 — Yield-Trap](../../57-yield-trap/)."
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
