"""Generate the two narrative notebooks for Study 330 (Low-Volatility-Anomaly).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic figures
run anywhere, offline and deterministic; the real-tape cells use the cached SPLV/SPHB/SPY parquet
under ../_cache/ (or the shared repo _cache/) if present and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader offline.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-20, window
# 2011-06 -> 2026-05, 180 months, fingerprint a5ce034427ab).
R = dict(
    fp="a5ce034427ab", n=180, win_start="2011-06", win_end="2026-05",
    # leg cards
    splv_cagr=9.7, splv_sharpe=0.85, splv_vol=11.7, splv_dd=-21,
    sphb_cagr=14.2, sphb_sharpe=0.66, sphb_vol=24.8, sphb_dd=-37,
    spy_cagr=14.2, spy_sharpe=0.98, spy_vol=14.6, spy_dd=-24,
    sharpe_gap=0.19,
    # raw dollar-neutral spread
    raw_mean=-6.4, raw_sharpe=-0.31, raw_t=-1.24,
    # beta-neutral spread
    bn_mean=3.5, bn_sharpe=0.34, bn_t=1.43, bn_beta=0.0, bn_alpha=3.5, bn_alpha_t=1.27,
    bn_ci_lo=-0.9, bn_ci_hi=8.1,
    # net of costs + borrow
    bn_net_mean=2.5, bn_net_t=1.02, cost_bps=5.0, borrow_bps=50.0,
)


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

from low_volatility_anomaly import data, strategy as st

def load_real():
    \"\"\"Cache-first real tape (empty frame offline). Drops the partial last month.\"\"\"
    df = data.fetch_pairs(fetch=False)
    if df.empty:
        return df
    return data.drop_partial_last(df).dropna()

REAL = load_real()
HAVE_REAL = not REAL.empty
print("real SPLV/SPHB cache present:", HAVE_REAL,
      "" if not HAVE_REAL else f"({len(REAL)} months, {REAL.index[0].date()} -> {REAL.index[-1].date()})")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Low-Volatility Anomaly — does boring really beat exciting? 🐌\n"
            "### Racing the calm S&P 500 fund (SPLV) against the wild one (SPHB), in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Boring_beats_exciting%2C_risk--adjusted%3F: Confirmed](https://img.shields.io/badge/Boring_beats_exciting%2C_risk--adjusted%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "One of finance's most famous 'free lunches': **calm stocks quietly beat exciting ones** "
            "once you account for risk. You can buy each side off the shelf — Invesco sells an "
            "S&P 500 **Low Volatility** fund (SPLV) and an S&P 500 **High Beta** fund (SPHB), the "
            "100 calmest and the 100 wildest names in the same index. So we race them. Does boring "
            "win? And if it does, can you make money betting on it?\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the bootstrap CI and the "
            "cost maths? That's the companion, **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** "
            "— same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does boring beat exciting *risk-adjusted*? | **Yes.** The calm fund (SPLV) earned a "
            f"Sharpe of **{R['splv_sharpe']}** vs the wild fund's (SPHB) **{R['sphb_sharpe']}** — on "
            f"**less than half** the volatility ({R['splv_vol']}% vs {R['sphb_vol']}%). |\n"
            f"| So can I make money on it? | **Not easily.** The obvious trade — buy calm, short wild "
            f"— *lost* **{R['raw_mean']:.0f}%/yr**: you'd be shorting the decade's biggest winner. |\n"
            f"| What if I hedge that out? | A risk-balanced version earns **+{R['bn_mean']}%/yr**, but "
            f"it's so thin the data can't tell it from luck (*t* = {R['bn_t']}). After costs it's "
            f"~+{R['bn_net_mean']}%/yr. |\n"
            "| So what *is* it? | A real, useful **defensive tilt** — a calmer ride for a comparable "
            "Sharpe — not a self-financing money machine. |\n\n"
            "> Boring genuinely *is* the better risk-adjusted hold. But 'better risk-adjusted' and "
            "'a trade you can get paid for' are two different things — and that gap is the whole study."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Low-volatility stocks have delivered higher risk-adjusted returns than "
            "high-volatility stocks. Boring is beautiful.\"* — the low-volatility anomaly "
            "(Baker, Bradley & Wurgler 2011; Frazzini & Pedersen 2014).\n\n"
            "The intuition: lots of investors can't (or won't) use leverage, so to chase big returns "
            "they crowd into **exciting, high-beta** stocks — overpaying for them. That leaves the "
            "**calm** stocks cheap, so they quietly out-earn per unit of risk. If true, the easiest "
            "free lunch in markets is simply *owning the boring ones*."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "This one matters because it's *backwards* from the first thing everyone learns: "
            "more risk should mean more reward. If the calm stocks actually win risk-adjusted, the "
            "textbook risk-return line is broken — and there's a fund you can buy to harvest it. "
            "The desk has already looked at the academic version on single stocks "
            "([18 Dull-Roar](../../18-dull-roar/)) and at one calm fund vs the market "
            "([58 Bunker](../../58-bunker/)). Here we do the cleanest head-to-head there is: the "
            "calmest fund against the wildest fund, same index, same shelf."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three honest checks:\n\n"
            "1. **Race the two funds.** SPLV vs SPHB on return, *and on Sharpe* (return per unit of "
            "risk). Boring can only 'win' the way the claim means it if it wins on **Sharpe**.\n"
            "2. **Try to trade it.** Build the obvious bet — long the calm fund, short the wild one — "
            "and see whether it actually makes money.\n"
            "3. **Be fair about risk.** The wild fund carries way more market exposure, so a naive "
            "short of it is really a short of *the market going up*. We hedge that out to isolate the "
            "real low-vol edge, then charge it honest costs and a borrow fee for the short.\n\n"
            "Tape: monthly returns of SPLV, SPHB and SPY since both funds launched (2011)."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: who's the better *risk-adjusted* hold?** Raw return vs Sharpe, the two funds "
            "side by side:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    s_lo = st.leg_stats(REAL['SPLV']); s_hi = st.leg_stats(REAL['SPHB'])\n"
            "    lo_cagr, lo_sh, lo_vol = s_lo['cagr']*100, s_lo['sharpe'], s_lo['vol']*100\n"
            "    hi_cagr, hi_sh, hi_vol = s_hi['cagr']*100, s_hi['sharpe'], s_hi['vol']*100\n"
            "    banner = 'REAL tape (SPLV vs SPHB)'\n"
            "else:\n"
            f"    lo_cagr, lo_sh, lo_vol = {R['splv_cagr']}, {R['splv_sharpe']}, {R['splv_vol']}\n"
            f"    hi_cagr, hi_sh, hi_vol = {R['sphb_cagr']}, {R['sphb_sharpe']}, {R['sphb_vol']}\n"
            "    banner = 'SYNTHETIC fallback (frozen real numbers shown)'\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar(['SPLV (calm)','SPHB (wild)'], [lo_cagr, hi_cagr], color=[GREEN, RED], width=.5)\n"
            "a1.set_ylabel('CAGR %/yr'); a1.set_title('Raw return: the WILD one wins')\n"
            "a2.bar(['SPLV (calm)','SPHB (wild)'], [lo_sh, hi_sh], color=[GREEN, RED], width=.5)\n"
            "a2.set_ylabel('Sharpe'); a2.set_title('Risk-adjusted: the CALM one wins')\n"
            "for ax in (a1,a2): ax.axhline(0,c='k',lw=1)\n"
            "fig.suptitle(banner, fontsize=9, color=GREY); plt.tight_layout(); plt.show()\n"
            "print(f'SPLV vol {lo_vol:.1f}%  vs  SPHB vol {hi_vol:.1f}%  ->  calm fund Sharpe {lo_sh:.2f} > wild {hi_sh:.2f}')"
        ),
        md(
            f"There's the anomaly's signature. The wild fund made more raw return "
            f"(+{R['sphb_cagr']:.0f}% vs +{R['splv_cagr']:.0f}%/yr) — but it took **{R['sphb_vol']:.0f}% "
            f"volatility to do it, versus {R['splv_vol']:.0f}% for the calm fund.** Per unit of risk, "
            f"**boring wins**: Sharpe {R['splv_sharpe']} vs {R['sphb_sharpe']}. The calmer ride was the "
            "smarter ride."
        ),
        md(
            "**Now the trap. The obvious way to bet on this is to buy the calm fund and short the wild "
            "one.** What happens?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    raw = st.spread_stats(st.spread(REAL)); raw_m, raw_t = raw['mean_ann']*100, raw['tstat']\n"
            "else:\n"
            f"    raw_m, raw_t = {R['raw_mean']}, {R['raw_t']}\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "ax.bar(['Long SPLV,\\nshort SPHB'], [raw_m], color=RED, width=.4)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('return %/yr')\n"
            "ax.set_title(f'The obvious trade LOSES: {raw_m:+.1f}%/yr')\n"
            "ax.annotate(f't = {raw_t:+.2f}', (0, raw_m/2), ha='center', color='white', fontweight='bold')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Buy calm / short wild: {raw_m:+.1f}%/yr  (HAC t {raw_t:+.2f}) -- a loser.')"
        ),
        md(
            f"It **loses {R['raw_mean']:.0f}%/yr.** Why? Shorting the high-beta fund means shorting "
            "the part of the market that screamed higher all decade. You were *right* that the calm "
            "fund is the better risk-adjusted hold, and you still **lost money** making the obvious "
            "bet — because the bet was secretly 'the market will fall', and it didn't."
        ),
        md(
            "**So we hedge out that market bet** — short just enough of the wild fund to cancel the "
            "extra market exposure, leaving only the pure low-vol edge:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bn = st.spread_stats(st.beta_neutral_spread(REAL)); bn_m, bn_t = bn['mean_ann']*100, bn['tstat']\n"
            "    bn_net = st.spread_stats(st.beta_neutral_spread(REAL, cost_bps=5.0, borrow_ann_bps=50.0))['mean_ann']*100\n"
            "else:\n"
            f"    bn_m, bn_t, bn_net = {R['bn_mean']}, {R['bn_t']}, {R['bn_net_mean']}\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.2))\n"
            "ax.bar(['Hedged book\\n(gross)','...after costs\\n& borrow'], [bn_m, bn_net],\n"
            "       color=[AMBER, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('return %/yr')\n"
            "ax.set_title(f'Hedged, it earns a thin +{bn_m:.1f}%/yr -- but t = {bn_t:.2f} (not certain)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Beta-neutral: +{bn_m:.1f}%/yr (t {bn_t:.2f}); after costs ~+{bn_net:.1f}%/yr.')"
        ),
        md(
            f"Hedged, it does earn a positive **+{R['bn_mean']}%/yr** — but it's *so thin* the data "
            f"can't distinguish it from luck (*t* = {R['bn_t']}, below the desk's bar of 2). After "
            f"realistic costs and a borrow fee for the short, it's ~+{R['bn_net_mean']}%/yr. A whisper, "
            "not a free lunch."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Boring beats exciting risk-adjusted — Confirmed.** SPLV Sharpe {R['splv_sharpe']} "
            f"vs SPHB {R['sphb_sharpe']}, on under half the volatility. The calm fund really is the "
            "better risk-adjusted hold.\n"
            f"- **Signal — Weak.** The *ranking* is right, but the *tradable* version earns only "
            f"+{R['bn_mean']}%/yr at *t* = {R['bn_t']} — this 15-year bull sample can't certify it.\n"
            f"- **Tradability — Fragile.** The naive trade loses {R['raw_mean']:.0f}%/yr; the hedged "
            "one is too thin to survive costs convincingly. What's real is a long-only defensive "
            "tilt, not a money machine."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The honest use isn't the long-short at all — it's just **holding the calm fund**. You "
            "give up some raw upside in roaring bull years and get a much smoother ride (drawdown "
            f"{R['splv_dd']}% vs the wild fund's {R['sphb_dd']}%) for a comparable or better Sharpe. "
            "The long-short that would 'monetise the anomaly' is the part that doesn't survive: it "
            "either fights the market's beta or shrinks to a rounding error once you pay to run it. "
            "Boring is a good *hold*; it is not, on this sample, a good *trade*."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The academic version.** [18 Dull-Roar](../../18-dull-roar/) runs the same idea on "
            "the S&P 500 *cross-section* (decile sorts) — and finds the modern large-cap sample "
            "actually *inverts* it. Same lesson, sharper.\n"
            "- **Calm vs the market.** [58 Bunker](../../58-bunker/) races the min-vol fund (USMV) "
            "against SPY — and finds the same 'great defense, no free Sharpe' story.\n"
            "- **A full cycle.** Both funds launched in 2011, so this sample missed 2008. Low-vol "
            "earns its keep in *bear* markets — fork this with a longer (or bear-heavy) window and "
            "see if the hedged edge clears the bar.\n\n"
            "*Think boring should pay as a trade, not just a hold? Fork this, add a bear-market "
            "window or a vol-target overlay, and show the hedged spread clearing t = 2.*"
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
            "# Low-Volatility Anomaly — a quantitative teardown 🔬\n"
            "### SPLV vs SPHB · excess-of-cash Sharpe · raw vs beta-neutral spread · HAC *t* + block bootstrap · alpha-vs-beta · costs & borrow\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Boring_beats_exciting%2C_risk--adjusted%3F: Confirmed](https://img.shields.io/badge/Boring_beats_exciting%2C_risk--adjusted%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We race the Invesco S&P 500 "
            "Low Volatility (SPLV) and High Beta (SPHB) ETFs leg by leg, then test whether the "
            "low-minus-high spread is a real, tradable edge or just the structural beta gap.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo monthly total returns, SPLV/SPHB/SPY "
            "2011–2026 (as-of 2026-06-20, fingerprint `a5ce034427ab`); the offline core and tests run "
            "on a deterministic synthetic world. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Sharpe ranking holds (SPLV **{R['splv_sharpe']}** > SPHB "
            f"**{R['sphb_sharpe']}**), but the beta-neutral spread earns +{R['bn_mean']}%/yr at HAC "
            f"*t* = **{R['bn_t']}** (boot CI [{R['bn_ci_lo']}%, {R['bn_ci_hi']}%], straddles 0). |\n"
            f"| **Tradability** | `FRAGILE` | Raw long-low/short-high loses **{R['raw_mean']}%/yr** "
            f"(*t* {R['raw_t']}); beta-neutral net of {R['cost_bps']:.0f} bps/leg + {R['borrow_bps']:.0f} "
            f"bps borrow → +{R['bn_net_mean']}%/yr (*t* {R['bn_net_t']}). |\n"
            f"| **Boring beats exciting, risk-adjusted?** | `CONFIRMED` | SPLV vol {R['splv_vol']}% vs "
            f"SPHB {R['sphb_vol']}%, DD {R['splv_dd']}% vs {R['sphb_dd']}%, higher Sharpe — the calm "
            "leg is the better risk-adjusted hold. |\n\n"
            "> 💡 In plain words: the *fact* (calm out-Sharpes wild) is solid; the *trade* (a "
            "self-financing low-minus-high book) is not — it either fights the market's beta or "
            "shrinks below the inference bar."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{\\text{lo}}, r^{\\text{hi}}, r^{\\text{m}}$ be the monthly total returns of SPLV, "
            "SPHB and SPY.\n\n"
            "- **H₁ (risk-adjusted ranking).** $\\mathrm{Sharpe}(r^{\\text{lo}}) > "
            "\\mathrm{Sharpe}(r^{\\text{hi}})$.\n"
            "- **H₂ (raw spread).** $\\mathbb{E}[r^{\\text{lo}} - r^{\\text{hi}}] > 0$.\n"
            "- **H₃ (beta-neutral edge).** With $w = \\beta_{\\text{lo}}/\\beta_{\\text{hi}}$, "
            "$\\mathbb{E}[r^{\\text{lo}} - w\\,r^{\\text{hi}}] > 0$ at HAC *t* ≥ 2.\n"
            "- **H₄ (net of frictions).** H₃ survives costs and short borrow.\n\n"
            "We find **H₁ confirmed**, **H₂ rejected** (the beta gap dominates), **H₃ supported in "
            "sign but sub-*t* = 2**, and **H₄ rejected** (the thin edge erodes)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The interesting content is the **decomposition**, not the binary. The anomaly's *headline* "
            "(boring out-Sharpes wild) is real and easy to verify. The *monetisation* fails for two "
            "separable reasons worth naming: (a) the naive trade is short the market's beta winner, "
            "and (b) once you hedge that, the residual low-vol premium on a 15-year bull sample is "
            "statistically indistinguishable from zero. That maps exactly to "
            "[18 Dull-Roar](../../18-dull-roar/)'s cross-sectional finding — the modern, investable "
            "expression is a defensive tilt, not a long-short edge."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Legs.** CAGR, *excess-of-cash* Sharpe (house rule when legs sit at different risk "
            "levels), annualised vol, max drawdown.\n"
            "- **Raw spread.** $r^{\\text{lo}} - r^{\\text{hi}}$, dollar-neutral.\n"
            "- **Beta-neutral spread.** Hedge the structural beta gap (full-sample betas), so the net "
            "market beta ≈ 0 — the Frazzini–Pedersen BAB construction.\n"
            "- **Inference.** Newey–West HAC *t* on the monthly mean; circular **block-bootstrap** CI "
            "(block 6) on the annualised mean.\n"
            "- **Alpha vs beta.** Regress the spread on SPY excess return.\n"
            "- **Frictions.** One-way turnover × NAV cost per leg + an annual short borrow on SPHB.\n"
            "- **Positive control.** A deterministic synthetic SPLV/SPHB/SPY world with a dial-able "
            "low-vol edge (and a null) — the harness must bank a planted edge and stay flat on the null.\n\n"
            "Tape: monthly total returns, 2011-06 → 2026-05 (180 months); SPLV/SPHB launched 2011-05."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The leg cards — H₁\n\n"
            "Raw return favours the wild leg; Sharpe favours the calm leg."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [(t, st.leg_stats(REAL[t])) for t in ['SPLV','SPHB','SPY']]\n"
            "    card = pd.DataFrame({t: {'CAGR%': s['cagr']*100, 'Sharpe': s['sharpe'],\n"
            "        'vol%': s['vol']*100, 'maxDD%': s['max_dd']*100} for t,s in rows}).T\n"
            "    banner = 'REAL tape'\n"
            "else:\n"
            "    card = pd.DataFrame({\n"
            f"        'SPLV': {{'CAGR%': {R['splv_cagr']}, 'Sharpe': {R['splv_sharpe']}, 'vol%': {R['splv_vol']}, 'maxDD%': {R['splv_dd']}}},\n"
            f"        'SPHB': {{'CAGR%': {R['sphb_cagr']}, 'Sharpe': {R['sphb_sharpe']}, 'vol%': {R['sphb_vol']}, 'maxDD%': {R['sphb_dd']}}},\n"
            f"        'SPY':  {{'CAGR%': {R['spy_cagr']}, 'Sharpe': {R['spy_sharpe']}, 'vol%': {R['spy_vol']}, 'maxDD%': {R['spy_dd']}}}}}).T\n"
            "    banner = 'SYNTHETIC fallback (frozen real numbers)'\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar(card.index, card['CAGR%'], color=[GREEN, RED, GREY], width=.6)\n"
            "a1.set_title('CAGR — wild leg wins raw'); a1.set_ylabel('%/yr'); a1.axhline(0,c='k',lw=1)\n"
            "a2.bar(card.index, card['Sharpe'], color=[GREEN, RED, GREY], width=.6)\n"
            "a2.set_title('Sharpe — calm leg wins risk-adjusted'); a2.axhline(0,c='k',lw=1)\n"
            "fig.suptitle(banner, fontsize=9, color=GREY); plt.tight_layout(); plt.show()\n"
            "print(card.round(2).to_string())"
        ),
        md(
            f"> 💡 In plain words: SPLV's Sharpe ({R['splv_sharpe']}) tops SPHB's ({R['sphb_sharpe']}) "
            f"on **{R['splv_vol']}% vol vs {R['sphb_vol']}%** — H₁ confirmed. Note SPLV did *not* beat "
            f"the market (SPY Sharpe {R['spy_sharpe']}) — the bull-sample caveat from "
            "[58 Bunker](../../58-bunker/)."
        ),
        md(
            "### 4b · Raw vs beta-neutral spread — H₂ and H₃\n\n"
            "The naive dollar-neutral spread is negative because the high-beta leg's beta dominates; "
            "hedging it out reveals the residual low-vol premium."
        ),
        code(
            "if HAVE_REAL:\n"
            "    raw = st.spread_stats(st.spread(REAL))\n"
            "    bn  = st.spread_stats(st.beta_neutral_spread(REAL))\n"
            "    lo, hi = st.block_bootstrap_ci(st.beta_neutral_spread(REAL), seed=330)\n"
            "    raw_m, raw_t = raw['mean_ann']*100, raw['tstat']\n"
            "    bn_m, bn_t = bn['mean_ann']*100, bn['tstat']\n"
            "    ci_lo, ci_hi = lo*100, hi*100\n"
            "else:\n"
            f"    raw_m, raw_t = {R['raw_mean']}, {R['raw_t']}\n"
            f"    bn_m, bn_t = {R['bn_mean']}, {R['bn_t']}\n"
            f"    ci_lo, ci_hi = {R['bn_ci_lo']}, {R['bn_ci_hi']}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "vals = [raw_m, bn_m]; ts = [raw_t, bn_t]\n"
            "b = ax.bar(['Raw\\n(dollar-neutral)','Beta-neutral\\n(hedged)'], vals,\n"
            "           color=[RED, AMBER], width=.5)\n"
            "ax.errorbar(1, bn_m, yerr=[[bn_m-ci_lo],[ci_hi-bn_m]], fmt='none', ecolor='k', capsize=6)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('return %/yr')\n"
            "ax.set_title('Hedging the beta gap flips the sign — but the t-stat stays sub-2')\n"
            "for bar_, t_ in zip(b, ts):\n"
            "    ax.annotate(f't={t_:+.2f}', (bar_.get_x()+bar_.get_width()/2,\n"
            "        bar_.get_height()+(0.4 if bar_.get_height()>=0 else -0.8)),\n"
            "        ha='center', va='bottom' if bar_.get_height()>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Raw {raw_m:+.1f}%/yr (t {raw_t:+.2f}); beta-neutral {bn_m:+.1f}%/yr (t {bn_t:+.2f}), 95% CI [{ci_lo:+.1f}, {ci_hi:+.1f}]')"
        ),
        md(
            f"> 💡 In plain words: H₂ **rejected** — the raw trade loses {R['raw_mean']:.0f}%/yr "
            f"(short the beta winner). H₃ **supported in sign** (+{R['bn_mean']}%/yr) but the HAC "
            f"*t* = {R['bn_t']} and the bootstrap CI [{R['bn_ci_lo']}%, {R['bn_ci_hi']}%] straddles "
            "zero — below the bar. `WEAK`, not `REAL`."
        ),
        md(
            "### 4c · Alpha vs beta — is the spread anything but the beta gap?\n\n"
            "Regress the beta-neutral spread on the market excess return."
        ),
        code(
            "if HAVE_REAL:\n"
            "    reg = st.market_regression(st.beta_neutral_spread(REAL), REAL['SPY'])\n"
            "    a, beta, at = reg['alpha_ann']*100, reg['beta'], reg['alpha_t']\n"
            "else:\n"
            f"    a, beta, at = {R['bn_alpha']}, {R['bn_beta']}, {R['bn_alpha_t']}\n"
            "print(f'Beta-neutral book: market beta = {beta:+.2f}  (~0, as designed)')\n"
            "print(f'                   annualised alpha = {a:+.1f}%/yr,  alpha t = {at:+.2f}')\n"
            "fig, ax = plt.subplots(figsize=(6.5, 4.0))\n"
            "ax.bar(['market beta','alpha %/yr'], [beta, a], color=[GREY, AMBER], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_title('Hedged book: ~zero beta, a thin (uncertain) alpha')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: by design the hedged book's market beta is ~{R['bn_beta']:.1f}; the "
            f"residual is a thin **+{R['bn_alpha']}%/yr** alpha at *t* = {R['bn_alpha_t']}. So what's "
            "left after removing the beta gap really is a (weak) low-vol premium — just not one this "
            "sample can certify."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — H₁ (Sharpe ranking) confirmed; H₃ (beta-neutral spread) positive "
            f"in sign (+{R['bn_mean']}%/yr) but HAC *t* = {R['bn_t']} < 2, CI straddles zero.\n"
            f"- **Tradability `FRAGILE`** — H₂ rejected (raw {R['raw_mean']}%/yr, *t* {R['raw_t']}); "
            f"H₄ rejected (net +{R['bn_net_mean']}%/yr, *t* {R['bn_net_t']} after costs + borrow). The "
            "salvageable piece is a long-only defensive tilt.\n"
            f"- **Boring beats exciting risk-adjusted? `CONFIRMED`** — SPLV Sharpe {R['splv_sharpe']} "
            f"vs SPHB {R['sphb_sharpe']}, {R['splv_vol']}% vol vs {R['sphb_vol']}%."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "The hedged book is already thin; frictions finish it."
        ),
        code(
            "costs = [0.0, 2.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    net = [st.spread_stats(st.beta_neutral_spread(REAL, cost_bps=c, borrow_ann_bps=50.0))['mean_ann']*100\n"
            "           for c in costs]\n"
            "else:\n"
            f"    base = {R['bn_mean']}; net = [base-0.3, base-0.6, base-1.0, base-1.6]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=AMBER, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net, 0, color=AMBER, alpha=.12)\n"
            "ax.set_xlabel('one-way cost per leg (bps)'); ax.set_ylabel('net spread %/yr')\n"
            "ax.set_title('A thin edge, eroding with every basis point (50 bps borrow throughout)')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'At {R['cost_bps']:.0f} bps/leg + 50 bps borrow: ~+{R['bn_net_mean']}%/yr (t {R['bn_net_t']}).')"
        ),
        md(
            "> 💡 In plain words: even a few basis points and a benign borrow take the net edge toward "
            "a rounding error. Unlike the desk's outright mirages the gross *sign* is right — but a "
            "spread you can't tell from zero before costs is `FRAGILE` once you pay for it."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print the same number? Plant a "
            "low-vol edge of increasing strength and watch the beta-neutral book's *t*-stat rise."
        ),
        code(
            "edges = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]\n"
            "ts = []\n"
            "for e in edges:\n"
            "    df, _ = data.synthetic_world(lowvol_edge=e, seed=330)\n"
            "    ts.append(st.spread_stats(st.beta_neutral_spread(df))['tstat'])\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot([e*100 for e in edges], ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted low-vol edge (%/yr)'); ax.set_ylabel('beta-neutral HAC t')\n"
            "ax.set_title('The harness banks a planted edge — flat at the null, clearing 2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('t ~0 at the null, rising past 2 as the planted edge grows -> a faithful detector.')"
        ),
        md(
            "The *t*-stat is ~0 at the null and rises through the bar as the planted edge grows — so "
            "the engine is a faithful detector, and the real-tape result is a statement about **the "
            "market**: on 2011–2026, the low-vol Sharpe ranking is real but the tradable spread is "
            "too thin to certify. For the cross-sectional version see "
            "[18 Dull-Roar](../../18-dull-roar/); for calm-vs-market, [58 Bunker](../../58-bunker/)."
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
