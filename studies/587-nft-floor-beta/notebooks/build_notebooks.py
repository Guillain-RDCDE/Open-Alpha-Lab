"""Generate the two narrative notebooks for Study 587 (NFT-Floor-Beta).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The whole study is a
deterministic, offline synthetic world — there is NO real tape (no clean free NFT floor data
exists), so every cell runs anywhere, and the frozen numbers in ``R`` mirror docs/results.md.
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30; synthetic null world,
# seed 587, 1500 days, series fp 01ef0bf578c2, lead_alpha = 0).
R = dict(
    fp="01ef0bf578c2", n_days=1500, seed=587, lookback=14, horizon=5,
    beta=1.61, beta_t=49.7, beta_corr=0.79, llc_peak_lag=-1, llc_peak=0.79,
    hac_t=-0.47, ols_t=-0.87, slope=-0.095, r2=0.0005, hac_t_ctrl=0.41, placebo_p=0.73,
    overlay_gross=1.4, overlay_net=-3.1, overlay_sharpe=-0.05, overlay_turn=0.124, frac_long=49.4,
    # sweep: (horizon, hac_t_raw, hac_t_ctrl)
    sweep=[(1, 0.22, 0.54), (3, 0.04, 0.92), (5, -0.47, 0.41), (10, -0.58, 0.89), (21, 0.11, 1.42)],
    # control: (lead_alpha, mean hac_t over 25 seeds)
    ctrl=[(0.0, -0.17), (0.05, 1.03), (0.10, 2.30), (0.20, 5.19), (0.30, 8.84)],
)


BOOT = """\
import sys, os, warnings
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
warnings.filterwarnings("ignore")
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from nft_floor_beta import data, strategy as st

SEED, LOOKBACK, HORIZON = 587, 14, 5

# The whole study is a deterministic OFFLINE synthetic world (no real NFT floor tape exists).
# NULL world: lead_alpha = 0 -> floors are a lagged high-beta echo of ETH.
DF, TRUTH = data.synthetic_series(lead_alpha=0.0, seed=SEED)
FP = data.fingerprint(DF[["eth_ret", "floor_ret"]])
print("synthetic null world:", len(DF), "days | series fp", FP, "| lead_alpha", TRUTH.lead_alpha)
print("(SYNTHETIC-ONLY: no free NFT floor tape exists -> signal capped below REAL)")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# NFT Floor-Beta — do NFT prices *lead* crypto, or just *follow* ETH? 🖼️\n"
            "### Whether 'blue-chip NFT floors are running hot' tells you anything about where ETH goes next\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![NFT_floor_leads_crypto%3F: Busted](https://img.shields.io/badge/NFT_floor_leads_crypto%3F-Busted-8b949e?style=flat-square)\n\n"
            "You'll hear it in every crypto cycle: *\"watch the NFT floors — when Bored Apes and Punks "
            "run, ETH is about to run.\"* The idea is that NFT **floor prices** (the cheapest one you "
            "can buy in a collection) are a thermometer for **risk appetite** — degen money flows into "
            "jpegs first, then into the rest of crypto. So floor **momentum** should *lead* ETH.\n\n"
            "It sounds smart. But there's a catch that kills it before we start: NFT floors are priced "
            "**in ETH**. When ETH goes up, a floor quoted in ETH tends to go up too — mechanically. So "
            "are floors *leading* ETH, or just *echoing* it a day late with extra noise? We build a "
            "clean synthetic world to find out.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Synthetic-only & not investment advice.** There is *no* clean, free feed of NFT "
            "floor prices (the APIs are locked down, wash-traded, and full of dead collections), so we "
            "test the *logic* on a deterministic synthetic world. Every chart is drawn by the code "
            "beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do NFT floors *lead* ETH? | **No — they follow.** Floors move a day *after* ETH and "
            f"amplify it ~{R['beta']}×. Their strongest link is to *yesterday's* ETH, not tomorrow's. |\n"
            f"| Does floor momentum predict *future* ETH? | **No.** The predictive *t* is **{R['hac_t']}** "
            f"(a shuffle test says *p* = {R['placebo_p']}) — indistinguishable from noise. |\n"
            f"| Could you trade it? | **No.** A 'buy ETH when floors run' rule is **+{R['overlay_gross']}%/yr "
            f"gross → {R['overlay_net']}%/yr net** after trading costs. |\n"
            "| So the 'floors lead crypto' idea is…? | **Busted.** It's a high-beta echo, not a "
            "leading indicator. |\n\n"
            "> Floors *feel* like they lead because they're jumpy and they move with ETH — but jumpy + "
            "moves-with-ETH is exactly what a **lagged, high-beta echo** looks like."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"Blue-chip NFT floor-price momentum is a risk-appetite signal that leads crypto "
            "returns.\"*\n\n"
            "The intuition: NFTs are the frothiest, most speculative corner of crypto. If risk appetite "
            "is rising, the theory goes, it shows up in jpeg floors *first* — degens ape into Punks "
            "before the money rotates into ETH and the rest of the market. So a rising floor today "
            "should forecast a rising ETH tomorrow."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If true, you'd have a free leading indicator for the whole crypto complex — watch a handful "
            "of floors, front-run ETH. That's a big deal. But there's an obvious confound: floors are "
            "**denominated in ETH**. A floor of '30 ETH' rising to '33 ETH' could just be… ETH doing "
            "nothing while the jpeg re-rates, *or* it could be ETH itself moving. We have to separate "
            "**leading** from **echoing**."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Build the floor's relationship to ETH.** Is the floor mostly *yesterday's* ETH, "
            "amplified? (That's 'follows'.) Or does it move *before* ETH? (That's 'leads'.)\n"
            "2. **Test the prediction directly.** Take today's floor momentum and ask: does it forecast "
            "ETH's return over the *next* few days? Put a proper *t*-stat on it.\n"
            "3. **Try to trade it.** Buy ETH when floor momentum is positive; count the money after "
            "trading costs.\n\n"
            "Catch we're honest about: there's **no clean free NFT floor data**, so we test the logic on "
            "a synthetic world where floors are *built* to be a lagged echo of ETH — and check whether "
            "the engine can still be *fooled* into seeing a lead. (It can't — and if we plant a real "
            "lead, it finds it.)\n\n"
            "*Timing:* the floor-momentum signal uses only data up to today's close; we act at that "
            "close and measure ETH's *forward* return. No peeking."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: does the floor move *before* ETH, or *after* it?** We line up the floor return "
            "against ETH's return at different day-offsets and measure the correlation."
        ),
        code(
            "llc = st.lead_lag_corr(DF, max_lag=5)\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "cols = [GREEN if k>0 else (RED if k<0 else GREY) for k in llc.index]\n"
            "ax.bar(llc.index, llc.values, color=cols, width=.7)\n"
            "ax.axvline(0, c='k', lw=1); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('offset k:  corr(floor today, ETH k days later)')\n"
            "ax.set_ylabel('correlation')\n"
            "ax.set_title('The spike is at k = -1: floors track YESTERDAY\\'s ETH (they FOLLOW)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('peak correlation at offset k =', int(llc.idxmax()), '->', round(llc.max(),3))\n"
            "print('(negative k = floors follow ETH; positive k = floors lead ETH)')"
        ),
        md(
            f"There it is. The correlation spikes at **k = {R['llc_peak_lag']}** (≈ {R['llc_peak']}): the "
            "floor is most correlated with **yesterday's** ETH move, and essentially *zero*-correlated "
            "with ETH's *future* moves. Floors **follow**. And they follow with a *high beta* — a 1% ETH "
            f"move yesterday maps to roughly a {R['beta']}% floor move today. That's the whole "
            "'signal' in floors: lagged, amplified ETH."
        ),
        md(
            "**Second: the real test.** Forget the echo — does today's floor *momentum* actually "
            "**predict ETH's next-5-day return**? If floors lead, this should be strongly positive."
        ),
        code(
            "mom = st.floor_momentum(DF, LOOKBACK); fwd = st.forward_return(DF, HORIZON)\n"
            "reg = st.predictive_regression(mom, fwd)\n"
            "p = st.placebo_pvalue(mom, fwd, n_perm=300, seed=SEED)\n"
            "d = pd.DataFrame({'mom': mom, 'fwd': fwd}).dropna()\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.6))\n"
            "ax.scatter(d['mom']*100, d['fwd']*100, s=8, color=GREY, alpha=.5)\n"
            "xs = np.linspace(d['mom'].min(), d['mom'].max(), 50)\n"
            "b0 = d['fwd'].mean() - reg['slope']*d['mom'].mean()\n"
            "ax.plot(xs*100, (reg['slope']*xs + b0)*100, c=RED, lw=2)\n"
            "ax.set_xlabel('floor momentum today (%)'); ax.set_ylabel('ETH return next 5 days (%)')\n"
            "ax.set_title(f'No lead: predictive t = {reg[\"hac_t\"]:+.2f} (placebo p = {p:.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'predictive HAC-t {reg[\"hac_t\"]:+.2f} | R^2 {reg[\"r2\"]:.4f} | placebo p {p:.2f}')"
        ),
        md(
            f"A flat, shapeless cloud. The predictive *t* is **{R['hac_t']}** — nowhere near the +2 "
            f"you'd need to call it a signal, and the wrong sign anyway — with R² essentially **zero** "
            f"({R['r2']}). The shuffle (placebo) test agrees: *p* = **{R['placebo_p']}**, i.e. random "
            "noise reproduces this 'edge' three-quarters of the time. Floor momentum tells you nothing "
            "about where ETH goes next."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Floor momentum doesn't predict forward ETH (*t* {R['hac_t']}, placebo "
            f"*p* {R['placebo_p']}); floors just echo ETH a day late. (And with no real NFT tape, we "
            "can't earn a REAL stamp anyway.)\n"
            "- **Tradability — Mirage.** 'Buy ETH when floors run' is positive gross, **negative net** "
            "after costs. It's just timed partial ETH beta.\n"
            "- **NFT floor leads crypto? — Busted.** Floors *follow* ETH (high-beta, one-day lag), they "
            "do not lead it."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No. Even the *gross* version of 'buy ETH when floor momentum is positive' is barely above "
            "zero — because it's really just holding ETH about half the time, chosen by a coin-flip "
            "dressed up as a floor signal. After you pay to flip in and out, it's underwater. And that's "
            "in a *clean* synthetic world; the real NFT floor tape is worse — wash-traded, thin, and "
            "riddled with collections that quietly died."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Why 'synthetic-only'?** Because there is no free, honest, no-key feed of blue-chip NFT "
            "floor prices — the marketplace APIs are locked, rate-limited, wash-traded and "
            "survivorship-ridden. Same trap as the desk's [Lego](../../273-lego-returns/), "
            "[Whisky](../../275-whisky-cask/) and [Sneaker](../../276-sneaker-resale/) alt-data "
            "studies.\n"
            "- **The engine is honest.** In [the quants notebook](02_for_the_quants.ipynb) we *plant* a "
            "real lead and watch the detector light up past *t* = 2 — so the flat result here is the "
            "economics, not a broken test.\n\n"
            "*Think NFT floors really do lead crypto? Fork this, wire in a real (de-washed, "
            "survivorship-free) floor tape, and show floor momentum forecasting forward ETH at t ≥ 2.*"
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
            "# NFT Floor-Beta — a quantitative teardown 🔬\n"
            "### floor-momentum signal · predictive HAC-*t* · lagged-beta & lead-lag decomposition · ETH-momentum control · block-shuffle placebo · horizon sweep · gross/net overlay · seed-robust synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![NFT_floor_leads_crypto%3F: Busted](https://img.shields.io/badge/NFT_floor_leads_crypto%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We ask whether NFT floor "
            "momentum forecasts forward ETH (a *lead*) or merely echoes it (a lagged high-beta "
            "*follow*).\n\n"
            "> ⚠️ **Synthetic-only & not investment advice.** No clean, free NFT floor tape exists "
            "(key-gated, wash-traded, survivorship-ridden APIs), so this runs on a deterministic "
            f"synthetic world (seed 587, {R['n_days']} days, series fp `{R['fp']}`, `lead_alpha = 0`). "
            "A synthetic-only study is **capped below REAL** on data availability. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Predictive HAC-*t* of floor momentum on forward ETH = "
            f"**{R['hac_t']}** (placebo *p* {R['placebo_p']}, R² {R['r2']}); **+{R['hac_t_ctrl']}** "
            "controlling for ETH's own momentum. Synthetic-only → below REAL. |\n"
            f"| **Tradability** | `MIRAGE` | Floor-momentum long/flat ETH overlay: gross "
            f"**+{R['overlay_gross']}%/yr** → net **{R['overlay_net']}%/yr** (10 bps × turnover), "
            f"Sharpe **{R['overlay_sharpe']}**. |\n"
            f"| **NFT floor leads crypto?** | `BUSTED` | Floor beta on *lagged* ETH = **{R['beta']}** "
            f"(*t* {R['beta_t']}), lead-lag corr peaks at lag **{R['llc_peak_lag']}** — floors follow, "
            "not lead. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it catches a planted "
            "lead), but a lagged high-beta echo of ETH carries no forward information about ETH."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $m_t$ be the NFT floor-index momentum (a trailing mean of floor returns) and $r_{t\\to "
            "t+h}$ the forward ETH return.\n\n"
            "- **H₁ (the lead).** In $r_{t\\to t+h} = \\alpha + \\beta\\,m_t + \\varepsilon$, the slope "
            "$\\beta > 0$ at HAC-*t* ≥ 2 (floor momentum forecasts forward ETH).\n"
            "- **H₂ (beyond ETH's own momentum).** H₁ survives adding ETH's own trailing momentum as a "
            "control.\n"
            "- **H₃ (lead not follow).** The lead-lag cross-correlation peaks at a *positive* lag "
            "(floors lead), not a negative one (floors follow).\n"
            "- **H₄ (net).** A floor-momentum ETH overlay pays after costs.\n\n"
            "We find **H₁ rejected** (HAC-*t* < 0), **H₂ rejected** (still < 1 with the control), **H₃ "
            "rejected** (peak at lag −1, floors *follow*), **H₄ rejected** (net negative)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why**. NFT floors are **ETH-denominated**, thin, and reflexive, so a "
            "floor index mechanically inherits ETH's beta and reacts *after* ETH clears. The empirical "
            "literature agrees: **Ante (2022)** finds ETH shocks drive NFT activity (not the reverse), "
            "and **Dowling (2022)** finds volatility spilling *from* crypto *into* NFTs. So the sceptical "
            "prior — floors follow, they don't lead — is exactly what we plant in the null world and "
            "then confirm the engine reads correctly."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $m_t$ = 14-day trailing mean of NFT floor-index returns (known at close $t$).\n"
            "- **Target.** Forward ETH return over $(t, t+h]$.\n"
            "- **Inference.** OLS of target on $m_t$ with a **Newey-West HAC** *t* (Bartlett weights, "
            "rule-of-thumb lag) — overlapping forward windows autocorrelate the residuals, so a plain "
            "*t* would overstate significance.\n"
            "- **Follow decomposition.** Beta of floor on *lagged* ETH; lead-lag cross-correlation.\n"
            "- **Control.** Re-run the predictive regression adding ETH's own trailing momentum.\n"
            "- **Placebo.** Circular-shift the signal against the target ($n$ = 500) and read the "
            "HAC-*t* tail.\n"
            "- **Frictions.** Long/flat ETH overlay, one-way cost × turnover (short variant pays "
            "borrow).\n"
            "- **Positive control.** Plant a genuine lead (`lead_alpha > 0`) and average the HAC-*t* "
            "over 25 seeds (house rule).\n\n"
            "Timing / execution lag: $m_t$ uses floor data through $t$ only; the position is entered at "
            "that close and earns the forward window. No future data enters the signal; a one-bar-later "
            "entry changes no stamp."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · Follow, not lead — H₃\n\n"
            "Beta of the floor on *lagged* ETH (the high-beta echo) and the lead-lag cross-correlation."
        ),
        code(
            "beta = st.contemporaneous_beta(DF, lag=1)\n"
            "llc = st.lead_lag_corr(DF, max_lag=5)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "d = DF[['eth_ret','floor_ret']].dropna().copy(); d['ethlag']=d['eth_ret'].shift(1); d=d.dropna()\n"
            "a1.scatter(d['ethlag']*100, d['floor_ret']*100, s=8, color=GREY, alpha=.5)\n"
            "xs=np.linspace(d['ethlag'].min(), d['ethlag'].max(),50)\n"
            "a1.plot(xs*100, (beta['beta']*xs)*100, c=RED, lw=2)\n"
            "a1.set_xlabel('ETH return yesterday (%)'); a1.set_ylabel('floor return today (%)')\n"
            "a1.set_title(f'Floor = {beta[\"beta\"]:.2f} x lagged ETH (t {beta[\"t\"]:.0f}, corr {beta[\"corr\"]:.2f})')\n"
            "cols=[GREEN if k>0 else (RED if k<0 else GREY) for k in llc.index]\n"
            "a2.bar(llc.index, llc.values, color=cols, width=.7); a2.axvline(0,c='k',lw=1); a2.axhline(0,c='k',lw=.8)\n"
            "a2.set_xlabel('offset k: corr(floor_t, eth_{t+k})'); a2.set_ylabel('correlation')\n"
            "a2.set_title('Peak at k=-1: floors FOLLOW ETH')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'lagged beta {beta[\"beta\"]:.2f} (t {beta[\"t\"]:.1f}) | lead-lag peak at k={int(llc.idxmax())} ({llc.max():.2f})')"
        ),
        md(
            f"> 💡 In plain words: H₃ **rejected.** The floor is a **{R['beta']}× high-beta echo of "
            f"yesterday's ETH** (*t* {R['beta_t']}, corr {R['beta_corr']}), and the cross-correlation "
            f"peaks at lag **{R['llc_peak_lag']}** — floors *follow* ETH, they don't lead it."
        ),
        md(
            "### 4b · The predictive test — H₁ & H₂\n\n"
            "Does floor momentum forecast forward ETH — raw, and controlling for ETH's own momentum?"
        ),
        code(
            "mom = st.floor_momentum(DF, LOOKBACK); fwd = st.forward_return(DF, HORIZON)\n"
            "ethm = st.eth_momentum(DF, LOOKBACK)\n"
            "raw = st.predictive_regression(mom, fwd)\n"
            "ctl = st.predictive_regression(mom, fwd, controls=ethm.to_frame())\n"
            "p = st.placebo_pvalue(mom, fwd, n_perm=500, seed=SEED)\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "ax.bar(['raw','+ ETH-mom\\ncontrol'], [raw['hac_t'], ctl['hac_t']], color=[GREY, GREY], width=.5)\n"
            "ax.axhline(2, ls='--', c=GREEN, lw=1, label='t = +2 bar (a real lead)')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('predictive HAC-t'); ax.set_ylim(-3, 3)\n"
            "ax.set_title(f'No lead: HAC-t {raw[\"hac_t\"]:+.2f} raw / {ctl[\"hac_t\"]:+.2f} controlled (placebo p {p:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'raw HAC-t {raw[\"hac_t\"]:+.2f} (OLS-t {raw[\"ols_t\"]:+.2f}, R2 {raw[\"r2\"]:.4f}) | '\n"
            "      f'controlled {ctl[\"hac_t\"]:+.2f} | placebo p {p:.2f}')"
        ),
        md(
            f"> 💡 In plain words: H₁ **and** H₂ **rejected.** Raw predictive HAC-*t* = **{R['hac_t']}** "
            f"(placebo *p* {R['placebo_p']}, R² {R['r2']}); controlling for ETH's own momentum it is "
            f"**+{R['hac_t_ctrl']}**. Floor momentum adds nothing about forward ETH."
        ),
        md(
            "### 4c · Robustness — the horizon sweep\n\n"
            "The predictive HAC-*t* across five forward horizons, raw and controlled."
        ),
        code(
            "sweep = st.horizon_sweep(DF, LOOKBACK)\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "x = np.arange(len(sweep))\n"
            "ax.bar(x-0.2, sweep['hac_t_raw'], width=.4, color=GREY, label='raw')\n"
            "ax.bar(x+0.2, sweep['hac_t_ctrl_ethmom'], width=.4, color=AMBER, label='+ ETH-mom control')\n"
            "ax.axhline(2, ls='--', c=GREEN, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in sweep.index])\n"
            "ax.set_ylabel('predictive HAC-t'); ax.set_ylim(-3, 3)\n"
            "ax.set_title('No horizon clears |t| = 2 — no lead to harvest')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(sweep[['hac_t_raw','hac_t_ctrl_ethmom']].round(2).to_string())"
        ),
        md(
            "> 💡 In plain words: every horizon's |HAC-*t*| stays under ~1.5 and wanders around zero. "
            "There is no forward-predictive edge at 1, 3, 5, 10 or 21 days."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — H₁ rejected (HAC-*t* {R['hac_t']}, placebo *p* {R['placebo_p']}); H₂ "
            f"rejected (+{R['hac_t_ctrl']} controlled); no horizon clears |*t*| = 2. Synthetic-only "
            "caps it below REAL regardless.\n"
            f"- **Tradability `MIRAGE`** — overlay gross +{R['overlay_gross']}%/yr → net "
            f"{R['overlay_net']}%/yr, Sharpe {R['overlay_sharpe']}.\n"
            f"- **NFT floor leads crypto? `BUSTED`** — lagged beta {R['beta']}, lead-lag peak at lag "
            f"{R['llc_peak_lag']}: floors follow, not lead."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the overlay, gross and net\n\n"
            "Long ETH when floor momentum > 0, else flat; enter next day; charge 10 bps per position "
            "change."
        ),
        code(
            "ov = st.overlay_backtest(DF, lookback=LOOKBACK, cost_bps=10.0, allow_short=False)\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "g, n = ov['gross_ann']*100, ov['net_ann']*100\n"
            "ax.bar(['gross','net\\n(10bps x turnover)'], [g, n], color=[AMBER, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised return %')\n"
            "ax.set_title(f'Barely positive gross ({g:+.1f}%), negative net ({n:+.1f}%), Sharpe {ov[\"sharpe_net\"]:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.1f}%/yr -> net {n:+.1f}%/yr | Sharpe {ov[\"sharpe_net\"]:+.2f} | '\n"
            "      f'turnover {ov[\"turnover\"]:.3f} | frac long {ov[\"frac_long\"]*100:.0f}%')"
        ),
        md(
            "> 💡 In plain words: the overlay is just timed *partial* ETH beta (long ~half the days, "
            "chosen by a noise signal). Positive gross, **negative net** once you pay to trade. "
            "`MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a genuine lead "
            "(`lead_alpha > 0`) of increasing strength — floor momentum now *does* inform future ETH — "
            "and watch the predictive HAC-*t* climb past +2, averaged over 25 seeds so no single lucky "
            "seed can fake it."
        ),
        code(
            "alphas = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]\n"
            "ts = [st.synthetic_mean_hac_t(data, lead_alpha=a, horizon=HORIZON, n_seeds=25) for a in alphas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(alphas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = +2 bar (a real lead)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted lead_alpha (0 = null, larger = stronger lead)')\n"
            "ax.set_ylabel('mean predictive HAC-t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted lead — flat at the null, past +2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for a, t in zip(alphas, ts): print(f'lead_alpha {a:.2f} -> mean HAC-t {t:+.2f}')"
        ),
        md(
            "The HAC-*t* is ≈ 0 at the null and rises monotonically past **+2** once a real lead is "
            "planted (around `lead_alpha` ≈ 0.10) — so the engine is a faithful detector. The flat "
            "real result is therefore a statement about the **economics of an NFT floor index**: it is "
            "a lagged high-beta echo of ETH, not a leading risk-appetite signal. And because no clean "
            "free NFT floor tape exists, the signal axis is capped below REAL on data availability — "
            "the same honesty applied to the desk's [Lego](../../273-lego-returns/), "
            "[Whisky](../../275-whisky-cask/) and [Sneaker](../../276-sneaker-resale/) alt-data "
            "studies."
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
