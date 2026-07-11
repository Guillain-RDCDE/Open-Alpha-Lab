"""Generate the two narrative notebooks for Study 659 (Costless Collar).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY tape under
../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY 1993-02-01 ->
# 2026-06-30; 397-month derived sample 1993-06-30 -> 2026-06-30).
R = dict(
    spy_rows=8410, spy_start="1993-02-01", spy_end="2026-06-30", fp_spy="44daf42aa058",
    n_months=397, m_start="1993-06-30", m_end="2026-06-30",
    cap_mean=5.93, cap_min=5.81, cap_max=6.67, floor_pct=-5.0,
    vol_min=5.6, vol_max=70.7, vol_mean=16.4,
    breakeven_bps=3.84,
    cost_table={5.0: (-2.3, -0.34, -0.33, 31.39), 10.0: (-12.3, -1.79, -1.76, 21.17),
                15.0: (-22.3, -3.25, -3.20, 14.27), 20.0: (-32.3, -4.70, -4.63, 9.62)},
    floor_n=37, floor_cushion=2.68, floor_t=6.39, floor_worst_spy=-16.52, floor_worst_collar=-5.10,
    cap_n=42, cap_share=10.6, cap_cost=-1.83, cap_t=-7.67, cap_best_spy=12.70,
    tw_spy=29.95, tw_collar=31.39, sharpe_spy=0.57, sharpe_collar=0.69,
    dd_spy=-50.8, dd_collar=-33.6,
    gfc_spy=-50.8, gfc_collar=-31.9, gfc_cut=18.9,
    covid_spy=-19.4, covid_collar=-10.1, covid_cut=9.4,
    exwin_n=375, exwin_diff_bps=-10.8, exwin_t=-1.95, exwin_nwt=-2.10,
    exwin_tw_spy=61.05, exwin_tw_collar=44.83, exwin_sh_spy=0.81, exwin_sh_collar=0.86,
    syn_null_fires=0, syn_planted_t=-5.08, syn_planted_sd=1.03, syn_planted_fire=20,
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Free_protection%3F: Busted](https://img.shields.io/badge/Free_protection%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from costless_collar import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    DAILY = data.load_real()
    MF = data.month_frame(DAILY)
    CAPS = st.collar_caps(MF["vol_in"], put_otm=data.PUT_OTM, r=data.RF_ANNUAL, T=data.OPTION_T)
    MF = MF.join(CAPS.rename("cap_pct"))
    COLL5 = st.collar_returns(MF["spy_ret"], MF["cap_pct"], put_otm=data.PUT_OTM, cost_bps=5.0)
    DF = pd.concat([MF, COLL5], axis=1)
else:
    DAILY = MF = CAPS = DF = None
print("real cache present:", HAVE_REAL, "| months on tape:", (0 if DF is None else len(DF)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Can you really get crash insurance for free? 🪢\n"
            "### The \"costless collar\" — a real trade that's real about the crash "
            "protection and quiet about the price you pay\n\n"
            + BADGES +
            "An options desk pitches you this: own the S&P 500. Buy a put that floors your "
            "losses at 5%. Sell a call — at a strike chosen so its premium exactly pays for "
            "the put. Net cost today: **zero**. You've just bought crash insurance and "
            "someone else paid for it.\n\n"
            "That's the pitch. It has a real name (a *collar*), a real published index (the "
            "CBOE's own 95-110 Collar Index), and it is not a scam — the mechanism is real. "
            "The question is whether \"free\" survives contact with what actually happens "
            "over the following month, year, and decade.\n\n"
            "> 📓 **Plain-language layer.** Want the Black-Scholes cap solver, the *t*-stats "
            "and the cost-sensitivity sweep? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** There's no historical SPY option chain on yfinance, so we "
            "build a stylized monthly collar priced off Black-Scholes and realized "
            "volatility — every approximation is named, not hidden. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the floor actually protect you? | **Yes — a lot, in the crashes that "
            f"matter.** The 2008 drawdown gets cut from **{R['gfc_spy']:.0f}%** to "
            f"**{R['gfc_collar']:.0f}%**; 2020 from **{R['covid_spy']:.0f}%** to "
            f"**{R['covid_collar']:.0f}%**. |\n"
            "| Does the cap actually cost you? | **Yes — reliably.** In the roughly 1-in-10 "
            f"months SPY beat that month's cap, you gave up **{abs(R['cap_cost']):.2f} "
            f"points on average** — once as much as {R['cap_best_spy']:.0f}% in a single "
            "month. |\n"
            "| So does it net out ahead? | **Only because 2008 and 2020 happened to be in "
            "the sample.** Strip out just those two windows — leave everything else "
            "(including the dot-com crash) in — and the collar **loses to buy & hold, "
            "statistically for real**. |\n"
            "| Is it really \"free\"? | **No.** You always pay in the cap, every time it "
            "binds, crash or no crash. \"Costless\" describes the day you put the trade on, "
            "not the trade. |\n\n"
            "> Real insurance. Real premium. The premium is just paid in upside, not dollars."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Buy a put 5% below the market to floor your losses. Sell a call at a "
            "strike chosen so its premium exactly covers what you paid for the put. Net "
            "premium: zero. You've hedged the crash and it didn't cost you a thing.\"*\n\n"
            "It's a real, tradable structure — the CBOE even publishes an index for a version "
            "of it (the 95-110 Collar Index). The appeal is obvious: nobody likes paying for "
            "insurance, and this pitch says you don't have to."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a collar really were \"free\" downside protection, every long-only investor "
            "should run one, always — you'd cut your worst drawdowns to a known floor and "
            "pay nothing for it. That would be a genuine free lunch, and free lunches in "
            "markets are the thing this whole desk exists to hunt down and, usually, "
            "debunk.\n\n"
            "So: does the floor really protect you? Does the cap really cost you? And when "
            "you add it all up over a real 33-year stretch of the market — crashes included "
            "— who actually comes out ahead?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **Build the collar, honestly.** No live option chain exists in our data, so "
            "each month we price a 5%-OTM put and solve for the call strike that makes its "
            "premium match (Black-Scholes, off trailing realized volatility) — that's our "
            "model's \"costless.\"\n"
            "- **Clip SPY's real monthly return** to [floor, that month's cap], charge a "
            "modest cost for rolling both legs, and compare to just holding SPY.\n"
            "- **Split the sample two ways**: months where the floor actually bound (crash "
            "months) and months where the cap actually bound (melt-up months) — is either "
            "effect statistically real, or just noise?\n"
            "- **The falsification test:** strip the two crash windows the pitch itself is "
            "selling (2008, 2020) out of the sample. If the \"free lunch\" survives without "
            "them, it's structural. If it vanishes, it was never free — it was a bet on "
            "crashes happening, and it paid off because two did."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: does the floor protect, and does the cap cost?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cf = st.crash_floor_effect(DF, thresh=-data.PUT_OTM)\n"
            "    cc = st.cap_cost_effect(DF)\n"
            "    fc, ft = cf['mean_cushion_pts'], cf['t_plain']\n"
            "    cco, cct = cc['mean_cost_pts'], cc['t_plain']\n"
            "else:\n"
            "    fc, ft = R['floor_cushion'], R['floor_t']\n"
            "    cco, cct = R['cap_cost'], R['cap_t']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['floor bites\\n(crash months)', 'cap bites\\n(melt-up months)'], [fc, cco],\n"
            "       color=[GREEN, RED], width=.55)\n"
            "for i,(v,t_) in enumerate([(fc,ft),(cco,cct)]):\n"
            "    ax.annotate(f'{v:+.2f} pts\\n(t={t_:+.2f})',(i,v),ha='center',\n"
            "        va='bottom' if v>0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('collar return minus SPY return (points, on the months it binds)')\n"
            "ax.set_title('Both effects are real and large -- in opposite directions')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'floor cushion {fc:+.2f} pts (t={ft:+.2f})   cap cost {cco:+.2f} pts (t={cct:+.2f})')"
        ),
        md(
            f"Both bars clear the bar for \"statistically real\" by a wide margin "
            f"(**|t| ≥ 6** either way). The floor cushioned the worst SPY month from "
            f"**{R['floor_worst_spy']:.1f}%** to **{R['floor_worst_collar']:.1f}%**. The cap "
            f"gave up as much as **{R['cap_best_spy']:.1f}%** in a single strong month. "
            "Neither of these is folklore — they're mechanical facts about a floor-and-cap "
            "structure.\n\n"
            "**Now the crashes the whole trade is sold on:**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    d_gfc_spy = st.window_drawdown(DF['spy_ret'], *data.GFC_WINDOW)\n"
            "    d_gfc_coll = st.window_drawdown(DF['collar_ret'], *data.GFC_WINDOW)\n"
            "    d_cov_spy = st.window_drawdown(DF['spy_ret'], *data.COVID_WINDOW)\n"
            "    d_cov_coll = st.window_drawdown(DF['collar_ret'], *data.COVID_WINDOW)\n"
            "else:\n"
            "    d_gfc_spy, d_gfc_coll = R['gfc_spy']/100, R['gfc_collar']/100\n"
            "    d_cov_spy, d_cov_coll = R['covid_spy']/100, R['covid_collar']/100\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "x = np.arange(2); w = .32\n"
            "ax.bar(x-w/2, [d_gfc_spy*100, d_cov_spy*100], width=w, color=GREY, label='SPY')\n"
            "ax.bar(x+w/2, [d_gfc_coll*100, d_cov_coll*100], width=w, color=GREEN, label='collar')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['2008 GFC', '2020 COVID'])\n"
            "ax.set_ylabel('drawdown inside the window (%)')\n"
            "ax.set_title('The floor genuinely cuts the two crashes the pitch is sold on')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'2008: SPY {d_gfc_spy*100:.1f}%  collar {d_gfc_coll*100:.1f}%')\n"
            "print(f'2020: SPY {d_cov_spy*100:.1f}%  collar {d_cov_coll*100:.1f}%')"
        ),
        md(
            "This part of the pitch is simply true: real, substantial drawdown protection "
            "in both headline crashes. If the story stopped here, it'd be a clean win.\n\n"
            "**But now the honest test — take the two crashes back out.** Not all of "
            "history, just the two specific windows the pitch itself invokes:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sub = st.exclude_windows(DF, [data.GFC_WINDOW, data.COVID_WINDOW])\n"
            "    tw_spy = st.terminal_wealth(sub['spy_ret'])\n"
            "    tw_coll = st.terminal_wealth(sub['collar_ret'])\n"
            "else:\n"
            "    tw_spy, tw_coll = R['exwin_tw_spy'], R['exwin_tw_collar']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.6))\n"
            "ax.bar(['SPY buy & hold', 'costless collar'], [tw_spy, tw_coll],\n"
            "       color=[GREEN, RED], width=.5)\n"
            "for i,v in enumerate([tw_spy, tw_coll]): ax.annotate(f'${v:,.2f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('$1 compounded, same 375 months, 2008 & 2020 excluded')\n"
            "ax.set_title('Take the two named crashes back out -- and the collar loses')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'ex-crash terminal wealth: SPY ${tw_spy:,.2f}  vs  collar ${tw_coll:,.2f}')"
        ),
        md(
            f"$1 in SPY becomes **${R['exwin_tw_spy']:.2f}**; the same $1 run through the "
            f"collar becomes **${R['exwin_tw_collar']:.2f}** — over the *identical* 375 "
            "months, with the two headline crashes removed (the 2000–2002 dot-com bear "
            "market is deliberately left in, so this isn't rigged in SPY's favor). The gap "
            f"is statistically real (Newey-West *t* = {R['exwin_nwt']:.2f}, clears the desk's "
            "bar). The \"free lunch\" wasn't free — it was a bet, and it happened to pay off "
            "because two real crashes landed inside this particular 33-year window."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Mixed.** The floor's crash cushion and the cap's upside cost are "
            "*both* mechanically real on the tape — the sign of the net effect just depends "
            "on whether a crash actually lands in your holding period, and you can't know "
            "that in advance.\n"
            "- **Tradability — Fragile.** The apparent \"win\" survives only under a "
            "generous cost assumption and only because 2 of 33 years happened to contain "
            "real crashes. Push costs up a little, or hold through a crash-free decade, and "
            "it flips.\n"
            "- **\"Truly free protection?\" — Busted.** You always pay in the cap. Whether "
            "that payment turns out to be \"worth it\" is a question about the *future*, not "
            "something the trade's structure can promise you."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The honest way to think about a collar** is as insurance you're pre-paying "
            "for in upside, not a free hedge — exactly like any other insurance, it's a good "
            "deal if the bad thing happens and a bad deal if it doesn't, and you don't get to "
            "know which in advance.\n"
            "- **This is a stylized model, not a live chain.** Real listed SPY/SPX collars "
            "price off a skewed implied-vol surface, not the flat, realized-vol proxy used "
            "here — see [02_for_the_quants](02_for_the_quants.ipynb) and "
            "[docs/references.md](../docs/references.md) for exactly how and why that likely "
            "understates the true cost of the put relative to the call.\n"
            "- **Sibling studies:** the [crash-insurance-cost](../../617-crash-insurance-cost/) "
            "study prices the naked put alone; [put-write-premium](../../658-put-write-premium/) "
            "is the mirror-image short-put trade; the "
            "[covered-call ETF study](../../337-covered-call-etf/) is this collar with the "
            "put leg removed; the [trailing-stop study](../../99-safety-net/) protects "
            "without options at all.\n\n"
            "*Think a real, skew-priced collar changes the answer? Show it on a live option "
            "chain, after realistic bid-ask, and we'll rerun the numbers.*"
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
            "# The Costless Collar — a quantitative teardown 🔬\n"
            "### The Black-Scholes cap solver · the floor/cap split tests · a cost-sensitivity "
            "and break-even sweep · the ex-crash-window robustness check · a 20-seed "
            "synthetic clip-mechanics control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — a **zero-net-premium collar gives crash protection for free** — has "
            "a real mechanism (a floor financed by a cap) and a real published analogue (the "
            "CBOE 95-110 Collar Index). The job here is to price the stylized structure "
            "honestly, measure both legs, and ask whether the sign of the net effect is a "
            "property of the trade or a property of the sample.\n\n"
            "> ⚠️ **Data note.** No historical SPY option chain exists on yfinance — every "
            "collar leg here is a Black-Scholes approximation priced off trailing realized "
            "volatility, our proxy for implied vol. SPY daily OHLC+AdjClose 1993→2026, "
            "yfinance, cached. No survivorship (SPY is a rules-based index ETF). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp_spy"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | floor cushion **+{R['floor_cushion']:.2f} pts, "
            f"t={R['floor_t']:.2f}**; cap cost **{R['cap_cost']:.2f} pts, t={R['cap_t']:.2f}**; "
            f"ex-2008/2020 drag **NW t={R['exwin_nwt']:.2f}** |\n"
            f"| **Tradability** | `FRAGILE` | break-even **{R['breakeven_bps']:.2f} bps/leg**; "
            f"full-sample win concentrated in {R['n_months']-R['exwin_n']} of {R['n_months']} months |\n"
            f"| **Free protection?** | `BUSTED` | the cap's cost clears t={R['cap_t']:.2f} "
            "every time it binds, crash or no crash |\n\n"
            "> 💡 In plain words: both halves of the trade are real; whether they net out in "
            "your favor is a bet on the future, not a property of the structure."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be SPY's realized total return in month $t$, $f=-5\\%$ the fixed put "
            "strike (floor), and $c_t$ the call strike (cap) solved so the Black-Scholes "
            "premiums of the put and call match at that month's trailing-realized-vol input "
            "$\\sigma_t$ (known through the close of month $t-1$ — zero look-ahead). The "
            "collar's realized return is $\\text{clip}(r_t,\\,f,\\,c_t)$ minus a 2-leg roll "
            "cost. The claims:\n\n"
            "- **H₁ (protection).** In months where $r_t < f$, the collar materially and "
            "significantly beats SPY.\n"
            "- **H₂ (cost).** In months where $r_t > c_t$, the collar materially and "
            "significantly trails SPY.\n"
            "- **H₃ (free lunch).** Averaged over a realistic multi-decade sample, H₁ and H₂ "
            "net to (at worst) zero — the structure doesn't cost you money on net.\n\n"
            "We find **H₁ and H₂ both strongly supported** (|t| > 6 each), and **H₃ "
            "unsupported once the 2008/2020 windows are excluded** (NW t = "
            f"{R['exwin_nwt']:.2f}) — the full-sample near-wash is a property of *this "
            "particular sample containing 2 real crashes*, not of the trade's structure."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design and the pricing model\n\n"
            "**Pricing.** Both legs use the same Black-Scholes formula "
            "($N(\\cdot)$ via `math.erf`, no scipy dependency), the same constant risk-free "
            "rate (3%/yr) and the same trailing-realized-vol input for a given month — the "
            "call strike is found by bisection so `bs_call(K_call) == bs_put(K_put=0.95)`. "
            "This is a **flat-vol, skew-free approximation** of a real listed collar (see "
            "`docs/references.md` for the direction of the resulting bias).\n\n"
            "**Inference.** Floor/cap bite tests are one-sample *t* on the paired monthly "
            "difference (collar − SPY), restricted to the months each leg actually binds. "
            "The full-sample and ex-crash-window drags additionally carry a **Newey-West "
            "(1987) 5→12-lag HAC** cross-check (monthly returns carry mild "
            "autocorrelation). The ex-crash-window split is **not** snooped to a favorable "
            "cut: it drops precisely the two windows the claim's own marketing invokes "
            "(2008 GFC, 2020 COVID) and *deliberately* leaves the 2000–2002 dot-com bear "
            "market in the \"ex-crash\" bucket."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** SPY daily OHLC+AdjClose {R['spy_start']} → {R['spy_end']} "
            f"({R['spy_rows']:,} rows); derived monthly sample {R['m_start']} → {R['m_end']} "
            f"({R['n_months']} months — the first ~4 months are a vol warm-up).\n"
            "- **Model.** Put fixed at 5% OTM; call solved for equal BS premium at that "
            "month's trailing-63-session realized vol (known before the month begins — the "
            "study's one documented execution lag).\n"
            "- **Headline.** Cost-sensitivity sweep (5/10/15/20 bps/leg) + a bisected "
            "break-even cost.\n"
            "- **Anatomy.** Floor-bite and cap-bite one-sample *t* tests, restricted to the "
            "months each leg actually binds.\n"
            "- **Aggregate.** Terminal wealth, Sharpe (excess of the same 3% cash rate) and "
            "max drawdown, full sample and the two named crash windows.\n"
            "- **Robustness.** The ex-2008/2020-window drag test, HAC *t* checked across "
            "4 lag choices.\n"
            "- **Control.** Synthetic monthly GBM, no roll costs, a never-binding null band "
            "vs. an always-binding planted band; the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The cap solver and its odd invariance\n\n"
            "The modeled cap barely moves across a huge range of trailing realized vol — "
            "worth seeing directly, because it's counter-intuitive and we didn't smooth it "
            "away."
        ),
        code(
            "if HAVE_REAL:\n"
            "    vmin, vmax, vmean = MF['vol_in'].min(), MF['vol_in'].max(), MF['vol_in'].mean()\n"
            "    cmin, cmax, cmean = MF['cap_pct'].min(), MF['cap_pct'].max(), MF['cap_pct'].mean()\n"
            "else:\n"
            "    vmin, vmax, vmean = R['vol_min']/100, R['vol_max']/100, R['vol_mean']/100\n"
            "    cmin, cmax, cmean = R['cap_min']/100, R['cap_max']/100, R['cap_mean']/100\n"
            "svals = np.linspace(max(vmin,0.03), vmax, 60)\n"
            "caps = [st.solve_costless_call_strike(1.0, 1-data.PUT_OTM, data.OPTION_T, data.RF_ANNUAL, s)-1 for s in svals]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.plot(svals*100, np.array(caps)*100, color=RED, lw=2)\n"
            "ax.axhline(cmean*100, ls='--', c=GREY, lw=1, label=f'observed sample mean cap {cmean*100:.2f}%')\n"
            "ax.set_xlabel('trailing realized vol input, annualized (%)')\n"
            "ax.set_ylabel('modeled costless cap (%)')\n"
            "ax.set_title('The cap is nearly vol-invariant at this floor/tenor/rate -- a model property')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'realized vol range on tape: {vmin*100:.1f}%-{vmax*100:.1f}% (mean {vmean*100:.1f}%)')\n"
            "print(f'modeled cap range: {cmin*100:.2f}%-{cmax*100:.2f}% (mean {cmean*100:.2f}%)')"
        ),
        md(
            f"> 💡 In plain words: from a calm **{R['vol_min']:.0f}%**-vol month to a COVID-"
            f"spike **{R['vol_max']:.0f}%**-vol month, the solved cap only moves from "
            f"**{R['cap_min']:.2f}%** to **{R['cap_max']:.2f}%**. That's an emergent property "
            "of equalizing two Black-Scholes premiums at a short tenor and modest moneyness "
            "— not a coding bug — and it means this model does **not** predict a wider "
            "safety margin in calm markets. Flagged, not hidden (`docs/references.md`)."
        ),
        md(
            "### 4b · Where the floor bites, where the cap bites\n\n"
            "One-sample *t* on the paired monthly difference, restricted to the months each "
            "leg actually binds."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cf = st.crash_floor_effect(DF, thresh=-data.PUT_OTM)\n"
            "    cc = st.cap_cost_effect(DF)\n"
            "else:\n"
            "    cf = dict(n=R['floor_n'], mean_cushion_pts=R['floor_cushion'], t_plain=R['floor_t'],\n"
            "              worst_spy_pct=R['floor_worst_spy'], worst_collar_pct=R['floor_worst_collar'])\n"
            "    cc = dict(n=R['cap_n'], mean_cost_pts=R['cap_cost'], t_plain=R['cap_t'],\n"
            "              share_of_months=R['cap_share'], best_spy_pct=R['cap_best_spy'])\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "vals = [cf['mean_cushion_pts'], cc['mean_cost_pts']]\n"
            "ts = [cf['t_plain'], cc['t_plain']]\n"
            "ax.bar(['floor bites\\n(SPY < -5%)', 'cap bites\\n(SPY > cap)'], vals,\n"
            "       color=[GREEN, RED], width=.55)\n"
            "for i,(v,t_) in enumerate(zip(vals, ts)):\n"
            "    ax.annotate(f'{v:+.2f} pts (t={t_:+.2f})',(i,v),ha='center',va='bottom' if v>0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.axhline(2, ls='--', c=GREY, lw=.8)\n"
            "ax.set_ylabel('collar - SPY (points, conditional on the leg binding)')\n"
            "ax.set_title('Both legs are individually decisive (|t| >> 2)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"floor: n={cf['n']}, {cf['mean_cushion_pts']:+.2f} pts, t={cf['t_plain']:+.2f}, \"\n"
            "      f\"worst SPY {cf['worst_spy_pct']:+.1f}% -> collar {cf['worst_collar_pct']:+.1f}%\")\n"
            "print(f\"cap:   n={cc['n']} ({cc['share_of_months']:.1f}%), {cc['mean_cost_pts']:+.2f} pts, \"\n"
            "      f\"t={cc['t_plain']:+.2f}, best SPY given up {cc['best_spy_pct']:+.1f}%\")"
        ),
        md(
            f"> 💡 In plain words: the floor cushions **{R['floor_n']} crash months** by "
            f"**+{R['floor_cushion']:.2f} points on average** (t = {R['floor_t']:.2f}); the "
            f"cap costs **{R['cap_n']} melt-up months** ({R['cap_share']:.1f}% of the sample) "
            f"**{R['cap_cost']:.2f} points on average** (t = {R['cap_t']:.2f}). Both clear "
            "the desk bar by a wide margin — neither is noise."
        ),
        md(
            "### 4c · Cost sensitivity and the break-even\n\n"
            "The full-sample mean(collar − SPY), swept across the 2-legs-per-month roll cost, "
            "with a bisected break-even."
        ),
        code(
            "if HAVE_REAL:\n"
            "    be = st.breakeven_cost_bps(MF['spy_ret'], MF['cap_pct'], put_otm=data.PUT_OTM)\n"
            "    costs = [5.0, 10.0, 15.0, 20.0]\n"
            "    diffs, ts = [], []\n"
            "    for cb in costs:\n"
            "        coll = st.collar_returns(MF['spy_ret'], MF['cap_pct'], put_otm=data.PUT_OTM, cost_bps=cb)\n"
            "        d = pd.concat([MF, coll], axis=1)\n"
            "        drag = st.full_sample_drag(d)\n"
            "        diffs.append(drag['mean_diff_bps']); ts.append(drag['t_nw'])\n"
            "else:\n"
            "    be = R['breakeven_bps']\n"
            "    costs = sorted(R['cost_table'])\n"
            "    diffs = [R['cost_table'][c][0] for c in costs]\n"
            "    ts = [R['cost_table'][c][2] for c in costs]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.plot(costs, diffs, 'o-', color=RED)\n"
            "a1.axhline(0, c='k', lw=.8); a1.axvline(be, ls='--', c=GREY, lw=1, label=f'break-even {be:.2f} bps')\n"
            "a1.set_xlabel('cost per leg (bps)'); a1.set_ylabel('mean(collar-SPY), bps/mo')\n"
            "a1.set_title('The edge is razor-thin in cost space'); a1.legend()\n"
            "a2.bar([str(c) for c in costs], ts, color=[RED if abs(t)>=2 else AMBER for t in ts], width=.6)\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_xlabel('cost per leg (bps)'); a2.set_ylabel('Newey-West t')\n"
            "a2.set_title('By 15 bps/leg the drag is decisive')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'break-even: {be:.2f} bps/leg')\n"
            "print({c: round(d,1) for c,d in zip(costs, diffs)})"
        ),
        md(
            f"> 💡 In plain words: break-even is **{R['breakeven_bps']:.2f} bps/leg** — "
            "thinner than the desk's own default 5 bps convention leaves comfortable room "
            "for (5 bps is already past it), and almost certainly thinner than a real OTM "
            "SPX/SPY option's bid-ask. By 15 bps the drag is decisively negative "
            f"(NW t = {R['cost_table'][15.0][2]:.2f})."
        ),
        md(
            "### 4d · Full sample vs. the two named crashes taken out\n\n"
            "The falsification test: does the full-sample near-wash survive removing "
            "*exactly* the two windows the claim's own marketing invokes?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tw_spy, tw_coll = st.terminal_wealth(DF['spy_ret']), st.terminal_wealth(DF['collar_ret'])\n"
            "    sub = st.exclude_windows(DF, [data.GFC_WINDOW, data.COVID_WINDOW])\n"
            "    drag_sub = st.full_sample_drag(sub)\n"
            "    tw_spy_sub = st.terminal_wealth(sub['spy_ret']); tw_coll_sub = st.terminal_wealth(sub['collar_ret'])\n"
            "    nwt_sub = drag_sub['t_nw']\n"
            "else:\n"
            "    tw_spy, tw_coll = R['tw_spy'], R['tw_collar']\n"
            "    tw_spy_sub, tw_coll_sub = R['exwin_tw_spy'], R['exwin_tw_collar']\n"
            "    nwt_sub = R['exwin_nwt']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.4))\n"
            "a1.bar(['SPY','collar'], [tw_spy, tw_coll], color=[GREY, GREEN], width=.5)\n"
            "for i,v in enumerate([tw_spy, tw_coll]): a1.annotate(f'${v:,.2f}',(i,v),ha='center',va='bottom')\n"
            "a1.set_title('Full sample (2008 + 2020 IN)\\n$1 compounded'); a1.set_ylabel('$')\n"
            "a2.bar(['SPY','collar'], [tw_spy_sub, tw_coll_sub], color=[GREEN, RED], width=.5)\n"
            "for i,v in enumerate([tw_spy_sub, tw_coll_sub]): a2.annotate(f'${v:,.2f}',(i,v),ha='center',va='bottom')\n"
            "a2.set_title(f'2008 + 2020 OUT (NW t={nwt_sub:+.2f})\\n$1 compounded, same 375 months')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'full sample: SPY ${tw_spy:,.2f}  collar ${tw_coll:,.2f}')\n"
            "print(f'ex-2008/2020: SPY ${tw_spy_sub:,.2f}  collar ${tw_coll_sub:,.2f}  (NW t={nwt_sub:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: with the two crashes IN the sample, the collar edges out "
            f"SPY (${R['tw_collar']:.2f} vs ${R['tw_spy']:.2f}) at a statistically "
            "insignificant gap. Pull *just* those two windows back out (dot-com bear market "
            f"stays in) and SPY pulls to **${R['exwin_tw_spy']:.2f}** against the collar's "
            f"**${R['exwin_tw_collar']:.2f}** — a real, HAC-robust shortfall "
            f"(NW t = {R['exwin_nwt']:.2f}, stable across 0/3/6/12 lags: −1.96 / −2.10 / "
            "−2.11 / −2.15). The full-sample \"win\" lives entirely inside 22 specific "
            f"months out of {R['n_months']}."
        ),
        md(
            "### 4e · Sharpe vs. terminal wealth — they don't have to agree\n\n"
            "Worth stating plainly: the collar's Sharpe stays *higher* than SPY's even "
            "ex-crash, because clipping both tails lowers volatility a lot — but the number "
            "that pays your bills is terminal wealth, and that one goes the other way."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sh_spy = st.sharpe_excess(DF['spy_ret'], data.RF_ANNUAL)\n"
            "    sh_coll = st.sharpe_excess(DF['collar_ret'], data.RF_ANNUAL)\n"
            "    sh_spy_sub = st.sharpe_excess(sub['spy_ret'], data.RF_ANNUAL)\n"
            "    sh_coll_sub = st.sharpe_excess(sub['collar_ret'], data.RF_ANNUAL)\n"
            "else:\n"
            "    sh_spy, sh_coll = R['sharpe_spy'], R['sharpe_collar']\n"
            "    sh_spy_sub, sh_coll_sub = R['exwin_sh_spy'], R['exwin_sh_collar']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "x = np.arange(2); w = .32\n"
            "ax.bar(x-w/2, [sh_spy, sh_spy_sub], width=w, color=GREY, label='SPY')\n"
            "ax.bar(x+w/2, [sh_coll, sh_coll_sub], width=w, color=GREEN, label='collar')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['full sample', 'ex-2008/2020'])\n"
            "ax.set_ylabel('Sharpe (excess of 3% cash)')\n"
            "ax.set_title('Sharpe still favors the collar ex-crash -- terminal wealth does not')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Sharpe full: SPY {sh_spy:.2f} vs collar {sh_coll:.2f}')\n"
            "print(f'Sharpe ex-crash: SPY {sh_spy_sub:.2f} vs collar {sh_coll_sub:.2f}')"
        ),
        md(
            f"> 💡 In plain words: ex-crash Sharpe is **{R['exwin_sh_spy']:.2f}** (SPY) vs "
            f"**{R['exwin_sh_collar']:.2f}** (collar) — the collar still *looks* better on a "
            "risk-adjusted basis, purely because clipping both tails cuts volatility. But "
            f"the dollar outcome over the same months is **${R['exwin_tw_spy']:.2f}** vs "
            f"**${R['exwin_tw_collar']:.2f}** — that's the version that matters. "
            "Sharpe rewards a smooth ride; it does not reward getting there faster. Don't "
            "let a favorable Sharpe substitute for the terminal-wealth number when someone's "
            "actually pitching you this trade."
        ),
        md(
            "### 4f · Faithful-engine control — the clip mechanics are unbiased\n\n"
            "Synthetic monthly GBM (16%/yr vol, 9%/yr drift), no roll costs (isolating the "
            "clip effect from the flat-cost assumption), 20 seeds."
        ),
        code(
            "null_diffs = []\n"
            "planted_ts = []\n"
            "for s_ in range(20):\n"
            "    world = data.synthetic_world(seed=659 + s_)['spy_ret']\n"
            "    null_r = st.synthetic_detect(world, floor=-0.99, cap=0.99)\n"
            "    null_diffs.append(null_r['mean_diff_bps'])\n"
            "    planted_ts.append(st.synthetic_detect(world, floor=-0.03, cap=0.02)['t_plain'])\n"
            "null_diffs = np.asarray(null_diffs); planted_ts = np.asarray(planted_ts)\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_diffs, color=GREY, s=40,\n"
            "           label='null (never-binding band), 20 seeds')\n"
            "ax.scatter(np.ones(20) + np.linspace(-.12,.12,20), planted_ts, color=RED, s=40,\n"
            "           label='planted (tight, always-binding band), 20 seeds')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks([0,1]); ax.set_xticklabels(['null: diff (bps)', 'planted: t-stat'])\n"
            "ax.set_title('Control: a no-op band invents nothing; a tight band is unmistakable')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: max |diff| across 20 seeds = {np.max(np.abs(null_diffs)):.6f} bps (exactly zero)')\n"
            "print(f'planted: mean t = {planted_ts.mean():+.2f} (sd {planted_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(planted_ts)>=2).sum()}/20 seeds')"
        ),
        md(
            f"> 💡 In plain words: a floor/cap wide enough to never bind at realistic vol "
            "produces **exactly zero** difference from raw returns in every single one of "
            "20 seeds — the strongest possible non-signal. A tight, always-binding band "
            f"lights up unmistakably (mean t = {R['syn_planted_t']:.2f}, "
            f"{R['syn_planted_fire']}/20 seeds clearing |t| ≥ 2). The clip-and-drag "
            "machinery reacts exactly as it should. *(A faithful-engine / power check only — "
            "never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — the floor cushions crashes (**+{R['floor_cushion']:.2f} "
            f"pts/event, t = {R['floor_t']:.2f}**) and the cap costs bull months "
            f"(**{R['cap_cost']:.2f} pts/event, t = {R['cap_t']:.2f}**), both decisively real "
            "on the tape. The net sign is regime-dependent: full sample with 2008/2020 IN, "
            f"it's a wash (t = {R['cost_table'][5.0][1]:.2f}); with just those two windows "
            f"OUT, it's a certified drag (NW t = {R['exwin_nwt']:.2f}).\n"
            f"- **Tradability `FRAGILE`** — break-even sits at a thin "
            f"{R['breakeven_bps']:.2f} bps/leg, and the entire full-sample edge is "
            f"concentrated in {R['n_months']-R['exwin_n']} of {R['n_months']} months. Neither "
            "a cost buffer nor a robustness margin survives realistic stress.\n"
            "- **\"Truly free protection?\" `BUSTED`** — the cap's cost is mechanically "
            "guaranteed and statistically certified every time it binds, independent of "
            "whether a crash ever arrives to \"justify\" it. \"Costless\" is a statement "
            "about day-one option premiums, not about the trade you actually end up holding."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general object is insurance pricing, not this specific structure.** Any "
            "self-financing hedge (collars, put spreads financed by calls, risk-reversals) "
            "faces the same arithmetic: the premium you \"don't pay\" in cash you pay in "
            "forgone optionality, and whether that trade nets out ahead is a statement about "
            "realized future volatility and drawdown timing, not about the structure itself.\n"
            "- **The obvious next test:** price the collar off a real, skewed implied-vol "
            "surface (once/if a historical SPX option chain is available) rather than a flat "
            "realized-vol proxy — our own reasoning in `docs/references.md` suggests this "
            "would widen the modeled cap somewhat, which would only make the cap-cost side "
            "of this study's finding *more* conservative, not less.\n"
            "- **Dedup map:** [617-crash-insurance-cost](../../617-crash-insurance-cost/) "
            "(the naked put, no financing), [658-put-write-premium](../../658-put-write-premium/) "
            "(the mirror-image short-put trade), "
            "[337-covered-call-etf](../../337-covered-call-etf/) (this collar with the put "
            "leg removed), [99-safety-net](../../99-safety-net/) (protection via exit rules, "
            "no options at all).\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
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
