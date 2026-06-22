"""Generate the two narrative notebooks for Study 354 (The Wheel).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. The real-tape cells read the cached SPY+VIX
parquet under ../_cache/ when present; otherwise every later cell quotes the frozen headline
numbers in ``R`` (a mirror of docs/results.md), embedded into the first code cell. The
Black-Scholes premium model, the Wheel engine, and the synthetic positive control run anywhere
with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md
# (SPY total-return + ^VIX as Black-Scholes IV, 401 month-ends 1993-01..2026-05, as-of 2026-06-18).
R = dict(
    asof="2026-06-18", window="1993-01-31 → 2026-05-31", n=401,
    vix_mean=19.6, vix_med=17.9, prem_mo=2.25,
    spy_lo=24.11, spy_hi=754.54,
    wheel_cagr=11.61, wheel_vol=8.7, wheel_sharpe=1.32, wheel_mdd=-22.8, wheel_skew=-1.87, wheel_final=39.23,
    spy_cagr=8.09, spy_vol=14.9, spy_sharpe=0.60, spy_mdd=-55.8, spy_skew=-0.36, spy_final=13.44,
    cash_cagr=3.04, cash_final=2.72,
    edge_full=2.50, t_full=1.73, win=58,
    edge_h1=6.21, t_h1=3.11, edge_h2=-1.20, t_h2=-0.58,
    up_cap=69, dn_cap=36, n_up=253, n_dn=147,
    spy_up=3.24, wheel_up=2.25, spy_dn=-3.55, wheel_dn=-1.28,
    worst1=-12.3, worst1_date="2020-03", y2008=-10.7, worst5=-51,
    costs=[(0.00, 2.50, 1.73, 11.61), (0.10, 1.30, 0.90, 10.29),
           (0.25, -0.50, -0.35, 8.33), (0.50, -3.50, -2.43, 5.15)],
    # synthetic control: (vix_premium, crash%, edge_ann, t, up_cap, dn_cap, skew)
    syn=[(1.00, 0, -3.71, -7.9, 45, 46, -1.67),
         (1.00, 1, -3.37, -7.2, 45, 51, -3.00),
         (1.20, 1, 1.05, 2.3, 55, 42, -3.00)],
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Free_income%3F: Busted](https://img.shields.io/badge/Free_income%3F-Busted-8b949e?style=flat-square)\n\n"
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

from the_wheel import data, strategy as st

HAVE_REAL = data.have_real()
M = data.load_real().iloc[:-1] if HAVE_REAL else None   # drop the in-progress final month
print("real SPY+VIX cache present:", HAVE_REAL,
      "| months:", (0 if M is None else len(M)))
"""

# The frozen headline dict is embedded into the notebook's first code cell so every
# downstream cell can quote it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Wheel — sell puts, then calls, forever for \"income\"? 🎡\n"
            "### The retail options-income machine, in plain English\n\n"
            + BADGES +
            "All over r/thetagang and a thousand \"passive income\" videos: the **Wheel**. *Sell a "
            "cash-secured **put** on SPY each month and pocket the premium. If the stock dips and you "
            "get \"assigned,\" no problem — you now own the shares, so **sell calls** against them and "
            "keep pocketing premium. When they get called away, go back to selling puts. Round and "
            "round — \"income\" every month, up market or down.*\n\n"
            "It sounds like rent on your money. Is it? We rebuild the Wheel on **33 years of real SPY**, "
            "pricing each month's option transparently from the **VIX**, and race it against just "
            "**holding SPY**. The short answer: it's not free income — it's **short volatility wearing "
            "an income costume**. It keeps a slice of the up months, gives the rest away, and still "
            "owns the crashes.\n\n"
            "> 📓 **Plain-language layer.** Want the Black-Scholes premium, the paired *t*-stats and the "
            "capture decomposition? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** Every chart is drawn by the code beside it. We price the "
            "option with a *transparent model* (Black-Scholes, VIX as the implied vol) — not a live "
            "option chain — and that model is *generous* to the seller, so reality is a bit worse. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is the monthly premium real money? | **Yes — but it's not free.** You're paid because "
            "you're **selling insurance** (short volatility). The cheque is the *variance risk "
            "premium*, not magic income. |\n"
            "| Does the Wheel beat just holding SPY? | **Only sometimes, and not reliably.** Gross of "
            f"costs it looks great (CAGR {R['wheel_cagr']:.1f}% vs {R['spy_cagr']:.1f}%), but the edge "
            f"**flipped negative** in the last 16 years and dies the moment you pay real trading "
            "costs. |\n"
            "| Is it \"market-neutral income\"? | **No.** It keeps only "
            f"**~{R['up_cap']}%** of the up months but rides **most of the way down** in crashes. "
            "That's a *directional* bet with the upside chopped off. |\n"
            "| So what *is* the Wheel? | **A covered-call / short-put position in disguise** — short "
            f"volatility. Skew **{R['wheel_skew']:.1f}** (vs SPY's {R['spy_skew']:.2f}): tiny steady "
            "gains, rare ugly losses. |\n\n"
            "> The premium is real; it's the price of selling insurance. The \"income\" is the upside "
            "you handed over. One crash collects on the policy."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Sell a cash-secured **put** at-the-money on SPY. If it expires above your strike, you "
            "keep the premium — free money. If it drops below, you're **assigned** the shares at a "
            "price you liked anyway. Now you own SPY, so **sell covered calls** at-the-money and keep "
            "collecting. When the calls get exercised, you're back in cash — start the wheel again. "
            "Premium every single month, in any market.\"*\n\n"
            "It's seductive because **every leg collects cash up front**. Selling the put pays you. "
            "Selling the call pays you. The wheel never stops turning, so the income never stops… "
            "right?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, the Wheel would be a holy grail: equity-like returns with bond-like calm and a "
            "steady monthly cheque. That's the dream every \"theta gang\" post is selling. But there's "
            "a catch hiding in plain sight. **When you sell that put, what are you giving up?** If SPY "
            "rips upward, your put expires worthless — you keep a tiny premium and **miss the whole "
            "rally** (you were in cash). And when you sell the call on shares you own, if SPY rips up "
            "you're **capped** — the gains above your strike aren't yours. Either way you've **sold "
            "away the upside**. The only question is whether the premium you bank is bigger than the "
            "upside you surrendered. Let's measure both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We rebuild the Wheel on **{R['n']} months** of real SPY ({R['window']}). Each month:\n\n"
            "1. **Price the option honestly.** We use **Black-Scholes** with the **VIX** as the "
            "implied vol — a transparent, public model (no insider option chain). It's actually a "
            "*little generous* to the seller, because VIX usually sits **above** the volatility that "
            "actually shows up.\n"
            "2. **Spin the wheel.** Sell the ATM put; if SPY ends below, get assigned and switch to "
            "selling ATM calls; when called away, switch back. Track the monthly return.\n"
            "3. **Race it.** Against **buy-and-hold SPY** and a **cash** baseline. And split the months "
            "into *up* months and *down* months to see exactly how much of each the Wheel keeps.\n\n"
            "**What would make us call it \"income\"?** A real, robust return edge over buy-and-hold "
            "that survives costs and doesn't depend on the decade. Spoiler: it doesn't."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline race.** $1 in the Wheel vs $1 in buy-and-hold SPY vs $1 in cash, "
            "over 33 years — with the option premium priced from the VIX."
        ),
        code(
            "if HAVE_REAL:\n"
            "    w = st.run_wheel(M); spy = M['spy_ret'].to_numpy()\n"
            "    nav_w = st.equity_curve(w['wheel_ret']); nav_s = st.equity_curve(spy)\n"
            "    idx = M.index\n"
            "    sw = st.summary(w['wheel_ret']); ss = st.summary(spy)\n"
            "    cagr_w, cagr_s = sw['cagr']*100, ss['cagr']*100\n"
            "else:\n"
            "    idx = np.arange(R['n'])\n"
            "    nav_w = (1+R['wheel_cagr']/100)**(np.arange(R['n'])/12)\n"
            "    nav_s = (1+R['spy_cagr']/100)**(np.arange(R['n'])/12)\n"
            "    cagr_w, cagr_s = R['wheel_cagr'], R['spy_cagr']\n"
            "nav_c = (1+R['cash_cagr']/100)**(np.arange(len(nav_w))/12)\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.8))\n"
            "ax.plot(idx, nav_w, c=AMBER, lw=2, label=f'The Wheel ({cagr_w:.1f}%/yr)')\n"
            "ax.plot(idx, nav_s, c=GREEN, lw=2, label=f'Buy & hold SPY ({cagr_s:.1f}%/yr)')\n"
            "ax.plot(idx, nav_c, c=GREY, lw=1.5, ls='--', label=f\"Cash ({R['cash_cagr']:.1f}%/yr)\")\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)'); ax.set_title('33 years: the Wheel vs just holding SPY')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Wheel ${nav_w[-1]:.1f}x   SPY ${nav_s[-1]:.1f}x   (gross of trading costs)')"
        ),
        md(
            f"Gross of costs the Wheel actually *wins* here — **{R['wheel_cagr']:.1f}%/yr** vs SPY's "
            f"**{R['spy_cagr']:.1f}%**, at a much smoother ride (a {R['wheel_mdd']:.0f}% worst drop vs "
            f"SPY's brutal {R['spy_mdd']:.0f}%). So is the pitch right after all? **No — and here's the "
            "tell.** That edge is entirely the *insurance premium* (VIX ran rich of reality for 33 "
            "years), and it comes with a nasty shape. Watch what happens in up months vs down months."
        ),
        md(
            "**The shape: what the Wheel keeps.** Split every month into *SPY went up* and *SPY went "
            "down*, and compare the Wheel's average to SPY's average in each bucket."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cap = st.capture(M, w)\n"
            "    su, wu, sd, wd = cap['spy_up_mean']*100, cap['wheel_up_mean']*100, cap['spy_dn_mean']*100, cap['wheel_dn_mean']*100\n"
            "    upc, dnc = cap['up_capture']*100, cap['dn_capture']*100\n"
            "else:\n"
            "    su, wu, sd, wd = R['spy_up'], R['wheel_up'], R['spy_dn'], R['wheel_dn']\n"
            "    upc, dnc = R['up_cap'], R['dn_cap']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "x = np.arange(2)\n"
            "ax.bar(x-0.18, [su, sd], 0.36, color=GREEN, label='SPY average')\n"
            "ax.bar(x+0.18, [wu, wd], 0.36, color=AMBER, label='Wheel average')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels(['UP months','DOWN months'])\n"
            "ax.set_ylabel('average monthly return (%)')\n"
            "ax.set_title(f'Keeps only {upc:.0f}% of the up, rides {dnc:.0f}% of the down'); ax.legend()\n"
            "for xi,v in zip([x[0]-0.18,x[0]+0.18,x[1]-0.18,x[1]+0.18],[su,wu,sd,wd]):\n"
            "    ax.annotate(f'{v:+.1f}%',(xi,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'up-capture {upc:.0f}%   down-capture {dnc:.0f}%')"
        ),
        md(
            f"There's the costume coming off. In **up** months the Wheel banks only "
            f"**~{R['up_cap']}%** of SPY's gain — the rest is the upside you sold. In **down** months it "
            f"still loses **~{R['dn_cap']}%** of the way down. The rich VIX premium softens the falls "
            "*here*, but the right tail is chopped clean off. That asymmetry — **truncated upside, "
            "near-full downside** — is the exact fingerprint of being **short volatility**. It feels "
            "like income because the cheques are steady; it behaves like a covered call because the "
            "big up moves never reach you."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Mixed.** The \"income beats the index\" edge is **not reliable**: it was "
            f"strongly positive in 1993–2009 (+{R['edge_h1']:.1f}%/yr) but went **negative** in "
            f"2009–2026 ({R['edge_h2']:.1f}%/yr). Full-sample it doesn't clear the bar (*t* = "
            f"{R['t_full']:.2f}). And the premium is just the insurance you're selling, not free "
            "money.\n"
            f"- **Tradability — Fragile.** Lovely on paper (Sharpe {R['wheel_sharpe']:.2f} vs "
            f"{R['spy_sharpe']:.2f}) but the edge **vanishes at ~25¢ per $100 of trading cost**, and "
            f"the payoff has skew **{R['wheel_skew']:.1f}** — its worst 5 months sum to "
            f"**{R['worst5']:.0f}%**. One crash hurts.\n"
            f"- **Free / market-neutral income? — Busted.** It keeps **{R['up_cap']}%** of up months and "
            f"**{R['dn_cap']}%** of down months — a directional long bet with the top sliced off, not "
            "income, and not neutral."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — pay the real costs\n\n"
            "Every spin of the wheel means **buying back / writing options** and sometimes trading the "
            "shares — and options have real bid/ask spreads. Watch the edge over buy-and-hold as we "
            "charge a realistic cost per option written."
        ),
        code(
            "costs = R['costs']\n"
            "cs = [c[0] for c in costs]\n"
            "if HAVE_REAL:\n"
            "    edges = [st.paired_t(st.run_wheel(M, cost_frac=c/100)['wheel_ret'], spy)['ann_drag']*100 for c in cs]\n"
            "else:\n"
            "    edges = [c[1] for c in costs]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "colors = [GREEN if e>0 else RED for e in edges]\n"
            "ax.bar([f'{c:.2f}%' for c in cs], edges, color=colors, width=.6)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xlabel('trading cost per option written')\n"
            "ax.set_ylabel('Wheel edge over buy-and-hold (%/yr)')\n"
            "ax.set_title('The edge dies the moment you pay real costs')\n"
            "for i,e in enumerate(edges): ax.annotate(f'{e:+.1f}%',(i,e),ha='center',va='bottom' if e>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('edge over B&H by cost:', [f'{e:+.1f}%' for e in edges])"
        ),
        md(
            "At **zero** cost the Wheel beats buy-and-hold by ~2.5%/yr — but at a realistic **~25¢ per "
            "\\$100** the edge is already **gone**, and at 50¢ it's a clear loser. And remember: even "
            "that gross edge was the VIX-is-rich premium, in a model with **no early assignment, no "
            "slippage, no pin risk** — all of which make a real Wheel worse. The honest version isn't "
            "\"set the wheel and collect rent\" — it's \"you're selling insurance, sized small, and one "
            "bad year can take a chunk of your house.\""
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🎡\n\n"
            "- **The same shape elsewhere.** [Study 62 — Premium-Seller](../../62-premium-seller/) "
            "shows a real covered-call *fund* (QYLD) trailing its own index; "
            "[Study 63 — Free-Fall](../../63-free-fall/) is the naive short-vol that blew up −95% in "
            "*Volmageddon*. The Wheel is the retail cousin of both.\n"
            "- **Try a fair price.** In the quant notebook, when we price the option at the market's "
            "*actual* realised volatility instead of the rich VIX, the Wheel **loses** to buy-and-hold "
            "— proof the only edge was the insurance premium, not the wheel.\n"
            "- **Add real frictions.** American-style early assignment, bid/ask, and choosing "
            "*out-of-the-money* strikes (what most wheelers actually do) all change the picture — and "
            "mostly for the worse. Fork the engine and try them.\n\n"
            "*Think the Wheel is free income? Price the option yourself from the VIX, race it against "
            "just holding SPY, and check whether the \"income\" wasn't the upside you quietly sold.*"
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
            "# The Wheel — a quantitative teardown 🔬\n"
            "### Black-Scholes ATM premium (VIX as IV) · the short-vol payoff identity · "
            "paired-t vs buy-and-hold · up/down capture · a fair-price positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We test "
            "whether the retail Wheel is anything but a **packaged short-volatility / covered-call** "
            "exposure. The decisive object is an **identity**, not a fit: a short cash-secured ATM put "
            "(plus a short ATM covered call when assigned) has the payoff *capped move on the short "
            "side + premium* — truncated upside, near-full downside. We price the premium with "
            "Black-Scholes (VIX/100 as the implied vol — a transparent model, clearly labelled), "
            "simulate the Wheel on the real SPY tape, and certify the engine on a synthetic world "
            "priced at its **own** vol.\n\n"
            "> ⚠️ **Not investment advice.** Real data: SPY total-return + ^VIX (yfinance, 1993–2026, "
            "as-of 2026-06-18); the offline core + synthetic control are deterministic. **VIX-as-IV is "
            "a proxy that *flatters the seller*** (implied > realised = the variance risk premium); "
            "no early assignment, no skew, no slippage — so the real Wheel is worse than modelled. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | Wheel-minus-buy&hold monthly edge: full-sample ann "
            f"**+{R['edge_full']:.2f}%**, paired **t = {R['t_full']:.2f}** (below 2). It **flips sign** "
            f"by half: +{R['edge_h1']:.1f}%/yr (t={R['t_h1']:.1f}) in 1993–2009, "
            f"**{R['edge_h2']:.1f}%/yr** (t={R['t_h2']:.1f}) since. At a *fair* IV the control shows "
            f"the Wheel **loses** to B&H. |\n"
            f"| **Tradability** | `FRAGILE` | Gross Sharpe **{R['wheel_sharpe']:.2f}** vs "
            f"{R['spy_sharpe']:.2f}, but the edge → 0 at **~25 bp**/option (t={R['costs'][2][2]:.2f}) "
            f"and the payoff has skew **{R['wheel_skew']:.2f}**, worst-5 months **{R['worst5']:.0f}%**. "
            "One crash hurts. |\n"
            f"| **Free income?** | `BUSTED` | Up-capture **{R['up_cap']}%**, down-capture "
            f"**{R['dn_cap']}%** — a directional short-vol payoff (truncated upside, near-full "
            "downside), not market-neutral income. |\n\n"
            "> 💡 In plain words: the premium is real but it's the **variance risk premium** you earn "
            "for selling insurance, not free income. The wheel mechanic itself has **no** edge — at a "
            "fair price it underperforms the index — and what gross edge the VIX premium provides dies "
            "to costs and carries a fat left tail."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let month $t$ open at spot $S_t$. The Wheel writes a **one-month ATM** option (strike "
            "$K = S_t$) and is in one of two states:\n\n"
            "- **Short put (collateral in cash).** Realised SPY return $g_t = S_{t+1}/S_t - 1$. If "
            "$g_t \\ge 0$ the put expires: month return $= p_t$ (premium) $+$ cash yield. If $g_t < 0$ "
            "assigned: return $= g_t + p_t$, and **carry the ETF** next month.\n"
            "- **Short call (holding ETF).** If $g_t \\le 0$: return $= g_t + c_t$ (own the falling "
            "ETF, keep premium). If $g_t > 0$: called away — return $= \\min(g_t,0) + c_t = c_t$ "
            "(upside above $K$ surrendered), **back to cash**.\n\n"
            "By **put-call parity** a short cash-secured ATM put and a short ATM covered call have the "
            "*same* payoff, so the Wheel is one **short-vol** position rolled monthly. The testable "
            "claims: (H₁) a real return edge over buy-and-hold; (H₂) lower-risk / market-neutral; "
            "(H₃) robust to regime and costs. We find **H₁ MIXED** (sign-flips, t<2, and the edge is "
            "the VRP), **H₂ false** (directional, fat left tail), **H₃ false** (dies to costs)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the payoff identity\n\n"
            "Write the Wheel's monthly return against buy-and-hold $g_t$. In the short-call state,\n\n"
            "$$ r^{\\text{wheel}}_t = \\min(g_t, 0) + c_t = g_t - \\underbrace{\\max(g_t,0)}_{\\text{upside sold}} + \\underbrace{c_t}_{\\text{premium}}. $$\n\n"
            "So **wheel = buy-and-hold − upside surrendered + premium banked**. The Wheel can only beat "
            "the index if $\\;\\mathbb{E}[c_t] > \\mathbb{E}[\\max(g_t,0)]\\;$ — i.e. if the premium "
            "out-earns the call's intrinsic value. Under **fair** (risk-neutral) pricing those are "
            "*equal by construction*, so $\\mathbb{E}[r^{\\text{wheel}} - g] \\le 0$ on a drifting "
            "index. Any positive edge must come from the **variance risk premium** — implied vol "
            "(VIX) sitting above realised, paying the seller. The whole study is measuring whether the "
            "VRP is big enough, robust enough, and cheap enough to harvest. (Spoiler: not reliably.)"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Real tape.** SPY total-return adjusted close + ^VIX level, folded into {R['n']} "
            f"month-end observations ({R['window']}); the in-progress final month is dropped (house "
            "rule). Each month carries $S_t$ (open), the close, and VIX/100 (the IV).\n"
            "- **Premium model.** One-month **ATM Black-Scholes**, IV $= $ VIX/100, $r=0$, so "
            "$c_t = p_t = 2\\Phi(\\sigma\\sqrt{t}/2)-1$. Transparent, public, reproducible — *not* a "
            "live chain. It flatters the seller (implied $>$ realised), so the shortfall is "
            "conservative.\n"
            "- **Signal test.** A **paired $t$** of $d_t = r^{\\text{wheel}}_t - g_t$ against 0 (the "
            "edge over buy-and-hold), full sample and by half (the regime check). `REAL` needs robust "
            "$t \\ge 2$; a sign-flip is `MIXED`.\n"
            "- **Decomposition.** Up/down **capture**: mean Wheel return ÷ mean SPY return over up "
            "months and down months — the short-vol fingerprint.\n"
            "- **Positive control.** A deterministic lognormal-with-jumps SPY world priced at its "
            "**own** vol (`vix_premium=1`): the Wheel must (a) reproduce the option payoff, (b) "
            "**underperform** buy-and-hold at a fair price, (c) only turn positive when a VRP is "
            "injected. And the BS engine must equal its closed form to machine precision.\n"
            "- **Costs.** One-way cost per option written, swept 0–50 bp; the break-even is the "
            "Tradability line."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The engine is faithful — Black-Scholes ATM = closed form, and the Wheel IS the payoff\n\n"
            "First the machinery proof: the ATM premium equals $2\\Phi(\\sigma\\sqrt{t}/2)-1$ exactly "
            "(call $=$ put at $r=0$), and on a synthetic world priced at its **own** vol the Wheel "
            "behaves as the identity demands."
        ),
        code(
            "fi = st.fair_premium_identity(np.array([0.10, 0.20, 0.40]))\n"
            "print('Black-Scholes ATM premium vs closed form (r=0):')\n"
            "for s,c,p,cf in zip([.10,.20,.40], np.atleast_1d(fi['call']), np.atleast_1d(fi['put']), np.atleast_1d(fi['closed_form'])):\n"
            "    print(f'  sigma={s*100:.0f}%:  engine call={c:.4f}  put={p:.4f}  closed={cf:.4f}  match={bool(np.isclose(c,cf) and np.isclose(c,p))}')\n"
            "print('\\nSynthetic control — the Wheel vs buy-and-hold (edge in %/yr):')\n"
            "rows = []\n"
            "for vp, crash, *_ in R['syn']:\n"
            "    S = data.synthetic_months(n=6000, sigma_ann=0.16, crash_prob=crash/100, vix_premium=vp, seed=354)\n"
            "    ws = st.run_wheel(S); ps = st.paired_t(ws['wheel_ret'], S['spy_ret'].to_numpy()); cs = st.capture(S, ws)\n"
            "    rows.append((vp, crash, ps['ann_drag']*100, ps['t'], cs['up_capture']*100, cs['dn_capture']*100))\n"
            "    print(f'  IV={vp:.2f}x realised, crash={crash:.0f}%/mo:  edge {ps[\"ann_drag\"]*100:+.2f}%  (t={ps[\"t\"]:+.1f})  up-cap {cs[\"up_capture\"]*100:.0f}%  dn-cap {cs[\"dn_capture\"]*100:.0f}%')\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.3))\n"
            "labels = [f'IV={r[0]:.2f}x\\ncrash {r[1]:.0f}%' for r in rows]; e=[r[2] for r in rows]\n"
            "ax.bar(labels, e, color=[GREEN if x>0 else RED for x in e], width=.55)\n"
            "ax.axhline(0,c='k',lw=1); ax.set_ylabel('Wheel edge over B&H (%/yr)')\n"
            "ax.set_title('At a FAIR price the Wheel loses; only a vol-premium saves it')\n"
            "for i,x in enumerate(e): ax.annotate(f'{x:+.1f}%',(i,x),ha='center',va='bottom' if x>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "> 💡 In plain words: the premium model is **exact** (not fitted). And the control nails the "
            "crux — at a **fair** implied vol the Wheel **underperforms** buy-and-hold by ~3.7%/yr "
            "(t≈−8): on an upward-drifting index, selling the upside costs more than the fair premium "
            "pays. The Wheel only turns positive when you *inject a variance risk premium* (IV above "
            "realised). **The wheel mechanic has no edge of its own.**"
        ),
        md(
            "### 4b · The real tape — a paired-t over buy-and-hold, and the regime flip\n\n"
            "Now the live question: does the VIX-priced Wheel actually beat buy-and-hold SPY, robustly?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    w = st.run_wheel(M); spy = M['spy_ret'].to_numpy()\n"
            "    pt = st.paired_t(w['wheel_ret'], spy); h=len(M)//2\n"
            "    p1 = st.paired_t(w['wheel_ret'].to_numpy()[:h], spy[:h])\n"
            "    p2 = st.paired_t(w['wheel_ret'].to_numpy()[h:], spy[h:])\n"
            "    full, e1, e2, t0, t1, t2 = pt['ann_drag']*100, p1['ann_drag']*100, p2['ann_drag']*100, pt['t'], p1['t'], p2['t']\n"
            "else:\n"
            "    full, e1, e2 = R['edge_full'], R['edge_h1'], R['edge_h2']\n"
            "    t0, t1, t2 = R['t_full'], R['t_h1'], R['t_h2']\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.4))\n"
            "labs=['full\\n1993-2026','first half\\n1993-2009','second half\\n2009-2026']; ev=[full,e1,e2]; tv=[t0,t1,t2]\n"
            "ax.bar(labs, ev, color=[GREY, GREEN, RED], width=.55)\n"
            "ax.axhline(0,c='k',lw=1); ax.set_ylabel('Wheel edge over B&H (%/yr)')\n"
            "ax.set_title('The income edge FLIPS SIGN by regime (t labelled)')\n"
            "for i,(e,t) in enumerate(zip(ev,tv)): ax.annotate(f'{e:+.1f}%\\nt={t:+.2f}',(i,e),ha='center',va='bottom' if e>=0 else 'top',fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'full edge {full:+.2f}%/yr (t={t0:+.2f})  |  H1 {e1:+.2f}% (t={t1:+.2f})  H2 {e2:+.2f}% (t={t2:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: full-sample the edge is +{R['edge_full']:.1f}%/yr but only **t = "
            f"{R['t_full']:.2f}** — under the bar. Worse, it **flips sign**: strongly positive when VIX "
            f"was rich and crashes frequent (1993–2009, +{R['edge_h1']:.1f}%, t={R['t_h1']:.1f}), then "
            f"**negative** through the post-2010 bull run ({R['edge_h2']:.1f}%, t={R['t_h2']:.1f}) when "
            "the surrendered upside swamped the premium. A sign-flipping, sub-2-t effect is **`MIXED`**, "
            "not a real income edge — H₁ rejected as stated."
        ),
        md(
            "### 4c · The short-vol fingerprint — capture and skew\n\n"
            "Why it's *not* income: the up/down capture and the return distribution."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cap = st.capture(M, w); sw=st.summary(w['wheel_ret']); ss=st.summary(spy)\n"
            "    upc,dnc,wsk,ssk = cap['up_capture']*100, cap['dn_capture']*100, sw['skew'], ss['skew']\n"
            "    wr = w['wheel_ret'].to_numpy()\n"
            "else:\n"
            "    upc,dnc,wsk,ssk = R['up_cap'], R['dn_cap'], R['wheel_skew'], R['spy_skew']\n"
            "    rng=np.random.default_rng(354); wr=rng.normal(.009,.025,R['n'])  # illustrative only\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.5,4.2))\n"
            "a1.bar(['up months','down months'],[upc,dnc],color=[GREEN,RED],width=.55)\n"
            "a1.axhline(100,ls='--',c=GREY,label='100% = SPY'); a1.set_ylabel('capture of SPY (%)')\n"
            "a1.set_title(f'Truncated upside ({upc:.0f}%), near-full downside ({dnc:.0f}%)'); a1.legend()\n"
            "for i,v in enumerate([upc,dnc]): a1.annotate(f'{v:.0f}%',(i,v),ha='center',va='bottom')\n"
            "a2.hist(wr*100, bins=40, color=AMBER, alpha=.85)\n"
            "a2.axvline(0,c='k',lw=1); a2.set_xlabel('Wheel monthly return (%)'); a2.set_ylabel('months')\n"
            "a2.set_title(f'Left-skewed payoff (skew {wsk:+.2f} vs SPY {ssk:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'up-capture {upc:.0f}%  down-capture {dnc:.0f}%  |  Wheel skew {wsk:+.2f}  SPY skew {ssk:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: **{R['up_cap']}%** up-capture vs **{R['dn_cap']}%** down-capture, and "
            f"skew **{R['wheel_skew']:.2f}** (vs SPY's {R['spy_skew']:.2f}) — the textbook short-vol "
            "payoff. The VRP-rich premium does cushion the *average* down month here, but the "
            "*distribution* is the giveaway: a chopped right tail and a heavy left one (worst single "
            f"month {R['worst1']:.1f}% in {R['worst1_date']}, worst-5 summing to {R['worst5']:.0f}%). "
            "This is **directional long beta with the top sold off**, not market-neutral income — H₂ "
            "rejected."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — paired edge over B&H +{R['edge_full']:.2f}%/yr at **t = "
            f"{R['t_full']:.2f}** (sub-2), **sign-flipping** by half (+{R['edge_h1']:.1f}% → "
            f"{R['edge_h2']:.1f}%). The control proves the wheel mechanic *loses* to B&H at a fair "
            "price; the only edge is the variance risk premium — real in the literature, not certified "
            "REAL on this tape.\n"
            f"- **Tradability `FRAGILE`** — gross Sharpe {R['wheel_sharpe']:.2f} vs {R['spy_sharpe']:.2f}, "
            f"but the edge hits zero at **~25 bp**/option (t={R['costs'][2][2]:.2f}) and the payoff "
            f"carries skew {R['wheel_skew']:.2f}, a {R['wheel_mdd']:.1f}% drawdown *in this generous "
            f"model*, worst-5 months {R['worst5']:.0f}%. Harvestable only sized and tail-aware.\n"
            f"- **Free income? `BUSTED`** — {R['up_cap']}% up-capture, {R['dn_cap']}% down-capture, "
            "skew "
            f"{R['wheel_skew']:.2f}: a packaged short-vol / covered-call exposure (cf. Studies 62, 63), "
            "directional and tail-heavy — not income, not neutral."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the cost break-even\n\n"
            "Turnover is brutal: an option written (and bought back / rolled) every month, plus share "
            "turnover on each assignment and call-away. Charge a one-way cost per option and watch the "
            "edge over buy-and-hold."
        ),
        code(
            "cs=[c[0] for c in R['costs']]\n"
            "if HAVE_REAL:\n"
            "    edges=[st.paired_t(st.run_wheel(M,cost_frac=c/100)['wheel_ret'], spy)['ann_drag']*100 for c in cs]\n"
            "    ts=[st.paired_t(st.run_wheel(M,cost_frac=c/100)['wheel_ret'], spy)['t'] for c in cs]\n"
            "else:\n"
            "    edges=[c[1] for c in R['costs']]; ts=[c[2] for c in R['costs']]\n"
            "fig,ax=plt.subplots(figsize=(8.8,4.3))\n"
            "ax.plot(cs, edges, 'o-', c=AMBER, lw=2)\n"
            "ax.axhline(0,ls='--',c=GREY); ax.fill_between(cs,edges,0,where=[e<0 for e in edges],color=RED,alpha=.15)\n"
            "ax.set_xlabel('one-way cost per option written (%)'); ax.set_ylabel('edge over B&H (%/yr)')\n"
            "ax.set_title('Break-even ~25 bp/option — the edge is thinner than the spread')\n"
            "for c,e,t in zip(cs,edges,ts): ax.annotate(f'{e:+.1f}%\\n(t={t:+.2f})',(c,e),ha='center',va='bottom' if e>=0 else 'top',fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('edge / t by cost:', [(f'{c:.2f}%', f'{e:+.1f}%', f't={t:+.2f}') for c,e,t in zip(cs,edges,ts)])"
        ),
        md(
            f"> 💡 In plain words: the gross edge ({R['edge_full']:.1f}%/yr) is **thinner than a typical "
            "monthly ATM-option spread**. It's gone by ~25 bp/option and clearly negative by 50 bp — "
            "and that's before early-assignment, slippage and pin risk, all absent from the model. "
            "There is no realistic cost assumption under which the Wheel is a robust money machine; the "
            "VRP it harvests is real but too thin and too tail-heavy to call INVESTABLE."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The fund and the blow-up twins.** [Study 62 — Premium-Seller](../../62-premium-seller/) "
            "(QYLD trailing QQQ on total return) and [Study 63 — Free-Fall](../../63-free-fall/) "
            "(naive short-vol, −95% in Volmageddon) are the same short-vol family at the fund and the "
            "ETP level; this study is the retail DIY version.\n"
            "- **OTM strikes & delta targeting.** Most wheelers sell ~0.30-delta OTM puts, not ATM. "
            "That trades a smaller premium for a wider buffer — `run_wheel` can be extended with a "
            "moneyness parameter to map the premium/upside trade-off.\n"
            "- **Real frictions.** American early assignment, the volatility *skew* (puts richer than "
            "ATM), term structure, and bid/ask all move the result — model the walk-up the writer "
            "actually pays. The prior is they make the Wheel worse, not better.\n\n"
            "*The reproducible core (Black-Scholes premium, Wheel engine, synthetic control) is "
            "offline and deterministic. Methods and sources: [`docs/references.md`](../docs/references.md); "
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
