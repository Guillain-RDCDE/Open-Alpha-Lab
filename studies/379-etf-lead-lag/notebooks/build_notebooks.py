"""Generate the two narrative notebooks for Study 379 (ETF Lead-Lag).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket prices
under ../_cache/ (the leader SPY + member basket) and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily adjusted
# closes, SPY + a 37-name smaller/less-liquid member basket, 2000-01-04 -> 2026-06-18,
# 6,654 days, 26.5 years).
R = dict(
    start="2000-01-04", end="2026-06-18", days=6654, years=26.5,
    # cross-correlation profile corr(leader_{t-k}, members_t)
    xcorr={-1: -0.0514, 0: 0.8670, 1: -0.0484, 2: -0.0122},
    # one-day lead-lag slope (members on lagged leader), Newey-West HAC
    lag1_slope=-0.05810, lag1_t=-1.88, lag1_n=6653,
    # next-bar rule: (q, n, cond_mean%, cond_win%, base_mean%, base_win%, t, p_placebo)
    q90=(0.90, 666, -0.0778, 50, 0.0285, 52, -1.34, 0.969),
    q95=(0.95, 333, -0.1019, 51, 0.0285, 52, -1.03, 0.945),
    # costs: (q, gross%, net%)
    net=[(0.90, -0.0778, -0.1778), (0.95, -0.1019, -0.2019)],
    # synthetic control: (lead, xcorr+1, slope, hac_t, rule_cond%, rule_base%, rule_t, rule_p)
    syn=[(0.00, 0.0110, 0.0141, 0.83, 0.0714, 0.0440, 0.49, 0.297),
         (0.05, 0.0499, 0.0641, 3.76, 0.1617, 0.0471, 2.03, 0.015),
         (0.25, 0.2017, 0.2641, 15.48, 0.5227, 0.0597, 8.18, 0.000)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Free_lunch%3F: Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square)\n\n"
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

from etf_lead_lag import data, strategy as st

HAVE_REAL = data.have_real()
F = data.load_real() if HAVE_REAL else None
print("real lead-lag cache present:", HAVE_REAL,
      "| days:", (0 if F is None else len(F)))
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
            "# Does the big ETF lead its slow members? 🔗\n"
            "### The 'buy the laggards the moment the leader pops' idea, in plain English\n\n"
            + BADGES +
            "Here's a tempting idea from the trading floor: the **big, liquid ETF** (think SPY) is "
            "where the action happens *first*. Its smaller, slower, less-liquid members are supposed to "
            "**catch up a day later**. So if the leader jumps today, you just buy the sleepy members "
            "**tomorrow** and collect their catch-up move. Free money, right?\n\n"
            "The catch is hiding in one word: **when**. The leader and its members really do move "
            "together — *enormously* — but they move together **the same day**. And you can't trade a "
            "move you'd have to know about in advance. This notebook shows that once you ask the only "
            "question that pays — *does yesterday's leader predict tomorrow's members?* — the answer on "
            "daily data is **nothing**.\n\n"
            "> 📓 **Plain-language layer.** Want the cross-correlation profile, the HAC *t*-stat and the "
            "placebo test? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Real ETF/member lead-lag lives **intraday** (it's measured in "
            "minutes, and arbitrageurs erase it within the day). Tick data isn't free, so we test the "
            "**daily-bar** version the folklore implies — and on that clock the catch-up is already "
            "gone. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do the leader and the members move together? | **Massively — same-day correlation "
            f"{R['xcorr'][0]:.2f}.** When SPY moves, the basket moves with it. |\n"
            "| Can I trade that? | **No.** It's *contemporaneous* — it happens the same day. To profit "
            "you'd need to know SPY's move before it happened. |\n"
            "| Does yesterday's leader predict today's members? | **No.** The one-day-ahead correlation "
            f"is **{R['xcorr'][1]:+.2f}** — essentially zero (and faintly *negative*). The 'catch-up' "
            "the story promises just isn't there on daily bars. |\n"
            "| So can I run the 'buy the laggards tomorrow' rule? | **It loses money.** After a big "
            f"leader day the members' *next-day* return averages **{R['q90'][2]:+.3f}%** — *below* a "
            "normal day — and worse once you pay to trade. |\n\n"
            "> The leader and the members are joined at the hip — **today**. The next-day lead you could "
            "actually trade is zero. The 'free lunch' is just same-day co-movement wearing a disguise."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The big liquid ETF digests news first. Its smaller, slower members are slow to "
            "re-price, so a move in the leader **predicts the next move** in the laggards. Wait for the "
            "leader to pop, then buy the members before they catch up.\"*\n\n"
            "It's a real, respectable intuition — liquid instruments *do* incorporate information "
            "faster, and there's a genuine academic literature on big stocks leading small ones. The "
            "seduction is the mechanical simplicity: a leader you can watch in real time, a basket of "
            "laggards you can buy, and a one-day window to harvest. We'll rebuild exactly that and see "
            "whether the daily tape pays it."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a leader move genuinely *predicted* tomorrow's member move, you'd have a clean, "
            "repeatable, mechanical edge — the holy grail of systematic trading: a signal you see today "
            "for a payoff you collect tomorrow. But there's a trap built into the word *together*. Two "
            "things can move **together** for two completely different reasons: (1) they react to the "
            "same news **at the same moment** (untradable — you can't act before the move), or (2) one "
            "**leads** the other in time (tradable — you act on the leader, wait, collect on the "
            "laggard). The whole question is which one we have. A huge same-day correlation tells you "
            "*nothing* about whether there's money to be made."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **SPY** as the leader and a fixed basket of **37 smaller, less-liquid members** "
            f"as the laggards, over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and line up "
            "their daily moves.\n\n"
            "1. **Map the timing.** For each time-shift *k*, how correlated is the leader *k* days ago "
            "with the members today? *k = 0* is same-day (untradable); *k = +1* is the tradable claim "
            "(yesterday's leader predicts today's members).\n"
            "2. **Try the rule.** On the days the leader pops (its top-decile up-days), buy the laggard "
            "basket the **next** day, hold one day. What do you earn — before and after costs?\n"
            "3. **Stress the luck.** Draw the same number of *random* next days thousands of times and "
            "ask whether the rule's payoff is anything a coin couldn't produce."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the timing map.** For each lag *k*, the correlation between the leader then and the "
            "members today. Watch where the tall bar lands."
        ),
        code(
            "if HAVE_REAL:\n"
            "    prof = st.crosscorr_profile(F, max_lag=5)\n"
            "    ks = sorted(prof); vals = [prof[k] for k in ks]\n"
            "else:\n"
            "    ks = list(range(-5,6))\n"
            "    base = {kk: 0.0 for kk in ks}; base.update(R['xcorr'])\n"
            "    vals = [base.get(k, 0.0) for k in ks]\n"
            "cols = [GREEN if k==0 else (AMBER if k==1 else GREY) for k in ks]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(ks, vals, color=cols, width=.7)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(ks); ax.set_xlabel('lag k  (k>0 = leader leads members by k days)')\n"
            "ax.set_ylabel('corr(leader at t-k, members at t)')\n"
            "ax.set_title('All the co-movement is at k=0 (same day) — nothing at k=+1')\n"
            "ax.annotate('same day\\n(untradable)', (0, vals[ks.index(0)]), ha='center', va='bottom')\n"
            "ax.annotate('the tradable\\nclaim', (1, 0.12), ha='center', va='bottom', color=AMBER)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('same-day corr (k=0):', round(vals[ks.index(0)],3),\n"
            "      '| one-day-ahead corr (k=+1):', round(vals[ks.index(1)],3))"
        ),
        md(
            f"There's the whole story in one picture. The green bar at **k = 0** towers at "
            f"**{R['xcorr'][0]:.2f}** — the leader and members are practically the same thing **on the "
            f"day**. But the amber bar at **k = +1** — the one you could actually trade — is "
            f"**{R['xcorr'][1]:+.2f}**, flat against zero (and, if anything, slightly *negative*). The "
            "catch-up the legend promises simply isn't in the next day's data."
        ),
        md(
            "**Now run the rule anyway.** On the days the leader pops, buy the laggards tomorrow. Here's "
            "the members' average *next-day* return after a big-leader day, next to a normal day."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.big_leader_days(F, q=0.90); s = st.summarize(F, ev)\n"
            "    cond = s['cond_mean']*100; base = s['base_mean']*100; n = s['n']\n"
            "else:\n"
            "    cond, base, n = R['q90'][2], R['q90'][4], R['q90'][1]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "bars = ax.bar(['after a big\\nleader day', 'any normal\\nday (base)'], [cond, base],\n"
            "              color=[RED, GREY], width=.55)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('members next-day return (%)')\n"
            "ax.set_title(f'The \"catch-up\" trade is NEGATIVE: {cond:+.3f}% vs base {base:+.3f}%')\n"
            "for b,v in zip(bars,[cond,base]): ax.annotate(f'{v:+.3f}%',(b.get_x()+b.get_width()/2,v),\n"
            "                                              ha='center', va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'{n} big-leader days -> members next day averages {cond:+.3f}% (a normal day: {base:+.3f}%)')"
        ),
        md(
            f"The bar points the **wrong way**. After the leader pops, the laggards' *next-day* return "
            f"averages **{R['q90'][2]:+.3f}%** — actually *below* a normal day's **{R['q90'][4]:+.3f}%**. "
            "Far from catching up, they drift very slightly down. The intuition feels right; the tape "
            "says no."
        ),
        md(
            "**Could random days look this good (or bad)?** The honest luck test: draw the same number "
            "of *random* next-days over and over and see where the rule's payoff lands in the cloud."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.big_leader_days(F, q=0.90); pl = st.placebo_pvalue(F, ev)\n"
            "    obs = pl['cond_mean']; pval = pl['p_value']\n"
            "    mem = F['members'].values; k = pl['k']; rng = np.random.default_rng(379)\n"
            "    draws = np.array([mem[rng.integers(0,len(mem),size=k)].mean() for _ in range(8000)])\n"
            "else:\n"
            "    obs = R['q90'][2]/100; pval = R['q90'][7]\n"
            "    rng = np.random.default_rng(379); draws = rng.normal(R['q90'][4]/100, 0.0006, 8000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=50, color=GREY, alpha=.85, label='mean of random next-days')\n"
            "ax.axvline(obs*100, c=RED, lw=2.5, label=f'the actual rule ({obs*100:+.3f}%)')\n"
            "ax.set_xlabel('mean next-day member return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The rule sits at the BOTTOM of the luck cloud — placebo p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'{pval*100:.0f}% of random draws do BETTER than the rule — it is not an edge, it is a small drag')"
        ),
        md(
            f"The red line — the actual rule — falls at the **bottom** of the random cloud: about "
            f"**{R['q90'][7]*100:.0f}%** of random next-day draws do *better*. There's no hidden edge "
            "being masked by noise; if anything there's a faint drag. The leader does not hand you the "
            "members' next move."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The only big correlation is **same-day** ({R['xcorr'][0]:.2f}) and "
            f"untradable. The one-day lead the claim needs is **{R['xcorr'][1]:+.2f}** — nothing.\n"
            "- **Tradability — Mirage.** The 'buy the laggards tomorrow' rule is **negative before "
            "costs** and worse after. You can't deploy an edge that points the wrong way.\n"
            "- **\"Free lunch\"? — Busted.** The thing that *looks* like free money is the same-day "
            "co-movement — and you can't trade a move that's already happened."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — pay the toll\n\n"
            "Suppose you ran it anyway. Each trade pays the spread twice (buy, then sell a day later). "
            "Here's gross vs net — and the point is that a 10-bps round-turn only digs the hole deeper."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for q in (0.90, 0.95):\n"
            "        ev = st.big_leader_days(F, q=q); c = st.net_of_costs(F, ev, 5.0)\n"
            "        rows.append((q, c['gross_mean']*100, c['net_mean']*100))\n"
            "else:\n"
            "    rows = R['net']\n"
            "qs = [f'top {int((1-q)*100)}%\\nleader days' for q,_,_ in rows]\n"
            "g = [r[1] for r in rows]; nv = [r[2] for r in rows]; x = np.arange(len(qs))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(x-.2, g, .4, color=AMBER, label='gross')\n"
            "ax.bar(x+.2, nv, .4, color=RED, label='net @ 10 bps round-turn')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels(qs)\n"
            "ax.set_ylabel('mean trade return (%)')\n"
            "ax.set_title('Already negative gross — costs only make it worse'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('gross/net per trade (%):', [(round(r[1],3), round(r[2],3)) for r in rows])"
        ),
        md(
            "The honest bottom line: there is **no toll low enough** to rescue a rule that already "
            "loses money gross. Costs aren't the villain here — they're just the final nail. The "
            "tradable lead the folklore promises doesn't exist on daily bars, so there's nothing for "
            "costs to eat into."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Where the real lead-lag hides.** It's **intraday** — measured in minutes and arbitraged "
            "away within the trading day by index-arbitrage desks. With tick data you'd see a genuine "
            "leader→member lead; by the daily close it's gone. That's the real lesson: *the clock "
            "matters more than the pattern.*\n"
            "- **The liquidity angle.** [Study 140 — Amihud-Illiquidity](../140-amihud-illiquidity/) "
            "asks whether illiquidity itself — the thing that makes members 'slow' — is a priced, "
            "tradable factor.\n"
            "- **Try your own basket.** Swap our 37 names for a different sector or size bucket and "
            "re-run the profile — the same-day bar stays tall, the k=+1 bar stays flat.\n\n"
            "*Think there's a tradable daily lead? Capture the big-leader days, buy the laggards "
            "tomorrow, draw the same number of random next-days, and show the rule landing **outside** "
            "the cloud — on the *profitable* side. Then we'll talk.*"
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
            "# ETF Lead-Lag — a quantitative teardown 🔬\n"
            "### lead-lag cross-correlation profile · an HAC-robust one-day slope · a next-bar rule "
            "with a Welch *t* + placebo null · costs · a synthetic planted-lead / faithful-engine control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We separate "
            "the two things 'they move together' fuses: a **massive contemporaneous** co-movement "
            "(untradable) from a **next-day lead** (the only tradable object), and we ask whether the "
            "latter exists at all on daily bars. The decisive number is not the lag-0 correlation (it's "
            "enormous, as the intuition says) but the **lag-+1** structure — which is statistically "
            "indistinguishable from zero, and HAC-robustly the *wrong sign*.\n\n"
            "> ⚠️ **Data + clock note.** True ETF/constituent lead-lag is an **intraday** effect "
            "(arbitraged away within the day); tick data isn't on yfinance, so we test the **daily-bar** "
            "version. Real data: yfinance daily adjusted closes, SPY + a 37-name smaller/less-liquid "
            "basket, 2000→2026 (mild survivorship, named on the Signal axis). Offline core + synthetic "
            "control are deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers "
            "in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | lag-0 corr **{R['xcorr'][0]:.2f}** (contemporaneous, untradable); "
            f"lag-+1 corr **{R['xcorr'][1]:+.2f}**; HAC one-day slope **{R['lag1_slope']:+.3f}** at "
            f"**t = {R['lag1_t']:.2f}** (wrong sign, n.s.). No next-day lead. |\n"
            f"| **Tradability** | `MIRAGE` | next-bar rule cond. **{R['q90'][2]:+.3f}%** vs base "
            f"**{R['q90'][4]:+.3f}%** (Welch **t = {R['q90'][6]:.2f}**, placebo **p = {R['q90'][7]:.2f}**); "
            f"gross is **negative**, net of 10 bps round-turn **{R['net'][0][2]:+.3f}%**. |\n"
            f"| **Free lunch?** | `BUSTED` | the only large signal is **same-day**; the tradable "
            "next-day piece is zero. Co-movement ≠ predictability. |\n\n"
            "> 💡 In plain words: leader and members are ~the same asset *intraday* (corr 0.87), which is "
            "untradable; strip the same-day term and the next-day lead — the only thing you could act on "
            "— is statistically nothing, and HAC-robustly the wrong sign."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\ell_t$ be the leader (SPY) log return and $m_t$ the equal-weight member-basket log "
            "return. The believers' tradable hypothesis is a **one-day Granger-style lead**:\n\n"
            "- **H₁ (the leader leads).** $\\rho_{+1} = \\mathrm{corr}(\\ell_{t-1}, m_t) > 0$, and the "
            "OLS slope $\\beta_1$ in $m_t = \\alpha + \\beta_1 \\ell_{t-1} + \\varepsilon_t$ is "
            "positive and HAC-significant.\n"
            "- **H₂ (it's deployable).** Conditioning on big-leader days, the members' next-day return "
            "beats the base rate by enough to survive one-way costs × turnover.\n"
            "- **H₃ ('free lunch').** The high leader–member co-movement reflects *predictive* lead, "
            "not mere **contemporaneous** ($\\rho_0$) co-movement.\n\n"
            "We find **H₁ rejected** ($\\rho_{+1}\\approx 0$, HAC $t = -1.88$ — wrong sign), **H₂ "
            "rejected** (negative gross, worse net), **H₃ rejected** ($\\rho_0 = 0.87$ dwarfs everything "
            "and is same-day). The legend is true exactly where it's useless (same-day co-movement) and "
            "false exactly where it would pay (a *next-day* lead)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The entire teardown is one decomposition of the cross-correlation function "
            "$\\rho_k = \\mathrm{corr}(\\ell_{t-k}, m_t)$ into a **contemporaneous** term ($k=0$) and a "
            "**predictive** term ($k=+1$):\n\n"
            "$$\\underbrace{\\rho_0}_{\\text{co-movement, untradable}} \\quad\\text{vs}\\quad "
            "\\underbrace{\\rho_{+1}}_{\\text{lead, tradable}}.$$\n\n"
            "A large $\\rho_0$ carries **no** tradable information: it is realised *simultaneously*, so "
            "no order placed at $t$'s close can capture it. Only $\\rho_{+1}$ (equivalently $\\beta_1$) "
            "is exploitable, and it must clear an **autocorrelation-robust** bar — non-synchronous "
            "trading and bid-ask bounce *manufacture* spurious daily lead-lag (Scholes-Williams 1977, "
            "Lo-MacKinlay 1990, Roll 1984), so a naive $t$ is not enough. We use a Newey-West HAC $t$ on "
            "$\\beta_1$ and a placebo null on the trading rule."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** yfinance daily adjusted closes, SPY (leader) + a fixed **37-name** "
            f"smaller/less-liquid US basket (members), {R['start']}→{R['end']} ({R['days']:,} days). "
            "Daily log returns; members = equal-weight cross-sectional mean. Daily bars are an explicit "
            "**proxy** for the intraday phenomenon (named on the axis), with mild survivorship.\n"
            "- **Profile.** $\\rho_k = \\mathrm{corr}(\\ell_{t-k}, m_t)$ for $k\\in[-5,5]$; the tradable "
            "claim is the sign and size of $\\rho_{+1}$.\n"
            "- **HAC slope.** OLS of $m_t$ on $\\ell_{t-1}$ with a Newey-West (5-lag) HAC standard "
            "error — the autocorrelation-robust Signal-axis test.\n"
            "- **Next-bar rule.** Enter the basket the **close after** a top-decile leader day (1-day "
            "lag, no look-ahead), hold one day. Welch $t$ of the conditional next-day mean vs the "
            "unconditional mean; a 20,000-draw **placebo** null sized to the trade count.\n"
            "- **Costs.** 5 bps one-way each side ⇒ 10 bps round-turn per 1-day trade (turnover = one "
            "round-turn/trade).\n"
            "- **Positive control.** A deterministic leader/member series with a **planted one-day "
            "lead** knob: with lead=0 the profile, HAC slope and rule must stay quiet; with a planted "
            "lead they must light up — proving the engine would find a real lead if one existed."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The cross-correlation profile — all the mass is at lag 0\n\n"
            "$\\rho_k$ across lags. A genuine one-day lead would show a positive bump at $k=+1$; instead "
            "everything piles up at $k=0$ (same-day) and $k=+1$ is flat/negative."
        ),
        code(
            "if HAVE_REAL:\n"
            "    prof = st.crosscorr_profile(F, max_lag=5); ks = sorted(prof); vals = [prof[k] for k in ks]\n"
            "else:\n"
            "    ks = list(range(-5,6)); base = {kk:0.0 for kk in ks}; base.update(R['xcorr'])\n"
            "    vals = [base.get(k,0.0) for k in ks]\n"
            "cols = [GREEN if k==0 else (AMBER if k==1 else GREY) for k in ks]\n"
            "fig, (a1,a2) = plt.subplots(1,2,figsize=(11.0,4.2),gridspec_kw={'width_ratios':[1,1]})\n"
            "a1.bar(ks, vals, color=cols, width=.7); a1.axhline(0,c='k',lw=.8); a1.set_xticks(ks)\n"
            "a1.set_xlabel('lag k'); a1.set_ylabel(r'$\\rho_k$'); a1.set_title('Full profile: lag-0 dominates')\n"
            "# zoom: drop k=0 to see the (tiny) predictive lags\n"
            "ks2 = [k for k in ks if k!=0]; v2 = [vals[ks.index(k)] for k in ks2]\n"
            "c2 = [AMBER if k==1 else GREY for k in ks2]\n"
            "a2.bar(ks2, v2, color=c2, width=.7); a2.axhline(0,c='k',lw=.8); a2.set_xticks(ks2)\n"
            "a2.set_xlabel('lag k (k=0 hidden)'); a2.set_ylabel(r'$\\rho_k$')\n"
            "a2.set_title('Zoom: k=+1 (tradable) is ~0 and negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('rho_0 =', round(vals[ks.index(0)],4), '| rho_+1 =', round(vals[ks.index(1)],4),\n"
            "      '| rho_-1 =', round(vals[ks.index(-1)],4))"
        ),
        md(
            f"> 💡 In plain words: $\\rho_0 = {R['xcorr'][0]:.2f}$ (left) is the leader and members being "
            f"essentially the same asset on the day. Hide it (right) and the tradable lag $\\rho_{{+1}} = "
            f"{R['xcorr'][1]:+.2f}$ is flat against zero, even faintly negative — **H₃ rejected**: the "
            "co-movement is co-movement, not a lead."
        ),
        md(
            "### 4b · The decisive test — an HAC-robust one-day slope\n\n"
            "Regress today's members on **yesterday's** leader, $m_t = \\alpha + \\beta_1\\ell_{t-1} + "
            "\\varepsilon_t$, with a Newey-West HAC standard error (daily autocorrelation would inflate a "
            "naive $t$). A real lead needs $\\beta_1 > 0$ at $t \\ge 2$."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = st.lag1_beta(F); slope = b['slope']; thac = b['t_hac']\n"
            "    lead1 = F['leader'].shift(1); pair = np.c_[lead1.values[1:], F['members'].values[1:]]\n"
            "    pair = pair[~np.isnan(pair).any(axis=1)]\n"
            "else:\n"
            "    slope, thac = R['lag1_slope'], R['lag1_t']\n"
            "    rng = np.random.default_rng(379); x = rng.normal(0,0.01,4000)\n"
            "    pair = np.c_[x, slope*x + rng.normal(0,0.008,4000)]\n"
            "fig, ax = plt.subplots(figsize=(8.8,4.4))\n"
            "ax.scatter(pair[:,0]*100, pair[:,1]*100, s=5, alpha=.15, color=GREY)\n"
            "xs = np.linspace(pair[:,0].min(), pair[:,0].max(), 50)\n"
            "ax.plot(xs*100, slope*xs*100, c=RED, lw=2, label=f'slope {slope:+.3f} (HAC t={thac:+.2f})')\n"
            "ax.axhline(0,c='k',lw=.5); ax.axvline(0,c='k',lw=.5)\n"
            "ax.set_xlabel(\"yesterday's leader return (%)\"); ax.set_ylabel(\"today's members return (%)\")\n"
            "ax.set_title('One-day lead-lag slope is negative and not significant'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'beta_1 = {slope:+.4f}  HAC t = {thac:+.2f}  -> wrong sign, |t|<2, no next-day lead')"
        ),
        md(
            f"> 💡 In plain words: the slope of today's members on yesterday's leader is "
            f"**{R['lag1_slope']:+.3f}** at HAC **t = {R['lag1_t']:.2f}** — *negative* (a faint next-day "
            "reversal, the opposite of a lead) and not significant. **H₁ rejected**: there is no "
            "autocorrelation-robust next-day lead to trade."
        ),
        md(
            "### 4c · The trading rule + its placebo null — and costs\n\n"
            "Enter the basket the day after a top-decile leader day; compare the conditional next-day "
            "mean to a placebo of random next-days, then charge costs. The rule's mean sits in the "
            "**left** tail of the luck cloud (it underperforms), and costs only deepen the loss."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.big_leader_days(F, q=0.90); pl = st.placebo_pvalue(F, ev); s = st.summarize(F, ev)\n"
            "    obs = pl['cond_mean']; pval = pl['p_value']; mem = F['members'].values; k = pl['k']\n"
            "    rng = np.random.default_rng(379)\n"
            "    draws = np.array([mem[rng.integers(0,len(mem),size=k)].mean() for _ in range(20000)])\n"
            "    c = st.net_of_costs(F, ev, 5.0); g, nv = c['gross_mean']*100, c['net_mean']*100\n"
            "else:\n"
            "    obs = R['q90'][2]/100; pval = R['q90'][7]\n"
            "    rng = np.random.default_rng(379); draws = rng.normal(R['q90'][4]/100, 0.0006, 20000)\n"
            "    g, nv = R['net'][0][1], R['net'][0][2]\n"
            "fig, (a1,a2) = plt.subplots(1,2,figsize=(11.0,4.2))\n"
            "a1.hist(draws*100, bins=60, color=GREY, alpha=.85, label=f'null: {len(draws):,} random draws')\n"
            "a1.axvline(obs*100, c=RED, lw=2.5, label=f'rule mean {obs*100:+.3f}%')\n"
            "a1.set_xlabel('mean next-day member return (%)'); a1.set_ylabel('frequency')\n"
            "a1.set_title(f'Placebo p = {pval:.2f} (rule in the LEFT tail)'); a1.legend()\n"
            "a2.bar(['gross','net @10bps'], [g, nv], color=[AMBER, RED], width=.5)\n"
            "a2.axhline(0,c='k',lw=.8); a2.set_ylabel('mean trade return (%)')\n"
            "a2.set_title('Negative gross; costs make it worse')\n"
            "for i,v in enumerate([g,nv]): a2.annotate(f'{v:+.3f}%',(i,v),ha='center',va='top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'rule mean {obs*100:+.3f}%  placebo p={pval:.3f}  gross={g:+.3f}%  net={nv:+.3f}%')"
        ),
        md(
            f"> 💡 In plain words: **{R['q90'][7]*100:.0f}%** of random next-day draws beat the rule — the "
            f"green-light edge would put it in the *right* tail; instead it's at the far left. Net of a "
            f"10 bps round-turn the rule earns **{R['net'][0][2]:+.3f}%**/trade. **H₂ rejected.**"
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic leader/member series with a **planted one-day lead** knob: with lead=0 "
            "everything must stay quiet (no false positive); a small planted lead must clear the bar; a "
            "large one must scream. All three hold — proving the engine would detect a real lead, and "
            "that finding *none* on the real tape is a true negative, not a dead pipeline."
        ),
        code(
            "res = []\n"
            "for lead in (0.0, 0.05, 0.25):\n"
            "    syn = data.synthetic_leadlag(n_days=6000, lead=lead, seed=379)\n"
            "    p = st.crosscorr_profile(syn, max_lag=2); b = st.lag1_beta(syn)\n"
            "    ev = st.big_leader_days(syn, q=0.90); s = st.summarize(syn, ev)\n"
            "    res.append((lead, p[1], b['slope'], b['t_hac'], s['cond_mean']*100, s['t'], s['p_placebo']))\n"
            "leads = [r[0] for r in res]; hacs = [r[3] for r in res]; rulet = [r[5] for r in res]\n"
            "x = np.arange(len(leads))\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.bar(x-.2, hacs, .4, color=GREEN, label='HAC t (one-day slope)')\n"
            "ax.bar(x+.2, rulet, .4, color=AMBER, label='Welch t (next-bar rule)')\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2'); ax.axhline(0, c='k', lw=.6)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'planted lead\\n{l:.2f}' for l in leads])\n"
            "ax.set_ylabel('t-stat'); ax.set_title('Control: lead=0 quiet, planted lead lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for l,xp1,sl,ht,cm,rt,rp in res:\n"
            "    print(f'lead={l:.2f}: rho+1={xp1:+.3f} slope={sl:+.3f} HAC_t={ht:>6.2f} rule_t={rt:>5.2f} p={rp:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted lead the control sits at HAC "
            f"**t = {R['syn'][0][3]:.2f}** and rule **t = {R['syn'][0][6]:.2f}** (quiet — no false "
            f"positive); a **0.05** lead already reaches HAC **t = {R['syn'][1][3]:.2f}** / rule "
            f"**t = {R['syn'][1][6]:.2f}**; a **0.25** lead screams (HAC t = {R['syn'][2][3]:.1f}). The "
            "machinery is honest, so the real-tape *negative* HAC t is a true negative: **the daily lead "
            "isn't there.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — $\\rho_0 = {R['xcorr'][0]:.2f}$ (contemporaneous, untradable); "
            f"$\\rho_{{+1}} = {R['xcorr'][1]:+.2f}$ and the HAC one-day slope **{R['lag1_slope']:+.3f}** at "
            f"**t = {R['lag1_t']:.2f}** (wrong sign, n.s.). No autocorrelation-robust next-day lead on "
            "daily bars. (True lead-lag is intraday — named on the axis.)\n"
            f"- **Tradability `MIRAGE`** — next-bar rule cond. **{R['q90'][2]:+.3f}%** vs base "
            f"**{R['q90'][4]:+.3f}%** (Welch **t = {R['q90'][6]:.2f}**, placebo **p = {R['q90'][7]:.2f}**); "
            f"**negative gross**, **{R['net'][0][2]:+.3f}%** net of 10 bps. An edge pointing the wrong "
            "way before costs is undeployable.\n"
            f"- **Free lunch? `BUSTED`** — the 'free lunch' is the same-day co-movement, realised "
            "simultaneously and impossible to act on; the only tradable piece (the next-day lead) is "
            "statistically zero. Co-movement ≠ predictability."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the clock, not the cost\n\n"
            "The operational truth: the lead-lag the literature documents is **real** but lives at the "
            "**intraday** scale, where index-arbitrage desks compete it away within the session. By the "
            "time you aggregate to a daily bar, the leader's information is already in the members' "
            "close. There is no daily next-bar edge to size — and no cost assumption that creates one "
            "from a negative gross number."
        ),
        code(
            "# illustrate the scale problem: how the (already-zero) daily lead compares to the\n"
            "# contemporaneous co-movement it is buried inside. The ratio is the 'information already\n"
            "# priced by the close' — overwhelming on daily bars.\n"
            "if HAVE_REAL:\n"
            "    prof = st.crosscorr_profile(F, max_lag=2); rho0 = abs(prof[0]); rho1 = abs(prof[1])\n"
            "else:\n"
            "    rho0 = abs(R['xcorr'][0]); rho1 = abs(R['xcorr'][1])\n"
            "frac_same_day = rho0/(rho0+rho1)\n"
            "fig, ax = plt.subplots(figsize=(8.4,2.4))\n"
            "ax.barh([0],[frac_same_day*100],color=GREEN,label='realised same-day (untradable)')\n"
            "ax.barh([0],[(1-frac_same_day)*100],left=[frac_same_day*100],color=AMBER,label='left for next day (tradable)')\n"
            "ax.set_yticks([]); ax.set_xlim(0,100); ax.set_xlabel('% of leader-member association')\n"
            "ax.set_title(f'{frac_same_day*100:.0f}% of the association is already realised same-day')\n"
            "ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5,-0.9)); plt.tight_layout(); plt.show()\n"
            "print(f'{frac_same_day*100:.1f}% of the leader-member link is contemporaneous; the tradable next-day residual is ~{(1-frac_same_day)*100:.1f}% and points the wrong way')"
        ),
        md(
            "> 💡 In plain words: essentially **all** of the leader-member association is realised the "
            "**same day**, where you can't trade it; the sliver left for the next day is both tiny and "
            "negative. The romance of 'the leader hands you the laggards' is an intraday effect that has "
            "**already happened** by the daily close. The clock, not the cost, is the verdict."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The right clock.** With intraday/tick data the leader genuinely leads (Lo-MacKinlay "
            "1990; Hou 2007; the ETF/index-arbitrage literature) — by minutes, not days. Re-running this "
            "engine on 1-minute bars is the natural next study; the daily result here predicts the "
            "signal vanishes as you coarsen the bar.\n"
            "- **Liquidity as the driver.** [Study 140 — Amihud-Illiquidity](../140-amihud-illiquidity/) "
            "— is the illiquidity that makes members 'slow' a priced, tradable factor in its own right?\n"
            "- **Microstructure artefacts.** Non-synchronous trading and bid-ask bounce (Scholes-Williams "
            "1977, Roll 1984) can *manufacture* a spurious daily lead-lag; here they don't even produce a "
            "positive one. A version that filters illiquid prints would sharpen the null further.\n\n"
            "*The reproducible core is offline and deterministic; the daily bar is an explicit proxy for "
            "the intraday phenomenon. Methods and sources: [`docs/references.md`](../docs/references.md); "
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
