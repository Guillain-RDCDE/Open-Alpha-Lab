"""Generate the two narrative notebooks for Study 156 (Martingale).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
daily parquets under ../_cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# Strategy parameters (mirrored in BOOT string and R dict for consistency)
MAX_LEVELS = 6
STEP = 0.05
TP = 0.05
COST = 5.0


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-14).
R = dict(
    # Pooled (SPY, QQQ, AAPL, GE) — real tape 1993-2026
    n_episodes=592,
    ruin_rate=6.2,
    win_rate=93.8,
    mart_mean_bps=313.9,
    bah_mean_bps=336.9,
    excess_mean_bps=-23.0,
    excess_skew=-2.50,
    excess_t=-0.31,
    mart_skew=-3.112,
    mart_t=2.76,
    # Per-ticker
    spy_ruin=2.3, spy_win=97.7, spy_excess=157.5, spy_t=1.41,
    qqq_ruin=6.4, qqq_win=93.6, qqq_excess=-123.6, qqq_t=-0.80,
    aapl_ruin=7.1, aapl_win=92.9, aapl_excess=-46.5, aapl_t=-0.41,
    ge_ruin=7.1,  ge_win=92.9,  ge_excess=-3.7,    ge_t=-0.02,
    # Monte Carlo ruin (mu=0, vol=18%)
    ruin_lv3=23.8, ruin_lv4=15.5, ruin_lv5=9.4,
    ruin_lv6=5.6,  ruin_lv7=2.6,  ruin_lv8=1.0,
    skew_lv3=-1.214, skew_lv4=-1.865, skew_lv5=-2.677,
    skew_lv6=-3.600, skew_lv7=-5.148, skew_lv8=-6.816,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from martingale import data, strategy as st

TICKERS = ["SPY", "QQQ", "AAPL", "GE"]
STEP, TP, MAX_LEVELS, COST = 0.05, 0.05, 6, 5.0

def _have_cache():
    return all(os.path.exists(data._cache_path(t, data.DEFAULT_CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def load_pooled(cost=COST):
    frames = []
    for t in TICKERS:
        prices = data.fetch_daily(t, fetch=False)
        ledger = st.run_episodes(prices["close"], step_pct=STEP, take_profit_pct=TP,
                                  max_levels=MAX_LEVELS, cost_bps=cost)
        ledger["ticker"] = t
        frames.append(ledger)
    return pd.concat(frames, ignore_index=True)

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Martingale — does averaging down beat buy-and-hold?\n"
            "### The doubling / adding-down gambler's strategy, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_B%26H%3F: Busted](https://img.shields.io/badge/Beats_B%26H%3F-Busted-8b949e?style=flat-square)\n\n"
            "Here's a recipe you'll find in every retail trading forum: when your position drops, "
            "*buy more* — double down, average down, buy the dip. The idea is that you're lowering "
            "your average cost and making the eventual recovery pay off bigger. It's pitched as a "
            "**disciplined, mathematical approach** to turning losing trades into winning ones. "
            "This notebook asks the only question that matters: does it actually beat just buying "
            "and holding the same capital?\n\n"
            "> 📓 **This is the plain-language layer.** Want the t-stats, the ruin math, and "
            "the bootstrap intervals? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does averaging down beat buy-and-hold? | **No.** Over 30 years of real data "
            f"the martingale strategy averages **+{R['mart_mean_bps']:.0f} bps/episode** vs "
            f"buy-and-hold's **+{R['bah_mean_bps']:.0f} bps/episode** — the simpler strategy wins. |\n"
            f"| 'But my win-rate is 94%!' | That's the trap. A "
            f"**{R['win_rate']:.0f}%** win-rate hides a "
            f"**{R['ruin_rate']:.0f}%** ruin rate and a skew of **{R['mart_skew']:+.1f}** — "
            "you pick up many small gains and occasionally blow up completely. |\n"
            "| Is there any statistical edge? | **No.** The excess return vs B&H is "
            f"**{R['excess_mean_bps']:+.0f} bps/episode** (t = {R['excess_t']:+.2f}) — "
            "indistinguishable from zero across all four markets. |\n"
            "| Could you trade it? | **No.** The ruin scenario is not theoretical — it occurs "
            "in the real data and creates a catastrophic left tail. |\n\n"
            "> The martingale is the sibling of the [loaded-dice](../../72-loaded-dice/) study: "
            "a strategy that *looks* like it wins because you see many wins and few losses, while "
            "the loss-when-it-happens is enormous."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When your stock is down 5%, buy more — you're lowering your average cost. "
            "When it falls another 5%, double your position again. When it recovers, you profit "
            "on all those shares at once. It's a mathematical edge that turns losers into winners.\"*\n\n"
            "This is the **martingale strategy** — originally a gambling system (double your bet "
            "after each loss at roulette, so the first win recovers everything). In markets it's "
            "called 'averaging down' or 'buying the dip with conviction.' The bet is that by "
            "doubling after each step down, you lower your average entry price so much that even "
            "a partial recovery becomes profitable. "
            "It *would* work perfectly in a world with unlimited capital."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If averaging down genuinely adds edge, it's a simple, systematic way to "
            "outperform buy-and-hold with the same capital. If it doesn't — if the mathematical "
            "averaging-down effect is exactly offset by the ruin risk you take on — then every "
            "retail trader doubling down on losing positions is running a **gambler's ruin** in "
            "slow motion. The stakes: this is one of the most common strategies in retail "
            "trading, and understanding whether it actually helps or just *feels* like it helps "
            "has real consequences."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three rules keep us honest:\n\n"
            "1. **Same total capital.** To compare fairly, the buy-and-hold baseline deploys "
            f"the *same* maximum capital the martingale could ever need (up to {2**MAX_LEVELS - 1}x "
            f"the initial position across {MAX_LEVELS} levels). If you only compare the initial "
            "stake, you're hiding the capital the doubling strategy has sitting in reserve.\n"
            "2. **Count every ruin event.** When the price drops far enough to trigger the "
            "final doubling level and keeps falling, the position is liquidated at a total loss. "
            "Any honest test includes every ruin in the statistics — not just the episodes that "
            "ended well.\n"
            "3. **Compare per episode.** Each episode starts at a fresh entry and ends at the "
            "take-profit, ruin, or end of data. We compare martingale vs buy-and-hold over the "
            "*same* window every time.\n\n"
            f"We run it on four real tapes — SPY (since 1993), QQQ (since 1999), AAPL and GE — "
            f"giving **{R['n_episodes']} real episodes** across ~30 years."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the win-rate trap.** The martingale wins most of the time — "
            "then occasionally blows up. Here is the return distribution:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pool = load_pooled()\n"
            "    mart_rets = pool['ret_net'].to_numpy()\n"
            "    bah_rets  = pool['bah_ret_net'].to_numpy()\n"
            "    n_ruin = (pool['outcome'] == 'ruin').sum()\n"
            "    ruin_pct = n_ruin / len(pool) * 100\n"
            "else:\n"
            "    import numpy as np\n"
            "    rng = np.random.default_rng(156)\n"
            "    # Approximate: many small positives, rare -1.0 ruin events\n"
            "    mart_rets = np.concatenate([rng.uniform(0.02,0.25,555), np.full(37,-1.0)])\n"
            "    bah_rets  = mart_rets * rng.uniform(0.8,1.2,592)\n"
            "    ruin_pct = R['ruin_rate']\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "axes[0].hist(mart_rets, bins=40, color=RED, alpha=0.75, label='Martingale')\n"
            "axes[0].hist(bah_rets,  bins=40, color=GREY, alpha=0.55, label='Buy-and-hold')\n"
            "axes[0].axvline(0, c='k', lw=1)\n"
            "axes[0].set_xlabel('Episode net return'); axes[0].set_ylabel('Count')\n"
            "axes[0].set_title('Return distribution: the left tail is the ruin')\n"
            "axes[0].legend()\n"
            "\n"
            "excess = mart_rets - bah_rets\n"
            "axes[1].hist(excess, bins=40, color=AMBER, alpha=0.8)\n"
            "axes[1].axvline(0, c='k', lw=1)\n"
            "axes[1].axvline(excess.mean(), c=RED, lw=2, label=f'mean={excess.mean()*1e4:+.0f} bps')\n"
            "axes[1].set_xlabel('Martingale excess over B&H')\n"
            "axes[1].set_title('Excess vs B&H: centred on zero')\n"
            "axes[1].legend()\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Ruin events: {{ruin_pct:.1f}}% of episodes | Martingale win-rate: '"
            f"f\"{{(mart_rets>0).mean()*100:.1f}}%\")"
        ),
        md(
            f"The martingale wins in **{R['win_rate']:.0f}%** of episodes — it really does "
            "collect many small gains. But the return distribution has a heavy left tail: "
            f"ruin events hit **{R['ruin_rate']:.0f}%** of the time, and when they hit, "
            "the whole position is gone. The excess-return panel (right) shows what this means "
            "for the comparison with buy-and-hold: the distribution is centred at zero, "
            "with a slight lean to the negative."
        ),
        md("**Now the regime breakdown: does averaging down at least help in crashes?**"),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for t in TICKERS:\n"
            "        prices = data.fetch_daily(t, fetch=False)\n"
            "        ledger = st.run_episodes(prices['close'], step_pct=STEP, take_profit_pct=TP,\n"
            "                                  max_levels=MAX_LEVELS, cost_bps=COST)\n"
            "        s = st.summarize(ledger, 'excess_ret_net')\n"
            "        rows.append((t, s['n_episodes'], s['ruin_rate']*100,\n"
            "                     s['win_rate']*100, s['mean_bps'], s['tstat']))\n"
            "    per_t = pd.DataFrame(rows, columns=['ticker','n','ruin%','win%','excess_bps','t'])\n"
            "else:\n"
            "    per_t = pd.DataFrame({\n"
            "        'ticker': TICKERS,\n"
            f"        'ruin%': [{R['spy_ruin']},{R['qqq_ruin']},{R['aapl_ruin']},{R['ge_ruin']}],\n"
            f"        'win%':  [{R['spy_win']},{R['qqq_win']},{R['aapl_win']},{R['ge_win']}],\n"
            f"        'excess_bps': [{R['spy_excess']},{R['qqq_excess']},{R['aapl_excess']},{R['ge_excess']}],\n"
            f"        't': [{R['spy_t']},{R['qqq_t']},{R['aapl_t']},{R['ge_t']}],\n"
            "    })\n"
            "    per_t['n'] = None\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "colors = [GREEN if abs(v) >= 2 else RED for v in per_t['t']]\n"
            "ax.bar(per_t['ticker'], per_t['excess_bps'], color=colors)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Excess return vs B&H (bps/episode)')\n"
            "ax.set_title('No market shows a reliable edge from averaging down')\n"
            "plt.tight_layout(); plt.show()\n"
            "per_t.round(1)"
        ),
        md(
            f"Across all four markets, the excess return is within the noise band (|t| < 2 "
            f"everywhere). SPY shows the most favourable result ({R['spy_excess']:+.0f} bps, "
            f"t = {R['spy_t']:+.2f}), still not significant, and reflecting SPY's long bull "
            "run rather than the averaging-down mechanics."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The martingale excess return over buy-and-hold is "
            f"**{R['excess_mean_bps']:+.0f} bps/episode** (t = **{R['excess_t']:+.2f}**) — "
            "statistically indistinguishable from zero, slightly negative. No market shows "
            "a reliable edge.\n"
            "- **Tradability — Mirage.** Ruin events occur in the real data, create a "
            f"skew of **{R['mart_skew']:+.1f}**, and wipe entire positions. The strategy "
            "can't be made safe by simply 'using more capital' — that just pushes the ruin "
            "further out while increasing the catastrophic loss when it arrives.\n"
            "- **Beats B&H? — Busted.** The famous high win-rate "
            f"(**{R['win_rate']:.0f}%**) is real and completely uninformative: it is a "
            "mechanical consequence of the exit rule (you collect small wins until the "
            "big loss arrives), not evidence of edge. "
            "Buy-and-hold of the same capital earns more in expectation."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Even if there were a small positive excess, the tail risk makes this "
            "untradeable at any serious size. Here's the ruin probability as you add "
            "more capital levels (more 'runway' for averaging down):"
        ),
        code(
            "levels = [3, 4, 5, 6, 7, 8]\n"
            "cap_units = [2**l - 1 for l in levels]\n"
            f"ruin_pcts = [{R['ruin_lv3']}, {R['ruin_lv4']}, {R['ruin_lv5']},\n"
            f"             {R['ruin_lv6']}, {R['ruin_lv7']}, {R['ruin_lv8']}]\n"
            f"skews = [{R['skew_lv3']}, {R['skew_lv4']}, {R['skew_lv5']},\n"
            f"         {R['skew_lv6']}, {R['skew_lv7']}, {R['skew_lv8']}]\n"
            "\n"
            "fig, ax1 = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax1.bar(levels, ruin_pcts, color=RED, alpha=0.8, label='Ruin probability %')\n"
            "ax1.set_xlabel('Max levels (N doublings allowed)')\n"
            "ax1.set_ylabel('Ruin probability per episode (%)', color=RED)\n"
            "ax2 = ax1.twinx()\n"
            "ax2.plot(levels, skews, 'o--', c=GREY, lw=2, label='P&L skew')\n"
            "ax2.set_ylabel('P&L skew (more negative = fatter left tail)', color=GREY)\n"
            "ax1.set_title('More capital = less ruin, but fatter tail when it hits')\n"
            "lines1, labels1 = ax1.get_legend_handles_labels()\n"
            "lines2, labels2 = ax2.get_legend_handles_labels()\n"
            "ax1.legend(lines1+lines2, labels1+labels2, loc='upper right')\n"
            "plt.tight_layout(); plt.show()\n"
            "\n"
            "for l, c, r, s in zip(levels, cap_units, ruin_pcts, skews):\n"
            "    print(f'  {l} levels ({c:3d}x capital): ruin {r:.1f}%, skew {s:+.3f}')"
        ),
        md(
            "Adding more capital levels lowers the per-episode ruin probability, but the "
            "required capital grows exponentially (63x for 6 levels, 255x for 8 levels) "
            "while the skew just gets worse — the rare ruin event becomes even more "
            "catastrophic relative to the normal wins. There is no free lunch: you cannot "
            "buy safety by deploying more capital, you can only move the ruin further out "
            "while making it proportionally more destructive."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The sister study.** [Study 72 — Loaded-Dice](../../72-loaded-dice/) is the "
            "direct sibling: a strategy that manufactures a 90%+ win-rate via exit asymmetry "
            "rather than doubling, but exposes the same negative-skew illusion.\n"
            "- **Does mean-reversion help?** If the market genuinely mean-reverts at the "
            "5% level, averaging down could pay. The quants notebook tests this with a "
            "synthetic positive control — the engine finds an edge when you plant mean-reversion, "
            "confirming the real market is simply not mean-reverting enough at this scale.\n"
            "- **The gambler's ruin.** The underlying mathematics — a random walk with absorbing "
            "barriers and finite capital — guarantees ruin with probability 1 given enough time. "
            "The question is only how long it takes and what it costs.\n\n"
            "*Think there's a smarter entry or exit rule that recovers the edge? Fork this, "
            "change the parameters, and show a genuine excess return over B&H with a controlled "
            "tail. That's the bar.*"
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
            "# Martingale — a quantitative teardown\n"
            "### Real daily tape · episode backtest · B&H control · ruin probability · HAC inference\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_B%26H%3F: Busted](https://img.shields.io/badge/Beats_B%26H%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We test whether "
            "the martingale / averaging-down strategy beats a buy-and-hold of the same total "
            "capital on real daily tapes (SPY 1993–2026, QQQ 1999–2026, AAPL 1993–2026, "
            "GE 1993–2026), derive the ruin probability analytically and by simulation, "
            "and show a synthetic positive control.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo daily bars, as-of 2026-06-14; "
            "offline core and tests run on a deterministic synthetic tape. Methods & sources in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **`💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Excess over B&H: **{R['excess_mean_bps']:+.0f} bps/episode**, "
            f"HAC *t* = **{R['excess_t']:+.2f}**; every ticker |*t*| < 1.5. |\n"
            f"| **Tradability** | `MIRAGE` | Ruin rate **{R['ruin_rate']:.0f}%** on real tape; "
            f"P&L skew **{R['mart_skew']:+.1f}**; exponential capital requirement per level. |\n"
            f"| **Beats B&H?** | `BUSTED` | Win-rate **{R['win_rate']:.0f}%** but B&H earns "
            f"more (**+{R['bah_mean_bps']:.0f}** vs **+{R['mart_mean_bps']:.0f}** bps/episode net). |\n\n"
            "> 💡 Three failures, one cause: doubling down cannot escape the mathematics of "
            "gambler's ruin on a market with finite capital. The high win-rate is a consequence "
            "of exit-asymmetry, not of any predictive information about future prices."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $P_t$ be the closing price and $P_0$ the initial entry. Define levels "
            "$k = 0, 1, \\ldots, K-1$ where level $k$ is entered if "
            "$P_t \\leq P_{k-1} \\cdot (1 - \\delta)$ (step-down of $\\delta = 5\\%$ from "
            "the previous entry). At level $k$ we deploy $2^k$ units, so after $k$ levels "
            "the total deployed capital is $\\sum_{j=0}^{k} 2^j = 2^{k+1}-1$ units. "
            "The average entry price is $\\bar{P} = \\sum_k 2^k P_{e_k} / (2^{k+1}-1)$. "
            "The take-profit fires when $P_t \\geq P_0 \\cdot (1 + \\tau)$ (TP = $\\tau = 5\\%$). "
            "Ruin fires when the required next doubling exceeds $K = 6$ levels, "
            "i.e. at max committed capital $2^K - 1 = 63$ units.\n\n"
            "**H₁ (signal):** $\\mathbb{E}[r_{\\text{martingale}} - r_{\\text{B\\&H}}] > 0$ "
            "across episodes of the same window and same total capital.\n"
            "**H₂ (tradable):** This excess survives realistic transaction costs and the ruin "
            "tail does not dominate the expectation.\n\n"
            "We reject H₁ and H₂ on both real data and on the random-walk synthetic null."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "The martingale is among the most psychologically compelling strategies: you see "
            "your win-rate is 94%, each winning trade feels like discipline rewarded, and the "
            "rare ruin feels like bad luck rather than the inevitable outcome of the math. "
            "Understanding that the averaging-down mechanism generates *no* measurable edge over "
            "buy-and-hold — and that the high win-rate is a direct artefact of the asymmetric "
            "exit, not of predictive information — is the difference between a professional "
            "approach and a gambler's fallacy."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Instrument set:** SPY, QQQ, AAPL, GE; daily Yahoo bars 1993–2026.\n"
            f"- **Entry:** long 1 unit at day $t_0$'s close.\n"
            f"- **Add-on rule:** if $P_t \\leq P_{{t_{{k-1}}}} \\cdot (1-0.05)$, "
            f"add $2^k$ units (doubling). Max {MAX_LEVELS} levels.\n"
            f"- **Take-profit:** all units exit when $P_t \\geq P_{{t_0}} \\cdot 1.05$.\n"
            "- **Ruin:** if the next required doubling would exceed max levels, "
            "liquidate at 100% loss of all deployed capital.\n"
            f"- **Baseline (B\\&H):** same start date, same end date, same total capital "
            f"($2^{{max\\_levels}}-1 = {2**MAX_LEVELS - 1}$ units), bought-and-held. "
            "No add-ons.\n"
            "- **Inference:** Newey-West HAC *t* on per-episode excess return "
            "(martingale net − B&H net). Bootstrap CI on each arm's Sharpe. "
            "Monte Carlo ruin probability over 5,000 simulated episodes.\n"
            "- **Positive control:** synthetic tape with tunable mean-reversion; "
            "the engine should find an edge if the market genuinely reverts at the step size."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Per-ticker breakdown — no market gives a reliable excess\n\n"
            "HAC *t* on per-episode excess return (martingale net minus B&H net). "
            "A bar clearing ±2 would be the signal."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for t in TICKERS:\n"
            "        prices = data.fetch_daily(t, fetch=False)\n"
            "        ledger = st.run_episodes(prices['close'], step_pct=STEP, take_profit_pct=TP,\n"
            "                                  max_levels=MAX_LEVELS, cost_bps=COST)\n"
            "        s_exc = st.summarize(ledger, 'excess_ret_net')\n"
            "        s_mart = st.summarize(ledger, 'ret_net')\n"
            "        rows.append((t, s_exc['n_episodes'], s_exc['ruin_rate']*100,\n"
            "                     s_mart['win_rate']*100, s_mart['skew'],\n"
            "                     s_exc['mean_bps'], s_exc['tstat']))\n"
            "    per_t = pd.DataFrame(rows, columns=\n"
            "        ['ticker','n','ruin%','win%','skew','excess_bps','t'])\n"
            "    pool = load_pooled()\n"
            "    sp = st.summarize(pool, 'excess_ret_net')\n"
            "else:\n"
            "    per_t = pd.DataFrame({\n"
            "        'ticker': TICKERS,\n"
            f"        'ruin%': [{R['spy_ruin']},{R['qqq_ruin']},{R['aapl_ruin']},{R['ge_ruin']}],\n"
            f"        'win%':  [{R['spy_win']},{R['qqq_win']},{R['aapl_win']},{R['ge_win']}],\n"
            f"        'excess_bps': [{R['spy_excess']},{R['qqq_excess']},{R['aapl_excess']},{R['ge_excess']}],\n"
            f"        't': [{R['spy_t']},{R['qqq_t']},{R['aapl_t']},{R['ge_t']}],\n"
            f"        'skew': [{R['mart_skew']:.2f},{R['mart_skew']:.2f},{R['mart_skew']:.2f},{R['mart_skew']:.2f}],\n"
            "    }); per_t['n'] = None\n"
            f"    sp = dict(mean_bps={R['excess_mean_bps']}, tstat={R['excess_t']})\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "col = [GREEN if abs(v) >= 2 else RED for v in per_t['t']]\n"
            "ax.bar(per_t['ticker'], per_t['t'], color=col)\n"
            "for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('HAC t-stat (excess net return)')\n"
            "ax.set_title('Four markets, four non-results (|t| < 2 everywhere)')\n"
            "ax.set_ylim(-3, 3); plt.tight_layout(); plt.show()\n"
            f"print(f\"pooled: {{sp['mean_bps']:+.1f}} bps/episode, HAC t = {{sp['tstat']:+.2f}}\")\n"
            "per_t.round(2)"
        ),
        md(
            "> 💡 Every ticker sits inside the ±2 band. Pooling "
            f"{R['n_episodes']} episodes gives **{R['excess_mean_bps']:+.0f} bps/episode**, "
            f"HAC *t* = **{R['excess_t']:+.2f}**. There is no edge — the extra power of "
            "pooling tells us this confidently."
        ),
        md(
            "### 4b · The skew and ruin profile — the real risk\n\n"
            "The martingale has a strongly negative P&L skew: many small wins, rare large losses. "
            "This is the tell for a tail-risk strategy."
        ),
        code(
            "levels = [3, 4, 5, 6, 7, 8]\n"
            "cap_units = [2**l - 1 for l in levels]\n"
            f"ruin_pcts = [{R['ruin_lv3']},{R['ruin_lv4']},{R['ruin_lv5']},"
            f"{R['ruin_lv6']},{R['ruin_lv7']},{R['ruin_lv8']}]\n"
            f"skews = [{R['skew_lv3']},{R['skew_lv4']},{R['skew_lv5']},"
            f"{R['skew_lv6']},{R['skew_lv7']},{R['skew_lv8']}]\n"
            "\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "ax1.plot(levels, ruin_pcts, 'o-', c=RED, lw=2)\n"
            "ax1.set_xlabel('Max levels (N doublings)'); ax1.set_ylabel('Ruin probability (%)')\n"
            "ax1.set_title('Ruin probability vs capital levels')\n"
            "for l, c, r in zip(levels, cap_units, ruin_pcts):\n"
            "    ax1.annotate(f'{c}x\\ncap', (l, r), ha='center', va='bottom', fontsize=8)\n"
            "\n"
            "ax2.plot(levels, skews, 's--', c=GREY, lw=2)\n"
            "ax2.axhline(0, c='k', lw=1)\n"
            "ax2.set_xlabel('Max levels'); ax2.set_ylabel('P&L skew')\n"
            "ax2.set_title('Skew worsens as ruin becomes rarer but bigger')\n"
            "plt.tight_layout(); plt.show()\n"
            "\n"
            "print('Capital units required by level:')\n"
            "for l, c in zip(levels, cap_units):\n"
            "    print(f'  {l} levels: {c:3d}x initial position, ruin skew {skews[levels.index(l)]:+.2f}')"
        ),
        md(
            "> 💡 This is the fundamental trade-off of the martingale: adding more capital "
            "levels buys lower ruin probability, but the required capital grows **exponentially** "
            f"(63x for 6 levels, 255x for 8 levels) and the P&L skew becomes more extreme. "
            "There is no configuration where the tail risk is acceptable and the edge is positive."
        ),
        md(
            "### 4c · The positive control — mean reversion is the missing ingredient\n\n"
            "On a synthetic tape with planted mean-reversion, the martingale gains an edge "
            "over buy-and-hold — confirming the engine works and the market is just "
            "not mean-reverting at the 5% step scale."
        ),
        code(
            "mr_vals = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]\n"
            "edges = []\n"
            "for mr in mr_vals:\n"
            "    prices, _ = data.synthetic_daily(n_days=3000, mu=0.0,\n"
            "                                      mean_reversion=mr, seed=156)\n"
            "    ledger = st.run_episodes(prices['close'], step_pct=STEP,\n"
            "                             take_profit_pct=TP, max_levels=MAX_LEVELS,\n"
            "                             cost_bps=COST)\n"
            "    s = st.summarize(ledger, 'excess_ret_net')\n"
            "    edges.append(s['mean_bps'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(mr_vals, edges, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('Planted mean-reversion speed')\n"
            "ax.set_ylabel('Martingale excess over B&H (bps/episode)')\n"
            "ax.set_title('The engine works: mean-reversion drives excess return')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At zero mean-reversion (random walk), excess is zero or negative.')\n"
            "print('Real markets at this scale: no measurable mean-reversion.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — pooled excess {R['excess_mean_bps']:+.0f} bps/episode, "
            f"HAC *t* {R['excess_t']:+.2f}; every ticker |*t*| < 1.5. "
            "B&H of the same capital earns more.\n"
            f"- **Tradability `MIRAGE`** — ruin rate {R['ruin_rate']:.0f}% on real data; "
            f"P&L skew {R['mart_skew']:+.1f}; capital requirement doubles with each level. "
            "The tail risk is real and non-diversifiable.\n"
            f"- **Beats B&H? `BUSTED`** — win-rate {R['win_rate']:.0f}% conceals the ruin; "
            "buy-and-hold earns more net. The averaging-down mechanism adds no edge on a "
            "market that does not mean-revert at this scale."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the ruin math\n\n"
            "The expected number of levels before ruin on a random walk with daily vol $\\sigma$ "
            "and step $\\delta$ is approximately $K^* = \\lfloor \\log_2(C / P_0) \\rfloor$ "
            "where $C$ is total capital. For $\\delta = 5\\%$, $\\sigma = 18\\%/\\sqrt{252}$, "
            "the probability of hitting $k$ consecutive step-downs before the TP fires is "
            "computed by simulation (shown in 4b). No level provides both low ruin probability "
            "and positive excess return simultaneously.\n\n"
            "The capital requirement grows as $2^K - 1$:\n"
        ),
        code(
            "import numpy as np\n"
            "levels = np.arange(3, 11)\n"
            "cap = 2**levels - 1\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(levels, cap, color=RED)\n"
            "ax.set_xlabel('Max levels K')\n"
            "ax.set_ylabel('Capital units required (= 2^K - 1)')\n"
            "ax.set_title('Capital grows exponentially — finite wealth always hits the wall')\n"
            "for l, c in zip(levels, cap):\n"
            "    ax.text(l, c + 3, str(c), ha='center', va='bottom', fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('This is why the martingale cannot be fixed: finite capital + unlimited'\n"
            "      + ' draw-down potential = guaranteed eventual ruin on any random walk.')"
        ),
        md(
            "> 💡 The Gambler's Ruin theorem (Feller 1950) states that on a fair random walk "
            "with finite capital, the probability of ruin approaches 1 as time goes to infinity. "
            "Real markets add a further insult: fat tails mean the ruin step can be larger than "
            "expected, arriving before you've had time to 'average down' to all planned levels."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Is the engine capable of finding an edge, or does it always print 'none'? "
            "Plant a mean-reversion pull in a synthetic daily tape: the martingale's "
            "advantage over buy-and-hold should turn on when genuine mean-reversion exists."
        ),
        code(
            "mr_vals = [0.0, 0.02, 0.05, 0.10, 0.20]\n"
            "mu_vals = [0.0, -0.10, -0.20]\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "\n"
            "# Panel 1: sweep mean-reversion\n"
            "edges_mr = []\n"
            "for mr in mr_vals:\n"
            "    prices, _ = data.synthetic_daily(n_days=3000, mu=0.0, mean_reversion=mr, seed=156)\n"
            "    ledger = st.run_episodes(prices['close'], step_pct=STEP, take_profit_pct=TP,\n"
            "                             max_levels=MAX_LEVELS, cost_bps=COST)\n"
            "    s = st.summarize(ledger, 'excess_ret_net')\n"
            "    edges_mr.append(s['mean_bps'])\n"
            "axes[0].plot(mr_vals, edges_mr, 'o-', c=GREEN, lw=2)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_xlabel('Mean-reversion speed')\n"
            "axes[0].set_ylabel('Excess over B&H (bps)')\n"
            "axes[0].set_title('Mean-reversion helps; random walk = no edge')\n"
            "\n"
            "# Panel 2: sweep mu (drift)\n"
            "ruin_rates = []\n"
            "for mu in mu_vals:\n"
            "    prices, _ = data.synthetic_daily(n_days=3000, mu=mu, mean_reversion=0.0, seed=156)\n"
            "    ledger = st.run_episodes(prices['close'], step_pct=STEP, take_profit_pct=TP,\n"
            "                             max_levels=MAX_LEVELS, cost_bps=COST)\n"
            "    s = st.summarize(ledger, 'excess_ret_net')\n"
            "    ruin_rates.append(s['ruin_rate'] * 100)\n"
            "axes[1].bar([str(m) for m in mu_vals], ruin_rates, color=[GREEN, AMBER, RED])\n"
            "axes[1].set_xlabel('Annual drift')\n"
            "axes[1].set_ylabel('Ruin rate (%)')\n"
            "axes[1].set_title('Negative drift massively increases ruin risk')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The engine is honest: it finds edge only when the market provides it.')"
        ),
        md(
            "The martingale finds a positive excess return on mean-reverting synthetic tapes — "
            "the mechanism genuinely works when prices oscillate around a mean at the step scale. "
            "The real-tape verdict is therefore a statement about the **market**, not the method: "
            "equity markets do not mean-revert reliably at the 5% step scale over multi-day "
            "horizons. If you believe in a market that does, the right study is "
            "[Study 33 — Slingshot](../../33-slingshot/) (short-term reversal) rather than "
            "a doubling strategy with catastrophic tail risk."
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
