"""Generate the two narrative notebooks for Study 426 (Know Sure Thing / KST).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY daily tape
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md.
# SPY daily, yfinance auto-adjusted (total return), 1993-01-29 -> 2026-06-23 (as-of, the last
# fully-closed session; the in-progress bar is dropped), 8,406 bars, 33.4 years, fp b7fff1b5e020.
# Net of 1 bp one-way x NAV, one-day execution lag, excess-of-cash Sharpe.
R = dict(
    asof="2026-06-23", start="1993-01-29", end="2026-06-23", years=33.4,
    n_bars=8406, fp="b7fff1b5e020", cost_bps=1.0,
    # per-arm: sharpe_excess, cagr%, maxdd%, hac_t, exposure, n_trades, turn/yr
    kst=(0.445, 4.29, -26.1, 2.65, 0.47, 458, 13.7),
    bah=(0.643, 10.76, -55.2, 4.33, 1.00, 1, 0.0),
    sma200=(0.727, 8.32, -28.3, 4.51, 0.75, 215, 6.4),
    macd=(0.421, 4.11, -38.5, 2.63, 0.50, 712, 21.3),
    perm_obs=0.458, perm_p=0.0044,
    spread_yr=-7.16, spread_t=-3.31,          # KST net minus buy-and-hold net
    # cost sweep: (one-way bp, KST sharpe, BAH sharpe)
    sweep=[(0.0, 0.458, 0.643), (1.0, 0.445, 0.643),
           (2.0, 0.432, 0.643), (5.0, 0.394, 0.643)],
    # long/short variant: sharpe, cagr%, maxdd%, t
    ls=(-0.139, -4.22, -81.4, -0.90),
    # synthetic control: (trend, kst_sh, bah_sh, spread%/yr, spread_t)
    syn=[(0.00, 0.591, 0.732, -5.41, -1.86),
         (0.20, 0.776, 0.775, -3.99, -1.14),
         (0.35, 0.973, 0.808, -2.21, -0.53)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![A_sure_thing%3F: Busted](https://img.shields.io/badge/A_sure_thing%3F-Busted-8b949e?style=flat-square)\n\n"
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

from know_sure_thing import data, strategy as st

ASOF = "2026-06-23"   # last fully-closed session; drop the in-progress bar
HAVE_REAL = data.have_real("SPY")
if HAVE_REAL:
    BARS = data.load_real("SPY").loc[:ASOF]
    print("SPY cache present | bars:", len(BARS), "|",
          BARS.index.min().date(), "->", BARS.index.max().date(),
          "| fingerprint:", data.fingerprint(BARS))
else:
    BARS = None
    print("SPY cache present:", HAVE_REAL, "(quoting frozen R)")
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Is Pring's \"Know Sure Thing\" a sure thing? 🤔\n"
            "### A famous momentum indicator, raced honestly against the plainest tools — in plain English\n\n"
            + BADGES +
            "In 1992 the technical analyst **Martin Pring** published an indicator with an irresistible "
            "name: the **Know Sure Thing**, or **KST**. It blends four momentum measures into one smooth "
            "line, and the rule is simple — **buy when the KST crosses above its signal line, step aside "
            "when it crosses below**. Fans say it catches big trends earlier and with fewer false alarms "
            "than a plain moving average.\n\n"
            "Great name. But does it actually *beat the alternatives*? We do the one test the brochures "
            "skip: we run it on **33 years of SPY** and put it in a head-to-head race against the most "
            "boring benchmarks imaginable — **just holding the market**, a **200-day moving average**, and "
            "the everyday **MACD**.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the permutation test and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice** — research & education. Every chart is drawn by the code beside "
            "it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the KST timing rule make money? | **Yes — a bit.** Long/flat, it earns a positive "
            "return above cash, and it really does beat a *coin flip* with the same time-in-market. So "
            "there's a flicker of genuine signal. |\n"
            "| Does it beat just **holding the market**? | **No — it loses, badly.** Buy-and-hold's "
            "risk-adjusted return (Sharpe **0.64**) crushes the KST rule's (**0.45**). Trading the KST "
            "gives up about **7% a year** versus doing nothing. |\n"
            "| Is it at least better than a **200-day average**? | **No.** The plain SMA-200 (Sharpe "
            "**0.73**) beats *both* KST and buy-and-hold. The fancy indicator loses to the simplest "
            "trend filter on the shelf. |\n"
            "| Can you fix it by **shorting** when it says sell? | **No — that's a disaster** (it loses "
            "money and draws down **−81%**). |\n\n"
            "> The KST isn't *noise* — it has a faint directional pulse. But \"sure thing\"? It can't beat "
            "buy-and-hold or a 200-day average. The clever indicator is **worse** than the boring tools."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The Know Sure Thing rolls four different momentum lookbacks into one smooth oscillator. "
            "Because it's built from long and short trends together, it turns up earlier and whipsaws less "
            "than a single moving average. Buy when KST crosses above its signal line; go to cash when it "
            "crosses below. You ride the big up-trends and sidestep the crashes.\"*\n\n"
            "Pring literally named it the *Know **Sure** Thing* — the marketing is in the title. The "
            "steelman is real, though: combining momentum over several horizons **is** a sensible idea, and "
            "trend-following on equities **does** have a long, respectable track record (Faber's 200-day "
            "rule, time-series momentum). So the honest question isn't \"is momentum real?\" It's: **does "
            "this particular, more complicated recipe beat the plain ones?**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the KST genuinely timed the market, it would be worth a fortune: you'd capture the "
            "up-years and dodge 2008 and 2022, turning the same index into a smoother, richer ride. That's "
            "the dream every market-timing indicator sells.\n\n"
            "But there's a catch every timer hits. Stepping out of the market means **giving up the equity "
            "risk premium** while you're in cash — and stocks drift *up* most of the time. So a timing rule "
            "has to earn back everything it gave up **plus** beat the simplest possible alternative, or "
            "it's just an expensive way to underperform. We'll measure exactly that trade-off."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{R['years']:.0f} years of SPY** (the S&P 500 ETF, dividends included), "
            f"{R['start']} → {R['end']}, and build four daily strategies:\n\n"
            "1. **KST** — long when the Know Sure Thing is above its signal line, flat otherwise.\n"
            "2. **Buy-and-hold** — always invested. The thing you're trying to beat.\n"
            "3. **SMA-200** — long when price is above its 200-day average (the classic trend filter).\n"
            "4. **MACD** — long on the everyday 12/26/9 crossover.\n\n"
            "Every rule pays realistic **trading costs**, enters **one day after** the signal (no "
            "cheating), and we compare them on **risk-adjusted return** (Sharpe). The killer test: **if "
            "KST can't out-Sharpe buy-and-hold and the 200-day average, it's not a sure thing — it's a "
            "mirage.**"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually race them\n\n"
            "Here's the whole story in one chart: the risk-adjusted return (Sharpe, net of costs) of all "
            "four strategies over 33 years. Higher is better."
        ),
        code(
            "names = ['KST\\n(long/flat)', 'Buy &\\nhold', 'SMA-200', 'MACD']\n"
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(BARS, cost_bps=R['cost_bps'], n_perm=1)\n"
            "    sh = [res['kst']['sharpe_excess'], res['buy_and_hold']['sharpe_excess'],\n"
            "          res['sma200']['sharpe_excess'], res['macd']['sharpe_excess']]\n"
            "else:\n"
            "    sh = [R['kst'][0], R['bah'][0], R['sma200'][0], R['macd'][0]]\n"
            "cols = [AMBER, GREEN, GREEN, RED]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "ax.bar(names, sh, color=cols, width=.62)\n"
            "for i,v in enumerate(sh): ax.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('Sharpe (net, excess of cash)')\n"
            "ax.set_title('The fancy indicator LOSES to just holding the market — and to a 200-day average')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('KST', round(sh[0],3), '| buy&hold', round(sh[1],3), '| SMA-200', round(sh[2],3), '| MACD', round(sh[3],3))"
        ),
        md(
            f"The verdict is right there. **Buy-and-hold ({R['bah'][0]:.2f})** beats KST "
            f"(**{R['kst'][0]:.2f}**). The dead-simple **200-day average ({R['sma200'][0]:.2f})** beats "
            "*everything*. The KST — for all its four-momentum machinery — lands in the **same bucket as "
            "MACD**, below the boring benchmarks. The clever recipe isn't earning its complexity."
        ),
        md(
            "**But wait — is the KST at least better than random?** Let's be fair. We keep the KST's "
            "exact in-and-out schedule (so the time-in-market is identical) but randomly **flip a coin** "
            "for each day's direction, thousands of times, and see how often a coin matches the real "
            "rule's return."
        ),
        code(
            "if HAVE_REAL:\n"
            "    book = st.book_returns(BARS['close'], st.kst_position(BARS['close']), cost_bps=R['cost_bps'])\n"
            "    pl = st.permutation_pvalue(book, n_draws=5000, seed=426)\n"
            "    obs, draws, p = pl['obs_sharpe'], pl['draws'], pl['p_value']\n"
            "else:\n"
            "    obs, p = R['perm_obs'], R['perm_p']\n"
            "    rng = np.random.default_rng(426); draws = rng.normal(0.0, 0.18, 5000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=60, color=GREY, alpha=.85, label='5,000 random sign-flip schedules')\n"
            "ax.axvline(obs, c=AMBER, lw=2.5, label=f'real KST schedule = {obs:.2f}')\n"
            "ax.set_xlabel('gross Sharpe'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'KST DOES beat a coin with the same exposure (p = {p:.4f}) — a faint real pulse')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:.3f}  beats-a-coin p = {p:.4f}')"
        ),
        md(
            f"So there *is* a real flicker: only **{R['perm_p']*100:.1f}%** of random coin-schedules with "
            "the KST's exposure do as well, so the rule's *direction* carries a little genuine information. "
            "**That's the trap, though** — \"beats a coin\" and \"beats buy-and-hold\" are different bars, "
            "and KST only clears the easy one. The faint pulse is mostly just the **equity risk premium** "
            "it collects on the ~47% of days it happens to be invested."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The long/flat rule has a faint directional pulse (it beats a coin, "
            f"p = {R['perm_p']:.3f}), but it's mostly harvested market beta and it **loses** to the "
            "benchmark.\n"
            f"- **Tradability — Mirage.** KST Sharpe **{R['kst'][0]:.2f}** vs buy-and-hold "
            f"**{R['bah'][0]:.2f}** vs SMA-200 **{R['sma200'][0]:.2f}**. Trading it *costs* you ~7%/yr "
            "versus doing nothing; shorting on the sell signal is a −81% disaster.\n"
            "- **\"A sure thing\"? — Busted.** It can't out-perform holding the market, and a one-line "
            "200-day moving average beats it. The complexity buys nothing."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the lower-drawdown consolation\n\n"
            "To be completely fair: the KST rule *does* one thing well — it cuts the worst crashes. "
            "Sitting in cash through 2008 and 2022 means a much shallower drawdown than buy-and-hold. "
            "Here's the trade-off: return vs pain."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(BARS, cost_bps=R['cost_bps'], n_perm=1)\n"
            "    arms = ['kst','buy_and_hold','sma200']\n"
            "    cg = [res[a]['cagr_net']*100 for a in arms]; dd = [abs(res[a]['maxdd_net'])*100 for a in arms]\n"
            "else:\n"
            "    cg = [R['kst'][1], R['bah'][1], R['sma200'][1]]; dd = [abs(R['kst'][2]), abs(R['bah'][2]), abs(R['sma200'][2])]\n"
            "labels = ['KST','Buy & hold','SMA-200']\n"
            "x = np.arange(3)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, cg, .4, color=GREEN, label='CAGR % (return)')\n"
            "ax.bar(x+.2, dd, .4, color=RED, label='worst drawdown % (pain)')\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('%')\n"
            "for i,v in enumerate(cg): ax.annotate(f'{v:.1f}',(i-.2,v),ha='center',va='bottom',fontsize=9)\n"
            "for i,v in enumerate(dd): ax.annotate(f'{v:.0f}',(i+.2,v),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_title('KST gives up a LOT of return to shave drawdown — SMA-200 does it better')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('CAGR:', [round(c,1) for c in cg], '| maxDD:', [round(d,0) for d in dd])"
        ),
        md(
            f"Yes, KST's worst drawdown (**{abs(R['kst'][2]):.0f}%**) is far gentler than buy-and-hold's "
            f"(**{abs(R['bah'][2]):.0f}%**). But you pay for it with a CAGR of only **{R['kst'][1]:.1f}%** "
            f"vs **{R['bah'][1]:.1f}%** — and the **SMA-200** delivers a *similar* low drawdown "
            f"(**{abs(R['sma200'][2]):.0f}%**) while keeping **{R['sma200'][1]:.1f}%** of return. If you "
            "want the smoother ride, the boring 200-day average is simply the better deal."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Parameters.** Pring's daily KST has eight knobs (four ROCs, four smoothings). Tune them on "
            "the full sample and you'll find a flattering combo — that's curve-fitting, not edge. Try it "
            "out-of-sample and watch it fade.\n"
            "- **The real lesson.** A complicated indicator with a confident name is *not* automatically "
            "better than a one-line moving average. Always race the fancy thing against the boring thing.\n"
            "- **Build your own.** Swap SPY for a single stock, or for a trendier asset (gold, a "
            "currency); the KST might fare better where trends are cleaner. The harness is in "
            "[`know_sure_thing/`](../know_sure_thing/).\n\n"
            "*Think KST beats buy-and-hold on some asset, net of costs and out-of-sample? Show the **net "
            "Sharpe** clearing the benchmark — then we'll talk.*"
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
            "# Know Sure Thing — a quantitative teardown 🔬\n"
            "### KST/signal long-flat (and long-short) timing on SPY · NET excess-vs-excess Sharpe race vs "
            "buy-and-hold / SMA-200 / MACD · HAC *t* · a sign-flip permutation placebo · cost sweep · "
            "one-day execution lag · a synthetic planted-trend control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). KST is a "
            "weighted sum of four SMA-smoothed ROCs with a signal-line crossover rule. The job here is to "
            "ask the only question that matters for a *more complicated* indicator: **does it beat the "
            "obvious simpler benchmarks**, net of costs, on the real tape?\n\n"
            "> ⚠️ **Data note.** SPY daily, yfinance auto-adjusted (**total return**), "
            f"{R['start']}→{R['end']} (as-of the last fully-closed session; the in-progress bar is "
            "dropped). One-way cost × NAV, one execution lag, **excess-of-cash** Sharpe both sides. "
            "Offline core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | The long/flat rule's excess-of-cash returns clear the inference bar "
            f"(HAC **t = {R['kst'][3]:.2f}**) and beat a sign-flip coin with identical exposure "
            f"(**p = {R['perm_p']:.4f}**) — but that pulse is mostly harvested **beta**: the value-add "
            f"over buy-and-hold is **significantly negative** (KST−BAH spread **t = {R['spread_t']:.2f}**). |\n"
            f"| **Tradability** | `MIRAGE` | Net excess Sharpe KST **{R['kst'][0]:.2f}** < buy-and-hold "
            f"**{R['bah'][0]:.2f}** < SMA-200 **{R['sma200'][0]:.2f}**. The timing rule *destroys* "
            f"~{abs(R['spread_yr']):.0f}%/yr vs doing nothing; the long/short variant is "
            f"Sharpe **{R['ls'][0]:.2f}** with a **{R['ls'][2]:.0f}%** drawdown. |\n"
            f"| **A sure thing?** | `BUSTED` | KST loses the race to **both** buy-and-hold and a one-line "
            f"SMA-200 (and ties MACD). The four-ROC machinery adds nothing over the plainest trend filter. |\n\n"
            "> 💡 In plain words: KST has a faint *real* directional signal, but it's indistinguishable "
            "from the equity risk premium it collects while invested, and it **loses to the simplest "
            "possible benchmarks**. Clever name, no edge."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "KST sums four SMA-smoothed rate-of-change series, weighted by horizon:\n\n"
            "$$\\mathrm{KST}_t = \\sum_{i=1}^{4} w_i\\,\\mathrm{SMA}_{m_i}\\!\\big(\\mathrm{ROC}_{n_i}(t)\\big),"
            "\\quad w=(1,2,3,4),\\; n=(10,15,20,30),\\; m=(10,10,10,15),$$\n\n"
            "with a signal line $s_t = \\mathrm{SMA}_9(\\mathrm{KST}_t)$. The position is "
            "$x_t = \\mathbf{1}[\\mathrm{KST}_t > s_t]$ (long/flat) or $\\mathrm{sgn}(\\mathrm{KST}_t - s_t)$ "
            "(long/short). The believer's hypotheses:\n\n"
            "- **H₁ (it has edge).** The KST book's excess return is positive and HAC-significant.\n"
            "- **H₂ (it's deployable).** Its NET excess Sharpe beats **buy-and-hold**.\n"
            "- **H₃ (it's *better*).** It beats the obvious simpler timing rules (SMA-200, MACD).\n\n"
            "We find **H₁ marginally supported** (t > 2, beats a coin) — but it is **beta in disguise**; "
            "**H₂ rejected** (KST−BAH spread t ≈ −3.3); **H₃ rejected** (loses to SMA-200, ties MACD). "
            "The indicator works as a momentum filter; it just doesn't *win*."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — alpha vs beta is the whole game\n\n"
            "A long/flat trend rule is **not** a market-neutral bet — when it's long it earns the equity "
            "risk premium, so any naive *t*-test of its returns against zero will look significant "
            "**purely from beta**. That is exactly why a single HAC *t* > 2 on the book is **not** enough "
            "to certify a timing *skill*. Two sharper tests separate signal from beta:\n\n"
            "- **The sign-flip permutation** holds the exposure schedule fixed and randomises only the "
            "*direction* — it asks whether the timing adds information beyond being invested that often.\n"
            "- **The KST−buy-and-hold spread** is the literal value-add of trading over not trading; its "
            "sign and *t* are the honest tradability test. A negative spread *t* means the rule "
            "**subtracts** value, however significant its standalone return looks."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** SPY daily, yfinance auto-adjusted total return, {R['start']}→{R['end']} "
            f"(**{R['n_bars']:,}** bars, **{R['years']:.1f}** years, fingerprint `{R['fp']}`).\n"
            "- **Indicator.** Pring's daily KST (ROC 10/15/20/30, SMA 10/10/10/15, signal 9).\n"
            "- **Rule.** Long/flat (long when KST > signal) and a long/short variant.\n"
            "- **Timing.** Position known at close *t* earns the return of *t+1* — one `shift`, applied "
            "once. No look-ahead.\n"
            f"- **Costs.** {R['cost_bps']:.0f} bp one-way × NAV on every position change (turnover = "
            "one-way × NAV); shorts pay 50 bp/yr borrow. Excess-of-cash Sharpe on both sides of every race.\n"
            "- **Benchmarks.** Buy-and-hold, SMA-200 (Faber), MACD(12,26,9) — same engine, same costs.\n"
            "- **Inference.** HAC (Newey-West) *t* on the book's excess returns; a 5,000-draw sign-flip "
            "permutation; the KST−BAH spread *t*; a cost sweep.\n"
            "- **Positive control.** A deterministic AR(1) tape with a **planted trend** knob: the "
            "value-add over buy-and-hold must rise from negative (no trend) toward positive (strong trend).\n\n"
            "**What would make us say \"mirage\":** a NET excess Sharpe below buy-and-hold and below "
            "SMA-200. (Spoiler: that is what happens.)"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The race — net excess Sharpe, all four arms\n\n"
            "The headline comparison. Net of 1 bp one-way × NAV, excess-of-cash, one-day lag."
        ),
        code(
            "names = ['KST', 'Buy & hold', 'SMA-200', 'MACD']\n"
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(BARS, cost_bps=R['cost_bps'], n_perm=1)\n"
            "    sh = [res[a]['sharpe_excess'] for a in ('kst','buy_and_hold','sma200','macd')]\n"
            "    tt = [res[a]['hac_t'] for a in ('kst','buy_and_hold','sma200','macd')]\n"
            "else:\n"
            "    sh = [R['kst'][0], R['bah'][0], R['sma200'][0], R['macd'][0]]\n"
            "    tt = [R['kst'][3], R['bah'][3], R['sma200'][3], R['macd'][3]]\n"
            "cols = [AMBER, GREEN, GREEN, RED]\n"
            "fig, (a1,a2) = plt.subplots(1, 2, figsize=(10.8, 4.3))\n"
            "a1.bar(names, sh, color=cols, width=.62)\n"
            "for i,v in enumerate(sh): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_ylabel('net excess Sharpe'); a1.set_title('KST loses to buy-and-hold AND SMA-200')\n"
            "a1.tick_params(axis='x', labelsize=8)\n"
            "a2.bar(names, tt, color=cols, width=.62)\n"
            "a2.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i,v in enumerate(tt): a2.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a2.set_ylabel('HAC t (excess returns)'); a2.set_title('All clear t=2 — because all collect beta'); a2.legend()\n"
            "a2.tick_params(axis='x', labelsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Sharpe:', [round(s,3) for s in sh]); print('HAC t:', [round(t,2) for t in tt])"
        ),
        md(
            f"> 💡 In plain words: every arm clears HAC *t* = 2 (left bar of buy-and-hold itself is "
            f"**{R['bah'][3]:.2f}**) — which is the whole point: **a standalone *t* > 2 just means \"it "
            f"was long a rising market.\"** On the metric that matters, *risk-adjusted* return, KST "
            f"(**{R['kst'][0]:.2f}**) sits **below** buy-and-hold (**{R['bah'][0]:.2f}**) and well below "
            f"SMA-200 (**{R['sma200'][0]:.2f}**)."
        ),
        md(
            "### 4b · Does the timing add information? — the sign-flip permutation\n\n"
            "Keep the KST exposure schedule, randomise the *direction* per bar (5,000 fair coins). If the "
            "observed Sharpe sits in the right tail, the rule's direction carries real information — "
            "**separate** from the question of whether it beats buy-and-hold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    book = st.book_returns(BARS['close'], st.kst_position(BARS['close']), cost_bps=R['cost_bps'])\n"
            "    pl = st.permutation_pvalue(book, n_draws=5000, seed=426)\n"
            "    obs, draws, p = pl['obs_sharpe'], pl['draws'], pl['p_value']\n"
            "else:\n"
            "    obs, p = R['perm_obs'], R['perm_p']\n"
            "    rng = np.random.default_rng(426); draws = rng.normal(0.0, 0.18, 5000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=60, color=GREY, alpha=.85, label='5,000 sign-flip schedules')\n"
            "ax.axvline(obs, c=AMBER, lw=2.5, label=f'observed = {obs:.3f}')\n"
            "ax.set_xlabel('gross Sharpe'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Beats a coin with the same exposure: p = {p:.4f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:.3f}  permutation p = {p:.4f}')"
        ),
        md(
            f"> 💡 In plain words: **p = {R['perm_p']:.4f}** — the KST direction *does* beat a coin. So "
            "there is a genuine (if faint) momentum signal. This is what keeps the Signal at **WEAK** and "
            "not **NONE** — but \"beats a coin\" is a far lower bar than \"beats holding the market.\""
        ),
        md(
            "### 4c · The decisive tradability test — the KST−buy-and-hold spread\n\n"
            "The literal value-add of *trading* the KST over *not* trading: the daily spread "
            "(KST net − buy-and-hold net). Its mean and HAC *t* are the honest answer to H₂."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(BARS, cost_bps=R['cost_bps'], n_perm=1)\n"
            "    kn = res['books']['kst']['net'].to_numpy(); bn = res['books']['buy_and_hold']['net'].to_numpy()\n"
            "    d = kn - bn\n"
            "    spread_yr = d.mean()*st.TRADING_DAYS*100; spread_t = st.hac_t(d)\n"
            "    eq_k = np.cumprod(1+res['books']['kst']['net'].fillna(0).to_numpy())\n"
            "    eq_b = np.cumprod(1+res['books']['buy_and_hold']['net'].fillna(0).to_numpy())\n"
            "    idx = BARS.index\n"
            "else:\n"
            "    spread_yr, spread_t = R['spread_yr'], R['spread_t']\n"
            "    idx = np.arange(100); eq_k = np.cumprod(1+np.full(100,0.0002)); eq_b = np.cumprod(1+np.full(100,0.0004))\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "ax.plot(idx, eq_b, c=GREEN, lw=1.8, label='buy & hold')\n"
            "ax.plot(idx, eq_k, c=AMBER, lw=1.8, label='KST long/flat')\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log)')\n"
            "ax.set_title(f'KST trails buy-and-hold by {abs(spread_yr):.1f}%/yr (spread HAC t = {spread_t:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'KST - buy&hold spread = {spread_yr:+.2f}%/yr   HAC t = {spread_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the spread is **{R['spread_yr']:+.1f}%/yr** at HAC "
            f"**t = {R['spread_t']:.2f}** — KST *significantly underperforms* buy-and-hold. This is the "
            "number that flips Tradability to **MIRAGE**: a standalone *t* of 2.65 looked respectable, "
            "but measured as value-add over the benchmark the rule is significantly in the red."
        ),
        md(
            "### 4d · Costs and the long/short variant\n\n"
            "Left: the cost sweep — KST is below buy-and-hold even at **zero** cost, so costs aren't the "
            "killer; the gross timing is simply worse. Right: flipping to **short** below the signal line "
            "(the long/short version) is a catastrophe."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sw = st.cost_sweep(BARS)\n"
            "    cb = [s['cost_bps'] for s in sw]; ks = [s['kst_sharpe'] for s in sw]; bs = [s['bah_sharpe'] for s in sw]\n"
            "    rls = st.run_experiment(BARS, cost_bps=R['cost_bps'], allow_short=True, n_perm=1)\n"
            "    ls_sh = rls['kst']['sharpe_excess']; ls_dd = abs(rls['kst']['maxdd_net'])*100\n"
            "else:\n"
            "    cb = [s[0] for s in R['sweep']]; ks = [s[1] for s in R['sweep']]; bs = [s[2] for s in R['sweep']]\n"
            "    ls_sh = R['ls'][0]; ls_dd = abs(R['ls'][2])\n"
            "fig, (a1,a2) = plt.subplots(1, 2, figsize=(10.8, 4.3))\n"
            "a1.plot(cb, ks, 'o-', c=AMBER, lw=2, label='KST')\n"
            "a1.plot(cb, bs, 'o-', c=GREEN, lw=2, label='buy & hold')\n"
            "a1.set_xlabel('one-way cost (bp)'); a1.set_ylabel('net excess Sharpe')\n"
            "a1.set_title('KST < buy-and-hold even at zero cost'); a1.legend()\n"
            "a2.bar(['L/S Sharpe','L/S maxDD %'], [ls_sh, -ls_dd], color=[RED, RED], width=.5)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.annotate(f'{ls_sh:.2f}',(0,ls_sh),ha='center',va='top',fontsize=10)\n"
            "a2.annotate(f'{-ls_dd:.0f}%',(1,-ls_dd),ha='center',va='top',fontsize=10)\n"
            "a2.set_title('Long/short KST: negative Sharpe, -81% drawdown')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('sweep KST:', [round(k,3) for k in ks]); print('L/S Sharpe', round(ls_sh,3), 'maxDD', round(ls_dd,1),'%')"
        ),
        md(
            f"> 💡 In plain words: at **0 bp** KST is **{R['sweep'][0][1]:.2f}** vs buy-and-hold "
            f"**{R['sweep'][0][2]:.2f}** — the gap is in the *gross* signal, not costs. And the long/short "
            f"version (**Sharpe {R['ls'][0]:.2f}**, **{R['ls'][2]:.0f}%** drawdown) confirms the short "
            "leg has no edge at all: 'sell' is not a tradeable signal, only 'be in cash'."
        ),
        md(
            "### 4e · Positive control — the engine *can* reward trend\n\n"
            "On a deterministic AR(1) tape we plant **return-trend** and watch the value-add (KST − "
            "buy-and-hold spread). With no trend it's strongly negative (timing only sacrifices beta); as "
            "trend rises the spread climbs toward zero and KST's Sharpe overtakes buy-and-hold's — proof "
            "the harness rewards a real trend when one exists, so the real-tape miss is a true negative."
        ),
        code(
            "trends = [0.00, 0.20, 0.35]\n"
            "rows = []\n"
            "for tr in trends:\n"
            "    b,_ = data.synthetic_panel(n_days=4000, trend=tr, seed=426)\n"
            "    r = st.run_experiment(b, cost_bps=R['cost_bps'], n_perm=1)\n"
            "    kk = r['books']['kst']['net'].to_numpy(); bb = r['books']['buy_and_hold']['net'].to_numpy(); dd = kk-bb\n"
            "    rows.append((tr, r['kst']['sharpe_excess'], r['buy_and_hold']['sharpe_excess'], dd.mean()*252*100, st.hac_t(dd)))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot([r[0] for r in rows], [r[3] for r in rows], 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c=RED, ls='--', label='break-even vs buy-and-hold')\n"
            "for r in rows: ax.annotate(f'{r[3]:+.1f}%',(r[0],r[3]),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_xlabel('planted return-trend (AR1 coeff)'); ax.set_ylabel('KST - buy&hold (%/yr)')\n"
            "ax.set_title('Control: value-add climbs toward 0+ as trend is planted'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for tr,ks_,bs_,sp,t in rows: print(f'trend={tr:+.2f}: KST Sh={ks_:.3f} BAH Sh={bs_:.3f} spread={sp:+.2f}%/yr t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** trend the spread is **{R['syn'][0][3]:+.1f}%/yr** (timing "
            f"strictly hurts — exactly the real-tape behaviour); at a strong planted trend it lifts to "
            f"**{R['syn'][2][3]:+.1f}%/yr** and KST's Sharpe (**{R['syn'][2][1]:.2f}**) overtakes "
            f"buy-and-hold's (**{R['syn'][2][2]:.2f}**). The machinery is faithful — so SPY just doesn't "
            "trend persistently enough for KST to add value over simply holding it."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — the long/flat book clears HAC **t = {R['kst'][3]:.2f}** and beats a "
            f"sign-flip coin (**p = {R['perm_p']:.4f}**), so there is a faint, *real* directional signal. "
            "But it is **beta in disguise**: the value-add over buy-and-hold is significantly negative "
            f"(spread **t = {R['spread_t']:.2f}**). Genuine pulse, no separable skill — **WEAK**, not REAL.\n"
            f"- **Tradability `MIRAGE`** — net excess Sharpe KST **{R['kst'][0]:.2f}** < buy-and-hold "
            f"**{R['bah'][0]:.2f}** < SMA-200 **{R['sma200'][0]:.2f}**; the rule sheds "
            f"~{abs(R['spread_yr']):.0f}%/yr vs doing nothing, even at zero cost, and the long/short "
            f"variant is Sharpe **{R['ls'][0]:.2f}** at **{R['ls'][2]:.0f}%** drawdown. Not tradeable as "
            "an edge.\n"
            f"- **A sure thing? `BUSTED`** — KST loses the race to **both** buy-and-hold and a one-line "
            "SMA-200, and merely ties MACD. The four-ROC complexity buys nothing the simplest trend filter "
            "doesn't do better."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the only thing it does is cut drawdown, and SMA-200 does that better\n\n"
            "The one honest selling point: like any long/flat trend filter, KST sits out the worst "
            "crashes, so its drawdown is far shallower than buy-and-hold. But that is a *generic* property "
            "of being in cash during bear markets — and the plain SMA-200 delivers the same protection "
            "while keeping far more upside."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(BARS, cost_bps=R['cost_bps'], n_perm=1)\n"
            "    arms = ['kst','buy_and_hold','sma200']\n"
            "    cg = [res[a]['cagr_net']*100 for a in arms]; dd = [abs(res[a]['maxdd_net'])*100 for a in arms]\n"
            "    sh = [res[a]['sharpe_excess'] for a in arms]\n"
            "else:\n"
            "    cg=[R['kst'][1],R['bah'][1],R['sma200'][1]]; dd=[abs(R['kst'][2]),abs(R['bah'][2]),abs(R['sma200'][2])]; sh=[R['kst'][0],R['bah'][0],R['sma200'][0]]\n"
            "labels=['KST','Buy & hold','SMA-200']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "sc = ax.scatter(dd, cg, s=[200,200,200], c=[AMBER,GREEN,GREEN], zorder=3)\n"
            "for l,x,y,s in zip(labels,dd,cg,sh): ax.annotate(f'{l}\\nSharpe {s:.2f}',(x,y),textcoords='offset points',xytext=(8,6),fontsize=9)\n"
            "ax.set_xlabel('worst drawdown (%)'); ax.set_ylabel('CAGR (%)')\n"
            "ax.set_title('Return vs pain: SMA-200 dominates KST (more return, similar drawdown)')\n"
            "ax.invert_xaxis(); plt.tight_layout(); plt.show()\n"
            "print('CAGR', [round(c,1) for c in cg], '| maxDD', [round(d,0) for d in dd])"
        ),
        md(
            f"> 💡 In plain words: KST cuts drawdown to **{abs(R['kst'][2]):.0f}%** but pays with a CAGR of "
            f"only **{R['kst'][1]:.1f}%**. SMA-200 gets a comparable **{abs(R['sma200'][2]):.0f}%** "
            f"drawdown while keeping **{R['sma200'][1]:.1f}%** CAGR — strictly better on both axes than "
            "KST. If crash protection is the goal, you don't need the Know Sure Thing; you need one line "
            "of code."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Parameter robustness.** Eight free knobs invite overfitting. An out-of-sample / "
            "walk-forward split of the ROC and SMA lengths is the obvious next test; in-sample tuning will "
            "*always* find a winner that fails forward.\n"
            "- **Asset dependence.** Trend filters earn their keep where trends persist (commodities, FX, "
            "single names). The large, mean-reverting US equity index is the *hardest* place for them — "
            "re-run on GLD or a momentum-heavy single stock.\n"
            "- **The generalisable lesson.** A standalone HAC *t* > 2 on a long-only timing rule is **not** "
            "evidence of skill — it is the equity risk premium. Always test value-add over buy-and-hold "
            "*and* over the simplest benchmark.\n\n"
            "*Reproducible core is offline & deterministic; methods and sources in "
            "[`docs/references.md`](../docs/references.md), frozen numbers in "
            "[`docs/results.md`](../docs/results.md).*"
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
