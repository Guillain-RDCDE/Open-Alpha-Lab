"""Generate the two narrative notebooks for Study 366 (Merger-Arbitrage).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the real deal book (the
documented entry/pre-deal closes live in ``merger_arbitrage/data.py``; the yfinance cache only
confirms still-listed names), and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (20 real all-cash US M&A
# deals, 2020-2024; 16 closed, 4 broke; entry = close ~1 day after announce).
R = dict(
    n=20, n_closed=16, n_breaks=4, win=80,
    entry_spread=8.99,        # gross spread at entry, all deals
    completed_spread=7.09,    # spread on the deals that closed (the "free lunch" number)
    break_mean=-19.39,        # mean realized return on the 4 breaks
    mean=1.79,                # MEAN realized arb return, all deals
    median=3.83, skew=-0.91,
    t=0.67, p_le0=0.240, boot_lo=-2.65, boot_hi=5.90,
    ann_mean=5.66, ann_t=2.09,
    costs=[(5, 1.79, 1.69, 0.63), (10, 1.79, 1.59, 0.59), (25, 1.79, 1.29, 0.48)],
    # synthetic: (edge, mean%, t, p_le0, breaks)
    syn=[(0.00, 0.52, 1.10, 0.142, 14), (0.02, 2.41, 4.77, 0.000, 14),
         (0.05, 5.24, 9.56, 0.000, 14)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
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

from merger_arbitrage import data, strategy as st

# the real deal book is reproducible offline (documented closes in the table); the yfinance
# cache only confirms still-listed names, so HAVE_REAL is effectively always True here.
HAVE_REAL = True
F = data.load_real()
print("real deal book :", len(F), "deals |", int((~F["completed"]).sum()), "breaks",
      "| yfinance cache present:", data.have_real())
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Free money in a takeover? — the merger-arbitrage spread 🤝\n"
            "### Buy the target below the offer, wait for the deal to close, pocket the difference — what's the catch?\n\n"
            + BADGES +
            "Here's a trade that sounds like free money. A company gets a **takeover offer** — say "
            "**$95 a share, all cash**. The next day the target trades at **$82**. Just *buy it at "
            "$82, wait for the deal to close, and collect $95.* A clean **+16%**, and it has nothing "
            "to do with whether the broad market goes up or down. People call it a *market-neutral "
            "coupon.*\n\n"
            "So what's the catch? The catch is the word **\"wait.\"** Deals don't always close. When one "
            "**breaks** — blocked by regulators, financing falls through, the buyer walks — the target "
            "**craters** back to where it traded before the offer, a 20–40% loss in a single day. This "
            "notebook shows that the spread isn't a free lunch at all: it's the **premium you collect "
            "for selling insurance** against the deal falling through.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the bootstrap and the skew? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** When a takeover closes, the target is **delisted** — its "
            "price history disappears from free data feeds. So our entry/exit prices are **documented "
            "public closes** carried in a real deal table (confirmed against live data for the two "
            "deals that *broke* and still trade). Every chart is drawn by the code beside it; house "
            "style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is there really a spread to collect? | **Yes.** Right after a deal is announced the "
            f"target trades about **{R['entry_spread']:.0f}% below** the cash offer, on average. |\n"
            "| Do most deals close and pay it? | **Yes — about 80% do.** That's the seductive part: you "
            "win most of the time. |\n"
            "| So is it free money? | **No.** In our book of 20 real deals, **4 broke** — and each "
            f"break cost about **{R['break_mean']:.0f}%**. Those four wipe out most of what the sixteen "
            f"winners earn: the *average* deal returns just **+{R['mean']:.1f}%**. |\n"
            "| Then why does it look so good? | **You're selling insurance.** Win small, often; lose "
            "big, rarely. A high win-rate with a rare disaster is what *writing insurance* looks like — "
            "not a free lunch. |\n\n"
            "> The spread is real. \"Free money\" is not. You're being paid a premium to bet the deal "
            "won't break — and on an honest tape that premium roughly equals the losses when it does."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When a company agrees to be bought for cash, its stock should jump to the offer price "
            "— but it doesn't quite. It trades a few percent below, because the deal takes months to "
            "close and isn't 100% certain. Buy the target, hold to the close, and harvest that gap. "
            "It's steady, it's market-neutral, and it wins almost every time.\"*\n\n"
            "This is **merger arbitrage** (\"risk arb\"), one of the oldest hedge-fund strategies. The "
            "logic is genuinely sound: the spread exists because closing takes time and carries risk. "
            "The seduction is the **win-rate** — most announced deals really do close. We'll rebuild "
            "the trade on real deals and look hard at the deals that *didn't*."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the spread were free money, it'd be one of the best risk-adjusted trades in markets: "
            "low correlation to stocks, a fat positive return, wins most of the time. Pensions and "
            "hedge funds would (and do) pile in. But there's a hidden cost the pitch glosses over: "
            "**what you give up when a deal breaks.** The right question isn't *\"how often do I win?\"* "
            "— it's *\"when I add up the small frequent wins and the rare giant losses, is there "
            "anything left?\"* A trade that wins 80% of the time can still have an **expected return of "
            "zero** if the 20% of losers are big enough. That's the whole game here."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We build a book of **{R['n']} real announced all-cash US deals** (2020–2024) — on purpose "
            f"a mix of clean closes **and** famous breaks (JetBlue/Spirit, TD/First Horizon, "
            "Avangrid/PNM, MaxLinear/Silicon Motion).\n\n"
            "1. **Enter after the news.** Buy the target at the close **one day after** the deal is "
            "announced (no cheating — you can't trade before it's public).\n"
            "2. **Mark the outcome honestly.** If the deal closes, you collect the offer. If it "
            "**breaks**, the target snaps back to where it traded *before* the bid — that's your loss.\n"
            "3. **Add it all up.** Average the realized return across *all* deals — winners and "
            "breakers together — and ask whether what's left is real, or just inside the noise of a "
            "small, lumpy sample."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, every deal in the book.** Green bars are the spread you *collected* (the deal "
            "closed); red bars are what you *lost* when it broke. Notice how a few deep red bars "
            "outweigh a lot of shallow green ones."
        ),
        code(
            "F2 = F.sort_values('ret')\n"
            "cols = [GREEN if c else RED for c in F2['completed']]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 5.0))\n"
            "ax.barh(F2['target'], F2['ret']*100, color=cols)\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('realized arb return per deal (%)')\n"
            "ax.set_title('16 small green wins, 4 deep red breaks — the merger-arb payoff')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('deals that closed:', int(F['completed'].sum()), ' deals that broke:', int((~F['completed']).sum()))"
        ),
        md(
            f"There's the shape. **{R['n_closed']} deals closed** and paid a tidy spread; "
            f"**{R['n_breaks']} broke** and cost about **{R['break_mean']:.0f}%** each. The breaks are "
            "few, but they're *deep* — and that asymmetry is the entire story."
        ),
        md(
            "**Now the punchline: what you *see* vs what you *get*.** The pitch quotes the spread on "
            "deals that close. The tape pays the average across *all* deals, breaks included."
        ),
        code(
            "comp = F[F['completed']]\n"
            "see = comp['spread'].mean()*100          # 'free lunch' number: spread if it closes\n"
            "get = st.summarize(F)['mean']*100         # honest mean across ALL deals\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "bars = ax.bar(['the spread you \\\"see\\\"\\n(deals that close)', 'what you actually \\\"get\\\"\\n(all deals, breaks too)'],\n"
            "              [see, get], color=[GREY, GREEN], width=.55)\n"
            "for b,v in zip(bars,[see,get]): ax.annotate(f'{v:.1f}%',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom',fontsize=12)\n"
            "ax.set_ylabel('average return per deal (%)'); ax.axhline(0,c='k',lw=.8)\n"
            "ax.set_title('The break tail eats most of the spread')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'spread if it closes: {see:.1f}%   ->   average across all deals: {get:.1f}%')"
        ),
        md(
            f"The headline **+{R['completed_spread']:.1f}%** coupon shrinks to **+{R['mean']:.1f}%** once "
            "the four breaks are counted. Most of the apparent free lunch was simply *not pricing the "
            "disaster*."
        ),
        md(
            "**Is even that +1.8% real?** With only 20 deals and a few huge losers, the average is "
            "shaky. Here's the honest test: resample the deal book thousands of times (bootstrap) and "
            "see how often the average comes out **zero or negative.**"
        ),
        code(
            "r = st.arb_returns(F)\n"
            "rng = np.random.default_rng(366)\n"
            "boot = np.array([r[rng.integers(0,len(r),len(r))].mean() for _ in range(20000)])\n"
            "p0 = (boot <= 0).mean()\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(boot*100, bins=60, color=GREY, alpha=.85, label='resampled average deal')\n"
            "ax.axvline(0, c=RED, lw=2, label='break-even (zero)')\n"
            "ax.axvline(r.mean()*100, c=GREEN, lw=2.5, label=f'our book ({r.mean()*100:.1f}%)')\n"
            "ax.set_xlabel('average realized return per deal (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'A quarter of resamples are <= 0 — the edge is inside the noise (p={p0:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'P[average deal <= 0] = {p0:.2f}  -> about 1 in 4. Not distinguishable from a fair bet.')"
        ),
        md(
            f"The green line (our book) sits well inside the cloud, and about **{R['p_le0']*100:.0f}%** of "
            "resamples land at zero or below. In plain terms: **the average deal's profit can't be told "
            "apart from a fair bet** — exactly what you'd expect if the spread is just the fair price of "
            "the break insurance."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The spread is real and you win **{R['win']:.0f}%** of the time — but "
            f"the average deal (**+{R['mean']:.1f}%**) is statistically indistinguishable from zero, and "
            "the headline spread was gross of a break tail that eats most of it. Real as lore, weak as "
            "edge.\n"
            "- **Tradability — Fragile.** Costs barely matter, but you're **short a deal-break put**: "
            "one blocked deal erases a year of spread, and breaks cluster (antitrust waves, financing "
            "freezes), so the tail doesn't diversify away.\n"
            "- **\"Free lunch\"? — Busted.** The steady coupon is an **insurance premium**. Win small "
            "often, lose big rarely — that's selling insurance, not finding free money."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the insurance you're really selling\n\n"
            "Forget statistics for a second. Picture the payoff: a row of small steady wins, then "
            "occasionally a cliff. That's the profile of an **insurance company** — and it tells you "
            "exactly how it fails. Not gradually, but all at once, when many deals break together."
        ),
        code(
            "wins = F[F['completed']]['ret']*100\n"
            "loss = F[~F['completed']]['ret']*100\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.scatter(np.arange(len(wins)), sorted(wins), c=GREEN, s=60, label=f'{len(wins)} wins (avg +{wins.mean():.1f}%)')\n"
            "ax.scatter(np.arange(len(loss))+len(wins)+1, sorted(loss), c=RED, s=80, label=f'{len(loss)} breaks (avg {loss.mean():.1f}%)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('realized return (%)'); ax.set_xticks([])\n"
            "ax.set_title('Picking up pennies in front of a steamroller')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'win {wins.mean():.1f}% on average, lose {loss.mean():.1f}% when it breaks -> short volatility')"
        ),
        md(
            "Real merger-arb desks survive by holding **dozens** of deals at once to diversify the "
            "idiosyncratic break risk. But the dangerous breaks are **correlated** — a hawkish "
            "antitrust regime or a frozen financing market can break a whole wave of deals together, "
            "and *that* is the loss no amount of diversification removes. Costs aren't the killer here; "
            "the **fat left tail** is. You can run this as a *style* (and pros do), but it is not the "
            "free coupon the pitch describes."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Price the tail directly.** Estimate each deal's break probability from the spread "
            "itself (a wide spread is the market screaming *this one's shaky*), then check whether the "
            "spread *over*-pays for that probability. Our finding says: barely, if at all.\n"
            "- **Diversify and stress it.** Build a 50-deal book and then force a *correlated* break "
            "wave (e.g. 2022's antitrust crackdown) — watch the 'market-neutral' coupon take a "
            "market-sized hit.\n"
            "- **The family resemblance.** Covered calls, short-vol carry, selling puts — same payoff "
            "shape (win small often, lose big rarely). If you understand one, you understand all of "
            "them.\n\n"
            "*Think the spread beats fair insurance pricing? Take a bigger, cleaner deal book, count "
            "every break, and show the average landing **above zero with a t past 2** — then we'll talk.*"
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
            "# Merger-Arbitrage — a quantitative teardown 🔬\n"
            "### Per-deal realized arb returns · the break tail & negative skew · a one-sample *t* + "
            "bootstrap null · costs · a synthetic fair-bet / planted-edge control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We separate "
            "what merger-arb's pitch fuses: a **real, visible spread** from the **break tail that pays "
            "for it**. The position is economically **short a deal-break put** (Mitchell & Pulvino "
            "2001) — written index-put-like returns, negatively skewed — so the decisive object is not "
            "the average spread (it's positive, as the lore says) but its **mean net of the tail**, and "
            "that mean's **standard error** on a small, fat-tailed sample.\n\n"
            "> ⚠️ **Data + survivorship note.** Acquired targets **delist** and vanish from free feeds, "
            "so a live-feed-only book would over-represent *broken* deals — survivorship that points "
            "*against* the strategy. We mark entry/exit from **documented closes** (confirmed against "
            "live data for the surviving names FHN, SIMO), named on the Signal axis. The book is small "
            "(20) and curated with a high break share by design. Methods in "
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
            f"| **Signal** | `WEAK` | mean realized arb return **+{R['mean']:.2f}%/deal** vs the "
            f"**+{R['completed_spread']:.1f}%** spread-if-it-closes; one-sample **t = {R['t']:.2f}**, "
            f"bootstrap **p(≤0) = {R['p_le0']:.2f}**, 90% CI **[{R['boot_lo']:.1f}%, {R['boot_hi']:.1f}%]** "
            "— positive but **fails t ≥ 2**, skew **" + f"{R['skew']:.2f}" + "**. |\n"
            f"| **Tradability** | `FRAGILE` | short a deal-break put: **{R['n_breaks']}/{R['n']}** breaks "
            f"at **{R['break_mean']:.0f}%** each; net of 25-bps costs the mean barely moves — the **tail "
            "and its correlation**, not cost, are binding. |\n"
            f"| **Free lunch?** | `BUSTED` | win-rate **{R['win']:.0f}%** with a deep, rare loser = a "
            "written put. On an honest tape the premium ≈ the expected loss. |\n\n"
            "> 💡 In plain words: the spread really is there, and most deals really do close — but strip "
            "out the four breaks and the average deal's profit can't be told apart from zero. You're "
            "collecting an insurance premium that, on this sample, roughly equals the claims you pay."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a deal offer cash $K$ per share; let $P_0$ be the target's close one day after "
            "announcement (the arb entry) and $S$ its un-bid standalone level. The realized arb return "
            "is\n\n"
            "$$ r = \\begin{cases} K/P_0 - 1 & \\text{deal closes (prob. } 1-q) \\\\ "
            "S/P_0 - 1 & \\text{deal breaks (prob. } q) \\end{cases} $$\n\n"
            "- **H₁ (the spread is an edge).** $\\mathbb{E}[r] > 0$ by *more* than fair compensation for "
            "the break tail — i.e. the spread $K/P_0-1$ exceeds $\\frac{q}{1-q}\\,(1-S/P_0)$.\n"
            "- **H₂ (it's deployable).** That excess survives costs and, crucially, the **correlation** "
            "of breaks (a book of many deals doesn't diversify a regime-driven break wave).\n"
            "- **H₃ (\"free lunch\").** The high win-rate reflects a genuine edge, not the geometry of a "
            "written put (win small often, lose big rarely).\n\n"
            "We find **H₁ not rejected but not supported** ($\\bar r>0$, $t<2$), **H₂ rejected** (thin "
            "edge, fat correlated tail), **H₃ rejected** (negative skew, premium ≈ expected loss). The "
            "claim is true exactly where it's uninformative (most deals close) and unproven exactly "
            "where it would pay (a spread that *over*-prices the break risk)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The whole teardown is one expectation against its standard error:\n\n"
            "$$ \\bar r = \\frac{1}{N}\\sum_i r_i,\\qquad t = \\frac{\\bar r}{s_r/\\sqrt{N}}. $$\n\n"
            "Two things make $t$ small here. **(i) $N$ is tiny** — a couple of dozen deals. **(ii) "
            "$s_r$ is inflated by the left tail** — a handful of $-20\\%$ to $-40\\%$ breaks dominate "
            "the variance, so $s_r/\\sqrt{N}$ swamps a few-percent $\\bar r$. A raw **win-rate** is the "
            "wrong lens entirely: a written put wins ~80–90% of the time *by construction*, so a high "
            "hit-rate is the null, not the signal. And because the distribution is skewed, the "
            "$t$-stat itself is fragile — the honest instrument is a **nonparametric bootstrap** of the "
            "mean, whose left tail (P[mean ≤ 0]) is the small-sample, fat-tail answer."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Deal book.** {R['n']} real announced all-cash US deals, 2020–2024 (offer, "
            "announce/resolve dates, outcome). A deliberate mix of clean closes and high-profile breaks.\n"
            "- **Entry.** Target close **1 day after** the announcement (no look-ahead). **Break mark.** "
            "The ~1-week pre-deal standalone close.\n"
            "- **Per-deal return.** $K/P_0-1$ if it closes, $S/P_0-1$ if it breaks; annualized only as "
            "an aside (annualizing short overlapping bets inflates $t$ — *not* a Signal statistic).\n"
            "- **Null #1 (one-sample t).** $\\bar r$ vs 0, reported with the caveat that skew weakens it.\n"
            "- **Null #2 (bootstrap).** 20,000 resamples of the book; report the 90% CI of the mean and "
            "$p=\\Pr[\\text{mean}\\le 0]$ — the small-sample workhorse.\n"
            "- **Costs.** One-way bps on entry **and** exit (one round-trip / deal); long-only target → "
            "no borrow.\n"
            "- **Positive control.** A deterministic deal book with a **known** break probability and "
            "the spread **pinned to fair-insurance value**, plus an `edge` knob: at `edge=0` the test "
            "must NOT reach significance; a planted edge must light it up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The distribution — small wins, fat-left losses\n\n"
            "The realized arb returns, sorted. The story is the asymmetry: a dense band of small "
            "positive spreads, then a few deep negative breaks that set the variance."
        ),
        code(
            "r = st.arb_returns(F)\n"
            "cols = [GREEN if x>0 else RED for x in r[np.argsort(r)]]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.3))\n"
            "ax.bar(range(len(r)), np.sort(r)*100, color=cols)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.axhline(r.mean()*100, c=AMBER, ls='--', lw=2, label=f'mean +{r.mean()*100:.1f}%')\n"
            "ax.set_ylabel('realized arb return (%)'); ax.set_xlabel('deal (sorted)')\n"
            "ax.set_title(f'Negative skew = {st.skewness(r):.2f}: the tail is the trade')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'mean {r.mean()*100:.2f}%  median {np.median(r)*100:.2f}%  skew {st.skewness(r):.2f}  (mean << median-ish, dragged by the tail)')"
        ),
        md(
            f"> 💡 In plain words: skew **{R['skew']:.2f}** and a mean (**+{R['mean']:.1f}%**) pulled well "
            f"below the median (**+{R['median']:.1f}%**) — the signature of a written put. The average "
            "deal is dragged down by a handful of breaks, not lifted by the many small wins."
        ),
        md(
            "### 4b · The decisive test — bootstrap the mean of a skewed book\n\n"
            "A *t*-stat assumes roughly symmetric errors; this book is left-skewed, so we bootstrap. "
            "20,000 resamples give the sampling distribution of the mean; the right-of-zero question is "
            "$\\Pr[\\text{mean}\\le 0]$."
        ),
        code(
            "rng = np.random.default_rng(366)\n"
            "boot = np.array([r[rng.integers(0,len(r),len(r))].mean() for _ in range(20000)])\n"
            "lo, hi = np.percentile(boot,[5,95])*100; p0 = (boot<=0).mean()\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(boot*100, bins=60, color=GREY, alpha=.85, label='bootstrap mean (20,000 draws)')\n"
            "ax.axvline(0, c=RED, lw=2, label='zero (fair bet)')\n"
            "ax.axvline(r.mean()*100, c=GREEN, lw=2.5, label=f'observed mean {r.mean()*100:.1f}%')\n"
            "ax.axvspan(lo, hi, color=AMBER, alpha=.15, label=f'90% CI [{lo:.1f}%, {hi:.1f}%]')\n"
            "ax.set_xlabel('mean realized arb return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'90% CI straddles zero; P[mean<=0] = {p0:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'one-sample t = {st.welch_t(r):.2f}   bootstrap p(mean<=0) = {p0:.3f}   90% CI [{lo:.1f}%, {hi:.1f}%]')"
        ),
        md(
            f"> 💡 In plain words: the 90% CI runs **[{R['boot_lo']:.1f}%, {R['boot_hi']:.1f}%]** — it "
            f"**contains zero** — and **{R['p_le0']*100:.0f}%** of resamples are non-positive. The "
            f"one-sample *t* is **{R['t']:.2f}**, far below 2. H₁ is **not rejected, not supported**: a "
            "positive whisper that a fair bet could easily have produced."
        ),
        md(
            "### 4c · The annualization trap + costs — neither rescues it\n\n"
            "Two checks. **Left:** annualizing per-deal returns by their short holding periods inflates "
            "the *t* to ~2 — a methodological mirage we explicitly **do not** lean on. **Right:** "
            "one-way costs barely dent a multi-month hold, so cost is **not** the binding constraint."
        ),
        code(
            "ra = st.annualized(F)\n"
            "tt = [st.welch_t(r), st.welch_t(ra)]\n"
            "fig, (a1,a2) = plt.subplots(1,2, figsize=(10.4,4.2))\n"
            "a1.bar(['per-deal\\n(honest)','annualized\\n(inflated)'], tt, color=[GREEN, GREY], width=.55)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tt): a1.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "a1.set_ylabel('t-stat'); a1.set_title('Annualizing short bets fakes significance'); a1.set_ylim(0,2.4); a1.legend()\n"
            "cs = [5,10,25]; nets = [st.net_of_costs(F, cost_bps=c) for c in cs]\n"
            "gm = [n['gross_mean']*100 for n in nets]; nm = [n['net_mean']*100 for n in nets]; xx=np.arange(len(cs))\n"
            "a2.bar(xx-.2, gm, .4, color=GREEN, label='gross'); a2.bar(xx+.2, nm, .4, color=GREY, label='net')\n"
            "a2.set_xticks(xx); a2.set_xticklabels([f'{c}bps' for c in cs]); a2.set_ylabel('mean per-deal return (%)')\n"
            "a2.set_title('Cost barely matters — the tail does'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-deal t', round(tt[0],2), '| annualized t', round(tt[1],2), '(inflated, NOT used) | net@25bps', round(nm[-1],2),'%')"
        ),
        md(
            f"> 💡 In plain words: the annualized **t = {R['ann_t']:.2f}** is where the \"merger-arb makes "
            "mid-single-digits a year\" lore comes from — but it's an artifact of stacking short "
            "overlapping bets, not an honest standard error. The per-deal **t = " + f"{R['t']:.2f}" +
            "** is the real one. Costs shave tenths of a point off a months-long hold; the binding "
            "constraint is the **break tail**, full stop."
        ),
        md(
            "### 4d · Fair-bet & power control — we know the truth here\n\n"
            "On a deterministic deal book (n = 250, break prob 8%, break loss −28%) with the spread "
            "**pinned to fair-insurance value**: at `edge = 0` the arb is a genuine fair bet and the "
            "test must stay below t = 2; a planted `edge` must light it up. Both hold — proving the "
            "engine is unbiased *and* that a fair spread does not fake significance."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.02, 0.05):\n"
            "    syn = data.synthetic_book(n_deals=250, edge=edge, seed=366); s = st.summarize(syn)\n"
            "    res.append((edge, s['mean']*100, s['t'], s['p_le0'], s['n_breaks']))\n"
            "labels = [f'edge\\n{e*100:.0f}%/deal' for e,_,_,_,_ in res]; tvals = [r[2] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN, GREEN], width=.55)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('one-sample t'); ax.set_title('Control: fair spread stays <2; a real edge lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,m,t,p,b in res: print(f'edge {e*100:+.0f}%/deal: mean={m:.2f}% t={t:.2f} p(<=0)={p:.3f} breaks={b}')"
        ),
        md(
            f"> 💡 In plain words: a **fair** spread (edge 0) gives **t = {R['syn'][0][2]:.2f}** — below 2, "
            f"no false positive — while a **+2%/deal** real edge jumps to **t = {R['syn'][1][2]:.2f}**. "
            "So the machinery is honest, and the real-tape *t* of ~0.7 is exactly what an *absent-or-"
            "tiny* per-deal edge looks like through a 20-deal keyhole. The sample size and the tail are "
            "the verdict."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — mean realized arb return **+{R['mean']:.2f}%/deal** at one-sample "
            f"**t = {R['t']:.2f}** / bootstrap **p(≤0) = {R['p_le0']:.2f}** (90% CI "
            f"[{R['boot_lo']:.1f}%, {R['boot_hi']:.1f}%], skew {R['skew']:.2f}). Literature support + a "
            "positive-but-insignificant, skew-fragile point estimate on a small, survivorship-shaped "
            "book ⇒ WEAK, not REAL (the t≥2 bar is not met).\n"
            f"- **Tradability `FRAGILE`** — short a deal-break put: **{R['n_breaks']}/{R['n']}** breaks at "
            f"**{R['break_mean']:.0f}%**. Net of 25-bps costs the mean barely moves; the binding "
            "constraints are the **fat left tail** and its **correlation** (a regime-driven break wave "
            "hits the whole book), not cost. No NAV-scale free lunch.\n"
            f"- **Free lunch? `BUSTED`** — win-rate **{R['win']:.0f}%** with a deep, rare loser is a "
            "**written put**; the premium ≈ the expected loss. A short-volatility carry style, not a "
            "market-neutral coupon."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — break probability vs the spread\n\n"
            "The operational truth in one picture: how big a break probability does each deal's spread "
            "*imply*, and how does that compare to the realized break rate? If the spread merely prices "
            "the break risk, there's no excess to harvest."
        ),
        code(
            "# implied break prob from the spread, assuming the standalone snap-back loss:\n"
            "# spread = q/(1-q) * loss  ->  q = spread / (spread + loss)\n"
            "sp = F['spread'].values\n"
            "loss = np.abs((F['predeal']/F['entry'] - 1.0).values)\n"
            "loss = np.where(loss < 0.02, 0.20, loss)   # floor for clean deals lacking a deep snap-back\n"
            "q_impl = sp / (sp + loss)\n"
            "realized_q = (~F['completed']).mean()\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.scatter(q_impl*100, sp*100, c=[RED if not c else GREEN for c in F['completed']], s=60)\n"
            "ax.axvline(realized_q*100, c=GREY, ls='--', label=f'realized break rate {realized_q*100:.0f}%')\n"
            "ax.set_xlabel('break probability IMPLIED by the spread (%)'); ax.set_ylabel('entry spread (%)')\n"
            "ax.set_title('Wide spreads sit on shaky deals — the spread prices the risk'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'mean implied break prob {q_impl.mean()*100:.0f}% vs realized {realized_q*100:.0f}% — the spread is paying for the tail, not beating it')"
        ),
        md(
            "> 💡 In plain words: the deals with the **fattest spreads are exactly the ones that broke** "
            "(the red points sit at high implied break probability) — the market wasn't handing out free "
            "money, it was *charging more where the deal was shakier*. Averaged across the book, the "
            "spread implies a break rate in the same ballpark as the one we realized: the premium is "
            "**compensation for the tail, not an edge over it.** There is no sizing, no cost assumption "
            "and no threshold that turns fairly-priced insurance into a free lunch — **the spread is the "
            "fair price of the risk you're carrying.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Mitchell & Pulvino, properly.** Regress the arb book on the market and on a written-"
            "put factor; the negative skew and the down-market beta are the real risk story (J. Finance "
            "2001). Our 20-deal book is the cartoon; their thousands-of-deals panel is the proof.\n"
            "- **Correlated breaks.** Model break probability with a common factor (antitrust regime, "
            "credit conditions) and watch the 'market-neutral' coupon acquire a fat down-market beta — "
            "the loss diversification can't remove.\n"
            "- **Spread → break-prob calibration.** Invert the spread for an implied break probability "
            "per deal and test whether it *over*-prices the realized rate. Our 4d-style control says: "
            "barely, if at all.\n\n"
            "*The reproducible core is offline and deterministic; entry/exit marks are documented closes "
            "(acquired targets delist). Methods and sources: [`docs/references.md`](../docs/references.md); "
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
