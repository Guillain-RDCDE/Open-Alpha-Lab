"""Generate the two narrative notebooks for Study 745 (Corporate-Jet-Index).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached monthly
closes under ../_cache/ (each firm + SPY) and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with no
network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance MONTHLY
# total-return closes, hardcoded ~24-firm jet-perk table, as-of 2026-06-30; 12 heavy /
# 12 low, panel 2011-01 -> 2026-06, 186 months; SPY benchmark, HAC/Newey-West t).
R = dict(
    asof="2026-06-30", n_firms=24, n_heavy=12, n_low=12,
    fingerprint="ad201b5ed46c",
    n_months=186, start="2011-01", end="2026-06",
    # long/short (low - heavy), excess of market
    ls_ann=2.11, ls_hac_t=0.47, ls_naive_t=0.48, ls_sharpe=0.12, ls_lags=4,
    heavy_x_ann=-0.34, low_x_ann=1.76,
    # alpha vs beta
    alpha_ann=8.81, alpha_t=2.10, ls_beta=-0.45, heavy_beta=1.31, low_beta=0.87,
    # raw compounding
    cagr_low=16.15, cagr_spy=14.17, cagr_heavy=12.19,
    # drop the mega-cap flyers
    ex_heavy_x=-3.83, ex_low_x=1.76, ex_ls_ann=5.80, ex_hac_t=1.08,
    ex_alpha_ann=12.08, ex_alpha_t=2.22,
    # costs
    gross_ann=2.11, cost_ann=0.24, borrow_ann=0.50, net_ann=1.36,
    # per-name excess-of-market ann (the founder-growth confound)
    winners=[("TSLA", 53.5), ("GOOGL", 10.1), ("META", 7.4), ("ANF", 4.7),
             ("GE", 1.9), ("ORCL", 1.6)],
    laggards=[("IEP", -14.1), ("CMCSA", -7.5), ("OXY", -5.7), ("LVS", -5.3),
              ("WYNN", -4.8), ("DIS", -3.9)],
    # synthetic control: (planted_bps_month, ls_ann, hac_t)
    syn=[(0.0, -1.17, -0.49), (-80.0, 8.76, 3.53)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Governance_discount%3F: Misattributed](https://img.shields.io/badge/Governance_discount%3F-Misattributed-8b949e?style=flat-square)\n\n"
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

from corporate_jet_index import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES, FIRMS = data.load_real()
    PANEL = st.long_short_panel(PRICES, FIRMS, min_names=4)
    S = st.summarize(PANEL)
else:
    PRICES = FIRMS = PANEL = S = None
print("real price cache present:", HAVE_REAL,
      "| panel months:", (0 if PANEL is None else len(PANEL)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Is a jet-loving CEO a governance short? ✈️\n"
            "### The corporate jet as a sell signal — a beautiful story, tested\n\n"
            + BADGES +
            "In 2006 a finance professor named David Yermack noticed something delicious. The moment a "
            "company first admits, in the fine print of its proxy, that the CEO flies the corporate jet "
            "for **personal** trips, the stock tends to **lag the market by about 4% a year** afterward. "
            "The jet, he argued, is a *tell*: a CEO helping themselves to the company plane is a CEO who "
            "might be helping themselves to a lot of things. Cue the obvious trade — **short the flyers, "
            "buy the penny-pinchers**, and pocket the governance discount.\n\n"
            "It's the kind of idea you *want* to be true. So we built it: a labelled list of ~24 "
            "large-caps split into **jet-loving CEOs** (Ellison, Musk, Zuckerberg, Adelson, Wynn, "
            "Jeffries…) and **frugal ones** (Buffett, Costco, Walmart, Texas Instruments…), and we ran "
            "the long/short. The answer is a great little lesson in *why the sign being right doesn't "
            "make the trade real.*\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the market-model alpha, the "
            "betting-against-beta confound spelled out? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Real governance-perk databases aren't free, so we **hardcode "
            "a transparent table** from proxy filings and press coverage (and the heavy/frugal call is "
            "ours — a judgement at the edges). And the most infamous jet abusers — Tyco, WorldCom, "
            "Enron — *went bankrupt and vanished from the tape*, which, as we'll see, biases the test "
            "**against** the very thing we're looking for. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did the frugal firms beat the flyers? | **Yes, actually.** Over 2011–2026 the frugal "
            f"basket compounded at **{R['cagr_low']:.1f}%/yr** vs the flyers' **{R['cagr_heavy']:.1f}%** "
            "— about the ~4pp gap Yermack found. The *sign* is real. |\n"
            "| So the short works? | **No.** Month to month that gap is **pure noise** "
            f"(*t* = **{R['ls_hac_t']:.2f}**, where you need 2). A net **+{R['net_ann']:.1f}%/yr** you "
            "can't distinguish from luck. |\n"
            "| But there's a *significant* number in here… | **There is — and it's a trick.** The one "
            f"stat that clears the bar (**+{R['alpha_ann']:.1f}%/yr** 'alpha') comes entirely from the "
            "long/short being **short high-risk stocks and long low-risk stocks** — a well-known "
            "low-volatility premium, *not* anything about jets. |\n"
            "| What breaks it? | **One Tesla.** The single best-performing 'flyer' (Elon Musk's Tesla, "
            f"**+{R['winners'][0][1]:.0f}%/yr** vs the market) single-handedly drags the short basket up. "
            "The surviving jet-lovers are disproportionately founder-run *winners*. |\n\n"
            "> The corporate jet is a real *correlate* of trouble — but on a tradable tape it's a "
            "correlate of **high-beta founder-growth stocks**, not an independent edge. Right sign, "
            "wrong reason, no trade."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When a CEO starts flying the company jet on personal time, it's a red flag — those "
            "firms underperform. Short the flyers, own the frugal ones.\"*\n\n"
            "This is a real, cited academic finding — Yermack's *Flights of Fancy* (2006) — not a "
            "message-board myth, which is exactly why it's worth taking seriously. The logic is clean: "
            "personal jet use is a **visible, disclosed symptom** of the deeper thing you can't see "
            "(weak boards, entrenched managers, empire-building). If the symptom is public and the "
            "disease is real, the stock should carry a discount you can trade. We'll test the strongest "
            "version: a long/short that's **long the frugal, short the flyers**, and see if the spread "
            "is anything more than the market's mood."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a *disclosed, public* perk predicted a 4%/yr underperformance, that would be genuinely "
            "shocking — it would mean the market ignores a red flag sitting in plain sight in every "
            "proxy statement, year after year. That's a big claim about market efficiency, and a "
            "standing money machine. But there's a catch the story never mentions: a jet-loving CEO is "
            "rarely *just* a jet-loving CEO. They tend to be **founders**, **empire-builders**, "
            "**high-conviction growth bosses** — and those traits come with a stock profile (high beta, "
            "high growth) that has its own, totally separate, expected return. So the real question "
            "isn't \"do flyers underperform?\" but \"do they underperform **for the governance reason**, "
            "*after* you strip out the fact that they're high-octane growth stocks?\" That's where the "
            "story usually dies."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We hardcode **~{R['n_firms']} large-caps** (as-of {R['asof']}; **{R['n_heavy']}** flagged "
            f"jet-heavy, **{R['n_low']}** frugal peers) and run a monthly **long/short**:\n\n"
            "1. **Two baskets.** Equal-weight the frugal names, equal-weight the flyers. Each flyer only "
            "enters the short **after** its jet perk is public (no cheating with hindsight).\n"
            "2. **Long/short = frugal − flyers**, measured *in excess of the market* so we're not just "
            "paid for stocks going up. Positive = Yermack was right.\n"
            "3. **Judge it honestly.** A Newey-West *t*-stat (the noise-robust kind) on the monthly "
            "spread. And — the decisive move — a **market-model** check that asks: is this 'edge' just "
            "**beta**? Then costs, short-borrow, and a stress test dropping the mega-cap winners.\n\n"
            "**What would make us say 'mirage':** if the tradable spread can't clear *t* = 2, or if the "
            "only significant number vanishes once we account for the baskets' different riskiness."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the part that flatters the story.** Here's a dollar invested in each basket "
            "(dividends reinvested) over the sample. The frugal basket really does finish ahead of the "
            "flyers — and even ahead of the S&P. The believers get their headline."
        ),
        code(
            "if HAVE_REAL:\n"
            "    hb = st.basket_returns(PRICES, FIRMS, heavy=True).reindex(PANEL.index).fillna(0)\n"
            "    lb = st.basket_returns(PRICES, FIRMS, heavy=False).reindex(PANEL.index).fillna(0)\n"
            "    mk = st.monthly_returns(PRICES)['SPY'].reindex(PANEL.index).fillna(0)\n"
            "    cl, ch, cm = (1+lb).cumprod(), (1+hb).cumprod(), (1+mk).cumprod()\n"
            "    idx = PANEL.index\n"
            "else:\n"
            "    idx = np.arange(R['n_months'])\n"
            "    cl = (1+R['cagr_low']/100/12)**np.arange(R['n_months'])\n"
            "    ch = (1+R['cagr_heavy']/100/12)**np.arange(R['n_months'])\n"
            "    cm = (1+R['cagr_spy']/100/12)**np.arange(R['n_months'])\n"
            "fig, ax = plt.subplots(figsize=(9.6, 5.0))\n"
            "ax.plot(idx, cl, c=GREEN, lw=2, label=f\"frugal basket ({R['cagr_low']:.1f}%/yr)\")\n"
            "ax.plot(idx, cm, c=GREY, lw=1.6, ls='--', label=f\"S&P 500 ({R['cagr_spy']:.1f}%/yr)\")\n"
            "ax.plot(idx, ch, c=RED, lw=2, label=f\"jet-loving basket ({R['cagr_heavy']:.1f}%/yr)\")\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log, total return)')\n"
            "ax.set_title('The frugal basket really did out-compound the flyers by ~4pp/yr'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"frugal {R['cagr_low']:.1f}%/yr  >  S&P {R['cagr_spy']:.1f}%/yr  >  flyers {R['cagr_heavy']:.1f}%/yr\")"
        ),
        md(
            f"So far, so Yermack: **{R['cagr_low']:.1f}%/yr** frugal vs **{R['cagr_heavy']:.1f}%/yr** "
            "flyers, the right sign and roughly the right size. If we stopped here we'd write it up as a "
            "win. But a compounding chart hides all the wiggles — and the wiggles are where the truth is."
        ),
        md(
            "**Now the honest test — is the monthly spread anything but noise?** We take the long/short "
            "(frugal minus flyers, in excess of the market) *month by month* and ask whether its average "
            "is reliably above zero. Here's the running total of that spread, with the *t*-stat."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = PANEL['ls']; t = S['ls_hac_t']; ann = S['ls_mean_ann']*100\n"
            "    cum = ls.cumsum()*100; idx = PANEL.index\n"
            "else:\n"
            "    rng=np.random.default_rng(745); ls=rng.normal(R['ls_ann']/100/12, 0.03, R['n_months'])\n"
            "    t=R['ls_hac_t']; ann=R['ls_ann']; cum=np.cumsum(ls)*100; idx=np.arange(R['n_months'])\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.6))\n"
            "ax.plot(idx, cum, c=AMBER, lw=2)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('cumulative long/short excess return (%)')\n"
            "ax.set_title(f'Frugal - flyers spread: {ann:+.1f}%/yr, but HAC t = {t:.2f} (noise)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'long/short {ann:+.1f}%/yr | HAC t = {t:.2f} -> you need |t|>=2, this is a coin flip')"
        ),
        md(
            f"A wandering line that ends up mildly positive — and a *t*-stat of **{R['ls_hac_t']:.2f}**. "
            "That's not an edge; that's a random walk that happened to drift up. The ~4pp compounding "
            "gap is **real but statistically silent**: too few names, too much month-to-month scatter. "
            "The believers' headline survives the eye and dies at the *t*-test."
        ),
        md(
            "**Where the whole story breaks — one chart.** Here's every flyer's return *versus the "
            "market*. The thesis says these should mostly be red (underperformers). Most are — but look "
            "at the one enormous green bar on the right."
        ),
        code(
            "names = [n for n,_ in R['laggards']] + [n for n,_ in R['winners']]\n"
            "vals  = [v for _,v in R['laggards']] + [v for _,v in R['winners']]\n"
            "order = np.argsort(vals)\n"
            "names = [names[i] for i in order]; vals = [vals[i] for i in order]\n"
            "cols = [RED if v < 0 else GREEN for v in vals]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.8))\n"
            "ax.barh(names, vals, color=cols)\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('flyer return vs the market, annualised (%)')\n"
            "ax.set_title('Most flyers DID lag - but Tesla (a founder-growth winner) breaks the short')\n"
            "ax.annotate('one name flips\\nthe whole basket', xy=(vals[-1], len(vals)-1),\n"
            "            xytext=(vals[-1]-28, len(vals)-2.4), color=GREEN, fontsize=9,\n"
            "            arrowprops=dict(arrowstyle='->', color=GREEN))\n"
            "plt.tight_layout(); plt.show()\n"
            "print('most flyers lag; TSLA %+.0f%%/yr, GOOGL %+.0f%%, META %+.0f%% drag the short basket UP'\n"
            "      % (R['winners'][0][1], R['winners'][1][1], R['winners'][2][1]))"
        ),
        md(
            f"There's the confound in one picture. Six of the flyers underperformed (IEP "
            f"**{R['laggards'][0][1]:.0f}%**, Comcast **{R['laggards'][1][1]:.0f}%**, Occidental "
            f"**{R['laggards'][2][1]:.0f}%**…) — Yermack would be pleased. But the flyers who *survived "
            "and thrived* are **founder-led growth machines** — Tesla **+"
            f"{R['winners'][0][1]:.0f}%/yr**, Alphabet **+{R['winners'][1][1]:.0f}%**, Meta "
            f"**+{R['winners'][2][1]:.0f}%**. Their jets are a footnote to a growth story. Shorting them "
            "for their aircraft would have cost you a fortune."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The sign is right (frugal **{R['cagr_low']:.1f}%** vs flyers "
            f"**{R['cagr_heavy']:.1f}%**; ex-mega-cap flyers lag the market **{R['ex_heavy_x']:.1f}%/yr**, "
            "almost exactly Yermack's −4%) — and survivorship works *against* the claim, so it's not "
            f"nothing. But the tradable spread is **HAC t = {R['ls_hac_t']:.2f}**, nowhere near "
            "significance. Real-as-a-sign, weak-as-a-signal.\n"
            f"- **Tradability — Mirage.** Net **+{R['net_ann']:.1f}%/yr** at *t* ≈ 0.5, and it's really a "
            "low-risk-vs-high-risk bet in disguise — one Tesla flips the short. Nothing to harvest.\n"
            "- **\"Governance discount?\" — Misattributed.** The only significant number is the reward "
            "for being **short risky stocks and long safe ones** (a low-volatility premium), not for "
            "spotting bad governance. The jet rides along with the risk profile; it doesn't drive the "
            "return."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the costs and the catch\n\n"
            "Say you tried anyway. You'd rebalance monthly, pay to trade both legs, and — because you're "
            "shorting the flyers — pay a **borrow fee** to your broker. Here's the gross spread and what "
            "survives."
        ),
        code(
            "if HAVE_REAL:\n"
            "    nc = st.net_of_costs(PANEL['ls'])\n"
            "    g, cst, bor, net = nc['gross_ann']*100, nc['rebal_cost_ann']*100, nc['borrow_ann']*100, nc['net_ann']*100\n"
            "else:\n"
            "    g, cst, bor, net = R['gross_ann'], R['cost_ann'], R['borrow_ann'], R['net_ann']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "labels = ['gross\\nspread', 'after\\ntrading costs', 'after\\nshort borrow']\n"
            "vals = [g, g-cst, g-cst-bor]; cols=[GREEN, AMBER, RED]\n"
            "ax.bar(labels, vals, color=cols)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('long/short return (%/yr)')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_title(f'A {net:+.1f}%/yr net spread - at t ~ 0.5, indistinguishable from zero')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.1f}%  -costs {cst:.1f}  -borrow {bor:.1f}  = net {net:+.1f}%/yr (t~0.5)')"
        ),
        md(
            f"Even before you argue about significance, there's only **+{R['net_ann']:.1f}%/yr** left — "
            "and it isn't real return, it's a low-volatility premium you could buy far more cheaply and "
            "reliably as an off-the-shelf factor, without betting the ranch on whether Elon Musk's next "
            "decade looks like his last. The jet angle adds single-name risk and subtracts nothing you "
            "were paid for."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🛫\n\n"
            "- **The governance-premium graveyard.** The famous [G-index] governance long/short "
            "(Gompers-Ishii-Metrick 2003) looked like huge alpha — then **decayed to zero** once the "
            "market learned it (Bebchuk-Cohen-Wang 2013). Our jet sort is the same shape: a governance "
            "story whose 'alpha' is really a risk factor.\n"
            "- **Add back the dead.** The cleanest fix is the hardest: rebuild the heavy basket with the "
            "*delisted* abusers (Tyco, WorldCom, Enron, Chesapeake) using their real price paths to "
            "zero. That's where Yermack's discount would actually live — and it's exactly what a "
            "survivor tape can't show.\n"
            "- **Neutralise the beta.** Re-run the long/short beta-hedged (or inside a low-vol-neutral "
            "universe) and see if *any* jet signal survives once the risk profile is matched.\n\n"
            "*Think the jet is a real, tradable red flag? Show the long/short clearing **t = 2** on a "
            "window where the frugal and flyer baskets have the **same market beta** — then it isn't "
            "just betting-against-beta in a costume.*"
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
            "# Corporate-Jet-Index — a quantitative long/short teardown 🔬\n"
            "### Long frugal / short jet-loving CEOs · HAC (Newey-West) *t* on the raw spread · a "
            "market-model α that is really a −0.45-beta BAB artifact · the founder-growth & survivorship "
            "confounds · costs + borrow · a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We take "
            "Yermack's (2006) *Flights of Fancy* result — jet-perk firms underperform ~4%/yr — at full "
            "strength and build the deployable long/short. The decisive move is **alpha vs beta**: the "
            "one statistic that clears *t* = 2 is a CAPM α manufactured by the long/short's **−0.45 "
            "beta** (frugal = low-beta staples, flyers = high-beta growth), i.e. the Frazzini-Pedersen "
            "betting-against-beta premium — not a governance discount. The raw, tradable spread is HAC "
            f"*t* = **{R['ls_hac_t']:.2f}**.\n\n"
            "> ⚠️ **Data + label note.** True governance-perk panels aren't free; we use a hardcoded, "
            "labelled table of ~24 large-caps (heavy/frugal is the believers' framing, subjective at the "
            "margin). **Survivorship points *against* the claim**: the archetypal abusers (Tyco, "
            "WorldCom, Enron, Chesapeake) delisted to ~zero and can't enter a yfinance tape, so a "
            "survivor basket that *still* can't short the flyers is a conservative refutation. Real data: "
            "yfinance **monthly total-return** closes. Offline core + synthetic control are "
            "deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Raw long/short (low−heavy, excess-of-mkt) **+{R['ls_ann']:.1f}%/yr** "
            f"at **HAC t = {R['ls_hac_t']:.2f}** (fails t≥2). Sign matches Yermack (ex-mega-cap flyers "
            f"**{R['ex_heavy_x']:.1f}%/yr** vs market) and survivorship biases *against* the claim ⇒ not "
            "nothing, but the tape can't certify it. |\n"
            f"| **Tradability** | `MIRAGE` | Net **+{R['net_ann']:.1f}%/yr** at t≈0.5; a −{abs(R['ls_beta']):.2f}-"
            "beta low-vol bet — one Tesla flips the short leg. |\n"
            f"| **Governance discount?** | `MISATTRIBUTED` | The lone t>2 is CAPM α **+{R['alpha_ann']:.1f}%/yr** "
            f"(t = {R['alpha_t']:.2f}) driven by β = **{R['ls_beta']:.2f}** (heavy β {R['heavy_beta']:.2f} "
            f"vs low β {R['low_beta']:.2f}) — betting-against-beta, not jets. |\n\n"
            "> 💡 In plain words: the frugal firms *did* out-compound the flyers — but that gap is (a) "
            "statistically indistinguishable from zero month-to-month, and (b) whatever is left after "
            "beta is just the reward for holding low-risk stocks and shorting high-risk ones. The jet is "
            "a passenger, not the pilot."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "For month $t$, form equal-weight basket returns $r^{\\text{low}}_t$ (frugal) and "
            "$r^{\\text{heavy}}_t$ (flyers, each name eligible only from the year after its perk is "
            "public), and the long/short in excess of the market $m_t=$ SPY:\n\n"
            "$$\\mathrm{LS}_t = (r^{\\text{low}}_t - m_t) - (r^{\\text{heavy}}_t - m_t) "
            "= r^{\\text{low}}_t - r^{\\text{heavy}}_t.$$\n\n"
            "- **H₁ (Yermack).** $\\mathbb{E}[\\mathrm{LS}] > 0$ — frugal beats flyers, risk-adjusted.\n"
            "- **H₂ (it's α, not β).** In $\\mathrm{LS}_t = \\alpha + \\beta m_t + \\varepsilon_t$, the "
            "$\\alpha$ is positive **and not just a beta tilt**.\n"
            "- **H₃ (deployable).** The spread survives a Newey-West *t*, costs, borrow, and dropping the "
            "mega-cap survivors.\n\n"
            f"We find **H₁ directionally supported but insignificant** (LS **+{R['ls_ann']:.1f}%/yr**, "
            f"HAC *t* = {R['ls_hac_t']:.2f}); **H₂ rejected** — the significant α (**+{R['alpha_ann']:.1f}%/yr**, "
            f"*t* = {R['alpha_t']:.2f}) is an artifact of $\\beta = {R['ls_beta']:.2f}$; **H₃ rejected** — "
            "the spread is a low-vol premium that one founder-growth name (Tesla) can flip. The claim is "
            "true as a *correlation with risk*, unproven as a *governance edge*."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the object is α, not the raw spread\n\n"
            "A raw long/short can be 'profitable' for a boring reason: if you are systematically **long "
            "low-beta and short high-beta**, you run a negative-beta book, and in a rising market a "
            "negative-beta portfolio earns a positive CAPM intercept *mechanically*. So the only "
            "governance-relevant quantity is the **beta-adjusted** α:\n\n"
            "$$\\hat\\alpha,\\ \\hat\\beta = \\arg\\min_{a,b}\\sum_t\\big(\\mathrm{LS}_t - a - b\\,m_t\\big)^2,"
            "\\qquad t_{\\hat\\alpha} = \\frac{\\hat\\alpha}{\\mathrm{se}_{\\text{HAC}}(\\hat\\alpha)}.$$\n\n"
            "The trap is reading a positive $\\hat\\alpha$ as 'governance works' when it is really "
            "$-\\hat\\beta\\cdot\\bar m$ leaking through — the low-volatility / betting-against-beta "
            "premium (Frazzini-Pedersen 2014; Baker-Bradley-Wurgler 2011). Our frugal basket "
            f"($\\beta = {R['low_beta']:.2f}$, defensive staples/retail) minus flyer basket "
            f"($\\beta = {R['heavy_beta']:.2f}$, growth) *is* a bet against beta, so the α it prints is "
            "the thing to be suspicious of, not to celebrate."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Perk table.** Hardcoded ~{R['n_firms']} large-caps (ticker, heavy/low, public-year); "
            f"**{R['n_heavy']}** heavy / **{R['n_low']}** low, as-of {R['asof']}, fingerprint "
            f"`{R['fingerprint']}`.\n"
            "- **Eligibility / lag.** A heavy name enters the short only from **Jan of the year after** "
            "its perk is public (no look-ahead). Low names enter whenever priced.\n"
            f"- **Panel.** Monthly total-return closes; kept where both baskets have ≥4 eligible names ⇒ "
            f"**{R['n_months']} months**, {R['start']} → {R['end']}.\n"
            "- **Signal test.** Newey-West (HAC, Bartlett) *t* on the monthly $\\mathrm{LS}$ mean.\n"
            "- **Alpha vs beta.** OLS $\\mathrm{LS}=\\alpha+\\beta m+\\varepsilon$ with a HAC *t* on "
            "$\\alpha$; report the basket betas.\n"
            "- **Confound stress tests.** Drop the mega-cap survivors (TSLA/META/GOOGL/ORCL); name the "
            "delisted abusers the tape *cannot* include.\n"
            "- **Costs.** One-way 10 bps × ~10%/mo turnover on both legs + 50 bps/yr short borrow.\n"
            "- **Positive control.** A deterministic monthly panel with a **plantable heavy-basket "
            "discount**; the engine must recover it and must NOT fabricate significance under the null."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The raw spread — right sign, no significance\n\n"
            "Long/short (low − heavy) in excess of the market, monthly, with its Newey-West *t*. The "
            "compounding gap is real; the *t* is a coin flip."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = PANEL['ls'].to_numpy(); hac = st.hac_tstat(ls)\n"
            "    ann = st.annualize_mean(hac['mean'])*100; t = hac['t']; lags = hac['lags']\n"
            "    naive = ls.mean()/(ls.std(ddof=1)/np.sqrt(len(ls)))\n"
            "else:\n"
            "    ann,t,lags,naive = R['ls_ann'], R['ls_hac_t'], R['ls_lags'], R['ls_naive_t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.bar(['naive iid t','HAC (Newey-West) t'], [naive, t], color=[GREY, AMBER], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 bar'); ax.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate([naive,t]): ax.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylim(0, 2.4); ax.set_ylabel('t-stat of the long/short mean')\n"
            "ax.set_title(f'Long/short {ann:+.1f}%/yr - both t-stats far below 2'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'LS {ann:+.2f}%/yr | naive t {naive:.2f} | HAC t {t:.2f} (lags {lags}) -> fails t>=2')"
        ),
        md(
            f"> 💡 In plain words: **+{R['ls_ann']:.1f}%/yr** at HAC *t* = **{R['ls_hac_t']:.2f}** (the "
            f"naive *t* is barely higher at {R['ls_naive_t']:.2f}, so autocorrelation isn't hiding an "
            "edge). With ~24 names and monthly scatter, a couple-of-point spread is deep inside its "
            "standard error. H₁ is directionally there and statistically absent."
        ),
        md(
            "### 4b · Alpha vs beta — the significant number is a beta artifact\n\n"
            "Regress the long/short on the market. The intercept α clears *t* = 2 — but look at the "
            "slope: a **negative beta**. The α is what a short-beta book earns in a bull market, not a "
            "governance premium."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mkt = st.monthly_returns(PRICES)['SPY'].reindex(PANEL.index).to_numpy()\n"
            "    mm = st.market_model_alpha(PANEL['ls'].to_numpy(), mkt)\n"
            "    a_ann = st.annualize_mean(mm['alpha'])*100; ta = mm['t_alpha']; beta = mm['beta']\n"
            "    x = mkt*100; y = PANEL['ls'].to_numpy()*100\n"
            "else:\n"
            "    a_ann,ta,beta = R['alpha_ann'], R['alpha_t'], R['ls_beta']\n"
            "    rng=np.random.default_rng(1); x=rng.normal(0.9,4.3,R['n_months']); y=beta*x+R['alpha_ann']/12+rng.normal(0,3,R['n_months'])\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.6))\n"
            "ax.scatter(x, y, s=16, color=GREY, alpha=.6)\n"
            "xs = np.linspace(x.min(), x.max(), 50)\n"
            "ax.plot(xs, beta*xs + a_ann/12, c=RED, lw=2, label=f'fit: beta={beta:+.2f}, alpha={a_ann:+.1f}%/yr')\n"
            "ax.axhline(0, c='k', lw=.6); ax.axvline(0, c='k', lw=.6)\n"
            "ax.set_xlabel('market return (%/mo)'); ax.set_ylabel('long/short return (%/mo)')\n"
            "ax.set_title(f'Long/short beta = {beta:+.2f}: the alpha (t={ta:.2f}) is a short-beta artifact'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'alpha {a_ann:+.1f}%/yr (HAC t={ta:.2f}) | LS beta {beta:+.2f} -> negative beta manufactures the alpha')"
        ),
        md(
            f"> 💡 In plain words: the down-sloping fit is the whole story. The long/short has "
            f"$\\beta = {R['ls_beta']:.2f}$ because the frugal basket ($\\beta = {R['low_beta']:.2f}$) is "
            f"defensive and the flyer basket ($\\beta = {R['heavy_beta']:.2f}$) is aggressive. Over a "
            "market that rose ~14%/yr, a −0.45-beta book *must* print a positive intercept. That α is "
            "**betting-against-beta**, a premium the desk has priced a hundred times — it has nothing to "
            "do with corporate jets."
        ),
        md(
            "### 4c · The founder-growth & survivorship confounds\n\n"
            "Two forces pull in opposite directions. **Survivorship** removes the delisted abusers (which "
            "would *help* the short) — yet the surviving flyers are **founder-growth winners** (which "
            "*hurt* it). Drop the four mega-cap survivors and the heavy basket finally underperforms the "
            "market by ~4% — Yermack's number — but the spread is *still* insignificant and *still* a "
            "beta artifact."
        ),
        code(
            "if HAVE_REAL:\n"
            "    firms2 = [f for f in FIRMS if f['ticker'] not in ('TSLA','META','GOOGL','ORCL')]\n"
            "    s2 = st.summarize(st.long_short_panel(PRICES, firms2, min_names=3))\n"
            "    full = (S['heavy_x_ann']*100, S['ls_mean_ann']*100, S['ls_hac_t'])\n"
            "    exmt = (s2['heavy_x_ann']*100, s2['ls_mean_ann']*100, s2['ls_hac_t'])\n"
            "else:\n"
            "    full = (R['heavy_x_ann'], R['ls_ann'], R['ls_hac_t'])\n"
            "    exmt = (R['ex_heavy_x'], R['ex_ls_ann'], R['ex_hac_t'])\n"
            "fig, (a1,a2) = plt.subplots(1,2, figsize=(11.0,4.3))\n"
            "a1.bar(['all flyers','drop mega-tech'], [full[0], exmt[0]], color=[GREY, RED])\n"
            "a1.axhline(0, c='k', lw=.8); a1.axhline(-4, ls='--', c=AMBER, label=\"Yermack -4%/yr\")\n"
            "a1.set_ylabel('heavy basket excess-of-market (%/yr)'); a1.set_title('Drop the survivors -> the -4% appears'); a1.legend()\n"
            "for i,v in enumerate([full[0],exmt[0]]): a1.annotate(f'{v:+.1f}%',(i,v),ha='center',va='top' if v<0 else 'bottom')\n"
            "a2.bar(['all flyers','drop mega-tech'], [full[2], exmt[2]], color=[GREY, RED])\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('long/short HAC t'); a2.set_title('...but the spread is STILL insignificant')\n"
            "for i,v in enumerate([full[2],exmt[2]]): a2.annotate(f't={v:.2f}',(i,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'all flyers: heavy {full[0]:+.1f}%/yr, HAC t {full[2]:.2f} | ex-mega-tech: heavy {exmt[0]:+.1f}%/yr, HAC t {exmt[2]:.2f}')"
        ),
        md(
            f"> 💡 In plain words: strip out Tesla/Meta/Alphabet/Oracle and the flyers lag the market by "
            f"**{R['ex_heavy_x']:.1f}%/yr** — spookily close to the paper. But (a) that's a snooped "
            f"subset, and (b) the spread is *still* only HAC *t* = **{R['ex_hac_t']:.2f}** with the same "
            f"low-beta α (**+{R['ex_alpha_ann']:.1f}%/yr**, *t* = {R['ex_alpha_t']:.2f}). The delisted "
            "abusers the tape can't hold would push the *other* way — so even this survivor-biased, "
            "hand-trimmed version can't manufacture a real governance short."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "A deterministic monthly panel (12 heavy / 12 low, **equal betas** so the null spread is "
            "centered on zero) with a **plantable heavy-basket discount**. With **no** discount the HAC "
            "*t* must stay inside ±2; with a large planted discount it must light up positive."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, -80.0):\n"
            "    syn = data.synthetic_panel(alpha_bps_month=edge, seed=745)\n"
            "    h = st.hac_tstat(syn['ls'])\n"
            "    res.append((edge, st.annualize_mean(h['mean'])*100, h['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = ['planted 0 bps/mo\\n(null)', 'planted -80 bps/mo\\n(~-9%/yr, large)']\n"
            "ts = [r[2] for r in res]\n"
            "ax.bar(labels, ts, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2'); ax.axhline(-2, ls='--', c=RED); ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(ts): ax.annotate(f't={t:+.2f}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "ax.set_ylabel('long/short HAC t'); ax.set_title('Control: quiet under the null, lights up on a real discount'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,a,t in res: print(f'planted {e:+.0f} bps/mo: LS {a:+.2f}%/yr  HAC t = {t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted discount the control's HAC *t* is "
            f"**{R['syn'][0][2]:.2f}** (no false positive from 186 months); a large −80 bps/mo discount "
            f"drives it to **{R['syn'][1][2]:.2f}**. So the engine is honest and powered — and the real "
            f"tape's HAC *t* of **{R['ls_hac_t']:.2f}** is exactly what an *absent, survivor-suppressed, "
            "beta-confounded* effect looks like. The machinery isn't the problem; the claim is."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — raw long/short **+{R['ls_ann']:.1f}%/yr** at **HAC t = {R['ls_hac_t']:.2f}** "
            "(naive t {0:.2f}), fails t ≥ 2. The sign matches Yermack (ex-mega-cap flyers "
            "{1:+.1f}%/yr vs market ≈ his −4%) and **survivorship biases *against*** the claim, so it "
            "isn't a hard zero — but literature + right sign on a sub-2 tape ⇒ WEAK, not REAL.\n"
            "- **Tradability `MIRAGE`** — net **+{2:.1f}%/yr** at t≈0.5; it is a −{3:.2f}-beta low-vol bet, "
            "and one founder-growth survivor (Tesla) flips the short leg. Not NAV-scale, not sign-stable.\n"
            "- **Governance discount? `MISATTRIBUTED`** — the only t>2 is CAPM α **+{4:.1f}%/yr** "
            "(t = {5:.2f}) driven by β = **{6:.2f}** (heavy β {7:.2f} vs low β {8:.2f}) — the "
            "betting-against-beta premium, not a jet red flag."
            .format(R['ls_naive_t'], R['ex_heavy_x'], R['net_ann'], abs(R['ls_beta']),
                    R['alpha_ann'], R['alpha_t'], R['ls_beta'], R['heavy_beta'], R['low_beta'])
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — costs, borrow, and the single-name risk\n\n"
            "The operational truth: a low-single-digit gross spread, most of it a factor you can rent "
            "cheaply, all of it hostage to a handful of founder-growth names on the short leg."
        ),
        code(
            "if HAVE_REAL:\n"
            "    nc = st.net_of_costs(PANEL['ls'])\n"
            "    g,cst,bor,net = nc['gross_ann']*100, nc['rebal_cost_ann']*100, nc['borrow_ann']*100, nc['net_ann']*100\n"
            "else:\n"
            "    g,cst,bor,net = R['gross_ann'], R['cost_ann'], R['borrow_ann'], R['net_ann']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "steps = ['gross', '- costs', '- borrow', '= net']\n"
            "vals = [g, -cst, -bor, None]; running=g; bottoms=[0]; heights=[g]\n"
            "for v in vals[1:3]:\n"
            "    bottoms.append(running+v if v<0 else running); heights.append(abs(v)); running+=v\n"
            "ax.bar(0, g, color=GREEN); ax.bar(1, cst, bottom=g-cst, color=AMBER); ax.bar(2, bor, bottom=g-cst-bor, color=AMBER)\n"
            "ax.bar(3, net, color=RED)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(range(4)); ax.set_xticklabels(steps)\n"
            "ax.set_ylabel('long/short (%/yr)'); ax.set_title(f'Gross {g:+.1f}%  ->  net {net:+.1f}%/yr (t~0.5)')\n"
            "ax.annotate(f'{net:+.1f}%', (3,net), ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.1f}%/yr | costs -{cst:.1f} | borrow -{bor:.1f} | net {net:+.1f}%/yr — a low-vol premium, not an edge')"
        ),
        md(
            "> 💡 In plain words: after frictions you're left with a **low-vol factor** you could buy as "
            "an ETF, wrapped in idiosyncratic single-name risk (short Tesla for a decade — no thanks). "
            "There is no jet-specific residual to capture. **The most vivid part of the story — famous "
            "flying founders — is exactly what makes the short both dangerous and empty.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The governance-alpha graveyard.** Gompers-Ishii-Metrick (2003) governance long/short "
            "looked like huge alpha; Bebchuk-Cohen-Wang (2013) showed it **decayed to zero out of "
            "sample**. The jet sort is the same species — a governance narrative whose only significant "
            "statistic is a risk factor.\n"
            "- **Resurrect the delisted.** Rebuild the heavy basket with the *bankrupt* abusers (Tyco, "
            "WorldCom, Enron, Chesapeake) on their real paths to zero — the honest way to see whether "
            "Yermack's discount lives where the survivor tape can't look.\n"
            "- **Beta-neutralise.** Re-run the long/short beta-hedged or in a low-vol-matched universe; "
            "if *any* jet signal clears *t* = 2 once the risk profile is equalised, that would be news.\n"
            "- **Bigger panel, real disclosure dates.** Swap the hardcoded table for a full "
            "perquisite-disclosure panel (hundreds of firms, exact DEF 14A dates); the error bars "
            "tighten, but the beta and founder-growth confounds won't vanish on their own.\n\n"
            "*The reproducible core is offline and deterministic; the perk table is an explicit "
            "hardcoded, labelled stand-in, survivor-biased *against* the claim. Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: [`docs/results.md`](../docs/results.md).*"
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
