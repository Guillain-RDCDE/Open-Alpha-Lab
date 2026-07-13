"""Generate the two narrative notebooks for Study 769 ("parks attendance as a DIS tell").

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic: real-tape cells read the cached month-end
yfinance pulls under ../_cache/ for DIS + SPY and the hardcoded (cited, approximate) parks
series from the package; on a cache miss they fall back to the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic planted-edge control runs anywhere.
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


# Frozen headline numbers — mirror of docs/results.md (parks series hardcoded/cited/approx;
# DIS + SPY month-end Adj Close via yfinance, as-of 2024-12-31). Single source of truth.
R = dict(
    win_eq="2010-01 → 2024-12",
    att_levels={2015: 138, 2016: 140, 2017: 150, 2018: 157, 2019: 156,
                2020: 47, 2021: 85, 2022: 122, 2023: 142},
    att_growth={2019: -1, 2020: -70, 2021: 81, 2022: 44, 2023: 16},
    release_month=7,
    dis_cagr=10.41, dis_vol=26.6, dis_sharpe=0.50, dis_mdd=-57.1,
    spy_cagr=14.14, spy_vol=14.5, spy_sharpe=0.99, spy_mdd=-23.9,
    excess_mean=-2.68, excess_t=-0.533, excess_p=0.603, excess_n=14,
    ll_abs_slope=0.00188, ll_abs_t=1.19, ll_n=137,
    ll_exc_slope=0.00002, ll_exc_t=0.02,
    ll_price_exc_t=0.07,
    reg_abs_cond=16.04, reg_abs_base=9.87, reg_abs_t=2.04, reg_abs_n=113,
    reg_exc_cond=-0.43, reg_exc_base=-4.77, reg_exc_t=1.80,
    bt_switches=3, bt_expo=70,
    bt_gross_sharpe=0.580, bt_net_sharpe=0.578, bt_net_ann=13.30, bt_net_vol=23.0,
    bt_bh_dis_sharpe=0.503, bt_bh_spy_sharpe=0.987,
    ctrl_null_t=0.60, ctrl_edge_t=3.02,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Parks_momentum_leads_DIS%3F: Busted](https://img.shields.io/badge/Parks_momentum_leads_DIS%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY, BLUE = "#c0392b", "#dab617", "#2ea44f", "#8b949e", "#3b6fb0"

from disney_parks import data, strategy as st

HAVE_EQ = data.have_equities()
ATT = data.load_attendance()                         # hardcoded, cited, APPROXIMATE proxy
GROWTH = data.attendance_growth()
FRAME = data.build_frame() if HAVE_EQ else None
print("equity cache present:", HAVE_EQ,
      "| attendance years:", ATT.index[0].year, "->", ATT.index[-1].year)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the Disney-parks crowd tell you when to buy DIS? 🏰\n"
            "### \"The parks are packed and prices keep rising — buy the stock,\" in plain English\n\n"
            + BADGES +
            "It's one of the most intuitive-sounding tips in retail investing: walk through a "
            "*rammed* Magic Kingdom in July, watch Disney jack the ticket price to \\$199, and "
            "conclude the obvious — *Disney the **stock** must be a screaming buy.* The parks are "
            "Disney's biggest profit engine, after all. If you can *see* the demand, you're ahead of "
            "Wall Street, right?\n\n"
            "This notebook takes that tip seriously and lines the **real, cited theme-park attendance "
            "numbers** up against `DIS` — with one honest rule the tip always forgets: **you can only "
            "trade on the attendance figure *after* it's published.** And it turns out that's the whole "
            "ballgame.\n\n"
            "> 📓 **Plain-language layer.** Want the Newey-West lead-lag *t*, the regime-dummy confound "
            "and the cost algebra? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice — and a data note.** The industry attendance report "
            "(TEA/AECOM *Theme Index*) is a paywalled PDF, not a free feed, so the attendance line "
            "below is a **small, clearly-cited, approximate** reconstruction of its public headline "
            "figures — a **proxy**, released with its real ~6-month lag. Every chart is drawn by the "
            "code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Are the parks a real, huge business? | **Yes.** Attendance climbed all through the "
            f"2010s to ~156 M visits in 2019, cratered to ~47 M in COVID-2020, and recovered. The "
            "demand story is completely real. |\n"
            "| Did owning DIS pay off like the crowds suggest? | **No.** Over 2010–2024 DIS compounded "
            f"at **~{R['dis_cagr']:.0f}%/yr** vs **~{R['spy_cagr']:.0f}%/yr** for the S&P 500 — with "
            f"*twice* the volatility and a **{R['dis_mdd']:.0f}%** drawdown. |\n"
            "| Does packed-parks momentum *lead* the stock? | **No.** Once you can only act on the "
            f"attendance print *after* it's public, it predicts DIS's market-beating return with "
            f"*t* = **{R['ll_exc_t']:.2f}** — a statistical zero. |\n"
            "| Why? | **You're late.** The attendance report lands ~6 months into the *next* year — "
            "long after Disney's quarterly results already told Wall Street how the parks did. |\n\n"
            "> The crowd is real. The *edge* is not. By the time you can trade the number, it's old news."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Theme-park attendance and Disney's pricing power are a leading indicator for the "
            "stock. The parks are Disney's crown jewel — packed gates and ever-higher ticket prices "
            "mean the business is firing, so `DIS` is a buy. You can literally see the alpha in the "
            "queue for Space Mountain.\"*\n\n"
            "It's a *steelman-able* claim. Parks, Experiences & Products **is** Disney's single "
            "largest profit centre, attendance really did grow for a decade, and ticket prices really "
            "did march from \\$79 (2010) to \\$199 (2024). The *business* intuition is sound. The "
            "question is whether it's a **tradable** intuition — whether the crowd tells *you* "
            "anything the market doesn't already know."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it were true, it would be a rare gift: a signal you can *observe with your own eyes* "
            "that front-runs a mega-cap stock. Alt-data desks pay millions for satellite car-counts of "
            "parking lots for exactly this reason. But 'the parks are doing great' and 'I can beat the "
            "market owning DIS' are different claims. The first is about the **business**; the second "
            "needs the information to be **early, DIS-specific, and net of cost**. We can check the "
            "second directly — and the release calendar is where it lives or dies."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three honest checks, each respecting the one rule the tip forgets — *no peeking at a "
            "number before it's public*:\n\n"
            "1. **Did DIS even beat the market?** Put `DIS` next to the S&P 500 (`SPY`) on the same "
            "2010–2024 clock — return, volatility, worst drawdown.\n"
            "2. **Does the crowd lead the stock?** Take the attendance figure *as you'd actually learn "
            "it* — released ~6 months into the next year — and ask if it predicts DIS's **market-"
            "beating** return over the following year.\n"
            "3. **Could you trade it?** Build the obvious rule — hold DIS when parks momentum is "
            "positive, else sit in the S&P — charge real costs, and see if it beats just owning an "
            "index fund.\n\n"
            "**What would make us say \"real tell\"?** The lagged attendance signal predicts DIS's "
            "*excess* return with a *t* past 2, **and** the rule beats buy-and-hold SPY net of cost. "
            "Anything less is a story about crowds, not an edge."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the crowds are real.** Here's the (approximate, cited) attendance line — a "
            "decade of growth, the COVID crater, the bounce."
        ),
        code(
            "yrs = list(ATT.index.year)\n"
            "lv = [float(v) for v in ATT.values]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.plot(yrs, lv, 'o-', c=BLUE, lw=2.4, label='Disney parks attendance (proxy, M visits)')\n"
            "ax.axvline(2020, ls='--', c=RED, alpha=.6)\n"
            "ax.annotate('COVID crater\\n~47 M', (2020, 47), textcoords='offset points',\n"
            "            xytext=(8, 18), color=RED, fontsize=9)\n"
            "ax.set_xlabel('year'); ax.set_ylabel('attendance (millions of visits)')\n"
            "ax.set_title('The parks demand story is completely real'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('attendance (M):', {y:int(round(v)) for y,v in zip(yrs, lv)})"
        ),
        md(
            "Steady growth into a **156 M-visit** peak in 2019, a COVID collapse to ~47 M, then a "
            "recovery. Nobody disputes the parks are a juggernaut. The tip's premise is sound. The "
            "question is what it does for a *stock investor*."
        ),
        md(
            "**Now the uncomfortable part: did owning DIS actually pay?** Same money, same years — "
            "DIS vs an S&P index fund."
        ),
        code(
            "if HAVE_EQ:\n"
            "    dis = FRAME['dis']; spy = FRAME['spy']\n"
            "    di = st.summarize(dis); ss = st.summarize(spy)\n"
            "    dcagr, scagr, dmdd, smdd = di['cagr']*100, ss['cagr']*100, di['mdd']*100, ss['mdd']*100\n"
            "    dn = dis/dis.iloc[0]*100; sn = spy/spy.iloc[0]*100\n"
            "    xd = dis.index; \n"
            "else:\n"
            "    dcagr, scagr, dmdd, smdd = R['dis_cagr'], R['spy_cagr'], R['dis_mdd'], R['spy_mdd']\n"
            "    xd = pd.date_range('2010-01-31', periods=180, freq='ME')\n"
            "    dn = pd.Series(100*(1+dcagr/100)**(np.arange(180)/12), index=xd)\n"
            "    sn = pd.Series(100*(1+scagr/100)**(np.arange(180)/12), index=xd)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.plot(xd, dn, c=BLUE, lw=2.0, label=f'DIS   ({dcagr:.0f}%/yr, maxDD {dmdd:.0f}%)')\n"
            "ax.plot(xd, sn, c=GREEN, lw=2.0, label=f'S&P 500  ({scagr:.0f}%/yr, maxDD {smdd:.0f}%)')\n"
            "ax.set_ylabel('$100 invested at start-2010'); ax.set_title('DIS lagged the market it is supposed to lead'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'DIS ~{dcagr:.1f}%/yr (maxDD {dmdd:.0f}%)  vs  SPY ~{scagr:.1f}%/yr (maxDD {smdd:.0f}%)')"
        ),
        md(
            f"The S&P didn't just win — it won with **half the drama**: ~{R['spy_cagr']:.0f}%/yr and a "
            f"−24% worst drop, versus DIS's ~{R['dis_cagr']:.0f}%/yr and a gut-churning "
            f"**{R['dis_mdd']:.0f}%** drawdown. All those packed parks, and the stock still trailed a "
            "boring index fund. Right there, the tip is in trouble."
        ),
        md(
            "**But maybe the *timing* is the edge?** The tip says: get in when the crowds are growing. "
            "Let's test it fairly — the attendance number for a year only becomes public ~6 months "
            "into the *next* year, so that's the earliest you could act. Does that signal predict DIS "
            "beating the market over the next 12 months?"
        ),
        code(
            "if HAVE_EQ:\n"
            "    lle = st.lead_lag(FRAME, 'pg', horizon=12, lag=1, excess=True)\n"
            "    lla = st.lead_lag(FRAME, 'pg', horizon=12, lag=1, excess=False)\n"
            "    t_exc, t_abs = lle['t'], lla['t']\n"
            "else:\n"
            "    t_exc, t_abs = R['ll_exc_t'], R['ll_abs_t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.2))\n"
            "bars = ax.bar(['predicting DIS\\n(raw return)', 'predicting DIS\\nBEATING the market'],\n"
            "              [t_abs, t_exc], color=[GREY, RED], width=.55)\n"
            "ax.axhline(2, ls='--', c=GREEN, alpha=.7); ax.axhline(-2, ls='--', c=GREEN, alpha=.7)\n"
            "ax.annotate('the bar for a real signal (|t|=2)', (0.5, 2.05), color=GREEN, fontsize=9)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('predictive t-stat'); ax.set_ylim(-1, 3)\n"
            "ax.set_title('Does lagged parks momentum lead DIS? Not once you can only trade it late')\n"
            "for b,v in zip(bars,[t_abs,t_exc]): ax.annotate(f't={v:.2f}',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'predicting DIS beating the S&P: t = {t_exc:.2f}  (a statistical zero)')"
        ),
        md(
            f"There it is. For the return that actually matters — DIS **beating** the market — the "
            f"lagged crowd signal scores *t* = **{R['ll_exc_t']:.2f}**. That's not a weak signal, it's "
            "*no* signal. Why? By the time the annual attendance report is published, Disney has "
            "already told Wall Street how the parks did in **three quarterly earnings calls**. You're "
            "not early. You're half a year late, reading yesterday's news."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The lagged parks signal predicts DIS's market-beating return with "
            f"*t* = {R['ll_exc_t']:.2f}; even the pricing-power version is a zero. No edge.\n"
            "- **Tradability — Mirage.** The obvious timing rule (next beat) trails a plain S&P index "
            "fund after costs — you'd have been richer doing nothing.\n"
            "- **Parks momentum leads DIS? — Busted.** A once-a-year, six-months-late attendance print "
            "can't front-run a stock whose quarterly reports already carry the same news."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — the honest bottom line\n\n"
            "Imagine two people at the start of 2010, each with \\$10,000. One buys an S&P index fund "
            "and forgets about it. The other runs the parks-tip strategy — hold DIS when the (lagged) "
            "crowd is growing, rotate to the S&P otherwise, paying real costs to switch. Who's richer "
            "by end-2024?"
        ),
        code(
            "start = 10_000.0; n_yr = 15\n"
            "spy_end = start*(1+R['spy_cagr']/100)**n_yr\n"
            "rule_end = start*(1+R['bt_net_ann']/100)**n_yr\n"
            "dis_end = start*(1+R['dis_cagr']/100)**n_yr\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.bar(['S&P index fund','the parks-tip rule\\n(net of cost)','buy & hold DIS'],\n"
            "       [spy_end, rule_end, dis_end], color=[GREEN, RED, BLUE], width=.6)\n"
            "for i,v in enumerate([spy_end, rule_end, dis_end]): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('value of $10,000 after 15 years')\n"
            "ax.set_title('Same $10k, 2010 -> 2024: the clever rule still trails the index fund'); plt.tight_layout(); plt.show()\n"
            "print(f'S&P index fund ${spy_end:,.0f}  |  parks-tip rule ${rule_end:,.0f}  |  buy-hold DIS ${dis_end:,.0f}')"
        ),
        md(
            "The index-fund investor wins, doing nothing. The parks-tip rule edges out buy-and-hold "
            "*DIS* — but only because it spends most of the decade hiding in the S&P anyway, and its "
            "few 'clever' moves were three stale, once-a-year rotations. You paid attention, tracked "
            "attendance, and traded — to end up **behind** the person who bought an index fund and went "
            "to the parks to have fun."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Use the real feed, earlier.** Our attendance line is a cited *approximation* released "
            "on the Theme Index's real calendar. If you can get a *faster* read (Disney's own quarterly "
            "segment revenue, or paid satellite/mobility data), test whether *that* leads — the whole "
            "point is being early.\n"
            "- **The alt-data trap.** Any public, journalism-friendly signal is the *least* edgy kind — "
            "it's priced by the time you read it (see [docs/references.md](../docs/references.md)).\n"
            "- **Same shape, other tickers.** [Study 358 — Watch Index](../../358-watch-index/) and "
            "[Study 708 — Eurovision](../../708-eurovision-effect/) run the same labelled-proxy, "
            "no-look-ahead discipline on other 'obvious' tells.\n\n"
            "*Think a faster or cleaner park signal beats the market net of cost? Wire it in, keep the "
            "release lag honest, and show the *t* — then check it wasn't just riding the post-COVID "
            "bounce.*"
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
            "# Parks attendance as a DIS tell — a quantitative teardown 🔬\n"
            "### DIS vs SPY (CAGR / vol / MDD + annual-excess *t*) · a Newey-West lead-lag on DIS's "
            "*excess* return · the regime-dummy confound · a costed rotation backtest · a synthetic "
            "planted-edge control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We test "
            "the strongest **strictly-lagged, no-look-ahead** form of \"theme-park momentum leads "
            "DIS\": (H₁) DIS out-returns SPY; (H₂) release-lagged attendance growth predicts DIS's "
            "*excess* return; (H₃) a rule built on it beats buy-and-hold SPY net of cost. We find "
            "**H₁ rejected** (DIS *lags*), **H₂ rejected** (*t* ≈ 0 on the excess tape), **H₃ "
            "rejected** (the rule trails SPY).\n\n"
            "> ⚠️ **Not investment advice — data provenance.** The attendance & ticket-price series "
            "are **hardcoded, cited, approximate** (public TEA/AECOM Theme Index — a *labelled proxy*, "
            "never a live feed), released with the report's real ~mid-following-year lag. `DIS`, `SPY` "
            "are month-end Adj Close via yfinance (as-of 2024-12-31). Offline core + synthetic control "
            "are deterministic. Methods in [`docs/references.md`](../docs/references.md); numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Lead-lag slope of DIS **excess-of-SPY** fwd-12m return on "
            f"release-lagged parks growth: **{R['ll_exc_slope']:+.5f}, NW *t* = {R['ll_exc_t']:+.2f}** "
            f"(n={R['ll_n']}); price-hike tell *t* = {R['ll_price_exc_t']:+.2f}. |\n"
            f"| **Tradability** | `MIRAGE` | Costed rotation rule nets **Sharpe {R['bt_net_sharpe']:.3f}** "
            f"vs buy-hold SPY **{R['bt_bh_spy_sharpe']:.3f}**. Three stale switches; un-scalable timing. |\n"
            f"| **Parks momentum leads DIS?** | `BUSTED` | Theme Index prints year-Y attendance "
            f"~{R['release_month']:02d}/Y+1 — after Disney's quarterly segment reports. Stale by construction. |\n\n"
            "> 💡 In plain words: DIS *lagged* the market over the window, the lagged crowd signal has "
            "essentially zero correlation with DIS beating that market, and the obvious timing rule "
            "trails an index fund. Every axis on which \"parks momentum leads DIS\" could survive, fails."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let the (release-lagged) parks-attendance growth be $g_t$, DIS's forward $h$-month return "
            "$r^{D}_{t\\to t+h}$ and SPY's $r^{B}_{t\\to t+h}$. The claim is a joint hypothesis:\n\n"
            "- **H₁ (DIS is worth owning).** $r^{D}$ out-returns $r^{B}$ risk-adjusted over the window.\n"
            "- **H₂ (the crowd leads the stock).** In "
            "$\\;(r^{D}_{t\\to t+h} - r^{B}_{t\\to t+h}) = \\alpha + \\beta\\, g_{t-1} + \\varepsilon\\;$ "
            "the slope $\\beta$ is positive with a Newey-West *t* > 2 — parks momentum predicts DIS "
            "**beating the market** (a DIS-specific tell, not broad beta).\n"
            "- **H₃ (it's harvestable).** A rule long DIS when $g>0$ else SPY, net of switching cost, "
            "out-Sharpes buy-and-hold SPY.\n\n"
            "The steelman is that Parks & Experiences is Disney's largest segment and attendance is "
            "real, cited data. The test is whether the *lagged, public* signal carries **DIS-specific, "
            "net-of-cost** predictive content — or is already in the price."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "If H₁–H₃ held, an eyeball-observable series would front-run a mega-cap — the dream of "
            "every alt-data desk. But each leg is separately falsifiable. H₁ is a **return race** on a "
            "common clock. H₂ is the crux: it must predict DIS's return *in excess of SPY*, because a "
            "signal that only tracks the broad-market regime is **not** a DIS tell — it's beta you can "
            "buy cheaper. H₃ is the **microstructure + capacity** reality: a once-a-year, six-months-"
            "late print gives you a handful of trades, not a strategy. Fail any leg and \"parks "
            "momentum leads DIS\" collapses to a business truism."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Parks series (proxy).** A hardcoded, cited, **approximate** annual attendance level "
            "(total Walt Disney Attractions, M visits) + WDW ticket price, reconstructed from public "
            "TEA/AECOM Theme Index figures. *Labelled a proxy* — its shape (2010s growth, 2020 crater, "
            "recovery) is defensible; its precise values are not a live feed.\n"
            f"- **Release lag (the crux).** The Theme Index for year Y is public ~{R['release_month']:02d}/"
            "Y+1; we step the signal onto the monthly frame at that date, then add **one** execution "
            "lag. Strict no-look-ahead.\n"
            "- **Equities.** `DIS`, `SPY` month-end Adj Close (yfinance, cached). Survivorship is not a "
            "concern (two named tickers, not a screen).\n"
            "- **Signal test.** (i) A small-sample annual-excess $t$ of DIS vs SPY. (ii) A **Newey-West "
            "(12-lag) HAC** $t$ of the lead-lag slope on DIS's **excess** return — the bar for `REAL` "
            "is *t* ≥ 2. (iii) A Welch regime split, absolute *and* excess, to expose the COVID-dummy "
            "confound.\n"
            "- **Cost (beat 6).** Rotation rule, 1-month lag, 10 bps per switch leg; raced vs "
            "buy-and-hold DIS *and* SPY on Sharpe.\n"
            "- **Positive control.** A deterministic momentum-plus-planted-edge path; the engine must "
            "stay null at edge=0 and light up at edge>0 — proof a null on the real tape is a real null.\n"
            "- **What would make us say \"tell\":** H₂ *t* > 2 on the **excess** tape **and** the rule "
            "beats SPY net of cost. We find neither."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The race — DIS vs SPY\n\n"
            "Does the stock even reward you for the demand you can see? Month-end Adj Close, rebased."
        ),
        code(
            "if HAVE_EQ:\n"
            "    dis, spy = FRAME['dis'], FRAME['spy']\n"
            "    di, ss = st.summarize(dis), st.summarize(spy)\n"
            "    ae = st.annual_excess_t(dis, spy)\n"
            "    dn, sn = dis/dis.iloc[0]*100, spy/spy.iloc[0]*100; xd = dis.index\n"
            "else:\n"
            "    di = {'cagr':R['dis_cagr']/100,'vol':R['dis_vol']/100,'sharpe':R['dis_sharpe'],'mdd':R['dis_mdd']/100}\n"
            "    ss = {'cagr':R['spy_cagr']/100,'vol':R['spy_vol']/100,'sharpe':R['spy_sharpe'],'mdd':R['spy_mdd']/100}\n"
            "    ae = {'mean_excess':R['excess_mean']/100,'t':R['excess_t'],'n':R['excess_n']}\n"
            "    xd = pd.date_range('2010-01-31', periods=180, freq='ME')\n"
            "    dn = pd.Series(100*(1+R['dis_cagr']/100)**(np.arange(180)/12), index=xd)\n"
            "    sn = pd.Series(100*(1+R['spy_cagr']/100)**(np.arange(180)/12), index=xd)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.plot(xd, dn, c=BLUE, lw=2.0, label=f\"DIS  CAGR {di['cagr']*100:.1f}%, Sharpe {di['sharpe']:.2f}, maxDD {di['mdd']*100:.0f}%\")\n"
            "ax.plot(xd, sn, c=GREEN, lw=2.0, label=f\"SPY  CAGR {ss['cagr']*100:.1f}%, Sharpe {ss['sharpe']:.2f}, maxDD {ss['mdd']*100:.0f}%\")\n"
            "ax.set_ylabel('rebased to 100 @ 2010'); ax.set_title('H1: DIS UNDER-performed SPY over the cycle'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"DIS: CAGR {di['cagr']*100:.2f}%  vol {di['vol']*100:.1f}%  Sharpe {di['sharpe']:.2f}  maxDD {di['mdd']*100:.1f}%\")\n"
            "print(f\"SPY: CAGR {ss['cagr']*100:.2f}%  vol {ss['vol']*100:.1f}%  Sharpe {ss['sharpe']:.2f}  maxDD {ss['mdd']*100:.1f}%\")\n"
            "print(f\"annual excess (DIS-SPY): {ae['mean_excess']*100:+.2f}%/yr  t={ae['t']:+.3f}  (n={ae['n']})\")"
        ),
        md(
            f"> 💡 In plain words: DIS compounded at **{R['dis_cagr']:.1f}%** vs SPY's "
            f"**{R['spy_cagr']:.1f}%**, at ~twice the vol and with a **{R['dis_mdd']:.0f}%** drawdown. "
            f"The mean annual excess is **{R['excess_mean']:+.1f}%/yr, *t* = {R['excess_t']:+.2f}** "
            f"(n={R['excess_n']}) — a losing point estimate, indistinct from zero. H₁ rejected: the "
            "stock did not reward the visible demand."
        ),
        md(
            "### 4b · The lead-lag — does the crowd predict DIS *beating the market*?\n\n"
            "Newey-West (12-lag) regression of DIS's forward-12m return on the release-lagged "
            "attendance growth. We run it two ways: on DIS's **raw** return, and on its **excess of "
            "SPY** — the return a genuine DIS tell must move. `REAL` needs $t \\ge 2$ on the excess."
        ),
        code(
            "if HAVE_EQ:\n"
            "    lla = st.lead_lag(FRAME, 'pg', 12, 1, excess=False)\n"
            "    lle = st.lead_lag(FRAME, 'pg', 12, 1, excess=True)\n"
            "    llp = st.lead_lag(FRAME, 'ph', 12, 1, excess=True)\n"
            "    ts = [lla['t'], lle['t'], llp['t']]\n"
            "else:\n"
            "    ts = [R['ll_abs_t'], R['ll_exc_t'], R['ll_price_exc_t']]\n"
            "labels = ['DIS raw\\n~ attendance', 'DIS EXCESS of SPY\\n~ attendance', 'DIS excess\\n~ price hike']\n"
            "cols = [GREY, RED, GREY]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "b = ax.bar(labels, ts, color=cols, width=.55)\n"
            "ax.axhline(2, ls='--', c=GREEN, alpha=.7, label='|t| = 2 (REAL bar)'); ax.axhline(-2, ls='--', c=GREEN, alpha=.7)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('Newey-West t of the slope'); ax.set_ylim(-1, 3)\n"
            "for bb,v in zip(b,ts): ax.annotate(f't={v:.2f}',(bb.get_x()+bb.get_width()/2,v),ha='center',va='bottom')\n"
            "ax.set_title('H2: lagged parks momentum does NOT predict DIS beating the market'); ax.legend(loc='upper right')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'DIS raw ~ attendance   : t={ts[0]:+.2f}')\n"
            "print(f'DIS EXCESS ~ attendance: t={ts[1]:+.2f}   <- the DIS-specific tell (a mechanical zero)')\n"
            "print(f'DIS excess ~ price hike: t={ts[2]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: on the return that *is* the claim — DIS beating the S&P — the slope "
            f"is **{R['ll_exc_slope']:+.5f}** with *t* = **{R['ll_exc_t']:+.2f}**. That is as close to a "
            f"mechanical zero as this desk prints. The pricing-power tell is the same nothing "
            f"(*t*={R['ll_price_exc_t']:+.2f}). Even DIS's *raw* return only reaches "
            f"*t*={R['ll_abs_t']:+.2f} — sub-2, and that's before benchmarking. **H₂ rejected.**"
        ),
        md(
            "### 4c · The one number that flirts with 2 — and why it's a confound\n\n"
            "A believer will reach for a **regime split**: DIS's forward return when parks momentum is "
            "positive vs the base rate. On *absolute* return that Welch *t* touches 2 — so let's show "
            "exactly why that's the COVID regime, not a tell, by re-running it on the **excess** return."
        ),
        code(
            "if HAVE_EQ:\n"
            "    ra = st.regime_split(FRAME, 12, 1, excess=False)\n"
            "    re = st.regime_split(FRAME, 12, 1, excess=True)\n"
            "    pairs = [('absolute DIS return', ra['cond_mean']*100, ra['base_mean']*100, ra['t']),\n"
            "             ('EXCESS of SPY', re['cond_mean']*100, re['base_mean']*100, re['t'])]\n"
            "else:\n"
            "    pairs = [('absolute DIS return', R['reg_abs_cond'], R['reg_abs_base'], R['reg_abs_t']),\n"
            "             ('EXCESS of SPY', R['reg_exc_cond'], R['reg_exc_base'], R['reg_exc_t'])]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "x = np.arange(2); w=.35\n"
            "ax.bar(x-w/2, [p[1] for p in pairs], w, color=GREEN, label='parks momentum > 0')\n"
            "ax.bar(x+w/2, [p[2] for p in pairs], w, color=GREY, label='base rate')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xticks(x); ax.set_xticklabels([p[0] for p in pairs])\n"
            "ax.set_ylabel('mean forward-12m return (%)')\n"
            "for i,p in enumerate(pairs): ax.annotate(f'Welch t={p[3]:.2f}',(i, max(p[1],p[2])),ha='center',va='bottom',fontsize=10)\n"
            "ax.set_title('The absolute gap (t~2) collapses the moment you benchmark to SPY'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for p in pairs: print(f'{p[0]:22s}: cond {p[1]:+.2f}%  base {p[2]:+.2f}%  Welch t={p[3]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: on *absolute* return the split gives *t*={R['reg_abs_t']:.2f} — just "
            "over the bar. But \"parks momentum > 0\" is essentially **\"not the COVID-crater regime\"** "
            "(the negative-momentum months are 2020's −70% print, public mid-2021). DIS simply had "
            f"lower absolute returns around that market-wide shock. Benchmark to SPY and the gap "
            f"**collapses to *t*={R['reg_exc_t']:.2f}** (sub-2); use the continuous signal and it's "
            f"*t*={R['ll_exc_t']:.2f}. It's market beta wearing a parks costume — not a DIS tell."
        ),
        md(
            "### 4d · The microstructure & capacity reality — net of cost\n\n"
            "The obvious rule: long DIS when release-lagged parks momentum > 0, else rotate to SPY; "
            "1-month execution lag, 10 bps per switch leg (a DIS↔SPY swap, no borrow). Raced vs both "
            "buy-and-holds."
        ),
        code(
            "if HAVE_EQ:\n"
            "    bt = st.timing_backtest(FRAME, lag=1, cost_bps=10.0, hold_bench=True)\n"
            "    sr = [bt['net']['sharpe'], bt['buy_hold_dis']['sharpe'], bt['buy_hold_spy']['sharpe']]\n"
            "    sw, expo = bt['n_switches'], bt['exposure_dis']*100\n"
            "else:\n"
            "    sr = [R['bt_net_sharpe'], R['bt_bh_dis_sharpe'], R['bt_bh_spy_sharpe']]\n"
            "    sw, expo = R['bt_switches'], R['bt_expo']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "cols = [RED, BLUE, GREEN]\n"
            "b = ax.bar(['parks-tip rule\\n(net of cost)','buy & hold DIS','buy & hold SPY'], sr, color=cols, width=.55)\n"
            "for bb,v in zip(b,sr): ax.annotate(f'{v:.3f}',(bb.get_x()+bb.get_width()/2,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('annualised Sharpe'); ax.set_title(f'H3: the rule ({sw:.0f} switches) trails a plain index fund')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'rule net Sharpe {sr[0]:.3f}  |  buy-hold DIS {sr[1]:.3f}  |  buy-hold SPY {sr[2]:.3f}  ({sw:.0f} switches, {expo:.0f}% DIS)')"
        ),
        md(
            f"> 💡 In plain words: the rule nets Sharpe **{R['bt_net_sharpe']:.3f}** — it beats "
            f"buy-and-hold *DIS* ({R['bt_bh_dis_sharpe']:.3f}) only because it spends ~30% of the time "
            f"hiding in SPY, and it **loses badly to just holding SPY ({R['bt_bh_spy_sharpe']:.3f})**. "
            f"Its whole 'edge' is **{R['bt_switches']}** stale, once-a-year rotations. **H₃ rejected** — "
            "and as *timing*, a six-months-late annual print has no capacity to be a strategy."
        ),
        md(
            "### 4e · Positive control — the engine recovers a planted edge\n\n"
            "A deterministic momentum-plus-price path (seed 769): the same Newey-West lead-lag must "
            "stay null when no edge is planted and light up when one is — proving the nulls above are "
            "real, not a broken pipeline."
        ),
        code(
            "c0 = st.control_recovers(data.synthetic(edge=0.0), 0.0)\n"
            "c1 = st.control_recovers(data.synthetic(edge=0.02), 0.02)\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.1))\n"
            "b = ax.bar(['edge = 0\\n(null)', 'edge = 0.02\\n(planted)'], [c0['t'], c1['t']], color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=GREEN, alpha=.7); ax.axhline(0, c='k', lw=1)\n"
            "for bb,v in zip(b,[c0['t'],c1['t']]): ax.annotate(f't={v:.2f}',(bb.get_x()+bb.get_width()/2,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('Newey-West lead-lag t'); ax.set_title('Machinery proof: null stays null, planted edge lights up')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"edge=0 -> t={c0['t']:+.2f} (must stay small)   edge=0.02 -> t={c1['t']:+.2f} (must light up)\")"
        ),
        md(
            f"> 💡 In plain words: at edge=0 the engine returns *t*={R['ctrl_null_t']:+.2f} (no "
            f"manufactured significance); with a planted edge, *t*={R['ctrl_edge_t']:+.2f}. A synthetic "
            "control is a machinery proof, never market evidence — but it certifies the `NONE`/`MIRAGE` "
            "stamps are a true null, not a pipeline that couldn't detect anything."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the lead-lag slope of DIS's **excess** return on release-lagged "
            f"parks growth is {R['ll_exc_slope']:+.5f}, NW *t* = {R['ll_exc_t']:+.2f} (n={R['ll_n']}); "
            f"the price-hike tell is *t* = {R['ll_price_exc_t']:+.2f}. The lone |*t*| ≥ 2 (a regime "
            f"dummy on *absolute* return, *t*={R['reg_abs_t']:.2f}) is a COVID/market-beta artifact "
            f"that vanishes vs SPY (*t*={R['reg_exc_t']:.2f}). No robust *t* ≥ 2 for the actual claim.\n"
            f"- **Tradability `MIRAGE`** — the costed rotation nets Sharpe {R['bt_net_sharpe']:.3f} vs "
            f"buy-hold SPY {R['bt_bh_spy_sharpe']:.3f}; {R['bt_switches']} stale switches, no capacity "
            "as timing.\n"
            "- **Parks momentum leads DIS? `BUSTED`** — a six-months-late annual print can't front-run "
            "a stock whose quarterly segment reports already carry the same news. The visible crowd is "
            "real; the tradable edge is not."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the capacity & cost reality\n\n"
            "Terminal wealth of \\$10,000, 2010→2024: an S&P index fund vs the parks-tip rule (net "
            "Sharpe/CAGR from 4d) vs buy-and-hold DIS. Capacity is the second wall: the signal updates "
            "**once a year, six months late** — there is no book to scale."
        ),
        code(
            "start=10_000.0; n=15\n"
            "paths={'S&P index fund':R['spy_cagr']/100, 'parks-tip rule (net)':R['bt_net_ann']/100,\n"
            "       'buy & hold DIS':R['dis_cagr']/100}\n"
            "labels=list(paths); ends=[start*(1+g)**n for g in paths.values()]; cols=[GREEN, RED, BLUE]\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.bar(labels, ends, .55, color=cols)\n"
            "for i,v in enumerate(ends): ax.annotate(f'${v:,.0f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('value of $10,000 after 15 years'); ax.set_title('Net of cost, the clever rule still trails the index fund')\n"
            "plt.tight_layout(); plt.show()\n"
            "for l,g in paths.items(): print(f'{l:24s} ${start*(1+g)**n:>10,.0f}  ({g*100:+.1f}%/yr)')"
        ),
        md(
            "> 💡 In plain words: the passive index fund wins outright. The rule's tiny edge over "
            "buy-and-hold DIS is entirely that it parks in SPY part-time — and it never catches SPY "
            "itself. With one stale signal a year, there is no sizing or venue that turns this into an "
            "edge. **MIRAGE** is the only honest stamp."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Beat the release calendar.** Our attendance line is released on the Theme Index's real "
            "~mid-following-year schedule. Swap in a *faster* read — Disney's own quarterly Parks & "
            "Experiences revenue, or paid satellite/mobility car-counts — and test whether *that* leads "
            "before it's priced. Being early is the entire game.\n"
            "- **Per-park / attendance-vs-revenue.** Attendance ≠ profit (pricing, per-cap spend, mix). "
            "Test operating-income momentum, or split domestic vs international parks — the aggregate "
            "may hide sign.\n"
            "- **The alt-data decay prior.** McLean & Pontiff and the crowding literature: a public, "
            "journalism-friendly signal is the least edgy kind ([docs/references.md](../docs/references.md)).\n\n"
            "*The reproducible core is offline and deterministic; the attendance & ticket series are "
            "**cited, approximate proxies** released with the report's real lag. Methods: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
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
