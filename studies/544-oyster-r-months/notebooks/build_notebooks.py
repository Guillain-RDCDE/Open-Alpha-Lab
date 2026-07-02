"""Generate the two narrative notebooks for Study 544 (Oyster-R-Months).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic figures run
anywhere, offline and deterministic; the real-tape cells use the cached monthly returns under
../_cache/ if present and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md), so the notebook re-runs for any reader offline.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-30).
# Market = S&P 500 ^GSPC 1928-2026 (1182 mo, fp e89dca918e03);
# Staples = XLP 1999-2026 (330 mo, fp 15bb025c5ac1).
R = dict(
    fp_mkt="e89dca918e03", fp_stp="15bb025c5ac1", n_mkt=1182, n_stp=330,
    asof="2026-06-30",
    # R-month split: (gap_bp, welch_t, placebo_p, ann_in_pct, ann_out_pct)
    mkt_gap_bp=-24.5, mkt_t=-0.72, mkt_p=0.457, mkt_ann_in=7.1, mkt_ann_out=10.3,
    stp_gap_bp=17.9, stp_t=0.45, stp_p=0.670, stp_ann_in=8.2, stp_ann_out=5.9,
    # sell-in-May cousin (market)
    hallo_gap_bp=40.6, hallo_t=1.31,
    # September diagnostic (market, 98 obs)
    sep_bp=-111.7,
    # tradability
    mkt_hold_cagr=6.3, mkt_hold_sharpe=0.43, mkt_hold_wealth=425,
    mkt_rule_cagr=3.4, mkt_rule_sharpe=0.31, mkt_rule_wealth=28,
    stp_hold_cagr=6.6, stp_hold_sharpe=0.58, stp_hold_wealth=5.8,
    stp_rule_cagr=4.7, stp_rule_sharpe=0.48, stp_rule_wealth=3.5,
    short_gross=-0.5, short_net=-0.8,
    # sub-periods (market gap-t)
    mkt_early_t=-0.79, mkt_late_t=0.01,
    # synthetic control: (r_premium_bp, mean gap-t over 25 seeds)
    ctrl=[(0.0, -0.05), (20.0, 0.63), (40.0, 1.30), (60.0, 1.98), (80.0, 2.65)],
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from oyster_r_months import data, strategy as st

def load_real(name):
    \"\"\"Cache-first real monthly returns (empty Series offline).\"\"\"
    return data.fetch_monthly(name, fetch=False)

MKT = load_real("market")
STP = load_real("staples")
HAVE_REAL = (not MKT.empty) and (not STP.empty)
print("real tapes present:", HAVE_REAL,
      "" if not HAVE_REAL else f"(market {len(MKT)} mo, staples {len(STP)} mo)")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The Oyster Rule — do markets only pay in months with an 'R'? 🦪\n"
            "### The 400-year-old 'eat oysters only in R months' rule, pointed at the stock market\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![R-rule_%E2%89%A0_sell-in-May%3F: Myth](https://img.shields.io/badge/R--rule_%E2%89%A0_sell--in--May%3F-Myth-8b949e?style=flat-square)\n\n"
            "There's an old kitchen rule: **only eat oysters in months with the letter *R* in their "
            "name** — September, October, November, December, January, February, March, April. Skip "
            "the R-less months: **May, June, July, August**. (Oysters spawn in summer and go milky, "
            "and shellfish spoils fastest in the heat — a food-safety rule from before "
            "refrigeration.)\n\n"
            "It's a fun rule. So let's ask a silly-but-honest question: do **stocks** follow the same "
            "calendar? Do R-months (Sep–Apr) beat the R-less summer months? We check the whole "
            "**stock market** (back to 1928) and, because the rule is about *food*, a basket of "
            "**consumer-staples** stocks (the food/drink/grocery names).\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the placebo test and the "
            "cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Did R-months beat R-less months on the market? | **No — the opposite.** R-months "
            f"earned **+{R['mkt_ann_in']}%/yr** vs the R-less summer's **+{R['mkt_ann_out']}%/yr**. "
            "The 'skip these' months actually paid *more*. |\n"
            f"| Is that gap real or noise? | **Noise.** A shuffle test gives *p* = {R['mkt_p']} — "
            "basically a coin flip. |\n"
            f"| What about food (staples) stocks? | R-months won by a whisker "
            f"(**+{R['stp_ann_in']}%** vs **+{R['stp_ann_out']}%**/yr) — but *t* = {R['stp_t']}, far "
            "too small to trust. |\n"
            f"| Is the oyster rule just 'sell in May' in disguise? | **Yes — and a worse one.** It "
            "drags **September** onto the 'buy' side, and September is the market's *worst* month. |\n\n"
            "> The oyster rule is great advice for oysters. For stocks it's a costume for "
            "'sell in May' — and the costume makes it *lose*."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"It is unwholesome in all months that have not an R in their name to eat an oyster.\"* "
            "— folk wisdom, at least as old as 1599.\n\n"
            "Turned into a market bet, it says: **hold stocks in the R-months (Sep–Apr), step aside "
            "in the R-less months (May–Aug).** If you've heard *\"sell in May and go away,\"* that "
            "should sound familiar — it's nearly the same calendar. The only difference: the oyster "
            "rule keeps you invested in **September** too."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "Calendar rules are catnip: they're simple, memorable, and *feel* like free money. But a "
            "rule that just re-labels 'sell in May' — and does it slightly *wrong* — is a perfect "
            "case study in how folklore dresses up. The desk already tested the straight sell-in-May "
            "seasonal in [Study 55 — Summer-Lull](../../55-summer-lull/); here we test the **oyster "
            "costume** of it, and see whether one extra month (September) matters."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Label every month.** Does its name contain an 'R'? Sep–Apr = yes; May–Aug = no.\n"
            "2. **Split the returns.** Average monthly return in R-months vs R-less months.\n"
            "3. **Compare.** The rule predicts R-months win. Do they?\n"
            "4. **Check it's not luck.** Shuffle the labels thousands of times and see how often pure "
            "chance beats what we saw.\n\n"
            "No forecasting here — a month's name is known before it starts, so there's nothing to "
            "'peek' at. The only real-world cost is *switching* in and out (we charge for that later)."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**R-months vs R-less months — who won, on the market and on staples?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    mk = st.split_stat(MKT, st.R_MONTHS); sp = st.split_stat(STP, st.R_MONTHS)\n"
            "    mkt_in, mkt_out = mk['ann_in']*100, mk['ann_out']*100\n"
            "    stp_in, stp_out = sp['ann_in']*100, sp['ann_out']*100\n"
            "    banner = 'REAL tape (market 1928-2026, staples 1999-2026)'\n"
            "else:\n"
            f"    mkt_in, mkt_out = {R['mkt_ann_in']}, {R['mkt_ann_out']}\n"
            f"    stp_in, stp_out = {R['stp_ann_in']}, {R['stp_ann_out']}\n"
            "    banner = 'frozen real numbers (offline)'\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "x = np.arange(2); w = 0.38\n"
            "ax.bar(x - w/2, [mkt_in, stp_in], w, label='R-months (Sep-Apr)', color=GREEN)\n"
            "ax.bar(x + w/2, [mkt_out, stp_out], w, label='R-less (May-Aug)', color=RED)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['MARKET', 'STAPLES'])\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised return %'); ax.legend()\n"
            "ax.set_title('The rule says R-months (green) win. On the market they LOSE.')\n"
            "fig.suptitle(banner, fontsize=9, color=GREY); plt.tight_layout(); plt.show()\n"
            "print(f'MARKET  R {mkt_in:.1f}%/yr vs R-less {mkt_out:.1f}%/yr')\n"
            "print(f'STAPLES R {stp_in:.1f}%/yr vs R-less {stp_out:.1f}%/yr')"
        ),
        md(
            f"On the **market**, the R-months earned *less* (**+{R['mkt_ann_in']}%/yr** vs the "
            f"R-less summer's **+{R['mkt_ann_out']}%/yr**) — the rule is **backwards**. On **staples** "
            "the R-months edge ahead, but as the quants notebook shows, that gap is statistical "
            "noise."
        ),
        md(
            "**Why is the oyster rule worse than plain 'sell in May'?** Because it keeps you invested "
            "in **September** — and September is the market's worst month. Look at the average return "
            "of every calendar month on the market:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    m = pd.DatetimeIndex(MKT.index).month\n"
            "    by_month = pd.Series(MKT.to_numpy(), index=m).groupby(level=0).mean()*100\n"
            "else:\n"
            "    by_month = pd.Series(\n"
            "        [1.3,-0.1,0.6,1.4,0.1,0.4,1.7,0.7,-1.12,0.5,0.9,1.3], index=range(1,13))\n"
            "labs = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']\n"
            "cols = [RED if i in (5,6,7,8) else GREEN for i in range(1,13)]\n"
            "cols[8] = '#7d1a12'  # September, darkest\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.bar(labs, by_month.values, color=cols, width=.7)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('avg monthly return %')\n"
            "ax.set_title('September (dark) is the worst month - and the oyster rule tells you to HOLD it')\n"
            "ax.annotate('the R-month rule\\nadds THIS to buy side', xy=(8, by_month.values[8]),\n"
            "            xytext=(5.2, -0.9), color=RED, fontsize=9,\n"
            "            arrowprops=dict(arrowstyle='->', color=RED))\n"
            "plt.tight_layout(); plt.show()\n"
            "print('September avg:', round(float(by_month.values[8]),2), '%/mo')"
        ),
        md(
            f"There's the culprit. September averages about **{R['sep_bp']} bp/month** — the single "
            "worst month on the tape. Plain 'sell in May' (skip May–Oct) *avoids* September; the "
            "oyster rule (skip only May–Aug) *holds* it. That one extra month is why the oyster "
            "version turns a soft-but-positive seasonal into a losing one."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** R-months earned *less* on the market (wrong sign, *t* {R['mkt_t']}, "
            f"shuffle *p* {R['mkt_p']}); on staples they won by a statistically invisible margin.\n"
            "- **Tradability — Mirage.** Sitting in cash all summer compounds to a fraction of just "
            "holding (see the quants notebook).\n"
            "- **R-rule ≠ sell-in-May? — Myth.** It *is* sell-in-May, in a costume — a worse one, "
            "because it hands September to the buy side."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No. Even if the sign had been right, the rule means being *out of the market for four "
            "months every year* — and on the market those four months historically **out-earned** "
            "the R-months. You'd pay switching costs to earn less. The quants notebook races the "
            "full 'hold-in-R-months' rule against buy-and-hold — it loses badly."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The straight seasonal.** [Study 55 — Summer-Lull](../../55-summer-lull/) tests plain "
            "'sell in May' on 98 years — it's *real but soft*, and still not tradable. The oyster "
            "rule is the *worse* cousin.\n"
            "- **Other calendar folklore.** [Study 95 — Holiday-Cheer](../../95-holiday-cheer/), "
            "[Study 158 — Super-Bowl](../../158-super-bowl/).\n\n"
            "*Think the oysters are onto something? Fork this, add more seafood/restaurant names or "
            "another market, and show the R-month gap clearing t = 2 the right way.*"
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
            "# The Oyster Rule — a quantitative teardown 🔬\n"
            "### R/non-R month split · two-sample Welch *t* · label-shuffle placebo · the sell-in-May cousin · the September diagnostic · hold-in-R-months vs buy-and-hold with costs · sub-periods · synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![R-rule_%E2%89%A0_sell-in-May%3F: Myth](https://img.shields.io/badge/R--rule_%E2%89%A0_sell--in--May%3F-Myth-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We split monthly returns by "
            "the R/non-R calendar, test whether R-months out-earn R-less months, and show the "
            "'signal' is sell-in-May in a costume.\n\n"
            "> ⚠️ **Not investment advice.** Real data: yfinance monthly total returns of the S&P 500 "
            f"(`^GSPC`, {R['n_mkt']} mo, fp `{R['fp_mkt']}`) and XLP (`{R['n_stp']}` mo, fp "
            f"`{R['fp_stp']}`), as-of {R['asof']}; the offline core and synthetic control run on a "
            "deterministic world. Methods in [`docs/references.md`](../docs/references.md), numbers "
            "in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Market R−nonR gap **{R['mkt_gap_bp']} bp/mo** (Welch *t* "
            f"**{R['mkt_t']}**, placebo *p* {R['mkt_p']}) — *wrong sign*; staples "
            f"**+{R['stp_gap_bp']} bp/mo** (*t* {R['stp_t']}, *p* {R['stp_p']}). No *t* ≥ 2. |\n"
            f"| **Tradability** | `MIRAGE` | Hold-in-R-months vs buy-and-hold: market **{R['mkt_rule_wealth']}x** "
            f"vs **{R['mkt_hold_wealth']}x** wealth (Sharpe {R['mkt_rule_sharpe']} vs {R['mkt_hold_sharpe']}). |\n"
            f"| **R-rule ≠ sell-in-May?** | `MYTH` | Halloween split +{R['hallo_gap_bp']} bp (*t* "
            f"+{R['hallo_t']}); the R-split adds September ({R['sep_bp']} bp/mo, worst month) → "
            f"flips to {R['mkt_gap_bp']} bp. |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), but on the real "
            "tape the R-month rule is a wrong-sign, insignificant null — a worse sell-in-May."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be the month-$t$ return and $\\mathbb{1}_R$ the indicator that month $t$ has an "
            "'R' (Sep–Apr).\n\n"
            "- **H₁ (the seasonal).** $\\mathbb{E}[r \\mid R] - \\mathbb{E}[r \\mid \\text{non-}R] > 0$ "
            "at Welch *t* ≥ 2 (R-months beat R-less).\n"
            "- **H₂ (not luck).** The gap sits in the tail of a label-shuffle null.\n"
            "- **H₃ (not just sell-in-May).** The R-split is materially different from the Halloween "
            "split.\n"
            "- **H₄ (tradable).** Hold-in-R-months beats buy-and-hold net of switching costs.\n\n"
            "We find **H₁ rejected** (wrong sign on the market, insignificant on staples), **H₂ "
            "rejected** (placebo *p* ≈ 0.46/0.67), **H₃ rejected** (it *is* sell-in-May, minus its "
            "September avoidance), and **H₄ rejected** (the rule underperforms buy-and-hold)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The content is the **why**. The Halloween seasonal (Nov–Apr > May–Oct) is real but soft "
            "([Study 55](../../55-summer-lull/)). The oyster rule re-labels it as (Sep–Apr) > "
            "(May–Aug), which moves **September** from the skip side to the hold side. Since September "
            "is the market's worst month, that single re-labelling is enough to flip the sign. This is "
            "a clean illustration of how folklore, transposed literally, becomes *anti*-signal."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Split.** $R = \\{$Sep, Oct, Nov, Dec, Jan, Feb, Mar, Apr$\\}$; non-$R = \\{$May, Jun, "
            "Jul, Aug$\\}$.\n"
            "- **Inference.** Welch (unequal-variance) two-sample *t* on $\\mathbb{E}[r\\mid R] - "
            "\\mathbb{E}[r\\mid \\text{non-}R]$; a **label-shuffle placebo** (5000 perms) preserving "
            "the count of R-months.\n"
            "- **Cousin.** The same statistic on the Halloween split (Nov–Apr vs May–Oct).\n"
            "- **Diagnostic.** The per-month means, to locate September's drag.\n"
            "- **Frictions.** Hold-in-R-months vs buy-and-hold, one-way cost on each switch; a "
            "long-R/short-R-less sensitivity with a borrow on the short leg.\n"
            "- **Sub-periods.** Pre/post-2000 gap-*t*.\n"
            "- **Positive control.** A deterministic synthetic world with a planted R-premium and a "
            "null — averaged over 25 seeds (house rule).\n\n"
            "Timing: the R/non-R label is a calendar fact known before each month begins — no "
            "estimation, no look-ahead. The only convention is the month-boundary fill; a one-day "
            "shift is immaterial on a monthly bar."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The R-month split — H₁ & H₂\n\n"
            "The Welch *t* on the R−nonR gap, with the label-shuffle placebo, on both instruments."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for lab, ser in [('market', MKT), ('staples', STP)]:\n"
            "        s = st.split_stat(ser, st.R_MONTHS)\n"
            "        p = st.placebo_pvalue(ser, st.R_MONTHS, n_perm=5000, seed=544)\n"
            "        rows.append((lab, s['gap']*1e4, s['t'], p))\n"
            "else:\n"
            f"    rows = [('market', {R['mkt_gap_bp']}, {R['mkt_t']}, {R['mkt_p']}),\n"
            f"            ('staples', {R['stp_gap_bp']}, {R['stp_t']}, {R['stp_p']})]\n"
            "tab = pd.DataFrame(rows, columns=['instrument','gap_bp','welch_t','placebo_p']).set_index('instrument')\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "cols = [GREEN if g>0 else RED for g in tab['gap_bp']]\n"
            "ax.bar(tab.index, tab['gap_bp'], color=cols, width=.45)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('R - nonR gap (bp/mo)')\n"
            "ax.set_title('R-month gap: wrong sign on market, tiny on staples')\n"
            "for i,(g,t) in enumerate(zip(tab['gap_bp'], tab['welch_t'])):\n"
            "    ax.annotate(f't={t:+.2f}', (i, g), ha='center',\n"
            "                va='bottom' if g>0 else 'top', fontsize=10)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(3).to_string())"
        ),
        md(
            f"> 💡 In plain words: **H₁ and H₂ rejected.** The market gap is **{R['mkt_gap_bp']} bp/mo** "
            f"the *wrong* way (Welch *t* {R['mkt_t']}, placebo *p* {R['mkt_p']}); staples is "
            f"**+{R['stp_gap_bp']} bp** but *t* {R['stp_t']}, *p* {R['stp_p']} — indistinguishable "
            "from noise. Neither clears *t* ≥ 2."
        ),
        md(
            "### 4b · The sell-in-May cousin & the September culprit — H₃\n\n"
            "Compare the R-split to the Halloween split on the market, then show the per-month means "
            "that explain the difference."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r_stat = st.split_stat(MKT, st.R_MONTHS)\n"
            "    h_stat = st.split_stat(MKT, st.HALLOWEEN_WINTER)\n"
            "    r_gap, r_t = r_stat['gap']*1e4, r_stat['t']\n"
            "    h_gap, h_t = h_stat['gap']*1e4, h_stat['t']\n"
            "    m = pd.DatetimeIndex(MKT.index).month\n"
            "    sep = float(MKT.to_numpy()[m==9].mean()*1e4)\n"
            "else:\n"
            f"    r_gap, r_t, h_gap, h_t, sep = {R['mkt_gap_bp']}, {R['mkt_t']}, {R['hallo_gap_bp']}, {R['hallo_t']}, {R['sep_bp']}\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.2))\n"
            "ax.bar(['Halloween\\n(Nov-Apr vs May-Oct)','R-months\\n(Sep-Apr vs May-Aug)'],\n"
            "       [h_gap, r_gap], color=[GREEN, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('in - out gap (bp/mo)')\n"
            "ax.set_title(f'Same seasonal, one month apart: adding Sept ({sep:.0f} bp) flips it')\n"
            "ax.annotate(f't={h_t:+.2f}', (0, h_gap), ha='center', va='bottom')\n"
            "ax.annotate(f't={r_t:+.2f}', (1, r_gap), ha='center', va='top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Halloween gap {h_gap:+.1f} bp (t {h_t:+.2f}) | R-month gap {r_gap:+.1f} bp (t {r_t:+.2f}) | September {sep:+.1f} bp')"
        ),
        md(
            f"> 💡 In plain words: **H₃ rejected — the R-rule IS sell-in-May.** The Halloween split is "
            f"positive (**+{R['hallo_gap_bp']} bp**, *t* +{R['hallo_t']}); the only structural change "
            f"in the R-split is that **September** ({R['sep_bp']} bp/mo, the worst month) moves to the "
            f"hold side, flipping the gap to **{R['mkt_gap_bp']} bp**. It's the same seasonal wearing "
            "an oyster costume — a *worse*-fitting one."
        ),
        md(
            "### 4c · Sub-periods — was it ever there?\n\n"
            "Split the market at 2000 and read the gap-*t* on each half."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp = st.subperiod_stats(MKT, '2000-01-01', st.R_MONTHS)\n"
            "    early_t, late_t = sp['early']['t'], sp['late']['t']\n"
            "else:\n"
            f"    early_t, late_t = {R['mkt_early_t']}, {R['mkt_late_t']}\n"
            "fig, ax = plt.subplots(figsize=(6.8, 4.0))\n"
            "ax.bar(['pre-2000','post-2000'], [early_t, late_t], color=[RED, GREY], width=.45)\n"
            "ax.axhline(0, c='k', lw=1); ax.axhline(2, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1); ax.set_ylabel('R-month gap Welch t')\n"
            "ax.set_title('No era with a real R-month edge (|t| never near 2)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'market gap-t: pre-2000 {early_t:+.2f} | post-2000 {late_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: negative-ish early (**{R['mkt_early_t']}**), ~0 late "
            f"(**{R['mkt_late_t']}**). There is no sub-period in which the R-month rule was a real "
            "edge — this is a null, not a decayed signal."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — H₁ rejected (market gap {R['mkt_gap_bp']} bp, *t* {R['mkt_t']}, "
            f"wrong sign; staples *t* {R['stp_t']}); H₂ rejected (placebo *p* {R['mkt_p']}/{R['stp_p']}).\n"
            f"- **Tradability `MIRAGE`** — the hold-in-R-months rule underperforms buy-and-hold (next "
            "beat).\n"
            "- **R-rule ≠ sell-in-May? `MYTH`** — H₃ rejected; it *is* sell-in-May, worsened by "
            "September."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the rule vs buy-and-hold, with costs\n\n"
            "Hold the instrument in R-months, sit in cash in R-less months; charge 5 bps per switch."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for lab, ser in [('market', MKT), ('staples', STP)]:\n"
            "        rv = st.rule_vs_hold(ser, st.R_MONTHS, cost_bps=5.0)\n"
            "        rows.append((lab, rv['hold_wealth'], rv['rule_net_wealth'],\n"
            "                     rv['hold_cagr']*100, rv['rule_net_cagr']*100,\n"
            "                     rv['hold_sharpe'], rv['rule_net_sharpe']))\n"
            "    tab = pd.DataFrame(rows, columns=['inst','hold_x','rule_x','hold_cagr','rule_cagr','hold_sh','rule_sh']).set_index('inst')\n"
            "else:\n"
            "    tab = pd.DataFrame({\n"
            f"        'hold_x':[{R['mkt_hold_wealth']},{R['stp_hold_wealth']}],\n"
            f"        'rule_x':[{R['mkt_rule_wealth']},{R['stp_rule_wealth']}],\n"
            f"        'hold_cagr':[{R['mkt_hold_cagr']},{R['stp_hold_cagr']}],\n"
            f"        'rule_cagr':[{R['mkt_rule_cagr']},{R['stp_rule_cagr']}],\n"
            f"        'hold_sh':[{R['mkt_hold_sharpe']},{R['stp_hold_sharpe']}],\n"
            f"        'rule_sh':[{R['mkt_rule_sharpe']},{R['stp_rule_sharpe']}]}}, index=['market','staples'])\n"
            "fig, ax = plt.subplots(figsize=(8, 4.3))\n"
            "x = np.arange(len(tab)); w=0.38\n"
            "ax.bar(x-w/2, tab['hold_cagr'], w, label='buy & hold', color=GREEN)\n"
            "ax.bar(x+w/2, tab['rule_cagr'], w, label='R-month rule (net)', color=RED)\n"
            "ax.set_xticks(x); ax.set_xticklabels(tab.index); ax.set_ylabel('CAGR %'); ax.legend()\n"
            "ax.set_title('Sitting out the summer compounds to less, at a lower Sharpe')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tab.round(2).to_string())"
        ),
        md(
            f"> 💡 In plain words: `MIRAGE`. On the market the rule turns **{R['mkt_hold_wealth']}x** "
            f"of buy-and-hold into **{R['mkt_rule_wealth']}x** (Sharpe {R['mkt_hold_sharpe']} → "
            f"{R['mkt_rule_sharpe']}). The literal long-R/short-R-less book is worse: gross "
            f"**{R['short_gross']}%/yr**, **{R['short_net']}%** net of a 50 bps short borrow."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a real R-month "
            "premium of increasing size and watch the Welch *t* climb past +2 — averaged over 25 "
            "seeds so no single lucky seed can fake it."
        ),
        code(
            "premia = [0.0, 20.0, 40.0, 60.0, 80.0]\n"
            "ts = [st.synthetic_mean_t(data, r_premium_bp=a, n_seeds=25) for a in premia]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(premia, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = +2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted R-month premium (bp/mo)')\n"
            "ax.set_ylabel('mean R-month Welch t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted R-premium - flat at the null, past +2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for a, t in zip(premia, ts): print(f'premium {a:+5.0f} bp -> mean t {t:+.2f}')"
        ),
        md(
            "The Welch *t* is ≈ 0 at the null and rises past +2 as the planted premium grows — so the "
            "engine is a faithful detector. The flat/wrong-sign real-tape result is therefore the "
            "**tape talking**: the R-month oyster rule is a genuine null, and a *worse* sell-in-May. "
            "For the straight seasonal see [Study 55 — Summer-Lull](../../55-summer-lull/)."
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
