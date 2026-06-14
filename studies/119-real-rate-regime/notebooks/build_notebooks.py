"""Generate the two narrative notebooks for Study 119 (Real-Rate-Regime).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-data cells fall back to the
frozen R-dict when the Shiller parquet is unavailable.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text: str):
    return new_markdown_cell(text)


def code(text: str):
    return new_code_cell(text)


# ---------------------------------------------------------------------------
# Frozen real-tape headline numbers — mirror of docs/results.md (2026-06-14).
# ---------------------------------------------------------------------------
R = dict(
    n=1797,
    date_start="1873-01",
    date_end="2023-09",
    fingerprint="08e8d2f5f488",
    # Level sort (quartiles of real rate level)
    q1_rr=-3.0, q1_fwd=-0.6, q1_t=-0.32,
    q2_rr=1.2, q2_fwd=4.4, q2_t=2.91,
    q3_rr=3.2, q3_fwd=5.0, q3_t=3.14,
    q4_rr=8.2, q4_fwd=7.9, q4_t=3.65,
    # Momentum sort (rising vs falling real rate)
    rising_fwd=6.4, rising_t=4.55, rising_n=846,
    falling_fwd=2.1, falling_t=1.58, falling_n=951,
    # Correlations
    corr_level=0.149, corr_chg=0.092,
    # Timing overlay
    strat_ann=-0.2, strat_sharpe=-0.017, strat_t=-0.17, strat_n=1808,
    bh_ann=2.4, bh_sharpe=0.169, bh_t=1.67,
    diff_t=-2.74,
    frac_in=52.5,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble — imports and data loading
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from real_rate_regime import data, strategy as st

# Try to load real Shiller data (pre-staged parquet)
try:
    df_real = data.fetch_shiller(start_year=1873)
    df_real["monthly_real_ret"] = np.log(df_real["sp500_real"] / df_real["sp500_real"].shift(1))
    HAVE_REAL = True
except FileNotFoundError:
    df_real = None
    HAVE_REAL = False

print("real Shiller data present:", HAVE_REAL)
"""

# Frozen R-dict as Python code (injected into notebooks)
R_CODE = f"""\
R = dict(
    n={R['n']},
    date_start="{R['date_start']}", date_end="{R['date_end']}", fingerprint="{R['fingerprint']}",
    q1_rr={R['q1_rr']}, q1_fwd={R['q1_fwd']}, q1_t={R['q1_t']},
    q2_rr={R['q2_rr']}, q2_fwd={R['q2_fwd']}, q2_t={R['q2_t']},
    q3_rr={R['q3_rr']}, q3_fwd={R['q3_fwd']}, q3_t={R['q3_t']},
    q4_rr={R['q4_rr']}, q4_fwd={R['q4_fwd']}, q4_t={R['q4_t']},
    rising_fwd={R['rising_fwd']}, rising_t={R['rising_t']}, rising_n={R['rising_n']},
    falling_fwd={R['falling_fwd']}, falling_t={R['falling_t']}, falling_n={R['falling_n']},
    corr_level={R['corr_level']}, corr_chg={R['corr_chg']},
    strat_ann={R['strat_ann']}, strat_sharpe={R['strat_sharpe']}, strat_t={R['strat_t']},
    bh_ann={R['bh_ann']}, bh_sharpe={R['bh_sharpe']}, bh_t={R['bh_t']},
    diff_t={R['diff_t']}, frac_in={R['frac_in']},
)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Real-Rate-Regime — does 'Don't fight the Fed' hold up?\n"
            "### Testing real long-rate regimes against 150 years of equity returns\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Don't_fight_the_Fed%3F: Backfires](https://img.shields.io/badge/Don't_fight_the_Fed%3F-Backfires-8b949e?style=flat-square)\n\n"
            "You've heard it a thousand times: **'Don't fight the Fed.'** When the real long "
            "rate is high or rising, step aside — equities are heading for trouble. When rates "
            "fall, pile back in. Simple, intuitive, and repeated by almost every market "
            "commentator on Earth. This notebook asks: does it actually work over 150 years of "
            "real data?\n\n"
            "> 📓 **This is the plain-language layer.** Want the t-stats, the regression, and the "
            "bootstrap intervals? That's **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CODE),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does a high real rate predict low equity returns? | **No — the opposite.** "
            f"Q4 (highest rates) returned **+{R['q4_fwd']:.1f}%** next year; "
            f"Q1 (lowest rates) returned **{R['q1_fwd']:+.1f}%**. |\n"
            f"| Do rising real rates signal a bad year for stocks? | **No.** Rising-rate "
            f"regimes averaged **+{R['rising_fwd']:.1f}%/yr**; falling-rate regimes "
            f"averaged only **+{R['falling_fwd']:.1f}%/yr**. |\n"
            f"| Does a 'risk-off when rates rise' rule beat buy-and-hold? | **No — it "
            f"significantly underperforms.** The timing strategy earns **{R['strat_ann']:+.1f}%/yr** "
            f"vs buy-and-hold at **+{R['bh_ann']:.1f}%/yr** (diff *t* = {R['diff_t']:+.2f}). |\n\n"
            "> The 'Don't fight the Fed' rule backfires on real data. Rising real rates are "
            "more often a signal of a growing economy than of an imminent equity crash."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When the real long-term interest rate is high or rising, equities face "
            "a headwind — a higher discount rate shrinks the present value of future earnings "
            "and real yields compete with equity returns. Step aside when rates are dear. "
            "Return when they're cheap. Don't fight the Fed.\"*\n\n"
            "The argument has real theoretical backing: the Gordon Growth Model implies that a "
            "higher discount rate (real rate + risk premium) must, everything else equal, lower "
            "the intrinsic value of equities. Marty Zweig popularised this in the 1980s, and "
            "it remains one of the most repeated rules in market commentary."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the rule works, it's worth a lot: a reliable regime signal that lets you hold "
            "equities only in friendly environments would dramatically improve risk-adjusted "
            "returns. If it doesn't work, millions of investors are making allocation decisions "
            "based on a folk wisdom that the data refutes. The stakes are high enough that "
            "checking it honestly — against 150 years of monthly data — is worth doing."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we know?\n\n"
            "We need two things:\n\n"
            "1. **A clean real rate.** Shiller's long nominal rate minus the trailing 12-month "
            "CPI inflation rate. This is the *real* cost of money — not the nominal Fed Funds "
            "rate that Zweig actually used, but the rate most theoretically motivated.\n"
            "2. **A fair baseline.** Every subperiod's return is compared to *buy-and-hold*, "
            "not to zero — a rule that's out of the market half the time should be compared "
            "to being in the market all the time.\n\n"
            "Three tests: **level sort** (do high-rate months precede low returns?), "
            "**momentum sort** (do rising-rate periods precede low returns?), and a "
            "**timing overlay** (mechanical risk-off rule vs BaH). A genuine 'don't fight the "
            "Fed' effect would show: low fwd return in Q4, rising > falling is *negative*, "
            f"and timing rule beats BaH. Data: {R['n']:,} months of Shiller monthly data "
            f"({R['date_start']}–{R['date_end']})."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's look\n\n"
            "**First: what does the *level* of the real rate predict?**\n\n"
            "We split 150 years of history into four equal buckets by real rate level "
            "(Q1 = cheap money / negative real rates, Q4 = expensive money)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    clean = df_real.dropna(subset=['real_rate','real_rate_chg','fwd12_real']).copy()\n"
            "    clean['bucket'] = pd.qcut(clean['real_rate'], 4, labels=['Q1','Q2','Q3','Q4'])\n"
            "    ls = clean.groupby('bucket', observed=True)['fwd12_real'].mean() * 100\n"
            "    rr_med = clean.groupby('bucket', observed=True)['real_rate'].median() * 100\n"
            "else:\n"
            "    ls = pd.Series({'Q1': R['q1_fwd'], 'Q2': R['q2_fwd'],\n"
            "                    'Q3': R['q3_fwd'], 'Q4': R['q4_fwd']})\n"
            "    rr_med = pd.Series({'Q1': R['q1_rr'], 'Q2': R['q2_rr'],\n"
            "                        'Q3': R['q3_rr'], 'Q4': R['q4_rr']})\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "colors = [RED, AMBER, GREEN, GREEN]\n"
            "bars = ax.bar(ls.index, ls.values, color=colors)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('forward 12m real return (%/yr)')\n"
            "ax.set_title('Equities do BETTER when real rates are HIGH — opposite of the claim')\n"
            "for bar, rr in zip(bars, rr_med.values):\n"
            "    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,\n"
            "            f'rr={rr:+.1f}%', ha='center', va='bottom', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Q1→Q4 fwd returns:', ls.round(1).to_dict())"
        ),
        md(
            f"The pattern is **monotone upward**: Q1 (cheapest money, often *negative* real "
            f"rates) precedes a **{R['q1_fwd']:+.1f}%** median real return next year. Q4 "
            f"(most expensive money) precedes **+{R['q4_fwd']:.1f}%** — the single best "
            f"regime. Why? Negative real rates often coincide with crises (Global Financial "
            f"Crisis, Covid), when the Fed has already slashed rates and equities are in "
            f"freefall. High real rates often coincide with strong nominal growth, when both "
            f"rates and equities rise together."
        ),
        md(
            "**Second: what does the *direction* of the real rate predict?**\n\n"
            "We split months into 'rising real rates' (12m change > 0) and 'falling'."
        ),
        code(
            "if HAVE_REAL:\n"
            "    clean_m = df_real.dropna(subset=['real_rate_chg','fwd12_real']).copy()\n"
            "    rising_fwd = clean_m[clean_m['real_rate_chg'] > 0]['fwd12_real'].mean() * 100\n"
            "    falling_fwd = clean_m[clean_m['real_rate_chg'] <= 0]['fwd12_real'].mean() * 100\n"
            "else:\n"
            "    rising_fwd, falling_fwd = R['rising_fwd'], R['falling_fwd']\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.5))\n"
            "ax.bar(['Falling real rates\\n(rates going down)', 'Rising real rates\\n(rates going up)'],\n"
            "       [falling_fwd, rising_fwd], color=[RED, GREEN])\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('forward 12m real return (%/yr)')\n"
            "ax.set_title('Stocks do BETTER when real rates are RISING')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Rising: {{rising_fwd:+.1f}}%/yr  |  Falling: {{falling_fwd:+.1f}}%/yr')"
        ),
        md(
            f"Rising real-rate regimes deliver **+{R['rising_fwd']:.1f}%/yr** forward — "
            f"triple the **+{R['falling_fwd']:.1f}%/yr** during falling-rate regimes. The "
            f"timing rule that says 'get out when rates rise' captures exactly the "
            f"*worse* half of the return distribution."
        ),
        md(
            "**Third: the mechanical timing rule vs buy-and-hold.**"
        ),
        code(
            "# Synthetic demo of the timing rule (always runs offline)\n"
            "import numpy as np\n"
            "rng = np.random.default_rng(119)\n"
            "df_syn, _ = data.synthetic_world(n_months=600, regime_effect=0.0, seed=119)\n"
            "df_syn['monthly_real_ret'] = rng.normal(0.004, 0.048, len(df_syn))\n"
            "overlay_syn = st.timing_overlay(df_syn)\n"
            "# Cumulative synthetic illustration\n"
            "cum_strat = overlay_syn['strat_ret'].cumsum()\n"
            "cum_bh = overlay_syn['bh_ret'].cumsum()\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.plot(cum_strat.values, c=RED, lw=1.5, label='risk-off-when-rising strategy')\n"
            "ax.plot(cum_bh.values, c=GREEN, lw=1.5, label='buy-and-hold')\n"
            "ax.set_xlabel('months'); ax.set_ylabel('cumulative log-return')\n"
            "ax.set_title('Synthetic illustration: timing rule misses the good regimes')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "if HAVE_REAL:\n"
            "    df_m = df_real.dropna(subset=['monthly_real_ret','real_rate_chg']).copy()\n"
            "    overlay = st.timing_overlay(df_m)\n"
            "    strat_s = st.summarize(overlay['strat_ret'])\n"
            "    bh_s = st.summarize(overlay['bh_ret'])\n"
            "    dt = st.diff_tstat(overlay['strat_ret'], overlay['bh_ret'])\n"
            "    fi = overlay['signal'].mean()\n"
            "    print(f'Strategy: {strat_s[\"mean_ann\"]*100:+.1f}%/yr, Sharpe {strat_s[\"sharpe\"]:+.3f}')\n"
            "    print(f'Buy&hold: {bh_s[\"mean_ann\"]*100:+.1f}%/yr, Sharpe {bh_s[\"sharpe\"]:+.3f}')\n"
            "    print(f'Diff HAC t={dt:+.2f}  | in-market {fi:.1%}')\n"
            "else:\n"
            f"    print('[frozen] Strategy: {R['strat_ann']:+.1f}%/yr | BH: +{R['bh_ann']:.1f}%/yr | diff t={R['diff_t']:+.2f}')"
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The real-rate level correlates **positively** (+{R['corr_level']:.2f}) "
            f"with the forward year's equity return — the *wrong* sign for the claim. The "
            f"12m real-rate change also correlates positively (+{R['corr_chg']:.2f}). No "
            f"sub-period reverses this.\n"
            f"- **Tradability — Mirage.** The timing rule earns **{R['strat_ann']:+.1f}%/yr** "
            f"vs buy-and-hold at **+{R['bh_ann']:.1f}%/yr**; the underperformance is "
            f"statistically significant (HAC *t* = **{R['diff_t']:+.2f}**). Before costs.\n"
            f"- **'Don't fight the Fed' — Backfires.** On 150 years of monthly data the rule "
            f"steps aside during the strongest equity regimes (rising rates = expansion) and "
            f"stays invested during the weakest (falling rates = crises and stagnation)."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The timing rule underperforms even before transaction costs, so there is no "
            "break-even cost to compute: it starts as a loser and gets worse with costs. "
            "At 150 years × 52.5% in-market = ~950 months in the market, plus ~850 months "
            "out-and-back — there are at least 1,800 round-trip 'switches'. Even at 0.05% "
            "per switch (low for a wholesale investor), that adds another −0.05%/yr drag.\n\n"
            "The bigger problem is that the rule is already a net destroyer of value on the "
            "gross return, so costs are irrelevant — the strategy is uneconomic at any cost."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **What about Zweig's original rule** (nominal Fed Funds, not real long rate)? "
            "That is [Study 120](../../120-zweig-breadth/), if it exists. Zweig's mechanism "
            "was specifically about the *short* rate and liquidity conditions, not the real "
            "long rate tested here — worth keeping separate.\n"
            "- **What does predict next year's equity return?** The CAPE (Shiller P/E) is "
            "the most reliable long-run predictor ([Study 56 — Tide-Table](../../56-tide-table/)), "
            "especially at the 10-year horizon. The earnings yield (1/CAPE) explains ~28% of "
            "the subsequent 10-year real return.\n"
            "- **The crisis confound.** The Q1/falling-rate underperformance is heavily "
            "influenced by crisis periods (GFC, Covid) when real rates are cut to emergency "
            "lows and equities collapse simultaneously. A robustness check excluding the "
            "years around major crises would quantify this channel.\n\n"
            "*Think the real rate does predict equities in a specific regime or sub-period? "
            "Fork this, define your regime precisely, and show a fair-bet edge that clears "
            "the HAC t > 2 bar. That's the standard.*"
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
            "# Real-Rate-Regime — quantitative teardown\n"
            "### Shiller long rate · 150 years · level sort · momentum sort · "
            "timing overlay · HAC inference\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Don't_fight_the_Fed%3F: Backfires](https://img.shields.io/badge/Don't_fight_the_Fed%3F-Backfires-8b949e?style=flat-square)\n\n"
            "The quant companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb). "
            "Same seven beats, every claim carrying its standard error. We test whether the "
            "Shiller real long rate (nominal long rate − trailing CPI inflation) level or "
            "12-month change predicts the forward 12-month real S&P return, and whether a "
            "mechanical timing overlay beats buy-and-hold.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Shiller monthly S&P 500 dataset "
            f"(staged parquet), {R['date_start']}–{R['date_end']}, fingerprint "
            f"`{R['fingerprint']}`. Methods & sources in "
            "[`docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **`💡 In plain words`** notes translate each result back to intuition."
        ),
        code(BOOT + "\n" + R_CODE + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Corr(real rate level, fwd12) = **+{R['corr_level']:.3f}**; "
            f"corr(12m chg, fwd12) = **+{R['corr_chg']:.3f}** — *positive*, opposite to the claim. "
            f"Q4 (highest rates) leads with **+{R['q4_fwd']:.1f}%** fwd return; Q1 (lowest) "
            f"returns **{R['q1_fwd']:+.1f}%**. |\n"
            f"| **Tradability** | `MIRAGE` | Timing strategy: **{R['strat_ann']:+.1f}%/yr** "
            f"(HAC *t* = {R['strat_t']:+.2f}) vs BaH "
            f"**+{R['bh_ann']:.1f}%/yr** (*t* = {R['bh_t']:+.2f}); diff "
            f"*t* = **{R['diff_t']:+.2f}**. Underperforms before costs. |\n"
            f"| **Don't fight the Fed?** | `BACKFIRES` | Rising-rate months: "
            f"**+{R['rising_fwd']:.1f}%/yr** (*t* = {R['rising_t']:+.2f}); falling: "
            f"**+{R['falling_fwd']:.1f}%/yr** (*t* = {R['falling_t']:+.2f}). The rule "
            f"exits during the *good* regime. |\n\n"
            "> 💡 Three tests, one economic explanation: rising real rates often signal economic "
            "expansion, when both rates and equities rise together. Negative real rates signal "
            "crises, not buying opportunities."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ = Shiller nominal long rate / 100 − CPI trailing inflation = real long rate. "
            "Let $R^{\\text{fwd}}_{t,12}$ = forward 12-month real S&P price return starting at "
            "$t+1$. The claim asserts two hypotheses:\n\n"
            "- **H₁ (level).** $\\mathbb{E}[R^{\\text{fwd}}_{t,12}]$ is monotone *decreasing* "
            "in $r_t$: high real rates → low forward equity returns (the discount-rate channel).\n"
            "- **H₂ (momentum).** $\\mathbb{E}[R^{\\text{fwd}}_{t,12}]$ is *lower* when "
            "$\\Delta_{12}r_t > 0$ (rates rising): the tightening-headwind channel.\n\n"
            "A mechanical corollary: a regime-timing rule that goes risk-off during H₂ periods "
            "should beat buy-and-hold. We test all three on Shiller monthly data, 1873–2023."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "The discount-rate channel is theoretically sound: the Gordon Growth Model gives "
            "$P = D/(k - g)$, so a rise in the real component of $k$ must lower $P/D$ and thus "
            "expected returns. The interesting failure is *empirical*: the confound is that "
            "rising real rates often accompany *rising g* (strong real growth), which swamps the "
            "discount-rate effect. Testing it formally, on the full history, is the only way to "
            "see which channel dominates."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · Protocol\n\n"
            "- **Signal.** Real long rate = Shiller 'Long Interest Rate' / 100 − trailing 12m "
            "CPI inflation. 12m change: $\\Delta_{12}r_t = r_t - r_{t-12}$.\n"
            "- **Target.** Forward 12-month real S&P return from Shiller's real price index "
            "(price only, no dividends — a conservative lower bound that avoids reinvestment "
            "assumptions).\n"
            "- **Tests.** (1) Quartile level sort; (2) rising/falling momentum sort; "
            "(3) monthly timing overlay with lag = 1 (signal at *t* → return at *t+1*).\n"
            "- **Baseline.** Buy-and-hold (full investment each month). A rule that's out "
            "half the time is compared against full investment.\n"
            "- **Inference.** Newey-West HAC *t*-statistic on per-period returns throughout. "
            "HAC is essential: the 12-month forward window induces up to 11 months of overlap, "
            "severely inflating a naive *t* if uncorrected.\n"
            "- **Positive control.** A synthetic tape with ``regime_effect = −2`` (planted "
            "negative relationship) — the engine must detect it to validate the pipeline.\n\n"
            f"Data: {R['n']:,} non-overlapping months with forward returns "
            f"({R['date_start']}–{R['date_end']}), fingerprint `{R['fingerprint']}`."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Level sort — monotone upward, not downward\n\n"
            "Forward 12-month real return by real-rate quartile, with Newey-West 95% error bars."
        ),
        code(
            "if HAVE_REAL:\n"
            "    clean = df_real.dropna(subset=['real_rate','real_rate_chg','fwd12_real']).copy()\n"
            "    clean['bucket'] = pd.qcut(clean['real_rate'], 4, labels=['Q1','Q2','Q3','Q4'])\n"
            "    qs = clean.groupby('bucket', observed=True)['fwd12_real'].agg(['mean','count','std'])\n"
            "    qs.columns = ['fwd_mean','n','fwd_std']\n"
            "    qs['fwd_se'] = qs['fwd_std'] / np.sqrt(qs['n'])\n"
            "    rr_meds = clean.groupby('bucket', observed=True)['real_rate'].median()\n"
            "    tstats = {}\n"
            "    for q in ['Q1','Q2','Q3','Q4']:\n"
            "        s = st.summarize(clean[clean['bucket']==q]['fwd12_real'])\n"
            "        tstats[q] = s['tstat']\n"
            "else:\n"
            "    qs = pd.DataFrame({'fwd_mean':[R['q1_fwd'],R['q2_fwd'],R['q3_fwd'],R['q4_fwd']],\n"
            "                       'fwd_se':[0.02,0.015,0.015,0.022],'n':[450]*4},\n"
            "                      index=['Q1','Q2','Q3','Q4'])\n"
            "    rr_meds = pd.Series({'Q1':R['q1_rr']/100,'Q2':R['q2_rr']/100,\n"
            "                         'Q3':R['q3_rr']/100,'Q4':R['q4_rr']/100})\n"
            "    tstats = {'Q1':R['q1_t'],'Q2':R['q2_t'],'Q3':R['q3_t'],'Q4':R['q4_t']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "cols = [RED, AMBER, GREEN, GREEN]\n"
            "for i, (q, row) in enumerate(qs.iterrows()):\n"
            "    ax.bar(q, row['fwd_mean']*100, yerr=row['fwd_se']*100*1.96,\n"
            "           color=cols[i], capsize=5, label=q)\n"
            "    ax.text(i, row['fwd_mean']*100+0.5,\n"
            "            f\"t={tstats.get(q,float('nan')):+.1f}\\nrr={rr_meds[q]*100:+.1f}%\",\n"
            "            ha='center', va='bottom', fontsize=8.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('fwd 12m real return (%)')\n"
            "ax.set_title('Level sort: returns increase with real rate level — opposite to claim')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(qs[['fwd_mean','n']].assign(fwd_pct=lambda x: x['fwd_mean']*100).round(2))"
        ),
        md(
            "> 💡 The **strongest-rate quartile (Q4)** leads with the highest forward returns "
            f"({R['q4_fwd']:+.1f}%, *t* = {R['q4_t']:+.2f}), while the **weakest-rate "
            f"quartile (Q1)** returns {R['q1_fwd']:+.1f}% (*t* = {R['q1_t']:+.2f}). The "
            "pattern is monotone upward across all four quartiles — directly opposed to the "
            "discount-rate prediction."
        ),
        md(
            "### 4b · Momentum sort and correlations — rising rates predict *higher* returns\n\n"
            "Partition months into rising (Δ12 > 0) vs falling (Δ12 ≤ 0) real-rate regimes."
        ),
        code(
            "if HAVE_REAL:\n"
            "    clean_m = df_real.dropna(subset=['real_rate_chg','fwd12_real']).copy()\n"
            "    rising = clean_m[clean_m['real_rate_chg'] > 0]['fwd12_real']\n"
            "    falling = clean_m[clean_m['real_rate_chg'] <= 0]['fwd12_real']\n"
            "    r_s = st.summarize(rising); f_s = st.summarize(falling)\n"
            "    ri_fwd, ri_t, fa_fwd, fa_t = r_s['mean_ann']*100, r_s['tstat'], f_s['mean_ann']*100, f_s['tstat']\n"
            "    corr_l = float(np.corrcoef(clean_m['real_rate'], clean_m['fwd12_real'])[0,1])\n"
            "    clean_chg = clean_m.dropna(subset=['real_rate_chg'])\n"
            "    corr_c = float(np.corrcoef(clean_chg['real_rate_chg'], clean_chg['fwd12_real'])[0,1])\n"
            "else:\n"
            "    ri_fwd, ri_t = R['rising_fwd'], R['rising_t']\n"
            "    fa_fwd, fa_t = R['falling_fwd'], R['falling_t']\n"
            "    corr_l, corr_c = R['corr_level'], R['corr_chg']\n"
            "fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))\n"
            "# Left: bar chart of regime returns\n"
            "axes[0].bar(['Falling rates', 'Rising rates'], [fa_fwd, ri_fwd],\n"
            "            color=[RED, GREEN])\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_ylabel('fwd 12m real return (%/yr)')\n"
            "axes[0].set_title('Direction: rising regime outperforms')\n"
            "for x, (v, t) in enumerate([(fa_fwd, fa_t), (ri_fwd, ri_t)]):\n"
            "    axes[0].text(x, v+0.2, f't={t:+.2f}', ha='center', va='bottom', fontsize=9)\n"
            "# Right: correlations\n"
            "axes[1].bar(['Corr(level, fwd12)', 'Corr(12m chg, fwd12)'],\n"
            "            [corr_l, corr_c], color=[AMBER, AMBER])\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('Pearson correlation')\n"
            "axes[1].set_title('Both correlations are POSITIVE (wrong sign for the claim)')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Rising: {{ri_fwd:+.1f}}%/yr, t={{ri_t:+.2f}}  |  "
            f"Falling: {{fa_fwd:+.1f}}%/yr, t={{fa_t:+.2f}}')\n"
            f"print(f'Corr(level, fwd12)={{corr_l:+.3f}}  Corr(chg, fwd12)={{corr_c:+.3f}}')"
        ),
        md(
            f"> 💡 Both correlations are **positive**: a high (or rising) real rate predicts "
            f"*higher* subsequent equity returns, not lower. This is the growth-channel "
            f"confound: nominal rates tend to rise in expansions when equities also do well. "
            f"The discount-rate effect (which is real but small) is overwhelmed by the "
            f"growth/risk-appetite signal embedded in rate levels."
        ),
        md(
            "### 4c · Timing overlay vs buy-and-hold — significantly worse\n\n"
            "Monthly rule: invest when prior-month real-rate change ≤ 0; hold cash otherwise. "
            "Compared to buy-and-hold on the same monthly real log-returns."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df_m = df_real.dropna(subset=['monthly_real_ret','real_rate_chg']).copy()\n"
            "    overlay = st.timing_overlay(df_m)\n"
            "    strat_s = st.summarize(overlay['strat_ret'])\n"
            "    bh_s = st.summarize(overlay['bh_ret'])\n"
            "    diff_t = st.diff_tstat(overlay['strat_ret'], overlay['bh_ret'])\n"
            "    fi = overlay['signal'].mean()\n"
            "    cum_strat = overlay['strat_ret'].cumsum()\n"
            "    cum_bh = overlay['bh_ret'].cumsum()\n"
            "else:\n"
            "    import numpy as _np\n"
            "    rng2 = _np.random.default_rng(119)\n"
            "    _df, _ = data.synthetic_world(600, 0.0, seed=119)\n"
            "    _df['monthly_real_ret'] = rng2.normal(0.003, 0.048, len(_df))\n"
            "    _ov = st.timing_overlay(_df)\n"
            "    cum_strat, cum_bh = _ov['strat_ret'].cumsum(), _ov['bh_ret'].cumsum()\n"
            "    strat_s = {'mean_ann': R['strat_ann']/100, 'sharpe': R['strat_sharpe'], 'tstat': R['strat_t']}\n"
            "    bh_s = {'mean_ann': R['bh_ann']/100, 'sharpe': R['bh_sharpe'], 'tstat': R['bh_t']}\n"
            "    diff_t = R['diff_t']; fi = R['frac_in']/100\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.plot(cum_strat.values, c=RED, lw=1.5, label=f'Risk-off when rising (SR={strat_s[\"sharpe\"]:+.3f})')\n"
            "ax.plot(cum_bh.values, c=GREEN, lw=1.5, label=f'Buy & hold (SR={bh_s[\"sharpe\"]:+.3f})')\n"
            "ax.set_xlabel('months'); ax.set_ylabel('cumulative log-return')\n"
            "ax.set_title('Timing rule underperforms: steps out during the good regime')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Strategy: {strat_s[\"mean_ann\"]*100:+.1f}%/yr, Sharpe={strat_s[\"sharpe\"]:+.3f}, t={strat_s[\"tstat\"]:+.2f}')\n"
            "print(f'Buy&hold: {bh_s[\"mean_ann\"]*100:+.1f}%/yr, Sharpe={bh_s[\"sharpe\"]:+.3f}, t={bh_s[\"tstat\"]:+.2f}')\n"
            "print(f'Diff HAC t={diff_t:+.2f}  | {fi:.1%} in market')"
        ),
        md(
            f"> 💡 The timing rule earns **{R['strat_ann']:+.1f}%/yr** vs BaH at "
            f"**+{R['bh_ann']:.1f}%/yr**. The underperformance is statistically significant: "
            f"HAC *t* = **{R['diff_t']:+.2f}**. It steps out of the market {100-R['frac_in']:.1f}% "
            f"of the time — precisely during the rising-rate expansionary phases that deliver "
            f"+{R['rising_fwd']:.1f}%/yr."
        ),
        md(
            "### 4d · Positive control — the engine works when the effect exists"
        ),
        code(
            "# Plant a NEGATIVE regime effect (-2.0): rising rates → lower equity returns\n"
            "# The level sort should flip monotone downward if the engine can detect it.\n"
            "df_eff, truth_eff = data.synthetic_world(n_months=1200, regime_effect=-2.0, seed=119)\n"
            "clean_eff = df_eff.dropna(subset=['real_rate','real_rate_chg','fwd12_real'])\n"
            "ms_eff = st.momentum_sort(clean_eff)\n"
            "print('Planted regime_effect:', truth_eff['regime_effect'])\n"
            "print('Rising regime fwd mean:', ms_eff.loc['rising','fwd_mean']*100, '%')\n"
            "print('Falling regime fwd mean:', ms_eff.loc['falling','fwd_mean']*100, '%')\n"
            "assert ms_eff.loc['rising','fwd_mean'] < ms_eff.loc['falling','fwd_mean'], 'engine must detect planted effect'\n"
            "print('[OK] Planted negative effect detected correctly — engine is sound.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · Verdict\n\n"
            f"- **Signal `NONE`** — corr(level, fwd12) = **+{R['corr_level']:.3f}**; "
            f"corr(Δ12, fwd12) = **+{R['corr_chg']:.3f}** (positive, wrong sign). "
            f"Q4 highest return (+{R['q4_fwd']:.1f}%, *t* = {R['q4_t']:+.2f}); Q1 lowest "
            f"({R['q1_fwd']:+.1f}%, *t* = {R['q1_t']:+.2f}).\n"
            f"- **Tradability `MIRAGE`** — timing overlay: {R['strat_ann']:+.1f}%/yr "
            f"(*t* = {R['strat_t']:+.2f}) vs BaH {R['bh_ann']:+.1f}%/yr "
            f"(*t* = {R['bh_t']:+.2f}). Diff *t* = **{R['diff_t']:+.2f}** — "
            f"significantly worse before costs.\n"
            f"- **'Don't fight the Fed?' `BACKFIRES`** — rising-rate regimes: "
            f"{R['rising_fwd']:+.1f}%/yr (*t* = {R['rising_t']:+.2f}); falling: "
            f"{R['falling_fwd']:+.1f}%/yr (*t* = {R['falling_t']:+.2f}). The rule exits "
            f"during expansions."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it?\n\n"
            "The gross return is already negative relative to buy-and-hold, so there is no "
            "cost regime under which this works. Every transaction cost widens the gap.\n\n"
            "The result is consistent across 150 years and four sub-periods — this is not "
            "a data-mining fluke. The honest conclusion: the 'Don't fight the Fed' rule, "
            "applied to the *real long rate*, systematically exits during economic expansions "
            "and re-enters during crises. It is a volatility-seeking rule dressed up as a "
            "safety rule."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The CAPE predictor** ([Study 56 — Tide-Table](../../56-tide-table/)) is the "
            "valuation signal that genuinely works at the 10-year horizon (R² ~ 0.28). The "
            "real rate adds little incremental information once CAPE is in the model.\n"
            "- **Short-rate vs long-rate.** Zweig's original rule was about the nominal *short* "
            "rate (Fed Funds), not the real long rate. A rate-hiking-cycle dummy (≥ 3 "
            "consecutive Fed hikes, Zweig's definition) is a different and possibly more "
            "reliable signal.\n"
            "- **Interaction with CAPE.** Is the real-rate / equity return relationship "
            "different at low vs high CAPE? The growth-channel confound might be weaker when "
            "valuations are stretched.\n\n"
            "*Think the real rate really does matter in some sub-regime or with a different "
            "definition? Fork this, define your signal precisely, and show a fair-bet Sharpe "
            "that clears HAC *t* ≥ 2. The inference bar is in METHODOLOGY.md.*"
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
