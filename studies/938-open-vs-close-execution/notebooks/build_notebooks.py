"""Generate the two narrative notebooks for Study 938 (Open or Close).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below, which mirrors ``docs/results.md``; the only live cells run the fast
offline synthetic control, and they are labelled as synthetic — never under a real banner.
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


# --------------------------------------------------------------------------- #
# Frozen real-tape headline — mirror of docs/results.md.
# SPY / IWM / EEM / EFA vs BIL, adjusted OHLC, 2007-05-30 -> 2026-06-30, 2 bps one-way.
# --------------------------------------------------------------------------- #
R = dict(
    start="2007-05-30", end="2026-06-30", n_days=4802, fp="f36d90ae4fdc",
    cost_bps=2.0, split="2017-01-01",
    # 10-month rule, per tape: (trades, gap bps/yr, HAC t, per-trade bps, win rate,
    #                           exSharpe open, exSharpe close)
    m_tapes={
        "SPY": (31, -4.8, -0.16, -2.9, 0.45, 0.587, 0.589),
        "IWM": (37, 50.7, 1.04, 25.1, 0.59, 0.363, 0.331),
        "EEM": (37, 66.6, 2.40, 32.9, 0.70, 0.313, 0.270),
        "EFA": (33, 35.4, 1.69, 19.6, 0.61, 0.299, 0.272),
    },
    m_gap=37.0, m_t=1.53, m_ci=(-12.0, 85.3), m_ci_neg=6.8,
    m_sh_open=0.444, m_sh_close=0.413,
    # growth rates: excess-of-cash first (the series the Sharpe uses), then raw cash-inclusive
    m_cagr_open=4.90, m_cagr_close=4.51, m_cagr_open_tot=6.21, m_cagr_close_tot=5.82,
    m_era_early=(57.4, 1.96), m_era_late=(17.9, 0.47),
    # 20-week rule
    w_tapes={
        "SPY": (88, -35.8, -0.81, -7.6, 0.50, 0.588, 0.615),
        "IWM": (114, -19.0, -0.29, -3.1, 0.46, 0.382, 0.392),
        "EEM": (102, -26.2, -0.54, -4.8, 0.45, 0.333, 0.349),
        "EFA": (96, -23.4, -0.66, -4.6, 0.45, 0.412, 0.432),
    },
    w_gap=-26.1, w_t=-0.80, w_ci=(-90.9, 35.7), w_ci_neg=80.7,
    w_sh_open=0.494, w_sh_close=0.515,
    w_cagr_open=5.27, w_cagr_close=5.54, w_cagr_open_tot=6.64, w_cagr_close_tot=6.92,
    w_era_early=(27.2, 0.58), w_era_late=(-78.0, -1.73),
    # pooled fills — naive first, then clustered on the trade date (the honest version:
    # 538 fills sit on 361 dates, four correlated tapes flipping together)
    n_trades=538, per_trade_bps=1.39, per_trade_t=0.33,
    wins=267, win_rate=49.6, wilson=(45.4, 53.8),
    n_dates=361, per_trade_t_clu=0.23, clu_mean_ci=(-10.8, 13.0), clu_win=(44.1, 55.1),
    # dispersion across the 8 rule x tape cells
    disp_mean=5.4, disp_sd=39.5, disp_lo=-35.8, disp_hi=66.6, disp_fires=1,
    # opening-auction penalty sweep (PROXY): extra bps -> (monthly gap, t), (weekly gap, t)
    pen={0.0: ((37.0, 1.53), (-26.1, -0.80)),
         1.0: ((35.1, 1.46), (-31.5, -0.97)),
         2.0: ((33.2, 1.38), (-36.8, -1.13)),
         5.0: ((27.5, 1.15), (-52.9, -1.62))},
    # cash-split PROXY sweep (SPY, 10-month): frac -> gap bps/yr
    cash_sweep={0.0: -4.87, 0.5: -4.84, 0.73: -4.85, 1.0: -4.89},
    # mechanism: entry / exit intraday bps, 10-month then 20-week spread
    mech_m={"SPY": (31.9, 39.8, -7.9, -0.22), "IWM": (0.9, -50.4, 51.2, 1.06),
            "EEM": (48.0, -17.1, 65.1, 2.51), "EFA": (25.2, -13.5, 38.7, 1.68)},
    mech_w_spreads={"SPY": (-15.4, -0.82), "IWM": (-6.4, -0.29),
                    "EEM": (-9.9, -0.57), "EFA": (-9.4, -0.69)},
    # synthetic control (offline, 8 seeds)
    syn_planted=(134.6, 28.2, 3.84, 8), syn_null=(13.8, 38.4, 0.55, 1),
)

HEADER = f"""# Study 938 — Open or Close 🔔

**The same timing rule, filled at tomorrow's open or at tomorrow's close. Does it matter?**

Every tactical back-test quietly picks a fill venue and almost none report the other choice.
We take Faber's moving-average filter — hold the ETF while its period-end close is above the
mean of the last `L` period-end closes, else hold **BIL** (T-bills) — run it in two flavours
(**10-month**, month-end; **20-week**, week-end) on **SPY / IWM / EEM / EFA**, and fill it
twice: once at the **open** of the next session, once at its **close**.

Window {R['start']} → {R['end']} ({R['n_days']:,} days), {R['cost_bps']:.0f} bps one-way × NAV,
long-only so no borrow. Adjusted OHLC (`auto_adjust=True`): the intraday leg is **price-only**,
the overnight leg carries the dividend.

*Real numbers below are the frozen headline (`docs/results.md`, Fingerprint `{R['fp']}`,
as-of 2026-06-30). The only live cells run the offline **synthetic** control and say so.*
"""


# --------------------------------------------------------------------------- #
def build_curious():
    m, w = R["m_tapes"], R["w_tapes"]
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. Why the choice looks free — and why it isn't obviously free\n\n"
           "Fix the rule and fix the lag: the signal is known at tonight's close, and you fill "
           "tomorrow. On every day you *don't* trade, the two versions of you own exactly the "
           "same thing. The only days they differ are the handful when the rule flips — and on "
           "those days the difference is one single session's move, the **open-to-close** move.\n\n"
           "So the whole question shrinks to: *is the trading session after a buy signal "
           "systematically greener than the session after a sell signal?* If trends carry over "
           "into the next day, filling at the open should win. If the opening auction is just "
           "the expensive place to trade, filling at the close should win."),
        md("## 2. The first answer looks exciting"),
        code(
            "R = " + repr({"m_tapes": R["m_tapes"], "m_gap": R["m_gap"], "m_t": R["m_t"]}) + "\n"
            "print('10-month rule — gap = (fill at the open) minus (fill at the close)')\n"
            "for tk, v in R['m_tapes'].items():\n"
            "    print(f\"  {tk}: {v[0]:3d} trades   {v[1]:+7.1f} bps/yr   HAC t {v[2]:+5.2f}   \"\n"
            "          f\"open wins {v[4]:.0%} of fills\")\n"
            "print(f\"  POOLED: {R['m_gap']:+.1f} bps/yr (HAC t {R['m_t']:+.2f})\")"
        ),
        md(f"Three of the four tapes lean the same way and the pooled book gains "
           f"**{R['m_gap']:+.1f} bps/yr** from filling at the open. Emerging markets even clears "
           f"the significance bar on its own (*t* = {m['EEM'][2]:+.2f}). This is the point at "
           f"which a lot of back-tests would stop, declare an execution edge, and hard-code "
           f"\"fill at the open\" into the production script."),
        md("## 3. Then you run a faster version of the same filter\n\n"
           "Same four tapes, same window, same costs — only now the filter is rebalanced weekly "
           "instead of monthly, which roughly triples the number of fills and therefore the "
           "statistical power. Note it is a *shorter* rule too (20 weeks ≈ 100 sessions of "
           "lookback against ≈ 210 for ten months), so this is a cousin, not a clone."),
        code(
            "W = " + repr({"w_tapes": R["w_tapes"], "w_gap": R["w_gap"], "w_t": R["w_t"]}) + "\n"
            "print('20-week rule — same tapes, ~3x the fills')\n"
            "for tk, v in W['w_tapes'].items():\n"
            "    print(f\"  {tk}: {v[0]:3d} trades   {v[1]:+7.1f} bps/yr   HAC t {v[2]:+5.2f}   \"\n"
            "          f\"open wins {v[4]:.0%} of fills\")\n"
            "print(f\"  POOLED: {W['w_gap']:+.1f} bps/yr (HAC t {W['w_t']:+.2f})  <- the sign flipped\")"
        ),
        md(f"**All four flip.** The same idea, measured with more data, says the opposite: "
           f"filling at the open now *loses* {abs(R['w_gap']):.0f} bps/yr. When a result reverses "
           f"the moment you give it more observations, the honest reading is that it was never "
           f"there.\n\n"
           f"> 🔬 **For the quants** — the two rules are not independent samples of the same "
           f"quantity, but they share the tapes, the window and the costs, so a genuine venue "
           f"effect would have to show up in both. Bootstrap CIs on the pooled gap: monthly "
           f"[{R['m_ci'][0]:+.0f}, {R['m_ci'][1]:+.0f}] bps/yr, weekly "
           f"[{R['w_ci'][0]:+.0f}, {R['w_ci'][1]:+.0f}] — both straddle zero."),
        md("## 4. Every fill on the desk, in one line"),
        code(
            "P = " + repr({k: R[k] for k in ("n_trades", "per_trade_bps", "per_trade_t",
                                             "wins", "win_rate", "wilson", "n_dates",
                                             "per_trade_t_clu", "clu_win")}) + "\n"
            "print(f\"all {P['n_trades']} fills, both rules, all four tapes\")\n"
            "print(f\"  mean slippage of the open fill : {P['per_trade_bps']:+.2f} bps  (t = {P['per_trade_t']:+.2f})\")\n"
            "print(f\"  the open fill won              : {P['wins']}/{P['n_trades']} = {P['win_rate']:.1f}%\")\n"
            "print(f\"  95% confidence on that win rate: [{P['wilson'][0]:.1f}%, {P['wilson'][1]:.1f}%]\")\n"
            "print(f\"\\n  but the fills are not independent - they sit on only {P['n_dates']} dates\")\n"
            "print(f\"  clustered on the trade date    : t = {P['per_trade_t_clu']:+.2f}, \"\n"
            "      f\"win rate 95% CI [{P['clu_win'][0]:.1f}%, {P['clu_win'][1]:.1f}%]\")\n"
            "print('\\n  -> a coin flip, and slightly more of one once you correct for that.')"
        ),
        md("## 5. The coin flip is free — but the noise it buys is not\n\n"
           f"Zero expected return does not mean zero consequence. Across the eight rule × tape "
           f"cells the *realised* venue gap ranges from **{R['disp_lo']:+.1f}** to "
           f"**{R['disp_hi']:+.1f} bps/yr** (sd {R['disp_sd']:.0f}). So an arbitrary, unexamined "
           f"line in your execution code moves the reported CAGR by roughly **±0.4 pp/yr** — "
           f"enough to make a mediocre strategy look respectable, or the reverse. It is the "
           f"same disease as rebalance-timing luck ([Study 836](../../836-timing-luck/)), one "
           f"dimension down: not *which day* you trade, but *what time of day*."),
        md("## 6. Is the harness even awake? (offline synthetic — not the real tape)\n\n"
           "A null result is only worth reading if the test could have found something. The "
           "synthetic tape below books a *fixed* daily move disproportionately into the trading "
           "session after a strong or weak month — a planted venue edge. The close-executed arm "
           "is mathematically untouched by the planting, so only the open arm can move."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..', '..', '..')))\n"
            "import numpy as np\n"
            "from open_close_exec import data, strategy as st\n"
            "for ss, tag in [(1.0, 'planted venue edge'), (0.0, 'null (no edge)')]:\n"
            "    t = np.array([st.synthetic_detect(\n"
            "            data.synthetic_daily(signal_strength=ss, seed=938 + 11 * s)[0])['gap_t_hac']\n"
            "        for s in range(8)])\n"
            "    print(f\"SYNTHETIC {tag:20s}: mean HAC t {t.mean():+.2f}, \"\n"
            "          f\"fires |t|>=2 on {int((abs(t) >= 2).sum())}/8 seeds\")"
        ),
        md("The detector fires on **every** planted seed and drops to roughly the nominal "
           "false-positive rate (1 seed in 8) once the edge is removed. The zero we measured on "
           "the real tape is a property of the tape, not a broken test."),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** Over {R['n_trades']} real fills the open-versus-close slippage "
           f"is **{R['per_trade_bps']:+.2f} bps per trade** (*t* = {R['per_trade_t']:+.2f} naive, "
           f"{R['per_trade_t_clu']:+.2f} clustered on the trade date) and the "
           f"open fill wins **{R['win_rate']:.1f}%** of the time, 95% CI "
           f"[{R['wilson'][0]:.1f}%, {R['wilson'][1]:.1f}%] and "
           f"[{R['clu_win'][0]:.1f}%, {R['clu_win'][1]:.1f}%] clustered. The monthly and weekly rules print "
           f"opposite signs on the same tapes; the era cut disagrees with itself; one cell in "
           f"eight clears |*t*| = 2, which is what chance delivers.\n"
           f"- **Tradability — Mirage.** There is no sign to commit to in advance, so no venue "
           f"rule can be written down. What survives is the noise: ±0.4 pp/yr of free "
           f"track-record luck. And once the opening auction is charged its real, wider spread "
           f"both rules drift toward the **close** — so fill at the close because it is cheaper "
           f"and deeper, and expect exactly nothing for it."),
    ]
    nb["cells"] = cells
    return nb


# --------------------------------------------------------------------------- #
def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 938 — Open or Close — the teardown\n\n"
           "The sliver algebra, per-tape HAC *t* on two rules, block-bootstrap CIs, the era cut, "
           "the entry-minus-exit intraday decomposition, both PROXY sweeps, and the live "
           "synthetic control. Every real number is frozen from `docs/results.md` "
           f"(Fingerprint `{R['fp']}`, as-of 2026-06-30); the synthetic cells are labelled."),
        code("R = " + repr(R)),
        md("## Setup — exactly one execution lag, two venues\n\n"
           "Let `target[t]` be the rule's weight formed from data through the close of day `t`. "
           "Both arms fill on `t+1`:\n\n"
           "* **open arm** — overnight leg of `t+1` at `target[t-1]`, intraday leg at `target[t]`;\n"
           "* **close arm** — the whole of `t+1` at `target[t-1]`, `target[t]` from that close.\n\n"
           "Hence on any day `u`,\n\n"
           "```\n"
           "r_open[u] - r_close[u] = (w_id[u] - w_on[u]) * (r_intraday[u] - r_intraday_cash[u])\n"
           "                         + O((daily return)^2)\n"
           "```\n\n"
           "which is exactly zero whenever `w_id == w_on` — i.e. on every non-trade day. The "
           "study is therefore a test on ~540 observations, not ~4,800, and its power is capped "
           "by the rules' turnover.\n\n"
           "> 💡 **In plain words** — the two versions of the portfolio own the same thing except "
           "on the days the rule flips, and on those days they differ by one trading session."),
        md("## Headline — 10-month rule (month-end rebalance), 2 bps one-way"),
        code(
            "print(f\"{'tape':5s} {'trades':>7s} {'gap bps/yr':>11s} {'HAC t':>7s} \"\n"
            "      f\"{'per-trade':>10s} {'open wins':>10s}  exSharpe open/close\")\n"
            "for tk, v in R['m_tapes'].items():\n"
            "    print(f\"{tk:5s} {v[0]:7d} {v[1]:+11.1f} {v[2]:+7.2f} {v[3]:+10.1f} \"\n"
            "          f\"{v[4]:10.0%}  {v[5]:+.3f} / {v[6]:+.3f}\")\n"
            "print(f\"\\npooled (EW 4 tapes): {R['m_gap']:+.1f} bps/yr  HAC t {R['m_t']:+.2f}  \"\n"
            "      f\"95% block-bootstrap CI [{R['m_ci'][0]:+.1f}, {R['m_ci'][1]:+.1f}] \"\n"
            "      f\"(share<0 {R['m_ci_neg']:.1f}%)\")\n"
            "print(f\"pooled exSharpe: open {R['m_sh_open']:+.3f} vs close {R['m_sh_close']:+.3f} \"\n"
            "      f\"(d {R['m_sh_open']-R['m_sh_close']:+.3f})\")\n"
            "print(f\"excess-of-cash CAGR {R['m_cagr_open']:.2f}% vs {R['m_cagr_close']:.2f}%  |  \"\n"
            "      f\"raw, cash-inclusive {R['m_cagr_open_tot']:.2f}% vs {R['m_cagr_close_tot']:.2f}%\")"
        ),
        md("Gross equals net here: both venues pay the same 2 bps one-way × NAV on the same "
           "trade days, and there is no short leg, so no borrow. Only EEM clears |*t*| = 2 — on "
           "37 trades. Both growth rates are printed and kept apart: the Sharpes are built on the "
           "**excess-of-cash** series, and the raw figure is ~1.3 pp/yr higher purely because the "
           "book sits in T-bills for a large share of the sample."),
        md("## A faster cousin, three times the fills — 20-week (week-end rebalance)\n\n"
           "Twenty weeks is ≈ 100 sessions of lookback against ≈ 210 for ten months, so this arm "
           "is shorter-horizon as well as higher-frequency. It is the power check, not a clean "
           "replication."),
        code(
            "print(f\"{'tape':5s} {'trades':>7s} {'gap bps/yr':>11s} {'HAC t':>7s} \"\n"
            "      f\"{'per-trade':>10s} {'open wins':>10s}  exSharpe open/close\")\n"
            "for tk, v in R['w_tapes'].items():\n"
            "    print(f\"{tk:5s} {v[0]:7d} {v[1]:+11.1f} {v[2]:+7.2f} {v[3]:+10.1f} \"\n"
            "          f\"{v[4]:10.0%}  {v[5]:+.3f} / {v[6]:+.3f}\")\n"
            "print(f\"\\npooled: {R['w_gap']:+.1f} bps/yr  HAC t {R['w_t']:+.2f}  \"\n"
            "      f\"95% CI [{R['w_ci'][0]:+.1f}, {R['w_ci'][1]:+.1f}] (share<0 {R['w_ci_neg']:.1f}%)\")\n"
            "print(f\"pooled exSharpe: open {R['w_sh_open']:+.3f} vs close {R['w_sh_close']:+.3f} \"\n"
            "      f\"(d {R['w_sh_open']-R['w_sh_close']:+.3f})\")\n"
            "print(f\"excess-of-cash CAGR {R['w_cagr_open']:.2f}% vs {R['w_cagr_close']:.2f}%  |  \"\n"
            "      f\"raw, cash-inclusive {R['w_cagr_open_tot']:.2f}% vs {R['w_cagr_close_tot']:.2f}%\")\n"
            "print('\\nall four tapes reverse sign relative to the monthly rule.')"
        ),
        md("## Pooled over every fill, and the era cut\n\n"
           "The pooled sample is **not** 538 independent draws: SPY / IWM / EEM / EFA are highly "
           "correlated and their moving-average signals flip on the same days, so the fills sit on "
           "361 distinct dates, and the two rules share one window. The naive *t* and the Wilson "
           "interval assume that away; clustering on the trade date is the minimum correction, and "
           "it is reported next to them."),
        code(
            "print(f\"{R['n_trades']} fills pooled: mean {R['per_trade_bps']:+.2f} bps \"\n"
            "      f\"(naive t = {R['per_trade_t']:+.2f}), open wins {R['wins']}/{R['n_trades']} = \"\n"
            "      f\"{R['win_rate']:.1f}%, 95% Wilson [{R['wilson'][0]:.1f}%, {R['wilson'][1]:.1f}%]\")\n"
            "print(f\"clustered on trade date ({R['n_dates']} dates): t = {R['per_trade_t_clu']:+.2f}, \"\n"
            "      f\"mean 95% CI [{R['clu_mean_ci'][0]:+.1f}, {R['clu_mean_ci'][1]:+.1f}] bps, \"\n"
            "      f\"win rate 95% CI [{R['clu_win'][0]:.1f}%, {R['clu_win'][1]:.1f}%]\")\n"
            "print(f\"\\nera cut at {R['split']} (pooled book):\")\n"
            "print(f\"  10-month : early {R['m_era_early'][0]:+7.1f} (t {R['m_era_early'][1]:+.2f})   \"\n"
            "      f\"late {R['m_era_late'][0]:+7.1f} (t {R['m_era_late'][1]:+.2f})\")\n"
            "print(f\"  20-week  : early {R['w_era_early'][0]:+7.1f} (t {R['w_era_early'][1]:+.2f})   \"\n"
            "      f\"late {R['w_era_late'][0]:+7.1f} (t {R['w_era_late'][1]:+.2f})\")"
        ),
        md("> 💡 **In plain words** — no half of the sample agrees with the other, and the two "
           "rules disagree with each other in the recent era."),
        md("## Mechanism — the entry-minus-exit intraday spread\n\n"
           "Entries carry `Δw = +1`, exits `Δw = −1`, and they alternate. A *constant* intraday "
           "drift therefore cancels over a full cycle; a venue edge requires the **conditional** "
           "means to differ. Mean intraday (price-only) return on trade days:"),
        code(
            "print('10-month rule:')\n"
            "print(f\"  {'tape':5s} {'entry bps':>10s} {'exit bps':>10s} {'spread':>9s} {'Welch t':>8s}\")\n"
            "for tk, v in R['mech_m'].items():\n"
            "    print(f\"  {tk:5s} {v[0]:+10.1f} {v[1]:+10.1f} {v[2]:+9.1f} {v[3]:+8.2f}\")\n"
            "print('\\n20-week rule (entry - exit spread only):')\n"
            "for tk, v in R['mech_w_spreads'].items():\n"
            "    print(f\"  {tk:5s} {v[0]:+9.1f} bps  (Welch t {v[1]:+.2f})\")\n"
            "print('\\nall four spreads flip negative on the higher-powered rule.')"
        ),
        md("The one monthly spread that clears |*t*| = 2 (EEM, +65.1 bps, Welch *t* = +2.51, on "
           "19 entries against 18 exits) does not survive the switch to weekly rebalancing, where "
           "the same tape's spread is **−9.9 bps** (*t* = −0.57). That is the signature of a "
           "small-sample artefact rather than a mechanism."),
        md("## Non-tape inputs — both PROXIES, both swept\n\n"
           "**(1) Opening-auction penalty.** The headline charges both venues the same 2 bps, "
           "which flatters the open (the opening auction is the wider, thinner book). We sweep an "
           "*extra* one-way charge on the open arm only. **(2) Cash split.** BIL's printed open is "
           "sub-penny noise, so the daily T-bill accrual is split night/day by an assumed "
           "fraction (default 0.73)."),
        code(
            "print('opening-auction penalty (PROXY) — extra one-way bps on the open arm only:')\n"
            "print(f\"  {'extra':>6s} {'10-month gap (t)':>22s} {'20-week gap (t)':>22s}\")\n"
            "for x, (mm, ww) in sorted(R['pen'].items()):\n"
            "    print(f\"  {x:6.1f} {mm[0]:+13.1f} ({mm[1]:+.2f}) {ww[0]:+13.1f} ({ww[1]:+.2f})\")\n"
            "print('\\ncash-split PROXY (SPY, 10-month) — overnight share of the T-bill accrual:')\n"
            "for f, g in sorted(R['cash_sweep'].items()):\n"
            "    print(f\"  f={f:.2f} -> gap {g:+.2f} bps/yr\")\n"
            "print('  -> inert, as a ~1 bp/day accrual must be.')"
        ),
        md("Both rules move toward the **close** as the opening spread is charged more honestly. "
           "That is the only monotone, sign-stable statement in the study — and it is a cost "
           "argument, not an alpha one."),
        md("## Dispersion — the cost of an arbitrary choice"),
        code(
            "print(f\"across the 8 rule x tape cells:\")\n"
            "print(f\"  mean {R['disp_mean']:+.1f} bps/yr, sd {R['disp_sd']:.1f}, \"\n"
            "      f\"range [{R['disp_lo']:+.1f}, {R['disp_hi']:+.1f}]\")\n"
            "print(f\"  |t| >= 2 in {R['disp_fires']}/8 cells (chance would give ~0.4)\")\n"
            "print(f\"  -> ~+/-{R['disp_sd']/100:.1f} pp/yr of unforecastable track-record luck\")"
        ),
        md("## Live synthetic control (offline — NOT the real tape)\n\n"
           "The generator fixes the close-to-close path and only re-splits it between the "
           "overnight and intraday legs, so the close-executed arm is bit-identical across the "
           "knob. `signal_strength=1` plants intraday continuation after strong/weak months "
           "(the open fill must win); `signal_strength=0` randomises the split (the venues must "
           "tie)."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..', '..', '..')))\n"
            "import numpy as np\n"
            "from open_close_exec import data, strategy as st\n"
            "for ss, tag in [(1.0, 'planted continuation'), (0.0, 'null')]:\n"
            "    rows = [st.synthetic_detect(\n"
            "                data.synthetic_daily(signal_strength=ss, seed=938 + 11 * s)[0])\n"
            "            for s in range(8)]\n"
            "    g = np.array([r['gap_ann_bps'] for r in rows])\n"
            "    t = np.array([r['gap_t_hac'] for r in rows])\n"
            "    print(f\"SYNTHETIC {tag:22s}: gap {g.mean():+7.1f} bps/yr (sd {g.std(ddof=1):.1f}), \"\n"
            "          f\"mean HAC t {t.mean():+.2f}, fires {int((abs(t) >= 2).sum())}/8\")\n"
            "print('\\n(frozen reference from docs/results.md: planted %+.1f / t %+.2f / %d of 8;'\n"
            "      ' null %+.1f / t %+.2f / %d of 8)'\n"
            "      % (R['syn_planted'][0], R['syn_planted'][2], R['syn_planted'][3],\n"
            "         R['syn_null'][0], R['syn_null'][2], R['syn_null'][3]))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** Pooled over {R['n_trades']} fills (on {R['n_dates']} distinct "
           f"dates) the open-minus-close slippage is **{R['per_trade_bps']:+.2f} bps**, *t* = "
           f"{R['per_trade_t']:+.2f} naive and **{R['per_trade_t_clu']:+.2f} clustered on the "
           f"trade date**; win rate **{R['win_rate']:.1f}%** with 95% Wilson "
           f"[{R['wilson'][0]:.1f}%, {R['wilson'][1]:.1f}%] — cluster-bootstrap "
           f"[{R['clu_win'][0]:.1f}%, {R['clu_win'][1]:.1f}%]. The 10-month rule says "
           f"{R['m_gap']:+.1f} bps/yr (*t* = {R['m_t']:+.2f}), the higher-powered 20-week rule says "
           f"{R['w_gap']:+.1f} (*t* = {R['w_t']:+.2f}); both bootstrap CIs contain zero; the era cut "
           f"reverses; and the one significant cell's mechanism (EEM's entry-minus-exit intraday "
           f"spread) flips sign under more data. The synthetic control fires "
           f"{R['syn_planted'][3]}/8 on a planted venue edge and {R['syn_null'][3]}/8 on the null, "
           f"so the harness is awake. *Survivorship: none — four continuously listed index ETFs; "
           f"intraday legs are price-only, overnight legs carry the dividend, labelled throughout.*\n"
           f"- **Tradability — Mirage.** No committable sign, hence no venue rule. The residual "
           f"is dispersion — realised gaps spanning "
           f"[{R['disp_lo']:+.1f}, {R['disp_hi']:+.1f}] bps/yr across the eight cells, ~±0.4 pp/yr "
           f"of free luck. The one robust statement is the friction one: charge the opening "
           f"auction its real spread and both rules prefer the **close**."),
    ]
    nb["cells"] = cells
    return nb


def main():
    for name, nb in [("01_for_the_curious", build_curious()),
                     ("02_for_the_quants", build_quants())]:
        path = os.path.join(HERE, f"{name}.ipynb")
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("wrote", path)


if __name__ == "__main__":
    main()
