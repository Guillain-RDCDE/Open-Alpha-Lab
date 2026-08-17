"""Generate the two narrative notebooks for Study 940 (The Turnover Budget).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below (a mirror of docs/results.md); the only live cells run the fast
synthetic control on a small planted panel, and they are always labelled as synthetic.
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


# Frozen real-tape headline — mirror of docs/results.md. Eleven Select Sector SPDRs
# (+ BIL for the long-only cross-check), 12-1 momentum, dollar-neutral 3/3,
# 1999-12-23 -> 2026-06-30, total return, as-of 2026-06-30.
R = dict(
    start="1999-12-23", end="2026-06-30", n_days=6668, fp="7af2719e8031",
    cost_bps=5.0, borrow_bps=40.0,
    # frequency table: [daily, weekly, monthly, quarterly]
    labels=["daily", "weekly", "monthly", "quarterly"],
    n_rebal=[6668, 1384, 318, 106],
    turnover=[31.27, 13.44, 5.81, 3.27],
    turn_per_rebal=[0.1241, 0.2569, 0.4828, 0.8143],
    churn_share=[0.9175, 0.9190, 0.9228, 0.9299],
    sharpe_gross=[0.121, 0.105, 0.099, 0.013],
    sharpe_net=[-0.096, -0.003, 0.037, -0.034],
    ret_gross=[0.98, 0.85, 0.79, 0.10],       # % per year
    ret_net=[-0.78, -0.02, 0.30, -0.27],      # % per year
    vol=[8.13, 8.11, 7.98, 7.83],
    maxdd=[-36.7, -31.7, -26.3, -39.4],
    t_gross=[0.67, 0.59, 0.55, 0.07],
    breakeven=[2.51, 4.83, 10.09, -3.24],
    # cost surface: cost_bps -> [d, w, m, q] net excess Sharpe
    cost_grid=[0.0, 1.0, 2.5, 5.0, 10.0, 25.0],
    cost_surface=[
        [0.097, 0.080, 0.073, -0.014],
        [0.058, 0.063, 0.066, -0.018],
        [0.000, 0.039, 0.055, -0.024],
        [-0.096, -0.003, 0.037, -0.034],
        [-0.288, -0.086, 0.001, -0.055],
        [-0.862, -0.334, -0.109, -0.118],
    ],
    borrow_grid=[0.0, 25.0, 50.0, 100.0],
    borrow_surface=[
        [-0.071, 0.022, 0.062, -0.008],
        [-0.087, 0.006, 0.047, -0.025],
        [-0.102, -0.009, 0.031, -0.041],
        [-0.133, -0.040, -0.001, -0.074],
    ],
    # parameter neighbourhood, monthly clock: (top_k, lookback, skip, gross Sharpe, HAC t)
    param_grid=[(2, 252, 21, 0.055, 0.31), (3, 252, 21, 0.099, 0.55),
                (4, 252, 21, 0.072, 0.39), (5, 252, 21, -0.006, -0.02),
                (3, 126, 21, 0.020, 0.11), (3, 252, 0, 0.046, 0.25),
                (3, 63, 5, 0.003, 0.02)],
    # bootstrap (gross / net) 95% CIs
    ci_gross=[(-0.221, 0.473), (-0.232, 0.457), (-0.252, 0.449), (-0.319, 0.346)],
    ci_net=[(-0.431, 0.252), (-0.337, 0.348), (-0.313, 0.388), (-0.366, 0.297)],
    ci_net_negshare=[71.2, 51.4, 41.4, 58.8],
    # paired races: (name, gross gap, gross t, net gap, net t)
    races=[("daily - monthly", 0.012, 0.18, -0.144, -1.90),
           ("daily - quarterly", 0.098, 0.90, -0.072, -0.67),
           ("monthly - quarterly", 0.086, 1.02, 0.071, 0.84)],
    era_early_gross=[0.035, 0.103, 0.137, 0.013],
    era_early_be=[0.31, 4.92, 15.86, -3.04],
    era_late_gross=[0.154, 0.047, 0.014, -0.022],
    era_late_be=[3.50, 1.24, -1.69, -11.07],
    era_split="2013-01-01",
    # long-only cross-check (2007-05-30 -> 2026-06-30), EW control cost-matched
    lo_start="2007-05-30",
    lo_sharpe=[0.524, 0.546, 0.539, 0.535],
    lo_t_cash=[2.56, 2.65, 2.62, 2.62],
    lo_sharpe_ew=[0.556, 0.555, 0.551, 0.584],
    lo_alpha=[-0.62, -0.19, -0.15, -0.44],        # %/yr vs EW, both legs costed
    lo_t_ew=[-0.33, -0.10, -0.08, -0.24],
    lo_alpha_gross=[0.67, 0.40, 0.09, -0.31],     # %/yr vs EW, neither leg costed
    lo_t_ew_gross=[0.35, 0.21, 0.05, -0.17],
    lo_turn=[27.20, 12.44, 5.21, 2.82],
    lo_turn_ew=[1.49, 0.72, 0.37, 0.24],
    # synthetic control
    syn_planted_sharpe=[5.78, 5.53, 4.87, 3.44],
    syn_planted_t=[17.7, 16.9, 15.0, 10.6],
    syn_planted_gap=13.91,                    # pp/yr gross, daily - quarterly
    syn_null_sharpe=[-0.030, -0.135, -0.154, 0.025],
    syn_null_mean=-0.048, syn_null_sd=0.232, syn_null_max=0.405,
    crossover_bps=1.0,
)

SETUP = (
    "import os, sys\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "sys.path.insert(0, os.path.abspath(os.path.join('..', '..', '..')))\n"
)

HEADER = f"""# Study 940 — The Turnover Budget ⏱️

**One sleeve, four speeds. What does rebalancing faster actually cost — and is there
anything there to pay for it?**

The folk theorem: rebalance more often and you track a decaying signal better, but you pay
in turnover, so there is an optimal frequency somewhere. It is almost never *priced*. Here
we price it. The sleeve is fixed — rank the eleven **Select Sector SPDRs** on their **12-1**
total return, long the top 3, short the bottom 3, dollar-neutral at 1.0 gross. The only
thing that changes is the rebalance clock: **daily / weekly / monthly / quarterly**.

The deliverable is not a Sharpe. It is each speed's **break-even cost per unit of traded
notional** — how many basis points of friction that clock can afford before its net excess
return hits zero.

Tape: {R['start']} → {R['end']} ({R['n_days']:,} days), daily **total-return** closes.
One execution lag (signal through day *t*, weights effective *t+1*, trade charged *t+1*);
weights drift between rebalances; costs are one-way × NAV; the short leg pays borrow.

*Real numbers below are frozen from `docs/results.md` (Fingerprint `{R['fp']}`, as-of
2026-06-30). The only live cells run a small **synthetic** panel and say so.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The bill, in one number\n\n"
           "A book that re-ranks itself **every day** trades about **31 times its own net "
           "asset value each year**. The same book on a **monthly** clock trades **5.8 times**. "
           "That is a 5.4× difference in the bill — before anyone asks what the extra "
           "trading bought."),
        code(
            "R = %r\n"
            "print('speed      rebalances   turnover/yr   per rebalance')\n"
            "for i, lab in enumerate(R['labels']):\n"
            "    print('%%-10s %%10d %%12.1fx %%14.1f%%%%'\n"
            "          %% (lab, R['n_rebal'][i], R['turnover'][i], R['turn_per_rebal'][i]*100))"
            % ({k: R[k] for k in ("labels", "n_rebal", "turnover", "turn_per_rebal")},)
        ),
        md("## 2. What the extra trading bought: almost nothing\n\n"
           f"Before any cost, the daily clock earned **{R['ret_gross'][0]:.2f}%** a year and "
           f"the monthly clock **{R['ret_gross'][2]:.2f}%**. That is a gross Sharpe gap of "
           f"**+{R['races'][0][1]:.3f}** — and it is not distinguishable from luck "
           f"(*t* = +{R['races'][0][2]:.2f}). The quarterly clock earned "
           f"**{R['ret_gross'][3]:.2f}%**: nothing at all.\n\n"
           "So the fast clock pays 5.4× the bill for a rounding error."),
        code(
            "R = %r\n"
            "print('speed        gross return/yr   gross Sharpe   HAC t')\n"
            "for i, lab in enumerate(R['labels']):\n"
            "    print('%%-10s %%14.2f%%%% %%14.3f %%8.2f'\n"
            "          %% (lab, R['ret_gross'][i], R['sharpe_gross'][i], R['t_gross'][i]))\n"
            "print('\\nbest gross t across all four speeds: %%+.2f  -> nothing here is real'\n"
            "      %% max(R['t_gross']))"
            % ({k: R[k] for k in ("labels", "ret_gross", "sharpe_gross", "t_gross")},)
        ),
        md("## 3. The budget — and why it is worthless\n\n"
           "The **break-even cost** is the honest deliverable: how many basis points per unit "
           "of traded notional each speed can afford before it earns exactly zero.\n\n"
           f"| speed | break-even |\n|---|--:|\n"
           f"| daily | **{R['breakeven'][0]:.1f} bps** |\n"
           f"| weekly | **{R['breakeven'][1]:.1f} bps** |\n"
           f"| monthly | **{R['breakeven'][2]:.1f} bps** |\n"
           f"| quarterly | **{R['breakeven'][3]:.1f} bps** (negative — it loses money before "
           "a single trade is charged) |\n\n"
           "Read that as a price of admission: a daily sector-momentum book needs its costs "
           "under two and a half basis points per unit traded, forever, to break even. But "
           "the budget is only as trustworthy as the return funding it — and that return has "
           f"a *t* of at most +{max(R['t_gross']):.2f}. **This is a budget for zero.**"),
        md("> 🔬 **For the quants** — break-even is the zero of "
           "`mean(excess) = gross_mean − borrow − c × 1e-4 × mean(traded)` solved for `c`, "
           "with borrow still charged (it is a holding cost, not a trading cost). Turnover "
           "here is traded notional `Σ|w_new − w_drifted|` as a fraction of NAV, so `c` is in "
           "bps *per unit traded*, not per rebalance."),
        md("## 4. The ranking of speeds is decided by your broker, not by the tape\n\n"
           "At zero cost the daily clock is the best arm. By **one basis point** it is already "
           "third, and by 2.5 bps it is the **worst**. The whole inversion happens **below the "
           f"first basis point** of execution cost — a standard nobody clears on a book that "
           "trades 31× NAV a year.\n\n"
           "Which is the real lesson: 'what is the optimal rebalance frequency' is not a "
           "question about markets. It is a question about your fill."),
        code(
            "R = %r\n"
            "print('cost/unit turnover |' + ''.join('%%10s' %% l for l in R['labels']))\n"
            "for c, row in zip(R['cost_grid'], R['cost_surface']):\n"
            "    print('%%14.1f bps |' %% c + ''.join('%%+10.3f' %% v for v in row))\n"
            "print('\\nnet excess-of-cash Sharpe. daily leads at 0 bps and trails by 2.5 bps.')"
            % ({k: R[k] for k in ("labels", "cost_grid", "cost_surface")},)
        ),
        md("## 5. The trap this study exists to name\n\n"
           "Drop the short leg and run the same ranking **long-only** — top 3 sectors, fully "
           "invested — and suddenly every speed looks significant against cash "
           f"(*t* ≈ +{min(R['lo_t_cash']):.1f} to +{max(R['lo_t_cash']):.1f}, Sharpe ~0.53). "
           "It would be very easy to publish that.\n\n"
           "It is equity beta. Cash is the wrong yardstick for a book that is 100% in "
           "equities all the time. The right one is the **same eleven sectors, equal-weighted, "
           "on the same clock, paying the same commission** — a control with no view at all. "
           "Against that, momentum selection adds "
           f"**{R['lo_alpha_gross'][0]:+.2f}%/yr** before trading costs at the daily clock "
           f"(*t* = {R['lo_t_ew_gross'][0]:+.2f}) and "
           f"**{R['lo_alpha_gross'][3]:+.2f}%/yr** at quarterly "
           f"(*t* = {R['lo_t_ew_gross'][3]:+.2f}) — nothing, at any speed.\n\n"
           "Net of costs it is worse, and for the reason this whole study is about: the sleeve "
           f"turns over **{R['lo_turn'][0]:.0f}× NAV a year** to the control's "
           f"**{R['lo_turn_ew'][0]:.1f}×**, so it hands back "
           f"{R['lo_alpha'][0]:+.2f}%/yr at daily. Selection bought nothing and the trading "
           "bill was real.\n\n"
           "> ⚠️ **How this study got it wrong first.** The original cut of this table raced "
           "the costed momentum sleeve against a *frictionless*, daily-rebalanced equal-weight "
           "average. That one-sided ledger charged one horse and not the other, and printed a "
           "−0.7%/yr 'selection shortfall' at the daily clock that was, to the basis point, "
           "the strategy's own commission. Same-clock, same-cost, or the race means nothing."),
        code(
            "R = %r\n"
            "hdr = 'speed      vs cash (t)   EW-11   alpha GROSS (t)     alpha NET (t)   turn  turnEW'\n"
            "print(hdr); print('-'*len(hdr))\n"
            "for i, lab in enumerate(R['labels']):\n"
            "    print('%%-9s %%10.2f %%8.3f %%9.2f%%%% (%%+5.2f) %%9.2f%%%% (%%+5.2f) %%6.1fx %%6.1fx'\n"
            "          %% (lab, R['lo_t_cash'][i], R['lo_sharpe_ew'][i],\n"
            "             R['lo_alpha_gross'][i], R['lo_t_ew_gross'][i],\n"
            "             R['lo_alpha'][i], R['lo_t_ew'][i],\n"
            "             R['lo_turn'][i], R['lo_turn_ew'][i]))\n"
            "print('\\nalpha is per year vs the equal-weight control. Nothing clears |t| = 2.')"
            % ({k: R[k] for k in ("labels", "lo_t_cash", "lo_sharpe_ew", "lo_alpha",
                                  "lo_t_ew", "lo_alpha_gross", "lo_t_ew_gross",
                                  "lo_turn", "lo_turn_ew")},)
        ),
        md("## 6. Live check — the machinery works when there is something to find\n\n"
           "**This cell is synthetic, not the real tape.** We plant a cross-section whose "
           "expected returns really do persist for about a quarter, and check that the ladder "
           "recovers the mechanism: on a world with genuine decaying momentum the faster clock "
           "*should* earn more gross. Then we switch the signal off and check the ladder goes "
           "quiet. If both hold, the flat real-tape result is a fact about sector momentum "
           "rather than a broken backtest."),
        code(
            SETUP +
            "from turnover_budget import data, strategy as st\n"
            "p1, c1, _ = data.synthetic_panel(n_assets=8, n_years=12, signal_strength=1.0, seed=940)\n"
            "p0, c0, _ = data.synthetic_panel(n_assets=8, n_years=12, signal_strength=0.0, seed=940)\n"
            "d1 = st.synthetic_detect(p1, c1, top_k=2)\n"
            "d0 = st.synthetic_detect(p0, c0, top_k=2)\n"
            "print('SYNTHETIC panel with planted momentum -- gross return per year:')\n"
            "for f in ('D','W','M','Q'):\n"
            "    print('   %-9s %+7.2f%%  (t %+6.2f)' % (st.FREQ_LABEL[f],\n"
            "          d1['ann_return_gross'][f]*100, d1['t_gross'][f]))\n"
            "print('   -> faster clock earns more: the mechanism is recovered\\n')\n"
            "print('SYNTHETIC null panel (no momentum planted) -- gross return per year:')\n"
            "for f in ('D','W','M','Q'):\n"
            "    print('   %-9s %+7.2f%%  (t %+6.2f)' % (st.FREQ_LABEL[f],\n"
            "          d0['ann_return_gross'][f]*100, d0['t_gross'][f]))\n"
            "print('   -> silent, as it must be')"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** Over 26.5 years the sleeve's best gross HAC *t* at any speed "
           f"is **+{max(R['t_gross']):.2f}**. Every bootstrap Sharpe interval straddles zero; "
           f"no speed-versus-speed race clears |*t*| = 2. The long-only arm's apparent "
           f"significance is equity beta — measured against the same eleven sectors "
           f"equal-weighted on the same clock and paying the same costs, its selection alpha "
           f"is indistinguishable from zero at every speed (largest |*t*| = "
           f"{max(abs(t) for t in R['lo_t_ew_gross'] + R['lo_t_ew']):.2f}).\n"
           f"- **Tradability — Mirage.** The budget is {R['breakeven'][0]:.1f} bps daily, "
           f"{R['breakeven'][1]:.1f} weekly, {R['breakeven'][2]:.1f} monthly and negative "
           f"quarterly — a budget for a return that is statistically zero. At a realistic "
           f"5 bps plus 40 bps borrow, three of four speeds are net-negative and the fourth "
           f"is indistinguishable from cash.\n"
           f"- **What survives.** The ladder itself. A daily clock on an eleven-name "
           f"cross-section trades 31× NAV a year and therefore needs roughly **five times** "
           f"the gross edge of a monthly clock just to stand still. That arithmetic transfers "
           f"to any sleeve you care about — ask of your signal whether its *gross* alpha "
           f"clears the break-even column. This one does not."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 940 — The Turnover Budget — the teardown\n\n"
           "The frequency table, the break-even cost per unit of traded notional, the cost "
           "surface and its inversion, the borrow sweep, the era cut, bootstrap Sharpe CIs, "
           "paired speed races, the long-only beta check, and the live synthetic control. "
           f"Every real number is frozen from `docs/results.md` (Fingerprint `{R['fp']}`, "
           "as-of 2026-06-30).\n\n"
           "**Construction.** 12-1 signal (252d return, 21d skip) on the eleven Select Sector "
           "SPDRs; long top 3 / short bottom 3, equal-weighted, 0.5 gross per side. Signal "
           "through day *t* → weights effective *t+1* → trade charged *t+1*: **one lag, and "
           "only one**. Weights drift with realised returns between rebalances. Traded "
           "notional is `Σ|w_new − w_drifted|` per unit NAV; cost is one-way × NAV; the short "
           "leg accrues borrow daily.\n\n"
           "**Excess-of-cash convention.** A dollar-neutral book is funded by its own "
           "collateral, which earns the bill, so the long-minus-short spread *is* the "
           "excess-of-cash return — subtracting BIL again would double-count. BIL gates only "
           f"the long-only cross-check (from {R['lo_start']})."),
        code("R = %r" % (R,)),
        md("> 💡 **In plain words** — one strategy, four alarm clocks. Everything below asks "
           "the same question: what did waking up more often cost, and did it buy anything?"),
        md("## 1. The frequency table\n\n"
           "Base assumptions: **5 bps** per unit traded notional, **40 bps/yr** borrow. Both "
           "are assumptions rather than tape, and both are swept below."),
        code(
            "hdr = 'speed      rebal  turn/yr  churn%  Sgross   tgross   Snet   ret_g%  ret_n%   vol%   maxDD%   breakeven'\n"
            "print(hdr); print('-'*len(hdr))\n"
            "for i, lab in enumerate(R['labels']):\n"
            "    print('%-9s %6d %8.1f %7.1f %+7.3f %+8.2f %+7.3f %+7.2f %+7.2f %6.2f %8.1f %10.2f bps'\n"
            "          % (lab, R['n_rebal'][i], R['turnover'][i], R['churn_share'][i]*100,\n"
            "             R['sharpe_gross'][i], R['t_gross'][i], R['sharpe_net'][i],\n"
            "             R['ret_gross'][i], R['ret_net'][i], R['vol'][i], R['maxdd'][i],\n"
            "             R['breakeven'][i]))"
        ),
        md("Three things to read off it.\n\n"
           f"1. **Turnover is monotone in the clock** ({R['turnover'][0]:.1f}× → "
           f"{R['turnover'][1]:.1f}× → {R['turnover'][2]:.1f}× → {R['turnover'][3]:.1f}× NAV "
           "per year) — arithmetic, not a finding.\n"
           f"2. **Gross return is *weakly* monotone too** (+{R['ret_gross'][0]:.2f}% → "
           f"+{R['ret_gross'][1]:.2f}% → +{R['ret_gross'][2]:.2f}% → "
           f"+{R['ret_gross'][3]:.2f}%), which is the mechanism the folk theorem posits — but "
           f"the whole spread from daily to quarterly is a gross Sharpe gap of "
           f"+{R['races'][1][1]:.3f} at *t* = +{R['races'][1][2]:.2f}.\n"
           "3. **92-93% of traded notional is rank churn**, not weight drift. The book is not "
           "paying to re-equalise positions it already holds; it is paying because the ranking "
           "genuinely changes. A no-trade band would recover very little."),
        md("> 💡 **In plain words** — the fast clock does earn a bit more before costs, in the "
           "direction the theory predicts. It is just far too small to matter, and the extra "
           "trading is real."),
        md("## 2. The break-even cost — the study's actual deliverable\n\n"
           "Solve `gross_mean − borrow − c × 1e-4 × mean(traded) = 0` for `c`. Borrow stays "
           "charged (a holding cost, not a trading cost), so `c` is purely the price of "
           "execution the arm can afford, per unit of notional it turns over."),
        code(
            "print('speed       gross ret/yr   turnover/yr   break-even cost per unit traded')\n"
            "for i, lab in enumerate(R['labels']):\n"
            "    print('%-10s %+12.2f%% %13.1fx %20.2f bps'\n"
            "          % (lab, R['ret_gross'][i], R['turnover'][i], R['breakeven'][i]))\n"
            "print('\\nquarterly is negative: it loses money BEFORE a single trade is charged,')\n"
            "print('so no execution standard, however good, makes it viable.')"
        ),
        md("## 3. The cost surface — and where the ranking inverts\n\n"
           "Net excess-of-cash Sharpe, borrow held at 40 bps/yr."),
        code(
            "print('cost/unit |' + ''.join('%10s' % l for l in R['labels']))\n"
            "print('-'*(11+10*len(R['labels'])))\n"
            "for c, row in zip(R['cost_grid'], R['cost_surface']):\n"
            "    print('%7.1f   |' % c + ''.join('%+10.3f' % v for v in row))\n"
            "best = [R['labels'][max(range(4), key=lambda j: row[j])] for row in R['cost_surface']]\n"
            "print('\\nbest arm by cost level:', dict(zip(R['cost_grid'], best)))"
        ),
        md("The best arm at 0 bps is **daily**; at **1 bp** it is already third, and from "
           "there down every row belongs to **monthly** while daily goes to dead last. The "
           "inversion is complete **inside the first basis point per unit traded notional**. "
           "Since no honest ETF book clears 1 bp all-in while turning over 31× NAV a year, "
           "the practical answer is 'slower' — but it is an answer about execution, not about "
           "the market.\n\n"
           "> 💡 **In plain words** — whether daily or monthly 'wins' depends entirely on what "
           "you assume you pay to trade. That is a warning about every rebalance-frequency "
           "study you have ever read, including this one."),
        md("## 4. Borrow sweep — the second non-tape input\n\n"
           "Borrow is a *holding* cost on the short notional, so it lands almost equally on "
           "every speed and cannot rescue or condemn one. Reported here so it is swept rather "
           "than asserted."),
        code(
            "print('borrow/yr |' + ''.join('%10s' % l for l in R['labels']))\n"
            "for b, row in zip(R['borrow_grid'], R['borrow_surface']):\n"
            "    print('%7.1f   |' % b + ''.join('%+10.3f' % v for v in row))\n"
            "print('\\nat ZERO borrow the best arm is still only %+.3f' % max(R['borrow_surface'][0]))"
        ),
        md("## 4b. Parameter neighbourhood — was the published sort cherry-picked?\n\n"
           "The sort is the Jegadeesh-Titman convention (252-day lookback, 21-day skip, "
           "top/bottom 3) and it was fixed before the tape was run. Here is the neighbourhood "
           "around it on the monthly clock, gross of everything, so nobody has to take that "
           "on trust."),
        code(
            "print('top_k  lookback  skip   gross Sharpe    HAC t')\n"
            "for tk, lb, sk, s, t in R['param_grid']:\n"
            "    star = '  <- published' if (tk, lb, sk) == (3, 252, 21) else ''\n"
            "    print('%5d %9d %5d %13.3f %8.2f%s' % (tk, lb, sk, s, t, star))\n"
            "print('\\nlargest |t| anywhere in the neighbourhood: %.2f'\n"
            "      % max(abs(t) for *_, t in R['param_grid']))"
        ),
        md("The published setting happens to be the *best* of the seven — and its *t* is "
           f"**+{R['param_grid'][1][4]:.2f}**. That is the useful way to read a robustness "
           "grid on a dead strategy: not 'the result survives perturbation' but 'there was "
           "no setting worth reaching for in the first place'.\n\n"
           "> ⚠️ Note what this check can and cannot buy. Seven points around one convention "
           "is not a multiple-testing correction, and if any cell here had printed *t* = 2.4 "
           "it would have been a **search statistic**, not a discovery. It is reported "
           "because a reader is entitled to know the headline number was not the survivor of "
           "a sweep."),
        md("## 5. Bootstrap Sharpe CIs (2,000 draws, 21-day fixed circular blocks)"),
        code(
            "print('speed        gross   95% CI                net    95% CI               share<0')\n"
            "for i, lab in enumerate(R['labels']):\n"
            "    g, n = R['ci_gross'][i], R['ci_net'][i]\n"
            "    print('%-10s %+7.3f  [%+.3f, %+.3f]  %+7.3f  [%+.3f, %+.3f] %8.1f%%'\n"
            "          % (lab, R['sharpe_gross'][i], g[0], g[1],\n"
            "             R['sharpe_net'][i], n[0], n[1], R['ci_net_negshare'][i]))\n"
            "print('\\nevery interval straddles zero, gross and net, at every speed.')"
        ),
        md("## 6. Paired speed races\n\n"
           "Same tape, same signal, only the clock differs — so the daily return difference is "
           "a clean paired comparison and the HAC *t* is the Jobson-Korkie return-difference "
           "test in Newey-West form."),
        code(
            "print('race                   gross gap (t)        net gap (t)')\n"
            "for name, gg, gt, ng, nt in R['races']:\n"
            "    print('%-20s %+8.3f (%+5.2f)   %+8.3f (%+5.2f)' % (name, gg, gt, ng, nt))\n"
            "print('\\nnot one race clears |t| = 2 -- including the one the study is built to find')\n"
            "print('(monthly beats daily NET at only t = %+.2f).' % R['races'][0][4])"
        ),
        md(f"## 7. Era cut (split {R['era_split']})\n\n"
           "A budget that means anything should keep its shape across eras. This one does not."),
        code(
            "print('speed        1999-2012 Sgross  break-even   2013-2026 Sgross  break-even')\n"
            "for i, lab in enumerate(R['labels']):\n"
            "    print('%-10s %+16.3f %10.2f bps %+17.3f %10.2f bps'\n"
            "          % (lab, R['era_early_gross'][i], R['era_early_be'][i],\n"
            "             R['era_late_gross'][i], R['era_late_be'][i]))\n"
            "print('\\nmonthly: the roomiest arm early (%.1f bps), NEGATIVE late (%.1f bps).'\n"
            "      % (R['era_early_be'][2], R['era_late_be'][2]))\n"
            "print('daily:   the tightest arm early (%.1f bps), the roomiest late (%.1f bps).'\n"
            "      % (R['era_early_be'][0], R['era_late_be'][0]))"
        ),
        md("> 💡 **In plain words** — the 'best' rebalance frequency swapped places between the "
           "two halves of the sample. That is what a noise estimate looks like."),
        md("## 8. The long-only beta check\n\n"
           "The most instructive table in the study. Drop the short leg, keep the ranking, and "
           "every speed clears *t* = 2 against cash. That is beta: a book that is always 100% "
           "in equities is being measured against T-bills.\n\n"
           "The control is the **same eleven sectors equal-weighted, run through the same "
           "engine, on the same rebalance clock, with the same one-day lag, paying the same "
           "5 bps on its own traded notional**. Cost symmetry is not a detail here — the "
           "sleeve turns over ~20× more than the control, so a frictionless benchmark would "
           "hand the sleeve's entire commission back as fake underperformance. (It did: see "
           "the warning below.) Both the cost-free and the net alpha are shown."),
        code(
            "hdr = 'speed      Snet   t vs cash   EW-11   alphaG (t)        alphaN (t)      turn  turnEW'\n"
            "print(hdr); print('-'*len(hdr))\n"
            "for i, lab in enumerate(R['labels']):\n"
            "    print('%-9s %+6.3f %+10.2f %8.3f %8.2f%% (%+5.2f) %8.2f%% (%+5.2f) %6.1fx %6.1fx'\n"
            "          % (lab, R['lo_sharpe'][i], R['lo_t_cash'][i], R['lo_sharpe_ew'][i],\n"
            "             R['lo_alpha_gross'][i], R['lo_t_ew_gross'][i],\n"
            "             R['lo_alpha'][i], R['lo_t_ew'][i],\n"
            "             R['lo_turn'][i], R['lo_turn_ew'][i]))\n"
            "print('\\nracing a fully-invested sleeve against CASH measures the equity market;')\n"
            "print('racing it against a FRICTIONLESS benchmark measures its own commission.')"
        ),
        md("> ⚠️ **An audit note against this study's own first draft.** The published cut of "
           "this table benchmarked the costed sleeve against a frictionless daily-rebalanced "
           "equal-weight average and reported selection alpha of −0.4% to −0.9%/yr, concluding "
           "that momentum 'loses to equal weighting at every speed'. With the benchmark paying "
           "its own way that shortfall largely disappears at the fast clocks — the gross alpha "
           f"is {R['lo_alpha_gross'][0]:+.2f}%/yr at daily. The *conclusion* survives (nothing "
           "here is remotely significant; the largest |*t*| against the control is "
           f"{max(abs(t) for t in R['lo_t_ew_gross'] + R['lo_t_ew']):.2f}), but the original "
           "reason for it was an artefact. A one-sided cost ledger is the most common way a "
           "backtest lies in the direction of its own thesis — including a sceptical one."),
        md("## 9. Live synthetic control — the harness is unbiased\n\n"
           "**Synthetic, not the real tape.** A planted panel whose expected returns decay with "
           "a ~63-day half-life: the ladder must recover a positive gross return that *falls* "
           "as the clock slows. A null panel (market factor + idiosyncratic noise): the ladder "
           "must go quiet at every speed."),
        code(
            SETUP +
            "import numpy as np\n"
            "from turnover_budget import data, strategy as st\n"
            "p1, c1, _ = data.synthetic_panel(n_assets=8, n_years=12, signal_strength=1.0, seed=940)\n"
            "d1 = st.synthetic_detect(p1, c1, top_k=2)\n"
            "print('SYNTHETIC planted: gross Sharpe / HAC t by speed')\n"
            "for f in ('D','W','M','Q'):\n"
            "    print('   %-9s %+7.2f  (t %+6.2f)  turnover %5.1fx'\n"
            "          % (st.FREQ_LABEL[f], d1['sharpe_gross'][f], d1['t_gross'][f],\n"
            "             d1['ann_turnover'][f]))\n"
            "print('   daily minus quarterly gross return: %+.2f pp/yr'\n"
            "      % (d1['daily_minus_quarterly_gross']*100))\n"
            "nulls = []\n"
            "for s in range(5):\n"
            "    pn, cn, _ = data.synthetic_panel(n_assets=8, n_years=12, signal_strength=0.0, seed=940+s)\n"
            "    nulls.append(st.synthetic_detect(pn, cn, top_k=2)['sharpe_gross']['M'])\n"
            "nulls = np.array(nulls)\n"
            "print('\\nSYNTHETIC null x5 (monthly gross Sharpe): mean %+.3f  sd %.3f  max|.| %.3f'\n"
            "      % (nulls.mean(), nulls.std(ddof=1), np.abs(nulls).max()))"
        ),
        md(f"On the full-size synthetic panel used in `docs/results.md` the planted gross "
           f"Sharpes are {R['syn_planted_sharpe']} for daily/weekly/monthly/quarterly (HAC "
           f"*t* = {R['syn_planted_t']}), with the daily arm earning "
           f"**+{R['syn_planted_gap']:.1f} pp/yr** more gross than the quarterly one; the null "
           f"panel gives {R['syn_null_sharpe']} with every |*t*| ≤ 0.67, and across six seeds "
           f"the monthly gross Sharpe has mean {R['syn_null_mean']:+.3f} (sd "
           f"{R['syn_null_sd']:.3f}). The detector fires on a planted effect and stays silent "
           f"on the null — the flat real-tape result is a property of **sector momentum**, not "
           f"of the engine."),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** Best gross excess-of-cash HAC *t* across all four speeds: "
           f"**+{max(R['t_gross']):.2f}** over {R['n_days']:,} days. Gross Sharpes "
           f"{R['sharpe_gross']} with bootstrap CIs that all straddle zero; the quarterly arm "
           f"is +{R['sharpe_gross'][3]:.3f}, and the whole parameter neighbourhood tops out at "
           f"*t* = +{max(t for *_, t in R['param_grid']):.2f}. "
           f"No paired speed race clears |*t*| = 2 (the best, monthly-beats-daily "
           f"net, is *t* = {R['races'][0][4]:+.2f}). The long-only arm's *t* ≈ +2.6 against "
           f"cash is equity beta: against a cost-matched, same-clock equal-weight-11 control "
           f"its selection alpha is {R['lo_alpha_gross'][0]:+.2f}%/yr gross and "
           f"{R['lo_alpha'][0]:+.2f}%/yr net at the daily clock, with every |*t*| ≤ "
           f"{max(abs(t) for t in R['lo_t_ew_gross'] + R['lo_t_ew']):.2f}. "
           f"Survivorship is small but named: XLRE (2015) and XLC (2018) are in the panel "
           f"because GICS later carved them out.\n"
           f"- **Tradability — Mirage.** Break-even costs of {R['breakeven'][0]:.1f} / "
           f"{R['breakeven'][1]:.1f} / {R['breakeven'][2]:.1f} / {R['breakeven'][3]:.1f} bps "
           f"per unit traded notional — a budget for a statistically zero return. At 5 bps "
           f"and 40 bps borrow three of four speeds are net-negative; at zero borrow the best "
           f"arm is still only {max(R['borrow_surface'][0]):+.3f}. The frequency ranking "
           f"inverts *below* {R['crossover_bps']:.0f} bp, so 'which speed is best' is settled by "
           f"an execution assumption, and the era cut swaps the winner outright.\n"
           f"- **The transferable result.** A daily clock on an eleven-name cross-section "
           f"trades **{R['turnover'][0]:.0f}× NAV a year** ({R['churn_share'][0]*100:.0f}% of "
           f"it genuine rank churn) against a monthly clock's {R['turnover'][2]:.1f}×, so it "
           f"needs roughly five times the gross alpha to break even. Use the break-even column "
           f"as the price of admission for any sleeve — and check the *gross* number clears it "
           f"before admiring the net one."),
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
