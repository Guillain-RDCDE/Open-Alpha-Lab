"""Generate the two narrative notebooks for Study 900 (Quality-Income).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from
the frozen ``R`` dict (mirroring docs/results.md); the live cells run only the fast
synthetic positive control, so execution is quick and network-free.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance total-return,
# SCHD+NOBL vs SPHD+VYM vs SPY, excess of BIL; common window 2013-11 -> 2026-06, 152 mo).
R = dict(
    start="2013-11", end="2026-06", n_months=152,
    q_cagr=11.00, q_vol=13.9, q_sharpe=0.697, q_maxdd=-22.4, q_wealth=3.75,
    y_cagr=10.13, y_vol=13.7, y_sharpe=0.650, y_maxdd=-27.5, y_wealth=3.40,
    spy_cagr=14.02, spy_vol=14.5, spy_sharpe=0.862, spy_maxdd=-23.9, spy_wealth=5.27,
    gap=0.047, diff_bps=6.7, diff_ann=0.81, t_1s=0.77, t_nw=0.57,
    ci_lo=-0.191, ci_hi=0.241, p_neg=0.355,
    era_e_gap=-0.007, era_e_diff=0.60, era_e_t=0.46, era_e_n=74,
    era_l_gap=0.060, era_l_diff=1.00, era_l_t=0.41, era_l_n=78,
    qspy_gap=-0.165, qspy_diff=-2.81, qspy_t=-1.45,
    yspy_gap=-0.212, yspy_diff=-3.62, yspy_t=-1.44,
    cost3_q=0.697, cost3_y=0.650, cost10_q=0.696, cost10_y=0.650,
    drag3=0.2, drag10=0.7, turn=0.6,
    null_mean_t=-0.40, null_sd_t=1.03, null_fire=1,
    planted_gap=0.505, planted_t=3.03, planted_diff=6.59,
    # calendar-year total returns (%), quality / yield / spy
    cal_years=[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    cal_q=[-4.4, 27.4, 11.7, 27.7, -4.9, 6.3, 9.2, 5.6],
    cal_y=[-6.0, 22.2, -4.7, 25.5, 0.1, 3.9, 17.9, 9.3],
    cal_spy=[-4.6, 31.2, 18.3, 28.7, -18.2, 26.2, 24.9, 17.7],
)


HEADER = f"""# Study 900 — Quality-Income 💎

**Does screening dividends for *quality* beat *chasing yield*?**

High dividend **yield** is a notorious value-trap magnet — the fattest yields often mark
distressed payers about to cut. **Quality**-dividend screens (durable, growing payers)
were sold as the fix. We race a **quality sleeve** (SCHD + NOBL) against a **raw
high-yield sleeve** (SPHD + VYM) and against **SPY**, all on monthly total returns, all
measured **excess of cash** (minus BIL), over the common window
{R['start']} → {R['end']} ({R['n_months']} months, NOBL-bound).

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Short-history / survivor caveat: young ETFs, one mostly-bull regime.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one line\n\n"
           "A stock can yield 9% because it's cheap and durable — or because the market "
           "knows the dividend is about to be cut. **Yield** screens can't tell the "
           "difference; **quality** screens (25 straight years of raises, strong cash "
           "flow) try to. If the value-trap story is real, the quality sleeve should ride "
           "*smoother* — especially through the years when traps blow up."),
        code(
            "R = %r\n"
            "print('QUALITY (SCHD+NOBL): CAGR %%5.2f%%%%  exSharpe %%.3f  maxDD %%.1f%%%%'\n"
            "      %% (R['q_cagr'], R['q_sharpe'], R['q_maxdd']))\n"
            "print('YIELD   (SPHD+VYM) : CAGR %%5.2f%%%%  exSharpe %%.3f  maxDD %%.1f%%%%'\n"
            "      %% (R['y_cagr'], R['y_sharpe'], R['y_maxdd']))\n"
            "print('SPY                : CAGR %%5.2f%%%%  exSharpe %%.3f  maxDD %%.1f%%%%'\n"
            "      %% (R['spy_cagr'], R['spy_sharpe'], R['spy_maxdd']))" % (R,)
        ),
        md("## 2. Where quality actually wins — the crisis years\n\n"
           "The whole thesis lives in the **stress years**, where yield-traps get "
           "punished. Look at 2020 (COVID) and 2022 (rate shock):"),
        code(
            "for yr, q, y, s in zip(R['cal_years'], R['cal_q'], R['cal_y'], R['cal_spy']):\n"
            "    star = '  <-- yield sleeve cratered' if (y < 0 and q > 5) else ''\n"
            "    print(f'{yr}:  quality {q:+6.1f}%   yield {y:+6.1f}%   SPY {s:+6.1f}%{star}')"
        ),
        md("In **2020** the quality sleeve returned **+11.7%** while the yield sleeve fell "
           "**−4.7%** (SPHD's high-yield names got crushed). That single dodge is why "
           "quality's worst drawdown (**−22.4%**) is ~5 points shallower than yield's "
           "(**−27.5%**). *But* yield's low-vol screen actually *won* 2022 and 2024 — it's "
           "a two-way trade."),
        md("## 3. Is it just luck? A live synthetic control\n\n"
           "We plant a real quality-over-yield edge in a seeded toy world (`edge>0`) and "
           "check the detector recovers it — and stays *silent* on the null (`edge=0`, no "
           "advantage). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from quality_income import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_world(n_months=150, edge=0.0, seed=900))\n"
            "planted = st.synthetic_detect(data.synthetic_world(n_months=150, edge=0.03, seed=900))\n"
            "print('null world   : gap NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: gap NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md(f"## 4. The honest verdict\n\n"
           f"On the real tape the quality sleeve edges the yield sleeve by only **+0.9 "
           f"pp/yr** of return and a Sharpe gap of **+{R['gap']:.3f}** — and that gap is "
           f"**not** statistically distinguishable from zero (HAC *t* = **+{R['t_nw']:.2f}**, "
           f"bootstrap 95% CI **[{R['ci_lo']:+.2f}, {R['ci_hi']:+.2f}]** straddles zero, "
           f"P(quality behind) = {R['p_neg']:.2f}). What quality *does* deliver is a real "
           f"**drawdown cushion** — dodging the yield-trap blowups — not a certified return "
           f"premium. And **both** dividend sleeves trailed plain **SPY** by ~3-4 pp/yr. "
           f"**Signal: Weak** (a real trap-avoidance profile, an insignificant Sharpe edge), "
           f"**Tradability: Fragile** (cheaply buyable, but a risk profile, not an edge)."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 900 — Quality-Income — the teardown\n\n"
           "The excess-of-cash Sharpe race, the quality-minus-yield HAC *t*, the paired "
           "block-bootstrap Sharpe-gap CI, the era cut, the vs-SPY races, the costed "
           "sleeves, and the 20-seed synthetic control."),
        code("R = %r" % (R,)),
        md("## The race — common window, excess of cash (minus BIL)"),
        code(
            "for name in ('q','y','spy'):\n"
            "    lbl = {'q':'QUALITY (SCHD+NOBL)','y':'YIELD  (SPHD+VYM) ','spy':'SPY               '}[name]\n"
            "    print(f\"{lbl}: CAGR {R[name+'_cagr']:5.2f}%  vol {R[name+'_vol']:4.1f}%  \"\n"
            "          f\"exSharpe {R[name+'_sharpe']:.3f}  maxDD {R[name+'_maxdd']:6.1f}%  \"\n"
            "          f\"$1->{R[name+'_wealth']:.2f}\")"
        ),
        md("## Headline — quality vs yield Sharpe gap + HAC t on the monthly difference\n\n"
           "The quality−yield monthly spread is cash-independent (cash cancels), so its "
           "mean / HAC *t* is the clean 'does quality out-earn yield?' statistic."),
        code(
            "print(f\"excess Sharpe : quality {R['q_sharpe']:.3f}  vs yield {R['y_sharpe']:.3f}  \"\n"
            "      f\"-> GAP {R['gap']:+.3f}\")\n"
            "print(f\"q-minus-y     : {R['diff_bps']:+.1f} bps/mo ({R['diff_ann']:+.2f}%/yr)  \"\n"
            "      f\"one-sample t {R['t_1s']:+.2f}  NW t {R['t_nw']:+.2f}\")\n"
            "print(f\"bootstrap gap : {R['gap']:+.3f}  95% CI [{R['ci_lo']:+.3f}, {R['ci_hi']:+.3f}]  \"\n"
            "      f\"P(gap<0) = {R['p_neg']:.3f}\")"
        ),
        md("The Sharpe advantage is **positive in sign but insignificant**: the CI "
           "straddles zero and one block-draw in three has quality behind."),
        md("## Era cut (split 2020-01-01) — no stable magnitude"),
        code(
            "print(f\"2013-11..2019-12 (n={R['era_e_n']}): gap {R['era_e_gap']:+.3f}  \"\n"
            "      f\"diff {R['era_e_diff']:+.2f}%/yr  NW t {R['era_e_t']:+.2f}\")\n"
            "print(f\"2020-01..2026-06 (n={R['era_l_n']}): gap {R['era_l_gap']:+.3f}  \"\n"
            "      f\"diff {R['era_l_diff']:+.2f}%/yr  NW t {R['era_l_t']:+.2f}\")"
        ),
        md("## Do the sleeves beat SPY? — both lag"),
        code(
            "print(f\"quality vs SPY: gap {R['qspy_gap']:+.3f}  diff {R['qspy_diff']:+.2f}%/yr  NW t {R['qspy_t']:+.2f}\")\n"
            "print(f\"yield   vs SPY: gap {R['yspy_gap']:+.3f}  diff {R['yspy_diff']:+.2f}%/yr  NW t {R['yspy_t']:+.2f}\")"
        ),
        md("## Costed — monthly rebalance turnover x one-way spread (long-only, no borrow)\n\n"
           "Turnover is only the drift back to 50/50 (~0.6%/mo), so even a fat spread costs "
           "under 1 bp/yr — costs do not move the verdict."),
        code(
            "print(f\" 3 bps/side: quality {R['q_sharpe']:.3f}->{R['cost3_q']:.3f}  \"\n"
            "      f\"yield {R['y_sharpe']:.3f}->{R['cost3_y']:.3f}  (drag {R['drag3']:.1f} bps/yr, turn {R['turn']:.1f}%/mo)\")\n"
            "print(f\"10 bps/side: quality {R['q_sharpe']:.3f}->{R['cost10_q']:.3f}  \"\n"
            "      f\"yield {R['y_sharpe']:.3f}->{R['cost10_y']:.3f}  (drag {R['drag10']:.1f} bps/yr)\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the gap detector must NOT fire on the null and must recover a planted "
           "quality premium."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from quality_income import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_world(n_months=150, edge=0.0, seed=900+s))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_world(n_months=150, edge=0.03, seed=900))\n"
            "print(f\"planted (edge=+3%/yr): gap {planted['sharpe_gap']:+.3f}  NW t {planted['t_nw']:+.2f}  diff {planted['diff_ann_pct']:+.2f}%/yr\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** Quality edges yield by +0.9 pp/yr and a Sharpe gap of "
           f"**+{R['gap']:.3f}**, with a genuinely shallower max drawdown "
           f"(**{R['q_maxdd']:.1f}%** vs **{R['y_maxdd']:.1f}%**) from dodging the 2020 "
           f"yield-sleeve crater — but the risk-adjusted edge is **not** significant "
           f"(NW *t* = **+{R['t_nw']:.2f}**, bootstrap CI **[{R['ci_lo']:+.2f}, "
           f"{R['ci_hi']:+.2f}]** straddles zero) and no era is significant. A real "
           f"trap-avoidance *profile*, not a certified premium. The synthetic control "
           f"recovers a *planted* edge cleanly (*t* = {R['planted_t']:.2f}, fires on "
           f"{R['null_fire']}/20 nulls), so a real premium *would* have shown — this one "
           f"didn't clear the bar. Short single-regime tape.\n"
           f"- **Tradability — Fragile.** The tilt is trivially buyable (cheap, liquid, "
           f"long-only, monthly rebalance costs < 1 bp/yr — not a Mirage) — but there is no "
           f"significant premium to bank; you buy a shallower-drawdown risk profile, not an "
           f"edge. And both dividend sleeves trailed plain SPY by ~3-4 pp/yr."),
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
