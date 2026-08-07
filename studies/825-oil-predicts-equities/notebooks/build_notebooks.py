"""Generate the two narrative notebooks for Study 825 (Oil Predicts Equities).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily adjusted
# close for USO + SPY, 2006-04-10 -> 2026-06-30; month-end resample; predictive regression
# of SPY forward-1M return on trailing-1M oil return). Fingerprint(SPY)=effe48bc1a4f.
R = dict(
    start="2006-05-31", end="2026-05-31", n_months=241, oil="USO", eq="SPY",
    fingerprint="effe48bc1a4f",
    beta=0.0136, t_nw=0.33, t_ols=0.52, r2_pct=0.11, alpha_pct=0.99,
    fwd_down_pct=0.95, fwd_up_pct=0.83, welch_t=-0.15,
    placebo_obs=0.0136, placebo_mean=0.0002, placebo_sd=0.0260, placebo_p=0.587,
    placebo_draws=2000,
    era1_n=116, era1_beta=0.0689, era1_t=1.23, era1_r2=2.15,
    era2_n=125, era2_beta=-0.0234, era2_t=-0.50, era2_r2=0.41,
    ls1_gross=0.200, ls1_net=0.170, ls1_t=0.58, ls1_sharpe=0.13, ls1_ann=2.0,
    ls5_net=0.132, ls5_t=0.45, hit=0.506,
    lf1_net=0.593, lf1_t=2.60, lf1_sharpe=0.58, lf1_ann=7.1,
    spy_bh=0.973, spy_bh_t=3.44, spy_bh_sharpe=0.77,
    null_mean_t=0.01, null_sd_t=1.03, null_fire=1,
    planted_beta=-0.1951, planted_t=-5.91, planted_r2=12.40,
)


HEADER = f"""# Study 825 — Oil Predicts Equities 🛢️📉

**Does *this month's* move in the oil price forecast *next month's* stock market — negatively?**

Driesprong, Jacobsen & Maat (2008), *"Striking Oil: Another Puzzle?"*, report that a rise
in the oil price this month is followed by **lower** equity returns next month: oil is a
slow-diffusing macro shock that the stock market prices in with a lag. We take the
self-contained monthly version — a predictive regression of the S&P 500 (**{R['eq']}**)
*forward* one-month return on the *trailing* one-month oil (**{R['oil']}**) return
({R['start']} → {R['end']}, {R['n_months']} months) — and stamp it on the real tape.

*Numbers below are the frozen headline (`docs/results.md`, fingerprint
`{R['fingerprint']}`); the live cells run the fast synthetic control. Survivorship: USO/SPY
are continuously-listed ETFs — no delisting bias on this pair, named on the Signal axis.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Oil is an input cost to almost every listed company and a barometer of global "
           "demand. Driesprong et al argue investors are *slow* to price an oil shock, so a "
           "rising oil price this month quietly drags on stocks **next** month. The test is "
           "a one-line forecast regression: today's oil return on the left of a one-month "
           "gap, tomorrow's equity return on the right. The claim fixes the sign: the slope "
           "should be **negative**."),
        code(
            "import numpy as np\n"
            "R = dict(beta=%r, t_nw=%r, r2_pct=%r, fwd_down_pct=%r, fwd_up_pct=%r)\n"
            "print('predictive slope beta = %%+.4f  (Newey-West t = %%+.2f,  R2 = %%.2f%%%%)'\n"
            "      %% (R['beta'], R['t_nw'], R['r2_pct']))\n"
            "print('  Driesprong predicts beta < 0; on the real tape it is ~0 and slightly POSITIVE')\n"
            "print('  next-month SPY after oil FELL: %%+.2f%%%%   after oil ROSE: %%+.2f%%%%'\n"
            "      %% (R['fwd_down_pct'], R['fwd_up_pct']))"
            % (R["beta"], R["t_nw"], R["r2_pct"], R["fwd_down_pct"], R["fwd_up_pct"])
        ),
        md("## 2. Is the machinery honest? A live synthetic control\n\n"
           "We plant the Driesprong link in a seeded toy tape (`edge>0` → a *negative* "
           "oil→equity slope) and check the regression recovers it with the right sign — and "
           "stays *silent* on the null (`edge=0`, oil and equities independent). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from oil_equities import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_series(edge=0.0, seed=825))\n"
            "planted = st.synthetic_detect(data.synthetic_series(edge=0.35, seed=825))\n"
            "print('null world   : slope t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: slope t = %+.2f  beta = %+.4f  (should light up NEGATIVE)'\n"
            "      % (planted['t_nw'], planted['beta']))"
        ),
        md(f"## 3. The honest verdict — the famous edge does *not* replicate\n\n"
           f"On 2006–2026 US ETFs the predictive slope is **{R['beta']:+.4f}** with a "
           f"Newey-West *t* of just **{R['t_nw']:+.2f}** and an R² of **{R['r2_pct']:.2f}%** — "
           f"statistically indistinguishable from zero, and if anything the *wrong* (positive) "
           f"sign versus the claimed negative one. Next-month SPY after the oil-up months "
           f"({R['fwd_up_pct']:+.2f}%) is barely different from after oil-down months "
           f"({R['fwd_down_pct']:+.2f}%). A 2,000-draw permutation placebo puts the observed "
           f"slope at p = {R['placebo_p']:.2f} (pure noise). The seeded synthetic control "
           f"recovers a *planted* negative slope cleanly, so this is a real null, not a broken "
           f"engine. **Signal: None.** And no version of the timer beats simply holding SPY, so "
           f"**Tradability: Mirage.**"),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 825 — Oil Predicts Equities — the teardown\n\n"
           "The predictive-regression slope with a Newey-West HAC *t*, the R², the "
           "tercile cross-check, the 2,000-permutation placebo, the two-era robustness cut, "
           "the costed monthly timer (vs a buy-&-hold benchmark), and the 20-seed synthetic "
           "control."),
        code("R = %r" % (R,)),
        md("## The headline — predictive regression  `r_equity[t+1] = a + b·r_oil[t]`\n\n"
           "Monthly, one documented lag: oil return known at the close of month `t`, equity "
           "return realised over month `t+1`. Slope `b` is the Driesprong coefficient."),
        code(
            "print(f\"slope beta   : {R['beta']:+.4f}   NW(6) t = {R['t_nw']:+.2f}   \"\n"
            "      f\"OLS t = {R['t_ols']:+.2f}   R2 = {R['r2_pct']:.2f}%\")\n"
            "print(f\"alpha        : {R['alpha_pct']:+.2f}%/mo   n = {R['n_months']} months\")\n"
            "print(f\"tercile check: fwd SPY after oil-down {R['fwd_down_pct']:+.2f}% vs \"\n"
            "      f\"after oil-up {R['fwd_up_pct']:+.2f}% (Welch t = {R['welch_t']:+.2f})\")\n"
            "print('  claim: beta < 0 (oil up -> stocks down next month). Found: beta ~ 0, wrong sign.')"
        ),
        md("## Placebo — permute the target, keep the predictor (2,000 draws)"),
        code(
            "print(f\"observed beta {R['placebo_obs']:+.4f} vs placebo mean {R['placebo_mean']:+.4f} \"\n"
            "      f\"(sd {R['placebo_sd']:.4f}) -> two-sided p = {R['placebo_p']:.3f}\")"
        ),
        md("## Robustness — two eras (split 2016-01-01)"),
        code(
            "print(f\"2006-2015 (n={R['era1_n']}): beta {R['era1_beta']:+.4f}  NW t = {R['era1_t']:+.2f}  R2 = {R['era1_r2']:.2f}%\")\n"
            "print(f\"2016-2026 (n={R['era2_n']}): beta {R['era2_beta']:+.4f}  NW t = {R['era2_t']:+.2f}  R2 = {R['era2_r2']:.2f}%\")\n"
            "print('  the slope even flips sign across eras -- no stable relation.')"
        ),
        md("## The timer — can you get paid for it?\n\n"
           "Trade `-sign(oil_ret[t])` of SPY next month; one-way cost × NAV per rebalance leg, "
           "50 bps/yr borrow on shorts. Compared against just buying and holding SPY."),
        code(
            "print(f\"long/short 1bp: gross {R['ls1_gross']:+.3f}%/mo -> net {R['ls1_net']:+.3f}%/mo \"\n"
            "      f\"(t={R['ls1_t']:+.2f}, Sharpe {R['ls1_sharpe']:.2f}, ~{R['ls1_ann']:+.1f}%/yr, hit {R['hit']:.3f})\")\n"
            "print(f\"long/short 5bp: net {R['ls5_net']:+.3f}%/mo (t={R['ls5_t']:+.2f})\")\n"
            "print(f\"long/flat  1bp: net {R['lf1_net']:+.3f}%/mo (t={R['lf1_t']:+.2f}, Sharpe {R['lf1_sharpe']:.2f})\")\n"
            "print(f\"SPY buy&hold  : {R['spy_bh']:+.3f}%/mo (t={R['spy_bh_t']:+.2f}, Sharpe {R['spy_bh_sharpe']:.2f})\")\n"
            "print('  the long/flat timer only \\'works\\' by inheriting the equity premium -- and still LOSES to buy&hold.')"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted negative slope."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from oil_equities import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_series(edge=0.0, seed=825+s))['t_nw'] for s in range(20)])\n"
            "print(f\"null (edge=0), 20 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), \"\n"
            "      f\"|t|>=2 in {(abs(null_t)>=2).sum()}/20\")\n"
            "planted = st.synthetic_detect(data.synthetic_series(edge=0.35, seed=825))\n"
            "print(f\"planted (edge=0.35): beta = {planted['beta']:+.4f}, NW t = {planted['t_nw']:+.2f}, R2 = {planted['r2_pct']:.2f}%\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The Driesprong negative oil→equity predictive slope does "
           f"**not** replicate on 2006–2026 US ETFs: the slope is **{R['beta']:+.4f}** "
           f"(NW *t* = **{R['t_nw']:+.2f}**, R² = **{R['r2_pct']:.2f}%**) — a flat, "
           f"wrong-signed nothing. It is p = {R['placebo_p']:.2f} in a 2,000-draw placebo and "
           f"flips sign across eras (*t* = {R['era1_t']:+.2f} / {R['era2_t']:+.2f}). The 20-seed "
           f"synthetic control recovers a *planted* negative slope cleanly (*t* = "
           f"{R['planted_t']:+.2f}, fires on {R['null_fire']}/20 nulls), so the flat real-tape "
           f"result is a genuine null, not a bug.\n"
           f"- **Tradability — Mirage.** The long/short timer is a coin-flip "
           f"(net {R['ls1_net']:+.3f}%/mo, *t* = {R['ls1_t']:+.2f}, hit {R['hit']:.3f}); the "
           f"long/flat variant looks positive (*t* = {R['lf1_t']:+.2f}) only because it inherits "
           f"the equity premium — and it still **loses to buying and holding SPY** "
           f"({R['spy_bh']:+.3f}%/mo, Sharpe {R['spy_bh_sharpe']:.2f} vs {R['lf1_sharpe']:.2f}). "
           f"No paycheck here."),
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
