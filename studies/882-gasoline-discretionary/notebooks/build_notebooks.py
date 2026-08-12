"""Generate the two narrative notebooks for Study 882 (Gas-Price → Discretionary).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from the
frozen ``R`` dict (mirroring docs/results.md); the live cells run only the fast synthetic
positive control, so execution is quick and network-free.
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
# close for RB=F + XLY + XLP + XLE + SPY, 2005-01-03 -> 2026-06-30; month-end resample;
# predictive regression of the XLY-XLP forward-1M spread on the trailing-1M gas return).
# Fingerprint(XLY)=c131f45ae347.
R = dict(
    start="2005-02-28", end="2026-05-31", n_months=256, gas="RB=F",
    fingerprint="c131f45ae347",
    beta=-0.0171, t_nw=-0.71, t_ols=-0.79, r2_pct=0.25, alpha_pct=0.26,
    fwd_down_pct=0.19, fwd_up_pct=-0.08, welch_t=-0.40,
    enr_beta=-0.0045, enr_t=-0.11, enr_r2=0.01,
    enr_fwd_down_pct=0.21, enr_fwd_up_pct=0.56, enr_welch_t=0.35,
    placebo_obs=-0.0171, placebo_mean=0.0004, placebo_sd=0.0218, placebo_p=0.431,
    placebo_draws=2000,
    era1_n=131, era1_beta=-0.0054, era1_t=-0.16, era1_r2=0.03,
    era2_n=125, era2_beta=-0.0289, era2_t=-0.88, era2_r2=0.57,
    ls1_gross=0.089, ls1_net=0.028, ls1_t=0.10, ls1_sharpe=0.02, ls1_ann=0.3,
    ls5_net=-0.046, ls5_t=-0.16, ls5_sharpe=-0.04, hit=0.504,
    null_mean_t=-0.61, null_sd_t=1.00, null_fire=1,
    planted_beta=-0.1294, planted_t=-5.92, planted_r2=10.44,
)


HEADER = f"""# Study 882 — Gas-Price → Discretionary ⛽

**Does *this month's* jump at the pump forecast *next month's* consumer-discretionary lagging staples — the "pump tax" rotation?**

A higher gasoline price is a regressive tax on the consumer's wallet: households burn a
near-fixed number of gallons, so dear gas drains the income that would otherwise buy autos,
apparel, travel and restaurants (**XLY**) while sparing food and household staples (**XLP**)
— and it lifts energy (**XLE**). The top-down trade is to rotate **out of discretionary into
staples** when gas rises. We take the self-contained monthly version — a predictive
regression of the **XLY − XLP** *forward* one-month spread on the *trailing* one-month
gasoline (**{R['gas']}**) return ({R['start']} → {R['end']}, {R['n_months']} months) — and
stamp it on the real tape.

*Numbers below are the frozen headline (`docs/results.md`, fingerprint `{R['fingerprint']}`);
the live cells run the fast synthetic control. Survivorship: RB=F/XLY/XLP/XLE/SPY are
continuously-listed futures/ETFs — no delisting bias, named on the Signal axis.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Gasoline is a large, salient line item in the household budget. When it spikes, "
           "the classic macro reading is that consumers cut *discretionary* spending first "
           "(you still eat, but you skip the new sneakers) while *staples* hold up — and "
           "energy names cash in. If the market is *slow* to price that, this month's gas "
           "move should quietly drag on the XLY − XLP spread **next** month. The claim fixes "
           "the sign: the predictive slope should be **negative**."),
        code(
            "R = dict(beta=%r, t_nw=%r, r2_pct=%r, fwd_down_pct=%r, fwd_up_pct=%r)\n"
            "print('predictive slope beta = %%+.4f  (Newey-West t = %%+.2f,  R2 = %%.2f%%%%)'\n"
            "      %% (R['beta'], R['t_nw'], R['r2_pct']))\n"
            "print('  pump-tax predicts beta < 0; on the real tape it IS negative -- but tiny')\n"
            "print('  fwd XLY-XLP after gas FELL: %%+.2f%%%%   after gas ROSE: %%+.2f%%%%'\n"
            "      %% (R['fwd_down_pct'], R['fwd_up_pct']))"
            % (R["beta"], R["t_nw"], R["r2_pct"], R["fwd_down_pct"], R["fwd_up_pct"])
        ),
        md("## 2. Is the machinery honest? A live synthetic control\n\n"
           "We plant the pump-tax link in a seeded toy tape (`edge>0` → a *negative* "
           "gas→(XLY−XLP) slope, and a *positive* energy tilt) and check the regression "
           "recovers it with the right sign — and stays *silent* on the null (`edge=0`, gas "
           "and the sector spreads independent). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from gas_discretionary import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_series(edge=0.0, seed=882))\n"
            "planted = st.synthetic_detect(data.synthetic_series(edge=0.35, seed=882))\n"
            "print('null world   : slope t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: slope t = %+.2f  beta = %+.4f  (should light up NEGATIVE)'\n"
            "      % (planted['t_nw'], planted['beta']))"
        ),
        md(f"## 3. The honest verdict — the pump-tax edge does *not* replicate\n\n"
           f"On 2005–2026 US ETFs the predictive slope is **{R['beta']:+.4f}** with a "
           f"Newey-West *t* of just **{R['t_nw']:+.2f}** and an R² of **{R['r2_pct']:.2f}%**. "
           f"The point estimate has the *right* (negative) sign, and the tercile split even "
           f"leans the predicted way (fwd XLY−XLP {R['fwd_down_pct']:+.2f}% after cheap gas "
           f"vs {R['fwd_up_pct']:+.2f}% after dear gas) — but it is statistically "
           f"indistinguishable from zero. A 2,000-draw permutation placebo puts the observed "
           f"slope at p = {R['placebo_p']:.2f} (pure noise). The seeded synthetic control "
           f"recovers a *planted* slope cleanly, so this is a real null, not a broken engine. "
           f"**Signal: None.** And the spread timer is a coin flip that doesn't clear costs, "
           f"so **Tradability: Mirage.** A sensible story, an absent edge."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 882 — Gas-Price → Discretionary — the teardown\n\n"
           "The predictive-regression slope with a Newey-West HAC *t*, the R², the tercile "
           "cross-check, the parallel energy-tilt regression, the 2,000-permutation placebo, "
           "the two-era robustness cut, the costed spread timer, and the 20-seed synthetic "
           "control."),
        code("R = %r" % (R,)),
        md("## The headline — predictive regression  `r_(XLY-XLP)[t+1] = a + b·r_gas[t]`\n\n"
           "Monthly, one documented lag: gas return known at the close of month `t`, the "
           "XLY−XLP spread return realised over month `t+1`. Slope `b` is the pump-tax "
           "coefficient (expected < 0)."),
        code(
            "print(f\"slope beta   : {R['beta']:+.4f}   NW(6) t = {R['t_nw']:+.2f}   \"\n"
            "      f\"OLS t = {R['t_ols']:+.2f}   R2 = {R['r2_pct']:.2f}%\")\n"
            "print(f\"alpha        : {R['alpha_pct']:+.2f}%/mo   n = {R['n_months']} months\")\n"
            "print(f\"tercile check: fwd XLY-XLP after gas-down {R['fwd_down_pct']:+.2f}% vs \"\n"
            "      f\"after gas-up {R['fwd_up_pct']:+.2f}% (Welch t = {R['welch_t']:+.2f})\")\n"
            "print('  claim: beta < 0 (gas up -> discretionary lags staples). Found: right sign, but ~0.')"
        ),
        md("## The energy tilt — `r_(XLE-SPY)[t+1] = a + b·r_gas[t]`  (claim: b > 0)"),
        code(
            "print(f\"slope beta   : {R['enr_beta']:+.4f}   NW(6) t = {R['enr_t']:+.2f}   R2 = {R['enr_r2']:.2f}%\")\n"
            "print(f\"tercile check: fwd XLE-SPY after gas-down {R['enr_fwd_down_pct']:+.2f}% vs \"\n"
            "      f\"after gas-up {R['enr_fwd_up_pct']:+.2f}% (Welch t = {R['enr_welch_t']:+.2f})\")\n"
            "print('  the tercile leans right (energy beats after dear gas) but the slope is a flat zero.')"
        ),
        md("## Placebo — permute the target, keep the predictor (2,000 draws)"),
        code(
            "print(f\"observed beta {R['placebo_obs']:+.4f} vs placebo mean {R['placebo_mean']:+.4f} \"\n"
            "      f\"(sd {R['placebo_sd']:.4f}) -> two-sided p = {R['placebo_p']:.3f}\")"
        ),
        md("## Robustness — two eras (split 2016-01-01)"),
        code(
            "print(f\"2005-2015 (n={R['era1_n']}): beta {R['era1_beta']:+.4f}  NW t = {R['era1_t']:+.2f}  R2 = {R['era1_r2']:.2f}%\")\n"
            "print(f\"2016-2026 (n={R['era2_n']}): beta {R['era2_beta']:+.4f}  NW t = {R['era2_t']:+.2f}  R2 = {R['era2_r2']:.2f}%\")\n"
            "print('  the sign is at least stable (negative in both) -- but insignificant in both. A stable nothing.')"
        ),
        md("## The timer — can you get paid for it?\n\n"
           "Trade `-sign(gas_ret[t])` of the XLY−XLP spread next month (a 2x NAV long/short "
           "book); one-way cost × NAV per rebalance leg on both legs, 50 bps/yr borrow on the "
           "short leg."),
        code(
            "print(f\"spread 1bp: gross {R['ls1_gross']:+.3f}%/mo -> net {R['ls1_net']:+.3f}%/mo \"\n"
            "      f\"(t={R['ls1_t']:+.2f}, Sharpe {R['ls1_sharpe']:.2f}, ~{R['ls1_ann']:+.1f}%/yr, hit {R['hit']:.3f})\")\n"
            "print(f\"spread 5bp: net {R['ls5_net']:+.3f}%/mo (t={R['ls5_t']:+.2f}, Sharpe {R['ls5_sharpe']:.2f})\")\n"
            "print('  a coin flip (hit 0.504); flat-to-negative once you pay realistic two-leg costs.')"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted negative slope."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from gas_discretionary import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_series(edge=0.0, seed=882+s))['t_nw'] for s in range(20)])\n"
            "print(f\"null (edge=0), 20 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), \"\n"
            "      f\"|t|>=2 in {(abs(null_t)>=2).sum()}/20\")\n"
            "planted = st.synthetic_detect(data.synthetic_series(edge=0.35, seed=882))\n"
            "print(f\"planted (edge=0.35): beta = {planted['beta']:+.4f}, NW t = {planted['t_nw']:+.2f}, R2 = {planted['r2_pct']:.2f}%\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The pump-tax rotation does **not** replicate on 2005–2026 US "
           f"ETFs: the slope is **{R['beta']:+.4f}** (NW *t* = **{R['t_nw']:+.2f}**, "
           f"R² = **{R['r2_pct']:.2f}%**) — the *right* (negative) sign, but a flat, "
           f"insignificant nothing. It is p = {R['placebo_p']:.2f} in a 2,000-draw placebo and "
           f"stays tiny and insignificant across both eras (*t* = {R['era1_t']:+.2f} / "
           f"{R['era2_t']:+.2f}); the energy tilt is flatter still (*t* = {R['enr_t']:+.2f}). "
           f"The 20-seed synthetic control recovers a *planted* slope cleanly (*t* = "
           f"{R['planted_t']:+.2f}), so the flat real-tape result is a genuine null, not a bug.\n"
           f"- **Tradability — Mirage.** The spread timer is a coin flip (hit {R['hit']:.3f}): "
           f"gross {R['ls1_gross']:+.3f}%/mo, net {R['ls1_net']:+.3f}%/mo at 1 bp and "
           f"{R['ls5_net']:+.3f}%/mo once you pay realistic two-leg costs (Sharpe ≈ 0). No "
           f"paycheck here."),
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
