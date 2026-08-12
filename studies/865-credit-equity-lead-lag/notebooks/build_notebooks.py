"""Generate the two narrative notebooks for Study 865 (Credit → Equity Lead-Lag).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily
# total-return closes for HYG/IEF/LQD/SPY, 2007-05-01 -> 2026-06-30; weekly Granger-style
# regression of next-week SPY on the trailing k-week HYG-excess-IEF credit trend).
R = dict(
    start="2007-05-01", end="2026-06-30", n_daily=4822, n_weekly=1001,
    fingerprint="c6ce5a5566a2",
    # signal — 4-week (primary) predictive regression
    K=4, n_weeks=996, on_frac=0.582,
    beta=-0.0546, beta_t=-1.70, per_sd_bps=-19.6, r2=0.601, corr=-0.078,
    # cross-checks
    beta1=-0.1073, per_sd1=-19.5, t1=-1.58, r2_1=0.592,
    beta2=-0.0117, per_sd2=-2.9, t2=-0.23, r2_2=0.013,
    # discrimination (risk-on vs risk-off next-week SPY)
    on_bps=19.5, off_bps=27.0, diff_bps=-7.5, disc_t=-0.48, welch_t=-0.43,
    n_on=580, n_off=416,
    # placebo
    placebo_obs=-7.5, placebo_mean=-0.2, placebo_sd=16.2, placebo_p=0.663,
    placebo_sigma=-0.4, placebo_draws=1000,
    # eras
    era_early_beta=-0.0439, era_early_t=-1.23, era_early_n=501,
    era_late_beta=-0.0910, era_late_t=-1.33, era_late_n=495,
    # overlay vs buy-and-hold
    n_switches=218,
    t1_net_sharpe=0.644, t1_bh_sharpe=0.645, t1_net_cagr=7.35, t1_bh_cagr=10.61,
    t1_net_dd=-32.7, t1_bh_dd=-54.6, t1_active_bps=-7.6, t1_active_t=-1.22, t1_cost_yr=22.8,
    t5_net_sharpe=0.569, t5_net_cagr=6.37, t5_active_bps=-9.3, t5_active_t=-1.49,
    # synthetic control
    null_mean_t=-0.26, null_sd_t=0.75, null_fire=0,
    planted_t=26.7, planted_per_sd=988, planted_r2=76.0, planted_active_t=6.90,
    planted_net_sharpe=3.81, planted_bh_sharpe=0.19,
)


HEADER = f"""# Study 865 — Credit → Equity Lead-Lag 🔗

**Does high-yield credit really *turn before* stocks?**

Desk lore says "credit leads equity": high-yield credit, measured *duration-hedged* —
**HYG in excess of IEF** — is supposed to inflect a beat ahead of the stock market. The
self-contained, testable version is a **Granger-style lead-lag**: does the trailing 1-4-week
HY-excess return **predict the NEXT week's SPY return**? We take the weekly form on the four
ETFs ({R['start']} → {R['end']}, {R['n_daily']:,} daily rows → {R['n_weekly']:,} weekly
closes) and ask the two honest questions: does the credit trend **lead** the equity leg, and
can a **costed** SPY↔IEF overlay beat buy-and-hold?

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Distinct from study 115 (credit-spread **level** warning), 832 (HY
momentum as a daily sign-timer), 131 (utilities canary), 379 (generic ETF lead-lag).*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "When high-yield bonds out-earn duration-matched Treasuries, the market is paying "
           "*up* for credit risk — risk appetite is rising. The folk rule: because credit "
           "turns *first*, a strong trailing HY-excess week should foreshadow a *strong SPY "
           "week to come*. So we regress **next** week's SPY return on the trailing 4-week "
           "credit trend. If credit leads, the slope is positive. Is it?"),
        code(
            "R = dict(beta=%r, beta_t=%r, per_sd_bps=%r, r2=%r, on_bps=%r, off_bps=%r, diff_bps=%r, disc_t=%r, on_frac=%r)\n"
            "print('predictive regression  r_SPY[t+1] ~ trailing 4-week HY-excess trend[t]:')\n"
            "print('  slope           : %%+.4f  (SPY move %%+.1f bps per 1-sigma credit trend)' %% (R['beta'], R['per_sd_bps']))\n"
            "print('  Newey-West t     : %%+.2f   R2 = %%.2f%%%%' %% (R['beta_t'], R['r2']))\n"
            "print('  -> WRONG SIGN and insignificant: credit up does NOT foreshadow stocks up')\n"
            "print()\n"
            "print('risk-on weeks (trend>0, %%.0f%%%% of weeks): next-week SPY %%+.1f bps' %% (R['on_frac']*100, R['on_bps']))\n"
            "print('risk-off weeks                     : next-week SPY %%+.1f bps' %% R['off_bps'])\n"
            "print('difference (on - off)              : %%+.1f bps/week  (NW t = %%+.2f)' %% (R['diff_bps'], R['disc_t']))"
            % (R["beta"], R["beta_t"], R["per_sd_bps"], R["r2"], R["on_bps"], R["off_bps"],
               R["diff_bps"], R["disc_t"], R["on_frac"])
        ),
        md("## 2. Is the machinery even able to see a lead? A live synthetic control\n\n"
           "We plant a real one-week lead of credit over equity in a seeded toy world "
           "(`edge>0`: the risk-on factor drives SPY *five trading days later*) and check "
           "the detector recovers it — and stays *silent* on the null (`edge=0`, credit "
           "trend present but leads nothing). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from credit_lead import data, strategy as st\n"
            "null_t = np.array([st.leadlag_regression(data.synthetic_panel(edge=0.0, seed=865+s, n_days=3000), 4)['beta_t_nw'] for s in range(6)])\n"
            "planted = st.leadlag_regression(data.synthetic_panel(edge=0.02, seed=865, n_days=3000), 4)\n"
            "print('null worlds  : regression NW t mean %+.2f over 6 seeds  (should be ~0)' % null_t.mean())\n"
            "print('planted world: regression NW t = %+.2f  (should light up, positive)' % planted['beta_t_nw'])"
        ),
        md("## 3. The honest verdict — credit does *not* lead equity here\n\n"
           f"On the real tape the trailing credit trend predicts next-week SPY with the "
           f"**wrong sign**: a stronger credit week foreshadows a slightly *weaker* SPY week "
           f"(slope {R['beta']:+.4f}, i.e. {R['per_sd_bps']:+.1f} bps per 1σ trend, NW *t* = "
           f"**{R['beta_t']:+.2f}**), and the risk-on−risk-off next-week difference "
           f"({R['diff_bps']:+.1f} bps) sits just {R['placebo_sigma']:+.1f}σ inside a "
           f"label-shift placebo (p = {R['placebo_p']:.2f}). The sign is **consistently wrong** "
           f"across both eras. The costed SPY↔IEF overlay never beats buy-and-hold (net Sharpe "
           f"{R['t1_net_sharpe']:.3f} vs {R['t1_bh_sharpe']:.3f} at 1 bp, giving up "
           f"{R['t1_bh_cagr']-R['t1_net_cagr']:.1f}%/yr of return) — it only trims the drawdown "
           f"({R['t1_net_dd']:.0f}% vs {R['t1_bh_dd']:.0f}%). **Signal: None. Tradability: "
           f"Mirage.**"),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 865 — Credit → Equity Lead-Lag — the teardown\n\n"
           "The Granger-style predictive-regression Newey-West *t*, the risk-on/off "
           "discrimination, the 1,000-draw label-shift placebo, the two-era cut, the costed "
           "SPY↔IEF overlay vs buy-and-hold, and the 20-seed synthetic positive control."),
        code("R = %r" % (R,)),
        md("## The signal — does the trailing credit trend LEAD next-week SPY?\n\n"
           "Regress `r_SPY[t+1] = a + b·trend_k[t] + u`, trend known at the Friday close of "
           "week `t`. Credit leading equity means a **positive, robust** slope."),
        code(
            "print(f\"4-week (headline): slope {R['beta']:+.4f}  per-sigma {R['per_sd_bps']:+.1f} bps  \"\n"
            "      f\"NW(6) t = {R['beta_t']:+.2f}  R2 = {R['r2']:.2f}%%  corr = {R['corr']:+.3f}  (n={R['n_weeks']})\")\n"
            "print(f\"1-week cross-check: slope {R['beta1']:+.4f}  per-sigma {R['per_sd1']:+.1f} bps  NW t = {R['t1']:+.2f}  R2 = {R['r2_1']:.2f}%%\")\n"
            "print(f\"2-week cross-check: slope {R['beta2']:+.4f}  per-sigma {R['per_sd2']:+.1f} bps  NW t = {R['t2']:+.2f}  R2 = {R['r2_2']:.2f}%%\")\n"
            "print('-> WRONG SIGN at every horizon, |t| < 2 throughout, R2 <= 0.6%% (credit explains ~none of next-week stocks)')"
        ),
        md("## Discrimination — next-week SPY on risk-on (trend>0) vs risk-off weeks"),
        code(
            "print(f\"risk-on  ({R['on_frac']*100:.1f}%%, n={R['n_on']}): next-week SPY {R['on_bps']:+.1f} bps\")\n"
            "print(f\"risk-off (       n={R['n_off']}): next-week SPY {R['off_bps']:+.1f} bps\")\n"
            "print(f\"difference (on - off): {R['diff_bps']:+.1f} bps/week  NW(6) t = {R['disc_t']:+.2f}  Welch t = {R['welch_t']:+.2f}\")\n"
            "print('-> risk-on weeks earn LESS the following week: conditioning on credit adds no forward edge')"
        ),
        md("## Placebo — circular-shift the risk-on labels (1,000 draws)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.1f} bps vs placebo mean {R['placebo_mean']:+.1f} \"\n"
            "      f\"(sd {R['placebo_sd']:.1f}) -> {R['placebo_sigma']:+.1f}sigma, p = {R['placebo_p']:.3f}\")\n"
            "print('the observed difference sits INSIDE the shuffle cloud -> nothing to explain')"
        ),
        md("## Robustness — two eras (split 2017-01-01), regression slope"),
        code(
            "print(f\"2007-2016 (n={R['era_early_n']}): slope {R['era_early_beta']:+.4f}  NW t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2017-2026 (n={R['era_late_n']}): slope {R['era_late_beta']:+.4f}  NW t = {R['era_late_t']:+.2f}\")\n"
            "print('the slope is WRONG-SIGNED in both halves and never clears |t|>=2 -> a weak, persistent anti-lead, not a signal')"
        ),
        md("## The overlay — SPY when trend>0 else IEF, weekly, costed, vs 100%-SPY buy-and-hold\n\n"
           "A switch turns the book over (2 legs); long-only, no borrow. 218 switches over 19 years."),
        code(
            "for tag,ns,cg,ac,at in [('1 bp',R['t1_net_sharpe'],R['t1_net_cagr'],R['t1_active_bps'],R['t1_active_t']),\n"
            "                        ('5 bps',R['t5_net_sharpe'],R['t5_net_cagr'],R['t5_active_bps'],R['t5_active_t'])]:\n"
            "    print(f\"{tag:>5}/leg: net Sharpe {ns:.3f} (B&H {R['t1_bh_sharpe']:.3f}) | net CAGR {cg:+.2f}%% \"\n"
            "          f\"(B&H {R['t1_bh_cagr']:+.2f}%%) | active {ac:+.1f} bps/wk (NW t={at:+.2f})\")\n"
            "print(f\"drawdown: overlay {R['t1_net_dd']:.0f}%% vs B&H {R['t1_bh_dd']:.0f}%% (the ONLY real benefit)\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted one-week "
           "lead of credit over equity. No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from credit_lead import data, strategy as st\n"
            "null_t = np.array([st.leadlag_regression(data.synthetic_panel(edge=0.0, seed=865+s, n_days=3000), 4)['beta_t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: regression NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.02, seed=865, n_days=3000), 4)\n"
            "print(f\"planted (edge=0.02): regression NW t = {planted['beta_t_nw']:+.2f}, per-sigma {planted['per_sd_bps']:+.0f} bps, \"\n"
            "      f\"active NW t = {planted['active_t_nw']:+.2f}, overlay Sharpe {planted['net_sharpe']:+.2f} vs B&H {planted['bh_sharpe']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The credit trend does not lead next-week equity: the "
           f"predictive-regression slope is wrong-signed and insignificant at every horizon "
           f"(**{R['per_sd_bps']:+.1f} bps/σ**, NW *t* = **{R['beta_t']:+.2f}** at 4 weeks; "
           f"*t* = {R['t1']:+.2f} at 1 week; *t* = {R['t2']:+.2f} at 2 weeks), R² never exceeds "
           f"0.6%, the risk-on−risk-off difference sits {R['placebo_sigma']:+.1f}σ inside a "
           f"label-shift placebo (p = {R['placebo_p']:.2f}), and the wrong sign is consistent "
           f"across eras. The 20-seed synthetic control recovers a *planted* lead (*t* = "
           f"{R['planted_t']:+.1f}, fires on {R['null_fire']}/20 nulls), so the flat real-tape "
           f"result is a true absence.\n"
           f"- **Tradability — Mirage.** The costed SPY↔IEF overlay never beats buy-and-hold "
           f"(net Sharpe {R['t1_net_sharpe']:.3f} vs {R['t1_bh_sharpe']:.3f} at 1 bp, active NW "
           f"*t* = {R['t1_active_t']:+.2f}; it forfeits {R['t1_bh_cagr']-R['t1_net_cagr']:.1f}%/yr "
           f"of CAGR) and falls further behind by 5 bps. Its one merit — a {R['t1_net_dd']:.0f}% "
           f"vs {R['t1_bh_dd']:.0f}% drawdown — is insurance paid for in forgone return, not an "
           f"edge."),
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
