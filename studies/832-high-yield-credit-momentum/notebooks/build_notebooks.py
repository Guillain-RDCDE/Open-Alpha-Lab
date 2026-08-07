"""Generate the two narrative notebooks for Study 832 (High-Yield Credit Momentum).

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
# total-return closes for HYG/IEF/LQD/SPY, 2007-05-01 -> 2026-06-30; 126d HYG-excess-IEF
# credit trend as an SPY<->IEF risk-on/off timer).
R = dict(
    start="2007-05-01", end="2026-06-30", n_rows=4822, fingerprint="3c027b47d765",
    # signal — 126d (6-month) primary
    L=126, n_days=4695, on_frac=0.636,
    on_bps=2.99, off_bps=4.60, diff_bps=-1.62, t_nw=-0.37, welch_t=-0.32,
    # signal — 63d (3-month) cross-check
    diff3_bps=-2.55, t3_nw=-0.61,
    # placebo
    placebo_obs=-1.62, placebo_mean=0.26, placebo_sd=4.67, placebo_p=0.646,
    placebo_sigma=-0.4, placebo_draws=1000,
    # eras
    era_early_diff=4.80, era_early_t=0.81, era_early_n=2310,
    era_late_diff=-12.27, era_late_t=-1.81, era_late_n=2385,
    # timer vs buy-and-hold
    n_switches=201, gross_sharpe=0.645,
    t1_net_sharpe=0.627, t1_bh_sharpe=0.619, t1_net_cagr=7.32, t1_bh_cagr=10.87,
    t1_net_dd=-41.3, t1_bh_dd=-54.7, t1_active_bps=-1.76, t1_active_t=-1.26, t1_cost_yr=21.6,
    t5_net_sharpe=0.558, t5_net_cagr=6.40, t5_active_bps=-2.10, t5_active_t=-1.50,
    # synthetic control
    null_mean_t=-0.16, null_sd_t=1.00, null_fire=1,
    planted_t=2.99, planted_welch=4.67, planted_active_t=3.08,
    planted_net_sharpe=0.52, planted_bh_sharpe=-1.01,
)


HEADER = f"""# Study 832 — High-Yield Credit Momentum 📉

**Does the *trend* of high-yield credit tell you when to be long stocks?**

High-yield credit is a risk asset; its recent total return is a popular risk-on/off gauge.
Measured *duration-hedged* — **HYG in excess of IEF** — a positive trailing trend is read
as risk appetite rising (be long SPY) and a negative trend as time to de-risk (hold IEF).
We take the self-contained daily version on the four ETFs ({R['start']} → {R['end']},
{R['n_rows']:,} rows) and ask the two honest questions: does the credit trend **predict**
the equity leg, and can a **costed** SPY↔IEF timer beat buy-and-hold?

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Distinct from study 115 (credit-spread **level** warning), 795
(cross-sectional bond momentum), 131 (utilities canary).*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "When high-yield bonds out-earn duration-matched Treasuries, the market is "
           "paying *up* for credit risk — risk appetite is rising. The folk rule: ride "
           "that into equities, and step aside into Treasuries when credit rolls over. "
           "It sounds like it should lead the stock market. Does it?"),
        code(
            "R = dict(diff_bps=%r, t_nw=%r, on_bps=%r, off_bps=%r, on_frac=%r)\n"
            "print('credit-trend risk-on days: SPY-over-IEF excess %%+.2f bps/day' %% R['on_bps'])\n"
            "print('        risk-off days     : SPY-over-IEF excess %%+.2f bps/day' %% R['off_bps'])\n"
            "print('difference (on - off)     : %%+.2f bps/day  (NW t = %%+.2f)' %% (R['diff_bps'], R['t_nw']))\n"
            "print('risk-on share of days     : %%.0f%%%%' %% (R['on_frac']*100))"
            % (R["diff_bps"], R["t_nw"], R["on_bps"], R["off_bps"], R["on_frac"])
        ),
        md("## 2. Is the machinery even able to see a signal? A live synthetic control\n\n"
           "We plant a real credit-times-equity effect in a seeded toy world (`edge>0`) "
           "and check the detector recovers it — and stays *silent* on the null "
           "(`edge=0`, credit trend present but not predictive). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from hy_credit_momentum import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=832+s, n_days=2200))['t_nw'] for s in range(6)])\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.006, seed=832, n_days=2200))\n"
            "print('null worlds  : discrimination NW t mean %+.2f over 6 seeds  (should be ~0)' % null_t.mean())\n"
            "print('planted world: discrimination NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — the credit trend does *not* time equities\n\n"
           f"On the real tape the risk-on days (positive credit trend) earned a SPY-over-IEF "
           f"excess of **{R['on_bps']:+.2f} bps/day** versus **{R['off_bps']:+.2f} bps/day** on "
           f"risk-off days — the *wrong* way round (difference **{R['diff_bps']:+.2f} bps/day**, "
           f"NW *t* = **{R['t_nw']:+.2f}**), and the 3-month trend agrees "
           f"({R['diff3_bps']:+.2f} bps, *t* = {R['t3_nw']:+.2f}). A label-shift placebo puts "
           f"the observed value at just {R['placebo_sigma']:+.1f}σ (p = {R['placebo_p']:.2f}), and "
           f"the sign **flips across eras** (+{R['era_early_diff']:.1f} bps early, "
           f"{R['era_late_diff']:+.1f} bps late). The costed SPY↔IEF timer never beats "
           f"buy-and-hold (net Sharpe {R['t1_net_sharpe']:.3f} vs {R['t1_bh_sharpe']:.3f} at 1 bp, "
           f"giving up {R['t1_bh_cagr']-R['t1_net_cagr']:.1f}%/yr of return) — it only trims the "
           f"drawdown ({R['t1_net_dd']:.0f}% vs {R['t1_bh_dd']:.0f}%). **Signal: None. "
           f"Tradability: Mirage.**"),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 832 — High-Yield Credit Momentum — the teardown\n\n"
           "The discrimination Newey-West *t*, the 1,000-draw label-shift placebo, the "
           "two-era cut, the costed SPY↔IEF timer vs buy-and-hold, and the 20-seed "
           "synthetic positive control."),
        code("R = %r" % (R,)),
        md("## The signal — does the credit trend discriminate the day-t equity excess?\n\n"
           "Split `xs = r_SPY − r_IEF` by the credit trend (HYG excess over IEF) known at "
           "the close of `t−1`. A positive trend should precede a *higher* `xs`."),
        code(
            "print(f\"126d (6mo): risk-on {R['on_bps']:+.2f} vs risk-off {R['off_bps']:+.2f} bps  \"\n"
            "      f\"-> diff {R['diff_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  Welch t = {R['welch_t']:+.2f}\")\n"
            "print(f\" 63d (3mo): diff {R['diff3_bps']:+.2f} bps/day  NW t = {R['t3_nw']:+.2f}\")\n"
            "print(f\"risk-on share: {R['on_frac']*100:.1f}%% of days (the timer is mostly long SPY anyway)\")"
        ),
        md("## Placebo — circular-shift the risk-on labels (1,000 draws)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.2f} \"\n"
            "      f\"(sd {R['placebo_sd']:.2f}) -> {R['placebo_sigma']:+.1f}sigma, p = {R['placebo_p']:.3f}\")\n"
            "print('the observed difference sits INSIDE the shuffle cloud -> nothing to explain')"
        ),
        md("## Robustness — two eras (split 2017-01-01)"),
        code(
            "print(f\"2007-2016 (n={R['era_early_n']}): diff {R['era_early_diff']:+.2f} bps  NW t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2017-2026 (n={R['era_late_n']}): diff {R['era_late_diff']:+.2f} bps  NW t = {R['era_late_t']:+.2f}\")\n"
            "print('the sign FLIPS across eras -> an unstable non-signal')"
        ),
        md("## The timer — SPY when trend>0 else IEF, costed, vs 100%-SPY buy-and-hold\n\n"
           "A switch turns the book over (2 legs); long-only, no borrow. 201 switches over 19 years."),
        code(
            "for tag,ns,cg,ac,at in [('1 bp',R['t1_net_sharpe'],R['t1_net_cagr'],R['t1_active_bps'],R['t1_active_t']),\n"
            "                        ('5 bps',R['t5_net_sharpe'],R['t5_net_cagr'],R['t5_active_bps'],R['t5_active_t'])]:\n"
            "    print(f\"{tag:>5}/leg: net Sharpe {ns:.3f} (B&H {R['t1_bh_sharpe']:.3f}) | net CAGR {cg:+.2f}%% \"\n"
            "          f\"(B&H {R['t1_bh_cagr']:+.2f}%%) | active {ac:+.2f} bps/day (NW t={at:+.2f})\")\n"
            "print(f\"drawdown: timer {R['t1_net_dd']:.0f}%% vs B&H {R['t1_bh_dd']:.0f}%% (the ONLY real benefit)\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted "
           "credit-times-equity relation. No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from hy_credit_momentum import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=832+s, n_days=2200))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.006, seed=832, n_days=2200))\n"
            "print(f\"planted (edge=0.006): discrimination NW t = {planted['t_nw']:+.2f}, active NW t = {planted['active_t_nw']:+.2f}, \"\n"
            "      f\"timer Sharpe {planted['net_sharpe']:+.2f} vs B&H {planted['bh_sharpe']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The credit trend does not time equity risk-on/off: the "
           f"risk-on−risk-off equity-excess difference is wrong-signed and negligible "
           f"(**{R['diff_bps']:+.2f} bps/day**, NW *t* = **{R['t_nw']:+.2f}** at 6 months; "
           f"{R['diff3_bps']:+.2f} bps, *t* = {R['t3_nw']:+.2f} at 3 months), sits "
           f"{R['placebo_sigma']:+.1f}σ inside a label-shift placebo (p = {R['placebo_p']:.2f}), and "
           f"flips sign across eras (+{R['era_early_diff']:.1f} → {R['era_late_diff']:+.1f} bps). "
           f"The 20-seed synthetic control recovers a *planted* effect (*t* = {R['planted_t']:+.2f}, "
           f"fires on {R['null_fire']}/20 nulls), so the flat real-tape result is a true absence.\n"
           f"- **Tradability — Mirage.** The costed SPY↔IEF timer never robustly beats "
           f"buy-and-hold (net Sharpe {R['t1_net_sharpe']:.3f} vs {R['t1_bh_sharpe']:.3f} at 1 bp, "
           f"active NW *t* = {R['t1_active_t']:+.2f}; it forfeits "
           f"{R['t1_bh_cagr']-R['t1_net_cagr']:.1f}%/yr of CAGR) and falls below B&H by 5 bps. "
           f"Its one merit — a {R['t1_net_dd']:.0f}% vs {R['t1_bh_dd']:.0f}% drawdown — is "
           f"insurance paid for in forgone return, not an edge."),
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
