"""Generate the two narrative notebooks for Study 894 (Trend Overlay on 60/40).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from
the frozen ``R`` dict (mirroring docs/results.md); the only live cell runs the fast
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily total-return
# SPY/IEF/AGG/BIL, 2007-05-30 -> 2026-06-30; 60/40 with a per-leg 200-day trend filter to BIL,
# excess of BIL, warm-up dropped -> 4,602-day window).
R = dict(
    start="2007-05-30", end="2026-06-30", n_days=4602, fingerprint="0dd2af7e1636",
    sharpe_strat=0.823, sharpe_bench=0.689, sharpe_adv=0.134,
    maxdd_strat=-12.5, maxdd_bench=-30.8, dd_cut=18.3,
    vol_strat=6.9, vol_bench=11.4, cagr_strat=6.94, cagr_bench=8.87,
    diff_bps=-0.87, t_nw=-1.18, t_1s=-0.96,
    boot_adv=0.134, boot_lo=-0.242, boot_hi=0.517, boot_ppos=0.744,
    boot_diff=-0.87, boot_diff_lo=-2.23, boot_diff_hi=0.59,
    era_early_adv=0.204, era_early_dd=21.3, era_early_n=2217,
    era_late_adv=0.071, era_late_dd=8.5, era_late_n=2385,
    y2008_ov=2.1, y2008_st=-13.2, y2022_ov=-8.5, y2022_st=-16.4,
    cost3_adv=0.104, cost5_adv=0.084,
    tax15_adv=-0.090, tax25_adv=-0.285, tax25_cagr=4.12,
    syn_plant_adv=0.31, syn_plant_dd=40.9, syn_plant_fire=11, syn_null_adv=0.04,
)


HEADER = f"""# Study 894 — Trend Overlay on 60/40 📉

**Does a 200-day trend filter cut the balanced book's drawdown *and* keep its return?**

Take the classic **60% SPY / 40% IEF** book and lay Faber's 200-day moving-average filter
over *each leg*: hold the equity sleeve while SPY is above its 200-day MA, the bond sleeve
while IEF is above its own — step whichever has rolled over to **BIL** cash. The pitch is a
free lunch: keep most of the 60/40's return while dodging its worst drawdowns
({R['start']} → {R['end']}, {R['n_days']:,} days, excess of cash).

*Numbers below are the frozen headline (`docs/results.md`, fingerprint `{R['fingerprint']}`);
the live cell runs the fast synthetic control. Short history: BIL (cash) launches 2007, so
this is a one-crash-each sample — named on the Signal axis.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "A 200-day moving average is a slow trend gauge: above it, the asset is in an "
           "uptrend; below it, a downtrend. Hold each leg only while it is *above* its own "
           "MA and park it in T-bills otherwise, and the book should step out of the way of "
           "long bear markets — 2008 for stocks, 2022 for bonds — the two episodes the "
           "static 60/40 is built to survive and doesn't."),
        code(
            "R = dict(sharpe_strat=%r, sharpe_bench=%r, sharpe_adv=%r,\n"
            "         maxdd_strat=%r, maxdd_bench=%r, dd_cut=%r,\n"
            "         vol_strat=%r, vol_bench=%r, cagr_strat=%r, cagr_bench=%r,\n"
            "         diff_bps=%r, t_nw=%r)\n"
            "print('DRAWDOWN : overlay %%+.1f%%%%  vs  static %%+.1f%%%%   (cut %%+.1f pp)'\n"
            "      %% (R['maxdd_strat'], R['maxdd_bench'], R['dd_cut']))\n"
            "print('VOL      : overlay %%.1f%%%%   vs  static %%.1f%%%%' %% (R['vol_strat'], R['vol_bench']))\n"
            "print('SHARPE   : overlay %%.2f   vs  static %%.2f   (adv %%+.2f)'\n"
            "      %% (R['sharpe_strat'], R['sharpe_bench'], R['sharpe_adv']))\n"
            "print('CAGR     : overlay %%.2f%%%%  vs  static %%.2f%%%%   <- gives up return for the calm'\n"
            "      %% (R['cagr_strat'], R['cagr_bench']))"
            % (R["sharpe_strat"], R["sharpe_bench"], R["sharpe_adv"],
               R["maxdd_strat"], R["maxdd_bench"], R["dd_cut"],
               R["vol_strat"], R["vol_bench"], R["cagr_strat"], R["cagr_bench"],
               R["diff_bps"], R["t_nw"])
        ),
        md("## 2. Where the edge lives — the two crash years\n\n"
           f"In **2008** the overlay returned **{R['y2008_ov']:+.1f}%** while the static book "
           f"lost **{R['y2008_st']:+.1f}%**; in **2022** (stocks *and* bonds down together) "
           f"**{R['y2022_ov']:+.1f}%** vs **{R['y2022_st']:+.1f}%**. In the calm years in "
           "between it *lags*, dripping away return on false signals and sitting in low-yield "
           "cash. It is tail insurance, not a return engine."),
        md("## 3. Is the sort just lucky? A live synthetic control\n\n"
           "Plant deep, persistent bear regimes the 200-day filter can duck (`edge=1`), and "
           "a flat null with no bear to duck (`edge=0`). The overlay must recover a drawdown "
           "cut and a Sharpe pickup in the planted world, and do neither on the null. Live, "
           "no network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from trend6040 import data, strategy as st\n"
            "# average a few seeds so the demo is representative, not one lucky/unlucky world\n"
            "plant = [st.synthetic_detect(data.synthetic_prices(edge=1.0, seed=894+s, n_days=5000)) for s in range(6)]\n"
            "null  = [st.synthetic_detect(data.synthetic_prices(edge=0.0, seed=894+s, n_days=5000)) for s in range(6)]\n"
            "pdd, padv = np.mean([d['dd_cut'] for d in plant])*100, np.mean([d['sharpe_adv'] for d in plant])\n"
            "ndd, nadv = np.mean([d['dd_cut'] for d in null])*100, np.mean([d['sharpe_adv'] for d in null])\n"
            "print('planted world (6 seeds): DD cut %+.1f pp, Sharpe adv %+.2f  (both light up)' % (pdd, padv))\n"
            "print('null world    (6 seeds): DD cut %+.1f pp, Sharpe adv %+.2f  (~ no skill)'   % (ndd, nadv))"
        ),
        md("## 4. The honest verdict — real risk cut, no bankable free lunch\n\n"
           f"On the real tape the overlay **roughly halves the drawdown** "
           f"({R['maxdd_strat']:+.1f}% vs {R['maxdd_bench']:+.1f}%) and the volatility — a "
           f"genuine, mechanical benefit. But it *keeps the return* it claims to only in a "
           f"loose sense: it **gives up ~{R['cagr_bench']-R['cagr_strat']:.1f} pp/yr of CAGR**, "
           f"the daily return difference is negative and insignificant ({R['diff_bps']:+.2f} "
           f"bps/day, NW *t* = {R['t_nw']:+.2f}), and the Sharpe advantage of {R['sharpe_adv']:+.2f} "
           f"has a bootstrap CI that **straddles zero** ([{R['boot_lo']:+.2f}, {R['boot_hi']:+.2f}], "
           f"positive in only {R['boot_ppos']*100:.0f}% of resamples). Worse, a **{25}% short-term "
           f"tax** on each forced move to cash flips that thin edge to {R['tax25_adv']:+.2f}. "
           f"**Signal: Weak** (real drawdown cut, no robust Sharpe edge), **Tradability: "
           f"Fragile** (survives trading costs, dies to tax and time)."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 894 — Trend Overlay on 60/40 — the teardown 📉\n\n"
           "The excess-of-cash vs excess-of-cash race, the HAC *t* on the return difference, "
           "the **paired** Sharpe-advantage bootstrap, the two-era cut, the calendar table, "
           "the switching-cost grid, the short-term-gains tax drag, and the synthetic control."),
        code("R = %r" % (R,)),
        md("## The headline — overlay vs static 60/40 (excess of BIL, gross)"),
        code(
            "print(f\"excess Sharpe : overlay {R['sharpe_strat']:.3f} vs static {R['sharpe_bench']:.3f}  \"\n"
            "      f\"(advantage {R['sharpe_adv']:+.3f})\")\n"
            "print(f\"max drawdown : {R['maxdd_strat']:+.1f}% vs {R['maxdd_bench']:+.1f}%  \"\n"
            "      f\"(cut {R['dd_cut']:+.1f} pp)\")\n"
            "print(f\"vol / CAGR   : {R['vol_strat']:.1f}% / {R['cagr_strat']:.2f}%  vs  \"\n"
            "      f\"{R['vol_bench']:.1f}% / {R['cagr_bench']:.2f}%\")\n"
            "print(f\"return diff  : {R['diff_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"(one-sample t = {R['t_1s']:+.2f})\")"
        ),
        md("## Is the Sharpe advantage real? Paired block-bootstrap\n\n"
           "Resample overlay and static excess returns with the *same* block indices each "
           "draw and take the Sharpe difference — a CI for the advantage itself."),
        code(
            "print(f\"Sharpe advantage {R['boot_adv']:+.3f}  95% CI [{R['boot_lo']:+.3f}, {R['boot_hi']:+.3f}]  \"\n"
            "      f\"P(adv>0) = {R['boot_ppos']:.3f}\")\n"
            "print(f\"return diff     {R['boot_diff']:+.2f} bps/day  95% CI [{R['boot_diff_lo']:+.2f}, {R['boot_diff_hi']:+.2f}]\")\n"
            "print('-> both CIs straddle zero: the risk-adjusted OUTperformance is not robust.')"
        ),
        md("## Robustness — two eras (split 2017-01-01)\n\n"
           "The advantage is front-loaded on 2008; post-2017 it fades."),
        code(
            "print(f\"2007-2016 (n={R['era_early_n']}): Sharpe adv {R['era_early_adv']:+.3f}, DD cut {R['era_early_dd']:+.1f} pp\")\n"
            "print(f\"2017-2026 (n={R['era_late_n']}): Sharpe adv {R['era_late_adv']:+.3f}, DD cut {R['era_late_dd']:+.1f} pp\")"
        ),
        md("## Calendar years — the edge is 2008 & 2022 (net of 3 bps switching)"),
        code(
            "print(f\"2008: overlay {R['y2008_ov']:+.1f}%  static {R['y2008_st']:+.1f}%  (diff {R['y2008_ov']-R['y2008_st']:+.1f} pp)\")\n"
            "print(f\"2022: overlay {R['y2022_ov']:+.1f}%  static {R['y2022_st']:+.1f}%  (diff {R['y2022_ov']-R['y2022_st']:+.1f} pp)\")"
        ),
        md("## Costs are trivial, tax is the killer\n\n"
           "A 200-day rule trades rarely, so switching costs barely move the edge — but each "
           "forced exit to cash realises a short-term gain a buy-and-hold book defers."),
        code(
            "print(f\"switching only  3bps: Sharpe adv {R['cost3_adv']:+.3f}   5bps: {R['cost5_adv']:+.3f}\")\n"
            "print(f\"+ 15% ST tax        : Sharpe adv {R['tax15_adv']:+.3f}  (already negative)\")\n"
            "print(f\"+ 25% ST tax        : Sharpe adv {R['tax25_adv']:+.3f}  (CAGR {R['tax25_cagr']:.2f}%)\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the overlay must recover a planted trend benefit and stay flat on the null."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from trend6040 import data, strategy as st\n"
            "adv_p = np.array([st.synthetic_detect(data.synthetic_prices(edge=1.0, seed=894+s, n_days=5000))['sharpe_adv'] for s in range(8)])\n"
            "adv_n = np.array([st.synthetic_detect(data.synthetic_prices(edge=0.0, seed=894+s, n_days=5000))['sharpe_adv'] for s in range(8)])\n"
            "print(f\"planted (edge=1), 8 seeds: Sharpe adv mean {adv_p.mean():+.3f}, adv>0 in {(adv_p>0).sum()}/8\")\n"
            "print(f\"null    (edge=0), 8 seeds: Sharpe adv mean {adv_n.mean():+.3f} (no bear to duck)\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** A **real, robust drawdown & vol cut** (max DD "
           f"{R['maxdd_strat']:+.1f}% vs {R['maxdd_bench']:+.1f}%, positive in both eras) — but the "
           f"*Sharpe advantage* ({R['sharpe_adv']:+.2f}) has a bootstrap CI that straddles zero "
           f"([{R['boot_lo']:+.2f}, {R['boot_hi']:+.2f}], P>0 = {R['boot_ppos']:.2f}), the return "
           f"difference is negative & insignificant ({R['diff_bps']:+.2f} bps/day, NW *t* = "
           f"{R['t_nw']:+.2f}), and it is front-loaded on 2008 (era adv "
           f"{R['era_early_adv']:+.2f} → {R['era_late_adv']:+.2f}). Real risk reduction, no robust "
           f"risk-adjusted outperformance. *Short history flatters the trend rule.*\n"
           f"- **Tradability — Fragile.** Trading costs are trivial ({R['cost5_adv']:+.2f} adv at "
           f"5 bps), so a tax-deferred account gets a genuinely calmer ride — but a **15% "
           f"short-term tax flips the advantage negative** ({R['tax15_adv']:+.2f}), 25% takes it to "
           f"{R['tax25_adv']:+.2f}, and the edge decays across eras. Real-but-thin & tax-eaten — "
           f"Fragile, not Investable, not a Mirage (the drawdown protection does survive costs)."),
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
