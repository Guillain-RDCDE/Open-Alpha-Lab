"""Generate the two narrative notebooks for Study 829 (Global Sovereign-Bond Momentum).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance month-end
# total-return closes, 5 global sovereign-bond ETFs, 2007-01-31 -> 2026-06-30; 12-1
# time-series momentum, one-month execution lag).
R = dict(
    start="2007-01-31", end="2026-06-30", n_etfs=5, n_months=221, fingerprint="fee295eff13f",
    bh_bps=24.56, bh_ann=2.95, bh_sharpe=0.399, bh_t=1.88,
    # long-short (+1/-1)
    ls_bps=0.26, ls_ann=0.03, ls_vol=5.99, ls_sharpe=0.005, ls_t_nw=0.02, ls_t1s=0.02, ls_hit=0.548,
    ls_pl_obs=0.26, ls_pl_mean=7.46, ls_pl_sd=12.47, ls_pl_p=0.7243,
    ls_e1_bps=4.65, ls_e1_t=0.23, ls_e2_bps=-9.80, ls_e2_t=-0.91, ls_e3_bps=5.55, ls_e3_t=0.25,
    ls_c5_net=-3.18, ls_c5_cost=3.44, ls_c5_netS=-0.064, ls_c5_t=-0.27,
    ls_c10_net=-4.48, ls_c10_cost=4.74, ls_c10_netS=-0.090, ls_c10_t=-0.39, ls_turn=0.26,
    # long-only (+1/0)
    lo_bps=12.27, lo_ann=1.47, lo_vol=4.56, lo_sharpe=0.323, lo_t_nw=1.75, lo_t1s=1.39, lo_hit=0.516,
    lo_pl_obs=12.27, lo_pl_mean=13.57, lo_pl_sd=6.92, lo_pl_p=0.5330,
    lo_e1_bps=26.94, lo_e1_t=1.66, lo_e2_bps=8.66, lo_e2_t=0.88, lo_e3_bps=2.25, lo_e3_t=0.29,
    lo_c5_net=11.59, lo_c5_netS=0.306, lo_c5_t=1.31,
    lo_c10_net=10.92, lo_c10_cost=1.35, lo_c10_netS=0.288, lo_c10_t=1.24, lo_turn=0.14,
    lb6_t=1.56, lb9_t=1.23, lb12_t=1.75, lb15_t=1.52,
    null_mean_t=-0.22, null_fire=0, planted_mean_t=13.38, planted_sharpe=2.45, planted_fire=20,
)


HEADER = f"""# Study 829 — Global Sovereign-Bond Momentum 🌍📉

**Does a bond market's own recent *trend* predict its next move?**

Moskowitz, Ooi & Pedersen (2012) *"Time Series Momentum"* find that across asset classes
— **including government bonds** — a market's own past 12-month return positively predicts
its next month. We take the bond sleeve on tradable ground: five global sovereign-bond
ETFs (`BWX`, `IGOV`, `BNDX`, `EMB`, `IEF`), each **signed by its 12-minus-1-month
total-return trend**, long the up-trend / short the down-trend (and a long-only variant),
{R['start']} → {R['end']}.

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: currently-listed funds only, tiny 5-ETF panel — an upper
bound with limited power.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Bond *levels* trend for years — a rate-cutting cycle lifts prices, a hiking "
           "cycle sinks them. So the sign of a market's trailing twelve-month total return "
           "(skipping the most recent month, the '12-1' window) *should* be an informative "
           "state for next month: own the up-trenders, short (or sit out) the down-trenders."),
        code(
            "R = dict(ls_bps=%r, ls_t_nw=%r, lo_bps=%r, lo_t_nw=%r, bh_bps=%r, bh_t=%r)\n"
            "print('LONG-SHORT 12-1 book : %%+.2f bps/mo (Newey-West t = %%+.2f)' %% (R['ls_bps'], R['ls_t_nw']))\n"
            "print('LONG-ONLY trend timing: %%+.2f bps/mo (Newey-West t = %%+.2f)' %% (R['lo_bps'], R['lo_t_nw']))\n"
            "print('NAIVE buy-and-hold    : %%+.2f bps/mo (Newey-West t = %%+.2f)  <- the real yardstick'\n"
            "      %% (R['bh_bps'], R['bh_t']))"
            % (R["ls_bps"], R["ls_t_nw"], R["lo_bps"], R["lo_t_nw"], R["bh_bps"], R["bh_t"])
        ),
        md("## 2. Is the trend real? A live synthetic control\n\n"
           "We plant a persistent trend in a seeded toy world (`edge>0`) and check the "
           "detector recovers it — and that it stays *silent* on a random walk (`edge=0`, "
           "no trend to find). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from sovereign_momentum import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=829))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.010, seed=829))\n"
            "print('random-walk world: NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted-trend world: NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — the trend does *not* pay here\n\n"
           f"On five global sovereign-bond ETFs the **long-short** 12-1 book is **flat**: "
           f"**{R['ls_bps']:+.2f} bps/mo**, Newey-West *t* = **{R['ls_t_nw']:+.2f}** — no "
           f"predictive sign at all. The **long-only** variant *looks* positive "
           f"(**{R['lo_bps']:+.2f} bps/mo**) but that is just **bond beta**: naive "
           f"equal-weight buy-and-hold earns **{R['bh_bps']:+.2f} bps/mo** — *more* than the "
           f"timed book — and a block-rotation placebo puts the long-only number *below* its "
           f"own null mean ({R['lo_pl_mean']:+.2f} bps, *p* = {R['lo_pl_p']:.2f}). The trend "
           "adds nothing you don't get by simply holding the bonds. **Signal: None** "
           "(claimed edge absent), **Tradability: Mirage** (the long-short book loses money "
           "after costs; the long-only book is costed bond beta that under-earns buy-and-hold)."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 829 — Global Sovereign-Bond Momentum — the teardown\n\n"
           "The Newey-West spread *t*, the buy-and-hold benchmark, the 3,000-draw "
           "block-rotation placebo, the three-era robustness cut, the lookback sweep, the "
           "costed backtest, and the 20-seed synthetic control."),
        code("R = %r" % (R,)),
        md("## The headline — long-short vs long-only vs just holding the bonds\n\n"
           "A trend edge has to beat *owning the bonds*. It doesn't."),
        code(
            "print(f\"buy-and-hold : {R['bh_bps']:+.2f} bps/mo  Sharpe {R['bh_sharpe']:.3f}  NW t = {R['bh_t']:+.2f}\")\n"
            "print(f\"long-short   : {R['ls_bps']:+.2f} bps/mo  Sharpe {R['ls_sharpe']:.3f}  NW t = {R['ls_t_nw']:+.2f}  (hit {R['ls_hit']:.3f})\")\n"
            "print(f\"long-only    : {R['lo_bps']:+.2f} bps/mo  Sharpe {R['lo_sharpe']:.3f}  NW t = {R['lo_t_nw']:+.2f}  (hit {R['lo_hit']:.3f})\")\n"
            "print('  -> long-only UNDER-earns buy-and-hold: the +12 bps is bond beta, not trend alpha')"
        ),
        md("## Placebo — block-rotate the returns (3,000 draws)\n\n"
           "Break the trend->forward-return link; does the signal beat its own return series' drift?"),
        code(
            "print(f\"long-short: observed {R['ls_pl_obs']:+.2f} bps vs null {R['ls_pl_mean']:+.2f} (sd {R['ls_pl_sd']:.2f}) -> p = {R['ls_pl_p']:.4f}\")\n"
            "print(f\"long-only : observed {R['lo_pl_obs']:+.2f} bps vs null {R['lo_pl_mean']:+.2f} (sd {R['lo_pl_sd']:.2f}) -> p = {R['lo_pl_p']:.4f}\")\n"
            "print('  long-only observation sits BELOW its own rotation-null mean -> no trend information')"
        ),
        md("## Robustness — three eras and a lookback sweep"),
        code(
            "print('era          long-short            long-only')\n"
            "print(f\"2007-2014    {R['ls_e1_bps']:+6.2f} (t={R['ls_e1_t']:+.2f})    {R['lo_e1_bps']:+6.2f} (t={R['lo_e1_t']:+.2f})\")\n"
            "print(f\"2014-2020    {R['ls_e2_bps']:+6.2f} (t={R['ls_e2_t']:+.2f})    {R['lo_e2_bps']:+6.2f} (t={R['lo_e2_t']:+.2f})\")\n"
            "print(f\"2020-2026    {R['ls_e3_bps']:+6.2f} (t={R['ls_e3_t']:+.2f})    {R['lo_e3_bps']:+6.2f} (t={R['lo_e3_t']:+.2f})\")\n"
            "print(f\"lookback NW t (long-only): 6m {R['lb6_t']:+.2f}  9m {R['lb9_t']:+.2f}  12m {R['lb12_t']:+.2f}  15m {R['lb15_t']:+.2f}  (never reaches 2)\")"
        ),
        md("## The costed backtest — one-way turnover + short borrow\n\n"
           "One-way cost x turnover per rebalance; short book pays 75 bps/yr borrow."),
        code(
            "print(f\"long-short  5 bps: gross {R['ls_bps']:+.2f} -> net {R['ls_c5_net']:+.2f} bps/mo (cost {R['ls_c5_cost']:.2f}, net Sharpe {R['ls_c5_netS']:+.3f}, t={R['ls_c5_t']:+.2f})\")\n"
            "print(f\"long-short 10 bps: gross {R['ls_bps']:+.2f} -> net {R['ls_c10_net']:+.2f} bps/mo (cost {R['ls_c10_cost']:.2f}, net Sharpe {R['ls_c10_netS']:+.3f}, t={R['ls_c10_t']:+.2f})\")\n"
            "print(f\"long-only   5 bps: gross {R['lo_bps']:+.2f} -> net {R['lo_c5_net']:+.2f} bps/mo (net Sharpe {R['lo_c5_netS']:+.3f}, t={R['lo_c5_t']:+.2f})\")\n"
            "print(f\"long-only  10 bps: gross {R['lo_bps']:+.2f} -> net {R['lo_c10_net']:+.2f} bps/mo (cost {R['lo_c10_cost']:.2f}, net Sharpe {R['lo_c10_netS']:+.3f}, t={R['lo_c10_t']:+.2f})\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on a random walk and must recover a planted trend."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from sovereign_momentum import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=829+s))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.010, seed=829))\n"
            "print(f\"planted (edge=0.010): NW t = {planted['t_nw']:+.2f}, Sharpe = {planted['sharpe']:.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The claimed global-sovereign-bond time-series-momentum edge does"
           f" **not** appear. The long-short 12-1 book is flat (**{R['ls_bps']:+.2f} bps/mo**, NW"
           f" *t* = **{R['ls_t_nw']:+.2f}**, placebo *p* = {R['ls_pl_p']:.2f}); the long-only"
           f" +{R['lo_bps']:.2f} bps/mo is **bond beta** (under-earns the {R['bh_bps']:.2f} bps/mo"
           f" buy-and-hold, placebo *p* = {R['lo_pl_p']:.2f}, NW *t* = {R['lo_t_nw']:+.2f} below"
           f" the bar, fragile across lookbacks). The 20-seed synthetic control fires on a planted"
           f" trend (mean *t* = {R['planted_mean_t']:+.2f}, {R['null_fire']}/20 on the null) — the"
           f" null is real.\n"
           f"- **Tradability — Mirage.** Long-short loses money once costed (net"
           f" **{R['ls_c5_net']:+.2f} bps/mo** at 5 bps, {R['ls_c10_net']:+.2f} at 10 bps; net"
           f" Sharpe < 0). Long-only's costed net Sharpe (~{R['lo_c10_netS']:.2f}) is diluted bond"
           f" beta that under-earns buy-and-hold — no costed net edge survives."),
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
