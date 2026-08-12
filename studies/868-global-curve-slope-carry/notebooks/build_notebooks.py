"""Generate the two narrative notebooks for Study 868 (Global Curve-Slope Carry).

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
# total-return closes, 6 sovereign-bond ETFs, 2007-01-31 -> 2026-06-30; cross-sectional
# carry sort, one-month execution lag).
R = dict(
    start="2007-01-31", end="2026-06-30", n_etfs=6, n_months=215, fingerprint="63122f29382a",
    bh_bps=21.36, bh_ann=2.56, bh_sharpe=0.398, bh_t=1.79,
    # yield-to-duration carry (primary)
    ytd_bps=-20.17, ytd_ann=-2.42, ytd_vol=8.10, ytd_sharpe=-0.299, ytd_t_nw=-1.45,
    ytd_t1s=-1.27, ytd_hit=0.447, ytd_long=9.99, ytd_short=30.16,
    ytd_pl_obs=-20.17, ytd_pl_mean=-0.23, ytd_pl_sd=10.91, ytd_pl_p=0.9387,
    ytd_e1_bps=-11.37, ytd_e1_t=-0.38, ytd_e2_bps=-40.24, ytd_e2_t=-3.02, ytd_e3_bps=2.27, ytd_e3_t=0.13,
    ytd_c5_net=-28.24, ytd_c5_cost=8.07, ytd_c5_netS=-0.419, ytd_c5_t=-1.77,
    ytd_c10_net=-30.06, ytd_c10_cost=9.89, ytd_c10_netS=-0.446, ytd_c10_t=-1.89, ytd_turn=0.36,
    # raw realized-yield carry
    raw_bps=3.16, raw_ann=0.38, raw_vol=8.79, raw_sharpe=0.043, raw_t_nw=0.20,
    raw_t1s=0.18, raw_hit=0.447, raw_long=16.98, raw_short=13.82,
    raw_pl_obs=3.16, raw_pl_mean=-0.03, raw_pl_sd=7.70, raw_pl_p=0.4390,
    raw_e1_bps=41.59, raw_e1_t=1.23, raw_e2_bps=-30.08, raw_e2_t=-1.70, raw_e3_bps=7.17, raw_e3_t=0.42,
    raw_c5_net=-4.81, raw_c5_netS=-0.066, raw_c5_t=-0.28,
    raw_c10_net=-6.53, raw_c10_cost=9.69, raw_c10_netS=-0.089, raw_c10_t=-0.38, raw_turn=0.34,
    # window sweep (yield-to-duration NW t)
    w24_t=-1.70, w36_t=-1.45, w48_t=-0.46, w60_t=0.29,
    null_mean_t=-0.05, null_fire=2, planted_mean_t=18.17, planted_sharpe=3.50, planted_fire=20,
)


HEADER = f"""# Study 868 — Global Curve-Slope Carry 🌍

**Does a *steep* yield curve pay a duration holder — long the high-carry, short the flat markets?**

Koijen, Moskowitz, Pedersen & Vrugt (2018) *"Carry"* find carry predicts returns across
asset classes, including bonds: a steep curve pays a holder the yield *and* the roll-down.
We take the sovereign-bond sleeve on tradable ground — six US + international government-bond
ETFs (`SHY`, `IEF`, `TLT`, `BWX`, `IGOV`, `BNDX`), ranked each month by a **yield-to-duration
carry proxy**, long the high-carry / short the low-carry markets, dollar-neutral,
{R['start']} → {R['end']}.

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: currently-listed funds only, a short full cross-section
(BNDX from 2013), and a price-only carry proxy — an upper bound with limited power.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "A **steep** curve is supposed to pay a duration holder twice — a higher yield "
           "*and* a capital gain as the bond ages and rolls *down* toward lower yields. So "
           "across bond markets a duration investor should tilt toward the steep-curve / "
           "high-carry sleeves and away from the flat / low-carry ones. We proxy each ETF's "
           "carry by its trailing realized yield ÷ its duration and sort the markets on it."),
        code(
            "R = dict(ytd_bps=%r, ytd_t_nw=%r, raw_bps=%r, raw_t_nw=%r, bh_bps=%r, bh_t=%r)\n"
            "print('YIELD-TO-DURATION carry sort: %%+.2f bps/mo (Newey-West t = %%+.2f)' %% (R['ytd_bps'], R['ytd_t_nw']))\n"
            "print('RAW realized-yield carry sort: %%+.2f bps/mo (Newey-West t = %%+.2f)' %% (R['raw_bps'], R['raw_t_nw']))\n"
            "print('NAIVE buy-and-hold           : %%+.2f bps/mo (Newey-West t = %%+.2f)  <- the real yardstick'\n"
            "      %% (R['bh_bps'], R['bh_t']))"
            % (R["ytd_bps"], R["ytd_t_nw"], R["raw_bps"], R["raw_t_nw"], R["bh_bps"], R["bh_t"])
        ),
        md("## 2. Is the carry real? A live synthetic control\n\n"
           "We plant a fixed structural carry spread in a seeded toy world (`edge>0`) and "
           "check the detector recovers it — and that it stays *silent* when every market "
           "yields the same (`edge=0`, nothing to sort on). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from curve_slope_carry import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=868))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.010, seed=868))\n"
            "print('equal-yield world  : NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted-carry world: NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — the carry sort does *not* pay here\n\n"
           f"On six sovereign-bond ETFs the **yield-to-duration** carry sort earns the "
           f"**wrong sign**: **{R['ytd_bps']:+.2f} bps/mo**, Newey-West *t* = "
           f"**{R['ytd_t_nw']:+.2f}** — its 'low-carry' short leg (+{R['ytd_short']:.2f} bps) "
           f"actually *out-earns* its 'high-carry' long leg (+{R['ytd_long']:.2f} bps), and a "
           f"random leg assignment beats it {R['ytd_pl_p']*100:.0f}% of the time. The plainer "
           f"**raw realized-yield** sort is **dead flat** (**{R['raw_bps']:+.2f} bps/mo**, NW "
           f"*t* = **{R['raw_t_nw']:+.2f}**). Naive equal-weight buy-and-hold earns "
           f"**{R['bh_bps']:+.2f} bps/mo** — *more* than either timed book. **Signal: None** "
           "(claimed edge absent, if anything perversely negative), **Tradability: Mirage** "
           "(every variant loses money after costs)."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 868 — Global Curve-Slope Carry — the teardown\n\n"
           "The Newey-West book *t*, the buy-and-hold benchmark, the 3,000-draw "
           "column-permutation placebo, the three-era robustness cut, the formation-window "
           "sweep, the costed backtest, and the 20-seed synthetic control."),
        code("R = %r" % (R,)),
        md("## The headline — yield-to-duration vs raw carry vs just holding the bonds\n\n"
           "A carry edge has to beat *owning the bonds*. Neither variant does — and the "
           "yield-to-duration sort is actively negative."),
        code(
            "print(f\"buy-and-hold      : {R['bh_bps']:+.2f} bps/mo  Sharpe {R['bh_sharpe']:.3f}  NW t = {R['bh_t']:+.2f}\")\n"
            "print(f\"yield-to-duration : {R['ytd_bps']:+.2f} bps/mo  Sharpe {R['ytd_sharpe']:+.3f}  NW t = {R['ytd_t_nw']:+.2f}  (long {R['ytd_long']:+.2f} / short {R['ytd_short']:+.2f} bps)\")\n"
            "print(f\"raw realized-yield: {R['raw_bps']:+.2f} bps/mo  Sharpe {R['raw_sharpe']:+.3f}  NW t = {R['raw_t_nw']:+.2f}  (long {R['raw_long']:+.2f} / short {R['raw_short']:+.2f} bps)\")\n"
            "print('  -> the yield-to-duration SHORT leg out-earns its LONG leg: the carry sort is inverted')"
        ),
        md("## Placebo — permute which market feeds each rank (3,000 draws)\n\n"
           "Break the carry->forward-return link; does the sort beat a random leg assignment?"),
        code(
            "print(f\"yield-to-duration: observed {R['ytd_pl_obs']:+.2f} bps vs null {R['ytd_pl_mean']:+.2f} (sd {R['ytd_pl_sd']:.2f}) -> right-tail p = {R['ytd_pl_p']:.4f}\")\n"
            "print(f\"raw realized-yield: observed {R['raw_pl_obs']:+.2f} bps vs null {R['raw_pl_mean']:+.2f} (sd {R['raw_pl_sd']:.2f}) -> right-tail p = {R['raw_pl_p']:.4f}\")\n"
            "print('  yield-to-duration observation sits DEEP in the wrong tail (p=0.94) -> beaten by random assignment')"
        ),
        md("## Robustness — three eras and a formation-window sweep"),
        code(
            "print('era          yield-to-duration      raw realized-yield')\n"
            "print(f\"2010-2016    {R['ytd_e1_bps']:+7.2f} (t={R['ytd_e1_t']:+.2f})   {R['raw_e1_bps']:+7.2f} (t={R['raw_e1_t']:+.2f})\")\n"
            "print(f\"2016-2021    {R['ytd_e2_bps']:+7.2f} (t={R['ytd_e2_t']:+.2f})   {R['raw_e2_bps']:+7.2f} (t={R['raw_e2_t']:+.2f})\")\n"
            "print(f\"2021-2026    {R['ytd_e3_bps']:+7.2f} (t={R['ytd_e3_t']:+.2f})   {R['raw_e3_bps']:+7.2f} (t={R['raw_e3_t']:+.2f})\")\n"
            "print(f\"window NW t (yield-to-duration): 24m {R['w24_t']:+.2f}  36m {R['w36_t']:+.2f}  48m {R['w48_t']:+.2f}  60m {R['w60_t']:+.2f}  (flips sign, never a robust positive)\")"
        ),
        md("## The costed backtest — one-way turnover + short borrow\n\n"
           "One-way cost x turnover per rebalance; short book pays 75 bps/yr borrow."),
        code(
            "print(f\"yield-to-duration  5 bps: gross {R['ytd_bps']:+.2f} -> net {R['ytd_c5_net']:+.2f} bps/mo (cost {R['ytd_c5_cost']:.2f}, net Sharpe {R['ytd_c5_netS']:+.3f}, t={R['ytd_c5_t']:+.2f})\")\n"
            "print(f\"yield-to-duration 10 bps: gross {R['ytd_bps']:+.2f} -> net {R['ytd_c10_net']:+.2f} bps/mo (cost {R['ytd_c10_cost']:.2f}, net Sharpe {R['ytd_c10_netS']:+.3f}, t={R['ytd_c10_t']:+.2f})\")\n"
            "print(f\"raw realized-yield  5 bps: gross {R['raw_bps']:+.2f} -> net {R['raw_c5_net']:+.2f} bps/mo (net Sharpe {R['raw_c5_netS']:+.3f}, t={R['raw_c5_t']:+.2f})\")\n"
            "print(f\"raw realized-yield 10 bps: gross {R['raw_bps']:+.2f} -> net {R['raw_c10_net']:+.2f} bps/mo (cost {R['raw_c10_cost']:.2f}, net Sharpe {R['raw_c10_netS']:+.3f}, t={R['raw_c10_t']:+.2f})\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire when every market yields the same, and must "
           "recover a planted carry spread."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from curve_slope_carry import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=868+s))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.010, seed=868))\n"
            "print(f\"planted (edge=0.010): NW t = {planted['t_nw']:+.2f}, Sharpe = {planted['sharpe']:.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The claimed global curve-slope carry premium does **not**"
           f" appear. The yield-to-duration sort is **wrong-signed** (**{R['ytd_bps']:+.2f}"
           f" bps/mo**, NW *t* = **{R['ytd_t_nw']:+.2f}**, placebo *p* = {R['ytd_pl_p']:.2f} —"
           f" beaten by random assignment), the raw-carry sort is flat"
           f" (**{R['raw_bps']:+.2f} bps/mo**, NW *t* = {R['raw_t_nw']:+.2f}, placebo *p* ="
           f" {R['raw_pl_p']:.2f}), and both flip sign across eras and windows. The 20-seed"
           f" synthetic control fires on a planted carry (mean *t* = {R['planted_mean_t']:+.2f},"
           f" {R['null_fire']}/20 on the null) — the null is real.\n"
           f"- **Tradability — Mirage.** Every variant loses money once costed"
           f" (yield-to-duration net **{R['ytd_c5_net']:+.2f} bps/mo** at 5 bps, raw carry net"
           f" **{R['raw_c5_net']:+.2f} bps/mo** at 5 bps; net Sharpe < 0 throughout) — no"
           f" costed net edge survives."),
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
