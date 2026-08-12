"""Generate the two narrative notebooks for Study 902 (Multi-Factor Composite).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from
the frozen ``R`` dict (mirroring docs/results.md); the live cells run only the fast
synthetic control, so execution is quick and network-free.
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


# Frozen real-tape headline numbers — mirror of docs/results.md. yfinance daily total-return
# closes; equal-weight VLUE/QUAL/MTUM/USMV/SIZE sleeve rebalanced monthly, vs SPY, minus BIL
# cash; common blend window 2013-08-31 -> 2026-06-30 (155 months); one-way 2 bps rebalance cost.
R = dict(
    start="2013-08-31", end="2026-06-30", n_months=155, fingerprint="8b05ab0c64b8",
    turnover_pct=1.17, cost_bps_yr=0.3,
    comp_cagr=13.32, comp_vol=14.1, comp_sharpe=0.841, comp_maxdd=-23.7,
    spy_cagr=14.14, spy_vol=14.5, spy_sharpe=0.874, spy_maxdd=-23.9,
    adv=-0.033, active_bps=-6.5, t_active=-0.73, win_rate=51,
    boot_lo=-0.205, boot_hi=0.077, boot_pneg=0.804,
    cbb_lo=0.389, cbb_hi=1.373,
    era_early_adv=0.078, era_early_t=-0.02, era_early_n=77,
    era_late_adv=-0.085, era_late_t=-0.77, era_late_n=78,
    mean_single_vol=15.2, min_single_vol=11.6,
    mean_single_sharpe=0.786, best_single_sharpe=0.928,
    cross_disp_pp=7.0, blend_annual_sd=11.9,
    single={"VLUE": 0.699, "QUAL": 0.841, "MTUM": 0.928, "USMV": 0.780, "SIZE": 0.681},
    invvol_sharpe=0.832, invvol_t=-1.30,
    null_adv=0.007, null_t=0.21, planted_adv=0.198, planted_t=3.66,
)


HEADER = f"""# Study 902 — Multi-Factor Composite 🧩

**"Single factors take turns working — so blend them." Does the blend beat the market?**

The practitioner's pitch for a multi-factor sleeve is *diversification*: value, quality,
momentum, min-vol and size each spend years out of favour, but a blend smooths the ride and
— the sell-side deck promises — earns a market-beating risk-adjusted return with less
factor-timing risk. We build the live version — an **equal-weight sleeve of VLUE + QUAL +
MTUM + USMV + SIZE**, rebalanced monthly — and race it against **SPY** on the
**excess-of-cash Sharpe** (both legs minus the BIL T-bill ETF), net of the rebalancing
turnover it actually pays ({R['start']} → {R['end']}, {R['n_months']} months common to all
five sleeves).

*Numbers below are the frozen headline (`docs/results.md`, fingerprint
`{R['fingerprint']}`); the live cells run the fast synthetic control. These five are the
flagship survivors of the 2010s smart-beta wave — the panel flatters the average factor-ETF
experience; named on the Signal axis.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "One factor is a bumpy ride: value spent 2017–2020 in the wilderness, momentum "
           "crashed in 2009, min-vol lags in melt-ups. Put all five in one basket, rebalance "
           "monthly, and the bad years of one are cushioned by the good years of another. "
           "That part is **real and mechanical** — a diversified blend has lower volatility "
           "than the average single sleeve. The open question is the *second* promise: does "
           "the smoother ride also **beat the market** once you net out costs?"),
        code(
            "R = dict(comp_sharpe=%r, spy_sharpe=%r, adv=%r, active_bps=%r, t_active=%r,\n"
            "         comp_vol=%r, spy_vol=%r, mean_single_vol=%r, mean_single_sharpe=%r)\n"
            "print('composite excess Sharpe : %%.3f' %% R['comp_sharpe'])\n"
            "print('SPY excess Sharpe        : %%.3f' %% R['spy_sharpe'])\n"
            "print('advantage (comp - SPY)   : %%+.3f  (active %%+.1f bps/mo, NW t %%+.2f)'\n"
            "      %% (R['adv'], R['active_bps'], R['t_active']))\n"
            "print('--- the diversification that DOES show up ---')\n"
            "print('composite vol %%.1f%%%% vs mean single sleeve %%.1f%%%%' %% (R['comp_vol'], R['mean_single_vol']))\n"
            "print('composite Sharpe %%.3f vs mean single sleeve %%.3f' %% (R['comp_sharpe'], R['mean_single_sharpe']))"
            % (R["comp_sharpe"], R["spy_sharpe"], R["adv"], R["active_bps"], R["t_active"],
               R["comp_vol"], R["spy_vol"], R["mean_single_vol"], R["mean_single_sharpe"])
        ),
        md("## 2. Is the machinery honest? A live synthetic control\n\n"
           "Before trusting the race on the real tape, we prove the detector on a seeded toy "
           "world: five members that share a market and each carry an independent style "
           "factor, plus a benchmark and cash. Plant a per-annum blend edge (`edge=+3%`) and "
           "the Sharpe-advantage *t* must light up; set it to zero and the *t* must stay "
           "quiet. No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from multi_factor import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_world(n_months=168, edge_ann=0.0, seed=902))\n"
            "planted = st.synthetic_detect(data.synthetic_world(n_months=168, edge_ann=0.03, seed=902))\n"
            "print('null world   : Sharpe adv %+.3f  active NW t = %+.2f  (should be ~0)'\n"
            "      % (null['sharpe_adv'], null['t_active_nw']))\n"
            "print('planted world: Sharpe adv %+.3f  active NW t = %+.2f  (should light up)'\n"
            "      % (planted['sharpe_adv'], planted['t_active_nw']))"
        ),
        md(f"## 3. The honest verdict — a real diversifier, not a market-beater\n\n"
           f"On the live tape the equal-weight sleeve earns an excess-of-cash Sharpe of "
           f"**{R['comp_sharpe']:.3f}** against SPY's **{R['spy_sharpe']:.3f}** — an "
           f"advantage of **{R['adv']:+.3f}**, i.e. it lands *just short* of the market. The "
           f"active return is **{R['active_bps']:+.1f} bps/mo** at NW *t* = "
           f"**{R['t_active']:+.2f}** (indistinguishable from zero, and the wrong side of "
           f"it), a paired bootstrap puts the advantage CI at "
           f"**[{R['boot_lo']:+.3f}, {R['boot_hi']:+.3f}]** straddling zero "
           f"(P(adv<0)={R['boot_pneg']:.2f}), and it **flips sign across the two eras** "
           f"({R['era_early_adv']:+.3f} early → {R['era_late_adv']:+.3f} late). Rebalancing "
           f"costs are a **non-issue** — the sleeve turns over just {R['turnover_pct']:.1f}% "
           f"of NAV/mo ({R['cost_bps_yr']:.1f} bps/yr) — so this isn't a cost story; there is "
           f"simply **no market-beating edge** to cost. What *is* real: the blend's vol "
           f"({R['comp_vol']:.1f}%) sits below the average single sleeve's "
           f"({R['mean_single_vol']:.1f}%) and its Sharpe ({R['comp_sharpe']:.3f}) beats the "
           f"average single sleeve ({R['mean_single_sharpe']:.3f}) — genuine diversification "
           f"of *factor-timing* risk, just not of *market* risk. **Signal: Weak** (the "
           f"diversification is real, the SPY-beating claim is not), **Tradability: "
           f"Fragile** (cheap to hold, but what you hold is a marginally-lower-Sharpe, "
           f"lower-vol version of the market)."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 902 — Multi-Factor Composite — the teardown\n\n"
           "The excess-of-cash Sharpe race, the HAC *t* on the active return, the paired "
           "moving-block bootstrap on the Sharpe advantage, the two-era robustness cut, the "
           "diversification decomposition, the per-sleeve breakdown, the inverse-vol "
           "robustness alt, and the planted-edge synthetic control. Real numbers are quoted "
           "from the frozen `R` dict (`examples/verify.py` reproduces them from the cache); "
           "the live cell runs the synthetic control offline."),
        code(
            "R = " + repr(R) + "\n"
            "print('window %s -> %s  (%d months, fingerprint %s)'\n"
            "      % (R['start'], R['end'], R['n_months'], R['fingerprint']))"
        ),
        md("## 1. The excess-of-cash Sharpe race (both legs minus BIL)\n\n"
           "The core comparison. Composite **net** of a one-way 2 bps rebalance cost vs SPY, "
           "both in excess of the tradable BIL T-bill ETF."),
        code(
            "print('leg           exSharpe   CAGR     vol     maxDD')\n"
            "print('composite net %8.3f  %6.2f%%  %5.1f%%  %6.1f%%'\n"
            "      % (R['comp_sharpe'], R['comp_cagr'], R['comp_vol'], R['comp_maxdd']))\n"
            "print('SPY           %8.3f  %6.2f%%  %5.1f%%  %6.1f%%'\n"
            "      % (R['spy_sharpe'], R['spy_cagr'], R['spy_vol'], R['spy_maxdd']))\n"
            "print()\n"
            "print('Sharpe advantage (comp - SPY): %+.3f' % R['adv'])\n"
            "print('active return: %+.1f bps/mo   NW t = %+.2f   win-rate %d%%'\n"
            "      % (R['active_bps'], R['t_active'], R['win_rate']))"
        ),
        md("## 2. Bootstrap CI on the advantage + two-era robustness\n\n"
           "A green Signal needs the advantage clear of zero *and* stable across sub-eras. "
           "Neither holds: the paired moving-block bootstrap CI straddles zero, and the "
           "advantage flips sign from the first half to the second."),
        code(
            "print('paired block bootstrap on Sharpe advantage:')\n"
            "print('  adv %+.3f   95%% CI [%+.3f, %+.3f]   P(adv<0) = %.2f'\n"
            "      % (R['adv'], R['boot_lo'], R['boot_hi'], R['boot_pneg']))\n"
            "print('  (composite excess Sharpe CBB CI [%.3f, %.3f] — wide on 155 months)'\n"
            "      % (R['cbb_lo'], R['cbb_hi']))\n"
            "print()\n"
            "print('two-era split:')\n"
            "print('  early (%dm): adv %+.3f   active NW t %+.2f'\n"
            "      % (R['era_early_n'], R['era_early_adv'], R['era_early_t']))\n"
            "print('  late  (%dm): adv %+.3f   active NW t %+.2f  <- sign flip'\n"
            "      % (R['era_late_n'], R['era_late_adv'], R['era_late_t']))"
        ),
        md("## 3. What the blend DOES diversify — the factor-timing pitch\n\n"
           "The diversification claim is not empty: the blend genuinely beats the *average* "
           "single sleeve on both vol and Sharpe, and it dampens the wide cross-sleeve "
           "dispersion each year. It just doesn't clear the *market*."),
        code(
            "print('composite vol %.1f%% < mean single sleeve %.1f%% (min single %.1f%%, SPY %.1f%%)'\n"
            "      % (R['comp_vol'], R['mean_single_vol'], R['min_single_vol'], R['spy_vol']))\n"
            "print('composite Sharpe %.3f > mean single %.3f (best single %.3f, SPY %.3f)'\n"
            "      % (R['comp_sharpe'], R['mean_single_sharpe'], R['best_single_sharpe'], R['spy_sharpe']))\n"
            "print('avg cross-sleeve annual dispersion %.1f pp -> blend year-to-year sd %.1f pp'\n"
            "      % (R['cross_disp_pp'], R['blend_annual_sd']))\n"
            "print()\n"
            "print('per single-factor sleeve excess Sharpe (common window):')\n"
            "for tk, s in R['single'].items():\n"
            "    print('  %-5s %.3f' % (tk, s))"
        ),
        md("## 4. Robustness — inverse-vol weighting\n\n"
           "Risk-weighting the sleeve (inverse trailing 12-month vol, point-in-time) doesn't "
           "rescue it: the advantage stays negative."),
        code(
            "print('inverse-vol sleeve net exSharpe %.3f vs SPY %.3f (adv %+.3f, active NW t %+.2f)'\n"
            "      % (R['invvol_sharpe'], R['spy_sharpe'], R['invvol_sharpe']-R['spy_sharpe'], R['invvol_t']))"
        ),
        md("## 5. Synthetic control — the machinery is faithful\n\n"
           "The detector recovers a *planted* per-annum blend edge and stays silent on the "
           "null. Machinery proof only — never cited in support of a stamp."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from multi_factor import data, strategy as st\n"
            "for label, edge in [('null    (edge=+0%/yr)', 0.0), ('planted (edge=+3%/yr)', 0.03)]:\n"
            "    d = st.synthetic_detect(data.synthetic_world(n_months=168, edge_ann=edge, seed=902))\n"
            "    print('%s: Sharpe adv %+.3f  active %+.1f bps/mo  NW t = %+.2f'\n"
            "          % (label, d['sharpe_adv'], d['active_bps'], d['t_active_nw']))"
        ),
        md(f"## Verdict\n\n"
           f"**Signal — Weak.** The diversification the sleeve is sold on is **real and "
           f"mechanical**: the equal-weight blend carries lower vol ({R['comp_vol']:.1f}%) "
           f"and a higher excess Sharpe ({R['comp_sharpe']:.3f}) than the *average* single "
           f"factor sleeve ({R['mean_single_vol']:.1f}% / {R['mean_single_sharpe']:.3f}), and "
           f"it tames the ~{R['cross_disp_pp']:.0f} pp/yr cross-sleeve dispersion. But the "
           f"headline claim — *beat the market on risk-adjusted return* — fails: the "
           f"excess-of-cash Sharpe advantage over SPY is **{R['adv']:+.3f}** (active NW *t* "
           f"**{R['t_active']:+.2f}**), the bootstrap CI **[{R['boot_lo']:+.3f}, "
           f"{R['boot_hi']:+.3f}]** straddles zero, and it flips sign across eras. No "
           f"SPY-beating edge; a genuine diversification benefit only. Flagship-survivor "
           f"selection is named.\n\n"
           f"**Tradability — Fragile.** Costs are *not* the obstacle — the sleeve turns over "
           f"{R['turnover_pct']:.1f}% of NAV/mo ({R['cost_bps_yr']:.1f} bps/yr), so the "
           f"gross and net numbers are identical to two decimals. The real, deliverable "
           f"benefit (diversification of factor-timing risk, one ticket per sleeve, penny "
           f"spreads) is trivially buyable — but what it buys is a marginally-lower-Sharpe, "
           f"lower-vol clone of SPY, not a bankable market-beating edge. Real but thin → "
           f"Fragile, not Investable."),
    ]
    nb["cells"] = cells
    return nb


def main():
    for name, builder in [("01_for_the_curious", build_curious),
                          ("02_for_the_quants", build_quants)]:
        nb = builder()
        path = os.path.join(HERE, f"{name}.ipynb")
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print("wrote", path)


if __name__ == "__main__":
    main()
