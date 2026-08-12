"""Generate the two narrative notebooks for Study 890 (Sector Risk-Parity).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from the
frozen ``R`` dict (mirroring docs/results.md); the only live cell runs the fast synthetic
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily total-return,
# quarterly inverse-vol / ERC sector books vs cap-weight SPY, both excess of BIL cash).
R = dict(
    # eleven-sector headline (joint window from XLC's 2018-06 inception)
    e_start="2018-10-01", e_end="2026-06-30", e_n=1946, e_rebals=31, e_fp="5e98273e423e", e_rows=2018,
    e_rp_sharpe=0.572, e_rp_ann=12.73, e_rp_vol=17.74, e_rp_dd=-34.7,
    e_spy_sharpe=0.667, e_spy_ann=15.67, e_spy_vol=19.62, e_spy_dd=-33.7,
    e_diff=-0.095, e_ci=(-0.302, 0.132), e_pneg=0.81, e_nwt=-1.50, e_turn=57, e_cost=1.7,
    e_erc_sharpe=0.583, e_erc_diff=-0.084,
    # nine-sector long history (from BIL's 2007-05 inception)
    n_start="2007-10-01", n_end="2026-06-30", n_n=4716, n_rebals=75, n_fp="2afe10148f92", n_rows=4802,
    n_rp_sharpe=0.555, n_rp_ann=11.16, n_rp_vol=17.76, n_rp_dd=-49.6,
    n_spy_sharpe=0.553, n_spy_ann=12.29, n_spy_vol=19.88, n_spy_dd=-55.2,
    n_diff=0.002, n_ci=(-0.099, 0.110), n_pneg=0.47, n_nwt=-1.12, n_turn=48, n_cost=1.4,
    n_erc_sharpe=0.517, n_erc_diff=-0.036,
    # era cut (9-sector inverse-vol)
    era_early="2007-2015", era_early_rp=0.414, era_early_spy=0.349, era_early_diff=0.064, era_early_n=2079,
    era_late="2016-2026", era_late_rp=0.695, era_late_spy=0.759, era_late_diff=-0.064, era_late_n=2637,
    # levered-to-SPY-vol timer (9-sector)
    lev_L=1.12, lev_sharpe=0.552, lev_spy_sharpe=0.553, lev_ann=10.97, lev_spy_ann=10.99,
    lev_fin=7.2, lev_dd=-54.2,
    # calendar-year highlights (9-sector inverse-vol vs SPY, pp)
    cy_2008=4.6, cy_2022=13.3, cy_2023=-14.6, cy_2024=-10.6, cy_wins=7, cy_years=20,
    # synthetic control
    null_mean=0.0031, null_sd=0.0214, null_fire=0, planted=0.136, planted_rp=1.32, planted_spy=1.18,
)


HEADER = f"""# Study 890 — Sector Risk-Parity ⚖️

**Cap-weight buries the S&P in a few mega-cap sectors. If you equal-*risk*-weight the eleven
GICS sectors instead, do you get a better risk-adjusted ride?**

The pitch is "All-Weather, but *within* equities": weight each sector by inverse volatility
(or full equal-risk-contribution), rebalance quarterly, and see whether the diversification
lifts the **excess-of-cash Sharpe** and cuts the drawdown versus **cap-weight SPY**, net of
costs. Diversification, not forecasting.

We test two real panels — an **eleven-sector** headline ({R['e_start']} → {R['e_end']}, short
because XLC only launched 2018-06) and a longer **nine-sector** panel back to {R['n_start']}
(BIL's inception) — both on yfinance daily total-return prices, everything excess of BIL cash.

*Numbers below are the frozen headline (`docs/results.md`); the only live cell runs the fast
synthetic control. Short history on the eleven-sector panel is named on the Signal axis.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Today a third of SPY *is* Information Technology — cap-weight lets the biggest, "
           "highest-vol sectors dominate the portfolio's risk. Inverse-vol weighting hands "
           "each sector the **same risk budget**: it trims the crowded high-vol sectors and "
           "lifts the sleepy low-vol ones (staples, utilities, health care). The hope is a "
           "smoother ride at a better Sharpe. The catch: when the crowded sector (tech) is "
           "exactly what powers the bull market, *under*-weighting it costs you return."),
        code(
            "R = " + repr(R) + "\n"
            "print('ELEVEN-SECTOR (2018-2026, tech-led):')\n"
            "print(f\"  risk-parity  Sharpe {R['e_rp_sharpe']:.2f}  ann {R['e_rp_ann']:+.1f}%  maxDD {R['e_rp_dd']:.0f}%\")\n"
            "print(f\"  cap-weight SPY Sharpe {R['e_spy_sharpe']:.2f}  ann {R['e_spy_ann']:+.1f}%  maxDD {R['e_spy_dd']:.0f}%\")\n"
            "print(f\"  -> Sharpe difference {R['e_diff']:+.3f}  (95% CI {R['e_ci']}) - RP LOST over the tech bull\")\n"
            "print()\n"
            "print('NINE-SECTOR (2007-2026, includes 2008):')\n"
            "print(f\"  risk-parity  Sharpe {R['n_rp_sharpe']:.2f}  ann {R['n_rp_ann']:+.1f}%  vol {R['n_rp_vol']:.1f}%  maxDD {R['n_rp_dd']:.0f}%\")\n"
            "print(f\"  cap-weight SPY Sharpe {R['n_spy_sharpe']:.2f}  ann {R['n_spy_ann']:+.1f}%  vol {R['n_spy_vol']:.1f}%  maxDD {R['n_spy_dd']:.0f}%\")\n"
            "print(f\"  -> Sharpe difference {R['n_diff']:+.3f}  (95% CI {R['n_ci']}) - a DEAD HEAT on Sharpe...\")\n"
            "print(f\"     ...but RP cut vol ({R['n_rp_vol']:.1f}% vs {R['n_spy_vol']:.1f}%) and drawdown ({R['n_rp_dd']:.0f}% vs {R['n_spy_dd']:.0f}%)\")"
        ),
        md("## 2. The tell — the 'advantage' flips sign every era\n\n"
           f"Split the long panel in half and the story is stark. In the crisis-heavy first "
           f"half risk-parity **won** ({R['era_early']}: Sharpe {R['era_early_rp']:.2f} vs "
           f"{R['era_early_spy']:.2f}, diff **{R['era_early_diff']:+.3f}**); in the tech-led "
           f"second half it **lost** ({R['era_late']}: {R['era_late_rp']:.2f} vs "
           f"{R['era_late_spy']:.2f}, diff **{R['era_late_diff']:+.3f}**). The two cancel to "
           f"~zero over the full sample. A real Sharpe edge should not swap signs with the "
           f"regime — this is diversification, not alpha."),
        code(
            "print(f\"{R['era_early']}: RP {R['era_early_rp']:.3f}  SPY {R['era_early_spy']:.3f}  diff {R['era_early_diff']:+.3f}  (RP wins the crises)\")\n"
            "print(f\"{R['era_late']}: RP {R['era_late_rp']:.3f}  SPY {R['era_late_spy']:.3f}  diff {R['era_late_diff']:+.3f}  (RP lags the tech bull)\")\n"
            "print()\n"
            "print(f\"Calendar-year: RP beat SPY by {R['cy_2008']:+.1f}pp in 2008 and {R['cy_2022']:+.1f}pp in 2022 (bear years),\")\n"
            "print(f\"               but lagged by {R['cy_2023']:.1f}pp in 2023 and {R['cy_2024']:.1f}pp in 2024 (AI melt-up).\")"
        ),
        md("## 3. Is the machinery even honest? A live synthetic control\n\n"
           "Before trusting any of that, check the detector on a toy world where we *know* the "
           "answer. Every asset has the same Sharpe but different vols; the cap-weight "
           "benchmark piles on the high-vol names. When vols are dispersed, inverse-vol "
           "*should* out-Sharpe cap-weight; when all vols are equal it *must* tie. No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from sector_rp import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_world(vol_spread=0.0, seed=890))\n"
            "planted = st.synthetic_detect(data.synthetic_world(vol_spread=0.02, seed=890))\n"
            "print('null world   (equal vols): Sharpe advantage %+.3f  (should be ~0)' % null['sharpe_advantage'])\n"
            "print('planted world(dispersed) : Sharpe advantage %+.3f  (should be clearly +)' % planted['sharpe_advantage'])"
        ),
        md("## 4. The honest verdict\n\n"
           f"On the real tape the promised **Sharpe improvement is not there**: the nine-sector "
           f"book ties SPY (diff **{R['n_diff']:+.3f}**, 95% CI {R['n_ci']} straddles zero) and "
           f"the eleven-sector book *loses* over 2018–2026 ({R['e_diff']:+.3f}). What *is* real "
           f"is the **risk reduction** — lower vol ({R['n_rp_vol']:.1f}% vs {R['n_spy_vol']:.1f}%) "
           f"and a milder drawdown ({R['n_rp_dd']:.0f}% vs {R['n_spy_dd']:.0f}%), with RP winning "
           f"every genuine bear year. So the drawdown half of the claim holds and the Sharpe half "
           f"does not: **Signal — Mixed**. And you cannot bank it: unlevered you simply earn "
           f"*less* than SPY for the smoother ride, and levering the book back to SPY's vol just "
           f"reproduces SPY ({R['lev_sharpe']:.2f} vs {R['lev_spy_sharpe']:.2f} Sharpe) — "
           f"**Tradability — Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 890 — Sector Risk-Parity — the teardown\n\n"
           "The excess-vs-excess Sharpe race (inverse-vol and ERC vs cap-weight SPY, both minus "
           "BIL), the paired block-bootstrap CI on the Sharpe difference, the Newey-West *t* on "
           "the mean return difference, the era cut, the costed and levered timers, and the "
           "20-seed synthetic control. All numbers frozen from `docs/results.md`."),
        code("R = %r" % (R,)),
        md("## The race — risk-parity vs cap-weight SPY, excess of cash\n\n"
           "An unlevered risk-parity book is *expected* to earn less than a tech-heavy cap-weight "
           "index, so the fair test is the risk-adjusted Sharpe (and the drawdown), not raw return."),
        code(
            "for tag,rp_s,rp_a,rp_v,rp_d,d,ci,p,t in [\n"
            "  ('11-sector 2018-26', R['e_rp_sharpe'],R['e_rp_ann'],R['e_rp_vol'],R['e_rp_dd'],R['e_diff'],R['e_ci'],R['e_pneg'],R['e_nwt']),\n"
            "  ('9-sector  2007-26', R['n_rp_sharpe'],R['n_rp_ann'],R['n_rp_vol'],R['n_rp_dd'],R['n_diff'],R['n_ci'],R['n_pneg'],R['n_nwt'])]:\n"
            "    print(f\"{tag}: RP Sharpe {rp_s:.3f} (ann {rp_a:+.1f}% vol {rp_v:.1f}% DD {rp_d:.0f}%)\")\n"
            "    print(f\"    Sharpe diff vs SPY {d:+.3f}  95% CI {ci}  P(diff<0)={p:.2f}  NW t(ret diff)={t:+.2f}\")\n"
            "print(f\"SPY (11-sec window) Sharpe {R['e_spy_sharpe']:.3f}; SPY (9-sec window) Sharpe {R['n_spy_sharpe']:.3f}\")\n"
            "print(f\"ERC variant: 11-sec Sharpe {R['e_erc_sharpe']:.3f} (diff {R['e_erc_diff']:+.3f}), 9-sec {R['n_erc_sharpe']:.3f} (diff {R['n_erc_diff']:+.3f})\")"
        ),
        md("The bootstrap CI on the Sharpe difference **straddles zero** in every case, and the "
           "Newey-West *t* on the mean daily excess-return difference is small and negative — no "
           "statistically distinguishable Sharpe advantage on either panel."),
        md("## Era cut — the advantage is entirely regime-dependent"),
        code(
            "print(f\"{R['era_early']} (n={R['era_early_n']}): RP {R['era_early_rp']:.3f} vs SPY {R['era_early_spy']:.3f}  diff {R['era_early_diff']:+.3f}  (crisis era: RP wins)\")\n"
            "print(f\"{R['era_late']} (n={R['era_late_n']}): RP {R['era_late_rp']:.3f} vs SPY {R['era_late_spy']:.3f}  diff {R['era_late_diff']:+.3f}  (tech bull: RP loses)\")\n"
            "print('A genuine edge holds across sub-eras; this one flips sign — the signature of diversification, not alpha.')"
        ),
        md("## The timer — can you get paid for it?\n\n"
           "Costs first: quarterly rebalancing turns over ~half the book a year, so at 3 bps "
           "one-way the drag is ~1–2 bps/yr — trivial. The problem is not costs; it is that the "
           "unlevered book earns *less* than SPY. Levering to SPY's vol to chase a return edge "
           "just reproduces SPY, because the Sharpe was never higher."),
        code(
            "print(f\"costs: turnover ~{R['n_turn']}%/yr x 3bps one-way = {R['n_cost']:.1f} bps/yr drag (negligible)\")\n"
            "print(f\"levered {R['lev_L']:.2f}x to SPY vol: Sharpe {R['lev_sharpe']:.3f} vs SPY {R['lev_spy_sharpe']:.3f}; \"\n"
            "      f\"ann {R['lev_ann']:+.1f}% vs {R['lev_spy_ann']:+.1f}% (financing {R['lev_fin']:.1f} bps/yr, maxDD {R['lev_dd']:.0f}%)\")\n"
            "print('No free lunch: the leverage route lands right back on SPY.')"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: inverse-vol must out-Sharpe the concentrated cap-weight benchmark ONLY when "
           "the assets' vols are dispersed, and tie when they are equal."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from sector_rp import data, strategy as st\n"
            "nulls = np.array([st.synthetic_detect(data.synthetic_world(vol_spread=0.0, seed=890+s))['sharpe_advantage'] for s in range(8)])\n"
            "print(f\"null (vol_spread=0), 8 seeds: mean advantage {nulls.mean():+.4f} (sd {nulls.std(ddof=1):.4f}), |adv|>0.1 in {(np.abs(nulls)>0.1).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_world(vol_spread=0.02, seed=890))\n"
            "print(f\"planted (vol_spread=0.02): advantage {planted['sharpe_advantage']:+.3f} (RP {planted['sr_rp']:.2f} vs cap-weight {planted['sr_bench']:.2f})\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Mixed.** The claim splits in two. The **drawdown / vol diversification "
           f"is real and robust**: over 2007–2026 inverse-vol cut volatility to "
           f"{R['n_rp_vol']:.1f}% (SPY {R['n_spy_vol']:.1f}%) and max drawdown to {R['n_rp_dd']:.0f}% "
           f"(SPY {R['n_spy_dd']:.0f}%), beating SPY by {R['cy_2008']:+.1f}pp in 2008 and "
           f"{R['cy_2022']:+.1f}pp in 2022. But the **excess-of-cash Sharpe advantage does not "
           f"clear the bar**: it is {R['n_diff']:+.3f} on the long panel (bootstrap 95% CI "
           f"{R['n_ci']} straddles zero, NW *t* = {R['n_nwt']:+.2f}), {R['e_diff']:+.3f} on the "
           f"2018–2026 panel, and it *flips sign* across eras ({R['era_early_diff']:+.3f} then "
           f"{R['era_late_diff']:+.3f}). Real risk reduction, no risk-adjusted edge. The 20-seed "
           f"synthetic control recovers a *planted* advantage cleanly (fires on {R['null_fire']}/20 "
           f"nulls). *Short history on the eleven-sector panel (XLC from 2018-06) is named here.*\n"
           f"- **Tradability — Mirage.** Costs are trivial ({R['n_cost']:.1f} bps/yr), but there is "
           f"no Sharpe edge to harvest: unlevered you earn *less* than SPY ({R['n_rp_ann']:+.1f}% vs "
           f"{R['n_spy_ann']:+.1f}%) for the smoother ride, and levering to SPY's vol reproduces "
           f"SPY ({R['lev_sharpe']:.2f} vs {R['lev_spy_sharpe']:.2f} Sharpe). The promised "
           f"risk-adjusted pickup is a mirage."),
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
