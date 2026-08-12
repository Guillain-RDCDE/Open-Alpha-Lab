"""Generate the two narrative notebooks for Study 880 (Aggregate Short Interest).

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


# Frozen real-tape headline numbers — mirror of docs/results.md. FINRA consolidated
# short interest, 50-name liquid panel, equal-weight mean days-to-cover index, 205
# bi-monthly settlement dates 2017-12-29 -> 2026-06-30; SPY total-return. Detrended-log
# index -> forward-SPY-return predictive regression, one publication lag.
R = dict(
    start="2017-12-29", end="2026-06-30", n_dates=205, n_panel=50,
    fingerprint="4ab933bfbc01", last_index=2.146, median_names=50,
    # headline horizon = 1 bi-monthly period (~2 weeks)
    h1_n=203, h1_beta=-17.2, h1_t=-0.66, h1_r2=0.28, h1_fwd=55,
    h2_n=202, h2_beta=-12.8, h2_t=-0.26, h2_r2=0.08, h2_fwd=110,
    h3_n=201, h3_beta=-35.5, h3_t=-0.50, h3_r2=0.41, h3_fwd=168,
    h6_n=198, h6_beta=-101.3, h6_t=-0.70, h6_r2=2.07, h6_fwd=338,
    terc_lo=67, terc_hi=21, terc_welch=-0.75, terc_n=68,
    placebo_obs=-17.22, placebo_mean=-0.13, placebo_sd=22.48, placebo_p=0.2170,
    era_e_n=107, era_e_beta=-21.4, era_e_t=-0.60, era_e_r2=0.50,
    era_l_n=96, era_l_beta=-13.2, era_l_t=-0.34, era_l_r2=0.13,
    timer1_net=35.3, timer1_ann=8.5, timer1_sharpe=0.71, timer1_sw=70,
    timer5_net=34.0, timer5_ann=8.2, timer5_sharpe=0.68, bh_ann=13.2,
    null_mean_t=0.47, null_sd_t=0.72, null_fire=1, null_100_mean=0.02,
    planted_beta=-103.8, planted_t=-4.10, planted_r2=7.0,
)


HEADER = f"""# Study 880 — Aggregate Short Interest 🐻

**Is market-wide short interest "arguably the strongest known predictor" of the market?**

Rapach, Ringgenberg & Zhou (2016) build a **market-level** short-interest index and
find it predicts the aggregate equity return with a strong **negative** slope: when
short sellers crowd the whole tape, forward market returns are *lower*. We rebuild the
aggregate index from the **FINRA consolidated short-interest** file (the official
bi-monthly, settlement-date report) for a liquid {R['n_panel']}-name panel — the
equal-weight average **days-to-cover** — and run the predictive regression of forward
SPY returns on its detrended level ({R['start']} → {R['end']}, {R['n_dates']} bi-monthly
prints).

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Availability: aggregate SI is bi-monthly (24/yr) with an ~8-day
publication lag — not a daily series. Index is a days-to-cover average (FINRA has no
shares-outstanding), and the panel is current-membership mega-caps.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea\n\n"
           "Short sellers are, on average, *informed* — so when they pile into the whole "
           "market at once (aggregate short interest spikes), it should be a bearish tell "
           "for the market as a whole. Rapach-Ringgenberg-Zhou call the detrended aggregate "
           "short-interest index the single strongest predictor of the market return, with "
           "a *negative* slope: high aggregate SI → lower forward return. We test exactly "
           "that regression on a modern, bi-monthly FINRA-built index."),
        code(
            "R = dict(h1_beta=%r, h1_t=%r, h1_r2=%r, terc_lo=%r, terc_hi=%r, terc_welch=%r)\n"
            "print('predictive slope (forward SPY return on detrended aggregate SI):')\n"
            "print('  beta = %%+.1f bps per 1sigma of the index   NW t = %%+.2f   R2 = %%.2f%%%%'\n"
            "      %% (R['h1_beta'], R['h1_t'], R['h1_r2']))\n"
            "print('  high-SI periods forward mean %%+d bps  vs low-SI %%+d bps  (Welch t = %%+.2f)'\n"
            "      %% (R['terc_hi'], R['terc_lo'], R['terc_welch']))"
            % (R["h1_beta"], R["h1_t"], R["h1_r2"], R["terc_lo"], R["terc_hi"], R["terc_welch"])
        ),
        md("## 2. Is the machinery honest? A live synthetic control\n\n"
           "We plant the RRZ effect in a seeded toy world (`edge>0`, high detrended SI "
           "depresses the next period's return) and check the regression recovers it — and "
           "stays *silent* on the null (`edge=0`). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from agg_short import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_frame(edge=0.0, seed=880, n_periods=200))\n"
            "planted = st.synthetic_detect(data.synthetic_frame(edge=0.015, seed=880, n_periods=200))\n"
            "print('null world   : slope NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: slope NW t = %+.2f  (should light up negative)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — right sign, no significance\n\n"
           f"On this modern bi-monthly tape the slope has the **correct RRZ sign** "
           f"(**{R['h1_beta']:+.1f} bps** of forward SPY return per 1σ of the index, and it "
           f"stays negative at every horizon out to 3 months) — but it is **nowhere near "
           f"significant**: NW *t* = **{R['h1_t']:+.2f}**, R² = **{R['h1_r2']:.2f}%**, and a "
           f"5,000-draw permutation places the observed slope only at *p* = {R['placebo_p']:.2f} "
           f"in the left tail. The high-SI periods do earn a little less "
           f"({R['terc_hi']:+d} vs {R['terc_lo']:+d} bps) but the gap is a coin-flip "
           f"(Welch *t* = {R['terc_welch']:+.2f}). The synthetic control recovers a *planted* "
           f"relation cleanly, so the flat real result is genuine, not a broken engine — the "
           f"celebrated aggregate-SI predictor simply does **not** show up on a 2017–2026 "
           f"mega-cap days-to-cover index. **Signal: None** (directionally consistent, "
           f"statistically absent), **Tradability: Mirage** (the de-risk-on-crowded-shorts "
           f"overlay just missed the bull market — {R['timer1_ann']:+.1f}%/yr vs "
           f"{R['bh_ann']:+.1f}%/yr buy-and-hold)."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 880 — Aggregate Short Interest — the teardown\n\n"
           "The horizon sweep, the Newey-West slope *t*, the high/low tercile split, the "
           "5,000-draw permutation placebo, the two-era cut, the costed timing overlay, and "
           "the 20-seed synthetic control. FINRA consolidated short interest, 50-name liquid "
           "panel, equal-weight mean days-to-cover; SPY total-return."),
        code("R = %r" % (R,)),
        md("## The headline — forward-SPY-return regression on the detrended index\n\n"
           "One publication lag (signal at settlement `t` acted on the next settlement "
           "`t+1`). Horizons in bi-monthly settlement periods (~0.5 mo each). RRZ predict "
           "beta < 0."),
        code(
            "for h in ('h1','h2','h3','h6'):\n"
            "    print(f\"H={h[1:]:>1} (~{int(h[1:])*0.5:.1f} mo): n={R[h+'_n']:3d}  \"\n"
            "          f\"beta={R[h+'_beta']:+7.1f} bps/sigma  NW t={R[h+'_t']:+.2f}  \"\n"
            "          f\"R2={R[h+'_r2']:+.2f}%  fwd mean={R[h+'_fwd']:+d} bps\")"
        ),
        md("## High vs low short-interest tercile — forward one-period SPY return"),
        code(
            "print(f\"low-SI tercile  {R['terc_lo']:+d} bps  (n={R['terc_n']})\")\n"
            "print(f\"high-SI tercile {R['terc_hi']:+d} bps  (n={R['terc_n']})  \"\n"
            "      f\"Welch t(high-low) = {R['terc_welch']:+.2f}\")"
        ),
        md("## Placebo — permute forward returns vs the index (5,000 draws)"),
        code(
            "print(f\"observed beta {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.2f} \"\n"
            "      f\"(sd {R['placebo_sd']:.2f}) -> left-tail p = {R['placebo_p']:.4f}\")"
        ),
        md("## Robustness — two eras (split 2022-06-01)"),
        code(
            "print(f\"2017-12 -> 2022-05 (n={R['era_e_n']}): beta={R['era_e_beta']:+.1f} bps  NW t={R['era_e_t']:+.2f}  R2={R['era_e_r2']:.2f}%\")\n"
            "print(f\"2022-06 -> 2026-06 (n={R['era_l_n']}): beta={R['era_l_beta']:+.1f} bps  NW t={R['era_l_t']:+.2f}  R2={R['era_l_r2']:.2f}%\")"
        ),
        md("## The timer — de-risk to cash when shorts are crowded (`sii>0`)\n\n"
           "One-way cost × NAV per switch. The overlay *underperforms* buy-and-hold — "
           "sitting out on crowded-short readings just gave up equity premium."),
        code(
            "for tag,net,ann,sh in [('1 bp',R['timer1_net'],R['timer1_ann'],R['timer1_sharpe']),\n"
            "                       ('5 bps',R['timer5_net'],R['timer5_ann'],R['timer5_sharpe'])]:\n"
            "    print(f\"{tag:>5}: overlay net {net:+.1f} bps/period ({ann:+.1f}%/yr, Sharpe {sh:.2f})\")\n"
            "print(f\"buy-and-hold: {R['bh_ann']:+.1f}%/yr  (the overlay loses to it)\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted relation."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from agg_short import data, strategy as st\n"
            "nt = np.array([st.synthetic_detect(data.synthetic_frame(edge=0.0, seed=880+s, n_periods=200))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: slope NW t mean {nt.mean():+.2f} (sd {nt.std(ddof=1):.2f}), |t|>=2 in {(abs(nt)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_frame(edge=0.015, seed=880, n_periods=200))\n"
            "print(f\"planted (edge=0.015): beta = {planted['beta']*1e4:+.1f} bps/sigma, NW t = {planted['t_nw']:+.2f}, R2 = {planted['r2']*100:.1f}%\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The RRZ aggregate-short-interest predictor does **not** "
           f"replicate on a 2017–2026 FINRA-built, mega-cap, days-to-cover index. The "
           f"forward-return slope has the **right sign** ({R['h1_beta']:+.1f} bps/σ, negative "
           f"at all four horizons) but NW *t* = **{R['h1_t']:+.2f}** (R² {R['h1_r2']:.2f}%); "
           f"both eras agree in sign yet neither is significant (*t* = {R['era_e_t']:+.2f} / "
           f"{R['era_l_t']:+.2f}); the permutation placebo puts it at *p* = {R['placebo_p']:.2f}. "
           f"The 20-seed synthetic control fires on the planted world (*t* = {R['planted_t']:+.2f}) "
           f"and stays quiet on the null, so the flat real result is genuine.\n"
           f"- **Tradability — Mirage.** The de-risk-on-crowded-shorts overlay earns "
           f"{R['timer1_ann']:+.1f}%/yr net (1 bp) — *below* buy-and-hold's {R['bh_ann']:+.1f}%/yr; "
           f"a weak, wrong-way-for-a-bull-market timing rule is a paycheck mirage.\n\n"
           f"*Caveats travelling with every number: aggregate SI is bi-monthly with an "
           f"~8-day publication lag; the index is a days-to-cover average not the paper's "
           f"shares-outstanding ratio; the panel is current-membership mega-caps "
           f"(survivorship, named on the Signal axis); and the sample is short (~8.5 years, "
           f"a mostly-bull era) versus RRZ's 1973–2014.*"),
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
