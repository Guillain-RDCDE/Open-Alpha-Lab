"""Generate the two narrative notebooks for Study 891 (Insurance Float Engine).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance total-return,
# KIE/IAK/SPY/KBE/BIL, 229 months 2007-06 -> 2026-06, as-of 2026-06-30, fp 3a54dbc09ab6).
R = dict(
    start="2007-06-30", end="2026-06-30", n=229, fp="3a54dbc09ab6",
    kie_cagr=7.86, kie_vol=21.73, kie_sharpe=0.398, kie_dd=-69.7,
    iak_cagr=6.80, iak_vol=20.85, iak_sharpe=0.359, iak_dd=-72.2,
    spy_cagr=10.68, spy_vol=15.58, spy_sharpe=0.644, spy_dd=-50.8,
    kbe_cagr=3.21, kbe_vol=27.64, kbe_sharpe=0.209, kbe_dd=-76.6,
    kie_adv=-0.246, kie_diff=-1.39, kie_tdiff=-0.49,
    iak_adv=-0.285, iak_diff=-2.56, iak_tdiff=-0.90,
    kie_ci=(-0.060, 0.966), iak_ci=(-0.116, 0.949), spy_ci=(0.180, 1.182),
    kie_capm_a=-2.48, kie_capm_ta=-0.84, kie_capm_b=1.109,
    iak_capm_a=-3.10, iak_capm_ta=-0.95, iak_capm_b=1.053,
    kie_two_a=-0.11, kie_two_ta=-0.04, kie_two_load=0.357, kie_two_tload=6.27,
    iak_two_a=-0.96, iak_two_ta=-0.34, iak_two_load=0.322, iak_two_tload=5.81,
    kie_kbe=2.89, kie_kbe_t=0.96, iak_kbe=1.71, iak_kbe_t=0.58,
    era_gfc=0.118, era_1015=-0.125, era_1620=-0.374, era_2126=-0.157, era_post=-0.207,
    rot_net=9.51, rot_sharpe=0.490, rot_mkt=11.41, rot_ins=10.03, rot_switches=30,
    iso_gross=-1.39, iso_tg=-0.49, iso_net=-1.89, iso_tn=-0.66, iso_charge=0.50,
    syn_null_adv=0.092, syn_null_ta=1.48, syn_edge_adv=0.305, syn_edge_ta=4.34,
    syn_load=0.387, syn_tload=10.70,
)


HEADER = f"""# Study 891 — Insurance Float Engine 🛡️

**P&C insurers earn on "float" — premiums held before claims are paid. It made Buffett.
Does a plain basket of insurers turn it into a market-beating edge?**

A property-and-casualty insurer collects your premium today and pays the claim later; the money
in between — the **float** — it invests for itself. Near-zero-cost leverage: the compounding
engine behind Berkshire. The folklore extrapolates: *so a broad insurer basket must be a quiet,
structurally-advantaged compounder.* We race two liquid wrappers — **KIE** (SPDR S&P Insurance,
equal-weight) and **IAK** (iShares U.S. Insurance) — against **SPY**, both **excess-of-cash**
(minus **BIL**), over {R['n']} months ({R['start']} → {R['end']}), with **KBE** (banks) as the
control that asks: *float premium, or just financial-sector beta?*

*Numbers below are the frozen headline (`docs/results.md`, fingerprint `{R['fp']}`); the live
cell runs the fast synthetic control. Short sample: 19 years, one of them the GFC — named on the
Signal axis.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The race, in one table\n\n"
           "Both legs are measured *excess-of-cash* (minus BIL) before we annualise the Sharpe, "
           "so a high short rate can't flatter anyone. If the float were a structural edge, the "
           "insurer baskets should out-Sharpe the market. They don't — they trail it, at higher "
           "volatility and a deeper drawdown."),
        code(
            "R = " + repr(R) + "\n"
            "print('basket   CAGR    vol   exSharpe   maxDD')\n"
            "for t in ['kie','iak','spy','kbe']:\n"
            "    print(f\"{t.upper():5s} {R[t+'_cagr']:6.2f}% {R[t+'_vol']:6.2f}% \"\n"
            "          f\"  {R[t+'_sharpe']:+.3f}   {R[t+'_dd']:6.1f}%\")\n"
            "print()\n"
            "print(f\"KIE vs SPY: Sharpe advantage {R['kie_adv']:+.3f}  \"\n"
            "      f\"(KIE-SPY {R['kie_diff']:+.2f}%/yr, HAC t={R['kie_tdiff']:+.2f})\")\n"
            "print(f\"IAK vs SPY: Sharpe advantage {R['iak_adv']:+.3f}  \"\n"
            "      f\"(IAK-SPY {R['iak_diff']:+.2f}%/yr, HAC t={R['iak_tdiff']:+.2f})\")"
        ),
        md("## 2. Where did the 'edge' go? It was financial-sector beta all along\n\n"
           "Regress the insurer's excess return on the market **plus** one financial-sector "
           "factor (banks minus market). If float were a real premium it would survive as a "
           "positive alpha. Instead the alpha vanishes to zero and the bank factor loads up hard:"),
        code(
            "print(f\"KIE CAPM alpha (market only): {R['kie_capm_a']:+.2f}%/yr (t={R['kie_capm_ta']:+.2f})\")\n"
            "print(f\"KIE alpha after adding the bank factor: {R['kie_two_a']:+.2f}%/yr \"\n"
            "      f\"(t={R['kie_two_ta']:+.2f})  <- collapses to zero\")\n"
            "print(f\"   ...while the bank-sector loading is {R['kie_two_load']:+.3f} \"\n"
            "      f\"(t={R['kie_two_tload']:+.2f})  <- big and significant\")\n"
            "print()\n"
            "print('Translation: the insurer basket IS financial-sector beta.')\n"
            "print('The float is real economics, but at the traded-basket level it is not a')\n"
            "print('distinct, market-beating premium.')"
        ),
        md("## 3. Is the machinery honest? A live synthetic control\n\n"
           "We plant a *real* +4 %/yr float edge in a seeded toy world (on top of the same "
           "market + bank beta) and check the detector recovers it — and stays silent on the "
           "null (edge = 0, only sector beta). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from insurance_float import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_world(edge_ann=0.0, seed=891))\n"
            "edge = st.synthetic_detect(data.synthetic_world(edge_ann=0.04, seed=891))\n"
            "print(f\"null  (no float edge): Sharpe adv {null['advantage']:+.3f}, \"\n"
            "      f\"CAPM alpha t={null['capm_t_alpha']:+.2f}  (should stay quiet)\")\n"
            "print(f\"planted +4%/yr edge  : Sharpe adv {edge['advantage']:+.3f}, \"\n"
            "      f\"CAPM alpha t={edge['capm_t_alpha']:+.2f}  (should light up)\")\n"
            "print(f\"bank loading detected in both: {edge['load_bank']:+.3f} \"\n"
            "      f\"(t={edge['t_load_bank']:+.2f}) — the confound is always found\")"
        ),
        md(f"## 4. The honest verdict\n\n"
           f"**Signal — None.** The claimed edge over the market is absent, and the sign runs the "
           f"wrong way: KIE/IAK excess Sharpe **{R['kie_sharpe']:.2f} / {R['iak_sharpe']:.2f}** "
           f"trails SPY's **{R['spy_sharpe']:.2f}**; the advantage is "
           f"**{R['kie_adv']:+.2f} / {R['iak_adv']:+.2f}** with the return difference "
           f"statistically zero, CAPM alpha is **negative**, and one financial-sector factor "
           f"drives the alpha to **{R['kie_two_a']:+.2f} %/yr (t = {R['kie_two_ta']:+.2f})**. "
           f"**Tradability — Mirage:** the long-insurer/short-market trade loses "
           f"(**{R['iso_net']:+.2f} %/yr** net), and the apparent engine is sector beta you can "
           f"rent more cheaply — and at a shallower drawdown — by just owning the market. "
           f"Buffett's float was real; a plain insurer basket does not inherit it. "
           f"*(One true aside: insurers did beat **banks** by {R['kie_kbe']:+.2f} %/yr — but only "
           f"at t = {R['kie_kbe_t']:.2f}, and that's a different claim.)*"),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 891 — Insurance Float Engine — the teardown\n\n"
           "The excess-vs-excess Sharpe race, the HAC *t* on the return difference, the bootstrap "
           "Sharpe CIs, the CAPM and the decisive two-factor decomposition, the era cut, the "
           "costed rotation/isolation trades, and the planted-edge synthetic control."),
        code("R = " + repr(R)),
        md("## The race — excess Sharpe (both legs minus BIL) + HAC *t* on the diff"),
        code(
            "print(f\"n = {R['n']} months  {R['start']} -> {R['end']}  fp {R['fp']}\")\n"
            "print(f\"KIE exSharpe {R['kie_sharpe']:+.3f}  vs SPY {R['spy_sharpe']:+.3f}  \"\n"
            "      f\"advantage {R['kie_adv']:+.3f}  | KIE-SPY {R['kie_diff']:+.2f}%/yr \"\n"
            "      f\"(HAC t={R['kie_tdiff']:+.2f})\")\n"
            "print(f\"IAK exSharpe {R['iak_sharpe']:+.3f}  vs SPY {R['spy_sharpe']:+.3f}  \"\n"
            "      f\"advantage {R['iak_adv']:+.3f}  | IAK-SPY {R['iak_diff']:+.2f}%/yr \"\n"
            "      f\"(HAC t={R['iak_tdiff']:+.2f})\")"
        ),
        md("## Bootstrap 95% CI on the excess Sharpe — you can't separate insurer from zero\n\n"
           "Circular block bootstrap. SPY's interval clears zero; the insurer intervals include it."),
        code(
            "for t in ['kie','iak','spy']:\n"
            "    lo,hi = R[t+'_ci']\n"
            "    print(f\"{t.upper()}: exSharpe {R[t+'_sharpe']:+.3f}  CI[{lo:+.3f}, {hi:+.3f}]\")"
        ),
        md("## CAPM — insurer excess on market excess (no alpha over the market)"),
        code(
            "print(f\"KIE: alpha {R['kie_capm_a']:+.2f}%/yr (t={R['kie_capm_ta']:+.2f})  beta {R['kie_capm_b']:.3f}\")\n"
            "print(f\"IAK: alpha {R['iak_capm_a']:+.2f}%/yr (t={R['iak_capm_ta']:+.2f})  beta {R['iak_capm_b']:.3f}\")"
        ),
        md("## The decisive test — add the bank-sector factor, and the alpha dies\n\n"
           "`insurer_ex = alpha + beta*market_ex + s*(bank_ex - market_ex)`. A float premium "
           "would survive; sector beta is absorbed."),
        code(
            "print(f\"KIE: alpha {R['kie_two_a']:+.2f}%/yr (t={R['kie_two_ta']:+.2f})  \"\n"
            "      f\"bank load {R['kie_two_load']:+.3f} (t={R['kie_two_tload']:+.2f})\")\n"
            "print(f\"IAK: alpha {R['iak_two_a']:+.2f}%/yr (t={R['iak_two_ta']:+.2f})  \"\n"
            "      f\"bank load {R['iak_two_load']:+.3f} (t={R['iak_two_tload']:+.2f})\")\n"
            "print()\n"
            "print('The insurer alpha over [market + bank factor] is indistinguishable from zero.')"
        ),
        md("## Within-financials — insurers DID beat banks (a different, weaker claim)"),
        code(
            "print(f\"KIE - KBE: {R['kie_kbe']:+.2f}%/yr (HAC t={R['kie_kbe_t']:+.2f})\")\n"
            "print(f\"IAK - KBE: {R['iak_kbe']:+.2f}%/yr (HAC t={R['iak_kbe_t']:+.2f})\")\n"
            "print('Real direction (float < spread-leverage in risk), but not t>=2 on 19 years.')"
        ),
        md("## Era cut — the deficit isn't one crisis; it's every calm era\n\n"
           "KIE minus SPY excess-Sharpe advantage by era. Positive only in the 2007-09 crash."),
        code(
            "for tag,v in [('GFC 2007-09',R['era_gfc']),('2010-15',R['era_1015']),\n"
            "              ('2016-20',R['era_1620']),('2021-26',R['era_2126']),\n"
            "              ('post-GFC 2010+',R['era_post'])]:\n"
            "    flag = '  <- only positive era (fell less in the crash)' if v>0 else ''\n"
            "    print(f\"{tag:16s} KIE-SPY advantage {v:+.3f}{flag}\")"
        ),
        md("## Tradability — nothing to pocket over the market"),
        code(
            "print(f\"(a) 1-month-lag rotation (own KIE when it's led SPY 12m): net {R['rot_net']:+.2f}%/yr\")\n"
            "print(f\"    vs always-SPY {R['rot_mkt']:+.2f}%/yr  always-KIE {R['rot_ins']:+.2f}%/yr  \"\n"
            "      f\"({R['rot_switches']} switches) -> underperforms just owning the market\")\n"
            "print(f\"(b) isolation (long KIE / short SPY, borrow+costs {R['iso_charge']:.2f}%/yr): \"\n"
            "      f\"gross {R['iso_gross']:+.2f}%/yr (t={R['iso_tg']:+.2f}) -> \"\n"
            "      f\"net {R['iso_net']:+.2f}%/yr (t={R['iso_tn']:+.2f})\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted +4 %/yr edge."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from insurance_float import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_world(edge_ann=0.0, seed=891))\n"
            "edge = st.synthetic_detect(data.synthetic_world(edge_ann=0.04, seed=891))\n"
            "print(f\"null  : Sharpe adv {null['advantage']:+.3f}, CAPM alpha t={null['capm_t_alpha']:+.2f}, \"\n"
            "      f\"two-factor alpha t={null['two_t_alpha']:+.2f}  (all quiet)\")\n"
            "print(f\"planted: Sharpe adv {edge['advantage']:+.3f}, CAPM alpha t={edge['capm_t_alpha']:+.2f}, \"\n"
            "      f\"two-factor alpha t={edge['two_t_alpha']:+.2f}  (recovered, survives bank control)\")\n"
            "print(f\"bank loading always found: {edge['load_bank']:+.3f} (t={edge['t_load_bank']:+.2f})\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** No edge over the market attributable to float — the sign runs "
           f"the wrong way. KIE/IAK excess Sharpe {R['kie_sharpe']:.2f}/{R['iak_sharpe']:.2f} vs "
           f"SPY {R['spy_sharpe']:.2f}; advantage {R['kie_adv']:+.2f}/{R['iak_adv']:+.2f} (HAC "
           f"*t* on the diff {R['kie_tdiff']:+.2f}/{R['iak_tdiff']:+.2f}); CAPM alpha "
           f"{R['kie_capm_a']:+.1f}/{R['iak_capm_a']:+.1f} %/yr; and the two-factor alpha "
           f"collapses to {R['kie_two_a']:+.2f} %/yr (*t* = {R['kie_two_ta']:+.2f}) against a "
           f"financial-sector loading of {R['kie_two_load']:+.2f} (*t* = {R['kie_two_tload']:.1f}). "
           f"Negative in every calm era; bootstrap can't clear zero. Short 19-year sample, one "
           f"crisis inside it. (Insurers > banks by {R['kie_kbe']:+.1f} %/yr at *t* = "
           f"{R['kie_kbe_t']:.2f} — a different, unclean claim.)\n"
           f"- **Tradability — Mirage.** The isolation trade nets {R['iso_net']:+.2f} %/yr; the "
           f"rotation nets {R['rot_net']:+.1f} %/yr below always-SPY's {R['rot_mkt']:.1f} %/yr. "
           f"The engine is sector beta — cheaper, and shallower-drawdown, as plain SPY."),
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
