"""Generate the two narrative notebooks for Study 906 (EM Local Bonds, FX-Hedged).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance total-return
# closes; EMLC/LEMB/EBND local-EM, EMB USD-EM, UUP overlay, BIL cash; 2010-08 -> 2026-06-30).
R = dict(
    start="2010-08-31", end="2026-06-30", n=191, fingerprint="32605dae0c9c",
    emlc_uup_beta=-1.12, t_emlc_uup=-11.91, emlc_uup_r2=0.50, hedge_b=-1.118,
    unhedged_exc=0.33, unhedged_sharpe=0.03, t_unhedged=0.14,
    hedged_exc=1.62, hedged_sharpe=0.20, t_hedged=0.94,
    emb_exc=3.10, emb_sharpe=0.34, t_emb=1.48,
    prem_diff=-1.48, t_prem_diff=-0.81, welch=-0.49,
    lemb_sharpe=0.23, lemb_t=0.99, lemb_prem=-1.03,
    ebnd_sharpe=0.26, ebnd_t=1.16, ebnd_prem=-1.33,
    boot_unh_lo=-0.38, boot_unh_hi=0.47,
    boot_hed_lo=-0.21, boot_hed_hi=0.69, boot_hed_fracneg=0.17,
    boot_emb_lo=-0.05, boot_emb_hi=0.91,
    wf_exc=1.60, wf_sharpe=0.20, wf_t=0.79, wf_resid_fx=0.10,
    era_early_hed=1.76, era_early_t=0.70, era_early_prem=-3.58, era_early_prem_t=-1.87, era_early_n=125,
    era_late_hed=1.34, era_late_t=0.84, era_late_prem=2.50, era_late_prem_t=0.71, era_late_n=66,
    dd_emlc=-32.3, dd_emb=-28.7, dd_hedged=-17.1,
    cost_charge=0.46, cost_gross=1.62, cost_net=1.15, cost_net_t=0.67, cost_net_sharpe=0.15,
    cost_net_prem=-1.95, cost_net_prem_t=-1.07,
    planted_exc=5.53, planted_t=3.25, planted_b=-1.10,
    null_t_mean=-0.01, null_t_sd=1.41, null_fire=1, null_seeds=20,
)


HEADER = f"""# Study 906 — EM Local Bonds FX-Hedged 🌏

**EM local-currency bonds pay a fat local rate — but does stripping the FX leave a real,
harvestable carry, or does the boring USD-EM ETF still win?**

Emerging-market **local-currency** government bonds (EMLC, LEMB, EBND) yield far more than
their US-dollar cousins because they pay the *local* short rate (6–13 % in Brazil, Mexico,
Indonesia, South Africa). The catch: your USD return is `local_bond + EM_FX`, and the
currency leg is so volatile — and, over 2010–2026, so *negative* as the dollar rose — that
the carry disappears. The obvious fix: **hedge the FX**. No clean FX-hedged EM-local ETF
exists on US tape, so we use a **proxy**: a long **UUP** (dollar-index) overlay that gains
when the broad dollar rallies and EM currencies fall together.

*Numbers below are the frozen headline (`docs/results.md`, fingerprint `{R['fingerprint']}`);
the live cells run the fast synthetic control. **Proxy caveat:** UUP tracks the
developed-market DXY basket, not the EMLC currency basket — it strips only part of the EM-FX.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "An EM-local bond fund earns the local rate **plus** whatever the currency does in "
           "dollars. When the dollar is strong, EM currencies weaken together and drown the "
           "carry. A **long dollar-index overlay** (UUP) rises exactly then — so adding it "
           "back approximately cancels the currency drag, leaving (mostly) the local rate."),
        code("R = %r" % (R,)),
        code(
            "print(f\"raw EMLC over cash : {R['unhedged_exc']:+.2f}%/yr  Sharpe {R['unhedged_sharpe']:+.2f}  (the FX drowns it)\")\n"
            "print(f\"FX-stripped (hedged): {R['hedged_exc']:+.2f}%/yr  Sharpe {R['hedged_sharpe']:+.2f}  (the local carry surfaces)\")\n"
            "print(f\"but plain USD-EM EMB: {R['emb_exc']:+.2f}%/yr  Sharpe {R['emb_sharpe']:+.2f}  (the boring sibling still wins)\")"
        ),
        md("## 2. Is the machinery honest? A live synthetic control\n\n"
           "We plant a known local-rate carry in a seeded toy world (`carry>0`) under a "
           "dollar-explained FX drag, and check the overlay recovers it — and stays *silent* "
           "on the null (`carry=0`, FX present but no carry). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from em_hedged import data, strategy as st\n"
            "planted = st.synthetic_detect(data.synthetic_world(carry_annual=0.04, seed=906))\n"
            "null    = st.synthetic_detect(data.synthetic_world(carry_annual=0.0,  seed=906))\n"
            "print('planted carry world: hedged HAC t = %+.2f  (should light up)' % planted['t_hedged'])\n"
            "print('null (no carry)    : hedged HAC t = %+.2f  (should be ~0)'    % null['t_hedged'])"
        ),
        md("## 3. Where the hedge genuinely helps — the drawdown\n\n"
           f"Stripping the dollar move roughly **halves** the drawdown "
           f"(EMLC **{R['dd_emlc']:.0f}%** → hedged **{R['dd_hedged']:.0f}%**). That is a real, "
           f"mechanical diversification win — you sleep better. It just doesn't turn into a "
           f"significant *return* premium."),
        code(
            "print(f\"EMLC unhedged max drawdown : {R['dd_emlc']:.1f}%\")\n"
            "print(f\"hedged-EMLC max drawdown   : {R['dd_hedged']:.1f}%   <- the FX-strip cuts the pain\")\n"
            "print(f\"EMB (USD-EM) max drawdown  : {R['dd_emb']:.1f}%\")"
        ),
        md("## 4. The honest verdict — a real mechanism, no bankable edge\n\n"
           f"The FX-strip is **mechanically real** (EMLC is ~50 % dollar-basket FX; hedging "
           f"lifts the Sharpe {R['unhedged_sharpe']:+.2f} → {R['hedged_sharpe']:+.2f} and halves "
           f"the drawdown). **But the leftover local carry isn't a robust premium** — "
           f"**{R['hedged_exc']:+.2f} %/yr at HAC *t* = {R['t_hedged']:+.2f}**, a bootstrap "
           f"Sharpe CI of [{R['boot_hed_lo']:+.2f}, {R['boot_hed_hi']:+.2f}] that straddles "
           f"zero — and it **loses to just owning USD-EM debt** (hedged − EMB = "
           f"**{R['prem_diff']:+.2f} %/yr**). After the overlay's cost the gap only widens. "
           f"**Signal: Weak. Tradability: Mirage** — the simpler EMB wins outright."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 906 — EM Local Bonds FX-Hedged — the teardown\n\n"
           "The EMLC~UUP hedge regression, the excess-vs-excess race, HAC *t*'s, the "
           "circular-block bootstrap Sharpe CIs, the walk-forward hedge and its residual "
           "EM-FX beta, the era split, drawdowns, the costed overlay, and the planted-carry "
           "synthetic control."),
        code("R = %r" % (R,)),
        md("## The hedge — EMLC is half dollar-basket FX\n\n"
           "Regress EMLC excess on the UUP overlay excess: the slope is the FX exposure; the "
           "variance-min hedge ratio `b` is that slope (negative ⇒ a **long-UUP** overlay)."),
        code(
            "print(f\"EMLC = a + b*UUP :  beta {R['emlc_uup_beta']:+.2f} (HAC t {R['t_emlc_uup']:+.2f})  R2 {R['emlc_uup_r2']:.2f}\")\n"
            "print(f\"hedge ratio b   :  {R['hedge_b']:+.3f}  -> a long dollar-index overlay of |b| x NAV\")"
        ),
        md("## The race — excess-vs-excess (minus BIL cash)"),
        code(
            "print(f\"unhedged EMLC : {R['unhedged_exc']:+.2f}%/yr  Sharpe {R['unhedged_sharpe']:+.2f}  (HAC t {R['t_unhedged']:+.2f})\")\n"
            "print(f\"hedged   EMLC : {R['hedged_exc']:+.2f}%/yr  Sharpe {R['hedged_sharpe']:+.2f}  (HAC t {R['t_hedged']:+.2f})\")\n"
            "print(f\"EMB (USD-EM)  : {R['emb_exc']:+.2f}%/yr  Sharpe {R['emb_sharpe']:+.2f}  (HAC t {R['t_emb']:+.2f})\")\n"
            "print(f\"hedged - EMB  : {R['prem_diff']:+.2f}%/yr  (HAC t {R['t_prem_diff']:+.2f}, Welch {R['welch']:+.2f})  <- NEGATIVE\")\n"
            "print(f\"confirms: LEMB Sharpe {R['lemb_sharpe']:+.2f} (t {R['lemb_t']:+.2f}), EBND {R['ebnd_sharpe']:+.2f} (t {R['ebnd_t']:+.2f})\")"
        ),
        md("## Bootstrap Sharpe CIs (circular block) — does the carry clear zero?"),
        code(
            "print(f\"unhedged EMLC : Sharpe {R['unhedged_sharpe']:+.2f}  95% CI [{R['boot_unh_lo']:+.2f}, {R['boot_unh_hi']:+.2f}]\")\n"
            "print(f\"hedged   EMLC : Sharpe {R['hedged_sharpe']:+.2f}  95% CI [{R['boot_hed_lo']:+.2f}, {R['boot_hed_hi']:+.2f}]  frac<0 {R['boot_hed_fracneg']:.2f}  <- straddles 0\")\n"
            "print(f\"EMB (USD-EM)  : Sharpe {R['emb_sharpe']:+.2f}  95% CI [{R['boot_emb_lo']:+.2f}, {R['boot_emb_hi']:+.2f}]\")"
        ),
        md("## Walk-forward hedge (36m rolling b, lag 1) — no look-ahead\n\n"
           "The in-sample `b` is an FX-strip upper bound; the implementable rolling hedge "
           "gives the same thin result and shows the residual EM-FX the DXY proxy can't reach."),
        code(
            "print(f\"walk-forward hedged: {R['wf_exc']:+.2f}%/yr  Sharpe {R['wf_sharpe']:+.2f}  (HAC t {R['wf_t']:+.2f})\")\n"
            "print(f\"residual EM-FX beta left by the proxy: {R['wf_resid_fx']:+.2f}  (the DXY basket != the EM basket)\")"
        ),
        md("## Robustness — two eras (split 2021-01-01)"),
        code(
            "print(f\"2010-2020 (n={R['era_early_n']}): hedged {R['era_early_hed']:+.2f}%/yr (t {R['era_early_t']:+.2f})  prem-vs-EMB {R['era_early_prem']:+.2f}% (t {R['era_early_prem_t']:+.2f})\")\n"
            "print(f\"2021-2026 (n={R['era_late_n']}): hedged {R['era_late_hed']:+.2f}%/yr (t {R['era_late_t']:+.2f})  prem-vs-EMB {R['era_late_prem']:+.2f}% (t {R['era_late_prem_t']:+.2f})\")"
        ),
        md("## The timer — cost the overlay"),
        code(
            "print(f\"overlay charge {R['cost_charge']:.2f}%/yr:  gross {R['cost_gross']:+.2f} -> net {R['cost_net']:+.2f}%/yr (t {R['cost_net_t']:+.2f}, Sharpe {R['cost_net_sharpe']:+.2f})\")\n"
            "print(f\"net premium vs EMB: {R['cost_net_prem']:+.2f}%/yr (t {R['cost_net_prem_t']:+.2f})  <- the USD-EM ETF dominates\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: recover a *planted* local carry, stay silent on the null. No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from em_hedged import data, strategy as st\n"
            "planted = st.synthetic_detect(data.synthetic_world(carry_annual=0.04, seed=906))\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_world(carry_annual=0.0, seed=906+s))['t_hedged'] for s in range(20)])\n"
            "print(f\"planted (carry=4%/yr): hedged {planted['hedged_exc_ann_pct']:+.2f}%/yr, HAC t {planted['t_hedged']:+.2f}\")\n"
            "print(f\"null (carry=0), 20 seeds: HAC t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/20\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — WEAK.** The FX-strip is a **real mechanism** — EMLC is ~50 % "
           f"dollar-basket FX (β = {R['emlc_uup_beta']:+.2f}, HAC *t* = {R['t_emlc_uup']:+.2f}), "
           f"hedging lifts the excess-of-cash Sharpe {R['unhedged_sharpe']:+.2f} → "
           f"{R['hedged_sharpe']:+.2f} and halves the drawdown ({R['dd_emlc']:.0f}% → "
           f"{R['dd_hedged']:.0f}%), the same on LEMB/EBND. **But the residual local carry is "
           f"not robust**: {R['hedged_exc']:+.2f} %/yr at HAC *t* = {R['t_hedged']:+.2f}, a "
           f"bootstrap Sharpe CI [{R['boot_hed_lo']:+.2f}, {R['boot_hed_hi']:+.2f}] straddling "
           f"zero, and it **loses to USD-EM debt** (hedged − EMB = {R['prem_diff']:+.2f} %/yr). "
           f"The 20-seed synthetic control recovers a planted carry (*t* = {R['planted_t']:+.2f}) "
           f"and fires on {R['null_fire']}/{R['null_seeds']} nulls, so the thin result is a true "
           f"small edge, not a biased estimator. Short ~15-year sample, one dollar super-cycle.\n"
           f"- **Tradability — MIRAGE.** Even gross the hedged carry (Sharpe "
           f"{R['hedged_sharpe']:+.2f}) is dominated by plain EMB (Sharpe {R['emb_sharpe']:+.2f}); "
           f"after the {R['cost_charge']:.2f} %/yr overlay cost net is {R['cost_net']:+.2f} %/yr "
           f"(*t* {R['cost_net_t']:+.2f}) and the net premium vs EMB is {R['cost_net_prem']:+.2f} "
           f"%/yr. A cheaper, simpler ETF wins — the local-hedged edge is a mirage."),
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
