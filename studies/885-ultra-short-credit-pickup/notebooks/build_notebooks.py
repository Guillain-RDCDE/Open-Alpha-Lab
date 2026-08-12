"""Generate the two narrative notebooks for Study 885 (Ultra-Short Credit Pickup).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted
from the frozen ``R`` dict (mirroring docs/results.md); the live cells run only the
fast synthetic control, so execution is quick and network-free.
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
# total-return closes; JPST/ICSH/MINT vs BIL/SHV; common 5-ETF sample 2017-05-22 ->
# 2026-06-30, 2289 days; as-of 2026-06-30; fingerprint 22e1cddb739d).
R = dict(
    fingerprint="22e1cddb739d", asof="2026-06-30",
    n_days=2289, start="2017-05-22", end="2026-06-30", years=9.08,
    # excess-of-BIL Sharpe + raw stats
    jpst_sharpe=0.61, icsh_sharpe=0.54, mint_sharpe=0.40, shv_sharpe=0.11,
    jpst_ret=3.00, icsh_ret=2.94, mint_ret=2.81, bil_ret=2.41, shv_ret=2.44,
    jpst_vol=0.93, icsh_vol=0.97, mint_vol=0.98, bil_vol=0.25, shv_vol=0.27,
    # the sleeve pickup (EW credit) minus BIL
    sleeve_bps=49.8, sleeve_t=1.30, sleeve_sharpe=0.62, sleeve_lags=8,
    jpst_bps=57.9, jpst_pt=1.60, icsh_bps=52.0, icsh_pt=1.46, mint_bps=39.5, mint_pt=0.84,
    shv_bps=2.8,
    # bootstrap CI on the sleeve excess Sharpe
    ci_lo=-0.26, ci_hi=2.21, ci_fracneg=0.091, ci_block=13,
    # sub-eras (pickup)
    early_bps=49.2, early_t=3.21, early_n=406,
    late_bps=49.9, late_t=1.08, late_n=1883, welch_t=-0.02,
    # drawdowns (common sample)
    bil_dd=-0.21, jpst_dd=-3.28, icsh_dd=-3.94, mint_dd=-4.62,
    # stress windows
    y2022_bil=1.40, y2022_jpst=1.14, y2022_icsh=0.96, y2022_mint=-1.01,
    covid_bil=0.26, covid_jpst=-2.75, covid_icsh=-3.55, covid_mint=-4.34,
    # MINT's long 2009-> history
    mint_long_bps=77.0, mint_long_t=2.82, mint_long_sharpe=0.89,
    mint_long_cilo=0.18, mint_long_cihi=1.92, mint_long_n=4177, mint_long_years=16.6,
    mint_long_early_t=7.44, mint_long_late_t=0.73,
    # costs
    cost1_net=47.8, cost1_sharpe=0.59, cost2_net=45.8, cost5_net=39.8,
    # synthetic control (multi-seed, 12 seeds)
    syn_null_tmean=0.61, syn_null_fire="0/12",
    syn_plant=120.0, syn_plant_tmean=3.51, syn_plant_fire="11/12", syn_plant_recovered=120.0,
)


HEADER = f"""# Study 885 — Ultra-Short Credit Pickup 💵

**Do ultra-short investment-grade credit ETFs (JPST / ICSH / MINT) pay you a real,
near-riskless pickup over plain T-bills (BIL / SHV)?**

The pitch is mechanical: hold ~AA-/A short-maturity IG corporates and ABS instead of
pure bills, and you are paid a small **spread over cash** for a *sliver* of credit and
duration risk. If that pickup is a genuine structural premium, the credit sleeve should
earn a **higher excess-of-bills Sharpe** than bills — better reward per unit of risk —
while drawing down only marginally more. We test it on the live, fee-paying tape
({R['start']} → {R['end']}, {R['n_days']} common days) and stay honest about 2020 & 2022.

*Numbers below are the frozen headline (`docs/results.md`, fingerprint `{R['fingerprint']}`);
the live cells run the fast synthetic control. Young ETFs → short live history, named on
the Signal axis.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one line\n\n"
           "Bills (BIL) pay the risk-free rate for essentially zero risk. Ultra-short "
           "credit (JPST/ICSH/MINT) buys slightly riskier paper — short IG corporates, "
           "a touch of ABS, ~0.3–0.9y duration — and pockets the **spread**. The hope: a "
           "boring, near-riskless *carry* you collect just by parking cash one notch up "
           "the risk ladder."),
        code(
            "R = %r\n"
            "print('sleeve pickup over BIL : %%+.1f bps/yr  (HAC t = %%+.2f)'\n"
            "      %% (R['sleeve_bps'], R['sleeve_t']))\n"
            "print('excess-of-BIL Sharpe   : JPST %%+.2f  ICSH %%+.2f  MINT %%+.2f  vs  SHV %%+.2f  vs  BIL 0.00'\n"
            "      %% (R['jpst_sharpe'], R['icsh_sharpe'], R['mint_sharpe'], R['shv_sharpe']))\n"
            "print('the sleeve out-earns bills by ~%%.0f bps/yr at ~5x their reward-per-risk...'\n"
            "      %% R['sleeve_bps'])" % (R,)
        ),
        md("## 2. ...but is it *riskless*? The honest stress test\n\n"
           "The whole appeal is 'near-riskless'. It isn't. When cash is exactly what you "
           "want — a liquidity crunch or a hiking cycle — the sliver of credit+duration "
           "bites:"),
        code(
            "print('March-2020 COVID crunch : BIL %+.2f%%  vs  JPST %+.2f%%  ICSH %+.2f%%  MINT %+.2f%%'\n"
            "      % (R['covid_bil'], R['covid_jpst'], R['covid_icsh'], R['covid_mint']))\n"
            "print('2022 rate-hike year     : BIL %+.2f%%  vs  JPST %+.2f%%  ICSH %+.2f%%  MINT %+.2f%%'\n"
            "      % (R['y2022_bil'], R['y2022_jpst'], R['y2022_icsh'], R['y2022_mint']))\n"
            "print('max drawdown            : BIL %.2f%%  vs sleeve %.2f%% to %.2f%%'\n"
            "      % (R['bil_dd'], R['jpst_dd'], R['mint_dd']))"
        ),
        md("## 3. Is the sort just lucky? A live synthetic control\n\n"
           "We plant a known pickup in a seeded toy world and check the detector recovers "
           "it — and that it stays *silent* on the null (credit tracks cash + a credit "
           "factor but with **no** structural carry). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from ultra_short import data, strategy as st\n"
            "def pickup_t(planted, seed):\n"
            "    w = data.synthetic_world(pickup_bps_yr=planted, seed=seed, n_days=2000)\n"
            "    return st.hac_mean((w['CREDIT'] - w['CASH']).dropna())['t_nw']\n"
            "null = np.array([pickup_t(0.0, 885+s) for s in range(12)])\n"
            "plant = np.array([pickup_t(120.0, 885+s) for s in range(12)])\n"
            "print('null (0 bps/yr), 12 seeds : mean HAC t = %+.2f, |t|>=2 in %d/12' % (null.mean(), int((abs(null)>=2).sum())))\n"
            "print('planted +120 bps/yr       : mean HAC t = %+.2f, |t|>=2 in %d/12' % (plant.mean(), int((abs(plant)>=2).sum())))"
        ),
        md(f"## 4. The honest verdict\n\n"
           f"The pickup is **real in the point estimate** — the sleeve out-earns bills by "
           f"**~{R['sleeve_bps']:.0f} bps/yr** at an excess Sharpe of **{R['sleeve_sharpe']:.2f}** "
           f"vs **{R['shv_sharpe']:.2f}** for short Treasuries and **0** for bills, and on "
           f"MINT's full {R['mint_long_years']:.0f}-year tape it is **+{R['mint_long_bps']:.0f} "
           f"bps/yr at HAC *t* = {R['mint_long_t']:.2f}**. But it does **not clear the desk's "
           f"robustness bar**: on the full 3-ETF sleeve the HAC *t* is only "
           f"**{R['sleeve_t']:.2f}**, the bootstrap Sharpe CI **crosses zero** "
           f"([{R['ci_lo']:+.2f}, {R['ci_hi']:+.2f}], {R['ci_fracneg']*100:.0f}% of resamples "
           f"negative), and — the killer — the whole edge lives in the **early** window "
           f"(MINT pre-2018 *t* = {R['mint_long_early_t']:.2f} vs post-2018 *t* = "
           f"{R['mint_long_late_t']:.2f}). And it is **not riskless**: −1% in 2022, −3 to −4% "
           f"in the COVID crunch, while bills stayed flat. **Signal: Weak. Tradability: "
           f"Fragile** — costs barely dent it (it's buy-and-hold), but a thin, era-contingent, "
           f"not-quite-significant carry is not something you can bank."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 885 — Ultra-Short Credit Pickup — the teardown\n\n"
           "The excess-of-bills Sharpe race, the HAC *t* on the credit-minus-bill pickup, "
           "a block-bootstrap Sharpe CI, the sub-era cut, the drawdown & stress windows, "
           "MINT's long 2009→ history, the cost math, and the synthetic control."),
        code("R = %r" % (R,)),
        md("## 1. The reward-per-risk race — annualised EXCESS-of-BIL Sharpe (rf = BIL)\n\n"
           "Every Sharpe is excess-of-cash (minus BIL). The credit sleeve genuinely wins "
           "on the point estimate; SHV (a hair more duration, ~zero credit) barely moves."),
        code(
            "print(f\"JPST : excess Sharpe {R['jpst_sharpe']:+.2f}  (ret {R['jpst_ret']:.2f}%/yr, vol {R['jpst_vol']:.2f}%)\")\n"
            "print(f\"ICSH : excess Sharpe {R['icsh_sharpe']:+.2f}  (ret {R['icsh_ret']:.2f}%/yr, vol {R['icsh_vol']:.2f}%)\")\n"
            "print(f\"MINT : excess Sharpe {R['mint_sharpe']:+.2f}  (ret {R['mint_ret']:.2f}%/yr, vol {R['mint_vol']:.2f}%)\")\n"
            "print(f\"SHV  : excess Sharpe {R['shv_sharpe']:+.2f}  (ret {R['shv_ret']:.2f}%/yr, vol {R['shv_vol']:.2f}%)\")\n"
            "print(f\"BIL  : excess Sharpe  0.00  (ret {R['bil_ret']:.2f}%/yr, vol {R['bil_vol']:.2f}%)  <- the cash leg\")"
        ),
        md("## 2. The pickup — equal-weight credit sleeve minus BIL, HAC t\n\n"
           "Ultra-short credit total returns are serially correlated (smooth NAV marks), "
           "so the Newey-West correction knocks the naive t down — this is why a 0.62 "
           "Sharpe over 9 years does *not* translate into a t≥2."),
        code(
            "print(f\"sleeve - BIL : {R['sleeve_bps']:+.1f} bps/yr  HAC t = {R['sleeve_t']:+.2f}  \"\n"
            "      f\"(n={R['n_days']}, lags={R['sleeve_lags']}, excess Sharpe {R['sleeve_sharpe']:+.2f})\")\n"
            "print(f\"  JPST-BIL {R['jpst_bps']:+.1f} (t={R['jpst_pt']:+.2f})  \"\n"
            "      f\"ICSH-BIL {R['icsh_bps']:+.1f} (t={R['icsh_pt']:+.2f})  \"\n"
            "      f\"MINT-BIL {R['mint_bps']:+.1f} (t={R['mint_pt']:+.2f})\")\n"
            "print(f\"  SHV-BIL {R['shv_bps']:+.1f} bps/yr (the near-zero-credit control)\")"
        ),
        md("## 3. Bootstrap CI on the sleeve excess Sharpe (circular block, 2000 draws)\n\n"
           "The interval crosses zero — the pickup is not distinguishable from cash at 95%."),
        code(
            "print(f\"sharpe {R['sleeve_sharpe']:+.2f}  95% CI [{R['ci_lo']:+.2f}, {R['ci_hi']:+.2f}]  \"\n"
            "      f\"frac<0 = {R['ci_fracneg']:.3f}  (block {R['ci_block']})\")"
        ),
        md("## 4. Sub-eras — the edge lives in the past\n\n"
           "Split the sleeve pickup at 2019, and split MINT's long history at 2018. The "
           "*mean* is stable, but the significance evaporates in the modern regime "
           "(post-GFC spreads compressed; 2021 ZIRP left no spread to earn; 2022 duration hurt)."),
        code(
            "print(f\"sleeve <2019 : {R['early_bps']:+.1f} bps/yr  HAC t = {R['early_t']:+.2f}  (n={R['early_n']})\")\n"
            "print(f\"sleeve >=2019: {R['late_bps']:+.1f} bps/yr  HAC t = {R['late_t']:+.2f}  (n={R['late_n']})  Welch t={R['welch_t']:+.2f}\")\n"
            "print(f\"MINT long-history {R['mint_long_years']:.0f}y : {R['mint_long_bps']:+.0f} bps/yr  HAC t = {R['mint_long_t']:+.2f}  \"\n"
            "      f\"Sharpe {R['mint_long_sharpe']:.2f}  CI [{R['mint_long_cilo']:+.2f},{R['mint_long_cihi']:+.2f}]\")\n"
            "print(f\"  MINT <2018 t = {R['mint_long_early_t']:+.2f}   vs   MINT >=2018 t = {R['mint_long_late_t']:+.2f}\")"
        ),
        md("## 5. It is not riskless — drawdowns & stress windows"),
        code(
            "print(f\"max DD (common sample): BIL {R['bil_dd']:.2f}%  JPST {R['jpst_dd']:.2f}%  \"\n"
            "      f\"ICSH {R['icsh_dd']:.2f}%  MINT {R['mint_dd']:.2f}%\")\n"
            "print(f\"COVID Feb-Mar 2020    : BIL {R['covid_bil']:+.2f}%  JPST {R['covid_jpst']:+.2f}%  \"\n"
            "      f\"ICSH {R['covid_icsh']:+.2f}%  MINT {R['covid_mint']:+.2f}%\")\n"
            "print(f\"2022 hiking year      : BIL {R['y2022_bil']:+.2f}%  JPST {R['y2022_jpst']:+.2f}%  \"\n"
            "      f\"ICSH {R['y2022_icsh']:+.2f}%  MINT {R['y2022_mint']:+.2f}%\")"
        ),
        md("## 6. Costs — the trade is buy-and-hold, so friction is trivial\n\n"
           "You buy the sleeve once; maturities roll inside the fund; ETF fees are already "
           "in the net-of-fee tape. One-way spread × NAV × ~1 turnover/yr barely registers "
           "— which is exactly why the *Signal*, not costs, is the binding constraint here."),
        code(
            "print(f\"1 bp one-way x1/yr: net {R['cost1_net']:+.1f} bps/yr (Sharpe {R['cost1_sharpe']:+.2f})\")\n"
            "print(f\"2 bp one-way x1/yr: net {R['cost2_net']:+.1f} bps/yr\")\n"
            "print(f\"5 bp one-way x1/yr: net {R['cost5_net']:+.1f} bps/yr  (gross was {R['sleeve_bps']:+.1f})\")"
        ),
        md("## 7. Synthetic control — the machinery is unbiased (never market evidence)\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted pickup "
           "by the exact planted amount."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from ultra_short import data, strategy as st\n"
            "def stats(planted, seed):\n"
            "    w = data.synthetic_world(pickup_bps_yr=planted, seed=seed, n_days=2000)\n"
            "    ex = (w['CREDIT'] - w['CASH']).dropna()\n"
            "    return st.hac_mean(ex)['t_nw'], st.hac_mean(ex)['mean_bps_yr']\n"
            "null_t = np.array([stats(0.0, 885+s)[0] for s in range(12)])\n"
            "plant = [stats(120.0, 885+s) for s in range(12)]\n"
            "plant_t = np.array([p[0] for p in plant])\n"
            "m0 = np.mean([stats(0.0, 885+s)[1] for s in range(12)])\n"
            "m1 = np.mean([p[1] for p in plant])\n"
            "print('null   (0 bps/yr): mean t %+.2f, |t|>=2 in %d/12' % (null_t.mean(), int((abs(null_t)>=2).sum())))\n"
            "print('planted (+120)   : mean t %+.2f, |t|>=2 in %d/12' % (plant_t.mean(), int((abs(plant_t)>=2).sum())))\n"
            "print('recovered carry  : %+.1f bps/yr (planted +120.0)' % (m1 - m0))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** The ultra-short credit pickup is genuinely there in the "
           f"point estimate — the sleeve out-earns bills by **{R['sleeve_bps']:+.1f} bps/yr** "
           f"at an excess Sharpe of **{R['sleeve_sharpe']:.2f}** (vs {R['shv_sharpe']:.2f} for "
           f"SHV, 0 for BIL), and on MINT's {R['mint_long_years']:.0f}-year tape it is "
           f"**+{R['mint_long_bps']:.0f} bps/yr at HAC *t* = {R['mint_long_t']:.2f}** with a "
           f"bootstrap CI clear of zero. But it **fails the robustness bar**: the full-sleeve "
           f"HAC *t* is only **{R['sleeve_t']:.2f}**, the bootstrap Sharpe CI crosses zero "
           f"([{R['ci_lo']:+.2f}, {R['ci_hi']:+.2f}], {R['ci_fracneg']*100:.0f}% negative), and "
           f"it does **not hold across sub-eras** — the significance is entirely a pre-2018 "
           f"post-GFC phenomenon (MINT *t* = {R['mint_long_early_t']:.2f} → "
           f"{R['mint_long_late_t']:.2f}). Sign is right and economically real everywhere, so "
           f"not None; robustness fails, so not Real. *Young ETFs → short live history.*\n"
           f"- **Tradability — Fragile.** Costs are trivial (buy-and-hold, fees inside the "
           f"tape; at 1 bp the net is {R['cost1_net']:+.1f} of the {R['sleeve_bps']:+.1f} "
           f"gross) — so this is **not** a cost Mirage. It is Fragile because the edge is thin "
           f"(~50 bps/yr), era-contingent (dead in the modern regime), and **not riskless**: "
           f"MINT lost {R['y2022_mint']:.1f}% in 2022 while bills made +{R['y2022_bil']:.1f}%, "
           f"and the sleeve drew down {R['mint_dd']:.1f}% vs bills' {R['bil_dd']:.2f}% in the "
           f"COVID crunch. A real-but-thin carry you cannot certify → Fragile."),
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
