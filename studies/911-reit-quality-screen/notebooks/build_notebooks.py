"""Generate the two narrative notebooks for Study 911 (REIT Quality Screen).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily total-return
# closes, common monthly sample 2007-06 -> 2026-06, 229 months, excess-of-BIL).
R = dict(
    start="2007-06", end="2026-06", n_months=229, fingerprint="b99a4946b405",
    # excess Sharpe race
    sh_rez=0.360, sh_vnq=0.290, sh_rwr=0.270, sh_rem=0.039, sh_spy=0.644,
    ann_rez=6.83, ann_vnq=5.40, ann_rwr=4.95, ann_rem=-1.13, ann_spy=10.68,
    vol_rez=20.9, vol_vnq=22.3, vol_rem=24.2,
    # leg 1: durable-income tilt
    rezvnq_bps=8.7, rezvnq_t=0.75, book_bps=2.0, book_t=0.41,
    adv=0.070, adv_lo=-0.070, adv_hi=0.202, adv_fracneg=0.17,
    era1_rezvnq_bps=1.4, era1_rezvnq_t=0.08, era2_rezvnq_bps=16.2, era2_rezvnq_t=1.09,
    # leg 2: the trap
    rezrem_bps=55.0, rezrem_t=1.72,
    era1_rezrem_bps=92.7, era1_rezrem_t=1.83, era2_rezrem_bps=17.1, era2_rezrem_t=0.47,
    # drawdowns
    dd_rez=-66.9, dd_vnq=-73.1, dd_rem=-74.7, dd_spy=-55.2,
    # costed book
    cost2_net=1.8, cost2_t=0.37, cost2_ann=0.22,
    cost5_net=1.5, cost5_t=0.31, cost5_ann=0.18,
    cost10_net=1.0, cost10_t=0.21, cost10_ann=0.12,
    # synthetic control
    null_adv_mean=-0.019, null_straddle=17, null_trapflag=20,
    planted_t=3.17, planted_adv=0.109, planted_adv_lo=0.040,
)


HEADER = f"""# Study 911 — REIT Quality Screen 🏢

**Not all REITs are equal — does a "quality" screen beat the broad index?**

Equity REITs own property and collect durable rents at moderate leverage; **mortgage REITs**
lever a thin spread between long mortgage assets and short funding and pay it out as a fat,
fragile dividend. The "quality REIT" screen holds the durable-income equity sleeve
(residential **REZ**, broad **VNQ**/**RWR**) and screens *out* the levered-carry sleeve
(**REM**), aiming to beat the broad index on a **risk-adjusted, net-of-cost** basis.

We race the live vehicles excess-of-BIL over **{R['start']} → {R['end']}** ({R['n_months']}
months). *Numbers below are the frozen headline (`docs/results.md`); the live cells run the
fast synthetic control. These sector ETFs are young — magnitudes are indicative.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The race in one table\n\n"
           "Excess-vs-excess Sharpe (every sleeve minus the BIL T-bill), 2007–2026. Watch "
           "the residential quality sleeve (REZ) edge the broad index (VNQ) — and watch the "
           "mortgage-REIT sleeve (REM) earn a **negative** 19-year total return despite its "
           "famous fat dividend."),
        code("R = %r" % (R,)),
        code(
            "print('sleeve         excessSharpe   ann.total-return')\n"
            "print(f\"REZ (quality)  {R['sh_rez']:+.3f}        {R['ann_rez']:+.2f}%/yr\")\n"
            "print(f\"VNQ (broad)    {R['sh_vnq']:+.3f}        {R['ann_vnq']:+.2f}%/yr\")\n"
            "print(f\"REM (mREIT)    {R['sh_rem']:+.3f}        {R['ann_rem']:+.2f}%/yr   <- the levered-carry trap\")"
        ),
        md("## 2. Is the machinery honest? A live synthetic control\n\n"
           "We plant a quality edge in a seeded toy world (`edge>0`) and check the "
           "Sharpe-advantage estimator recovers it — and that it stays centred at zero on "
           "the null (`edge=0`), while the trap detector always flags the inferior-Sharpe "
           "levered leg. No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from reit_quality import data, strategy as st\n"
            "null = st.sharpe_advantage(data.synthetic_world(edge_ann=0.0, seed=911), 'QUAL','BROAD', rf='CASH', n_boot=600)\n"
            "plant = st.sharpe_advantage(data.synthetic_world(edge_ann=0.03, seed=911), 'QUAL','BROAD', rf='CASH', n_boot=600)\n"
            "print('null   : Sharpe adv %+.3f  CI [%+.3f, %+.3f]  (straddles 0)' % (null['advantage'], null['ci_low'], null['ci_high']))\n"
            "print('planted: Sharpe adv %+.3f  CI [%+.3f, %+.3f]  (clear of 0)' % (plant['advantage'], plant['ci_low'], plant['ci_high']))"
        ),
        md(f"## 3. The honest verdict — one real distinction, no bankable edge\n\n"
           f"**The durable-income tilt is *not* certified.** REZ edges VNQ on Sharpe "
           f"({R['sh_rez']:.2f} vs {R['sh_vnq']:.2f}), but the monthly spread is only "
           f"**{R['rezvnq_bps']:+.1f} bps/mo at HAC *t* = {R['rezvnq_t']:.2f}**, the "
           f"bootstrap Sharpe-advantage CI **[{R['adv_lo']:+.3f}, {R['adv_hi']:+.3f}] "
           f"straddles zero**, and all of the (insignificant) tilt lives in the second half "
           f"(t = {R['era1_rezvnq_t']:.2f} → {R['era2_rezvnq_t']:.2f}).\n\n"
           f"**The levered-carry *trap* is real and structural.** Mortgage REITs earned "
           f"**{R['ann_rem']:+.2f}%/yr for 19 years** at excess Sharpe {R['sh_rem']:.3f} — an "
           f"order of magnitude below the equity sleeves. *But* the broad index VNQ already "
           f"holds ≈ no mortgage REITs, so avoiding the trap is **already free**.\n\n"
           f"**Signal: Mixed** (tilt absent, trap real) · **Tradability: Fragile** (the one "
           f"robust action is already inside the index you'd otherwise hold; the incremental "
           f"tilt nets ~+0.1–0.2%/yr at *t* ≈ 0.3)."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 911 — REIT Quality Screen — the teardown\n\n"
           "The excess-Sharpe race, the durable-income tilt (HAC spread *t* + bootstrap "
           "Sharpe-advantage CI + era cut), the levered-carry trap, the daily drawdowns, the "
           "costed quality book, and the 20-seed synthetic control."),
        code("R = %r" % (R,)),
        md("## The race — excess-vs-excess Sharpe (minus BIL)"),
        code(
            "for s,(sh,a,v) in {'REZ':(R['sh_rez'],R['ann_rez'],R['vol_rez']),\n"
            "                   'VNQ':(R['sh_vnq'],R['ann_vnq'],R['vol_vnq']),\n"
            "                   'REM':(R['sh_rem'],R['ann_rem'],R['vol_rem'])}.items():\n"
            "    print(f'{s}: excessSharpe {sh:+.3f}  ann {a:+.2f}%  vol {v:.1f}%')"
        ),
        md("## Leg 1 — the durable-income tilt (REZ vs VNQ): thin, not certified"),
        code(
            "print(f\"REZ-VNQ spread : {R['rezvnq_bps']:+.1f} bps/mo  HAC t = {R['rezvnq_t']:+.2f}\")\n"
            "print(f\"quality book   : {R['book_bps']:+.1f} bps/mo  HAC t = {R['book_t']:+.2f}\")\n"
            "print(f\"Sharpe adv     : {R['adv']:+.3f}  95% CI [{R['adv_lo']:+.3f}, {R['adv_hi']:+.3f}]  frac<0 = {R['adv_fracneg']:.2f}\")\n"
            "print(f\"  era 2007-2016: {R['era1_rezvnq_bps']:+.1f} bps  t = {R['era1_rezvnq_t']:+.2f}\")\n"
            "print(f\"  era 2017-2026: {R['era2_rezvnq_bps']:+.1f} bps  t = {R['era2_rezvnq_t']:+.2f}\")"
        ),
        md("## Leg 2 — the leveraged-carry trap (mortgage REITs)"),
        code(
            "print(f\"REZ-REM spread : {R['rezrem_bps']:+.1f} bps/mo  HAC t = {R['rezrem_t']:+.2f}  (GFC-concentrated)\")\n"
            "print(f\"  era 2007-2016: {R['era1_rezrem_bps']:+.1f} bps  t = {R['era1_rezrem_t']:+.2f}\")\n"
            "print(f\"  era 2017-2026: {R['era2_rezrem_bps']:+.1f} bps  t = {R['era2_rezrem_t']:+.2f}\")\n"
            "print(f\"REM total return {R['ann_rem']:+.2f}%/yr vs VNQ {R['ann_vnq']:+.2f}%/yr -- a yield trap on total-return basis\")"
        ),
        md("## Risk — daily total-return max drawdowns"),
        code(
            "for s,d in [('REZ',R['dd_rez']),('VNQ',R['dd_vnq']),('REM',R['dd_rem']),('SPY',R['dd_spy'])]:\n"
            "    print(f'{s}: {d:+.1f}%')"
        ),
        md("## Tradability — the quality book, costed vs buy-and-hold VNQ"),
        code(
            "for tag,n,t,a in [('2 bps',R['cost2_net'],R['cost2_t'],R['cost2_ann']),\n"
            "                  ('5 bps',R['cost5_net'],R['cost5_t'],R['cost5_ann']),\n"
            "                  ('10 bps',R['cost10_net'],R['cost10_t'],R['cost10_ann'])]:\n"
            "    print(f'{tag:>6} one-way: net {n:+.1f} bps/mo (t={t:+.2f}, ~{a:+.2f}%/yr)')"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: on the null the Sharpe-advantage CI must straddle zero; a planted edge must "
           "light up; the trap detector must always flag the inferior-Sharpe leg."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from reit_quality import data, strategy as st\n"
            "straddle = trap = 0\n"
            "for s in range(8):\n"
            "    w = data.synthetic_world(edge_ann=0.0, seed=911+s)\n"
            "    a = st.sharpe_advantage(w, 'QUAL','BROAD', rf='CASH', n_boot=500)\n"
            "    straddle += int(a['ci_low'] < 0 < a['ci_high'])\n"
            "    trap += int(st.excess_sharpe(w,'TRAP','CASH') < st.excess_sharpe(w,'BROAD','CASH'))\n"
            "print(f'null (edge=0), 8 seeds: CI straddles zero in {straddle}/8; trap flagged {trap}/8')\n"
            "d = st.synth_detect(data.synthetic_world(edge_ann=0.03, seed=911))\n"
            "print(f\"planted (+3%/yr): QUAL-BROAD HAC t = {d['spread_t']:+.2f}, Sharpe adv = {d['adv']:+.3f} (CI low {d['adv_ci_low']:+.3f})\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Mixed.** The durable-income tilt is **not certified**: REZ − VNQ is "
           f"**{R['rezvnq_bps']:+.1f} bps/mo at HAC *t* = {R['rezvnq_t']:.2f}**, Sharpe-advantage "
           f"CI **[{R['adv_lo']:+.3f}, {R['adv_hi']:+.3f}]** straddles zero, not era-robust "
           f"(t = {R['era1_rezvnq_t']:.2f} → {R['era2_rezvnq_t']:.2f}). The levered-carry "
           f"**trap** *is* real: mortgage REITs earned **{R['ann_rem']:+.2f}%/yr** at Sharpe "
           f"{R['sh_rem']:.3f} — but the broad index already excludes it.\n"
           f"- **Tradability — Fragile.** The costed quality book nets **~+{R['cost5_ann']:.2f}%/yr "
           f"at *t* = {R['cost5_t']:.2f}** over VNQ (costs barely matter — the gross edge is only "
           f"~{R['book_bps']:.0f} bps/mo); the one robust action (avoid mortgage REITs) is "
           f"already free inside the broad ETF you'd hold anyway."),
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
