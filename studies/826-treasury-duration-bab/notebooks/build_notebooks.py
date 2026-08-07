"""Generate the two narrative notebooks for Study 826 (Treasury Duration BAB).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily
# total-return closes for SHY/IEI/IEF/TLH/TLT, 2010-01-04 -> 2026-06-30; trailing-252d
# beta to the equal-weight duration factor; Frazzini-Pedersen rank-weighted BAB book).
R = dict(
    start="2010-01-04", end="2026-06-30", n_etfs=5, n_days=3894, rows=4147,
    fingerprint="e423191a2863",
    beta_shy=0.123, beta_iei=0.470, beta_ief=0.911, beta_tlh=1.402, beta_tlt=2.094,
    bab_bps=1.31, t_nw=2.50, t_1s=2.37, sharpe=0.60, gross_sharpe=0.60,
    lev_lo_bps=2.00, lev_hi_bps=0.69, welch_t=1.18,
    beta_lo=0.238, beta_hi=1.864, gross_lev=5.04,
    alpha_bps=1.32, t_alpha=2.51, beta_resid=-0.007,
    placebo_obs=1.31, placebo_mean=2.573, placebo_sd=0.728,
    placebo_p=0.99100, placebo_sigma=-1.74, placebo_draws=1000,
    era_early_bps=0.32, era_early_t=0.49, era_early_n=1760,
    era_late_bps=2.13, era_late_t=2.68, era_late_n=2134,
    timer_1_gross=1.31, timer_1_cost=0.02, timer_1_borrow=0.11,
    timer_1_net=1.19, timer_1_t=2.14, timer_1_sharpe=0.55, timer_1_ann=3.0,
    timer_5_gross=1.31, timer_5_cost=0.08, timer_5_borrow=0.11,
    timer_5_net=1.13, timer_5_t=2.03, timer_5_sharpe=0.52, timer_5_ann=2.8,
    null_mean_t=0.42, null_sd_t=1.08, null_fire=1,
    planted_t=30.79, planted_welch=15.10, planted_beta_resid=0.006,
)


HEADER = f"""# Study 826 — Treasury Duration BAB 🏦📉

**Does *betting-against-beta* earn a low-risk alpha inside the Treasury curve?**

Frazzini & Pedersen (2014) show that low-beta assets beat high-beta assets on a
*risk-adjusted* basis, so a **BAB** book — long the low-beta legs levered to unit beta,
short the high-beta legs, beta-neutral — earns a positive alpha; they document it across
asset classes, **US Treasuries by maturity included**. We rebuild the Treasury-curve
version from five iShares ETFs that ladder the curve —
SHY (1-3y) → IEI → IEF → TLH → TLT (20y+) — estimate each ETF's beta to an equal-weight
**duration factor**, and form the classic rank-weighted BAB book
({R['start']} → {R['end']}, {R['n_etfs']} ETFs).

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Fingerprint `{R['fingerprint']}`.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Leverage-constrained investors who want more return can't just borrow — so "
           "they over-buy high-beta (here: long-duration) assets and bid them up, leaving "
           "low-beta (short-duration) assets *cheap per unit of risk*. Frazzini-Pedersen's "
           "fix: **lever up the boring low-beta leg** to the same risk as the exciting "
           "high-beta leg, short the high-beta leg, and pocket the low-risk premium. Inside "
           "the Treasury curve that means: lever up SHY/IEI, short TLH/TLT, beta-neutral."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(bab_bps=%r, t_nw=%r, sharpe=%r, lev_lo_bps=%r, lev_hi_bps=%r,\n"
            "         beta_lo=%r, beta_hi=%r, gross_lev=%r)\n"
            "print('BAB book: long low-beta (levered) / short high-beta, beta-neutral')\n"
            "print('  BAB spread : %%+.2f bps/day  (Newey-West t = %%+.2f, Sharpe %%.2f)'\n"
            "      %% (R['bab_bps'], R['t_nw'], R['sharpe']))\n"
            "print('  levered legs: low-beta %%+.2f vs high-beta %%+.2f bps/day'\n"
            "      %% (R['lev_lo_bps'], R['lev_hi_bps']))\n"
            "print('  the cage   : beta_lo %%.2f / beta_hi %%.2f -> %%.1fx gross leverage'\n"
            "      %% (R['beta_lo'], R['beta_hi'], R['gross_lev']))"
            % (R["bab_bps"], R["t_nw"], R["sharpe"], R["lev_lo_bps"], R["lev_hi_bps"],
               R["beta_lo"], R["beta_hi"], R["gross_lev"])
        ),
        md("## 2. Is the sort just lucky? A live synthetic control\n\n"
           "We plant a Frazzini-Pedersen low-beta alpha in a seeded toy curve (`edge>0`) "
           "and check the detector recovers it — and stays *silent* on the null (`edge=0`, "
           "betas spread out but no alpha). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from duration_bab import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=826, n_days=1300))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0015, seed=826, n_days=1600))\n"
            "print('null world   : BAB NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: BAB NW t = %+.2f  (should light up)' % planted['t_nw'])\n"
            "print('planted book residual beta to factor = %+.3f (beta-neutral)' % planted['beta_resid'])"
        ),
        md("## 3. The honest verdict — the low-risk alpha does *not* hold here\n\n"
           f"On the real curve the BAB book prints **{R['bab_bps']:+.2f} bps/day** with "
           f"Newey-West *t* = **{R['t_nw']:+.2f}** — right sign, and it *nominally* clears "
           "the significance bar. But two checks knock it down:\n\n"
           f"1. **The placebo refutes the signal.** Permute which ETF's return lands in "
           f"each leg of the *same* leverage cage: the random assignment earns **more** "
           f"(placebo mean {R['placebo_mean']:+.2f} bps) than the real beta-sorted book "
           f"({R['placebo_obs']:+.2f} bps) — the observed sits ~{abs(R['placebo_sigma']):.1f}σ "
           f"into the *left* tail (right-tail p = {R['placebo_p']:.2f}). The **beta signal "
           "adds no value**; the small positive number is mechanical *levered carry* from "
           "the 1/β scaling on the low-vol leg.\n"
           f"2. **It's one era.** 2010–2017 the BAB is flat (*t* = {R['era_early_t']:+.2f}); "
           f"the whole result lives in 2018–2026 (*t* = {R['era_late_t']:+.2f}).\n\n"
           "The betas ladder cleanly (SHY 0.12 → TLT 2.09) and the book is beta-neutral "
           f"(residual β = {R['beta_resid']:+.3f}), so the machinery is sound — the claimed "
           "Frazzini-Pedersen low-risk edge is simply **absent as a signal** on this curve. "
           "**Signal: None**, **Tradability: Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 826 — Treasury Duration BAB — the teardown\n\n"
           "The beta ladder, the Newey-West BAB *t*, the factor-regression alpha and "
           "residual beta, the 1,000-permutation placebo, the two-era robustness cut, the "
           "costed leveraged timer, and the 20-seed synthetic control."),
        code("R = %r" % (R,)),
        md("## The beta ladder — each ETF's trailing-252d beta to the duration factor\n\n"
           "The equal-weight duration factor; betas rise monotonically with maturity."),
        code(
            "print('  SHY %.3f  IEI %.3f  IEF %.3f  TLH %.3f  TLT %.3f'\n"
            "      % (R['beta_shy'], R['beta_iei'], R['beta_ief'], R['beta_tlh'], R['beta_tlt']))"
        ),
        md("## The headline — Frazzini-Pedersen BAB return\n\n"
           "Long low-beta legs levered to unit beta, short high-beta legs, beta-neutral."),
        code(
            "print(f\"BAB          : {R['bab_bps']:+.2f} bps/day  NW(10) t = {R['t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['t_1s']:+.2f}  Sharpe = {R['sharpe']:.2f}\")\n"
            "print(f\"levered legs : low-beta {R['lev_lo_bps']:+.2f} vs high-beta {R['lev_hi_bps']:+.2f} bps \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print(f\"factor reg   : alpha {R['alpha_bps']:+.2f} bps/day (NW t = {R['t_alpha']:+.2f}), \"\n"
            "      f\"residual beta = {R['beta_resid']:+.3f} (beta-neutral)\")\n"
            "print(f\"the cage     : beta_lo {R['beta_lo']:.3f} / beta_hi {R['beta_hi']:.3f} -> {R['gross_lev']:.2f}x gross leverage\")"
        ),
        md("## Placebo — permute the returns into the SAME leverage cage (1,000 permutations)\n\n"
           "Keep the beta-rank weights and the 1/β leverage; permute which ETF's return "
           "feeds each leg. A real beta signal ⇒ observed in the far *right* tail. Here it "
           "is the opposite — random assignment does *better*."),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> right-tail p = {R['placebo_p']:.5f}  \"\n"
            "      f\"({R['placebo_sigma']:+.2f}sigma vs placebo mean)\")\n"
            "print('=> the beta SIGNAL adds no value; the positive number is levered carry.')"
        ),
        md("## Robustness — two eras (split 2018-01-01)"),
        code(
            "print(f\"2010-2017 (n={R['era_early_n']}): {R['era_early_bps']:+.2f} bps  NW t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2018-2026 (n={R['era_late_n']}): {R['era_late_bps']:+.2f} bps  NW t = {R['era_late_t']:+.2f}\")"
        ),
        md("## The timer — can you get paid for it?\n\n"
           "One-way cost × turnover of the *levered* weights; short leg pays 50 bps/yr "
           "borrow. (Financing the ~5x gross leverage at the short rate is **not** charged "
           "here — with `rf≈0` the levered carry is flattered; a realistic financing rate "
           "erodes the little that remains.)"),
        code(
            "for tag,g,c,b,n,t in [('1 bp',R['timer_1_gross'],R['timer_1_cost'],R['timer_1_borrow'],R['timer_1_net'],R['timer_1_t']),\n"
            "                      ('5 bps',R['timer_5_gross'],R['timer_5_cost'],R['timer_5_borrow'],R['timer_5_net'],R['timer_5_t'])]:\n"
            "    print(f\"{tag:>5}: gross {g:+.2f} -> net {n:+.2f} bps/day (cost {c:.2f} + borrow {b:.2f}/day, t={t:+.2f})\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted "
           "low-beta alpha with a beta-neutral book."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from duration_bab import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=826+s, n_days=1300))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0015, seed=826, n_days=1600))\n"
            "print(f\"planted (edge=0.0015): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}, residual beta = {planted['beta_resid']:+.3f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The Frazzini-Pedersen low-risk / BAB alpha does **not** "
           f"replicate inside the Treasury curve. The book prints **{R['bab_bps']:+.2f} "
           f"bps/day** (NW *t* = **{R['t_nw']:+.2f}**, right sign) — but the permutation "
           f"placebo shows the beta sort earns *less* than a random assignment into the "
           f"same 1/β cage (observed {R['placebo_obs']:+.2f} vs placebo {R['placebo_mean']:+.2f} "
           f"bps, ~{abs(R['placebo_sigma']):.1f}σ into the left tail), and it is entirely a "
           f"2018–2026 phenomenon (*t* = {R['era_early_t']:+.2f} / {R['era_late_t']:+.2f}). "
           f"The 20-seed synthetic control recovers a *planted* alpha cleanly "
           f"(*t* = {R['planted_t']:+.1f}, fires on {R['null_fire']}/20 nulls, residual β ≈ 0), "
           f"so the machinery is sound — the beta *signal* adds nothing; the positive number "
           f"is mechanical levered carry.\n"
           f"- **Tradability — Mirage.** What remains is ~{R['timer_1_ann']:.0f}%/yr net "
           f"(net *t* ≈ {R['timer_1_t']:+.1f} at 1–5 bps costs) but it rests on "
           f"~{R['gross_lev']:.0f}x gross leverage financed at `rf≈0`, is beaten by random "
           f"assignment, and lives in one era — a mirage, not a low-risk premium."),
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
