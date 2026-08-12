"""Generate the two narrative notebooks for Study 889 (Broad Dollar-Hedge Overlay).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from the frozen
``R`` dict (mirroring docs/results.md); the live cells run only the fast synthetic positive control,
so execution is quick and network-free.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance monthly total-return,
# HEFA/EFA 2014-03->2026-06, DBEF/EFA 2011-07->2026-06; fingerprint 7afa89fb9f2f).
R = dict(
    fp="7afa89fb9f2f", asof="2026-06-30",
    h_n=148, h_start="2014-03", h_carry=1.68, h_t=4.74, h_obs=1.35,
    h_beta=0.93, h_tbeta=34.8, h_alpha=1.79, h_talpha=5.38, h_r2=0.91,
    h_ci_lo=0.90, h_ci_hi=2.45, h_frac0=0.000,
    h_pre_carry=1.15, h_pre_t=2.89, h_pre_n=94, h_pre_obs=0.89,
    h_post_carry=2.61, h_post_t=4.88, h_post_n=54, h_post_obs=2.14,
    h_sh=0.75, h_sh_lo=0.20, h_sh_hi=1.35, u_sh=0.40, u_sh_lo=-0.11, u_sh_hi=0.95, adv=0.35,
    h_dd=-20.7, u_dd=-27.6,
    d_n=180, d_carry=1.37, d_t=2.56, d_obs=1.09, d_beta=0.90, d_r2=0.69,
    d_ci_lo=-0.02, d_ci_hi=2.84,
    ov_switches=1, ov_share=0.93, ov_sh=0.67, ov_hedged=0.75, ov_unhedged=0.40,
    ov_adv_u=0.27, ov_adv_h=-0.08, ov_cost=0.005,
    sp_net=2.53, sp_t=1.41, sp_charge=0.62,
    null_mean=0.03, null_sd=0.89, null_fire=0, null_seeds=20,
    planted_carry=2.56, planted_t=7.08, planted_beta=1.00,
)


HEADER = f"""# Study 889 — Broad Dollar-Hedge Overlay 🔁

**Is the currency-hedged-minus-unhedged gap for *broad* developed international the
US-vs-foreign rate differential — mechanically — and can you time a hedge on it?**

[Study 613](../../613-currency-hedged-etf-carry/) showed that for **one** market (Japan) the
return gap between a currency-*hedged* equity ETF and its *unhedged* twin is the covered-interest-
parity short-rate differential — "free carry hidden in a share class". Here we **generalise to
broad EAFE** (developed ex-US), where the differential is now *positive* (the Fed out-yields the
ECB/BoJ/BoE/SNB): does `hedged − unhedged` still equal the differential, and does a systematic
"hedge when the US out-yields" overlay add Sharpe?

*Numbers below are the frozen headline (`docs/results.md`, fingerprint `{R['fp']}`, as-of
{R['asof']}); the live cells run the fast synthetic control. Young-ETF caveat: HEFA's clean tape
starts {R['h_start']} and the whole sample is one US-out-yields / rising-dollar era.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. What a hedged EAFE fund actually does\n\n"
           "An *unhedged* EAFE fund (EFA) earns the local stocks **plus** whatever the euro, yen, "
           "pound and franc do against the dollar. A *hedged* fund (HEFA = EFA + one-month FX "
           "forwards) sells that currency basket forward and keeps only the local stocks — **plus** "
           "the forward's carry. Covered interest parity prices the forward at the rate gap, so the "
           "hedged fund quietly pockets `(r_US − r_foreign)`. Since 2022 the Fed sits *above* the "
           "EAFE central banks, so for a dollar holder the hedge now **pays**."),
        code(
            "R = dict(h_carry=%r, h_t=%r, h_obs=%r, h_beta=%r, h_r2=%r)\n"
            "print('HEFA minus EFA, currency stripped out:')\n"
            "print('  carry_hat = %%+.2f%%%%/yr  (HAC t = %%+.2f)' %% (R['h_carry'], R['h_t']))\n"
            "print('  observable US-EAFE policy differential: %%+.2f%%%%/yr' %% R['h_obs'])\n"
            "print('  the hedge is a %%.2f short of the currency basket (R2 = %%.2f)'\n"
            "        %% (R['h_beta'], R['h_r2']))"
            % (R["h_carry"], R["h_t"], R["h_obs"], R["h_beta"], R["h_r2"])
        ),
        md("The carry the hedge pockets (**+1.68 %/yr**) sits right on the *observable* policy "
           "differential (**+1.35 %/yr**), and the hedge is a near-full short of the foreign "
           "currency basket (β = 0.93). Exactly 613's Japan mechanics — now broad, and "
           "dollar-favourable."),
        md("## 2. Is it just luck? A live synthetic control\n\n"
           "We plant a known carry in a seeded toy world and check the estimator recovers it — and "
           "that it stays silent when the planted carry is zero. No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from dollar_hedge import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_world(n_months=180, carry_annual=0.0, seed=889))\n"
            "planted = st.synthetic_detect(data.synthetic_world(n_months=180, carry_annual=0.03, seed=889))\n"
            "print('null world    : carry %+.2f%%/yr  HAC t = %+.2f  (should be ~0)'\n"
            "      % (null['carry_ann_pct'], null['t_carry']))\n"
            "print('planted (+3%%) : carry %+.2f%%/yr  HAC t = %+.2f  beta %.2f  (recovers it)'\n"
            "      % (planted['carry_ann_pct'], planted['t_carry'], planted['beta']))"
        ),
        md("## 3. So can you *time* it? The honest answer: no — you just hold it\n\n"
           f"The obvious overlay is *hedge when the US out-yields, unhedge when it doesn't*. But the "
           f"US has out-yielded EAFE for **{R['ov_share']*100:.0f}% of the post-2014 sample**, so "
           f"the switch fires **{R['ov_switches']} time in twelve years** — it is effectively "
           f"'always hedge', and its brief unhedged spell only *lowers* the Sharpe below "
           f"always-hedging ({R['ov_sh']:.2f} vs {R['ov_hedged']:.2f})."),
        code(
            "R = dict(ov_share=%r, ov_switches=%r, ov_sh=%r, ov_hedged=%r, ov_unhedged=%r)\n"
            "print('overlay Sharpe   %%.2f  (switches: %%d, share hedged %%.0f%%%%)'\n"
            "      %% (R['ov_sh'], R['ov_switches'], R['ov_share']*100))\n"
            "print('always-hedged    %%.2f   <- just holding the hedged wrapper wins'  %% R['ov_hedged'])\n"
            "print('always-unhedged  %%.2f' %% R['ov_unhedged'])"
            % (R["ov_share"], R["ov_switches"], R["ov_sh"], R["ov_hedged"], R["ov_unhedged"])
        ),
        md(f"## 4. The honest verdict\n\n"
           f"- **Signal — Real.** The 613 carry identity **generalises to broad EAFE**: HEFA minus "
           f"EFA (currency stripped) is **+{R['h_carry']:.2f} %/yr at HAC *t* = +{R['h_t']:.2f}**, "
           f"on the observable +{R['h_obs']:.2f} %/yr differential, β = {R['h_beta']:.2f} short of "
           f"the basket, holding in both eras. A genuine, dollar-favourable mechanical premium.\n"
           f"- **Tradability — Fragile.** It is real but you *hold* it, you don't *time* it: the "
           f"US out-yields EAFE ~{R['ov_share']*100:.0f}% of the time so the switch adds nothing "
           f"({R['ov_sh']:.2f} vs {R['ov_hedged']:.2f} Sharpe), the raw hedged-sleeve win rests on "
           f"one dollar regime, and the pure carry can't be cleanly isolated. Thin & un-timeable."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 889 — Broad Dollar-Hedge Overlay — the teardown\n\n"
           "The carry decomposition, the HAC *t* on the same-basket pair, the β ≈ 1 hedge "
           "regression, the 2022 era split, the excess-of-cash Sharpe race, the costed overlay, the "
           "UUP collateral-yield trap, and the planted-carry synthetic control."),
        code("R = %r" % (R,)),
        md("## The identity — `carry_hat = (hedged − unhedged) + fx_foreign`\n\n"
           "`fx_foreign` is the **spot** USD return of an EAFE-weighted EUR/JPY/GBP/CHF basket. The "
           "hedge regression is `diff = α + β·(−fx_foreign)`: β ≈ 1 = a full short of the basket."),
        code(
            "print(f\"HEFA/EFA (same basket, n={R['h_n']}):\")\n"
            "print(f\"  carry_hat {R['h_carry']:+.2f}%/yr  HAC t = {R['h_t']:+.2f}   \"\n"
            "      f\"observed rate diff {R['h_obs']:+.2f}%/yr\")\n"
            "print(f\"  hedge reg: beta = {R['h_beta']:.2f} (t {R['h_tbeta']:+.1f}), \"\n"
            "      f\"alpha = {R['h_alpha']:+.2f}%/yr (t {R['h_talpha']:+.2f}), R2 = {R['h_r2']:.2f}\")\n"
            "print(f\"  carry bootstrap 95% CI [{R['h_ci_lo']:+.2f}, {R['h_ci_hi']:+.2f}]  \"\n"
            "      f\"frac<=0 {R['h_frac0']:.3f}\")\n"
            "print(f\"DBEF/EFA (provider diff, n={R['d_n']}): carry {R['d_carry']:+.2f}%/yr \"\n"
            "      f\"HAC t = {R['d_t']:+.2f}  beta {R['d_beta']:.2f}  R2 {R['d_r2']:.2f}  \"\n"
            "      f\"CI [{R['d_ci_lo']:+.2f}, {R['d_ci_hi']:+.2f}]\")"
        ),
        md("## Era split (cut 2022-01-01) — the carry grows with the differential"),
        code(
            "print(f\"HEFA/EFA pre-2022 (n={R['h_pre_n']}): carry {R['h_pre_carry']:+.2f}%/yr \"\n"
            "      f\"HAC t = {R['h_pre_t']:+.2f}  (obs diff {R['h_pre_obs']:+.2f})\")\n"
            "print(f\"HEFA/EFA 2022+   (n={R['h_post_n']}): carry {R['h_post_carry']:+.2f}%/yr \"\n"
            "      f\"HAC t = {R['h_post_t']:+.2f}  (obs diff {R['h_post_obs']:+.2f})\")\n"
            "print('  -> clears t>=2 in BOTH eras and grows as the gap widens: the mechanical signature')"
        ),
        md("## The excess-of-cash Sharpe race (both legs minus BIL)\n\n"
           "The hedged sleeve out-Sharped the unhedged one and cut the drawdown — but the CIs "
           "**overlap**, so the *advantage* rests on one realised dollar regime, not on the carry."),
        code(
            "print(f\"hedged   ex-cash Sharpe {R['h_sh']:.2f}  95% CI [{R['h_sh_lo']:.2f}, {R['h_sh_hi']:.2f}]\")\n"
            "print(f\"unhedged ex-cash Sharpe {R['u_sh']:.2f}  95% CI [{R['u_sh_lo']:.2f}, {R['u_sh_hi']:.2f}]\")\n"
            "print(f\"advantage {R['adv']:+.2f} (CIs overlap)   max DD hedged {R['h_dd']:.1f}% vs unhedged {R['u_dd']:.1f}%\")"
        ),
        md("## The overlay and the isolation spread — why it is *held*, not *timed*"),
        code(
            "print(f\"overlay: {R['ov_switches']} switch, share hedged {R['ov_share']:.2f}, \"\n"
            "      f\"Sharpe {R['ov_sh']:.2f} vs always-hedged {R['ov_hedged']:.2f} \"\n"
            "      f\"(adv vs hedged {R['ov_adv_h']:+.2f}; cost drag {R['ov_cost']:.3f}%/yr)\")\n"
            "print(f\"isolation spread (long hedged/short unhedged = long the dollar): \"\n"
            "      f\"net {R['sp_net']:+.2f}%/yr  HAC t = {R['sp_t']:+.2f}  (fx vol swamps the carry)\")"
        ),
        md("## The UUP trap — why we use spot, not the tradeable dollar ETF\n\n"
           "`UUP`'s total return also earns the US-bill collateral yield, so a naive "
           "`carry_hat = diff − UUP` subtracts that yield and **cancels most of the carry** — it "
           "prints *negative*, worst in the high-rate 2022+ era. The live cell shows it on the "
           "synthetic world's clean fx leg vs a collateral-contaminated dollar."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from dollar_hedge import data, strategy as st\n"
            "w = data.synthetic_world(n_months=180, carry_annual=0.03, seed=889)\n"
            "pf = st.pair_frame(w, 'HEFA', 'EFA')\n"
            "clean = st.nw_mean_t((pf['diff'] + pf['fx_foreign']).values)[0]*12*100\n"
            "# a UUP-like dollar that ALSO earns ~4%/yr collateral: subtracting it eats the carry\n"
            "uup_like = -pf['fx_foreign'] + 0.04/12\n"
            "contaminated = st.nw_mean_t((pf['diff'] - uup_like).values)[0]*12*100\n"
            "print(f'carry via SPOT fx basket   : {clean:+.2f}%/yr  (recovers the planted +3%)')\n"
            "print(f'carry via UUP-like (w/ 4pct yield): {contaminated:+.2f}%/yr  (collateral yield cancels it)')"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the estimator must NOT fire on the null and must recover a planted carry with β ≈ 1."),
        code(
            "null_t = np.array([st.synthetic_detect(\n"
            "    data.synthetic_world(n_months=180, carry_annual=0.0, seed=889+s))['t_carry'] for s in range(8)])\n"
            "print(f\"null (carry=0), 8 seeds: HAC t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), \"\n"
            "      f\"|t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_world(n_months=180, carry_annual=0.03, seed=889))\n"
            "print(f\"planted (+3%/yr): recovered {planted['carry_ann_pct']:+.2f}%/yr \"\n"
            "      f\"(HAC t {planted['t_carry']:+.2f}), hedge beta {planted['beta']:.2f}\")\n"
            "print(f\"(frozen full 20-seed run: null t mean {R['null_mean']:+.2f}, fires {R['null_fire']}/{R['null_seeds']})\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Real.** The 613 currency-hedge carry identity **generalises to broad "
           f"developed international.** On the clean same-basket HEFA/EFA pair the hedge is a "
           f"near-full short of the foreign basket (β = {R['h_beta']:.2f}, R² = {R['h_r2']:.2f}) and "
           f"pockets **+{R['h_carry']:.2f} %/yr at HAC *t* = +{R['h_t']:.2f}** — on the observable "
           f"+{R['h_obs']:.2f} %/yr differential, bootstrap CI [+{R['h_ci_lo']:.2f}, "
           f"+{R['h_ci_hi']:.2f}] clear of zero, *t* ≥ 2 in both eras (+{R['h_pre_t']:.2f} / "
           f"+{R['h_post_t']:.2f}), growing with the gap. DBEF/EFA corroborates (*t* = "
           f"+{R['d_t']:.2f}). The synthetic control recovers a *planted* carry (β = "
           f"{R['planted_beta']:.2f}, fires {R['null_fire']}/{R['null_seeds']} nulls).\n"
           f"- **Tradability — Fragile.** The carry is real and cheap to *hold* (~2 bp wrapper gap), "
           f"but **not bankable as an overlay**: the 'hedge when the US out-yields' switch adds "
           f"nothing over just always-hedging ({R['ov_sh']:.2f} vs {R['ov_hedged']:.2f} Sharpe, "
           f"{R['ov_switches']} switch in 12 yr); the excess-Sharpe advantage's CI overlaps and rests "
           f"on one dollar regime; isolating the pure carry is a dollar-long spread that nets "
           f"+{R['sp_net']:.2f} %/yr at only *t* = +{R['sp_t']:.2f}. Real but thin & un-timeable."),
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
