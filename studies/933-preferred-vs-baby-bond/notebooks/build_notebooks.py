"""Generate the two narrative notebooks for Study 933 (Same Issuer, Two Ladders).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from
the frozen ``R`` dict (mirroring docs/results.md); the only live cells run the fast
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


# Frozen real-tape headline — mirror of docs/results.md. Eight issuer pairs, daily
# total-return closes, equal-weight ladders, monthly rebalance with one execution lag,
# 25 bps one-way, excess-of-cash (BIL), 2022-02-02 -> 2026-06-30.
R = dict(
    start="2022-02-02", end="2026-06-30", n_days=1105, fp="9ae51d88019e", n_pairs=8,
    cost_bps=25, turnover=75,
    pref_sharpe=0.368, pref_cagr=5.35, pref_vol=18.96, pref_dd=-30.82, pref_t=0.68,
    bond_sharpe=0.202, bond_cagr=1.82, bond_vol=13.31, bond_dd=-20.40, bond_t=0.38,
    adv=0.166, t_ladder=0.65, ret_spread=4.29,
    pairwise=3.10, t_pairwise=0.55, vol_ratio=1.42,
    ci_pref_lo=-0.736, ci_pref_hi=1.351, ci_bond_lo=-0.823, ci_bond_hi=1.256,
    ci_adv_lo=-0.470, ci_adv_hi=0.804, ci_adv_neg=32.3,
    ci_pw_lo=-6.80, ci_pw_hi=13.57,
    wins=6, n_pairs_win=8, wilson_lo=0.41, wilson_hi=0.93,
    median_pair=0.70, riskier=6, ex_rily=0.81, t_ex_rily=0.17,
    rily_spread=19.15, rily_pref_cagr=-7.71, rily_bond_cagr=2.58,
    era_e_n=542, era_e_adv=-0.16, era_e_pw=-1.16, era_e_t=-0.21,
    era_l_n=564, era_l_adv=0.10, era_l_pw=7.17, era_l_t=0.72,
    cost5_adv=0.166, cost5_spread=4.33, cost100_adv=0.167, cost100_spread=4.14,
    ls_gross=4.29, ls_sharpe0=0.282, ls_t=0.65, ls_300=1.29, ls_sharpe300=0.085, ls_dd=-25.9,
    st_2022=0.06, st_banks=2.89, st_2024=5.28, st_tariff=-1.03, st_covid=9.53,
    covid_pref=-10.25, covid_bond=-19.78,
    long_n=1821, long_pw=-0.43, long_t=-0.14, long_adv=-0.064, long_volratio=0.95,
    alt_pw=3.84, alt_t=0.68, alt_adv=0.222,
    sach_pref=-17.4, sach_bond=21.6, sach_spread=-10.63, sach_t=-0.85,
    pff_sharpe=-0.191, pff_cagr=-2.60, pff_gap=-4.75, pff_t=-0.90, pff_corr=0.42,
    syn_pl=5.74, syn_pl_fire=20, syn_nl=-0.26, syn_nl_fire=0, syn_planted=6.0,
    # --- the audit probes: microstructure/frequency and concentration ---
    stale_worst="CMS-PB", stale_worst_zero=15.5, stale_worst_ac1=-0.28,
    stale_worst_vol_d=24.0, stale_worst_vol_m=11.4, stale_worst_bond_vol_m=13.5,
    stale_pref_mean=5.4, stale_bond_mean=5.0,
    riskier_d=6, riskier_w=4, riskier_m=2,
    ratio_d=1.31, ratio_w=1.30, ratio_m=1.30,
    ex2_pairs=6, ex2_pw=-0.60, ex2_t=-0.21, ex2_adv=-0.116,
    ex2_ratio_d=0.94, ex2_ratio_m=0.79, ex2_riskier_m=0,
    ex2_dd_pref=-20.1, ex2_dd_bond=-19.3,
    ex1_ratio_m=1.27, ex1_riskier_m=1, ex1_pairs=7,
    wide_step_pairs=2, narrow_step_pairs=6,
)


HEADER = f"""# Study 933 — Same Issuer, Two Ladders 🪜

**For one and the same balance sheet, does the junior rung pay you for standing behind the senior one?**

Eight US issuers list **two** $25-par instruments side by side: a **preferred** (the junior
rung — discretionary coupon, usually no maturity, behind every debt claim) and an
exchange-traded **baby bond**. Textbook risk pricing says the preferred must pay more. This
study holds the **obligor fixed** and measures how much.

*(The rungs are not uniform: only **{R['wide_step_pairs']} of 8** pairs are the textbook
"perpetual preferred vs *senior* note" — in four the note is itself junior subordinated or
perpetual, and in two the "preferred" is a dated term preferred. A one-rung step cannot pay a
two-rung premium; `data.RUNG_STRUCTURE` names each one.)*

The pairs: CMS Energy, Duke Energy, Brookfield Renewable, Brookfield Infrastructure,
B. Riley Financial, Oxford Lane Capital, Eagle Point Credit, Babcock & Wilcox — daily
**total-return** closes, {R['start']} → {R['end']} ({R['n_days']:,} days), two equal-weight
ladders rebalanced monthly with **one execution lag**, {R['cost_bps']} bps one-way, both
**excess-of-cash** (BIL).

*Numbers below are the frozen headline (`docs/results.md`, Fingerprint `{R['fp']}`); the live
cells run only the fast synthetic control. As-of 2026-06-30.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. Two rungs of one ladder\n\n"
           "Picture a company that owes money in two publicly listed ways. The **baby bond** is "
           "debt: a fixed maturity, a contractual coupon, and a legal queue position ahead of "
           "shareholders. The **preferred** is the rung below: no maturity date, a coupon the "
           "board can switch off, and a place in the queue behind every lender. Same company, "
           "same trucks and power plants and loan book — different seat on the lifeboat.\n\n"
           "So the preferred should pay you extra. The question is how much, and whether four "
           "and a half years of tape can actually see it."),
        code(
            "R = dict(pref_sharpe=%r, bond_sharpe=%r, pref_cagr=%r, bond_cagr=%r,\n"
            "         pref_vol=%r, bond_vol=%r, pref_dd=%r, bond_dd=%r,\n"
            "         pairwise=%r, t_pairwise=%r, vol_ratio=%r,\n"
            "         ex2_pw=%r, ex2_ratio_m=%r)\n"
            "print('Preferred ladder (junior): %%+.2f%%%% a year over cash, %%.1f%%%% swings, worst fall %%.1f%%%%'\n"
            "      %% (R['pref_cagr'], R['pref_vol'], R['pref_dd']))\n"
            "print('Baby-bond ladder (senior): %%+.2f%%%% a year over cash, %%.1f%%%% swings, worst fall %%.1f%%%%'\n"
            "      %% (R['bond_cagr'], R['bond_vol'], R['bond_dd']))\n"
            "print()\n"
            "print('extra paid for standing further back: %%+.2f%%%% a year  (t = %%+.2f)'\n"
            "      %% (R['pairwise'], R['t_pairwise']))\n"
            "print('extra risk taken for it            : %%.2fx the volatility'  %% R['vol_ratio'])\n"
            "print()\n"
            "print('...but drop the two distressed issuers and BOTH numbers invert:')\n"
            "print('   paid %%+.2f%%%% a year, and %%.2fx the volatility (i.e. the CALMER leg)'\n"
            "      %% (R['ex2_pw'], R['ex2_ratio_m']))"
            % (R["pref_sharpe"], R["bond_sharpe"], R["pref_cagr"], R["bond_cagr"],
               R["pref_vol"], R["bond_vol"], R["pref_dd"], R["bond_dd"],
               R["pairwise"], R["t_pairwise"], R["vol_ratio"],
               R["ex2_pw"], R["ex2_ratio_m"])
        ),
        md(f"## 2. The payment is a rumour\n\n"
           f"At first glance the numbers look like the textbook: the junior ladder swung "
           f"**{R['vol_ratio']:.2f}×** as hard, fell **{R['pref_dd']:.1f}%** at worst against "
           f"**{R['bond_dd']:.1f}%**, and was the more volatile leg in "
           f"**{R['riskier_d']} of {R['n_pairs']}** pairs — more risk, and "
           f"**+{R['pairwise']:.2f}%/yr** more return to go with it.\n\n"
           f"Both halves of that sentence fall apart on inspection. Start with the return: "
           f"**+{R['pairwise']:.2f}%/yr** sounds like a premium until you see the error bars — "
           f"the plausible range runs from **{R['ci_pw_lo']:.1f}%** to "
           f"**+{R['ci_pw_hi']:.1f}%**. That is not a measurement, it is a shrug. (Sections 3 "
           f"and 4 take the risk half apart.)\n\n"
           f"> 🔬 *For the quants:* HAC (Newey-West) *t* = **{R['t_pairwise']:+.2f}** on the "
           f"pairwise daily difference; the Sharpe advantage **{R['adv']:+.3f}** carries a "
           f"joint block-bootstrap CI of **[{R['ci_adv_lo']:+.2f}, {R['ci_adv_hi']:+.2f}]**, "
           f"with **{R['ci_adv_neg']:.0f}%** of resamples negative. And note what the pairwise "
           f"spread *is*: an arithmetic mean of daily differences, re-equal-weighted every day "
           f"and charged no cost — a statistic, not a portfolio."),
        md(f"## 3. Two companies are carrying the whole average\n\n"
           f"**B. Riley Financial** went through a genuine distress episode and back. Its "
           f"preferred crashed and rebounded so violently that, averaged day by day, it "
           f"contributes a **+{R['rily_spread']:.1f}%/yr** spread — enough on its own to make "
           f"the panel look like it pays a premium. Take it out and the number falls to "
           f"**+{R['ex_rily']:.2f}%/yr** (*t* = {R['t_ex_rily']:+.2f}).\n\n"
           f"Take out **Babcock & Wilcox** as well — the other issuer that spent this window in "
           f"trouble — and the premium doesn't just shrink, it **changes sign**: "
           f"**{R['ex2_pw']:.2f}%/yr** across the remaining {R['ex2_pairs']} pairs, with a "
           f"Sharpe advantage of **{R['ex2_adv']:+.3f}**. The typical pair pays "
           f"**+{R['median_pair']:.2f}%/yr** — about a fifth of a percent a quarter, for "
           f"standing behind every lender.\n\n"
           f"> 🔬 *For the quants:* and even B. Riley's headline number is a **Jensen trap** — "
           f"its *arithmetic* daily spread is +{R['rily_spread']:.1f}%/yr while its *compounded* "
           f"outcome is the opposite ({R['rily_pref_cagr']:+.2f}%/yr CAGR on the preferred vs "
           f"{R['rily_bond_cagr']:+.2f}% on the bond). The investor who actually held it lost."),
        md(f"## 4. And the 'at least it's riskier' consolation prize isn't real either\n\n"
           f"If the junior rung isn't *paid* more, you would at least expect the textbook's "
           f"other half to hold — that it is reliably the more dangerous seat. It isn't. Two "
           f"things were hiding inside that **{R['riskier_d']}/{R['n_pairs']}** count.\n\n"
           f"**These tickers barely trade.** A $25-par preferred can go a whole day without a "
           f"single print, and then bounce between bid and offer the next. CMS Energy's "
           f"preferred shows **no price change at all on {R['stale_worst_zero']:.1f}%** of days "
           f"and an up-down-up ping-pong pattern (lag-1 autocorrelation "
           f"**{R['stale_worst_ac1']:+.2f}**). Measured day by day it looks like a "
           f"**{R['stale_worst_vol_d']:.1f}%**-volatility instrument; measured month by month, "
           f"where the ping-pong cancels out, it is **{R['stale_worst_vol_m']:.1f}%** — *below* "
           f"its own baby bond's {R['stale_worst_bond_vol_m']:.1f}%. Widen the measuring window "
           f"and the count falls **{R['riskier_d']} → {R['riskier_w']} → {R['riskier_m']}** "
           f"(daily → weekly → monthly).\n\n"
           f"**And the two survivors are the same two distressed companies.** Drop B. Riley and "
           f"Babcock & Wilcox and the junior ladder becomes the *calmer* one: volatility ratio "
           f"**{R['ex2_ratio_m']:.2f}** monthly, junior riskier in **{R['ex2_riskier_m']} of "
           f"{R['ex2_pairs']}** pairs, worst falls of **{R['ex2_dd_pref']:.1f}%** vs "
           f"**{R['ex2_dd_bond']:.1f}%** — indistinguishable.\n\n"
           f"So the junior rung is the riskier one *when the company is in trouble*. That is a "
           f"statement about trouble, not about seniority — and it is not something the market "
           f"was paying you in advance to bear."),
        md(f"## 5. And seniority didn't save you when it mattered\n\n"
           f"Being senior is supposed to buy protection. On this tape it mostly didn't:\n\n"
           f"| when it got ugly | preferred | baby bond | who won |\n"
           f"|---|--:|--:|---|\n"
           f"| 2022 rate shock | −18.7% | −18.7% | a dead heat |\n"
           f"| March-2023 bank stress | +0.6% | −2.3% | **the junior rung** |\n"
           f"| 2024 issuer stress | +13.9% | +8.6% | **the junior rung** |\n"
           f"| 2025 tariff shock | −10.6% | −9.5% | the senior rung, barely |\n"
           f"| COVID crash (2 long-history pairs) | **{R['covid_pref']:.1f}%** | "
           f"**{R['covid_bond']:.1f}%** | **the junior rung, by 9.5 pp** |\n\n"
           f"The reason is unglamorous: between two rungs of one issuer, what usually differs "
           f"most is **duration**, not seniority. A perpetual preferred and a 40-year note are "
           f"both long; a 2028 note is not. In 2022 and in COVID the market was pricing "
           f"interest rates, and the 'safer' rung was simply the more rate-sensitive one."),
        md(f"## 6. Two more reasons to distrust the +{R['pairwise']:.1f}%\n\n"
           f"**The screen throws away the counter-examples.** We could only include pairs where "
           f"*both* rungs are still listed in June 2026 — which quietly deletes every redeemed "
           f"note and called preferred. The one dead pair we can still measure (Sachem Capital, "
           f"whose bond matured in December 2024) ran the other way entirely: the preferred "
           f"returned **{R['sach_pref']:.1f}%** while the bond returned "
           f"**+{R['sach_bond']:.1f}%** — a spread of **{R['sach_spread']:.2f}%/yr**.\n\n"
           f"**The broad market disagrees with our eight names.** Over exactly this window the "
           f"whole preferred sector (PFF) **lost {abs(R['pff_gap']):.2f}%/yr** to our "
           f"baby-bond basket. Our preferreds won; preferreds in general did not."),
        md("## 7. Live check — the yardstick works (offline synthetic)\n\n"
           "Before believing a null result, check the instrument can detect anything at all. We "
           "build a fake world of eight issuers where the junior rung genuinely *is* paid an "
           "extra 6% a year, and a matched world where it is paid exactly nothing."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from two_ladders import data, strategy as st\n"
            "paid   = st.synthetic_detect(*data.synthetic_panel(signal_strength=1.0, seed=933)[:3])\n"
            "unpaid = st.synthetic_detect(*data.synthetic_panel(signal_strength=0.0, seed=933)[:3])\n"
            "print('world where the junior rung IS paid +6.0%%/yr : measured %+.2f%%/yr  (t = %+.2f)'\n"
            "      % (paid['pairwise_spread_ann']*100, paid['t_pairwise']))\n"
            "print('world where it is paid nothing               : measured %+.2f%%/yr  (t = %+.2f)'\n"
            "      % (unpaid['pairwise_spread_ann']*100, unpaid['t_pairwise']))"
        ),
        md(f"On this single draw the yardstick picks the real premium out clearly (a *t* well "
           f"past 2) and fails to reach significance in the world where nothing is paid — and "
           f"across 20 draws (`docs/results.md`) it fires **{R['syn_pl_fire']}/20** times on the "
           f"planted world and **{R['syn_nl_fire']}/20** on the null. So the silence on the real "
           f"tape is the tape's, not the measuring stick's."),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The seniority premium is not measurable here: "
           f"**+{R['pairwise']:.2f}%/yr** with *t* = **{R['t_pairwise']:+.2f}** and a CI of "
           f"**[{R['ci_pw_lo']:.1f}%, +{R['ci_pw_hi']:.1f}%]**; the sign flips between the two "
           f"halves of the sample; it collapses to **+{R['ex_rily']:.2f}%** without one issuer "
           f"and to **{R['ex2_pw']:.2f}%** without two; and over the longer seven-year two-pair "
           f"tape it is **{R['long_pw']:+.2f}%/yr**. Nor is there a consolation risk premium to "
           f"point at: the junior rung's apparent extra risk is a **thin-tape artefact** plus "
           f"**two distressed names**, and outside them the junior rung is the calmer leg in "
           f"**{R['ex2_riskier_m']} of {R['ex2_pairs']}** pairs.\n"
           f"- **Tradability — Mirage.** Long the preferreds, short the bonds: "
           f"**+{R['ls_gross']:.2f}%/yr** gross at Sharpe {R['ls_sharpe0']:.2f} on a *t* of "
           f"{R['ls_t']:.2f}, down to Sharpe {R['ls_sharpe300']:.2f} once you assume a realistic "
           f"borrow cost, with a {R['ls_dd']:.0f}% drawdown along the way. If you want the "
           f"issuer's credit, the senior rung gave you two-thirds of the return for two-thirds "
           f"of the risk — and slept better."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 933 — Same Issuer, Two Ladders — the teardown\n\n"
           "The pairwise same-issuer estimator, the Newey-West *t*, joint block-bootstrap CIs, "
           "the per-issuer table and its Jensen trap, the era cut, the cost and borrow sweeps, "
           "the stress windows, the long-history extension, the survivorship probe, the PFF "
           "fallback and the live synthetic control. Every real number is frozen from "
           "`docs/results.md` (Fingerprint `%s`).\n\n"
           "**Design.** Two equal-weight ladders over %d issuer pairs, monthly rebalance, "
           "weights formed through day *t* and applied at *t+1* (the study's only execution "
           "lag), %d bps one-way x NAV on turnover (a PROXY), both series taken "
           "excess-of-cash (BIL total return). Daily **total-return** closes throughout — "
           "price-only is meaningless between two coupon instruments."
           % (R["fp"], R["n_pairs"], R["cost_bps"])),
        code("R = %r" % (R,)),
        md("## 1. Why *pairwise*\n\n"
           "A preferred-vs-bond race across *different* issuers confounds seniority with credit "
           "quality, sector and duration. Holding the obligor fixed makes the issuer's spread "
           "factor cancel in the difference:\n\n"
           "$$ d_{i,t} = r^{pref}_{i,t} - r^{bond}_{i,t} = (\\beta^{pref}_i - \\beta^{bond}_i) f_{i,t} + \\Delta\\text{carry}_i + \\varepsilon_{i,t} $$\n\n"
           "so $\\overline{d}$ estimates the paid seniority premium plus a residual beta term, "
           "not a cross-sectional credit tilt. The cost is $n$: only a handful of US issuers "
           "have ever had both rungs listed at once.\n\n"
           "> 💡 *In plain words:* comparing two rungs of the same company removes the question "
           "'is this a better company?' and leaves only 'is this a worse seat?'."),
        code(
            "print(f\"preferred ladder : exSharpe {R['pref_sharpe']:+.3f}  CAGR {R['pref_cagr']:+.2f}%  \"\n"
            "      f\"vol {R['pref_vol']:.2f}%  DD {R['pref_dd']:.1f}%  HAC t {R['pref_t']:+.2f}\")\n"
            "print(f\"baby-bond ladder : exSharpe {R['bond_sharpe']:+.3f}  CAGR {R['bond_cagr']:+.2f}%  \"\n"
            "      f\"vol {R['bond_vol']:.2f}%  DD {R['bond_dd']:.1f}%  HAC t {R['bond_t']:+.2f}\")\n"
            "print()\n"
            "print(f\"Sharpe advantage (pref - bond) : {R['adv']:+.3f}\")\n"
            "print(f\"ladder return spread          : {R['ret_spread']:+.2f}%/yr  (HAC t = {R['t_ladder']:+.2f})\")\n"
            "print(f\"PAIRWISE same-issuer spread   : {R['pairwise']:+.2f}%/yr  (HAC t = {R['t_pairwise']:+.2f})\")\n"
            "print(f\"vol ratio pref/bond {R['vol_ratio']:.2f}   annual ladder turnover {R['turnover']}%\")"
        ),
        md("## 2. Bootstrap — the arms are resampled *jointly*\n\n"
           "The two rungs of one balance sheet are strongly contemporaneously correlated; "
           "resampling them independently would badly overstate the precision of the "
           "difference. 2,000 circular block draws, 21-day blocks, same block index for both "
           "arms."),
        code(
            "print(f\"pref ladder exSharpe {R['pref_sharpe']:+.3f}  95% CI [{R['ci_pref_lo']:+.3f}, {R['ci_pref_hi']:+.3f}]\")\n"
            "print(f\"bond ladder exSharpe {R['bond_sharpe']:+.3f}  95% CI [{R['ci_bond_lo']:+.3f}, {R['ci_bond_hi']:+.3f}]\")\n"
            "print(f\"Sharpe ADVANTAGE     {R['adv']:+.3f}  95% CI [{R['ci_adv_lo']:+.3f}, {R['ci_adv_hi']:+.3f}]  \"\n"
            "      f\"share<0 {R['ci_adv_neg']:.1f}%\")\n"
            "print(f\"pairwise spread      {R['pairwise']:+.2f}%/yr  95% CI [{R['ci_pw_lo']:+.2f}%, {R['ci_pw_hi']:+.2f}%]\")"
        ),
        md(f"## 3. The cross-section — and a Jensen trap\n\n"
           f"Wins (spread > 0): **{R['wins']}/{R['n_pairs_win']}**, Wilson 95% "
           f"**[{R['wilson_lo']:.2f}, {R['wilson_hi']:.2f}]** — the hit rate cannot reject a "
           f"coin. Median pair spread **+{R['median_pair']:.2f}%/yr** against a panel mean of "
           f"**+{R['pairwise']:.2f}%**: the mean is not the typical pair.\n\n"
           f"**B. Riley** contributes an arithmetic **+{R['rily_spread']:.1f}%/yr** while its "
           f"compounded outcome is the *opposite sign* ({R['rily_pref_cagr']:+.2f}% vs "
           f"{R['rily_bond_cagr']:+.2f}% CAGR). A 97%-vol crash-and-rebound inflates the "
           f"arithmetic mean by roughly $\\sigma^2/2$; the geometric investor never saw it.\n\n"
           f"> 💡 *In plain words:* the one number that makes the premium look real is an "
           f"artefact of averaging daily percentage moves on something that halved and doubled."),
        code(
            "print(f\"wins {R['wins']}/{R['n_pairs_win']}  Wilson [{R['wilson_lo']:.2f}, {R['wilson_hi']:.2f}]\")\n"
            "print(f\"median pair spread {R['median_pair']:+.2f}%/yr   vs panel mean {R['pairwise']:+.2f}%/yr\")\n"
            "print(f\"drop B. Riley -> pairwise {R['ex_rily']:+.2f}%/yr (t = {R['t_ex_rily']:+.2f})\")\n"
            "print(f\"drop B. Riley + B&W -> pairwise {R['ex2_pw']:+.2f}%/yr (t = {R['ex2_t']:+.2f})  \"\n"
            "      f\"adv {R['ex2_adv']:+.3f}  -> SIGN FLIP on 6 remaining pairs\")"
        ),
        md(f"## 3b. The risk ordering is an artefact too\n\n"
           f"The tempting fallback — *fine, the premium is unmeasurable, but at least the junior "
           f"rung is reliably riskier* — does not survive two checks.\n\n"
           f"**Frequency.** These are thin retail tapes: mean stale-print share "
           f"**{R['stale_pref_mean']:.1f}% / {R['stale_bond_mean']:.1f}%** (pref / bond), with "
           f"{R['stale_worst']} at **{R['stale_worst_zero']:.1f}%** and lag-1 AC "
           f"**{R['stale_worst_ac1']:+.2f}** — textbook bid-ask bounce, which *inflates* measured "
           f"daily variance. Compounding the same returns up:\n\n"
           f"| frequency | vol ratio | pref riskier |\n|---|--:|--:|\n"
           f"| daily | {R['ratio_d']:.2f} | **{R['riskier_d']}/8** |\n"
           f"| weekly | {R['ratio_w']:.2f} | {R['riskier_w']}/8 |\n"
           f"| monthly | {R['ratio_m']:.2f} | **{R['riskier_m']}/8** |\n\n"
           f"{R['stale_worst']} alone goes from **{R['stale_worst_vol_d']:.1f}%** daily vol to "
           f"**{R['stale_worst_vol_m']:.1f}%** monthly — below its own bond's "
           f"**{R['stale_worst_bond_vol_m']:.1f}%**.\n\n"
           f"**Concentration.** The panel ratio holds up only because two names dominate an "
           f"equal-weight variance. Drop B. Riley and Babcock & Wilcox:\n\n"
           f"| panel | pairwise (*t*) | adv | ratio d / m | pref riskier (m) | DD pref vs bond |\n"
           f"|---|--:|--:|--:|--:|--:|\n"
           f"| all 8 | +{R['pairwise']:.2f}% ({R['t_pairwise']:+.2f}) | {R['adv']:+.3f} | "
           f"{R['ratio_d']:.2f} / {R['ratio_m']:.2f} | {R['riskier_m']}/8 | "
           f"{R['pref_dd']:.1f}% vs {R['bond_dd']:.1f}% |\n"
           f"| ex B. Riley | +{R['ex_rily']:.2f}% ({R['t_ex_rily']:+.2f}) | +0.176 | "
           f"1.30 / {R['ex1_ratio_m']:.2f} | {R['ex1_riskier_m']}/{R['ex1_pairs']} | −23.1% vs −19.4% |\n"
           f"| **ex both** | **{R['ex2_pw']:.2f}%** ({R['ex2_t']:+.2f}) | **{R['ex2_adv']:+.3f}** | "
           f"**{R['ex2_ratio_d']:.2f} / {R['ex2_ratio_m']:.2f}** | "
           f"**{R['ex2_riskier_m']}/{R['ex2_pairs']}** | {R['ex2_dd_pref']:.1f}% vs "
           f"{R['ex2_dd_bond']:.1f}% |\n\n"
           f"> 💡 *In plain words:* the junior rung is the wilder one **when the issuer is in "
           f"distress** — a tautology about distress, not a price of subordination. In the "
           f"{R['ex2_pairs']} non-distressed pairs there is no step in either direction, and it "
           f"is worth noting those are also the pairs whose seniority gap is narrowest "
           f"(`data.RUNG_STRUCTURE`: only {R['wide_step_pairs']} of 8 pairs put a *senior* note "
           f"against a perpetual preferred)."),
        code(
            "for f, ratio, riskier in ((\"daily\", R['ratio_d'], R['riskier_d']),\n"
            "                          (\"weekly\", R['ratio_w'], R['riskier_w']),\n"
            "                          (\"monthly\", R['ratio_m'], R['riskier_m'])):\n"
            "    print(f\"{f:8s} vol ratio {ratio:.2f}   pref riskier in {riskier}/8 pairs\")\n"
            "print()\n"
            "print(f\"ex B.Riley + B&W: ratio {R['ex2_ratio_d']:.2f} daily / {R['ex2_ratio_m']:.2f} monthly, \"\n"
            "      f\"pref riskier {R['ex2_riskier_m']}/{R['ex2_pairs']}, \"\n"
            "      f\"DD {R['ex2_dd_pref']:.1f}% vs {R['ex2_dd_bond']:.1f}%\")\n"
            "print('-> neither the return step nor the risk step is a seniority fact')"
        ),
        md("## 4. Era cut, cost sweep, borrow sweep\n\n"
           "The spread changes sign across the halves; cost is irrelevant (it cancels between "
           "two near-identical-turnover ladders); the borrow assumption is what actually kills "
           "the dollar-neutral expression."),
        code(
            "print(f\"early n={R['era_e_n']}: adv {R['era_e_adv']:+.2f}  pairwise {R['era_e_pw']:+.2f}% (t={R['era_e_t']:+.2f})\")\n"
            "print(f\"late  n={R['era_l_n']}: adv {R['era_l_adv']:+.2f}  pairwise {R['era_l_pw']:+.2f}% (t={R['era_l_t']:+.2f})  <- sign flip\")\n"
            "print()\n"
            "print(f\"cost   5 bps: adv {R['cost5_adv']:+.3f}  spread {R['cost5_spread']:+.2f}%\")\n"
            "print(f\"cost 100 bps: adv {R['cost100_adv']:+.3f}  spread {R['cost100_spread']:+.2f}%  -> cost is not the story\")\n"
            "print()\n"
            "print(f\"long-short, borrow   0 bp: {R['ls_gross']:+.2f}%/yr  Sharpe {R['ls_sharpe0']:.3f}  t {R['ls_t']:+.2f}\")\n"
            "print(f\"long-short, borrow 300 bp: {R['ls_300']:+.2f}%/yr  Sharpe {R['ls_sharpe300']:.3f}  DD {R['ls_dd']:.1f}%\")"
        ),
        md("## 5. Stress windows, long history, and what the screen deletes\n\n"
           "Seniority is meant to be worth something precisely when the issuer's credit is "
           "questioned. It wasn't, in four of five windows — because between two rungs of one "
           "issuer the dominant difference is usually **duration**, not subordination."),
        code(
            "print(f\"2022 rate shock   gap {R['st_2022']:+.2f} pp   (a dead heat)\")\n"
            "print(f\"Mar-2023 banks    gap {R['st_banks']:+.2f} pp   (junior rung won)\")\n"
            "print(f\"2024 issuer strss gap {R['st_2024']:+.2f} pp   (junior rung won)\")\n"
            "print(f\"2025 tariff shock gap {R['st_tariff']:+.2f} pp   (senior rung won, barely)\")\n"
            "print(f\"COVID (2-pair)    gap {R['st_covid']:+.2f} pp   (junior rung won by 9.5 pp)\")\n"
            "print()\n"
            "print(f\"long-history CMS+Duke, n={R['long_n']}: pairwise {R['long_pw']:+.2f}%/yr (t={R['long_t']:+.2f})  \"\n"
            "      f\"adv {R['long_adv']:+.3f}  vol ratio {R['long_volratio']:.2f}\")\n"
            "print(f\"alternative rung (CMSA/BEPI/BIPI): pairwise {R['alt_pw']:+.2f}%/yr (t={R['alt_t']:+.2f})  \"\n"
            "      f\"adv {R['alt_adv']:+.3f}  -> not a CUSIP artefact\")\n"
            "print()\n"
            "print(f\"SURVIVORSHIP  Sachem SACH-PA vs SACC (bond matured 2024-12):\")\n"
            "print(f\"   pref cum {R['sach_pref']:+.1f}%  bond cum {R['sach_bond']:+.1f}%  \"\n"
            "      f\"spread {R['sach_spread']:+.2f}%/yr (t={R['sach_t']:+.2f})  -> the screen deletes the losers\")\n"
            "print(f\"FALLBACK      PFF exSharpe {R['pff_sharpe']:+.3f} (CAGR {R['pff_cagr']:+.2f}%);  \"\n"
            "      f\"PFF - baby bonds {R['pff_gap']:+.2f}%/yr (t={R['pff_t']:+.2f})\")"
        ),
        md("## 6. Live synthetic control — power and calibration\n\n"
           "Planted world: eight issuers, junior rung paid a real +6.0%/yr carry on top of its "
           "higher factor loading. Null world: identical betas, premium exactly zero. The "
           "estimator must fire on the first and stay silent on the second. This proves the "
           "machinery; it never supports the real-tape stamp."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from two_ladders import data, strategy as st\n"
            "for ss, tag in ((1.0, 'planted +6.0%/yr'), (0.0, 'exact null    ')):\n"
            "    sp, ts = [], []\n"
            "    for k in range(8):\n"
            "        d = st.synthetic_detect(*data.synthetic_panel(signal_strength=ss, seed=933+k)[:3])\n"
            "        sp.append(d['pairwise_spread_ann']); ts.append(d['t_pairwise'])\n"
            "    sp, ts = np.array(sp), np.array(ts)\n"
            "    print(f\"{tag}: spread {sp.mean()*100:+.2f}%/yr (sd {sp.std(ddof=1)*100:.2f}%)  \"\n"
            "          f\"|t|>=2 in {int((abs(ts)>=2).sum())}/8\")"
        ),
        md(f"(The frozen 20-seed run in `docs/results.md`: planted **+{R['syn_pl']:.2f}%/yr**, "
           f"fires **{R['syn_pl_fire']}/20**; null **{R['syn_nl']:+.2f}%/yr**, fires "
           f"**{R['syn_nl_fire']}/20**.)"),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** Pairwise same-issuer spread **+{R['pairwise']:.2f}%/yr**, HAC "
           f"*t* = **{R['t_pairwise']:+.2f}**, bootstrap CI **[{R['ci_pw_lo']:.2f}%, "
           f"+{R['ci_pw_hi']:.2f}%]**; Sharpe advantage **{R['adv']:+.3f}**, CI "
           f"**[{R['ci_adv_lo']:+.2f}, {R['ci_adv_hi']:+.2f}]** with "
           f"{R['ci_adv_neg']:.0f}% of draws negative. Sign flips by era "
           f"({R['era_e_pw']:+.2f}% / {R['era_l_pw']:+.2f}%); collapses to "
           f"**+{R['ex_rily']:.2f}% (t={R['t_ex_rily']:+.2f})** ex-B. Riley; median pair "
           f"**+{R['median_pair']:.2f}%**; hit rate {R['wins']}/{R['n_pairs_win']} with a Wilson "
           f"interval spanning a coin flip; the seven-year two-pair extension is "
           f"**{R['long_pw']:+.2f}%/yr**; the survivorship probe runs **{R['sach_spread']:.2f}%/yr** "
           f"the *other* way; and PFF **lost {abs(R['pff_gap']):.2f}%/yr** to the baby-bond basket "
           f"on the same window. The synthetic control recovers a planted +6%/yr on "
           f"{R['syn_pl_fire']}/20 seeds and is silent on {R['syn_nl_fire']}/20 nulls, so the "
           f"estimator has the power the tape lacks. And the risk ordering that once looked "
           f"robust ({R['vol_ratio']:.2f}x vol, riskier in {R['riskier_d']}/{R['n_pairs']} pairs) "
           f"is **daily microstructure plus two distressed names**: {R['riskier_m']}/8 at "
           f"monthly frequency, and {R['ex2_riskier_m']}/{R['ex2_pairs']} once those two names "
           f"leave, where the ratio itself is {R['ex2_ratio_m']:.2f}.\n"
           f"- **Tradability — Mirage.** Dollar-neutral long-pref / short-bond: "
           f"**{R['ls_gross']:+.2f}%/yr** gross, Sharpe {R['ls_sharpe0']:.3f}, *t* "
           f"{R['ls_t']:+.2f}, DD {R['ls_dd']:.1f}%; Sharpe {R['ls_sharpe300']:.3f} at a 300 bp "
           f"borrow ASSUMPTION. Cost is irrelevant "
           f"(adv {R['cost5_adv']:+.3f} → {R['cost100_adv']:+.3f} from 5 to 100 bps) — the "
           f"signal is what is missing. The directional tilt is no better: the long-history "
           f"extension pays the junior rung **less**, and the live-listing screen deletes the "
           f"pairs that would drag the panel down.\n"
           f"- **Named limits.** 4.4-year common window, one rate cycle, no default cycle, "
           f"8 correlated pairs; survivorship on both rungs; a 25 bps cost PROXY and a borrow "
           f"ASSUMPTION, both swept; thin tapes (stale-print shares to "
           f"{R['stale_worst_zero']:.1f}%, lag-1 AC to {R['stale_worst_ac1']:+.2f}), which is why "
           f"every risk claim is stated at three frequencies; and a seniority step that is only "
           f"the textbook one in {R['wide_step_pairs']} of 8 pairs "
           f"(`data.RUNG_STRUCTURE`) — a limit of the *listed* universe, not a fixable choice."),
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
