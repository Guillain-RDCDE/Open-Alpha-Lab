"""Generate the two narrative notebooks for Study 915 (K-1 vs 1099).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below, which mirrors ``docs/results.md`` line for line; the only live
cells run the fast synthetic control, and they are always labelled as synthetic. No cell
reads the real cache, so the notebooks execute identically on a fresh checkout.
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


# --------------------------------------------------------------------------- #
# Frozen real-tape headline — mirror of docs/results.md.
# PDBC (Form 1099) vs DBC (Schedule K-1), cash BIL, total return,
# 2014-11-07 -> 2026-06-30, as-of 2026-06-30, fingerprint 0e33b87210d5.
# --------------------------------------------------------------------------- #
R = dict(
    start="2014-11-07", end="2026-06-30", n_days=2926, fp="0e33b87210d5",
    # layer 1 — the pre-tax race
    ric_sharpe=0.151, ric_cagr=3.01, ric_vol=18.1, ric_dd=-49.3, ric_t=0.51,
    k1_sharpe=0.159, k1_cagr=3.16, k1_vol=17.9, k1_dd=-50.6, k1_t=0.53,
    diff_ann=-0.1274, t_diff=-0.21, te_ann=5.30, corr=0.9566, beta=0.9626, r2=0.9151,
    sharpe_adv=-0.0081,
    ci_lo=-1.0323, ci_hi=0.7768, ci_pneg=0.618,
    adv_ci_lo=-0.0587, adv_ci_hi=0.0421,
    # fee proxy
    er_k1=0.87, er_ric=0.59, fee_gap_bp=28,
    # microstructure
    te_monthly=1.46, n_months=140, acf1=-0.430, diff_monthly=-0.1945, t_monthly=-0.68,
    # power
    se_iid=1.56, se_hac=0.61, se_boot=0.45,
    mde_iid=3.11, mde_hac=1.22, mde_boot=0.91,
    # eras
    era_e_n=1547, era_e_diff=0.008, era_e_t=0.01, era_e_te=7.04,
    era_l_n=1377, era_l_diff=-0.319, era_l_t=-0.73, era_l_te=2.00,
    # calendar years
    n_years=11, wins=3, win_rate=27, win_lo=10, win_hi=57,
    worst_year=2025, worst_diff=-2.14, best_year=2015, best_diff=0.74,
    # cost sweep
    cost50_diff=-0.2136, cost50_t=-0.35, cost50_drag_bp=8.62,
    # after-tax model
    gap_min=-0.360, gap_max=0.588,
    gap_top_p0=0.588, gap_top_p1=0.219, gap_top_p2=-0.340,
    gap_mid_p1=0.149, gap_low_p1=0.021,
    regime_gap_low=0.241, regime_gap_mid=0.380, regime_gap_top=0.438,
    pretax_cagr_k1=3.328, pretax_cagr_ric=3.055,
    aftertax_k1_mid=2.034, aftertax_ric_mid=2.183,
    # context
    usci_cagr=4.68, uso_cagr=-6.70, bno_cagr=1.85, dispersion_pp=11.38,
    # synthetic control
    syn_planted=4.00, syn_recovered=4.95, syn_t=-4.03,
    syn_null_mean=0.453, syn_null_sd=1.057, syn_null_fire=0,
)


HEADER = f"""# Study 915 — K-1 vs 1099 🧾

**Does the tax-friendly commodity wrapper give back its convenience in performance?**

Buy a commodity ETP that holds futures directly and you own a **partnership**. Every
spring it sends you a **Schedule K-1** — often late enough to force a filing extension —
and inside an IRA it can generate taxable income the account was supposed to shelter. In
exchange, its futures gains fall under **Section 1256**: taxed **60% long-term / 40%
short-term** whatever your holding period, but **marked to market every 31 December**
whether or not you sell.

**PDBC** is the same manager's answer, branded literally *"No K-1"*. It pushes the futures
into a Cayman subsidiary so it can be a normal 1940-Act fund and send you a **Form 1099**.
The trade: you lose the 60/40 rate, you receive ordinary-income distributions — and you
gain **deferral** on everything else.

The desk's question is not which structure is prettier. It is whether the convenience shows
up as a **cost on the tape**. We race **PDBC (1099)** against **DBC (K-1)** — same manager,
same index family, same commodity beta — on daily **total-return** closes,
{R['start']} → {R['end']} ({R['n_days']:,} days), both legs **excess of cash** (BIL).

*Numbers below are the frozen headline (`docs/results.md`, Fingerprint `{R['fp']}`); the
only live cells run a synthetic control and say so. As-of 2026-06-30.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. Two envelopes, one commodity\n\n"
           "DBC and PDBC are run by the same house, on the same strategy family, holding "
           "essentially the same futures. They differ in the *legal envelope* around them — "
           "and therefore in the piece of paper you get in the post each spring. If the "
           "convenient envelope quietly costs you money, the two tapes should drift apart. "
           "Here is what eleven and a half years of tape says."),
        code(
            "R = " + repr(R) + "\n\n"
            "print(f\"DBC  (Schedule K-1): CAGR {R['k1_cagr']:+.2f}%   \"\n"
            "      f\"excess-of-cash Sharpe {R['k1_sharpe']:+.3f}\")\n"
            "print(f\"PDBC (Form 1099)  : CAGR {R['ric_cagr']:+.2f}%   \"\n"
            "      f\"excess-of-cash Sharpe {R['ric_sharpe']:+.3f}\")\n"
            "print()\n"
            "print(f\"difference (1099 - K-1) : {R['diff_ann']:+.3f}% a year    t = {R['t_diff']:+.2f}\")\n"
            "print(f\"95% confidence interval : [{R['ci_lo']:+.2f}%, {R['ci_hi']:+.2f}%] a year\")\n"
            "print(f\"day-to-day correlation  : {R['corr']:.3f}\")"
        ),
        md(f"## 2. The answer: nothing to see\n\n"
           f"The 1099 wrapper trailed its K-1 twin by **{abs(R['diff_ann']):.2f}% a year** — "
           f"about a tenth of one percent. That is not a finding; it is a rounding error. "
           f"The *t*-statistic is **{R['t_diff']:+.2f}** (anything under 2 means "
           f"\"indistinguishable from zero\"), and the honest range around it runs from "
           f"**{R['ci_lo']:+.2f}% to {R['ci_hi']:+.2f}%** a year — comfortably straddling "
           f"nothing at all.\n\n"
           f"On risk-adjusted terms the gap is even smaller: an excess-of-cash Sharpe "
           f"difference of **{R['sharpe_adv']:+.4f}**, with a range of "
           f"[{R['adv_ci_lo']:+.3f}, {R['adv_ci_hi']:+.3f}]. For practical purposes these "
           f"are the same investment wearing different paperwork."),
        md(f"## 3. \"But my null might just be blind\"\n\n"
           f"A finding of *nothing* is only worth reading if the test could have found "
           f"*something*. So we asked the sample how big a wrapper cost it would have "
           f"caught. Answer: anything above roughly **{R['mde_boot']:.1f}-{R['mde_hac']:.1f}% "
           f"a year** would have shown up clearly.\n\n"
           f"So the correct sentence is not \"there is no cost\" — it is **\"there is no cost "
           f"bigger than about 1% a year\"**. Given that the entire fee difference between "
           f"the two funds is {R['fee_gap_bp']} basis points, that is a useful bound.\n\n"
           f"> 🔬 **For the quants:** the minimum detectable difference at |*t*| = 2 is "
           f"{R['mde_hac']:.2f}%/yr on the HAC standard error and {R['mde_boot']:.2f}%/yr on "
           f"the 21-day block bootstrap; the naive i.i.d. figure ({R['mde_iid']:.2f}%/yr) is "
           f"three times too pessimistic because the daily difference is strongly "
           f"*negatively* autocorrelated."),
        md(f"## 4. The 5% \"tracking error\" that isn't\n\n"
           f"Measured day by day, the two funds look like they wander {R['te_ann']:.1f}% "
           f"apart a year — which sounds alarming. Measure the same thing month by month and "
           f"it collapses to **{R['te_monthly']:.2f}%**.\n\n"
           f"The reason is mundane: the two closing prints do not clear at the same instant, "
           f"and each carries its own small premium or discount to the fund's true value. "
           f"They disagree on Monday and re-converge on Tuesday — the daily difference has a "
           f"lag-1 correlation of **{R['acf1']:+.2f}**. That is stock-exchange plumbing, not "
           f"divergence a holder ever experiences."),
        md(f"## 5. Year by year — and the one year that looks like a story\n\n"
           f"Across the {R['n_years']} complete calendar years the 1099 wrapper won "
           f"**{R['wins']} of {R['n_years']}** ({R['win_rate']}%, and a coin flip sits inside "
           f"the honest range of {R['win_lo']}-{R['win_hi']}%). Nine of the eleven annual "
           f"differences are inside ±1 percentage point.\n\n"
           f"The exception is **{R['worst_year']}**, where PDBC trailed by "
           f"**{abs(R['worst_diff']):.2f} pp**. PDBC is *actively* managed — it does not "
           f"mechanically copy DBC's index — so it is allowed to hold a different basket, "
           f"and manager drift is the natural explanation. Be clear that this is an "
           f"**interpretation, not a measurement**: we did not open the holdings files, so "
           f"we cannot prove it was the basket rather than the envelope. What we can say is "
           f"that the gap points the other way in {R['best_year']} "
           f"({R['best_diff']:+.2f} pp), which is not how a structural wrapper cost behaves."),
        md(f"## 6. Now the tax — and the honest bit\n\n"
           f"Pre-tax, the two are a tie. So does tax break the tie? We modelled both regimes "
           f"on the realised annual returns: the K-1 leg marked to market every year at the "
           f"60/40 rate, the 1099 leg paying ordinary tax on its distributions and deferring "
           f"the rest until sale.\n\n"
           f"**Every input to that model is an assumption, not a measurement** — the marginal "
           f"rates, and above all how much the 1099 fund distributes each year. So we swept "
           f"both. The result across the whole grid runs from "
           f"**{R['gap_min']:+.2f} to {R['gap_max']:+.2f} percentage points a year** — and "
           f"**the sign flips inside it**.\n\n"
           f"If the fund distributes little, deferral wins and the 1099 wrapper is ahead by a "
           f"few tenths of a point. If it distributes twice the collateral interest — which "
           f"a fund of this type can do when its subsidiary realises gains, and which we did "
           f"**not** measure for PDBC — the K-1's 60/40 rate wins by about the "
           f"same margin. The after-tax winner is chosen by *your assumption about the "
           f"distribution*, not by the tape. We report the range and refuse to pick."),
        md(f"## 7. The decision you are actually making\n\n"
           f"Over the same window, choosing between commodity vehicles was worth "
           f"**{R['dispersion_pp']:.1f} percentage points a year**: USCI compounded at "
           f"{R['usci_cagr']:+.2f}%, USO at **{R['uso_cagr']:+.2f}%**. Choosing between the "
           f"two *wrappers* was worth **{abs(R['diff_ann']):.2f} points**.\n\n"
           f"The paperwork decision is roughly **90 times smaller** than the decision it is "
           f"usually bundled with — and investors spend far more time on it."),
        md("## 8. Live check — the machinery isn't asleep (synthetic, not the real tape)\n\n"
           "Before trusting a null, check that the same code finds a cost when a cost is "
           "genuinely there. The cells below build **synthetic** wrapper pairs — two funds on "
           "a shared commodity factor — one with a real drag planted in it, one with none.\n\n"
           "*This is simulated data. None of the numbers below are DBC or PDBC.*"),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from wrapper_tax import data, strategy as st\n"
            "\n"
            "planted, truth = data.synthetic_daily(signal_strength=1.0, seed=915)\n"
            "det = st.synthetic_detect(planted, n_boot=400)\n"
            "print('SYNTHETIC — planted wrapper drag  : %.2f%% a year' % (truth['planted_drag_ann']*100))\n"
            "print('SYNTHETIC — the race recovered    : %.2f%% a year  (t = %+.2f)'\n"
            "      % (det['estimated_drag_ann']*100, det['t_diff']))\n"
            "\n"
            "null, _ = data.synthetic_daily(signal_strength=0.0, seed=915)\n"
            "det0 = st.synthetic_detect(null, n_boot=400)\n"
            "print('SYNTHETIC — identical wrappers    : %+.2f%% a year  (t = %+.2f — well inside'\n"
            "      % (det0['diff_ann']*100, det0['t_diff']))\n"
            "print('                                     +/-2, i.e. indistinguishable from zero)')"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The 1099 wrapper costs **{abs(R['diff_ann']):.2f}%/yr** "
           f"against its K-1 twin, with *t* = {R['t_diff']:+.2f} and a range of "
           f"[{R['ci_lo']:+.2f}%, {R['ci_hi']:+.2f}%]; null in both halves of the sample, "
           f"{R['wins']}/{R['n_years']} annual wins, and unmoved by a punitive cost sweep. The "
           f"test had the power to catch anything above ~1%/yr, so this is an informative "
           f"null. The after-tax model does not rescue a signal either: its gap spans "
           f"[{R['gap_min']:+.2f}, {R['gap_max']:+.2f}] pp/yr and **changes sign** with the "
           f"distribution assumption.\n"
           f"- **Tradability — Mirage.** There is no edge here to bank — the difference is "
           f"tens of basis points against {R['te_monthly']:.2f}-{R['te_ann']:.2f}% of tracking "
           f"error and {R['dispersion_pp']:.1f} pp/yr of index-choice dispersion. What *is* "
           f"worth knowing is the non-finding: the No-K-1 convenience appears to be **free**. "
           f"That is a good reason to prefer it, and it is not alpha."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 915 — K-1 vs 1099 — the teardown\n\n"
           "PDBC (Form 1099, 1940-Act fund with a Cayman subsidiary) against DBC (Schedule "
           "K-1 commodity pool, Section 1256 futures), same manager, same index family. "
           "Three layers in increasing order of assumption: **(1)** the pre-tax race — pure "
           "tape; **(2)** the fee-gap reconciliation — one proxy; **(3)** the after-tax "
           "regime model — all assumption, fully swept.\n\n"
           "Every real number below is frozen from `docs/results.md` (Fingerprint `%s`, "
           "as-of 2026-06-30). The live cells run synthetic data only and are labelled.\n\n"
           "> 💡 **In plain words:** two funds, one commodity, different tax paperwork. Does "
           "the easy paperwork cost you anything?"
           % R["fp"]),
        code("R = %r" % (R,)),
        md("## Layer 1 — the pre-tax race\n\n"
           "Both legs buy-and-hold, both measured **excess of cash** (BIL), so the collateral "
           "T-bill yield — ~5%/yr in 2023-2026 and a large fraction of a commodity fund's "
           "return in those years — is netted out of *both* sides. **One execution lag**: the "
           "wrapper is bought at the first common close and the first return counted is the "
           "next session's. **No shorting**, hence no borrow anywhere in this study."),
        code(
            "print(f\"DBC  (K-1)  : exSharpe {R['k1_sharpe']:+.3f}  CAGR {R['k1_cagr']:+.2f}%  \"\n"
            "      f\"vol {R['k1_vol']:.1f}%  MaxDD {R['k1_dd']:.1f}%  HAC t {R['k1_t']:+.2f}\")\n"
            "print(f\"PDBC (1099) : exSharpe {R['ric_sharpe']:+.3f}  CAGR {R['ric_cagr']:+.2f}%  \"\n"
            "      f\"vol {R['ric_vol']:.1f}%  MaxDD {R['ric_dd']:.1f}%  HAC t {R['ric_t']:+.2f}\")\n"
            "print()\n"
            "print(f\"difference (1099 - K-1) : {R['diff_ann']:+.4f}%/yr   HAC t = {R['t_diff']:+.2f}\")\n"
            "print(f\"  paired block-bootstrap 95% CI : [{R['ci_lo']:+.4f}%, {R['ci_hi']:+.4f}%]  \"\n"
            "      f\"P(<0) = {R['ci_pneg']:.3f}\")\n"
            "print(f\"excess-of-cash Sharpe advantage : {R['sharpe_adv']:+.4f}  \"\n"
            "      f\"CI [{R['adv_ci_lo']:+.4f}, {R['adv_ci_hi']:+.4f}]\")\n"
            "print(f\"tracking: corr {R['corr']:.4f}  beta {R['beta']:.4f}  R2 {R['r2']:.4f}\")"
        ),
        md("## Layer 2 — what the fee gap predicted vs what the tape delivered\n\n"
           "**PROXY / ASSUMPTION carrying hindsight:** the two **current** prospectus expense "
           "ratios, applied to the whole 2014-2026 window. They are *not* measured here and "
           "are never subtracted from a return series — both tapes are already net of the "
           "real, time-varying fees. They are quoted only to check the sign of the prediction, "
           "so the hindsight never touches a backtest."),
        code(
            "print(f\"expense ratio (PROXY): DBC {R['er_k1']:.2f}%  PDBC {R['er_ric']:.2f}%\")\n"
            "print(f\"  -> the fee gap predicts the 1099 wrapper WINS by {R['fee_gap_bp']} bp/yr\")\n"
            "print(f\"  -> the tape says it LOST by {abs(R['diff_ann'])*100:.0f} bp/yr\")\n"
            "print(f\"  -> a {R['fee_gap_bp'] + abs(R['diff_ann'])*100:.0f} bp/yr shortfall, \"\n"
            "      f\"inside one standard error ({R['se_hac']:.2f}%/yr)\")\n"
            "print('  -> i.e. the tape cannot tell the fee prediction apart from zero either;')\n"
            "print('     active/basket drift is a candidate, but it is NOT measured here.')"
        ),
        md("## The nuisance term — daily TE is mostly microstructure\n\n"
           "The daily difference carries a lag-1 autocorrelation of −0.43: the two closes "
           "disagree and re-converge. Naive √252 scaling therefore inflates the tracking "
           "error threefold relative to the divergence a holder lives through.\n\n"
           "> 💡 **In plain words:** the two prices bounce against each other day to day, then "
           "snap back. Look monthly and the wobble mostly cancels."),
        code(
            "print(f\"daily-scaled TE {R['te_ann']:.2f}%/yr   vs   monthly TE {R['te_monthly']:.2f}%/yr \"\n"
            "      f\"({R['n_months']} months)\")\n"
            "print(f\"lag-1 autocorrelation of the daily difference: {R['acf1']:+.3f}\")\n"
            "print(f\"monthly difference {R['diff_monthly']:+.4f}%/yr (HAC t = {R['t_monthly']:+.2f}) \"\n"
            "      f\"-> same null, cleaner measurement\")"
        ),
        md("## Power — is this null informative or merely blind?\n\n"
           "Three standard errors for the same quantity. The i.i.d. one is wrong (negative "
           "autocorrelation); the HAC and bootstrap ones agree."),
        code(
            "for tag, se, mde in [('i.i.d.', R['se_iid'], R['mde_iid']),\n"
            "                     ('HAC (Newey-West)', R['se_hac'], R['mde_hac']),\n"
            "                     ('block bootstrap', R['se_boot'], R['mde_boot'])]:\n"
            "    print(f\"{tag:18s}: SE {se:.2f}%/yr   minimum detectable at |t|=2: {mde:.2f}%/yr\")\n"
            "print()\n"
            "print('=> the claim is \"no wrapper cost above ~0.9-1.2%/yr\", not \"no wrapper cost\".')"
        ),
        md("## Era cut — and a converging pair\n\n"
           "Null in both halves. Note the tracking error *falling* as PDBC grew out of its "
           "illiquid first years: the wrappers converged rather than diverged. (Both TEs are "
           "the naive √252-scaled daily figure the cell above shows to be inflated — the "
           "*fall* is the point, not the level.)"),
        code(
            "print(f\"2014-11..2020-12 (n={R['era_e_n']}): diff {R['era_e_diff']:+.3f}%/yr \"\n"
            "      f\"(t={R['era_e_t']:+.2f})  TE {R['era_e_te']:.2f}%\")\n"
            "print(f\"2021-01..2026-06 (n={R['era_l_n']}): diff {R['era_l_diff']:+.3f}%/yr \"\n"
            "      f\"(t={R['era_l_t']:+.2f})  TE {R['era_l_te']:.2f}%   <- TE more than halved\")"
        ),
        md("## Calendar years, sign test, and the cost sweep\n\n"
           "Costs are amortised over the holding period — buy-and-hold means one round trip, "
           "so 2 x spread / 11.6 years. The sweep is deliberately asymmetric: the extra spread "
           "is charged to the 1099 leg only."),
        code(
            "print(f\"1099 wins {R['wins']}/{R['n_years']} complete years \"\n"
            "      f\"({R['win_rate']}%, Wilson 95% CI [{R['win_lo']}%, {R['win_hi']}%]) \"\n"
            "      f\"-> a coin flip is inside the interval\")\n"
            "print(f\"worst year {R['worst_year']}: {R['worst_diff']:+.2f} pp   \"\n"
            "      f\"best year {R['best_year']}: {R['best_diff']:+.2f} pp   \"\n"
            "      f\"(sign reverses -> not a structural wrapper cost)\")\n"
            "print('   PDBC is actively managed, so basket drift is the candidate --')\n"
            "print('   INTERPRETATION, not a measurement: no holdings file was parsed.')\n"
            "print()\n"
            "print(f\"+50 bp one-way charged to the 1099 leg -> {R['cost50_drag_bp']:.2f} bp/yr drag, \"\n"
            "      f\"difference {R['cost50_diff']:+.4f}%/yr (t={R['cost50_t']:+.2f})\")\n"
            "print('=> at an 11.6-year horizon, trading friction is not the story.')"
        ),
        md("## Layer 3 — the after-tax model. Every input is an ASSUMPTION\n\n"
           "**K-1 leg:** §1256 contracts, marked to market each 31 December, taxed at the "
           "60/40 blend; collateral interest (proxied by BIL's total return on the opening "
           "balance) taxed as ordinary income; net 1256 losses carried forward; tax paid out "
           "of the account; annual MTM steps the basis up so **nothing is owed at "
           "liquidation**.\n\n"
           "**1099 leg:** ordinary distributions = `payout_share` × the same interest proxy, "
           "taxed annually and reinvested (basis rises, the tax withdrawal removes basis pro "
           "rata); the remainder **deferred** and taxed once at the long-term rate on sale.\n\n"
           "**Not modelled:** state tax, the accountant time a K-1 costs, filing extensions, "
           "UBTI inside a retirement account, and PDBC's IRA eligibility. No filing was "
           "parsed — this is a transparent model, not a measurement.\n\n"
           "> 💡 **In plain words:** the K-1 fund gets a better tax rate but pays every year; "
           "the 1099 fund gets a worse rate on part of the return but delays the rest. Which "
           "wins is arithmetic — and it depends on how much the 1099 fund hands out."),
        code(
            "print('after-tax gap (1099 - K-1), pp/yr, across the assumption grid')\n"
            "print(f\"  top bracket 40.8/23.8: payout 0.0 -> {R['gap_top_p0']:+.3f}   \"\n"
            "      f\"payout 1.0 -> {R['gap_top_p1']:+.3f}   payout 2.0 -> {R['gap_top_p2']:+.3f}\")\n"
            "print(f\"  32%+NIIT   35.8/18.8: payout 1.0 -> {R['gap_mid_p1']:+.3f}\")\n"
            "print(f\"  24%/15%             : payout 1.0 -> {R['gap_low_p1']:+.3f}\")\n"
            "print()\n"
            "print(f\"full-grid range: [{R['gap_min']:+.3f}, {R['gap_max']:+.3f}] pp/yr -- THE SIGN FLIPS.\")\n"
            "print('=> the after-tax winner is chosen by the payout assumption, not by the tape.')"
        ),
        md("## The regime alone — both tax rules on the SAME return stream\n\n"
           "Applying both regimes to DBC's own annual returns removes the tracking difference, "
           "leaving only the value of deferral against the 60/40 rate."),
        code(
            "print(f\"22/15      : regime gap {R['regime_gap_low']:+.3f} pp/yr to the 1099 wrapper\")\n"
            "print(f\"35.8/18.8  : regime gap {R['regime_gap_mid']:+.3f} pp/yr\")\n"
            "print(f\"40.8/23.8  : regime gap {R['regime_gap_top']:+.3f} pp/yr\")\n"
            "print()\n"
            "print(f\"pre-tax CAGR (complete years): K-1 {R['pretax_cagr_k1']:+.3f}%  \"\n"
            "      f\"1099 {R['pretax_cagr_ric']:+.3f}%\")\n"
            "print(f\"after tax at 35.8/18.8, payout 1.0: K-1 {R['aftertax_k1_mid']:+.3f}%  \"\n"
            "      f\"1099 {R['aftertax_ric_mid']:+.3f}%\")\n"
            "print('=> deferral is worth 24-44 bp/yr; the tracking difference takes back ~27 bp/yr.')"
        ),
        md("## Context — the size of the decision next door\n\n"
           "Survivorship note: no dead commodity ETP was dropped, and both wrappers are the "
           "live vehicles an investor actually faced from PDBC's inception. The *pair* was "
           "chosen because it is the famous K-1/No-K-1 twin — selection on salience, not on "
           "performance."),
        code(
            "print(f\"USCI (K-1)  CAGR {R['usci_cagr']:+.2f}%\")\n"
            "print(f\"DBC  (K-1)  CAGR {R['k1_cagr']:+.2f}%\")\n"
            "print(f\"PDBC (1099) CAGR {R['ric_cagr']:+.2f}%\")\n"
            "print(f\"BNO  (K-1)  CAGR {R['bno_cagr']:+.2f}%\")\n"
            "print(f\"USO  (K-1)  CAGR {R['uso_cagr']:+.2f}%\")\n"
            "print()\n"
            "print(f\"index/vehicle dispersion {R['dispersion_pp']:.2f} pp/yr  vs  \"\n"
            "      f\"wrapper difference {abs(R['diff_ann']):.2f} pp/yr  (~90x)\")"
        ),
        md("## Synthetic control — SYNTHETIC DATA, never the real tape\n\n"
           "Two wrappers on a shared commodity factor plus independent tracking noise. Plant a "
           "known drag on one: the race must recover it at |*t*| ≥ 2. Switch the drag off: the "
           "race must stay quiet across a panel of independent seeds. This proves the null "
           "above is a property of the wrappers, not of a sleepy estimator."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from wrapper_tax import data, strategy as st\n"
            "\n"
            "planted, truth = data.synthetic_daily(signal_strength=1.0, seed=915)\n"
            "det = st.synthetic_detect(planted, n_boot=600)\n"
            "print('SYNTHETIC planted drag %.2f%%/yr -> recovered %.2f%%/yr  (HAC t %+.2f, CI [%+.2f%%, %+.2f%%])'\n"
            "      % (truth['planted_drag_ann']*100, det['estimated_drag_ann']*100,\n"
            "         det['t_diff'], det['ci_low']*100, det['ci_high']*100))\n"
            "\n"
            "panel = data.synthetic_panel(n_pairs=8, signal_strength=0.0, seed=915)\n"
            "diffs = np.array([st.race(p, 'ric', 'k1', 'cash')['diff_ann'] for p, _ in panel])\n"
            "ts = np.array([st.race(p, 'ric', 'k1', 'cash')['t_diff'] for p, _ in panel])\n"
            "print('SYNTHETIC null x8: mean drift %+.3f%%/yr (sd %.3f%%), |t|>=2 in %d/8'\n"
            "      % (diffs.mean()*100, diffs.std(ddof=1)*100, int((np.abs(ts) >= 2).sum())))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The pre-tax difference is **{R['diff_ann']:+.4f}%/yr** at "
           f"**HAC *t* = {R['t_diff']:+.2f}**, paired bootstrap CI "
           f"[{R['ci_lo']:+.2f}%, {R['ci_hi']:+.2f}%]; the excess-of-cash Sharpe advantage is "
           f"{R['sharpe_adv']:+.4f}, CI [{R['adv_ci_lo']:+.3f}, {R['adv_ci_hi']:+.3f}]. Null "
           f"in both eras (|*t*| ≤ {abs(R['era_l_t']):.2f}), {R['wins']}/{R['n_years']} annual "
           f"wins with a coin flip inside the Wilson interval, and a 50 bp punitive spread "
           f"moves the answer {abs(R['cost50_diff']-R['diff_ann'])*100:.0f} bp. The sample had "
           f"the power to detect anything above ~{R['mde_boot']:.1f}-{R['mde_hac']:.1f}%/yr, so "
           f"this is an informative null, not a blind one. The after-tax layer cannot promote "
           f"it: the modelled gap spans [{R['gap_min']:+.2f}, {R['gap_max']:+.2f}] pp/yr and "
           f"**changes sign** inside the assumption grid. The synthetic control recovers a "
           f"planted {R['syn_planted']:.1f}%/yr drag ({R['syn_recovered']:.2f}%/yr, "
           f"*t* = {R['syn_t']:+.2f}) and stays quiet on the null "
           f"({R['syn_null_fire']}/8), so the estimator works.\n"
           f"- **Tradability — Mirage.** Nothing to bank. The wrapper difference is tens of "
           f"basis points against {R['te_monthly']:.2f}-{R['te_ann']:.2f}%/yr of tracking error "
           f"and {R['dispersion_pp']:.1f} pp/yr of index-choice dispersion, and the after-tax "
           f"gap is smaller than the assumptions that generate it. The usable conclusion is the "
           f"non-finding — the No-K-1 convenience is, on this evidence, **free** — and free "
           f"convenience is not alpha."),
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
