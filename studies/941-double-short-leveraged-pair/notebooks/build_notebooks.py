"""Generate the two narrative notebooks for Study 941 (Short Both Legs).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below (a mirror of docs/results.md); the only live cells run the fast
synthetic control, and they are never presented under a real-tape banner.
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


# Frozen real-tape headline — mirror of docs/results.md. TQQQ + SQQQ shorted in equal
# dollars against BIL cash, QQQ as the neutrality benchmark, 2010-02-11 -> 2026-06-30.
R = dict(
    start="2010-02-11", end="2026-06-30", n_days=4120, fp="8f75619ff8e2",
    tqqq_mean=55.48, sqqq_mean=-57.25, qqq_mean=20.30, bil_mean=1.32,
    tqqq_vol=61.2, sqqq_vol=61.3, qqq_vol=20.7,
    implied_load=2.20,
    sf_long=2.79, sf_short=1.61, sf_share_long=63,
    decay_3x=12.9, decay_m3x=25.7,
    mean=2.06, sharpe=1.64, t=8.82, vol=1.26, dd=-1.37,
    worst_day=-1.14, worst_month=-0.23, skew=1.64, kurt=47.3, turnover=6.8,
    beta=0.0032, alpha=2.00, t_alpha=8.49, r2=0.003, qqq_sharpe=0.92,
    ci_sh_lo=1.36, ci_sh_hi=1.90, ci_m_lo=1.67, ci_m_hi=2.64, ci_frac_neg=0.0,
    wk_mean=2.69, wk_sharpe=0.55, wk_t=2.18, wk_dd=-11.4, wk_wm=-7.1, wk_beta=0.030,
    mo_mean=5.26, mo_sharpe=0.44, mo_t=2.39, mo_dd=-15.5, mo_wm=-12.1, mo_beta=0.190,
    mo_alpha=1.64, mo_t_alpha=0.68,
    nv_mean=-61.15, nv_dd=-100.0, nv_gross=51.7, nv_date="2013-08-01",
    be_daily=2.06, be_weekly=2.70, be_monthly=5.27,
    b1_mean=1.06, b1_t=4.54, b2_mean=0.06, b2_t=0.26, b3_mean=-0.94, b3_t=-4.02,
    b5_mean=-2.94, b5_t=-12.57, b8_mean=-5.94, b15_mean=-12.94,
    nr_mean=0.75, nr_sharpe=0.60, nr_t=3.02, nr_be=0.75,
    nr_early=1.45, nr_early_t=7.86, nr_late=0.10, nr_late_t=0.22,
    fr_early=1.50, fr_early_t=7.98, fr_late=2.59, fr_late_t=6.39,
    cost0=2.20, cost0_t=9.31, cost5=1.86, cost10=1.52,
    yr2020=6.79, yr2022=3.63, yr2016=0.92, yr_all_positive=17, yrs_below_be=11,
    med=1.74, trim1=1.82, frac_pos=55.9,
    drop5=1.77, drop5_t=13.48, drop20=1.40, drop20_t=9.60, drop50=0.95, drop50_t=5.53,
    ex2020=1.77, ex2020_t=15.97,
    syn_planted=2.45, syn_recovered=2.45, syn_t=15.06,
    syn_null=0.00, syn_null_t=0.05, syn_decay_l=14.5, syn_decay_s=29.0,
)


HEADER = f"""# Study 941 — Short Both Legs ⚖️

**Short the 3x long *and* the 3x short of the same index — is the decay harvest free money?**

TQQQ (3x long Nasdaq-100) and SQQQ (−3x inverse) both reset their leverage every day, and
both therefore bleed the textbook compounding decay — roughly **{R['decay_3x']:.0f}%/yr**
for the 3x and **{R['decay_m3x']:.0f}%/yr** for the −3x at the index's ~21% volatility.
Short both in equal dollars and the market exposure cancels. The forums call it free money.

We test it on **TQQQ + SQQQ vs BIL** (cash), with **QQQ** as the neutrality check, daily
total-return closes {R['start']} → {R['end']} ({R['n_days']:,} days), everything
**excess-of-cash**, one execution lag, 2 bps one-way cost, and the borrow fee swept from
0 to 15%/yr because it is an *assumption*, not tape.

**Every harvest number below is before any stock-borrow fee** — no free borrow-fee tape
exists, so the rate is an assumption and the *breakeven* borrow is what the verdict rests on.

*Numbers below are the frozen headline (`docs/results.md`, Fingerprint `{R['fp']}`); the
live cells run the fast offline synthetic control. As-of 2026-06-30.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The chart that sells the trade\n\n"
           "Hold TQQQ and SQQQ together and your money melts — that much is true and easy "
           "to chart. The leap the forums make is: *therefore shorting both must print "
           "money*. Let's see what the tape actually paid."),
        code(
            "R = dict(mean=%r, sharpe=%r, t=%r, vol=%r, dd=%r, beta=%r,\n"
            "         implied_load=%r, decay_3x=%r, decay_m3x=%r)\n"
            "print('short both legs, daily reset, 2 bps, no borrow:')\n"
            "print('  excess-of-cash %%+.2f%%%%/yr   Sharpe %%+.2f   HAC t %%+.2f'\n"
            "      %% (R['mean'], R['sharpe'], R['t']))\n"
            "print('  volatility %%.2f%%%%   worst drawdown %%.2f%%%%   beta to QQQ %%+.4f'\n"
            "      %% (R['vol'], R['dd'], R['beta']))\n"
            "print()\n"
            "print('decay supposedly on offer: %%.0f%%%% + %%.0f%%%% per year'\n"
            "      %% (R['decay_3x'], R['decay_m3x']))\n"
            "print('what actually turned up  : %%+.2f%%%% per year' %% R['mean'])"
            % (R["mean"], R["sharpe"], R["t"], R["vol"], R["dd"], R["beta"],
               R["implied_load"], R["decay_3x"], R["decay_m3x"])
        ),
        md(f"## 2. Where did the other 36 points go?\n\n"
           f"They were never collectable. The decay is a fact about the *compounded path* "
           f"of a levered fund, not about its *average daily return*. A short position that "
           f"is reset to the same dollar size every evening earns the average daily return "
           f"of what it is short — and on that measure TQQQ is exactly 3x the index and "
           f"SQQQ exactly −3x, so the index cancels and all that is left is **what the two "
           f"funds cost you**.\n\n"
           f"Add up the two legs on the real tape and the leftover is "
           f"**+{R['implied_load']:.2f}%/yr**. Two caveats we owe you, because they are the "
           f"easiest place in this study to fool yourself:\n\n"
           f"1. That +{R['implied_load']:.2f}% and the {R['mean']:+.2f}% the backtest earned "
           f"are **the same arithmetic**, not two independent measurements agreeing. On a "
           f"daily reset the book's excess return *is* `cash − 0.5×(TQQQ + SQQQ)`, to fifteen "
           f"decimal places. The backtest earns its keep on everything else — other "
           f"schedules, borrow, the rebate, costs, compounding, ruin — not here.\n"
           f"2. It is a **residual**, and calling it a fee is a reading. Expense ratios, the "
           f"financing spread, the funds' own end-of-day trading, tracking error and the "
           f"cash-proxy basis all land in the same bucket, and this study separates none of "
           f"them. What *is* established is the negative: it is not the compounding decay.\n\n"
           f"> 🔬 **For the quants.** Arithmetic vs geometric: "
           f"`E[r_L] = L*E[r_index] - (L-1)*r_cash - fees` has no σ² term; the "
           f"`0.5*L*(L-1)*σ²` decay only appears in `E[log r_L]`. A constant-dollar short "
           f"is an arithmetic-mean bet, which is why it never touches the decay."),
        md(f"## 3. So it *is* a real edge — a small, dull one\n\n"
           f"+{R['mean']:.2f}%/yr over cash, at {R['vol']:.2f}% volatility, with a worst "
           f"drawdown of {R['dd']:.2f}% and a beta to the Nasdaq of {R['beta']:+.4f}. It was "
           f"positive in **all {R['yr_all_positive']} calendar years** in the sample (2026 "
           f"is a half-year stub), best "
           f"in the stressed ones (2020: +{R['yr2020']:.2f}%, 2022: +{R['yr2022']:.2f}%) "
           f"because that is when the funds' own trading costs widen. Statistically it is "
           f"about as solid as this desk ever sees: HAC *t* = {R['t']:+.2f} — though on a "
           f"series this thin that *t* is really saying **a cost accrual is stable**, not "
           f"that an anomaly was discovered.\n\n"
           f"And it is not a tail bet, which for a series with +{R['kurt']:.0f} excess "
           f"kurtosis is worth checking: {R['frac_pos']:.1f}% of days are positive, the "
           f"median day annualises to +{R['med']:.2f}%, and throwing away the **50 best days** "
           f"out of {R['n_days']:,} still leaves +{R['drop50']:.2f}%/yr. Drop the whole of "
           f"2020 and it is +{R['ex2020']:.2f}%/yr. (Those are *magnitude* checks. We do not "
           f"quote a *t* for them: deleting the best observations strips the right tail, so "
           f"the *t* rises mechanically even as the mean falls — it would look like better "
           f"evidence when it is less.)"),
        md(f"## 4. And then the borrow fee arrives\n\n"
           f"You have to *borrow* both funds to short them, and the lender charges a fee. "
           f"Every 1%/yr of blended fee takes 1 pp straight off the top, so the trade dies "
           f"at a blended borrow of **{R['be_daily']:.2f}%/yr**. TQQQ is easy to borrow; "
           f"SQQQ is the crowded side. The edge and the fee are the same size — which is "
           f"not a coincidence: a lending fee exists precisely to price the reason someone "
           f"wants to short a thing. And at exactly that rate, "
           f"**{R['yrs_below_be']} of the {R['yr_all_positive']} calendar years lose "
           f"money** — 'positive every year' is not 'profitable every year'.\n\n"
           f"Worse for an ordinary account. A short position generates cash, and a prime "
           f"broker pays you the cash rate on it; a retail broker pays you nothing. That "
           f"interest is not a *source* of the edge — the two funds' own financing carry "
           f"averages to +1x cash on the liability side and the proceeds interest is what "
           f"offsets it — but **losing** it costs you the cash rate itself, "
           f"{R['bil_mean']:.2f}%/yr on this sample. Without it the trade earned "
           f"+{R['nr_mean']:.2f}%/yr over the whole window and "
           f"**+{R['nr_late']:.2f}%/yr (t = {R['nr_late_t']:+.2f}) since 2018** — nothing — "
           f"because the higher rates went, the more the missing interest cost."),
        md(f"## 5. Can't you just rebalance less often and earn more?\n\n"
           f"That is the classic response, and the tape does show a bigger headline: "
           f"monthly resets returned +{R['mo_mean']:.2f}%/yr instead of "
           f"+{R['mean']:.2f}%. But almost all of the extra is an accidental *long Nasdaq* "
           f"bet (beta {R['mo_beta']:+.3f}); its market-neutral alpha is only "
           f"+{R['mo_alpha']:.2f}%/yr with *t* = {R['mo_t_alpha']:+.2f}. Meanwhile the "
           f"Sharpe collapses from {R['sharpe']:+.2f} to {R['mo_sharpe']:+.2f} and the worst "
           f"month goes from {R['worst_month']:.2f}% to {R['mo_wm']:.1f}%.\n\n"
           f"Never rebalancing is not a strategy at all. The 3x leg keeps rising, your short "
           f"of it keeps growing, and by **{R['nv_date']}** the book is **wiped out** — "
           f"gross short {R['nv_gross']:.1f}x equity at the end. Short-both is a short-gamma "
           f"position: you must keep feeding it, and the feeding is the strategy."),
        md("## 6. Live check — the machinery is unbiased (offline synthetic)\n\n"
           "We build two toy worlds with **identical, enormous** decay in their levered "
           "legs. In one, the funds charge a fee; in the other they are free. If the trade "
           "harvested *decay*, it would pay in both. Watch it pay in one."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from short_pair import data, strategy as st\n"
            "for tag, ss in [('funds charge a fee', 1.0), ('funds are free    ', 0.0)]:\n"
            "    px, truth = data.synthetic_daily(signal_strength=ss, seed=941)\n"
            "    d = st.synthetic_detect(px, cost_bps=0.0)\n"
            "    print('%s | decay in the legs %.1f%%/%.1f%% per yr -> harvest %+.2f%%/yr (t=%+.2f)'\n"
            "          % (tag, truth['decay_long_ann']*100, truth['decay_short_ann']*100,\n"
            "             d['ann_mean']*100, d['t_hac']))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Real.** A genuine, market-neutral **+{R['mean']:.2f}%/yr** over "
           f"cash **before borrow** (HAC *t* = {R['t']:+.2f}, beta {R['beta']:+.4f}, positive "
           f"every year, +{R['drop50']:.2f}% without its 50 best days). It is simply not the "
           f"effect the folklore names: it is what the two funds cost their holders, and the "
           f"synthetic control proves the point by paying nothing when identical decay "
           f"carries no fee.\n"
           f"- **Tradability — Fragile.** Breakeven blended borrow is "
           f"**{R['be_daily']:.2f}%/yr** — the same size as the edge, and at that rate "
           f"{R['yrs_below_be']} of {R['yr_all_positive']} years are losers. Worse, the "
           f"crowded SQQQ leg breaks even at only {R['sf_short']:.2f}%/yr on its own; the "
           f"easy-to-borrow TQQQ leg is paying for it. Without a rebate "
           f"it is +{R['nr_late']:.2f}%/yr since 2018, which for a retail account makes it a "
           f"Mirage outright. Rebalancing less to earn more just buys residual beta and a "
           f"{R['mo_wm']:.0f}% month, and not rebalancing at all destroys the account. A "
           f"prime-brokerage carry trade at best, free money never."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 941 — Short Both Legs — the teardown\n\n"
           "The arithmetic-vs-geometric decomposition, the excess-of-cash harvest with its "
           "Newey-West *t*, block-bootstrap CIs, the residual-beta regression, the borrow "
           "and rebate sweeps, the reset-schedule race, and the live synthetic control that "
           "separates cost load from decay. Every real number is frozen from "
           "`docs/results.md` (Fingerprint `%s`), and every harvest number is **before any "
           "stock-borrow fee**." % R["fp"]),
        code("R = %r" % (R,)),
        md("## 1. The two legs, arithmetically\n\n"
           "> 💡 **In plain words:** add up what the two funds returned on an average day, "
           "and whatever does not cancel is what shorting them both can pay you.\n\n"
           "Annualised arithmetic means (total return, `auto_adjust=True`). Daily-return "
           "betas to QQQ are +2.96 / −2.96, so the index cancels almost exactly; what is "
           "left is each fund's financing carry and cost load.\n\n"
           "⚠️ **This quantity and the section-2 harvest are the same algebra.** On a daily "
           "reset with a full rebate, zero borrow and zero cost, `e_nav[t] = r_cash[t] − "
           "0.5·(r_long[t] + r_short[t])` **identically** (agreement to 5e-16). Treat their "
           "match as a unit test of the simulator, never as corroboration. And the residual "
           "itself is unattributed: expenses, financing spread, the funds' own end-of-day "
           "trading, tracking and the BIL-versus-funding basis all sit inside it.\n\n"
           "Split by leg, the same residual is lopsided — and the lopsidedness runs against "
           "the trade. Each leg pays borrow on its own half of the notional, so each has its "
           "own breakeven equal to its own shortfall, and the **crowded, expensive-to-borrow "
           "SQQQ leg is the thin one (1.61%/yr)**. The blended 2.06% only exists because the "
           "general-collateral TQQQ leg (2.79%/yr, 63% of the harvest) subsidises it."),
        code(
            "print(f\"TQQQ  mean {R['tqqq_mean']:+.2f}%/yr  vol {R['tqqq_vol']:.1f}%\")\n"
            "print(f\"SQQQ  mean {R['sqqq_mean']:+.2f}%/yr  vol {R['sqqq_vol']:.1f}%\")\n"
            "print(f\"QQQ   mean {R['qqq_mean']:+.2f}%/yr  vol {R['qqq_vol']:.1f}%\")\n"
            "print(f\"BIL   mean {R['bil_mean']:+.2f}%/yr\")\n"
            "print()\n"
            "print(f\"cash - 0.5*(TQQQ + SQQQ) = implied average fund shortfall \"\n"
            "      f\"{R['implied_load']:+.2f}%/yr  <- IDENTITY with the harvest in section 2\")\n"
            "print(f\"geometric decay on offer  : {R['decay_3x']:.1f}% (3x) + {R['decay_m3x']:.1f}% (-3x) per yr\")\n"
            "print('the decay lives in E[log r], the short book earns E[r] -> uncollectable')\n"
            "print()\n"
            "print('split by leg (shortfall vs its own costless replication):')\n"
            "print(f\"  TQQQ {R['sf_long']:+.2f}%/yr ({R['sf_share_long']:.0f}% of the harvest)  \"\n"
            "      f\"SQQQ {R['sf_short']:+.2f}%/yr -> the crowded leg is the THIN one\")"
        ),
        md("## 2. Headline — daily reset, excess-of-cash, zero borrow, 2 bps"),
        code(
            "print(f\"mean {R['mean']:+.2f}%/yr  Sharpe {R['sharpe']:+.2f}  HAC t {R['t']:+.2f}  \"\n"
            "      f\"vol {R['vol']:.2f}%  maxDD {R['dd']:.2f}%\")\n"
            "print(f\"worst day {R['worst_day']:.2f}%  worst month {R['worst_month']:.2f}%  \"\n"
            "      f\"skew {R['skew']:+.2f}  excess kurtosis {R['kurt']:+.1f}  turnover {R['turnover']:.1f}x/yr\")\n"
            "print(f\"residual beta to QQQ {R['beta']:+.4f} (R2 {R['r2']:.3f}), \"\n"
            "      f\"alpha {R['alpha']:+.2f}%/yr (t={R['t_alpha']:+.2f})\")\n"
            "print(f\"QQQ's own excess Sharpe over the window: {R['qqq_sharpe']:+.2f} \"\n"
            "      f\"-> the book is neutral to it, not a disguised long\")\n"
            "print()\n"
            "print(f\"bootstrap (2000 draws, 21d blocks): Sharpe 95% CI \"\n"
            "      f\"[{R['ci_sh_lo']:+.2f}, {R['ci_sh_hi']:+.2f}], \"\n"
            "      f\"mean 95% CI [{R['ci_m_lo']:+.2f}%, {R['ci_m_hi']:+.2f}%], \"\n"
            "      f\"frac<0 {R['ci_frac_neg']:.1f}%\")"
        ),
        md("## 2b. Accrual or a handful of days?\n\n"
           "> 💡 **In plain words:** with +47 excess kurtosis, we have to check that the "
           "profit is not just the March-2020 week wearing a disguise.\n\n"
           "Drop-the-best-days, a trimmed mean, and a straight 2020 exclusion. Note the last "
           "line: strip the 50 best days and what survives is *below the breakeven borrow*.\n\n"
           "⚠️ **Read the mean column, not the *t* column.** Deleting the best observations "
           "and then re-running a *t*-test is not valid inference — it strips the right tail, "
           "so the *t* rises **mechanically** (+8.82 → +13.48 after five days) while the mean "
           "falls. Those *t*'s are printed as smoothness diagnostics; nothing in the verdict "
           "rests on them."),
        code(
            "print(f\"mean {R['mean']:+.2f}%/yr   median day {R['med']:+.2f}%/yr   \"\n"
            "      f\"days > 0: {R['frac_pos']:.1f}%   1% trimmed mean {R['trim1']:+.2f}%/yr\")\n"
            "print(f\"drop  5 best days: {R['drop5']:+.2f}%/yr (t {R['drop5_t']:+.2f})\")\n"
            "print(f\"drop 20 best days: {R['drop20']:+.2f}%/yr (t {R['drop20_t']:+.2f})\")\n"
            "print(f\"drop 50 best days: {R['drop50']:+.2f}%/yr (t {R['drop50_t']:+.2f})  \"\n"
            "      f\"<- below the {R['be_daily']:.2f}%/yr breakeven borrow\")\n"
            "print(f\"excluding 2020   : {R['ex2020']:+.2f}%/yr (t {R['ex2020_t']:+.2f})\")\n"
            "print()\n"
            "print(f\"and at a borrow equal to the breakeven, {R['yrs_below_be']} of \"\n"
            "      f\"{R['yr_all_positive']} calendar years are losers\")"
        ),
        md("## 3. The reset schedule is a risk dial, not a return dial\n\n"
           "> 💡 **In plain words:** letting the position drift looks more profitable, but "
           "the extra profit is an accidental bet on the Nasdaq, and the risk is not small.\n\n"
           "The equal-dollar short is short gamma on its own rebalance schedule: whichever "
           "leg is winning against you grows into the book between resets."),
        code(
            "print(f\"daily  : mean {R['mean']:+.2f}%  Sharpe {R['sharpe']:+.2f}  t {R['t']:+.2f}  \"\n"
            "      f\"DD {R['dd']:.1f}%  worst month {R['worst_month']:.1f}%  beta {R['beta']:+.3f}\")\n"
            "print(f\"weekly : mean {R['wk_mean']:+.2f}%  Sharpe {R['wk_sharpe']:+.2f}  t {R['wk_t']:+.2f}  \"\n"
            "      f\"DD {R['wk_dd']:.1f}%  worst month {R['wk_wm']:.1f}%  beta {R['wk_beta']:+.3f}\")\n"
            "print(f\"monthly: mean {R['mo_mean']:+.2f}%  Sharpe {R['mo_sharpe']:+.2f}  t {R['mo_t']:+.2f}  \"\n"
            "      f\"DD {R['mo_dd']:.1f}%  worst month {R['mo_wm']:.1f}%  beta {R['mo_beta']:+.3f}\")\n"
            "print(f\"never  : mean {R['nv_mean']:+.2f}%  DD {R['nv_dd']:.1f}%  \"\n"
            "      f\"gross short peaked at {R['nv_gross']:.1f}x  -> RUINED {R['nv_date']}\")\n"
            "print()\n"
            "print(f\"monthly's market-neutral alpha is only {R['mo_alpha']:+.2f}%/yr \"\n"
            "      f\"(t={R['mo_t_alpha']:+.2f}) -> the headline gain is residual beta, not harvest\")"
        ),
        md("## 4. The crux — borrow, an ASSUMPTION swept end to end\n\n"
           "> 💡 **In plain words:** the fee for borrowing the shares is the same size as "
           "the profit, so the whole verdict hangs on a number we cannot observe for free.\n\n"
           "Borrow is charged on the gross short notional (1.0x NAV), so the harvest falls "
           "one-for-one with the assumed rate."),
        code(
            "print(f\"borrow  0%: mean {R['mean']:+.2f}%/yr  (t {R['t']:+.2f})\")\n"
            "print(f\"borrow  1%: mean {R['b1_mean']:+.2f}%/yr  (t {R['b1_t']:+.2f})\")\n"
            "print(f\"borrow  2%: mean {R['b2_mean']:+.2f}%/yr  (t {R['b2_t']:+.2f})\")\n"
            "print(f\"borrow  3%: mean {R['b3_mean']:+.2f}%/yr  (t {R['b3_t']:+.2f})\")\n"
            "print(f\"borrow  5%: mean {R['b5_mean']:+.2f}%/yr  (t {R['b5_t']:+.2f})\")\n"
            "print(f\"borrow 15%: mean {R['b15_mean']:+.2f}%/yr\")\n"
            "print()\n"
            "print(f\"breakeven blended borrow: daily {R['be_daily']:.2f}%/yr, \"\n"
            "      f\"weekly {R['be_weekly']:.2f}%/yr, monthly {R['be_monthly']:.2f}%/yr\")\n"
            "print('(the weekly/monthly headroom is bought with the tail in section 3)')"
        ),
        md("## 5. Account terms — the rebate is a precondition, not a source\n\n"
           "The two legs' own financing carry averages to **+1x cash on the liability side**; "
           "the interest on the short proceeds is exactly what offsets it. So a full-rebate "
           "account's excess return is the cost residual in full, and a no-rebate account "
           f"*forfeits the cash rate itself* ({R['bil_mean']:.2f}%/yr here) — a cost of the "
           "account terms, not a component of the alpha. It happens to be worth more than "
           "half the harvest, and it grows with the level of rates."),
        code(
            "print(f\"full rebate: mean {R['mean']:+.2f}%/yr  Sharpe {R['sharpe']:+.2f}  t {R['t']:+.2f}  \"\n"
            "      f\"breakeven borrow {R['be_daily']:.2f}%/yr\")\n"
            "print(f\"   eras: 2010-2017 {R['fr_early']:+.2f}% (t={R['fr_early_t']:+.2f})  |  \"\n"
            "      f\"2018-2026 {R['fr_late']:+.2f}% (t={R['fr_late_t']:+.2f})\")\n"
            "print(f\"no rebate  : mean {R['nr_mean']:+.2f}%/yr  Sharpe {R['nr_sharpe']:+.2f}  t {R['nr_t']:+.2f}  \"\n"
            "      f\"breakeven borrow {R['nr_be']:.2f}%/yr\")\n"
            "print(f\"   eras: 2010-2017 {R['nr_early']:+.2f}% (t={R['nr_early_t']:+.2f})  |  \"\n"
            "      f\"2018-2026 {R['nr_late']:+.2f}% (t={R['nr_late_t']:+.2f})  <- dead\")\n"
            "print()\n"
            "print(f\"cost sweep (one-way bps): 0 -> {R['cost0']:+.2f}% (t {R['cost0_t']:+.2f}), \"\n"
            "      f\"5 -> {R['cost5']:+.2f}%, 10 -> {R['cost10']:+.2f}%  (turnover {R['turnover']:.1f}x/yr)\")"
        ),
        md("## 6. Live synthetic control — fee load vs decay, by construction\n\n"
           "> 💡 **In plain words:** two toy markets, identical melt in the levered funds, "
           "one with fees and one without. Only the fee shows up in the harvest.\n\n"
           "Both worlds carry the same large `0.5*L*(L-1)*σ²` decay; `signal_strength` "
           "scales only the funds' expense-plus-financing load. Six seeds each."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from short_pair import data, strategy as st\n"
            "for tag, ss in [('costly funds (planted)', 1.0), ('free funds (null)     ', 0.0)]:\n"
            "    means, ts = [], []\n"
            "    for px, truth in data.synthetic_panel(n_seeds=6, signal_strength=ss):\n"
            "        d = st.synthetic_detect(px, cost_bps=0.0)\n"
            "        means.append(d['ann_mean']); ts.append(d['t_hac'])\n"
            "    print('%s planted %+.2f%%/yr -> recovered %+.2f%%/yr (sd %.2f%%), max|t| %.2f, '\n"
            "          'decay in legs %.1f%%/%.1f%% per yr'\n"
            "          % (tag, truth['expected_harvest_ann']*100, np.mean(means)*100,\n"
            "             np.std(means, ddof=1)*100, max(abs(t) for t in ts),\n"
            "             truth['decay_long_ann']*100, truth['decay_short_ann']*100))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Real.** Excess-of-cash **+{R['mean']:.2f}%/yr before borrow**, HAC "
           f"*t* = **{R['t']:+.2f}**, bootstrap mean CI "
           f"[+{R['ci_m_lo']:.2f}%, +{R['ci_m_hi']:.2f}%] clear of zero, residual beta "
           f"{R['beta']:+.4f} to QQQ with alpha *t* = {R['t_alpha']:+.2f}, positive in every "
           f"calendar year, in both eras (+{R['fr_early']:.2f}% / +{R['fr_late']:.2f}%), "
           f"and holding up as a magnitude without its 50 best days (+{R['drop50']:.2f}%) or "
           f"without 2020 (+{R['ex2020']:.2f}%) — no *t* quoted for those two, per §2b. "
           f"What is *not* "
           f"evidence: the harvest equalling the implied load (+{R['implied_load']:.2f}%/yr) "
           f"is an **identity**, and the load is an **unattributed residual**. What is "
           f"evidence: the synthetic control recovers a planted load "
           f"({R['syn_planted']:+.2f}% → {R['syn_recovered']:+.2f}%, max|*t*| {R['syn_t']:.1f}) "
           f"while returning {R['syn_null']:+.2f}% (max|*t*| {R['syn_null_t']:.2f}) when the "
           f"same {R['syn_decay_l']:.0f}%/{R['syn_decay_s']:.0f}% decay carries no fee — so "
           f"the volatility decay is *not* what is being collected. **Pair selection:** the "
           f"universe is n = 1 and hindsight-chosen. For the *signal* that is conservative "
           f"(TQQQ/SQQQ is the cheapest, best-run levered pair, so its residual is near a "
           f"floor); for the *borrow* it runs the other way, and a fund being wound up "
           f"mid-sample would **not** have paid a short — an ETF liquidation pays NAV and the "
           f"short closes flat.\n"
           f"- **Tradability — Fragile.** Breakeven blended borrow **{R['be_daily']:.2f}%/yr**, "
           f"but split by leg the cheap TQQQ side carries {R['sf_share_long']:.0f}% of the "
           f"harvest ({R['sf_long']:.2f}%/yr) and the crowded SQQQ side breaks even at only "
           f"**{R['sf_short']:.2f}%/yr**; at the blended rate "
           f"{R['yrs_below_be']} of {R['yr_all_positive']} calendar years lose money; no "
           f"rebate on the proceeds cuts "
           f"the harvest to +{R['nr_mean']:.2f}%/yr and to +{R['nr_late']:.2f}%/yr "
           f"(*t* = {R['nr_late_t']:+.2f}) since 2018 — a Mirage at retail terms. The "
           f"apparent fix — reset less often — "
           f"buys residual beta, not alpha ({R['mo_alpha']:+.2f}%/yr, "
           f"*t* = {R['mo_t_alpha']:+.2f}), at Sharpe {R['mo_sharpe']:+.2f} and a "
           f"{R['mo_wm']:.1f}% month; not resetting at all ruins the book on {R['nv_date']} "
           f"at {R['nv_gross']:.1f}x gross. It is a prime-brokerage carry of ~2%/yr on 1.3% "
           f"vol, entirely at the mercy of a fee that exists to price it away."),
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
