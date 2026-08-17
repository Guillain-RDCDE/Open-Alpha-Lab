"""Generate the two narrative notebooks for Study 956 (the ADR custody fee).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below, which mirrors ``docs/results.md``; the only live cells run the fast
offline synthetic control, and they are never placed under a real-tape banner.
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


# Frozen real-tape headline — mirror of docs/results.md. Fifteen ADR / home-line / FX
# triples, total-return and price-only closes, 2000-01-03 -> 2026-06-30, as-of 2026-06-30.
R = dict(
    start="2000-01-03", end="2026-06-30", n_rows=6649, fp="ccd87508f03e",
    n_pairs=15, n_kept=10, n_dropped=5,
    uk_home_yield_lo=0.04, uk_home_yield_hi=0.05, uk_adr_yield_lo=3.58, uk_adr_yield_hi=5.82,
    uk_ratio_lo=97, uk_ratio_hi=114, uk_fake_fee=-5.4,
    gap_mean=13.80, gap_median=7.53, gap_sd=18.48, gap_t=2.36, gap_pos=9, gap_n=10,
    gap_ci_lo=4.96, gap_ci_hi=25.65, gap_frac_le0=0.000, sign_p=0.0107,
    cents_mean=9.01, cents_median=5.34, cents_ci_lo=3.09, cents_ci_hi=17.57,
    total_mean=16.71, total_t=1.92, placebo_mean=3.00, placebo_t=0.58,
    loo_lo=8.75, loo_hi=15.43, loo_t_lo=1.92, loo_t_hi=2.67,
    loo_worst="E", loo_best_drop="NVS",
    boot_clear=3, boot_names="NVS, TSM, TM",
    era_e_gap=19.99, era_e_t=2.65, era_e_pos=10, era_l_gap=10.23, era_l_t=2.09, era_l_pos=6,
    wht0=13.80, wht0_t=2.36, wht05=-14.67, wht1=-43.15, wht1_t=-5.50, wht15=-71.62,
    wht_cost_lo=26, wht_cost_hi=96,
    block_t=1.84, block_n=5, drop_nvs_mean=8.75,
    raw_sw_bp=944.7, raw_sw_t=1.05, clean_rows=24, clean_total=83728,
    brk006=20.11, brk006_t=2.56, brk010=16.71, brk010_t=1.92,
    sw_n=4071, sw_adr_sharpe=0.547, sw_loc_sharpe=0.608,
    sw_gross_bp=14.6, sw_gross_t=0.11, sw_fx30_bp=12.8, sw_mid_bp=-2.2, sw_full_bp=-18.5,
    syn_planted=77.5, syn_recovered=76.31, syn_custody=24.33, syn_null=0.11,
    syn_null8_mean=-0.28, syn_null8_sd=1.69, syn_break=74.51,
    fee_schedule_lo=1, fee_schedule_hi=5,
    names=(("TTE", 5.1, 1.58), ("SNY", -0.8, -0.42), ("SAP", 10.0, 1.78),
           ("PHG", 1.9, 2.65), ("ING", 4.4, 2.81), ("E", 31.6, 6.83),
           ("NVS", 59.3, 18.26), ("NVO", 1.5, 12.10), ("TM", 11.9, 7.69),
           ("TSM", 13.2, 13.62)),
)


# The small slice of R the plain-language notebook prints, so its first code cell stays
# readable instead of dumping the whole frozen dict.
CURIOUS_R = dict(
    uk_home=R["uk_home_yield_hi"], uk_adr=R["uk_adr_yield_hi"], uk_ratio=R["uk_ratio_hi"],
    fake=R["uk_fake_fee"], kept=R["n_kept"], pairs=R["n_pairs"],
    gap_mean=R["gap_mean"], gap_median=R["gap_median"], gap_pos=R["gap_pos"],
    gap_n=R["gap_n"], sign_p=R["sign_p"], cents_median=R["cents_median"],
    cents_ci_lo=R["cents_ci_lo"], cents_ci_hi=R["cents_ci_hi"],
)


HEADER = f"""# Study 956 — The Custody Fee 🏦

**The depositary bank charges you 1-5 cents per ADS per year and never sends a bill. Can the
tape see it?**

An American Depositary Receipt is a claim on foreign shares held by a depositary bank. The
bank charges the *holder* a pass-through custody fee — published in the deposit agreement,
typically **{R['fee_schedule_lo']}-{R['fee_schedule_hi']} cents per ADS per year** — and nets
it out of a dividend rather than billing it. The foreign tax authority withholds tax on the
same dividend. Neither leak ever appears on a price chart.

We test it on **{R['n_pairs']} ADR / home-line pairs** across ten countries,
{R['start']} → {R['end']} ({R['n_rows']:,} rows), using daily **total-return** and
**price-only** closes.

*Real-tape numbers below are the frozen headline (`docs/results.md`, fingerprint
`{R['fp']}`); the live cells run the fast offline synthetic control and are labelled as such.
As-of 2026-06-30.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The bill nobody sends you\n\n"
           "When you buy **TM** you are not buying a Toyota share. You are buying a receipt "
           "that a bank in New York issues against Toyota shares it holds in Tokyo. The bank "
           "does real work — collecting the dividend, converting the yen, handling the tax "
           "paperwork — and it charges you for it. A few cents per receipt per year, taken "
           "quietly out of the dividend before it lands in your account.\n\n"
           "You will never see it on a chart, because the *price* of the receipt tracks the "
           "share almost perfectly. It only shows up in what you are actually **paid**.\n\n"
           "> 🔬 *For the quants:* the estimand is the slope in time of "
           "`log(ADR total return) − log(home line total return × FX)`. Daily differences are "
           "hopeless here — non-synchronous closes put 1-2 % of noise a day around a fee "
           "worth 0.1 % a *year* — but that noise is stationary and reverses, so it does not "
           "accumulate, while the fee does. A trend fit on the level sees it; a mean of "
           "returns never will."),
        md("## 2. First, the trap — London has no dividends\n\n"
           f"We started with fifteen pairs. Five of them — Shell, BP, HSBC, Unilever, Rio "
           f"Tinto — are London listings, and the naive comparison said their ADRs were "
           f"bleeding **{abs(R['uk_fake_fee'])} % a year**. A hundred times any real fee.\n\n"
           f"The reason is not finance, it is plumbing: the data vendor's \"adjusted close\" "
           f"for the London Stock Exchange is **split-adjusted only**. The home leg's entire "
           f"dividend stream is missing. Its measured yield is "
           f"**{R['uk_home_yield_lo']}-{R['uk_home_yield_hi']} %/yr** against the ADR's "
           f"**{R['uk_adr_yield_lo']}-{R['uk_adr_yield_hi']} %** — a ratio of "
           f"**{R['uk_ratio_lo']}-{R['uk_ratio_hi']}×**.\n\n"
           f"So we built a screen that reads only the *home* line's own yield — never the "
           f"ADR's, never the gap — and it throws all five out automatically. "
           f"**{R['n_kept']} of {R['n_pairs']} pairs survive.** Anyone who skips this step "
           f"will publish the 5 % number as a discovery."),
        code(
            "R = " + repr(CURIOUS_R) + "\n"
            "print(f\"London home leg yield : {R['uk_home']:.2f} %/yr   <- dividends missing entirely\")\n"
            "print(f\"London ADR  leg yield : {R['uk_adr']:.2f} %/yr\")\n"
            "print(f\"ratio                 : {R['uk_ratio']} x  ->  a fabricated \"\n"
            "      f\"{R['fake']:.1f} %/yr 'custody fee'\")\n"
            "print(f\"pairs surviving the coverage screen: {R['kept']} of {R['pairs']}\")"
        ),
        md(f"## 3. On the ten pairs that work, something is there\n\n"
           f"Across the surviving ten, the ADR hands its holder **{R['gap_mean']:.1f} basis "
           f"points a year** less income than the home line — median "
           f"**{R['gap_median']:.1f} bp**. In the unit the depositaries publish, that is a "
           f"median of **{R['cents_median']:.1f} cents per ADS per year**, which lands right "
           f"on their stated **{R['fee_schedule_lo']}-{R['fee_schedule_hi']} cent** band. "
           f"Hold that thought — section 5 explains why landing on the band is *not* the same "
           f"as having measured the fee.\n\n"
           f"**{R['gap_pos']} of {R['gap_n']}** names point the same way — a coin would do "
           f"that about once in a hundred tries (*p* = {R['sign_p']:.3f}). Resampling the "
           f"issuers puts the average at **[{R['gap_ci_lo']:.1f}, {R['gap_ci_hi']:.1f}] "
           f"bp/yr**, clear of zero."),
        code(
            "names = " + repr(list(R["names"])) + "\n"
            "print(f\"{'name':<6} {'gap bp/yr':>10} {'HAC t':>8}\")\n"
            "for n, g, t in names:\n"
            "    print(f\"{n:<6} {g:>10.1f} {t:>8.2f}\")\n"
            "print()\n"
            "print(f\"pooled mean {R['gap_mean']:+.2f} bp/yr, median {R['gap_median']:+.2f}, \"\n"
            "      f\"sign test {R['gap_pos']}/{R['gap_n']} positive (p = {R['sign_p']:.3f})\")\n"
            "print(f\"in cents per ADS per year: median {R['cents_median']:.2f} c, \"\n"
            "      f\"95% CI [{R['cents_ci_lo']:.2f}, {R['cents_ci_hi']:.2f}]\")\n"
            "print('published depositary schedules: 1-5 c/ADS/yr')"
        ),
        md(f"## 4. But it is a whisper, not a shout\n\n"
           f"Honesty first. The pooled *t* is **+{R['gap_t']:.2f}** — barely over the desk's "
           f"bar — and dropping a single name (Eni) takes it to **+{R['loo_t_lo']:.2f}**, "
           f"under it. Worse, the ten names are not ten independent facts: six of them are "
           f"euro-area issuers sharing one currency, one treaty rate and one dividend "
           f"calendar. Count each currency block once and the *t* is "
           f"**+{R['block_t']:.2f}** on {R['block_n']} observations.\n\n"
           f"Name by name the leak is invisible: a block bootstrap clears zero on only "
           f"**{R['boot_clear']} of {R['gap_n']}** issuers. And the average is propped up by "
           f"Novartis, whose +59 bp/yr is really two spin-offs (Alcon 2019, Sandoz 2023) that "
           f"the two data feeds record differently — drop it and the mean halves to "
           f"**+{R['drop_nvs_mean']:.1f} bp/yr**.\n\n"
           f"It is also shrinking. Split the sample in 2015: **{R['era_e_gap']:.1f} bp/yr** "
           f"before, **{R['era_l_gap']:.1f} bp/yr** since. Both positive — but half the size, "
           f"and only {R['era_l_pos']} of {R['gap_n']} names still point the right way.\n\n"
           f"One more thing we cannot fix: every issuer here is still listed on both venues in "
           f"2026. That is a **survivor panel**, and the ADRs where fee disputes actually end "
           f"up — the de-sponsored ones — are absent by construction."),
        md(f"## 5. And we cannot tell fee from tax\n\n"
           f"The plan was to split the leak in two: the depositary's fee, and the foreign "
           f"government's withholding tax. It failed, and the failure is informative.\n\n"
           f"At the treaty rates, withholding alone should cost "
           f"**{R['wht_cost_lo']}-{R['wht_cost_hi']} bp/yr** on these dividend yields — three "
           f"to seven times the *entire* gap we measure. Subtract it and every single name's "
           f"\"custody fee\" goes **negative** ({R['wht1']:.1f} bp/yr on average). That is not "
           f"a finding about fees; it is proof that the withholding tax **is not in the "
           f"total-return series at all** — the per-ADS dividend the data vendor records is "
           f"the gross declared amount.\n\n"
           f"The fallback plan failed too. The UK charges *no* dividend withholding tax, so "
           f"the five London pairs would have pinned the fee exactly — except those are the "
           f"same five names section 2 threw out.\n\n"
           f"And now follow that first argument one step further, because it is the honest "
           f"punchline of the study. If the vendor records the **gross declared** per-ADS "
           f"dividend, then a fee the depositary bills straight to your brokerage account "
           f"through DTC — which is how a great deal of the schedule is actually collected — "
           f"never reaches this tape either. So **{R['gap_mean']:.1f} bp/yr is an upper bound "
           f"on the depositary fee, not a measurement of it**. It is a combined income "
           f"shortfall that is also consistent with the depositary's FX-conversion spread, "
           f"with rounding of the per-ADS rate, and with feed differences on special "
           f"dividends. Landing on the published 1-5 cent band is suggestive. It is not "
           f"proof. **This study measures a leak; it does not name it.**"),
        md("## 6. Live check — the machinery is unbiased (offline synthetic)\n\n"
           "Below we build a world where we *know* the answer: a home line, an FX cross, and "
           "an ADR whose dividend is docked by a planted fee and a planted tax. The estimator "
           "must recover the planted number, and must report ~zero when we switch the fee "
           "off. **This cell is synthetic — none of its output is a real-tape result.**"),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from adr_drag import data, strategy as st\n"
            "\n"
            "frames, truth = data.synthetic_panel(n_names=8, drag_bps_per_year=25.0, signal_strength=1.0)\n"
            "whts = {k: truth['per_name'][k]['wht'] for k in frames}\n"
            "planted = st.synthetic_detect(frames, whts)\n"
            "print('SYNTHETIC world with a planted fee')\n"
            "print('  planted total leak  : %+.2f bp/yr' % (truth['planted_gap_per_year']*1e4))\n"
            "print('  recovered           : %+.2f bp/yr' % (planted['income_gap']['mean']*1e4))\n"
            "print('  residual custody    : %+.2f bp/yr (planted %.1f)'\n"
            "      % (planted['custody']['mean']*1e4, truth['custody_drag_per_year']*1e4))\n"
            "\n"
            "frames0, truth0 = data.synthetic_panel(n_names=8, drag_bps_per_year=25.0, signal_strength=0.0)\n"
            "null = st.synthetic_detect(frames0, {k: 0.0 for k in frames0})\n"
            "print('SYNTHETIC world with NO fee and NO tax')\n"
            "print('  recovered           : %+.2f bp/yr  (should be ~0)' % (null['income_gap']['mean']*1e4))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** An income shortfall is on the tape: "
           f"**+{R['gap_mean']:.1f} bp/yr**, {R['gap_pos']}/{R['gap_n']} names, sign test "
           f"*p* = {R['sign_p']:.3f}, a median of **{R['cents_median']:.1f} cents per ADS/yr** "
           f"in the range of the published schedules, positive in both eras. But "
           f"*t* = +{R['gap_t']:.2f} is knife-edge (leave-one-out +{R['loo_t_lo']:.2f}; "
           f"+{R['block_t']:.2f} once the euro names are counted once), a third of the "
           f"intended sample was destroyed by a vendor data defect, the panel is a survivor "
           f"panel, and the leak cannot be attributed — not to the tax, and not to the fee.\n"
           f"- **Tradability — Mirage.** Owning the home lines instead wins "
           f"**+{R['sw_gross_bp']:.1f} bp/yr gross at *t* = +{R['sw_gross_t']:.2f}**, and a "
           f"15 bp/yr foreign safekeeping charge flips it negative. Budget ~5-10 cents per ADS "
           f"per year of invisible wrapper cost and get on with your life."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md(f"# Study 956 — The Custody Fee — the teardown\n\n"
           f"The trend-in-levels estimator with break segmentation, its price-ratio placebo, "
           f"the currency-free income decomposition, HAC versus block bootstrap, the sign "
           f"test and leave-one-out, the era cut, the withholding sweep that demonstrates "
           f"non-identification, and the live synthetic control. Every real number is frozen "
           f"from `docs/results.md` (fingerprint `{R['fp']}`)."),
        code("R = %r" % (R,)),
        md("## The estimator\n\n"
           "For each pair, `x_t = log(ADR_TR_t) − log(home_TR_t × FX_t)`. Regress `x` on time "
           "in years, **centred within each break segment**, with one dummy per segment; the "
           "drag is minus the common slope. Newey-West at **252 lags** because the regression "
           "error is the arbitrage band, which is near-unit-root.\n\n"
           "Two companion fits: the same regression on the **price-only** ratio (a placebo "
           "that must be flat — an ADS is a fixed number of shares), and the **income gap**, "
           "`[log(TR) − log(price)]_ADR − [log(TR) − log(price)]_home`, which is the headline "
           "because FX multiplies a leg's two closes identically and therefore cancels.\n\n"
           "> 💡 *In plain words:* we are not asking whether the receipt's price drifts from "
           "the share's. We are asking whether the receipt pays you less."),
        md("## The coverage screen and what it removes"),
        code(
            "print(f\"pairs loaded {R['n_pairs']}, kept {R['n_kept']}, dropped {R['n_dropped']} (all LSE)\")\n"
            "print(f\"dropped names' HOME yield {R['uk_home_yield_lo']:.2f}-{R['uk_home_yield_hi']:.2f} %/yr \"\n"
            "      f\"vs ADR yield {R['uk_adr_yield_lo']:.2f}-{R['uk_adr_yield_hi']:.2f} %/yr \"\n"
            "      f\"-> ratio {R['uk_ratio_lo']}-{R['uk_ratio_hi']}x\")\n"
            "print(f\"naive drag on those names: {R['uk_fake_fee']:.1f} %/yr -- a vendor artefact, not a fee\")\n"
            "print('the gate reads the HOME leg only, so it cannot select on the estimand')"
        ),
        md("## Headline — pooled across names\n\n"
           "Names are the observations; the estimation error is dominated by each pair's own "
           "arbitrage band while the *fee* is a common institutional parameter, so the "
           "cross-name statistic tests \"is the average leak non-zero\", not any one name."),
        code(
            "print(f\"total drag    mean {R['total_mean']:+6.2f} bp/yr  t {R['total_t']:+5.2f}\")\n"
            "print(f\"price placebo mean {R['placebo_mean']:+6.2f} bp/yr  t {R['placebo_t']:+5.2f}  <- flat, as it must be\")\n"
            "print(f\"income gap    mean {R['gap_mean']:+6.2f} bp/yr  median {R['gap_median']:+6.2f}  \"\n"
            "      f\"sd {R['gap_sd']:.2f}  t {R['gap_t']:+5.2f}  positive {R['gap_pos']}/{R['gap_n']}\")\n"
            "print(f\"  name-bootstrap 95% CI [{R['gap_ci_lo']:+.2f}, {R['gap_ci_hi']:+.2f}] bp/yr, \"\n"
            "      f\"share<=0 {R['gap_frac_le0']:.3f}\")\n"
            "print(f\"  sign test p = {R['sign_p']:.4f}\")\n"
            "print(f\"  in cents/ADS/yr: mean {R['cents_mean']:.2f}, median {R['cents_median']:.2f}, \"\n"
            "      f\"CI [{R['cents_ci_lo']:.2f}, {R['cents_ci_hi']:.2f}] vs a published \"\n"
            "      f\"{R['fee_schedule_lo']}-{R['fee_schedule_hi']} c/ADS/yr schedule\")"
        ),
        md("## Where it breaks down\n\n"
           "The HAC *t* is generous and the block bootstrap is not. Report both. And the "
           "cross-name *t* assumes ten independent issuers, which is false: six of the ten "
           "are euro-area names sharing a currency, a treaty rate and a dividend calendar, "
           "so the honest unit of resampling is the currency block, not the name."),
        code(
            "print(f\"leave-one-out mean range [{R['loo_lo']:.2f}, {R['loo_hi']:.2f}] bp/yr, \"\n"
            "      f\"t range [{R['loo_t_lo']:+.2f}, {R['loo_t_hi']:+.2f}]\")\n"
            "print(f\"  dropping {R['loo_worst']} alone takes t under 2; dropping {R['loo_best_drop']} \"\n"
            "      f\"(two spin-offs) halves the mean to {R['drop_nvs_mean']:.2f} bp/yr\")\n"
            "print(f\"collapse the 6 euro names to one obs (EUR/CHF/DKK/JPY/TWD): \"\n"
            "      f\"t {R['block_t']:+.2f} on n = {R['block_n']}  <- below the bar\")\n"
            "print(f\"per-name 63-day block bootstrap clears zero on {R['boot_clear']}/{R['gap_n']} \"\n"
            "      f\"names ({R['boot_names']}) -- name by name the leak is invisible\")\n"
            "print(f\"price placebo contaminated on PHG (+32.5), NVS (+23.2), TSM (-30.6) bp/yr \"\n"
            "      f\"-> the TOTAL-drag column is unusable for those; the income gap is not\")\n"
            "print('per-name HAC t on the income leg is NOT to be trusted (near-deterministic')\n"
            "print('  staircase => tiny residual => NVO reads t +12.10 on a 1.5 bp/yr gap)')\n"
            "print('survivorship: every issuer is still dual-listed in 2026 -- a survivor panel')"
        ),
        md("## Era cut (split 2015-01-01) and break-threshold sweep"),
        code(
            "print(f\"2000-2014: income gap {R['era_e_gap']:+6.2f} bp/yr (t {R['era_e_t']:+5.2f}), \"\n"
            "      f\"positive {R['era_e_pos']}/{R['gap_n']}\")\n"
            "print(f\"2015-2026: income gap {R['era_l_gap']:+6.2f} bp/yr (t {R['era_l_t']:+5.2f}), \"\n"
            "      f\"positive {R['era_l_pos']}/{R['gap_n']}  <- same sign, half the size\")\n"
            "print()\n"
            "print(f\"break threshold 0.06 -> total drag {R['brk006']:+6.2f} (t {R['brk006_t']:+5.2f})\")\n"
            "print(f\"break threshold 0.10 -> total drag {R['brk010']:+6.2f} (t {R['brk010_t']:+5.2f})  \"\n"
            "      f\"(0.15 and 0.25 identical: exactly ONE level shift exists in the kept \"\n"
            "      f\"panel -- ING, 2009-11-23, the state-aid rights issue)\")"
        ),
        md("## The ASSUMPTION sweep — the fee/tax split is not identified\n\n"
           "The withholding rate is the only non-tape input (treaty rates, 15 % to 21 %). "
           "Scaling it moves the residual \"custody\" term straight through zero and out the "
           "other side.\n\n"
           "The intended anchor was the **UK** — no dividend withholding tax at all, so a UK "
           "pair's income gap *is* the custody fee by law. All five UK pairs are London "
           "listings and all five die on the coverage screen. Nothing identifies the split.\n\n"
           "> ⚠️ **And the same logic bounds the fee, not just the tax.** If the vendor's "
           "per-ADS dividend is the *gross declared* amount — which is what this sweep "
           "proves — then a depositary fee billed through DTC as a separate account charge "
           "never enters the series either. The measured gap is an **upper bound on the "
           "fee**, and is equally consistent with the depositary's FX-conversion spread, "
           "per-ADS rounding, and feed differences on special dividends and spin-offs. This "
           "study measures a leak; it does not name it."),
        code(
            "print(f\"0.0 x treaty -> residual custody {R['wht0']:+7.2f} bp/yr (t {R['wht0_t']:+5.2f})\")\n"
            "print(f\"0.5 x treaty -> residual custody {R['wht05']:+7.2f} bp/yr\")\n"
            "print(f\"1.0 x treaty -> residual custody {R['wht1']:+7.2f} bp/yr (t {R['wht1_t']:+5.2f})\")\n"
            "print(f\"1.5 x treaty -> residual custody {R['wht15']:+7.2f} bp/yr\")\n"
            "print()\n"
            "print(f\"treaty withholding alone would cost {R['wht_cost_lo']}-{R['wht_cost_hi']} bp/yr \"\n"
            "      f\"on these yields -- 3x to 7x the WHOLE measured gap of {R['gap_mean']:.1f} bp/yr\")\n"
            "print('=> the withholding is absent from the ADR total-return series; only the')\n"
            "print('   0.0x column is defensible, and it makes the gap an UPPER BOUND on the fee')"
        ),
        md("## The only traded leg — own the home line instead\n\n"
           "Equal-weight baskets, excess-of-cash versus BIL **on both legs**, long-only (no "
           "short leg, hence no borrow), one-way FX conversion × NAV executed at **t+1** on a "
           "decision formed at *t*. Both frictions are assumptions and both are swept.\n\n"
           "> ⚠️ **This leg is a measurement, not a backtest.** Two reasons. (1) It inherits "
           "the bad-print screen from `data.load_pair`, whose rolling median is *centred* and "
           "therefore peeks forward. That is inert on the headline — the pooled income gap is "
           "identical with the filter off — but decisive here: switch it off and the same race "
           "reads **+944.7 bp/yr at HAC t = +1.05**, because 24 corrupt rows out of 83,728 "
           "(0.03 %) dominate an arithmetic mean of daily returns. (2) Rebalancing turnover is "
           "charged on neither leg, and only the home leg would really pay FX on every trade. "
           "Both readings are Mirage; the cleaned one is the honest magnitude."),
        code(
            "print(f\"n = {R['sw_n']} common days (BIL inception gates the cash leg)\")\n"
            "print(f\"ADR basket excess-of-cash Sharpe {R['sw_adr_sharpe']:+.3f} \"\n"
            "      f\"vs home basket {R['sw_loc_sharpe']:+.3f}\")\n"
            "print(f\"gross          : {R['sw_gross_bp']:+6.1f} bp/yr  HAC t {R['sw_gross_t']:+.2f}\")\n"
            "print(f\"30 bp FX       : {R['sw_fx30_bp']:+6.1f} bp/yr\")\n"
            "print(f\"30 bp + 15 bp/yr custody: {R['sw_mid_bp']:+6.1f} bp/yr  <- already negative\")\n"
            "print(f\"50 bp + 30 bp/yr custody: {R['sw_full_bp']:+6.1f} bp/yr\")\n"
            "print()\n"
            "print(f\"LOOK-AHEAD CHECK -- hygiene filter OFF: {R['raw_sw_bp']:+.1f} bp/yr \"\n"
            "      f\"HAC t {R['raw_sw_t']:+.2f}  ({R['clean_rows']} of {R['clean_total']:,} \"\n"
            "      f\"rows = {100*R['clean_rows']/R['clean_total']:.3f}% drive the difference)\")\n"
            "print('  same check on the HEADLINE income gap: +13.80 bp/yr either way -- inert')"
        ),
        md("## Live synthetic control — recovery, silence on the null, immunity to a ratio break\n\n"
           "**Synthetic only.** Planted fee: the estimator must recover it and split it "
           "correctly when the withholding rate is known. Null: it must report ~zero. Planted "
           "ADS-ratio step: the segment fixed effects must absorb it."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from adr_drag import data, strategy as st\n"
            "\n"
            "fr, tr = data.synthetic_panel(n_names=8, drag_bps_per_year=25.0, signal_strength=1.0)\n"
            "w = {k: tr['per_name'][k]['wht'] for k in fr}\n"
            "d = st.synthetic_detect(fr, w)\n"
            "print('planted gap %+.2f -> recovered %+.2f bp/yr (t %+.1f); residual custody %+.2f (planted %.1f)'\n"
            "      % (tr['planted_gap_per_year']*1e4, d['income_gap']['mean']*1e4,\n"
            "         d['income_gap']['t'], d['custody']['mean']*1e4, tr['custody_drag_per_year']*1e4))\n"
            "print('price placebo %+.2f bp/yr (the fee is netted from the dividend, never the price)'\n"
            "      % (d['price_drift']['mean']*1e4))\n"
            "\n"
            "nulls = []\n"
            "for s in range(6):\n"
            "    f0, t0 = data.synthetic_panel(n_names=6, n_years=10, drag_bps_per_year=25.0,\n"
            "                                  signal_strength=0.0, seed=956 + 11*s)\n"
            "    nulls.append(st.synthetic_detect(f0, {k: 0.0 for k in f0})['income_gap']['mean'])\n"
            "nulls = np.array(nulls) * 1e4\n"
            "print('null x6 seeds: mean %+.2f bp/yr (sd %.2f), |mean| >= 5 bp on %d/6'\n"
            "      % (nulls.mean(), nulls.std(ddof=1), (np.abs(nulls) >= 5).sum()))\n"
            "\n"
            "fb, tb = data.synthetic_panel(n_names=4, drag_bps_per_year=25.0, ratio_break=0.70)\n"
            "db = st.synthetic_detect(fb, {k: tb['per_name'][k]['wht'] for k in fb})\n"
            "print('with a planted 0.70-log ADS-ratio step: total drag %+.2f bp/yr (planted %.1f)'\n"
            "      % (db['drag']['mean']*1e4, tb['planted_gap_per_year']*1e4))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** The income shortfall is real and correctly signed: "
           f"**+{R['gap_mean']:.2f} bp/yr** pooled (median +{R['gap_median']:.2f}), "
           f"{R['gap_pos']}/{R['gap_n']} positive (sign test *p* = {R['sign_p']:.3f}), "
           f"name-bootstrap CI **[+{R['gap_ci_lo']:.2f}, +{R['gap_ci_hi']:.2f}] bp/yr** with "
           f"share ≤ 0 of {R['gap_frac_le0']:.3f}, positive in both eras "
           f"(+{R['era_e_gap']:.1f} / +{R['era_l_gap']:.1f}), and a median "
           f"**{R['cents_median']:.1f} c/ADS/yr** in the range of the published "
           f"{R['fee_schedule_lo']}-{R['fee_schedule_hi']} c schedules. It is not robust: "
           f"*t* = +{R['gap_t']:.2f} → +{R['loo_t_lo']:.2f} on leave-one-out and "
           f"+{R['block_t']:.2f} on {R['block_n']} currency blocks, "
           f"{R['boot_clear']}/{R['gap_n']} names clear a block-bootstrap CI, dropping "
           f"Novartis halves the mean to +{R['drop_nvs_mean']:.2f}, the price "
           f"placebo is contaminated on three names, {R['n_dropped']}/{R['n_pairs']} pairs "
           f"are unusable on a vendor defect, and the panel is a **survivor panel**. Nor is "
           f"the leak attributable: assumed withholding "
           f"{R['wht_cost_lo']}-{R['wht_cost_hi']} bp/yr exceeds the whole gap, so the tax is "
           f"absent from the tape — and by the same argument a DTC-billed depositary fee "
           f"would be too, making +{R['gap_mean']:.2f} bp/yr an **upper bound** rather than a "
           f"fee. The synthetic control recovers a "
           f"planted {R['syn_planted']:.1f} bp/yr as {R['syn_recovered']:.2f}, is silent on "
           f"the null (mean {R['syn_null8_mean']:+.2f} bp/yr over 8 seeds) and survives a "
           f"planted ratio break ({R['syn_break']:.2f}) — the ambiguity is the tape's, not "
           f"the harness's.\n"
           f"- **Tradability — Mirage.** The home basket beats the ADR basket by "
           f"**+{R['sw_gross_bp']:.1f} bp/yr gross at HAC *t* = +{R['sw_gross_t']:.2f}** and "
           f"turns negative ({R['sw_mid_bp']:+.1f} bp/yr) at a 15 bp/yr foreign safekeeping "
           f"charge, with rebalancing turnover charged on neither leg and a forward-peeking "
           f"hygiene filter baked in (without it: {R['raw_sw_bp']:+.1f} bp/yr at "
           f"*t* = +{R['raw_sw_t']:.2f}, driven by "
           f"{100*R['clean_rows']/R['clean_total']:.3f} % of the rows). A real, small, "
           f"unavoidable cost of the wrapper — not an edge."),
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
