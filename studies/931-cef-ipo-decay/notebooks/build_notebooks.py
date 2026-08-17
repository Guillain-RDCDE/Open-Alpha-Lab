"""Generate the two narrative notebooks for Study 931 (the CEF IPO hole).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below (a mirror of docs/results.md); the only live cells run the fast
offline **synthetic** control, and they are labelled as such — no synthetic number ever
appears under a real-tape banner.
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
# 28 CEF IPOs (2012-2022 vintages) vs asset-class benchmark ETFs, total return,
# entry at the offering price, as-of 2026-06-30.
# --------------------------------------------------------------------------- #
R = dict(
    start="2012-05-25", end="2026-06-30", n_days=3543, fp="d69758d98b1a",
    n_funds=28, n_vintages=9, offer_ok=21, worst_gap=-7.2,
    # abnormal total return vs asset-class benchmark
    m1=-1.47, m1_med=-1.12, m1_neg=0.68, m1_t=-1.34,
    m3=-5.13, m3_med=-4.22, m3_neg=0.79, m3_t=-3.94,
    m6=-7.06, m6_med=-8.19, m6_neg=0.82, m6_t=-4.06,
    m12=-10.26, m12_med=-9.45, m12_neg=0.89, m12_t=-3.65,
    m12_wilson_lo=0.73, m12_wilson_hi=0.96,
    # vintage-cluster bootstrap
    cb3_lo=-7.93, cb3_hi=-3.47, cb6_lo=-10.15, cb6_hi=-3.59,
    cb12_lo=-16.39, cb12_hi=-3.65, cb_clusters=9,
    # calendar-time portfolio
    ct1_ann=-13.79, ct1_t=-1.37,
    ct3_ann=-21.30, ct3_t=-2.86, ct3_lo=-34.29, ct3_hi=-9.23,
    ct6_ann=-11.58, ct6_t=-2.12,
    ct12_ann=-5.80, ct12_t=-1.57, ct12_lo=-12.63, ct12_hi=0.03,
    ct_book=1.37, ct_book_max=3, ct_byfund=-8.37,
    # time-to-discount proxy
    ttd3_med=27, ttd5_med=34, ttd5_q1=23, ttd5_q3=60, ttd5_cross=27,
    ttd10_med=97,
    # era cut
    era_e_n=12, era_e=-7.14, era_e_neg=0.83, era_e_t=-2.10,
    era_l_n=16, era_l=-12.60, era_l_neg=0.94, era_l_t=-2.99,
    # sweeps
    map_spy=-16.84, map_spy_t=-5.22, map_aor=-9.11, map_aor_t=-3.08,
    entry_close=-9.55, entry_close_t=-3.35,
    # placebo + seasoned mirror
    plac=1.90, plac_sd=2.30, plac_p05=-1.66, plac_p95=5.80, plac_draws=500,
    seas=3.02, seas_med=2.85, seas_neg=0.43, seas_t=0.83,
    seas_ct=2.74, seas_ct_t=1.10,
    # the costed mirror trade
    sh0=9.08, sh0_t=3.18, sh300=6.08, sh300_t=2.13, sh500=4.08, sh500_t=1.43,
    sh1000=-0.92, sh1000_t=-0.32, breakeven_bps=908, breakeven_3m_bps=1660,
    sh3m_500=2.90, sh3m_500_t=2.39,
    # beta control (market-model CAR, beta fitted on the SEASONED window)
    beta_mean=0.90, beta_med=0.92, beta_min=0.22, beta_max=1.13,
    car_b1=-8.40, car_b1_neg=0.79, car_b1_t=-2.94,
    car_bf=-7.34, car_bf_neg=0.75, car_bf_t=-2.56,
    # universe
    empty_vintages="2017, 2018",
    # synthetic control (machinery proof)
    syn_planted=-10.0, syn_recovered=-9.70, syn_t=-6.70,
    syn_null=0.15, syn_null_t=0.10, syn_null_fire=0,
    syn_lev_b1=3.30, syn_lev_bf=0.80, syn_lev_beta=1.60,
)


HEADER = f"""# Study 931 — The CEF IPO Hole 🕳️

**A closed-end fund IPOs at $20. What is it worth the morning after?**

Not $20. The underwriting syndicate's 3-6% comes out of the proceeds, so the fund starts
life owning less than the subscriber paid — and unlike a company IPO there is no growth
story to reprice, because the fund is a basket of securities anyone can buy directly. The
folk claim: the price holds while the syndicate stabilises it, then slides to the discount
seasoned closed-end funds trade at.

We test it on **{R['n_funds']} US closed-end funds that came to market between 2012 and
2022** ({R['n_vintages']} vintages — these are the large launches we could name and that
still trade under their IPO ticker, *not* every closed-end fund that ever came to market),
each raced against **one asset-class benchmark ETF**, on daily **total-return** closes
({R['start']} → {R['end']}, {R['n_days']:,} days), entering at the **offering price the
subscriber pays**.

*Numbers below are the frozen headline (`docs/results.md`, Fingerprint `{R['fp']}`); the
only live cells run the offline **synthetic** control and are labelled as such.
As-of 2026-06-30.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. What you actually buy for $20\n\n"
           "You wire $20 a share. The underwriters keep roughly **$0.90** of it. The fund "
           "buys bonds or shares with the remaining **$19.10**. From the first minute, you "
           "own $19.10 of assets and a receipt saying you paid $20 — and there is no "
           "business inside that might grow into the difference. That is the whole idea "
           "this study measures.\n\n"
           "> 🔬 **For the quants:** the load is quoted for context and is *never subtracted* "
           "anywhere in this study. It is the mechanism, not an extra cost — the price tape "
           "already contains its consequences, and subtracting it would double-count."),
        md("## 2. The month of quiet, then the fall\n\n"
           "We compare each fund with an ETF holding the same *kind* of assets — a tech CEF "
           "against a tech ETF, a junk-bond CEF against a junk-bond ETF — so a fund launched "
           "into a bad year isn't blamed for the year."),
        code(
            "R = dict(m1=%r, m1_t=%r, m3=%r, m3_t=%r, m6=%r, m6_t=%r, m12=%r, m12_t=%r,\n"
            "         m12_neg=%r, n_funds=%r)\n"
            "for lab in ('1', '3', '6', '12'):\n"
            "    print('%%3s months after the IPO: %%+6.2f%%%% vs its own asset class   (t = %%+.2f)'\n"
            "          %% (lab, R['m'+lab], R['m'+lab+'_t']))\n"
            "print()\n"
            "print('%%.0f%%%% of the %%d funds are behind their asset class one year in.'\n"
            "      %% (R['m12_neg']*100, R['n_funds']))"
            % (R["m1"], R["m1_t"], R["m3"], R["m3_t"], R["m6"], R["m6_t"],
               R["m12"], R["m12_t"], R["m12_neg"], R["n_funds"])
        ),
        md(f"Month one is **quiet** — {R['m1']:+.2f}%, which is statistical nothing "
           f"(*t* = {R['m1_t']:+.2f}). That is the syndicate holding the price up. Then the "
           f"support ends and the hole opens: **{R['m3']:+.2f}%** by month three, "
           f"**{R['m12']:+.2f}%** by month twelve. Twenty-five of twenty-eight funds are "
           f"behind. This is not a bad year for one sleeve — it happened to tech funds, "
           f"health funds, loan funds, preferred funds and credit funds alike, in "
           f"**{R['n_vintages']} different vintage years**."),
        md("## 3. Seven weeks to a discount\n\n"
           f"We cannot watch the discount itself — daily net asset values aren't on the free "
           f"tape — so we watch the next best thing: the day the fund first falls a given "
           f"distance behind the assets it holds. **{R['ttd5_cross']} of {R['n_funds']}** "
           f"funds fall 5% behind within two years, and the median takes "
           f"**{R['ttd5_med']} trading days** — about seven weeks from the closing bell of "
           f"the IPO. Ten percent behind takes a median {R['ttd10_med']} days.\n\n"
           "> 🔬 **For the quants:** a PROXY for time-to-discount, not the discount. It "
           "conflates the price/NAV gap with any tracking difference between the fund's "
           "portfolio and the benchmark ETF — which is why the *level* is indicative and "
           "only the *timing* pattern is taken seriously."),
        md("## 4. Is it the IPO, or are these just bad funds?\n\n"
           f"The obvious objection. So we ran the identical measurement on the identical "
           f"funds from **fake** start dates drawn at random from their seasoned life. The "
           f"average abnormal return over those fake windows is **{R['plac']:+.2f}%** "
           f"(sd {R['plac_sd']:.2f}%) — nothing like {R['m12']:+.2f}%. And measured from "
           f"their first birthday to their third, the same funds show **{R['seas']:+.2f}%** "
           f"(*t* = {R['seas_t']:+.2f}): the bleeding simply stops.\n\n"
           f"The damage lives **in the IPO window**. These are not permanently bad assets — "
           f"they are permanently bad *purchases at issue*."),
        md("## 5. So can you make money from it?\n\n"
           f"The trade you would want is to short the new fund and buy the benchmark. On "
           f"paper it pays **{R['sh0']:+.2f}%** over twelve months. In practice you have to "
           f"borrow the shares, and the trade breaks even at about "
           f"**{R['breakeven_bps']} bps a year** of borrow cost — while a closed-end fund in "
           f"the weeks after its IPO is the hardest thing on the market to borrow: the whole "
           f"issue is sitting in the retail accounts the syndicate placed it in, and almost "
           f"none of it is lendable.\n\n"
           f"There is no long version either. You cannot buy a hole. What is left is one "
           f"instruction, and it is free: **never subscribe to a closed-end fund IPO — wait a "
           f"year and buy it in the market**, most likely at a discount, from the person who "
           f"did subscribe."),
        md("## 6. Live check — is the measuring stick straight? *(synthetic, not the real tape)*\n\n"
           "Before believing a −10% result, check that the estimator finds a hole when one is "
           "planted and finds nothing when none is. Both worlds below are **simulated** — no "
           "real fund appears in this cell."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from cef_ipo import data, strategy as st\n"
            "planted, truth = data.synthetic_panel(signal_strength=1.0, seed=931)\n"
            "null, _ = data.synthetic_panel(signal_strength=0.0, seed=931)\n"
            "p = st.synthetic_detect(planted)\n"
            "n = st.synthetic_detect(null)\n"
            "print('SYNTHETIC world with a planted %+.0f%% first-year hole -> measured %+.2f%% (t = %+.2f)'\n"
            "      % (-truth['planted_slide']*100, p['mean_abn_pct'], p['tstat']))\n"
            "print('SYNTHETIC world with no hole at all               -> measured %+.2f%% (t = %+.2f)'\n"
            "      % (n['mean_abn_pct'], n['tstat']))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Real.** New closed-end funds lose ground to their own asset class: "
           f"**{R['m3']:+.2f}%** by month three (*t* = **{R['m3_t']:+.2f}**) and about "
           f"**{R['m12']:+.2f}%** by month twelve, in both vintage halves, under every "
           f"benchmark choice, not because the funds are leveraged, and only inside the IPO "
           f"window. Honest caveats: at twelve months the most conservative test does not "
           f"clear (so treat 10% as the size, three months as the proof); the 28 names are "
           f"the big launches we could name, not every closed-end fund that ever IPO'd; and "
           f"they are funds that still exist under their IPO ticker (the worst ones get "
           f"merged away, which makes the hole look *smaller*).\n"
           f"- **Tradability — Mirage.** The short that would monetise it cannot be borrowed "
           f"at a price that leaves anything (break-even ~{R['breakeven_bps']} bps/yr of "
           f"borrow). The finding is worth exactly one behaviour — wait — and behaviour "
           f"doesn't pay a coupon."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 931 — The CEF IPO Hole — the teardown\n\n"
           "The abnormal-return table, the vintage-cluster bootstrap, the calendar-time "
           "portfolio with its Newey-West *t*, the pseudo-IPO placebo, the seasoned mirror, "
           "the era and mapping sweeps, the borrow-swept short, and a live synthetic control. "
           "Every real number is frozen from `docs/results.md` (Fingerprint `%s`); the only "
           "live cell is explicitly synthetic.\n\n"
           "> 💡 **In plain words:** we are asking whether a brand-new closed-end fund "
           "under-performs the very assets it holds, how sure we can be given that only 28 "
           "funds and 9 vintages exist, and whether anyone could get paid for knowing it."
           % R["fp"]),
        code("R = %r" % (R,)),
        md("## Design\n\n"
           "- **Sample.** 28 US CEF IPOs, vintages 2012-2022, one asset-class benchmark ETF "
           "each (XLK / XLV / IGF / AOR / BKLN / HYG / LQD / PFF).\n"
           "- **Entry.** The offering price ($20; $25 for the four preferred funds) — an "
           "ASSUMPTION, corroborated by the tape for %d/28 funds within 1%% (worst %+.1f%%, "
           "RIV, whose Yahoo history starts ~2 weeks late — a bias *against* the effect) and "
           "swept against a day-one-close entry.\n"
           "- **Both legs total return** (`auto_adjust=True`). A price-only comparison would "
           "manufacture a fake decay out of an 8-12%% distribution rate.\n"
           "- **One execution lag, once:** the event study's entry *is* the IPO, so the lag "
           "lives in the calendar-time portfolio and the mirror trade, both entered at the "
           "**t+1** close.\n"
           "- **The 3-6%% load is never subtracted** — it is the mechanism; the tape contains "
           "it. Subtracting it would double-count.\n"
           "- **No excess-of-cash adjustment**, on purpose: every number is a difference of "
           "two fully invested legs or a self-financing spread, so the cash rate cancels "
           "identically on both sides. BIL is cached and deliberately unread.\n"
           "- **Universe:** a hand-built **convenience sample** of 28 large, still-listed "
           "launches — *not* a census. Several hundred CEFs IPO'd in the window, no free "
           "listing-date screen exists, and the **%s vintages are empty**. "
           "Representativeness cannot be checked from inside this study.\n"
           "- **Survivorship:** surviving tickers only; BIGZ (2021) left the tape by ETF "
           "conversion and XFLT (2017) was dropped for corrupt Yahoo split factors. Worse "
           "funds disappear, so the measured hole is biased *toward zero*."
           % (R["offer_ok"], R["worst_gap"], R["empty_vintages"])),
        md("## The event study"),
        code(
            "print(f\"{'h':>4s} {'mean':>8s} {'median':>8s} {'neg':>6s} {'t':>7s}\")\n"
            "for lab in ('1','3','6','12'):\n"
            "    print(f\"{lab+'m':>4s} {R['m'+lab]:+8.2f} {R['m'+lab+'_med']:+8.2f} \"\n"
            "          f\"{R['m'+lab+'_neg']:6.0%} {R['m'+lab+'_t']:+7.2f}\")\n"
            "print(f\"\\n12m share-negative Wilson 95% CI: \"\n"
            "      f\"[{R['m12_wilson_lo']:.2f}, {R['m12_wilson_hi']:.2f}]\")\n"
            "print('month 1 is the syndicate stabilisation window; the hole opens in months 2-6')"
        ),
        md("## Dependence-robust inference\n\n"
           "Twenty-eight overlapping event windows across nine vintages are not 28 "
           "independent draws — the six 2021-vintage funds all met the 2022 rate shock "
           "together. Two answers.\n\n"
           "**(a) Vintage-cluster bootstrap** — resample whole IPO *years* with replacement.\n\n"
           "**(b) Calendar-time portfolio** (Fama 1998) — collapse the 28 windows into ONE "
           "daily equal-weight long-fund / short-benchmark spread and take its Newey-West "
           "*t*, which assumes nothing about cross-fund independence.\n\n"
           "> 💡 **In plain words:** (a) asks 'what if we'd drawn different market years?', "
           "(b) asks 'what does one honest daily P&L of this idea look like?'"),
        code(
            "print('vintage-cluster bootstrap (%d clusters):' % R['cb_clusters'])\n"
            "for lab, lo, hi in (('3m', R['cb3_lo'], R['cb3_hi']), ('6m', R['cb6_lo'], R['cb6_hi']),\n"
            "                    ('12m', R['cb12_lo'], R['cb12_hi'])):\n"
            "    print(f\"  {lab:>4s}: 95% CI [{lo:+6.2f}%, {hi:+6.2f}%]  -> entirely below zero\")\n"
            "print()\n"
            "print('calendar-time portfolio (HAC t):')\n"
            "for lab, ann, t in (('age<=1m', R['ct1_ann'], R['ct1_t']), ('age<=3m', R['ct3_ann'], R['ct3_t']),\n"
            "                    ('age<=6m', R['ct6_ann'], R['ct6_t']), ('age<=12m', R['ct12_ann'], R['ct12_t'])):\n"
            "    flag = '  <- clears' if abs(t) >= 2 else ''\n"
            "    print(f\"  {lab:>8s}: {ann:+8.2f}%/yr  HAC t = {t:+.2f}{flag}\")\n"
            "print(f\"\\nthe 3m book holds {R['ct_book']:.2f} funds on an average day (max {R['ct_book_max']}) \"\n"
            "      f\"-> it equal-weights DAYS, not FUNDS, and throws away the cross-sectional averaging;\")\n"
            "print('this is the most conservative test available, not the most powerful one.')"
        ),
        md(f"The 12-month calendar-time reading (*t* = {R['ct12_t']:+.2f}) is the honest weak "
           f"point of this study and is named on the Signal axis: **at twelve months the most "
           f"conservative test does not clear**, so the 12-month figure is this study's "
           f"*magnitude* and the 3m/6m figures are its *evidence*. It is an efficiency limit "
           f"rather than a contradiction — with staggered IPOs the book holds ~1.4 names, "
           f"equal-weights days rather than funds, and therefore under-weights the crowded "
           f"(and worst-hit) 2019-2021 cohort; equal-weighting by fund over the same windows "
           f"gives {R['ct_byfund']:+.2f}%/yr.\n\n"
           f"> ⚖️ **Audit note.** This test was *tightened* after the build: the book now "
           f"earns ages 2…window, so it is never credited with the IPO-close → next-close "
           f"move nobody entering at t+1 could have held. Every horizon moved further from "
           f"zero (3m −2.69 → {R['ct3_t']:+.2f}, 6m −1.95 → {R['ct6_t']:+.2f}, "
           f"12m −1.45 → {R['ct12_t']:+.2f}). The 12-month row fails on either convention."),
        md("## Is the hole just leverage?\n\n"
           "Subtracting **one** unit of benchmark is an assumption, and closed-end funds are "
           "usually 25-40% leveraged: a levered fund launched at the top of its sleeve would "
           "print a negative 'abnormal' return with no IPO mechanism at all. So we re-run the "
           "event window as a market-model CAR (sum of daily *r*<sub>fund</sub> − β·*r*"
           "<sub>bench</sub> from the t+1 close) with each fund's **own β fitted on its "
           "seasoned life**, months 12-36.\n\n"
           "> ⚠️ **This is a diagnostic, not a rule.** The β comes from data *after* the "
           "window it corrects. Nobody standing at the IPO knows it, and it never touches the "
           "mirror trade."),
        code(
            "print(f\"seasoned beta to own benchmark: mean {R['beta_mean']:.2f} \"\n"
            "      f\"median {R['beta_med']:.2f}  (range {R['beta_min']:.2f}-{R['beta_max']:.2f})\")\n"
            "print(f\"CAR, beta = 1       : {R['car_b1']:+6.2f}%  {R['car_b1_neg']:.0%} negative  \"\n"
            "      f\"t = {R['car_b1_t']:+.2f}\")\n"
            "print(f\"CAR, beta = fitted  : {R['car_bf']:+6.2f}%  {R['car_bf_neg']:.0%} negative  \"\n"
            "      f\"t = {R['car_bf_t']:+.2f}\")\n"
            "print('\\n-> mean beta is 0.90, not 1.4: against these benchmarks the funds are not')\n"
            "print('   high-beta, and adjusting for beta removes ~a fifth of the hole and none')\n"
            "print('   of its significance. The hole is not leverage.')"
        ),
        md("## Is it the IPO window, or the funds?\n\n"
           "The placebo re-runs the identical 12-month estimator on the identical funds and "
           "benchmarks from **random seasoned start dates**. If these were simply bad assets, "
           "it would reproduce the hole."),
        code(
            "print(f\"real IPO window       : {R['m12']:+6.2f}%  (t = {R['m12_t']:+.2f})\")\n"
            "print(f\"pseudo-IPO placebo    : {R['plac']:+6.2f}%  (sd {R['plac_sd']:.2f}%, \"\n"
            "      f\"5-95pct [{R['plac_p05']:+.2f}%, {R['plac_p95']:+.2f}%], {R['plac_draws']} draws)\")\n"
            "print(f\"same funds, months 12-36: {R['seas']:+6.2f}%  (t = {R['seas_t']:+.2f}, \"\n"
            "      f\"{R['seas_neg']:.0%} negative)\")\n"
            "print(f\"  calendar-time version : {R['seas_ct']:+6.2f}%/yr (HAC t = {R['seas_ct_t']:+.2f})\")\n"
            "print('\\n-> the damage is concentrated in the IPO window and stops once seasoned.')"
        ),
        md("## Robustness — era cut, benchmark mapping, entry convention"),
        code(
            "print(f\"vintages 2012-2016 (n={R['era_e_n']}): {R['era_e']:+6.2f}%  \"\n"
            "      f\"{R['era_e_neg']:.0%} negative  t={R['era_e_t']:+.2f}\")\n"
            "print(f\"vintages 2019-2022 (n={R['era_l_n']}): {R['era_l']:+6.2f}%  \"\n"
            "      f\"{R['era_l_neg']:.0%} negative  t={R['era_l_t']:+.2f}\")\n"
            "print()\n"
            "print(f\"benchmark = asset class : {R['m12']:+6.2f}%  t={R['m12_t']:+.2f}  (headline)\")\n"
            "print(f\"benchmark = SPY         : {R['map_spy']:+6.2f}%  t={R['map_spy_t']:+.2f}\")\n"
            "print(f\"benchmark = AOR (60/40) : {R['map_aor']:+6.2f}%  t={R['map_aor_t']:+.2f}\")\n"
            "print()\n"
            "print(f\"entry = offering price  : {R['m12']:+6.2f}%  t={R['m12_t']:+.2f}\")\n"
            "print(f\"entry = day-one close   : {R['entry_close']:+6.2f}%  t={R['entry_close_t']:+.2f}\")\n"
            "print('\\n-> entering at the day-one close recovers only 0.7pp of the 10.3pp hole:')\n"
            "print('   the damage is post-listing drift, not the day-one gap.')"
        ),
        md("## Time-to-discount PROXY"),
        code(
            "for thr, med in ((3, R['ttd3_med']), (5, R['ttd5_med']), (10, R['ttd10_med'])):\n"
            "    print(f\"first day {thr:2d}% behind the asset class: median {med:3d} trading days\")\n"
            "print(f\"  (-5% threshold: {R['ttd5_cross']}/{R['n_funds']} funds cross within 2 years, \"\n"
            "      f\"IQR [{R['ttd5_q1']}, {R['ttd5_q3']}] days)\")\n"
            "print('\\nPROXY: daily NAV is not on the free tape, so this conflates the price/NAV')\n"
            "print('discount with fund-vs-ETF tracking difference. The timing pattern is the read.')"
        ),
        md("## Could you trade it? Short the new issue, long the benchmark\n\n"
           "Equal notional, opened at the **t+1** close, 20 bps one-way × NAV on each of four "
           "legs (two on entry, two on exit), borrow charged pro-rata on the short leg.\n\n"
           "> 💡 **In plain words:** the edge is real but it lives on the short side, and new "
           "closed-end funds are among the least borrowable securities in existence."),
        code(
            "print(f\"{'borrow':>10s} {'net 12m':>9s} {'t':>7s}\")\n"
            "for bo, net, t in ((0, R['sh0'], R['sh0_t']), (300, R['sh300'], R['sh300_t']),\n"
            "                   (500, R['sh500'], R['sh500_t']), (1000, R['sh1000'], R['sh1000_t'])):\n"
            "    print(f\"{bo:8d}bp {net:+9.2f}% {t:+7.2f}\")\n"
            "print(f\"\\nbreak-even borrow: ~{R['breakeven_bps']} bps/yr at 12m, \"\n"
            "      f\"~{R['breakeven_3m_bps']} bps/yr at 3m\")\n"
            "print(f\"3m version at 500bp borrow: {R['sh3m_500']:+.2f}% (t = {R['sh3m_500_t']:+.2f})\")\n"
            "print('\\nA CEF weeks after its IPO has essentially no lendable float: the whole issue')\n"
            "print('sits in the retail accounts the syndicate placed it into. Paper trade.')"
        ),
        md("## Live synthetic control — the estimator is unbiased *(synthetic, not the real tape)*\n\n"
           "Planted panel: 28 staggered synthetic IPOs each carrying a −10% first-year slide, "
           "which the estimator must recover. Null panel: the same generator with nothing "
           "planted, on which it must stay quiet across seeds. **No real fund appears below.**"),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from cef_ipo import data, strategy as st\n"
            "planted, truth = data.synthetic_panel(signal_strength=1.0, seed=931)\n"
            "p = st.synthetic_detect(planted)\n"
            "print('SYNTHETIC planted %+.1f%%: recovered %+.2f%% (t = %+.2f, %.0f%% of funds negative)'\n"
            "      % (-truth['planted_slide']*100, p['mean_abn_pct'], p['tstat'], p['share_negative']*100))\n"
            "ts = np.array([st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=931+s)[0])['tstat']\n"
            "               for s in range(8)])\n"
            "print('SYNTHETIC null x8 seeds : t mean %+.2f (sd %.2f), |t|>=2 in %d/8'\n"
            "      % (ts.mean(), ts.std(ddof=1), int((np.abs(ts) >= 2).sum())))\n"
            "half = st.synthetic_detect(data.synthetic_panel(signal_strength=0.5, seed=931)[0])\n"
            "print('SYNTHETIC dose-response : half-strength %+.2f%% vs full %+.2f%%'\n"
            "      % (half['mean_abn_pct'], p['mean_abn_pct']))\n"
            "print('SYNTHETIC levered null  : beta %.1f planted and NO hole -> beta=1 CAR %+.2f%% '\n"
            "      '(biased), beta-fitted CAR %+.2f%% (unbiased)'\n"
            "      % (R['syn_lev_beta'], R['syn_lev_b1'], R['syn_lev_bf']))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Real.** The stamp rests on the horizons where *every* estimator "
           f"agrees: {R['m3']:+.2f}% by month three (one-sample *t* = **{R['m3_t']:+.2f}**, "
           f"calendar-time HAC *t* = **{R['ct3_t']:+.2f}**, vintage-cluster CI "
           f"[{R['cb3_lo']:+.2f}%, {R['cb3_hi']:+.2f}%]) and {R['m6']:+.2f}% by month six "
           f"(*t* = {R['m6_t']:+.2f}, HAC *t* = **{R['ct6_t']:+.2f}**). The twelve-month "
           f"magnitude is {R['m12']:+.2f}% ({R['m12_neg']:.0%} of funds negative, Wilson "
           f"[{R['m12_wilson_lo']:.2f}, {R['m12_wilson_hi']:.2f}]). Both vintage halves clear "
           f"|*t*| ≥ 2 alone ({R['era_e_t']:+.2f} / {R['era_l_t']:+.2f}); all three benchmark "
           f"mappings reproduce it ({R['map_spy']:+.2f}% to {R['map_aor']:+.2f}%); each "
           f"fund's own fitted beta removes only a fifth of it ({R['car_b1']:+.2f}% → "
           f"{R['car_bf']:+.2f}%, *t* = {R['car_bf_t']:+.2f}); the pseudo-IPO placebo reads "
           f"{R['plac']:+.2f}% and the seasoned window {R['seas']:+.2f}% "
           f"(*t* = {R['seas_t']:+.2f}), so the damage is IPO-specific. The synthetic control "
           f"recovers a planted −10% as {R['syn_recovered']:+.2f}% (*t* = {R['syn_t']:+.2f}) "
           f"and fires on {R['syn_null_fire']}/8 null seeds. **Named on this axis:** at "
           f"**12 months the calendar-time HAC *t* is {R['ct12_t']:+.2f} and does not clear** "
           f"on a ~{R['ct_book']:.1f}-name book — the 12m number is the magnitude, not the "
           f"evidence; the **universe is a hand-built convenience sample of 28 large "
           f"launches, not a census** ({R['empty_vintages']} contribute nothing); "
           f"survivorship (surviving tickers only, BIGZ converted away, XFLT dropped for "
           f"corrupt splits) biases the hole toward zero; 28 funds across 9 vintages; the "
           f"offering price, the benchmark mapping and beta = 1 are assumptions, all "
           f"corroborated and swept.\n"
           f"- **Tradability — Mirage.** {R['sh0']:+.2f}% over twelve months at zero borrow, "
           f"but break-even borrow is ~{R['breakeven_bps']} bps/yr and a newly issued CEF has "
           f"no lendable float; there is no long expression of a hole. The residual is an "
           f"abstention rule — do not subscribe, buy seasoned — which avoids a loss and banks "
           f"nothing."),
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
