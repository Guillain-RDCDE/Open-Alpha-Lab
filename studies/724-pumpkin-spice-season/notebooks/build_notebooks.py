"""Build the two narrative notebooks for Study 724 (Pumpkin-Spice-Season).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace 01_for_the_curious.ipynb 02_for_the_quants.ipynb

Figures inline from the cached real run (_cache/pumpkin_spice_season.parquet, built by
examples/verify.py --fetch). Every code cell falls back to the synthetic tape OFFLINE and banners
which tape it is showing. Real headline numbers live ONLY in RESULTS below, mirroring docs/results.md."""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Real numbers from the fingerprinted run (docs/results.md, fp 1d50b4e1e153) ───────────────────
RESULTS = {
    "date_start": "1993-02-28",
    "date_end": "2026-05-31",
    "n_months": 400,
    "fingerprint": "1d50b4e1e153",
    "season_mean": 1.25,      # Aug–Nov excess (SBUX−SPY), %/mo
    "off_mean": 0.69,         # Dec–Jul excess, %/mo
    "spread": 0.56,           # season − off, %/mo
    "spread_t": 0.65,         # Welch
    "ci_lo": -0.84,
    "ci_hi": 1.92,
    "best_month": "Mar",      # largest per-month excess t — off-thesis
    "best_month_t": 1.71,     # HAC
    "aug_t": 0.54,            # the PSL-launch month itself: weakest of the four season months
    "psl_rank": 2,            # PSL window's rank among the 12 four-month windows (1 = strongest)
    "top_window": "Mar-Apr-May-Jun",
    "top_window_spread": 0.74,
    "rot_cagr": 15.1,
    "rot_sharpe": 0.66,
    "rot_net_sharpe": 0.65,
    "rot_maxdd": -64,
    "spy_cagr": 10.9,
    "spy_sharpe": 0.61,
    "sbux_cagr": 17.6,
    "sbux_sharpe": 0.58,
    "sbux_maxdd": -76,
    "timer_sharpe": 0.27,     # market-neutral seasonal pair (Aug–Nov only)
    "timer_net_sharpe": 0.26,
    "timer_cagr": 5.6,
    "allyear_spread_sharpe": 0.35,  # always-hold SBUX−SPY pair (the season DILUTES this)
    "allyear_mean": 0.87,     # %/mo
    "sub_early": -0.69,       # 1993–2009 season−off spread, %
    "sub_early_t": -0.47,
    "sub_late": 1.84,         # 2010-on season−off spread, %
    "sub_late_t": 2.06,
    "basket_months": 244,
    "basket_spread": 0.43,
    "basket_t": 0.71,
    "basket_fp": "624b662c1d9b",
}

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath("../../.."))  # repo root (quantlab/)
sys.path.insert(0, os.path.abspath(".."))        # study package (pumpkin_spice_season/)
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 5.5); plt.rcParams["axes.grid"] = True
import numpy as np, pandas as pd
from pumpkin_spice_season import data, strategy as st

# Cache-first real tape; fall back to the synthetic control OFFLINE and banner which tape we show.
try:
    from quantlab import repro
    d = repro.as_of(data.fetch_data())   # cache-first (examples/verify.py --fetch)
    if d.empty:
        raise RuntimeError("cache miss")
    TAPE = "REAL  (SBUX/SPY total-return monthly, 1993-02..2026-05, 400 months, fp 1d50b4e1e153)"
except Exception as e:
    d, _ = data.synthetic_world(psl_premium=0.0, seed=724)  # the NULL synthetic world
    TAPE = f"SYNTHETIC NULL control (offline fallback: {e}) -- NOT the real tape"
print("TAPE:", TAPE)
rf     = d["tbill"]
excess = d["excess"]                             # SBUX - SPY, the object of study
ms     = st.month_stats(excess)
se     = st.season_tstat(excess)
rot    = st.seasonal_rotation(d["sbux"], d["spy"])
rot_net= st.apply_costs(rot, n_trades_per_year=2, cost_bps_one_way=5)
timer  = st.spread_timer(excess, tbill=rf)       # market-neutral, Aug-Nov only
timer_net = st.apply_costs(timer, n_trades_per_year=4, cost_bps_one_way=5)
bh_spy = st.buy_hold(d["spy"]); bh_sbux = st.buy_hold(d["sbux"])
"""

VERDICT = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Pumpkin premium?: Busted](https://img.shields.io/badge/Pumpkin_premium%3F-Busted-8b949e?style=flat-square)\n\n"
)


def md(t):
    return new_markdown_cell(t)


def code(t):
    return new_code_cell(t)


def build_curious():
    cells = [
        md("# Pumpkin-Spice-Season 🎃\n### The PSL launches in late August — so does Starbucks beat the market into the fall?\n\n"
           + VERDICT +
           "It's the most reliable ritual in American retail: every late August, Starbucks brings back the "
           "Pumpkin Spice Latte, the internet loses its mind, and \"pumpkin spice season\" runs from the PSL "
           "launch through Thanksgiving. Surely all that autumn foot-traffic shows up in the stock — buy SBUX "
           "in August, ride the pumpkin-spice quarter, beat the market? We checked 33 years of Starbucks "
           "returns *relative to the S&P 500*. The answer is no — and the way it fails is a little lesson in "
           "how a great stock fools you into seeing a calendar.\n\n"
           "> **Plain-language layer.** The HAC t-stats, the window placebo and the rotation math are in "
           "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n>\n"
           "> ⚠️ Not investment advice. Every chart is generated by the code beside it; the cell banners "
           "say whether it is the real SBUX/SPY tape or the offline synthetic control."),
        code(BOOT),
        md("## The answer first 🎯\n\n"
           "| The claim | The data (SBUX − SPY, 1993–2026, 400 months) |\n|---|---|\n"
           "| \"Pumpkin-spice season beats the market\" | Aug–Nov excess **+1.25%/mo** vs off-season **+0.69%** "
           "— a spread of **+0.56%, t = 0.65**, CI **[−0.84%, +1.92%]**. A coin flip. |\n"
           "| \"August is the launch, so it's the month\" | The PSL-launch month August is the **weakest** of "
           "the four season months (t = 0.54); the biggest month is *March* (off-thesis). |\n"
           "| \"Aug–Nov is the special window\" | Among all twelve 4-month windows, Aug–Nov ranks **#2** — an "
           "off-thesis **Mar–Jun** window is stronger. Nothing picks out the pumpkin months. |"),
        md("## 1 · The claim\n\nStarbucks launched the Pumpkin Spice Latte in 2003; it's now the company's "
           "most popular seasonal drink ever, with the launch date pulled earlier and earlier (late August in "
           "recent years) to front-run the craving. \"Pumpkin spice season\" is a genuine cultural and "
           "revenue event. So the folklore writes itself: **be long Starbucks from the August PSL launch "
           "through the Aug–Nov pumpkin quarter, and you'll beat the market.**"),
        md("## 2 · So what?\n\nIf a soft-drink marketing calendar reliably moved the stock *relative to the "
           "market*, you'd have a free lunch: rotate into SBUX for four months, back to an index fund the rest "
           "of the year, and collect a seasonal premium with no forecasting. The catch is the word *reliably*. "
           "Starbucks has been a spectacular stock for 30 years — so the real question isn't \"does SBUX go up "
           "in the fall\" (it goes up most of the time), it's **\"does its market-beating happen *especially* "
           "in pumpkin-spice season?\"**"),
        md("## 3 · How would we even know?\n\nWe study SBUX **minus SPY** — the market-relative return, "
           "because \"beats the market\" is the claim. We line up every August, every September, … and ask "
           "whether the Aug–Nov excess is reliably bigger than the rest of the year (with an "
           "autocorrelation-robust t-stat and an honest confidence interval). Then the placebo: we slide a "
           "four-month window around the whole calendar and see whether Aug–Nov is *special* or just one slice "
           "among twelve. Finally we build the obvious rotation and race it. **If the season's confidence "
           "interval straddles zero and Aug–Nov isn't the standout window, it's a mirage.**"),
        code("month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']\n"
             "means = [ms.loc[m,'mean']*100 for m in range(1,13)]\n"
             "tstats = [ms.loc[m,'tstat'] for m in range(1,13)]\n"
             "season=[8,9,10,11]\n"
             "colors = ['#c0392b' if abs(tstats[m-1])>=2 else ('#ff8c1a' if m in season else '#8b949e') for m in range(1,13)]\n"
             "fig,ax = plt.subplots(figsize=(10,4))\n"
             "ax.bar(month_names, means, color=colors)\n"
             "ax.axhline(0, color='black', lw=0.8)\n"
             "ax.set_ylabel('Mean SBUX-minus-SPY excess (%)')\n"
             "ax.set_title(f'Per-month SBUX excess over SPY (orange = pumpkin-spice season Aug-Nov)\\nTAPE: {TAPE}')\n"
             "plt.tight_layout(); plt.show()\n"
             "print(f\"Season Aug-Nov: {se['season_mean']*100:+.2f}%/mo  (n={se['n_season']})\")\n"
             "print(f\"Off-season:     {se['off_mean']*100:+.2f}%/mo  (n={se['n_off']})\")\n"
             "print(f\"Season - Off spread: {se['spread']*100:+.2f}%  Welch t={se['tstat']:.2f}\")"),
        md("## 4 · The teardown — let's actually look\n\nLook at the orange (pumpkin-spice) bars. They're not "
           "taller than the grey ones — and August, the launch month everyone points to, is one of the "
           "*smallest*. The season-minus-off spread is **+0.56%/month with t = 0.65**: indistinguishable from "
           "zero, with a confidence interval running **−0.84% to +1.92%**. SBUX beats the market in plenty of "
           "months; it just doesn't do it *especially* when the pumpkin drinks are out."),
        code("wp = st.window_placebo(excess)\n"
             "disp = wp.copy(); disp['spread'] = (disp['spread']*100).round(2)\n"
             "disp['label'] = np.where(disp['is_psl'], disp['months']+'  <-- PSL', disp['months'])\n"
             "print(f'TAPE: {TAPE}\\n')\n"
             "print('All twelve 4-month windows, excess spread vs rest-of-year (best first):')\n"
             "for rk,row in disp.iterrows():\n"
             "    print(f\"  #{rk+1}: {row['label']:22s} spread {row['spread']:+.2f}%  t={row['tstat']:+.2f}\")\n"
             "psl_rank = int(wp[wp['is_psl']].index[0]) + 1\n"
             "print(f'\\nPumpkin-spice window (Aug-Nov) ranks #{psl_rank} of {len(wp)} -- an off-thesis window wins.')"),
        md("**Read it plainly.** If pumpkin-spice season were a real market edge, the Aug–Nov window would sit "
           "at the top of that list. It's #2 — beaten by a spring window (Mar–Jun) that has nothing to do with "
           "pumpkins. The \"season\" is a story we drew around one ordinary quarter of a great stock's year."),
        code("res = {\n"
             "    'seasonal rotation (long SBUX Aug-Nov, else SPY)': st.summary(rot, rf=rf),\n"
             "    'rotation net 5bp':                                st.summary(rot_net, rf=rf),\n"
             "    'buy & hold SPY':                                  st.summary(bh_spy, rf=rf),\n"
             "    'buy & hold SBUX':                                 st.summary(bh_sbux, rf=rf),\n"
             "}\n"
             "display(pd.DataFrame(res).T[['cagr','sharpe','max_drawdown']].round(3))\n"
             "print('Sharpe = excess of T-bill; the rotation only *looks* to edge SPY because it borrows')\n"
             "print('SBUX beta for 4 months -- and a DIFFERENT 4 months (Mar-Jun) would have been better.')"),
        md("## 5 · The verdict\n\n**Signal: None** — the season-vs-off excess spread is t = 0.65 with a CI "
           "that straddles zero; no month clears |t| ≥ 2, and the PSL window isn't even the best 4-month "
           "slice. **Tradability: Mirage** — the market-neutral \"be in the pair only for the season\" trade "
           "earns Sharpe 0.27, *less* than just holding the SBUX-vs-SPY pair all year (0.35). The season "
           "*dilutes* the edge, it doesn't concentrate it. **\"Pumpkin premium\"? Busted** — the spread is "
           "*negative* in 1993–2009 and only turns up in a snooped 2010-on half."),
        md("## 6 · Could you actually trade it?\n\nNot as a *seasonal* edge. The long-only rotation's headline "
           "Sharpe (0.66 vs SPY's 0.61) is a magic trick: it substitutes a higher-returning stock into four "
           "months of the year. But (a) you had to *know in advance* that SBUX would be a 30-year winner "
           "(survivorship — most \"buy the beloved brand\" bets don't end up as Starbucks), and (b) the pumpkin "
           "months aren't even the right four. Strip out the borrowed SBUX beta and the pure seasonal signal "
           "is worth nothing."),
        md("## 7 · Going further 🚪\n\nForks: (a) test the calendar on **fundamentals** — do SBUX's autumn-"
           "quarter same-store sales actually beat its other quarters, or is even *that* a myth? (b) a "
           "**QSR-basket** version (SBUX/MCD/YUM/CMG) to see if any \"autumn consumer\" pop is broader than one "
           "name (spoiler in the quants notebook: t = 0.71); (c) compare to "
           "[307 Coffee-Seasonality](../../307-coffee-seasonality/) — the *commodity* calendar behind the cup, "
           "another None/Mirage."),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "01_for_the_curious.ipynb")


def build_quants():
    cells = [
        md("# Pumpkin-Spice-Season — a quantitative teardown 🔬\n"
           "### Per-month HAC t-stats on SBUX−SPY · season-vs-off Welch spread + block-bootstrap CI · a 12-window placebo · rotation race · sub-period split\n\n"
           + VERDICT +
           "The deep companion to [the notebook for the curious](01_for_the_curious.ipynb) — *same seven "
           "beats, every claim now carrying its standard error.* We test whether Starbucks' market-relative "
           "return concentrates in the Aug–Nov pumpkin-spice window across 400 months and find no signal, a "
           "window placebo that an off-thesis quarter wins, a sign-flipping sub-period, and a rotation whose "
           "only edge is borrowed single-name beta.\n\n"
           "> ⚠️ Not investment advice. SBUX and SPY **total-return** (dividends reinvested), 13-week T-bill "
           "(^IRX) the cash leg, monthly 1993-02 → 2026-05, 400 months (Yahoo Finance, daily closes resampled "
           "to month-end, grid asserted hole-free). The object of study is the **excess** SBUX − SPY. Offline, "
           "every cell falls back to the synthetic NULL and banners the tape. Sources in "
           "[`docs/references.md`](../docs/references.md).\n>\n"
           "> 💡 The `💡 In plain words` notes translate each result back into intuition."),
        code(BOOT),
        md("## Verdict, up front\n\n| Axis | Stamp | Why |\n|---|---|---|\n"
           "| Signal | **None** | season−off excess spread t = 0.65, 95% CI [−0.84%, +1.92%]; no month \\|t\\| ≥ 2; PSL window ranks #2 of 12 |\n"
           "| Tradability | **Mirage** | market-neutral seasonal pair Sharpe 0.27 < always-hold pair 0.35 — the season *dilutes* the edge |\n"
           "| Pumpkin premium? | **Busted** | −0.69% in 1993–2009 (t = −0.47), only +1.84% in a snooped 2010-on half (t = 2.06) |\n\n"
           "> 💡 In plain words: Starbucks beats the market in lots of months. It does **not** do so "
           "*especially* in pumpkin-spice season — the vivid calendar story leaves no statistical fingerprint."),
        md("## 1 · The claim, steelmanned\n\n"
           "- **H₁:** at least one pumpkin-spice-season month (Aug–Nov) has a significantly positive SBUX−SPY "
           "excess.\n"
           "- **H₂:** the season group (Aug–Nov) has a significantly higher mean excess than the off-season — "
           "Welch t-test on pooled groups.\n"
           "- **H₃:** the season-minus-off spread's block-bootstrap 95% CI excludes zero.\n"
           "- **H₄:** the pumpkin-spice window is the *strongest* four-month window (placebo — it should stand "
           "out among all twelve).\n"
           "- **H₅:** a long-SBUX-in-season / SPY-otherwise rotation beats buy-and-hold *after* accounting for "
           "the fact that it's just borrowing SBUX beta, and the pattern is stable across sub-periods."),
        md("## 2 · So what? — what rides on each\n\nIf H₁–H₅ hold, a marketing calendar prints market-relative "
           "alpha and you harvest it with a fixed-month rotation. If they fail, \"pumpkin-spice season\" is "
           "**single-name survivorship dressed as a calendar**: SBUX was a great stock, so *any* slice of its "
           "year looks good, and we remember the slice with the cultural story attached."),
        md("## 3 · How we'd know — the protocol\n\nOne-sample t-stats (naive **and** Newey-West HAC) for each "
           "of the 12 calendar months of SBUX−SPY vs 0 (Bonferroni threshold |t| ≈ 3 for α = 0.05/12); a Welch "
           "two-sample test for season (Aug–Nov) vs off-season; a circular block-bootstrap (12-month blocks) "
           "95% CI on the spread; a **12-window placebo** sliding a 4-month window around the calendar; the "
           "rotation (long SBUX Aug–Nov, SPY otherwise — calendar-known, **no execution lag**) and a "
           "market-neutral spread timer, Sharpe in **excess of the T-bill**, gross and net of 5 bp/leg; and a "
           "1993–2009 / 2010-on sub-period split. **Survivorship is on the Signal axis:** SBUX is one "
           "hand-picked survivor, so we read the raw-return win with maximum suspicion and lean on the "
           "market-relative and placebo tests."),
        md("## 4 · The teardown"),
        md("### 4.1 Per-month excess t-stats, naive and HAC (H₁)"),
        code("month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']\n"
             "print(f'TAPE: {TAPE}\\n')\n"
             "print(f'{\"Month\":6s}  {\"Mean\":>8s}  {\"t-naive\":>8s}  {\"t-HAC\":>8s}  {\"n\":>4s}  Signal?')\n"
             "for m in range(1,13):\n"
             "    row = ms.loc[m]\n"
             "    tag = ' (PSL)' if m in [8,9,10,11] else ''\n"
             "    sig = '|t|>=3 (Bonf.)' if abs(row['tstat_hac'])>=3 else ('|t|>=2' if abs(row['tstat_hac'])>=2 else 'noise')\n"
             "    print(f'{month_names[m-1]+tag:6s}  {row[\"mean\"]*100:+7.2f}%  {row[\"tstat\"]:+8.2f}  '\n"
             "          f'{row[\"tstat_hac\"]:+8.2f}  {int(row[\"n\"]):4d}  {sig}')"),
        md("> 💡 In plain words: not one month clears |t| ≥ 2 on either statistic. The largest excess "
           "(March, t ≈ 1.71 HAC) is off-thesis, and the PSL-launch month August is one of the *weakest* season "
           "months (t = 0.54). **H₁ rejected.**"),
        md("### 4.2 Season vs off-season spread + block-bootstrap CI (H₂, H₃)"),
        code("print(f'Season (Aug-Nov): {se[\"season_mean\"]*100:+.2f}%/mo  n={se[\"n_season\"]}')\n"
             "print(f'Off-season:       {se[\"off_mean\"]*100:+.2f}%/mo  n={se[\"n_off\"]}')\n"
             "print(f'Spread: {se[\"spread\"]*100:+.2f}%/mo  Welch t={se[\"tstat\"]:.2f}')\n"
             "ci = st.spread_bootstrap_ci(excess, n_boot=5000, seed=724)\n"
             "print(f'Block-bootstrap 95% CI on spread: [{ci[\"lo\"]*100:.2f}%, {ci[\"hi\"]*100:.2f}%]  '\n"
             "      f'(point {ci[\"point\"]*100:+.2f}%, n_boot={ci[\"n_boot\"]})')\n"
             "print('CI straddles 0:', ci['lo'] < 0 < ci['hi'])"),
        md("> 💡 In plain words: the Welch t is 0.65 and the bootstrap CI runs roughly −0.8% to +1.9% — it "
           "swamps the 0.56% point estimate. The season spread is noise. **H₂ and H₃ rejected.**"),
        md("### 4.3 The window placebo — is Aug–Nov special? (H₄)"),
        code("wp = st.window_placebo(excess)\n"
             "disp = wp.copy(); disp['spread']=(disp['spread']*100).round(2)\n"
             "for rk,row in disp.iterrows():\n"
             "    flag = '  <-- PUMPKIN SPICE' if row['is_psl'] else ''\n"
             "    print(f\"  #{rk+1}: {row['months']:22s} spread {row['spread']:+.2f}%  t={row['tstat']:+.2f}{flag}\")\n"
             "psl_rank=int(wp[wp['is_psl']].index[0])+1; top=wp.iloc[0]\n"
             "print(f'\\nPSL window ranks #{psl_rank} of {len(wp)}; the strongest is {top[\"months\"]} '\n"
             "      f'({top[\"spread\"]*100:+.2f}%), which has nothing to do with pumpkins.')"),
        md("> 💡 In plain words: a real seasonal edge would put Aug–Nov at the top of the placebo ranking. It "
           "lands **#2**, behind an off-thesis spring window. When the story-window isn't the winning window, "
           "the story is decoration. **H₄ rejected.**"),
        md("### 4.4 Rotation race + the survivorship trap (H₅a)"),
        code("res = {\n"
             "    'seasonal rotation (SBUX Aug-Nov, else SPY)': st.summary(rot, rf=rf),\n"
             "    'rotation net 5bp':                           st.summary(rot_net, rf=rf),\n"
             "    'market-neutral pair, season only':           st.summary(timer, rf=rf),\n"
             "    'market-neutral pair, ALL YEAR':              st.summary(st.buy_hold(excess)),\n"
             "    'buy & hold SPY':                             st.summary(bh_spy, rf=rf),\n"
             "    'buy & hold SBUX':                            st.summary(bh_sbux, rf=rf),\n"
             "}\n"
             "display(pd.DataFrame(res).T[['cagr','sharpe','vol_ann','max_drawdown','n']].round(3))\n"
             "print('The season-only pair (Sharpe ~0.27) is WORSE than holding the pair all year (~0.35):')\n"
             "print('the pumpkin window does not concentrate SBUX alpha, it dilutes it. The rotation only')\n"
             "print('edges SPY by borrowing SBUX beta 4 months/yr -- pure single-name survivorship.')"),
        md("> 💡 In plain words: the market-neutral SBUX-vs-SPY pair earns a *higher* Sharpe when held all year "
           "(0.35) than when held only in pumpkin-spice season (0.27). If the season were where the alpha "
           "lived, restricting to it would *raise* the Sharpe. It lowers it. **H₅a rejected.**"),
        md("### 4.5 Sub-period stability (H₅b)"),
        code("d.index = pd.DatetimeIndex(d.index)\n"
             "for lab, yr in [('1993-2009',(1993,2009)), ('2010-on',(2010,2030))]:\n"
             "    sl = d[(d.index.year>=yr[0]) & (d.index.year<=yr[1])]\n"
             "    r = st.season_tstat(sl['excess'])\n"
             "    print(f'{lab}: season={r[\"season_mean\"]*100:+.2f}%  off={r[\"off_mean\"]*100:+.2f}%  '\n"
             "          f'spread={r[\"spread\"]*100:+.2f}%  Welch t={r[\"tstat\"]:.2f}  n_on={r[\"n_season\"]}')\n"
             "print('\\nThe season spread FLIPS SIGN across halves -> the +2.06 in 2010-on is a snooped split,')\n"
             "print('not a stable law; the full-sample t=0.65 is the honest headline.')"),
        md("> 💡 In plain words: the season spread is *negative* in 1993–2009 (t = −0.47) and only nominally "
           "positive 2010-on (t = 2.06 — but on a split *we chose*, over half the sample, off a full-sample "
           "t = 0.65). Per the desk's inference bar, a snooped, sign-flipping half reads **WEAK at most**, "
           "never REAL. **H₅b rejected.**"),
        md("### 4.6 Robustness — the QSR basket"),
        code("try:\n"
             "    b = repro.as_of(data.fetch_basket())\n"
             "    if b.empty: raise RuntimeError('basket cache miss')\n"
             "    rb = st.season_tstat(b['excess'])\n"
             "    print(f'QSR basket (SBUX/MCD/YUM/CMG) excess over SPY, {b.index.min().date()}..{b.index.max().date()} ({len(b)} months):')\n"
             "    print(f'  season {rb[\"season_mean\"]*100:+.2f}%  off {rb[\"off_mean\"]*100:+.2f}%  '\n"
             "          f'spread {rb[\"spread\"]*100:+.2f}%  Welch t={rb[\"tstat\"]:.2f}')\n"
             "except Exception as e:\n"
             "    print('basket leg needs the cached parquet (examples/verify.py --fetch):', e)"),
        md("> 💡 In plain words: broadening from Starbucks to a coffee/QSR basket doesn't rescue the effect — "
           "the season spread is +0.43%/mo at t = 0.71, still noise. Whatever autumn pop the story imagines is "
           "neither a Starbucks quirk nor a broad QSR law; it isn't there."),
        md("## 5 · The verdict\n\nH₁ rejected (no significant month). H₂/H₃ rejected (spread t = 0.65, CI "
           "[−0.84%, +1.92%]). H₄ rejected (PSL window ranks #2 of 12). H₅a rejected (season-only pair 0.27 < "
           "all-year pair 0.35). H₅b rejected (sign-flipping sub-periods). → Signal `NONE`, Tradability "
           "`MIRAGE`, pumpkin premium `BUSTED`."),
        md("## 6 · Could you trade it?\n\nNo — not as a seasonal edge. The long-only rotation's Sharpe 0.66 vs "
           "SPY's 0.61 is entirely *borrowed SBUX beta* dropped into four months, and it's not even the right "
           "four (Mar–Jun beats it in the placebo). The market-neutral seasonal pair — the honest isolation of "
           "\"the season\" — is Sharpe 0.27 gross, 0.26 net, and *below* the all-year pair. There is no seasonal "
           "alpha to defend, so the break-even-cost question is moot. What's real here is survivorship: you'd "
           "have needed to know, in 1993, that SBUX would be one of the great stocks of the era."),
        md("## 7 · Going further\n\nForks: (a) key the test to SBUX's **autumn-quarter same-store sales** — is "
           "even the *fundamental* pumpkin bump real, or does the fall quarter just look like the rest of a "
           "growth story? (b) an event-study around each year's **PSL launch date** (a few trading days), where "
           "a marketing pop, if any, would actually live — not a four-month window; (c) cross-check against "
           "[307 Coffee-Seasonality](../../307-coffee-seasonality/) (the bean, not the brand) and the "
           "calendar-effect family. Companion caveat: any \"buy the beloved brand\" seasonal is one survivor "
           "away from a data-mined mirage."),
    ]
    _write(new_notebook(cells=cells, metadata=_meta()), "02_for_the_quants.ipynb")


def _meta():
    return {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}


def _write(nb, name):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
