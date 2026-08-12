"""Generate the two narrative notebooks for Study 903 (Sector-Neutral Low-Vol).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily OHLC,
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; trailing-63d volatility,
# long bottom30% / short top30%; raw vs sector-neutral).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, n_rows=4147, n_days=4083,
    fingerprint="357fd262912f",
    # raw (sector-tilted) book
    raw_spread_bps=-4.74, raw_t_nw=-2.61, raw_t_1s=-2.47,
    raw_lo_bps=5.21, raw_hi_bps=9.95, raw_welch=-1.66, raw_sharpe=-0.61,
    raw_lo_sh=0.98, raw_hi_sh=0.98, raw_lo_vol=0.13, raw_hi_vol=0.26,
    raw_long_def=44.7, raw_short_def=5.2, raw_ls_def=39.5, univ_def=20.0,
    # sector-neutral book
    neu_spread_bps=-3.53, neu_t_nw=-2.67, neu_t_1s=-2.52,
    neu_lo_bps=6.54, neu_hi_bps=10.08, neu_welch=-1.26, neu_sharpe=-0.63,
    neu_lo_sh=1.05, neu_hi_sh=1.07, neu_lo_vol=0.16, neu_hi_vol=0.24,
    neu_long_def=13.9, neu_short_def=15.7, neu_ls_def=-1.8,
    delta_spread=1.20,
    # placebo (sector-neutral)
    placebo_obs=-3.53, placebo_mean=-0.005, placebo_sd=0.993,
    placebo_sigma=3.55, placebo_p=1.0000, placebo_draws=1000,
    # eras (sector-neutral)
    era_early_bps=-3.08, era_early_t=-1.93, era_early_n=1949,
    era_late_bps=-3.95, era_late_t=-1.90, era_late_n=2134,
    # timer (sector-neutral)
    timer_1_gross=-3.53, timer_1_cost=2.14, timer_1_net=-5.67, timer_1_t=-4.04,
    timer_1_sh=-1.00, timer_1_ann=-14.3,
    timer_5_gross=-3.53, timer_5_cost=10.14, timer_5_net=-13.67, timer_5_t=-9.74,
    timer_5_sh=-2.42, timer_5_ann=-34.4,
    # synthetic control
    null_mean_t=0.20, null_sd_t=0.64, null_fire=0,
    planted_t=4.23, planted_welch=4.35, planted_edge=0.1,
    confound_raw_t=3.46, confound_raw_fire=18, confound_neu_fire=0,
)


HEADER = f"""# Study 903 — Sector-Neutral Low-Vol 🧮

**Is the low-vol anomaly a real stock-level effect, or just a bet on defensive sectors?**

The low-volatility anomaly (study 330) says calm stocks out-earn wild ones risk-adjusted.
But a naive low-vol sort quietly loads the structurally calm **sectors** — utilities,
staples, health care — and shorts the wild ones (tech, energy). So how much of the "edge"
is a defensive-**sector** bet rather than a stock-level effect? We strip the sector out:
rank each name on trailing volatility **within its own sector** (demean by the sector
median), then long the low-vol / short the high-vol names **sector-neutrally**, on a liquid
US cross-section ({R['start']} → {R['end']}, {R['n_names']} names).

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: current-membership mega-caps — magnitudes are an upper
bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The sector bet, made visible\n\n"
           "Before we touch returns, look at *what the naive low-vol sort actually buys*. "
           "The calmest 30% of names are overwhelmingly the defensive sectors; the wildest "
           "30% are tech and energy. So a raw low-vol book is, mechanically, long-defensive "
           "/ short-cyclical — a sector bet wearing a factor costume."),
        code(
            "R = dict(raw_long_def=%r, raw_short_def=%r, raw_ls_def=%r, univ_def=%r,\n"
            "         neu_long_def=%r, neu_short_def=%r, neu_ls_def=%r)\n"
            "print('RAW low-vol sort   : long book %%.1f%%%% defensive vs short %%.1f%%%% '\n"
            "      '(universe %%.1f%%%%) -> long-short tilt %%+.1f%%%%'\n"
            "      %% (R['raw_long_def'], R['raw_short_def'], R['univ_def'], R['raw_ls_def']))\n"
            "print('SECTOR-NEUTRAL sort: long book %%.1f%%%% defensive vs short %%.1f%%%% '\n"
            "      '-> long-short tilt %%+.1f%%%% (bet removed)'\n"
            "      %% (R['neu_long_def'], R['neu_short_def'], R['neu_ls_def']))"
            % (R["raw_long_def"], R["raw_short_def"], R["raw_ls_def"], R["univ_def"],
               R["neu_long_def"], R["neu_short_def"], R["neu_ls_def"])
        ),
        md("## 2. Does the low-vol edge survive the strip?\n\n"
           "Now the returns. We run the identical long-low-vol / short-high-vol book twice — "
           "once raw, once after demeaning each name's vol within its sector — and compare."),
        code(
            "R = dict(raw_spread_bps=%r, raw_t_nw=%r, neu_spread_bps=%r, neu_t_nw=%r,\n"
            "         raw_lo_sh=%r, raw_hi_sh=%r, neu_lo_sh=%r, neu_hi_sh=%r)\n"
            "print('spread (low-vol minus high-vol):')\n"
            "print('  RAW           : %%+.2f bps/day  (NW t = %%+.2f)' %% (R['raw_spread_bps'], R['raw_t_nw']))\n"
            "print('  SECTOR-NEUTRAL: %%+.2f bps/day  (NW t = %%+.2f)' %% (R['neu_spread_bps'], R['neu_t_nw']))\n"
            "print()\n"
            "print('per-leg Sharpe (the real, risk-adjusted low-vol claim):')\n"
            "print('  RAW           : low-vol %%.2f vs high-vol %%.2f  (a tie)' %% (R['raw_lo_sh'], R['raw_hi_sh']))\n"
            "print('  SECTOR-NEUTRAL: low-vol %%.2f vs high-vol %%.2f  (high-vol edges it)' %% (R['neu_lo_sh'], R['neu_hi_sh']))"
            % (R["raw_spread_bps"], R["raw_t_nw"], R["neu_spread_bps"], R["neu_t_nw"],
               R["raw_lo_sh"], R["raw_hi_sh"], R["neu_lo_sh"], R["neu_hi_sh"])
        ),
        md("## 3. Is the demean real? A live synthetic control\n\n"
           "The whole method rests on the demean genuinely stripping the sector bet. We prove "
           "it in a seeded toy world: plant **only** a defensive-sector premium (no stock-level "
           "effect at all) and watch the *raw* sort fire while the *sector-neutral* sort stays "
           "silent. Then plant a real within-sector low-vol effect and watch the neutral sort "
           "recover it. No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import pandas as pd\n"
            "from sn_lowvol import data, strategy as st\n"
            "secmap = pd.Series(data.synthetic_sectors(40, 8))\n"
            "# only a sector premium (edge=0): raw should fire, neutral should not\n"
            "conf = data.synthetic_panel(edge=0.0, seed=903, n_assets=40, n_days=1500, n_sectors=8, sector_prem_ann=0.08)\n"
            "rc = st.close_returns(conf)\n"
            "raw_t = st.vol_stats(st.vol_spreads(rc, secmap, 63, 0.3, neutral=False))['t_nw']\n"
            "neu_t = st.vol_stats(st.vol_spreads(rc, secmap, 63, 0.3, neutral=True))['t_nw']\n"
            "print('CONFOUND (sector premium only): raw NW t = %+.2f (fooled) -> neutral NW t = %+.2f (silent)' % (raw_t, neu_t))\n"
            "# a genuine within-sector low-vol effect: neutral should recover it\n"
            "pl = data.synthetic_panel(edge=0.1, seed=903, n_assets=40, n_days=1500, n_sectors=8)\n"
            "pt = st.vol_stats(st.vol_spreads(st.close_returns(pl), secmap, 63, 0.3, neutral=True))['t_nw']\n"
            "print('PLANTED stock-level low-vol effect: neutral NW t = %+.2f (recovered)' % pt)"
        ),
        md(f"## 4. The honest verdict — the low-vol edge does *not* survive here\n\n"
           f"On this liquid mega-cap tape the low-vol book is **negative** — the *wild* names "
           f"(tech mega-caps) out-earned the calm ones — both raw (**{R['raw_spread_bps']:+.2f} "
           f"bps/day**) and sector-neutral (**{R['neu_spread_bps']:+.2f} bps/day**, NW *t* = "
           f"**{R['neu_t_nw']:+.2f}**). Stripping the sector bet shifts the spread by only "
           f"**{R['delta_spread']:+.2f} bps** and leaves it significantly *wrong-signed* vs the "
           f"anomaly. Even on the anomaly's own turf — risk-adjusted **Sharpe** — the low-vol "
           f"leg has *no* advantage once sector-neutral ({R['neu_lo_sh']:.2f} vs "
           f"{R['neu_hi_sh']:.2f}). The synthetic control confirms the machinery is sound (a "
           f"pure sector premium fools the raw sort at *t* = {R['confound_raw_t']:+.2f} but the "
           f"neutral sort stays silent), so this is a real null, not a bug. **Signal: None** "
           f"(the low-vol edge is absent — and what character the raw sort had was largely a "
           f"defensive-sector tilt), **Tradability: Mirage** (the book loses money at any cost)."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 903 — Sector-Neutral Low-Vol — the teardown\n\n"
           "The raw-vs-neutral spread splits, the per-leg Sharpe race, the defensive-tilt "
           "diagnostic, the Newey-West spread *t*, the 1,000-permutation placebo, the two-era "
           "cut, the costed timer, and the synthetic control (null + planted + the sector-"
           "confound proof)."),
        code("R = %r" % (R,)),
        md("## The headline — raw vs sector-neutral low-vol spread\n\n"
           "Daily equal-weight bottom-30% minus top-30% trailing-63d-vol spread "
           "(long low-vol, short high-vol), on the same panel."),
        code(
            "print(f\"RAW    spread : {R['raw_spread_bps']:+.2f} bps/day  NW(10) t = {R['raw_t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['raw_t_1s']:+.2f}  (Welch {R['raw_welch']:+.2f})\")\n"
            "print(f\"NEUTRAL spread: {R['neu_spread_bps']:+.2f} bps/day  NW(10) t = {R['neu_t_nw']:+.2f}  \"\n"
            "      f\"one-sample t = {R['neu_t_1s']:+.2f}  (Welch {R['neu_welch']:+.2f})\")\n"
            "print(f\"stripping the sector bet moved the spread {R['delta_spread']:+.2f} bps -> still wrong-signed\")"
        ),
        md("## The per-leg Sharpe race — the *risk-adjusted* low-vol claim\n\n"
           "The anomaly is really about return **per unit of risk**. Each leg's own Sharpe "
           "(≈ excess-of-cash on a daily book), with its annualised vol."),
        code(
            "print(f\"RAW    : low-vol Sharpe {R['raw_lo_sh']:.2f} (vol {R['raw_lo_vol']:.2f}) vs \"\n"
            "      f\"high-vol {R['raw_hi_sh']:.2f} (vol {R['raw_hi_vol']:.2f})  -> a tie\")\n"
            "print(f\"NEUTRAL: low-vol Sharpe {R['neu_lo_sh']:.2f} (vol {R['neu_lo_vol']:.2f}) vs \"\n"
            "      f\"high-vol {R['neu_hi_sh']:.2f} (vol {R['neu_hi_vol']:.2f})  -> high-vol edges it\")"
        ),
        md("## The defensive-sector tilt — what the raw sort actually buys"),
        code(
            "print(f\"RAW    : long book {R['raw_long_def']:.1f}% defensive vs short {R['raw_short_def']:.1f}% \"\n"
            "      f\"(universe {R['univ_def']:.1f}%)  -> long-short tilt {R['raw_ls_def']:+.1f}%\")\n"
            "print(f\"NEUTRAL: long book {R['neu_long_def']:.1f}% defensive vs short {R['neu_short_def']:.1f}% \"\n"
            "      f\"-> long-short tilt {R['neu_ls_def']:+.1f}% (neutralised)\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations, sector-neutral)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> ~{R['placebo_sigma']:.2f} sigma into the LEFT tail \"\n"
            "      f\"(right-tail p = {R['placebo_p']:.4f})\")"
        ),
        md("## Robustness — two eras (split 2018-01-01, sector-neutral)"),
        code(
            "print(f\"2010-2017 (n={R['era_early_n']}): {R['era_early_bps']:+.2f} bps  NW t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2018-2026 (n={R['era_late_n']}): {R['era_late_bps']:+.2f} bps  NW t = {R['era_late_t']:+.2f}\")"
        ),
        md("## The timer — can you get paid for it (sector-neutral book)?\n\n"
           "2 sides × one-way cost × NAV per day on the long-short book; short (high-vol) pays 50 bps/yr borrow."),
        code(
            "for tag,g,c,n,t,sh in [('1 bp',R['timer_1_gross'],R['timer_1_cost'],R['timer_1_net'],R['timer_1_t'],R['timer_1_sh']),\n"
            "                       ('5 bps',R['timer_5_gross'],R['timer_5_cost'],R['timer_5_net'],R['timer_5_t'],R['timer_5_sh'])]:\n"
            "    print(f\"{tag:>5} one-way: gross {g:+.2f} -> net {n:+.2f} bps/day (cost {c:.2f}/day, t={t:+.2f}, Sharpe {sh:.2f})\")"
        ),
        md("## Synthetic positive control — the machinery is sound\n\n"
           "Live: the sector-neutral detector must NOT fire on the null, must recover a planted "
           "within-sector low-vol effect, and — crucially — a pure sector premium must fool the "
           "RAW sort but leave the NEUTRAL sort silent."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np, pandas as pd\n"
            "from sn_lowvol import data, strategy as st\n"
            "secmap = pd.Series(data.synthetic_sectors(40, 8))\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=903+s, n_assets=40, n_days=1200, n_sectors=8, sector_prem_ann=0.0), secmap, neutral=True)['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds, neutral: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.1, seed=903, n_assets=40, n_days=1500, n_sectors=8), secmap, neutral=True)\n"
            "print(f\"planted within-sector low-vol (edge=0.1), neutral: NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")\n"
            "conf = data.synthetic_panel(edge=0.0, seed=903, n_assets=40, n_days=1500, n_sectors=8, sector_prem_ann=0.08)\n"
            "rc = st.close_returns(conf)\n"
            "raw_t = st.vol_stats(st.vol_spreads(rc, secmap, 63, 0.3, neutral=False))['t_nw']\n"
            "neu_t = st.vol_stats(st.vol_spreads(rc, secmap, 63, 0.3, neutral=True))['t_nw']\n"
            "print(f\"confound (sector premium only): RAW fires (t={raw_t:+.2f}) but NEUTRAL silent (t={neu_t:+.2f})\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The low-vol edge does **not** survive on 50 liquid US "
           f"mega-caps. The sector-neutral long-low-vol / short-high-vol spread is "
           f"**{R['neu_spread_bps']:+.2f} bps/day** (NW *t* = **{R['neu_t_nw']:+.2f}**) — "
           f"significant but *opposite in sign* to the claim (the wild tech names out-earned), "
           f"holding in both eras (*t* = {R['era_early_t']:+.2f} / {R['era_late_t']:+.2f}) and "
           f"≈{R['placebo_sigma']:.1f}σ into the left tail of a 1,000-permutation placebo. On the "
           f"anomaly's own **Sharpe** axis the low-vol leg has no advantage once sector-neutral "
           f"({R['neu_lo_sh']:.2f} vs {R['neu_hi_sh']:.2f}). The naive sort's character was "
           f"largely a **defensive-sector tilt** (long book {R['raw_long_def']:.0f}% defensive; "
           f"neutralising moved the spread {R['delta_spread']:+.2f} bps), and the synthetic "
           f"control shows a pure sector premium fools the raw sort (*t* = {R['confound_raw_t']:+.2f}) "
           f"while the neutral sort stays silent ({R['confound_neu_fire']}/8) — a clean null, not machinery.\n"
           f"- **Tradability — Mirage.** The sector-neutral book loses money gross and net "
           f"(**{R['timer_1_net']:+.2f} bps/day** at 1 bp one-way, {R['timer_5_net']:+.2f} at 5 "
           f"bps); even the data-mined sign-flip is eaten by the {R['timer_1_cost']:.2f} bps/day "
           f"round-trip friction. *Survivorship: current-membership mega-caps — an upper bound.*"),
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
