"""Generate the two narrative notebooks for Study 866 (Flight-to-Quality Beta).

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
# total-return, 50 liquid US mega-caps + TLT/SPY, 2010-01-04 -> 2026-06-30; trailing-252d
# flight-to-quality beta, monthly long-low-FTQ / short-high-FTQ, bottom/top 20%).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, rows=4147, n_months=193, names_per_mo=50,
    spread_pct=0.52, spread_ann_pct=6.24, t_nw=1.36, t_1s=1.25,
    lo_pct=1.79, hi_pct=1.27, welch_t=0.92,
    placebo_obs=0.52, placebo_mean=-0.0064, placebo_sd=0.2237, placebo_p=0.0080,
    placebo_sigma=2.35, placebo_draws=1000,
    era_early_n=91, era_early_pct=0.41, era_early_t=0.73,
    era_late_n=102, era_late_pct=0.61, era_late_t=1.18,
    crash_days=203, spy_crash_pct=-2.59, lo_crash=-3.25, hi_crash=-2.13,
    crash_cushion=1.13, crash_welch=6.78, allday_himinuslo=-0.02,
    timer1_gross=0.52, timer1_cost=0.08, timer1_net=0.44, timer1_t=1.05, timer1_ann=5.26,
    timer10_gross=0.52, timer10_cost=0.44, timer10_net=0.08, timer10_t=0.19, timer10_ann=0.94,
    null_mean_t=0.28, null_sd_t=1.11, null_fire=1,
    planted_t=12.39, planted_welch=7.70,
    fingerprint="357fd262912f",
)


HEADER = f"""# Study 866 — Flight-to-Quality Beta 🛟

**Which stocks are *true* defensives — the ones that actually rally with Treasuries when
the market sells off? Do those hedges under-earn (you pay for the protection), and do
they really cushion crashes?**

For each name we estimate a **flight-to-quality beta** (`beta_ftq`): its beta to the
**TLT** long-Treasury daily return, measured **only on down-SPY days**. A high FTQ beta
is a stock that reliably co-moves with the safe-haven bid in sell-offs — a good crash
hedge. The CAPM-of-insurance prediction is two-sided: such names should (a) earn a
**lower** average return (you pay an insurance premium) yet (b) deliver **real crash
protection**. We test both on a liquid US cross-section ({R['start']} → {R['end']},
{R['n_names']} names).

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: current-membership mega-caps — magnitudes are an upper
bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "When the market panics, investors dump stocks and pile into the safest bonds — "
           "long Treasuries rally as equities fall. A stock that *rises with that bond bid* "
           "on the worst days is a genuine hedge. The theory: hedges are expensive, so they "
           "should quietly **under-earn** in calm times — but pay you back by **losing less** "
           "when it all goes wrong. We measure each name's flight-to-quality beta and check "
           "both halves."),
        code(
            "import numpy as np, pandas as pd\n"
            "R = dict(spread_pct=%r, t_nw=%r, spread_ann_pct=%r, lo_pct=%r, hi_pct=%r,\n"
            "         crash_cushion=%r, crash_welch=%r, lo_crash=%r, hi_crash=%r)\n"
            "print('pay-for-the-hedge spread (long low-FTQ / short high-FTQ):')\n"
            "print('  %%+.2f %%%%/mo  (%%+.2f %%%%/yr)   Newey-West t = %%+.2f'\n"
            "      %% (R['spread_pct'], R['spread_ann_pct'], R['t_nw']))\n"
            "print('  low-FTQ book %%+.2f vs high-FTQ book %%+.2f %%%%/mo' %% (R['lo_pct'], R['hi_pct']))\n"
            "print()\n"
            "print('crash protection (worst 5%% of SPY days):')\n"
            "print('  low-FTQ book %%+.2f%%%% vs high-FTQ book %%+.2f%%%% -> cushion %%+.2f%%%%/day (t = %%+.2f)'\n"
            "      %% (R['lo_crash'], R['hi_crash'], R['crash_cushion'], R['crash_welch']))"
            % (R["spread_pct"], R["t_nw"], R["spread_ann_pct"], R["lo_pct"], R["hi_pct"],
               R["crash_cushion"], R["crash_welch"], R["lo_crash"], R["hi_crash"])
        ),
        md("## 2. Is the sort just lucky? A live synthetic control\n\n"
           "We plant the pay-for-the-hedge effect in a seeded toy world (`edge>0`) — names "
           "with a high flight-to-quality loading get a lower forward mean — and check the "
           "detector recovers it, while staying *silent* on the null (`edge=0`, FTQ betas "
           "present but unpriced). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from ftq_beta import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=866, n_assets=40, n_days=1400), min_stocks=10)\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.004, seed=866, n_assets=40, n_days=1500), min_stocks=10)\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — a real hedge, a weak premium\n\n"
           f"The **crash-protection** half of the claim is **confirmed and strong**: on the "
           f"worst 5% of market days the high-FTQ (hedge) book lost only {R['hi_crash']:+.2f}% "
           f"vs the low-FTQ book's {R['lo_crash']:+.2f}% — a **{R['crash_cushion']:+.2f}%/day** "
           f"cushion (Welch *t* = {R['crash_welch']:+.2f}). FTQ beta really does pick the "
           f"crash-cushioning names.\n\n"
           f"But the **pay-for-the-hedge** half is only **weak**: the long-low-FTQ / "
           f"short-high-FTQ spread is **{R['spread_pct']:+.2f} %/mo** (the right sign — the "
           f"risky names out-earned the hedges) but the Newey-West *t* is just "
           f"**{R['t_nw']:+.2f}**, short of significance, and it holds in neither era. And it "
           f"does not survive costs. **Signal: Weak · Tradability: Mirage · Crash-protection: "
           f"Confirmed.**"),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 866 — Flight-to-Quality Beta — the teardown\n\n"
           "The per-leg splits, the Newey-West spread *t*, the pooled Welch book test, the "
           "1,000-permutation placebo, the two-era robustness cut, the crash-day drawdown "
           "comparison, the costed timer, and the 20-seed synthetic control."),
        code("R = %r" % (R,)),
        md("## The headline — long-low-FTQ / short-high-FTQ spread (the pay-for-the-hedge premium)\n\n"
           "Monthly equal-weight bottom-20% minus top-20% FTQ-beta spread."),
        code(
            "print(f\"spread        : {R['spread_pct']:+.2f} %/mo ({R['spread_ann_pct']:+.2f} %/yr)  \"\n"
            "      f\"NW(6) t = {R['t_nw']:+.2f}  one-sample t = {R['t_1s']:+.2f}\")\n"
            "print(f\"books         : low-FTQ {R['lo_pct']:+.2f} vs high-FTQ {R['hi_pct']:+.2f} %/mo \"\n"
            "      f\"(Welch t = {R['welch_t']:+.2f})\")\n"
            "print('right sign (risky names out-earn hedges) but |t| < 2 -> WEAK')"
        ),
        md("## Placebo — permute the forward returns within each month (1,000 draws)\n\n"
           "Cross-sectionally the tilt *is* real, even though the time-series HAC t is weak."),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} %/mo vs placebo mean {R['placebo_mean']:+.4f} \"\n"
            "      f\"(sd {R['placebo_sd']:.4f}) -> right-tail p = {R['placebo_p']:.4f} \"\n"
            "      f\"(~{R['placebo_sigma']:+.2f} sd-units into the right tail)\")"
        ),
        md("## Robustness — two eras (split 2018-01-01)"),
        code(
            "print(f\"2011-2018 (n={R['era_early_n']}): {R['era_early_pct']:+.2f} %/mo  NW t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2018-2026 (n={R['era_late_n']}): {R['era_late_pct']:+.2f} %/mo  NW t = {R['era_late_t']:+.2f}\")\n"
            "print('sign stable (positive both halves) but neither clears |t|>=2')"
        ),
        md("## Crash protection — the *other* half of the claim (worst 5% of SPY days)\n\n"
           "This is where the FTQ sort earns its keep — a big, highly significant cushion."),
        code(
            "print(f\"{R['crash_days']} crash days, mean SPY {R['spy_crash_pct']:+.2f}%:\")\n"
            "print(f\"  low-FTQ book {R['lo_crash']:+.2f}%/day  vs  high-FTQ book {R['hi_crash']:+.2f}%/day\")\n"
            "print(f\"  cushion (high-low) {R['crash_cushion']:+.2f}%/day  (Welch t = {R['crash_welch']:+.2f})\")\n"
            "print(f\"  [across ALL days the same difference is a negligible {R['allday_himinuslo']:+.2f}%/day]\")"
        ),
        md("## The timer — can you get paid the premium?\n\n"
           "2 legs × round-trip × one-way × NAV per month; short pays 50 bps/yr borrow."),
        code(
            "for tag,g,c,n,t,a in [('1 bp',R['timer1_gross'],R['timer1_cost'],R['timer1_net'],R['timer1_t'],R['timer1_ann']),\n"
            "                      ('10 bps',R['timer10_gross'],R['timer10_cost'],R['timer10_net'],R['timer10_t'],R['timer10_ann'])]:\n"
            "    print(f\"{tag:>6} one-way: gross {g:+.2f} -> net {n:+.2f} %/mo (cost {c:.2f}/mo, t={t:+.2f}, ~{a:+.2f}%/yr)\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted relation."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from ftq_beta import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=866+s, n_assets=40, n_days=1400), min_stocks=10)['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.004, seed=866, n_assets=40, n_days=1500), min_stocks=10)\n"
            "print(f\"planted (edge=0.004): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** The pay-for-the-hedge premium (long low-FTQ / short high-FTQ) "
           f"is **{R['spread_pct']:+.2f} %/mo** ({R['spread_ann_pct']:+.2f} %/yr) with the "
           f"**right sign** and sits ≈{R['placebo_sigma']:+.2f} sd-units into the right tail of "
           f"a 1,000-permutation placebo (p = {R['placebo_p']:.4f}) — cross-sectionally real, "
           f"not a lucky sort. But the conservative Newey-West *t* is only **{R['t_nw']:+.2f}**, "
           f"it fails |*t*| ≥ 2, and neither era clears significance "
           f"({R['era_early_t']:+.2f} / {R['era_late_t']:+.2f}). A 20-seed synthetic control "
           f"recovers a *planted* relation cleanly (*t* = {R['planted_t']:+.2f}), so this is an "
           f"honest read of a thin tape.\n"
           f"- **Tradability — Mirage.** Insignificant even gross; at 1 bp net "
           f"**{R['timer1_net']:+.2f} %/mo** (*t* = {R['timer1_t']:+.2f}), at 10 bps net "
           f"**{R['timer10_net']:+.2f} %/mo** (*t* = {R['timer10_t']:+.2f}) — costs plus borrow "
           f"eat it.\n"
           f"- **Crash-protection (descriptive) — Confirmed.** The sort really cushions crashes: "
           f"the high-FTQ book lost **{R['crash_cushion']:+.2f}%/day less** than the low-FTQ book "
           f"on the worst 5% of SPY days (Welch *t* = {R['crash_welch']:+.2f}). FTQ beta is a "
           f"genuine risk characteristic — just not a robustly *priced* one on mega-caps."),
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
