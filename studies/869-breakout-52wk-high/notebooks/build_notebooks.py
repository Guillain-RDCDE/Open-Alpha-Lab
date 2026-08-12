"""Generate the two narrative notebooks for Study 869 (52-Week-High Breakout Drift).

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
# total-return, 50 liquid US large-caps, 2010-01-04 -> 2026-06-30; fresh-52w-high
# breakout event, long just-broke-out / short the rest, forward 5d & 20d).
R = dict(
    start="2010-01-04", end="2026-06-30", n_names=50, fingerprint="357fd262912f",
    # 5-day horizon
    h5_days=3039, h5_breakouts=14630,
    h5_spread=7.21, h5_t_nw=1.14, h5_t_1s=1.70, h5_brk=31.14, h5_rest=23.93, h5_welch=1.22,
    h5_hit=0.515,
    h5_placebo_obs=7.21, h5_placebo_mean=-0.044, h5_placebo_sd=5.660, h5_placebo_p=0.099,
    h5_era_early=-1.50, h5_era_early_t=-0.24, h5_era_late=14.72, h5_era_late_t=1.40,
    h5_t1_gross=7.21, h5_t1_cost=4.68, h5_t1_net=2.52, h5_t1_t=0.60,
    h5_t5_gross=7.21, h5_t5_cost=20.68, h5_t5_net=-13.48, h5_t5_t=-3.18,
    # 20-day horizon
    h20_days=3025, h20_breakouts=14584,
    h20_spread=21.64, h20_t_nw=1.20, h20_t_1s=2.48, h20_brk=131.09, h20_rest=109.45,
    h20_welch=1.75, h20_hit=0.506,
    h20_placebo_obs=21.64, h20_placebo_mean=-0.284, h20_placebo_sd=15.522, h20_placebo_p=0.083,
    h20_era_early=10.34, h20_era_early_t=0.57, h20_era_late=31.47, h20_era_late_t=1.06,
    h20_t1_gross=21.64, h20_t1_cost=6.74, h20_t1_net=14.90, h20_t1_t=1.71,
    h20_t5_gross=21.64, h20_t5_cost=22.74, h20_t5_net=-1.10, h20_t5_t=-0.13,
    # synthetic control
    null_mean_t=-0.07, null_sd_t=1.46, null_fire=1,
    planted_t=9.62, planted_welch=14.78,
)


HEADER = f"""# Study 869 — 52-Week-High Breakout Drift 🚀

**When a stock closes at a *fresh 52-week high*, does it drift up — or fade?**

This is the **event** of a new 52-week-high *breakout*, not George-Hwang **nearness** to
the high (that is study 236). Two stories are in tension: **breakout momentum** (a
decisive break above the old high releases pent-up demand, the name keeps running) versus
**resistance / anchoring** (the salient 52-week high is a ceiling, so the break fades). We
flag every fresh-52w-high day point-in-time on a liquid US cross-section
({R['start']} → {R['end']}, {R['n_names']} names) and measure the forward 5- and 20-day
return of a long-just-broke-out book vs the rest.

*Numbers below are the frozen headline (`docs/results.md`, fingerprint
`{R['fingerprint']}`); the live cells run the fast synthetic control. Survivorship:
current-membership mega-caps — survivors over-print new highs, so magnitudes are an upper
bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "A stock spends a year below some old peak, then one day *closes above it* for "
           "the first time in 52 weeks. Traders call this a **breakout**. The bullish read: "
           "the old high was a lid, and breaking it releases pent-up demand — buy new highs, "
           "they beget more highs. The bearish read: the 52-week high is a **resistance** "
           "level where anchored sellers dump and the move **fades**. Which one does the "
           "tape actually show over the next week and month?"),
        code(
            "R = dict(h5_spread=%r, h5_t_nw=%r, h5_brk=%r, h5_rest=%r,\n"
            "         h20_spread=%r, h20_t_nw=%r, h20_brk=%r, h20_rest=%r)\n"
            "print('FORWARD 5-DAY  : breakout book %%+.2f bps vs rest %%+.2f bps -> spread %%+.2f (NW t=%%+.2f)'\n"
            "      %% (R['h5_brk'], R['h5_rest'], R['h5_spread'], R['h5_t_nw']))\n"
            "print('FORWARD 20-DAY : breakout book %%+.2f bps vs rest %%+.2f bps -> spread %%+.2f (NW t=%%+.2f)'\n"
            "      %% (R['h20_brk'], R['h20_rest'], R['h20_spread'], R['h20_t_nw']))"
            % (R["h5_spread"], R["h5_t_nw"], R["h5_brk"], R["h5_rest"],
               R["h20_spread"], R["h20_t_nw"], R["h20_brk"], R["h20_rest"])
        ),
        md("## 2. Is the sort just lucky? A live synthetic control\n\n"
           "We plant a breakout-drift in a seeded toy world (`edge>0`: names that just "
           "broke out get extra forward drift) and check the detector recovers it — and "
           "stays *silent* on the null (`edge=0`: fresh highs still happen but predict "
           "nothing). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from breakout_high import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=869, n_assets=40, n_days=1200))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0015, seed=869, n_assets=40, n_days=1500))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up)' % planted['t_nw'])"
        ),
        md("## 3. The honest verdict — a drift in the *right direction* that doesn't clear the bar\n\n"
           f"On this liquid mega-cap tape the fresh-breakout name **does** out-earn the rest — "
           f"the sign is **breakout momentum, not fade**: **{R['h5_spread']:+.2f} bps** over 5 "
           f"days, **{R['h20_spread']:+.2f} bps** over 20 days. But the honest, "
           f"overlap-corrected Newey-West *t* is only **{R['h5_t_nw']:+.2f}** / "
           f"**{R['h20_t_nw']:+.2f}** — below the |*t*| = 2 bar. The permutation placebo puts "
           f"it at p ≈ {R['h5_placebo_p']:.2f}–{R['h20_placebo_p']:.2f}, and the whole effect "
           f"is a **2018-2026** phenomenon (2010-2017 is flat-to-negative). Directionally there, "
           f"statistically absent. **Signal: Weak**, and once you pay to trade it "
           f"(**{R['h5_t5_net']:+.2f} bps net** at 5 bps/side over 5 days) — **Tradability: Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 869 — 52-Week-High Breakout Drift — the teardown\n\n"
           "The per-horizon splits, the Newey-West spread *t* (lags scaled to the overlap), "
           "the pooled Welch book test, the 1,000-permutation placebo, the two-era "
           "robustness cut, the costed timer, and the 20-seed synthetic control."),
        code("R = %r" % (R,)),
        md("## The headline — long-breakout / short-rest forward-return spread\n\n"
           "Daily equal-weight breakout-minus-rest forward return; NW lags = 2×horizon "
           "(forward windows overlap and breakouts cluster)."),
        code(
            "for h in ('h5','h20'):\n"
            "    lab = '5-day' if h=='h5' else '20-day'\n"
            "    print(f\"{lab:>7}: spread {R[h+'_spread']:+.2f} bps  NW t = {R[h+'_t_nw']:+.2f}  \"\n"
            "          f\"one-sample t = {R[h+'_t_1s']:+.2f}  |  breakout {R[h+'_brk']:+.2f} vs \"\n"
            "          f\"rest {R[h+'_rest']:+.2f} bps (Welch {R[h+'_welch']:+.2f})  hit {R[h+'_hit']:.3f}\")"
        ),
        md("## Placebo — column-permute the forward returns (1,000 permutations)"),
        code(
            "for h in ('h5','h20'):\n"
            "    lab = '5-day' if h=='h5' else '20-day'\n"
            "    print(f\"{lab:>7}: observed {R[h+'_placebo_obs']:+.2f} bps vs placebo mean \"\n"
            "          f\"{R[h+'_placebo_mean']:+.3f} (sd {R[h+'_placebo_sd']:.3f}) -> p = {R[h+'_placebo_p']:.3f}\")"
        ),
        md("## Robustness — two eras (split 2018-01-01)\n\n"
           "The (weak) drift is entirely a 2018-2026 phenomenon — neither era clears |*t*| = 2."),
        code(
            "for h in ('h5','h20'):\n"
            "    lab = '5-day' if h=='h5' else '20-day'\n"
            "    print(f\"{lab:>7}: 2010-2017 {R[h+'_era_early']:+.2f} bps (NW t={R[h+'_era_early_t']:+.2f})  |  \"\n"
            "          f\"2018-2026 {R[h+'_era_late']:+.2f} bps (NW t={R[h+'_era_late_t']:+.2f})\")"
        ),
        md("## The timer — can you get paid for it?\n\n"
           "2 sides × (in+out) one-way cost per event; short pays 50 bps/yr borrow."),
        code(
            "for h in ('h5','h20'):\n"
            "    lab = '5-day' if h=='h5' else '20-day'\n"
            "    print(f\"{lab:>7} @1bp : gross {R[h+'_t1_gross']:+.2f} -> net {R[h+'_t1_net']:+.2f} bps \"\n"
            "          f\"(cost {R[h+'_t1_cost']:.2f}, t={R[h+'_t1_t']:+.2f})\")\n"
            "    print(f\"{lab:>7} @5bps: gross {R[h+'_t5_gross']:+.2f} -> net {R[h+'_t5_net']:+.2f} bps \"\n"
            "          f\"(cost {R[h+'_t5_cost']:.2f}, t={R[h+'_t5_t']:+.2f})\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted breakout drift."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from breakout_high import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=869+s, n_assets=40, n_days=1200))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.0015, seed=869, n_assets=40, n_days=1500))\n"
            "print(f\"planted (edge=0.0015): NW t = {planted['t_nw']:+.2f}, Welch t = {planted['welch_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — Weak.** A fresh 52-week-high breakout is followed by a **drift up**, "
           f"not a fade (the sign matches breakout momentum): +{R['h5_spread']:.2f} bps over 5 "
           f"days, +{R['h20_spread']:.2f} bps over 20 days, breakout book beats the rest on "
           f"both. But the overlap-corrected **NW *t* is only +{R['h5_t_nw']:.2f} / "
           f"+{R['h20_t_nw']:.2f}**, the placebo p is {R['h5_placebo_p']:.2f}–"
           f"{R['h20_placebo_p']:.2f}, and it is **entirely a 2018-2026 phenomenon** "
           f"(2010-2017 flat-to-negative). It fails the |*t*| ≥ 2 bar and does not hold across "
           f"eras. The 20-seed synthetic control recovers a *planted* drift cleanly "
           f"(*t* = +{R['planted_t']:.2f}, fires on {R['null_fire']}/20 nulls), so the weak "
           f"real drift is a feature of the tape, not an engine bug.\n"
           f"- **Tradability — Mirage.** Even at an optimistic 1 bp one-way the net edge is "
           f"insignificant (*t* = {R['h5_t1_t']:+.2f} / {R['h20_t1_t']:+.2f}); at a realistic "
           f"5 bps it goes negative ({R['h5_t5_net']:+.2f} / {R['h20_t5_net']:+.2f} bps). No "
           f"cost leaves a paycheck."),
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
