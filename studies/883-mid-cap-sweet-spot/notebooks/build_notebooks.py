"""Generate the two narrative notebooks for Study 883 (Mid-Cap Sweet Spot).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance total-return,
# IJH/MDY/SPY/IWM/BIL, as-of 2026-06-30, fingerprint d294f0cdb517).
R = dict(
    as_of="2026-06-30", fingerprint="d294f0cdb517",
    common_n=4801, common="2007-05 -> 2026-06",
    ijh_sh=0.453, mdy_sh=0.440, spy_sh=0.542, iwm_sh=0.394,
    ijh_ret=11.58, spy_ret=12.12, iwm_ret=11.15,
    ijh_vol=22.5, spy_vol=19.8, iwm_vol=24.8,
    ijh_dd=-55.1, spy_dd=-55.2, iwm_dd=-58.6,
    adv_spy=-0.089, adv_spy_lo=-0.225, adv_spy_hi=0.050,
    adv_iwm=+0.060, adv_iwm_lo=-0.038, adv_iwm_hi=0.169,
    ijh_spy_d=1.74, ijh_spy_t=1.19, ijh_iwm_d=0.44, ijh_iwm_t=0.38,
    mdy_spy_d=0.95, mdy_spy_t=0.67, mdy_spy_n=7839, mdy_iwm_d=0.32, mdy_iwm_t=0.27,
    era1_d=3.21, era1_t=0.94, era2_d=3.94, era2_t=1.48,
    era3_d=1.49, era3_t=0.74, era4_d=-3.50, era4_t=-1.23,
    cost_spy_g=1.74, cost_spy_n=1.00, cost_spy_t=0.68,
    cost_iwm_g=0.44, cost_iwm_n=-0.30, cost_iwm_t=-0.25, charge=0.74,
    null_beats=1, planted_advL=1.011, planted_tL=5.78, planted_advS=1.110, planted_tS=5.02,
)


HEADER = f"""# Study 883 — Mid-Cap Sweet Spot 🎯

**Are mid-caps the "forgotten middle" — a better risk-adjusted return than BOTH large
(SPY) and small (IWM)?**

Folklore (and a lot of ETF marketing) says the mid-cap band is the sweet spot: past
small-cap fragility, still growing faster than mega-caps, and under-followed by analysts.
We test whether the mid-cap ETF (**IJH**, plus **MDY** for the longer S&P MidCap 400 tape)
delivers a genuine **excess-of-cash Sharpe advantage over BOTH** neighbours
({R['common']}, cash = BIL), whether it holds across eras, and whether it survives costs.

*Numbers below are the frozen headline (`docs/results.md`, fingerprint
`{R['fingerprint']}`); the live cells run the fast synthetic control. Short cash history:
BIL lists only from 2007, so the Sharpe race misses the 1995-2006 mid heyday — named on
the Signal axis.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Small-caps promise growth but carry fragility, illiquidity and default risk. "
           "Mega-caps are safe but slow and picked over by every analyst alive. The "
           "**middle** — big enough to be stable, small enough to still compound, and "
           "under-covered — is supposed to be where the best *risk-adjusted* return "
           "hides. If true, the mid-cap ETF should out-Sharpe **both** SPY and IWM."),
        code(
            "R = dict(ijh_sh=%r, spy_sh=%r, iwm_sh=%r)\n"
            "print('excess-of-cash Sharpe, 2007-2026 (higher = better):')\n"
            "print(f\"  large SPY : {R['spy_sh']:.3f}\")\n"
            "print(f\"  MID  IJH : {R['ijh_sh']:.3f}   <- the 'sweet spot'\")\n"
            "print(f\"  small IWM : {R['iwm_sh']:.3f}\")\n"
            "print('mid is literally in the MIDDLE - below large, just above small.')"
            % (R["ijh_sh"], R["spy_sh"], R["iwm_sh"])
        ),
        md("## 2. So did the middle win? No — it sat in the middle\n\n"
           f"On the excess-of-cash Sharpe race mid-cap **fails the 'beats BOTH' test on its "
           f"own terms**: IJH's {R['ijh_sh']:.3f} is *below* large SPY's {R['spy_sh']:.3f} "
           f"and only just above small IWM's {R['iwm_sh']:.3f}. The paired bootstrap says "
           f"the advantage is not distinguishable from zero either way — vs SPY "
           f"**{R['adv_spy']:+.3f}** (95% CI [{R['adv_spy_lo']:+.3f}, {R['adv_spy_hi']:+.3f}]), "
           f"vs IWM **{R['adv_iwm']:+.3f}** (CI [{R['adv_iwm_lo']:+.3f}, {R['adv_iwm_hi']:+.3f}]). "
           f"The 2010s mega-cap tech run handed the best Sharpe to *large*, not the middle."),
        md("## 3. But mid *did* out-return both over the long run — just not reliably\n\n"
           f"The return **difference** doesn't need the cash leg, so it reaches back to "
           f"1995. Over the full tape mid-cap out-returned both: **MDY − SPY = "
           f"{R['mdy_spy_d']:+.2f}%/yr** (1995-2026), **IJH − SPY = {R['ijh_spy_d']:+.2f}%/yr**, "
           f"IJH − IWM = {R['ijh_iwm_d']:+.2f}%/yr. The *sign* is real — the forgotten middle "
           f"tilt exists — but **not one difference clears a HAC *t* of 2** (best is "
           f"{R['ijh_spy_t']:+.2f} over 26 years). And it is not stable:"),
        code(
            "R = dict(era1_d=%r, era1_t=%r, era4_d=%r, era4_t=%r)\n"
            "print('MDY - SPY by era (+ = mid beats large):')\n"
            "print(f\"  1995-2002 : {R['era1_d']:+.2f}%%/yr  (HAC t {R['era1_t']:+.2f})\")\n"
            "print(f\"  2017-2026 : {R['era4_d']:+.2f}%%/yr  (HAC t {R['era4_t']:+.2f})  <- REVERSED\")"
            % (R["era1_d"], R["era1_t"], R["era4_d"], R["era4_t"])
        ),
        md("## 4. A live synthetic control — the detector works\n\n"
           "To be sure the 'no robust advantage' reading isn't a dead detector, we plant a "
           "real mid Sharpe edge in a seeded toy world and check it fires — and stays quiet "
           "on the null. No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from midcap import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_world(n_days=3000, edge=0.0, seed=883))\n"
            "planted = st.synthetic_detect(data.synthetic_world(n_days=3000, edge=0.0006, seed=883))\n"
            "print('null world   : mid adv vs large = %+.3f (t %+.2f) -> beats_both=%s'\n"
            "      % (null['adv_large'], null['t_large'], null['beats_both']))\n"
            "print('planted world: mid adv vs large = %+.3f (t %+.2f) -> beats_both=%s'\n"
            "      % (planted['adv_large'], planted['t_large'], planted['beats_both']))"
        ),
        md(f"## 5. The honest verdict\n\n"
           f"- **Signal: Weak.** Mid-cap is real folklore with a real *sign* — it out-returned "
           f"both neighbours over the long run — but the advantage **never reaches "
           f"significance, sits below large on a modern Sharpe basis, and reversed in the "
           f"last decade** ({R['era4_d']:+.2f}%/yr vs large in 2017-2026). A fragile, "
           f"era-dependent tilt, not a dependable sweet spot.\n"
           f"- **Tradability: Mirage.** Long-mid/short-large nets only **+{R['cost_spy_n']:.2f}%/yr** "
           f"(net *t* = +{R['cost_spy_t']:.2f}, and on the leg that inverted); "
           f"long-mid/short-small nets **{R['cost_iwm_n']:+.2f}%/yr** after costs. Nothing "
           f"bankable."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 883 — Mid-Cap Sweet Spot — the teardown\n\n"
           "The excess-vs-excess Sharpe race, the paired-bootstrap advantage CIs, the HAC "
           "*t* on the cash-independent pairwise difference, the four-era myth-check, the "
           "costed dollar-neutral spread, and the planted-edge synthetic control."),
        code("R = %r" % (R,)),
        md("## The race — excess-of-cash Sharpe (2007-2026 common window)\n\n"
           "Every leg minus BIL cash, so the race is apples-to-apples."),
        code(
            "print(f\"n = {R['common_n']} days ({R['common']})\")\n"
            "for tag, sh, rt, vl, dd in [('IJH mid ', R['ijh_sh'], R['ijh_ret'], R['ijh_vol'], R['ijh_dd']),\n"
            "                            ('SPY large', R['spy_sh'], R['spy_ret'], R['spy_vol'], R['spy_dd']),\n"
            "                            ('IWM small', R['iwm_sh'], R['iwm_ret'], R['iwm_vol'], R['iwm_dd'])]:\n"
            "    print(f\"  {tag}: exSharpe {sh:.3f}  ret {rt:+.2f}%  vol {vl:.1f}%  maxDD {dd:.1f}%\")\n"
            "print('mid sits BELOW large and just above small -> fails beats-both.')"
        ),
        md("## The advantage — mid excess-Sharpe minus each neighbour, paired block bootstrap\n\n"
           "2,000 draws, 21-day blocks, resampled jointly to keep the cross-correlation."),
        code(
            "print(f\"IJH - SPY : adv {R['adv_spy']:+.3f}  95% CI [{R['adv_spy_lo']:+.3f}, {R['adv_spy_hi']:+.3f}]  -> spans 0\")\n"
            "print(f\"IJH - IWM : adv {R['adv_iwm']:+.3f}  95% CI [{R['adv_iwm_lo']:+.3f}, {R['adv_iwm_hi']:+.3f}]  -> spans 0\")"
        ),
        md("## The pairwise return difference — cash-independent, full tape\n\n"
           "`mid − large` doesn't need the cash leg, so MDY reaches back to 1995."),
        code(
            "print(f\"IJH - SPY : {R['ijh_spy_d']:+.2f}%/yr  HAC t {R['ijh_spy_t']:+.2f}\")\n"
            "print(f\"IJH - IWM : {R['ijh_iwm_d']:+.2f}%/yr  HAC t {R['ijh_iwm_t']:+.2f}\")\n"
            "print(f\"MDY - SPY : {R['mdy_spy_d']:+.2f}%/yr  HAC t {R['mdy_spy_t']:+.2f}  (n={R['mdy_spy_n']}, since 1995)\")\n"
            "print(f\"MDY - IWM : {R['mdy_iwm_d']:+.2f}%/yr  HAC t {R['mdy_iwm_t']:+.2f}\")\n"
            "print('sign-correct (mid out-returns both) but NONE clears |t|=2.')"
        ),
        md("## Robustness — MDY − SPY by era (the myth-check)"),
        code(
            "for lbl, d, t in [('1995-2002', R['era1_d'], R['era1_t']), ('2003-2009', R['era2_d'], R['era2_t']),\n"
            "                  ('2010-2016', R['era3_d'], R['era3_t']), ('2017-2026', R['era4_d'], R['era4_t'])]:\n"
            "    flag = '  <- REVERSED' if d < 0 else ''\n"
            "    print(f\"  {lbl}: {d:+.2f}%/yr  HAC t {t:+.2f}{flag}\")"
        ),
        md("## The costed spread — long mid / short neighbour (dollar-neutral)\n\n"
           "50 bps/yr borrow on the short + 2 sides × 3 bps × 4 rebalances/yr."),
        code(
            "print(f\"long IJH / short SPY: gross {R['cost_spy_g']:+.2f} -> net {R['cost_spy_n']:+.2f}%/yr (t {R['cost_spy_t']:+.2f})\")\n"
            "print(f\"long IJH / short IWM: gross {R['cost_iwm_g']:+.2f} -> net {R['cost_iwm_n']:+.2f}%/yr (t {R['cost_iwm_t']:+.2f})\")\n"
            "print(f\"charge = {R['charge']:.2f}%/yr; only the reversed leg is net-positive, at t=+0.68.\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: a common market factor drives large/mid/small; `edge` lifts mid's excess "
           "mean. The detector must fire on a planted edge and stay quiet on the null."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from midcap import data, strategy as st\n"
            "beats = 0\n"
            "for s in range(20):\n"
            "    sig = st.synthetic_detect(data.synthetic_world(n_days=3000, edge=0.0, seed=883+s))\n"
            "    beats += int(sig['beats_both'] and abs(sig['t_large'])>=2 and abs(sig['t_small'])>=2)\n"
            "planted = st.synthetic_detect(data.synthetic_world(n_days=3000, edge=0.0006, seed=883))\n"
            "print(f\"null (edge=0), 20 seeds: strict-significant-beats-both in {beats}/20 (~nominal 5%)\")\n"
            "print(f\"planted (edge=0.0006): adv vs large {planted['adv_large']:+.3f} (t {planted['t_large']:+.2f}), \"\n"
            "      f\"vs small {planted['adv_small']:+.3f} (t {planted['t_small']:+.2f}), beats_both={planted['beats_both']}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — WEAK.** The 'mid beats BOTH' sweet-spot claim does not clear a robust "
           f"bar. Mid's excess Sharpe ({R['ijh_sh']:.3f}) sits *below* large ({R['spy_sh']:.3f}) "
           f"and barely above small ({R['iwm_sh']:.3f}); both advantage CIs span zero. The "
           f"long-run return tilt is sign-correct (MDY − SPY {R['mdy_spy_d']:+.2f}%/yr since 1995) "
           f"but never significant (best HAC *t* = +{R['ijh_spy_t']:.2f}) and it **reversed** "
           f"({R['era4_d']:+.2f}%/yr in 2017-2026). A fragile, era-dependent tilt. The synthetic "
           f"control fires cleanly on a planted edge (*t* ≈ 5-6) and fires on {R['null_beats']}/20 "
           f"nulls, so the detector is honest.\n"
           f"- **Tradability — MIRAGE.** No costed spread clears the bar: long-mid/short-large "
           f"nets +{R['cost_spy_n']:.2f}%/yr at *t* = +{R['cost_spy_t']:.2f} (and on the leg that "
           f"inverted), long-mid/short-small nets {R['cost_iwm_n']:+.2f}%/yr. The Sharpe edge over "
           f"small is a lower-return / similar-vol artifact — a Mirage after costs."),
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
