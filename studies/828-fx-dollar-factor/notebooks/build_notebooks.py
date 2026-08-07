"""Generate the two narrative notebooks for Study 828 (FX Dollar Factor).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily FX,
# month-end, 7 G10 currencies vs USD, 2003-12-31 -> 2026-06-30; DOL = equal-weight mean
# of foreign-vs-USD spot returns; timing = trailing-12m DOL proxy for the avg. fwd discount).
R = dict(
    start="2003-12-31", end="2026-06-30", n_ccy=7, n_months=270, fingerprint="de324e1d9334",
    spot_bps=0.52, spot_ann=0.06, spot_t_nw=0.04, spot_t_1s=0.04, spot_sharpe=0.01, vol=7.7,
    ex_bps=1.24, ex_ann=0.15, ex_t_nw=0.09,
    tim_beta=-0.0307, tim_t=-1.46, tim_r2=0.0101, tim_n=258,
    placebo_obs=0.0307, placebo_sd=0.0238, placebo_p=0.084, placebo_rot=1000,
    era_early_bps=8.91, era_early_ann=1.07, era_early_t=0.38, era_early_n=132,
    era_late_bps=-7.50, era_late_ann=-0.90, era_late_t=-0.55, era_late_n=138,
    static_gross=-0.29, static_net=-0.41, static_sharpe=-0.05, static_t=-0.25,
    timed_gross=-0.88, timed_net=-0.96, timed_sharpe=-0.18, timed_t=-0.86,
    switches=1.63, invested=51,
    null_prem_mean_t=0.32, null_prem_sd=0.84, null_prem_fire=0, null_tim_fire=1,
    planted_prem_bps=24.41, planted_prem_t=3.63, planted_tim_beta=0.055, planted_tim_t=5.23,
)


HEADER = f"""# Study 828 — FX Dollar Factor 💵

**Is the "dollar factor" a priced risk premium — and can the forward discount time it?**

Lustig, Roussanov & Verdelhan (2011) name two currency factors: the carry slope **HML_FX**
and the **dollar factor DOL** — the equal-weight average return of a basket of foreign
currencies against the USD. DOL's *unconditional* premium is famously small; their claim is
that it is priced **conditionally**, timed by the **average forward discount**. We build DOL on
a {R['n_ccy']}-currency G10 basket ({R['start']} → {R['end']}, {R['n_months']} months) and test
both — carefully inverting the USD-base quotes so every pair reads USD-per-foreign.

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Survivorship: current-membership G10 — magnitudes are an upper bound.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "The **dollar factor** is just the average of every foreign currency's move against "
           "the USD. If you borrow dollars and hold a basket of foreign currencies, DOL is what "
           "you earn on the spot leg — positive when the dollar *weakens*. LRV say this level "
           "factor is priced, but only **conditionally**: it should pay more when the average "
           "**forward discount** (foreign rates minus US rates) is high. We test the premium and "
           "the timing."),
        code(
            "R = dict(spot_bps=%r, spot_ann=%r, spot_t_nw=%r, tim_t=%r, era_early_ann=%r, era_late_ann=%r)\n"
            "print('DOL premium (spot): %%+.2f bps/mo = %%+.2f%%%%/yr  (Newey-West t = %%+.2f)'\n"
            "      %% (R['spot_bps'], R['spot_ann'], R['spot_t_nw']))\n"
            "print('  -> a t of %%+.2f is indistinguishable from zero' %% R['spot_t_nw'])\n"
            "print('dollar-timing slope t = %%+.2f  (insignificant, and wrong sign)' %% R['tim_t'])\n"
            "print('era split: %%+.2f%%%%/yr then %%+.2f%%%%/yr -> flips sign'\n"
            "      %% (R['era_early_ann'], R['era_late_ann']))"
            % (R["spot_bps"], R["spot_ann"], R["spot_t_nw"], R["tim_t"],
               R["era_early_ann"], R["era_late_ann"])
        ),
        md("## 2. Is the engine honest? A live synthetic control\n\n"
           "We plant a real dollar premium (`edge>0`) and a real timing relation (`timing>0`) in "
           "a seeded toy world and check the detector recovers each — and that it stays *silent* "
           "on the null (`edge=0, timing=0`). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from dollar_factor import data, strategy as st\n"
            "null    = st.synthetic_detect(data.synthetic_panel(edge=0.0,   timing=0.0,  seed=828, n_months=480))\n"
            "prem    = st.synthetic_detect(data.synthetic_panel(edge=0.004, timing=0.0,  seed=828, n_months=480))\n"
            "timing  = st.synthetic_detect(data.synthetic_panel(edge=0.0,   timing=0.02, seed=828, n_months=480))\n"
            "print('null world    : premium t = %+.2f, timing t = %+.2f  (both ~0)' % (null['prem_t_nw'], null['timing_t']))\n"
            "print('planted premium: premium t = %+.2f  (should light up)' % prem['prem_t_nw'])\n"
            "print('planted timing : timing  t = %+.2f  (should light up)' % timing['timing_t'])"
        ),
        md("## 3. The honest verdict — the dollar factor pays nothing here\n\n"
           f"On a {R['n_ccy']}-currency G10 basket the unconditional DOL premium is "
           f"**{R['spot_bps']:+.2f} bps/mo = {R['spot_ann']:+.2f}%/yr** with NW *t* = "
           f"**{R['spot_t_nw']:+.2f}** — a premium indistinguishable from zero, which is exactly "
           f"LRV's own starting point. The **dollar-timing** test, run with the only conditioning "
           f"variable we can build from yfinance spot (a trailing-dollar-trend proxy for the "
           f"average forward discount), is **insignificant and wrong-signed** (NW *t* = "
           f"**{R['tim_t']:+.2f}**, placebo p = {R['placebo_p']:.3f}), and the premium flips sign "
           f"across eras. The synthetic control recovers *planted* versions of both cleanly, so "
           f"this is a genuine absence, not a bug. **Signal: None** (the claimed edge is absent, "
           f"with the true rate-based forward discount not reconstructable from spot — a data "
           f"limit), **Tradability: Mirage** (both books lose money net)."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 828 — FX Dollar Factor — the teardown\n\n"
           "The DOL premium Newey-West *t*, the dollar-timing regression + block-shuffle placebo, "
           "the two-era cut, the costed static/timed books, and the 20-seed synthetic control."),
        code("R = %r" % (R,)),
        md("## The headline — the DOL premium\n\n"
           "Equal-weight foreign-currency-vs-USD basket, monthly."),
        code(
            "print(f\"DOL spot   : {R['spot_bps']:+.2f} bps/mo ({R['spot_ann']:+.2f}%/yr)  \"\n"
            "      f\"NW(6) t = {R['spot_t_nw']:+.2f}  one-sample t = {R['spot_t_1s']:+.2f}  Sharpe {R['spot_sharpe']:+.2f}\")\n"
            "print(f\"DOL excess : {R['ex_bps']:+.2f} bps/mo ({R['ex_ann']:+.2f}%/yr)  NW(6) t = {R['ex_t_nw']:+.2f}  (+carry proxy)\")\n"
            "print(f\"vol        : {R['vol']:.1f}%/yr\")"
        ),
        md("## Dollar-timing — predictive regression DOL_{t+1} = a + b·signal_t\n\n"
           "signal = trailing-12m DOL (a spot-only proxy for the average forward discount)."),
        code(
            "print(f\"slope b   : {R['tim_beta']:+.4f}   NW(6) t = {R['tim_t']:+.2f}   R2 = {R['tim_r2']:.4f}   n = {R['tim_n']}\")\n"
            "print(f\"placebo   : |obs b| {R['placebo_obs']:.4f} vs placebo sd {R['placebo_sd']:.4f} \"\n"
            "      f\"({R['placebo_rot']} rotations) -> p = {R['placebo_p']:.3f}\")\n"
            "print('  the slope is insignificant AND wrong-signed (high trend -> lower next DOL)')"
        ),
        md("## Robustness — two eras (split 2015-01-01)"),
        code(
            "print(f\"2004-2014 (n={R['era_early_n']}): {R['era_early_bps']:+.2f} bps/mo ({R['era_early_ann']:+.2f}%/yr)  NW t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2015-2026 (n={R['era_late_n']}): {R['era_late_bps']:+.2f} bps/mo ({R['era_late_ann']:+.2f}%/yr)  NW t = {R['era_late_t']:+.2f}\")\n"
            "print('  the premium flips sign across the halves -> not a stable factor')"
        ),
        md("## The timer — can you get paid for it?\n\n"
           "Static long basket (20% turnover) and a timed overlay (long when trend>0), 5 bps one-way."),
        code(
            "print(f\"static long DOL : gross {R['static_gross']:+.2f}%/yr -> net {R['static_net']:+.2f}%/yr \"\n"
            "      f\"(Sharpe {R['static_sharpe']:+.2f}, t {R['static_t']:+.2f})\")\n"
            "print(f\"timed DOL       : gross {R['timed_gross']:+.2f}%/yr -> net {R['timed_net']:+.2f}%/yr \"\n"
            "      f\"(Sharpe {R['timed_sharpe']:+.2f}, t {R['timed_t']:+.2f}, {R['switches']:.2f} switches/yr, invested {R['invested']}%)\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover planted premium + timing."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from dollar_factor import data, strategy as st\n"
            "npt = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, timing=0.0, seed=828+s, n_months=480))['prem_t_nw'] for s in range(10)])\n"
            "ntt = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, timing=0.0, seed=828+s, n_months=480))['timing_t'] for s in range(10)])\n"
            "print(f\"null (10 seeds): premium NW t mean {npt.mean():+.2f} (|t|>=2 in {(abs(npt)>=2).sum()}/10); timing |t|>=2 in {(abs(ntt)>=2).sum()}/10\")\n"
            "prem = st.synthetic_detect(data.synthetic_panel(edge=0.004, timing=0.0, seed=828, n_months=480))\n"
            "tim  = st.synthetic_detect(data.synthetic_panel(edge=0.0, timing=0.02, seed=828, n_months=480))\n"
            "print(f\"planted premium: DOL {prem['prem_mean_bps']:+.2f} bps/mo, NW t = {prem['prem_t_nw']:+.2f}\")\n"
            "print(f\"planted timing : slope {tim['timing_beta']:+.4f}, t = {tim['timing_t']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The unconditional dollar factor earns **{R['spot_ann']:+.2f}%/yr** "
           f"(NW *t* = **{R['spot_t_nw']:+.2f}**) — indistinguishable from zero, and LRV's own "
           f"premise. The dollar-timing test with the spot-only forward-discount proxy is "
           f"**insignificant and wrong-signed** (NW *t* = **{R['tim_t']:+.2f}**, placebo p = "
           f"{R['placebo_p']:.3f}), and the premium flips sign across eras "
           f"(*t* = {R['era_early_t']:+.2f} / {R['era_late_t']:+.2f}). The 20-seed synthetic control "
           f"recovers a *planted* premium (*t* = {R['planted_prem_t']:+.2f}) and *planted* timing "
           f"(*t* = {R['planted_tim_t']:+.2f}) cleanly, firing on {R['null_prem_fire']}/20 premium "
           f"nulls — so the flat tape is a real absence. *(The true rate-based average forward "
           f"discount is not reconstructable from spot — a data limit, honestly flagged.)*\n"
           f"- **Tradability — Mirage.** The static long basket nets **{R['static_net']:+.2f}%/yr** "
           f"and the timed overlay **{R['timed_net']:+.2f}%/yr** at 5 bps — no premium to pay for "
           f"the friction."),
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
