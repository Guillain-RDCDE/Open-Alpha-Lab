"""Generate the two narrative notebooks for Study 874 (IPO-Price Anchoring).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape headline numbers are quoted from the
frozen ``R`` dict (mirroring docs/results.md); the live cells run only the fast synthetic
positive control, so execution is quick and network-free.
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30, fingerprint
# 1eaa178051af; 44 curated listings vs SPY, yfinance daily closes 2014-01 -> 2026-06).
R = dict(
    as_of="2026-06-30", fingerprint="1eaa178051af",
    n_names=44, n_ipo=40, n_direct=4, start="2014-01-31", end="2026-06-30",
    active_months=86, n_obs=2698, avg_names=30.3, below_share=42.0,
    # anchoring pull (FM slope): all vs ipo-only
    anchor_slope=-0.0023, anchor_bps10=-2.3, anchor_t=-0.39, anchor_t1s=-0.40,
    anchor_share_neg=44, anchor_n=86,
    anchor_slope_ipo=-0.0018, anchor_t_ipo=-0.30,
    # below-offer drag (spread bps/mo, ann %, NW t, welch t, legs, n)
    below_bps=-56.84, below_ann=-6.61, below_t=-0.56, below_welch=-0.35,
    below_leg=13.07, above_leg=69.91, below_n=76,
    below_bps_ipo=-99.93, below_t_ipo=-0.94,
    # placebo
    plac_obs=-0.00233, plac_mean=-0.00026, plac_sd=0.00478,
    plac_p_left=0.322, plac_p_two=0.610, plac_draws=1000,
    # eras (cut 2022-07)
    era_early_bps=-43.31, era_early_t=-0.18, era_early_n=29,
    era_late_bps=-65.19, era_late_t=-0.87, era_late_n=47,
    # timer: (cost, borrow, gross, gross_t, net, net_ann, net_t, dd)
    timer=[(10.0, 3.0, 56.84, 0.56, 11.84, 1.43, 0.12, -57.0),
           (20.0, 5.0, 56.84, 0.56, -24.82, -2.94, -0.24, -61.7)],
    # synthetic control: (edge, mean_slope, mean_t, reject%)
    ctrl=[(0.0, 0.0023, 0.41, 0), (0.15, -0.1441, -12.54, 100), (0.30, -0.2946, -19.76, 100)],
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
import numpy as np
from ipo_anchor import data, strategy as st
"""


HEADER = f"""# Study 874 — IPO-Price Anchoring ⚓

**Do investors anchor on the IPO *offer price*?**

The offer price is the one round number every newly public stock is introduced by. Two pieces
of folklore follow: (1) an **anchoring pull** — a name stretched far above its offer should get
pulled back down, one below pulled back up (forward return *negatively* related to the
gap-from-offer); and (2) a **below-offer drag** — crossing below the offer, the cohort's
collective cost basis, is a persistent weight. We hard-code a curated table of
**{R['n_names']} famous recent US listings** (offer/reference price + first-trade date, public
record) and test both against **market-adjusted** forward returns (name − SPY),
{R['start']} → {R['end']}.

*Numbers below are the frozen headline (`docs/results.md`, fingerprint `{R['fingerprint']}`); the
live cells run the fast synthetic control. Curation bias (a small, one-dominant-cohort set) is
named on the Signal axis.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "A stock IPOs at, say, $68. That round number sticks. Anchoring says the price gets "
           "*pulled* back toward it — expensive-above gets sold, cheap-below gets bought — and "
           "loss-aversion lore adds that once you're *below* the offer (everyone who bought the "
           "deal is under water) the name carries a drag. We measure the gap `log(price/offer)` "
           "every month and ask whether it predicts next month's **market-adjusted** return."),
        code(
            "R = dict(anchor_slope=%r, anchor_t=%r, below_bps=%r, below_t=%r,\n"
            "         below_leg=%r, above_leg=%r)\n"
            "print('anchoring pull  : FM slope %%+.4f  (NW t = %%+.2f)  <- right sign, no significance'\n"
            "      %% (R['anchor_slope'], R['anchor_t']))\n"
            "print('below-offer drag: %%+.1f bps/mo  (NW t = %%+.2f)'\n"
            "      %% (R['below_bps'], R['below_t']))\n"
            "print('   below-offer basket %%+.1f vs above-offer basket %%+.1f bps/mo (market-adj)'\n"
            "      %% (R['below_leg'], R['above_leg']))"
            % (R["anchor_slope"], R["anchor_t"], R["below_bps"], R["below_t"],
               R["below_leg"], R["above_leg"])
        ),
        md("## 2. The trap: a loud number with a quiet *t*\n\n"
           f"Below-offer names trailed above-offer names by **{abs(R['below_bps']):.0f} bps/mo** "
           f"market-adjusted — that *looks* like a real drag. But the Newey-West *t* is "
           f"**{R['below_t']:+.2f}**: with ~45 names that mostly IPO'd in one 2020-21 wave and "
           "rose and crashed together, there just isn't enough independent information to call it. "
           "The desk's whole job is to separate a big number from a *significant* one."),
        md("## 3. Is the pipeline honest? A live synthetic control\n\n"
           "We plant an anchoring pull in a seeded toy world (`edge>0`, forward return reverts "
           "toward the anchor) and check the detector recovers it — and stays *silent* on the "
           "null (`edge=0`, gap present but unpriced). No network."),
        code(
            BOOT +
            "null = st.synthetic_control(0.0, n_seeds=20)\n"
            "planted = st.synthetic_control(0.15, n_seeds=20)\n"
            "print('null world   : mean FM slope %+.4f  (|t|>=2 in %.0f%% of seeds)'\n"
            "      % (null['mean_slope'], null['reject_rate']*100))\n"
            "print('planted world: mean FM slope %+.4f  (|t|>=2 in %.0f%% of seeds)'\n"
            "      % (planted['mean_slope'], planted['reject_rate']*100))"
        ),
        md(f"## 4. The honest verdict\n\n"
           f"The offer price *feels* like an anchor, and the below-offer gap even points the "
           f"predicted way — but on this curated sample **neither leg clears |t| ≥ 2** "
           f"(anchoring NW *t* = **{R['anchor_t']:+.2f}**, drag NW *t* = **{R['below_t']:+.2f}**), "
           f"the permutation placebo can't tell the slope from noise (two-sided *p* = "
           f"{R['plac_p_two']:.2f}), and the tradable book earns a coin-flip gross that dies "
           f"after borrow and costs. **Signal: None** (underpowered, one-cohort), "
           f"**Tradability: Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 874 — IPO-Price Anchoring — the teardown\n\n"
           "The Fama-MacBeth anchoring slope, the below-offer basket spread, the "
           "1,000-permutation placebo, the two-era cut, the costed timer, and the 20-seed "
           "synthetic control. Real-tape numbers are the frozen headline (`docs/results.md`)."),
        code("R = %r" % (R,)),
        md("## Data stamp"),
        code(
            "print(f\"{R['n_names']} curated listings ({R['n_ipo']} IPOs + {R['n_direct']} direct), \"\n"
            "      f\"vs SPY, {R['start']} -> {R['end']}\")\n"
            "print(f\"{R['active_months']} active months, {R['n_obs']} name-months, \"\n"
            "      f\"avg {R['avg_names']:.1f} names/month, below-offer share {R['below_share']:.1f}%\")\n"
            "print(f\"as-of {R['as_of']}   fingerprint {R['fingerprint']}\")"
        ),
        md("## Test 1 — the anchoring pull (Fama-MacBeth cross-sectional slope, HAC t)\n\n"
           "Forward market-adjusted return regressed on `gap = log(price/offer)` each month; "
           "average the monthly slopes; NW(6) *t*. Anchoring ⇒ negative slope."),
        code(
            "print(f\"all listings: slope {R['anchor_slope']:+.4f} ({R['anchor_bps10']:+.1f} bps/mo \"\n"
            "      f\"per +10% above offer)  NW t = {R['anchor_t']:+.2f}  \"\n"
            "      f\"({R['anchor_share_neg']}% of months negative, n={R['anchor_n']})\")\n"
            "print(f\"IPOs only  : slope {R['anchor_slope_ipo']:+.4f}  NW t = {R['anchor_t_ipo']:+.2f}\")"
        ),
        md("## Test 2 — the below-offer drag (below − above basket spread, HAC t)"),
        code(
            "print(f\"all listings: {R['below_bps']:+.2f} bps/mo ({R['below_ann']:+.2f}%/yr)  \"\n"
            "      f\"NW t = {R['below_t']:+.2f}  (Welch t = {R['below_welch']:+.2f}, n={R['below_n']})\")\n"
            "print(f\"   below-offer basket {R['below_leg']:+.2f} vs above-offer basket {R['above_leg']:+.2f} bps/mo\")\n"
            "print(f\"IPOs only  : {R['below_bps_ipo']:+.2f} bps/mo  NW t = {R['below_t_ipo']:+.2f}\")"
        ),
        md("## Placebo — shuffle gap→forward-return within each month (1,000 permutations)"),
        code(
            "print(f\"observed slope {R['plac_obs']:+.5f} vs placebo mean {R['plac_mean']:+.5f} \"\n"
            "      f\"(sd {R['plac_sd']:.5f}) over {R['plac_draws']:,} draws\")\n"
            "print(f\"left-tail p = {R['plac_p_left']:.3f}   two-sided p = {R['plac_p_two']:.3f}  \"\n"
            "      f\"-> indistinguishable from a random pairing\")"
        ),
        md("## Robustness — below-offer spread, two eras (split 2022-07)"),
        code(
            "print(f\"pre-2022-07: {R['era_early_bps']:+.2f} bps/mo  NW t = {R['era_early_t']:+.2f} (n={R['era_early_n']})\")\n"
            "print(f\"2022-07 on : {R['era_late_bps']:+.2f} bps/mo  NW t = {R['era_late_t']:+.2f} (n={R['era_late_n']})\")\n"
            "print('same (negative) sign both halves, insignificant in each -> one correlated cohort')"
        ),
        md("## The timer — SHORT below-offer / LONG above-offer, net of borrow + costs"),
        code(
            "for cb, br, g, gt, n, na, nt, dd in R['timer']:\n"
            "    print(f\"cost={cb:>4.1f} bps, borrow={br:.0f}%/yr: gross {g:+.2f} (t={gt:+.2f}) \"\n"
            "          f\"-> net {n:+.2f} bps/mo ({na:+.2f}%/yr, t={nt:+.2f})  max DD {dd:.1f}%\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the FM detector must NOT fire on the null and must recover a planted pull."),
        code(
            BOOT +
            "for edge in (0.0, 0.15):\n"
            "    r = st.synthetic_control(edge, n_seeds=20)\n"
            "    print(f\"edge={edge:.2f}: mean slope {r['mean_slope']:+.4f}  mean HAC t = {r['mean_t']:+.2f}  \"\n"
            "          f\"|t|>=2 rejection rate = {r['reject_rate']*100:.0f}%\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** Anchoring-pull FM slope **{R['anchor_slope']:+.4f}** "
           f"(NW *t* = **{R['anchor_t']:+.2f}**, placebo two-sided *p* = {R['plac_p_two']:.2f}); "
           f"below-offer drag **{R['below_bps']:+.1f} bps/mo** (NW *t* = **{R['below_t']:+.2f}**), "
           f"same sign in both eras (*t* = {R['era_early_t']:+.2f} / {R['era_late_t']:+.2f}) but "
           f"insignificant throughout. Both point the claimed way; neither clears |t| ≥ 2 — an "
           f"underpowered, one-cohort curated cross-section. The 20-seed synthetic control fires "
           f"on 0/20 nulls and 100% on a planted pull, so the flatness is real, not machinery.\n"
           f"- **Tradability — Mirage.** Short-below / long-above earns +{R['timer'][0][2]:.1f} "
           f"bps/mo gross at *t* = +{R['timer'][0][3]:.2f}; borrow + costs take net to "
           f"{R['timer'][0][4]:+.1f} → {R['timer'][1][4]:+.1f} bps/mo on a ~60% drawdown."),
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
