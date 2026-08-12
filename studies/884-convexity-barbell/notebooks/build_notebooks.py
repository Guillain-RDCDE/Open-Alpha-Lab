"""Generate the two narrative notebooks for Study 884 (Convexity Barbell).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily
# total-return closes for SHY/IEF/TLT/BIL, 2010-01-04 -> 2026-06-30; trailing-252d empirical
# duration; duration-matched SHY+TLT barbell vs the IEF bullet).
R = dict(
    start="2010-01-04", end="2026-06-30", n_bonds=3, n_days=3894, rows=4147,
    fingerprint="32356eb6aefe",
    beta_shy=0.116, beta_ief=0.864, beta_tlt=2.020,
    w_short=0.605, w_long=0.395,
    barbell_ann=2.15, barbell_vol=6.41, barbell_mdd=-23.5,
    bullet_ann=2.28, bullet_vol=6.52, bullet_mdd=-23.9, corr=0.944,
    spread_bps=-0.054, t_nw=-0.27, t_1s=-0.25, spread_sharpe=-0.06,
    boot_lo=-0.433, boot_hi=0.344,
    sharpe_barbell_x=0.147, sharpe_bullet_x=0.165, sharpe_adv=-0.018, welch_t=-0.06,
    resid_dur_slope=0.009, conv_slope=-0.222, spread_conv_bps=-0.047, spread_carry_bps=-0.007,
    smile=[0.305, 0.014, 0.053, -0.693, 0.052],
    y2021_bar=-1.46, y2021_bul=-3.33, y2021_sp=1.89,
    y2022_bar=-15.40, y2022_bul=-15.16, y2022_sp=-0.38,
    y2025_bar=4.79, y2025_bul=8.03, y2025_sp=-3.06,
    era_early_bps=0.034, era_early_t=0.11, era_early_n=1760,
    era_late_bps=-0.126, era_late_t=-0.47, era_late_n=2134,
    placebo_obs=-0.054, placebo_mean=-0.129, placebo_sd=0.061, placebo_p=0.122,
    timer_05_net=-0.056, timer_1_net=-0.057, timer_2_net=-0.061,
    timer_1_t=-0.26, timer_1_ann=-0.14, turnover=0.0037,
    null_mean_t=0.01, null_sd_t=0.73, null_fire=0,
    planted_t=4.30, planted_conv_slope=0.70,
)


HEADER = f"""# Study 884 — Convexity Barbell 🏋️

**Does a duration-matched SHY+TLT barbell out-earn the IEF bullet on its extra convexity?**

Textbook fixed income (Fabozzi; Ilmanen): a **barbell** (short + long ends) weighted to the
same **duration** as a **bullet** (the belly) carries more **convexity**, so the second-order
term `+½·C·(Δy)²` should make it out-earn the bullet whenever yields move a lot. We rebuild
it from three iShares Treasury ETFs + a cash leg —
**bullet = IEF (7-10y)**, **barbell = {R['w_short']:.3f}·SHY (1-3y) + {R['w_long']:.3f}·TLT (20y+)**,
duration-matched each day — over {R['start']} → {R['end']} ({R['rows']:,} rows).

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. Fingerprint `{R['fingerprint']}`.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one picture\n\n"
           "Two Treasury books with the **same duration** (same first-order rate exposure): "
           "the **bullet** just holds the belly (IEF); the **barbell** holds the short and "
           "long ends (SHY + TLT) weighted to match. Because convexity grows with the square "
           "of maturity, spreading out to the wings gives the barbell **more curvature** — "
           "so on a big yield move `−D·Δy + ½·C·(Δy)²` should leave the barbell ahead, up or "
           "down. That is the textbook 'barbells are convex' free lunch. The catch the desk "
           "tests: the market makes you *pay* for convexity with a lower yield, and the "
           "barbell carries curve-reshaping (butterfly) risk the bullet doesn't."),
        code(
            "R = dict(w_short=%r, w_long=%r, barbell_ann=%r, bullet_ann=%r, barbell_vol=%r,\n"
            "         bullet_vol=%r, spread_bps=%r, t_nw=%r, sharpe_adv=%r, corr=%r)\n"
            "print('duration-matched barbell = %%.3f*SHY + %%.3f*TLT   vs   bullet = IEF'\n"
            "      %% (R['w_short'], R['w_long']))\n"
            "print('  barbell : %%+.2f%%%%/yr  vol %%.2f%%%%' %% (R['barbell_ann'], R['barbell_vol']))\n"
            "print('  bullet  : %%+.2f%%%%/yr  vol %%.2f%%%%  (corr %%.3f)' %% (R['bullet_ann'], R['bullet_vol'], R['corr']))\n"
            "print('  spread  : %%+.3f bps/day  (NW t = %%+.2f, Sharpe advantage %%+.3f)'\n"
            "      %% (R['spread_bps'], R['t_nw'], R['sharpe_adv']))"
            % (R["w_short"], R["w_long"], R["barbell_ann"], R["bullet_ann"], R["barbell_vol"],
               R["bullet_vol"], R["spread_bps"], R["t_nw"], R["sharpe_adv"], R["corr"])
        ),
        md("## 2. Is the detector any good? A live synthetic control\n\n"
           "We plant an **under-priced** convexity in a seeded toy curve (`edge>0` ⇒ the "
           "barbell's extra convexity is a genuine net pickup) and check the spread lights "
           "up — and stays *silent* on the null (`edge=0`, convexity present but exactly "
           "paid for by a yield give-up). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from barbell import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=884, n_days=1300))\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.6, seed=884, n_days=1800))\n"
            "print('null world   : spread NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: spread NW t = %+.2f  (should light up)' % planted['t_nw'])\n"
            "print('convexity slope > 0 in BOTH worlds (structural): null %+.2f / planted %+.2f'\n"
            "      % (null['conv_slope'], planted['conv_slope']))"
        ),
        md("## 3. The honest verdict — the free lunch isn't free\n\n"
           f"On the real Treasury tape the duration-matched barbell earns "
           f"**{R['barbell_ann']:+.2f}%/yr** vs the bullet's **{R['bullet_ann']:+.2f}%** at "
           f"the same vol — it *under*-earns. The daily spread is **{R['spread_bps']:+.3f} "
           f"bps** (Newey-West *t* = **{R['t_nw']:+.2f}**), its bootstrap CI straddles zero "
           f"(**[{R['boot_lo']:+.2f}, {R['boot_hi']:+.2f}]**), and the excess-vs-excess "
           f"Sharpe advantage is **{R['sharpe_adv']:+.2f}**. Two tells seal it:\n\n"
           f"1. **The convexity is invisible in total return.** The `f²` slope is "
           f"*wrong-signed* (**{R['conv_slope']:+.2f}**) and the convexity smile is absent — "
           f"the barbell does **not** systematically win when yields move most.\n"
           f"2. **2022 — the biggest move — went the wrong way.** In the historic selloff the "
           f"barbell **{R['y2022_bar']:+.2f}%** *lost* to the bullet **{R['y2022_bul']:+.2f}%** "
           f"(spread {R['y2022_sp']:+.2f}%): exactly the scenario the claim needs, and it "
           f"failed.\n\n"
           "A barbell really is more convex — but the market prices that convexity into the "
           "wings' lower yield and charges butterfly (curve-reshaping) risk the bullet avoids. "
           "**Signal: None**, **Tradability: Mirage**."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 884 — Convexity Barbell — the teardown\n\n"
           "The duration ladder & match, the spread's Newey-West *t* and bootstrap CI, the "
           "convexity regression + smile, the 2022 tell, the two-era cut, the leg-permutation "
           "placebo, the costed timer, and the 20-seed synthetic control."),
        code("R = %r" % (R,)),
        md("## The duration ladder & the match\n\n"
           "Each bond's empirical duration = its trailing-252d beta to the equal-weight rates "
           "factor; solving `w·β_SHY + (1-w)·β_TLT = β_IEF` gives the barbell weight."),
        code(
            "print('empirical durations: SHY %.3f  IEF %.3f  TLT %.3f'\n"
            "      % (R['beta_shy'], R['beta_ief'], R['beta_tlt']))\n"
            "print('=> barbell = %.3f*SHY + %.3f*TLT (duration-matched to IEF)'\n"
            "      % (R['w_short'], R['w_long']))\n"
            "print('spread residual duration slope on the factor = %+.4f (~0 => matched)'\n"
            "      % R['resid_dur_slope'])"
        ),
        md("## The headline — barbell vs bullet, total return"),
        code(
            "print(f\"barbell : {R['barbell_ann']:+.2f}%/yr  vol {R['barbell_vol']:.2f}%  maxDD {R['barbell_mdd']:.1f}%\")\n"
            "print(f\"bullet  : {R['bullet_ann']:+.2f}%/yr  vol {R['bullet_vol']:.2f}%  maxDD {R['bullet_mdd']:.1f}%  (corr {R['corr']:.3f})\")\n"
            "print(f\"spread  : {R['spread_bps']:+.3f} bps/day  NW(10) t = {R['t_nw']:+.2f}  one-sample t = {R['t_1s']:+.2f}  Sharpe = {R['spread_sharpe']:+.3f}\")\n"
            "print(f\"bootstrap spread-mean CI95 = [{R['boot_lo']:+.3f}, {R['boot_hi']:+.3f}] bps (straddles zero)\")\n"
            "print(f\"excess-vs-excess Sharpe: barbell {R['sharpe_barbell_x']:+.3f} vs bullet {R['sharpe_bullet_x']:+.3f} -> advantage {R['sharpe_adv']:+.3f} (Welch t = {R['welch_t']:+.2f})\")"
        ),
        md("## Convexity — the `f²` regression and the smile\n\n"
           "Regress the spread on `[1, f, f²]`. The claim needs a **positive** `f²` slope "
           "(barbell captures its extra convexity); the tape gives a *wrong-signed* one, and "
           "the smile (mean spread by |move| quintile) shows no monotone rise into big moves."),
        code(
            "print(f\"convexity slope on f^2 = {R['conv_slope']:+.3f}  (claim: > 0 -> WRONG-SIGNED)\")\n"
            "print(f\"mean-spread split: convexity {R['spread_conv_bps']:+.4f} + carry/drift {R['spread_carry_bps']:+.4f} bps\")\n"
            "labels=['small','.','.','.','big']\n"
            "print('smile (mean spread bps by |move| quintile):')\n"
            "for lab,v in zip(labels, R['smile']): print(f'   {lab:>5}: {v:+.3f}')"
        ),
        md("## The 2022 tell — the biggest rate move in the sample\n\n"
           "The claim says the barbell wins when yields move a lot. In 2022 — the historic "
           "selloff — it **lost** to the bullet."),
        code(
            "for y,bar,bul,sp in [(2021,R['y2021_bar'],R['y2021_bul'],R['y2021_sp']),\n"
            "                     (2022,R['y2022_bar'],R['y2022_bul'],R['y2022_sp']),\n"
            "                     (2025,R['y2025_bar'],R['y2025_bul'],R['y2025_sp'])]:\n"
            "    print(f'{y}: barbell {bar:+.2f}%  bullet {bul:+.2f}%  spread {sp:+.2f}%')"
        ),
        md("## Robustness — two eras (split 2018-01-01)"),
        code(
            "print(f\"2010-2017 (n={R['era_early_n']}): {R['era_early_bps']:+.3f} bps  NW t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2018-2026 (n={R['era_late_n']}): {R['era_late_bps']:+.3f} bps  NW t = {R['era_late_t']:+.2f}\")"
        ),
        md("## Placebo — permute the two barbell legs in time\n\n"
           "Break the day-by-day alignment of the two legs; the observed spread should sit "
           "inside the placebo cloud (no convexity-alignment signal)."),
        code(
            "print(f\"observed {R['placebo_obs']:+.3f} bps vs placebo mean {R['placebo_mean']:+.3f} \"\n"
            "      f\"(sd {R['placebo_sd']:.3f}) -> right-tail p = {R['placebo_p']:.3f}\")"
        ),
        md("## The timer — nothing to harvest, costs only subtract\n\n"
           "The barbell turns over slowly, so frictions are tiny — but the gross spread is "
           "already ≈ 0, so the net is negative at every cost level."),
        code(
            "for tag,net in [('0.5 bp',R['timer_05_net']),('1 bp',R['timer_1_net']),('2 bps',R['timer_2_net'])]:\n"
            "    print(f\"{tag:>6}: net {net:+.3f} bps/day\")\n"
            "print(f\"turnover {R['turnover']:.4f}/day, net t = {R['timer_1_t']:+.2f}, ~{R['timer_1_ann']:+.2f}%/yr\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null (convexity fairly priced) and must "
           "recover a planted under-priced convexity — while the convexity slope is positive "
           "in *both* worlds (convexity is structural, an edge only when under-priced)."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from barbell import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_panel(edge=0.0, seed=884+s, n_days=1300))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: spread NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_panel(edge=0.6, seed=884, n_days=1800))\n"
            "print(f\"planted (edge=0.6): spread NW t = {planted['t_nw']:+.2f}, convexity slope = {planted['conv_slope']:+.3f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** A duration-matched SHY+TLT barbell does **not** out-earn the "
           f"IEF bullet: **{R['barbell_ann']:+.2f}%/yr vs {R['bullet_ann']:+.2f}%**, spread "
           f"**{R['spread_bps']:+.3f} bps** (NW *t* = **{R['t_nw']:+.2f}**), bootstrap CI "
           f"**[{R['boot_lo']:+.2f}, {R['boot_hi']:+.2f}]** straddling zero, Sharpe advantage "
           f"**{R['sharpe_adv']:+.2f}**, flat in both eras. The convexity is genuine but "
           f"invisible in total return (`f²` slope {R['conv_slope']:+.2f}, no smile, and 2022 "
           f"went the wrong way). The 20-seed synthetic control recovers a *planted* edge "
           f"cleanly (*t* = {R['planted_t']:+.1f}, 0/20 nulls fire), so the machinery is sound "
           f"— the net edge is simply absent.\n"
           f"- **Tradability — Mirage.** No gross edge to cost; the spread is ≈ 0 before "
           f"frictions and negative after (net ~{R['timer_1_ann']:+.2f}%/yr). The free "
           f"convexity lunch is fully offset by carry give-up and curve risk."),
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
