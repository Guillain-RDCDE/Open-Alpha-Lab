"""Generate the two narrative notebooks for Study 881 (Jobless-Claims Sector Rotation).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (FRED IC4WSA 4-week-MA
# claims + yfinance total-return sector ETFs, 1998-12-31 -> 2026-06-30; monthly frame;
# predictive NW regression of the cyclical-minus-defensive forward spread on the 4-week
# claims change).
R = dict(
    start="1998-12-31", end="2026-06-30", n_months=331, n=329, fingerprint="43e0a9ec1a15",
    spread_mean_pct=0.251, spread_sd_pct=4.37,
    slope=0.0177, t_nw=6.39, r2=0.0127, corr=0.1125,
    ex_covid_slope=0.0623, ex_covid_t=1.49, ex_covid_n=318,
    winsor_slope=0.0504, winsor_t=1.43,
    spearman_rho=0.0189, spearman_p=0.733,
    era_early_slope=0.1222, era_early_t=1.41, era_early_n=156,
    era_late_slope=0.0164, era_late_t=6.83, era_late_n=173,
    placebo_obs=0.0177, placebo_mean=-0.00003, placebo_sd=0.0089, placebo_p=0.053,
    timer0_gross=0.89, timer0_net=0.39, timer0_t=0.14,
    timer10_gross=0.89, timer10_net=-1.91, timer10_t=-0.66, n_switches=168,
    null_mean_t=0.16, null_sd_t=0.95, null_fire=1,
    planted_slope=-0.4651, planted_t=-13.85,
)


HEADER = f"""# Study 881 — Jobless-Claims Sector Rotation 📋

**Does a rise in initial jobless claims tilt the market from cyclicals to defensives?**

The nowcast folk-rule: rising **initial jobless claims** signal a cooling labour market,
so the risk-off playbook says rotate from **cyclicals** (XLY consumer-discretionary,
XLI industrials) into **defensives** (XLP staples, XLU utilities). We test the sharp
version — does the **4-week change in claims** *predict* the forward
**cyclical-minus-defensive** sector spread? — on a monthly frame ({R['start']} →
{R['end']}, {R['n_months']} months). This is a *rotation*, not a market-timer (that is
Study 385's question).

*Numbers below are the frozen headline (`docs/results.md`); the live cells run the fast
synthetic control. The sector ETFs trade continuously since 1998 — no survivorship in
the outcome; the one honest hazard is the 2020 outlier.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one line\n\n"
           "Claims come out weekly and lead the cycle; defensives out-earn into "
           "slowdowns. So a claims *uptick* should push the **cyclical − defensive** "
           "return spread **down** next month. That means a **negative** predictive "
           "slope. Let's see what the tape says."),
        code(
            "R = %r\n"
            "print('predictive slope of (cyclical - defensive) spread on the 4-week claims change:')\n"
            "print('  slope %%+.4f   NW(6) t = %%+.2f   R2 = %%.4f   (n=%%d)' %% (R['slope'], R['t_nw'], R['r2'], R['n']))\n"
            "need = 'negative'; got = 'NEGATIVE' if R['slope'] < 0 else 'POSITIVE (wrong sign!)'\n"
            "print('  claim needs a %%s slope; observed is %%s' %% (need, got))"
            % (R,)
        ),
        md("## 2. The catch — it is *one* month (2020)\n\n"
           "The fitted slope is not just wrong-signed, it is a mirage of a single "
           "episode. In 2020 the claims 4-week MA exploded (211k → 4,174k) exactly as "
           "the market bottomed and cyclicals ripped off that bottom — a few enormous "
           "same-signed points that drag the regression line."),
        code(
            "print('full sample : slope %+.4f   NW t = %+.2f' % (R['slope'], R['t_nw']))\n"
            "print('ex-COVID 2020: slope %+.4f   NW t = %+.2f  (n=%d)' % (R['ex_covid_slope'], R['ex_covid_t'], R['ex_covid_n']))\n"
            "print('winsor 1/99  : slope %+.4f   NW t = %+.2f' % (R['winsor_slope'], R['winsor_t']))\n"
            "print('Spearman rank: rho %+.4f   p = %.3f  (outlier-robust -> ~zero)' % (R['spearman_rho'], R['spearman_p']))"
        ),
        md("## 3. Is the sort just lucky? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world (`edge>0`: rising claims really "
           "do knock cyclicals down) and check the detector recovers the **negative** "
           "slope — and stays *silent* on the null (`edge=0`, claims move but predict "
           "nothing). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from claims_nowcast import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic_frame(edge=0.0, seed=881, n_months=360))\n"
            "planted = st.synthetic_detect(data.synthetic_frame(edge=0.5, seed=881, n_months=360))\n"
            "print('null world   : slope NW t = %+.2f  (should be ~0)' % null['t_nw'])\n"
            "print('planted world: slope NW t = %+.2f  (should light up NEGATIVE)' % planted['t_nw'])"
        ),
        md(f"## 4. The honest verdict\n\n"
           f"On the real tape the predictive slope is **wrong-signed and significant** "
           f"(**{R['slope']:+.4f}**, NW *t* = **{R['t_nw']:+.2f}**) — but that "
           f"significance is a **single-outlier (COVID-2020)** artefact: it collapses to "
           f"*t* ≈ {R['ex_covid_t']:.1f} once 2020 is dropped, to a Spearman ρ = "
           f"{R['spearman_rho']:+.2f} (p = {R['spearman_p']:.2f}), and to *t* = "
           f"{R['era_early_t']:+.2f} in the pre-2020 era. The seeded control recovers a "
           f"*planted* rotation cleanly, so the engine is fine — the effect simply is "
           f"not there. Rising claims carry **no** robust cyclical-vs-defensive rotation "
           f"signal. **Signal: None. Tradability: Mirage.**"),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 881 — Jobless-Claims Sector Rotation — the teardown\n\n"
           "The predictive Newey-West slope, the COVID-sensitivity / winsor / Spearman "
           "triple, the two-era cut, the permutation placebo, the costed rotation timer, "
           "and the 20-seed synthetic control."),
        code("R = %r" % (R,)),
        md("## The headline — predictive regression  `spread_{t+1} ~ dclaims_t`\n\n"
           "Newey-West(6) regression of the forward **cyclical − defensive** spread on "
           "the 4-week claims change. The claim needs a **negative** slope."),
        code(
            "print(f\"slope        : {R['slope']:+.4f}   NW(6) t = {R['t_nw']:+.2f}   R2 = {R['r2']:.4f}\")\n"
            "print(f\"correlation  : {R['corr']:+.4f}   (n={R['n']})\")\n"
            "print(f\"spread mean  : {R['spread_mean_pct']:+.3f}%/mo (sd {R['spread_sd_pct']:.2f}%)\")\n"
            "print('sign check   :', 'NEGATIVE (claim)' if R['slope']<0 else 'POSITIVE -> wrong sign -> None')"
        ),
        md("## It is one outlier — the 2020 claims spike"),
        code(
            "print(f\"full sample   : slope {R['slope']:+.4f}  NW t = {R['t_nw']:+.2f}\")\n"
            "print(f\"ex-COVID 2020 : slope {R['ex_covid_slope']:+.4f}  NW t = {R['ex_covid_t']:+.2f}  (n={R['ex_covid_n']})\")\n"
            "print(f\"winsor 1/99   : slope {R['winsor_slope']:+.4f}  NW t = {R['winsor_t']:+.2f}\")\n"
            "print(f\"Spearman rank : rho {R['spearman_rho']:+.4f}  p = {R['spearman_p']:.3f}\")"
        ),
        md("## Robustness — two eras (split 2012-01-01)"),
        code(
            "print(f\"1999-2011 (n={R['era_early_n']}): slope {R['era_early_slope']:+.4f}  NW t = {R['era_early_t']:+.2f}\")\n"
            "print(f\"2012-2026 (n={R['era_late_n']}): slope {R['era_late_slope']:+.4f}  NW t = {R['era_late_t']:+.2f}  (<- contains 2020)\")"
        ),
        md("## Placebo — shuffle the claims change vs the forward spread (2,000 draws)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.4f} vs placebo mean {R['placebo_mean']:+.5f} \"\n"
            "      f\"(sd {R['placebo_sd']:.4f}) -> two-sided p = {R['placebo_p']:.3f}\")"
        ),
        md("## The timer — can you get paid for the rotation?\n\n"
           "Flip a long-short cyclical/defensive book with the claim's sign; one-way × "
           "NAV per leg + 50 bps/yr borrow on the short."),
        code(
            "for tag,g,n,t in [('0 bp',R['timer0_gross'],R['timer0_net'],R['timer0_t']),\n"
            "                  ('10 bp',R['timer10_gross'],R['timer10_net'],R['timer10_t'])]:\n"
            "    print(f\"{tag:>5} one-way: gross {g:+.2f}%/yr -> net {n:+.2f}%/yr (t_net {t:+.2f})\")\n"
            "print(f\"({R['n_switches']} switches over the sample)\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the detector must NOT fire on the null and must recover a planted "
           "**negative** slope."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from claims_nowcast import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic_frame(edge=0.0, seed=881+s, n_months=360))['t_nw'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic_frame(edge=0.5, seed=881, n_months=360))\n"
            "print(f\"planted (edge=0.5): slope {planted['slope']:+.4f}, NW t = {planted['t_nw']:+.2f}\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The claimed labour-nowcast rotation does not exist. The "
           f"predictive slope is **wrong-signed** (positive, {R['t_nw']:+.2f} *t*), and "
           f"even that is a **single-outlier (COVID-2020)** artefact: *t* ≈ "
           f"{R['ex_covid_t']:.1f} ex-COVID / {R['winsor_t']:.1f} winsorised, Spearman "
           f"ρ = {R['spearman_rho']:+.2f} (p = {R['spearman_p']:.2f}), pre-2020 era "
           f"*t* = {R['era_early_t']:+.2f}, placebo p = {R['placebo_p']:.2f}. The 20-seed "
           f"synthetic control recovers a *planted* rotation (*t* = {R['planted_t']:+.2f}, "
           f"fires on {R['null_fire']}/20 nulls), so the engine is sound — the effect is "
           f"absent.\n"
           f"- **Tradability — Mirage.** The sign-correct rotation earns only "
           f"**{R['timer0_gross']:+.2f}%/yr gross** (*t* ≈ 0) and goes negative "
           f"(**{R['timer10_net']:+.2f}%/yr**) once its {R['n_switches']} flips pay a 10 bp "
           f"one-way cost."),
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
