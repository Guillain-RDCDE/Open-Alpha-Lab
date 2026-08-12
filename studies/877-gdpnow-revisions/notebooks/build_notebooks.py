"""Generate the two narrative notebooks for Study 877 (GDPNow Revisions).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (Atlanta Fed GDPNow daily
# nowcast history 2011-2026, within-quarter revision; SPY total-return; lag-0 headline).
R = dict(
    start="2011-08-30", end="2026-06-26", n=2042, n_forecasts=2102, n_qtr=60,
    fingerprint="6161199ecd72",
    b1=-3.22, t1=-1.02, r2_1=0.043,
    b5=1.47, t5=0.15, r2_5=0.002,
    b1_l1=3.27, t1_l1=0.89, b5_l1=-3.08, t5_l1=-0.36,
    base_bps=5.56, up_thr=0.300, down_thr=-0.350,
    up_n=205, up_bps=-18.71, up_t=-2.74,
    down_n=205, down_bps=0.09, down_t=0.01, updown_welch=-1.64,
    up_l1_bps=11.24, up_l1_t=1.33,
    era_e_b=1.99, era_e_t=0.19, era_e_n=918,
    era_l_b=-3.57, era_l_t=-1.08, era_l_n=1124,
    placebo_obs=-3.22, placebo_sd=3.50, placebo_p=0.316,
    timer1_net=0.36, timer1_ann=0.9, timer1_sharpe=0.07,
    timer5_net=-1.68, timer5_ann=-4.2, timer5_sharpe=-0.34,
    bh_sharpe=0.80, exposure=0.48,
    null_mean_t=-0.18, null_sd_t=1.01, null_fire=0,
    planted_t=6.65, planted_b=41.1, planted_r2=2.14,
)


HEADER = f"""# Study 877 — GDPNow Revisions 📉

**The Atlanta Fed's GDPNow nowcast is revised almost every day as new data lands. Is that
daily *revision* a real-time growth surprise you can trade in SPY?**

The believers' story: an **upward** revision means the incoming data beat the model's running
estimate, so stocks should rise over the next day or week; a big **downward** revision should
precede weakness. We take GDPNow's full daily forecast history ({R['start']} → {R['end']},
{R['n_forecasts']:,} forecasts over {R['n_qtr']} quarters), form the within-quarter revision,
and regress forward SPY returns on it.

*Numbers below are the frozen headline (`docs/results.md`, fingerprint
`{R['fingerprint']}`); the live cells run the fast synthetic control. The nowcast posts
intraday, so the headline is the **most generous** execution (trade the release-day close).*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The idea in one line\n\n"
           "GDPNow is a model that keeps a running guess of this quarter's GDP growth and "
           "nudges it every time a new report (jobs, ISM, retail sales) comes out. The **size "
           "of the nudge** — today's nowcast minus yesterday's — is a clean, real-time "
           "*growth surprise*. If markets hadn't already priced that surprise, an up-nudge "
           "should be followed by higher stocks. Should."),
        code(
            "R = dict(b1=%r, t1=%r, r2_1=%r, up_bps=%r, up_t=%r, down_bps=%r, down_t=%r)\n"
            "print('predictive slope (1-day fwd SPY on the revision):')\n"
            "print('  %%+.2f bps per 1pp of revision   NW t = %%+.2f   R2 = %%.3f%%%%'\n"
            "      %% (R['b1'], R['t1'], R['r2_1']))\n"
            "print('after the BIGGEST UP revisions  : %%+.2f bps next day (t = %%+.2f)'\n"
            "      %% (R['up_bps'], R['up_t']))\n"
            "print('after the BIGGEST DOWN revisions: %%+.2f bps next day (t = %%+.2f)'\n"
            "      %% (R['down_bps'], R['down_t']))"
            % (R["b1"], R["t1"], R["r2_1"], R["up_bps"], R["up_t"], R["down_bps"], R["down_t"])
        ),
        md("## 2. What the tape says\n\n"
           f"The predictive slope is **insignificant** (NW *t* = {R['t1']:+.2f}) and the *R²* "
           f"is a rounding error (**{R['r2_1']:.3f}%**). The one number that *is* significant "
           f"goes the **wrong way**: after the biggest **up**-revisions SPY *falls* the next "
           f"day (**{R['up_bps']:+.2f} bps**, *t* = {R['up_t']:+.2f} — a 'sell the good news' "
           f"blip), while big **down**-revisions are dead flat ({R['down_bps']:+.2f} bps, "
           f"*t* = {R['down_t']:+.2f}). So the claim fails on both halves: up-revisions don't "
           f"predict strength, and down-revisions don't precede weakness."),
        md("## 3. Is the sort just lucky? A live synthetic control\n\n"
           "We plant the effect in a seeded toy world (`edge>0`: an up-revision really does "
           "lift the next-day return) and check the regression recovers it — and stays "
           "*silent* on the null (`edge=0`, revisions present but unpriced). No network."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from gdpnow import data, strategy as st\n"
            "null = st.synthetic_detect(data.synthetic(edge=0.0, seed=877, n=2000))\n"
            "planted = st.synthetic_detect(data.synthetic(edge=0.005, seed=877, n=2000))\n"
            "print('null world   : slope NW t = %+.2f  (should be ~0)' % null['t'])\n"
            "print('planted world: slope NW t = %+.2f  (should light up)' % planted['t'])"
        ),
        md("## 4. The honest verdict\n\n"
           f"On {R['n']:,} genuine daily revisions the GDPNow nowcast revision **does not "
           f"predict** forward SPY: the slope is insignificant (*t* = {R['t1']:+.2f}), it "
           f"**flips sign** the moment you can't trade the release-day close, and the only "
           f"significant piece is *wrong-signed* and fragile. A timer that buys after "
           f"up-revisions earns a Sharpe of **{R['timer1_sharpe']:.2f}** at 1 bp cost — versus "
           f"**{R['bh_sharpe']:.2f}** for just holding SPY. **Signal: None. Tradability: "
           f"Mirage.** The revision is a real-time *restatement* of news the tape already "
           f"absorbed minutes earlier — not a forecast of tomorrow."),
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 877 — GDPNow Revisions — the teardown\n\n"
           "The predictive regression with a Newey-West slope *t*, the top/bottom-decile "
           "conditional test, the execution-lag robustness, the two-era cut, the permutation "
           "placebo, the costed timer, and the 20-seed synthetic control."),
        code("R = %r" % (R,)),
        md("## The headline — forward SPY return on the revision (HAC t)"),
        code(
            "print(f\"1-day fwd: beta {R['b1']:+.2f} bps/pp  NW t = {R['t1']:+.2f}  \"\n"
            "      f\"R2 = {R['r2_1']:.3f}%  (n = {R['n']})\")\n"
            "print(f\"5-day fwd: beta {R['b5']:+.2f} bps/pp  NW t = {R['t5']:+.2f}  \"\n"
            "      f\"R2 = {R['r2_5']:.3f}%\")\n"
            "print(f\"lag-1 robustness: 1-day beta {R['b1_l1']:+.2f} (t {R['t1_l1']:+.2f}) \"\n"
            "      f\"-> SIGN FLIPS vs lag 0: no stable slope\")"
        ),
        md("## Decile conditional — biggest up vs biggest down revisions (fwd 1-day)\n\n"
           "The claim: up-revisions → strength, down-revisions → weakness. The tape says the "
           "reverse-or-nothing."),
        code(
            "print(f\"base forward 1-day: {R['base_bps']:+.2f} bps\")\n"
            "print(f\"top-decile UP  (rev>={R['up_thr']:+.3f}): n={R['up_n']}  \"\n"
            "      f\"{R['up_bps']:+.2f} bps  NW t = {R['up_t']:+.2f}\")\n"
            "print(f\"bot-decile DOWN(rev<={R['down_thr']:+.3f}): n={R['down_n']}  \"\n"
            "      f\"{R['down_bps']:+.2f} bps  NW t = {R['down_t']:+.2f}\")\n"
            "print(f\"up-minus-down Welch t = {R['updown_welch']:+.2f}\")\n"
            "print(f\"[lag 1] top-decile UP flips to {R['up_l1_bps']:+.2f} bps \"\n"
            "      f\"(t = {R['up_l1_t']:+.2f}) -> fragile intraday artefact\")"
        ),
        md("## Robustness — two eras (split 2019-01-01)"),
        code(
            "print(f\"2011-2018 (n={R['era_e_n']}): beta {R['era_e_b']:+.2f} bps  NW t = {R['era_e_t']:+.2f}\")\n"
            "print(f\"2019-2026 (n={R['era_l_n']}): beta {R['era_l_b']:+.2f} bps  NW t = {R['era_l_t']:+.2f}\")\n"
            "print('the (already-insignificant) slope changes sign across halves')"
        ),
        md("## Placebo — shuffle forward returns against revisions (5,000 draws)"),
        code(
            "print(f\"observed {R['placebo_obs']:+.2f} bps vs shuffled sd {R['placebo_sd']:.2f} \"\n"
            "      f\"-> two-sided p = {R['placebo_p']:.3f}\")"
        ),
        md("## The timer — long SPY one day after an up-revision, flat otherwise"),
        code(
            "for tag,net,ann,shp in [('1 bp',R['timer1_net'],R['timer1_ann'],R['timer1_sharpe']),\n"
            "                        ('5 bps',R['timer5_net'],R['timer5_ann'],R['timer5_sharpe'])]:\n"
            "    print(f\"{tag:>5} cost: net {net:+.2f} bps/day ({ann:+.1f}%/yr, Sharpe {shp:+.2f})\")\n"
            "print(f\"vs buy-and-hold Sharpe {R['bh_sharpe']:.2f} on the same dates \"\n"
            "      f\"(rule in the market only {R['exposure']*100:.0f}% of the time)\")"
        ),
        md("## Synthetic positive control — the machinery is unbiased\n\n"
           "Live: the regression must NOT fire on the null and must recover a planted "
           "revision→return edge."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from gdpnow import data, strategy as st\n"
            "null_t = np.array([st.synthetic_detect(data.synthetic(edge=0.0, seed=877+s, n=1500))['t'] for s in range(8)])\n"
            "print(f\"null (edge=0), 8 seeds: NW t mean {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t)>=2).sum()}/8\")\n"
            "planted = st.synthetic_detect(data.synthetic(edge=0.005, seed=877, n=2000))\n"
            "print(f\"planted (edge=0.005): beta {planted['beta_bps']:+.1f} bps, NW t = {planted['t']:+.2f}, R2 = {planted['r2']*100:.2f}%\")"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** The GDPNow revision does **not** predict forward SPY with the "
           f"claimed sign: 1-day slope NW *t* = **{R['t1']:+.2f}**, *R²* = **{R['r2_1']:.3f}%**, "
           f"sign-flipping under a one-day lag and across eras (*t* = {R['era_e_t']:+.2f} / "
           f"{R['era_l_t']:+.2f}), inside a permutation placebo (*p* = {R['placebo_p']:.2f}). The "
           f"only significant piece — top-decile **up**-revisions preceding next-day "
           f"**weakness** ({R['up_bps']:+.2f} bps, *t* = {R['up_t']:+.2f}) — is **wrong-signed** "
           f"vs the claim and flips to {R['up_l1_bps']:+.2f} bps once you can't trade the "
           f"release-day close. Down-revisions are flat (*t* = {R['down_t']:+.2f}). The 20-seed "
           f"synthetic control fires on {R['null_fire']}/20 nulls and recovers a planted edge "
           f"(*t* = {R['planted_t']:+.2f}), so the flat real result is genuine, not a bug.\n"
           f"- **Tradability — Mirage.** An up-revision timer earns Sharpe "
           f"**{R['timer1_sharpe']:.2f}** at 1 bp (vs {R['bh_sharpe']:.2f} buy-and-hold) and "
           f"goes negative ({R['timer5_sharpe']:+.2f}) at 5 bps."),
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
