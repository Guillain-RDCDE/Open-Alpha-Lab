"""Generate the two narrative notebooks for Study 787 (Heatwave-Utilities).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached XLU/SPY
tapes under ../_cache/ (fetching once on a cache miss) and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive control
runs anywhere with no network.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (XLU + SPY, yfinance,
# 1998-12-22 -> 2026-06-30; 27 of 27 summers resolved). Fingerprint 7e03507631b7.
R = dict(
    n_events=27, n_included=27, fp="7e03507631b7", rows=6921,
    pre_s_mean=-0.216, pre_s_t=-0.410, pre_s_hit=13, pre_s_n=27,
    pre_l_mean=-0.194, pre_l_t=-0.331, pre_l_hit=15,
    post_s_mean=+0.367, post_s_t=+0.559, post_s_hit=14,
    post_l_mean=+0.718, post_l_t=+0.814, post_l_hit=14,
    pl_pre_p=0.6228, pl_post_p=0.2535, pl_mean=-0.032, pl_sd=0.604,
    jk_lo=+0.365, jk_hi=+1.092,
    post_l_net5=+0.618, post_l_t5=+0.70, post_l_net10=+0.518, post_l_t10=+0.59,
    null_mean_t=+0.20, null_sd_t=0.94, null_hits=1,
    planted1_mean=+1.570, planted1_t=+2.09, planted2_mean=+2.570, planted2_t=+3.43,
)

PRELUDE = """\
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from heatwave_utilities import data as dt, strategy as st

if not dt.have_real():
    print("cache miss -> fetching XLU + SPY once (needs network)")
    dt.fetch()
prices = dt.load_real()
ev = st.build_event_table(prices, cost_bps=5.0)
inc = ev[ev["included"]]
print(f"panel loaded; {len(inc)} of {len(dt.EVENTS)} summers resolved")
"""


def build_curious():
    cells = [
        md("# Study 787 — Heatwave-Utilities \U0001F321️\n\n"
           "*For the curious.* The dog-days of summer bake the country, air conditioners run "
           "flat out, and electricity demand hits its annual peak. Surely the **utilities** "
           "sector — the companies selling all that power — rallies through the heat? We put "
           "27 summers (1999→2025) on the stand and asked the tape one plain question: "
           "**does XLU beat the market across the peak-heat weeks?** The answer is a polite "
           "*no*."),
        md("## The claim, and why it's a clean test\n\n"
           "Peak US heat is **calendar-known**: because of the ~4-5 week *seasonal "
           "temperature lag*, the hottest average temperatures land in mid-to-late July, not "
           "on the June solstice (NOAA/NWS). So we anchor every year on a fixed **July 22** "
           "and hold XLU across the peak-heat window — zero look-ahead. We measure XLU's "
           "**abnormal** return (XLU − SPY, total-return) so we're not just measuring a "
           "summer that was good for *everything*."),
        code(PRELUDE),
        md("## The peak-heat calendar we test (fixed July-22 climatological centre)"),
        code("pd.DataFrame(dt.EVENTS, columns=['year', 'peak_heat_anchor'])"),
        md("## The picture: mean cumulative abnormal return around the peak\n\n"
           "Offset 0 is the July-22 peak-heat anchor. Left of zero is the *ramp into* the "
           "heat; right of zero is the *through / past-peak* window (late July into August)."),
        code(
            "car = st.car_path(ev, prices)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.axvline(0, color='0.4', lw=1, ls='--')\n"
            "ax.axhline(0, color='0.7', lw=0.8)\n"
            "ax.plot(car.index, car.values * 100, color='#e67e22', lw=2)\n"
            "ax.set_xlabel('trading days from the peak-heat anchor (0 = July 22)')\n"
            "ax.set_ylabel('mean cumulative AR, XLU − SPY (%)')\n"
            "ax.set_title('Utilities barely twitch through the peak of summer')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(f"## The numbers\n\n"
           f"| window | mean AR | *t* | hit rate |\n|---|--:|--:|--:|\n"
           f"| 2-week into-heat | {R['pre_s_mean']:+.2f}% | {R['pre_s_t']:+.2f} | {R['pre_s_hit']}/27 |\n"
           f"| 1-month into-heat | {R['pre_l_mean']:+.2f}% | {R['pre_l_t']:+.2f} | {R['pre_l_hit']}/27 |\n"
           f"| 2-week past-peak | {R['post_s_mean']:+.2f}% | {R['post_s_t']:+.2f} | {R['post_s_hit']}/27 |\n"
           f"| 1-month past-peak | **{R['post_l_mean']:+.2f}%** | **{R['post_l_t']:+.2f}** | {R['post_l_hit']}/27 |\n\n"
           f"The 'rally through the heat' window is *directionally* right — XLU beats SPY "
           f"by a whisker in the late-July-into-August month (+{R['post_l_mean']:.2f}%) — but "
           f"the *t* is only {R['post_l_t']:+.2f} and the hit rate is a coin-flip {R['post_l_hit']}/27. "
           f"That's noise, not a season."),
        code(
            "rows = [('2wk into-heat','pre_s'),('1mo into-heat','pre_l'),('2wk past-peak','post_s'),('1mo past-peak','post_l')]\n"
            "for label, col in rows:\n"
            "    s = st.one_sample_t(inc[col].values); hr = st.hit_rate(inc[col].values)\n"
            "    print(f'{label:<14s} n={s[\"n\"]:2d}  mean={s[\"mean\"]*100:+.3f}%  t={s[\"t\"]:+.3f}  hit {hr[\"k\"]}/{hr[\"n\"]}')"
        ),
        md("## So what?\n\n"
           "The 'hold utilities through the dog-days' trade is folklore. Yes, summer heat "
           "spikes electricity *load* — but regulated rates, fuel pass-throughs and an "
           "efficient market mean that well-known demand pattern never shows up as a tradable "
           "*stock* edge. The best window is +0.72% with *t* < 1 and dies the moment you charge "
           "costs. Verdict: **None signal, Mirage tradability.** The quants' notebook has the "
           "placebo, the jackknife, the costed leg and the synthetic control."),
    ]
    return new_notebook(cells=cells)


def build_quants():
    cells = [
        md("# Study 787 — Heatwave-Utilities — for the quants \U0001F52C\n\n"
           "The full battery: one-sample *t* per window, a random-window placebo, a "
           "leave-one-out jackknife, the costed net leg, and a seeded synthetic positive "
           "control. Everything offline once cached; fingerprint "
           f"`{R['fp']}`."),
        code(PRELUDE),
        md("## 1. One-sample *t* across independent summers\n\n"
           "Each summer is one independent event, so the unit is a one-sample *t* of the "
           "per-year abnormal return — **not** a daily panel (which would fake precision)."),
        code(
            "for label, col in [('2wk into-heat','pre_s'),('1mo into-heat','pre_l'),('2wk past-peak','post_s'),('1mo past-peak','post_l')]:\n"
            "    s = st.one_sample_t(inc[col].values)\n"
            "    print(f'{label:<14s} n={s[\"n\"]}  mean={s[\"mean\"]*100:+.3f}%  sd={s[\"sd\"]*100:.2f}%  t={s[\"t\"]:+.3f}')"
        ),
        md("## 2. Random-window placebo — is the past-peak bump inside the luck cloud?\n\n"
           "For each summer we redraw a random, non-peak 2-week window on XLU vs SPY and "
           "recompute the abnormal return; 20 seeds × 200 draws. If the observed mean sits "
           "in the tail of that null, it isn't ordinary tracking noise. Spoiler: it's centre-field."),
        code(
            "pl = st.placebo_pvalue(ev, prices, 'post_s', k=10, tail='right')\n"
            "import numpy as np\n"
            "xlu, spy = prices[dt.INSTRUMENT], prices[dt.BENCHMARK]\n"
            "common = xlu.index.intersection(spy.index).sort_values()\n"
            "rng = np.random.default_rng(999); draws = []\n"
            "for _ in range(4000):\n"
            "    vals = []\n"
            "    for _e in range(int(inc.shape[0])):\n"
            "        p = int(rng.integers(0, len(common)-11))\n"
            "        vals.append(float(xlu.loc[common[p+10]]/xlu.loc[common[p]] - spy.loc[common[p+10]]/spy.loc[common[p]]))\n"
            "    draws.append(np.mean(vals))\n"
            "fig, ax = plt.subplots(figsize=(9,4))\n"
            "ax.hist(np.array(draws)*100, bins=50, color='#8b949e', alpha=0.8)\n"
            "ax.axvline(pl['obs']*100, color='#e67e22', lw=2, label=f\"observed {pl['obs']*100:+.2f}%\")\n"
            "ax.set_xlabel('mean 2-week AR of random windows (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f\"past-peak window vs luck cloud (right-tail p={pl['p_value']:.3f})\"); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('placebo:', {k: round(v,4) if isinstance(v,float) else v for k,v in pl.items()})"
        ),
        md("## 3. Jackknife — is the (tiny) past-peak tilt one good year, or broad?"),
        code(
            "x = inc['post_l'].values\n"
            "jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "print(f'full-sample t = {st.one_sample_t(x)[\"t\"]:+.3f}')\n"
            "print(f'jackknife t range [{min(jk):+.3f}, {max(jk):+.3f}] over {len(x)} leave-one-out draws')"
        ),
        md("## 4. Tradability — net of costs\n\n"
           "Calendar-known entry, so the signal window and the tradable window are the same "
           "(gross vs net). The best cut is +0.72%/yr gross with *t* < 1 — and it only "
           "shrinks once you charge a couple of round-trips of cost."),
        code(
            "ev10 = st.build_event_table(prices, cost_bps=10.0); inc10 = ev10[ev10['included']]\n"
            "for base, label in [('post_l','1mo past-peak'),('pre_l','1mo into-heat')]:\n"
            "    g = st.one_sample_t(inc[base].values); n5 = st.one_sample_t(inc[base+'_net'].values); n10 = st.one_sample_t(inc10[base+'_net'].values)\n"
            "    print(f'{label:<14s} gross {g[\"mean\"]*100:+.3f}% (t={g[\"t\"]:+.2f})  net@5 {n5[\"mean\"]*100:+.3f}% (t={n5[\"t\"]:+.2f})  net@10 {n10[\"mean\"]*100:+.3f}% (t={n10[\"t\"]:+.2f})')"
        ),
        md("## 5. Synthetic positive control — the detector works, and the null is quiet\n\n"
           "The one-sample-*t* detector must stay quiet on a planted-null world and recover a "
           "planted into-the-heat bump. At n = 27 the null is well-behaved (|*t*| ≥ 2 on "
           "~1/20 seeds), and a planted ≥1% seasonal rally is caught cleanly — so the "
           "empty real result is a true absence, not a blind detector."),
        code(
            "null_ts = np.array([st.synthetic_detect(bump=0.0, seed=814+s, k=10)['t'] for s in range(20)])\n"
            "print(f'null: mean t={null_ts.mean():+.2f} sd={null_ts.std(ddof=1):.2f}  |t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "for b in (0.01, 0.02):\n"
            "    r = st.synthetic_detect(bump=b, seed=814, k=10)\n"
            "    print(f'planted +{b*100:.0f}%: mean AR {r[\"mean\"]*100:+.3f}%  t={r[\"t\"]:+.2f}')\n"
            "bumps = np.linspace(0, 0.03, 13)\n"
            "ts = [st.synthetic_detect(bump=b, seed=814, k=10)['t'] for b in bumps]\n"
            "fig, ax = plt.subplots(figsize=(8,4)); ax.plot(bumps*100, ts, 'o-', color='#2ea44f')\n"
            "ax.axhline(2, color='0.6', ls='--'); ax.set_xlabel('planted into-heat bump (%)'); ax.set_ylabel('detector t')\n"
            "ax.set_title('planted bump is recovered monotonically'); plt.tight_layout(); plt.show()"
        ),
        md("## Verdict\n\n"
           "**Signal: None.** No window separates from noise — into-heat is a "
           "slightly-negative zero (|*t*| < 0.5) and the 'rally through the heat' past-peak "
           "window is directionally positive but tiny (+0.72%, *t* = +0.81, placebo right-tail "
           "p = 0.25, coin-flip 14/27). The synthetic control confirms the detector *would* "
           "have caught a real ≥1% seasonal rally — there just isn't one. "
           "**Tradability: Mirage.** Best cut +0.72%/yr gross, *t* < 1, decaying with costs, "
           "inside a 4-window search."),
    ]
    return new_notebook(cells=cells)


def main():
    for name, nb in (("01_for_the_curious", build_curious()),
                     ("02_for_the_quants", build_quants())):
        path = os.path.join(HERE, f"{name}.ipynb")
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                     "language": "python"}
        with open(path, "w", encoding="utf-8") as fh:
            nbf.write(nb, fh)
        print("wrote", path)


if __name__ == "__main__":
    main()
