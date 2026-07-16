"""Generate the two narrative notebooks for Study 780 (Long-Weekend-Drift).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY tape under
../_cache/ (fetching once on a cache miss) and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with no
network.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (SPY, yfinance,
# 2004-06-01 -> 2026-06-30; 192 of 192 holidays resolved). Fingerprint 120ff40dea08.
R = dict(
    n_events=192, n_included=192, fp="120ff40dea08", rows=5555, baseline=+0.0484,
    pre1_mean=+0.0964, pre1_t=+1.532, pre1_hit=112, pre1_n=192,
    pre3_mean=+0.2986, pre3_t=+2.195, pre3_hit=113,
    post1_mean=-0.0314, post1_t=-0.384, post1_hit=90,
    pl_pre1_p=0.1263, pl_pre1_mean=-0.0005, pl_pre1_sd=0.0850,
    pl_pre3_p=0.0145, pl_pre3_mean=-0.0022, pl_pre3_sd=0.1376,
    jk_lo=+1.273, jk_hi=+1.815,
    sub_old_mean=+0.0718, sub_old_t=+0.701, sub_new_mean=+0.1210, sub_new_t=+1.644,
    pre1_net5=-0.0036, pre1_t5=-0.06, pre1_net10=-0.1036, pre1_t10=-1.65,
    pre3_net5=+0.1986, pre3_t5=+1.46, pre3_net10=+0.0986, pre3_t10=+0.72,
    null_mean_t=+0.06, null_sd_t=1.05, null_hits=2,
    planted1_mean=+0.1610, planted1_t=+1.97, planted2_mean=+0.3537, planted2_t=+4.34,
)

PRELUDE = """\
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from long_weekend_drift import data as dt, strategy as st

if not dt.have_real():
    print("cache miss -> fetching SPY once (needs network)")
    dt.fetch()
prices = dt.load_real()
ev = st.build_event_table(prices, cost_bps=5.0)
inc = ev[ev["included"]]
print(f"panel loaded; {len(inc)} of {len(dt.EVENTS)} holidays resolved")
"""


def build_curious():
    cells = [
        md("# Study 780 — Long-Weekend-Drift 🏖️\n\n"
           "*For the curious.* Traders have said for a century that the market drifts up into "
           "a holiday — buy the day before a long weekend. We put **192 US market holidays** "
           "(2005→2025) on the stand and asked the tape one plain question: **does SPY beat an "
           "ordinary day on the pre-holiday session?** The answer: a faint yes that dies the "
           "moment you pay a spread."),
        md("## The claim, and why it's a clean test\n\n"
           "The NYSE holiday schedule is **published years ahead**, so *buy K sessions before "
           "the holiday, sell on the eve close* is calendar-known and zero-look-ahead. SPY is "
           "tested on its own, so **'abnormal' means excess over SPY's own average day** — does "
           "the holiday-eve session beat a normal session, not another asset."),
        code(PRELUDE),
        md("## The holiday calendar we test (hardcoded from the NYSE schedule)"),
        code("pd.DataFrame(dt.EVENTS, columns=['holiday_date', 'name']).tail(12)"),
        md("## The picture: mean cumulative excess return around the holiday-eve\n\n"
           "Offset 0 is the holiday-eve close. Left of zero leads *into* the holiday (the "
           "supposed run-up); right of zero is the post-holiday sessions."),
        code(
            "car = st.car_path(ev, prices, pre=5, post=5)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.axvline(0, color='0.4', lw=1, ls='--')\n"
            "ax.axhline(0, color='0.7', lw=0.8)\n"
            "ax.plot(car.index, car.values * 100, color='#1f77b4', lw=2, marker='o')\n"
            "ax.set_xlabel('trading days from the holiday-eve (0 = eve close)')\n"
            "ax.set_ylabel('mean cumulative excess return, SPY (%)')\n"
            "ax.set_title('SPY drifts up INTO a holiday — faintly')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(f"## The numbers\n\n"
           f"| window | mean excess | *t* | hit rate |\n|---|--:|--:|--:|\n"
           f"| holiday-eve (pre1) | **{R['pre1_mean']:+.2f}%** | **{R['pre1_t']:+.2f}** | {R['pre1_hit']}/192 |\n"
           f"| 3-session run-up (pre3) | {R['pre3_mean']:+.2f}% | {R['pre3_t']:+.2f} | {R['pre3_hit']}/192 |\n"
           f"| post-holiday (post1) | {R['post1_mean']:+.2f}% | {R['post1_t']:+.2f} | {R['post1_hit']}/192 |\n\n"
           f"The folklore's **direction is there** — the eve beats an ordinary day by "
           f"{R['pre1_mean']:.2f}% and the 3-session run-up by {R['pre3_mean']:.2f}%, both with "
           f"~58% up-days. But the classic single-eve cut is **not significant** "
           f"(*t* = {R['pre1_t']:+.2f}); only the wider run-up crosses the line "
           f"(*t* = {R['pre3_t']:+.2f}). And the day *after*? A clean zero."),
        code(
            "rows = [('eve pre1','pre1'),('run-up pre3','pre3'),('post1','post1')]\n"
            "for label, col in rows:\n"
            "    s = st.one_sample_t(inc[col].values); hr = st.hit_rate(inc[col].values)\n"
            "    print(f'{label:<12s} n={s[\"n\"]:3d}  mean={s[\"mean\"]*100:+.4f}%  t={s[\"t\"]:+.3f}  hit {hr[\"k\"]}/{hr[\"n\"]}')"
        ),
        md("## So what?\n\n"
           "The famous *buy into the long weekend* trade is a **faint, cost-fragile echo** of "
           "the old pre-holiday anomaly: the drift is there in the right direction, but the "
           "classic single-day version is statistically unremarkable and a 5 bps round-trip "
           "zeroes it out. Verdict: **Weak signal, Fragile tradability.** The quants' notebook "
           "has the placebo, the jackknife, the decay split, the costed leg and the synthetic "
           "control."),
    ]
    return new_notebook(cells=cells)


def build_quants():
    cells = [
        md("# Study 780 — Long-Weekend-Drift — for the quants 🔬\n\n"
           "The full battery: one-sample *t* per window, a random-window placebo, a "
           "leave-one-out jackknife, an old/recent decay split, the costed net leg, and a "
           "seeded synthetic positive control. Everything offline once cached; fingerprint "
           f"`{R['fp']}`."),
        code(PRELUDE),
        md("## 1. One-sample *t* across independent holiday events\n\n"
           "Each holiday is one independent event, so the unit is a one-sample *t* of the "
           "per-holiday excess return (over SPY's own mean daily return) — **not** a daily "
           "panel (which would fake precision)."),
        code(
            "mu = st.baseline_daily(prices)\n"
            "print(f'baseline ordinary-day return = {mu*100:+.4f}%')\n"
            "for label, col in [('eve pre1','pre1'),('run-up pre3','pre3'),('post1','post1')]:\n"
            "    s = st.one_sample_t(inc[col].values)\n"
            "    print(f'{label:<12s} n={s[\"n\"]}  mean={s[\"mean\"]*100:+.4f}%  sd={s[\"sd\"]*100:.3f}%  t={s[\"t\"]:+.3f}')"
        ),
        md("## 2. Random-window placebo — is the drift inside the luck cloud?\n\n"
           "For each event we redraw a random, non-holiday window on SPY and recompute the "
           "excess return; 20 seeds × 200 draws. If the observed mean sits in the tail of that "
           "null, it isn't ordinary noise. Note the split verdict: the single-eve is inside the "
           "cloud, the 3-session run-up is in the tail."),
        code(
            "pl1 = st.placebo_pvalue(ev, prices, 'pre1', k=1, tail='right')\n"
            "pl3 = st.placebo_pvalue(ev, prices, 'pre3', k=3, tail='right')\n"
            "spy = prices[dt.INSTRUMENT]; idx = spy.index.sort_values(); mu = st.baseline_daily(prices)\n"
            "rng = np.random.default_rng(999); draws = []\n"
            "for _ in range(4000):\n"
            "    vals = []\n"
            "    for _e in range(int(inc.shape[0])):\n"
            "        p = int(rng.integers(3, len(idx)-2))\n"
            "        vals.append(float(spy.iloc[p]/spy.iloc[p-3]-1.0) - 3*mu)\n"
            "    draws.append(np.mean(vals))\n"
            "fig, ax = plt.subplots(figsize=(9,4))\n"
            "ax.hist(np.array(draws)*100, bins=50, color='#8b949e', alpha=0.8)\n"
            "ax.axvline(pl3['obs']*100, color='#1f77b4', lw=2, label=f\"observed pre3 {pl3['obs']*100:+.3f}%\")\n"
            "ax.set_xlabel('mean 3-session excess of random windows (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f\"run-up vs luck cloud (right-tail p={pl3['p_value']:.3f})\"); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('pre1 placebo:', {k: round(v,4) if isinstance(v,float) else v for k,v in pl1.items()})\n"
            "print('pre3 placebo:', {k: round(v,4) if isinstance(v,float) else v for k,v in pl3.items()})"
        ),
        md("## 3. Jackknife — is the eve drift one holiday, or broad?"),
        code(
            "x = inc['pre1'].values\n"
            "jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "print(f'full-sample t = {st.one_sample_t(x)[\"t\"]:+.3f}')\n"
            "print(f'jackknife t range [{min(jk):+.3f}, {max(jk):+.3f}] over {len(x)} leave-one-out draws')"
        ),
        md("## 4. Did the anomaly decay? (old vs recent halves)\n\n"
           "Published calendar effects usually fade *after* discovery. Here the eve drift is "
           "mildly *stronger* in the recent half — but both halves are individually "
           "insignificant, so it's persistent weakness, not confirmation."),
        code(
            "half = len(inc)//2\n"
            "for label, sub in [('old 2005->2015', inc.iloc[:half]), ('recent 2015->2025', inc.iloc[half:])]:\n"
            "    s = st.one_sample_t(sub['pre1'].values)\n"
            "    print(f'{label:<18s} n={s[\"n\"]:3d}  mean={s[\"mean\"]*100:+.4f}%  t={s[\"t\"]:+.3f}')"
        ),
        md("## 5. Tradability — net of costs\n\n"
           "Calendar-known entry, so the signal window and the tradable window are the same "
           "(gross vs net). The single-day eve trade is **zeroed by a 5 bps round-trip**; the "
           "3-session run-up keeps a positive but marginal net edge that dies by 10 bps."),
        code(
            "ev10 = st.build_event_table(prices, cost_bps=10.0); inc10 = ev10[ev10['included']]\n"
            "for base, label in [('pre1','eve'),('pre3','3-sess run-up')]:\n"
            "    g = st.one_sample_t(inc[base].values); n5 = st.one_sample_t(inc[base+'_net'].values); n10 = st.one_sample_t(inc10[base+'_net'].values)\n"
            "    print(f'{label:<14s} gross {g[\"mean\"]*100:+.4f}% (t={g[\"t\"]:+.2f})  net@5 {n5[\"mean\"]*100:+.4f}% (t={n5[\"t\"]:+.2f})  net@10 {n10[\"mean\"]*100:+.4f}% (t={n10[\"t\"]:+.2f})')"
        ),
        md("## 6. Synthetic positive control — the detector works, and the null is clean\n\n"
           "The one-sample-*t* detector must stay quiet on a planted-null world and recover a "
           "planted pre-holiday bump. At n ≈ 190 the null is well-behaved (~5% FPR), so the "
           "observed *t* = 1.53 really is unremarkable — and a real 0.4% bump would land at "
           "*t* ≈ 4."),
        code(
            "null_ts = np.array([st.synthetic_detect(bump=0.0, seed=792+s, k=1)['t'] for s in range(20)])\n"
            "print(f'null: mean t={null_ts.mean():+.2f} sd={null_ts.std(ddof=1):.2f}  |t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "for b in (0.002, 0.004):\n"
            "    r = st.synthetic_detect(bump=b, seed=792, k=1)\n"
            "    print(f'planted +{b*100:.1f}%: mean excess {r[\"mean\"]*100:+.4f}%  t={r[\"t\"]:+.2f}')\n"
            "bumps = np.linspace(0, 0.006, 13)\n"
            "ts = [st.synthetic_detect(bump=b, seed=792, k=1)['t'] for b in bumps]\n"
            "fig, ax = plt.subplots(figsize=(8,4)); ax.plot(bumps*100, ts, 'o-', color='#2ea44f')\n"
            "ax.axhline(2, color='0.6', ls='--'); ax.set_xlabel('planted eve bump (%)'); ax.set_ylabel('detector t')\n"
            "ax.set_title('planted bump is recovered monotonically'); plt.tight_layout(); plt.show()"
        ),
        md("## Verdict\n\n"
           "**Signal: Weak.** The drift is directionally real and jackknife-stable (eve +0.10%, "
           "hit 58%; 3-session run-up +0.30%, *t* = +2.20, placebo p = 0.014) — but the classic "
           "single-eve cut is insignificant (*t* = +1.53, p = 0.13), significance only appears "
           "on the wider window, and the detector is well-powered at n = 190. **Tradability: "
           "Fragile.** Survives gross on the run-up with ample SPY capacity, but the single-day "
           "version is zeroed by a 5 bps round-trip and the run-up's net edge is marginal and "
           "gone by 10 bps — a break-even, cost-sensitive echo, not a bankable edge."),
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
