"""Generate the two narrative notebooks for Study 781 (Quad-Witching-Hangover).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY tape
under ../_cache/ (fetching once on a cache miss) and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere
with no network.
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
# 2004-06-01 -> 2026-06-30; 84 of 84 quad-witching Fridays resolved). Fingerprint a7fba5aedc7e.
R = dict(
    n_events=84, n_included=84, fp="a7fba5aedc7e", rows=5555,
    post_s_mean=+0.075, post_s_t=+0.276, post_s_hit=35, post_s_n=84,
    post_l_mean=+0.375, post_l_t=+0.937, post_l_hit=53,
    pre_mean=+0.188, pre_t=+0.660, pre_hit=51,
    pl_post_s_obs=+0.075, pl_post_s_mean=+0.239, pl_post_s_sd=0.256, pl_post_s_p=0.248,
    pl_post_l_obs=+0.375, pl_post_l_mean=+0.475, pl_post_l_sd=0.346, pl_post_l_p=0.364,
    jk_lo=-0.226, jk_hi=+0.592,
    post_s_net5=-0.025, post_s_t5=-0.09, post_s_net10=-0.125, post_s_t10=-0.46,
    post_l_net5=+0.275, post_l_t5=+0.69, post_l_net10=+0.175, post_l_t10=+0.44,
    null_mean_t=+0.67, null_sd_t=0.93, null_hits=3,
    planted1_mean=+0.040, planted1_t=+0.16, planted2_mean=-0.560, planted2_t=-2.19,
)

PRELUDE = """\
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from quad_witching_hangover import data as dt, strategy as st

if not dt.have_real():
    print("cache miss -> fetching SPY once (needs network)")
    dt.fetch()
prices = dt.load_real()
ev = st.build_event_table(prices, cost_bps=5.0)
inc = ev[ev["included"]]
print(f"panel loaded; {len(inc)} of {len(dt.EVENTS)} quad-witching Fridays resolved")
"""


def build_curious():
    cells = [
        md("# Study 781 — Quad-Witching-Hangover 🗓️\n\n"
           "*For the curious.* Four times a year — the third Friday of March, June, "
           "September and December — four classes of derivatives expire at once: "
           "\"quadruple witching.\" Trader-lore says the market is left **hungover** the "
           "following week, drifting lower after the expiration churn. We put all 84 "
           "quad-witching Fridays (2005→2025) on the stand and asked the tape one plain "
           "question: **does the week after quad-witching underperform?**"),
        md("## The claim, and why it's a clean test\n\n"
           "Quad-witching is a fixture of the exchange calendar — literally *third Friday of "
           "the quarter's last month* — so it is **known years ahead**. A \"sit out / short "
           "the week after\" rule is calendar-known and zero-look-ahead. We test `SPY`'s own "
           "forward return (SPY *is* the index, so there's no separate benchmark) and judge "
           "\"underperform\" against SPY's ordinary drift via a random-window placebo."),
        code(PRELUDE),
        md("## The quad-witching calendar we test (hardcoded, third Fridays)"),
        code("pd.DataFrame(dt.EVENTS, columns=['date','year','quarter']).head(12)"),
        md("## The picture: mean cumulative SPY return around quad-witching\n\n"
           "Offset 0 is the quad-witching close. Left of zero is the *run-in*; right of zero "
           "is the *hangover* window the folklore says should sag."),
        code(
            "car = st.car_path(ev, prices)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.axvline(0, color='0.4', lw=1, ls='--')\n"
            "ax.axhline(0, color='0.7', lw=0.8)\n"
            "ax.plot(car.index, car.values * 100, color='#2c7fb8', lw=2, marker='o', ms=3)\n"
            "ax.set_xlabel('trading days from quad-witching (0 = expiration close)')\n"
            "ax.set_ylabel('mean cumulative SPY return (%)')\n"
            "ax.set_title('No hangover: SPY drifts mildly UP the week after quad-witching')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(f"## The numbers\n\n"
           f"| window | mean | *t* | up-rate |\n|---|--:|--:|--:|\n"
           f"| 1-week hangover | **{R['post_s_mean']:+.2f}%** | **{R['post_s_t']:+.2f}** | {R['post_s_hit']}/84 |\n"
           f"| 2-week give-back | {R['post_l_mean']:+.2f}% | {R['post_l_t']:+.2f} | {R['post_l_hit']}/84 |\n"
           f"| 1-week run-in | {R['pre_mean']:+.2f}% | {R['pre_t']:+.2f} | {R['pre_hit']}/84 |\n\n"
           f"The folklore is **wrong on the sign**: the week *after* quad-witching returns "
           f"{R['post_s_mean']:+.2f}% on average (*t* = {R['post_s_t']:+.2f}) — mildly "
           f"*positive*, not a hangover. Two weeks out it's more positive still "
           f"({R['post_l_mean']:+.2f}%). There is simply no underperformance to trade."),
        code(
            "rows = [('1wk hangover','post_s'),('2wk give-back','post_l'),('1wk run-in','pre')]\n"
            "for label, col in rows:\n"
            "    s = st.one_sample_t(inc[col].values); hr = st.hit_rate(inc[col].values)\n"
            "    print(f'{label:<14s} n={s[\"n\"]:2d}  mean={s[\"mean\"]*100:+.3f}%  t={s[\"t\"]:+.3f}  up {hr[\"k\"]}/{hr[\"n\"]}')"
        ),
        md("## So what?\n\n"
           "The \"quad-witching hangover\" is a myth. SPY's week after quarterly quad-witching "
           "is *positive* (+0.08%), sits dead-centre of the random-window placebo, and its "
           "*t*-stat is a rounding error. Verdict: **Signal None, Tradability Mirage.** The "
           "quants' notebook has the placebo, the jackknife, the costed leg and the synthetic "
           "control that proves the detector *would* catch a real hangover if one existed."),
    ]
    return new_notebook(cells=cells)


def build_quants():
    cells = [
        md("# Study 781 — Quad-Witching-Hangover — for the quants 🔬\n\n"
           "The full battery: one-sample *t* per window, a random-window placebo, a "
           "leave-one-out jackknife, the costed net leg, and a seeded synthetic positive "
           "control. Everything offline once cached; fingerprint "
           f"`{R['fp']}`."),
        code(PRELUDE),
        md("## 1. One-sample *t* across independent quad-witching quarters\n\n"
           "Each quad-witching Friday is one independent, non-overlapping event, so the unit "
           "is a one-sample *t* of the per-quarter forward return — **not** a daily panel "
           "(which would fake precision)."),
        code(
            "for label, col in [('1wk hangover','post_s'),('2wk give-back','post_l'),('1wk run-in','pre')]:\n"
            "    s = st.one_sample_t(inc[col].values)\n"
            "    print(f'{label:<14s} n={s[\"n\"]}  mean={s[\"mean\"]*100:+.3f}%  sd={s[\"sd\"]*100:.2f}%  t={s[\"t\"]:+.3f}')"
        ),
        md("## 2. Random-window placebo — is the hangover inside the luck cloud?\n\n"
           "For each event we redraw a random, non-quad-witching 1-week window on SPY and "
           "recompute the return; 20 seeds × 200 draws. SPY drifts up, so the placebo mean is "
           "positive; if the observed hangover were real it would sit in the **left** tail."),
        code(
            "pl = st.placebo_pvalue(ev, prices, 'post_s', k=5, tail='left')\n"
            "spy = prices[dt.INSTRUMENT]; common = spy.index.sort_values()\n"
            "rng = np.random.default_rng(999); draws = []\n"
            "for _ in range(4000):\n"
            "    vals = []\n"
            "    for _e in range(int(inc.shape[0])):\n"
            "        p = int(rng.integers(0, len(common)-6))\n"
            "        vals.append(float(spy.loc[common[p+5]]/spy.loc[common[p]] - 1.0))\n"
            "    draws.append(np.mean(vals))\n"
            "fig, ax = plt.subplots(figsize=(9,4))\n"
            "ax.hist(np.array(draws)*100, bins=50, color='#8b949e', alpha=0.8)\n"
            "ax.axvline(pl['obs']*100, color='#2c7fb8', lw=2, label=f\"observed {pl['obs']*100:+.2f}%\")\n"
            "ax.set_xlabel('mean 1-week SPY return of random windows (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f\"post-quad-witching week vs luck cloud (left-tail p={pl['p_value']:.3f})\"); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('placebo:', {k: round(v,4) if isinstance(v,float) else v for k,v in pl.items()})"
        ),
        md("## 3. Jackknife — is the (non-)effect robust, or one quarter?"),
        code(
            "x = inc['post_s'].values\n"
            "jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "print(f'full-sample t = {st.one_sample_t(x)[\"t\"]:+.3f}')\n"
            "print(f'jackknife t range [{min(jk):+.3f}, {max(jk):+.3f}] over {len(x)} leave-one-out draws')"
        ),
        md("## 4. Tradability — net of costs\n\n"
           "Calendar-known entry, so the signal window and the tradable window are the same. "
           "To \"trade the hangover\" you'd **short** a window whose gross return is *positive* "
           "— you lose before costs and more after. There is no edge in either direction."),
        code(
            "ev10 = st.build_event_table(prices, cost_bps=10.0); inc10 = ev10[ev10['included']]\n"
            "for base, label in [('post_s','1wk hangover'),('post_l','2wk give-back')]:\n"
            "    g = st.one_sample_t(inc[base].values); n5 = st.one_sample_t(inc[base+'_net'].values); n10 = st.one_sample_t(inc10[base+'_net'].values)\n"
            "    print(f'{label:<14s} gross {g[\"mean\"]*100:+.3f}% (t={g[\"t\"]:+.2f})  net@5 {n5[\"mean\"]*100:+.3f}% (t={n5[\"t\"]:+.2f})  net@10 {n10[\"mean\"]*100:+.3f}% (t={n10[\"t\"]:+.2f})')"
        ),
        md("## 5. Synthetic positive control — the detector works, the tape has nothing\n\n"
           "The one-sample-*t* detector must recover a *planted* post-event hangover and stay "
           "controlled on the null. The null mean *t* is slightly positive because the "
           "synthetic world (like SPY) has +drift leaking into a self-benchmarked *t* — the "
           "same reason the real hangover reads mildly positive."),
        code(
            "null_ts = np.array([st.synthetic_detect(dip=0.0, seed=793+s, k=5)['t'] for s in range(20)])\n"
            "print(f'null: mean t={null_ts.mean():+.2f} sd={null_ts.std(ddof=1):.2f}  |t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "for d in (0.006, 0.012):\n"
            "    r = st.synthetic_detect(dip=d, seed=793, k=5)\n"
            "    print(f'planted -{d*100:.1f}%: mean {r[\"mean\"]*100:+.3f}%  t={r[\"t\"]:+.2f}')\n"
            "dips = np.linspace(0, 0.016, 13)\n"
            "ts = [st.synthetic_detect(dip=d, seed=793, k=5)['t'] for d in dips]\n"
            "fig, ax = plt.subplots(figsize=(8,4)); ax.plot(dips*100, ts, 'o-', color='#c0392b')\n"
            "ax.axhline(-2, color='0.6', ls='--'); ax.set_xlabel('planted hangover dip (%)'); ax.set_ylabel('detector t')\n"
            "ax.set_title('planted hangover is recovered monotonically'); plt.tight_layout(); plt.show()"
        ),
        md("## Verdict\n\n"
           "**Signal: None.** The week after quad-witching returns +0.08% (*t* = +0.28) — "
           "*positive*, opposite the folklore; the 2-week window is +0.38% (*t* = +0.94); the "
           "placebo is dead-centre (left-tail p = 0.25 / 0.36) and the jackknife *t* is pinned "
           "near zero (−0.23…+0.59). **Tradability: Mirage.** \"Shorting the hangover\" shorts "
           "a positive window and loses gross; costs sink the 1-week leg outright. No edge in "
           "either direction."),
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
