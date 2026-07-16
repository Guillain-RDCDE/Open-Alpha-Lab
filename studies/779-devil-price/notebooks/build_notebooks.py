"""Generate the two narrative notebooks for Study 779 (Devil-Price).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached S&P 100 / SPY
panel under ../_cache/ (fetching once on a cache miss) and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs
anywhere with no network.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (S&P 100 + SPY, yfinance,
# 2007-01-03 -> 2026-06-30; 98 of 99 names returned data; 976 weekly rebalances).
# Fingerprint b72f770274d9.
R = dict(
    fp="b72f770274d9", rows=4903, n_names=98, n_rebal=976,
    devil_s_dm=-0.0061, devil_s_t=-0.216, devil_l_dm=-0.0455, devil_l_t=-0.872,
    ls_s=+0.0365, ls_s_t=+0.823, ls_s_hit=530, ls_s_n=976,
    ls_l=+0.1827, ls_l_t=+2.281, ls_l_hit=532,
    spy_s=+0.0729, spy_s_t=+1.954, spy_l=+0.2685, spy_l_t=+3.770,
    pl_s_obs=+0.0365, pl_s_mean=-0.0000, pl_s_sd=0.0396, pl_s_p=0.179,
    pl_l_obs=+0.1827, pl_l_mean=+0.0004, pl_l_sd=0.0769, pl_l_p=0.006,
    jk_full=+0.823, jk_lo=+0.353, jk_hi=+1.960,
    mo_ls=+0.2456, mo_ls_t=+1.415, mo_n=232, mo_p=0.061, mo_devil_t=-0.505,
    null_mean_t=-0.24, null_sd_t=0.92, null_hits=0,
    planted1_mean=-0.245, planted1_t=-5.58, planted2_mean=-0.427, planted2_t=-8.94,
)

PRELUDE = """\
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from devil_price import data as dt, strategy as st

if not dt.have_real():
    print("cache miss -> fetching S&P 100 + SPY once (needs network)")
    dt.fetch()
panel, spy = dt.load_real()
events = st.build_panel_events(panel, spy)
table = st.build_sort_table(events)
print(f"panel loaded; {len(events)} weekly rebalances over {panel.shape[1]} names")
"""


def build_curious():
    cells = [
        md("# Study 779 — Devil-Price 😈\n\n"
           "*For the curious.* 666 is the number of the beast — the most famous unlucky "
           "number there is — and the S&P 500's 2009 crash bottom near **666** is annotated "
           "to death as a 'devilish low' (it was actually the start of a decade-long bull). "
           "So does a **stock** whose price sits on a 666 handle ($6.66, $66.6, $666…) carry "
           "a curse and then lag? We sorted the S&P 100 every week and asked the tape."),
        md("## The claim, and why it's a clean test\n\n"
           "A share price mantissa is visible in real time, so ranking names by how close "
           "their price is to the nearest 666 level and shorting the closest quintile is "
           "zero-look-ahead. The trick: the distance is **decade-invariant** — $6.66, $66.6 "
           "and $666 all score identically — so the 'devil' bucket is full every week. We "
           "measure **cross-sectionally demeaned** returns, so the market drops out and we're "
           "not just measuring beta."),
        code(PRELUDE),
        md("## Where the devil sits on the price circle\n\n"
           "Every price maps to a point on the log-price circle; the beast is at "
           "`log10(6.66) = 0.82`. Distance 0 = on a 666 handle, 0.5 = maximally far."),
        code(
            "xs = np.linspace(1, 100, 4000)\n"
            "fig, ax = plt.subplots(figsize=(9, 4))\n"
            "ax.plot(xs, dt.devil_distance(xs), color='#8e44ad', lw=1.8)\n"
            "for lv in (6.66, 66.6):\n"
            "    ax.axvline(lv, color='#c0392b', ls='--', lw=1)\n"
            "    ax.annotate(f'${lv}', (lv, 0.02), color='#c0392b')\n"
            "ax.set_xscale('log'); ax.set_xlabel('price ($, log scale)')\n"
            "ax.set_ylabel('devil distance  [0, 0.5]')\n"
            "ax.set_title('The devil level repeats every decade (6.66, 66.6, 666, ...)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md("## The picture: forward return by devil-distance decile\n\n"
           "Bin 0 is *nearest* the beast; bin 9 is farthest. If the curse were real, the "
           "left bars would be deeply negative."),
        code(
            "prof = st.distance_profile(events, horizon='s')\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "colors = ['#c0392b' if i < 2 else '#7f8c8d' for i in range(len(prof))]\n"
            "ax.bar(prof.index, prof.values * 100, color=colors)\n"
            "ax.axhline(0, color='0.4', lw=0.8)\n"
            "ax.set_xlabel('devil-distance decile (0 = on the 666 handle)')\n"
            "ax.set_ylabel('mean demeaned 1-week fwd return (%)')\n"
            "ax.set_title('The 666-touchers (left) are not visibly cursed')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(f"## The numbers\n\n"
           f"| cut | mean | *t* | hit rate |\n|---|--:|--:|--:|\n"
           f"| devil quintile, 1-week (demeaned) | {R['devil_s_dm']:+.3f}% | **{R['devil_s_t']:+.2f}** | — |\n"
           f"| clean − devil spread, 1-week | {R['ls_s']:+.3f}% | {R['ls_s_t']:+.2f} | {R['ls_s_hit']}/{R['ls_s_n']} |\n"
           f"| clean − devil spread, 1-month (~4x overlap) | {R['ls_l']:+.3f}% | **{R['ls_l_t']:+.2f}** | {R['ls_l_hit']}/{R['ls_s_n']} |\n\n"
           f"The 666-touchers do **not** measurably underperform at the clean weekly horizon "
           f"(*t* = {R['devil_s_t']:+.2f} — a dead zero). The spread carries the folklore's "
           f"sign (clean beats devil) and the *monthly* cut even shows *t* = {R['ls_l_t']:+.2f}… "
           f"but that's an illusion — see the quants' notebook, where de-overlapping the "
           f"windows drops it to *t* = {R['mo_ls_t']:+.2f}."),
        code(
            "for label, col in [('devil 1wk (dm)','devil_s_dm'),('LS 1wk','ls_s'),('LS 1mo','ls_l')]:\n"
            "    s = st.one_sample_t(table[col].values)\n"
            "    print(f'{label:<14s} n={s[\"n\"]:4d}  mean={s[\"mean\"]*100:+.4f}%  t={s[\"t\"]:+.3f}')"
        ),
        md("## So what?\n\n"
           "The number-of-the-beast price sort is a **faint, folklore-signed lean** (the "
           "clean names edge out the 666-touchers) that only *looks* significant through a "
           "window-overlap trick and evaporates under a clean non-overlapping test. And a "
           "stock's nominal price digit is economically meaningless (it's just split history), "
           "so there's no mechanism to hang it on. Verdict: **Weak signal, Mirage "
           "tradability.** The quants' notebook has the placebo, the jackknife, the "
           "de-overlapped re-run and the synthetic control."),
    ]
    return new_notebook(cells=cells)


def build_quants():
    cells = [
        md("# Study 779 — Devil-Price — for the quants 🔬\n\n"
           "The full battery: one-sample *t* per cut, a random-**bucket** placebo, a "
           "leave-one-year-out jackknife, the decisive **non-overlapping** monthly re-run, "
           "and a seeded synthetic positive control. Everything offline once cached; "
           f"fingerprint `{R['fp']}`."),
        code(PRELUDE),
        md("## 1. One-sample *t* across weekly rebalances\n\n"
           "The unit is the per-rebalance cross-sectional spread. Note the horizon caveat: "
           "the 1-week window is (near-)non-overlapping and honest; the 1-month window is "
           "rebalanced weekly, so consecutive windows overlap ~4× and its *t* is inflated."),
        code(
            "for label, col in [('devil 1wk (dm)','devil_s_dm'),('devil 1mo (dm)','devil_l_dm'),\n"
            "                   ('LS 1wk','ls_s'),('LS 1mo','ls_l'),\n"
            "                   ('devil vs SPY 1mo','devil_l_abn')]:\n"
            "    s = st.one_sample_t(table[col].values)\n"
            "    print(f'{label:<18s} n={s[\"n\"]:4d}  mean={s[\"mean\"]*100:+.4f}%  t={s[\"t\"]:+.3f}')\n"
            "print('\\n(the +vs-SPY figure is an equal-weight-vs-cap-weight artefact, not a devil effect)')"
        ),
        md("## 2. Random-bucket placebo — does the *devil* characteristic carry the spread?\n\n"
           "Replace the devil sort with two random quintiles of the same size and recompute "
           "clean−devil; 20 seeds × 100 draws. If the real spread sits in the tail of that "
           "null, the characteristic (not bucket size) carried it."),
        code(
            "pl_s = st.placebo_pvalue(events, horizon='s', tail='right')\n"
            "pl_l = st.placebo_pvalue(events, horizon='l', tail='right')\n"
            "fig, ax = plt.subplots(figsize=(9,4))\n"
            "rng = np.random.default_rng(123)\n"
            "# rebuild the 1-month null cloud for the histogram\n"
            "fwds = [e['fwd_l'] for e in events]; sizes=[max(1,int(round(len(f)*st.QFRAC))) for f in fwds]\n"
            "cloud=[]\n"
            "for _ in range(1500):\n"
            "    sp=[]\n"
            "    for f,k in zip(fwds,sizes):\n"
            "        perm=rng.permutation(len(f)); sp.append(f[perm[k:2*k]].mean()-f[perm[:k]].mean())\n"
            "    cloud.append(np.mean(sp))\n"
            "ax.hist(np.array(cloud)*100, bins=50, color='#8e8e8e', alpha=0.8)\n"
            "ax.axvline(pl_l['obs']*100, color='#c0392b', lw=2, label=f\"observed {pl_l['obs']*100:+.3f}%\")\n"
            "ax.set_xlabel('mean 1-month clean-devil spread, random buckets (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f\"1-month spread vs random-bucket cloud (right-tail p={pl_l['p_value']:.3f})\"); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('1wk placebo p =', round(pl_s['p_value'],3), '| 1mo placebo p =', round(pl_l['p_value'],3))"
        ),
        md("## 3. The decisive test — de-overlap the monthly windows\n\n"
           "Rebalance **monthly** (spacing = 21) so the 21-day windows no longer overlap. If "
           "the monthly *t* = +2.28 was real, it survives; if it was overlap inflation, it "
           "collapses."),
        code(
            "ev21 = st.build_panel_events(panel, spy, fwd_s=5, fwd_l=21, spacing=21)\n"
            "t21 = st.build_sort_table(ev21)\n"
            "s = st.one_sample_t(t21['ls_l'].values); d = st.one_sample_t(t21['devil_l_dm'].values)\n"
            "pl = st.placebo_pvalue(ev21, horizon='l', tail='right')\n"
            "print(f'weekly-overlap 1mo LS:   t=+2.281  (inflated)')\n"
            "print(f'NON-overlap  1mo LS:     n={s[\"n\"]}  mean={s[\"mean\"]*100:+.4f}%  t={s[\"t\"]:+.3f}  placebo p={pl[\"p_value\"]:.3f}')\n"
            "print(f'NON-overlap  devil (dm):  t={d[\"t\"]:+.3f}')\n"
            "print('-> the monthly edge was window overlap, not a cross-sectional signal')"
        ),
        md("## 4. Leave-one-year-out jackknife — 1-week LS spread\n\n"
           "Is the (small, positive) weekly spread broad across years, or one lucky year?"),
        code(
            "yrs = pd.DatetimeIndex(table['date']).year\n"
            "full = st.one_sample_t(table['ls_s'].values)['t']\n"
            "ts = [st.one_sample_t(table[yrs!=y]['ls_s'].values)['t'] for y in sorted(set(yrs))]\n"
            "print(f'full-sample t = {full:+.3f}')\n"
            "print(f'jackknife t range [{min(ts):+.3f}, {max(ts):+.3f}] over {len(ts)} leave-one-year-out draws')\n"
            "print('-> persistently positive (folklore sign) but never significant')"
        ),
        md("## 5. Synthetic positive control — the detector works, and the null is quiet\n\n"
           "The sort detector must stay quiet on a planted-null world and recover a planted "
           "devil-quintile underperformance monotonically."),
        code(
            "null_ts = np.array([st.synthetic_detect(bump=0.0, seed=789+s, side='devil')['t'] for s in range(20)])\n"
            "print(f'null: mean t={null_ts.mean():+.2f} sd={null_ts.std(ddof=1):.2f}  |t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "for b in (0.01, 0.02):\n"
            "    r = st.synthetic_detect(bump=b, seed=789, side='devil')\n"
            "    print(f'planted -{b*100:.0f}%: devil-quintile mean {r[\"mean\"]*100:+.3f}%  t={r[\"t\"]:+.2f}')\n"
            "bumps = np.linspace(0, 0.025, 11)\n"
            "ts = [st.synthetic_detect(bump=b, seed=789, side='devil')['t'] for b in bumps]\n"
            "fig, ax = plt.subplots(figsize=(8,4)); ax.plot(bumps*100, ts, 'o-', color='#8e44ad')\n"
            "ax.axhline(-2, color='0.6', ls='--'); ax.set_xlabel('planted devil drag (%)'); ax.set_ylabel('detector t')\n"
            "ax.set_title('planted underperformance is recovered monotonically'); plt.tight_layout(); plt.show()"
        ),
        md("## Verdict\n\n"
           "**Signal: Weak.** The clean weekly tests are quiet (devil quintile *t* = −0.22, "
           "LS *t* = +0.82, placebo p = 0.18). There is a persistent folklore-signed lean "
           "(LS positive both horizons, hit ~54.5%, jackknife-stable positive) and the "
           "monthly spread hits placebo p = 0.006 — but that is window-overlap inflation: "
           "de-overlapped, the monthly *t* falls to +1.42 (p = 0.06). Borderline, "
           "horizon-fragile, and mechanism-free (nominal price digit ≠ fundamentals). "
           "**Tradability: Mirage.** ~18-25 bps/month gross on a weekly-turnover, ~20-a-side "
           "long/short of an arbitrary price digit — no clean significance, no mechanism, "
           "nothing to bank."),
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
