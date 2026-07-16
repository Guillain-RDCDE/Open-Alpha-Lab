"""Generate the two narrative notebooks for Study 786 (Flu-Season).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached CVS/SPY
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (CVS + SPY, yfinance,
# 2006-01-03 -> 2026-06-30; 19 of 19 flu seasons resolved). Fingerprint 804082b9d653.
R = dict(
    n_events=19, n_included=19, fp="804082b9d653", rows=5154,
    pre_s_mean=-0.078, pre_s_t=-0.109, pre_s_hit=9, pre_s_n=19,
    pre_l_mean=+1.124, pre_l_t=+1.018, pre_l_hit=11,
    post_s_mean=-0.231, post_s_t=-0.225, post_s_hit=10,
    post_l_mean=-1.402, post_l_t=-0.909, post_l_hit=8,
    pl_pre_p=0.530, pl_mean=+0.003, pl_sd=1.078, pl_post_p=0.412,
    jk_lo=-0.529, jk_hi=+0.310,
    pre_s_net5=-0.178, pre_s_t5=-0.25, pre_s_net10=-0.278, pre_s_t10=-0.39,
    null_mean_t=+0.28, null_sd_t=0.91, null_hits=0,
    planted1_mean=+0.208, planted1_t=+0.20, planted2_mean=+1.208, planted2_t=+1.15,
)

PRELUDE = """\
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from flu_season import data as dt, strategy as st

if not dt.have_real():
    print("cache miss -> fetching CVS + SPY once (needs network)")
    dt.fetch()
prices = dt.load_real()
ev = st.build_event_table(prices, cost_bps=5.0)
inc = ev[ev["included"]]
print(f"panel loaded; {len(inc)} of {len(dt.EVENTS)} flu seasons resolved")
"""


def build_curious():
    cells = [
        md("# Study 786 — Flu-Season 🤧\n\n"
           "*For the curious.* Everyone *knows* you buy the drugstores into flu season — "
           "CVS gets bid up ahead of its big flu-shot / cold-and-flu revenue window. We put "
           "19 flu seasons (2007→2025) on the stand and asked the tape one plain question: "
           "**does CVS rally into the October flu-season start?** The answer is a flat no."),
        md("## The claim, and why it's a clean test\n\n"
           "The U.S. flu season, by CDC/WHO definition, runs **MMWR weeks 40-20** — it "
           "*begins in early October* every year, a fixed calendar convention known years "
           "ahead. So *buy K sessions before October 1, hold into it* is calendar-known and "
           "zero-look-ahead. We measure CVS's **abnormal** return (CVS − SPY, total-return) "
           "so we're not just measuring its (below-market) beta."),
        code(PRELUDE),
        md("## The flu-season calendar we test (fixed CDC Oct-1 convention)"),
        code("pd.DataFrame(dt.EVENTS, columns=['year', 'season_start'])"),
        md("## The picture: mean cumulative abnormal return around the season start\n\n"
           "Offset 0 is the October-1 season start. Left of zero is the *run-up* (the "
           "supposed rally in); right of zero is the *in-season* window."),
        code(
            "car = st.car_path(ev, prices)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.axvline(0, color='0.4', lw=1, ls='--')\n"
            "ax.axhline(0, color='0.7', lw=0.8)\n"
            "ax.plot(car.index, car.values * 100, color='#2c7fb8', lw=2)\n"
            "ax.set_xlabel('trading days from the flu-season start (0 = Oct 1)')\n"
            "ax.set_ylabel('mean cumulative AR, CVS − SPY (%)')\n"
            "ax.set_title('CVS does not rally into flu season — it just wanders')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(f"## The numbers\n\n"
           f"| window | mean AR | *t* | hit rate |\n|---|--:|--:|--:|\n"
           f"| 2-week run-up | {R['pre_s_mean']:+.2f}% | {R['pre_s_t']:+.2f} | {R['pre_s_hit']}/19 |\n"
           f"| 1-month run-up | {R['pre_l_mean']:+.2f}% | {R['pre_l_t']:+.2f} | {R['pre_l_hit']}/19 |\n"
           f"| 2-week in-season | {R['post_s_mean']:+.2f}% | {R['post_s_t']:+.2f} | {R['post_s_hit']}/19 |\n"
           f"| 1-month in-season | {R['post_l_mean']:+.2f}% | {R['post_l_t']:+.2f} | {R['post_l_hit']}/19 |\n\n"
           f"Every cut is indistinguishable from noise: the 2-week run-up is essentially "
           f"zero ({R['pre_s_mean']:+.2f}%, *t* = {R['pre_s_t']:+.2f}), and the biggest "
           f"number in the table (+{R['pre_l_mean']:.2f}% at a month) still only makes "
           f"*t* = {R['pre_l_t']:+.2f}. No rally-in, no give-back."),
        code(
            "rows = [('2wk run-up','pre_s'),('1mo run-up','pre_l'),('2wk in-season','post_s'),('1mo in-season','post_l')]\n"
            "for label, col in rows:\n"
            "    s = st.one_sample_t(inc[col].values); hr = st.hit_rate(inc[col].values)\n"
            "    print(f'{label:<14s} n={s[\"n\"]:2d}  mean={s[\"mean\"]*100:+.3f}%  t={s[\"t\"]:+.3f}  hit {hr[\"k\"]}/{hr[\"n\"]}')"
        ),
        md("## So what?\n\n"
           "The famous *buy the drugstores into flu season* trade is folklore: CVS's abnormal "
           "return into and through the CDC-defined October season start looks like any random "
           "two-week window. Verdict: **None signal, Mirage tradability.** The quants' notebook "
           "has the placebo, the jackknife, the costed leg and the synthetic control."),
    ]
    return new_notebook(cells=cells)


def build_quants():
    cells = [
        md("# Study 786 — Flu-Season — for the quants 🔬\n\n"
           "The full battery: one-sample *t* per window, a random-window placebo, a "
           "leave-one-out jackknife, the costed net leg, and a seeded synthetic positive "
           "control. Everything offline once cached; fingerprint "
           f"`{R['fp']}`."),
        code(PRELUDE),
        md("## 1. One-sample *t* across independent flu-season years\n\n"
           "Each flu season is one independent event, so the unit is a one-sample *t* of the "
           "per-year abnormal return — **not** a daily panel (which would fake precision)."),
        code(
            "for label, col in [('2wk run-up','pre_s'),('1mo run-up','pre_l'),('2wk in-season','post_s'),('1mo in-season','post_l')]:\n"
            "    s = st.one_sample_t(inc[col].values)\n"
            "    print(f'{label:<14s} n={s[\"n\"]}  mean={s[\"mean\"]*100:+.3f}%  sd={s[\"sd\"]*100:.2f}%  t={s[\"t\"]:+.3f}')"
        ),
        md("## 2. Random-window placebo — is anything outside the luck cloud?\n\n"
           "For each event we redraw a random, non-season 2-week window on CVS vs SPY and "
           "recompute the abnormal return; 20 seeds × 200 draws. If the observed mean sits "
           "in the tail of that null, it isn't ordinary tracking noise. It doesn't."),
        code(
            "pl = st.placebo_pvalue(ev, prices, 'pre_s', k=10, tail='right')\n"
            "import numpy as np\n"
            "cvs, spy = prices[dt.INSTRUMENT], prices[dt.BENCHMARK]\n"
            "common = cvs.index.intersection(spy.index).sort_values()\n"
            "rng = np.random.default_rng(999); draws = []\n"
            "for _ in range(4000):\n"
            "    vals = []\n"
            "    for _e in range(int(inc.shape[0])):\n"
            "        p = int(rng.integers(0, len(common)-11))\n"
            "        vals.append(float(cvs.loc[common[p+10]]/cvs.loc[common[p]] - spy.loc[common[p+10]]/spy.loc[common[p]]))\n"
            "    draws.append(np.mean(vals))\n"
            "fig, ax = plt.subplots(figsize=(9,4))\n"
            "ax.hist(np.array(draws)*100, bins=50, color='#8b949e', alpha=0.8)\n"
            "ax.axvline(pl['obs']*100, color='#2c7fb8', lw=2, label=f\"observed {pl['obs']*100:+.2f}%\")\n"
            "ax.set_xlabel('mean 2-week AR of random windows (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f\"pre-season run-up vs luck cloud (right-tail p={pl['p_value']:.3f})\"); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('placebo:', {k: round(v,4) if isinstance(v,float) else v for k,v in pl.items()})"
        ),
        md("## 3. Jackknife — is a null hiding a single strong year?"),
        code(
            "x = inc['pre_s'].values\n"
            "jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "print(f'full-sample t = {st.one_sample_t(x)[\"t\"]:+.3f}')\n"
            "print(f'jackknife t range [{min(jk):+.3f}, {max(jk):+.3f}] over {len(x)} leave-one-out draws')"
        ),
        md("## 4. Tradability — net of costs\n\n"
           "Calendar-known entry, so the signal window and the tradable window are the same "
           "(gross vs net). The gross edge is already ~zero on every window, and costs only "
           "push it negative — there is no positive-expectancy side."),
        code(
            "ev10 = st.build_event_table(prices, cost_bps=10.0); inc10 = ev10[ev10['included']]\n"
            "for base, label in [('pre_s','2wk run-up'),('post_s','2wk in-season')]:\n"
            "    g = st.one_sample_t(inc[base].values); n5 = st.one_sample_t(inc[base+'_net'].values); n10 = st.one_sample_t(inc10[base+'_net'].values)\n"
            "    print(f'{label:<14s} gross {g[\"mean\"]*100:+.3f}% (t={g[\"t\"]:+.2f})  net@5 {n5[\"mean\"]*100:+.3f}% (t={n5[\"t\"]:+.2f})  net@10 {n10[\"mean\"]*100:+.3f}% (t={n10[\"t\"]:+.2f})')"
        ),
        md("## 5. Synthetic positive control — the detector works; the tape is just empty\n\n"
           "The one-sample-*t* detector must stay quiet on a planted-null world and recover a "
           "planted pre-season bump. It fires cleanly on a ~3% plant — so the flat real-tape "
           "result is a true absence of signal, not a broken detector."),
        code(
            "null_ts = np.array([st.synthetic_detect(bump=0.0, seed=813+s, k=10)['t'] for s in range(20)])\n"
            "print(f'null: mean t={null_ts.mean():+.2f} sd={null_ts.std(ddof=1):.2f}  |t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "for b in (0.01, 0.02):\n"
            "    r = st.synthetic_detect(bump=b, seed=813, k=10)\n"
            "    print(f'planted +{b*100:.0f}%: mean AR {r[\"mean\"]*100:+.3f}%  t={r[\"t\"]:+.2f}')\n"
            "bumps = np.linspace(0, 0.03, 13)\n"
            "ts = [st.synthetic_detect(bump=b, seed=813, k=10)['t'] for b in bumps]\n"
            "fig, ax = plt.subplots(figsize=(8,4)); ax.plot(bumps*100, ts, 'o-', color='#2ea44f')\n"
            "ax.axhline(2, color='0.6', ls='--'); ax.set_xlabel('planted run-up bump (%)'); ax.set_ylabel('detector t')\n"
            "ax.set_title('planted bump is recovered monotonically'); plt.tight_layout(); plt.show()"
        ),
        md("## Verdict\n\n"
           "**Signal: None.** Every cut is indistinguishable from noise (2-week run-up "
           "−0.08%, *t* = −0.11; 1-month +1.12%, *t* = +1.02; in-season legs both |*t*| < 1; "
           "placebo p ≈ 0.53/0.41; jackknife *t* never leaves [−0.53, +0.31]). "
           "**Tradability: Mirage.** The gross edge is ~zero and costs only make it worse — "
           "there is nothing to bank."),
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
