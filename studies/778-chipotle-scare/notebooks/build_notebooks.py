"""Generate the two narrative notebooks for Study 778 (Chipotle-Scare).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached CMG/SPY
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (CMG + SPY, yfinance,
# 2013-01-02 -> 2026-06-30; 6 of 6 scares resolved). Fingerprint 846252f83830.
R = dict(
    n_events=6, n_included=6, fp="846252f83830", rows=3393,
    pre_s_mean=-3.877, pre_s_t=-3.451, pre_s_hit=0, pre_s_n=6,
    pre_l_mean=-9.806, pre_l_t=-2.640, pre_l_hit=1,
    post_s_mean=-9.329, post_s_t=-2.355, post_s_hit=1,
    post_l_mean=-10.111, post_l_t=-1.877, post_l_hit=1,
    pl_reb_p=0.9968, pl_mean=+0.220, pl_sd=3.725, pl_shock_p=0.0022,
    jk_lo=-4.027, jk_hi=-1.742,
    post_s_net5=-9.429, post_s_t5=-2.38, post_s_net10=-9.529, post_s_t10=-2.41,
    null_mean_t=-0.29, null_sd_t=0.76, null_hits=0,
    planted3_mean=+2.768, planted3_t=+2.17, planted6_mean=+5.768, planted6_t=+4.52,
)

PRELUDE = """\
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from chipotle_scare import data as dt, strategy as st

if not dt.have_real():
    print("cache miss -> fetching CMG + SPY once (needs network)")
    dt.fetch()
prices = dt.load_real()
ev = st.build_event_table(prices, cost_bps=5.0)
inc = ev[ev["included"]]
print(f"panel loaded; {len(inc)} of {len(dt.EVENTS)} scares resolved")
"""


def build_curious():
    cells = [
        md("# Study 778 — Chipotle-Scare 🌯🦠\n\n"
           "*For the curious.* Every few years a headline breaks that a Chipotle location "
           "poisoned its customers — E. coli, norovirus, Salmonella — the stock craters, and "
           "somebody on finance Twitter says *buy the dip, the burritos will be back*. We put "
           "**six real Chipotle food-safety scares** (2015→2018) on the stand and asked the "
           "tape one plain question: **if you buy CMG the day the scare goes public, do you "
           "get paid?** The answer is a hard no — and in a direction that should scare *you*."),
        md("## The claim, and why it's a clean test\n\n"
           "A food-safety scare is *unscheduled* — you can't front-run it. But once the news is "
           "public you absolutely *can* buy the panic. So *buy at the announcement close, hold K "
           "sessions* is executable and zero-look-ahead by construction. We measure CMG's "
           "**abnormal** return (CMG − SPY, total-return) so we're grading the burrito, not the "
           "market. Two loud caveats up front: **n = 6**, and four of those six are the *same* "
           "2015 contamination crisis — so this is a story, not a statistic."),
        code(PRELUDE),
        md("## The six scares we test (hardcoded from CDC / health depts / coverage)"),
        code("pd.DataFrame(dt.EVENTS, columns=['year', 'anchor_date', 'what broke'])"),
        md("## The picture: mean cumulative abnormal return around the scare\n\n"
           "Offset 0 is the day the scare went public. Left of zero is the run-in (the dip "
           "forming); right of zero is the *buy-the-dip* window. If the folklore were right, "
           "the line would turn **up** after zero."),
        code(
            "car = st.car_path(ev, prices)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.axvline(0, color='0.4', lw=1, ls='--')\n"
            "ax.axhline(0, color='0.7', lw=0.8)\n"
            "ax.plot(car.index, car.values * 100, color='#c0392b', lw=2)\n"
            "ax.set_xlabel('trading days from the scare (0 = news is public)')\n"
            "ax.set_ylabel('mean cumulative AR, CMG − SPY (%)')\n"
            "ax.set_title('The dip does NOT bounce — CMG keeps falling after a scare')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(f"## The numbers\n\n"
           f"| window | mean AR | *t* | hit rate |\n|---|--:|--:|--:|\n"
           f"| 2-week shock (into the news) | {R['pre_s_mean']:+.2f}% | {R['pre_s_t']:+.2f} | {R['pre_s_hit']}/6 |\n"
           f"| 1-month shock | {R['pre_l_mean']:+.2f}% | {R['pre_l_t']:+.2f} | {R['pre_l_hit']}/6 |\n"
           f"| **1-month rebound (buy the dip)** | **{R['post_s_mean']:+.2f}%** | **{R['post_s_t']:+.2f}** | {R['post_s_hit']}/6 |\n"
           f"| 1-quarter rebound | {R['post_l_mean']:+.2f}% | {R['post_l_t']:+.2f} | {R['post_l_hit']}/6 |\n\n"
           f"The scare is a real dip — CMG lost ~3.9% vs SPY in the week *into* the news (0 of 6 "
           f"events positive). But the 'buy the dip' half is **inverted**: in the month *after* "
           f"the scare CMG fell another **{abs(R['post_s_mean']):.1f}%** (*t* = {R['post_s_t']:+.2f}). "
           f"Only 1 of 6 scares (Powell, OH 2018) rewarded the dip-buyer; the rest kept bleeding."),
        code(
            "rows = [('2wk shock','pre_s'),('1mo shock','pre_l'),('1mo rebound','post_s'),('1q rebound','post_l')]\n"
            "for label, col in rows:\n"
            "    s = st.one_sample_t(inc[col].values); hr = st.hit_rate(inc[col].values)\n"
            "    print(f'{label:<12s} n={s[\"n\"]:2d}  mean={s[\"mean\"]*100:+.3f}%  t={s[\"t\"]:+.3f}  hit {hr[\"k\"]}/{hr[\"n\"]}')"
        ),
        md("## So what?\n\n"
           "The famous *buy the dip on a Chipotle scare* trade is a **falling knife**: the scare "
           "is not a one-day repricing but the start of a slow bleed (spectacularly so in the "
           "2015 crisis, which dragged for months). The one comforting number — a *t* of −2.36 on "
           "the keep-falling drift — is a mirage of small samples: four of six events are one "
           "autocorrelated 2015 episode. Verdict: **Weak signal, Mirage tradability** — you can't "
           "bank a 6-event, one-crisis curiosity, and the folklore as stated loses money. The "
           "quants' notebook has the placebo, the jackknife, the costed leg and the synthetic "
           "control."),
    ]
    return new_notebook(cells=cells)


def build_quants():
    cells = [
        md("# Study 778 — Chipotle-Scare — for the quants 🔬\n\n"
           "The full battery: one-sample *t* per window, a random-window placebo, a "
           "leave-one-out jackknife, the costed net leg, and a seeded synthetic positive "
           "control. Everything offline once cached; fingerprint "
           f"`{R['fp']}`. **Read every *t* through the n = 6 / one-2015-crisis / overlapping-"
           "December-windows caveat.**"),
        code(PRELUDE),
        md("## 1. One-sample *t* across the scare events\n\n"
           "Each scare is one event, so the unit is a one-sample *t* of the per-event abnormal "
           "return — **not** a daily panel (which would fake precision). The claim ('buy the "
           "dip') is a *positive* post-event (rebound) mean."),
        code(
            "for label, col in [('2wk shock','pre_s'),('1mo shock','pre_l'),('1mo rebound','post_s'),('1q rebound','post_l')]:\n"
            "    s = st.one_sample_t(inc[col].values)\n"
            "    print(f'{label:<12s} n={s[\"n\"]}  mean={s[\"mean\"]*100:+.3f}%  sd={s[\"sd\"]*100:.2f}%  t={s[\"t\"]:+.3f}')"
        ),
        md("## 2. Random-window placebo — is the keep-falling drift inside the luck cloud?\n\n"
           "For each event we redraw a random, non-scare one-month window on CMG vs SPY and "
           "recompute the abnormal return; 20 seeds × 200 draws. The buy-the-dip claim is the "
           "*right* tail (a positive rebound); the observed rebound sits at the far *left*."),
        code(
            "pl = st.placebo_pvalue(ev, prices, 'post_s', k=21, tail='right')\n"
            "aapl_cmg, spy = prices[dt.INSTRUMENT], prices[dt.BENCHMARK]\n"
            "common = aapl_cmg.index.intersection(spy.index).sort_values()\n"
            "rng = np.random.default_rng(999); draws = []\n"
            "for _ in range(4000):\n"
            "    vals = []\n"
            "    for _e in range(int(inc.shape[0])):\n"
            "        p = int(rng.integers(0, len(common)-22))\n"
            "        vals.append(float(aapl_cmg.loc[common[p+21]]/aapl_cmg.loc[common[p]] - spy.loc[common[p+21]]/spy.loc[common[p]]))\n"
            "    draws.append(np.mean(vals))\n"
            "fig, ax = plt.subplots(figsize=(9,4))\n"
            "ax.hist(np.array(draws)*100, bins=50, color='#8b949e', alpha=0.8)\n"
            "ax.axvline(pl['obs']*100, color='#c0392b', lw=2, label=f\"observed rebound {pl['obs']*100:+.2f}%\")\n"
            "ax.set_xlabel('mean 1-month AR of random CMG windows (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f\"buy-the-dip rebound vs luck cloud (right-tail p={pl['p_value']:.3f})\"); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('placebo:', {k: round(v,4) if isinstance(v,float) else v for k,v in pl.items()})"
        ),
        md("## 3. Jackknife — is the drift one bad year, or broad?\n\n"
           "The jackknife *t* never leaves negative territory — but that is **not** real "
           "robustness: four of six events are the single 2015 crisis, so leave-one-out can't "
           "break the dependence."),
        code(
            "x = inc['post_s'].values\n"
            "jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "print(f'full-sample t = {st.one_sample_t(x)[\"t\"]:+.3f}')\n"
            "print(f'jackknife t range [{min(jk):+.3f}, {max(jk):+.3f}] over {len(x)} leave-one-out draws')"
        ),
        md("## 4. Tradability — net of costs\n\n"
           "'Buy the dip' loses ~9–10% of abnormal return per event *before* borrow. The inverse "
           "(short the scare, keep shorting) 'works' in-sample, but on 6 events across 11 years, "
           "one 2015 crisis, in a hard-to-borrow single name — capacity-trivial and un-bankable."),
        code(
            "ev10 = st.build_event_table(prices, cost_bps=10.0); inc10 = ev10[ev10['included']]\n"
            "for base, label in [('post_s','1mo rebound'),('post_l','1q rebound')]:\n"
            "    g = st.one_sample_t(inc[base].values); n5 = st.one_sample_t(inc[base+'_net'].values); n10 = st.one_sample_t(inc10[base+'_net'].values)\n"
            "    print(f'{label:<12s} gross {g[\"mean\"]*100:+.3f}% (t={g[\"t\"]:+.2f})  net@5 {n5[\"mean\"]*100:+.3f}% (t={n5[\"t\"]:+.2f})  net@10 {n10[\"mean\"]*100:+.3f}% (t={n10[\"t\"]:+.2f})')"
        ),
        md("## 5. Synthetic positive control — the detector works, the null is quiet\n\n"
           "The one-sample-*t* detector must stay quiet on a planted-null world and recover a "
           "planted buy-the-dip rebound. It does both — so the *absence* of a rebound on the real "
           "tape is informative, not a dead detector."),
        code(
            "null_ts = np.array([st.synthetic_detect(bump=0.0, seed=784+s, k=10, side='post')['t'] for s in range(20)])\n"
            "print(f'null: mean t={null_ts.mean():+.2f} sd={null_ts.std(ddof=1):.2f}  |t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "for b in (0.03, 0.06):\n"
            "    r = st.synthetic_detect(bump=b, seed=784, k=10, side='post')\n"
            "    print(f'planted rebound +{b*100:.0f}%: mean AR {r[\"mean\"]*100:+.3f}%  t={r[\"t\"]:+.2f}')\n"
            "bumps = np.linspace(0, 0.06, 13)\n"
            "ts = [st.synthetic_detect(bump=b, seed=784, k=10, side='post')['t'] for b in bumps]\n"
            "fig, ax = plt.subplots(figsize=(8,4)); ax.plot(bumps*100, ts, 'o-', color='#2ea44f')\n"
            "ax.axhline(2, color='0.6', ls='--'); ax.set_xlabel('planted rebound bump (%)'); ax.set_ylabel('detector t')\n"
            "ax.set_title('planted rebound is recovered monotonically'); plt.tight_layout(); plt.show()"
        ),
        md("## Verdict\n\n"
           "**Signal: Weak.** The *claimed* effect — a positive buy-the-dip rebound — is "
           "decisively absent; the rebound is negative (−9.3%, *t* = −2.36, right-tail p ≈ 0.997), "
           "the reverse of the folklore. There's a suggestive keep-falling downdrift, but it's "
           "fragile: n = 6, four of them one autocorrelated 2015 crisis, two December windows "
           "overlapping. Not noise, not robust. **Tradability: Mirage.** 'Buy the dip' loses money "
           "net; the inverse short is a 6-event, one-episode, hard-to-borrow curiosity with no "
           "capacity and no independence — nothing to bank."),
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
