"""Generate the two narrative notebooks for Study 783 (IPO-Deal-Of-Year).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached marquee-IPO /
SPY tapes under ../_cache/ (fetching once on a cache miss) and otherwise quote the frozen
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (17 marquee US IPOs + SPY,
# yfinance, SPY 2011-01-03 -> 2026-06-30; 17 of 17 debuts resolved). Fingerprint 523eb8c97e19.
R = dict(
    n_events=17, n_included=17, fp="523eb8c97e19", rows=30098,
    fwd_3m_mean=+7.96, fwd_3m_t=+0.530, fwd_3m_hit=6,
    fwd_6m_mean=-11.06, fwd_6m_t=-0.887, fwd_6m_hit=5,
    fwd_12m_mean=-8.70, fwd_12m_t=-0.541, fwd_12m_hit=4, fwd_n=17,
    pl_3m_p=0.797, pl_3m_mean=+1.59, pl_3m_sd=7.81,
    pl_12m_p=0.0985, pl_12m_mean=+14.29, pl_12m_sd=18.57,
    jk_lo=-1.264, jk_hi=-0.297,
    net5_12m=-8.80, net5_t12=-0.55, net10_12m=-8.90, net10_t12=-0.55,
    null_mean_t=-0.15, null_sd_t=1.20, null_hits=2,
    planted10_mean=-17.50, planted10_t=-3.07, planted20_mean=-27.46, planted20_t=-4.82,
)

PRELUDE = """\
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from ipo_deal_of_year import data as dt, strategy as st

if not dt.have_real():
    print("cache miss -> fetching marquee names + SPY once (needs network)")
    dt.fetch()
prices = dt.load_real()
ev = st.build_event_table(prices, cost_bps=5.0)
inc = ev[ev["included"]]
print(f"panel loaded; {len(inc)} of {len(dt.EVENTS)} marquee IPOs resolved")
"""


def build_curious():
    cells = [
        md("# Study 783 — IPO-Deal-Of-Year 🏦\n\n"
           "*For the curious.* Every year the bankers crown an **'IPO of the year'** — the "
           "biggest, splashiest, most oversubscribed debut on the tape. Folklore (an old cousin "
           "of Ritter's IPO-underperformance result) says the loudest deals then lag: the hype "
           "prices the pop and the newly-public stock drifts down. We put 17 marquee US debuts "
           "(Facebook → Reddit) on the stand and measured their forward return against SPY. The "
           "answer is a **split decision** — with a fat-tailed twist."),
        md("## The claim, and the honest caveat\n\n"
           "We anchor on each name's **first trading close** and measure its **abnormal** return "
           "(name − SPY) at 3, 6 and 12 months. **Selection is *ex post* by design** — these are "
           "the debuts we remember *because* they were the marquee deal, and you can't buy 'the "
           "IPO of the year' at its open (the crown is awarded later). So this is a *descriptive* "
           "autopsy, not a live rule — which already caps tradability at **Mirage**."),
        code(PRELUDE),
        md("## The marquee-IPO calendar we test (hardcoded real tickers + first-trade dates)"),
        code("pd.DataFrame(dt.EVENTS, columns=['ticker','first_trade','deal'])"),
        md("## The picture: mean cumulative abnormal return after the debut\n\n"
           "Offset 0 is the first close; the line traces the average name-minus-SPY path over "
           "the first year."),
        code(
            "car = st.car_path(ev, prices)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.axhline(0, color='0.7', lw=0.8)\n"
            "ax.plot(car.index, car.values * 100, color='#c0392b', lw=2)\n"
            "ax.set_xlabel('trading days after the first close (0 = debut)')\n"
            "ax.set_ylabel('mean cumulative AR, name − SPY (%)')\n"
            "ax.set_title('Marquee IPOs sag against the tape after the pop fades')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md("## The other picture: it's all skew\n\n"
           "The *mean* is a wash because a handful of monsters (Palantir, Reddit, Arm) drown out "
           "the many losers. Look at the 12-month ledger per name:"),
        code(
            "d = inc.sort_values('fwd_12m')\n"
            "fig, ax = plt.subplots(figsize=(9, 5))\n"
            "cols = ['#2ea44f' if v > 0 else '#c0392b' for v in d['fwd_12m']]\n"
            "ax.barh(d['ticker'], d['fwd_12m']*100, color=cols)\n"
            "ax.axvline(0, color='0.4', lw=1)\n"
            "ax.set_xlabel('12-month forward abnormal return, name − SPY (%)')\n"
            "ax.set_title('13 of 17 lag SPY — but 4 lottery winners cancel the mean')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(f"## The numbers\n\n"
           f"| horizon | mean AR | *t* | beat SPY |\n|---|--:|--:|--:|\n"
           f"| 3-month | **{R['fwd_3m_mean']:+.2f}%** | {R['fwd_3m_t']:+.2f} | {R['fwd_3m_hit']}/17 |\n"
           f"| 6-month | {R['fwd_6m_mean']:+.2f}% | {R['fwd_6m_t']:+.2f} | {R['fwd_6m_hit']}/17 |\n"
           f"| 12-month | {R['fwd_12m_mean']:+.2f}% | {R['fwd_12m_t']:+.2f} | {R['fwd_12m_hit']}/17 |\n\n"
           f"So: **most** marquee debuts (13 of 17) trail SPY over their first year, and the "
           f"typical one is ~9% behind — but the *mean* is statistically a wash "
           f"(*t* = {R['fwd_12m_t']:+.2f}) because Palantir (+123%), Reddit (+137%) and Arm "
           f"(+91%) blow the average back to noise. And at 3 months the basket is actually "
           f"*up* {R['fwd_3m_mean']:+.2f}% (the pop). The 'underperformance' is real in the "
           f"median but sign-flips with horizon and never clears significance."),
        code(
            "for label, col in [('3-month','fwd_3m'),('6-month','fwd_6m'),('12-month','fwd_12m')]:\n"
            "    s = st.one_sample_t(inc[col].values); hr = st.hit_rate(inc[col].values)\n"
            "    print(f'{label:<10s} n={s[\"n\"]:2d}  mean={s[\"mean\"]*100:+.2f}%  t={s[\"t\"]:+.3f}  beat SPY {hr[\"k\"]}/{hr[\"n\"]}')"
        ),
        md("## So what?\n\n"
           "The bankers' 'IPO of the year' does tend to disappoint the *typical* buyer — three "
           "in four lag the index over the following year — but as a **basket mean** it's a "
           "coin-flip, and as a **trade** it's a mirage: you can't buy the crown at the open, "
           "the payoff is dominated by a few un-shortable lottery winners, and nothing clears "
           "significance. Verdict: **Weak signal, Mirage tradability.** The quants' notebook has "
           "the placebo, the jackknife, the costed leg and the synthetic control."),
    ]
    return new_notebook(cells=cells)


def build_quants():
    cells = [
        md("# Study 783 — IPO-Deal-Of-Year — for the quants 🔬\n\n"
           "The full battery: one-sample *t* per horizon, a random-window placebo drawn from "
           "*each name's own* tape, a leave-one-out jackknife, the costed net leg, and a seeded "
           "synthetic positive control. Everything offline once cached; fingerprint "
           f"`{R['fp']}`."),
        code(PRELUDE),
        md("## 1. One-sample *t* across independent debut events\n\n"
           "Each marquee IPO is one independent event, so the unit is a one-sample *t* of the "
           "per-name forward abnormal return — **not** a daily panel (which would fake "
           "precision). Note the *sign flip*: positive at 3m (the pop), negative at 6m/12m."),
        code(
            "for label, col in [('3-month','fwd_3m'),('6-month','fwd_6m'),('12-month','fwd_12m')]:\n"
            "    s = st.one_sample_t(inc[col].values)\n"
            "    print(f'{label:<10s} n={s[\"n\"]}  mean={s[\"mean\"]*100:+.2f}%  sd={s[\"sd\"]*100:.1f}%  t={s[\"t\"]:+.3f}')"
        ),
        md("## 2. Random-window placebo — is the 12m sag inside the luck cloud?\n\n"
           "For each name we redraw a random, non-debut 12-month window on **that name's own** "
           "post-listing history vs SPY and recompute the abnormal return; 20 seeds × 200 draws. "
           "Drawing from each stock's own life controls for the fact that new listings are just "
           "more volatile. The null here is strongly *positive* (+14%) — these survivors mooned "
           "on average — so the debut window at −8.7% sits in the left ~10% tail (borderline)."),
        code(
            "pl = st.placebo_pvalue(ev, prices, 'fwd_12m', k=252, tail='left')\n"
            "spy = prices[dt.BENCHMARK]\n"
            "rng = np.random.default_rng(999); draws = []\n"
            "per_name = []\n"
            "for _, r in inc.iterrows():\n"
            "    nm = prices[r['ticker']]; common = nm.index.intersection(spy.index).sort_values()\n"
            "    if len(common) > 253: per_name.append((nm, common))\n"
            "for _ in range(3000):\n"
            "    vals = []\n"
            "    for nm, common in per_name:\n"
            "        p = int(rng.integers(0, len(common)-253))\n"
            "        vals.append(float(nm.loc[common[p+252]]/nm.loc[common[p]] - spy.loc[common[p+252]]/spy.loc[common[p]]))\n"
            "    draws.append(np.mean(vals))\n"
            "fig, ax = plt.subplots(figsize=(9,4))\n"
            "ax.hist(np.array(draws)*100, bins=50, color='#8b949e', alpha=0.8)\n"
            "ax.axvline(pl['obs']*100, color='#c0392b', lw=2, label=f\"observed {pl['obs']*100:+.1f}%\")\n"
            "ax.set_xlabel('mean 12-month AR of random own-name windows (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f\"debut window vs each name's luck cloud (left-tail p={pl['p_value']:.3f})\"); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('placebo:', {k: round(v,4) if isinstance(v,float) else v for k,v in pl.items()})"
        ),
        md("## 3. Jackknife — is the sag broad, or a couple of names?"),
        code(
            "x = inc['fwd_12m'].values\n"
            "jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "print(f'full-sample t = {st.one_sample_t(x)[\"t\"]:+.3f}')\n"
            "print(f'jackknife t range [{min(jk):+.3f}, {max(jk):+.3f}] over {len(x)} leave-one-out draws')\n"
            "print('-> never clears |t|>=2; the mean is a wash however you slice it')"
        ),
        md("## 4. Tradability — net of costs (descriptive only)\n\n"
           "The crown is awarded *ex post*, so this is not a live entry — but even taken at face "
           "value the 12-month basket short is insignificant gross and net, and the four "
           "un-shortable monsters (PLTR/RDDT/ARM/BYND) would have detonated a naive short."),
        code(
            "ev10 = st.build_event_table(prices, cost_bps=10.0); inc10 = ev10[ev10['included']]\n"
            "for base, label in [('fwd_3m','3-month'),('fwd_12m','12-month')]:\n"
            "    g = st.one_sample_t(inc[base].values); n5 = st.one_sample_t(inc[base+'_net'].values); n10 = st.one_sample_t(inc10[base+'_net'].values)\n"
            "    print(f'{label:<10s} gross {g[\"mean\"]*100:+.2f}% (t={g[\"t\"]:+.2f})  net@5 {n5[\"mean\"]*100:+.2f}% (t={n5[\"t\"]:+.2f})  net@10 {n10[\"mean\"]*100:+.2f}% (t={n10[\"t\"]:+.2f})')"
        ),
        md("## 5. Synthetic positive control — the detector works, and the null is quiet\n\n"
           "The one-sample-*t* detector must stay quiet on a planted-null world and recover a "
           "planted post-IPO forward drift. It does: the null barely fires and a planted "
           "under-performance is recovered monotonically — so the insignificant real result is "
           "the *tape's* verdict, not a dead detector."),
        code(
            "null_ts = np.array([st.synthetic_detect(bump=0.0, seed=802+s)['t'] for s in range(20)])\n"
            "print(f'null: mean t={null_ts.mean():+.2f} sd={null_ts.std(ddof=1):.2f}  |t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "for b in (-0.10, -0.20):\n"
            "    r = st.synthetic_detect(bump=b, seed=802)\n"
            "    print(f'planted {b*100:+.0f}%: mean AR {r[\"mean\"]*100:+.2f}%  t={r[\"t\"]:+.2f}')\n"
            "bumps = np.linspace(-0.25, 0.05, 13)\n"
            "ts = [st.synthetic_detect(bump=b, seed=802)['t'] for b in bumps]\n"
            "fig, ax = plt.subplots(figsize=(8,4)); ax.plot(bumps*100, ts, 'o-', color='#2ea44f')\n"
            "ax.axhline(-2, color='0.6', ls='--'); ax.axhline(0, color='0.8', lw=0.8)\n"
            "ax.set_xlabel('planted 12m forward drift (%)'); ax.set_ylabel('detector t')\n"
            "ax.set_title('planted drift is recovered monotonically'); plt.tight_layout(); plt.show()"
        ),
        md("## Verdict\n\n"
           "**Signal: Weak.** Most marquee debuts (13/17) trail SPY over their first year and the "
           "typical name is ~9% behind, and the debut window sits in the left ~10% of each name's "
           "own history — a genuine directional tilt. But the *mean* is a wash "
           "(*t* = −0.54, jackknife never reaches |*t*| ≥ 1.3), it **sign-flips positive at 3 "
           "months** (the pop), and the whole thing is dominated by fat right-tail skew. "
           "Borderline and fragile, not robust. **Tradability: Mirage.** The crown is awarded "
           "*ex post* (no live entry), the payoff is ruled by a few un-shortable lottery winners, "
           "and nothing clears significance net of costs."),
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
