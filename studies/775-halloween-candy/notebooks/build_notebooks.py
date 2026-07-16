"""Generate the two narrative notebooks for Study 775 (Halloween-Candy).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached HSY/SPY
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (HSY + SPY, yfinance,
# 1993-01-29 -> 2026-06-30; 32 of 32 Halloweens resolved). Fingerprint 22c38ea72789.
R = dict(
    n_events=32, n_included=32, fp="22c38ea72789", rows=8411,
    pre_s_mean=-1.192, pre_s_t=-1.339, pre_s_hit=14, pre_s_n=32,
    pre_l_mean=-2.012, pre_l_t=-1.759, pre_l_hit=12,
    post_s_mean=-1.080, post_s_t=-1.319, post_s_hit=9,
    post_l_mean=-1.119, post_l_t=-0.919, post_l_hit=10,
    pl_pre_p=0.9305, pl_mean=+0.037, pl_sd=0.830, pl_post_p=0.0872,
    jk_lo=-2.242, jk_hi=-1.458,
    pre_l_net5=-2.112, pre_l_t5=-1.85, pre_l_net10=-2.212, pre_l_t10=-1.93,
    null_mean_t=+0.06, null_sd_t=1.20, null_hits=2,
    planted1_mean=+1.083, planted1_t=+1.26, planted2_mean=+2.083, planted2_t=+2.41,
)

PRELUDE = """\
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from halloween_candy import data as dt, strategy as st

if not dt.have_real():
    print("cache miss -> fetching HSY + SPY once (needs network)")
    dt.fetch()
prices = dt.load_real()
ev = st.build_event_table(prices, cost_bps=5.0)
inc = ev[ev["included"]]
print(f"panel loaded; {len(inc)} of {len(dt.EVENTS)} Halloweens resolved")
"""


def build_curious():
    cells = [
        md("# Study 775 — Halloween-Candy 🎃\n\n"
           "*For the curious.* Everyone *knows* Halloween is Hershey's Super Bowl — the "
           "biggest candy weekend of the year — so surely you buy `HSY` into Oct 31 and ride "
           "the trick-or-treat haul. We put 32 Halloweens (1994→2025) on the stand and asked "
           "the tape two plain questions: **does HSY rally into Halloween?** and **does it "
           "fade after?** The answer is a double bust — with an unexpected direction."),
        md("## The claim, and why it's a clean test\n\n"
           "Halloween is a **fixed public holiday — always October 31** — so *buy K sessions "
           "before, sell on the day* is calendar-known decades ahead and zero-look-ahead. We "
           "measure HSY's **abnormal** return (HSY − SPY, total-return) so we're not just "
           "measuring a slow staple drifting with the market."),
        code(PRELUDE),
        md("## The Halloween calendar we test (one event per year, Oct 31)"),
        code("pd.DataFrame(dt.EVENTS, columns=['year', 'halloween_date']).head(8)"),
        md("## The picture: mean cumulative abnormal return around Halloween\n\n"
           "Offset 0 is Halloween. Left of zero is the *run-up* (the supposed rally in); "
           "right of zero is the *fade* window."),
        code(
            "car = st.car_path(ev, prices)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.axvline(0, color='0.4', lw=1, ls='--')\n"
            "ax.axhline(0, color='0.7', lw=0.8)\n"
            "ax.plot(car.index, car.values * 100, color='#e67e22', lw=2)\n"
            "ax.set_xlabel('trading days from Halloween (0 = Oct 31)')\n"
            "ax.set_ylabel('mean cumulative AR, HSY − SPY (%)')\n"
            "ax.set_title('Hershey does NOT rally into Halloween — it mildly lags')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(f"## The numbers\n\n"
           f"| window | mean AR | *t* | hit rate |\n|---|--:|--:|--:|\n"
           f"| 2-week run-up | {R['pre_s_mean']:+.2f}% | {R['pre_s_t']:+.2f} | {R['pre_s_hit']}/32 |\n"
           f"| 1-month run-up | **{R['pre_l_mean']:+.2f}%** | **{R['pre_l_t']:+.2f}** | {R['pre_l_hit']}/32 |\n"
           f"| 2-week fade | {R['post_s_mean']:+.2f}% | {R['post_s_t']:+.2f} | {R['post_s_hit']}/32 |\n"
           f"| 1-month fade | {R['post_l_mean']:+.2f}% | {R['post_l_t']:+.2f} | {R['post_l_hit']}/32 |\n\n"
           f"The bullish folklore is **absent, if anything inverted**: in the month *into* "
           f"Halloween HSY has under-performed SPY by {abs(R['pre_l_mean']):.2f}% on average "
           f"(*t* = {R['pre_l_t']:+.2f}) — the opposite of a rally in. And the 'sell the "
           f"season' fade? Also mildly negative, and not significant either. Every window "
           f"points down, none of them clears the significance bar (|*t*| < 2)."),
        code(
            "rows = [('2wk run-up','pre_s'),('1mo run-up','pre_l'),('2wk fade','post_s'),('1mo fade','post_l')]\n"
            "for label, col in rows:\n"
            "    s = st.one_sample_t(inc[col].values); hr = st.hit_rate(inc[col].values)\n"
            "    print(f'{label:<12s} n={s[\"n\"]:2d}  mean={s[\"mean\"]*100:+.3f}%  t={s[\"t\"]:+.3f}  hit {hr[\"k\"]}/{hr[\"n\"]}')"
        ),
        md("## So what?\n\n"
           "The famous *buy Hershey for the Halloween candy season* trade is folklore: the "
           "stock doesn't rally in (it mildly lags the market) and it doesn't pop after "
           "either. Nothing crosses the significance bar. The best cut — the 1-month "
           "pre-Halloween softness — is a sub-threshold, *opposite-direction* whiff. Verdict: "
           "**No signal, Mirage tradability.** The quants' notebook has the placebo, the "
           "jackknife, the costed leg and the synthetic control that shows the detector "
           "*would* have caught a real seasonal had one existed."),
    ]
    return new_notebook(cells=cells)


def build_quants():
    cells = [
        md("# Study 775 — Halloween-Candy — for the quants 🔬\n\n"
           "The full battery: one-sample *t* per window, a random-window placebo, a "
           "leave-one-out jackknife, the costed net leg, and a seeded synthetic positive "
           "control. Everything offline once cached; fingerprint "
           f"`{R['fp']}`."),
        code(PRELUDE),
        md("## 1. One-sample *t* across independent Halloween years\n\n"
           "Each Halloween is one independent event, so the unit is a one-sample *t* of the "
           "per-year abnormal return — **not** a daily panel (which would fake precision)."),
        code(
            "for label, col in [('2wk run-up','pre_s'),('1mo run-up','pre_l'),('2wk fade','post_s'),('1mo fade','post_l')]:\n"
            "    s = st.one_sample_t(inc[col].values)\n"
            "    print(f'{label:<12s} n={s[\"n\"]}  mean={s[\"mean\"]*100:+.3f}%  sd={s[\"sd\"]*100:.2f}%  t={s[\"t\"]:+.3f}')"
        ),
        md("## 2. Random-window placebo — is the run-up softness inside the luck cloud?\n\n"
           "For each event we redraw a random, non-Halloween 2-week window on HSY vs SPY and "
           "recompute the abnormal return; 20 seeds × 200 draws. If the observed mean sits "
           "in the tail of that null, it isn't ordinary tracking noise."),
        code(
            "pl = st.placebo_pvalue(ev, prices, 'pre_s', k=10, tail='right')\n"
            "import numpy as np\n"
            "hsy, spy = prices[dt.INSTRUMENT], prices[dt.BENCHMARK]\n"
            "common = hsy.index.intersection(spy.index).sort_values()\n"
            "rng = np.random.default_rng(999); draws = []\n"
            "for _ in range(4000):\n"
            "    vals = []\n"
            "    for _e in range(int(inc.shape[0])):\n"
            "        p = int(rng.integers(0, len(common)-11))\n"
            "        vals.append(float(hsy.loc[common[p+10]]/hsy.loc[common[p]] - spy.loc[common[p+10]]/spy.loc[common[p]]))\n"
            "    draws.append(np.mean(vals))\n"
            "fig, ax = plt.subplots(figsize=(9,4))\n"
            "ax.hist(np.array(draws)*100, bins=50, color='#8b949e', alpha=0.8)\n"
            "ax.axvline(pl['obs']*100, color='#e67e22', lw=2, label=f\"observed {pl['obs']*100:+.2f}%\")\n"
            "ax.set_xlabel('mean 2-week AR of random windows (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f\"pre-Halloween run-up vs luck cloud (right-tail p={pl['p_value']:.3f})\"); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('placebo:', {k: round(v,4) if isinstance(v,float) else v for k,v in pl.items()})"
        ),
        md("## 3. Jackknife — is the 1-month softness one bad year, or broad?"),
        code(
            "x = inc['pre_l'].values\n"
            "jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "print(f'full-sample t = {st.one_sample_t(x)[\"t\"]:+.3f}')\n"
            "print(f'jackknife t range [{min(jk):+.3f}, {max(jk):+.3f}] over {len(x)} leave-one-out draws')"
        ),
        md("## 4. Tradability — net of costs\n\n"
           "Calendar-known entry, so the signal window and the tradable window are the same "
           "(gross vs net). The long-into-Halloween trade *loses* money; costs only make it "
           "worse. Even the negative-drift 'short' version never reaches significance, and "
           "shorting a staple a few weeks a year is a capacity-trivial curiosity."),
        code(
            "ev10 = st.build_event_table(prices, cost_bps=10.0); inc10 = ev10[ev10['included']]\n"
            "for base, label in [('pre_s','2wk run-up'),('pre_l','1mo run-up')]:\n"
            "    g = st.one_sample_t(inc[base].values); n5 = st.one_sample_t(inc[base+'_net'].values); n10 = st.one_sample_t(inc10[base+'_net'].values)\n"
            "    print(f'{label:<12s} gross {g[\"mean\"]*100:+.3f}% (t={g[\"t\"]:+.2f})  net@5 {n5[\"mean\"]*100:+.3f}% (t={n5[\"t\"]:+.2f})  net@10 {n10[\"mean\"]*100:+.3f}% (t={n10[\"t\"]:+.2f})')"
        ),
        md("## 5. Synthetic positive control — the detector works, the absence is informative\n\n"
           "The one-sample-*t* detector must stay quiet on a planted-null world and recover a "
           "planted pre-Halloween bump. A real +2% seasonal run-up would light up at *t* ≈ "
           "+2.4 — so the flat/negative real result is a genuine null, not a power failure."),
        code(
            "null_ts = np.array([st.synthetic_detect(bump=0.0, seed=781+s, k=10)['t'] for s in range(20)])\n"
            "print(f'null: mean t={null_ts.mean():+.2f} sd={null_ts.std(ddof=1):.2f}  |t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "for b in (0.01, 0.02):\n"
            "    r = st.synthetic_detect(bump=b, seed=781, k=10)\n"
            "    print(f'planted +{b*100:.0f}%: mean AR {r[\"mean\"]*100:+.3f}%  t={r[\"t\"]:+.2f}')\n"
            "bumps = np.linspace(0, 0.03, 13)\n"
            "ts = [st.synthetic_detect(bump=b, seed=781, k=10)['t'] for b in bumps]\n"
            "fig, ax = plt.subplots(figsize=(8,4)); ax.plot(bumps*100, ts, 'o-', color='#2ea44f')\n"
            "ax.axhline(2, color='0.6', ls='--'); ax.set_xlabel('planted run-up bump (%)'); ax.set_ylabel('detector t')\n"
            "ax.set_title('planted bump is recovered monotonically'); plt.tight_layout(); plt.show()"
        ),
        md("## Verdict\n\n"
           "**Signal: None.** The bullish folklore is absent — every run-up and fade window "
           "is *negative* (HSY mildly lags SPY into and out of Oct 31), and no cut clears "
           "|*t*| ≥ 2 (strongest is the 1-month run-up at *t* = −1.76, placebo right-tail "
           "0.93, jackknife [−2.24, −1.46]). The synthetic control shows a real +2% bump "
           "would fire at *t* ≈ +2.4, so the absence is informative. **Tradability: Mirage.** "
           "The long trade loses money; the negative-drift short never reaches significance, "
           "is capacity-trivial, and sits inside a 4-window search."),
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
