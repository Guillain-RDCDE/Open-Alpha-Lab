"""Generate the two narrative notebooks for Study 773 (Spotify-Wrapped).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPOT/SPY
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (SPOT + SPY, yfinance,
# 2018-04-03 -> 2026-06-30; 8 of 10 Wrapped launches resolved). Fingerprint ee0e942455bb.
R = dict(
    n_events=10, n_included=8, fp="ee0e942455bb", rows=2072,
    pre_s_mean=-0.676, pre_s_t=-0.177, pre_s_hit=5, pre_s_n=8,
    pre_l_mean=-1.940, pre_l_t=-0.373, pre_l_hit=3,
    post_s_mean=+0.614, post_s_t=+0.239, post_s_hit=4,
    post_l_mean=+0.324, post_l_t=+0.152, post_l_hit=4,
    pl_pre_p=0.6352, pl_mean=+0.399, pl_sd=2.886, pl_post_p=0.5373,
    jk_lo=-0.817, jk_hi=+0.424,
    pre_s_net5=-0.776, pre_s_t5=-0.20, pre_s_net10=-0.876, pre_s_t10=-0.23,
    null_mean_t=-0.43, null_sd_t=1.43, null_hits=4,
    planted1_mean=+1.192, planted1_t=+1.37, planted2_mean=+2.192, planted2_t=+2.52,
)

PRELUDE = """\
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from spotify_wrapped import data as dt, strategy as st

if not dt.have_real():
    print("cache miss -> fetching SPOT + SPY once (needs network)")
    dt.fetch()
prices = dt.load_real()
ev = st.build_event_table(prices, cost_bps=5.0)
inc = ev[ev["included"]]
print(f"panel loaded; {len(inc)} of {len(dt.EVENTS)} Wrapped launches resolved")
"""


def build_curious():
    cells = [
        md("# Study 773 — Spotify-Wrapped 🎧\n\n"
           "*For the curious.* Every December, **Spotify Wrapped** takes over your feed — the "
           "viral, personalised year-in-review that everyone screenshots. The folklore trade: "
           "buy SPOT into the launch, because all that free buzz surely lifts the stock. We put "
           "the 8 post-IPO Wrapped seasons (2018→2025) on the stand and asked two plain "
           "questions: **does SPOT rally into the launch?** and **does it fade after?** "
           "The answer is a double bust."),
        md("## The claim, and why it's a clean test\n\n"
           "Wrapped ships in the same **late-Nov/early-Dec slot every year** (Nov 29 → Dec 6 "
           "since 2016), so the date is known well in advance. That makes *buy K sessions "
           "before, sell on the launch* calendar-known and zero-look-ahead. We measure SPOT's "
           "**abnormal** return (SPOT − SPY, total-return) so we're not just measuring beta. "
           "Caveat up front: SPOT only lists in April 2018, so this is a thin **n = 8** sample."),
        code(PRELUDE),
        md("## The Wrapped calendar we test (hardcoded from Spotify Newsroom / Wikipedia)\n\n"
           "2016 and 2017 are carried for completeness but have no SPOT tape (the direct "
           "listing is 2018-04-03), so they're excluded downstream."),
        code("pd.DataFrame(dt.EVENTS, columns=['year', 'wrapped_launch_date'])"),
        md("## The picture: mean cumulative abnormal return around the launch\n\n"
           "Offset 0 is the Wrapped launch. Left of zero is the *run-up* (the supposed rally "
           "in); right of zero is the *fade* window."),
        code(
            "car = st.car_path(ev, prices)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.axvline(0, color='0.4', lw=1, ls='--')\n"
            "ax.axhline(0, color='0.7', lw=0.8)\n"
            "ax.plot(car.index, car.values * 100, color='#1DB954', lw=2)\n"
            "ax.set_xlabel('trading days from the Wrapped launch (0 = event)')\n"
            "ax.set_ylabel('mean cumulative AR, SPOT − SPY (%)')\n"
            "ax.set_title('Spotify does NOT rally into its Wrapped launch')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(f"## The numbers\n\n"
           f"| window | mean AR | *t* | hit rate |\n|---|--:|--:|--:|\n"
           f"| 2-week run-up | {R['pre_s_mean']:+.2f}% | {R['pre_s_t']:+.2f} | {R['pre_s_hit']}/8 |\n"
           f"| 1-month run-up | {R['pre_l_mean']:+.2f}% | {R['pre_l_t']:+.2f} | {R['pre_l_hit']}/8 |\n"
           f"| 2-week fade | {R['post_s_mean']:+.2f}% | {R['post_s_t']:+.2f} | {R['post_s_hit']}/8 |\n"
           f"| 1-month fade | {R['post_l_mean']:+.2f}% | {R['post_l_t']:+.2f} | {R['post_l_hit']}/8 |\n\n"
           f"The bullish folklore is simply **absent**: both run-up windows are mildly "
           f"*negative* ({R['pre_s_mean']:+.2f}% and {R['pre_l_mean']:+.2f}%), and every single "
           f"*t* is under 0.4 in magnitude. The 'sell the news' fade? A clean +0.3–0.6% wisp, "
           f"a dead 50/50 hit rate. Nothing here is distinguishable from noise."),
        code(
            "rows = [('2wk run-up','pre_s'),('1mo run-up','pre_l'),('2wk fade','post_s'),('1mo fade','post_l')]\n"
            "for label, col in rows:\n"
            "    s = st.one_sample_t(inc[col].values); hr = st.hit_rate(inc[col].values)\n"
            "    print(f'{label:<12s} n={s[\"n\"]:2d}  mean={s[\"mean\"]*100:+.3f}%  t={s[\"t\"]:+.3f}  hit {hr[\"k\"]}/{hr[\"n\"]}')"
        ),
        md("## So what?\n\n"
           "The 'buy Spotify into Wrapped' trade is folklore. The stock doesn't rally in (it "
           "mildly, insignificantly sags) and it doesn't fade after. On just 8 post-IPO events, "
           "every cut sits dead-centre of its own noise — and SPOT's ordinary two-week swings "
           "(placebo sd ≈ 2.9%) dwarf any 'Wrapped' tilt. Verdict: **Signal None, Tradability "
           "Mirage.** The quants' notebook has the placebo, the jackknife, the costed leg and "
           "the synthetic control."),
    ]
    return new_notebook(cells=cells)


def build_quants():
    cells = [
        md("# Study 773 — Spotify-Wrapped — for the quants 🔬\n\n"
           "The full battery: one-sample *t* per window, a random-window placebo, a "
           "leave-one-out jackknife, the costed net leg, and a seeded synthetic positive "
           "control. Everything offline once cached; fingerprint "
           f"`{R['fp']}`. Note the sample size: **n = 8** (SPOT lists April 2018)."),
        code(PRELUDE),
        md("## 1. One-sample *t* across independent Wrapped years\n\n"
           "Each Wrapped launch is one independent event, so the unit is a one-sample *t* of "
           "the per-year abnormal return — **not** a daily panel (which would fake precision). "
           "With n = 8 these are small-sample statistics; read the point estimates."),
        code(
            "for label, col in [('2wk run-up','pre_s'),('1mo run-up','pre_l'),('2wk fade','post_s'),('1mo fade','post_l')]:\n"
            "    s = st.one_sample_t(inc[col].values)\n"
            "    print(f'{label:<12s} n={s[\"n\"]}  mean={s[\"mean\"]*100:+.3f}%  sd={s[\"sd\"]*100:.2f}%  t={s[\"t\"]:+.3f}')"
        ),
        md("## 2. Random-window placebo — is anything outside the luck cloud?\n\n"
           "For each event we redraw a random, non-launch 2-week window on SPOT vs SPY and "
           "recompute the abnormal return; 20 seeds × 200 draws. If the observed mean sits in "
           "the tail of that null, it isn't ordinary tracking noise. (It sits dead-centre.)"),
        code(
            "pl = st.placebo_pvalue(ev, prices, 'pre_s', k=10, tail='right')\n"
            "import numpy as np\n"
            "spot, spy = prices[dt.INSTRUMENT], prices[dt.BENCHMARK]\n"
            "common = spot.index.intersection(spy.index).sort_values()\n"
            "rng = np.random.default_rng(999); draws = []\n"
            "for _ in range(4000):\n"
            "    vals = []\n"
            "    for _e in range(int(inc.shape[0])):\n"
            "        p = int(rng.integers(0, len(common)-11))\n"
            "        vals.append(float(spot.loc[common[p+10]]/spot.loc[common[p]] - spy.loc[common[p+10]]/spy.loc[common[p]]))\n"
            "    draws.append(np.mean(vals))\n"
            "fig, ax = plt.subplots(figsize=(9,4))\n"
            "ax.hist(np.array(draws)*100, bins=50, color='#8b949e', alpha=0.8)\n"
            "ax.axvline(pl['obs']*100, color='#1DB954', lw=2, label=f\"observed {pl['obs']*100:+.2f}%\")\n"
            "ax.set_xlabel('mean 2-week AR of random windows (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f\"pre-launch run-up vs luck cloud (right-tail p={pl['p_value']:.3f})\"); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('placebo:', {k: round(v,4) if isinstance(v,float) else v for k,v in pl.items()})"
        ),
        md("## 3. Jackknife — is there any effect to be stable about?"),
        code(
            "x = inc['pre_s'].values\n"
            "jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "print(f'full-sample t = {st.one_sample_t(x)[\"t\"]:+.3f}')\n"
            "print(f'jackknife t range [{min(jk):+.3f}, {max(jk):+.3f}] over {len(x)} leave-one-out draws')"
        ),
        md("## 4. Tradability — net of costs\n\n"
           "Calendar-known entry, so the signal window and the tradable window are the same "
           "(gross vs net). But there is no gross edge to erode: every cut is a sub-1% wisp on "
           "8 events, swamped by a ~2.9% window-noise sd."),
        code(
            "ev10 = st.build_event_table(prices, cost_bps=10.0); inc10 = ev10[ev10['included']]\n"
            "for base, label in [('pre_s','2wk run-up'),('post_s','2wk fade')]:\n"
            "    g = st.one_sample_t(inc[base].values); n5 = st.one_sample_t(inc[base+'_net'].values); n10 = st.one_sample_t(inc10[base+'_net'].values)\n"
            "    print(f'{label:<12s} gross {g[\"mean\"]*100:+.3f}% (t={g[\"t\"]:+.2f})  net@5 {n5[\"mean\"]*100:+.3f}% (t={n5[\"t\"]:+.2f})  net@10 {n10[\"mean\"]*100:+.3f}% (t={n10[\"t\"]:+.2f})')"
        ),
        md("## 5. Synthetic positive control — the detector works, the tape has no bump\n\n"
           "The one-sample-*t* detector must stay quiet on a planted-null world and recover a "
           "planted pre-launch bump. Note the honest small-sample false-positive rate at "
           "n = 18 with a 10-day AR: |*t*| ≥ 2 fires on ~1/5 of null seeds — and the real study "
           "runs on n = 8, thinner still. The SPOT tape simply has no bump to find."),
        code(
            "null_ts = np.array([st.synthetic_detect(bump=0.0, seed=777+s, k=10)['t'] for s in range(20)])\n"
            "print(f'null: mean t={null_ts.mean():+.2f} sd={null_ts.std(ddof=1):.2f}  |t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "for b in (0.01, 0.02):\n"
            "    r = st.synthetic_detect(bump=b, seed=777, k=10)\n"
            "    print(f'planted +{b*100:.0f}%: mean AR {r[\"mean\"]*100:+.3f}%  t={r[\"t\"]:+.2f}')\n"
            "bumps = np.linspace(0, 0.03, 13)\n"
            "ts = [st.synthetic_detect(bump=b, seed=777, k=10)['t'] for b in bumps]\n"
            "fig, ax = plt.subplots(figsize=(8,4)); ax.plot(bumps*100, ts, 'o-', color='#1DB954')\n"
            "ax.axhline(2, color='0.6', ls='--'); ax.set_xlabel('planted run-up bump (%)'); ax.set_ylabel('detector t')\n"
            "ax.set_title('planted bump is recovered monotonically'); plt.tight_layout(); plt.show()"
        ),
        md("## Verdict\n\n"
           "**Signal: None.** Both the run-up (−0.68% / −1.94%, |*t*| < 0.4) and the fade "
           "(+0.61% / +0.32%, |*t*| < 0.25) are indistinguishable from zero; the placebo puts "
           "the observed means dead-centre (p ≈ 0.64, 0.54); the jackknife *t* straddles and "
           "flips sign; and it all rests on n = 8 post-IPO events. **Tradability: Mirage.** No "
           "gross edge, sign-unstable across horizon and leave-one-out, dwarfed by SPOT's "
           "~2.9% two-week idiosyncratic noise. The 'buy Spotify into Wrapped' trade is a story."),
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
