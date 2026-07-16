"""Generate the two narrative notebooks for Study 772 (Album-Drop).

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
# 2018-04-03 -> 2026-06-30; 27 of 27 drops resolved). Fingerprint 5406b18c6109.
R = dict(
    n_events=27, n_included=27, fp="5406b18c6109", rows=2072,
    pre_s_mean=+0.219, pre_s_t=+0.123, pre_s_hit=10, pre_s_n=27,
    pre_l_mean=+1.436, pre_l_t=+0.602, pre_l_hit=15,
    post_s_mean=-1.158, post_s_t=-0.709, post_s_hit=14,
    post_l_mean=-1.622, post_l_t=-0.928, post_l_hit=13,
    pl_pre_p=0.533, pl_mean=+0.368, pl_sd=1.557, pl_post_p=0.833,
    jk_lo=-1.178, jk_hi=-0.275,
    post_s_net5=-1.258, post_s_t5=-0.77, post_s_net10=-1.358, post_s_t10=-0.83,
    null_mean_t=+0.11, null_sd_t=1.21, null_hits=2,
    planted1_mean=+2.017, planted1_t=+2.16, planted2_mean=+3.017, planted2_t=+3.23,
)

PRELUDE = """\
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from album_drop import data as dt, strategy as st

if not dt.have_real():
    print("cache miss -> fetching SPOT + SPY once (needs network)")
    dt.fetch()
prices = dt.load_real()
ev = st.build_event_table(prices, cost_bps=5.0)
inc = ev[ev["included"]]
print(f"panel loaded; {len(inc)} of {len(dt.EVENTS)} drops resolved")
"""


def build_curious():
    cells = [
        md("# Study 772 — Album-Drop 🎧\n\n"
           "*For the curious.* When Taylor Swift, Drake or Adele drops a record that shatters "
           "Spotify's single-day streaming record, surely the *stock* jumps? We put 27 "
           "genuinely blockbuster album releases (2018→2024) on the stand and asked the tape "
           "two plain questions: **does SPOT rally into the drop?** and **does it pop after?** "
           "The answer is a flat, boring zero."),
        md("## The claim, and why it's a clean test\n\n"
           "A mega-album's release date is **known weeks ahead** (pre-orders, teaser singles), "
           "so *buy K sessions before, sell on the day* is calendar-known and zero-look-ahead. "
           "We measure SPOT's **abnormal** return (SPOT − SPY, total-return) so we're not just "
           "measuring the market or SPOT's beta. The economic prior is flat: Spotify earns from "
           "*subscriptions*, not a per-stream cut one album can move."),
        code(PRELUDE),
        md("## The album calendar we test (hardcoded from press releases)"),
        code("pd.DataFrame(dt.EVENTS, columns=['artist — album', 'release_date'])"),
        md("## The picture: mean cumulative abnormal return around the drop\n\n"
           "Offset 0 is the release. Left of zero is the *run-up* (the supposed rally in); "
           "right of zero is the *reaction* window."),
        code(
            "car = st.car_path(ev, prices)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.axvline(0, color='0.4', lw=1, ls='--')\n"
            "ax.axhline(0, color='0.7', lw=0.8)\n"
            "ax.plot(car.index, car.values * 100, color='#1db954', lw=2)\n"
            "ax.set_xlabel('trading days from the album release (0 = drop)')\n"
            "ax.set_ylabel('mean cumulative AR, SPOT − SPY (%)')\n"
            "ax.set_title('A blockbuster drop barely wiggles Spotify stock')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(f"## The numbers\n\n"
           f"| window | mean AR | *t* | hit rate |\n|---|--:|--:|--:|\n"
           f"| 2-week run-up | {R['pre_s_mean']:+.2f}% | {R['pre_s_t']:+.2f} | {R['pre_s_hit']}/27 |\n"
           f"| 1-month run-up | {R['pre_l_mean']:+.2f}% | {R['pre_l_t']:+.2f} | {R['pre_l_hit']}/27 |\n"
           f"| 2-week reaction | {R['post_s_mean']:+.2f}% | {R['post_s_t']:+.2f} | {R['post_s_hit']}/27 |\n"
           f"| 1-month reaction | {R['post_l_mean']:+.2f}% | {R['post_l_t']:+.2f} | {R['post_l_hit']}/27 |\n\n"
           f"Every single cut is indistinguishable from noise — all |*t*| < 1, every hit rate "
           f"near a coin-flip. The drop lights up the *charts*, not the *stock*."),
        code(
            "rows = [('2wk run-up','pre_s'),('1mo run-up','pre_l'),('2wk reaction','post_s'),('1mo reaction','post_l')]\n"
            "for label, col in rows:\n"
            "    s = st.one_sample_t(inc[col].values); hr = st.hit_rate(inc[col].values)\n"
            "    print(f'{label:<14s} n={s[\"n\"]:2d}  mean={s[\"mean\"]*100:+.3f}%  t={s[\"t\"]:+.3f}  hit {hr[\"k\"]}/{hr[\"n\"]}')"
        ),
        md("## So what?\n\n"
           "The famous *buy Spotify when the big album drops* idea is folklore: the stock "
           "doesn't rally in and doesn't pop after. One record-breaking streaming week is a "
           "rounding error against a ~600M-user subscription business the market has already "
           "priced. Verdict: **None signal, Mirage tradability.** The quants' notebook has the "
           "placebo, the jackknife, the costed leg and the synthetic control that *would* have "
           "caught a real effect."),
    ]
    return new_notebook(cells=cells)


def build_quants():
    cells = [
        md("# Study 772 — Album-Drop — for the quants 🔬\n\n"
           "The full battery: one-sample *t* per window, a random-window placebo, a "
           "leave-one-out jackknife, the costed net leg, and a seeded synthetic positive "
           "control. Everything offline once cached; fingerprint "
           f"`{R['fp']}`."),
        code(PRELUDE),
        md("## 1. One-sample *t* across independent album events\n\n"
           "Each album drop is one independent event, so the unit is a one-sample *t* of the "
           "per-event abnormal return — **not** a daily panel (which would fake precision)."),
        code(
            "for label, col in [('2wk run-up','pre_s'),('1mo run-up','pre_l'),('2wk reaction','post_s'),('1mo reaction','post_l')]:\n"
            "    s = st.one_sample_t(inc[col].values)\n"
            "    print(f'{label:<14s} n={s[\"n\"]}  mean={s[\"mean\"]*100:+.3f}%  sd={s[\"sd\"]*100:.2f}%  t={s[\"t\"]:+.3f}')"
        ),
        md("## 2. Random-window placebo — is either move inside the luck cloud?\n\n"
           "For each event we redraw a random, non-drop 2-week window on SPOT vs SPY and "
           "recompute the abnormal return; 20 seeds × 200 draws. If the observed mean sits in "
           "the tail of that null, it isn't ordinary tracking noise. (Spoiler: it is.)"),
        code(
            "pl = st.placebo_pvalue(ev, prices, 'post_s', k=10, tail='right')\n"
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
            "ax.axvline(pl['obs']*100, color='#1db954', lw=2, label=f\"observed {pl['obs']*100:+.2f}%\")\n"
            "ax.set_xlabel('mean 2-week AR of random windows (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f\"post-drop reaction vs luck cloud (right-tail p={pl['p_value']:.3f})\"); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('placebo:', {k: round(v,4) if isinstance(v,float) else v for k,v in pl.items()})"
        ),
        md("## 3. Jackknife — is the tiny negative one bad album, or broad?"),
        code(
            "x = inc['post_s'].values\n"
            "jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "print(f'full-sample t = {st.one_sample_t(x)[\"t\"]:+.3f}')\n"
            "print(f'jackknife t range [{min(jk):+.3f}, {max(jk):+.3f}] over {len(x)} leave-one-out draws')"
        ),
        md("## 4. Tradability — net of costs\n\n"
           "Calendar-known entry, so the signal window and the tradable window are the same "
           "(gross vs net). The run-up is a rounding error that costs erase; the reaction is a "
           "sub-1-*t* negative. No edge in any cut."),
        code(
            "ev10 = st.build_event_table(prices, cost_bps=10.0); inc10 = ev10[ev10['included']]\n"
            "for base, label in [('pre_s','2wk run-up'),('post_s','2wk reaction')]:\n"
            "    g = st.one_sample_t(inc[base].values); n5 = st.one_sample_t(inc[base+'_net'].values); n10 = st.one_sample_t(inc10[base+'_net'].values)\n"
            "    print(f'{label:<14s} gross {g[\"mean\"]*100:+.3f}% (t={g[\"t\"]:+.2f})  net@5 {n5[\"mean\"]*100:+.3f}% (t={n5[\"t\"]:+.2f})  net@10 {n10[\"mean\"]*100:+.3f}% (t={n10[\"t\"]:+.2f})')"
        ),
        md("## 5. Synthetic positive control — the detector works, the effect is just absent\n\n"
           "The one-sample-*t* detector must stay quiet on a planted-null world and recover a "
           "planted pre-drop bump. It does both — so the flat real result is a true null, not a "
           "broken test."),
        code(
            "null_ts = np.array([st.synthetic_detect(bump=0.0, seed=776+s, k=10)['t'] for s in range(20)])\n"
            "print(f'null: mean t={null_ts.mean():+.2f} sd={null_ts.std(ddof=1):.2f}  |t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "for b in (0.01, 0.02):\n"
            "    r = st.synthetic_detect(bump=b, seed=776, k=10)\n"
            "    print(f'planted +{b*100:.0f}%: mean AR {r[\"mean\"]*100:+.3f}%  t={r[\"t\"]:+.2f}')\n"
            "bumps = np.linspace(0, 0.03, 13)\n"
            "ts = [st.synthetic_detect(bump=b, seed=776, k=10)['t'] for b in bumps]\n"
            "fig, ax = plt.subplots(figsize=(8,4)); ax.plot(bumps*100, ts, 'o-', color='#1db954')\n"
            "ax.axhline(2, color='0.6', ls='--'); ax.set_xlabel('planted run-up bump (%)'); ax.set_ylabel('detector t')\n"
            "ax.set_title('planted bump is recovered monotonically'); plt.tight_layout(); plt.show()"
        ),
        md("## Verdict\n\n"
           "**Signal: None.** Every real cut is indistinguishable from noise (run-up +0.22% "
           "*t* +0.12 / +1.44% *t* +0.60; reaction −1.16% *t* −0.71 / −1.62% *t* −0.93), placebo "
           "p = 0.53 / 0.83 dead-centre, jackknife never near significance — while the synthetic "
           "control cleanly recovers a planted +1%/+2% bump. **Tradability: Mirage.** No edge in "
           "any window, gross or net; a record streaming week is a rounding error against "
           "Spotify's subscription base."),
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
