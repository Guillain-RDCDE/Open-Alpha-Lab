"""Generate the two narrative notebooks for Study 774 (Nintendo-Direct).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached NTDOY/SPY
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (NTDOY + SPY, yfinance,
# 2012-01-03 -> 2026-06-30; 25 of 25 Directs resolved). Fingerprint a7162a8938fd.
R = dict(
    n_events=25, n_included=25, fp="a7162a8938fd", rows=3643,
    pre_s_mean=-1.474, pre_s_t=-1.552, pre_s_hit=10, pre_s_n=25,
    pre_l_mean=-1.203, pre_l_t=-0.688, pre_l_hit=12,
    post_s_mean=+0.925, post_s_t=+0.774, post_s_hit=13,
    post_l_mean=-0.096, post_l_t=-0.053, post_l_hit=13,
    pl_pre_p=0.844, pl_mean=-0.028, pl_sd=1.507, pl_post_p=0.774,
    jk_lo=-1.829, jk_hi=-1.140,
    pre_s_net5=-1.574, pre_s_t5=-1.66, pre_s_net10=-1.674, pre_s_t10=-1.76,
    null_mean_t=-0.25, null_sd_t=1.06, null_hits=2,
    planted1_mean=+1.659, planted1_t=+1.44, planted2_mean=+2.659, planted2_t=+2.31,
)

PRELUDE = """\
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from nintendo_direct import data as dt, strategy as st

if not dt.have_real():
    print("cache miss -> fetching NTDOY + SPY once (needs network)")
    dt.fetch()
prices = dt.load_real()
ev = st.build_event_table(prices, cost_bps=5.0)
inc = ev[ev["included"]]
print(f"panel loaded; {len(inc)} of {len(dt.EVENTS)} Directs resolved")
"""


def build_curious():
    cells = [
        md("# Study 774 — Nintendo-Direct 🎮\n\n"
           "*For the curious.* Gaming Twitter *knows* you buy Nintendo into a Direct — the "
           "hype builds, the stock rips, and then you either ride a great showing or sell the "
           "news. We put 25 flagship Nintendo Directs (2013→2025) on the stand and asked the "
           "tape two plain questions: **does NTDOY rally into the broadcast?** and **does it "
           "move after?** The answer is a quiet double bust."),
        md("## The claim, and the honest catch\n\n"
           "A Nintendo Direct is Nintendo's ~40-minute video showcase — its biggest owned-media "
           "catalyst. The folklore: buy the hype in, sell the news out. The catch: a Direct is "
           "usually **announced only ~1-3 days ahead**, so *buy 2 weeks before* isn't really "
           "something you could have traded — you'd have needed to know the date. We test the "
           "run-up anyway (it's the folklore) and measure NTDOY's **abnormal** return "
           "(NTDOY − SPY, total-return) so we're not just measuring the market."),
        code(PRELUDE),
        md("## The Direct calendar we test (hardcoded from Wikipedia / Nintendo Life)"),
        code("pd.DataFrame(dt.EVENTS, columns=['air_date', 'label'])"),
        md("## The picture: mean cumulative abnormal return around the Direct\n\n"
           "Offset 0 is the Direct. Left of zero is the *run-up* (the supposed hype rally); "
           "right of zero is the *fade/pop* window."),
        code(
            "car = st.car_path(ev, prices)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.axvline(0, color='0.4', lw=1, ls='--')\n"
            "ax.axhline(0, color='0.7', lw=0.8)\n"
            "ax.plot(car.index, car.values * 100, color='#e60012', lw=2)\n"
            "ax.set_xlabel('trading days from the Direct (0 = broadcast)')\n"
            "ax.set_ylabel('mean cumulative AR, NTDOY − SPY (%)')\n"
            "ax.set_title('Nintendo does NOT rally into its Direct')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(f"## The numbers\n\n"
           f"| window | mean AR | *t* | hit rate |\n|---|--:|--:|--:|\n"
           f"| 2-week run-up | {R['pre_s_mean']:+.2f}% | {R['pre_s_t']:+.2f} | {R['pre_s_hit']}/25 |\n"
           f"| 1-month run-up | {R['pre_l_mean']:+.2f}% | {R['pre_l_t']:+.2f} | {R['pre_l_hit']}/25 |\n"
           f"| 2-week fade | {R['post_s_mean']:+.2f}% | {R['post_s_t']:+.2f} | {R['post_s_hit']}/25 |\n"
           f"| 1-month fade | {R['post_l_mean']:+.2f}% | {R['post_l_t']:+.2f} | {R['post_l_hit']}/25 |\n\n"
           f"The bullish folklore just **isn't there**: the run-up sign is mildly *negative* "
           f"({R['pre_s_mean']:+.2f}%) but nowhere near significant (*t* = {R['pre_s_t']:+.2f}, "
           f"hit rate {R['pre_s_hit']}/25 — a coin flip). And the 'sell the news' fade? A clean "
           f"zero (both cuts |*t*| < 0.8)."),
        code(
            "rows = [('2wk run-up','pre_s'),('1mo run-up','pre_l'),('2wk fade','post_s'),('1mo fade','post_l')]\n"
            "for label, col in rows:\n"
            "    s = st.one_sample_t(inc[col].values); hr = st.hit_rate(inc[col].values)\n"
            "    print(f'{label:<12s} n={s[\"n\"]:2d}  mean={s[\"mean\"]*100:+.3f}%  t={s[\"t\"]:+.3f}  hit {hr[\"k\"]}/{hr[\"n\"]}')"
        ),
        md("## So what?\n\n"
           "The famous *buy the hype into a Nintendo Direct* trade is folklore: NTDOY doesn't "
           "rally in (a mild, insignificant dip if anything) and doesn't reliably move after. "
           "On top of the missing signal, you couldn't have traded a clean 2-week run-up anyway "
           "(Directs are announced ~1-3 days ahead) and NTDOY is a thin OTC ADR. Verdict: "
           "**None signal, Mirage tradability.** The quants' notebook has the placebo, the "
           "jackknife, the costed leg and the synthetic control."),
    ]
    return new_notebook(cells=cells)


def build_quants():
    cells = [
        md("# Study 774 — Nintendo-Direct — for the quants 🔬\n\n"
           "The full battery: one-sample *t* per window, a random-window placebo, a "
           "leave-one-out jackknife, the costed net leg, and a seeded synthetic positive "
           "control. Everything offline once cached; fingerprint "
           f"`{R['fp']}`."),
        code(PRELUDE),
        md("## 1. One-sample *t* across independent Directs\n\n"
           "Each Direct is one independent event, so the unit is a one-sample *t* of the "
           "per-event abnormal return — **not** a daily panel (which would fake precision)."),
        code(
            "for label, col in [('2wk run-up','pre_s'),('1mo run-up','pre_l'),('2wk fade','post_s'),('1mo fade','post_l')]:\n"
            "    s = st.one_sample_t(inc[col].values)\n"
            "    print(f'{label:<12s} n={s[\"n\"]}  mean={s[\"mean\"]*100:+.3f}%  sd={s[\"sd\"]*100:.2f}%  t={s[\"t\"]:+.3f}')"
        ),
        md("## 2. Random-window placebo — is the dip inside the luck cloud?\n\n"
           "For each event we redraw a random, non-Direct 2-week window on NTDOY vs SPY and "
           "recompute the abnormal return; 20 seeds × 200 draws. If the observed mean sits "
           "in the tail of that null, it isn't ordinary tracking noise. Here it does not."),
        code(
            "pl = st.placebo_pvalue(ev, prices, 'pre_s', k=10, tail='right')\n"
            "import numpy as np\n"
            "ntdoy, spy = prices[dt.INSTRUMENT], prices[dt.BENCHMARK]\n"
            "common = ntdoy.index.intersection(spy.index).sort_values()\n"
            "rng = np.random.default_rng(999); draws = []\n"
            "for _ in range(4000):\n"
            "    vals = []\n"
            "    for _e in range(int(inc.shape[0])):\n"
            "        p = int(rng.integers(0, len(common)-11))\n"
            "        vals.append(float(ntdoy.loc[common[p+10]]/ntdoy.loc[common[p]] - spy.loc[common[p+10]]/spy.loc[common[p]]))\n"
            "    draws.append(np.mean(vals))\n"
            "fig, ax = plt.subplots(figsize=(9,4))\n"
            "ax.hist(np.array(draws)*100, bins=50, color='#8b949e', alpha=0.8)\n"
            "ax.axvline(pl['obs']*100, color='#e60012', lw=2, label=f\"observed {pl['obs']*100:+.2f}%\")\n"
            "ax.set_xlabel('mean 2-week AR of random windows (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f\"pre-Direct dip vs luck cloud (right-tail p={pl['p_value']:.3f})\"); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('placebo:', {k: round(v,4) if isinstance(v,float) else v for k,v in pl.items()})"
        ),
        md("## 3. Jackknife — is the dip one bad Direct, or broad?"),
        code(
            "x = inc['pre_s'].values\n"
            "jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "print(f'full-sample t = {st.one_sample_t(x)[\"t\"]:+.3f}')\n"
            "print(f'jackknife t range [{min(jk):+.3f}, {max(jk):+.3f}] over {len(x)} leave-one-out draws')"
        ),
        md("## 4. Tradability — net of costs (and the look-ahead problem)\n\n"
           "Even gross there is no significant edge; costs only enlarge the (untradeable) short "
           "of a negative window. And the entry date isn't known 2 weeks out — Directs are "
           "announced ~1-3 days ahead — while NTDOY is a thin OTC ADR with wide spreads."),
        code(
            "ev10 = st.build_event_table(prices, cost_bps=10.0); inc10 = ev10[ev10['included']]\n"
            "for base, label in [('pre_s','2wk run-up'),('post_s','2wk fade')]:\n"
            "    g = st.one_sample_t(inc[base].values); n5 = st.one_sample_t(inc[base+'_net'].values); n10 = st.one_sample_t(inc10[base+'_net'].values)\n"
            "    print(f'{label:<12s} gross {g[\"mean\"]*100:+.3f}% (t={g[\"t\"]:+.2f})  net@5 {n5[\"mean\"]*100:+.3f}% (t={n5[\"t\"]:+.2f})  net@10 {n10[\"mean\"]*100:+.3f}% (t={n10[\"t\"]:+.2f})')"
        ),
        md("## 5. Synthetic positive control — the detector works; the tape has no bump\n\n"
           "The one-sample-*t* detector must stay quiet on a planted-null world and recover a "
           "planted pre-Direct bump. At n = 25 with a 10-day AR the null fires |*t*| ≥ 2 on only "
           "~2/20 seeds — an honest, modest false-positive rate — and a real planted bump is "
           "recovered monotonically. The machinery is fine; the real NTDOY tape simply has "
           "nothing to find."),
        code(
            "null_ts = np.array([st.synthetic_detect(bump=0.0, seed=778+s, k=10)['t'] for s in range(20)])\n"
            "print(f'null: mean t={null_ts.mean():+.2f} sd={null_ts.std(ddof=1):.2f}  |t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "for b in (0.01, 0.02):\n"
            "    r = st.synthetic_detect(bump=b, seed=778, k=10)\n"
            "    print(f'planted +{b*100:.0f}%: mean AR {r[\"mean\"]*100:+.3f}%  t={r[\"t\"]:+.2f}')\n"
            "bumps = np.linspace(0, 0.03, 13)\n"
            "ts = [st.synthetic_detect(bump=b, seed=778, k=10)['t'] for b in bumps]\n"
            "fig, ax = plt.subplots(figsize=(8,4)); ax.plot(bumps*100, ts, 'o-', color='#e60012')\n"
            "ax.axhline(2, color='0.6', ls='--'); ax.set_xlabel('planted run-up bump (%)'); ax.set_ylabel('detector t')\n"
            "ax.set_title('planted bump is recovered monotonically'); plt.tight_layout(); plt.show()"
        ),
        md("## Verdict\n\n"
           "**Signal: None.** Neither half of the folklore survives — the run-up is an "
           "insignificant mild dip (*t* = −1.55, placebo right-tail ≈ 0.84, hit 40%) and the "
           "fade is a clean zero (|*t*| < 0.8). Nothing reaches significance or a placebo tail, "
           "while the synthetic control shows a real bump *would* have been caught. "
           "**Tradability: Mirage.** No edge gross or net, the entry date isn't known 2 weeks "
           "out (Directs announced ~1-3 days ahead), and NTDOY is a thin OTC ADR."),
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
