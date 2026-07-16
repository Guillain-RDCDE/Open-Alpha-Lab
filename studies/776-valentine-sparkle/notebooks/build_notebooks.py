"""Generate the two narrative notebooks for Study 776 (Valentine-Sparkle).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SIG/SPY
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (SIG + SPY, yfinance,
# 2008-01-02 -> 2026-06-30; 18 of 18 Valentine's Days resolved). Fingerprint 55e1ce2a38e2.
R = dict(
    n_events=18, n_included=18, fp="55e1ce2a38e2", rows=4652,
    pre_s_mean=-1.275, pre_s_t=-0.929, pre_s_hit=7, pre_s_n=18,
    pre_l_mean=-2.714, pre_l_t=-1.021, pre_l_hit=8,
    post_s_mean=+1.137, post_s_t=+0.516, post_s_hit=9,
    post_l_mean=+0.869, post_l_t=+0.224, post_l_hit=8,
    pl_pre_p=0.786, pl_mean=+0.469, pl_sd=2.270, pl_post_p=0.625,
    jk_lo=-1.956, jk_hi=-0.521,
    pre_s_net5=-1.375, pre_s_t5=-1.00, pre_s_net10=-1.475, pre_s_t10=-1.08,
    null_mean_t=-0.61, null_sd_t=1.49, null_hits=3,
    planted1_mean=+0.449, planted1_t=+0.52, planted2_mean=+1.449, planted2_t=+1.68,
)

PRELUDE = """\
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from valentine_sparkle import data as dt, strategy as st

if not dt.have_real():
    print("cache miss -> fetching SIG + SPY once (needs network)")
    dt.fetch()
prices = dt.load_real()
ev = st.build_event_table(prices, cost_bps=5.0)
inc = ev[ev["included"]]
print(f"panel loaded; {len(inc)} of {len(dt.EVENTS)} Valentine's Days resolved")
"""


def build_curious():
    cells = [
        md("# Study 776 — Valentine-Sparkle 💎\n\n"
           "*For the curious.* Everyone *knows* jewellers rally into Valentine's Day — Signet "
           "(Kay, Zales, Jared) owns the biggest gifting quarter on the retail calendar, so the "
           "stock should be bid up into February 14 and fade after. We put 18 Valentine's Days "
           "(2009→2026) on the stand and asked the tape two plain questions: **does SIG rally "
           "into the holiday?** and **does it fade after?** The answer is a double bust."),
        md("## The claim, and why it's a clean test\n\n"
           "Valentine's Day is the year's first big diamond/engagement-gift peak, and its date "
           "is **fixed years ahead** (February 14, every year). So *buy K sessions before, sell "
           "on/after the date* is calendar-known and zero-look-ahead. We measure SIG's "
           "**abnormal** return (SIG − SPY, total-return) so we're not just measuring beta — "
           "Signet is a volatile mid-cap with a market beta well above 1."),
        code(PRELUDE),
        md("## The Valentine's calendar we test (fixed February 14 each year)"),
        code("pd.DataFrame(dt.EVENTS, columns=['year', 'valentines_date'])"),
        md("## The picture: mean cumulative abnormal return around Valentine's Day\n\n"
           "Offset 0 is the Valentine's anchor. Left of zero is the *run-up* (the supposed "
           "rally in); right of zero is the *fade* window."),
        code(
            "car = st.car_path(ev, prices)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.axvline(0, color='0.4', lw=1, ls='--')\n"
            "ax.axhline(0, color='0.7', lw=0.8)\n"
            "ax.plot(car.index, car.values * 100, color='#c0392b', lw=2)\n"
            "ax.set_xlabel('trading days from Valentine\\'s Day (0 = Feb 14 anchor)')\n"
            "ax.set_ylabel('mean cumulative AR, SIG − SPY (%)')\n"
            "ax.set_title('Signet does NOT rally into Valentine\\'s Day')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(f"## The numbers\n\n"
           f"| window | mean AR | *t* | hit rate |\n|---|--:|--:|--:|\n"
           f"| 2-week run-up | **{R['pre_s_mean']:+.2f}%** | **{R['pre_s_t']:+.2f}** | {R['pre_s_hit']}/18 |\n"
           f"| 1-month run-up | {R['pre_l_mean']:+.2f}% | {R['pre_l_t']:+.2f} | {R['pre_l_hit']}/18 |\n"
           f"| 2-week fade | {R['post_s_mean']:+.2f}% | {R['post_s_t']:+.2f} | {R['post_s_hit']}/18 |\n"
           f"| 1-month fade | {R['post_l_mean']:+.2f}% | {R['post_l_t']:+.2f} | {R['post_l_hit']}/18 |\n\n"
           f"The bullish folklore is **absent — and mildly inverted**: in the two weeks *into* "
           f"Valentine's, SIG has *under*-performed SPY by {abs(R['pre_s_mean']):.2f}% on average "
           f"(*t* = {R['pre_s_t']:+.2f}), and over a month by {abs(R['pre_l_mean']):.2f}%. Neither "
           f"is close to significant — but both point the *wrong* way for a rally. And the 'sell "
           f"the holiday' fade? SIG is, if anything, *up* afterwards (both |*t*| < 0.6)."),
        code(
            "rows = [('2wk run-up','pre_s'),('1mo run-up','pre_l'),('2wk fade','post_s'),('1mo fade','post_l')]\n"
            "for label, col in rows:\n"
            "    s = st.one_sample_t(inc[col].values); hr = st.hit_rate(inc[col].values)\n"
            "    print(f'{label:<12s} n={s[\"n\"]:2d}  mean={s[\"mean\"]*100:+.3f}%  t={s[\"t\"]:+.3f}  hit {hr[\"k\"]}/{hr[\"n\"]}')"
        ),
        md("## So what?\n\n"
           "The famous *buy the jeweller into Valentine's Day* trade is folklore twice over: "
           "the stock doesn't rally in (it drifts mildly *below* SPY) and it doesn't fade after. "
           "Every cut sits inside the noise. Verdict: **None signal, Mirage tradability.** The "
           "quants' notebook has the placebo, the jackknife, the costed leg and the synthetic "
           "control."),
    ]
    return new_notebook(cells=cells)


def build_quants():
    cells = [
        md("# Study 776 — Valentine-Sparkle — for the quants 🔬\n\n"
           "The full battery: one-sample *t* per window, a random-window placebo, a "
           "leave-one-out jackknife, the costed net leg, and a seeded synthetic positive "
           "control. Everything offline once cached; fingerprint "
           f"`{R['fp']}`."),
        code(PRELUDE),
        md("## 1. One-sample *t* across independent Valentine's years\n\n"
           "Each Valentine's Day is one independent event, so the unit is a one-sample *t* of "
           "the per-year abnormal return — **not** a daily panel (which would fake precision)."),
        code(
            "for label, col in [('2wk run-up','pre_s'),('1mo run-up','pre_l'),('2wk fade','post_s'),('1mo fade','post_l')]:\n"
            "    s = st.one_sample_t(inc[col].values)\n"
            "    print(f'{label:<12s} n={s[\"n\"]}  mean={s[\"mean\"]*100:+.3f}%  sd={s[\"sd\"]*100:.2f}%  t={s[\"t\"]:+.3f}')"
        ),
        md("## 2. Random-window placebo — is the run-up inside the luck cloud?\n\n"
           "For each event we redraw a random, non-Valentine's 2-week window on SIG vs SPY and "
           "recompute the abnormal return; 20 seeds × 200 draws. If the observed mean sits in "
           "the tail of that null, it isn't ordinary tracking noise — here it lands dead-centre."),
        code(
            "pl = st.placebo_pvalue(ev, prices, 'pre_s', k=10, tail='right')\n"
            "import numpy as np\n"
            "sig, spy = prices[dt.INSTRUMENT], prices[dt.BENCHMARK]\n"
            "common = sig.index.intersection(spy.index).sort_values()\n"
            "rng = np.random.default_rng(999); draws = []\n"
            "for _ in range(4000):\n"
            "    vals = []\n"
            "    for _e in range(int(inc.shape[0])):\n"
            "        p = int(rng.integers(0, len(common)-11))\n"
            "        vals.append(float(sig.loc[common[p+10]]/sig.loc[common[p]] - spy.loc[common[p+10]]/spy.loc[common[p]]))\n"
            "    draws.append(np.mean(vals))\n"
            "fig, ax = plt.subplots(figsize=(9,4))\n"
            "ax.hist(np.array(draws)*100, bins=50, color='#8b949e', alpha=0.8)\n"
            "ax.axvline(pl['obs']*100, color='#c0392b', lw=2, label=f\"observed {pl['obs']*100:+.2f}%\")\n"
            "ax.set_xlabel('mean 2-week AR of random windows (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f\"pre-Valentine's run-up vs luck cloud (right-tail p={pl['p_value']:.3f})\"); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('placebo:', {k: round(v,4) if isinstance(v,float) else v for k,v in pl.items()})"
        ),
        md("## 3. Jackknife — is there any dominant year, or just noise?"),
        code(
            "x = inc['pre_s'].values\n"
            "jk = [st.one_sample_t(np.delete(x, i))['t'] for i in range(len(x))]\n"
            "print(f'full-sample t = {st.one_sample_t(x)[\"t\"]:+.3f}')\n"
            "print(f'jackknife t range [{min(jk):+.3f}, {max(jk):+.3f}] over {len(x)} leave-one-out draws')"
        ),
        md("## 4. Tradability — net of costs\n\n"
           "Calendar-known entry, so the signal window and the tradable window are the same "
           "(gross vs net). Costs only nudge the (insignificant) run-up more negative and the "
           "(insignificant) fade less positive — no window ever clears |*t*| = 1.1."),
        code(
            "ev10 = st.build_event_table(prices, cost_bps=10.0); inc10 = ev10[ev10['included']]\n"
            "for base, label in [('pre_s','2wk run-up'),('post_s','2wk fade')]:\n"
            "    g = st.one_sample_t(inc[base].values); n5 = st.one_sample_t(inc[base+'_net'].values); n10 = st.one_sample_t(inc10[base+'_net'].values)\n"
            "    print(f'{label:<12s} gross {g[\"mean\"]*100:+.3f}% (t={g[\"t\"]:+.2f})  net@5 {n5[\"mean\"]*100:+.3f}% (t={n5[\"t\"]:+.2f})  net@10 {n10[\"mean\"]*100:+.3f}% (t={n10[\"t\"]:+.2f})')"
        ),
        md("## 5. Synthetic positive control — the detector works, and the null is noisy\n\n"
           "The one-sample-*t* detector must stay quiet on a planted-null world and recover a "
           "planted pre-Valentine's bump. Note the honest small-sample false-positive rate at "
           "n = 18 with a 10-day AR: |*t*| ≥ 2 fires on ~3/20 null seeds — a reason not to read "
           "much into any single |*t*| near 2."),
        code(
            "null_ts = np.array([st.synthetic_detect(bump=0.0, seed=782+s, k=10)['t'] for s in range(20)])\n"
            "print(f'null: mean t={null_ts.mean():+.2f} sd={null_ts.std(ddof=1):.2f}  |t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "for b in (0.01, 0.02):\n"
            "    r = st.synthetic_detect(bump=b, seed=782, k=10)\n"
            "    print(f'planted +{b*100:.0f}%: mean AR {r[\"mean\"]*100:+.3f}%  t={r[\"t\"]:+.2f}')\n"
            "bumps = np.linspace(0, 0.03, 13)\n"
            "ts = [st.synthetic_detect(bump=b, seed=782, k=10)['t'] for b in bumps]\n"
            "fig, ax = plt.subplots(figsize=(8,4)); ax.plot(bumps*100, ts, 'o-', color='#2ea44f')\n"
            "ax.axhline(2, color='0.6', ls='--'); ax.set_xlabel('planted run-up bump (%)'); ax.set_ylabel('detector t')\n"
            "ax.set_title('planted bump is recovered monotonically'); plt.tight_layout(); plt.show()"
        ),
        md("## Verdict\n\n"
           "**Signal: None.** No rally into Valentine's and no fade after — every window is "
           "insignificant (2wk run-up −1.28%, *t* = −0.93; 1mo run-up −2.71%, *t* = −1.02; both "
           "fades |*t*| < 0.6), the placebo sits dead-centre (right-tail p = 0.79), the jackknife "
           "*t* never leaves [−1.96, −0.52], and the null itself fires |*t*| ≥ 2 on 3/20 seeds. "
           "The point estimates even lean the *wrong* way for the folklore. **Tradability: "
           "Mirage.** No window clears |*t*| = 1.1 gross or net; the only faintly-directional cut "
           "is a capacity-trivial, statistically empty short inside a 4-window search."),
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
