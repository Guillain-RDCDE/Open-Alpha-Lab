"""Generate the two narrative notebooks for Study 782 (CEO-Name-Length).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached tapes under
../_cache/ (fetching once on a cache miss) and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with no
network.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (40 names + SPY, yfinance,
# 2015-01-02 -> 2026-06-30; 137 monthly cross-sections). Fingerprint a7f273eb002d.
R = dict(
    n_names=40, n_months=137, fp="a7f273eb002d", rows=2889,
    ch_min=2, ch_max=11, ch_mean=6.05, ch_sd=1.80,
    ls_mean=-0.930, ls_sd=3.559, ls_t=-3.058, ls_sharpe=-0.905,
    hit_k=58, hit_n=137, hit_rate=42.3, hit_lo=34.4, hit_hi=50.7,
    ls_ann=-11.2, short_leg=+2.326, long_leg=+1.396,
    pl_obs=-0.930, pl_mean=-0.214, pl_sd=0.407, pl_p=0.0385, pl_draws=4000,
    jk_lo=-3.447, jk_hi=-2.232,
    net5_mean=-1.030, net5_t=-3.39, net10_mean=-1.130, net10_t=-3.72,
    q20_mean=-0.986, q20_t=-2.17, q33_mean=-0.930, q33_t=-3.06, q50_mean=-0.695, q50_t=-3.48,
    ls_beta_spy=-0.31,
    null_mean_t=-0.23, null_sd_t=1.22, null_hits=2,
    planted1_mean=+0.876, planted1_t=+4.27, planted2_mean=+1.779, planted2_t=+8.66,
)

PRELUDE = """\
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from ceo_name_length import data as dt, strategy as st

if not dt.have_real():
    print("cache miss -> fetching 40 names + SPY once (needs network)")
    dt.fetch()
prices = dt.load_real()
ch = dt.characteristics()
rets = dt.monthly_returns(prices)
ls = st.long_short_series(rets, ch)
print(f"panel loaded; {rets.shape[0]} monthly cross-sections x {rets.shape[1]} names")
"""


def build_curious():
    cells = [
        md("# Study 782 — CEO-Name-Length \U0001f4db\n\n"
           "*For the curious.* Here's a factor so silly it has to be a joke: **sort stocks by "
           "how many letters are in the CEO's surname.** Long the longest surnames, short the "
           "shortest, hold it market-neutral. It *cannot* work... so why does the tape hand us "
           "a *t*-stat of −3? This notebook is a lesson in how a nonsense signal fools you."),
        md("## The setup\n\n"
           "40 large caps, each tagged with its CEO's surname length (a **static 2026 "
           "snapshot** — we don't track who was CEO when; that's fine here *because the label "
           "is meaningless by design*). Sort into terciles, hold **long longest-surname / "
           "short shortest-surname**, rebalance monthly. A market with any sense should give a "
           "flat, boring zero."),
        code(PRELUDE),
        md("## The universe and its surname lengths"),
        code("pd.DataFrame([(t, s, dt.surname_len(s)) for t, s in dt.UNIVERSE],\n"
             "             columns=['ticker', 'ceo_surname', 'len']).sort_values('len').reset_index(drop=True)"),
        md(f"## The 'result' — and it looks real\n\n"
           f"| metric | value |\n|---|--:|\n"
           f"| monthly long/short mean | **{R['ls_mean']:+.2f}%** |\n"
           f"| *t* (n = {R['n_months']}) | **{R['ls_t']:+.2f}** |\n"
           f"| annualised | ≈ {R['ls_ann']:+.1f}%/yr |\n"
           f"| hit rate | {R['hit_k']}/{R['hit_n']} = {R['hit_rate']:.1f}% |\n\n"
           f"The **shortest**-surname tercile out-earned the longest by ~0.9%/month "
           f"({R['short_leg']:+.2f}% vs {R['long_leg']:+.2f}%). A *t* of −3 would get a real "
           f"factor published. So... surname length works?"),
        code(
            "s = st.one_sample_t(ls.values); hr = st.hit_rate(ls.values)\n"
            "print(f\"LS mean={s['mean']*100:+.3f}%/mo  t={s['t']:+.3f}  Sharpe={st.sharpe(ls.values):+.2f}  \"\n"
            "      f\"hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}%\")"
        ),
        md("## The reveal: *who* is in each leg\n\n"
           "Print the two legs and the illusion collapses — the **short** (shortest-surname) "
           "leg is just the megacap-tech winners list."),
        code(
            "long_names, short_names = st.leg_masks(ch)\n"
            "sn = {t: s for t, s in dt.UNIVERSE}\n"
            "print('SHORT leg (shortest surnames):', [f'{t}/{sn[t]}' for t in sorted(short_names)])\n"
            "print('LONG  leg (longest  surnames):', [f'{t}/{sn[t]}' for t in sorted(long_names)])"
        ),
        md("## The picture: it's a fixed sector bet, not a signal\n\n"
           "The cumulative long/short line just bleeds lower the whole decade — a persistent "
           "short-tech / long-value tilt that has nothing to do with orthography."),
        code(
            "cum = (1 + ls).cumprod()\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.axhline(1.0, color='0.7', lw=0.8)\n"
            "ax.plot(cum.index, cum.values, color='#c0392b', lw=2)\n"
            "ax.set_ylabel('growth of $1 (long long-names / short short-names)')\n"
            "ax.set_xlabel('date')\n"
            "ax.set_title('The \"surname-length factor\" is just short-tech / long-value')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(f"## So what?\n\n"
           f"The number is real; its *cause* is not. Short surnames coincidentally belonged to "
           f"the megacap-tech cohort (Cook, Jassy, Huang, Musk, Su), so a static 2026 snapshot "
           f"secretly encodes 'which names became winners.' There's no mechanism, the *t* "
           f"overstates a single fixed decade-long bet, and the placebo is only borderline "
           f"(p ≈ {R['pl_p']:.2f}). Verdict: **Weak signal, Mirage tradability** — the classic "
           f"data-snooping trap. The quants' notebook has the placebo, the jackknife, the costs "
           f"and the synthetic control."),
    ]
    return new_notebook(cells=cells)


def build_quants():
    cells = [
        md("# Study 782 — CEO-Name-Length — for the quants \U0001f52c\n\n"
           "The full battery on a null-by-design characteristic: one-sample *t* of the monthly "
           "long/short, a label-shuffle placebo, a leave-one-ticker-out jackknife, alternate "
           "cuts, the costed net leg, and a seeded synthetic positive control. Everything "
           f"offline once cached; fingerprint `{R['fp']}`."),
        code(PRELUDE),
        md("## 1. The headline long/short and its (over-stated) *t*\n\n"
           "One-sample *t* of the monthly LS return. Note the unit trap up front: the book is "
           "a **fixed** portfolio, so 137 months are *not* 137 independent bets — they're one "
           "decade-long sector tilt (Sharpe ≈ −0.9)."),
        code(
            "s = st.one_sample_t(ls.values); hr = st.hit_rate(ls.values)\n"
            "print(f\"n={s['n']}  mean={s['mean']*100:+.3f}%/mo  sd={s['sd']*100:.3f}%  t={s['t']:+.3f}  \"\n"
            "      f\"Sharpe={st.sharpe(ls.values):+.2f}  hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}%\")\n"
            "spy = prices['SPY'].resample('ME').last().pct_change().reindex(rets.index)\n"
            "beta = np.polyfit(spy.dropna().values, ls.reindex(spy.dropna().index).values, 1)[0]\n"
            "print(f'realised LS beta to SPY = {beta:+.3f}  (dollar-neutral but NOT beta-neutral)')"
        ),
        md("## 2. Label-shuffle placebo — is the spread anything but tercile luck?\n\n"
           "Permute the surname-length labels across the 40 names and recompute the LS mean; "
           "20 seeds × 200 draws. If surname length carried information the observed spread "
           "would sit in the tail. It sits only *borderline* in the tail (two-tail p ≈ 0.04) — "
           "and that's before penalising the direction + spec search."),
        code(
            "pl = st.placebo_pvalue(rets, ch, tail='two')\n"
            "vals = ch.values\n"
            "rng = np.random.default_rng(4242); draws = []\n"
            "for _ in range(4000):\n"
            "    perm = pd.Series(rng.permutation(vals), index=ch.index)\n"
            "    draws.append(float(st.long_short_series(rets, perm).mean()))\n"
            "fig, ax = plt.subplots(figsize=(9, 4))\n"
            "ax.hist(np.array(draws)*100, bins=50, color='#8b949e', alpha=0.85)\n"
            "ax.axvline(pl['obs']*100, color='#c0392b', lw=2, label=f\"observed {pl['obs']*100:+.2f}%\")\n"
            "ax.set_xlabel('mean monthly LS of shuffled surname-length labels (%)'); ax.set_ylabel('count')\n"
            "ax.set_title(f\"surname-length spread vs label-shuffle cloud (two-tail p={pl['p_value']:.3f})\"); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('placebo:', {k: round(v,4) if isinstance(v,float) else v for k,v in pl.items()})"
        ),
        md("## 3. Jackknife — one name, or the whole cohort?"),
        code(
            "jk = st.jackknife_t(rets, ch)\n"
            "print(f\"full-sample t = {st.one_sample_t(ls.values)['t']:+.3f}\")\n"
            "print(f\"jackknife t range [{jk['lo']:+.3f}, {jk['hi']:+.3f}] over {jk['n']} leave-one-out draws\")\n"
            "print('=> stable, but stability of a whole-cohort sector tilt is not evidence of a real signal')"
        ),
        md("## 4. Alternate cuts + costs\n\n"
           "The sign survives every breakpoint (the sector tilt is present at all of them), "
           "and costs make the short look 'stronger' — but it's an ex-post short-tech bet, not "
           "a repeatable edge."),
        code(
            "for q, lab in [(0.20,'quintile'),(1/3,'tercile'),(0.50,'median')]:\n"
            "    sq = st.one_sample_t(st.long_short_series(rets, ch, q=q).values)\n"
            "    print(f'{lab:<9s} q={q:.2f}: mean {sq[\"mean\"]*100:+.3f}%/mo  t={sq[\"t\"]:+.2f}')\n"
            "for cb in (0.0, 5.0, 10.0):\n"
            "    sn = st.one_sample_t(st.long_short_series(rets, ch, cost_bps=cb).values)\n"
            "    tag = 'gross' if cb == 0 else f'net@{cb:.0f}bps'\n"
            "    print(f'{tag:<10s}: mean {sn[\"mean\"]*100:+.3f}%/mo  t={sn[\"t\"]:+.2f}')"
        ),
        md("## 5. Synthetic positive control — the detector works, the real spread is a confound\n\n"
           "The sort must stay quiet on an inert label (bump = 0) and recover a *planted* "
           "characteristic→return slope monotonically. It does — which is precisely why we "
           "trust that the real-tape −3.06 is a **confound**, not a planted slope: there is no "
           "mechanism to plant."),
        code(
            "null_ts = np.array([st.synthetic_detect(bump=0.0, seed=798+i)['t'] for i in range(20)])\n"
            "print(f'null: mean t={null_ts.mean():+.2f} sd={null_ts.std(ddof=1):.2f}  |t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "for b in (0.004, 0.008):\n"
            "    r = st.synthetic_detect(bump=b, seed=798)\n"
            "    print(f'planted +{b:.3f}: mean {r[\"mean\"]*100:+.3f}%/mo  t={r[\"t\"]:+.2f}')\n"
            "bumps = np.linspace(0, 0.01, 11)\n"
            "ts = [st.synthetic_detect(bump=b, seed=798)['t'] for b in bumps]\n"
            "fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(bumps, ts, 'o-', color='#2ea44f')\n"
            "ax.axhline(2, color='0.6', ls='--'); ax.set_xlabel('planted characteristic->return slope'); ax.set_ylabel('detector t')\n"
            "ax.set_title('planted slope is recovered monotonically'); plt.tight_layout(); plt.show()"
        ),
        md(f"## Verdict\n\n"
           f"**Signal: Weak.** The nominal stats cross the bar (*t* = {R['ls_t']:+.2f}, placebo "
           f"two-tail p ≈ {R['pl_p']:.2f}, jackknife-stable [{R['jk_lo']:+.2f}, {R['jk_hi']:+.2f}], "
           f"consistent across cuts) so it is **not** noise — but it is a textbook confound: a "
           f"static 2026 CEO snapshot turns the sort into a fixed short-tech/long-value bet "
           f"(short leg = AAPL/AMZN/NVDA/TSLA/AMD), the monthly *t* overstates one ~11-yr "
           f"portfolio (Sharpe {R['ls_sharpe']:+.2f}, driven by 2020 & 2026), and there is zero "
           f"mechanism. Real number, wrong cause. **Tradability: Mirage.** An unrepeatable, "
           f"look-ahead-contaminated sector coincidence dressed up as a name-length factor."),
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
