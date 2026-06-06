# The source articles (the work this repository responds to)

This project is an **independent, reproducible response** to Bruce Knuteson's
claim that the overnight-vs-intraday return pattern is the fingerprint of
large-scale market manipulation. To judge our response, read his work first.

## Get the papers — one command

```bash
python papers/download_papers.py
```

This fetches every openly available paper below **straight from its official
source** (arXiv, the authors' university pages, the NY Fed, publishers' open
repositories) into this folder. We deliberately do **not** commit the PDFs: they
are the authors' copyrighted work (e.g. the arXiv papers are under arXiv's
[`nonexclusive-distrib/1.0`](http://arxiv.org/licenses/nonexclusive-distrib/1.0/)
licence, which grants distribution rights to arXiv, not to third parties). The
right thing — and the more credible thing — is to point you at the primary source.

## Knuteson's articles (the work this responds to)

| Work | Year | Source | Auto-fetch | Role |
|---|---|---|---|---|
| **Nothing to See Here: How to Say It When You Need to** | 2023 | [SSRN 4619084](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4619084) | ✋ SSRN only | the central pamphlet; no free copy exists, but its data/figures are fully covered by the 2019 + 2020 papers below |
| **Celebrating Three Decades of Worldwide Stock Market Manipulation** | 2019 | [arXiv 1912.01708](https://arxiv.org/abs/1912.01708) | ✅ | the figures we reproduce (Fig. 1c, the "25 most problematic", the India Fig. 8) |
| **Strikingly Suspicious Overnight and Intraday Returns** | 2020 | [arXiv 2010.01727](https://arxiv.org/abs/2010.01727) | ✅ | the substantive overnight/intraday data paper |
| **They Still Haven't Told You** | 2022 | [arXiv 2201.00223](https://arxiv.org/abs/2201.00223) | ✅ | the follow-up on the attribution argument |

Author's data/code thread: <https://bruceknuteson.github.io/spy-day-and-night/>

## Supporting / counterpoint literature

| Work | Year | Source | Auto-fetch | Role |
|---|---|---|---|---|
| Lou, Polk, Skouras, *A Tug of War: Overnight vs Intraday Expected Returns* | 2019 | [LSE PDF](https://personal.lse.ac.uk/polk/research/TugOfWar.pdf) · [JFE](https://www.sciencedirect.com/science/article/abs/pii/S0304405X19300650) | ✅ | investor-clientele explanation; the canonical academic treatment |
| Haghani, Ragulin, Dewey, *Night Moves: Is the Overnight Drift the Grandmother of All Market Anomalies?* | 2022 | [Elm Wealth PDF](https://elmwealth.com/night-moves-overnight-drift/) · [SSRN 4139328](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4139328) | ✅ | balanced quant discussion; retail-trading explanation |
| Qiao and Dam, *The Overnight Return Puzzle and the "T+1" Trading Rule in Chinese Stock Markets* | 2020 | [U. Groningen PDF](https://pure.rug.nl/ws/files/132141807/1_s2.0_S1386418120300033_main.pdf) · [JFM](https://www.sciencedirect.com/science/article/abs/pii/S1386418120300033) | ✅ | the Chinese T+1 case that *inverts* the pattern |
| Boyarchenko, Larsen, and Whelan, *The Overnight Drift* | 2023 | [NY Fed Staff Report 917](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr917.pdf) (2020 wp) · [RFS](https://doi.org/10.1093/rfs/hhad020) | ✅ | funding/microstructure mechanism for the drift |
| Cooper, Cliff, Gulen, *Return Differences between Trading and Non-Trading Hours: Like Night and Day* | 2008 | [Purdue copy (Wayback)](https://web.archive.org/web/20210530111718/https://www.krannert.purdue.edu/faculty/hgulen/Day_and_Night.pdf) · [SSRN 1004081](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1004081) | ✅ | early documentation of the night/day split |

> Downloaded PDFs stay local (git-ignored). Copyright remains with the authors.
> Only **Nothing to See Here** (✋) is unobtainable without SSRN access — no free
> copy exists anywhere, and its content is covered by the 2019/2020 Knuteson papers.

---

For formatted, author–date citations of all of the above — plus a **literature
map** of which explanation each paper argues — see
[`../docs/references.md`](../docs/references.md). Machine-readable BibTeX entries
are in [`../references.bib`](../references.bib).
