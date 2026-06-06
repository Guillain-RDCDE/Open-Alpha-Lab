# The source articles (the work this repository responds to)

This project is an **independent, reproducible response** to Bruce Knuteson's
claim that the overnight-vs-intraday return pattern is the fingerprint of
large-scale market manipulation. To judge our response, read his work first.

## Get the papers — one command

```bash
python papers/download_papers.py
```

This downloads the openly available arXiv papers **straight from arXiv** into
this folder. We deliberately do **not** commit the PDFs: they are the authors'
copyrighted work (the arXiv papers are under arXiv's
[`nonexclusive-distrib/1.0`](http://arxiv.org/licenses/nonexclusive-distrib/1.0/)
licence, which grants distribution rights to arXiv, not to third parties). The
right thing — and the more credible thing — is to point you at the primary source.

## The articles

| Work | Year | Source | Role |
|---|---|---|---|
| **Nothing to See Here: How to Say It When You Need to** | 2023 | [SSRN 4619084](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4619084) *(login wall)* | the central pamphlet; its figures are reproduced in the paper below |
| **Celebrating Three Decades of Worldwide Stock Market Manipulation** | 2019 | [arXiv 1912.01708](https://arxiv.org/abs/1912.01708) · [PDF](https://arxiv.org/pdf/1912.01708) | the figures we reproduce (Fig. 1c, the "25 most problematic", the India Fig. 8) |
| **They Still Haven't Told You** | 2022 | [arXiv 2201.00223](https://arxiv.org/abs/2201.00223) · [PDF](https://arxiv.org/pdf/2201.00223) | the follow-up on the attribution argument |

Author's data/code thread: <https://bruceknuteson.github.io/spy-day-and-night/>

## Supporting / counterpoint literature

- Lou, Polk, Skouras (2019), *A Tug of War: Overnight vs Intraday Expected Returns*
- Cooper, Cliff, Gulen (2008), *Return Differences between Trading and Non-Trading Hours*
- Haghani et al. / Elm Wealth (2022), *Night Moves* — balanced quant discussion
- Qiao & Dam (2020) — the Chinese T+1 case that inverts the pattern
- Boyarchenko et al. (Federal Reserve Bank of New York)

> Downloaded PDFs stay local (git-ignored). Copyright remains with the authors.
