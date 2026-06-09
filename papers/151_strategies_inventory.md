# Inventaire trié — *151 Trading Strategies* (Kakushadze & Serur, 2018)

Source : Z. Kakushadze & J. A. Serur, *151 Trading Strategies*, Palgrave Macmillan 2018.
PDF (édition espagnole, formules identiques) : arXiv [1912.04492](https://arxiv.org/abs/1912.04492) · SSRN `3247865`.
**Nature du livre** : catalogue descriptif (+550 formules), **pas** un recueil de backtests. Chaque entrée
est donc une *hypothèse candidate*, à transformer en étude 7-beat — « 151 à tester », pas « 151 testées ».

## Répartition par chapitre (19 classes d'actifs, ~151 stratégies)

| Ch | Classe d'actifs        | # strat. | Tier données dominant |
|----|------------------------|:-------:|-----------------------|
| 2  | Options                |   50    | **C** — chaînes + OI/IV |
| 3  | Actions                |   20    | **A/B** — cœur exploitable |
| 4  | ETFs                   |   6     | **A** |
| 5  | Renta fija / obligataire |  13   | **C** — données bond |
| 6  | Indices                |   3     | A / C (intraday) |
| 7  | Volatilité (asset class) | 5     | **C** — VIX futures, var swaps |
| 8  | Devises (FX)           |   5     | A / B |
| 9  | Commodities            |   6     | B — term structure, COT |
| 10 | Futures                |   4     | **A** |
| 11 | Actifs structurés      |   6     | **C** |
| 12 | Convertibles           |   2     | **C** |
| 13 | Arbitrage fiscal       |   2     | **C** |
| 14 | Divers (météo, énergie…)|  4     | C |
| 15 | Distressed             |   3     | **C** |
| 16 | Immobilier             |   5     | **C** |
| 17 | Cash / monétaire       |   5     | B |
| 18 | Crypto                 |   2     | **A** |
| 19 | Macro global           |   4     | B — données FRED |
| 20 | Infrastructure         |   ~3    | C |

**Lecture clé** : ~50 stratégies (1/3 du livre) sont des montages d'options, hors de portée sans chaîne
fiable avec OI/IV (cf. ton constat sur les sources options). Le gisement réellement exploitable se
concentre dans les chapitres **3, 4, 8, 9, 10, 18, 19** — soit ~40 stratégies, dont ~20 frappables tout de suite.

---

## Tier A — Frappable maintenant (OHLCV daily + ton flux overnight close→open)

| § | Stratégie | Note d'implémentation | Proche de |
|---|-----------|----------------------|-----------|
| 3.1  | Price-momentum | classement cross-section sur rendements passés | fil rouge momentum |
| 3.4  | Anomalie de basse volatilité | tri par vol réalisée, long bas-vol | Study 16 Storm-Shy |
| 3.7  | Momentum résiduel | momentum sur résidus d'un modèle factoriel simple | |
| 3.8  | Trading de paires | cointégration / spread z-score | |
| 3.9  | Réversion à la moyenne (groupe unique) | z-score sur panier | |
| 3.11 | Moyenne mobile | croisement prix/MA | |
| 3.12 | Deux moyennes mobiles | croisement rapide/lent | |
| 3.13 | Trois moyennes mobiles | filtre de tendance | |
| 3.14 | Support & résistance | cassure de niveau | **Study 17 Glass-Ceiling (déjà fait !)** |
| 3.15 | Canal | breakout de canal (Donchian) | Study 17 |
| 3.18 | Arbitrage statistique — optimisation | portefeuille mean-reversion optimisé | |
| 3.20 | Combo alpha | agrégation de signaux | |
| 4.1  | Rotation de momentum sectoriel | top-k secteurs par momentum | |
| 4.4  | Réversion à la moyenne (ETFs) | mean-reversion court terme ETF | |
| 4.6  | Suivi de tendance multi-actifs | trend-following diversifié | |
| 8.1  | Moyennes mobiles avec filtre HP | détrend Hodrick-Prescott puis MA | |
| 10.3 | Contrarian futures (réversion) | mean-reversion sur futures liquides | |
| 10.4 | Suivi de tendance futures (momentum) | le « managed futures » classique | |
| 18.2 | Réseau de neurones (crypto) | OHLCV → ANN directionnel | |
| 18.3 | Sentiment — naïve Bayes (crypto) | OHLCV + flux texte gratuit | |

## Tier B — Frappable avec **une** source publique gratuite en plus

| § | Stratégie | Source d'appoint (gratuite) |
|---|-----------|------------------------------|
| 3.2  | Earnings-momentum | dates de résultats (yfinance / Nasdaq) |
| 3.3  | Value | fondamentaux (yfinance, SimFin) |
| 3.6  | Portefeuille multifactoriel | fondamentaux + factors (Ken French) |
| 3.16 | Événements M&A | flux deals |
| 8.2 / 8.3 | Carry trade (G10 / dollar) | taux courts **FRED** |
| 8.4  | Combo momentum & carry | taux **FRED** + OHLCV |
| 9.1  | Roll yields | structure par terme des futures |
| 9.2  | Pression de couverture | **COT** (CFTC, gratuit) |
| 9.4  | Value commodities | term structure |
| 17.x | Stratégies cash / monétaire | courbe des taux **FRED** |
| 19.2 | Macro-momentum fondamental | séries macro **FRED** |
| 19.3 | Couverture macro vs inflation | CPI / breakevens **FRED** |
| 19.5 | Trading d'annonces économiques | calendrier éco |

## Tier C — Données premium / indisponibles (à écarter pour l'instant)

- **Ch 2 — Options (50 strat.)** : toutes nécessitent chaînes + OI/IV fiables → bloquant (cf. [[options-data-sources]]).
- **Ch 5 — Renta fija (13)** : prix/yields obligataires granulaires.
- **Ch 7 — Volatilité (5)** : VIX futures, variance swaps.
- **Ch 11/12/13/15/16 — structurés, convertibles, arb. fiscal, distressed, immobilier** : données spécialisées.
- **3.19 Market-making, 6.4 arb intraday ETF, 8.5 arb triangulaire** : tick / carnet d'ordres.
- **3.5 Volatilité implicite** : surface d'IV (options).

---

## Top candidats à transformer en études (priorité desk)

1. **3.4 Basse volatilité** — prolonge directement *Study 16 Storm-Shy* (vol-scaling) ; même machinerie, autre angle.
2. **3.15 Canal / Donchian** & **3.14 Support-résistance** — *Study 17 Glass-Ceiling* est déjà une instance ;
   le livre fournit le cadre formel pour une famille « breakout » complète.
3. **10.4 Trend-following futures** — le test « managed futures » canonique, multi-actifs, données faciles.
4. **3.8 Paires / 3.9 réversion** — brique mean-reversion overnight (gap close→open).
5. **8.2 Carry trade** — première incursion FX, une seule source d'appoint (FRED).

> Reste à faire : ~8 stratégies à cheval sur deux lignes de TOC non auto-capturées (parse 143/151) —
> à compléter manuellement à la lecture des chapitres concernés si on attaque ces familles.
