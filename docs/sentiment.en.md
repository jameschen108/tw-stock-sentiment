# Study 1: details of daily sentiment

> English translation, added 2026-10-05, of [sentiment.md](sentiment.md). Where the two differ, the Chinese original governs.

A supplement to study 1 in the [README](../README.md): how texts are selected, how the four sentiment measures are trained and how accurate they are, the LLM labelling prompt, and the details of prediction and backtests.

## Pipeline

```
PTT raw data ──> texts about 2330 ──> score per text ──> daily features ──┬─> 04 analysis (is there an effect?)
prices, ex-rights, market index ──────────────────────────────────────────┴─> 05 prediction (is it usable?)
```

The default target is TSMC (2330); to change stocks, edit `ticker` in `config.yaml` or add `--ticker` to each step.
Prices are the FinMind daily prices in the shared data folder (267 stocks + 0050, 2014 to 2025/03), adjusted on ex-rights/ex-dividend dates with the reference prices in `raw/twse_exrights/`.

## Which texts count as "discussing this stock"

`src/pttsent/ptt.py`; study 3's pre-open discussion volume uses the same texts.
- Posts: the title mentions the ticker or an alias (台積電, 台積, TSMC, 護國神山). A mention only in the body does not count; in practice those are mostly posting templates, margin trading tables or passing references.
- When an alias collides with another company, the words listed under `exclude` in `config.yaml` are removed from the text before matching.
  For example, 長榮 collides with 長榮航 (EVA Air, 2618), so 2603 excludes 長榮航 / 長榮空 / 長榮鋼.
- Comments: the title of their post mentions the stock (then the whole thread counts), or the comment itself mentions it. This catches comments about TSMC in intraday chat threads.
- Quoted lines and the `[標的]` post template are removed first; a comment timestamped before its post, or 30 or more days after it, is treated as a suspicious timestamp and excluded.

Returns beyond the price limit (for example 2603 when it resumed trading after its capital reduction on 2022-09-19) cannot be real; there is no capital-reduction reference price on hand, so they are set to missing.

## Three levels

| Method | How to run | Pros and cons |
|---|---|---|
| Lexicon (default) | Runs as is | Fast and interpretable; cannot tell a description of the past from a prediction, and misses ironic posts (反串) |
| Weak-label classifier | `python scripts/train_classifier.py --labels weak`, then add `--method classifier_weak` to steps 2-5 | Uses the "分類：多／空" (category: long/short) that `[標的]` authors fill in as labels (7,462 posts in 2019-2023; 2024 is kept for the final test and not used in training), at no cost; but the labels are post-level, so they fit short comments poorly, and long labels outnumber short ones 5 to 1 |
| LLM labels + classifier | `scripts/llm_label.py` (see the file header), then `train_classifier.py --labels llm`, and add `--method classifier_llm` to steps 2-5 | The most accurate labels, and they can separate banter and irony; costs API fees (Batches API plus caching, about US$0.002 per text), and the model may "know" what the market did later (the prompt tells it to judge only the text and gives no dates) |

On a time-split test set (2022/04 to the end of 2023), the weak-label classifier reaches macro-F1 0.83 (short-class F1 0.70).
On the same posts, the lexicon can score only 77% of the texts, with 66% directional accuracy.

Applied to **comments**, however, the weak-label classifier calls 93% of them bullish, and its correlation with same-day returns falls to 0.03 (the lexicon's is 0.11).
The reason is that it never learned "neutral / irrelevant" and never saw short comments. Measuring comment sentiment therefore still needs LLM labels.

## LLM labelling

`src/pttsent/sentiment/llm_label.py` uses Claude Opus 5.5 (effort medium) to label each text bullish / bearish / neutral / irrelevant,
and also flags `sarcasm` and `banter`. The main points of the prompt:
- Label a direction only when the *author's own* view of where the price is going is visible; describing or complaining about moves that already happened, mocking other people or current events, memes, and showing off gains or losses all count as irrelevant.
- Neutral is limited to serious discussion of prospects without a direction; a one-line judgment about fundamentals, demand or competition also counts as neutral rather than irrelevant.
- For posts, the author's own commentary (心得) section decides; quoted news text is not taken as the author's view.

The prompt was spot-checked and revised version by version on 50 texts: the first version forced banter into long/short or dumped it into neutral; after raising effort from low to medium, agreement across 3 reruns on ambiguous cases rose from 3/9 to 6/9.
Sampling excludes 2024 (the final test period) and is proportional by year and by post/comment.

| | 2330 TSMC | 2603 Evergreen |
|---|---|---|
| Labels (2019-2023) | 4,000 | 1,902 |
| API cost | US$8.28 | US$3.16 |
| Irrelevant / bullish / neutral / bearish | 62% / 18% / 12% / 8% | 64% / 17% / 8% / 11% |
| Share of banter | 55% | 56% |
| Classifier F1 (bullish / bearish / neutral / irrelevant) | 0.39 / 0.25 / 0.42 / 0.80 | 0.49 / 0.26 / 0.27 / 0.77 |

The small classifier (character n-gram TF-IDF + logistic regression) can pick out banter but separates bullish from bearish poorly (bearish F1 about 0.25):
irony and meaning cannot be learned from surface characters, and bearish examples are few.

## Pooling labels from two stocks

`train_classifier.py --labels llm_pooled` pools the LLM labels of both stocks (5,902 texts) to train one shared classifier (`--method classifier_llm_pooled`).
With the same time split and the same test texts, compared with each stock's own labels only: 2330 macro-F1 0.464 → 0.509 and bearish 0.245 → 0.379 (bootstrap 95% CI excludes 0);
2603 0.441 → 0.455, within noise. But 2330's daily sentiment correlates 0.87 with the original (averaging about 60 accounts a day cancels out most single-text errors),
so downstream results barely changed.

2317 got another 500 labels, used only as a test set and not for training (`llm.pool_tickers` in `config.yaml` decides which stocks are pooled).
The pooled classifier reaches macro-F1 0.440 on them (comments 0.421), similar to 2603. This batch cost US$3.50 (about US$0.007 per text):
in a small batch the 1-hour cache almost never hit, so every text paid for a cache write (2× the input price).

## BERT

`train_classifier.py --labels llm_pooled --model bert`, and `--method classifier_bert` for steps 2-5.
It needs torch and transformers; training takes about 30 minutes on an M2, and scoring runs at about 300 texts a second. It fine-tunes `ckiplab/bert-base-chinese`;
a comment gets its post's title as the first segment, a long post keeps its first 64 tokens plus its end (256 at most), and the epoch is chosen on the last 10% of the training set.
Against the pooled TF-IDF on the same test set as above (1,181 texts):

| | macro-F1 | Bearish | Bullish | Neutral | Irrelevant | Comment macro-F1 | Long/short flipped |
|---|---|---|---|---|---|---|---|
| TF-IDF (pooled) | 0.495 | 0.351 | 0.473 | 0.369 | 0.786 | 0.474 | 35 |
| BERT | **0.550** | 0.388 | **0.574** | 0.413 | **0.827** | **0.517** | 30 |

The difference is +0.056 (bootstrap 95% CI [+0.017, +0.096]), with gains on both 2330 and 2603. The first version kept only the first 128 tokens and did not win (0.472),
because the author's commentary section got cut off. The change in the second version was decided after seeing the first version's test-set results, so the gain may be a little optimistic.
It was therefore also checked on 2317 (500 texts), which took no part in any training or tuning:

| 2317 | macro-F1 | Bearish | Bullish | Neutral | Irrelevant | Comment macro-F1 | Long/short flipped |
|---|---|---|---|---|---|---|---|
| TF-IDF (pooled) | 0.440 | 0.324 | 0.473 | 0.159 | 0.803 | 0.421 | 12 / 119 |
| BERT | **0.519** | 0.369 | 0.576 | 0.362 | 0.769 | 0.456 | 19 / 119 |

The difference is +0.079 (95% CI [+0.024, +0.139]), so the gain holds. But BERT commits to a direction more readily (bearish recall 0.28 → 0.44, precision about 0.31),
and it flips long and short more often. The score is P(bullish) − P(bearish), so flips do the most damage, and daily sentiment does not necessarily get more accurate.

## Model files

Trained models are in `data/models/`: the weak-label model is `sentiment_clf_weak.joblib`, the LLM-label models are one per stock as `sentiment_clf_llm_{ticker}.joblib`,
the pooled one is `sentiment_clf_llm_pooled.joblib`, and BERT is the folder `sentiment_bert_llm_pooled/` (training log in its `training.json`).
A `.json` file with the same name next to each model records the label source, training period, test scores and scikit-learn version. Loading a model under a different version gives a warning; retrain in that case.

## Daily features, prediction and backtests

- Besides mean sentiment, the daily features include the bullish share, sentiment dispersion (standard deviation), attention (post count, abnormal post count) and the push/boo ratio; days with no sentiment signal are flagged with `has_sent` and imputed.
- The prediction target is whether tomorrow is up (`up_next`), with two baselines: always guess up, and guess up tomorrow if today was up. The logit C, and the reason gradient-boosted trees were dropped, are in `models.py`.
  The gbm result in the supplementary line of [PREREGISTRATION.en.md](prereg/PREREGISTRATION.en.md) was run before they were dropped.
- Backtest: buy or keep holding at the close when the model predicts up, otherwise stay flat; fees are in `config.yaml`. For `open_to_close` there are also short day-trade strategies,
  including "short the day after a low-sentiment day" (the threshold is computed on a rolling basis from past data only; see `backtest.py`).
- In the development period every backtest loses to buy-and-hold after costs; buy-and-hold returned 2330 +87%, 2603 +1057% (the 2020-2021 shipping rally), 2317 +39%.

## More on the results

- Evergreen's LLM TF-IDF sentiment at first seemed not to follow prices (return → sentiment p 0.50-0.68). At the time I thought the prompt's exclusion of "describing today's move" explained it;
  but with BERT p = 0.001, so the more likely cause is that the old classifier measured only about 8 accounts a day, which is too coarse.
- The only hint (2603 LLM oc): in the two lowest sentiment quintiles, the next day's mean intraday return is −47 and −96 bp; the other quintiles are about 0.
- Hon Hai: Granger p for return → sentiment is 0.022, and for sentiment → return 0.47. The median number of accounts with a sentiment signal per day is only 4 (8 for 2603), so the measurement is coarser and the same-day correlation lower.
- BERT rerun on all three stocks: accounts with a sentiment signal per day went 2330 59 → 97, 2603 8 → 15, 2317 4 → 10; the new daily sentiment correlates only 0.38-0.52 with the earlier methods,
  so it really does measure something different, but the next-day correlation is only 0.001-0.029, none of it significant.

## Predicting returns and long-short

`scripts/13_return_predict.py`; results are in `output/{ticker}/classifier_bert/return_{dev|2024}{_oc}/`.
The README lists only group B's main numbers; these are the full tables.

- The evaluation period is the set of trading days on which all four models (A/B × regression/logit) have forecasts; in the development period it starts in 2020-01 (the first 250 days are training only).
  It is not exactly the same as the evaluation period of `05_predict.py`, so the buy-and-hold numbers differ slightly (for example 2603 +1287% vs +1057%).
- The regression training target is winsorized at the 1% / 99% quantiles within the training window. Out-of-sample R² uses the historical mean as its benchmark (Campbell-Thompson); a version against 0 is also saved in `metrics_reg.csv`.
- The ridge penalty candidates are 10⁰ to 10⁶. For 2330 every re-estimation picks 10⁶; for 2603 and 2317 it is mostly 10³ or 10⁶, so the forecasts are almost constant.
- Cost of flipping from long to short: sell the long, then short through securities borrowing, paying two sell-side costs plus the borrowing fee; flipping from short to long pays two buy-side costs. A short's daily return is approximated as −(close at t+1 / close at t − 1), ignoring interest on short-sale margin and forced buy-ins.
- Futures are approximated with stock returns, ignoring basis and rolls; commissions are per contract and converted to 0.02% in `config.yaml`; adjust this for your broker and the stock price.
- Thresholds: close to close uses half the round-trip cost (stocks 0.29%, futures 0.02%), keeping the current position when the forecast does not clear it; day trades use the full round-trip cost (stocks 0.44%, futures 0.04%), with no trade when it is not cleared.
- Break-even cost = total gross profit ÷ total traded value, a simple-interest approximation; a volatile stock can still lose money under compounding even when it clears the cost.

### Predictions

| | R² ridge (A → B) | R² OLS (A → B) | IC OLS (A → B) | CW t ridge / OLS | AUC logit (A → B) | AUC diff |
|---|---|---|---|---|---|---|
| **Development** | | | | | | |
| 2330 cc | −0.11% → −0.15% | −3.03% → −4.24% | −0.033 → −0.048 | −1.27 / −0.27 | 0.508 → 0.494 | −0.014 [−0.036, +0.002] |
| 2330 oc | −0.14% → −0.25% | −2.26% → −3.82% | +0.017 → −0.007 | −0.34 / −0.38 | 0.494 → 0.488 | −0.006 [−0.028, +0.008] |
| 2603 cc | −0.20% → −0.28% | −3.02% → −3.74% | +0.039 → +0.038 | +0.33 / −0.47 | 0.519 → 0.510 | −0.008 [−0.019, +0.002] |
| 2603 oc | +0.13% → +0.28% | −1.98% → −2.93% | +0.058 → +0.043 | +1.46 / −1.03 | 0.524 → 0.522 | −0.002 [−0.014, +0.009] |
| 2317 cc | −0.07% → −0.17% | −1.14% → −2.56% | +0.063 → +0.046 | −0.37 / −0.96 | 0.523 → 0.529 | +0.007 [−0.024, +0.033] |
| 2317 oc | +0.46% → +0.34% | −0.20% → −1.61% | +0.076 → +0.056 | −0.09 / −1.35 | 0.540 → 0.534 | −0.006 [−0.037, +0.016] |
| **2024** | | | | | | |
| 2330 cc | −0.02% → −0.02% | −0.82% → −0.84% | −0.102 → −0.088 | +1.46 / +0.45 | 0.485 → 0.511 | +0.027 [−0.021, +0.061] |
| 2330 oc | −0.59% → −0.61% | −1.21% → −2.38% | +0.046 → +0.003 | −0.02 / −0.91 | 0.503 → 0.525 | +0.021 [−0.003, +0.057] |
| 2603 cc | −0.34% → −0.02% | −3.43% → −3.29% | −0.118 → −0.081 | +1.30 / +0.74 | 0.491 → 0.476 | −0.015 [−0.062, +0.008] |
| 2603 oc | −0.22% → +0.03% | −0.86% → −0.34% | +0.070 → +0.110 | +1.02 / +1.04 | 0.529 → 0.512 | −0.017 [−0.041, +0.001] |
| 2317 cc | −0.60% → −0.08% | −0.75% → −1.86% | +0.032 → +0.011 | +1.41 / −2.10 | 0.480 → 0.485 | +0.005 [−0.025, +0.035] |
| 2317 oc | −1.09% → −0.95% | −0.79% → −0.83% | +0.034 → +0.032 | +0.59 / +0.19 | 0.509 → 0.510 | +0.001 [−0.015, +0.021] |

### Long-short backtests (group B, total return after costs)

The benchmark is buy-and-hold for cc and shorting every day for oc. Parentheses give the share of days spent long in the long-short position.

| | Stocks: benchmark | OLS long-short | logit long-short | OLS threshold | Futures: benchmark | OLS long-short | logit long-short |
|---|---|---|---|---|---|---|---|
| **Development** | | | | | | | |
| 2330 cc | +87% | −92% (71%) | −83% (74%) | −43% | +87% | −51% | −15% |
| 2330 oc | −98% | −99% (38%) | −98% (19%) | −18% | +5% | −48% | +6% |
| 2603 cc | +1287% | −33% (64%) | −56% (80%) | +25% | +1289% | +302% | +16% |
| 2603 oc | −95% | −99% (35%) | −99% (29%) | −57% | +116% | −61% | −58% |
| 2317 cc | +39% | −94% (60%) | −90% (72%) | +12% | +40% | −26% | −22% |
| 2317 oc | −97% | −98% (34%) | −98% (17%) | −16% | +23% | −24% | −19% |
| **2024** | | | | | | | |
| 2330 cc | +84% | −10% (79%) | −14% (67%) | +41% | +84% | +29% | +47% |
| 2330 oc | −69% | −70% (31%) | −65% (19%) | −3% | −21% | −23% | −11% |
| 2603 cc | +57% | −58% (79%) | +3% (92%) | +23% | +57% | −31% | +24% |
| 2603 oc | −55% | −48% (42%) | −59% (13%) | +5% | +14% | +35% | +6% |
| 2317 cc | +79% | −45% (62%) | −57% (64%) | +47% | +80% | +6% | −24% |
| 2317 oc | −72% | −72% (36%) | −58% (19%) | −11% | −27% | −28% | +9% |

- With flat days left out of logit training, the close-to-close model guesses up on 60-90% of days in most cases; intraday it guesses down on 70-90%, consistent with the negative average intraday return.
- Random benchmark (`random_sign.csv`): of the 36 long-short combinations in the development period, only group A logit for 2317 oc beats 95% of random positions (96%); in 2024 it is again the only one (97%).
