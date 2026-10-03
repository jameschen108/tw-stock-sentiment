# PTT Sentiment and Taiwan Stocks

[中文版](README.zh-TW.md)

My individual part of a five-person undergraduate capstone at Soochow University. The team studies Taiwan stocks with different methods, and the advisor asked us to work individually first.

The project tests whether the Stock board on PTT, Taiwan's largest bulletin board system, carries information beyond price and volume.
The question I care about is how much a price-and-volume model improves once PTT is added.
I explored each of the three studies on a development period first, then preregistered its hypotheses and tested them once on a period I had not looked at.

## Results

| Study | Question | Answer | Preregistration |
|---|---|---|---|
| 1. Daily sentiment | Does it predict next-day returns? | **No.** Sentiment reacts to prices that have already moved; the only hint in the development period did not recur in 2024; predicting returns and going long-short on the sign fails as well | [PREREGISTRATION.md](docs/prereg/PREREGISTRATION.md) |
| 2. [標的] posts | Do the long/short calls authors declare predict later returns? | **No.** Authors chase rallies; the small gap comes from continuation after limit-locked closes, and buying on bullish posts loses money after costs | [PREREGISTRATION_TARGET.md](docs/prereg/PREREGISTRATION_TARGET.md) |
| 3. Pre-open discussion volume | Does discussion between the close and the next open predict that day's trading volume? | **Yes**, but the effect is small (out-of-sample MSE down 0.5-3%); passed both out-of-sample tests, 2024 and 2025-2026 | [PREREGISTRATION_VOLUME.md](docs/prereg/PREREGISTRATION_VOLUME.md), [replication](docs/prereg/PREREGISTRATION_VOLUME_REPLICATION.md) |

PTT reflects prices that have already moved, but it does signal whether the day will be busier than usual.
Details for study 1 are in [docs/sentiment.md](docs/sentiment.md), and for studies 2 and 3 in their [preregistration documents](docs/prereg/). Everything under `docs/` is in Chinese.

## Terms

- **PTT**: Taiwan's largest bulletin board system. The Stock board is its stock discussion board; comments under a post are called pushes (推文).
- **[標的] posts**: the Stock board's format for single-stock pitches. Board rules require the author to state long or short.
- **Price limit (漲跌停)**: a Taiwan stock can move at most ±10% a day. When the close is locked at the limit, orders usually cannot be filled.
- **Disposition stocks (處置股)**: stocks the exchange puts under trading restrictions after abnormal trading.
- **Day trade (當沖)**: buying and selling the same stock on the same day. A sell-first day trade needs no securities borrowing.
- **Tickers**: 2330 TSMC, 2603 Evergreen Marine, 2317 Hon Hai (Foxconn), 0050 Yuanta Taiwan 50 ETF.

## Data

Raw data is not in this repo. The code reads it from a local folder, `/Users/jameschen/Project/data` (read-only; set the environment variable `PTT_DATA_DIR` to point elsewhere):

| Use | Path | Notes |
|---|---|---|
| PTT posts and comments | `pttcc/stock_{year}.jsonl` | 2019-2024: 158k posts, 15.63M comments; 2025 to 2026/09/26: 45k posts, 5.56M comments (used only in the study 3 replication) |
| PTT [標的] posts | pttweb | 2015/04 to 2025/01, including deleted posts (study 2) |
| Daily stock prices, market index | `raw/finmind/price/`, `raw/twse_taiex/` | 267 stocks + 0050, 2014 to 2025/03 (study 1) |
| Ex-rights/ex-dividend, disposition stocks | `raw/twse_exrights/`, `raw/twse_punish/` | Return adjustment; controls in study 3 |
| cnyes news, TAIEX futures | `raw/cnyes/headline/`, `raw/taifex_tx/` | Controls in study 3; exploratory analysis in study 2 |

Studies 2 and 3 also use a full-market daily price panel I downloaded myself (TWSE + TPEx, 2015 to 2026/09, including delisted stocks and adjusted for capital reductions) and US data (TSM, ^SOX), stored in `data/prices/`.
Against FinMind's 267 stocks, closing prices match on all 610k stock-days. Returns differ in only 38 cases, all ex-dividend adjustments during a stock's TPEx-listed period that the original FinMind pipeline missed.

## Running

```bash
pip install -r requirements.txt
python -m pytest                          # time alignment, no look-ahead, cost calculations

# Study 1: daily sentiment (default 2330)
python scripts/01_prepare_ptt.py          # PTT to parquet + extract texts about the target stock
python scripts/02_score_sentiment.py      # sentiment score for each text
python scripts/03_build_features.py       # daily features + prediction targets
python scripts/04_analyze.py              # correlation, Granger, regression, event study
python scripts/05_predict.py              # rolling prediction + backtest
python scripts/06_prereg_test.py --final  # preregistered 2024 test (run once already; do not rerun)
python scripts/13_return_predict.py       # predict return and direction, long-short on the sign; --period 2024 is supplementary (run already)
python scripts/14_explore_directions.py   # exploration: weekly, extreme sentiment, excess return (development period only)

# Study 2: [標的] event study
python scripts/fetch_prices.py --market twse   # full-market daily quotes, about 3.5 hours; add --end 2026-09-24 for the study 3 replication
python scripts/fetch_prices.py --market tpex   # can run in parallel
python scripts/build_panel.py                  # full-market panel + validation
python scripts/07_target_events.py             # [標的] posts -> event table
python scripts/08_target_dev.py                # development period 2016–2023
python scripts/09_target_explore.py            # exploratory analysis (FDR-corrected)
python scripts/10_target_prereg.py --final     # 2024 final test (run once already; do not rerun)

# Study 3: pre-open discussion volume and trading volume
python scripts/fetch_us.py                                  # TSM, ^SOX, through 2025-06
python scripts/fetch_us.py --end 2026-09-27 --out us_2026   # for the replication, saved separately without overwriting
python scripts/11_volume_prereg.py              # development period 2019–2023
python scripts/11_volume_prereg.py --final      # 2024 final test (run once already; do not rerun)
python scripts/12_volume_replication.py --final # 2025-01 .. 2026-09 replication (run once already; do not rerun)
```

- Steps 2-5 and 13 accept `--ticker` and `--method` (`lexicon` / `classifier_weak` / `classifier_llm` / `classifier_llm_pooled` / `classifier_bert`).
  Steps 3-5 and 13 accept `--target open_to_close`, which predicts next-day open-to-close instead and writes results to `*_oc`. Classifier training is described in [docs/sentiment.md](docs/sentiment.md).
- LLM labelling needs `ANTHROPIC_API_KEY=...` in `.env` at the project root (gitignored).
- Results go to `output/{ticker}/{method}/` and intermediate files to `data/`; neither is in git.

## Shared design

- Which texts count as discussing a stock (`ptt.py`): posts whose title mentions the ticker or an alias, with their whole comment thread, plus comments that mention it themselves.
  A mention only in the body does not count, since those are mostly templates or passing references. When an alias collides with another company, `exclude` in `config.yaml` removes those words first (for Evergreen Marine, 長榮航, EVA Air).
- Time alignment (`calendar.py`): a post before the 13:30 close counts toward that day; later posts, weekends and holidays count toward the next trading day. Features for day t use only posts visible before day t's close. With `--target open_to_close` the day is cut at 09:00 instead, so features run up to the open of t+1.
- Returns (`prices.py`): adjusted with reference prices on ex-rights/ex-dividend dates. Returns beyond the price limit that cannot be adjusted are set to missing.
- Validation: development and final test periods are kept separate. I commit the hypotheses, thresholds and test code first, run the final test once, and record the result whichever way it goes.
  2024 and 2025-01 to 2026-09 have both been used up, so new ideas have to be tested on data from 2026-10 onward.

## Study 1: daily sentiment and next-day returns

- Daily sentiment (`features.py`): averaged within each account for the day first, then across accounts, so that spamming accounts do not dominate. A day with no sentiment signal is missing, not 0.
- Sentiment measures: a lexicon; a weak-label classifier, whose labels are the long/short that [標的] authors declare; a TF-IDF classifier trained on LLM labels (Claude Opus 5.5); and BERT fine-tuned on the same labels.
  BERT reaches macro-F1 0.550 on the test set against 0.495 for TF-IDF, and 0.519 against 0.440 on 2317, which was not used in training. Details in [docs/sentiment.md](docs/sentiment.md).
- Prediction (`models.py`): predict whether tomorrow is up. Group A uses price and volume only; group B adds sentiment. Logistic regression (C=0.01), retrained every 21 trading days on past data only.
- Backtest (`backtest.py`): hold when the model predicts up, net of commissions and securities transaction tax. For open-to-close, every day is a day trade.

Development period 2019-2023; cc = close to close, oc = next-day open to close. AUC diff is the out-of-sample AUC of B − A with a bootstrap 95% CI. IC is the Spearman correlation between the predicted probability and the next-day return.

| Stock × method × target | corr(sentiment, same day) | corr(sentiment, target) | Regression p (sentiment) | Granger p, return to sentiment | AUC diff | IC (A → B) |
|---|---|---|---|---|---|---|
| 2330 lexicon cc | 0.112 | −0.012 | 0.61 | 0.003 | −0.008 [−0.021, +0.004] | +0.006 → −0.003 |
| 2330 weak labels cc | 0.024 | −0.021 | 0.23 | — | −0.010 [−0.027, +0.006] | +0.006 → −0.002 |
| 2330 LLM cc | 0.132 | −0.024 | 0.19 | 0.0001 | +0.001 [−0.014, +0.014] | +0.006 → +0.008 |
| 2330 LLM pooled cc | 0.124 | −0.025 | 0.17 | 0.0004 | −0.001 [−0.019, +0.015] | +0.006 → +0.004 |
| 2330 BERT cc | **0.194** | 0.001 | 0.88 | <0.0001 | −0.014 [−0.034, +0.006] | +0.006 → −0.013 |
| 2330 lexicon oc | — | 0.017 | 0.28 | — | −0.006 [−0.023, +0.006] | +0.019 → +0.000 |
| 2330 LLM oc | — | −0.001 | 0.65 | — | −0.005 [−0.024, +0.008] | +0.019 → −0.003 |
| 2603 lexicon cc | 0.122 | 0.021 | 0.47 | <0.0001 | −0.007 [−0.023, +0.005] | +0.047 → +0.033 |
| 2603 LLM cc | 0.082 | 0.036 | 0.066 | 0.50 | +0.002 [−0.010, +0.017] | +0.047 → +0.051 |
| 2603 lexicon oc | — | 0.045 | 0.047 | 0.30 | +0.003 [−0.012, +0.017] | +0.013 → +0.014 |
| **2603 LLM oc** | — | **0.064** | **0.028** | 0.68 | +0.003 [−0.007, +0.014] | +0.013 → +0.016 |
| 2603 BERT cc | 0.108 | 0.029 | 0.39 | 0.001 | −0.007 [−0.017, +0.001] | +0.047 → +0.037 |
| 2317 LLM pooled cc | 0.075 | −0.025 | 0.50 | 0.022 | −0.008 [−0.029, +0.011] | +0.028 → +0.026 |
| 2317 BERT cc | 0.127 | 0.024 | 0.21 | <0.0001 | +0.000 [−0.022, +0.021] | +0.028 → +0.030 |

1. Sentiment reacts to prices. Its correlation with the same-day return is 0.075-0.19, and the Granger test from return to sentiment is mostly significant. (The weak labels call 90% of comments bullish and measure almost nothing.)
2. Better measurement raises the same-day correlation, but the next day still shows nothing. For TSMC the same-day correlation rises from 0.112 with the lexicon to 0.194 with BERT, and BERT picks up about twice as many accounts per day, yet the next-day correlation stays near 0.
3. Adding sentiment does not significantly raise AUC in any combination, and every backtest loses to buy-and-hold after costs.
   I chose Hon Hai as a new sample before looking at the results, and it agrees.
4. The only hint is 2603 LLM oc. After low-sentiment days the next day's intraday return leans negative, and short day trades earn +7% after costs, but I picked this out of about 10 combinations.

2024 final test:

| Hypothesis | Development | 2024 | Verdict |
|---|---|---|---|
| H1 sentiment regression coefficient > 0 | +0.0084, p = 0.014 | −0.0048, p = 0.69 | Failed |
| H2 lower next-day intraday return after low-sentiment days | −59.6 vs −1.4 bp, p = 0.003 | −4.8 vs −21.7 bp (reversed), p = 0.73 | Failed |
| H3 short day trades after low-sentiment days profitable after costs | +7.3% | −36.0% | Failed |

### Predicting returns and long-short

The models above only predict direction and only go long. Here they predict both the return and the direction, and take a side on the sign: buy at the close when the forecast is > 0, short when it is < 0 (`13_return_predict.py`).

- Models: same features and rolling training as above. Returns come from ridge, with the penalty chosen by time-series cross-validation inside the training window, and from unshrunk OLS. Direction comes from logit, with flat days left out of training.
  Ridge picks the maximum penalty almost every time, so its forecast collapses to the historical mean. That mean is always > 0, which turns the long-short into buy-and-hold. I added OLS after seeing this, so that the sign can change.
- Costs, two cases. Stocks: shorting close to close requires securities borrowing, which adds a 0.08% borrowing fee; a sell-first day trade does not. Single-stock futures: commission of about 0.02% plus futures transaction tax of 0.002%.
- Additional numbers:
  - Break-even cost: gross profit ÷ traded value.
  - Random benchmark: shuffle the positions while keeping the number of long and short days fixed, and count what share of the random positions the actual gross return beats.
  - Threshold variant: trade only when the forecast return exceeds the cost, with the threshold fixed in advance.

Development period 2019-2023, BERT sentiment, group B. The benchmark for cc is buy-and-hold. For oc it is shorting every day, because the average intraday return is negative for all three stocks.

| | Out-of-sample R² (OLS) | CW t (OLS) | AUC diff | Stocks: benchmark / OLS long-short / logit long-short | Futures: benchmark / OLS long-short / logit long-short | Break-even vs actual cost (logit) |
|---|---|---|---|---|---|---|
| 2330 cc | −4.2% | −0.27 | −0.014 [−0.036, +0.002] | +87% / −92% / −83% | +87% / −51% / −15% | 0.02% vs 0.31% |
| 2330 oc | −3.8% | −0.38 | −0.006 [−0.028, +0.008] | −98% / −99% / −98% | +5% / −48% / +6% | 0.03% vs 0.22% |
| 2603 cc | −3.7% | −0.47 | −0.008 [−0.019, +0.002] | +1287% / −33% / −56% | +1289% / +302% / +16% | 0.25% vs 0.31% |
| 2603 oc | −2.9% | −1.03 | −0.002 [−0.014, +0.009] | −95% / −99% / −99% | +116% / −61% / −58% | −0.00% vs 0.22% |
| 2317 cc | −2.6% | −0.96 | +0.007 [−0.024, +0.033] | +39% / −94% / −90% | +40% / −26% / −22% | 0.00% vs 0.31% |
| 2317 oc | −1.6% | −1.35 | −0.006 [−0.037, +0.016] | −97% / −98% / −98% | +23% / −24% / −19% | 0.02% vs 0.22% |

1. Returns cannot be predicted. The out-of-sample R² of OLS is negative everywhere, worse than simply using the historical mean; ridge shrinks to the historical mean, with R² within ±0.5%.
   With sentiment added, the highest Clark-West t is 1.46 (ridge), and every AUC-difference CI contains 0.
2. With stock costs, long-short on the sign loses heavily. All 12 combinations (6 groups × OLS, logit) lose 33-99%, and every close-to-close one loses to buy-and-hold.
   Gross profit per dollar traded is mostly under 0.05%, while the actual cost is 0.22-0.31%.
   The only one above cost is 2603 cc OLS (0.35% vs 0.31%), and Evergreen is volatile enough that it still loses 33% compounded.
3. Futures-level costs do not rescue it. Group B in the table loses to buy-and-hold in every close-to-close case; intraday, only 2330 logit narrowly beats shorting every day (+6% vs +5%).
4. Of 36 long-short combinations, including group A and ridge, only 1 has a gross return beating 95% of random positions with the same long/short split. At the 5% level about 2 would be expected anyway.
   That one is the price-and-volume-only logit for 2317 oc, so PTT plays no part in it (see point 6 of [Next steps](#next-steps)).
5. With stock costs, ridge forecasts almost never clear the threshold. OLS trades much less under the threshold, but still loses to buy-and-hold close to close.

2024 was already used by study 1, so this part is not preregistered; I committed the code before running 2024. The conclusion is the same:
- With sentiment added, the highest CW t is 1.46, and every AUC-difference CI contains 0.
- 11 of the 12 stock long-short combinations lose money. The only profitable one is 2603 cc logit (+3%), while buy-and-hold returns +57%.

Full tables in [docs/sentiment.md](docs/sentiment.md#預測報酬與多空).

## Study 2: [標的] posts as stock recommendations

Board rules require [標的] authors to state long or short themselves, so these posts work like stock recommendations. Each post is one event, entered at the first close after posting, with returns measured relative to the market.
Standard errors are two-way clustered by date and stock. I hand-checked the event parsing on 200 posts; 198 were correct.

Development period 2016-2023: 10,705 posts (8,709 long, 1,996 short) on 1,467 stocks.

| Long − short (market-adjusted) | Difference | t |
|---|---|---|
| 5 days after posting (H1) | +0.41% | 1.99 |
| 20 days after posting (H2) | +0.54% | 1.44 |
| Same, excluding the 4.7% whose entry close is locked at the price limit | +0.09% / +0.28% | 0.47 / 0.77 |
| 5 days before posting | +0.69% | 2.81 |

1. The small gap comes from price limits. Long posts whose entry close is locked limit-up gain +2.4% over the next 5 days, and short posts locked limit-down lose −6.5%, but a locked close usually cannot be traded. Excluding them, the gap is close to 0.
2. Authors chase rallies: stocks in both long and short posts had already risen 4-5% in the 20 days before posting.
3. Following them loses money. Buying only the long posts returns −0.83% over 5 days and −1.51% over 20 days after costs, and trails stocks with similar liquidity and past returns by only 0.11%.
4. Exploratory analysis finds nothing. None of 14 features is significant after FDR correction; they include the author's past accuracy, push/boo ratio, comment sentiment, an attached brokerage statement, deleted posts, and whether there was news.
   For the 656 market-index posts, the direction is about as good as random. Of the 1,509 posts that give a target price and a stop loss, about 50% hit the stop first within 60 days and about 31% hit the target first.

Final test, 2024/01 to 2025/01: 596 posts, only 57 of them short. I knew in advance that power was only about 5%.

| Hypothesis | Development | Final test | Verdict |
|---|---|---|---|
| H1 5-day long − short > 0 | +0.41%, one-sided p = 0.023 | −1.32% (reversed), p = 0.89 | Failed |
| H2 20-day long − short > 0 | +0.54%, p = 0.075 | +1.44%, p = 0.24 | Failed |
| H3 a difference in the 20 days before posting | −1.05%, two-sided p = 0.048 | −2.46%, p = 0.27 | No significant difference |

## Study 3: pre-open discussion volume and same-day trading volume

Post counts with the day cut at 13:30 mostly react to that day's intraday price and volume, and add nothing for the next day's volume or volatility (see point 1 of [Next steps](#next-steps)).
Discussion between the close and the next open happens after the last trade, so the price-and-volume model has not seen it yet.

- Target: log trading volume on day T (main) and log intraday range. The forecast is made before T's open.
- Signal: posts plus comments from T−1 13:30 to T 09:00, log1p, minus the mean of the same window over the previous 20 trading days (`volume.py`).
- Baseline: recent price and volume, calendar and settlement days, the TAIEX futures night session, overnight moves in TSM and the Philadelphia Semiconductor Index, cnyes news volume, a limit-up or limit-down close the day before, disposition periods, and ex-dividend days.
- Model: expanding-window OLS, re-estimated every 21 trading days. The Clark-West test compares models with and without discussion volume.

In the development period, with only the night session as a control, discussion volume cuts the MSE of TSMC's trading volume by 3.96%. After adding the ADR and SOX the cut is 0.96%, so about 3/4 of it was relaying the ADR's overnight move.

| CW t (TSMC / Evergreen / Hon Hai) | Development 2019-2023 | 2024 final test | Replication, 2025-01 to 2026-09 |
|---|---|---|---|
| **H1** trading volume, three stocks pooled | 5.48 | 3.76, p = 0.0001, **passed** | 2.94, p = 0.002, **passed** |
| H2 trading volume per stock (Holm) | 2.40 / 3.92 / 3.33 | 1.67 / 2.63 / 1.65, only Evergreen passed | 2.22 / 2.49 / 1.68, all three passed |
| H3 intraday range, three stocks pooled | 3.18 | 2.92, **passed** | 2.47, **passed** |
| H1 with the opening gap added | 4.56 | 3.19 | 2.78 |
| Out-of-sample MSE change | −0.69 / −1.90 / −1.08% | −1.26 / −2.96 / −1.10% | −1.06 / −1.99 / −0.48% |
| Volume change per +1 SD of discussion | +4.5 / +8.4 / +5.8% | +3.6 / +10.5 / +4.4% | +4.0 / +10.0 / +3.6% |

- The result stays significant after adding the opening gap, which sums up the market's reaction to all overnight news. So the signal carries more than the size of the overnight news.
- It predicts how busy the day will be and says nothing about direction, so the conclusions of studies 1 and 2 stand.
- Monthly discussion of Evergreen since 2025 is only about 1/8 of what it was before (about 380k comments in 2019-2024, 13k in 2025-01 to 2026-09), yet its increment is still the largest of the three.
- The replication covers 420 trading days; 395 of them have trading volume for all three stocks.

## Next steps

1. Change the target. My first attempt used post counts cut at 13:30 to predict the next day's |return|, and it added nothing beyond past volatility
   (out-of-sample MSE change 2330 −0.2%, 2603 +0.1%, 2317 +0.4%, all CIs contain 0; the in-sample t = 3.2 for 2603 comes from heavy posting and high volatility coinciding in the 2021 shipping boom).
   Switching to discussion between the close and the next open is what worked (study 3). 5-day returns or returns relative to the market are also worth trying.
2. Add controls. The sentiment models do not yet include news volume or the night session, and institutional investor flows are not used anywhere yet. Study 3 does not yet control for freight rates (Evergreen), Apple and Nvidia (Hon Hai), or earnings call and monthly revenue release dates.
3. Trade less often. I have tried weekly frequency, trading only on extreme sentiment, and predicting returns relative to the market; none of the 12 pre-specified tests passed FDR correction
   ([docs/explore_directions.md](docs/explore_directions.md)).
4. Cover more stocks. Of 301 stocks scanned, only 2330, 2603 and 2317 had a median of ≥ 20 comments per trading day in 2019-2023.
   UMC, Yang Ming, China Steel and AUO had enough only around 2021, almost all others had ≤ 5, and short names such as 統一, 南亞, 大成 and 華電 mostly collide with other words.
   A daily panel is not feasible. It would need weekly frequency, or a cross-stock event study on days when discussion spikes.
5. Test again on new out-of-sample data from 2026-10 onward.
6. Check an intraday price-and-volume signal for 2317. A logit using only price and volume to predict Hon Hai's next-day intraday direction beats 96% of random positions in gross return in the development period and 97% in 2024.
   With futures costs, its long-short returns +64% and +16%, against +23% and −27% for shorting every day over the same periods. I picked it out of 36 combinations and it has nothing to do with PTT; it needs to be preregistered and tested on data from 2026-10 onward.

## Layout

```
config.yaml           all parameters (tickers, aliases and exclusions, cutoff times, splits, costs, LLM)
src/pttsent/
  calendar.py         trading-day alignment
  prices.py           stock prices, ex-rights/ex-dividend adjustment, market index
  ptt.py              PTT conversion, extracting texts about the target stock
  sentiment/          lexicon / weak_labels / classifier / bert / llm_label
  features.py         daily features and prediction targets, development / final test split
  analysis.py         correlation, Granger, HAC regression, event study
  models.py           rolling prediction (logit, ridge, OLS), baselines, out-of-sample R², bootstrap CIs for AUC differences
  backtest.py         backtests with costs (long-short, day trades, borrowing and futures costs, thresholds, break-even, random benchmark)
  plots.py            charts
  panel.py            full-market daily price panel (TWSE + TPEx, adjustments, price limits, market index)
  events.py           merging [標的] posts, ticker parsing, entry-time alignment
  event_study.py      abnormal returns, two-way clustered standard errors, calendar-time portfolios
  volume.py           pre-open discussion volume, night-session and US alignment, rolling OLS, Clark–West test
  explore.py          weekly data, extreme-sentiment days, tests of return differences on extreme days, hedged excess returns
scripts/              01–06, 13, 14 study 1; 07–10, fetch_prices.py, build_panel.py study 2; 11, 12, fetch_us.py study 3;
                      train_classifier.py, llm_label.py train the sentiment classifiers
tests/                invariants: time alignment, no look-ahead, costs
docs/                 (in Chinese)
  sentiment.md        study 1 details (text attribution, LLM labelling, classifier and BERT accuracy, backtests and extra results)
  explore_directions.md  plan and results for weekly, extreme-sentiment and excess-return exploration
  report_outline.md   capstone report outline
  prereg/             four preregistrations with results: studies 1, 2, 3, and the study 3 replication
```
