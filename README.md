# PTT Sentiment and Taiwan Stocks

[中文版](README.zh-TW.md)

My individual part of a five-person undergraduate capstone at Soochow University. The team studies Taiwan stocks with different methods, and the advisor asked us to work individually first.

The project tests whether the Stock board on PTT, Taiwan's largest bulletin board system, carries information beyond price and volume.
The question I care about is how much a price-and-volume model improves once PTT is added.
I explored each of the four studies on a development period first, then preregistered its hypotheses and tested them once on a period I had not looked at.

## Background

Before this repo, I ported the design of Li, Liu, Ye, Zhao & Zhao, *It Depends on When You Search* (MIS Quarterly), to Taiwan long-tail stocks, with PTT post counts in place of Google search volume: [taiwan-market-attention](https://github.com/jameschen108/taiwan-market-attention).
The main hypothesis, H1, was significant only under the paper's inference standard, which clusters by stock alone (t = 2.68-3.49), and not with two-way clustering (t = 1.26-1.63). So I switched methods and started this repo.
The PTT corpus, the FinMind prices for the 267-stock universe, and the ex-rights and disposition-stock data carry over from that round. The code here is new.

## Results

| Study | Question | Answer | Preregistration |
|---|---|---|---|
| 1. Daily sentiment | Does it predict next-day returns? | **No.** Sentiment reacts to prices that have already moved; the only hint in the development period did not recur in 2024; predicting returns and going long-short on the sign fails as well | [PREREGISTRATION.md](docs/prereg/PREREGISTRATION.en.md) |
| 2. [標的] posts | Do the long/short calls authors declare predict later returns? | **No.** Authors chase rallies; the small gap comes from continuation after limit-locked closes, and buying on bullish posts loses money after costs | [PREREGISTRATION_TARGET.md](docs/prereg/PREREGISTRATION_TARGET.en.md) |
| 3. Pre-open discussion volume | Does discussion between the close and the next open predict that day's trading volume? | **Yes**, but the effect is small (out-of-sample MSE down 0.5-3%); passed both out-of-sample tests, 2024 and 2025-2026 | [PREREGISTRATION_VOLUME.md](docs/prereg/PREREGISTRATION_VOLUME.en.md), [replication](docs/prereg/PREREGISTRATION_VOLUME_REPLICATION.en.md) |
| 4. Large moves | Do news and discussion show anything before a stock's large up or down days? | **Pre-open discussion signals whether the day will bring a large move** (direction unknown): after controlling for cnyes news and MOPS announcements it passed out of sample, 2024-01 to 2026-09 (t = 4.61). LLM attribution (κ = 0.63 against hand labels) finds no cause in the surrounding week's news and announcements for over half of large moves; identified causes are mostly industry themes, earnings and company events; moves without news show no clear reversal afterwards. **Pre-open sentiment slightly predicts the intraday direction**: passed out of sample, 2024-01 to 2026-09 (t = 2.08), but the effect is too small to survive trading costs | [PREREGISTRATION_LARGE_MOVES.md](docs/prereg/PREREGISTRATION_LARGE_MOVES.md), [PREREGISTRATION_DIRECTION.md](docs/prereg/PREREGISTRATION_DIRECTION.md) (Chinese only) |

PTT reflects prices that have already moved, but pre-open discussion does signal whether the day will be busier than usual and whether a large move is coming; pre-open sentiment also hints at the intraday direction, though too weakly to make money.
Details for study 1 are in [docs/sentiment.md](docs/sentiment.en.md), for studies 2 and 3 in their [preregistration documents](docs/prereg/), and for study 4 in [docs/large_moves.md](docs/large_moves.md) (Chinese only for now). The documents under `docs/` are written in Chinese; the `.en.md` files are English translations added later, and the Chinese originals are the registered versions.

## Terms

- **PTT**: Taiwan's largest bulletin board system. The Stock board is its stock discussion board. Each comment under a post (推文) is a push (推, up), a boo (噓, down) or a neutral arrow (→).
- **[標的] posts**: the Stock board's format for single-stock pitches. Board rules require the author to state long or short.
- **TWSE, TPEx, TAIEX**: the Taiwan Stock Exchange, the Taipei Exchange (the OTC market), and the TWSE's main index. TAIEX futures also trade in a night session after the stock market closes.
- **Institutional investors (三大法人)**: foreign investors, investment trusts and securities dealers, whose daily net buying is published by the exchange.
- **Price limit (漲跌停)**: a Taiwan stock can move at most ±10% a day. When the close is locked at the limit, orders usually cannot be filled.
- **Disposition stocks (處置股)**: stocks the exchange puts under trading restrictions after abnormal trading.
- **Day trade (當沖)**: buying and selling the same stock on the same day. A sell-first day trade needs no securities borrowing.
- **Tickers**: 2330 TSMC, 2603 Evergreen Marine, 2317 Hon Hai (Foxconn), 0050 Yuanta Taiwan 50 ETF. TSM is TSMC's ADR on the NYSE.
- **Data sources**: FinMind is an open API for Taiwan market data, cnyes (鉅亨網) is a Taiwanese financial news site, and pttweb is a third-party PTT archive whose copy includes deleted posts.
- **MOPS (公開資訊觀測站)**: the Market Observation Post System, where listed companies file material announcements (重大訊息), financial reports and monthly revenue.

## Data

Raw data is not in this repo. The code reads it from a local folder, `/Users/jameschen/Project/data` (read-only; set the environment variable `PTT_DATA_DIR` to point elsewhere):

| Use | Path | Notes |
|---|---|---|
| PTT posts and comments | `pttcc/stock_{year}.jsonl` | 2019-2024: 158k posts, 15.63M comments; 2025 to 2026/09/26: 45k posts, 5.56M comments (used only in the study 3 replication and study 4's period A) |
| PTT [標的] posts | pttweb | 2015/04 to 2025/01, including deleted posts (study 2) |
| Daily stock prices, market index | `raw/finmind/price/`, `raw/twse_taiex/` | 267 stocks + 0050, 2014 to 2025/03 (study 1) |
| Ex-rights/ex-dividend, disposition stocks | `raw/twse_exrights/`, `raw/twse_punish/` | Return adjustment; controls in study 3 |
| cnyes news, TAIEX futures | `raw/cnyes/headline/`, `raw/taifex_tx/` | Controls in studies 3 and 4; exploratory analysis in study 2 |
| Day-trade volume | `raw/twse_twtb4u/` | TWSE TWTB4U, 2015-2024, listed stocks only (descriptive part of study 4) |

Studies 2, 3 and 4 also use a full-market daily price panel I downloaded myself (TWSE + TPEx, 2015 to 2026/09, including delisted stocks and adjusted for capital reductions) and US data (TSM, ^SOX), stored in `data/prices/`.
Study 4 also uses MOPS material announcements I fetched myself (267 stocks, 2019 to 2026/10, `scripts/fetch_mops.py`), stored in `data/mops/`.
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

# Out-of-sample tests from 2026-10 (PREREGISTRATION_2026.md)
python scripts/15_oos_2026.py power             # power simulation (periods already used only)
python scripts/15_oos_2026.py volume --final    # third study 3 test: run once after 2027-09-30, when the data is complete
python scripts/15_oos_2026.py honhai --final    # 2317 intraday signal: run once after 2028-09-30

# Study 4: news and discussion around large moves (docs/large_moves.md)
python scripts/16_large_moves.py names          # name matching, round 1; recheck and rules are round 2 and the final rules (settled; do not resample with --force)
python scripts/16_large_moves.py events         # abnormal returns, large-move days, follow-on events, control days
python scripts/16_large_moves.py windows        # news and PTT counts in five windows per stock-day, about 10 minutes the first time
python scripts/16_large_moves.py analyze        # development-period P1-P4 (run once already; do not rerun)
python scripts/fetch_mops.py                    # MOPS announcements 2019-2023, about 1.5 hours; add --start 2024 --end 2026 for period A
python scripts/16_large_moves.py supplement     # supplementary analysis with MOPS announcements (run once already)
python scripts/17_large_moves_prereg.py power   # power (development period only)
python scripts/17_large_moves_prereg.py final A # out-of-sample test, 2024-01 to 2026-09 (run once already; do not rerun)
python scripts/17_large_moves_prereg.py final B # 2026-10 to 2027-09: run once after the period ends and the data is complete
python scripts/18_large_moves_causes.py build   # cause classification: build LLM inputs, samples and the hand-labelling sheet (no cost)
python scripts/18_large_moves_causes.py submit --effort low   # send to the Batches API, about $9 (run once already); then status, collect
python scripts/18_large_moves_causes.py report  # category shares, agreement and κ against hand labels
python scripts/19_large_moves_direction.py texts     # pre-open sentiment: extract pre-open PTT texts (docs/large_moves_direction.md)
python scripts/19_large_moves_direction.py score     # BERT scoring, about 1 hour on an Apple GPU
python scripts/19_large_moves_direction.py analyze   # development-period D1, D2 (run once already; do not rerun); gapcheck is the post-hoc opening-gap check
python scripts/20_direction_prereg.py final A        # out-of-sample test, 2024-01 to 2026-09 (run once already; do not rerun); score A and check A come first
python scripts/20_direction_prereg.py final B        # 2026-10 to 2029-09: run once after the period ends and the data is complete
```

- Steps 2-5 and 13 accept `--ticker` and `--method` (`lexicon` / `classifier_weak` / `classifier_llm` / `classifier_llm_pooled` / `classifier_bert`).
  Steps 3-5 and 13 accept `--target open_to_close`, which predicts next-day open-to-close instead and writes results to `*_oc`. Classifier training is described in [docs/sentiment.md](docs/sentiment.en.md).
- LLM labelling needs `ANTHROPIC_API_KEY=...` in `.env` at the project root (gitignored).
- Results go to `output/{ticker}/{method}/` and intermediate files to `data/`; neither is in git.

## Shared design

- Which texts count as discussing a stock (`ptt.py`): posts whose title mentions the ticker or an alias, with their whole comment thread, plus comments that mention it themselves.
  A mention only in the body does not count, since those are mostly templates or passing references. When an alias collides with another company, `exclude` in `config.yaml` removes those words first (for Evergreen Marine, 長榮航, EVA Air).
- Time alignment (`calendar.py`): a post before the 13:30 close counts toward that day; later posts, weekends and holidays count toward the next trading day. Features for day t use only posts visible before day t's close. With `--target open_to_close` the day is cut at 09:00 instead, so features run up to the open of t+1.
- Returns (`prices.py`): adjusted with reference prices on ex-rights/ex-dividend dates. Returns beyond the price limit that cannot be adjusted are set to missing.
- Validation: development and final test periods are kept separate. I commit the hypotheses, thresholds and test code first, run the final test once, and record the result whichever way it goes.
  The hypotheses of studies 1-3 have used up 2024 and 2025-01 to 2026-09.
  Study 4 is a new hypothesis, so its out-of-sample test uses 2024-01 to 2026-09, which it had not touched; study 3 had already looked at a closely related relationship over the same period, and the preregistration says so.
  Data from 2026-10 onward is preregistered for a third study 3 test, the 2317 intraday signal ([PREREGISTRATION_2026.md](docs/prereg/PREREGISTRATION_2026.en.md)), the study 4 replication ([PREREGISTRATION_LARGE_MOVES.md](docs/prereg/PREREGISTRATION_LARGE_MOVES.md)), and the pre-open sentiment replication ([PREREGISTRATION_DIRECTION.md](docs/prereg/PREREGISTRATION_DIRECTION.md), through 2029-09).

## Study 1: daily sentiment and next-day returns

- Daily sentiment (`features.py`): averaged within each account for the day first, then across accounts, so that spamming accounts do not dominate. A day with no sentiment signal is missing, not 0.
- Sentiment measures: a lexicon; a weak-label classifier, whose labels are the long/short that [標的] authors declare; a TF-IDF classifier trained on LLM labels (Claude Opus 5.5); and BERT fine-tuned on the same labels.
  BERT reaches macro-F1 0.550 on the test set against 0.495 for TF-IDF, and 0.519 against 0.440 on 2317, which was not used in training. Details in [docs/sentiment.md](docs/sentiment.en.md).
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

Full tables in [docs/sentiment.md](docs/sentiment.en.md#predicting-returns-and-long-short).

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

## Study 4: news and discussion around large moves

Studies 1-3 use PTT to predict prices or volume. This one goes the other way: find the days a stock moved unusually far, then look back at the news and discussion around them.

- Events: 267 stocks; the absolute abnormal return (net of β × market) is ≥ 3σ and ≥ 3%.
  Ex-rights days, disposition periods and the first day after a trading halt are excluded. A large move within 5 trading days of an earlier one counts as a continuation and stays out of the main analysis.
  The development period, 2019-03 to 2023-11, has 3,656 main events (2,671 up, 985 down). Control days are days with no event in the 5 days before or after, for the same stock.
- Name matching: two-character short names often collide with ordinary words (冠軍 "champion", 全國 "nationwide", 新興 "emerging", 統一 "unify"). I used cnyes stock tags to flag 76 stocks and hand-checked samples of PTT titles in two rounds.
  The 19 short names below 80% precision match by ticker only, and the year-like tickers (2006-2015) match by name only. The rules were committed before any events were computed.
- Windows: the week before, pre-open (previous close to 09:00), intraday, next day, and the week after.
- Regressions: stock × year and date fixed effects, with standard errors two-way clustered by stock and date.

Development period 2019-2023, primary tests (BH correction, FDR 10%):

| Test | Coefficient | t | Verdict |
|---|---|---|---|
| P1 pre-open PTT abnormal → large move that day | +0.0060 | 7.87 | **Passed** |
| P2 previous-week PTT abnormal → large move that day | −0.0024 | −5.58 | Failed (wrong sign) |
| P3 large moves with no cnyes news → next 5 days | −0.50% | −1.48 | Failed |
| P4 same, next 20 days | +0.03% | 0.06 | Failed |

1. News and discussion mostly come after the price has moved. cnyes news does not rise in the week before a large move; news and PTT pick up from the pre-open window and peak intraday and the next day.
   The day-trade share is 10 points above normal on large up days and unchanged on large down days.
2. Most large moves have no identifiable news. Between the previous close and the day's close, 82.5% have no cnyes headline; adding MOPS material announcements, 69.6% still have neither.
   94% of MOPS announcements are filed after the close, and they cluster on the night before a large move: 22% of large down days had one the night before, against 7.5% of ordinary days.
3. Moves without news show no clear reversal. Their next 5 days trail moves with news by 0.5%, in the direction the US literature finds (Chan 2003), but not significantly.
   Events that closed locked at the price limit continue by 1.75% over the next 5 days, consistent with study 2.
4. P1 is not just relaying announcements. This analysis was added after seeing P1. With MOPS announcements as controls, the coefficient falls from 0.0060 to 0.0050.
   Splitting the pre-open window at 08:30, the signal is almost all before 08:30 (t = 5.95); the 08:30-09:00 trial-matching window gives 0.0004 (t = 0.34), so the discussion is not a reaction to indicative opening prices.

Preregistered out-of-sample test L1, using the specification from the supplementary analysis: PTT abnormal from the previous close to 08:30, controlling for cnyes news, MOPS announcements, and price and volume.

| | Period | Main events | Coefficient | t | Per +1 SD | Verdict |
|---|---|---|---|---|---|---|
| Development (reference) | 2019-03 to 2023-11 | 3,656 | 0.0047 | 5.95 | — | — |
| **A** | 2024-01 to 2026-09 | 2,277 | 0.0051 | 4.61 | +0.29 points (base rate 1.35%) | **Passed** |
| B | 2026-10 to 2027-09 | — | — | — | — | runs after 2027-09-30 |

- The coefficient in A is no smaller than in development, and t matches the 4.49 expected beforehand.
- The L1 specification was chosen after seeing the development period, and study 3 had already looked at a related relationship (pre-open discussion → volume and range) over period A.
- The signal predicts whether a large move happens, not its direction, so it cannot decide long or short.
- Monthly revenue filing times are not controlled: MOPS history has the amounts but not when they were filed.

Causes of large moves (descriptive): Claude Opus 5.5 sorts each event's cnyes news, MOPS announcement subjects and PTT titles from the surrounding week into 7 categories (`18_large_moves_causes.py`, cost $9.36).
Before seeing any answers I hand-labelled 100 events: 72% agreement, κ = 0.63, just above the pre-set threshold of 0.6.

| Category (3,526 main events, development period) | As specified | Strict rule (post hoc) |
|---|---|---|
| No cnyes news and no MOPS announcement in the surrounding week | 38.6% | 38.6% |
| News exists, but no cause found | 17.2% | 34.2% |
| Industry themes | 12.9% | 9.3% |
| Earnings, guidance, earnings calls | 13.8% | 7.9% |
| Company events | 12.2% | 7.6% |
| Monthly revenue, analyst reports and flows, price-only reports, other | 5.4% | 2.4% |

1. The LLM is more lenient than I am. In 22 of the 28 disagreements I found no cause while the LLM inferred one from timing alone, for example when the night before had only a "board approved the quarterly report" subject line.
   Counting the LLM's low-confidence answers as no cause (a rule chosen after seeing the validation) gives 82% agreement and κ = 0.72. Either way, over half of large moves have no identifiable cause (56% and 73%).
2. Large moves lean toward industry themes. Compared with ordinary days that have news, industry themes are 21% against 13%, and the company's own events 20% against 30%. Industry themes are 15.5% of large up moves but only 5.7% of large down moves.
3. Causes are mostly public before the open: 70% of identified causes appear before the open. Over 80% of earnings and company-event causes are announcements from the night before; a third of industry themes appear intraday, and 40% of analyst reports come after the close, which looks like explanation after the fact.
4. Monthly revenue is only 2%, which may be a data limit: monthly revenue filings are not MOPS material announcements, so only those that made a cnyes headline are visible.

Pre-open sentiment and direction ([docs/large_moves_direction.md](docs/large_moves_direction.md), Chinese only): discussion volume only says whether a large move is coming, so this asks whether the sentiment of the same posts says which way.
PTT texts mentioning the stock from the previous close to 08:30 are scored with study 1's BERT classifier and averaged by account; days with fewer than 3 accounts expressing a view are left out.

| Test | Sample | Coefficient | t | Verdict |
|---|---|---|---|---|
| D1 pre-open sentiment → large up vs down (development) | 233 large-move events | 0.650 | 5.79 | Passed (BH) |
| D2 pre-open sentiment → open-to-close return, on days with pre-open discussion at least 2.7× normal (development) | 6,067 days | 0.0047 | 2.78 | Passed (BH) |
| Post hoc: D2 controlling for the opening gap (development) | 6,067 days | 0.0048 | 2.86 | — |
| **E1 = D2 with the gap control, preregistered, A: 2024-01 to 2026-09** | 3,303 days, 219 stocks | 0.0048 | 2.08 | **Passed** |
| E1, B: 2026-10 to 2029-09 | — | — | — | runs after 2029-09-30 |

1. Direction of large moves: 79% of large moves with bullish pre-open sentiment were up, against 47% with bearish sentiment. Controlling for the opening gap halves the coefficient, so about half of it is PTT and the gap both reacting to the previous night's news. D1 cannot be traded and is descriptive only.
2. Intraday returns: controlling for the opening gap leaves the coefficient unchanged, and the gap itself does not continue intraday. So sentiment carries a little information beyond the opening price, and the out-of-sample coefficient matches development.
3. No money in it. One standard deviation of sentiment adds only 0.10% to the intraday return. Trading the sign of sentiment from open to close earns 0.033% gross per trade in period A (t = 0.71): −0.40% after stock day-trade costs and −0.011% with futures costs.
4. No conflict with study 1, which used daily sentiment cut at the close to predict the next day's close-to-close return. Here only sentiment before 08:30 is used, the return is that day's open to close, and only days with unusually heavy discussion count. Sentiment does react to the previous day's move (correlation 0.17), as in study 1, and the regression controls for it.
5. It applies only to stocks people discuss: just 6% of large-move events have 3 or more accounts expressing a pre-open view. The 08:30 cutoff, the discussion threshold and the gap control were all chosen after seeing the development period.

## Next steps

1. Change the target. My first attempt used post counts cut at 13:30 to predict the next day's |return|, and it added nothing beyond past volatility
   (out-of-sample MSE change 2330 −0.2%, 2603 +0.1%, 2317 +0.4%, all CIs contain 0; the in-sample t = 3.2 for 2603 comes from heavy posting and high volatility coinciding in the 2021 shipping boom).
   Switching to discussion between the close and the next open is what worked (study 3). 5-day returns or returns relative to the market are also worth trying.
2. Add controls. The sentiment models do not yet include news volume or the night session, and institutional investor flows are not used anywhere yet. Study 3 does not yet control for freight rates (Evergreen), Apple and Nvidia (Hon Hai), or earnings call and monthly revenue release dates.
   Study 4 now includes MOPS announcements, which cover most earnings calls, financial reports and dividends. Monthly revenue filing times cannot be recovered for past years; they can only be collected from now on by fetching MOPS's same-day filings (t51sb10) every day.
3. Trade less often. I have tried weekly frequency, trading only on extreme sentiment, and predicting returns relative to the market; none of the 12 pre-specified tests passed FDR correction
   ([docs/explore_directions.md](docs/explore_directions.en.md)).
4. Cover more stocks. Of 301 stocks scanned, only 2330, 2603 and 2317 had a median of ≥ 20 comments per trading day in 2019-2023.
   UMC, Yang Ming, China Steel and AUO had enough only around 2021, almost all others had ≤ 5, and short names such as 統一, 南亞, 大成 and 華電 mostly collide with other words.
   A daily panel is not feasible. It would need weekly frequency or a cross-stock event study. Study 4 is the latter: large-move days across 267 stocks are the events, and colliding short names match by ticker only after a hand check.
5. Test on new out-of-sample data. A third study 3 test is preregistered for target dates 2026-10-01 to 2027-09-30 and runs once after the period ends.
   With the 2025-26 effect, its power is about 75% ([PREREGISTRATION_2026.md](docs/prereg/PREREGISTRATION_2026.en.md)).
6. Check an intraday price-and-volume signal for 2317. A logit using only price and volume to predict Hon Hai's next-day intraday direction beats 96% of random positions in gross return in the development period and 97% in 2024.
   With futures costs, its long-short returns +64% and +16%, against +23% and −27% for shorting every day over the same periods. I picked it out of 36 combinations and it has nothing to do with PTT.
   The random-position test overstates significance when long days cluster; with a Newey-West t test instead, the development period gives only t = 1.40.
   It is preregistered for target dates 2026-10-01 to 2028-09-30; power is at most about 40%, so a failure would only count as inconclusive.
7. Study 4 from here:
   - Power for L1's period B (2026-10 to 2027-09) is about 86% if the effect matches the development period and about 39% at half that.
   - Power for pre-open sentiment E1's period B (2026-10 to 2029-09) is about 74% at the development effect and about 31% at half; a failure would only count as inconclusive.
   - The cause classification only has MOPS subject lines. With the full announcement text, filings such as "board approved the quarterly report" would show whether the news was good or bad, which is where the LLM and I disagreed most.

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
  large_moves.py      study 4: name matching, abnormal returns and large-move days, window counts, fixed-effects regressions with two-way clustering
  mops.py             parsing MOPS material announcements
  causes.py           study 4: LLM inputs, prompt and output schema for classifying the causes of large moves
  direction.py        study 4: pre-open PTT texts and daily sentiment (direction)
scripts/              01–06, 13, 14 study 1; 07–10, fetch_prices.py, build_panel.py study 2; 11, 12, fetch_us.py study 3; 15 out-of-sample tests from 2026-10;
                      16-20, fetch_mops.py study 4;
                      train_classifier.py, llm_label.py train the sentiment classifiers
tests/                invariants: time alignment, no look-ahead, costs
docs/                 Chinese originals, with English translations in *.en.md
  sentiment.md        study 1 details (text attribution, LLM labelling, classifier and BERT accuracy, backtests and extra results)
  explore_directions.md  plan and results for weekly, extreme-sentiment and excess-return exploration
  large_moves.md      study 4 plan, name check, results, supplementary analysis and cause classification (Chinese only for now); large_moves/ holds the name check sheets, matching rules and hand labels for the cause classification
  large_moves_direction.md  study 4 extension: plan, results and opening-gap check for pre-open sentiment and direction (Chinese only for now)
  report_outline.md   capstone report outline (Chinese only)
  prereg/             seven preregistrations with results: studies 1, 2, 3, the study 3 replication, the out-of-sample tests from 2026-10, study 4, and study 4's pre-open sentiment
```
