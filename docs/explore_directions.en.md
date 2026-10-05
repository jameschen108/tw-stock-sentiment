# Exploration: weekly frequency, extreme sentiment, excess returns

> English translation, added 2026-10-05, of [explore_directions.md](explore_directions.md). The plan and code were committed in `94fddff` on 2026-10-02 before any result was seen; the results were added in `54c6d1b`. Where the two differ, the Chinese original governs.

Daily long-short cannot pay its costs (see "Predicting returns and long-short" under study 1 in the README). This document tries three directions, all on the development period 2019-2023 only.
This document and its code were committed **before** looking at any result; all settings, the primary tests and the multiple-comparison correction were fixed here first, and the results, good or bad, are added in the last section.
This is exploration, not confirmation: any signal that passes has to be preregistered separately and tested on data from 2026-10 onward.

## Common settings

- Stocks: 2330, 2603, 2317; sentiment from BERT (`classifier_bert`), close to close.
- No data from 2024 onward is touched: rows whose target date falls in 2024 are never used.
- Costs as in study 1: stocks (shorting requires securities borrowing) and single-stock futures; backtests also report the break-even cost and a random benchmark with the same long/short split.

## Direction 1: weekly frequency

The position for next week is set at the close of each week's last trading day and held to the close of next week's last trading day; that is at most about 52 trades a year, 1/5 of daily.

- Target: next week's log return (the sum of next week's daily `ret`). Only complete weeks: the target week's last day must be before 2024-01-01.
- Features: price and volume from the row of the week's last day (`PRICE_FEATURES`; `ret_5` is this week's return); sentiment is the mean of the week's daily `SENT_FEATURES`.
- Models: as daily, A / B × ridge, OLS, logit; predictions start after at least 52 weeks, retrained every 4 weeks.
- Strategy: long when the forecast is > 0, short when < 0; long-only is reported too.

## Direction 2: trade only on extreme sentiment

Trade only when sentiment falls in the bottom 10% or top 10% of the previous 250 trading days (excluding the day itself, at least 60 days); otherwise stay flat.

- The direction is fixed in advance as **contrarian**: buy at the next day's close after extreme pessimism and short after extreme optimism, to test the claim that "PTT is a contrarian indicator". Following the crowd is the same signal reversed, and its gross return differs only in sign.
- Holding period: 1 day and 5 days. With 5 days, a new extreme day during the holding period replaces the old one.
- Test: `future return ~ pessimistic day + optimistic day + ret + ret_5 + mkt_ret + vol_20` (OLS, Newey-West, 5 lags for 1 day and 10 for 5 days),
  comparing the coefficient difference "pessimistic − optimistic", two-sided. The price and volume controls separate "extreme sentiment" from "the stock just moved".

## Direction 3: predict returns relative to the market

Stock return minus market return, with the position hedged by TAIEX futures (beta fixed at 1). This removes the bull-market drift, so the long-short is no longer dominated by "on average it goes up".

- Target: `excess_next` = next-day stock return − next-day TAIEX return (log).
- Models and strategy as daily: A / B × ridge, OLS, logit, long-short on the sign.
- Costs: the stock leg at stock costs, plus one futures cost for the hedge leg; in the all-futures version, both legs at futures costs.

## Primary tests and multiple comparisons

12 primary tests in total, with Benjamini-Hochberg controlling FDR at 10%:

| Direction | Test | Count |
|---|---|---|
| 1. Weekly | OLS B vs A, Clark-West, one-sided | 3 (three stocks) |
| 2. Extreme sentiment | Coefficient difference pessimistic − optimistic, two-sided | 6 (three stocks × 1 day, 5 days) |
| 3. Excess return | OLS B vs A, Clark-West, one-sided | 3 (three stocks) |

Ridge, logit AUC differences, backtests and the random benchmark are secondary results: described, not judged.
Only tests that pass FDR correction get written up as preregistrations and tested on data from 2026-10 onward; the rest are recorded as no finding.

## Results

Run: `python scripts/14_explore_directions.py`; the code and this plan were committed before the run (94fddff).
**None of the 12 primary tests passed**: the smallest p is 0.149, and after FDR correction all are ≥ 0.76. None of the three directions has a signal worth preregistering.

| Direction | Test | 2330 | 2603 | 2317 |
|---|---|---|---|---|
| 1. Weekly | CW t (OLS, B vs A) | 0.21 (p 0.42) | −0.41 (p 0.66) | 1.04 (p 0.15) |
| 2. Extreme sentiment | Pessimistic − optimistic, 1 day | +0.23% (p 0.23) | −0.27% (p 0.49) | −0.14% (p 0.46) |
| 2. Extreme sentiment | Pessimistic − optimistic, 5 days | +0.48% (p 0.37) | −0.86% (p 0.47) | +0.19% (p 0.57) |
| 3. Excess return | CW t (OLS, B vs A) | −0.71 (p 0.76) | −1.24 (p 0.89) | −0.17 (p 0.57) |

The extreme-sentiment numbers are coefficient differences after controlling for price and volume. The three stocks disagree in direction, so the claim that "PTT is a contrarian indicator" does not show up here.
There are 86-122 pessimistic days and as many optimistic days per stock.

### Secondary results (described, not judged)

- **Weekly**:
  - 198-204 weeks evaluated; group B trades 0.3-27 times a year, far less than daily.
  - With sentiment added, the ridge CW t is 0.22 / −1.46 / 1.61.
  - AUC difference: 2330 −0.011 [−0.042, +0.014], **2603 +0.054 [+0.009, +0.096]**, 2317 +0.022 [−0.037, +0.078].
    2603 is the only one of the 6 secondary AUC differences (3 weekly, 3 excess) that excludes 0, but the primary test for the same stock has a CW t of −0.41.
  - Stock long-short (B logit) against buy-and-hold: 2330 +107% vs +90%, 2603 +146% vs +1349%, 2317 −6% vs +41%.
    In the 2330 case 94% of the weeks are long, and it beats only 88% of random positions; it amounts to buy-and-hold with a few weeks out.
- **Extreme sentiment**:
  - All 6 stock long-short combinations lose 56-78%, and all of them lose to buy-and-hold under futures costs too.
  - The evaluation period here starts in 2019-01 (no training needed), so the benchmark differs from the other directions (2330 +209%).
- **Excess return**:
  - The benchmark becomes "always long the stock, short the market": 2330 +32%, 2603 +768%, 2317 −7%.
  - Stock long-short (group B OLS, logit) loses 78-93% in every case.
  - With futures costs, group B OLS returns +12% / +42% / −32%, still losing to the benchmark in every case.
- **Random benchmark**: of 42 long-short combinations, none beats 95% of random positions.
  3 fall below 5% (ridge for 2317, where the model guesses backwards), which is within what multiple comparisons would produce; reversing the signal after the fact would be data snooping, so it is not used.

### Conclusion

Trading less often, trading only on extreme sentiment and removing the market drift all failed to make PTT sentiment useful.
The only thing worth noting is the weekly AUC difference for 2603, but it is a secondary result that the primary test for the same stock does not support, so it is not preregistered.
