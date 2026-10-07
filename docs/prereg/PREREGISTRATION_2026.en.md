# Preregistration: out-of-sample tests from 2026-10 (third volume test, Hon Hai intraday signal)

> English translation, added 2026-10-07. The registered document is the Chinese original, [PREREGISTRATION_2026.md](PREREGISTRATION_2026.md), committed in `a1ed12a` on 2026-10-07 before any data dated 2026-10-01 or later was downloaded. Where the two differ, the Chinese original governs.

Registered: 2026-10-07. This document and `scripts/15_oos_2026.py` were committed **before** downloading any data dated 2026-10-01 or later.
`--final` first checks three things and refuses to run if any of them fails:
1. These two files and the registered code they depend on (`scripts/11_volume_prereg.py`, `src/pttsent/volume.py`, `features.py`, `models.py`) are all committed and unchanged since.
2. The evaluation period has ended.
3. This test has not produced a result before.

Each test is run only once, and the results, good or bad, are recorded as they are in the last section.

## Background

- **V (trading volume)**: the incremental power of pre-open PTT discussion volume to predict same-day trading volume has passed in two independent out-of-sample periods:
  2024 with CW t = 3.76 ([PREREGISTRATION_VOLUME.en.md](PREREGISTRATION_VOLUME.en.md)), and 2025-01 to 2026-09 with CW t = 2.94
  ([PREREGISTRATION_VOLUME_REPLICATION.en.md](PREREGISTRATION_VOLUME_REPLICATION.en.md)). The third test uses exactly the same settings to see whether the effect persists.
- **R (Hon Hai intraday)**: in the study 1 extension that predicts returns and goes long-short on the sign, only one of 36 combinations beat 95% of random positions in the development period:
  the price-and-volume-only logit for Hon Hai's next-day intraday (open to close) direction, which beat 96% in the development period and 97% in 2024 (see study 1 in the README).
  It was picked out of 36 combinations and has nothing to do with PTT; until it is confirmed on new data, it is only a lead.

## What has been seen from 2026-10 onward

All data on hand at registration ends before 2026-10-01: PTT to 2026-09-26, the price panel to 2026-09-24, US data to 2026-09-25, cnyes news to 2026-09-26, TAIEX futures to 2026-09.
**No** data dated 2026-10-01 or later has been downloaded or looked at.
The three trading days 2026-09-28 to 09-30 belong to neither evaluation period; they will be downloaded with the new data and used only for training.

## The two tests

| | V: pre-open discussion volume → same-day trading volume | R: Hon Hai next-day intraday direction |
|---|---|---|
| Evaluation period (target dates) | 2026-10-01 .. 2027-09-30 (about 245 trading days) | 2026-10-01 .. 2028-09-30 (about 490 trading days) |
| When it runs | After 2027-09-30, once the data is complete: `15_oos_2026.py volume --final` | After 2028-09-30: `15_oos_2026.py honhai --final` |
| Primary hypothesis | V1 | R1 |
| How the length was chosen | With the 2025-26 effect, power for 12 months is about 75% (see "Power") | Power is low: about 39% for 24 months and 52% for 36 months. 24 months is a compromise between waiting time and power |

V and R are two different questions; each uses α = 0.05, with no correction across them.

## V: pre-open discussion volume → same-day trading volume (third test)

Features, targets, baseline model, model and test all follow "Fixed settings" in [PREREGISTRATION_VOLUME.en.md](PREREGISTRATION_VOLUME.en.md).
`15_oos_2026.py` rewrites no logic; it loads `build` and `evaluate` directly from `11_volume_prereg.py`.
The hypotheses are renamed, with the same content:

- **V1 (primary)**: the Clark-West loss differential for `y_lv` (log volume), averaged across the three stocks on common trading days, is > 0; Newey-West with 5 lags, one-sided p < 0.05.
- **V2 (secondary)**: for each stock, the mean loss differential for `y_lv` is > 0, Holm-corrected at overall α = 0.05.
- **V3 (secondary)**: the mean loss differential for `y_vol` (log range) pooled across the three stocks is > 0, one-sided p < 0.05.

Also reported (not hypotheses), as in the previous two tests: each stock's out-of-sample MSE change, the effect of 1 standard deviation, and the increment after adding the opening gap.

| Item | 2025-26 test | This test |
|---|---|---|
| Evaluation period | Target dates 2025-01-02 .. 2026-09-24 | Target dates 2026-10-01 .. 2027-09-30 |
| Training data | From 2019-02 up to the start of each block | Same, so it includes 2024 to 2026-09 |
| US data | `data/prices/us_2026/` | `data/prices/us_2027/`, downloaded after the period ends with `fetch_us.py --end 2027-10-01 --out us_2027` |

Data to add after the evaluation period ends:

| Data | Source and method | Checks after adding |
|---|---|---|
| PTT 2026-09-27 .. 2027-09-30 | The same `ptt-stock-crawler` as for 2025-26; add 2027 to `ptt_years` in `config.yaml` | No duplicate IDs, all dates in range; in the rebuilt `texts_{ticker}.parquet`, every row before 2026-09-27 is identical to before the rebuild |
| cnyes news, TAIEX futures, disposition stocks, ex-rights/ex-dividend | The same code as for 2025-26 | No truncation; re-download one old month and compare sha256 |
| Price panel | `fetch_prices.py --end 2027-09-30` + `build_panel.py` | Rows up to 2026-09-24 identical column by column to before the rebuild |

After adding the data, first run `15_oos_2026.py volume` (without `--final`) to confirm the usable days for each stock, then run `--final`.

Decision rules:
- V1 passes: the effect also holds in a third independent period.
- V1 fails: the conclusion becomes "the 2024 and 2025-26 results did not recur in 2026-10 to 2027-09", written into the README as is, with a discussion of possible reasons (for example, changes in discussion volume). The earlier two results are still reported, but can no longer be described as "the effect continues to hold".
- V2 and V3 are reported separately and do not change the V1 decision.

## R: Hon Hai next-day intraday direction, price and volume only

### Fixed settings

| Item | Setting |
|---|---|
| Stock, timing | 2317 Hon Hai. Each row is trading day t, predicted after t's close; the position is entered at t+1's open and exited at its close (a day trade) |
| Target | log(close / open) on t+1; the training label is its sign, and flat days are not used for training |
| Features | Study 1's price-and-volume features `features.PRICE_FEATURES`, computed with `features.price_features`: the day's return, returns 1 to 4 days back, 5-day cumulative return, 20-day volatility, volume z-score, the day's intraday return, and the market return. Rows with any missing feature are not used |
| Data | 2317's open, close, shares traded, adjusted return and intraday return from the self-built panel `data/prices/panel.parquet`; the market return is the TAIEX log return in `market.parquet` |
| Model | Study 1's logit (standardized, then `LogisticRegression(C=0.01)`); `models.walk_forward` with at least 250 rows, retrained every 21 rows; training rows start on 2019-01-01 and run up to the start of each block, using only rows whose target date is no later than the end of the evaluation period |
| Position | Long if the probability of a rise is > 0.5, short if < 0.5, flat at exactly 0.5 (excluded from the test) |
| Test | OLS of t+1's simple intraday return on a constant and a "long" dummy, Newey-West with 5 lags. The coefficient equals the mean return on long days minus the mean on short days; one-sided test that it is > 0 |
| Evaluation period | Target dates 2026-10-01 .. 2028-09-30 |

**R1 (the only hypothesis)**: the coefficient above is > 0, one-sided p < 0.05. If there are fewer than 10 long days or fewer than 10 short days, it cannot be tested and counts as not passed.

### Differences from study 1, and why

- **Price data source**: study 1's price-and-volume features read FinMind daily prices, which end in 2025-03, so the self-built panel is used instead.
  I reproduced study 1's positions on periods already used (`15_oos_2026.py honhai`):
  - In the development period, the long/short direction is the same on 99.17% of 963 common trading days; in 2024, 100% of 241 days.
  - The differences come from three days. On 2018-10-26 Hon Hai resumed trading after a capital reduction; the FinMind version has a missing return there, while the panel adjusts for it. The panel's TAIEX is missing 2019-04-29 and 04-30; those two rows are not used for training, which shifts the start of every later retraining block by two rows.
- **Test**: study 1 used a "random position" test: shuffle the positions, keep the number of long and short days fixed, and see what share of random positions the actual gross return beats.
  When the effect is set to 0 in simulation, this test rejects 6-9% of the time, above the nominal 5%. The reason is that long days cluster, and intraday volatility clusters too; shuffling the positions destroys that structure.
  The Newey-West t test rejects 3-4% of the time, so it is used instead.
  With it, the original evidence is weaker than study 1 reported: t = 1.40 in the development period (one-sided p ≈ 0.08), t = 1.96 in 2024, and t = 2.30 for the two combined.

### Also reported (not hypotheses)

- Study 1's random-position test pctile.
- Backtests of long-short and of shorting every day, under single-stock futures and stock costs: total return, Sharpe, maximum drawdown.
- Share of long days, and break-even cost compared with the actual cost.

### Decision rules

- R1 passes: Hon Hai's price-and-volume-only intraday direction signal holds out of sample. The report must state two things clearly: the signal was picked out of 36 combinations, and it has nothing to do with PTT. Whether it makes money is judged by the futures-cost backtest, not by R1.
- R1 fails: the conclusion is "inconclusive", not "no effect", because power is at most about 40% (see the next section). The evaluation period will not be extended after seeing the result, and the same data will not be retested with a different test; any later test needs data from 2028-10 onward and a new registration.

## Power

`python scripts/15_oos_2026.py power`: a block bootstrap (blocks of 20 days) resamples the reference period to the length of the evaluation period, 600 times each, one-sided α = 0.05.
"Half" means shrinking the effect to half of the reference period's.

**V1, log volume pooled across three stocks** (bootstrap; in parentheses, the analytic result from scaling the t value by the number of days)

| Effect from | 6 months, full | 12 months, full | 12 months, half |
|---|---|---|---|
| Development period | 69% (64%) | 90% (88%) | 46% (41%) |
| 2024 | 92% (85%) | 100% (98%) | 70% (60%) |
| 2025-26 | 54% (50%) | 75% (75%) | 31% (31%) |

The most recent period (2025-26) has the smallest effect. With that effect, power for 6 months is only about half, so the evaluation period is 12 months.

**R1, Hon Hai intraday** (Newey-West t test)

| Effect from | δ (long days − short days) | 12 months | 24 months | 36 months |
|---|---|---|---|---|
| Development period | +0.21% | 14% | 16% | 27% |
| 2024 | +0.63% | 69% | 88% | 97% |
| Development period + 2024 | +0.32% | 24% | 39% | 52% |
| Development period + 2024, half | +0.16% | 11% | 17% | 18% |
| Effect = 0 (type I error) | 0 | 4% | 3% | 4% |

With the effect at 0, the random-position test rejects 9% / 6% / 7% of the time at 12 / 24 / 36 months.

The reference effects for both tests are upper bounds:
- V's development-period effect was settled only after exploration.
- R was picked out of 36 combinations, and its 2024 effect is three times the development-period effect, likely an overestimate.
- A more realistic estimate for R is the "development period + 2024, half" row: power for 24 months is under 20%.

## Changes

None.

## Results

### V (to be filled in after 2027-09-30)

Not run yet.

### R (to be filled in after 2028-09-30)

Not run yet.
