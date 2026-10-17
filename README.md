# PTT 情緒與台股走勢

[English version below](#ptt-sentiment-and-taiwan-stocks)

畢業專題：檢驗 PTT Stock 板的內容能不能提供量價以外的資訊。
核心問題是：加入 PTT 之後，比只用量價的模型好多少？
四個研究都先在開發期探索，再把假設預先登記，在沒看過的期間只測一次。

## 背景

在這個 repo 之前，我把 Li, Liu, Ye, Zhao & Zhao《It Depends on When You Search》（MIS Quarterly）的設計移植到台股長尾個股，用 PTT 發文數代替 Google 搜尋量：[taiwan-market-attention](https://github.com/jameschen108/taiwan-market-attention)。
主要假設 H1 只在原論文的推論標準（只按個股 cluster）下顯著（t = 2.68–3.49），改成雙重 cluster 就不顯著（t = 1.26–1.63），所以我換了方法，開了這個 repo。
上一輪的 PTT 語料、267 檔的 FinMind 股價、除權息與處置股資料沿用下來；程式是重寫的。

## 結論

| 研究 | 問題 | 結論 | 預先登記 |
|---|---|---|---|
| 一、每日情緒 | 能預測隔天報酬嗎？ | **不能**。情緒在反應已發生的股價；開發期唯一的跡象在 2024 沒有重現；預測報酬、照正負號做多空也一樣 | [PREREGISTRATION.md](docs/prereg/PREREGISTRATION.md) |
| 二、[標的] 文 | 作者自標的多空能預測之後的報酬嗎？ | **不能**。作者在追漲，小差距來自漲跌停鎖住的延續，跟著看多文買扣成本後賠錢 | [PREREGISTRATION_TARGET.md](docs/prereg/PREREGISTRATION_TARGET.md) |
| 三、開盤前討論量 | 收盤後到開盤前的討論量能預測當天成交量嗎？ | **能**，但效果小（樣本外 MSE 降 0.5–3%）；2024 與 2025–2026 兩次樣本外檢定都通過 | [PREREGISTRATION_VOLUME.md](docs/prereg/PREREGISTRATION_VOLUME.md)、[重複驗證](docs/prereg/PREREGISTRATION_VOLUME_REPLICATION.md) |
| 四、大漲跌日 | 個股大漲大跌之前，新聞與討論有沒有跡象？ | **開盤前的討論量能預告當天會不會大漲跌**（不分方向）：扣掉鉅亨新聞與重大訊息後，2024-01 到 2026-09 樣本外通過（t = 4.61）。用 LLM 歸因（人工驗證 κ = 0.63），過半的大漲跌在前後一週的新聞與公告裡找不到原因；找得到的多是產業題材、財報與公司事件；沒新聞的大漲跌之後沒有明顯反轉。**開盤前的情緒能稍微預告盤中方向**：2024-01 到 2026-09 樣本外通過（t = 2.08），但效果小到扣掉成本就沒了 | [PREREGISTRATION_LARGE_MOVES.md](docs/prereg/PREREGISTRATION_LARGE_MOVES.md)、[PREREGISTRATION_DIRECTION.md](docs/prereg/PREREGISTRATION_DIRECTION.md) |
| 五、槓桿 ETF（和 PTT 無關） | 用 0050 與 00631L 放大曝險，扣成本後的總報酬能贏過大盤嗎？ | **事先定的主要規則（波動目標）沒有顯著贏**：2024-01 到 2026-09 總報酬 +280% 對大盤 +187%，但 t = 1.39；固定 1.5 倍與 200 日均線通過了，但這段大盤走勢事先已知是大多頭，只持有 0050 也顯著贏。開發期最大回檔 −67% 到 −73% | [PREREGISTRATION_LEVERAGE.md](docs/prereg/PREREGISTRATION_LEVERAGE.md) |

PTT 反映的是已經發生的價格，但開盤前的討論能預告當天會不會比較熱、會不會大漲跌；開盤前的情緒還能稍微預告盤中的方向，只是效果小到賺不到錢。
研究五換一個問題：不找訊號，只放大市場曝險；三年的評估期太短，主要規則沒有顯著贏，槓桿的回檔則照樣出現。
研究一的細節在 [docs/sentiment.md](docs/sentiment.md)，研究二、三、五的細節在各自的[預先登記文件](docs/prereg/)，研究四在 [docs/large_moves.md](docs/large_moves.md)。

## 資料

原始資料讀自共用資料夾 `/Users/jameschen/Project/data`（唯讀，可用環境變數 `PTT_DATA_DIR` 改位置）：

| 用途 | 路徑 | 說明 |
|---|---|---|
| PTT 文章與推文 | `pttcc/stock_{年}.jsonl` | 2019–2024：15.8 萬篇、1,563 萬則推文；2025–2026/09/26：4.5 萬篇、556 萬則（只用在研究三的重複驗證與研究四的評估期 A） |
| PTT [標的] 文 | pttweb | 2015/04–2025/01，含被刪的文章（研究二） |
| 個股日價、大盤 | `raw/finmind/price/`、`raw/twse_taiex/` | 267 檔 + 0050，2014–2025/03（研究一） |
| 除權息、處置股 | `raw/twse_exrights/`、`raw/twse_punish/` | 還原報酬、研究三的控制變數 |
| 鉅亨新聞、台指期 | `raw/cnyes/headline/`、`raw/taifex_tx/` | 研究三、四的控制變數，研究二的探索性分析 |
| 當沖成交 | `raw/twse_twtb4u/` | 證交所 TWTB4U，2015–2024，只有上市股票（研究四的描述） |

研究二、三、四另外用自己下載的全市場日價面板（證交所＋櫃買，2015–2026/09，含已下市與減資還原）和美股資料（TSM、^SOX），存在 `data/prices/`。
研究四另外用自己抓的公開資訊觀測站重大訊息（267 檔，2019–2026/10，`scripts/fetch_mops.py`），存在 `data/mops/`。
面板和 FinMind 的 267 檔比對，61 萬個檔日的收盤全部一致；報酬只有 38 筆不同，都是原本用 FinMind 的流程漏掉上櫃期間的除息。

## 執行

```bash
pip install -r requirements.txt
python -m pytest                          # 時間對齊、不偷看未來、成本計算

# 研究一：每日情緒（預設 2330）
python scripts/01_prepare_ptt.py          # PTT 轉 parquet＋挑出目標股票的文字
python scripts/02_score_sentiment.py      # 每則文字打情緒分數
python scripts/03_build_features.py       # 每日特徵＋預測目標
python scripts/04_analyze.py              # 相關、Granger、迴歸、事件研究
python scripts/05_predict.py              # 滾動式預測＋回測
python scripts/06_prereg_test.py --final  # 預先登記的 2024 檢定（已跑過，不要再跑）
python scripts/13_return_predict.py       # 預測報酬與漲跌、照正負號多空；--period 2024 是補充（已跑過）
python scripts/14_explore_directions.py   # 探索：週頻、情緒極端、超額報酬（只用開發期）

# 研究二：[標的] 事件研究
python scripts/fetch_prices.py --market twse   # 全市場日行情，約 3.5 小時；研究三的重複驗證另加 --end 2026-09-24
python scripts/fetch_prices.py --market tpex   # 可以同時跑
python scripts/build_panel.py                  # 全市場面板＋驗證
python scripts/07_target_events.py             # [標的] 文 -> 事件表
python scripts/08_target_dev.py                # 開發期 2016–2023
python scripts/09_target_explore.py            # 探索性分析（FDR 校正）
python scripts/10_target_prereg.py --final     # 2024 最終測試（已跑過，不要再跑）

# 研究三：開盤前討論量與成交量
python scripts/fetch_us.py                                  # TSM、^SOX，到 2025-06
python scripts/fetch_us.py --end 2026-09-27 --out us_2026   # 重複驗證用，另存不覆蓋
python scripts/11_volume_prereg.py              # 開發期 2019–2023
python scripts/11_volume_prereg.py --final      # 2024 最終測試（已跑過，不要再跑）
python scripts/12_volume_replication.py --final # 2025-01 .. 2026-09 重複驗證（已跑過，不要再跑）

# 2026-10 起的樣本外檢定（PREREGISTRATION_2026.md）
python scripts/15_oos_2026.py power             # 檢定力模擬（只用已經用過的期間）
python scripts/15_oos_2026.py volume --final    # 研究三第三次驗證：2027-09-30 之後、資料補齊再跑一次
python scripts/15_oos_2026.py honhai --final    # 2317 盤中訊號：2028-09-30 之後跑一次

# 研究四：大漲跌日前後的新聞與討論（docs/large_moves.md）
python scripts/16_large_moves.py names          # 名稱比對第一輪；recheck、rules 是第二輪與定案（已定案，不要加 --force 重抽）
python scripts/16_large_moves.py events         # 超額報酬、大漲跌日、後續事件、控制日
python scripts/16_large_moves.py windows        # 每個股票日五個時間窗的新聞、PTT 則數，第一次約 10 分鐘
python scripts/16_large_moves.py analyze        # 開發期 P1–P4（已跑過，不要再跑）
python scripts/fetch_mops.py                    # 重大訊息 2019–2023，約 1.5 小時；評估期 A 另加 --start 2024 --end 2026
python scripts/16_large_moves.py supplement     # 加入重大訊息的補充分析（已跑過）
python scripts/17_large_moves_prereg.py power   # 檢定力（只用開發期）
python scripts/17_large_moves_prereg.py final A # 2024-01 .. 2026-09 樣本外檢定（已跑過，不要再跑）
python scripts/17_large_moves_prereg.py final B # 2026-10 .. 2027-09：期間結束、資料補齊後跑一次
python scripts/18_large_moves_causes.py build   # 原因分類：組 LLM 的輸入、抽樣、產生人工驗證表（不花錢）
python scripts/18_large_moves_causes.py submit --effort low   # 送 Batches API，約 9 美元（已跑過）；之後 status、collect
python scripts/18_large_moves_causes.py report  # 各類別比例、和人工標註的一致率與 κ
python scripts/19_large_moves_direction.py texts     # 開盤前情緒：抽出開盤前的 PTT 文字（docs/large_moves_direction.md）
python scripts/19_large_moves_direction.py score     # BERT 打分數，Apple GPU 約 1 小時
python scripts/19_large_moves_direction.py analyze   # 開發期 D1、D2（已跑過，不要再跑）；gapcheck 是控制開盤跳空的事後檢查
python scripts/20_direction_prereg.py final A        # 2024-01 .. 2026-09 樣本外檢定（已跑過，不要再跑）；之前要先 score A、check A
python scripts/20_direction_prereg.py final B        # 2026-10 .. 2029-09：期間結束、資料補齊後跑一次

# 研究五：槓桿 ETF（PREREGISTRATION_LEVERAGE.md）
python scripts/fetch_long_history.py      # 台指期 1998 起、報酬指數 2003 起、價格指數 1999 起、央行利率，約 35 分鐘
python scripts/21_leverage.py dev         # 開發期 2003–2023 的 C0–C3（會覆寫 output/leverage/dev_meta.json）
python scripts/21_leverage.py power       # 檢定力（只用開發期的 t 值）
python scripts/21_leverage.py final A     # 2024-01 .. 2026-09 樣本外檢定（已跑過，不要再跑）；之前要先 check A
python scripts/21_leverage.py final B     # 2026-10 .. 2029-09：期間結束、資料補齊後跑一次
```

- 步驟 2–5 與 13 可加 `--ticker`、`--method`（`lexicon` / `classifier_weak` / `classifier_llm` / `classifier_llm_pooled` / `classifier_bert`）；
  步驟 3–5 與 13 可加 `--target open_to_close` 改預測隔天開盤到收盤，結果存在 `*_oc`。分類器的訓練見 [docs/sentiment.md](docs/sentiment.md)。
- LLM 標註要在專案根目錄的 `.env`（已 gitignore）寫 `ANTHROPIC_API_KEY=...`。
- 結果在 `output/{代號}/{方法}/`，中間產物在 `data/`，兩者都不進 git。

## 共通設計

- **哪些文字算在討論這檔股票**（`ptt.py`）：標題提到代號或別名的文章與整串留言，加上本身提到的留言。
  只在內文提到的不算（大多是範本或順帶一提）；別名會撞到別家公司時，`config.yaml` 的 `exclude` 先把那些詞刪掉（例如長榮航）。
- **時間對齊**（`calendar.py`）：13:30 收盤前的貼文算當天，之後、週末、假日算下一個交易日。第 t 天的特徵只用 t 天收盤前看得到的貼文；`--target open_to_close` 時改以 09:00 切日，收到 t+1 開盤前。
- **報酬**（`prices.py`）：除權息日用參考價還原；超過漲跌幅限制、無法還原的報酬設成缺值。
- **驗證**：開發期和最終測試期分開；假設、門檻與檢定程式先 commit，最終測試只跑一次，結果不論好壞都照實記錄。
  研究一到三的假設已經用掉 2024 與 2025-01 .. 2026-09。
  研究四是新的假設，它的樣本外檢定用 2024-01 .. 2026-09，因為這個假設在這段期間沒用過；但研究三已經在同一段期間看過相近的關係，登記文件裡有寫明。
  2026-10 起的資料已經預先登記給研究三的第三次驗證、2317 盤中訊號（[PREREGISTRATION_2026.md](docs/prereg/PREREGISTRATION_2026.md)），研究四的重現（[PREREGISTRATION_LARGE_MOVES.md](docs/prereg/PREREGISTRATION_LARGE_MOVES.md)），以及開盤前情緒的重現（[PREREGISTRATION_DIRECTION.md](docs/prereg/PREREGISTRATION_DIRECTION.md)，到 2029-09）。
  研究五和 PTT 無關，評估期 A 也用 2024-01 .. 2026-09；這段的大盤走勢在登記前已經看過，登記文件裡有寫明。B 是 2026-10 .. 2029-09。

## 研究一：每日情緒 → 隔天報酬

- **每日情緒**（`features.py`）：同一帳號當天先平均再跨帳號平均，避免洗版；沒有情緒訊號的日子是缺值，不是 0。
- **情緒量法**：詞典、弱標籤分類器（[標的] 文自填的多空當標籤）、LLM 標註（Claude Opus 5.5）訓練的 TF-IDF 分類器、同一批標籤微調的 BERT。
  BERT 在測試集的 macro-F1 0.550，TF-IDF 0.495；在沒參與訓練的 2317 上是 0.519 對 0.440。細節見 [docs/sentiment.md](docs/sentiment.md)。
- **預測**（`models.py`）：預測明天漲不漲。A 組只用量價、B 組加情緒，logistic regression（C=0.01），每 21 個交易日只用過去資料重訓。
- **回測**（`backtest.py`）：預測漲就持有，扣手續費與證交稅；開盤到收盤時每天都是一次當沖。

開發期 2019–2023；cc = 收盤到收盤，oc = 隔天開盤到收盤。AUC 差是 B − A 的樣本外 AUC（bootstrap 95% CI）；IC 是預測機率和隔天報酬的 Spearman 相關。

| 股票 × 方法 × 目標 | corr(情緒, 當天) | corr(情緒, 目標) | 迴歸情緒 p | Granger 報酬→情緒 p | AUC 差 | IC（A → B） |
|---|---|---|---|---|---|---|
| 2330 詞典 cc | 0.112 | −0.012 | 0.61 | 0.003 | −0.008 [−0.021, +0.004] | +0.006 → −0.003 |
| 2330 弱標籤 cc | 0.024 | −0.021 | 0.23 | — | −0.010 [−0.027, +0.006] | +0.006 → −0.002 |
| 2330 LLM cc | 0.132 | −0.024 | 0.19 | 0.0001 | +0.001 [−0.014, +0.014] | +0.006 → +0.008 |
| 2330 LLM 合併 cc | 0.124 | −0.025 | 0.17 | 0.0004 | −0.001 [−0.019, +0.015] | +0.006 → +0.004 |
| 2330 BERT cc | **0.194** | 0.001 | 0.88 | <0.0001 | −0.014 [−0.034, +0.006] | +0.006 → −0.013 |
| 2330 詞典 oc | — | 0.017 | 0.28 | — | −0.006 [−0.023, +0.006] | +0.019 → +0.000 |
| 2330 LLM oc | — | −0.001 | 0.65 | — | −0.005 [−0.024, +0.008] | +0.019 → −0.003 |
| 2603 詞典 cc | 0.122 | 0.021 | 0.47 | <0.0001 | −0.007 [−0.023, +0.005] | +0.047 → +0.033 |
| 2603 LLM cc | 0.082 | 0.036 | 0.066 | 0.50 | +0.002 [−0.010, +0.017] | +0.047 → +0.051 |
| 2603 詞典 oc | — | 0.045 | 0.047 | 0.30 | +0.003 [−0.012, +0.017] | +0.013 → +0.014 |
| **2603 LLM oc** | — | **0.064** | **0.028** | 0.68 | +0.003 [−0.007, +0.014] | +0.013 → +0.016 |
| 2603 BERT cc | 0.108 | 0.029 | 0.39 | 0.001 | −0.007 [−0.017, +0.001] | +0.047 → +0.037 |
| 2317 LLM 合併 cc | 0.075 | −0.025 | 0.50 | 0.022 | −0.008 [−0.029, +0.011] | +0.028 → +0.026 |
| 2317 BERT cc | 0.127 | 0.024 | 0.21 | <0.0001 | +0.000 [−0.022, +0.021] | +0.028 → +0.030 |

1. **情緒在反應股價**：和當天報酬相關 0.075–0.19（弱標籤把 9 成留言判成看多，幾乎量不到東西），「報酬 → 情緒」的 Granger 檢定大多顯著。
2. **量得越準，當天相關越高，但隔天仍然沒有**：台積電從詞典 0.112 到 BERT 0.194；BERT 每天量到的帳號約多一倍，隔天相關仍在 0 附近。
3. **沒有預測力**：所有組合加入情緒後 AUC 都沒有顯著提升，回測扣成本後都輸給買進持有。
   鴻海是看結果之前就選定的新樣本，結果一致。
4. **唯一的跡象**：2603 LLM oc，低情緒日隔天盤中偏跌、放空當沖扣成本後 +7%，但它是從約 10 種組合中挑出來的。

**2024 最終測試**：

| 假設 | 開發期 | 2024 | 判定 |
|---|---|---|---|
| H1 情緒迴歸係數 > 0 | +0.0084，p = 0.014 | −0.0048，p = 0.69 | 未通過 |
| H2 低情緒日隔天盤中報酬較低 | −59.6 vs −1.4 bp，p = 0.003 | −4.8 vs −21.7 bp（相反），p = 0.73 | 未通過 |
| H3 低情緒日放空當沖扣成本後賺錢 | +7.3% | −36.0% | 未通過 |

### 預測報酬與多空

前面只預測漲不漲、只做多。這裡改成同時預測報酬和漲跌，照預測的正負號做多空：預測 > 0 收盤買進，< 0 放空（`13_return_predict.py`）。

- **模型**：特徵與滾動訓練同上。預測報酬用 ridge（懲罰在訓練窗內以時間序列交叉驗證選）與不縮減的 OLS；預測漲跌用 logit，平盤日不放進訓練。
  ridge 幾乎每次都選到懲罰上限，預測只剩歷史平均，永遠 > 0，多空等於買進持有。OLS 是看到這個結果後才加的，用來讓正負號真的會變。
- **成本**：現股（收盤到收盤放空要融券，多付 0.08% 借券費；當沖先賣後買不用）與個股期貨（手續費約 0.02% 加期交稅十萬分之二）兩種。
- **另外報的數字**：
  - 損益兩平成本：毛利 ÷ 成交金額。
  - 隨機基準：把部位隨機打亂、多空天數不變，看實際毛報酬贏過幾成的隨機部位。
  - 門檻變化：預測報酬超過成本才進場，門檻事先定好。

開發期 2019–2023（BERT 情緒，B 組）。基準在 cc 是買進持有；oc 是每天放空，因為三檔的盤中報酬平均都是負的。

| | 樣本外 R²（OLS） | CW t（OLS） | AUC 差 | 現股：基準／OLS 多空／logit 多空 | 期貨：基準／OLS 多空／logit 多空 | 損益兩平 vs 實際成本（logit） |
|---|---|---|---|---|---|---|
| 2330 cc | −4.2% | −0.27 | −0.014 [−0.036, +0.002] | +87%／−92%／−83% | +87%／−51%／−15% | 0.02% vs 0.31% |
| 2330 oc | −3.8% | −0.38 | −0.006 [−0.028, +0.008] | −98%／−99%／−98% | +5%／−48%／+6% | 0.03% vs 0.22% |
| 2603 cc | −3.7% | −0.47 | −0.008 [−0.019, +0.002] | +1287%／−33%／−56% | +1289%／+302%／+16% | 0.25% vs 0.31% |
| 2603 oc | −2.9% | −1.03 | −0.002 [−0.014, +0.009] | −95%／−99%／−99% | +116%／−61%／−58% | −0.00% vs 0.22% |
| 2317 cc | −2.6% | −0.96 | +0.007 [−0.024, +0.033] | +39%／−94%／−90% | +40%／−26%／−22% | 0.00% vs 0.31% |
| 2317 oc | −1.6% | −1.35 | −0.006 [−0.037, +0.016] | −97%／−98%／−98% | +23%／−24%／−19% | 0.02% vs 0.22% |

1. **預測不了報酬**：OLS 的樣本外 R² 全部是負的，比直接用歷史平均還差；ridge 縮到只剩歷史平均，R² 在 ±0.5% 以內。
   加入情緒後，Clark–West t 最高 1.46（ridge），AUC 差的信賴區間都含 0。
2. **照正負號多空，現股全部大賠**：12 個組合（6 組 × OLS、logit）都賠 33–99%，收盤到收盤全部輸給買進持有。
   每換手一塊錢的毛利大多不到 0.05%，實際成本是 0.22–0.31%。
   唯一超過成本的是 2603 cc 的 OLS（0.35% 對 0.31%），但長榮波動大，複利下仍然賠 33%。
3. **成本降到期貨的水準也不夠**：表中的 B 組在收盤到收盤全部輸給買進持有；盤中只有 2330 logit 小贏每天放空（+6% 對 +5%）。
4. **贏不過隨機部位**：36 個多空組合（含 A 組、ridge）中，毛報酬贏過 95% 同比例隨機部位的只有 1 個，5% 的顯著水準下本來就預期會有約 2 個。
   這個是 2317 oc 只用量價的 logit，不是 PTT 的貢獻（見[可以往哪裡做](#可以往哪裡做)第 6 點）。
5. **門檻變化**：現股成本下，ridge 的預測幾乎從來沒超過門檻；OLS 的換手少很多，但收盤到收盤仍然輸給買進持有。

2024（研究一已經用過，不是預先登記；程式在跑 2024 之前就 commit 了）的結論相同：
- 加入情緒後 CW t 最高 1.46，AUC 差的信賴區間都含 0。
- 現股多空 12 組中 11 組虧損，唯一賺錢的是 2603 cc logit（+3%），買進持有則是 +57%。

完整的表在 [docs/sentiment.md](docs/sentiment.md#預測報酬與多空)。

## 研究二：[標的] 文當作個股推薦

[標的] 文的作者照板規自己標明多空，比較像個股推薦。一篇文是一個事件，從發文後第一個收盤進場，看相對大盤的報酬；
標準誤依日期與股票雙向叢集。事件解析人工抽查 200 篇，正確 198 篇。

開發期 2016–2023：10,705 篇（看多 8,709、看空 1,996），1,467 檔。

| 看多 − 看空（市場調整） | 差 | t |
|---|---|---|
| 發文後 5 天（H1） | +0.41% | 1.99 |
| 發文後 20 天（H2） | +0.54% | 1.44 |
| 同上，排除進場收盤鎖漲跌停的 4.7% | +0.09% / +0.28% | 0.47 / 0.77 |
| 發文前 5 天 | +0.69% | 2.81 |

1. **小差距來自漲跌停**：收盤鎖漲停的看多文之後 5 天 +2.4%、鎖跌停的看空文 −6.5%，收盤鎖住時通常買不到；排除後差距接近 0。
2. **作者在追漲**：看多、看空的股票發文前 20 天都已經漲了 4–5%。
3. **跟著買會賠**：只買看多文，扣成本後 5 天 −0.83%、20 天 −1.51%；和流動性、過去報酬相近的股票比只落後 0.11%。
4. **探索性分析沒有發現**：作者過去準不準、推噓比、留言情緒、附對帳單、被刪文、有沒有新聞等 14 個特徵，FDR 校正後都不顯著。
   大盤文 656 篇，方向對錯接近隨機；有寫目標價與停損的 1,509 篇，60 天內先碰到停損約 50%、先碰到目標約 31%。

**2024/01–2025/01 最終測試**：596 篇，看空只有 57 篇，事先就知道檢定力只有約 5%。

| 假設 | 開發期 | 最終測試 | 判定 |
|---|---|---|---|
| H1 5 天 看多 − 看空 > 0 | +0.41%，單尾 p = 0.023 | −1.32%（相反），p = 0.89 | 未通過 |
| H2 20 天 看多 − 看空 > 0 | +0.54%，p = 0.075 | +1.44%，p = 0.24 | 未通過 |
| H3 發文前 20 天有差別 | −1.05%，雙尾 p = 0.048 | −2.46%，p = 0.27 | 沒有顯著差別 |

## 研究三：開盤前討論量 → 當天成交量

13:30 切日的發文量大多在反應當天盤中的量價，對隔天的量與波動沒有增量（見[可以往哪裡做](#可以往哪裡做)第 1 點）。
收盤後到隔天開盤前的討論發生在最後一筆成交之後，量價模型還沒看到。

- 目標：T 日的 log 成交量（主要）與 log 振幅；預測時點是 T 開盤前。
- 訊號：T−1 13:30 到 T 09:00 的發文加留言數取 log1p，減去前 20 個交易日同一窗口的平均（`volume.py`）。
- 基準：前幾天的量價、日曆與結算日、台指期夜盤、TSM 與費半的隔夜變動、鉅亨新聞量、前一天收漲跌停、處置期間、除權息日。
- 擴張視窗的 OLS，每 21 個交易日重估；加入討論量前後用 Clark–West 檢定比較。

開發期只控制夜盤時，台積電成交量的 MSE 降 3.96%；加入 ADR 與費半後剩 0.96%，約 3/4 是在轉述 ADR 的隔夜變動。

| CW t（台積電／長榮／鴻海） | 開發期 2019–2023 | 2024 最終測試 | 2025-01 .. 2026-09 重複驗證 |
|---|---|---|---|
| **H1** 三檔合併成交量 | 5.48 | 3.76，p = 0.0001，**通過** | 2.94，p = 0.002，**通過** |
| H2 各檔成交量（Holm） | 2.40／3.92／3.33 | 1.67／2.63／1.65，只有長榮通過 | 2.22／2.49／1.68，三檔都通過 |
| H3 三檔合併振幅 | 3.18 | 2.92，**通過** | 2.47，**通過** |
| H1 再加入開盤跳空 | 4.56 | 3.19 | 2.78 |
| 樣本外 MSE 變化 | −0.69／−1.90／−1.08% | −1.26／−2.96／−1.10% | −1.06／−1.99／−0.48% |
| 討論量 +1 SD → 成交量 | +4.5／+8.4／+5.8% | +3.6／+10.5／+4.4% | +4.0／+10.0／+3.6% |

- 加入開盤跳空（市場對所有隔夜消息的加總）後仍然顯著，所以這個訊號除了隔夜消息的大小，還有別的資訊。
- 預測的是當天會不會比較熱，不是漲跌方向，所以研究一、二的結論不變。
- 長榮的討論 2025 年以後每月只剩以前的約 1/8（2019–2024 年約 38 萬則，2025-01 到 2026-09 只有 1.3 萬則），增量仍是三檔最大。
- 重複驗證共 420 個交易日，成交量三檔共同可用 395 天。

## 研究四：大漲跌日前後的新聞與討論

研究一到三都是拿 PTT 去預測價格或成交量。這裡反過來：先找出個股單日漲跌特別大的日子，再回頭看前後的新聞與討論。

- **事件**：267 檔，超額報酬（扣掉 β × 大盤）的絕對值 ≥ 3σ 而且 ≥ 3%。
  排除除權息、處置期間與停牌後恢復；5 個交易日內又出現的大漲跌算延續，不進主要分析。
  開發期 2019-03 到 2023-11 共 3,656 個主要事件（大漲 2,671、大跌 985），控制日是同一檔前後 5 天都沒有事件的日子。
- **名稱比對**：兩字簡稱常撞名（「冠軍」「全國」「新興」「統一」）。我用鉅亨的股票標記篩出 76 檔，抽 PTT 標題人工檢查兩輪。
  正確率低於 80% 的 19 個簡稱改成只比對代號，像年份的代號（2006–2015）改成只比對簡稱。規則在算事件之前就 commit 了。
- **時間窗**：事前一週、開盤前（前一天收盤到 09:00）、盤中、隔天、事後一週。
- **回歸**：個股×年與日期固定效果，標準誤依個股、日期雙向 cluster。

**開發期 2019–2023 的主要檢定**（BH 校正，FDR 10%）：

| 檢定 | 係數 | t | 判定 |
|---|---|---|---|
| P1 開盤前 PTT 異常 → 當天大漲跌 | +0.0060 | 7.87 | **通過** |
| P2 前一週 PTT 異常 → 當天大漲跌 | −0.0024 | −5.58 | 未通過（方向相反） |
| P3 沒有鉅亨新聞的大漲跌 → 之後 5 天 | −0.50% | −1.48 | 未通過 |
| P4 同上，之後 20 天 | +0.03% | 0.06 | 未通過 |

1. **新聞與討論多半在價格動了之後才出現**：大漲跌前一週鉅亨新聞沒有變多；新聞與 PTT 從開盤前開始增加，盤中到隔天最熱。
   大漲日的當沖比例比平常高 10 個百分點，大跌日沒有變化。
2. **大部分大漲跌找不到消息**：t−1 收盤到 t 收盤之間，82.5% 沒有鉅亨頭條；加上公開資訊觀測站的重大訊息，仍有 69.6% 兩者都沒有。
   重大訊息 94% 在收盤後發布，集中在大漲跌的前一晚：大跌日 22% 前一晚有公告，平常日子 7.5%。
3. **沒消息的大漲跌沒有明顯反轉**：之後 5 天比有新聞的少 0.5%，方向和美股文獻（Chan 2003）一致，但不顯著。
   收盤鎖漲跌停的事件之後 5 天多延續 1.75%，和研究二一致。
4. **P1 不只是在轉述消息**：這是看過 P1 之後才補的分析。加入重大訊息，係數從 0.0060 降到 0.0050。
   把開盤前切在 08:30，訊號幾乎都在 08:30 之前（t = 5.95）；試撮期間 08:30–09:00 是 0.0004（t = 0.34），所以不是看到試撮價才在討論。

**預先登記的樣本外檢定 L1**：設定用補充分析之後的版本，也就是前一天收盤到 08:30 的 PTT 異常，控制鉅亨新聞、重大訊息與量價。

| | 期間 | 主要事件 | 係數 | t | 多一個標準差 | 判定 |
|---|---|---|---|---|---|---|
| 開發期（參考） | 2019-03 .. 2023-11 | 3,656 | 0.0047 | 5.95 | — | — |
| **A** | 2024-01 .. 2026-09 | 2,277 | 0.0051 | 4.61 | +0.29 個百分點（基準 1.35%） | **通過** |
| B | 2026-10 .. 2027-09 | — | — | — | — | 2027-09-30 之後跑 |

- A 的係數沒有比開發期小，t 值也和事先估計的 4.49 相符。
- L1 的設定是看過開發期之後才選的；A 這段期間研究三已經看過相近的關係（開盤前討論量 → 成交量、振幅）。
- 這個訊號預測的是會不會大漲跌，不是方向，不能拿來決定做多或放空。
- 月營收的申報時間還沒控制：公開資訊觀測站的歷史資料只有金額，沒有申報時間。

**大漲跌的原因（描述）**：用 Claude Opus 5.5 把每個事件前後一週的鉅亨新聞、重大訊息主旨與 PTT 標題分成 7 類（`18_large_moves_causes.py`，費用 9.36 美元）。
我先在看不到答案的情況下人工標 100 個事件：一致率 72%、κ = 0.63，剛好超過事先定的門檻 0.6。

| 類別（開發期主要事件 3,526 個） | 原設定 | 嚴格規則（事後） |
|---|---|---|
| 前後一週沒有鉅亨新聞，也沒有重大訊息 | 38.6% | 38.6% |
| 有消息，但找不到原因 | 17.2% | 34.2% |
| 產業題材 | 12.9% | 9.3% |
| 財報、財測、法說會 | 13.8% | 7.9% |
| 公司重大事件 | 12.2% | 7.6% |
| 月營收、法人籌碼、只描述股價、其他 | 5.4% | 2.4% |

1. **LLM 比人寬鬆**：不一致的 28 個事件中，22 個是我判斷找不到原因，LLM 卻依時間上的巧合推測了原因，例如前一晚只有「董事會通過季報」的主旨。
   把 LLM 信心 low 的都算成找不到原因（這是看過驗證之後才定的規則），一致率 82%、κ = 0.72。兩種算法下，找不到原因的都超過一半（56%、73%）。
2. **大漲跌日比較常是產業題材**：和平常有消息的日子比，產業題材 21% 對 13%，公司自己的重大事件 20% 對 30%。產業題材佔大漲的 15.5%，只佔大跌的 5.7%。
3. **原因多半在開盤前就公布了**：找得到原因的事件中，70% 的原因在開盤前公布。財報與公司事件八成以上是前一晚的公告；產業題材有三分之一在盤中才出現，法人報告有四成在收盤後，比較像事後解釋。
4. 月營收只佔 2%，可能是資料限制：月營收申報不在重大訊息裡，只有上了鉅亨頭條的才看得到。

**開盤前的情緒與漲跌方向**（[docs/large_moves_direction.md](docs/large_moves_direction.md)）：討論量只說會不會大漲跌，這裡看同一段時間的情緒能不能說方向。
前一天收盤到 08:30 提到這檔的 PTT 文字，用研究一的 BERT 分類器打分數，按帳號加權平均；有表達看法的帳號少於 3 個就不算。

| 檢定 | 樣本 | 係數 | t | 判定 |
|---|---|---|---|---|
| D1 開盤前情緒 → 大漲還是大跌（開發期） | 233 個大漲跌事件 | 0.650 | 5.79 | 通過（BH 校正） |
| D2 開盤前情緒 → 開盤到收盤的報酬，開盤前討論是平常 2.7 倍以上的日子（開發期） | 6,067 天 | 0.0047 | 2.78 | 通過（BH 校正） |
| 事後：D2 再控制開盤跳空（開發期） | 6,067 天 | 0.0048 | 2.86 | — |
| **E1 = 控制跳空的 D2，預先登記，A：2024-01 .. 2026-09** | 3,303 天、219 檔 | 0.0048 | 2.08 | **通過** |
| E1，B：2026-10 .. 2029-09 | — | — | — | 2029-09-30 之後跑 |

1. **大漲跌日的方向**：開盤前情緒偏多的大漲跌 79% 是大漲，偏空的只有 47%。但控制開盤跳空之後係數減半，大約一半是 PTT 和開盤跳空都在反應前一晚的消息；D1 不能交易，只當描述。
2. **盤中報酬**：控制開盤跳空之後，係數完全沒變，跳空本身也不會在盤中延續。所以情緒在開盤價之外還有一點資訊，樣本外的係數也和開發期一樣。
3. **賺不到錢**：情緒每多一個標準差，盤中報酬只多 0.10%。照情緒的正負號開盤買、收盤賣，評估期 A 的毛報酬每筆 0.033%（t = 0.71），扣掉現股當沖成本是 −0.40%，用期貨成本也是 −0.011%。
4. **和研究一不矛盾**：研究一看的是收盤切日的每日情緒對隔天收盤報酬，這裡只看 08:30 以前的情緒、當天盤中的報酬，而且只看討論異常熱的日子。情緒會反應前一天的漲跌（相關 0.17），這點和研究一相同，回歸裡已經控制。
5. **只適用於有人討論的股票**：大漲跌事件中，開盤前有 3 個以上帳號表達看法的只有 6%。窗口切在 08:30、討論量門檻、控制跳空都是看過開發期才定的。

## 研究五：用槓桿 ETF 打贏大盤的總報酬

研究一到四都在找 PTT 的方向訊號，扣掉成本後都賺不到錢。研究五換一個問題，和 PTT 無關：不找訊號，只放大市場曝險、再管控波動拖累，看扣掉成本後的含息總報酬能不能贏過 TAIEX 報酬指數。
只用現股與 ETF（0050、00631L），不用期貨、選擇權與融資。這是研究，不是投資建議。

- **資料**：
  - 面板的 10.5% 過濾會把 2 倍 ETF 的大跌、大漲日刪掉（00631L 有 17 天），所以 `etf.py` 從收盤與基準價重算；0050（2025-06 一拆四）與 00631L（2026-03 一拆二十二）的分割照公告還原。
  - 2015 年以前沒有 00631L，用台指期（1998 年起）合成，年成本只用 2015–2023 校準（2.55%）。
    用現貨價格指數合成會每年少算約 8%（沒有股息）。另外 2 倍 ETF 在大跌日會比 2 倍現貨跌得多，因為它持有的期貨 13:45 才結算。
- **候選策略**（看開發期結果之前寫死）：C0 只持有 0050；C1 固定 50% 00631L + 50% 0050，每月調回；
  **C2（主要假設）** 依過去 60 天的波動調整槓桿（1 到 2 倍），每週檢查；C3 在 200 日均線之上持有 00631L，之下持有 0050。
- **檢定**：每日對數超額報酬（策略 − TAIEX TR）的平均 > 0，Newey-West t，單尾 α = 0.05；C1、C3 是次要假設，用 Holm 校正。

| | 開發期 2003–2023 年化超額 | 開發期 t | 開發期最大回檔 | A：2024-01 .. 2026-09 總報酬 | A 的 t | A 判定 |
|---|---|---|---|---|---|---|
| TAIEX TR | — | — | −56% | +187% | — | — |
| C0（對照） | −0.3% | −0.48 | −56% | +255% | 2.81 | 不是假設 |
| **C2** | +4.4% | 1.40 | −67% | +280% | 1.39 | **未通過** |
| C1 | +5.1% | 1.90 | −73% | +354% | 2.09 | 通過（Holm） |
| C3 | +3.9% | 0.97 | −72% | +496% | 2.22 | 通過（Holm） |

1. **主要假設 C2 在開發期和 A 都不顯著**：突發崩盤之前通常很平靜，C2 常是滿 2 倍進場；波動拉高後才降槓桿，之後的反彈只拿到 1 倍。
2. **A 的結果證據力弱**：A 的大盤走勢在登記前已經看過，三年漲了 187%，2 倍曝險在這種多頭本來就會贏。只持有 0050 的 C0 也顯著贏，因為 0050 和 00631L 都偏重台積電。
3. **槓桿的代價**：開發期的最大回檔是 −67% 到 −73%；合成的 2 倍 ETF 在 2000 年跌 76%、2008 年跌 70%。1999–2002 只有價格指數，四個策略都輸給大盤。
4. **3 年太短**：C2 在 3 年內的檢定力只有約 13%。B（2026-10 .. 2029-09）沒通過也只能算無法判定。

## 可以往哪裡做

1. **換預測目標**：最早試的是 13:30 切日的發文量預測隔天 |報酬|，在過去波動之外沒有增量
   （樣本外 MSE 變化 2330 −0.2%、2603 +0.1%、2317 +0.4%，信賴區間都含 0；樣本內 2603 的 t = 3.2 是 2021 航運熱潮時發文多、波動大同時發生）；
   改用收盤後到開盤前的討論量才有了（研究三）。也可以試 5 日報酬或相對大盤的超額報酬。
2. **加控制變數**：情緒模型還沒加新聞量與夜盤，三大法人都還沒用；研究三還沒控制長榮的運價、鴻海的蘋果與輝達、法說會與月營收公布日。
   研究四已經加入重大訊息（多數法說會、財報、股利都在裡面），但月營收的申報時間歷史上抓不到，只能從現在開始每天抓公開資訊觀測站的當日申報（t51sb10）。
3. **降低交易頻率**：已試過週頻、只在情緒極端時進場、預測相對大盤的超額報酬，12 個事先定好的檢定在 FDR 校正後都沒通過
   （[docs/explore_directions.md](docs/explore_directions.md)）。
4. **更多股票**：掃過 301 檔，2019–2023 每交易日討論量中位數 ≥ 20 則的只有 2330、2603、2317；
   聯電、陽明、中鋼、友達只有 2021 前後夠多，其他幾乎都 ≤ 5 則，「統一」「南亞」「大成」「華電」這類簡稱又大多撞名。
   逐日的 panel 做不起來，要改成週頻，或做跨股票的事件研究。研究四就是後者：以 267 檔的大漲跌日為事件，撞名的簡稱人工檢查後只比對代號。
5. **新的樣本外資料**：研究三的第三次驗證已預先登記，目標日 2026-10-01 .. 2027-09-30，期間結束後跑一次。
   依 2025–26 年的效果，檢定力約 75%（[PREREGISTRATION_2026.md](docs/prereg/PREREGISTRATION_2026.md)）。
6. **2317 盤中的量價訊號**：只用量價的 logit 預測鴻海隔天盤中漲跌，毛報酬在開發期贏過 96% 的隨機部位，2024 是 97%；
   期貨成本下多空 +64%、+16%，同期每天放空是 +23%、−27%。這是 36 組裡挑出來的，而且和 PTT 無關。
   隨機部位檢定在做多日集中時會高估顯著性，改用 Newey–West t 檢定後，開發期只有 t = 1.40。
   已預先登記，目標日 2026-10-01 .. 2028-09-30；檢定力最多約 40%，沒通過只能算無法判定。
7. **研究四接下來**：
   - L1 評估期 B（2026-10 .. 2027-09）的檢定力：效果和開發期相同時約 86%，只有一半時約 39%。
   - 開盤前情緒 E1 的評估期 B（2026-10 .. 2029-09）的檢定力：效果相同時約 74%，只有一半時約 31%；沒通過只能算無法判定。
   - 原因分類的重大訊息只有主旨。補抓全文之後，「通過季報」這類公告就看得出好壞，這也是人和 LLM 最常不一致的地方。
8. **研究五接下來**：評估期 B（2026-10 .. 2029-09）期間結束、資料補齊後跑一次；檢定力約 13%。要真的分辨槓桿策略和大盤，需要比 3 年長很多的樣本外期間。

## 目錄

```
config.yaml           所有參數（標的、別名與排除詞、截止時間、切分、成本、LLM）
src/pttsent/
  config.py           讀 config.yaml、.env，資料路徑（PTT_DATA_DIR）
  calendar.py         交易日對齊
  prices.py           股價、除權息還原、大盤
  ptt.py              PTT 轉檔、挑出目標股票的文字
  sentiment/          lexicon / weak_labels / classifier / bert / llm_label
  features.py         每日特徵與預測目標、開發期／最終測試期切分
  analysis.py         相關、Granger、HAC 迴歸、事件研究
  models.py           滾動式預測（logit、ridge、OLS）、基準、樣本外 R²、AUC 差異的 bootstrap 信賴區間
  backtest.py         含成本回測（多空、當沖、融券與期貨成本、門檻、損益兩平、隨機基準）
  plots.py            圖表
  panel.py            全市場日價面板（上市＋上櫃、還原、漲跌停、大盤）
  events.py           [標的] 文合併、股票代號解析、進場時間對齊
  event_study.py      異常報酬、雙向叢集標準誤、日曆時間組合
  volume.py           開盤前討論量、夜盤與美股對齊、滾動 OLS、Clark–West 檢定
  explore.py          週頻資料、情緒極端日、極端日報酬差檢定、對沖後的超額報酬
  large_moves.py      研究四：名稱比對、超額報酬與大漲跌日、時間窗則數、固定效果＋雙向 cluster 迴歸
  mops.py             公開資訊觀測站重大訊息的解析
  causes.py           研究四：大漲跌原因分類的 LLM 輸入、提示詞與輸出格式
  direction.py        研究四：開盤前的 PTT 文字與每日情緒（漲跌方向）
  etf.py              研究五：ETF 報酬（不過濾、分割還原）、台指期連續近月、合成 2 倍、指數與央行利率
  leverage.py         研究五：候選策略 C0–C3、評估指標、block bootstrap、Holm 校正
scripts/              01–06、13、14 研究一；07–10、fetch_prices.py、build_panel.py 研究二；11、12、fetch_us.py 研究三；15 2026-10 起的樣本外檢定；
                      16–20、fetch_mops.py 研究四；21、fetch_long_history.py 研究五；
                      train_classifier.py、llm_label.py 訓練情緒分類器
tests/                時間對齊、不偷看未來、成本等不變量
docs/
  sentiment.md        研究一的細節（文字歸屬、LLM 標註、分類器與 BERT 的準確度、回測與補充結果）
  explore_directions.md  週頻、情緒極端、超額報酬的探索計畫與結果
  large_moves.md      研究四的計畫、名稱檢查、結果、補充分析與原因分類；large_moves/ 是名稱檢查表、比對規則與原因分類的人工標註
  large_moves_direction.md  研究四延伸：開盤前情緒與漲跌方向的探索計畫、結果與跳空檢查
  report_outline.md   專題報告大綱
  leverage/           研究五：ETF 分割表（附公告來源）
  prereg/             八份預先登記與結果：研究一、研究二、研究三、研究三的重複驗證、2026-10 起的樣本外檢定、研究四、研究四的開盤前情緒、研究五
```

---

# PTT Sentiment and Taiwan Stocks

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
| 5. Leveraged ETFs (not about PTT) | Can scaling up market exposure with 0050 and 00631L beat the market's total return after costs? | **The pre-specified primary rule (volatility targeting) did not win significantly**: +280% total return in 2024-01 to 2026-09 against +187% for the market, but t = 1.39. A fixed 1.5x mix and a 200-day moving-average rule passed, but the market's path in that period was already known to be a strong bull run, and holding 0050 alone also won significantly. Development-period maximum drawdowns were −67% to −73% | [PREREGISTRATION_LEVERAGE.md](docs/prereg/PREREGISTRATION_LEVERAGE.md) (Chinese only) |

PTT reflects prices that have already moved, but pre-open discussion does signal whether the day will be busier than usual and whether a large move is coming; pre-open sentiment also hints at the intraday direction, though too weakly to make money.
Study 5 asks a different question: no signal, just more market exposure. Three years of evaluation is too short for the primary rule to win significantly, while the drawdowns from leverage showed up as expected.
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

# Study 5: leveraged ETFs (PREREGISTRATION_LEVERAGE.md)
python scripts/fetch_long_history.py      # TX futures from 1998, total return index from 2003, price index from 1999, central bank rate; about 35 minutes
python scripts/21_leverage.py dev         # C0-C3 over the development period 2003-2023 (overwrites output/leverage/dev_meta.json)
python scripts/21_leverage.py power       # power (development-period t values only)
python scripts/21_leverage.py final A     # out-of-sample test, 2024-01 to 2026-09 (run once already; do not rerun); check A comes first
python scripts/21_leverage.py final B     # 2026-10 to 2029-09: run once after the period ends and the data is complete
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
  Study 5 does not use PTT; its period A is also 2024-01 to 2026-09. The market's path over that period was already seen before registration, and the preregistration says so. Period B is 2026-10 to 2029-09.

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

## Study 5: beating the market's total return with leveraged ETFs

Studies 1-4 look for a directional signal in PTT, and none of it makes money after costs. Study 5 asks a different question that has nothing to do with PTT: with no signal at all, can scaling up market exposure, while managing volatility drag, beat the TAIEX total return index after costs?
Only stocks and ETFs (0050, 00631L); no futures, options or margin. This is research, not investment advice.

- Data:
  - The panel's 10.5% filter drops the largest up and down days of 2x ETFs (17 days for 00631L), so `etf.py` recomputes returns from closes and base prices; the 0050 split (1-for-4, 2025-06) and the 00631L split (1-for-22, 2026-03) are restored from the announcements.
  - Before 2015 there is no 00631L, so it is synthesized from TX futures (from 1998), with its yearly cost calibrated on 2015-2023 only (2.55%).
    Synthesizing from the spot price index would miss about 8% a year (no dividends). A 2x ETF also falls more than twice the spot index on crash days, because the futures it holds settle at 13:45.
- Candidates, fixed before seeing the development results: C0 holds 0050 only; C1 holds a fixed 50% 00631L + 50% 0050, rebalanced monthly;
  **C2 (primary)** sets leverage between 1x and 2x from the past 60 days' volatility, checked weekly; C3 holds 00631L above the 200-day moving average and 0050 below it.
- Test: mean daily log excess return (strategy − TAIEX TR) > 0, Newey-West t, one-sided α = 0.05; C1 and C3 are secondary, with Holm correction.

| | Dev 2003-2023 annual excess | Dev t | Dev max drawdown | A: 2024-01 to 2026-09 total return | A t | A verdict |
|---|---|---|---|---|---|---|
| TAIEX TR | — | — | −56% | +187% | — | — |
| C0 (reference) | −0.3% | −0.48 | −56% | +255% | 2.81 | not a hypothesis |
| **C2** | +4.4% | 1.40 | −67% | +280% | 1.39 | **failed** |
| C1 | +5.1% | 1.90 | −73% | +354% | 2.09 | passed (Holm) |
| C3 | +3.9% | 0.97 | −72% | +496% | 2.22 | passed (Holm) |

1. The primary rule C2 was not significant in either the development period or A. Sudden crashes usually follow calm stretches, so C2 often goes in at the full 2x; it only cuts leverage after volatility jumps, and then catches the rebound at 1x.
2. A is weak evidence. The market's path in A was seen before registration, and it rose 187% in three years; 2x exposure wins in a bull run like that anyway. C0, holding only 0050, also won significantly, because 0050 and 00631L both lean heavily on TSMC.
3. The cost of leverage: development-period maximum drawdowns were −67% to −73%; the synthetic 2x ETF lost 76% in 2000 and 70% in 2008. In 1999-2002, with only the price index, all four strategies lost to the market.
4. Three years is too short: C2's power over three years is only about 13%, so a failure in B (2026-10 to 2029-09) would also count as inconclusive.

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
8. Study 5 from here: period B (2026-10 to 2029-09) runs once after it ends and the data is complete; power is about 13%. Telling a leveraged strategy apart from the market would need an out-of-sample period much longer than three years.

## Layout

```
config.yaml           all parameters (tickers, aliases and exclusions, cutoff times, splits, costs, LLM)
src/pttsent/
  config.py           loads config.yaml and .env, data paths (PTT_DATA_DIR)
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
  etf.py              study 5: ETF returns (no limit filter, split adjustment), continuous TX front month, synthetic 2x, indexes and central bank rate
  leverage.py         study 5: candidate strategies C0-C3, metrics, block bootstrap, Holm correction
scripts/              01–06, 13, 14 study 1; 07–10, fetch_prices.py, build_panel.py study 2; 11, 12, fetch_us.py study 3; 15 out-of-sample tests from 2026-10;
                      16-20, fetch_mops.py study 4; 21, fetch_long_history.py study 5;
                      train_classifier.py, llm_label.py train the sentiment classifiers
tests/                invariants: time alignment, no look-ahead, costs
docs/                 Chinese originals, with English translations in *.en.md
  sentiment.md        study 1 details (text attribution, LLM labelling, classifier and BERT accuracy, backtests and extra results)
  explore_directions.md  plan and results for weekly, extreme-sentiment and excess-return exploration
  large_moves.md      study 4 plan, name check, results, supplementary analysis and cause classification (Chinese only for now); large_moves/ holds the name check sheets, matching rules and hand labels for the cause classification
  large_moves_direction.md  study 4 extension: plan, results and opening-gap check for pre-open sentiment and direction (Chinese only for now)
  report_outline.md   capstone report outline (Chinese only)
  leverage/           study 5: ETF split table (with announcement sources)
  prereg/             eight preregistrations with results: studies 1, 2, 3, the study 3 replication, the out-of-sample tests from 2026-10, study 4, study 4's pre-open sentiment, and study 5
```
