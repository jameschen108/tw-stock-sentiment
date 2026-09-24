"""圖表（存成 PNG）。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang TC", "Arial Unicode MS",
                                   "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def price_vs_sentiment(df, path, title):
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.plot(df.index, df["close"], color="#333", lw=1, label="收盤價")
    ax1.set_ylabel("收盤價")
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["sent_mean"].rolling(20, min_periods=5).mean(),
             color="#d62728", lw=1, label="情緒（20 日平均）")
    ax2.axhline(0, color="#d62728", lw=0.5, ls=":")
    ax2.set_ylabel("情緒")
    ax1.set_title(title)
    fig.legend(loc="upper left", bbox_to_anchor=(0.07, 0.9))
    _save(fig, path)


def cross_correlation(xc, path, title):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(xc["lag"], xc["corr"], color=["#1f77b4" if k > 0 else "#aaa" for k in xc["lag"]])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("k（>0：情緒領先報酬 k 天；<0：報酬領先情緒）")
    ax.set_ylabel("corr(情緒_t, 報酬_{t+k})")
    ax.set_title(title)
    _save(fig, path)


def event_car(ev, path, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    for g, color in [("high", "#d62728"), ("low", "#2ca02c")]:
        e = ev[ev["group"] == g]
        if e.empty:
            continue
        label = f"{'情緒最高' if g == 'high' else '情緒最低'} 10%（{e['n_events'].iloc[0]} 次）"
        ax.plot(e["offset"], e["car"] * 100, color=color, marker="o", ms=3, label=label)
        ax.fill_between(e["offset"], (e["car"] - 2 * e["se"]) * 100,
                        (e["car"] + 2 * e["se"]) * 100, color=color, alpha=0.12)
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("相對事件日（0 = 情緒極端當天，當天報酬與情緒同時發生）")
    ax.set_ylabel("累積超額報酬（%）")
    ax.set_title(title)
    ax.legend()
    _save(fig, path)


def equity_curves(curves: dict, path, title):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, eq in curves.items():
        ax.plot(eq.index, eq, lw=1, label=name)
    ax.set_ylabel("淨值（起始 = 1，已扣成本）")
    ax.set_title(title)
    ax.legend()
    _save(fig, path)
