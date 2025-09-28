import yfinance as yf
import pandas as pd


def get_multi_timeframe_data(
    ticker: str,
    daily_period: str = "30d",
    daily_count: int = 20,
    hourly_period: str = "5d",
    hourly_count: int = 48,
    minute_period: str = "2d",
    minute_count: int = 60,
    hourly_interval: str = "1h",
    minute_interval: str = "5m",
) -> dict:
    """
    指定された銘柄の長期・中期・短期の株価データを取得する関数。

    Args:
        ticker (str): 銘柄のティッカーシンボル。
        daily_period (str): 日足データの取得期間（例: '30d'）。
        daily_count (int): 日足で返す最新本数（tailに使用）。
        hourly_period (str): 1時間足データの取得期間（例: '5d'）。
        hourly_count (int): 1時間足で返す最新本数。
        minute_period (str): 5分足データの取得期間（例: '2d'）。
        minute_count (int): 5分足で返す最新本数。
        hourly_interval (str): 1時間足のinterval（例: '1h'）。
        minute_interval (str): 5分足のinterval（例: '5m'）。

    Returns:
        dict: 各時間軸のDataFrameを格納した辞書。
              {'daily': df_daily, 'hourly': df_hourly, 'minute': df_minute}
    """
    data_frames = {}

    print(f"[{ticker}] 複数時間軸のデータを取得中...")

    # 1. 長期足（日足）
    # yfinanceでは週末等を含むため、少し多めに期間を指定してtail()で末尾を取得するのが確実
    df_daily = yf.download(
        ticker,
        period=daily_period,  # 日足の取得期間
        interval="1d",  # 日足
        progress=False,
        auto_adjust=False
    )
    if not df_daily.empty:
        df_daily.columns = df_daily.columns.droplevel(1)
        data_frames['daily'] = df_daily.tail(daily_count)
    else:
        print(f"警告: [{ticker}] の日足データを取得できませんでした。")
        data_frames['daily'] = pd.DataFrame()

    # 2. 中期足（1時間足）
    df_hourly = yf.download(
        ticker,
        period=hourly_period,  # 1時間足の取得期間
        interval=hourly_interval,  # 1時間足のinterval
        progress=False,
        auto_adjust=False
    )
    if not df_hourly.empty:
        data_frames['hourly'] = df_hourly.tail(hourly_count)
    else:
        print(f"警告: [{ticker}] の1時間足データを取得できませんでした。")
        data_frames['hourly'] = pd.DataFrame()

    # 3. 短期足（5分足）
    df_minute = yf.download(
        ticker,
        period=minute_period,  # 5分足の取得期間
        interval=minute_interval,  # 5分足のinterval
        progress=False,
        auto_adjust=False
    )
    if not df_minute.empty:
        data_frames['minute'] = df_minute.tail(minute_count)
    else:
        print(f"警告: [{ticker}] の5分足データを取得できませんでした。")
        data_frames['minute'] = pd.DataFrame()

    print(f"[{ticker}] データ取得完了。")
    return data_frames


# --- このファイル単体でテストするためのコード ---
if __name__ == '__main__':
    # トヨタ自動車のデータを取得してみる
    toyota_ticker = '7203.T'
    multi_data = get_multi_timeframe_data(toyota_ticker)

    if multi_data:
        print("\n--- 取得データサンプル ---")
        for timeframe, df in multi_data.items():
            print(f"\n▼ {timeframe} データ (取得本数: {len(df)})")
            if not df.empty:
                print(df.head(3))  # 各データの先頭3行を表示
            else:
                print("データなし")