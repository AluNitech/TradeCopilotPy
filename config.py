from dataclasses import dataclass


@dataclass
class DataHandlerOptions:
    """
    get_multi_timeframe_data に渡すオプション。
    値を変更して運用環境ごとに調整してください。
    """
    daily_period: str = "30d"
    daily_count: int = 20
    hourly_period: str = "5d"
    hourly_count: int = 48
    minute_period: str = "2d"
    minute_count: int = 60
    hourly_interval: str = "1h"
    minute_interval: str = "5m"

@dataclass
class LLMOptions:
    """
    LLM関連の設定。APIキーやデフォルトモデルをここに設定できます。
    - openai_api_key: OpenAI APIキー（環境変数が未設定の場合に使用）
    - model: 既定のモデル名（未指定なら環境変数 LLM_MODEL、さらに未設定ならコード側のデフォルト）
    """
    openai_api_key: str = ""  # 例: "sk-xxxxxxxx..."
    model: str = "gpt-4o"           # 例: "gpt-4o" など


# アプリが参照するオプションのインスタンス
DATA_HANDLER_OPTIONS = DataHandlerOptions()
LLM_OPTIONS = LLMOptions()