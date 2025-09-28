import os
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import httpx
import time
import numpy as np

# 設定からAPIキー/モデルを取得（存在しない場合は環境変数を使用）
try:
    from config import LLM_OPTIONS  # type: ignore
except Exception:
    LLM_OPTIONS = None  # type: ignore


# Optional LangChain backend (only used if available)
try:
    # langchain_openai is the modern provider package; classic `langchain` may not be installed.
    from langchain_openai import ChatOpenAI  # type: ignore
    _LANGCHAIN_AVAILABLE = True
except Exception:
    ChatOpenAI = None  # type: ignore
    _LANGCHAIN_AVAILABLE = False


@dataclass
class InvestmentDecision:
    """
    投資判断の構造化結果
    """
    action: str  # "BUY" | "SELL" | "HOLD"
    conviction: float  # 0-100
    reasoning: str
    risks: List[str] = field(default_factory=list)
    timeframe: Optional[str] = None  # 例: "短期（1-2週間）"
    targets: Dict[str, Union[float, str]] = field(default_factory=dict)  # 例: {"entry": 1234.5, "tp": 1300, "sl": 1180}
    category_value: Optional[str] = None  # LLMが返す5段階カテゴリ（英語コード or 日本語）

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "InvestmentDecision":
        def _norm_action(a: str) -> str:
            a_up = (a or "").strip().upper()
            if a_up in ("BUY", "LONG"):
                return "BUY"
            if a_up in ("SELL", "SHORT"):
                return "SELL"
            return "HOLD"

        def _norm_category(cat: Optional[str]) -> Optional[str]:
            if not cat:
                return None
            s = str(cat).strip().upper()
            # 日本語対応
            ja_map = {
                "強い買い": "STRONG_BUY",
                "買い": "BUY",
                "ホールド": "HOLD",
                "様子見": "HOLD",
                "売り": "SELL",
                "強い売り": "STRONG_SELL",
            }
            if s in ("STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"):
                return s
            if cat in ja_map:
                return ja_map[cat]
            # 英語の表記揺れ
            s2 = s.replace(" ", "_")
            if s2 in ("STRONG_BUY", "STRONG_SELL"):
                return s2
            return None

        return InvestmentDecision(
            action=_norm_action(d.get("action", "")),
            conviction=float(d.get("conviction", 0)),
            reasoning=str(d.get("reasoning", "")),
            risks=list(d.get("risks", [])) if isinstance(d.get("risks", []), list) else [],
            timeframe=d.get("timeframe"),
            targets=d.get("targets", {}) if isinstance(d.get("targets", {}), dict) else {},
            category_value=_norm_category(d.get("category")),
        )

    def category(self) -> str:
        """
        5段階のカテゴリを返す: STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL
        LLMがcategoryを返している場合はそれを優先し、無ければaction/convictionから導出
        """
        if self.category_value:
            return self.category_value
        a = (self.action or "").upper()
        c = float(self.conviction or 0)
        if a == "BUY":
            if c >= 80:
                return "STRONG_BUY"
            elif c >= 55:
                return "BUY"
            else:
                return "HOLD"
        if a == "SELL":
            if c >= 80:
                return "STRONG_SELL"
            elif c >= 55:
                return "SELL"
            else:
                return "HOLD"
        return "HOLD"

    def category_ja(self) -> str:
        """
        カテゴリの日本語表記を返す（LLMのcategoryを優先）
        """
        mapping = {
            "STRONG_BUY": "強い買い",
            "BUY": "買い",
            "HOLD": "ホールド",
            "SELL": "売り",
            "STRONG_SELL": "強い売り",
        }
        return mapping.get(self.category(), "ホールド")


def _pick_close_column(df: pd.DataFrame) -> Optional[Union[str, Tuple]]:
    """
    DataFrameから終値に相当するカラム名を推定して返す
    - 優先度: Adj Close > Close
    - MultiIndex列にも対応（例: ("Adj Close", "7203.T") や ("Close", "7203.T")）
    """
    # 単一列名（優先: Adj Close）
    for name in ["Adj Close", "Close", "adj_close", "close", "AdjClose"]:
        if name in df.columns:
            return name

    # MultiIndex（優先: Adj Close）
    if isinstance(df.columns, pd.MultiIndex):
        # まずAdj Closeを探す
        for col in df.columns:
            if isinstance(col, tuple) and any(str(level) == "Adj Close" for level in col):
                return col
        # 次にCloseを探す
        for col in df.columns:
            if isinstance(col, tuple) and any(str(level) == "Close" for level in col):
                return col

    return None

def _to_series_1d(obj: Union[pd.Series, pd.DataFrame, Any]) -> pd.Series:
    """
    可能なら1次元のSeriesに変換する。DataFrameなら先頭列を選択。
    """
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            return pd.Series(dtype=float)
        return obj.iloc[:, 0]
    if isinstance(obj, pd.Series):
        return obj
    # その他はSeriesにラップ
    return pd.Series([obj])


def _as_float(x: Any) -> float:
    """
    単一要素のSeriesやnumpyスカラーを安全にfloatへ変換（FutureWarning回避）
    """
    # 単一要素Series
    if isinstance(x, pd.Series):
        if len(x) == 0:
            return float("nan")
        x = x.iloc[0]
    # numpyスカラー or 0次元配列
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    return float(x)


def _as_int(x: Any) -> int:
    """
    単一要素のSeriesやnumpyスカラーを安全にintへ変換（FutureWarning回避）
    """
    # 単一要素Series
    if isinstance(x, pd.Series):
        if len(x) == 0:
            return 0
        x = x.iloc[0]
    # numpyスカラー or 0次元配列
    if hasattr(x, "item"):
        try:
            return int(x.item())
        except Exception:
            pass
    return int(x)



def _summarize_frame(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    """
    1つの時間軸DataFrameについて、LLMに渡すのに十分な要約統計を作成
    """
    if df is None or df.empty:
        return {"label": label, "empty": True}

    close_col = _pick_close_column(df)
    if close_col is None:
        return {"label": label, "empty": True}

    series = _to_series_1d(df[close_col]).dropna()
    if series.empty:
        return {"label": label, "empty": True}

    last = _as_float(series.iloc[-1])
    prev = _as_float(series.iloc[-2]) if len(series) >= 2 else last
    chg = last - prev
    chg_pct = (chg / prev * 100.0) if prev != 0 else 0.0

    def sma(n: int) -> Optional[float]:
        if len(series) >= n:
            return _as_float(series.tail(n).mean())
        return None

    stats = {
        "label": label,
        "empty": False,
        "last": round(last, 6),
        "prev": round(prev, 6),
        "change": round(chg, 6),
        "change_pct": round(chg_pct, 4),
        "sma_5": round(sma(5), 6) if sma(5) is not None else None,
        "sma_20": round(sma(20), 6) if sma(20) is not None else None,
        "sma_50": round(sma(50), 6) if sma(50) is not None else None,
        "min_last_20": round(_as_float(series.tail(20).min()), 6) if len(series) >= 20 else None,
        "max_last_20": round(_as_float(series.tail(20).max()), 6) if len(series) >= 20 else None,
    }

    # Volumeの概要（単一列名とMultiIndexの両対応）
    vol_series = None

    # 1) 単一列名
    for name in ["Volume", "volume"]:
        if name in df.columns:
            vol_series = _to_series_1d(df[name]).dropna()
            break

    # 2) MultiIndex（例: ("Volume", "7203.T")）
    if vol_series is None and isinstance(df.columns, pd.MultiIndex):
        # Volumeを含むタプルを優先的に選択
        candidate = None
        for col in df.columns:
            if isinstance(col, tuple) and any(str(level) == "Volume" for level in col):
                candidate = col
                break
        if candidate is not None:
            vol_series = _to_series_1d(df[candidate]).dropna()

    if vol_series is not None and not vol_series.empty:
        stats["volume_last"] = _as_int(vol_series.iloc[-1])
        stats["volume_avg5"] = int(round(_as_float(vol_series.tail(5).mean()))) if len(vol_series) >= 5 else None
        stats["volume_avg20"] = int(round(_as_float(vol_series.tail(20).mean()))) if len(vol_series) >= 20 else None

    return stats


def summarize_market_data(multi_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    stock_data.get_multi_timeframe_data() が返すような辞書を要約する
    """
    return {
        "daily": _summarize_frame(multi_data.get("daily", pd.DataFrame()), "daily"),
        "hourly": _summarize_frame(multi_data.get("hourly", pd.DataFrame()), "hourly"),
        "minute": _summarize_frame(multi_data.get("minute", pd.DataFrame()), "minute"),
    }


def _flatten_columns_to_str(df: pd.DataFrame) -> pd.DataFrame:
    """
    列名を文字列にフラット化（MultiIndex対応）
    """
    df2 = df.copy()
    if isinstance(df2.columns, pd.MultiIndex):
        df2.columns = [" | ".join([str(x) for x in col]) for col in df2.columns.to_list()]
    else:
        df2.columns = [str(c) for c in df2.columns]
    return df2


def _df_to_records(df: pd.DataFrame) -> Dict[str, Any]:
    """
    DataFrameをJSONレコード配列に変換（日時はISO文字列）
    """
    if df is None or df.empty:
        return {"columns": [], "records": [], "count": 0}

    df2 = _flatten_columns_to_str(df).reset_index()
    # pandasのto_jsonでISO文字列に変換し、それをPythonオブジェクトへ戻す
    try:
        records = json.loads(df2.to_json(orient="records", date_format="iso", force_ascii=False))
    except Exception:
        # フォールバック: 非シリアライズ型を文字列化
        records = []
        for _, row in df2.iterrows():
            obj = {}
            for k, v in row.to_dict().items():
                try:
                    json.dumps(v)  # そのままいけるか確認
                    obj[str(k)] = v
                except Exception:
                    obj[str(k)] = str(v)
            records.append(obj)
    return {"columns": list(df2.columns), "records": records, "count": len(records)}


def raw_market_data(multi_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    stock_data.get_multi_timeframe_data() が返す各時間軸の DataFrame を生データとしてJSONレコード化
    """
    return {
        "daily": _df_to_records(multi_data.get("daily", pd.DataFrame())),
        "hourly": _df_to_records(multi_data.get("hourly", pd.DataFrame())),
        "minute": _df_to_records(multi_data.get("minute", pd.DataFrame())),
    }


def _build_prompt(ticker: str, data_payload: Dict[str, Any], objective: Optional[str] = None) -> str:
    """
    LLMに投げるプロンプトを構築（JSON出力を指示）
    """
    instructions = f"""
あなたはプロのトレーディングアシスタントです。銘柄 {ticker} のマルチタイムフレームの生データ（各行をJSONレコード）に基づいて、
5段階（強い買い/買い/ホールド/売り/強い売り）の投資判断を日本語の理由付きで行い、JSONのみで出力してください。

要件:
- JSONのみを返し、余計なテキストは一切含めないこと
- フィールド:
  - category ∈ ["STRONG_BUY","BUY","HOLD","SELL","STRONG_SELL"]  ← 5段階のいずれか（この値を必ず返すこと）
  - action ∈ ["BUY","SELL","HOLD"]  ← categoryと整合する値
  - conviction (0-100)
  - reasoning (日本語)
  - risks (配列)
  - timeframe (任意)
  - targets (entry/tp/slなど任意)
- 短期〜中期のテクニカル観点（トレンド、平均線、出来高）を簡潔に考慮
- リスクも2-4点程度で簡潔に
- {('目的: ' + objective) if objective else ''}

入力データ（生）:
{json.dumps(data_payload, ensure_ascii=False, indent=2)}
"""
    return instructions.strip()


def _parse_llm_json(text: str) -> Dict[str, Any]:
    """
    LLM応答からJSONを抽出・解析
    """
    # 1) そのままJSONとして試す
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) ```json ... ``` ブロック抽出
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 3) 最初の { … } を抽出
    m2 = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            pass
    # 失敗時はHOLDデフォルト
    return {
        "action": "HOLD",
        "conviction": 0,
        "reasoning": "モデル応答の解析に失敗しました。",
        "risks": [],
        "timeframe": None,
        "targets": {},
    }


def _call_openai_chat(prompt: str, model: Optional[str] = None, temperature: float = 0.2, timeout: int = 30) -> Dict[str, Any]:
    """
    OpenAI Chat Completions APIを直接呼び出してJSONを得る（httpxベース）
    - APIキーとモデルは config.LLM_OPTIONS → 環境変数 の順で参照
    """
    # config からの取得を優先し、未設定の場合は環境変数へフォールバック
    cfg_key = ""
    cfg_model = ""
    try:
        if LLM_OPTIONS is not None:
            cfg_key = getattr(LLM_OPTIONS, "openai_api_key", "") or ""
            cfg_model = getattr(LLM_OPTIONS, "model", "") or ""
    except Exception:
        pass

    api_key = (cfg_key.strip() if cfg_key else "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OpenAI APIキーが設定されていません。config.py の LLM_OPTIONS.openai_api_key か環境変数 OPENAI_API_KEY を設定してください。")

    used_model = (
        model
        or (cfg_model.strip() if cfg_model else "")
        or os.getenv("LLM_MODEL", "").strip()
        or "gpt-4o-mini"
    )
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": used_model,
        "messages": [
            {"role": "system", "content": "You are an expert trading assistant. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        # JSON出力強制（対応モデルのみ）
        "response_format": {"type": "json_object"},
    }

    # 明示的なタイムアウト設定（接続/読み取り/書き込み）
    timeout_cfg = httpx.Timeout(connect=10.0, read=float(timeout), write=10.0, pool=float(timeout))
    max_retries = 2
    backoff = 1.5
    fallback_model = "gpt-4o-mini"
    tried_fallback = False

    # HTTP/2は依存関係(h2)が必要なため無効化し、HTTP/1.1で送信
    with httpx.Client(timeout=timeout_cfg) as client:
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                resp = client.post(url, headers=headers, json=payload)

                # 400時はエラーメッセージを取得し、モデル/フォーマット関連ならフォールバック
                if resp.status_code == 400:
                    err_text = ""
                    try:
                        err_json = resp.json()
                        err_text = err_json.get("error", {}).get("message", "") or resp.text
                    except Exception:
                        err_text = resp.text

                    lower = err_text.lower()
                    model_issue = ("model" in lower and ("exist" in lower or "not found" in lower))
                    format_issue = "response_format" in lower

                    if (model_issue or format_issue) and not tried_fallback:
                        # フォールバックモデルで再試行し、response_formatも削除して互換性確保
                        payload["model"] = fallback_model
                        if "response_format" in payload:
                            del payload["response_format"]
                        tried_fallback = True
                        continue  # 即リトライ

                    raise httpx.HTTPStatusError(f"400 Bad Request: {err_text}", request=resp.request, response=resp)

                resp.raise_for_status()
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return _parse_llm_json(content)
            except (httpx.TimeoutException, httpx.HTTPError) as e:
                last_exc = e
                if attempt >= max_retries:
                    raise
                time.sleep(backoff * (attempt + 1))
        # 到達しない想定
        if last_exc:
            raise last_exc


def _call_langchain(prompt: str, model: Optional[str] = None, temperature: float = 0.2) -> Dict[str, Any]:
    """
    LangChain経由でOpenAIチャットモデルを呼び出す（インストール時のみ）
    - モデル/キーは config.LLM_OPTIONS → 環境変数 の順に参照
    """
    if not _LANGCHAIN_AVAILABLE or ChatOpenAI is None:
        raise RuntimeError("LangChainバックエンドは利用できません（langchain_openai が見つかりません）。")

    cfg_key = ""
    cfg_model = ""
    try:
        if LLM_OPTIONS is not None:
            cfg_key = getattr(LLM_OPTIONS, "openai_api_key", "") or ""
            cfg_model = getattr(LLM_OPTIONS, "model", "") or ""
    except Exception:
        pass

    used_model = (
        model
        or (cfg_model.strip() if cfg_model else "")
        or os.getenv("LLM_MODEL", "").strip()
        or "gpt-4o-mini"
    )
    api_key = (cfg_key.strip() if cfg_key else "") or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OpenAI APIキーが設定されていません。config.py の LLM_OPTIONS.openai_api_key か環境変数 OPENAI_API_KEY を設定してください。")

    # ChatOpenAIはapi_key引数を受け付けます（新しいlangchain_openaiパッケージ）
    llm = ChatOpenAI(model=used_model, temperature=temperature, api_key=api_key)  # type: ignore
    msg = llm.invoke(_build_system_user_messages(prompt))
    return _parse_llm_json(msg.content)


def _build_system_user_messages(user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are an expert trading assistant. Output JSON only."},
        {"role": "user", "content": user_prompt},
    ]


def advise_from_data(
    ticker: str,
    multi_data: Dict[str, pd.DataFrame],
    *,
    objective: Optional[str] = None,
    prefer_backend: Optional[str] = None,  # "langchain" | "openai"
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> InvestmentDecision:
    """
    取得済みの株価データ（daily/hourly/minuteのDataFrame辞書）をLLMへ渡し、投資判断を受け取る。

    Args:
        ticker: 銘柄コードやティッカー（例: "7203.T"）
        multi_data: get_multi_timeframe_data等で取得した辞書
        objective: 追加の目的（例: "短期のデイトレ観点で"）
        prefer_backend: "langchain"（インストール時のみ）または "openai"。未指定なら自動判定。
        model: LLMモデル名（未指定なら環境変数 LLM_MODEL またはデフォルトを使用）
        temperature: 生成温度

    Returns:
        InvestmentDecision: 構造化された投資判断
    """
    data_payload = raw_market_data(multi_data)
    prompt = _build_prompt(ticker, data_payload, objective)

    backend = (prefer_backend or "").lower().strip()
    use_langchain = backend == "langchain"

    if use_langchain:
        result = _call_langchain(prompt, model=model, temperature=temperature)
    else:
        # 明示指定がなければOpenAI API直呼びを優先（依存を増やさないため）
        result = _call_openai_chat(prompt, model=model, temperature=temperature)

    return InvestmentDecision.from_dict(result)


# 簡易CLIテスト用（任意実行）
if __name__ == "__main__":
    print("llm_strategy quick self-check: module loaded.")