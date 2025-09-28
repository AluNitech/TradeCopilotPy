import httpx
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import re
from dataclasses import dataclass
from decimal import Decimal


@dataclass
class RankingStock:
    """デイトレ適性ランキングの株式データを表現するデータクラス"""
    rank: int
    code: str
    name: str
    market: str
    current_price: float
    price_change: float
    price_change_percent: float
    volume: int
    trading_value: int  # 概算売買代金
    volatility_percent: float  # 株価変動率
    
    def __post_init__(self):
        """データの後処理・バリデーション"""
        if self.rank <= 0:
            raise ValueError("順位は1以上である必要があります")
        if not self.code or not re.match(r'^\d{3,4}[A-Z]?$', self.code):
            raise ValueError(f"証券コードの形式が正しくありません: {self.code}")
        if not self.name:
            raise ValueError("銘柄名は空であってはなりません")


def parse_number(value: str) -> float:
    """
    文字列から数値を抽出する
    
    Args:
        value: 解析対象の文字列
        
    Returns:
        float: 抽出された数値
    """
    if not value:
        return 0.0
    
    # カンマと全角数字を除去
    cleaned = re.sub(r'[,，\s]', '', value)
    # 数字と小数点、マイナス記号のみを抽出
    match = re.search(r'-?\d+(?:\.\d+)?', cleaned)
    return float(match.group()) if match else 0.0


def parse_percentage(value: str) -> float:
    """
    パーセント文字列から数値を抽出
    
    Args:
        value: "%"を含むパーセント文字列
        
    Returns:
        float: パーセント値（例："+5.5%" → 5.5）
    """
    if not value:
        return 0.0
    
    # %記号を除去して数値を抽出
    cleaned = value.replace('%', '').replace('％', '')
    return parse_number(cleaned)


def extract_stock_info_from_cell(cell_text: str) -> tuple[str, str, str]:
    """
    銘柄セルから銘柄名、証券コード、市場を抽出
    
    Args:
        cell_text: 銘柄セルのテキスト（例："レーザーテック 6920 東P" や "オリオンビール409A 東P"）
        
    Returns:
        tuple: (銘柄名, 証券コード, 市場)
    """
    # 証券コード（3-4桁数字 + 任意のアルファベット1文字）を検索
    # 4桁コード（例: 6920, 9984）と3桁+アルファベット（例: 409A, 285A）の両方に対応
    code_matches = re.findall(r'(\d{3,4}[A-Z]?)', cell_text)
    if not code_matches:
        return cell_text.strip(), "", ""
    
    # 最長のマッチを選択（アルファベット付きを優先）
    code = max(code_matches, key=len)
    
    # 4桁数字のみの場合と3桁+アルファベットの場合を区別
    # 3桁+アルファベットでない場合は4桁数字のみを採用
    digit_only_codes = [c for c in code_matches if c.isdigit() and len(c) == 4]
    alpha_codes = [c for c in code_matches if not c.isdigit()]
    
    if alpha_codes:
        code = alpha_codes[0]  # アルファベット付きコードを優先
    elif digit_only_codes:
        code = digit_only_codes[0]  # 4桁数字コード
    else:
        code = code_matches[0]  # その他
    
    # 市場（東P, 東S, 東Gなど）を検索
    market_match = re.search(r'(東[PSG]|名[12]|札|福)', cell_text)
    market = market_match.group(1) if market_match else ""
    
    # 銘柄名を抽出（コードと市場を除去）
    name = cell_text
    if code:
        name = name.replace(code, '')
    if market:
        name = name.replace(market, '')
    name = name.strip()
    
    return name, code, market


def parse_matsui_ranking_html(html_content: str) -> List[RankingStock]:
    """
    松井証券のランキングHTMLを解析してRankingStockオブジェクトのリストを返す
    
    Args:
        html_content: 松井証券ランキングページのHTML
        
    Returns:
        List[RankingStock]: 解析されたランキングデータ
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # ランキングテーブルを特定
    table = soup.select_one('table.m-table[data-type="rankingDt"]')
    if not table:
        # フォールバック: 他のテーブル形式を試す
        table = soup.select_one('table.m-table') or soup.select_one('table')
    
    if not table:
        raise ValueError("ランキングテーブルが見つかりませんでした")
    
    # tbody内の行を取得
    rows = table.select('tbody tr')
    if not rows:
        # フォールバック: 全行からヘッダーを除外
        rows = table.select('tr')[1:]
    
    results = []
    
    for row in rows:
        try:
            # 全てのtdセルを取得（クラス指定も含む）
            all_cells = row.find_all('td')
            if len(all_cells) < 7:
                continue  # 必要な列数に満たない場合はスキップ
            
            # 順位（1列目）
            rank = int(all_cells[0].get_text(strip=True))
            
            # 銘柄情報（2列目）- 銘柄名、コード、市場
            name, code, market = extract_stock_info_from_cell(all_cells[1].get_text(strip=True))
            
            # 現在値（3列目）
            current_price_text = all_cells[2].get_text(strip=True)
            current_price = parse_number(current_price_text)
            
            # 前日比情報（4列目、モバイル専用）
            price_change = 0.0
            price_change_percent = 0.0
            mobile_change_cell = all_cells[3]
            if 'm-sp-only' in mobile_change_cell.get('class', []):
                change_text = mobile_change_cell.get_text(strip=True)
                # "+210.0(+1.64%)" のような形式を解析
                change_match = re.search(r'([+-]?\d+(?:\.\d+)?)\s*\(([+-]?\d+(?:\.\d+)?)%\)', change_text)
                if change_match:
                    price_change = float(change_match.group(1))
                    price_change_percent = float(change_match.group(2))
            
            # 出来高（5列目）- "出来高：10,062,600"形式
            volume_text = all_cells[4].get_text(strip=True)
            # "出来高："プレフィックスを除去
            volume_clean = volume_text.replace('出来高：', '').replace('出来高:', '')
            volume = int(parse_number(volume_clean))
            
            # 概算売買代金（6列目）- "概算売買代金：205,675,947,000"形式
            trading_value_text = all_cells[5].get_text(strip=True)
            # "概算売買代金："プレフィックスを除去
            trading_value_clean = trading_value_text.replace('概算売買代金：', '').replace('概算売買代金:', '')
            trading_value = int(parse_number(trading_value_clean))
            
            # 株価変動率（7列目）- "株価変動率：+6.16%"形式や直接"+6.16%"形式
            volatility_text = all_cells[6].get_text(strip=True) if len(all_cells) > 6 else "0%"
            # "株価変動率："プレフィックスを除去
            volatility_clean = volatility_text.replace('株価変動率：', '').replace('株価変動率:', '')
            volatility_percent = parse_percentage(volatility_clean)
            
            stock = RankingStock(
                rank=rank,
                code=code,
                name=name,
                market=market,
                current_price=current_price,
                price_change=price_change,
                price_change_percent=price_change_percent,
                volume=volume,
                trading_value=trading_value,
                volatility_percent=volatility_percent
            )
            
            results.append(stock)
            
        except (ValueError, IndexError) as e:
            # デバッグ情報を出力
            print(f"警告: 行の解析に失敗しました - {e}")
            cell_data = [cell.get_text(strip=True) for cell in all_cells[:8]]
            print(f"      セル数: {len(all_cells)}, データ: {cell_data}")
            if len(all_cells) >= 2:
                name, code, market = extract_stock_info_from_cell(all_cells[1].get_text(strip=True))
                print(f"      解析結果: name='{name}', code='{code}', market='{market}'")
            continue
    
    return results


def fetch_daytrading_morning_ranking(timeout: int = 15) -> List[RankingStock]:
    """
    松井証券のデイトレ朝ランキングをスクレイピングして取得する関数。

    Args:
        timeout (int): HTTPリクエストのタイムアウト秒数。

    Returns:
        List[RankingStock]: ランキングデータのリスト
    """
    URL = "https://finance.matsui.co.jp/ranking-day-trading-morning/index"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        print("松井証券デイトレランキングを取得中...")
        resp = httpx.get(URL, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        
        results = parse_matsui_ranking_html(resp.text)
        print(f"ランキングデータ取得完了: {len(results)}件")
        return results

    except httpx.RequestError as e:
        print(f"エラー: HTTP リクエストに失敗しました - {e}")
        return []
    except httpx.HTTPStatusError as e:
        print(f"エラー: HTTPステータスエラー - {e.response.status_code}")
        return []
    except Exception as e:
        print(f"エラー: ランキング取得中に予期しない問題が発生しました - {e}")
        return []


# --- このファイル単体でテストするためのコード ---
if __name__ == '__main__':
    # 提供されたHTMLデータでテスト（実際の構造に基づく）
    sample_html = """
    <table class="m-table" data-type="rankingDt">
      <thead>
      <tr>
        <th>順位</th>
        <th>銘柄名(コード/市場)</th>
        <th>現在値</th>
        <th>出来高</th>
        <th>概算売買代金</th>
        <th>株価変動率</th>
        <th>注文</th>
      </tr>
      </thead>
      <tbody>
        <tr>
          <td>1</td>
          <td>
            <a href="/stock/6920/index">レーザーテック</a>
            <span>6920 東P</span>
          </td>
          <td><span class="m-down-icon">20,245.0</span></td>
          <td class="m-sp-only m-align-right"><span class="m-down-color">-1,285.0(-5.97%)</span></td>
          <td><span>出来高：</span>10,062,600</td>
          <td><span>概算売買代金：</span>205,675,947,000</td>
          <td><span class="m-sp-only">株価変動率：</span>+6.16%</td>
          <td>...</td>
        </tr>
        <tr>
          <td>2</td>
          <td>
            <a href="/stock/9984/index">ソフトバンクグループ</a>
            <span>9984 東P</span>
          </td>
          <td><span class="m-down-icon">19,025.0</span></td>
          <td class="m-sp-only m-align-right"><span class="m-down-color">-555.0(-2.83%)</span></td>
          <td><span>出来高：</span>9,424,000</td>
          <td><span>概算売買代金：</span>181,291,203,000</td>
          <td><span class="m-sp-only">株価変動率：</span>+4.55%</td>
          <td>...</td>
        </tr>
      </tbody>
    </table>
    """
    
    print("=== HTMLデータ解析テスト ===")
    try:
        stocks = parse_matsui_ranking_html(sample_html)
        print(f"解析成功: {len(stocks)}件のデータを抽出")
        
        for stock in stocks:
            print(f"{stock.rank:2d}位. [{stock.code}] {stock.name} ({stock.market})")
            print(f"     現在値: {stock.current_price:,.1f}円")
            print(f"     出来高: {stock.volume:,}株")
            print(f"     売買代金: {stock.trading_value:,}円")
            print(f"     変動率: {stock.volatility_percent:+.2f}%")
            if stock.price_change != 0:
                print(f"     前日比: {stock.price_change:+.1f}円 ({stock.price_change_percent:+.2f}%)")
            print()
            
    except Exception as e:
        print(f"HTMLデータ解析エラー: {e}")
    
    print("\n=== 実際のWebサイトからのデータ取得テスト ===")
    ranking = fetch_daytrading_morning_ranking()
    if ranking:
        print(f"取得成功: {len(ranking)}件")
        print("\n--- デイトレランキング上位10位 ---")
        for stock in ranking:
            print(f"{stock.rank:2d}位. [{stock.code}] {stock.name} ({stock.market})")
            print(f"     現在値: {stock.current_price:,.1f}円")
            print(f"     出来高: {stock.volume:,}株")
            print(f"     売買代金: {stock.trading_value:,}円")
            print(f"     変動率: {stock.volatility_percent:+.2f}%")
            if stock.price_change != 0:
                print(f"     前日比: {stock.price_change:+.1f}円 ({stock.price_change_percent:+.2f}%)")
            print()
    else:
        print("ランキングデータを取得できませんでした。")
