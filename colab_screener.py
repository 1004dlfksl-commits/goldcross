# -*- coding: utf-8 -*-
# 📱 스마트폰 사용자 안내: 상단 메뉴에서 [런타임] -> [모두 실행]을 누르면 원터치로 분석이 시작됩니다.

# ==========================================
# 1. 필수 라이브러리 설치 및 임포트
# ==========================================
try:
    import IPython
    shell = IPython.get_ipython()
    if shell is not None:
        print("📥 필수 라이브러리(pykrx, finance-datareader)를 백그라운드 설치 중...")
        shell.system("pip install -q pykrx finance-datareader")
except Exception:
    pass

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import concurrent.futures
import requests
from bs4 import BeautifulSoup
import time
from IPython.display import display, HTML

# 구글 코랩 인터랙티브 데이터 테이블 활성화
try:
    from google.colab import data_table
    data_table.enable_dataframe_formatter()
except ImportError:
    pass

# ==========================================
# 2. 구글 코랩 입력 폼 설정 (Colab Forms)
# ==========================================
#@title 🎯 GOLDCROSS 주식 스크리너 설정 { run: "auto" }

#@markdown ### ⚙️ 분석 대상 및 필터 설정
시장_선택 = "KOSPI + KOSDAQ" #@param ["KOSPI", "KOSDAQ", "KOSPI + KOSDAQ"]
최소_시가총액_억원 = 1000 #@param {type:"slider", min:500, max:10000, step:500}
최소_일거래대금_억원 = 10 #@param {type:"slider", min:1, max:50, step:1}
최대_분석_종목수 = 500 #@param {type:"slider", min:100, max:1500, step:100}
골드크로스_범위_거래일 = 3 #@param {type:"slider", min:1, max:5, step:1}
적자_기업_제외 = True #@param {type:"boolean"}
데이터_강제_새로고침 = False #@param {type:"boolean"}

#@markdown ### 📉 바닥 지지선 기준 (조건 B)
지지선_기준 = "120일선" #@param ["120일선", "60일선", "20일선", "5일선"]

#@markdown ### 🛡️ 고점 제외 안전 모드 설정
안전_모드_고점제외 = True #@param {type:"boolean"}
RSI_과열_컷트라인 = 65 #@param {type:"slider", min:40, max:80, step:5}

#@markdown ### 🎯 단기 저격 모드 (스윙) 설정
단기_저격_모드_스윙 = False #@param {type:"boolean"}

#@markdown ### 📱 모바일 화면 최적화
모바일_화면_최적화 = True #@param {type:"boolean"}
차트_조회_기간_거래일 = 60 #@param [30, 60, 120, 150] {type:"raw"}
비교_분석_종목코드 = "005930" #@param {type:"string"}


# ==========================================
# 3. 데이터 수집 및 연산 헬퍼 함수
# ==========================================

# 시가총액을 한글 형태로 변환하는 함수
def format_marcap(val):
    if pd.isna(val) or val <= 0:
        return "N/A"
    jo = val // 1000000000000
    eok = (val % 1000000000000) // 100000000
    if jo > 0:
        if eok > 0:
            return f"{int(jo)}조 {int(eok)}억"
        return f"{int(jo)}조"
    return f"{int(eok)}억"

# 최근 N일 이내 골드크로스 여부 탐지
def detect_recent_golden_cross(df, gc_days=3):
    for offset in range(gc_days):
        idx = -1 - offset
        if idx - 1 < -len(df):
            break
        ma5_today = df['MA_5'].iloc[idx]
        ma20_today = df['MA_20'].iloc[idx]
        ma5_yesterday = df['MA_5'].iloc[idx - 1]
        ma20_yesterday = df['MA_20'].iloc[idx - 1]
        
        if ma5_today > ma20_today and ma5_yesterday <= ma20_yesterday:
            cross_date = df.index[idx].strftime("%Y-%m-%d")
            return True, cross_date, offset
    return False, None, None

# RSI 지표 계산
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period-1, adjust=False).mean()
    avg_loss = loss.ewm(com=period-1, adjust=False).mean()
    rs = np.where(avg_loss == 0, np.nan, avg_gain / avg_loss)
    rsi = np.where(avg_loss == 0, 100.0, 100.0 - (100.0 / (1.0 + rs)))
    rsi = np.where((avg_gain == 0) & (avg_loss == 0), 50.0, rsi)
    return pd.Series(rsi, index=prices.index)

# 시장 종목 리스트 획득
def get_market_tickers():
    try:
        df_kospi = fdr.StockListing('KOSPI')
        df_kosdaq = fdr.StockListing('KOSDAQ')
        df_kospi['Market'] = 'KOSPI'
        df_kosdaq['Market'] = 'KOSDAQ'
        df = pd.concat([df_kospi, df_kosdaq], ignore_index=True)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"❌ 종목 리스트 획득 실패: {e}")
        return pd.DataFrame()

# 개별 종목 주가 정보 병렬 다운로드
def fetch_stock_history(ticker, start_date, end_date):
    try:
        df = fdr.DataReader(ticker, start_date, end_date)
        if df is not None and not df.empty and len(df) >= 120:
            return ticker, df
    except:
        pass
    return ticker, None

# 네이버 금융 PER/적자 여부 크롤링
def check_is_deficit(ticker):
    url = f"https://finance.naver.com/item/main.naver?code={ticker}"
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}, timeout=3)
        if r.status_code != 200:
            return False, "N/A"
        soup = BeautifulSoup(r.text, 'html.parser')
        per_tag = soup.find('em', id='_per')
        per_val = per_tag.text.strip() if per_tag else "N/A"
        eps_tag = soup.find('em', id='_eps')
        eps_val = eps_tag.text.strip() if eps_tag else "N/A"
        
        per_clean = per_val.replace(',', '')
        eps_clean = eps_val.replace(',', '')
        
        if per_clean == "N/A" or "-" in per_clean or "-" in eps_clean:
            return True, per_val
        try:
            val = float(per_clean)
            if val <= 0:
                return True, per_val
        except ValueError:
            return True, per_val
        return False, per_val
    except Exception:
        return False, "Error"

# ==========================================
# 4. 메모리 캐시 로직 (중복 다운로드 차단)
# ==========================================
if 'stock_cache' not in globals() or 데이터_강제_새로고침:
    stock_cache = {}
    print("🔄 메모리 캐시가 초기화되었습니다. 데이터를 새로 수집합니다.")
else:
    print(f"✅ 기존 데이터 캐시({len(stock_cache)}개 종목)를 활용하여 초고속 분석을 수행합니다.")

# ==========================================
# 5. 본문 실행 영역
# ==========================================
def main():
    # 📱 모바일 가독성 및 재생 버튼 시각화 안내 CSS 주입
    display(HTML("""
    <style>
        .colab-output-container {
            font-family: 'Inter', 'Noto Sans KR', sans-serif;
            font-size: 8px !important;
            color: #e2e8f0;
            line-height: 1.4;
        }
        .colab-output-container h3 {
            font-size: 9.5px !important;
            font-weight: 700;
            color: #ffffff;
            margin-top: 10px;
            margin-bottom: 4px;
        }
        .colab-output-container p {
            font-size: 8px !important;
            color: #94a3b8;
            margin-bottom: 6px;
        }
        .notice-card {
            background-color: #0f172a;
            border: 1px solid #334155;
            border-left: 4px solid #3b82f6;
            padding: 6px 8px;
            border-radius: 6px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
        }
        .notice-card-text {
            font-size: 8px !important;
            color: #cbd5e1;
            font-weight: 600;
            line-height: 1.4;
        }
        /* 프리미엄 금융 다크테마 테이블 */
        .goldcross-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 7.5px !important;
            margin-top: 4px;
            margin-bottom: 16px;
            background-color: #0f172a;
            color: #f8fafc;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }
        .goldcross-table th {
            background-color: #1e293b;
            color: #cbd5e1;
            font-weight: 600;
            text-align: left;
            padding: 4px 6px;
            border-bottom: 1.5px solid #334155;
            white-space: nowrap;
        }
        .goldcross-table td {
            padding: 4px 6px;
            border-bottom: 1px solid #1e293b;
            color: #e2e8f0;
            white-space: nowrap;
        }
        .goldcross-table tr:nth-child(even) {
            background-color: #131c2e;
        }
        .goldcross-table tr:hover {
            background-color: #1e293b;
        }
        /* 종목명 링크 */
        .stock-link {
            color: #60a5fa !important;
            text-decoration: none !important;
            font-weight: 700;
        }
        .stock-link:active {
            color: #93c5fd !important;
            text-decoration: underline !important;
        }
        /* 바로가기 버튼 */
        .btn-naver-link {
            display: inline-block;
            background-color: #1e40af;
            color: #ffffff !important;
            text-decoration: none !important;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 9px !important;
            font-weight: 600;
            text-align: center;
        }
        .btn-naver-link:hover {
            background-color: #1d4ed8;
        }
        .table-responsive {
            width: 100%;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }
    </style>
    
    <div class='notice-card'>
        <span style='font-size: 16px; margin-right: 8px;'>💡</span>
        <div class='notice-card-text'>
            <b>[ 실행 방법 ]</b> 설정을 변경한 뒤, 코드 셀 바로 왼쪽 끝에 있는 <b>동그란 재생(▶) 아이콘</b>을 누르면 분석이 원터치로 다시 수행됩니다!
        </div>
    </div>
    """))

    # 분석 일자 정의
    today_dt = datetime.today()
    start_dt = today_dt - timedelta(days=250)
    start_date_str = start_dt.strftime("%Y-%m-%d")
    today_date_str = today_dt.strftime("%Y-%m-%d")
    
    # 이평선 지지 기준 값 수치 추출
    support_window = 120
    if "60일선" in 지지선_기준:
        support_window = 60
    elif "20일선" in 지지선_기준:
        support_window = 20
    elif "5일선" in 지지선_기준:
        support_window = 5
        
    markets = []
    if "KOSPI" in 시장_선택:
        markets.append("KOSPI")
    if "KOSDAQ" in 시장_선택:
        markets.append("KOSDAQ")
        
    print("📥 시장 원천 리스트 수집 중...")
    raw_list = get_market_tickers()
    if raw_list.empty:
        print("❌ 종목 데이터 로드 실패")
        return
        
    # 1차 필터링
    filtered_list = raw_list[
        (raw_list['Market'].isin(markets)) & 
        (raw_list['Marcap'] >= 최소_시가총액_억원 * 100000000) &
        (raw_list['Amount'] >= 최소_일거래대금_억원 * 100000000)
    ]
    filtered_list = filtered_list.sort_values(by='Amount', ascending=False).head(최대_분석_종목수)
    tickers = filtered_list['Code'].tolist()
    
    ticker_name_map = dict(zip(filtered_list['Code'], filtered_list['Name']))
    ticker_market_map = dict(zip(filtered_list['Code'], filtered_list['Market']))
    ticker_amount_map = dict(zip(filtered_list['Code'], filtered_list['Amount']))
    ticker_marcap_map = dict(zip(filtered_list['Code'], filtered_list['Marcap']))
    
    print(f"🔎 분석 대상 후보군: {len(tickers)}개 선정 완료.")
    
    # 미수집 종목만 다운로드 진행
    missing_tickers = [t for t in tickers if t not in stock_cache]
    if missing_tickers:
        print(f"📥 신규 데이터 다운로드 중 ({len(missing_tickers)}개 종목)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            futures = [executor.submit(fetch_stock_history, t, start_date_str, today_date_str) for t in missing_tickers]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                t, df = future.result()
                if df is not None:
                    stock_cache[t] = df
                if i > 0 and i % 50 == 0:
                    print(f"  └ {i}/{len(missing_tickers)} 다운로드 완료")
                    
    # 결과 분석
    passed_defensive = []
    passed_gc = []
    passed_support = []
    passed_outperform = []
    passed_sniper = []
    
    # 지수 가져오기
    try:
        kospi_hist = fdr.DataReader('KS11', start_date_str, today_date_str)
        kosdaq_hist = fdr.DataReader('KQ11', start_date_str, today_date_str)
    except:
        kospi_hist, kosdaq_hist = pd.DataFrame(), pd.DataFrame()
        
    for ticker in tickers:
        df = stock_cache.get(ticker)
        if df is None or len(df) < 120:
            continue
            
        try:
            # 기술적 보조 지표 계산
            df['MA_5'] = df['Close'].rolling(window=5).mean()
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['MA_support'] = df['Close'].rolling(window=support_window).mean()
            df['RSI'] = calculate_rsi(df['Close'], period=14)
            
            close_today = df['Close'].iloc[-1]
            ma5_today = df['MA_5'].iloc[-1]
            ma20_today = df['MA_20'].iloc[-1]
            ma_support_today = df['MA_support'].iloc[-1]
            rsi_today = df['RSI'].iloc[-1]
            
            disparity_today = (close_today / ma_support_today) * 100
            gap_pct = ((close_today - ma20_today) / ma20_today) * 100
            
            has_gc, gc_date, gc_offset = detect_recent_golden_cross(df, 골드크로스_범위_거래일)
            gc_text = "오늘 발생" if gc_offset == 0 else (f"{gc_offset}일 전" if gc_offset is not None else "-")
            is_supported = close_today > ma_support_today
            
            market = ticker_market_map[ticker]
            idx_hist = kospi_hist if market == 'KOSPI' else kosdaq_hist
            stock_ret_5d = (df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6]
            idx_ret_5d = (idx_hist['Close'].iloc[-1] - idx_hist['Close'].iloc[-6]) / idx_hist['Close'].iloc[-6] if not idx_hist.empty else 0
            
            if idx_ret_5d < 0:
                is_defensive = stock_ret_5d >= 0
            else:
                is_defensive = (stock_ret_5d >= 0) or (stock_ret_5d >= idx_ret_5d)
                
            df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
            volume_3d_ago = df['Volume'].iloc[-4]
            vol_ma20_3d_ago = df['Volume_MA20'].iloc[-4]
            volume_today = df['Volume'].iloc[-1]
            recent_3d_vol_mean = df['Volume'].iloc[-3:].mean()
            max_amount_3d = (df['Close'].iloc[-3:] * df['Volume'].iloc[-3:]).max()
            
            stock_info = {
                'Code': ticker,
                'Name': ticker_name_map[ticker],
                'Market': market,
                'Close': int(close_today),
                'Amount': int(ticker_amount_map[ticker] / 100000000),
                'Marcap': ticker_marcap_map[ticker],
                'Stock_Return_5D': stock_ret_5d,
                'Index_Return_5D': idx_ret_5d,
                'Gap_Pct': gap_pct,
                'Disparity': disparity_today,
                'RSI': rsi_today,
                'Cross_Date': gc_date if has_gc else "-",
                'Cross_Days_Ago': gc_text,
                'MA5': ma5_today,
                'MA20': ma20_today,
                'MA_support': ma_support_today,
                'Volume_Today': int(volume_today),
                'Volume_3D_Ago': int(volume_3d_ago),
                'Max_Amount_3D': float(max_amount_3d / 100000000)
            }
            
            # --- A. 일반 조건 필터링 ---
            passed_safe_filter = True
            if 안전_모드_고점제외:
                if disparity_today > 110.0 or pd.isna(rsi_today) or rsi_today > RSI_과열_컷트라인:
                    passed_safe_filter = False
            if passed_safe_filter:
                if has_gc:
                    passed_gc.append(stock_info.copy())
                if is_supported:
                    passed_support.append(stock_info.copy())
                if is_defensive:
                    passed_outperform.append(stock_info.copy())
                if has_gc and is_supported and is_defensive:
                    passed_defensive.append(stock_info.copy())
                    
            # --- B. 단기 저격 조건 필터링 (스윙) ---
            is_trend_aligned = ma5_today > ma20_today
            is_vol_burst_3d_ago = volume_3d_ago > vol_ma20_3d_ago
            is_vol_pulled_back = volume_today <= 0.6 * recent_3d_vol_mean
            is_rsi_cond = (not pd.isna(rsi_today)) and (45.0 <= rsi_today <= 55.0)
            
            if is_trend_aligned and is_vol_burst_3d_ago and is_vol_pulled_back and is_rsi_cond:
                if not 안전_모드_고점제외 or (disparity_today <= 110.0):
                    passed_sniper.append(stock_info.copy())
        except:
            pass

    # 재무 건전성 및 바로가기 일괄 검증 필터
    def apply_financial_filter(candidates_list):
        if not candidates_list:
            return []
        filtered = []
        for item in candidates_list:
            ticker = item['Code']
            is_deficit, per = check_is_deficit(ticker)
            item['PER'] = per
            item['Is_Deficit'] = is_deficit
            item['네이버증권'] = f"https://m.stock.naver.com/domestic/stock/{ticker}"
            if 적자_기업_제외 and is_deficit:
                continue
            filtered.append(item)
        return filtered

    print("📋 후보군 재무 건전성 필터링 중...")
    passed_defensive = apply_financial_filter(passed_defensive)
    passed_gc = apply_financial_filter(passed_gc)
    passed_support = apply_financial_filter(passed_support)
    passed_outperform = apply_financial_filter(passed_outperform)
    passed_sniper = apply_financial_filter(passed_sniper)
    
    # ==========================================
    # 6. 테이블 결과물 출력 영역 (Google Colab 최적화)
    # ==========================================
    def show_table(title, desc, data_list, is_sniper_tbl=False):
        # 30% 축소된 글씨가 들어가는 최적화 컨테이너
        display(HTML(f"<div class='colab-output-container'><div style='border-left: 4px solid #3b82f6; padding-left: 8px; margin-top: 20px;'><h3>{title}</h3></div><p>{desc}</p></div>"))
        if not data_list:
            display(HTML("<div class='colab-output-container'><p style='color: #ef4444;'>조건을 만족하는 종목이 존재하지 않습니다.</p></div>"))
            return
        
        df = pd.DataFrame(data_list)
        
        if 모바일_화면_최적화:
            # 모바일 최적화: 가로 스크롤 없이 가로 폭에 딱 맞춤 (5개 핵심 컬럼만)
            # 종목명 자체를 링크로 감싸서 클릭 가능하게 함
            df['종목명'] = df.apply(lambda r: f"<a class='stock-link' href='{r['네이버증권']}' target='_blank'>{r['Name']}</a>", axis=1)
            df['현재가'] = df['Close'].map(lambda x: f"{x:,}")
            
            if is_sniper_tbl:
                df = df.sort_values(by='Max_Amount_3D', ascending=False)
                df['거래대금'] = df['Max_Amount_3D'].map(lambda x: f"{x:,.0f}억")
                df['RSI'] = df['RSI'].map(lambda x: f"{x:.1f}" if not pd.isna(x) else "-")
                # 20일선이격도 색상 입히기
                df['20일선이격'] = df['Gap_Pct'].map(lambda x: f"<span style='color: #ef4444; font-weight:600;'>{x:+.1f}%</span>" if x > 0 else (f"<span style='color: #3b82f6; font-weight:600;'>{x:+.1f}%</span>" if x < 0 else f"{x:+.1f}%"))
                df_show = df[['종목명', '현재가', '거래대금', 'RSI', '20일선이격']].copy()
                df_show.columns = ['종목명', '현재가', '거래대금(최대)', 'RSI', '20일이격']
            else:
                df = df.sort_values(by='Amount', ascending=False)
                df['거래대금'] = df['Amount'].map(lambda x: f"{x:,}억")
                # 5일 수익률 색상 입히기
                df['5일수익률'] = df['Stock_Return_5D'].map(lambda x: f"<span style='color: #ef4444; font-weight:600;'>{x:+.1%}</span>" if x > 0 else (f"<span style='color: #3b82f6; font-weight:600;'>{x:+.1%}</span>" if x < 0 else f"{x:+.1%}"))
                df['크로스일'] = df['Cross_Days_Ago']
                df_show = df[['종목명', '현재가', '거래대금', '5일수익률', '크로스일']].copy()
                df_show.columns = ['종목명', '현재가', '거래대금', '5일수익률', '크로스일']
        else:
            # 전체 화면 모드 (기존 14개 컬럼 전체 출력 + 가로 스크롤 허용)
            if is_sniper_tbl:
                df = df.sort_values(by='Max_Amount_3D', ascending=False)
                df_show = df[['Code', 'Name', 'Market', 'Close', 'Max_Amount_3D', 'Volume_Today', 'Volume_3D_Ago', 'RSI', 'Disparity', 'Gap_Pct', 'PER', '네이버증권']].copy()
                df_show.columns = ['종목코드', '종목명', '소속시장', '현재가(원)', '3일간최대거래대금', '오늘거래량', '3일전거래량', 'RSI', '이격도(%)', '20일선이격도', 'PER', '네이버증권']
                df_show['3일간최대거래대금'] = df_show['3일간최대거래대금'].map(lambda x: f"{x:,.1f}억")
                df_show['오늘거래량'] = df_show['오늘거래량'].map(lambda x: f"{x:,}")
                df_show['3일전거래량'] = df_show['3일전거래량'].map(lambda x: f"{x:,}")
            else:
                df = df.sort_values(by='Amount', ascending=False)
                df_show = df[['Code', 'Name', 'Market', 'Close', 'Amount', 'Marcap', 'Stock_Return_5D', 'Index_Return_5D', 'Disparity', 'RSI', 'Gap_Pct', 'Cross_Days_Ago', 'PER', '네이버증권']].copy()
                df_show.columns = ['종목코드', '종목명', '소속시장', '현재가(원)', '일거래대금', '시가총액', '5일 종목수익률', '5일 지수수익률', '이격도(%)', 'RSI', '20일선이격도', '최근 크로스일', 'PER', '네이버증권']
                df_show['일거래대금'] = df_show['일거래대금'].map(lambda x: f"{x:,}억")
                df_show['시가총액'] = df_show['시가총액'].apply(format_marcap)
                df_show['5일 종목수익률'] = df_show['5일 종목수익률'].map(lambda x: f"{x:+.2%}")
                df_show['5일 지수수익률'] = df_show['5일 지수수익률'].map(lambda x: f"{x:+.2%}")
                
            df_show['20일선이격도'] = df_show['20일선이격도'].map(lambda x: f"{x:+.2f}%")
            df_show['이격도(%)'] = df_show['이격도(%)'].map(lambda x: f"{x:.1f}%")
            df_show['RSI'] = df_show['RSI'].map(lambda x: f"{x:.1f}" if not pd.isna(x) else "-")
            df_show['현재가(원)'] = df_show['현재가(원)'].map(lambda x: f"{x:,}")
            df_show['네이버증권'] = df_show.apply(lambda r: f"<a class='btn-naver-link' href='{r['네이버증권']}' target='_blank'>바로가기 ↗</a>", axis=1)
        
        # HTML 렌더링 후 화면 출력
        html_table = df_show.to_html(escape=False, index=False, classes='goldcross-table')
        display(HTML(f"<div class='table-responsive'>{html_table}</div>"))

    # 모드에 따른 분기 렌더링
    if 단기_저격_모드_스윙:
        show_table(
            "🎯 단기 저격 포트폴리오 (스윙 눌림목)",
            "3거래일 전 대량 거래대금이 들어온 후, 오늘 거래량이 최근 3일 평균 거래량의 60% 이하로 급감하며 숨 고르기 중인 5/20 정배열 단기 스윙 종목군입니다.",
            passed_sniper,
            is_sniper_tbl=True
        )
    else:
        show_table(
            "🛡️ 통합 포트폴리오 (조건 A+B+C 만족)",
            f"골드크로스, {지지선_기준} 이평선 지지, 그리고 지수 대비 가격 방어력을 동시에 만족하는 통합 헤지 포트폴리오 후보군입니다.",
            passed_defensive
        )
        show_table(
            "⚡ 최근 골든크로스 종목 (조건 A 만족)",
            f"최근 영업일 기준 지정된 범위 이내에 5일 이동평균선이 20일 이동평균선을 상향 돌파하여 상승 전환된 종목군입니다.",
            passed_gc
        )
        show_table(
            f"📈 {지지선_기준} 지지 종목 (조건 B 만족)",
            f"현재 주가가 선택된 바닥 지지선({지지선_기준}) 위에 위치하여 탄탄한 하방 지지력이 확인되는 종목군입니다.",
            passed_support
        )
        show_table(
            "💪 지수 대비 방어우수 종목 (조건 C 만족)",
            "최근 5거래일 동안 시장의 강한 하락 압력에도 가격 방어력을 보여주거나 지수보다 뛰어난 초과수익을 낸 종목군입니다.",
            passed_outperform
        )
        
    # ==========================================
    # 7. 개별 비교 분석 차트 출력 영역
    # ==========================================
    stk_df = stock_cache.get(비교_분석_종목코드)
    if stk_df is not None:
        stock_sub = stk_df.tail(차트_조회_기간_거래일).copy()
        sel_market = "KOSPI"
        # 소속시장 확인
        all_found = passed_defensive + passed_gc + passed_support + passed_outperform + passed_sniper
        sel_name = f"종목코드: {비교_분석_종목코드}"
        for itm in all_found:
            if itm['Code'] == 비교_분석_종목코드:
                sel_market = itm['Market']
                sel_name = itm['Name']
                break
                
        idx_hist = kospi_hist if sel_market == 'KOSPI' else kosdaq_hist
        if not idx_hist.empty:
            index_sub = idx_hist.tail(차트_조회_기간_거래일).copy()
            merged_plot = pd.DataFrame(index=stock_sub.index)
            merged_plot['Stock_Close'] = stock_sub['Close']
            merged_plot = merged_plot.join(index_sub['Close'], how='inner', rsuffix='_Index')
            merged_plot = merged_plot.dropna()
            
            if not merged_plot.empty:
                merged_plot['Stock_Norm'] = (merged_plot['Stock_Close'] / merged_plot['Stock_Close'].iloc[0]) * 100
                merged_plot['Index_Close'] = merged_plot['Close']
                merged_plot['Index_Norm'] = (merged_plot['Index_Close'] / merged_plot['Index_Close'].iloc[0]) * 100
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=merged_plot.index,
                    y=merged_plot['Stock_Norm'],
                    mode='lines+markers',
                    name=f"{sel_name} (정규화)",
                    line=dict(color='#3b82f6', width=3),
                    marker=dict(size=5),
                    hovertemplate='날짜: %{x}<br>종목 지수: %{y:.2f}%<br>'
                ))
                fig.add_trace(go.Scatter(
                    x=merged_plot.index,
                    y=merged_plot['Index_Norm'],
                    mode='lines',
                    name=f"{sel_market} 지수 (정규화)",
                    line=dict(color='#9ca3af', width=2, dash='dash'),
                    hovertemplate='날짜: %{x}<br>시장 지수: %{y:.2f}%<br>'
                ))
                
                # 모바일 화면 크기 최적화 반응형 제어
                chart_width = 450 if 모바일_화면_최적화 else None
                chart_height = 400 if 모바일_화면_최적화 else 500
                
                fig.update_layout(
                    title=dict(
                        text=f"📊 {sel_name} vs {sel_market} 상대 수익률 추이",
                        x=0.5,
                        font=dict(size=15, color='#1e293b')
                    ),
                    xaxis=dict(title="날짜", gridcolor='#e2e8f0'),
                    yaxis=dict(title="상대 수익률 (시작점 = 100%)", gridcolor='#e2e8f0'),
                    legend=dict(
                        x=0.01,
                        y=0.99,
                        bgcolor='rgba(255, 255, 255, 0.8)',
                        bordercolor='#cbd5e1',
                        borderwidth=1
                    ),
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    margin=dict(l=20, r=20, t=50, b=20),
                    width=chart_width,
                    height=chart_height
                )
                
                display(HTML("<div style='margin-top:40px; border-left:4px solid #10b981; padding-left:10px;'><h3>종목 vs 시장 가격 추이 비교 차트</h3></div>"))
                fig.show()
            else:
                print("❌ 주가와 지수의 비교를 위한 병합 데이터가 부족합니다.")
        else:
            print("❌ 시장 지수 데이터가 비어있어 그래프를 출력할 수 없습니다.")
    else:
        print("\n🔍 개별 비교 차트를 출력하려면 위에 있는 '비교_분석_종목코드'란에 6자리 종목번호(예: 005930)를 입력하고 실행해 주세요.")

if __name__ == '__main__':
    main()
