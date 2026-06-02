import streamlit as st
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

# 1. 페이지 설정 및 디자인 스타일 정의 (Premium Dark/Modern Theme)
st.set_page_config(
    page_title="✨ GoldCross | Premium Stock Screener",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS로 UI 고도화 (신뢰도 높은 금융 기관 스타일)
st.markdown("""
    <style>
    /* 전체 폰트 및 배경 고도화 */
    html, body, [data-testid="stSidebarView"] {
        font-family: 'Inter', 'Noto Sans KR', sans-serif;
    }
    
    /* 대시보드 카드 디자인 */
    .dashboard-card {
        background-color: #1E293B;
        padding: 22px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.5px;
    }
    .metric-label {
        font-size: 13px;
        color: #E2E8F0; /* 더 선명하고 밝은 회색으로 가독성 대폭 향상 */
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    
    /* 사이드바 스타일링 및 가독성 극대화 */
    section[data-testid="stSidebar"] {
        background-color: #0B0F19 !important;
        border-right: 1px solid #1E293B !important;
    }
    /* 사이드바 위젯 라벨 및 설명 텍스트 밝은 색상으로 오버라이드 (가독성 최우선) */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] span {
        color: #F8FAFC !important; /* 순백색에 가까운 아주 밝은 컬러로 변경 */
        font-weight: 500 !important;
        font-size: 14px !important;
    }
    /* 도움말/설명 문구 가독성 */
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #E2E8F0 !important; /* 가독성을 위해 한 단계 더 밝은 실버로 변경 */
        font-size: 13px !important;
        line-height: 1.4 !important;
    }
    /* 사이드바 헤더 및 서브헤더 가독성 */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    
    /* 메인 화면 탭 스타일링 커스텀 */
    button[data-baseweb="tab"] {
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
    }
    button[aria-selected="true"] {
        color: #3B82F6 !important;
        border-bottom-color: #3B82F6 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 시가총액을 한글 형태로 변환하는 함수 (예: 2031조 5,818억)
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

# 최근 N일 이내에 골드크로스가 발생했는지 확인하는 함수
def detect_recent_golden_cross(df, gc_days=3):
    # 가장 최근 거래일부터 과거 방향으로 순회
    for offset in range(gc_days):
        idx = -1 - offset
        # 만약 전체 데이터 범위를 넘어가면 중단
        if idx - 1 < -len(df):
            break
        ma5_today = df['MA_5'].iloc[idx]
        ma20_today = df['MA_20'].iloc[idx]
        ma5_yesterday = df['MA_5'].iloc[idx - 1]
        ma20_yesterday = df['MA_20'].iloc[idx - 1]
        
        # 골드크로스 조건 만족 여부 확인
        if ma5_today > ma20_today and ma5_yesterday <= ma20_yesterday:
            cross_date = df.index[idx].strftime("%Y-%m-%d")
            return True, cross_date, offset
    return False, None, None

# RSI(Relative Strength Index) 계산 함수
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # 지수이동평균(EMA)을 사용하여 Wilder의 RSI 계산 방식 구현
    avg_gain = gain.ewm(com=period-1, adjust=False).mean()
    avg_loss = loss.ewm(com=period-1, adjust=False).mean()
    
    # 분모가 0이 되는 것을 방지하여 에러를 예방합니다.
    rs = np.where(avg_loss == 0, np.nan, avg_gain / avg_loss)
    rsi = np.where(avg_loss == 0, 100.0, 100.0 - (100.0 / (1.0 + rs)))
    rsi = np.where((avg_gain == 0) & (avg_loss == 0), 50.0, rsi)
    return pd.Series(rsi, index=prices.index)

# 캐싱을 사용해 KOSPI / KOSDAQ 종목 리스트 데이터 불러오기
@st.cache_data(ttl=3600)
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
        st.error(f"시장 종목 리스트를 불러오는 중 오류 발생: {e}")
        return pd.DataFrame()

# 캐싱을 사용해 지수 데이터 불러오기 (KS11: 코스피, KQ11: 코스닥)
@st.cache_data(ttl=3600)
def get_index_data(start_date, end_date):
    try:
        kospi = fdr.DataReader('KS11', start_date, end_date)
        kosdaq = fdr.DataReader('KQ11', start_date, end_date)
        return kospi, kosdaq
    except Exception as e:
        st.error(f"지수 데이터를 불러오는 중 오류 발생: {e}")
        return pd.DataFrame(), pd.DataFrame()

# 개별 종목 주가 정보 병렬 다운로드를 위한 함수
def fetch_stock_history(ticker, start_date, end_date):
    try:
        df = fdr.DataReader(ticker, start_date, end_date)
        if df is not None and not df.empty and len(df) >= 120:
            return ticker, df
    except:
        pass
    return ticker, None

# 네이버 금융에서 PER 및 적자 여부 크롤링 (캐싱 적용)
@st.cache_data(ttl=86400)
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

# 대시보드 메인 화면 타이틀 (Premium 헤더)
st.markdown("""
    <div style="background: #0F172A; padding: 26px 28px; border-radius: 12px; border: 1px solid #1E293B; border-left: 4px solid #3B82F6; margin-bottom: 28px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
        <h1 style="color: #F8FAFC; margin: 0; font-size: 30px; font-weight: 700; letter-spacing: -0.5px;">✨ GOLDCROSS</h1>
        <p style="color: #94A3B8; margin: 8px 0 0 0; font-size: 14px; font-weight: 400; line-height: 1.5; max-width: 900px;">
            골드크로스(GoldCross) 스크리너는 시장 하락 압력을 극복하고 다양한 이평선 지지를 기반으로 상승 변곡점에 진입한 최우량 종목을 탐색하는 전문 퀀트 스캐너입니다.
        </p>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 사이드바 설정 영역
# ==========================================
st.sidebar.header("⚙️ 분석 필터 설정")

markets = st.sidebar.multiselect(
    "검색 대상 시장",
    options=["KOSPI", "KOSDAQ"],
    default=["KOSPI", "KOSDAQ"]
)

min_marcap = st.sidebar.slider(
    "최소 시가총액 (억 원)",
    min_value=500,
    max_value=10000,
    value=1000,
    step=500,
    help="설정한 시가총액 이상의 탄탄한 기업들만 필터링합니다."
)

min_amount = st.sidebar.slider(
    "최소 일 거래대금 (억 원)",
    min_value=1,
    max_value=50,
    value=10,
    step=1,
    help="설정한 거래대금 이상으로 활발히 거래되는 종목만 찾습니다."
)

max_scan = st.sidebar.slider(
    "최대 분석 대상 종목 수",
    min_value=100,
    max_value=1500,
    value=500,
    step=100,
    help="분석할 상위 거래대금 순 종목의 개수입니다. 개수가 적을수록 분석 속도가 빨라집니다."
)

gc_days_window = st.sidebar.slider(
    "골드크로스 발생 범위 (최근 N거래일 이내)",
    min_value=1,
    max_value=5,
    value=3,
    step=1,
    help="설정한 일수 이내에 골드크로스가 한 번이라도 발생한 종목을 모두 탐지합니다."
)

exclude_deficit = st.sidebar.checkbox(
    "최근 적자 기업 제외 (PER 마이너스/NA)",
    value=True,
    help="네이버 금융 기준으로 PER이 마이너스이거나 적자인 기업을 리스트에서 필터링합니다."
)

support_ma_selection = st.sidebar.radio(
    "바닥 지지선 기준 (조건 B)",
    options=[
        "120일선 (장기 바닥 지지)",
        "60일선 (중기 바닥 지지)",
        "20일선 (단기 추세 지지)",
        "5일선 (초단기 돌파 지지)"
    ],
    index=0,
    help="정배열 바닥을 판단할 이동평균선 기준을 선택합니다."
)

safe_mode = st.sidebar.toggle(
    "고점 과열 종목 제외 (안전 모드)",
    value=True,
    help="주가가 바닥선 대비 너무 높거나(이격도 110% 초과) RSI 지표가 과열된 종목을 필터링합니다."
)

rsi_cutoff = st.sidebar.slider(
    "RSI 과열 컷트라인 설정",
    min_value=40,
    max_value=80,
    value=65,
    step=5,
    disabled=not safe_mode,
    help="안전 모드가 켜져 있을 때만 작동하며, 지정한 RSI 값 이하인 종목만 필터링합니다."
)

sniper_mode = st.sidebar.toggle(
    "🎯 단기 저격 모드 (스윙)",
    value=False,
    help="최근 대량 거래량이 유입된 후 거래량이 급감한 5일/20일 정배열 눌림목 종목을 탐지합니다."
)

chart_days = st.sidebar.selectbox(
    "비교 차트 조회 기간 (거래일)",
    options=[30, 60, 120, 150],
    index=1
)

# 데이터 탐색 및 계산 일자 준비
today_dt = datetime.today()
start_dt = today_dt - timedelta(days=250)

start_date_str = start_dt.strftime("%Y-%m-%d")
today_date_str = today_dt.strftime("%Y-%m-%d")

# ==========================================
# 메인화면 - 시장 현황 영역
# ==========================================
st.subheader("📊 최근 5거래일 국내 시장 지수 추이")

kospi_hist, kosdaq_hist = get_index_data(start_date_str, today_date_str)

if not kospi_hist.empty and not kosdaq_hist.empty:
    k_close = kospi_hist['Close']
    kq_close = kosdaq_hist['Close']
    
    kospi_ret_5d = (k_close.iloc[-1] - k_close.iloc[-6]) / k_close.iloc[-6]
    kosdaq_ret_5d = (kq_close.iloc[-1] - kq_close.iloc[-6]) / kq_close.iloc[-6]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="KOSPI 지수",
            value=f"{k_close.iloc[-1]:,.2f}pt",
            delta=f"{k_close.iloc[-1] - k_close.iloc[-2]:+,.2f}pt (전일대비)"
        )
    with col2:
        st.metric(
            label="KOSPI 최근 5일 수익률",
            value=f"{kospi_ret_5d:+.2%}",
            delta="하락세 (방어주 탐색 적합)" if kospi_ret_5d < 0 else "상승/보합세"
        )
    with col3:
        st.metric(
            label="KOSDAQ 지수",
            value=f"{kq_close.iloc[-1]:,.2f}pt",
            delta=f"{kq_close.iloc[-1] - kq_close.iloc[-2]:+,.2f}pt (전일대비)"
        )
    with col4:
        st.metric(
            label="KOSDAQ 최근 5일 수익률",
            value=f"{kosdaq_ret_5d:+.2%}",
            delta="하락세 (방어주 탐색 적합)" if kosdaq_ret_5d < 0 else "상승/보합세"
        )
else:
    st.warning("시장 지수 데이터를 불러오지 못했습니다.")

# ==========================================
# 분석 실행 및 알고리즘 적용
# ==========================================
if st.sidebar.button("🔍 조건 만족 종목 탐색 시작", use_container_width=True):
    with st.spinner("1차 후보 종목 데이터 로드 중..."):
        raw_list = get_market_tickers()
        
    if not raw_list.empty:
        # 필터링 1: 시장 선택 및 시가총액, 거래대금 기준
        filtered_list = raw_list[
            (raw_list['Market'].isin(markets)) & 
            (raw_list['Marcap'] >= min_marcap * 100000000) &
            (raw_list['Amount'] >= min_amount * 100000000)
        ]
        
        filtered_list = filtered_list.sort_values(by='Amount', ascending=False).head(max_scan)
        
        tickers = filtered_list['Code'].tolist()
        ticker_name_map = dict(zip(filtered_list['Code'], filtered_list['Name']))
        ticker_market_map = dict(zip(filtered_list['Code'], filtered_list['Market']))
        ticker_amount_map = dict(zip(filtered_list['Code'], filtered_list['Amount']))
        ticker_marcap_map = dict(zip(filtered_list['Code'], filtered_list['Marcap']))
        
        st.write(f"⚙️ **1차 필터링 완료**: 시가총액 {min_marcap}억 및 거래대금 {min_amount}억 이상 대상 중 상위 **{len(tickers)}개** 종목 상세 분석 중...")
        
        histories = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total = len(tickers)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            futures = [executor.submit(fetch_stock_history, ticker, start_date_str, today_date_str) for ticker in tickers]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                ticker, df = future.result()
                if df is not None:
                    histories[ticker] = df
                
                if i % 10 == 0 or i == total - 1:
                    pct = (i + 1) / total
                    progress_bar.progress(pct)
                    status_text.text(f"📥 주가 데이터 수집 및 기술적 지표 분석 중: {i+1} / {total} 완료")
                    
        progress_bar.empty()
        status_text.empty()
        
        # 사용자 이평선 설정값 추출
        if "120일선" in support_ma_selection:
            support_window = 120
        elif "60일선" in support_ma_selection:
            support_window = 60
        elif "20일선" in support_ma_selection:
            support_window = 20
        else:
            support_window = 5
        
        # 각 조건별 만족 리스트 초기화
        passed_defensive = []   # 모든 조건을 통과한 통합 방어주 (A + B + C)
        passed_gc = []          # 조건 A: 최근 N일 골드크로스
        passed_support = []     # 조건 B: N일선 지지 (정배열 바닥)
        passed_outperform = []  # 조건 C: 지수 대비 방어력 우수
        passed_sniper = []      # 🎯 단기 저격주 후보
        
        for ticker, df in histories.items():
            try:
                # 이동평균선(MA) 계산
                df['MA_5'] = df['Close'].rolling(window=5).mean()
                df['MA_20'] = df['Close'].rolling(window=20).mean()
                df['MA_support'] = df['Close'].rolling(window=support_window).mean()
                
                # RSI 계산
                df['RSI'] = calculate_rsi(df['Close'], period=14)
                
                close_today = df['Close'].iloc[-1]
                ma5_today = df['MA_5'].iloc[-1]
                ma20_today = df['MA_20'].iloc[-1]
                ma_support_today = df['MA_support'].iloc[-1]
                rsi_today = df['RSI'].iloc[-1]
                
                # 설정된 바닥선 대비 이격도(%) 계산 (예: 105.4%)
                disparity_today = (close_today / ma_support_today) * 100
                
                # 20일선 기준 이격도(갭%) 계산
                gap_pct = ((close_today - ma20_today) / ma20_today) * 100
                
                # 1. [조건 A - 골드크로스 여부 탐지]
                has_gc, gc_date, gc_offset = detect_recent_golden_cross(df, gc_days_window)
                gc_text = "오늘 발생" if gc_offset == 0 else (f"{gc_offset}일 전" if gc_offset is not None else "-")
                
                # 2. [조건 B - 선택된 이평선 지지 여부]
                is_supported = close_today > ma_support_today
                
                # 3. [조건 C - 지수 대비 방어력]
                market = ticker_market_map[ticker]
                idx_hist = kospi_hist if market == 'KOSPI' else kosdaq_hist
                
                stock_ret_5d = (df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6]
                idx_ret_5d = (idx_hist['Close'].iloc[-1] - idx_hist['Close'].iloc[-6]) / idx_hist['Close'].iloc[-6]
                
                if idx_ret_5d < 0:
                    is_defensive = stock_ret_5d >= 0
                else:
                    is_defensive = (stock_ret_5d >= 0) or (stock_ret_5d >= idx_ret_5d)
                
                # 추가 정보: 거래량 및 3일 최대 거래대금 계산
                df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
                volume_3d_ago = df['Volume'].iloc[-4]
                vol_ma20_3d_ago = df['Volume_MA20'].iloc[-4]
                volume_today = df['Volume'].iloc[-1]
                recent_3d_vol_mean = df['Volume'].iloc[-3:].mean()
                
                # 최근 3일간의 최대 거래대금 계산 (종가 * 거래량으로 근사화)
                max_amount_3d = (df['Close'].iloc[-3:] * df['Volume'].iloc[-3:]).max()
                
                # 공통 정보 객체
                stock_info = {
                    'Code': ticker,
                    'Name': ticker_name_map[ticker],
                    'Market': market,
                    'Close': int(close_today),
                    'Amount': int(ticker_amount_map[ticker] / 100000000), # 억 단위
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
                    'Support_Window': support_window,
                    'Volume_Today': int(volume_today),
                    'Volume_3D_Ago': int(volume_3d_ago),
                    'Vol_MA20_3D_Ago': float(vol_ma20_3d_ago),
                    'Max_Amount_3D': float(max_amount_3d / 100000000) # 억 단위
                }
                
                # --- 일반 조건 필터링 ---
                passed_safe_filter = True
                if safe_mode:
                    if disparity_today > 110.0 or pd.isna(rsi_today) or rsi_today > rsi_cutoff:
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
                
                # --- 단기 저격 조건 필터링 (스윙) ---
                is_trend_aligned = ma5_today > ma20_today
                is_vol_burst_3d_ago = volume_3d_ago > vol_ma20_3d_ago
                is_vol_pulled_back = volume_today <= 0.6 * recent_3d_vol_mean
                is_rsi_cond = (not pd.isna(rsi_today)) and (45.0 <= rsi_today <= 55.0)
                
                if is_trend_aligned and is_vol_burst_3d_ago and is_vol_pulled_back and is_rsi_cond:
                    # 고점 제외(안전 모드)와의 선택적 결합: 켜져 있으면 이격도 110% 이하 필터 추가 적용
                    if not safe_mode or (disparity_today <= 110.0):
                        passed_sniper.append(stock_info.copy())
                        
            except Exception:
                pass
                
        # 각 리스트에 재무 건전성(적자 제외) 적용
        def apply_financial_filter(candidates_list):
            if not candidates_list:
                return []
            filtered = []
            for item in candidates_list:
                ticker = item['Code']
                is_deficit, per = check_is_deficit(ticker)
                item['PER'] = per
                item['Is_Deficit'] = is_deficit
                
                # 네이버 링크 주소 저장
                item['Link'] = f"https://m.stock.naver.com/domestic/stock/{ticker}"
                
                if exclude_deficit and is_deficit:
                    continue
                filtered.append(item)
            return filtered
            
        with st.spinner("📋 후보 종목들의 재무 건전성 및 바로가기 생성 중..."):
            st.session_state['passed_defensive'] = apply_financial_filter(passed_defensive)
            st.session_state['passed_gc'] = apply_financial_filter(passed_gc)
            st.session_state['passed_support'] = apply_financial_filter(passed_support)
            st.session_state['passed_outperform'] = apply_financial_filter(passed_outperform)
            st.session_state['passed_sniper'] = apply_financial_filter(passed_sniper)
            st.session_state['histories'] = histories
            
# ==========================================
# 메인화면 - 결과 탭 및 차트 출력 영역
# ==========================================
if 'passed_defensive' in st.session_state:
    # 사용자 이평선 설정값 추출
    if "120일선" in support_ma_selection:
        support_window = 120
    elif "60일선" in support_ma_selection:
        support_window = 60
    elif "20일선" in support_ma_selection:
        support_window = 20
    else:
        support_window = 5
    
    if sniper_mode:
        tab_sniper = st.tabs(["🎯 단기 저격 포트폴리오 (스윙 눌림목)"])[0]
        with tab_sniper:
            st.write("3거래일 전 대량 거래대금이 들어온 후, 오늘 거래량이 최근 3일 평균 거래량의 60% 이하로 급감하며 숨 고르기 중인 5/20 정배열 단기 스윙 종목군입니다.")
            
            def render_sniper_table(candidates_list):
                if not candidates_list:
                    st.warning("단기 저격 조건에 만족하는 종목이 없습니다. 시장 조건 검색 설정을 완화해 보세요.")
                    return pd.DataFrame()
                    
                df = pd.DataFrame(candidates_list)
                df = df.sort_values(by='Max_Amount_3D', ascending=False) # 3일 최대 거래대금 순 정렬
                
                # 출력 데이터 가공
                df_show = df[[
                    'Code', 'Name', 'Market', 'Close', 'Max_Amount_3D', 'Volume_Today', 'Volume_3D_Ago', 'RSI', 'Disparity', 'Gap_Pct', 'PER', 'Link'
                ]].copy()
                
                # 컬럼 한글화
                df_show.columns = [
                    '종목코드', '종목명', '소속시장', '현재가(원)', '3일간최대거래대금(억 원)', '오늘거래량(주)', '3일전거래량(주)', 'RSI', '이격도(%)', '20일선이격도', 'PER', '네이버 증권'
                ]
                
                # 서식 변환
                df_show['3일간최대거래대금(억 원)'] = df_show['3일간최대거래대금(억 원)'].map(lambda x: f"{x:,.1f}억")
                df_show['오늘거래량(주)'] = df_show['오늘거래량(주)'].map(lambda x: f"{x:,}")
                df_show['3일전거래량(주)'] = df_show['3일전거래량(주)'].map(lambda x: f"{x:,}")
                df_show['20일선이격도'] = df_show['20일선이격도'].map(lambda x: f"{x:+.2f}%")
                df_show['이격도(%)'] = df_show['이격도(%)'].map(lambda x: f"{x:.1f}%")
                df_show['RSI'] = df_show['RSI'].map(lambda x: f"{x:.1f}" if not pd.isna(x) else "-")
                df_show['현재가(원)'] = df_show['현재가(원)'].map(lambda x: f"{x:,}")
                
                # LinkColumn을 활성화한 데이터프레임 렌더링
                st.dataframe(
                    df_show, 
                    column_config={
                        "네이버 증권": st.column_config.LinkColumn(
                            "네이버 증권",
                            help="클릭하면 해당 종목의 네이버 증권 페이지로 이동합니다.",
                            display_text="바로가기 ↗"
                        )
                    },
                    use_container_width=True, 
                    hide_index=True
                )
                return df
                
            df_sniper = render_sniper_table(st.session_state.get('passed_sniper', []))
    else:
        # 탭 메뉴 정의
        tab_def, tab_gc, tab_support, tab_out = st.tabs([
            "🛡️ 통합 포트폴리오 (조건 A+B+C)",
            "⚡ 골든크로스 종목 (조건 A)",
            f"📈 {support_window}일선 지지 종목 (조건 B)",
            "💪 지수 대비 강세 종목 (조건 C)"
        ])
        
        # 탭별 데이터를 데이터프레임 형식으로 렌더링하는 헬퍼 함수
        def render_candidate_table(candidates_list):
            if not candidates_list:
                st.warning("이 조건에 만족하는 종목이 없습니다. 사이드바 설정의 검색 강도를 완화해 보세요.")
                return pd.DataFrame()
                
            df = pd.DataFrame(candidates_list)
            df = df.sort_values(by='Amount', ascending=False) # 일 거래대금 순 정렬
            
            # 출력 데이터 가공
            df_show = df[[
                'Code', 'Name', 'Market', 'Close', 'Amount', 'Marcap', 'Stock_Return_5D', 'Index_Return_5D', 'Disparity', 'RSI', 'Gap_Pct', 'Cross_Days_Ago', 'PER', 'Link'
            ]].copy()
            
            # 컬럼 한글화
            df_show.columns = [
                '종목코드', '종목명', '소속시장', '현재가(원)', '일거래대금(억 원)', '시가총액', '5일 종목수익률', '5일 지수수익률', '이격도(%)', 'RSI', '20일선이격도', '최근 크로스일', 'PER', '네이버 증권'
            ]
            
            # 서식 변환
            df_show['5일 종목수익률'] = df_show['5일 종목수익률'].map(lambda x: f"{x:+.2%}")
            df_show['5일 지수수익률'] = df_show['5일 지수수익률'].map(lambda x: f"{x:+.2%}")
            df_show['20일선이격도'] = df_show['20일선이격도'].map(lambda x: f"{x:+.2f}%")
            df_show['이격도(%)'] = df_show['이격도(%)'].map(lambda x: f"{x:.1f}%")
            df_show['RSI'] = df_show['RSI'].map(lambda x: f"{x:.1f}" if not pd.isna(x) else "-")
            df_show['현재가(원)'] = df_show['현재가(원)'].map(lambda x: f"{x:,}")
            df_show['일거래대금(억 원)'] = df_show['일거래대금(억 원)'].map(lambda x: f"{x:,}")
            df_show['시가총액'] = df_show['시가총액'].apply(format_marcap)
            
            # LinkColumn을 활성화한 데이터프레임 렌더링
            st.dataframe(
                df_show, 
                column_config={
                    "네이버 증권": st.column_config.LinkColumn(
                        "네이버 증권",
                        help="클릭하면 해당 종목의 네이버 증권 페이지로 이동합니다.",
                        display_text="바로가기 ↗"
                    )
                },
                use_container_width=True, 
                hide_index=True
            )
            return df
            
        with tab_def:
            st.write(f"골드크로스, {support_window}일 이평선 지지, 그리고 지수 대비 가격 방어력을 동시에 만족하는 통합 헤지 포트폴리오 후보 종목군입니다.")
            df_def = render_candidate_table(st.session_state['passed_defensive'])
            
        with tab_gc:
            st.write("최근 영업일 기준 지정된 범위 이내에 5일 이동평균선이 20일 이동평균선을 골든크로스하여 단기 상승 추세로 전환된 종목군입니다.")
            df_gc = render_candidate_table(st.session_state['passed_gc'])
            
        with tab_support:
            ma_type_desc = "초단기" if support_window == 5 else "단기" if support_window == 20 else "중기" if support_window == 60 else "장기"
            st.write(f"현재 주가가 {ma_type_desc} 이동평균선({support_window}일선) 위에 위치하여 하방 지지력이 확인되는 기술적 정배열 초입 종목군입니다.")
            df_support = render_candidate_table(st.session_state['passed_support'])
            
        with tab_out:
            st.write("최근 5거래일 동안 시장 지수(KOSPI/KOSDAQ)의 하락에도 불구하고 가격 방어력 또는 시장 대비 초과 수익을 기록한 강세 종목군입니다.")
            df_out = render_candidate_table(st.session_state['passed_outperform'])
            
    # 하단 비교 그래프 시각화 영역
    st.markdown("---")
    st.subheader("종목 및 지수 가격 추이 상대 분석 (100% 정규화)")
    
    # 모든 탐지된 유니크 종목 수집하여 차트 대행 리스트업
    all_found = []
    for category in ['passed_defensive', 'passed_gc', 'passed_support', 'passed_outperform', 'passed_sniper']:
        if category in st.session_state:
            all_found.extend(st.session_state[category])
        
    if all_found:
        # 중복 제거
        unique_candidates = list({item['Code']: item for item in all_found}.values())
        unique_candidates = sorted(unique_candidates, key=lambda x: x['Amount'], reverse=True)
        
        stock_options = [f"{row['Name']} ({row['Code']})" for row in unique_candidates]
        selected_stock_str = st.selectbox("비교 분석할 종목을 선택하세요", options=stock_options)
        
        if selected_stock_str:
            sel_code = selected_stock_str.split('(')[1].replace(')', '').strip()
            sel_name = selected_stock_str.split('(')[0].strip()
            
            # 주가 데이터 가져오기
            sel_df = st.session_state['histories'].get(sel_code)
            
            # 소속 시장 및 PER 확인
            sel_item = next(item for item in unique_candidates if item['Code'] == sel_code)
            sel_market = sel_item['Market']
            sel_index_df = kospi_hist if sel_market == 'KOSPI' else kosdaq_hist
            
            if sel_df is not None and not sel_df.empty and not sel_index_df.empty:
                stock_sub = sel_df.tail(chart_days).copy()
                index_sub = sel_index_df.tail(chart_days).copy()
                
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
                    
                    fig.update_layout(
                        title=dict(
                            text=f"{sel_name} vs {sel_market} 지수 상대 수익률 추이 (최근 {chart_days} 거래일)",
                            x=0.5,
                            font=dict(size=18, color='#F8FAFC')
                        ),
                        xaxis=dict(title="날짜", gridcolor='#1E293B', tickfont=dict(color='#94A3B8')),
                        yaxis=dict(title="상대 가격 (시작점 = 100%)", gridcolor='#1E293B', tickfont=dict(color='#94A3B8')),
                        legend=dict(
                            x=0.01,
                            y=0.99,
                            bgcolor='rgba(15, 23, 42, 0.8)',
                            bordercolor='#334155',
                            borderwidth=1,
                            font=dict(color='#F8FAFC')
                        ),
                        plot_bgcolor='#0F172A',
                        paper_bgcolor='#0F172A',
                        margin=dict(l=40, r=40, t=60, b=40),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 요약 정보 카드 출력
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div class="dashboard-card">
                            <span class="metric-label">기업 정보</span><br>
                            <span class="metric-value">{sel_name} ({sel_code})</span>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="dashboard-card">
                            <span class="metric-label">종목 PER</span><br>
                            <span class="metric-value">{sel_item['PER']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        # 5일 초과 수익률 계산
                        excess_ret = sel_item['Stock_Return_5D'] - sel_item['Index_Return_5D']
                        color = "#10b981" if excess_ret >= 0 else "#ef4444"
                        st.markdown(f"""
                        <div class="dashboard-card">
                            <span class="metric-label">지수 대비 5일 성과</span><br>
                            <span class="metric-value" style="color:{color};">+{excess_ret:+.2%} 초과수익</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("데이터 매칭 오류가 발생했습니다.")
    else:
        st.info("검색 완료된 종목이 없습니다. 먼저 탐색을 시작해 주세요!")
else:
    st.info("왼쪽 사이드바의 조건을 설정한 후 **[조건 만족 종목 탐색 시작]** 버튼을 눌러 탐색을 시작해 주세요!")
