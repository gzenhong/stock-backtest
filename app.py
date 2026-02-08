import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. 網頁頁面配置
st.set_page_config(page_title="多股績效與回撤區間分析", layout="wide")
st.title("⚖️ 多支股票投資對比 (含 MDD 發生區間)")

# 2. 側邊欄設定
with st.sidebar:
    st.header("1. 設定投資參數")
    start_date = st.date_input("開始日期", value=datetime(2009, 12, 31), min_value=datetime(1900, 1, 1), max_value=datetime.today())
    end_date = st.date_input("結束日期", value=datetime.today(), min_value=datetime(1900, 1, 1), max_value=datetime.today())
    initial_capital = 10000 

    st.divider()
    st.header("2. 輸入股票代號")
    # 預設顯示常用代號
    input_df = pd.DataFrame([
        {"代號": "2330.TW"}, {"代號": "0050.TW"}, {"代號": "QQQ"}, 
        {"代號": "SPY"}, {"代號": ""}, {"代號": ""}, 
        {"代號": ""}, {"代號": ""}, {"代號": ""}
    ])
    edited_df = st.data_editor(input_df, num_rows="fixed", hide_index=True)

    symbols = [
        str(s["代號"]).strip().upper() 
        for s in edited_df.to_dict('records') 
        if s["代號"] is not None and str(s["代號"]).strip() != ""
    ]

    analyze_btn = st.button("🚀 開始執行比較分析")

# 3. 核心處理函數
def get_adjusted_data(symbol, start, end):
    # 多抓緩衝日期以確保能取得前一年的最後一日收盤價
    buffer_start = start - timedelta(days=400)
    data = yf.download(symbol, start=buffer_start, end=end, auto_adjust=False, progress=False)
    if data.empty: return None

    # 處理 yfinance 可能回傳的 MultiIndex 欄位
    if isinstance(data.columns, pd.MultiIndex):
        series = data["Adj Close"][symbol] if "Adj Close" in data.columns.get_level_values(0) else data["Close"][symbol]
    else:
        series = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]

    series = series.dropna().copy()
    
    # 台股分割修正邏輯
    if symbol == "0050.TW":
        series.loc[series.index < pd.Timestamp("2014-01-02")] /= 4
    elif symbol == "0052.TW":
        series.loc[series.index < pd.Timestamp("2025-11-17")] /= 7
    return series

# 4. 主要執行邏輯
if analyze_btn and symbols:
    try:
        raw_series_dict = {}
        stock_start_info = {}

        with st.spinner('正在抓取數據並對齊區間...'):
            for sym in symbols:
                res = get_adjusted_data(sym, start_date, end_date)
                if res is not None:
                    # 找出在理想開始日期後的第一筆有效資料日期
                    actual_data_in_range = res[res.index >= pd.Timestamp(start_date)]
                    if not actual_data_in_range.empty:
                        raw_series_dict[sym] = res
                        stock_start_info[sym] = actual_data_in_range.index[0]

        if raw_series_dict:
            # ✨ 核心修正：精準找出「最晚開始」的日期 (瓶頸)
            latest_start_date = max(stock_start_info.values())
            # ✨ 核心修正：精準反查是哪一支股票限制了起始日
            reference_stock = [s for s, d in stock_start_info.items() if d == latest_start_date][0]
            common_end_date = min([s.index[-1] for s in raw_series_dict.values()])

            st.success(f"📌 **同步計算基準：** 由於各股票歷史資料時間不同，已取最短共同區間進行對比。")
            st.info(f"📅 **實際回測期間：** `{latest_start_date.strftime('%Y-%m-%d')}` 至 `{common_end_date.strftime('%Y-%m-%d')}` (以資料日期最短的 `{reference_stock}` 為準)")

            all_assets_df = pd.DataFrame()
            all_roi_df = pd.DataFrame()
            summary_data = []

            for sym, series in raw_series_dict.items():
                # 對齊區間
                invest_series = series[series.index >= latest_start_date]
                if invest_series.empty: continue

                # --- ✨ 計算最大回撤 (MDD) ---
                rolling_max = invest_series.cummax()
                drawdowns = (invest_series - rolling_max) / rolling_max
                max_drawdown = drawdowns.min()
                mdd_end_date = drawdowns.idxmin()
                mdd_start_date = invest_series[:mdd_end_date].idxmax()
                mdd_period = f"{mdd_start_date.strftime('%Y-%m-%d')} ~ {mdd_end_date.strftime('%Y-%m-%d')}"

                # --- ✨ 年度報酬與資產計算 (精確跨年對齊) ---
                years = sorted(list(set(invest_series.index.year)))
                temp_assets, temp_rois = {}, {}
                
                # 以對齊日的第一筆價格作為 $10,000 的基準
                s_price = float(invest_series.iloc[0])
                current_assets = initial_capital

                for year in years:
                    year_data = invest_series[invest_series.index.year == year]
                    if year_data.empty: continue
                    
                    # 尋找基準價格：嘗試找「前一年最後一個交易日」
                    prev_year_data = series[series.index.year < year]
                    if not prev_year_data.empty:
                        base_price = float(prev_year_data.iloc[-1])
                    else:
                        base_price = s_price # 資料起始年

                    year_end_price = float(year_data.iloc[-1])
                    year_roi = (year_end_price - base_price) / base_price
                    
                    # 更新資產與報酬
                    current_assets = (year_end_price / s_price) * initial_capital
                    temp_assets[year] = round(current_assets, 0)
                    temp_rois[year] = f"{year_roi * 100:.2f}%"

                # 建立資產曲線 (供繪圖使用)
                asset_curve = (invest_series / s_price) * initial_capital
                all_assets_df[sym] = asset_curve
                all_roi_df[sym] = pd.Series(temp_rois)

                # 計算總指標
                total_roi = (current_assets - initial_capital) / initial_capital
                days = (invest_series.index[-1] - invest_series.index[0]).days
                cagr = (current_assets / initial_capital) ** (365.25 / days) - 1 if days > 0 else 0

                summary_data.append({
                    "股票代號": sym,
                    "最終資產": f"${current_assets:,.0f}",
                    "累積報酬(ROI)": f"{total_roi * 100:.2f}%",
                    "年化(CAGR)": f"{cagr * 100:.2f}%",
                    "最大回撤(MDD)": f"{max_drawdown * 100:.2f}%",
                    "MDD 期間 (高點 → 低點)": mdd_period
                })

            # 呈現圖表與表格
            st.subheader(f"📊 多股累積資產成長圖 (起始資產 ${initial_capital:,.0f})")
            st.line_chart(all_assets_df)

            st.subheader("📋 績效與風險總結 (對齊區間)")
            st.table(pd.DataFrame(summary_data).set_index("股票代號"))

            st.divider()
            st.subheader("📅 年度報酬率明細 (%)")
            st.write("💡 計算基準：前一年最後一個交易日 vs 當年最後一個交易日")
            st.dataframe(all_roi_df.T, use_container_width=True)
            
        else:
            st.error("查無有效數據，請檢查代號或日期設定。")

    except Exception as e:
        st.error(f"分析失敗: {e}")
