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
    input_df = pd.DataFrame([
        {"代號": "2330.TW"}, {"代號": "0050.TW"}, {"代號": "QQQ"}, 
        {"代號": "SPY"}, {"代號": ""}, {"代號": ""}, 
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
    buffer_start = start - timedelta(days=400)
    data = yf.download(symbol, start=buffer_start, end=end, auto_adjust=False, progress=False)
    if data.empty: return None

    if isinstance(data.columns, pd.MultiIndex):
        series = data["Adj Close"][symbol] if "Adj Close" in data.columns.get_level_values(0) else data["Close"][symbol]
    else:
        series = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]

    series = series.dropna().copy()
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

        with st.spinner('正在精確計算對齊區間績效...'):
            for sym in symbols:
                res = get_adjusted_data(sym, start_date, end_date)
                if res is not None:
                    actual_in_range = res[res.index >= pd.Timestamp(start_date)]
                    if not actual_in_range.empty:
                        raw_series_dict[sym] = res
                        stock_start_info[sym] = actual_in_range.index[0]

        if raw_series_dict:
            # ✨ 判定基準日 (找出最晚開始有資料的那一天)
            latest_start_date = max(stock_start_info.values())
            # ✨ 判定基準股票 (解決標示錯誤)
            reference_stock = [s for s, d in stock_start_info.items() if d == latest_start_date][0]
            common_end_date = min([s.index[-1] for s in raw_series_dict.values()])

            st.success(f"📌 **同步計算基準：** 由於各股票歷史資料時間不同，已取最短共同區間進行對比。")
            st.info(f"📅 **實際回測期間：** `{latest_start_date.strftime('%Y-%m-%d')}` 至 `{common_end_date.strftime('%Y-%m-%d')}` (以 `{reference_stock}` 為準)")

            all_assets_df = pd.DataFrame()
            all_roi_df = pd.DataFrame()
            summary_data = []

            for sym, series in raw_series_dict.items():
                # 核心：強制取共同起始日之後的資料
                invest_series = series[series.index >= latest_start_date]
                if invest_series.empty: continue

                # ✨ 核心修正：累積報酬必須以「共同起始日」的價格為 100% 基準
                base_price_at_start = float(invest_series.iloc[0])
                
                # 計算資產曲線：起始日當天大家都是 $10,000
                asset_curve = (invest_series / base_price_at_start) * initial_capital
                all_assets_df[sym] = asset_curve

                # --- 1. 計算年度報酬 (跨年對齊) ---
                years = sorted(list(set(invest_series.index.year)))
                temp_rois = {}
                for year in years:
                    year_data = invest_series[invest_series.index.year == year]
                    prev_year_data = series[series.index.year < year]
                    # 邏輯：前一年最後一日 vs 該年最後一日
                    ref_price = float(prev_year_data.iloc[-1]) if not prev_year_data.empty else base_price_at_start
                    y_end_price = float(year_data.iloc[-1])
                    temp_rois[year] = f"{((y_end_price - ref_price) / ref_price) * 100:.2f}%"
                all_roi_df[sym] = pd.Series(temp_rois)

                # --- 2. 計算 MDD ---
                rolling_max = invest_series.cummax()
                drawdowns = (invest_series - rolling_max) / rolling_max
                max_drawdown = drawdowns.min()
                mdd_end_date = drawdowns.idxmin()
                mdd_start_date = invest_series[:mdd_end_date].idxmax()

                # --- 3. 計算總指標 (ROI 也基於對齊區間) ---
                final_asset = float(asset_curve.iloc[-1])
                # ✨ 這裡的累積報酬 (ROI) 已經正確限制在最短區間內
                total_roi_aligned = (final_asset - initial_capital) / initial_capital
                
                days = (invest_series.index[-1] - invest_series.index[0]).days
                cagr = (final_asset / initial_capital) ** (365.25 / days) - 1 if days > 0 else 0

                summary_data.append({
                    "股票代號": sym,
                    "最終資產": f"${final_asset:,.0f}",
                    "累積報酬(ROI)": f"{total_roi_aligned * 100:.2f}%",
                    "年化(CAGR)": f"{cagr * 100:.2f}%",
                    "最大回撤(MDD)": f"{max_drawdown * 100:.2f}%",
                    "MDD 發生期間": f"{mdd_start_date.strftime('%Y-%m-%d')} ~ {mdd_end_date.strftime('%Y-%m-%d')}"
                })

            # 繪圖與表格呈現
            st.subheader(f"📊 多股累積資產成長圖 (起始資產 ${initial_capital:,.0f})")
            st.line_chart(all_assets_df)

            st.subheader("📋 績效與風險總結 (對齊區間)")
            st.table(pd.DataFrame(summary_data).set_index("股票代號"))

            st.divider()
            st.subheader("📅 年度報酬率明細 (%)")
            st.dataframe(all_roi_df.T, use_container_width=True)
            
        else:
            st.error("查無數據。")

    except Exception as e:
        st.error(f"分析失敗: {e}")
