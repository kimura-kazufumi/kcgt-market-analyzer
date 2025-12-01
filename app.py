import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- ページ設定 ---
st.set_page_config(
    page_title="KCGT Market Analyzer",
    page_icon="🌌",
    layout="wide"
)

# --- スタイル調整 ---
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main {
        background: #0e1117;
        color: #fafafa;
    }
    h1, h2, h3 {
        color: #00BFFF;
    }
    .stAlert {
        background-color: #330f0f;
        color: #ff4b4b;
        border: 1px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# --- 関数定義: KCGT計算ロジック (対数モード) ---
def calculate_kcgt_metrics(price_series, window=20):
    # 1. 対数変換 (スケール不変性のため)
    log_prices = np.log(price_series)
    
    # 2. 曲率計算 (2階微分)
    curvature = np.diff(log_prices, n=2)
    curvature = np.pad(curvature, (2, 0), 'constant', constant_values=0)
    
    # 3. 幾何学的粗さ (Roughness) = Σ界面のストレス
    series_curvature = pd.Series(curvature)
    roughness = series_curvature.rolling(window=window).std()
    
    # 4. 表示用にスケーリング
    return roughness * 1000

# --- 関数定義: データ取得 ---
@st.cache_data(ttl=60) # リアルタイム性を重視してキャッシュ時間を60秒に短縮
def get_data(ticker, interval, period):
    try:
        # period引数を使って取得（start/endよりも柔軟）
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        return data
    except Exception as e:
        return None

# --- メインコンテンツ ---

# ヘッダー
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🌌 KCGT Market Analyzer")
    st.markdown("**構界宇宙幾何理論 (Kōkai Cosmic Geometry Theory) に基づく市場構造診断**")
with col2:
    st.image("https://img.icons8.com/color/96/000000/physics.png", width=80)

# 免責事項
st.info("⚠️ **免責事項:** 本ツールは物理幾何学モデルの実験的応用です。投資助言ではありません。相場の「幾何学的な無理（歪み）」を可視化するものであり、将来の価格を保証するものではありません。")

# --- サイドバー設定 ---
ticker_input = st.sidebar.text_input("銘柄コード", value="BTC-USD")

# 時間足の選択 (New!)
interval = st.sidebar.selectbox(
    "時間足 (Interval)",
    options=["1d", "1h", "15m", "5m", "1m"],
    index=0,
    help="1m, 5m 等は直近のデータしか取得できない場合があります。"
)

# 期間設定（分足の場合は期間を短く自動調整するロジック）
if interval in ["1m", "5m", "15m", "1h"]:
    period = "7d" # 分足は最大7日〜60日程度しか取れない制限があるため
    st.sidebar.info(f"※ 短期足 ({interval}) 選択中は、直近 {period} のデータを解析します。")
else:
    period = "2y" # 日足なら2年分
    # 日付指定は日足の時のみ有効にするなどの制御も可能ですが、今回は簡易的にperiod指定を使います
# パラメータ調整
st.sidebar.markdown("---")
st.sidebar.subheader("📐 解析パラメータ")
window_size = st.sidebar.slider("平滑化ウィンドウ (日)", 10, 50, 20, help="値を大きくすると長期トレンド重視、小さくすると敏感になります。")
sensitivity = st.sidebar.slider("検知感度 (σ)", 1.0, 4.0, 2.0, help="閾値の高さ。値を下げると警告が出やすくなります。")

# --- 解析実行 ---
if ticker_input:
    with st.spinner(f'{ticker_input} の構界データを取得中...'):
        # 引数を変更
        df = get_data(ticker_input, interval, period)

    if df is not None and not df.empty:
        # データ準備
        prices = df['Close'].values.flatten() if df['Close'].ndim > 1 else df['Close'].values
        dates = df.index
        
        # KCGT計算
        stress_index = calculate_kcgt_metrics(prices, window=window_size)
        
        # 閾値の動的計算 (キャリブレーション: 最初の1/4期間を基準とする)
        calib_len = max(30, int(len(stress_index) * 0.25))
        base_stress = stress_index[:calib_len]
        # NaNを除去
        base_stress = base_stress[~np.isnan(base_stress)]
        
        if len(base_stress) > 0:
            threshold = np.mean(base_stress) + sensitivity * np.std(base_stress)
            # 最低ライン設定（あまりに動きがない銘柄での誤検知防止）
            threshold = max(threshold, 0.01)
        else:
            threshold = 1.0 # フォールバック

        # 危険判定
        danger_mask = stress_index > threshold
        
        # 最新の状態判定
        latest_stress = stress_index.iloc[-1]
        is_danger = latest_stress > threshold
        
        # --- 結果表示エリア ---
        
        # メトリクス表示
        m1, m2, m3 = st.columns(3)
        current_price = prices[-1]
        m1.metric("現在価格", f"{current_price:,.2f}")
        m2.metric("KCGTストレス指数", f"{latest_stress:.2f}", delta=f"{latest_stress-threshold:.2f} (vs Limit)", delta_color="inverse")
        
        if is_danger:
            m3.error("**構造的警告 (WARNING)**")
            st.warning(f"🚨 **警告:** 現在、市場の幾何学的ストレスが限界値を超えています。$\Sigma$界面の相転移（トレンド崩壊）のリスクが高まっています。")
        else:
            m3.success("**構造的安定 (STABLE)**")
            st.success(f"✅ **安定:** 現在、市場の構造は幾何学的許容範囲内に収まっています。")

        # --- メインチャート描画 ---
        st.subheader(f"📊 KCGT 構造診断チャート: {ticker_input}")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # 上段: 価格と警告
        ax1.plot(dates, prices, color='#00BFFF', linewidth=1.5, label='Price (Σ Interface)')
        ax1.set_title('Price Trend & Structural Warning Zones', fontsize=12, color='white')
        ax1.grid(True, alpha=0.1, color='white')
        ax1.set_facecolor('#0e1117')
        
        # 警告ゾーンの描画
        y_min, y_max = ax1.get_ylim()
        ax1.fill_between(dates, y_min, y_max, where=danger_mask, color='#ff4b4b', alpha=0.2, label='KCGT Warning Zone')
        
        # 軸の色調整
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y', colors='white')
        for spine in ax1.spines.values(): spine.set_edgecolor('white')
        ax1.legend(loc='upper left', facecolor='#0e1117', labelcolor='white')

        # 下段: ストレス指数
        ax2.plot(dates, stress_index, color='#DDA0DD', linewidth=1.5, label='Geometric Roughness')
        ax2.axhline(y=threshold, color='#FFA500', linestyle='--', linewidth=1.5, label='Elastic Limit (Threshold)')
        ax2.fill_between(dates, 0, stress_index, color='#DDA0DD', alpha=0.1)
        
        ax2.set_title('Internal Geometric Stress (log-space curvature)', fontsize=12, color='white')
        ax2.grid(True, alpha=0.1, color='white')
        ax2.set_facecolor('#0e1117')
        ax2.set_ylabel('Stress Level', color='white')
        
        ax2.tick_params(axis='x', colors='white')
        ax2.tick_params(axis='y', colors='white')
        for spine in ax2.spines.values(): spine.set_edgecolor('white')
        ax2.legend(loc='upper left', facecolor='#0e1117', labelcolor='white')

        fig.patch.set_facecolor('#0e1117')
        plt.tight_layout()
        st.pyplot(fig)

        # --- 理論解説セクション ---
        with st.expander("📚 **理論解説: なぜ「崩壊」が予知できるのか？**"):
            st.markdown("""
            ### 構界宇宙幾何理論 (KCGT) による市場解釈
            
            KCGTにおいて、市場価格の推移は「時間軸上の1次元の線」ではなく、**「構界 $\Sigma$（界面）の幾何学的な形状」**として解釈されます。
            
            1.  **膨張エネルギー ($\Delta^+$):** 買い圧力。界面を押し広げようとする力。
            2.  **収縮エネルギー ($\Delta^-$):** 売り圧力。界面を引き戻そうとする力。
            3.  **幾何学的ストレス (Roughness):** 上記のグラフ（紫線）です。
            
            **通常の指標との違い:**
            多くの指標は「価格が下がった」ときに反応しますが、KCGTは**「価格は上がっているが、その上がり方が『汚い（幾何学的に無理がある）』」**ときに反応します。
            
            * **警告ゾーン (赤):** $\Sigma$界面の歪みが「弾性限界」を超えた状態です。ここでは、わずかな衝撃で「相転移（暴落）」が発生する確率が極めて高くなります。
            * **サイレント期間:** バブルの最終局面では、抵抗がなくなり一時的にストレスが下がることがあります（慣性飛行）。チャート上の警告が消えた直後の最高値更新は、最も警戒すべきシグナルです。
            """)

    else:
        st.error(f"データが見つかりませんでした。銘柄コード '{ticker_input}' を確認してください。")
        st.markdown("主なコード例: `^N225` (日経平均), `^GSPC` (S&P500), `BTC-USD`, `ETH-USD`, `7203.T` (トヨタ自動車)")

# フッター
st.markdown("---")
st.markdown("© 2025 KCGT Research Lab. | Powered by Python & Streamlit")
