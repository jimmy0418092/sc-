import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# 設置頁面標題和佈局
st.set_page_config(page_title="Invest Mania", layout="wide")

# 自定義樣式
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }

    .big-font {
        font-size:30px !important;
        background: -webkit-linear-gradient(#FF5733, #FFC300);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        font-size: 20px;
        padding: 10px 24px;
    }

    .highlight-positive {
        background-color: #90EE90;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .highlight-negative {
        background-color: #FFCCCB;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .highlight-neutral {
        background-color: #FFFFE0;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 15px;
    }

    .stTextInput > div > input {
        background-color: #f1f1f1;
        border-radius: 8px;
        padding: 8px;
        font-size: 16px;
    }

    .sidebar .sidebar-content {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True
)

# 定義 get_matrix_and_state 函數
def get_matrix_and_state(code, day_avgs):
    st.write(f"股票代碼: {code}")
    
    # 假設你已經有該股票的數據
    df = pd.read_csv(os.path.join('data', f'{code}.csv'))

    # 填充日期數據
    if 'date' not in df.columns:
        df['date'] = pd.date_range(start='2023-01-01', periods=len(df))
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    df['Day_Avg'] = (df['open'] + df['close']) / 2
    df['Pct_Change'] = df['Day_Avg'].pct_change() * 100
    df['Pct_Change'].fillna(df['Pct_Change'].rolling(window=3, min_periods=1).mean(), inplace=True)

    # 根據百分比變化範圍定義狀態
    def categorize_state(change):
        if change > 6:
            return '++'
        elif change > 0.5:
            return '+'
        elif change >= -0.5 and change <= 0.5:
            return '0'
        elif change > -6:
            return '-'
        else:
            return '--'

    df['State'] = df['Pct_Change'].apply(categorize_state)

    states = ['++', '+', '0', '-', '--']
    transition_matrix = pd.DataFrame(0, index=states, columns=states)
    
    for (i, current_state), (_, next_state) in zip(df['State'].shift().items(), df['State'].items()):
        if pd.notna(current_state) and pd.notna(next_state):
            transition_matrix.at[current_state, next_state] += 1

    transition_matrix.fillna(0, inplace=True)
    transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0)

    P = transition_matrix.values
    evals, evecs = np.linalg.eig(P.T)
    steady_state = np.real(evecs[:, np.isclose(evals, 1)])
    steady_state = steady_state / steady_state.sum()
    steady_state = steady_state.flatten()
    
    steady_state_probs = pd.Series(steady_state, index=states)

    return df, transition_matrix, steady_state_probs, 1/steady_state_probs

# 主選單
option = st.selectbox("選擇您要進行的操作：", ["進行股票預測", "股票比較", "查看使用說明"])

# 股票預測功能
if option == "進行股票預測":
    
    st.sidebar.title("股票預測輸入")
    code = st.sidebar.text_input("輸入股票代碼:")
    num_days = st.sidebar.number_input("輸入你要輸入的日均價天數", min_value=1, max_value=30, value=5)
    day_avgs = [st.sidebar.number_input(f"Day {i+1}:", value=0.0) for i in range(num_days)]

    if st.sidebar.button("預測"):
        df, transition_matrix, steady_state_probs, return_days = get_matrix_and_state(code, day_avgs)

        # 計算最後一個日均價的變化百分比，決定最後狀態
        last_change = (day_avgs[-1] - day_avgs[-2]) / day_avgs[-2] * 100

        def categorize_state(change):
            if change > 6:
                return '++'
            elif change > 0.5:
                return '+'
            elif change >= -0.5 and change <= 0.5:
                return '0'
            elif change > -6:
                return '-'
            else:
                return '--'

        last_state = categorize_state(last_change)

        # 預測未來狀態
        next_state_probabilities = transition_matrix.loc[last_state]
        predicted_next_state = next_state_probabilities.idxmax()

        # 根據預測結果顯示動態樣式
        if predicted_next_state == '++':
            st.markdown('<div class="highlight-positive big-font">預測結果為大漲</div>', unsafe_allow_html=True)
        elif predicted_next_state == '+':
            st.markdown('<div class="highlight-positive big-font">預測結果為漲</div>', unsafe_allow_html=True)
        elif predicted_next_state == '0':
            st.markdown('<div class="highlight-neutral big-font">預測結果為持平</div>', unsafe_allow_html=True)
        elif predicted_next_state == '-':
            st.markdown('<div class="highlight-negative big-font">預測結果為跌</div>', unsafe_allow_html=True)
        elif predicted_next_state == '--':
            st.markdown('<div class="highlight-negative big-font">預測結果為大跌</div>', unsafe_allow_html=True)

        # Yahoo 股市連結
        yahoo_url = f"https://tw.stock.yahoo.com/quote/{code}"
        st.write(f"### 預測結果: [🔗 Yahoo 股市 {code}](<{yahoo_url}>)", unsafe_allow_html=True)

        # 顯示股票價格走勢圖
        st.write("### 股票價格走勢圖")
        plt.figure(figsize=(10, 6))
        plt.plot(df['date'], df['Day_Avg'], label='平均價格', marker='o', linestyle='-', color='blue')
        plt.fill_between(df['date'], df['Day_Avg'], color="skyblue", alpha=0.3)
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(plt)

        # 顯示 Transition Matrix
        st.write("### Transition Matrix:")
        st.dataframe(transition_matrix)

        # 顯示穩態機率
        st.write("### Steady State Probabilities:")
        st.dataframe(steady_state_probs)

        # 顯示回轉天數
        st.write("### Return Days:")
        st.write(return_days)

# 股票比較功能
elif option == "股票比較":
    st.sidebar.title("股票比較")

    # 股票 1 的輸入
    st.sidebar.subheader("股票 1")
    code1 = st.sidebar.text_input("輸入股票代碼 1:")
    day_avgs1 = [st.sidebar.number_input(f"股票 1 - Day {i+1}:", value=0.0) for i in range(5)]

    # 股票 2 的輸入
    st.sidebar.subheader("股票 2")
    code2 = st.sidebar.text_input("輸入股票代碼 2:")
    day_avgs2 = [st.sidebar.number_input(f"股票 2 - Day {i+1}:", value=0.0) for i in range(5)]

    if st.sidebar.button("比較"):
        df1, transition_matrix1, steady_state_probs1, return_days1 = get_matrix_and_state(code1, day_avgs1)
        df2, transition_matrix2, steady_state_probs2, return_days2 = get_matrix_and_state(code2, day_avgs2)

        # 顯示比較表格
        comparison_data = {
            "股票代碼": [code1, code2],
            "最近狀態": [df1['State'].iloc[-1], df2['State'].iloc[-1]],
            "穩態機率（++）": [f"{steady_state_probs1['++']:.2f}", f"{steady_state_probs2['++']:.2f}"],
            "穩態機率（+）": [f"{steady_state_probs1['+']:.2f}", f"{steady_state_probs2['+']:.2f}"],
            "穩態機率（0）": [f"{steady_state_probs1['0']:.2f}", f"{steady_state_probs2['0']:.2f}"],
            "穩態機率（-）": [f"{steady_state_probs1['-']:.2f}", f"{steady_state_probs2['-']:.2f}"],
            "穩態機率（--）": [f"{steady_state_probs1['--']:.2f}", f"{steady_state_probs2['--']:.2f}"],
            "回轉天數": [f"{return_days1.min():.2f}", f"{return_days2.min():.2f}"]
        }

        comparison_df = pd.DataFrame(comparison_data)
        st.write("### 股票比較結果")
        st.dataframe(comparison_df)

        # 股票 1 走勢圖
        st.write("### 股票 1 走勢圖")
        plt.figure(figsize=(10, 6))
        plt.plot(df1['date'], df1['Day_Avg'], label=f'{code1} 平均價格', marker='o', linestyle='-', color='blue')
        plt.fill_between(df1['date'], df1['Day_Avg'], color="skyblue", alpha=0.3)
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(plt)

        # 股票 2 走勢圖
        st.write("### 股票 2 走勢圖")
        plt.figure(figsize=(10, 6))
        plt.plot(df2['date'], df2['Day_Avg'], label=f'{code2} 平均價格', marker='o', linestyle='-', color='green')
        plt.fill_between(df2['date'], df2['Day_Avg'], color="lightgreen", alpha=0.3)
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(plt)

# 使用說明功能
elif option == "查看使用說明":
    st.subheader("使用說明")
    st.write("""
        1. 在側邊欄輸入股票代碼和最近幾天的日均價。
        2. 點擊「預測」按鈕，系統將自動根據您的輸入進行股票趨勢預測。
        3. 結果會顯示預測的未來價格走勢，以及股票的過渡矩陣與穩態機率。
        4. 在股票比較功能中，可以比較兩隻股票的結果。
    """)
