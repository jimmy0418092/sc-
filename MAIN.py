import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox

def get_matrix_and_state(code, day_avgs):
    print("股票代碼", code)
    
    # 假設你已經有相應的歷史數據文件
    df = pd.read_csv(os.path.join('data', f'{code}.csv'))

    # 計算歷史數據的均價
    df['Day_Avg'] = (df['open'] + df['close']) / 2
    df['Pct_Change'] = df['Day_Avg'].pct_change() * 100

    # 使用滑動平均填充 NaN 值
    df['Pct_Change'].fillna(df['Pct_Change'].rolling(window=3, min_periods=1).mean(), inplace=True)

    # 將輸入的日均價新增到資料框中
    last_days = pd.DataFrame({
        'Day_Avg': day_avgs,
        'Pct_Change': pd.Series(day_avgs).pct_change() * 100
    })

    # 合併新輸入的日均價
    df = pd.concat([df, last_days], ignore_index=True)

    # 定義狀態分類
    def categorize_state(change):
        if change > 6:  # 6% 到 10% 轉為 '++'
            return '++'
        elif change > 0.5:  # 0.5% 到 6% 轉為 '+'
            return '+'
        elif change >= -0.5 and change <= 0.5:  # -0.5% 到 0.5% 轉為 '0'
            return '0'
        elif change > -6 and change <= -0.5:  # -0.5% 到 -6% 轉為 '-'
            return '-'
        else:  # -7% 到 -10% 轉為 '--'
            return '--'

    # 應用狀態分類
    df['State'] = df['Pct_Change'].apply(categorize_state)

    # 定義狀態順序
    states = ['++', '+', '0', '-', '--']

    # 初始化轉移矩陣
    transition_matrix = pd.DataFrame(0, index=states, columns=states)

    # 計算轉移次數
    for (i, current_state), (_, next_state) in zip(df['State'].shift().items(), df['State'].items()):
        if pd.notna(current_state) and pd.notna(next_state):
            transition_matrix.at[current_state, next_state] += 1

    # 將 NaN 替換為 0
    transition_matrix.fillna(0, inplace=True)

    # 將每一行的值轉換為機率
    transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0)

    # 將轉移矩陣轉換為 numpy 陣列
    P = transition_matrix.values

    # 計算穩態機率
    evals, evecs = np.linalg.eig(P.T)
    steady_state = np.real(evecs[:, np.isclose(evals, 1)])

    # 正規化穩態機率向量
    steady_state = steady_state / steady_state.sum()
    steady_state = steady_state.flatten()

    # 將穩態機率映射到狀態
    steady_state_probs = pd.Series(steady_state, index=states)

    return transition_matrix, steady_state_probs, 1/steady_state_probs

def format_matrix(matrix):
    result = []
    for index, row in matrix.iterrows():
        formatted_row = [f'{index}'] + [f'{value:.2f}' for value in row]
        result.append(" | ".join(formatted_row))
    header = '  ' + " | ".join([""] + list(matrix.columns))
    return "\n".join([header] + result)

def calculate():
    try:
        code = entry_code.get()
        day_avgs = [float(entry.get()) for entry in day_avg_entries]

        transition_matrix, steady_state_probs, return_days = get_matrix_and_state(code, day_avgs)
        
        # 計算最後狀態
        last_change = (day_avgs[-1] - day_avgs[-2]) / day_avgs[-2] * 100

        # 狀態分類
        def categorize_state(change):
            if change > 6:  # 6% 到 10% 轉為 '++'
                return '++'
            elif change > 0.5:  # 0.5% 到 6% 轉為 '+'
                return '+'
            elif change >= -0.5 and change <= 0.5:  # -0.5% 到 0.5% 轉為 '0'
                return '0'
            elif change > -6 and change <= -0.5:  # -0.5% 到 -6% 轉為 '-'
                return '-'
            else:  # -7% 到 -10% 轉為 '--'
                return '--'
        
        last_state = categorize_state(last_change)

        # 預測下一個狀態
        next_state_probabilities = transition_matrix.loc[last_state]
        predicted_next_state = next_state_probabilities.idxmax()

        result = f"Transition Matrix:\n{format_matrix(transition_matrix)}\n\n"
        result += f"Steady State Probabilities:\n{steady_state_probs.to_string(float_format='%.2f')}\n\n"
        result += f"Return days:\n{return_days.to_string(float_format='%.2f')}\n\n"
        result += f"Last State: {last_state} (Change: {last_change:.2f}%)\n"
        result += f"Predicted Next State: {predicted_next_state}"

        messagebox.showinfo("Prediction Result", result)
    except ValueError:
        messagebox.showerror("Input Error", "請填寫正確的均價")
    except FileNotFoundError:
        messagebox.showerror("Input Error", "找不到此股票的資料")

def create_day_avg_inputs(num_days):
    global day_avg_entries
    for widget in day_avg_frame.winfo_children():
        widget.destroy()
    day_avg_entries = []
    for i in range(num_days):
        tk.Label(day_avg_frame, text=f"Day {i+1}:").grid(row=i, column=0)
        entry = tk.Entry(day_avg_frame)
        entry.grid(row=i, column=1)
        day_avg_entries.append(entry)

# 建立主視窗
root = tk.Tk()
root.title("Stock Prediction")

# 股票代碼輸入
tk.Label(root, text="Enter Stock Code:").grid(row=0, column=0)
entry_code = tk.Entry(root)
entry_code.grid(row=0, column=1)

# 輸入天數選擇
tk.Label(root, text="輸入你要輸入的日均價天數 (1-30)").grid(row=1, column=0, columnspan=2)
num_days = tk.IntVar(value=5)
tk.Spinbox(root, from_=1, to=30, textvariable=num_days, command=lambda: create_day_avg_inputs(num_days.get())).grid(row=2, column=0, columnspan=2)

# 均價輸入區域
day_avg_frame = tk.Frame(root)
day_avg_frame.grid(row=3, column=0, columnspan=2)
create_day_avg_inputs(5)

# 按鈕
calculate_button = tk.Button(root, text="Predict", command=calculate)
calculate_button.grid(row=4, columnspan=2)

# 運行應用程式
root.mainloop()

