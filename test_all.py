import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import joblib  # Để tải mô hình GBM
import time
from datetime import datetime
import pytz
import tensorflow as tf  # Thêm import TensorFlow

# Vô hiệu hóa cảnh báo
tf.get_logger().setLevel('ERROR')  # Chỉ hiển thị lỗi
tf.config.experimental_run_functions_eagerly(False)  # Tắt chế độ eager execution

# Hàm để lấy dữ liệu giá Bitcoin từ Yahoo Finance
def fetch_yahoo_finance_bitcoin_data():
    df = yf.download('BTC-USD', period='1d', interval='1m')
    return df

# Hàm để chuẩn hóa và tạo dataset cho mô hình
def create_dataset(data, time_step=1):
    X = []
    if len(data) < time_step:
        return np.array(X)

    for i in range(len(data) - time_step + 1):
        X.append(data[i:(i + time_step), 0])
    return np.array(X)

# Tải các mô hình đã huấn luyện
price_model = load_model('bitcoin_price_prediction_model.h5')
signal_model = joblib.load('trading_signal_model_gbm.pkl')  # Tải mô hình tín hiệu giao dịch
hybrid_model = load_model('hybrid_trend_prediction.h5')  # Tải mô hình dự đoán xu hướng

# Hàm tính toán chỉ số RSI
def compute_rsi(data, window=14):
    delta = np.diff(data)
    gain = (delta[delta > 0].sum() / window) if window > 0 else 0
    loss = (-delta[delta < 0].sum() / window) if window > 0 else 0
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Hàm tính toán đường trung bình động (SMA)
def compute_sma(data, window=14):
    return np.mean(data[-window:])

# Hàm tính toán đường trung bình lũy thừa (EMA)
def compute_ema(data, window=14):
    return pd.Series(data).ewm(span=window, adjust=False).mean().iloc[-1]

# Hàm tính toán Bollinger Bands
def compute_bollinger_bands(data, window=20, num_std_dev=2):
    sma = compute_sma(data, window)
    std_dev = np.std(data[-window:])
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, lower_band

# Hàm tính toán MACD
def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    ema_short = pd.Series(data).ewm(span=short_window, adjust=False).mean().iloc[-1]
    ema_long = pd.Series(data).ewm(span=long_window, adjust=False).mean().iloc[-1]
    macd = ema_short - ema_long
    signal = pd.Series([macd]).ewm(span=signal_window, adjust=False).mean().iloc[-1]
    return macd, signal

# Hàm tính toán Stochastic Oscillator
def compute_stochastic_oscillator(data, k_window=14, d_window=3):
    low_min = np.min(data[-k_window:])
    high_max = np.max(data[-k_window:])
    k = 100 * ((data[-1] - low_min) / (high_max - low_min))
    d = np.mean([k]) if len(data[-d_window:]) > 0 else 0
    return k, d

# Hàm tính toán Average True Range (ATR)
def compute_atr(data, window=14):
    tr = []
    for i in range(1, len(data)):
        tr.append(max(data[i] - data[i - 1], abs(data[i] - data[i - 1]), abs(data[i - 1] - data[i])))
    return np.mean(tr[-window:]) if len(tr) >= window else 0

# Hàm để dự đoán theo từng phút
# Adjust the call to np.array(data_list).flatten() where necessary in the indicator functions
def calculate_technical_indicators(data_list):
    # Flatten data_list to avoid shape issues
    data_array = np.array(data_list).flatten()

    rsi = compute_rsi(data_array, window=14)
    sma_50 = compute_sma(data_array, window=50)
    sma_200 = compute_sma(data_array, window=200)
    ema_20 = compute_ema(data_array, window=20)
    upper_band, lower_band = compute_bollinger_bands(data_array, window=20)
    macd, signal = compute_macd(data_array)
    stochastic_k, stochastic_d = compute_stochastic_oscillator(data_array)
    atr = compute_atr(data_array)
    momentum = data_array[-1] - data_array[-2]  # Calculate momentum
    volume_change = np.random.random()  # Replace with volume calculation method if applicable

    return {
        'RSI': rsi,
        'MACD': macd,
        'MACD_signal': signal,
        'MACD_diff': macd - signal,  # Calculate MACD_diff
        'EMA_20': ema_20,
        'SMA_50': sma_50,
        'SMA_200': sma_200,
        'Stoch': stochastic_k,  # Select Stochastic_K or Stochastic_D as required
        'ATR': atr,
        'Momentum': momentum,
        'Volume_change': volume_change,
        'BB_high': upper_band,
        'BB_low': lower_band,
    }

# Hàm để dự đoán theo từng phút
def predict_real_time():
    df = fetch_yahoo_finance_bitcoin_data()

    if df.empty:
        print("Không có dữ liệu mới. Thử lại sau.")
        return

    data_list = df['Close'].tail(60).values.tolist()

    while True:
        try:
            df = fetch_yahoo_finance_bitcoin_data()

            if df.empty:
                print("Không có dữ liệu mới. Thử lại sau.")
                time.sleep(600)
                continue

            if df.isnull().values.any():
                print("Dữ liệu chứa giá trị NaN.")
                time.sleep(600)
                continue

            last_close_price = df['Close'].iloc[-1]
            data_list.append(last_close_price)

            if len(data_list) > 60:
                data_list.pop(0)

            if len(data_list) < 60:
                print(f"Chưa đủ dữ liệu để dự đoán: {len(data_list)} giá trị có sẵn.")
                time.sleep(600)
                continue

            # Chuẩn hóa dữ liệu
            data_array = np.array(data_list).reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data_array)

            # Tạo dữ liệu cho dự đoán
            X = create_dataset(scaled_data, time_step=60)

            if X.shape[0] == 0:
                print("Dữ liệu đầu vào không hợp lệ cho mô hình.")
                time.sleep(600)
                continue

            X = X.reshape(X.shape[0], X.shape[1], 1)

            # Dự đoán giá cho phút hiện tại
            predicted_price_current = price_model.predict(X[-1].reshape(1, X.shape[1], 1))
            predicted_price_current = scaler.inverse_transform(predicted_price_current)

            # Dự đoán giá cho phút tiếp theo
            if len(X) > 0:
                next_X = np.append(X[-1][1:], predicted_price_current)  # Thêm giá dự đoán vào cuối
                next_X = next_X.reshape((1, 60, 1))  # Reshape cho mô hình

                predicted_price_next = price_model.predict(next_X)
                predicted_price_next = scaler.inverse_transform(predicted_price_next)

            # Tính toán thời gian cho dự đoán
            current_time_utc = datetime.now(pytz.utc)
            vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
            current_time_vn = current_time_utc.astimezone(vietnam_tz)

            print(f"Dự đoán giá Bitcoin hiện tại: {predicted_price_current[0][0]:.2f} USD")
            print(f"Dự đoán giá Bitcoin cho phút tiếp theo: {predicted_price_next[0][0]:.2f} USD")
            print(f"Khung giờ dự đoán hiện tại: {current_time_vn.strftime('%Y-%m-%d %H:%M:%S')} (Giờ Việt Nam)")

            # Tính toán các chỉ số kỹ thuật
            if len(data_list) > 20:
                indicators = calculate_technical_indicators(data_list)

                # Chuẩn bị dữ liệu đầu vào cho mô hình hybrid với 14 đặc trưng
                hybrid_features = np.array([
                    predicted_price_next[0][0],  # Giá dự đoán tiếp theo
                    indicators['RSI'],
                    indicators['MACD'],
                    indicators['MACD_signal'],
                    indicators['MACD_diff'],
                    indicators['EMA_20'],
                    indicators['SMA_50'],
                    indicators['SMA_200'],
                    indicators['Stoch'],
                    indicators['ATR'],
                    # Thêm 4 đặc trưng nữa nếu cần
                    # Ví dụ:
                    indicators['Momentum'],
                    indicators['Volume_change'],
                    indicators['BB_high'],
                    indicators['BB_low']
                ])

                # Reshape hybrid_features thành (1, 100, 14)
                # Giả sử bạn muốn lặp lại các giá trị trong hybrid_features cho đủ 100 bước
                hybrid_features = np.tile(hybrid_features, (100, 1))  # Lặp lại 100 lần
                hybrid_features = hybrid_features.reshape(1, 100, 14)  # Reshape cho mô hình

                # Chuẩn bị dữ liệu đầu vào cho mô hình tín hiệu giao dịch với 10 đặc trưng
                trading_features = np.array([
                    predicted_price_next[0][0],  # Giá dự đoán tiếp theo
                    indicators['RSI'],
                    indicators['MACD'],
                    indicators['MACD_signal'],
                    indicators['MACD_diff'],
                    indicators['EMA_20'],
                    indicators['SMA_50'],
                    indicators['SMA_200'],
                    indicators['Stoch'],
                    indicators['ATR']
                ]).reshape(1, -1)  # Reshape cho mô hình tín hiệu giao dịch

                # Dự đoán tín hiệu giao dịch
                trading_signal = signal_model.predict(trading_features)
                predicted_signal = 'Mua' if trading_signal[0] > 0.5 else 'Bán'

                # Dự đoán xu hướng từ mô hình hybrid
                hybrid_prediction = hybrid_model.predict(hybrid_features)
                predicted_trend = 'Tăng' if hybrid_prediction[0][0] > 0.5 else 'Giảm'

                print(f"Tín hiệu giao dịch dự đoán: {predicted_signal}")
                print(f"Dự đoán xu hướng (hybrid model): {predicted_trend}")

            time.sleep(60)  # Dừng 60 giây trước khi lấy dữ liệu tiếp theo
        except Exception as e:
            print(f"Có lỗi xảy ra: {e}")
            time.sleep(60)  # Dừng 60 giây trước khi thử lại

# Gọi hàm dự đoán theo thời gian thực
predict_real_time()
