import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Hàm tính toán các chỉ số kỹ thuật
def compute_indicators(data):
    rsi = compute_rsi(data['Close'])
    sma = compute_sma(data['Close'])
    ema = compute_ema(data['Close'])
    upper_band, lower_band = compute_bollinger_bands(data['Close'])
    macd, signal = compute_macd(data['Close'])
    stochastic_k, stochastic_d = compute_stochastic_oscillator(data['Close'])
    atr = compute_atr(data['Close'])

    return {
        'RSI': rsi,
        'SMA': sma,
        'EMA': ema,
        'Upper_Band': upper_band,
        'Lower_Band': lower_band,
        'MACD': macd,
        'Signal': signal,
        'Stochastic_K': stochastic_k,
        'Stochastic_D': stochastic_d,
        'ATR': atr
    }

# Lấy dữ liệu lịch sử Bitcoin
def fetch_data():
    df = yf.download('BTC-USD', start='2022-01-01', end='2024-10-22', interval='1d')
    return df

# Tạo dataset từ các chỉ số kỹ thuật
def create_dataset(df):
    indicators = []
    signals = []

    for i in range(20, len(df)):
        data_slice = df.iloc[i-20:i]
        indicator_values = compute_indicators(data_slice)
        indicators.append(indicator_values)

        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            signals.append(1)  # Tín hiệu mua
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            signals.append(-1)  # Tín hiệu bán
        else:
            signals.append(0)  # Không hành động

    return pd.DataFrame(indicators), signals

# Tải dữ liệu và tạo dataset
data = fetch_data()
indicators_df, signals = create_dataset(data)

# Chuyển đổi DataFrame thành mảng NumPy cho mô hình
X = indicators_df.values
y = signals

# Chia dữ liệu thành tập huấn luyện và kiểm tra (sử dụng TimeSeriesSplit)
tscv = TimeSeriesSplit(n_splits=5)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Xử lý mất cân bằng lớp
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Huấn luyện mô hình Gradient Boosting với GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=tscv)
grid_search.fit(X_resampled, y_resampled)

# Dự đoán trên tập kiểm tra
y_pred = grid_search.predict(X_scaled)

# Đánh giá mô hình
print(classification_report(y, y_pred))

# Lưu mô hình
joblib.dump(grid_search.best_estimator_, 'trading_signal_model_gbm.pkl')
