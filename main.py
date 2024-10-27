import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import ta
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Bidirectional, Conv1D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from keras.optimizers import Adam
from keras.regularizers import l2

# Tải dữ liệu Bitcoin
df = yf.download('BTC-USD', start='2023-01-01', end='2024-10-22', interval='1h')

# Thêm chỉ số kỹ thuật bổ sung
df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
macd = ta.trend.MACD(df['Close'])
df['MACD'] = macd.macd()
df['MACD_signal'] = macd.macd_signal()
df['MACD_diff'] = macd.macd_diff()
df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()
df['Stoch'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
df['Momentum'] = df['Close'] - df['Close'].shift(10)
df['Volume_change'] = df['Volume'].pct_change()  # Thay đổi khối lượng
indicator_bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
df['BB_high'] = indicator_bb.bollinger_hband()
df['BB_low'] = indicator_bb.bollinger_lband()

# Loại bỏ các giá trị NaN
df.dropna(inplace=True)

# Kiểm tra và xử lý giá trị vô cực
df = df.replace([np.inf, -np.inf], np.nan)
df.dropna(inplace=True)

# Tạo nhãn: 1 nếu giá tăng, 0 nếu giá giảm
df['Trend'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Chọn các đặc trưng để dự đoán xu hướng
features = df[['Close', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'EMA_20', 'SMA_50', 
               'SMA_200', 'Stoch', 'ATR', 'Momentum', 'Volume_change', 'BB_high', 'BB_low']].values
labels = df['Trend'].values

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# Tạo dữ liệu theo cửa sổ thời gian (window)
def create_dataset(X, y, time_step=100):
    Xs, ys = [], []
    for i in range(len(X) - time_step):
        Xs.append(X[i:i+time_step])
        ys.append(y[i+time_step])
    return np.array(Xs), np.array(ys)

# Tạo tập dữ liệu
X, y = create_dataset(scaled_features, labels)

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Xây dựng mô hình kết hợp LSTM và GRU với nhiều lớp hơn
def build_hybrid_model(input_shape):
    model = Sequential()
    
    # Lớp Conv1D để trích xuất đặc trưng
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # LSTM Layer
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Thêm lớp LSTM
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # GRU Layer
    model.add(Bidirectional(GRU(256, return_sequences=False)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

# Compile mô hình
hybrid_model = build_hybrid_model((X_train.shape[1], X_train.shape[2]))
hybrid_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Định nghĩa các callback
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=0.00001)

# Huấn luyện mô hình
hybrid_history = hybrid_model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_test, y_test),
                                   callbacks=[early_stop, reduce_lr], verbose=1)

# Đánh giá mô hình trên tập test
hybrid_test_loss, hybrid_test_acc = hybrid_model.evaluate(X_test, y_test)
print(f'Hybrid Model Test Accuracy: {hybrid_test_acc}')

# Dự đoán trên tập test
y_pred_hybrid = hybrid_model.predict(X_test)
y_pred_hybrid = (y_pred_hybrid > 0.5).astype(int)

# Đánh giá độ chính xác
print(confusion_matrix(y_test, y_pred_hybrid))
print(classification_report(y_test, y_pred_hybrid))

# Lưu mô hình
hybrid_model.save('hybrid_trend_prediction.h5')