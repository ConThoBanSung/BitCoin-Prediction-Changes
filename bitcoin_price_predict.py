from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.3))  # Tăng Dropout để tránh overfitting
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(25))  # Thêm lớp Dense với nhiều nơron hơn
model.add(Dense(1))

# Biên dịch mô hình
model.compile(optimizer='adam', loss='mean_squared_error')

# Sử dụng EarlyStopping và ReduceLROnPlateau để điều chỉnh quá trình huấn luyện
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=0.00001)

# Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[early_stop, reduce_lr], verbose=1)

# Dự đoán giá với tập kiểm tra
predicted_prices = model.predict(X_test)

# Inverse transform để đưa dữ liệu về dạng ban đầu
predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
model.save('bitcoin_price_prediction_model.h5')


# Đánh giá mô hình
import matplotlib.pyplot as plt
plt.plot(df.index[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), color='blue', label='Giá thực tế')
plt.plot(df.index[-len(y_test):], predicted_prices, color='red', label='Giá dự đoán')
plt.legend()
plt.show()
