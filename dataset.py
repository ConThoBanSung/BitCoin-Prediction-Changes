import yfinance as yf

def fetch_yahoo_finance_bitcoin_data(start_date, end_date):
    # Lấy dữ liệu giá Bitcoin từ Yahoo Finance
    df = yf.download('BTC-USD', start=start_date, end=end_date, interval='1h')
    return df

# Ví dụ sử dụng
df = fetch_yahoo_finance_bitcoin_data('2023-01-01', '2024-10-21')
print(df.head())
