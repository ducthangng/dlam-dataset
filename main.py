import pandas as pd

train = pd.read_csv("train.csv")
forecast_val = pd.read_csv("forecast_index_validation.csv")

print("=== TRAIN ===")
print(train.shape)
print(train['timestamp'].min(), "→", train['timestamp'].max())
print(train.dtypes)
print(train.head(3))

print("\n=== FORECAST INDEX ===")
print(forecast_val.shape)
print(forecast_val['timestamp'].min(), "→", forecast_val['timestamp'].max())
print(forecast_val.head(3))

# Chạy ngay cái này
print("\n=== SERIES ===")
print(forecast_val['series_id'].nunique())
print(forecast_val['series_id'].unique())
print(32256 / 24)  # = 1344 hours = 56 ngày?
print(32256 / 336) # = 96