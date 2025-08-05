import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from db_utils import get_connection, fetch_sales_data
from config import CONFIG

# Feature Engineering
def generate_features(df):
    df = df.copy()
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['lag_1'] = df['y'].shift(1)
    df['lag_7'] = df['y'].shift(7)
    df['rolling_mean_7'] = df['y'].rolling(window=7).mean()
    df['rolling_std_14'] = df['y'].rolling(window=14).std()
    df.dropna(inplace=True)
    return df

# Train XGBoost Model
def train_xgboost_model(df):
    df = generate_features(df)
    X = df.drop(columns=['ds', 'y'])
    y = df['y']

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    print(f"\nXGBoost Accuracy:\nMAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return model, df, X_test, y_test, preds, mae, rmse

# Forecast Future Demand
def forecast_with_xgboost(model, df, forecast_days=30):
    last_known = df.tail(14).copy()
    future_dates = pd.date_range(start=last_known['ds'].max() + timedelta(days=1), periods=forecast_days)
    forecast_rows = []

    for date in future_dates:
        row = {
            'ds': date,
            'day_of_week': date.dayofweek,
            'month': date.month,
            'is_weekend': int(date.dayofweek in [5, 6]),
            'lag_1': last_known['y'].iloc[-1],
            'lag_7': last_known['y'].iloc[-7] if len(last_known) >= 7 else last_known['y'].mean(),
            'rolling_mean_7': last_known['y'].rolling(window=7).mean().iloc[-1],
            'rolling_std_14': last_known['y'].rolling(window=14).std().iloc[-1]
        }
        pred_input = pd.DataFrame([row]).drop(columns=['ds'])
        yhat = model.predict(pred_input)[0]
        row['yhat'] = yhat
        forecast_rows.append(row)

        last_known = pd.concat([last_known, pd.DataFrame({'ds': [date], 'y': [yhat]})], ignore_index=True)

    forecast_df = pd.DataFrame(forecast_rows)
    return forecast_df[['ds', 'yhat']]

# Plot Test Set Predictions
def plot_test_predictions(X_test, y_test, preds, sku):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual', marker='o')
    plt.plot(y_test.index, preds, label='XGBoost Prediction', linestyle='--')
    plt.title(f"Test Set Prediction for SKU {sku}")
    plt.xlabel("Index")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Entry Point
if __name__ == "__main__":
    sql_cfg = CONFIG["sql"]
    forecast_cfg = CONFIG["forecast"]

    conn = get_connection(sql_cfg["server"], sql_cfg["database"])
    sku = sql_cfg["sku_filter"]#"1CDCE0"
    df = fetch_sales_data(conn, sql_cfg["table"], sku)

    if df.empty:
        print(f"No sales data found for SKU: {sku}")
    else:
        df['ds'] = pd.to_datetime(df['ds'])
        df = df[df['y'] > 0]

        model, enriched_df, X_test, y_test, preds, mae, rmse = train_xgboost_model(df)
        forecast_df = forecast_with_xgboost(model, enriched_df, forecast_days=forecast_cfg["horizon"])

        print("\nForecast Preview:")
        print(forecast_df.head(10))

        plot_test_predictions(X_test, y_test, preds, sku)