import pandas as pd
import numpy as np
from datetime import datetime
from prophet import Prophet
from db_utils import get_connection, fetch_sales_data
from config import CONFIG
import matplotlib.pyplot as plt
import pyodbc

print("Available ODBC Drivers:", pyodbc.drivers())

#  Accuracy Evaluation
def evaluate_forecast_accuracy(forecast_df, actual_df, min_actual_threshold=5):
    filtered_actuals = actual_df[actual_df['y'] > min_actual_threshold]
    filtered_forecast = forecast_df[forecast_df['ds'].isin(filtered_actuals['ds'])]

    merged = pd.merge(
        filtered_forecast[['ds', 'yhat']],
        filtered_actuals[['ds', 'y']],
        on='ds',
        how='inner'
    )

    mae = (merged['yhat'] - merged['y']).abs().mean()
    rmse = ((merged['yhat'] - merged['y']) ** 2).mean() ** 0.5
    mape = ((merged['yhat'] - merged['y']).abs() / merged['y']).mean() * 100
    smape = 100 * np.mean(2 * np.abs(merged['yhat'] - merged['y']) / (np.abs(merged['yhat']) + np.abs(merged['y'])))

    return mae, rmse, mape, smape

#  Forecast Generation
def generate_forecast_preview(server, database, sales_table, sku, forecast_days=30, noise_level=0.01):
    conn = get_connection(server, database)
    df = fetch_sales_data(conn, sales_table, sku)

    if df.empty:
        print(f"No sales data found for SKU: {sku}")
        return pd.DataFrame(), conn, None, None, None

    df['ds'] = pd.to_datetime(df['ds'])
    df['y_raw'] = df['y']  # preserve original
    df['y'] = np.log1p(df['y'])  # log-transform

    #  Inject small Gaussian noise to reduce overfitting
    df['y'] += np.random.normal(loc=0, scale=noise_level, size=len(df))

    # Optional: Add holidays
    holidays = pd.DataFrame({
        'ds': pd.to_datetime(['2025-05-01', '2025-06-15']),  # example dates
        'holiday': ['LaborDay', 'Eid']
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.5,
        holidays=holidays
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # Inverse transform predictions
    forecast['yhat'] = np.expm1(forecast['yhat'])
    forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
    forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])

    # Evaluate accuracy on historical portion only
    actual_df = fetch_sales_data(conn, sales_table, sku)
    actual_df['ds'] = pd.to_datetime(actual_df['ds'])
    history_forecast = forecast[forecast['ds'].isin(actual_df['ds'])]
    mae, rmse, mape, smape = evaluate_forecast_accuracy(history_forecast, actual_df)

    print(f"\n Forecast Accuracy for SKU {sku} (filtered actuals > 5):")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, SMAPE: {smape:.2f}%")

    preview_df = pd.DataFrame({
        'SKU': sku if sku else 'ALL',
        'ForecastDate': forecast['ds'].dt.date,
        'ForecastedDemand': forecast['yhat'].round().astype(int),
        'ReorderPoint': (forecast['yhat'] * 0.8).round().astype(int),
        'ModelUsed': 'Prophet',
        'ConfidenceIntervalLow': forecast['yhat_lower'].round().astype(int),
        'ConfidenceIntervalHigh': forecast['yhat_upper'].round().astype(int),
        'CreatedAt': datetime.now()
    })

    preview_df = preview_df[preview_df['ForecastDate'] > datetime.today().date()]

    return preview_df, conn, (mae, rmse, mape, smape), forecast, actual_df

#  Insert Forecast to SQL
def insert_forecast_results(conn, df, table_name="ForecastResults"):
    cursor = conn.cursor()

    insert_query = f"""
        INSERT INTO {table_name} (
            SKU, ForecastDate, ForecastedDemand, ReorderPoint,
            ModelUsed, ConfidenceIntervalLow, ConfidenceIntervalHigh, CreatedAt
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    for _, row in df.iterrows():
        cursor.execute(insert_query, (
            str(row['SKU']),
            pd.to_datetime(row['ForecastDate']).date(),
            int(row['ForecastedDemand']),
            int(row['ReorderPoint']),
            str(row['ModelUsed']),
            int(row['ConfidenceIntervalLow']),
            int(row['ConfidenceIntervalHigh']),
            pd.to_datetime(row['CreatedAt'])
        ))

    conn.commit()
    cursor.close()

#  Plot Forecast vs Actuals
def plot_forecast_vs_actual(forecast_df, actual_df, sku):
    merged = pd.merge(
        forecast_df[['ds', 'yhat']],
        actual_df[['ds', 'y']],
        on='ds',
        how='inner'
    )

    if merged.empty:
        print(f" No overlapping dates found for SKU {sku}. Plot skipped.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(merged['ds'], merged['y'], label='Actual', marker='o')
    plt.plot(merged['ds'], merged['yhat'], label='Forecast', linestyle='--')
    plt.title(f"Forecast vs Actuals for SKU {sku}")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#  Entry Point
if __name__ == "__main__":
    sql_cfg = CONFIG["sql"]
    forecast_cfg = CONFIG["forecast"]

    preview, conn, metrics, full_forecast, actual_df = generate_forecast_preview(
        server=sql_cfg["server"],
        database=sql_cfg["database"],
        sales_table=sql_cfg["table"],
        sku=forecast_cfg["sku"],#"1CDCE0",
        forecast_days=forecast_cfg["horizon"],
        noise_level=0.01  # Inject 1% noise
    )

    if not preview.empty:
        print("\n Forecast Preview:")
        print(preview.head(10))

        insert_forecast_results(conn, preview)
        print(f"\n Forecast inserted into 'ForecastResults' table.")

        if metrics:
            mae, rmse, mape, smape = metrics
            print(f"\n Final Accuracy Metrics (filtered actuals > 5):")
            print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, SMAPE: {smape:.2f}%")

            history_forecast = full_forecast[full_forecast['ds'].isin(actual_df['ds'])]
            plot_forecast_vs_actual(history_forecast, actual_df, "1CDCE0")
    else:
        print("\n No forecast generated.")