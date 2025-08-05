import pandas as pd
import numpy as np
from datetime import datetime
from prophet import Prophet
from db_utils import get_connection, fetch_sales_data
from config import CONFIG
import matplotlib.pyplot as plt
import pyodbc

print("Available ODBC Drivers:", pyodbc.drivers())

# Accuracy Evaluation
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

# 🔮 Forecast Generation
def generate_forecast_preview(server, database, sales_table, sku=None, forecast_days=30, noise_level=0.01, show_plot=True):
    sku = sku or "ALL"
    conn = get_connection(server, database)

    if sku.lower() == "all":
        sku_query = f"SELECT DISTINCT SKU FROM {sales_table}"
        all_skus = pd.read_sql(sku_query, conn)["SKU"].dropna().unique().tolist()

        combined_preview = []
        combined_forecast = []
        combined_actuals = []

        for single_sku in all_skus:
            preview_df, _, metrics, forecast, actual_df = generate_forecast_preview(
                server, database, sales_table, single_sku, forecast_days, noise_level, show_plot=False
            )
            if not preview_df.empty:
                combined_preview.append(preview_df)
                combined_forecast.append(forecast.assign(SKU=single_sku))
                combined_actuals.append(actual_df.assign(SKU=single_sku))

                mae, rmse, mape, smape = metrics
                print(f"\nSKU {single_sku} Accuracy:")
                print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, SMAPE: {smape:.2f}%")

        if combined_preview:
            final_df = pd.concat(combined_preview, ignore_index=True)
            full_forecast = pd.concat(combined_forecast, ignore_index=True)
            full_actuals = pd.concat(combined_actuals, ignore_index=True)

            return final_df, conn, None, full_forecast, full_actuals
        else:
            print("No forecast generated for any SKU.")
            return pd.DataFrame(), conn, None, None, None

    # Single SKU logic
    df = fetch_sales_data(conn, sales_table, sku)
    if df.empty:
        print(f"No sales data found for SKU: {sku}")
        return pd.DataFrame(), conn, None, None, None

    df['ds'] = pd.to_datetime(df['ds'])
    df['y_raw'] = df['y']
    df['y'] = np.log1p(df['y']) + np.random.normal(loc=0, scale=noise_level, size=len(df))

    holidays = pd.DataFrame({
        'ds': pd.to_datetime(['2025-05-01', '2025-06-15']),
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

    forecast['yhat'] = np.expm1(forecast['yhat'])
    forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
    forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])

    actual_df = fetch_sales_data(conn, sales_table, sku)
    actual_df['ds'] = pd.to_datetime(actual_df['ds'])
    history_forecast = forecast[forecast['ds'].isin(actual_df['ds'])]

    mae, rmse, mape, smape = evaluate_forecast_accuracy(history_forecast, actual_df)

    print(f"\nForecast Accuracy for SKU {sku} (filtered actuals > 5):")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, SMAPE: {smape:.2f}%")

    preview_df = pd.DataFrame({
        'SKU': sku,
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

# Insert Forecast to SQL
def insert_forecast_results(conn, df, table_name="ForecastResults2"):
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

# Plot Forecast vs Actuals
def plot_forecast_vs_actual(forecast_df, actual_df, sku):
    merged = pd.merge(
        forecast_df[['ds', 'yhat']],
        actual_df[['ds', 'y']],
        on='ds',
        how='inner'
    )
    if merged.empty:
        print(f"No overlapping dates found for SKU {sku}. Plot skipped.")
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

# Entry Point
if __name__ == "__main__":
    sql_cfg = CONFIG["sql"]
    forecast_cfg = CONFIG["forecast"]

    sku = sql_cfg.get("sku_filter", "ALL")

    preview, conn, metrics, full_forecast, actual_df = generate_forecast_preview(
        server=sql_cfg["server"],
        database=sql_cfg["database"],
        sales_table=sql_cfg["table"],
        sku=sku,
        forecast_days=forecast_cfg.get("horizon", 30),
        noise_level=0.01,
        show_plot=True
    )

    if not preview.empty:
        print("\nForecast Preview:")
        print(preview.head(10))

        insert_forecast_results(conn, preview)
        print(f"\nForecast inserted into 'ForecastResults' table.")

        if sku.lower() == "all" and full_forecast is not None and actual_df is not None:
            history_forecast = full_forecast[full_forecast['ds'].isin(actual_df['ds'])]
            plot_forecast_vs_actual(history_forecast, actual_df, sku)
    else:
        print("\nNo forecast generated.")
