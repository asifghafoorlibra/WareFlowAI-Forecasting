import pandas as pd
from datetime import datetime
from prophet import Prophet
from db_utils import get_connection, fetch_sales_data
from config import CONFIG

import pyodbc
print("Available ODBC Drivers:", pyodbc.drivers())

def generate_forecast_preview(server, database, sales_table, sku, forecast_days=30):
    # Step 1: Connect and fetch data
    conn = get_connection(server, database)
    df = fetch_sales_data(conn, sales_table, sku)

    if df.empty:
        print(f"No sales data found for SKU: {sku}")
        return pd.DataFrame(), conn

    # Step 2: Train Prophet
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # Step 3: Format preview DataFrame
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

    # Step 4: Filter future dates only
    preview_df = preview_df[preview_df['ForecastDate'] > datetime.today().date()]

    return preview_df, conn

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
            pd.to_datetime(row['ForecastDate']).date(),  # Ensure it's a native Python date
            int(row['ForecastedDemand']),
            int(row['ReorderPoint']),
            str(row['ModelUsed']),
            int(row['ConfidenceIntervalLow']),
            int(row['ConfidenceIntervalHigh']),
            pd.to_datetime(row['CreatedAt'])  # Ensure it's a native Python datetime
        ))

    conn.commit()
    cursor.close()

# 🔍 Entry point
if __name__ == "__main__":
    sql_cfg = CONFIG["sql"]
    forecast_cfg = CONFIG["forecast"]

    preview, conn = generate_forecast_preview(
        server=sql_cfg["server"],
        database=sql_cfg["database"],
        sales_table=sql_cfg["table"],
        sku= "1CDCE0", #sql_cfg["sku_filter"],
        forecast_days=forecast_cfg["horizon"]
    )

    if not preview.empty:
        print("\n Forecast Preview:")
        print(preview.head(10))

        # Save to database
        insert_forecast_results(conn, preview)
        print(f"\n Forecast inserted into 'ForecastResults' table.")
    else:
        print("\n No forecast generated.")


def evaluate_forecast_accuracy(forecast_df, actual_df):
    merged = pd.merge(
        forecast_df[['ds', 'yhat']],
        actual_df[['ds', 'y']],
        on='ds',
        how='inner'
    )
    mae = (merged['yhat'] - merged['y']).abs().mean()
    rmse = ((merged['yhat'] - merged['y']) ** 2).mean() ** 0.5
    mape = ((merged['yhat'] - merged['y']).abs() / merged['y']).mean() * 100
    return mae, rmse, mape


