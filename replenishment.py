import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from db_utils import get_connection, fetch_sales_data, fetch_inventory_data
from config import CONFIG

# 🔄 Calculate Reorder Quantities
def calculate_reorder_quantity(forecast_df, inventory_df, buffer_ratio=0.2):
    forecast_df = forecast_df.copy()
    forecast_df['SKU'] = forecast_df.get('SKU', '1CDCE0')  # default if missing

    merged = pd.merge(
        forecast_df[['SKU', 'ds', 'yhat']],
        inventory_df[['SKU', 'CurrentStock']],
        on='SKU',
        how='left'
    )

    merged['ReorderPoint'] = (merged['yhat'] * (1 + buffer_ratio)).round()
    merged['ReorderQty'] = (merged['ReorderPoint'] - merged['CurrentStock']).clip(lower=0).astype(int)

    return merged[['SKU', 'ds', 'yhat', 'CurrentStock', 'ReorderPoint', 'ReorderQty']]

# 🗃️ Insert Replenishment Plan into SQL
def insert_replenishment_queue(conn, df, table_name="ReplenishmentQueue"):
    cursor = conn.cursor()
    insert_query = f"""
        INSERT INTO {table_name} (
            SKU, ForecastDate, ForecastedDemand, CurrentStock,
            ReorderPoint, ReorderQty, CreatedAt
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    for _, row in df.iterrows():
        cursor.execute(insert_query, (
            str(row['SKU']),
            pd.to_datetime(row['ds']).date(),
            int(row['yhat']),
            int(row['CurrentStock']),
            int(row['ReorderPoint']),
            int(row['ReorderQty']),
            datetime.now()
        ))
    conn.commit()
    cursor.close()

# 📊 Visualize Replenishment Plan
def plot_reorder_plan(df, sku):
    plt.figure(figsize=(10, 5))
    plt.plot(df['ds'], df['yhat'], label='Forecasted Demand', marker='o')
    plt.plot(df['ds'], df['ReorderPoint'], label='Reorder Point', linestyle='--')
    plt.bar(df['ds'], df['ReorderQty'], label='Reorder Qty', alpha=0.5)
    plt.title(f"Replenishment Plan for SKU {sku}")
    plt.xlabel("Date")
    plt.ylabel("Units")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 🚀 Entry Point
if __name__ == "__main__":
    sql_cfg = CONFIG["sql"]
    forecast_cfg = CONFIG["forecast"]

    conn = get_connection(sql_cfg["server"], sql_cfg["database"])
    sku = "1CDCE0"

    # Load forecast from previous step (e.g., XGBoost or Prophet)
    forecast_df = pd.read_sql(f"SELECT * FROM ForecastResults WHERE SKU = '{sku}'", conn)
    forecast_df['ds'] = pd.to_datetime(forecast_df['ForecastDate'])
    forecast_df['yhat'] = forecast_df['ForecastedDemand']

    # Load current inventory
    inventory_df = fetch_inventory_data(conn, sku)  # assumes SKU-level inventory

    if forecast_df.empty or inventory_df.empty:
        print(f"⚠️ Missing data for SKU {sku}. Replenishment skipped.")
    else:
        plan_df = calculate_reorder_quantity(forecast_df, inventory_df, buffer_ratio=0.2)

        print("\n📦 Replenishment Plan Preview:")
        print(plan_df.head(10))

        insert_replenishment_queue(conn, plan_df)
        print(f"\n✅ Replenishment plan inserted into 'ReplenishmentQueue' table.")

        plot_reorder_plan(plan_df, sku)