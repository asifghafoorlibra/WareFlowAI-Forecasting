# clustering_pipeline.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from db_utils import get_connection, fetch_bulk_sales
from config import CONFIG
import pyodbc

def run_inventory_clustering():
    # --- Load config ---
    server = CONFIG['sql']['server']
    database = CONFIG['sql']['database']
    sales_table = CONFIG['sql']['table']
    lookback_days = 90

    # --- Step 1: Connect and fetch sales data ---
    conn = get_connection(server, database)
    sales_df = fetch_bulk_sales(conn, sales_table, lookback_days=lookback_days)

    # --- Step 2: Feature Engineering ---
    total_sales = sales_df.groupby('SKU')['UnitsSold'].sum().rename('TotalSales')
    demand_df = sales_df.groupby(['SKU', 'SaleDate'])['UnitsSold'].sum().unstack(fill_value=0)
    demand_std = demand_df.std(axis=1)
    demand_mean = demand_df.mean(axis=1)
    demand_cv = (demand_std / demand_mean).fillna(0).rename('DemandCV')
    days_sold = sales_df.groupby('SKU')['SaleDate'].nunique().rename('DaysSold')

    features_df = pd.concat([total_sales, demand_cv, days_sold], axis=1).reset_index()

    # --- Step 3: Normalize features ---
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df[['TotalSales', 'DemandCV', 'DaysSold']])

    # --- Step 4: Apply K-Means ---
    kmeans = KMeans(n_clusters=4, random_state=42)
    features_df['Cluster'] = kmeans.fit_predict(scaled_features)

    # --- Step 5: Save clusters to SQL ---
    insert_query = """
    MERGE INTO SKUClusters AS target
    USING (SELECT ? AS SKU, ? AS TotalSales, ? AS DemandCV, ? AS DaysSold, ? AS Cluster) AS source
    ON target.SKU = source.SKU
    WHEN MATCHED THEN
        UPDATE SET 
            TotalSales = source.TotalSales,
            DemandCV = source.DemandCV,
            DaysSold = source.DaysSold,
            Cluster = source.Cluster,
            CreatedAt = GETDATE()
    WHEN NOT MATCHED THEN
        INSERT (SKU, TotalSales, DemandCV, DaysSold, Cluster, CreatedAt)
        VALUES (source.SKU, source.TotalSales, source.DemandCV, source.DaysSold, source.Cluster, GETDATE());
    """

    cursor = conn.cursor()
    for _, row in features_df.iterrows():
        cursor.execute(insert_query, row['SKU'], int(row['TotalSales']), float(row['DemandCV']), int(row['DaysSold']), int(row['Cluster']))
    conn.commit()
    cursor.close()
    conn.close()

    return features_df 