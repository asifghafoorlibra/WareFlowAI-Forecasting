import pyodbc
import pandas as pd

def get_connection(server, database):
    return pyodbc.connect(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;"
    )

def fetch_sales_data(conn, table, sku=None):
    query = f"""
        SELECT SaleDate, SUM(UnitsSold) AS TotalUnits
        FROM {table}
        {"WHERE SKU = '" + sku + "'" if sku else ""}
        GROUP BY SaleDate
        ORDER BY SaleDate
    """
    df = pd.read_sql(query, conn)
    df.rename(columns={"SaleDate": "ds", "TotalUnits": "y"}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'])
    return df

def fetch_bulk_sales(conn, table, lookback_days=90):
    query = f"""
        SELECT SKU, SaleDate, UnitsSold
        FROM {table}
        WHERE SaleDate >= DATEADD(DAY, -{lookback_days}, GETDATE())
    """
    return pd.read_sql(query, conn)

def fetch_forecast_data(conn):
    query = """
        SELECT 
            Id,
            SKU,
            ForecastDate,
            ForecastedDemand,
            ReorderPoint,
            ModelUsed,
            ConfidenceIntervalLow,
            ConfidenceIntervalHigh,
            CreatedAt
        FROM ForecastResults  
        ORDER BY ForecastDate DESC
    """
    return pd.read_sql(query, conn)


def fetch_cluster_data(conn):
    query = """
        SELECT 
            SKU,
            TotalSales,
            DemandCV,
            DaysSold,
            Cluster,
            CreatedAt
        FROM SKUClusters
        ORDER BY CreatedAt DESC
    """
    df = pd.read_sql(query, conn)
    df['CreatedAt'] = pd.to_datetime(df['CreatedAt'])
    return df