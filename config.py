CONFIG = {
    "sql": {
        "server": "DESKTOP-M64NLAS",
        "database": "AIWareFlowDB",
        "table": "SalesHistory",
        "sku_filter": None  # or 'ABC123'
    },
    "forecast": {
        "horizon": 30,
        "include_promo": True
    },
    "output_path": "models/prophet/forecast.csv"
}