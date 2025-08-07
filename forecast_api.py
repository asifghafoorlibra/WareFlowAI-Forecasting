from flask import Flask, request, jsonify
from flask_cors import CORS
from forecast_utils2 import generate_forecast_preview, insert_forecast_results
from config import CONFIG
from db_utils import get_connection, fetch_forecast_data
app = Flask(__name__)
CORS(app)  # Enable CORS for Angular frontend

    
#---------------- USE THIS -------------------

@app.route("/api/forecast", methods=["POST"])
def forecast():
    try:
        data = request.get_json(force=True)

        sku = data.get("sku", "ALL")
        forecast_days = int(data.get("forecast_days", CONFIG["forecast"]["horizon"]))
        noise_level = float(data.get("noise_level", 0.01))

        sql_cfg = CONFIG["sql"]
        preview_df, conn, metrics, _, _ = generate_forecast_preview(
            server=sql_cfg["server"],
            database=sql_cfg["database"],
            sales_table=sql_cfg["table"],
            sku=sku,
            forecast_days=forecast_days,
            noise_level=noise_level
        )

        if preview_df.empty:
            return jsonify({"message": f"No forecast generated for SKU(s): {sku}"}), 404

        insert_forecast_results(conn, preview_df)

        metrics_response = {}
        if metrics:
            mae, rmse, mape, smape = metrics
            metrics_response = {
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "mape": round(mape, 2),
                "smape": round(smape, 2)
            }

        return jsonify({
            "sku": sku,
            "forecast": preview_df.to_dict(orient="records"),
            "metrics": metrics_response
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/forecast/run", methods=["POST"])
def run_forecast_and_return():
    try:
        # Step 1: Trigger forecast generation
        data = request.get_json(force=True)
        with app.test_request_context('/api/forecast', method='POST', json=data):
            response = forecast()

        if isinstance(response, tuple) and response[1] != 200:
            return response  # Return error if forecast failed

        # Step 2: Read forecast data from DB
        sql_cfg = CONFIG["sql"]
        conn = get_connection(sql_cfg["server"], sql_cfg["database"])
        forecast_df = fetch_forecast_data(conn)
        conn.close()

        return jsonify({
            "forecast_data": forecast_df.to_dict(orient="records")
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)

 