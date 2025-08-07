# clusters.py

from flask import Flask, jsonify
from flask_cors import CORS
from clusters import run_inventory_clustering
from db_utils import get_connection, fetch_cluster_data
from config import CONFIG
app = Flask(__name__)
CORS(app)

@app.route('/api/getclusters', methods=['GET'])
def run_clustering():
    try:
        result_df = run_inventory_clustering()
        summary = result_df.groupby('Cluster').agg({
            'SKU': 'count',
            'TotalSales': 'mean',
            'DemandCV': 'mean',
            'DaysSold': 'mean'
        }).reset_index().round(2)

        return jsonify(summary.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clusterdata', methods=['GET'])
def get_cluster_data():
    try:
        conn = get_connection(CONFIG['sql']['server'], CONFIG['sql']['database'])
        df = fetch_cluster_data(conn)
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)