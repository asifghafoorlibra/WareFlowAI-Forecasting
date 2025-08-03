# clusters_api.py

from flask import Flask, jsonify
from clusters import run_inventory_clustering

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True)