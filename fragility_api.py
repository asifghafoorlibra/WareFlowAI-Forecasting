from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
from predict_fragility import predict_fragility  
 

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)
# --- Helpers ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes ---
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'message': 'Image uploaded successfully', 'filename': filename}), 200

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/detect', methods=['POST'])
def detect_fragility():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({'error': 'Filename is required'}), 400

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(image_path):
        return jsonify({'error': 'File not found'}), 404

    try:
        # Redirect stdout to capture printed output
        import io
        import sys
        buffer = io.StringIO()
        sys.stdout = buffer

        predict_fragility(image_path)  # Calls your existing function

        sys.stdout = sys.__stdout__
        output = buffer.getvalue()

        # Extract prediction line from output
        lines = output.strip().split('\n')
        prediction_line = next((line for line in lines if line.startswith("Prediction:")), None)

        if prediction_line:
            label, confidence = prediction_line.replace("Prediction:", "").strip().split("(", 1)
            confidence = confidence.replace("% confidence)", "").strip()
            return jsonify({
                'label': label.strip(),
                'confidence': float(confidence)
            }), 200
        else:
            return jsonify({'error': 'Prediction failed'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Run ---
if __name__ == '__main__':
    app.run(debug=True)