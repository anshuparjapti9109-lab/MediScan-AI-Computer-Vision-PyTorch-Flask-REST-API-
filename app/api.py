"""
MediScan AI - Flask REST API
Endpoints: /api/analyze, /api/health, /api/classes
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import sys
import os
import base64
import io
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import get_engine, CLASSES

app = Flask(__name__, static_folder='../static', template_folder='../templates')
CORS(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────
#  GET /api/health
# ─────────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model": "ResNet-50",
        "version": "1.0.0",
        "classes": 14,
        "device": "CPU",
        "quantization": "dynamic"
    })


# ─────────────────────────────────────────
#  GET /api/classes
# ─────────────────────────────────────────
@app.route('/api/classes', methods=['GET'])
def get_classes():
    return jsonify({"classes": CLASSES, "count": len(CLASSES)})


# ─────────────────────────────────────────
#  POST /api/analyze
# ─────────────────────────────────────────
@app.route('/api/analyze', methods=['POST'])
def analyze():
    start_time = time.time()

    # ── Validate input ──────────────────
    if 'image' not in request.files:
        return jsonify({"error": "No image file in request. Use key 'image'"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Use: {ALLOWED_EXTENSIONS}"}), 400

    image_bytes = file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        return jsonify({"error": "File too large. Max 10MB"}), 413

    # ── Run inference pipeline ──────────
    try:
        engine = get_engine()
        result = engine.predict(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    inference_time = round((time.time() - start_time) * 1000, 2)

    # ── Encode Grad-CAM image ───────────
    gradcam_b64 = None
    if result.get("gradcam_data") is not None:
        try:
            gradcam_arr = result["gradcam_data"]
            gradcam_img = Image.fromarray(gradcam_arr.astype(np.uint8))
            buf = io.BytesIO()
            gradcam_img.save(buf, format="PNG")
            gradcam_b64 = base64.b64encode(buf.getvalue()).decode()
        except Exception:
            gradcam_b64 = None

    # ── Build response ──────────────────
    response = {
        "success": True,
        "inference_time_ms": inference_time,
        "top_finding": result["top_finding"],
        "detected_conditions": result["detected_conditions"],
        "all_predictions": result["predictions"],
        "report": result["report"],
        "gradcam_image": gradcam_b64,
        "metadata": {
            "filename": file.filename,
            "model": "ResNet-50",
            "dataset": "NIH ChestX-ray14",
            "threshold": 0.5
        }
    }

    return jsonify(response)


# ─────────────────────────────────────────
#  POST /api/analyze/base64  (alternative)
# ─────────────────────────────────────────
@app.route('/api/analyze/base64', methods=['POST'])
def analyze_base64():
    """Accept base64 encoded image for JS frontend"""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "JSON body with 'image' key (base64) required"}), 400

    try:
        image_b64 = data['image']
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        return jsonify({"error": "Invalid base64 image data"}), 400

    start_time = time.time()
    try:
        engine = get_engine()
        result = engine.predict(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    inference_time = round((time.time() - start_time) * 1000, 2)

    gradcam_b64 = None
    if result.get("gradcam_data") is not None:
        try:
            gradcam_arr = result["gradcam_data"]
            gradcam_img = Image.fromarray(gradcam_arr.astype(np.uint8))
            buf = io.BytesIO()
            gradcam_img.save(buf, format="PNG")
            gradcam_b64 = base64.b64encode(buf.getvalue()).decode()
        except Exception:
            gradcam_b64 = None

    return jsonify({
        "success": True,
        "inference_time_ms": inference_time,
        "top_finding": result["top_finding"],
        "detected_conditions": result["detected_conditions"],
        "all_predictions": result["predictions"],
        "report": result["report"],
        "gradcam_image": gradcam_b64,
    })


# ─────────────────────────────────────────
#  Serve frontend
# ─────────────────────────────────────────
@app.route('/')
def index():
    return send_file('../templates/index.html')


if __name__ == '__main__':
    print("🩺 MediScan AI starting...")
    print("   Model: ResNet-50 | 14 classes | NIH ChestX-ray14")
    app.run(debug=True, host='0.0.0.0', port=5000)
