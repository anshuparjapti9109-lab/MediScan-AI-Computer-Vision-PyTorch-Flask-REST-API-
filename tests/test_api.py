"""
MediScan AI - API Test Suite
Usage: python tests/test_api.py
"""

import requests
import json
import sys
import os
import base64
import numpy as np
from PIL import Image
import io

BASE_URL = "http://localhost:5000"

def create_dummy_xray():
    """Create a synthetic grayscale X-ray-like image for testing"""
    img_array = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
    # Add circular bright region (simulate lung)
    for i in range(224):
        for j in range(224):
            if (i-80)**2 + (j-70)**2 < 50**2:
                img_array[i, j] = min(255, img_array[i, j] + 80)
            if (i-80)**2 + (j-150)**2 < 50**2:
                img_array[i, j] = min(255, img_array[i, j] + 80)
    img = Image.fromarray(img_array, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf

def test_health():
    print("\n[1] Testing /api/health ...")
    r = requests.get(f"{BASE_URL}/api/health")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data["status"] == "healthy"
    print(f"    ✅ Status: {data['status']} | Model: {data['model']} | Classes: {data['classes']}")

def test_classes():
    print("\n[2] Testing /api/classes ...")
    r = requests.get(f"{BASE_URL}/api/classes")
    assert r.status_code == 200
    data = r.json()
    assert len(data["classes"]) == 14
    print(f"    ✅ {data['count']} classes returned")
    print(f"    Classes: {', '.join(data['classes'][:5])}...")

def test_analyze_multipart():
    print("\n[3] Testing /api/analyze (multipart/form-data) ...")
    img_buf = create_dummy_xray()
    r = requests.post(
        f"{BASE_URL}/api/analyze",
        files={"image": ("test_xray.png", img_buf, "image/png")}
    )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["success"] == True
    assert "top_finding" in data
    assert "all_predictions" in data
    assert len(data["all_predictions"]) == 14
    print(f"    ✅ Inference time: {data['inference_time_ms']}ms")
    print(f"    Top finding: {data['top_finding']['class']} ({data['top_finding']['probability']:.2%})")
    print(f"    Detected: {len(data['detected_conditions'])} conditions above threshold")

def test_analyze_base64():
    print("\n[4] Testing /api/analyze/base64 (JSON) ...")
    img_buf = create_dummy_xray()
    b64 = base64.b64encode(img_buf.getvalue()).decode()
    r = requests.post(
        f"{BASE_URL}/api/analyze/base64",
        json={"image": b64},
        headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["success"] == True
    print(f"    ✅ Base64 endpoint works | Grad-CAM: {'✅' if data.get('gradcam_image') else '❌'}")

def test_error_no_file():
    print("\n[5] Testing error handling (no file) ...")
    r = requests.post(f"{BASE_URL}/api/analyze")
    assert r.status_code == 400
    print(f"    ✅ Correct 400 error: {r.json()['error']}")

if __name__ == "__main__":
    print("=" * 50)
    print("  MediScan AI - API Test Suite")
    print("=" * 50)

    tests = [test_health, test_classes, test_analyze_multipart, test_analyze_base64, test_error_no_file]
    passed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"    ❌ FAILED: {e}")

    print(f"\n{'='*50}")
    print(f"  Results: {passed}/{len(tests)} tests passed")
    print("=" * 50)
    sys.exit(0 if passed == len(tests) else 1)
