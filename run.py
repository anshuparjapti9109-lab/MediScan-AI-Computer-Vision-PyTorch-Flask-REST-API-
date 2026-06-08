"""
MediScan AI - Entry Point
Run: python run.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.api import app

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  🩺  MediScan AI — Medical Image Analysis")
    print("="*50)
    print("  Model    : ResNet-50")
    print("  Classes  : 14 (NIH ChestX-ray14)")
    print("  Backend  : Flask REST API")
    print("  Frontend : http://localhost:5000")
    print("  API Docs : http://localhost:5000/api/health")
    print("="*50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
