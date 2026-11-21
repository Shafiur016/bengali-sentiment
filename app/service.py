"""Flask inference service; requires a trained model artifact."""
import os
from pathlib import Path
from flask import Flask, request, jsonify
from .inference import predict

app = Flask(__name__)
MODEL_FILE = os.getenv('MODEL_FILE', 'artifacts/sentiment_cnn_lstm.pt')

@app.get('/healthz')
def health():
    return jsonify({'status': 'ok'})

@app.get('/readyz')
def ready():
    if not Path(MODEL_FILE).exists():
        return jsonify({'ready': False, 'message': 'Train model first'}), 503
    return jsonify({'ready': True})

@app.post('/predict')
def classify():
    body = request.get_json(silent=True) or {}
    text = body.get('text')
    if not isinstance(text, str) or not text.strip():
        return jsonify({'error': 'text must be a nonempty string'}), 400
    if len(text) > 10000:
        return jsonify({'error': 'text must be at most 10000 characters'}), 413
    if not Path(MODEL_FILE).exists():
        return jsonify({'error': 'Model not trained. See README.'}), 503
    return jsonify(predict(text, MODEL_FILE))
