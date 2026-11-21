# Bengali Sentiment Analysis — CNN–BiLSTM + Flask API

PyTorch text classifier with a convolutional feature extractor and bidirectional LSTM; Bengali Unicode preprocessing, training-only vocabulary, stratified train/validation/test splits, confusion matrix and classification report. Includes a Flask inference service.

## Install and train (Python 3.11/3.12, CPU or GPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python data/generate_demo.py
python -m app.train --data data/demo_reviews.csv --epochs 3 --output artifacts
python -m flask --app app.service run --port 5000
```

API:

```bash
curl -X POST http://127.0.0.1:5000/predict -H 'Content-Type: application/json' \
 -d '{"text":"খাবারটি খুব সুস্বাদু"}'
```

Or after training: `docker compose up --build` for the Flask API.

## Your real dataset

Replace the tiny **hand-authored demo examples** with a suitable Bengali sentiment dataset in UTF-8 CSV format:

```csv
text,label
"খাবারটি ভালো ছিল",positive
"সেবাটি খারাপ ছিল",negative
"আজ দোকানটি খোলা",neutral
```

The split must contain enough examples per class (especially if you use 3 labels). If duplicated sentences share sources, split by source/group to avoid leakage; the default stratification alone does not handle group leakage. Recheck label quality, imbalance and out-of-domain performance.

## Output and limitations

- `artifacts/sentiment_cnn_lstm.pt`: model parameters, train-only vocabulary, class labels.
- `artifacts/metrics.json`: measured held-out accuracy, precision/recall/F1, confusion matrix and sample counts.
- **No 86% accuracy claim is made**: such a number cannot be verified without the original dataset, split and trained weights. Demo dataset is not a valid benchmark.
- This is a scratch-trained model, not a multilingual pretrained transformer. The Flask app has no authentication/rate limiting; protect it for public deployment.
