"""Usage: python -m app.train --data data/demo_reviews.csv --epochs 3 --output artifacts"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from .data import read_csv, split_records, build_vocab, encode
from .model import CNNLSTM


def train(data: str, output: str, epochs: int = 10, seed: int = 42, max_len: int = 64, batch_size: int = 32):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    records = read_csv(data)
    train_all, test = split_records(records, test_size=0.2, seed=seed)
    train_rows, val_rows = split_records(train_all, test_size=0.2, seed=seed + 1)
    vocab = build_vocab([r['text'] for r in train_rows])
    classes = sorted({r['label'] for r in records})
    label_to_idx = {v: i for i, v in enumerate(classes)}

    def dataset(rows):
        x = torch.tensor([encode(r['text'], vocab, max_len) for r in rows], dtype=torch.long)
        y = torch.tensor([label_to_idx[r['label']] for r in rows], dtype=torch.long)
        return TensorDataset(x, y)

    train_loader = DataLoader(dataset(train_rows), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset(val_rows), batch_size=batch_size)
    test_loader = DataLoader(dataset(test), batch_size=batch_size)
    model = CNNLSTM(vocab_size=len(vocab), num_classes=len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    best_loss = float('inf'); best_state = None
    for epoch in range(epochs):
        model.train(); loss_sum = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); loss_sum += loss.item()
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                val_loss += loss_fn(model(x), y).item()
        history.append({'epoch': epoch + 1, 'train_batch_loss_sum': round(loss_sum, 4), 'val_batch_loss_sum': round(val_loss, 4)})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    truth, predicted = [], []
    with torch.no_grad():
        for x, y in test_loader:
            predicted += model(x).argmax(dim=1).tolist()
            truth += y.tolist()
    metrics = {
        'accuracy': accuracy_score(truth, predicted),
        'classification_report': classification_report(truth, predicted, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(truth, predicted, labels=list(range(len(classes)))).tolist(),
        'classes': classes,
        'sample_counts': {'train': len(train_rows), 'validation': len(val_rows), 'test': len(test)},
        'seed': seed,
        'epochs': epochs,
        'history': history,
        'note': 'Demo-data metrics are illustrative only; evaluate on a real held-out dataset before reporting.'
    }
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'vocab': vocab, 'classes': classes, 'max_len': max_len}, out / 'sentiment_cnn_lstm.pt')
    (out / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'accuracy': metrics['accuracy'], 'sample_counts': metrics['sample_counts'], 'artifact': str(out / 'sentiment_cnn_lstm.pt')}, indent=2))
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/demo_reviews.csv')
    parser.add_argument('--output', default='artifacts')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.seed, batch_size=args.batch_size)
