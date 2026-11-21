"""Reusable inference code shared between CLI and Flask HTTP service."""
from functools import lru_cache
import torch
from .model import CNNLSTM
from .data import encode

@lru_cache(maxsize=4)
def load_bundle(path: str):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = CNNLSTM(vocab_size=len(checkpoint['vocab']), num_classes=len(checkpoint['classes']))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint

def predict(text: str, path: str) -> dict:
    model, config = load_bundle(path)
    tokens = torch.tensor([encode(text, config['vocab'], config['max_len'])], dtype=torch.long)
    with torch.no_grad():
        scores = torch.softmax(model(tokens), dim=1)[0].tolist()
    pairs = sorted(zip(config['classes'], scores), key=lambda x: x[1], reverse=True)
    return {'label': pairs[0][0], 'confidence': round(pairs[0][1], 4),
            'probabilities': {label: round(prob, 4) for label, prob in pairs}}
