"""Dataset ingestion and leakage-aware, stratified splitting for Bengali text."""
import csv
import re
from collections import Counter
from sklearn.model_selection import train_test_split


def normalize_text(text: str) -> str:
    return ' '.join(str(text).strip().split())


def tokenize(text: str) -> list[str]:
    text = normalize_text(text).lower()
    return re.findall(r'[\w]+', text, flags=re.UNICODE)


def validate_records(rows: list[dict]) -> list[dict]:
    if not rows:
        raise ValueError('Dataset must have at least one example')
    result = []
    seen = {}
    for i, row in enumerate(rows, start=1):
        text = normalize_text(row.get('text', ''))
        label = normalize_text(row.get('label', ''))
        if not text or not label:
            raise ValueError(f'Missing text or label on row {i}')
        key = ' '.join(tokenize(text))
        if key in seen:
            raise ValueError(f'Duplicate text on row {i}; deduplicate before splitting to prevent train/test leakage')
        seen[key] = label
        result.append({'text': text, 'label': label})
    if len({row['label'] for row in result}) < 2:
        raise ValueError('At least two sentiment labels are required')
    return result


def read_csv(path: str) -> list[dict]:
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'text', 'label'} <= set(reader.fieldnames or []):
            raise ValueError('CSV must have text,label columns')
        return validate_records(list(reader))


def split_records(rows: list[dict], test_size: float = 0.2, seed: int = 42):
    rows = validate_records(rows)
    counts = Counter(row['label'] for row in rows)
    if min(counts.values()) < 2:
        raise ValueError('Need at least two examples per class for stratified split')
    train, test = train_test_split(rows, test_size=test_size, stratify=[r['label'] for r in rows], random_state=seed)
    return train, test


def build_vocab(texts: list[str], max_words: int = 12000) -> dict[str, int]:
    counts = Counter(token for text in texts for token in tokenize(text))
    return {'<PAD>': 0, '<UNK>': 1, **{w: i + 2 for i, (w, _) in enumerate(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_words - 2])}}


def encode(text: str, vocab: dict[str, int], max_len: int = 64) -> list[int]:
    ids = [vocab.get(word, 1) for word in tokenize(text)[:max_len]]
    return ids + [0] * (max_len - len(ids))
