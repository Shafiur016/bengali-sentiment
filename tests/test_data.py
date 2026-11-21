import pytest
from app.data import normalize_text, validate_records, split_records


def test_normalization_preserves_bengali_text():
    assert normalize_text('  এই   খাবার\n দারুণ!  ') == 'এই খাবার দারুণ!'


def test_data_validation_rejects_single_class_and_missing_text():
    with pytest.raises(ValueError):
        validate_records([{'text': 'ভালো', 'label': 'positive'}, {'text': 'খুব ভালো', 'label': 'positive'}])
    with pytest.raises(ValueError):
        validate_records([{'text': '', 'label': 'negative'}, {'text': 'ভালো', 'label': 'positive'}])


def test_stratified_split_retains_classes():
    rows = [{'text': f'নমুনা {i}', 'label': 'positive' if i < 12 else 'negative'} for i in range(24)]
    train, test = split_records(rows, test_size=0.25, seed=9)
    assert len(train) == 18 and len(test) == 6
    assert {r['label'] for r in test} == {'positive', 'negative'}


def test_duplicate_text_is_rejected_to_prevent_split_leakage():
    with pytest.raises(ValueError, match='Duplicate'):
        validate_records([{'text': 'খুব ভালো', 'label': 'positive'}, {'text': 'খুব ভালো', 'label': 'positive'}, {'text': 'খুব খারাপ', 'label': 'negative'}])
