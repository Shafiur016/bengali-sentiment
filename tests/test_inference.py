import torch
from app.model import CNNLSTM
from app.data import build_vocab
from app.inference import predict


def test_saved_checkpoint_can_be_loaded_for_classification(tmp_path):
    vocab = build_vocab(['খুব ভালো খাবার', 'খুব খারাপ খাবার'])
    model = CNNLSTM(len(vocab), 2)
    file = tmp_path / 'sentiment.pt'
    torch.save({'state_dict': model.state_dict(), 'vocab': vocab, 'classes': ['negative', 'positive'], 'max_len': 64}, file)
    result = predict('খাবার ভালো', str(file))
    assert result['label'] in {'negative', 'positive'}
    assert abs(sum(result['probabilities'].values()) - 1) < 0.01
