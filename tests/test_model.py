import torch
from app.model import CNNLSTM
from app.data import encode, build_vocab


def test_forward_shape_and_probabilities():
    vocab = build_vocab(['এটি ভালো', 'এটি খারাপ'])
    model = CNNLSTM(len(vocab), 3)
    logits = model(torch.tensor([encode('এটি ভালো', vocab), encode('এটি খারাপ', vocab)]))
    assert logits.shape == (2, 3)
    assert torch.allclose(torch.softmax(logits, 1).sum(1), torch.ones(2), atol=1e-6)
