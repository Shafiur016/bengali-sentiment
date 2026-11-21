"""Trainable CNN → bidirectional LSTM classifier, not a pretrained model."""
import torch
from torch import nn

class CNNLSTM(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, embed_dim: int = 64, conv_channels: int = 96, hidden_dim: int = 48):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Sequential(nn.Conv1d(embed_dim, conv_channels, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2))
        self.lstm = nn.LSTM(conv_channels, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(hidden_dim * 2, num_classes))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        sequence = self.embedding(tokens).transpose(1, 2)
        sequence = self.conv(sequence).transpose(1, 2)
        _, (h, _) = self.lstm(sequence)
        return self.classifier(torch.cat((h[-2], h[-1]), dim=1))
