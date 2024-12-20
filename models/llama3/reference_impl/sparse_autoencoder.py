import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()
    def encode(self, x):
        # Subtract decoder bias before encoding
        x = x - self.decoder.bias
        return self.relu(self.encoder(x))
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    def normalize_decoder_weights(self):
        """Normalize decoder weights to prevent degenerate solutions"""
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(
                self.decoder.weight.data, p=2, dim=1
            )