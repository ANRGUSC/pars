import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class AttentionTransformer(nn.Module):
    def __init__(self, input_dim, embed_size, num_heads, hidden_dim, output_dim, num_layers):
        super(AttentionTransformer, self).__init__()
        self.input_dim = input_dim
        self.embed_size = embed_size
        # Linear layer to project the input features to the embedding space.
        self.embedding = nn.Linear(input_dim, embed_size)
        # Transformer Encoder Layer
        encoder_layers = TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_layers)
        # Output linear layer
        self.fc_out = nn.Linear(embed_size, output_dim)

    def forward(self, x):
        # Reshape the input x to have dimensions: batch size, sequence length, input dimension.
        x = x.view(x.size(0), -1, self.input_dim)
        # Apply the embedding layer to each element in the input sequence.
        x = self.embedding(x)
        # Adjust x's dimensions to fit the transformer's expected input shape: (seq_length, batch, embed_size)
        x = x.permute(1, 0, 2)
        # Apply the transformer encoder to the embedded input.
        x = self.transformer_encoder(x)
        # Take the last sequence element's output for the final prediction.
        out = self.fc_out(x[-1])
        return out
