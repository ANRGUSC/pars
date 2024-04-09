import torch
import torch.nn as nn
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')


class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        # embed_size: Dimensionality of the input feature vector.

        self.embed_size = embed_size
        
        # Linear transformations for input sequences.
        # These will project the input sequences into "value", "key", and "query" spaces.
        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False) # Transformation for values.
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False) # Transformation for keys.
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False) # Transformation for queries.

    def forward(self, values, keys, query):
        # Apply linear transformations to project inputs into value, key, and query spaces.
        values = self.values(values) # Transformed values.
        keys = self.keys(keys) # Transformed keys.
        queries = self.queries(query) # Transformed queries.
        
        # Compute the dot product of queries and keys to get the raw attention scores.
        attn_values = torch.bmm(queries, keys.transpose(1, 2))
        # Softmax is applied to normalize the attention scores on the last dimension.
        attention = torch.nn.functional.softmax(attn_values, dim=2)
        
        # Multiply the normalized attention scores with the values to get the output.
        # This step aggregates the information based on the computed attention weights.
        out = torch.bmm(attention, values)
        return out


class AttentionGRU(nn.Module):
    def __init__(self, input_dim, embed_size, hidden_dim, output_dim):
        super(AttentionGRU, self).__init__()
        # input_dim: The size of each input feature vector.
        # embed_size: The size of the embedding vector that the input features will be mapped to.
        # hidden_dim: The size of the GRU's hidden state.
        # output_dim: The size of the output vector after processing by the network.

        self.input_dim = input_dim
        # A linear layer that maps the input features to the embedding space.
        self.embedding = nn.Linear(input_dim, embed_size)
        # The self-attention mechanism defined earlier, taking embeddings as input.
        self.attention = SelfAttention(embed_size)
        # GRU layer that processes sequences, with input dimensions from the embedding size to the hidden state size.
        self.gru = nn.GRU(embed_size, hidden_dim, batch_first=True)
        # Final linear layer that maps the GRU's hidden state to the desired output size.
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Reshape the input x to have dimensions: batch size, sequence length, input dimension.
        x = x.view(x.size(0), -1, self.input_dim)
        # Apply the embedding layer to each element in the input sequence.
        x = self.embedding(x)
        # Apply the self-attention mechanism to the embedded input.
        x = self.attention(x, x, x)
        # Process the output of the attention mechanism with the GRU layer.
        # The GRU returns its last layer's output for each timestep, as well as the last hidden state.
        _, h_n = self.gru(x)
        # The final hidden state (h_n) is squeezed to remove the first dimension, then passed through the final linear layer to produce the output.
        out = self.fc_out(h_n.squeeze(0))
        return out






