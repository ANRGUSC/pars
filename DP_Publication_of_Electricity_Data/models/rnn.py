import torch
import torch.nn as nn

class AttentionRNN(nn.Module):
    def __init__(self, input_dim, embed_size, hidden_dim, output_dim):
        super(AttentionRNN, self).__init__()
        self.input_dim = input_dim
        # Linear layer to project the input features to the embedding space.
        self.embedding = nn.Linear(input_dim, embed_size)
        # RNN layer that processes sequences, taking input from the embedding size to the hidden state size.
        self.rnn = nn.RNN(embed_size, hidden_dim, batch_first=True)
        # Output linear layer that maps the RNN's hidden state to the output size.
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Reshape the input x to have dimensions: batch size, sequence length, input dimension.
        x = x.view(x.size(0), -1, self.input_dim)
        # Apply the embedding layer to each element in the input sequence.
        x = self.embedding(x)
        # Process the embedded input with the RNN layer.
        # The RNN returns its output for each timestep, as well as the last hidden state.
        _, h_n = self.rnn(x)
        # The final hidden state is passed through the output linear layer to produce the model's output.
        out = self.fc_out(h_n.squeeze(0))
        return out
