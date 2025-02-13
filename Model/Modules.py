import torch
from torch import nn
import math

# Set device to CPU explicitly
device = torch.device('cpu')

# Helper Function: Shape Validation
def validate_shape(tensor, expected_shape, name="Tensor"):
    assert tensor.shape == expected_shape, f"{name} has unexpected shape {tensor.shape}, expected {expected_shape}"

# Feature Extraction, Preference Detection
class Module_1_6(nn.Module):
    def __init__(self, device, n_items, embedding_dim, dropout, gru=None):
        super(Module_1_6, self).__init__()
        self.device = device
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_size = embedding_dim
        self.gru = gru  # Accept the GRU layer or initialize it
        self.designModelArch(dropout)

    def designModelArch(self, dropout):
        n_embeddings = self.n_items
        self.item_embedding = nn.Embedding((n_embeddings + 1), self.embedding_dim, padding_idx=n_embeddings)
        self.hour_embedding_a = nn.Embedding(25, self.embedding_dim, padding_idx=24)
        self.minute_embedding_a = nn.Embedding(61, self.embedding_dim, padding_idx=60)
        self.second_embedding_a = nn.Embedding(61, self.embedding_dim, padding_idx=60)

        self.hour_embedding_b = nn.Embedding(25, self.embedding_dim, padding_idx=24)
        self.minute_embedding_b = nn.Embedding(61, self.embedding_dim, padding_idx=60)
        self.second_embedding_b = nn.Embedding(61, self.embedding_dim, padding_idx=60)

        self.bias_last = torch.nn.Parameter(torch.Tensor(self.n_items))
        self.dropout = nn.Dropout(dropout)

    def forward(self, session, h_a_o, m_a_o, s_a_o, h_b_o, m_b_o, s_b_o):
        session = session.long()
        session_embedding = self.item_embedding(session)

        h_b_o = torch.clamp(h_b_o, 0, 23).long().to(self.device)
        m_b_o = torch.clamp(m_b_o, 0, 60).long().to(self.device)
        s_b_o = torch.clamp(s_b_o, 0, 60).long().to(self.device)

        h_a_o = torch.clamp(h_a_o, 0, 23).long().to(self.device)
        m_a_o = torch.clamp(m_a_o, 0, 60).long().to(self.device)
        s_a_o = torch.clamp(s_a_o, 0, 60).long().to(self.device)

        hour_embedding_a = self.hour_embedding_a(h_a_o)
        minute_embedding_a = self.minute_embedding_a(m_a_o)
        second_embedding_a = self.second_embedding_a(s_a_o)

        hour_embedding_b = self.hour_embedding_b(h_b_o)
        minute_embedding_b = self.minute_embedding_b(m_b_o)
        second_embedding_b = self.second_embedding_b(s_b_o)

        embeddings = [
            session_embedding,
            hour_embedding_a,
            minute_embedding_a,
            second_embedding_a,
            hour_embedding_b,
            minute_embedding_b,
            second_embedding_b,
        ]

        return embeddings

    def get_scores(self, output):
        # Assuming the output is the logits from a neural network layer and you want to return them as-is
        return output  # Or apply activation like softmax if needed



class Module_2(nn.Module):
    def __init__(self, device, embedding_dim, hidden_size, dropout):
        super(Module_2, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.designModelArch()

    def designModelArch(self):
        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = self.dropout_layer(gru_out)
        return gru_out


class Module_3(nn.Module):
    def __init__(self, device, embedding_dim, dropout):
        super(Module_3, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.designModelArch()

    def designModelArch(self):
        self.a_linear = nn.Linear(self.embedding_dim * 3, self.embedding_dim)
        self.b_linear = nn.Linear(self.embedding_dim * 3, self.embedding_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, a_hour, a_minute, a_second, b_hour, b_minute, b_second):
        a_embedding = torch.cat((a_hour, a_minute, a_second), dim=-1)
        b_embedding = torch.cat((b_hour, b_minute, b_second), dim=-1)

        a_embedding = self.a_linear(a_embedding)
        b_embedding = self.b_linear(b_embedding)

        a_embedding = self.dropout_layer(a_embedding)
        b_embedding = self.dropout_layer(b_embedding)

        return b_embedding, a_embedding


class Module_4(nn.Module):
    def __init__(self, device, dropout):
        super(Module_4, self).__init__()
        self.device = device
        self.dropout = dropout
        self.designModelArch()

    def designModelArch(self):
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, gru_out, b_embedding, a_embedding):
        

        # Adjust b_embedding to match the shape of gru_out
        b_embedding_expanded = b_embedding.unsqueeze(0).expand(gru_out.size(0), -1, b_embedding.size(1), b_embedding.size(2))
        b_embedding_expanded = b_embedding_expanded.view(gru_out.size(0), -1, gru_out.size(2))

        # Adjust a_embedding to match the shape of gru_out
        a_embedding_expanded = a_embedding.unsqueeze(0).expand(gru_out.size(0), -1, a_embedding.size(1), a_embedding.size(2))
        a_embedding_expanded = a_embedding_expanded.view(gru_out.size(0), -1, gru_out.size(2))

        

        # Now perform element-wise multiplication
        weighted_gru_out = gru_out * b_embedding_expanded
        weighted_a_embedding = a_embedding_expanded * weighted_gru_out

        # Apply dropout
        weighted_gru_out = self.dropout_layer(weighted_gru_out)
        weighted_a_embedding = self.dropout_layer(weighted_a_embedding)

        # Return the weighted outputs
        return weighted_gru_out, weighted_a_embedding, gru_out, b_embedding, a_embedding






class Module_5(nn.Module):
    def __init__(self, device, embedding_dim, hidden_size, dropout=0.0, noise_epsilon=0.0):
        super(Module_5, self).__init__()
        self.device = device
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 512)  # This should output 512 units
        self.dropout_layer = nn.Dropout(dropout)
        self.noise_epsilon = noise_epsilon

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(1, x.size(0), x.size(2)).to(x.device)
        c0 = torch.zeros(1, x.size(0), x.size(2)).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        #print(f"LSTM output shape: {out.shape}")  # Print shape of LSTM output

        # Take only the last time step output
        out = out[:, -1, :]
        

        # Add noise if needed
        noise = torch.randn_like(out) * self.noise_epsilon
        out = out + noise.clamp(min=-1.0, max=1.0)

        # Pass through first fully connected layer (fc1)
        out = torch.relu(self.fc1(out))
        #print(f"Output after first FC layer: {out.shape}")  # Print shape after first FC layer

        # Apply dropout
        out = self.dropout_layer(out)

        # Apply fc2 layer
        out = self.fc2(out)
        #print(f"Output after applying fc2: {out.shape}")  # Print shape after fc2

        # Ensure output shape is [1, 512] before returning
        if out.shape != (1, 512):
            raise ValueError(f"Expected output shape [1, 512], got {out.shape}")

        return out








