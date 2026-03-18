from torch import nn


class ActionEncoder(nn.Module):
    def __init__(self, num_actions: int = 9, embedding_dim: int = 32, output_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, embedding_dim)
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, action_ids):
        embedding = self.embedding(action_ids)
        return self.proj(embedding)
