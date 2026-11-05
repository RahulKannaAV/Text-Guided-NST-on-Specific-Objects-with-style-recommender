import torch
from torch import nn

EMBED_DIM = 8
HIDDEN_DIM = 32

class StyleSelector(nn.Module):
    def __init__(
        self, num_contents, num_metrics, num_styles
    ):
        super().__init__()

        self.content_embedding = nn.Embedding(
            num_contents, EMBED_DIM
        )
        self.metric_embedding = nn.Embedding(
            num_metrics, EMBED_DIM
        )

        self.network = nn.Sequential(
            nn.Linear(EMBED_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, num_styles),
        )

    def forward(self, content_id, metric_id):
        content_vec = self.content_embedding(content_id)
        metric_vec = self.metric_embedding(metric_id)

        combined = torch.cat(
            [content_vec, metric_vec], dim=1
        )

        # Logits for S1, S2, S3, ...
        return self.network(combined)

