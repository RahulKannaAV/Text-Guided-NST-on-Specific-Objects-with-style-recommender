import re
import pandas as pd
import torch
from torch import nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# 1. Configuration
# -----------------------------

CSV_PATH = "results.csv"

METRICS = ["Color", "Structure", "Semantic"]

# True means higher scores are better.
# Change to False for metrics where lower scores are better.
HIGHER_IS_BETTER = {
    "Color": True,
    "Structure": True,
    "Semantic": True,
}

EMBED_DIM = 8
HIDDEN_DIM = 32
EPOCHS = 300
BATCH_SIZE = 4
LEARNING_RATE = 0.005

torch.manual_seed(42)

# -----------------------------
# 2. Read the original dataset
# -----------------------------

df = pd.read_csv(CSV_PATH)

def parse_label(value):
    match = re.fullmatch(
        r"(.+)_on_(.+)", str(value).strip()
    )
    if not match:
        raise ValueError(
            f"Invalid ContentStyle label: {value}"
        )
    return match.group(1), match.group(2)

parsed = df["ContentStyle"].apply(parse_label)

df["Content"] = parsed.apply(lambda x: x[0])
df["Style"] = parsed.apply(lambda x: x[1])

contents = sorted(df["Content"].unique())
styles = sorted(df["Style"].unique())
metrics = METRICS

content_to_id = {
    name: i for i, name in enumerate(contents)
}
style_to_id = {
    name: i for i, name in enumerate(styles)
}
metric_to_id = {
    name: i for i, name in enumerate(metrics)
}

# -----------------------------
# 3. Derive training targets
# -----------------------------
# One example = (content, metric) -> best style

examples = []

for content in contents:
    subset = df[df["Content"] == content]

    for metric in metrics:
        scores = subset[metric].astype(float)

        if scores.isna().any():
            raise ValueError(
                f"Missing {metric} scores for {content}"
            )

        if HIGHER_IS_BETTER[metric]:
            best_idx = scores.idxmax()
        else:
            best_idx = scores.idxmin()

        best_style = df.loc[best_idx, "Style"]

        examples.append((
            content_to_id[content],
            metric_to_id[metric],
            style_to_id[best_style],
        ))

X_content = torch.tensor(
    [x[0] for x in examples], dtype=torch.long
)
X_metric = torch.tensor(
    [x[1] for x in examples], dtype=torch.long
)
y = torch.tensor(
    [x[2] for x in examples], dtype=torch.long
)

dataset = TensorDataset(X_content, X_metric, y)
loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True
)

# -----------------------------
# 4. Define the neural network
# -----------------------------

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


model = StyleSelector(
    num_contents=len(contents),
    num_metrics=len(metrics),
    num_styles=len(styles),
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)

# -----------------------------
# 5. Train
# -----------------------------

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for content_ids, metric_ids, targets in loader:
        optimizer.zero_grad()

        logits = model(content_ids, metric_ids)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(targets)

    if (epoch + 1) % 50 == 0:
        avg_loss = total_loss / len(dataset)
        print(
            f"Epoch {epoch+1}/{EPOCHS}, "
            f"Loss: {avg_loss:.4f}"
        )

# -----------------------------
# 6. Predict the best style
# -----------------------------

def predict_style(content_label, metric_label):
    if content_label not in content_to_id:
        raise ValueError(
            f"Unknown content: {content_label}. "
            "This ID-based model only supports "
            "contents seen during training."
        )

    if metric_label not in metric_to_id:
        raise ValueError(
            f"Unknown metric: {metric_label}"
        )

    model.eval()

    content_id = torch.tensor(
        [content_to_id[content_label]]
    )
    metric_id = torch.tensor(
        [metric_to_id[metric_label]]
    )

    with torch.no_grad():
        logits = model(content_id, metric_id)
        probabilities = torch.softmax(logits, dim=1)
        predicted_id = probabilities.argmax(dim=1).item()

    return {
        "content": content_label,
        "metric": metric_label,
        "predicted_style": styles[predicted_id],
        "probabilities": {
            style: round(probabilities[0, i].item(), 4)
            for i, style in enumerate(styles)
        },
    }


# Example inference
print(predict_style("Bear", "Color"))
print(predict_style("Squirrel", "Semantic"))
# print(predict_style("C2", "E3"))

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), MODEL_PATH / "rec_model_weights.pth")
print("Model weights saved successfully.")