import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Load your CSV with logits
df = pd.read_csv("/lab/projects/fire_smoke_awr/src/pipeline/2-stage/A_phash10_single_objects_pred_lt_100.csv")

# Extract logits and labels
logits = torch.tensor(df[["logit_bg", "logit_fire", "logit_smoke"]].values, dtype=torch.float32)
labels = torch.tensor(df["label"].values, dtype=torch.long)  # assumes 0=bg, 1=fire/smoke, 2=smoke if multiclass

# Criterion
nll = nn.CrossEntropyLoss()

# Temperature parameter
T = nn.Parameter(torch.ones(1) * 1.0)

# Optimizer
optimizer = optim.LBFGS([T], lr=0.01, max_iter=50)

def closure():
    optimizer.zero_grad()
    loss = nll(logits / T, labels)
    loss.backward()
    return loss

optimizer.step(closure)

print(f"✅ Optimal temperature: {T.item():.4f}")
