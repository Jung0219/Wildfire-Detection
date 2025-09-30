import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os

# ========== CONFIG ==========
CSV_PATH = "/lab/projects/fire_smoke_awr/src/pipeline/2-stage/output.csv"
MODEL_PATH = "/lab/projects/fire_smoke_awr/src/pipeline/2-stage/logreg_model.pt"
EPOCHS = 50
BATCH_SIZE = 32
LR = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ============================

# ---- Load data ----
df = pd.read_csv(CSV_PATH)
X = df[["det_conf", "cls_conf"]].values.astype("float32")
y = df["label"].values.astype("float32")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ---- Define logistic regression model ----
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # 2 features -> 1 logit
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = LogisticRegression(input_dim=2).to(DEVICE)

# ---- Loss and optimizer ----
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---- Training loop ----
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# ---- Evaluation ----
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
        preds = model(xb)
        predicted = (preds >= 0.5).float()
        correct += (predicted == yb).sum().item()
        total += yb.size(0)
print(f"Test Accuracy: {correct/total:.4f}")

# ---- Save model ----
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")

# ---- Example: load and use ----
loaded_model = LogisticRegression(input_dim=2)
loaded_model.load_state_dict(torch.load(MODEL_PATH))
loaded_model.eval()

sample = torch.tensor([[0.7, 0.8]])  # det_conf, cls_conf
print("Example prob:", loaded_model(sample).item())
