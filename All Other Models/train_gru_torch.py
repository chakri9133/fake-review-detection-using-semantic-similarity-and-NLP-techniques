import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =====================================================
# 1. CONFIG
# =====================================================
BASE_DIR = r"D:\Fake Review Detection"
DATA_PATH = BASE_DIR + r"\Datasets\kindle_reviews_clean.csv"

MAX_LEN = 120
BATCH_SIZE = 64
EPOCHS = 5
EMB_DIM = 100
HIDDEN_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# =====================================================
# 2. LOAD DATA
# =====================================================
df = pd.read_csv(DATA_PATH)
texts = df["clean_text"].astype(str).tolist()
labels = df["label"].values

# =====================================================
# 3. TOKENIZER + VOCAB
# =====================================================
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

vocab = {"<PAD>": 0, "<UNK>": 1}
for text in texts:
    for word in tokenize(text):
        if word not in vocab:
            vocab[word] = len(vocab)

# =====================================================
# 4. DATASET
# =====================================================
class ReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def encode(self, text):
        tokens = tokenize(text)[:MAX_LEN]
        ids = [vocab.get(w, 1) for w in tokens]
        return ids + [0] * (MAX_LEN - len(ids))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encode(self.texts[idx])),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# =====================================================
# 5. SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

train_loader = DataLoader(
    ReviewDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    ReviewDataset(X_test, y_test),
    batch_size=BATCH_SIZE
)

# =====================================================
# 6. GRU MODEL
# =====================================================
class GRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), EMB_DIM, padding_idx=0)
        self.gru = nn.GRU(
            EMB_DIM,
            HIDDEN_DIM,
            batch_first=True
        )
        self.fc = nn.Linear(HIDDEN_DIM, 2)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))

model = GRUClassifier().to(DEVICE)

# =====================================================
# 7. TRAINING
# =====================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training GRU...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.2f}")

# =====================================================
# 8. EVALUATION
# =====================================================
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for x, y in test_loader:
        preds = torch.argmax(model(x.to(DEVICE)), dim=1)
        y_true.extend(y.numpy())
        y_pred.extend(preds.cpu().numpy())

print("\n=== GRU CLASSIFICATION REPORT ===\n")
print(classification_report(y_true, y_pred))
