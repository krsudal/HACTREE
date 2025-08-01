#%%
# train_hs_model.py
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# 🔧 설정
# 현재 파일 기준으로 절대 경로 지정
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "sample_hs_data_10000.csv")
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
VOCAB_PATH = BASE_DIR / "vocab.json"
MODEL_PATH = BASE_DIR / "hs_model.pt"
TOKENIZER_NAME = "jhgan/ko-sroberta-multitask"
EMBEDDING_DIM = 100
HIDDEN_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5
MAX_LEN = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔤 데이터 로드 및 전처리
df = pd.read_csv(DATA_PATH)
le = LabelEncoder()
df["label"] = le.fit_transform(df["hs_code"])

# 저장
with open(LABEL_ENCODER_PATH, "wb") as f:
    pickle.dump(le, f)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
with open(VOCAB_PATH, "w") as f:
    json.dump(tokenizer.get_vocab(), f)

# 📦 Dataset
class HSDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 🧠 모델 정의
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# 📊 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(
    df["item_name"], df["label"], test_size=0.2, stratify=df["label"])

train_dataset = HSDataset(X_train.tolist(), y_train.tolist())
val_dataset = HSDataset(X_val.tolist(), y_val.tolist())
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 🚀 모델 학습
model = BiLSTMClassifier(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    num_classes=len(le.classes_)
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

# 💾 모델 저장
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ 모델 저장 완료: {MODEL_PATH}")


# %%
