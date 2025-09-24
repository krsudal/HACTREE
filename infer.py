#%%
# ✅ 첫 번째 셀: 경로 설정 추가
import os, sys
sys.path.append(os.path.abspath(".."))  # 상위 폴더인 rag_project 기준

import torch
from hs_predict.model import BiLSTMClassifier
import json
import pickle

def predict_hs_code(text, model_path, vocab_path, label_encoder_path, device='cpu'):
    # Load metadata
    vocab = load_vocab(vocab_path)
    label_encoder = load_label_encoder(label_encoder_path)

    # Hyperparameters (must match training)
    vocab_size = len(vocab)
    embedding_dim = 100
    hidden_dim = 128
    output_dim = len(label_encoder.classes_)

    # Load model
    model = BiLSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Preprocess input
    x = text_to_tensor(text, vocab, device)

    # Predict
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()

    return label_encoder.inverse_transform([pred])[0]

def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        return json.load(f)

def text_to_tensor(text, vocab, device='cpu'):
    tokens = text.lower().split()
    ids = [vocab.get(tok, vocab.get("<unk>", 0)) for tok in tokens]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

def load_label_encoder(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# %%
