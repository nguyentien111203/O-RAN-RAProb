import csv
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import PPOtrain.input  # Gọi tới createEnvironmentInput()
import numpy as np

# ---------------------------------------------
# TRÍCH ĐẶC TRƯNG - Mỗi RB là 1 sample
# ---------------------------------------------
def extract_features(K, I, B, H, P, RminK, Thrmin, BandW, N0):
    features = []
    labels = []
    rb_map = []

    for i in I:
        for b_idx, b in enumerate(B[i]):
            for k in K:
                vec = [
                    len(K), len(I), len(B[i]),
                    RminK[k], Thrmin, BandW, H[i][b][k], P[i][b], N0
                ]
                features.append(vec)
                labels.append(k)  # placeholder, bạn có thể thay bằng nhãn thật
                rb_map.append((i, b))
    return features, labels, rb_map

# ---------------------------------------------
# XÂY TẬP DỮ LIỆU TỪ CSV
# ---------------------------------------------
def build_dataset_from_csv(csv_path):
    all_features = []
    all_labels = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            numuser, numRU, RBeachRU, Pmax, RminK, Thrmin, BandW, N0 = row

            numuser = int(numuser)
            numRU = int(numRU)
            RBeachRU = ast.literal_eval(RBeachRU)
            Pmax = ast.literal_eval(Pmax)
            RminK = ast.literal_eval(RminK)
            Thrmin = float(Thrmin)
            BandW = float(BandW)
            N0 = float(N0)

            K, I, B, H, P = PPOtrain.input.createEnvironmentInput(numuser, numRU, RBeachRU, Pmax)

            feats, labels, _ = extract_features(K, I, B, H, P, RminK, Thrmin, BandW, N0)
            all_features.extend(feats)
            all_labels.extend(labels)

    X = torch.tensor(all_features, dtype=torch.float32)
    y = torch.tensor(all_labels, dtype=torch.long)
    return X, y, numuser

# ---------------------------------------------
# MÔ HÌNH PHÂN LOẠI USER CHO MỖI RB
# ---------------------------------------------
class RBClassifier(nn.Module):
    def __init__(self, input_dim, num_users, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_users)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------
# HUẤN LUYỆN MÔ HÌNH
# ---------------------------------------------
def train_model(X, y, num_users, epochs=20, batch_size=64):
    model = RBClassifier(X.shape[1], num_users)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, loss: {loss.item():.4f}")
    return model

# ---------------------------------------------
# LƯU / LOAD MÔ HÌNH
# ---------------------------------------------
def save_model(model, path="rb_classifier.pt"):
    torch.save(model.state_dict(), path)

def load_model(input_dim, num_users, path="rb_classifier.pt"):
    model = RBClassifier(input_dim, num_users)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# ---------------------------------------------
# DỰ ĐOÁN PHÂN BỔ allocation[i][b] = k
# ---------------------------------------------
def predict_allocation(model, K, I, B, H, P, RminK, Thrmin, BandW, N0):
    features, _, rb_map = extract_features(K, I, B, H, P, RminK, Thrmin, BandW, N0)
    X = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X)
        predictions = torch.argmax(logits, dim=1).numpy()

    allocation = {i: {} for i in I}
    idx = 0
    for (i, b) in rb_map:
        allocation[i][b] = predictions[idx]
        idx += 1

    return allocation

# ---------------------------------------------
# TÍNH THROUGHPUT DỰA TRÊN ALLOCATION[i][b] = k
# ---------------------------------------------
def compute_throughput(K, I, B, H, P, RminK, BandW, N0, allocation):
    throughput = {k: 0 for k in K}
    for i in I:
        for b in B[i]:
            k = allocation[i].get(b, None)
            if k is None:
                continue
            sinr = H[i][b][k] * P[i][b] / (N0 * BandW[k])
            throughput[k] += BandW[k] * np.log2(1 + sinr)

    num_served = sum(1 for k in K if throughput[k] >= RminK[k])
    return throughput, num_served
