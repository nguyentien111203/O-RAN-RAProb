import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# -------------------------
# Trích đặc trưng RB-user
# -------------------------
def extract_pointwise_features(K, I, B, H, P, RminK, Thrmin, BandW, N0):
    features = []
    labels = []
    rb_map = []

    for i in I:
        for b in B[i]:
            for k in K:
                vec = [
                    len(K), len(I), len(B[i]),
                    k / len(K),  # user identity (normalized)
                    RminK[k], Thrmin, BandW,
                    H[i][b][k], P[i][b], N0
                ]
                features.append(vec)
                labels.append(k)  # chỉ dùng nếu có ground truth
                rb_map.append((i, b, k))
    return torch.tensor(features, dtype=torch.float32), rb_map

# -------------------------
# Mô hình dự đoán Q-score
# -------------------------
class RBScoringNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Q-score
        )

    def forward(self, x):
        return self.model(x)

# -------------------------
# Huấn luyện Q-score model
# -------------------------
def train_model_qscore(X, y_scores=None, epochs=20, batch_size=128):
    model = RBScoringNN(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if y_scores is None:
        y_scores = torch.rand(len(X), 1)  # nếu không có nhãn thực

    dataset = TensorDataset(X, y_scores)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, loss: {loss.item():.4f}")
    return model

# -------------------------
# Dự đoán allocation từ Q-score
# -------------------------
def predict_allocation_q_scoring(model, K, I, B, H, P, RminK, Thrmin, BandW, N0):
    X, rb_map = extract_pointwise_features(K, I, B, H, P, RminK, Thrmin, BandW, N0)
    with torch.no_grad():
        scores = model(X).squeeze().numpy()

    allocation = {i: {} for i in I}
    idx_map = {}

    for idx, (i, b, k) in enumerate(rb_map):
        if (i, b) not in idx_map:
            idx_map[(i, b)] = []
        idx_map[(i, b)].append((k, scores[idx]))

    for (i, b), candidates in idx_map.items():
        best_k = max(candidates, key=lambda x: x[1])[0]
        allocation[i][b] = best_k

    return allocation

def compute_throughput_and_served_users(K, I, B, H, P, RminK, BandW, N0, allocation):
    """
    Tính throughput cho từng user, tổng throughput và số người dùng được phục vụ
    """
    throughput = {k: 0.0 for k in K}
    
    for i in I:
        for b in B[i]:
            k = allocation[i].get(b, None)
            if k is None:
                continue
            sinr = H[i][b][k] * P[i][b] / (N0 * BandW)
            rate = BandW * np.log2(1 + sinr)
            throughput[k] += rate

    num_served = sum(1 for k in K if throughput[k] >= RminK[k])
    return throughput, num_served
