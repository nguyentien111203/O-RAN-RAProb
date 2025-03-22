import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Định nghĩa mạng chính sách
class PolicyNet(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = torch.softmax(self.fc3(x), dim=-1)
        return action_probs

# Định nghĩa PPO trainer
class PPOTrainer:
    def __init__(self, data_dir, action_dim=10, hidden_dim=128, lr=0.001, batch_size=64, epochs=1000):
        self.data_dir = data_dir
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load dữ liệu từ nhiều file
        self.states, self.rewards = self.load_all_data()

        # Khởi tạo mô hình
        self.input_dim = self.states.shape[1]
        self.model = PolicyNet(self.input_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Tạo DataLoader
        self.dataset = TensorDataset(self.states, self.rewards)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def load_all_data(self):
        all_states, all_rewards = [], []

        # Duyệt qua tất cả các file .npz trong thư mục
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".npz"):
                file_path = os.path.join(self.data_dir, file_name)
                data = np.load(file_path)

                x = data["x"]  # Allocation (Batch, I, B, K)
                p = data["p"]  # Power (Batch, I, B, K)
                energy = data["energy"]  # Reward (Batch)

                # Ghép x và p thành state đầu vào
                state = np.concatenate([x, p], axis=1)  # (Batch, 2*I, B, K)

                all_states.append(state)
                all_rewards.append(energy)

        # Nối tất cả dữ liệu thành một mảng lớn
        states = np.concatenate(all_states, axis=0)  # (Total_Batch, 2*I, B, K)
        rewards = np.concatenate(all_rewards, axis=0)  # (Total_Batch,)

        # Chuyển đổi thành Tensor
        states = torch.tensor(states.reshape(states.shape[0], -1), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)

        return states, rewards

    def train(self):
        self.model.train()
        for epoch in tqdm(range(self.epochs), desc="Training PPO"):
            total_loss = 0
            for batch_states, batch_rewards in self.dataloader:
                self.optimizer.zero_grad()

                # Tính toán xác suất hành động
                action_probs = self.model(batch_states)

                # Lấy chỉ số hành động theo phân phối chính sách
                actions = torch.multinomial(action_probs, num_samples=1)

                # Tính log-prob của hành động được chọn
                selected_log_probs = torch.log(action_probs.gather(1, actions))

                # PPO Policy Gradient Loss
                loss = -(selected_log_probs * batch_rewards).mean()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        # Lưu mô hình sau khi train
        self.save_model()
        print("Mô hình PPO đã được lưu thành công!")

    def save_model(self, path="./DRL/ppo_policy.pth"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="./DRL/ppo_policy.pth"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            action_probs = self.model(state)
            action = torch.multinomial(action_probs, num_samples=1).item()
        
        return action

# Chạy Training 
if __name__ == "__main__":
    trainer = PPOTrainer(data_dir="./DRL/Data_DRL", action_dim=10, epochs=1000, lr=0.001, batch_size=64)
    trainer.train()
