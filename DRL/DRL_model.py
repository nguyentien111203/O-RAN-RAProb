import torch
import numpy as np
from DRL_train import PPOTrainer  # Import mô hình PPO đã train

# Lớp mô hình thực hiện map
class Predictor:
    def __init__(self, model_path, I, B, K):
        self.trainer = PPOTrainer(I, B, K)
        self.trainer.load_model(model_path)  # Load mô hình đã train

    def predict_and_save(self, state, output_file="prediction.npz"):
        action = self.trainer.select_action(state)  # Dự đoán hành động
        x_pred = state[:self.trainer.I]  # Lấy x từ input
        p_pred = state[self.trainer.I:]  # Lấy p từ input
        
        # Lưu kết quả vào file .npz
        np.savez_compressed(output_file, x=x_pred, p=p_pred, action=action)
        print(x_pred)
        print(p_pred)
        print(f"Kết quả đã lưu vào {output_file}")

# Thử dự đoán
if __name__ == "__main__":
    model_path = "ppo_policy.pth"  # Đường dẫn mô hình đã train
    I, B, K = 3, 3, 3  # Số lượng Input

    predictor = Predictor(model_path, I, B, K)

    # Giả lập một trạng thái đầu vào
    state = np.random.rand(2 * I, B, K)

    # Dự đoán và lưu kết quả
    predictor.predict_and_save(state, output_file="result.npz")