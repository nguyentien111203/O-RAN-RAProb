import numpy as np
from itertools import product

class Environment:
    def __init__(self, numuser, numRU, B, PeachRB, HeachRU, RminK, BandW, N0):
        self.numuser = numuser  # Số người dùng
        self.numRU = numRU      # Số RU
        self.B = B              # Số RB mỗi RU
        self.PeachRB = PeachRB  # Công suất mỗi RB
        self.HeachRU = HeachRU  # Ma trận kênh mỗi RU: shape (numRU, numuser)
        self.RminK = RminK      # Yêu cầu throughput tối thiểu
        self.BandW = BandW      # Băng thông mỗi PRB
        self.N0 = N0            # Mật độ công suất tạp âm
        self.Allocation_matrix = np.zeros((numRU, numuser))  # Ma trận tỷ lệ phân bổ
        self.R_k = np.zeros(numuser)  # Throughput hiện tại
        self.R_gap = self.RminK.copy()  # Khoảng cách throughput
        self.discrete_levels = np.arange(0, 1.01, 0.05)  # Mức rời rạc cho Allocation
        self.rgap_levels = [0, 1, 2, 3]  # Mức rời rạc cho R_gap

    def reset(self):
        """Khởi tạo lại môi trường"""
        self.Allocation_matrix = np.ones((self.numRU, self.numuser))
        for i in range(self.numRU):
            for k in range(self.numuser):
                self.Allocation_matrix[i][k] *= self.RminK[k] / sum(self.RminK)
        self.R_k = np.zeros(self.numuser)
        self.R_gap = np.ones(self.numuser)
        return self.get_state(0)  # Trạng thái của RU đầu tiên

    def discretize_allocation(self, allocation):
        """Rời rạc hóa tỷ lệ phân bổ"""
        return np.round(allocation / 0.05) * 0.05

    def discretize_rgap(self, rgap):
        """Rời rạc hóa R_gap"""
        discretized = np.zeros_like(rgap, dtype=int)
        for k in range(self.numuser):
            if rgap[k] < 0:
                discretized[k] = 0  # Đã đạt
            elif 0 <= rgap[k] <= self.RminK[k] / 3:
                discretized[k] = 1  # Gần đạt
            elif self.RminK[k] / 3 < rgap[k] <= 2 * self.RminK[k] / 3:
                discretized[k] = 2  # Thiếu vừa
            else:
                discretized[k] = 3  # Thiếu nhiều
        return discretized

    def get_state(self, ru_idx):
        """Lấy trạng thái của RU ru_idx"""
        allocation = self.Allocation_matrix[ru_idx]
        rgap = self.R_gap
        return {
            'allocation': self.discretize_allocation(allocation),
            'rgap': self.discretize_rgap(rgap)
        }

    def compute_throughput(self):
        """Tính throughput cho mỗi người dùng dựa trên HeachRU"""
        self.R_k = np.zeros(self.numuser)
        for k in range(self.numuser):
            for i in range(self.numRU):
                    sinr = (self.PeachRB[i] * np.abs(self.HeachRU[i, k])**2) / (self.N0 * self.BandW)
                    self.R_k[k] += self.BandW * self.Allocation_matrix[i, k] * self.B[i] * np.log2(1 + sinr)
        self.R_gap = (self.RminK - self.R_k)/self.RminK

    def compute_reward(self, ru_idx):
        """Tính phần thưởng cho RU ru_idx"""
        num_satisfied = np.sum(self.R_k >= self.RminK)
        unsatisfied_penalty = np.sum(np.maximum(0, (self.RminK - self.R_k)/self.RminK))
        w1, w2 = 100, 1
        reward = w1 * (num_satisfied / self.numuser) - w2 * unsatisfied_penalty 
        return reward

    def step(self, ru_idx, action):
        """Thực hiện hành động và trả về trạng thái mới, phần thưởng"""
        user_from, user_to = action
        delta = 0.05
        if self.Allocation_matrix[ru_idx, user_from] >= delta:
            self.Allocation_matrix[ru_idx, user_from] -= delta
            self.Allocation_matrix[ru_idx, user_to] += delta
            self.compute_throughput()
            reward = self.compute_reward(ru_idx)
            next_state = self.get_state(ru_idx)
            return next_state, reward, False
        else:
            return self.get_state(ru_idx), 0, False  # Hành động không khả thi