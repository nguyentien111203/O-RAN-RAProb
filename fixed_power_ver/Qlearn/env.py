import numpy as np
from itertools import product
import common

class Environment:
    def __init__(self, numuser, numRU, B, PeachRB, HeachRU, RminK, BandW, N0, delta = 1):
        self.numuser = numuser  # Số người dùng
        self.numRU = numRU      # Số RU
        self.B = B              # Số RB mỗi RU
        self.PeachRB = PeachRB  # Công suất mỗi RB
        self.HeachRU = HeachRU  # Ma trận kênh mỗi RU: shape (numRU, numuser)
        self.RminK = RminK      # Yêu cầu throughput tối thiểu
        self.BandW = BandW      # Băng thông mỗi PRB
        self.N0 = N0            # Mật độ công suất tạp âm

        # Thông tin cho giải thuật
        self.delta = delta      # Mức rời rạc cho Allocation
        self.Allocation_matrix = np.zeros((numRU, numuser))  # Ma trận tỷ lệ phân bổ
        self.R_k = np.zeros(numuser)  # Throughput hiện tại
        self.R_gap = self.RminK.copy()  # Khoảng cách throughput
        self.discrete_levels = np.arange(0, 1.01, self.delta)  # Mức rời rạc cho Allocation
        self.rgap_levels = [0, 1]  # Mức rời rạc cho R_gap

    def reset(self):
        """Khởi tạo lại môi trường"""
        scores = self.RminK * self.HeachRU**0.25
        self.Allocation_matrix = np.ones((self.numRU, self.numuser))
        for i in range(self.numRU):
            # 1. Phân bố tỉ lệ thực
            raw_alloc = [self.B[i] * scores[i][k] / sum(scores[i]) for k in range(self.numuser)]

            # 2. Làm tròn xuống để lấy phần nguyên
            int_alloc = [int(x) for x in raw_alloc]

            # 3. Tính số PRB còn lại sau khi làm tròn
            remaining = self.B[i] - sum(int_alloc)

            # 4. Tính phần dư (xếp hạng theo độ lớn phần lẻ nhất)
            remaining = self.B[i] - sum(int_alloc)
            fractional_parts = [(k, raw_alloc[k] - int_alloc[k]) for k in range(self.numuser)]
            fractional_parts.sort(key=lambda x: x[1], reverse=True)
            for j in range(remaining):
                k = fractional_parts[j][0]
                int_alloc[k] += 1

            # 6. Lưu vào Allocation_matrix
            for k in range(self.numuser):
                self.Allocation_matrix[i][k] = int_alloc[k]
        self.R_k = np.zeros(self.numuser)
        self.R_gap = np.ones(self.numuser)
        return self.get_state(0)  # Trạng thái của RU đầu tiên

    def discretize_rgap(self, rgap):
        """Rời rạc hóa R_gap"""
        discretized = np.zeros_like(rgap, dtype=int)
        for k in range(self.numuser):
            if rgap[k] < 0:
                discretized[k] = 0  # Đã đạt
            else:
                discretized[k] = 1  # Thiếu 
        return discretized

    def get_state(self, ru_idx):
        """Lấy trạng thái của RU ru_idx"""
        allocation = self.Allocation_matrix[ru_idx]
        rgap = self.R_gap
        return {
            'allocation': allocation,
            'rgap': self.discretize_rgap(rgap)
        }

    def compute_throughput(self):
        """Tính throughput cho mỗi người dùng dựa trên HeachRU"""
        self.R_k = np.zeros(self.numuser)
        for k in range(self.numuser):
            for i in range(self.numRU):
                    sinr = (self.PeachRB[i] * np.abs(self.HeachRU[i, k])**2) / (self.N0 * self.BandW)
                    self.R_k[k] += self.BandW * self.Allocation_matrix[i, k] * np.log2(1 + sinr)
        self.R_gap = (self.RminK - self.R_k)/self.RminK

    def compute_reward(self, ru_idx):
        """Tính phần thưởng cho RU ru_idx"""

        reward = common.tunning * np.sum(self.R_k >= self.RminK) + (1-common.tunning) * np.sum(self.R_k)

        return reward

    def step(self, ru_idx, action):
        """Thực hiện hành động và trả về trạng thái mới, phần thưởng"""
        user_from, user_to = action
        if self.Allocation_matrix[ru_idx, user_from] >= self.delta:
            self.Allocation_matrix[ru_idx, user_from] -= self.delta
            self.Allocation_matrix[ru_idx, user_to] += self.delta
            self.compute_throughput()
            reward = self.compute_reward(ru_idx)
            next_state = self.get_state(ru_idx)
            return next_state, reward, False
        else:
            return self.get_state(ru_idx), 0, False  # Hành động không khả thi