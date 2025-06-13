import numpy as np
from itertools import product

class Environment:
    def __init__(self, numuser, numRU, B, PeachRB, HeachRU, RminK, Thrmin, BandW, N0, delta = 1):
        self.numuser = numuser  # Số người dùng
        self.numRU = numRU      # Số RU
        self.B = B              # Số RB mỗi RU
        self.PeachRB = PeachRB  # Công suất mỗi RB
        self.HeachRU = HeachRU  # Ma trận kênh mỗi RU: shape (numRU, numuser)
        self.RminK = RminK      # Yêu cầu throughput tối thiểu
        self.Thrmin = Thrmin    # Giá trị scale theo throughput (hiện đang để là Mbps)
        self.BandW = BandW      # Băng thông mỗi PRB
        self.N0 = N0            # Mật độ công suất tạp âm

        # Thông tin cho giải thuật
        self.delta = delta      # Mức rời rạc cho Allocation
        self.Allocation_matrix = np.zeros((numRU, numuser))  # Ma trận tỷ lệ phân bổ
        self.R_k = np.zeros(numuser)  # Throughput hiện tại
        self.served = np.zeros(numuser) # Phục vụ người dùng
        self.initial_thr = np.zeros(numuser)
        self.initial_user_served = np.zeros(numuser)
        self.budget = B
        self.prev_served = np.zeros(self.numuser)

    def reset(self):
        self.Allocation_matrix = np.zeros((self.numRU, self.numuser))
        
        score = self.RminK / np.sum(self.RminK)  # Tính điểm phân bổ

        for ru in range(self.numRU):
            # Phân bổ theo tỉ lệ điểm
            for user in range(self.numuser):
                self.Allocation_matrix[ru][user] = int(self.B[ru] * score[user])
            remaining = self.B[ru] - int(np.sum(self.Allocation_matrix[ru]))
            
            if remaining != 0:
                for rb in range(remaining):
                    k = np.random.choice(range(self.numuser))
                    self.Allocation_matrix[ru][k] += 1 

        # Tính throughput và lưu trạng thái phục vụ ban đầu
        self.compute_throughput()
        self.prev_served = (self.R_k >= self.RminK).astype(int)
        self.R_gap = np.ones(self.numuser)
        
        return self.get_state(0)

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
        return {
            'allocation': allocation
        }

    def compute_throughput(self):
        """Tính throughput cho mỗi người dùng dựa trên HeachRU"""
        self.R_k = np.zeros(self.numuser)
        for k in range(self.numuser):
            for i in range(self.numRU):
                    sinr = (self.PeachRB[i] * np.abs(self.HeachRU[i, k])**2) / (self.N0 * self.BandW)
                    self.R_k[k] += self.BandW * self.Allocation_matrix[i, k] * np.log2(1 + sinr)
        for k in range(self.numuser):
            if self.R_k[k] >= self.RminK[k]:
                self.served[k] = 1
            else :
                self.served[k] = 0

    def step(self, ru_idx, action):
        """Thực hiện hành động: chuyển delta RB từ user_from sang user_to nếu hợp lệ."""
        if action == (-1,-1):
            return self.get_state(ru_idx), False, True
        else:
            user_from, user_to = action

            valid = False

            # Kiểm tra xem có thể chuyển delta từ user_from sang user_to không
            if self.Allocation_matrix[ru_idx, user_from] >= self.delta:
                self.Allocation_matrix[ru_idx, user_from] -= self.delta
                self.Allocation_matrix[ru_idx, user_to] += self.delta
                valid = True

            next_state = self.get_state(ru_idx)

            return next_state, False, valid

        