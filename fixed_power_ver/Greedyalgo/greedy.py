import numpy as np
import random
import time

class GreedyAllocation:
    def __init__(self, K, I, H, B, P, RminK, Thrmin, BandW, N0):
        # Khởi tạo các tham số môi trường
        self.K = K  # Tập người dùng
        self.I = I  # Tập RU
        self.H = H  # Ma trận channel gain: H[i][b][k]
        self.B = B  # Danh sách RB của mỗi RU
        self.P = P  # Công suất mỗi RB của RU
        self.RminK = RminK  # Yêu cầu throughput tối thiểu từng user
        self.Thrmin = Thrmin  # Tổng throughput mục tiêu
        self.BandW = BandW  # Băng thông mỗi RB
        self.N0 = N0  # Noise power density

        # Tạo ma trận phân bổ và công suất: allocation[i][b][k], power[i][b][k]
        self.pi = {k : 0 for k in self.K}
        self.allocation = np.zeros((len(self.I), max(len(self.B[i]) for i in self.I), len(self.K)))
        self.power = np.zeros((len(self.I), max(len(self.B[i]) for i in self.I), len(self.K)))
        self.num_user_serve = 0
        self.throughput_Greedy = 0
        self.throughput = {}
        self.runtime = 0
    
    def check_solution_constraints(self, x, p):
        # Tính toán y và pi từ x
        y = {i: np.zeros(len(self.K)) for i in self.I}

        for i in self.I:
                for k in self.K:
                    if sum(x[i][b][k] for b in self.B[i]) > 0:
                        y[i][k] = 1
                    else :
                        y[i][k] = 0

        # Tính biến pi_k cho từng người dùng ở đây
        for k in self.K:
            if sum(y[i][k] for i in self.I) > 0 :
                self.pi[k] = 1
            else :
                self.pi[k] = 0
            
        # Constraint 1: Mỗi RB chỉ được gán tối đa 1 user
        for i in self.I:
            for b in self.B[i]:
                sum_b = sum([x[i][b][k] for k in self.K])
                if sum_b > 1.0:
                    return 1  # Vi phạm điều kiện 1

        # Constraint 5: Tổng công suất truyền không vượt quá Pmax của RU
        for i in self.I:
            total_power = sum(x[i][b][k] * p[i][b][k] for b in self.B[i] for k in self.K)
            t = total_power - self.Pmax[i]
            if t > 0.1e-6:
                return 5  # Vi phạm điều kiện 5

        return 0  # Không có lỗi nào

    def run(self):
        # RU được xét theo thứ tự giảm dần công suất tối đa
        start = time.time()
        condition = True
        sorted_RU = sorted(self.I, key=lambda i: -self.Pmax[i])
        while condition:
            allocation_temp = self.allocation.copy()
            power_temp = self.power.copy()

            for i in self.I:
                ru_power_left = self.Pmax[i]
                for b in self.B[i]:
                    # Chọn người dùng có channel tốt nhất
                    k = np.random.choice(self.K)
                    allocation_temp[i][b][k] = 1

                    # Cấp công suất theo tỉ lệ kênh truyền
                    share = min(ru_power_left, self.Pmax[i] / len(self.B[i]))
                    power_temp[i][b][k] = share
                    ru_power_left -= share
            
            if self.check_solution_constraints(allocation_temp, power_temp) == 0:
                condition = False
                end = time.time()
                self.allocation = allocation_temp
                self.power = power_temp
                self.re_calculate()
                self.num_user_serve = sum(self.pi[k] for k in self.K)
                self.runtime = end - start

    def compute_throughput(self):
        SINR = np.zeros((len(self.I), max(len(self.B[i]) for i in self.I), len(self.K)))
        for i in self.I:
            for b in self.B[i]:
                for k in self.K:
                    # SINR = công suất x channel gain^2 / (N0 * BW)
                    signal = self.allocation[i][b][k] * self.power[i][b][k] * self.H[i][b][k]
                    SINR[i][b][k] = signal / (self.N0 * self.BW)

        return {k: sum(self.BW * np.log2(1 + SINR[i][b][k]) for i in self.I for b in self.B[i]) for k in self.K}
    
    def re_calculate(self):
        # Cập nhật pi[k] dựa trên throughput và RminK[k]
        throughput = self.compute_throughput()
        self.pi = {k: 1 if throughput[k] >= self.RminK[k] else 0 for k in self.K}

        # Cập nhật lại trạng thái phân bổ RB (x) dựa trên pi[k]
        for k in self.K:
            if self.pi[k] == 0:  
                for i in self.I:
                    for b in self.B[i]:
                        self.allocation[i][b][k] = 0 
                        self.power[i][b][k] = 0

        throughput = self.compute_throughput()
        self.pi = {k: 1 if throughput[k] >= self.RminK[k] else 0 for k in self.K}

        # Cập nhật throughput tổng cộng
        self.throughput_Greedy = sum(
                throughput[k] for k in self.K
            )
