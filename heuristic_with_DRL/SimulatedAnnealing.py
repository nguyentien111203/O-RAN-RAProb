import numpy as np
from simanneal import Annealer
import matplotlib.pyplot as plt

class RBAllocationSA(Annealer):
    def __init__(self, K, I, H, B, P, Pmax, RminK, Tmin, BW, N0):
        self.K = K  # Tập người dùng
        self.I = I  # Tập RU
        self.H = H  # Ma trận channel vector
        self.B = B  # Tập số RB của từng RU
        self.P = P
        self.Pmax = Pmax  # Công suất tối đa của từng RU
        self.RminK = RminK  # Data rate tối thiểu của từng user
        self.Tmin = Tmin  # Throughput tối thiểu
        self.BW = BW  # Băng thông của từng RB
        self.N0 = N0  # Mật độ công suất tạp âm
        self.energy_history = []

        state = self.initialize_state()
        super().__init__(state)

    def initialize_state(self):
        allocation = {i: np.zeros((len(self.B[i]), len(self.K))) for i in self.I}
        power = {i: np.zeros((len(self.B[i]), len(self.K))) for i in self.I}
        
        for k in self.K:
            i = np.random.choice(self.I)
            b = np.random.randint(0, len(self.B[i]))
            allocation[i][b][k] = 1
            power[i][b][k] = np.random.uniform(0, self.Pmax[i])
        
        return allocation, power

    def compute_SINR(self, allocation, power):
        SINR = {k: 0 for k in self.K}
        for k in self.K:
            signal = sum(
                power[i][b][k] * self.H[k][i][b]
                for i in self.I
                for b in range(len(self.B[i]))
                if allocation[i][b][k] == 1
            )
            SINR[k] = signal / (self.N0 * self.BW)
        return SINR

    def compute_throughput(self, SINR):
        return {k: self.BW * np.log2(1 + SINR[k]) for k in self.K}

    def energy(self):
        allocation, power = self.state
        SINR = self.compute_SINR(allocation, power)
        throughput = self.compute_throughput(SINR)
        cost = -sum(np.log(1 + SINR[k]) for k in self.K)
        penalty = sum(max(0, self.RminK[k] - throughput[k]) for k in self.K)
        return cost + penalty

    def move(self):
        allocation, power = self.state
        Cmax = 3  # Số lần cấp tài nguyên tối đa cho một trạm liên tiếp
        Cmax_k = 5  # Số lần cấp tài nguyên tối đa cho một thiết bị
        
        # Chọn k chưa đạt R_min, nếu không có thì chọn ngẫu nhiên
        underperforming_users = [k for k in self.K if self.compute_throughput(self.compute_SINR(allocation, power))[k] < self.RminK[k]]
        if underperforming_users:
            k = np.random.choice(underperforming_users)
        else:
            k = np.random.choice(self.K)
        
        # Chọn một trạm i chưa phục vụ k hoặc luân phiên nếu đã phục vụ hết
        candidate_stations = [i for i in self.I if not np.any(allocation[i][:, k])]
        if candidate_stations:
            i = np.random.choice(candidate_stations)
        else:
            i = np.random.choice(self.I)
        
        # Chọn một RB mới cho k tại trạm i
        b = np.random.randint(0, len(self.B[i]))
        allocation[i][:, k] = 0  # Xóa RB cũ của k
        allocation[i][b][k] = 1  # Gán RB mới
        
        # Điều chỉnh công suất
        throughput = self.compute_throughput(self.compute_SINR(allocation, power))
        if throughput[k] >= self.RminK[k]:  # Nếu đã đạt yêu cầu, giảm công suất nhẹ
            alpha = 1 / (self.T + 10)  # Giảm nhẹ theo nhiệt độ của SA
            power[i][b][k] = max(0, power[i][b][k] * (1 - alpha))
        else:  # Nếu chưa đạt, cấp thêm công suất với giới hạn
            if self.tranmission_count[i] < Cmax and self.user_attempts[k] < Cmax_k:
                delta_power = min(0.1 * self.Pmax[i], power[i][b][k] * 1.1)
                power[i][b][k] = min(self.Pmax[i], power[i][b][k] + delta_power)
                self.tranmission_count[i] += 1
                self.user_attempts[k] += 1
        
        self.state = allocation, power
        
        # Lưu giá trị năng lượng sau mỗi vòng lặp
        self.energy_history.append(self.energy())

    def run(self):
        self.copy_strategy = "deepcopy"
        best_state, best_energy = self.anneal()
        return best_state, best_energy
    

    def draw_figures(self): # Vẽ hình xem tiến trình hội tụ
        plt.plot(self.energy_history)
        plt.xlabel("Iteration")
        plt.ylabel("Energy")
        plt.title("Simulated Annealing Energy Progression")
        plt.show() 

    #def transfer_var(self): # Xác định giá trị các biến sau khi thực hiện

