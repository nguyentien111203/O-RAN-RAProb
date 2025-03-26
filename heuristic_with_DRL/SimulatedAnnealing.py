import numpy as np
from simanneal import Annealer
import matplotlib.pyplot as plt
import time
import os
import common
import math


class RBAllocationSA(Annealer):
    id = 0
    def __init__(self, K, I, H, B, Pmax, RminK, Tmin, BW, N0, step_SA, test_id):
        self.K = K  # Tập người dùng
        self.I = I  # Tập RU
        self.H = H  # Ma trận channel vector
        self.B = B  # Tập số RB của từng RU
        self.Pmax = Pmax  # Công suất tối đa của từng RU
        self.RminK = RminK  # Data rate tối thiểu của từng user
        self.Tmin = Tmin  # Throughput tối thiểu
        self.BW = BW  # Băng thông của từng RB
        self.N0 = N0  # Mật độ công suất tạp âm
        
        self.input_prob = (self.K, self.I, self.H, self.B, self.Pmax, self.RminK, self.Tmin, self.BW, self.N0)
        # Khởi tạo biến người dùng
        self.pi = {k : 0 for k in self.K}
        self.throughput_SA = 0

        self.num_user_serve = 0
        self.Tmax = 1000
        self.Tmin = 100
        self.steps = step_SA
        self.energy_history = []
        self.time = 0

        self.test_id = test_id
        self.best_state = []
        self.best_energy = 0
        self.all_x = []
        self.all_p = []

        state = self.initialize_state()
        super().__init__(state)

    def initialize_state(self):
        x = np.array([np.zeros((len(self.B[i]), len(self.K))) for i in self.I], dtype = object)
        p = np.array([np.zeros((len(self.B[i]), len(self.K))) for i in self.I], dtype = object)

        condition = True
        while condition:
            """
            Tạo giải pháp ban đầu hợp lệ:
            - Đảm bảo mỗi RB chỉ gán 1 user (Constraint 1)
            - Đảm bảo công suất phân bổ hợp lý không vượt quá Pmax (Constraint 5)
            """
            
            for i in self.I:
                total_power = self.Pmax[i]  # Tổng công suất đã cấp phát trong RU i

                for b in self.B[i]:
                    # Chọn ngẫu nhiên 1 user để gán RB (đảm bảo RB chỉ gán cho 1 user)
                    k = np.random.choice(self.K)  
                    x[i][b][k] = 1  # Gán RB b cho user k

                # Phân bổ công suất ngẫu nhiên
                for b in self.B[i]:
                    for k in self.K:
                        if x[i][b][k] == 1:
                            p[i][b][k] = np.random.uniform(0.1, 0.5) * self.Pmax[i]  # Gán công suất ngẫu nhiên
                            if total_power <= p[i][b][k] :
                                p[i][b][k] = total_power
                                total_power = 0
                            else :
                                total_power-=p[i][b][k]
            if self.check_solution_constraints(x, p) == 0 :
                print("Initial state completed!")
                condition = False
                
        return x, p

    def compute_SINR(self, state):
        allocation, power = state
        SINR = {i : np.zeros((len(self.B[i]), len(self.K))) for i in self.I}
        for i in self.I:
            for b in self.B[i]:
                for k in self.K:
                    signal = allocation[i][b][k] * power[i][b][k] * self.H[k][i][b]
                    SINR[i][b][k] = signal / (self.N0 * self.BW)
        return SINR

    def compute_throughput(self, SINR):
        return {k: sum(self.BW * np.log2(1 + SINR[i][b][k]) for i in self.I for b in self.B[i]) for k in self.K}
    
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
                if sum(x[i][b][k] for k in self.K) > 1:
                    return 1  # Vi phạm điều kiện 1

        # Constraint 5: Tổng công suất truyền không vượt quá Pmax của RU
        for i in self.I:
            total_power = sum(x[i][b][k] * p[i][b][k] for b in self.B[i] for k in self.K)
            if total_power - self.Pmax[i] > 1e-5:
                return 5  # Vi phạm điều kiện 5

        return 0  # Không có lỗi nào


    def energy(self):
        SINR = self.compute_SINR(self.state)
        throughput = self.compute_throughput(SINR)
        
        # Hàm tối ưu của bài toán
        cost = -sum((throughput[k]/self.Tmin) + self.pi[k] for k in self.K)

        # Đại lượng phạt nếu không đảm bảo throughput cho người dùng
        penalty = common.lamda_penalty * sum(np.log1p(max(0, (self.RminK[k] - throughput[k]) / self.Tmin)) for k in self.K)

        # Tối đa hóa hàm mục tiêu và tối thiểu hóa đại lượng phạt 
        return cost + penalty
    
    def move(self):
        allocation, power = self.state  # Lấy trạng thái hiện tại
        condition = True

        # Xác định số RB tối đa ban đầu cho mỗi user để tránh phân tán công suất
        max_RB_per_user = max(1, len(self.K) // len(self.I))

        while condition:
            i = np.random.choice(self.I)  # Chọn ngẫu nhiên một RU
            k = np.random.choice(self.K)  # Chọn ngẫu nhiên một user

            # Danh sách RB đã cấp phát và chưa cấp phát cho user k
            b_used = np.where(allocation[i][:, k] == 1)[0] if np.any(allocation[i][:, k]) else []
            b_unused = [b for b in range(len(self.B[i])) if b not in b_used]  # Các RB chưa dùng

            action = np.random.choice(["add", "swap", "relocate"], p=[1/3, 1/3, 1/3])

            if action == "add" and len(b_unused) > 0 and len(b_used) < max_RB_per_user:
                # Chọn RB có channel gain tốt nhất chưa dùng
                b_new = max(b_unused, key=lambda b: self.H[k][i][b])
                b_existing = np.random.choice(b_used) if len(b_used) > 0 else None

                if sum(allocation[i][b_new]) == 0:
                    allocation[i][b_new][k] = 1
                    if b_existing is not None:
                        power_share = power[i][b_existing][k] * 0.20
                        power[i][b_existing][k] *= 0.80
                        power[i][b_new][k] = power_share
                    else:
                        power[i][b_new][k] = 0.1 * self.Pmax[i]

            elif action == "swap" and len(b_used) > 0:
                k_other = np.random.choice(self.K)
                if k_other != k:
                    b_swap = np.random.choice(b_used)
                    if allocation[i][b_swap][k_other] == 1:
                        allocation[i][b_swap][k], allocation[i][b_swap][k_other] = (
                            allocation[i][b_swap][k_other],
                            allocation[i][b_swap][k],
                        )

            elif action == "relocate" and len(b_used) > 0 and len(b_unused) > 0:
                b_old = np.random.choice(b_used)
                b_new_candidates = [b for b in b_unused if self.H[k][i][b] > self.H[k][i][b_old]]
                
                if b_new_candidates:
                    b_new = max(b_new_candidates, key=lambda b: self.H[k][i][b])
                    allocation[i][b_old][k] = 0
                    allocation[i][b_new][k] = 1
                    power[i][b_new][k] = power[i][b_old][k]
                    power[i][b_old][k] = 0

            # Kiểm tra lại throughput để đảm bảo không user nào bị thiếu
            throughput = self.compute_throughput(self.compute_SINR((allocation, power)))
            for k in self.K:
                if throughput[k] < self.RminK[k]:
                    # Cấp thêm công suất nếu chưa đạt yêu cầu
                    for i in self.I:
                        for b in self.B[i]:
                            if allocation[i][b][k] == 1:
                                power[i][b][k] = min(self.Pmax[i], power[i][b][k] * 1.1)

            # Loại bỏ RB cấp phát nhưng không có công suất
            for b in self.B[i]:
                if allocation[i][b][k] == 1 and power[i][b][k] == 0:
                    allocation[i][b][k] = 0

            # Nếu trạng thái hợp lệ, cập nhật trạng thái mới
            self.state = allocation, power  
            self.energy_history.append(self.energy())  # Lưu lại năng lượng
            condition = False


    def re_calculate(self):
        # Cập nhật pi[k] dựa trên throughput và RminK[k]
        throughput = self.compute_throughput(self.compute_SINR(self.best_state))
        self.pi = {k: 1 if throughput[k] >= self.RminK[k] else 0 for k in self.K}

        # Cập nhật lại trạng thái phân bổ RB (x) dựa trên pi[k]
        allocation, power = self.best_state  # Lấy trạng thái tốt nhất
        for k in self.K:
            if self.pi[k] == 0:  
                for i in self.I:
                    for b in self.B[i]:
                        allocation[i][b][k] = 0 

        # Tính lại SINR và throughput sau khi cập nhật x
        sinr = self.compute_SINR((allocation, power))
        throughput = self.compute_throughput(sinr)

        # Cập nhật throughput tổng cộng
        self.throughput_SA = sum(
                sum(self.BW * np.log2(1 + sinr[i][b][k]) for i in self.I for b in self.B[i])
                for k in self.K
            )
        
        with open("./output_SA.txt", 'w') as opf:
            opf.writelines([str(allocation), str(power)])

                
    def run(self):
        self.copy_strategy = "deepcopy"
        start = time.time()
        self.best_state, self.best_energy = self.anneal()
        end = time.time()
        self.time = end - start
        self.save_file()
        self.re_calculate()
        self.num_user_serve = sum(self.pi[k] for k in self.K)
        

    def draw_figures(self): # Vẽ hình xem tiến trình hội tụ
        plt.plot(self.energy_history)
        plt.xlabel("Iteration")
        plt.ylabel("Energy")
        plt.title("Simulated Annealing Energy Progression")
        
        # Lưu ảnh vào file
        plt.savefig(f"./Output/figures/energy_plot_{RBAllocationSA.id}_{self.test_id}.png")  
        plt.close()  # Đóng hình để tránh lỗi khi vẽ nhiều hình liên tiếp

    def save_file(self):
        # Tạo file để lưu
        save_file = f"./DRL/Data_DRL/data_{RBAllocationSA.id}_{self.test_id}.npz"
        np.savez_compressed(save_file, x = np.array(self.all_x, dtype = object)
                                     , p = np.array(self.all_p, dtype = object)
                                     , energy = np.array(self.energy_history, dtype = np.float32),
                                     allow_pickle = True)

    

    

