import numpy as np
from simanneal import Annealer
import matplotlib.pyplot as plt
import time
import os
import common


class RBAllocationSA(Annealer):
    def __init__(self, K, I, H, B, Pmax, RminK, Tmin, BW, N0, step_SA):
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

        self.num_user_serve = 0
        self.Tmax = 2000
        self.Tmin = 100
        self.steps = step_SA
        self.energy_history = []

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
                condition = False
                print("Initial state created successfully!")
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
        """
        Hàm tạo lân cận mới với hai lựa chọn:
        - Điều chỉnh công suất một RB đang cấp phát
        - Thêm một RB mới chưa được cấp phát và chia sẻ công suất
        Đảm bảo:
        - Điều kiện 1: Mỗi RB chỉ gán tối đa 1 user
        - Điều kiện 5: Tổng công suất không vượt quá Pmax của RU
        """
        allocation, power = self.state  # Lấy trạng thái hiện tại
        condition = True

        while condition:
            i = np.random.choice(self.I)  # Chọn ngẫu nhiên một RU
            k = np.random.choice(self.K)  # Chọn ngẫu nhiên một user

            # Danh sách RB đã cấp phát và chưa cấp phát cho user k
            b_used = np.where(allocation[i][:, k] == 1)[0] if np.any(allocation[i][:, k]) else []
            b_unused = [b for b in range(len(self.B[i])) if b not in b_used]  # Các RB chưa dùng

            action = np.random.choice(["move", "add"], p=[0.5, 0.5])  # 50% di dời, 50% thêm RB

            if action == "move" and len(b_used) > 0:  # Điều chỉnh công suất RB đã cấp phát
                b_selected = np.random.choice(b_used)  # Chọn 1 RB đang dùng
                delta = np.random.uniform(0.98, 1.02)  # ±2% thay đổi
                power[i][b_selected][k] *= delta
                power[i][b_selected][k] = min(self.Pmax[i], max(0, power[i][b_selected][k]))  # Giữ trong giới hạn

            elif action == "add" and len(b_unused) > 0 and len(b_used) > 0:  # Thêm RB mới, chia sẻ công suất
                b_new = np.random.choice(b_unused)  # Chọn 1 RB chưa dùng
                b_existing = np.random.choice(b_used)  # Chọn 1 RB đang cấp phát

                # Kiểm tra điều kiện 1: RB mới chưa có user nào sử dụng
                if sum(allocation[i][b_new]) == 0:
                    allocation[i][b_new][k] = 1  # Cấp phát RB mới
                    power_share = power[i][b_existing][k] * 0.10  # Chia sẻ công suất
                    power[i][b_existing][k] *= 0.90  # Công suất cũ giảm
                    power[i][b_new][k] = power_share  # Công suất mới nhận phần còn lại

            # Kiểm tra điều kiện 5: Tổng công suất của RU không vượt quá Pmax
            total_power = sum(power[i][b][k] for b in self.B[i] for k in self.K)
            if total_power > self.Pmax[i]:  
                # Nếu vượt quá Pmax, rollback và chọn nước đi khác
                allocation, power = self.state  
                continue  # Thử lại với một nước đi khác

            # Nếu trạng thái hợp lệ, cập nhật trạng thái mới
            self.state = allocation, power  
            energy = self.energy()
            self.all_x.append(allocation)
            self.all_p.append(power)
            self.energy_history.append(energy)  # Lưu lại năng lượng
            condition = False  # Thoát vòng lặp nếu hợp lệ
    
    def re_calculate(self):
        sinr = self.compute_SINR(self.best_state)
        throughput = self.compute_throughput(sinr)
        for k in self.K:
            if (throughput[k] >= self.RminK[k] * self.pi[k]):
                self.pi[k] = 1
            else :
                self.pi[k] = 0
        
    def run(self):
        self.copy_strategy = "deepcopy"
        self.best_state, self.best_energy = self.anneal()
        self.save_file()
        self.re_calculate()
        self.num_user_serve = sum(self.pi[k] for k in self.K)


    def draw_figures(self): # Vẽ hình xem tiến trình hội tụ
        plt.plot(self.energy_history)
        plt.xlabel("Iteration")
        plt.ylabel("Energy")
        plt.title("Simulated Annealing Energy Progression")
        
        id = len(os.listdir(path = "./Output/figures"))
        # Lưu ảnh vào file
        plt.savefig(f"./Output/figures/energy_plot_{id}.png")  
        plt.close()  # Đóng hình để tránh lỗi khi vẽ nhiều hình liên tiếp

    def save_file(self):
        # Tạo file để lưu
        id = len(os.listdir("./DRL/Data_DRL"))
        save_file = f"./DRL/Data_DRL/data_{id}.npz"
        np.savez_compressed(save_file, x = np.array(self.all_x, dtype = object)
                                     , p = np.array(self.all_p, dtype = object)
                                     , energy = np.array(self.energy_history, dtype = np.float32))

    

    

