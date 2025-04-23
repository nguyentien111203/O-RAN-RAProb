import numpy as np
from simanneal import Annealer
import matplotlib.pyplot as plt
import time
import os
import common
import math
import random



class RBAllocationSA(Annealer):
    id = 7 #5
    def __init__(self, K, I, H, B, Pmax, RminK, Thrmin, BW, N0, step_SA, Tmax, Tmin, test_id):
        self.K = K  # Tập người dùng
        self.I = I  # Tập RU
        self.H = H  # Ma trận channel vector
        self.B = B  # Tập số RB của từng RU
        self.Pmax = Pmax  # Công suất tối đa của từng RU
        self.RminK = RminK  # Data rate tối thiểu của từng user
        self.Thrmin = Thrmin  # Throughput tối thiểu
        self.BW = BW  # Băng thông của từng RB
        self.N0 = N0  # Mật độ công suất tạp âm
        
        # Khởi tạo biến người dùng
        self.pi = {k : 0 for k in self.K}
        self.throughput_SA = 0

        self.num_user_serve = 0
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.steps = step_SA
        self.energy_history = []
        self.time = 0
        self.current_step = 0
        self.all_x = []
        self.all_p = []
        self.actions = []

        self.test_id = test_id
        state = self.initialize_state()
        super().__init__(state)

    def initialize_state(self):
        condition = True
        while condition :
            max_rb = max(len(self.B[i]) for i in self.I)
            allocation = np.zeros((len(self.I), max_rb, len(self.K)))
            power = np.zeros((len(self.I), max_rb, len(self.K)))

            for i in self.I:
                ru_power_remaining = self.Pmax[i]
                bws = len(self.B[i])

                # Chuẩn hóa tổng H[i][b][k] để cấp power tỉ lệ thuận
                h_weights = np.zeros((bws, len(self.K)))
                for b in range(bws):
                    h_values = self.H[i][b]
                    sorted_k = sorted(range(len(self.K)), key=lambda k: h_values[k], reverse=True)
                    top_k = sorted_k[:max(3, len(self.K)//3)]  # scale theo số user

                    selected_k = random.choice(top_k)
                    allocation[i][b][selected_k] = 1
                    h_weights[b][selected_k] = h_values[selected_k]

                # Phân phối power theo tỷ lệ h_weights
                total_weight = np.sum(h_weights)
                if total_weight > 0:
                    for b in range(bws):
                        for k in range(len(self.K)):
                            if allocation[i][b][k] == 1:
                                pw = (h_weights[b][k] / total_weight) * self.Pmax[i]
                                power[i][b][k] = pw

                # Chốt lại tổng power không vượt Pmax[i]
                total_power = np.sum(power[i])
                if total_power > self.Pmax[i]:
                    power[i] *= self.Pmax[i] / total_power

            if self.check_solution_constraints(allocation, power) == 0:
                condition = False
                throughput = sum(self.compute_throughput(self.compute_SINR((allocation, power)))[k] for k in self.K)
                with open("./throughput.txt", "a") as opf:
                    opf.write(f"{throughput}\n")
                return (allocation, power)    
            
    def compute_SINR(self, state):
        allocation, power = state
        SINR = {i : np.zeros((len(self.B[i]), len(self.K))) for i in self.I}
        for i in self.I:
            for b in self.B[i]:
                for k in self.K:
                    signal = allocation[i][b][k] * power[i][b][k] * self.H[i][b][k]
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
                sum_b = sum([x[i][b][k] for k in self.K])
                if sum_b > 1.0:
                    return 1  # Vi phạm điều kiện 1

        # Constraint 5: Tổng công suất truyền không vượt quá Pmax của RU
        for i in self.I:
            total_power = sum(x[i][b][k] * p[i][b][k] for b in self.B[i] for k in self.K)
            t = total_power - self.Pmax[i]
            if t > 0.1e-3:
                return 5  # Vi phạm điều kiện 5

        return 0  # Không có lỗi nào

    def compute_obj(self, state):
        SINR = self.compute_SINR(state)
        throughput = self.compute_throughput(SINR)
        
        # Hàm tối ưu của bài toán
        cost = -sum(common.tunning * (throughput[k] / self.Thrmin) + (1 - common.tunning) * self.pi[k] for k in self.K)

        # Đại lượng phạt nếu không đảm bảo throughput cho người dùng
        penalty = common.lamda_penalty * sum(np.log1p(max(0, (self.RminK[k] - throughput[k]) / self.Thrmin)) for k in self.K)

        # Tối đa hóa hàm mục tiêu và tối thiểu hóa đại lượng phạt 
        return cost + penalty
    

    def energy(self):
        return self.compute_obj(self.state)
    
    def move(self):
        condition = True
        while condition:
            allocation, power = self.state
            throughput = self.compute_throughput(self.compute_SINR(self.state))
            delta = 0.01

            action = random.choices(
                ["boost", "remove", "balance", "rescue_user"], weights=[0.4, 0.2, 0.2, 0.2]
            )[0]

            i = random.randint(0, len(self.I) - 1)
            b = random.randint(0, len(self.B[i]) - 1)
            used_k = [k for k in range(len(self.K)) if allocation[i][b][k] == 1 and power[i][b][k] > 0]

            if action == "boost" and used_k:
                k = min(used_k, key=lambda k: throughput[k])  # Ưu tiên user yếu hơn
                increase = delta * self.Pmax[i]
                if np.sum(power[i]) + increase <= self.Pmax[i]:
                    power[i][b][k] += increase

            elif action == "remove" and used_k:
                k = max(used_k, key=lambda k: throughput[k])  # Giảm với user thừa
                decrease = delta * self.Pmax[i]
                if power[i][b][k] > decrease:
                    power[i][b][k] -= decrease
                    if power[i][b][k] < 1e-6:
                        allocation[i][b][k] = 0
                        power[i][b][k] = 0

            elif action == "balance":
                avg_power = self.Pmax[i] / len(self.B[i])
                total_rb_power = np.sum(power[i][b])
                if total_rb_power > 1.2 * avg_power and used_k:
                    k = used_k[0]
                    reduce = min(delta * self.Pmax[i], power[i][b][k])
                    power[i][b][k] -= reduce
                    if power[i][b][k] < 1e-6:
                        allocation[i][b][k] = 0

                elif total_rb_power < 0.8 * avg_power:
                    best_k = max(range(len(self.K)), key=lambda k: self.H[i][b][k])
                    allocation[i][b] = np.zeros(len(self.K))
                    power[i][b] = np.zeros(len(self.K))
                    allocation[i][b][best_k] = 1
                    power[i][b][best_k] = delta * self.Pmax[i]

            # clean
            for i in self.I:
                for b in self.B[i]:
                    for k in range(len(self.K)):
                        if allocation[i][b][k] == 1 and power[i][b][k] < 1e-6:
                            allocation[i][b][k] = 0
                            power[i][b][k] = 0

                if self.check_solution_constraints(allocation, power) == 0 :
                    is_initial_step = len(self.actions) == 0
                    is_new_action = is_initial_step or (action != self.actions[-1])

                    if is_new_action:
                        self.state = (allocation, power)
                        self.energy_history.append(self.energy())
                        self.all_x.append(allocation)
                        self.all_p.append(power)
                        self.actions.append(action)
                        
                        throughput = sum(self.compute_throughput(self.compute_SINR(self.state))[k] for k in self.K)
                        with open("./throughput.txt", "a") as opf:
                            opf.write(f"{action}, {throughput}\n")
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
                        power[i][b][k] = 0

        # Tính lại SINR và throughput sau khi cập nhật x
        sinr = self.compute_SINR((allocation, power))
        throughput = self.compute_throughput(sinr)

        # Cập nhật throughput tổng cộng
        self.throughput_SA = sum(
                throughput[k] for k in self.K
            )
        with open("./power.txt", "a") as opp:
            opp.write(str(power))

    def run(self):
        self.copy_strategy = "deepcopy"
        start = time.time()
        self.best_state, self.best_energy = self.anneal()
        end = time.time()
        allocate, power = self.best_state
        self.all_x.append(allocate)
        self.all_p.append(power)
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
                                     , energy = np.array(self.energy_history, dtype = np.float32)
                                     , action = np.array(self.actions, dtype = np.str_) 
                                     , allow_pickle = True)

    

    

