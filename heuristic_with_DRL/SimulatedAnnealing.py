import numpy as np
from simanneal import Annealer
import matplotlib.pyplot as plt
import time
import os
import common
import math
import random



class RBAllocationSA(Annealer):
    id = 6 #5
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
            # Initialize state ensuring each RB serves at most one user"""
            # Create empty allocation and power matrices
            num_rbs = max(len(self.B[i]) for i in self.I)
            allocation = np.zeros((len(self.I), num_rbs, len(self.K)))
            power = np.zeros_like(allocation)
            
            # Track available RBs and remaining power
            available_rbs = {(i, b) for i in self.I for b in self.B[i]}
            ru_power_remaining = {i: self.Pmax[i] for i in self.I}
            
            # Create list of all possible user-RB allocations with their channel quality
            candidate_allocations = []
            for i in self.I:
                for b in self.B[i]:
                    for k in self.K:
                        candidate_allocations.append((
                            self.H[i][b][k],  # Channel quality as primary sort key
                            random.random(),   # Secondary random key for diversity
                            i, b, k           # Allocation identifiers
                        ))
            
            # Sort by channel quality (descending) and random factor
            candidate_allocations.sort(reverse=True, key=lambda x: (x[0], x[1]))
            
            # Assign best user-RB pairs first
            for h_val, _, i, b, k in candidate_allocations:
                if (i, b) not in available_rbs:
                    continue  # RB already assigned
                    
                if ru_power_remaining[i] <= 0:
                    continue  # No power left in this RU
                    
                # Calculate reasonable power allocation
                rb_power = min(
                    0.9 * ru_power_remaining[i] / len(self.B[i]),  # Equal share with headroom
                    (self.N0 * self.BW) / h_val * (2**(self.RminK[k]/self.BW) - 1) if self.RminK[k] > 0 else float('inf')
                )
                
                if rb_power > 1e-6:  # Significant power allocation
                    allocation[i][b][k] = 1
                    power[i][b][k] = rb_power
                    available_rbs.remove((i, b))
                    ru_power_remaining[i] -= rb_power
            
            # Second pass to ensure minimum rate requirements
            for k in self.K:
                current_rate = sum(
                    self.BW * np.log2(1 + (power[i][b][k] * self.H[i][b][k] / (self.N0 * self.BW)))
                    for i in self.I for b in self.B[i] if allocation[i][b][k] > 0
                )
                
                if self.RminK[k] > 0 and current_rate < self.RminK[k]:
                    # Find best available RB to add for this user
                    best_rb = None
                    best_h = 0
                    for i in self.I:
                        for b in self.B[i]:
                            if (i, b) in available_rbs:  # Only unassigned RBs
                                if self.H[i][b][k] > best_h:
                                    best_h = self.H[i][b][k]
                                    best_rb = (i, b)
                    
                    if best_rb and ru_power_remaining[best_rb[0]] > 0:
                        i, b = best_rb
                        needed_power = (self.N0 * self.BW) / self.H[i][b][k] * (
                            2**((self.RminK[k] - current_rate)/self.BW) - 1
                        )
                        allocated_power = min(needed_power, ru_power_remaining[i])
                        
                        if allocated_power > 1e-6:
                            allocation[i][b][k] = 1
                            power[i][b][k] = allocated_power
                            available_rbs.remove((i, b))
                            ru_power_remaining[i] -= allocated_power         

            if self.check_solution_constraints(allocation, power) == 0:
                condition = False
                throughput = sum(self.compute_throughput(self.compute_SINR((allocation, power)))[k] for k in self.K)
                with open("./throughput.txt", "a") as opf:
                    opf.write(f"{throughput}")
                with open("./power.txt", "a") as opp:
                    opf.write(str(power))
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
            #throughput = self.compute_throughput(self.compute_SINR(self.state))
            delta = 0.01  # Power adjustment factor
            action = random.choices(
                ["boost", "remove", "balance", "refocus"], weights=[0.4, 0.1, 0.3, 0.2]
            )[0]

            i = random.choice(self.I)
            b = random.choice(self.B[i])
            used_k = [k for k in self.K if allocation[i][b][k] == 1 and power[i][b][k] > 0]


            if action == "boost" and used_k:
                k = used_k[0]
                if np.sum(power[i]) + delta * self.Pmax[i] <= self.Pmax[i]:
                    power[i][b][k] += delta * self.Pmax[i]


            elif action == "remove" and used_k:
                k = used_k[0]
                if power[i][b][k] > delta * self.Pmax[i]:
                    power[i][b][k] -= delta * self.Pmax[i]
                    if power[i][b][k] < 1e-6:
                        allocation[i][b][k] = 0
                        power[i][b][k] = 0


            elif action == "balance":
                avg_power = self.Pmax[i] / len(self.B[i])
                total_rb_power = np.sum(power[i][b])
                if total_rb_power > 1.2 * avg_power and used_k:
                    k = used_k[0]
                    reduce = min(0.5 * delta * self.Pmax[i], power[i][b][k])
                    power[i][b][k] -= reduce
                    if power[i][b][k] < 1e-6:
                        allocation[i][b][k] = 0
                        power[i][b][k] = 0

                elif total_rb_power < 0.8 * avg_power:
                    best_k = max(self.K, key=lambda k: self.H[i][b][k])
                    for k2 in self.K:
                        if k2 != best_k:
                            allocation[i][b][k2] = 0
                            power[i][b][k2] = 0
                    allocation[i][b][best_k] = 1
                    power[i][b][best_k] += delta * self.Pmax[i]

            elif action == "refocus":
                # Assign RB to a better user
                if not used_k:
                    best_k = max(self.K, key=lambda k: self.H[i][b][k])
                    for k2 in self.K:
                        allocation[i][b][k2] = 0
                        power[i][b][k2] = 0
                    allocation[i][b][best_k] = 1
                    power[i][b][best_k] = delta * self.Pmax[i]

                else:
                    current_k = used_k[0]
                    better_k = max(self.K, key=lambda k: self.H[i][b][k])
                    if better_k != current_k and self.H[i][b][better_k] > self.H[i][b][current_k] * 1.3:
                        transfer = power[i][b][current_k]
                        allocation[i][b][current_k] = 0
                        power[i][b][current_k] = 0
                        allocation[i][b][better_k] = 1
                        power[i][b][better_k] = transfer


            if self.check_solution_constraints(allocation, power) == 0:
                self.state = (allocation, power)
                self.energy_history.append(self.energy())
                self.all_x.append(allocation)
                self.all_p.append(power)
                self.actions.append(action)
                
                throughput = sum(self.compute_throughput(self.compute_SINR(self.state))[k] for k in self.K)
                with open("./throughput.txt", "a") as opf:
                    opf.write(f"{throughput}")
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
                throughput[k] for k in self.K
            )

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

    

    

