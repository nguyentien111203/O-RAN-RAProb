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

        # Ngưỡng 
        self.global_threshold_thr = 0.98 # Ngưỡng throughput giảm
        self.user_eff = 0.1              # Ngưỡng giảm số người được phục vụ

        self.test_id = test_id
        state = self.initialize_state()
        super().__init__(state)

    def initialize_state(self):
        condition = True
        while condition:
            max_rb = max(len(self.B[i]) for i in self.I)
            allocation = np.zeros((len(self.I), max_rb, len(self.K)))
            power = np.zeros((len(self.I), max_rb, len(self.K)))
            throughput = np.zeros(len(self.K))

            for i in self.I:  # Duyệt qua tất cả RU
                bws = len(self.B[i])  # Số lượng RB trong RU
                h_weights = np.zeros((bws, len(self.K)))  # Ma trận tỷ lệ hiệu suất kênh

                for b in range(bws):
                    h_values = self.H[i][b]  # Lấy hiệu suất kênh của RB này
                    sorted_k = sorted(range(len(self.K)), key=lambda k: h_values[k], reverse=True)  # Sắp xếp theo hiệu suất

                    # Ưu tiên user có mức thiếu hụt lớn nhất trong danh sách hiệu suất cao
                    top_k = sorted_k[:max(3, len(self.K) // 3)]  # Chọn nhóm top
                    selected_k = max(top_k, key=lambda k: self.RminK[k] - throughput[k])  # Ưu tiên thiếu hụt

                    # Phân bổ cho user được chọn
                    allocation[i][b][selected_k] = 1
                    h_weights[b][selected_k] = h_values[selected_k]

                # Phân phối công suất theo tỷ lệ
                total_weight = np.sum(h_weights)
                if total_weight > 0:
                    for b in range(bws):
                        for k in range(len(self.K)):
                            if allocation[i][b][k] == 1:
                                # Tính toán công suất cấp phát
                                pw = (h_weights[b][k] / total_weight) * self.Pmax[i]
                                power[i][b][k] = pw
                                throughput[k] += self.H[i][b][k] * pw

                # Kiểm soát tổng công suất không vượt quá 
                total_power = np.sum(power[i])
                if total_power > self.Pmax[i]:
                    power[i] *= self.Pmax[i] / total_power  # Điều chỉnh lại tỷ lệ công suất

            if self.check_solution_constraints(allocation, power) == 0:
                condition = False
                throughput = sum(self.compute_throughput(self.compute_SINR((allocation, power)))[k] for k in self.K)
                with open("./throughput.txt", "a") as opf:
                    opf.write(f"{throughput}\n")
                return (allocation, power)    
    
    def calculate_single(self, i, b, k, power):
        return self.BW * np.log2(1 + ((power * self.H[i][b][k]) / (self.BW * self.N0)))
            
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
            # Định kỳ mỗi 30 bước thực hiện rescue
            if len(self.actions) > 0 and len(self.actions) % 30 == 0:
                allocation, power = self.global_rescue()
                action = "global-rescue"
            else :
                throughput = self.compute_throughput(self.compute_SINR(self.state))
                delta = 0.01

                action = random.choices(
                    ["boost", "remove", "balance", "reassign_rb"],
                    weights=[0.25, 0.2, 0.3, 0.25]
                )[0]

                i = random.choice(self.I)
                b = random.choice(self.B[i]) 
                used_k = [k for k in self.K if allocation[i][b][k] == 1 and power[i][b][k] > 0]

                # BOOST: Ưu tiên user gần đạt
                if action == "boost" and used_k:
                    k = min(used_k, key=lambda k: max(0, self.RminK[k] - throughput[k]))
                    increase = delta * self.Pmax[i]
                    if np.sum(power[i]) + increase <= self.Pmax[i]:
                        power[i][b][k] += increase

                # REMOVE: Giảm nhưng không làm tụt dưới RminK
                elif action == "remove" and used_k:
                    k = max(used_k, key=lambda k: throughput[k] / self.RminK[k])
                    decrease = delta * self.Pmax[i]
                    if power[i][b][k] > decrease:
                        power[i][b][k] -= decrease
                        tp_new = self.compute_throughput(self.compute_SINR((allocation, power)))[k]
                        if tp_new < self.RminK[k]:
                            power[i][b][k] += decrease  # rollback local
                        elif power[i][b][k] < 1e-6:
                            allocation[i][b][k] = 0
                            power[i][b][k] = 0

                # BALANCE: phân phối lại power theo nhu cầu
                elif action == "balance" and used_k:
                    avg_power = self.Pmax[i] / len(self.B[i])
                    total_rb_power = np.sum(power[i][b])
                    if total_rb_power > 1.2 * avg_power:
                        reduce = total_rb_power - 1.2 * avg_power
                        for k in used_k:
                            portion = power[i][b][k] / total_rb_power
                            power[i][b][k] -= reduce * portion
                            if power[i][b][k] < 1e-6:
                                allocation[i][b][k] = 0
                                power[i][b][k] = 0
                        # Dồn lại cho các RB yếu hơn
                        under_b = [bb for bb in self.B[i] if np.sum(power[i][bb]) < 0.8 * avg_power]
                        for bb in under_b:
                            under_k = [k for k in self.K if allocation[i][bb][k] == 1]
                            for k in under_k:
                                power[i][bb][k] += reduce / max(1, len(under_b) * len(under_k))

                # REASSIGN_RB: chuyển RB từ user đủ --> user thiếu, nhưng giữ pi[k_from] = 1
                elif action == "reassign_rb":
                    over_k = [k for k in self.K if throughput[k] >= 1.1 * self.RminK[k]]
                    under_k = [k for k in self.K if throughput[k] < self.RminK[k]]
                    if over_k and under_k:
                        k_from = random.choice(over_k)
                        k_to = min(under_k, key=lambda k: self.RminK[k] - throughput[k])
                        for b in self.B[i]:
                            if allocation[i][b][k_from] == 1:
                                # Tạm chuyển RB
                                p_share = power[i][b][k_from]
                                allocation[i][b][k_from] = 0
                                power[i][b][k_from] = 0
                                allocation[i][b][k_to] = 1
                                power[i][b][k_to] = p_share

                                # Kiểm tra nếu k_from vẫn đạt RminK
                                tp_new = self.compute_throughput(self.compute_SINR((allocation, power)))
                                if tp_new[k_from] >= self.RminK[k_from]:
                                    throughput = tp_new  # commit
                                else:
                                    # rollback nếu move khiến k_from bị mất phục vụ
                                    allocation[i][b][k_from] = 1
                                    power[i][b][k_from] = p_share
                                    allocation[i][b][k_to] = 0
                                    power[i][b][k_to] = 0
                                break  # chỉ chuyển 1 RB

                # Cleanup
                for ii in self.I:
                    for bb in self.B[ii]:
                        for kk in self.K:
                            if allocation[ii][bb][kk] == 1 and power[ii][bb][kk] < 1e-6:
                                allocation[ii][bb][kk] = 0
                                power[ii][bb][kk] = 0

            if self.check_solution_constraints(allocation, power) == 0 :
                is_initial_step = len(self.actions) == 0
                is_new_action = is_initial_step or (action != self.actions[-1]) or action == "global-rescue"

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

    def global_rescue(self):
        allocation, power = self.state
        throughput = self.compute_throughput(self.compute_SINR((allocation, power)))
        pi_old = sum(1 for k in self.K if throughput[k] >= self.RminK[k])

        under_k = [k for k in self.K if throughput[k] < self.RminK[k]]
        over_k = [k for k in self.K if throughput[k] >= 1.2 * self.RminK[k]]

        # Thu thập RB có thể thu hồi từ user đủ
        reclaimable_rb = []
        for i in self.I:
            for b in self.B[i]:
                for k in over_k:
                    if self.state[0][i][b][k] == 1:
                        utility_loss = throughput[k] - self.RminK[k]
                        reclaimable_rb.append((i, b, k, power[i][b][k], utility_loss))

        # Sắp xếp theo utility loss tăng dần
        reclaimable_rb.sort(key=lambda x: x[4])

        for (i, b, k_old, p, _) in reclaimable_rb:
            if not under_k:
                break
            # Tạm ngắt RB từ user đủ
            self.state[0][i][b][k_old] = 0
            self.state[1][i][b][k_old] = 0

            # Gán lại RB cho user cần nhất
            k_new = max(under_k, key=lambda k: (self.RminK[k] - throughput[k]) * self.H[i][b][k])
            self.state[0][i][b][k_new] = 1
            self.state[1][i][b][k_new] = p

            # Cập nhật under_k
            throughput = self.compute_throughput(self.compute_SINR(self.state))
            under_k = [k for k in self.K if throughput[k] < self.RminK[k]]

        # Đánh giá lại
        pi_new = sum(1 for k in self.K if throughput[k] >= self.RminK[k])
        energy_new = self.compute_obj(self.state)

        if pi_new >= pi_old:
            self.energy_history.append(energy_new)
            self.all_x.append(self.state[0].copy())
            self.all_p.append(self.state[1].copy())
            self.actions.append("global_rescue")
            with open("./throughput.txt", "a") as f:
                f.write(f"global_rescue, {energy_new:.2f}, pi: {pi_new}/{len(self.K)}\n")

            return (allocation, power)


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

    

    

