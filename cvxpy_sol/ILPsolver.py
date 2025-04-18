import cvxpy as cp
import numpy as np
import gurobipy
import common
import mosek

"""
    Lớp bài toán O-RAN Resource Allocation
    Yêu cầu đầy đủ các đầu vào :
        K : tập người dùng
        I : tập các RU
        H : tập giá trị tính toán từ vector kênh truyền tương ứng
        B : tập các RB của RU
        Pmax : công suất tối đa của từng RU (W)
        RminK : datarate tối thiểu cho từng người dùng (Mbps)
        Tmin : throughput tối thiểu (Mbps)
        BW : băng thông mỗi RB chiếm (Hz)
        N0 : mật độ công suất tạp âm (W/Hz)
"""
class AllocationProblemILP():
    def __init__(self, K, I, H, B, Pmax, RminK, Thrmin, BW, N0):
        self.K = K
        self.I = I
        self.H = H
        self.B = B
        self.Pmax = Pmax
        self.RminK = RminK
        self.Thrmin = Thrmin
        self.BW = BW
        self.N0 = N0

        self.sol_map = {}
        self.time = 0
        self.num_user_serve = 0
        self.check = True
        self.throughput = 0
        

    def createProblem(self):
        # Biến quyết định
        pi = { k : cp.Variable(name = f"pi_{k}" ,boolean=True) for k in self.K}
        x = { (i, b, k): cp.Variable(name = f"x_{i}_{b}_{k}", boolean=True) for i in self.I for b in self.B[i] for k in self.K }
        y = { (i, k): cp.Variable(name = f"y_{i}_{k}", boolean = True) for i in self.I for k in self.K }
        p = { (i, b, k): cp.Variable(name = f"p_{i}_{b}_{k}", nonneg=True) for i in self.I for b in self.B[i] for k in self.K }
        u = { (i, b, k): cp.Variable(name = f"u_{i}_{b}_{k}", nonneg=True) for i in self.I for b in self.B[i] for k in self.K }

        # Biểu thức tính toán SINR và throughput
        SINR = { (i, b, k): (u[(i, b, k)] * self.H[i][b][k]) / (self.BW * self.N0)
            for i in self.I for b in self.B[i] for k in self.K}
       
        dataRate = { 
        k: cp.sum([
            self.BW * cp.log1p(1 + SINR[(i, b, k)]) / np.log(2) 
            for i in self.I for b in self.B[i]
            ]) 
            for k in self.K
        }

        # Danh sách ràng buộc
        constraints = []

        # Ràng buộc 1: Mỗi RB chỉ có thể gán cho một user
        for i in self.I:
            for b in self.B[i]:
                constraints.append(cp.sum([x[(i, b, k)] for k in self.K]) <= 1)

        # Ràng buộc 2: Đảm bảo tốc độ dữ liệu tối thiểu
        for k in self.K:
            constraints.append(dataRate[k] >= self.RminK[k] * pi[k])

        # Ràng buộc 3: Liên kết giữa x và y
        for k in self.K:
            for i in self.I:
                constraints.append(cp.sum([x[(i, b, k)] for b in self.B[i]]) / len(self.B[i]) <= y[(i, k)])
                constraints.append(cp.sum([x[(i, b, k)] for b in self.B[i]]) / len(self.B[i]) + 1 - 1e-5 >= y[(i, k)])

        # Ràng buộc 4: Liên kết giữa pi và y
        for k in self.K:
            constraints.append(cp.sum([y[(i, k)] for i in self.I]) / len(self.I) <= pi[k])
            constraints.append(cp.sum([y[(i, k)] for i in self.I]) / len(self.I) + 1 - 1e-5 >= pi[k])

        # Ràng buộc 5: Tổng công suất không vượt quá giới hạn của RU
        for i in self.I:
            constraints.append(cp.sum([u[(i, b, k)] for b in self.B[i] for k in self.K]) <= self.Pmax[i])

        # Ràng buộc 6: Liên kết giữa u, p, x
        for k in self.K:
            for i in self.I:
                for b in self.B[i]:
                    constraints.append(u[(i, b, k)] <= p[(i, b, k)])
                    constraints.append(u[(i, b, k)] <= self.Pmax[i] * x[(i, b, k)])
                    constraints.append(u[(i, b, k)] >= p[(i, b, k)] - self.Pmax[i] * x[(i, b, k)])

        # Hàm mục tiêu: Tối đa throughput và số lát mạng được chấp nhận
        objective = cp.Maximize(cp.sum([(1 - common.tunning) * (dataRate[k]/ self.Thrmin) + common.tunning * pi[k] for k in self.K]))

        # Giải bài toán tối ưu
        problem = cp.Problem(objective, constraints)
        
        return problem

    def solve(self):
       
        problem = self.createProblem()

        # Thiết lập số luồng cho MOSEK
        mosek_params = {
            "MSK_IPAR_NUM_THREADS": 6,  # Sử dụng 8 luồng (có thể điều chỉnh)
        }

        problem.solve(solver=cp.MOSEK, verbose = True, mosek_params = mosek_params)

        self.sol_map = problem.var_dict
        self.num_user_serve = sum(self.sol_map.get(f"pi_{k}").value for k in self.K)
        self.check = self.check_solution()
        self.throughput = (problem.objective.value - common.tunning * self.num_user_serve) * self.Thrmin/(1 - common.tunning)
        
        
        self.time = problem._solve_time
        
        
    
    def check_solution(self):
        x = {(i, b, k): self.sol_map.get(f"x_{i}_{b}_{k}", 0).value for i in self.I for b in self.B[i] for k in self.K}
        y = {(i, k): self.sol_map.get(f"y_{i}_{k}", 0).value for i in self.I for k in self.K}
        pi = {k: self.sol_map.get(f"pi_{k}", 0).value for k in self.K}
        p = {(i, b, k): self.sol_map.get(f"p_{i}_{b}_{k}", 0).value for i in self.I for b in self.B[i] for k in self.K}
        u = {(i, b, k): self.sol_map.get(f"u_{i}_{b}_{k}", 0).value for i in self.I for b in self.B[i] for k in self.K}
        
        # Biểu thức tính toán SINR và throughput
        SINR = { (i, b, k): (u[(i, b, k)] * self.H[i][b][k]) / (self.BW * self.N0)
            for i in self.I for b in self.B[i] for k in self.K}
        dataRate = { 
        k: cp.sum([
            self.BW * np.log1p(1 + SINR[(i, b, k)]) / np.log(2) 
            for i in self.I for b in self.B[i]
            ]) 
            for k in self.K
        }

        self.throughput = sum([dataRate[k] for k in self.K])
    
        # Constraint 1: Mỗi RB chỉ được gán tối đa 1 user
        for i in self.I:
            for b in self.B[i]:
                if (sum(x[(i, b, k)] for k in self.K) >= (1 + 1e-5)):
                    return False  # Vi phạm điều kiện 1

        # Constraint 2: Đảm bảo min data rate cho mỗi user (đã tuyến tính hóa hàm log2(1+x))
        for k in self.K:
            data_rate = sum(
                self.BW * np.log2(1 + ((u[(i, b, k)] * self.H[i][b][k]) / (self.BW * self.N0)))
                for i in self.I for b in self.B[i]
            )
            if data_rate < self.RminK[k] * pi[k]:
                return False  # Vi phạm điều kiện 2

        # Constraint 3: Mối quan hệ giữa x và y
        for k in self.K:
            for i in self.I:
                lhs = sum(x[(i, b, k)] for b in self.B[i]) / len(self.B[i])
                if (lhs > y[(i, k)] + 1e-5) :  # Tránh lỗi số thực
                    return False  # Vi phạm điều kiện 3

        # Constraint 4: Mối quan hệ giữa pi và y
        for k in self.K:
            lhs = sum(y[(i, k)] for i in self.I) / len(self.I)
            if (lhs > pi[k] + 1e-5):
                return False  # Vi phạm điều kiện 4

        # Constraint 5: Tổng công suất truyền không vượt quá Pmax của RU
        for i in self.I:
            total_power = sum(u[(i, b, k)] for b in self.B[i] for k in self.K)
            if total_power - self.Pmax[i] > 1e-5:
                return False  # Vi phạm điều kiện 5

        # Constraint 6: Quan hệ giữa u, p và x
        for k in self.K:
            for i in self.I:
                for b in self.B[i]:
                    if not (u[(i, b, k)] <= p[(i, b, k)] and u[(i, b, k)] <= self.Pmax[i] * x[(i, b, k)] 
                            and u[(i, b, k)] >= p[(i, b, k)] - self.Pmax[i] * x[(i, b, k)]):
                        return False  # Vi phạm điều kiện 6

        return True  # Không có lỗi nào

        
