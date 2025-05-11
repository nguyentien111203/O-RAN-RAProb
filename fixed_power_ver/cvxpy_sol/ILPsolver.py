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
        P : tập công suất của từng RB cho từng RU
        RminK : datarate tối thiểu cho từng người dùng (Mbps)
        Tmin : throughput tối thiểu (Mbps)
        BandW : băng thông mỗi RB chiếm (Hz) (khác nhau)
        N0 : mật độ công suất tạp âm (W/Hz)
"""
class AllocationProblemILP():
    def __init__(self, K, I, H, B, P, RminK, Thrmin, BandW, N0):
        self.K = K
        self.I = I
        self.H = H
        self.B = B
        self.P = P
        self.RminK = RminK
        self.Thrmin = Thrmin
        self.BandW = BandW
        self.N0 = N0

        self.sol_map = {}
        self.time = 0
        self.num_user_serve = 0
        self.check = True
        self.throughput = 0
        

    def createProblem(self):
        # Biến quyết định
        pi = {k : cp.Variable(name = f"pi_{k}" ,boolean=True) for k in self.K}
        # x : tỷ lệ phân bổ
        x = { (i, k): cp.Variable(name = f"x_{i}_{k}", nonneg = True) for i in self.I for k in self.K }
        
        # Biểu thức tính toán SINR và throughput
        SINR = { (i, k): (self.P[i] * self.H[i][k]) / (self.BandW * self.N0) for i in self.I for k in self.K}
       
        dataRate = { 
        k: cp.sum([
             x[(i, k)] * self.B[i] * self.BandW * cp.log1p(SINR[(i, k)]) / np.log(2) 
            for i in self.I
            ]) 
            for k in self.K
        }

        # Danh sách ràng buộc
        constraints = []

        # Ràng buộc 1 : Tổng các tỷ lệ phải bằng 1
        for i in self.I:
            constraints.append(cp.sum([x[(i, k)] for k in self.K]) == 1.0) 
        # Ràng buộc 2: Đảm bảo tốc độ dữ liệu tối thiểu
        for k in self.K:
            constraints.append(dataRate[k] >= self.RminK[k] * pi[k])

        # Ràng buộc 4: Liên kết giữa pi và x
        for k in self.K:
            constraints.append(cp.sum([x[(i, k)] for i in self.I]) / len(self.I) <= pi[k])
            constraints.append(cp.sum([x[(i, k)] for i in self.I]) / len(self.I) + 1 - 1e-5 >= pi[k])

        # Hàm mục tiêu: Tối đa throughput và số lát mạng được chấp nhận
        objective = cp.Maximize(cp.sum([(1 - common.tunning) * (dataRate[k]/ self.Thrmin) + common.tunning * pi[k] for k in self.K]))

        # Giải bài toán tối ưu
        problem = cp.Problem(objective, constraints)
        
        return problem

    def solve(self):
       
        problem = self.createProblem()

        print(problem)

        problem.solve(solver=cp.MOSEK, verbose = True)

        self.sol_map = problem.var_dict
        self.check = self.check_solution()
        
        self.time = problem._solve_time
        
        
    def check_solution(self):
        x = {(i, k): self.sol_map.get(f"x_{i}_{k}", 0).value for i in self.I for k in self.K}
        pi = {k: self.sol_map.get(f"pi_{k}", 0).value for k in self.K}
        
        # Biểu thức tính toán SINR và throughput
        SINR = { (i, k): (self.P[i] * self.H[i][k]) / (self.BandW * self.N0)
            for i in self.I for k in self.K}
        print(x)
        dataRate = { 
        k: cp.sum([
            self.BandW * x[(i, k)] * self.B[i] * np.log1p(SINR[(i, k)]) / np.log(2) 
            for i in self.I
            ]) 
            for k in self.K
        }

        self.throughput = sum([dataRate[k] for k in self.K])
        self.num_user_serve = sum([pi[k] for k in self.K])

        # Constraint 1 : Tổng các tỷ lệ phải bằng 1
        for i in self.I:
            sum_rate = sum(x[(i, k)] for k in self.K)
            if sum_rate - 1 > 0:
                return False # Vi phạm điều kiện 1

        # Constraint 2: Đảm bảo min data rate cho mỗi user 
        for k in self.K:
            data_rate = sum(
                self.BandW * np.log2(1 + ((x[(i, k)] * self.B[i] * self.P[i] * self.H[i][k]) / (self.BandW * self.N0)))
                for i in self.I
            )
            if data_rate < self.RminK[k] * pi[k]:
                return False  # Vi phạm điều kiện 2


        # Constraint 4: Mối quan hệ giữa pi và x
        for k in self.K:
            lhs = sum(x[(i, k)] for i in self.I) / len(self.I)
            if (lhs > pi[k] + 1e-5):
                return False  # Vi phạm điều kiện 4

        return True  # Không có lỗi nào

        
