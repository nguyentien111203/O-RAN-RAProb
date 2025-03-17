import cvxpy as cp
import numpy as np
import gurobipy
import common


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
    def __init__(self, K, I, H, B, Pmax, RminK, Tmin, BW, N0):
        self.K = K
        self.I = I
        self.H = H
        self.B = B
        self.Pmax = Pmax
        self.RminK = RminK
        self.Tmin = Tmin
        self.BW = BW
        self.N0 = N0
        

    def createProblem(self):
        # Biến quyết định
        pi = { k : cp.Variable(name = f"pi_{k}" ,boolean=True) for k in self.K}
        x = { (i, b, k): cp.Variable(name = f"x_{i}_{b}_{k}", boolean=True) for i in self.I for b in self.B[i] for k in self.K }
        y = { (i, k): cp.Variable(name = f"y_{i}_{k}", boolean = True) for i in self.I for k in self.K }
        p = { (i, b, k): cp.Variable(name = f"p_{i}_{b}_{k}", nonneg=True) for i in self.I for b in self.B[i] for k in self.K }
        u = { (i, b, k): cp.Variable(name = f"u_{i}_{b}_{k}", nonneg=True) for i in self.I for b in self.B[i] for k in self.K }

        # Biểu thức tính toán SINR và throughput
        SINR = { (i, b, k): (u[(i, b, k)] * self.H[k][i][b]) / (self.BW * self.N0)
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
                    constraints.append(p[(i, b, k)] <= self.Pmax[i] * x[(i, b, k)])   # Điều kiện thêm

        # Hàm mục tiêu: Tối đa throughput và số lát mạng được chấp nhận
        objective = cp.Maximize(cp.sum([common.tunning * dataRate[k] + (1 - common.tunning) * pi[k] for k in self.K]))

        # Giải bài toán tối ưu
        problem = cp.Problem(objective, constraints)
        
        return problem

    def solve(self):
        problem = self.createProblem()

        problem.solve(solver=cp.MOSEK, verbose = True)
        print(problem.is_dcp())  # Nếu là True, bài toán là convex và solver có thể giải

        variable_name_map = { var.id: var.name() for var in problem.variables() }

        return problem.solution.primal_vars, variable_name_map, problem._solve_time 
