import pulp
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
        pi = pulp.LpVariable.dicts("pi", self.K, cat=pulp.LpBinary)
        x = pulp.LpVariable.dicts("x", [(i, b, k) for i in self.I for b in self.B[i] for k in self.K], cat=pulp.LpBinary)
        y = pulp.LpVariable.dicts("y", [(i, k) for i in self.I for k in self.K], cat=pulp.LpBinary)
        p = pulp.LpVariable.dicts("p", [(i, b, k) for i in self.I for b in self.B[i] for k in self.K], lowBound = 0)
        u = pulp.LpVariable.dicts("u", [(i, b, k) for i in self.I for b in self.B[i] for k in self.K], lowBound = 0)
        
        # Tạo hàm tính datarate (xấp xỉ tuyến tính log2(1 + x) = ax + b để có thể giải bài toán tuyến tính)
        dataRate = {
            k: pulp.lpSum(self.BW * (common.a * ((u[(i, b, k)] * self.H[k][i][b]) / (self.BW * self.N0)) + common.b)
                           for i in self.I for b in self.B[i]) for k in self.K
        }
        
        allocateProb = pulp.LpProblem("Allocation Problem", pulp.LpMaximize)

        # Ràng buộc 1: Mỗi RB chỉ có thể gán cho một user
        for i in self.I:
            for b in self.B[i]:
                allocateProb += pulp.lpSum(x[(i, b, k)] for k in self.K) <= 1

        # Ràng buộc 2: Đảm bảo tốc độ dữ liệu tối thiểu
        for k in self.K:
            allocateProb += dataRate[k] >= self.RminK[k] * pi[k]

        # Ràng buộc 3: Liên kết tuyến tính giữa x và y
        for k in self.K:
            for i in self.I:
                allocateProb += (pulp.lpSum(x[(i, b, k)] for b in self.B[i]) / len(self.B[i])) <= y[i, k]
                allocateProb += (pulp.lpSum(x[(i, b, k)] for b in self.B[i]) / len(self.B[i])) + 1 - 1e-5 >= y[i, k]

        # Ràng buộc 4: Liên kết giữa pi và y
        for k in self.K:
            allocateProb += (pulp.lpSum(y[(i, k)] for i in self.I) / len(self.I)) <= pi[k]
            allocateProb += (pulp.lpSum(y[(i, k)] for i in self.I) / len(self.I)) + 1 - 1e-5 >= pi[k]

        # Ràng buộc 5: Tổng công suất không vượt quá giới hạn của RU
        for i in self.I:
            allocateProb += (pulp.lpSum(u[(i, b, k)] for b in self.B[i] for k in self.K)) <= self.Pmax[i]

        # Ràng buộc 6: Liên kết tuyến tính giữa u, p, x
        for k in self.K:
            for i in self.I:
                for b in self.B[i]:
                    allocateProb += u[(i, b, k)] <= p[(i, b, k)]
                    allocateProb += u[(i, b, k)] <= self.Pmax[i] * x[(i, b, k)]
                    allocateProb += u[(i, b, k)] >= p[(i, b, k)] - self.Pmax[i] * x[(i, b, k)]

        # Hàm mục tiêu: Tối đa throughput và số lát mạng được chấp nhận
        allocateProb += pulp.lpSum(common.tunning * dataRate[k] + (1 - common.tunning) * pi[k] for k in self.K)

        return allocateProb

    def solve(self):
        allocateProb = self.createProblem()
        allocateProb.solve(solver=pulp.GUROBI_CMD())
        
        return {v.name: v.varValue for v in allocateProb.variables()}, allocateProb.solutionTime
