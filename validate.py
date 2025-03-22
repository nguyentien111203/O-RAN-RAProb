import common
import cvxpy_sol.ILPsolver as ILPsolver
import numpy as np

"""
    Kiểm tra kết quả giải của một bài toán
    Đầu vào :
        solution : 1 thư viện chứa các biến và giá trị của nó
        problem : bài toán thuộc lớp AllocationProblemILP
    Đầu ra : Một trong các số [0, 1, 2, 3, 4, 5, 6] với
        0 : Tất cả được đáp ứng
        1,2,3,4,5,6 : Điều kiện 1,2,3,4,5,6 không thỏa mãn
"""
def check_solution_constraints(solution, problem : ILPsolver.AllocationProblemILP):
    # Giải nén thông tin từ problem
    I, K, B, Pmax, RminK, BW, N0, H = (
        problem.I, problem.K, problem.B,
        problem.Pmax, problem.RminK, problem.BW, problem.N0, problem.H
    )

    x = {(i, b, k): solution.get(f"x_{i}_{b}_{k}", 0) for i in I for b in B[i] for k in K}
    y = {(i, k): solution.get(f"y_{i}_{k}", 0) for i in I for k in K}
    pi = {k: solution.get(f"pi_{k}", 0) for k in K}
    p = {(i, b, k): solution.get(f"p_{i}_{b}_{k}", 0) for i in I for b in B[i] for k in K}
    u = {(i, b, k): solution.get(f"p_{i}_{b}_{k}", 0) for i in I for b in B[i] for k in K}


    # Constraint 1: Mỗi RB chỉ được gán tối đa 1 user
    for i in I:
        for b in B[i]:
            if sum(x[(i, b, k)] for k in K) > 1:
                return 1  # Vi phạm điều kiện 1

    # Constraint 2: Đảm bảo min data rate cho mỗi user (đã tuyến tính hóa hàm log2(1+x))
    for k in K:
        data_rate = sum(
            BW * np.log2(1 + ((u[(i, b, k)] * H[k][i][b]) / (BW * N0)))
            for i in I for b in B[i]
        )
        if data_rate < RminK[k] * pi[k]:
            return 2  # Vi phạm điều kiện 2

    # Constraint 3: Mối quan hệ giữa x và y
    for k in K:
        for i in I:
            lhs = sum(x[(i, b, k)] for b in B[i]) / len(B[i])
            if (lhs > y[(i, k)] + 1e-5) :  # Tránh lỗi số thực
                return 3  # Vi phạm điều kiện 3

    # Constraint 4: Mối quan hệ giữa pi và y
    for k in K:
        lhs = sum(y[(i, k)] for i in I) / len(I)
        if (lhs > pi[k] + 1e-5):
            return 4  # Vi phạm điều kiện 4

    # Constraint 5: Tổng công suất truyền không vượt quá Pmax của RU
    for i in I:
        total_power = sum(u[(i, b, k)] for b in B[i] for k in K)
        if total_power - Pmax[i] > 1e-5:
            return 5  # Vi phạm điều kiện 5

    # Constraint 6: Quan hệ giữa u, p và x
    for k in K:
        for i in I:
            for b in B[i]:
                if not (u[(i, b, k)] <= p[(i, b, k)] and u[(i, b, k)] <= Pmax[i] * x[(i, b, k)] 
                        and u[(i, b, k)] >= p[(i, b, k)] - Pmax[i] * x[(i, b, k)]):
                    return 6  # Vi phạm điều kiện 6

    return 0  # Không có lỗi nào
