import common
import ILPsolver

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

    # Constraint 1: Mỗi RB chỉ được gán tối đa 1 user
    for i in I:
        for b in B[i]:
            if sum(solution.get(f"x_({i}_{b}_{k})", 0) for k in K) > 1:
                return 1  # Vi phạm điều kiện 1

    # Constraint 2: Đảm bảo min data rate cho mỗi user (đã tuyến tính hóa hàm log2(1+x))
    for k in K:
        data_rate = sum(
            BW * (0.2927 * (solution.get(f"u_({i}_{b}_{k})", 0) * H[k][i][b] / (BW * N0)) + 0.8927)
            for i in I for b in B[i]
        )
        if data_rate < RminK[k] * solution.get(f"pi_{k}", 0):
            return 2  # Vi phạm điều kiện 2

    # Constraint 3: Mối quan hệ giữa x và y
    for k in K:
        for i in I:
            lhs = sum(solution.get(f"x_({i}_{b}_{k})", 0) for b in B[i]) / len(B[i])
            if lhs > solution.get(f"y_({i}_{k})", 0) + 1e-5:  # Tránh lỗi số thực
                return 3  # Vi phạm điều kiện 3

    # Constraint 4: Mối quan hệ giữa pi và y
    for k in K:
        lhs = sum(solution.get(f"y_({i}_{k})", 0) for i in I) / len(I)
        if lhs > solution.get(f"pi_{k}", 0) + 1e-5:
            return 4  # Vi phạm điều kiện 4

    # Constraint 5: Tổng công suất truyền không vượt quá Pmax của RU
    for i in I:
        total_power = sum(solution.get(f"u_({i}_{b}_{k})", 0) for b in B[i] for k in K)
        if total_power > Pmax[i]:
            return 5  # Vi phạm điều kiện 5

    # Constraint 6: Quan hệ giữa u, p và x
    for k in K:
        for i in I:
            for b in B[i]:
                u_val = solution.get(f"u_({i}_{b}_{k})", 0)
                p_val = solution.get(f"p_({i}_{b}_{k})", 0)
                x_val = solution.get(f"x_({i}_{b}_{k})", 0)
                if not (u_val <= p_val and u_val <= Pmax[i] * x_val and u_val >= p_val - Pmax[i] * x_val):
                    return 6  # Vi phạm điều kiện 6

    return 0  # Không có lỗi nào
