import ILPsolver
import input
import validate

def main():
    # Tạo đầu vào với số người dùng, số RU, số RB ở mỗi RU
    K, I, B, H = input.createEnvironmentInput(3, 3, [5,5,5])

    # Tạo bài toán với đầu vào trên và giải
    prob = ILPsolver.AllocationProblemILP(K = K, I = I, H = H, B = B, Pmax = [200, 250, 200],
                                          RminK = [50, 50, 50], Tmin = 100, BW = 180000, N0 = 7.02e-6)
    # Nên có chỗ để tạo Pmax, RminK, Tmin, BW và N0?
    solution, map, time = prob.solve()

    sol_map = {}
    for var_id, value in solution.items():
        var_name = map.get(var_id, f"Unknown_{var_id}")
        sol_map[var_name] = value
        print(f"{var_name} = {value}")

    
    print(validate.check_solution_constraints(sol_map, prob))


main()
    
    