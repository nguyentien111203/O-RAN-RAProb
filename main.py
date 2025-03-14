import ILPsolver
import input
import validate

def main():
    # Tạo đầu vào với số người dùng, số RU, số RB ở mỗi RU
    K, I, B, H = input.createEnvironmentInput(2, 2, [4,4])

    # Tạo bài toán với đầu vào trên và giải
    prob = ILPsolver.AllocationProblemILP(K = K, I = I, H = H, B = B, Pmax = [200, 250],
                                          RminK = [50, 50], Tmin = 100, BW = 180000, N0 = 4.002e-21)
    # Nên có chỗ để tạo Pmax, RminK, Tmin, BW và N0?
    solution, time = prob.solve()

    print(solution)
    print("\n" + str(time))
    
    print(validate.check_solution_constraints(solution, prob))


main()
    
    