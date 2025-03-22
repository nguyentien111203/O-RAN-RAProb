import cvxpy_sol.ILPsolver as ILPsolver
import Input.input as input
import validate
from heuristic_with_DRL import SimulatedAnnealing
import csv
import createtest
import ast

def main():
    """
    # Tạo đầu vào với số người dùng, số RU, số RB ở mỗi RU
    K, I, B, H = input.createEnvironmentInput(4, 4, [5,5,5,5])

    
    # Tạo bài toán với đầu vào trên và giải
    prob = ILPsolver.AllocationProblemILP(K = K, I = I, H = H, B = B, Pmax = [300, 350, 400, 450],
                                          RminK = [100, 100, 120, 140], Tmin = 100, BW = 180000, N0 = 7.02e-6)
    
    prob.solve()

    prob.write_file("./CVX_sol.txt")
    
    
    prob_SA = SimulatedAnnealing.RBAllocationSA(K = K, I = I, H = H, B = B, P, BW = 180000, N0 = 7.02e-6, T = 100)
    prob_SA.run()
    prob_SA.write_file("./heuristic_with_DRL/solution.txt")
    prob_SA.draw_figures()
    """
    # Mở CSV input_file với open
    input_file = "./Input/input_file.csv"
    with open(input_file, mode = 'r') as csvfile:
        # Tạo một csv reader
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            numuser, numRU, RBeachRU, Pmax, RminK, Tmin, step_SA = row

            numuser = int(numuser)
            numRU = int(numRU)
            RBeachRU = ast.literal_eval(RBeachRU)
            Pmax = ast.literal_eval(Pmax)
            RminK = ast.literal_eval(RminK)
            Tmin = float(Tmin)
            step_SA = int(step_SA)

            # Tạo đầu vào cho bài toán
            K, I, B, H  = input.createEnvironmentInput(numuser, numRU, RBeachRU)
            """
            # Giải bài toán với CVXPY
            prob = ILPsolver.AllocationProblemILP(K = K, I = I, H = H, B = B, Pmax = Pmax,
                                            RminK = RminK, Tmin = Tmin, BW = 180000, N0 = 7.02e-6)
            
            prob.solve()
            """
            # Giải với SA
            prob_SA = SimulatedAnnealing.RBAllocationSA(K = K, I = I, H = H, B = B, Pmax = Pmax,
                                            RminK = RminK, Tmin = Tmin, BW = 180000, N0 = 7.02e-6, step_SA = step_SA)
            prob_SA.run()
            prob_SA.draw_figures()

            # Ghi kết quả vào file
            createtest.write_data_test("./Output/output.csv",numuser, numRU, RBeachRU, 0,
                                    0, step_SA = prob_SA.steps,
                                    numuser_SA = prob_SA.num_user_serve)

main()
    
    