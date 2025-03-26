import cvxpy_sol.ILPsolver as ILPsolver
import Input.input as input
import validate
from heuristic_with_DRL import SimulatedAnnealing
import csv
import createtest
import ast
import numpy as np

def main():

    # Mở CSV input_file với open
    input_file = "./Input/input_file.csv"
    with open(input_file, mode = 'r') as csvfile:
        # Tạo một csv reader
        reader = csv.reader(csvfile)
        next(reader)
        line = 0
        for row in reader:
            line += 1
            numuser, numRU, RBeachRU, Pmax, RminK, Tmin, BW, N0, step_SA = row

            numuser = int(numuser)
            numRU = int(numRU)
            RBeachRU = ast.literal_eval(RBeachRU)
            Pmax = ast.literal_eval(Pmax)
            RminK = ast.literal_eval(RminK)
            Tmin = float(Tmin)
            BW = float(BW)
            N0 = float(N0)
            step_SA = int(step_SA)
            

            # Tạo đầu vào cho bài toán
            K, I, B, H  = input.createEnvironmentInput(numuser, numRU, RBeachRU)

            np.savez_compressed(f"./Input/Input_data/input_data_{line}.npz", I = I, B = B, K = K, H = H, RminK = RminK, Pmax=Pmax, 
                                Tmin = Tmin, BW = BW, N0 = N0, allow_pickle = True)
            
            # Giải bài toán với CVXPY
            prob = ILPsolver.AllocationProblemILP(K = K, I = I, H = H, B = B, Pmax = Pmax,
                                            RminK = RminK, Tmin = Tmin, BW = BW, N0 = N0)
            # RminK : Mbps, BW : MHz, N0 : mW/MHz, Pmax : mW
            
            prob.solve()
            
            for t in range(10):
                throughput_SA_1 = []
                # Giải với SA
                prob_SA = SimulatedAnnealing.RBAllocationSA(K = K, I = I, H = H, B = B, Pmax = Pmax,
                                                RminK = RminK, Tmin = Tmin, BW = BW, N0 = N0, step_SA = step_SA, test_id = t)
                prob_SA.run()
                prob_SA.draw_figures()
                
                # Ghi kết quả vào file
                createtest.write_data_test("./Output/output.csv", t, numuser, numRU, RBeachRU, prob.time, prob.throughput,
                                        prob.num_user_serve, prob.check, 
                                        step_SA = prob_SA.steps,
                                        numuser_SA = prob_SA.num_user_serve, throughput_SA = prob_SA.throughput_SA,
                                        time_SA = prob_SA.time)
                
                throughput_SA_1.append(prob_SA.throughput_SA)

                
            a = (np.var(throughput_SA_1, ddof = 0))/(np.average(throughput_SA_1))

            with open(file = "./eval.csv", mode = "a") as opf:
                writer = csv.writer(opf)
                writer.writerow([SimulatedAnnealing.RBAllocationSA.id,a])

            SimulatedAnnealing.RBAllocationSA.id += 1
        
main()
    
    