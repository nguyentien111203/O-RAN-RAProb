import cvxpy_sol.ILPsolver as ILPsolver
import Input.input as input
from heuristic_with_DRL import SimulatedAnnealing
from Greedyalgo.greedy import GreedyAllocation
import csv
import createtest
import ast
import numpy as np
import os
def main():
    
    #for file in os.listdir("./Input/Input_data"): 
    # Mở CSV input_file với open
    input_file = "./Input/input_file.csv"
    with open(input_file, mode = 'r') as csvfile:
        # Tạo một csv reader
        reader = csv.reader(csvfile)
        next(reader)
        line = 7 #4
        for row in reader:
            line += 1
            numuser, numRU, RBeachRU, Pmax, RminK, Thrmin, BandW, N0, step_SA, Tmax, Tmin = row

            numuser = int(numuser)
            numRU = int(numRU)
            RBeachRU = ast.literal_eval(RBeachRU)
            Pmax = ast.literal_eval(Pmax)
            RminK = ast.literal_eval(RminK)
            PowerRB = ast.literal_eval(PowerRB)
            Thrmin = float(Thrmin)
            BandW = ast.literal_eval(BandW)
            N0 = float(N0)
            step_SA = int(step_SA)
            Tmax = int(Tmax)
            Tmin = int(Tmin)
            
            # Tạo đầu vào cho bài toán
            K, I, B, H  = input.createEnvironmentInput(numuser, numRU, RBeachRU)

            """
            K, I, B, H, RminK, Pmax, Thrmin, BW, N0 = input.input_from_npz("./Input/Input_data/" + file)
            """
            # Giải bài toán với CVXPY
            
            prob = ILPsolver.AllocationProblemILP(K = K, I = I, H = H, B = B, Pmax = Pmax,
                                            RminK = RminK, Thrmin = Thrmin, BandW = BandW, N0 = N0)

            # RminK : Mbps, BW : MHz, N0 : mW/MHz, Pmax : mW

            prob.solve()
            
            pro_greedy = GreedyAllocation(K = K, I = I, H = H, B = B, Pmax = Pmax, RminK = RminK,
                                        Thrmin = Thrmin, BandW = BandW, N0 = N0)

            pro_greedy.run()
            
            # Giải với SA
            for t in range(3):
                prob_SA = SimulatedAnnealing.RBAllocationSA(K = K, I = I, H = H, B = B, Pmax = Pmax,
                                                RminK = RminK, Thrmin = Thrmin, BandW = BandW, N0 = N0, step_SA = step_SA, Tmax = Tmax,
                                                Tmin = Tmin, test_id = t)
                prob_SA.run()
                prob_SA.draw_figures()

                np.savez_compressed(f"./Input/Input_data/input_data_{line}_{t}.npz", I = I, B = B, K = K, H = H, RminK = RminK, Pmax=Pmax, 
                                Thrmin = Thrmin, BandW = BandW, N0 = N0, allow_pickle = True)
            
                createtest.write_data_test("./Output/output.csv", t, numuser, numRU, RBeachRU, 
                                prob.time, 
                                prob.throughput,
                                prob.num_user_serve, 
                                prob.check, 
                                step_SA = prob_SA.steps,
                                numuser_SA = prob_SA.num_user_serve, 
                                throughput_SA = prob_SA.throughput_SA,
                                time_SA = prob_SA.time, 
                                throughput_Greedy = pro_greedy.throughput_Greedy, 
                                num_user_Greedy = pro_greedy.num_user_serve,
                                runtime_Greedy = pro_greedy.runtime)
                

            SimulatedAnnealing.RBAllocationSA.id += 1
    
        
main()
    
    