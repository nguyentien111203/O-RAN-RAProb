import cvxpy_sol.ILPsolver as ILPsolver
import Input.input as input
from Greedyalgo.greedy import GreedyAllocation
import csv
import createtest
import ast
import torch
import numpy as np
import os
from stable_baselines3 import PPO  # Sử dụng mô hình đã được train
import os
import time
from Qlearn import env, MultiQ

def main():
    
    #for file in os.listdir("./Input/Input_data"): 
    # Mở CSV input_file với open
    input_file = "./Input/input_real.csv"
    # Đường dẫn tới file mô hình
    model_path = "./ppo_rb_allocation.zip"

    with open(input_file, mode = 'r') as csvfile:
        # Tạo một csv reader
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            numuser, numRU, B, P, RminK, Thrmin, BandW, N0 = row

            numuser = int(numuser)
            numRU = int(numRU)
            B = ast.literal_eval(B)
            P = ast.literal_eval(P)
            RminK = ast.literal_eval(RminK)
            Thrmin = float(Thrmin)
            BandW = float(BandW)
            N0 = float(N0)
            
            # Tạo đầu vào cho bài toán
            K, I, H  = input.createEnvironmentInput(numuser, numRU)

            """
            K, I, B, H, RminK, Pmax, Thrmin, BW, N0 = input.input_from_npz("./Input/Input_data/" + file)
            """
            # Giải bài toán với CVXPY
            
            prob = ILPsolver.AllocationProblemILP(K = K, I = I, H = H, B = B, P = P,
                                            RminK = RminK, Thrmin = Thrmin, BandW = BandW, N0 = N0)

            # RminK : Mbps, BW : MHz, N0 : mW/MHz, Pmax : mW
            prob.solve()
            # Đưa vào tensor
            start = time.time()
            # Khởi tạo môi trường và thuật toán
            envir = env.Environment(numuser, numRU, B, P, H, RminK, BandW, N0)
            q_learning = MultiQ.MultiAgentQLearning(envir, numuser, numRU)
            
            # Huấn luyện
            q_learning.train(max_episodes=10)

            end = time.time()  

            # Kết quả
            allocation = q_learning.get_allocation()
            print("Final Allocation_matrix:")
            print(allocation)
            envir.compute_throughput()
            print("Final Throughput (R_k):", envir.R_k, "Mbps")
            print("RminK:", RminK, "Mbps")

            throughput = envir.R_k
            
            num_served = sum(1 for k in K if throughput[k] >= RminK[k])

            createtest.write_data_test("./Output/output.csv", 0, numuser, numRU, B, 
                                prob.time, 
                                prob.throughput,
                                prob.num_user_serve, 
                                prob.check, 
                                numuser_SA = num_served, 
                                throughput_SA = sum(throughput[k] for k in K),
                                time_SA = end - start
                                )

    
        
main()
    