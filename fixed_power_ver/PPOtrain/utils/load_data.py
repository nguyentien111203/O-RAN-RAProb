import pandas as pd
import numpy as np
import ast
from data.input import createEnvironmentInput

def load_env_params_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    env_params_list = []

    for _, row in df.iterrows():
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
        K, I, H  = createEnvironmentInput(numuser, numRU)

        env_params = {
            'K': K,
            'I': I,
            'B': B,
            'H': H,
            'P': P,
            'RminK': RminK,
            'Thrmin': Thrmin,
            'BandW': BandW,
            'N0': N0
        }

        print(f"H shape: {np.array(H, dtype=object).shape}")
        print(f"P: {P}, type: {[type(p) for p in P]}")
        print(f"RminK: {RminK}, type: {[type(r) for r in RminK]}")

        env_params_list.append(env_params)

    return env_params_list
