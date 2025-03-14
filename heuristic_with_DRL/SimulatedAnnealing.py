import numpy as np
from simanneal import Annealer

class RBAllocationSA(Annealer):
    def __init__(self, K, I, H, B, P, Pmax, RminK, Tmin, BW, N0):
        self.K = K  # Tập người dùng
        self.I = I  # Tập RU
        self.H = H  # Ma trận channel vector
        self.B = B  # Tập số RB của từng RU
        self.P = P
        self.Pmax = Pmax  # Công suất tối đa của từng RU
        self.RminK = RminK  # Data rate tối thiểu của từng user
        self.Tmin = Tmin  # Throughput tối thiểu
        self.BW = BW  # Băng thông của từng RB
        self.N0 = N0  # Mật độ công suất tạp âm

        state = self.initialize_state()
        super().__init__(state)

    def initialize_state(self):
        allocation = {i: np.zeros((len(self.B[i]), len(self.K))) for i in self.I}
        power = {i: np.zeros((len(self.B[i]), len(self.K))) for i in self.I}
        
        for k in self.K:
            i = np.random.choice(self.I)
            b = np.random.randint(0, len(self.B[i]))
            allocation[i][b][k] = 1
            power[i][b][k] = np.random.uniform(0, self.Pmax[i])
        
        return allocation, power

    def compute_SINR(self, allocation, power):
        SINR = {k: 0 for k in self.K}
        for k in self.K:
            signal = sum(
                power[i][b][k] * self.H[k][i][b]
                for i in self.I
                for b in range(len(self.B[i]))
                if allocation[i][b][k] == 1
            )
            SINR[k] = signal / (self.N0 * self.BW)
        return SINR

    def compute_throughput(self, SINR):
        return {k: self.BW * np.log2(1 + SINR[k]) for k in self.K}

    def energy(self):
        allocation, power = self.state
        SINR = self.compute_SINR(allocation, power)
        throughput = self.compute_throughput(SINR)
        cost = -sum(np.log(1 + SINR[k]) for k in self.K)
        penalty = sum(max(0, self.RminK[k] - throughput[k]) for k in self.K)
        return cost + penalty

    def move(self):
        allocation, power = self.state
        i = np.random.choice(self.I)
        b = np.random.randint(0, len(self.B[i]))
        k = np.random.choice(self.K)
        
        allocation[i][:, k] = 0
        new_b = np.random.randint(0, len(self.B[i]))
        allocation[i][new_b][k] = 1

        delta_power = np.random.uniform(-0.1, 0.1) * self.Pmax[i]
        power[i][new_b][k] = max(0, min(self.Pmax[i], power[i][new_b][k] + delta_power))
        
        self.state = allocation, power

    def run(self):
        self.copy_strategy = "deepcopy"
        best_state, best_energy = self.anneal()
        return best_state, best_energy
