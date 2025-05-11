import numpy as np
import gym
from gym import spaces

class ResourceEnv(gym.Env):
    def __init__(self, K, I, B, H, P, RminK, Thrmin, BandW, N0):
        super().__init__()
        self.K = K
        self.I = I
        self.B = B
        self.H = H
        self.P = P
        self.RminK = RminK
        self.Thrmin = Thrmin
        self.BandW = BandW
        self.N0 = N0

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(len(I), len(K)), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([len(K), len(K), len(I)])

    def reset(self):
        self.r = np.random.dirichlet(np.ones(len(self.K)), size=len(self.I))
        return self.r.copy()

    def step(self, action):
        k_from, k_to, i = action
        delta = 0.05
        if self.r[i, k_from] >= delta:
            self.r[i, k_from] -= delta
            self.r[i, k_to] += delta

        throughput = self.compute_throughput()
        reward = self.compute_reward(throughput)
        done = False
        return self.r.copy(), reward, done, {}

    def compute_throughput(self):
        T = np.zeros(len(self.K))
        for k in self.K:
            for i in self.I:
                SINR = (self.H[i][k] * self.P[i]) / self.N0
                T[k] += self.BandW * self.r[i][k] * np.log2(1 + SINR)
        return T

    def compute_reward(self, T):
        ratio = T / self.RminK
        ratio = np.clip(ratio, 0, 2)
        return np.sum(ratio)
    
    def state_space_dim(self):
        return len(self.I) * len(self.K)  # Vì state là ma trận (I x K), flatten lại

    def action_space_dim(self):
        return 3  # Vì action là tuple (k_from, k_to, ru_index)

