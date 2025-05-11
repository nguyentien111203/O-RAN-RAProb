import numpy as np
from itertools import product
import uuid
from collections import defaultdict
from agent import RUAgent
from env import Environment


class MultiAgentQLearning:
    def __init__(self, env, numuser, numRU):
        self.env = env  # Môi trường
        self.numuser = numuser  # Số lượng người dùng
        self.numRU = numRU  # Số lượng RU
        self.Q_table = defaultdict(lambda: {a: 0 for a in range(numuser)})  # Q_table chung
        self.agents = [RUAgent(i, numuser, env.RminK, self.Q_table) for i in range(numRU)] # Các agent ở mỗi RU

    def train(self, max_episodes=100, alpha=0.1, gamma=0.9, epsilon=0.5):
        for episode in range(max_episodes):
            state = self.env.reset()
            for ru_idx in range(self.numRU):
                # Tạo ra agent ở mỗi RU, đưa ra state, và cho ra hành động
                agent = self.agents[ru_idx]
                state = self.env.get_state(ru_idx)
                action = agent.choose_action(state, epsilon)
                next_state, reward, done = self.env.step(ru_idx, action)
                agent.update_q_table(state, action, reward, next_state, alpha, gamma)
            #if episode % 10 == 0:
            #    print(f"Episode {episode}, Allocation_matrix:\n{self.env.Allocation_matrix}")
            #    self.env.compute_throughput()
            #    print(f"Throughput (R_k): {self.env.R_k} Mbps, RminK: {self.env.RminK} Mbps")

    def get_allocation(self):
        """Trả về ma trận phân bổ cuối cùng"""
        return self.env.Allocation_matrix