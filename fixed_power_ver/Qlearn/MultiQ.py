import numpy as np
from itertools import product
from collections import defaultdict
from Qlearn.agent import RUAgent
from Qlearn.env import Environment
from matplotlib import pyplot as plt
import common


from collections import defaultdict

class MultiAgentQLearning:
    def __init__(self, env, numuser, numRU, alpha, gamma):
        self.env = env
        self.numuser = numuser
        self.numRU = numRU
        self.Q_table = defaultdict(lambda: {(uf, ut): 0 for uf in range(numuser) for ut in range(numuser) if uf != ut})
        self.Q_table[(-1,-1)] = 0
        self.agents = [RUAgent(i, numuser, env.RminK, self.env.B[i], self.Q_table) for i in range(numRU)]
        self.rewards = []
        self.moving_avgs = []
        self.alpha = alpha
        self.gamma = gamma

    def train(self, max_episodes=500, steps_per_ru=3):
        epsilon = 1.0
        epsilon_min = 0.05
        decay_rate = 0.005

        for episode in range(max_episodes):
            self.env.reset()
            total_reward = 0

            for ru_idx in range(self.numRU):
                agent = self.agents[ru_idx]
                state = self.env.get_state(ru_idx)

                for _ in range(steps_per_ru):
                    valid = False

                    action = agent.choose_action(state, epsilon)
                    next_state, done, valid = self.env.step(ru_idx, action)
                    while not valid:
                        action = agent.choose_action(state, epsilon)
                        next_state, done, valid = self.env.step(ru_idx, action)
                        
                    reward = self.compute_total_reward()
                    agent.update_q_table(state, action, reward, next_state, self.alpha, self.gamma)
                    state = next_state
                    total_reward = reward  # cộng dần reward thay vì ghi đè  
            epsilon = max(epsilon_min, epsilon - decay_rate)
            self.rewards.append(total_reward)
            print(f"Episode {episode}: reward = {total_reward}")

        rewards = np.array(self.rewards)
        self.moving_avgs = np.convolve(rewards, np.ones(10)/10, mode='valid')


    def get_allocation(self):
        return self.env.Allocation_matrix
    
    def compute_total_reward(self):
        """Reward kết hợp: tối ưu throughput, thưởng khi có dư, phạt nếu thiếu."""
    
        self.env.compute_throughput()
        
        reward = (1 - common.tunning) * sum(self.env.R_k) + common.tunning * sum(self.env.served) \
            + common.lamda_penalty * (1-common.tunning) * sum(self.env.R_k - self.env.RminK)
        
        return reward


    def draw_figure(self, window=10):
        plt.figure(figsize=(10, 5))
        rewards = np.array(self.rewards)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(rewards, alpha=0.3, label='Reward')
        plt.plot(moving_avg, color='blue', label = 'Average reward')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.title("Q-learning Progress")
        plt.tight_layout()
        plt.savefig(f"./Picture/Qlearning_process_{self.numuser}_{self.alpha}_{self.gamma}.png")
        plt.show()
        