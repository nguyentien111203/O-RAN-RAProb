import numpy as np
from Qlearn.agent import RUAgent
from Qlearn.env import Environment
from matplotlib import pyplot as plt
import common
import pickle


from collections import defaultdict

class MultiAgentQLearning:
    def __init__(self, env, numuser, numRU, Q_table, alpha, gamma):
        self.env = env
        self.numuser = numuser
        self.numRU = numRU
        self.agents = [RUAgent(i, numuser, env.RminK, self.env.B[i], Q_table) for i in range(numRU)]
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

        with open("./Qlearn/Q_table.pkl", "wb") as f:    # Lưu bảng Q_table
            pickle.dump([agent.Q_table], f)


    def get_allocation(self):
        return self.env.Allocation_matrix
    
    def compute_total_reward(self):
        """Reward kết hợp: tối ưu throughput, thưởng khi có dư, phạt nếu thiếu."""
    
        self.env.compute_throughput()
        
        reward = (1 - common.tunning) * (sum(self.env.R_k)/self.env.Thrmin) + common.tunning * sum(self.env.served) \
            + common.lamda_penalty * (1-common.tunning) * (sum(self.env.R_k - self.env.RminK)*self.env.Thrmin)
        
        return reward
    
    """
        Sử dụng MultiQ hiện tại với mô trường để đưa ra hành động
        Input :
            env : Môi trường
            q_table : Q_table đã được train từ trước
            epsilon : Xác suất lựa chọn hành động ngẫu nhiên
            episode : Số episodes để các agent thử nghiệm
            steps_per_ru : Số hành động mà mỗi ru lựa chọn hành động
        Output :
            env.Allocation_matrix : Ma trận phân bổ từ từng RU tới các UE
    """
    def run_inference(self, env, q_table, epsilon, episode, steps_per_ru):
        for ep in range(episode):
            env.reset()
            self.agents = [RUAgent(i, env.numuser, env.RminK, env.B[i], q_table) for i in range(env.numRU)]

            for ru_idx in range(env.numRU):
                agent = self.agents[ru_idx]
                state = env.get_state(ru_idx)  

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

        return env.Allocation_matrix


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
        