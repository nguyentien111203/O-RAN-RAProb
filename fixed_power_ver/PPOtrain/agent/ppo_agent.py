import torch
import numpy as np
from model.actor import Actor
from model.critic import Critic
from model.memory import Memory
from torch.distributions import Categorical 

class PPOAgent:
    def __init__(self, state_dim, action_dim, numuser, numRU, hidden_dim=128, lr=3e-4, gamma=0.99):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)
        self.memory = Memory()
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.numuser = numuser
        self.numRU = numRU

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state)

        dist = Categorical(probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)

        # Giải mã index thành (k_from, k_to, i)
        k_from = action_idx.item() // (self.numuser * self.numRU)
        k_to   = (action_idx.item() // self.numRU) % self.numuser
        i      = action_idx.item() % self.numRU

        action = (k_from, k_to, i)

        # Nếu k_from == k_to, thì chọn lại (trong bản cứng có thể bỏ qua luôn các action kiểu này)
        if k_from == k_to:
            return self.select_action(state.squeeze(0))  # Đệ quy chọn lại (có thể tối ưu sau)

        # Lưu lại vào memory
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.logprobs.append(log_prob)

        return action

    def update(self):
        states = torch.FloatTensor(self.memory.states)
        actions = torch.LongTensor([a[1] for a in self.memory.actions])
        rewards = torch.FloatTensor(self.memory.rewards)

        values = self.critic(states).squeeze()
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        advantage = returns - values
        probs = self.actor(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        loss_actor = -(log_probs * advantage.detach()).mean()

        loss_critic = advantage.pow(2).mean()

        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        self.memory.clear()

    def save(self, path):
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
