import gym
import numpy as np

class WrappedRBEnv(gym.Wrapper):
    def __init__(self, env, max_K, max_obs_len):
        super(WrappedRBEnv, self).__init__(env)
        self.max_K = max_K
        self.max_obs_len = max_obs_len

        # Đảm bảo observation_space đúng với kích thước mô hình
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(max_obs_len,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([max_K] * env._num_rb())

    def reset(self):
        obs = self.env.reset()
        return self._pad_obs(obs)

    def step(self, action):
        # Map action về trong range thực tế của env
        mapped_action = [a % len(self.env.K) for a in action]
        obs, reward, done, info = self.env.step(mapped_action)
        return self._pad_obs(obs), reward, done, info

    def _pad_obs(self, obs):
        obs = np.array(obs, dtype=np.float32)
        if obs.shape[0] < self.max_obs_len:
            obs = np.pad(obs, (0, self.max_obs_len - obs.shape[0]))
        elif obs.shape[0] > self.max_obs_len:
            obs = obs[:self.max_obs_len]
        return obs

