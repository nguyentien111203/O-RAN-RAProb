import os
import torch
import numpy as np
from tqdm import tqdm
from agent.ppo_agent import PPOAgent
from env.environment import ResourceEnv
from model.memory import Memory


def train_model(env_params_list, total_epochs=1000, save_path='checkpoints/'):
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize agent
    sample_env = ResourceEnv(**env_params_list[0])
    agent = PPOAgent(
        state_dim=sample_env.state_space_dim(),
        action_dim=sample_env.action_space_dim(),
        numRU = len(sample_env.I),
        numuser = len(sample_env.K)
    )

    for idx, env_params in enumerate(env_params_list):
        print(f"\nTraining on Environment #{idx+1}")
        env = ResourceEnv(**env_params)
        memory = Memory()

        for epoch in tqdm(range(total_epochs)):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                memory.rewards.append(reward)
                memory.dones.append(done)
                state = next_state
                total_reward += reward

            agent.update(memory)
            memory.clear()

            if (epoch + 1) % 100 == 0:
                print(f"Env {idx+1} | Epoch {epoch+1} | Total Reward: {total_reward:.4f}")

        # Save fine-tuned model after each environment
        torch.save(agent.actor.state_dict(), os.path.join(save_path, f"actor_env{idx+1}.pth"))
        torch.save(agent.critic.state_dict(), os.path.join(save_path, f"critic_env{idx+1}.pth"))

    print("Training and fine-tuning completed.")
