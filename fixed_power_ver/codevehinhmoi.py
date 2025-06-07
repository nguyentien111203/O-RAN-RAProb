import matplotlib.pyplot as plt
import torch 


def draw_figure(rewards, numuser, varying, character):
    plt.title("QL-Training")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.plot(rewards[0], color='silver', label = f"{character} = {varying[0]}")  # plot cumulative reward
    plt.plot(rewards[1], color='green', label = f"{character} = {varying[1]}")  # plot cumulative reward
    plt.plot(rewards[2], color='blue', label = f"{character} = {varying[2]}")  # plot cumulative reward
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.ylim(bottom = 0, top = numuser + 2)
    plt.legend()
    plt.grid(True)
    plt.title("Q-learning Progress")
    plt.tight_layout()
    plt.savefig(f"./Picture/Qlearning_process_varying{character}_{numuser}_1.png")
    plt.show()