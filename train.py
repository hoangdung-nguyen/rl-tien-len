import os
import torch
import rlcard
from rlcard.agents import DMCAgent
from rlcard.envs.registration import register
from envs.tienlen_env import TienLenEnv

# 1. Register and Make Env
register(env_id='tien-len-v1', entry_point=TienLenEnv)
env = rlcard.make('tien-len-v1')

# 2. Initialize DMC Agents
agents = []
for i in range(env.num_players):
    agent = DMCAgent(
        state_shape=env.state_shape[i],
        action_shape=[52],  # The 52-bit bitmap feature vector
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    agents.append(agent)

env.set_agents(agents)


def train(num_episodes=10000):
    print("Starting Training...")
    for episode in range(num_episodes):
        # env.run() collects a full game trajectory using self-play
        trajectories, payoffs = env.run(is_training=True)

        # Feed the data into the agents for learning
        for i in range(env.num_players):
            for ts in trajectories[i]:
                # DMC agents update their internal networks here
                agents[i].feed(ts, payoffs[i])

        if episode % 100 == 0:
            print(f"Episode {episode} completed. Winner Payoff: {max(payoffs)}")
            # Save the first agent as our "Main" model
            agents[0].save(os.path.join('models', f'tienlen_model_{episode}.pth'))


if __name__ == "__main__":
    if not os.path.exists('models'): os.makedirs('models')
    train()
