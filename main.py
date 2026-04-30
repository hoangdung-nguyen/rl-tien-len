import rlcard
from rlcard.envs.registration import register
from envs.tienlen_env import TienLenEnv
from rlcard.agents import RandomAgent

# 1. Register
register(env_id='tien-len-v1', entry_point=TienLenEnv)

# 2. Setup
env = rlcard.make('tien-len-v1')
agents = [RandomAgent(num_actions=env.num_actions) for _ in range(4)]
env.set_agents(agents)

# 3. Test Run
print("Starting match...")
trajectories, payoffs = env.run()
print("Final Payoffs:", payoffs)
