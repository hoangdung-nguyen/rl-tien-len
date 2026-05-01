import os
import torch
import rlcard
import numpy as np
from rlcard.agents.dmc_agent import DMCTrainer
from rlcard.envs.registration import register
from rlcard.agents.dmc_agent.model import DMCModel
from rlcard.agents.dmc_agent.model import DMCAgent
from rlcard.utils import tournament
import math
import random

register(env_id='tien-len', entry_point='envs.tienlen_env:TienLenEnv')
env = rlcard.make('tien-len')


orig_init = DMCAgent.__init__
def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.use_raw = True
DMCAgent.__init__ = new_init

def train():
    env = rlcard.make('tien-len')
    trainer = DMCTrainer(
        env,
        cuda='0',
        xpid='rl-tien-len',
        savedir='models',
        num_actors=5,
    )
    trainer.start()

def load_model():
    env = rlcard.make('tien-len')

    model = DMCModel(env.state_shape, [52])
    checkpoint = torch.load('models/rl-tien-len/model.tar', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set to evaluation mode

    return TrainedAgent(model)


def evaluate(trained_agent):
    env = rlcard.make('tien-len')
    env.set_agents([trained_agent, NaiveTienLenAgent(), HeuristicSearchAgent(env.game.judger), TienLenRandomAgent()])
    results = tournament(env, 1000)
    print(f"Tournament Results (Avg Payoffs): {results}")


def play_test_game(trained_agent):
    env = rlcard.make('tien-len')
    env.set_agents([trained_agent, NaiveTienLenAgent(), HeuristicSearchAgent(env.game.judger), TienLenRandomAgent()])

    state, player_id = env.reset()
    print("--- Game Start ---")

    while not env.is_over():
        # Get action from the current player's agent
        current_agent = env.agents[player_id]
        action, _ = current_agent.eval_step(state)

        # Display the play
        move_desc = action if action else "PASS"
        print(f"Player {player_id} plays: {move_desc}")

        # Advance environment
        state, player_id = env.step(action)

    print("--- Game Over ---")
    print(f"Final Payoffs: {env.get_payoffs()}")


if not os.path.exists('models'): os.makedirs('models')
train()
agent = load_model()

evaluate(agent)

play_test_game(agent)