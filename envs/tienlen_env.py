import numpy as np
from rlcard.envs.env import Env
from game.tienlen_game import TienLenGame

class TienLenEnv(Env):
    def __init__(self, config):
        self.name = 'tien-len'
        self.game = TienLenGame()
        super().__init__(config)
        # 52 bits for hand + 52 for table = 104
        self.state_shape = [[104] for _ in range(4)]
        self.action_shape = [None for _ in range(4)]

    def _extract_state(self, state):
        obs = np.zeros(104, dtype=int)
        for rank, suit in state['hand']:
            obs[(rank-3)*4 + suit] = 1
        
        # RLCard agents expect legal_actions to be a dict or list of ints
        legal_actions = state['legal_actions']
        
        return {
            'obs': obs,
            'legal_actions': {action: None for action in legal_actions}, # Dict for some agents
            'raw_legal_actions': legal_actions # For reference
        }

    def _decode_action(self, action_id):
        return action_id # Map back to game logic cards

    def get_payoffs(self):
        # Simplified: +1 for first player with empty hand
        return np.array([1.0 if not p.hand else -0.33 for p in self.game.players])
