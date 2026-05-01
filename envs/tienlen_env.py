
import numpy as np
from rlcard.envs.env import Env
from game.tienlen_game import TienLenGame

class TienLenEnv(Env):
    def __init__(self, config):
        self.name = 'tien-len'
        self.game = TienLenGame()
        super().__init__(config)

        # State: 52 (hand) + 52 (top of stack) + 52 (history) + 3 (opponents' counts)
        self.state_shape = [[159] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

    def get_action_feature(self, action):
        feature = np.zeros(52, dtype=np.float32)
        if action:  # not pass
            for r, s in action:
                feature[(r - 3) * 4 + s] = 1
        return feature

    def _extract_state(self, state):
        """
        Turns the game state into numerical vectors for the NN.
        Returns:
            obs: The state vector
            legal_actions: A dictionary where keys are the card tuples
            action_features: A vector for each legal move
        """
        # 1. Build State Vector (159 features)
        obs = np.zeros(159, dtype=np.float32)

        # Bits 0-51: My Hand
        for r, s in state['hand']:
            obs[(r - 3) * 4 + s] = 1

        # Bits 52-103: Top of the Stack (what I must beat)
        if state['current_stack']:
            for r, s in state['current_stack'][-1]:
                obs[52 + (r - 3) * 4 + s] = 1

        # Bits 104-155: Discard Pile (History - Optional but helpful)
        # You can track this in TienLenGame to help the agent "count cards"
            for h in state['current_stack']:
                for r, s in h:
                    obs[104 + (r - 3) * 4 + s] = 1

        # Bits 156-158: Opponent Hand Sizes (Normalized)
        # Helps the agent know if someone is about to win
        other_counts = [len(p.hand) / 13.0 for i, p in enumerate(self.game.players) if i != state['player_id']]
        obs[156:159] = other_counts

        # 2. Build Action Features
        # Every legal move becomes a 52-bit vector
        legal_actions = state['legal_actions']  # This is a list of card tuples
        action_features = {}

        for move in legal_actions:
            feature = np.zeros(52, dtype=np.float32)
            for r, s in move:
                feature[(r - 3) * 4 + s] = 1
            action_features[move] = feature

        return {
            'obs': obs,
            'legal_actions': action_features,  # Move (tuple) -> 52-bit vector
            'raw_obs': state,
            'raw_legal_actions': legal_actions
        }

    def step(self, action):
        next_state, player_id = self.game.step(action)
        return self._extract_state(next_state), player_id

    def get_payoffs(self):
        payoffs = list(self.game.initial_payoffs)
        for idx, p in enumerate(self.game.players):
            if p.hand and any(card[0] == 15 for card in p.hand):
                payoffs[idx] -= 0.5
        return np.array(payoffs)
