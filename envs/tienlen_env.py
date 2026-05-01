
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

    def _get_one_action_feature(self, move, current_hand, current_state):
        """
        Creates a 67-dim feature vector for a specific move.
        67 dims = 52 (cards) + 8 (types) + 12 (heuristic/future metrics)
        """
        feature = np.zeros(67, dtype=np.float32)
        m_type = "NONE"
        # 1. Card Bitmap (0-51) & Pass Bit (58)
        if move == ():
            feature[58] = 1  # Pass bit
            future_hand = current_hand  # Hand doesn't change
        else:
            for r, s in move:
                feature[(r - 3) * 4 + s] = 1

            # 2. Type Bits (52-57)
            m_type, m_power = self.game.judger.get_type(move)
            n = len(move)

            # 1. Map the 'SAME' type into specific bits based on count
            if m_type == "SAME":
                if n == 1:
                    # If rank is 15, it's a PIG, else it's a SINGLE
                    feature[52 if m_power[0] != 15 else 59] = 1
                elif n == 2:
                    feature[53] = 1  # PAIR
                elif n == 3:
                    feature[54] = 1  # TRIPLE
                elif n == 4:
                    feature[55] = 1  # FOUR_OF_A_KIND / BOMB

            # 2. Map the others
            elif m_type == "RUN":
                feature[56] = 1
            elif m_type == "HANG":
                feature[57] = 1
            elif m_type == "PIG":
                feature[59] = 1  # Explicit Pig Bit

            future_hand = list(current_hand)
            for card in move:
                if card in future_hand:
                    future_hand.remove(card)

        # 3. HEURISTIC METRICS (59-71)
        future_moves = self.game.judger._get_all_types(future_hand)

        # 60. Future Mobility: Normalized via x / (x + 20)
        m_count = len(future_moves)
        feature[60] = m_count / (m_count + 20.0)

        # 61. Progress: Normalized 0 to 1
        feature[61] = (13 - len(future_hand)) / 13.0

        # 62. Hand Strength: Normalized (Max rank is 15)
        if future_hand:
            avg_rank = sum(c[0] for c in future_hand) / len(future_hand)
            feature[62] = avg_rank / 15.0

        # 63. Same-State Response: Normalized via x / (x + 5)
        if move:
            res_moves = self.game.judger.get_legal_actions(future_hand, move, m_type, False, (3, 0))
            res_count = len(res_moves)
            feature[63] = res_count / (res_count + 5.0)

        # 64. Cutting Action: Binary (0 or 1)
        is_bomb = (n == 4 and self.game.judger.is_same_rank(move)) or m_type == "HANG"
        if is_bomb and current_state in ["PIG", "SAME"]:
            feature[64] = 1.0

        # 65. Stack Multiplier: Normalized (Assuming max 5-layer cut chain)
        feature[65] = min(len(self.game.current_stack) / 5.0, 1.0)

        # 66. Mobility Quality: Normalized (Max size is usually 4 or 5)
        if future_moves:
            avg_size = sum(len(m) for m in future_moves) / len(future_moves)
            feature[66] = min(avg_size / 4.0, 1.0)

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
            action_features[move] = self._get_one_action_feature(move, state['hand'], state['state'])

        return {
            'obs': obs,
            'legal_actions': action_features,
            'raw_obs': state,
            'raw_legal_actions': legal_actions
        }

    def step(self, action, raw_action=False):
        next_state, player_id = self.game.step(action)
        return self._extract_state(next_state), player_id

    def get_payoffs(self):
        payoffs = list(self.game.initial_payoffs)
        for idx, p in enumerate(self.game.players):
            if p.hand and any(card[0] == 15 for card in p.hand):
                payoffs[idx] -= 0.5
        return np.array(payoffs)
