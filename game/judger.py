class TienLenJudger:
    @staticmethod
    def get_type(cards):
        """Returns type of combination (SINGLE, PAIR, SEQUENCE, etc.)"""
        cards = sorted(cards)
        n = len(cards)
        if n == 1: return "SINGLE"
        if n == 2 and cards[0][0] == cards[1][0]: return "PAIR"
        # Simplified: add logic for sequences and bombs here
        return "INVALID"

    def get_legal_actions(self, hand, last_move):
        # Map (rank, suit) to an ID between 1 and 52
        # Example mapping: (rank-3)*4 + suit + 1
        legal_actions = [0] # 0 represents 'pass'
        for rank, suit in hand:
            action_id = (rank - 3) * 4 + suit + 1
            legal_actions.append(action_id)
        return legal_actions
