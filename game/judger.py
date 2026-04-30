import itertools

class TienLenJudger:
    def __init__(self):
        self.action_space = self._generate_all_possible_combinations()
        pass

    def _generate_all_possible_combinations(self):
        # 3-10 are 3-10, J:11, Q:12, K:13, A:14, 2:15
        ranks = list(range(3, 16))
        suits = list(range(4))
        all_cards = [(r, s) for r in ranks for s in suits]
        return self.get_all_combinations(all_cards)

    def get_all_combinations(self, hand):
        """
        Generates every valid combination (Single, Pair, Triple, Run, Hang) from a hand.
        hand: list of (rank, suit)
        """
        combinations = []
        rank_groups = {}
        for r, s in hand:
            rank_groups.setdefault(r, []).append((r, s))
            
        # 1. Singles
        combinations.extend([[card] for card in hand])
        
        # 2. Pairs, Triples, Quads
        for r, cards in rank_groups.items():
            for n in range(2, 5):
                if len(cards) >= n:
                    combinations.extend([list(c) for c in itertools.combinations(cards, n)])

        # 3. Runs (Sequences) - Minimum 3, '2' (rank 15) excluded
        unique_ranks = sorted([r for r in rank_groups.keys() if r < 15])
        for length in range(3, len(unique_ranks) + 1):
            for i in range(len(unique_ranks) - length + 1):
                subset = unique_ranks[i:i+length]
                if all(subset[j] == subset[0] + j for j in range(length)):
                    # Get all possible card variations for this rank sequence
                    rank_cards = [rank_groups[r] for r in subset]
                    for combo in itertools.product(*rank_cards):
                        combinations.append(list(combo))

        # 4. Hangs (3+ consecutive pairs)
        pair_ranks = sorted([r for r, cards in rank_groups.items() if len(cards) >= 2 and r < 15])
        for length in range(3, len(pair_ranks) + 1):
            for i in range(len(pair_ranks) - length + 1):
                subset = pair_ranks[i:i+length]
                if all(subset[j] == subset[0] + j for j in range(length)):
                    rank_pairs = [list(itertools.combinations(rank_groups[r], 2)) for r in subset]
                    for combo_pairs in itertools.product(*rank_pairs):
                        combinations.append([card for pair in combo_pairs for card in pair])
                        
        return combinations
    
    @staticmethod
    def is_same_rank(cards):
        if not cards: return False
        return all(c[0] == cards[0][0] for c in cards)

    @staticmethod
    def is_run(cards):
        """ Checks for a sequence (3-4-5...). In Tien Len, '2' cannot be in a sequence. """
        if len(cards) < 3: return False
        ranks = sorted([c[0] for c in cards])
        if 15 in ranks: return False # '2' is rank 15
        # Check if they are consecutive: [3, 4, 5] -> [0, 0, 0] after subtraction
        return all(ranks[i] == ranks[0] + i for i in range(len(ranks)))

    @staticmethod
    def is_hang(cards):
        """ 
        Checks for 'Pairs-of-Sequences' (Double Runs).
        Matches Java logic for pairsOf.size() >= 3 and pairsOf.get(rank) == 2
        """
        if len(cards) < 6 or len(cards) % 2 != 0: return False
        ranks = sorted([c[0] for c in cards])
        # Ensure every rank has exactly a pair
        for i in range(0, len(ranks), 2):
            if ranks[i] != ranks[i+1]: return False
        
        # Ensure the unique ranks are a sequence
        unique_ranks = sorted(list(set(ranks)))
        if 15 in unique_ranks: return False
        return all(unique_ranks[i] == unique_ranks[0] + i for i in range(len(unique_ranks)))

    @staticmethod
    def get_type(self, cards):
        """ 
        Matches your Java 'getMoveState'.
        Returns a string type and the 'power rank' (highest card in combo)
        """
        if not cards: return "NONE", 0
        cards = sorted(cards, key=lambda x: (x[0], x[1])) # Sort by rank then suit
        power_rank = cards[-1] # Highest card defines the strength
        
        n = len(cards)
        if n == 1: return "SINGLE", power_rank
        if self.is_same_rank(cards):
            if n == 2: return "PAIR", power_rank
            if n == 3: return "TRIPLE", power_rank
            if n == 4: return "FOUR_OF_A_KIND", power_rank # A "Bomb"
        if self.is_run(cards): return "RUN", power_rank
        if self.is_hang(cards): return "HANG", power_rank # Pairs-of-sequences
        
        return "INVALID", 0


    def get_legal_actions(self, player, last_move, action_to_id):
        """
        Args:
            player (TienLenPlayer): The player object whose turn it is
            last_move (list): The list of cards currently on the table (None if new round)
            action_to_id (dict): The mapping to convert combinations to IDs
            
        Returns:
            legal_actions (list): A list of integer IDs (e.g., [0, 15, 42])
        """
        legal_actions = [0] if last_move is not None else []
        
        all_hand_combos = self.find_all_combos_in_hand(player.hand)
        
        for combo in all_hand_combos:
            if last_move is None:
                combo_id = action_to_id[tuple(sorted(combo))]
                legal_actions.append(combo_id)
            else:
                if self.can_beat(combo, last_move):
                    combo_id = action_to_id[tuple(sorted(combo))]
                    legal_actions.append(combo_id)
                    
        return legal_actions


    def can_beat(self, move, target_move):
        """
        move: list of (rank, suit) chosen by agent
        target_move: list of (rank, suit) currently on top of the stack
        """
        m_type, m_power = self.get_combination_type(move)
        t_type, t_power = self.get_combination_type(target_move)

        if m_type == "INVALID": return False
        
        if m_type == t_type and len(move) == len(target_move):
            return m_power > t_power

        if (m_type == "FOUR_OF_A_KIND" or (m_type == "HANG" and len(move)>6)) and t_type == "DOUBLE" and t_power[0] == 15:
            return True
        if (m_type == "FOUR_OF_A_KIND" or m_type == "HANG") and t_type == "SINGLE" and t_power[0] == 15:
            return True
            
        return False
