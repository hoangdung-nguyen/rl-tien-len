import itertools
from collections import defaultdict

class TienLenJudger:
    def __init__(self):
        pass

    def _generate_all_possible_combinations(self):
        # 3-10 are 3-10, J:11, Q:12, K:13, A:14, 2:15
        all_cards = [(r, s) for r in range(3, 16) for s in range(4)]

        all_combos = self._get_all_types(all_cards)

        unique_actions = [None] # ID 0 is 'Pass'
        seen = set()
        for combo in all_combos:
            c_tuple = tuple(sorted(combo))
            if c_tuple not in seen:
                unique_actions.append(c_tuple)
                seen.add(c_tuple)

        return unique_actions

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

    def get_legal_actions(self, player, last_move, state, action_to_id, is_first, lowest):
        """
        Args:
            player (TienLenPlayer): The player object whose turn it is
            last_move (list): The list of cards currently on the table (None if new round)
            action_to_id (dict): The mapping to convert combinations to IDs

        Returns:
            legal_actions (list): A list of integer IDs (e.g., [0, 15, 42])
        """
        valid_combos = []
        hand = sorted(player.hand)

        if is_first:
            # Java: getMovesThatContains(lowest)
            valid_combos = self._get_combos_containing(hand, lowest)
        elif state == "none":
            # Java: getMoves(hand, moves) -> All types
            valid_combos = self._get_all_types(hand)
        elif state == "same":
            # Java: getSames(hand, moves, peek)
            valid_combos = self._get_sames(hand, last_move)
        elif state == "run":
            # Java: getRuns(hand, moves, peek)
            valid_combos = self._get_runs(hand, last_move)
        elif state == "pig":
            # Java: getPigs(hand, moves, peek)
            valid_combos = self._get_pigs(hand, last_move)
        elif state == "hang":
            # Java: getHang(hand, moves, peek)
            valid_combos = self._get_hangs(hand, last_move)

        # Map to IDs
        legal_ids = [action_to_id[tuple(sorted(c))] for c in valid_combos if tuple(sorted(c)) in action_to_id]

        # Add 'Pass' (0) if not a new round
        if state != "none" and not is_first:
            legal_ids.append(0)

        return list(set(legal_ids))

    @staticmethod
    def _get_rank_groups(hand):
        groups = defaultdict(list)
        for r, s in hand:
            groups[r].append((r, s))
        return groups

    def _get_sames(self, hand, peek):
        """ Java: getSames - Find Pairs/Triples/Quads of same length but higher power """
        n, peek_power = len(peek), peek[-1]
        valid, groups = [], self._get_rank_groups(hand)
        for r, cards in groups.items():
            if len(cards) >= n:
                for combo in itertools.combinations(cards, n):
                    if combo[-1] > peek_power:
                        valid.append(list(combo))
        return valid

    def _get_all_sames(self, hand):
        """ Helper for building the global action space: returns all singles, pairs, etc. """
        valid = []
        groups = self._get_rank_groups(hand)
        for r, cards in groups.items():
            for n in range(1, 5): # 1=Single, 2=Pair, 3=Triple, 4=Quad
                for combo in itertools.combinations(cards, n):
                    valid.append(list(combo))
        return valid

    def _get_runs(self, hand, peek=None):
        """ Find sequences where length >= peek length """
        target_len = len(peek) if peek else 3
        peek_power = peek[-1] if peek else (0, 0)
        valid, groups = [], self._get_rank_groups(hand)

        available_ranks = sorted([r for r in groups.keys() if r < 15]) # 2s (15) excluded
        # Loop through all possible lengths starting from target_len
        for length in range(target_len, 13):
            for i in range(len(available_ranks) - length + 1):
                subset = available_ranks[i:i+length]
                if all(subset[j] == subset[0] + j for j in range(length)):
                    # Get card variations for this rank sequence
                    for combo in itertools.product(*[groups[r] for r in subset]):
                        if length > target_len or combo[-1] > peek_power:
                            valid.append(list(combo))
        return valid

    def _get_pigs(self, hand, peek):
        """ Higher 2s or 'Cutting' moves (Bombs/Hangs) """
        valid = self._get_sames(hand, peek) # Higher 2s
        groups = self._get_rank_groups(hand)
        # Four of a kind cuts any 2
        for r, cards in groups.items():
            if len(cards) == 4: valid.append(cards)
        # 3+ Consecutive pairs cuts single 2s
        if len(peek) == 1:
            valid.extend(self._get_hangs(hand, None))
        return valid

    def _get_hangs(self, hand, peek=None):
        """ Consecutive pairs (Pairs-of-sequences) """
        target_pairs = len(peek) // 2 if peek else 3
        peek_power = peek[-1] if peek else (0, 0)
        valid, groups = [], self._get_rank_groups(hand)
        pair_ranks = sorted([r for r, cards in groups.items() if len(cards) >= 2 and r < 15])

        for length in range(target_pairs, 7):
            for i in range(len(pair_ranks) - length + 1):
                subset = pair_ranks[i:i+length]
                if all(subset[j] == subset[0] + j for j in range(length)):
                    rank_pairs = [list(itertools.combinations(groups[r], 2)) for r in subset]
                    for combo_pairs in itertools.product(*rank_pairs):
                        combo = [card for pair in combo_pairs for card in pair]
                        if length > target_pairs or combo[-1] > peek_power:
                            valid.append(combo)
        return valid

    def _get_all_types(self, hand):
        """ All valid moves for a new trick """
        all_combos = []

        all_combos.extend(self._get_all_sames(hand))

        all_combos.extend(self._get_runs(hand))

        all_combos.extend(self._get_hangs(hand))
        return all_combos

    def _get_combos_containing(self, hand, card):
        """ Force play of 3s on first turn """
        return [m for m in self._get_all_types(hand) if card in m]

    def can_beat(self, move, target_move):
        """
        move: list of (rank, suit) chosen by agent
        target_move: list of (rank, suit) currently on top of the stack
        """
        m_type, m_power = self.get_type(move)
        t_type, t_power = self.get_type(target_move)

        if m_type == "INVALID": return False
        
        if m_type == t_type and len(move) == len(target_move):
            return m_power > t_power

        if (m_type == "FOUR_OF_A_KIND" or (m_type == "HANG" and len(move)>6)) and t_type == "PAIR" and t_power[0] == 15:
            return True
        if (m_type == "FOUR_OF_A_KIND" or m_type == "HANG") and t_type == "SINGLE" and t_power[0] == 15:
            return True
            
        return False
