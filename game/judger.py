import itertools
from collections import defaultdict

class TienLenJudger:
    def __init__(self):
        pass

    def _generate_all_possible_combinations(self):
        # 3-10 are 3-10, J:11, Q:12, K:13, A:14, 2:15
        all_cards = [(r, s) for r in range(3, 16) for s in range(4)]

        all_combos = self._get_all_types(all_cards)

        unique_actions = [()]
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
        """ Checks for 'Pairs-of-Sequences' (Double Runs). """
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
        """ Returns a string type and the 'power rank' (highest card in combo) """
        if not cards: return "NONE", 0
        cards = sorted(cards, key=lambda x: (x[0], x[1])) # Sort by rank then suit
        power_rank = cards[-1] # Highest card defines the strength

        if self.is_same_rank(cards): return "SAME", power_rank
        if self.is_run(cards): return "RUN", power_rank
        if power_rank[0] == 15: return "PIG", power_rank
        if self.is_hang(cards): return "HANG", power_rank # Pairs-of-sequences
        
        return "INVALID", 0

    def get_legal_actions(self, hand, last_move, state, is_first, lowest):
        """
        Args:
            hand: The hand from which the move must be built
            last_move (list): The list of cards currently on the table
        Returns:
            legal_actions (list): A list of tuples (e.g., [(3, 0), (4, 3), (5, 1)])
        """
        valid_combos = []

        if is_first:
            valid_combos = self._get_combos_containing(hand, lowest)
        elif state == "NONE":
            valid_combos = self._get_all_types(hand)
        elif state == "SAME":
            valid_combos = self._get_sames(hand, last_move)
        elif state == "RUN":
            valid_combos = self._get_runs(hand, last_move)
        elif state == "PIG":
            valid_combos = self._get_pigs(hand, last_move)
        elif state == "HANG":
            valid_combos = self._get_hangs(hand, last_move)

        legal_actions = [tuple(sorted(c)) for c in valid_combos]

        # Add 'Pass' (empty tuple) if not a new round
        if state != "NONE" and not is_first:
            legal_actions.append(())

        return list(set(legal_actions))

    @staticmethod
    def _get_rank_groups(hand):
        groups = defaultdict(list)
        for r, s in hand:
            groups[r].append((r, s))
        return groups

    def _get_sames(self, hand, peek):
        """ Find Pairs/Triples/Quads of same length but higher power """
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
        """ Finds runs of the same length but higher power """
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
                        if length == target_len and combo[-1] > peek_power:
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