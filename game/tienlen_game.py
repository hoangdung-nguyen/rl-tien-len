from .dealer import TienLenDealer
from .judger import TienLenJudger
from .player import TienLenPlayer

class TienLenGame:
    def __init__(self):
        self.num_players = 4
        self.judger = TienLenJudger()
        self.lowest_card = (3, 0) # 3 of Spades (rank 3, suit 0)

        # RLCard Action Space Mapping
        #self.actions = self.judger._generate_all_possible_combinations()
        self.num_actions = 1442
        #self.action_to_id = {tuple(sorted(c)) if c else None: i for i, c in enumerate(self.actions)}
        #self.id_to_action = {i: c for i, c in enumerate(self.actions)}

    def init_game(self):
        self.dealer = TienLenDealer()
        self.players = [TienLenPlayer(i) for i in range(self.num_players)]
        self.dealer.deal_cards(self.players)

        self.current_stack = []
        self.players_in_play = [True] * self.num_players
        self.players_in_game = [True] * self.num_players
        self.state = "NONE"
        self.is_first_round = True
        self.last_move = None
        self.last_player = None
        self.initial_payoffs = [0] * self.num_players

        self.current_player = 0
        for i, p in enumerate(self.players):
            if self.lowest_card in p.hand:
                self.current_player = i
                break

        return self.get_state(self.current_player), self.current_player

    def step(self, action_cards):
        """ Logic to process an agent's move and update the game state. """
        player = self.players[self.current_player]
        if action_cards == ():
            self.players_in_play[self.current_player] = False
        else:
            self.state, _ = self.judger.get_type(action_cards)

            if self.state == "HANG":
                self._play_hang()

            self.current_stack.append(list(action_cards))

            for card in action_cards:
                player.hand.remove(card)

            self.is_first_round = False

            self.last_move = list(action_cards)
            self.last_player = self.current_player

            # Check if last player won
            if not player.hand:
                self.players_in_game[self.last_player] = False
                self.players_in_play[self.last_player] = False

                for idx, player in enumerate(self.players_in_game):
                    if not player: self.initial_payoffs[idx] += 1

        # Check if everyone passed except one person
        if sum(self.players_in_play) <= 1:
            # Round reset: the winner of the trick gets to start a new round
            self.state = "NONE"
            self.players_in_play = self.players_in_game.copy()
            self.current_player = self.last_player
        else:
            # Move to next player who hasn't passed yet
            self.current_player = (self.current_player + 1) % self.num_players

        while not self.players_in_play[self.current_player]:
            self.current_player = (self.current_player + 1) % self.num_players

        return self.get_state(self.current_player), self.current_player

    def get_state(self, player_id):
        player = self.players[player_id]

        legal_actions = self.judger.get_legal_actions(
            player.hand,
            self.last_move,
            self.state,
            self.is_first_round,
            self.lowest_card
        )

        return {
            'hand': player.hand,
            'current_stack': self.current_stack,
            'state': self.state,
            'legal_actions': legal_actions,
            'player_id': player_id
        }

    def _play_hang(self):
        if len(self.current_stack) < 2:
            return

        the_gotten = []
        temp_stack = list(self.current_stack)

        while temp_stack:
            move = temp_stack.pop()
            the_gotten.append(move)
            if len(move) <= 3:
                break

        multiplier = len(the_gotten)

        if len(the_gotten) >= 2:
            target_move = the_gotten[-1]
            card = target_move[0 if len(target_move)==1 else -1]
            rank, suit = card
            penalty = (suit + 1) * multiplier

            self.initial_payoffs[self.last_player] -= penalty
            self.initial_payoffs[self.current_player] += penalty

    def get_num_players(self):
        return self.num_players

    def get_num_actions(self):
        return self.num_actions

    def is_over(self):
        """Check if every player has run out of cards."""
        # return any(len(p.hand) == 0 for p in self.players)
        return sum(self.players_in_game) < 2

    def get_player_id(self):
        """ Return the current player's ID (0, 1, 2, or 3) """
        return self.current_player
