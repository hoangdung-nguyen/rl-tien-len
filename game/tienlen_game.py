from .dealer import TienLenDealer
from .judger import TienLenJudger

class TienLenGame:

    def __init__(self):
        self.num_players = 4
        self.judger = TienLenJudger()
        self.lowest_card = (3, 0) # 3 of Spades (rank 3, suit 0)

        # RLCard Action Space Mapping
        self.actions = self.judger._generate_all_possible_combinations()
        self.num_actions = len(self.actions)
        self.action_to_id = {tuple(sorted(c)) if c else None: i for i, c in enumerate(self.actions)}
        self.id_to_action = {i: c for i, c in enumerate(self.actions)}

    def get_num_players(self):
        return self.num_players

    def get_num_actions(self):
        return self.num_actions

    def is_over(self):
        """Check if every player has run out of cards."""
        return sum(self.players_in_play) == 1

    def get_player_id(self):
        """ Return the current player's ID (0, 1, 2, or 3) """
        return self.current_player

    def init_game(self):
        self.dealer = TienLenDealer()
        self.players = [Player(i) for i in range(self.num_players)]
        self.dealer.deal_cards(self.players)

        # Java-conformant state tracking
        self.current_stack = []      # stack<TreeSet<Card>> currentStack
        self.players_in_play = [True] * self.num_players
        self.players_in_game = [True] * self.num_players
        self.state = "none"          # stackState state
        self.is_first_round = True

        self.current_player = 0
        for i, p in enumerate(self.players):
            if self.lowest_card in p.hand:
                self.current_player = i
                break

        return self.get_state(self.current_player), self.current_player

    def step(self, action_id):
        """ Logic to process an agent's move and update the game state. """
        action_cards = self.id_to_action[action_id]
        player = self.players[self.current_player]

        if action_cards is None: # This is a "Pass" (ID 0)
            self.players_in_play[self.current_player] = False
        else:
            self.current_stack.append(list(action_cards))
            self.state = self.judger.get_type(action_cards)
            for card in action_cards:
                player.hand.remove(card)

            self.is_first_round = False

            self.latest_move = list(action_cards)
            self.last_player = self.current_player

        # Check if everyone passed except one person
        if sum(self.players_in_play) == 1:
            # Round reset: the winner of the trick gets to start a new round
            self.last_move = None
            self.state = "none"
            self.players_in_play = self.players_in_game.copy()
            self.current_player = self.last_player # Winner starts
        else:
            # Move to next player who hasn't passed yet
            self.current_player = (self.current_player + 1) % self.num_players
            while not self.players_in_play[self.current_player]:
                self.current_player = (self.current_player + 1) % self.num_players

        return self.get_state(self.current_player), self.current_player

    def get_state(self, player_id):
        player = self.players[player_id]

        # The top of the stack is what the Judger compares against
        top_of_stack = self.current_stack[-1] if self.current_stack else None

        # Judger needs stackState to filter moves like your Java getValidMoves(state, peek)
        legal_actions = self.judger.get_legal_actions(
            player,
            top_of_stack,
            self.state,
            self.action_to_id,
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

class Player:
    def __init__(self, player_id):
        self.player_id = player_id
        self.hand = []
