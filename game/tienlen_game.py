from .dealer import TienLenDealer
from .judger import TienLenJudger

class TienLenGame:

    def __init__(self):
        self.num_players = 4
        self.judger = TienLenJudger()

        self.actions = self.judger._generate_all_possible_combinations()
        self.num_actions = len(self.actions)
        
        self.action_to_id = {combo: i for i, combo in enumerate(self.actions)}
        self.id_to_action = {i: combo for i, combo in enumerate(self.actions)}

    def get_num_actions(self):
        return self.num_actions

    def get_num_players(self):
        return self.num_players

    def get_num_actions(self):
        return self.num_actions

    def is_over(self):
        """Check if any player has run out of cards."""
        for player in self.players:
            if len(player.hand) != 0:
                return False
        return True

    def get_player_id(self):
        """ Return the current player's ID (0, 1, 2, or 3) """
        return self.current_player

    def init_game(self):
        """ Initializes all game variables at the start of a new match. """
        self.dealer = TienLenDealer()
        self.players = [Player(i) for i in range(self.num_players)]
        self.dealer.deal_cards(self.players)

        for (i, player) in enumerate(self.players):
            if player.hand.index((16,0)) > -1:
                self.current_player = 0 # In real Tien Len, find the 3 of Spades
        self.last_move = None
        self.last_player = None

        # Round management: who is still allowed to play in the current trick?
        self.active_players = [True] * self.num_players

        return self.get_state(self.current_player), self.current_player


    def init_game(self):
        """ Equivalent to Java's startArraysSetup() and startGameSetup() """
        self.dealer = TienLenDealer()
        self.players = [Player(i) for i in range(self.num_players)]
        self.dealer.deal_cards(self.players)

        # Java-conformant state tracking
        self.current_stack = []      # stack<TreeSet<Card>> currentStack
        self.players_in_play = [True] * self.num_players
        self.state = "none"          # stackState state
        self.is_first_round = True

        # Determine starting player (who has 3 of Spades)
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
            self.active_players[self.current_player] = False
        else:
            # 1. Update the table
            self.last_move = list(action_cards)
            self.last_player = self.current_player
            # 2. Remove cards from player's hand
            for card in action_cards:
                player.hand.remove(card)

        # Check if everyone passed except one person
        if sum(self.active_players) == 1:
            # Round reset: the winner of the trick gets to start a new round
            self.last_move = None
            self.active_players = [True] * self.num_players
            self.current_player = self.last_player # Winner starts
        else:
            # Move to next player who hasn't passed yet
            self.current_player = (self.current_player + 1) % self.num_players
            while not self.active_players[self.current_player]:
                self.current_player = (self.current_player + 1) % self.num_players

        return self.get_state(self.current_player), self.current_player

    def get_state(self, player_id):
        """ Formats the data the Env needs to build the observation vector. """
        player = self.players[player_id]

        # Note: You pass 'self.action_to_id' so the judger knows how to return IDs
        legal_actions = self.judger.get_legal_actions(
            player, self.last_move, self.action_to_id, self.active_players
        )

        return {
            'player_id': player_id,
            'hand': player.hand,
            'last_move': self.last_move,
            'legal_actions': legal_actions,
            'others_hand_count': [len(p.hand) for i, p in enumerate(self.players) if i != player_id]
        }

class Player:
    def __init__(self, player_id):
        self.player_id = player_id
        self.hand = []
