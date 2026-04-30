from .dealer import TienLenDealer
from .judger import TienLenJudger

class TienLenGame:

    def __init__(self):
        self.num_players = 4
        self.judger = TienLenJudger()

        self.actions = self.judger.generate_all_possible_combinations()
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

    def step(self, action):
        """ 
        1. Process the action (e.g., remove card from hand)
        2. Advance to the next player
        """
        # For now, just a placeholder to keep the loop moving:
        self.current_player = (self.current_player + 1) % self.num_players
        
        # Return next state and next player id
        return self.get_state(self.current_player), self.current_player
    
    def init_game(self):
        self.dealer = TienLenDealer()
        self.players = [Player(i) for i in range(self.num_players)]
        self.dealer.deal_cards(self.players)
        self.current_player = 0
        self.last_move = None
        return self.get_state(0), 0

    def get_state(self, player_id):
        player = self.players[player_id]
        return {
            'hand': player.hand,
            'last_move': self.last_move,
            'legal_actions': self.judger.get_legal_actions(player.hand, self.last_move)
        }

class Player:
    def __init__(self, player_id):
        self.player_id = player_id
        self.hand = []
