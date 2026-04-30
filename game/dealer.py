import random

class TienLenDealer:
    def __init__(self):
        # 3-10 are 3-10, J=11, Q=12, K=13, A=14, 2=15
        self.deck = [(rank, suit) for rank in range(3, 16) for suit in range(4)]
        
    def deal_cards(self, players):
        random.shuffle(self.deck)
        for i, player in enumerate(players):
            player.hand = sorted(self.deck[i*13:(i+1)*13])