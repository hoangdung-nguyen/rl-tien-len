class TienLenPlayer(Player):
    def __init__(self, player_id, np_random):
        super().__init__(player_id, np_random)
        self.hand = []
