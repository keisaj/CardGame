# from player import Player
from card_game import CardGame, Player, Card

class GambitPlayer(Player):

    name = 'Gambit Player'
    def __init__(self) -> None:
        global player
        self.number = player
        player += 1
        
    def make_move(self, game_state: dict) -> Card:
        """
        The player will receive a dict with:
        - 'hand': list of held cards
        - 'discard': list of discarded cards in this round
        """
        """
        move = model.predict(game_state)
        """

        played_card = super().make_move(game_state)
        return played_card

    def get_name(self) -> str:
        return self.name + f'{self.number}'
    
    def set_temp_reward(self, discarded_cards: dict, point_deltas: dict):
        """
        After four cards get played, every player will receive a dict with:
        - 'discarded_cards': dict of these discarded cards by each player (ordered!)
        - 'point_deltas': dict of points received in this round by each player
        """
        return super().set_temp_reward(discarded_cards, point_deltas)
    
    def set_final_reward(self, points: dict):
        """
        After all cards have been played, every player will receive a dict with
        points received in this round by each player. A game consists of eleven such full rounds.
        """
        return super().set_final_reward(points)

