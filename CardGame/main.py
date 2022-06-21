# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import random

from card_game import CardGame, Player, Card

player = 1


class RandomPlayer(Player):
    """
    Makes random moves (but according to the rules)
    """
    def __init__(self):
        global player
        self.number = player
        player += 1

    def make_move(self, game_state: dict) -> Card:
        if not game_state["discard"]:
            return random.choice(game_state["hand"])
        else:
            options = list(filter(lambda card: card.suit == list(game_state["discard"])[0].suit, game_state["hand"]))
            if len(options) > 0:
                return random.choice(options)
            else:
                return random.choice(game_state["hand"])

    def get_name(self):
        return f"RandomPlayer{self.number}"

    def set_temp_reward(self, discarded_cards: dict, point_deltas: dict):
        pass

    def set_final_reward(self, points: dict):
        pass

<<<<<<< HEAD
class DebugPlayer(RandomPlayer):
    """
    Makes random moves (but according to the rules)
    """
    def __init__(self):
        global player
        self.number = player
        player += 1

    def make_move(self, game_state: dict) -> Card:
        return super().make_move(game_state)

    def get_name(self):
        return f"DebugPlayer{self.number}"

    def set_temp_reward(self, discarded_cards: dict, point_deltas: dict):
        return super().set_temp_reward(discarded_cards, point_deltas)

    def set_final_reward(self, points: dict):
        return super().set_final_reward(points)

=======
>>>>>>> 775a9d215b0b00de82939534141a437dc5137d8a
class GambitPlayer(Player):

    name = 'Gambit Player'
    def __init__(self) -> None:
        global player
        self.number = player
        player += 1
<<<<<<< HEAD
        
    def make_move(self, game_state: dict) -> Card:
        """
        The player will receive a dict with:
        - 'hand': list of held cards
        - 'discard': list of discarded cards in this round
        """
        played_card = super().make_move(game_state)
        return played_card

=======

    def make_move(self, game_state: dict) -> Card:
        return super().make_move(game_state)
>>>>>>> 775a9d215b0b00de82939534141a437dc5137d8a

    def get_name(self) -> str:
        return self.name + f'{self.number}'
    
    def set_temp_reward(self, discarded_cards: dict, point_deltas: dict):
<<<<<<< HEAD
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
=======
        return super().set_temp_reward(discarded_cards, point_deltas)
    
    def set_final_reward(self, points: dict):
>>>>>>> 775a9d215b0b00de82939534141a437dc5137d8a
        return super().set_final_reward(points)


def main():
    game = CardGame(RandomPlayer(), RandomPlayer(), DebugPlayer(), RandomPlayer(), delay=100, display=True, full_deck=False)
    print(game.start())
    # print(game.start())


if __name__ == '__main__':
    main()
