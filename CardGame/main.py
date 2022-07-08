import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

from card_game import CardGame, Player, Card
from keras.optimizers import Adam, SGD
from card_game.gambitPlayer import GambitPlayer


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
            options = list(filter(lambda card: card.suit == list(
                game_state["discard"])[0].suit, game_state["hand"]))
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


def main():
    players = [RandomPlayer(), RandomPlayer(), GambitPlayer(), RandomPlayer()]
    players_list = [players[0], players[1], players[2], players[3]]
    players_wl_ratio = {
        players[0]: 0,
        players[1]: 0,
        players[2]: 0,
        players[3]: 0
    }
    for i, ep in enumerate(range(100)):
        random.shuffle(players)
        game = CardGame(players[0], players[1], players[2], players[3], delay=0, display=False,
                        full_deck=False)
        results = game.start()

        loser = None
        for result in results:
            if loser is None:
                loser = result
            else:
                if results[loser] < results[result]:
                    loser = result
        players_wl_ratio[loser] = players_wl_ratio[loser] + 1
        resultStr = players_list[0].get_name() + " : " + str(players_wl_ratio[players_list[0]]), \
            players_list[1].get_name() + " : " + str(players_wl_ratio[players_list[1]]), \
            players_list[2].get_name() + " : " + str(players_wl_ratio[players_list[2]]), \
            players_list[3].get_name() + " : " + \
            str(players_wl_ratio[players_list[3]])

        print('Episode:', i, 'results:', resultStr)

    for p in players:
        if type(p) == type(GambitPlayer()):
            p.model.save_weights('wagi')
            break

    # Play final game
    random.shuffle(players)
    game = CardGame(players[0], players[1], players[2], players[3], delay=100, display=True,
                    full_deck=False)
    print(game.start())


if __name__ == '__main__':
    main()
