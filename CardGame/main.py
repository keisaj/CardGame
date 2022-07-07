import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

from card_game import CardGame, Player, Card

player = 1

class Memory:
    def __init__(self):
        self.clear()

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)

    def __len__(self):
        return len(self.actions)

class GambitPlayer(Player):
    name = 'Gambit Player'

    def __init__(self) -> None:
        global player
        self.number = player
        player += 1
        self.epsilon = 1.0
        self.model = self.create_model()
        self.card_map = self.get_card_map()
        self.memory = Memory()

    def make_move(self, game_state: dict) -> Card:
        pred = self.get_action(game_state)
        played_card = self.decode_number_to_card_from_hand(pred, game_state)
        return played_card

    def get_name(self) -> str:
        return self.name + f'{self.number}'

    def set_temp_reward(self, discarded_cards: dict, point_deltas: dict):
        # dodaje do memory
        return super().set_temp_reward(discarded_cards, point_deltas)

    def set_final_reward(self, points: dict):
        # nagradzam
        return super().set_final_reward(points)

    def get_action(self, game_state):
        self.observation = self.get_input_vector(game_state)
        self.observation = np.expand_dims(self.observation, axis=0)
        logits = self.model.predict(self.observation, verbose=0)
        action = tf.random.categorical(logits, num_samples=1)
        action = action.numpy().flatten()
        return action[0]

    def create_model(self):
        learning_rate = 0.001
        action_size = 24
        state_size = 9  # [1, 2, 3, 4, 5, 0,| 5, 6, 7 ]

        model = keras.Sequential()
        model.add(keras.layers.Dense(64, input_dim=state_size, activation='relu'))
        model.add(keras.layers.Dense(64, activation="relu"))
        model.add(keras.layers.Dense(action_size, activation="linear"))
        model.compile(loss="mse",
                      optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
        return model
        # na wyjsciu sieci chcemy miec pojedyncza liczbe z zakresu 1-24 wskazujaca karte z naszej mapy
        # te liczbe otrzymana od sieci podajemy do funkcji decode_number_to_card_from_hand() zeby dostac karte do zagrania

    def decode_number_to_card_from_hand(self, prediction, game_state):
        rank = None
        suit = None
        for r in self.card_map.keys():
            for s in self.card_map[r]:
                if self.card_map[r][s] == prediction:
                    rank = r
                    suit = s
                    break
        for card in game_state['hand']:
            if card.rank is rank and card.suit is suit:
                if not game_state["discard"]:
                    return random.choice(game_state["hand"])
                else:
                    options = list(
                        filter(lambda c: c.suit == list(game_state["discard"])[0].suit, game_state["hand"]))
                    if len(options) > 0:
                        return random.choice(options)
                    else:
                        return card

        # tu by sie przydalo ukarac siec bo chciala zagrac karte ktorej nie mamy w rece
        if not game_state["discard"]:
            return random.choice(game_state["hand"])
        else:
            options = list(
                filter(lambda c: c.suit == list(game_state["discard"])[0].suit, game_state["hand"]))
            if len(options) > 0:
                return random.choice(options)
            else:
                return random.choice(game_state['hand'])

    # returns vector of 6 elements with mapped card -> number
    # if there is less than 6 cards on hand it will be filled to 6 with zeros
    def get_input_vector(self, game_state):
        vect = []
        for card in game_state['hand']:
            vect.append(self.decode(card))
        for i in range(6 - len(vect)):
            vect.append(0)
        for discard in game_state['discard']:
            vect.append(self.decode(discard))
        for i in range(9 - len(vect)):
            vect.append(0)
        return np.array(vect)

    def get_card_map(self):
        ranks = ['9', '10', 'Jack', 'Queen', 'King', 'Ace']
        suits = ['Clovers', 'Diamonds', 'Hearts', 'Spades']
        counter = 0
        map = {}
        for i in range(6):
            map[ranks[i]] = {}
            for j in range(4):
                map[ranks[i]][suits[j]] = counter
                counter = counter + 1
        return map

    def decode(self, card):
        return self.card_map.get(card.rank).get(card.suit)


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


def main():
    game = CardGame(RandomPlayer(), RandomPlayer(), GambitPlayer(), RandomPlayer(), delay=100, display=False, full_deck=False)
    print(game.start())


if __name__ == '__main__':
    main()
