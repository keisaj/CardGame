# from player import Player
from card_game import CardGame, Player, Card
import random
import numpy as np
from tensorflow import keras

class GambitPlayer(Player):

    name = 'Gambit Player'
    def __init__(self) -> None:
        global player
        self.number = player
        player += 1
        self.epsilon = 1.0 
        self.model = self.create_model()
        
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

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """
        possible_actions = self.get_legal_actions(state)       

        return random.choice(possible_actions) if (np.random.random() <= self.epsilon) else self.get_best_action(state)

    def get_best_action(self, state):
        """
        Compute the best action to take in a state.
        """
        return np.argmax(self.model.predict(state)[0])
    
    def get_legal_actions(self, state):
        return state['hand']

    def create_model(self):
        learning_rate = 0.001
        action_size = 6 # 6 kart dostajemy na start [1, 2, 3, 4, 5, 0,| 5, 6, 7, 8 ]

        model = keras.Sequential()
        model.add(keras.layers.Dense(64, input_dim=state_size, activation='relu'))
        model.add(keras.layers.Dense(64, activation="relu"))
        model.add(keras.layers.Dense(action_size))
        model.compile(loss="mse",
              optimizer=keras.optimizers.Adam(learning_rate=learning_rate))




