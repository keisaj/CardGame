# from player import Player
from card_game import CardGame, Player, Card
import random
import numpy as np
from tensorflow import keras
import tensorflow as tf

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
# Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) — Q[state, action])


def discount_rewards(rewards, gamma=0.95):
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R
    return discounted_rewards


def compute_loss(logits, actions, rewards):
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=actions)
    loss = tf.reduce_mean(neg_logprob * rewards)
    return loss


def train_step(model, loss_function, optimizer, observations, actions, discounted_rewards, custom_fwd_fn=None):
    with tf.GradientTape() as tape:
        if custom_fwd_fn is not None:
            prediction = custom_fwd_fn(observations)
        else:
            prediction = model(observations)
        loss = loss_function(prediction, actions, discounted_rewards)
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 2)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


class GambitPlayer(Player):
    name = 'Gambit Player'

    def __init__(self) -> None:
        global player
        self.number = player
        player += 1
        self.epsilon = 1.0
        self.optimizer = Adam()
        self.model = self.create_model()
        self.card_map = self.get_card_map()
        self.memory = Memory()
        self.reward = 0
        self.action = 0
        self.observation = 0

    def get_name(self) -> str:
        return self.name + f'{self.number}'

    def set_temp_reward(self, discarded_cards: dict, point_deltas: dict):
        for point in point_deltas:
            if type(point) == type(GambitPlayer()):
                self.reward += -point_deltas[point]
                self.memory.add_to_memory(
                    self.observation, self.action, self.reward)

    def set_final_reward(self, points: dict):
        total_reward = sum(self.memory.rewards)
        train_step(self.model, compute_loss, self.optimizer,
                   observations=np.vstack(self.memory.observations),
                   actions=np.array(self.memory.actions),
                   discounted_rewards=discount_rewards(self.memory.rewards))
        self.memory.clear()

    def make_move(self, game_state: dict) -> Card:
        self.reward = 0
        pred = self.get_action(game_state)
        card = self.decode_number_to_card_from_hand(pred, game_state)
        if self.is_legal_move(card, game_state):
            print('Yay, legal action')
            self.action = self.decode(card)
            return card

        self.punish(pred)
        card = self.get_any_legal_move(game_state)
        self.action = self.decode(card)
        return card

    def is_legal_move(self, card, game_state):
        if card is None:
            return False

        if not game_state["discard"]:
            return True
        else:
            options = list(
                filter(lambda c: c.suit == list(game_state["discard"])[0].suit, game_state["hand"]))
            if len(options) > 0:
                return card in options
            else:
                return True

    def punish(self, pred):
        print('Illegal action')
        self.memory.add_to_memory(self.observation, pred, -500)

    def get_any_legal_move(self, game_state):
        if not game_state["discard"]:
            return random.choice(game_state["hand"])
        else:
            options = list(
                filter(lambda c: c.suit == list(game_state["discard"])[0].suit, game_state["hand"]))
            if len(options) > 0:
                return random.choice(options)
            else:
                return random.choice(game_state['hand'])

    def get_action(self, game_state):
        self.observation = self.get_input_vector(game_state)
        self.observation = np.expand_dims(self.observation, axis=0)
        logits = self.model.predict(self.observation, verbose=0)
        action = tf.random.categorical(logits, num_samples=1)
        action = action.numpy().flatten()
        return action[0]

    def create_model(self):
        # learning_rate = 0.01
        action_size = 24
        state_size = 9  # [1, 2, 3, 4, 5, 0,| 5, 6, 7 ]

        model = keras.Sequential()
        model.add(keras.layers.Dense(
            64, input_dim=state_size, activation='relu'))
        model.add(keras.layers.Dense(64, activation="relu"))
        model.add(keras.layers.Dense(action_size, activation="softmax"))
        model.compile()
        return model

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
                return card
        return None

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
