import random
from tabnanny import verbose
import keras
import tensorflow as tf
import numpy as np
from card_game import CardGame, Player, Card, card_game
import itertools

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

# --------------------------------------------------------------------------------------------------------------------
def choose_action(model, observation, single=True):
    observation = np.expand_dims(observation, axis=0) if single else observation
    logits = model.predict(observation, verbose=0)
    action = tf.random.categorical(logits, num_samples=1)
    action = action.numpy().flatten()
    return action[0] if single else action


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


def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x.astype(np.float32)


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


class Agent(Player):
    def __init__(self):
        ranks = ['9', '10', 'Jack', 'Queen', 'King', 'Ace']
        suits = ['Clovers', 'Diamonds', 'Hearts', 'Spades']
        self.states = {}
        self.reversed = {}
        counter = 1
        for i in itertools.product(ranks, suits):
            self.states[i[0] + i[1]] = counter
            self.reversed[counter] = [i[0], i[1]]
            counter += 1
        self.states[0] = 0
        self.reversed[0] = ['none', 'none']

        self.memory = Memory()
        self.model = keras.models.Sequential([
            keras.layers.Dense(units=(6 + 3 + 20), activation="relu"),
            keras.layers.Dense(units=(64), activation="relu"),
            keras.layers.Dense(units=(64), activation="relu"),
            keras.layers.Dense(units=(64), activation="relu"),
            keras.layers.Dense(units=len(self.states), activation='linear')
        ])
        self.optimizer = keras.optimizers.Adam()
        self.model.compile(self.optimizer)
        self.action = 0
        self.observation = 0
        self.reward = 0
        self.prev_discards = []


    def make_move(self, game_state: dict) -> Card:
        hand = []
        for card in game_state['hand']:
            hand.append(self.states[card.rank + card.suit])
        while len(hand) < 6:
            hand.append(0)

        discard = []
        for card in game_state['discard']:
            discard.append(self.states[card.rank + card.suit])
        while len(discard) < 3:
            discard.append(0)

        old_discards = []
        for card in self.prev_discards:
            old_discards.append(self.states[card.rank + card.suit])
        while len(old_discards) < 20:
            old_discards.append(0)

        self.observation = np.array(hand + discard + old_discards)
        self.action = choose_action(self.model, self.observation)
        r = self.reversed[self.action]
        pActions = [a for a in game_state['hand'] if (a.rank == r[0] and a.suit == r[1])]

        if len(pActions) != 1:
            self.memory.add_to_memory(self.observation, self.action, -500)
            if not game_state["discard"]:
                rc = random.choice(game_state["hand"])
                self.action = self.states[rc.rank + rc.suit]
                return rc
            else:
                options = list(
                    filter(lambda card: card.suit == list(game_state["discard"])[0].suit, game_state["hand"]))
                if len(options) > 0:
                    rc = random.choice(options)
                    self.action = self.states[rc.rank + rc.suit]
                    return rc
                else:
                    rc = random.choice(game_state["hand"])
                    self.action = self.states[rc.rank + rc.suit]
                    return rc
        else:
            if not game_state["discard"]:
                return pActions[0]
            else:
                options = list(
                    filter(lambda card: card.suit == list(game_state["discard"])[0].suit, game_state["hand"]))
                if len(options) > 0:
                    rc = random.choice(options)
                    self.action = self.states[rc.rank + rc.suit]
                    return rc
                else:
                    return pActions[0]

    def get_name(self) -> str:
        return f"Player"

    def set_temp_reward(self, discarded_cards: dict, point_deltas: dict):
        for dc in discarded_cards:
            self.prev_discards.append(discarded_cards[dc])
        for point in point_deltas:
            if type(point) == type(Agent()):
                self.reward = -point_deltas[point]
                self.memory.add_to_memory(self.observation, self.action, self.reward)

    def set_final_reward(self, points: dict):
        self.prev_discards = []
        total_reward = sum(self.memory.rewards)
        train_step(self.model, compute_loss, self.optimizer,
                   observations=np.vstack(self.memory.observations),
                   actions=np.array(self.memory.actions),
                   discounted_rewards=discount_rewards(self.memory.rewards))
        self.memory.clear()


def main():

    game = CardGame(Agent(), RandomPlayer(), RandomPlayer(), RandomPlayer(), delay=0, display=False, full_deck=False)
    players = {}
    for player in game.players:
        players[player.get_name()] = 0

    for ep in range(1000):
        points = game.start()
        print(f'episode: {ep}', players)
        for player in points:
            players[player.get_name()] += points[player]

    for player in game.players:
        if type(player) == type(Agent()):
            model = player.model
    game = CardGame(Agent(), RandomPlayer(), RandomPlayer(), RandomPlayer(), delay=1000, display=True, full_deck=False)
    game.players[0].model = model
    print(game.start())


if __name__ == '__main__':
    main()
