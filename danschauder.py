import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering
from os.path import join, isfile
import math
import pickle


def count_states(m, n):
    return math.factorial(m + n - 1) / (
        math.factorial(n - 1) * math.factorial(m)
    )


count_states(5, 5)

dict_size = 1000

class Interpreter:
    def __init__(self, k=5):
        location = join("data", "valid-wordle-words.txt")
        with open(location, "r") as words:
            data = words.read()
            word_list = data.split()
        self.words_full = random.sample(word_list, dict_size)
        self.words_full = [word.lower() for word in self.words_full]
        self.words = self.words_full.copy()
        self.k = k
        self.n = len(self.words)

    def calculate_word_similarity(self, a, b):
        assert len(a) == len(b), f"{a} is a different length than {b}"
        score = np.sum([1 for i in range(len(a)) if a[i] == b[i]])
        a_set = set(a)
        b_set = set(b)
        return (score + len(a_set.intersection(b_set))) / 10

    def make_similarity_graph(self):
        n = len(self.words)
        full = n == dict_size
        sim = join("data", f"similarity_graph{dict_size}")
        sim_csr = join("data", f"similarity_graph_csr{dict_size}")
        if isfile(sim) and isfile(sim_csr) and full:
            with open(sim, "rb") as pickled_graph:
                self.G = pickle.load(pickled_graph)
            with open(sim_csr, "rb") as pickled_csr:
                self.G_csr = pickle.load(pickled_csr)
        else:
            n = len(self.words)
            self.G = np.zeros((n, n))
            for i in range(n):
                for j in range(i,n):
                    similarity = self.calculate_word_similarity(
                        self.words[i], self.words[j]
                    )
                    self.G[i, j] = self.G[j, i] = similarity

            self.G_csr = csr_matrix(self.G)
            if full:
                with open(sim, "wb") as pickled_graph:
                    pickle.dump(self.G, pickled_graph)
                with open(sim_csr, "wb") as pickled_csr:
                    pickle.dump(self.G_csr, pickled_csr)

    def save_similarity_graph(self):
        self.G_full = self.G.copy()
        self.G_csr_full = self.G_csr.copy()

    def make_clusters(self):
        if len(self.words) > self.k:
            self.clusters = SpectralClustering(
                n_clusters=self.k,
                assign_labels="discretize",
                affinity="precomputed",
                random_state=13,
            ).fit_predict(self.G_csr)
        else:
            self.clusters = np.array([i for i in range(self.n)])

    def make_cluster_frequencies(self):
        labels, cluster_counts = np.unique(self.clusters, return_counts=True)
        self.cluster_freqs = cluster_counts / np.sum(cluster_counts)

    def make_word_groups(self):
        self.words_arr = np.array(self.words)
        self.word_groups = []
        for i in range(self.k):
            if i >= len(self.clusters):
                self.word_groups.append(np.array([]))
            else:
                self.word_groups.append(np.asarray(self.clusters == i).nonzero()[0])

    def get_state(self) -> str:
        cluster_bins = ""
        for i in range(self.k):
            if i >= len(self.cluster_freqs):
                cluster_bins = cluster_bins + "0"
            elif self.cluster_freqs[i] > 0.8:
                cluster_bins = cluster_bins + "4"
            elif self.cluster_freqs[i] > 0.6:
                cluster_bins = cluster_bins + "3"
            elif self.cluster_freqs[i] > 0.4:
                cluster_bins = cluster_bins + "2"
            elif self.cluster_freqs[i] > 0.2:
                cluster_bins = cluster_bins + "1"
            else:
                cluster_bins = cluster_bins + "0"
        return cluster_bins

    def get_word_cluster(self, cluster_index):
        return self.words_arr[self.word_groups[cluster_index]]

    def get_cluster_centroid(self, cluster_index) -> str:
        if len(self.word_groups[cluster_index]) > 0:
            idx = self.word_groups[cluster_index]
        else:
            idx = np.array([0])
        return self.words_arr[idx][np.argmax(np.sum(self.G[idx, :], axis=1))]

    def filter_words(self, guess, feedback) -> None:
        self.new_words = []
        for word in self.words:
            add_word = True
            i = 0
            allowed_chars = set()
            for j in range(len(feedback)):
                if feedback[j] > 0:
                    allowed_chars.add(guess[j])
            while add_word and i < len(feedback):
                if (
                    feedback[i] == 0
                    and guess[i] in word
                    and guess[i] not in allowed_chars
                ):
                    add_word = False
                elif feedback[i] == 1 and guess[i] not in word:
                    add_word = False
                elif feedback[i] == 1 and word[i] == guess[i]:
                    add_word = False
                elif feedback[i] == 2 and word[i] != guess[i]:
                    add_word = False
                i += 1
            if add_word:
                self.new_words.append(word)
        assert (
            self.new_words
        ), f"Filtering failed with guess: {guess} and feedback: {feedback}"

    def get_reward(self, feedback):
        shrink_factor = self.n / len(self.new_words)
        if np.sum(feedback) == 10:
            shrink_factor += 1000
        return np.sum(feedback) + shrink_factor
    
    def update_word_list(self):
        self.words = self.new_words
        self.n = len(self.words)

    def reset(self):
            self.words = self.words_full.copy()
            self.n = len(self.words)
            self.G = self.G_full.copy()
            self.G_csr = self.G_csr_full.copy()


class QLearner:
    def __init__(
        self,
        action_count=5,
        alpha=0.2,
        gamma=0.9,
        random_action_rate=0.5,
        random_action_decay_rate=0.99,
    ):
        self.action_count = action_count
        self.alpha = alpha
        self.gamma = gamma
        self.random_action_rate = random_action_rate
        self.random_action_decay_rate = random_action_decay_rate
        self.s = "10001"
        self.a = 0
        self.qtable = {"10001": np.zeros(self.action_count)}

    def get_random_action(self):
        return random.randint(0, self.action_count - 1)

    def update_qtable(self, s, a, s_prime, r):
        if s_prime in self.qtable.keys():
            a_prime = np.argmax(self.qtable[s_prime])
        else:
            self.qtable[s_prime] = np.zeros(self.action_count)
            a_prime = random.randint(0, self.action_count - 1)

        self.qtable[s][a] = (1 - self.alpha) * self.qtable[s][a] + self.alpha * (
            r + self.gamma * self.qtable[s_prime][a_prime]
        )
        return a_prime

    def get_action(self, s_prime, r):
        a_prime = self.update_qtable(self.s, self.a, s_prime, r)
        self.random_action_rate = (
            self.random_action_rate * self.random_action_decay_rate
        )
        random_action = np.random.random() <= self.random_action_rate
        if random_action:
            action = self.get_random_action()
        else:
            action = a_prime

        self.s = s_prime
        self.a = action
        return action


class Grader:
    def __init__(self):
        pass

    def pick_word(self, words):
        self.word = np.random.choice(words)

    def evaluate_guess(self, guess):
        word_copy = self.word
        feedback = np.zeros(5)
        for i in range(5):
            if guess[i] == self.word[i]:
                feedback[i] = 2
            elif guess[i] in word_copy:
                feedback[i] = 1
                word_char_index = word_copy.index(guess[i])
                word_copy = (
                    word_copy[:word_char_index] + word_copy[word_char_index + 1 :]
                )
            else:
                feedback[i] = 0
        return feedback


def initialize_simulation():
    interpreter = Interpreter()
    interpreter.make_similarity_graph()
    print("similarity graph constructed!")
    interpreter.save_similarity_graph()
    qlearner = QLearner()
    grader = Grader()
    return interpreter, qlearner, grader


def run_simulation(interpreter:Interpreter, qlearner:QLearner, grader:Grader, verbose=True):
    guess_count = 0
    grader.pick_word(interpreter.words)
    won = False
    r = 0
    while not won:
        interpreter.make_clusters()
        interpreter.make_cluster_frequencies()
        interpreter.make_word_groups()
        s = interpreter.get_state()
        a = qlearner.get_action(s, r)
        guess = interpreter.get_cluster_centroid(a)
        feedback = grader.evaluate_guess(guess)
        guess_count += 1
        if np.sum(feedback) == 10:
            r = interpreter.get_reward(feedback)
            a = qlearner.get_action(s, r)
            won = True
        else:
            interpreter.filter_words(guess, feedback)
            r = interpreter.get_reward(feedback)
            interpreter.update_word_list()
            interpreter.make_similarity_graph()
    if verbose:
        print(f"Guessed {grader.word} in {guess_count} guesses")
    interpreter.reset()
    return guess_count, interpreter, qlearner, grader


interpreter, qlearner, grader = initialize_simulation()
training_epochs = 1000
epochs = np.arange(training_epochs)
guesses = np.zeros(training_epochs)
for i in range(training_epochs):
    guess_count, interpreter, qlearner, grader = run_simulation(
        interpreter, qlearner, grader, verbose=False
    )
    guesses[i] = guess_count
    if (i+1) % 100 == 0:
        print(f"{(i+1)/training_epochs*100}% done...")


plt.bar(epochs,guesses)
plt.show()

print(f'Average guesses: {np.mean(guesses)}')

print(f'Total game losses out of {training_epochs}: {np.sum(guesses>6)}')

print(f'Overall win rate: {(training_epochs-np.sum(guesses>6))/training_epochs*100}%')