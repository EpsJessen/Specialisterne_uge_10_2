from wordle_res import Wordle
from random import sample
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
import torch

plt.ion()

class RandomPlayer():

    def __init__(self):    
        self.game = Wordle()
        self.words = self.game.full_dict
        self.current_words = deepcopy(self.words)
        self.rounds = []

    def reduce_words(self, guess, res):
        new_words = []
        guess = list(guess)
        for word in self.current_words:
            word_str = word
            word = list(word)
            guess_copy = deepcopy(guess)
            result = [0, 0, 0, 0, 0]
            for pos, letter in enumerate(guess_copy):
                if letter == word[pos]:
                    result[pos] = 2
                    word[pos] = "_"
                    guess_copy[pos] = "*"
            for pos, letter in enumerate(guess_copy):
                if letter in word:
                    result[pos] = 1
                    word[word.index(letter)] = "_"
                    guess_copy[pos] = "*"
            if result == res:
                new_words.append(word_str)
        self.current_words = new_words

    def make_guess(self):
        guess = sample(self.current_words, 1)[0]
        won, result = self.game.make_guess(guess)
        if won != 0:
            return True
        else:
            self.reduce_words(guess, result)
            return False
        
    def new_game(self):
        self.game.start_game()

    def play_game(self):
        rounds = 0
        done = False
        self.current_words = self.words
        self.new_game()
        while not done:
            done = self.make_guess()
            rounds += 1
        return rounds

    def play_n_games(self, n):
        guesses_used = []
        for _ in range(n):
            rounds = self.play_game()
            self.rounds.append(rounds)
            self.plot_durations()
        self.plot_durations(True)

    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.rounds, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Playing...')
        plt.xlabel('Game')
        plt.ylabel('Guesses')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

def main():
    player = RandomPlayer()
    player.play_n_games(1000)
    total_rounds = sum(player.rounds)
    average = total_rounds/1000
    print(f"The average number of rounds used was {average} rounds!")
    print(sum([1 for nr in player.rounds if nr < 7])/1000)
    input("Press enter to quit: ")




if __name__ == "__main__":
    main()
