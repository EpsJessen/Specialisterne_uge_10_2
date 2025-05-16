from os.path import join
from random import randrange, sample
import torch

class Wordle:
    def __init__(self, nr_words=None, dict_size=None):
        self.round = 0
        self.max_rounds = 100
        self.current_state = torch.tensor([[[0 for i in range(3)]for j in range(5)] for k in range(26)])
        self.full_dict = self.get_word_list()
        if dict_size is not None:
            self.full_dict = sample(self.full_dict, dict_size)
        
        if nr_words is not None:
            self.words = sample(self.full_dict, nr_words)
        else:
            self.words = self.full_dict
        self._chosen_word = ""
        self.prev_guesses = {}
        self.reward = 0
        self.games_played = -1

    def get_word_list(self) -> list[str]:
        location = join("data", "valid-wordle-words.txt")
        with open(location, "r") as words:
            data = words.read()
            word_list = data.split()
        return word_list

    def get_word(self) -> str:
        word_nr = randrange(0, len(self.words))
        return self.words[word_nr]
    
    def start_game(self):
        self.round = 1
        self._chosen_word = self.get_word()

            

    def compare(self, guess):
        target = list(self._chosen_word)
        guess = list(guess)
        result = [0, 0, 0, 0, 0]
        for pos, letter in enumerate(guess):
            if letter == target[pos]:
                result[pos] = 2
                target[pos] = "_"
                guess[pos] = "*"
        for pos, letter in enumerate(guess):
            if letter in target:
                result[pos] = 1
                target[target.index(letter)] = "_"
                guess[pos] = "*"
        return result

    def make_guess(self, guess: str):
        self.reward = 0
        # Update round number and compare word to correct
        self.round += 1
        res = self.compare(guess)
        if guess == self._chosen_word:
            return 1, res
        if self.round > self.max_rounds:
            return -1, res
        return 0, res
        
    def step(self, word_nr):
        word = self.full_dict[word_nr]
        res, reward = self.make_guess(word)
        won = False
        lost = False
        if res == 1:
            won = True
        if res == -1:
            lost = True
        return self.get_state(), reward, won, lost, None
    
    def get_state(self):
        return self.current_state.flatten()
    

def main():
    game = Wordle()
    game.start_game()
    game.make_guess("crane")
    game.make_guess("socks")
    game.make_guess("hello")
    game.make_guess("ideas")
    print(game.current_state)
    print(game._chosen_word)
    print(game.get_state)
    print(game.get_state().flatten())


if __name__ == "__main__":
    main()
