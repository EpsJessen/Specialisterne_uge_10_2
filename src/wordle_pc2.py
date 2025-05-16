from os.path import join
from random import randrange, sample
import torch

class Wordle:
    def __init__(self, nr_words=None, dict_size=None):
        self.round = 0
        self.max_rounds = 25
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
        if self.games_played < 3 * len(self.words):
            return self.words[self.games_played % len(self.words)]
        return self.words[word_nr]

    def start_game(self):
        self.games_played += 1
        if self.games_played < len(self.words) * 3:
            self.max_rounds = float("inf")
        else:
            self.max_rounds = 25
        self.prev_guesses = {}
        self.round = 1
        self._chosen_word = self.get_word()
        self.current_state = torch.tensor([[[0 for i in range(3)]for j in range(5)] for k in range(26)])

    def update_state(self, letter, pos, state):
        letter_nr = ord(letter)-97
        match state:
            # PT nem løsning hvor vi kun opdatere med positionerne fra det gættede ord
            # Letter is incorrect (possibly only this occurance)
            case 0:
                # Check if we already knew the letter was not in the word
                pre_wrongs = False
                for i in range(5):
                    if self.current_state[letter_nr][i][state] == 1:
                        pre_wrongs = True
                        break
                # If so reduce score
                if pre_wrongs:
                    self.reward -= 2
                # Else give small reward
                else:
                    self.reward += 0.05
                self.current_state[letter_nr][pos][state] = 1
            # Letter exists in word but not at this position
            case 1:
                # Check if we already knew
                pre_correct = False
                for i in range(5):
                    if self.current_state[letter_nr][i][2] == 1:
                        pre_correct = True
                        break
                pre_incorrect = False
                for i in range(5):
                    if self.current_state[letter_nr][i][state] == 1:
                        pre_incorrect = True
                        break
                # If we didn't know about the letter before
                if not (pre_incorrect or pre_correct):
                    self.reward += 0.08
                # If we didn't know the correct position and didn't know this was wrong 
                elif not (pre_correct or self.current_state[letter_nr][pos][state] == 1):
                    self.reward += 0.04
                # If we did know this was wrong
                elif self.current_state[letter_nr][pos][state] == 1:
                    self.reward -= 0.1
                self.current_state[letter_nr][pos][state] = 1
                # Other locations cannot be updated as we do not know where letter belongs
            # Letter is correctly placed
            case 2:
                # If this is new knowledge
                if self.current_state[letter_nr][pos][state] == 0:
                    self.reward += 0.40
                self.current_state[letter_nr][pos][state] = 1
            

    def compare(self, guess):
        target = list(self._chosen_word)
        guess = list(guess)
        result = [0, 0, 0, 0, 0]
        for pos, letter in enumerate(guess):
            if letter == target[pos]:
                self.update_state(letter, pos, 2)
                result[pos] = 2
                target[pos] = "_"
                guess[pos] = "*"
        for pos, letter in enumerate(guess):
            if letter in target:
                self.update_state(letter, pos, 1)
                result[pos] = 1
                target[target.index(letter)] = "_"
                guess[pos] = "*"
        for pos, letter in enumerate(guess):
            if letter != '*':
                self.update_state(letter, pos, 0)
        return result

    def make_guess(self, guess: str):
        self.reward = 0
        # Update round number and compare word to correct
        self.round += 1
        self.compare(guess)
        prev = self.prev_guesses.get(guess, 0)
        if prev == 0:
            self.reward += 2
        else:
            self.reward -= 5 + prev ** 2
        current_guesses = self.prev_guesses.get(guess, 0)
        self.prev_guesses[guess] = current_guesses + 1
        if guess == self._chosen_word:
            self.reward += 1000 - 5 * self.round
            return 1, self.reward
        if self.round > self.max_rounds:
            self.reward -= 100
            return -1, self.reward
        return 0, self.reward
        
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
