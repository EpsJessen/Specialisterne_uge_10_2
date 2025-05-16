from os.path import join
from random import randrange


class Wordle:
    def __init__(self):
        self.round = 0
        self.max_rounds = 6
        self.words = self.get_word_list()
        self._chosen_word = ""

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
        result = [-1, -1, -1, -1, -1]
        for pos, letter in enumerate(guess):
            if letter == target[pos]:
                result[pos] = 1
                target[pos] = "_"
                guess[pos] = "*"
        for pos, letter in enumerate(guess):
            if letter in target:
                result[pos] = 0
                target[target.index(letter)] = "_"
                guess[pos] = "*"
        return result

    def result_print(self, result):
        colors = []
        for val in result:
            match val:
                case -1:
                    colors.append("🟥")
                case 0:
                    colors.append("🟨")
                case 1:
                    colors.append("🟩")
        color_string = ''.join(colors)
        print(color_string)

    def make_guess(self, guess: str):
        if guess in ["quit", "exit", "q", "q()"]:
            return True
        # Check round
        current_round = self.round
        if current_round < 1:
            print("Game not started yet!")
            return True
        if current_round > self.max_rounds:
            print("Game already over!")
            return True

        # Validate guess
        if len(guess) != 5:
            print(f"Guesses must have length 5, but {guess} has length {len(guess)}!")
            return False
        guess = guess.lower()
        if guess not in self.words:
            print(f"{guess} is not in our list of (English) words...")
            return False

        # Update round number and compare word to correct
        self.round += 1
        result = self.compare(guess)
        self.result_print(result)
        if guess == self._chosen_word:
            print("Congratulations, you've won!")
            return True
        if self.round >= self.max_rounds:
            print(f"Womp womp, you've lost\nThe correct word was {self._chosen_word}")
            return True
        return False
        


def main():
    game = Wordle()
    game.start_game()
    while True:
        done = game.make_guess(input("Make your guess: "))
        if done:
            break



if __name__ == "__main__":
    main()
