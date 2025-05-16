import torch
from torch import nn
from wordle_pc import Wordle

class Neural_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        letters = 26
        positions = 5
        states = 3
        game = Wordle()
        nr_words = len(game.get_word_list())
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(letters * positions * states, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, nr_words)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def get_nn():
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    return Neural_Network().to(device)




def main():
    pass


if __name__ == "__main__":
    main()
