# Specialisterne_uge_10_2
reinforced learning for wordle


# A game of wordle

I've implemented the game wordle in a few different forms, with the main difference being the form of the feedback.

In the standard version `wordle_game.py`, the feedback (and input) is given in(/taken from) the terminal. In `wordle_res.py` the gamestate ({1:won, 0:ongoing, -1:lost}) and the result (list of length 5 with each index representing the "color" of the letter {0:grey, 1:yellow, 2:green}) is returned.
Finally, in `wordle_pcX.py` the gamestate (as above), the game status (a tensor representing the collected knowledge of the game), and the score of the guess is returned.

# Learning is not easy

In `wordle_learner.py`, I've implemented a d-q-learner, using as input the game status. It makes guesses by indicating which english 5-letter word it finds most likely based on the game satus.

Unfortunately, this learner achives a very low rate of success, ending up guessing at the same word many times in a row, especially after the initial learning period.

# Status cHeck

To see how well the learner is doing, I've implemented a random player, which makes a guess at a random 5-letter word matching the results of previous guesses.
Although not the best tactic for winning games in few moves (or at all), it achives an average number of guesses of ~5.3. Not too shabby!

But shows we have a long way to go yet with our learner (\*^\*)