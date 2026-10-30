import random

actual_number = random.randint(1,100)
guessed_number = 0
player_won = False

guessed_number = int(input("Guess a number between 1 and 100: "))
while not player_won:
    if guessed_number == actual_number:
        print('You guessed it right!! You won!!')
        player_won = True
        break
    elif guessed_number < actual_number:
        guessed_number = int(input("Guess something higher: "))
    elif guessed_number > actual_number:
        guessed_number = int(input("Guess something lower: "))