import random

starting_number = 1
ending_number = 100

bot_won = False

while not bot_won:
    guessed_number = random.randint(starting_number,ending_number)

    human_input = input(f'is the number {guessed_number} right: ?')

    if human_input == 'yes':
        print('Yay!! I won!!')
        bot_won = True
        break
    elif human_input == 'no, go higher':
        starting_number = guessed_number+1
        guessed_number = random.randint(starting_number,ending_number)
    elif human_input == 'no, go lower':
        ending_number = guessed_number-1
        guessed_number = random.randint(starting_number,ending_number)