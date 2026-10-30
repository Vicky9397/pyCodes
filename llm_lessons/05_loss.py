"""
Lesson 5: Loss - one number that says "how wrong was the guess"

Training needs a score to make smaller. For language models that score is:
    loss = -log(percentage the model gave to the CORRECT next letter)
Confident and right -> tiny loss. Unsure or wrong -> big loss.
"""

import math

print("Percentage given to the correct letter -> loss")
for p in [0.99, 0.9, 0.7, 0.5, 0.2, 0.04, 0.01]:
    print(f"  {p:5.0%}  ->  loss {-math.log(p):.2f}")
print()

# Random guessing among 26 letters gives each one 1/26.
print(f"Random guessing over 26 letters: loss {-math.log(1 / 26):.2f}")
print("This is why your tiny GPT started at about 3.3.")
print()

# Example: two models predict the letter after "the ca". The answer is "t".
model_a = {"t": 0.70, "r": 0.20, "n": 0.10}
model_b = {"t": 0.10, "r": 0.60, "n": 0.30}
for name, preds in [("Model A", model_a), ("Model B", model_b)]:
    print(f"{name} gave 't' {preds['t']:.0%} -> loss {-math.log(preds['t']):.2f}")
print("Lower loss = better model. Training = adjusting weights to lower this.")
print()

# Over a whole sentence we just average the loss of every guess.
guesses = [0.9, 0.6, 0.3, 0.8]
avg = sum(-math.log(p) for p in guesses) / len(guesses)
print(f"Four guesses gave the correct letter {guesses} -> average loss {avg:.2f}")
