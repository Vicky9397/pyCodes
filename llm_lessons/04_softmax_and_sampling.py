"""
Lesson 4: Scores -> percentages (softmax) and picking the next letter

A neural network does not output percentages. It outputs raw scores
(any number, even negative). Softmax turns them into percentages that add up
to 100%. Then we roll a weighted dice to pick the next letter.
"""

import math
import random

random.seed(3)

letters = ["t", "r", "n", "x"]
scores = [2.0, 0.5, 0.1, -1.0]  # made-up raw scores from a model


def softmax(values, temperature=1.0):
    # e^score makes every number positive and makes big scores much bigger.
    exps = [math.exp(v / temperature) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


print("Raw scores :", dict(zip(letters, scores)))
probs = softmax(scores)
print("Softmax    :", {l: f"{p:.0%}" for l, p in zip(letters, probs)})
print("Adds up to :", f"{sum(probs):.0%}")
print()

# Sampling: pick 100 times using the percentages as dice weights.
picks = random.choices(letters, weights=probs, k=100)
print("Picked 100 times:", {l: picks.count(l) for l in letters})
print("Mostly 't', but sometimes others. That randomness is why the same")
print("prompt can give different answers.")
print()

# Temperature: divide scores before softmax.
#   low  (0.3) -> safe, repetitive, almost always the top letter
#   high (2.0) -> wild, more random choices
for t in [0.3, 1.0, 2.0]:
    p = softmax(scores, temperature=t)
    print(f"Temperature {t}: " + "  ".join(f"{l}={x:.0%}" for l, x in zip(letters, p)))
