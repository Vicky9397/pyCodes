"""
Lesson 10: Attention - the key idea inside GPT

No training here. Small made-up numbers so you can follow every step by hand.

Each token makes three lists of numbers:
  query : "what am I looking for?"
  key   : "what do I contain?"
  value : "what information do I pass on if someone picks me?"

To update the latest token:
  1. Compare its query with every earlier token's key (dot product).
     Bigger number = better match.
  2. Softmax those match scores into percentages (attention weights).
  3. Blend all the values using those percentages.
The result is a new list that carries information from the tokens that matter.
In a real model, query/key/value are produced by learned weight tables
(wq, wk, wv in 11_tiny_gpt.py). Here we just write them by hand.
"""

import math

tokens = ["the", "cat", "sat", "on"]

# Made-up 2-number lists. Idea: "on" is looking for something animal-like.
keys = {
    "the": [0.1, 0.0],
    "cat": [1.0, 0.2],   # strongly "animal"
    "sat": [0.2, 1.0],   # strongly "action"
    "on":  [0.3, 0.3],
}
values = {
    "the": [0.0, 0.1],
    "cat": [5.0, 0.0],   # carries "the subject is a cat"
    "sat": [0.0, 5.0],   # carries "the action is sitting"
    "on":  [1.0, 1.0],
}
query_of_on = [2.0, 0.5]  # "on" asks: who is the subject? (mostly animal-ish)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(xs):
    exps = [math.exp(x) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


# Step 1: match scores. "on" may only look at itself and EARLIER tokens.
# (This "no peeking ahead" rule is why GPT can only see the past.)
print('Updating the token "on". It looks back at:', tokens)
print()
print("Step 1 - match score (query of 'on' . key of each token):")
scores = []
for t in tokens:
    s = dot(query_of_on, keys[t])
    scores.append(s)
    print(f"  {t:4}: {s:.2f}")
print()

# Step 2: percentages.
weights = softmax(scores)
print("Step 2 - softmax into attention weights:")
for t, w in zip(tokens, weights):
    print(f"  {t:4}: {w:5.1%}  " + "#" * int(w * 50))
print()

# Step 3: blend the values.
blended = [sum(w * values[t][i] for t, w in zip(tokens, weights)) for i in range(2)]
print("Step 3 - blend values using those weights:")
print(f"  new vector for 'on' = [{blended[0]:.2f}, {blended[1]:.2f}]")
print()
print("'on' paid most attention to 'cat', so its new vector now carries")
print("'cat' information. That is how context flows between words.")
print("GPT runs several of these at once (heads), on every token, stacked")
print("in layers. 11_tiny_gpt.py puts all the lessons together.")
