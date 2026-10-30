"""
Lesson 8: First neural language model (embeddings + learning)

Same job as Lesson 3 (look at one letter, guess the next), but now it LEARNS:
  - Each letter gets an embedding: a list of 8 random numbers.
  - A weight table turns that list into one score per possible next letter.
  - Softmax -> percentages, loss -> how wrong, backward -> gradients, nudge.
It should end up about as good as the counting model, proving learning works.
"""

import math
import os
import random

random.seed(1)


# ---- Autograd engine (Lesson 7, plus exp, log and a fast dot product) ----
class Value:
    __slots__ = ("data", "grad", "children", "local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data, self.grad = data, 0.0
        self.children, self.local_grads = children, local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __truediv__(self, other):
        if isinstance(other, Value):
            return self * Value(1 / other.data, (other,), (-1 / other.data ** 2,))
        return self * (1 / other)

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def backward(self):
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                order.append(node)
            elif node not in seen:
                seen.add(node)
                stack.append((node, True))
                stack.extend((c, False) for c in node.children if c not in seen)
        self.grad = 1.0
        for node in reversed(order):
            for child, local in zip(node.children, node.local_grads):
                child.grad += local * node.grad


def dot(a, b):
    return Value(sum(x.data * y.data for x, y in zip(a, b)), tuple(a) + tuple(b),
                 tuple(y.data for y in b) + tuple(x.data for x in a))


def vsum(values):
    return Value(sum(v.data for v in values), tuple(values), (1.0,) * len(values))


def softmax(scores):
    peak = max(s.data for s in scores)
    exps = [(s - peak).exp() for s in scores]
    total = vsum(exps)
    return [e / total for e in exps]


# ---- Data ----
with open(os.path.join(os.path.dirname(__file__), "data.txt"), encoding="utf-8") as f:
    text = f.read()
chars = sorted(set(text))
V = len(chars)
to_id = {c: i for i, c in enumerate(chars)}
data = [to_id[c] for c in text]
pairs = list(zip(data, data[1:]))  # (current letter, correct next letter)

# ---- Model: embedding table + output table ----
EMB = 8
embedding = [[Value(random.gauss(0, 0.5)) for _ in range(EMB)] for _ in range(V)]
output = [[Value(random.gauss(0, 0.5)) for _ in range(EMB)] for _ in range(V)]
params = [p for table in (embedding, output) for row in table for p in row]


def scores_for(letter_id):
    vec = embedding[letter_id]               # letter -> its 8 numbers
    return [dot(row, vec) for row in output]  # 8 numbers -> one score per letter


# ---- Training ----
STEPS, BATCH, LR = 1000, 32, 1.0
print(f"vocab {V} | weights {len(params)} | random-guess loss {math.log(V):.2f}")
for step in range(STEPS):
    batch = random.sample(pairs, BATCH)
    losses = [-softmax(scores_for(cur))[nxt].log() for cur, nxt in batch]
    loss = vsum(losses) / BATCH

    for p in params:
        p.grad = 0.0
    loss.backward()
    for p in params:
        p.data -= LR * p.grad

    if step % 100 == 0 or step == STEPS - 1:
        print(f"step {step:3}: loss {loss.data:.3f}")

# Average loss over ALL pairs (a fairer number than one random batch).
full = sum(-math.log(softmax(scores_for(c))[n].data) for c, n in pairs) / len(pairs)
print(f"Neural model loss over the whole text: {full:.3f}")

# Best possible score for a one-letter-back model = the counting model's loss.
counts = {}
for cur, nxt in pairs:
    counts.setdefault(cur, {})
    counts[cur][nxt] = counts[cur].get(nxt, 0) + 1
best = sum(-math.log(counts[c][n] / sum(counts[c].values())) for c, n in pairs) / len(pairs)
print(f"Counting model (Lesson 3) loss: {best:.3f}  <- the best this design can reach")
print()

# ---- Generate ----
out = [to_id["t"]]
for _ in range(150):
    probs = [p.data for p in softmax(scores_for(out[-1]))]
    out.append(random.choices(range(V), weights=probs)[0])
print("Generated text:")
print("".join(chars[i] for i in out))
