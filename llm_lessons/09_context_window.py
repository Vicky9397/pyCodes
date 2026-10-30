"""
Lesson 9: Looking further back - a 3-letter context window

Lesson 8 only saw ONE previous letter, so it could never beat counting.
Now the model sees the last 3 letters:
  1. Look up the embedding (8 numbers) of each of the 3 letters.
  2. Glue them into one list of 24 numbers.
  3. A hidden layer (32 "neurons") processes them. ReLU keeps positives only,
     which lets the network learn patterns that aren't straight lines.
  4. Output layer -> one score per possible next letter.
The loss should go clearly below Lesson 8's best, and words get cleaner.
"""

import math
import os
import random

random.seed(1)


# ---- Autograd engine (same as Lesson 8, plus relu) ----
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

    def relu(self):
        return Value(max(0.0, self.data), (self,), (1.0 if self.data > 0 else 0.0,))

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


def table(rows, cols, std):
    return [[Value(random.gauss(0, std)) for _ in range(cols)] for _ in range(rows)]


# ---- Data: every (3 letters -> next letter) example ----
with open(os.path.join(os.path.dirname(__file__), "data.txt"), encoding="utf-8") as f:
    text = f.read()
chars = sorted(set(text))
V = len(chars)
to_id = {c: i for i, c in enumerate(chars)}
data = [to_id[c] for c in text]
CONTEXT = 3
examples = [(data[i:i + CONTEXT], data[i + CONTEXT]) for i in range(len(data) - CONTEXT)]

# ---- Model ----
EMB, HIDDEN = 8, 32
embedding = table(V, EMB, 0.5)
hidden_w = table(HIDDEN, CONTEXT * EMB, 0.3)
hidden_b = [Value(0.0) for _ in range(HIDDEN)]
output_w = table(V, HIDDEN, 0.3)
params = [p for t in (embedding, hidden_w, output_w) for row in t for p in row] + hidden_b


def scores_for(context_ids):
    x = [n for i in context_ids for n in embedding[i]]  # 3 x 8 = 24 numbers
    h = [(dot(row, x) + b).relu() for row, b in zip(hidden_w, hidden_b)]
    return [dot(row, h) for row in output_w]


# ---- Training ----
STEPS, BATCH, LR = 600, 16, 0.3
print(f"vocab {V} | weights {len(params)} | random-guess loss {math.log(V):.2f}")
print("Lesson 8 could not go below about 1.64 (the counting model's best). Watch this one:")
for step in range(STEPS):
    batch = random.sample(examples, BATCH)
    loss = vsum([-softmax(scores_for(ctx))[nxt].log() for ctx, nxt in batch]) / BATCH

    for p in params:
        p.grad = 0.0
    loss.backward()
    lr = LR if step < STEPS * 0.7 else LR / 3  # smaller steps near the end
    for p in params:
        p.data -= lr * p.grad

    if step % 100 == 0 or step == STEPS - 1:
        print(f"step {step:3}: loss {loss.data:.3f}")

# Average loss over ALL examples (a fairer number than one random batch).
full = sum(-math.log(softmax(scores_for(c))[n].data) for c, n in examples) / len(examples)
print(f"Loss over the whole text: {full:.3f}")
print()

# ---- Generate ----
out = [to_id[c] for c in "the"]
for _ in range(150):
    probs = [p.data for p in softmax(scores_for(out[-CONTEXT:]))]
    out.append(random.choices(range(V), weights=probs)[0])
print("Generated text:")
print("".join(chars[i] for i in out))
print()
print("Better, but a fixed window of 3 is still short. Attention (Lesson 10)")
print("lets the model look back as far as it wants and pick what matters.")
