"""
tiny_gpt.py

A tiny character-level GPT (the same family as ChatGPT, at toy scale),
written in pure Python. Only the standard library is used: math, random, os.

How it works, in order:
  1. Autograd   - a small engine that tracks every math operation so it can
                  work out how to nudge each weight to reduce the error.
  2. Tokenizer  - turns each character into a number and back.
  3. Model      - a 1-layer transformer: embeddings, self-attention, MLP.
  4. Training   - show it text, measure how wrong its next-character guess is,
                  adjust weights with the Adam optimizer. Repeat.
  5. Generation - feed it a start and let it predict one character at a time.

Run:  python tiny_gpt.py
Put your own text in input.txt (same folder) to train on that instead.
"""

import math
import os
import random

random.seed(42)


# ---------------------------------------------------------------------------
# 1. Autograd engine
# ---------------------------------------------------------------------------
# Every number in the model is a Value. A Value remembers which Values it was
# made from (children) and how much it changes when each child changes
# (local_grads). backward() walks this chain to fill in .grad for every weight.

class Value:
    __slots__ = ("data", "grad", "children", "local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0.0
        self.children = children
        self.local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, k):
        return Value(self.data ** k, (self,), (k * self.data ** (k - 1),))

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        return Value(math.log(self.data), (self,), (1.0 / self.data,))

    def relu(self):
        return Value(max(0.0, self.data), (self,), (1.0 if self.data > 0 else 0.0,))

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1.0 if isinstance(other, Value) else 1.0 / other)

    __radd__ = __add__
    __rmul__ = __mul__

    def backward(self):
        # Order nodes so each one comes after everything it was built from.
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if node in seen:
                continue
            seen.add(node)
            stack.append((node, True))
            for child in node.children:
                if child not in seen:
                    stack.append((child, False))

        self.grad = 1.0
        for node in reversed(order):
            for child, local in zip(node.children, node.local_grads):
                child.grad += local * node.grad


def vsum(values):
    """Sum a list of Values as a single node (faster than chaining +)."""
    return Value(sum(v.data for v in values), tuple(values), (1.0,) * len(values))


def dot(a, b):
    """Dot product of two Value lists as a single node."""
    return Value(
        sum(x.data * y.data for x, y in zip(a, b)),
        tuple(a) + tuple(b),
        tuple(y.data for y in b) + tuple(x.data for x in a),
    )


# ---------------------------------------------------------------------------
# 2. Data and tokenizer
# ---------------------------------------------------------------------------

DEFAULT_TEXT = """the cat sat on the mat.
the dog sat on the rug.
the cat ran to the park.
the dog ran to the park.
the sun is hot and the sky is blue.
the moon is cold and the night is dark.
a bird sang in the tree.
a fish swam in the lake.
the boy ate a red apple.
the girl ate a green apple.
the cat saw the bird in the tree.
the dog saw the fish in the lake.
i like the cat and the cat likes me.
i like the dog and the dog likes me.
the rain fell on the town.
the wind blew over the hill.
the boy and the girl ran to the hill.
the cat and the dog sat in the sun.
we went to the park in the morning.
we went to the lake at night.
the old man sat by the fire.
the old woman read a book by the fire.
the bird flew over the lake.
the fish jumped out of the lake.
the dog likes to run and the cat likes to sleep.
the boy likes to read and the girl likes to sing.
a small cat slept on a warm mat.
a big dog slept on a soft rug.
the tree is tall and the grass is green.
the sea is deep and the sand is warm.
"""


def load_text():
    if os.path.exists("input.txt"):
        with open("input.txt", "r", encoding="utf-8") as f:
            return f.read()
    return DEFAULT_TEXT


text = load_text()
chars = sorted(set(text))
vocab_size = len(chars)
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for i, c in enumerate(chars)}


def encode(s):
    return [char_to_id[c] for c in s]


def decode(ids):
    return "".join(id_to_char[i] for i in ids)


data = encode(text)


# ---------------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------------

N_EMBD = 16       # size of each token's vector
N_HEAD = 4        # attention heads
N_LAYER = 1       # transformer blocks
BLOCK_SIZE = 16   # how many past characters the model can look at
HEAD_DIM = N_EMBD // N_HEAD


def matrix(rows, cols, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(cols)] for _ in range(rows)]


weights = {
    "wte": matrix(vocab_size, N_EMBD),   # token embeddings
    "wpe": matrix(BLOCK_SIZE, N_EMBD),   # position embeddings
    "lm_head": matrix(vocab_size, N_EMBD),
}
for layer in range(N_LAYER):
    weights[f"l{layer}.wq"] = matrix(N_EMBD, N_EMBD)
    weights[f"l{layer}.wk"] = matrix(N_EMBD, N_EMBD)
    weights[f"l{layer}.wv"] = matrix(N_EMBD, N_EMBD)
    weights[f"l{layer}.wo"] = matrix(N_EMBD, N_EMBD)
    weights[f"l{layer}.fc1"] = matrix(4 * N_EMBD, N_EMBD)
    weights[f"l{layer}.fc2"] = matrix(N_EMBD, 4 * N_EMBD)

params = [p for mat in weights.values() for row in mat for p in row]


def linear(x, w):
    return [dot(row, x) for row in w]


def softmax(logits):
    peak = max(v.data for v in logits)
    exps = [(v - peak).exp() for v in logits]
    total = vsum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    mean_sq = vsum([xi * xi for xi in x]) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def forward_token(token_id, pos, keys, values):
    """Process one token at position pos. keys/values hold earlier tokens."""
    x = [t + p for t, p in zip(weights["wte"][token_id], weights["wpe"][pos])]
    x = rmsnorm(x)

    for layer in range(N_LAYER):
        # Self-attention: look back at earlier tokens and mix in what matters.
        residual = x
        x = rmsnorm(x)
        q = linear(x, weights[f"l{layer}.wq"])
        k = linear(x, weights[f"l{layer}.wk"])
        v = linear(x, weights[f"l{layer}.wv"])
        keys[layer].append(k)
        values[layer].append(v)

        attn_out = []
        for h in range(N_HEAD):
            s, e = h * HEAD_DIM, (h + 1) * HEAD_DIM
            q_h = q[s:e]
            scores = [dot(q_h, k_t[s:e]) / math.sqrt(HEAD_DIM) for k_t in keys[layer]]
            attn = softmax(scores)
            for j in range(s, e):
                attn_out.append(dot(attn, [v_t[j] for v_t in values[layer]]))

        x = linear(attn_out, weights[f"l{layer}.wo"])
        x = [a + b for a, b in zip(x, residual)]

        # MLP: think about the mixed information.
        residual = x
        x = rmsnorm(x)
        x = [h.relu() for h in linear(x, weights[f"l{layer}.fc1"])]
        x = linear(x, weights[f"l{layer}.fc2"])
        x = [a + b for a, b in zip(x, residual)]

    return linear(x, weights["lm_head"])  # one score per character


def predict_next(context):
    """Return logits for the character that follows context."""
    context = context[-BLOCK_SIZE:]
    keys = [[] for _ in range(N_LAYER)]
    values = [[] for _ in range(N_LAYER)]
    logits = None
    for pos, token_id in enumerate(context):
        logits = forward_token(token_id, pos, keys, values)
    return logits


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------

STEPS = 1000
LEARNING_RATE = 0.01
BETA1, BETA2, EPS = 0.9, 0.99, 1e-8

m = [0.0] * len(params)  # Adam: running average of gradients
v = [0.0] * len(params)  # Adam: running average of squared gradients


def train():
    print(f"text: {len(data)} chars | vocab: {vocab_size} | params: {len(params)}")
    for step in range(STEPS):
        start = random.randint(0, len(data) - BLOCK_SIZE - 1)
        chunk = data[start:start + BLOCK_SIZE + 1]
        inputs, targets = chunk[:-1], chunk[1:]

        keys = [[] for _ in range(N_LAYER)]
        values = [[] for _ in range(N_LAYER)]
        losses = []
        for pos, (token_id, target) in enumerate(zip(inputs, targets)):
            probs = softmax(forward_token(token_id, pos, keys, values))
            losses.append(-probs[target].log())
        loss = vsum(losses) / len(losses)

        for p in params:
            p.grad = 0.0
        loss.backward()

        lr = LEARNING_RATE * (1 - step / STEPS)  # slowly lower the step size
        for i, p in enumerate(params):
            m[i] = BETA1 * m[i] + (1 - BETA1) * p.grad
            v[i] = BETA2 * v[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m[i] / (1 - BETA1 ** (step + 1))
            v_hat = v[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr * m_hat / (math.sqrt(v_hat) + EPS)

        if step % 50 == 0 or step == STEPS - 1:
            print(f"step {step:4d} | loss {loss.data:.4f}")


# ---------------------------------------------------------------------------
# 5. Generation
# ---------------------------------------------------------------------------

def generate(prompt="the ", length=200, temperature=0.8):
    ids = encode(prompt)
    for _ in range(length):
        logits = predict_next(ids)
        scaled = [l.data / temperature for l in logits]
        peak = max(scaled)
        weights_ = [math.exp(s - peak) for s in scaled]
        ids.append(random.choices(range(vocab_size), weights=weights_)[0])
    return decode(ids)


if __name__ == "__main__":
    train()
    print("\n--- generated text ---")
    print(generate())
