"""
Lesson 3: The simplest language model - just counting

No learning yet. Read the text and count: "after letter X, which letters came
next, and how often?" Then generate text by picking the next letter using
those counts. This is called a bigram model (bi = two: current + next).
"""

import os
import random

random.seed(7)

with open(os.path.join(os.path.dirname(__file__), "data.txt"), encoding="utf-8") as f:
    text = f.read()

# Step 1: count every (current, next) pair.
counts = {}
for current, nxt in zip(text, text[1:]):
    counts.setdefault(current, {})
    counts[current][nxt] = counts[current].get(nxt, 0) + 1

# Step 2: look at what follows a few letters.
for letter in ["t", "h", "q"]:
    if letter not in counts:
        print(f"After {letter!r}: never seen in the text")
        continue
    follow = sorted(counts[letter].items(), key=lambda kv: -kv[1])
    total = sum(n for _, n in follow)
    top = ", ".join(f"{c!r} {n * 100 // total}%" for c, n in follow[:5])
    print(f"After {letter!r}: {top}")
print()

# Step 3: generate. Start with "t", then keep picking a next letter
# in proportion to how often it followed the current letter.
out = "t"
for _ in range(150):
    options = counts[out[-1]]
    out += random.choices(list(options), weights=list(options.values()))[0]

print("Generated text:")
print(out)
print()
print("It knows 'h' usually follows 't', but it only ever looks at ONE letter")
print("back, so it cannot form real words reliably. Later lessons fix that.")
