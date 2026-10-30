"""
Lesson 2: Word pieces (how ChatGPT-style tokenizers are built)

Big models do not use single characters. They use word pieces, built by
BPE (Byte Pair Encoding):
  1. Start with every character as its own piece.
  2. Find the pair of pieces that sits side by side most often.
  3. Join that pair into one new piece.
  4. Repeat.
Watch the text get split into fewer, bigger pieces each round.
"""

text = "the cat sat on the mat. the hat is on the cat. the cat is fat."
MERGES = 10

tokens = list(text)
print(f"Start: {len(tokens)} pieces (one per character)")
print(tokens)
print()

for round_no in range(1, MERGES + 1):
    # Count how often each side-by-side pair appears.
    pair_counts = {}
    for a, b in zip(tokens, tokens[1:]):
        pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    best = max(pair_counts, key=pair_counts.get)

    # Join every occurrence of the best pair into one piece.
    merged, i = [], 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best:
            merged.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    tokens = merged

    print(f"Round {round_no:2}: joined {best[0]!r} + {best[1]!r} "
          f"(seen {pair_counts[best]} times) -> {len(tokens)} pieces")

print()
print("Final pieces:")
print(tokens)
print()
print("Same text, far fewer pieces. Common chunks like 'the ' and 'at' became")
print("single tokens. Real tokenizers do this until they have ~100,000 pieces.")
