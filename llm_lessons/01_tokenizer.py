"""
Lesson 1: Tokenizer (letters -> numbers -> letters)

A computer can only do maths, so every character needs an ID number.
This is the very first step of every language model.
"""

text = "the cat sat on the mat."

# Step 1: find every unique character, sorted so the order never changes.
chars = sorted(set(text))
print("Unique characters:", chars)
print("Vocabulary size  :", len(chars))
print()

# Step 2: build two lookup tables, one for each direction.
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for i, c in enumerate(chars)}
print("Character -> ID table:")
for c, i in char_to_id.items():
    print(f"  {c!r:6} -> {i}")
print()


def encode(s):
    """Text to list of numbers."""
    return [char_to_id[c] for c in s]


def decode(ids):
    """List of numbers back to text."""
    return "".join(id_to_char[i] for i in ids)


# Step 3: try it.
ids = encode("cat")
print('encode("cat")     ->', ids)
print(f"decode({ids})  ->", repr(decode(ids)))
print('encode("the mat") ->', encode("the mat"))
print()

# Step 4: a character the tokenizer has never seen cannot be encoded.
# This is why real models build their vocabulary from huge amounts of text.
try:
    encode("dog")
except KeyError as missing:
    print(f'encode("dog") fails: {missing} was never in the training text.')
