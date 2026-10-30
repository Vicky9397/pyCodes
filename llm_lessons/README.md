# Build a Language Model From Scratch - 11 Lessons

Pure Python. No libraries to install. Each lesson is one script that teaches one idea.
Do them in order. Each one builds on the one before.

**How to do every lesson:**
1. Open the script and read the comment block at the top.
2. Run it: `python 01_tokenizer.py` (from inside this folder).
3. Read the output next to the code.
4. Stop when the "Done when" line is true. Then move to the next lesson.

`data.txt` is the shared training text used by lessons 3, 8 and 9.

---

### Lesson 1 - `01_tokenizer.py` (runs instantly)
**Idea:** letters become ID numbers, and back again.
**Done when:** you can explain why `encode("dog")` fails.

### Lesson 2 - `02_word_pieces_bpe.py` (instant)
**Idea:** how ChatGPT-style tokenizers join common letter pairs into word pieces.
**Done when:** you can say why `'at'` was the first pair to be joined.

### Lesson 3 - `03_counting_model.py` (instant)
**Idea:** the simplest language model - count which letter follows which, then generate.
**Done when:** you can explain why the output has real letter patterns but few real words.

### Lesson 4 - `04_softmax_and_sampling.py` (instant)
**Idea:** raw scores become percentages (softmax), then a weighted dice picks the next letter. Temperature controls randomness.
**Done when:** you can say what happens to the percentages at temperature 0.3 vs 2.0.

### Lesson 5 - `05_loss.py` (instant)
**Idea:** loss is one number that says how wrong a guess was.
**Done when:** you can explain why random guessing over 26 letters gives a loss of about 3.26.

### Lesson 6 - `06_learning_one_weight.py` (instant)
**Idea:** learning = nudge a weight, see if the loss drops, step downhill. Repeat.
**Done when:** you see `w` climb from 5 to about 30 and can explain what the gradient told it each step.

### Lesson 7 - `07_autograd.py` (instant)
**Idea:** every number remembers how it was made, so one backward pass gives every gradient at once.
**Done when:** you see Part 2's step-0 gradient (-487.50) exactly match Lesson 6's step-0 gradient.

### Lesson 8 - `08_neural_bigram.py` (~10 seconds)
**Idea:** the first neural language model - embeddings plus a weight table, trained with Lessons 4-7.
**Done when:** its final loss lands near the counting model's 1.64. That proves learning works.

### Lesson 9 - `09_context_window.py` (~10 seconds)
**Idea:** look at the last 3 letters instead of 1, with a hidden layer.
**Done when:** you see its loss go clearly below 1.64 and the text has more real words (`the boy`, `the girl`).

### Lesson 10 - `10_attention.py` (instant)
**Idea:** attention - each token looks back, scores every earlier token, and blends in information from the ones that matter.
**Done when:** you can explain why `on` paid 58% attention to `cat`.

### Lesson 11 - `11_tiny_gpt.py` (~1 minute)
**Idea:** everything combined into a real tiny GPT.
**Done when:** you can point to where each lesson appears in this file:
tokenizer (1), softmax (4), loss (5), Adam weight updates (6), `Value` class (7), embeddings (8), MLP layer (9), attention (10).
