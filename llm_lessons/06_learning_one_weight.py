"""
Lesson 6: How a machine learns - with just ONE weight

Problem: tomatoes cost a fixed price per kg, but we don't know the price.
We only have past bills (kg -> rupees). The real price is 30 per kg.

Model:  rupees = w * kg       (w is the one weight to learn)
Start with a bad guess for w, then repeat:
  1. Measure how wrong it is (loss).
  2. Nudge w a tiny bit and see if loss goes up or down (the gradient).
  3. Move w the opposite way (downhill).
This loop is exactly how GPT learns, just with billions of weights.
"""

bills = [(1, 30), (2, 60), (3, 90), (5, 150)]  # (kg, rupees paid)

w = 5.0               # bad starting guess
learning_rate = 0.01  # how big each step is


def loss_for(weight):
    # Average of (guess - real)^2. Squaring makes every error positive.
    return sum((weight * kg - paid) ** 2 for kg, paid in bills) / len(bills)


for step in range(30):
    loss = loss_for(w)

    # Gradient by nudging: how much does loss change if w moves a tiny bit?
    tiny = 0.0001
    gradient = (loss_for(w + tiny) - loss) / tiny

    if step % 3 == 0:
        print(f"step {step:2}: w = {w:6.2f}   loss = {loss:9.2f}   gradient = {gradient:9.2f}")

    # Positive gradient means "loss rises if w rises", so move w down.
    w = w - learning_rate * gradient

print()
print(f"Learned w = {w:.2f}  (real price is 30)")
print(f"Prediction for 4 kg: {w * 4:.0f} rupees")
print()
print("Nudging each weight one by one works here, but GPT has billions of")
print("weights. Lesson 7 shows the fast way to get every gradient at once.")
