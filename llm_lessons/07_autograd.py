"""
Lesson 7: Autograd - getting every gradient at once (backpropagation)

In Lesson 6 we nudged w to find its gradient. With millions of weights that is
far too slow. Instead, every number remembers how it was made. Then one
backward pass works out every gradient in a single sweep.
This Value class is the same engine used in 11_tiny_gpt.py.
"""


class Value:
    def __init__(self, data, children=(), local_grads=(), label=""):
        self.data = data
        self.grad = 0.0              # filled in by backward()
        self.children = children     # the Values this one was made from
        self.local_grads = local_grads  # how much this changes per child
        self.label = label

    def __add__(self, other):
        # a + b: if a goes up by 1, result goes up by 1. Same for b.
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        # a * b: if a goes up by 1, result goes up by b. And vice versa.
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def backward(self):
        # Visit nodes from the final answer back to the start,
        # passing blame (gradient) to each child.
        order, seen = [], set()

        def visit(node):
            if node not in seen:
                seen.add(node)
                for child in node.children:
                    visit(child)
                order.append(node)

        visit(self)
        self.grad = 1.0
        for node in reversed(order):
            for child, local in zip(node.children, node.local_grads):
                child.grad += local * node.grad


# Part 1: a small sum.  L = a * b + c
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
L = a * b + c
L.backward()

print(f"L = a*b + c = {a.data}*{b.data} + {c.data} = {L.data}")
for v in (a, b, c):
    print(f"  gradient of {v.label} = {v.grad:5.1f}   "
          f"(if {v.label} goes up by 1, L changes by about {v.grad})")

# Check one by nudging, like Lesson 6.
nudged = (2.0001 * -3.0 + 10.0 - L.data) / 0.0001
print(f"  check a by nudging: {nudged:.1f}  -> matches")
print()

# Part 2: redo Lesson 6 (tomato price) using autograd instead of nudging.
bills = [(1, 30), (2, 60), (3, 90), (5, 150)]
w = Value(5.0)
for step in range(30):
    loss = Value(0.0)
    for kg, paid in bills:
        error = w * Value(kg) + Value(-paid)
        loss = loss + error * error
    loss = loss * Value(1 / len(bills))

    w.grad = 0.0
    loss.backward()          # one sweep gives the exact gradient of w

    if step % 5 == 0:
        print(f"step {step:2}: w = {w.data:6.2f}   loss = {loss.data:9.2f}   gradient = {w.grad:8.2f}")

    w.data -= 0.01 * w.grad

print(f"Learned w = {w.data:.2f} (real price is 30), with no nudging needed.")
